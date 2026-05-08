#!/usr/bin/env python3
"""
Auto-detekcia hovoriacich z input videa cez librosa F0 (pitch) analýzu
+ obsahová detekcia cez Whisper sample + keyword regex.

Detekuje:
  - Počet hovoriacich (1 vs >1) — bimodálna distribúcia pitch
  - Pohlavie (male/female/child) — mean F0
  - Typ obsahu (general/programming/technical/sql/educational/...) — keyword scan
  - Odporúčaný voice design preset

Použitie:
  ./auto_detect_speakers.py video.mp4
  ./auto_detect_speakers.py video.mp4 --json   # machine-readable output
  ./auto_detect_speakers.py video.mp4 --no-content   # vynech content scan (rýchlejšie)
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import librosa

REPO = Path(__file__).resolve().parent
DEMUCS_CACHE = REPO / "work" / "demucs_cache"


def get_demucs_vocals(video_path: Path) -> Path:
    """Extract vocals via demucs (cached). Returns vocals.wav path."""
    stem = video_path.stem
    cached = DEMUCS_CACHE / stem / "vocals.wav"
    if cached.exists() and cached.stat().st_size > 1000:
        return cached
    # Fallback: extract raw audio (works less well but doesn't need demucs)
    DEMUCS_CACHE.mkdir(parents=True, exist_ok=True)
    raw = DEMUCS_CACHE / f"{stem}_raw.wav"
    if not raw.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(raw)],
            capture_output=True, check=True,
        )
    return raw


def analyze_pitch(audio_path: Path, sample_rate: int = 16000) -> dict:
    """librosa pyin pitch tracker. Returns stats + raw F0 array."""
    print(f"[ANALYZE] Loading {audio_path.name}...", flush=True)
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    print(f"[ANALYZE] Duration: {len(y)/sr:.1f}s, running pyin...", flush=True)
    # pyin = probabilistic YIN, robust pitch tracker
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=70,    # min male F0
        fmax=400,   # max child/female F0
        sr=sr,
        frame_length=2048,
        hop_length=512,  # 32ms hop @ 16kHz
    )
    # Filter voiced frames with high confidence
    voiced = f0[~np.isnan(f0) & (voiced_probs > 0.5)]
    if len(voiced) < 50:
        return {"error": "Too few voiced frames", "n_voiced": len(voiced)}
    return {
        "f0": voiced.tolist(),
        "n_voiced": int(len(voiced)),
        "n_total": int(len(f0)),
        "voiced_ratio": float(len(voiced) / len(f0)),
        "mean_f0": float(np.mean(voiced)),
        "median_f0": float(np.median(voiced)),
        "std_f0": float(np.std(voiced)),
        "p25_f0": float(np.percentile(voiced, 25)),
        "p75_f0": float(np.percentile(voiced, 75)),
    }


def classify_gender(median_f0: float, p25: float = 0, p75: float = 0, std: float = 0) -> tuple:
    """F0 → (gender, confidence, alt). Vráti hlavnú classifikáciu + alternative pre borderline.

    Thresholds (Hz):
      male:    < 175
      female:  175-265
      child:   > 265

    POZN.: pyin občas robí octave-doubling pri mužskom hlase a aj kvôli reverb/format
    detekuje 2× fundamental. Borderline cases (240-280) sú nespoľahlivé.
    """
    if median_f0 < 165:
        return ("male", "high", None)
    elif median_f0 < 195:
        return ("male", "medium", "female")  # young men, occasional doubling
    elif median_f0 < 235:
        return ("female", "high", None)
    elif median_f0 < 270:
        return ("female", "medium", "male")  # could be young man w/ pitch errors
    else:
        return ("child", "medium", "female")


def detect_multispeaker(f0_list: list) -> dict:
    """Bimodality test on pitch distribution.
    If clearly bimodal (2 peaks far apart) → likely 2 speakers (m+f)."""
    f0 = np.array(f0_list)
    # Find histogram peaks
    hist, edges = np.histogram(f0, bins=30, range=(70, 400))
    # Smooth
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(hist.astype(float), sigma=1.5)
    # Find local maxima
    peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > smooth.max() * 0.20:
            peaks.append((edges[i], smooth[i]))
    n_peaks = len(peaks)
    is_bimodal = False
    speaker_count_est = 1
    if n_peaks >= 2:
        # Check if peaks are FAR apart (>50 Hz) and significant
        peaks.sort(key=lambda x: -x[1])
        top2 = peaks[:2]
        if abs(top2[0][0] - top2[1][0]) > 50 and top2[1][1] / top2[0][1] > 0.30:
            is_bimodal = True
            speaker_count_est = 2
    return {
        "is_bimodal": is_bimodal,
        "speaker_count_estimated": speaker_count_est,
        "n_peaks_detected": n_peaks,
        "peaks_hz": [round(p[0], 0) for p in peaks[:3]],
    }


def suggest_omnivoice_params(gender_confidence: str, multispeaker: bool,
                             content_type: str = "general") -> dict:
    """Decision tree → (diffusion_steps, guidance_scale).
    Vyššie hodnoty pre náročnejšie podmienky (tech termíny, multispeaker, low conf).
    """
    # Default — bezpečný stred pre väčšinu videí
    steps = 64
    guidance = 2.5

    # Content type zvýši kvalitu pre technické/programmné videá
    if content_type in ("programming", "technical", "sql"):
        steps = 96
        guidance = 2.8

    # Multi-speaker → silnejšie ref tracking (aby sa hlasy nemiešali)
    if multispeaker:
        steps = max(steps, 96)
        guidance = max(guidance, 3.0)

    # Low confidence → nestabilný pitch v zdrojovom audiu, treba viac steps
    if gender_confidence == "low":
        steps = max(steps, 96)
        guidance = max(guidance, 3.0)
    elif gender_confidence == "medium":
        steps = max(steps, 80)
        guidance = max(guidance, 2.7)

    # Cap pre rozumný čas inferencie
    steps = min(steps, 128)
    guidance = round(min(guidance, 3.5), 2)
    return {"diffusion_steps": int(steps), "guidance_scale": float(guidance)}


def suggest_voice_design(gender: str, multispeaker: bool, lang: str = "sk") -> str:
    """Suggest voice design preset based on gender + lang."""
    flag = "🇸🇰" if lang.lower() in ("sk", "slk") else "🇨🇿" if lang.lower() in ("cs", "ces", "cz") else "🇸🇰"
    if multispeaker:
        return f"(viac hlasov — multi-voice, default voice design)"
    if gender == "male":
        return f"{flag} {'slovenský' if flag=='🇸🇰' else 'český'} muž"
    elif gender == "female":
        return f"{flag} {'slovenská' if flag=='🇸🇰' else 'česká'} žena"
    else:
        return f"{flag} dieťa (skús muža/ženu manuálne)"


# ───────────────────────── content-type detection ──────────────────────────
# Keyword sets — case-insensitive word-boundary matches. Short, hand-curated.
# Cieľ: rozhodnúť medzi general / programming / technical / sql / educational /
# podcast / review / news. Default = general.
_KW_PROGRAMMING = [
    r"c\+\+", r"\bpython\b", r"\bjavascript\b", r"\btypescript\b", r"\brust\b",
    r"\bgolang\b", r"\bgo\s+lang\b", r"\bjava(?!script)\b", r"\bkotlin\b",
    r"\bswift\b", r"\bruby\b", r"\bphp\b", r"\bscala\b", r"\bhaskell\b",
    r"\bfunction\b", r"\bvariable\b", r"\bcompil(?:e|er|ing)\b", r"\bdebug(?:ger|ging)?\b",
    r"\bgithub\b", r"\bgit\s+repo\b", r"\brepository\b", r"\bpull\s+request\b",
    r"\bcommit\b", r"\bcodebase\b", r"\brefactor\b", r"\bframework\b",
    r"\blibrary\b", r"\bAPI\b", r"\bSDK\b", r"\bIDE\b", r"\bvisual\s+studio\b",
    r"\bvscode\b", r"\bvs\s+code\b", r"\bnpm\b", r"\bpip\b", r"\bcargo\b",
    r"\breact(?:js)?\b", r"\bvue(?:js)?\b", r"\bangular\b", r"\bnode(?:js)?\b",
    r"\bclass\b.{0,30}\b(method|inherit|extend)", r"\basync\b", r"\bawait\b",
    r"\bcallback\b", r"\bclosure\b", r"\brecursion\b", r"\bbinary\s+tree\b",
    r"\blinked\s+list\b", r"\barray\b", r"\bhash\s*map\b", r"\bdictionary\b",
]
_KW_TECHNICAL = [
    r"\bGPU\b", r"\bCPU\b", r"\bRAM\b", r"\bVRAM\b", r"\bSSD\b", r"\bHDD\b",
    r"\bengine\b", r"\bunity\b", r"\bunreal\b", r"\bgodot\b",
    r"\bshader\b", r"\brender(?:er|ing)?\b", r"\braytrac\w+\b",
    r"\bphysics\b", r"\bvector\b", r"\bmatrix\b", r"\bquaternion\b",
    r"\bkernel\b", r"\bdriver\b", r"\bfirmware\b", r"\bregister\b",
    r"\bFPGA\b", r"\bARM\b", r"\bx86\b", r"\bRISC-?V\b", r"\bmicrocontroller\b",
    r"\bvoltage\b", r"\bcurrent\b", r"\bfrequency\b", r"\bsignal\b", r"\bcircuit\b",
    r"\balgorithm\b", r"\bneural\s+network\b", r"\btransformer\b",
    r"\bdeep\s+learning\b", r"\bmachine\s+learning\b", r"\bLLM\b",
    r"\binference\b", r"\btraining\b", r"\bdataset\b", r"\bquantiz\w+\b",
    r"\btensor\b", r"\bpytorch\b", r"\btensorflow\b", r"\bCUDA\b", r"\bOpenGL\b",
    r"\bDirectX\b", r"\bVulkan\b", r"\bOpenCL\b",
]
_KW_SQL = [
    r"\bSQL\b", r"\bSELECT\b\s+\w", r"\bFROM\s+\w+\s+(WHERE|JOIN|GROUP)",
    r"\bWHERE\s+\w", r"\bJOIN\b", r"\bINNER\s+JOIN\b", r"\bLEFT\s+JOIN\b",
    r"\bGROUP\s+BY\b", r"\bORDER\s+BY\b", r"\bINSERT\s+INTO\b", r"\bUPDATE\s+\w+\s+SET\b",
    r"\bDELETE\s+FROM\b", r"\bschema\b", r"\bquery\b", r"\bdatabase\b",
    r"\bpostgres(?:ql)?\b", r"\bmysql\b", r"\bmongodb\b", r"\bsqlite\b",
    r"\bmariadb\b", r"\boracle\s+db\b", r"\bredis\b",
    r"\bNULL\b", r"\bforeign\s+key\b", r"\bprimary\s+key\b", r"\btable\b",
    r"\bcolumn\b", r"\brow\b", r"\bindex(?:ed|ing|es)?\b",
    r"\bnormaliz\w+\b", r"\bACID\b", r"\btransaction\b", r"\bORM\b",
]
_KW_EDUCATIONAL = [
    r"\bin\s+this\s+(lesson|tutorial|video|course)\b",
    r"\btoday\s+(we'll|we\s+will|i'll|i\s+will)\s+(learn|cover|teach|explain)\b",
    r"\blet's\s+(learn|cover|explore|dive)\b",
    r"\bstep\s+by\s+step\b", r"\bbeginner('?s)?\s+guide\b",
    r"\btutorial\b", r"\bexplain\w*\b", r"\bunderstand\w*\b",
    r"\blecture\b", r"\bcourse\b", r"\bclassroom\b",
]
_KW_PODCAST = [
    r"\bwelcome\s+(back|to\s+(the|our))\b",
    r"\bthanks\s+for\s+(joining|tuning)\b",
    r"\bon\s+today's\s+(show|episode|podcast)\b",
    r"\bmy\s+guest\b", r"\bjoining\s+me\b",
    r"\bpodcast\b", r"\bepisode\b",
]
_KW_REVIEW = [
    r"\btoday\s+i'?m\s+reviewing\b",
    r"\bpros\s+and\s+cons\b", r"\bverdict\b",
    r"\bwould\s+i\s+recommend\b", r"\bworth\s+(buying|the\s+money)\b",
    r"\bmy\s+(review|rating|verdict)\b",
    r"\bunboxing\b", r"\bfirst\s+impression\b",
]
_KW_NEWS = [
    r"\bbreaking\s+news\b", r"\baccording\s+to\s+(officials|sources|reports)\b",
    r"\breports\s+(indicate|suggest)\b",
    r"\bin\s+other\s+news\b", r"\bheadlines\b",
]

# SK/CZ keyword extensions — pre prípady keď Whisper transkribuje už-dabbed video
# alebo originál v SK/CZ. Tech termíny (C++, SQL, GitHub, …) ostávajú v EN sade.
_KW_PROGRAMMING_SK = [
    r"\bprogramov\w+\b", r"\bprogramátor\w*\b",
    r"\bfunkci[au]\b", r"\bpremenn[áéú]\b",
    r"\bkompil\w+\b", r"\bdebugov\w+\b",
    r"\bknižnic\w+\b", r"\bzdrojov[ýé]\s+kód\b", r"\bkód\w*\b",
    r"\brepozitár\w*\b", r"\baplikác\w+\b",
]
_KW_TECHNICAL_SK = [
    r"\bgrafick[áýé]\s+kart\w+\b", r"\bprocesor\w*\b",
    r"\bhardvér\w*\b", r"\bsoftvér\w*\b",
    r"\bjadr\w+\b", r"\bovládač\w*\b", r"\bdriver\w*\b",
    r"\balgoritm\w+\b", r"\bneurón\w+\s+sieť\w*\b",
    r"\bstrojov[éý]\s+učen\w+\b", r"\bumel\w+\s+inteligenc\w+\b",
    r"\bnapätie\b", r"\bprúd\w*\b", r"\bfrekvenci\w+\b", r"\bsignál\w*\b",
]
_KW_SQL_SK = [
    r"\bdatabáz\w+\b", r"\btabuľk\w+\b", r"\btabúľk\w+\b",  # whisper občas tabúľka
    r"\bstĺpc\w+\b", r"\briadk\w+\b", r"\bdotaz\w*\b",
    r"\bschém\w+\b", r"\bnulov[éý]\s+hodnot\w+\b",
    r"\bcudz[íý]\s+kľúč\w*\b", r"\bprimárn\w+\s+kľúč\w*\b",
    r"\bdatabáz\w+\s+jazyk\w*\b",
]
_KW_EDUCATIONAL_SK = [
    r"\bv\s+tejto\s+(lekcii|hodine|kapitole)\b",
    r"\bdnes\s+sa\s+(naučíme|pozrieme|budeme)\b",
    r"\bnaučíme\s+sa\b", r"\bvysvetlím\b", r"\bukážem\s+(vám|si)\b",
    r"\btutoriál\w*\b", r"\bnávod\w*\b", r"\blekci\w+\b",
]


def _kw_count(text: str, patterns: list) -> int:
    n = 0
    for pat in patterns:
        n += len(re.findall(pat, text, flags=re.IGNORECASE))
    return n


def transcribe_sample(video_path: Path, sample_seconds: int = 90,
                      offset_seconds: int = 30) -> tuple:
    """Vyextrahuje sample (default 90s od 30s) a transkribuje cez faster-whisper.
    Multilingual model + auto-detect language (funguje pre EN aj SK/CZ dabbed).
    Vracia (text, lang). Pri chybe vráti ("", "").
    """
    import tempfile
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return ("", "")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sample_wav = Path(tf.name)
    try:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(offset_seconds), "-t", str(sample_seconds),
                 "-i", str(video_path), "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(sample_wav)],
                capture_output=True, check=True, timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ("", "")
        if not sample_wav.exists() or sample_wav.stat().st_size < 1000:
            return ("", "")

        # Multilingual `small` model — auto-detect language. C++/SQL/GitHub
        # ostávajú v transcripte v EN aj keď je rest SK/CZ.
        print("[CONTENT] Transcribing sample (small, multilingual)...", flush=True)
        try:
            model = WhisperModel("small", device="cuda", compute_type="float16")
        except Exception:
            try:
                model = WhisperModel("small", device="cpu", compute_type="int8")
            except Exception as e:
                print(f"[CONTENT] Whisper init failed: {e}", flush=True)
                return ("", "")
        segments, info = model.transcribe(
            str(sample_wav), beam_size=1,
            condition_on_previous_text=False, vad_filter=False,
        )
        chunks = [s.text for s in segments]
        lang = getattr(info, "language", "") or ""
        try:
            del model
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return (" ".join(chunks).strip(), lang)
    finally:
        sample_wav.unlink(missing_ok=True)


def classify_content(transcript: str) -> dict:
    """Spočíta keyword hits (EN + SK/CZ) → vyberie najsilnejší typ obsahu.
    Vracia dict so 'content_type', 'confidence', 'scores'.
    """
    if not transcript or len(transcript) < 80:
        return {"content_type": "general", "confidence": "low",
                "scores": {}, "transcript_chars": len(transcript or "")}
    # EN + SK keywords sčítame — EN tech termíny (SQL, C++, GitHub) ostávajú aj
    # v SK transcripte, SK keywords zachytávajú lokalizovaný kontext.
    scores = {
        "programming": _kw_count(transcript, _KW_PROGRAMMING + _KW_PROGRAMMING_SK),
        "technical":   _kw_count(transcript, _KW_TECHNICAL + _KW_TECHNICAL_SK),
        "sql":         _kw_count(transcript, _KW_SQL + _KW_SQL_SK),
        "educational": _kw_count(transcript, _KW_EDUCATIONAL + _KW_EDUCATIONAL_SK),
        "podcast":     _kw_count(transcript, _KW_PODCAST),
        "review":      _kw_count(transcript, _KW_REVIEW),
        "news":        _kw_count(transcript, _KW_NEWS),
    }
    # Priorita: SQL > programming > technical (špecifickejšie ako technical)
    # Ostatné kategórie sa berú iba ak top tech score je nízky.
    top = max(scores, key=scores.get)
    top_score = scores[top]

    # SQL má prioritu ak má aspoň 3 hits (špecifickejšie keywords)
    if scores["sql"] >= 3:
        chosen = "sql"
    elif scores["programming"] >= 4:
        chosen = "programming"
    elif scores["technical"] >= 4:
        chosen = "technical"
    elif top_score >= 3:
        chosen = top
    else:
        chosen = "general"

    # Confidence: high ak top je 2× viac ako 2nd, jinak medium/low
    sorted_scores = sorted(scores.values(), reverse=True)
    if top_score == 0:
        conf = "low"
    elif len(sorted_scores) > 1 and sorted_scores[1] > 0 and top_score / sorted_scores[1] < 1.5:
        conf = "medium"
    elif top_score >= 5:
        conf = "high"
    else:
        conf = "medium"

    return {
        "content_type": chosen,
        "confidence": conf,
        "scores": scores,
        "transcript_chars": len(transcript),
    }


def main():
    ap = argparse.ArgumentParser(description="Auto-detekcia hovoriacich z videa (gender + count + content)")
    ap.add_argument("video", help="Input video MP4")
    ap.add_argument("--lang", default="sk", help="Cieľový jazyk pre voice design suggestion")
    ap.add_argument("--json", action="store_true", help="JSON output (machine-readable)")
    ap.add_argument("--no-content", action="store_true",
                    help="Vynechaj content-type detekciu (rýchlejšie, bez Whisper)")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        sys.exit(f"Video neexistuje: {video_path}")

    vocals_path = get_demucs_vocals(video_path)
    pitch = analyze_pitch(vocals_path)

    if "error" in pitch:
        result = {"video": str(video_path), "error": pitch["error"], "n_voiced": pitch.get("n_voiced", 0)}
        print(json.dumps(result) if args.json else f"CHYBA: {pitch['error']}")
        sys.exit(1)

    gender, confidence, alt_gender = classify_gender(
        pitch["median_f0"], pitch["p25_f0"], pitch["p75_f0"], pitch["std_f0"])
    multi = detect_multispeaker(pitch["f0"])
    suggestion = suggest_voice_design(gender, multi["is_bimodal"], args.lang)

    # Content type detection (cez Whisper sample + keyword regex)
    content = {"content_type": "general", "confidence": "low", "scores": {}}
    transcript_lang = ""
    if not args.no_content:
        transcript, transcript_lang = transcribe_sample(video_path)
        if transcript:
            content = classify_content(transcript)
            preview = transcript[:200].replace("\n", " ")
            print(f"[CONTENT] Lang={transcript_lang}, preview: {preview}...", flush=True)
        else:
            print("[CONTENT] Whisper transcription failed — content_type=general", flush=True)

    ov_params = suggest_omnivoice_params(
        gender_confidence=confidence,
        multispeaker=multi["is_bimodal"],
        content_type=content["content_type"],
    )

    result = {
        "video": video_path.name,
        "median_f0_hz": round(pitch["median_f0"], 1),
        "mean_f0_hz": round(pitch["mean_f0"], 1),
        "std_f0": round(pitch["std_f0"], 1),
        "voiced_ratio": round(pitch["voiced_ratio"], 2),
        "gender": gender,
        "confidence": confidence,
        "alt_gender": alt_gender,
        "multispeaker": multi["is_bimodal"],
        "speaker_count_estimated": multi["speaker_count_estimated"],
        "peaks_hz": multi["peaks_hz"],
        "suggested_voice_design": suggestion,
        "suggested_speaker_gender": "feminine" if gender == "female" else "masculine",
        "suggested_content_type": content["content_type"],
        "content_confidence": content["confidence"],
        "content_scores": content.get("scores", {}),
        "suggested_diffusion_steps": ov_params["diffusion_steps"],
        "suggested_guidance_scale": ov_params["guidance_scale"],
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print()
        print(f"=== Speaker analysis: {video_path.name} ===")
        print(f"Median F0:        {result['median_f0_hz']} Hz")
        print(f"Mean F0:          {result['mean_f0_hz']} Hz  (std={result['std_f0']})")
        print(f"Voiced ratio:     {result['voiced_ratio']:.0%}")
        print(f"Detected peaks:   {result['peaks_hz']} Hz")
        print()
        conf_label = {"high":"VYSOKÁ", "medium":"STREDNÁ", "low":"NÍZKA"}.get(confidence, confidence)
        alt_str = f" (alt: {alt_gender})" if alt_gender else ""
        print(f"GENDER:           {gender.upper()}{alt_str}")
        print(f"CONFIDENCE:       {conf_label}")
        print(f"MULTISPEAKER:     {'YES (~%d speakers)' % result['speaker_count_estimated'] if multi['is_bimodal'] else 'NO (single speaker)'}")
        print()
        print(f"💡 Suggestion: voice design = '{suggestion}'")
        print(f"   speaker_gender = '{result['suggested_speaker_gender']}'")
        if confidence == "medium":
            print(f"   ⚠ Borderline F0 ({pitch['median_f0']:.0f} Hz) — možno chceš {alt_gender} namiesto {gender}")
        print()
        ct = result["suggested_content_type"]
        cc = result["content_confidence"]
        cc_label = {"high":"VYSOKÁ","medium":"STREDNÁ","low":"NÍZKA"}.get(cc, cc)
        print(f"CONTENT TYPE:     {ct.upper()} (confidence: {cc_label})")
        if result["content_scores"]:
            top3 = sorted(result["content_scores"].items(), key=lambda x: -x[1])[:3]
            print(f"   Top hits:      {', '.join(f'{k}={v}' for k,v in top3)}")
        print()
        print(f"OMNIVOICE PARAMS: diffusion_steps={result['suggested_diffusion_steps']}, "
              f"guidance_scale={result['suggested_guidance_scale']}")


if __name__ == "__main__":
    main()
