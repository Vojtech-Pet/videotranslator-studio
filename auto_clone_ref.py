#!/usr/bin/env python3
"""
Auto-clone ref voice z input videa:
  1. Demucs separate vocals (use cached if exists)
  2. Find clean ~5s speech section (high RMS, low silence, no noise bursts)
  3. Apply cleanup pipeline (highpass, denoise, fade-out)
  4. Save as voices/<video_stem>_clone.wav + ref_text cache (manual or Whisper)

Použitie:
  ./auto_clone_ref.py video.mp4 [--out voices/clone.wav] [--ref-lang en]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent
VOICES_DIR = REPO / "voices"
CACHE_REF_TEXT = VOICES_DIR / ".ref_text_cache"
DEMUCS_CACHE = REPO / "work" / "demucs_cache"


def get_demucs_vocals(video_path: Path) -> Path:
    """Run demucs (cached) on video, return path to vocals.wav."""
    stem = video_path.stem
    cached = DEMUCS_CACHE / stem / "vocals.wav"
    if cached.exists() and cached.stat().st_size > 1000:
        print(f"[CLONE] Cached vocals: {cached}")
        return cached

    DEMUCS_CACHE.mkdir(parents=True, exist_ok=True)
    audio_in = DEMUCS_CACHE / f"{stem}_full.wav"
    if not audio_in.exists():
        print(f"[CLONE] Extract audio z {video_path.name}...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "2", "-ar", "44100",
             "-c:a", "pcm_s16le", str(audio_in)],
            capture_output=True, check=True,
        )
    print(f"[CLONE] Demucs separate vocals (htdemucs)...")
    out_dir = DEMUCS_CACHE / "tmp"
    out_dir.mkdir(exist_ok=True)
    py = "/mnt/tts_data/miniforge3/envs/musetalk_env/bin/python"
    subprocess.run(
        [py, "-m", "demucs.separate",
         "-n", "htdemucs", "--two-stems", "vocals",
         str(audio_in), "-o", str(out_dir)],
        check=True,
    )
    src = out_dir / "htdemucs" / audio_in.stem / "vocals.wav"
    target = DEMUCS_CACHE / stem / "vocals.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.move(str(src), str(target))
    # Aj no_vocals do cache pre master_pro
    src2 = out_dir / "htdemucs" / audio_in.stem / "no_vocals.wav"
    if src2.exists():
        shutil.move(str(src2), str(target.parent / "no_vocals.wav"))
    shutil.rmtree(out_dir, ignore_errors=True)
    audio_in.unlink(missing_ok=True)
    return target


def find_clean_section(vocals_path: Path, target_dur: float = 5.5) -> tuple[float, float]:
    """Find clean speech section in vocals — high RMS, low silence, no noise bursts.
    Returns (start_s, dur_s)."""
    # Load mono 24kHz
    tmp = vocals_path.with_name(vocals_path.stem + "_24k.wav")
    if not tmp.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(vocals_path), "-ac", "1", "-ar", "24000",
             "-c:a", "pcm_s16le", str(tmp)],
            capture_output=True, check=True,
        )
    audio, sr = sf.read(str(tmp))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    tmp.unlink(missing_ok=True)

    win = int(target_dur * sr)
    hop = sr  # 1s hop
    best = None
    # Skip first 30s (intro), last 30s (outro)
    start_pos = min(30 * sr, len(audio) // 4)
    end_pos = max(len(audio) - 30 * sr, len(audio) * 3 // 4)
    for start in range(start_pos, end_pos - win, hop):
        s = audio[start:start + win]
        rms_db = 20 * np.log10(np.sqrt(np.mean(s**2)) + 1e-9)
        silence = float(np.mean(np.abs(s) < 0.005))
        # Per 100ms ZC, max value
        max_zc = 0
        for i in range(0, len(s) - sr // 10, sr // 10):
            sub = s[i:i + sr // 10]
            zc = np.sum(np.diff(np.sign(sub)) != 0) / len(sub)
            if zc > max_zc:
                max_zc = zc
        # Penalty pre noise bursts: ZC > 0.25 je veľmi nežiaduce (scream, breath, click)
        if max_zc > 0.30:
            continue  # skip — explicit reject
        # Score: hlasno, málo silence, žiadne high-ZC
        score = rms_db - silence * 50 - max(0, max_zc - 0.15) * 200
        if best is None or score > best[0]:
            best = (score, start / sr, rms_db, silence, max_zc)
    if best is None:
        # Fallback — relaxnut max_zc
        for start in range(start_pos, end_pos - win, hop):
            s = audio[start:start + win]
            rms_db = 20 * np.log10(np.sqrt(np.mean(s**2)) + 1e-9)
            silence = float(np.mean(np.abs(s) < 0.005))
            score = rms_db - silence * 50
            if best is None or score > best[0]:
                best = (score, start / sr, rms_db, silence, 0.0)
    print(f"[CLONE] Best section: t={best[1]:.1f}s RMS={best[2]:.1f}dB "
          f"silence={best[3]*100:.0f}% maxZC={best[4]:.2f}")
    return best[1], target_dur


def cleanup_ref(in_path: Path, out_path: Path, start: float, dur: float, european: bool = False):
    """Extract section + cleanup + fade-out.
    european=True: aplikuje EQ ktorý posunie hlas k slavskému/európskemu charakteru.
    Cieľ: redukovať charakteristické US/EN frekvencie ktoré OmniVoice klonuje
    (twang ~2.2kHz, bright sibilanty ~3.5kHz, americký vowel spread ~1kHz, upper presence ~5.5kHz)
    a pridať Slavic warmth (200Hz). Lowpass 7500Hz potlačí najjasnejšie EN konsonanty.
    """
    step1 = out_path.with_name(out_path.stem + "_step1.wav")
    af1_parts = [
        "highpass=f=70",
        "afftdn=nr=8:nt=w:om=o",
        "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB:detection=peak",
        "areverse,silenceremove=start_periods=1:start_duration=0.15:start_threshold=-40dB:detection=peak,areverse",
    ]
    if european:
        # EQ-only shaping (formant shift trick rozbíjal audio → revertnuté).
        # Posilnená verzia (2026-05-08) — viac aggressive proti EN/US prízvuku.
        af1_parts += [
            "equalizer=f=200:t=q:w=0.9:g=+2.5",   # warm bass foundation (Slavic)
            "equalizer=f=400:t=q:w=1.0:g=-1.0",   # tame mild boxiness
            "equalizer=f=1000:t=q:w=1.0:g=-1.5",  # tame American vowel spread (open vowels)
            "equalizer=f=2200:t=q:w=1.4:g=-3.5",  # KEY: stronger twang tame (-1dB more)
            "equalizer=f=3500:t=q:w=1.2:g=-3.0",  # stronger upper sibilant tame
            "equalizer=f=5500:t=q:w=1.0:g=-2.0",  # tame upper presence (US brightness)
            "lowpass=f=7500",                      # tighter HF cap = smoother SK feel
        ]
    af1_parts.append("loudnorm=I=-20:TP=-2:LRA=7")
    af1 = ",".join(af1_parts)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(dur), "-i", str(in_path),
         "-af", af1, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(step1)],
        capture_output=True, check=True,
    )
    # Fade-out 200ms
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(step1)],
        capture_output=True, text=True, check=True,
    )
    actual_dur = float(r.stdout.strip())
    fade_start = max(0.0, actual_dur - 0.20)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(step1),
         "-af", f"afade=t=out:st={fade_start:.3f}:d=0.20:curve=hsin",
         "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(out_path)],
        capture_output=True, check=True,
    )
    step1.unlink(missing_ok=True)


def blend_with_sk_ref(video_clone: Path, sk_ref: Path, sk_ref_dur: float = 2.0,
                      xfade: float = 0.3) -> Path | None:
    """Strategy B (mix): vyextrahuje prvých `sk_ref_dur` sekúnd zo `sk_ref`, urobí
    cross-fade s `video_clone` a vráti blended WAV v rovnakom formáte (24kHz mono).

    Idea: OmniVoice dostane "obojaké" referenčné audio — krátky SK natívny segment
    (prozódia, akcent) potom video-speakerov hlas (timbre, identita). Model klonuje
    blend; v praxi výsledok záleží od dĺžok a poradia.

    Vrátí Path k blended WAV, alebo None pri zlyhaní.
    """
    blended = video_clone.with_name(video_clone.stem + "_blend.wav")
    sk_trim = video_clone.with_name(video_clone.stem + "_sktrim.wav")
    try:
        # 1) Trim SK ref na sk_ref_dur, normalize to 24kHz mono, level-match (-20 LUFS)
        subprocess.run(
            ["ffmpeg", "-y", "-t", f"{sk_ref_dur:.2f}", "-i", str(sk_ref),
             "-af", "loudnorm=I=-20:TP=-2:LRA=7,afade=t=out:st="
                    f"{max(0, sk_ref_dur - 0.10):.2f}:d=0.10",
             "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(sk_trim)],
            capture_output=True, check=True, timeout=30,
        )
        # 2) Cross-fade concat: [sk_trim][video_clone] s acrossfade
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(sk_trim), "-i", str(video_clone),
             "-filter_complex",
             f"[0:a][1:a]acrossfade=d={xfade:.2f}:c1=tri:c2=tri[a]",
             "-map", "[a]", "-ac", "1", "-ar", "24000",
             "-c:a", "pcm_s16le", str(blended)],
            capture_output=True, check=True, timeout=30,
        )
        sk_trim.unlink(missing_ok=True)
        return blended
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-300:]
        print(f"[CLONE] Blend zlyhal: {err}")
        sk_trim.unlink(missing_ok=True)
        blended.unlink(missing_ok=True)
        return None


# Seed text (~6s SK speech) — Chatterbox prejde s týmto textom + video clone ref → vyrobí
# SK-flavored ref WAV, ktorý nahradí raw video clone pre OmniVoice. Cieľ: dať OmniVoice
# referenciu ktorá má SK akcent zapečený, ale stále s timbre videoo speakera.
_CHATTERBOX_SEED_TEXT = (
    "Vitajte, dnes si spolu vysvetlíme niekoľko zaujímavých vecí. "
    "Začneme od základov a postupne sa dostaneme aj k zložitejším témam."
)


def chatterbox_make_sk_ref(video_clone_path: Path, out_path: Path,
                           model_path: Path, src_dir: Path) -> bool:
    """Run Chatterbox SK with video_clone as audio_prompt + SK seed text.
    Output → SK-flavored ref WAV (replaces raw video clone for OmniVoice).

    PRO config (CONFIG_ANTICUT — production-tested defaults pre demo kvalitu):
      - exaggeration=0.50, cfg_weight=0.50, temperature=0.60 (best demo quality)
      - safe_mode=False (PRO neutruncuje, plná prozódia)
      - retry až 3× ak output je príliš krátky / failed; pri opakovaných failoch
        fallback na safe_mode (anti-halucinácia poistka)
    Returns True on success.
    """
    py = "/mnt/tts_data/miniforge3/envs/musetalk_env/bin/python"
    scripts_dir = REPO / "scripts"
    def _make_code(safe: bool) -> str:
        if safe:
            params = "    safe_mode=True,\n    exaggeration=0.45,\n    cfg_weight=0.50,\n    temperature=0.55,\n"
        else:
            # PRO / CONFIG_ANTICUT — production preset
            params = "    safe_mode=False,\n    exaggeration=0.50,\n    cfg_weight=0.50,\n    temperature=0.60,\n"
        return (
            f"import sys\n"
            f"sys.path.insert(0, '{scripts_dir}')\n"
            f"sys.path.insert(0, '{src_dir}')\n"
            f"from tts import chatterbox_tts\n"
            f"from pathlib import Path\n"
            f"chatterbox_tts(\n"
            f"    text={_CHATTERBOX_SEED_TEXT!r},\n"
            f"    output_path=Path('{out_path}'),\n"
            f"    model_path='{model_path}',\n"
            f"    audio_prompt_path='{video_clone_path}',\n"
            f"    device='cuda',\n"
            f"    ultra_safe=False,\n"
            f"{params}"
            f"    use_torch_compile=True,\n"
            f")\n"
            f"print('OK', flush=True)\n"
        )

    # Try PRO config first (3 attempts), fallback to safe_mode if all fail
    print(f"[CLONE] Chatterbox SK ref-gen (PRO config: CONFIG_ANTICUT)...", flush=True)
    for attempt in range(3):
        out_path.unlink(missing_ok=True)
        r = subprocess.run([py, "-c", _make_code(safe=False)],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[CLONE] ✓ Chatterbox SK ref (PRO) vytvorený "
                  f"(attempt {attempt+1}/3, {out_path.stat().st_size//1024} KB)", flush=True)
            return True
        err = (r.stderr or "")[-200:]
        print(f"[CLONE] PRO attempt {attempt+1} zlyhal: {err}", flush=True)

    # Fallback: safe_mode (poistka proti halucináciam)
    print(f"[CLONE] PRO 3× zlyhal — fallback na safe_mode...", flush=True)
    for attempt in range(2):
        out_path.unlink(missing_ok=True)
        r = subprocess.run([py, "-c", _make_code(safe=True)],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[CLONE] ✓ Chatterbox SK ref (safe fallback) vytvorený "
                  f"({out_path.stat().st_size//1024} KB)", flush=True)
            return True
    print(f"[CLONE] Chatterbox ref-gen úplne zlyhal", flush=True)
    return False


def cache_ref_text_whisper(ref_path: Path, lang: str = "en"):
    """Whisper transcribe ref audio, save to .ref_text_cache/<stem>.txt."""
    CACHE_REF_TEXT.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_REF_TEXT / f"{ref_path.stem}.txt"
    py = "/mnt/tts_data/miniforge3/envs/musetalk_env/bin/python"
    code = (
        f"from faster_whisper import WhisperModel\n"
        f"wm = WhisperModel('base', device='cuda', compute_type='float16')\n"
        f"segs, _ = wm.transcribe('{ref_path}', language='{lang}', beam_size=1)\n"
        f"text = ' '.join(s.text.strip() for s in segs).strip()\n"
        f"open('{cache_file}', 'w').write(text)\n"
        f"print(text)\n"
    )
    r = subprocess.run([py, "-c", code], capture_output=True, text=True)
    text = r.stdout.strip()
    print(f"[CLONE] ref_text ({lang}): {text!r}")
    return text


def main():
    ap = argparse.ArgumentParser(description="Auto-clone voice ref z input videa")
    ap.add_argument("video", help="Input video MP4")
    ap.add_argument("--out", default=None,
                    help="Output ref WAV (default: voices/<stem>_clone.wav)")
    ap.add_argument("--ref-lang", default="en",
                    help="Language pre Whisper transcribe ref text (default: en)")
    ap.add_argument("--target-dur", type=float, default=5.5,
                    help="Target ref dur (default: 5.5s)")
    ap.add_argument("--european", action=argparse.BooleanOptionalAction, default=True,
                    help="Aplikuje EQ + formant shift pre európsky/slavský charakter "
                         "(default ON pre clone z EN videa). --no-european vypne.")
    ap.add_argument("--sk-ref-blend", default=None,
                    help="Cesta k SK natívnej WAV referencii. Prependuje sa pred video clone "
                         "s cross-fade — OmniVoice klonuje 'blend' (SK prozódia/akcent + "
                         "video timbre). Default: žiadny blend (čistý video clone).")
    ap.add_argument("--sk-ref-dur", type=float, default=2.0,
                    help="Koľko sekúnd SK ref použiť pri blende (default 2.0s — krátke, "
                         "aby video timbre ostalo dominantný).")
    ap.add_argument("--via-chatterbox", action=argparse.BooleanOptionalAction, default=True,
                    help="Použi Chatterbox SK ako ref-generator: vyrobí SK-flavored ref WAV "
                         "(SK akcent + video timbre) ktorý sa použije pre OmniVoice. "
                         "Default ON ak je Chatterbox dostupný; auto-disable ak chýba. "
                         "--no-via-chatterbox = raw video clone (americký prizvuk).")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        sys.exit(f"Video neexistuje: {video_path}")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        VOICES_DIR.mkdir(exist_ok=True)
        out_path = VOICES_DIR / f"{video_path.stem}_clone.wav"

    print(f"==> Auto-clone ref z {video_path.name}")
    print(f"   output: {out_path}")
    print(f"   lang:   {args.ref_lang}")
    print()

    # 1) Demucs vocals
    vocals = get_demucs_vocals(video_path)

    # 2) Find clean section
    start, dur = find_clean_section(vocals, args.target_dur)

    # 3) Cleanup + extract
    cleanup_ref(vocals, out_path, start, dur, european=args.european)
    if args.european:
        print(f"[CLONE] European EQ applied (Slavic warmth +200Hz, twang -3.5dB @ 2.2kHz, "
              f"sibilants -3dB @ 3.5kHz, presence -2dB @ 5.5kHz, LP 7.5kHz)")

    # 3b) Blend with SK native reference (Strategy B — quick proof-of-concept)
    if args.sk_ref_blend:
        sk_ref = Path(args.sk_ref_blend).expanduser().resolve()
        if not sk_ref.exists():
            print(f"[CLONE] ⚠ SK ref blend súbor neexistuje: {sk_ref} — preskakujem blend")
        else:
            blended = blend_with_sk_ref(out_path, sk_ref, sk_ref_dur=args.sk_ref_dur)
            if blended:
                # Replace clean clone with blended version
                blended.replace(out_path)
                print(f"[CLONE] ✓ Blended s SK ref ({sk_ref.name}, prvých {args.sk_ref_dur:.1f}s)")

    # 3c) Chatterbox SK ref-generator → SK-flavored ref pre OmniVoice
    # Chatterbox vezme video clone WAV ako audio_prompt + krátky SK seed text
    # → vyrobí ~6s SK speech v cloned voice (SK akcent + video timbre).
    # Tento WAV nahradí raw video clone — OmniVoice bude klonovať z neho a output
    # bude mať natívny SK akcent zachovaný cez celé video.
    chatterbox_used = False
    if args.via_chatterbox:
        cb_model_paths = [
            REPO / "models" / "chatterbox" / "t3_sk_v2.2.safetensors",
            Path("/mnt/tts_data/VideoTranslator_studio/models/chatterbox/t3_sk_v2.2.safetensors"),
        ]
        cb_src_dirs = [
            REPO.parent / "Ai" / "chatterbox_git" / "src",
            REPO.parent / "chatterbox_git" / "src",
        ]
        cb_model = next((p for p in cb_model_paths if p.exists()), None)
        cb_src = next((d for d in cb_src_dirs if d.exists()), None)
        if cb_model and cb_src:
            cb_ref_path = out_path.with_name(out_path.stem + "_sk.wav")
            ok = chatterbox_make_sk_ref(out_path, cb_ref_path, cb_model, cb_src)
            if ok:
                cb_ref_path.replace(out_path)  # SK-flavored ref nahradí raw video clone
                chatterbox_used = True
                print(f"[CLONE] ✓ Chatterbox SK ref-gen použitý — OmniVoice bude klonovať SK-flavored ref")
        else:
            missing = "model" if not cb_model else "source dir"
            print(f"[CLONE] Chatterbox {missing} chýba — používa sa raw video clone (môže priniesť US prizvuk)")

    # 4) Ref text:
    # - Chatterbox použitý → ref_text = seed text (vieme presne čo Chatterbox povedal,
    #   nemusíme to znova transkribovať Whisperom — Whisper na Chatterbox SK output
    #   produkuje garbled transcript, čo spôsobí mismatch a halucinácie v OmniVoice)
    # - Inak → Whisper transcribe (raw video clone má originálny EN text)
    if chatterbox_used:
        # Save ref_text do OBOCH cache lokácií:
        #   voices/.ref_text_cache/<stem>.txt — legacy umiestnenie
        #   <ref_dir>/.ref_text_cache/<stem>.txt — kde test_omnivoice_batch hľadá
        CACHE_REF_TEXT.mkdir(parents=True, exist_ok=True)
        (CACHE_REF_TEXT / f"{out_path.stem}.txt").write_text(_CHATTERBOX_SEED_TEXT, encoding="utf-8")
        local_cache = out_path.parent / ".ref_text_cache"
        local_cache.mkdir(parents=True, exist_ok=True)
        (local_cache / f"{out_path.stem}.txt").write_text(_CHATTERBOX_SEED_TEXT, encoding="utf-8")
        print(f"[CLONE] ref_text (sk, seed): {_CHATTERBOX_SEED_TEXT[:100]}...")
    else:
        cache_ref_text_whisper(out_path, args.ref_lang)
        # Mirror Whisper cache do <ref_dir>/.ref_text_cache/ pre test_omnivoice_batch
        whisper_cache = CACHE_REF_TEXT / f"{out_path.stem}.txt"
        if whisper_cache.exists():
            local_cache = out_path.parent / ".ref_text_cache"
            local_cache.mkdir(parents=True, exist_ok=True)
            (local_cache / f"{out_path.stem}.txt").write_text(
                whisper_cache.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\n✓ Clone ref: {out_path}")
    print(f"  Použij: ./test_omnivoice_batch.py --ref '{out_path}'")


if __name__ == "__main__":
    main()
