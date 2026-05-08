#!/usr/bin/env python3
"""
Batch OmniVoice tester — vygeneruje audio pre VŠETKY segmenty z preloženého
videa, KOMPLETNE BEZ PIPELINE (žiadny mini_level11, ADAPT, post-processing).

Použitie:
  ./test_omnivoice_batch.py                     # default: 2 Years of C++ Programming
  ./test_omnivoice_batch.py --segments cesta_k_segments.json
  ./test_omnivoice_batch.py --out /tmp/audio.wav
  ./test_omnivoice_batch.py --num-step 96 --guidance 1.8

Výstup: jeden WAV súbor (všetky segmenty zlepené raw concat) + MP3.
Žiadny silence trim, žiadne EQ, žiadny timeline alignment.

Model load 1× v subprocess, generuje všetky segmenty postupne. Rýchlejšie ako 94×
volanie jednotlivého test_omnivoice.py.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def omnivoice_speak_symbols(text: str) -> str:
    """Inline kópia z tts.py — pre-process text pre OmniVoice."""
    out = text
    # Tech symboly
    out = re.sub(r'\bC\+\+', 'C plus plus', out)
    out = re.sub(r'\bc\+\+', 'C plus plus', out)
    out = re.sub(r'\bC#', 'C sharp', out)
    out = re.sub(r'\bF#', 'F sharp', out)
    out = re.sub(r'\.NET\b', 'dot NET', out)
    # Úvodzovky
    out = re.sub(r'[„""""\'\']', '', out)
    # Em-dash / en-dash → čiarka (predtým spôsoboval 0.6-0.9s pauzu v audio)
    out = out.replace('—', ',').replace('–', ',')
    # Ellipsis: mid-sentence → čiarka, trailing → bodka
    out = out.replace('…', '.')
    out = re.sub(r'\.{3,}\s*([A-Za-zÁ-žÀ-ÿ])', r', \1', out)
    out = re.sub(r'(?:\.\s*){2,}\.', '.', out)
    out = re.sub(r'\.{2,}', '.', out)
    out = re.sub(r'\s+', ' ', out).strip()
    if out and out[-1] == ',':
        out = out[:-1] + '.'
    return out


def smart_trim_tail(wav_path: Path, threshold_db: float = -42.0, fade_ms: int = 60):
    """Smart trailing trim: silenceremove (-42dB threshold) + fade-out 60ms na koniec.
    Eliminuje breath/click/noise artefakty po reči bez rezania samohlasok.
    Fade-out je krátky (60ms) → nepočuteľný v reči, ale potlačí ostré transienty na konci."""
    tmp = wav_path.with_name(wav_path.stem + "_trimmed.wav")
    chain = (
        f"areverse,"
        f"silenceremove=start_periods=1:start_duration=0.05:start_threshold={threshold_db}dB,"
        f"areverse,"
        f"afade=t=out:st=0:d={fade_ms/1000:.3f}:curve=tri"
    )
    # Note: afade with st=0 + areverse trick doesn't work directly. Use a 2-step approach.
    # Step 1: trim trailing silence
    step1 = wav_path.with_name(wav_path.stem + "_s1.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path),
         "-af", f"areverse,silenceremove=start_periods=1:start_duration=0.05:start_threshold={threshold_db}dB,areverse",
         "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(step1)],
        capture_output=True, timeout=30,
    )
    if not step1.exists() or step1.stat().st_size < 1000:
        step1.unlink(missing_ok=True)
        return
    # Step 2: get duration, apply fade-out on last fade_ms
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(step1)],
        capture_output=True, text=True,
    )
    try:
        dur = float(r.stdout.strip())
    except Exception:
        step1.unlink(missing_ok=True)
        return
    fade_start = max(0.0, dur - fade_ms / 1000)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(step1),
         "-af", f"afade=t=out:st={fade_start:.3f}:d={fade_ms/1000:.3f}:curve=tri",
         "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True, timeout=30,
    )
    step1.unlink(missing_ok=True)
    if tmp.exists() and tmp.stat().st_size > 1000:
        tmp.replace(wav_path)
    else:
        tmp.unlink(missing_ok=True)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REF = REPO_ROOT / "voices" / "chatterbox_sk_greeting_studio.wav"
DEFAULT_PY = "/mnt/tts_data/miniforge3/envs/f5tts_env/bin/python"
DEFAULT_REPO = "k2-fsa/OmniVoice"
DEFAULT_SEGMENTS = REPO_ROOT / "temp" / "2 Years of C++ Programming" / "2 Years of C++ Programming_sk_segments.json"
DEFAULT_OUT_DIR = REPO_ROOT / "input" / "sk"

# Subprocess runner — beží v f5tts_env, loaduje model raz, generuje všetky segmenty.
# Podporuje multi-voice cez per-segment ref_audios + ref_texts (parallel lists).
RUNNER = r'''
import json, sys, torch, soundfile as sf
from pathlib import Path
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

p = json.loads(sys.stdin.read())

texts = p['texts']                 # list of strings
N = len(texts)

# Per-segment refs (multi-voice): if "ref_audios" present, use that list.
# Else broadcast single "ref_audio" to all segments.
if p.get("ref_audios"):
    ref_audios = p["ref_audios"]   # list of paths, len=N
    ref_texts = p.get("ref_texts") or [""] * N  # list, len=N
    multi = True
else:
    ref_audios = [p["ref_audio"]] * N
    ref_texts = [p.get("ref_text") or ""] * N
    multi = False

# Whisper auto-transcribe pre prázdne ref_text-y
unique_refs_to_transcribe = sorted({(a, ref_texts[i]) for i, a in enumerate(ref_audios)
                                     if a and not ref_texts[i]})
if unique_refs_to_transcribe:
    try:
        from faster_whisper import WhisperModel
        wm = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
        text_cache = {}
        for ra, _ in unique_refs_to_transcribe:
            print(f"[REF] Whisper transcribe {ra}...", flush=True)
            try:
                segs, _ = wm.transcribe(ra, language=p.get("ref_lang", "sk"), beam_size=1)
                text_cache[ra] = " ".join(s.text.strip() for s in segs).strip() or "Toto je vzor slovenského hlasu."
                print(f"[REF]   {text_cache[ra]!r}", flush=True)
            except Exception as e:
                print(f"[REF] {ra} Whisper failed: {e}", flush=True)
                text_cache[ra] = "Toto je vzor slovenského hlasu."
        del wm
        # Apply transcribed text to all empty slots
        for i in range(N):
            if not ref_texts[i] and ref_audios[i] in text_cache:
                ref_texts[i] = text_cache[ref_audios[i]]
    except Exception as e:
        print(f"[REF] Whisper batch zlyhal: {e}", flush=True)

print(f"[OMNIVOICE] Loading {p['repo']}...", flush=True)
model = OmniVoice.from_pretrained(p['repo'], device_map='cuda:0', dtype=torch.float16)

cfg = OmniVoiceGenerationConfig(
    num_step=p['num_step'],
    guidance_scale=p['guidance_scale'],
    position_temperature=p.get('position_temperature', 3.0),
    postprocess_output=p.get('postprocess', False),
)

out_dir = Path(p['chunks_dir'])
out_dir.mkdir(parents=True, exist_ok=True)

import time
t0 = time.time()
unique_speakers_used = set()
for i, text in enumerate(texts):
    if not text or not text.strip():
        import numpy as np
        sf.write(str(out_dir / f"chunk_{i:04d}.wav"), np.zeros(int(0.3*24000), dtype="float32"), 24000)
        print(f"[{i+1}/{N}] EMPTY → silence", flush=True)
        continue
    seg_ref = ref_audios[i]
    seg_ref_text = ref_texts[i] or "Toto je vzor slovenského hlasu."
    if multi:
        unique_speakers_used.add(Path(seg_ref).stem)
    try:
        audio = model.generate(
            text=text, language=p['language'],
            ref_audio=seg_ref, ref_text=seg_ref_text,
            generation_config=cfg,
        )
        out_wav = out_dir / f"chunk_{i:04d}.wav"
        sf.write(str(out_wav), audio[0], 24000)
        dur = len(audio[0]) / 24000.0
        speaker_tag = f" [{Path(seg_ref).stem}]" if multi else ""
        print(f"[{i+1}/{N}]{speaker_tag} {dur:5.2f}s  {text[:60]!r}", flush=True)
    except Exception as e:
        print(f"[{i+1}/{N}] FAILED: {e}", flush=True)
        import numpy as np
        sf.write(str(out_dir / f"chunk_{i:04d}.wav"), np.zeros(int(0.3*24000), dtype="float32"), 24000)

if multi:
    print(f"[MULTI] Used {len(unique_speakers_used)} unique speaker refs: "
          f"{sorted(unique_speakers_used)}", flush=True)
print(f"[DONE] {N} segments in {time.time()-t0:.0f}s", flush=True)
'''


def main():
    ap = argparse.ArgumentParser(description="Batch OmniVoice TTS — všetky segmenty bez pipeline")
    ap.add_argument("--segments", default=str(DEFAULT_SEGMENTS),
                    help=f"JSON so segmentmi (default: {DEFAULT_SEGMENTS.name})")
    ap.add_argument("--out", "-o", default=None,
                    help="Output WAV cesta (default: input/sk/<stem>_STANDALONE.wav)")
    ap.add_argument("--ref", default=str(DEFAULT_REF), help="Reference audio")
    ap.add_argument("--ref-text", default="", help="Reference transcript (default: auto)")
    ap.add_argument("--lang", default="sk", help="TTS language")
    ap.add_argument("--num-step", type=int, default=64, help="Diffusion steps")
    ap.add_argument("--guidance", type=float, default=2.5, help="Guidance scale")
    ap.add_argument("--postprocess", action="store_true", help="Zapnúť OmniVoice interný silence trim")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--python", default=DEFAULT_PY)
    ap.add_argument("--chunks-dir", default="/tmp/standalone_chunks", help="Kam ukladať jednotlivé chunky")
    ap.add_argument("--keep-chunks", action="store_true", help="Nemazať chunks po concat")
    ap.add_argument("--speaker-refs", default="",
                    help="JSON path s mappingom {speaker_id: ref_wav_path}. "
                         "Ak je nastavené + segments majú speaker_id → multi-voice TTS. "
                         "Inak: single-ref použitý pre všetky segmenty.")
    args = ap.parse_args()

    seg_path = Path(args.segments).expanduser().resolve()
    if not seg_path.exists():
        sys.exit(f"Chyba: segments JSON neexistuje: {seg_path}")

    data = json.loads(seg_path.read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    raw_texts = [(s.get("text") or "").strip() for s in segs]

    # Aplikuj symbol expansion (C++ → C plus plus) — OmniVoice nečíta '++'.
    texts = [omnivoice_speak_symbols(t) for t in raw_texts]
    n_changed = sum(1 for r, t in zip(raw_texts, texts) if r != t)
    if n_changed:
        print(f"   symbol expansion: {n_changed}/{len(texts)} segmentov upravených (C++ → C plus plus, atď.)")
    print(f"==> {len(texts)} segmentov z {seg_path.name}")

    chunks_dir = Path(args.chunks_dir).expanduser().resolve()
    if chunks_dir.exists():
        for f in chunks_dir.glob("chunk_*.wav"):
            f.unlink()
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    if args.out:
        out_wav = Path(args.out).expanduser().resolve()
    else:
        stem = seg_path.stem.replace("_sk_segments", "").replace("_segments", "")
        out_wav = DEFAULT_OUT_DIR / f"{stem}_STANDALONE.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Multi-voice mode: --speaker-refs JSON + segments majú speaker_id
    multi_voice = False
    speaker_refs_map = {}
    if args.speaker_refs:
        spk_refs_path = Path(args.speaker_refs).expanduser().resolve()
        if spk_refs_path.exists():
            speaker_refs_map = json.loads(spk_refs_path.read_text(encoding="utf-8"))
            # Sanity check: aspoň 1 segment má speaker_id
            n_with_speaker = sum(1 for s in segs if s.get("speaker_id"))
            if n_with_speaker > 0 and speaker_refs_map:
                multi_voice = True
                print(f"   MULTI-VOICE mode ON ({len(speaker_refs_map)} speakers, "
                      f"{n_with_speaker}/{len(segs)} segments majú speaker_id)")
        else:
            print(f"   ⚠ speaker_refs JSON neexistuje: {spk_refs_path}")

    if multi_voice:
        # Per-segment ref_audio + ref_text (parallel lists)
        # Pre každý speaker_id pozri jeho ref WAV + cached ref_text
        per_seg_ref_audios = []
        per_seg_ref_texts = []
        # Pre fallback (ak segment nemá speaker_id) použijeme prvého speakera v mape
        default_ref = next(iter(speaker_refs_map.values()))
        for s in segs:
            spk = s.get("speaker_id") or ""
            ref_path_seg = speaker_refs_map.get(spk, default_ref)
            per_seg_ref_audios.append(ref_path_seg)
            # Try load cached ref_text
            rp = Path(ref_path_seg)
            cache = rp.parent / ".ref_text_cache" / f"{rp.stem}.txt"
            txt = cache.read_text(encoding="utf-8").strip() if cache.exists() else ""
            per_seg_ref_texts.append(txt)
        payload = {
            "texts": texts,
            "chunks_dir": str(chunks_dir),
            "repo": args.repo,
            "ref_audios": per_seg_ref_audios,    # multi-voice
            "ref_texts": per_seg_ref_texts,      # multi-voice
            "ref_lang": args.lang,
            "language": args.lang,
            "num_step": args.num_step,
            "guidance_scale": args.guidance,
            "postprocess": args.postprocess,
        }
        print(f"   refs:     {len(set(per_seg_ref_audios))} unique speaker WAVs")
    else:
        # Single-ref (existujúce správanie)
        ref_path = Path(args.ref).expanduser().resolve()
        ref_text = args.ref_text
        if not ref_text:
            cache_path = ref_path.parent / ".ref_text_cache" / f"{ref_path.stem}.txt"
            if cache_path.exists():
                ref_text = cache_path.read_text(encoding="utf-8").strip()
                print(f"   ref_text: cached from {cache_path.name}: {ref_text!r}")
            else:
                print(f"   ref_text: WARN no cache, fallback bude použitý")
        payload = {
            "texts": texts,
            "chunks_dir": str(chunks_dir),
            "repo": args.repo,
            "ref_audio": str(ref_path),
            "ref_text": ref_text,
            "ref_lang": args.lang,
            "language": args.lang,
            "num_step": args.num_step,
            "guidance_scale": args.guidance,
            "postprocess": args.postprocess,
        }

    if multi_voice:
        print(f"   ref:      [multi-voice mode: {len(speaker_refs_map)} speakers]")
    else:
        print(f"   ref:      {payload['ref_audio']}")
    print(f"   out_wav:  {out_wav}")
    print(f"   chunks:   {chunks_dir}")
    print(f"   num_step: {args.num_step}")
    print(f"   guidance: {args.guidance}")
    print(f"   postproc: {args.postprocess}")
    print()

    # Spusti TTS subprocess
    proc = subprocess.run(
        [args.python, "-c", RUNNER],
        input=json.dumps(payload).encode("utf-8"),
        timeout=3600,  # 1 hour max
    )
    if proc.returncode != 0:
        sys.exit(f"OmniVoice subprocess failed (rc={proc.returncode})")

    # Concat chunks → out_wav
    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunks:
        sys.exit("No chunks generated")
    print(f"\n==> Concat {len(chunks)} chunks → {out_wav}")
    list_file = chunks_dir / "concat.txt"
    list_file.write_text("\n".join(f"file '{c.absolute()}'" for c in chunks) + "\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(out_wav)],
        capture_output=True,
    )

    # Aj MP3
    out_mp3 = out_wav.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out_wav), "-codec:a", "libmp3lame", "-b:a", "192k", str(out_mp3)],
        capture_output=True,
    )

    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out_wav)],
        capture_output=True, text=True,
    ).stdout.strip()

    print(f"\n✓ WAV: {out_wav} ({out_wav.stat().st_size//1024//1024} MB, {dur}s)")
    print(f"✓ MP3: {out_mp3} ({out_mp3.stat().st_size//1024//1024} MB)")

    if not args.keep_chunks:
        for f in chunks_dir.glob("chunk_*.wav"):
            f.unlink()
        list_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
