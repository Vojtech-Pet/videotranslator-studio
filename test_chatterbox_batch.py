#!/usr/bin/env python3
"""
Chatterbox SK 2.2 batch — paralela k test_omnivoice_batch.py.

Generuje per-segment WAV z segments JSON cez Chatterbox SK 2.2 fine-tune.
Output má **natívny SK akcent** (model bol trénovaný na SK speech),
voice clone preberá iba timbre z reference, NIE akcent.

Použitie:
  ./test_chatterbox_batch.py --segments segments.json --ref voices/clone.wav
  ./test_chatterbox_batch.py --segments seg.json --ref ref.wav --out out.wav
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _fallback_to_omnivoice(args, seg_path: Path, ref_path: Path):
    """Spustí test_omnivoice_batch.py s rovnakými args. Použité keď Chatterbox
    nie je dostupný (chýbajúci model alebo source).
    """
    import subprocess
    omni_script = REPO / "test_omnivoice_batch.py"
    if not omni_script.exists():
        sys.exit(f"OmniVoice fallback skript nenájdený: {omni_script}")
    out_path = (Path(args.out).expanduser().resolve() if args.out
                else REPO / "input" / "sk" / f"{seg_path.stem.replace('_sk_segments','')}_chatterbox.wav")
    cmd = [sys.executable, str(omni_script),
           "--segments", str(seg_path),
           "--ref", str(ref_path),
           "--out", str(out_path),
           "--chunks-dir", args.chunks_dir]
    if args.keep_chunks:
        cmd.append("--keep-chunks")
    print(f"[FALLBACK] {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(description="Chatterbox SK batch TTS")
    ap.add_argument("--segments", required=True, help="Segments JSON")
    ap.add_argument("--ref", required=True, help="Reference WAV pre voice clone")
    ap.add_argument("--out", "-o", default=None, help="Output WAV (default: input/sk/<stem>_chatterbox.wav)")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="Nezmaže per-segment chunks (pre downstream alignment)")
    ap.add_argument("--chunks-dir", default="/tmp/chatterbox_chunks",
                    help="Adresár pre chunky (default /tmp/chatterbox_chunks)")
    ap.add_argument("--model", default=str(REPO / "models" / "chatterbox" / "t3_sk_v2.2.safetensors"),
                    help="Path k Chatterbox SK 2.2 .safetensors")
    args = ap.parse_args()

    seg_path = Path(args.segments).expanduser().resolve()
    if not seg_path.exists():
        sys.exit(f"Segments JSON neexistuje: {seg_path}")
    ref_path = Path(args.ref).expanduser().resolve()
    if not ref_path.exists():
        sys.exit(f"Ref WAV neexistuje: {ref_path}")

    data = json.loads(seg_path.read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    print(f"==> Chatterbox batch")
    print(f"   segments: {seg_path}")
    print(f"   ref:      {ref_path}")
    print(f"   N:        {len(segs)} segments")
    print()

    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — torch + chatterbox model load ~5-10s
    print("Loading Chatterbox SK 2.2 model...", flush=True)
    from tts import chatterbox_tts

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        # Fallback to v1 dir
        v1 = Path("/mnt/tts_data/VideoTranslator_studio/models/chatterbox/t3_sk_v2.2.safetensors")
        if v1.exists():
            model_path = v1
        else:
            print(f"\n[CHATTERBOX] Model nenájdený ({args.model})", flush=True)
            print(f"[CHATTERBOX] → fallback na OmniVoice (test_omnivoice_batch.py)", flush=True)
            _fallback_to_omnivoice(args, seg_path, ref_path)
            return

    # Check chatterbox source dir je dostupné
    chatterbox_src = REPO.parent / "Ai" / "chatterbox_git" / "src"
    if not chatterbox_src.exists():
        chatterbox_src = REPO.parent / "chatterbox_git" / "src"
    if not chatterbox_src.exists():
        print(f"\n[CHATTERBOX] Source dir nenájdený (Ai/chatterbox_git/src)", flush=True)
        print(f"[CHATTERBOX] → fallback na OmniVoice", flush=True)
        _fallback_to_omnivoice(args, seg_path, ref_path)
        return

    print(f"   model:    {model_path}")

    all_audio = []
    sr = 24000  # Chatterbox default output SR
    for i, seg in enumerate(segs):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
        try:
            chatterbox_tts(
                text=text,
                output_path=chunk_path,
                model_path=str(model_path),
                audio_prompt_path=str(ref_path),
                device="cuda",
            )
            audio, sr = sf.read(str(chunk_path))
        except Exception as e:
            print(f"   [{i+1:3d}/{len(segs)}] FAIL: {e}", flush=True)
            audio = np.zeros(int(0.5 * sr), dtype=np.float32)
            sf.write(str(chunk_path), audio.astype(np.float32), sr)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        all_audio.append(audio.astype(np.float32))
        print(f"   [{i+1:3d}/{len(segs)}] {len(audio)/sr:.2f}s  '{text[:60]}...'", flush=True)

    print(f"\n==> Concat {len(all_audio)} chunks", flush=True)
    full = np.concatenate(all_audio) if all_audio else np.zeros(int(0.1 * sr), dtype=np.float32)

    out_path = (Path(args.out).expanduser().resolve() if args.out
                else REPO / "input" / "sk" / f"{seg_path.stem.replace('_sk_segments','')}_chatterbox.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), full, sr)
    print(f"✓ WAV: {out_path} ({out_path.stat().st_size//1024//1024} MB, {len(full)/sr:.2f}s)")

    if not args.keep_chunks:
        for f in chunks_dir.glob("chunk_*.wav"):
            f.unlink(missing_ok=True)
        print(f"  chunks removed")


if __name__ == "__main__":
    main()
