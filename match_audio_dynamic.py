#!/usr/bin/env python3
"""
Dynamický timeline-aware match: per-segment atempo aby každý chunk sedel do svojho
slotu z segments JSON. Lepšie ako uniform speed-up — rýchlo sa zrýchlia iba dlhé
segmenty, krátke ostávajú prirodzené, gaps medzi rečou sa vyplnia tichom.

Použitie:
  ./match_audio_dynamic.py
    # default: temp/<video>/<stem>_sk_segments.json + chunks dir from last batch

  ./match_audio_dynamic.py --segments path/to/segments.json --chunks-dir path/to/chunks/
  ./match_audio_dynamic.py --out final.wav --max-factor 1.50

Vstup:
  - segments JSON: [{"start": 0.20, "end": 4.81, "text": "..."}, ...]
  - chunks dir: chunk_0000.wav, chunk_0001.wav, ... (1 súbor per segment)

Výstup:
  - WAV ktorý časovo presne sedí s videom (timeline aligned).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = REPO / "input" / "sk"
SR = 24000


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def atempo_chain(factor: float) -> str:
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor:.6f}"
    parts = []
    rem = factor
    while rem > 2.0:
        parts.append("atempo=2.0")
        rem /= 2.0
    while rem < 0.5:
        parts.append("atempo=0.5")
        rem /= 0.5
    parts.append(f"atempo={rem:.6f}")
    return ",".join(parts)


def stretch_chunk(in_wav: Path, target_dur: float, max_factor: float = 1.50,
                  min_factor: float = 0.85) -> np.ndarray:
    """Time-stretch chunk to target_dur. Returns mono float32 audio at SR.

    factor = cur_dur / target_dur
    - factor > 1.0 = audio dlhšie ako slot → speed up (atempo > 1.0), capped at max_factor
    - factor < min_factor = audio kratšie ako slot → ZACHOVÁ prirodzené tempo
      (žiadny slow-down; ticho rozdelíme okolo chunku v main())
    - min_factor ≤ factor < 1.0 = mierne pomalšie, presne sadne do slotu (lepšie sync)
    """
    audio, sr = sf.read(str(in_wav))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    cur_dur = len(audio) / sr
    if target_dur <= 0.05:
        return np.zeros(int(0.05 * SR), dtype=np.float32)
    factor = cur_dur / target_dur
    if abs(factor - 1.0) < 0.02:
        if sr != SR:
            return _resample(audio, sr, SR)
        return audio
    # Pri factor < min_factor zachovaj prirodzené tempo (nikdy nepoužiť pomalý slow-down).
    # Pri min_factor ≤ factor < 1.0 aplikuj mierny slow-down → presný fit do slotu.
    if factor < min_factor:
        if sr != SR:
            audio = _resample(audio, sr, SR)
        return audio  # natural tempo, no slow-down
    if factor > max_factor:
        factor = max_factor

    # Use ffmpeg atempo (better quality than scipy)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_out:
        in_path = Path(tf_in.name)
        out_path = Path(tf_out.name)
    sf.write(str(in_path), audio, sr)
    chain = atempo_chain(factor)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path), "-af", chain,
         "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(out_path)],
        capture_output=True, timeout=30,
    )
    out_audio, _ = sf.read(str(out_path))
    if out_audio.ndim > 1:
        out_audio = out_audio.mean(axis=1)
    in_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return out_audio.astype(np.float32)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        in_path = Path(tf.name)
    sf.write(str(in_path), audio, src_sr)
    out_path = in_path.with_suffix(".out.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path), "-ar", str(dst_sr),
         "-ac", "1", "-c:a", "pcm_s16le", str(out_path)],
        capture_output=True,
    )
    out, _ = sf.read(str(out_path))
    if out.ndim > 1:
        out = out.mean(axis=1)
    in_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Dynamic per-segment timeline aligner")
    ap.add_argument("--segments", default=None, help="Segments JSON (default: temp/.../sk_segments.json)")
    ap.add_argument("--chunks-dir", default="/tmp/standalone_chunks",
                    help="Adresár s chunk_NNNN.wav (default: /tmp/standalone_chunks)")
    ap.add_argument("--video", default=None, help="Source video pre overall length (optional)")
    ap.add_argument("--out", "-o", default=None, help="Output WAV")
    ap.add_argument("--max-factor", type=float, default=1.50,
                    help="Max speed factor per segment (default 1.50; 1.40 safer)")
    ap.add_argument("--min-factor", type=float, default=0.85,
                    help="Min factor — pod ním sa NEAPLIKUJE slow-down, audio ostane "
                         "naturálne, slot sa vyplní tichom rozdelene okolo chunku (default 0.85). "
                         "Medzi min_factor a 1.0 sa aplikuje mierny slow-down pre presný fit.")
    ap.add_argument("--center-short", action=argparse.BooleanOptionalAction, default=True,
                    help="Pri natural-tempo chunkoch (factor < min_factor) umiestni "
                         "chunk do stredu slotu, nie na začiatok — lepší lipsync, menej "
                         "dlhých ticha za slovami (default ON).")
    ap.add_argument("--use-gaps", action=argparse.BooleanOptionalAction, default=True,
                    help="Smart placement — keď segment chunk pretieka cez slot, použije "
                         "gap (silence) k ďalšiemu segmentu ako buffer pred atempo. "
                         "Default ON — výrazne menej overlap pri tesných segmentoch.")
    args = ap.parse_args()

    # Auto-detect segments JSON
    if args.segments is None:
        # Find newest sk_segments.json in temp/
        candidates = list((REPO / "temp").glob("*/*sk_segments.json"))
        if not candidates:
            sys.exit("Žiadny segments JSON nenájdený v temp/. Použi --segments")
        args.segments = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected segments: {args.segments}")

    seg_path = Path(args.segments).expanduser().resolve()
    chunks_dir = Path(args.chunks_dir).expanduser().resolve()

    data = json.loads(seg_path.read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data

    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunks:
        sys.exit(f"Žiadne chunk_*.wav v {chunks_dir}. Spusti najprv test_omnivoice_batch.py --keep-chunks")
    if len(chunks) != len(segs):
        print(f"⚠ Chunks ({len(chunks)}) != segments ({len(segs)}). Poradie podľa indexu.")

    # Compute total timeline duration
    total_dur = max(float(s.get("end", 0)) for s in segs)
    if args.video:
        v_dur = get_duration(Path(args.video))
        total_dur = max(total_dur, v_dur)
        print(f"Video duration: {v_dur:.2f}s")
    print(f"Timeline duration: {total_dur:.2f}s")
    print(f"Segments: {len(segs)}, Chunks: {len(chunks)}")

    # Initialize timeline
    timeline = np.zeros(int(total_dur * SR) + SR, dtype=np.float32)

    # Place each chunk at its slot
    n_stretched = 0
    n_clamped = 0
    for i, seg in enumerate(segs):
        if i >= len(chunks):
            break
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        slot_dur = end - start
        if slot_dur < 0.05:
            continue

        chunk_dur = get_duration(chunks[i])

        # Smart placement: ak chunk_dur > slot_dur, skús použiť gap k next segment
        # ako extra buffer (effective_slot = slot + min(gap, 0.5s)). Tým sa znižuje
        # potreba atempo a audio nepretrieka cez nasledujúci slot.
        effective_slot = slot_dur
        if args.use_gaps and chunk_dur > slot_dur:
            # Find gap to next segment
            if i + 1 < len(segs):
                next_start = float(segs[i+1].get("start", end))
                gap = next_start - end
                if gap > 0.10:
                    # Use up to 80% of gap (leave 20% breathing room)
                    extra = min(gap * 0.8, chunk_dur - slot_dur)
                    effective_slot = slot_dur + extra
                    if extra > 0.05:
                        print(f"  seg {i+1}: gap-extend slot {slot_dur:.2f}s → {effective_slot:.2f}s "
                              f"(gap={gap:.2f}s)")

        factor = chunk_dur / effective_slot if effective_slot > 0 else 1.0

        if abs(factor - 1.0) > 0.02:
            n_stretched += 1
        if factor > args.max_factor:
            n_clamped += 1
            print(f"  seg {i+1}: factor {factor:.2f}× clamped to {args.max_factor:.2f}× "
                  f"(slot={effective_slot:.2f}s, chunk={chunk_dur:.2f}s)")

        stretched = stretch_chunk(chunks[i], effective_slot, args.max_factor, args.min_factor)
        stretched_dur = len(stretched) / SR

        # Center-short: ak chunk je výrazne kratší ako slot (factor < min_factor),
        # vystredíme ho — pol-tichom pred + pol-tichom po. Lepší lipsync, menej
        # "audio sa stratilo" pocitu pri dlhých slotoch.
        offset_sec = 0.0
        if args.center_short and stretched_dur < (slot_dur * 0.95):
            free_space = slot_dur - stretched_dur
            offset_sec = max(0.0, free_space * 0.5)

        place_start = start + offset_sec
        start_idx = int(place_start * SR)
        end_idx = start_idx + len(stretched)
        if end_idx > len(timeline):
            timeline = np.pad(timeline, (0, end_idx - len(timeline) + SR))
        timeline[start_idx:end_idx] += stretched

    # Trim to total_dur
    timeline = timeline[: int(total_dur * SR)]

    # Save
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        stem = seg_path.stem.replace("_sk_segments", "").replace("_segments", "")
        out_path = DEFAULT_OUT_DIR / f"{stem}_TIMELINE.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), timeline, SR)

    print(f"\n✓ Timeline audio: {out_path}")
    print(f"  Duration: {len(timeline)/SR:.2f}s")
    print(f"  Stretched segments: {n_stretched}/{len(segs)}")
    print(f"  Clamped (factor > {args.max_factor}): {n_clamped}")

    # MP3
    mp3 = out_path.with_suffix(".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", str(out_path),
                    "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)],
                   capture_output=True)
    print(f"  MP3:      {mp3} ({mp3.stat().st_size//1024//1024} MB)")


if __name__ == "__main__":
    main()
