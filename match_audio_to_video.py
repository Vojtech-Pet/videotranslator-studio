#!/usr/bin/env python3
"""
Time-stretch audio aby sa zmestilo presne do dĺžky videa.

Použitie:
  ./match_audio_to_video.py video.mp4 audio.wav [--out matched.wav]

Defaults:
  --out: <audio_stem>_matched.wav v rovnakom adresári

Time-stretch cez ffmpeg atempo (best dostupná metóda pre reč).
"""
import argparse
import subprocess
import sys
from pathlib import Path


def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def atempo_chain(factor: float) -> str:
    """ffmpeg atempo cap is 0.5–2.0 per filter. Chain multiple if outside."""
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor:.6f}"
    parts = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Time-stretch audio to match video duration")
    ap.add_argument("video", help="Source video MP4")
    ap.add_argument("audio", help="Audio WAV/MP3 to stretch")
    ap.add_argument("--out", "-o", default=None, help="Output WAV (default: <audio>_matched.wav)")
    ap.add_argument("--max-factor", type=float, default=1.40,
                    help="Max speed factor (safety; default 1.40). Above this, sounds unnatural.")
    ap.add_argument("--dry-run", action="store_true", help="Iba spočíta factor, nestreuje")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    if not video_path.exists():
        sys.exit(f"Video neexistuje: {video_path}")
    if not audio_path.exists():
        sys.exit(f"Audio neexistuje: {audio_path}")

    video_dur = get_duration(str(video_path))
    audio_dur = get_duration(str(audio_path))
    factor = audio_dur / video_dur
    print(f"Video: {video_dur:.2f}s")
    print(f"Audio: {audio_dur:.2f}s")
    print(f"Factor: {factor:.4f}× ({'speed up' if factor > 1 else 'slow down'})")

    if abs(factor - 1.0) < 0.01:
        print("Audio už sedí s videom (delta < 1%) — žiadna úprava netreba.")
        return

    if factor > args.max_factor:
        print(f"⚠ Factor {factor:.2f}× presahuje max {args.max_factor}. Audio bude clamped na max.")
        factor = args.max_factor

    if args.dry_run:
        print(f"\n[DRY RUN] Použil by som: {atempo_chain(factor)}")
        return

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = audio_path.with_name(audio_path.stem + "_matched.wav")

    chain = atempo_chain(factor)
    print(f"\nStretching: {chain}")
    cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-af", chain,
           "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0:
        sys.exit(f"FFmpeg failed: {r.stderr.decode('utf-8', errors='replace')[-500:]}")

    new_dur = get_duration(str(out_path))
    print(f"\n✓ Saved: {out_path}")
    print(f"  Duration: {new_dur:.2f}s (target {video_dur:.2f}s, delta {new_dur - video_dur:+.2f}s)")


if __name__ == "__main__":
    main()
