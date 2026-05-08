#!/usr/bin/env python3
"""
PRO master script — replikuje style YouTube SK AI dubbing:
  voice (TIMELINE.wav, dominant) + bg music (z pôvodného EN demucs) → heavy compress
  → EBU R128 -20 LUFS → limit -2 dBTP → stereo MP4

Použitie:
  ./master_pro.py video.mp4 timeline.wav [--out final.mp4]
  ./master_pro.py video.mp4 timeline.wav --bg-volume -15 --lufs -20

Pipeline:
  1. Demucs separate vocals/no_vocals z pôvodného EN videa (cache)
  2. Mix: timeline.wav (voice center) + no_vocals.wav (bg, -15 dB)
  3. Compress: heavy ratio (3:1, threshold -18dB) — konsistentnejšia úroveň
  4. Loudnorm 2-pass: I=-20 LUFS, TP=-2 dBTP, LRA=5
  5. Mux do MP4 stereo
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
CACHE_DIR = REPO / "work" / "demucs_cache"


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def demucs_separate(video_path: Path, cache_dir: Path) -> Path:
    """Extract instrumental track from video. Cached."""
    stem = video_path.stem
    cached = cache_dir / stem / "no_vocals.wav"
    if cached.exists() and cached.stat().st_size > 1000:
        print(f"[BG] Cached instrumental: {cached}")
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Extract audio first
    audio_in = cache_dir / f"{stem}_full.wav"
    if not audio_in.exists():
        print(f"[BG] Extract audio z {video_path.name}...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "2", "-ar", "44100",
             "-c:a", "pcm_s16le", str(audio_in)],
            capture_output=True, check=True,
        )

    print(f"[BG] Demucs separate (htdemucs --two-stems vocals)...")
    out_dir = cache_dir / "tmp"
    out_dir.mkdir(exist_ok=True)
    # Použiť musetalk_env Python kde je demucs nainštalovaný
    py_exec = "/mnt/tts_data/miniforge3/envs/musetalk_env/bin/python"
    subprocess.run(
        [py_exec, "-m", "demucs.separate",
         "-n", "htdemucs", "--two-stems", "vocals",
         str(audio_in), "-o", str(out_dir)],
        check=True,
    )
    src = out_dir / "htdemucs" / audio_in.stem / "no_vocals.wav"
    if not src.exists():
        sys.exit(f"Demucs neuložil: {src}")
    target = cache_dir / stem / "no_vocals.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    src.rename(target)
    # Cleanup tmp
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    audio_in.unlink(missing_ok=True)
    return target


def main():
    ap = argparse.ArgumentParser(description="PRO master — replikuje YouTube SK AI dubbing style")
    ap.add_argument("video", help="Pôvodný EN video MP4 (zdroj background music)")
    ap.add_argument("timeline", help="Timeline-aligned voice WAV (z match_audio_dynamic.py)")
    ap.add_argument("--out", "-o", default=None, help="Output MP4")
    ap.add_argument("--bg-volume", type=float, default=-15.0,
                    help="Background music volume offset v dB (default -15)")
    ap.add_argument("--lufs", type=float, default=-20.0,
                    help="Target integrated loudness (default -20 LUFS — YouTube)")
    ap.add_argument("--no-bg", action="store_true", help="Bez background music (iba voice mastered)")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR), help="Demucs cache adresár")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    timeline_path = Path(args.timeline).expanduser().resolve()
    if not video_path.exists() or not timeline_path.exists():
        sys.exit(f"Vstup neexistuje. video={video_path.exists()}, timeline={timeline_path.exists()}")

    print(f"==> PRO master")
    print(f"   video:    {video_path.name}")
    print(f"   timeline: {timeline_path.name}")
    print(f"   bg vol:   {args.bg_volume} dB")
    print(f"   target:   {args.lufs} LUFS")

    # 1) Mix voice + bg
    if args.no_bg:
        print(f"\n[MIX] Skip background — iba voice")
        mixed = timeline_path
    else:
        bg_path = demucs_separate(video_path, Path(args.cache_dir))
        print(f"\n[MIX] Voice center + bg @ {args.bg_volume}dB...")
        mixed = timeline_path.with_name(timeline_path.stem + "_mixed.wav")
        # voice (mono) → stereo center, bg (stereo) at offset volume, mix duration=longest
        filter_complex = (
            f"[0:a]volume=0dB,pan=stereo|c0=c0|c1=c0[v];"
            f"[1:a]volume={args.bg_volume}dB[b];"
            f"[v][b]amix=inputs=2:duration=longest:dropout_transition=0:weights=1.0 1.0,"
            f"alimiter=limit=0.95:level=disabled"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(timeline_path), "-i", str(bg_path),
             "-filter_complex", filter_complex,
             "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(mixed)],
            check=True, capture_output=True,
        )

    # 2) PRO EQ + Compress + loudnorm 2-pass — matches YouTube SK reference profile
    # EQ shape: +1.5dB bass warmth, tame boxy 400Hz, soften 4kHz twang, lowpass 9.5kHz
    # → output má warm, polished, broadcast-ready charakter
    print(f"\n[MASTER] PRO EQ + Compress + EBU R128 loudnorm I={args.lufs} LUFS TP=-2 LRA=4.5...")
    mastered = timeline_path.with_name(timeline_path.stem + "_mastered.wav")
    af_master = (
        # PRO EQ — open clarity + extended bass + air profile
        "equalizer=f=70:t=q:w=0.7:g=+2.0,"      # sub-bass / chest body (NOVÉ)
        "equalizer=f=120:t=q:w=0.9:g=+2.0,"     # warmth (zvýšené z +1.0)
        "equalizer=f=400:t=q:w=1.0:g=-1.0,"     # mild box tame
        "equalizer=f=1500:t=q:w=1.5:g=-1.5,"    # subtle de-nasal
        "equalizer=f=3500:t=q:w=1.0:g=+2.0,"    # presence — voice clarity
        "equalizer=f=8000:t=q:w=0.8:g=+3.5,"    # air boost (zvýšené z +2.5)
        "equalizer=f=12000:t=q:w=0.8:g=+3.0,"   # ultra-air / sparkle (zvýšené z +2.0)
        "equalizer=f=15000:t=q:w=0.7:g=+2.5,"   # top-end shimmer (NOVÉ)
        "lowpass=f=16000,"                       # very open highs (bolo 14.5k)
        # Compander pre even level
        "compand=attacks=0.05:decays=0.1:points=-90/-90|-30/-22|-15/-14|-5/-7|0/-4,"
        # 2-pass EBU R128 loudnorm
        f"loudnorm=I={args.lufs}:TP=-2:LRA=4.5:print_format=summary"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mixed), "-af", af_master,
         "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(mastered)],
        check=True, capture_output=True,
    )

    # 3) Mux do MP4
    out_path = Path(args.out) if args.out else video_path.parent.parent / "sk" / f"{video_path.stem}_PRO.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[MUX] {out_path}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-i", str(mastered),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-map", "0:v:0", "-map", "1:a:0", "-shortest",
         str(out_path)],
        check=True, capture_output=True,
    )

    # Cleanup intermediate
    if mixed != timeline_path:
        mixed.unlink(missing_ok=True)
    mastered.unlink(missing_ok=True)

    print(f"\n✓ PRO mastered MP4: {out_path}")
    print(f"  Size: {out_path.stat().st_size//1024//1024} MB")
    print(f"  Dur:  {get_duration(out_path):.2f}s")


if __name__ == "__main__":
    main()
