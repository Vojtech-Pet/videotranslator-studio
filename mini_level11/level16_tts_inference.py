#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for entry in (ROOT, SCRIPTS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import soundfile as sf  # noqa: E402

from audio_pipeline import trim_edge_silence_ffmpeg  # noqa: E402
from tts import s2_pro_tts  # noqa: E402

TARGET_SR = 44100


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Level 16 S2-Pro batch TTS inference.")
    ap.add_argument("--input", required=True, help="Input tts_ready.json from Level 15.")
    ap.add_argument("--out-dir", default="level16_out", help="Output directory.")
    ap.add_argument("--checkpoint", required=True, help="Fish S2-Pro checkpoint path.")
    ap.add_argument("--reference-wav", default="", help="Reference WAV for cloning.")
    ap.add_argument(
        "--reference-text",
        default=(
            "This is a calm, clear and professional technical narrator voice. "
            "The speech is natural, confident, medium-paced and easy to understand."
        ),
        help="Reference transcript text.",
    )
    ap.add_argument("--server-python", default="", help="Optional python for Fish API server.")
    ap.add_argument("--server-port", type=int, default=8091, help="Fish API server port.")
    ap.add_argument("--timeout", type=float, default=600.0, help="Single TTS request timeout in seconds.")
    ap.add_argument("--no-trim", action="store_true", help="Disable silence trim.")
    ap.add_argument("--no-normalize", action="store_true", help="Disable loudness normalize.")
    ap.add_argument("--target-lufs", type=float, default=-16.0, help="Target loudness for ffmpeg loudnorm.")
    return ap


def _load_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict) and "segments" in data:
        return list(data["segments"])
    raise ValueError("Unsupported input format.")


def wav_duration_seconds(path: Path) -> float:
    data, sr = sf.read(path)
    return len(data) / sr


def normalize_loudness_ffmpeg(in_wav: Path, out_wav: Path, target_lufs: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-af",
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar",
        str(TARGET_SR),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)


def process_one(
    idx: int,
    item: dict[str, Any],
    *,
    wav_dir: Path,
    checkpoint_path: str,
    reference_wav: str | None,
    reference_text: str | None,
    server_python: str,
    server_port: int,
    timeout_s: float,
    trim_enabled: bool,
    normalize_enabled: bool,
    target_lufs: float,
) -> dict[str, Any]:
    text = str(item.get("text") or "").strip()
    start = float(item.get("start", 0.0) or 0.0)
    end = float(item.get("end", 0.0) or 0.0)
    slot = max(0.01, end - start)

    raw_wav = wav_dir / f"seg_{idx:04d}_raw.wav"
    s2_pro_tts(
        text=text,
        output_path=raw_wav,
        checkpoint_path=checkpoint_path,
        reference_audio_path=reference_wav or None,
        reference_text=reference_text or None,
        server_python=server_python,
        server_port=server_port,
        timeout_s=timeout_s,
    )

    final_wav = raw_wav
    if trim_enabled:
        trimmed = wav_dir / f"seg_{idx:04d}_trim.wav"
        if trim_edge_silence_ffmpeg(final_wav, trimmed):
            final_wav = trimmed

    if normalize_enabled:
        normalized = wav_dir / f"seg_{idx:04d}.wav"
        normalize_loudness_ffmpeg(final_wav, normalized, target_lufs)
        final_wav = normalized

    duration = wav_duration_seconds(final_wav)
    delta = round(duration - slot, 3)

    return {
        "idx": idx,
        "start": start,
        "end": end,
        "slot": round(slot, 3),
        "text": text,
        "wav": str(final_wav),
        "wav_duration": round(duration, 3),
        "delta": delta,
        "ok": abs(delta) <= 0.35,
    }


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_dir = out_dir / "wav_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "tts_manifest.json"
    failed_path = out_dir / "tts_failed.json"

    items = _load_items(input_path)
    manifest: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] TTS")
        try:
            row = process_one(
                idx,
                item,
                wav_dir=wav_dir,
                checkpoint_path=args.checkpoint,
                reference_wav=args.reference_wav,
                reference_text=args.reference_text,
                server_python=args.server_python,
                server_port=args.server_port,
                timeout_s=args.timeout,
                trim_enabled=not args.no_trim,
                normalize_enabled=not args.no_normalize,
                target_lufs=args.target_lufs,
            )
        except Exception as exc:
            row = {
                "idx": idx,
                "start": item.get("start"),
                "end": item.get("end"),
                "text": item.get("text"),
                "error": str(exc),
                "ok": False,
            }
        manifest.append(row)
        if not row.get("ok", False):
            failed.append(row)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    failed_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")

    print("HOTOVO:")
    print(f" - {manifest_path}")
    print(f" - {failed_path}")


if __name__ == "__main__":
    main()
