#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

random.seed(42)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare Level 17 S2 training pack from Level 16 manifest.")
    ap.add_argument("--input", required=True, help="Input Level 16 tts_manifest.json")
    ap.add_argument("--out-dir", default="level17_train_pack", help="Output directory")
    ap.add_argument("--train-split", type=float, default=0.95, help="Train split ratio")
    ap.add_argument("--min-duration", type=float, default=1.5, help="Minimum audio duration")
    ap.add_argument("--max-duration", type=float, default=14.0, help="Maximum audio duration")
    ap.add_argument("--speaker", default="sk_tech_01", help="Speaker id for manifest rows")
    ap.add_argument("--allow-missing-audio", action="store_true", help="Do not reject rows with missing audio files.")
    return ap


BAD_PATTERNS = (
    "ajznul",
    "koalesk",
    "esíkjúel",
    "splynutie splynutia",
    "dostávajú to",
)


def classify_item(
    item: dict[str, Any],
    *,
    min_duration: float,
    max_duration: float,
    allow_missing_audio: bool,
) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "invalid_row"
    if not item.get("ok", False):
        return False, "not_ok"

    wav = str(item.get("wav") or "").strip()
    if not wav:
        return False, "missing_wav"

    try:
        duration = float(item.get("wav_duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False, "invalid_duration"
    if duration <= 0:
        return False, "missing_duration"
    if duration < min_duration:
        return False, "too_short"
    if duration > max_duration:
        return False, "too_long"

    text = str(item.get("text") or "").strip()
    if len(text) < 5:
        return False, "short_text"

    lowered = text.lower()
    if any(pattern in lowered for pattern in BAD_PATTERNS):
        return False, "bad_text"

    if not allow_missing_audio and not Path(wav).expanduser().exists():
        return False, "missing_audio_file"

    return True, "ok"


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, speaker: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "audio": row["wav"],
                "text": row["text"],
                "speaker": speaker,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_training_config(path: Path, *, train_file: Path, val_file: Path) -> None:
    yaml_text = (
        "dataset:\n"
        f"  train_manifest: {train_file.name}\n"
        f"  val_manifest: {val_file.name}\n"
        "\n"
        "audio:\n"
        "  sample_rate: 44100\n"
        "\n"
        "training:\n"
        "  batch_size: 4\n"
        "  grad_accum: 4\n"
        "  epochs: 3\n"
        "\n"
        "optimizer:\n"
        "  lr: 1e-5\n"
        "\n"
        "logging:\n"
        "  eval_steps: 500\n"
        "  save_steps: 1000\n"
        "\n"
        "model:\n"
        "  freeze_text_encoder: false\n"
        "  freeze_audio_encoder: true\n"
    )
    path.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("Expected Level 16 manifest as a JSON list.")

    rejected_counts: dict[str, int] = {}
    clean: list[dict[str, Any]] = []
    looks_like_level16 = 0
    for item in items:
        if isinstance(item, dict) and {"wav", "wav_duration", "ok"} & set(item.keys()):
            looks_like_level16 += 1
        keep, reason = classify_item(
            item,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            allow_missing_audio=args.allow_missing_audio,
        )
        if keep:
            clean.append(item)
        else:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    if looks_like_level16 == 0:
        raise ValueError(
            "Input does not look like a Level 16 manifest. Use level16_out/tts_manifest.json, not tts_ready.json."
        )
    if not clean:
        raise ValueError(
            "No usable rows after filtering. Check stats.json or pass a real Level 16 manifest with wav/wav_duration/ok fields."
        )

    random.shuffle(clean)
    split_idx = int(len(clean) * args.train_split)
    if len(clean) > 1:
        split_idx = min(max(split_idx, 1), len(clean) - 1)
    else:
        split_idx = len(clean)
    train = clean[:split_idx]
    val = clean[split_idx:]

    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"
    stats_file = out_dir / "stats.json"
    config_file = out_dir / "config_s2_sk.yaml"

    write_jsonl(train_file, train, speaker=args.speaker)
    write_jsonl(val_file, val, speaker=args.speaker)
    write_training_config(config_file, train_file=train_file, val_file=val_file)

    durations = [float(item.get("wav_duration", 0.0) or 0.0) for item in clean]
    stats = {
        "input_manifest": str(input_path),
        "total_input": len(items),
        "clean_used": len(clean),
        "train": len(train),
        "val": len(val),
        "speaker": args.speaker,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "allow_missing_audio": bool(args.allow_missing_audio),
        "avg_duration": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "min_kept_duration": round(min(durations), 3) if durations else 0.0,
        "max_kept_duration": round(max(durations), 3) if durations else 0.0,
        "rejected": rejected_counts,
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("HOTOVO:")
    print(f" - {train_file}")
    print(f" - {val_file}")
    print(f" - {stats_file}")
    print(f" - {config_file}")


if __name__ == "__main__":
    main()
