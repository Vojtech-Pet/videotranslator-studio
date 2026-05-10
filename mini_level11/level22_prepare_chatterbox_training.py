#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Prosody quality scoring — uses librosa, optional (graceful fallback)
# ---------------------------------------------------------------------------

def score_prosody(audio_path: Path) -> dict[str, float] | None:
    """Zmeria prosodickú kvalitu WAV súboru.

    Vráti:
      voiced_ratio   — podiel znelých frames (0–1); <0.40 = príliš veľa ticha/šumu
      energy_score   — normalizovaná RMS energia (0–1); <0.12 = príliš tichý
      pitch_range_hz — rozsah fundamentálnej frekvencie (p90–p10); <20 Hz = monotónny
      prosody_tier   — "expressive_high" / "expressive_mid" / "neutral" / "flat_reject"

    Ak sa analýza nepodarí (librosa nie je dostupná, súbor poškodený), vráti None.
    """
    try:
        import numpy as np
        import librosa
    except ImportError:
        return None

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        if y.size < int(sr * 0.3):
            return None

        # Energy
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        if rms.size == 0:
            return None
        mean_rms = float(np.mean(rms))
        energy_score = float(np.clip((mean_rms - 0.015) / 0.095, 0.0, 1.0))

        # Voiced ratio + pitch via pyin
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=60.0, fmax=500.0, sr=sr,
                frame_length=2048, hop_length=256,
            )
            voiced_ratio = float(voiced_flag.mean()) if voiced_flag.size else 0.0
            voiced_f0 = f0[voiced_flag] if voiced_flag.any() else None
            if voiced_f0 is not None and len(voiced_f0) >= 10:
                pitch_range = float(np.percentile(voiced_f0, 90) - np.percentile(voiced_f0, 10))
            else:
                pitch_range = 0.0
        except Exception:
            voiced_ratio = 0.0
            pitch_range = 0.0

        # Tier
        if voiced_ratio < 0.40 or energy_score < 0.12:
            tier = "flat_reject"
        elif pitch_range >= 90.0 and energy_score >= 0.45:
            tier = "expressive_high"
        elif pitch_range >= 40.0:
            tier = "expressive_mid"
        else:
            tier = "neutral"

        return {
            "voiced_ratio": round(voiced_ratio, 3),
            "energy_score": round(energy_score, 3),
            "pitch_range_hz": round(pitch_range, 1),
            "prosody_tier": tier,
        }
    except Exception:
        return None


DEFAULT_TRAIN_ROOT = Path("/home/vojtech/Ai/chatterbox-finetuning")
DEFAULT_MODEL_DIR = Path(
    "/home/vojtech/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9"
)
DEFAULT_RESUME_CHECKPOINT = Path("/home/vojtech/VideoTranslator_studio/models/chatterbox/t3_sk_v2.2.safetensors")
ALLOWED_BUCKETS = ("core_clean", "technical_sql", "hard_examples", "expressive_high", "expressive_mid")
BAD_PATTERNS = (
    "ajznul",
    "koalesk",
    "esíkjúel",
    "splynutie splynutia",
    "dostávajú to",
    "dostavaju to",
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare Chatterbox SQL pilot training pack from Level 21 ingest manifests.")
    ap.add_argument(
        "--manifest-dirs",
        nargs="+",
        required=True,
        help="One or more Level 21 manifest directories or candidate_segments.json files.",
    )
    ap.add_argument("--out-dir", required=True, help="Output directory for the training pack")
    ap.add_argument("--speaker", default="sk_sql_01", help="Speaker id stored in preview manifests")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--min-duration", type=float, default=1.5, help="Minimum segment duration in seconds")
    ap.add_argument("--max-duration", type=float, default=12.0, help="Maximum segment duration in seconds")
    ap.add_argument("--min-chars", type=int, default=18, help="Minimum number of text characters")
    ap.add_argument("--clip-sample-rate", type=int, default=24000, help="Output sample rate for segment wavs")
    ap.add_argument("--reuse-clips", action="store_true", help="Reuse existing clipped wavs if present")
    ap.add_argument("--eval-ratio", type=float, default=0.1, help="Held-out eval ratio per bucket")
    ap.add_argument(
        "--min-eval-bucket-size",
        type=int,
        default=8,
        help="Minimum bucket size before creating held-out eval rows for that bucket",
    )
    ap.add_argument("--core-weight", type=int, default=5, help="Oversampling weight for core_clean")
    ap.add_argument("--sql-weight", type=int, default=3, help="Oversampling weight for technical_sql")
    ap.add_argument("--hard-weight", type=int, default=2, help="Oversampling weight for hard_examples")
    ap.add_argument("--expressive-high-weight", type=int, default=7, help="Oversampling weight for expressive_high (prosody rebucket)")
    ap.add_argument("--expressive-mid-weight", type=int, default=5, help="Oversampling weight for expressive_mid (prosody rebucket)")
    ap.add_argument("--train-root", default=str(DEFAULT_TRAIN_ROOT), help="Path to chatterbox-finetuning repo")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Chatterbox base model dir for from_local()")
    ap.add_argument("--prosody-filter", action="store_true", help="Filtruj tréningové dáta podľa prosodickej kvality (vyžaduje librosa)")
    ap.add_argument("--prosody-rebucket", action="store_true", help="Automaticky presuň vzorky do bucketu podľa prosody tieru (expressive_high/mid/neutral)")
    ap.add_argument(
        "--resume-checkpoint",
        default=str(DEFAULT_RESUME_CHECKPOINT),
        help="Checkpoint to resume from for the SQL pilot",
    )
    ap.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    ap.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    ap.add_argument("--learning-rate", type=float, default=5e-6, help="Pilot learning rate")
    ap.add_argument("--epochs", type=float, default=2.0, help="Pilot epoch count")
    ap.add_argument("--save-steps", type=int, default=100, help="Checkpoint/save frequency")
    ap.add_argument("--warmup-steps", type=int, default=25, help="Warmup steps")
    ap.add_argument("--val-split-ratio", type=float, default=0.05, help="Internal Trainer val split ratio")
    ap.add_argument("--turbo", action="store_true", help="Generuj launch script pre Chatterbox Turbo (CHATTERBOX_IS_TURBO=1)")
    return ap


def clean_text(text: str) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    out = re.sub(r"\s+([,.:;!?])", r"\1", out)
    out = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", out)
    return out.strip()


def resolve_candidate_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / "candidate_segments.json",
        path / "manifests" / "candidate_segments.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No candidate_segments.json found under: {path}")


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for item in paths:
        manifest_path = resolve_candidate_path(item)
        try:
            rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Cannot read manifest {manifest_path}: {exc}") from exc
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON list in {manifest_path}")
        loaded.extend(rows)
    return loaded


def filter_row(row: dict[str, Any], *, min_duration: float, max_duration: float, min_chars: int) -> tuple[bool, str]:
    if not isinstance(row, dict):
        return False, "invalid_row"
    bucket = str(row.get("bucket") or "")
    if bucket not in ALLOWED_BUCKETS:
        return False, "bad_bucket"

    text = clean_text(str(row.get("text") or ""))
    if len(text) < min_chars:
        return False, "short_text"
    lowered = text.lower()
    if any(pattern in lowered for pattern in BAD_PATTERNS):
        return False, "bad_text"

    try:
        start = float(row.get("start", 0.0) or 0.0)
        end = float(row.get("end", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False, "bad_timing"
    duration = float(row.get("duration", max(0.0, end - start)) or 0.0)
    if duration < min_duration:
        return False, "too_short"
    if duration > max_duration:
        return False, "too_long"

    audio = Path(str(row.get("audio") or "")).expanduser()
    if not audio.exists():
        return False, "missing_audio"
    return True, "ok"


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("source_id"),
            round(float(row.get("start", 0.0) or 0.0), 3),
            round(float(row.get("end", 0.0) or 0.0), 3),
            clean_text(row.get("text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def make_clip_id(row: dict[str, Any]) -> str:
    source_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(row.get("source_id") or "src")).strip("_") or "src"
    segment_index = int(row.get("segment_index", 0) or 0)
    start_ms = int(round(float(row.get("start", 0.0) or 0.0) * 1000))
    end_ms = int(round(float(row.get("end", 0.0) or 0.0) * 1000))
    return f"{source_id}_seg{segment_index:05d}_{start_ms:09d}_{end_ms:09d}"


def clip_audio(src: Path, *, start: float, duration: float, dst: Path, sample_rate: int, reuse: bool) -> None:
    if reuse and dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)


def stratified_eval_split(
    rows: list[dict[str, Any]],
    *,
    eval_ratio: float,
    min_eval_bucket_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["bucket"])].append(row)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for bucket in ALLOWED_BUCKETS:
        bucket_rows = list(by_bucket.get(bucket, []))
        rng.shuffle(bucket_rows)
        if len(bucket_rows) >= min_eval_bucket_size:
            eval_count = max(1, int(round(len(bucket_rows) * eval_ratio)))
            eval_count = min(eval_count, max(1, len(bucket_rows) - 1))
        else:
            eval_count = 0
        eval_rows.extend(bucket_rows[:eval_count])
        train_rows.extend(bucket_rows[eval_count:])
    return train_rows, eval_rows


def oversample_rows(rows: list[dict[str, Any]], *, weights: dict[str, int]) -> list[dict[str, Any]]:
    weighted: list[dict[str, Any]] = []
    for row in rows:
        copies = max(1, int(weights.get(str(row["bucket"]), 1)))
        for index in range(copies):
            clone = dict(row)
            clone["id"] = f"{row['id']}__w{index + 1:02d}"
            weighted.append(clone)
    return weighted


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, speaker: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "audio": row["audio_filepath"],
                "text": row["text"],
                "speaker": speaker,
                "bucket": row["bucket"],
                "id": row["id"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def render_launch_script(path: Path, *, train_root: Path, model_dir: Path, resume_checkpoint: Path, metadata_path: Path, clips_dir: Path, preprocessed_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TRAIN_ROOT="{train_root}"
PYTHON="${{PYTHON:-/home/vojtech/miniforge3/envs/chatterbox_env/bin/python3}}"

export CHATTERBOX_MODEL_DIR="${{CHATTERBOX_MODEL_DIR:-{model_dir}}}"
export CHATTERBOX_METADATA_PATH="${{CHATTERBOX_METADATA_PATH:-{metadata_path}}}"
export CHATTERBOX_WAV_DIR="${{CHATTERBOX_WAV_DIR:-{clips_dir}}}"
export CHATTERBOX_PREPROCESSED_DIR="${{CHATTERBOX_PREPROCESSED_DIR:-{preprocessed_dir}}}"
export CHATTERBOX_OUTPUT_DIR="${{CHATTERBOX_OUTPUT_DIR:-{output_dir}}}"
export CHATTERBOX_RESUME_CHECKPOINT="${{CHATTERBOX_RESUME_CHECKPOINT:-{resume_checkpoint}}}"

export CHATTERBOX_USE_MTL="${{CHATTERBOX_USE_MTL:-1}}"
export CHATTERBOX_LANGUAGE_ID="${{CHATTERBOX_LANGUAGE_ID:-sk}}"
export CHATTERBOX_JSON_FORMAT="${{CHATTERBOX_JSON_FORMAT:-1}}"
export CHATTERBOX_LJSPEECH="${{CHATTERBOX_LJSPEECH:-0}}"
export CHATTERBOX_IS_TURBO="${{CHATTERBOX_IS_TURBO:-{1 if args.turbo else 0}}}"
export CHATTERBOX_PREPROCESS="${{CHATTERBOX_PREPROCESS:-1}}"
export CHATTERBOX_NEW_VOCAB_SIZE="${{CHATTERBOX_NEW_VOCAB_SIZE:-2454}}"

export CHATTERBOX_BATCH_SIZE="${{CHATTERBOX_BATCH_SIZE:-{args.batch_size}}}"
export CHATTERBOX_GRAD_ACCUM="${{CHATTERBOX_GRAD_ACCUM:-{args.grad_accum}}}"
export CHATTERBOX_LEARNING_RATE="${{CHATTERBOX_LEARNING_RATE:-{args.learning_rate}}}"
export CHATTERBOX_NUM_EPOCHS="${{CHATTERBOX_NUM_EPOCHS:-{args.epochs}}}"
export CHATTERBOX_SAVE_STEPS="${{CHATTERBOX_SAVE_STEPS:-{args.save_steps}}}"
export CHATTERBOX_WARMUP_STEPS="${{CHATTERBOX_WARMUP_STEPS:-{args.warmup_steps}}}"
export CHATTERBOX_VAL_SPLIT_RATIO="${{CHATTERBOX_VAL_SPLIT_RATIO:-{args.val_split_ratio}}}"
export CHATTERBOX_SAVE_TOTAL_LIMIT="${{CHATTERBOX_SAVE_TOTAL_LIMIT:-3}}"

cd "$TRAIN_ROOT"
exec "$PYTHON" -u train.py
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    manifest_rows = load_rows(args.manifest_dirs)
    filtered: list[dict[str, Any]] = []
    rejected = Counter()
    for row in manifest_rows:
        keep, reason = filter_row(
            row,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_chars=args.min_chars,
        )
        if keep:
            filtered.append(row)
        else:
            rejected[reason] += 1
    filtered = dedupe_rows(filtered)
    if not filtered:
        raise RuntimeError("No usable Level 21 candidates after filtering.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    clips_dir = out_dir / "clips"
    manifests_dir = out_dir / "manifests"
    preprocessed_dir = out_dir / "preprocessed"
    output_dir = out_dir / "output_chatterbox_sql_pilot"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    prepared_rows: list[dict[str, Any]] = []
    source_counts = Counter()
    bucket_counts = Counter()
    for row in filtered:
        text = clean_text(row["text"])
        start = round(float(row["start"]), 3)
        end = round(float(row["end"]), 3)
        duration = round(float(row.get("duration", end - start) or (end - start)), 3)
        source_audio = Path(str(row["audio"])).expanduser().resolve()
        clip_id = make_clip_id(row)
        clip_path = clips_dir / f"{clip_id}.wav"
        clip_audio(
            source_audio,
            start=start,
            duration=duration,
            dst=clip_path,
            sample_rate=args.clip_sample_rate,
            reuse=args.reuse_clips,
        )
        final_bucket = str(row["bucket"])

        # Prosody analýza clipped WAV (opt-in)
        prosody_meta: dict[str, Any] = {}
        if (args.prosody_filter or args.prosody_rebucket) and clip_path.exists():
            pscore = score_prosody(clip_path)
            if pscore:
                prosody_meta = pscore
                tier = pscore["prosody_tier"]

                if args.prosody_filter and tier == "flat_reject":
                    rejected["prosody_flat_reject"] += 1
                    print(
                        f"[PROSODY] Rejected {clip_id}: voiced={pscore['voiced_ratio']:.2f}"
                        f" energy={pscore['energy_score']:.2f} pitch={pscore['pitch_range_hz']:.0f}Hz",
                        flush=True,
                    )
                    continue

                if args.prosody_rebucket and tier in ("expressive_high", "expressive_mid"):
                    # Nezasahuj do technical_sql/hard_examples — len core_clean rebucket
                    if final_bucket == "core_clean":
                        final_bucket = tier
                        print(
                            f"[PROSODY] Rebucket {clip_id}: core_clean → {tier}"
                            f" (pitch={pscore['pitch_range_hz']:.0f}Hz"
                            f" energy={pscore['energy_score']:.2f})",
                            flush=True,
                        )

        prepared = {
            "id": clip_id,
            "text": text,
            "audio_filepath": str(clip_path),
            "bucket": final_bucket,
            "source_id": str(row["source_id"]),
            "source_title": str(row.get("source_title") or ""),
            "source_channel": str(row.get("source_channel") or ""),
            "source_url": str(row.get("source_url") or ""),
            "segment_index": int(row.get("segment_index", 0) or 0),
            "start": start,
            "end": end,
            "duration": duration,
            "source_audio": str(source_audio),
            **prosody_meta,
        }
        prepared_rows.append(prepared)
        source_counts[prepared["source_id"]] += 1
        bucket_counts[final_bucket] += 1

    train_rows, eval_rows = stratified_eval_split(
        prepared_rows,
        eval_ratio=args.eval_ratio,
        min_eval_bucket_size=args.min_eval_bucket_size,
        rng=rng,
    )
    weights = {
        "core_clean": args.core_weight,
        "technical_sql": args.sql_weight,
        "hard_examples": args.hard_weight,
        "expressive_high": args.expressive_high_weight,
        "expressive_mid": args.expressive_mid_weight,
    }
    weighted_rows = oversample_rows(train_rows, weights=weights)
    rng.shuffle(weighted_rows)

    metadata_base_path = manifests_dir / "metadata_base_train.json"
    metadata_weighted_path = manifests_dir / "metadata_weighted_pilot.json"
    eval_path = manifests_dir / "eval_segments.json"
    train_preview_path = manifests_dir / "train_preview.jsonl"
    eval_preview_path = manifests_dir / "eval_preview.jsonl"
    stats_path = manifests_dir / "stats.json"
    launch_path = out_dir / "launch_chatterbox_sql_pilot.sh"

    write_json(metadata_base_path, train_rows)
    write_json(metadata_weighted_path, weighted_rows)
    write_json(eval_path, eval_rows)
    write_jsonl(train_preview_path, train_rows, speaker=args.speaker)
    write_jsonl(eval_preview_path, eval_rows, speaker=args.speaker)
    render_launch_script(
        launch_path,
        train_root=Path(args.train_root).expanduser().resolve(),
        model_dir=Path(args.model_dir).expanduser().resolve(),
        resume_checkpoint=Path(args.resume_checkpoint).expanduser().resolve(),
        metadata_path=metadata_weighted_path,
        clips_dir=clips_dir,
        preprocessed_dir=preprocessed_dir,
        output_dir=output_dir,
        args=args,
    )

    stats = {
        "input_rows": len(manifest_rows),
        "filtered_rows": len(filtered),
        "prepared_rows": len(prepared_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "weighted_rows": len(weighted_rows),
        "speaker": args.speaker,
        "weights": weights,
        "bucket_counts": dict(bucket_counts),
        "source_counts": dict(source_counts),
        "rejected": dict(rejected),
        "paths": {
            "metadata_base_train": str(metadata_base_path),
            "metadata_weighted_pilot": str(metadata_weighted_path),
            "eval_segments": str(eval_path),
            "train_preview_jsonl": str(train_preview_path),
            "eval_preview_jsonl": str(eval_preview_path),
            "clips_dir": str(clips_dir),
            "preprocessed_dir": str(preprocessed_dir),
            "output_dir": str(output_dir),
            "launch_script": str(launch_path),
        },
        "training": {
            "train_root": str(Path(args.train_root).expanduser().resolve()),
            "model_dir": str(Path(args.model_dir).expanduser().resolve()),
            "resume_checkpoint": str(Path(args.resume_checkpoint).expanduser().resolve()),
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "save_steps": args.save_steps,
            "warmup_steps": args.warmup_steps,
            "val_split_ratio": args.val_split_ratio,
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("HOTOVO:")
    print(f" - {metadata_base_path}")
    print(f" - {metadata_weighted_path}")
    print(f" - {eval_path}")
    print(f" - {launch_path}")
    print(f" - {stats_path}")


if __name__ == "__main__":
    main()
