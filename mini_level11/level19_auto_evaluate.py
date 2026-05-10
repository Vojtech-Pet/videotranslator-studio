#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for entry in (ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import soundfile as sf  # noqa: E402

from src.level5_manager import asr_backcheck_score  # noqa: E402
from src.terminology_memory import TerminologyMemory  # noqa: E402
from src.utils_text import normalize_text  # noqa: E402

try:
    from stt import transcribe_whisper  # noqa: E402
except Exception:
    transcribe_whisper = None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Level 19 auto evaluation for TTS manifests.")
    ap.add_argument("--input", required=True, help="Input Level 16 tts_manifest.json")
    ap.add_argument("--output", default="", help="Output segment report JSON. Defaults beside input.")
    ap.add_argument("--summary", default="", help="Output summary JSON. Defaults beside input.")
    ap.add_argument(
        "--asr-backend",
        choices=["disabled", "faster_whisper"],
        default="disabled",
        help="Optional ASR backend for back-check.",
    )
    ap.add_argument("--whisper-model", default="large-v3-turbo", help="faster-whisper model name.")
    ap.add_argument("--download-root", default="", help="Optional faster-whisper download root.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="ASR device.")
    ap.add_argument("--report-threshold", type=float, default=0.82, help="Minimum final score for pass.")
    return ap


def _load_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected Level 16 manifest as a JSON list.")
    return data


def _wav_duration_seconds(path: Path) -> float:
    data, sr = sf.read(str(path))
    return len(data) / float(sr)


def _timing_score(delta: float | None) -> float:
    if delta is None:
        return 0.5
    absolute = abs(delta)
    if absolute <= 0.20:
        return 1.0
    if absolute <= 0.35:
        return 0.8
    if absolute <= 0.60:
        return 0.5
    return 0.2


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    prev = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        curr = [i]
        for j, token_b in enumerate(seq_b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _tokenize_words(text: str) -> list[str]:
    clean = normalize_text(text).lower()
    return [token for token in clean.replace(".", " ").replace(",", " ").replace(":", " ").replace(";", " ").split() if token]


def _tokenize_chars(text: str) -> list[str]:
    return list(normalize_text(text).lower().replace(" ", ""))


def _wer(expected_text: str, actual_text: str) -> float:
    expected = _tokenize_words(expected_text)
    actual = _tokenize_words(actual_text)
    if not expected:
        return 0.0 if not actual else 1.0
    return round(_levenshtein(expected, actual) / len(expected), 4)


def _cer(expected_text: str, actual_text: str) -> float:
    expected = _tokenize_chars(expected_text)
    actual = _tokenize_chars(actual_text)
    if not expected:
        return 0.0 if not actual else 1.0
    return round(_levenshtein(expected, actual) / len(expected), 4)


def _asr_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _transcribe_audio(
    wav_path: Path,
    *,
    backend: str,
    whisper_model: str,
    download_root: str,
    device: str,
) -> str:
    if backend == "disabled":
        return ""
    if backend == "faster_whisper":
        if transcribe_whisper is None:
            raise RuntimeError("faster-whisper backend unavailable: failed to import transcribe_whisper.")
        result = transcribe_whisper(
            str(wav_path),
            whisper_model,
            download_root or str((ROOT / "models" / "faster_whisper").resolve()),
            _asr_device(device),
        )
        return normalize_text(str(result.get("text") or ""))
    raise ValueError(f"Unsupported ASR backend: {backend}")


def _term_score(expected_text: str, actual_text: str, terminology: TerminologyMemory) -> float:
    return terminology.term_preservation_score(expected_text, actual_text)


def _evaluate_row(
    row: dict[str, Any],
    *,
    terminology: TerminologyMemory,
    asr_backend: str,
    whisper_model: str,
    download_root: str,
    device: str,
    report_threshold: float,
) -> dict[str, Any]:
    text = normalize_text(str(row.get("text") or ""))
    wav_path = Path(str(row.get("wav") or "")).expanduser()
    slot = row.get("slot")
    if slot is None:
        try:
            start = float(row.get("start", 0.0) or 0.0)
            end = float(row.get("end", 0.0) or 0.0)
            slot = max(0.01, end - start)
        except (TypeError, ValueError):
            slot = 0.01
    else:
        slot = max(0.01, float(slot))

    wav_duration = None
    if wav_path.exists():
        wav_duration = _wav_duration_seconds(wav_path)
    elif row.get("wav_duration") is not None:
        wav_duration = float(row.get("wav_duration") or 0.0)

    if wav_duration is None:
        delta = None
    else:
        delta = wav_duration - slot

    asr_text = normalize_text(str(row.get("asr_text") or row.get("asr_backcheck") or ""))
    asr_mode = "provided" if asr_text else "generated"
    if not asr_text and wav_path.exists():
        asr_text = _transcribe_audio(
            wav_path,
            backend=asr_backend,
            whisper_model=whisper_model,
            download_root=download_root,
            device=device,
        )
    if asr_text:
        asr_mode = "generated" if asr_mode == "generated" else asr_mode
    elif asr_backend == "disabled":
        asr_text = text
        asr_mode = "proxy_expected"

    timing_score = _timing_score(delta)
    term_score = _term_score(text, asr_text or text, terminology)
    asr_match = asr_backcheck_score(text, asr_text or text, terminology=terminology)
    wer = _wer(text, asr_text) if asr_text else 1.0
    cer = _cer(text, asr_text) if asr_text else 1.0
    wer_score = max(0.0, 1.0 - wer)
    cer_score = max(0.0, 1.0 - cer)

    final_score = round(
        (0.35 * asr_match)
        + (0.20 * term_score)
        + (0.15 * wer_score)
        + (0.10 * cer_score)
        + (0.20 * timing_score),
        3,
    )

    return {
        "idx": row.get("idx"),
        "start": row.get("start"),
        "end": row.get("end"),
        "slot": round(slot, 3),
        "text": text,
        "wav": str(wav_path),
        "wav_duration": round(wav_duration, 3) if wav_duration is not None else None,
        "delta": round(delta, 3) if delta is not None else None,
        "asr_text": asr_text,
        "asr_mode": asr_mode,
        "wer": wer,
        "cer": cer,
        "term_score": round(term_score, 3),
        "timing_score": round(timing_score, 3),
        "asr_match_score": round(asr_match, 3),
        "final_score": final_score,
        "ok": final_score >= report_threshold,
    }


def _build_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not rows:
        return {
            "segments": 0,
            "passed": 0,
            "failed": 0,
            "avg_final_score": 0.0,
            "avg_wer": 0.0,
            "avg_cer": 0.0,
            "avg_timing_score": 0.0,
            "avg_asr_match_score": 0.0,
            "threshold": threshold,
        }
    return {
        "segments": len(rows),
        "passed": sum(1 for row in rows if row.get("ok")),
        "failed": sum(1 for row in rows if not row.get("ok")),
        "avg_final_score": round(sum(float(row.get("final_score", 0.0) or 0.0) for row in rows) / len(rows), 3),
        "avg_wer": round(sum(float(row.get("wer", 0.0) or 0.0) for row in rows) / len(rows), 4),
        "avg_cer": round(sum(float(row.get("cer", 0.0) or 0.0) for row in rows) / len(rows), 4),
        "avg_timing_score": round(sum(float(row.get("timing_score", 0.0) or 0.0) for row in rows) / len(rows), 3),
        "avg_asr_match_score": round(sum(float(row.get("asr_match_score", 0.0) or 0.0) for row in rows) / len(rows), 3),
        "proxy_rows": sum(1 for row in rows if row.get("asr_mode") == "proxy_expected"),
        "threshold": threshold,
    }


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_level19_report.json")
    )
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else input_path.with_name(f"{input_path.stem}_level19_summary.json")
    )

    items = _load_items(input_path)
    terminology = TerminologyMemory()
    rows: list[dict[str, Any]] = []
    total = len(items)
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{total}] level19")
        rows.append(
            _evaluate_row(
                item,
                terminology=terminology,
                asr_backend=args.asr_backend,
                whisper_model=args.whisper_model,
                download_root=args.download_root,
                device=args.device,
                report_threshold=args.report_threshold,
            )
        )

    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(_build_summary(rows, args.report_threshold), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"HOTOVO: {output_path}")
    print(f"SUMMARY: {summary_path}")


if __name__ == "__main__":
    main()
