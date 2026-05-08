#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export Level 15 pack from Level 14 JSON.")
    ap.add_argument("--input", required=True, help="Level 14 JSON file.")
    ap.add_argument("--out-dir", default="", help="Output directory. Defaults to input parent.")
    ap.add_argument("--score-threshold", type=float, default=0.82, help="Minimum final_score for clean dataset.")
    ap.add_argument("--cps-limit", type=float, default=17.5, help="Maximum CPS for clean dataset.")
    return ap


def _format_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _load_segments(path: Path) -> tuple[list[dict[str, Any]], Any, bool]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "segments" in data:
        return list(data["segments"]), data, True
    if isinstance(data, list):
        return list(data), data, False
    raise ValueError("Unsupported input JSON format.")


def export_tts_json(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": float(seg.get("start", 0.0) or 0.0),
            "end": float(seg.get("end", 0.0) or 0.0),
            "text": str(seg.get("tts_input") or seg.get("text") or "").strip(),
        }
        for seg in segments
    ]


def export_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        text = str(seg.get("tts_input") or seg.get("text") or "").strip()
        lines.append(str(idx))
        lines.append(f"{_format_time(float(seg.get('start', 0.0) or 0.0))} --> {_format_time(float(seg.get('end', 0.0) or 0.0))}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def export_review(
    segments: list[dict[str, Any]],
    *,
    score_threshold: float,
    cps_limit: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        score = float(seg.get("final_score", 0.0) or 0.0)
        cps = float(seg.get("cps_final", 0.0) or 0.0)
        if score >= score_threshold and cps <= cps_limit:
            continue
        reason_parts: list[str] = []
        if score < score_threshold:
            reason_parts.append("low_score")
        if cps > cps_limit:
            reason_parts.append("high_cps")
        if seg.get("level12_suggested_split"):
            reason_parts.append("suggested_split")
        out.append(
            {
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "text": str(seg.get("tts_input") or seg.get("text") or "").strip(),
                "score": round(score, 3),
                "cps": round(cps, 2),
                "reason": ",".join(reason_parts) or "low_score_or_high_cps",
            }
        )
    return out


def export_clean(
    segments: list[dict[str, Any]],
    *,
    score_threshold: float,
    cps_limit: float,
) -> list[dict[str, Any]]:
    return [
        seg
        for seg in segments
        if float(seg.get("final_score", 0.0) or 0.0) >= score_threshold
        and float(seg.get("cps_final", 0.0) or 0.0) <= cps_limit
    ]


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    segments, _, _ = _load_segments(input_path)

    out_tts_json = out_dir / "tts_ready.json"
    out_srt = out_dir / "tts_preview.srt"
    out_review = out_dir / "manual_review.json"
    out_clean = out_dir / "dataset_clean.json"

    tts_data = export_tts_json(segments)
    out_tts_json.write_text(json.dumps(tts_data, ensure_ascii=False, indent=2), encoding="utf-8")
    out_srt.write_text(export_srt(segments), encoding="utf-8")
    out_review.write_text(
        json.dumps(
            export_review(segments, score_threshold=args.score_threshold, cps_limit=args.cps_limit),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    out_clean.write_text(
        json.dumps(
            export_clean(segments, score_threshold=args.score_threshold, cps_limit=args.cps_limit),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("HOTOVO:")
    print(f" - {out_tts_json}")
    print(f" - {out_srt}")
    print(f" - {out_review}")
    print(f" - {out_clean}")


if __name__ == "__main__":
    main()
