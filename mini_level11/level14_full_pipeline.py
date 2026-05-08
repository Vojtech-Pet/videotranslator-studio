#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for entry in (ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from sql_logic_guard import guard_sql_text  # noqa: E402
from src.coherence_checker import spoken_naturalness_score  # noqa: E402
from src.level12_manager import Level12Manager, estimate_cps, target_cps_for_type  # noqa: E402
from src.level13_manager import Level13Manager  # noqa: E402
from src.terminology_memory import TerminologyMemory  # noqa: E402
from src.utils_text import normalize_text  # noqa: E402

HARD_CPS_LIMIT = 17.5


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Level 14 full JSON pre-TTS runner.")
    ap.add_argument("--input", required=True, help="Input JSON with segments.")
    ap.add_argument("--output", default="", help="Output JSON path. Defaults to *_LEVEL14.json beside input.")
    ap.add_argument("--report", default="", help="Output report path. Defaults to *_LEVEL14_report.json beside input.")
    ap.add_argument("--provider", choices=["ollama", "disabled"], default="ollama", help="Rewrite provider for Level 13.")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama generate endpoint.")
    ap.add_argument("--ollama-model", default="gemma3:27b", help="Ollama model.")
    ap.add_argument("--ollama-timeout", type=int, default=120, help="Ollama timeout in seconds.")
    ap.add_argument("--max-ollama-segments", type=int, default=24, help="Maximum segments sent to Ollama in Level 13.")
    return ap


def _load_segments(path: Path) -> tuple[list[dict[str, Any]], Any, bool]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "segments" in data:
        return list(data["segments"]), data, True
    if isinstance(data, list):
        return list(data), data, False
    raise ValueError("Unsupported JSON format. Expected list or object with 'segments'.")


def _save_segments(path: Path, wrapper: Any, wrapped: bool, segments: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        if wrapped:
            wrapper["segments"] = segments
            json.dump(wrapper, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(segments, handle, ensure_ascii=False, indent=2)


def _preclean_segment(seg: dict[str, Any]) -> dict[str, Any]:
    out = dict(seg)
    source_text = normalize_text(str(out.get("text_src") or out.get("source_text") or ""))
    base_text = normalize_text(str(out.get("tts_input") or out.get("text") or source_text))
    cleaned = guard_sql_text(base_text, source_text=source_text, fallback_text=base_text)
    out["text"] = cleaned
    out["tts_input"] = cleaned
    return out


def _score_segment(seg: dict[str, Any], terminology: TerminologyMemory) -> dict[str, Any]:
    text = normalize_text(str(seg.get("tts_input") or seg.get("text") or ""))
    source_text = normalize_text(str(seg.get("text_src") or seg.get("source_text") or text))
    seg_type = str(seg.get("level13_segment_type") or seg.get("level12_segment_type") or seg.get("segment_type") or "default")
    target_cps = float(seg.get("level13_target_cps") or seg.get("level12_target_cps") or target_cps_for_type(seg_type))
    duration = max(0.01, float(seg.get("end", 0.0) or 0.0) - float(seg.get("start", 0.0) or 0.0))
    cps_value = estimate_cps(text, duration)

    term_score = terminology.term_preservation_score(source_text or text, text)
    spoken_score = spoken_naturalness_score(text)
    if cps_value <= target_cps:
        cps_score = 1.0
    elif cps_value <= target_cps + 1.0:
        cps_score = 0.8
    elif cps_value <= HARD_CPS_LIMIT:
        cps_score = 0.5
    else:
        cps_score = 0.2

    final_score = round((0.35 * term_score) + (0.35 * cps_score) + (0.30 * spoken_score), 3)

    out = dict(seg)
    out["tts_input"] = text
    out["segment_type"] = seg_type
    out["target_cps"] = round(target_cps, 2)
    out["cps_final"] = round(cps_value, 2)
    out["term_score"] = round(term_score, 3)
    out["spoken_score"] = round(spoken_score, 3)
    out["cps_score"] = round(cps_score, 3)
    out["final_score"] = final_score
    return out


def _build_report(segments: list[dict[str, Any]]) -> dict[str, Any]:
    segment_types: dict[str, int] = {}
    for seg in segments:
        seg_type = str(seg.get("segment_type") or "default")
        segment_types[seg_type] = segment_types.get(seg_type, 0) + 1

    return {
        "total_segments": len(segments),
        "too_long_after_level14": sum(
            1
            for seg in segments
            if float(seg.get("cps_final", 0.0) or 0.0) > float(seg.get("target_cps", 16.0) or 16.0) + 1.0
        ),
        "low_score_after_level14": sum(1 for seg in segments if float(seg.get("final_score", 0.0) or 0.0) < 0.82),
        "suggested_split_count": sum(1 for seg in segments if seg.get("level12_suggested_split")),
        "segment_type_breakdown": segment_types,
    }


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_name(f"{input_path.stem}_LEVEL14.json")
    report_path = Path(args.report).expanduser().resolve() if args.report else input_path.with_name(f"{input_path.stem}_LEVEL14_report.json")

    segments, wrapper, wrapped = _load_segments(input_path)
    cleaned_segments = [_preclean_segment(seg) for seg in segments]

    level12 = Level12Manager()
    level12_segments = level12.process_segments(cleaned_segments)

    level13 = Level13Manager(
        provider=args.provider,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        ollama_timeout=args.ollama_timeout,
        max_ollama_segments=args.max_ollama_segments,
        verbose=True,
    )
    level13_segments = level13.process_segments(level12_segments)

    terminology = TerminologyMemory()
    final_segments: list[dict[str, Any]] = []
    total = len(level13_segments)
    for idx, seg in enumerate(level13_segments, start=1):
        print(f"[{idx}/{total}] level14")
        final_segments.append(_score_segment(seg, terminology))

    _save_segments(output_path, wrapper, wrapped, final_segments)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_build_report(final_segments), handle, ensure_ascii=False, indent=2)

    print(f"HOTOVO: {output_path}")
    print(f"REPORT: {report_path}")


if __name__ == "__main__":
    main()
