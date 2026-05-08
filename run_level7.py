#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.level7_manager import Level7Manager, summarize_level7_segments
from src.utils_io import load_segments, save_json, save_segments

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output" / "level6_scene_checked.json"
DEFAULT_OUTPUT = ROOT / "output" / "level7_project_adaptive.json"
DEFAULT_MEMORY = ROOT / "memory" / "project_memory.json"
DEFAULT_SUMMARY = ROOT / "output" / "final" / "level7_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 7 project-aware adaptive pass over Level 6 output")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to level6_scene_checked.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to Level 7 output JSON")
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY), help="Path to project memory JSON")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY), help="Path to Level 7 summary JSON")
    parser.add_argument("--acceptance-threshold", type=float, default=0.82, help="Fallback threshold if accepted is missing")
    return parser


def process(args: argparse.Namespace) -> None:
    segments, wrapper = load_segments(args.input)
    manager = Level7Manager(args.memory_file)
    enriched = manager.process_segments(segments)

    for seg in enriched:
        if "accepted" not in seg:
            duration_status = str(seg.get("duration_status") or "").lower()
            has_audio = duration_status not in {"pending_audio", "missing_wav"}
            seg["accepted"] = bool(has_audio and float(seg.get("final_score", 0.0) or 0.0) >= args.acceptance_threshold)
        manager.finalize_segment(seg)

    manager.finalize_project()
    summary = summarize_level7_segments(enriched)
    summary["memory_file"] = str(Path(args.memory_file).expanduser().resolve())

    save_segments(args.output, enriched, wrapper)
    save_json(args.summary_output, summary)

    print(
        f"Level 7 adaptive pass: accepted={summary['accepted']} "
        f"needs_retry={summary['needs_retry']} "
        f"forced_locked_terms={summary['forced_locked_terms']}"
    )
    print(f"Hotovo: {Path(args.output).expanduser().resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
