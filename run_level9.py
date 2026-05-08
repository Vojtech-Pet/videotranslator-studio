#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.level9_manager import Level9Manager, summarize_level9_segments
from src.utils_io import load_segments, save_json, save_segments

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output" / "level7_project_adaptive.json"
DEFAULT_OUTPUT = ROOT / "output" / "level9_learned_decisions.json"
DEFAULT_MEMORY = ROOT / "memory" / "project_memory.json"
DEFAULT_SUMMARY = ROOT / "output" / "final" / "level9_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 9 learned decision layer over Level 7 output")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to level7_project_adaptive.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to Level 9 output JSON")
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY), help="Path to project memory JSON")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY), help="Path to Level 9 summary JSON")
    return parser


def process(args: argparse.Namespace) -> None:
    segments, wrapper = load_segments(args.input)
    manager = Level9Manager(args.memory_file)
    prepared = manager.process_segments(segments)
    for seg in prepared:
        if seg.get("final_score") is not None:
            manager.learn_from_result(seg)
    manager.finalize_project()

    summary = summarize_level9_segments(prepared)
    summary["memory_file"] = str(Path(args.memory_file).expanduser().resolve())

    save_segments(args.output, prepared, wrapper)
    save_json(args.summary_output, summary)

    print(
        f"Level 9 learned decisions: segment_types={len(summary['segment_types'])} "
        f"prompt_variants={len(summary['prompt_names'])} "
        f"reranked={summary['reranked_segments']}"
    )
    print(f"Hotovo: {Path(args.output).expanduser().resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
