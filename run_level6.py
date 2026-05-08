#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.level6_manager import evaluate_scene, summarize_scene_evaluation
from src.utils_io import ensure_output_tree, load_segments, save_json, save_segments

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output" / "level4_final.json"
DEFAULT_OUTPUT = ROOT / "output" / "level6_scene_checked.json"
DEFAULT_SUMMARY = ROOT / "output" / "final" / "level6_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 6 scene-aware evaluation over Level 4 output")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to level4_final.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to Level 6 output JSON")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY), help="Path to Level 6 summary JSON")
    parser.add_argument("--acceptance-threshold", type=float, default=0.82, help="Final score threshold for acceptance")
    return parser


def process(args: argparse.Namespace) -> None:
    ensure_output_tree(ROOT)
    segments, wrapper = load_segments(args.input)
    evaluated = evaluate_scene(segments, acceptance_threshold=args.acceptance_threshold)
    summary = summarize_scene_evaluation(evaluated)

    save_segments(args.output, evaluated, wrapper)
    save_json(args.summary_output, summary)

    print(
        f"Level 6 scene check: accepted={summary['accepted']} "
        f"needs_retry={summary['needs_retry']} "
        f"avg_final_score={summary['avg_final_score']}"
    )
    print(f"Hotovo: {Path(args.output).expanduser().resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
