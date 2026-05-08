#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.level5_manager import Level5Manager, summarize_level5_segments
from src.utils_io import ensure_output_tree, load_segments, save_json, save_segments

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output" / "level4_final.json"
DEFAULT_OUTPUT = ROOT / "output" / "level5_quality_checked.json"
DEFAULT_SUMMARY = ROOT / "output" / "final" / "level5_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 5 quality-aware autonomous dubbing loop over Level 4 output")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to level4_final.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to Level 5 output JSON")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY), help="Path to Level 5 summary JSON")
    parser.add_argument("--acceptance-threshold", type=float, default=0.82, help="Final score threshold for acceptance")
    parser.add_argument("--best-of-n-margin", type=float, default=0.04, help="Minimum score margin required to switch from rendered audio to a rerender candidate")
    return parser


def process(args: argparse.Namespace) -> None:
    ensure_output_tree(ROOT)
    segments, wrapper = load_segments(args.input)
    manager = Level5Manager(
        acceptance_threshold=args.acceptance_threshold,
        best_of_n_margin=args.best_of_n_margin,
    )
    processed = manager.process_segments(segments)
    summary = summarize_level5_segments(processed)

    save_segments(args.output, processed, wrapper)
    save_json(args.summary_output, summary)

    print(
        f"Level 5 quality loop: accepted={summary['accepted']} "
        f"needs_retry={summary['needs_retry']} "
        f"candidate_wins={summary['candidate_wins']}"
    )
    print(f"Hotovo: {Path(args.output).expanduser().resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
