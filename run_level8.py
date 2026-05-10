#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.level8_manager import Level8Manager
from src.utils_io import save_json

ROOT = Path(__file__).resolve().parent
DEFAULT_MEMORY = ROOT / "memory" / "project_memory.json"
DEFAULT_ANALYSIS = ROOT / "memory" / "level8_analysis.json"
DEFAULT_REWRITE_DATASET = ROOT / "memory" / "rewrite_dataset.json"
DEFAULT_NEGATIVE_DATASET = ROOT / "memory" / "rewrite_negative_dataset.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 8 rule mining and dataset export from project memory")
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY), help="Path to project memory JSON")
    parser.add_argument("--analysis-output", default=str(DEFAULT_ANALYSIS), help="Path to Level 8 analysis JSON")
    parser.add_argument("--rewrite-dataset-output", default=str(DEFAULT_REWRITE_DATASET), help="Path to positive rewrite dataset JSON")
    parser.add_argument("--negative-dataset-output", default=str(DEFAULT_NEGATIVE_DATASET), help="Path to negative rewrite dataset JSON")
    return parser


def process(args: argparse.Namespace) -> None:
    manager = Level8Manager(args.memory_file)
    analysis = manager.analyze()
    manager.update_memory(analysis)

    save_json(args.analysis_output, analysis)
    save_json(args.rewrite_dataset_output, analysis.get("rewrite_dataset", []))
    save_json(args.negative_dataset_output, analysis.get("negative_dataset", []))

    print(
        f"Level 8 analysis: history={analysis.get('history_size', 0)} "
        f"rewrite_dataset={analysis.get('rewrite_dataset_size', 0)} "
        f"negative_dataset={analysis.get('negative_dataset_size', 0)}"
    )
    print(f"Hotovo: {Path(args.analysis_output).expanduser().resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
