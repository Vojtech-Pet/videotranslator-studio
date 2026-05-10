#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.level11_manager import Level11Manager, summarize_level11_segments
from src.utils_io import load_segments, save_json, save_segments

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output" / "level9_learned_decisions.json"
DEFAULT_OUTPUT = ROOT / "output" / "level11_multi_agent_results.json"
DEFAULT_MEMORY = ROOT / "memory" / "project_memory.json"
DEFAULT_REGISTRY = ROOT / "memory" / "policy_registry.json"
DEFAULT_SUMMARY = ROOT / "output" / "final" / "level11_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 11 autonomous multi-agent dubbing scaffold")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to level9/level10 segment JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to Level 11 output JSON")
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY), help="Path to project memory JSON")
    parser.add_argument("--registry-file", default=str(DEFAULT_REGISTRY), help="Path to policy registry JSON")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY), help="Path to Level 11 summary JSON")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum autonomous retry rounds per segment")
    return parser


def process(args: argparse.Namespace) -> None:
    segments, wrapper = load_segments(args.input)
    manager = Level11Manager(args.memory_file, args.registry_file, max_retries=args.max_retries)
    processed = manager.process_segments(segments)
    summary = summarize_level11_segments(processed)
    summary["memory_file"] = str(Path(args.memory_file).expanduser().resolve())
    summary["registry_file"] = str(Path(args.registry_file).expanduser().resolve())
    manager.finalize(summary)

    save_segments(args.output, processed, wrapper)
    save_json(args.summary_output, summary)

    print(
        f"Level 11 multi-agent run: ok={summary['ok_segments']} "
        f"needs_retry={summary['needs_retry']} "
        f"complex={summary['complex_segments']}"
    )
    print(f"Hotovo: {Path(args.output).expanduser().resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
