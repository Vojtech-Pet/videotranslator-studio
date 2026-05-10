#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.level10_manager import Level10Manager
from src.utils_io import load_segments, save_json

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output" / "level9_learned_decisions.json"
DEFAULT_MEMORY = ROOT / "memory" / "project_memory.json"
DEFAULT_REGISTRY = ROOT / "memory" / "policy_registry.json"
DEFAULT_ANALYSIS = ROOT / "memory" / "level10_analysis.json"
DEFAULT_AB_RESULTS = ROOT / "memory" / "level10_ab_results.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 10 self-optimizing dubbing platform scaffold")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to level9_learned_decisions.json")
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY), help="Path to project memory JSON")
    parser.add_argument("--registry-file", default=str(DEFAULT_REGISTRY), help="Path to policy registry JSON")
    parser.add_argument("--analysis-output", default=str(DEFAULT_ANALYSIS), help="Path to Level 10 analysis JSON")
    parser.add_argument("--ab-output", default=str(DEFAULT_AB_RESULTS), help="Path to Level 10 A/B results JSON")
    return parser


def process(args: argparse.Namespace) -> None:
    segments, _wrapper = load_segments(args.input)
    manager = Level10Manager(args.registry_file, args.memory_file)
    manager.bootstrap()
    active_before = manager.registry.get_active_policy_name()

    analysis = manager.build_analysis_summary()
    synthesized_prompt = manager.synthesize_candidate_prompt(analysis)

    candidate_summaries: list[dict] = []
    ab_results: list[dict] = []
    promoted: list[str] = []

    for candidate in manager.create_candidates():
        candidate_policy = dict(candidate)
        candidate_policy["base_prompt"] = synthesized_prompt
        manager.register_candidate_policy(candidate_policy)
        comparison = manager.compare_active_vs_candidate(segments, candidate_policy)
        promoted_now = manager.maybe_promote(candidate_policy["name"], comparison["ab_result"])
        if promoted_now:
            promoted.append(candidate_policy["name"])

        candidate_summaries.append(
            {
                "name": candidate_policy["name"],
                "target_cps": candidate_policy.get("target_cps"),
                "aggressive_shorten_threshold": candidate_policy.get("aggressive_shorten_threshold"),
                "force_locked_terms_on_sql": candidate_policy.get("force_locked_terms_on_sql"),
                "base_prompt": candidate_policy.get("base_prompt"),
                "promoted": promoted_now,
                "ab_result": comparison["ab_result"],
            }
        )
        ab_results.append(comparison)

    payload = {
        "active_policy_before": active_before,
        "active_policy_after": manager.registry.get_active_policy_name(),
        "analysis": analysis,
        "synthesized_prompt": synthesized_prompt,
        "candidate_policies": candidate_summaries,
        "promoted_candidates": promoted,
    }
    save_json(args.analysis_output, payload)
    save_json(args.ab_output, ab_results)

    print(
        f"Level 10 policy evolution: candidates={len(candidate_summaries)} "
        f"promoted={len(promoted)} active={manager.registry.get_active_policy_name()}"
    )
    print(f"Hotovo: {Path(args.analysis_output).expanduser().resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
