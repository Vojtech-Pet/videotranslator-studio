from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ab_tester import compare_policy_results
from src.auto_promoter import promote_if_better
from src.evaluator_model import evaluate_segment, simulate_policy_effect
from src.policy_registry import PolicyRegistry
from src.project_memory import ProjectMemory
from src.prompt_synthesizer import synthesize_prompt
from src.strategy_search import generate_candidate_policies


class Level10Manager:
    def __init__(self, registry_path: str | Path, memory_path: str | Path):
        self.registry = PolicyRegistry(registry_path)
        self.memory = ProjectMemory(Path(memory_path).expanduser().resolve())
        self.memory.load()

    def bootstrap(self) -> None:
        if "policy_v1" in self.registry.data.get("policies", {}):
            return
        prompt_ranking = self.memory.data.get("prompt_ranking", {})
        base_prompt = "Prepis vetu prirodzene pre technicky dabing."
        if prompt_ranking:
            best_prompt_name = max(prompt_ranking.items(), key=lambda item: item[1])[0]
            base_prompt = str(self.memory.data.get("prompt_templates", {}).get(best_prompt_name, base_prompt))
        policy = {
            "target_cps": 16.0,
            "aggressive_shorten_threshold": 17.0,
            "force_locked_terms_on_sql": True,
            "base_prompt": base_prompt,
        }
        self.registry.register_policy("policy_v1", policy)
        self.registry.save()

    def build_analysis_summary(self) -> dict[str, Any]:
        history = list(self.memory.data.get("segment_history", []) or [])
        total = max(1, len(history))
        duration_failures = sum(1 for item in history if float(item.get("duration_score", 1.0) or 1.0) < 0.6)
        literal_style = sum(1 for item in history if float(item.get("spoken_score", 1.0) or 1.0) < 0.75)
        sql_corruption = sum(1 for item in history if float(item.get("term_score", 1.0) or 1.0) < 0.85)
        return {
            "history_size": len(history),
            "sql_term_corruption_rate": round(sql_corruption / total, 3),
            "literal_style_rate": round(literal_style / total, 3),
            "duration_failure_rate": round(duration_failures / total, 3),
            "failure_clusters": dict(self.memory.data.get("last_level8_analysis", {}).get("failure_clusters", {})),
            "prompt_ranking": dict(self.memory.data.get("prompt_ranking", {})),
        }

    def create_candidates(self) -> list[dict[str, Any]]:
        active = self.registry.get_active_policy()
        return generate_candidate_policies(active)

    def synthesize_candidate_prompt(self, analysis: dict[str, Any]) -> str:
        active = self.registry.get_active_policy()
        return synthesize_prompt(str(active.get("base_prompt", "")), analysis)

    def register_candidate_policy(self, policy: dict[str, Any]) -> None:
        name = str(policy.get("name") or "")
        if not name:
            raise ValueError("Candidate policy must have a name")
        self.registry.register_policy(name, policy)
        self.registry.save()

    def evaluate_policy_on_segments(self, policy: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for seg in segments:
            simulated = simulate_policy_effect(seg, policy)
            scores = evaluate_segment(simulated)
            results.append(
                {
                    "seg": seg.get("seg"),
                    "policy_name": policy.get("name"),
                    "final_score": scores["final_score"],
                    "term_score": scores["term_score"],
                    "cps_score": scores["cps_score"],
                }
            )
        return results

    def compare_active_vs_candidate(
        self,
        segments: list[dict[str, Any]],
        candidate_policy: dict[str, Any],
    ) -> dict[str, Any]:
        active = self.registry.get_active_policy()
        active_named = dict(active)
        active_named["name"] = self.registry.get_active_policy_name()
        active_results = self.evaluate_policy_on_segments(active_named, segments)
        candidate_results = self.evaluate_policy_on_segments(candidate_policy, segments)
        ab_result = compare_policy_results(active_results, candidate_results)
        return {
            "active_policy": active_named["name"],
            "candidate_policy": candidate_policy["name"],
            "ab_result": ab_result,
            "active_results": active_results,
            "candidate_results": candidate_results,
        }

    def maybe_promote(self, candidate_name: str, ab_result: dict[str, Any]) -> bool:
        return promote_if_better(self.registry, candidate_name, ab_result)
