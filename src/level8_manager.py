from __future__ import annotations

from pathlib import Path
from typing import Any

from src.dataset_builder import build_negative_examples, build_rewrite_dataset
from src.evaluation_set import EVAL_SET
from src.failure_clusterer import cluster_failures
from src.policy_updater import apply_rule_updates
from src.project_memory import ProjectMemory
from src.prompt_ranker import rank_prompts
from src.rule_miner import mine_banned_forms, mine_retry_recommendations


class Level8Manager:
    def __init__(self, memory_path: str | Path):
        self.memory = ProjectMemory(Path(memory_path).expanduser().resolve())
        self.memory.load()

    def analyze(self) -> dict[str, Any]:
        history = list(self.memory.data.get("segment_history", []) or [])
        banned_forms = mine_banned_forms(history)
        retry_recommendations = mine_retry_recommendations(history)
        clusters = cluster_failures(history)
        prompt_ranking = rank_prompts(history)
        rewrite_dataset = build_rewrite_dataset(history)
        negative_dataset = build_negative_examples(history)

        return {
            "history_size": len(history),
            "banned_forms": banned_forms,
            "retry_recommendations": retry_recommendations,
            "failure_clusters": {name: len(records) for name, records in clusters.items()},
            "prompt_ranking": prompt_ranking,
            "rewrite_dataset_size": len(rewrite_dataset),
            "negative_dataset_size": len(negative_dataset),
            "rewrite_dataset": rewrite_dataset,
            "negative_dataset": negative_dataset,
            "evaluation_set_size": len(EVAL_SET),
            "evaluation_set": EVAL_SET,
        }

    def update_memory(self, analysis: dict[str, Any]) -> None:
        mined_rules = {
            "banned_forms": analysis.get("banned_forms", []),
            "retry_recommendations": analysis.get("retry_recommendations", {}),
            "prompt_ranking": analysis.get("prompt_ranking", {}),
        }
        apply_rule_updates(self.memory, mined_rules)
        self.memory.data["last_level8_analysis"] = {
            "history_size": analysis.get("history_size", 0),
            "failure_clusters": analysis.get("failure_clusters", {}),
            "rewrite_dataset_size": analysis.get("rewrite_dataset_size", 0),
            "negative_dataset_size": analysis.get("negative_dataset_size", 0),
        }
        self.memory.save()
