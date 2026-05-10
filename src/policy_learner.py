from __future__ import annotations

from collections import defaultdict
from typing import Any


class PolicyLearner:
    def __init__(self) -> None:
        self.stats: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    def observe(self, features: dict[str, Any], decision: dict[str, Any], outcome: dict[str, Any]) -> None:
        key = (
            str(features.get("segment_type") or "unknown"),
            str(decision.get("prompt_name") or "balanced"),
            str(decision.get("retry_mode") or "balanced"),
        )
        self.stats[key].append(float(outcome.get("final_score", 0.0) or 0.0))

    def best_prompt_for_segment_type(self, segment_type: str) -> str | None:
        best_prompt: str | None = None
        best_score = -1.0
        for (seg_type, prompt_name, _retry_mode), scores in self.stats.items():
            if seg_type != segment_type or not scores:
                continue
            average = sum(scores) / len(scores)
            if average > best_score:
                best_score = average
                best_prompt = prompt_name
        return best_prompt

    def best_retry_mode_for_segment_type(self, segment_type: str) -> str | None:
        best_retry: str | None = None
        best_score = -1.0
        for (seg_type, _prompt_name, retry_mode), scores in self.stats.items():
            if seg_type != segment_type or not scores:
                continue
            average = sum(scores) / len(scores)
            if average > best_score:
                best_score = average
                best_retry = retry_mode
        return best_retry

    def export_summary(self) -> dict[str, Any]:
        by_segment_type: dict[str, dict[str, Any]] = {}
        for (segment_type, prompt_name, retry_mode), scores in self.stats.items():
            bucket = by_segment_type.setdefault(
                segment_type,
                {
                    "best_prompt": None,
                    "best_retry_mode": None,
                    "prompt_scores": {},
                    "retry_scores": {},
                },
            )
            average = round(sum(scores) / len(scores), 3)
            current_prompt = bucket["prompt_scores"].get(prompt_name, -1.0)
            current_retry = bucket["retry_scores"].get(retry_mode, -1.0)
            if average > current_prompt:
                bucket["prompt_scores"][prompt_name] = average
            if average > current_retry:
                bucket["retry_scores"][retry_mode] = average

        for segment_type, bucket in by_segment_type.items():
            if bucket["prompt_scores"]:
                bucket["best_prompt"] = max(bucket["prompt_scores"].items(), key=lambda item: item[1])[0]
            if bucket["retry_scores"]:
                bucket["best_retry_mode"] = max(bucket["retry_scores"].items(), key=lambda item: item[1])[0]
        return {
            "entries": len(self.stats),
            "segment_types": by_segment_type,
        }
