from __future__ import annotations

from collections import defaultdict
from typing import Any


def rank_prompts(segment_history: list[dict[str, Any]]) -> dict[str, float]:
    stats: dict[str, list[float]] = defaultdict(list)
    for record in segment_history:
        prompt_name = str(record.get("prompt_name") or "").strip()
        final_score = record.get("final_score")
        if not prompt_name or final_score is None:
            continue
        try:
            stats[prompt_name].append(float(final_score))
        except (TypeError, ValueError):
            continue

    averages = {name: round(sum(scores) / len(scores), 3) for name, scores in stats.items() if scores}
    return dict(sorted(averages.items(), key=lambda item: item[1], reverse=True))
