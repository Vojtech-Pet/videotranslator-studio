from __future__ import annotations

from typing import Any

from src.project_memory import ProjectMemory


def apply_rule_updates(memory: ProjectMemory, mined_rules: dict[str, Any]) -> None:
    for bad_form in mined_rules.get("banned_forms", []):
        memory.add_banned_form(str(bad_form))

    retry_recommendations = mined_rules.get("retry_recommendations", {})
    memory.data.setdefault("policy_recommendations", {})
    memory.data["policy_recommendations"].update(retry_recommendations)

    prompt_ranking = mined_rules.get("prompt_ranking", {})
    if prompt_ranking:
        memory.data["prompt_ranking"] = prompt_ranking
