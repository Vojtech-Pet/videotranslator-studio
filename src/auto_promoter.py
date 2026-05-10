from __future__ import annotations

from src.policy_registry import PolicyRegistry


def promote_if_better(registry: PolicyRegistry, candidate_name: str, ab_result: dict) -> bool:
    if ab_result.get("winner") == "B" and float(ab_result.get("delta", 0.0) or 0.0) >= 0.02:
        registry.set_active_policy(candidate_name)
        registry.save()
        return True
    return False
