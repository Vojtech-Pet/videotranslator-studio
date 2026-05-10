from __future__ import annotations

from typing import Any


def generate_candidate_policies(base_policy: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(base_policy)
    candidates: list[dict[str, Any]] = []

    lower_cps = dict(base)
    lower_cps["target_cps"] = max(14.5, float(base.get("target_cps", 16.0) or 16.0) - 0.5)
    lower_cps["name"] = "candidate_lower_cps"
    candidates.append(lower_cps)

    locked_terms = dict(base)
    locked_terms["force_locked_terms_on_sql"] = True
    locked_terms["name"] = "candidate_locked_terms"
    candidates.append(locked_terms)

    earlier_shorten = dict(base)
    earlier_shorten["aggressive_shorten_threshold"] = float(
        base.get("aggressive_shorten_threshold", 17.0) or 17.0
    ) - 0.5
    earlier_shorten["name"] = "candidate_earlier_shorten"
    candidates.append(earlier_shorten)

    return candidates
