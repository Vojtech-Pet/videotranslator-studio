from __future__ import annotations

from typing import Any


def compare_policy_results(results_a: list[dict[str, Any]], results_b: list[dict[str, Any]]) -> dict[str, Any]:
    avg_a = sum(float(item.get("final_score", 0.0) or 0.0) for item in results_a) / max(1, len(results_a))
    avg_b = sum(float(item.get("final_score", 0.0) or 0.0) for item in results_b) / max(1, len(results_b))
    winner = "A" if avg_a >= avg_b else "B"
    return {
        "avg_score_a": round(avg_a, 3),
        "avg_score_b": round(avg_b, 3),
        "winner": winner,
        "delta": round(abs(avg_a - avg_b), 3),
    }
