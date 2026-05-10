from __future__ import annotations

from typing import Any


def candidate_score(candidate: dict[str, Any]) -> float:
    return round(
        0.25 * float(candidate.get("meaning_score", 1.0) or 1.0)
        + 0.20 * float(candidate.get("term_score", 1.0) or 1.0)
        + 0.15 * float(candidate.get("cps_score", 1.0) or 1.0)
        + 0.15 * float(candidate.get("duration_score", 1.0) or 1.0)
        + 0.10 * float(candidate.get("spoken_score", 1.0) or 1.0)
        + 0.10 * float(candidate.get("asr_match_score", 1.0) or 1.0)
        + 0.05 * float(candidate.get("coherence_score", 1.0) or 1.0),
        3,
    )


def rerank_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {}
    ranked = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["candidate_rerank_score"] = candidate_score(item)
        item["candidate_rank_index"] = index
        ranked.append(item)
    ranked.sort(key=lambda item: item["candidate_rerank_score"], reverse=True)
    best = dict(ranked[0])
    best["candidate_rank"] = 1
    best["candidate_pool_size"] = len(ranked)
    return best
