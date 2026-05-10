from __future__ import annotations

from typing import Any


PROMPT_MAP = {
    "technical_definition": "locked_terms",
    "technical_explanation": "balanced",
    "example_explanation": "dub_friendly",
    "summary": "balanced",
    "general_explanation": "dub_friendly",
}


def choose_prompt(
    segment_type: str,
    features: dict[str, Any],
    *,
    learned_prompt: str | None = None,
) -> str:
    cps = float(features.get("cps", 0.0) or 0.0)
    asr_match = float(features.get("asr_match_score", 1.0) or 1.0)
    has_sql_terms = bool(features.get("has_sql_terms"))

    if cps > 17.0:
        return "aggressive_shorten"
    if has_sql_terms and asr_match < 0.9:
        return "locked_terms"
    if learned_prompt:
        return learned_prompt
    return PROMPT_MAP.get(segment_type, "balanced")
