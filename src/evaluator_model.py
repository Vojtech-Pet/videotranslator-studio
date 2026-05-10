from __future__ import annotations

import unicodedata
from typing import Any


BAD_TERMS = ("ajznul", "koalesk", "esikjuel", "esikjul", "iz not null", "izn ot null")


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def evaluate_segment(seg: dict[str, Any]) -> dict[str, float]:
    text = str(seg.get("tts_input", "") or "")
    folded = _fold(text)

    term_score = 1.0
    for bad in BAD_TERMS:
        if bad in folded:
            term_score -= 0.3
    term_score = max(0.0, term_score)

    final_cps = float(seg.get("final_chars_per_sec", 16.0) or 16.0)
    if final_cps <= 16.0:
        cps_score = 1.0
    elif final_cps <= 17.0:
        cps_score = 0.8
    else:
        cps_score = 0.5

    duration_score = float(seg.get("duration_score", 1.0) or 1.0)
    asr_match_score = float(seg.get("asr_match_score", 1.0) or 1.0)
    coherence_score = float(seg.get("coherence_score", 1.0) or 1.0)
    spoken_score = float(seg.get("spoken_score", 1.0) or 1.0)

    final_score = (
        0.25 * term_score
        + 0.20 * cps_score
        + 0.20 * duration_score
        + 0.15 * asr_match_score
        + 0.10 * coherence_score
        + 0.10 * spoken_score
    )

    return {
        "term_score": round(term_score, 3),
        "cps_score": round(cps_score, 3),
        "duration_score": round(duration_score, 3),
        "asr_match_score": round(asr_match_score, 3),
        "coherence_score": round(coherence_score, 3),
        "spoken_score": round(spoken_score, 3),
        "final_score": round(final_score, 3),
    }


def simulate_policy_effect(seg: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    simulated = dict(seg)
    current_cps = float(simulated.get("final_chars_per_sec", 16.0) or 16.0)
    target_cps = float(policy.get("target_cps", 16.0) or 16.0)
    shorten_threshold = float(policy.get("aggressive_shorten_threshold", 17.0) or 17.0)
    force_locked = bool(policy.get("force_locked_terms_on_sql"))
    has_sql_terms = bool(simulated.get("force_locked_terms")) or "SQL" in str(simulated.get("tts_input") or "")

    effective_cps = current_cps
    if current_cps > shorten_threshold:
        effective_cps = max(target_cps, current_cps - 0.6)
    elif current_cps > target_cps:
        effective_cps = max(target_cps, current_cps - 0.25)
    simulated["final_chars_per_sec"] = round(effective_cps, 3)

    duration_score = float(simulated.get("duration_score", 1.0) or 1.0)
    if effective_cps <= target_cps and duration_score < 1.0:
        duration_score = min(1.0, duration_score + 0.08)
    simulated["duration_score"] = round(duration_score, 3)

    term_score = float(simulated.get("term_score", 1.0) or 1.0)
    if force_locked and has_sql_terms:
        term_score = min(1.0, term_score + 0.08)
    simulated["term_score"] = round(term_score, 3)

    spoken_score = float(simulated.get("spoken_score", 1.0) or 1.0)
    if effective_cps <= target_cps:
        spoken_score = min(1.0, spoken_score + 0.04)
    simulated["spoken_score"] = round(spoken_score, 3)

    return simulated
