from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any


COMMON_BAD_FORMS = (
    "ajznul",
    "koalesk",
    "esikjuel",
    "esikjul",
    "iz not null",
    "izn ot null",
    "iz null",
)


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def mine_banned_forms(segment_history: list[dict[str, Any]]) -> list[str]:
    counter: Counter[str] = Counter()
    for record in segment_history:
        text = _fold(str(record.get("tts_input") or ""))
        for bad_form in COMMON_BAD_FORMS:
            if bad_form in text:
                counter[bad_form] += 1
    return sorted(form for form, count in counter.items() if count >= 2)


def mine_retry_recommendations(segment_history: list[dict[str, Any]]) -> dict[str, str]:
    recommendations: dict[str, str] = {}

    sql_fail_count = 0
    sql_locked_success = 0
    duration_fail_count = 0
    aggressive_success = 0
    asr_fail_count = 0
    pronunciation_success = 0

    for record in segment_history:
        text = _fold(str(record.get("tts_input") or ""))
        retry_mode = str(record.get("retry_mode") or "")
        accepted = bool(record.get("accepted", False))
        duration_score = float(record.get("duration_score", 1.0) or 1.0)
        asr_match_score = float(record.get("asr_match_score", 1.0) or 1.0)

        if any(bad_form in text for bad_form in COMMON_BAD_FORMS):
            sql_fail_count += 1
            if retry_mode == "locked_terms" and accepted:
                sql_locked_success += 1

        if duration_score < 0.6:
            duration_fail_count += 1
            if retry_mode == "aggressive_shorten" and accepted:
                aggressive_success += 1

        if asr_match_score < 0.75:
            asr_fail_count += 1
            if retry_mode == "pronunciation_safe" and accepted:
                pronunciation_success += 1

    if sql_fail_count >= 3 and sql_locked_success / max(1, sql_fail_count) >= 0.5:
        recommendations["sql_term_corruption"] = "locked_terms"
    if duration_fail_count >= 3 and aggressive_success / max(1, duration_fail_count) >= 0.4:
        recommendations["duration_mismatch"] = "aggressive_shorten"
    if asr_fail_count >= 3 and pronunciation_success / max(1, asr_fail_count) >= 0.4:
        recommendations["asr_mismatch"] = "pronunciation_safe"
    return recommendations
