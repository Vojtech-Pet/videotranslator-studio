from __future__ import annotations

import unicodedata
from typing import Any


CRITICAL_TERMS = [
    "SQL",
    "NULL",
    "IS NULL",
    "IS NOT NULL",
    "ISNULL",
    "COALESCE",
    "TRUE",
    "FALSE",
    "BOOLEAN",
]


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def extract_features(seg: dict[str, Any]) -> dict[str, Any]:
    text = str(seg.get("tts_input") or "")
    slot = seg.get("slot", max(0.01, float(seg.get("end", 0) or 0) - float(seg.get("start", 0) or 0)))
    try:
        slot_value = max(0.01, float(slot))
    except (TypeError, ValueError):
        slot_value = 0.01

    folded = _fold(text)
    cps = len(text) / slot_value if slot_value > 0 else 999.0
    term_count = sum(1 for term in CRITICAL_TERMS if _fold(term) in folded)

    return {
        "slot": round(slot_value, 3),
        "text_len": len(text),
        "cps": round(cps, 3),
        "term_count": term_count,
        "has_sql_terms": term_count > 0,
        "has_numbers": any(ch.isdigit() for ch in text),
        "coherence_score": float(seg.get("coherence_score", 1.0) or 1.0),
        "asr_match_score": float(seg.get("asr_match_score", 1.0) or 1.0),
        "duration_score": float(seg.get("duration_score", 1.0) or 1.0),
        "duration_delta": float(seg.get("delta", 0.0) or 0.0),
        "spoken_score": float(seg.get("spoken_score", 1.0) or 1.0),
        "retry_count": int(seg.get("retry_count", 0) or 0),
        "scene_type": str(seg.get("scene_type", "unknown") or "unknown"),
        "segment_type": str(seg.get("segment_type", "unknown") or "unknown"),
    }
