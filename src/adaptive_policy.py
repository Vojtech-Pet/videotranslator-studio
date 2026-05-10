from __future__ import annotations

import unicodedata
from typing import Any

from src.project_memory import ProjectMemory


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


CRITICAL_BAD_FORMS = (
    "ajznul",
    "koalesk",
    "esikjuel",
    "esikjul",
    "iz not null",
    "izn ot null",
)

CRITICAL_TERMS = (
    "SQL",
    "NULL",
    "IS NULL",
    "IS NOT NULL",
    "ISNULL",
    "COALESCE",
    "TRUE",
    "FALSE",
    "BOOLEAN",
)


class AdaptivePolicy:
    def __init__(self, memory: ProjectMemory):
        self.memory = memory

    def choose_retry_mode(self, seg: dict[str, Any]) -> str:
        text = _fold(str(seg.get("tts_input") or ""))
        recommendations = self.memory.data.get("policy_recommendations", {})

        if any(term in text for term in CRITICAL_BAD_FORMS):
            return str(recommendations.get("sql_term_corruption") or "locked_terms")

        if float(seg.get("term_score", 1.0) or 1.0) < 0.85:
            return "locked_terms"

        if float(seg.get("final_chars_per_sec", 0.0) or 0.0) > 17.0 or str(seg.get("duration_status") or "") == "too_long":
            locked_rate = self.memory.retry_success_rate("locked_terms")
            shorten_rate = self.memory.retry_success_rate("aggressive_shorten")
            if shorten_rate >= locked_rate:
                return "aggressive_shorten"
            return "locked_terms"

        if float(seg.get("asr_match_score", 1.0) or 1.0) < 0.75:
            return str(recommendations.get("asr_mismatch") or "pronunciation_safe")

        if float(seg.get("coherence_score", 1.0) or 1.0) < 0.75:
            return str(recommendations.get("style_incoherence") or "dub_friendly")

        if float(seg.get("spoken_score", 1.0) or 1.0) < 0.75:
            return "dub_friendly"

        if str(seg.get("duration_status") or "") in {"missing_wav", "pending_audio"}:
            return "render_audio"

        return "balanced"

    def should_force_locked_terms(self, seg: dict[str, Any]) -> bool:
        text = str(seg.get("tts_input") or "")
        source = str(seg.get("source_text") or "")
        haystack = f"{text} {source}"
        return any(term in haystack for term in CRITICAL_TERMS)

    def choose_prompt_mode(self, seg: dict[str, Any], retry_mode: str) -> str:
        if retry_mode != "balanced":
            return retry_mode
        scene_type = str(seg.get("scene_type") or "")
        if scene_type == "technical_tutorial":
            return "balanced"
        return "dub_friendly"
