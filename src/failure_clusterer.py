from __future__ import annotations

import unicodedata
from typing import Any


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def classify_failure(record: dict[str, Any]) -> str:
    text = _fold(str(record.get("tts_input") or ""))

    if any(token in text for token in ("ajznul", "koalesk", "esikjuel", "esikjul", "iz not null", "izn ot null")):
        return "sql_term_corruption"
    if float(record.get("term_score", 1.0) or 1.0) < 0.8:
        return "term_preservation"
    if float(record.get("duration_score", 1.0) or 1.0) < 0.6:
        return "duration_mismatch"
    if float(record.get("asr_match_score", 1.0) or 1.0) < 0.75:
        return "asr_mismatch"
    if float(record.get("coherence_score", 1.0) or 1.0) < 0.75:
        return "style_incoherence"
    if float(record.get("spoken_score", 1.0) or 1.0) < 0.75:
        return "unnatural_spoken_style"
    return "other"


def cluster_failures(segment_history: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for record in segment_history:
        cluster = classify_failure(record)
        clusters.setdefault(cluster, []).append(record)
    return clusters
