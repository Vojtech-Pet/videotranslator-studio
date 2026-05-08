from __future__ import annotations

from typing import Any


def get_slot(seg: dict[str, Any]) -> float:
    start = float(seg.get("start", 0.0) or 0.0)
    end = float(seg.get("end", 0.0) or 0.0)
    if end > start:
        return end - start
    return 0.01


def estimate_cps(text: str, slot: float) -> float:
    if slot <= 0:
        return 999.0
    return len(text or "") / slot


def choose_prompt_name(text: str, cps: float) -> str:
    upper = (text or "").upper()
    if any(term in upper for term in ["SQL", "NULL", "IS NULL", "IS NOT NULL", "ISNULL", "COALESCE"]):
        if cps > 17.0:
            return "aggressive_shorten"
        return "locked_terms"
    if cps > 17.0:
        return "aggressive_shorten"
    return "dub_friendly"
