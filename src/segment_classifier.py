from __future__ import annotations

import unicodedata
from typing import Any


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def classify_segment(features: dict[str, Any], text: str) -> str:
    low = _fold(text)

    if any(token in low for token in ("syntax", "klucove slovo", "funkcia", "stlpec", "tabulka", "databaza")):
        return "technical_definition"

    if any(token in low for token in ("priklad", "pozrime sa", "mame dve", "v tomto scenari", "teraz si ukazeme")):
        return "example_explanation"

    if any(token in low for token in ("celkovy obraz", "prehlad", "zhrnutie")):
        return "summary"

    if bool(features.get("has_sql_terms")):
        return "technical_explanation"

    return "general_explanation"
