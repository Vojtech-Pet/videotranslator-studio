from __future__ import annotations

import re
from dataclasses import dataclass, field


WATCHLIST_TERMS = (
    "SQL",
    "NULL",
    "IS NULL",
    "IS NOT NULL",
    "ISNULL",
    "COALESCE",
    "NULLIF",
    "TRUE",
    "FALSE",
    "BOOLEAN",
)


def _term_pattern(term: str) -> str:
    return re.escape(term).replace(r"\ ", r"\s+")


def _extract_pattern(term: str) -> str:
    if term == "NULLIF":
        return r"(?:\bNULLIF\b|\b(?:function|called|sql\s+function)\s+null\s+if\b)"
    return rf"\b{_term_pattern(term)}\b"


@dataclass(slots=True)
class TerminologyMemory:
    preferred_terms: dict[str, str] = field(
        default_factory=lambda: {
            "sql": "SQL",
            "null": "NULL",
            "is null": "IS NULL",
            "is not null": "IS NOT NULL",
            "isnull": "ISNULL",
            "coalesce": "COALESCE",
            "nullif": "NULLIF",
            "true": "TRUE",
            "false": "FALSE",
            "boolean": "BOOLEAN",
        }
    )
    banned_forms: dict[str, str] = field(
        default_factory=lambda: {
            "ajznul": "ISNULL",
            "koalesk": "COALESCE",
            "iz not null": "IS NOT NULL",
            "izn ot null": "IS NOT NULL",
            "iz null": "IS NULL",
            "esikjuel": "SQL",
            "esikjul": "SQL",
        }
    )

    def normalize_terms(self, text: str) -> str:
        out = text or ""
        for bad, good in self.banned_forms.items():
            out = re.sub(rf"\b{_term_pattern(bad)}\b", good, out, flags=re.IGNORECASE)
        for raw, good in self.preferred_terms.items():
            out = re.sub(rf"\b{_term_pattern(raw)}\b", good, out, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", out).strip()

    def extract_terms_present(self, text: str) -> list[str]:
        clean = self.normalize_terms(text)
        found = []
        for term in WATCHLIST_TERMS:
            if re.search(_extract_pattern(term), clean, flags=re.IGNORECASE):
                found.append(term)
        return found

    def term_consistency_score(self, text: str) -> float:
        text_low = (text or "").lower()
        penalty = 0.0
        for bad in self.banned_forms:
            if re.search(rf"\b{_term_pattern(bad)}\b", text_low, flags=re.IGNORECASE):
                penalty += 0.25
        return round(max(0.0, 1.0 - penalty), 3)

    def term_preservation_score(self, reference_text: str, candidate_text: str) -> float:
        expected = set(self.extract_terms_present(reference_text))
        actual = set(self.extract_terms_present(candidate_text))
        consistency = self.term_consistency_score(candidate_text)
        if not expected:
            return consistency
        coverage = len(expected & actual) / len(expected)
        return round((0.7 * coverage) + (0.3 * consistency), 3)
