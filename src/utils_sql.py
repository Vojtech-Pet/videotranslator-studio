from __future__ import annotations

import re

from scripts.text_adaptation import SQL_PROTECTED_TERMS, fix_sql_terms as _fix_sql_terms


def fix_sql_terms(text: str) -> str:
    return _fix_sql_terms(text)


def contains_sql_terms(text: str) -> bool:
    clean = text or ""
    for term in SQL_PROTECTED_TERMS:
        pattern = re.escape(term).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", clean, flags=re.IGNORECASE):
            return True
    return False


def protected_terms_in(text: str) -> set[str]:
    clean = fix_sql_terms(text or "")
    found: set[str] = set()
    for term in SQL_PROTECTED_TERMS:
        pattern = re.escape(term).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", clean, flags=re.IGNORECASE):
            found.add(term)
    return found

