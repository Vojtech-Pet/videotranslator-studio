from __future__ import annotations

import re


SQL_REPLACEMENTS = {
    r"\bajznul\b": "ISNULL",
    r"\bkoalesk\b": "COALESCE",
    r"\biz not null\b": "IS NOT NULL",
    r"\bizn ot null\b": "IS NOT NULL",
    r"\biz null\b": "IS NULL",
    r"\bnull\b": "NULL",
    r"\bsql\b": "SQL",
    r"\btrue\b": "TRUE",
    r"\bfalse\b": "FALSE",
    r"\bboolean\b": "BOOLEAN",
}


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_punctuation(text: str) -> str:
    out = re.sub(r"\s+([,.:;!?])", r"\1", text or "")
    out = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", out)
    return clean_spaces(out)


def fix_sql_terms(text: str) -> str:
    out = text or ""
    for pattern, replacement in SQL_REPLACEMENTS.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return clean_punctuation(out)
