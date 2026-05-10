from __future__ import annotations

from src.utils_sql import fix_sql_terms
from src.utils_text import cleanup_spacing, maybe_normalize_numbers, normalize_text


def level1_process_text(
    text: str,
    *,
    normalize_numbers: bool = True,
    lang: str = "sk",
    preserve_sql_terms: bool = True,
) -> str:
    out = cleanup_spacing(text)
    if preserve_sql_terms:
        out = fix_sql_terms(out)
    out = maybe_normalize_numbers(out, enabled=normalize_numbers, lang=lang)
    if preserve_sql_terms:
        out = fix_sql_terms(out)
    return normalize_text(out)

