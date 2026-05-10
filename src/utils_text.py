from __future__ import annotations

import re

from scripts.text_normalizer import normalize_for_tts


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def cleanup_spacing(text: str) -> str:
    out = normalize_text(text)
    out = re.sub(r"\s+([,.:;!?])", r"\1", out)
    out = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", out)
    return normalize_text(out)


def choose_segment_text(seg: dict, *, prefer_text_original: bool = True) -> str:
    preferred = [
        "text_original",
        "level3_text",
        "level2_text",
        "level1_text",
        "tts_input",
        "text",
        "translated",
    ]
    fallback = [
        "text",
        "translated",
        "tts_input",
        "level3_text",
        "level2_text",
        "level1_text",
        "text_original",
    ]
    keys = preferred if prefer_text_original else fallback
    for key in keys:
        value = normalize_text(seg.get(key))
        if value:
            return value
    return ""


def maybe_normalize_numbers(text: str, *, enabled: bool = True, lang: str = "sk") -> str:
    clean = normalize_text(text)
    if not enabled or not re.search(r"\d", clean):
        return clean
    try:
        normalized = normalize_for_tts(
            clean,
            content_type="technical",
            lang=lang,
            mode="light",
        )
        return cleanup_spacing(normalized)
    except Exception:
        return clean

