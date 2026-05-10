from __future__ import annotations

from typing import Any


def _style_flags(text: str) -> dict[str, bool]:
    clean = (text or "").lower()
    return {
        "formal": any(x in clean for x in ("uvedieme si", "nasledne", "vykoname", "realizujeme")),
        "spoken": any(x in clean for x in ("pozrime sa", "podme", "teraz si ukazeme", "mozeme pouzit")),
        "too_literal": any(
            x in clean
            for x in (
                "hlboky ponor",
                "deep dive",
                "nasich udajoch",
                "go a",
                "odstranit ho a nahradit ho",
            )
        ),
    }


def spoken_naturalness_score(text: str) -> float:
    clean = (text or "").lower()
    flags = _style_flags(clean)
    score = 1.0
    if flags["too_literal"]:
        score -= 0.35
    if len(clean) > 220:
        score -= 0.10
    if clean.count(",") >= 4:
        score -= 0.10
    if any(x in clean for x in ("toto je to, co", "v poriadku priatelia", "inside", "specialnych funkcii sql o tom")):
        score -= 0.15
    return round(max(0.0, score), 3)


def coherence_score(
    prev_seg: dict[str, Any] | None,
    curr_seg: dict[str, Any],
    next_seg: dict[str, Any] | None,
) -> float:
    text = curr_seg.get("tts_input", "")
    flags = _style_flags(text)
    score = 1.0
    if flags["too_literal"]:
        score -= 0.35
    if prev_seg:
        prev_flags = _style_flags(prev_seg.get("tts_input", ""))
        if prev_flags["formal"] and flags["spoken"]:
            score -= 0.10
        if prev_flags["spoken"] and flags["formal"]:
            score -= 0.10
    if next_seg:
        next_flags = _style_flags(next_seg.get("tts_input", ""))
        if flags["formal"] and next_flags["spoken"]:
            score -= 0.05
        if flags["spoken"] and next_flags["formal"]:
            score -= 0.05
    return round(max(0.0, score), 3)
