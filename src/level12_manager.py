from __future__ import annotations

import re
from collections import Counter
from typing import Any

from sql_logic_guard import guard_sql_text, protected_sql_terms_in, source_guided_template
from src.utils_text import cleanup_spacing, normalize_text

TARGET_CPS_DEFAULT = 16.0
TARGET_CPS_DEFINITION = 15.2
TARGET_CPS_EXAMPLE = 16.2
TARGET_CPS_SUMMARY = 16.5

SPLIT_MARKERS = (":", ";", ",", " a ", " ale ", " takže ", " preto ", " ak ")


def slot_duration(seg: dict[str, Any]) -> float:
    try:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.01
    return max(0.01, end - start)


def estimate_cps(text: str, duration: float) -> float:
    return len(normalize_text(text)) / max(duration, 0.01)


def classify_segment_type(text: str) -> str:
    low = normalize_text(text).lower()
    if any(x in low for x in ("isnull", "is null", "is not null", "coalesce", "nullif", "syntax", "argumenty")):
        return "definition"
    if any(x in low for x in ("príklad", "example", "objednávky", "scenár", "scenario", "pozrime sa", "adres")):
        return "example"
    if any(x in low for x in ("prehľad", "overview", "zhrnutie", "celkový obraz", "big picture")):
        return "summary"
    return "default"


def target_cps_for_type(seg_type: str) -> float:
    if seg_type == "definition":
        return TARGET_CPS_DEFINITION
    if seg_type == "example":
        return TARGET_CPS_EXAMPLE
    if seg_type == "summary":
        return TARGET_CPS_SUMMARY
    return TARGET_CPS_DEFAULT


def _shorten_text(text: str) -> str:
    out = normalize_text(text)
    rules = [
        (r"\bTakže\b,?\s*", ""),
        (r"\bTeraz\b,?\s*", ""),
        (r"\bAko môžete vidieť\b,?\s*", ""),
        (r"\bV tomto scenári\b,?\s*", ""),
        (r"\bPoďme sa pozrieť\b", "Pozrime sa"),
        (r"\bPoďme si ukázať\b", "Ukážme si"),
        (r"\bveľmi\b", ""),
        (r"\bnaozaj\b", ""),
        (r"\bskutočne\b", ""),
        (r"\bhodnota je NULL\b", "je NULL"),
        (r"\bhodnota nie je NULL\b", "nie je NULL"),
        (r"\bpreddefinovanou hodnotou\b", "predvolenou hodnotou"),
        (r"^Ukážme si príklad\.$", "Príklad."),
    ]
    for pattern, replacement in rules:
        candidate = cleanup_spacing(re.sub(pattern, replacement, out, flags=re.IGNORECASE))
        if len(candidate) < len(out):
            out = candidate
    return cleanup_spacing(out)


def _best_split(text: str) -> dict[str, str] | None:
    clean = normalize_text(text)
    if len(clean) < 50:
        return None
    candidates: list[tuple[str, str, int]] = []
    for marker in SPLIT_MARKERS:
        start = 0
        while True:
            idx = clean.find(marker, start)
            if idx == -1:
                break
            left_end = idx + (1 if marker in {":", ";", ","} else 0)
            left = cleanup_spacing(clean[:left_end])
            right = cleanup_spacing(clean[idx + len(marker) :])
            start = idx + len(marker)
            if len(left) < 20 or len(right) < 20:
                continue
            candidates.append((left, right, abs(len(left) - len(right))))
    if not candidates:
        return None
    left, right, _ = min(candidates, key=lambda item: item[2])
    return {"left": left, "right": right}


class Level12Manager:
    def process_segment(self, seg: dict[str, Any]) -> dict[str, Any]:
        result = dict(seg)
        original_text = normalize_text(str(result.get("tts_input") or result.get("text") or ""))
        source_text = normalize_text(str(result.get("text_src") or result.get("source_text") or ""))
        slot = slot_duration(result)
        seg_type = classify_segment_type(source_text or original_text)
        target = target_cps_for_type(seg_type)
        cps_before = estimate_cps(original_text, slot)
        notes: list[str] = []
        candidate_text = original_text

        template = source_guided_template(source_text) if source_text else None
        if template:
            template_text = cleanup_spacing(guard_sql_text(template, source_text=source_text, fallback_text=original_text))
            template_cps = estimate_cps(template_text, slot)
            if template_cps <= cps_before + 0.25 or not protected_sql_terms_in(original_text).issuperset(
                protected_sql_terms_in(template_text)
            ):
                candidate_text = template_text
                notes.append("source_template")

        shortened = _shorten_text(candidate_text)
        if estimate_cps(shortened, slot) + 0.05 < estimate_cps(candidate_text, slot):
            candidate_text = shortened
            notes.append("shorten")

        candidate_text = cleanup_spacing(guard_sql_text(candidate_text, source_text=source_text, fallback_text=original_text))
        cps_after = estimate_cps(candidate_text, slot)
        split = _best_split(candidate_text) if cps_after > target + 1.0 else None
        if split:
            notes.append("suggest_split")

        result["text"] = candidate_text
        result["tts_input"] = candidate_text
        result["segment_type"] = result.get("segment_type") or seg_type
        result["level12_text"] = candidate_text
        result["level12_changed"] = candidate_text != original_text
        result["level12_segment_type"] = seg_type
        result["level12_target_cps"] = round(target, 2)
        result["level12_cps_before"] = round(cps_before, 2)
        result["level12_cps_after"] = round(cps_after, 2)
        result["level12_notes"] = notes
        if split:
            result["level12_suggested_split"] = split
        elif "level12_suggested_split" in result:
            result.pop("level12_suggested_split", None)
        return result

    def process_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.process_segment(seg) for seg in segments]


def summarize_level12_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "segments": len(segments),
        "changed_segments": 0,
        "suggested_splits": 0,
        "over_target_before": 0,
        "over_target_after": 0,
        "avg_cps_before": 0.0,
        "avg_cps_after": 0.0,
        "segment_types": {},
    }
    if not segments:
        return summary

    cps_before_values = [float(seg.get("level12_cps_before", 0.0) or 0.0) for seg in segments]
    cps_after_values = [float(seg.get("level12_cps_after", 0.0) or 0.0) for seg in segments]
    summary["changed_segments"] = sum(1 for seg in segments if seg.get("level12_changed"))
    summary["suggested_splits"] = sum(1 for seg in segments if seg.get("level12_suggested_split"))
    summary["over_target_before"] = sum(
        1
        for seg in segments
        if float(seg.get("level12_cps_before", 0.0) or 0.0) > float(seg.get("level12_target_cps", TARGET_CPS_DEFAULT))
    )
    summary["over_target_after"] = sum(
        1
        for seg in segments
        if float(seg.get("level12_cps_after", 0.0) or 0.0) > float(seg.get("level12_target_cps", TARGET_CPS_DEFAULT))
    )
    summary["avg_cps_before"] = round(sum(cps_before_values) / len(cps_before_values), 3)
    summary["avg_cps_after"] = round(sum(cps_after_values) / len(cps_after_values), 3)
    summary["segment_types"] = dict(
        Counter(str(seg.get("level12_segment_type") or seg.get("segment_type") or "default") for seg in segments)
    )
    return summary
