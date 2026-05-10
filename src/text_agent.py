from __future__ import annotations

import re
from typing import Any

from src.level4_audio_feedback import find_split_index, tighten_text_for_retry
from src.utils_sql import contains_sql_terms, fix_sql_terms
from src.utils_text import cleanup_spacing, normalize_text


FILLER_PATTERNS = (
    r"\bv poriadku priatelia\b",
    r"\btak toto je to, co\b",
    r"\bgo a\b",
    r"\bdeep dive do\b",
)


def _remove_fillers(text: str) -> str:
    out = text
    for pattern in FILLER_PATTERNS:
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bdeep dive\b", "pozrieme sa", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnas stol\b", "tabulka", out, flags=re.IGNORECASE)
    return cleanup_spacing(out)


def _candidate_modes(prompt_name: str) -> list[str]:
    ordered = [prompt_name, "balanced", "dub_friendly", "aggressive_shorten", "locked_terms"]
    seen: set[str] = set()
    result: list[str] = []
    for item in ordered:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class TextAgent:
    def generate(
        self,
        seg: dict[str, Any],
        prompt_bundle: dict[str, str],
        *,
        force_mode: str | None = None,
        max_candidates: int = 3,
    ) -> list[dict[str, Any]]:
        base_text = normalize_text(str(seg.get("tts_input") or seg.get("level3_text") or seg.get("text") or ""))
        base_text = cleanup_spacing(fix_sql_terms(base_text))
        slot = float(seg.get("slot") or max(0.01, float(seg.get("end", 0.0) or 0.0) - float(seg.get("start", 0.0) or 0.0)))
        prompt_name = force_mode or prompt_bundle.get("prompt_name") or "balanced"

        candidates: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        for mode in _candidate_modes(str(prompt_name)):
            candidate_text = base_text
            if mode == "dub_friendly":
                candidate_text = _remove_fillers(base_text)
            elif mode == "aggressive_shorten":
                candidate_text = cleanup_spacing(
                    fix_sql_terms(
                        tighten_text_for_retry(
                            _remove_fillers(base_text),
                            slot,
                            preserve_sql_terms=contains_sql_terms(base_text),
                        )
                    )
                )
            elif mode == "locked_terms":
                candidate_text = cleanup_spacing(fix_sql_terms(base_text))

            if contains_sql_terms(base_text):
                candidate_text = cleanup_spacing(fix_sql_terms(candidate_text))

            split_index = find_split_index(candidate_text)
            split_suggestion = None
            if split_index is not None:
                split_suggestion = [
                    cleanup_spacing(candidate_text[:split_index].strip()),
                    cleanup_spacing(candidate_text[split_index:].strip()),
                ]

            normalized_candidate = normalize_text(candidate_text)
            if not normalized_candidate or normalized_candidate in seen_texts:
                continue
            seen_texts.add(normalized_candidate)
            candidates.append(
                {
                    "candidate_name": mode,
                    "tts_input": normalized_candidate,
                    "prompt_name": prompt_bundle.get("prompt_name"),
                    "prompt_text": prompt_bundle.get("prompt_text"),
                    "split_suggestion": split_suggestion,
                }
            )
            if len(candidates) >= max_candidates:
                break
        return candidates
