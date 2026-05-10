from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.coherence_checker import coherence_score, spoken_naturalness_score
from src.failure_memory import FailureMemory
from src.persona_controller import get_persona, persona_stability_score
from src.scene_planner import make_scene_plan
from src.terminology_memory import TerminologyMemory
from src.utils_text import normalize_text


STOPWORDS = {
    "a",
    "aj",
    "ale",
    "ako",
    "by",
    "co",
    "ho",
    "ich",
    "je",
    "ju",
    "na",
    "sa",
    "si",
    "sme",
    "som",
    "so",
    "su",
    "tak",
    "teraz",
    "to",
    "toho",
    "tohoto",
    "u",
    "v",
    "vo",
    "z",
    "ze",
}


def duration_score_from_delta(delta: float | None, status: str | None = None) -> float:
    if status in {"missing_wav", "pending_audio"}:
        return 0.35
    if delta is None:
        return 0.5
    absolute_delta = abs(delta)
    if absolute_delta <= 0.20:
        return 1.0
    if absolute_delta <= 0.40:
        return 0.8
    if absolute_delta <= 0.70:
        return 0.5
    return 0.2


def cps_score(cps: float, target: float, hard_limit: float) -> float:
    if cps <= target:
        return 1.0
    if cps <= hard_limit:
        return 0.8
    if cps <= hard_limit + 1.0:
        return 0.5
    return 0.2


def weighted_final_score(seg: dict[str, Any]) -> float:
    return round(
        0.22 * float(seg.get("meaning_score", 1.0))
        + 0.15 * float(seg.get("term_score", 1.0))
        + 0.12 * float(seg.get("cps_score", 1.0))
        + 0.12 * float(seg.get("duration_score", 1.0))
        + 0.10 * float(seg.get("spoken_score", 1.0))
        + 0.10 * float(seg.get("asr_match_score", 1.0))
        + 0.09 * float(seg.get("coherence_score", 1.0))
        + 0.05 * float(seg.get("terminology_consistency_score", 1.0))
        + 0.05 * float(seg.get("persona_stability_score", 1.0)),
        3,
    )


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-zÀ-ž_]+", (text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def _heuristic_meaning_score(reference_text: str, candidate_text: str) -> float:
    reference_tokens = _tokenize(reference_text)
    candidate_tokens = _tokenize(candidate_text)
    if not reference_tokens:
        return 1.0 if candidate_text.strip() else 0.0
    if not candidate_tokens:
        return 0.0
    common = len(reference_tokens & candidate_tokens)
    recall = common / len(reference_tokens)
    precision = common / len(candidate_tokens)
    score = (0.65 * recall) + (0.35 * precision)
    return round(max(0.0, min(1.0, score)), 3)


def _best_candidate_text(seg: dict[str, Any]) -> str:
    for key in ("tts_input", "level3_text", "level2_text", "level1_text", "translated", "text_original", "text"):
        value = normalize_text(str(seg.get(key) or ""))
        if value:
            return value
    return ""


def _slot(seg: dict[str, Any]) -> float:
    if seg.get("slot") is not None:
        try:
            return max(0.01, float(seg["slot"]))
        except (TypeError, ValueError):
            pass
    audit = seg.get("level4_audit") or {}
    if audit.get("slot") is not None:
        try:
            return max(0.01, float(audit["slot"]))
        except (TypeError, ValueError):
            pass
    start = float(seg.get("start", 0.0) or 0.0)
    end = float(seg.get("end", 0.0) or 0.0)
    return max(0.01, end - start)


def _duration_fields(seg: dict[str, Any]) -> tuple[float | None, str]:
    audit = seg.get("level4_audit") or {}
    delta = audit.get("delta", seg.get("delta"))
    try:
        delta = None if delta is None else float(delta)
    except (TypeError, ValueError):
        delta = None
    status = str(audit.get("status") or seg.get("duration_status") or "").strip().lower()
    return delta, status


def _carry_score(seg: dict[str, Any], key: str) -> float | None:
    value = seg.get(key)
    if value is None:
        level5 = seg.get("level5_scores")
        if isinstance(level5, dict):
            value = level5.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_scene(
    segments: list[dict[str, Any]],
    *,
    acceptance_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    term_mem = TerminologyMemory()
    fail_mem = FailureMemory()
    scene_plan = make_scene_plan(segments)
    persona = get_persona(scene_plan.scene_type)

    prepared: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments, 1):
        seg_copy = dict(seg)
        source_text = normalize_text(str(seg_copy.get("text_src") or seg_copy.get("text") or ""))
        base_text = _best_candidate_text(seg_copy)
        normalized_text = term_mem.normalize_terms(base_text)
        seg_copy["seg"] = int(seg_copy.get("seg") or idx)
        seg_copy["tts_input"] = normalized_text
        seg_copy["source_text"] = source_text
        seg_copy["slot"] = round(_slot(seg_copy), 3)
        seg_copy["level6_terms_present"] = term_mem.extract_terms_present(normalized_text)
        prepared.append(seg_copy)

    for index, seg in enumerate(prepared):
        prev_seg = prepared[index - 1] if index > 0 else None
        next_seg = prepared[index + 1] if index < len(prepared) - 1 else None
        text = seg["tts_input"]
        reference_text = seg.get("level3_text") or seg.get("level2_text") or seg.get("level1_text") or seg.get("text") or ""

        delta, duration_status = _duration_fields(seg)
        slot = float(seg["slot"])
        final_cps = len(text) / slot if slot > 0 else 999.0

        meaning_score = _carry_score(seg, "meaning_score")
        if meaning_score is None:
            meaning_score = _heuristic_meaning_score(str(reference_text), text)

        term_score = _carry_score(seg, "term_score")
        if term_score is None:
            term_score = term_mem.term_preservation_score(str(reference_text), text)

        spoken_score = _carry_score(seg, "spoken_score")
        if spoken_score is None:
            spoken_score = spoken_naturalness_score(text)

        asr_match_score = _carry_score(seg, "asr_match_score")
        if asr_match_score is None:
            asr_match_score = 1.0

        seg["terminology_consistency_score"] = term_mem.term_consistency_score(text)
        seg["coherence_score"] = coherence_score(prev_seg, seg, next_seg)
        seg["persona_stability_score"] = persona_stability_score(text, persona)
        seg["final_chars_per_sec"] = round(final_cps, 2)
        seg["cps_score"] = cps_score(final_cps, scene_plan.target_cps, scene_plan.hard_cps_limit)
        seg["duration_score"] = duration_score_from_delta(delta, duration_status)
        seg["meaning_score"] = round(meaning_score, 3)
        seg["term_score"] = round(term_score, 3)
        seg["spoken_score"] = round(spoken_score, 3)
        seg["asr_match_score"] = round(asr_match_score, 3)
        seg["scene_type"] = scene_plan.scene_type
        seg["scene_base_style"] = scene_plan.base_style
        seg["scene_target_cps"] = scene_plan.target_cps
        seg["scene_hard_cps_limit"] = scene_plan.hard_cps_limit
        seg["persona_name"] = persona.name
        seg["persona_pace"] = persona.pace
        seg["persona_energy"] = persona.energy
        seg["persona_clarity"] = persona.clarity
        seg["duration_status"] = duration_status or "unknown"
        seg["delta"] = None if delta is None else round(delta, 3)
        seg["final_score"] = weighted_final_score(seg)

        has_audio = seg["duration_status"] not in {"pending_audio", "missing_wav"}
        seg["accepted"] = bool(has_audio and seg["final_score"] >= acceptance_threshold)
        if not seg["accepted"]:
            retry_mode = fail_mem.suggest_retry_mode(seg)
            seg["retry_mode"] = retry_mode
            fail_mem.log_event(
                seg_id=int(seg["seg"]),
                problem_type="low_final_score" if has_audio else "missing_audio",
                details={
                    "final_score": seg["final_score"],
                    "retry_mode": retry_mode,
                    "duration_status": seg["duration_status"],
                },
            )
        else:
            seg["retry_mode"] = None
    return prepared


def summarize_scene_evaluation(segments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "segments": len(segments),
        "accepted": 0,
        "needs_retry": 0,
        "avg_final_score": 0.0,
        "scene_type": "",
        "persona_name": "",
        "retry_modes": {},
        "duration_status_counts": {},
    }
    if not segments:
        return summary

    summary["accepted"] = sum(1 for seg in segments if seg.get("accepted"))
    summary["needs_retry"] = len(segments) - summary["accepted"]
    summary["avg_final_score"] = round(
        sum(float(seg.get("final_score", 0.0)) for seg in segments) / len(segments),
        3,
    )
    summary["scene_type"] = str(segments[0].get("scene_type") or "")
    summary["persona_name"] = str(segments[0].get("persona_name") or "")
    summary["retry_modes"] = dict(Counter(str(seg.get("retry_mode") or "accepted") for seg in segments))
    summary["duration_status_counts"] = dict(
        Counter(str(seg.get("duration_status") or "unknown") for seg in segments)
    )
    return summary
