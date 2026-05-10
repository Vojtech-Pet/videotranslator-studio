from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from src.audio_agent import AudioAgent
from src.coherence_checker import spoken_naturalness_score
from src.terminology_memory import TerminologyMemory
from src.text_agent import TextAgent
from src.utils_text import normalize_text


TARGET_CPS = 16.0
HARD_CPS_LIMIT = 17.0
DEFAULT_ACCEPTANCE_THRESHOLD = 0.82
BEST_OF_N_IMPROVEMENT_MARGIN = 0.04
BAD_ASR_FORMS = ("ajznul", "koalesk", "esikjuel", "esikjul", "iz not null", "izn ot null")
TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ž_]+")


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(_fold(text)) if len(token) > 1}


def _heuristic_meaning_score(reference_text: str, candidate_text: str) -> float:
    reference_tokens = _tokenize(reference_text)
    candidate_tokens = _tokenize(candidate_text)
    if not reference_tokens:
        return 1.0 if normalize_text(candidate_text) else 0.0
    if not candidate_tokens:
        return 0.0
    common = len(reference_tokens & candidate_tokens)
    recall = common / len(reference_tokens)
    precision = common / len(candidate_tokens)
    return round(max(0.0, min(1.0, (0.65 * recall) + (0.35 * precision))), 3)


def asr_backcheck_score(
    expected_text: str,
    asr_text: str,
    *,
    terminology: TerminologyMemory | None = None,
) -> float:
    term_mem = terminology or TerminologyMemory()
    expected = term_mem.normalize_terms(normalize_text(expected_text))
    actual = term_mem.normalize_terms(normalize_text(asr_text))
    if not expected:
        return 0.0
    if not actual:
        return 0.0
    if _fold(expected) == _fold(actual):
        return 1.0

    expected_words = _tokenize(expected)
    actual_words = _tokenize(actual)
    overlap = len(expected_words & actual_words) / max(1, len(expected_words))

    expected_terms = set(term_mem.extract_terms_present(expected))
    actual_terms = set(term_mem.extract_terms_present(actual))
    if expected_terms:
        term_coverage = len(expected_terms & actual_terms) / len(expected_terms)
    else:
        term_coverage = 1.0

    penalty = 0.0
    actual_folded = _fold(actual)
    for bad in BAD_ASR_FORMS:
        if bad in actual_folded:
            penalty += 0.15

    score = (0.65 * overlap) + (0.35 * term_coverage) - penalty
    return round(max(0.0, min(1.0, score)), 3)


def _cps_score(text: str, slot: float) -> tuple[float, float]:
    if slot <= 0.0:
        return 0.2, 999.0
    cps = len(normalize_text(text)) / max(slot, 0.01)
    if cps <= TARGET_CPS:
        score = 1.0
    elif cps <= HARD_CPS_LIMIT:
        score = 0.8
    elif cps <= HARD_CPS_LIMIT + 1.0:
        score = 0.5
    else:
        score = 0.2
    return round(score, 3), round(cps, 3)


def _duration_score(delta: float | None, status: str | None) -> float:
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


def _base_reference_text(seg: dict[str, Any]) -> str:
    for key in ("level3_text", "level2_text", "level1_text", "tts_input", "text", "translated"):
        value = normalize_text(str(seg.get(key) or ""))
        if value:
            return value
    return ""


def _source_text(seg: dict[str, Any]) -> str:
    for key in ("source_text", "text_src", "text_original"):
        value = normalize_text(str(seg.get(key) or ""))
        if value:
            return value
    return _base_reference_text(seg)


def _slot(seg: dict[str, Any]) -> float:
    try:
        return max(0.01, float(seg.get("slot") or 0.0))
    except (TypeError, ValueError):
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        return max(0.01, end - start)


def _rendered_variant(seg: dict[str, Any]) -> dict[str, Any]:
    audit = dict(seg.get("level4_audit") or {})
    delta = audit.get("delta", seg.get("delta"))
    try:
        delta_value = None if delta is None else float(delta)
    except (TypeError, ValueError):
        delta_value = None
    status = str(audit.get("status") or seg.get("duration_status") or "pending_audio")
    asr_text = normalize_text(
        str(
            seg.get("asr_backcheck")
            or seg.get("tts_debug", {}).get("asr_backcheck")
            or seg.get("level5_asr_backcheck")
            or ""
        )
    )
    return {
        "source": "rendered",
        "candidate_name": "rendered_current",
        "prompt_name": str(seg.get("prompt_name") or seg.get("selected_prompt") or "rendered"),
        "prompt_text": None,
        "prosody_name": str(seg.get("tts_engine") or "rendered"),
        "tts_input": normalize_text(str(seg.get("tts_input") or seg.get("text") or "")),
        "slot": _slot(seg),
        "delta": delta_value,
        "status": status,
        "estimated_duration": audit.get("wav_duration"),
        "asr_backcheck": asr_text,
        "audio_source": "rendered",
    }


def _choose_generation_mode(seg: dict[str, Any], rendered: dict[str, Any]) -> str:
    text = str(rendered.get("tts_input") or "")
    slot = float(rendered.get("slot") or _slot(seg))
    _, cps = _cps_score(text, slot)
    if float(seg.get("term_score", 1.0) or 1.0) < 0.85:
        return "locked_terms"
    if rendered.get("status") == "too_long" or cps > HARD_CPS_LIMIT:
        return "aggressive_shorten"
    if float(seg.get("asr_match_score", 1.0) or 1.0) < 0.75:
        return "pronunciation_safe"
    if spoken_naturalness_score(text) < 0.7:
        return "dub_friendly"
    return "balanced"


def _choose_retry_mode(metrics: dict[str, Any]) -> str | None:
    if float(metrics.get("term_score", 1.0) or 1.0) < 0.8 or bool(metrics.get("watchlist_alert")):
        return "locked_terms"
    if float(metrics.get("delta", 0.0) or 0.0) > 0.25 or str(metrics.get("duration_status") or "") == "too_long":
        return "aggressive_shorten"
    if float(metrics.get("asr_match_score", 1.0) or 1.0) < 0.75:
        return "pronunciation_safe"
    if float(metrics.get("spoken_score", 1.0) or 1.0) < 0.7:
        return "dub_friendly"
    return None


def _score_variant(
    seg: dict[str, Any],
    variant: dict[str, Any],
    *,
    terminology: TerminologyMemory,
    acceptance_threshold: float,
) -> dict[str, Any]:
    reference_text = _base_reference_text(seg)
    source_text = _source_text(seg)
    text = terminology.normalize_terms(normalize_text(str(variant.get("tts_input") or "")))
    slot = float(variant.get("slot") or _slot(seg))
    cps_score_value, cps_value = _cps_score(text, slot)
    delta = variant.get("delta")
    try:
        delta_value = None if delta is None else float(delta)
    except (TypeError, ValueError):
        delta_value = None
    duration_status = str(variant.get("status") or "unknown")
    duration_score_value = _duration_score(delta_value, duration_status)
    meaning_reference = reference_text or source_text
    meaning_score = _heuristic_meaning_score(meaning_reference, text)
    term_score = terminology.term_preservation_score(meaning_reference or text, text)
    spoken_score = spoken_naturalness_score(text)

    asr_text = normalize_text(str(variant.get("asr_backcheck") or ""))
    if asr_text:
        asr_match = asr_backcheck_score(text, asr_text, terminology=terminology)
    else:
        existing = variant.get("asr_match_score", seg.get("asr_match_score"))
        if existing is None:
            asr_match = 1.0
        else:
            try:
                asr_match = float(existing)
            except (TypeError, ValueError):
                asr_match = 1.0

    expected_terms = set(terminology.extract_terms_present(meaning_reference))
    actual_terms = set(terminology.extract_terms_present(text))
    watchlist_alert = bool(expected_terms and not expected_terms.issubset(actual_terms))
    if any(bad in _fold(asr_text) for bad in BAD_ASR_FORMS):
        watchlist_alert = True

    final_score = (
        0.30 * meaning_score
        + 0.20 * term_score
        + 0.15 * cps_score_value
        + 0.15 * duration_score_value
        + 0.10 * spoken_score
        + 0.10 * asr_match
    )

    scored = dict(variant)
    scored.update(
        {
            "tts_input": text,
            "meaning_score": round(meaning_score, 3),
            "term_score": round(term_score, 3),
            "cps_score": round(cps_score_value, 3),
            "duration_score": round(duration_score_value, 3),
            "spoken_score": round(spoken_score, 3),
            "asr_match_score": round(asr_match, 3),
            "final_score": round(final_score, 3),
            "final_chars_per_sec": round(cps_value, 3),
            "duration_status": duration_status,
            "delta": delta_value,
            "watchlist_alert": watchlist_alert,
        }
    )
    scored["accepted"] = bool(
        scored["final_score"] >= acceptance_threshold
        and duration_status == "ok"
        and scored["term_score"] >= 0.85
    )
    scored["retry_mode"] = None if scored["accepted"] else _choose_retry_mode(scored)
    return scored


class Level5Manager:
    def __init__(
        self,
        *,
        acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
        best_of_n_margin: float = BEST_OF_N_IMPROVEMENT_MARGIN,
    ) -> None:
        self.acceptance_threshold = acceptance_threshold
        self.best_of_n_margin = best_of_n_margin
        self.terminology = TerminologyMemory()
        self.text_agent = TextAgent()
        self.audio_agent = AudioAgent()

    def process_segment(self, seg: dict[str, Any]) -> dict[str, Any]:
        result = dict(seg)
        rendered_variant = _rendered_variant(result)
        prompt_name = _choose_generation_mode(result, rendered_variant)
        prompt_bundle = {
            "prompt_name": prompt_name,
            "prompt_text": prompt_name,
        }

        text_candidates = self.text_agent.generate(result, prompt_bundle, force_mode=prompt_name, max_candidates=3)
        audio_variants = self.audio_agent.synthesize(result, text_candidates)

        rendered_scored = _score_variant(
            result,
            rendered_variant,
            terminology=self.terminology,
            acceptance_threshold=self.acceptance_threshold,
        )

        scored_variants: list[dict[str, Any]] = []
        for idx, variant in enumerate(audio_variants, 1):
            variant_copy = dict(variant)
            variant_copy["source"] = "predicted"
            variant_copy["variant_index"] = idx
            variant_copy["audio_source"] = "predicted"
            variant_copy["asr_backcheck"] = normalize_text(str(variant_copy.get("tts_input") or ""))
            scored = _score_variant(
                result,
                variant_copy,
                terminology=self.terminology,
                acceptance_threshold=self.acceptance_threshold,
            )
            scored_variants.append(scored)

        best_predicted = max(scored_variants, key=lambda item: item["final_score"], default=None)
        selected = rendered_scored
        needs_rerender = False
        if best_predicted is not None:
            should_switch = (
                best_predicted["final_score"] >= rendered_scored["final_score"] + self.best_of_n_margin
                and (
                    not rendered_scored["accepted"]
                    or best_predicted["term_score"] > rendered_scored["term_score"]
                    or best_predicted["duration_score"] > rendered_scored["duration_score"]
                )
            )
            if should_switch:
                selected = best_predicted
                needs_rerender = True

        level5_retry_required = needs_rerender or not rendered_scored["accepted"]
        retry_mode = selected.get("retry_mode")
        if not retry_mode and level5_retry_required:
            retry_mode = _choose_retry_mode(selected)

        result["level5_rendered_variant"] = rendered_scored
        result["level5_candidate_variants"] = scored_variants
        result["level5_text_candidates"] = text_candidates
        result["level5_audio_variant_count"] = len(audio_variants)
        result["candidates"] = len(text_candidates)
        result["chosen_candidate"] = int(selected.get("variant_index", 0) or 0)
        result["chosen_candidate_name"] = str(selected.get("candidate_name") or "rendered_current")
        result["chosen_candidate_source"] = str(selected.get("source") or "rendered")
        result["best_candidate"] = {
            "candidate_name": result["chosen_candidate_name"],
            "source": result["chosen_candidate_source"],
            "prompt_name": selected.get("prompt_name"),
            "prosody_name": selected.get("prosody_name"),
            "final_score": selected.get("final_score"),
        }
        result["level5_retry_required"] = level5_retry_required
        result["level5_selected_rendered"] = bool(result["chosen_candidate_source"] == "rendered")
        result["level5_best_predicted_score"] = None if best_predicted is None else best_predicted["final_score"]
        result["level5_rendered_score"] = rendered_scored["final_score"]
        result["retry_count"] = int(level5_retry_required)
        result["retry_mode"] = retry_mode
        result["prompt_name"] = str(selected.get("prompt_name") or prompt_name)
        result["selected_prompt"] = str(selected.get("prompt_name") or prompt_name)

        result["tts_input"] = str(selected.get("tts_input") or result.get("tts_input") or "")
        result["meaning_score"] = selected["meaning_score"]
        result["term_score"] = selected["term_score"]
        result["cps_score"] = selected["cps_score"]
        result["duration_score"] = selected["duration_score"]
        result["spoken_score"] = selected["spoken_score"]
        result["asr_match_score"] = selected["asr_match_score"]
        result["final_score"] = selected["final_score"]
        result["final_chars_per_sec"] = selected["final_chars_per_sec"]
        result["level5_scores"] = {
            "meaning_score": selected["meaning_score"],
            "term_score": selected["term_score"],
            "cps_score": selected["cps_score"],
            "duration_score": selected["duration_score"],
            "spoken_score": selected["spoken_score"],
            "asr_match_score": selected["asr_match_score"],
            "final_score": selected["final_score"],
        }
        result["level5_asr_backcheck"] = selected.get("asr_backcheck") or rendered_scored.get("asr_backcheck") or ""
        result["accepted"] = bool(
            selected["final_score"] >= self.acceptance_threshold
            and not needs_rerender
            and rendered_scored["duration_status"] == "ok"
        )
        return result

    def process_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.process_segment(seg) for seg in segments]


def summarize_level5_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "segments": len(segments),
        "accepted": 0,
        "needs_retry": 0,
        "rendered_wins": 0,
        "candidate_wins": 0,
        "avg_final_score": 0.0,
        "avg_asr_match_score": 0.0,
        "prompt_names": {},
        "retry_modes": {},
        "watchlist_alerts": 0,
    }
    if not segments:
        return summary

    summary["accepted"] = sum(1 for seg in segments if seg.get("accepted"))
    summary["needs_retry"] = len(segments) - summary["accepted"]
    summary["rendered_wins"] = sum(1 for seg in segments if seg.get("chosen_candidate_source") == "rendered")
    summary["candidate_wins"] = sum(1 for seg in segments if seg.get("chosen_candidate_source") == "predicted")
    summary["avg_final_score"] = round(
        sum(float(seg.get("final_score", 0.0) or 0.0) for seg in segments) / max(1, len(segments)),
        3,
    )
    summary["avg_asr_match_score"] = round(
        sum(float(seg.get("asr_match_score", 0.0) or 0.0) for seg in segments) / max(1, len(segments)),
        3,
    )
    summary["prompt_names"] = dict(Counter(str(seg.get("prompt_name") or "balanced") for seg in segments))
    summary["retry_modes"] = dict(Counter(str(seg.get("retry_mode") or "accepted") for seg in segments))
    summary["watchlist_alerts"] = sum(
        1
        for seg in segments
        if any(variant.get("watchlist_alert") for variant in seg.get("level5_candidate_variants", []))
        or bool(seg.get("level5_rendered_variant", {}).get("watchlist_alert"))
    )
    return summary
