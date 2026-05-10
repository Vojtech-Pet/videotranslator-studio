from __future__ import annotations

import re
from collections import Counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from sql_logic_guard import SQL_PHONETIC_PATTERNS, guard_sql_text, protected_sql_terms_in, source_guided_template
from src.coherence_checker import spoken_naturalness_score
from src.level12_manager import classify_segment_type, estimate_cps, slot_duration, target_cps_for_type
from src.utils_text import cleanup_spacing, normalize_text

LOCKED_TERMS = (
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


def _ollama_probe_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path or ""
    if path.endswith("/api/generate"):
        path = path[: -len("/api/generate")] + "/api/tags"
    elif not path.endswith("/api/tags"):
        path = path.rstrip("/") + "/api/tags"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _clean_candidate(text: str, *, source_text: str, fallback_text: str) -> str:
    return cleanup_spacing(guard_sql_text(text, source_text=source_text, fallback_text=fallback_text))


def _contains_suspicious_sql(text: str) -> bool:
    return any(re.search(pattern, text or "", flags=re.IGNORECASE) for pattern in SQL_PHONETIC_PATTERNS)


def _prompt_pass1(source_text: str) -> str:
    return f"""
Prelož vetu z angličtiny do prirodzenej technickej slovenčiny.

Pravidlá:
- Zachovaj význam.
- Zachovaj presne tieto výrazy:
  SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, NULLIF, TRUE, FALSE, BOOLEAN
- Nepoužívaj fonetické tvary ani halucinácie.
- Vráť iba finálnu vetu.

Text:
"{source_text}"
""".strip()


def _prompt_pass2(slovak_text: str) -> str:
    return f"""
Prepíš vetu do prirodzenej slovenčiny vhodnej pre technický voiceover.

Pravidlá:
- Zachovaj význam.
- Zachovaj technické výrazy presne.
- Uprednostni krátke a jasné formulácie.
- Nepoužívaj doslovné anglické konštrukcie.
- Vráť iba finálnu vetu.

Text:
"{slovak_text}"
""".strip()


def _prompt_pass3(slovak_text: str, *, slot: float, seg_type: str) -> str:
    return f"""
Prepíš vetu tak, aby sa prirodzene zmestila do časového slotu.

Pravidlá:
- Zachovaj význam.
- Zachovaj presne:
  SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, NULLIF, TRUE, FALSE, BOOLEAN
- Skráť vetu iba ak je to potrebné.
- Musí byť prirodzená na hlasné čítanie.
- Vráť iba finálnu vetu.

Typ segmentu: {seg_type}
Slot: {slot:.2f} sekundy

Text:
"{slovak_text}"
""".strip()


def _candidate_score(text: str, *, source_text: str, current_text: str, slot: float, target_cps: float) -> float:
    clean = normalize_text(text)
    if not clean:
        return -999.0

    expected_terms = protected_sql_terms_in(source_text or current_text)
    actual_terms = protected_sql_terms_in(clean)
    if expected_terms:
        term_score = len(expected_terms & actual_terms) / len(expected_terms)
    else:
        term_score = 1.0

    cps_value = estimate_cps(clean, slot)
    if cps_value <= target_cps:
        timing_score = 1.0
    elif cps_value <= target_cps + 1.0:
        timing_score = 0.75
    else:
        timing_score = 0.35

    spoken_score = spoken_naturalness_score(clean)
    suspicious_penalty = 0.55 if _contains_suspicious_sql(clean) else 0.0
    too_short_penalty = 0.20 if len(clean) < max(12, int(len(current_text) * 0.35)) else 0.0
    return round((0.4 * term_score) + (0.35 * timing_score) + (0.25 * spoken_score) - suspicious_penalty - too_short_penalty, 4)


class Level13Manager:
    def __init__(
        self,
        *,
        provider: str = "llama_cpp",
        ollama_url: str = "http://localhost:11434/api/generate",
        ollama_model: str = "gemma3:27b",
        ollama_timeout: int = 120,
        max_ollama_segments: int = 24,
        verbose: bool = False,
    ):
        self.provider = provider
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.ollama_timeout = ollama_timeout
        self.max_ollama_segments = max(0, int(max_ollama_segments))
        self.verbose = verbose
        self._ollama_checked = False
        self._ollama_available = False
        self._probe_error = ""
        self._ollama_processed = 0

    def _ensure_ollama(self) -> bool:
        if self.provider != "ollama":
            return False
        if self._ollama_checked:
            return self._ollama_available
        self._ollama_checked = True
        try:
            response = requests.get(_ollama_probe_url(self.ollama_url), timeout=min(self.ollama_timeout, 5))
            response.raise_for_status()
            self._ollama_available = True
        except Exception as exc:
            self._ollama_available = False
            self._probe_error = str(exc)
        return self._ollama_available

    def _generate(self, prompt: str) -> str:
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.ollama_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["response"]).strip()

    def _should_process(self, seg: dict[str, Any], current_text: str, source_text: str, seg_type: str, target: float) -> bool:
        if not current_text:
            return False
        current_cps = estimate_cps(current_text, slot_duration(seg))
        if current_cps > target + 0.25:
            return True
        if seg.get("level12_suggested_split"):
            return True
        if _contains_suspicious_sql(current_text):
            return True
        if spoken_naturalness_score(current_text) < 0.82:
            return True
        if source_guided_template(source_text) and current_cps > target:
            return True
        if seg_type in {"definition", "example", "summary"} and len(current_text) >= 35:
            return current_cps > target or spoken_naturalness_score(current_text) < 0.9
        return False

    def process_segment(self, seg: dict[str, Any]) -> dict[str, Any]:
        result = dict(seg)
        source_text = normalize_text(str(result.get("text_src") or result.get("source_text") or ""))
        current_text = normalize_text(str(result.get("tts_input") or result.get("text") or ""))
        slot = slot_duration(result)
        seg_type = str(result.get("level12_segment_type") or result.get("segment_type") or classify_segment_type(source_text or current_text))
        target = float(result.get("level12_target_cps") or target_cps_for_type(seg_type))

        result["level13_provider"] = "skipped"
        result["level13_changed"] = False
        result["level13_segment_type"] = seg_type
        result["level13_target_cps"] = round(target, 2)
        result["level13_cps_before"] = round(estimate_cps(current_text, slot), 2)

        if not self._should_process(result, current_text, source_text, seg_type, target):
            result["level13_tts_input"] = current_text
            result["level13_cps_after"] = result["level13_cps_before"]
            return result

        template = source_guided_template(source_text) if source_text else None
        candidates: dict[str, str] = {"current": current_text}
        if template:
            candidates["template"] = _clean_candidate(template, source_text=source_text, fallback_text=current_text)

        ollama_used = False
        if self._ensure_ollama() and source_text and self._ollama_processed < self.max_ollama_segments:
            try:
                pass1 = _clean_candidate(self._generate(_prompt_pass1(source_text)), source_text=source_text, fallback_text=current_text)
                candidates["llm_pass1"] = pass1
                pass2 = _clean_candidate(self._generate(_prompt_pass2(pass1)), source_text=source_text, fallback_text=current_text)
                candidates["llm_pass2"] = pass2
                if estimate_cps(pass2, slot) > target or result.get("level12_suggested_split"):
                    pass3 = _clean_candidate(
                        self._generate(_prompt_pass3(pass2, slot=slot, seg_type=seg_type)),
                        source_text=source_text,
                        fallback_text=current_text,
                    )
                else:
                    pass3 = pass2
                candidates["llm_pass3"] = pass3
                ollama_used = True
                self._ollama_processed += 1
            except Exception as exc:
                result["level13_error"] = str(exc)
        elif self.provider == "ollama" and self.verbose and self._probe_error:
            result["level13_error"] = self._probe_error

        scored = {
            name: _candidate_score(text, source_text=source_text, current_text=current_text, slot=slot, target_cps=target)
            for name, text in candidates.items()
        }
        best_name = max(scored, key=scored.get)
        best_text = candidates[best_name]

        current_score = scored.get("current", -999.0)
        if best_name != "current" and scored[best_name] <= current_score + 0.03:
            best_name = "current"
            best_text = current_text

        best_text = _clean_candidate(best_text, source_text=source_text, fallback_text=current_text)
        result["text"] = best_text
        result["tts_input"] = best_text
        result["level13_tts_input"] = best_text
        result["level13_best_candidate"] = best_name
        result["level13_candidates"] = {name: text for name, text in candidates.items() if text}
        result["level13_scores"] = scored
        result["level13_changed"] = best_text != current_text
        result["level13_provider"] = "ollama" if ollama_used else ("template" if best_name == "template" else "fallback")
        result["level13_cps_after"] = round(estimate_cps(best_text, slot), 2)
        return result

    def process_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        total = len(segments)
        for idx, seg in enumerate(segments, start=1):
            processed = self.process_segment(seg)
            out.append(processed)
            if self.verbose:
                provider = str(processed.get("level13_provider") or "skipped")
                if provider != "skipped":
                    preview = str(processed.get("text") or "")[:90]
                    print(
                        f"[LEVEL13] seg {idx}/{total} provider={provider} "
                        f"cps={processed.get('level13_cps_before', 0.0)}→{processed.get('level13_cps_after', 0.0)} "
                        f"text='{preview}'",
                        flush=True,
                    )
        return out


def summarize_level13_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "segments": len(segments),
        "changed_segments": 0,
        "providers": {},
        "best_candidates": {},
        "avg_cps_before": 0.0,
        "avg_cps_after": 0.0,
        "errors": 0,
    }
    if not segments:
        return summary

    cps_before = [float(seg.get("level13_cps_before", 0.0) or 0.0) for seg in segments]
    cps_after = [float(seg.get("level13_cps_after", seg.get("level13_cps_before", 0.0)) or 0.0) for seg in segments]
    summary["changed_segments"] = sum(1 for seg in segments if seg.get("level13_changed"))
    summary["providers"] = dict(Counter(str(seg.get("level13_provider") or "skipped") for seg in segments))
    summary["best_candidates"] = dict(Counter(str(seg.get("level13_best_candidate") or "current") for seg in segments))
    summary["avg_cps_before"] = round(sum(cps_before) / len(cps_before), 3)
    summary["avg_cps_after"] = round(sum(cps_after) / len(cps_after), 3)
    summary["errors"] = sum(1 for seg in segments if seg.get("level13_error"))
    return summary
