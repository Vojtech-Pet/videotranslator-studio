from __future__ import annotations

import concurrent.futures
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ATTEMPT_TIMEOUT_S: int = 45   # max sekúnd na jeden attempt (LLM + eval)


def _make_timeout_fallback(base: dict[str, Any]) -> dict[str, Any]:
    """Fallback výsledok keď L11 attempt timeoutuje — zachová originálny text."""
    return {
        "ok": False,
        "final_score": 0.0,
        "status": "timeout",
        "tts_input": base.get("tts_input") or base.get("level3_text") or base.get("text", ""),
        "candidate_name": "timeout_fallback",
        "prosody_name": "default",
        "delta": 0.0,
        "estimated_duration": 0.0,
        "evaluated_variants": 0,
    }

from src.audio_agent import AudioAgent
from src.eval_agent import EvalAgent
from src.feature_extractor import extract_features
from src.memory_agent import MemoryAgent
from src.policy_agent import PolicyAgent
from src.policy_registry import PolicyRegistry
from src.prompt_agent import PromptAgent
from src.segment_classifier import classify_segment
from src.text_agent import TextAgent


class Level11Manager:
    def __init__(self, memory_path: str | Path, registry_path: str | Path, *, max_retries: int = 1):
        self.memory_path = Path(memory_path).expanduser().resolve()
        self.registry_path = Path(registry_path).expanduser().resolve()
        self.max_retries = max_retries
        self.prompt_agent = PromptAgent(self.memory_path, self.registry_path)
        self.text_agent = TextAgent()
        self.audio_agent = AudioAgent()
        self.eval_agent = EvalAgent()
        self.policy_agent = PolicyAgent(self.memory_path, self.registry_path)
        self.memory_agent = MemoryAgent(self.memory_path)
        self.registry = PolicyRegistry(self.registry_path)

    def _is_complex(self, seg: dict[str, Any], features: dict[str, Any]) -> bool:
        return bool(
            features.get("has_sql_terms")
            or float(features.get("cps", 0.0) or 0.0) > 16.0
            or str(seg.get("segment_type") or "") in {"technical_definition", "technical_explanation"}
            or not bool(seg.get("accepted", True))
        )

    def _run_single_attempt(
        self,
        base: dict[str, Any],
        segment_type: str,
        retry_mode: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        """Jeden attempt — volá LLM + eval. Spúšťa sa v thread s timeoutom."""
        prompt_bundle = self.prompt_agent.get_prompt(base, segment_type=segment_type, retry_mode=retry_mode)
        candidates = self.text_agent.generate(base, prompt_bundle, force_mode=retry_mode)
        audio_variants = self.audio_agent.synthesize(base, candidates)
        best = self.eval_agent.select_best(base, audio_variants)
        return best, prompt_bundle, len(candidates)

    def process_segment(self, seg: dict[str, Any]) -> dict[str, Any]:
        base = dict(seg)
        features = extract_features(base)
        segment_type = str(base.get("segment_type") or classify_segment(features, str(base.get("tts_input") or "")))
        features["segment_type"] = segment_type
        base["segment_type"] = segment_type
        base["orchestrator_complex"] = self._is_complex(base, features)
        base["policy_name"] = self.registry.get_active_policy_name()

        attempts: list[dict[str, Any]] = []
        retry_mode: str | None = None
        best_result: dict[str, Any] | None = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _executor:
            for attempt_index in range(self.max_retries + 1):
                _future = _executor.submit(self._run_single_attempt, base, segment_type, retry_mode)
                try:
                    best, prompt_bundle, candidate_count = _future.result(timeout=_ATTEMPT_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "[L11] Attempt %d TIMEOUT (%ds) — seg '%s...' → fallback",
                        attempt_index, _ATTEMPT_TIMEOUT_S, str(base.get("tts_input", ""))[:40],
                    )
                    if best_result is None:
                        best_result = _make_timeout_fallback(base)
                    break
                except Exception as _err:
                    logger.error("[L11] Attempt %d ERROR: %s", attempt_index, _err)
                    if best_result is None:
                        best_result = _make_timeout_fallback(base)
                    break

                best["attempt_index"] = attempt_index
                best["prompt_name"] = prompt_bundle.get("prompt_name")
                best["prompt_text"] = prompt_bundle.get("prompt_text")
                best["policy_name"] = prompt_bundle.get("policy_name")
                best["candidate_count"] = candidate_count
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "retry_mode": retry_mode,
                        "prompt_name": prompt_bundle.get("prompt_name"),
                        "candidate_count": candidate_count,
                        "best_candidate_name": best.get("candidate_name"),
                        "best_status": best.get("status"),
                        "best_final_score": best.get("final_score"),
                    }
                )

                if best_result is None or float(best.get("final_score", 0.0) or 0.0) > float(best_result.get("final_score", 0.0) or 0.0):
                    best_result = best

                if best.get("ok") or attempt_index >= self.max_retries:
                    break

                retry_plan = self.policy_agent.suggest(base, best)
                retry_mode = str(retry_plan.get("retry_mode") or "balanced")

        assert best_result is not None
        result = dict(base)
        result["attempt_count"] = len(attempts)
        result["orchestrator_attempts"] = attempts
        result["best_candidate_name"] = best_result.get("candidate_name")
        result["best_prosody_name"] = best_result.get("prosody_name")
        result["prompt_name"] = best_result.get("prompt_name")
        result["prompt_text"] = best_result.get("prompt_text")
        result["retry_mode"] = retry_mode or result.get("retry_mode")
        result["multi_agent_score"] = best_result.get("final_score")
        result["final_score"] = max(float(result.get("final_score", 0.0) or 0.0), float(best_result.get("final_score", 0.0) or 0.0))
        result["duration_status"] = best_result.get("status")
        result["delta"] = best_result.get("delta")
        result["ok"] = bool(best_result.get("ok"))
        result["selected_tts_input"] = best_result.get("tts_input")
        result["multi_agent_selected"] = {
            "candidate_name": best_result.get("candidate_name"),
            "prosody_name": best_result.get("prosody_name"),
            "prompt_name": best_result.get("prompt_name"),
            "estimated_duration": best_result.get("estimated_duration"),
            "status": best_result.get("status"),
            "final_score": best_result.get("final_score"),
        }
        result["audio_variants_evaluated"] = int(best_result.get("evaluated_variants", 0) or 0)
        self.memory_agent.store(result)
        return result

    def process_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.process_segment(seg) for seg in segments]

    def finalize(self, summary: dict[str, Any] | None = None) -> None:
        self.memory_agent.finalize(summary)


def summarize_level11_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "segments": len(segments),
        "ok_segments": 0,
        "needs_retry": 0,
        "complex_segments": 0,
        "avg_multi_agent_score": 0.0,
        "prompt_names": {},
        "best_candidate_names": {},
        "best_prosody_names": {},
        "duration_statuses": {},
    }
    if not segments:
        return summary

    summary["ok_segments"] = sum(1 for seg in segments if seg.get("ok"))
    summary["needs_retry"] = len(segments) - summary["ok_segments"]
    summary["complex_segments"] = sum(1 for seg in segments if seg.get("orchestrator_complex"))
    summary["avg_multi_agent_score"] = round(
        sum(float(seg.get("multi_agent_score", 0.0) or 0.0) for seg in segments) / max(1, len(segments)),
        3,
    )
    summary["prompt_names"] = dict(Counter(str(seg.get("prompt_name") or "balanced") for seg in segments))
    summary["best_candidate_names"] = dict(Counter(str(seg.get("best_candidate_name") or "unknown") for seg in segments))
    summary["best_prosody_names"] = dict(Counter(str(seg.get("best_prosody_name") or "unknown") for seg in segments))
    summary["duration_statuses"] = dict(Counter(str(seg.get("duration_status") or "unknown") for seg in segments))
    return summary
