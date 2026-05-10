from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.candidate_reranker import rerank_candidates
from src.feature_extractor import extract_features
from src.policy_learner import PolicyLearner
from src.project_memory import ProjectMemory
from src.prompt_router import choose_prompt
from src.segment_classifier import classify_segment
from src.threshold_tuner import tuned_cps_target, tuned_hard_limit


class Level9Manager:
    def __init__(self, memory_path: str | Path):
        self.memory = ProjectMemory(Path(memory_path).expanduser().resolve())
        self.memory.load()
        self.learner = PolicyLearner()
        self._bootstrap_from_history()

    def _bootstrap_from_history(self) -> None:
        history = list(self.memory.data.get("segment_history", []) or [])
        for record in history:
            features = extract_features(record)
            segment_type = str(record.get("segment_type") or "")
            if not segment_type or segment_type == "unknown":
                segment_type = classify_segment(features, str(record.get("tts_input") or ""))
            features["segment_type"] = segment_type
            self.learner.observe(
                features=features,
                decision={
                    "prompt_name": record.get("prompt_name"),
                    "retry_mode": record.get("retry_mode"),
                },
                outcome={
                    "final_score": record.get("final_score", 0.0),
                },
            )

    def _default_retry_mode(self, features: dict[str, Any], prompt_name: str, fallback: str | None = None) -> str:
        if prompt_name in {"locked_terms", "aggressive_shorten", "pronunciation_safe", "dub_friendly"}:
            return prompt_name
        if float(features.get("cps", 0.0) or 0.0) > 17.0:
            return "aggressive_shorten"
        if bool(features.get("has_sql_terms")) and float(features.get("asr_match_score", 1.0) or 1.0) < 0.9:
            return "locked_terms"
        if float(features.get("spoken_score", 1.0) or 1.0) < 0.75:
            return "dub_friendly"
        if fallback:
            return fallback
        return "balanced"

    def prepare_segment(self, seg: dict[str, Any]) -> dict[str, Any]:
        seg_copy = dict(seg)
        existing_prompt = str(seg_copy.get("prompt_name") or "").strip() or None
        existing_retry = str(seg_copy.get("retry_mode") or "").strip() or None

        features = extract_features(seg_copy)
        segment_type = classify_segment(features, str(seg_copy.get("tts_input", "") or ""))
        features["segment_type"] = segment_type

        learned_prompt = self.learner.best_prompt_for_segment_type(segment_type)
        prompt_name = choose_prompt(segment_type, features, learned_prompt=learned_prompt)
        cps_target = tuned_cps_target(segment_type)
        cps_hard_limit = tuned_hard_limit(segment_type)
        learned_retry = self.learner.best_retry_mode_for_segment_type(segment_type)
        retry_mode = self._default_retry_mode(features, prompt_name, fallback=learned_retry or existing_retry)

        seg_copy["features"] = features
        seg_copy["segment_type"] = segment_type
        seg_copy["previous_prompt_name"] = existing_prompt
        seg_copy["previous_retry_mode"] = existing_retry
        seg_copy["prompt_name"] = prompt_name
        seg_copy["retry_mode"] = retry_mode
        seg_copy["cps_target"] = cps_target
        seg_copy["cps_hard_limit"] = cps_hard_limit
        seg_copy["learned_prompt_preference"] = learned_prompt
        seg_copy["learned_retry_preference"] = learned_retry
        seg_copy["decision_source"] = "learned_history" if learned_prompt or learned_retry else "heuristic_router"
        return seg_copy

    def choose_best_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return rerank_candidates(candidates)

    def process_segment(self, seg: dict[str, Any]) -> dict[str, Any]:
        prepared = self.prepare_segment(seg)
        candidates = prepared.get("candidates") or prepared.get("candidate_variants")
        if isinstance(candidates, list) and len(candidates) > 1:
            best_candidate = self.choose_best_candidate(candidates)
            prepared["chosen_candidate"] = best_candidate
            prepared["chosen_candidate_index"] = int(best_candidate.get("candidate_rank_index", 0))
            prepared["candidate_rerank_score"] = best_candidate.get("candidate_rerank_score")
            prepared["candidate_pool_size"] = best_candidate.get("candidate_pool_size", len(candidates))
        else:
            prepared["candidate_pool_size"] = len(candidates) if isinstance(candidates, list) else 0
        return prepared

    def process_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.process_segment(seg) for seg in segments]

    def learn_from_result(self, seg: dict[str, Any]) -> None:
        features = dict(seg.get("features") or extract_features(seg))
        features["segment_type"] = str(seg.get("segment_type") or features.get("segment_type") or "unknown")
        decision = {
            "prompt_name": seg.get("prompt_name"),
            "retry_mode": seg.get("retry_mode"),
        }
        outcome = {
            "final_score": seg.get("final_score", 0.0),
        }
        self.learner.observe(features=features, decision=decision, outcome=outcome)
        self.memory.add_decision_history(
            {
                "seg": seg.get("seg"),
                "segment_type": features.get("segment_type"),
                "prompt_name": seg.get("prompt_name"),
                "retry_mode": seg.get("retry_mode"),
                "final_score": seg.get("final_score"),
                "accepted": seg.get("accepted"),
                "decision_source": seg.get("decision_source"),
                "cps_target": seg.get("cps_target"),
                "cps_hard_limit": seg.get("cps_hard_limit"),
            }
        )

    def finalize_project(self) -> None:
        self.memory.set_decision_memory(self.learner.export_summary())
        self.memory.save()


def summarize_level9_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "segments": len(segments),
        "segment_types": {},
        "prompt_names": {},
        "retry_modes": {},
        "decision_sources": {},
        "reranked_segments": 0,
        "avg_cps_target": 0.0,
    }
    if not segments:
        return summary

    summary["segment_types"] = dict(Counter(str(seg.get("segment_type") or "unknown") for seg in segments))
    summary["prompt_names"] = dict(Counter(str(seg.get("prompt_name") or "balanced") for seg in segments))
    summary["retry_modes"] = dict(Counter(str(seg.get("retry_mode") or "balanced") for seg in segments))
    summary["decision_sources"] = dict(Counter(str(seg.get("decision_source") or "heuristic_router") for seg in segments))
    summary["reranked_segments"] = sum(1 for seg in segments if int(seg.get("candidate_pool_size", 0) or 0) > 1)
    summary["avg_cps_target"] = round(
        sum(float(seg.get("cps_target", 16.0) or 16.0) for seg in segments) / max(1, len(segments)),
        3,
    )
    return summary
