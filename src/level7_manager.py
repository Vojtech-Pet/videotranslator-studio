from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.adaptive_policy import AdaptivePolicy
from src.learning_logger import log_segment_outcome
from src.project_memory import ProjectMemory
from src.prompt_memory import PromptMemory


DEFAULT_PREFERRED_TERMS = {
    "sql": "SQL",
    "null": "NULL",
    "is null": "IS NULL",
    "is not null": "IS NOT NULL",
    "isnull": "ISNULL",
    "coalesce": "COALESCE",
    "true": "TRUE",
    "false": "FALSE",
    "boolean": "BOOLEAN",
}

DEFAULT_BANNED_FORMS = (
    "ajznul",
    "koalesk",
    "esikjuel",
    "esikjul",
    "iz not null",
    "izn ot null",
)


class Level7Manager:
    def __init__(self, memory_path: str | Path):
        self.memory = ProjectMemory(path=Path(memory_path).expanduser().resolve())
        self.memory.load()
        self.policy = AdaptivePolicy(self.memory)
        self.prompts = PromptMemory.default()
        self.prompts.merge_overrides(self.memory.data.get("prompt_templates"))

    def bootstrap_defaults(self) -> None:
        for raw, preferred in DEFAULT_PREFERRED_TERMS.items():
            self.memory.update_term(raw, preferred)
        for bad_form in DEFAULT_BANNED_FORMS:
            self.memory.add_banned_form(bad_form)
        for prompt_name, prompt_text in self.prompts.prompts.items():
            self.memory.set_prompt_template(prompt_name, prompt_text)

    def enrich_segment(self, seg: dict[str, Any]) -> dict[str, Any]:
        seg_copy = dict(seg)
        retry_mode = self.policy.choose_retry_mode(seg_copy)
        prompt_name = self.policy.choose_prompt_mode(seg_copy, retry_mode)
        seg_copy["retry_mode"] = retry_mode
        seg_copy["prompt_name"] = prompt_name
        seg_copy["selected_prompt"] = self.prompts.get(prompt_name)
        seg_copy["force_locked_terms"] = self.policy.should_force_locked_terms(seg_copy)
        return seg_copy

    def finalize_segment(self, seg: dict[str, Any]) -> None:
        accepted = bool(seg.get("accepted", False))
        retry_mode = str(seg.get("retry_mode") or "balanced")
        prompt_name = str(seg.get("prompt_name") or retry_mode)
        final_score = float(seg.get("final_score", 0.0) or 0.0)
        self.memory.log_retry_result(retry_mode, accepted)
        self.memory.log_prompt_result(prompt_name, final_score, accepted)
        self.memory.log_persona_result(
            str(seg.get("scene_type") or "unknown"),
            str(seg.get("persona_name") or "unknown"),
            pace=str(seg.get("persona_pace") or "medium"),
            energy=str(seg.get("persona_energy") or "medium"),
            clarity=str(seg.get("persona_clarity") or "medium"),
            final_score=final_score,
        )
        self.memory.log_style_result(
            str(seg.get("scene_type") or "unknown"),
            str(seg.get("scene_base_style") or "unknown"),
            accepted,
            final_score,
        )
        log_segment_outcome(self.memory, seg)

    def finalize_project(self) -> None:
        self.memory.save()

    def process_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.bootstrap_defaults()
        return [self.enrich_segment(seg) for seg in segments]


def summarize_level7_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "segments": len(segments),
        "accepted": 0,
        "needs_retry": 0,
        "retry_modes": {},
        "prompt_names": {},
        "forced_locked_terms": 0,
        "avg_final_score": 0.0,
    }
    if not segments:
        return summary
    summary["accepted"] = sum(1 for seg in segments if seg.get("accepted"))
    summary["needs_retry"] = len(segments) - summary["accepted"]
    summary["retry_modes"] = dict(Counter(str(seg.get("retry_mode") or "balanced") for seg in segments))
    summary["prompt_names"] = dict(Counter(str(seg.get("prompt_name") or "balanced") for seg in segments))
    summary["forced_locked_terms"] = sum(1 for seg in segments if seg.get("force_locked_terms"))
    summary["avg_final_score"] = round(
        sum(float(seg.get("final_score", 0.0) or 0.0) for seg in segments) / max(1, len(segments)),
        3,
    )
    return summary
