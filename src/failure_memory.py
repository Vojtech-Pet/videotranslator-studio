from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FailureMemory:
    events: list[dict[str, Any]] = field(default_factory=list)

    def log_event(self, seg_id: int, problem_type: str, details: dict[str, Any]) -> None:
        self.events.append(
            {
                "seg_id": seg_id,
                "problem_type": problem_type,
                "details": details,
            }
        )

    def count_problem(self, problem_type: str) -> int:
        return sum(1 for event in self.events if event["problem_type"] == problem_type)

    def suggest_retry_mode(self, segment: dict[str, Any]) -> str:
        duration_status = str(segment.get("duration_status") or "").lower()
        if duration_status in {"missing_wav", "pending_audio"}:
            return "render_audio"
        if segment.get("term_score", 1.0) < 0.85 or segment.get("terminology_consistency_score", 1.0) < 0.85:
            return "locked_terms"
        if duration_status == "too_long" or float(segment.get("final_chars_per_sec", 0.0)) > 17.0:
            return "aggressive_shorten"
        if segment.get("asr_match_score", 1.0) < 0.75:
            return "pronunciation_safe"
        if segment.get("spoken_score", 1.0) < 0.75 or segment.get("coherence_score", 1.0) < 0.75:
            return "dub_friendly"
        return "balanced"
