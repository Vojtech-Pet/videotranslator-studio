from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_memory_payload() -> dict[str, Any]:
    return {
        "preferred_terms": {},
        "banned_forms": [],
        "style_memory": {},
        "retry_stats": {},
        "persona_memory": {},
        "prompt_memory": {},
        "prompt_ranking": {},
        "prompt_templates": {},
        "policy_recommendations": {},
        "last_level8_analysis": {},
        "decision_memory": {},
        "decision_history": [],
        "segment_history": [],
    }


@dataclass(slots=True)
class ProjectMemory:
    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.data = _default_memory_payload()
        else:
            self.data = _default_memory_payload()
        for key, value in _default_memory_payload().items():
            self.data.setdefault(key, value if not isinstance(value, (dict, list)) else value.copy())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_term(self, raw: str, preferred: str) -> None:
        self.data.setdefault("preferred_terms", {})
        self.data["preferred_terms"][raw] = preferred

    def add_banned_form(self, bad_form: str) -> None:
        clean = str(bad_form or "").strip()
        if not clean:
            return
        self.data.setdefault("banned_forms", [])
        if clean not in self.data["banned_forms"]:
            self.data["banned_forms"].append(clean)

    def log_retry_result(self, mode: str, success: bool) -> None:
        key = str(mode or "balanced")
        self.data.setdefault("retry_stats", {})
        stats = self.data["retry_stats"].setdefault(key, {"attempts": 0, "successes": 0})
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
        stats["success_rate"] = round(stats["successes"] / max(1, stats["attempts"]), 3)

    def retry_success_rate(self, mode: str) -> float:
        stats = self.data.get("retry_stats", {}).get(mode)
        if not stats:
            return 0.0
        attempts = int(stats.get("attempts", 0) or 0)
        successes = int(stats.get("successes", 0) or 0)
        if attempts <= 0:
            return 0.0
        return successes / attempts

    def log_persona_result(
        self,
        scene_type: str,
        persona_name: str,
        *,
        pace: str,
        energy: str,
        clarity: str,
        final_score: float,
    ) -> None:
        scene_key = str(scene_type or "unknown")
        self.data.setdefault("persona_memory", {})
        persona_bucket = self.data["persona_memory"].setdefault(scene_key, {})
        stats = persona_bucket.setdefault(
            persona_name or "unknown",
            {
                "persona_name": persona_name or "unknown",
                "pace": pace,
                "energy": energy,
                "clarity": clarity,
                "runs": 0,
                "score_sum": 0.0,
                "score_avg": 0.0,
            },
        )
        stats["pace"] = pace
        stats["energy"] = energy
        stats["clarity"] = clarity
        stats["runs"] += 1
        stats["score_sum"] = round(float(stats.get("score_sum", 0.0)) + float(final_score), 3)
        stats["score_avg"] = round(stats["score_sum"] / max(1, stats["runs"]), 3)

    def log_prompt_result(self, prompt_name: str, final_score: float, accepted: bool) -> None:
        key = str(prompt_name or "balanced")
        self.data.setdefault("prompt_memory", {})
        stats = self.data["prompt_memory"].setdefault(
            key,
            {"prompt_name": key, "uses": 0, "accepted": 0, "score_sum": 0.0, "score_avg": 0.0},
        )
        stats["uses"] += 1
        if accepted:
            stats["accepted"] += 1
        stats["score_sum"] = round(float(stats.get("score_sum", 0.0)) + float(final_score), 3)
        stats["score_avg"] = round(stats["score_sum"] / max(1, stats["uses"]), 3)
        stats["accept_rate"] = round(stats["accepted"] / max(1, stats["uses"]), 3)

    def log_style_result(self, scene_type: str, base_style: str, accepted: bool, final_score: float) -> None:
        key = str(scene_type or "unknown")
        self.data.setdefault("style_memory", {})
        stats = self.data["style_memory"].setdefault(
            key,
            {
                "scene_type": key,
                "preferred_style": base_style,
                "runs": 0,
                "accepted": 0,
                "score_sum": 0.0,
                "score_avg": 0.0,
            },
        )
        stats["preferred_style"] = base_style
        stats["runs"] += 1
        if accepted:
            stats["accepted"] += 1
        stats["score_sum"] = round(float(stats.get("score_sum", 0.0)) + float(final_score), 3)
        stats["score_avg"] = round(stats["score_sum"] / max(1, stats["runs"]), 3)
        stats["accept_rate"] = round(stats["accepted"] / max(1, stats["runs"]), 3)

    def set_prompt_template(self, prompt_name: str, prompt_text: str) -> None:
        self.data.setdefault("prompt_templates", {})
        self.data["prompt_templates"][prompt_name] = prompt_text

    def set_decision_memory(self, payload: dict[str, Any]) -> None:
        self.data["decision_memory"] = payload

    def add_decision_history(self, record: dict[str, Any]) -> None:
        self.data.setdefault("decision_history", [])
        self.data["decision_history"].append(record)

    def add_segment_history(self, record: dict[str, Any]) -> None:
        self.data.setdefault("segment_history", [])
        self.data["segment_history"].append(record)
