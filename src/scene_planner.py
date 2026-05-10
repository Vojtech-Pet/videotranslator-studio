from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ScenePlan:
    scene_type: str
    base_style: str
    pace: str
    energy: str
    clarity: str
    term_strictness: str
    target_cps: float
    hard_cps_limit: float


def _segment_text(seg: dict[str, Any]) -> str:
    for key in ("tts_input", "level3_text", "level2_text", "level1_text", "text"):
        value = str(seg.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def detect_scene_type(segments: list[dict[str, Any]]) -> str:
    joined = " ".join(_segment_text(seg) for seg in segments)
    tech_hits = [
        "sql",
        "null",
        "is null",
        "is not null",
        "isnull",
        "coalesce",
        "stlpec",
        "funkcia",
        "tabulka",
        "databaza",
        "hodnota",
        "boolean",
    ]
    count = sum(1 for term in tech_hits if term in joined)
    if count >= 4:
        return "technical_tutorial"
    return "general_explanatory"


def make_scene_plan(segments: list[dict[str, Any]]) -> ScenePlan:
    scene_type = detect_scene_type(segments)
    if scene_type == "technical_tutorial":
        return ScenePlan(
            scene_type="technical_tutorial",
            base_style="clear_calm_explanatory",
            pace="medium",
            energy="low_medium",
            clarity="high",
            term_strictness="hard",
            target_cps=16.0,
            hard_cps_limit=17.0,
        )
    return ScenePlan(
        scene_type="general_explanatory",
        base_style="natural_explanatory",
        pace="medium",
        energy="medium",
        clarity="medium_high",
        term_strictness="medium",
        target_cps=17.0,
        hard_cps_limit=18.0,
    )
