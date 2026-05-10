from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PersonaProfile:
    name: str
    pace: str
    energy: str
    clarity: str
    style: str


def get_persona(scene_type: str) -> PersonaProfile:
    if scene_type == "technical_tutorial":
        return PersonaProfile(
            name="calm_technical_lecturer",
            pace="medium",
            energy="low_medium",
            clarity="high",
            style="clear_calm_explanatory",
        )
    return PersonaProfile(
        name="neutral_explainer",
        pace="medium",
        energy="medium",
        clarity="medium_high",
        style="natural_explanatory",
    )


def persona_stability_score(segment_text: str, persona: PersonaProfile) -> float:
    text = (segment_text or "").lower()
    score = 1.0
    if persona.style == "clear_calm_explanatory":
        if any(x in text for x in ("mega", "brutalne", "uplne top", "wow")):
            score -= 0.30
        if any(x in text for x in ("hlboky ponor", "deep dive", "nasho stola")):
            score -= 0.20
    return round(max(0.0, score), 3)
