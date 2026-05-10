from __future__ import annotations

from typing import Any

BAD_TERMS = ["ajznul", "koalesk", "esikjuel", "esikjul", "iz not null", "izn ot null"]


class EvalAgent:
    def __init__(self) -> None:
        pass

    def score_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        text = str(candidate["text"])
        low = text.lower()

        term_score = 1.0
        for bad in BAD_TERMS:
            if bad in low:
                term_score -= 0.4
        term_score = max(0.0, term_score)

        cps = float(candidate["cps"])
        if cps <= 16.0:
            cps_score = 1.0
        elif cps <= 17.0:
            cps_score = 0.8
        else:
            cps_score = 0.5

        spoken_score = 1.0
        for bad_phrase in ["hlboky ponor", "nas ho stola", "nasho stola", "odide a nahradi", "go a"]:
            if bad_phrase in low:
                spoken_score -= 0.3
        spoken_score = max(0.0, spoken_score)

        final_score = (
            0.4 * term_score
            + 0.3 * float(candidate["duration_score"])
            + 0.2 * cps_score
            + 0.1 * spoken_score
        )

        return {
            **candidate,
            "term_score": round(term_score, 3),
            "cps_score": round(cps_score, 3),
            "spoken_score": round(spoken_score, 3),
            "final_score": round(final_score, 3),
        }

    def choose_best(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [self.score_candidate(candidate) for candidate in candidates]
        if not scored:
            return {}
        scored.sort(key=lambda item: item["final_score"], reverse=True)
        return scored[0]
