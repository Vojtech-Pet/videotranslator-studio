from __future__ import annotations

from typing import Any


def _status_from_delta(delta: float) -> str:
    if abs(delta) <= 0.20:
        return "ok"
    if delta > 0.20:
        return "too_long"
    if delta < -0.40:
        return "too_short"
    return "ok"


class AudioAgent:
    def synthesize(
        self,
        seg: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        slot = max(0.01, float(seg.get("slot") or (float(seg.get("end", 0.0) or 0.0) - float(seg.get("start", 0.0) or 0.0))))
        baseline_cps = float(seg.get("final_chars_per_sec", 0.0) or 0.0)
        if baseline_cps <= 0:
            baseline_cps = max(8.0, len(str(seg.get("tts_input") or "")) / slot)

        variants: list[dict[str, Any]] = []
        prosodies = [("technical_default", 1.00), ("clear_precise", 1.06)]
        for candidate in candidates:
            text = str(candidate.get("tts_input") or "")
            base_duration = len(text) / max(6.0, baseline_cps)
            for prosody_name, factor in prosodies:
                duration = round(base_duration * factor, 3)
                delta = round(duration - slot, 3)
                variants.append(
                    {
                        "candidate_name": candidate.get("candidate_name"),
                        "prompt_name": candidate.get("prompt_name"),
                        "prompt_text": candidate.get("prompt_text"),
                        "tts_input": text,
                        "split_suggestion": candidate.get("split_suggestion"),
                        "prosody_name": prosody_name,
                        "estimated_duration": duration,
                        "slot": round(slot, 3),
                        "delta": delta,
                        "status": _status_from_delta(delta),
                    }
                )
        return variants
