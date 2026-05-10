from __future__ import annotations

from typing import Any

from src.evaluator_model import evaluate_segment


def _duration_score(delta: float, status: str) -> float:
    if status in {"missing_wav", "pending_audio"}:
        return 0.35
    absolute_delta = abs(delta)
    if absolute_delta <= 0.20:
        return 1.0
    if absolute_delta <= 0.40:
        return 0.8
    if absolute_delta <= 0.70:
        return 0.5
    return 0.2


class EvalAgent:
    def score_variant(self, seg: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
        slot = max(0.01, float(variant.get("slot", seg.get("slot", 0.01)) or 0.01))
        text = str(variant.get("tts_input") or "")
        duration_status = str(variant.get("status") or "unknown")
        delta = float(variant.get("delta", 0.0) or 0.0)

        payload = dict(seg)
        payload["tts_input"] = text
        payload["final_chars_per_sec"] = round(len(text) / slot, 3)
        payload["duration_score"] = _duration_score(delta, duration_status)
        payload["duration_status"] = duration_status
        payload["delta"] = delta

        spoken_score = float(seg.get("spoken_score", 1.0) or 1.0)
        if str(variant.get("candidate_name")) == "dub_friendly":
            spoken_score = min(1.0, spoken_score + 0.08)
        elif str(variant.get("candidate_name")) == "aggressive_shorten":
            spoken_score = min(1.0, spoken_score + 0.04)
        payload["spoken_score"] = round(spoken_score, 3)

        asr_match = float(seg.get("asr_match_score", 1.0) or 1.0)
        if duration_status == "too_short":
            asr_match = max(0.7, asr_match - 0.05)
        payload["asr_match_score"] = round(asr_match, 3)

        scores = evaluate_segment(payload)
        result = dict(variant)
        result.update(scores)
        result["ok"] = bool(duration_status == "ok" and scores["term_score"] >= 0.85 and scores["final_score"] >= 0.78)
        return result

    def select_best(self, seg: dict[str, Any], audio_variants: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [self.score_variant(seg, variant) for variant in audio_variants]
        if not scored:
            return {"ok": False, "final_score": 0.0, "status": "missing_variants"}
        scored.sort(key=lambda item: (item.get("ok", False), float(item.get("final_score", 0.0))), reverse=True)
        best = dict(scored[0])
        best["evaluated_variants"] = len(scored)
        best["scored_variants"] = scored
        return best
