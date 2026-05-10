from __future__ import annotations

from typing import Any

from core.utils import estimate_cps, get_slot


class AudioAgent:
    def __init__(self) -> None:
        pass

    def prepare_audio_candidates(
        self,
        seg: dict[str, Any],
        text_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        slot = get_slot(seg)
        out: list[dict[str, Any]] = []

        for candidate in text_candidates:
            text = str(candidate["text"])
            cps = estimate_cps(text, slot)

            if cps <= 16.0:
                duration_score = 1.0
            elif cps <= 17.0:
                duration_score = 0.8
            else:
                duration_score = 0.5

            out.append(
                {
                    **candidate,
                    "slot": round(slot, 3),
                    "cps": round(cps, 2),
                    "duration_score": duration_score,
                    "tts_ready_text": text,
                }
            )

        return out
