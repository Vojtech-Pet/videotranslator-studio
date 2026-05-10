from __future__ import annotations

from typing import Any


class ASRAgent:
    def __init__(self) -> None:
        pass

    def transcribe(self, text_simulation: str) -> str:
        return str(text_simulation or "").lower()
