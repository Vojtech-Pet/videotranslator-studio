from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MEMORY_FILE = ROOT / "memory" / "pipeline_memory.json"
BAD_TERMS = {"ajznul", "koalesk", "esikjuel", "esikjul"}


class MemoryAgent:
    def __init__(self) -> None:
        self.path = MEMORY_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self.memory = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.memory = {
                    "bad_terms": {},
                    "bad_patterns": {},
                    "good_prompts": {},
                    "failures": [],
                }
        else:
            self.memory = {
                "bad_terms": {},
                "bad_patterns": {},
                "good_prompts": {},
                "failures": [],
            }

    def save(self) -> None:
        self.path.write_text(json.dumps(self.memory, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_failure(self, seg: dict[str, Any]) -> None:
        self.memory.setdefault("failures", []).append(
            {
                "text": seg.get("text"),
                "tts_input": seg.get("tts_input"),
                "score": seg.get("final_score"),
                "selected_prompt": seg.get("selected_prompt"),
                "retry_mode": seg.get("retry_mode"),
            }
        )

        text = str(seg.get("tts_input", "") or "").lower()
        for word in text.split():
            normalized = word.strip(".,:;!?\"'()[]{}")
            if normalized in BAD_TERMS:
                self.memory.setdefault("bad_terms", {})
                self.memory["bad_terms"][normalized] = self.memory["bad_terms"].get(normalized, 0) + 1

        if float(seg.get("cps", 0.0) or 0.0) > 17.0:
            self.memory.setdefault("bad_patterns", {})
            self.memory["bad_patterns"]["high_cps"] = self.memory["bad_patterns"].get("high_cps", 0) + 1

        self.save()

    def record_success(self, seg: dict[str, Any]) -> None:
        prompt = str(seg.get("selected_prompt") or "unknown")
        self.memory.setdefault("good_prompts", {})
        self.memory["good_prompts"][prompt] = self.memory["good_prompts"].get(prompt, 0) + 1
        self.save()

    def get_hint(self) -> str | None:
        bad_terms = self.memory.get("bad_terms", {})
        if bad_terms.get("ajznul", 0) > 2:
            return "lock_terms"
        if self.memory.get("bad_patterns", {}).get("high_cps", 0) > 3:
            return "aggressive_shorten"
        return None
