from __future__ import annotations

from pathlib import Path
from typing import Any

from src.project_memory import ProjectMemory


class MemoryAgent:
    def __init__(self, memory_path: str | Path):
        self.memory = ProjectMemory(Path(memory_path).expanduser().resolve())
        self.memory.load()

    def store(self, result: dict[str, Any]) -> None:
        self.memory.data.setdefault("agent_runs", [])
        self.memory.data["agent_runs"].append(
            {
                "seg": result.get("seg"),
                "segment_type": result.get("segment_type"),
                "prompt_name": result.get("prompt_name"),
                "retry_mode": result.get("retry_mode"),
                "final_score": result.get("final_score"),
                "ok": result.get("ok"),
                "attempts": result.get("attempt_count"),
                "policy_name": result.get("policy_name"),
                "best_candidate_name": result.get("best_candidate_name"),
                "best_prosody_name": result.get("best_prosody_name"),
            }
        )

    def finalize(self, summary: dict[str, Any] | None = None) -> None:
        if summary is not None:
            self.memory.data["last_level11_summary"] = dict(summary)
        self.memory.save()
