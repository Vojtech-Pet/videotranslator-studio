from __future__ import annotations

from pathlib import Path
from typing import Any

from src.policy_registry import PolicyRegistry
from src.project_memory import ProjectMemory


class PolicyAgent:
    def __init__(self, memory_path: str | Path, registry_path: str | Path):
        self.memory = ProjectMemory(Path(memory_path).expanduser().resolve())
        self.memory.load()
        self.registry = PolicyRegistry(registry_path)

    def suggest(self, seg: dict[str, Any], best_variant: dict[str, Any]) -> dict[str, Any]:
        active_policy = self.registry.get_active_policy()
        shorten_threshold = float(active_policy.get("aggressive_shorten_threshold", 17.0) or 17.0)
        cps = float(seg.get("final_chars_per_sec", 0.0) or 0.0)

        if float(best_variant.get("term_score", 1.0) or 1.0) < 0.85:
            retry_mode = "locked_terms"
            reason = "term_integrity"
        elif str(best_variant.get("status") or "") == "too_long" or cps > shorten_threshold:
            retry_mode = "aggressive_shorten"
            reason = "duration_overflow"
        elif str(best_variant.get("status") or "") == "too_short":
            retry_mode = "dub_friendly"
            reason = "duration_underflow"
        elif float(best_variant.get("asr_match_score", 1.0) or 1.0) < 0.85:
            retry_mode = "pronunciation_safe"
            reason = "pronunciation_risk"
        else:
            retry_mode = "balanced"
            reason = "general_cleanup"

        return {"retry_mode": retry_mode, "reason": reason}
