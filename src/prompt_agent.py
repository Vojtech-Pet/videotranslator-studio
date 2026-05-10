from __future__ import annotations

from pathlib import Path
from typing import Any

from src.policy_registry import PolicyRegistry
from src.project_memory import ProjectMemory


SEGMENT_PROMPT_MAP = {
    "technical_definition": "locked_terms",
    "technical_explanation": "balanced",
    "example_explanation": "dub_friendly",
    "summary": "balanced",
    "general_explanation": "dub_friendly",
}


class PromptAgent:
    def __init__(self, memory_path: str | Path, registry_path: str | Path):
        self.memory = ProjectMemory(Path(memory_path).expanduser().resolve())
        self.memory.load()
        self.registry = PolicyRegistry(registry_path)

    def get_prompt(
        self,
        seg: dict[str, Any],
        *,
        segment_type: str,
        retry_mode: str | None = None,
    ) -> dict[str, str]:
        active_policy = self.registry.get_active_policy()
        prompt_templates = dict(self.memory.data.get("prompt_templates", {}))

        prompt_name = retry_mode or str(seg.get("prompt_name") or "").strip()
        if not prompt_name:
            prompt_name = SEGMENT_PROMPT_MAP.get(segment_type, "balanced")

        template = str(prompt_templates.get(prompt_name) or active_policy.get("base_prompt") or "").strip()
        if not template:
            template = "Prepis vetu prirodzene pre technicky dabing."

        additions: list[str] = []
        if segment_type == "technical_definition":
            additions.append("Zachovaj technicke terminy uplne presne.")
        elif segment_type == "example_explanation":
            additions.append("Pouzi prirodzenejsiu hovorenost a kratsie vety.")
        elif segment_type == "summary":
            additions.append("Uprednostni strucnost a jasnost.")

        return {
            "prompt_name": prompt_name,
            "prompt_text": template + ("\n\n" + "\n".join(additions) if additions else ""),
            "policy_name": self.registry.get_active_policy_name(),
        }
