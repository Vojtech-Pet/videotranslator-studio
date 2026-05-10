from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PolicyRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.data: dict[str, Any] = {"active_policy": "policy_v1", "policies": {}}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass  # poškodený súbor → zachovaj default data
        self.data.setdefault("active_policy", "policy_v1")
        self.data.setdefault("policies", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def register_policy(self, name: str, policy: dict[str, Any]) -> None:
        self.data.setdefault("policies", {})
        payload = dict(policy)
        payload["name"] = name
        self.data["policies"][name] = payload

    def get_policy(self, name: str) -> dict[str, Any]:
        return dict(self.data.get("policies", {}).get(name, {}))

    def get_active_policy_name(self) -> str:
        return str(self.data.get("active_policy") or "policy_v1")

    def get_active_policy(self) -> dict[str, Any]:
        return self.get_policy(self.get_active_policy_name())

    def set_active_policy(self, name: str) -> None:
        if name in self.data.get("policies", {}):
            self.data["active_policy"] = name
