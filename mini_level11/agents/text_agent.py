from __future__ import annotations

import re
from typing import Any

import requests

from core.prompts import SYSTEM_PROMPTS
from core.sql_fixer import fix_sql_terms
from core.utils import choose_prompt_name, estimate_cps, get_slot

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:27b"


def _heuristic_rewrite(text: str, prompt_name: str) -> str:
    out = text
    if prompt_name in {"dub_friendly", "aggressive_shorten"}:
        out = re.sub(r"\bv poriadku priatelia\b", " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\btak toto je to, co\b", " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bgo a\b", " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bdeep dive\b", "pozrieme sa", out, flags=re.IGNORECASE)
    if prompt_name == "aggressive_shorten":
        out = re.sub(r"\bteraz\b", " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bv tomto pripade\b", " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bpovedzme, ze\b", " ", out, flags=re.IGNORECASE)
    return fix_sql_terms(out)


class TextAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self._session = requests.Session()
        self._llm_available: bool | None = None

    def call_llm(self, prompt: str, *, fallback_text: str, timeout: int = 5) -> str:
        if self._llm_available is False:
            return fallback_text
        try:
            response = self._session.post(
                OLLAMA_URL,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            text = str(payload.get("response") or "").strip()
            self._llm_available = True
            return text or fallback_text
        except Exception:
            self._llm_available = False
            return fallback_text

    def build_prompt(self, text: str, slot: float, prompt_name: str) -> str:
        system = SYSTEM_PROMPTS[prompt_name]
        return f'{system}\n\nSlot: {slot:.2f} sekundy\nText:\n"{text}"\n'

    def generate_candidates(self, seg: dict[str, Any], hint: str | None = None) -> list[dict[str, Any]]:
        raw_text = str(seg.get("text") or seg.get("tts_input") or "")
        fixed_text = fix_sql_terms(raw_text)
        slot = get_slot(seg)
        cps = estimate_cps(fixed_text, slot)
        prompt_name = choose_prompt_name(fixed_text, cps)
        if hint == "lock_terms":
            prompt_name = "locked_terms"
        elif hint == "aggressive_shorten":
            prompt_name = "aggressive_shorten"

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        fixed_candidate = fix_sql_terms(fixed_text)
        candidates.append({"text": fixed_candidate, "source": "fixed", "prompt_name": "fixed", "hint": hint})
        seen.add(fixed_candidate)

        prompt = self.build_prompt(fixed_text, slot, prompt_name)
        llm_text = fix_sql_terms(
            self.call_llm(prompt, fallback_text=_heuristic_rewrite(fixed_text, prompt_name))
        )
        if llm_text not in seen:
            candidates.append({"text": llm_text, "source": "llm", "prompt_name": prompt_name, "hint": hint})
            seen.add(llm_text)

        if prompt_name != "dub_friendly":
            prompt2 = self.build_prompt(fixed_text, slot, "dub_friendly")
            llm_text2 = fix_sql_terms(
                self.call_llm(prompt2, fallback_text=_heuristic_rewrite(fixed_text, "dub_friendly"))
            )
            if llm_text2 not in seen:
                candidates.append({"text": llm_text2, "source": "llm", "prompt_name": "dub_friendly", "hint": hint})
                seen.add(llm_text2)

        return candidates
