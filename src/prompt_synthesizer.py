from __future__ import annotations

from typing import Any


def synthesize_prompt(base_prompt: str, analysis: dict[str, Any]) -> str:
    additions: list[str] = []

    if float(analysis.get("sql_term_corruption_rate", 0.0) or 0.0) > 0.1:
        additions.append(
            "KRITICKE: SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE musia zostat presne v tomto tvare."
        )

    if float(analysis.get("literal_style_rate", 0.0) or 0.0) > 0.1:
        additions.append("Vyhybaj sa doslovnym anglickym konstrukciam a neprirodzenemu slovosledu.")

    if float(analysis.get("duration_failure_rate", 0.0) or 0.0) > 0.1:
        additions.append("Ak je veta dlha, skrat ju bez straty vyznamu.")

    if not additions:
        return (base_prompt or "").strip()

    clean_base = (base_prompt or "").strip()
    if not clean_base:
        clean_base = "Prepis vetu prirodzene pre technicky dabing."
    return clean_base + "\n\n" + "\n".join(additions)
