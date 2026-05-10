from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PromptMemory:
    prompts: dict[str, str]

    @classmethod
    def default(cls) -> "PromptMemory":
        return cls(
            prompts={
                "balanced": (
                    "Prepis vetu do prirodzenej technickej slovenciny vhodnej pre voiceover.\n"
                    "Zachovaj vyznam a technicke terminy presne.\n"
                    "Vrat iba finalnu vetu."
                ),
                "dub_friendly": (
                    "Prepis vetu do prirodzenej hovorenej slovenciny vhodnej pre dabing.\n"
                    "Skrat vetu, ak sa da. Zachovaj technicke terminy presne.\n"
                    "Vrat iba finalnu vetu."
                ),
                "aggressive_shorten": (
                    "Prepis vetu co najstrucnejsie bez straty vyznamu.\n"
                    "Zachovaj presne: SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, TRUE, FALSE, BOOLEAN.\n"
                    "Vrat iba finalnu vetu."
                ),
                "locked_terms": (
                    "Prepis vetu prirodzene pre dabing.\n\n"
                    "KRITICKE:\n"
                    "SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, TRUE, FALSE, BOOLEAN\n"
                    "musia zostat presne v tomto tvare.\n"
                    "Nesmu byt prepisane foneticky.\n"
                    "Vrat iba finalnu vetu."
                ),
                "pronunciation_safe": (
                    "Prepis vetu tak, aby sa lahko vyslovovala v TTS.\n"
                    "Pouzi kratsie a jasnejsie konstrukcie.\n"
                    "Zachovaj technicke terminy presne.\n"
                    "Vrat iba finalnu vetu."
                ),
                "render_audio": (
                    "Pouzi aktualny text a vyrenderuj novu audio verziu bez zmeny terminov.\n"
                    "Ak je to mozne, zachovaj technicky kludny styl."
                ),
            }
        )

    def merge_overrides(self, prompts: dict[str, str] | None) -> None:
        if not prompts:
            return
        for key, value in prompts.items():
            clean = str(value or "").strip()
            if clean:
                self.prompts[str(key)] = clean

    def get(self, mode: str) -> str:
        return self.prompts.get(mode, self.prompts["balanced"])
