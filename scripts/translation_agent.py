"""
TranslationAgentPipeline — EN → SK prekladový agent pre dabing, titulky a hovorený text.

Architektúra:
    vstup → Analyzer → Planner → Translator → Validator → (Refiner) → výstup

Moduly:
    A. Analyzer   — zistí typ textu, riziká, chránené termíny
    B. Planner    — vygeneruje pravidlá pre LLM podľa analýzy
    C. Translator — samotný LLM preklad s pravidlami
    D. Validator  — skontroluje výstup voči originálu
    E. Refiner    — opraví ak validator nájde problém

Módy:
    dub_friendly       — kratšie, hovorové, rytmus dabingu
    technical_precise  — zachovanie terminológie, presnosť
    subtitle_compact   — úsporné titulky, max dĺžka
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

# ─── Prompty ─────────────────────────────────────────────────────────────────

_ANALYZER_PROMPT = """\
You are a translation analysis agent for English to Slovak dubbing, subtitles, and technical videos.

Analyze the input sentence and return JSON only. No explanation, no markdown.

Tasks:
1. Detect domain: general | technical | dubbing | subtitles | tutorial | narration
2. Detect style: casual | neutral | formal | emotional
3. Detect speaker_intent: statement | question | irony | warning | instruction | dry_comment | resigned_statement
4. Extract protected_terms: technical terms, product names, acronyms, code, commands that must NOT be translated
5. Extract numbers_units: all numbers and measurements
6. Extract entities: proper names, place names, character names
7. Detect idioms: English idioms or figurative phrases that need free translation
8. Decide translation_mode: literal | balanced | natural | dub_friendly
9. Decide length_constraint: strict | medium | free

Return valid JSON only:
{
  "domain": "...",
  "style": "...",
  "speaker_intent": "...",
  "protected_terms": [],
  "numbers_units": [],
  "entities": [],
  "idioms": [],
  "translation_mode": "...",
  "length_constraint": "..."
}"""

_TRANSLATOR_PROMPT = """\
You are a high-quality English to Slovak translation engine for dubbing and subtitles.

Translate the sentence according to the analysis and rules below.

Hard rules (never break):
- Preserve meaning exactly — do not add or remove information
- Preserve all numbers, units, and measurements exactly
- Preserve all protected terms listed in the analysis (keep them in English)
- Do not hallucinate — only translate what is in the source

Soft rules (follow unless they conflict with hard rules):
- Prefer natural spoken Slovak over literal translation
- Avoid bookish or stiff wording
- Use colloquial Slovak for dialogue and dubbing
- Keep output length within the requested constraint

Mode-specific rules:
{mode_rules}

Analysis:
{analysis_json}

Source sentence:
{source_text}

Return ONLY the final Slovak translation. Nothing else."""

_VALIDATOR_PROMPT = """\
You are a strict translation validation agent for English to Slovak dubbing.

Compare the English source and the Slovak translation. Return JSON only. No explanation.

Source (EN): {source_text}
Translation (SK): {translation}
Slot duration: {slot:.1f} seconds
Analysis: {analysis_json}

Check:
1. meaning_ok — is the meaning fully preserved?
2. missing_info — list any information present in EN but missing in SK
3. hallucinations — list anything in SK that is NOT in EN
4. numbers_ok — are all numbers/units preserved exactly?
5. protected_terms_ok — are all protected terms kept untranslated?
6. spoken_naturalness — how natural is this for spoken Slovak: poor | ok | good
7. length_ratio — estimate len(SK_words) / len(EN_words)
8. needs_revision — true if any hard rule is violated OR spoken_naturalness is poor
9. revision_reason — brief reason if needs_revision is true

Return valid JSON only:
{{
  "meaning_ok": true,
  "missing_info": [],
  "hallucinations": [],
  "numbers_ok": true,
  "protected_terms_ok": true,
  "spoken_naturalness": "good",
  "length_ratio": 1.0,
  "needs_revision": false,
  "revision_reason": ""
}}"""

_REFINER_PROMPT = """\
You are a Slovak dubbing refinement agent.

Revise the Slovak translation based on the validator feedback below.

Hard rules:
- Keep the original meaning from SOURCE_EN
- Do not change numbers, units, or protected terms
- Do not add information that is not in SOURCE_EN

Fix what the validator flagged:
{revision_reason}

SOURCE_EN: {source_text}
CURRENT_SK: {translation}
SLOT: {slot:.1f} seconds
VALIDATOR: {validator_json}

Return ONLY the corrected final Slovak translation. Nothing else."""

# ─── Mód-špecifické pravidlá ─────────────────────────────────────────────────

_MODE_RULES = {
    "dub_friendly": """\
- This is DUBBING — prefer shorter, spoken-rhythm Slovak
- Colloquial Slovak is strongly preferred over formal
- A slight paraphrase is allowed if it sounds more natural
- Avoid literal calques from English
- Penalize overly long translations — keep it speakable in {slot:.1f}s""",

    "technical_precise": """\
- This is TECHNICAL content — precision over naturalness
- Keep all technical terms in English (SQL, API, JSON, Docker, Python, etc.)
- Use established Slovak technical vocabulary if it exists
- Do NOT over-translate: 'API vracia JSON' is better than 'rozhranie vracia formát'""",

    "subtitle_compact": """\
- This is a SUBTITLE — be maximally compact
- Remove filler words that don't carry meaning
- Keep it under {slot:.1f}s reading speed (~15 chars/second)
- Clarity and brevity over literary style""",

    "balanced": """\
- Balance accuracy and naturalness
- Natural Slovak preferred but don't sacrifice meaning
- Keep length similar to source""",

    "natural": """\
- Prioritize natural, fluent Slovak
- Moderate freedom to paraphrase if it reads better
- Avoid literal translation when a natural equivalent exists""",

    "literal": """\
- Prioritize literal accuracy
- Stay close to the source structure where possible
- Only deviate when Slovak grammar absolutely requires it""",
}


# ─── Backend abstrakcia ───────────────────────────────────────────────────────

class _LLMBackend:
    """Abstraktný backend — OpenAI API alebo lokálny LLM."""

    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 512) -> str:
        raise NotImplementedError


class OpenAIBackend(_LLMBackend):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def chat(self, messages, temperature=0.1, max_tokens=2000) -> str:
        kwargs: dict = {"max_completion_tokens": max_tokens}
        if temperature != 1.0:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, **kwargs,
                )
            except Exception:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs,
                )
        else:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs,
            )
        return (resp.choices[0].message.content or "").strip()


class LocalLLMBackend(_LLMBackend):
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048):
        from llama_cpp import Llama
        print(f"[AGENT] Načítavam lokálny model: {model_path}", flush=True)
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)

    def chat(self, messages, temperature=0.1, max_tokens=2000) -> str:
        resp = self.llm.create_chat_completion(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()

    def unload(self):
        del self.llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class TranslationAgentPipeline:
    """
    Plná 5-modulová prekladová pipeline pre EN → SK dabing/titulky.

    Použitie:
        agent = TranslationAgentPipeline.from_openai(api_key="sk-...")
        result = agent.process_segment("Well, that didn't go as planned.", slot=2.5)

        # Alebo celé video:
        translated_segments = agent.process_segments(segments)
    """

    def __init__(
        self,
        backend: _LLMBackend,
        *,
        default_mode: str = "dub_friendly",
        speaker_gender: str = "m",       # "m" | "f" | "auto"
        glossary: dict[str, str] | None = None,
        max_refine_attempts: int = 2,
        verbose: bool = True,
    ):
        self.backend = backend
        self.default_mode = default_mode
        self.speaker_gender = speaker_gender
        self.glossary: dict[str, str] = glossary or {}
        self.max_refine_attempts = max_refine_attempts
        self.verbose = verbose
        self._context: list[dict] = []   # pamäť posledných segmentov

    # ── Továrenské metódy ────────────────────────────────────────────────────

    @classmethod
    def from_openai(
        cls,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        **kwargs,
    ) -> "TranslationAgentPipeline":
        return cls(OpenAIBackend(api_key=api_key, model=model), **kwargs)

    @classmethod
    def from_local_llm(
        cls,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        **kwargs,
    ) -> "TranslationAgentPipeline":
        return cls(LocalLLMBackend(model_path, n_gpu_layers, n_ctx), **kwargs)

    # ── A. Analyzer ──────────────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        """Analyzuje vetu a vráti JSON s metadátami."""
        raw = self.backend.chat(
            messages=[
                {"role": "system", "content": _ANALYZER_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        try:
            return json.loads(_extract_json(raw))
        except Exception:
            return {
                "domain": "general", "style": "neutral", "speaker_intent": "statement",
                "protected_terms": [], "numbers_units": [], "entities": [],
                "idioms": [], "translation_mode": self.default_mode, "length_constraint": "medium",
            }

    # ── B. Planner ───────────────────────────────────────────────────────────

    def plan(self, analysis: dict, slot: float) -> str:
        """Z analýzy vyberie mód a vygeneruje mode_rules string."""
        mode = analysis.get("translation_mode", self.default_mode)
        template = _MODE_RULES.get(mode, _MODE_RULES["balanced"])
        return template.format(slot=slot)

    # ── C. Translator ────────────────────────────────────────────────────────

    def translate(self, text: str, analysis: dict, mode_rules: str) -> str:
        """Preloží vetu podľa analýzy a pravidiel."""
        system = _build_translator_system(
            analysis=analysis,
            mode_rules=mode_rules,
            source_text=text,
            glossary=self.glossary,
            speaker_gender=self.speaker_gender,
            context=self._context,
        )
        raw = self.backend.chat(
            messages=[{"role": "user", "content": system}],
            temperature=0.15,
            max_tokens=2000,
        )
        return raw.strip()

    # ── D. Validator ─────────────────────────────────────────────────────────

    def validate(self, source: str, translation: str, analysis: dict, slot: float) -> dict:
        """Skontroluje výstup voči originálu. Vráti JSON dict."""
        prompt = _VALIDATOR_PROMPT
        prompt = prompt.replace("{source_text}", source)
        prompt = prompt.replace("{translation}", translation)
        prompt = prompt.replace("{slot:.1f}", f"{slot:.1f}")
        prompt = prompt.replace("{analysis_json}", json.dumps(analysis, ensure_ascii=False))
        raw = self.backend.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        try:
            return json.loads(_extract_json(raw))
        except Exception:
            return {"needs_revision": False, "spoken_naturalness": "ok", "length_ratio": 1.0}

    # ── E. Refiner ───────────────────────────────────────────────────────────

    def refine(self, source: str, translation: str, validation: dict, slot: float) -> str:
        """Opraví preklad podľa feedbacku validátora."""
        prompt = _REFINER_PROMPT
        prompt = prompt.replace("{source_text}", source)
        prompt = prompt.replace("{translation}", translation)
        prompt = prompt.replace("{slot:.1f}", f"{slot:.1f}")
        prompt = prompt.replace("{validator_json}", json.dumps(validation, ensure_ascii=False))
        prompt = prompt.replace("{revision_reason}", validation.get("revision_reason", "improve naturalness"))
        raw = self.backend.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        return raw.strip()

    # ── Hlavná pipeline ──────────────────────────────────────────────────────

    def process_segment(self, text: str, slot: float = 3.0) -> tuple[str, dict]:
        """
        Spustí plnú pipeline na jeden segment.
        Vracia (final_translation, pipeline_log).
        """
        text = text.strip()
        if not text:
            return text, {}

        log: dict[str, Any] = {"source": text, "slot": slot}

        # A. Analyze
        analysis = self.analyze(text)
        log["analysis"] = analysis
        if self.verbose:
            mode = analysis.get("translation_mode", "?")
            domain = analysis.get("domain", "?")
            print(f"[AGENT] Analyze: domain={domain} mode={mode}", flush=True)

        # B. Plan
        mode_rules = self.plan(analysis, slot)

        # C. Translate
        translation = self.translate(text, analysis, mode_rules)
        log["translation_raw"] = translation
        if self.verbose:
            print(f"[AGENT] Translate: {translation[:70]}", flush=True)

        # Guard: príliš dlhý alebo prázdny výstup → vráť originál
        if not translation or len(translation) > len(text) * 3:
            log["skipped"] = "too long or empty"
            return text, log

        # D. Validate
        validation = self.validate(text, translation, analysis, slot)
        log["validation"] = validation
        if self.verbose:
            nat = validation.get("spoken_naturalness", "?")
            needs = validation.get("needs_revision", False)
            print(f"[AGENT] Validate: naturalness={nat} needs_revision={needs}", flush=True)

        # E. Refine (ak treba)
        final = translation
        for attempt in range(self.max_refine_attempts):
            if not validation.get("needs_revision"):
                break
            if self.verbose:
                reason = validation.get("revision_reason", "")
                print(f"[AGENT] Refine #{attempt+1}: {reason}", flush=True)
            refined = self.refine(text, final, validation, slot)
            if refined and len(refined) <= len(text) * 3:
                final = refined
                log[f"refined_{attempt+1}"] = final
                # Znovu validuj
                validation = self.validate(text, final, analysis, slot)
            else:
                break

        log["final"] = final

        # Aktualizuj kontext
        self._context.append({"src": text, "tgt": final})
        if len(self._context) > 5:
            self._context.pop(0)

        return final, log

    def process_segments(
        self,
        segments: list[dict],
        src_context: str = "",
        slot_field: str | None = None,
    ) -> list[dict]:
        """
        Preloží zoznam segmentov (formát pipeline: [{start, end, text, ...}]).
        Zachová všetky ostatné polia, zmení len 'text'.
        """
        if src_context and self.verbose:
            print(f"[AGENT] Kontext videa: {src_context}", flush=True)

        total = len(segments)
        result = []
        for i, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            if not text:
                result.append(seg)
                continue

            slot = float(seg.get("end", 0)) - float(seg.get("start", 0))
            if slot <= 0:
                slot = 3.0

            if self.verbose:
                print(f"[AGENT] --- Segment {i+1}/{total} (slot={slot:.1f}s) ---", flush=True)

            final, log = self.process_segment(text, slot=slot)

            seg_copy = dict(seg)
            seg_copy["text_src"] = text
            seg_copy["text"] = final
            seg_copy["_agent_log"] = log
            result.append(seg_copy)

        return result

    def update_glossary(self, term_en: str, term_sk: str):
        self.glossary[term_en] = term_sk

    def reset_context(self):
        self._context.clear()


# ─── Pomocné funkcie ─────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Extrahuje JSON z textu (aj keď je obalený v markdowne)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return m.group(1)
    m = re.search(r"(\{[\s\S]+\})", text)
    if m:
        return m.group(1)
    return text


def _build_translator_system(
    analysis: dict,
    mode_rules: str,
    source_text: str,
    glossary: dict,
    speaker_gender: str,
    context: list[dict],
) -> str:
    # Použiť string replace namiesto .format() — analysis_json obsahuje { } ktoré by rozobili format()
    prompt = _TRANSLATOR_PROMPT
    prompt = prompt.replace("{mode_rules}", mode_rules)
    prompt = prompt.replace("{analysis_json}", json.dumps(analysis, ensure_ascii=False))
    prompt = prompt.replace("{source_text}", source_text)
    parts = [prompt]

    if speaker_gender == "f":
        parts.append("\nSpeaker gender: FEMALE — use feminine verb forms (urobila, bola, videla...).")
    elif speaker_gender == "m":
        parts.append("\nSpeaker gender: MALE — use masculine verb forms (urobil, bol, videl...).")

    if glossary:
        glines = "\n".join(f"  {k} → {v}" for k, v in glossary.items())
        parts.append(f"\nGlossary for this video:\n{glines}")

    if context:
        clines = "\n".join(f"  EN: {c['src']}\n  SK: {c['tgt']}" for c in context[-3:])
        parts.append(f"\nRecent context (for consistency):\n{clines}")

    return "\n".join(parts)


# ─── Drop-in funkcia pre pipeline ────────────────────────────────────────────

def translate_with_agent(
    segments: list[dict],
    target_lang: str = "sk",
    speaker_gender: str = "m",
    mode: str = "dub_friendly",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    local_model_path: str | None = None,
    glossary: dict | None = None,
    src_context: str = "",
    verbose: bool = True,
) -> list[dict]:
    """
    Drop-in náhrada za translate_segments() — používa plnú agent pipeline.

    Parametre:
        segments         : [{start, end, text}]
        speaker_gender   : "m" / "f" / "auto"
        mode             : "dub_friendly" | "technical_precise" | "subtitle_compact"
        model            : OpenAI model (ignorované ak local_model_path je nastavený)
        local_model_path : cesta k GGUF modelu pre offline použitie
        glossary         : vlastný glosár {EN: SK}
        src_context      : popis obsahu videa

    Príklad:
        from translation_agent import translate_with_agent
        segs = translate_with_agent(segments, mode="technical_precise", src_context="Docker tutoriál")
    """
    if local_model_path:
        agent = TranslationAgentPipeline.from_local_llm(
            model_path=local_model_path,
            default_mode=mode,
            speaker_gender=speaker_gender,
            glossary=glossary,
            verbose=verbose,
        )
    else:
        agent = TranslationAgentPipeline.from_openai(
            api_key=api_key,
            model=model,
            default_mode=mode,
            speaker_gender=speaker_gender,
            glossary=glossary,
            verbose=verbose,
        )
    return agent.process_segments(segments, src_context=src_context)


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Nastav OPENAI_API_KEY env premennú.")
        sys.exit(1)

    test_cases = [
        # (text, slot, popis)
        ("Well, that didn't go as planned.", 2.5, "dabing — casual"),
        ("The server didn't respond within 30 seconds, so the request failed.", 4.0, "technický"),
        ("I don't think that's really a good idea.", 3.0, "dabing — dialóg"),
        ("First, let's run git commit -m 'fix' to save our changes.", 4.5, "tech tutoriál"),
        ("If you have any questions, leave them in the comments below.", 3.5, "narration"),
    ]

    print("=== TranslationAgentPipeline — test ===\n")
    agent = TranslationAgentPipeline.from_openai(api_key=api_key, default_mode="dub_friendly")

    for text, slot, desc in test_cases:
        print(f"[{desc}]")
        print(f"  EN: {text}")
        final, log = agent.process_segment(text, slot=slot)
        print(f"  SK: {final}")
        v = log.get("validation", {})
        print(f"  naturalness={v.get('spoken_naturalness','?')} ratio={v.get('length_ratio','?')} revised={'yes' if 'refined_1' in log else 'no'}")
        print()
