"""
text_adaptation.py
==================
Timing-aware text adaptation pre dubbing.

Upraví preložený text tak, aby sa zmestil do časového slotu segmentu.
Ak je TTS odhad príliš dlhý → rewrite (concise / dub-friendly).

Logika:
  estimated ≤ slot * 1.05  → nechaj tak
  estimated > slot * 1.05  → concise rewrite (1. kolo)
  stále preteká            → dub-friendly rewrite (2. kolo, agresívnejší)
  stále preteká            → fallback (stretch v pipeline)

Kalibrácia:
  SK_CHARS_PER_SEC = 13.0 — Chatterbox SK pri normálnom tempe
  Skratky (2+ veľké písmená) dostávajú bonus +0.3s/skratka (expandujú sa foneticky)

Integrácia do pipeline.py:
  text = adapt_for_timing(
      raw_translated_text, src_text=src, slot_duration=slot_dur,
      llm=llm_instance,  # alebo None → len odhad, žiadny rewrite
  ).text
"""

import re
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from sql_logic_guard import (
    SQL_PHONETIC_PATTERNS,
    SQL_PROTECTED_TERMS as SQL_GUARD_TERMS,
    guard_sql_text,
    protected_sql_terms_in as _protected_sql_terms_in_guard,
)

# ── Kalibrácia ──────────────────────────────────────────────────────────────

SK_CHARS_PER_SEC: float = 14.5
ABBREV_BONUS_S:   float = 0.30
THRESHOLD_SOFT:   float = 1.20
THRESHOLD_HARD:   float = 1.40
THRESHOLD_FORCE_DUB: float = 1.60

TARGET_CPS_TECHNICAL: float = 16.0
HARD_CPS_LIMIT_TECHNICAL: float = 17.0
TARGET_CPS_GENERAL: float = 17.0
HARD_CPS_LIMIT_GENERAL: float = 18.0
TARGET_CPS_FAST: float = 18.0
HARD_CPS_LIMIT_FAST: float = 18.5

SQL_PROTECTED_TERMS = (
    *SQL_GUARD_TERMS,
)

_EXAMPLE_HINTS = (
    r"\bfor example\b",
    r"\blet'?s say\b",
    r"\bnapríklad\b",
    r"\bpovedzme\b",
    r"\bukáž(?:me)? si\b",
)

_SUMMARY_HINTS = (
    r"\bso\b",
    r"\bto sum up\b",
    r"\bin summary\b",
    r"\bna záver\b",
    r"\bstručne\b",
    r"\bzhrňme si\b",
)

_PROSODY_HINTS = {
    "technical_spoken": "Tón: pokojný, presný, technický voiceover.",
    "example_explanation": "Tón: vysvetľujúci a mierne živší pri príkladoch.",
    "summary": "Tón: stručný a uzatvárajúci.",
    "general_spoken": "Tón: prirodzený hovorený voiceover.",
}


# ── Výsledok adaptácie ───────────────────────────────────────────────────────

@dataclass
class AdaptResult:
    text:               str     # finálny text (original alebo rewritten)
    mode:               str     # "ok_no_change" / "concise_rewrite" / "dub_friendly_rewrite"
                                # / "fallback_stretch" / "meta_response_rejected"
    estimated_duration: float   # odhad dĺžky TTS [s]
    slot_duration:      float   # slot [s]
    ratio:              float   # estimated / slot (>1.0 = preteká)
    ratio_before:       float   = 0.0   # ratio pred rewritom (pre štatistiky)
    rewrite_rounds:     int     = 0
    rewrite_mode:       str     = "balanced"
    style:              str     = "general_spoken"
    cps_before:         float   = 0.0
    cps_after:          float   = 0.0
    target_cps:         float   = 0.0
    hard_cps_limit:     float   = 0.0
    analysis:           dict[str, Any] = field(default_factory=dict)
    notes:              list    = field(default_factory=list)


@dataclass
class SegmentAnalysis:
    slot: float
    text_len: int
    chars_per_sec: float
    target_cps: float
    hard_cps_limit: float
    has_sql_terms: bool
    has_code_like_sql: bool
    has_numbers: bool
    style: str
    needs_rewrite: bool
    needs_merge: bool
    needs_split: bool
    priority: str


# ── Adaptation štatistiky ────────────────────────────────────────────────────

@dataclass
class AdaptStats:
    total:                  int   = 0
    ok_no_change:           int   = 0
    concise_rewrite:        int   = 0
    dub_friendly_rewrite:   int   = 0
    aggressive_rewrite:     int   = 0
    fallback_stretch:       int   = 0
    meta_response_rejected: int   = 0
    skip_sql:               int   = 0
    skip_short:             int   = 0
    ratio_before_sum:       float = 0.0   # suma ratio pred adaptáciou (len overflow)
    ratio_after_sum:        float = 0.0   # suma ratio po adaptácii (len overflow)
    overflow_count:         int   = 0     # segmenty kde ratio_before > 1.0

    def record(self, result: "AdaptResult"):
        self.total += 1
        mode = result.mode
        if   mode == "ok_no_change":           self.ok_no_change += 1
        elif mode == "concise_rewrite":        self.concise_rewrite += 1
        elif mode == "dub_friendly_rewrite":   self.dub_friendly_rewrite += 1
        elif mode == "aggressive_rewrite":     self.aggressive_rewrite += 1
        elif mode == "fallback_stretch":       self.fallback_stretch += 1
        elif mode == "meta_response_rejected": self.meta_response_rejected += 1
        if result.ratio_before > 1.0:
            self.overflow_count   += 1
            self.ratio_before_sum += result.ratio_before
            self.ratio_after_sum  += result.ratio

    def summary(self) -> str:
        avg_before = self.ratio_before_sum / self.overflow_count if self.overflow_count else 0
        avg_after  = self.ratio_after_sum  / self.overflow_count if self.overflow_count else 0
        lines = [
            f"[ADAPT STATS] celkovo={self.total}  overflow={self.overflow_count}",
            f"  ok_no_change:         {self.ok_no_change}",
            f"  concise_rewrite:      {self.concise_rewrite}",
            f"  dub_friendly_rewrite: {self.dub_friendly_rewrite}",
            f"  aggressive_rewrite:   {self.aggressive_rewrite}",
            f"  fallback_stretch:     {self.fallback_stretch}",
            f"  meta_rejected:        {self.meta_response_rejected}",
            f"  skip_sql:             {self.skip_sql}",
            f"  skip_short:           {self.skip_short}",
        ]
        if self.overflow_count:
            lines.append(f"  avg ratio before: {avg_before:.2f}  →  after: {avg_after:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        avg_before = self.ratio_before_sum / self.overflow_count if self.overflow_count else 0
        avg_after  = self.ratio_after_sum  / self.overflow_count if self.overflow_count else 0
        return {
            "total": self.total, "overflow": self.overflow_count,
            "ok_no_change": self.ok_no_change,
            "concise_rewrite": self.concise_rewrite,
            "dub_friendly_rewrite": self.dub_friendly_rewrite,
            "aggressive_rewrite": self.aggressive_rewrite,
            "fallback_stretch": self.fallback_stretch,
            "meta_response_rejected": self.meta_response_rejected,
            "skip_sql": self.skip_sql,
            "skip_short": self.skip_short,
            "avg_ratio_before": round(avg_before, 3),
            "avg_ratio_after":  round(avg_after,  3),
        }


# ── Odhad TTS dĺžky ─────────────────────────────────────────────────────────

def estimate_tts_duration(
    text: str,
    chars_per_sec: float = SK_CHARS_PER_SEC,
    abbrev_bonus: float   = ABBREV_BONUS_S,
) -> float:
    """
    Odhadne dĺžku TTS bez spustenia modelu.

    Skratky (2+ veľké písmená v slove, napr. API, HTTP, CI/CD) sa foneticky expandujú
    (API → éj-pí-áj = ~3× dlhšie). Pridáme bonus za každú skratku.
    """
    base = len(text) / max(chars_per_sec, 1.0)
    abbrevs = re.findall(r'\b[A-Z]{2,}(?:/[A-Z]+)?\b', text)
    return base + len(abbrevs) * abbrev_bonus


def needs_adaptation(
    text: str,
    slot_duration: float,
    threshold: float = THRESHOLD_SOFT,
) -> bool:
    """True ak odhadovaná TTS dĺžka preteká slot o viac ako threshold."""
    if slot_duration <= 0:
        return False
    est = estimate_tts_duration(text)
    return est > slot_duration * threshold


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fix_sql_terms(text: str, src_text: str = "", fallback_text: str = "") -> str:
    out = normalize_spaces(text)
    if not out:
        return out
    return guard_sql_text(out, source_text=src_text, fallback_text=fallback_text)


def _contains_sql_terms(text: str, src_text: str = "") -> bool:
    haystack = f"{text}\n{src_text}"
    for term in SQL_PROTECTED_TERMS:
        pattern = re.escape(term).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", haystack, flags=re.IGNORECASE):
            return True
    return bool(re.search(r"\b(?:database|query|table|column|row|join|where)\b", haystack, flags=re.IGNORECASE))


def _looks_like_sql_code(text: str) -> bool:
    return bool(
        re.search(r"\b(?:SELECT\s+\S|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|WITH\s+\w+\s+AS)\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:ISNULL|COALESCE|NULLIF|IFNULL|NVL|IIF)\s*\(", text, flags=re.IGNORECASE)
        or re.search(r"[=<>!()*_]", text)
    )


def estimate_chars_per_second(text: str, slot_duration: float) -> float:
    if slot_duration <= 0.0:
        return 999.0
    return len(normalize_spaces(text)) / max(slot_duration, 0.01)


def _style_limits(style: str) -> tuple[float, float]:
    if style == "technical_spoken":
        return TARGET_CPS_TECHNICAL, HARD_CPS_LIMIT_TECHNICAL
    if style == "example_explanation":
        return 15.5, 16.5
    if style == "summary":
        return 16.5, 17.5
    if style == "fast_spoken":
        return TARGET_CPS_FAST, HARD_CPS_LIMIT_FAST
    return TARGET_CPS_GENERAL, HARD_CPS_LIMIT_GENERAL


def choose_rewrite_mode(chars_per_sec: float, target_cps: float, hard_cps_limit: float) -> str:
    if chars_per_sec <= target_cps:
        return "balanced"
    if chars_per_sec <= hard_cps_limit:
        return "dub_friendly"
    return "aggressive"


def _detect_style(text: str, src_text: str, has_sql_terms: bool) -> str:
    haystack = f"{text}\n{src_text}"
    if has_sql_terms or re.search(r"\b(?:function|query|table|column|database)\b", haystack, flags=re.IGNORECASE):
        return "technical_spoken"
    if any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in _EXAMPLE_HINTS):
        return "example_explanation"
    if any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in _SUMMARY_HINTS):
        return "summary"
    return "general_spoken"


def analyze_segment(text: str, src_text: str = "", slot_duration: float = 0.0) -> SegmentAnalysis:
    clean = normalize_spaces(text)
    clean_src = normalize_spaces(src_text)
    has_sql_terms = _contains_sql_terms(clean, clean_src)
    has_code_like_sql = _looks_like_sql_code(clean)
    style = _detect_style(clean, clean_src, has_sql_terms)
    target_cps, hard_cps_limit = _style_limits(style)
    chars_per_sec = estimate_chars_per_second(clean, slot_duration)
    needs_rewrite = chars_per_sec > target_cps or needs_adaptation(clean, slot_duration)
    needs_split = chars_per_sec > (hard_cps_limit + 0.8) and bool(re.search(r"[:,;]|\s(?:a|ale|ktorý|ktorá|ktoré)\s", clean, flags=re.IGNORECASE))
    needs_merge = chars_per_sec < max(8.0, target_cps * 0.55) and len(clean) < 24
    priority = "shorten" if chars_per_sec > hard_cps_limit else ("preserve_terms" if has_sql_terms else "natural_flow")
    return SegmentAnalysis(
        slot=slot_duration,
        text_len=len(clean),
        chars_per_sec=chars_per_sec,
        target_cps=target_cps,
        hard_cps_limit=hard_cps_limit,
        has_sql_terms=has_sql_terms,
        has_code_like_sql=has_code_like_sql,
        has_numbers=bool(re.search(r"\d", clean)),
        style=style,
        needs_rewrite=needs_rewrite,
        needs_merge=needs_merge,
        needs_split=needs_split,
        priority=priority,
    )


# ── LLM rewrite prompty ──────────────────────────────────────────────────────

def _build_prompt_slot_aware(
    translated: str,
    src: str,
    analysis: SegmentAnalysis,
    rewrite_mode: str,
    target_chars: int,
) -> tuple[str, str]:
    """Časovo-aware prompt pre balanced / dub-friendly / aggressive rewrite."""
    mode_hint = {
        "balanced": "Jemne uhladť vetu a skráť ju len mierne — ZACHOVAJ takmer všetok obsah.",
        "dub_friendly": (
            "Sprav vetu hovorenú a rytmicky vhodnú pre dabing. "
            "Skráť IBA toľko, koľko treba aby sa zmestila do cieľovej dĺžky. "
            "ZACHOVAJ všetky fakty zo zdrojovej vety."
        ),
        "aggressive": (
            "Skráť vetu na cieľovú dĺžku. ZACHOVAJ všetky kľúčové fakty a technické termíny. "
            "Nikdy nevyhadzuj polovicu obsahu — radšej použi kompaktnejšiu syntaxia. "
            "Text MUSÍ byť aspoň 85% cieľovej dĺžky (nie kratší)."
        ),
    }.get(rewrite_mode, "Sprav vetu stručnú a hovorenú.")

    min_chars = max(20, int(target_chars * 0.85))
    system = (
        "Si expert na prepis technického textu pre slovenský dabing.\n\n"
        "Úloha:\n"
        "Prepíš vetu do prirodzenej hovorenej slovenčiny tak, aby sa zmestila do určeného časového slotu.\n\n"
        "Pravidlá:\n"
        "- Zachovaj význam — všetky fakty, mená, čísla, technické termíny.\n"
        "- Zachovaj technické výrazy presne: SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, NULLIF, TRUE, FALSE, BOOLEAN.\n"
        "- Nepoužívaj fonetické prepisy.\n"
        "- Neprekladaj SQL keywordy.\n"
        "- Rozlišuj presne: ISNULL je funkcia na nahradenie NULL hodnoty, IS NULL je podmienka na kontrolu NULL, NULLIF je funkcia, ktorá pri zhode vracia NULL.\n"
        "- Výstup musí byť hovorený, prirodzený a kompletný (nesmie pôsobiť osekane alebo polovičato).\n"
        "- Nepoužívaj doslovné anglické konštrukcie.\n"
        f"- Cieľová dĺžka: {min_chars}–{target_chars} znakov (DOLNÁ hranica je dôležitá!).\n"
        f"- Ak by si mal výstup pod {min_chars} znakov, radšej použi širšie synonymá alebo plnšie formulácie.\n"
        f"- Režim rewrite: {rewrite_mode}.\n"
        f"- Štýl segmentu: {analysis.style}.\n"
        f"- {_PROSODY_HINTS.get(analysis.style, _PROSODY_HINTS['general_spoken'])}\n"
        f"- {mode_hint}\n"
        "- Vráť iba finálnu vetu. Bez vysvetlenia. Bez úvodzoviek."
    )
    user = (
        f"Slot: {analysis.slot:.2f} sekundy\n"
        f"Max chars per second: {analysis.target_cps:.1f}\n"
        f"Aktuálne chars per second: {analysis.chars_per_sec:.1f}\n"
        f"Priorita: {analysis.priority}\n"
        f"Treba split: {'áno' if analysis.needs_split else 'nie'}\n"
        f"Anglický originál: {src}\n"
        f"Text:\n\"{translated}\""
    )
    return system, user


# ── LLM wrapper ──────────────────────────────────────────────────────────────

def _call_llm(
    system: str,
    user: str,
    llm,
    max_tokens: int = 256,
    temperature: float = 0.10,
) -> Optional[str]:
    """
    Zavolá LLM model. Akceptuje:
      - llama_cpp.Llama inštanciu
      - callable(system, user) → str
    Vráti None pri chybe.
    """
    if llm is None:
        return None
    try:
        if callable(llm) and not hasattr(llm, "create_chat_completion"):
            return llm(system, user)
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = resp["choices"][0]["message"]["content"].strip()
        # Odstrán úvodzovky ak LLM obalil text
        result = result.strip('"\'„"«»')
        return result if result else None
    except Exception as e:
        print(f"[ADAPT] LLM chyba: {e}", flush=True)
        return None


# ── Meta-response detekcia ───────────────────────────────────────────────────

_META_STARTERS = (
    "Rozumiem", "Rozumím", "Dobre,", "Dobre.", "OK,", "Samozrejme",
    "Áno,", "Áno.", "Jasne,", "Jasne.", "Pochopil", "Upravím",
    "Tu je", "Tu je upravený", "Skrátená verzia", "Verzia:",
    "Preformulovaný", "Skrátený text",
)

def _protected_sql_terms_in(text: str) -> set[str]:
    return _protected_sql_terms_in_guard(text)


def _is_valid_rewrite(original: str, rewritten: str, analysis: SegmentAnalysis | None = None) -> bool:
    """Overí či LLM vrátil skutočný rewrite a nie meta-odpoveď."""
    if not rewritten:
        return False
    if rewritten.startswith(_META_STARTERS):
        return False
    # Ak je výstup dlhší ako originál → LLM pridával text namiesto skracovania
    if len(rewritten) > len(original) * 1.05:
        return False
    # Minimálna dĺžka — nechceme jednoznakový výstup
    if len(rewritten) < 4:
        return False
    # Ak je rewrite < 35% pôvodnej dĺžky → LLM odsekol obsah, audio bude pôsobiť osekané.
    # 35% je bezpečný prah — aggressive_rewrite typicky drží 50-70%, pod tým ide o haluc/skip.
    if len(rewritten) < len(original) * 0.35:
        return False
    if analysis and analysis.has_sql_terms:
        if any(re.search(pattern, rewritten, flags=re.IGNORECASE) for pattern in SQL_PHONETIC_PATTERNS):
            return False
        original_terms = _protected_sql_terms_in(original)
        rewritten_terms = _protected_sql_terms_in(rewritten)
        if not original_terms.issubset(rewritten_terms):
            return False
    return True


# ── Hlavná funkcia ────────────────────────────────────────────────────────────

def adapt_for_timing(
    translated:    str,
    src_text:      str          = "",
    slot_duration: float        = 0.0,
    llm                         = None,
    chars_per_sec: float        = SK_CHARS_PER_SEC,
    threshold_soft: float       = THRESHOLD_SOFT,
    threshold_hard: float       = THRESHOLD_HARD,
    threshold_force_dub: float  = THRESHOLD_FORCE_DUB,
    verbose:       bool         = False,
) -> AdaptResult:
    """
    Hlavná vstupná funkcia text adaptation layer.

    Args:
        translated:     preložený text (SK/CS)
        src_text:       zdrojový text (EN) — pre LLM kontext
        slot_duration:  trvanie slotu [s] (seg["end"] - seg["start"])
        llm:            llama_cpp.Llama inštancia alebo callable(system,user)->str
                        Ak None → len odhad, žiadny rewrite
        chars_per_sec:  kalibrácia rýchlosti reči
        threshold_soft: prahový pomer pre prvý rewrite (default 1.08)
        threshold_hard: prahový pomer pre priamy dub-friendly (default 1.20)
        verbose:        loguj detaily

    Returns:
        AdaptResult s finálnym textom a metadátami
    """
    text = normalize_spaces(translated)
    src_text = normalize_spaces(src_text)
    analysis = analyze_segment(text, src_text, slot_duration)
    if analysis.has_sql_terms:
        text = fix_sql_terms(text, src_text=src_text, fallback_text=translated)
        analysis = analyze_segment(text, src_text, slot_duration)
    est = estimate_tts_duration(text, chars_per_sec)
    cps_before = analysis.chars_per_sec

    # Slot ≤ 0 → nedá sa odhadnúť, vráť originál
    if slot_duration <= 0:
        return AdaptResult(text=text, mode="ok_no_change",
                           estimated_duration=est, slot_duration=slot_duration,
                           ratio=0.0, ratio_before=0.0, style=analysis.style,
                           cps_before=cps_before, cps_after=cps_before,
                           target_cps=analysis.target_cps, hard_cps_limit=analysis.hard_cps_limit,
                           analysis=asdict(analysis), notes=["no_slot"])

    # Príliš krátky text (< 20 znakov) → rewrite by nemal zmysel
    if len(text) < 20:
        ratio = est / slot_duration if slot_duration > 0 else 0.0
        return AdaptResult(text=text, mode="ok_no_change",
                           estimated_duration=est, slot_duration=slot_duration,
                           ratio=ratio, ratio_before=ratio, style=analysis.style,
                           cps_before=cps_before, cps_after=cps_before,
                           target_cps=analysis.target_cps, hard_cps_limit=analysis.hard_cps_limit,
                           analysis=asdict(analysis), notes=["short_text"])

    ratio = est / slot_duration if slot_duration > 0 else 0.0
    ratio_before = ratio

    if verbose:
        print(
            f"[ADAPT] slot={slot_duration:.2f}s est={est:.2f}s ratio={ratio:.2f} "
            f"cps={cps_before:.1f}/{analysis.target_cps:.1f} style={analysis.style} "
            f"text={text[:50]!r}",
            flush=True,
        )

    # Nepretečie → vráť originál
    if ratio <= threshold_soft and cps_before <= analysis.target_cps:
        return AdaptResult(text=text, mode="ok_no_change",
                           estimated_duration=est, slot_duration=slot_duration,
                           ratio=ratio, ratio_before=ratio_before, style=analysis.style,
                           cps_before=cps_before, cps_after=cps_before,
                           target_cps=analysis.target_cps, hard_cps_limit=analysis.hard_cps_limit,
                           analysis=asdict(analysis))

    # Bez LLM → ohlás fallback
    if llm is None:
        if verbose:
            print(f"[ADAPT] ratio={ratio:.2f} > {threshold_soft} — no LLM, fallback stretch", flush=True)
        return AdaptResult(text=text, mode="fallback_stretch",
                           estimated_duration=est, slot_duration=slot_duration,
                           ratio=ratio, ratio_before=ratio_before, style=analysis.style,
                           cps_before=cps_before, cps_after=cps_before,
                           target_cps=analysis.target_cps, hard_cps_limit=analysis.hard_cps_limit,
                           analysis=asdict(analysis), notes=["no_llm"])

    target_chars = max(12, int(slot_duration * analysis.target_cps * 0.95))
    rewrite_mode = choose_rewrite_mode(cps_before, analysis.target_cps, analysis.hard_cps_limit)
    if ratio >= threshold_force_dub:
        rewrite_mode = "aggressive"

    notes: list[str] = []
    rewrite_rounds = 0
    attempts: list[tuple[str, int]] = []
    if rewrite_mode == "balanced" and ratio < threshold_hard:
        attempts = [
            ("balanced", target_chars),
            ("dub_friendly", target_chars),
            ("aggressive", max(10, int(target_chars * 0.95))),
        ]
    elif rewrite_mode == "dub_friendly":
        attempts = [
            ("dub_friendly", target_chars),
            ("aggressive", max(10, int(target_chars * 0.95))),
        ]
    else:
        attempts = [
            ("aggressive", target_chars),
            ("aggressive", max(10, int(target_chars * 0.92))),
        ]

    best_text = text
    best_est = est
    best_ratio = ratio
    best_cps = cps_before
    best_mode = rewrite_mode

    for mode_name, mode_target_chars in attempts:
        sys_p, usr_p = _build_prompt_slot_aware(best_text, src_text, analysis, mode_name, mode_target_chars)
        rewritten = _call_llm(
            sys_p,
            usr_p,
            llm,
            max_tokens=256,
            temperature=0.08 if mode_name == "balanced" else 0.05,
        )
        rewrite_rounds += 1

        if not _is_valid_rewrite(text, rewritten, analysis):
            notes.append(f"{mode_name}_rejected")
            continue

        rewritten = normalize_spaces(rewritten or "")
        if analysis.has_sql_terms:
            rewritten = fix_sql_terms(rewritten, src_text=src_text, fallback_text=text)

        new_analysis = analyze_segment(rewritten, src_text, slot_duration)
        new_est = estimate_tts_duration(rewritten, chars_per_sec)
        new_ratio = new_est / slot_duration if slot_duration > 0 else 0.0
        new_cps = new_analysis.chars_per_sec

        if verbose:
            print(
                f"[ADAPT] {mode_name} → est={new_est:.2f}s ratio={new_ratio:.2f} "
                f"cps={new_cps:.1f}/{new_analysis.target_cps:.1f} text={rewritten[:60]!r}",
                flush=True,
            )

        # Lower bound — rewrite ktorý vyprodukuje audio < 75% slotu nechá výrazné ticho
        # (>25% slotu silence padding) a pôsobí "osekane". Skip a skús ďalší mód /
        # fallback na pôvodný (časový stretch je menej rušivý ako dlhé ticho).
        ADAPT_RATIO_FLOOR = 0.75
        if new_ratio < ADAPT_RATIO_FLOOR:
            notes.append(f"{mode_name}_too_short_ratio={new_ratio:.2f}")
            continue

        # Best-tracking: preferuj kandidátov v rozumnom rozsahu (0.75-1.05),
        # cieľ ~0.90 — necháva natural breath gap ale nie veľa ticha.
        IDEAL_RATIO = 0.90
        cur_dist = abs(best_ratio - IDEAL_RATIO) if ADAPT_RATIO_FLOOR <= best_ratio <= threshold_soft else 999.0
        new_dist = abs(new_ratio - IDEAL_RATIO)
        if new_dist < cur_dist or (abs(new_dist - cur_dist) < 1e-6 and new_cps < best_cps):
            best_text = rewritten
            best_est = new_est
            best_ratio = new_ratio
            best_cps = new_cps
            best_mode = mode_name

        if new_ratio <= threshold_soft and new_cps <= new_analysis.target_cps:
            result_mode = (
                "concise_rewrite" if mode_name == "balanced"
                else "dub_friendly_rewrite" if mode_name == "dub_friendly"
                else "aggressive_rewrite"
            )
            return AdaptResult(
                text=rewritten,
                mode=result_mode,
                estimated_duration=new_est,
                slot_duration=slot_duration,
                ratio=new_ratio,
                ratio_before=ratio_before,
                rewrite_rounds=rewrite_rounds,
                rewrite_mode=mode_name,
                style=new_analysis.style,
                cps_before=cps_before,
                cps_after=new_cps,
                target_cps=new_analysis.target_cps,
                hard_cps_limit=new_analysis.hard_cps_limit,
                analysis=asdict(new_analysis),
                notes=notes,
            )

        notes.append(f"{mode_name}_still_over")

    best_analysis = analyze_segment(best_text, src_text, slot_duration)
    if best_text != text:
        notes.append(f"best_mode={best_mode}")

    return AdaptResult(
        text=best_text,
        mode="fallback_stretch",
        estimated_duration=best_est,
        slot_duration=slot_duration,
        ratio=best_ratio,
        ratio_before=ratio_before,
        rewrite_rounds=rewrite_rounds,
        rewrite_mode=best_mode,
        style=best_analysis.style,
        cps_before=cps_before,
        cps_after=best_cps,
        target_cps=best_analysis.target_cps,
        hard_cps_limit=best_analysis.hard_cps_limit,
        analysis=asdict(best_analysis),
        notes=notes,
    )


# ── Batch adaptácia ───────────────────────────────────────────────────────────

def adapt_segments_batch(
    segments:      list[dict],
    llm                       = None,
    src_key:       str        = "text_src",
    tgt_key:       str        = "text",
    verbose:       bool       = False,
    skip_sql:      bool       = True,
    stats_path:    str        = "",   # ak zadaný → uloží AdaptStats do JSON
    chars_per_sec: float      = SK_CHARS_PER_SEC,  # kalibrácia z CPSCalibrator
) -> tuple[list[dict], "AdaptStats"]:
    """
    Aplikuje adapt_for_timing na všetky segmenty.

    Modifikuje segmenty in-place:
      seg[tgt_key]           → adaptovaný text
      seg["text_original"]   → záloha pôvodného prekladu
      seg["adapt_mode"]      → "ok_no_change" / "concise_rewrite" /
                               "dub_friendly_rewrite" / "aggressive_rewrite" /
                               "fallback_stretch" / "meta_response_rejected" / "skip_sql"
      seg["adapt_ratio"]     → ratio estimated/slot po adaptácii
      seg["adapt_ratio_before"] → ratio pred adaptáciou

    Returns:
        (segments, AdaptStats)
    """
    stats = AdaptStats()

    for seg in segments:
        text = (seg.get(tgt_key) or "").strip()
        if not text:
            continue

        # Preskočiť segmenty s AKTUÁLNYM SQL kódom — term preservation > timing
        # Používame _looks_like_sql_query (prísnejší ako is_sql_heavy) — skip len pre skutočný SQL kód,
        # nie pre prirodzené vety ktoré len SPOMÍNAJÚ SQL pojmy (null, SQL, isNull...)
        if skip_sql:
            try:
                from translation import _looks_like_sql_query
                if _looks_like_sql_query(text):
                    seg["adapt_mode"]  = "skip_sql"
                    seg["adapt_ratio"] = 0.0
                    stats.skip_sql += 1
                    continue
            except ImportError:
                pass

        # Preskočiť krátky text — nie je čo skracovať
        if len(text) < 20:
            seg["adapt_mode"]  = "ok_no_change"
            seg["adapt_ratio"] = 0.0
            stats.skip_short += 1
            continue

        slot_dur = float(seg.get("end", 0)) - float(seg.get("start", 0))

        # Preskočiť veľmi krátke sloty — nie je čo skracovať
        if slot_dur < 0.5:
            seg["adapt_mode"]  = "ok_no_change"
            seg["adapt_ratio"] = 0.0
            stats.skip_short += 1
            continue

        src_text = (seg.get(src_key) or seg.get("text_src") or "").strip()

        result = adapt_for_timing(
            translated=text, src_text=src_text,
            slot_duration=slot_dur, llm=llm, verbose=verbose,
            chars_per_sec=chars_per_sec,
        )

        seg["text_original"]      = text
        seg[tgt_key]              = result.text
        seg["adapt_mode"]         = result.mode
        seg["adapt_ratio"]        = round(result.ratio, 3)
        seg["adapt_ratio_before"] = round(result.ratio_before, 3)
        seg["adapt_style"]        = result.style
        seg["adapt_rewrite_mode"] = result.rewrite_mode
        seg["adapt_cps_before"]   = round(result.cps_before, 2)
        seg["adapt_cps_after"]    = round(result.cps_after, 2)
        seg["adapt_target_cps"]   = round(result.target_cps, 2)
        seg["adapt_hard_cps_limit"] = round(result.hard_cps_limit, 2)
        seg["adapt_analysis"]     = result.analysis
        seg["adapt_has_sql_terms"] = bool(result.analysis.get("has_sql_terms"))
        seg["adapt_preserve_sql_terms"] = bool(
            result.analysis.get("has_sql_terms") and not result.analysis.get("has_code_like_sql")
        )
        seg["adapt_notes"]        = result.notes

        stats.record(result)

        if result.mode not in ("ok_no_change",) and (verbose or result.ratio_before > 1.3):
            print(
                f"[ADAPT] {seg.get('start',0):.1f}s  "
                f"{result.mode}  "
                f"ratio {result.ratio_before:.2f}→{result.ratio:.2f}  "
                f"cps {result.cps_before:.1f}→{result.cps_after:.1f}/{result.target_cps:.1f}  "
                f"mode={result.rewrite_mode} rounds={result.rewrite_rounds}",
                flush=True,
            )

    print(stats.summary(), flush=True)

    if stats_path:
        p = Path(stats_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ADAPT] Stats uložené: {p}", flush=True)

    return segments, stats


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Text adaptation layer test")
    parser.add_argument("--text",     required=True, help="Preložený SK text")
    parser.add_argument("--src",      default="",    help="Zdrojový EN text")
    parser.add_argument("--slot",     type=float, default=3.0, help="Slot duration [s]")
    parser.add_argument("--llm",      default="",    help="Cesta k GGUF modelu")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    llm_instance = None
    if args.llm:
        from llama_cpp import Llama
        llm_instance = Llama(model_path=args.llm, n_ctx=2048, n_gpu_layers=56, verbose=False)

    result = adapt_for_timing(
        translated=args.text, src_text=args.src,
        slot_duration=args.slot, llm=llm_instance, verbose=True,
    )

    est_orig = estimate_tts_duration(args.text)
    print(f"\n{'─'*60}")
    print(f"Originál   : {args.text}")
    print(f"Est. dur.  : {est_orig:.2f}s  slot: {args.slot:.2f}s  ratio: {est_orig/args.slot:.2f}")
    print(f"{'─'*60}")
    print(f"Výsledok   : {result.text}")
    print(f"Mód        : {result.mode}")
    print(f"Est. dur.  : {result.estimated_duration:.2f}s  ratio: {result.ratio:.2f}")
    print(f"Style      : {result.style}")
    print(f"CPS        : {result.cps_before:.1f} → {result.cps_after:.1f}  target={result.target_cps:.1f}")
    print(f"Rounds     : {result.rewrite_rounds}")
    if result.notes:
        print(f"Notes      : {result.notes}")
