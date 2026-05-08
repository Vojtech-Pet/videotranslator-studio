"""
duration_planner.py — Slot-aware duration planning vrstva pre Chatterbox TTS.

Architektúra:
    plan(source_text, translated_text, slot_dur, content_type) -> TargetSpec
    select_preset(slot_dur, content_type)                      -> Preset
    evaluate(wav_path, slot_dur, spec)                         -> GuardResult
    decide(guard_result)                                       -> Action
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

# ── Konštanty ─────────────────────────────────────────────────────────────────

# Priemerná SK rečová hustota znakov/s podľa content type
_CPS: dict[str, float] = {
    "sql":       12.5,
    "technical": 12.5,
    "general":   13.5,
    "fast":      15.0,
}
_CPS_DEFAULT = 13.0

# Slová/s (pre target_words)
_WPS_DEFAULT = 2.2

# Tolerancie pre guard (sekundy)
ACCEPT_DELTA_S      =  0.20   # ±0.20s → accept
STRETCH_MAX_DELTA_S =  0.80   # ±0.20–0.80s → stretch
REWRITE_MAX_DELTA_S =  2.00   # ±0.80–2.00s → rewrite expand/shrink
# nad 2.00s → split_or_retry

# Stretch limity
STRETCH_UP_DEFAULT   = 1.15   # max zrýchlenie (atempo > 1)
STRETCH_DOWN_DEFAULT = 0.88   # max spomalenie (atempo < 1)

# Minimálne vyplnenie slotu
MIN_FILL_RATIO_DEFAULT = 0.70  # TTS musí pokryť aspoň 70% slotu

# ── Dátové typy ───────────────────────────────────────────────────────────────

class RewriteMode(str, Enum):
    KEEP    = "keep"
    SHRINK  = "shrink"
    EXPAND  = "expand"
    SPLIT   = "split"


@dataclass
class TargetSpec:
    slot_dur: float
    target_chars: int
    target_words: int
    min_fill_ratio: float
    max_stretch_up: float    # atempo cap pri skrátení (> 1 = rýchlejšie)
    max_stretch_down: float  # atempo cap pri predĺžení (< 1 = pomalšie)
    rewrite_mode: RewriteMode
    source_cps: float        # hustota originálu (chars/s)
    draft_cps: float         # hustota draftu (chars/s)
    content_type: str


@dataclass
class Preset:
    exaggeration: float
    cfg_weight: float
    temperature: float
    label: str = ""


@dataclass
class GuardResult:
    """Výsledok merania — čistý popis situácie, bez rozhodnutia."""
    slot_dur: float
    wav_dur: float
    delta: float            # wav_dur - slot_dur  (záporné = príliš krátky)
    fill_ratio: float       # wav_dur / slot_dur
    stretch_ratio: float    # koľko by treba atempo (wav_dur / slot_dur)
    silence_tail_s: float   # odhadovaný chvost ticha
    # Technický guard
    tech_ok: bool
    tech_issue: Literal["none", "too_short", "too_long", "low_fill", "silence_tail"] = "none"
    # Kvalitatívny guard (rozšíriteľné)
    qual_ok: bool = True
    qual_issue: str = ""


class Action(str, Enum):
    ACCEPT            = "accept"
    STRETCH           = "stretch"           # jemný atempo
    REWRITE_EXPAND    = "rewrite_expand"    # text treba predĺžiť
    REWRITE_SHRINK    = "rewrite_shrink"    # text treba skrátiť
    SPLIT_OR_RETRY    = "split_or_retry"    # segment príliš dlhý / text nevyhovujúci
    FALLBACK_ACCEPT   = "fallback_accept"   # po všetkých pokusoch — akceptuj najlepší


# ── Hlavné funkcie ─────────────────────────────────────────────────────────────

def plan(
    source_text: str,
    translated_text: str,
    slot_dur: float,
    content_type: str = "general",
) -> TargetSpec:
    """
    Vypočíta TargetSpec pre daný slot.

    Parametre:
        source_text     — pôvodný EN text (pre hustotu originálu)
        translated_text — SK draft (môže byť prázdny ak ešte neexistuje)
        slot_dur        — dĺžka časového slotu v sekundách
        content_type    — "sql" | "technical" | "general" | "fast"
    """
    cps = _CPS.get(content_type, _CPS_DEFAULT)
    slot_safe = max(slot_dur, 0.1)

    target_chars = max(5, int(slot_safe * cps))
    target_words = max(1, int(slot_safe * _WPS_DEFAULT))

    src_chars = len(source_text.strip())
    draft_chars = len(translated_text.strip())

    source_cps = src_chars / slot_safe if src_chars > 0 else cps
    draft_cps  = draft_chars / slot_safe if draft_chars > 0 else 0.0

    # Rozhodnutie o rewrite_mode podľa porovnania draftu s cieľom
    draft_ratio = draft_chars / target_chars if target_chars > 0 else 1.0
    if draft_ratio < 0.60:
        rewrite_mode = RewriteMode.EXPAND
    elif draft_ratio > 1.35:
        rewrite_mode = RewriteMode.SHRINK
    elif slot_dur > 10.0 and draft_chars < target_chars * 0.75:
        # Dlhý slot s krátkym textom → split môže byť lepší ako expand
        rewrite_mode = RewriteMode.SPLIT
    else:
        rewrite_mode = RewriteMode.KEEP

    # Stretch limity — pre dlhé segmenty trochu konzervatívnejšie
    if slot_dur > 8.0:
        max_up   = min(STRETCH_UP_DEFAULT,   1.12)
        max_down = max(STRETCH_DOWN_DEFAULT, 0.90)
    elif slot_dur < 3.0:
        max_up   = STRETCH_UP_DEFAULT
        max_down = STRETCH_DOWN_DEFAULT
    else:
        max_up   = STRETCH_UP_DEFAULT
        max_down = STRETCH_DOWN_DEFAULT

    # Min fill ratio — pri krátkych slotoch trochu voľnejšie
    min_fill = MIN_FILL_RATIO_DEFAULT if slot_dur >= 3.0 else 0.60

    return TargetSpec(
        slot_dur=round(slot_safe, 3),
        target_chars=target_chars,
        target_words=target_words,
        min_fill_ratio=min_fill,
        max_stretch_up=max_up,
        max_stretch_down=max_down,
        rewrite_mode=rewrite_mode,
        source_cps=round(source_cps, 2),
        draft_cps=round(draft_cps, 2),
        content_type=content_type,
    )


def select_preset(slot_dur: float, content_type: str = "general") -> Preset:
    """
    Vráti odporúčaný Chatterbox preset pre daný slot.

    Filozofia:
        <3s  → kompaktný (vyšší cfg_weight, nižší temp, nižší exaggeration)
        3–8s → balanced
        >8s  → mierne uvoľnenejší, ale NIE agresívne
                (dlhé segmenty sa môžu rozpadnúť pri príliš voľnom nastavení)
    """
    if slot_dur < 3.0:
        return Preset(
            exaggeration=0.40,
            cfg_weight=0.70,
            temperature=0.70,
            label="compact",
        )
    elif slot_dur <= 8.0:
        return Preset(
            exaggeration=0.50,
            cfg_weight=0.60,
            temperature=0.78,
            label="balanced",
        )
    else:
        # Dlhý slot — mierne uvoľnenejší, ale stále konzervatívny
        return Preset(
            exaggeration=0.48,
            cfg_weight=0.55,
            temperature=0.82,
            label="relaxed",
        )


def evaluate(
    wav_path: str | Path,
    slot_dur: float,
    spec: TargetSpec | None = None,
) -> GuardResult:
    """
    Meria TTS výstup a vracia GuardResult — čistý popis, bez rozhodnutia.
    (Rozhodnutie robí decide().)
    """
    import soundfile as sf

    try:
        info = sf.info(str(wav_path))
        wav_dur = info.frames / float(info.samplerate)
    except Exception:
        wav_dur = 0.0

    slot_safe   = max(slot_dur, 0.01)
    delta       = round(wav_dur - slot_safe, 3)
    fill_ratio  = round(wav_dur / slot_safe, 4)
    stretch_r   = round(wav_dur / slot_safe, 4) if wav_dur > 0 else 0.0

    # Odhadni silence tail — jednoducho: ak je wav_dur < slot * 0.85 → chvost
    silence_tail = max(0.0, slot_safe - wav_dur) if wav_dur < slot_safe else 0.0

    min_fill = spec.min_fill_ratio if spec else MIN_FILL_RATIO_DEFAULT

    # Technický guard
    tech_ok = True
    tech_issue: Literal["none", "too_short", "too_long", "low_fill", "silence_tail"] = "none"

    if wav_dur <= 0.05:
        tech_ok = False
        tech_issue = "too_short"
    elif fill_ratio < min_fill:
        tech_ok = False
        tech_issue = "low_fill"
    elif silence_tail > 1.5:
        tech_ok = False
        tech_issue = "silence_tail"
    elif delta > ACCEPT_DELTA_S:
        tech_ok = False
        tech_issue = "too_long"
    elif delta < -ACCEPT_DELTA_S:
        tech_ok = False
        tech_issue = "too_short"

    return GuardResult(
        slot_dur=round(slot_safe, 3),
        wav_dur=round(wav_dur, 3),
        delta=delta,
        fill_ratio=fill_ratio,
        stretch_ratio=stretch_r,
        silence_tail_s=round(silence_tail, 3),
        tech_ok=tech_ok,
        tech_issue=tech_issue,
    )


def decide(guard: GuardResult, spec: TargetSpec | None = None) -> Action:
    """
    Rozhodnutie na základe GuardResult.

    Stavový automat:
        accept
          ↓
        small_miss  → stretch
          ↓
        medium_miss → rewrite_expand / rewrite_shrink
          ↓
        large_miss  → split_or_retry
          ↓
        (po max pokusoch) → fallback_accept
    """
    if guard.tech_ok and guard.qual_ok:
        return Action.ACCEPT

    abs_delta = abs(guard.delta)

    # Low fill alebo silence tail → expand / split
    if guard.tech_issue in ("low_fill", "silence_tail"):
        if guard.slot_dur > 8.0:
            return Action.SPLIT_OR_RETRY
        return Action.REWRITE_EXPAND

    # Too short
    if guard.tech_issue == "too_short":
        if abs_delta <= STRETCH_MAX_DELTA_S:
            # Skontroluj či je stretch v limitu
            max_down = spec.max_stretch_down if spec else STRETCH_DOWN_DEFAULT
            if guard.stretch_ratio >= max_down:
                return Action.STRETCH
            return Action.REWRITE_EXPAND
        elif abs_delta <= REWRITE_MAX_DELTA_S:
            return Action.REWRITE_EXPAND
        else:
            return Action.SPLIT_OR_RETRY

    # Too long
    if guard.tech_issue == "too_long":
        if abs_delta <= STRETCH_MAX_DELTA_S:
            max_up = spec.max_stretch_up if spec else STRETCH_UP_DEFAULT
            if guard.stretch_ratio <= max_up:
                return Action.STRETCH
            return Action.REWRITE_SHRINK
        elif abs_delta <= REWRITE_MAX_DELTA_S:
            return Action.REWRITE_SHRINK
        else:
            return Action.SPLIT_OR_RETRY

    return Action.FALLBACK_ACCEPT
