"""
Level 4 audio feedback helpers.

Post-TTS timing audit, retry heuristics, and merge/split helpers for
slot-aware dubbing pipelines.
"""

from __future__ import annotations

import re
from pathlib import Path

import soundfile as sf

from phonetic_guard import shorten_text
from text_adaptation import estimate_chars_per_second, fix_sql_terms, normalize_spaces

OK_DELTA_S = 0.20
TOO_SHORT_DELTA_S = -0.40
DEFAULT_TARGET_CPS = 16.0
DEFAULT_HARD_CPS_LIMIT = 17.0

# Prahy pre too_short retry
TOO_SHORT_PAUSE_MAX_S = 0.80   # do tohto → len pause injection (ticho na konci)
TOO_SHORT_REWRITE_MIN_S = 0.80  # od tohto → LLM rewrite s požiadavkou na dlhší text

# Automatická segmentácia dlhých slotov
AUTO_SPLIT_THRESHOLD_S = 10.0   # slot dlhší než toto → automaticky rozbiť

_RETRY_SHORTEN_RULES = [
    (r"\bV poriadku,\s*", ""),
    (r"\bpriatelia,?\s*", ""),
    (r"\btakže\b", ""),
    (r"\bteraz\b", ""),
    (r"\bpoďme si\b", "poďme"),
    (r"\burobíme\b", ""),
    (r"\bhlboký ponor do\b", "pozrieme sa na"),
    (r"\bsme schopní\b", "vieme"),
    (r"\bmáme k dispozícii\b", "máme"),
    (r"\bktoré slúžia na\b", "na"),
    (r"\bv podstate\b", ""),
    (r"\bna to máme\b", "máme"),
]

_SPLIT_PATTERNS = (
    r":",
    r";",
    r",",
    r"\s+a\s+",
    r"\s+ale\s+",
    r"\s+takže\s+",
)


def wav_duration_seconds(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def audit_duration(slot: float, wav_path: str | Path) -> dict:
    duration = wav_duration_seconds(wav_path)
    delta = round(duration - float(slot or 0.0), 3)

    if abs(delta) <= OK_DELTA_S:
        status = "ok"
    elif delta > OK_DELTA_S:
        status = "too_long"
    elif delta < TOO_SHORT_DELTA_S:
        status = "too_short"
    else:
        status = "ok"

    return {
        "slot": round(float(slot or 0.0), 3),
        "wav_duration": round(duration, 3),
        "delta": delta,
        "status": status,
    }


def should_retry_from_audit(audit: dict, *, attempt_idx: int = 0, max_attempts: int = 1) -> bool:
    return attempt_idx < max_attempts and audit.get("status") == "too_long"


def should_retry_too_short(audit: dict, *, attempt_idx: int = 0, max_attempts: int = 1) -> bool:
    """Retry keď TTS vygenerovalo príliš krátke audio pre daný slot."""
    return attempt_idx < max_attempts and audit.get("status") == "too_short"


def too_short_needs_rewrite(audit: dict) -> bool:
    """
    Vráti True ak rozdiel je veľký → treba LLM rewrite.
    Vráti False ak stačí pause injection.
    """
    delta = float(audit.get("delta") or 0.0)  # záporné = TTS kratšie než slot
    return delta < -TOO_SHORT_REWRITE_MIN_S


def expand_text_hint(text: str, slot: float, tts_dur: float) -> str:
    """
    Vráti hint pre LLM rewrite — koľko sekúnd chýba a aký text treba vygenerovať.
    Používa sa ako system prompt doplnok pri too_short rewrite.
    """
    missing = round(slot - tts_dur, 2)
    target_chars = int(slot * DEFAULT_TARGET_CPS)
    current_chars = len(text)
    return (
        f"Text je príliš krátky pre slot {slot:.1f}s (TTS trvalo len {tts_dur:.1f}s, chýba ~{missing:.1f}s). "
        f"Prepíš ho tak aby mal ~{target_chars} znakov (teraz má {current_chars}). "
        f"Zachovaj presný význam, prirodzený slovenský prejav a SQL termíny."
    )


def should_merge(
    seg1: dict,
    seg2: dict,
    *,
    max_gap: float = 0.25,
    target_cps: float = DEFAULT_TARGET_CPS,
    text_key: str = "tts_input",
) -> bool:
    gap = float(seg2["start"]) - float(seg1["end"])
    if gap > max_gap:
        return False
    combined_slot = float(seg2["end"]) - float(seg1["start"])
    if combined_slot <= 0.05:
        return False
    combined_text = f"{seg1.get(text_key, '')} {seg2.get(text_key, '')}".strip()
    cps = estimate_chars_per_second(combined_text, combined_slot)
    return cps <= target_cps


def find_split_index(text: str) -> int | None:
    clean = normalize_spaces(text)
    if len(clean) < 32:
        return None

    for pattern in _SPLIT_PATTERNS:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            idx = match.start()
            left = clean[:idx].strip(" ,:;")
            right = clean[match.end():].strip(" ,:;")
            if len(left) >= 12 and len(right) >= 12:
                return idx
    return None


def should_split(
    text: str,
    slot: float,
    *,
    hard_cps_limit: float = DEFAULT_HARD_CPS_LIMIT,
) -> bool:
    return estimate_chars_per_second(text, slot) > hard_cps_limit and find_split_index(text) is not None


def tighten_text_for_retry(
    text: str,
    slot: float,
    *,
    target_cps: float = DEFAULT_TARGET_CPS,
    preserve_sql_terms: bool = False,
) -> str:
    """
    Deterministic shortening for one post-TTS retry.

    Keeps protected SQL terms exact when requested.
    """
    base = normalize_spaces(text)
    if not base:
        return base
    if preserve_sql_terms:
        base = fix_sql_terms(base)

    candidates: list[str] = [base]

    shortened = shorten_text(base)
    if shortened:
        candidates.append(normalize_spaces(shortened))

    aggressive = shortened or base
    for pattern, replacement in _RETRY_SHORTEN_RULES:
        aggressive = re.sub(pattern, replacement, aggressive, flags=re.IGNORECASE)
    aggressive = normalize_spaces(aggressive)
    if aggressive:
        candidates.append(aggressive)

    best = base
    best_cps = estimate_chars_per_second(base, slot)
    for candidate in candidates:
        candidate = normalize_spaces(candidate)
        if preserve_sql_terms:
            candidate = fix_sql_terms(candidate)
        if not candidate or len(candidate) < 4:
            continue
        cps = estimate_chars_per_second(candidate, slot)
        if cps <= target_cps and len(candidate) < len(best):
            return candidate
        if len(candidate) < len(best) or (len(candidate) == len(best) and cps < best_cps):
            best = candidate
            best_cps = cps

    return best
