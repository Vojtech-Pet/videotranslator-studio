#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phonetic_guard_rewrite.py
=========================
Post-processing pipeline pre debug alebo segments JSON:
  1. Ochrana SQL/technických termínov (PROTECTED_REGEX)
  2. Bezpečná normalizácia čísel (voliteľná)
  3. Phonetic vrstva — preskočí, ak sú protected terms
  4. Rewrite layer — heuristické skrátenie pre príliš dlhé segmenty (voliteľné)
  5. Guard: adapt_ratio >= 1.40 alebo chars_per_sec >= 18.5 → rewrite

Prepínače:
    PREFER_TEXT_ORIGINAL      — True: zdrojom je text_original/adapted pred translated/tts_input
    ENABLE_REWRITE            — True: aktivuje rewrite+shorten pravidlá
    ENABLE_NUMBER_NORMALIZATION — True: číslice → slová podľa NUMBER_WORD_MAP

Podporuje oba formáty:
    [{...}, ...]                   (debug_list)
    {"segments": [{...}, ...]}     (segments_dict)

Použitie:
    python3 phonetic_guard_rewrite.py
    python3 phonetic_guard_rewrite.py --input moj.json
    python3 phonetic_guard_rewrite.py --input moj.json --output opraveny.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =========================================================
# KONFIG
# =========================================================

DEFAULT_INPUT = "sql_part_007_part001_sk_tts_debug.json"

PREFER_TEXT_ORIGINAL       = True   # True: text_original pred translated
ENABLE_REWRITE             = True   # True: zapne rewrite + shorten pravidlá
ENABLE_NUMBER_NORMALIZATION = True   # True: číslice → slová

MAX_ADAPT_RATIO:   float = 1.40
MAX_CHARS_PER_SEC: float = 18.5

# =========================================================
# Ochrana technických termínov
# =========================================================

PROTECTED_REGEX = [
    r"\bSQL\b",
    r"\bNULL\b",
    r"\bIS\s+NULL\b",
    r"\bIS\s+NOT\s+NULL\b",
    r"\bISNULL\b",
    r"\bCOALESCE\b",
    r"\bTRUE\b",
    r"\bFALSE\b",
    r"\bBOOLEAN\b",
]

# Bezpečné fonetické náhrady (NIE SQL keywordy)
PHONETIC_MAP: Dict[str, str] = {
    # "GPU": "džípijú",
    # "CPU": "sípijú",
}

NUMBER_WORD_MAP: Dict[str, str] = {
    "40": "štyridsať",
    "0":  "nula",
}

# =========================================================
# Rewrite heuristiky — skrátenie dlhých segmentov
# =========================================================

REWRITE_REPLACEMENTS = [
    (r"\bV poriadku, priatelia, takže teraz\b",                               "Teraz"),
    (r"\burobíme hlboký ponor do\b",                                          "sa pozrieme na"),
    (r"\bO tom, ako zaobchádzať s\b",                                         "Ako pracovať s"),
    (r"\bv našich údajoch\b",                                                 "v dátach"),
    (r"\bteraz v niektorých scenároch\b",                                     "v niektorých prípadoch"),
    (r"\bA chceli by sme ju odstrániť a nahradiť\b",                          "A chceli by sme ju nahradiť"),
    (r"\bAby sme to urobili v\b",                                             "Na to v"),
    (r"\bprvá volaná je\b",                                                   "prvá je"),
    (r"\bNazývané\b",                                                         ""),
    (r"\bv ktorom máme hodnotu vo vnútri\b",                                  "kde máme hodnotu v"),
    (r"\bnášho stola\b",                                                      "našej tabuľky"),
    (r"\burobiť ako nulový\b",                                                "nastaviť na NULL"),
    (r"\brobíme presný opak\b",                                               "robíme opak"),
    (r"\bPoďme si dať príklad\.",                                             "Pozrime sa na príklad."),
    (r"\bMôžeme ísť a použiť\b",                                             "Môžeme použiť"),
    (r"\bTakže kontrolujeme NULL vo vnútri\.",                                "Teda kontrolujeme, či je tam NULL."),
    (r"\bA ak SQL narazí na nejakú hodnotu NULL,\b",                          "Ak SQL narazí na hodnotu NULL,"),
    (r"\bodíde a nahradí ju\b",                                               "nahradí ju"),
    (r"\bTakže toto bude ako predvolená hodnota pre hodnoty NULL\.",          "To bude predvolená hodnota pre NULL."),
    (r"\bTakže\b",                                                            ""),
]

FILLER_PATTERNS = [
    r"\bako vy\?\b",
    r"\bchoďte ju vymeniť\b",
]

SHORTEN_RULES = [
    (r"\bteraz\b",          ""),
    (r"\btakže\b",          ""),
    (r"\bv našich\b",       ""),
    (r"\bnašich\b",         ""),
    (r"\bveľmi\b",          ""),
    (r"\bskutočne\b",       ""),
    (r"\bpoďme si\b",       "poďme"),
    (r"\bmôžeme\b",         "vieme"),
    (r"\bpozrime sa na\b",  "príklad"),
]

# =========================================================
# Pomocné funkcie
# =========================================================

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_punctuation(text: str) -> str:
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", text)
    return normalize_spaces(text)


def find_protected_terms(text: str) -> List[str]:
    found = []
    for pattern in PROTECTED_REGEX:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append(match.group(0))
    seen: set = set()
    out = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def contains_protected_term(text: str) -> bool:
    return bool(find_protected_terms(text))


def find_protected_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for pattern in PROTECTED_REGEX:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged: List[List[int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(s, e) for s, e in merged]


def is_inside_spans(index: int, spans: List[Tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def replace_outside_protected(text: str, replacements: Dict[str, str]) -> str:
    if not replacements:
        return text
    spans = find_protected_spans(text)
    result = []
    i = 0
    while i < len(text):
        if is_inside_spans(i, spans):
            for start, end in spans:
                if start <= i < end:
                    result.append(text[i:end])
                    i = end
                    break
            continue
        replaced = False
        for src, dst in replacements.items():
            if text.startswith(src, i):
                overlap = any(i < end and i + len(src) > start for start, end in spans)
                if not overlap:
                    result.append(dst)
                    i += len(src)
                    replaced = True
                    break
        if not replaced:
            result.append(text[i])
            i += 1
    return "".join(result)


def safe_number_normalization(text: str) -> str:
    if not ENABLE_NUMBER_NORMALIZATION:
        return text
    def repl(m: re.Match) -> str:
        return NUMBER_WORD_MAP.get(m.group(0), m.group(0))
    return normalize_spaces(re.sub(r"\b\d+\b", repl, text))


def apply_phonetic_layer(text: str) -> str:
    if contains_protected_term(text):
        return text
    return normalize_spaces(replace_outside_protected(text, PHONETIC_MAP))


def apply_rewrite_rules(text: str) -> str:
    out = text
    for pattern, replacement in REWRITE_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    for pattern in FILLER_PATTERNS:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhodnotu NULL\b", "NULL hodnotu", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhodnoty NULL\b", "NULL hodnoty", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnuly\b",         "NULL hodnoty", out, flags=re.IGNORECASE)
    out = clean_punctuation(normalize_spaces(out))
    return (out[0].upper() + out[1:]) if out else out


def shorten_text_iteratively(text: str) -> str:
    out = text
    for pattern, replacement in SHORTEN_RULES:
        candidate = clean_punctuation(normalize_spaces(
            re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        ))
        if len(candidate) < len(out):
            out = candidate
    return out


# =========================================================
# Výber zdrojového textu
# =========================================================

def get_base_text(seg: Dict[str, Any]) -> str:
    """
    Vyberie zdrojový text podľa PREFER_TEXT_ORIGINAL.

    PREFER_TEXT_ORIGINAL=True  → adapted → text_original → translated → text → tts_input
    PREFER_TEXT_ORIGINAL=False → adapted → translated → text → text_original → tts_input
    """
    if PREFER_TEXT_ORIGINAL:
        priority = ["adapted", "text_original", "translated", "text", "tts_input"]
    else:
        priority = ["adapted", "translated", "text", "text_original", "tts_input"]

    for key in priority:
        val = seg.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_spaces(val)
    return ""


def get_slot(seg: Dict[str, Any]) -> float:
    slot = seg.get("slot")
    if isinstance(slot, (int, float)) and slot > 0:
        return float(slot)
    start = seg.get("start")
    end   = seg.get("end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return float(end - start)
    return 0.0


def get_adapt_ratio(seg: Dict[str, Any]) -> float:
    for key in ("adapt_ratio_after", "adapt_ratio_before", "adapt_ratio"):
        val = seg.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 1.0


def estimate_chars_per_sec(text: str, slot: float) -> float:
    return round(len(text) / slot, 1) if slot > 0 else 999.0


def _needs_rewrite(seg: Dict[str, Any], text: str) -> bool:
    slot = get_slot(seg)
    return (
        get_adapt_ratio(seg) >= MAX_ADAPT_RATIO or
        estimate_chars_per_sec(text, slot) >= MAX_CHARS_PER_SEC
    )


# =========================================================
# Spracovanie segmentu
# =========================================================

def process_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    base_text = get_base_text(seg)

    protected_terms_found = find_protected_terms(base_text)
    phonetic_skipped = bool(protected_terms_found)

    after_normalize = safe_number_normalization(base_text)
    after_phonetic  = apply_phonetic_layer(after_normalize)

    rewritten       = after_phonetic
    rewrite_applied = False
    rewrite_reason: List[str] = []

    if ENABLE_REWRITE and _needs_rewrite(seg, rewritten):
        rewrite_applied = True
        rewrite_reason.append("ratio_or_cps")
        rewritten = apply_rewrite_rules(rewritten)

    if ENABLE_REWRITE and _needs_rewrite(seg, rewritten):
        rewrite_reason.append("iterative_shorten")
        rewritten = shorten_text_iteratively(rewritten)

    final_text = clean_punctuation(normalize_spaces(rewritten))
    slot = get_slot(seg)

    seg["after_normalize"]        = after_normalize
    seg["after_phonetic"]         = after_phonetic
    seg["after_sk_fixes"]         = final_text
    seg["tts_input"]              = final_text
    seg["phonetic_skipped"]       = phonetic_skipped
    seg["protected_terms_found"]  = protected_terms_found
    seg["rewrite_applied"]        = rewrite_applied
    seg["rewrite_reason"]         = rewrite_reason
    seg["slot"]                   = round(slot, 3) if slot > 0 else slot
    seg["final_chars_per_sec"]    = estimate_chars_per_sec(final_text, slot)

    return seg


# =========================================================
# I/O
# =========================================================

def load_json(path: Path) -> Tuple[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return "debug_list", data
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return "segments_dict", data
    raise ValueError("Nepodporovaný JSON formát. Očakávam list alebo {segments: [...]}")


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phonetic guard + rewrite pre debug/segments JSON")
    parser.add_argument("--input",  default=DEFAULT_INPUT,
                        help=f"Vstupný JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=None,
                        help="Výstupný JSON (default: <vstup>.fixed.json)")
    cli = parser.parse_args()

    input_path  = Path(cli.input)
    output_path = Path(cli.output) if cli.output else input_path.with_suffix(".fixed.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Vstupný súbor neexistuje: {input_path}")

    fmt, data = load_json(input_path)

    phonetic_skipped_count = 0
    rewrite_count          = 0
    fixed: List[Dict] = []

    segments = data if fmt == "debug_list" else data["segments"]
    for seg in segments:
        new_seg = process_segment(seg)
        fixed.append(new_seg)
        if new_seg.get("phonetic_skipped"):
            phonetic_skipped_count += 1
        if new_seg.get("rewrite_applied"):
            rewrite_count += 1

    output_data = fixed if fmt == "debug_list" else {**data, "segments": fixed}
    save_json(output_path, output_data)

    print("Hotovo.")
    print(f"Formát                    : {fmt}")
    print(f"Načítaných segmentov      : {len(fixed)}")
    print(f"Preskočená fonetizácia    : {phonetic_skipped_count}")
    print(f"Rewrite použitý           : {rewrite_count}")
    print(f"PREFER_TEXT_ORIGINAL      = {PREFER_TEXT_ORIGINAL}")
    print(f"ENABLE_REWRITE            = {ENABLE_REWRITE}")
    print(f"ENABLE_NUMBER_NORMALIZATION = {ENABLE_NUMBER_NORMALIZATION}")
    print(f"Výstup                    : {output_path}")


if __name__ == "__main__":
    main()
