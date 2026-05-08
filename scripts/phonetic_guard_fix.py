#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phonetic_guard_fix.py
=====================
Opraví phonetic polia v debug JSON súbore — ochráni SQL/technické termíny.

Načíta:  sql_part_007_part001_sk_tts_debug.json  (alebo ľubovoľný debug JSON)
Uloží:   <vstup>.fixed.json

Použitie:
    python3 phonetic_guard_fix.py
    python3 phonetic_guard_fix.py --input moj_debug.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

# =========================
# Nastavenia
# =========================

DEFAULT_INPUT  = "sql_part_007_part001_sk_tts_debug.json"

# Technické termíny, ktoré sa NESMÚ meniť foneticky
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

# Jednoduchá ukážka normalizácie čísel
NUMBER_WORD_MAP = {
    "40": "štyridsať",
    "0":  "nula",
}


# =========================
# Pomocné funkcie
# =========================

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_protected_terms(text: str) -> List[str]:
    found = []
    for pattern in PROTECTED_REGEX:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append(match.group(0))
    seen: set = set()
    out = []
    for item in found:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            out.append(item)
    return out


def contains_protected_term(text: str) -> bool:
    return len(find_protected_terms(text)) > 0


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


def safe_number_normalization(text: str) -> str:
    def repl(match: re.Match) -> str:
        return NUMBER_WORD_MAP.get(match.group(0), match.group(0))
    text = re.sub(r"\b\d+\b", repl, text)
    return normalize_spaces(text)


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


def apply_phonetic_layer(text: str) -> str:
    if contains_protected_term(text):
        return text
    text = replace_outside_protected(text, PHONETIC_MAP)
    return normalize_spaces(text)


def process_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    # preferuje "adapted" → "translated" → "tts_input"
    base_text = (
        seg.get("adapted") or
        seg.get("translated") or
        seg.get("tts_input") or ""
    )
    base_text = normalize_spaces(base_text)

    protected_terms_found = find_protected_terms(base_text)
    phonetic_skipped = len(protected_terms_found) > 0

    after_normalize = safe_number_normalization(base_text)
    after_phonetic  = apply_phonetic_layer(after_normalize)

    seg["after_normalize"]        = after_normalize
    seg["after_phonetic"]         = after_phonetic
    seg["after_sk_fixes"]         = after_phonetic
    seg["tts_input"]              = after_phonetic
    seg["phonetic_skipped"]       = phonetic_skipped
    seg["protected_terms_found"]  = protected_terms_found

    return seg


def main() -> None:
    parser = argparse.ArgumentParser(description="Opraví phonetic polia v debug JSON")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Vstupný JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=None,
                        help="Výstupný JSON (default: <vstup>.fixed.json)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".fixed.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Vstupný súbor neexistuje: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # podpora oboch formátov: priamy list alebo {"segments": [...]}
    if isinstance(data, dict) and "segments" in data:
        segments   = data["segments"]
        wrap_dict  = True
    elif isinstance(data, list):
        segments   = data
        wrap_dict  = False
    else:
        raise ValueError("Neznámy formát JSON — očakávam list alebo {segments: [...]}")

    skipped_count = 0
    fixed = []
    for seg in segments:
        new_seg = process_segment(seg)
        if new_seg.get("phonetic_skipped"):
            skipped_count += 1
        fixed.append(new_seg)

    output_data = {"segments": fixed} if wrap_dict else fixed

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Hotovo.")
    print(f"Načítaných segmentov : {len(fixed)}")
    print(f"Preskočená fonetizácia: {skipped_count}")
    print(f"Výstup               : {output_path}")


if __name__ == "__main__":
    main()
