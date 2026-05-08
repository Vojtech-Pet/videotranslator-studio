#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sql_logic_guard import clean_punctuation, guard_sql_text, normalize_spaces, source_guided_template

DEFAULT_INPUT_CANDIDATES = (
    ROOT_DIR / "temp/sql_part_007_part001/sql_part_007_part001_sk_segments.json",
    ROOT_DIR / "mini_level11/input/sql_part_007_part001_sk_segments.json",
)

KEYWORDS = (
    "SQL",
    "NULL",
    "IS NULL",
    "IS NOT NULL",
    "ISNULL",
    "COALESCE",
    "NULLIF",
    "TRUE",
    "FALSE",
    "BOOLEAN",
)

KNOWN_SQL_SEGMENT_OVERRIDES = {
    5: "V SQL na to máme dve funkcie: ISNULL a COALESCE.",
    8: "Ak chceme nahradiť hodnotu za NULL, použijeme funkciu NULLIF.",
    20: "Začnime funkciou ISNULL, ktorá nahradí NULL konkrétnou hodnotou.",
    21: "Syntax funkcie ISNULL je veľmi jednoduchá.",
    22: "Použijeme funkciu ISNULL, ktorá prijíma dva argumenty.",
    26: "Takto skontrolujeme, či stĺpec obsahuje NULL hodnoty.",
    27: 'Ak SQL narazí na NULL, nahradí ju hodnotou "unknown".',
    28: "Táto hodnota funguje ako predvolená náhrada za NULL.",
    35: "Ak je hodnota NULL, použije sa hodnota z adresy fakturácie.",
    39: "Takto nahrádzame NULL hodnoty pomocou iného stĺpca.",
    40: "Ak hodnota nie je NULL, zobrazíme ju. Pozrime sa na príklad.",
    44: "Ak hodnota nie je NULL, SQL vráti rovnakú hodnotu.",
    46: "Pri druhej objednávke je adresa zásielky NULL. Čo sa stane v takom prípade?",
    48: 'Hodnota je "na", takže vo výstupe nedostaneme NULL, ale "na".',
    54: "Máme teda dva stĺpce a logika zostáva rovnaká.",
    58: "Pri prvej objednávke adresa nie je NULL, takže obsahuje hodnotu A.",
}


def find_default_input() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[-1]


def derive_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_segments"):
        stem = f"{stem[:-len('_segments')]}_segments_FIXED"
    else:
        stem = f"{stem}_FIXED"
    return input_path.with_name(f"{stem}{input_path.suffix}")


def clean(text: str) -> str:
    return clean_punctuation(normalize_spaces(text))


def fix_null_words(text: str) -> str:
    replacements = [
        (r"\bnuly\b", "NULL hodnoty"),
        (r"\bnulová\b", "NULL"),
        (r"\bnulova\b", "NULL"),
        (r"\bnulové\b", "NULL"),
        (r"\bnulove\b", "NULL"),
        (r"\bnulovú\b", "NULL"),
        (r"\bnulovu\b", "NULL"),
        (r"\bnulovou\b", "NULL"),
        (r"\bpr[aá]zdna hodnota\b", "NULL hodnota"),
        (r"\bpr[aá]zdne hodnoty\b", "NULL hodnoty"),
        (r"\bnulová hodnota\b", "NULL hodnota"),
        (r"\bnulova hodnota\b", "NULL hodnota"),
        (r"\bnulové hodnoty\b", "NULL hodnoty"),
        (r"\bnulove hodnoty\b", "NULL hodnoty"),
    ]
    out = clean(text)
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return clean(out)


def simplify_sentence(text: str) -> str:
    rules = [
        (r"\bTakže,\s*", ""),
        (r"\bTakze,\s*", ""),
        (r"\bTakže\b", ""),
        (r"\bTakze\b", ""),
        (r"\bV poriadku,? priatelia,?\s*", ""),
        (r"\bDobre,? priatelia,?\s*", ""),
        (r"\bpoďme si\b", ""),
        (r"\bpojdme si\b", ""),
        (r"\bpoďme\b", ""),
        (r"\bpojdme\b", ""),
        (r"\bteraz\b", ""),
        (r"\bteraz\b", ""),
        (r"\bponoríme sa do\b", "pozrieme sa na"),
        (r"\bponorime sa do\b", "pozrieme sa na"),
        (r"\bhlboký ponor\b", "prehľad"),
        (r"\bhlboky ponor\b", "prehlad"),
        (r"\bgo a\b", ""),
        (r"\bnášho stola\b", "tabuľky"),
        (r"\bnasho stola\b", "tabulky"),
        (r"\bnáš stôl\b", "tabuľka"),
        (r"\bnas stol\b", "tabulka"),
        (r"\bv našej tabuľke\b", "v tabuľke"),
        (r"\bnasej tabulke\b", "tabulke"),
        (r"\bis a null\b", "IS NULL"),
        (r"\bnot nul\b", "IS NOT NULL"),
        (r"\bakceptuje\b", "prijíma"),
        (r"\bakceptuje\b", "prijima"),
        (r"\bdefault value\b", "predvolená hodnota"),
        (r"\bdefault value\b", "predvolena hodnota"),
    ]
    out = clean(text)
    for pattern, repl in rules:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return clean(out)


def fix_logic(text: str) -> str:
    replacements = [
        (r"Syntax funkcie IS NULL", "Syntax funkcie ISNULL"),
        (
            r"Použijeme.*IS NULL.*argumenty",
            "Použijeme funkciu ISNULL, ktorá prijíma dva argumenty.",
        ),
        (
            r"Pouzijeme.*IS NULL.*argumenty",
            "Pouzijeme funkciu ISNULL, ktora prijima dva argumenty.",
        ),
        (r"\bfunkciu NULL\b", "funkciu NULLIF"),
        (r"\bfunkciu NULL\b", "funkciu NULLIF"),
        (r"\bfunkcia NULL\b", "funkcia NULLIF"),
        (r"\bfunkcia NULL\b", "funkcia NULLIF"),
        (r"\bISNULL,?\s*nie,?\s*m[aá]me hodnotu\b", "Ak hodnota nie je NULL, máme hodnotu"),
        (r"\bISNULL, nie, mame hodnotu\b", "Ak hodnota nie je NULL, mame hodnotu"),
        (r"\bhodnota nie je ISNULL\b", "hodnota nie je NULL"),
        (r"\bhodnota nie ISNULL\b", "hodnota nie je NULL"),
        (r"\bto nie je ISNULL\b", "to nie je NULL"),
        (r"\bto nie ISNULL\b", "to nie je NULL"),
        (r"\badresa zásielky ako NULL\b", "adresa zásielky je NULL"),
        (r'\bdostaneme na\b', 'dostaneme "na"'),
    ]
    out = clean(text)
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return clean(out)


def final_cleanup(text: str, *, source_text: str, fallback_text: str) -> str:
    out = guard_sql_text(text, source_text=source_text, fallback_text=fallback_text)
    out = re.sub(r"\bNULL NULL\b", "NULL", out, flags=re.IGNORECASE)
    out = re.sub(r"\bNULL NULL hodnoty\b", "NULL hodnoty", out, flags=re.IGNORECASE)
    for keyword in KEYWORDS:
        out = re.sub(rf"\b{re.escape(keyword)}\b", keyword, out, flags=re.IGNORECASE)
    return clean(out)


def process_segment(seg: dict[str, Any], *, seg_index: int, dataset_name: str) -> dict[str, Any]:
    seg_out = dict(seg)
    source_text = clean(str(seg_out.get("text_src") or ""))
    original_text = clean(str(seg_out.get("text") or ""))
    working_text = clean(str(seg_out.get("tts_input") or original_text or source_text))

    working_text = fix_null_words(working_text)
    working_text = simplify_sentence(working_text)
    working_text = fix_logic(working_text)
    working_text = final_cleanup(
        working_text,
        source_text=source_text,
        fallback_text=original_text,
    )

    provider = "rule_fix"
    template = source_guided_template(source_text)
    if template:
        working_text = final_cleanup(
            fix_logic(simplify_sentence(fix_null_words(template))),
            source_text=source_text,
            fallback_text=original_text,
        )
        provider = "source_template"

    if "sql_part_007_part001" in dataset_name and seg_index in KNOWN_SQL_SEGMENT_OVERRIDES:
        working_text = final_cleanup(
            KNOWN_SQL_SEGMENT_OVERRIDES[seg_index],
            source_text=source_text,
            fallback_text=original_text,
        )
        provider = "dataset_override"

    seg_out["tts_input"] = working_text
    seg_out["fix_provider"] = provider
    seg_out["fix_source"] = "text_src" if source_text else "text"
    return seg_out


def load_segments(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read segments from {path}: {exc}") from exc
    if isinstance(payload, dict) and "segments" in payload:
        return list(payload["segments"]), payload
    if isinstance(payload, list):
        return list(payload), None
    raise ValueError("Ocakavam list segmentov alebo objekt s klucom 'segments'.")


def save_segments(path: Path, segments: list[dict[str, Any]], wrapper: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if wrapper is not None:
            wrapper["segments"] = segments
            json.dump(wrapper, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(segments, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministicky fixer pre cely SQL segment JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=find_default_input(),
        help="Vstupny JSON so segmentmi.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Vystupny JSON. Ak chyba, vytvori sa sibling _FIXED.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Vstupny subor neexistuje: {input_path}")

    output_path = args.output.expanduser().resolve() if args.output else derive_output_path(input_path)
    segments, wrapper = load_segments(input_path)

    processed: list[dict[str, Any]] = []
    dataset_name = input_path.stem
    total = len(segments)

    for idx, seg in enumerate(segments, start=1):
        processed.append(process_segment(seg, seg_index=idx, dataset_name=dataset_name))
        print(f"[{idx}/{total}] done", flush=True)

    save_segments(output_path, processed, wrapper)
    print(f"\nHOTOVO: {output_path}", flush=True)


if __name__ == "__main__":
    main()
