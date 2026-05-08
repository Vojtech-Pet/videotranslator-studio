#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from core.utils import estimate_cps, get_slot
from pipeline_whisperx_fish_retry import (
    DEFAULT_HARD_CPS_LIMIT,
    DEFAULT_INPUT_JSON,
    DEFAULT_TARGET_CPS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    choose_rewrite_mode,
    clean_punctuation,
    clean_spaces,
    effective_cps_limits,
    enforce_spoken_safety,
    enforce_sql_terms,
    fix_sql_terms,
    ollama_generate,
    simple_rewrite,
)

LOCKED_TERMS = [
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
]

KNOWN_SQL_SEGMENT_OVERRIDES = {
    5: "V SQL na to máme dve funkcie: ISNULL a COALESCE.",
    8: "Ak chceme nahradiť hodnotu za NULL, použijeme funkciu NULLIF.",
    20: 'Začnime funkciou ISNULL, ktorá nahradí NULL konkrétnou hodnotou.',
    21: "Syntax funkcie ISNULL je veľmi jednoduchá.",
    22: "Použijeme funkciu ISNULL, ktorá prijíma dva argumenty.",
    26: "Takto skontrolujeme, či stĺpec obsahuje NULL hodnoty.",
    27: 'Ak SQL narazí na NULL, nahradí ju hodnotou "unknown".',
    28: "Táto hodnota funguje ako predvolená náhrada za NULL.",
    35: "Ak je hodnota NULL, použije sa hodnota z adresy fakturácie.",
    39: "Takto nahrádzame NULL hodnoty pomocou iného stĺpca.",
}


def derive_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_segments"):
        stem = f"{stem}_rewritten"
    else:
        stem = f"{stem}_rewritten"
    return input_path.with_name(f"{stem}{input_path.suffix}")


def classify_segment_type(text: str) -> str:
    low = clean_spaces(text).lower()
    if any(x in low for x in ["syntax", "isnull", "coalesce", "nullif", "is null", "is not null"]):
        return "definition"
    if any(x in low for x in ["example", "for example", "scenario", "orders", "address", "príklad", "pozrime sa"]):
        return "example"
    if any(x in low for x in ["big picture", "overview", "summary", "prehľad", "zhrnutie"]):
        return "summary"
    return "default"


def build_rewrite_prompt(text_en: str, slot: float, mode: str, seg_type: str) -> str:
    return f"""
Si expert na slovenský technický dabing.

Úloha:
Prelož vetu z angličtiny do prirodzenej technickej slovenčiny vhodnej pre voiceover.

KRITICKÉ PRAVIDLÁ:
- Zachovaj význam.
- Zachovaj presne tieto výrazy:
  SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, NULLIF, TRUE, FALSE, BOOLEAN
- Nikdy nepíš:
  ajznul, koalesk, esíkjúel
- Neprekladaj SQL keywordy.
- Rozlišuj presne:
  ISNULL = funkcia na nahradenie NULL hodnoty
  IS NULL = podmienka na kontrolu NULL
  NULLIF = funkcia, ktorá pri zhode vráti NULL
- Nepoužívaj doslovné anglické formulácie.
- Výstup musí byť vhodný na hlasné čítanie.

REŽIM:
{mode}

TYP SEGMENTU:
{seg_type}

SLOT:
{slot:.2f} sekundy

TEXT:
"{text_en}"

Vráť iba finálnu slovenskú vetu.
""".strip()


def source_guided_template(source_text: str) -> str | None:
    low = clean_spaces(source_text).lower()

    if "in order to do that in sql we have two functions" in low and "called isnull" in low:
        return "V SQL na to máme dve funkcie: ISNULL a COALESCE."
    if "replace the value with a null" in low and "nullif" in low:
        return "Ak chceme nahradiť hodnotu za NULL, použijeme funkciu NULLIF."
    if "between the is and null there is like space" in low:
        return "Na to máme podmienku IS NULL, medzi IS a NULL je medzera."
    if "if you apply is null" in low and ("true or false" in low or "pull in true or false" in low):
        return "Keď použijeme IS NULL, dostaneme hodnotu TRUE alebo FALSE."
    if "syntax of the is null is very simple" in low:
        return "Syntax funkcie ISNULL je veľmi jednoduchá."
    if "accepts two arguments" in low and "replacement value" in low:
        return "Funkcia ISNULL prijíma dva argumenty: najprv hodnotu a potom náhradnú hodnotu."
    if "use the is null for the column called shipping address" in low:
        return "Funkciu ISNULL môžeme použiť napríklad na stĺpec s adresou doručenia."
    if "if sql encounters any null" in low and "replace it with the" in low:
        return 'Ak SQL narazí na NULL, nahradí ju hodnotou "unknown".'
    if "default value" in low and "unknown" in low:
        return "Táto hodnota funguje ako predvolená náhrada za NULL."
    if "don't want to have it always like the unknown" in low and "use another" in low:
        return 'V niektorých prípadoch však nechceme vždy použiť hodnotu "unknown".'
    if "column to help the first one" in low:
        return "Namiesto statickej hodnoty môžeme použiť iný stĺpec ako náhradu."
    if "whether we have a null value" in low:
        return "Takto skontrolujeme, či stĺpec obsahuje NULL hodnoty."
    if "shipping address is null" in low and "billing address" in low:
        return "Ak je hodnota NULL, použije sa hodnota z adresy fakturácie."
    if "replacing the nulls using the help of other column" in low:
        return "Takto nahrádzame NULL hodnoty pomocou iného stĺpca."
    return None


def apply_known_dataset_override(dataset_name: str, seg_index: int) -> str | None:
    if "sql_part_007_part001" not in dataset_name:
        return None
    return KNOWN_SQL_SEGMENT_OVERRIDES.get(seg_index)


def enforce_sql_logic(text: str, *, source_text: str = "", fallback_text: str = "") -> str:
    out = text or ""

    replacements = [
        (r"Syntax funkcie IS NULL", "Syntax funkcie ISNULL"),
        (r"Použijeme (?:kľúčové slovo |funkciu )IS NULL", "Použijeme funkciu ISNULL"),
        (r"IS NULL, ktorá prijíma dva argumenty", "ISNULL, ktorá prijíma dva argumenty"),
        (r"funkci[ao]u NULL\b", "funkciou NULLIF"),
        (r"\bSQL NULL\b", "NULLIF"),
        (r"\bprázdna hodnota\b", "NULL hodnota"),
        (r"\bprázdne hodnoty\b", "NULL hodnoty"),
    ]
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    for term in LOCKED_TERMS:
        out = re.sub(rf"\b{re.escape(term)}\b", term, out, flags=re.IGNORECASE)

    low_src = clean_spaces(source_text).lower()
    low_fallback = clean_spaces(fallback_text).lower()
    joined = f"{low_src} {low_fallback}".strip()

    if "true or false" in joined or "pull in true or false" in joined or "boolean" in joined:
        out = re.sub(r"\bISNULL\b", "IS NULL", out, flags=re.IGNORECASE)

    if "two arguments" in joined or "replacement value" in joined or "default value" in joined:
        out = re.sub(r"Syntax funkcie IS NULL", "Syntax funkcie ISNULL", out, flags=re.IGNORECASE)
        out = re.sub(r"funkci[ao]u IS NULL", "funkciou ISNULL", out, flags=re.IGNORECASE)
        out = re.sub(r"\bIS NULL, ktorá prijíma dva argumenty", "ISNULL, ktorá prijíma dva argumenty", out, flags=re.IGNORECASE)

    if "nullif" in joined:
        out = re.sub(r"\bNULL\b", "NULLIF", out, count=1, flags=re.IGNORECASE)
        out = re.sub(r"\bfunkcia ISNULL\b", "funkcia NULLIF", out, flags=re.IGNORECASE)

    if "coalesce" in joined:
        out = re.sub(r"\bkoalesk\b", "COALESCE", out, flags=re.IGNORECASE)

    if "isnull" in joined and "coalesce" in joined and "two functions" in joined:
        out = "V SQL na to máme dve funkcie: ISNULL a COALESCE."

    return clean_punctuation(out)


def normalize_rewrite_text(text: str, *, source_text: str = "", fallback_text: str = "") -> str:
    out = fix_sql_terms(text or "")
    out = enforce_sql_terms(out)
    out = enforce_sql_logic(out, source_text=source_text, fallback_text=fallback_text)
    out = enforce_spoken_safety(out)
    out = clean_punctuation(out)
    return out


def rewrite_segment(
    seg: dict[str, Any],
    *,
    enable_ollama: bool,
    ollama_url: str,
    ollama_model: str,
    prefer_text_src: bool,
    overwrite_text_field: bool,
    write_debug_fields: bool,
    target_cps: float,
    hard_cps_limit: float,
    seg_index: int,
    dataset_name: str,
) -> dict[str, Any]:
    seg_out = dict(seg)
    slot = get_slot(seg_out)

    source_text = clean_spaces(str(seg_out.get("text_src") or "")) if prefer_text_src else ""
    fallback_text = clean_spaces(str(seg_out.get("text") or seg_out.get("tts_input") or ""))
    base_text = source_text or fallback_text or clean_spaces(str(seg_out.get("text_src") or ""))

    seg_type = classify_segment_type(base_text)
    effective_target, effective_hard = effective_cps_limits(
        seg_type,
        target_cps=target_cps,
        hard_cps_limit=hard_cps_limit,
    )
    mode = choose_rewrite_mode(
        slot,
        base_text,
        target_cps=effective_target,
        hard_cps_limit=effective_hard,
    )

    output_text = ""
    rewrite_provider = "fallback"
    rewrite_error = ""

    if enable_ollama:
        prompt = build_rewrite_prompt(base_text, slot, mode, seg_type)
        try:
            output_text = ollama_generate(
                prompt,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
            )
            rewrite_provider = "ollama"
        except Exception as exc:  # pragma: no cover - runtime-only network path
            rewrite_error = str(exc)

    if not clean_spaces(output_text):
        heuristic_source = fallback_text or base_text
        output_text = simple_rewrite(heuristic_source, slot, aggressive=(mode == "aggressive"))
        rewrite_provider = "fallback"

    output_text = normalize_rewrite_text(
        output_text,
        source_text=source_text,
        fallback_text=fallback_text,
    )

    source_override = source_guided_template(source_text)
    if source_override:
        output_text = normalize_rewrite_text(
            source_override,
            source_text=source_text,
            fallback_text=fallback_text,
        )
        rewrite_provider = "source_template"

    dataset_override = apply_known_dataset_override(dataset_name, seg_index)
    if dataset_override:
        output_text = normalize_rewrite_text(
            dataset_override,
            source_text=source_text,
            fallback_text=fallback_text,
        )
        rewrite_provider = "dataset_override"

    if not output_text:
        output_text = normalize_rewrite_text(
            simple_rewrite(fallback_text or base_text, slot, aggressive=(mode == "aggressive")),
            source_text=source_text,
            fallback_text=fallback_text,
        )
        rewrite_provider = "fallback"

    seg_out["tts_input"] = output_text
    if overwrite_text_field:
        seg_out["text"] = output_text

    if write_debug_fields:
        seg_out["rewrite_mode"] = mode
        seg_out["segment_type"] = seg_type
        seg_out["slot"] = round(slot, 3)
        seg_out["tts_cps"] = round(estimate_cps(output_text, slot), 2)
        seg_out["rewrite_source"] = "text_src" if source_text and rewrite_provider in {"ollama", "source_template", "dataset_override"} else "text"
        seg_out["rewrite_provider"] = rewrite_provider
        seg_out["target_cps"] = round(effective_target, 2)
        seg_out["hard_cps_limit"] = round(effective_hard, 2)
        if rewrite_error:
            seg_out["rewrite_error"] = rewrite_error

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
    raise ValueError("Očakávam list segmentov alebo objekt s kľúčom 'segments'.")


def save_segments(
    path: Path,
    segments: list[dict[str, Any]],
    wrapper: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if wrapper is not None:
            wrapper["segments"] = segments
            json.dump(wrapper, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(segments, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full dataset rewrite pre SQL/TTS segment JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Vstupný JSON so segmentmi.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Výstupný JSON. Ak chýba, vytvorí sa sibling súbor s príponou _rewritten.json.",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help="Ollama generate endpoint.",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--disable-ollama",
        action="store_true",
        help="Vypne Ollama rewrite a použije len fallback heuristiky.",
    )
    parser.add_argument(
        "--prefer-text-field",
        action="store_true",
        help="Nepreferovať text_src, ale pôvodné text pole.",
    )
    parser.add_argument(
        "--overwrite-text-field",
        action="store_true",
        help="Prepísať aj pôvodné pole text výsledným tts_input.",
    )
    parser.add_argument(
        "--no-debug-fields",
        action="store_true",
        help="Nevypisovať debug metadata do segmentov.",
    )
    parser.add_argument(
        "--target-cps",
        type=float,
        default=DEFAULT_TARGET_CPS,
        help="Cieľové CPS.",
    )
    parser.add_argument(
        "--hard-cps-limit",
        type=float,
        default=DEFAULT_HARD_CPS_LIMIT,
        help="Tvrdý CPS limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Vstupný súbor neexistuje: {input_path}")

    output_path = (args.output.expanduser().resolve() if args.output else derive_output_path(input_path))
    segments, wrapper = load_segments(input_path)

    rewritten: list[dict[str, Any]] = []
    total = len(segments)
    ollama_enabled = not bool(args.disable_ollama)
    dataset_name = input_path.stem

    for idx, seg in enumerate(segments, start=1):
        print(f"[{idx}/{total}] rewriting...", flush=True)
        rewritten.append(
            rewrite_segment(
                seg,
                enable_ollama=ollama_enabled,
                ollama_url=args.ollama_url,
                ollama_model=args.ollama_model,
                prefer_text_src=not bool(args.prefer_text_field),
                overwrite_text_field=bool(args.overwrite_text_field),
                write_debug_fields=not bool(args.no_debug_fields),
                target_cps=float(args.target_cps),
                hard_cps_limit=float(args.hard_cps_limit),
                seg_index=idx,
                dataset_name=dataset_name,
            )
        )

    save_segments(output_path, rewritten, wrapper)

    avg_cps = 0.0
    if rewritten:
        avg_cps = sum(float(seg.get("tts_cps") or 0.0) for seg in rewritten) / len(rewritten)
    print(f"Hotovo: {output_path}", flush=True)
    print(
        f"Segmenty: {len(rewritten)} | Ollama: {'on' if ollama_enabled else 'off'} | "
        f"avg tts_cps: {avg_cps:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
