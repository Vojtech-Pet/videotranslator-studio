#!/usr/bin/env python3
"""Level 1 — text hygiene + merger diarization/emotion metadát do preloženého JSON.

Dva režimy:
  A) --translated existing_sk.json  → zlúči speaker/emotion z Level 0b do existujúceho prekladu
  B) bez --translated               → Level 0b JSON musí mať aj pole "text" (preklad)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.level1_text_hygiene import level1_process_text
from src.utils_io import ensure_output_tree, load_segments, save_json, save_segments
from src.utils_text import choose_segment_text

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output" / "level0b_emotion.json"
DEFAULT_OUTPUT = ROOT / "output" / "level1_fixed.json"
DEFAULT_SUMMARY = ROOT / "output" / "final" / "level1_summary.json"

_META_KEYS = ("speaker_id", "speaker_ref_audio", "speaker_gender", "emotion")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 1 — hygiene + speaker/emotion merge")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Level 0b JSON (so speaker_id, emotion)")
    parser.add_argument("--translated", default="", help="Existujúci preložený JSON (sk segments) — ak nie je, 'text' musí byť v --input")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--lang", default="sk")
    parser.add_argument("--no-number-normalization", action="store_true")
    return parser


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_match(seg: dict, candidates: list[dict]) -> dict | None:
    """Find translated segment with most time overlap."""
    best, best_ov = None, 0.0
    for c in candidates:
        ov = _overlap(seg["start"], seg["end"], c["start"], c["end"])
        if ov > best_ov:
            best_ov = ov
            best = c
    return best if best_ov > 0.05 else None


def merge_meta(diarized: list[dict], translated: list[dict]) -> list[dict]:
    """Merge speaker/emotion fields from diarized into translated segments."""
    result = []
    for tr_seg in translated:
        merged = dict(tr_seg)
        match = _best_match(tr_seg, diarized)
        if match:
            for key in _META_KEYS:
                if key in match:
                    merged[key] = match[key]
        result.append(merged)
    return result


def process(args: argparse.Namespace) -> None:
    ensure_output_tree(ROOT)

    diarized_segs, diarized_wrapper = load_segments(args.input)

    if args.translated:
        print(f"Zlučujem speaker/emotion dáta do preloženého JSON: {args.translated}")
        translated_segs, wrapper = load_segments(args.translated)
        segments = merge_meta(diarized_segs, translated_segs)
    else:
        segments = diarized_segs
        wrapper = diarized_wrapper
        missing = sum(1 for s in segments if not s.get("text") and not s.get("text_original"))
        if missing:
            print(f"Pozor: {missing}/{len(segments)} segmentov nemá pole 'text' (preklad). Použi --translated.")

    out_segments = []
    for seg in segments:
        seg_copy = dict(seg)
        raw = choose_segment_text(seg_copy, prefer_text_original=True)
        if not raw:
            raw = seg_copy.get("text_src", "")
        seg_copy["level1_text"] = level1_process_text(
            raw,
            normalize_numbers=not args.no_number_normalization,
            lang=args.lang,
            preserve_sql_terms=True,
        )
        out_segments.append(seg_copy)

    speakers = sorted({s.get("speaker_id", "?") for s in out_segments})
    genders = {s.get("speaker_id", "?"): s.get("speaker_gender", "?") for s in out_segments}
    summary = {
        "total_segments": len(out_segments),
        "speakers": speakers,
        "speaker_genders": {k: v for k, v in genders.items() if k != "?"},
        "with_emotion": sum(1 for s in out_segments if "emotion" in s),
        "with_speaker": sum(1 for s in out_segments if "speaker_id" in s),
    }

    save_segments(args.output, out_segments, wrapper)
    save_json(args.summary_output, summary)

    print(f"Level 1 hotovo: {len(out_segments)} segmentov")
    for spk in speakers:
        g = genders.get(spk, "?")
        count = sum(1 for s in out_segments if s.get("speaker_id") == spk)
        print(f"  {spk} ({g}): {count} segmentov")
    print(f"Output: {Path(args.output).resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
