#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.level1_text_hygiene import level1_process_text
from src.level2_rewrite_agent import load_llm, rewrite_text
from src.level3_timing_agent import process_timing_segment
from src.level4_audio_feedback import audit_segments, summarize_audit
from src.utils_io import ensure_output_tree, load_segments, save_segments
from src.utils_text import choose_segment_text

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input" / "sql_part_007_split" / "sql_part_007_part001_sk_segments.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 1 -> Level 4 dubbing pipeline orchestrator")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to segments JSON")
    parser.add_argument("--output-dir", default=str(ROOT / "output"), help="Directory for level outputs")
    parser.add_argument("--wav-dir", default=str(ROOT / "output" / "wav_segments"), help="Directory with rendered segment WAVs")
    parser.add_argument("--llama-model", default="", help="Local GGUF model used for Level 2 and Level 3 rewrite")
    parser.add_argument("--llama-ctx", type=int, default=2048, help="Context size for llama.cpp")
    parser.add_argument("--llama-gpu-layers", type=int, default=-1, help="GPU layers for llama.cpp")
    parser.add_argument("--lang", default="sk", help="Language for deterministic hygiene steps")
    parser.add_argument("--no-level2", action="store_true", help="Skip Level 2 rewrite")
    parser.add_argument("--no-level3", action="store_true", help="Skip Level 3 timing rewrite")
    parser.add_argument("--no-number-normalization", action="store_true", help="Disable Level 1 number normalization")
    parser.add_argument("--prefer-text-original", action=argparse.BooleanOptionalAction, default=True, help="Prefer text_original over text when both exist")
    return parser


def process(args: argparse.Namespace) -> None:
    ensure_output_tree(ROOT)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wav_segments").mkdir(parents=True, exist_ok=True)
    (output_dir / "final").mkdir(parents=True, exist_ok=True)

    segments, wrapper = load_segments(args.input)
    llm = load_llm(
        args.llama_model,
        n_ctx=args.llama_ctx,
        n_gpu_layers=args.llama_gpu_layers,
        verbose=False,
    )

    try:
        level1_segments: list[dict] = []
        for seg in segments:
            seg_copy = dict(seg)
            raw = choose_segment_text(seg_copy, prefer_text_original=args.prefer_text_original)
            seg_copy["level1_text"] = level1_process_text(
                raw,
                normalize_numbers=not args.no_number_normalization,
                lang=args.lang,
                preserve_sql_terms=True,
            )
            level1_segments.append(seg_copy)
        save_segments(output_dir / "level1_fixed.json", level1_segments, wrapper)

        level2_segments: list[dict] = []
        for seg in level1_segments:
            seg_copy = dict(seg)
            level1_text = seg_copy.get("level1_text", "")
            if args.no_level2:
                level2_text = level1_text
            else:
                level2_text = rewrite_text(
                    level1_text,
                    source_text=seg_copy.get("text_src", ""),
                    llm=llm,
                )
            seg_copy["level2_text"] = level2_text
            level2_segments.append(seg_copy)
        save_segments(output_dir / "level2_rewritten.json", level2_segments, wrapper)

        level3_segments: list[dict] = []
        for seg in level2_segments:
            if args.no_level3:
                seg_copy = dict(seg)
                base_text = seg_copy.get("level2_text") or seg_copy.get("level1_text") or seg_copy.get("text", "")
                slot = max(0.0, float(seg_copy.get("end", 0.0)) - float(seg_copy.get("start", 0.0)))
                seg_copy["slot"] = round(slot, 3)
                seg_copy["cps_before"] = round(len(base_text) / slot, 2) if slot > 0.0 else 999.0
                seg_copy["cps_after"] = seg_copy["cps_before"]
                seg_copy["rewrite_mode"] = "keep"
                seg_copy["level3_text"] = base_text
                seg_copy["tts_input"] = base_text
                level3_segments.append(seg_copy)
            else:
                level3_segments.append(process_timing_segment(seg, llm=llm))
        save_segments(output_dir / "level3_timing.json", level3_segments, wrapper)

        wav_dir = Path(args.wav_dir).expanduser().resolve()
        if wav_dir.exists() and any(wav_dir.glob("*.wav")):
            level4_segments = audit_segments(level3_segments, wav_dir=wav_dir)
            summary = summarize_audit(level4_segments)
            print(
                f"Level 4 audit: ok={summary.get('ok',0)} "
                f"too_long={summary.get('too_long',0)} "
                f"too_short={summary.get('too_short',0)} "
                f"missing_wav={summary.get('missing_wav',0)}"
            )
        else:
            level4_segments = []
            for idx, seg in enumerate(level3_segments, 1):
                seg_copy = dict(seg)
                seg_copy["level4_audit"] = {
                    "seg": idx,
                    "slot": round(float(seg_copy.get("slot") or 0.0), 3),
                    "wav_duration": None,
                    "delta": None,
                    "status": "pending_audio",
                    "wav_path": "",
                }
                level4_segments.append(seg_copy)
            print("Level 4 audit preskoceny: vo wav-dir zatial nie su ziadne segmenty.")

        save_segments(output_dir / "level4_final.json", level4_segments, wrapper)
        save_segments(output_dir / "final" / "level4_final.json", level4_segments, wrapper)

        print(f"Hotovo: Level 1 -> Level 4 vystupy su v {output_dir}")
    finally:
        if llm is not None:
            del llm


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
