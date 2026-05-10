#!/usr/bin/env python3
# Run inside: conda run -n musetalk_env python run_level0.py ...
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.level0_diarize import (
    attach_refs,
    build_segments,
    extract_speaker_references,
    summarize_level0,
    transcribe_and_diarize,
)
from src.utils_io import ensure_output_tree, save_json, save_segments

ROOT = Path(__file__).resolve().parent
GUI_CONFIG = ROOT / "musetalk_gui_config.json"


def _load_hf_token_from_config() -> str:
    try:
        cfg = json.loads(GUI_CONFIG.read_text(encoding="utf-8"))
        return cfg.get("hf_api_key", "")
    except Exception:
        return ""
DEFAULT_OUTPUT = ROOT / "output" / "level0_diarized.json"
DEFAULT_REFS_DIR = ROOT / "output" / "speaker_refs"
DEFAULT_SUMMARY = ROOT / "output" / "final" / "level0_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Level 0 — WhisperX transcription + diarization + speaker ref extraction"
    )
    parser.add_argument("input", help="Path to video or audio file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--refs-dir", default=str(DEFAULT_REFS_DIR), help="Dir for speaker reference WAVs")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--language", default="en", help="Source language code (default: en)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token pre Pyannote (fallback: GUI config / HF_TOKEN env)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="float16", choices=["float16", "int8", "float32"])
    parser.add_argument("--no-refs", action="store_true", help="Skip reference audio extraction")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    return parser


def process(args: argparse.Namespace) -> None:
    hf_token = args.hf_token or _load_hf_token_from_config()
    if not hf_token:
        raise SystemExit(
            "HuggingFace token nenájdený.\n"
            "Zadaj ho v GUI (HF API Key) a ulož, alebo použi --hf-token / HF_TOKEN env."
        )
    args.hf_token = hf_token

    ensure_output_tree(ROOT)
    input_path = Path(args.input).expanduser().resolve()

    print(f"Transkripcia + diarizácia: {input_path.name}")
    raw_segments, diarize_df = transcribe_and_diarize(
        input_path,
        hf_token=args.hf_token,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    segments = build_segments(raw_segments, diarize_df)

    refs: dict[str, str] = {}
    if not args.no_refs:
        print("Extrahujem referenčné audio pre každého rečníka...")
        refs = extract_speaker_references(input_path, segments, args.refs_dir)
        segments = attach_refs(segments, refs)

    summary = summarize_level0(segments, refs)

    wrapper = {
        "source_file": str(input_path),
        "language": args.language,
        "level": 0,
        "segments": segments,
    }
    save_segments(args.output, segments, wrapper)
    save_json(args.summary_output, summary)

    print(
        f"Level 0 hotovo: {summary['total_segments']} segmentov, "
        f"{summary['speaker_count']} rečník(ov): {', '.join(summary['speakers'])}"
    )
    if refs:
        for spk, path in refs.items():
            print(f"  {spk} → {path}")
    print(f"Output: {Path(args.output).resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
