#!/usr/bin/env python3
# Run inside: conda run -n chatterbox_env python run_level0b.py ...
from __future__ import annotations

import argparse
from pathlib import Path

from src.level0b_emotion import (
    analyze_segments, attach_genders, detect_speaker_genders,
    load_emotion_pipeline, load_gender_pipeline, summarize_emotions,
)
from src.utils_io import ensure_output_tree, load_segments, save_json, save_segments

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output" / "level0_diarized.json"
DEFAULT_OUTPUT = ROOT / "output" / "level0b_emotion.json"
DEFAULT_SUMMARY = ROOT / "output" / "final" / "level0b_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 0b — Emotion analysis per segment (Wav2Vec2)")
    parser.add_argument("source_audio", help="Pôvodný video/audio súbor (pre extrakciu segmentov)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Level 0 diarized JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser


def process(args: argparse.Namespace) -> None:
    ensure_output_tree(ROOT)
    segments, wrapper = load_segments(args.input)

    print(f"Načítavam emotion model ({args.device})...")
    emotion_pipe = load_emotion_pipeline(args.device)

    print("Načítavam gender model...")
    gender_pipe = load_gender_pipeline(args.device)

    print(f"Detekujem pohlavie rečníkov...")
    genders = detect_speaker_genders(segments, args.source_audio, gender_pipe)
    for spk, g in genders.items():
        print(f"  {spk} → {g}")
    segments = attach_genders(segments, genders)

    print(f"Analyzujem emócie pre {len(segments)} segmentov...")
    segments = analyze_segments(segments, args.source_audio, emotion_pipe)

    summary = summarize_emotions(segments)
    summary["speaker_genders"] = genders
    save_segments(args.output, segments, wrapper)
    save_json(args.summary_output, summary)

    print(f"Emotion analýza hotová: {summary['total']} segmentov")
    print(f"  Distribúcia: {summary['distribution']}")
    print(f"  Avg arousal: {summary['avg_arousal']}  Avg valence: {summary['avg_valence']}")
    print(f"Output: {Path(args.output).resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
