#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from agents.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "input" / "sql_part_007_part001_sk_segments.json"
OUTPUT_FILE = ROOT / "output" / "mini_level11_output.json"


def main() -> None:
    input_path = INPUT_FILE
    output_path = OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
        wrapped = True
    elif isinstance(data, list):
        segments = data
        wrapped = False
    else:
        raise ValueError("Nepodporovany JSON format.")

    orchestrator = Orchestrator()
    out_segments = []

    for index, seg in enumerate(segments, start=1):
        print(f"Processing segment {index}/{len(segments)}")
        processed = orchestrator.process_segment(seg)
        out_segments.append(processed)

    with output_path.open("w", encoding="utf-8") as handle:
        if wrapped:
            data["segments"] = out_segments
            json.dump(data, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(out_segments, handle, ensure_ascii=False, indent=2)

    print(f"Hotovo: {output_path}")


if __name__ == "__main__":
    main()
