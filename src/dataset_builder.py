from __future__ import annotations

from typing import Any


def build_rewrite_dataset(segment_history: list[dict[str, Any]]) -> list[dict[str, str]]:
    dataset: list[dict[str, str]] = []
    for record in segment_history:
        source = str(record.get("source_text") or "").strip()
        target = str(record.get("tts_input") or "").strip()
        accepted = bool(record.get("accepted", False))
        final_score = float(record.get("final_score", 0.0) or 0.0)
        if source and target and accepted and final_score >= 0.9:
            dataset.append({"input": source, "output": target})
    return dataset


def build_negative_examples(segment_history: list[dict[str, Any]]) -> list[dict[str, str]]:
    dataset: list[dict[str, str]] = []
    for record in segment_history:
        source = str(record.get("source_text") or "").strip()
        target = str(record.get("tts_input") or "").strip()
        accepted = bool(record.get("accepted", True))
        if source and target and not accepted:
            dataset.append({"input": source, "bad_output": target})
    return dataset
