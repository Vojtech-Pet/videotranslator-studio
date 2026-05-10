from __future__ import annotations


def tuned_cps_target(segment_type: str) -> float:
    if segment_type == "technical_definition":
        return 15.5
    if segment_type == "example_explanation":
        return 16.5
    if segment_type == "summary":
        return 16.8
    return 16.0


def tuned_hard_limit(segment_type: str) -> float:
    if segment_type == "technical_definition":
        return 16.5
    if segment_type == "example_explanation":
        return 17.2
    if segment_type == "summary":
        return 17.5
    return 17.0
