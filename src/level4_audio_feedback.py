from __future__ import annotations

from pathlib import Path

from scripts.level4_audio_feedback import (
    audit_duration,
    find_split_index,
    should_merge,
    should_split,
    tighten_text_for_retry,
    wav_duration_seconds,
)

from src.utils_audio import resolve_segment_wav


def audit_segments(
    segments: list[dict],
    *,
    wav_dir: str | Path,
    text_key: str = "tts_input",
) -> list[dict]:
    base = Path(wav_dir).expanduser().resolve()
    audited: list[dict] = []
    for idx, seg in enumerate(segments, 1):
        seg_copy = dict(seg)
        wav_path = resolve_segment_wav(base, idx)
        slot = float(seg_copy.get("slot") or (float(seg_copy.get("end", 0.0)) - float(seg_copy.get("start", 0.0))))
        if wav_path is None:
            seg_copy["level4_audit"] = {
                "seg": idx,
                "slot": round(slot, 3),
                "wav_duration": None,
                "delta": None,
                "status": "missing_wav",
                "wav_path": "",
            }
        else:
            audit = audit_duration(slot, wav_path)
            audit["seg"] = idx
            audit["wav_path"] = str(wav_path)
            audit["text_chars"] = len((seg_copy.get(text_key) or "").strip())
            seg_copy["level4_audit"] = audit
        audited.append(seg_copy)
    return audited


def summarize_audit(segments: list[dict]) -> dict[str, int]:
    summary = {"ok": 0, "too_long": 0, "too_short": 0, "missing_wav": 0}
    for seg in segments:
        status = (seg.get("level4_audit") or {}).get("status", "missing_wav")
        summary[status] = summary.get(status, 0) + 1
    return summary

