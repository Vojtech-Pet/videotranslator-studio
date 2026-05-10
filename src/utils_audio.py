from __future__ import annotations

import re
from pathlib import Path

from scripts.level4_audio_feedback import audit_duration, wav_duration_seconds


def resolve_segment_wav(wav_dir: str | Path, seg_index: int) -> Path | None:
    base = Path(wav_dir).expanduser().resolve()
    if not base.exists():
        return None
    candidates = [
        base / f"tts_{seg_index:04d}.wav",
        base / f"seg_{seg_index:04d}.wav",
        base / f"{seg_index:04d}.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_wav_map(wav_dir: str | Path) -> dict[int, Path]:
    base = Path(wav_dir).expanduser().resolve()
    if not base.exists():
        return {}
    result: dict[int, Path] = {}
    for wav_path in sorted(base.glob("*.wav")):
        match = re.search(r"(\d+)$", wav_path.stem)
        if not match:
            continue
        result[int(match.group(1))] = wav_path
    return result

