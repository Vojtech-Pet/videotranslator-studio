from __future__ import annotations

import json
from pathlib import Path


def ensure_output_tree(base_dir: str | Path) -> dict[str, Path]:
    root = Path(base_dir).expanduser().resolve()
    output_dir = root / "output"
    wav_segments_dir = output_dir / "wav_segments"
    final_dir = output_dir / "final"
    prompts_dir = root / "prompts"
    src_dir = root / "src"
    for path in (output_dir, wav_segments_dir, final_dir, prompts_dir, src_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "output": output_dir,
        "wav_segments": wav_segments_dir,
        "final": final_dir,
        "prompts": prompts_dir,
        "src": src_dir,
    }


def load_segments(path: str | Path) -> tuple[list[dict], dict | None]:
    source = Path(path).expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "segments" in data:
        return list(data.get("segments") or []), dict(data)
    if isinstance(data, list):
        return list(data), None
    raise ValueError(f"Nepodporovany JSON format: {source}")


def save_segments(path: str | Path, segments: list[dict], wrapper: dict | None = None) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if wrapper is not None:
        payload = dict(wrapper)
        payload["segments"] = segments
    else:
        payload = segments
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def save_json(path: str | Path, payload: dict | list) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

