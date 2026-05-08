"""
Model bootstrap — first-run download manager.

Stiahne potrebné modely z HuggingFace pri prvom spustení.
Každý model má `repo_id`, voliteľne `filename` (single-file repos) alebo `pattern`
(snapshot s allow_patterns).

Použitie:
    from scripts.model_bootstrap import REQUIRED_MODELS, bootstrap_models, missing_models

    missing = missing_models(models_dir)
    if missing:
        results = bootstrap_models(models_dir, only=missing,
                                   progress_callback=lambda mid, pct: print(f"{mid}: {pct}%"))
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class ModelSpec:
    id: str                      # local identifier (used as subdir under models_dir)
    repo_id: str                 # HuggingFace repo id (e.g. "Vojtech-Pet/videotranslator-gemma-26b-translator")
    description: str             # human-readable label
    size_gb: float               # approximate download size
    essential: bool              # required for default pipeline
    filename: Optional[str] = None     # single-file repo: GGUF or specific weight file
    allow_patterns: Optional[list] = None  # snapshot subset (e.g. ["*.safetensors", "config.json"])


# ── Model registry ────────────────────────────────────────────────────────────
# UPRAVTE PO PRVOM UPLOAD-E:
# - repo_id musí zodpovedať reálnemu HF repu
# - filename pre GGUF single-file repos
# - size_gb pre user-side disk-space check
REQUIRED_MODELS: list[ModelSpec] = [
    # ── ESSENTIAL (povinné pre default pipeline) ──────────────────────────────
    # OmniVoice — upstream zero-shot TTS (k2-fsa, public, žiadny re-upload)
    ModelSpec(
        id="omnivoice",
        repo_id="k2-fsa/OmniVoice",
        description="OmniVoice zero-shot TTS (k2-fsa, 600+ jazykov, primary)",
        size_gb=3.3,
        essential=True,
    ),
    # Whisper STT — community CTranslate2 INT8 build
    ModelSpec(
        id="whisper-large-v3-turbo",
        repo_id="deepdml/faster-whisper-large-v3-turbo-ct2",
        description="Faster-Whisper large-v3-turbo (CTranslate2 INT8)",
        size_gb=1.5,
        essential=True,
    ),
    # Gemma 4 26B base — original quantized (public, Google → ggml-org community GGUF)
    ModelSpec(
        id="gemma-4-26b",
        repo_id="ggml-org/gemma-4-26B-A4B-it-GGUF",
        description="Gemma 4 26B A4B Q4_K_M (base, original Google model)",
        size_gb=16.0,
        essential=True,
        filename="gemma-4-26B-A4B-it-Q4_K_M.gguf",
    ),

    # ── OPTIONAL (vlastné fine-tunes pekiskola) ──────────────────────────────
    # Chatterbox SK 2.2 — existujúci HF repo
    ModelSpec(
        id="chatterbox-sk",
        repo_id="pekiskol/chatterbox-tts-slovak",
        description="Chatterbox SK 2.2 (slovenský fine-tune, MIT)",
        size_gb=1.5,
        essential=False,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.txt"],
    ),
    # Gemma 3 27B multitask v2 — ODSTRÁNENÉ 2026-05-06 (refine echoed EN, broken on tech texts)
    # Gemma 3 12B translator v1 — primary fine-tune (used aj ako default refine)
    ModelSpec(
        id="g3-12b-translator",
        repo_id="pekiskol/gemma-3-12b-translator-v1",
        description="Gemma 3 12B EN→SK translator v1 (SK fine-tune, default refine model)",
        size_gb=8.0,
        essential=False,
        filename="gemma-3-12b-translator-v1-q5km.gguf",
    ),
    # Gemma 3 12B length adjuster v1 — natrénovaný (TODO upload)
    ModelSpec(
        id="g3-12b-length-adjuster",
        repo_id="pekiskol/gemma-3-12b-length-adjuster-v1",
        description="Gemma 3 12B SK length adjuster v1 (compress/expand)",
        size_gb=8.0,
        essential=False,
        filename="gemma-3-12b-length-adjuster-v1-q5km.gguf",
    ),
]


def models_disk_path(custom: Optional[Path] = None) -> Path:
    """Vráti default cieľ pre modely (override cez VTS_MODELS_DIR alebo paths.PATHS.models_root)."""
    if custom:
        return custom
    env = os.environ.get("VTS_MODELS_DIR", "")
    if env:
        return Path(env)
    try:
        from paths import PATHS
        if PATHS.models_root and Path(PATHS.models_root).exists():
            return Path(PATHS.models_root)
    except Exception:
        pass
    return Path.home() / ".local" / "share" / "videotranslator" / "models"


def is_model_present(spec: ModelSpec, models_dir: Path) -> bool:
    target = models_dir / spec.id
    if not target.exists():
        return False
    if spec.filename:
        return (target / spec.filename).exists()
    # snapshot — staci ak adresar nie je prazdny
    return any(target.iterdir())


def missing_models(models_dir: Path, only_essential: bool = False) -> list[ModelSpec]:
    out = []
    for spec in REQUIRED_MODELS:
        if only_essential and not spec.essential:
            continue
        if not is_model_present(spec, models_dir):
            out.append(spec)
    return out


def disk_space_gb(path: Path) -> float:
    """Vráti voľné miesto na disku v GB pre danú cestu."""
    path.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(str(path))
    return stat.free / (1024 ** 3)


def bootstrap_models(
    models_dir: Path,
    only: Optional[list[ModelSpec]] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    hf_token: Optional[str] = None,
) -> dict[str, str]:
    """Stiahne chýbajúce modely. Vracia {model_id: status}."""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError as e:
        return {"error": f"huggingface_hub missing — pip install huggingface_hub: {e}"}

    models_dir.mkdir(parents=True, exist_ok=True)
    targets = only if only is not None else missing_models(models_dir)
    results: dict[str, str] = {}

    for spec in targets:
        target_dir = models_dir / spec.id
        target_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback:
            progress_callback(spec.id, 0)

        try:
            if spec.filename:
                # Single-file download
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=spec.filename,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    token=hf_token,
                )
            else:
                snapshot_download(
                    repo_id=spec.repo_id,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    token=hf_token,
                    allow_patterns=spec.allow_patterns,
                )
            results[spec.id] = "ok"
            if progress_callback:
                progress_callback(spec.id, 100)
        except Exception as e:
            results[spec.id] = f"failed: {type(e).__name__}: {e}"
            if progress_callback:
                progress_callback(spec.id, -1)

    return results


def total_download_size_gb(specs: list[ModelSpec]) -> float:
    return sum(s.size_gb for s in specs)


if __name__ == "__main__":
    # CLI diagnostika: vypíše stav modelov
    import sys
    md = models_disk_path()
    print(f"Models dir: {md}")
    print(f"  exists: {md.exists()}")
    print(f"  free disk: {disk_space_gb(md):.1f} GB\n")
    print("=== Required models ===")
    for spec in REQUIRED_MODELS:
        present = is_model_present(spec, md)
        marker = "✓" if present else "✗"
        ess = " (essential)" if spec.essential else ""
        print(f"  {marker} {spec.id} [{spec.size_gb:.1f} GB]{ess}")
        print(f"      {spec.description}")
        print(f"      repo: {spec.repo_id}")

    missing = missing_models(md)
    if missing:
        print(f"\nMissing: {len(missing)} models, {total_download_size_gb(missing):.1f} GB")
        if "--download" in sys.argv:
            print("Spúšťam download...")
            results = bootstrap_models(md, only=missing,
                                        progress_callback=lambda mid, pct: print(f"  [{pct:3d}%] {mid}"))
            for mid, status in results.items():
                print(f"  {mid}: {status}")
