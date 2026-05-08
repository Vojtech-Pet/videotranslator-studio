"""
Centrálne cesty pre VideoTranslator Studio.

Filozofia:
  - 5 koreňových ciest (miniforge_root, models_root, knihy_root, ollama_bin, lmstudio_bin)
  - Všetky ostatné cesty sú odvodené (conda envs, GGUF modely)
  - Auto-detect pri prvom štarte; override cez ~/.config/videotranslator/paths.json
  - Bootstrap dialog v GUI ak chýbajú esenciálne cesty

Použitie:
    from scripts.paths import PATHS
    print(PATHS.gemma3_12b_q8)
    print(PATHS.python_chatterbox)
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Config location ───────────────────────────────────────────────────────────
CONFIG_DIR = Path.home() / ".config" / "videotranslator"
CONFIG_FILE = CONFIG_DIR / "paths.json"


# ── Auto-detect helpers ───────────────────────────────────────────────────────
def _first_existing(candidates: list[Path], fallback: Path | None = None) -> Path:
    """Vráti prvú cestu zo zoznamu kandidátov, ktorá existuje."""
    for p in candidates:
        if p.exists():
            return p
    return fallback if fallback is not None else candidates[0]


def _detect_miniforge() -> Path:
    return _first_existing([
        Path.home() / "miniforge3",
        Path("/mnt/tts_data/miniforge3"),
        Path("/opt/miniforge3"),
        Path("/usr/local/miniforge3"),
        Path("/opt/conda"),
    ])


def _detect_models_root() -> Path:
    return _first_existing([
        Path("/mnt/tts_data/VideoTranslator_studio/models"),
        Path.home() / "Ai" / "models",
        Path.home() / ".local" / "share" / "videotranslator" / "models",
        Path("/var/lib/videotranslator/models"),
    ])


def _detect_knihy_root() -> Path:
    return _first_existing([
        Path("/mnt/tts_data/knihy"),
        Path.home() / "Ai" / "knihy",
        Path.home() / ".local" / "share" / "videotranslator" / "knihy",
    ])


def _detect_ollama_bin() -> Path:
    return _first_existing([
        Path("/mnt/tts_data/ollama/bin/ollama"),
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path.home() / ".local" / "bin" / "ollama",
    ])


def _detect_lmstudio_bin() -> Path:
    return _first_existing([
        Path.home() / ".lmstudio" / "bin" / "lms",
        Path("/usr/local/bin/lms"),
    ])


def _detect_fish_legacy() -> Path:
    return _first_existing([
        Path.home() / "VideoTranslator" / "fish_speech",
        Path("/mnt/tts_data/VideoTranslator_studio/fish_speech"),
    ])


def _detect_dataset_root() -> Path:
    return _first_existing([
        Path("/mnt/tts_data/dataset"),
        Path.home() / "Videos" / "dataset",
        Path.home() / "Music" / "dataset",
    ])


# ── Paths dataclass ───────────────────────────────────────────────────────────
@dataclass
class Paths:
    """Konfigurovateľné koreňové cesty. Všetky derivácie cez @property."""
    miniforge_root: str = ""
    models_root: str = ""
    knihy_root: str = ""
    ollama_bin: str = ""
    lmstudio_bin: str = ""
    fish_legacy_root: str = ""
    dataset_root: str = ""

    # ── Conda environments (pythony) ─────────────────────────────────────────
    @property
    def python_chatterbox(self) -> Path:
        return Path(self.miniforge_root) / "envs/chatterbox_env/bin/python3"

    @property
    def python_musetalk(self) -> Path:
        return Path(self.miniforge_root) / "envs/musetalk_env/bin/python"

    @property
    def python_xtts(self) -> Path:
        return Path(self.miniforge_root) / "envs/xtts_env/bin/python"

    @property
    def python_fish(self) -> Path:
        return Path(self.miniforge_root) / "envs/fish_env/bin/python3"

    @property
    def python_f5tts(self) -> Path:
        # OmniVoice + F5-TTS env (3.3 GB OmniVoice model + dependencies)
        return Path(self.miniforge_root) / "envs/f5tts_env/bin/python"

    @property
    def python_omnivoice(self) -> Path:
        # Alias k python_f5tts — OmniVoice je nainštalovaný v f5tts_env (alebo override)
        custom = Path(self.miniforge_root) / "envs/omnivoice_env/bin/python"
        if custom.exists():
            return custom
        return self.python_f5tts

    @property
    def python_finetune(self) -> Path:
        return Path(self.miniforge_root) / "envs/finetune_env/bin/python"

    @property
    def python_heygen_diar(self) -> Path:
        return Path(self.miniforge_root) / "envs/heygen_diar/python"

    # ── Modely (HF + GGUF) v models_root ─────────────────────────────────────
    @property
    def gemma3_12b_q8(self) -> Path:
        return Path(self.models_root) / "gemma-3-12b-it" / "google_gemma-3-12b-it-Q8_0.gguf"

    @property
    def gemma3_12b_dir(self) -> Path:
        return Path(self.models_root) / "gemma-3-12b-it"

    @property
    def gemma3_27b_q6(self) -> Path:
        return Path(self.models_root) / "gemma-3-27b-it" / "google_gemma-3-27b-it-Q6_K.gguf"

    @property
    def gemma4_26b_base_q4km(self) -> Path:
        return Path(self.models_root) / "gemma-4-26B" / "gemma-4-26B-A4B-it-Q4_K_M.gguf"

    @property
    def gemma4_e4b_hf(self) -> Path:
        return Path(self.models_root) / "gemma-4-E4B-it"

    @property
    def eurollm_q6kl(self) -> Path:
        return Path(self.models_root) / "eurollm" / "EuroLLM-9B-Instruct-Q6_K_L.gguf"

    @property
    def llama3_8b_q8(self) -> Path:
        return Path(self.models_root) / "llama-3-8b-abliterated-v3" / "Meta-Llama-3-8B-Instruct-Q8_0.gguf"

    # ── Fine-tuned modely v knihy_root ───────────────────────────────────────
    @property
    def e4b_v9_translate(self) -> Path:
        return Path(self.knihy_root) / "gguf_translate_e4b_v9" / "gemma4-e4b-sk-translate-v9-q4km.gguf"

    @property
    def _26b_q4km_ft(self) -> Path:
        return Path(self.knihy_root) / "26b_translator_q4km.gguf"

    @property
    def _26b_q5km_ft(self) -> Path:
        return Path(self.knihy_root) / "26b_translator_q5km.gguf"

    @property
    def _26b_q5kl_ft(self) -> Path:
        return Path(self.knihy_root) / "26b_translator_q5kl.gguf"

    @property
    def _26b_q6k_ft(self) -> Path:
        return Path(self.knihy_root) / "26b_translator_q6k.gguf"

    # ── Mapping ollama meno → GGUF cesta (pre fallback) ──────────────────────
    def ollama_to_gguf(self, ollama_name: str) -> Optional[Path]:
        mapping = {
            "26b-translator-q5:latest":  self._26b_q5km_ft,
            "26b-translator-q5l:latest": self._26b_q5kl_ft,
            "26b-translator-q6:latest":  self._26b_q6k_ft,
            "26b-translator:latest":     self._26b_q4km_ft,
        }
        return mapping.get(ollama_name)

    # ── Validation ───────────────────────────────────────────────────────────
    def validate(self) -> dict[str, bool]:
        """Vráti slovník {meno_cesty: existuje?} pre všetky koreňové cesty."""
        return {
            "miniforge_root":  Path(self.miniforge_root).exists(),
            "models_root":     Path(self.models_root).exists(),
            "knihy_root":      Path(self.knihy_root).exists(),
            "ollama_bin":      Path(self.ollama_bin).exists(),
            "lmstudio_bin":    Path(self.lmstudio_bin).exists(),
            "fish_legacy":     Path(self.fish_legacy_root).exists(),
            "dataset_root":    Path(self.dataset_root).exists(),
        }


# ── Load / save ───────────────────────────────────────────────────────────────
def load_paths() -> Paths:
    """Načíta cesty z ~/.config/videotranslator/paths.json alebo auto-detect."""
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return Paths(**{k: v for k, v in data.items() if k in Paths.__dataclass_fields__})
        except Exception as e:
            print(f"[paths] Config load error: {e}, using defaults")

    return Paths(
        miniforge_root=str(_detect_miniforge()),
        models_root=str(_detect_models_root()),
        knihy_root=str(_detect_knihy_root()),
        ollama_bin=str(_detect_ollama_bin()),
        lmstudio_bin=str(_detect_lmstudio_bin()),
        fish_legacy_root=str(_detect_fish_legacy()),
        dataset_root=str(_detect_dataset_root()),
    )


def save_paths(paths: Paths) -> None:
    """Uloží koreňové cesty do ~/.config/videotranslator/paths.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(paths)
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Singleton ─────────────────────────────────────────────────────────────────
PATHS: Paths = load_paths()


if __name__ == "__main__":
    # Diagnostika: vypíše všetky cesty + validáciu
    print(f"Config file: {CONFIG_FILE}")
    print(f"  exists: {CONFIG_FILE.exists()}\n")
    print("=== PATHS (root) ===")
    for k, v in asdict(PATHS).items():
        exists = "✓" if Path(v).exists() else "✗"
        print(f"  {exists} {k:20s} = {v}")
    print("\n=== DERIVED PATHS ===")
    derived = [
        ("python_chatterbox", PATHS.python_chatterbox),
        ("python_musetalk",   PATHS.python_musetalk),
        ("python_finetune",   PATHS.python_finetune),
        ("gemma3_12b_q8",     PATHS.gemma3_12b_q8),
        ("gemma4_26b_base",   PATHS.gemma4_26b_base_q4km),
        ("gemma4_e4b_hf",     PATHS.gemma4_e4b_hf),
        ("e4b_v9_translate",  PATHS.e4b_v9_translate),
        ("_26b_q5km_ft",      PATHS._26b_q5km_ft),
        ("_26b_q6k_ft",       PATHS._26b_q6k_ft),
    ]
    for name, p in derived:
        exists = "✓" if Path(p).exists() else "✗"
        print(f"  {exists} {name:20s} = {p}")
    print("\n=== VALIDATION ===")
    for k, v in PATHS.validate().items():
        print(f"  {'✓' if v else '✗'} {k}")
