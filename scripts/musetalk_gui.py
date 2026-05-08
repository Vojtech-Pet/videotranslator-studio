"""
Simple Tkinter GUI wrapper for MUSETALK_MENU.bat workflows.

Exposes three main flows:
1) Unified pipeline (pipeline.py) for local files.
2) YouTube download + translate + optional lipsync (pipeline.py + MuseTalk).
3) Local Whisper + Translate + TTS with optional lipsync.
4) Configure API for ChatGPT/Gemini/Grok translation engines.

The GUI keeps the original defaults from MUSETALK_MENU.bat where possible.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import json
import threading
import ctypes
import time
import os
import platform
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False  # Linux/Mac
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import tkinter.font as tkfont


BASE_DIR = Path(__file__).parent.resolve()
VOICES_DIR = BASE_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)
CHATTERBOX_MODELS_DIR = BASE_DIR.parent / "models" / "chatterbox"
CHECKPOINT_FILE = BASE_DIR / "batch_checkpoint.json"
VIDEO_LISTS_FILE = BASE_DIR / "video_lists.json"
SAVED_CHECKPOINTS_FILE = BASE_DIR / "saved_checkpoints.json"

# Ensure writable cache dirs when launched from desktop
_cache_root = BASE_DIR / ".cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
os.environ.setdefault("TORCH_HOME", str(_cache_root / "torch"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache_root / "numba"))
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

EQ_BUILTIN_PRESETS: Dict[str, List[str]] = {
    "Default (Voice EQ)": [
        "highpass=f=45",
        "equalizer=f=90:t=q:w=1:g=2.8",
        "equalizer=f=180:t=q:w=1:g=1.5",
        "equalizer=f=300:t=q:w=1:g=0.8",
        "lowpass=f=9500",
        "equalizer=f=3200:t=q:w=1:g=-2.0",
        "equalizer=f=4300:t=q:w=1.2:g=-2.8",
        "equalizer=f=5800:t=q:w=1:g=-2.2",
        "equalizer=f=7000:t=q:w=1:g=-1.8",
    ],
    "Warm Voice": [
        "highpass=f=45",
        "equalizer=f=90:t=q:w=1:g=3.2",
        "equalizer=f=220:t=q:w=1:g=1.8",
        "equalizer=f=2800:t=q:w=1:g=-1.2",
        "lowpass=f=10500",
    ],
    "Speech Clarity": [
        "highpass=f=60",
        "equalizer=f=170:t=q:w=1:g=-1.0",
        "equalizer=f=3200:t=q:w=1:g=2.2",
        "equalizer=f=4500:t=q:w=1:g=1.6",
        "lowpass=f=12000",
    ],
    "De-ess Soft": [
        "highpass=f=55",
        "equalizer=f=4200:t=q:w=1.2:g=-2.2",
        "equalizer=f=6200:t=q:w=1.2:g=-2.4",
        "equalizer=f=7800:t=q:w=1:g=-1.5",
        "lowpass=f=13000",
    ],
    "Flat (Bypass-ish)": [
        "highpass=f=30",
        "lowpass=f=16000",
    ],
    "Podcast Deep": [
        "highpass=f=40",
        "equalizer=f=85:t=q:w=0.9:g=3.6",
        "equalizer=f=160:t=q:w=1:g=2.1",
        "equalizer=f=2800:t=q:w=1:g=-1.4",
        "equalizer=f=5000:t=q:w=1.2:g=-1.8",
        "lowpass=f=10000",
    ],
    "Bright Presence": [
        "highpass=f=55",
        "equalizer=f=220:t=q:w=1:g=-1.2",
        "equalizer=f=3200:t=q:w=1:g=2.8",
        "equalizer=f=5200:t=q:w=1:g=2.2",
        "equalizer=f=7600:t=q:w=1:g=1.4",
        "lowpass=f=14500",
    ],
    "Low Noise Speech": [
        "highpass=f=65",
        "equalizer=f=140:t=q:w=1:g=-0.8",
        "equalizer=f=250:t=q:w=1:g=-0.6",
        "equalizer=f=3000:t=q:w=1:g=1.5",
        "equalizer=f=6200:t=q:w=1.2:g=-2.8",
        "lowpass=f=11500",
    ],
    "Radio Mid Focus": [
        "highpass=f=80",
        "equalizer=f=180:t=q:w=1:g=-2.0",
        "equalizer=f=1200:t=q:w=1:g=1.8",
        "equalizer=f=2800:t=q:w=1:g=2.6",
        "equalizer=f=5200:t=q:w=1:g=-1.4",
        "lowpass=f=9000",
    ],
    "Soft Voiceover": [
        "highpass=f=45",
        "equalizer=f=100:t=q:w=1:g=2.0",
        "equalizer=f=260:t=q:w=1:g=1.2",
        "equalizer=f=3500:t=q:w=1:g=-1.3",
        "equalizer=f=6200:t=q:w=1:g=-2.0",
        "lowpass=f=12000",
    ],
    "Hegen": [
        # XTTS v2 → HeyGen-quality feel: warm, clean, natural presence
        "highpass=f=100",
        "equalizer=f=250:t=q:w=1.2:g=-1.5",    # reduce boxiness
        "equalizer=f=450:t=q:w=1.0:g=+1.2",    # add warmth
        "equalizer=f=900:t=q:w=0.8:g=-0.8",    # reduce muddiness
        "equalizer=f=2200:t=q:w=1.0:g=+1.0",   # add presence/clarity
        "equalizer=f=4000:t=q:w=1.2:g=-1.5",   # tame harshness
        "equalizer=f=8000:t=q:w=1.0:g=+0.8",   # add air
        "lowpass=f=12000",
    ],
    # ── Chatterbox TTS profiles ─────────────────────────────────────────────
    # HiFTGenerator vocoder (24kHz): no content >12kHz, boxiness 300-600Hz,
    # nasality 800-1.3kHz, harshness 3-5kHz, weak high-end (vocoder ceiling).
    # Source: GitHub issues devnen/Chatterbox-TTS-Server, Hacker News #44251411,
    #         arxiv HiFi-GAN vocoder research, Podigy podcast EQ guidelines.
    "Chatterbox Natural": [
        # Light general-purpose: remove boxiness, tame harshness, restore air
        "highpass=f=80",
        "equalizer=f=350:t=q:w=1.2:g=-2.5",    # cut boxiness (300-400 Hz)
        "equalizer=f=700:t=q:w=1.0:g=-1.5",    # cut nasality / mud
        "equalizer=f=2500:t=q:w=1.0:g=+1.5",   # presence / intelligibility
        "equalizer=f=5000:t=q:w=0.8:g=-1.5",   # tame male sibilance / buzz
        "highshelf=f=9000:g=+2.0",              # restore air (vocoder ceiling)
        "lowpass=f=11500",
    ],
    "Chatterbox Voice Clone": [
        # Voice cloning mode – typically more muffled/boxy, needs stronger treatment
        "highpass=f=80",
        "equalizer=f=280:t=q:w=1.2:g=-3.5",    # strong boxiness cut
        "equalizer=f=600:t=q:w=1.0:g=-2.5",    # mud cut
        "equalizer=f=950:t=q:w=1.0:g=-2.0",    # nasality cut
        "equalizer=f=2200:t=q:w=1.0:g=+1.8",   # presence
        "equalizer=f=4500:t=q:w=0.8:g=-2.0",   # harshness / vocoder buzz
        "equalizer=f=7000:t=q:w=1.0:g=-1.5",   # de-ess (female sibilance zone)
        "highshelf=f=9000:g=+2.5",              # air restoration
        "lowpass=f=11000",
    ],
    "Chatterbox Audiobook": [
        # ACX-compatible: clean, warm, intelligible – no harsh peaks
        # Target: -18 LUFS / -3 dBFS TP (configured in loudnorm during polish)
        "highpass=f=80",
        "equalizer=f=350:t=q:w=1.5:g=-3.0",    # boxiness
        "equalizer=f=700:t=q:w=1.0:g=-2.0",    # mud
        "equalizer=f=2500:t=q:w=1.0:g=+1.0",   # clarity
        "equalizer=f=5000:t=q:w=0.5:g=-1.5",   # harshness
        "highshelf=f=9500:g=+2.0",              # air
        "lowpass=f=11500",
    ],
    "Chatterbox Podcast": [
        # Punchy, forward-sounding for narration / dubbing
        "highpass=f=80",
        "equalizer=f=200:t=q:w=1.0:g=+1.5",    # body / warmth
        "equalizer=f=400:t=q:w=1.2:g=-2.0",    # boxiness
        "equalizer=f=800:t=q:w=1.0:g=-1.8",    # mud / honk
        "equalizer=f=3000:t=q:w=1.0:g=+2.0",   # presence / punch
        "equalizer=f=5500:t=q:w=0.8:g=-2.0",   # harshness
        "highshelf=f=9000:g=+1.5",              # air
        "lowpass=f=11500",
    ],
}

DEFAULT_CHATTERBOX_MODELS = [
    "t3_sk_v2.2.safetensors",
    "t3_sk_v2.3-edge-r2.safetensors",
    "t3_cs.safetensors",
    "t3_finetuned_v0.1.safetensors",
]


# ---------------------------------------------------------------------------
# Saved checkpoints management (named batch states with progress)
# ---------------------------------------------------------------------------
def _load_saved_checkpoints() -> Dict[str, Any]:
    """Load all named checkpoints from JSON."""
    if SAVED_CHECKPOINTS_FILE.exists():
        try:
            with open(SAVED_CHECKPOINTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_saved_checkpoints(data: Dict[str, Any]) -> None:
    """Save all named checkpoints to JSON."""
    try:
        with open(SAVED_CHECKPOINTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save checkpoints: {e}")


# ---------------------------------------------------------------------------
# Video list management (named collections of videos)
# ---------------------------------------------------------------------------
def _load_video_lists() -> Dict[str, List[str]]:
    """Load all named video lists from JSON."""
    if VIDEO_LISTS_FILE.exists():
        try:
            with open(VIDEO_LISTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_video_lists(lists: Dict[str, List[str]]) -> None:
    """Save all named video lists to JSON."""
    try:
        with open(VIDEO_LISTS_FILE, "w", encoding="utf-8") as f:
            json.dump(lists, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save video lists: {e}")


# ---------------------------------------------------------------------------
# Batch checkpoint management
# ---------------------------------------------------------------------------
@dataclass
class BatchItem:
    path: str
    status: str = "pending"  # pending, processing, done, failed
    error: str = ""

def save_checkpoint(items: List[BatchItem], current_index: int, settings_snapshot: Dict[str, Any]) -> None:
    """Save batch progress to checkpoint file."""
    data = {
        "items": [{"path": i.path, "status": i.status, "error": i.error} for i in items],
        "current_index": current_index,
        "settings": settings_snapshot,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

def load_checkpoint() -> tuple[List[BatchItem], int, Dict[str, Any]] | None:
    """Load batch progress from checkpoint file."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = [BatchItem(path=i["path"], status=i["status"], error=i.get("error", "")) for i in data["items"]]
        return items, data["current_index"], data.get("settings", {})
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None

def clear_checkpoint() -> None:
    """Remove checkpoint file after successful completion."""
    try:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
    except Exception:
        pass
# Cross-platform support (Windows/Linux)
import platform
IS_WINDOWS = platform.system() == "Windows"
USER_PROFILE = Path(os.environ.get("USERPROFILE" if IS_WINDOWS else "HOME", os.path.expanduser("~")))
PYTHON_EXE = "python.exe" if IS_WINDOWS else "python"

if IS_WINDOWS:
    DEFAULT_PY = USER_PROFILE / "miniconda3" / "envs" / "heygen_local" / PYTHON_EXE
    if not DEFAULT_PY.exists():
        DEFAULT_PY = USER_PROFILE / "miniconda3" / PYTHON_EXE
    DEFAULT_DIAR_PY = USER_PROFILE / "miniconda3" / "envs" / "heygen_diar" / PYTHON_EXE
else:
    DEFAULT_PY = USER_PROFILE / "miniforge3" / "envs" / "musetalk_env" / "bin" / PYTHON_EXE
    if not DEFAULT_PY.exists():
        DEFAULT_PY = USER_PROFILE / "miniforge3" / "bin" / PYTHON_EXE
    DEFAULT_DIAR_PY = USER_PROFILE / "miniforge3" / "envs" / "heygen_diar" / PYTHON_EXE
DEFAULT_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _path_str(p: Path | str) -> str:
    return str(Path(p))


def _latest_mp4(search_root: Path) -> Path | None:
    candidates = list(search_root.glob("*.mp4"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _log_command(cmd: List[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class CommandRunner:
    def __init__(self, log_fn, start_cb=None, done_cb=None, progress_cb=None):
        self.log_fn = log_fn
        self.start_cb = start_cb
        self.done_cb = done_cb
        self.progress_cb = progress_cb
        self.proc: subprocess.Popen | None = None

    def run(
        self,
        cmd: List[str],
        workdir: Path | None = None,
        base_progress: float = 0,
        progress_range: float = 100,
        on_line=None,
        extra_env: Dict[str, str] | None = None,
    ) -> int:
        """Run command and parse progress from output.

        Args:
            base_progress: Starting progress value (0-100)
            progress_range: How much progress this command represents
        """
        import re
        workdir = workdir or BASE_DIR
        self.log_fn(f"[RUN] {workdir}> {_log_command(cmd)}")
        flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags = subprocess.CREATE_NO_WINDOW
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # Better CUDA error messages
        # Avoid CUDA visibility being altered mid-process
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("NVIDIA_VISIBLE_DEVICES", None)
        if extra_env:
            env.update(extra_env)
        if self.start_cb:
            self.start_cb()
        if self.progress_cb:
            self.progress_cb(base_progress + 1)
        self.log_fn("[INFO] Spúšťam proces (načítavanie modelov môže trvať chvíľu)...")

        # Patterns to detect progress
        segment_pattern = re.compile(r'[Ss]egment\s+(\d+)\s*/\s*(\d+)', re.IGNORECASE)
        step_pattern = re.compile(r'\[STEP\s+(\d+)\s*/\s*(\d+)\]', re.IGNORECASE)
        percent_pattern = re.compile(r'(\d+)%\|')  # tqdm style
        tqdm_pattern = re.compile(r'(\d+)/(\d+)\s*\[')  # tqdm iterations
        # Pattern to extract ETA from tqdm: [00:49<18:15, 14.10it/s] -> extracts "18:15"
        tqdm_eta_pattern = re.compile(r'\[[\d:]+<([\d:]+)')  # tqdm iterations

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=_path_str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
                env=env,
            )
            assert self.proc.stdout is not None
            
            # Read character by character to handle \r (progress bars)
            buffer = []
            while True:
                char = self.proc.stdout.read(1)
                if not char:
                    if buffer:
                        line = "".join(buffer).strip()
                        if line:
                            self.log_fn(line)
                    break
                
                if char == '\r' or char == '\n':
                    line = "".join(buffer).strip()
                    buffer = []
                    if not line:
                        continue
                        
                    self.log_fn(line)

                    # Custom line handler (if provided, can override default progress parsing)
                    if on_line and on_line(line):
                        continue

                    # Try to parse progress from output
                    if self.progress_cb:
                        progress = None

                        # Check for segment progress
                        m = segment_pattern.search(line)
                        if m:
                            current, total = int(m.group(1)), int(m.group(2))
                            if total > 0: progress = (current / total) * 100

                        # Check for step progress
                        if progress is None:
                            m = step_pattern.search(line)
                            if m:
                                current, total = int(m.group(1)), int(m.group(2))
                                if total > 0: progress = (current / total) * 100

                        # Check for tqdm percentage
                        if progress is None:
                            m = percent_pattern.search(line)
                            if m: progress = float(m.group(1))

                        # Check for tqdm iterations
                        if progress is None:
                            m = tqdm_pattern.search(line)
                            if m:
                                current, total = int(m.group(1)), int(m.group(2))
                                if total > 0: progress = (current / total) * 100

                        # Apply progress within the range
                        if progress is not None:
                            actual_progress = base_progress + (progress / 100) * progress_range
                            self.progress_cb(min(99, actual_progress))
                else:
                    buffer.append(char)

            self.proc.wait()
            return_code = self.proc.returncode
            self.log_fn(f"[EXIT] {return_code}")
            if self.progress_cb:
                self.progress_cb(base_progress + progress_range)
            return return_code
        finally:
            self.proc = None
            if self.done_cb:
                self.done_cb()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.log_fn(f"[STOP] Ukončujem proces PID: {self.proc.pid}...")
            try:
                # Use taskkill to terminate the entire process tree on Windows
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    check=True,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                self.log_fn(f"[STOP] Proces {self.proc.pid} a jeho podprocesy boli úspešne ukončené.")
            except Exception as e:
                self.log_fn(f"[ERROR] taskkill zlyhal: {e}. Skúšam proc.kill().")
                try:
                    self.proc.kill()
                    self.log_fn(f"[STOP] Proces {self.proc.pid} bol ukončený cez proc.kill().")
                except Exception as e2:
                    self.log_fn(f"[ERROR] proc.kill() tiež zlyhal: {e2}")
        else:
            self.log_fn("[INFO] Žiadny aktívny proces na zastavenie.")

# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------
@dataclass
class AppSettings:
    py: Path = DEFAULT_PY
    diar_py: Path = DEFAULT_DIAR_PY
    ffmpeg: str = DEFAULT_FFMPEG
    export_onnx_path: str = ""
    def_tts_engine: str = "chatterbox"
    def_trans_engine: str = "madlad"
    def_clone_voice: bool = False
    def_emotion_clone: bool = False
    def_local_keep_music: bool = False
    def_denoise: bool = False
    def_denoise_preset: str = "mild"
    def_auto_speech_gain: bool = False
    openai_api_key: str = ""
    openai_translate_model: str = "gpt-5-mini-2025-08-07"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_tts_base_url: str = "https://api.openai.com/v1"
    grok_api_key: str = ""
    grok_translate_model: str = "grok-3-mini"
    grok_tts_model: str = "grok-3-mini"
    grok_tts_voice: str = "alloy"
    grok_tts_base_url: str = "https://api.x.ai/v1"
    gemini_api_key: str = ""
    gemini_translate_model: str = "gemini-2.5-flash"
    gemini_tts_model: str = "gemini-2.5-flash"
    gemini_tts_voice: str = "alloy"
    gemini_tts_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    hf_api_key: str = ""
    def_chatterbox_model: str = "t3_sk_v2.2.safetensors"
    def_speech_gain: str = "1.0"
    def_music_volume: str = "0.20"
    def_base_tempo: str = "1.08"
    def_stretch_min: str = "0.85"
    def_stretch_max: str = "1.70"
    def_pause_gap_s: str = "2.50"
    def_content_type: str = "general"
    def_vad_min_speech_ms: str = "4000"
    def_vad_min_silence_ms: str = "5000"
    def_vad_max_speech_s: str = "90.0"
    def_log_height: int = 12
    def_window_width: int = 1150
    def_window_height: int = 1100
    def_dark_mode: bool = True
    def_batch_qa: bool = False

    def ffmpeg_dir(self) -> str:
        p = Path(self.ffmpeg)
        if p.is_file():
            return str(p.parent)
        return ""

    def _secret_free_dict(self) -> dict:
        return {
            "py": str(self.py), "diar_py": str(self.diar_py), "ffmpeg": self.ffmpeg,
            "export_onnx_path": self.export_onnx_path,
            "def_tts_engine": self.def_tts_engine,
            "def_trans_engine": self.def_trans_engine,
            "def_clone_voice": self.def_clone_voice,
            "def_emotion_clone": self.def_emotion_clone,
            "def_local_keep_music": self.def_local_keep_music,
            "def_denoise": self.def_denoise,
            "def_denoise_preset": self.def_denoise_preset,
            "def_auto_speech_gain": self.def_auto_speech_gain,
            "openai_api_key": "",
            "openai_translate_model": self.openai_translate_model,
            "openai_tts_model": self.openai_tts_model,
            "openai_tts_voice": self.openai_tts_voice,
            "openai_tts_base_url": self.openai_tts_base_url,
            "grok_api_key": "",
            "grok_translate_model": self.grok_translate_model,
            "grok_tts_model": self.grok_tts_model,
            "grok_tts_voice": self.grok_tts_voice,
            "grok_tts_base_url": self.grok_tts_base_url,
            "gemini_api_key": "",
            "gemini_translate_model": self.gemini_translate_model,
            "gemini_tts_model": self.gemini_tts_model,
            "gemini_tts_voice": self.gemini_tts_voice,
            "gemini_tts_base_url": self.gemini_tts_base_url,
            "hf_api_key": "",
            "def_chatterbox_model": self.def_chatterbox_model,
            "def_speech_gain": self.def_speech_gain,
            "def_music_volume": self.def_music_volume,
            "def_base_tempo": self.def_base_tempo,
            "def_stretch_min": self.def_stretch_min,
            "def_stretch_max": self.def_stretch_max,
            "def_pause_gap_s": self.def_pause_gap_s,
            "def_content_type": self.def_content_type,
            "def_vad_min_speech_ms": self.def_vad_min_speech_ms,
            "def_vad_min_silence_ms": self.def_vad_min_silence_ms,
            "def_vad_max_speech_s": self.def_vad_max_speech_s,
            "def_log_height": self.def_log_height,
            "def_window_width": self.def_window_width,
            "def_window_height": self.def_window_height,
            "def_dark_mode": self.def_dark_mode,
            "def_batch_qa": self.def_batch_qa,
        }

    def load(self):
        config_path = BASE_DIR / "musetalk_gui_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data.setdefault("openai_translate_model", "gpt-5-mini-2025-08-07")
                    data.setdefault("grok_translate_model", data.get("grok_tts_model", "grok-3-mini"))
                    data.setdefault("gemini_translate_model", data.get("gemini_tts_model", "gemini-2.5-flash"))
                    if "py" in data: self.py = Path(data["py"])
                    if "diar_py" in data: self.diar_py = Path(data["diar_py"])
                    if "ffmpeg" in data: self.ffmpeg = data["ffmpeg"]
                    if "export_onnx_path" in data: self.export_onnx_path = data["export_onnx_path"]
                    if "def_tts_engine" in data: self.def_tts_engine = data["def_tts_engine"]
                    if "def_trans_engine" in data: self.def_trans_engine = data["def_trans_engine"]
                    if "def_clone_voice" in data: self.def_clone_voice = data["def_clone_voice"]
                    if "def_emotion_clone" in data: self.def_emotion_clone = bool(data["def_emotion_clone"])
                    if "def_local_keep_music" in data: self.def_local_keep_music = bool(data["def_local_keep_music"])
                    if "def_denoise" in data: self.def_denoise = bool(data["def_denoise"])
                    if "def_denoise_preset" in data: self.def_denoise_preset = data["def_denoise_preset"]
                    if "def_auto_speech_gain" in data: self.def_auto_speech_gain = bool(data["def_auto_speech_gain"])
                    if "openai_translate_model" in data: self.openai_translate_model = data["openai_translate_model"]
                    if "openai_tts_model" in data: self.openai_tts_model = data["openai_tts_model"]
                    if "openai_tts_voice" in data: self.openai_tts_voice = data["openai_tts_voice"]
                    if "openai_tts_base_url" in data: self.openai_tts_base_url = data["openai_tts_base_url"]
                    if "grok_translate_model" in data: self.grok_translate_model = data["grok_translate_model"]
                    if "grok_tts_model" in data: self.grok_tts_model = data["grok_tts_model"]
                    if "grok_tts_voice" in data: self.grok_tts_voice = data["grok_tts_voice"]
                    if "grok_tts_base_url" in data: self.grok_tts_base_url = data["grok_tts_base_url"]
                    if "gemini_translate_model" in data: self.gemini_translate_model = data["gemini_translate_model"]
                    if "gemini_tts_model" in data: self.gemini_tts_model = data["gemini_tts_model"]
                    if "gemini_tts_voice" in data: self.gemini_tts_voice = data["gemini_tts_voice"]
                    if "gemini_tts_base_url" in data: self.gemini_tts_base_url = data["gemini_tts_base_url"]
                    if "def_chatterbox_model" in data: self.def_chatterbox_model = data["def_chatterbox_model"]
                    if "def_speech_gain" in data: self.def_speech_gain = data["def_speech_gain"]
                    if "def_music_volume" in data: self.def_music_volume = data["def_music_volume"]
                    if "def_base_tempo" in data: self.def_base_tempo = data["def_base_tempo"]
                    if "def_stretch_min" in data: self.def_stretch_min = data["def_stretch_min"]
                    if "def_stretch_max" in data: self.def_stretch_max = data["def_stretch_max"]
                    if "def_pause_gap_s" in data: self.def_pause_gap_s = data["def_pause_gap_s"]
                    if "def_content_type" in data: self.def_content_type = data["def_content_type"]
                    if "def_vad_min_speech_ms" in data: self.def_vad_min_speech_ms = data["def_vad_min_speech_ms"]
                    if "def_vad_min_silence_ms" in data: self.def_vad_min_silence_ms = data["def_vad_min_silence_ms"]
                    if "def_vad_max_speech_s" in data: self.def_vad_max_speech_s = data["def_vad_max_speech_s"]
                    if "def_log_height" in data: self.def_log_height = int(data["def_log_height"])
                    if "def_window_width" in data: self.def_window_width = int(data["def_window_width"])
                    if "def_window_height" in data: self.def_window_height = int(data["def_window_height"])
                    if "def_dark_mode" in data: self.def_dark_mode = bool(data["def_dark_mode"])
                    if "def_batch_qa" in data: self.def_batch_qa = bool(data["def_batch_qa"])
            except Exception as e:
                print(f"Failed to load config: {e}")
        if self.def_tts_engine not in ("chatterbox", "xtts", "edge"):
            self.def_tts_engine = "chatterbox"
        if self.def_trans_engine not in ("llama", "madlad", "multislav5lang", "hybrid", "google_madlad", "chatgpt", "grok", "gemini", "google"):
            self.def_trans_engine = "madlad"
        if not self.py.exists():
            self.py = DEFAULT_PY

    def save(self):
        config_path = BASE_DIR / "musetalk_gui_config.json"
        data = self._secret_free_dict()
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")


# ---------------------------------------------------------------------------
# Tooltip helper (pyvideotrans-style hover help)
# ---------------------------------------------------------------------------
class ToolTip:
    """Lightweight hover tooltip for any Tkinter widget."""
    def __init__(self, widget: tk.Widget, text: str, delay: int = 600) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self, _event=None) -> None:
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    def _show(self) -> None:
        if not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self.text, justify="left",
            background="#fffacd", relief="solid", borderwidth=1,
            font=("DejaVu Sans", 9), wraplength=340, padx=5, pady=4,
        ).pack()


def tip(widget: tk.Widget, text: str) -> None:
    """Attach a hover tooltip to *widget*."""
    ToolTip(widget, text)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class MuseTalkGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VideoTranslator GUI")
        self.settings = AppSettings()
        self.settings.load()
        self.font_ui_family, self.font_mono_family = self._detect_fonts()
        self._configure_default_fonts()
        self.geometry(f"{self.settings.def_window_width}x{self.settings.def_window_height}")
        self.dark_mode = tk.BooleanVar(value=self.settings.def_dark_mode)
        self.log_queue: queue.Queue = queue.Queue()
        self.busy_count = 0
        self.runner = CommandRunner(self._queue_log, self._queue_start, self._queue_stop, self._queue_progress)
        self.keep_awake = tk.BooleanVar(value=True)
        self.shutdown_after = tk.BooleanVar(value=False)
        self.progress_start_ts: float | None = None
        self.eta_history: list[tuple[float, float]] = []  # (timestamp, progress) pairs for smoothed ETA
        self.eta_smoothed: float | None = None  # exponential moving average of ETA
        self._awake_timer_id = None  # Timer for periodic sleep prevention refresh
        self._eta_timer_id = None  # Timer for periodic ETA updates
        self._last_progress_change: float = 0  # Timestamp of last progress change
        self._eta_calculated_at: float = 0  # When ETA was last calculated
        self._target_progress: float = 0  # Target progress for smooth animation
        self._progress_anim_id = None  # Timer for progress animation
        self._theme = {}
        # Batch ETA tracking
        self.batch_start_ts: float | None = None  # When batch started
        self.batch_total_videos: int = 0  # Total videos in batch
        self.batch_completed_videos: int = 0  # Completed videos
        self.batch_video_times: list[float] = []  # Time taken for each completed video
        # default toggles for forms
        self.def_local_lipsync = tk.BooleanVar(value=True)
        self.def_local_keep_music = tk.BooleanVar(value=self.settings.def_local_keep_music)
        self.def_local_context = tk.BooleanVar(value=True)
        self.def_local_voice_eq = tk.BooleanVar(value=True)
        self.def_local_multi_voice = tk.BooleanVar(value=False)
        self.def_local_clone_voice = tk.BooleanVar(value=self.settings.def_clone_voice)
        self.def_local_emotion_clone = tk.BooleanVar(value=self.settings.def_emotion_clone)
        self.def_local_denoise = tk.BooleanVar(value=self.settings.def_denoise)
        self.def_local_auto_speech_gain = tk.BooleanVar(value=self.settings.def_auto_speech_gain)
        self.export_type = tk.StringVar(value="onnx")
        self.export_models = tk.StringVar(value="all")
        self.export_fp16 = tk.BooleanVar(value=True)
        self.fps25_src = tk.StringVar(value="")
        self.fps25_out = tk.StringVar(value="")
        self.split_src = tk.StringVar(value="")
        self.split_minutes = tk.DoubleVar(value=1.0)
        self.join_dir = tk.StringVar(value="")
        self._build_ui()
        self._apply_theme(self.dark_mode.get())
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(150, self._drain_log)
        # Check for existing checkpoint on startup
        self.after(500, self._check_startup_checkpoint)

    def _check_startup_checkpoint(self) -> None:
        """Check if there's an unfinished batch on startup and notify user."""
        checkpoint = load_checkpoint()
        if checkpoint:
            items, _, _ = checkpoint
            pending = [i for i in items if i.status in ("pending", "failed")]
            if pending:
                self._queue_log(f"[INFO] Nájdený nedokončený batch ({len(pending)} videí). Kliknite 'Pokračovať' pre obnovenie.")

    # -------------------- UI builders --------------------
    # ---- Left nav items (key, label) ----
    _NAV_ITEMS = [
        ("local",    "Lokálny súbor"),
        ("youtube",  "Stiahnutie videa"),
        ("api",      "API kľúče"),
        ("tools",    "Nástroje"),
        ("eq",       "EQ Profil"),
        ("settings", "Nastavenia"),
    ]

    def _build_ui(self) -> None:
        self._init_styles()
        self._pages: dict[str, ttk.Frame] = {}
        self._nav_btns: dict[str, tk.Button] = {}
        self._current_page: str = "local"

        # ── Outer container (nav | content | steps-sidebar) ──────────────────
        main_container = tk.Frame(self)
        main_container.pack(fill="both", expand=True)

        # Left navigation bar
        self._nav_bar = tk.Frame(main_container, width=148, bg="#19232D")
        self._nav_bar.pack(side="left", fill="y")
        self._nav_bar.pack_propagate(False)

        _nav_title = tk.Label(
            self._nav_bar, text="Video\nTranslator",
            bg="#19232D", fg="#DFE1E2",
            font=(self.font_ui_family, 10, "bold"),
            pady=12,
        )
        _nav_title.pack(fill="x")
        tk.Frame(self._nav_bar, height=1, bg="#455364").pack(fill="x")

        for page_key, label in self._NAV_ITEMS:
            btn = tk.Button(
                self._nav_bar,
                text=label,
                bg="#19232D", fg="#DFE1E2",
                activebackground="#1A72BB", activeforeground="#ffffff",
                relief="flat", anchor="w",
                padx=14, pady=9,
                cursor="hand2",
                wraplength=130, justify="left",
                font=(self.font_ui_family, 9),
                command=lambda k=page_key: self._show_nav_page(k),
            )
            btn.pack(fill="x")
            self._nav_btns[page_key] = btn

        # Content area — all pages stacked here
        self._content_area = ttk.Frame(main_container)
        self._content_area.pack(side="left", fill="both", expand=True)
        self._content_area.grid_rowconfigure(0, weight=1)
        self._content_area.grid_columnconfigure(0, weight=1)

        # Right side: Steps Sidebar
        self.sidebar = ttk.Frame(main_container, width=240, padding=10)
        self.sidebar.pack(side="right", fill="y", padx=(0, 10), pady=10)

        # Sidebar Title
        ttk.Label(self.sidebar, text="Postup spracovania", font=(self.font_ui_family, 10, "bold")).pack(anchor="w", pady=(0, 10))

        self.steps_frame = ttk.Frame(self.sidebar)
        self.steps_frame.pack(fill="x")

        # Placeholder
        ttk.Label(self.steps_frame, text="Pripravené", foreground="gray").pack(anchor="w")

        # Per-task progress list (pyvideotrans-style)
        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", pady=(12, 6))
        ttk.Label(self.sidebar, text="Videá", font=(self.font_ui_family, 9, "bold")).pack(anchor="w")
        self._task_list_frame = ttk.Frame(self.sidebar)
        self._task_list_frame.pack(fill="both", expand=True, pady=(4, 0))
        self._task_widgets: list[dict] = []

        self.local_tab(self._content_area)
        self.youtube_tab(self._content_area)
        self.api_tab(self._content_area)
        self.tools_tab(self._content_area)
        self.eq_tab(self._content_area)
        self.settings_tab(self._content_area)

        # Show first page and highlight its nav button
        self._show_nav_page("local")

        self._apply_progress_style()
        self._apply_button_styles()

        # Bottom frame for controls and log
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        # Control frame for buttons
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill="x", pady=(5, 2))
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        self.start_button = ttk.Button(button_frame, text="Spustiť", style="Green.TButton", command=self._run_active_tab)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.stop_button = ttk.Button(button_frame, text="Stop", style="Red.TButton", command=self._stop_all_processes, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # Progress bar — current file
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            bottom_frame, mode="determinate", maximum=100.0, variable=self.progress_var, style="Success.Horizontal.TProgressbar"
        )
        self.progress.pack(fill="x", pady=(2, 1), ipady=4)

        self.progress_label = tk.Label(
            self.progress,
            text="",
            anchor="center",
            bg="#e0e0e0",
            fg="#222",
            font=(self.font_ui_family, 9, "bold"),
        )
        self.progress_label.place(relx=0.5, rely=0.5, anchor="center")

        # Batch progress bar — overall batch (hidden when batch has <= 1 file)
        self.batch_progress_var = tk.DoubleVar(value=0.0)
        self.batch_progress = ttk.Progressbar(
            bottom_frame, mode="determinate", maximum=100.0, variable=self.batch_progress_var,
            style="Success.Horizontal.TProgressbar"
        )
        self.batch_progress_label = tk.Label(
            self.batch_progress,
            text="",
            anchor="center",
            bg="#e0e0e0",
            fg="#222",
            font=(self.font_ui_family, 8),
        )
        self.batch_progress_label.place(relx=0.5, rely=0.5, anchor="center")
        # Initially hidden; shown when batch has > 1 item
        self._batch_bar_visible = False

        self.log_text = tk.Text(bottom_frame, height=self.settings.def_log_height, wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=(5, 0))
        self.log_text.configure(state="disabled", font=(self.font_mono_family, 10))

        # Status bar (pyvideotrans-style bottom toolbar)
        self._status_bar = tk.Frame(self, height=28)
        self._status_bar.pack(side="bottom", fill="x")
        self._status_bar.pack_propagate(False)
        self._status_label = tk.Label(
            self._status_bar, text="  VideoTranslator  |  Pripravené",
            anchor="w", font=(self.font_ui_family, 9),
        )
        self._status_label.pack(side="left", fill="y", padx=4)
        ttk.Button(self._status_bar, text="Otvoriť výstup",
                   command=self._open_output_dir).pack(side="right", padx=4, pady=2)
        ttk.Button(self._status_bar, text="Vymazať log",
                   command=self._clear_log).pack(side="right", padx=2, pady=2)
        ttk.Checkbutton(self._status_bar, text="Tmavý režim",
                        variable=self.dark_mode,
                        command=self._toggle_dark_mode).pack(side="right", padx=8, pady=2)

    # ── Nav page switching ────────────────────────────────────────────────────
    def _show_nav_page(self, name: str) -> None:
        self._current_page = name
        if name in self._pages:
            self._pages[name].tkraise()
        # Highlight active nav button
        dark = getattr(self, "dark_mode", None)
        is_dark = dark.get() if dark else True
        active_bg  = "#1A72BB"
        active_fg  = "#ffffff"
        normal_bg  = "#19232D" if is_dark else "#32414B"
        normal_fg  = "#DFE1E2"
        for k, btn in self._nav_btns.items():
            if k == name:
                btn.configure(bg=active_bg, fg=active_fg)
            else:
                btn.configure(bg=normal_bg, fg=normal_fg)

    def _detect_fonts(self) -> tuple[str, str]:
        if IS_WINDOWS:
            return "Segoe UI", "Consolas"
        if platform.system() == "Darwin":
            return "Helvetica", "Menlo"
        return "DejaVu Sans", "DejaVu Sans Mono"

    def _configure_default_fonts(self) -> None:
        try:
            tkfont.nametofont("TkDefaultFont").configure(family=self.font_ui_family, size=10)
            tkfont.nametofont("TkTextFont").configure(family=self.font_ui_family, size=10)
            tkfont.nametofont("TkFixedFont").configure(family=self.font_mono_family, size=10)
        except Exception:
            pass

    def _init_styles(self) -> None:
        self.style = ttk.Style()
        # Use a theme that supports color customization consistently
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

    def _apply_progress_style(self) -> None:
        if self.dark_mode.get():
            trough = "#32414B"
            bg = "#1A72BB"      # pyvideotrans blue
            dark = "#346792"
        else:
            trough = "#c8d8e8"
            bg = "#1A72BB"
            dark = "#346792"
        self.style.configure(
            "Success.Horizontal.TProgressbar",
            troughcolor=trough,
            background=bg,
            bordercolor=trough,
            lightcolor=bg,
            darkcolor=dark,
        )

    def _apply_button_styles(self) -> None:
        if self.dark_mode.get():
            green_bg = "#1A72BB"      # pyvideotrans blue for Start
            green_active = "#259AE9"  # bright blue on hover
            red_bg = "#8B1A1A"
            red_active = "#c0392b"
            fg = "#DFE1E2"
        else:
            green_bg = "#1A72BB"
            green_active = "#259AE9"
            red_bg = "#dc3545"
            red_active = "#c82333"
            fg = "#ffffff"
        self.style.configure("Green.TButton", padding=(10, 6), font=(self.font_ui_family, 10, "bold"), foreground=fg, background=green_bg)
        self.style.map("Green.TButton", background=[("active", green_active), ("disabled", "#455364")], foreground=[("active", fg), ("disabled", "#788D9C")])
        self.style.configure("Red.TButton", padding=(10, 6), font=(self.font_ui_family, 10, "bold"), foreground=fg, background=red_bg)
        self.style.map("Red.TButton", background=[("active", red_active), ("disabled", "#32414B")], foreground=[("active", fg), ("disabled", "#788D9C")])

    def _apply_theme(self, dark: bool) -> None:
        # pyvideotrans-inspired color palette
        if dark:
            self._theme = {
                "bg":       "#19232D",   # primary dark bg
                "bg_alt":   "#32414B",   # secondary panels
                "bg_panel": "#455364",   # toolbar / button bg
                "fg":       "#DFE1E2",   # primary text
                "muted":    "#788D9C",   # muted / disabled text
                "accent":   "#1A72BB",   # primary blue accent
                "accent2":  "#259AE9",   # bright blue (hover)
                "entry_bg": "#19232D",   # input field bg
                "entry_fg": "#DFE1E2",
                "text_bg":  "#19232D",   # log area bg
                "text_fg":  "#DFE1E2",
                "hover_bg": "#54687A",   # hover bg
                "border":   "#455364",   # border color
            }
        else:
            self._theme = {
                "bg":       "#f0f4f8",
                "bg_alt":   "#ffffff",
                "bg_panel": "#dce6f0",
                "fg":       "#1a2633",
                "muted":    "#6a7f8e",
                "accent":   "#1A72BB",
                "accent2":  "#259AE9",
                "entry_bg": "#ffffff",
                "entry_fg": "#1a2633",
                "text_bg":  "#ffffff",
                "text_fg":  "#111111",
                "hover_bg": "#c8d8e8",
                "border":   "#b0c4d8",
            }

        t = self._theme
        self.configure(bg=t["bg"])

        # ttk styles — pyvideotrans inspired
        self.style.configure("TFrame", background=t["bg"])
        self.style.configure("TLabel", background=t["bg"], foreground=t["fg"])
        self.style.configure("TLabelframe", background=t["bg"], foreground=t["muted"],
                             bordercolor=t["border"], relief="solid")
        self.style.configure("TLabelframe.Label", background=t["bg"], foreground=t["accent2"],
                             font=(self.font_ui_family, 9, "bold"))
        self.style.configure("TCheckbutton", background=t["bg"], foreground=t["fg"])
        self.style.map("TCheckbutton", background=[("active", t["bg"])], foreground=[("active", t["fg"])])
        self.style.configure("TRadiobutton", background=t["bg"], foreground=t["fg"])
        self.style.map("TRadiobutton", background=[("active", t["bg"])], foreground=[("active", t["fg"])])
        self.style.configure("TButton", background=t["bg_panel"], foreground=t["fg"],
                             bordercolor=t["border"], padding=(8, 4))
        self.style.map(
            "TButton",
            background=[("active", t["hover_bg"]), ("pressed", t["accent"]), ("disabled", t["bg_alt"])],
            foreground=[("active", t["fg"]), ("disabled", t["muted"])],
        )
        self.style.configure("TEntry", fieldbackground=t["entry_bg"], foreground=t["entry_fg"],
                             bordercolor=t["border"], insertcolor=t["fg"])
        self.style.configure("TCombobox", fieldbackground=t["entry_bg"], foreground=t["entry_fg"],
                             selectbackground=t["accent"], selectforeground="#ffffff",
                             arrowcolor=t["fg"])
        self.style.configure("TSpinbox", fieldbackground=t["entry_bg"], foreground=t["entry_fg"],
                             arrowcolor=t["fg"])
        self.style.configure("TSeparator", background=t["border"])

        # Nav bar theming
        if hasattr(self, "_nav_bar"):
            nav_bg = "#19232D"  # always dark sidebar
            self._nav_bar.configure(bg=nav_bg)
            for w in self._nav_bar.winfo_children():
                if isinstance(w, tk.Label):
                    w.configure(bg=nav_bg, fg="#DFE1E2")
                elif isinstance(w, tk.Frame):
                    w.configure(bg="#455364")
            # Re-apply active button highlight
            active = getattr(self, "_current_page", None)
            for k, btn in getattr(self, "_nav_btns", {}).items():
                if k == active:
                    btn.configure(bg="#1A72BB", fg="#ffffff")
                else:
                    btn.configure(bg=nav_bg, fg="#DFE1E2")

        self._apply_progress_style()
        self._apply_button_styles()

        # Non-ttk widgets
        if hasattr(self, "log_text"):
            self.log_text.configure(bg=t["text_bg"], fg=t["text_fg"], insertbackground=t["text_fg"])
        if hasattr(self, "progress_label"):
            self.progress_label.configure(bg=t["bg_alt"], fg=t["fg"])
        if hasattr(self, "_status_bar"):
            self._status_bar.configure(bg=t["bg_panel"])
            if hasattr(self, "_status_label"):
                self._status_label.configure(bg=t["bg_panel"], fg=t["muted"])

        # Update step indicators with theme
        if hasattr(self, "steps_frame"):
            for w in self.steps_frame.winfo_children():
                if isinstance(w, ttk.Frame):
                    for child in w.winfo_children():
                        if isinstance(child, tk.Canvas):
                            child.configure(bg=t["bg"])

        self._apply_theme_to_widget_tree(self)

    def _apply_theme_to_widget_tree(self, root: tk.Misc) -> None:
        t = self._theme
        list_bg = t.get("text_bg", "#141414")
        list_fg = t.get("text_fg", "#e6e6e6")
        sel_bg = t.get("accent", "#007acc")
        sel_fg = "#ffffff"
        entry_bg = t.get("entry_bg", list_bg)
        entry_fg = t.get("entry_fg", list_fg)

        try:
            if isinstance(root, tk.Toplevel):
                root.configure(bg=t.get("bg", "#1e1f22"))
        except Exception:
            pass

        for w in root.winfo_children():
            try:
                if isinstance(w, tk.Listbox):
                    w.configure(
                        bg=list_bg,
                        fg=list_fg,
                        selectbackground=sel_bg,
                        selectforeground=sel_fg,
                        highlightbackground=t.get("bg", list_bg),
                    )
                elif isinstance(w, tk.Text):
                    w.configure(bg=t.get("text_bg", list_bg), fg=t.get("text_fg", list_fg), insertbackground=list_fg)
                elif isinstance(w, tk.Entry):
                    w.configure(bg=entry_bg, fg=entry_fg, insertbackground=entry_fg)
                elif isinstance(w, tk.Canvas):
                    w.configure(bg=t.get("bg", "#1e1f22"))
                elif isinstance(w, tk.Frame):
                    w.configure(bg=t.get("bg_alt", t.get("bg", "#1e1f22")))
                elif isinstance(w, tk.Label):
                    w.configure(bg=t.get("bg_alt", t.get("bg", "#1e1f22")), fg=t.get("fg", "#e6e6e6"))
            except Exception:
                pass
            self._apply_theme_to_widget_tree(w)

    def _run_active_tab(self):
        page = getattr(self, "_current_page", "local")
        if page == "local":
            self._run_local()
        elif page == "youtube":
            self._run_youtube()
        elif page == "eq":
            self._run_eq_apply()

    # -------------------- Logging --------------------
    def _queue_log(self, msg) -> None:
        self.log_queue.put(msg)

    def _queue_start(self) -> None:
        self.log_queue.put(("start", None))

    def _queue_stop(self) -> None:
        self.log_queue.put(("stop", None))

    def _queue_progress(self, value: float) -> None:
        self.log_queue.put(("progress", value))

    # -------------------- Step UI Helpers --------------------
    def _queue_step_init(self, steps: List[str]) -> None:
        self.log_queue.put(("step_init", steps))

    def _queue_step_active(self, index: int) -> None:
        self.log_queue.put(("step_active", index))

    def _queue_step_done(self, index: int) -> None:
        self.log_queue.put(("step_done", index))

    def _ui_init_steps(self, steps: List[str]) -> None:
        for widget in self.steps_frame.winfo_children():
            widget.destroy()

        self.step_widgets = []
        self._step_start_times: Dict[int, float] = {}
        bg_color = self._theme.get("bg", self.cget("bg"))
        outline = self._theme.get("muted", "#999")

        for i, step_name in enumerate(steps):
            frame = ttk.Frame(self.steps_frame)
            frame.pack(fill="x", pady=2)

            # Indicator (Canvas)
            canvas = tk.Canvas(frame, width=20, height=20, highlightthickness=0, bg=bg_color)
            canvas.pack(side="left", padx=(0, 6))
            # Draw empty circle
            oval = canvas.create_oval(2, 2, 18, 18, outline=outline, width=2)

            # Step name label
            lbl = ttk.Label(frame, text=step_name, font=(self.font_ui_family, 9))
            lbl.pack(side="left", anchor="w")

            # Timing label (shown when step is active/done)
            time_lbl = ttk.Label(frame, text="", font=(self.font_ui_family, 8),
                                 foreground=self._theme.get("muted", "#888"))
            time_lbl.pack(side="right", anchor="e", padx=(4, 0))

            self.step_widgets.append({"canvas": canvas, "oval": oval, "label": lbl, "time_lbl": time_lbl})

    def _ui_set_step(self, index: int, state: str) -> None:
        if index < 0 or index >= len(self.step_widgets):
            return

        w = self.step_widgets[index]
        canvas = w["canvas"]
        oval = w["oval"]
        lbl = w["label"]
        time_lbl = w.get("time_lbl")
        accent = self._theme.get("accent", "#007acc")
        ok = "#34c759"
        fg = self._theme.get("fg", "#222")
        muted = self._theme.get("muted", "#888")

        if state == "active":
            canvas.itemconfig(oval, outline=accent, width=3)
            lbl.configure(font=(self.font_ui_family, 9, "bold"), foreground=accent)
            # Record start time and show clock
            self._step_start_times[index] = time.time()
            if time_lbl is not None:
                start_str = time.strftime("%H:%M:%S")
                time_lbl.configure(text=start_str, foreground=accent)
        elif state == "done":
            canvas.itemconfig(oval, outline=ok, fill=ok, width=0)
            # Draw checkmark
            canvas.create_line(5, 10, 9, 14, 15, 6, fill="white", width=2)
            lbl.configure(font=(self.font_ui_family, 9), foreground=fg)
            # Show elapsed time
            if time_lbl is not None and index in self._step_start_times:
                elapsed = time.time() - self._step_start_times[index]
                m, s = divmod(int(elapsed), 60)
                elapsed_str = f"{m}:{s:02d}" if m else f"{s}s"
                time_lbl.configure(text=elapsed_str, foreground=muted)

    # -------------------- Per-task progress (pyvideotrans-style) --------------------
    def _queue_task_list_init(self, filenames: list[str]) -> None:
        self.log_queue.put(("task_list_init", filenames))

    def _queue_task_status(self, index: int, status: str) -> None:
        self.log_queue.put(("task_status", (index, status)))

    def _ui_init_task_list(self, filenames: list[str]) -> None:
        for w in self._task_list_frame.winfo_children():
            w.destroy()
        self._task_widgets = []
        _STATUS_COLORS = {"pending": "#999", "processing": "#007acc", "done": "#34c759", "failed": "#e74c3c"}
        _STATUS_SYMBOLS = {"pending": "○", "processing": "●", "done": "✓", "failed": "✗"}
        for i, fname in enumerate(filenames):
            row = ttk.Frame(self._task_list_frame)
            row.pack(fill="x", pady=1)
            sym_lbl = tk.Label(row, text="○", width=2, font=(self.font_ui_family, 10),
                               fg=_STATUS_COLORS["pending"], bg=self._theme.get("bg", self.cget("bg")))
            sym_lbl.pack(side="left", padx=(0, 4))
            name_lbl = ttk.Label(row, text=Path(fname).name[:28], font=(self.font_ui_family, 8))
            name_lbl.pack(side="left", anchor="w")
            self._task_widgets.append({"sym": sym_lbl, "name": name_lbl, "status": "pending"})

    def _ui_set_task_status(self, index: int, status: str) -> None:
        if index < 0 or index >= len(self._task_widgets):
            return
        _STATUS_COLORS = {"pending": "#999", "processing": "#007acc", "done": "#34c759", "failed": "#e74c3c"}
        _STATUS_SYMBOLS = {"pending": "○", "processing": "●", "done": "✓", "failed": "✗"}
        w = self._task_widgets[index]
        w["status"] = status
        w["sym"].configure(text=_STATUS_SYMBOLS.get(status, "○"), fg=_STATUS_COLORS.get(status, "#999"))
        bold = status == "processing"
        w["name"].configure(font=(self.font_ui_family, 8, "bold" if bold else ""))

    # ---- Edge-TTS voice catalogue (name, lang_code, gender m/f) ----
    _ALL_EDGE_VOICES: list[tuple[str, str, str]] = [
        ("cs-CZ-AntoninNeural", "cs", "m"),
        ("cs-CZ-VlastaNeural",  "cs", "f"),
        ("sk-SK-LukasNeural",   "sk", "m"),
        ("sk-SK-ViktoriaNeural","sk", "f"),
        ("en-US-AriaNeural",    "en", "f"),
        ("en-US-GuyNeural",     "en", "m"),
        ("en-US-JennyNeural",   "en", "f"),
        ("en-US-DavisNeural",   "en", "m"),
        ("en-US-AmberNeural",   "en", "f"),
        ("en-US-BrianNeural",   "en", "m"),
        ("en-GB-RyanNeural",    "en", "m"),
        ("en-GB-SoniaNeural",   "en", "f"),
        ("de-DE-KatjaNeural",   "de", "f"),
        ("de-DE-ConradNeural",  "de", "m"),
        ("fr-FR-DeniseNeural",  "fr", "f"),
        ("fr-FR-HenriNeural",   "fr", "m"),
        ("es-ES-AlvaroNeural",  "es", "m"),
        ("es-ES-ElviraNeural",  "es", "f"),
        ("pl-PL-MarekNeural",   "pl", "m"),
        ("pl-PL-ZofiaNeural",   "pl", "f"),
        ("ru-RU-DmitryNeural",  "ru", "m"),
        ("ru-RU-SvetlanaNeural","ru", "f"),
    ]

    # ---- Tab badge helpers ----
    _TAB_BASE_NAMES = {
        "Lokálny súbor": "Lokálny súbor",
        "YouTube preklad": "YouTube preklad",
        "API": "API",
        "Nástroje": "Nástroje",
        "EQ": "EQ",
        "Nastavenia": "Nastavenia",
    }

    def _set_tab_badge(self, badge: str) -> None:
        """Append a badge symbol to the active nav button label."""
        try:
            page = getattr(self, "_current_page", None)
            if not page:
                return
            self._active_page_key = page
            btn = self._nav_btns.get(page)
            if btn:
                base = next((lbl for k, lbl in self._NAV_ITEMS if k == page), page)
                btn.configure(text=f"{base} {badge}" if badge else base)
        except Exception:
            pass

    def _clear_tab_badge(self, success: bool = True) -> None:
        """Replace spinning badge with ✓ or ✗ on the nav button that started the job."""
        try:
            page = getattr(self, "_active_page_key", None)
            if not page:
                return
            btn = self._nav_btns.get(page)
            if btn:
                base = next((lbl for k, lbl in self._NAV_ITEMS if k == page), page)
                btn.configure(text=f"{base} {'✓' if success else '✗'}")
        except Exception:
            pass

    def _spin_tab_badge(self) -> None:
        """Animate a rotating indicator on the active nav button while busy."""
        _frames = ("◐", "◓", "◑", "◒")
        if not hasattr(self, "_spin_frame"):
            self._spin_frame = 0
        if self.busy_count > 0:
            try:
                page = getattr(self, "_active_page_key", None)
                btn = self._nav_btns.get(page) if page else None
                if btn:
                    base = next((lbl for k, lbl in self._NAV_ITEMS if k == page), page)
                    frame = _frames[self._spin_frame % len(_frames)]
                    btn.configure(text=f"{base} {frame}")
            except Exception:
                pass
            self._spin_frame += 1
            self._spin_timer_id = self.after(250, self._spin_tab_badge)
        else:
            self._spin_frame = 0

    # ---- Edge-TTS voice filter ----
    def _filter_edge_voices(self) -> None:
        """Filter voice combobox(es) by language and gender."""
        lang_v = getattr(self, "edge_lang_filter_var", None)
        gender_v = getattr(self, "edge_gender_filter_var", None)
        lang = lang_v.get() if lang_v else "všetky"
        gender = gender_v.get() if gender_v else "všetky"
        filtered = [
            name for name, lc, gc in self._ALL_EDGE_VOICES
            if (lang == "všetky" or lc == lang)
            and (gender == "všetky" or (gc == "m") == (gender == "muž"))
        ]
        if not filtered:
            filtered = [n for n, _, _ in self._ALL_EDGE_VOICES]
        cb = getattr(self, "_edge_voice_cb", None)
        if cb:
            cb["values"] = filtered
            if self.edge_tts_voice_var.get() not in filtered:
                self.edge_tts_voice_var.set(filtered[0])
        cb2 = getattr(self, "_edge_voice_cb2", None)
        if cb2:
            cb2["values"] = [""] + filtered

    def _start_progress(self):
        self.busy_count += 1
        if self.busy_count == 1:
            # Clear log from previous run
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")
            self.start_button.configure(text="Spustiť", style="Green.TButton", state="disabled")
            self.stop_button.configure(text="Stop", style="Red.TButton", state="normal")
            self.progress_var.set(1)
            self._target_progress = 1
            self._set_awake(True)
            self.progress_start_ts = time.time()
            self.eta_history = [(time.time(), 0)]  # Reset ETA history
            self.eta_smoothed = None
            self._last_progress_change = time.time()
            self._eta_calculated_at = time.time()
            self._start_eta_timer()  # Start periodic ETA updates
            # Tab badge: start spinner
            self._spin_frame = 0
            self._spin_tab_badge()
        self._update_progress_label()

    def _stop_progress(self):
        _was_busy = self.busy_count > 0
        if self.busy_count > 0:
            self.busy_count -= 1
        if self.busy_count == 0:
            self.start_button.configure(text="Spustiť znovu", style="Green.TButton", state="normal")
            self.stop_button.configure(text="Stop", style="Red.TButton", state="disabled")
            self._stop_progress_animation()
            self._target_progress = 0
            self.progress_var.set(0)
            self._set_awake(False)
            self.progress_start_ts = None
            self.eta_history = []
            self.eta_smoothed = None
            self._stop_eta_timer()  # Stop periodic ETA updates
        self._update_progress_label()

    def _start_eta_timer(self):
        """Start timer for periodic ETA updates every second."""
        if hasattr(self, '_eta_timer_id') and self._eta_timer_id:
            return
        def tick():
            if self.busy_count > 0:
                self._update_progress_label()
                self._eta_timer_id = self.after(1000, tick)
            else:
                self._eta_timer_id = None
        self._eta_timer_id = self.after(1000, tick)

    def _stop_eta_timer(self):
        """Stop the ETA update timer."""
        if hasattr(self, '_eta_timer_id') and self._eta_timer_id:
            self.after_cancel(self._eta_timer_id)
            self._eta_timer_id = None

    def _start_progress_animation(self):
        """Animate progress bar smoothly toward target value."""
        if self._progress_anim_id:
            return  # Already animating

        def animate():
            current = self.progress_var.get()
            target = self._target_progress

            if current < target:
                # Increment by 1% but don't exceed target
                new_val = min(current + 1, target)
                self.progress_var.set(new_val)
                self._update_progress_label()

                if new_val < target:
                    # Calculate delay - faster when far from target
                    diff = target - new_val
                    delay = max(50, min(500, int(1000 / max(diff, 1))))
                    self._progress_anim_id = self.after(delay, animate)
                else:
                    self._progress_anim_id = None
            else:
                self._progress_anim_id = None

        self._progress_anim_id = self.after(100, animate)

    def _stop_progress_animation(self):
        """Stop progress animation."""
        if self._progress_anim_id:
            self.after_cancel(self._progress_anim_id)
            self._progress_anim_id = None

    def _drain_log(self) -> None:
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            if isinstance(msg, tuple):
                tag = msg[0]
                if tag == "start":
                    self._start_progress()
                    continue
                if tag == "stop":
                    self._stop_progress()
                    continue
                if tag == "progress":
                    try:
                        val = float(msg[1])
                        val = max(0, min(100, val))
                        self._target_progress = val
                        self._start_progress_animation()
                    except Exception:
                        pass
                    continue
                if tag == "step_init":
                    self._ui_init_steps(msg[1])
                    continue
                if tag == "step_active":
                    self._ui_set_step(msg[1], "active")
                    continue
                if tag == "step_done":
                    self._ui_set_step(msg[1], "done")
                    continue
                if tag == "success_button":
                    self.stop_button.configure(text="Hotovo", style="Green.TButton")
                    self._clear_tab_badge(success=True)
                    continue
                if tag == "task_list_init":
                    self._ui_init_task_list(msg[1])
                    continue
                if tag == "task_status":
                    self._ui_set_task_status(msg[1][0], msg[1][1])
                    continue
                if tag == "error":
                    self._set_error_state(msg[1])
                    self._clear_tab_badge(success=False)
                    continue
                if tag == "flash_window":
                    try:
                        self.bell()
                        self.focus_force()
                    except Exception:
                        pass
                    continue
            self.log_text.configure(state="normal")
            self.log_text.insert("end", str(msg) + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(150, self._drain_log)

    # -------------------- Defaults sync --------------------
    def _apply_defaults(self):
        self.local_lipsync.set(self.def_local_lipsync.get())
        self.local_keep_music.set(self.def_local_keep_music.get())
        self.local_context.set(self.def_local_context.get())
        self.local_voice_eq.set(self.def_local_voice_eq.get())
        self.local_multi_voice.set(self.def_local_multi_voice.get())
        self.local_clone_voice.set(self.def_local_clone_voice.get())
        self.local_emotion_clone.set(self.def_local_emotion_clone.get())
        self.local_denoise.set(self.def_local_denoise.get())
        self.local_auto_speech_gain.set(self.def_local_auto_speech_gain.get())
        # Apply engine defaults
        if hasattr(self, 'def_tts_var'):
            self.local_tts.set(self.def_tts_var.get())
        if hasattr(self, 'def_trans_var'):
            self.local_engine.set(self.def_trans_var.get())

    # -------------------- Tools helpers --------------------
    def _run_export_onnx(self):
        if not self.settings.py.exists():
            messagebox.showerror("Python nenájdený", self.settings.py)
            return
        
        # Search for script in likely locations
        candidates = [BASE_DIR / "MuseTalk" / "export_onnx.py", BASE_DIR / "export_onnx.py"]
        script_path = None
        work_dir = None
        for p in candidates:
            if p.exists():
                script_path = p
                work_dir = p.parent
                break
        if not script_path:
            messagebox.showerror("Chýba export_onnx.py", f"Súbor sa nenašiel:\n{candidates[0]}")
            return
        args = ["--fp16"] if self.export_fp16.get() else []
        models = self.export_models.get()
        if models == "all":
            args.append("--export_all")
        elif models == "unet":
            args.append("--export_unet")
        elif models == "vae":
            args.append("--export_vae")
        elif models == "bisenet":
            args.append("--export_bisenet")
        elif models == "whisper":
            args.append("--export_whisper")
        if self.export_type.get() == "tensorrt":
            args.append("--tensorrt")

        cmd = [_path_str(self.settings.py), target_script.name, *args]

        def worker():
            self._queue_progress(0)
            self._queue_log(f"[INFO] Spúšťam export modelov...")
            
            # Capture output to detect specific errors
            output_log = []
            def on_line(line):
                output_log.append(line)
                return False

            ret = self.runner.run(cmd, work_dir, on_line=on_line)
            if ret == 0:
                self._queue_log("[OK] Export dokončený (MuseTalk\\models\\onnx)")
                self._queue_progress(100)
                self._notify_success("Export modelov dokončený.")
            else:
                full_log = "".join(output_log)
                if "torchvision" in full_log and "nms" in full_log:
                    self._notify_error("Chyba prostredia! Spustite 'Oprava prostredia' v Nástrojoch.")
                    self._queue_log("[TIP] Táto chyba sa opraví tlačidlom 'Spustiť opravu' v záložke Nástroje.")
                else:
                    self._notify_error("Export modelov zlyhal. Skontrolujte log.")

        threading.Thread(target=worker, daemon=True).start()

    def _pick_fps_src(self):
        path = filedialog.askopenfilename(title="Zdrojové video")
        if path:
            self.fps25_src.set(path)
            if not self.fps25_out.get():
                out = Path(path).with_name(Path(path).stem + "_25fps.mp4")
                self.fps25_out.set(str(out))

    def _pick_fps_out(self):
        path = filedialog.asksaveasfilename(
            title="Výstup 25fps",
            defaultextension=".mp4",
            filetypes=[("Video", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self.fps25_out.set(path)

    def _run_fps25(self):
        src = self.fps25_src.get().strip()
        dst = self.fps25_out.get().strip()
        if not src or not Path(src).exists():
            messagebox.showerror("Chýba vstup", "Vyber zdrojové video.")
            return
        if not dst:
            dst = str(Path(src).with_name(Path(src).stem + "_25fps.mp4"))
            self.fps25_out.set(dst)
        ffmpeg = self.settings.ffmpeg if self.settings.ffmpeg else "ffmpeg"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            src,
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            dst,
        ]

        def worker():
            self._queue_progress(0)
            ret = self.runner.run(cmd, BASE_DIR)
            if ret == 0:
                self._queue_log(f"[OK] 25fps video: {dst}")
                self._queue_progress(100)

        threading.Thread(target=worker, daemon=True).start()

    # -------------------- Video Split --------------------
    def _pick_split_src(self):
        path = filedialog.askopenfilename(
            title="Zdrojové video",
            filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov *.flv"), ("All files", "*.*")],
        )
        if path:
            self.split_src.set(path)

    def _run_split_video(self):
        import math
        src = self.split_src.get().strip()
        minutes = self.split_minutes.get()
        if not src or not Path(src).exists():
            messagebox.showerror("Chýba vstup", "Vyber zdrojové video.")
            return
        if minutes <= 0:
            messagebox.showerror("Chyba", "Počet minút musí byť väčší ako 0.")
            return

        # Clear log before each run
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        ffmpeg = self.settings.ffmpeg if self.settings.ffmpeg else "ffmpeg"
        ffprobe = shutil.which("ffprobe") or "ffprobe"
        self.split_btn.configure(state="disabled", text="Analyzujem ticho…")

        def worker():
            try:
                self._queue_progress(0)
                flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

                # Get video duration
                cmd_probe = [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    src,
                ]
                result = subprocess.run(cmd_probe, capture_output=True, text=True, creationflags=flags)
                total_duration = float(result.stdout.strip())

                segment_seconds = minutes * 60
                vp = Path(src)
                output_dir = vp.parent / f"{vp.stem}_split"
                output_dir.mkdir(exist_ok=True)

                self._queue_log(f"Video: {vp.name}")
                self._queue_log(f"Dĺžka: {total_duration / 60:.1f} min ({total_duration:.0f}s)")
                self._queue_log(f"Cieľová dĺžka časti: {minutes} min")
                self._queue_log("-" * 50)

                # --- Step 1: Detect silent moments ---
                self._queue_log("Hľadám tiché miesta vo videu…")
                self.after(0, lambda: self.split_btn.configure(text="Hľadám ticho…"))
                # silence_threshold: -35dB, min silence duration: 0.3s
                cmd_silence = [
                    ffmpeg, "-i", src,
                    "-af", "silencedetect=noise=-35dB:d=0.3",
                    "-f", "null", "-",
                ]
                proc_sil = subprocess.run(cmd_silence, capture_output=True, text=True, creationflags=flags)
                # Parse silence_end timestamps from stderr
                import re as _re
                silence_points = []
                for line in proc_sil.stderr.splitlines():
                    m = _re.search(r'silence_end:\s*([\d.]+)', line)
                    if m:
                        silence_points.append(float(m.group(1)))

                self._queue_log(f"Nájdených {len(silence_points)} tichých miest")

                # --- Step 2: Find best split points near target times ---
                # For each target split point, find nearest silence within ±15s window
                search_window = 15.0  # seconds to search around target
                split_times = [0.0]  # always start at 0

                num_target_splits = math.ceil(total_duration / segment_seconds) - 1
                for i in range(1, num_target_splits + 1):
                    target = i * segment_seconds
                    if target >= total_duration:
                        break
                    # Find nearest silence to target within window
                    best = None
                    best_dist = float('inf')
                    for sp in silence_points:
                        dist = abs(sp - target)
                        if dist < best_dist and dist <= search_window:
                            best = sp
                            best_dist = dist
                    if best is not None:
                        split_times.append(best)
                        self._queue_log(f"  Bod {i}: {best:.1f}s (ticho, cieľ bol {target:.0f}s, odchýlka {best_dist:.1f}s)")
                    else:
                        # No silence found nearby, use exact target
                        split_times.append(target)
                        self._queue_log(f"  Bod {i}: {target:.1f}s (presný, žiadne ticho v okolí)")

                split_times.append(total_duration)
                num_segments = len(split_times) - 1

                self._queue_log(f"Rozdelenie na {num_segments} častí")
                self._queue_log("-" * 50)
                self.after(0, lambda: self.split_btn.configure(text=f"Delím 0/{num_segments}…"))

                # --- Step 3: Split at silence points ---
                ok_count = 0
                for i in range(num_segments):
                    start_time = split_times[i]
                    duration = split_times[i + 1] - start_time
                    out_file = output_dir / f"{vp.stem}_part{i + 1:03d}{vp.suffix}"

                    self.after(0, lambda idx=i: self.split_btn.configure(text=f"Delím {idx+1}/{num_segments}…"))

                    cmd_split = [
                        ffmpeg, "-y",
                        "-ss", f"{start_time:.3f}",
                        "-i", src,
                        "-t", f"{duration:.3f}",
                        "-c", "copy",
                        "-avoid_negative_ts", "make_start_zero",
                        str(out_file),
                    ]
                    proc = subprocess.run(cmd_split, capture_output=True, text=True, creationflags=flags)
                    pct = (i + 1) / num_segments * 100
                    self._queue_progress(pct)

                    if proc.returncode == 0 and out_file.exists():
                        size_mb = out_file.stat().st_size / (1024 * 1024)
                        dur_min = duration / 60
                        self._queue_log(
                            f"[{i + 1}/{num_segments}] {out_file.name} — OK "
                            f"({dur_min:.1f} min, {size_mb:.1f} MB, "
                            f"{start_time:.1f}s–{split_times[i+1]:.1f}s)"
                        )
                        ok_count += 1
                    else:
                        err = proc.stderr.strip().splitlines()[-1] if proc.stderr else "neznáma chyba"
                        self._queue_log(f"[{i + 1}/{num_segments}] {out_file.name} — CHYBA: {err}")

                self._queue_log("-" * 50)
                self._queue_log(f"Hotovo! {ok_count}/{num_segments} častí v: {output_dir}")
                self._queue_progress(100)
            except Exception as e:
                self._queue_log(f"[ERROR] {e}")
            finally:
                self.after(0, lambda: self.split_btn.configure(state="normal", text="Rozdeliť"))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------- Video Join --------------------
    def _pick_join_dir(self):
        path = filedialog.askdirectory(title="Adresár s časťami videa")
        if path:
            self.join_dir.set(path)
            # Auto-detect how many video files are inside
            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}
            parts = sorted(
                [f for f in Path(path).iterdir() if f.suffix.lower() in video_exts],
                key=lambda p: p.name,
            )
            self._queue_log(f"Nájdených {len(parts)} videí v: {path}")

    def _run_join_video(self):
        src_dir = self.join_dir.get().strip()
        if not src_dir or not Path(src_dir).is_dir():
            messagebox.showerror("Chýba vstup", "Vyber adresár s časťami videa.")
            return

        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}
        parts = sorted(
            [f for f in Path(src_dir).iterdir() if f.suffix.lower() in video_exts],
            key=lambda p: p.name,
        )
        if len(parts) < 2:
            messagebox.showerror("Chyba", f"V adresári je len {len(parts)} video. Treba aspoň 2.")
            return

        # Ask for output file
        default_name = Path(src_dir).stem.replace("_split", "") + "_joined.mp4"
        out_path = filedialog.asksaveasfilename(
            title="Uložiť spojené video ako",
            initialdir=Path(src_dir).parent,
            initialfile=default_name,
            filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv"), ("All files", "*.*")],
        )
        if not out_path:
            return

        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        ffmpeg = self.settings.ffmpeg if self.settings.ffmpeg else "ffmpeg"
        self.join_btn.configure(state="disabled", text="Spájanie…")

        def worker():
            try:
                self._queue_progress(0)
                # Create concat list file
                concat_file = Path(src_dir) / "_concat_list.txt"
                with open(concat_file, "w", encoding="utf-8") as f:
                    for p in parts:
                        # ffmpeg concat requires forward slashes and escaping
                        escaped = str(p).replace("\\", "/").replace("'", "'\\''")
                        f.write(f"file '{escaped}'\n")

                self._queue_log(f"Spájanie {len(parts)} častí:")
                for i, p in enumerate(parts, 1):
                    self._queue_log(f"  {i}. {p.name}")
                self._queue_log(f"Výstup: {out_path}")
                self._queue_log("-" * 50)

                flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                cmd = [
                    ffmpeg, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-c", "copy",
                    out_path,
                ]
                self._queue_log(f"Príkaz: {' '.join(cmd)}")
                proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)

                # Cleanup concat list
                concat_file.unlink(missing_ok=True)

                if proc.returncode == 0 and Path(out_path).exists():
                    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
                    self._queue_log("-" * 50)
                    self._queue_log(f"Hotovo! Výstup: {Path(out_path).name} ({size_mb:.1f} MB)")
                    self._queue_progress(100)
                else:
                    err = proc.stderr.strip().splitlines()[-1] if proc.stderr else "neznáma chyba"
                    self._queue_log(f"[CHYBA] {err}")
            except Exception as e:
                self._queue_log(f"[ERROR] {e}")
            finally:
                self.after(0, lambda: self.join_btn.configure(state="normal", text="Spojiť"))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------- Progress label --------------------
    def _update_progress_label(self):
        val = float(self.progress_var.get())
        now = time.time()

        if self.progress_start_ts and val >= 1:
            elapsed = now - self.progress_start_ts

            # Record progress changes for speed calculation
            if not self.eta_history or self.eta_history[-1][1] != val:
                self.eta_history.append((now, val))
                if len(self.eta_history) > 30:
                    self.eta_history = self.eta_history[-30:]
                self._last_progress_change = now

                # Recalculate ETA when progress changes
                if len(self.eta_history) >= 2:
                    t_start, p_start = self.eta_history[0]
                    t_end, p_end = self.eta_history[-1]
                    dt = t_end - t_start
                    dp = p_end - p_start
                    if dp > 0 and dt > 0:
                        speed = dp / dt
                        new_eta = (100 - val) / speed
                        if self.eta_smoothed is None:
                            self.eta_smoothed = new_eta
                        else:
                            self.eta_smoothed = 0.3 * new_eta + 0.7 * self.eta_smoothed
                        self._eta_calculated_at = now

            # Calculate remaining time for current video (countdown)
            if self.eta_smoothed is not None and hasattr(self, '_eta_calculated_at'):
                time_passed = now - self._eta_calculated_at
                remaining = max(0, self.eta_smoothed - time_passed)
            else:
                remaining = elapsed * (100 - val) / max(val, 1)

            # Format elapsed
            el_m, el_s = divmod(int(elapsed), 60)
            elapsed_txt = f"{el_m}:{el_s:02d}"

            # Format remaining for current video
            rem_m, rem_s = divmod(int(remaining), 60)
            remain_txt = f"{rem_m}:{rem_s:02d}"

            # Calculate finish time (always show)
            finish_time_txt = ""
            batch_info_txt = ""

            if self.batch_total_videos > 1 and self.batch_start_ts:
                # Batch mode - calculate total remaining time
                remaining_videos = self.batch_total_videos - self.batch_completed_videos - 1

                if self.batch_video_times:
                    avg_time = sum(self.batch_video_times) / len(self.batch_video_times)
                    batch_remaining = remaining + (remaining_videos * avg_time)
                elif val > 5:
                    est_video_time = elapsed / (val / 100)
                    batch_remaining = remaining + (remaining_videos * est_video_time)
                else:
                    batch_remaining = remaining

                # Calculate and format finish time
                finish_time = now + batch_remaining
                finish_str = time.strftime("%H:%M", time.localtime(finish_time))

                # Format batch remaining
                if batch_remaining >= 3600:
                    batch_h, batch_rem = divmod(int(batch_remaining), 3600)
                    batch_m, _ = divmod(batch_rem, 60)
                    batch_remain_txt = f"{batch_h}h{batch_m:02d}m"
                else:
                    batch_m, batch_s = divmod(int(batch_remaining), 60)
                    batch_remain_txt = f"{batch_m}:{batch_s:02d}"

                batch_info_txt = f"  [{self.batch_completed_videos + 1}/{self.batch_total_videos}]  |  Zostáva: {batch_remain_txt}"
                finish_time_txt = f"  |  Hotovo: ~{finish_str}"
            else:
                # Single video - show finish time for current video
                finish_time = now + remaining
                finish_str = time.strftime("%H:%M", time.localtime(finish_time))
                finish_time_txt = f"  |  Hotovo: ~{finish_str}"

            text = f"{val:.1f}%  ⏱ {remain_txt}{batch_info_txt}{finish_time_txt}  |  Uplynulo: {elapsed_txt}"
        else:
            text = f"{val:.0f}%"

        self.progress_label.config(text=text)
        # When the green bar covers text, switch to white for contrast
        if val >= 55:
            self.progress_label.config(fg="white")
        else:
            self.progress_label.config(fg=self._theme.get("fg", "#222"))

    def _update_batch_progress(self) -> None:
        """Show/update the overall batch progress bar. Called from main thread only."""
        total = getattr(self, "batch_total_videos", 0)
        done = getattr(self, "batch_completed_videos", 0)
        if total <= 1:
            # Hide batch bar for single-video jobs
            if self._batch_bar_visible:
                self.batch_progress.pack_forget()
                self._batch_bar_visible = False
            return
        # Show batch bar if not visible
        if not self._batch_bar_visible:
            self.batch_progress.pack(fill="x", pady=(0, 4), ipady=3, after=self.progress)
            self._batch_bar_visible = True
        pct = (done / total) * 100.0
        self.batch_progress_var.set(pct)
        self.batch_progress_label.config(
            text=f"Celkový batch: {done}/{total}  ({pct:.0f}%)",
            fg="white" if pct >= 55 else self._theme.get("fg", "#222"),
        )

    # -------------------- Sleep prevention --------------------
    def _set_awake(self, enable: bool):
        if not IS_WINDOWS:
            return
        if not self.keep_awake.get():
            return
        try:
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            if enable:
                result = ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                )
                if result:
                    self._queue_log("[AWAKE] Sleep prevention aktivovaný")
                # Start periodic refresh timer
                self._start_awake_timer()
            else:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                self._stop_awake_timer()
        except Exception as e:
            self._queue_log(f"[WARN] Sleep prevention zlyhalo: {e}")

    def _start_awake_timer(self):
        """Periodically refresh awake state every 30 seconds."""
        if not IS_WINDOWS:
            return
        if hasattr(self, '_awake_timer_id') and self._awake_timer_id:
            return  # Already running
        def refresh():
            if self.busy_count > 0 and self.keep_awake.get():
                try:
                    ES_CONTINUOUS = 0x80000000
                    ES_SYSTEM_REQUIRED = 0x00000001
                    ES_DISPLAY_REQUIRED = 0x00000002
                    ctypes.windll.kernel32.SetThreadExecutionState(
                        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                    )
                except:
                    pass
                self._awake_timer_id = self.after(30000, refresh)  # Every 30 sec
            else:
                self._awake_timer_id = None
        self._awake_timer_id = self.after(30000, refresh)

    def _stop_awake_timer(self):
        """Stop the awake refresh timer."""
        if hasattr(self, '_awake_timer_id') and self._awake_timer_id:
            self.after_cancel(self._awake_timer_id)
            self._awake_timer_id = None

    def _on_closing(self):
        """Handle window close event."""
        if self.busy_count > 0:
            if not messagebox.askokcancel("Ukončiť", "Proces stále beží. Naozaj chcete skončiť?"):
                return
            try:
                self.runner.stop()
            except Exception:
                pass
        try:
            self.settings.def_trans_engine = self.local_engine.get()
            self.settings.def_batch_qa = self.local_use_madlad_gemma_qa.get()
            self.settings.save()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
        # Force exit - background threads can keep the process alive
        os._exit(0)

    def _stop_all_processes(self):
        """Stops the running process when the Stop button is clicked."""
        self.runner.stop()

    def _set_error_state(self, message: str):
        """Changes Start button to Red and shows error message."""
        self.start_button.configure(text=f"CHYBA: {message}", style="Red.TButton", state="normal")

    # -------------------- Notifications --------------------
    def _notify_success(self, message: str = "Spracovanie dokončené!"):
        """Show success notification with sound."""
        if HAS_WINSOUND:
            try:
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
            except:
                pass
        else:
            # Linux / macOS: desktop notification + audio chime
            try:
                subprocess.Popen(
                    ["notify-send", "-i", "emblem-default", "VideoTranslator", message],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                pass
            try:
                subprocess.Popen(
                    ["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                pass
        self._queue_log(f"[✓] {message}")
        self.log_queue.put(("success_button", None))
        self.log_queue.put(("flash_window", None))
        # Shutdown PC if requested
        if self.shutdown_after.get():
            self._queue_log("[INFO] Vypínam PC o 60 sekúnd... (shutdown /a pre zrušenie)")
            try:
                subprocess.Popen(["shutdown", "/s", "/t", "60"], creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception as e:
                self._queue_log(f"[ERROR] Nepodarilo sa vypnúť PC: {e}")

    def _notify_error(self, message: str = "Niečo sa pokazilo!"):
        """Show error notification with sound."""
        if HAS_WINSOUND:
            try:
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
            except:
                pass
        else:
            try:
                subprocess.Popen(
                    ["notify-send", "-i", "dialog-error", "VideoTranslator — Chyba", message],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                pass
            try:
                subprocess.Popen(
                    ["paplay", "/usr/share/sounds/freedesktop/stereo/dialog-error.oga"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                pass
        self._queue_log(f"[✗] CHYBA: {message}")
        self.log_queue.put(("error", message))
        self.log_queue.put(("flash_window", None))

    # -------------------- Tab: Stiahnutie videa --------------------
    def youtube_tab(self, parent: ttk.Frame) -> None:
        tab = ttk.Frame(parent)
        tab.grid(row=0, column=0, sticky="nsew")
        self._pages["youtube"] = tab

        self.yt_url = tk.StringVar()
        self.yt_quality = tk.StringVar(value="1080")
        self.yt_format = tk.StringVar(value="video")
        self.yt_output_dir = tk.StringVar(value=str(Path.home() / "Stiahnuté"))
        self.yt_playlist = tk.BooleanVar(value=False)
        self.yt_cookies_browser = tk.StringVar(value="")

        # hint
        hint = ttk.Label(tab, text="Podporované: YouTube, Vimeo, TikTok, Twitter/X, Facebook, Twitch, Dailymotion, PornHub a 1000+ ďalších (yt-dlp)", foreground="#7a9cbf")
        hint.grid(row=0, column=0, columnspan=3, sticky="w", pady=(6, 2))

        ttk.Label(tab, text="URL").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(tab, textvariable=self.yt_url, width=70).grid(row=1, column=1, columnspan=2, padx=5, pady=8, sticky="ew")

        ttk.Label(tab, text="Formát").grid(row=2, column=0, sticky="w", pady=5)
        fmt_frame = ttk.Frame(tab)
        fmt_frame.grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(fmt_frame, text="Video (MP4)", value="video", variable=self.yt_format).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(fmt_frame, text="Iba audio (MP3)", value="audio", variable=self.yt_format).pack(side="left")

        ttk.Label(tab, text="Max kvalita").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(tab, textvariable=self.yt_quality, values=["2160", "1440", "1080", "720", "480", "360"], width=8, state="readonly").grid(
            row=3, column=1, sticky="w", pady=5
        )

        ttk.Label(tab, text="Výstupný priečinok").grid(row=4, column=0, sticky="w", pady=5)
        out_frame = ttk.Frame(tab)
        out_frame.grid(row=4, column=1, columnspan=2, sticky="ew", pady=5)
        ttk.Entry(out_frame, textvariable=self.yt_output_dir, width=55).pack(side="left", padx=(0, 6))
        ttk.Button(out_frame, text="Prehľadávať…", command=lambda: (
            d := filedialog.askdirectory(initialdir=self.yt_output_dir.get()),
            self.yt_output_dir.set(d) if d else None,
        )).pack(side="left")

        opt_frame = ttk.Frame(tab)
        opt_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ttk.Checkbutton(opt_frame, text="Stiahnuť celý playlist", variable=self.yt_playlist).pack(side="left", padx=(0, 24))
        ttk.Label(opt_frame, text="Cookies z prehliadača:").pack(side="left", padx=(0, 6))
        ttk.Combobox(opt_frame, textvariable=self.yt_cookies_browser,
                     values=["", "firefox", "chrome", "chromium", "brave", "edge"],
                     width=10, state="readonly").pack(side="left")
        ttk.Label(opt_frame, text="(pre prihlásené stránky)", foreground="#7a9cbf").pack(side="left", padx=(6, 0))

    def _run_youtube(self) -> None:
        url = self.yt_url.get().strip()
        if not url:
            messagebox.showerror("Chýba URL", "Vložte URL adresu videa.")
            return
        yt_dlp_bin = shutil.which("yt-dlp")
        if not yt_dlp_bin:
            messagebox.showerror("yt-dlp chýba", "Nainštalujte yt-dlp (pip install yt-dlp).")
            return

        yt_quality      = self.yt_quality.get() or "1080"
        yt_fmt          = self.yt_format.get()
        output_dir      = self.yt_output_dir.get().strip() or str(Path.home() / "Stiahnuté")
        yt_playlist     = self.yt_playlist.get()
        yt_cookies_br   = self.yt_cookies_browser.get().strip()
        shutdown_after  = self.shutdown_after.get()

        self.pipeline_state = {"phase": "init", "base": 0.0, "range": 100.0}

        def worker():
            try:
                self._queue_progress(0)
                steps = ["Sťahovanie"]
                self.current_steps = steps
                self._queue_step_init(steps)
                self._queue_step_active(0)

                Path(output_dir).mkdir(parents=True, exist_ok=True)
                # For playlists use subfolder per playlist
                if yt_playlist:
                    out_tmpl = str(Path(output_dir) / "%(playlist_title)s" / "%(playlist_index)s - %(title)s.%(ext)s")
                else:
                    out_tmpl = str(Path(output_dir) / "%(title)s.%(ext)s")

                playlist_args = [] if yt_playlist else ["--no-playlist"]
                cookies_args  = ["--cookies-from-browser", yt_cookies_br] if yt_cookies_br else []

                if yt_fmt == "audio":
                    cmd_dl = [
                        yt_dlp_bin,
                        "-x", "--audio-format", "mp3",
                        "-o", out_tmpl,
                        *playlist_args, *cookies_args,
                        url,
                    ]
                else:
                    cmd_dl = [
                        yt_dlp_bin,
                        "-f", f"bestvideo[height<={yt_quality}]+bestaudio/best[height<={yt_quality}]",
                        "--merge-output-format", "mp4",
                        "-o", out_tmpl,
                        *playlist_args, *cookies_args,
                        url,
                    ]

                if self.runner.run(cmd_dl, Path(output_dir), base_progress=0, progress_range=100) != 0:
                    self._notify_error("Sťahovanie zlyhalo!")
                    return

                self._queue_step_done(0)
                self._queue_progress(100)
                self._notify_success(f"Stiahnuté do: {output_dir}", shutdown_after=shutdown_after)

            except Exception as _exc:
                import traceback as _tb
                self._queue_log(f"[CRASH] Neočakávaná chyba: {_exc}\n{_tb.format_exc()}")
                self._notify_error(f"Neočakávaná chyba: {_exc}")

        threading.Thread(target=worker, daemon=True).start()

    # -------------------- Tab: Local Whisper/TTS --------------------
    def local_tab(self, parent: ttk.Frame) -> None:
        tab = ttk.Frame(parent)
        tab.grid(row=0, column=0, sticky="nsew")
        self._pages["local"] = tab

        self.local_src = tk.StringVar()
        self.local_lang = tk.StringVar(value="cs")
        self.local_engine = tk.StringVar(value=self.settings.def_trans_engine)
        self.local_madlad_model = tk.StringVar(value="google/madlad400-7b-mt")
        self.local_llm_model = tk.StringVar(value="models/gemma-3-27b-it/google_gemma-3-27b-it-Q6_K.gguf")
        default_local_tts = self.settings.def_tts_engine if self.settings.def_tts_engine in ("chatterbox", "xtts", "edge") else "chatterbox"
        self.local_tts = tk.StringVar(value=default_local_tts)
        self.local_lipsync = tk.BooleanVar(value=self.def_local_lipsync.get())
        self.local_keep_music = tk.BooleanVar(value=self.def_local_keep_music.get())
        self.local_context = tk.BooleanVar(value=self.def_local_context.get())
        self.local_voice_eq = tk.BooleanVar(value=self.def_local_voice_eq.get())
        self.local_multi_voice = tk.BooleanVar(value=self.def_local_multi_voice.get())
        self.local_multi_voice_n = tk.IntVar(value=2)
        self.local_clone_voice = tk.BooleanVar(value=self.def_local_clone_voice.get())
        self.local_emotion_clone = tk.BooleanVar(value=self.def_local_emotion_clone.get())
        self.local_denoise = tk.BooleanVar(value=self.def_local_denoise.get())
        self.local_auto_speech_gain = tk.BooleanVar(value=self.def_local_auto_speech_gain.get())
        self.local_denoise_preset = tk.StringVar(value=self.settings.def_denoise_preset)
        self.local_base_tempo = tk.StringVar(value=self.settings.def_base_tempo)
        self.local_max_chars = tk.StringVar(value="300")
        self.local_out = tk.StringVar(value="")
        self.local_tts_eq_profile = tk.StringVar(value="")
        self.local_pitch_shift = tk.StringVar(value="0")
        self.local_use_madlad_gemma_qa = tk.BooleanVar(value=self.settings.def_batch_qa)
        self.local_tts_safe_mode = tk.BooleanVar(value=False)
        self.local_use_text_adaptation = tk.BooleanVar(value=True)
        self.local_preset = tk.StringVar(value="production_dub")

        # Batch processing state
        self.batch_items: List[BatchItem] = []
        self.batch_current_index = 0

        # --- Batch file list with scrollbar ---
        ttk.Label(tab, text="Vstupné videá (batch)").grid(row=0, column=0, sticky="nw", pady=5)

        list_frame = ttk.Frame(tab)
        list_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.batch_listbox = tk.Listbox(list_frame, height=5, width=65, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.batch_listbox.yview)
        self.batch_listbox.configure(yscrollcommand=scrollbar.set)
        self.batch_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=0, column=2, padx=5, pady=5, sticky="n")
        ttk.Button(btn_frame, text="Pridať…", command=self._pick_local_batch).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Odstrániť", command=self._remove_batch_selected).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Vyčistiť", command=self._clear_batch_list).pack(fill="x", pady=2)
        ttk.Separator(btn_frame, orient="horizontal").pack(fill="x", pady=4)
        ttk.Button(btn_frame, text="Uložiť zoznam…", command=self._save_batch_list).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Zoznamy…", command=self._open_list_manager).pack(fill="x", pady=2)
        ttk.Separator(btn_frame, orient="horizontal").pack(fill="x", pady=4)
        ttk.Button(btn_frame, text="Uložiť checkpoint…", command=self._save_checkpoint_named).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Checkpointy…", command=self._open_checkpoint_manager).pack(fill="x", pady=2)

        # Checkpoint resume button
        self.resume_btn = ttk.Button(btn_frame, text="Pokračovať ❯", command=self._resume_from_checkpoint)
        self.resume_btn.pack(fill="x", pady=(10, 2))
        self._update_resume_button()

        ttk.Label(tab, text="Výstupný priečinok").grid(row=1, column=0, sticky="w", pady=(8, 2))
        ttk.Entry(tab, textvariable=self.local_out, width=70).grid(row=1, column=1, padx=5, pady=2, sticky="we")
        ttk.Button(tab, text="Vybrať…", command=self._pick_local_out_dir).grid(row=1, column=2, padx=5, pady=2, sticky="e")
        ttk.Label(tab, text="(ponechaj prázdne = uloží sa vedľa zdroja)", foreground="gray").grid(
            row=2, column=1, columnspan=2, sticky="w", padx=5, pady=(0, 6)
        )

        ttk.Label(tab, text="Cieľový jazyk").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=self.local_lang, width=10).grid(row=3, column=1, sticky="w", pady=5)

        tts_frame = ttk.LabelFrame(tab, text="TTS engine")
        tts_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5, padx=2)

        # Row 0: Radio buttons
        _tts_radio_row = ttk.Frame(tts_frame)
        _tts_radio_row.pack(fill="x", padx=4, pady=(4, 2))
        _tts_tips = {
            "chatterbox": "Chatterbox — lokálny neurálny TTS, najlepší pre klónovanie hlasu. Vyžaduje GPU.",
            "xtts": "XTTS v2 — lokálny TTS od Coqui, podpora viac jazykov, klónovanie hlasu cez WAV súbor.",
            "edge": "Edge-TTS — Microsoft cloud TTS, zadarmo, bez GPU. Prirodzený hlas. Vyžaduje internet.",
        }
        for txt, val in (("Chatterbox", "chatterbox"), ("XTTS v2", "xtts"), ("Edge-TTS", "edge")):
            _rb = ttk.Radiobutton(_tts_radio_row, text=txt, value=val, variable=self.local_tts)
            _rb.pack(side="left", padx=8)
            tip(_rb, _tts_tips[val])
        self.xtts_speaker_wav = tk.StringVar(value="")

        # Row 1a: Edge-TTS voice filter (lang + gender) + voice combobox + rate + pitch
        _all_voice_names = [n for n, _, _ in self._ALL_EDGE_VOICES]
        self.edge_tts_voice_var = tk.StringVar(value="cs-CZ-AntoninNeural")
        self.edge_tts_rate_var = tk.StringVar(value="+0%")
        self.edge_tts_pitch_var = tk.StringVar(value="+0Hz")
        self.edge_lang_filter_var = tk.StringVar(value="cs")
        self.edge_gender_filter_var = tk.StringVar(value="všetky")

        # Filter row
        edge_filter_row = ttk.Frame(tts_frame)
        edge_filter_row.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(edge_filter_row, text="Jazyk:").pack(side="left", padx=(0, 2))
        _lang_cb = ttk.Combobox(edge_filter_row, textvariable=self.edge_lang_filter_var,
                     values=["všetky", "cs", "sk", "en", "de", "fr", "es", "pl", "ru"],
                     width=7, state="readonly")
        _lang_cb.pack(side="left", padx=(0, 8))
        tip(_lang_cb, "Filtruj hlasy podľa jazyka.")
        ttk.Label(edge_filter_row, text="Pohlavie:").pack(side="left", padx=(0, 2))
        _gender_cb = ttk.Combobox(edge_filter_row, textvariable=self.edge_gender_filter_var,
                      values=["všetky", "muž", "žena"], width=7, state="readonly")
        _gender_cb.pack(side="left")
        tip(_gender_cb, "Filtruj hlasy podľa pohlavia.")
        self.edge_lang_filter_var.trace_add("write", lambda *_: self._filter_edge_voices())
        self.edge_gender_filter_var.trace_add("write", lambda *_: self._filter_edge_voices())

        # Voice + rate + pitch row
        edge_row = ttk.Frame(tts_frame)
        edge_row.pack(fill="x", padx=8, pady=2)
        ttk.Label(edge_row, text="Edge hlas:").pack(side="left", padx=(0, 4))
        self._edge_voice_cb = ttk.Combobox(edge_row, textvariable=self.edge_tts_voice_var,
                     values=_all_voice_names, width=22)
        self._edge_voice_cb.pack(side="left", padx=2)
        tip(self._edge_voice_cb, "Hlas pre Edge-TTS. Editable — môžeš zadať ľubovoľný hlas.\ncs-CZ = čeština, sk-SK = slovenčina, en-US = angličtina.\nAntonin/Lukas/Guy = mužský, Vlasta/Viktoria/Aria/Jenny = ženský.\nEn-US-AriaNeural/GuyNeural podporujú štýly (emócie).")
        ttk.Label(edge_row, text="Rate:").pack(side="left", padx=(8, 2))
        _edge_rate_entry = ttk.Entry(edge_row, textvariable=self.edge_tts_rate_var, width=6)
        _edge_rate_entry.pack(side="left", padx=2)
        tip(_edge_rate_entry, "Rýchlosť: +0% = normálna, +15% = rýchlejšia, -10% = pomalšia.")
        ttk.Label(edge_row, text="Pitch:").pack(side="left", padx=(8, 2))
        _edge_pitch_entry = ttk.Entry(edge_row, textvariable=self.edge_tts_pitch_var, width=6)
        _edge_pitch_entry.pack(side="left", padx=2)
        tip(_edge_pitch_entry, "Výška hlasu: +0Hz = normálna, +5Hz = vyšší, -5Hz = nižší.")
        # Apply initial filter (cs voices only)
        self.after_idle(self._filter_edge_voices)

        # Row 1b: Edge-TTS style (emócia) + Voice 2 pre multi-voice
        _edge_styles = [
            "", "cheerful", "sad", "angry", "excited", "friendly",
            "hopeful", "shouting", "whispering", "terrified", "unfriendly",
            "newscast", "customerservice", "narration-professional",
        ]
        self.edge_tts_style_var = tk.StringVar(value="")
        self.edge_tts_voice2_var = tk.StringVar(value="")
        edge_row2 = ttk.Frame(tts_frame)
        edge_row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(edge_row2, text="Štýl:").pack(side="left", padx=(0, 4))
        _style_cb = ttk.Combobox(edge_row2, textvariable=self.edge_tts_style_var,
                      values=_edge_styles, width=22)
        _style_cb.pack(side="left", padx=2)
        tip(_style_cb, "Emócia/štýl hlasu (SSML). Funguje IBA pre vybrané hlasy (napr. en-US-AriaNeural, en-US-GuyNeural).\nPrečitinka a de-CZ hlasy štýly nepodporujú — parameter sa ignoruje.")
        ttk.Label(edge_row2, text="Hlas 2:").pack(side="left", padx=(12, 2))
        self._edge_voice_cb2 = ttk.Combobox(edge_row2, textvariable=self.edge_tts_voice2_var,
                      values=[""] + _all_voice_names, width=22)
        self._edge_voice_cb2.pack(side="left", padx=2)
        tip(self._edge_voice_cb2, "Druhý hlas pre multi-voice režim (rôzni rečníci).\nVyžaduje zaškrtnuté Multi-hlas. Nechaj prázdne pre single-voice.")

        # Edge-TTS Row 3: backaudio volume
        edge_row3 = ttk.Frame(tts_frame)
        edge_row3.pack(fill="x", padx=8, pady=2)
        self.backaudio_volume_var = tk.StringVar(value="0.0")
        ttk.Label(edge_row3, text="Pozadie orig.:").pack(side="left", padx=(0, 4))
        _ba_spin = ttk.Spinbox(edge_row3, from_=0.0, to=1.0, increment=0.05,
                               textvariable=self.backaudio_volume_var, width=6, format="%.2f")
        _ba_spin.pack(side="left", padx=2)
        tip(_ba_spin, "Objem originálneho zvuku mixovaný ako pozadie (0.0=vypnuté, 0.2=jemné, 0.5=výrazné).\nPodporuje ambienciu/hudbu počas ticha — štýl pyvideotrans.")
        ttk.Label(edge_row3, text="(0=vyp, 0.2=jemné)", foreground="#888888").pack(side="left", padx=4)

        # Row 2: Chatterbox voice picker
        cb_voice_row = ttk.Frame(tts_frame)
        cb_voice_row.pack(fill="x", padx=8, pady=2)
        ttk.Label(cb_voice_row, text="CB hlas:").pack(side="left", padx=(0, 4))
        self.chatterbox_voice_var = tk.StringVar(value="(predvolený)")
        self._chatterbox_voice_cb = ttk.Combobox(
            cb_voice_row, textvariable=self.chatterbox_voice_var,
            values=self._get_chatterbox_voices(), state="readonly", width=22,
        )
        self._chatterbox_voice_cb.pack(side="left", padx=2)
        self._chatterbox_voice_cb.bind("<<ComboboxSelected>>", self._on_chatterbox_voice_selected)
        ttk.Button(cb_voice_row, text="↺", width=3, command=self._refresh_chatterbox_voices).pack(side="left", padx=2)
        ttk.Button(cb_voice_row, text="Otvoriť voices/", command=lambda: os.startfile(str(VOICES_DIR)) if os.name == "nt" else subprocess.Popen(["xdg-open", str(VOICES_DIR)])).pack(side="left", padx=(4, 2))
        ttk.Label(cb_voice_row, text="Model:").pack(side="left", padx=(12, 4))
        self.chatterbox_model_var = tk.StringVar(
            value=self._normalize_chatterbox_model_name(self.settings.def_chatterbox_model)
        )
        self._chatterbox_model_cb = ttk.Combobox(
            cb_voice_row,
            textvariable=self.chatterbox_model_var,
            values=self._get_chatterbox_models(),
            state="readonly",
            width=24,
        )
        self._chatterbox_model_cb.pack(side="left", padx=2)
        ttk.Button(cb_voice_row, text="↺", width=3, command=self._refresh_chatterbox_models).pack(side="left", padx=2)

        # Row 3: XTTS speaker + pitch shift
        _xtts_speakers = [
            "Damien Black", "Craig Gutsy", "Torcull Diarmuid", "Viktor Menelaos",
            "Luis Moray", "Marcos Rudaski", "Wulf Carlevaro", "Aaron Dreschner",
            "Kumar Dahl", "Eugenio Mataracı", "Ferran Simen", "Xavier Hayasaka",
            "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka",
            "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Gilberto Mathias",
            "Ilkin Urbano", "Ludvig Milivoj", "Suad Qasim", "Zacharie Aimilios",
            "Filip Traverse", "Damjan Chapman",
            # female (for completeness)
            "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema",
            "Ana Florence", "Brenda Stern", "Sofia Hellen", "Nova Hogarth",
        ]
        self.xtts_speaker = tk.StringVar(value="Damien Black")
        self.xtts_pitch_shift = tk.StringVar(value="0")
        tts_row2 = ttk.Frame(tts_frame)
        tts_row2.pack(fill="x", padx=8, pady=(2, 4))
        ttk.Label(tts_row2, text="XTTS hlas:").pack(side="left", padx=(0, 4))
        ttk.Combobox(tts_row2, textvariable=self.xtts_speaker, values=_xtts_speakers,
                     state="readonly", width=20).pack(side="left", padx=2)
        ttk.Label(tts_row2, text="Pitch (st):").pack(side="left", padx=(12, 2))
        ttk.Spinbox(tts_row2, textvariable=self.xtts_pitch_shift,
                    from_=-12, to=12, increment=0.5, width=6).pack(side="left", padx=2)

        eng_frame = ttk.LabelFrame(tab, text="Prekladový engine")
        eng_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=5, padx=2)
        _eng_tips = {
            "llama": "Llama/Gemma — lokálny LLM preklad (najlepšia kvalita pre náučné texty). Vyžaduje GPU + veľký model.",
            "madlad": "MADLAD-400 — lokálny seq2seq prekladač od Google. Rýchly, bez internetu.",
            "multislav5lang": "MultiSlav-5lang — veľmi rýchly lokálny Marian prekladač. Podporuje cs/en/pl/sk/sl, dobre drží technické skratky.",
            "hybrid": "Hybrid — MADLAD preklad + Gemma QA kontrola kvality. Kompromis medzi rýchlosťou a kvalitou.",
            "chatgpt": "ChatGPT (OpenAI API) — vynikajúca kvalita prekladu, platené, vyžaduje API kľúč.",
            "grok": "Grok (xAI API) — alternatíva k ChatGPT, vyžaduje API kľúč.",
            "gemini": "Gemini (Google API) — alternatíva k ChatGPT, vyžaduje API kľúč.",
            "google": "Google Translate (zadarmo, bez API kľúča) — mobilný endpoint. Dobrá kvalita pre bežné texty.",
            "google_madlad": "Google+MADLAD — Google Translate ako primárny (lepšia kvalita), MADLAD ako záloha pri zlyhaní alebo SQL/kód segmentoch.",
        }
        for txt, val in (("Llama", "llama"), ("MADLAD", "madlad"), ("MultiSlav", "multislav5lang"), ("Hybrid", "hybrid"), ("Google+M", "google_madlad"), ("ChatGPT", "chatgpt"), ("Grok", "grok"), ("Gemini", "gemini"), ("Google", "google")):
            _rb = ttk.Radiobutton(eng_frame, text=txt, value=val, variable=self.local_engine)
            _rb.pack(side="left", padx=6, pady=4)
            tip(_rb, _eng_tips[val])
        ttk.Label(eng_frame, text="  MADLAD model:").pack(side="left", padx=(12, 2))
        ttk.Combobox(eng_frame, textvariable=self.local_madlad_model, values=[
            "google/madlad400-3b-mt",
            "google/madlad400-7b-mt",
            "google/madlad400-10b-mt",
        ], width=28).pack(side="left", padx=4, pady=4)
        ttk.Label(eng_frame, text="  LLM model:").pack(side="left", padx=(8, 2))
        ttk.Combobox(eng_frame, textvariable=self.local_llm_model, values=[
            "models/gemma-3-27b-it/google_gemma-3-27b-it-Q6_K.gguf",
            "models/gemma-3-12b-it/google_gemma-3-12b-it-Q8_0.gguf",
            "scripts/models/eurollm/EuroLLM-9B-Instruct-Q6_K_L.gguf",
            "models/translategemma/translategemma-12b-it-Q8_0.gguf",
        ], width=48).pack(side="left", padx=4, pady=4)
        local_batch_qa_cb = ttk.Checkbutton(eng_frame, text="Batch QA (MADLAD→Gemma→QA)",
                        variable=self.local_use_madlad_gemma_qa)
        local_batch_qa_cb.pack(side="left", padx=(10, 4), pady=4)

        def _update_local_batch_qa_state(*_):
            state = "normal" if self.local_engine.get() in ("hybrid", "google_madlad") else "disabled"
            local_batch_qa_cb.configure(state=state)

        self.local_engine.trace_add("write", _update_local_batch_qa_state)
        _update_local_batch_qa_state()

        # Content type for translation style (only active for Llama)
        content_frame = ttk.LabelFrame(tab, text="Typ obsahu (štýl prekladu) — Llama / Hybrid")
        content_frame.grid(row=6, column=0, columnspan=3, sticky="w", pady=5, padx=2)
        self.local_content_type = tk.StringVar(value=self.settings.def_content_type)
        content_options = [
            ("Všeobecný", "general"),
            ("Náučné/Prednáška", "educational"),
            ("Podcast/Rozhovor", "podcast"),
            ("Recenzia", "review"),
            ("Technický/Vedecký", "technical"),
            ("SQL/Databázy", "sql"),
            ("Správy", "news"),
            ("Zábava", "entertainment"),
        ]
        self._content_radios = []
        for txt, val in content_options:
            rb = ttk.Radiobutton(content_frame, text=txt, value=val, variable=self.local_content_type)
            rb.pack(side="left", padx=4, pady=4)
            self._content_radios.append(rb)

        def _update_content_state(*_):
            engine = self.local_engine.get()
            state = "normal" if engine in ("llama", "hybrid", "chatgpt", "grok", "gemini") else "disabled"
            for rb in self._content_radios:
                rb.configure(state=state)


        self.local_engine.trace_add("write", _update_content_state)
        _update_content_state()

        _cb_lipsync = ttk.Checkbutton(tab, text="Použiť lipsync (MuseTalk)", variable=self.local_lipsync)
        _cb_lipsync.grid(row=7, column=0, columnspan=2, sticky="w", pady=4)
        tip(_cb_lipsync, "Lipsync prepíše pohyb pier v origináli podľa nového hlasu pomocou MuseTalk. Vyžaduje GPU + MuseTalk modely.")

        _cb_music = ttk.Checkbutton(tab, text="Zachovať hudbu", variable=self.local_keep_music)
        _cb_music.grid(row=8, column=0, columnspan=2, sticky="w", pady=4)
        tip(_cb_music, "Oddelí hudbu od reči (Demucs), zachová hudbu v pozadí finálneho videa. Výrazne predlžuje spracovanie.")

        _cb_ctx = ttk.Checkbutton(tab, text="Kontextový preklad", variable=self.local_context)
        _cb_ctx.grid(row=9, column=0, columnspan=2, sticky="w", pady=4)
        tip(_cb_ctx, "Prekladá viac segmentov naraz s kontextom predchádzajúcich viet — lepšia konzistencia termínov. Len pre Llama/LLM engine.")

        _cb_eq = ttk.Checkbutton(tab, text="Voice EQ", variable=self.local_voice_eq)
        _cb_eq.grid(row=10, column=0, sticky="w", pady=4)
        tip(_cb_eq, "Aplikuje EQ filter na TTS hlas (zvýraznenie výšok, odstránenie bum). Vyberte profil vedľa.")
        _eq_profile_names = [""] + list(EQ_BUILTIN_PRESETS.keys())
        ttk.Label(tab, text="EQ profil").grid(row=10, column=1, sticky="e", pady=4, padx=(0, 4))
        ttk.Combobox(
            tab,
            textvariable=self.local_tts_eq_profile,
            values=_eq_profile_names,
            state="readonly",
            width=18,
        ).grid(row=10, column=2, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Multi-voice (Chatterbox)", variable=self.local_multi_voice).grid(
            row=11, column=0, sticky="w", pady=4
        )
        ttk.Label(tab, text="Počet hlasov:").grid(row=11, column=1, sticky="e", pady=4, padx=(0, 4))
        ttk.Spinbox(tab, textvariable=self.local_multi_voice_n, from_=2, to=6, increment=1, width=4).grid(
            row=11, column=2, sticky="w", pady=4
        )
        _cb_shutdown = ttk.Checkbutton(tab, text="Vypnúť PC po dokončení", variable=self.shutdown_after)
        _cb_shutdown.grid(row=7, column=2, sticky="w", pady=4)
        tip(_cb_shutdown, "Po dokončení celého batchu automaticky vypne počítač (poweroff).")

        # Collapsible "Pokročilé" section (pyvideotrans-style "More settings")
        self._adv_local_open = tk.BooleanVar(value=False)
        _adv_toggle_btn = ttk.Button(
            tab, text="▶  Pokročilé nastavenia",
            command=lambda: self._toggle_adv_local(_adv_frame, _adv_toggle_btn),
        )
        _adv_toggle_btn.grid(row=12, column=0, columnspan=3, sticky="w", pady=(6, 0), padx=2)
        tip(_adv_toggle_btn, "Zobrazí / skryje pokročilé možnosti: klónovanie hlasu, denoise, pitch, safe mode…")

        _adv_frame = ttk.Frame(tab)
        # (hidden by default — shown on toggle)

        _cb_clone = ttk.Checkbutton(_adv_frame, text="Voice clone (Chatterbox)", variable=self.local_clone_voice)
        _cb_clone.grid(row=0, column=0, columnspan=2, sticky="w", pady=3)
        tip(_cb_clone, "Klonuje hlas zo zdrojového videa a použije ho pre Chatterbox TTS. Vyžaduje dobrý zdrojový zvuk.")

        _cb_safe = ttk.Checkbutton(_adv_frame, text="TTS Safe mode (menej EOS artefaktov)", variable=self.local_tts_safe_mode)
        _cb_safe.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        tip(_cb_safe, "Bezpečný režim TTS: zabraňuje artefaktom na konci segmentov (EOS = end-of-sequence glitches). Mierne spomaľuje.")

        _cb_text_adapt = ttk.Checkbutton(_adv_frame, text="Text adaptation (timing-aware rewrite)",
                                          variable=self.local_use_text_adaptation)
        _cb_text_adapt.grid(row=4, column=0, columnspan=2, sticky="w", pady=3)
        tip(_cb_text_adapt,
            "Ak TTS odhadovaná dĺžka preteká slot, pokúsi sa vetu skrátiť (concise → dub-friendly rewrite).\n"
            "Vyžaduje --llama_model pre LLM rewrite. Bez neho len hlási overflow, stretch zachráni zvyšok.")

        _cb_autogain = ttk.Checkbutton(_adv_frame, text="Auto speech gain", variable=self.local_auto_speech_gain)
        _cb_autogain.grid(row=5, column=0, sticky="w", pady=3)
        tip(_cb_autogain, "Automaticky normalizuje hlasitosť TTS výstupu podľa hlasitosti originálu (RMS matching).")

        # Pipeline preset row
        _preset_frame = ttk.Frame(_adv_frame)
        _preset_frame.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 2))
        ttk.Label(_preset_frame, text="Pipeline preset:").pack(side="left", padx=(0, 6))
        _preset_cb = ttk.Combobox(_preset_frame, textvariable=self.local_preset, state="readonly", width=18,
                                   values=["", "production_dub", "teaching_sql"])
        _preset_cb.pack(side="left")
        tip(_preset_cb,
            "production_dub: čistý dubbing — phonetic OFF, SQL TTS OFF, light normalizer. Odporúčané pre filmy/dokumenty.\n"
            "teaching_sql: SQL/IT výukové video — všetky normalizéry ON, SQL fonetika ON.\n"
            "(prázdne) = manuálne nastavenia")

        ttk.Label(_adv_frame, text="Pitch hlasu (st):").grid(row=4, column=1, sticky="e", pady=3, padx=(0, 4))
        _pitch_spin = ttk.Spinbox(_adv_frame, textvariable=self.local_pitch_shift,
                    from_=-12, to=12, increment=0.5, width=6)
        _pitch_spin.grid(row=4, column=2, sticky="w", pady=3)
        tip(_pitch_spin, "Posun výšky hlasu v poltónoch. 0 = bez zmeny. +3 = jemne vyšší, -3 = nižší.")

        # max_chars widget is in Settings tab

        # Variables for timing/volume/VAD (widgets are in Settings tab)
        self.local_stretch_min = tk.StringVar(value=self.settings.def_stretch_min)
        self.local_stretch_max = tk.StringVar(value=self.settings.def_stretch_max)
        self.local_pause_gap_s = tk.StringVar(value=self.settings.def_pause_gap_s)
        self.local_music_volume = tk.StringVar(value=self.settings.def_music_volume)
        self.local_speech_gain = tk.StringVar(value=self.settings.def_speech_gain)
        self.local_vad_min_speech = tk.StringVar(value=self.settings.def_vad_min_speech_ms)
        self.local_vad_min_silence = tk.StringVar(value=self.settings.def_vad_min_silence_ms)
        self.local_vad_max_speech = tk.StringVar(value=self.settings.def_vad_max_speech_s)
        self.local_whisper_prompt = tk.StringVar(value="")

    def _pick_local(self) -> None:
        # Legacy single file pick (for backwards compatibility)
        path = filedialog.askopenfilename(title="Vyber video")
        if path:
            self.local_src.set(path)

    def _get_chatterbox_voices(self) -> list[str]:
        """Return list of WAV filenames in voices/ directory."""
        wavs = sorted(p.name for p in VOICES_DIR.glob("*.wav"))
        return ["(predvolený)"] + wavs

    def _normalize_chatterbox_model_name(self, value: str | None) -> str:
        raw = (value or "").strip()
        if not raw:
            return DEFAULT_CHATTERBOX_MODELS[0]
        if raw.endswith(".safetensors"):
            return Path(raw).name
        return raw

    def _get_chatterbox_models(self) -> list[str]:
        names = set(DEFAULT_CHATTERBOX_MODELS)
        if CHATTERBOX_MODELS_DIR.exists():
            names.update(p.name for p in CHATTERBOX_MODELS_DIR.glob("*.safetensors"))
        names.add(self._normalize_chatterbox_model_name(self.settings.def_chatterbox_model))
        current = getattr(self, "chatterbox_model_var", None)
        if current is not None:
            names.add(self._normalize_chatterbox_model_name(current.get()))
        return sorted(name for name in names if name)

    def _get_selected_chatterbox_model_path(self) -> str:
        current = getattr(self, "chatterbox_model_var", None)
        model_name = self._normalize_chatterbox_model_name(
            current.get() if current is not None else self.settings.def_chatterbox_model
        )
        return f"models/chatterbox/{model_name}"

    def _on_chatterbox_voice_selected(self, event=None) -> None:
        """When a voice is selected from dropdown, update xtts_speaker_wav path."""
        name = self.chatterbox_voice_var.get()
        if name and name != "(predvolený)":
            self.xtts_speaker_wav.set(str(VOICES_DIR / name))
        else:
            self.xtts_speaker_wav.set("")

    def _refresh_chatterbox_voices(self) -> None:
        """Refresh the voice dropdown list."""
        voices = self._get_chatterbox_voices()
        self._chatterbox_voice_cb["values"] = voices
        current = self.chatterbox_voice_var.get()
        if current not in voices:
            self.chatterbox_voice_var.set("(predvolený)")
            self.xtts_speaker_wav.set("")

    def _refresh_chatterbox_models(self) -> None:
        """Refresh the Chatterbox model dropdown list."""
        models = self._get_chatterbox_models()
        self._chatterbox_model_cb["values"] = models
        current = self._normalize_chatterbox_model_name(self.chatterbox_model_var.get())
        if current not in models:
            self.chatterbox_model_var.set(DEFAULT_CHATTERBOX_MODELS[0])

    def _toggle_adv_local(self, frame: ttk.Frame, btn: ttk.Button) -> None:
        """Show/hide the advanced local settings panel (pyvideotrans-style collapsible)."""
        if self._adv_local_open.get():
            frame.grid_forget()
            self._adv_local_open.set(False)
            btn.configure(text="▶  Pokročilé nastavenia")
        else:
            frame.grid(row=13, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 4))
            self._adv_local_open.set(True)
            btn.configure(text="▼  Pokročilé nastavenia")

    def _pick_local_batch(self) -> None:
        """Add multiple files to the batch list."""
        paths = filedialog.askopenfilenames(
            title="Vyber videá (môžeš vybrať viacero)",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All files", "*.*")],
        )
        for p in paths:
            if p and p not in [item.path for item in self.batch_items]:
                item = BatchItem(path=p)
                self.batch_items.append(item)
                self._update_batch_listbox()

    def _remove_batch_selected(self) -> None:
        """Remove selected items from batch list."""
        selected = list(self.batch_listbox.curselection())
        for idx in reversed(selected):
            if idx < len(self.batch_items):
                del self.batch_items[idx]
        self._update_batch_listbox()

    def _clear_batch_list(self) -> None:
        """Clear all items from batch list."""
        self.batch_items.clear()
        self._update_batch_listbox()

    def _save_batch_list(self) -> None:
        """Save current batch list under a user-chosen name."""
        if not self.batch_items:
            messagebox.showinfo("Prázdny zoznam", "Nemáte žiadne videá na uloženie.")
            return
        name = simpledialog.askstring("Uložiť zoznam", "Názov zoznamu:", parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        lists = _load_video_lists()
        paths = [item.path for item in self.batch_items]
        if name in lists:
            if not messagebox.askyesno("Prepísať?", f"Zoznam '{name}' už existuje. Prepísať?"):
                return
        lists[name] = paths
        _save_video_lists(lists)
        messagebox.showinfo("Uložené", f"Zoznam '{name}' uložený ({len(paths)} videí).")

    def _load_list_into_batch(self, name: str) -> None:
        """Load a named list into the batch listbox."""
        lists = _load_video_lists()
        paths = lists.get(name, [])
        if not paths:
            return
        self.batch_items.clear()
        for p in paths:
            self.batch_items.append(BatchItem(path=p))
        self._update_batch_listbox()

    def _open_list_manager(self) -> None:
        """Open a window to browse, load, and delete saved video lists."""
        win = tk.Toplevel(self)
        win.title("Uložené zoznamy videí")
        win.geometry("700x500")
        win.transient(self)
        win.grab_set()

        # Left: list of saved names
        left = ttk.Frame(win)
        left.pack(side="left", fill="y", padx=6, pady=6)

        ttk.Label(left, text="Zoznamy:", font=(self.font_ui_family, 10, "bold")).pack(anchor="w")
        name_listbox = tk.Listbox(left, width=25, height=20)
        name_listbox.pack(fill="y", expand=True, pady=4)

        # Right: detail view
        right = ttk.Frame(win)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        ttk.Label(right, text="Videá v zozname:", font=(self.font_ui_family, 10, "bold")).pack(anchor="w")
        detail_frame = ttk.Frame(right)
        detail_frame.pack(fill="both", expand=True, pady=4)
        detail_listbox = tk.Listbox(detail_frame, width=60, height=18, selectmode=tk.EXTENDED)
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=detail_listbox.yview)
        detail_listbox.configure(yscrollcommand=detail_scroll.set)
        detail_listbox.pack(side="left", fill="both", expand=True)
        detail_scroll.pack(side="right", fill="y")

        info_label = ttk.Label(right, text="", foreground="gray")
        info_label.pack(anchor="w")

        # Buttons
        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", pady=6)

        lists_data = _load_video_lists()

        def refresh_names():
            nonlocal lists_data
            lists_data = _load_video_lists()
            name_listbox.delete(0, tk.END)
            for n in sorted(lists_data.keys()):
                count = len(lists_data[n])
                name_listbox.insert(tk.END, f"{n}  ({count})")

        def on_name_select(event=None):
            sel = name_listbox.curselection()
            if not sel:
                return
            # Extract name from "name  (count)" format
            entry = name_listbox.get(sel[0])
            name = entry.rsplit("  (", 1)[0]
            paths = lists_data.get(name, [])
            detail_listbox.delete(0, tk.END)
            for p in paths:
                detail_listbox.insert(tk.END, Path(p).name)
            # Check which files exist
            existing = sum(1 for p in paths if Path(p).exists())
            info_label.configure(text=f"{len(paths)} videí, {existing} existuje na disku")

        def get_selected_name():
            sel = name_listbox.curselection()
            if not sel:
                messagebox.showinfo("Výber", "Vyberte zoznam.", parent=win)
                return None
            entry = name_listbox.get(sel[0])
            return entry.rsplit("  (", 1)[0]

        def load_selected():
            name = get_selected_name()
            if name:
                self._load_list_into_batch(name)
                win.destroy()

        def delete_selected():
            name = get_selected_name()
            if not name:
                return
            if not messagebox.askyesno("Odstrániť?", f"Odstrániť zoznam '{name}'?", parent=win):
                return
            lists_data.pop(name, None)
            _save_video_lists(lists_data)
            refresh_names()
            detail_listbox.delete(0, tk.END)
            info_label.configure(text="")

        def remove_video_from_list():
            """Remove selected videos from the currently viewed list."""
            name = get_selected_name()
            if not name:
                return
            sel = list(detail_listbox.curselection())
            if not sel:
                messagebox.showinfo("Výber", "Vyberte videá na odstránenie.", parent=win)
                return
            paths = lists_data.get(name, [])
            for idx in reversed(sel):
                if idx < len(paths):
                    del paths[idx]
            lists_data[name] = paths
            _save_video_lists(lists_data)
            on_name_select()  # refresh detail view
            refresh_names()  # refresh counts

        def rename_selected():
            name = get_selected_name()
            if not name:
                return
            new_name = simpledialog.askstring("Premenovať", f"Nový názov pre '{name}':", parent=win)
            if not new_name or not new_name.strip():
                return
            new_name = new_name.strip()
            if new_name in lists_data:
                messagebox.showerror("Existuje", f"Zoznam '{new_name}' už existuje.", parent=win)
                return
            lists_data[new_name] = lists_data.pop(name)
            _save_video_lists(lists_data)
            refresh_names()

        ttk.Button(btn_row, text="Načítať do fronty", command=load_selected).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Odstrániť video", command=remove_video_from_list).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Premenovať", command=rename_selected).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Zmazať zoznam", command=delete_selected).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Zavrieť", command=win.destroy).pack(side="right", padx=4)

        name_listbox.bind("<<ListboxSelect>>", on_name_select)
        refresh_names()
        self._apply_theme_to_widget_tree(win)

    # ------------------------------------------------------------------
    # Named checkpoint management (save/load batch state with progress)
    # ------------------------------------------------------------------
    def _save_checkpoint_named(self) -> None:
        """Save current batch state (items + settings) as a named checkpoint."""
        if not self.batch_items:
            messagebox.showinfo("Prázdny zoznam", "Nemáte žiadne videá na uloženie checkpointu.")
            return
        name = simpledialog.askstring("Uložiť checkpoint", "Názov checkpointu:", parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        checkpoints = _load_saved_checkpoints()
        if name in checkpoints:
            if not messagebox.askyesno("Prepísať?", f"Checkpoint '{name}' už existuje. Prepísať?"):
                return
        # Gather current settings
        settings_snapshot = {
            "lang": self.local_lang.get().strip() or "cs",
            "lipsync": self.local_lipsync.get(),
            "keep_music": self.local_keep_music.get(),
            "denoise": self.local_denoise.get(),
            "denoise_preset": self.local_denoise_preset.get(),
            "auto_speech_gain": self.local_auto_speech_gain.get(),
            "base_tempo": self.local_base_tempo.get(),
            "tts": self.local_tts.get(),
            "trans_engine": self.local_engine.get(),
            "out_dir": self.local_out.get().strip(),
            "content_type": self.local_content_type.get(),
            "stretch_min": self.local_stretch_min.get(),
            "stretch_max": self.local_stretch_max.get(),
            "pause_gap_s": self.local_pause_gap_s.get(),
            "vad_min_speech_ms": self.local_vad_min_speech.get(),
            "vad_min_silence_ms": self.local_vad_min_silence.get(),
            "vad_max_speech_s": self.local_vad_max_speech.get(),
        }
        items_data = [{"path": i.path, "status": i.status, "error": i.error} for i in self.batch_items]
        done_count = sum(1 for i in self.batch_items if i.status == "done")
        total = len(self.batch_items)
        checkpoints[name] = {
            "items": items_data,
            "settings": settings_snapshot,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "done": done_count,
            "total": total,
        }
        _save_saved_checkpoints(checkpoints)
        messagebox.showinfo("Uložené", f"Checkpoint '{name}' uložený ({done_count}/{total} hotových).")

    def _load_checkpoint_named(self, name: str) -> None:
        """Load a named checkpoint into the batch and restore settings."""
        checkpoints = _load_saved_checkpoints()
        ckpt = checkpoints.get(name)
        if not ckpt:
            return
        items = [BatchItem(path=i["path"], status=i["status"], error=i.get("error", "")) for i in ckpt["items"]]
        # Reset "processing" status to "pending"
        for item in items:
            if item.status == "processing":
                item.status = "pending"
        self.batch_items = items
        # Restore settings
        s = ckpt.get("settings", {})
        if "lang" in s:
            self.local_lang.set(s["lang"])
        if "lipsync" in s:
            self.local_lipsync.set(s["lipsync"])
        if "keep_music" in s:
            self.local_keep_music.set(s["keep_music"])
        if "denoise" in s:
            self.local_denoise.set(s["denoise"])
        if "denoise_preset" in s:
            self.local_denoise_preset.set(s["denoise_preset"])
        if "auto_speech_gain" in s:
            self.local_auto_speech_gain.set(s["auto_speech_gain"])
        if "base_tempo" in s:
            self.local_base_tempo.set(s["base_tempo"])
        if "tts" in s:
            self.local_tts.set(s["tts"])
        if "trans_engine" in s:
            self.local_engine.set(s["trans_engine"])
        if "out_dir" in s:
            self.local_out.set(s["out_dir"])
        if "content_type" in s:
            self.local_content_type.set(s["content_type"])
        if "stretch_min" in s:
            self.local_stretch_min.set(s["stretch_min"])
        if "stretch_max" in s:
            self.local_stretch_max.set(s["stretch_max"])
        if "pause_gap_s" in s:
            self.local_pause_gap_s.set(s["pause_gap_s"])
        if "vad_min_speech_ms" in s:
            self.local_vad_min_speech.set(s["vad_min_speech_ms"])
        if "vad_min_silence_ms" in s:
            self.local_vad_min_silence.set(s["vad_min_silence_ms"])
        if "vad_max_speech_s" in s:
            self.local_vad_max_speech.set(s["vad_max_speech_s"])
        self._update_batch_listbox()
        self._queue_log(f"Checkpoint '{name}' načítaný ({sum(1 for i in items if i.status == 'done')}/{len(items)} hotových)")

    def _open_checkpoint_manager(self) -> None:
        """Open a window to browse, load, and delete saved checkpoints."""
        win = tk.Toplevel(self)
        win.title("Uložené checkpointy")
        win.geometry("750x520")
        win.transient(self)
        win.grab_set()

        # Left: list of saved checkpoint names
        left = ttk.Frame(win)
        left.pack(side="left", fill="y", padx=6, pady=6)

        ttk.Label(left, text="Checkpointy:", font=(self.font_ui_family, 10, "bold")).pack(anchor="w")
        name_listbox = tk.Listbox(left, width=30, height=20)
        name_listbox.pack(fill="y", expand=True, pady=4)

        # Right: detail view
        right = ttk.Frame(win)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        info_label = ttk.Label(right, text="", font=(self.font_ui_family, 9))
        info_label.pack(anchor="w", pady=(0, 4))

        ttk.Label(right, text="Videá:", font=(self.font_ui_family, 10, "bold")).pack(anchor="w")
        detail_frame = ttk.Frame(right)
        detail_frame.pack(fill="both", expand=True, pady=4)
        detail_listbox = tk.Listbox(detail_frame, width=60, height=16)
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=detail_listbox.yview)
        detail_listbox.configure(yscrollcommand=detail_scroll.set)
        detail_listbox.pack(side="left", fill="both", expand=True)
        detail_scroll.pack(side="right", fill="y")

        ckpt_data = _load_saved_checkpoints()

        def refresh_names():
            nonlocal ckpt_data
            ckpt_data = _load_saved_checkpoints()
            name_listbox.delete(0, tk.END)
            for n in sorted(ckpt_data.keys()):
                c = ckpt_data[n]
                done = c.get("done", 0)
                total = c.get("total", 0)
                name_listbox.insert(tk.END, f"{n}  ({done}/{total})")

        def on_name_select(event=None):
            sel = name_listbox.curselection()
            if not sel:
                return
            entry = name_listbox.get(sel[0])
            name = entry.rsplit("  (", 1)[0]
            ckpt = ckpt_data.get(name, {})
            items = ckpt.get("items", [])
            s = ckpt.get("settings", {})
            ts = ckpt.get("timestamp", "?")
            done = sum(1 for i in items if i["status"] == "done")
            failed = sum(1 for i in items if i["status"] == "failed")
            pending = sum(1 for i in items if i["status"] in ("pending", "processing"))
            info_label.configure(
                text=f"Uložené: {ts}  |  Jazyk: {s.get('lang','?')}  |  TTS: {s.get('tts','?')}  |  "
                     f"Hotové: {done}  Zostáva: {pending}  Zlyhané: {failed}"
            )
            detail_listbox.delete(0, tk.END)
            status_icons = {"done": "\u2713", "failed": "\u2717", "pending": "\u25cb", "processing": "\u25b6"}
            for item in items:
                icon = status_icons.get(item["status"], "?")
                detail_listbox.insert(tk.END, f"{icon}  {Path(item['path']).name}")
                idx = detail_listbox.size() - 1
                if item["status"] == "done":
                    detail_listbox.itemconfig(idx, fg="green")
                elif item["status"] == "failed":
                    detail_listbox.itemconfig(idx, fg="red")

        def get_selected_name():
            sel = name_listbox.curselection()
            if not sel:
                messagebox.showinfo("Výber", "Vyberte checkpoint.", parent=win)
                return None
            entry = name_listbox.get(sel[0])
            return entry.rsplit("  (", 1)[0]

        def load_selected():
            name = get_selected_name()
            if name:
                self._load_checkpoint_named(name)
                win.destroy()

        def delete_selected():
            name = get_selected_name()
            if not name:
                return
            if not messagebox.askyesno("Odstrániť?", f"Odstrániť checkpoint '{name}'?", parent=win):
                return
            ckpt_data.pop(name, None)
            _save_saved_checkpoints(ckpt_data)
            refresh_names()
            detail_listbox.delete(0, tk.END)
            info_label.configure(text="")

        def rename_selected():
            name = get_selected_name()
            if not name:
                return
            new_name = simpledialog.askstring("Premenovať", f"Nový názov pre '{name}':", parent=win)
            if not new_name or not new_name.strip():
                return
            new_name = new_name.strip()
            if new_name in ckpt_data:
                messagebox.showerror("Existuje", f"Checkpoint '{new_name}' už existuje.", parent=win)
                return
            ckpt_data[new_name] = ckpt_data.pop(name)
            _save_saved_checkpoints(ckpt_data)
            refresh_names()

        # Buttons
        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", pady=6)
        ttk.Button(btn_row, text="Načítať", command=load_selected).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Premenovať", command=rename_selected).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Zmazať", command=delete_selected).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Zavrieť", command=win.destroy).pack(side="right", padx=4)

        name_listbox.bind("<<ListboxSelect>>", on_name_select)
        refresh_names()
        self._apply_theme_to_widget_tree(win)

    def _update_batch_listbox(self) -> None:
        """Refresh the listbox display with current batch items."""
        self.batch_listbox.delete(0, tk.END)
        for item in self.batch_items:
            status_icon = {"pending": "○", "processing": "▶", "done": "✓", "failed": "✗"}.get(item.status, "?")
            name = Path(item.path).name
            self.batch_listbox.insert(tk.END, f"{status_icon} {name}")
            # Color coding
            idx = self.batch_listbox.size() - 1
            if item.status == "done":
                self.batch_listbox.itemconfig(idx, fg="green")
            elif item.status == "failed":
                self.batch_listbox.itemconfig(idx, fg="red")
            elif item.status == "processing":
                self.batch_listbox.itemconfig(idx, fg="blue")

    def _update_resume_button(self) -> None:
        """Show/hide resume button based on checkpoint existence."""
        checkpoint = load_checkpoint()
        if checkpoint:
            items, _idx, _ = checkpoint
            # "processing" means app crashed during that video - treat as pending
            pending_count = sum(1 for i in items if i.status in ("pending", "failed", "processing"))
            if pending_count > 0:
                self.resume_btn.configure(text=f"Pokračovať ({pending_count}) ❯", state="normal")
                return
        self.resume_btn.configure(text="Pokračovať ❯", state="disabled")

    def _resume_from_checkpoint(self) -> None:
        """Load checkpoint and resume batch processing."""
        checkpoint = load_checkpoint()
        if not checkpoint:
            messagebox.showinfo("Checkpoint", "Žiadny checkpoint na obnovenie.")
            return
        items, current_idx, settings = checkpoint
        # Reset "processing" status to "pending" (app crashed during that video)
        for item in items:
            if item.status == "processing":
                item.status = "pending"
        # Confirm with user
        pending = [i for i in items if i.status in ("pending", "failed")]
        done = [i for i in items if i.status == "done"]
        msg = f"Nájdený checkpoint:\n• Dokončené: {len(done)}\n• Zostáva: {len(pending)}\n\nPokračovať?"
        if not messagebox.askyesno("Obnoviť checkpoint", msg):
            return
        # Restore state
        self.batch_items = items
        self.batch_current_index = current_idx
        # Restore settings if available
        if settings:
            if "lang" in settings:
                self.local_lang.set(settings["lang"])
            if "lipsync" in settings:
                self.local_lipsync.set(settings["lipsync"])
            if "keep_music" in settings:
                self.local_keep_music.set(settings["keep_music"])
            if "denoise" in settings:
                self.local_denoise.set(settings["denoise"])
            if "denoise_preset" in settings:
                self.local_denoise_preset.set(settings["denoise_preset"])
            if "auto_speech_gain" in settings:
                self.local_auto_speech_gain.set(settings["auto_speech_gain"])
            if "base_tempo" in settings:
                self.local_base_tempo.set(settings["base_tempo"])
            if "tts" in settings:
                self.local_tts.set(settings["tts"])
            if "trans_engine" in settings:
                self.local_engine.set(settings["trans_engine"])
            if "out_dir" in settings:
                self.local_out.set(settings["out_dir"])
            if "content_type" in settings:
                self.local_content_type.set(settings["content_type"])
            if "stretch_min" in settings:
                self.local_stretch_min.set(settings["stretch_min"])
            if "stretch_max" in settings:
                self.local_stretch_max.set(settings["stretch_max"])
            if "pause_gap_s" in settings:
                self.local_pause_gap_s.set(settings["pause_gap_s"])
            if "vad_min_speech_ms" in settings:
                self.local_vad_min_speech.set(settings["vad_min_speech_ms"])
            if "vad_min_silence_ms" in settings:
                self.local_vad_min_silence.set(settings["vad_min_silence_ms"])
            if "vad_max_speech_s" in settings:
                self.local_vad_max_speech.set(settings["vad_max_speech_s"])
        self._update_batch_listbox()
        # Start processing
        self._run_local()

    def _pick_local_out_dir(self) -> None:
        """Pick output directory for batch processing."""
        path = filedialog.askdirectory(title="Výstupný priečinok")
        if path:
            self.local_out.set(path)

    def _pick_local_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Kam uložiť výsledné video",
            defaultextension=".mp4",
            filetypes=[("Video", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self.local_out.set(path)

    def _run_local(self) -> None:
        # Build batch list from listbox if empty (support legacy single-file mode too)
        if not self.batch_items:
            src = self.local_src.get().strip()
            if src and Path(src).exists():
                self.batch_items = [BatchItem(path=src)]
            else:
                messagebox.showerror("Chýba vstup", "Pridajte aspoň jedno video do zoznamu.")
                return

        # Validate - also include "processing" (app crashed during that video)
        pending_items = [item for item in self.batch_items if item.status in ("pending", "failed", "processing")]
        if not pending_items:
            # All done - ask user if they want to re-run
            if messagebox.askyesno("Batch dokončený", "Všetky videá už boli spracované.\nChcete ich spustiť znova?"):
                for item in self.batch_items:
                    item.status = "pending"
                    item.error = ""
                self._update_batch_listbox()
            else:
                return
        if not self.settings.py.exists():
            messagebox.showerror("Python nenájdený", str(self.settings.py))
            return

        tgt = self.local_lang.get().strip() or "cs"
        custom_out = self.local_out.get().strip()

        content_type = self.local_content_type.get()

        # Save settings snapshot for checkpoint
        settings_snapshot = {
            "lang": tgt,
            "lipsync": self.local_lipsync.get(),
            "keep_music": self.local_keep_music.get(),
            "denoise": self.local_denoise.get(),
            "denoise_preset": self.local_denoise_preset.get(),
            "auto_speech_gain": self.local_auto_speech_gain.get(),
            "base_tempo": self.local_base_tempo.get(),
            "tts": self.local_tts.get(),
            "trans_engine": self.local_engine.get(),
            "out_dir": custom_out,
            "content_type": content_type,
            "stretch_min": self.local_stretch_min.get(),
            "stretch_max": self.local_stretch_max.get(),
            "pause_gap_s": self.local_pause_gap_s.get(),
            "vad_min_speech_ms": self.local_vad_min_speech.get(),
            "vad_min_silence_ms": self.local_vad_min_silence.get(),
            "vad_max_speech_s": self.local_vad_max_speech.get(),
        }

        # Pipeline progress state for Local
        def pipeline_line_handler(line: str) -> bool:
            offset = 1 if self.current_steps and self.current_steps[0].startswith("Sťahovanie") else 0
            if "[WHISPER] device=" in line:
                self._queue_step_active(offset + 1)
            elif "[WHISPER]" in line and ("Segment" in line or "VAD" in line):
                self._queue_step_active(offset + 2)
            elif "[LLAMA]" in line or "[MADLAD]" in line or "[MULTISLAV]" in line:
                self._queue_step_active(offset + 3)
            elif any(x in line for x in ["[CHATTERBOX]", "[ONNX-TTS]", "[PIPER]", "[ALLTALK]"]):
                self._queue_step_active(offset + 4)
            elif "[TIMELINE]" in line or "[SYNC]" in line:
                self._queue_step_active(offset + 5)
            return False

        def process_single_video(src: str, video_idx: int, total_videos: int) -> bool:
            """Process a single video. Returns True if successful."""
            self._queue_log(f"\n{'='*50}")
            self._queue_log(f"[BATCH] Video {video_idx + 1}/{total_videos}: {Path(src).name}")
            self._queue_log(f"{'='*50}")

            # Define steps
            steps = [
                "Extrakcia audia z videa",
                "VAD/Segmentácia",
                "Whisper (prepis reči)",
                "Preklad (do SK/CZ)",
                "TTS (syntéza hlasu)",
                "Časovanie audia",
            ]
            if self.local_lipsync.get():
                steps.append("Lipsync (pohyb pier)")
            else:
                steps.append("Export/Mux videa")
            steps.append("Čistenie")

            self.current_steps = steps
            self._queue_step_init(steps)
            self._queue_step_active(0)

            trans_engine = self.local_engine.get()
            suffix = {"llama": "llama", "madlad": "madlad", "multislav5lang": "multislav5lang", "hybrid": "hybrid", "google_madlad": "google_madlad", "chatgpt": "chatgpt", "grok": "grok", "gemini": "gemini", "google": "google"}.get(trans_engine, "madlad")

            trans_args: list[str] = []
            _llm_model = self.local_llm_model.get() or "/mnt/tts_data/VideoTranslator_studio/models/gemma-3-12b-it/google_gemma-3-12b-it-Q8_0.gguf"
            _eurollm_path = "/mnt/tts_data/VideoTranslator_studio/models/eurollm/EuroLLM-9B-Instruct-Q6_K_L.gguf"
            _gemma27_path = "/mnt/tts_data/VideoTranslator_studio/models/gemma-3-27b-it/google_gemma-3-27b-it-Q6_K.gguf"
            if trans_engine == "llama":
                trans_args = ["--llama_model", _llm_model, "--llama_gpu_layers", "56"]
            elif trans_engine == "eurollm_gemma":
                trans_args = [
                    "--llama_model", _eurollm_path, "--llama_gpu_layers", "56",
                    "--refine_translation",
                    "--refine_model", _gemma27_path, "--refine_gpu_layers", "56",
                ]
            elif trans_engine == "google":
                trans_args = ["--use_google_translate"]
            elif trans_engine == "google_madlad":
                madlad_m = self.local_madlad_model.get() or "google/madlad400-7b-mt"
                if self.local_use_madlad_gemma_qa.get():
                    # Google Phase 1 + Gemma/Llama QA
                    trans_args = [
                        "--use_google_translate", "--use_madlad_gemma_qa",
                        "--madlad_model", madlad_m,
                        "--llama_model", _llm_model,
                        "--llama_gpu_layers", "56",
                        "--qa_model_path", "models/llama-3-8b-abliterated-v3/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
                        "--qa_gpu_layers", "32",
                    ]
                else:
                    # Google primary + MADLAD fallback only
                    trans_args = ["--hybrid_translate", "--madlad_model", madlad_m]
            elif trans_engine == "hybrid":
                madlad_m = self.local_madlad_model.get() or "google/madlad400-7b-mt"
                trans_args = [
                    "--hybrid_translate", "--madlad_model", madlad_m,
                    "--hybrid_qa",
                    "--llama_model", _llm_model,
                    "--llama_gpu_layers", "56",
                ]
            elif trans_engine == "madlad":
                madlad_m = self.local_madlad_model.get() or "google/madlad400-7b-mt"
                trans_args = ["--use_madlad", "--madlad_model", madlad_m]
            elif trans_engine == "multislav5lang":
                trans_args = ["--use_multislav5lang"]
            elif trans_engine in ("chatgpt", "grok", "gemini"):
                if trans_engine == "chatgpt":
                    api_key = (
                        (self.openai_api_key_var.get().strip() if hasattr(self, "openai_api_key_var") else self.settings.openai_api_key.strip())
                        or os.environ.get("API_TRANS_KEY", "").strip()
                        or os.environ.get("OPENAI_API_KEY", "").strip()
                    )
                    model = (self.openai_translate_model_var.get().strip() if hasattr(self, "openai_translate_model_var") else self.settings.openai_translate_model) or "gpt-5.2-mini"
                    base_url = (self.openai_tts_base_url_var.get().strip() if hasattr(self, "openai_tts_base_url_var") else self.settings.openai_tts_base_url) or "https://api.openai.com/v1"
                elif trans_engine == "grok":
                    api_key = (
                        (self.grok_api_key_var.get().strip() if hasattr(self, "grok_api_key_var") else self.settings.grok_api_key.strip())
                        or os.environ.get("API_TRANS_KEY", "").strip()
                        or os.environ.get("GROK_API_KEY", "").strip()
                        or os.environ.get("XAI_API_KEY", "").strip()
                    )
                    model = (self.grok_translate_model_var.get().strip() if hasattr(self, "grok_translate_model_var") else self.settings.grok_translate_model) or "grok-3-mini"
                    base_url = (self.grok_tts_base_url_var.get().strip() if hasattr(self, "grok_tts_base_url_var") else self.settings.grok_tts_base_url) or "https://api.x.ai/v1"
                else:
                    api_key = (
                        (self.gemini_api_key_var.get().strip() if hasattr(self, "gemini_api_key_var") else self.settings.gemini_api_key.strip())
                        or os.environ.get("API_TRANS_KEY", "").strip()
                        or os.environ.get("GEMINI_API_KEY", "").strip()
                        or os.environ.get("GOOGLE_API_KEY", "").strip()
                    )
                    model = (self.gemini_translate_model_var.get().strip() if hasattr(self, "gemini_translate_model_var") else self.settings.gemini_translate_model) or "gemini-2.5-flash"
                    base_url = (self.gemini_tts_base_url_var.get().strip() if hasattr(self, "gemini_tts_base_url_var") else self.settings.gemini_tts_base_url) or "https://generativelanguage.googleapis.com/v1beta/openai"
                if not api_key:
                    self._queue_log(f"[ERROR] Chýba API key pre prekladový engine {trans_engine.upper()} v karte API.")
                    return False
                trans_args = [
                    "--use_api_translate",
                    "--api_translate_provider", trans_engine,
                    "--api_translate_model", model,
                    "--api_translate_base_url", base_url,
                ]
            else:
                # Hybrid = EuroLLM preklad + Gemma 27B refine pass
                trans_args = [
                    "--llama_model", _eurollm_path, "--llama_gpu_layers", "56",
                    "--refine_translation",
                    "--refine_model", _gemma27_path, "--refine_gpu_layers", "56",
                ]

            tts_engine = self.local_tts.get()
            tts_args: list[str] = []
            if tts_engine == "xtts":
                tts_args = ["--use_xtts", "--xtts_lang", tgt]
                spk = self.xtts_speaker_wav.get().strip() if hasattr(self, "xtts_speaker_wav") else ""
                if spk:
                    tts_args += ["--xtts_speaker_wav", spk]
                else:
                    xtts_spk = self.xtts_speaker.get().strip() if hasattr(self, "xtts_speaker") else "Damien Black"
                    if xtts_spk:
                        tts_args += ["--xtts_speaker", xtts_spk]
                try:
                    ps = float(self.xtts_pitch_shift.get())
                    if ps != 0.0:
                        tts_args += ["--xtts_pitch_shift", str(ps)]
                except (ValueError, AttributeError):
                    pass
            elif tts_engine == "chatterbox":
                tts_args = ["--use_chatterbox", "--chatterbox_model", self._get_selected_chatterbox_model_path()]
                cb_voice = self.xtts_speaker_wav.get().strip()
                if cb_voice and Path(cb_voice).exists():
                    tts_args += ["--xtts_speaker_wav", cb_voice]
                elif self.local_clone_voice.get():
                    tts_args.append("--clone_voice")
                if self.local_multi_voice.get():
                    tts_args.append("--multi_voice")
                    tts_args += ["--multi_voice_n", str(self.local_multi_voice_n.get())]
                if self.local_emotion_clone.get():
                    tts_args.append("--emotion_clone")
                if self.local_tts_safe_mode.get():
                    tts_args.append("--tts_safe_mode")
                if self.local_use_text_adaptation.get():
                    tts_args += ["--use_text_adaptation",
                                 "--llama_model", "models/gemma-3-12b-it/google_gemma-3-12b-it-Q8_0.gguf",
                                 "--llama_gpu_layers", "56"]
            elif tts_engine == "edge":
                tts_args = [
                    "--use_edge_tts",
                    "--edge_tts_voice", self.edge_tts_voice_var.get() or "cs-CZ-AntoninNeural",
                    "--edge_tts_rate", self.edge_tts_rate_var.get() or "+0%",
                    "--edge_tts_pitch", self.edge_tts_pitch_var.get() or "+0Hz",
                ]
                _e_style = getattr(self, "edge_tts_style_var", None)
                if _e_style and _e_style.get().strip():
                    tts_args += ["--edge_tts_style", _e_style.get().strip()]
                _e_v2 = getattr(self, "edge_tts_voice2_var", None)
                if _e_v2 and _e_v2.get().strip():
                    _voices = ",".join([
                        self.edge_tts_voice_var.get() or "cs-CZ-AntoninNeural",
                        _e_v2.get().strip(),
                    ])
                    tts_args += ["--edge_tts_voices", _voices, "--multi_voice"]
                _ba_vol = getattr(self, "backaudio_volume_var", None)
                if _ba_vol:
                    try:
                        _ba = float(_ba_vol.get())
                        if _ba > 0.0:
                            tts_args += ["--backaudio_volume", f"{_ba:.2f}"]
                    except ValueError:
                        pass
            else:
                tts_args = ["--use_chatterbox", "--chatterbox_model", self._get_selected_chatterbox_model_path()]

            music_vol = self.local_music_volume.get() or "0.25"
            music_args = ["--keep_music", "--music_volume", music_vol] if self.local_keep_music.get() else []
            context_args = ["--context_aware"] if (self.local_context.get() and trans_engine in ("llama",)) else []
            eq_args = ["--voice_eq"] if self.local_voice_eq.get() else []
            tts_eq_profile = self.local_tts_eq_profile.get().strip()
            if tts_eq_profile:
                eq_args += ["--tts_eq_profile", tts_eq_profile]
            content_args = ["--content_type", content_type] if content_type != "general" else []

            # Speech gain + pitch shift
            speech_gain = self.local_speech_gain.get() or "1.0"
            gain_args = ["--speech_gain", speech_gain]
            auto_gain_args = ["--auto-speech-gain"] if self.local_auto_speech_gain.get() else ["--no-auto-speech-gain"]
            try:
                ps_val = float(self.local_pitch_shift.get())
            except (ValueError, AttributeError):
                ps_val = 0.0
            pitch_args = ["--pitch_shift", str(ps_val)] if ps_val != 0.0 else []
            if self.local_denoise.get():
                denoise_args = ["--denoise_preset", self.local_denoise_preset.get() or "mild"]
            else:
                denoise_args = ["--no_denoise"]

            # Stretch/timing limits
            stretch_min = self.local_stretch_min.get() or "0.70"
            stretch_max = self.local_stretch_max.get() or "1.60"
            stretch_args = ["--stretch_min", stretch_min, "--stretch_max", stretch_max]
            base_tempo = self.local_base_tempo.get() or "1.06"
            tempo_args = ["--base_tempo", base_tempo]
            pause_gap_args = ["--max_pause_gap_s", self.local_pause_gap_s.get() or "2.50"]

            # VAD segmentation params
            vad_args = [
                "--vad_min_speech_ms", self.local_vad_min_speech.get() or "2000",
                "--vad_min_silence_ms", self.local_vad_min_silence.get() or "2500",
                "--vad_max_speech_s", self.local_vad_max_speech.get() or "90.0",
            ]

            # Whisper terminology hint
            _wp = self.local_whisper_prompt.get().strip()
            whisper_prompt_args = ["--whisper_prompt", _wp] if _wp else []

            # Pipeline preset
            _preset_val = self.local_preset.get().strip()
            preset_args = ["--preset", _preset_val] if _preset_val else []

            cmd = [
                _path_str(self.settings.py), _path_str(BASE_DIR / "pipeline.py"),
                "--input", src, "--tgt_lang", tgt,
                "--whisper_root", "models/faster-whisper",
                "--max_chars", self.local_max_chars.get() or "200",
                "--text_out", f"work/translation_{tgt}_{suffix}.txt",
                "--timed", "--vad", "--timeline_mode",
                *music_args, *context_args, *eq_args, *content_args, *preset_args, *stretch_args, *tempo_args, *pause_gap_args, *gain_args, *auto_gain_args, *pitch_args, *denoise_args, *vad_args, *tts_args, *trans_args, *whisper_prompt_args,
            ]
            hf_key = (
                (self.hf_api_key_var.get().strip() if hasattr(self, "hf_api_key_var") else self.settings.hf_api_key.strip())
                or os.environ.get("HF_TOKEN", "").strip()
                or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
            )
            run_env = None
            if trans_engine in ("chatgpt", "grok", "gemini") or hf_key:
                run_env = os.environ.copy()
            if trans_engine in ("chatgpt", "grok", "gemini"):
                run_env["API_TRANS_KEY"] = api_key
            if hf_key:
                run_env["HF_TOKEN"] = hf_key
                run_env["HUGGINGFACE_HUB_TOKEN"] = hf_key
            if self.runner.run(cmd, BASE_DIR.parent, base_progress=0, progress_range=20, on_line=pipeline_line_handler, extra_env=run_env) != 0:
                return False

            for i in range(0, 6):
                self._queue_step_done(i)

            src_path = Path(src)
            stem_name = src_path.stem
            output_suffix = "_lipsync.mp4" if self.local_lipsync.get() else "_tts.mp4"

            out_file = None
            if custom_out:
                p = Path(custom_out)
                if p.exists() and p.is_dir():
                    out_file = p / f"{stem_name}_{tgt}{output_suffix}"
                else:
                    out_file = p
                    if out_file.suffix.lower() != ".mp4":
                        out_file = out_file.with_suffix(".mp4")
            else:
                out_file = src_path.with_name(f"{stem_name}_{tgt}{output_suffix}")

            stem = Path(src).with_suffix("").as_posix()
            audio = Path(f"{stem}_{tgt}_tts.wav")           # speech only — always use for MuseTalk
            music_audio = Path(f"{stem}_{tgt}_tts_music.wav")  # music mix — added after lipsync

            if self.local_lipsync.get():
                self._queue_step_active(6)
                self._queue_progress(20)
                lipsync_ok = (self._run_musetalk(video_path=Path(src), audio_path=audio, base_progress=20, progress_range=78) == 0)
                final = self._finalize_musetalk_output(out_file, video_path=Path(src), audio_path=audio) if lipsync_ok else None

                if final:
                    # Always mux audio into lipsync output — MuseTalk may produce video-only
                    mux_src = music_audio if music_audio.exists() else audio
                    self._mux_music_into_lipsync(final, mux_src)
                    self._queue_log(f"[OK] Lipsync video hotové: {final}")
                    self._queue_step_done(6)
                    self._queue_step_active(7)
                    self._cleanup_temp_files(src)
                    self._queue_step_done(7)
                    self._queue_progress(100)
                    return True

                self._queue_log("[WARN] Lipsync zlyhal – robím len mux.")
                self._queue_step_done(6)

            # Mux fallback (use music mix if available, else speech only)
            self._queue_step_active(6)
            if Path(self.settings.ffmpeg).exists() or shutil.which(self.settings.ffmpeg):
                if not audio.exists():
                    self._queue_log(f"[ERROR] Audio súbor neexistuje: {audio}")
                else:
                    dest_mux = out_file.with_name(out_file.name.replace("_lipsync.mp4", "_tts.mp4")) if self.local_lipsync.get() else out_file
                    mux_audio = music_audio if music_audio.exists() else audio
                    mux_cmd = [
                        self.settings.ffmpeg, "-y", "-i", src, "-i", _path_str(mux_audio),
                        "-filter_complex", "[1:a]apad=pad_dur=120[a]",
                        "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest",
                        _path_str(dest_mux),
                    ]
                    rc = self.runner.run(mux_cmd, BASE_DIR)
                    if rc != 0:
                        self._queue_log(f"[ERROR] Mux zlyhal (exit {rc}): {dest_mux}")
                    else:
                        self._queue_log(f"[OK] Video vytvorené: {dest_mux}")
            else:
                self._queue_log("[ERROR] FFmpeg nenájdený — mux preskočený.")
            self._queue_step_done(6)
            self._queue_step_active(7)
            self._cleanup_temp_files(src)
            self._queue_step_done(7)
            self._queue_progress(100)
            return True

        def worker():
            # Wrap entire batch in a single start/stop pair so that
            # busy_count stays >= 1 throughout.  This prevents the log
            # from being cleared between videos and keeps the UI in the
            # "running" state for the whole batch.
            self._queue_start()
            self._queue_progress(0)
            total = len(self.batch_items)
            done_count = 0
            fail_count = 0

            # Initialize batch ETA tracking
            self.batch_start_ts = time.time()
            self.batch_total_videos = total
            self.batch_completed_videos = sum(1 for i in self.batch_items if i.status == "done")
            self.batch_video_times = []
            self.after(0, self._update_batch_progress)

            # Initialize per-task sidebar list (pyvideotrans-style)
            self._queue_task_list_init([item.path for item in self.batch_items])
            for i, item in enumerate(self.batch_items):
                if item.status == "done":
                    self._queue_task_status(i, "done")

            try:
                for idx, item in enumerate(self.batch_items):
                    if item.status == "done":
                        done_count += 1
                        continue

                    # Mark as processing
                    item.status = "processing"
                    self._queue_task_status(idx, "processing")
                    self.batch_current_index = idx
                    save_checkpoint(self.batch_items, idx, settings_snapshot)
                    self.after(0, self._update_batch_listbox)

                    video_start_time = time.time()
                    try:
                        success = process_single_video(item.path, idx, total)
                        video_duration = time.time() - video_start_time

                        if success:
                            item.status = "done"
                            self._queue_task_status(idx, "done")
                            done_count += 1
                            self.batch_completed_videos += 1
                            self.batch_video_times.append(video_duration)
                        else:
                            item.status = "failed"
                            self._queue_task_status(idx, "failed")
                            item.error = "Preklad/TTS zlyhalo"
                            fail_count += 1
                    except Exception as e:
                        item.status = "failed"
                        self._queue_task_status(idx, "failed")
                        item.error = str(e)
                        fail_count += 1
                        self._queue_log(f"[ERROR] {e}")

                    # Save checkpoint after each video
                    save_checkpoint(self.batch_items, idx + 1, settings_snapshot)
                    self.after(0, self._update_batch_listbox)
                    self.after(0, self._update_batch_progress)

                # Batch complete - show total time
                total_time = time.time() - self.batch_start_ts
                if total_time >= 3600:
                    time_h, time_rem = divmod(int(total_time), 3600)
                    time_m, time_s = divmod(time_rem, 60)
                    total_time_str = f"{time_h}h {time_m}m {time_s}s"
                else:
                    time_m, time_s = divmod(int(total_time), 60)
                    total_time_str = f"{time_m}m {time_s}s"

                self._queue_log(f"\n{'='*50}")
                self._queue_log(f"[BATCH] Dokončené: {done_count}/{total} | Zlyhalo: {fail_count}")
                self._queue_log(f"[BATCH] Celkový čas: {total_time_str}")
                self._queue_log(f"{'='*50}")

                # Reset batch tracking
                self.batch_start_ts = None
                self.batch_total_videos = 0
                self.batch_completed_videos = 0
                self.batch_video_times = []
                self.after(0, self._update_batch_progress)  # Hides batch bar

                if fail_count == 0:
                    clear_checkpoint()
                    self._notify_success(f"Batch dokončený: {done_count} videí")
                else:
                    self._notify_error(f"Batch: {fail_count} videí zlyhalo (checkpoint uložený)")

                self.after(0, self._update_resume_button)
            finally:
                self._queue_stop()

        threading.Thread(target=worker, daemon=True).start()

    # -------------------- Tab: API --------------------
    def api_tab(self, parent: ttk.Frame) -> None:
        tab = ttk.Frame(parent)
        tab.grid(row=0, column=0, sticky="nsew")
        self._pages["api"] = tab
        tab.columnconfigure(1, weight=1)

        self.openai_api_key_var = tk.StringVar(value=self.settings.openai_api_key)
        self.openai_translate_model_var = tk.StringVar(value=self.settings.openai_translate_model)
        self.openai_tts_model_var = tk.StringVar(value=self.settings.openai_tts_model)
        self.openai_tts_voice_var = tk.StringVar(value=self.settings.openai_tts_voice)
        self.openai_tts_base_url_var = tk.StringVar(value=self.settings.openai_tts_base_url)
        self.grok_api_key_var = tk.StringVar(value=self.settings.grok_api_key)
        self.grok_translate_model_var = tk.StringVar(value=self.settings.grok_translate_model)
        self.grok_tts_model_var = tk.StringVar(value=self.settings.grok_tts_model)
        self.grok_tts_voice_var = tk.StringVar(value=self.settings.grok_tts_voice)
        self.grok_tts_base_url_var = tk.StringVar(value=self.settings.grok_tts_base_url)
        self.gemini_api_key_var = tk.StringVar(value=self.settings.gemini_api_key)
        self.gemini_translate_model_var = tk.StringVar(value=self.settings.gemini_translate_model)
        self.gemini_tts_model_var = tk.StringVar(value=self.settings.gemini_tts_model)
        self.gemini_tts_voice_var = tk.StringVar(value=self.settings.gemini_tts_voice)
        self.gemini_tts_base_url_var = tk.StringVar(value=self.settings.gemini_tts_base_url)
        self.hf_api_key_var = tk.StringVar(value=self.settings.hf_api_key)

        ttk.Label(tab, text="ChatGPT API Key").grid(row=0, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.openai_api_key_var, show="*", width=70).grid(row=0, column=1, sticky="we", padx=6, pady=6)

        ttk.Label(tab, text="ChatGPT model (preklad)").grid(row=1, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.openai_translate_model_var, width=40).grid(row=1, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="ChatGPT model (TTS, voliteľné)").grid(row=2, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.openai_tts_model_var, width=40).grid(row=2, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="ChatGPT voice (TTS, voliteľné)").grid(row=3, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.openai_tts_voice_var, width=20).grid(row=3, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="ChatGPT Base URL").grid(row=4, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.openai_tts_base_url_var, width=60).grid(row=4, column=1, sticky="we", padx=6, pady=6)

        ttk.Separator(tab, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="we", padx=6, pady=6)

        ttk.Label(tab, text="Grok API Key").grid(row=6, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.grok_api_key_var, show="*", width=70).grid(row=6, column=1, sticky="we", padx=6, pady=6)

        ttk.Label(tab, text="Grok model (preklad)").grid(row=7, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.grok_translate_model_var, width=40).grid(row=7, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="Grok model (TTS, voliteľné)").grid(row=8, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.grok_tts_model_var, width=40).grid(row=8, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="Grok voice (TTS, voliteľné)").grid(row=9, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.grok_tts_voice_var, width=20).grid(row=9, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="Grok Base URL").grid(row=10, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.grok_tts_base_url_var, width=60).grid(row=10, column=1, sticky="we", padx=6, pady=6)

        ttk.Separator(tab, orient="horizontal").grid(row=11, column=0, columnspan=2, sticky="we", padx=6, pady=6)

        ttk.Label(tab, text="Gemini API Key").grid(row=12, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.gemini_api_key_var, show="*", width=70).grid(row=12, column=1, sticky="we", padx=6, pady=6)

        ttk.Label(tab, text="Gemini model (preklad)").grid(row=13, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.gemini_translate_model_var, width=40).grid(row=13, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="Gemini model (TTS, voliteľné)").grid(row=14, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.gemini_tts_model_var, width=40).grid(row=14, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="Gemini voice (TTS, voliteľné)").grid(row=15, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.gemini_tts_voice_var, width=20).grid(row=15, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(tab, text="Gemini Base URL").grid(row=16, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.gemini_tts_base_url_var, width=60).grid(row=16, column=1, sticky="we", padx=6, pady=6)

        ttk.Separator(tab, orient="horizontal").grid(row=17, column=0, columnspan=2, sticky="we", padx=6, pady=6)

        ttk.Label(tab, text="Hugging Face API Key").grid(row=18, column=0, sticky="w", pady=6, padx=6)
        ttk.Entry(tab, textvariable=self.hf_api_key_var, show="*", width=70).grid(row=18, column=1, sticky="we", padx=6, pady=6)

        ttk.Label(
            tab,
            text="Tieto hodnoty sa používajú hlavne pre prekladové enginy ChatGPT/Grok/Gemini a HF autorizáciu.",
            foreground="gray",
        ).grid(row=16, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 8))

        ttk.Button(tab, text="Uložiť API nastavenia", command=self._save_settings).grid(row=17, column=0, columnspan=2, pady=8)

    # -------------------- Tab: Nástroje --------------------
    def tools_tab(self, parent: ttk.Frame) -> None:
        tab = ttk.Frame(parent)
        tab.grid(row=0, column=0, sticky="nsew")
        self._pages["tools"] = tab

        # ONNX export
        onnx_frame = ttk.LabelFrame(tab, text="Export modelov (ONNX/TensorRT)")
        onnx_frame.grid(row=0, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        ttk.Radiobutton(onnx_frame, text="ONNX", value="onnx", variable=self.export_type).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(onnx_frame, text="ONNX + TensorRT", value="tensorrt", variable=self.export_type).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(onnx_frame, text="Všetky", value="all", variable=self.export_models).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(onnx_frame, text="Len UNet", value="unet", variable=self.export_models).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(onnx_frame, text="Len VAE", value="vae", variable=self.export_models).grid(row=1, column=2, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(onnx_frame, text="Len BiSeNet", value="bisenet", variable=self.export_models).grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(onnx_frame, text="Len Whisper", value="whisper", variable=self.export_models).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(onnx_frame, text="FP16", variable=self.export_fp16).grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Button(onnx_frame, text="Spusti export", command=self._run_export_onnx).grid(row=4, column=0, columnspan=3, pady=6)

        # FPS convert
        fps_frame = ttk.LabelFrame(tab, text="Konverzia na 25 fps (libx264)")
        fps_frame.grid(row=1, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        ttk.Label(fps_frame, text="Zdrojové video").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(fps_frame, textvariable=self.fps25_src, width=60).grid(row=0, column=1, padx=5, pady=2, sticky="we")
        ttk.Button(fps_frame, text="Vybrať…", command=self._pick_fps_src).grid(row=0, column=2, padx=4, pady=2)
        ttk.Label(fps_frame, text="Výstup 25fps").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(fps_frame, textvariable=self.fps25_out, width=60).grid(row=1, column=1, padx=5, pady=2, sticky="we")
        ttk.Button(fps_frame, text="Vybrať…", command=self._pick_fps_out).grid(row=1, column=2, padx=4, pady=2)
        ttk.Button(fps_frame, text="Konvertovať na 25 fps", command=self._run_fps25).grid(row=2, column=0, columnspan=3, pady=6)

        # Video Split
        split_frame = ttk.LabelFrame(tab, text="Rozdelenie videa podľa minút")
        split_frame.grid(row=2, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        ttk.Label(split_frame, text="Zdrojové video").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(split_frame, textvariable=self.split_src, width=60).grid(row=0, column=1, padx=5, pady=2, sticky="we")
        ttk.Button(split_frame, text="Vybrať…", command=self._pick_split_src).grid(row=0, column=2, padx=4, pady=2)
        ttk.Label(split_frame, text="Dĺžka časti (min)").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(split_frame, textvariable=self.split_minutes, from_=0.1, to=999, increment=0.5, width=10).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.split_btn = ttk.Button(split_frame, text="Rozdeliť", command=self._run_split_video)
        self.split_btn.grid(row=2, column=0, columnspan=3, pady=6)

        # Video Join
        join_frame = ttk.LabelFrame(tab, text="Spojenie videí (po preklade)")
        join_frame.grid(row=3, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        join_frame.columnconfigure(1, weight=1)
        ttk.Label(join_frame, text="Adresár s časťami").grid(row=0, column=0, sticky="w", pady=4, padx=4)
        ttk.Entry(join_frame, textvariable=self.join_dir, width=60).grid(row=0, column=1, padx=5, pady=2, sticky="we")
        ttk.Button(join_frame, text="Vybrať…", command=self._pick_join_dir).grid(row=0, column=2, padx=4, pady=2)
        ttk.Label(join_frame, text="Vyberte adresár s preloženými časťami videa (napr. *_split). Súbory sa spoja podľa abecedy.", wraplength=500, foreground="gray").grid(row=1, column=0, columnspan=3, sticky="w", padx=4)
        self.join_btn = ttk.Button(join_frame, text="Spojiť", command=self._run_join_video)
        self.join_btn.grid(row=2, column=0, columnspan=3, pady=6)

        # Environment Fix
        env_frame = ttk.LabelFrame(tab, text="Oprava prostredia (Torch/Torchvision)")
        env_frame.grid(row=4, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        
        msg = "Ak MuseTalk padá na chybe 'operator torchvision::nms does not exist', spustite túto opravu.\n" \
              "Toto preinštaluje torch a torchvision na kompatibilné verzie."
        ttk.Label(env_frame, text=msg, wraplength=500, justify="left").pack(anchor="w", padx=5, pady=2)
        
        btn_frame = ttk.Frame(env_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="Spustiť opravu (FIX_TORCH_FINAL.bat)", command=self._run_fix_torch).pack(side="left", padx=(0, 5))
        ttk.Button(btn_frame, text="Otestovať funkčnosť (NMS)", command=self._test_torch_env).pack(side="left")

    # -------------------- Tab: EQ --------------------
    def eq_tab(self, parent: ttk.Frame) -> None:
        tab = ttk.Frame(parent)
        tab.grid(row=0, column=0, sticky="nsew")
        self._pages["eq"] = tab
        tab.columnconfigure(1, weight=1)

        self.eq_input_var = tk.StringVar(value=str(BASE_DIR / "EQ" / "samples" / "voice_sample_raw.wav"))
        self.eq_output_var = tk.StringVar(value=str(BASE_DIR / "EQ" / "samples" / "voice_sample_eq.wav"))
        self.eq_preset_var = tk.StringVar(value=str(BASE_DIR / "EQ" / "eq_preset.json"))
        self.eq_sample_source_var = tk.StringVar(value=str(BASE_DIR / "MuseTalk" / "data" / "audio" / "eng.wav"))
        self.eq_sample_duration_var = tk.StringVar(value="18")
        self.eq_builtin_preset_var = tk.StringVar(value="Default (Voice EQ)")

        ttk.Label(tab, text="Zdroj audio").grid(row=0, column=0, sticky="w", pady=4, padx=4)
        ttk.Entry(tab, textvariable=self.eq_input_var).grid(row=0, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(tab, text="Vybrať…", command=self._pick_eq_input).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(tab, text="Výstup audio").grid(row=1, column=0, sticky="w", pady=4, padx=4)
        ttk.Entry(tab, textvariable=self.eq_output_var).grid(row=1, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(tab, text="Vybrať…", command=self._pick_eq_output).grid(row=1, column=2, padx=4, pady=4)

        ttk.Label(tab, text="EQ preset JSON").grid(row=2, column=0, sticky="w", pady=4, padx=4)
        ttk.Entry(tab, textvariable=self.eq_preset_var).grid(row=2, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(tab, text="Vybrať…", command=self._pick_eq_preset).grid(row=2, column=2, padx=4, pady=4)

        ttk.Label(tab, text="Prednastavenie EQ").grid(row=3, column=0, sticky="w", pady=4, padx=4)
        self.eq_preset_combo = ttk.Combobox(
            tab,
            textvariable=self.eq_builtin_preset_var,
            values=list(EQ_BUILTIN_PRESETS.keys()),
            state="readonly",
        )
        self.eq_preset_combo.grid(row=3, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(tab, text="Použiť preset", command=self._apply_eq_builtin_preset).grid(row=3, column=2, padx=4, pady=4)

        filter_frame = ttk.LabelFrame(tab, text="Filtre (1 filter na riadok)")
        filter_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
        filter_frame.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)

        self.eq_filters_text = tk.Text(filter_frame, height=10, wrap="word")
        self.eq_filters_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.eq_filters_text.configure(font=(self.font_mono_family, 10))
        self._load_eq_preset_to_ui()

        row_btn = ttk.Frame(filter_frame)
        row_btn.grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Button(row_btn, text="Načítať preset", command=self._load_eq_preset_to_ui).pack(side="left", padx=(0, 4))
        ttk.Button(row_btn, text="Uložiť preset", command=self._save_eq_preset_from_ui).pack(side="left", padx=4)

        sample_frame = ttk.LabelFrame(tab, text="Generovanie test vzorky")
        sample_frame.grid(row=5, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        sample_frame.columnconfigure(1, weight=1)

        ttk.Label(sample_frame, text="Sample source").grid(row=0, column=0, sticky="w", pady=4, padx=4)
        ttk.Entry(sample_frame, textvariable=self.eq_sample_source_var).grid(row=0, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(sample_frame, text="Vybrať…", command=self._pick_eq_sample_source).grid(row=0, column=2, padx=4, pady=4)
        ttk.Label(sample_frame, text="Trvanie (s)").grid(row=1, column=0, sticky="w", pady=4, padx=4)
        ttk.Entry(sample_frame, textvariable=self.eq_sample_duration_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(sample_frame, text="Vygenerovať sample", command=self._run_eq_generate_sample).grid(row=1, column=2, padx=4, pady=4)

        action_row = ttk.Frame(tab)
        action_row.grid(row=6, column=0, columnspan=3, sticky="w", padx=6, pady=6)
        ttk.Button(action_row, text="Aplikovať EQ", command=self._run_eq_apply).pack(side="left")
        ttk.Button(action_row, text="Prehrať vstup", command=lambda: self._run_eq_play(self.eq_input_var.get())).pack(side="left", padx=(8, 0))
        ttk.Button(action_row, text="Prehrať výstup", command=lambda: self._run_eq_play(self.eq_output_var.get())).pack(side="left", padx=(4, 0))

    def _pick_eq_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Vyber zdrojové audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("All files", "*.*")],
        )
        if path:
            self.eq_input_var.set(path)
            if not self.eq_output_var.get().strip():
                p = Path(path)
                self.eq_output_var.set(str(p.with_name(f"{p.stem}_eq.wav")))

    def _pick_eq_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Vyber výstup EQ",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.eq_output_var.set(path)

    def _pick_eq_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Vyber EQ preset JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.eq_preset_var.set(path)
            self._load_eq_preset_to_ui()

    def _pick_eq_sample_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Vyber source pre test sample",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("All files", "*.*")],
        )
        if path:
            self.eq_sample_source_var.set(path)

    def _apply_eq_builtin_preset(self) -> None:
        preset_name = self.eq_builtin_preset_var.get().strip()
        filters = EQ_BUILTIN_PRESETS.get(preset_name)
        if not filters:
            return
        self.eq_filters_text.delete("1.0", "end")
        self.eq_filters_text.insert("1.0", "\n".join(filters))
        self._queue_log(f"[EQ] Použitý preset: {preset_name}")

    def _load_eq_preset_to_ui(self) -> None:
        preset_path = Path(self.eq_preset_var.get().strip())
        if not preset_path.exists():
            return
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            filters = data.get("filters", [])
            self.eq_filters_text.delete("1.0", "end")
            self.eq_filters_text.insert("1.0", "\n".join(filters))
        except Exception as e:
            self._queue_log(f"[EQ] Nepodarilo sa načítať preset: {e}")

    def _save_eq_preset_from_ui(self) -> None:
        preset_path = Path(self.eq_preset_var.get().strip())
        if not preset_path:
            messagebox.showerror("EQ", "Vyber preset JSON.")
            return
        try:
            filters_raw = self.eq_filters_text.get("1.0", "end").splitlines()
            filters = [line.strip() for line in filters_raw if line.strip()]
            data = {
                "name": "voice_eq_custom",
                "description": "EQ preset edited in GUI",
                "filters": filters,
                "output_codec": "pcm_s16le",
            }
            preset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._queue_log(f"[EQ] Preset uložený: {preset_path}")
        except Exception as e:
            messagebox.showerror("EQ", f"Nepodarilo sa uložiť preset:\n{e}")

    def _run_eq_generate_sample(self) -> None:
        if not self.settings.py.exists():
            messagebox.showerror("Python nenájdený", str(self.settings.py))
            return
        source = Path(self.eq_sample_source_var.get().strip())
        if not source.exists():
            messagebox.showerror("EQ", f"Source neexistuje:\n{source}")
            return
        out = BASE_DIR / "EQ" / "samples" / "voice_sample_raw.wav"
        tts_script = BASE_DIR / "pipeline.py"
        if not tts_script.exists():
            messagebox.showerror("EQ", f"Skript chýba:\n{tts_script}")
            return
        duration = self.eq_sample_duration_var.get().strip() or "18"  # kept for compatibility/UI
        _ = duration

        text_path = BASE_DIR / "work" / "eq_sample_text_cs.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        sample_text_cs = (
            "Ahoj, toto je testovacia česká vzorka hlasu pre ekvalizér, teda ekvalizer. "
            "Zachovávam prirodzenú artikuláciu, sykavky aj výšky, aby bolo počuť rozdiel po úprave EQ. "
            "Kúl technologie a kvalitní zvuk jsou důležité."
        )
        text_path.write_text(sample_text_cs, encoding="utf-8")

        generated = source.with_suffix("").with_name(f"{source.stem}_cs_tts.wav")
        cmd = [
            _path_str(self.settings.py),
            _path_str(tts_script),
            "--input",
            _path_str(source),
            "--tgt_lang",
            "cs",
            "--text_in",
            _path_str(text_path),
            "--use_chatterbox",
            "--chatterbox_model",
            self._get_selected_chatterbox_model_path(),
            "--no-auto-speech-gain",
            "--speech_gain",
            "1.0",
        ]

        def worker():
            self._queue_progress(0)
            ret = self.runner.run(cmd, BASE_DIR.parent)
            if ret == 0:
                if generated.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(generated, out)
                else:
                    self._queue_log(f"[EQ] Upozornenie: očakávaný výstup neexistuje: {generated}")
                self.eq_input_var.set(str(out))
                if not self.eq_output_var.get().strip():
                    self.eq_output_var.set(str(BASE_DIR / "EQ" / "samples" / "voice_sample_eq.wav"))
                self._queue_progress(100)
                self._queue_log(f"[EQ] Český Chatterbox sample vygenerovaný: {out}")

        threading.Thread(target=worker, daemon=True).start()

    def _run_eq_apply(self) -> None:
        if not self.settings.py.exists():
            messagebox.showerror("Python nenájdený", str(self.settings.py))
            return
        self._save_eq_preset_from_ui()
        inp = Path(self.eq_input_var.get().strip())
        if not inp.exists():
            messagebox.showerror("EQ", f"Zdroj neexistuje:\n{inp}")
            return
        out_txt = self.eq_output_var.get().strip()
        out = Path(out_txt) if out_txt else inp.with_name(f"{inp.stem}_eq.wav")
        self.eq_output_var.set(str(out))

        preset = Path(self.eq_preset_var.get().strip())
        script = BASE_DIR / "EQ" / "apply_eq.py"
        if not script.exists():
            messagebox.showerror("EQ", f"Skript chýba:\n{script}")
            return
        ffmpeg_bin = self.settings.ffmpeg.strip() if self.settings.ffmpeg.strip() else "ffmpeg"
        cmd = [
            _path_str(self.settings.py),
            _path_str(script),
            "--input",
            _path_str(inp),
            "--output",
            _path_str(out),
            "--preset",
            _path_str(preset),
            "--ffmpeg",
            ffmpeg_bin,
        ]

        def worker():
            self._queue_progress(0)
            ret = self.runner.run(cmd, BASE_DIR)
            if ret == 0:
                self._queue_progress(100)
                self._queue_log(f"[EQ] Hotovo: {out}")

        threading.Thread(target=worker, daemon=True).start()

    def _resolve_ffplay_bin(self) -> str:
        ffmpeg_cfg = (self.settings.ffmpeg or "").strip()
        if ffmpeg_cfg:
            ffmpeg_path = Path(ffmpeg_cfg)
            if ffmpeg_path.is_file():
                suffix = ffmpeg_path.suffix.lower()
                if suffix == ".exe":
                    candidate = ffmpeg_path.with_name("ffplay.exe")
                else:
                    candidate = ffmpeg_path.with_name("ffplay")
                if candidate.exists():
                    return str(candidate)
        return "ffplay"

    def _run_eq_play(self, audio_path_txt: str) -> None:
        audio_path = Path((audio_path_txt or "").strip())
        if not audio_path_txt or not audio_path.exists():
            messagebox.showerror("EQ", f"Audio neexistuje:\n{audio_path}")
            return
        ffplay_bin = self._resolve_ffplay_bin()
        if not (Path(ffplay_bin).exists() or shutil.which(ffplay_bin)):
            messagebox.showerror("EQ", "ffplay nebol nájdený. Nainštaluj FFmpeg/ffplay alebo nastav cestu k ffmpeg v Nastaveniach.")
            return

        cmd = [ffplay_bin, "-nodisp", "-autoexit", _path_str(audio_path)]

        def worker():
            self._queue_log(f"[EQ] Prehrávam: {audio_path.name}")
            self.runner.run(cmd, BASE_DIR)

        threading.Thread(target=worker, daemon=True).start()

    # -------------------- Tab: Settings --------------------
    def settings_tab(self, parent: ttk.Frame) -> None:
        tab = ttk.Frame(parent)
        tab.grid(row=0, column=0, sticky="nsew")
        self._pages["settings"] = tab

        self.py_var = tk.StringVar(value=_path_str(self.settings.py))
        self.ffmpeg_var = tk.StringVar(value=self.settings.ffmpeg)

        ttk.Label(tab, text="Python exe").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=self.py_var, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Vybrať…", command=self._pick_py).grid(row=0, column=2, padx=5)

        ttk.Label(tab, text="FFmpeg (ffmpeg.exe alebo PATH)").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=self.ffmpeg_var, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Vybrať…", command=self._pick_ffmpeg).grid(row=1, column=2, padx=5)

        ttk.Checkbutton(tab, text="Zabrániť uspatiu počas spracovania", variable=self.keep_awake).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=6
        )

        ttk.Checkbutton(tab, text="Tmavý režim", variable=self.dark_mode, command=self._toggle_dark_mode).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=6
        )

        # Engine defaults
        engine_frame = ttk.LabelFrame(tab, text="Predvolené enginy")
        engine_frame.grid(row=4, column=0, columnspan=3, sticky="we", pady=6, padx=2)

        self.def_tts_var = tk.StringVar(value=self.settings.def_tts_engine)
        ttk.Label(engine_frame, text="TTS engine:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        for i, (txt, val) in enumerate((("Chatterbox", "chatterbox"), ("XTTS v2", "xtts"), ("Edge-TTS", "edge"))):
            ttk.Radiobutton(engine_frame, text=txt, value=val, variable=self.def_tts_var).grid(row=0, column=1 + i, sticky="w", padx=4, pady=2)

        self.def_trans_var = tk.StringVar(value=self.settings.def_trans_engine)
        ttk.Label(engine_frame, text="Prekladový engine:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        for i, (txt, val) in enumerate((("Llama", "llama"), ("MADLAD", "madlad"), ("MultiSlav", "multislav5lang"), ("Hybrid", "hybrid"), ("ChatGPT", "chatgpt"), ("Grok", "grok"), ("Gemini", "gemini"), ("Google", "google"))):
            ttk.Radiobutton(engine_frame, text=txt, value=val, variable=self.def_trans_var).grid(row=1, column=1 + i, sticky="w", padx=4, pady=2)

        # Toggle defaults
        defaults_frame = ttk.LabelFrame(tab, text="Predvolené prepínače")
        defaults_frame.grid(row=5, column=0, columnspan=3, sticky="we", pady=6, padx=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: lipsync", variable=self.def_local_lipsync, command=self._apply_defaults).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: hudba", variable=self.def_local_keep_music, command=self._apply_defaults).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: kontext", variable=self.def_local_context, command=self._apply_defaults).grid(row=0, column=2, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: voice EQ", variable=self.def_local_voice_eq, command=self._apply_defaults).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: multi-voice", variable=self.def_local_multi_voice, command=self._apply_defaults).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: voice clone", variable=self.def_local_clone_voice, command=self._apply_defaults).grid(row=1, column=2, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: denoise", variable=self.def_local_denoise, command=self._apply_defaults).grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: auto gain", variable=self.def_local_auto_speech_gain, command=self._apply_defaults).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(defaults_frame, text="Lokálny: emotion clone", variable=self.def_local_emotion_clone, command=self._apply_defaults).grid(row=2, column=2, sticky="w", padx=4, pady=2)
        ttk.Button(defaults_frame, text="Použiť tieto defaulty na formuláre", command=self._apply_defaults).grid(row=3, column=0, columnspan=3, pady=6)

        # Timing/stretch + Volume settings
        adv_frame = ttk.LabelFrame(tab, text="Časovanie / Hlasitosť")
        adv_frame.grid(row=6, column=0, columnspan=3, sticky="we", pady=6, padx=2)

        row1 = ttk.Frame(adv_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Stretch min:").pack(side="left", padx=(5, 2))
        ttk.Entry(row1, textvariable=self.local_stretch_min, width=5).pack(side="left", padx=2)
        ttk.Label(row1, text="max:").pack(side="left", padx=(10, 2))
        ttk.Entry(row1, textvariable=self.local_stretch_max, width=5).pack(side="left", padx=2)
        ttk.Label(row1, text="Base tempo:").pack(side="left", padx=(10, 2))
        ttk.Entry(row1, textvariable=self.local_base_tempo, width=5).pack(side="left", padx=2)
        ttk.Label(row1, text="(0.70-1.60)", foreground="gray").pack(side="left", padx=5)

        row1b = ttk.Frame(adv_frame)
        row1b.pack(fill="x", pady=2)
        ttk.Label(row1b, text="Max pauza (s):").pack(side="left", padx=(5, 2))
        ttk.Entry(row1b, textvariable=self.local_pause_gap_s, width=5).pack(side="left", padx=2)
        ttk.Label(row1b, text="(min pauza medzi segmentmi; väčšia = lepší sync, menšia = kratšie ticho)", foreground="gray").pack(side="left", padx=5)

        row2 = ttk.Frame(adv_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Hlasitosť reči:").pack(side="left", padx=(5, 2))
        ttk.Entry(row2, textvariable=self.local_speech_gain, width=5).pack(side="left", padx=2)
        ttk.Label(row2, text="Hlasitosť hudby:").pack(side="left", padx=(10, 2))
        ttk.Entry(row2, textvariable=self.local_music_volume, width=5).pack(side="left", padx=2)
        ttk.Label(row2, text="(0.15=jemná, 0.25=stredná, 0.50=výrazná, max 1.0)", foreground="gray").pack(side="left", padx=5)

        row3 = ttk.Frame(adv_frame)
        row3.pack(fill="x", pady=2)
        _cb_denoise_s = ttk.Checkbutton(row3, text="Odšumenie (DENOISE)", variable=self.local_denoise)
        _cb_denoise_s.pack(side="left", padx=(5, 6))
        tip(_cb_denoise_s, "Odstráni šum zo zdrojového zvuku pred STT. Pomáha pri šumnom mikrofóne alebo komprimovanom videu.")
        ttk.Label(row3, text="Preset:").pack(side="left", padx=(0, 2))
        ttk.Combobox(row3, textvariable=self.local_denoise_preset, values=["mild", "strong"], width=8, state="readonly").pack(side="left", padx=2)
        ttk.Label(row3, text="(mild = menej artefaktov, strong = silnejšie čistenie)", foreground="gray").pack(side="left", padx=5)

        # Max chars settings
        chars_frame = ttk.LabelFrame(tab, text="Veľkosť textových blokov")
        chars_frame.grid(row=7, column=0, columnspan=3, sticky="we", pady=6, padx=2)

        chars_row = ttk.Frame(chars_frame)
        chars_row.pack(fill="x", pady=2)
        ttk.Label(chars_row, text="max_chars:").pack(side="left", padx=(5, 2))
        ttk.Entry(chars_row, textvariable=self.local_max_chars, width=6).pack(side="left", padx=2)
        ttk.Label(chars_frame, text="Max znakov na kúsok textu pre preklad / TTS", foreground="gray").pack(anchor="w", padx=5, pady=(0, 2))

        # VAD settings
        vad_frame = ttk.LabelFrame(tab, text="VAD segmentácia (počet častí audia)")
        vad_frame.grid(row=8, column=0, columnspan=3, sticky="we", pady=6, padx=2)

        vad_row = ttk.Frame(vad_frame)
        vad_row.pack(fill="x", pady=2)
        ttk.Label(vad_row, text="Min reč (ms):").pack(side="left", padx=(5, 2))
        ttk.Entry(vad_row, textvariable=self.local_vad_min_speech, width=6).pack(side="left", padx=2)
        ttk.Label(vad_row, text="Min ticho (ms):").pack(side="left", padx=(10, 2))
        ttk.Entry(vad_row, textvariable=self.local_vad_min_silence, width=6).pack(side="left", padx=2)
        ttk.Label(vad_row, text="Max segment (s):").pack(side="left", padx=(10, 2))
        ttk.Entry(vad_row, textvariable=self.local_vad_max_speech, width=6).pack(side="left", padx=2)
        ttk.Label(vad_frame, text="Vyššie hodnoty min reč/ticho = menej segmentov = rýchlejšie spracovanie", foreground="gray").pack(anchor="w", padx=5, pady=(0, 2))

        # Whisper hint
        whisper_frame = ttk.LabelFrame(tab, text="Whisper nápoveda (terminológia)")
        whisper_frame.grid(row=9, column=0, columnspan=3, sticky="we", pady=6, padx=2)
        w_row = ttk.Frame(whisper_frame)
        w_row.pack(fill="x", pady=2)
        ttk.Label(w_row, text="Hint:").pack(side="left", padx=(5, 2))
        ttk.Entry(w_row, textvariable=self.local_whisper_prompt, width=55).pack(side="left", padx=2, fill="x", expand=True)
        ttk.Label(whisper_frame, text='Napr. "SQL, COALESCE, ISNULL, NULLIF, Python, numpy" — pomáha Whisperu správne rozpoznať odborné slová', foreground="gray").pack(anchor="w", padx=5, pady=(0, 2))

        # Window / Log size
        ui_frame = ttk.LabelFrame(tab, text="Veľkosť okna")
        ui_frame.grid(row=10, column=0, columnspan=3, sticky="we", pady=6, padx=2)

        self.win_width_var = tk.StringVar(value=str(self.settings.def_window_width))
        self.win_height_var = tk.StringVar(value=str(self.settings.def_window_height))
        self.log_height_var = tk.StringVar(value=str(self.settings.def_log_height))

        ui_row1 = ttk.Frame(ui_frame)
        ui_row1.pack(fill="x", pady=2)
        ttk.Label(ui_row1, text="Šírka:").pack(side="left", padx=(5, 2))
        ttk.Entry(ui_row1, textvariable=self.win_width_var, width=6).pack(side="left", padx=2)
        ttk.Label(ui_row1, text="Výška:").pack(side="left", padx=(10, 2))
        ttk.Entry(ui_row1, textvariable=self.win_height_var, width=6).pack(side="left", padx=2)
        ttk.Label(ui_row1, text="Log (riadky):").pack(side="left", padx=(10, 2))
        ttk.Entry(ui_row1, textvariable=self.log_height_var, width=5).pack(side="left", padx=2)

        ttk.Button(tab, text="Uložiť nastavenia", command=self._save_settings).grid(row=11, column=0, columnspan=3, pady=8)

    def _pick_py(self) -> None:
        path = filedialog.askopenfilename(title="Vyber python.exe")
        if path:
            self.py_var.set(path)

    def _pick_ffmpeg(self) -> None:
        path = filedialog.askopenfilename(title="Vyber ffmpeg.exe")
        if path:
            self.ffmpeg_var.set(path)

    def _save_settings(self) -> None:
        py_path = Path(self.py_var.get().strip())
        if not py_path.exists():
            messagebox.showerror("Python nenájdený", py_path)
            return
        self.settings.py = py_path
        ffmpeg = self.ffmpeg_var.get().strip() or "ffmpeg"
        self.settings.ffmpeg = ffmpeg
        self.settings.def_tts_engine = self.def_tts_var.get() if self.def_tts_var.get() in ("chatterbox", "xtts", "edge") else "chatterbox"
        self.settings.def_trans_engine = self.def_trans_var.get()
        self.settings.def_clone_voice = self.def_local_clone_voice.get()
        self.settings.def_local_keep_music = self.def_local_keep_music.get()
        self.settings.def_denoise = self.def_local_denoise.get()
        self.settings.def_denoise_preset = self.local_denoise_preset.get() or "mild"
        self.settings.def_auto_speech_gain = self.def_local_auto_speech_gain.get()
        if hasattr(self, "openai_api_key_var"):
            self.settings.openai_api_key = ""
        if hasattr(self, "openai_translate_model_var"):
            self.settings.openai_translate_model = self.openai_translate_model_var.get().strip() or "gpt-5-mini-2025-08-07"
        if hasattr(self, "openai_tts_model_var"):
            self.settings.openai_tts_model = self.openai_tts_model_var.get().strip() or "gpt-4o-mini-tts"
        if hasattr(self, "openai_tts_voice_var"):
            self.settings.openai_tts_voice = self.openai_tts_voice_var.get().strip() or "alloy"
        if hasattr(self, "openai_tts_base_url_var"):
            self.settings.openai_tts_base_url = self.openai_tts_base_url_var.get().strip() or "https://api.openai.com/v1"
        if hasattr(self, "grok_api_key_var"):
            self.settings.grok_api_key = ""
        if hasattr(self, "grok_translate_model_var"):
            self.settings.grok_translate_model = self.grok_translate_model_var.get().strip() or "grok-3-mini"
        if hasattr(self, "grok_tts_model_var"):
            self.settings.grok_tts_model = self.grok_tts_model_var.get().strip() or "grok-3-mini"
        if hasattr(self, "grok_tts_voice_var"):
            self.settings.grok_tts_voice = self.grok_tts_voice_var.get().strip() or "alloy"
        if hasattr(self, "grok_tts_base_url_var"):
            self.settings.grok_tts_base_url = self.grok_tts_base_url_var.get().strip() or "https://api.x.ai/v1"
        if hasattr(self, "gemini_api_key_var"):
            self.settings.gemini_api_key = ""
        if hasattr(self, "gemini_translate_model_var"):
            self.settings.gemini_translate_model = self.gemini_translate_model_var.get().strip() or "gemini-2.5-flash"
        if hasattr(self, "gemini_tts_model_var"):
            self.settings.gemini_tts_model = self.gemini_tts_model_var.get().strip() or "gemini-2.5-flash"
        if hasattr(self, "gemini_tts_voice_var"):
            self.settings.gemini_tts_voice = self.gemini_tts_voice_var.get().strip() or "alloy"
        if hasattr(self, "gemini_tts_base_url_var"):
            self.settings.gemini_tts_base_url = self.gemini_tts_base_url_var.get().strip() or "https://generativelanguage.googleapis.com/v1beta/openai"
        if hasattr(self, "hf_api_key_var"):
            self.settings.hf_api_key = ""
        if hasattr(self, "chatterbox_model_var"):
            self.settings.def_chatterbox_model = self._normalize_chatterbox_model_name(self.chatterbox_model_var.get())
        self.settings.def_dark_mode = self.dark_mode.get()
        self.settings.def_speech_gain = self.local_speech_gain.get()
        self.settings.def_music_volume = self.local_music_volume.get()
        self.settings.def_base_tempo = self.local_base_tempo.get() or "1.06"
        self.settings.def_stretch_min = self.local_stretch_min.get()
        self.settings.def_stretch_max = self.local_stretch_max.get()
        self.settings.def_pause_gap_s = self.local_pause_gap_s.get() or "0.70"
        self.settings.def_content_type = self.local_content_type.get()
        self.settings.def_batch_qa = self.local_use_madlad_gemma_qa.get()
        self.settings.def_emotion_clone = self.local_emotion_clone.get()
        self.settings.def_vad_min_speech_ms = self.local_vad_min_speech.get()
        self.settings.def_vad_min_silence_ms = self.local_vad_min_silence.get()
        self.settings.def_vad_max_speech_s = self.local_vad_max_speech.get()
        try:
            self.settings.def_log_height = int(self.log_height_var.get())
        except ValueError:
            pass
        try:
            self.settings.def_window_width = int(self.win_width_var.get())
            self.settings.def_window_height = int(self.win_height_var.get())
        except ValueError:
            pass
        self.settings.save()
        # Apply live
        self.log_text.configure(height=self.settings.def_log_height)
        self.geometry(f"{self.settings.def_window_width}x{self.settings.def_window_height}")
        self._apply_theme(self.dark_mode.get())

    def _toggle_dark_mode(self) -> None:
        self.settings.def_dark_mode = self.dark_mode.get()
        self.settings.save()
        self._apply_theme(self.dark_mode.get())

    def _clear_log(self) -> None:
        """Clear the log text widget."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _open_output_dir(self) -> None:
        """Open the default output folder in file manager."""
        folder = BASE_DIR / "results"
        folder.mkdir(exist_ok=True)
        try:
            if platform.system() == "Windows":
                os.startfile(str(folder))
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as e:
            messagebox.showerror("Chyba", f"Nepodarilo sa otvoriť priečinok:\n{e}")

    def _run_fix_torch(self):
        bat_path = BASE_DIR / "bat" / "FIX_TORCH_FINAL.bat"
        if not bat_path.exists():
            messagebox.showerror("Chýba skript", f"Nenašiel sa: {bat_path}")
            return
        
        try:
            # Run in a new console window so user can see output and interact (pause)
            subprocess.Popen(["start", "cmd.exe", "/c", str(bat_path)], shell=True, cwd=str(BASE_DIR))
            self._queue_log("[INFO] Spustený opravný skript v novom okne. Postupujte podľa pokynov v okne.")
        except Exception as e:
            self._notify_error(f"Nepodarilo sa spustiť opravu: {e}")

    def _test_torch_env(self):
        try:
            import torch
            import torchvision
            # Simple NMS test
            boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]], dtype=torch.float32)
            scores = torch.tensor([1.0], dtype=torch.float32)
            # This will crash if torchvision is not compatible with torch
            torchvision.ops.nms(boxes, scores, 0.5)
            
            msg = (f"Všetko vyzerá v poriadku!\n\n"
                   f"Torch: {torch.__version__}\n"
                   f"Torchvision: {torchvision.__version__}\n"
                   f"CUDA dostupná: {torch.cuda.is_available()}\n"
                   f"NMS operátor: Funkčný")
            messagebox.showinfo("Test úspešný", msg)
        except Exception as e:
            messagebox.showerror("Test zlyhal", f"Chyba prostredia:\n{e}\n\nOdporúčam spustiť opravu znova.")

    # -------------------- MuseTalk helpers --------------------
    def _prepare_musetalk_inputs(self, video_path: Path, audio_path: Path) -> bool:
        """Prepare input files and create dynamic config for MuseTalk."""
        musetalk_dir = BASE_DIR / "MuseTalk"

        if not video_path.exists():
            self._queue_log(f"[ERROR] Input video neexistuje: {video_path}")
            return False
        if not audio_path.exists():
            self._queue_log(f"[ERROR] Input audio neexistuje: {audio_path}")
            return False

        # Create dynamic config using absolute paths
        config_dir = musetalk_dir / "configs" / "inference"
        config_dir.mkdir(parents=True, exist_ok=True)
        dynamic_config = config_dir / "dynamic.yaml"

        # Use absolute paths, formatted with forward slashes for yaml compatibility
        abs_video_path = video_path.resolve().as_posix()
        abs_audio_path = audio_path.resolve().as_posix()

        config_content = f"""task_0:
  video_path: "{abs_video_path}"
  audio_path: "{abs_audio_path}"
"""
        self._queue_log(f"[INFO] Obsah dynamic.yaml:\n---\n{config_content}\n---")
        dynamic_config.write_text(config_content, encoding="utf-8")
        self._queue_log(f"[OK] Dynamický config vytvorený: {dynamic_config}")

        return True

    def _run_musetalk(self, video_path: Path, audio_path: Path, base_progress: float = 0, progress_range: float = 100) -> int:
        """Run MuseTalk inference using the same defaults as MUSETALK_MENU.bat."""
        # Prepare inputs and dynamic config
        if not self._prepare_musetalk_inputs(video_path, audio_path):
            return 1

        ffmpeg_dir = self.settings.ffmpeg_dir()
        if not ffmpeg_dir:
            self._queue_log("[ERROR] FFmpeg cesta nie je nastavená.")
            return 1
        self._queue_log("[INFO] Inicializujem MuseTalk (načítavanie modelov môže trvať 1-2 minúty)...")
        musetalk_dir = BASE_DIR / "MuseTalk"
        accel_args: list[str] = []
        if (musetalk_dir / "models/onnx/unet_fp16.trt").exists():
            accel_args = ["--use_onnx", "--use_tensorrt"]
        elif (musetalk_dir / "models/onnx/unet_fp16.onnx").exists():
            accel_args = ["--use_onnx"]
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")

        cmd = [
            _path_str(self.settings.py),
            "-m",
            "scripts.inference",
            "--inference_config",
            "configs/inference/dynamic.yaml",
            "--result_dir",
            "results/test",
            "--unet_model_path",
            "models/musetalkV15/unet.pth",
            "--unet_config",
            "models/musetalkV15/musetalk.json",
            "--version",
            "v15",
            "--parsing_mode",
            "mouth",
            "--left_cheek_width",
            "90",
            "--right_cheek_width",
            "90",
            "--extra_margin",
            "10",
            "--max_width",
            "1280",
            "--saved_coord",
            "--use_saved_coord",
            "--detect_every_n",
            "1",
            "--batch_size",
            "16",
            "--ffmpeg_path",
            ffmpeg_dir,
            *accel_args,
        ]

        # --- Error detection logic ---
        error_detected = False
        def error_handler(line: str) -> bool:
            nonlocal error_detected
            line_lower = line.lower()
            if "division by zero" in line_lower or "error occurred during processing" in line_lower:
                self._queue_log(f"[ERROR] MuseTalk chyba detegovaná v logu: {line.strip()}")
                error_detected = True
            return False # always let default parsing happen

        return_code = self.runner.run(cmd, musetalk_dir, base_progress=base_progress, progress_range=progress_range, on_line=error_handler)

        if error_detected and return_code == 0:
            self._queue_log("[WARN] MuseTalk vrátil exit code 0, ale v logu bola nájdená chyba. Vraciam chybový kód 1.")
            return 1

        return return_code

    def _mux_music_into_lipsync(self, lipsync_path: Path, music_audio: Path) -> bool:
        """Replace audio track in lipsync video with the music-mixed version."""
        tmp = lipsync_path.with_suffix(".music_mux.mp4")
        cmd = [
            self.settings.ffmpeg, "-y",
            "-i", _path_str(lipsync_path),
            "-i", _path_str(music_audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            _path_str(tmp),
        ]
        result = self.runner.run(cmd, BASE_DIR)
        if result == 0 and tmp.exists() and tmp.stat().st_size > 100_000:
            shutil.move(str(tmp), str(lipsync_path))
            self._queue_log("[MUSIC] Pridaná hudba do lipsync videa")
            return True
        else:
            Path(str(tmp)).unlink(missing_ok=True)
            self._queue_log("[WARN] Pridanie hudby do lipsync zlyhalo, ponechávam len reč")
            return False

    def _finalize_musetalk_output(self, dest_file: Path, video_path: Path = None, audio_path: Path = None) -> Path | None:
        v15_dir = BASE_DIR / "MuseTalk" / "results" / "test" / "v15"
        main_dir = BASE_DIR / "MuseTalk" / "results" / "test"

        # Try to find the exact expected output file by MuseTalk's naming convention
        latest = None
        if video_path and audio_path:
            expected_name = f"{video_path.stem}_{audio_path.stem}.mp4"
            for search_dir in (v15_dir, main_dir):
                candidate = search_dir / expected_name
                if candidate.exists():
                    latest = candidate
                    break

        # Fallback to latest-by-mtime
        if not latest:
            latest = _latest_mp4(v15_dir) or _latest_mp4(main_dir)

        if not latest:
            self._queue_log("[ERROR] MuseTalk výstup sa nenašiel.")
            return None

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(latest, dest_file)

        self._queue_log(f"[OK] MuseTalk výstup (final): {dest_file}")
        return dest_file

    def _cleanup_temp_files(self, video_src: str | None = None) -> None:
        """Clean up temporary files after pipeline completion.

        Args:
            video_src: Optional path to source video. If given, also deletes
                       generated .wav and .srt files from its parent directory.
        """
        self._queue_log("[INFO] Čistím dočasné súbory...")
        temp_patterns = [
            BASE_DIR / "work" / "whisper_input.wav",
            BASE_DIR / "input" / "input.mp4",
            BASE_DIR / "out" / "audio_sk.wav",
            BASE_DIR / "out" / "audio_cs.wav",
            BASE_DIR / "out" / "audio_en.wav",
            BASE_DIR / "MuseTalk" / "data" / "video" / "input.mp4",
            BASE_DIR / "MuseTalk" / "data" / "audio" / "input.wav",
            BASE_DIR / "MuseTalk" / "configs" / "inference" / "dynamic.yaml",
            BASE_DIR / "work" / "concat_list.txt",
        ]
        # Glob patterns for wildcard cleanup
        glob_patterns = [
            (BASE_DIR / "work", "*.wav"),
            (BASE_DIR / "work", "*.srt"),
            (BASE_DIR / "work" / "tts_chunks", "*"),
            (BASE_DIR / "work" / "temp_timeline", "*"),
            (BASE_DIR / "input", "*_tts.wav"),
            (BASE_DIR / "input", "*_tts_music.wav"),
            (BASE_DIR / "input", "*.srt"),
        ]
        # Clean wav/srt next to source video (e.g. movie_cs_tts.wav, movie_cs.srt)
        # Also clean temp_stretched/temp_timeline directories created there
        if video_src:
            vp = Path(video_src)
            vid_dir = vp.parent
            stem = vp.stem
            if vid_dir.exists() and vid_dir != BASE_DIR / "input":
                glob_patterns.extend([
                    (vid_dir, f"{stem}_*_tts.wav"),
                    (vid_dir, f"{stem}_*_tts_music.wav"),
                    (vid_dir, f"{stem}_*.srt"),
                ])
                # Clean temp directories next to video
                for temp_name in ["temp_stretched", "temp_timeline"]:
                    temp_path = vid_dir / temp_name
                    if temp_path.exists():
                        try:
                            shutil.rmtree(temp_path, ignore_errors=True)
                        except Exception:
                            pass
        deleted = 0
        for p in temp_patterns:
            try:
                if p.exists():
                    p.unlink()
                    deleted += 1
            except Exception:
                pass
        for folder, pattern in glob_patterns:
            try:
                for f in folder.glob(pattern):
                    if f.is_file():
                        f.unlink()
                        deleted += 1
            except Exception:
                pass
        # Cleanup temp directories
        for temp_dir in ["demucs_output", "work/temp_stretched", "work/temp_timeline", "work/tts_chunks"]:
            try:
                shutil.rmtree(BASE_DIR / temp_dir, ignore_errors=True)
            except Exception:
                pass
        self._queue_log(f"[OK] Vymazaných {deleted} dočasných súborov.")


def main() -> None:
    app = MuseTalkGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
