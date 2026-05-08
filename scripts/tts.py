"""
TTS engines & quality control module.
Piper, Chatterbox, API TTS, ONNX TTS, inspect, silence fallback.
"""

import atexit
import base64
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch

from config import SCRIPT_DIR, TTS_SR, FFMPEG_TIMEOUT

_VT_DIR = Path(__file__).parent.parent
_AI_CHATTERBOX_DIR = _VT_DIR.parent / "Ai" / "models" / "chatterbox"
_FISH_REPO_CANDIDATES = [
    _VT_DIR / "fish_speech" / "repo",
    _VT_DIR.parent / "VideoTranslator" / "fish_speech" / "repo",
]
_DEFAULT_S2_PRO_PYTHON = Path.home() / "miniforge3" / "envs" / "fish_env" / "bin" / "python"
_S2_SERVER_STATE: dict[str, object] = {
    "proc": None,
    "checkpoint": "",
    "port": None,
    "log_handle": None,
    "log_path": "",
}


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _fish_repo_relative_tail(path: Path) -> Path | None:
    parts = path.parts
    for i in range(len(parts) - 1):
        if parts[i] == "fish_speech" and parts[i + 1] == "repo":
            return Path(*parts[i + 2:]) if i + 2 < len(parts) else Path()
    return None


_DEFAULT_FISH_REPO_DIR = _first_existing_path(_FISH_REPO_CANDIDATES)
_DEFAULT_S2_PRO_MODEL = _first_existing_path(
    [repo / "checkpoints" / "s2-pro" for repo in _FISH_REPO_CANDIDATES]
)


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _resolve_chatterbox_asset(path_str: str | None, fallback_name: str | None = None) -> Path:
    raw = (path_str or "").strip()
    candidates: list[Path] = []

    if raw:
        p = Path(raw).expanduser()
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(_VT_DIR / p)
            candidates.append(_VT_DIR.parent / p)
        if p.name:
            candidates.append(_VT_DIR / "models" / "chatterbox" / p.name)
            candidates.append(_AI_CHATTERBOX_DIR / p.name)

    if fallback_name:
        candidates.append(_VT_DIR / "models" / "chatterbox" / fallback_name)
        candidates.append(_AI_CHATTERBOX_DIR / fallback_name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    if raw:
        return Path(raw).expanduser()
    if fallback_name:
        return _AI_CHATTERBOX_DIR / fallback_name
    return Path("")


def _resolve_s2_pro_asset(path_str: str | None) -> Path:
    raw = (path_str or "").strip()
    candidates: list[Path] = []

    if raw:
        p = Path(raw).expanduser()
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(_VT_DIR / p)
            candidates.append(_VT_DIR.parent / p)
            candidates.extend(repo / p for repo in _FISH_REPO_CANDIDATES)
        repo_tail = _fish_repo_relative_tail(p)
        if repo_tail is not None:
            candidates.extend(repo / repo_tail for repo in _FISH_REPO_CANDIDATES)
    else:
        candidates.extend(repo / "checkpoints" / "s2-pro" for repo in _FISH_REPO_CANDIDATES)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    if raw and Path(raw).expanduser().name == "s2-pro":
        return _DEFAULT_S2_PRO_MODEL
    return Path(raw).expanduser() if raw else _DEFAULT_S2_PRO_MODEL


def _resolve_s2_repo_dir(checkpoint: Path) -> Path:
    candidates = [checkpoint.parent.parent, _DEFAULT_FISH_REPO_DIR, *_FISH_REPO_CANDIDATES]
    repo_tail = _fish_repo_relative_tail(checkpoint)
    if repo_tail is not None:
        repo_root = checkpoint
        for _ in repo_tail.parts:
            repo_root = repo_root.parent
        candidates.insert(0, repo_root)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "tools" / "api_server.py").exists():
            return candidate

    return _DEFAULT_FISH_REPO_DIR


def _resolve_s2_server_python(path_str: str | None) -> Path:
    raw = (path_str or "").strip()
    candidates: list[Path] = []

    if raw:
        p = Path(raw).expanduser()
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(_VT_DIR / p)
            candidates.append(_VT_DIR.parent / p)

    candidates.append(_DEFAULT_S2_PRO_PYTHON)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path(raw).expanduser() if raw else _DEFAULT_S2_PRO_PYTHON


def _s2_health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/health"


def _s2_tts_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/tts"


def _s2_server_ready(port: int) -> bool:
    try:
        resp = requests.get(_s2_health_url(port), timeout=1.5)
        return resp.ok and resp.json().get("status") == "ok"
    except Exception:
        return False


def _cleanup_s2_pro_server() -> None:
    proc = _S2_SERVER_STATE.get("proc")
    if proc is not None and getattr(proc, "poll", None) is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    log_handle = _S2_SERVER_STATE.get("log_handle")
    if log_handle is not None:
        try:
            log_handle.close()
        except Exception:
            pass

    _S2_SERVER_STATE.update(
        {
            "proc": None,
            "checkpoint": "",
            "port": None,
            "log_handle": None,
            "log_path": "",
        }
    )


atexit.register(_cleanup_s2_pro_server)


def ensure_s2_pro_server(
    checkpoint_path: str,
    *,
    server_python: str = "",
    port: int = 8091,
    startup_timeout_s: float = 180.0,
) -> str:
    checkpoint = _resolve_s2_pro_asset(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"S2-Pro checkpoint not found: {checkpoint}")
    repo_dir = _resolve_s2_repo_dir(checkpoint)
    if not (repo_dir / "tools" / "api_server.py").exists():
        raise FileNotFoundError(f"Fish Speech repo not found: {repo_dir}")

    server_py = _resolve_s2_server_python(server_python)
    if not server_py.exists():
        raise FileNotFoundError(f"S2-Pro server python not found: {server_py}")

    current_proc = _S2_SERVER_STATE.get("proc")
    current_checkpoint = str(_S2_SERVER_STATE.get("checkpoint") or "")
    current_port = _S2_SERVER_STATE.get("port")
    if (
        current_proc is not None
        and getattr(current_proc, "poll", None) is not None
        and current_proc.poll() is None
        and current_checkpoint == str(checkpoint)
        and current_port == port
        and _s2_server_ready(port)
    ):
        return _s2_tts_url(port)

    if current_proc is not None:
        _cleanup_s2_pro_server()

    if _s2_server_ready(port):
        print(f"[S2-PRO] Reusing running Fish server on port {port}", flush=True)
        _S2_SERVER_STATE.update(
            {
                "proc": None,
                "checkpoint": str(checkpoint),
                "port": port,
                "log_handle": None,
                "log_path": "",
            }
        )
        return _s2_tts_url(port)

    log_path = _VT_DIR / "temp" / "s2_pro_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    cmd = [
        str(server_py),
        "tools/api_server.py",
        "--llama-checkpoint-path",
        str(checkpoint),
        "--decoder-checkpoint-path",
        str(checkpoint / "codec.pth"),
        "--decoder-config-name",
        "modded_dac_vq",
        "--listen",
        f"127.0.0.1:{port}",
        "--half",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[S2-PRO] Starting Fish server from {checkpoint}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _S2_SERVER_STATE.update(
        {
            "proc": proc,
            "checkpoint": str(checkpoint),
            "port": port,
            "log_handle": log_handle,
            "log_path": str(log_path),
        }
    )

    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if _s2_server_ready(port):
            print(f"[S2-PRO] Fish server ready on port {port}", flush=True)
            return _s2_tts_url(port)
        if proc.poll() is not None:
            log_tail = ""
            try:
                log_tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
            except Exception:
                pass
            raise RuntimeError(
                "S2-Pro server exited before becoming ready.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Log tail:\n{log_tail}"
            )
        time.sleep(1.0)

    raise TimeoutError(f"S2-Pro server did not become ready within {startup_timeout_s:.0f}s")


def _preprocess_tts_text(text: str) -> str:
    """Pre-TTS normalizácia skratiek a čísel ktoré Fish Speech zle číta.

    Opravuje:
    - "20 %" → "20 percent" (Fish Speech inak číta "200%")
    - "NDA" → "eN-Dý-Á" (Fish Speech číta ako "enda")
    - "Y100" → "ypsilon sto" (Fish Speech zlyháva)
    - "2.0" → "dva-nula" (pre éra enžinov 2.0)
    - "C++" → "sí-plus-plus" (ak už nie je fonetizované)
    """
    if not text:
        return text
    # 20 % → 20 percent  (len izolované číslo + %)
    text = re.sub(r'(\d+)\s*%', r'\1 percent', text)
    # NDA (izolované, 3 veľké písmená) → hláskovať
    text = re.sub(r'\bNDA\b', 'eN-Dý-Á', text)
    text = re.sub(r'\bNDAs\b', 'eN-Dý-Áčka', text)
    # Y100, X100 atď. (písmeno + číslo bez medzery)
    text = re.sub(r'\bY(\d+)\b', r'ypsilon \1', text)
    text = re.sub(r'\bX(\d+)\b', r'iks \1', text)
    # 2.0, 3.0 atď. vo verziách
    text = re.sub(r'\b(\d+)\.0\b', r'\1-nula', text)
    return text


def s2_pro_tts(
    *,
    text: str,
    output_path: Path,
    checkpoint_path: str,
    reference_audio_path: str | None = None,
    reference_text: str | None = None,
    server_python: str = "",
    server_port: int = 8091,
    timeout_s: float = 600.0,
) -> Path:
    url = ensure_s2_pro_server(
        checkpoint_path,
        server_python=server_python,
        port=server_port,
    )

    # Pre-TTS normalizácia — Fish Speech zle číta %, NDA, Y100, 2.0 atď.
    text = _preprocess_tts_text(text)

    payload: dict[str, object] = {
        "text": unicodedata.normalize("NFC", text or "").strip(),
        "format": "wav",
        "chunk_length": 200,
        "max_new_tokens": 1024,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
        "normalize": True,
    }

    ref_audio = Path(reference_audio_path).expanduser() if reference_audio_path else None
    ref_text_clean = re.sub(r"\s+", " ", unicodedata.normalize("NFC", reference_text or "")).strip()
    if ref_audio is not None and ref_audio.exists() and ref_text_clean:
        payload["references"] = [
            {
                "audio": base64.b64encode(ref_audio.read_bytes()).decode("ascii"),
                "text": ref_text_clean,
            }
        ]

    print(f"[S2-PRO] Synthesizing via {url}", flush=True)
    response = requests.post(url, json=payload, timeout=timeout_s)
    if response.status_code != 200:
        detail = response.text[:400] if response.text else f"HTTP {response.status_code}"
        raise RuntimeError(f"S2-Pro TTS request failed: {detail}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path


@lru_cache(maxsize=1)
def _load_sk_normalize_fn():
    candidates = [
        _VT_DIR.parent / "Ai" / "chatterbox-finetuning" / "src" / "text_normalize_sk.py",
        _VT_DIR.parent / "chatterbox-finetuning" / "src" / "text_normalize_sk.py",
    ]

    for module_path in candidates:
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("vt_text_normalize_sk", module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        normalize_fn = getattr(module, "normalize_sk_turbo", None)
        if not callable(normalize_fn):
            normalize_fn = getattr(module, "normalize_sk", None)
        if callable(normalize_fn):
            return normalize_fn
    return None


def _normalize_sk_turbo_text(text: str) -> str:
    """
    Turbo SK model was trained on normalized text (numbers, dates, currencies).
    Reuse the same normalizer at inference when available.
    """
    raw = unicodedata.normalize("NFC", text or "").strip()
    if not raw:
        return raw

    try:
        normalize_fn = _load_sk_normalize_fn()
        if callable(normalize_fn):
            normalized = normalize_fn(raw)
            return unicodedata.normalize("NFC", normalized).strip()
    except Exception as e:
        print(f"[TURBO] WARNING: SK text normalization unavailable: {e}", flush=True)

    return raw


def _stabilize_short_tts_text(text: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return raw

    core = raw.rstrip(".!?…,:;").strip()
    if not core:
        return raw

    words = core.split()
    if len(core) <= 2:
        return core + "..."
    if len(words) == 1 and len(core) <= 5:
        return core + "..."
    if len(core) < 10 and raw[-1] not in ".!?…,:;":
        return raw + "."
    return raw


def _is_short_tts_reply(text: str) -> bool:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return False
    core = raw.rstrip(".!?…,:;").strip()
    if not core:
        return False
    return len(core.split()) <= 2 and len(core) <= 12


def cleanup_generated_tts_audio(
    wav_path: Path,
    short_text: bool = False,
    aggressive: bool = False,
) -> bool:
    """Light in-place cleanup for generated TTS audio.

    Clonevoice outputs can carry a faint hiss/interference floor that becomes
    much more obvious after compression and loudnorm. Keep this pass gentler
    than reference cleanup so we reduce noise without making speech metallic.
    """
    if not wav_path.exists():
        return False

    try:
        dur_s = float(sf.info(str(wav_path)).duration)
    except Exception:
        dur_s = 0.0

    if dur_s <= 0.05:
        return False

    tmp_path = wav_path.with_name(f"{wav_path.stem}_ttsclean.wav")
    if aggressive:
        af_parts = [
            "highpass=f=70",
            "afftdn=nr=10:nt=w:om=o",
            "equalizer=f=90:t=q:w=0.7:g=+5.0",   # torch bass
            "equalizer=f=160:t=q:w=1.0:g=+2.5",  # bass warmth
            "lowpass=f=5500",
            "equalizer=f=3800:t=q:w=1.0:g=-4.5",  # cut presence
            "equalizer=f=5200:t=q:w=1.2:g=-6.0",  # destroy highs
        ]
    elif short_text:
        af_parts = [
            "highpass=f=55",
            "afftdn=nr=5:nt=w:om=o",
            "equalizer=f=90:t=q:w=0.7:g=+5.0",   # torch bass
            "equalizer=f=160:t=q:w=1.0:g=+2.5",  # bass warmth
            "lowpass=f=5800",
            "equalizer=f=5000:t=q:w=1.1:g=-5.5",  # destroy highs
        ]
    else:
        af_parts = [
            "highpass=f=60",
            "afftdn=nr=7:nt=w:om=o",
            "equalizer=f=90:t=q:w=0.7:g=+5.0",   # torch bass
            "equalizer=f=160:t=q:w=1.0:g=+2.5",  # bass warmth
            "lowpass=f=5500",
            "equalizer=f=3800:t=q:w=1.0:g=-3.5",  # cut presence
            "equalizer=f=5200:t=q:w=1.2:g=-5.5",  # destroy highs
        ]

    timeout_s = max(60, int(dur_s * 4) + 20)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-af", ",".join(af_parts),
        "-ac", "1",
        "-ar", str(TTS_SR),
        "-c:a", "pcm_s16le",
        str(tmp_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[TTS-CLEAN] Timeout after {timeout_s}s: {wav_path.name}", flush=True)
        tmp_path.unlink(missing_ok=True)
        return False

    if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        print(f"[TTS-CLEAN] Failed: {result.stderr[-300:]}", flush=True)
        tmp_path.unlink(missing_ok=True)
        return False

    tmp_path.replace(wav_path)
    _mode = "strong" if aggressive else ("short" if short_text else "mild")
    print(f"[TTS-CLEAN] Output cleaned ({_mode}): {wav_path.name}", flush=True)
    return True

def write_wav_safe(path: Path | str, wav: np.ndarray, sr: int) -> None:
    """Write audio as PCM16 after handling NaNs and scaling glitches."""
    arr = np.asarray(wav)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    arr = arr.astype(np.float32)
    if peak > 1.5:
        arr = arr / 32768.0
    sf.write(str(path), arr, sr, subtype="PCM_16")


def write_silence_chunk(path: Path, duration_s: float, sr: int = TTS_SR) -> None:
    """Create a deterministic silence fallback chunk when TTS retries fail."""
    dur = max(0.12, float(duration_s))
    n = max(1, int(dur * sr))
    silence = np.zeros(n, dtype=np.float32)
    write_wav_safe(path, silence, sr)


# ---------------------------------------------------------------------------
# TTS chunk quality inspection
# ---------------------------------------------------------------------------

def inspect_tts_chunk(path: Path, expected_sec: float | None = None) -> tuple[bool, str]:
    """
    Validate generated TTS chunk to catch silent/glitched outputs.
    Returns (is_bad, reason).
    """
    if not path.exists():
        return True, "missing file"
    try:
        y, sr = sf.read(str(path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if y.size == 0:
            return True, "empty audio"
        dur = len(y) / float(sr)
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y ** 2)))
        if dur < 0.08:
            return True, f"too short ({dur:.3f}s)"
        if expected_sec is not None:
            min_expected = max(0.10, min(1.20, expected_sec * 0.22))
            if dur < min_expected:
                return True, f"short vs expected ({dur:.3f}s < {min_expected:.3f}s)"
        if peak < 0.004:
            return True, f"very low peak ({peak:.5f})"
        if rms < 0.0008:
            return True, f"very low RMS ({rms:.5f})"
        if not np.isfinite(y).all():
            return True, "non-finite samples"
        if y.size > 1600:
            signs = np.signbit(y)
            zcr = float(np.mean(signs[1:] != signs[:-1]))
            crest = peak / max(rms, 1e-9)
            # Thresholds tuned for Czech TTS: sibilants (š,č,ž,s,z) have ZCR 0.30-0.45
            # and crest 2.0-3.0. Only flag clearly pathological noise (near-white-noise level).
            if zcr > 0.46 and crest < 1.6:
                return True, f"noisy chunk (zcr={zcr:.3f}, crest={crest:.2f})"
        clipped_ratio = float(np.mean(np.abs(y) >= 0.999))
        if clipped_ratio > 0.18:
            return True, f"heavy clipping ({clipped_ratio:.1%})"

        # --- Spectral analysis to catch distorted/horror-sounding TTS ---
        min_fft_samples = int(sr * 0.15)
        if y.size >= min_fft_samples:
            frame_len = min(2048, y.size)
            hop = frame_len // 2
            n_frames = max(1, (y.size - frame_len) // hop + 1)

            flatness_vals = []
            centroid_vals = []
            rolloff_vals = []
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)

            for fi in range(n_frames):
                start = fi * hop
                frame = y[start : start + frame_len]
                if len(frame) < frame_len:
                    break
                window = np.hanning(frame_len)
                spectrum = np.abs(np.fft.rfft(frame * window))
                spectrum_pow = spectrum ** 2
                total_energy = spectrum_pow.sum()
                if total_energy < 1e-12:
                    continue
                log_spec = np.log(spectrum_pow + 1e-12)
                geo_mean = np.exp(log_spec.mean())
                arith_mean = spectrum_pow.mean()
                flatness = geo_mean / max(arith_mean, 1e-12)
                flatness_vals.append(float(flatness))
                centroid = float(np.sum(freqs * spectrum_pow) / total_energy)
                centroid_vals.append(centroid)
                cumsum = np.cumsum(spectrum_pow)
                rolloff_idx = np.searchsorted(cumsum, 0.85 * total_energy)
                rolloff_freq = float(freqs[min(rolloff_idx, len(freqs) - 1)])
                rolloff_vals.append(rolloff_freq)

            if flatness_vals:
                avg_flatness = float(np.median(flatness_vals))
                avg_centroid = float(np.median(centroid_vals))
                avg_rolloff = float(np.median(rolloff_vals))

                if avg_flatness > 0.40:
                    return True, f"spectral flatness too high ({avg_flatness:.3f}) - distorted/noisy"
                # centroid and rolloff checks removed — Czech sibilants (š,č,ž) at 24kHz
                # push median centroid above 5000 Hz and rolloff above 9000 Hz on valid speech.

        # --- Repetition/looping detection ---
        min_loop_samples = int(sr * 0.60)
        if y.size >= min_loop_samples:
            if expected_sec is not None and expected_sec >= 0.1:
                ratio = dur / expected_sec
                # For very short windows (<0.5s) allow more expansion (translation overhead);
                # for normal windows flag at 3.5x.
                max_ratio = 3.5 if expected_sec >= 0.5 else 12.0
                if ratio > max_ratio:
                    return True, f"duration ratio too high ({ratio:.1f}x expected) - likely looping"

            analysis_len = min(y.size, int(sr * 4.0))
            segment = y[:analysis_len]
            segment = segment - segment.mean()
            seg_energy = float(np.sum(segment ** 2))
            if seg_energy > 1e-10:
                min_lag = int(sr * 0.10)
                max_lag = int(sr * 0.40)
                max_lag = min(max_lag, analysis_len // 2)

                if max_lag > min_lag:
                    autocorr_vals = []
                    for lag in range(min_lag, max_lag, max(1, (max_lag - min_lag) // 60)):
                        corr = float(np.sum(segment[:-lag] * segment[lag:])) / seg_energy
                        autocorr_vals.append(corr)

                    if autocorr_vals:
                        peak_corr = max(autocorr_vals)
                        high_corr_count = sum(1 for c in autocorr_vals if c > 0.50)
                        high_corr_ratio = high_corr_count / len(autocorr_vals)

                        if peak_corr > 0.72:
                            return True, (
                                f"repetition detected (autocorr={peak_corr:.3f}) - TTS looping"
                            )
                        if high_corr_ratio > 0.50 and peak_corr > 0.55:
                            return True, (
                                f"repetitive pattern ({high_corr_ratio:.0%} lags correlated, "
                                f"peak={peak_corr:.3f}) - TTS looping"
                            )

        # --- Mid-chunk distortion detection ---
        # Chatterbox can lose coherence mid-utterance, producing quiet garbled
        # sections surrounded by normal speech.  Scan 500ms windows and flag
        # if any window has RMS < 12% of overall RMS while BOTH neighbours
        # are loud.  Only runs on chunks >8s to avoid false positives on
        # natural breath pauses in shorter auto-split sub-chunks.
        window_s = 0.5
        window_samples = int(sr * window_s)
        if dur > 8.0 and y.size > window_samples * 3 and rms > 0.01:
            n_windows = y.size // window_samples
            win_rms = np.array([
                float(np.sqrt(np.mean(y[j * window_samples:(j + 1) * window_samples] ** 2)))
                for j in range(n_windows)
            ])
            drop_thresh = rms * 0.12
            for wi in range(1, n_windows - 1):
                if win_rms[wi] < drop_thresh:
                    # Both neighbours must be at normal level
                    left_ok = win_rms[wi - 1] > rms * 0.40
                    right_ok = win_rms[wi + 1] > rms * 0.40
                    if left_ok and right_ok:
                        t_drop = wi * window_s
                        return True, (
                            f"mid-chunk distortion at ~{t_drop:.1f}s "
                            f"(window RMS={win_rms[wi]:.4f} vs avg={rms:.4f})"
                        )

        # Diagnostic log for tuning thresholds
        diag_parts = [f"dur={dur:.2f}s rms={rms:.4f} peak={peak:.3f}"]
        if y.size >= int(sr * 0.15) and 'avg_flatness' in dir():
            pass
        try:
            if flatness_vals:
                diag_parts.append(f"flat={avg_flatness:.3f} cent={avg_centroid:.0f}Hz roll={avg_rolloff:.0f}Hz")
        except NameError:
            pass
        try:
            if autocorr_vals:
                diag_parts.append(f"acorr={peak_corr:.3f} hratio={high_corr_ratio:.0%}")
        except NameError:
            pass
        if expected_sec:
            diag_parts.append(f"ratio={dur/expected_sec:.2f}x")
        print(f"[INSPECT] {path.name}: {' | '.join(diag_parts)}", flush=True)

        return False, ""
    except Exception as e:
        return True, f"inspect error: {e}"


# ---------------------------------------------------------------------------
# Voice sample helper
# ---------------------------------------------------------------------------

def cleanup_reference_audio(wav_path: Path, aggressive: bool = False) -> bool:
    """Clean a voice-reference WAV in place for cloning stability.

    The goal is not hi-fi restoration, but removing hum/hiss/edge silence so
    clone prompts carry the speaker identity with less background interference.
    """
    if not wav_path.exists():
        return False

    try:
        dur_s = float(sf.info(str(wav_path)).duration)
    except Exception:
        dur_s = 0.0

    tmp_path = wav_path.with_name(f"{wav_path.stem}_refclean.wav")
    if aggressive:
        af = ",".join([
            "highpass=f=90",
            "afftdn=nr=18:nt=w:om=o",
            "lowpass=f=9000",
            "equalizer=f=220:t=q:w=1.0:g=-2.5",
            "equalizer=f=6400:t=q:w=1.2:g=-2.2",
            "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-48dB",
            "areverse",
            "silenceremove=start_periods=1:start_silence=0.08:start_threshold=-45dB",
            "areverse",
        ])
    else:
        af = ",".join([
            "highpass=f=70",
            "afftdn=nr=12:nt=w:om=o",
            "lowpass=f=11000",
            "equalizer=f=6800:t=q:w=1.2:g=-1.5",
            "silenceremove=start_periods=1:start_silence=0.04:start_threshold=-50dB",
            "areverse",
            "silenceremove=start_periods=1:start_silence=0.06:start_threshold=-46dB",
            "areverse",
        ])

    timeout_s = max(60, int(dur_s * 4) + 20)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-af", af,
        "-ac", "1",
        "-ar", str(TTS_SR),
        "-c:a", "pcm_s16le",
        str(tmp_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[VOICE] Reference cleanup timed out after {timeout_s}s: {wav_path.name}", flush=True)
        if tmp_path.exists():
            tmp_path.unlink()
        return False

    if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        print(f"[VOICE] Reference cleanup failed: {result.stderr[-300:]}", flush=True)
        if tmp_path.exists():
            tmp_path.unlink()
        return False

    tmp_path.replace(wav_path)
    print(
        f"[VOICE] Reference cleaned ({'strong' if aggressive else 'mild'}): {wav_path.name}",
        flush=True,
    )
    return True


def _score_reference_window_audio(
    window: np.ndarray,
    sr: int,
    *,
    librosa_mod=None,
    silence_threshold: float = 0.01,
) -> float:
    """Score a reference-audio window for clone quality.

    Higher score means: denser speech, less silence, less clipping, less hiss.
    """
    if window.ndim > 1:
        window = window.mean(axis=1)
    window = np.asarray(window, dtype=np.float32)
    if window.size < max(1, int(sr * 0.35)):
        return 0.0

    rms = float(np.sqrt(np.mean(window ** 2)))
    silence_frac = float(np.mean(np.abs(window) < silence_threshold))
    clipped_frac = float(np.mean(np.abs(window) > 0.98))

    penalty = 1.0 + min(1.0, clipped_frac * 6.0)
    activity = max(0.05, 1.0 - silence_frac)

    if librosa_mod is not None:
        try:
            frame_rms = librosa_mod.feature.rms(
                y=window,
                frame_length=1024,
                hop_length=256,
            )[0]
            if frame_rms.size:
                activity_thr = max(float(np.median(frame_rms) * 1.10), 0.008)
                active_ratio = float(np.mean(frame_rms > activity_thr))
                activity *= max(0.35, active_ratio)
            flatness = float(np.mean(librosa_mod.feature.spectral_flatness(
                y=window,
                n_fft=1024,
                hop_length=256,
            )))
            zcr = float(np.mean(librosa_mod.feature.zero_crossing_rate(
                window,
                frame_length=1024,
                hop_length=256,
            )))
            penalty += min(0.9, flatness * 2.4) + min(0.5, zcr * 3.5)
        except Exception:
            pass

    return (rms * activity) / penalty


def optimize_chatterbox_reference_prompt(
    wav_path: Path,
    *,
    target_dur: float = 5.8,
    min_dur: float = 3.2,
) -> bool:
    """Crop a clone reference to the densest speech-heavy window.

    Chatterbox clone quality is usually better with ~4-6 s of dense speech
    than with a long prompt full of pauses or noisy tails.
    """
    if not wav_path.exists():
        return False

    try:
        import librosa
    except Exception:
        librosa = None

    try:
        y, sr = sf.read(str(wav_path), dtype="float32")
    except Exception as e:
        print(f"[VOICE] Reference optimize failed to read {wav_path.name}: {e}", flush=True)
        return False

    if y.ndim > 1:
        y = y.mean(axis=1)

    dur_s = len(y) / float(sr) if sr > 0 else 0.0
    if dur_s < min_dur or dur_s <= target_dur + 0.35:
        return False

    keep_dur = min(max(min_dur, target_dur), dur_s)
    keep_samples = max(1, int(keep_dur * sr))
    hop_samples = max(1, int(sr * 0.35))

    best_score = -1.0
    best_start = 0
    pos = 0
    while pos + keep_samples <= len(y):
        window = y[pos:pos + keep_samples]
        score = _score_reference_window_audio(window, sr, librosa_mod=librosa)
        if score > best_score:
            best_score = score
            best_start = pos
        pos += hop_samples

    if best_score <= 0.0:
        return False

    cropped = y[best_start:best_start + keep_samples].copy()
    fade_n = min(int(sr * 0.02), len(cropped) // 8)
    if fade_n > 8:
        cropped[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        cropped[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    sf.write(str(wav_path), cropped, sr, subtype="PCM_16")
    print(
        f"[VOICE] Reference optimized: {wav_path.name} "
        f"({best_start / sr:.1f}s–{(best_start + keep_samples) / sr:.1f}s of {dur_s:.1f}s)",
        flush=True,
    )
    return True


def prepare_chatterbox_reference_audio(
    wav_path: Path,
    *,
    aggressive: bool = False,
    target_dur: float = 5.8,
) -> bool:
    """Clean and tighten a reference WAV for Chatterbox cloning."""
    clean_ok = cleanup_reference_audio(wav_path, aggressive=aggressive)
    optimize_chatterbox_reference_prompt(
        wav_path,
        target_dur=5.2 if aggressive else target_dur,
        min_dur=2.8 if aggressive else 3.2,
    )
    return clean_ok or wav_path.exists()


def extract_chatterbox_voice_sample(
    audio_path: Path,
    output_path: Path,
    start_sec: float = 0,
    duration_sec: float = 10,
) -> bool:
    if not extract_voice_sample(audio_path, output_path, start_sec=start_sec, duration_sec=duration_sec):
        return False
    optimize_chatterbox_reference_prompt(
        output_path,
        target_dur=min(5.8, max(3.4, float(duration_sec) - 1.0)),
    )
    return output_path.exists()


def extract_best_chatterbox_voice_sample(
    audio_path: Path,
    output_path: Path,
    sample_dur: float = 10.0,
    step: float = 5.0,
    min_start: float = 0.0,
    max_end: float | None = None,
    silence_threshold: float = 0.01,
) -> bool:
    if not extract_best_voice_sample(
        audio_path,
        output_path,
        sample_dur=sample_dur,
        step=step,
        min_start=min_start,
        max_end=max_end,
        silence_threshold=silence_threshold,
    ):
        return False
    optimize_chatterbox_reference_prompt(
        output_path,
        target_dur=min(5.8, max(3.6, float(sample_dur) - 1.2)),
    )
    return output_path.exists()


def extract_best_chatterbox_reference_from_segments(
    audio_path: Path,
    output_path: Path,
    *,
    segments: list[dict],
    target_dur: float = 6.0,
    max_total_dur: float = 8.0,
) -> bool:
    """Build a Chatterbox clone prompt from the best source speech segments."""
    import shutil
    import tempfile

    try:
        import librosa
    except Exception:
        librosa = None

    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value or "")).strip()

    candidates: list[dict[str, object]] = []
    for idx, seg in enumerate(segments or []):
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        dur = max(0.0, end - start)
        text = _clean_text(seg.get("text_src") or seg.get("text") or "")
        if dur < 1.3 or dur > 10.5 or not text:
            continue
        if len(text.split()) < 2 and len(text) < 8:
            continue
        meta_score = min(dur, 5.5) * 1.5 + min(len(text), 140) * 0.025 - abs(dur - 4.0) * 0.7
        if dur > 7.5:
            meta_score -= (dur - 7.5) * 0.9
        candidates.append({
            "idx": idx,
            "start": start,
            "end": end,
            "dur": dur,
            "text": text,
            "meta_score": meta_score,
        })

    if not candidates:
        print("[VOICE] No good transcript segments for Chatterbox clone prompt.", flush=True)
        return False

    candidates.sort(key=lambda item: float(item["meta_score"]), reverse=True)
    probe_limit = min(12, len(candidates))
    scratch_dir = Path(tempfile.mkdtemp(prefix="cb_ref_"))
    analyzed: list[dict[str, object]] = []
    try:
        for cand in candidates[:probe_limit]:
            tmp = scratch_dir / f"cand_{int(cand['idx']):04d}.wav"
            pad_s = 0.10 if float(cand["dur"]) >= 4.0 else 0.16
            t_start = max(0.0, float(cand["start"]) - pad_s)
            duration = (float(cand["end"]) + pad_s) - t_start
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(t_start),
                "-t", str(duration),
                "-i", str(audio_path),
                "-ac", "1",
                "-ar", str(TTS_SR),
                "-c:a", "pcm_s16le",
                str(tmp),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
                if result.returncode != 0 or not tmp.exists():
                    continue
                y, sr = sf.read(str(tmp), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                audio_score = _score_reference_window_audio(y, sr, librosa_mod=librosa)
                analyzed.append({
                    **cand,
                    "probe_path": tmp,
                    "audio_score": audio_score,
                    "score": float(cand["meta_score"]) + audio_score * 120.0,
                })
            except Exception:
                continue

        if not analyzed:
            print("[VOICE] Segment-based clone prompt failed, falling back to window search.", flush=True)
            return extract_best_chatterbox_voice_sample(
                audio_path,
                output_path,
                sample_dur=max_total_dur,
                step=max(1.5, target_dur / 2.0),
            )

        analyzed.sort(key=lambda item: float(item["score"]), reverse=True)
        selected: list[dict[str, object]] = []
        selected_total = 0.0
        for cand in analyzed:
            start = float(cand["start"])
            end = float(cand["end"])
            overlaps = any(not (end <= float(prev["start"]) - 0.35 or start >= float(prev["end"]) + 0.35) for prev in selected)
            if overlaps:
                continue
            selected.append(cand)
            selected_total += float(cand["dur"])
            if selected_total >= target_dur or len(selected) >= 3:
                break

        if not selected:
            selected = [analyzed[0]]

        parts: list[np.ndarray] = []
        out_sr = TTS_SR
        for si, cand in enumerate(selected):
            probe_path = Path(str(cand["probe_path"]))
            y, sr = sf.read(str(probe_path), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = np.asarray(y, dtype=np.float32)
            fade_n = min(int(sr * 0.02), len(y) // 8)
            if fade_n > 8:
                y[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
                y[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
            parts.append(y)
            out_sr = sr
            if si < len(selected) - 1:
                parts.append(np.zeros(int(sr * 0.12), dtype=np.float32))

        if not parts:
            return False

        combined = np.concatenate(parts)
        sf.write(str(output_path), combined, out_sr, subtype="PCM_16")
        prepare_chatterbox_reference_audio(
            output_path,
            aggressive=False,
            target_dur=min(5.8, max(4.0, target_dur)),
        )
        print(
            "[VOICE] Segment-based clone prompt: "
            + ", ".join(
                f"seg {int(c['idx']) + 1} ({float(c['dur']):.1f}s)"
                for c in selected
            ),
            flush=True,
        )
        return output_path.exists()
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

def extract_voice_sample(audio_path: Path, output_path: Path, start_sec: float = 0, duration_sec: float = 10) -> bool:
    print(f"[VOICE] Extracting {duration_sec}s voice sample from {start_sec}s", flush=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-ac", "1",
        "-ar", str(TTS_SR),
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if result.returncode != 0:
        print(f"[VOICE] Extraction failed: {result.stderr}", flush=True)
        return False
    cleanup_reference_audio(output_path, aggressive=False)
    print(f"[VOICE] Sample saved: {output_path}", flush=True)
    return True


def extract_best_voice_sample(
    audio_path: Path,
    output_path: Path,
    sample_dur: float = 10.0,
    step: float = 5.0,
    min_start: float = 0.0,
    max_end: float | None = None,
    silence_threshold: float = 0.01,
) -> bool:
    """Extract the highest-quality speech segment from audio_path.

    Scans the audio in overlapping windows, scores each window by RMS energy
    and silence fraction, then extracts the best window to output_path.
    Returns True on success, False on failure.
    """
    import tempfile

    try:
        import librosa
    except Exception:
        librosa = None

    # Decode to a temporary mono WAV for analysis
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
        tmp_path = Path(tmp_f.name)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", str(TTS_SR),
            "-c:a", "pcm_s16le",
            str(tmp_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        if result.returncode != 0:
            print(f"[VOICE] Decode failed: {result.stderr[-300:]}", flush=True)
            return False

        wav, sr = sf.read(str(tmp_path))
        wav = wav.astype(np.float32)
        total_dur = len(wav) / sr

        end_limit = min(max_end, total_dur) if max_end is not None else total_dur
        win_samples = int(sample_dur * sr)
        step_samples = int(step * sr)

        best_score = -1.0
        best_start_s = min_start
        pos = int(min_start * sr)

        while pos + win_samples <= int(end_limit * sr):
            window = wav[pos: pos + win_samples]
            rms = float(np.sqrt(np.mean(window ** 2)))
            silence_frac = float(np.mean(np.abs(window) < silence_threshold))
            clipped_frac = float(np.mean(np.abs(window) > 0.98))
            noise_penalty = 1.0
            if librosa is not None:
                try:
                    flatness = float(np.mean(librosa.feature.spectral_flatness(
                        y=window.astype(np.float32),
                        n_fft=1024,
                        hop_length=256,
                    )))
                    zcr = float(np.mean(librosa.feature.zero_crossing_rate(
                        window.astype(np.float32),
                        frame_length=1024,
                        hop_length=256,
                    )))
                    noise_penalty += min(0.75, flatness * 2.2) + min(0.35, zcr * 3.0)
                except Exception:
                    pass
            score = (rms * max(0.0, 1.0 - silence_frac) * max(0.2, 1.0 - clipped_frac * 4.0)) / noise_penalty
            if score > best_score:
                best_score = score
                best_start_s = pos / sr
            pos += step_samples

        if best_score <= 0:
            best_start_s = min_start

        print(
            f"[VOICE] Best sample window: {best_start_s:.1f}s–{best_start_s + sample_dur:.1f}s "
            f"(score={best_score:.4f}, total={total_dur:.1f}s)",
            flush=True
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return extract_voice_sample(audio_path, output_path, start_sec=best_start_s, duration_sec=sample_dur)


def prepare_s2_pro_reference(
    *,
    audio_path: Path,
    output_path: Path,
    segments: list[dict],
    auto: bool = True,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
) -> tuple[str, str] | None:
    """Prepare a Fish S2-Pro reference clip plus matching source transcript."""

    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value or "")).strip()

    candidates: list[dict] = []
    for seg in segments or []:
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", seg_start) or seg_start)
        seg_dur = max(0.0, seg_end - seg_start)
        seg_text = _clean_text(seg.get("text_src") or seg.get("text") or "")
        if not seg_text or seg_dur < 1.2:
            continue
        candidates.append(
            {
                "start": seg_start,
                "end": seg_end,
                "dur": seg_dur,
                "text": seg_text,
            }
        )

    if not candidates:
        print("[S2-PRO] No suitable source segments for reference cloning; using default voice.", flush=True)
        return None

    ref_start = max(0.0, float(start_sec))
    ref_dur = max(1.0, float(duration_sec))
    ref_text = ""

    if auto:
        def _score(seg: dict) -> float:
            dur = float(seg["dur"])
            text_len = len(seg["text"])
            target = min(max(ref_dur, 3.0), 10.0)
            return min(dur, target) * 12.0 + min(text_len, 220) * 0.08 - abs(dur - target) * 2.5

        best = max(candidates, key=_score)
        ref_start = best["start"]
        ref_dur = min(max(2.0, float(best["dur"])), ref_dur)
        ref_text = best["text"]
    else:
        ref_end = ref_start + ref_dur
        overlapping: list[dict] = []
        for seg in candidates:
            overlap = max(0.0, min(seg["end"], ref_end) - max(seg["start"], ref_start))
            if overlap >= 0.35:
                overlapping.append(seg)
        if not overlapping:
            print("[S2-PRO] Manual reference range has no transcript overlap; using default voice.", flush=True)
            return None
        ref_text = _clean_text(" ".join(seg["text"] for seg in overlapping))

    if not ref_text:
        print("[S2-PRO] Reference transcript is empty; using default voice.", flush=True)
        return None

    if not extract_voice_sample(audio_path, output_path, start_sec=ref_start, duration_sec=ref_dur):
        return None

    return str(output_path), ref_text


# ---------------------------------------------------------------------------
# API TTS (OpenAI-compatible)
# ---------------------------------------------------------------------------

def api_tts(api_key: str, base_url: str, model: str, voice: str, text: str, output_path: Path, timeout_s: int = 300) -> Path:
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text).strip()
    if not text:
        raise RuntimeError("API TTS got empty text")
    base = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    url = f"{base}/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "wav",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if not r.ok:
        raise RuntimeError(f"API TTS HTTP {r.status_code}: {r.text[:400]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(r.content)
    return output_path


def openai_tts(api_key: str, model: str, voice: str, text: str, output_path: Path, timeout_s: int = 300) -> Path:
    return api_tts(api_key, "https://api.openai.com/v1", model, voice, text, output_path, timeout_s)


# ---------------------------------------------------------------------------
# Piper TTS
# ---------------------------------------------------------------------------

def piper_tts(model_path: str, text: str, output_path: Path, speaker_id: int = 0) -> Path:
    try:
        from piper import PiperVoice
    except ImportError:
        raise RuntimeError("Piper TTS not installed. Run: pip install piper-tts")
    print(f"[PIPER] Generating: {text[:50]}...", flush=True)
    if not hasattr(piper_tts, "_voice") or piper_tts._model_path != model_path:
        print(f"[PIPER] Loading model: {model_path}", flush=True)
        use_cuda = False
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                use_cuda = True
        except Exception:
            use_cuda = False
        try:
            piper_tts._voice = PiperVoice.load(model_path, use_cuda=use_cuda)
            piper_tts._model_path = model_path
            print(f"[PIPER] CUDA={use_cuda}", flush=True)
        except Exception as e:
            if use_cuda:
                print(f"[PIPER] CUDA load failed ({e}), retrying on CPU...", flush=True)
                piper_tts._voice = PiperVoice.load(model_path, use_cuda=False)
                piper_tts._model_path = model_path
            else:
                raise
    voice = piper_tts._voice
    import wave
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    print(f"[PIPER] Saved: {output_path}", flush=True)
    return output_path


def enhance_piper_audio(input_path: Path, output_path: Path) -> bool:
    print(f"[ENHANCE] Improving Piper audio quality...", flush=True)
    filters = ["lowpass=f=4000"]
    filter_str = ",".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", filter_str,
        "-ar", "48000",
        "-resampler", "soxr",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if result.returncode != 0:
        print(f"[ENHANCE] Failed: {result.stderr}", flush=True)
        return False
    print(f"[ENHANCE] Audio enhanced: {output_path}", flush=True)
    return True


def piper_available(model_path: str) -> bool:
    try:
        from piper import PiperVoice
        return Path(model_path).exists()
    except ImportError:
        return False


_F5_DEFAULT_CKPT = os.environ.get("VTS_F5_CKPT", "")
_F5_DEFAULT_REF_AUDIO = os.environ.get("VTS_F5_REF_AUDIO", "")
_F5_DEFAULT_REF_TEXT = "Zem je čerstvá, včera v noci som si to poriadne nevšímol."
_F5_DEFAULT_PYTHON = os.environ.get("VTS_F5_PYTHON", "")  # F5 odstránený, ale konstanta zachovaná pre backward compat


def f5_tts(
    text: str,
    output_path: Path,
    ckpt_path: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    f5_python: str | None = None,
    device: str = "cuda",
    seed: int = 42,
) -> Path:
    """F5-TTS SK inference cez f5tts_env subprocess. use_ema=False (kritické pre fine-tune).

    Default: SK fine-tuned ckpt z f5_sk_v1_full/model_last.pt (auto-update keď trening pokračuje).
    """
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path  = ckpt_path  or _F5_DEFAULT_CKPT
    ref_audio  = ref_audio  or _F5_DEFAULT_REF_AUDIO
    ref_text   = ref_text   or _F5_DEFAULT_REF_TEXT
    f5_python  = f5_python  or _F5_DEFAULT_PYTHON

    if not Path(ckpt_path).exists():
        raise RuntimeError(f"[F5] Checkpoint missing: {ckpt_path}")
    if not Path(ref_audio).exists():
        raise RuntimeError(f"[F5] Reference audio missing: {ref_audio}")
    if not Path(f5_python).exists():
        raise RuntimeError(f"[F5] Python env missing: {f5_python} (install f5_tts package)")

    print(f"[F5] Generating: {text[:50]}...", flush=True)

    runner = (
        "import sys, json\n"
        "args = json.loads(sys.stdin.read())\n"
        "from f5_tts.api import F5TTS\n"
        "f5 = F5TTS(model='F5TTS_v1_Base', ckpt_file=args['ckpt'], device=args['device'], use_ema=False)\n"
        "f5.infer(ref_file=args['ref_audio'], ref_text=args['ref_text'], gen_text=args['text'],\n"
        "         file_wave=args['out'], seed=args['seed'], show_info=lambda *a, **kw: None)\n"
        "print('OK')\n"
    )
    payload = {
        "text": text, "out": str(output_path), "ckpt": ckpt_path,
        "ref_audio": ref_audio, "ref_text": ref_text, "device": device, "seed": seed,
    }
    proc = subprocess.run(
        [f5_python, "-c", runner],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True, timeout=600,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"[F5] inference failed: {err}")
    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError(f"[F5] output missing/empty: {output_path}")
    # Post-processing: rovnaký TTS-CLEAN pass ako Chatterbox (highpass + denoise + EQ).
    # F5 výstup občas obsahuje hiss / prefix šum z reference audio prompt.
    cleanup_generated_tts_audio(
        output_path,
        short_text=_is_short_tts_reply(text),
        aggressive=False,
    )
    print(f"[F5] Saved: {output_path} ({output_path.stat().st_size//1024} KB)", flush=True)
    return output_path


def f5_available(ckpt_path: str | None = None, f5_python: str | None = None) -> bool:
    ckpt = ckpt_path or _F5_DEFAULT_CKPT
    py   = f5_python  or _F5_DEFAULT_PYTHON
    return Path(ckpt).exists() and Path(py).exists()


def piper_rs_tts(
    cli_path: str,
    config_path: str,
    model_path: str,
    text: str,
    output_path: Path,
    ort_dll_path: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = None
    if ort_dll_path:
        ort_path = Path(ort_dll_path)
        if ort_path.exists():
            env = dict(**os.environ)
            env["ORT_DYLIB_PATH"] = str(ort_path)
            env["PATH"] = str(ort_path.parent) + os.pathsep + env.get("PATH", "")
            cuda_path = env.get("CUDA_PATH")
            if cuda_path:
                cuda_bin = Path(cuda_path) / "bin"
                if cuda_bin.exists():
                    env["PATH"] = str(cuda_bin) + os.pathsep + env["PATH"]
    cmd = [
        cli_path,
        config_path,
        text,
        "-m",
        model_path,
        "-o",
        str(output_path),
    ]
    result = subprocess.run(cmd, env=env, timeout=FFMPEG_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError("piper-rs failed (non-zero exit)")
    if not output_path.exists():
        raise RuntimeError(f"piper-rs output missing: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Chatterbox TTS helpers
# ---------------------------------------------------------------------------

def _smart_chunk_for_tts(text: str, max_chars: int = 300) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    import re
    sentences = re.split(r'([.!?]+\s+)', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        punctuation = sentences[i+1] if i+1 < len(sentences) else ""
        full_sentence = sentence + punctuation
        if not current_chunk:
            current_chunk = full_sentence
        elif len(current_chunk) + len(full_sentence) <= max_chars:
            current_chunk += full_sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = full_sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            words = chunk.split()
            sub_chunk = ""
            for word in words:
                if not sub_chunk:
                    sub_chunk = word
                elif len(sub_chunk) + 1 + len(word) <= max_chars:
                    sub_chunk += " " + word
                else:
                    final_chunks.append(sub_chunk)
                    sub_chunk = word
            if sub_chunk:
                final_chunks.append(sub_chunk)
    return final_chunks


# ---------------------------------------------------------------------------
# Chatterbox Multilingual TTS
# ---------------------------------------------------------------------------

def chatterbox_tts(
    text: str,
    output_path: Path,
    model_path: str,
    device: str = "cuda",
    voice_id: str = None,
    use_torch_compile: bool = True,
    audio_prompt_path: str = None,
    safe_mode: bool = False,
    ultra_safe: bool = False,
    exaggeration: float = -1.0,
    cfg_weight: float = -1.0,
    temperature: float = -1.0,
    chunk_overrides: list[dict] | None = None,
    chunk_prompts: list[str | None] | None = None,
) -> Path:
    import sys
    import torch
    import torchaudio as ta
    from safetensors.torch import load_file as load_safetensors

    if device == "cuda" and not torch.cuda.is_available():
        print(f"[CHATTERBOX] CUDA not available, using CPU", flush=True)
        device = "cpu"

    _vt_dir = Path(__file__).parent.parent
    chatterbox_path = _vt_dir.parent / "Ai" / "chatterbox_git" / "src"
    if not chatterbox_path.exists():
        chatterbox_path = _vt_dir.parent / "chatterbox_git" / "src"
    if str(chatterbox_path) not in sys.path:
        sys.path.insert(0, str(chatterbox_path))

    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as e:
        raise RuntimeError(f"Chatterbox multilingual not available: {e}")

    logging.getLogger("chatterbox.models.t3.inference.alignment_stream_analyzer").setLevel(logging.ERROR)
    logging.getLogger("chatterbox.models.t3.t3").setLevel(logging.WARNING)  # suppress "✅ EOS token detected!" info spam
    print(f"[CHATTERBOX] Generating: {text[:50]}...", flush=True)

    def _oom_check(exc: Exception) -> bool:
        return isinstance(exc, getattr(torch, "OutOfMemoryError", type(None))) \
            or isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", type(None))) \
            or "out of memory" in str(exc).lower()

    if not hasattr(chatterbox_tts, '_model') or chatterbox_tts._device != device:
        # Po predchádzajúcom zlyhaní nenačítavaj znova (šetrí čas a HF requesty)
        if getattr(chatterbox_tts, '_model_load_failed', False):
            raise RuntimeError(
                "[CHATTERBOX] Model sa predtým nepodarilo načítať — preskakujem (OOM/RAM)"
            )

        # Skontroluj voľnú VRAM pred načítaním modelu (~3.5 GB pre Chatterbox)
        if device == "cuda" and torch.cuda.is_available():
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            _free_gb = (torch.cuda.get_device_properties(0).total_memory
                        - torch.cuda.memory_allocated(0)) / 1e9
            if _free_gb < 3.5:
                chatterbox_tts._model_load_failed = True
                raise RuntimeError(
                    f"[CHATTERBOX] Nedostatok VRAM pre load modelu"
                    f" ({_free_gb:.1f} GB voľné, potrebné ~3.5 GB)"
                )

        # Skontroluj voľnú RAM (model potrebuje ~3 GB RAM počas načítavania)
        try:
            import psutil as _psutil
            _free_ram_gb = _psutil.virtual_memory().available / 1e9
            if _free_ram_gb < 3.0:
                chatterbox_tts._model_load_failed = True
                raise RuntimeError(
                    f"[CHATTERBOX] Nedostatok RAM pre load modelu"
                    f" ({_free_ram_gb:.1f} GB voľné, potrebné ~3 GB)"
                )
        except ImportError:
            pass

        print(f"[CHATTERBOX] Loading multilingual model on {device}...", flush=True)
        try:
            chatterbox_tts._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        except Exception as _load_err:
            chatterbox_tts._model_load_failed = True
            if _oom_check(_load_err):
                raise RuntimeError(
                    f"[CHATTERBOX] OOM pri načítaní modelu na {device}: {_load_err}"
                ) from _load_err
            raise

        model_file = _resolve_chatterbox_asset(model_path)
        if model_file.exists():
            print(f"[CHATTERBOX] Loading Czech weights: {model_file}", flush=True)
            t3_state = load_safetensors(str(model_file), device="cpu")
            try:
                text_emb = t3_state.get("text_emb.weight")
                text_head = t3_state.get("text_head.weight")
                if text_emb is not None and text_head is not None:
                    target_vocab = chatterbox_tts._model.t3.text_emb.weight.shape[0]
                    src_vocab = text_emb.shape[0]
                    if src_vocab != target_vocab:
                        if src_vocab > target_vocab:
                            print(f"[CHATTERBOX] Vocab mismatch: {src_vocab}->{target_vocab}, trimming.", flush=True)
                            t3_state["text_emb.weight"] = text_emb[:target_vocab, :]
                            t3_state["text_head.weight"] = text_head[:target_vocab, :]
                        else:
                            print(f"[CHATTERBOX] Vocab mismatch: {src_vocab}->{target_vocab}, padding.", flush=True)
                            pad_rows = target_vocab - src_vocab
                            emb_pad = text_emb.mean(dim=0, keepdim=True).repeat(pad_rows, 1)
                            head_pad = text_head.mean(dim=0, keepdim=True).repeat(pad_rows, 1)
                            t3_state["text_emb.weight"] = torch.cat([text_emb, emb_pad], dim=0)
                            t3_state["text_head.weight"] = torch.cat([text_head, head_pad], dim=0)
            except Exception as e:
                print(f"[CHATTERBOX] WARNING: Vocab resize failed: {e}", flush=True)

            chatterbox_tts._model.t3.load_state_dict(t3_state, strict=True)
            if device != "cpu":
                try:
                    chatterbox_tts._model.t3.to(device)
                except Exception as _to_err:
                    if _oom_check(_to_err):
                        raise RuntimeError(
                            f"[CHATTERBOX] OOM pri t3.to({device}): {_to_err}"
                        ) from _to_err
                    raise
            chatterbox_tts._model.t3.eval()
            print(f"[CHATTERBOX] Czech model loaded successfully!", flush=True)
        else:
            print(f"[CHATTERBOX] WARNING: Czech weights not found at {model_file}", flush=True)

        if use_torch_compile and hasattr(torch, 'compile') and device == "cuda":
            try:
                print(f"[CHATTERBOX] Applying torch.compile() optimization...", flush=True)
                chatterbox_tts._model.t3 = torch.compile(chatterbox_tts._model.t3, mode="default", dynamic=True)
                chatterbox_tts._model.s3gen = torch.compile(chatterbox_tts._model.s3gen, mode="default", dynamic=True)
                print(f"[CHATTERBOX] torch.compile() applied (first inference will be slower)", flush=True)
                chatterbox_tts._compiled = True
            except Exception as e:
                print(f"[CHATTERBOX] torch.compile() failed (non-critical): {e}", flush=True)
                chatterbox_tts._compiled = False
        else:
            chatterbox_tts._compiled = False

        chatterbox_tts._device = device
        chatterbox_tts._model.t3.eval()
        # Nový model nemá žiadne conds — vymažeme cache aby sa znovu pripravili
        chatterbox_tts._audio_prompt_path = None

    model = chatterbox_tts._model

    text_stripped = text.strip()
    if not text_stripped:
        print("[CHATTERBOX] WARNING: Empty text, padding with silence", flush=True)
        silence = torch.zeros(1, int(model.sr * 0.1))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(output_path), silence, model.sr)
        return output_path

    text_to_generate = _stabilize_short_tts_text(text_stripped)
    if len(text_stripped) < 3:
        print(f"[CHATTERBOX] Short-text stabilization: '{text_stripped}' -> '{text_to_generate}'", flush=True)
    elif text_to_generate != text_stripped:
        print(f"[CHATTERBOX] WARNING: Short text, stabilized as: '{text_to_generate}'", flush=True)

    if voice_id:
        language_id = voice_id
        print(f"[CHATTERBOX] Using voice: {voice_id}", flush=True)
    else:
        # t3_cs.safetensors — Czech fine-tune používal language_id="pl" (Polish token id=717)
        #   → pri inferencii treba "pl" inak nestabilná generácia + predčasný EOS
        # t3_finetuned_v*.safetensors — SK fine-tune bol trénovaný BEZ language_id
        #   → pri inferencii treba None, inak model dostane poľský prefix a ignoruje mäkčene
        resolved_model_path = _resolve_chatterbox_asset(model_path)
        model_filename = resolved_model_path.name if resolved_model_path else ""
        if "finetuned" in model_filename or "sk" in model_filename.lower():
            language_id = "sk"  # SK fine-tuned v0.6+: trénovaný s MTL tokenizérom, language_id="sk"
            print(f"[CHATTERBOX] language_id=sk (SK fine-tuned model — MTL tokenizer)", flush=True)
        elif resolved_model_path.exists():
            language_id = "pl"  # CS base model
        else:
            language_id = "en"

    cached_prompt = getattr(chatterbox_tts, '_audio_prompt_path', None)
    if ultra_safe:
        if cached_prompt is not None:
            print(f"[CHATTERBOX] Ultra-safe: clearing voice clone conditionals", flush=True)
            if hasattr(model, 'conds'):
                model.conds = None
            chatterbox_tts._audio_prompt_path = None
    elif audio_prompt_path:
        resolved_prompt = _resolve_chatterbox_asset(audio_prompt_path)
        if resolved_prompt.exists() and cached_prompt != str(resolved_prompt):
            print(f"[CHATTERBOX] Preparing voice clone from: {resolved_prompt}", flush=True)
            model.prepare_conditionals(str(resolved_prompt))
            chatterbox_tts._audio_prompt_path = str(resolved_prompt)
        audio_prompt_path = None
    else:
        # No prompt requested — clear any cached conditionals from a previous segment
        if cached_prompt is not None:
            if hasattr(model, 'conds'):
                model.conds = None
            chatterbox_tts._audio_prompt_path = None

    if ultra_safe:
        gen_exaggeration = 0.30
        gen_cfg_weight = 0.40
        gen_temperature = 0.45
        gen_top_p = 0.92
        gen_rep_penalty = 1.25
        max_chunk_size = 90
    elif safe_mode:
        gen_exaggeration = 0.45
        gen_cfg_weight = 0.50
        gen_temperature = 0.60
        gen_top_p = 0.92
        gen_rep_penalty = 1.25
        max_chunk_size = 110
    else:
        # CONFIG_ANTICUT — overené na regression testoch (v2.1), najlepší výsledok:
        # 9/13 stable_ok (69%), prirodzený hlas, minimum artefaktov
        gen_exaggeration = 0.50
        gen_cfg_weight = 0.50
        gen_temperature = 0.60
        gen_top_p = 0.92
        gen_rep_penalty = 1.25
        max_chunk_size = 180

    # Manual overrides from CLI args (--chatterbox_exaggeration etc.)
    if exaggeration >= 0.0:
        gen_exaggeration = exaggeration
    if cfg_weight >= 0.0:
        gen_cfg_weight = cfg_weight
    if temperature >= 0.0:
        gen_temperature = temperature
    if exaggeration >= 0.0 or cfg_weight >= 0.0 or temperature >= 0.0:
        print(f"[CHATTERBOX] Override params: exaggeration={gen_exaggeration:.2f} cfg_weight={gen_cfg_weight:.2f} temperature={gen_temperature:.2f}", flush=True)

    print(f"[CHATTERBOX] Params: exag={gen_exaggeration} cfg={gen_cfg_weight} temp={gen_temperature} top_p={gen_top_p} rep_pen={gen_rep_penalty}", flush=True)

    MAX_CHUNK_SIZE = max_chunk_size
    if len(text_to_generate) > MAX_CHUNK_SIZE:
        chunks = _smart_chunk_for_tts(text_to_generate, max_chars=MAX_CHUNK_SIZE)
        print(f"[CHATTERBOX] Text too long ({len(text_to_generate)} chars), splitting into {len(chunks)} chunks", flush=True)

        wav_chunks = []
        silence_pause = torch.zeros(1, int(model.sr * 0.1))

        for i, chunk_text in enumerate(chunks, 1):
            ci = i - 1  # 0-based index

            # Layer 4: per-chunk energy overrides
            if chunk_overrides and ci < len(chunk_overrides):
                ov = chunk_overrides[ci]
                c_exag = float(ov.get("exaggeration", gen_exaggeration))
                c_cfg = float(ov.get("cfg_weight", gen_cfg_weight))
                c_temp = float(ov.get("temperature", gen_temperature))
            else:
                c_exag, c_cfg, c_temp = gen_exaggeration, gen_cfg_weight, gen_temperature

            # Layer 5: per-chunk reference prompt
            c_prompt = audio_prompt_path
            if chunk_prompts and ci < len(chunk_prompts) and chunk_prompts[ci]:
                c_prompt = chunk_prompts[ci]

            print(
                f"[CHATTERBOX] Chunk {i}/{len(chunks)}"
                f" exag={c_exag:.2f} cfg={c_cfg:.2f} temp={c_temp:.2f}"
                f"{' [custom_ref]' if c_prompt != audio_prompt_path else ''}"
                f": '{chunk_text[:60]}'",
                flush=True,
            )
            try:
                with torch.inference_mode():
                    chunk_wav = model.generate(
                        chunk_text, language_id=language_id,
                        audio_prompt_path=c_prompt,
                        exaggeration=c_exag, cfg_weight=c_cfg,
                        temperature=c_temp, top_p=gen_top_p,
                        repetition_penalty=gen_rep_penalty,
                    )
                    chunk_wav_cpu = chunk_wav.detach().cpu()
                    wav_chunks.append(chunk_wav_cpu)
                    if i < len(chunks):
                        wav_chunks.append(silence_pause)
                    if device == "cuda" and i % 5 == 0:
                        torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                print(f"[CHATTERBOX] OOM in chunk {i}: {e}", flush=True)
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                # Retry ten istý chunk raz po cache clear
                try:
                    with torch.inference_mode():
                        chunk_wav = model.generate(
                            chunk_text, language_id=language_id,
                            audio_prompt_path=c_prompt,
                            exaggeration=c_exag, cfg_weight=c_cfg,
                            temperature=c_temp, top_p=gen_top_p,
                            repetition_penalty=gen_rep_penalty,
                        )
                        chunk_wav_cpu = chunk_wav.detach().cpu()
                        wav_chunks.append(chunk_wav_cpu)
                        if i < len(chunks):
                            wav_chunks.append(silence_pause)
                        print(f"[CHATTERBOX] OOM chunk {i} retry OK", flush=True)
                except Exception as _e2:
                    print(f"[CHATTERBOX] OOM chunk {i} retry failed: {_e2}, inserting silence", flush=True)
                    wav_chunks.append(torch.zeros(1, int(model.sr * 0.1)))
            except Exception as e:
                print(f"[CHATTERBOX] ERROR in chunk {i}: {e}", flush=True)
                wav_chunks.append(torch.zeros(1, int(model.sr * 0.1)))

        if wav_chunks:
            wav = torch.cat(wav_chunks, dim=1)
            del wav_chunks
        else:
            wav = torch.zeros(1, int(model.sr * 0.1))
    else:
        print(f"[CHATTERBOX] TTS text: '{text_to_generate}'", flush=True)
        _retry_seeds = [42, 0, 123, 7, 99]
        _min_dur = max(0.3, len(text_to_generate) / 25.0)
        wav = None
        for _attempt, _seed in enumerate(_retry_seeds):
            try:
                torch.manual_seed(_seed)
                with torch.inference_mode():
                    _wav = model.generate(text_to_generate, language_id=language_id, audio_prompt_path=audio_prompt_path,
                                          exaggeration=gen_exaggeration, cfg_weight=gen_cfg_weight,
                                          temperature=gen_temperature, top_p=gen_top_p,
                                          repetition_penalty=gen_rep_penalty)
                _dur = _wav.shape[-1] / model.sr
                if _dur >= _min_dur:
                    wav = _wav
                    if _attempt > 0:
                        print(f"[CHATTERBOX] Retry {_attempt} OK (seed={_seed}, dur={_dur:.1f}s)", flush=True)
                    break
                print(f"[CHATTERBOX] Too short (seed={_seed}, dur={_dur:.2f}s < {_min_dur:.2f}s), retrying...", flush=True)
                if _attempt == len(_retry_seeds) - 1:
                    wav = _wav  # use best available
            except IndexError as e:
                print(f"[CHATTERBOX] ERROR: Text tokenization failed: {e}", flush=True)
                break
            except torch.cuda.OutOfMemoryError as e:
                print(f"[CHATTERBOX] OOM (attempt {_attempt}): {e}", flush=True)
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                if _attempt < len(_retry_seeds) - 1:
                    print("[CHATTERBOX] OOM recovery: retrying after cache clear...", flush=True)
                    continue
                # Všetky pokusy zlyhali na OOM
                print("[CHATTERBOX] OOM: all retries exhausted, saving silence", flush=True)
                break
            except Exception as e:
                print(f"[CHATTERBOX] ERROR: Generation failed: {e}", flush=True)
                break
        if wav is None:
            wav = torch.zeros(1, int(model.sr * 0.1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output_path), wav, model.sr)
    del wav
    if device == "cuda":
        torch.cuda.empty_cache()

    cleanup_generated_tts_audio(
        output_path,
        short_text=_is_short_tts_reply(text_stripped),
        aggressive=False,
    )
    print(f"[CHATTERBOX] Saved: {output_path}", flush=True)
    if device == "cuda":
        torch.cuda.empty_cache()
    return output_path


def chatterbox_turbo_tts(
    text: str,
    output_path: Path,
    model_path: str,
    device: str = "cuda",
    audio_prompt_path: str = None,
    temperature: float = 0.8,
    top_k: int = 1000,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> Path:
    """Chatterbox TURBO TTS — 1-step MeanFlow decoder, ~3.4× rýchlejší ako MTL."""
    import sys
    import torch
    import torchaudio as ta
    from safetensors.torch import load_file as load_safetensors

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    _vt_dir = Path(__file__).parent.parent
    chatterbox_path = _vt_dir.parent / "Ai" / "chatterbox_git" / "src"
    if not chatterbox_path.exists():
        chatterbox_path = _vt_dir.parent / "chatterbox_git" / "src"
    if str(chatterbox_path) not in sys.path:
        sys.path.insert(0, str(chatterbox_path))

    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError as e:
        raise RuntimeError(f"Chatterbox Turbo not available: {e}")

    logging.getLogger("chatterbox.models.t3.inference.alignment_stream_analyzer").setLevel(logging.ERROR)
    logging.getLogger("chatterbox.models.t3.t3").setLevel(logging.WARNING)
    print(f"[TURBO] Generating: {text[:50]}...", flush=True)

    TURBO_BASE_DIR = "/home/vojtech/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd"
    sk_fallback_ref = _resolve_chatterbox_asset("models/chatterbox/reference_sk.wav", fallback_name="reference_sk.wav")

    if not hasattr(chatterbox_turbo_tts, '_model') or chatterbox_turbo_tts._device != device:
        print(f"[TURBO] Loading Turbo model on {device}...", flush=True)
        chatterbox_turbo_tts._model = ChatterboxTurboTTS.from_local(TURBO_BASE_DIR, device=device)

        model_file = _resolve_chatterbox_asset(model_path)
        if model_file.exists():
            print(f"[TURBO] Loading SK fine-tuned weights: {model_file}", flush=True)
            t3_state = load_safetensors(str(model_file), device="cpu")
            chatterbox_turbo_tts._model.t3.load_state_dict(t3_state, strict=False)
            if device != "cpu":
                chatterbox_turbo_tts._model.t3.to(device)
            chatterbox_turbo_tts._model.t3.eval()
            print(f"[TURBO] SK Turbo model loaded!", flush=True)
        else:
            print(f"[TURBO] WARNING: weights not found at {model_path}, using base model", flush=True)

        chatterbox_turbo_tts._device = device
        chatterbox_turbo_tts._audio_prompt_path = None

    model = chatterbox_turbo_tts._model

    text_stripped = _normalize_sk_turbo_text(text)
    if not text_stripped:
        silence = torch.zeros(1, int(model.sr * 0.1))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(output_path), silence, model.sr)
        return output_path

    text_to_generate = _stabilize_short_tts_text(text_stripped)
    if text_to_generate != text.strip():
        print(f"[TURBO] Normalized text: {text_to_generate[:90]}...", flush=True)

    # Voice clone conditionals — preferujeme SK reference, nie EN built-in conds.pt
    cached_prompt = getattr(chatterbox_turbo_tts, '_audio_prompt_path', None)
    resolved_prompt = _resolve_chatterbox_asset(audio_prompt_path) if audio_prompt_path else None
    effective_prompt = resolved_prompt if (resolved_prompt and resolved_prompt.exists()) else (sk_fallback_ref if sk_fallback_ref.exists() else None)
    if effective_prompt and cached_prompt != str(effective_prompt):
        print(f"[TURBO] Preparing voice from: {effective_prompt}", flush=True)
        model.prepare_conditionals(str(effective_prompt), exaggeration=0.0)
        chatterbox_turbo_tts._audio_prompt_path = str(effective_prompt)
    elif not effective_prompt:
        print(f"[TURBO] WARNING: No SK reference found, using built-in EN voice", flush=True)

    # Chunk long texts
    MAX_CHUNK = 180
    if len(text_to_generate) > MAX_CHUNK:
        chunks = _smart_chunk_for_tts(text_to_generate, max_chars=MAX_CHUNK)
        print(f"[TURBO] Splitting into {len(chunks)} chunks", flush=True)
        wav_chunks = []
        silence_pause = torch.zeros(1, int(model.sr * 0.1))
        for i, chunk in enumerate(chunks, 1):
            try:
                with torch.inference_mode():
                    w = model.generate(chunk, temperature=temperature, top_k=top_k,
                                       top_p=top_p, repetition_penalty=repetition_penalty)
                wav_chunks.append(w.detach().cpu())
                if i < len(chunks):
                    wav_chunks.append(silence_pause)
            except Exception as e:
                print(f"[TURBO] ERROR chunk {i}: {e}", flush=True)
                wav_chunks.append(torch.zeros(1, int(model.sr * 0.1)))
        wav = torch.cat(wav_chunks, dim=1) if wav_chunks else torch.zeros(1, int(model.sr * 0.1))
    else:
        _min_dur = max(0.3, len(text_to_generate) / 25.0)
        wav = None
        for _attempt in range(3):
            try:
                torch.manual_seed([42, 0, 123][_attempt])
                with torch.inference_mode():
                    _wav = model.generate(text_to_generate, temperature=temperature, top_k=top_k,
                                          top_p=top_p, repetition_penalty=repetition_penalty)
                _dur = _wav.shape[-1] / model.sr
                if _dur >= _min_dur:
                    wav = _wav
                    if _attempt > 0:
                        print(f"[TURBO] Retry {_attempt} OK (dur={_dur:.1f}s)", flush=True)
                    break
                print(f"[TURBO] Too short (dur={_dur:.2f}s < {_min_dur:.2f}s), retrying...", flush=True)
                if _attempt == 2:
                    wav = _wav
            except Exception as e:
                print(f"[TURBO] ERROR: {e}", flush=True)
                break
        if wav is None:
            wav = torch.zeros(1, int(model.sr * 0.1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output_path), wav, model.sr)
    del wav
    if device == "cuda":
        torch.cuda.empty_cache()
    cleanup_generated_tts_audio(
        output_path,
        short_text=_is_short_tts_reply(text_stripped),
        aggressive=False,
    )
    print(f"[TURBO] Saved: {output_path}", flush=True)
    return output_path


def chatterbox_available(model_path: str) -> bool:
    import sys
    _vt_dir = Path(__file__).parent.parent
    chatterbox_path = _vt_dir.parent / "Ai" / "chatterbox_git" / "src"
    if not chatterbox_path.exists():
        chatterbox_path = _vt_dir.parent / "chatterbox_git" / "src"
    if str(chatterbox_path) not in sys.path:
        sys.path.insert(0, str(chatterbox_path))
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Chatterbox ONNX TTS
# ---------------------------------------------------------------------------

_chatterbox_onnx_instance = None


def _select_onnx_device() -> str:
    if torch.cuda.is_available():
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def chatterbox_onnx_tts(
    text: str,
    output_path: Path,
    precision: str = "fp32",
    device: str | None = None,
    reference_voice: Path | str | None = None,
) -> Path:
    global _chatterbox_onnx_instance
    import soundfile as sf

    if device is None:
        device = _select_onnx_device()
    if device is None:
        device = "cpu"

    model_dir = SCRIPT_DIR / "models" / "chatterbox-onnx-cs" / "onnx"
    lm_by_precision = {
        "fp32": model_dir / "language_model.onnx",
        "fp16": model_dir / "language_model_fp16.onnx",
        "q4": model_dir / "language_model_q4.onnx",
        "q4f16": model_dir / "language_model_q4f16.onnx",
    }
    selected_precision = precision
    if not lm_by_precision.get(selected_precision, lm_by_precision["fp32"]).exists():
        print(f"[ONNX-TTS] WARNING: precision '{selected_precision}' model not found, falling back to fp32", flush=True)
        selected_precision = "fp32"

    if _chatterbox_onnx_instance is None:
        from chatterbox_onnx import ChatterboxONNX
        print(f"[ONNX-TTS] Loading Chatterbox ONNX ({selected_precision}) on {device}...", flush=True)
        _chatterbox_onnx_instance = ChatterboxONNX(
            model_dir=str(SCRIPT_DIR / "models" / "chatterbox-onnx-cs"),
            precision=selected_precision,
            device=device,
        )
        print(f"[ONNX-TTS] Model loaded!", flush=True)

    tts = _chatterbox_onnx_instance

    text_stripped = text.strip()
    if len(text_stripped) < 3:
        print(f"[ONNX-TTS] WARNING: Text too short, generating silence", flush=True)
        silence = np.zeros(int(TTS_SR * 0.1))
        write_wav_safe(output_path, silence, TTS_SR)
        return output_path

    print(f"[ONNX-TTS] Generating: {text_stripped[:50]}...", flush=True)

    try:
        onnx_chunks = _smart_chunk_for_tts(text_stripped, max_chars=120)
        wav_parts = []
        pause = np.zeros(int(TTS_SR * 0.06), dtype=np.float32)

        def _max_tokens_for_chunk(chunk_text: str) -> int:
            words = len(chunk_text.split())
            chars = len(chunk_text)
            estimate = int(22 + words * 8 + chars * 0.20)
            return max(48, min(140, estimate))

        for idx, chunk_text in enumerate(onnx_chunks, 1):
            max_new_tokens = _max_tokens_for_chunk(chunk_text)
            print(f"[ONNX-TTS] Chunk {idx}/{len(onnx_chunks)} (max_new_tokens={max_new_tokens})", flush=True)
            chunk_wav = tts.generate(
                chunk_text,
                exaggeration=0.45,
                max_new_tokens=max_new_tokens,
                show_progress=False,
                voice_path=str(reference_voice) if reference_voice else None,
            )
            chunk_wav = np.asarray(chunk_wav, dtype=np.float32).squeeze()
            expected_sec = max(0.45, len(chunk_text) / 14.0)
            max_sec = expected_sec * 1.9
            max_samples = int(max_sec * TTS_SR)
            if chunk_wav.size > max_samples:
                chunk_wav = chunk_wav[:max_samples]
            wav_parts.append(chunk_wav)
            if idx < len(onnx_chunks):
                wav_parts.append(pause)

        wav = np.concatenate(wav_parts) if wav_parts else np.zeros(int(TTS_SR * 0.1), dtype=np.float32)
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        print(f"[ONNX-TTS] Raw shape={wav.shape}, dtype={wav.dtype}, "
              f"min={wav.min():.6f}, max={wav.max():.6f}, "
              f"samples={len(wav)}, duration={len(wav)/TTS_SR:.2f}s", flush=True)
        peak = np.abs(wav).max()
        if peak > 1e-6:
            target = 0.9
            wav = wav * (target / peak)
            print(f"[ONNX-TTS] Normalized: peak {peak:.6f} → {target}", flush=True)
        else:
            print(f"[ONNX-TTS] WARNING: Audio is silent (peak={peak})!", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav_safe(output_path, wav, TTS_SR)
        print(f"[ONNX-TTS] Saved: {output_path}", flush=True)
    except Exception as e:
        print(f"[ONNX-TTS] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        silence = np.zeros(int(TTS_SR * 0.1))
        write_wav_safe(output_path, silence, TTS_SR)

    return output_path


def chatterbox_onnx_available() -> bool:
    onnx_dir = SCRIPT_DIR / "models" / "chatterbox-onnx-cs" / "onnx"
    return onnx_dir.exists() and (onnx_dir / "language_model.onnx").exists()


# ---------------------------------------------------------------------------
# XTTS v2 TTS (cez subprocess v separátnom xtts_env prostredí)
# ---------------------------------------------------------------------------

def xtts_tts(
    text: str,
    output_path: Path,
    lang: str = "cs",
    speaker_wav: str = "",
    speaker: str = "Damien Black",
    pitch_shift: float = 0.0,
    device: str = "cuda",
    xtts_python: str = "",
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
) -> Path:
    """Synthesize text using XTTS v2 via subprocess (separate xtts_env)."""
    import subprocess as _sp

    script = Path(__file__).parent / "xtts_worker.py"
    if not script.exists():
        raise RuntimeError(f"[XTTS] Worker script not found: {script}")
    if not xtts_python:
        # Auto-detect xtts_env python — env override → paths.PATHS → standard locations
        env_override = os.environ.get("VTS_XTTS_PYTHON", "")
        if env_override and Path(env_override).exists():
            xtts_python = env_override
        else:
            candidates = []
            try:
                from paths import PATHS
                candidates.append(Path(PATHS.python_xtts))
            except Exception:
                pass
            candidates += [
                Path.home() / "miniforge3/envs/xtts_env/bin/python",
                Path("/opt/conda/envs/xtts_env/bin/python"),
            ]
            for c in candidates:
                if c.exists():
                    xtts_python = str(c)
                    break
        if not xtts_python:
            raise RuntimeError("[XTTS] xtts_env python not found. Set VTS_XTTS_PYTHON or xtts_python.")

    cmd = [
        xtts_python, str(script),
        "--text", text,
        "--output", str(output_path),
        "--lang", lang,
        "--device", device,
        "--model_name", model_name,
    ]
    if speaker_wav and Path(speaker_wav).exists():
        cmd += ["--speaker_wav", speaker_wav]
    elif speaker:
        cmd += ["--speaker", speaker]
    if pitch_shift != 0.0:
        cmd += ["--pitch_shift", str(pitch_shift)]

    result = _sp.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"[XTTS] Worker failed (exit {result.returncode})")

    if not output_path.exists():
        raise RuntimeError(f"[XTTS] Output not created: {output_path}")

    return output_path


def xtts_available() -> bool:
    if os.environ.get("VTS_XTTS_PYTHON") and Path(os.environ["VTS_XTTS_PYTHON"]).exists():
        return True
    candidates = [
        Path.home() / "miniforge3/envs/xtts_env/bin/python",
        Path("/opt/conda/envs/xtts_env/bin/python"),
    ]
    try:
        from paths import PATHS
        candidates.insert(0, Path(PATHS.python_xtts))
    except Exception:
        pass
    return any(c.exists() for c in candidates)
# 2026-05-05: Test ukázal výrazne lepšie skóre než F5 alebo Chatterbox SK
# pre slovenčinu (1760 vs 1572 vs 1268 z 34 testov), preto pridaný ako engine.
# ---------------------------------------------------------------------------
# Dynamické cesty — auto-detect cez paths.PATHS, override cez env premenné.
# Užívateľ môže nastaviť: VTS_OMNIVOICE_PYTHON, VTS_OMNIVOICE_REF_AUDIO, VTS_VOICES_DIR
def _detect_omnivoice_python() -> str:
    """OmniVoice je nainštalovaný v f5tts_env (alebo separátnom omnivoice_env).
    Detection priority: env override → paths.python_omnivoice → standard locations.
    """
    env_override = os.environ.get("VTS_OMNIVOICE_PYTHON", "")
    if env_override and Path(env_override).exists():
        return env_override
    try:
        from paths import PATHS
        candidate = Path(PATHS.python_omnivoice)
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    # Fallbacks — skús niekoľko bežných env mien (omnivoice → f5tts → fish_env)
    for env_name in ("omnivoice_env", "f5tts_env", "fish_env"):
        for root in (Path.home() / "miniforge3", Path("/opt/conda"),
                     Path("/mnt/tts_data/miniforge3")):
            c = root / "envs" / env_name / "bin" / "python"
            if c.exists():
                # Ešte raz over že má omnivoice modul
                try:
                    import subprocess as _sp
                    r = _sp.run([str(c), "-c", "import omnivoice"],
                                capture_output=True, timeout=10)
                    if r.returncode == 0:
                        return str(c)
                except Exception:
                    pass
    return "python3"  # last resort — bude hlásiť chybu pri prvom volani

def _detect_default_voices_dir() -> Path:
    env_override = os.environ.get("VTS_VOICES_DIR", "")
    if env_override:
        return Path(env_override)
    # Hľadaj voices/ relatívne k tomuto súboru
    here = Path(__file__).resolve().parent.parent  # scripts/ → repo root
    if (here / "voices").is_dir():
        return here / "voices"
    return Path.home() / ".local" / "share" / "videotranslator" / "voices"

_OMNIVOICE_DEFAULT_REPO = "k2-fsa/OmniVoice"
_OMNIVOICE_VOICES_DIR = _detect_default_voices_dir()
_OMNIVOICE_DEFAULT_REF_AUDIO = str(_OMNIVOICE_VOICES_DIR / "juraj_sk_studio.wav")
_OMNIVOICE_DEFAULT_REF_TEXT = "Náhrávka vznikla na základe predlohy publikovanej knižne vo vydavateľstve Slovart v roku dvetisíc dvadsaťpäť."
_OMNIVOICE_DEFAULT_PYTHON = _detect_omnivoice_python()
# Inference defaults (z testov: num_step=64, guidance=2.5 ≈ +5dB SNR vs default)
_OMNIVOICE_DEFAULT_NUM_STEP = 64
_OMNIVOICE_DEFAULT_GUIDANCE = 2.5
_OMNIVOICE_DEFAULT_POSITION_TEMP = 3.0
_OMNIVOICE_DEFAULT_LANGUAGE = "sk"
# Issue #144 fix: max 100 chars per chunk redukuje crackling z 65% na 17%.
# 2026-05-07: bumpnuté na 999 — všetok náš chunking vypnutý, OmniVoice dostane celý
# segment textu a rozhoduje sám podľa audio_chunk_threshold=30s. Naše chunkovanie
# pri 100/200 znakoch spôsobovalo "stop+restart" cuts → user vníma ako odsekávanie.
# Pre tech video so 7-8s segmentmi je každý pod 30s threshold → bez internal chunking.
_OMNIVOICE_MAX_CHUNK_CHARS = 999


_OMNIVOICE_VALID_INSTRUCT = frozenset({
    "male", "female",
    "child", "teenager", "young adult", "middle-aged", "elderly",
    "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
    "whisper",
    "american accent", "british accent", "australian accent", "chinese accent",
    "canadian accent", "indian accent", "korean accent", "portuguese accent",
    "russian accent", "japanese accent",
})


def omnivoice_validate_instruct(instruct: str) -> str:
    """Filter out unsupported instruct tokens. Vracia comma-separated string len validných."""
    if not instruct:
        return ""
    items = [s.strip().lower() for s in instruct.split(",")]
    valid = [s for s in items if s in _OMNIVOICE_VALID_INSTRUCT]
    return ", ".join(valid)


def omnivoice_make_expressive(text: str) -> str:
    """Pridaj čiarky pred slová 'teda/napríklad/ale/však/totiž' pre expresívnu prozódiu."""
    import re
    pattern = r'\s+(teda|napríklad|ale|však|totiž|pretože|preto|však)\b'
    # Insert čiarka len ak ešte nie je
    return re.sub(
        pattern,
        lambda m: (', ' + m.group(1)) if not m.string[max(0, m.start()-1):m.start()].rstrip().endswith(',') else m.group(0),
        text,
        flags=re.IGNORECASE,
    )


def omnivoice_speak_symbols(text: str) -> str:
    """Pre-processing pred OmniVoice TTS. Eliminuje text triggery ktoré model interpretuje
    ako "trailing off" alebo neukončenú vetu (= fade-out audio output, vnímané ako odsekávanie).

    Verified fixes (run 13-14 + Strategy A/B/C test 2026-05-07):
    1. Tech symboly: model ignoruje plus-plus, hash v texte → C++ znelo C.
    2. Trailing čiarka: model interpretuje ako veta pokračuje → fade-out koniec.
    3. Ellipsis tri bodky / single ellipsis char: trailing-off prosody, audio sa stratí.
    4. Úvodzovky (slovenské aj anglické): rozbíjajú prozódiu pri opakovaní.

    POZN.: volá sa iba v omnivoice_tts(). Chatterbox má pôvodný flow nezmenený.
    """
    import re
    out = text

    # 1) Tech symboly (verified C++ test — bez expansion model hovorí len "C")
    out = re.sub(r'\bC\+\+', 'C plus plus', out)
    out = re.sub(r'\bc\+\+', 'C plus plus', out)
    out = re.sub(r'\bC#', 'C sharp', out)
    out = re.sub(r'\bF#', 'F sharp', out)
    out = re.sub(r'\.NET\b', 'dot NET', out)

    # 2) Úvodzovky NAJPRV — odstránené (TTS ich nečíta, ale rozbíjajú prozódiu).
    out = re.sub(r'[„""""\'\']', '', out)

    # 3) EM-DASH a EN-DASH → čiarka (verified 2026-05-08: "–" v texte spôsobí
    #    0.6-0.9s pauzu v generovanom audio v 2 z 14 problematických segmentov).
    out = out.replace('—', ',')  # em-dash U+2014
    out = out.replace('–', ',')  # en-dash U+2013

    # 4) Mid-sentence "..." → čiarka (krátka pauza, nie dlhá fade-off).
    #    Trailing "..." → bodka. Toto rieši 8+ segmentov s "trailing off" fade.
    out = out.replace('…', '.')
    # Mid-sentence (nasleduje neprázdne písmeno) → ", "
    out = re.sub(r'\.{3,}\s*([A-Za-zÁ-žÀ-ÿ])', r', \1', out)
    out = re.sub(r'(?:\.\s*){2,}\.', '.', out)
    out = re.sub(r'\.{2,}', '.', out)

    # 5) Cleanup: trailing čiarka → bodka.
    out = re.sub(r'\s+', ' ', out).strip()
    if out and out[-1] == ',':
        out = out[:-1] + '.'

    return out


def trim_trailing_silence_chunk(wav_path: Path, threshold_db: float = -45.0) -> bool:
    """Strategy B (verified Wyhrala v A/B/C teste 2026-05-07) — odstráni trailing silence
    z OmniVoice chunku. Krátke texty (Dovidenia, tie motory) majú 35-43% trailing silence
    kvôli minimálnej dur diffusion modelu (~1.5s). Post-trim zníži z 9/94 na 4/94 problémových.

    Použitie: per-chunk po OmniVoice generation, pred concat do segment WAV.
    """
    import subprocess as _sp
    if not wav_path.exists():
        return False
    tmp = wav_path.with_name(wav_path.stem + '_trim.wav')
    cmd = ['ffmpeg', '-y', '-i', str(wav_path),
           '-af', f'areverse,silenceremove=start_periods=1:start_duration=0.05:start_threshold={threshold_db}dB,areverse',
           '-ac', '1', '-ar', str(TTS_SR), '-c:a', 'pcm_s16le', str(tmp)]
    r = _sp.run(cmd, capture_output=True, timeout=30)
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1000:
        tmp.replace(wav_path)
        return True
    tmp.unlink(missing_ok=True)
    return False


def omnivoice_tts(
    text: str,
    output_path: Path,
    repo_or_path: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    omnivoice_python: str | None = None,
    language: str = "sk",
    num_step: int = _OMNIVOICE_DEFAULT_NUM_STEP,
    guidance_scale: float = _OMNIVOICE_DEFAULT_GUIDANCE,
    position_temperature: float = _OMNIVOICE_DEFAULT_POSITION_TEMP,
    instruct: str = "",
    speed: float = 1.0,
    expressive: bool = False,
) -> Path:
    """OmniVoice (k2-fsa) zero-shot TTS cez f5tts_env subprocess.

    Default model: k2-fsa/OmniVoice (Apache 2.0, 600+ jazykov).
    Default ref: juraj_sk_studio.wav (cleaned cez cleanup_reference_audio).

    Test matrix (Whisper QA, 34 SK testov, 2026-05-05):
        OmniVoice  → 1760 (avg 51.8/test, 0 negatívnych skóre)
        F5+smart   → 1572 (avg 46.2/test)
        Chatterbox → 1268 (avg 37.3/test, 4 negatívne skóre)
    """
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_or_path = repo_or_path or _OMNIVOICE_DEFAULT_REPO
    ref_audio = ref_audio or _OMNIVOICE_DEFAULT_REF_AUDIO
    omnivoice_python = omnivoice_python or _OMNIVOICE_DEFAULT_PYTHON

    if not Path(ref_audio).exists():
        raise RuntimeError(f"[OMNIVOICE] Reference audio missing: {ref_audio}")
    if not Path(omnivoice_python).exists():
        raise RuntimeError(f"[OMNIVOICE] Python interpreter missing: {omnivoice_python}")

    # ref_text musí zodpovedať ref_audio. Ak user dal custom ref_audio bez ref_text,
    # default text by spôsobil halucináciu (model dostane mismatched audio↔text).
    # Auto-transcribe cez Whisper. Pre default ref_audio použijeme známy default text.
    if not ref_text:
        if Path(ref_audio).resolve() == Path(_OMNIVOICE_DEFAULT_REF_AUDIO).resolve():
            ref_text = _OMNIVOICE_DEFAULT_REF_TEXT
        else:
            print(f"[OMNIVOICE] Custom ref_audio bez ref_text → auto-Whisper transcribe…", flush=True)
            try:
                ref_text = whisper_transcribe_ref_audio(ref_audio, language=language)
                print(f"[OMNIVOICE] Auto-transcribed ref_text: {ref_text[:80]}…", flush=True)
            except Exception as e:
                print(f"[OMNIVOICE] WARN: Whisper transcribe failed ({e}), padajúci back na default — TTS môže halucinovať!", flush=True)
                ref_text = _OMNIVOICE_DEFAULT_REF_TEXT

    # Subprocess runner — chunkuje text na <= 100 chars (issue #144 fix)
    runner = """
import sys, json, re, tempfile, subprocess
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

p = json.loads(sys.stdin.read())

def _chunk(text, max_len=100):
    out = []
    for para in re.split(r'\\n+', text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len:
            out.append(para); continue
        sentences = re.split(r'(?<=[.!?])\\s+', para)
        buf = ''
        for s in sentences:
            if len(buf) + len(s) + 1 <= max_len:
                buf = (buf + ' ' + s).strip()
            else:
                if buf: out.append(buf)
                if len(s) <= max_len:
                    buf = s
                else:
                    for sub in re.split(r',\\s*', s):
                        if buf and len(buf) + len(sub) + 2 <= max_len:
                            buf = buf + ', ' + sub
                        else:
                            if buf: out.append(buf)
                            buf = sub
        if buf: out.append(buf)
    return out

chunks = _chunk(p['text'], max_len=p['max_chunk'])
model = OmniVoice.from_pretrained(p['repo'], device_map='cuda:0', dtype=torch.float16)
cfg = OmniVoiceGenerationConfig(
    num_step=p['num_step'],
    guidance_scale=p['guidance_scale'],
    position_temperature=p['position_temperature'],
    postprocess_output=False,  # vypnuté: model interne trim long silences a môže odsekať pauzy
                                # medzi slovami / vetami → pôsobí "osekane". Naše post-processing
                                # (silence_repair s -60dB threshold) je bezpečnejšie.
)

# Save chunks to temp dir for post-processing (trim + crossfade)
out_path = Path(p['output_path'])
tmp_dir = out_path.with_name(out_path.stem + '_chunks_tmp')
tmp_dir.mkdir(exist_ok=True, parents=True)

chunk_files = []
for i, chunk in enumerate(chunks):
    kw = dict(text=chunk, language=p['language'],
              ref_audio=p['ref_audio'], ref_text=p['ref_text'],
              generation_config=cfg)
    if p.get('instruct'):
        kw['instruct'] = p['instruct']
    if p.get('speed') and p['speed'] != 1.0:
        kw['speed'] = p['speed']
    audio = model.generate(**kw)
    cf = tmp_dir / f'chunk_{i:04d}.wav'
    sf.write(str(cf), audio[0], 24000)
    chunk_files.append(cf)

# Trim silence + crossfade concat (post-processing)
import sys as _sys, importlib.util as _il
# Inline post-processing helpers (subprocess-safe)
import subprocess as _sp
TTS_SR_LOCAL = 24000

def _trim_edges(wav_path, threshold_db=-65.0):
    # MIN-CONSERVATIVE trim: silenceremove s peak detekciou pri -55dB chytala tiché spoluhlasky
    # (š/s/k/t fade-out na konci slov). Threshold znížený na -65 dB + RMS detekcia, len ak
    # sú aspoň 0.4s ticha (skutočne dlhá pauza). Inak audio prechádza bez zmeny.
    tmp = wav_path.with_name(wav_path.stem + '_t.wav')
    cmd = ['ffmpeg', '-y', '-i', str(wav_path),
           '-af', (
               # Leading silence trim — len 0.4s+ tichá medzera (skutočne ticho, nie quiet onset)
               f'silenceremove=start_periods=1:start_duration=0.4:start_threshold={threshold_db}dB,'
               # Trailing silence trim — cez reverse trick s rovnakou konzervatívnou hranicou
               f'areverse,silenceremove=start_periods=1:start_duration=0.4:start_threshold={threshold_db}dB,areverse,'
               # Prirodzený breath gap: 60ms začiatok, 40ms koniec
               f'adelay=60|60,apad=pad_dur=0.04'
           ),
           '-ac', '1', '-ar', str(TTS_SR_LOCAL), '-c:a', 'pcm_s16le', str(tmp)]
    r = _sp.run(cmd, capture_output=True, timeout=60)
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1000:
        tmp.replace(wav_path)

def _concat_simple(files, output):
    # Simple concat — žiadny crossfade. Crossfade=0.05s zlieval posledných 25ms predošlého
    # chunku s prvými 25ms ďalšieho → cut sound + premazaný začiatok slov pri chunk hraniciach.
    # Pri OmniVoice chunkoch (rozdelené na vetné hranice) je raw concat bezpečnejší.
    if len(files) == 1:
        import shutil; shutil.copy(str(files[0]), str(output)); return
    # Build concat list file (safer pre rôzne sample rates)
    list_path = output.with_suffix('.concatlist.txt')
    lines = [f"file '{f}'" for f in files]
    list_path.write_text(chr(10).join(lines) + chr(10))
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(list_path),
           '-ac', '1', '-ar', str(TTS_SR_LOCAL),
           '-c:a', 'pcm_s16le', str(output)]
    _sp.run(cmd, capture_output=True, timeout=300)
    list_path.unlink(missing_ok=True)

# Step 1: trim — minimálne, iba ak sú reálne 0.4s+ ticha
for cf in chunk_files:
    _trim_edges(cf, threshold_db=-65.0)

# Step 2: concat — bez crossfade (raw concat, zachová slovné hranice)
_concat_simple(chunk_files, out_path)

# Cleanup temp chunks
import shutil as _shutil
_shutil.rmtree(tmp_dir, ignore_errors=True)
print('OK', flush=True)
"""

    # Validate instruct + apply expressive preprocessing
    # Najprv nahradíme symboly (C++ → C plus plus) — model ich občas ignoruje (najmä
    # po prvom výskyte v texte). Potom expressive úpravy interpunkcie.
    final_text = omnivoice_speak_symbols(text)
    if expressive:
        final_text = omnivoice_make_expressive(final_text)
    final_instruct = omnivoice_validate_instruct(instruct) if instruct else ""

    payload = {
        "text": final_text,
        "output_path": str(output_path),
        "repo": repo_or_path,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "language": language,
        "num_step": num_step,
        "guidance_scale": guidance_scale,
        "position_temperature": position_temperature,
        "max_chunk": _OMNIVOICE_MAX_CHUNK_CHARS,
        "instruct": final_instruct,
        "speed": float(speed),
    }
    proc = subprocess.run(
        [omnivoice_python, "-c", runner],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True, timeout=600,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"[OMNIVOICE] inference failed: {err}")
    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError(f"[OMNIVOICE] output missing/empty: {output_path}")

    # Strategy B (verified A/B/C test 2026-05-07): post-trim trailing silence z OmniVoice
    # output. Krátke texty / texty končiace tichom majú 35-43% trailing silence kvôli
    # diffusion modelu (min ~1.5s dur). Trim zníži problémové segmenty z 9/94 na 4/94.
    # IBA pre OmniVoice — Chatterbox má svoje pôvodné post-processing, nemenené.
    try:
        trim_trailing_silence_chunk(output_path, threshold_db=-45.0)
    except Exception as _e:
        print(f"[OMNIVOICE] trailing-trim warning: {_e}", flush=True)

    print(f"[OMNIVOICE] Saved: {output_path} ({output_path.stat().st_size//1024} KB)", flush=True)
    return output_path


def trim_chunk_edges(
    wav_path: Path,
    silence_threshold_db: float = -55.0,
    min_silence_dur_s: float = 0.05,
    output_path: Path | None = None,
) -> bool:
    """Trim leading + trailing silence z TTS chunku in-place (alebo do output_path).

    Použité pre OmniVoice / F5 / Chatterbox chunky pred concat aby sa eliminovali
    artefakty: leading silence (~1s prefix), trailing silence (krátke pauzy >500ms
    medzi vetami pri concat).
    """
    if not wav_path.exists():
        return False
    out = output_path or wav_path
    tmp = wav_path.with_name(wav_path.stem + "_trimedge.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-af", (
            f"silenceremove=start_periods=1:start_duration={min_silence_dur_s}:"
            f"start_threshold={silence_threshold_db}dB,"
            f"areverse,"
            f"silenceremove=start_periods=1:start_duration={min_silence_dur_s}:"
            f"start_threshold={silence_threshold_db}dB,"
            f"areverse"
        ),
        "-ac", "1", "-ar", str(TTS_SR), "-c:a", "pcm_s16le",
        str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 1000:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


def concat_chunks_crossfade(
    chunk_files: list,
    output_path: Path,
    crossfade_dur_s: float = 0.05,
) -> bool:
    """Concat WAV súborov s acrossfade medzi každou dvojicou (eliminuje energy clicks na boundaries).

    Použité po `trim_chunk_edges()` - chunks majú tight edges a crossfade dáva
    smooth join bez click/pop artefaktov. Default 50ms crossfade je počuteľne
    plynulé bez efektu "rozplývania" reči.
    """
    if not chunk_files:
        return False
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(chunk_files) == 1:
        import shutil
        shutil.copy(str(chunk_files[0]), str(output_path))
        return True

    inputs = []
    for f in chunk_files:
        inputs.extend(["-i", str(f)])

    # Iteratívny acrossfade chain: [0][1]→[a1]; [a1][2]→[a2]; ...
    filter_parts = []
    last_label = "0"
    for i in range(1, len(chunk_files)):
        next_label = f"a{i}"
        prev = f"[{last_label}]" if last_label.startswith("a") else "[0]"
        filter_parts.append(
            f"{prev}[{i}]acrossfade=d={crossfade_dur_s}:c1=tri:c2=tri[{next_label}]"
        )
        last_label = next_label

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", "; ".join(filter_parts),
        "-map", f"[{last_label}]",
        "-ac", "1", "-ar", str(TTS_SR), "-c:a", "pcm_s16le",
        str(output_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000


def omnivoice_post_process(
    chunks_dir: Path,
    output_wav: Path,
    glob_pattern: str = "chunk_*.wav",
    silence_threshold_db: float = -55.0,
    crossfade_dur_s: float = 0.05,
) -> bool:
    """End-to-end post-processing pre OmniVoice chunked output.

    1. Trim leading + trailing silence z každého chunku (rieši 1.15s prefix + pauzy)
    2. Concat s crossfade 50ms (rieši energy jumps na hranicach chunkov)

    Vracia True ak úspech.
    """
    chunks_dir = Path(chunks_dir)
    output_wav = Path(output_wav)
    chunk_files = sorted(chunks_dir.glob(glob_pattern))
    if not chunk_files:
        return False

    for c in chunk_files:
        trim_chunk_edges(c, silence_threshold_db=silence_threshold_db)

    return concat_chunks_crossfade(chunk_files, output_wav, crossfade_dur_s=crossfade_dur_s)


def omnivoice_available(omnivoice_python: str | None = None) -> bool:
    py = omnivoice_python or _OMNIVOICE_DEFAULT_PYTHON
    if not Path(py).exists():
        return False
    try:
        proc = subprocess.run(
            [py, "-c", "import omnivoice; print('ok')"],
            capture_output=True, timeout=15,
        )
        return proc.returncode == 0 and b"ok" in proc.stdout
    except Exception:
        return False


# Cache transkriptov ref_audio — vždy v rovnakom voices/ adresári ako sám audio súbor
# (cache_file vie byť potom presunutý spolu s voices/ pri zálohe).
_REF_TEXT_CACHE_DIR = _OMNIVOICE_VOICES_DIR / ".ref_text_cache"


def whisper_transcribe_ref_audio(ref_audio_path: str, language: str = "sk") -> str:
    """Auto-transkribuj reference audio cez faster-whisper (cached).

    Vracia transcript ako string. Cache uložená v voices/.ref_text_cache/<hash>.txt.
    Whisper beží v subprocess (video_prekladac env), aby sa neflopol main pipeline.
    """
    ref_path = Path(ref_audio_path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Ref audio missing: {ref_audio_path}")

    # Cache key z file mtime + size + path
    import hashlib
    stat = ref_path.stat()
    cache_key = hashlib.md5(
        f"{ref_path.name}|{stat.st_size}|{int(stat.st_mtime)}|{language}".encode()
    ).hexdigest()
    _REF_TEXT_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    cache_file = _REF_TEXT_CACHE_DIR / f"{cache_key}.txt"
    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass  # ak corrupt, refresh

    # Subprocess s faster_whisper. Skús env názvy v poradí, fallback na current python.
    def _detect_whisper_python() -> str:
        env_override = os.environ.get("VTS_WHISPER_PYTHON", "")
        if env_override and Path(env_override).exists():
            return env_override
        try:
            from paths import PATHS
            for env_name in ("video_prekladac", "musetalk_env", "f5tts_env"):
                candidate = Path(PATHS.miniforge_root) / "envs" / env_name / "bin" / "python"
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass
        return sys.executable
    whisper_py = _detect_whisper_python()
    # Hľadaj NVIDIA CUDA libs v site-packages aktuálneho python env-u
    def _detect_cuda_lib() -> str:
        py_prefix = Path(whisper_py).resolve().parent.parent  # bin/python → env_root
        for pyver in ("python3.10", "python3.11", "python3.12"):
            sp = py_prefix / "lib" / pyver / "site-packages" / "nvidia"
            if sp.is_dir():
                return str(sp)
        return ""
    cuda_lib = _detect_cuda_lib()
    runner = """
import sys, os
from faster_whisper import WhisperModel
m = WhisperModel('large-v3', device='cuda', compute_type='float16')
segs, _ = m.transcribe(sys.argv[1], language=sys.argv[2], beam_size=5)
print(' '.join(s.text.strip() for s in segs))
"""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        (f"{cuda_lib}/cublas/lib:{cuda_lib}/cudnn/lib:" if cuda_lib else "") + env.get("LD_LIBRARY_PATH", "")
    )
    proc = subprocess.run(
        [whisper_py, "-c", runner, str(ref_path), language],
        capture_output=True, env=env, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Whisper transcribe failed: {proc.stderr.decode('utf-8', errors='replace')[-500:]}"
        )
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    if not text:
        raise RuntimeError("Whisper returned empty transcript")
    # Cache
    try:
        cache_file.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return text
