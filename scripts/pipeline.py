"""
VideoTranslator pipeline — independent orchestrator.
STT -> Translation -> TTS -> Audio assembly
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# Ensure scripts/ directory is on sys.path for sibling module imports
_SCRIPTS_DIR = Path(__file__).parent.resolve()
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_ROOT_DIR = _SCRIPTS_DIR.parent.resolve()
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import torch

from config import FFMPEG_TIMEOUT, FFPROBE_TIMEOUT, SCRIPT_DIR, TTS_SR, build_parser, target_lang_name
from linux_logic_guard import apply_linux_mixed_mode, normalize_linux_for_tts
from stt import (
    SQL_WHISPER_PROMPT,
    extract_audio_ffmpeg,
    fix_segments_transcription,
    get_audio_duration,
    merge_adjacent_segments,
    normalize_segments,
    split_long_segments,
    transcribe_whisper,
    transcribe_whisper_vad,
)
from translation import (
    apply_phonetic_respelling,
    normalize_camelcase_for_tts,
    free_madlad_model,
    free_multislav5lang_model,
    is_sql_heavy,
    normalize_sql_for_chatterbox_tts,
    normalize_sql_for_tts,
    split_sql_clauses,
    protect_phonetic_terms,
    restore_phonetic_terms,
    strip_guillemets,
    translate_batch_madlad_gemma_qa,
    translate_segment_with_google,
    translate_segment_with_llama,
    translate_segment_with_madlad,
    translate_segment_with_multislav5lang,
    refine_translation_with_llama,
    refine_fulldoc_with_llama,
    refine_fulldoc_with_ollama,
    refine_with_sk_corrector,
    apply_gender_fix_sk_feminine,
    apply_linux_source_guided_cleanup,
    cleanup_with_gemma,
    qa_check_tone,
    fix_sk_conditional_grammar,
    fix_sk_foreign_name_declension,
    fix_sk_sql_function_names,
    apply_sql_source_guided_cleanup,
    fix_sk_noun_declension,
    translate_full_document_with_translategemma,
    translate_fulldoc_with_gemma4_hf,
    translate_fulldoc_with_lmstudio,
    translate_segments_sliding_window,
)
from translategemma import translate_segment_with_translategemma, free_translategemma
from text_normalizer import normalize_for_tts
from phonetic_guard import needs_rewrite, shorten_text
from level4_audio_feedback import (
    audit_duration,
    expand_text_hint,
    find_split_index,
    tighten_text_for_retry,
    AUTO_SPLIT_THRESHOLD_S,
)
from tts import (
    chatterbox_tts,
    chatterbox_turbo_tts,
    extract_best_chatterbox_reference_from_segments,
    extract_best_chatterbox_voice_sample,
    extract_chatterbox_voice_sample,
    inspect_tts_chunk,
    prepare_chatterbox_reference_audio,
    prepare_s2_pro_reference,
    s2_pro_tts,
    write_silence_chunk,
)
from sk_glossary import apply_postfix
from audio_pipeline import (
    apply_noise_reduction,
    apply_speech_gain,
    apply_tts_master_chain,
    apply_voice_eq,
    assemble_timeline_ffmpeg,
    assemble_concat_style,
    concat_wavs,
    extract_music_demucs,
    extract_vocals_mdx,
    mix_audio_with_music,
    mix_with_backaudio,
    repair_silence_gaps,
    trim_and_smooth_chunk,
    trim_edge_silence_ffmpeg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def _log_memory(label: str = "") -> None:
    """Vypíše aktuálny stav RAM/SWAP/VRAM do logu. Varuje pri prekročení prahu."""
    if not _PSUTIL:
        return
    mem  = _psutil.virtual_memory()
    swap = _psutil.swap_memory()
    ram_pct  = mem.percent
    swap_pct = swap.percent
    ram_gb   = mem.used  / 1024**3
    ram_tot  = mem.total / 1024**3
    swap_gb  = swap.used  / 1024**3
    swap_tot = swap.total / 1024**3

    vram_str = ""
    try:
        import torch as _t
        if _t.cuda.is_available():
            props = _t.cuda.get_device_properties(0)
            vram_tot = props.total_memory / 1024**3
            vram_used = _t.cuda.memory_reserved(0) / 1024**3
            vram_pct  = vram_used / vram_tot * 100
            vram_str  = f" | VRAM {vram_used:.1f}/{vram_tot:.1f}GB ({vram_pct:.1f}%)"
            if vram_pct >= 95.0:
                logging.warning(f"[MEM] !! KRITICKÁ VRAM {vram_pct:.1f}% — modely môžu spillovať do RAM!")
            elif vram_pct >= 85.0:
                logging.warning(f"[MEM] VRAM WARN {vram_pct:.1f}%")
    except Exception:
        pass

    tag = f" [{label}]" if label else ""
    logging.info(f"[MEM]{tag} RAM {ram_gb:.1f}/{ram_tot:.1f}GB ({ram_pct:.1f}%) | SWAP {swap_gb:.1f}/{swap_tot:.1f}GB ({swap_pct:.1f}%){vram_str}")

    if ram_pct >= 90.0:
        logging.warning(f"[MEM] !! KRITICKÁ RAM {ram_pct:.1f}% — pipeline môže zlyhať alebo swapovať!")
    elif ram_pct >= 80.0:
        logging.warning(f"[MEM] RAM WARN {ram_pct:.1f}%")
    if swap_pct >= 80.0:
        logging.warning(f"[MEM] !! KRITICKÝ SWAP {swap_pct:.1f}% — systém je veľmi pomalý!")
    elif swap_pct >= 50.0:
        logging.warning(f"[MEM] SWAP WARN {swap_pct:.1f}%")


def _hash_tts_cache_payload(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _format_srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ollama_aux_url(generate_url: str, suffix: str) -> str:
    parts = urlsplit(generate_url)
    path = parts.path or ""
    if path.endswith("/api/generate"):
        path = path[: -len("/api/generate")] + suffix
    elif not path.endswith(suffix):
        path = path.rstrip("/") + suffix
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _unload_ollama_model(generate_url: str, model: str, timeout: int = 20) -> bool:
    try:
        import requests as _requests

        response = _requests.post(
            generate_url,
            json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        print(f"[OLLAMA] Unloaded model from VRAM: {model}", flush=True)
        return True
    except Exception as e:
        print(f"[OLLAMA] Unload skipped/failed for {model}: {e}", flush=True)
        return False


def _extract_segment_audio(src_wav: Path, start: float, end: float, out_path: Path, pad_s: float = 0.3) -> bool:
    """Extract audio segment [start-pad .. end+pad] from src_wav to out_path at 24kHz mono."""
    import subprocess
    t_start = max(0.0, start - pad_s)
    duration = (end + pad_s) - t_start
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t_start), "-t", str(duration),
             "-i", str(src_wav), "-ar", "24000", "-ac", "1", str(out_path)],
            check=True, capture_output=True, timeout=FFMPEG_TIMEOUT,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        print(f"[EMOTION] ffmpeg trim failed: {e}", flush=True)
        return False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _derive_style_transfer_params(
    src_wav: Path,
    start: float,
    end: float,
    scratch_dir: Path,
    seg_idx: int,
    base_exaggeration: float,
    base_cfg_weight: float,
    base_temperature: float,
    pad_s: float = 0.18,
) -> dict[str, float] | None:
    """Estimate speaking style from the original segment without cloning its voice.

    Used when the user selected an explicit custom WAV voice together with
    emotion transfer. In that case we keep the custom voice as the prompt and
    only modulate Chatterbox generation parameters from the source segment's
    energy/activity profile.
    """
    import numpy as np
    import librosa

    probe_wav = scratch_dir / f"style_probe_{seg_idx:04d}.wav"
    try:
        if not _extract_segment_audio(src_wav, start, end, probe_wav, pad_s=pad_s):
            return None
        y, sr = librosa.load(str(probe_wav), sr=16000, mono=True)
        if y.size < int(sr * 0.20):
            return None

        # Speech activity and energy are the most stable cheap signals we can
        # reuse here. We intentionally avoid using the full segment as a clone
        # prompt because that would also import speaker identity.
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        if rms.size == 0:
            return None

        activity_threshold = max(float(np.median(rms) * 1.35), 0.010)
        active = rms > activity_threshold
        if int(active.sum()) < 3:
            activity_threshold = max(float(np.percentile(rms, 60)), 0.007)
            active = rms > activity_threshold

        active_ratio = float(np.mean(active)) if active.size else 0.0
        rms_active = rms[active] if active.any() else rms
        mean_rms = float(np.mean(rms_active))
        dyn_rms = float(np.std(rms_active) / (mean_rms + 1e-6))

        energy_score = _clamp((mean_rms - 0.025) / 0.085, 0.0, 1.0)
        activity_score = _clamp((active_ratio - 0.12) / 0.55, 0.0, 1.0)
        dynamics_score = _clamp(dyn_rms / 0.65, 0.0, 1.0)
        style_score = _clamp(
            0.50 * energy_score + 0.30 * activity_score + 0.20 * dynamics_score,
            0.0,
            1.0,
        )

        base_exag = 0.50 if base_exaggeration < 0.0 else float(base_exaggeration)
        base_cfg = 0.50 if base_cfg_weight < 0.0 else float(base_cfg_weight)
        base_temp = 0.60 if base_temperature < 0.0 else float(base_temperature)

        # Pitch analýza — doplnok k energie
        pitch_stats = _extract_pitch_stats(src_wav, start, end, scratch_dir, seg_idx, pad_s=pad_s)
        pitch_score = 0.5  # neutral fallback
        if pitch_stats:
            # Pitch range: väčší rozsah = expresívnejší prejav
            # Typický rozsah pre expresívnu reč: 80–200 Hz, pre monotónnu: <40 Hz
            pitch_score = _clamp((pitch_stats["pitch_range_hz"] - 30) / 130, 0.0, 1.0)
            # Voiced ratio: viac znelých frames = súvislá, aktívna reč
            voiced_boost = _clamp((pitch_stats["voiced_ratio"] - 0.5) / 0.4, -0.2, 0.2)
            pitch_score = _clamp(pitch_score + voiced_boost, 0.0, 1.0)

        # Kombinovaný skóre: energia (35%) + pitch (45%) + dynamika (20%)
        # pitch má vyššiu váhu — priamo riadi intonáciu hlasu
        combined_score = _clamp(
            0.35 * style_score + 0.45 * pitch_score + 0.20 * dynamics_score,
            0.0, 1.0,
        )

        exag = _clamp(base_exag + (combined_score - 0.5) * 0.38, 0.22, 0.80)
        cfg = _clamp(base_cfg + (0.5 - combined_score) * 0.14, 0.30, 0.82)
        temp = _clamp(base_temp + (dynamics_score - 0.5) * 0.10 + (pitch_score - 0.5) * 0.10, 0.42, 0.88)

        result = {
            "exaggeration": round(exag, 3),
            "cfg_weight": round(cfg, 3),
            "temperature": round(temp, 3),
            "style_score": round(style_score, 3),
            "energy_score": round(energy_score, 3),
            "activity_score": round(activity_score, 3),
            "dynamics_score": round(dynamics_score, 3),
            "pitch_score": round(pitch_score, 3),
            "combined_score": round(combined_score, 3),
        }
        if pitch_stats:
            result.update(pitch_stats)
        return result
    except Exception as e:
        print(f"[EMOTION-STYLE] Segment {seg_idx}: analysis failed: {e}", flush=True)
        return None
    finally:
        probe_wav.unlink(missing_ok=True)


def _guard_style_transfer_params(
    style_overrides: dict[str, float] | None,
    *,
    tgt_lang: str = "",
    has_explicit_voice: bool = False,
) -> dict[str, float] | None:
    """Clamp style-transfer params for safer cross-lingual SK dubbing.

    Full emotion transfer from source audio is helpful, but on Slovak dubbing it
    can easily over-drive Chatterbox into foreign prosody, slower pacing, and
    lexical drift. Keep the effect, just soften it.
    """
    if not style_overrides:
        return style_overrides

    out = dict(style_overrides)
    if (tgt_lang or "").lower() != "sk":
        return out

    # Cross-lingual Slovak dubbing is much more stable with milder style
    # transfer. Explicit custom voices need even tighter caps because their
    # pacing is already typically slower than the source sample.
    if has_explicit_voice:
        exag_low, exag_high = 0.28, 0.50
        cfg_low, cfg_high = 0.42, 0.56
        temp_low, temp_high = 0.42, 0.50
    else:
        exag_low, exag_high = 0.30, 0.55
        cfg_low, cfg_high = 0.42, 0.60
        temp_low, temp_high = 0.42, 0.54

    if "exaggeration" in out:
        out["exaggeration"] = round(_clamp(float(out["exaggeration"]), exag_low, exag_high), 3)
    if "cfg_weight" in out:
        out["cfg_weight"] = round(_clamp(float(out["cfg_weight"]), cfg_low, cfg_high), 3)
    if "temperature" in out:
        out["temperature"] = round(_clamp(float(out["temperature"]), temp_low, temp_high), 3)

    return out


def _extract_pause_map(
    src_wav: Path,
    start: float,
    end: float,
    scratch_dir: Path,
    seg_idx: int,
    min_pause_s: float = 0.25,
    pad_s: float = 0.18,
) -> list[float]:
    """Vráti relatívne pozície páuz (0.0–1.0) vnútri segmentu.

    Používa VAD cez RMS energiu na originálnom segmente.
    Hodnota 0.35 = pauza nastala v 35% trvania segmentu.
    Toto umožní vložiť pauzy do preloženého textu na zodpovedajúcich miestach.
    """
    try:
        import numpy as np
        import librosa

        probe_wav = scratch_dir / f"pause_probe_{seg_idx:04d}.wav"
        if not _extract_segment_audio(src_wav, start, end, probe_wav, pad_s=pad_s):
            return []
        try:
            y, sr = librosa.load(str(probe_wav), sr=16000, mono=True)
            if y.size < int(sr * 0.3):
                return []

            hop = 256
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
            threshold = max(float(np.percentile(rms, 20)), 0.004)
            silent = rms < threshold

            # Nájdi skupiny tichých frames → pauzy
            pauses: list[float] = []
            in_silence = False
            silence_start = 0
            total_frames = len(silent)

            for frame_idx, is_silent in enumerate(silent):
                if is_silent and not in_silence:
                    in_silence = True
                    silence_start = frame_idx
                elif not is_silent and in_silence:
                    in_silence = False
                    silence_dur_s = (frame_idx - silence_start) * hop / sr
                    if silence_dur_s >= min_pause_s:
                        pause_center = (silence_start + frame_idx) / 2
                        relative_pos = round(pause_center / max(total_frames, 1), 3)
                        # Ignoruj pauzy na okrajoch (prvých/posledných 8% = padding artefakty)
                        if 0.08 < relative_pos < 0.92:
                            pauses.append(relative_pos)

            return pauses
        finally:
            probe_wav.unlink(missing_ok=True)
    except Exception as e:
        print(f"[PAUSE-MAP] Segment {seg_idx}: failed: {e}", flush=True)
        return []


def _inject_pauses(text: str, pause_positions: list[float]) -> str:
    """Vloží pauzy (čiarky / tri bodky) do textu podľa relatívnych pozícií.

    Rozdelí text na slová a vloží pauzy na zodpovedajúce miesta.
    Krátkej pauze (1 pozícia) = čiarka, dlhšej (≥2 blízke) = tri bodky.
    Existujúca interpunkcia sa neprepíše.
    """
    if not pause_positions or not text.strip():
        return text

    words = text.split()
    n = len(words)
    if n < 3:
        return text

    result = list(words)
    for pos in sorted(set(pause_positions)):
        word_idx = max(1, min(int(pos * n), n - 2))
        word = result[word_idx - 1]
        # Nevkladaj ak slovo už má interpunkciu na konci
        if not word[-1] in ".,:;!?…":
            result[word_idx - 1] = word + ","

    return " ".join(result)


def _extract_pitch_stats(
    src_wav: Path,
    start: float,
    end: float,
    scratch_dir: Path,
    seg_idx: int,
    pad_s: float = 0.18,
) -> dict[str, float] | None:
    """Extrahuje pitch štatistiky (medián, rozsah) z originálneho segmentu.

    Používa librosa.pyin — robustnejší ako yin pre rečový signál.
    Vráti None ak sa analýza nepodarí alebo segment je príliš tichý.
    """
    try:
        import numpy as np
        import librosa

        probe_wav = scratch_dir / f"pitch_probe_{seg_idx:04d}.wav"
        if not _extract_segment_audio(src_wav, start, end, probe_wav, pad_s=pad_s):
            return None
        try:
            y, sr = librosa.load(str(probe_wav), sr=16000, mono=True)
            if y.size < int(sr * 0.4):
                return None

            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=60.0, fmax=500.0, sr=sr,
                frame_length=2048, hop_length=256,
            )
            voiced = f0[voiced_flag] if voiced_flag.any() else None
            if voiced is None or len(voiced) < 10:
                return None

            pitch_median = float(np.median(voiced))
            pitch_p10 = float(np.percentile(voiced, 10))
            pitch_p90 = float(np.percentile(voiced, 90))
            pitch_range = pitch_p90 - pitch_p10
            # voiced_ratio: koľko percent frames má pitch (expresívnosť reči)
            voiced_ratio = float(voiced_flag.mean())

            return {
                "pitch_median_hz": round(pitch_median, 1),
                "pitch_range_hz": round(pitch_range, 1),
                "voiced_ratio": round(voiced_ratio, 3),
            }
        finally:
            probe_wav.unlink(missing_ok=True)
    except Exception as e:
        print(f"[PITCH] Segment {seg_idx}: failed: {e}", flush=True)
        return None


def _extract_energy_contour(
    src_wav: Path,
    start: float,
    end: float,
    n_chunks: int,
    scratch_dir: Path,
    seg_idx: int,
    base_exag: float,
    base_cfg: float,
    base_temp: float,
) -> list[dict[str, float]]:
    """Vráti zoznam per-chunk parametrov odvodzených z energy kontúry originálu.

    Segment sa rozdelí na n_chunks časových okien. Pre každé okno sa zmeria
    RMS energia a dynamika → upravia sa exaggeration/cfg_weight/temperature.

    Výstup: list dĺžky n_chunks, každý prvok {exaggeration, cfg_weight, temperature}.
    Pri zlyhaní: fallback na globálne parametre pre všetky chunky.
    """
    fallback = [{"exaggeration": base_exag, "cfg_weight": base_cfg, "temperature": base_temp}] * n_chunks
    if n_chunks <= 1:
        return fallback

    try:
        import numpy as np
        import librosa

        probe_wav = scratch_dir / f"energy_probe_{seg_idx:04d}.wav"
        if not _extract_segment_audio(src_wav, start, end, probe_wav, pad_s=0.05):
            return fallback
        try:
            y, sr = librosa.load(str(probe_wav), sr=16000, mono=True)
            if y.size < int(sr * 0.3):
                return fallback

            samples_per_chunk = len(y) // n_chunks
            result = []

            for ci in range(n_chunks):
                c_start = ci * samples_per_chunk
                c_end = c_start + samples_per_chunk if ci < n_chunks - 1 else len(y)
                chunk_y = y[c_start:c_end]

                if len(chunk_y) < 256:
                    result.append({"exaggeration": base_exag, "cfg_weight": base_cfg, "temperature": base_temp})
                    continue

                rms = librosa.feature.rms(y=chunk_y, frame_length=512, hop_length=128)[0]
                if rms.size == 0:
                    result.append({"exaggeration": base_exag, "cfg_weight": base_cfg, "temperature": base_temp})
                    continue

                mean_rms = float(np.mean(rms))
                dyn_rms = float(np.std(rms) / (mean_rms + 1e-6))

                energy_score = _clamp((mean_rms - 0.020) / 0.090, 0.0, 1.0)
                dynamics_score = _clamp(dyn_rms / 0.70, 0.0, 1.0)
                chunk_score = _clamp(0.65 * energy_score + 0.35 * dynamics_score, 0.0, 1.0)

                exag = _clamp(base_exag + (chunk_score - 0.5) * 0.28, 0.20, 0.88)
                cfg = _clamp(base_cfg + (0.5 - chunk_score) * 0.10, 0.35, 0.85)
                temp = _clamp(base_temp + (dynamics_score - 0.5) * 0.08, 0.42, 0.82)
                result.append({
                    "exaggeration": round(exag, 3),
                    "cfg_weight": round(cfg, 3),
                    "temperature": round(temp, 3),
                    "energy_score": round(energy_score, 3),
                    "dynamics_score": round(dynamics_score, 3),
                })

            return result
        finally:
            probe_wav.unlink(missing_ok=True)
    except Exception as e:
        print(f"[ENERGY-CONTOUR] Segment {seg_idx}: failed: {e}", flush=True)
        return fallback


def _relative_dur_limit(seg_dur_s: float) -> float:
    """Relatívny hard limit na duration drift podľa dĺžky segmentu.

    Krátke segmenty sú pri dabingu kritickejšie — aj 120ms drift môže
    zničiť lipsync. Dlhé segmenty tolerujú viac.

      do 1.5 s  → max 120 ms
      1.5–4.0 s → lineárne 120–300 ms
      nad 4.0 s → max 500 ms
    """
    if seg_dur_s <= 1.5:
        return 0.12
    if seg_dur_s <= 4.0:
        return round(0.12 + (seg_dur_s - 1.5) / (4.0 - 1.5) * (0.30 - 0.12), 3)
    return 0.50


def _postproc_quality_score(
    metrics: dict,
    dur_limit: float = 0.50,
    seg_dur_s: float = 0.0,
) -> tuple[float, str]:
    """Vypočíta skóre kvality post-processingu a rozhodne o fallbacku.

    dur_limit: externý hard limit (z --postproc_dur_limit).
    seg_dur_s: dĺžka segmentu — ak > 0, použije sa relatívny limit
               (_relative_dur_limit) namiesto pevného dur_limit, ak je prísnejší.

    Skóre 0.0–1.0:
      >= 0.70 → "pause+f0"   (plný post-processing OK)
      >= 0.45 a dur OK → "pause_only"  (F0 transfer škodí, pauzy OK)
      <  0.45 → "raw"        (post-processing zhoršil výstup)

    Trestné body:
      dur drift > efektívny limit   → -0.50
      dur drift > 60% limitu        → -0.20
      dur drift > 30% limitu        → -0.10
      rms drift > 4 dB              → -0.30
      rms drift > 2 dB              → -0.10
      f0_var ratio > 2.5× AND abs delta > 15 Hz → -0.30 (vocoder artefakt)
      f0_var ratio > 1.8× AND abs delta > 8 Hz  → -0.10
    """
    score = 1.0
    dur_delta = abs(metrics.get("dur_delta_s", 0.0))
    rms_drift = abs(metrics.get("rms_drift_db", 0.0))
    f0_before = metrics.get("f0_var_before", 0.0)
    f0_after = metrics.get("f0_var_after", 0.0)

    # Relatívny limit — použije prísnejší z relatívneho a externého
    _eff_limit = dur_limit
    if seg_dur_s > 0.0:
        _rel = _relative_dur_limit(seg_dur_s)
        _eff_limit = min(dur_limit, _rel)

    # Duration drift — najkritickejší faktor pre dabing
    if dur_delta > _eff_limit:
        score -= 0.50
    elif dur_delta > _eff_limit * 0.60:
        score -= 0.20
    elif dur_delta > _eff_limit * 0.30:
        score -= 0.10

    # RMS drift — pyworld resyntéza môže zmeniť hlasitosť
    if rms_drift > 4.0:
        score -= 0.30
    elif rms_drift > 2.0:
        score -= 0.10

    # F0 variancia — explózia = vocoder artefakt (sykavky, transienty)
    # Fix: ratio je zradný pri before≈0 → pridaj absolútny delta floor
    _f0_abs_delta = f0_after - f0_before
    if f0_before > 8.0:  # floor: ignoruj monotónne segmenty
        if f0_after > f0_before * 2.5 and _f0_abs_delta > 15.0:
            score -= 0.30
        elif f0_after > f0_before * 1.8 and _f0_abs_delta > 8.0:
            score -= 0.10

    score = round(max(0.0, min(1.0, score)), 3)

    if score >= 0.70:
        decision = "pause+f0"
    elif score >= 0.45 and dur_delta <= _eff_limit:
        decision = "pause_only"
    else:
        decision = "raw"

    return score, decision


# ---------------------------------------------------------------------------
# Cross-segment deduplication (Level 14 – heuristický post-L11 krok)
# ---------------------------------------------------------------------------
_XSEG_INTRO_PATTERNS = [
    # "Druhá sa volá X", "Prvá sa volá X", "Ďalšia sa volá X"
    r"^(druhá?|prv[aá]|ďalš[ia]|tretia?|štvrtá?)\s+sa\s+vol[aá]\s+",
    r"^(druhá?|prv[aá]|ďalš[ia]|tretia?|štvrtá?)\s+vol[aá]\s+sa\s+",
    # "Nazýva sa X", "Volá sa X"
    r"^nazýva\s+sa\s+",
    r"^vol[aá]\s+sa\s+",
    # "To je X", "To je funkcia X"
    r"^to\s+je\s+(funkcia\s+|príkaz\s+|metóda\s+)?",
    # "Čiže X", "Teda X"
    r"^(čiže|teda|totiž)\s+",
    # "Je to X"
    r"^je\s+to\s+(funkcia\s+|príkaz\s+)?",
    # "Ide o X"
    r"^ide\s+o\s+(funkciu\s+|príkaz\s+)?",
    # "Hovoríme o X"
    r"^hovoríme\s+o\s+",
    # "Táto funkcia sa volá X"
    r"^t[aá]to\s+(funkcia|metóda|príkaz)\s+sa\s+vol[aá]\s+",
]

_XSEG_INTRO_MAX_WORDS = 8
_XSEG_INTRO_MAX_CHARS = 50
_XSEG_TAIL_WORDS = 5  # koľko posledných slov Seg N kontrolujeme


def _build_xseg_lexicon() -> dict[str, str]:
    """Vráti {lowercase_variant: canonical_key} pre SQL/tech termíny z PHONETIC_MAP."""
    try:
        from translation import PHONETIC_MAP as _PM
    except ImportError:
        try:
            import sys as _sys
            _sys.path.insert(0, str(__file__).replace("pipeline.py", ""))
            from translation import PHONETIC_MAP as _PM
        except Exception:
            return {}
    lexicon: dict[str, str] = {}
    for k, v in _PM.items():
        # Preskočiť bežné slová (len veľkými písmenami = SQL, alebo tech termíny)
        if k == k.upper() and len(k) >= 3:  # SQL keywords / functions
            canonical = k.upper()
            lexicon[k.lower()] = canonical
            lexicon[v.lower()] = canonical  # fonetický prepis
        elif k[0].isupper() and not k.isupper() and len(k) >= 4:  # Tech (Java, Docker...)
            canonical = k
            lexicon[k.lower()] = canonical
            lexicon[v.lower()] = canonical
    return lexicon


def _xseg_extract_tail_terms(text: str, lexicon: dict[str, str]) -> set[str]:
    """Vráti množinu canonical kľúčov z posledných N slov textu."""
    import re as _re
    words = _re.findall(r"[\w\-]+", text.lower())
    tail = words[-_XSEG_TAIL_WORDS:]
    found: set[str] = set()
    for w in tail:
        if w in lexicon:
            found.add(lexicon[w])
    # Tiež skús bigrams (napr. "IS NULL", "LEFT JOIN")
    for i in range(len(tail) - 1):
        bigram = tail[i] + " " + tail[i + 1]
        if bigram in lexicon:
            found.add(lexicon[bigram])
    return found


def _xseg_find_intro_sentence(text: str) -> str | None:
    """Ak text začína krátkou intro frázou (podľa vzorov), vráti ju; inak None."""
    import re as _re
    text = text.strip()
    # Rozdeľ na vety
    sentences = _re.split(r"(?<=[.!?…])\s+", text)
    if not sentences:
        return None
    first = sentences[0].strip()
    if not first:
        return None
    # Kontrola dĺžky
    word_count = len(first.split())
    if word_count > _XSEG_INTRO_MAX_WORDS or len(first) > _XSEG_INTRO_MAX_CHARS:
        return None
    # Kontrola intro vzoru
    first_lower = first.lower()
    for pattern in _XSEG_INTRO_PATTERNS:
        if _re.match(pattern, first_lower):
            return first
    return None


def _xseg_intro_contains_term(intro: str, canonical_terms: set[str], lexicon: dict[str, str]) -> bool:
    """Skontroluje, či intro veta obsahuje rovnaký canonical termín ako koniec predch. segmentu."""
    import re as _re
    words = _re.findall(r"[\w\-]+", intro.lower())
    for w in words:
        if w in lexicon and lexicon[w] in canonical_terms:
            return True
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        if bigram in lexicon and lexicon[bigram] in canonical_terms:
            return True
    return False


def _cross_segment_dedup_check(
    segments: list[dict],
    audit_log: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Odstráni intro vety v Seg N+1 ktoré opakujú kľúčový termín z konca Seg N.

    Podmienky (všetky musia platiť):
    1. Seg N końí termínom z SQL/tech lexikónu (PHONETIC_MAP)
    2. Seg N+1 začína krátkou intro frázou (≤8 slov / ≤50 znakov)
    3. Intro fráza matchuje SK intro vzor a obsahuje rovnaký canonical termín
    4. Po odstránení intro frázy zostane v N+1 aspoň 3 slová

    Vracia: (upravené segmenty, počet zmien)
    """
    import re as _re
    lexicon = _build_xseg_lexicon()
    if not lexicon:
        return segments, 0

    result = [dict(s) for s in segments]
    changed = 0

    for i in range(len(result) - 1):
        seg_n = result[i]
        seg_n1 = result[i + 1]

        text_n = (seg_n.get("text") or seg_n.get("translated") or "").strip()
        text_n1 = (seg_n1.get("text") or seg_n1.get("translated") or "").strip()
        if not text_n or not text_n1:
            continue

        # Krok 1: nájdi canonical termíny na konci Seg N
        tail_terms = _xseg_extract_tail_terms(text_n, lexicon)
        if not tail_terms:
            continue

        # Krok 2: nájdi intro vetu na začiatku Seg N+1
        intro = _xseg_find_intro_sentence(text_n1)
        if not intro:
            continue

        # Krok 3: intro musí obsahovať rovnaký termín
        if not _xseg_intro_contains_term(intro, tail_terms, lexicon):
            continue

        # Krok 4: po odstránení musí zostať aspoň 3 slová
        remainder = text_n1[len(intro):].lstrip(" ").lstrip()
        # Odstrán aj vedúcu interpunkciu a medzery
        remainder = _re.sub(r"^[\s\.,;:!?–—]+", "", remainder)
        if len(remainder.split()) < 3:
            continue

        # Prijmi zmenu
        original_n1 = text_n1
        result[i + 1] = dict(seg_n1)
        result[i + 1]["text"] = remainder
        if "translated" in result[i + 1]:
            result[i + 1]["translated"] = remainder
        changed += 1

        matched_terms = ", ".join(sorted(tail_terms))
        print(
            f"[XSEG-DEDUP] Seg {seg_n1.get('id', i+2)}: removed intro repetition "
            f"\"{intro}\" (term: {matched_terms})",
            flush=True,
        )
        if audit_log is not None:
            audit_log.append({
                "seg_n": seg_n.get("id", i + 1),
                "seg_n1": seg_n1.get("id", i + 2),
                "original_n1": original_n1,
                "new_n1": remainder,
                "removed_intro": intro,
                "matched_terms": matched_terms,
            })

    return result, changed


def _f0_blend_for_duration(dur_s: float, scale: float = 1.0) -> float:
    """Vráti blend_weight pre F0 transfer podľa dĺžky segmentu.

    < 1.2 s  → 0.0  (preskočiť — contour nestabilná)
    1.2–3.0s → lineárne 0.20–0.35  (short/medium — konzervatívne)
    3.0–6.0s → lineárne 0.35–0.45  (long — mierny transfer)
    > 6.0 s  → 0.45                 (max strop)

    scale: škálovací faktor (--f0_blend_scale). 1.0 = default. 1.44 ≈ staré agresívne váhy.
    """
    if dur_s < 1.2:
        return 0.0
    if dur_s <= 3.0:
        base = 0.20 + (dur_s - 1.2) / (3.0 - 1.2) * (0.35 - 0.20)
    elif dur_s <= 6.0:
        base = 0.35 + (dur_s - 3.0) / (6.0 - 3.0) * (0.45 - 0.35)
    else:
        base = 0.45
    return round(min(base * scale, 0.95), 3)


def _compute_postproc_metrics(
    wav_before: Path,
    wav_after: Path,
    seg_idx: int,
    n_pauses: int = 0,
) -> dict:
    """Vypočíta metriky pre porovnanie RAW a post-processed WAV.

    Vracia: dur_before_s, dur_after_s, dur_delta_s,
            f0_var_before, f0_var_after, f0_var_delta,
            rms_before, rms_after, rms_drift_db,
            n_pauses_injected
    """
    out: dict = {"seg": seg_idx, "n_pauses_injected": n_pauses}
    try:
        import numpy as np
        import soundfile as sf
        import librosa

        def _load(p: Path):
            y, sr = sf.read(str(p), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            return y.astype(np.float64), int(sr)

        y_b, sr_b = _load(wav_before)
        y_a, sr_a = _load(wav_after)

        out["dur_before_s"] = round(len(y_b) / sr_b, 3)
        out["dur_after_s"] = round(len(y_a) / sr_a, 3)
        out["dur_delta_s"] = round(out["dur_after_s"] - out["dur_before_s"], 3)

        # RMS
        rms_b = float(np.sqrt(np.mean(y_b ** 2))) + 1e-9
        rms_a = float(np.sqrt(np.mean(y_a ** 2))) + 1e-9
        out["rms_before"] = round(rms_b, 5)
        out["rms_after"] = round(rms_a, 5)
        out["rms_drift_db"] = round(20 * np.log10(rms_a / rms_b), 2)

        # F0 variance via pyin
        def _f0_var(y, sr):
            try:
                f0, vf, _ = librosa.pyin(y.astype(np.float32), fmin=60, fmax=500,
                                         sr=sr, hop_length=256, frame_length=2048)
                valid = f0[vf & np.isfinite(f0)]
                return round(float(np.std(valid)), 2) if len(valid) > 3 else 0.0
            except Exception:
                return 0.0

        out["f0_var_before"] = _f0_var(y_b, sr_b)
        out["f0_var_after"] = _f0_var(y_a, sr_a)
        out["f0_var_delta"] = round(out["f0_var_after"] - out["f0_var_before"], 2)

    except Exception as e:
        out["error"] = str(e)
    return out


def _lp_f0_trend(f0: "np.ndarray", voiced: "np.ndarray", window_frames: int) -> "np.ndarray":
    """Extrahuje low-frequency trend F0 kontúry sliding window priemerom cez voiced frames.

    Unvoiced frames (f0==0) sú interpolované pred filtrovaním a maskované späť po ňom.
    Výsledok: pomalý trend — veta stúpa/klesá — bez mikrovariácií slabík.
    """
    import numpy as np
    n = len(f0)
    if n == 0:
        return f0.copy()

    # Interpoluj cez unvoiced medzery pre plynulý filter
    f0_interp = f0.copy()
    indices = np.arange(n)
    voiced_idx = indices[voiced]
    if len(voiced_idx) < 2:
        return f0.copy()
    f0_interp = np.interp(indices, voiced_idx, f0[voiced_idx])

    # Sliding window average (uniform low-pass)
    w = max(3, window_frames | 1)  # musí byť odd
    pad = w // 2
    padded = np.pad(f0_interp, pad, mode="edge")
    trend = np.convolve(padded, np.ones(w) / w, mode="valid")[:n]

    # Maskovanie späť — unvoiced frames dostanú 0
    trend[~voiced] = 0.0
    return trend


def _apply_f0_transfer(
    src_wav: Path,
    src_start: float,
    src_end: float,
    tts_wav: Path,
    seg_idx: int,
    blend_weight: float = 0.30,
    lp_window_s: float = 0.60,
) -> bool:
    """Low-frequency F0 trend transfer — prenáša len pomalý intonačný tvar.

    Zmena oproti pôvodnej verzii:
    - NEPRENESIE celú raw F0 kontúru (zabíjala mikrovariácie TTS hlasu)
    - Low-pass filter (window ~0.6s) extrahuje len trend vety (stúpa/klesá)
    - Blend weight 0.20–0.45 (nie 0.65) — WORLD resyntéza sama pridáva stratu
    - Mikrovariácia (prirodzenosť hlasu) zostáva z TTS
    - Ak f0_var_after < f0_var_before × 0.80 → vráti False (transfer monotonizoval)
    """
    try:
        import numpy as np
        import librosa
        import pyworld as pw
        import soundfile as sf

        # --- Načítaj source segment ---
        probe_tmp = tts_wav.with_suffix(".f0src.wav")
        if not _extract_segment_audio(src_wav, src_start, src_end, probe_tmp, pad_s=0.05):
            return False
        try:
            y_src, _sr = librosa.load(str(probe_tmp), sr=None, mono=True)
            sr_src = int(_sr)
        finally:
            probe_tmp.unlink(missing_ok=True)

        # --- Načítaj TTS výstup ---
        y_tts, sr_tts = sf.read(str(tts_wav), dtype="float64")
        if y_tts.ndim > 1:
            y_tts = y_tts.mean(axis=1)
        if len(y_tts) < sr_tts * 0.15:
            return False

        if sr_src != sr_tts:
            y_src = librosa.resample(y_src.astype(np.float64), orig_sr=sr_src, target_sr=sr_tts)
        else:
            y_src = y_src.astype(np.float64)
        sr = sr_tts

        # --- HARVEST F0 (frame_period=5ms → 200 frames/s) ---
        _fp = 5.0
        _f0_src, _t_src = pw.harvest(y_src, sr, f0_floor=60.0, f0_ceil=600.0, frame_period=_fp)
        _f0_tts, _t_tts = pw.harvest(y_tts, sr, f0_floor=60.0, f0_ceil=600.0, frame_period=_fp)

        _voiced_src = _f0_src > 0
        _voiced_tts = _f0_tts > 0

        if _voiced_src.sum() < 8 or _voiced_tts.sum() < 8:
            return False

        # --- Low-pass trend extrakcia (window_frames = lp_window_s / frame_period_s) ---
        _frames_per_s = 1000.0 / _fp  # 200 frames/s
        _lp_win = max(5, int(lp_window_s * _frames_per_s) | 1)

        # Trend zo source (time-warped na dĺžku TTS)
        _n_src, _n_tts = len(_f0_src), len(_f0_tts)
        _src_interp_idx = np.linspace(0, _n_src - 1, _n_tts)
        _f0_src_warped = np.interp(_src_interp_idx, np.arange(_n_src), _f0_src)
        _voiced_src_w = np.interp(_src_interp_idx, np.arange(_n_src), _voiced_src.astype(float)) > 0.5

        _trend_src = _lp_f0_trend(_f0_src_warped, _voiced_src_w, _lp_win)
        _trend_tts = _lp_f0_trend(_f0_tts, _voiced_tts, _lp_win)

        # Normalizácia trendov na TTS mediánový pitch (zachovaj výšku hlasu TTS)
        _med_tts = float(np.median(_f0_tts[_voiced_tts]))
        _med_src_trend = float(np.median(_trend_src[_voiced_src_w])) if _voiced_src_w.any() else _med_tts
        _med_tts_trend = float(np.median(_trend_tts[_voiced_tts])) if _voiced_tts.any() else _med_tts

        # Correction = (source_trend - source_trend_median) * blend_weight
        # Aplikuj len tam kde TTS je voiced — zachovaj transienty a sykavky
        _f0_new = _f0_tts.copy()
        _std_tts = float(np.std(_f0_tts[_voiced_tts])) + 1e-6

        for fi in range(_n_tts):
            if not _voiced_tts[fi]:
                continue
            # Odchýlka source trendu od jeho mediánu (tvar bez absolútnej výšky)
            _src_trend_delta = _trend_src[fi] - _med_src_trend if _trend_src[fi] > 0 else 0.0
            # Blend: TTS micro-variácia + blend_weight × source trend tvar
            _tts_micro = _f0_tts[fi] - _trend_tts[fi] if _trend_tts[fi] > 0 else 0.0
            _new_trend = _trend_tts[fi] + blend_weight * _src_trend_delta
            _f0_new[fi] = max(60.0, _new_trend + _tts_micro)

        # --- WORLD re-synthesis ---
        _sp = pw.cheaptrick(y_tts, _f0_tts, _t_tts, sr)
        _ap = pw.d4c(y_tts, _f0_tts, _t_tts, sr)
        y_out = pw.synthesize(_f0_new, _sp, _ap, sr, frame_period=_fp)

        if len(y_out) < sr * 0.1:
            return False

        # RMS normalizácia
        _rms_in = float(np.sqrt(np.mean(y_tts ** 2))) + 1e-9
        _rms_out = float(np.sqrt(np.mean(y_out ** 2))) + 1e-9
        y_out = np.clip(y_out * (_rms_in / _rms_out), -1.0, 1.0)

        # Guardrail: ak transfer monotonizoval hlas → zahoď výsledok
        _f0_var_before = float(np.std(_f0_tts[_voiced_tts]))
        _f0_var_after  = float(np.std(_f0_new[_voiced_tts]))
        if _f0_var_after < _f0_var_before * 0.80:
            print(
                f"[F0-TRANSFER] Seg {seg_idx}: ZAHODENÉ — monotonizácia "
                f"f0_var {_f0_var_before:.1f}→{_f0_var_after:.1f}Hz",
                flush=True,
            )
            return False

        sf.write(str(tts_wav), y_out.astype(np.float32), sr)
        print(
            f"[F0-TRANSFER] Seg {seg_idx}: OK | blend={blend_weight} lp_win={_lp_win}fr"
            f" | f0_var {_f0_var_before:.1f}→{_f0_var_after:.1f}Hz",
            flush=True,
        )
        return True

    except ImportError as _ie:
        print(f"[F0-TRANSFER] Seg {seg_idx}: chýba knižnica ({_ie})", flush=True)
        return False
    except Exception as e:
        print(f"[F0-TRANSFER] Seg {seg_idx}: failed ({e}) — wav nezmenený", flush=True)
        return False


def _inject_pauses_wav(
    tts_wav: Path,
    pause_positions: list[float],
    seg_idx: int,
    min_pause_ms: int = 120,
    max_pause_ms: int = 450,
) -> bool:
    """Vloží presné tichá (ms) do syntetizovaného WAV podľa relatívnych pozícií.

    Na rozdiel od textového _inject_pauses (vkladá čiarky), táto funkcia pracuje
    priamo s audio — identifikuje VAD hranice slov a vkladá ticho v presnom čase.

    pause_positions: relatívne pozície 0.0–1.0 (výstup _extract_pause_map)
    min/max_pause_ms: rozsah dĺžky vloženého ticha

    Vracia True ak aspoň jedna pauza bola vložená.
    """
    if not pause_positions:
        return False
    try:
        import numpy as np
        import soundfile as sf
        import librosa

        y, sr = sf.read(str(tts_wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        n = len(y)
        if n < sr * 0.2:
            return False

        # VAD: nájdi "tiché okná" v TTS audia kde môžeme vložiť pauzu
        hop = 256
        rms = librosa.feature.rms(y=y.astype(np.float32), frame_length=1024, hop_length=hop)[0]
        threshold = max(float(np.percentile(rms, 25)), 0.003)

        # Pre každú požadovanú pozíciu nájdi najbližšie tiché okno v ±10% okolí
        inserts: list[tuple[int, int]] = []  # (sample_position, silence_samples)
        for pos in sorted(set(pause_positions)):
            target_sample = int(pos * n)
            search_radius = int(0.10 * n)
            s_start = max(0, target_sample - search_radius)
            s_end = min(n, target_sample + search_radius)

            # Hľadaj najtiší frame v okolí
            f_start = s_start // hop
            f_end = min(len(rms), s_end // hop + 1)
            if f_end <= f_start:
                continue

            local_rms = rms[f_start:f_end]
            quietest_local = int(np.argmin(local_rms))
            insert_sample = (f_start + quietest_local) * hop

            # Dĺžka pauzy: proportional to how quiet the frame is
            quietness = _clamp(1.0 - local_rms[quietest_local] / (threshold + 1e-9), 0.0, 1.0)
            pause_ms = int(min_pause_ms + quietness * (max_pause_ms - min_pause_ms))
            pause_samples = int(sr * pause_ms / 1000)
            inserts.append((insert_sample, pause_samples))

        if not inserts:
            return False

        # Vloži tichá od konca (zachovaj indexy)
        inserts_sorted = sorted(inserts, key=lambda x: x[0], reverse=True)
        for insert_sample, pause_samples in inserts_sorted:
            silence = np.zeros(pause_samples, dtype=np.float32)
            y = np.concatenate([y[:insert_sample], silence, y[insert_sample:]])

        sf.write(str(tts_wav), y, sr)
        total_ms = sum(ps * 1000 // sr for _, ps in inserts)
        print(
            f"[PAUSE-WAV] Seg {seg_idx}: {len(inserts)} pauza(y) vložená, "
            f"celkom +{total_ms}ms",
            flush=True,
        )
        return True

    except Exception as e:
        print(f"[PAUSE-WAV] Seg {seg_idx}: failed ({e}) — wav nezmenený", flush=True)
        return False


def _extract_pitch_contour(
    src_wav: Path,
    start: float,
    end: float,
    n_chunks: int,
    scratch_dir: Path,
    seg_idx: int,
    base_exag: float,
    base_cfg: float,
    base_temp: float,
) -> list[dict[str, float]]:
    """Per-chunk TTS parametre odvodené z F0 (pitch) kontúry originálu.

    Extrahuje základnú frekvenciu cez librosa.pyin(). Vysoká variabilita F0
    (expresívna intonácia) → vyššia exaggeration. Monotónna reč → nižšia
    exaggeration, vyšší cfg_weight pre stabilitu.

    Výstup: list dĺžky n_chunks, každý prvok {exaggeration, cfg_weight, temperature,
    mean_f0, pitch_variation, voiced_ratio}. Pri zlyhaní: fallback na base params.
    """
    fallback = [{"exaggeration": base_exag, "cfg_weight": base_cfg, "temperature": base_temp}] * n_chunks
    if n_chunks <= 1:
        return fallback

    try:
        import numpy as np
        import librosa

        probe_wav = scratch_dir / f"pitch_probe_{seg_idx:04d}.wav"
        if not _extract_segment_audio(src_wav, start, end, probe_wav, pad_s=0.05):
            return fallback
        try:
            y, sr = librosa.load(str(probe_wav), sr=16000, mono=True)
            if y.size < int(sr * 0.3):
                return fallback

            hop = 256
            f0, voiced_flag, _voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                sr=sr, hop_length=hop, frame_length=2048,
            )

            frames_per_chunk = max(1, len(f0) // n_chunks)
            result = []

            for ci in range(n_chunks):
                f_start = ci * frames_per_chunk
                f_end = f_start + frames_per_chunk if ci < n_chunks - 1 else len(f0)
                chunk_f0 = f0[f_start:f_end]
                chunk_voiced = voiced_flag[f_start:f_end]

                valid_f0 = chunk_f0[chunk_voiced & np.isfinite(chunk_f0)]
                if len(valid_f0) < 3:
                    result.append({"exaggeration": base_exag, "cfg_weight": base_cfg, "temperature": base_temp})
                    continue

                mean_f0 = float(np.mean(valid_f0))
                std_f0 = float(np.std(valid_f0))
                voiced_ratio = len(valid_f0) / max(len(chunk_f0), 1)

                # pitch_variation: std/mean normalizovaný na bežnú reč (~0.1–0.35)
                pitch_variation = _clamp(std_f0 / (mean_f0 + 1e-6) / 0.35, 0.0, 1.0)
                voiced_score = _clamp(voiced_ratio / 0.7, 0.0, 1.0)
                pitch_score = _clamp(0.7 * pitch_variation + 0.3 * voiced_score, 0.0, 1.0)

                exag = _clamp(base_exag + (pitch_score - 0.5) * 0.24, 0.20, 0.88)
                cfg = _clamp(base_cfg + (0.5 - pitch_score) * 0.12, 0.35, 0.85)
                temp = _clamp(base_temp + (pitch_variation - 0.5) * 0.06, 0.42, 0.82)
                result.append({
                    "exaggeration": round(exag, 3),
                    "cfg_weight": round(cfg, 3),
                    "temperature": round(temp, 3),
                    "mean_f0": round(mean_f0, 1),
                    "pitch_variation": round(pitch_variation, 3),
                    "voiced_ratio": round(voiced_ratio, 3),
                })

            return result
        finally:
            probe_wav.unlink(missing_ok=True)
    except Exception as e:
        print(f"[PITCH-CONTOUR] Segment {seg_idx}: failed: {e}", flush=True)
        return fallback


def _extract_phoneme_duration(
    src_wav: Path,
    start: float,
    end: float,
    n_chunks: int,
    scratch_dir: Path,
    seg_idx: int,
) -> list[float]:
    """Relatívna hustota onsetov reči (0.0–1.0) pre každý chunk segmentu.

    Používa librosa onset detection na odhad rýchlosti reči.
    0.0 = pomalá reč / dlhé slabiky, 1.0 = rýchla reč / krátke slabiky.

    Výstup sa používa na jemnú úpravu temperature: rýchla reč → nižšia temp
    (stabilnejší prejav), pomalá reč → vyššia temp (prirodzenejšia variácia).
    Pri zlyhaní: fallback 0.5 pre všetky chunky.
    """
    fallback = [0.5] * n_chunks
    if n_chunks <= 1:
        return fallback

    try:
        import numpy as np
        import librosa

        probe_wav = scratch_dir / f"phoneme_probe_{seg_idx:04d}.wav"
        if not _extract_segment_audio(src_wav, start, end, probe_wav, pad_s=0.05):
            return fallback
        try:
            y, sr = librosa.load(str(probe_wav), sr=16000, mono=True)
            if y.size < int(sr * 0.2):
                return fallback

            hop = 256
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="frames")
            frames_per_chunk = max(1, len(y) // hop // n_chunks)
            result = []

            for ci in range(n_chunks):
                f_start = ci * frames_per_chunk
                f_end = f_start + frames_per_chunk if ci < n_chunks - 1 else (len(y) // hop)
                chunk_onsets = onset_frames[(onset_frames >= f_start) & (onset_frames < f_end)]
                chunk_dur_s = (f_end - f_start) * hop / sr

                if chunk_dur_s < 0.1:
                    result.append(0.5)
                    continue

                # onsets per second; normalize: ~4–8 onsets/s = normal speech
                onset_rate = len(chunk_onsets) / chunk_dur_s
                score = _clamp((onset_rate - 2.0) / 8.0, 0.0, 1.0)
                result.append(round(score, 3))

            return result
        finally:
            probe_wav.unlink(missing_ok=True)
    except Exception as e:
        print(f"[PHONEME-DUR] Segment {seg_idx}: failed: {e}", flush=True)
        return fallback


_TEMP_BY_EMOTION: dict[str, float] = {
    "angry": 0.85, "surprised": 0.78, "happy": 0.75,
    "fearful": 0.70, "disgust": 0.68, "neutral": 0.65,
    "calm": 0.58, "sad": 0.55,
}


def _emotion_vec_to_cb_params(
    emotion: dict,
    base_exag: float,
    base_cfg: float,
    base_temp: float,
    blend: float = 0.45,
) -> tuple[float, float, float]:
    """Convert Level 0b emotion vector to Chatterbox (exaggeration, cfg_weight, temperature).

    arousal  → exaggeration  (0→0.30, 1→0.80)
    valence  → cfg_weight    (0→0.40, 1→0.70)
    dominant → temperature   (via _TEMP_BY_EMOTION)
    blend    → weight of emotion vs base params
    """
    arousal = float(emotion.get("arousal", 0.4))
    valence = float(emotion.get("valence", 0.4))
    dominant = str(emotion.get("dominant", "neutral")).lower()

    em_exag = 0.30 + arousal * 0.50
    em_cfg  = 0.40 + valence * 0.30
    em_temp = _TEMP_BY_EMOTION.get(dominant, 0.65)

    exag = round(_clamp(base_exag * (1 - blend) + em_exag * blend, 0.25, 0.90), 3)
    cfg  = round(_clamp(base_cfg  * (1 - blend) + em_cfg  * blend, 0.30, 0.80), 3)
    temp = round(_clamp(base_temp * (1 - blend) + em_temp * blend, 0.42, 0.88), 3)
    return exag, cfg, temp


def _blend_prosody_layers(
    energy: list[dict],
    pitch: list[dict],
    phoneme_dur: list[float],
) -> list[dict]:
    """Zlúči energy contour, pitch contour a phoneme duration do finálnych per-chunk parametrov.

    Váhy: energy 50%, pitch 50%. Phoneme duration jemne dolaďuje temperature.
    """
    n = len(energy)
    result = []
    for i in range(n):
        e = energy[i]
        p = pitch[i] if i < len(pitch) else e
        d = phoneme_dur[i] if i < len(phoneme_dur) else 0.5

        exag = round(0.50 * e["exaggeration"] + 0.50 * p["exaggeration"], 3)
        cfg = round(0.50 * e["cfg_weight"] + 0.50 * p["cfg_weight"], 3)
        temp_base = 0.50 * e["temperature"] + 0.50 * p["temperature"]
        # Rýchla reč (d→1) → znížiť temp (stabilnejšie), pomalá (d→0) → zvýšiť
        temp = round(_clamp(temp_base + (0.5 - d) * 0.06, 0.42, 0.82), 3)
        result.append({"exaggeration": exag, "cfg_weight": cfg, "temperature": temp})
    return result


def _extract_chunk_references(
    src_wav: Path,
    start: float,
    end: float,
    n_chunks: int,
    scratch_dir: Path,
    seg_idx: int,
    min_ref_s: float = 1.5,
    pad_s: float = 0.15,
) -> list[str | None]:
    """Pre každý chunk vráti cestu k zodpovedajúcemu úseku originálneho audia.

    Segment [start, end] sa rovnomerne rozdelí na n_chunks okien. Každé okno
    je extrahované ako samostatný WAV súbor → Chatterbox ho použije ako
    reference prompt pre daný chunk namiesto globálneho promptu.

    Ak je okno príliš krátke (< min_ref_s), vráti None → fallback na globálny prompt.
    Súbory sa vymažú po TTS syntéze (volajúci je zodpovedný za cleanup).
    """
    if n_chunks <= 1:
        return [None]

    seg_dur = end - start
    chunk_dur = seg_dur / n_chunks
    result: list[str | None] = []

    for ci in range(n_chunks):
        c_start = start + ci * chunk_dur
        c_end = c_start + chunk_dur
        effective_dur = chunk_dur + pad_s * 2

        if effective_dur < min_ref_s:
            # Príliš krátke — rozšír do susedných okien, max do hraníc segmentu
            extend = (min_ref_s - effective_dur) / 2
            c_start = max(start, c_start - extend)
            c_end = min(end, c_end + extend)
            effective_dur = c_end - c_start

        if effective_dur < min_ref_s * 0.7:
            result.append(None)
            continue

        ref_path = scratch_dir / f"chunk_ref_{seg_idx:04d}_{ci:02d}.wav"
        if _extract_segment_audio(src_wav, c_start, c_end, ref_path, pad_s=pad_s):
            result.append(str(ref_path))
        else:
            result.append(None)

    return result


_TURBO_TECH_RE = re.compile(
    r"\b(?:API|HTTPS?|JSON|SQL|HTML|CSS|VPN|DNS|URL|REST|GraphQL|OCR|RAG|LLM|CPU|GPU|RAM|Docker|Git(?:Hub)?|Node\.js|TypeScript|JavaScript|USB-C|Wi-?Fi|GPT(?:-\d+(?:\.\d+)?)?)\b|C\+\+|C#",
    re.IGNORECASE,
)
_TURBO_ACRONYM_RE = re.compile(r"\b(?=[A-Z0-9+#/.-]*[A-Z])[A-Z0-9]{2,}(?:[-+#/.][A-Z0-9]+)*\b|C\+\+|C#")
_SK_ACRONYM_TTS_MAP = {
    "HTTP": "há-té-té-pé",
    "HTTPS": "há-té-té-pé-es",
    "SSL": "es-es-el",
    "CPU": "cé-pé-ú",
    "GPU": "gé-pé-ú",
    "RAM": "ram",
    "SSD": "es-es-dé",
    "API": "á-pé-í",
    "JSON": "džejsón",
    "XML": "iks-em-el",
    "CSV": "cé-es-vé",
}


def _turbo_fallback_reasons(text: str, slot_dur: float, sql_mode: bool = False) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    reasons: list[str] = []
    has_digits = any(ch.isdigit() for ch in raw)
    has_tech = bool(_TURBO_TECH_RE.search(raw))
    acronym_tokens = _TURBO_ACRONYM_RE.findall(raw)

    if sql_mode:
        reasons.append("sql_mode")
    if has_digits:
        reasons.append("digits")
    if re.search(r"\b(?:19|20)\d{2}\b", raw):
        reasons.append("year")
    if re.search(r"\d+[.,]\d+%?|\d+\s*%", raw):
        reasons.append("numeric_format")
    if has_tech:
        reasons.append("tech_term")
    if len(acronym_tokens) >= 2:
        reasons.append("multiple_acronyms")
    if len(raw) <= 28 and (has_digits or has_tech or acronym_tokens):
        reasons.append("short_sensitive")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def _is_acronym_heavy_tts_text(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    acronym_tokens = _TURBO_ACRONYM_RE.findall(raw)
    if len(acronym_tokens) >= 3:
        return True
    if len(acronym_tokens) >= 2 and raw.count(",") >= 2:
        return True
    return False


def _probe_audio_duration(path: Path) -> float:
    try:
        import soundfile as _sf
        return float(_sf.info(str(path)).duration)
    except Exception:
        import subprocess
        try:
            res = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=FFPROBE_TIMEOUT,
            )
            return float((res.stdout or "").strip() or 0.0)
        except Exception:
            return 0.0


def _timing_status(slot_duration: float, wav_duration: float) -> str:
    delta = wav_duration - slot_duration
    if delta > max(0.18, slot_duration * 0.05):
        return "too_long"
    if delta < -max(0.60, slot_duration * 0.22):
        return "too_short"
    return "ok"


def _respell_sk_acronym_heavy_tts_text(text: str) -> str:
    out = text
    for token, spoken in sorted(_SK_ACRONYM_TTS_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        out = re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", spoken, out)
    return out


def _is_multi_voice_cue_text(text: str) -> bool:
    raw = (text or "").strip().lower()
    if not raw:
        return False

    norm = re.sub(r"[^\w\s]", " ", raw, flags=re.UNICODE)
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return False

    cue_exact = {
        "boj",
        "bojujte",
        "prvé kolo",
        "druhé kolo",
        "tretie kolo",
        "záverečné kolo",
        "záverečné kolo boj",
        "kratos vyhráva",
        "fatalita",
        "brutalita",
        "bez milosti",
    }
    if norm in cue_exact:
        return True

    cue_tokens = {"kolo", "boj", "bojujte", "vyhráva", "fatalita", "brutalita"}
    tokens = norm.split()
    if len(tokens) <= 4 and cue_tokens.intersection(tokens):
        return True
    return False


def _is_multi_voice_local_clone_text(text: str, slot_dur: float) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if slot_dur > 3.2 or len(raw) > 36:
        return False

    norm = re.sub(r"[^\w\s]", " ", raw.lower(), flags=re.UNICODE)
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return False

    # Short combat taunts sound wrong when conditioned on a calm cluster sample.
    # Prefer a direct local clone prompt from the original segment instead.
    exact = {
        "ty si nič",
        "kratos vyhráva",
        "bojujte",
        "prvé kolo",
        "druhé kolo",
        "tretie kolo",
        "záverečné kolo boj",
        "fatalita",
        "brutalita",
        "bez milosti",
    }
    if norm in exact:
        return True

    if "!" in raw and len(norm.split()) <= 6:
        return True

    trigger_phrases = (
        "ty si",
        "vyhráva",
        "bojujte",
        "kolo",
        "fatalita",
        "brutalita",
        "bez milosti",
    )
    return any(p in norm for p in trigger_phrases)


def _estimate_pitch_from_wav(wav_path: Path) -> float | None:
    try:
        import librosa
        import numpy as np

        y, _ = librosa.load(str(wav_path), sr=16000)
        if y.size < 2048:
            return None

        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        active = rms > 0.005
        if active.sum() > 5:
            mask = np.repeat(active, 256)[:len(y)]
            y = y[mask]
        if y.size < 2048:
            return None

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=70,
            fmax=420,
            sr=16000,
            frame_length=2048,
            hop_length=512,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
        if voiced_f0 is None:
            return None
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        if len(voiced_f0) < 5:
            return None
        return float(np.median(voiced_f0))
    except Exception:
        return None


def _resolve_cluster_gender_labels(cluster_pitch_hz: dict[int, float | None]) -> dict[int, str]:
    labels: dict[int, str] = {}
    valid = [(sid, pitch) for sid, pitch in cluster_pitch_hz.items() if pitch is not None]

    if len(valid) >= 2:
        low_sid, low_pitch = min(valid, key=lambda item: item[1])
        high_sid, high_pitch = max(valid, key=lambda item: item[1])
        if (high_pitch - low_pitch) >= 22.0:
            labels[low_sid] = "male"
            labels[high_sid] = "female"

    for sid, pitch in cluster_pitch_hz.items():
        if sid in labels:
            continue
        if pitch is None:
            labels[sid] = "unknown"
        else:
            labels[sid] = "female" if pitch >= 165.0 else "male"

    return labels


def _diarization_from_segments(
    segments: list[dict],
) -> tuple[list[str | None], list[int], dict[int, dict[str, object]]]:
    """Build speaker_refs/ids/meta from pre-computed Level 0 diarization fields.

    Uses speaker_id + speaker_ref_audio + speaker_gender already in each segment.
    Skips re-clustering entirely.
    """
    label_to_int: dict[str, int] = {}
    speaker_refs: list[str | None] = []
    speaker_ids: list[int] = []
    speaker_meta: dict[int, dict[str, object]] = {}

    for seg in segments:
        raw_label = str(seg.get("speaker_id") or "SPEAKER_00")
        if raw_label not in label_to_int:
            sid = len(label_to_int)
            label_to_int[raw_label] = sid
            gender = str(seg.get("speaker_gender") or "unknown").lower()
            ref = seg.get("speaker_ref_audio") or None
            speaker_meta[sid] = {"gender": gender, "pitch_hz": None, "label": raw_label, "ref": ref}
        sid = label_to_int[raw_label]
        speaker_ids.append(sid)
        speaker_refs.append(speaker_meta[sid]["ref"])

    spk_summary = ", ".join(
        f"{k}→{v}({speaker_meta[v]['gender']})" for k, v in label_to_int.items()
    )
    print(f"[MULTI] Using pre-computed diarization: {len(label_to_int)} speaker(s): {spk_summary}", flush=True)
    return speaker_refs, speaker_ids, speaker_meta


def _has_diarization(segments: list[dict], min_coverage: float = 0.5) -> bool:
    """Return True if enough segments have pre-computed speaker_id + speaker_ref_audio."""
    with_meta = sum(1 for s in segments if s.get("speaker_id") and s.get("speaker_ref_audio"))
    return len(segments) > 0 and (with_meta / len(segments)) >= min_coverage


def _merge_diarization_json(segments: list[dict], diarization_json: str) -> list[dict]:
    """Merge speaker_id/speaker_ref_audio/speaker_gender/emotion from Level 0b JSON into segments."""
    import json as _json

    _KEYS = ("speaker_id", "speaker_ref_audio", "speaker_gender", "emotion")
    try:
        raw = _json.loads(Path(diarization_json).read_text(encoding="utf-8"))
        diar_segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    except Exception as e:
        print(f"[DIAR] Nepodarilo sa načítať diarization_json: {e}", flush=True)
        return segments

    def _overlap(as_: float, ae: float, bs: float, be: float) -> float:
        return max(0.0, min(ae, be) - max(as_, bs))

    merged = 0
    for seg in segments:
        best, best_ov = None, 0.0
        for d in diar_segs:
            ov = _overlap(seg["start"], seg["end"], float(d["start"]), float(d["end"]))
            if ov > best_ov:
                best_ov, best = ov, d
        if best and best_ov > 0.05:
            for k in _KEYS:
                if k in best:
                    seg[k] = best[k]
            merged += 1

    print(f"[DIAR] Merged diarization into {merged}/{len(segments)} segments from {Path(diarization_json).name}", flush=True)
    return segments


def _cluster_speaker_voices(
    audio_wav: Path,
    segments: list[dict],
    n_speakers: int,
    out_dir: Path,
) -> tuple[list[str | None], list[int], dict[int, dict[str, object]]]:
    """Cluster segments by speaker via resemblyzer d-vector embeddings + AgglomerativeClustering.

    resemblyzer produces 256-dim speaker-identity embeddings trained specifically
    to distinguish speakers regardless of content — far more reliable than MFCC+F0.
    AgglomerativeClustering with cosine distance does not require knowing cluster
    count upfront; we constrain it to n_speakers via n_clusters param.

    Falls back to MFCC+KMeans if resemblyzer is not installed.
    """
    import collections
    import subprocess
    import numpy as np

    # --- Try resemblyzer, fall back to MFCC if unavailable ---
    _use_resemblyzer = False
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        _enc = VoiceEncoder(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
        _use_resemblyzer = True
        print("[MULTI] resemblyzer VoiceEncoder loaded (256-dim d-vectors).", flush=True)
    except Exception as _re_err:
        print("="*60, flush=True)
        print("[MULTI] WARNING: resemblyzer is not installed!", flush=True)
        print("[MULTI] Speaker clustering will use MFCC fallback which is very inaccurate.", flush=True)
        print("[MULTI] For good multi-voice, run: pip install resemblyzer", flush=True)
        print("="*60, flush=True)
        print(f"[MULTI] resemblyzer not available ({_re_err}), falling back to MFCC+KMeans.", flush=True)

    print(f"[MULTI] Computing speaker embeddings for {len(segments)} segments...", flush=True)
    features: list[np.ndarray | None] = []
    tmp_wavs: list[Path | None] = []
    valid_idx: list[int] = []
    segment_pitch_hz: list[float | None] = [None] * len(segments)

    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        dur = end - start
        if dur < 0.5:
            features.append(None)
            tmp_wavs.append(None)
            continue
        tmp = out_dir / f"_spk_seg_{i:04d}.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(max(0.0, start)), "-t", str(dur),
                 "-i", str(audio_wav), "-ar", "16000", "-ac", "1", str(tmp)],
                check=True, capture_output=True, timeout=FFMPEG_TIMEOUT,
            )
            segment_pitch_hz[i] = _estimate_pitch_from_wav(tmp)
            if _use_resemblyzer:
                wav_re = preprocess_wav(str(tmp))
                embed = _enc.embed_utterance(wav_re)  # (256,) float32
                features.append(embed.astype(np.float32))
                tmp_wavs.append(tmp)
                valid_idx.append(i)
            else:
                import librosa
                y, _ = librosa.load(str(tmp), sr=16000)
                rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
                active = rms > 0.005
                if active.sum() > 5:
                    mask = np.repeat(active, 256)[:len(y)]
                    y_active = y[mask]
                else:
                    y_active = y
                mfcc = librosa.feature.mfcc(y=y_active, sr=16000, n_mfcc=20)
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                try:
                    f0, voiced_flag, _ = librosa.pyin(
                        y_active, fmin=50, fmax=400, sr=16000,
                        frame_length=2048, hop_length=512,
                    )
                    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
                    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
                    f0_mean = float(np.median(voiced_f0)) if len(voiced_f0) > 5 else 130.0
                    f0_std_val  = float(np.std(voiced_f0))   if len(voiced_f0) > 5 else 20.0
                    if len(voiced_f0) > 5:
                        segment_pitch_hz[i] = float(np.median(voiced_f0))
                except Exception:
                    f0_mean, f0_std_val = 130.0, 20.0
                features.append(np.concatenate([mfcc_mean, mfcc_std, [f0_mean / 5.0, f0_std_val / 5.0]]))
                tmp_wavs.append(tmp)
                valid_idx.append(i)
        except Exception as e:
            print(f"[MULTI] Feature extraction failed for seg {i+1}: {e}", flush=True)
            features.append(None)
            tmp_wavs.append(None)

    speaker_refs: list[str | None] = [None] * len(segments)
    speaker_meta: dict[int, dict[str, object]] = {}
    label_map: dict[int, int] = {}

    if len(valid_idx) < 2:
        print("[MULTI] Too few valid segments for clustering — skipping.", flush=True)
    else:
        n_clusters = min(n_speakers, len(valid_idx))
        valid_feats = np.array([features[i] for i in valid_idx])

        if _use_resemblyzer:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            # Cosine distance on d-vectors — standard for speaker diarization
            agg = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels = agg.fit_predict(valid_feats)
        else:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            valid_feats = scaler.fit_transform(valid_feats)
            km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            labels = km.fit_predict(valid_feats)

        # --- Silhouette check: log only, do NOT block ---
        # User explicitly enabled multi-voice → they know there are 2 speakers.
        # Cosine silhouette with d-vectors naturally sits in 0.03-0.20 range even
        # for well-separated speakers, so a hard threshold causes false fallbacks.
        if n_clusters > 1 and len(set(labels)) > 1:
            from sklearn.metrics import silhouette_score
            _metric = "cosine" if _use_resemblyzer else "euclidean"
            _sil = float(silhouette_score(valid_feats, labels, metric=_metric))
            _sil_warn = _sil < 0.05
            print(
                f"[MULTI] Silhouette score ({_metric}): {_sil:.3f}"
                + (" ⚠ very low — možno 1 rečník" if _sil_warn else ""),
                flush=True,
            )

        # Log cluster sizes
        for _ck in range(n_clusters):
            _cnt = int(np.sum(labels == _ck))
            print(f"[MULTI] Cluster {_ck}: {_cnt} segs", flush=True)

        label_map = {vi: int(labels[li]) for li, vi in enumerate(valid_idx)}
        cluster_pitch_hz: dict[int, float | None] = {}
        for k in range(n_clusters):
            cluster_pitches = [
                segment_pitch_hz[i]
                for i in valid_idx
                if label_map.get(i) == k and segment_pitch_hz[i] is not None
            ]
            cluster_pitch_hz[k] = float(np.median(cluster_pitches)) if cluster_pitches else None
        cluster_gender = _resolve_cluster_gender_labels(cluster_pitch_hz)
        for k in range(n_clusters):
            speaker_meta[k] = {
                "gender": cluster_gender.get(k, "unknown"),
                "pitch_hz": cluster_pitch_hz.get(k),
            }

        cue_idx = [
            i for i in valid_idx
            if _is_multi_voice_cue_text((segments[i].get("text") or "").strip())
        ]
        cue_counts = collections.Counter(label_map[i] for i in cue_idx if i in label_map)
        if cue_counts:
            cue_speaker = cue_counts.most_common(1)[0][0]
            for i in cue_idx:
                prev = label_map.get(i)
                if prev is None or prev == cue_speaker:
                    continue
                label_map[i] = cue_speaker
                print(
                    f"[MULTI] Cue speaker fix: seg {i+1} -> Speaker {cue_speaker} "
                    f"(was Speaker {prev}) text={segments[i].get('text', '').strip()!r}",
                    flush=True,
                )

        # Max duration for a speaker reference prompt.
        # Chatterbox produces artefacts ("chrcanie") with prompts longer than ~10s.
        _MAX_REF_S = 8.0

        for k in range(n_clusters):
            cluster_segs = [i for i in valid_idx if label_map.get(i) == k]
            if not cluster_segs:
                continue
            # Prefer segments that are long enough to be a good reference but not
            # excessively long.  Pick the best segment that is >= 3 s and closest
            # to _MAX_REF_S; fall back to the longest if none qualify.
            def _ref_score(idx: int) -> float:
                dur = float(segments[idx]["end"]) - float(segments[idx]["start"])
                if dur < 3.0:
                    return -1.0
                # Prefer durations in [4, _MAX_REF_S]; penalise very long segments
                return min(dur, _MAX_REF_S) - max(0.0, dur - _MAX_REF_S) * 2.0

            best = max(cluster_segs, key=_ref_score)
            # If best score is -1 (all segments < 3s), fall back to longest available
            if _ref_score(best) < 0:
                best = max(cluster_segs, key=lambda idx: float(segments[idx]["end"]) - float(segments[idx]["start"]))
            seg_start = float(segments[best]["start"])
            seg_end   = float(segments[best]["end"])
            seg_dur   = seg_end - seg_start

            # If the best segment is longer than _MAX_REF_S, take the last
            # _MAX_REF_S seconds (speech tail — speaker is fully warmed up,
            # less likely to have intro noise or overlap with another speaker).
            if seg_dur > _MAX_REF_S:
                seg_start = seg_end - _MAX_REF_S

            rep_wav = out_dir / f"speaker_{k}.wav"

            # ----------------------------------------------------------------
            # For minority speakers with short segments (< 4s), concatenate
            # multiple segments from the cluster to build a longer reference.
            # ----------------------------------------------------------------
            _MIN_CONCAT_S = 4.0
            _ref_built_by_concat = False
            if seg_dur < _MIN_CONCAT_S and len(cluster_segs) > 1:
                # Sort cluster segments by duration (descending), pick until we
                # reach _MAX_REF_S total.
                _sorted_by_dur = sorted(
                    cluster_segs,
                    key=lambda idx: float(segments[idx]["end"]) - float(segments[idx]["start"]),
                    reverse=True,
                )
                _concat_parts: list[Path] = []
                _concat_total = 0.0
                for _cidx, _seg_i in enumerate(_sorted_by_dur):
                    _s = float(segments[_seg_i]["start"])
                    _e = float(segments[_seg_i]["end"])
                    _d = _e - _s
                    if _concat_total + _d > _MAX_REF_S:
                        # Take only what fits
                        _d_take = _MAX_REF_S - _concat_total
                        _s = _e - _d_take
                        _d = _d_take
                    _part_wav = out_dir / f"speaker_{k}_part{_cidx}.wav"
                    if _extract_segment_audio(audio_wav, _s, _e, _part_wav, pad_s=0.1):
                        _concat_parts.append(_part_wav)
                    _concat_total += _d
                    if _concat_total >= _MAX_REF_S:
                        break

                if len(_concat_parts) >= 2:
                    try:
                        import soundfile as _sf
                        import numpy as _np
                        _concat_arrays = []
                        for _p in _concat_parts:
                            _y, _sr = _sf.read(str(_p))
                            if _y.ndim > 1: _y = _y.mean(axis=1)
                            _fade_len = int(_sr * 0.05)
                            if len(_y) > _fade_len * 2:
                                _y[:_fade_len] *= _np.linspace(0, 1, _fade_len, dtype=_y.dtype)
                                _y[-_fade_len:] *= _np.linspace(1, 0, _fade_len, dtype=_y.dtype)
                            _concat_arrays.append(_y)
                            _concat_arrays.append(_np.zeros(int(_sr * 0.15), dtype=_y.dtype))
                        if _concat_arrays:
                            _final_y = _np.concatenate(_concat_arrays)
                            _sf.write(str(rep_wav), _final_y, 24000, subtype='PCM_16')
                            _ref_built_by_concat = rep_wav.exists() and rep_wav.stat().st_size > 0
                    except Exception as _cc_err:
                        print(f"[MULTI] Speaker {k}: concat failed ({_cc_err}), using single seg", flush=True)
                    finally:
                        for _pp in _concat_parts:
                            _pp.unlink(missing_ok=True)
                    if _ref_built_by_concat:
                        seg_dur = _concat_total
                        print(
                            f"[MULTI] Speaker {k}: built concat reference from "
                            f"{len(_concat_parts)} segs ({_concat_total:.1f}s)",
                            flush=True,
                        )

            ok = _ref_built_by_concat or _extract_segment_audio(
                audio_wav, seg_start, seg_end,
                rep_wav, pad_s=0.3,
            )
            if ok:
                prepare_chatterbox_reference_audio(rep_wav, aggressive=True)

                for i in cluster_segs:
                    speaker_refs[i] = str(rep_wav)
                speaker_meta.setdefault(k, {})
                speaker_meta[k]["ref_path"] = str(rep_wav)
                _ref_dur = min(seg_dur, _MAX_REF_S)
                _pitch = speaker_meta[k].get("pitch_hz")
                _pitch_str = f", pitch≈{_pitch:.1f}Hz" if isinstance(_pitch, (int, float)) else ""
                print(
                    f"[MULTI] Speaker {k}: {len(cluster_segs)} segs, "
                    f"ref=seg {best+1} ({seg_start:.1f}s–{seg_end:.1f}s, ref={_ref_dur:.1f}s), "
                    f"{speaker_meta[k].get('gender', 'unknown')}{_pitch_str}",
                    flush=True,
                )

    # Cleanup temp segment files
    for p in tmp_wavs:
        if p and p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    # Build per-segment speaker ID list (None → -1 = unknown/too short)
    speaker_ids: list[int] = []
    for i in range(len(segments)):
        if i in label_map:
            speaker_ids.append(label_map[i])
        else:
            speaker_ids.append(-1)

    return speaker_refs, speaker_ids, speaker_meta


def _align_srt_whisperx(
    tts_wav: Path,
    tgt_lang: str,
    device: str,
    whisper_root: str,
    whisper_model: str = "large-v3",
) -> None:
    """Re-transcribe TTS audio with WhisperX forced alignment and write SRT.

    Requires whisperx (already in musetalk_env).  Falls back to raw Whisper
    timestamps if the language-specific alignment model is unavailable.
    """
    try:
        import whisperx
    except ImportError:
        print("[SRT-ALIGN] whisperx not found — skipping (pip install whisperx)", flush=True)
        return

    _LANG = {
        "ces": "cs", "cs": "cs", "slk": "sk", "sk": "sk",
        "eng": "en", "en": "en", "deu": "de", "de": "de",
        "fra": "fr", "fr": "fr", "spa": "es", "es": "es",
        "rus": "ru", "ru": "ru", "ita": "it", "it": "it",
        "pol": "pl", "pl": "pl",
    }
    wx_lang = _LANG.get(tgt_lang, tgt_lang[:2])
    srt_path = tts_wav.with_suffix(".srt")

    print(f"[SRT-ALIGN] Transcribing {tts_wav.name} (lang={wx_lang})...", flush=True)

    # Zisti voľnú VRAM — ak menej ako 3 GB, použi CPU priamo (pyannote VAD tiež potrebuje VRAM)
    _wx_device = device
    if device == "cuda" and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        _free_vram_gb = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_allocated(0)) / 1e9
        if _free_vram_gb < 8.0:
            print(
                f"[SRT-ALIGN] Málo VRAM ({_free_vram_gb:.1f} GB < 8 GB) — fallback na CPU",
                flush=True,
            )
            _wx_device = "cpu"

    def _is_oom(exc: Exception) -> bool:
        _oom_types: list = []
        try:
            _oom_types.append(torch.OutOfMemoryError)
        except AttributeError:
            pass
        try:
            _oom_types.append(torch.cuda.OutOfMemoryError)
        except AttributeError:
            pass
        if _oom_types and isinstance(exc, tuple(_oom_types)):
            return True
        return "out of memory" in str(exc).lower()

    import os as _os
    # Ak bežíme na CPU, skry CUDA pred pyannote VAD aby neskúšal použiť GPU
    _hide_cuda = _wx_device == "cpu"
    _orig_cuda_visible = _os.environ.get("CUDA_VISIBLE_DEVICES")
    if _hide_cuda:
        _os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        audio = whisperx.load_audio(str(tts_wav))
        compute_type = "int8_float16" if _wx_device == "cuda" else "int8"

        def _load_wx_model(dev: str, ct: str):
            return whisperx.load_model(
                whisper_model, dev,
                compute_type=ct,
                download_root=whisper_root,
                language=wx_lang,
            )

        try:
            wx_model = _load_wx_model(_wx_device, compute_type)
        except Exception as _e:
            if _is_oom(_e) and _wx_device == "cuda":
                print("[SRT-ALIGN] OOM pri načítaní modelu — fallback na CPU", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                _wx_device = "cpu"
                wx_model = _load_wx_model("cpu", "int8")
            else:
                raise

        try:
            wx_result = wx_model.transcribe(audio, batch_size=8 if _wx_device == "cuda" else 4)
        except Exception as _e:
            _e_str = str(_e).lower()
            if _is_oom(_e) and _wx_device == "cuda":
                print("[SRT-ALIGN] OOM pri transkripcii — fallback na CPU", flush=True)
                del wx_model
                gc.collect()
                torch.cuda.empty_cache()
                _wx_device = "cpu"
                wx_model = _load_wx_model("cpu", "int8")
                wx_result = wx_model.transcribe(audio, batch_size=4)
            elif "batch_size" in _e_str and "too large" in _e_str:
                # Interný pyannote batch_size=32 — nie je ovplyvniteľný cez transcribe()
                # SRT-ALIGN je voliteľný — preskočíme ticho
                print(f"[SRT-ALIGN] pyannote batch_size chyba — preskakujem SRT align", flush=True)
                return
            else:
                raise

        del wx_model
        if _wx_device == "cuda":
            torch.cuda.empty_cache()

        segs = wx_result.get("segments") or []
        if not segs:
            print("[SRT-ALIGN] No speech detected in TTS audio.", flush=True)
            return

        # Forced alignment: word-level timestamps for each segment
        try:
            align_model, align_meta = whisperx.load_align_model(
                language_code=wx_lang, device=_wx_device,
            )
            aligned = whisperx.align(
                segs, align_model, align_meta, audio, _wx_device,
                return_char_alignments=False,
            )
            del align_model
            if _wx_device == "cuda":
                torch.cuda.empty_cache()
            segs = aligned.get("segments", segs)
        except Exception as align_err:
            if _is_oom(align_err):
                print("[SRT-ALIGN] OOM pri word alignment — preskakujem", flush=True)
                if _wx_device == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                print(f"[SRT-ALIGN] Word alignment skipped ({align_err}) — using segment timestamps", flush=True)

        _write_srt(segs, srt_path)
        print(f"[SRT-ALIGN] Saved TTS subtitles: {srt_path}", flush=True)

    except Exception as exc:
        print(f"[SRT-ALIGN] Failed: {exc}", flush=True)
    finally:
        # Obnov CUDA_VISIBLE_DEVICES
        if _hide_cuda:
            if _orig_cuda_visible is None:
                _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                _os.environ["CUDA_VISIBLE_DEVICES"] = _orig_cuda_visible


def _write_srt(segments: list[dict], path: Path) -> None:
    lines: list[str] = []
    idx = 1
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines += [str(idx), f"{_format_srt_time(seg['start'])} --> {_format_srt_time(seg['end'])}", text, ""]
        idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SRT] Saved subtitles with timing: {path}", flush=True)


def _write_segments_json(segments: list[dict], path: Path) -> None:
    import json as _json
    path.write_text(
        _json.dumps({"segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[JSON] Segments uložené: {path}", flush=True)


def _build_effective_tts_segments(segments: list[dict], tts_debug_log: list[dict]) -> list[dict]:
    effective: list[dict] = []
    for i, seg in enumerate(segments):
        merged = dict(seg)
        if i < len(tts_debug_log):
            tts_text = (tts_debug_log[i].get("tts_input") or merged.get("text") or "").strip()
            merged["text"] = tts_text
        effective.append(merged)
    return effective


def _build_level4_payload(
    base_segments: list[dict],
    effective_segments: list[dict],
    tts_debug_log: list[dict],
    timing_audit_log: list[dict],
    wav_paths: list[Path],
) -> list[dict]:
    payload: list[dict] = []
    total = max(
        len(base_segments),
        len(effective_segments),
        len(tts_debug_log),
        len(timing_audit_log),
    )
    for idx in range(total):
        base = dict(base_segments[idx]) if idx < len(base_segments) else {}
        effective = dict(effective_segments[idx]) if idx < len(effective_segments) else {}
        debug = dict(tts_debug_log[idx]) if idx < len(tts_debug_log) else {}
        audit = dict(timing_audit_log[idx]) if idx < len(timing_audit_log) else {}

        merged: dict = {}
        merged.update(base)
        merged.update(effective)
        merged["seg"] = int(merged.get("seg") or idx + 1)

        tts_input = str(debug.get("tts_input") or merged.get("text") or "").strip()
        if tts_input:
            merged["tts_input"] = tts_input
            merged["text"] = tts_input

        if debug.get("translated") and not merged.get("translated"):
            merged["translated"] = debug["translated"]
        if debug:
            merged["tts_debug"] = debug
        if debug.get("tts_engine") and not merged.get("tts_engine"):
            merged["tts_engine"] = debug["tts_engine"]

        if idx < len(wav_paths):
            merged["wav"] = str(Path(wav_paths[idx]).expanduser().resolve())

        if audit:
            merged["level4_audit"] = audit
            merged["delta"] = audit.get("delta")
            merged["duration_status"] = audit.get("status")
            if audit.get("slot") is not None:
                merged["slot"] = audit.get("slot")
        else:
            start = float(merged.get("start", 0.0) or 0.0)
            end = float(merged.get("end", 0.0) or 0.0)
            slot = round(max(0.01, end - start), 3)
            merged["slot"] = slot
            merged["delta"] = None
            merged["duration_status"] = "pending_audio"
            merged["level4_audit"] = {
                "slot": slot,
                "wav_duration": None,
                "delta": None,
                "status": "pending_audio",
            }

        if not merged.get("source_text"):
            merged["source_text"] = str(
                merged.get("text_src")
                or merged.get("text_original")
                or debug.get("translated")
                or merged.get("translated")
                or ""
            ).strip()

        payload.append(merged)
    return payload


def _run_level6_plus_sidecar(
    *,
    base_segments: list[dict],
    effective_segments: list[dict],
    tts_debug_log: list[dict],
    timing_audit_log: list[dict],
    wav_paths: list[Path],
    out_dir: Path,
    stem: str,
    tgt: str,
    args,
) -> None:
    try:
        from src.level10_manager import Level10Manager
        from src.level11_manager import Level11Manager, summarize_level11_segments
        from src.level5_manager import Level5Manager, summarize_level5_segments
        from src.level6_manager import evaluate_scene, summarize_scene_evaluation
        from src.level7_manager import Level7Manager, summarize_level7_segments
        from src.level8_manager import Level8Manager
        from src.level9_manager import Level9Manager, summarize_level9_segments
        from src.utils_io import save_json
    except Exception as exc:
        print(f"[LEVELS] Import failed: {exc}", flush=True)
        return

    try:
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        memory_file = Path(
            getattr(args, "level_memory_file", _ROOT_DIR / "memory" / "project_memory.json")
        ).expanduser().resolve()
        registry_file = Path(
            getattr(args, "level_policy_registry_file", _ROOT_DIR / "memory" / "policy_registry.json")
        ).expanduser().resolve()
        level11_max_retries = max(0, int(getattr(args, "level11_max_retries", 1) or 1))

        level4_segments = _build_level4_payload(
            base_segments=base_segments,
            effective_segments=effective_segments,
            tts_debug_log=tts_debug_log,
            timing_audit_log=timing_audit_log,
            wav_paths=wav_paths,
        )

        level4_path = out_dir / f"{stem}_{tgt}_level4_final.json"
        level5_path = out_dir / f"{stem}_{tgt}_level5_quality_checked.json"
        level6_path = out_dir / f"{stem}_{tgt}_level6_scene_checked.json"
        level7_path = out_dir / f"{stem}_{tgt}_level7_project_adaptive.json"
        level8_path = out_dir / f"{stem}_{tgt}_level8_analysis.json"
        rewrite_dataset_path = out_dir / f"{stem}_{tgt}_rewrite_dataset.json"
        negative_dataset_path = out_dir / f"{stem}_{tgt}_rewrite_negative_dataset.json"
        level9_path = out_dir / f"{stem}_{tgt}_level9_learned_decisions.json"
        level10_analysis_path = out_dir / f"{stem}_{tgt}_level10_analysis.json"
        level10_ab_path = out_dir / f"{stem}_{tgt}_level10_ab_results.json"
        level11_path = out_dir / f"{stem}_{tgt}_level11_multi_agent_results.json"

        _write_segments_json(level4_segments, level4_path)

        level5_manager = Level5Manager()
        level5_segments = level5_manager.process_segments(level4_segments)
        level5_summary = summarize_level5_segments(level5_segments)
        _write_segments_json(level5_segments, level5_path)
        save_json(final_dir / f"{stem}_{tgt}_level5_summary.json", level5_summary)

        level6_segments = evaluate_scene(level5_segments)
        level6_summary = summarize_scene_evaluation(level6_segments)
        level6_summary["memory_file"] = str(memory_file)
        level6_summary["registry_file"] = str(registry_file)
        _write_segments_json(level6_segments, level6_path)
        save_json(final_dir / f"{stem}_{tgt}_level6_summary.json", level6_summary)

        level7_manager = Level7Manager(memory_file)
        level7_segments = level7_manager.process_segments(level6_segments)
        for seg in level7_segments:
            level7_manager.finalize_segment(seg)
        level7_manager.finalize_project()
        level7_summary = summarize_level7_segments(level7_segments)
        level7_summary["memory_file"] = str(memory_file)
        _write_segments_json(level7_segments, level7_path)
        save_json(final_dir / f"{stem}_{tgt}_level7_summary.json", level7_summary)

        level8_manager = Level8Manager(memory_file)
        level8_analysis = level8_manager.analyze()
        level8_manager.update_memory(level8_analysis)
        save_json(level8_path, level8_analysis)
        save_json(rewrite_dataset_path, level8_analysis.get("rewrite_dataset", []))
        save_json(negative_dataset_path, level8_analysis.get("negative_dataset", []))
        save_json(
            final_dir / f"{stem}_{tgt}_level8_summary.json",
            {
                "history_size": level8_analysis.get("history_size", 0),
                "failure_clusters": level8_analysis.get("failure_clusters", {}),
                "prompt_ranking": level8_analysis.get("prompt_ranking", {}),
                "rewrite_dataset_size": level8_analysis.get("rewrite_dataset_size", 0),
                "negative_dataset_size": level8_analysis.get("negative_dataset_size", 0),
                "evaluation_set_size": level8_analysis.get("evaluation_set_size", 0),
            },
        )

        level9_manager = Level9Manager(memory_file)
        level9_segments = level9_manager.process_segments(level7_segments)
        for seg in level9_segments:
            level9_manager.learn_from_result(seg)
        level9_manager.finalize_project()
        level9_summary = summarize_level9_segments(level9_segments)
        level9_summary["memory_file"] = str(memory_file)
        _write_segments_json(level9_segments, level9_path)
        save_json(final_dir / f"{stem}_{tgt}_level9_summary.json", level9_summary)

        level10_manager = Level10Manager(registry_file, memory_file)
        level10_manager.bootstrap()
        active_policy_before = level10_manager.registry.get_active_policy_name()
        level10_analysis = level10_manager.build_analysis_summary()
        synthesized_prompt = level10_manager.synthesize_candidate_prompt(level10_analysis)
        candidate_summaries: list[dict] = []
        ab_results: list[dict] = []
        promoted_candidates: list[str] = []

        for candidate in level10_manager.create_candidates():
            candidate_policy = dict(candidate)
            candidate_policy["base_prompt"] = synthesized_prompt
            level10_manager.register_candidate_policy(candidate_policy)
            comparison = level10_manager.compare_active_vs_candidate(level9_segments, candidate_policy)
            promoted_now = level10_manager.maybe_promote(candidate_policy["name"], comparison["ab_result"])
            if promoted_now:
                promoted_candidates.append(candidate_policy["name"])
            candidate_summaries.append(
                {
                    "name": candidate_policy["name"],
                    "target_cps": candidate_policy.get("target_cps"),
                    "aggressive_shorten_threshold": candidate_policy.get("aggressive_shorten_threshold"),
                    "force_locked_terms_on_sql": candidate_policy.get("force_locked_terms_on_sql"),
                    "base_prompt": candidate_policy.get("base_prompt"),
                    "promoted": promoted_now,
                    "ab_result": comparison["ab_result"],
                }
            )
            ab_results.append(comparison)

        level10_payload = {
            "active_policy_before": active_policy_before,
            "active_policy_after": level10_manager.registry.get_active_policy_name(),
            "analysis": level10_analysis,
            "synthesized_prompt": synthesized_prompt,
            "candidate_policies": candidate_summaries,
            "promoted_candidates": promoted_candidates,
            "memory_file": str(memory_file),
            "registry_file": str(registry_file),
        }
        save_json(level10_analysis_path, level10_payload)
        save_json(level10_ab_path, ab_results)
        save_json(final_dir / f"{stem}_{tgt}_level10_summary.json", level10_payload)

        level11_manager = Level11Manager(
            memory_file,
            registry_file,
            max_retries=level11_max_retries,
        )
        level11_segments = level11_manager.process_segments(level9_segments)
        level11_summary = summarize_level11_segments(level11_segments)
        level11_summary["memory_file"] = str(memory_file)
        level11_summary["registry_file"] = str(registry_file)
        level11_manager.finalize(level11_summary)
        _write_segments_json(level11_segments, level11_path)
        save_json(final_dir / f"{stem}_{tgt}_level11_summary.json", level11_summary)

        save_json(
            final_dir / f"{stem}_{tgt}_level_sidecar_summary.json",
            {
                "enabled": True,
                "memory_file": str(memory_file),
                "registry_file": str(registry_file),
                "level5": level5_summary,
                "level6": level6_summary,
                "level7": level7_summary,
                "level8": {
                    "history_size": level8_analysis.get("history_size", 0),
                    "rewrite_dataset_size": level8_analysis.get("rewrite_dataset_size", 0),
                    "negative_dataset_size": level8_analysis.get("negative_dataset_size", 0),
                },
                "level9": level9_summary,
                "level10": {
                    "active_policy_before": active_policy_before,
                    "active_policy_after": level10_payload["active_policy_after"],
                    "promoted_candidates": promoted_candidates,
                },
                "level11": level11_summary,
            },
        )
        print(
            f"[LEVELS] Level 5-11 sidecar saved → "
            f"{out_dir / f'{stem}_{tgt}_level11_multi_agent_results.json'}",
            flush=True,
        )
    except Exception as exc:
        import traceback as _traceback

        print(f"[LEVELS] Sidecar failed: {exc}", flush=True)
        print(_traceback.format_exc(), flush=True)


def _run_pre_tts_level6_plus(
    *,
    translated_segments: list[dict],
    out_dir: Path,
    stem: str,
    tgt: str,
    args,
) -> tuple[list[dict], dict | None]:
    try:
        from src.level10_manager import Level10Manager
        from src.level11_manager import Level11Manager, summarize_level11_segments
        from src.level5_manager import Level5Manager, summarize_level5_segments
        from src.level6_manager import evaluate_scene, summarize_scene_evaluation
        from src.level7_manager import Level7Manager, summarize_level7_segments
        from src.level8_manager import Level8Manager
        from src.level9_manager import Level9Manager, summarize_level9_segments
        from src.utils_io import save_json
    except Exception as exc:
        print(f"[LEVELS/PRE-TTS] Import failed: {exc}", flush=True)
        return translated_segments, None

    try:
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        memory_file = Path(
            getattr(args, "level_memory_file", _ROOT_DIR / "memory" / "project_memory.json")
        ).expanduser().resolve()
        registry_file = Path(
            getattr(args, "level_policy_registry_file", _ROOT_DIR / "memory" / "policy_registry.json")
        ).expanduser().resolve()
        level11_max_retries = max(0, int(getattr(args, "level11_max_retries", 1) or 1))

        level4_segments = _build_level4_payload(
            base_segments=translated_segments,
            effective_segments=translated_segments,
            tts_debug_log=[],
            timing_audit_log=[],
            wav_paths=[],
        )

        level5_path = out_dir / f"{stem}_{tgt}_pre_tts_level5_quality_checked.json"
        level6_path = out_dir / f"{stem}_{tgt}_pre_tts_level6_scene_checked.json"
        level7_path = out_dir / f"{stem}_{tgt}_pre_tts_level7_project_adaptive.json"
        level8_path = out_dir / f"{stem}_{tgt}_pre_tts_level8_analysis.json"
        level9_path = out_dir / f"{stem}_{tgt}_pre_tts_level9_learned_decisions.json"
        level10_analysis_path = out_dir / f"{stem}_{tgt}_pre_tts_level10_analysis.json"
        level10_ab_path = out_dir / f"{stem}_{tgt}_pre_tts_level10_ab_results.json"
        level11_path = out_dir / f"{stem}_{tgt}_pre_tts_level11_multi_agent_results.json"

        level8_manager = Level8Manager(memory_file)
        level8_analysis = level8_manager.analyze()
        level8_manager.update_memory(level8_analysis)
        save_json(level8_path, level8_analysis)

        level10_manager = Level10Manager(registry_file, memory_file)
        level10_manager.bootstrap()
        active_policy_before = level10_manager.registry.get_active_policy_name()

        level5_manager = Level5Manager()
        level5_segments = level5_manager.process_segments(level4_segments)
        level5_summary = summarize_level5_segments(level5_segments)
        _write_segments_json(level5_segments, level5_path)
        save_json(final_dir / f"{stem}_{tgt}_pre_tts_level5_summary.json", level5_summary)

        level6_segments = evaluate_scene(level5_segments)
        level6_summary = summarize_scene_evaluation(level6_segments)
        level6_summary["memory_file"] = str(memory_file)
        level6_summary["registry_file"] = str(registry_file)
        _write_segments_json(level6_segments, level6_path)
        save_json(final_dir / f"{stem}_{tgt}_pre_tts_level6_summary.json", level6_summary)

        level7_manager = Level7Manager(memory_file)
        level7_segments = level7_manager.process_segments(level6_segments)
        level7_summary = summarize_level7_segments(level7_segments)
        level7_summary["memory_file"] = str(memory_file)
        _write_segments_json(level7_segments, level7_path)
        save_json(final_dir / f"{stem}_{tgt}_pre_tts_level7_summary.json", level7_summary)

        level9_manager = Level9Manager(memory_file)
        level9_segments = level9_manager.process_segments(level7_segments)
        level9_summary = summarize_level9_segments(level9_segments)
        level9_summary["memory_file"] = str(memory_file)
        _write_segments_json(level9_segments, level9_path)
        save_json(final_dir / f"{stem}_{tgt}_pre_tts_level9_summary.json", level9_summary)

        level10_analysis = level10_manager.build_analysis_summary()
        synthesized_prompt = level10_manager.synthesize_candidate_prompt(level10_analysis)
        candidate_summaries: list[dict] = []
        ab_results: list[dict] = []
        promoted_candidates: list[str] = []

        for candidate in level10_manager.create_candidates():
            candidate_policy = dict(candidate)
            candidate_policy["base_prompt"] = synthesized_prompt
            level10_manager.register_candidate_policy(candidate_policy)
            comparison = level10_manager.compare_active_vs_candidate(level9_segments, candidate_policy)
            promoted_now = level10_manager.maybe_promote(candidate_policy["name"], comparison["ab_result"])
            if promoted_now:
                promoted_candidates.append(candidate_policy["name"])
            candidate_summaries.append(
                {
                    "name": candidate_policy.get("name"),
                    "target_cps": candidate_policy.get("target_cps"),
                    "aggressive_shorten_threshold": candidate_policy.get("aggressive_shorten_threshold"),
                    "force_locked_terms_on_sql": candidate_policy.get("force_locked_terms_on_sql"),
                    "base_prompt": candidate_policy.get("base_prompt"),
                    "promoted": promoted_now,
                    "ab_result": comparison["ab_result"],
                }
            )
            ab_results.append(comparison)

        level10_payload = {
            "active_policy_before": active_policy_before,
            "active_policy_after": level10_manager.registry.get_active_policy_name(),
            "analysis": level10_analysis,
            "synthesized_prompt": synthesized_prompt,
            "candidate_policies": candidate_summaries,
            "promoted_candidates": promoted_candidates,
            "memory_file": str(memory_file),
            "registry_file": str(registry_file),
        }
        save_json(level10_analysis_path, level10_payload)
        save_json(level10_ab_path, ab_results)
        save_json(final_dir / f"{stem}_{tgt}_pre_tts_level10_summary.json", level10_payload)

        level11_manager = Level11Manager(
            memory_file,
            registry_file,
            max_retries=level11_max_retries,
        )
        level11_segments = level11_manager.process_segments(level9_segments)
        level11_summary = summarize_level11_segments(level11_segments)
        level11_summary["memory_file"] = str(memory_file)
        level11_summary["registry_file"] = str(registry_file)
        _write_segments_json(level11_segments, level11_path)
        save_json(final_dir / f"{stem}_{tgt}_pre_tts_level11_summary.json", level11_summary)

        merged_segments: list[dict] = []
        changed = 0
        selected_by_agent = 0
        for idx, seg in enumerate(translated_segments):
            merged = dict(seg)
            enriched = dict(level11_segments[idx]) if idx < len(level11_segments) else {}
            original_text = str(merged.get("text") or "").strip()
            selected_text = str(
                enriched.get("selected_tts_input")
                or enriched.get("tts_input")
                or original_text
            ).strip()
            if selected_text:
                merged["text"] = selected_text
                merged["tts_input"] = selected_text
            if selected_text and selected_text != original_text:
                changed += 1
            if str(enriched.get("best_candidate_name") or "").strip():
                selected_by_agent += 1

            for key in (
                "prompt_name",
                "selected_prompt",
                "retry_mode",
                "segment_type",
                "scene_type",
                "scene_base_style",
                "scene_target_cps",
                "scene_hard_cps_limit",
                "force_locked_terms",
                "meaning_score",
                "term_score",
                "cps_score",
                "duration_score",
                "spoken_score",
                "asr_match_score",
                "final_score",
                "final_chars_per_sec",
                "multi_agent_score",
                "chosen_candidate_name",
                "chosen_candidate_source",
                "best_candidate_name",
                "best_prosody_name",
                "duration_status",
                "delta",
            ):
                if enriched.get(key) is not None:
                    merged[key] = enriched[key]

            merged["pre_tts_level5_scores"] = dict(enriched.get("level5_scores") or {})
            merged["pre_tts_multi_agent_selected"] = dict(enriched.get("multi_agent_selected") or {})
            merged["pre_tts_orchestrator_attempts"] = list(enriched.get("orchestrator_attempts") or [])
            merged_segments.append(merged)

        summary_payload = {
            "enabled": True,
            "changed_segments": changed,
            "selected_by_agent": selected_by_agent,
            "memory_file": str(memory_file),
            "registry_file": str(registry_file),
            "level5": level5_summary,
            "level6": level6_summary,
            "level7": level7_summary,
            "level8": {
                "history_size": level8_analysis.get("history_size", 0),
                "rewrite_dataset_size": level8_analysis.get("rewrite_dataset_size", 0),
                "negative_dataset_size": level8_analysis.get("negative_dataset_size", 0),
            },
            "level9": level9_summary,
            "level10": {
                "active_policy_before": active_policy_before,
                "active_policy_after": level10_payload["active_policy_after"],
                "promoted_candidates": promoted_candidates,
            },
            "level11": level11_summary,
        }
        save_json(final_dir / f"{stem}_{tgt}_pre_tts_level_sidecar_summary.json", summary_payload)
        print(
            f"[LEVELS/PRE-TTS] changed={changed} "
            f"active_policy={level10_payload['active_policy_after']} "
            f"level11_ok={level11_summary.get('ok_segments', 0)}/{level11_summary.get('segments', 0)} "
            f"| {level11_path}",
            flush=True,
        )
        return merged_segments, summary_payload
    except Exception as exc:
        import traceback as _traceback

        print(f"[LEVELS/PRE-TTS] Failed: {exc}", flush=True)
        print(_traceback.format_exc(), flush=True)
        return translated_segments, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"[ERROR] Input not found: {input_path}")

    # Working dirs
    work_dir = _SCRIPTS_DIR.parent / "work"
    chunks_dir = work_dir / "tts_chunks"
    logs_dir = work_dir / "logs"
    # Pred novým behom vyčisti tts_chunks/ — chunky z predošlého behu (často iný engine)
    # by sa inak preskočili a pipeline by miešala enginy. Resume v rámci jedného behu
    # nepotrebujeme — chunky sa generujú v sekundách a re-generation je rýchla.
    if chunks_dir.exists():
        import shutil as _sh
        _removed = sum(1 for _ in chunks_dir.iterdir())
        _sh.rmtree(chunks_dir, ignore_errors=True)
        if _removed:
            print(f"[CLEANUP] Vymazané {_removed} starých chunkov z work/tts_chunks/", flush=True)
    for d in (work_dir, chunks_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Output dir for non-video files (JSON, WAV, SRT) — defaults to input's parent dir
    _out_dir_arg = getattr(args, "out_dir", "")
    out_dir = Path(_out_dir_arg).resolve() if _out_dir_arg else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _setup_logging(logs_dir / f"run_{stamp}.log")
    print(f"[LOG] Logging to: work/logs/run_{stamp}.log", flush=True)
    _log_memory("štart pipeline")

    os.chdir(_SCRIPTS_DIR.parent)  # relative model paths resolve from VideoTranslator/

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgt = args.tgt_lang
    stem = input_path.stem

    # ── G3 27B multitask = single-step (translate + length-aware) ─────────────
    # Vypneme ADAPT/CPS pass — 27B robí length-aware preklad sám cez constrained prompt.
    if getattr(args, "use_g3_27b_translator", False):
        args.no_adapt_llm = True
        args.compress_overflow = False
        args.cps_expand_threshold = 0.0
        print(f"[27B-MULTITASK] Single-step mód: vypnuté ADAPT/CPS-EXPAND/compress-overflow",
              flush=True)

    # ── Auto-override ADAPT/CPS-EXPAND model na G3 v1 length adjuster ──────────
    # Keď uzivatel explicitne zvolil --use_g3_v1_length_adjuster, nezhoduje sa to
    # s default adapt_ollama_model=sk-gemma4-e4b-v6 ani llama_cpp Gemma-4-26B.
    # Force adaptation pass aby používal ten istý g3-12b-length-adjuster-v1 model.
    if getattr(args, "use_g3_v1_length_adjuster", False):
        _g3_la_model = getattr(args, "g3_v1_length_adjuster_model", "g3-12b-length-adjuster-v1")
        _orig_adapt = getattr(args, "adapt_ollama_model", "")
        if _orig_adapt != _g3_la_model:
            args.adapt_ollama_model = _g3_la_model
            args.adapt_provider = "ollama"  # nelloaduj llama_cpp 26B
            args.no_adapt_llm = True        # vypni llama_cpp 26B
            print(f"[ADAPT-OVERRIDE] use_g3_v1_length_adjuster=True → "
                  f"adapt_ollama_model: {_orig_adapt or '(default)'} → {_g3_la_model}, "
                  f"provider=ollama, no_adapt_llm=True", flush=True)

    _startup_ollama_url   = getattr(args, "adapt_ollama_url",   "")
    _startup_ollama_model = getattr(args, "adapt_ollama_model", "")
    if (
        getattr(args, "adapt_provider", "llama_cpp") == "ollama"
        or (_startup_ollama_url and _startup_ollama_model)
    ):
        if not _startup_ollama_url:
            _startup_ollama_url = "http://localhost:11434/api/generate"
        if not _startup_ollama_model:
            _startup_ollama_model = "gemma4:26b"
        _startup_ollama_timeout = int(getattr(args, "adapt_ollama_timeout", 120))
        print(f"[STARTUP] Uvoľňujem ollama pred spustením pipeline: {_startup_ollama_model}", flush=True)
        _unload_ollama_model(
            _startup_ollama_url,
            _startup_ollama_model,
            timeout=min(_startup_ollama_timeout, 20),
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Audio extraction ---
    audio_wav = work_dir / "whisper_input.wav"
    extract_audio_ffmpeg(str(input_path), str(audio_wav))
    total_dur = get_audio_duration(audio_wav)

    # --- Keep music: separate before whisper ---
    music_wav: Path | None = None
    vocals_wav: Path | None = None   # clean vocals — used for speaker clustering
    if args.keep_music:
        _vsep = getattr(args, "vocal_separator", "demucs")
        sep_out = work_dir / "demucs_output"
        sep_out.mkdir(exist_ok=True)
        try:
            if _vsep == "mdx_net":
                from pathlib import Path as _Path
                _mdx_model = getattr(args, "mdx_model", "UVR-MDX-NET-Voc_FT")
                print(f"[MUSIC] Separating vocals with MDX-Net ({_mdx_model})...", flush=True)
                music_wav = extract_vocals_mdx(audio_wav, sep_out, model_name=_mdx_model)
                # MDX-Net also produces a vocals stem alongside the instrumental
                if music_wav and music_wav.exists():
                    _voc_candidate = music_wav.parent / music_wav.name.replace("(Instrumental)", "(Vocals)")
                    if not _voc_candidate.exists():
                        # fallback: any file with "Vocals" in name in same dir
                        for _f in music_wav.parent.iterdir():
                            if "vocal" in _f.name.lower() and "instrumental" not in _f.name.lower():
                                _voc_candidate = _f
                                break
                    if _voc_candidate.exists():
                        vocals_wav = _voc_candidate
            else:
                _model_flag = ["--name", "htdemucs_ft"] if _vsep == "htdemucs_ft" else []
                print(f"[MUSIC] Separating music with Demucs ({_vsep})...", flush=True)
                import sys as _sys, subprocess as _sp
                cmd = [_sys.executable, "-m", "demucs", "--two-stems", "vocals",
                       *_model_flag, "-o", str(sep_out), str(audio_wav)]
                result = _sp.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr[-500:])
                stem = audio_wav.stem
                model_subdir = "htdemucs_ft" if _vsep == "htdemucs_ft" else "htdemucs"
                candidate = sep_out / model_subdir / stem / "no_vocals.wav"
                _voc_cand = sep_out / model_subdir / stem / "vocals.wav"
                if not candidate.exists():
                    # fallback search
                    for d in sep_out.iterdir():
                        c = d / stem / "no_vocals.wav"
                        if c.exists():
                            candidate = c
                            _voc_cand = c.parent / "vocals.wav"
                            break
                if candidate.exists():
                    music_wav = candidate
                if _voc_cand.exists():
                    vocals_wav = _voc_cand
            if music_wav and music_wav.exists():
                print(f"[MUSIC] Music track: {music_wav}", flush=True)
            if vocals_wav and vocals_wav.exists():
                print(f"[MUSIC] Vocals track: {vocals_wav}", flush=True)
        except Exception as e:
            print(f"[MUSIC] Separation failed ({e}), skipping music", flush=True)
        finally:
            # Uvoľni VRAM po separácii pred Whisperom
            gc.collect()
            torch.cuda.empty_cache()

    # ── Preset resolver ────────────────────────────────────────────────────────
    # Resolve effective pipeline flags from --preset (can be overridden by explicit flags)
    _preset = getattr(args, "preset", "") or ""
    # args.phonetic_respelling is None if not explicitly passed (BooleanOptionalAction, default=None)
    _phon_explicit = getattr(args, "phonetic_respelling", None)  # None = use preset default
    if _preset == "production_dub":
        # Clean dubbing: trust the TTS model, minimal text mutation. Phonetic OFF unless user forces.
        _phonetic_on     = _phon_explicit if _phon_explicit is not None else False
        _sql_tts_on      = getattr(args, "sql_tts_normalize", False)     # OFF unless user forces
        _normalizer_mode = getattr(args, "normalizer_mode", "") or "light"
        # Disable AUTO voice cloning from source (EN reference audio causes artifacts with SK model).
        # Manual cloning via --clone_voice or --xtts_speaker_wav still works.
        if not getattr(args, "clone_voice", False) and not getattr(args, "xtts_speaker_wav", ""):
            args.no_voice_clone = True
    elif _preset == "teaching_sql":
        # SQL/IT teaching video: all normalizers ON + backaudio fill for silent slots
        _phonetic_on     = True
        _sql_tts_on      = True
        _normalizer_mode = getattr(args, "normalizer_mode", "") or "full"
        if not getattr(args, "backaudio_volume", 0.0):
            args.backaudio_volume = 0.15
    else:
        # Legacy / no preset: phonetic ON by default (PHONETIC_MAP fixes cool→kúl, Java→Džava, etc.)
        _phonetic_on     = _phon_explicit if _phon_explicit is not None else True
        _sql_tts_on      = getattr(args, "sql_tts_normalize", False)
        _normalizer_mode = getattr(args, "normalizer_mode", "") or "full"

    # content_type=sql always enables SQL TTS normalization + backaudio fill
    _content_type_sql = getattr(args, "content_type", "general") == "sql"
    if _content_type_sql:
        _sql_tts_on = True
        if not getattr(args, "backaudio_volume", 0.0):
            args.backaudio_volume = 0.15
        # SQL teaching videos have no background music — disable Demucs
        # so backaudio (original audio at 0.15 vol) can fill silent gaps
        args.keep_music = False

    print(f"[PRESET] preset={_preset or 'none'}  phonetic={_phonetic_on}  "
          f"sql_tts={_sql_tts_on}  normalizer={_normalizer_mode}", flush=True)

    # --- Shortcut: skip STT + translation + adaptation, use existing segments JSON ---
    segments: list[dict] = []
    if getattr(args, "use_existing_segments", False):
        import json as _json
        _seg_path = out_dir / f"{stem}_{tgt}_segments.json"
        if not _seg_path.exists():
            raise SystemExit(f"[ERROR] --use_existing_segments: segments file not found: {_seg_path}")
        _seg_data = _json.loads(_seg_path.read_text(encoding="utf-8"))
        translated = _seg_data.get("segments", _seg_data) if isinstance(_seg_data, dict) else _seg_data
        print(f"[SEGMENTS] Loaded {len(translated)} segments from {_seg_path}", flush=True)
        # Jump directly to TTS (skip all blocks below until TTS engine selection)
        goto_tts = True
    else:
        goto_tts = False

    content_type = getattr(args, "content_type", "general")
    _mixed_technical_terms = bool(getattr(args, "mixed_technical_terms", False))
    _whisper_lang: str = "en"

    if not goto_tts:
        # --- Transcription ---
        # Build effective Whisper prompt: user prompt + auto SQL terminology when content_type == "sql"
        _user_prompt = getattr(args, "whisper_prompt", "") or ""
        if content_type == "sql":
            _effective_prompt = (_user_prompt + ", " + SQL_WHISPER_PROMPT).lstrip(", ")
            print(f"[WHISPER] SQL mode: injecting terminology prompt ({len(_effective_prompt)} chars)", flush=True)
        else:
            _effective_prompt = _user_prompt

        print(f"[WHISPER] device={device}", flush=True)
        if args.vad:
            result = transcribe_whisper_vad(
                str(audio_wav),
                model_name=args.whisper_model,
                download_root=args.whisper_root,
                device=device,
                initial_prompt=_effective_prompt,
                vad_threshold=args.vad_threshold,
                min_speech_ms=args.vad_min_speech_ms,
                min_silence_ms=args.vad_min_silence_ms,
                speech_pad_ms=args.vad_speech_pad_ms,
                max_speech_s=args.vad_max_speech_s,
                whisper_no_speech_max=args.whisper_no_speech_max,
                whisper_avg_logprob_min=args.whisper_avg_logprob_min,
                whisper_compression_max=args.whisper_compression_max,
            )
        else:
            result = transcribe_whisper(
                str(audio_wav),
                model_name=args.whisper_model,
                download_root=args.whisper_root,
                device=device,
                initial_prompt=_effective_prompt,
                whisper_no_speech_max=args.whisper_no_speech_max,
                whisper_avg_logprob_min=args.whisper_avg_logprob_min,
                whisper_compression_max=args.whisper_compression_max,
            )
        print("[WHISPER] done", flush=True)

        segments: list[dict] = result.get("segments") or []
        if not segments:
            raise SystemExit("[ERROR] Whisper returned no segments.")
        _whisper_lang = result.get("language", "en")  # ISO 639-1, used by NLLB engine

        # --- Segment normalization / merge / split ---
        segments = normalize_segments(segments)
        # ASR corrections — fix Whisper mishearings for specific content types (e.g. SQL)
        segments = fix_segments_transcription(segments, content_type)
        if args.merge_segments or args.merge_max_gap_s > 0:
            segments = merge_adjacent_segments(
                segments,
                max_gap_s=args.merge_max_gap_s or 0.5,
                max_dur_s=args.merge_max_dur_s or 8.0,
            )
        if getattr(args, "split_segments", False):
            segments = split_long_segments(
                segments,
                max_duration=args.max_segment_dur,
                min_duration=args.min_segment_dur,
            )
        else:
            # Automatická segmentácia dlhých slotov aj bez --split_segments
            # Sloty dlhšie než auto_split_max_dur (default 10s) sa rozložia; 0 = vypnuté
            _auto_split_max = float(getattr(args, "auto_split_max_dur", AUTO_SPLIT_THRESHOLD_S) or 0.0)
            _long_segs = sum(1 for s in segments if (s.get("end", 0) - s.get("start", 0)) > _auto_split_max) if _auto_split_max > 0 else 0
            if _long_segs > 0:
                print(
                    f"[AUTO-SPLIT] {_long_segs} segmentov dlhších než {_auto_split_max:.1f}s → automatické rozdelenie",
                    flush=True,
                )
                segments = split_long_segments(
                    segments,
                    max_duration=_auto_split_max,
                    min_duration=2.0,
                )

    # --- Translation ---
    translated: list[dict] = [] if not goto_tts else translated
    if goto_tts:
        segments = translated
    _sw_src_lang = _whisper_lang if (_whisper_lang and _whisper_lang != "auto") else "en"

    if not goto_tts and getattr(args, "hybrid_translate", False):
        # Hybrid: Google primary (better quality) + MADLAD fallback (SQL/code + Google failures)
        print("[HYBRID] Google Translate primary + MADLAD fallback...", flush=True)
        _madlad_name = getattr(args, "madlad_model", "google/madlad400-7b-mt")
        _cs_caps = tgt in ("cs", "ces")
        _sk_caps = tgt in ("sk", "slk")
        _CAPS_MAP = {
            "WHEN": "když" if _cs_caps else "keď" if _sk_caps else None,
            "WHERE": "kde", "ELSE": "jinak" if _cs_caps else "inak" if _sk_caps else None,
            "WITH": "s",
            "THEN": "pak" if _cs_caps else "potom" if _sk_caps else None,
            "IF": "pokud" if _cs_caps else "ak" if _sk_caps else None,
        }
        import re as _re
        for i, seg in enumerate(segments):
            src = (seg.get("text") or "").strip()
            print(f"[SRC] {src}", flush=True)
            if not src:
                translated.append({"start": seg["start"], "end": seg["end"], "text": ""})
                continue
            use_madlad_for_seg = is_sql_heavy(src)
            trans = ""
            if not use_madlad_for_seg:
                trans = translate_segment_with_google(src, tgt)
                if trans:
                    print(f"[HYBRID/GOOGLE] Seg {i+1}: {trans[:60]}", flush=True)
                else:
                    print(f"[HYBRID] Google failed seg {i+1}, falling back to MADLAD", flush=True)
                    use_madlad_for_seg = True
            if use_madlad_for_seg:
                protected_src, token_map = protect_phonetic_terms(src, use_guillemets=True)
                trans = translate_segment_with_madlad(protected_src, tgt, model_name=_madlad_name)
                trans = restore_phonetic_terms(trans, token_map)
                trans = strip_guillemets(trans)
                if _phonetic_on:
                    trans = apply_phonetic_respelling(trans)
                for _eng, _cz in _CAPS_MAP.items():
                    if _cz and _re.search(r'\b' + _eng + r'\b', trans):
                        trans = _re.sub(r'\b' + _eng + r'\b', _cz, trans)
                print(f"[HYBRID/MADLAD] Seg {i+1}: {trans[:60]}", flush=True)
            if not trans:
                trans = src
            print(f"[TR ] {trans}", flush=True)
            translated.append({"start": seg["start"], "end": seg["end"], "text": trans, "text_src": src})

        # ── Hybrid QA pass — LLM dostane EN + SK, opraví gramatiku a vernosť ──
        if getattr(args, "hybrid_qa", False):
            _qa_model_path = getattr(args, "llama_model", "")
            if _qa_model_path:
                print(f"[HYBRID-QA] Loading LLM: {_qa_model_path}", flush=True)
                try:
                    from llama_cpp import Llama as _QALlama
                    _qa_llm = _QALlama(
                        model_path=_qa_model_path,
                        n_ctx=getattr(args, "llama_ctx", 2048),
                        n_gpu_layers=getattr(args, "llama_gpu_layers", 56),
                        verbose=False,
                    )
                    _tgt_name = target_lang_name(tgt)
                    _qa_fixed = _qa_skipped = 0
                    for _qi, _seg in enumerate(translated):
                        _sk = (_seg.get("text") or "").strip()
                        _en = (_seg.get("text_src") or "").strip()
                        if not _sk:
                            continue
                        # Ochrana technických termínov — LLM ich nesmie prepísať
                        _sk_protected, _token_map = protect_phonetic_terms(_sk, use_guillemets=True)
                        # Krok 1: oprav gramatiku/plynulosť
                        _refined_protected = cleanup_with_gemma(
                            _sk_protected,
                            llama_model=_qa_model_path,
                            n_gpu_layers=getattr(args, "llama_gpu_layers", 56),
                            llm_instance=_qa_llm,
                            tgt_lang_name=_tgt_name,
                        )
                        # Obnov chránené termíny
                        _refined = restore_phonetic_terms(_refined_protected, _token_map)
                        _refined = strip_guillemets(_refined)
                        # Krok 2: ak máme EN originál, skontroluj vernosť prekladu
                        if _en and _refined:
                            _qa_ok, _qa_fb = qa_check_tone(
                                original_en=_en,
                                translated=_refined,
                                llama_model=_qa_model_path,
                                llm_instance=_qa_llm,
                                tgt_lang=tgt,
                            )
                            if not _qa_ok:
                                print(f"[HYBRID-QA] Seg {_qi+1} QA FAIL: {_qa_fb}", flush=True)
                                # pri QA fail ponechaj cleanup verziu — aspoň gramatika je ok
                        if _refined and _refined != _sk:
                            _seg["text"] = _refined
                            _seg["text_before_qa"] = _sk
                            _qa_fixed += 1
                        else:
                            _qa_skipped += 1
                    del _qa_llm
                    print(f"[HYBRID-QA] Hotovo — opravených: {_qa_fixed}, nezmenených: {_qa_skipped}", flush=True)
                except Exception as _e:
                    print(f"[HYBRID-QA] QA pass zlyhal ({_e}), pokračujem bez QA.", flush=True)
            else:
                print("[HYBRID-QA] --hybrid_qa: žiaden --llama_model, preskakujem QA.", flush=True)

    elif not goto_tts and getattr(args, "use_google_translate", False):
        _sw_batch = getattr(args, "sw_batch_size", 80)
        _sw_memory = getattr(args, "sw_memory_size", 15)
        print(f"[GOOGLE-SW] Sliding window preklad (batch={_sw_batch}, memory={_sw_memory})", flush=True)
        translated = translate_segments_sliding_window(
            segments=segments,
            tgt_lang=tgt,
            engine="google",
            src_lang=_sw_src_lang,
            content_type=content_type,
            batch_size=_sw_batch,
            memory_size=_sw_memory,
        )
        for i, seg in enumerate(translated):
            seg["text_src"] = (segments[i].get("text") or "").strip()
            print(f"[TR ] {seg.get('text','')}", flush=True)

    elif not goto_tts and getattr(args, "use_multislav5lang", False):
        _multislav_model = getattr(args, "multislav_model", "allegro/multislav-5lang")
        _multislav_supported = {"cs", "ces", "cz", "en", "eng", "pl", "pol", "sk", "slk", "sl", "slv"}
        if (tgt or "").lower() not in _multislav_supported:
            raise ValueError(f"--use_multislav5lang supports only cs/en/pl/sk/sl target languages (got: {tgt})")
        print(f"[MULTISLAV] Translating with {_multislav_model}...", flush=True)
        for i, seg in enumerate(segments):
            src = (seg.get("text") or "").strip()
            print(f"[SRC] {src}", flush=True)
            if not src:
                translated.append({"start": seg["start"], "end": seg["end"], "text": ""})
                continue
            print(f"[MULTISLAV] Segment {i+1}/{len(segments)}", flush=True)
            trans = translate_segment_with_multislav5lang(src, tgt, model_name=_multislav_model)
            if _phonetic_on:
                trans = apply_phonetic_respelling(trans)
            print(f"[TR ] {trans}", flush=True)
            translated.append({"start": seg["start"], "end": seg["end"], "text": trans, "text_src": src})

    elif not goto_tts and getattr(args, "use_madlad_gemma_qa", False):
        _use_google_p1 = getattr(args, "use_google_translate", False)
        print(f"[MGQ] Starting 3-phase batch pipeline (Phase1={'Google' if _use_google_p1 else 'MADLAD'})...", flush=True)
        results = translate_batch_madlad_gemma_qa(
            segments=segments,
            tgt_lang=tgt,
            madlad_model=args.madlad_model,
            gemma_model=args.llama_model,
            gemma_n_ctx=args.llama_ctx,
            gemma_n_gpu_layers=args.llama_gpu_layers,
            qa_model=getattr(args, "qa_model_path", ""),
            qa_n_gpu_layers=getattr(args, "qa_gpu_layers", 32),
            use_google_phase1=_use_google_p1,
        )
        for r in results:
            translated.append({
                "start": r["start"], "end": r["end"],
                "text": r["text_final"], "text_src": r.get("text", ""),
            })

    elif not goto_tts and getattr(args, "use_nllb", False):
        from transformers import pipeline as hf_pipeline
        from term_protection import TermProtector, translate_with_protection
        # NLLB language code map (ISO 639-1/2 → NLLB format)
        _NLLB_MAP = {
            "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "cs": "ces_Latn",
            "sk": "slk_Latn", "pl": "pol_Latn", "hu": "hun_Latn", "uk": "ukr_Cyrl",
            "ru": "rus_Cyrl", "es": "spa_Latn", "it": "ita_Latn",
            # ISO 639-2 / pipeline tgt_lang codes
            "ces": "ces_Latn", "slk": "slk_Latn", "deu": "deu_Latn", "fra": "fra_Latn",
            "pol": "pol_Latn", "hun": "hun_Latn", "ukr": "ukr_Cyrl", "rus": "rus_Cyrl",
            "eng": "eng_Latn", "spa": "spa_Latn", "ita": "ita_Latn",
        }
        _nllb_src_raw = getattr(args, "nllb_src_lang", "auto")
        nllb_src = _NLLB_MAP.get(_whisper_lang, "eng_Latn") if _nllb_src_raw == "auto" \
                   else _nllb_src_raw
        nllb_tgt = _NLLB_MAP.get(tgt, tgt)
        nllb_model = getattr(args, "nllb_model", "facebook/nllb-200-1.3B")
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[NLLB] {nllb_model}  {nllb_src} → {nllb_tgt}  device={_dev}", flush=True)
        _nllb_translator = hf_pipeline(
            "translation", model=nllb_model,
            device=0 if _dev == "cuda" else -1,
            torch_dtype=torch.float16 if _dev == "cuda" else torch.float32,
        )
        _tp = TermProtector()
        _nllb_results = translate_with_protection(segments, _nllb_translator, nllb_src, _tp,
                                                   tgt_lang=nllb_tgt)
        for seg, res in zip(segments, _nllb_results):
            t = res.get("sk_text", "").strip() or seg.get("text", "")
            if _phonetic_on:
                t = apply_phonetic_respelling(t)
            print(f"[TR ] {t}", flush=True)
            translated.append({"start": seg["start"], "end": seg["end"],
                                "text": t, "text_src": seg.get("text", "")})
        del _nllb_translator; gc.collect(); torch.cuda.empty_cache()
        print("[NLLB] done", flush=True)

    elif not goto_tts and getattr(args, "use_madlad", False):
        for i, seg in enumerate(segments):
            src = (seg.get("text") or "").strip()
            print(f"[SRC] {src}", flush=True)
            if not src:
                translated.append({"start": seg["start"], "end": seg["end"], "text": ""})
                continue
            print(f"[MADLAD] Segment {i+1}/{len(segments)}", flush=True)
            protected_src, token_map = protect_phonetic_terms(src, use_guillemets=True)
            trans = translate_segment_with_madlad(protected_src, tgt, model_name=args.madlad_model)
            trans = restore_phonetic_terms(trans, token_map)
            trans = strip_guillemets(trans)
            if _phonetic_on:
                trans = apply_phonetic_respelling(trans)
            # Fix MADLAD caps leakage (SQL keywords left as English: WHEN, WHERE, ELSE, etc.)
            _cs_caps = tgt in ("cs", "ces")
            _sk_caps = tgt in ("sk", "slk")
            _CAPS_MAP = {
                "WHEN": "když" if _cs_caps else "keď" if _sk_caps else None,
                "WHERE": "kde",
                "ELSE": "jinak" if _cs_caps else "inak" if _sk_caps else None,
                "WITH": "s",
                "THEN": "pak" if _cs_caps else "potom" if _sk_caps else None,
                "IF": "pokud" if _cs_caps else "ak" if _sk_caps else None,
            }
            import re as _re
            for _eng, _cz in _CAPS_MAP.items():
                if _cz and _re.search(r'\b' + _eng + r'\b', trans):
                    trans = _re.sub(r'\b' + _eng + r'\b', _cz, trans)
                    print(f"  [CAPS-FIX] {_eng} → {_cz}", flush=True)
            print(f"[TR ] {trans}", flush=True)
            translated.append({"start": seg["start"], "end": seg["end"], "text": trans, "text_src": src})

    elif not goto_tts and getattr(args, "use_gemma4_hf", False):
        _g4_model = getattr(args, "gemma4_hf_model", "/mnt/tts_data/VideoTranslator_studio/models/gemma-4-E4B-it")
        print(f"[GEMMA4-HF] Gemma 4 E4B full-document translation...", flush=True)
        _log_memory("pred gemma4-hf")
        translated = translate_fulldoc_with_gemma4_hf(
            segments=segments,
            tgt_lang=tgt,
            model_id=_g4_model,
        )
        # Subprocess E4B skončil — explicitne uvoľni RAM + VRAM pred refine passem
        gc.collect()
        gc.collect()  # druhý pass zachytí cyklické referencie
        torch.cuda.empty_cache()
        _log_memory("po gemma4-hf")
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        print("[GEMMA4-HF] RAM + VRAM uvoľnená, pripravujem refine pass...", flush=True)

    elif not goto_tts and getattr(args, "use_translategemma_fulldoc", False):
        _tg_model = getattr(args, "translategemma_model", "")
        _tg_layers = getattr(args, "translategemma_gpu_layers", -1)
        _ctx_hint = getattr(args, "refine_context", "")
        print(f"[FULLDOC] Full-document TranslateGemma translation...", flush=True)
        translated = translate_full_document_with_translategemma(
            segments=segments,
            tgt_lang=tgt,
            model_path=_tg_model,
            n_gpu_layers=_tg_layers,
            n_ctx=args.llama_ctx,
            context_hint=_ctx_hint,
        )

    elif not goto_tts and getattr(args, "use_translategemma", False):
        _tg_model = getattr(args, "translategemma_model", "")
        _tg_layers = getattr(args, "translategemma_gpu_layers", -1)
        _sw_batch = getattr(args, "sw_batch_size", 80)
        _sw_memory = getattr(args, "sw_memory_size", 15)
        print(f"[TRANSLATEGEMMA-SW] Sliding window preklad (batch={_sw_batch}, memory={_sw_memory})", flush=True)
        translated = translate_segments_sliding_window(
            segments=segments,
            tgt_lang=tgt,
            engine="translategemma",
            tg_model_path=_tg_model,
            n_gpu_layers=_tg_layers,
            n_ctx=args.llama_ctx,
            content_type=content_type,
            batch_size=_sw_batch,
            memory_size=_sw_memory,
            src_lang=_sw_src_lang,
        )
        # Pridaj text_src a aplikuj phonetic
        for i, seg in enumerate(translated):
            src_text = (segments[i].get("text") or "").strip()
            seg["text_src"] = src_text
            if _phonetic_on and seg.get("text"):
                seg["text"] = apply_phonetic_respelling(seg["text"])
            print(f"[TR ] {seg.get('text','')}", flush=True)
        from translategemma import free_translategemma as _free_tg
        _free_tg()
        gc.collect()
        torch.cuda.empty_cache()
        print("[TRANSLATEGEMMA-SW] Model released from memory", flush=True)

    elif not goto_tts and getattr(args, "use_api_translate", False):
        _sw_batch = getattr(args, "sw_batch_size", 80)
        _sw_memory = getattr(args, "sw_memory_size", 15)
        _api_provider = getattr(args, "api_translate_provider", "chatgpt")
        _api_model = getattr(args, "api_translate_model", "gpt-5-mini-2025-08-07")
        _api_base = getattr(args, "api_translate_base_url", "https://api.openai.com/v1")
        _api_key = (getattr(args, "api_translate_key", "") or os.environ.get("API_TRANS_KEY", "")).strip()
        if not _api_key:
            raise RuntimeError("API translate key missing. Set --api_translate_key or API_TRANS_KEY.")
        print(
            f"[API-SW] Sliding window preklad provider={_api_provider} model={_api_model} "
            f"(batch={_sw_batch}, memory={_sw_memory})",
            flush=True,
        )
        translated = translate_segments_sliding_window(
            segments=segments,
            tgt_lang=tgt,
            engine="api",
            api_key=_api_key,
            api_base_url=_api_base,
            api_model=_api_model,
            api_provider=_api_provider,
            n_ctx=args.llama_ctx,
            temperature=min(args.llama_temp, 0.10),
            content_type=content_type,
            batch_size=_sw_batch,
            memory_size=_sw_memory,
            src_lang=_sw_src_lang,
        )
        for i, seg in enumerate(translated):
            src_text = (segments[i].get("text") or "").strip()
            seg["text_src"] = src_text
            if _phonetic_on and seg.get("text"):
                seg["text"] = apply_phonetic_respelling(seg["text"])
            print(f"[TR ] {seg.get('text','')}", flush=True)

    elif not goto_tts and getattr(args, "grammar_fix_only", False):
        # grammar_fix_only: preskočí preklad, použije zdrojový text priamo (SK audio → transkript → grammar fix)
        print("[GRAMMAR-FIX-ONLY] Preskakujem preklad — zdrojový text pôjde priamo na grammar fix.", flush=True)
        translated = []
        for seg in segments:
            src_text = (seg.get("text") or "").strip()
            translated.append({**seg, "text": src_text, "text_src": src_text})
        print(f"[GRAMMAR-FIX-ONLY] Pripravených {len(translated)} segmentov.", flush=True)

    elif not goto_tts and getattr(args, "use_ollama_translate", False):
        import requests as _requests
        _ol_model   = getattr(args, "ollama_translate_model", "sk-gemma4-26b:latest")
        _ol_url     = getattr(args, "ollama_translate_url", "http://localhost:11434/api/generate")
        _ol_timeout = getattr(args, "ollama_translate_timeout", 180)
        print(f"[OLLAMA-TR] Prekladám cez Ollama model={_ol_model}", flush=True)
        translated = []
        for i, seg in enumerate(segments):
            src_text = (seg.get("text") or "").strip()
            if not src_text:
                translated.append({**seg, "text": "", "text_src": ""})
                continue
            protected_src, _tok_map = protect_phonetic_terms(src_text)
            _prompt = (
                "Si profesionálny prekladateľ. Prelož nasledujúci anglický text do slovenčiny.\n"
                "Dodržuj prirodzený hovorený slovenský jazyk. Zachovaj tón a dĺžku vety.\n"
                "Technické výrazy 'engine', 'engines', 'shader', 'rendering', 'framework' NEPRELEKÁŠ.\n"
                "Slovo 'era'/'eras' prekladaj ako 'éra'/'éry'. Slovo 'went from' prekladaj ako 'prešiel od'.\n"
                "Odpoveď: IBA preložený text, bez vysvetlení.\n\n"
                f"EN: {protected_src}\nSK:"
            )
            try:
                _r = _requests.post(_ol_url, json={"model": _ol_model, "prompt": _prompt, "stream": False}, timeout=_ol_timeout)
                _r.raise_for_status()
                _sk = _r.json().get("response", "").strip()
            except Exception as _e:
                print(f"[OLLAMA-TR] Seg {i} chyba: {_e} — vraciam prázdny reťazec", flush=True)
                _sk = ""
            _sk = restore_phonetic_terms(_sk, _tok_map)
            if _phonetic_on and _sk:
                _sk = apply_phonetic_respelling(_sk)
            translated.append({**seg, "text": _sk, "text_src": src_text})
            print(f"[TR ] {_sk}", flush=True)
        print(f"[OLLAMA-TR] Preložených {len(translated)} segmentov.", flush=True)

    elif not goto_tts and getattr(args, "use_g3_27b_translator", False):
        # Gemma 3 27B multitask v2 (QLoRA fine-tuned, ollama).
        # Single-step: prelozi + adjust length naraz cez constrained prompt.
        # NEPOTREBUJE separátny length adjuster.
        from llm_g3_27b import translate_g3_27b, _ollama_generate
        print(f"[G3-27B-TRANSLATOR] Volám g3-27b-multitask-v2 cez ollama (single-step multitask)...", flush=True)

        # ── Pre-warm Ollama: prvé volanie po štarte načíta 16 GB Q4_K_M model do VRAM,
        # čo trvá ~30-60s. Bez warmupu prvé 1-2 volania timeoutujú a fallbackujú na EN.
        try:
            import time as _t27
            print(f"[G3-27B-TRANSLATOR] Pre-warming model (load do VRAM)...", flush=True)
            _warm_t0 = _t27.time()
            _warm_resp = _ollama_generate("Translate to Slovak.\nHello.", num_predict=20, timeout=180)
            print(f"[G3-27B-TRANSLATOR] Pre-warm OK ({_t27.time()-_warm_t0:.1f}s, resp={_warm_resp[:50]!r})", flush=True)
        except Exception as _warm_err:
            print(f"[G3-27B-TRANSLATOR] Pre-warm zlyhalo (ignorujem): {_warm_err}", flush=True)

        # ── Sentence-merge pre 27B v2 ──────────────────────────────────────────
        # Whisper rozdeľuje vety na sub-segmenty (napr. "I have gone from literally knowing"
        # | "to simple console programs..."). Translator dostane orphan fragment a vyrobí
        # neprirodzenú slovenčinu ("z nevedomia som prešiel k tomu, že viem"). Riešenie:
        # 1. Spojiť za sebou idúce segmenty do logických viet (predošlý nekončí .!?)
        # 2. Preložiť celú vetu ako kontext
        # 3. Časovo proporčne rozdeliť SK späť na N pôvodných segmentov pre TTS timing
        import re as _re_sm

        def _ends_sentence(t: str) -> bool:
            t = (t or "").strip()
            return bool(t) and t[-1] in ".!?…"

        def _starts_lower(t: str) -> bool:
            t = (t or "").lstrip()
            return bool(t) and t[0].islower()

        # Build groups of segment indices that form one sentence
        _groups: list[list[int]] = []
        for _idx, _seg in enumerate(segments):
            _t = (_seg.get("text") or "").strip()
            if not _t:
                _groups.append([_idx])
                continue
            # Pridaj do predošlej skupiny ak: predošlá nekončí .!? AND aktuálna začína malým
            if (_groups and _groups[-1]
                and _starts_lower(_t)
                and not _ends_sentence((segments[_groups[-1][-1]].get("text") or ""))):
                _groups[-1].append(_idx)
            else:
                _groups.append([_idx])

        _n_groups = len(_groups)
        _n_merged = sum(1 for g in _groups if len(g) > 1)
        print(f"[G3-27B-TRANSLATOR] Sentence-merge: {len(segments)} segs → {_n_groups} groups "
              f"({_n_merged} merged)", flush=True)

        def _split_sk_to_segments(sk_full: str, en_lengths: list[int]) -> list[str]:
            """Rozdeľ SK preklad proporčne podľa pomeru EN dĺžok. Hľadá najbližšiu medzeru."""
            sk_full = (sk_full or "").strip()
            if len(en_lengths) == 1:
                return [sk_full]
            total_en = sum(en_lengths) or 1
            sk_len = len(sk_full)
            cuts = []
            cum = 0
            for n in en_lengths[:-1]:
                cum += n
                target = int(round(cum / total_en * sk_len))
                # Nájdi najbližšiu medzeru v okne ±20 znakov
                best = target
                for off in range(0, 25):
                    if target + off < sk_len and sk_full[target + off] == " ":
                        best = target + off
                        break
                    if target - off >= 0 and sk_full[target - off] == " ":
                        best = target - off
                        break
                cuts.append(best)
            # Vyrob fragmenty
            parts = []
            prev = 0
            for c in cuts:
                parts.append(sk_full[prev:c].strip())
                prev = c
            parts.append(sk_full[prev:].strip())
            return parts

        translated = [None] * len(segments)
        for _g in _groups:
            _en_texts = [(segments[i].get("text") or "").strip() for i in _g]
            _merged_en = " ".join(_en_texts)
            if not _merged_en:
                for i in _g:
                    translated[i] = {**segments[i], "text": ""}
                continue
            # Slot total: súčet trvaní v skupine
            _slot_total = sum(
                max(0.0, float(segments[i].get("end", 0)) - float(segments[i].get("start", 0)))
                for i in _g
            )
            if _slot_total > 0.1:
                _sk_full = translate_g3_27b(_merged_en, target_seconds=_slot_total,
                                             target_cps=15.0, fallback=_merged_en)
            else:
                _sk_full = translate_g3_27b(_merged_en, fallback=_merged_en)

            if len(_g) == 1:
                translated[_g[0]] = {**segments[_g[0]], "text": _sk_full}
            else:
                # Rozdeľ SK proporčne podľa pomeru EN dĺžok
                _en_lens = [len(t) for t in _en_texts]
                _sk_parts = _split_sk_to_segments(_sk_full, _en_lens)
                if len(_sk_parts) != len(_g):
                    # Fallback: rovnaký SK pre prvý, prázdny pre ostatné
                    _sk_parts = [_sk_full] + [""] * (len(_g) - 1)
                for i, sk_part in zip(_g, _sk_parts):
                    translated[i] = {**segments[i], "text": sk_part}

        for i, seg in enumerate(translated):
            src_text = (segments[i].get("text") or "").strip()
            seg["text_src"] = src_text
            if _phonetic_on and seg.get("text"):
                seg["text"] = apply_phonetic_respelling(seg["text"])

    elif not goto_tts and getattr(args, "use_g3_v1_translator", False):
        # Gemma 3 12B v1 translator (QLoRA fine-tuned, ollama).
        # content_type=technical/programming/sql → model zachová EN tech termy
        _ct = getattr(args, "content_type", "general")
        print(f"[G3-V1-TRANSLATOR] Volám g3-12b-translator-v1 cez ollama (content_type={_ct})...", flush=True)
        from translation import translate_fulldoc_with_g3_v1
        translated = translate_fulldoc_with_g3_v1(
            segments=segments, tgt_lang=tgt, src_lang=_sw_src_lang, content_type=_ct)
        for i, seg in enumerate(translated):
            src_text = (segments[i].get("text") or "").strip()
            seg["text_src"] = src_text
            if _phonetic_on and seg.get("text"):
                seg["text"] = apply_phonetic_respelling(seg["text"])

    elif not goto_tts and getattr(args, "use_finetuned_translator", False):
        _ft_model = getattr(args, "finetuned_translator_model", "26b-translator")
        _ft_url = "http://localhost:11434/api/generate"
        print(f"[FINETUNED-TRANSLATOR] Volám {_ft_model} cez ollama (raw API)...", flush=True)
        from translation import translate_fulldoc_with_ollama_translator
        translated = translate_fulldoc_with_ollama_translator(
            segments=segments,
            tgt_lang=tgt,
            ollama_url=_ft_url,
            model=_ft_model,
            src_lang=_sw_src_lang,
        )
        for i, seg in enumerate(translated):
            src_text = (segments[i].get("text") or "").strip()
            seg["text_src"] = src_text
            if _phonetic_on and seg.get("text"):
                seg["text"] = apply_phonetic_respelling(seg["text"])

    elif not goto_tts and getattr(args, "use_lmstudio_translate", False):
        _sw_batch  = getattr(args, "sw_batch_size", 40)
        _sw_memory = getattr(args, "sw_memory_size", 15)
        _ls_url    = getattr(args, "lmstudio_translate_url",     "http://localhost:1234/v1")
        _ls_model  = getattr(args, "lmstudio_translate_model",   "")
        _ls_timeout = int(getattr(args, "lmstudio_translate_timeout", 180))
        _ls_model_key = getattr(args, "lmstudio_model_key", "gemma-4-26b-a4b-it")
        _lms_cli = "/home/vojtech/.lmstudio/bin/lms"

        # Auto-start LM Studio server + model ak nebeží
        import requests as _req_ls
        _ls_api_url = _ls_url.rstrip("/") + "/models"
        _ls_running = False
        try:
            _ls_running = _req_ls.get(_ls_api_url, timeout=3).ok
        except Exception:
            pass

        if not _ls_running:
            import subprocess as _subp, time as _time
            import os as _os
            _lms_env = {**_os.environ, "PATH": f"/home/vojtech/.lmstudio/bin:{_os.environ.get('PATH','')}"}
            print(f"[LMSTUDIO] Server nebeží — spúšťam...", flush=True)
            _subp.Popen([_lms_cli, "server", "start"], env=_lms_env,
                        stdout=_subp.DEVNULL, stderr=_subp.DEVNULL)
            # Čakaj kým server naštartuje (max 30s)
            for _i in range(30):
                _time.sleep(1)
                try:
                    if _req_ls.get(_ls_api_url, timeout=2).ok:
                        print(f"[LMSTUDIO] Server štart OK ({_i+1}s)", flush=True)
                        _ls_running = True
                        break
                except Exception:
                    pass
            if not _ls_running:
                print("[LMSTUDIO] Server sa nespustil — skúšam aj tak.", flush=True)

        # Načítaj model ak nie je v pamäti
        if _ls_running and _ls_model_key:
            try:
                _loaded = _req_ls.get(_ls_api_url, timeout=5).json()
                _loaded_ids = [m.get("id","") for m in _loaded.get("data",[])]
                if not any(_ls_model_key in _id for _id in _loaded_ids):
                    print(f"[LMSTUDIO] Načítavam model {_ls_model_key}...", flush=True)
                    import subprocess as _subp2
                    import os as _os2
                    _lms_env2 = {**_os2.environ, "PATH": f"/home/vojtech/.lmstudio/bin:{_os2.environ.get('PATH','')}"}
                    _subp2.Popen([_lms_cli, "load", _ls_model_key, "--gpu", "max",
                                  "--context-length", "16384"],
                                 env=_lms_env2, stdout=_subp2.DEVNULL, stderr=_subp2.DEVNULL)
                    # Čakaj na načítanie modelu (max 120s)
                    import time as _time2
                    for _j in range(120):
                        _time2.sleep(1)
                        try:
                            _ld2 = _req_ls.get(_ls_api_url, timeout=3).json()
                            if any(_ls_model_key in m.get("id","") for m in _ld2.get("data",[])):
                                print(f"[LMSTUDIO] Model načítaný ({_j+1}s)", flush=True)
                                break
                        except Exception:
                            pass
                    else:
                        print(f"[LMSTUDIO] Model sa nenačítal v čase — pokračujem.", flush=True)
                else:
                    print(f"[LMSTUDIO] Model {_ls_model_key} už v pamäti.", flush=True)
            except Exception as _ls_err:
                print(f"[LMSTUDIO] Kontrola modelu zlyhala: {_ls_err}", flush=True)

        # batch_size: --ls_batch_size override; ak --whole_doc_translate, posiela všetko v 1 dávke
        _ls_batch = int(getattr(args, "ls_batch_size", 50))
        if getattr(args, "whole_doc_translate", False):
            _ls_batch = max(len(segments), 1)  # 1 batch = celý dokument
            print(f"[LMSTUDIO-FULLDOC] WHOLE-DOC mode — všetkých {len(segments)} segmentov v 1 dávke pre maximum kontextu", flush=True)
        print(f"[LMSTUDIO-FULLDOC] Full-doc preklad cez LM Studio API ({_ls_url}, model={_ls_model or _ls_model_key}, batch={_ls_batch})", flush=True)
        translated = translate_fulldoc_with_lmstudio(
            segments=segments,
            tgt_lang=tgt,
            lmstudio_url=_ls_url,
            lmstudio_model=_ls_model,
            lmstudio_timeout=_ls_timeout,
            content_type=content_type,
            src_lang=_sw_src_lang,
            batch_size=_ls_batch,
        )
        for i, seg in enumerate(translated):
            src_text = (segments[i].get("text") or "").strip()
            seg["text_src"] = src_text
            if _phonetic_on and seg.get("text"):
                seg["text"] = apply_phonetic_respelling(seg["text"])
            print(f"[TR ] {seg.get('text','')}", flush=True)

        # Compression / length-adjust pass — buď G3 v1 (compress + expand) alebo 26B compress-only
        _compress_enabled = getattr(args, "compress_overflow", True)
        _target_cps = float(getattr(args, "compress_target_cps", 13.0))
        _use_g3_adj = getattr(args, "use_g3_v1_length_adjuster", False)
        if _compress_enabled and translated:
            _c_stats = {"noop": 0, "deterministic": 0, "lmstudio": 0, "g3_v1": 0, "partial": 0}
            if _use_g3_adj:
                from translation import compress_with_g3_v1
                print(f"[ADJUST] Spúšťam length-adjuster G3 v1 (compress + expand, target_cps={_target_cps})", flush=True)
            else:
                from translation import compress_text_pipeline
                print(f"[COMPRESS] Spúšťam compression pass 26B (target_cps={_target_cps})", flush=True)
            for i, seg in enumerate(translated):
                text = (seg.get("text") or "").strip()
                dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
                target_len = max(20, int(dur * _target_cps))
                if not text:
                    _c_stats["noop"] += 1
                    continue
                if _use_g3_adj:
                    if abs(len(text) - target_len) < 5:
                        _c_stats["noop"] += 1
                        continue
                    adjusted, method = compress_with_g3_v1(text, target_len)
                else:
                    if len(text) <= target_len:
                        _c_stats["noop"] += 1
                        continue
                    adjusted, method = compress_text_pipeline(
                        text, target_len,
                        use_lmstudio_fallback=True,
                        lmstudio_url=_ls_url,
                        lmstudio_model=_ls_model,
                    )
                if adjusted != text:
                    seg["text_original_precompress"] = text
                    seg["text"] = adjusted
                    seg["compress_method"] = method
                _c_stats[method if method in _c_stats else "partial"] += 1
            print(f"[ADJUST] {_c_stats}", flush=True)

        # Uvoľni 26B model pred TTS — VRAM treba pre Fish Speech
        try:
            import subprocess as _subp_unload, os as _os_unload
            _lms_env_unload = {**_os_unload.environ, "PATH": f"/home/vojtech/.lmstudio/bin:{_os_unload.environ.get('PATH','')}"}
            print(f"[LMSTUDIO] Uvoľňujem model {_ls_model_key} pred TTS...", flush=True)
            _subp_unload.run([_lms_cli, "unload", _ls_model_key],
                             env=_lms_env_unload, capture_output=True, timeout=30)
            print(f"[LMSTUDIO] Model uvoľnený.", flush=True)
        except Exception as _unload_err:
            print(f"[LMSTUDIO] Unload varování: {_unload_err}", flush=True)

        # SK corrector pass — opraví skloňovanie po 26B preklade (po compression)
        _sk_corrector_model = getattr(args, "sk_corrector_model", "sk-corrector-v1")
        if _sk_corrector_model and translated:
            print(f"[SK-CORRECTOR] Spúšťam grammar correction pass (model={_sk_corrector_model})...", flush=True)
            try:
                translated = refine_with_sk_corrector(
                    segments=translated,
                    model=_sk_corrector_model,
                )
            except Exception as _corr_err:
                print(f"[SK-CORRECTOR] Zlyhal ({_corr_err}) — pokračujem bez korekcie.", flush=True)

    elif not goto_tts:
        from llama_cpp import Llama
        _sw_batch = getattr(args, "sw_batch_size", 60)
        _sw_memory = getattr(args, "sw_memory_size", 15)
        print(f"[LLAMA-SW] Loading model: {args.llama_model}", flush=True)
        _log_memory("pred llama-sw")
        llm = Llama(model_path=args.llama_model, n_ctx=args.llama_ctx, n_gpu_layers=args.llama_gpu_layers, verbose=False)
        print(f"[LLAMA-SW] Sliding window preklad (batch={_sw_batch}, memory={_sw_memory})", flush=True)
        translated = translate_segments_sliding_window(
            segments=segments,
            tgt_lang=tgt,
            engine="llama",
            llm_instance=llm,
            n_ctx=args.llama_ctx,
            temperature=args.llama_temp,
            content_type=content_type,
            batch_size=_sw_batch,
            memory_size=_sw_memory,
            src_lang=_sw_src_lang,
        )
        for i, seg in enumerate(translated):
            src_text = (segments[i].get("text") or "").strip()
            seg["text_src"] = src_text
            if _phonetic_on and seg.get("text"):
                seg["text"] = apply_phonetic_respelling(seg["text"])
            print(f"[TR ] {seg.get('text','')}", flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        print("[LLAMA-SW] Model released from memory", flush=True)

    # --- Refinement pass (fix grammar/declension with LLM) ---
    # Pre lmstudio engine refinement preskočíme — 26B model už produkuje kvalitnú gramatiku
    _skip_refine_lmstudio = getattr(args, "use_lmstudio_translate", False)
    if not goto_tts and getattr(args, "refine_translation", False) and translated and not _skip_refine_lmstudio:
        _refine_model = getattr(args, "refine_model", "") or args.llama_model
        _refine_layers = getattr(args, "refine_gpu_layers", -1)
        _refine_ctx = getattr(args, "refine_context", "")
        _refine_model_explicit = bool(getattr(args, "refine_model", ""))
        _use_ollama_refine = bool(
            not _refine_model_explicit  # ak je --refine_model nastavený, vždy llama.cpp
            and (
                getattr(args, "use_ollama_adaptation", False)
                or getattr(args, "adapt_provider", "llama_cpp") == "ollama"
            )
        )
        if _use_ollama_refine:
            print(f"[REFINE] Starting full-document refinement pass ({len(translated)} segments)...", flush=True)
            _log_memory("pred refine-ollama")
            try:
                translated = refine_fulldoc_with_ollama(
                    segments=translated,
                    tgt_lang=tgt,
                    url=getattr(args, "adapt_ollama_url", "http://localhost:11434/api/generate"),
                    model=getattr(args, "adapt_ollama_model", "gemma4:26b"),
                    timeout=int(getattr(args, "adapt_ollama_timeout", 120)),
                    n_ctx=args.llama_ctx,
                    context_hint=_refine_ctx,
                )
            except Exception as _refine_err:
                print(f"[REFINE] Zlyhal ({_refine_err}) — pokračujem bez refine.", flush=True)
        else:
            # Overenie či je llama_cpp dostupné v aktuálnom prostredí
            try:
                from llama_cpp import Llama as _LlamaCheck
                _llama_available = True
            except Exception:
                _llama_available = False
            if not _llama_available:
                print("[REFINE] llama_cpp nie je dostupné v tomto prostredí — refine preskočený.", flush=True)
            elif not _refine_model or not Path(_refine_model).exists():
                print(f"[REFINE] Model nenájdený: {_refine_model!r} — refine preskočený.", flush=True)
            else:
                print(f"[REFINE] Starting full-document refinement pass ({len(translated)} segments)...", flush=True)
                _log_memory("pred refine-llama")
                try:
                    translated = refine_fulldoc_with_llama(
                        segments=translated,
                        tgt_lang=tgt,
                        llama_model=_refine_model,
                        n_ctx=args.llama_ctx,
                        n_gpu_layers=_refine_layers,
                        context_hint=_refine_ctx,
                    )
                except Exception as _refine_err:
                    print(f"[REFINE] Zlyhal ({_refine_err}) — pokračujem bez refine.", flush=True)

    # --- Remove consecutive duplicate translations ---
    if not goto_tts and translated:
        _deduped = []
        for _seg in translated:
            if _deduped and _seg.get("text", "").strip() == _deduped[-1].get("text", "").strip() and _seg.get("text", "").strip():
                print(f"[DEDUP] Removed duplicate segment at {_seg['start']:.1f}s: '{_seg['text'][:60]}'", flush=True)
            else:
                _deduped.append(_seg)
        translated = _deduped

    # --- Gender fix (deterministic regex, no LLM needed) ---
    _gender_fix = getattr(args, "gender_fix", "")
    if not goto_tts and _gender_fix and translated:
        print(f"[GENDER-FIX] Applying gender fix: {_gender_fix}", flush=True)
        if _gender_fix == "sk_f":
            translated = apply_gender_fix_sk_feminine(translated)

    # Slovak grammar fixes — SQL function rewrite only when SQL mode active
    if not goto_tts and tgt in ("slk", "sk") and translated:
        translated = fix_sk_conditional_grammar(translated)
        translated = fix_sk_noun_declension(translated)
        translated = fix_sk_foreign_name_declension(translated)
        if _sql_tts_on:
            translated = fix_sk_sql_function_names(translated)
            translated = apply_sql_source_guided_cleanup(translated)
        if _mixed_technical_terms and content_type == "technical":
            translated = apply_linux_source_guided_cleanup(translated, mixed_mode=True)

    # Fallback: ak segment neobsahuje slovenské znaky → Google Translate
    if not goto_tts and tgt in ("slk", "sk") and translated:
        _SK_CHARS = set("áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ")
        _fallback_count = 0
        for _seg in translated:
            _txt = (_seg.get("text") or "").strip()
            if len(_txt) > 8 and not any(c in _SK_CHARS for c in _txt):
                try:
                    _fixed = translate_segment_with_google(_txt, tgt, src_lang="en")
                    if _fixed and _fixed.strip() != _txt:
                        print(f"[FALLBACK-GT] {_txt[:60]} → {_fixed[:60]}", flush=True)
                        _seg["text"] = _fixed
                        _fallback_count += 1
                except Exception as _e:
                    print(f"[FALLBACK-GT] chyba: {_e}", flush=True)
        if _fallback_count:
            print(f"[FALLBACK-GT] Opravených {_fallback_count} nepreložených segmentov.", flush=True)
        translated = apply_sql_source_guided_cleanup(translated)
        if _mixed_technical_terms and content_type == "technical":
            translated = apply_linux_source_guided_cleanup(translated, mixed_mode=True)

    # --- Text output ---
    if not goto_tts and args.text_out:
        txt_path = Path(args.text_out)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(
            "\n\n".join(f"[{s['start']:.1f}s] {s['text']}" for s in translated),
            encoding="utf-8",
        )

    if not goto_tts:
        # --- SRT ---
        srt_path = out_dir / f"{stem}_{tgt}.srt"
        _write_srt(translated, srt_path)

        # --- JSON (for QC / downstream tools) ---
        json_out_path = Path(args.json_out) if getattr(args, "json_out", "") else \
                        out_dir / f"{stem}_{tgt}_segments.json"
        import json as _json
        _write_segments_json(translated, json_out_path)

        # --- SK TTS audit (scoring + bezpečné fixy) ---
        _audit_results = []
        if tgt in ("sk", "slk"):
            try:
                from audit_sk_tts_json import audit_json, write_report, compute_metrics, write_metrics
                _audited_segs, _audit_results = audit_json(json_out_path, apply_fixes=True)
                for _ai, _aseg in enumerate(_audited_segs):
                    if _ai < len(translated):
                        translated[_ai]["text"] = _aseg["text"]
                        translated[_ai]["_audit"] = _aseg.get("_audit", {})
                _audit_report = out_dir / f"{stem}_{tgt}_audit_report.txt"
                write_report(_audit_results, _audit_report)
                _flagged = sum(1 for r in _audit_results if r["score"] > 0)
                print(f"[AUDIT] {len(_audit_results)} segmentov | flagged: {_flagged} | report: {_audit_report}", flush=True)
            except Exception as _ae:
                print(f"[AUDIT] Chyba pri audite: {_ae}", flush=True)

        # --- SK Gemma repair (len flagged segmenty) ---
        _repair_count = 0
        if tgt in ("sk", "slk") and _audit_results:
            _use_ollama_repair = bool(
                getattr(args, "use_ollama_adaptation", False)
                or getattr(args, "adapt_provider", "llama_cpp") == "ollama"
            )
            _repair_model = getattr(args, "refine_model", "") or getattr(args, "llama_model", "")
            if _use_ollama_repair:
                try:
                    from translation import repair_sk_flagged_with_ollama
                    _before = [s.get("text","") for s in translated]
                    translated = repair_sk_flagged_with_ollama(
                        translated,
                        _audit_results,
                        url=getattr(args, "adapt_ollama_url", "http://localhost:11434/api/generate"),
                        model=getattr(args, "adapt_ollama_model", "gemma4:26b"),
                        timeout=int(getattr(args, "adapt_ollama_timeout", 120)),
                        min_score=1,
                    )
                    _repair_count = sum(1 for a, b in zip(_before, [s.get("text","") for s in translated]) if a != b)
                    print(f"[REPAIR/OLLAMA] Opravených segmentov: {_repair_count}", flush=True)
                except Exception as _re:
                    print(f"[REPAIR/OLLAMA] Chyba: {_re}", flush=True)
            elif _repair_model and Path(_repair_model).exists():
                try:
                    from translation import repair_sk_flagged_with_gemma
                    _repair_layers = getattr(args, "refine_gpu_layers", -1)
                    _before = [s.get("text","") for s in translated]
                    translated = repair_sk_flagged_with_gemma(
                        translated, _audit_results,
                        llama_model=_repair_model,
                        n_gpu_layers=int(_repair_layers),
                        min_score=1,
                    )
                    _repair_count = sum(1 for a, b in zip(_before, [s.get("text","") for s in translated]) if a != b)
                    print(f"[REPAIR] Opravených segmentov: {_repair_count}", flush=True)
                except Exception as _re:
                    print(f"[REPAIR] Chyba: {_re}", flush=True)
            else:
                print("[REPAIR] Žiadny refine_model — preskočené.", flush=True)

        # --- SK metriky (uloží históriu behov) ---
        if tgt in ("sk", "slk") and _audit_results:
            try:
                _metrics = compute_metrics(_audit_results, repaired_count=_repair_count)
                _metrics["video"] = stem
                _metrics_path = out_dir / f"{stem}_{tgt}_metrics.json"
                write_metrics(_metrics, _metrics_path)
                print(
                    f"[METRICS] flagged={_metrics['flagged_count']} "
                    f"bad_terms={_metrics['bad_term_count']} "
                    f"hallucinations={_metrics['hallucination_count']} "
                    f"repaired={_metrics['gemma_repaired_count']} "
                    f"| {_metrics_path}", flush=True
                )
            except Exception as _me:
                print(f"[METRICS] Chyba: {_me}", flush=True)

        if tgt in ("sk", "slk") and translated:
            if _sql_tts_on:
                translated = apply_sql_source_guided_cleanup(translated)
            if _mixed_technical_terms and content_type == "technical":
                translated = apply_linux_source_guided_cleanup(translated, mixed_mode=True)
            _write_srt(translated, srt_path)
            _write_segments_json(translated, json_out_path)

        # Translation-only now continues through all pre-TTS text passes
        # (mini_level11, adaptation, pre-TTS orchestration) and returns later.

        # --- Free MADLAD VRAM before loading TTS model ---
        free_madlad_model()
        free_multislav5lang_model()
        gc.collect()
        torch.cuda.empty_cache()

    # --- mini_level11 rewrite pass (EN source -> SK spoken rewrite + SQL guard) ---
    _mini_level11_stats = None
    if not goto_tts and getattr(args, "use_mini_level11", False) and translated:
        if tgt in ("sk", "slk"):
            try:
                from mini_level11_bridge import rewrite_segments_batch as _mini_level11_rewrite

                _mini_stats_path = str(out_dir / f"{stem}_mini_level11_stats.json")
                _mini_use_ollama = bool(
                    getattr(args, "use_ollama_adaptation", False)
                    or getattr(args, "adapt_provider", "llama_cpp") == "ollama"
                )
                translated, _mini_level11_stats = _mini_level11_rewrite(
                    translated,
                    use_ollama=_mini_use_ollama,
                    ollama_url=getattr(args, "adapt_ollama_url", "http://localhost:11434/api/generate"),
                    ollama_model=getattr(args, "adapt_ollama_model", "gemma4:26b"),
                    ollama_timeout=int(getattr(args, "adapt_ollama_timeout", 120)),
                    verbose=True,
                    stats_path=_mini_stats_path,
                )
                _mini_json_path = out_dir / f"{stem}_{tgt}_mini_level11_segments.json"
                _write_segments_json(translated, _mini_json_path)
                print(
                    f"[MINI-L11] changed={_mini_level11_stats.changed} "
                    f"ollama={_mini_level11_stats.ollama} "
                    f"template={_mini_level11_stats.source_template} "
                    f"fallback={_mini_level11_stats.fallback} "
                    f"| {_mini_json_path}",
                    flush=True,
                )
            except Exception as _mini_err:
                print(f"[MINI-L11] Chyba: {_mini_err}", flush=True)
        else:
            print("[MINI-L11] Preskočené — aktuálne podporuje len slovenský výstup.", flush=True)

    # --- Pre-TTS text stack (Level 12 / Level 13 / adaptation / orchestration) ---
    _level12_summary = None
    _level13_summary = None
    _pre_tts_levels_summary = None
    _adapt_stats = None
    _cps_cal = None
    _cps_cal_path = None
    _ollama_release_after_text = bool(
        not goto_tts
        and (
            getattr(args, "use_ollama_adaptation", False)
            or getattr(args, "adapt_provider", "llama_cpp") == "ollama"
        )
    )
    _ollama_release_url = getattr(args, "adapt_ollama_url", "http://localhost:11434/api/generate")
    _ollama_release_model = getattr(args, "adapt_ollama_model", "gemma4:26b")
    _ollama_release_timeout = int(getattr(args, "adapt_ollama_timeout", 120))
    try:
        if not goto_tts and getattr(args, "use_level6_plus", False) and translated and tgt in ("sk", "slk"):
            try:
                from src.level12_manager import Level12Manager, summarize_level12_segments
                from src.utils_io import save_json as _save_level_json

                _level12_manager = Level12Manager()
                translated = _level12_manager.process_segments(translated)
                _level12_summary = summarize_level12_segments(translated)
                _level12_path = out_dir / f"{stem}_{tgt}_pre_tts_level12_smart_timing.json"
                _write_segments_json(translated, _level12_path)
                (out_dir / "final").mkdir(parents=True, exist_ok=True)
                _save_level_json(out_dir / "final" / f"{stem}_{tgt}_pre_tts_level12_summary.json", _level12_summary)
                print(
                    f"[LEVEL12] changed={_level12_summary['changed_segments']} "
                    f"splits={_level12_summary['suggested_splits']} "
                    f"over_target={_level12_summary['over_target_before']}→{_level12_summary['over_target_after']} "
                    f"| {_level12_path}",
                    flush=True,
                )
            except Exception as _level12_err:
                print(f"[LEVEL12] Chyba: {_level12_err}", flush=True)

        if not goto_tts and getattr(args, "use_level6_plus", False) and translated and tgt in ("sk", "slk"):
            try:
                from src.level13_manager import Level13Manager, summarize_level13_segments
                from src.utils_io import save_json as _save_level_json

                _level13_manager = Level13Manager(
                    provider=getattr(args, "adapt_provider", "llama_cpp"),
                    ollama_url=_ollama_release_url,
                    ollama_model=_ollama_release_model,
                    ollama_timeout=_ollama_release_timeout,
                    max_ollama_segments=8,
                    verbose=True,
                )
                translated = _level13_manager.process_segments(translated)
                _level13_summary = summarize_level13_segments(translated)
                _level13_path = out_dir / f"{stem}_{tgt}_pre_tts_level13_multipass_rewrite.json"
                _write_segments_json(translated, _level13_path)
                (out_dir / "final").mkdir(parents=True, exist_ok=True)
                _save_level_json(out_dir / "final" / f"{stem}_{tgt}_pre_tts_level13_summary.json", _level13_summary)
                print(
                    f"[LEVEL13] changed={_level13_summary['changed_segments']} "
                    f"providers={_level13_summary['providers']} "
                    f"avg_cps={_level13_summary['avg_cps_before']:.2f}→{_level13_summary['avg_cps_after']:.2f} "
                    f"| {_level13_path}",
                    flush=True,
                )
            except Exception as _level13_err:
                print(f"[LEVEL13] Chyba: {_level13_err}", flush=True)

        # --- Cross-segment deduplication (Level 14 — po L11/L13, pred adaptation) ---
        if not goto_tts and getattr(args, "tgt_lang", "").startswith("sk"):
            try:
                _xseg_audit: list[dict] = []
                translated, _xseg_changed = _cross_segment_dedup_check(translated, _xseg_audit)
                if _xseg_changed:
                    print(
                        f"[XSEG-DEDUP] Odstránené intro repetície: {_xseg_changed} segmentov",
                        flush=True,
                    )
                    # Uložiť do audit JSON ak je k dispozícii out_dir
                    try:
                        _xseg_path = out_dir / f"{stem}_{tgt}_xseg_dedup_audit.json"
                        import json as _xseg_json
                        with open(_xseg_path, "w", encoding="utf-8") as _xf:
                            _xseg_json.dump(_xseg_audit, _xf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
            except Exception as _xseg_err:
                print(f"[XSEG-DEDUP] Chyba: {_xseg_err}", flush=True)

        if not goto_tts and getattr(args, "use_text_adaptation", True):
            from text_adaptation import adapt_segments_batch
            _adapt_llm = None
            _adapt_use_ollama = bool(
                getattr(args, "use_ollama_adaptation", False)
                or getattr(args, "adapt_provider", "llama_cpp") == "ollama"
            )
            if _adapt_use_ollama:
                _ollama_url = _ollama_release_url
                _ollama_model = _ollama_release_model
                _ollama_timeout = _ollama_release_timeout
                try:
                    import requests as _requests

                    _probe_url = _ollama_aux_url(_ollama_url, "/api/tags")
                    _probe_resp = _requests.get(_probe_url, timeout=min(_ollama_timeout, 5))
                    _probe_resp.raise_for_status()

                    def _adapt_llm(system: str, user: str) -> str:
                        prompt = f"{system}\n\n{user}".strip()
                        response = _requests.post(
                            _ollama_url,
                            json={
                                "model": _ollama_model,
                                "prompt": prompt,
                                "stream": False,
                            },
                            timeout=_ollama_timeout,
                        )
                        response.raise_for_status()
                        payload = response.json()
                        return str(payload["response"]).strip()

                    print(
                        f"[ADAPT] Using Ollama for text adaptation: model={_ollama_model} url={_ollama_url}",
                        flush=True,
                    )
                except Exception as _e:
                    print(f"[ADAPT] Ollama unavailable ({_e}), adaptation will estimate-only", flush=True)
                    _adapt_llm = None
            else:
                _adapt_llm_path = "" if getattr(args, "no_adapt_llm", False) else getattr(args, "llama_model", "")
                if _adapt_llm_path:
                    try:
                        from llama_cpp import Llama as _Llama
                        print(f"[ADAPT] Loading LLM for text adaptation: {_adapt_llm_path}", flush=True)
                        _adapt_llm = _Llama(
                            model_path=_adapt_llm_path,
                            n_ctx=getattr(args, "llama_ctx", 2048),
                            n_gpu_layers=getattr(args, "llama_gpu_layers", -1),
                            verbose=False,
                        )
                    except Exception as _e:
                        print(f"[ADAPT] LLM load failed ({_e}), adaptation will estimate-only", flush=True)
            _adapt_stats_path = str(out_dir / f"{stem}_adapt_stats.json")
            # ── CPS kalibrácia z predchádzajúcich runov ───────────────────────
            from cps_calibrator import CPSCalibrator
            _cps_cal_path = out_dir / "cps_calibration.json"
            _cps_cal = CPSCalibrator()
            _cps_cal.load(_cps_cal_path)
            print(
                f"[CPS] Kalibrácia: {_cps_cal.effective_cps:.2f} c/s "
                f"(konfidencia={_cps_cal.confidence}, n={_cps_cal._n})",
                flush=True,
            )
            translated, _adapt_stats = adapt_segments_batch(
                translated, llm=_adapt_llm, verbose=True,
                stats_path=_adapt_stats_path,
                chars_per_sec=_cps_cal.effective_cps,
            )
            del _adapt_llm

        if not goto_tts and getattr(args, "use_level6_plus", False) and translated:
            translated, _pre_tts_levels_summary = _run_pre_tts_level6_plus(
                translated_segments=translated,
                out_dir=out_dir,
                stem=stem,
                tgt=tgt,
                args=args,
            )

        if not goto_tts and tgt in ("sk", "slk") and translated:
            if _sql_tts_on:
                translated = apply_sql_source_guided_cleanup(translated)
            if _mixed_technical_terms and content_type == "technical":
                translated = apply_linux_source_guided_cleanup(translated, mixed_mode=True)

        # --- Slovak grammar fix: oprava predložkových väzieb (skloňovanie) ---
        _grammar_fix_enabled = (
            not goto_tts
            and tgt in ("sk", "slk")
            and translated
            and getattr(args, "adapt_provider", "ollama") == "ollama"
            and getattr(args, "adapt_ollama_url", "")
        )
        if _grammar_fix_enabled:
            from translation import fix_grammar_ollama
            _gf_url   = getattr(args, "adapt_ollama_url",   "http://localhost:11434/api/generate")
            _gf_model = getattr(args, "adapt_ollama_model", "llama3.1:8b")
            print(f"[GRAMMAR] Opravujem predložkové väzby ({_gf_model})...", flush=True)
            translated = fix_grammar_ollama(
                translated,
                ollama_url=_gf_url,
                ollama_model=_gf_model,
                timeout=30,
                batch_size=10,
            )

        if _pre_tts_levels_summary is not None and (_level12_summary is not None or _level13_summary is not None):
            from src.utils_io import save_json as _save_level_json

            if _level12_summary is not None:
                _pre_tts_levels_summary["level12"] = _level12_summary
            if _level13_summary is not None:
                _pre_tts_levels_summary["level13"] = _level13_summary
            _save_level_json(
                out_dir / "final" / f"{stem}_{tgt}_pre_tts_level_sidecar_summary.json",
                _pre_tts_levels_summary,
            )
    finally:
        if _ollama_release_after_text:
            _unload_ollama_model(
                _ollama_release_url,
                _ollama_release_model,
                timeout=min(_ollama_release_timeout, 20),
            )

    # Keep JSON/SRT in sync with the final pre-TTS text that will feed TTS.
    if not goto_tts:
        srt_path = out_dir / f"{stem}_{tgt}.srt"
        json_out_path = Path(args.json_out) if getattr(args, "json_out", "") else out_dir / f"{stem}_{tgt}_segments.json"
        _write_srt(translated, srt_path)
        _write_segments_json(translated, json_out_path)

    if getattr(args, "translation_only", False):
        print("[TRANSLATION-ONLY] Končím po všetkých pre-TTS textových krokoch / JSON bez TTS.", flush=True)
        free_madlad_model()
        free_multislav5lang_model()
        gc.collect()
        torch.cuda.empty_cache()
        return

    # ── Pre-TTS CPS expansion pass ──────────────────────────────────────────────
    # Segmenty kde je text oveľa kratší ako slot (CPS < prah) dostanú ollama expand.
    # Toto rieši prípady ako "Príklad." v 5s slote kde adaptácia nezmenila text.
    # POZNÁMKA: aj po gemma4 preklade môžu dlhé segmenty (10-11s) mať krátke preklady —
    # gemma4 ich zhrnie namiesto plného prekladu. Preto je CPS repair aktívny vždy.
    # Pre gemma4 beh: používame llama3.1:8b (malý, CPU-friendly, nezaberá VRAM).
    _cps_expand_threshold = float(getattr(args, "cps_expand_threshold", 9.0))
    _cps_expand_min_slot  = float(getattr(args, "cps_expand_min_slot", 2.0))
    _gemma4_did_translate = (
        getattr(args, "hybrid_translate", False)
        or getattr(args, "use_gemma4_hf", False)
    )
    _cps_expand_enabled   = (
        not getattr(args, "no_cps_expand", False)
        and float(getattr(args, "cps_expand_threshold", 9.0)) > 0
        and not goto_tts
    )

    if _cps_expand_enabled and translated:
        from translation import expand_text_single_ollama as _cps_expand_fn
        _ollama_url_exp   = getattr(args, "adapt_ollama_url",     "http://localhost:11434/api/generate")
        # Po gemma4 preklade: použij --cps_repair_model (default llama3.1:8b, CPU-friendly)
        # Inak (štandardný CPS expand): --adapt_ollama_model (default gemma4 26B cez ollama)
        if _gemma4_did_translate:
            _ollama_model_exp = getattr(args, "cps_repair_model", "llama3.1:8b")
        else:
            _ollama_model_exp = getattr(args, "adapt_ollama_model", "llama3.1:8b")
        # llama3.1:8b načíta rýchlo — kratší timeout; gemma4 26B potrebuje viac
        _default_timeout = 60 if _gemma4_did_translate else 180
        _ollama_to_exp    = max(_default_timeout, int(getattr(args, "adapt_ollama_timeout", _default_timeout)))
        _cps_expanded_n   = 0

        # Connectivity check — skip CPS-EXPAND ak Ollama nebeží
        _ollama_alive = False
        try:
            import requests as _req_warmup
            _chk = _req_warmup.get(
                _ollama_url_exp.replace("/api/generate", "/api/tags").replace("/v1/chat/completions", "/api/tags"),
                timeout=3,
            )
            _ollama_alive = _chk.status_code < 500
        except Exception:
            pass

        if not _ollama_alive:
            print(f"[CPS-EXPAND] Ollama nedostupný ({_ollama_url_exp}) — CPS-EXPAND preskočený", flush=True)
            _cps_expand_enabled = False
        elif not _gemma4_did_translate:
            # Pre-warm: len pre veľké modely (gemma4 26B), nie pre llama3.1:8b
            try:
                print(f"[CPS-EXPAND] Pre-warming ollama: {_ollama_model_exp}...", flush=True)
                _warmup_resp = _req_warmup.post(
                    _ollama_url_exp,
                    json={"model": _ollama_model_exp, "prompt": "Hi", "stream": False},
                    timeout=120,
                )
                if _warmup_resp.status_code == 200:
                    print("[CPS-EXPAND] Ollama warm-up OK", flush=True)
                else:
                    print(f"[CPS-EXPAND] Ollama warm-up status={_warmup_resp.status_code}", flush=True)
            except Exception as _wup_err:
                print(f"[CPS-EXPAND] Ollama warm-up failed: {_wup_err}", flush=True)
        elif _gemma4_did_translate:
            print(f"[CPS-REPAIR] Gemma4 post-repair model: {_ollama_model_exp} (threshold={_cps_expand_threshold})", flush=True)

        for _seg in translated if _cps_expand_enabled else []:
            _slot = float(_seg.get("end", 0)) - float(_seg.get("start", 0))
            if _slot < _cps_expand_min_slot:
                continue
            _txt = (_seg.get("tts_input") or _seg.get("text") or "").strip()
            if not _txt:
                continue
            _cps = len(_txt) / _slot
            if _cps >= _cps_expand_threshold:
                continue

            _src = (_seg.get("text_src") or "").strip()
            print(
                f"[CPS-EXPAND] Seg t={_seg.get('start',0):.1f}s"
                f" cps={_cps:.1f} slot={_slot:.1f}s"
                f" ({len(_txt)}ch) → expand",
                flush=True,
            )
            _expanded = _cps_expand_fn(
                _txt, _src, _slot,
                content_type=content_type,
                ollama_url=_ollama_url_exp,
                ollama_model=_ollama_model_exp,
                timeout=_ollama_to_exp,
            )
            if _expanded and _expanded != _txt and len(_expanded) > len(_txt):
                _seg["tts_input"] = _expanded
                _seg["cps_expand_original"] = _txt
                _seg["cps_expand_new_cps"] = round(len(_expanded) / _slot, 2)
                _cps_expanded_n += 1
                print(
                    f"[CPS-EXPAND]   {len(_txt)}→{len(_expanded)}ch"
                    f" cps={_cps:.1f}→{len(_expanded)/_slot:.1f}",
                    flush=True,
                )

        if _cps_expanded_n:
            print(f"[CPS-EXPAND] Rozšírených {_cps_expanded_n} segmentov", flush=True)

    # ── Bezpodmienečné uvoľnenie ollama modelov pred TTS ────────────────────────
    # Ollama (translator/length-adjuster/cascade) drží 7-20 GB VRAM aj po skončení
    # textu → blokuje TTS engine (F5/Chatterbox/S2). Uvoľníme všetky tri ak boli
    # použité.
    _pre_tts_ollama_url   = getattr(args, "adapt_ollama_url",   "http://localhost:11434/api/generate")
    _pre_tts_unload: list[str] = []
    # 1) length adjuster (compress/expand pass)
    _adapt_model = getattr(args, "adapt_ollama_model", "")
    if _adapt_model:
        _pre_tts_unload.append(_adapt_model)
    # 2a) G3 27B multitask (gemma 3 27B Q4_K_M) — ~16 GB VRAM
    if getattr(args, "use_g3_27b_translator", False):
        _g3_27b_model = getattr(args, "g3_27b_translator_model", "g3-27b-multitask-v2")
        if _g3_27b_model and _g3_27b_model not in _pre_tts_unload:
            _pre_tts_unload.append(_g3_27b_model)
    # 2) G3 v1 translator (gemma 3 12B) — ~10-12 GB VRAM
    if getattr(args, "use_g3_v1_translator", False):
        _g3v1_model = getattr(args, "g3_v1_translator_model", "g3-12b-translator-v1")
        if _g3v1_model:
            _pre_tts_unload.append(_g3v1_model)
        _g3v1_la = getattr(args, "g3_v1_length_adjuster_model", "")
        if _g3v1_la and _g3v1_la not in _pre_tts_unload:
            _pre_tts_unload.append(_g3v1_la)
    # 3) 26B finetuned translator
    if getattr(args, "use_finetuned_translator", False):
        _ft_model = getattr(args, "finetuned_translator_model", "")
        if _ft_model and _ft_model not in _pre_tts_unload:
            _pre_tts_unload.append(_ft_model)
    # 4) Cascade fallback model — 26b-translator-q6 vyložený cez llm_v1._cascade_to_26b
    if "26b-translator-q6:latest" not in _pre_tts_unload:
        _pre_tts_unload.append("26b-translator-q6:latest")
    if _pre_tts_unload and _pre_tts_ollama_url:
        for _m in _pre_tts_unload:
            print(f"[PRE-TTS] Uvoľňujem ollama: {_m}", flush=True)
            _unload_ollama_model(_pre_tts_ollama_url, _m, timeout=20)
        import gc as _gc_pretts
        _gc_pretts.collect()
        import torch as _torch_pretts
        _torch_pretts.cuda.empty_cache()
        _log_memory("po uvoľnení ollama pred TTS")

    # --- TTS engine selection (needed before voice sample) ---
    use_torch_compile = not getattr(args, "no_torch_compile", False)
    wav_paths: list[Path] = []
    n = len(translated)
    use_s2 = getattr(args, "use_s2_pro", False)
    use_turbo = getattr(args, "use_chatterbox_turbo", False)
    use_f5 = getattr(args, "use_f5", False)
    use_omnivoice = getattr(args, "use_omnivoice", False)
    use_cb = (
        args.use_chatterbox
        or (not use_s2 and not use_f5 and not use_omnivoice and not args.use_xtts and not args.use_piper and not args.use_piper_rs)
    ) and not use_turbo and not use_s2 and not use_f5 and not use_omnivoice

    # --- Voice sample for cloning ---
    voice_sample: str | None = None
    s2_reference: tuple[str, str] | None = None
    # --turbo_reference_voice overrides the default SK fallback for Turbo TTS
    _turbo_ref = getattr(args, "turbo_reference_voice", "") or ""
    if use_turbo and _turbo_ref and Path(_turbo_ref).exists():
        voice_sample = _turbo_ref
        print(f"[TURBO] Using custom SK reference: {_turbo_ref}", flush=True)
    _explicit_voice = getattr(args, "xtts_speaker_wav", "") or ""
    _has_explicit_voice = bool(_explicit_voice and Path(_explicit_voice).exists())
    if _explicit_voice and not Path(_explicit_voice).exists():
        print(f"[VOICE] WARNING: Zadaný voice WAV neexistuje: {_explicit_voice}", flush=True)
    if not voice_sample and _has_explicit_voice:
        # User provided explicit voice reference — use it directly (highest priority)
        voice_sample = _explicit_voice
        print(f"[VOICE] Using provided voice reference: {voice_sample}", flush=True)
    elif use_s2 and args.clone_voice:
        s2_ref_path = work_dir / "s2_reference.wav"
        s2_reference = prepare_s2_pro_reference(
            audio_path=audio_wav,
            output_path=s2_ref_path,
            segments=segments,
            auto=False,
            start_sec=getattr(args, "voice_sample_start", 0.0),
            duration_sec=getattr(args, "voice_sample_duration", 10.0),
        )
        if s2_reference:
            print(f"[S2-PRO] Manual reference prepared: {s2_ref_path}", flush=True)
    elif not voice_sample and args.clone_voice:
        # Explicit: extract fixed time range specified by user
        _voice_src = vocals_wav if (vocals_wav and vocals_wav.exists()) else audio_wav
        vs_path = work_dir / "voice_sample.wav"
        ok = extract_chatterbox_voice_sample(
            _voice_src, vs_path,
            start_sec=args.voice_sample_start,
            duration_sec=args.voice_sample_duration,
        )
        if ok:
            voice_sample = str(vs_path)
            print(f"[VOICE] Extracted sample: {vs_path}", flush=True)
    elif not voice_sample and use_cb and not getattr(args, "no_voice_clone", False):
        # Auto: build a denser clone prompt from the best source speech segments.
        _voice_src = vocals_wav if (vocals_wav and vocals_wav.exists()) else audio_wav
        vs_path = work_dir / "voice_sample.wav"
        ok = extract_best_chatterbox_reference_from_segments(
            _voice_src,
            vs_path,
            segments=segments,
            target_dur=min(6.0, max(4.5, float(getattr(args, "voice_sample_duration", 10.0)) - 1.5)),
            max_total_dur=min(8.0, max(5.5, float(getattr(args, "voice_sample_duration", 10.0)))),
        )
        if not ok:
            ok = extract_best_chatterbox_voice_sample(
                _voice_src,
                vs_path,
                sample_dur=getattr(args, "voice_sample_duration", 10.0),
                step=5.0,
                min_start=0.0,
            )
        if ok:
            print(f"[VOICE] Chatterbox reference source: {_voice_src.name}", flush=True)
        if ok:
            voice_sample = str(vs_path)
            print(f"[VOICE] Auto-extracted best sample: {vs_path}", flush=True)
        else:
            print("[VOICE] Auto-extraction failed — using default Chatterbox voice", flush=True)
    elif use_s2 and not getattr(args, "no_voice_clone", False):
        s2_ref_path = work_dir / "s2_reference.wav"
        s2_reference = prepare_s2_pro_reference(
            audio_path=audio_wav,
            output_path=s2_ref_path,
            segments=segments,
            auto=True,
            duration_sec=getattr(args, "voice_sample_duration", 10.0),
        )
        if s2_reference:
            print(f"[S2-PRO] Auto reference prepared: {s2_ref_path}", flush=True)
        else:
            print("[S2-PRO] Auto reference unavailable — using default voice", flush=True)

    if use_s2:
        print(f"[S2-PRO] Using Fish Speech S2-Pro with checkpoint: {args.s2_pro_model}", flush=True)
    elif use_turbo:
        print(f"[TURBO] Using Chatterbox TURBO TTS with model: {args.chatterbox_turbo_model}", flush=True)
    elif use_f5:
        print(f"[F5] Using F5-TTS SK fine-tuned: {getattr(args, 'f5_model', '')}", flush=True)
    elif use_omnivoice:
        print(f"[OMNIVOICE] Using OmniVoice (zero-shot): {getattr(args, 'omnivoice_repo', '')}", flush=True)
    elif use_cb:
        print(f"[CHATTERBOX] Using Chatterbox TTS with model: {args.chatterbox_model}", flush=True)
    elif args.use_xtts:
        print("[XTTS] Using XTTS v2 TTS", flush=True)
    elif args.use_piper:
        print(f"[PIPER] Using Piper TTS: {args.piper_model}", flush=True)

    # --- Merge pre-computed diarization (Level 0b) if available ---
    _diar_json = getattr(args, "diarization_json", "") or ""
    if not _diar_json:
        _auto = Path(input_path).parent / "output" / f"{Path(input_path).stem}_level0b.json"
        if not _auto.exists():
            _auto = Path(str(BASE_DIR) if "BASE_DIR" in dir() else Path(input_path).parent) / "output" / f"{Path(input_path).stem}_level0b.json"
        if _auto.exists():
            _diar_json = str(_auto)
            print(f"[DIAR] Auto-detected diarization JSON: {_auto.name}", flush=True)
    if _diar_json:
        translated = _merge_diarization_json(translated, _diar_json)

    # --- Multi-voice speaker clustering (before TTS loop) ---
    speaker_refs: list[str | None] = [None] * len(translated)
    speaker_ids: list[int] = [-1] * len(translated)
    speaker_meta: dict[int, dict[str, object]] = {}
    # For Chatterbox: cluster → wav sample path
    if use_cb and getattr(args, "multi_voice", False):
        n_spk = getattr(args, "multi_voice_n", 2)
        if _has_diarization(translated):
            print("[MULTI] Pre-computed diarization found in segments — skipping re-clustering.", flush=True)
            speaker_refs, speaker_ids, speaker_meta = _diarization_from_segments(translated)
        else:
            print(f"[MULTI] Speaker clustering: detecting {n_spk} voices...", flush=True)
            _cluster_src = vocals_wav if (vocals_wav and vocals_wav.exists()) else audio_wav
            if _cluster_src != audio_wav:
                print(f"[MULTI] Using clean vocals for clustering: {_cluster_src.name}", flush=True)
            speaker_refs, speaker_ids, speaker_meta = _cluster_speaker_voices(_cluster_src, translated, n_spk, chunks_dir)

        _male_voice_wav = (getattr(args, "multi_voice_male_wav", "") or "").strip()
        _female_voice_wav = (getattr(args, "multi_voice_female_wav", "") or "").strip()
        for _label, _path in (("male", _male_voice_wav), ("female", _female_voice_wav)):
            if _path and not Path(_path).exists():
                print(f"[MULTI] WARNING: {_label} fallback WAV not found: {_path}", flush=True)
        _gender_voice_map = {
            "male": _male_voice_wav if _male_voice_wav and Path(_male_voice_wav).exists() else "",
            "female": _female_voice_wav if _female_voice_wav and Path(_female_voice_wav).exists() else "",
        }
        _gender_voice_routed = 0
        if any(_gender_voice_map.values()):
            for i, sid in enumerate(speaker_ids):
                if sid < 0:
                    continue
                _gender = str((speaker_meta.get(sid) or {}).get("gender") or "")
                _gender_ref = _gender_voice_map.get(_gender) or ""
                if _gender_ref:
                    speaker_refs[i] = _gender_ref
                    _gender_voice_routed += 1
            if _gender_voice_routed:
                print(
                    f"[MULTI] Gender fallback WAV routing applied to {_gender_voice_routed} segments "
                    f"(male={bool(_gender_voice_map['male'])}, female={bool(_gender_voice_map['female'])})",
                    flush=True,
                )

        for i, sid in enumerate(speaker_ids):
            if sid < 0:
                continue
            _meta = speaker_meta.get(sid) or {}
            translated[i]["speaker_gender_est"] = _meta.get("gender", "unknown")
            translated[i]["speaker_pitch_hz"] = _meta.get("pitch_hz")

        _female_indices = [
            i for i, sid in enumerate(speaker_ids)
            if sid >= 0 and (speaker_meta.get(sid) or {}).get("gender") == "female"
        ]
        if _female_indices and getattr(args, "tgt_lang", "") in ("sk", "slk"):
            _fixed_female = apply_gender_fix_sk_feminine([translated[i] for i in _female_indices])
            for _idx, _fixed_seg in zip(_female_indices, _fixed_female):
                _fixed_seg["speaker_gender_est"] = translated[_idx].get("speaker_gender_est", "female")
                _fixed_seg["speaker_pitch_hz"] = translated[_idx].get("speaker_pitch_hz")
                translated[_idx] = _fixed_seg
            print(
                f"[MULTI] Applied feminine grammar fix to {len(_female_indices)} female-speaker segments",
                flush=True,
            )

        # Save speaker diarization log
        import json as _json
        spk_log = [
            {
                "seg": i,
                "speaker": f"Speaker {sid}" if sid >= 0 else "unknown",
                "speaker_id": sid,
                "speaker_gender_est": (speaker_meta.get(sid) or {}).get("gender", "unknown"),
                "speaker_pitch_hz": (
                    round(float((speaker_meta.get(sid) or {}).get("pitch_hz")), 1)
                    if isinstance((speaker_meta.get(sid) or {}).get("pitch_hz"), (int, float))
                    else None
                ),
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": seg.get("text", "").strip(),
            }
            for i, (seg, sid) in enumerate(zip(translated, speaker_ids))
        ]
        spk_log_path = out_dir / f"{stem}_speakers.json"
        spk_log_path.write_text(_json.dumps(spk_log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[MULTI] Speaker log saved → {spk_log_path}", flush=True)
    # --- TTS pre-flight setup ---
    import re as _re
    import json as _json

    # ── A. Anti-meta filter ──────────────────────────────────────────────────
    # LLM sometimes returns task description / commentary instead of translation.
    # Caught here as last resort before audio is generated.
    _META_PREFIXES = (
        "tu je skrátená", "tu je opravená", "tu je preklad", "tu je verzia",
        "skrátená verzia", "opravená verzia", "opravený preklad",
        "prepis:", "vysvetlenie:", "odpoveď:", "preklad:",
        "rozumiem,", "rozumiem.", "jasne,", "jasne.", "jasné,", "jasné.",
        "samozrejme,", "samozrejme.", "samozřejmě,", "samozřejmě.",
        "rozumím,", "rozumím.", "iste,", "iste.",
        "tu je moja odpoveď", "tu je výsledok",
        "nasledujúci text", "nasledujúci preklad",
        "toto je preklad", "toto je výsledok", "toto je opravená",
        "nasleduje preklad", "nasleduje opravený",
        "poznámka:", "note:", "pozn.:", "remark:",
        # Gemma4 refusal / meta-commentary patterns
        "toto je technický text", "tento text je v rozpore", "tento text je v rozporu",
        "preklad nebol poskytnutý", "(preklad nebol", "preklad nie je možný",
        "nie je možné preložiť", "text neobsahuje", "text je v rozpore",
        "nie je reálny", "nie reálny text", "prosím, poskytnite",
        "neobsahuje žiadny", "neobsahuje žiadnu",
        # Whisper hallucination patterns (common false transcriptions)
        "subtituly urobil", "titulky urobil", "subtitles by", "translated by",
        "subscribe", "like and subscribe",
    )
    _META_ANYWHERE = (
        "halucin", "znak halucin", "nesprávny preklad", "incorrect translation",
        "skrátená verzia:", "tu je skrátený", "pre dabing by", "lepší preklad",
    )

    def _is_meta_text(t: str) -> bool:
        """Vráti True ak text vyzerá ako LLM meta-odpoveď, nie preklad."""
        tl = t.lower().strip()
        if any(tl.startswith(p) for p in _META_PREFIXES):
            return True
        if any(p in tl for p in _META_ANYWHERE):
            return True
        # Prvý riadok < 40 znakov a končí ":" = typické "Prepis: ..." artefakty
        first_line = tl.split("\n")[0]
        if len(first_line) < 40 and first_line.endswith(":"):
            return True
        return False

    # Debug log: accumulated per-segment TTS input info → saved as JSON after loop
    _tts_debug_log: list[dict] = []
    _timing_audit_log: list[dict] = []
    # Hallucination bucket: segments where QA flagged bad audio
    _bad_seg_log: list[dict] = []
    _timing_retry_enabled = bool(getattr(args, "tts_timing_retry", True))
    _timing_retry_max_attempts = max(0, int(getattr(args, "tts_timing_retry_max_attempts", 1) or 0))
    _turbo_fallback_enabled = bool(use_turbo and getattr(args, "turbo_auto_fallback", False))
    _turbo_fallback_model = (getattr(args, "turbo_fallback_model", "") or args.chatterbox_model).strip()
    _turbo_fallback_device_pref = getattr(args, "turbo_fallback_device", "cpu") or "cpu"
    _turbo_fallback_preroute = 0
    _turbo_fallback_qa = 0

    def _turbo_fallback_device() -> str:
        return device if _turbo_fallback_device_pref == "auto" else _turbo_fallback_device_pref

    # --- Emotion pass: per-segment exaggeration (voliteľné, --emotion_llm) ---
    # Primárna metóda: Prosody classifier (RMS + F0 std, librosa, bez modelu)
    # Fallback: LLM (Ollama) ak prosody zlyhá
    _emotion_levels: list[str] = ["low"] * len(translated)
    _emotion_llm_enabled = use_cb and getattr(args, "emotion_llm", False)
    if _emotion_llm_enabled:
        from translation import _EMOTION_EXAG
        print(f"[EMOTION] Detekujem intenzitu prosody pre {len(translated)} segmentov...", flush=True)
        try:
            from audio_pipeline import classify_prosody_levels
            _emotion_levels = classify_prosody_levels(translated, input_path)
            _em_ok = any(lv != "low" for lv in _emotion_levels)
            if not _em_ok:
                raise ValueError("Všetky segmenty sú 'low' — prosody classifier pravdepodobne zlyhal")
        except Exception as _prosody_err:
            print(f"[EMOTION] Prosody zlyhal ({_prosody_err}), skúšam LLM fallback...", flush=True)
            try:
                from translation import detect_emotion_levels_ollama
                _em_url   = getattr(args, "adapt_ollama_url",  "http://localhost:11434/api/generate")
                _em_model = getattr(args, "emotion_llm_model", "gemma4:26b")
                print(f"[EMOTION] LLM fallback: {_em_model}", flush=True)
                _emotion_levels = detect_emotion_levels_ollama(
                    translated,
                    ollama_url=_em_url,
                    ollama_model=_em_model,
                    timeout=90,
                )
            except Exception as _llm_err:
                print(f"[EMOTION] LLM fallback zlyhal ({_llm_err}), emócia vypnutá", flush=True)
                _emotion_llm_enabled = False
        _em_counts = {k: _emotion_levels.count(k) for k in ("low", "medium", "high")}
        print(f"[EMOTION] low={_em_counts['low']} medium={_em_counts['medium']} high={_em_counts['high']}", flush=True)

    for i, seg in enumerate(translated):
        _step: dict = {"seg": i + 1,
                       "start": float(seg.get("start", 0)),
                       "end": float(seg.get("end", 0))}
        _slot_dur = float(seg["end"]) - float(seg["start"])
        _step["slot"] = round(_slot_dur, 3)

        text = seg.get("text", "").strip()
        _turbo_route_probe_text = text
        _step["translated"] = text
        # Adaptation metadata (set by adapt_segments_batch before TTS loop)
        if seg.get("adapt_mode") and seg["adapt_mode"] != "ok_no_change":
            _step["adapted"] = seg.get("text_original", text)
            _step["adapt_mode"] = seg["adapt_mode"]
            _step["adapt_ratio_before"] = seg.get("adapt_ratio_before", 0)
            _step["adapt_ratio_after"] = seg.get("adapt_ratio", 0)
            _step["adapt_style"] = seg.get("adapt_style")
            _step["adapt_rewrite_mode"] = seg.get("adapt_rewrite_mode")
            _step["adapt_cps_before"] = seg.get("adapt_cps_before", 0)
            _step["adapt_cps_after"] = seg.get("adapt_cps_after", 0)
            _step["adapt_target_cps"] = seg.get("adapt_target_cps", 0)

        # Strip any unresolved phonetic placeholders (MADLAD T5 tokenizer corrupts __PHON_N__
        # into "__ PHON _0__", "__phone_4__", etc. — clean before TTS to avoid garbage speech)
        text = _re.sub(r'_+[\s_]*[Pp][Hh][Oo][Nn][Ee]?[\s_]*_?[\s_]*\d+[\s_]*_+', '', text).strip()
        # Final phonetic pass + guillemet cleanup right before TTS (catches all translation paths)
        text = strip_guillemets(text)
        # CamelCase SQL function names → TTS-readable (always, even when phonetic_respelling=OFF)
        # isNull→"is nul", COALESCE→"koalesk", etc.
        text = normalize_camelcase_for_tts(text)

        # ── C. Safe mode for short slots ─────────────────────────────────────
        # Ak je slot < 1.5s a text technický/SQL → vypni agresívne normalizácie
        _norm_lang = "sk" if tgt in ("slk", "sk") else "cs"
        _short_slot = _slot_dur < 1.5
        _chars_per_sec = len(text) / _slot_dur if _slot_dur > 0.05 else 999

        # NeMo-style normalization: numbers → words, dates, times, units, percentages
        # mode="light" (production_dub): only dates/times/percentages/cardinals — no unit/ordinal rewrites
        # Optimization: skip entirely when segment has no digits or special tokens (saves regex overhead
        # and avoids any accidental text mutation on pure natural-language sentences)
        _has_normalizable = bool(_re.search(r'\d|%|€|\$|Kč', text))
        if _has_normalizable:
            text = normalize_for_tts(text, content_type=content_type, lang=_norm_lang,
                                     mode=_normalizer_mode)
        _step["after_normalize"] = text

        # Fish Speech S2 Pro číta SQL natívne — SQL TTS konverzie (IS NULL→ajznul) sú zbytočné
        _skip_sql_ph = use_s2

        _forced_acronym_phonetics = False
        if tgt in ("slk", "sk") and not _phonetic_on and _is_acronym_heavy_tts_text(text):
            text = _respell_sk_acronym_heavy_tts_text(text)
            _forced_acronym_phonetics = True
        elif tgt in ("cs", "ces") and not _phonetic_on and _is_acronym_heavy_tts_text(text):
            text = apply_phonetic_respelling(text, skip_sql_phonetic=_skip_sql_ph)
            _forced_acronym_phonetics = True
        elif _phonetic_on:
            text = apply_phonetic_respelling(text, skip_sql_phonetic=_skip_sql_ph)
        _step["after_phonetic"] = text
        if _forced_acronym_phonetics:
            _step["forced_acronym_phonetics"] = True

        if _mixed_technical_terms and content_type == "technical":
            _before_linux_mixed = text
            text = apply_linux_mixed_mode(text, source_text=(seg.get("text_src") or ""), fallback_text=_before_linux_mixed)
            if use_cb or use_turbo:
                text = normalize_linux_for_tts(text)
            if text != _before_linux_mixed:
                _step["after_linux_mixed"] = text

        # Re-apply deterministic SK post-fixes after text adaptation.
        # This catches phrase regressions that can be reintroduced after audit,
        # e.g. "Určite sa pýtate", "časový kód", or "Prize Repository".
        if tgt in ("slk", "sk"):
            text, _postfix_hits = apply_postfix(text)
            if _postfix_hits:
                _step["after_postfix"] = text
                _step["postfix_hits"] = _postfix_hits

        # SK TTS mispronunciation fixes — always on for SK, independent of phonetic_on
        # Handles cases where the fine-tuned SK model consistently mispronounces specific forms
        if tgt in ("slk", "sk"):
            _sk_fixes = [
                # "time code" preložený ako "časový kód" → správne "časová značka"
                (r'\bčasový kód\b',    'časová značka'),
                (r'\bčasového kódu\b', 'časovej značky'),
                (r'\bčasovému kódu\b', 'časovej značke'),
                (r'\bčasovom kóde\b',  'časovej značke'),
                (r'\bčasovým kódom\b', 'časovou značkou'),
                # "-iesť" verb infinitives: model reads as "-ás" (previesť→prevás)
                (r'\bpreviesť\b', 'previjest'),
                (r'\bPreviesť\b', 'Previjest'),
                (r'\bpriviesť\b', 'privijest'),
                (r'\bPriviesť\b', 'Privijest'),
                (r'\badviesť\b',  'advijest'),
                (r'\bAdviesť\b',  'Advijest'),
                (r'\bodriesť\b',  'odrijest'),
                (r'\bOdriesť\b',  'Odrijest'),
                (r'\bzaviesť\b',  'zavijest'),
                (r'\bZaviesť\b',  'Zavijest'),
                (r'\bvyviesť\b',  'vyvijest'),
                (r'\bVyviesť\b',  'Vyvijest'),
                (r'\botviesť\b',  'otvijest'),
                (r'\bOtviesť\b',  'Otvijest'),
            ]
            for _pat, _rep in _sk_fixes:
                text = _re.sub(_pat, _rep, text)
        _step["after_sk_fixes"] = text if tgt in ("slk", "sk") else None

        # ── D. Post-phonetic length guard ────────────────────────────────────
        # Phonetic respelling môže predĺžiť text (napr. "API" → "éj-pí-áj").
        # Ak chars_per_sec stále prekračuje limit → generické filler-removal.
        _cps_after_phonetic = len(text) / _slot_dur if _slot_dur > 0.05 else 0
        _adapt_ratio_final  = float(seg.get("adapt_ratio", seg.get("adapt_ratio_after", 1.0)) or 1.0)
        _cps_guard_limit = float(seg.get("adapt_hard_cps_limit", 18.5) or 18.5)
        _step["cps_guard_limit"] = round(_cps_guard_limit, 2)
        if needs_rewrite(_adapt_ratio_final, _cps_after_phonetic, max_chars_per_sec=_cps_guard_limit):
            _text_before_guard = text
            text = shorten_text(text)
            if text != _text_before_guard:
                _step["after_phonetic_guard"] = text
                _step["phonetic_guard_triggered"] = True

        # SQL normalization: only when explicitly enabled (_sql_tts_on)
        # production_dub: OFF; teaching_sql / content_type=sql: ON
        sql_mode = _sql_tts_on and (not _short_slot)
        _preserve_sql_terms = bool(seg.get("adapt_preserve_sql_terms", False))
        if sql_mode and not (use_s2 or _preserve_sql_terms):
            if use_cb or use_turbo:
                text = normalize_sql_for_chatterbox_tts(text)
                _step["after_sql_mode"] = "chatterbox_selective"
            else:
                text = normalize_sql_for_tts(text)
                _step["after_sql_mode"] = "generic"
            _step["after_sql"] = text
        elif sql_mode:
            _step["after_sql"] = text
            _step["sql_terms_preserved"] = True

        out_chunk = chunks_dir / f"tts_{i+1:04d}.wav"

        # ── A. Anti-meta filter ───────────────────────────────────────────────
        if text and _is_meta_text(text):
            print(f"[TTS] Seg {i+1} META-TEXT detected — replacing with silence: {text[:80]!r}", flush=True)
            _bad_seg_log.append({"seg": i+1, "reason": "meta_text", "slot": _slot_dur,
                                  "translated": _step.get("translated"), "tts_input": text})
            _step["tts_input"] = text
            _step["dropped"] = "meta_text"
            _tts_debug_log.append(_step)
            write_silence_chunk(out_chunk, _slot_dur)
            wav_paths.append(out_chunk)
            continue

        if not text:
            write_silence_chunk(out_chunk, _slot_dur)
            wav_paths.append(out_chunk)
            continue

        # Ensure text ends with punctuation to prevent early EOS and to keep
        # the cache hash tied to the real rendered text.
        if (use_cb or use_turbo or use_s2) and text and text[-1] not in ".!?…,:;":
            text = text + "."
        _short_reply_core = text.rstrip(".!?…,:;").strip()
        _short_reply_like = (
            bool(_short_reply_core)
            and len(_short_reply_core.split()) <= 2
            and len(_short_reply_core) <= 10
            and _slot_dur <= 1.8
            and not sql_mode
        )

        # Short-slot warning
        if _short_slot and _chars_per_sec > 15:
            print(f"[TTS] Seg {i+1} SHORT SLOT: {_slot_dur:.2f}s / {len(text)} chars ({_chars_per_sec:.0f} c/s)", flush=True)

        # ── B. Debug log entry ────────────────────────────────────────────────
        _step["chars_per_sec"] = round(_chars_per_sec, 1)
        _step["tts_input"] = text
        _speaker_ref = speaker_refs[i] if i < len(speaker_refs) else ""
        _tts_cache_meta_path = out_chunk.with_suffix(".meta.json")

        # --- Urči finálne TTS parametre pre cache kľúč ---
        _cb_exaggeration = getattr(args, "chatterbox_exaggeration", -1.0)
        _cb_cfg_weight = getattr(args, "chatterbox_cfg_weight", -1.0)
        _cb_temperature = getattr(args, "chatterbox_temperature", -1.0)

        if use_cb or (use_turbo and getattr(args, "turbo_auto_fallback", False)):
            _temp_style_overrides = None
            if _emotion_llm_enabled:
                from translation import _EMOTION_EXAG
                _em_level = _emotion_levels[i] if i < len(_emotion_levels) else "low"
                _em_exag = _EMOTION_EXAG.get(_em_level, 0.50)
                if _cb_exaggeration < 0:
                    _cb_exaggeration = _em_exag
                else:
                    _cb_exaggeration = round(0.60 * _cb_exaggeration + 0.40 * _em_exag, 3)

            _seg_emotion = seg.get("emotion")
            if _seg_emotion and isinstance(_seg_emotion, dict) and _seg_emotion.get("confidence", 0) > 0.15:
                _base_e = _cb_exaggeration if _cb_exaggeration >= 0 else float(getattr(args, "chatterbox_exaggeration", 0.50))
                _base_c = _cb_cfg_weight   if _cb_cfg_weight   >= 0 else float(getattr(args, "chatterbox_cfg_weight",   0.60))
                _base_t = _cb_temperature  if _cb_temperature  >= 0 else float(getattr(args, "chatterbox_temperature",  0.78))
                _cb_exaggeration, _cb_cfg_weight, _cb_temperature = _emotion_vec_to_cb_params(
                    _seg_emotion, _base_e, _base_c, _base_t,
                )

            if getattr(args, "emotion_clone", False) and _has_explicit_voice:
                _temp_style_overrides = _derive_style_transfer_params(
                    src_wav=audio_wav, start=float(seg["start"]), end=float(seg["end"]),
                    scratch_dir=chunks_dir, seg_idx=i + 1,
                    base_exaggeration=_cb_exaggeration, base_cfg_weight=_cb_cfg_weight,
                    base_temperature=_cb_temperature, pad_s=getattr(args, "emotion_clone_pad_s", 0.3),
                )
                _temp_style_overrides = _guard_style_transfer_params(
                    _temp_style_overrides, tgt_lang=getattr(args, "tgt_lang", ""), has_explicit_voice=True,
                )

            if _temp_style_overrides:
                _cb_exaggeration = _temp_style_overrides.get("exaggeration", _cb_exaggeration)
                _cb_cfg_weight = _temp_style_overrides.get("cfg_weight", _cb_cfg_weight)
                _cb_temperature = _temp_style_overrides.get("temperature", _cb_temperature)

        _tts_cache_payload = {
            "version": 2,
            "seg": i + 1,
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "slot": round(_slot_dur, 3),
            "input_text": text,
            "engine": (
                "s2_pro" if use_s2 else
                "chatterbox_turbo" if use_turbo else
                "chatterbox" if use_cb else
                "xtts" if args.use_xtts else
                "piper" if args.use_piper else
                "unknown"
            ),
            "voice_sample": str(voice_sample or ""),
            "speaker_ref": str(_speaker_ref or ""),
            "speaker_id": int(speaker_ids[i]) if i < len(speaker_ids) else -1,
            "speaker_gender_est": str(seg.get("speaker_gender_est", "") or ""),
            "speaker_pitch_hz": (
                round(float(seg.get("speaker_pitch_hz")), 1)
                if isinstance(seg.get("speaker_pitch_hz"), (int, float))
                else None
            ),
            "s2_reference_audio": str(s2_reference[0] if s2_reference else ""),
            "s2_reference_text": str(s2_reference[1] if s2_reference else ""),
            "sql_mode": bool(sql_mode),
            "normalizer_mode": _normalizer_mode,
            "phonetic_on": bool(_phonetic_on),
            "multi_voice": bool(getattr(args, "multi_voice", False)),
            "multi_voice_male_wav": str(getattr(args, "multi_voice_male_wav", "") or ""),
            "multi_voice_female_wav": str(getattr(args, "multi_voice_female_wav", "") or ""),
            "emotion_clone": bool(getattr(args, "emotion_clone", False)),
            "voice_eq": bool(getattr(args, "voice_eq", False)),
            "chatterbox_model": str(getattr(args, "chatterbox_model", "") or ""),
            "chatterbox_turbo_model": str(getattr(args, "chatterbox_turbo_model", "") or ""),
            "s2_pro_model": str(getattr(args, "s2_pro_model", "") or ""),
            "xtts_speaker_wav": str(getattr(args, "xtts_speaker_wav", "") or ""),
            "xtts_speaker": str(getattr(args, "xtts_speaker", "") or ""),
            "piper_model": str(getattr(args, "piper_model", "") or ""),
            "chatterbox_exaggeration": _cb_exaggeration,
            "chatterbox_cfg_weight": _cb_cfg_weight,
            "chatterbox_temperature": _cb_temperature,
        }
        _tts_cache_hash = _hash_tts_cache_payload(_tts_cache_payload)
        _step["tts_cache_hash"] = _tts_cache_hash[:12]
        _tts_debug_log.append(_step)

        # --reuse_tts: reuse only when the semantic cache hash matches.
        if getattr(args, "reuse_tts", False) and out_chunk.exists() and out_chunk.stat().st_size > 200:
            _reuse_ok = False
            _reuse_reason = "missing_meta"
            if _tts_cache_meta_path.exists():
                try:
                    _cached_meta = json.loads(_tts_cache_meta_path.read_text(encoding="utf-8"))
                    if _cached_meta.get("semantic_hash") == _tts_cache_hash:
                        _reuse_ok = True
                        _reuse_reason = "hash_ok"
                    else:
                        _reuse_reason = "hash_mismatch"
                except Exception as _reuse_err:
                    _reuse_reason = f"meta_error:{_reuse_err}"
            if _reuse_ok:
                print(f"[TTS] Segment {i+1}/{n} — reusing existing chunk (hash ok)", flush=True)
                wav_paths.append(out_chunk)
                continue
            print(f"[TTS] Segment {i+1}/{n} — regenerating ({_reuse_reason})", flush=True)

        print(f"[TTS] Segment {i+1}/{n}", flush=True)
        _sql_clause_split = False  # set True below when SQL clause-split is used
        _used_cb_for_segment = False
        audio_prompt = None
        ref_audio_path = s2_reference[0] if s2_reference else None
        ref_text = s2_reference[1] if s2_reference else None
        _turbo_route_reasons: list[str] = []
        if _turbo_fallback_enabled:
            _turbo_route_reasons = _turbo_fallback_reasons(
                _turbo_route_probe_text,
                _slot_dur,
                sql_mode=sql_mode,
            )
            if _turbo_route_reasons:
                _step["turbo_fallback_route"] = _turbo_route_reasons
                _step["turbo_fallback_device"] = _turbo_fallback_device()
                _turbo_fallback_preroute += 1
                print(
                    f"[TURBO->MTL] Seg {i+1}: pre-route {', '.join(_turbo_route_reasons)} "
                    f"-> {_turbo_fallback_model} on {_turbo_fallback_device()}",
                    flush=True,
                )

        if use_turbo and not _turbo_route_reasons:
            _step["tts_engine"] = "turbo"
            audio_prompt = voice_sample
            if speaker_refs[i]:
                audio_prompt = speaker_refs[i]
            chatterbox_turbo_tts(
                text=text,
                output_path=out_chunk,
                model_path=args.chatterbox_turbo_model,
                device=device,
                audio_prompt_path=audio_prompt,
            )

        elif use_s2:
            _step["tts_engine"] = "s2_pro_clone" if s2_reference else "s2_pro"
            s2_pro_tts(
                text=text,
                output_path=out_chunk,
                checkpoint_path=args.s2_pro_model,
                reference_audio_path=ref_audio_path,
                reference_text=ref_text,
                server_python=getattr(args, "s2_pro_server_python", ""),
                server_port=int(getattr(args, "s2_pro_server_port", 8091)),
            )

        elif use_cb or _turbo_route_reasons:
            _used_cb_for_segment = True
            _step["tts_engine"] = "chatterbox_fallback" if _turbo_route_reasons else "chatterbox"
            _style_overrides = None
            _base_exag = getattr(args, "chatterbox_exaggeration", -1.0)
            _base_cfg = getattr(args, "chatterbox_cfg_weight", -1.0)
            _base_temp = getattr(args, "chatterbox_temperature", -1.0)
            # LLM emotion override — upraví exaggeration podľa intenzity segmentu
            if _emotion_llm_enabled and not _style_overrides:
                from translation import _EMOTION_EXAG
                _em_level = _emotion_levels[i] if i < len(_emotion_levels) else "low"
                _em_exag = _EMOTION_EXAG.get(_em_level, 0.50)
                # Použi len ak user nenastavil exaggeration manuálne (default -1.0)
                if _base_exag < 0:
                    _base_exag = _em_exag
                else:
                    # Blend: 60% user nastavenie + 40% emotion
                    _base_exag = round(0.60 * _base_exag + 0.40 * _em_exag, 3)
                print(f"[EMOTION] Seg {i+1}: {_em_level} → exaggeration={_base_exag:.3f}", flush=True)
            # Priority: multi_voice > explicit custom voice > emotion_clone > clone_voice > None
            if speaker_refs[i]:
                audio_prompt = speaker_refs[i]
                _spk_id = speaker_ids[i] if i < len(speaker_ids) else -1
                print(f"[MULTI] Seg {i+1}: Speaker {_spk_id} ref={Path(audio_prompt).name}", flush=True)
            elif _has_explicit_voice:
                audio_prompt = voice_sample
                if getattr(args, "emotion_clone", False):
                    _style_overrides = _derive_style_transfer_params(
                        src_wav=audio_wav,
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        scratch_dir=chunks_dir,
                        seg_idx=i + 1,
                        base_exaggeration=_base_exag,
                        base_cfg_weight=_base_cfg,
                        base_temperature=_base_temp,
                        pad_s=getattr(args, "emotion_clone_pad_s", 0.3),
                    )
                    _style_overrides = _guard_style_transfer_params(
                        _style_overrides,
                        tgt_lang=getattr(args, "tgt_lang", ""),
                        has_explicit_voice=True,
                    )
                    if _style_overrides:
                        _step["emotion_style_transfer"] = dict(_style_overrides)
                        _pitch_info = (
                            f" pitch_range={_style_overrides['pitch_range_hz']:.0f}Hz"
                            f" pitch_score={_style_overrides['pitch_score']:.2f}"
                            if "pitch_range_hz" in _style_overrides else ""
                        )
                        print(
                            f"[EMOTION-STYLE] Segment {i+1}: custom voice + style transfer "
                            f"exag={_style_overrides['exaggeration']:.2f} "
                            f"cfg={_style_overrides['cfg_weight']:.2f} "
                            f"temp={_style_overrides['temperature']:.2f}"
                            f"{_pitch_info}",
                            flush=True,
                        )
            elif getattr(args, "emotion_clone", False) and not _has_explicit_voice:
                # For SK dubbing, full per-segment prompts from the original (often EN)
                # audio can bend pronunciation or lexicalize foreign prosody into the
                # generated Slovak. Prefer a stable source-voice sample plus style-only
                # transfer, and keep the old per-segment ref path only as fallback.
                _prefer_stable_source_style = bool(voice_sample and getattr(args, "tgt_lang", "") == "sk")
                if _prefer_stable_source_style:
                    audio_prompt = voice_sample
                    # Cross-lingual EN->SK dubbing is much more stable when we keep
                    # the source voice identity but *avoid* per-segment style
                    # modulation from the foreign-language audio. Otherwise Slovak
                    # pronunciation often bends toward the source prosody.
                    _style_overrides = {
                        "exaggeration": round(min(_base_exag, 0.50), 3),
                        "cfg_weight": round(_clamp(_base_cfg, 0.45, 0.55), 3),
                        "temperature": round(_clamp(_base_temp, 0.42, 0.50), 3),
                    }
                    _step["emotion_mode"] = "stable_source_voice"
                    print(
                        f"[EMOTION] Segment {i+1}: source voice sample only "
                        f"(style transfer disabled for SK stability) "
                        f"exag={_style_overrides['exaggeration']:.2f} "
                        f"cfg={_style_overrides['cfg_weight']:.2f} "
                        f"temp={_style_overrides['temperature']:.2f}",
                        flush=True,
                    )
                else:
                    # Very short emotion prompts destabilize Chatterbox on SK and can
                    # produce lexical drift or gibberish unrelated to the requested text.
                    # Fall back to the stable global voice sample for short slots.
                    _emo_seg_dur = float(seg["end"]) - float(seg["start"])
                    _emo_text_len = len((text or "").strip())
                    _emo_min_slot_s = 2.8
                    _emo_min_chars = 24
                    if _emo_seg_dur < _emo_min_slot_s or _emo_text_len < _emo_min_chars:
                        audio_prompt = voice_sample
                        _step["emotion_clone_skipped"] = {
                            "reason": "short_segment",
                            "slot_s": round(_emo_seg_dur, 3),
                            "text_chars": _emo_text_len,
                            "min_slot_s": _emo_min_slot_s,
                            "min_chars": _emo_min_chars,
                        }
                        print(
                            f"[EMOTION] Segment {i+1}: skipped short ref "
                            f"({_emo_seg_dur:.2f}s, {_emo_text_len} chars) -> stable voice sample",
                            flush=True,
                        )
                    else:
                        ref_path = chunks_dir / f"ref_{i+1:04d}.wav"
                        if _extract_segment_audio(
                            audio_wav, float(seg["start"]), float(seg["end"]), ref_path,
                            pad_s=getattr(args, "emotion_clone_pad_s", 0.3),
                        ):
                            audio_prompt = str(ref_path)
                            print(f"[EMOTION] Segment {i+1}: using ref {ref_path.name}", flush=True)
                        else:
                            audio_prompt = voice_sample
            else:
                audio_prompt = voice_sample

            # Pause map: extrahuj pauzy z originálneho segmentu a vlož do textu
            # Aktívuje sa pri emotion_clone kde máme prístup k originálu
            if getattr(args, "emotion_clone", False) and audio_wav.exists() and not sql_mode:
                _pauses = _extract_pause_map(
                    src_wav=audio_wav,
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    scratch_dir=chunks_dir,
                    seg_idx=i + 1,
                    min_pause_s=0.28,
                )
                if _pauses:
                    _text_with_pauses = _inject_pauses(text, _pauses)
                    if _text_with_pauses != text:
                        print(
                            f"[PAUSE-MAP] Segment {i+1}: {len(_pauses)} pause(s) injected"
                            f" at {[f'{p:.2f}' for p in _pauses]}",
                            flush=True,
                        )
                        text = _text_with_pauses
                        _step["pause_map"] = {"positions": _pauses, "count": len(_pauses)}

            # SQL breath-chunking: split actual SQL queries at clause boundaries
            # Each clause (SELECT..., FROM..., WHERE...) generates separately → natural pacing
            if sql_mode:
                sql_clauses = split_sql_clauses(text)
            else:
                sql_clauses = [text]

            # Layers 4+5: energy contour + per-chunk style embedding
            # Aktívuje sa pri emotion_clone na segmentoch dlhých ≥2 chunky
            _chunk_overrides: list[dict] | None = None
            _chunk_prompts: list[str | None] | None = None
            _chunk_refs_to_cleanup: list[str] = []

            if (
                not sql_mode
                and getattr(args, "emotion_clone", False)
                and audio_wav.exists()
                and len(sql_clauses) == 1  # len==1 → text ide do chatterbox_tts kde sa splituje
            ):
                # Zistíme počet chunkov ktoré chatterbox_tts vytvorí interným splitom
                # Použijeme rovnaký limit ako chatterbox_tts (180 pre normal, 110 safe, 90 ultra_safe)
                if getattr(args, "tts_ultra_safe", False):
                    _max_chunk = 90
                elif getattr(args, "tts_safe_mode", False):
                    _max_chunk = 110
                else:
                    _max_chunk = 180

                if len(text) > _max_chunk:
                    from tts import _smart_chunk_for_tts as _sct
                    _preview_chunks = _sct(text, max_chars=_max_chunk)
                    _n = len(_preview_chunks)
                else:
                    _n = 1

                if _n >= 2:
                    _exag_base = _style_overrides["exaggeration"] if _style_overrides else _base_exag
                    _cfg_base = _style_overrides["cfg_weight"] if _style_overrides else _base_cfg
                    _temp_base = _style_overrides["temperature"] if _style_overrides else _base_temp

                    # Layer 4a: energy contour → per-chunk exag/cfg/temp
                    _energy_ov = _extract_energy_contour(
                        src_wav=audio_wav,
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        n_chunks=_n,
                        scratch_dir=chunks_dir,
                        seg_idx=i + 1,
                        base_exag=_exag_base,
                        base_cfg=_cfg_base,
                        base_temp=_temp_base,
                    )

                    # Layer 4b: pitch contour → F0-based exag/cfg/temp
                    _pitch_ov = _extract_pitch_contour(
                        src_wav=audio_wav,
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        n_chunks=_n,
                        scratch_dir=chunks_dir,
                        seg_idx=i + 1,
                        base_exag=_exag_base,
                        base_cfg=_cfg_base,
                        base_temp=_temp_base,
                    )

                    # Layer 4c: phoneme duration → onset density per chunk
                    _phoneme_dur = _extract_phoneme_duration(
                        src_wav=audio_wav,
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        n_chunks=_n,
                        scratch_dir=chunks_dir,
                        seg_idx=i + 1,
                    )

                    # Blend energy + pitch + phoneme_dur → finálne per-chunk parametre
                    _chunk_overrides = _blend_prosody_layers(_energy_ov, _pitch_ov, _phoneme_dur)

                    # Layer 5: per-chunk style embedding → každý chunk má vlastný ref
                    _raw_refs = _extract_chunk_references(
                        src_wav=audio_wav,
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        n_chunks=_n,
                        scratch_dir=chunks_dir,
                        seg_idx=i + 1,
                        min_ref_s=1.5,
                    )
                    # Fallback: ak ref neexistuje, použi globálny audio_prompt
                    _chunk_prompts = [r if r else audio_prompt for r in _raw_refs]
                    _chunk_refs_to_cleanup = [r for r in _raw_refs if r]

                    _pitch_vars = [ov.get("pitch_variation", 0.0) for ov in _pitch_ov]
                    _onset_rates = _phoneme_dur
                    print(
                        f"[PROSODY] Segment {i+1}: {_n} chunks"
                        f" | energy+pitch+phoneme blended"
                        f" | pitch_var={[round(v,2) for v in _pitch_vars]}"
                        f" | onset_rate={[round(d,2) for d in _onset_rates]}"
                        f" | chunk_refs={sum(1 for r in _raw_refs if r)}/{_n}",
                        flush=True,
                    )
                    _step["prosody_layers"] = {
                        "n_chunks": _n,
                        "energy_contour": [
                            {"exag": ov["exaggeration"], "energy": ov.get("energy_score")}
                            for ov in _energy_ov
                        ],
                        "pitch_contour": [
                            {"mean_f0": ov.get("mean_f0"), "pitch_var": ov.get("pitch_variation")}
                            for ov in _pitch_ov
                        ],
                        "phoneme_duration": _phoneme_dur,
                        "blended": [
                            {"exag": ov["exaggeration"], "cfg": ov["cfg_weight"], "temp": ov["temperature"]}
                            for ov in _chunk_overrides
                        ],
                        "chunk_refs": [bool(r) for r in _raw_refs],
                    }

            if len(sql_clauses) > 1:
                print(f"[SQL] Clause split → {len(sql_clauses)} parts: {[c[:40] for c in sql_clauses]}", flush=True)
                import torchaudio as _ta
                clause_wavs = []
                silence_100ms = None
                for ci, clause_text in enumerate(sql_clauses):
                    if not clause_text.strip():
                        continue
                    if clause_text[-1] not in ".!?…,:;":
                        clause_text = clause_text + ","
                    clause_tmp = out_chunk.with_suffix(f".clause{ci}.wav")
                    try:
                        chatterbox_tts(
                            text=clause_text, output_path=clause_tmp,
                            model_path=_turbo_fallback_model if _turbo_route_reasons else args.chatterbox_model,
                            device=_turbo_fallback_device() if _turbo_route_reasons else device,
                            use_torch_compile=use_torch_compile,
                            audio_prompt_path=audio_prompt,
                            safe_mode=getattr(args, "tts_safe_mode", False),
                            ultra_safe=getattr(args, "tts_ultra_safe", False),
                            exaggeration=_style_overrides["exaggeration"] if _style_overrides else _base_exag,
                            cfg_weight=_style_overrides["cfg_weight"] if _style_overrides else _base_cfg,
                            temperature=_style_overrides["temperature"] if _style_overrides else _base_temp,
                        )
                    except RuntimeError as _cb_clause_err:
                        if "out of memory" in str(_cb_clause_err).lower() or "Nedostatok VRAM" in str(_cb_clause_err):
                            print(f"[CHATTERBOX] OOM SQL clause {ci} Seg {i+1} — preskakujem klauzu", flush=True)
                            gc.collect(); torch.cuda.empty_cache()
                        else:
                            raise
                    if clause_tmp.exists():
                        wav_data, sr = _ta.load(str(clause_tmp))
                        clause_wavs.append(wav_data)
                        if silence_100ms is None:
                            silence_100ms = torch.zeros(1, int(sr * 0.10))
                        if ci < len(sql_clauses) - 1:
                            clause_wavs.append(silence_100ms)
                        clause_tmp.unlink(missing_ok=True)
                if clause_wavs:
                    combined = torch.cat(clause_wavs, dim=1)
                    _ta.save(str(out_chunk), combined, sr)
                    del clause_wavs, combined
                    _sql_clause_split = True
            else:
                try:
                    chatterbox_tts(
                        text=text,
                        output_path=out_chunk,
                        model_path=_turbo_fallback_model if _turbo_route_reasons else args.chatterbox_model,
                        device=_turbo_fallback_device() if _turbo_route_reasons else device,
                        use_torch_compile=use_torch_compile,
                        audio_prompt_path=audio_prompt,
                        safe_mode=getattr(args, "tts_safe_mode", False),
                        ultra_safe=getattr(args, "tts_ultra_safe", False),
                        exaggeration=_style_overrides["exaggeration"] if _style_overrides else _base_exag,
                        cfg_weight=_style_overrides["cfg_weight"] if _style_overrides else _base_cfg,
                        temperature=_style_overrides["temperature"] if _style_overrides else _base_temp,
                        chunk_overrides=_chunk_overrides,
                        chunk_prompts=_chunk_prompts,
                    )
                except RuntimeError as _cb_err:
                    if "out of memory" in str(_cb_err).lower() or "Nedostatok VRAM" in str(_cb_err):
                        print(f"[CHATTERBOX] OOM Seg {i+1} — ukladám ticho: {_cb_err}", flush=True)
                        _oom_dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
                        write_silence_chunk(out_chunk, _oom_dur)
                        gc.collect(); torch.cuda.empty_cache()
                        _bad_seg_log.append({"seg": i+1, "reason": "oom", "slot": _oom_dur, "text": text})
                    else:
                        raise

            # Cleanup per-chunk reference WAVs (Layer 5)
            for _ref_path in _chunk_refs_to_cleanup:
                try:
                    Path(_ref_path).unlink(missing_ok=True)
                except OSError:
                    pass
        elif use_f5:
            from tts import f5_tts
            f5_tts(
                text=text, output_path=out_chunk,
                ckpt_path=getattr(args, "f5_model", None),
                ref_audio=getattr(args, "f5_ref_audio", None) or voice_sample,
                ref_text=getattr(args, "f5_ref_text", None),
                f5_python=getattr(args, "f5_python", None),
            )
        elif use_omnivoice:
            from tts import omnivoice_tts
            omnivoice_tts(
                text=text, output_path=out_chunk,
                repo_or_path=getattr(args, "omnivoice_repo", None),
                ref_audio=getattr(args, "omnivoice_ref_audio", None) or voice_sample,
                ref_text=getattr(args, "omnivoice_ref_text", None),
                omnivoice_python=getattr(args, "omnivoice_python", None),
                language=getattr(args, "omnivoice_language", "sk"),
                num_step=getattr(args, "omnivoice_num_step", 64),
                guidance_scale=getattr(args, "omnivoice_guidance", 2.5),
                instruct=getattr(args, "omnivoice_instruct", "") or "",
                speed=getattr(args, "omnivoice_speed", 1.0),
                expressive=getattr(args, "omnivoice_expressive", False),
            )
        elif args.use_xtts:
            from tts import xtts_tts
            xtts_tts(text=text, output_path=out_chunk, lang=args.xtts_lang,
                     speaker_wav=args.xtts_speaker_wav or voice_sample or "",
                     speaker=args.xtts_speaker,
                     xtts_python=getattr(args, "xtts_python", ""))
        elif args.use_piper:
            from tts import piper_tts
            piper_tts(model_path=args.piper_model, text=text, output_path=out_chunk)

        # Per-chunk quality gate: detect repetition loops, silence, distortion.
        # One retry with safe_mode; if still bad → silence (not ultra_safe — voice style
        # change mid-pipeline is more jarring than a brief silence).
        # SQL mode: skip duration check — normalized SQL (SELECT/FROM/WHERE → words) is
        # always much longer than the original segment window, both for single and split clauses.
        if out_chunk.exists():
            _seg_dur = float(seg["end"]) - float(seg["start"])
            _qa_expected = None if (sql_mode or _sql_clause_split) else _seg_dur
            _is_bad, _reason = inspect_tts_chunk(out_chunk, expected_sec=_qa_expected)
            if _is_bad and _used_cb_for_segment:
                _retry_prompt = audio_prompt
                _retry_note = "safe_mode"
                # Full per-segment emotion refs from the source audio can destabilize
                # SK dubbing on cross-language material. For retry, fall back to the
                # stable source-wide sample instead of the per-segment ref.
                if getattr(args, "emotion_clone", False) and not _has_explicit_voice and voice_sample and audio_prompt != voice_sample:
                    _retry_prompt = voice_sample
                    _retry_note = "safe_mode + stable_source_sample"
                    _step["emotion_retry_mode"] = "stable_source_sample"
                # With an explicit custom WAV voice, the first pass may use emotion
                # style-transfer overrides. If QA flags it, retry with the same custom
                # voice but with true safe defaults (no override params).
                elif getattr(args, "emotion_clone", False) and _has_explicit_voice:
                    _retry_note = "safe_mode + custom_voice_no_style"
                    _step["emotion_retry_mode"] = "custom_voice_no_style"
                print(f"[QA] Seg {i+1} BAD ({_reason}) — retry {_retry_note}", flush=True)
                try:
                    chatterbox_tts(
                        text=text,
                        output_path=out_chunk,
                        model_path=_turbo_fallback_model if _turbo_route_reasons else args.chatterbox_model,
                        device=_turbo_fallback_device() if _turbo_route_reasons else device,
                        use_torch_compile=False,
                        audio_prompt_path=_retry_prompt,
                        safe_mode=True,
                    )
                except RuntimeError as _cb_qa_err:
                    if "out of memory" in str(_cb_qa_err).lower() or "Nedostatok VRAM" in str(_cb_qa_err):
                        print(f"[CHATTERBOX] OOM Seg {i+1} QA retry — ukladám ticho", flush=True)
                        write_silence_chunk(out_chunk, _seg_dur)
                        gc.collect(); torch.cuda.empty_cache()
                        _bad_seg_log.append({"seg": i+1, "reason": "oom_qa_retry", "slot": _seg_dur, "text": text})
                    else:
                        raise
                _is_bad, _reason = inspect_tts_chunk(out_chunk, expected_sec=_seg_dur)
                if _is_bad:
                    # HYBRID: try S2 Pro before silence
                    if getattr(args, "chatterbox_s2_lastresort", False) and ref_audio_path and ref_text:
                        print(f"[HYBRID] Seg {i+1} Chatterbox QA failed ({_reason}) → trying S2 Pro lastresort", flush=True)
                        try:
                            s2_pro_tts(text=text, output_path=out_chunk,
                                       checkpoint_path=args.s2_pro_model,
                                       reference_audio_path=ref_audio_path, reference_text=ref_text,
                                       server_python=getattr(args, "s2_pro_server_python", ""),
                                       server_port=int(getattr(args, "s2_pro_server_port", 8091)))
                            _is_bad, _reason = inspect_tts_chunk(out_chunk, expected_sec=_seg_dur)
                            if not _is_bad:
                                _step["tts_engine"] = "s2_pro_lastresort"
                                print(f"[HYBRID] Seg {i+1} S2 Pro lastresort OK", flush=True)
                            else:
                                print(f"[HYBRID] Seg {i+1} S2 Pro also BAD ({_reason}) — silence", flush=True)
                        except Exception as _s2_err:
                            print(f"[HYBRID] Seg {i+1} S2 Pro fallback exception: {_s2_err} — silence", flush=True)
                            _is_bad = True
                    if _is_bad:
                        print(f"[QA] Seg {i+1} still BAD ({_reason}) — replacing with silence", flush=True)
                        write_silence_chunk(out_chunk, _seg_dur)
                        _bad_seg_log.append({"seg": i+1, "reason": _reason, "slot": _seg_dur, "text": text})
            elif _is_bad and use_turbo and _turbo_fallback_enabled:
                print(
                    f"[QA] Seg {i+1} BAD ({_reason}) — retry multilingual fallback on {_turbo_fallback_device()}",
                    flush=True,
                )
                _step["tts_engine"] = "chatterbox_fallback_qa"
                _step["turbo_qa_fallback"] = _reason
                _step["turbo_fallback_device"] = _turbo_fallback_device()
                _turbo_fallback_qa += 1
                try:
                    chatterbox_tts(
                        text=text,
                        output_path=out_chunk,
                        model_path=_turbo_fallback_model,
                        device=_turbo_fallback_device(),
                        use_torch_compile=False,
                        audio_prompt_path=audio_prompt,
                        safe_mode=True,
                    )
                except RuntimeError as _cb_fb_err:
                    if "out of memory" in str(_cb_fb_err).lower() or "Nedostatok VRAM" in str(_cb_fb_err):
                        print(f"[CHATTERBOX] OOM Seg {i+1} turbo fallback — ukladám ticho", flush=True)
                        write_silence_chunk(out_chunk, _seg_dur)
                        gc.collect(); torch.cuda.empty_cache()
                        _bad_seg_log.append({"seg": i+1, "reason": "oom_turbo_fallback", "slot": _seg_dur, "text": text})
                    else:
                        raise
                _used_cb_for_segment = True
                _is_bad, _reason = inspect_tts_chunk(out_chunk, expected_sec=_seg_dur)
                if _is_bad:
                    # HYBRID: try S2 Pro before silence
                    if getattr(args, "chatterbox_s2_lastresort", False) and ref_audio_path and ref_text:
                        print(f"[HYBRID] Seg {i+1} Chatterbox turbo-fallback also failed → trying S2 Pro lastresort", flush=True)
                        try:
                            s2_pro_tts(text=text, output_path=out_chunk,
                                       checkpoint_path=args.s2_pro_model,
                                       reference_audio_path=ref_audio_path, reference_text=ref_text,
                                       server_python=getattr(args, "s2_pro_server_python", ""),
                                       server_port=int(getattr(args, "s2_pro_server_port", 8091)))
                            _is_bad, _reason = inspect_tts_chunk(out_chunk, expected_sec=_seg_dur)
                            if not _is_bad:
                                _step["tts_engine"] = "s2_pro_lastresort_after_turbo"
                                print(f"[HYBRID] Seg {i+1} S2 Pro lastresort OK", flush=True)
                        except Exception as _s2_err:
                            print(f"[HYBRID] Seg {i+1} S2 Pro lastresort exception: {_s2_err}", flush=True)
                            _is_bad = True
                    if _is_bad:
                        print(f"[QA] Seg {i+1} fallback still BAD ({_reason}) — replacing with silence", flush=True)
                        write_silence_chunk(out_chunk, _seg_dur)
                        _bad_seg_log.append({"seg": i+1, "reason": _reason, "slot": _seg_dur, "text": text})
            elif _is_bad:
                # Non-Chatterbox TTS: no retry, just replace with silence
                print(f"[QA] Seg {i+1} BAD ({_reason}) — replacing with silence", flush=True)
                write_silence_chunk(out_chunk, _seg_dur)
                _bad_seg_log.append({"seg": i+1, "reason": _reason, "slot": _seg_dur, "text": text})
            if out_chunk.exists():
                # Trim Chatterbox leading noise: model generates ~-30dB hiss before speech starts.
                # Use -25dB threshold + 100ms min-silence to avoid cutting sibilant onsets (š/č/ž
                # grow from -35dB to speech level in <50ms, won't trigger 100ms threshold).
                if _used_cb_for_segment:
                    _trimmed = out_chunk.with_suffix(".lead_trim.wav")
                    _trim_threshold = -25.0
                    _trim_keep = 0.10
                    _tail_threshold = -45.0
                    _tail_keep = 0.10
                    if _short_reply_like:
                        _trim_threshold = -30.0
                        _trim_keep = 0.03
                        _tail_threshold = -42.0
                        _tail_keep = 0.04
                    if trim_edge_silence_ffmpeg(
                        out_chunk,
                        _trimmed,
                        threshold_db=_trim_threshold,
                        keep_s=_trim_keep,
                        tail_threshold_db=_tail_threshold,
                        tail_keep_s=_tail_keep,
                    ):
                        if _trimmed.exists() and _trimmed.stat().st_size > 4096:
                            import shutil as _shutil
                            _shutil.move(str(_trimmed), str(out_chunk))
                        else:
                            _trimmed.unlink(missing_ok=True)
                repair_silence_gaps(out_chunk, max_silence_s=0.35)

        if out_chunk.exists():
            # ── Duration Planner — jediné miesto pravdy pre timing rozhodnutia ──
            from duration_planner import plan as _dp_plan, evaluate as _dp_eval, decide as _dp_decide, Action as _DPAction, select_preset as _dp_select_preset

            _src_text = (seg.get("text_src") or "").strip()
            _dp_spec  = _dp_plan(_src_text, text, _slot_dur, content_type)
            _dp_guard = _dp_eval(out_chunk, _slot_dur, _dp_spec)
            _dp_action = _dp_decide(_dp_guard, _dp_spec)

            # Pomocná funkcia — zachová kompatibilitu s timing audit logom
            def _make_timing_entry(_text_for_audio: str) -> dict:
                _g = _dp_eval(out_chunk, _slot_dur, _dp_spec)
                _wav_dur = _g.wav_dur
                _actual_cps = (len(_text_for_audio) / _wav_dur) if _wav_dur > 0.05 else 0.0
                _status = "ok" if _g.tech_ok else ("too_long" if _g.delta > 0 else "too_short")
                return {
                    "seg": i + 1,
                    "slot": round(_slot_dur, 3),
                    "wav_duration": round(_wav_dur, 3),
                    "delta": _g.delta,
                    "status": _status,
                    "text_chars": len(_text_for_audio),
                    "target_cps": round(float(seg.get("adapt_target_cps", 0.0) or 0.0), 2),
                    "actual_cps": round(_actual_cps, 2),
                    "fill_ratio": _g.fill_ratio,
                    "action": _dp_action.value,
                }

            _timing_entry = _make_timing_entry(text)
            _timing_retry_attempts = 0

            # ── Fáza 1: planner rozhoduje, akcie sú rovnaké ako pred tým ──────

            # REWRITE_SHRINK pre Chatterbox:
            # 1. LLM shrink (Ollama) → regenerate TTS s kratším textom
            # 2. Ak LLM shrink nestačí alebo zlyhal → atempo fallback (max 1.20x)
            # Pri 27B v2 multitask: LLM shrink VYPNUTÝ (sk-gemma4-e4b-v6 by stratil
            #  fine-tuned vzory ako negation transfer). Atempo fallback sám stačí.
            _g3_27b_active = bool(getattr(args, "use_g3_27b_translator", False))
            if _timing_retry_enabled and not use_s2 and _dp_action == _DPAction.REWRITE_SHRINK and not _g3_27b_active:
                _shrink_accepted = False
                _shrink_ollama_url   = getattr(args, "adapt_ollama_url",   "http://localhost:11434/api/generate")
                _shrink_ollama_model = getattr(args, "adapt_ollama_model", "sk-gemma4-e4b-v6")

                # --- Krok 1: LLM shrink + Chatterbox regenerácia ---
                if _shrink_ollama_url and _shrink_ollama_model:
                    from translation import shrink_text_single_ollama as _shrink_fn
                    _src_text_shrink = (seg.get("text_src") or seg.get("text_en") or "").strip()
                    print(
                        f"[PLANNER] Seg {i+1} rewrite_shrink: LLM skracuje"
                        f" ({len(text)}→~{int(_slot_dur * 13)} znakov, delta={_dp_guard.delta:+.2f}s)",
                        flush=True,
                    )
                    _shrunk_text = _shrink_fn(
                        text, _src_text_shrink, _slot_dur,
                        content_type=content_type,
                        ollama_url=_shrink_ollama_url,
                        ollama_model=_shrink_ollama_model,
                        timeout=int(getattr(args, "adapt_ollama_timeout", 120)),
                    )
                    if _shrunk_text and len(_shrunk_text) < len(text):
                        _backup_chunk = out_chunk.with_suffix(".shrink_llm_bak.wav")
                        shutil.copy2(out_chunk, _backup_chunk)
                        try:
                            _shrink_render = _shrunk_text if _shrunk_text[-1] in ".!?…,:;" else _shrunk_text + "."
                            if use_f5:
                                from tts import f5_tts
                                f5_tts(
                                    text=_shrink_render, output_path=out_chunk,
                                    ckpt_path=getattr(args, "f5_model", None),
                                    ref_audio=getattr(args, "f5_ref_audio", None) or voice_sample,
                                    ref_text=getattr(args, "f5_ref_text", None),
                                    f5_python=getattr(args, "f5_python", None),
                                )
                            else:
                                chatterbox_tts(
                                    text=_shrink_render,
                                    output_path=out_chunk,
                                    model_path=str(args.chatterbox_model),
                                    audio_prompt_path=audio_prompt,
                                    exaggeration=_cb_exaggeration,
                                    cfg_weight=_cb_cfg_weight,
                                    temperature=_cb_temperature,
                                    device=device,
                                )
                            _shrink_guard = _dp_eval(out_chunk, _slot_dur, _dp_spec)
                            if abs(_shrink_guard.delta) < abs(_dp_guard.delta):
                                text = _shrunk_text
                                _step["tts_input"] = text
                                _dp_guard = _shrink_guard
                                _dp_action = _dp_decide(_dp_guard, _dp_spec)
                                _shrink_accepted = True
                                _backup_chunk.unlink(missing_ok=True)
                                print(
                                    f"[PLANNER] Seg {i+1} LLM shrink accepted"
                                    f" ({len(_shrunk_text)} znakov, delta={_shrink_guard.delta:+.2f}s)",
                                    flush=True,
                                )
                                _step.setdefault("timing_retry", []).append({
                                    "attempt": 1, "action": "rewrite_shrink_llm",
                                    "text_before": text, "text_after": _shrunk_text,
                                    "delta_before": round(_dp_guard.delta, 3),
                                    "delta_after": round(_shrink_guard.delta, 3),
                                })
                            else:
                                shutil.move(str(_backup_chunk), str(out_chunk))
                                print(
                                    f"[PLANNER] Seg {i+1} LLM shrink rejected (delta nezlepšil)",
                                    flush=True,
                                )
                        except Exception as _shrink_err:
                            _backup_chunk.unlink(missing_ok=True) if not _backup_chunk.exists() else shutil.move(str(_backup_chunk), str(out_chunk))
                            print(f"[PLANNER] Seg {i+1} LLM shrink TTS failed: {_shrink_err}", flush=True)
                    else:
                        print(f"[PLANNER] Seg {i+1} LLM shrink: text nezmenený, skipping", flush=True)

                # --- Krok 2: atempo fallback ak LLM shrink nestačil ---
                if not _shrink_accepted or _dp_action == _DPAction.REWRITE_SHRINK:
                    _atempo_shrink = round(min(1.18, max(1.0, _dp_guard.stretch_ratio)), 4)
                    print(
                        f"[PLANNER] Seg {i+1} rewrite_shrink→atempo fallback"
                        f" atempo={_atempo_shrink:.4f} delta={_dp_guard.delta:+.2f}s",
                        flush=True,
                    )
                    _stretched_shrink = out_chunk.with_suffix(".shrink_atempo.wav")
                    import subprocess as _sp_shrink
                    _rc_shrink = _sp_shrink.run(
                        ["ffmpeg", "-y", "-i", str(out_chunk),
                         "-af", f"atempo={_atempo_shrink:.4f}", str(_stretched_shrink)],
                        capture_output=True,
                    )
                    if _rc_shrink.returncode == 0 and _stretched_shrink.exists():
                        shutil.move(str(_stretched_shrink), str(out_chunk))
                        _dp_guard = _dp_eval(out_chunk, _slot_dur, _dp_spec)
                        _timing_entry = _make_timing_entry(text)
                        _step.setdefault("timing_retry", []).append({
                            "attempt": 1, "action": "rewrite_shrink_atempo",
                            "atempo": _atempo_shrink,
                            "after": _timing_entry, "accepted": True,
                        })
                    else:
                        _stretched_shrink.unlink(missing_ok=True)
                        print(f"[PLANNER] Seg {i+1} rewrite_shrink→atempo failed", flush=True)

            elif _timing_retry_enabled and use_s2 and _dp_action == _DPAction.REWRITE_SHRINK and not _g3_27b_active:
                while _timing_retry_attempts < _timing_retry_max_attempts:
                    _retry_target_cps = float(seg.get("adapt_target_cps", 16.0) or 16.0)
                    _retry_text = tighten_text_for_retry(
                        text, _slot_dur,
                        target_cps=_retry_target_cps,
                        preserve_sql_terms=bool(seg.get("adapt_preserve_sql_terms", False) or use_s2),
                    )
                    if not _retry_text or len(_retry_text) >= len(text):
                        break

                    _timing_retry_attempts += 1
                    _retry_render_text = _retry_text
                    if _retry_render_text[-1] not in ".!?…,:;":
                        _retry_render_text += "."
                    _backup_chunk = out_chunk.with_suffix(f".timing_retry_{_timing_retry_attempts}.bak.wav")
                    shutil.copy2(out_chunk, _backup_chunk)
                    _before_timing = dict(_timing_entry)
                    _before_text = text
                    print(
                        f"[PLANNER] Seg {i+1} {_dp_action.value} delta={_dp_guard.delta:+.2f}s"
                        f" → rewrite_shrink attempt {_timing_retry_attempts}",
                        flush=True,
                    )
                    s2_pro_tts(
                        text=_retry_render_text,
                        output_path=out_chunk,
                        checkpoint_path=args.s2_pro_model,
                        reference_audio_path=ref_audio_path,
                        reference_text=ref_text,
                        server_python=getattr(args, "s2_pro_server_python", ""),
                        server_port=int(getattr(args, "s2_pro_server_port", 8091)),
                    )
                    _retry_bad, _retry_reason = inspect_tts_chunk(out_chunk, expected_sec=_slot_dur)
                    if not _retry_bad:
                        repair_silence_gaps(out_chunk, max_silence_s=0.35)
                        _retry_timing = _make_timing_entry(_retry_text)
                        _accepted = abs(float(_retry_timing["delta"])) < abs(float(_before_timing["delta"]))
                        if _accepted:
                            text = _retry_text
                            _step["tts_input"] = text
                            _timing_entry = _retry_timing
                            _dp_guard = _dp_eval(out_chunk, _slot_dur, _dp_spec)
                            _dp_action = _dp_decide(_dp_guard, _dp_spec)
                        else:
                            shutil.move(str(_backup_chunk), str(out_chunk))
                    else:
                        shutil.move(str(_backup_chunk), str(out_chunk))
                        _accepted = False

                    _step.setdefault("timing_retry", []).append({
                        "attempt": _timing_retry_attempts,
                        "action": "rewrite_shrink",
                        "text_before": _before_text,
                        "text_after": _retry_text,
                        "before": _before_timing,
                        "after": _retry_timing if not _retry_bad else _before_timing,
                        "accepted": _accepted,
                        "reason": None if _accepted else (_retry_reason if _retry_bad else "not_better"),
                    })
                    _backup_chunk.unlink(missing_ok=True)
                    if not _accepted or _dp_action != _DPAction.REWRITE_SHRINK:
                        break

            # STRETCH → atempo filter (malá odchýlka, v rámci stretch limitov)
            elif _timing_retry_enabled and _dp_action == _DPAction.STRETCH:
                # stretch_ratio = wav_dur / slot_dur → priamo atempo hodnota
                _atempo = round(max(0.5, min(2.0, _dp_guard.stretch_ratio)), 4)
                print(
                    f"[PLANNER] Seg {i+1} stretch atempo={_atempo:.4f} delta={_dp_guard.delta:+.2f}s",
                    flush=True,
                )
                _stretched = out_chunk.with_suffix(".stretched.wav")
                import subprocess as _sp
                _stretch_rc = _sp.run(
                    ["ffmpeg", "-y", "-i", str(out_chunk),
                     "-af", f"atempo={_atempo:.4f}", str(_stretched)],
                    capture_output=True,
                )
                if _stretch_rc.returncode == 0 and _stretched.exists():
                    shutil.move(str(_stretched), str(out_chunk))
                    _dp_guard = _dp_eval(out_chunk, _slot_dur, _dp_spec)
                    _timing_entry = _make_timing_entry(text)
                    _step.setdefault("timing_retry", []).append({
                        "attempt": 1, "action": "atempo",
                        "atempo": _atempo,
                        "after": _timing_entry, "accepted": True,
                    })
                else:
                    _stretched.unlink(missing_ok=True)
                    print(
                        f"[PLANNER] Seg {i+1} atempo failed (rc={_stretch_rc.returncode})",
                        flush=True,
                    )

            # REWRITE_EXPAND → ollama expand + re-TTS, fallback na apad
            # Pri 27B v2 multitask: VYPNUTÝ (sk-gemma4-e4b-v6 by stratil fine-tuned vzory
            #  ako negation transfer). Atempo/apad fallback sám stačí.
            elif _timing_retry_enabled and _dp_action == _DPAction.REWRITE_EXPAND and not _g3_27b_active:
                from translation import expand_text_single_ollama as _expand_text
                _ollama_url   = getattr(args, "adapt_ollama_url",     "http://localhost:11434/api/generate")
                _ollama_model = getattr(args, "adapt_ollama_model",   "gemma4:26b")
                _ollama_to    = int(getattr(args, "adapt_ollama_timeout", 45))
                _src_for_exp  = (seg.get("text_src") or "").strip()

                _expanded = _expand_text(
                    text, _src_for_exp, _slot_dur,
                    content_type=content_type,
                    ollama_url=_ollama_url,
                    ollama_model=_ollama_model,
                    timeout=_ollama_to,
                )

                _expand_accepted = False
                if _expanded and _expanded != text and len(_expanded) > len(text) * 0.95:
                    _backup_expand = out_chunk.with_suffix(".expand_bak.wav")
                    shutil.copy2(out_chunk, _backup_expand)
                    print(
                        f"[PLANNER] Seg {i+1} rewrite_expand: {len(text)}→{len(_expanded)} chars, re-TTS",
                        flush=True,
                    )
                    # Re-render s planner presetom (exag/cfg/temp)
                    _exp_preset = _dp_select_preset(_slot_dur, content_type)
                    if _used_cb_for_segment:
                        try:
                            chatterbox_tts(
                                text=_expanded,
                                output_path=out_chunk,
                                model_path=_turbo_fallback_model if _turbo_route_reasons else args.chatterbox_model,
                                device=_turbo_fallback_device() if _turbo_route_reasons else device,
                                use_torch_compile=False,
                                audio_prompt_path=audio_prompt,
                                safe_mode=getattr(args, "tts_safe_mode", False),
                                exaggeration=_exp_preset.exaggeration,
                                cfg_weight=_exp_preset.cfg_weight,
                                temperature=_exp_preset.temperature,
                            )
                        except RuntimeError as _cb_exp_err:
                            if "out of memory" in str(_cb_exp_err).lower() or "Nedostatok VRAM" in str(_cb_exp_err):
                                print(f"[PLANNER] OOM Seg {i+1} expand re-TTS — zachovávam pôvodný chunk", flush=True)
                                if _backup_expand.exists():
                                    shutil.move(str(_backup_expand), str(out_chunk))
                                gc.collect(); torch.cuda.empty_cache()
                                _expand_accepted = False
                                _dp_action = _DPAction.ACCEPT  # preskočíme dp_eval nižšie
                            else:
                                raise
                    if out_chunk.exists() and _backup_expand.exists():
                        _new_guard = _dp_eval(out_chunk, _slot_dur, _dp_spec)
                        if abs(_new_guard.delta) < abs(_dp_guard.delta):
                            text = _expanded
                            _step["tts_input"] = text
                            _dp_guard = _new_guard
                            _timing_entry = _make_timing_entry(text)
                            _expand_accepted = True
                            _backup_expand.unlink(missing_ok=True)
                            print(
                                f"[PLANNER] Seg {i+1} expand accepted"
                                f" delta={_dp_guard.delta:+.2f}s",
                                flush=True,
                            )
                        else:
                            shutil.move(str(_backup_expand), str(out_chunk))
                            print(
                                f"[PLANNER] Seg {i+1} expand rejected (not better), reverting",
                                flush=True,
                            )
                    else:
                        if _backup_expand.exists():
                            shutil.move(str(_backup_expand), str(out_chunk))

                    _step.setdefault("timing_retry", []).append({
                        "attempt": 1, "action": "rewrite_expand",
                        "text_before": seg.get("tts_input", text),
                        "text_after": _expanded if _expand_accepted else text,
                        "accepted": _expand_accepted,
                    })

                # Fallback: apad ak expand nezafungoval / bol odmietnutý
                # Preskočiť apad keď:
                #  - TTS vygenerovalo < 0.5s → pravdepodobné zlyhanie → nechaj bez paddingu
                #  - text je prirodzene riedky (CPS < 7) → dlhý slot s krátkym textom = zámerná pauza
                _apad_text_cps = len(text) / _slot_dur if _slot_dur > 0 else 0
                _apad_skip = (
                    _dp_guard.wav_dur < 0.5  # Chatterbox zlyhal
                    or _apad_text_cps < 7.0  # text je prirodzene krátky pre slot
                )
                if not _expand_accepted:
                    _ts_missing = _slot_dur - _dp_guard.wav_dur
                    if _ts_missing > 0.05 and not _apad_skip:
                        _padded = out_chunk.with_suffix(".padded.wav")
                        import subprocess as _sp
                        _pad_rc = _sp.run([
                            "ffmpeg", "-y", "-i", str(out_chunk),
                            "-af", f"apad=pad_dur={_ts_missing:.3f}",
                            "-t", str(_slot_dur), str(_padded),
                        ], capture_output=True)
                        if _pad_rc.returncode == 0 and _padded.exists():
                            shutil.move(str(_padded), str(out_chunk))
                            _timing_entry = _make_timing_entry(text)
                            _step.setdefault("timing_retry", []).append({
                                "attempt": 2, "action": "apad_fallback",
                                "missing_s": round(_ts_missing, 3),
                                "after": _timing_entry, "accepted": True,
                            })
                        else:
                            _padded.unlink(missing_ok=True)
                    _hint = expand_text_hint(text, _slot_dur, _dp_guard.wav_dur)
                    _step["too_short_rewrite_hint"] = _hint

            # SPLIT_OR_RETRY — last resort: trim (too_long) alebo apad (too_short/low_fill)
            elif _dp_action == _DPAction.SPLIT_OR_RETRY:
                _sor_issue = _dp_guard.tech_issue
                _sor_action = "unknown"
                import subprocess as _sp

                if _sor_issue == "too_long":
                    # Tvrdý trim na slot_dur — audio je >2s príliš dlhé
                    _trimmed = out_chunk.with_suffix(".trimmed.wav")
                    _trim_rc = _sp.run(
                        ["ffmpeg", "-y", "-i", str(out_chunk),
                         "-t", str(_slot_dur), str(_trimmed)],
                        capture_output=True,
                    )
                    if _trim_rc.returncode == 0 and _trimmed.exists():
                        shutil.move(str(_trimmed), str(out_chunk))
                        _dp_guard = _dp_eval(out_chunk, _slot_dur, _dp_spec)
                        _timing_entry = _make_timing_entry(text)
                        _sor_action = "hard_trim"
                    else:
                        _trimmed.unlink(missing_ok=True)
                        _sor_action = "hard_trim_failed"

                else:
                    # too_short / low_fill / silence_tail — apad na slot_dur
                    # Preskočiť keď TTS zlyhal alebo text je prirodzene krátky pre slot
                    _sor_text_cps = len(text) / _slot_dur if _slot_dur > 0 else 0
                    _sor_apad_skip = (
                        _dp_guard.wav_dur < 0.5  # TTS zlyhalo
                        or _sor_text_cps < 7.0   # zámerná pauza v origináli
                    )
                    _sor_missing = _slot_dur - _dp_guard.wav_dur
                    if _sor_missing > 0.05 and not _sor_apad_skip:
                        _padded = out_chunk.with_suffix(".sor_padded.wav")
                        _pad_rc = _sp.run([
                            "ffmpeg", "-y", "-i", str(out_chunk),
                            "-af", f"apad=pad_dur={_sor_missing:.3f}",
                            "-t", str(_slot_dur), str(_padded),
                        ], capture_output=True)
                        if _pad_rc.returncode == 0 and _padded.exists():
                            shutil.move(str(_padded), str(out_chunk))
                            _dp_guard = _dp_eval(out_chunk, _slot_dur, _dp_spec)
                            _timing_entry = _make_timing_entry(text)
                            _sor_action = "apad_last_resort"
                        else:
                            _padded.unlink(missing_ok=True)
                            _sor_action = "apad_failed"

                    # Skús nájsť split bod — hint pre budúce re-spracovanie
                    _split_idx = find_split_index(text)
                    if _split_idx:
                        _step["split_hint"] = {
                            "split_at": _split_idx,
                            "part1": text[:_split_idx].strip(),
                            "part2": text[_split_idx:].strip(),
                        }

                print(
                    f"[PLANNER] Seg {i+1} split_or_retry({_sor_issue})"
                    f" delta={_dp_guard.delta:+.2f}s fill={_dp_guard.fill_ratio:.2f}"
                    f" → {_sor_action}",
                    flush=True,
                )
                _step.setdefault("timing_retry", []).append({
                    "attempt": 1, "action": f"split_or_retry_{_sor_action}",
                    "issue": _sor_issue,
                    "after": _timing_entry,
                })
                _hint = expand_text_hint(text, _slot_dur, _dp_guard.wav_dur)
                _step["too_short_rewrite_hint"] = _hint

            _timing_audit_log.append(_timing_entry)
            _step["timing_audit"] = _timing_entry
            if _timing_entry["status"] != "ok":
                _timing_wav_duration = float(_timing_entry.get("wav_duration") or 0.0)
                _timing_delta = float(_timing_entry.get("delta") or 0.0)
                print(
                    f"[TIMING] Seg {i+1} {_timing_entry['status']} "
                    f"slot={_slot_dur:.2f}s wav={_timing_wav_duration:.2f}s delta={_timing_delta:+.2f}s",
                    flush=True,
                )

        if out_chunk.exists():
            try:
                _tts_cache_meta = dict(_tts_cache_payload)
                _tts_cache_meta["final_text"] = text
                _tts_cache_meta["wav"] = str(out_chunk)
                _tts_cache_meta["wav_size"] = int(out_chunk.stat().st_size)
                _tts_cache_meta["semantic_hash"] = _tts_cache_hash
                _tts_cache_meta_path.write_text(
                    json.dumps(_tts_cache_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as _cache_write_err:
                print(
                    f"[TTS] Seg {i+1} cache meta save failed: {_cache_write_err}",
                    flush=True,
                )

        # ── Post-processing: F0 transfer + WAV pause injection ──────────────────
        if out_chunk.exists() and getattr(args, "emotion_clone", False) and audio_wav.exists() and not sql_mode:
            import shutil as _sh
            _src_s = float(seg["start"])
            _src_e = float(seg["end"])
            _seg_dur = _src_e - _src_s
            _dur_limit = float(getattr(args, "postproc_dur_limit", 0.50))

            # Guardrail: blend weight podľa dĺžky segmentu (škálovaný --f0_blend_scale)
            _f0_scale = float(getattr(args, "f0_blend_scale", 1.0))
            _blend_w = _f0_blend_for_duration(_seg_dur, scale=_f0_scale)

            # Vždy ulož raw kópiu — potrebná pre metriky aj fallback
            _pp_raw = out_chunk.with_suffix(".pp_raw.wav")
            _pp_pause: Path | None = None
            _sh.copy2(out_chunk, _pp_raw)

            # A/B export dir (voliteľný)
            _ab_dir = chunks_dir / "ab_export"
            _do_ab = getattr(args, "ab_export", False)
            if _do_ab:
                _ab_dir.mkdir(exist_ok=True)
                _sh.copy2(_pp_raw, _ab_dir / f"seg{i+1:04d}_raw.wav")

            # ── Krok 1: WAV pause injection ──────────────────────────────────
            _pm = _step.get("pause_map")
            _n_pauses_injected = 0
            if _pm and getattr(args, "wav_pause_injection", True) and _seg_dur >= 0.8:
                _max_p_ms = 200 if _seg_dur < 2.0 else 450
                _pw_ok = _inject_pauses_wav(
                    tts_wav=out_chunk,
                    pause_positions=_pm["positions"],
                    seg_idx=i + 1,
                    max_pause_ms=_max_p_ms,
                )
                _step["wav_pause_injection"] = _pw_ok
                if _pw_ok:
                    _n_pauses_injected = _pm.get("count", 0)
                    _pp_pause = out_chunk.with_suffix(".pp_pause.wav")
                    _sh.copy2(out_chunk, _pp_pause)
                    if _do_ab:
                        _sh.copy2(_pp_pause, _ab_dir / f"seg{i+1:04d}_pause_only.wav")
            elif _seg_dur < 0.8:
                print(f"[PAUSE-WAV] Seg {i+1}: preskočené (dur={_seg_dur:.2f}s < 0.8s)", flush=True)

            # ── Krok 2: F0 transfer ──────────────────────────────────────────
            if getattr(args, "f0_transfer", True):
                if _blend_w == 0.0:
                    print(f"[F0-TRANSFER] Seg {i+1}: preskočené (dur={_seg_dur:.2f}s < 1.2s)", flush=True)
                    _step["f0_transfer"] = "skipped_short"
                else:
                    _f0_ok = _apply_f0_transfer(
                        src_wav=audio_wav,
                        src_start=_src_s,
                        src_end=_src_e,
                        tts_wav=out_chunk,
                        seg_idx=i + 1,
                        blend_weight=_blend_w,
                    )
                    _step["f0_transfer"] = _f0_ok
                    _step["f0_blend_weight"] = _blend_w

            # ── Krok 3: Metriky + quality score + auto-fallback ──────────────
            _metrics = _compute_postproc_metrics(
                wav_before=_pp_raw,
                wav_after=out_chunk,
                seg_idx=i + 1,
                n_pauses=_n_pauses_injected,
            )
            _score, _decision = _postproc_quality_score(_metrics, dur_limit=_dur_limit, seg_dur_s=_seg_dur)
            _metrics["quality_score"] = _score
            _metrics["decision"] = _decision
            _metrics["seg_dur_s"] = round(_seg_dur, 3)
            _metrics["eff_dur_limit"] = round(min(_dur_limit, _relative_dur_limit(_seg_dur)), 3)
            _step["postproc_metrics"] = _metrics

            # Auto-fallback: vyber najlepší variant
            _fallback_applied = False
            if _decision == "raw":
                _sh.copy2(_pp_raw, out_chunk)
                _fallback_applied = True
                print(
                    f"[POSTPROC] ⚠ Seg {i+1}: FALLBACK→raw "
                    f"score={_score:.2f} dur_delta={_metrics.get('dur_delta_s',0):+.3f}s "
                    f"rms={_metrics.get('rms_drift_db',0):+.1f}dB "
                    f"f0_var={_metrics.get('f0_var_before',0):.0f}→{_metrics.get('f0_var_after',0):.0f}Hz",
                    flush=True,
                )
            elif _decision == "pause_only" and _pp_pause:
                _sh.copy2(_pp_pause, out_chunk)
                _fallback_applied = True
                print(
                    f"[POSTPROC] ⚠ Seg {i+1}: FALLBACK→pause_only "
                    f"score={_score:.2f} dur_delta={_metrics.get('dur_delta_s',0):+.3f}s",
                    flush=True,
                )
            else:
                # pause+f0 OK — len info ak bol drift nad 30% limitu
                dur_d = abs(_metrics.get("dur_delta_s", 0.0))
                if dur_d > _dur_limit * 0.30:
                    print(
                        f"[POSTPROC] Seg {i+1}: score={_score:.2f} ({_decision}) "
                        f"dur_delta={_metrics.get('dur_delta_s',0):+.3f}s "
                        f"rms={_metrics.get('rms_drift_db',0):+.1f}dB",
                        flush=True,
                    )

            _step["postproc_decision"] = _decision
            _step["postproc_fallback"] = _fallback_applied

            # A/B: finálny variant
            if _do_ab:
                _sh.copy2(out_chunk, _ab_dir / f"seg{i+1:04d}_final_{_decision.replace('+','_')}.wav")

            # Uprac temp súbory
            _pp_raw.unlink(missing_ok=True)
            if _pp_pause:
                _pp_pause.unlink(missing_ok=True)

        wav_paths.append(out_chunk)

    # Save postproc metrics summary (ak boli počítané)
    _postproc_all = [
        s.get("postproc_metrics")
        for s in _tts_debug_log
        if s.get("postproc_metrics")
    ]
    if _postproc_all:
        _n_pp = len(_postproc_all)

        # ── Fallback rate podľa typu segmentu ────────────────────────────────
        # Typy: short (<1.5s), medium (1.5–4s), long (>4s)
        _buckets: dict[str, dict[str, int]] = {
            "short":  {"pause+f0": 0, "pause_only": 0, "raw": 0, "total": 0},
            "medium": {"pause+f0": 0, "pause_only": 0, "raw": 0, "total": 0},
            "long":   {"pause+f0": 0, "pause_only": 0, "raw": 0, "total": 0},
        }
        for _m in _postproc_all:
            _sd = _m.get("seg_dur_s", 0.0)
            _bk = "short" if _sd < 1.5 else ("medium" if _sd <= 4.0 else "long")
            _dec = _m.get("decision", "pause+f0")
            _buckets[_bk][_dec] = _buckets[_bk].get(_dec, 0) + 1
            _buckets[_bk]["total"] += 1

        def _pct(n, total):
            return round(100.0 * n / total, 1) if total else 0.0

        _fallback_summary: dict = {}
        for _bk, _bc in _buckets.items():
            if _bc["total"] == 0:
                continue
            _t = _bc["total"]
            _fallback_summary[_bk] = {
                "total": _t,
                "pause+f0":   {"n": _bc["pause+f0"],   "pct": _pct(_bc["pause+f0"], _t)},
                "pause_only": {"n": _bc["pause_only"],  "pct": _pct(_bc["pause_only"], _t)},
                "raw":        {"n": _bc["raw"],         "pct": _pct(_bc["raw"], _t)},
            }

        _pp_path = out_dir / f"{stem}_{tgt}_postproc_metrics.json"
        _pp_summary = {
            "total_segments": _n_pp,
            "fallback_rate": _fallback_summary,
            "overall": {
                "pause+f0":   _pct(sum(b["pause+f0"]   for b in _buckets.values()), _n_pp),
                "pause_only": _pct(sum(b["pause_only"]  for b in _buckets.values()), _n_pp),
                "raw":        _pct(sum(b["raw"]         for b in _buckets.values()), _n_pp),
            },
            "total_dur_delta_s":   round(sum(m.get("dur_delta_s", 0.0)   for m in _postproc_all), 3),
            "avg_rms_drift_db":    round(sum(m.get("rms_drift_db", 0.0)  for m in _postproc_all) / _n_pp, 2),
            "avg_f0_var_delta":    round(sum(m.get("f0_var_delta", 0.0)  for m in _postproc_all) / _n_pp, 2),
            "total_pauses_injected": sum(m.get("n_pauses_injected", 0)   for m in _postproc_all),
            "segments": _postproc_all,
        }
        _pp_path.write_text(_json.dumps(_pp_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        # Log súhrn — kompaktný formát pre rýchle čítanie
        print("[POSTPROC] ── Fallback rate ──────────────────────────────────", flush=True)
        for _bk, _fs in _fallback_summary.items():
            print(
                f"[POSTPROC]  {_bk:6s} (n={_fs['total']:3d}): "
                f"pause+f0={_fs['pause+f0']['pct']:5.1f}%  "
                f"pause_only={_fs['pause_only']['pct']:5.1f}%  "
                f"raw={_fs['raw']['pct']:5.1f}%",
                flush=True,
            )
        print(
            f"[POSTPROC]  overall  (n={_n_pp:3d}): "
            f"pause+f0={_pp_summary['overall']['pause+f0']:5.1f}%  "
            f"pause_only={_pp_summary['overall']['pause_only']:5.1f}%  "
            f"raw={_pp_summary['overall']['raw']:5.1f}%",
            flush=True,
        )
        print(
            f"[POSTPROC] dur_delta={_pp_summary['total_dur_delta_s']:+.3f}s"
            f" | rms_drift={_pp_summary['avg_rms_drift_db']:+.1f}dB"
            f" | f0_var_delta={_pp_summary['avg_f0_var_delta']:+.1f}Hz"
            f" | pauses={_pp_summary['total_pauses_injected']}"
            f" → {_pp_path.name}",
            flush=True,
        )

    # Save TTS debug log and hallucination bucket
    _tts_log_path = out_dir / f"{stem}_{tgt}_tts_debug.json"
    _tts_log_path.write_text(_json.dumps(_tts_debug_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TTS] Debug log saved → {_tts_log_path}", flush=True)
    if _timing_audit_log:
        _timing_path = out_dir / f"{stem}_{tgt}_timing_audit.json"
        _timing_path.write_text(_json.dumps(_timing_audit_log, ensure_ascii=False, indent=2), encoding="utf-8")
        _timing_counts = {"ok": 0, "too_long": 0, "too_short": 0}
        for _item in _timing_audit_log:
            _timing_counts[_item["status"]] = _timing_counts.get(_item["status"], 0) + 1
        print(
            f"[TIMING] Audit saved → {_timing_path} "
            f"(ok={_timing_counts.get('ok',0)} too_long={_timing_counts.get('too_long',0)} "
            f"too_short={_timing_counts.get('too_short',0)})",
            flush=True,
        )
        # ── Aktualizuj CPS kalibráciu z nameraných WAV dĺžok ─────────────────
        if _cps_cal is not None:
            _cps_updated = _cps_cal.update_from_audit_log(_timing_audit_log)
            if _cps_updated > 0:
                _cps_cal.save(_cps_cal_path)
                print(
                    f"[CPS] Kalibrácia aktualizovaná → {_cps_cal.effective_cps:.2f} c/s "
                    f"({_cps_updated} meraní, konfidencia={_cps_cal.confidence})",
                    flush=True,
                )
    if _bad_seg_log:
        _bad_log_path = out_dir / f"{stem}_{tgt}_tts_bad_segments.json"
        _bad_log_path.write_text(_json.dumps(_bad_seg_log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[TTS] Bad segments log ({len(_bad_seg_log)}) saved → {_bad_log_path}", flush=True)

    # Release TTS model from VRAM
    _turbo_used_cb_fallback = (_turbo_fallback_preroute + _turbo_fallback_qa) > 0
    if (use_cb or _turbo_used_cb_fallback) and hasattr(chatterbox_tts, "_model") and chatterbox_tts._model is not None:
        del chatterbox_tts._model
        chatterbox_tts._model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[CHATTERBOX] Model released from memory", flush=True)
    # Reset load-failed flag pre ďalší beh
    chatterbox_tts._model_load_failed = False
    if use_turbo and hasattr(chatterbox_turbo_tts, "_model") and chatterbox_turbo_tts._model is not None:
        del chatterbox_turbo_tts._model
        chatterbox_turbo_tts._model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[TURBO] Model released from memory", flush=True)

    # --- Gap compression (pre-assembly) ---
    # Disabled by default for timeline-synced dubbing because any timestamp shift
    # can make speech run ahead of the on-screen action.
    _max_pause = getattr(args, "max_pause_gap_s", 0.0) or 0.0
    _timeline_synced = bool(getattr(args, "timeline_mode", False) or getattr(args, "timed", False))
    _compress_timeline_gaps = bool(getattr(args, "compress_timeline_gaps", False))
    if _timeline_synced and not _compress_timeline_gaps:
        print("[GAP] Preserving original timeline gaps for sync-safe assembly", flush=True)
    elif _max_pause > 0 and len(translated) > 1:
        _shift = 0.0
        _gaps_compressed = 0
        _max_total_shift = max(3.0, min(20.0, total_dur * 0.02))
        _shift_cap_hit = False
        for _gi in range(1, len(translated)):
            _prev_end   = translated[_gi - 1]["end"]          # already shifted (modified in prev iter)
            _this_start = translated[_gi]["start"] - _shift    # original − accumulated shift so far
            _gap = _this_start - _prev_end
            if _gap > _max_pause:
                _remaining = max(0.0, _max_total_shift - _shift)
                if _remaining > 0:
                    _excess = min(_gap - _max_pause, _remaining)
                    _shift  += _excess
                    _gaps_compressed += 1
                    if _excess < (_gap - _max_pause):
                        _shift_cap_hit = True
            if _shift > 0:
                translated[_gi]["start"] -= _shift
                translated[_gi]["end"]   -= _shift
        if _gaps_compressed:
            total_dur_orig = total_dur
            total_dur = max(total_dur - _shift, translated[-1]["end"] + 0.5)
            _cap_msg = f"  [cap hit at {_max_total_shift:.1f}s]" if _shift_cap_hit or _shift >= (_max_total_shift - 1e-6) else ""
            print(f"[GAP] Compressed {_gaps_compressed} long pauses (>{_max_pause}s) → "
                  f"total shift={_shift:.1f}s  duration: {total_dur_orig:.1f}s → {total_dur:.1f}s{_cap_msg}", flush=True)

    # Save effective spoken sidecars after all TTS text fixes and timeline shifts.
    _effective_tts_segments = _build_effective_tts_segments(translated, _tts_debug_log)
    _tts_segments_path = out_dir / f"{stem}_{tgt}_tts_segments.json"
    _tts_input_srt_path = out_dir / f"{stem}_{tgt}_tts_input.srt"
    _write_segments_json(_effective_tts_segments, _tts_segments_path)
    _write_srt(_effective_tts_segments, _tts_input_srt_path)
    _effective_mismatch = sum(
        1
        for _base, _eff in zip(translated, _effective_tts_segments)
        if (_base.get("text") or "").strip() != (_eff.get("text") or "").strip()
    )
    if _effective_mismatch:
        print(f"[TTS] Effective spoken text differs from base segments in {_effective_mismatch} segments", flush=True)
    if getattr(args, "use_level6_plus", False):
        _run_level6_plus_sidecar(
            base_segments=translated,
            effective_segments=_effective_tts_segments,
            tts_debug_log=_tts_debug_log,
            timing_audit_log=_timing_audit_log,
            wav_paths=wav_paths,
            out_dir=out_dir,
            stem=stem,
            tgt=tgt,
            args=args,
        )

    # --- Assembly ---
    out_wav = out_dir / f"{stem}_{tgt}_tts.wav"
    if args.timeline_mode or args.timed:
        # Use pyvideotrans concat style by default (gap-collapse + pydub + librosa)
        # Fall back to adelay/amix only when explicitly requested via --legacy_assembly
        if getattr(args, "legacy_assembly", False):
            assemble_timeline_ffmpeg(
                segments=translated, seg_audio=wav_paths, out_path=out_wav,
                total_dur=total_dur, stretch_min=args.stretch_min,
                stretch_max=args.stretch_max, base_tempo=args.base_tempo,
                gap_collapse=False,
                fill_slot=getattr(args, "fill_slot", False),
            )
        else:
            assemble_concat_style(
                segments=translated, seg_audio=wav_paths, out_path=out_wav,
                total_dur=total_dur, stretch_max=args.stretch_max,
                base_tempo=args.base_tempo,
                min_fill_tempo=getattr(args, "assembly_min_fill_tempo", 0.70),
            )
    else:
        concat_wavs(wav_paths, out_wav)
    print(f"[DONE] Audio assembled: {out_wav}", flush=True)

    # --- Post-run pipeline metrics ---
    try:
        import subprocess as _sp
        _atempo_used = 0
        _trim_used   = 0
        _ok_fit      = 0
        for _i, (_seg, _wp) in enumerate(zip(translated, wav_paths)):
            if not _wp or not Path(_wp).exists():
                continue
            _slot_dur = float(_seg["end"]) - float(_seg["start"])
            if _i + 1 < len(translated):
                _slot_dur = float(translated[_i + 1]["start"]) - float(_seg["start"])
            _r = _sp.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(_wp)],
                capture_output=True, text=True,
            )
            _tts_dur = float(_r.stdout.strip()) if _r.stdout.strip() else _slot_dur
            _ratio   = _tts_dur / _slot_dur if _slot_dur > 0 else 1.0
            if _ratio > args.stretch_max:
                _trim_used += 1
            elif _ratio > 1.02:
                _atempo_used += 1
            else:
                _ok_fit += 1

        _total = len(translated)
        _adapt_d = vars(_adapt_stats) if _adapt_stats is not None else {}
        _adapted = (
            _adapt_d.get("concise_rewrite", 0)
            + _adapt_d.get("dub_friendly_rewrite", 0)
            + _adapt_d.get("aggressive_rewrite", 0)
        )
        _ok_adapt = _adapt_d.get("ok_no_change", 0) + _adapt_d.get("skip_sql", 0)
        _avg_b = round(_adapt_d.get("ratio_before_sum", 0) / max(_adapt_d.get("overflow_count", 1), 1), 2)
        _avg_a = round(_adapt_d.get("ratio_after_sum",  0) / max(_adapt_d.get("overflow_count", 1), 1), 2)
        print(f"\n{'═'*55}", flush=True)
        print(f"[PIPELINE METRICS]  {_total} segmentov", flush=True)
        print(f"  adapt  ok_no_change : {_ok_adapt}", flush=True)
        print(
            f"  adapt  rewritten    : {_adapted}  "
            f"(concise={_adapt_d.get('concise_rewrite',0)}  "
            f"dub={_adapt_d.get('dub_friendly_rewrite',0)}  "
            f"aggr={_adapt_d.get('aggressive_rewrite',0)})",
            flush=True,
        )
        print(f"  adapt  fallback     : {_adapt_d.get('fallback_stretch',0)}", flush=True)
        print(f"  adapt  avg ratio    : {_avg_b:.2f}x → {_avg_a:.2f}x", flush=True)
        if _timing_audit_log:
            _too_long = sum(1 for _item in _timing_audit_log if _item.get("status") == "too_long")
            _too_short = sum(1 for _item in _timing_audit_log if _item.get("status") == "too_short")
            print(f"  timing too_long     : {_too_long}", flush=True)
            print(f"  timing too_short    : {_too_short}", flush=True)
        if use_turbo:
            print(f"  turbo  MTL preroute : {_turbo_fallback_preroute}", flush=True)
            print(f"  turbo  MTL QA retry : {_turbo_fallback_qa}", flush=True)
        print(f"  assembly ok_fit     : {_ok_fit}", flush=True)
        print(f"  assembly atempo     : {_atempo_used}", flush=True)
        print(f"  assembly TRIM ⚠     : {_trim_used}  {'✓ cieľ splnený' if _trim_used == 0 else '← znížiť'}", flush=True)
        print(f"{'═'*55}\n", flush=True)
    except Exception as _me:
        print(f"[METRICS] Chyba pri výpočte metrík: {_me}", flush=True)

    # --- Post-processing ---
    if not args.no_denoise:
        apply_noise_reduction(out_wav, preset=args.denoise_preset)
    apply_tts_master_chain(out_wav, denoise=not args.no_denoise)
    print(f"[POLISH] Applied tone-shape+compressor+loudnorm to {out_wav.name}", flush=True)
    if args.voice_eq:
        apply_voice_eq(out_wav, out_wav)
        print("[EQ] Voice EQ applied", flush=True)

    apply_speech_gain(out_wav, speech_gain=args.speech_gain, auto_gain=args.auto_speech_gain)
    print(f"[GAIN] Applied speech gain x{args.speech_gain:.2f} auto={args.auto_speech_gain} to {out_wav}", flush=True)

    # --- Odstrániť dlhé pauzy z finálnej zostavy ---
    # WARNING: on timeline-synced video dubbing this would shorten the whole track
    # and break A/V sync. Per-chunk silence repair already ran before assembly.
    if _timeline_synced:
        print("[SILENCE] Skipping post-assembly silence repair for timeline-synced dub", flush=True)
    else:
        _max_pause_final = getattr(args, "max_pause_gap_s", 0.0) or 0.0
        _silence_cap = max(_max_pause_final, 1.0)  # min 1s, inak by sme rezali medzi slovami
        repair_silence_gaps(out_wav, max_silence_s=_silence_cap)
        print(f"[SILENCE] Post-assembly silence repair (max {_silence_cap:.1f}s) done", flush=True)

    # --- Oprava single-frame glitchov (assembly timing artefakty) ---
    from audio_pipeline import repair_click_glitches
    repair_click_glitches(out_wav)

    # --- SRT alignment with WhisperX (speech-only, before music is mixed in) ---
    if getattr(args, "srt_align", False):
        try:
            _align_srt_whisperx(out_wav, args.tgt_lang, device, args.whisper_root, args.whisper_model)
        except Exception as _srt_exc:
            print(f"[SRT-ALIGN] Skipped (error): {_srt_exc}", flush=True)

    # --- Mix original audio as background (pyvideotrans-style ambience fill) ---
    backaudio_vol = getattr(args, "backaudio_volume", 0.0)
    # Skip when keep_music was requested and music was extracted — mixing full original (with speech)
    # on top of a clean vocal-separated track would cause overlap.
    # For content_type=sql, keep_music is forced to False above, so backaudio always applies.
    if backaudio_vol > 0.0 and not args.keep_music and not (music_wav and music_wav.exists()):
        mix_with_backaudio(out_wav, input_path, out_wav, volume=backaudio_vol)

    # --- Mix music ---
    if music_wav and music_wav.exists():
        mix_audio_with_music(out_wav, music_wav, out_wav, music_volume=args.music_volume)
        print(f"[MUSIC] Mixed background music (vol={args.music_volume})", flush=True)

    print(f"[DONE] CHATTERBOX audio: {out_wav}", flush=True)


if __name__ == "__main__":
    main()
