"""
Audio assembly, post-processing, music separation & mixing module.
Timeline assembly, time-stretch, mastering, Demucs, sidechain mix.
"""

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from stt import get_audio_duration
from config import TTS_SR, FFMPEG_TIMEOUT
from paths import PATHS


# ---------------------------------------------------------------------------
# FFmpeg audio utilities
# ---------------------------------------------------------------------------

def atempo_filter_for_ratio(ratio: float) -> str:
    # atempo supports 0.5..2.0; chain if needed
    if ratio <= 0:
        return "atempo=1.0"
    parts = []
    while ratio > 2.0:
        parts.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        parts.append("atempo=0.5")
        ratio /= 0.5
    parts.append(f"atempo={ratio:.6f}")
    return ",".join(parts)


def time_stretch_librosa(input_path: Path, output_path: Path, speed_ratio: float) -> bool:
    """Pitch-preserving time-stretch via librosa phase vocoder (inspired by pyvideotrans/_rate.py)."""
    try:
        import librosa
        y, sr = sf.read(str(input_path), dtype="float32")
        mono = y[:, 0] if y.ndim > 1 else y
        # n_fft=4096 → 171ms window at 24kHz: better spectral resolution, less metallic artifacts
        stretched = librosa.effects.time_stretch(mono, rate=speed_ratio, n_fft=4096, hop_length=1024)
        sf.write(str(output_path), stretched, sr, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"  [STRETCH] librosa failed ({e}), falling back to atempo", flush=True)
        return False


def time_stretch_rubberband(input_path: Path, output_path: Path, speed_ratio: float) -> bool:
    """FFmpeg rubberband filter: PSOLA-based, lepšia kvalita reči pri >30% stretch.
    Pitch-preserving, žiadny echo ako librosa phase vocoder.
    Vyžaduje: ffmpeg compiled with --enable-librubberband.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter:a", f"rubberband=tempo={speed_ratio:.6f}:pitch=1.0:transients=crisp",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    return result.returncode == 0


def time_stretch_ffmpeg(input_path: Path, output_path: Path, speed_ratio: float) -> bool:
    # Skús najprv rubberband (PSOLA, lepšia kvalita), fallback na atempo.
    # rubberband: pitch-preserving, no phase smearing — lepšie pre reč pri >30% stretch.
    if abs(speed_ratio - 1.0) > 0.30:
        if time_stretch_rubberband(input_path, output_path, speed_ratio):
            return True
        print(f"  [STRETCH] rubberband failed, falling back to atempo", flush=True)

    atempo_filter = atempo_filter_for_ratio(speed_ratio)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter:a", atempo_filter,
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    return result.returncode == 0


def trim_edge_silence_ffmpeg(
    input_path: Path,
    output_path: Path,
    threshold_db: float = -48.0,
    keep_s: float = 0.08,
    trim_tail: bool = True,
    tail_threshold_db: float = -45.0,
    tail_keep_s: float = 0.10,
) -> bool:
    """
    Trim leading (and optionally trailing) silence from a TTS chunk.

    trim_tail=True removes the Chatterbox trailing hiss/silence that follows
    every utterance (~0.15-0.35s of near-silence at the end of each chunk).
    This prevents stuttery gaps between segments in the assembled timeline.
    """
    # Leading trim only (original behaviour)
    af_parts = [
        f"silenceremove=start_periods=1"
        f":start_silence={keep_s:.3f}"
        f":start_threshold={threshold_db:.1f}dB"
    ]
    # Trailing trim — reverse → trim leading → reverse back (FFmpeg idiom)
    if trim_tail:
        af_parts += [
            "areverse",
            f"silenceremove=start_periods=1"
            f":start_silence={tail_keep_s:.3f}"
            f":start_threshold={tail_threshold_db:.1f}dB",
            "areverse",
        ]
    af = ",".join(af_parts)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", af,
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    return result.returncode == 0 and output_path.exists()


def calculate_safe_tempo(tts_duration: float, target_duration: float, stretch_min: float = 0.70, stretch_max: float = 1.60) -> tuple[float, str]:
    if target_duration <= 0 or tts_duration <= 0:
        return 1.0, "skip"
    tempo = tts_duration / target_duration
    if 0.90 <= tempo <= 1.10:
        return tempo, "safe"
    elif 0.85 <= tempo <= 1.20:
        return tempo, "ok"
    elif stretch_min <= tempo <= stretch_max:
        return tempo, "risky"
    else:
        clamped = max(stretch_min, min(stretch_max, tempo))
        return clamped, "clamped"


# ---------------------------------------------------------------------------
# WAV concatenation
# ---------------------------------------------------------------------------

def concat_wavs(wavs: list[Path], out_path: Path) -> None:
    audio = []
    sr = None
    for p in wavs:
        y, this_sr = sf.read(str(p))
        if y.ndim > 1:
            y = y[:, 0]
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            y = np.interp(
                np.linspace(0, len(y), int(len(y) * sr / this_sr), endpoint=False),
                np.arange(len(y)),
                y,
            )
        audio.append(y)
    if not audio:
        raise RuntimeError("No audio chunks to concatenate.")
    out = np.concatenate(audio)
    sf.write(str(out_path), out, sr)


def concat_wavs_crossfade(
    wavs: list[Path],
    out_path: Path,
    crossfade_ms: int = 60,
) -> None:
    """Concatenate WAV files with short cross-fade to avoid hard cuts."""
    segments: list[np.ndarray] = []
    sr = None
    for p in wavs:
        y, this_sr = sf.read(str(p))
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(np.float32)
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            y = np.interp(
                np.linspace(0, len(y), int(len(y) * sr / this_sr), endpoint=False),
                np.arange(len(y)),
                y,
            ).astype(np.float32)
        segments.append(y)
    if not segments:
        raise RuntimeError("No audio chunks to concatenate.")
    if len(segments) == 1 or crossfade_ms <= 0:
        sf.write(str(out_path), np.concatenate(segments), sr)
        return

    xfade_samples = int(sr * crossfade_ms / 1000.0)
    result = segments[0]
    for seg in segments[1:]:
        overlap = min(xfade_samples, len(result), len(seg))
        if overlap < 4:
            result = np.concatenate([result, seg])
            continue
        fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        tail = result[-overlap:] * fade_out
        head = seg[:overlap] * fade_in
        mixed = tail + head
        result = np.concatenate([result[:-overlap], mixed, seg[overlap:]])
    sf.write(str(out_path), result, sr)


# ---------------------------------------------------------------------------
# Timeline assembly methods
# ---------------------------------------------------------------------------

def assemble_timed_audio(
    segments: list[dict],
    seg_audio: list[Path],
    out_path: Path,
    total_dur: float,
    sr: int = TTS_SR,
) -> None:
    concat_list = out_path.parent / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for seg_path in seg_audio:
            abs_path = Path(seg_path).resolve()
            safe_path = str(abs_path).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")
    temp_concat = out_path.parent / "temp_concat.wav"
    cmd_concat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:a", "pcm_s16le",
        str(temp_concat)
    ]
    subprocess.run(cmd_concat, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
    combined_duration = get_audio_duration(temp_concat)
    print(f"  Combined audio: {combined_duration:.2f}s, Target: {total_dur:.2f}s")
    if combined_duration > total_dur * 1.05:
        speed_ratio = combined_duration / total_dur
        speed_ratio = min(2.0, speed_ratio)
        print(f"  Speeding up: {speed_ratio:.2f}x using FFmpeg atempo")
        atempo_filter = atempo_filter_for_ratio(speed_ratio)
        cmd_stretch = [
            "ffmpeg", "-y",
            "-i", str(temp_concat),
            "-filter:a", atempo_filter,
            "-c:a", "pcm_s16le",
            str(out_path)
        ]
        subprocess.run(cmd_stretch, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
    else:
        if combined_duration < total_dur:
            print(f"  Audio shorter than video - padding with silence")
            cmd_pad = [
                "ffmpeg", "-y",
                "-i", str(temp_concat),
                "-af", f"apad=whole_dur={total_dur}",
                "-c:a", "pcm_s16le",
                str(out_path)
            ]
            subprocess.run(cmd_pad, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        else:
            shutil.copy(temp_concat, out_path)
    try:
        concat_list.unlink()
        temp_concat.unlink()
    except Exception:
        pass


def assemble_synced_audio(
    segments: list[dict],
    seg_audio: list[Path],
    out_path: Path,
    total_dur: float,
    sr: int = TTS_SR,
) -> None:
    print(f"  Syncing to original timing ({len(segments)} segments)...")
    total_samples = int(total_dur * sr)
    output_audio = np.zeros(total_samples, dtype=np.float32)
    temp_dir = out_path.parent / "temp_stretched"
    temp_dir.mkdir(exist_ok=True)
    for i, seg_path in enumerate(seg_audio):
        start_time = segments[i]["start"]
        end_time = segments[i]["end"]
        available_time = end_time - start_time
        if available_time <= 0.1:
            continue
        seg_duration = get_audio_duration(seg_path)
        start_sample = int(start_time * sr)
        actual_path = seg_path
        if seg_duration > available_time * 1.05:
            speed_ratio = seg_duration / available_time
            if speed_ratio <= 2.5:
                stretched_path = temp_dir / f"stretched_{i:04d}.wav"
                if time_stretch_ffmpeg(seg_path, stretched_path, speed_ratio):
                    actual_path = stretched_path
                    print(f"    Segment {i+1}: {seg_duration:.2f}s -> {available_time:.2f}s ({speed_ratio:.2f}x)")
        seg_data, seg_sr = sf.read(str(actual_path))
        if seg_data.ndim > 1:
            seg_data = seg_data.mean(axis=1)
        seg_data = seg_data.astype(np.float32)
        if seg_sr != sr:
            target_len = int(len(seg_data) * sr / seg_sr)
            seg_data = np.interp(
                np.linspace(0, len(seg_data), target_len, endpoint=False),
                np.arange(len(seg_data)),
                seg_data,
            ).astype(np.float32)
        available_samples = int(available_time * sr)
        seg_len = min(len(seg_data), available_samples, total_samples - start_sample)
        if seg_len > 0 and start_sample < total_samples:
            output_audio[start_sample:start_sample + seg_len] = seg_data[:seg_len]
    sf.write(str(out_path), output_audio, sr)
    print(f"  Synced audio saved: {out_path}")
    shutil.rmtree(temp_dir, ignore_errors=True)


def assemble_timeline_ffmpeg(
    segments: list[dict],
    seg_audio: list[Path],
    out_path: Path,
    total_dur: float,
    safe_stretch: bool = True,
    stretch_min: float = 0.70,
    stretch_max: float = 1.60,
    base_tempo: float = 1.0,
    prefer_silence_trim: bool = False,
    speedup_headroom: float = 1.35,
    crossfade_dur: float = 0.18,
    gap_collapse: bool = False,
    fill_slot: bool = False,
) -> None:
    """
    fill_slot=True: time-stretch every TTS chunk to exactly match the slot duration
    in BOTH directions — slow down when TTS is shorter, speed up when longer.
    Max slowdown is capped at stretch_min (default 0.70 = 1.43× slower).
    """
    if len(segments) != len(seg_audio):
        raise ValueError(f"Segment count ({len(segments)}) != audio file count ({len(seg_audio)})")
    print(f"[TIMELINE] Assembling {len(segments)} segments with FFmpeg adelay/amix", flush=True)
    print(f"[TIMELINE] Stretch limits: {stretch_min:.0%} - {stretch_max:.0%}", flush=True)
    temp_dir = out_path.parent / "temp_timeline"
    temp_dir.mkdir(exist_ok=True)

    processed_files = []
    stretch_stats = {"safe": 0, "ok": 0, "risky": 0, "clamped": 0, "skip": 0}

    for i, (seg, audio_path) in enumerate(zip(segments, seg_audio)):
        target_dur = seg["end"] - seg["start"]
        source_audio_path = audio_path
        tts_dur = get_audio_duration(source_audio_path)
        if i + 1 < len(segments):
            next_start = segments[i + 1]["start"]
            max_allowed_dur = (next_start - seg["start"]) - 0.08
        else:
            max_allowed_dur = total_dur - seg["start"]
        max_allowed_dur = max(target_dur, max_allowed_dur)

        # Gap-collapse: if TTS fits in gap (up to next segment), no stretch needed.
        # Only stretch when audio overflows even the silence gap (pyvideotrans strategy).
        if gap_collapse and tts_dur <= max_allowed_dur:
            tempo, status = 1.0, "skip"
        elif fill_slot and target_dur > 0.1:
            # Fill mode: stretch in BOTH directions to exactly fill the slot.
            # Slow down when TTS is shorter, speed up when longer.
            # Clamped to [stretch_min, stretch_max] to stay audible.
            raw_tempo = tts_dur / target_dur
            tempo = max(stretch_min, min(stretch_max, raw_tempo))
            if abs(tempo - 1.0) < 0.02:
                status = "skip"
                tempo = 1.0
            elif raw_tempo < 1.0:
                status = "fill"
            else:
                status = "risky" if raw_tempo > stretch_max else "ok"
        else:
            collapse_target = max_allowed_dur if gap_collapse else target_dur
            tempo, status = calculate_safe_tempo(tts_dur, collapse_target, stretch_min, stretch_max)
            if status != "skip":
                tempo = max(stretch_min, min(stretch_max, tempo * base_tempo))
        stretch_stats[status] += 1

        stretched_dur = tts_dur / tempo if tempo > 0 else tts_dur
        overflow = stretched_dur - max_allowed_dur

        if prefer_silence_trim and overflow > 0.25:
            nosil_path = temp_dir / f"seg_{i:04d}_nosil.wav"
            if trim_edge_silence_ffmpeg(source_audio_path, nosil_path):
                nosil_dur = get_audio_duration(nosil_path)
                saved = tts_dur - nosil_dur
                if 0.08 <= saved <= 0.80 and nosil_dur >= 0.35:
                    source_audio_path = nosil_path
                    tts_dur = nosil_dur
                    tempo, status = calculate_safe_tempo(tts_dur, target_dur, stretch_min, stretch_max)
                    if status != "skip":
                        tempo = max(stretch_min, min(stretch_max, tempo * base_tempo))
                    stretched_dur = tts_dur / tempo if tempo > 0 else tts_dur
                    overflow = stretched_dur - max_allowed_dur
                    print(f"    Seg {i+1}: shortened leading silence {saved:.2f}s before fit", flush=True)

        if overflow > 0 and tts_dur > 0 and max_allowed_dur > 0:
            required_tempo = tts_dur / max_allowed_dur
            soft_max_tempo = max(stretch_max, stretch_max * max(1.0, speedup_headroom))
            if required_tempo <= soft_max_tempo and required_tempo > tempo:
                print(f"    Seg {i+1}: prefer speed over trim, tempo {tempo:.2f} -> {required_tempo:.2f}", flush=True)
                tempo = required_tempo
                stretched_dur = tts_dur / tempo
                overflow = stretched_dur - max_allowed_dur

        needs_trim = overflow > 0.60   # zvýšená tolerancia — menej agresívne odstrihnutie

        if status == "skip" or abs(tempo - 1.0) < 0.02:
            if needs_trim:
                trimmed_path = temp_dir / f"seg_{i:04d}_trimmed.wav"
                fade_dur = min(0.55, max(0.35, overflow * 0.6))  # dlhší fade = prirodzenejší koniec
                fade_start = max(0.0, max_allowed_dur - fade_dur)
                cmd = [
                    "ffmpeg", "-y", "-i", str(source_audio_path),
                    "-filter:a", f"afade=t=in:st=0:d={crossfade_dur:.3f},atrim=end={max_allowed_dur:.3f},afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}",
                    "-c:a", "pcm_s16le", str(trimmed_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
                processed_files.append(trimmed_path if result.returncode == 0 else audio_path)
            else:
                faded_path = temp_dir / f"seg_{i:04d}_faded.wav"
                natural_fade = max(crossfade_dur, 0.28)
                fade_out_start = max(0.0, tts_dur - natural_fade)
                cmd = [
                    "ffmpeg", "-y", "-i", str(source_audio_path),
                    "-filter:a", f"afade=t=in:st=0:d={crossfade_dur:.3f},afade=t=out:st={fade_out_start:.3f}:d={natural_fade:.3f}",
                    "-c:a", "pcm_s16le", str(faded_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
                processed_files.append(faded_path if result.returncode == 0 else source_audio_path)
            continue

        stretched_path = temp_dir / f"seg_{i:04d}_stretched.wav"
        if safe_stretch and status in ("risky", "clamped"):
            print(f"    Seg {i+1}: tempo={tempo:.2f} ({status}) - may sound unnatural", flush=True)

        atempo_filter = atempo_filter_for_ratio(tempo)
        if needs_trim:
            fade_dur = min(0.55, max(0.35, overflow * 0.6))
            fade_start = max(0.0, max_allowed_dur - fade_dur)
            audio_filter = f"{atempo_filter},afade=t=in:st=0:d={crossfade_dur:.3f},atrim=end={max_allowed_dur:.3f},afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}"
        else:
            natural_fade = max(crossfade_dur, 0.28)
            fade_out_start = max(0.0, stretched_dur - natural_fade)
            audio_filter = f"{atempo_filter},afade=t=in:st=0:d={crossfade_dur:.3f},afade=t=out:st={fade_out_start:.3f}:d={natural_fade:.3f}"

        cmd = [
            "ffmpeg", "-y", "-i", str(source_audio_path),
            "-filter:a", audio_filter,
            "-c:a", "pcm_s16le",
            str(stretched_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        if result.returncode == 0 and stretched_path.exists():
            processed_files.append(stretched_path)
        else:
            print(f"    Seg {i+1}: stretch failed, using original", flush=True)
            processed_files.append(source_audio_path)

    print(f"[TIMELINE] Stretch stats: {stretch_stats}", flush=True)

    if len(processed_files) <= 32:
        _assemble_with_filter_complex(segments, processed_files, out_path, total_dur, crossfade_dur)
    else:
        _assemble_batched(segments, processed_files, out_path, total_dur, temp_dir, crossfade_dur)

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def _assemble_with_filter_complex(
    segments: list[dict],
    audio_files: list[Path],
    out_path: Path,
    total_dur: float,
    crossfade_dur: float = 0.08,
) -> None:
    missing_files = [str(f) for f in audio_files if not f.exists()]
    if missing_files:
        print(f"[TIMELINE] ERROR: Missing audio files: {missing_files[:5]}", flush=True)
        raise RuntimeError(f"Missing {len(missing_files)} audio files")
    inputs = []
    for audio_path in audio_files:
        inputs.extend(["-i", str(audio_path)])
    filters = []
    amix_inputs = []
    overlap_ms = crossfade_dur * 1000.0
    for i, seg in enumerate(segments):
        start_ms = max(0.0, seg["start"] * 1000.0 - (overlap_ms if i > 0 else 0.0))
        filters.append(f"[{i}]adelay={start_ms:.3f}|{start_ms:.3f}[d{i}]")
        amix_inputs.append(f"[d{i}]")
    inputs.extend(["-f", "lavfi", "-i", f"anullsrc=r={TTS_SR}:cl=mono:d={total_dur}"])
    base_idx = len(audio_files)
    all_inputs = f"[{base_idx}]" + "".join(amix_inputs)
    filters.append(f"{all_inputs}amix=inputs={len(segments) + 1}:duration=first:dropout_transition=0:normalize=0[out]")
    filter_complex = ";".join(filters)
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        str(out_path)
    ]
    print(f"[TIMELINE] Running FFmpeg with {len(segments)} inputs...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if result.returncode != 0:
        error_log = out_path.parent / "logs" / "ffmpeg_error.log"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log, "w") as f:
            f.write("=== FFMPEG COMMAND ===\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("=== STDERR ===\n")
            f.write(result.stderr + "\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout + "\n")
        print(f"[TIMELINE] FFmpeg failed! Full log: {error_log}", flush=True)
        print(f"[TIMELINE] Command length: {len(' '.join(cmd))} chars", flush=True)
        print(f"[TIMELINE] Filter complex length: {len(filter_complex)} chars", flush=True)
        stderr_lines = result.stderr.split('\n')
        print(f"[TIMELINE] Last 20 stderr lines:", flush=True)
        for line in stderr_lines[-20:]:
            if line.strip():
                print(f"  {line}", flush=True)
        raise RuntimeError(f"FFmpeg failed. See {error_log}")
    print(f"[TIMELINE] Assembled: {out_path}", flush=True)


def _assemble_batched(
    segments: list[dict],
    audio_files: list[Path],
    out_path: Path,
    total_dur: float,
    temp_dir: Path,
    crossfade_dur: float = 0.08,
    batch_size: int = 20,
) -> None:
    print(f"[TIMELINE] Using batched assembly ({len(segments)} segments)", flush=True)
    batch_outputs = []
    for batch_idx in range(0, len(segments), batch_size):
        batch_end = min(batch_idx + batch_size, len(segments))
        batch_segs = segments[batch_idx:batch_end]
        batch_files = audio_files[batch_idx:batch_end]
        batch_out = temp_dir / f"batch_{batch_idx:04d}.wav"
        batch_start_time = batch_segs[0]["start"]
        batch_end_time = batch_segs[-1]["end"]
        batch_dur = batch_end_time - batch_start_time + 0.5
        adjusted_segs = [
            {"start": s["start"] - batch_start_time, "end": s["end"] - batch_start_time}
            for s in batch_segs
        ]
        _assemble_with_filter_complex(adjusted_segs, batch_files, batch_out, batch_dur, crossfade_dur)
        batch_outputs.append({"path": batch_out, "start": batch_start_time})

    print(f"[TIMELINE] Combining {len(batch_outputs)} batches...", flush=True)
    sr = TTS_SR
    total_samples = int(total_dur * sr)
    output_audio = np.zeros(total_samples, dtype=np.float32)
    for batch in batch_outputs:
        batch_data, batch_sr = sf.read(str(batch["path"]))
        if batch_data.ndim > 1:
            batch_data = batch_data.mean(axis=1)
        if batch_sr != sr:
            target_len = int(len(batch_data) * sr / batch_sr)
            batch_data = np.interp(
                np.linspace(0, len(batch_data), target_len, endpoint=False),
                np.arange(len(batch_data)),
                batch_data,
            )
        start_sample = int(batch["start"] * sr)
        end_sample = min(start_sample + len(batch_data), total_samples)
        copy_len = end_sample - start_sample
        if copy_len > 0:
            output_audio[start_sample:end_sample] += batch_data[:copy_len]
    sf.write(str(out_path), output_audio.astype(np.float32), sr)
    print(f"[TIMELINE] Final output: {out_path}", flush=True)
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# pyvideotrans-style concat assembly (gap-collapse + pydub concat + FFmpeg atempo stretch)
# ---------------------------------------------------------------------------

def assemble_concat_style(
    segments: list[dict],
    seg_audio: list[Path],
    out_path: Path,
    total_dur: float,
    stretch_max: float = 1.60,
    base_tempo: float = 1.0,
    min_fill_tempo: float = 0.70,
) -> None:
    """
    Clean assembly matching pyvideotrans SpeedRate strategy:
    - Gap collapse: each segment slot = seg.start … next_seg.start
    - TTS fits slot → pad tail with silence
    - TTS overflows → FFmpeg atempo time-stretch (or hard trim)
    - Sequential pydub concat (no adelay/amix mixing artifacts)
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError(f"assemble_concat_style requires pydub: {e}")

    SR = TTS_SR
    print(f"[CONCAT] Assembling {len(segments)} segments (pyvideotrans concat style)", flush=True)

    audio_parts: list[AudioSegment] = []
    timeline_ms: int = 0
    _stats_stretched = 0
    _stats_clamped = 0

    for i, (seg, wav_path) in enumerate(zip(segments, seg_audio)):
        start_ms = int(seg["start"] * 1000)
        seg_text = re.sub(r"\s+", " ", str(seg.get("text") or "").strip())
        seg_core = seg_text.rstrip(".!?…,:;").strip()
        seg_word_count = len(seg_core.split()) if seg_core else 0

        # Fill gap before this segment
        gap_ms = start_ms - timeline_ms
        if gap_ms > 10:
            audio_parts.append(AudioSegment.silent(duration=gap_ms, frame_rate=SR))
            timeline_ms += gap_ms

        # Slot end: next segment's start (gap collapse) or total_dur for last
        if i < len(segments) - 1:
            slot_end_ms = int(segments[i + 1]["start"] * 1000)
        else:
            slot_end_ms = int(total_dur * 1000)
        slot_dur_ms = max(10, slot_end_ms - start_ms)
        short_reply = bool(seg_core) and seg_word_count <= 2 and len(seg_core) <= 10 and slot_dur_ms <= 1800

        if wav_path and Path(wav_path).exists():
            try:
                tts = (AudioSegment.from_file(str(wav_path))
                       .set_frame_rate(SR).set_channels(1).set_sample_width(2))
            except Exception as e:
                print(f"  Seg {i+1}: failed to load audio ({e}), inserting silence", flush=True)
                tts = AudioSegment.silent(duration=slot_dur_ms, frame_rate=SR)

            tts_dur_ms = len(tts)
            # Fade-OUT prevents click at TTS→silence boundary.
            # No fade-IN: trim_and_smooth_chunk already ensures each chunk starts at zero.
            _fade_out_ms = min(40, tts_dur_ms // 8)

            if tts_dur_ms <= slot_dur_ms:
                # Vypočítaj koľko ticha by ostalo
                tail_ms = slot_dur_ms - tts_dur_ms
                _stretch_applied: float | None = None
                _stretch_clamped = False
                # Ak je ticho väčšie ako 1.2s a min_fill_tempo > 0 → jemne spomaľ reč
                _short_reply_fill = short_reply and tail_ms > 220 and tts_dur_ms < slot_dur_ms * 0.82
                if (tail_ms > 1200 or _short_reply_fill) and min_fill_tempo > 0.0:
                    _fill_floor = min_fill_tempo
                    if _short_reply_fill:
                        _fill_floor = min(min_fill_tempo, 0.62)
                    slow_ratio = tts_dur_ms / slot_dur_ms  # ideálne tempo (< 1.0)
                    effective_tempo = max(slow_ratio, _fill_floor)
                    _stretch_clamped = (effective_tempo > slow_ratio)  # narazili sme na limit
                    if effective_tempo < 0.98:
                        import tempfile as _tf
                        try:
                            with _tf.NamedTemporaryFile(suffix="_slow_in.wav", delete=False) as _fi, \
                                 _tf.NamedTemporaryFile(suffix="_slow_out.wav", delete=False) as _fo:
                                _in = Path(_fi.name); _out = Path(_fo.name)
                            tts.export(str(_in), format="wav")
                            import subprocess as _sp2
                            _atempo_val = round(effective_tempo, 4)
                            _sp2.run(
                                ["ffmpeg", "-y", "-i", str(_in),
                                 "-af", f"atempo={_atempo_val}",
                                 "-ar", str(SR), str(_out)],
                                capture_output=True, timeout=30,
                            )
                            if _out.stat().st_size > 1000:
                                from pydub import AudioSegment as _AS
                                tts = _AS.from_file(str(_out)).set_frame_rate(SR).set_channels(1).set_sample_width(2)
                                tail_ms = slot_dur_ms - len(tts)
                                _stretch_applied = _atempo_val
                            _in.unlink(missing_ok=True); _out.unlink(missing_ok=True)
                        except Exception as _e:
                            print(f"  Seg {i+1}: slow-stretch failed ({_e}), using original", flush=True)
                            tail_ms = slot_dur_ms - tts_dur_ms
                # Fits: fade-out to prevent chunk→silence click, then pad silence
                _fade_out_ms = min(40, len(tts) // 8)
                tts = tts.fade_out(_fade_out_ms)
                audio_parts.append(tts)
                if tail_ms > 10:
                    audio_parts.append(AudioSegment.silent(duration=tail_ms, frame_rate=SR))
                # Log
                if _stretch_applied is not None:
                    _stats_stretched += 1
                    if _stretch_clamped:
                        _stats_clamped += 1
                    _clamp_tag = " (clamped)" if _stretch_clamped else ""
                    print(
                        f"  Seg {i+1}: stretch_applied={_stretch_applied:.2f}{_clamp_tag}"
                        f"{' [short-reply]' if short_reply else ''}"
                        f" → {len(tts)}ms (+{tail_ms}ms silence)",
                        flush=True,
                    )
                else:
                    print(f"  Seg {i+1}: {tts_dur_ms}ms fits in {slot_dur_ms}ms slot (+{tail_ms}ms silence)", flush=True)
            else:
                ratio = tts_dur_ms / slot_dur_ms  # speed-up factor
                effective_ratio = ratio * base_tempo
                if effective_ratio > stretch_max:
                    # Too much: hard trim with fade-out
                    fade_ms = min(320, slot_dur_ms // 4)
                    tts = tts[:slot_dur_ms].fade_out(fade_ms)
                    audio_parts.append(tts)
                    print(f"  Seg {i+1}: TRIMMED {tts_dur_ms}ms → {slot_dur_ms}ms (ratio {ratio:.2f}x > max)", flush=True)
                else:
                    # FFmpeg atempo time-stretch (better for speech than phase vocoder)
                    import tempfile
                    try:
                        with tempfile.NamedTemporaryFile(suffix="_in.wav", delete=False) as tf_in, \
                             tempfile.NamedTemporaryFile(suffix="_out.wav", delete=False) as tf_out:
                            in_path = Path(tf_in.name)
                            out_stretch_path = Path(tf_out.name)
                        tts.export(str(in_path), format="wav")
                        time_stretch_ffmpeg(in_path, out_stretch_path, effective_ratio)
                        stretched = (AudioSegment.from_file(str(out_stretch_path))
                                     .set_frame_rate(SR).set_channels(1).set_sample_width(2))
                        in_path.unlink(missing_ok=True)
                        out_stretch_path.unlink(missing_ok=True)
                        max_ms = slot_dur_ms
                        if len(stretched) > max_ms:
                            stretched = stretched[:max_ms]
                        fo = min(40, len(stretched) // 8)
                        stretched = stretched.fade_out(fo)
                        audio_parts.append(stretched)
                        tail_ms = slot_dur_ms - len(stretched)
                        if tail_ms > 10:
                            audio_parts.append(AudioSegment.silent(duration=tail_ms, frame_rate=SR))
                        print(f"  Seg {i+1}: stretched {tts_dur_ms}ms → {len(stretched)}ms (x{effective_ratio:.2f})", flush=True)
                    except Exception as e:
                        print(f"  Seg {i+1}: atempo stretch failed ({e}), using trimmed", flush=True)
                        tts = tts[:slot_dur_ms].fade_out(min(320, slot_dur_ms // 4))
                        audio_parts.append(tts)
        else:
            # No audio: silence for full slot
            audio_parts.append(AudioSegment.silent(duration=slot_dur_ms, frame_rate=SR))
            print(f"  Seg {i+1}: missing audio → {slot_dur_ms}ms silence", flush=True)

        timeline_ms = slot_end_ms

    # Pad to total duration
    total_ms = int(total_dur * 1000)
    if timeline_ms < total_ms:
        audio_parts.append(AudioSegment.silent(duration=total_ms - timeline_ms, frame_rate=SR))

    # Concat and export
    combined = AudioSegment.empty()
    for part in audio_parts:
        combined = combined + part
    combined.export(str(out_path), format="wav")
    _clamp_note = f"  clamped_at_{min_fill_tempo:.2f}={_stats_clamped}" if _stats_clamped else ""
    print(
        f"[CONCAT] Done: {len(combined) / 1000:.1f}s output"
        f"  slow_stretched={_stats_stretched}{_clamp_note}",
        flush=True,
    )


def mix_with_backaudio(tts_wav: Path, orig_video: Path, out_path: Path, volume: float) -> None:
    """
    Mix TTS audio with original video audio as low-volume background.
    pyvideotrans-style: fills silence gaps with ambience/music from original.
    Resamples original to TTS sample rate (mono). Output replaces out_path.
    """
    tmp = out_path.with_suffix(".backaudio_tmp.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tts_wav),
        "-i", str(orig_video),
        "-filter_complex",
        (
            f"[1:a]aresample={TTS_SR},aformat=channel_layouts=mono,"
            f"volume={volume:.3f}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[out]"
        ),
        "-map", "[out]",
        "-ar", str(TTS_SR), "-ac", "1",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
    tmp.replace(out_path)
    print(f"[BACKAUDIO] Mixed original audio at vol={volume:.2f} into {out_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Music / source separation (Demucs + MDX-Net)
# ---------------------------------------------------------------------------

# Best vocal removal model (SDR 12.1 vocals / 16.3 instrumental)
MDX_DEFAULT_MODEL = "model_bs_roformer_ep_368_sdr_12.9628.ckpt"
MDX_MODELS_DIR    = Path(PATHS.models_root) / "mdx_net"


def extract_vocals_mdx(
    audio_path: Path,
    output_dir: Path,
    model_name: str = MDX_DEFAULT_MODEL,
    models_dir: Path = MDX_MODELS_DIR,
) -> Path:
    """Separate vocals using audio-separator (MDX-Net / UVR5 models).

    Returns path to the instrumental (no-vocals) WAV file.
    Model is downloaded automatically on first use to *models_dir*.
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        raise RuntimeError(
            "audio-separator not installed. "
            "Run: pip install 'audio-separator[gpu]'"
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    import logging
    print(f"[MDX] Separating vocals with {model_name}...", flush=True)
    print(f"[MDX] Input:  {audio_path}", flush=True)
    print(f"[MDX] Output: {output_dir}", flush=True)

    sep = Separator(
        log_level=logging.WARNING,
        model_file_dir=str(models_dir),
        output_dir=str(output_dir),
        output_format="WAV",
        normalization_threshold=0.9,
    )
    print(f"[MDX] Device: {getattr(sep, 'torch_device', 'auto')}", flush=True)
    sep.load_model(model_name)
    output_files = sep.separate(str(audio_path))

    # audio-separator returns [vocals_path, instrumental_path]
    # instrumental = no-vocals track
    if len(output_files) < 2:
        raise RuntimeError(f"[MDX] Expected 2 output files, got: {output_files}")

    # Find the instrumental (non-Vocals) file
    # audio-separator returns filenames only (not full paths), so prepend output_dir
    instrumental = None
    for f in output_files:
        fname = Path(f).name.lower()
        if "instrumental" in fname or "no_vocal" in fname or "_(no" in fname:
            instrumental = output_dir / Path(f).name
            break
    if instrumental is None:
        # fallback: second file is typically instrumental
        instrumental = output_dir / Path(output_files[1]).name

    if not instrumental.exists():
        raise RuntimeError(f"[MDX] Instrumental file not found: {instrumental}")

    print(f"[MDX] Instrumental extracted: {instrumental}", flush=True)
    return instrumental


def extract_music_demucs(audio_path: Path, output_dir: Path) -> Path:
    print(f"[DEMUCS] Separating music from vocals...", flush=True)
    print(f"[DEMUCS] Input: {audio_path}", flush=True)
    print(f"[DEMUCS] Output dir: {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", str(output_dir),
        str(audio_path)
    ]
    print(f"[DEMUCS] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[DEMUCS] stdout: {result.stdout}", flush=True)
        print(f"[DEMUCS] stderr: {result.stderr}", flush=True)
        raise RuntimeError(f"Demucs failed with code {result.returncode}")
    print(f"[DEMUCS] Demucs completed successfully", flush=True)
    stem_name = audio_path.stem
    no_vocals_path = output_dir / "htdemucs" / stem_name / "no_vocals.wav"
    print(f"[DEMUCS] Looking for: {no_vocals_path}", flush=True)
    if not no_vocals_path.exists():
        print(f"[DEMUCS] Not found, searching alternatives...", flush=True)
        for model_dir in output_dir.iterdir():
            print(f"[DEMUCS] Checking: {model_dir}", flush=True)
            if model_dir.is_dir():
                candidate = model_dir / stem_name / "no_vocals.wav"
                if candidate.exists():
                    no_vocals_path = candidate
                    break
                candidate2 = model_dir / "no_vocals.wav"
                if candidate2.exists():
                    no_vocals_path = candidate2
                    break
    if not no_vocals_path.exists():
        print(f"[DEMUCS] Contents of output_dir:", flush=True)
        for p in output_dir.rglob("*"):
            print(f"  {p}", flush=True)
        raise RuntimeError(f"Demucs output not found at {no_vocals_path}")
    print(f"[DEMUCS] Music extracted: {no_vocals_path}", flush=True)
    return no_vocals_path


def _cut_speech_band(signal: np.ndarray, sr: int) -> np.ndarray:
    """Parametric EQ: cut the speech intelligibility band (300 Hz – 3.5 kHz) by ~7-8 dB.

    BS-Roformer leaves residual voice bleed mostly in the 300 Hz–3.5 kHz range.
    A gentle peaking-EQ cut here reduces vocal bleed without destroying bass or air.
    Uses Audio EQ Cookbook second-order peaking filters via scipy.signal.sosfilt.
    """
    from scipy.signal import sosfilt

    def _peaking_sos(f0: float, Q: float, gain_db: float, fs: int) -> np.ndarray:
        """Audio EQ Cookbook peaking EQ — positive gain_db = boost, negative = cut."""
        A = 10 ** (gain_db / 40.0)          # sqrt of linear amplitude gain
        w0 = 2.0 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2.0 * Q)
        b0 =  1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 =  1 - alpha * A
        a0 =  1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 =  1 - alpha / A
        return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

    result = signal.copy()
    # Two overlapping peaking filters shaping a broad speech-band cut:
    #   Band 1: centre 700 Hz, Q=0.6, -7 dB  — fundamentals + low formants
    #   Band 2: centre 2500 Hz, Q=0.7, -6 dB — high formants / presence
    for fc, Q, gain_db in [(700, 0.6, -7.0), (2500, 0.7, -6.0)]:
        if fc >= sr / 2:
            continue
        sos = _peaking_sos(fc, Q, gain_db, sr)
        result = sosfilt(sos, result).astype(np.float32)
    return result


def mix_audio_with_music(speech_path: Path, music_path: Path, output_path: Path, music_volume: float = 0.25) -> None:
    print(f"[MIX] Mixing speech with music (music volume: {music_volume})...", flush=True)
    speech, speech_sr = sf.read(str(speech_path))
    music, music_sr = sf.read(str(music_path))
    print(f"[MIX] Speech: {len(speech)} samples, {speech_sr}Hz, peak={np.abs(speech).max():.3f}", flush=True)
    print(f"[MIX] Music: {len(music)} samples, {music_sr}Hz, peak={np.abs(music).max():.3f}", flush=True)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    # Keep music stereo if available; store original channels count for output
    music_stereo = music.ndim > 1 and music.shape[1] >= 2
    if not music_stereo and music.ndim > 1:
        music = music.mean(axis=1)
    # Upsample SPEECH to music_sr to preserve full music frequency range.
    # Downsampling music to speech_sr (24kHz) would kill all content above 12kHz.
    out_sr = music_sr if music_sr > speech_sr else speech_sr
    if speech_sr != out_sr:
        target_len = int(len(speech) * out_sr / speech_sr)
        speech = np.interp(
            np.linspace(0, len(speech), target_len, endpoint=False),
            np.arange(len(speech)),
            speech,
        )
    if music_sr != out_sr:
        target_len = int(len(music) * out_sr / music_sr)
        xs = np.linspace(0, len(music), target_len, endpoint=False)
        xp = np.arange(len(music))
        if music_stereo:
            music = np.stack([np.interp(xs, xp, music[:, 0]),
                              np.interp(xs, xp, music[:, 1])], axis=1)
        else:
            music = np.interp(xs, xp, music)
    if len(music) > len(speech):
        music = music[:len(speech)]
    elif len(music) < len(speech):
        pad = len(speech) - len(music)
        if music_stereo:
            music = np.concatenate([music, music[:pad] if pad <= len(music) else np.tile(music, (pad//len(music)+1, 1))[:pad]], axis=0)
        else:
            music = np.pad(music, (0, pad), mode='wrap')
    speech = speech.astype(np.float32)
    if music_stereo:
        music = music.astype(np.float32)
        music_mono = music.mean(axis=1)
    else:
        music = music.astype(np.float32)
        music_mono = music
    # Cut 300 Hz–3.5 kHz from instrumental to suppress BS-Roformer vocal bleed
    if music_stereo:
        music[:, 0] = _cut_speech_band(music[:, 0], speech_sr)
        music[:, 1] = _cut_speech_band(music[:, 1], speech_sr)
    else:
        music = _cut_speech_band(music, speech_sr)
        music_mono = music
    print("[MIX] Speech-band EQ applied to instrumental (reduces vocal bleed)", flush=True)
    music_volume = min(music_volume, 1.0)
    active_mask = np.abs(speech) > 0.01
    if active_mask.sum() > 256:
        speech_rms = float(np.sqrt(np.mean(speech[active_mask] ** 2))) + 1e-9
    else:
        speech_rms = float(np.sqrt(np.mean(speech ** 2))) + 1e-9
    active_music_mask = np.abs(music_mono) > 0.01
    if active_music_mask.sum() > 256:
        music_rms = float(np.sqrt(np.mean(music_mono[active_music_mask] ** 2))) + 1e-9
    else:
        music_rms = float(np.sqrt(np.mean(music_mono ** 2))) + 1e-9
    scale_factor = (speech_rms / music_rms) * music_volume
    music_scaled = music * scale_factor
    print(f"[MIX] Speech active-RMS={speech_rms:.4f}, Music RMS={music_rms:.4f}, music_volume ratio={music_volume}", flush=True)
    # Output stereo: TTS voice centred (L+R identical), music keeps stereo channels
    if music_stereo:
        speech_stereo = np.stack([speech, speech], axis=1)
        mixed = speech_stereo + music_scaled
    else:
        mixed = np.stack([speech + music_scaled, speech + music_scaled], axis=1)
    print(f"[MIX] Mixed peak before limiting: {np.abs(mixed).max():.3f}", flush=True)
    max_val = np.abs(mixed).max()
    if max_val > 0.95:
        speech_peak = np.abs(speech).max()
        if speech_peak < 0.90:
            headroom = 0.95 - speech_peak
            music_peak = np.abs(music_scaled).max()
            if music_peak > 0:
                scale = min(1.0, headroom / music_peak)
                music_scaled = music * scale_factor * scale
                if music_stereo:
                    mixed = speech_stereo + music_scaled
                else:
                    mixed = np.stack([speech + music_scaled, speech + music_scaled], axis=1)
                print(f"[MIX] Music reduced to fit headroom (scale={scale:.3f})", flush=True)
            else:
                mixed = np.stack([speech, speech], axis=1)
        else:
            mixed = mixed * (0.95 / max_val)
    sf.write(str(output_path), mixed.astype(np.float32), out_sr)
    print(f"[MIX] Output stereo {out_sr}Hz: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Audio mastering / post-processing
# ---------------------------------------------------------------------------

def measure_lufs_ebur128(audio_path: Path) -> tuple[float | None, float | None]:
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", str(audio_path),
        "-filter_complex", "ebur128=peak=true",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stderr or "") + "\n" + (proc.stdout or "")
    lufs_matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", log)
    peak_matches = re.findall(r"Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS", log)
    lufs_val = float(lufs_matches[-1]) if lufs_matches else None
    peak_val = float(peak_matches[-1]) if peak_matches else None
    return lufs_val, peak_val


def ffmpeg_apply_filter(input_path: Path, output_path: Path, af: str) -> bool:
    # Probe input sample rate — loudnorm internally upsamples to 192kHz and without
    # an explicit -ar the output would be saved at 192kHz (8× too slow on playback).
    try:
        import subprocess as _sp
        _r = _sp.run(["ffprobe","-v","quiet","-show_entries","stream=sample_rate",
                      "-of","default=noprint_wrappers=1:nokey=1",str(input_path)],
                     capture_output=True, text=True)
        _in_sr = int(_r.stdout.strip().splitlines()[0]) if _r.stdout.strip() else TTS_SR
    except Exception:
        _in_sr = TTS_SR
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", af,
        "-ar", str(_in_sr),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FFMPEG] Filter failed: {af}", flush=True)
        print(result.stderr[-500:], flush=True)
        return False
    return True


def apply_voice_eq(input_path: Path, output_path: Path) -> bool:
    print(f"[EQ] Balancing voice frequencies...", flush=True)
    filters = [
        "highpass=f=45",
        "equalizer=f=90:t=q:w=1:g=2.8",
        "equalizer=f=180:t=q:w=1:g=1.5",
        "equalizer=f=300:t=q:w=1:g=0.8",
        "lowpass=f=9500",
        "equalizer=f=3200:t=q:w=1:g=-2.0",
        "equalizer=f=4300:t=q:w=1.2:g=-2.8",
        "equalizer=f=5800:t=q:w=1:g=-2.2",
        "equalizer=f=7000:t=q:w=1:g=-1.8",
    ]
    filter_str = ",".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", filter_str,
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    # FFmpeg nevie písať in-place — ak vstup == výstup, použij temp súbor
    _in_place = Path(input_path).resolve() == Path(output_path).resolve()
    _actual_out = Path(output_path).with_suffix(".eq_tmp.wav") if _in_place else Path(output_path)
    cmd[-1] = str(_actual_out)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[EQ] Failed: {result.stderr}", flush=True)
        if _in_place:
            _actual_out.unlink(missing_ok=True)
        return False

    if _in_place:
        import shutil as _sh
        _sh.move(str(_actual_out), str(output_path))

    print(f"[EQ] Voice balanced: {output_path}", flush=True)
    return True


def trim_and_smooth_chunk(wav_path: Path, silence_thresh_db: float = -50.0, pad_ms: int = 25, fade_ms: int = 10) -> bool:
    """Remove leading/trailing near-silence and apply short fades to avoid clicks at segment joins."""
    try:
        y, sr = sf.read(str(wav_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if y.size < 8:
            return False
        thr = float(10 ** (silence_thresh_db / 20.0))
        nz = np.where(np.abs(y) > thr)[0]
        if nz.size > 0:
            pad = int(sr * (pad_ms / 1000.0))
            start = max(0, int(nz[0]) - pad)
            end = min(len(y), int(nz[-1]) + pad + 1)
            y = y[start:end]
        fade_n = int(sr * (fade_ms / 1000.0))
        if fade_n > 0 and len(y) > fade_n * 2:
            y[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            y[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        sf.write(str(wav_path), y, sr)
        return True
    except Exception:
        return False


def repair_click_glitches(wav_path: Path, frame_ms: float = 20.0, thresh_ratio: float = 0.08) -> int:
    """Opraví single-frame dropout glitche v assemblovanom audu.

    Nájde miesta kde RMS klesne na < thresh_ratio * susedných framov a interpoluje.
    Vracia počet opravených glitchov.
    """
    try:
        y, sr = sf.read(str(wav_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        frame = int(sr * frame_ms / 1000.0)
        n_frames = len(y) // frame
        if n_frames < 4:
            return 0

        rms = np.array([
            float(np.sqrt(np.mean(y[i*frame:(i+1)*frame]**2)))
            for i in range(n_frames)
        ])

        repaired = 0
        for i in range(1, n_frames - 1):
            neighbor_avg = (rms[i-1] + rms[i+1]) / 2.0
            if neighbor_avg > 0.01 and rms[i] < neighbor_avg * thresh_ratio:
                # Interpoluj samples v tomto frame
                s = i * frame
                e = min(s + frame, len(y))
                prev_end = y[s - 1] if s > 0 else 0.0
                next_start = y[e] if e < len(y) else 0.0
                y[s:e] = np.linspace(prev_end, next_start, e - s, dtype=np.float32)
                repaired += 1

        if repaired > 0:
            sf.write(str(wav_path), y, sr)
            print(f"[GLITCH] Opravených {repaired} single-frame glitchov v {wav_path.name}", flush=True)
        return repaired
    except Exception as e:
        print(f"[GLITCH] repair_click_glitches zlyhalo: {e}", flush=True)
        return 0


def repair_silence_gaps(wav_path: Path, max_silence_s: float = 0.40) -> bool:
    """Shorten internal silence gaps longer than max_silence_s. Returns True if any gaps were shortened."""
    try:
        y, sr = sf.read(str(wav_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)

        win_s = 0.02  # 20ms windows
        win_samples = int(sr * win_s)
        n_windows = len(y) // win_samples
        if n_windows < 10:
            return False

        # Threshold 0.001 (~-60dB) namiesto 0.008 (~-42dB) — predtým agresívne chytalo
        # tiché samohlásky a dlhé slabiky, vyhadzovalo platné audio. -60dB je iba
        # skutočné digital silence (typicky pauza medzi vetami v TTS modeli).
        threshold = 0.001
        min_gap_windows = int(max_silence_s / win_s)
        keep_windows = int(max_silence_s / win_s)

        is_silent = np.array([
            float(np.sqrt(np.mean(y[j * win_samples:(j + 1) * win_samples] ** 2))) < threshold
            for j in range(n_windows)
        ])

        gaps = []
        in_gap = False
        gap_start = 0
        for wi in range(n_windows):
            if is_silent[wi]:
                if not in_gap:
                    in_gap = True
                    gap_start = wi
            else:
                if in_gap:
                    if wi - gap_start > min_gap_windows:
                        gaps.append((gap_start, wi))
                    in_gap = False
        if in_gap and n_windows - gap_start > min_gap_windows:
            gaps.append((gap_start, n_windows))

        if not gaps:
            return False

        result_parts = []
        prev_end_sample = 0
        total_removed = 0.0
        for gap_start_w, gap_end_w in gaps:
            gap_start_s = gap_start_w * win_samples
            gap_end_s = min(gap_end_w * win_samples, len(y))
            result_parts.append(y[prev_end_sample:gap_start_s])
            gap_len_samples = gap_end_s - gap_start_s
            keep_samples = keep_windows * win_samples
            if gap_len_samples > keep_samples:
                half_keep = keep_samples // 2
                result_parts.append(y[gap_start_s:gap_start_s + half_keep])
                result_parts.append(y[gap_end_s - half_keep:gap_end_s])
                total_removed += (gap_len_samples - keep_samples) / float(sr)
            else:
                result_parts.append(y[gap_start_s:gap_end_s])
            prev_end_sample = gap_end_s
        result_parts.append(y[prev_end_sample:])

        result = np.concatenate(result_parts)
        if total_removed > 0.05:
            print(f"[SILENCE-REPAIR] {wav_path.name}: removed {total_removed:.1f}s silence", flush=True)
            sf.write(str(wav_path), result, sr)
            return True
        return False
    except Exception as e:
        print(f"[SILENCE-REPAIR] Failed: {e}", flush=True)
        return False


def apply_noise_reduction(input_path: Path, preset: str = "mild") -> bool:
    temp_path = input_path.with_name(f"{input_path.stem}_denoised.wav")
    if preset == "strong":
        filter_str = (
            "highpass=f=85,"
            "afftdn=nr=17:nt=w:om=o,"
            "lowpass=f=9800,"
            "equalizer=f=220:t=q:w=1.0:g=-1.8,"
            "equalizer=f=6500:t=q:w=1.2:g=-2.6"
        )
    else:
        filter_str = (
            "highpass=f=65,"
            "afftdn=nr=9:nt=w:om=o,"
            "lowpass=f=12500,"
            "equalizer=f=4200:t=q:w=1.0:g=-0.6,"
            "equalizer=f=7000:t=q:w=1.1:g=-1.6"
        )

    # afftdn is slow — compute timeout from file duration (4× realtime + 60s buffer, min 10 min)
    try:
        dur_s = sf.info(str(input_path)).duration
    except Exception:
        dur_s = 0.0
    denoise_timeout = max(600, int(dur_s * 4) + 60)
    print(f"[DENOISE] Starting ({preset}), file={dur_s:.0f}s, timeout={denoise_timeout}s ...", flush=True)

    try:
        _in_sr = int(sf.info(str(input_path)).samplerate)
    except Exception:
        _in_sr = TTS_SR
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", filter_str,
        "-ar", str(_in_sr),
        "-c:a", "pcm_s16le",
        str(temp_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=denoise_timeout)
    except subprocess.TimeoutExpired:
        print(f"[DENOISE] Timed out after {denoise_timeout}s — skipping", flush=True)
        if temp_path.exists():
            temp_path.unlink()
        return False

    if result.returncode == 0 and temp_path.exists():
        shutil.move(str(temp_path), str(input_path))
        print(f"[DENOISE] Done ({preset}): {input_path}", flush=True)
        return True
    else:
        print(f"[DENOISE] Failed: {result.stderr[-300:]}", flush=True)
        if temp_path.exists():
            temp_path.unlink()
        return False


def apply_soft_high_cut(tts_path: Path) -> bool:
    tmp_out = tts_path.with_name(f"{tts_path.stem}_tone.wav")
    # Agresívne tlmenie výšok + boost basov pre OmniVoice ostrý/tenký hlas.
    af = (
        "lowpass=f=6500,"
        "equalizer=f=2800:t=q:w=1.0:g=-2.5,"
        "equalizer=f=4000:t=q:w=1.0:g=-3.5,"
        "equalizer=f=5500:t=q:w=1.0:g=-4.0,"
        "highpass=f=70,"
        "equalizer=f=110:t=q:w=0.9:g=+3.5,"
        "equalizer=f=180:t=q:w=1.0:g=+2.0,"
        "equalizer=f=350:t=q:w=1.0:g=+1.0"
    )
    if not ffmpeg_apply_filter(tts_path, tmp_out, af):
        return False
    try:
        shutil.move(str(tmp_out), str(tts_path))
        print(f"[TONE] Highs softened: {af}", flush=True)
        return True
    except Exception as e:
        print(f"[TONE] Failed to replace softened file: {e}", flush=True)
        return False


def _loudnorm_two_pass(input_path: Path, output_path: Path, target_i: float = -22.0,
                       target_tp: float = -2.0, target_lra: float = 5.0) -> bool:
    """Two-pass EBU R128 loudnorm — accurate LUFS for speech (single-pass overshoots ~3-4 dB)."""
    import json as _json
    import re as _re
    # Probe SR so output stays at same rate
    try:
        _r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=sample_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
            capture_output=True, text=True,
        )
        in_sr = int(_r.stdout.strip().splitlines()[0]) if _r.stdout.strip() else TTS_SR
    except Exception:
        in_sr = TTS_SR
    # Pass 1: measure loudness stats
    p1 = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(input_path),
         "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = _re.search(r'\{[^{}]+\}', p1.stderr, _re.DOTALL)
    if not m:
        print("[LOUDNORM] Pass-1 JSON parse failed, falling back to single-pass", flush=True)
        p = subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path),
             "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}",
             "-ar", str(in_sr), "-c:a", "pcm_s16le", str(output_path)],
            capture_output=True, text=True,
        )
        return p.returncode == 0
    st = _json.loads(m.group(0))
    # Pass 2: apply with measured values (linear mode = accurate)
    af2 = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        f":measured_I={st['input_i']}:measured_TP={st['input_tp']}"
        f":measured_LRA={st['input_lra']}:measured_thresh={st['input_thresh']}"
        f":offset={st['target_offset']}:linear=true:print_format=summary"
    )
    p2 = subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-af", af2,
         "-ar", str(in_sr), "-c:a", "pcm_s16le", str(output_path)],
        capture_output=True, text=True,
    )
    if p2.returncode != 0:
        print(f"[LOUDNORM] Pass-2 failed: {p2.stderr[-300:]}", flush=True)
    return p2.returncode == 0


def apply_tts_master_chain(tts_path: Path, denoise: bool = True, fast_mode: bool = True) -> bool:
    tmp_out = tts_path.with_name(f"{tts_path.stem}_master.wav")
    # Measured HeyGen reference: -21.8 LUFS, LRA=5.0, TP=-1.9 dBFS, dynamic range 12.2 dB.
    # Compressor: ratio=3.5 to achieve LRA ~5 LU (was ratio=2 → 18 dB dynamic range, too wide).
    _compress = "acompressor=threshold=-20dB:ratio=3.5:attack=5:release=80:makeup=3dB"
    # EQ: HeyGen spectral analysis vs our TTS (normalized to 500-1kHz):
    #   400 Hz: -3 dB (mud cut). 1.2 kHz: +1.5 dB (body). 5 kHz shelf: +5 dB (HeyGen 6-10kHz is
    #   9.7 dB higher than ours — Chatterbox TTS rolls off above 8kHz; shelf compensates partially).
    _tone = [
        "highpass=f=100",
        "lowpass=f=17000",
        "equalizer=f=400:t=q:w=2.0:g=-3",
        "equalizer=f=1200:t=q:w=1.5:g=+1.5",
        "equalizer=f=5000:t=s:g=+5",
    ]
    tmp_eq = tts_path.with_name(f"{tts_path.stem}_eq.wav")
    if fast_mode:
        chain = list(_tone)
        chain.extend([_compress, "alimiter=limit=-2.0dB"])
    else:
        if denoise:
            chain = ["afftdn=nr=5:nt=w:om=o"]
        else:
            chain = []
        chain.extend(_tone)
        chain.extend([_compress, "alimiter=limit=-2.0dB"])
    af = ",".join(chain)
    ok = ffmpeg_apply_filter(tts_path, tmp_eq, af)
    if not ok:
        return False
    # Two-pass loudnorm for accurate -22 LUFS (single-pass overshoots ~3-4 dB on speech).
    ok2 = _loudnorm_two_pass(tmp_eq, tmp_out, target_i=-22.0, target_tp=-2.0, target_lra=5.0)
    try:
        tmp_eq.unlink(missing_ok=True)
    except Exception:
        pass
    if not ok2:
        return False
    try:
        shutil.move(str(tmp_out), str(tts_path))
        print(f"[MASTER] Applied TTS master chain (EQ+compress+2-pass loudnorm -22 LUFS LRA=5): {tts_path.name}", flush=True)
        return True
    except Exception as e:
        print(f"[MASTER] Failed to replace mastered file: {e}", flush=True)
        return False


def match_tts_to_reference_voice(tts_path: Path, ref_vocals_path: Path, output_path: Path) -> bool:
    ref_lufs, ref_peak = measure_lufs_ebur128(ref_vocals_path)
    if ref_lufs is None:
        print("[CLONE-MATCH] Could not read reference LUFS, skipping voice match", flush=True)
        return False
    print(
        f"[CLONE-MATCH] Reference vocals: I={ref_lufs:.2f} LUFS"
        + (f", Peak={ref_peak:.2f} dBFS" if ref_peak is not None else ""),
        flush=True,
    )
    tmp_norm = output_path.with_name(f"{output_path.stem}_norm.wav")
    tmp_eq = output_path.with_name(f"{output_path.stem}_eq.wav")
    if not ffmpeg_apply_filter(tts_path, tmp_norm, f"loudnorm=I={ref_lufs:.2f}:TP=-1.5:LRA=11"):
        return False
    if not ffmpeg_apply_filter(tmp_norm, tmp_eq, "highpass=f=80,lowpass=f=12000"):
        return False
    if not ffmpeg_apply_filter(
        tmp_eq, output_path,
        "acompressor=threshold=-18dB:ratio=3:attack=10:release=120,alimiter=limit=-1.5dB",
    ):
        return False
    for p in (tmp_norm, tmp_eq):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    return True


def mix_with_sidechain_ffmpeg(
    speech_path: Path,
    music_path: Path,
    output_path: Path,
    ref_vocals_lufs: float | None = None,
    ref_music_lufs: float | None = None,
    music_trim_ratio: float = 1.0,
) -> bool:
    music_gain_db = 0.0
    speech_lufs, _ = measure_lufs_ebur128(speech_path)
    music_lufs, _ = measure_lufs_ebur128(music_path)
    if (
        ref_vocals_lufs is not None
        and ref_music_lufs is not None
        and speech_lufs is not None
        and music_lufs is not None
    ):
        target_delta = ref_music_lufs - ref_vocals_lufs
        current_delta = music_lufs - speech_lufs
        music_gain_db = target_delta - current_delta
        music_gain_db = max(-12.0, min(12.0, music_gain_db))
        print(f"[MIX] Music gain match: {music_gain_db:+.2f} dB", flush=True)
    if music_trim_ratio and music_trim_ratio > 0:
        trim_db = 20.0 * math.log10(music_trim_ratio)
        trim_db = max(-6.0, min(6.0, trim_db))
        music_gain_db += trim_db
        print(f"[MIX] Music trim: {trim_db:+.2f} dB (ratio={music_trim_ratio:.3f})", flush=True)
    filter_complex = (
        f"[0:a]volume={music_gain_db:+.2f}dB[m0];"
        "[m0][1:a]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=300[mduck];"
        "[mduck][1:a]amix=inputs=2:normalize=0[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(music_path),
        "-i", str(speech_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[MIX] Sidechain mix failed: {result.stderr[-500:]}", flush=True)
        return False
    return True


def apply_speech_gain(
    wav_path: Path, speech_gain: float, auto_gain: bool, max_auto_gain: float = 200.0, auto_target_peak: float = 0.90
) -> float:
    if not auto_gain and abs(speech_gain - 1.0) < 1e-9:
        return 1.0
    samples, sr = sf.read(str(wav_path))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    peak = np.abs(samples).max()
    gain = speech_gain
    if auto_gain and peak > 0:
        target_gain = auto_target_peak / peak
        target_gain = min(target_gain, max_auto_gain)
        gain = max(gain, target_gain)
    if abs(gain - 1.0) < 1e-5 and not auto_gain:
        return gain
    samples *= gain
    peak = np.abs(samples).max()
    if peak > 0.99:
        samples *= 0.99 / peak
    sf.write(str(wav_path), samples, sr)
    print(f"[GAIN] Applied speech gain x{gain:.2f} to {wav_path}", flush=True)


# ---------------------------------------------------------------------------
# Prosody-based emotion intensity classifier
# ---------------------------------------------------------------------------

def classify_prosody_levels(
    segments: list[dict],
    audio_source: "Path | str",
    sr: int = 16000,
) -> list[str]:
    """Klasifikuje intenzitu emócie každého segmentu na základe prosody originálu.

    Používa RMS energiu a štandardnú odchýlku F0 (základná frekvencia).
    Normalizuje percentilmi v rámci videa — bez nutnosti kalibrácie.

    Returns:
        List of "low" | "medium" | "high" per segment.
    """
    try:
        import librosa
    except ImportError:
        print("[PROSODY] librosa nie je nainštalovaná — fallback na 'low'", flush=True)
        return ["low"] * len(segments)

    audio_source = Path(audio_source)
    # Ak je video súbor, extrahujeme audio cez ffmpeg do temp wav
    _tmp_wav: "Path | None" = None
    load_path = audio_source
    if audio_source.suffix.lower() not in (".wav", ".flac", ".ogg", ".mp3"):
        import tempfile, subprocess as _sp
        _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        _tmp_wav = Path(_tmp.name)
        _tmp.close()
        try:
            _sp.run(
                ["ffmpeg", "-y", "-i", str(audio_source),
                 "-ac", "1", "-ar", str(sr), str(_tmp_wav)],
                capture_output=True, timeout=120,
            )
            load_path = _tmp_wav
        except Exception as e:
            print(f"[PROSODY] ffmpeg extrakcia zlyhala: {e}", flush=True)
            return ["low"] * len(segments)

    try:
        y_full, _sr = librosa.load(str(load_path), sr=sr, mono=True)
    except Exception as e:
        print(f"[PROSODY] Načítanie audia zlyhalo: {e}", flush=True)
        return ["low"] * len(segments)
    finally:
        if _tmp_wav and _tmp_wav.exists():
            _tmp_wav.unlink(missing_ok=True)

    rms_scores: list[float] = []
    f0_scores: list[float] = []

    for seg in segments:
        t_start = float(seg.get("start", 0))
        t_end   = float(seg.get("end", t_start + 1.0))
        i_start = int(t_start * _sr)
        i_end   = int(t_end   * _sr)
        chunk   = y_full[i_start:i_end]

        if len(chunk) < _sr * 0.1:          # kratší ako 100 ms → neutrálne
            rms_scores.append(0.0)
            f0_scores.append(0.0)
            continue

        # RMS energia
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_scores.append(rms)

        # F0 štandardná odchýlka (expresivita výšky hlasu)
        try:
            f0, _, _ = librosa.pyin(
                chunk,
                fmin=librosa.note_to_hz("C2"),   # ~65 Hz
                fmax=librosa.note_to_hz("C6"),   # ~1047 Hz
                sr=_sr,
            )
            f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            f0_scores.append(float(f0_valid.std()) if len(f0_valid) > 1 else 0.0)
        except Exception:
            f0_scores.append(0.0)

    if not rms_scores:
        return ["low"] * len(segments)

    # Normalizácia: rank v rámci videa (0–1)
    def _rank_norm(vals: list[float]) -> list[float]:
        arr = np.array(vals, dtype=float)
        if arr.max() - arr.min() < 1e-9:
            return [0.5] * len(vals)
        return list((arr - arr.min()) / (arr.max() - arr.min()))

    rms_norm = _rank_norm(rms_scores)
    f0_norm  = _rank_norm(f0_scores)

    # Kombinácia: 60% RMS energia + 40% F0 variabilita
    combined = [0.60 * r + 0.40 * f for r, f in zip(rms_norm, f0_norm)]

    # Percentilové prahy: bottom 33% = low, top 33% = high
    p33 = float(np.percentile(combined, 33))
    p67 = float(np.percentile(combined, 67))

    levels = []
    for score in combined:
        if score <= p33:
            levels.append("low")
        elif score <= p67:
            levels.append("medium")
        else:
            levels.append("high")

    low_c    = levels.count("low")
    medium_c = levels.count("medium")
    high_c   = levels.count("high")
    print(f"[PROSODY] low={low_c} medium={medium_c} high={high_c}", flush=True)
    return levels
    return gain
