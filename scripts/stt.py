"""
STT (Speech-to-Text) & segment processing module.
Bootstrap, audio extraction, Whisper transcription, segment manipulation.
"""

import os
import re
import sys
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa

from config import WHISPER_SR, FFPROBE_TIMEOUT, FFMPEG_TIMEOUT


# ---------------------------------------------------------------------------
# Bootstrap: re-exec in configured conda env
# ---------------------------------------------------------------------------

def _preferred_python_candidates() -> list[Path]:
    home = Path.home()
    if os.name == "nt":
        return [
            home / "miniconda3" / "envs" / "heygen_local" / "python.exe",
            home / "miniconda3" / "envs" / "musetalk_env" / "python.exe",
        ]
    return [
        home / "miniforge3" / "envs" / "musetalk_env" / "bin" / "python",
        home / "miniconda3" / "envs" / "musetalk_env" / "bin" / "python",
    ]


def _bootstrap_preferred_python() -> None:
    # Re-exec in configured conda env so CUDA/PyTorch stack is consistent.
    if os.environ.get("MUSETALK_PY_BOOTSTRAPPED") == "1":
        return
    current = Path(sys.executable).resolve()
    for candidate in _preferred_python_candidates():
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if not resolved.exists():
            continue
        if resolved == current:
            return
        env = os.environ.copy()
        env["MUSETALK_PY_BOOTSTRAPPED"] = "1"
        os.execvpe(str(resolved), [str(resolved), *sys.argv], env)
    return


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def extract_audio_ffmpeg(input_path: str, output_wav: str, ffmpeg_path: str = "ffmpeg") -> None:
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(WHISPER_SR),
        "-c:a",
        "pcm_s16le",
        output_wav,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"[STT] ffmpeg timed out extracting audio from: {input_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[STT] ffmpeg failed (rc={e.returncode}) for: {input_path}") from e


def get_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def get_video_duration(input_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=FFPROBE_TIMEOUT).strip()
        return float(out)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"[STT] ffprobe timed out on: {input_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[STT] ffprobe failed (rc={e.returncode}) for: {input_path}") from e
    except ValueError:
        raise RuntimeError(f"[STT] ffprobe returned non-numeric duration for: {input_path}")


def format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def split_text(text: str, max_chars: int) -> list[str]:
    def hard_split(sentence: str) -> list[str]:
        words = sentence.split()
        out = []
        buf = ""
        for w in words:
            if not buf:
                buf = w
            elif len(buf) + 1 + len(w) <= max_chars:
                buf = f"{buf} {w}"
            else:
                out.append(buf)
                buf = w
        if buf:
            out.append(buf)
        final = []
        for seg in out:
            if len(seg) <= max_chars:
                final.append(seg)
            else:
                for i in range(0, len(seg), max_chars):
                    final.append(seg[i : i + max_chars])
        return final

    parts = text.strip().splitlines()
    chunks = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(hard_split(p))
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# Whisper hallucination filter
# ---------------------------------------------------------------------------

_WHISPER_NO_SPEECH_MAX = 0.6
_WHISPER_AVG_LOGPROB_MIN = -1.0
_WHISPER_COMPRESSION_MAX = 2.4

_WHISPER_BOH = {
    # Universal / English
    "thank you for watching", "thanks for watching", "please subscribe",
    "like and subscribe", "subtitles by", "transcribed by", "translation by",
    "subtitles made by", "captions by", "closed captions",
    "amara.org", "www.", ".com", ".sk", ".cz",
    "dimatorzok", "subscribestar", "patreon.com",
    # Czech
    "děkuji za pozornost", "děkuji vám za pozornost", "díky za sledování",
    "titulky", "přepis", "přeloženo",
    # Slovak
    "ďakujem za pozornosť", "ďakujem vám za pozornosť", "ďakujem za sledovanie",
    "ďakujem", "subscribe", "titulky", "prepis",
    # Sexual content — Whisper halucinations over silent/music segments
    "big booty", "cam slut", "dog dick", "sucking", "making out",
    "onlyfans", "pornhub", "xvideos", "xxx", "anal", "blowjob",
    "sex tape", "leaked video", "nude", "naked",
}

import re as _re
_WHISPER_EXPLICIT_RE = _re.compile(
    r"\b(fuck|fucking|fucked|shit|porn|porny|slut|slutty|dick|pussy|cock|booty|"
    r"bitch|whore|hentai|dildo|orgasm|masturbat|cumshot|creampie|gangbang|"
    r"deepthroat|handjob|rimjob|threesome|milf|gilf)\b",
    _re.IGNORECASE,
)


def _seg_is_hallucination(
    seg,
    *,
    no_speech_max: float = _WHISPER_NO_SPEECH_MAX,
    avg_logprob_min: float = _WHISPER_AVG_LOGPROB_MIN,
    compression_max: float = _WHISPER_COMPRESSION_MAX,
) -> bool:
    """Return True if the Whisper segment looks like a hallucination."""
    if seg.no_speech_prob > no_speech_max:
        return True
    if seg.avg_logprob < avg_logprob_min:
        return True
    if seg.compression_ratio > compression_max:
        return True
    text_lower = seg.text.strip().lower()
    if any(phrase in text_lower for phrase in _WHISPER_BOH):
        return True
    if _WHISPER_EXPLICIT_RE.search(text_lower):
        return True
    return False


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_whisper(
    audio_path: str,
    model_name: str,
    download_root: str,
    device: str,
    initial_prompt: str = "",
    whisper_no_speech_max: float = _WHISPER_NO_SPEECH_MAX,
    whisper_avg_logprob_min: float = _WHISPER_AVG_LOGPROB_MIN,
    whisper_compression_max: float = _WHISPER_COMPRESSION_MAX,
):
    from faster_whisper import WhisperModel
    compute_type = "int8_float16" if device == "cuda" and torch.cuda.is_available() else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=download_root)
    try:
        segments_gen, _info = model.transcribe(
            audio_path,
            initial_prompt=initial_prompt or None,
            word_timestamps=True,
            # Predtým False — spôsobovalo lámanie viet uprostred (model nevidel kontext).
            # Zapnuté na True dáva model lepšiu segmentáciu (~10-15% menej fragmentov).
            # Halucination filter (_seg_is_hallucination) chytí prípadné rozbehy.
            condition_on_previous_text=True,
            no_speech_threshold=whisper_no_speech_max,
            log_prob_threshold=whisper_avg_logprob_min,
            compression_ratio_threshold=whisper_compression_max,
        )
        result_segments = []
        full_text_parts = []
        _prev_text_norm = ""
        for seg in segments_gen:
            if _seg_is_hallucination(
                seg,
                no_speech_max=whisper_no_speech_max,
                avg_logprob_min=whisper_avg_logprob_min,
                compression_max=whisper_compression_max,
            ):
                print(f"[WHISPER] Halucinácia preskočená (no_speech={seg.no_speech_prob:.2f} logprob={seg.avg_logprob:.2f} cr={seg.compression_ratio:.2f}): {seg.text.strip()[:60]}", flush=True)
                continue
            _cur_norm = seg.text.strip().lower()
            if _cur_norm and _cur_norm == _prev_text_norm:
                print(f"[WHISPER] Repetícia preskočená: {seg.text.strip()[:60]!r}", flush=True)
                continue
            _prev_text_norm = _cur_norm
            result_segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            full_text_parts.append(seg.text.strip())
        return {
            "segments": result_segments,
            "text": " ".join(full_text_parts),
        }
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def transcribe_whisper_vad(
    audio_path: str,
    model_name: str,
    download_root: str,
    device: str,
    vad_threshold: float,
    min_silence_ms: int,
    min_speech_ms: int,
    speech_pad_ms: int,
    max_speech_s: float,
    initial_prompt: str = "",
    whisper_no_speech_max: float = _WHISPER_NO_SPEECH_MAX,
    whisper_avg_logprob_min: float = _WHISPER_AVG_LOGPROB_MIN,
    whisper_compression_max: float = _WHISPER_COMPRESSION_MAX,
) -> dict:
    """Returns {"segments": [{"start", "end", "text"}, ...], "text": "..."}"""
    from faster_whisper import WhisperModel
    compute_type = "int8_float16" if device == "cuda" and torch.cuda.is_available() else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=download_root)
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )
    get_speech_timestamps, _save_audio, _read_audio, _vad_iter, _collect_chunks = vad_utils
    wav_np, sr = sf.read(audio_path)
    if wav_np.ndim > 1:
        wav_np = wav_np[:, 0]
    if sr != WHISPER_SR:
        wav_np = librosa.resample(wav_np.astype(np.float32), orig_sr=sr, target_sr=WHISPER_SR)
    wav = torch.from_numpy(wav_np.astype(np.float32))
    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=WHISPER_SR,
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    result_segments: list[dict] = []
    full_texts: list[str] = []
    try:
        if not timestamps:
            return {"segments": [], "text": ""}
        prompt = initial_prompt or None
        _prev_vad_text_norm = ""
        for i, t in enumerate(timestamps, 1):
            chunk_start_sample = t["start"]
            chunk_end_sample = t["end"]
            if chunk_end_sample <= chunk_start_sample:
                continue
            chunk_start_s = chunk_start_sample / WHISPER_SR
            seg = wav[chunk_start_sample:chunk_end_sample].numpy()
            if max_speech_s and (len(seg) / WHISPER_SR) > max_speech_s:
                step = int(max_speech_s * WHISPER_SR)
                for j in range(0, len(seg), step):
                    chunk = seg[j : j + step]
                    if len(chunk) < 1600:
                        continue
                    offset_s = chunk_start_s + j / WHISPER_SR
                    print(f"[WHISPER] VAD chunk {i} split {j//step + 1}", flush=True)
                    segs_gen, _ = model.transcribe(chunk, initial_prompt=prompt, word_timestamps=True,
                        condition_on_previous_text=False,
                        no_speech_threshold=whisper_no_speech_max,
                        log_prob_threshold=whisper_avg_logprob_min,
                        compression_ratio_threshold=whisper_compression_max,
                    )
                    for s in segs_gen:
                        text = s.text.strip()
                        if not text or _seg_is_hallucination(
                            s,
                            no_speech_max=whisper_no_speech_max,
                            avg_logprob_min=whisper_avg_logprob_min,
                            compression_max=whisper_compression_max,
                        ):
                            if text:
                                print(f"[WHISPER] Halucinácia preskočená: {text[:60]}", flush=True)
                            continue
                        _cur_norm = text.lower()
                        if _cur_norm and _cur_norm == _prev_vad_text_norm:
                            print(f"[WHISPER] Repetícia preskočená: {text[:60]!r}", flush=True)
                            continue
                        _prev_vad_text_norm = _cur_norm
                        result_segments.append({
                            "start": round(offset_s + s.start, 3),
                            "end": round(offset_s + s.end, 3),
                            "text": text,
                        })
                        full_texts.append(text)
            else:
                if len(seg) < 1600:
                    continue
                print(f"[WHISPER] VAD chunk {i}", flush=True)
                segs_gen, _ = model.transcribe(seg, initial_prompt=prompt, word_timestamps=True,
                    condition_on_previous_text=False,
                    no_speech_threshold=whisper_no_speech_max,
                    log_prob_threshold=whisper_avg_logprob_min,
                    compression_ratio_threshold=whisper_compression_max,
                )
                for s in segs_gen:
                    text = s.text.strip()
                    if not text or _seg_is_hallucination(
                        s,
                        no_speech_max=whisper_no_speech_max,
                        avg_logprob_min=whisper_avg_logprob_min,
                        compression_max=whisper_compression_max,
                    ):
                        if text:
                            print(f"[WHISPER] Halucinácia preskočená: {text[:60]}", flush=True)
                        continue
                    _cur_norm = text.lower()
                    if _cur_norm and _cur_norm == _prev_vad_text_norm:
                        print(f"[WHISPER] Repetícia preskočená: {text[:60]!r}", flush=True)
                        continue
                    _prev_vad_text_norm = _cur_norm
                    result_segments.append({
                        "start": round(chunk_start_s + s.start, 3),
                        "end": round(chunk_start_s + s.end, 3),
                        "text": text,
                    })
                    full_texts.append(text)
    finally:
        del wav, wav_np, model, vad_model
        if device == "cuda":
            torch.cuda.empty_cache()
    return {"segments": result_segments, "text": " ".join(full_texts)}


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------

def normalize_segments(segments: list[dict], min_gap: float = 0.08, tail_trim: float = 0.03) -> list[dict]:
    # Ensure non-overlapping, ordered segments and add small tail trim
    segs = sorted(segments, key=lambda s: s["start"])
    cleaned = []
    prev_end = 0.0
    for i, seg in enumerate(segs):
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        if tail_trim > 0:
            end = max(start, end - tail_trim)
        if start < prev_end + min_gap:
            start = prev_end + min_gap
        if end <= start:
            continue
        cleaned.append({"start": start, "end": end, "text": seg.get("text", "")})
        prev_end = end
    return cleaned


def merge_adjacent_segments(segments: list[dict], max_gap_s: float, max_dur_s: float) -> list[dict]:
    """Merge nearby segments while keeping max combined duration."""
    if not segments or max_gap_s <= 0 or max_dur_s <= 0:
        return segments
    segs = sorted(segments, key=lambda s: float(s["start"]))
    merged = []
    cur = {
        "start": float(segs[0]["start"]),
        "end": float(segs[0]["end"]),
        "text": (segs[0].get("text") or "").strip(),
    }
    for nxt in segs[1:]:
        nstart = float(nxt["start"])
        nend = float(nxt["end"])
        ntext = (nxt.get("text") or "").strip()
        gap = max(0.0, nstart - cur["end"])
        combined_dur = nend - cur["start"]
        if gap <= max_gap_s and combined_dur <= max_dur_s:
            cur["end"] = nend
            if ntext:
                cur["text"] = f"{cur['text']} {ntext}".strip() if cur["text"] else ntext
        else:
            merged.append(cur)
            cur = {"start": nstart, "end": nend, "text": ntext}
    merged.append(cur)
    return merged


def merge_short_segments(
    segments: list[dict],
    min_dur_s: float = 1.2,
    min_words: int = 2,
    max_gap_s: float = 0.8,
    max_dur_s: float = 25.0,
) -> list[dict]:
    """
    Merge very short filler-like segments into neighbors.
    Rule:
    - if duration < min_dur_s OR word_count < min_words
    - and gap to neighbor <= max_gap_s
    """
    if not segments:
        return segments

    segs = [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": (s.get("text") or "").strip(),
        }
        for s in sorted(segments, key=lambda x: float(x["start"]))
    ]

    i = 0
    while i < len(segs):
        seg = segs[i]
        dur = seg["end"] - seg["start"]
        words = len(seg["text"].split())
        is_short = dur < min_dur_s or words < min_words
        if not is_short or len(segs) == 1:
            i += 1
            continue

        candidates = []
        if i > 0:
            prev = segs[i - 1]
            gap_prev = max(0.0, seg["start"] - prev["end"])
            dur_prev = seg["end"] - prev["start"]
            if gap_prev <= max_gap_s and dur_prev <= max_dur_s:
                candidates.append(("prev", gap_prev))
        if i + 1 < len(segs):
            nxt = segs[i + 1]
            gap_next = max(0.0, nxt["start"] - seg["end"])
            dur_next = nxt["end"] - seg["start"]
            if gap_next <= max_gap_s and dur_next <= max_dur_s:
                candidates.append(("next", gap_next))

        if not candidates:
            i += 1
            continue

        direction = min(candidates, key=lambda x: x[1])[0]
        if direction == "prev":
            prev = segs[i - 1]
            prev["end"] = seg["end"]
            if seg["text"]:
                prev["text"] = f"{prev['text']} {seg['text']}".strip() if prev["text"] else seg["text"]
            segs.pop(i)
            i = max(i - 1, 0)
        else:
            nxt = segs[i + 1]
            nxt["start"] = seg["start"]
            if seg["text"]:
                nxt["text"] = f"{seg['text']} {nxt['text']}".strip() if nxt["text"] else seg["text"]
            segs.pop(i)
            i = max(i - 1, 0)

    return segs


def _split_text_into_parts(text: str, max_dur: float, total_dur: float, min_dur: float) -> list[str]:
    """
    Rozdelí text na časti tak aby každá časť dostala slot úmerný svojej dĺžke.
    Priorita rozdeľovacích miest: vety (. ! ?) > bodkočiarka/dvojbodka > čiarka > medzera.
    Vracia list textových častí, každá min 1 slovo.
    """
    text = text.strip()
    if not text:
        return [text]

    num_parts = max(2, int(total_dur / max_dur) + (1 if total_dur % max_dur > min_dur else 0))

    # Rozdeľovacie vzory v poradí priority
    split_patterns = [
        r'(?<=[.!?])\s+',           # vety
        r'(?<=[;:])\s+',             # bodkočiarka, dvojbodka
        r'(?<=,)\s+',                # čiarka
        r'\s+',                      # medzera (fallback)
    ]

    # Nájdi najlepšie rozdeľovacie miesta
    chunks = [text]
    for pattern in split_patterns:
        parts = re.split(pattern, text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= num_parts:
            chunks = parts
            break

    if len(chunks) < 2:
        # Posledná záchrana — rozdeliť podľa slov
        words = text.split()
        mid = len(words) // 2
        chunks = [" ".join(words[:mid]), " ".join(words[mid:])]
        chunks = [c for c in chunks if c]

    # Zlúč krátke chunks do skupín tak aby žiadna skupina neprekračovala max_dur
    # Pridelenie slotov proporcionálne podľa počtu znakov
    total_chars = sum(len(c) for c in chunks)
    if total_chars == 0:
        return [text]

    # Greedy grouping — pridávaj chunks kým slot skupiny neprekročí max_dur
    parts: list[str] = []
    current_group: list[str] = []
    current_chars = 0

    for chunk in chunks:
        chunk_chars = len(chunk)
        group_dur = (current_chars + chunk_chars) / total_chars * total_dur
        if current_group and group_dur > max_dur * 1.1:
            parts.append(" ".join(current_group))
            current_group = [chunk]
            current_chars = chunk_chars
        else:
            current_group.append(chunk)
            current_chars += chunk_chars

    if current_group:
        parts.append(" ".join(current_group))

    return [p for p in parts if p.strip()] or [text]


def split_long_segments(segments: list[dict], max_duration: float = 6.0, min_duration: float = 2.0) -> list[dict]:
    """
    Rozdelí segmenty dlhšie než max_duration na menšie časti.
    Časové sloty sa prideľujú proporcionálne podľa počtu znakov textu
    (dlhšia veta = dlhší slot), nie rovnomerne.
    """
    result = []

    for seg in segments:
        duration = float(seg["end"]) - float(seg["start"])
        text = (seg.get("text") or "").strip()

        if duration <= max_duration or not text:
            result.append(seg)
            continue

        parts_text = _split_text_into_parts(text, max_duration, duration, min_duration)

        if len(parts_text) < 2:
            result.append(seg)
            continue

        # Prideliť sloty proporcionálne podľa počtu znakov
        total_chars = sum(len(p) for p in parts_text)
        t_cursor = float(seg["start"])
        for i, part_text in enumerate(parts_text):
            char_ratio = len(part_text) / total_chars if total_chars > 0 else 1.0 / len(parts_text)
            part_dur = duration * char_ratio
            # Posledná časť dostane zvyšok (zabraní driftom)
            if i == len(parts_text) - 1:
                part_end = float(seg["end"])
            else:
                part_end = round(t_cursor + part_dur, 3)

            new_seg = dict(seg)  # zachová všetky pôvodné polia (text_src, adapt_*, ...)
            new_seg["text"] = part_text
            new_seg["start"] = round(t_cursor, 3)
            new_seg["end"] = part_end
            result.append(new_seg)
            t_cursor = part_end

        slot_str = " + ".join(f"{float(r['end'])-float(r['start']):.1f}s" for r in result[-len(parts_text):])
        print(
            f"[SPLIT] {duration:.1f}s → {len(parts_text)} častí: {slot_str}",
            flush=True,
        )

    return result


def get_sliding_window_context(segments: list[dict], current_idx: int, window_size: int = 2) -> tuple[str, str]:
    """
    Get context from surrounding segments for better translation.
    Returns (before_context, after_context) as strings.
    """
    before_texts = []
    after_texts = []

    # Get previous segments
    for i in range(max(0, current_idx - window_size), current_idx):
        text = segments[i].get("text", "").strip()
        if text:
            before_texts.append(text)

    # Get following segments
    for i in range(current_idx + 1, min(len(segments), current_idx + 1 + window_size)):
        text = segments[i].get("text", "").strip()
        if text:
            after_texts.append(text)

    return " ".join(before_texts), " ".join(after_texts)


# ---------------------------------------------------------------------------
# ASR transcription error corrections — Whisper mishears technical terms
# ---------------------------------------------------------------------------

# Default initial_prompt injected automatically when content_type == "sql".
# Listing SQL terms in the prompt biases Whisper to use correct spellings
# (e.g. "COALESCE" instead of "koalas", "NULLIF" instead of "null if").
SQL_WHISPER_PROMPT = (
    "SQL, SELECT, FROM, WHERE, GROUP BY, ORDER BY, HAVING, DISTINCT, "
    "INNER JOIN, LEFT JOIN, RIGHT JOIN, CROSS JOIN, "
    "COALESCE, ISNULL, NULLIF, CAST, CONVERT, GETDATE, DATEPART, DATEDIFF, "
    "VARCHAR, NVARCHAR, DATETIME, BIGINT, INT, NULL, NOT NULL, "
    "primary key, foreign key, stored procedure, index, schema, table, column, "
    "COUNT, SUM, AVG, MIN, MAX, UNION, EXCEPT, INTERSECT, ROW_NUMBER, DENSE_RANK"
)


# SQL: Whisper mishears SQL function names as similar-sounding English words.
# Applied to source segments BEFORE translation when content_type == "sql".
_SQL_ASR_FIXES = [
    # COALESCE — instructor's English pronunciation sounds like "koalas" to Whisper
    (r'\bkoalas?\b',                                                    'COALESCE'),
    (r'\bco[\s-]?aless?e?\b',                                          'COALESCE'),
    # NULLIF — spoken as "null if" only when clearly a function name
    (r'\b(?:function\s+(?:called\s+)?null\s+if|called\s+null\s+if|sql\s+function\s+null\s+if)\b', 'NULLIF'),
    # ISNULL — spoken as "is null" (the FUNCTION, not the operator); catch instructor phrasings
    (r'\bfunction\s+(?:called\s+)?is\s+(?:a\s+)?null\b',              'function ISNULL'),
    (r'\bcalled\s+is\s+(?:a\s+)?null\b',                               'called ISNULL'),
    (r'\bone\s+called\s+is\s+(?:a\s+)?null\b',                        'one called ISNULL'),
    # GROUP BY — Whisper sometimes writes "group buy" or "group by" correctly
    (r'\bgroup\s+buy\b',                                                'GROUP BY'),
    # ORDER BY — "order buy", "order by"
    (r'\border\s+buy\b',                                                'ORDER BY'),
    # WHERE — "ware", "wear", "where" (Whisper usually gets this right but just in case)
    (r'\b(?:ware|wear)\s+(?=[a-zA-Z_])',                               'WHERE '),
    # INNER JOIN — "inner john", "inner gene"
    (r'\binner\s+john\b',                                               'INNER JOIN'),
]


# General: Whisper mishearings that occur regardless of content_type.
# Negative lookbehind for digits prevents changing "5 pennies" (coin context).
_GENERAL_ASR_FIXES = [
    # "penis" → "pennies" / "pennis" / "penny's" (phonetically similar to Whisper)
    (r'(?<!\d )(?<!\d)\bpenni(?:es|s)?\b', 'penis'),
    (r'(?<!\d )(?<!\d)\bpennis\b',          'penis'),
]


def fix_transcription_errors(text: str, content_type: str) -> str:
    """Fix known Whisper ASR mishearing errors for specific content types."""
    if not text or not content_type:
        return text
    # General fixes run for all content types
    for pattern, replacement in _GENERAL_ASR_FIXES:
        new = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new != text:
            print(f"    [ASR-FIX] '{text.strip()[:80]}' → '{new.strip()[:80]}'", flush=True)
            text = new
    if content_type == "sql":
        for pattern, replacement in _SQL_ASR_FIXES:
            new = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new != text:
                print(f"    [ASR-FIX] '{text.strip()[:80]}' → '{new.strip()[:80]}'", flush=True)
                text = new
    return text


def fix_segments_transcription(segments: list, content_type: str) -> list:
    """Apply fix_transcription_errors to all segments."""
    if not content_type:
        return segments
    fixed = []
    for seg in segments:
        s = dict(seg)
        if s.get("text"):
            s["text"] = fix_transcription_errors(s["text"], content_type)
        fixed.append(s)
    return fixed
