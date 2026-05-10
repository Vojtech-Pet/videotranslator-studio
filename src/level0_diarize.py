from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)

import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


_DEVICE = "cuda"
_BATCH_SIZE = 16
_COMPUTE_TYPE = "float16"
_MIN_REF_SECONDS = 5.0
_MAX_REF_SECONDS = 20.0


def _resolve_hf_token(token: str) -> str:
    if token:
        return token
    for path in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if path.exists():
            return path.read_text().strip()
    return ""


def transcribe_and_diarize(
    audio_path: str | Path,
    hf_token: str,
    language: str = "en",
    device: str = _DEVICE,
    batch_size: int = _BATCH_SIZE,
    compute_type: str = _COMPUTE_TYPE,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """Run WhisperX transcription + word alignment + Pyannote diarization.

    Returns list of segments with speaker_id assigned.
    """
    audio_path = str(audio_path)
    hf_token = _resolve_hf_token(hf_token)
    audio = whisperx.load_audio(str(audio_path))

    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    del model

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)
    del align_model

    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        token=hf_token,
        device=device,
    )
    # lower min_cluster_size so short speaker turns (classroom responses, brief replies) are detected
    diarize_model.model.instantiate({
        "segmentation": {"min_duration_off": 0.0},
        "clustering": {"method": "centroid", "min_cluster_size": 2, "threshold": 0.7045654963945799},
    })
    diarize_df = diarize_model(
        str(audio_path),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    return result["segments"], diarize_df


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_segments(whisperx_segments: list[dict], diarize_df=None) -> list[dict]:
    """Convert WhisperX segments to pipeline segment format.

    Uses segment-level overlap with diarization DataFrame for reliable
    speaker assignment — more robust than word-level majority vote for
    short SPEAKER turns.
    """
    out = []
    for seg in whisperx_segments:
        start, end = seg["start"], seg["end"]
        speaker_id = _assign_speaker_from_df(start, end, diarize_df) or "SPEAKER_00"
        out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text_src": seg.get("text", "").strip(),
            "speaker_id": speaker_id,
        })
    return out


def _assign_speaker_from_df(start: float, end: float, df) -> str | None:
    if df is None or df.empty:
        return None
    overlap_by_spk: dict[str, float] = {}
    for _, row in df.iterrows():
        spk = row.get("speaker") or row.get("label")
        if not spk:
            continue
        ov = _overlap(start, end, float(row["start"]), float(row["end"]))
        if ov > 0:
            overlap_by_spk[spk] = overlap_by_spk.get(spk, 0.0) + ov
    if not overlap_by_spk:
        return None
    return max(overlap_by_spk, key=lambda k: overlap_by_spk[k])


def extract_speaker_references(
    audio_path: str | Path,
    segments: list[dict],
    output_dir: str | Path,
) -> dict[str, str]:
    """Extract one clean reference audio clip per speaker using ffmpeg.

    Returns mapping speaker_id -> path.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # collect candidate segments per speaker, pick longest within [5s, 20s]
    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        spk = seg["speaker_id"]
        by_speaker.setdefault(spk, []).append(seg)

    refs: dict[str, str] = {}
    for spk, spk_segs in by_speaker.items():
        best = _pick_best_ref_segment(spk_segs)
        if best is None:
            continue
        out_path = output_dir / f"ref_{spk}.wav"
        _ffmpeg_cut(audio_path, best["start"], best["end"], out_path)
        refs[spk] = str(out_path)

    return refs


def _pick_best_ref_segment(segs: list[dict]) -> dict | None:
    candidates = [s for s in segs if (s["end"] - s["start"]) >= _MIN_REF_SECONDS]
    if not candidates:
        candidates = segs
    # prefer segments closest to _MAX_REF_SECONDS but not over
    under = [s for s in candidates if (s["end"] - s["start"]) <= _MAX_REF_SECONDS]
    pool = under if under else candidates
    return max(pool, key=lambda s: s["end"] - s["start"])


def _ffmpeg_cut(src: Path, start: float, end: float, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(src),
            "-ar", "22050",
            "-ac", "1",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def attach_refs(segments: list[dict], refs: dict[str, str]) -> list[dict]:
    for seg in segments:
        spk = seg["speaker_id"]
        if spk in refs:
            seg["speaker_ref_audio"] = refs[spk]
    return segments


def summarize_level0(segments: list[dict], refs: dict[str, str]) -> dict:
    speakers = sorted({s["speaker_id"] for s in segments})
    return {
        "total_segments": len(segments),
        "speakers": speakers,
        "speaker_count": len(speakers),
        "ref_audios": refs,
        "segments_without_speaker": sum(1 for s in segments if not s.get("speaker_id")),
    }
