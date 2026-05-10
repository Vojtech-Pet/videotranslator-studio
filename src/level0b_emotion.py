from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
from typing import Any

import numpy as np

_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
_GENDER_MODEL_ID = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
_SAMPLE_RATE = 16000
_MAX_SECONDS = 10.0

_AROUSAL_MAP = {
    "angry": 0.9, "fear": 0.85, "disgust": 0.75,
    "surprised": 0.7, "happy": 0.6, "sad": 0.3, "neutral": 0.2,
}
_VALENCE_MAP = {
    "happy": 0.9, "surprised": 0.6, "neutral": 0.5,
    "sad": 0.2, "fear": 0.15, "disgust": 0.1, "angry": 0.1,
}


def load_emotion_pipeline(device: str = "cuda"):
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "audio-classification",
        model=_MODEL_ID,
        device=0 if device == "cuda" else -1,
    )


def load_gender_pipeline(device: str = "cuda"):
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "audio-classification",
        model=_GENDER_MODEL_ID,
        device=0 if device == "cuda" else -1,
    )


def detect_speaker_genders(
    segments: list[dict],
    source_audio: str | Path,
    gender_pipe,
) -> dict[str, str]:
    """Detect gender for each unique speaker_id from their longest segment."""
    by_speaker: dict[str, dict] = {}
    for seg in segments:
        spk = seg.get("speaker_id", "SPEAKER_00")
        dur = seg["end"] - seg["start"]
        if spk not in by_speaker or dur > (by_speaker[spk]["end"] - by_speaker[spk]["start"]):
            by_speaker[spk] = seg

    genders: dict[str, str] = {}
    for spk, seg in by_speaker.items():
        try:
            audio_np = _cut_segment_audio(source_audio, seg["start"], seg["end"], None, max_seconds=15.0)
            if len(audio_np) < 800:
                genders[spk] = "unknown"
                continue
            result = gender_pipe({"raw": audio_np, "sampling_rate": _SAMPLE_RATE})
            top = max(result, key=lambda x: x["score"])
            genders[spk] = top["label"].lower()
        except Exception:
            genders[spk] = "unknown"
    return genders


def attach_genders(segments: list[dict], genders: dict[str, str]) -> list[dict]:
    for seg in segments:
        spk = seg.get("speaker_id", "SPEAKER_00")
        seg["speaker_gender"] = genders.get(spk, "unknown")
    return segments


def _load_wav_numpy(wav_path: str | Path, max_seconds: float = _MAX_SECONDS) -> np.ndarray:
    """Load WAV segment as 16kHz mono float32 numpy array via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-f", "f32le", "-ac", "1", "-ar", str(_SAMPLE_RATE),
        "-t", str(max_seconds), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def _cut_segment_audio(
    source: str | Path,
    start: float,
    end: float,
    tmp_path: str | Path,
    max_seconds: float = _MAX_SECONDS,
) -> np.ndarray:
    duration = min(end - start, max_seconds)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-t", str(duration),
        "-i", str(source),
        "-f", "f32le", "-ac", "1", "-ar", str(_SAMPLE_RATE), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def _scores_to_emotion(scores: list[dict]) -> dict[str, Any]:
    top = max(scores, key=lambda x: x["score"])
    label = top["label"].lower().strip()
    score_map = {s["label"].lower().strip(): s["score"] for s in scores}
    arousal = sum(_AROUSAL_MAP.get(l, 0.4) * v for l, v in score_map.items())
    valence = sum(_VALENCE_MAP.get(l, 0.4) * v for l, v in score_map.items())
    return {
        "dominant": label,
        "confidence": round(top["score"], 3),
        "arousal": round(arousal, 3),
        "valence": round(valence, 3),
        "scores": {s["label"].lower(): round(s["score"], 3) for s in scores},
    }


def analyze_segments(
    segments: list[dict],
    source_audio: str | Path,
    emotion_pipe,
) -> list[dict]:
    """Add emotion_vector to each segment in-place."""
    for seg in segments:
        try:
            audio_np = _cut_segment_audio(source_audio, seg["start"], seg["end"], None)
            if len(audio_np) < 800:
                seg["emotion"] = {"dominant": "neutral", "confidence": 0.0, "arousal": 0.2, "valence": 0.5, "scores": {}}
                continue
            result = emotion_pipe({"raw": audio_np, "sampling_rate": _SAMPLE_RATE})
            seg["emotion"] = _scores_to_emotion(result)
        except Exception as e:
            seg["emotion"] = {"dominant": "neutral", "confidence": 0.0, "arousal": 0.2, "valence": 0.5, "error": str(e)}
    return segments


def summarize_emotions(segments: list[dict]) -> dict:
    emotions = [s["emotion"]["dominant"] for s in segments if "emotion" in s]
    counts: dict[str, int] = {}
    for e in emotions:
        counts[e] = counts.get(e, 0) + 1
    return {
        "total": len(emotions),
        "distribution": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "avg_arousal": round(sum(s["emotion"]["arousal"] for s in segments if "emotion" in s) / max(len(emotions), 1), 3),
        "avg_valence": round(sum(s["emotion"]["valence"] for s in segments if "emotion" in s) / max(len(emotions), 1), 3),
    }
