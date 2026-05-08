#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import requests
import sys
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.sql_fixer import fix_sql_terms
from core.utils import estimate_cps

ROOT = Path(__file__).resolve().parents[1]
MINI_ROOT = Path(__file__).resolve().parent
LEGACY_ROOT = Path("/home/vojtech/VideoTranslator")


def _prefer_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_AUDIO = _prefer_existing(
    ROOT / "work" / "whisper_input.wav",
    LEGACY_ROOT / "work" / "whisper_input.wav",
)
DEFAULT_INPUT_JSON = MINI_ROOT / "input" / "sql_part_007_part001_sk_segments.json"
DEFAULT_OUTPUT_DIR = MINI_ROOT / "output" / "whisperx_fish_retry"
DEFAULT_FISH_CHECKPOINT = _prefer_existing(
    ROOT / "fish_speech" / "repo" / "checkpoints" / "s2-pro",
    LEGACY_ROOT / "fish_speech" / "repo" / "checkpoints" / "s2-pro",
)
DEFAULT_FISH_SERVER_PYTHON = Path.home() / "miniforge3" / "envs" / "fish_env" / "bin" / "python"
DEFAULT_REFERENCE_WAV = _prefer_existing(
    ROOT / "work" / "s2_reference.wav",
    LEGACY_ROOT / "work" / "s2_reference.wav",
)
DEFAULT_REFERENCE_TEXT = (
    "This is a calm, clear and professional technical narrator voice. "
    "The speech is natural, confident, medium-paced and easy to understand."
)
DEFAULT_TARGET_CPS = 16.0
DEFAULT_HARD_CPS_LIMIT = 17.0
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b"
ENABLE_OLLAMA_REWRITE = True

VOICE_PERSONA = {
    "name": "calm_technical_lecturer",
    "description": "pokojný, jasný, technický hlas, bez prehnanej expresie",
    "pace": "medium",
    "energy": "low_medium",
    "clarity": "high",
    "style": "clear_calm_explanatory",
}

PROSODY_PRESETS = {
    "definition": {
        "instruction": "Hovor pokojne, presne a zrozumiteľne. Dôraz na technické termíny. Bez emócií navyše."
    },
    "example": {
        "instruction": "Hovor prirodzene a mierne živšie, ale stále technicky a jasne."
    },
    "summary": {
        "instruction": "Hovor stručne, prirodzene a uzatvorene. Mierne dôrazne."
    },
    "default": {
        "instruction": "Hovor pokojne, prirodzene a zrozumiteľne."
    },
}


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_punctuation(text: str) -> str:
    out = re.sub(r"\s+([,.:;!?])", r"\1", text or "")
    out = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", out)
    return clean_spaces(out)


def enforce_sql_terms(text: str) -> str:
    fixes = {
        r"\bajznul\b": "ISNULL",
        r"\bkoalesk\b": "COALESCE",
        r"\besíkjúel\b": "SQL",
        r"\besikjuel\b": "SQL",
        r"\besikjul\b": "SQL",
        r"\bíz null\b": "IS NULL",
        r"\biz null\b": "IS NULL",
        r"\bíz not null\b": "IS NOT NULL",
        r"\biz not null\b": "IS NOT NULL",
    }
    out = text or ""
    for pattern, replacement in fixes.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return clean_punctuation(out)


def enforce_spoken_safety(text: str) -> str:
    fixes = [
        (r"\b(?:naš\w*\s+){1,2}stola\b", "našej tabuľky"),
        (r"\bnášho stola\b", "našej tabuľky"),
        (r"\bnasho stola\b", "našej tabuľky"),
        (r"\b(?:rob[ií]me|urob[ií]me)\s+hlbok[ýy]\s+ponor\b", "pozrieme sa"),
        (r"\bhlboký ponor\b", "pozrieme sa"),
        (r"\bhlboky ponor\b", "pozrieme sa"),
        (r"\bdeep dive\b", "pozrieme sa"),
    ]
    out = text or ""
    for pattern, replacement in fixes:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    out = re.sub(r"\bv\s+našej tabuľky\b", "v našej tabuľke", out, flags=re.IGNORECASE)
    return clean_punctuation(out)


def load_optional_module(name: str):
    try:
        return __import__(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python module '{name}'. Run this script with "
            "/home/vojtech/miniforge3/envs/musetalk_env/bin/python or install the dependency there."
        ) from exc


@lru_cache(maxsize=4)
def load_faster_whisper_model(model_size: str, device: str, compute_type: str):
    fw = load_optional_module("faster_whisper")
    return fw.WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_faster_whisper(
    audio_path: Path,
    *,
    model_size: str,
    device: str,
    compute_type: str,
    language: str,
) -> list[dict[str, Any]]:
    model = load_faster_whisper_model(model_size, device, compute_type)
    segments, _info = model.transcribe(str(audio_path), beam_size=5, language=language)
    out: list[dict[str, Any]] = []
    for seg in segments:
        out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text_src": clean_spaces(seg.text.strip()),
            }
        )
    return out


def align_whisperx(
    audio_path: Path,
    fw_segments: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    whisperx = load_optional_module("whisperx")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    del model

    wx_segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text_src"]} for seg in fw_segments]
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(wx_segments, model_a, metadata, audio, device)

    out: list[dict[str, Any]] = []
    for seg in aligned["segments"]:
        out.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text_src": clean_spaces(seg["text"].strip()),
            }
        )
    return out


def simple_rewrite(text: str, slot: float, *, aggressive: bool = False) -> str:
    t = fix_sql_terms(text)
    rules = [
        (r"\bV poriadku, priatelia, takže teraz\b", "Teraz"),
        (r"\burobíme hlboký ponor do\b", "sa pozrieme na"),
        (r"\bdeep dive do\b", "sa pozrieme na"),
        (r"\bO tom, ako zaobchádzať s\b", "Ako pracovať s"),
        (r"\bv našich údajoch\b", "v dátach"),
        (r"\bnášho stola\b", "našej tabuľky"),
        (r"\bPoďme si dať príklad\b", "Pozrime sa na príklad"),
        (r"\bMôžeme ísť a použiť\b", "Môžeme použiť"),
        (r"\bTakže\b", ""),
    ]
    for pat, repl in rules:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)

    if aggressive:
        shorten_rules = [
            (r"\bteraz\b", ""),
            (r"\bskutočne\b", ""),
            (r"\bveľmi\b", ""),
            (r"\bv našich\b", ""),
            (r"\bnašich\b", ""),
            (r"\bpovedzme, že\b", ""),
        ]
        for pat, repl in shorten_rules:
            t2 = re.sub(pat, repl, t, flags=re.IGNORECASE)
            if len(t2) < len(t):
                t = t2

    return clean_punctuation(t)


def classify_segment_type(text: str) -> str:
    low = clean_spaces(text).lower()
    if any(x in low for x in ["syntax", "funkcia", "is null", "isnull", "coalesce", "null hodnota"]):
        return "definition"
    if any(x in low for x in ["príklad", "priklad", "v tomto scenári", "v tomto scenari", "máme dve objednávky", "mame dve objednavky", "pozrime sa"]):
        return "example"
    if any(x in low for x in ["prehľad", "prehlad", "celkový obraz", "celkovy obraz", "zhrnutie"]):
        return "summary"
    return "default"


def target_cps_for_type(seg_type: str) -> float:
    if seg_type == "definition":
        return 15.5
    if seg_type == "example":
        return 16.5
    if seg_type == "summary":
        return 16.8
    return DEFAULT_TARGET_CPS


def hard_cps_for_type(seg_type: str) -> float:
    return target_cps_for_type(seg_type) + 1.0


def effective_cps_limits(seg_type: str, *, target_cps: float, hard_cps_limit: float) -> tuple[float, float]:
    typed_target = target_cps_for_type(seg_type)
    typed_hard = hard_cps_for_type(seg_type)
    if abs(float(target_cps) - DEFAULT_TARGET_CPS) < 1e-6:
        effective_target = typed_target
    else:
        effective_target = min(float(target_cps), typed_target)
    if abs(float(hard_cps_limit) - DEFAULT_HARD_CPS_LIMIT) < 1e-6:
        effective_hard = typed_hard
    else:
        effective_hard = min(float(hard_cps_limit), typed_hard)
    return effective_target, effective_hard


def choose_rewrite_mode(slot: float, text: str, *, target_cps: float, hard_cps_limit: float) -> str:
    cps = estimate_cps(text, slot)
    if cps <= target_cps:
        return "balanced"
    if cps <= hard_cps_limit:
        return "dub_friendly"
    return "aggressive"


def build_rewrite_prompt(text_en: str, slot: float, mode: str, seg_type: str) -> str:
    prosody_instruction = PROSODY_PRESETS.get(seg_type, PROSODY_PRESETS["default"])["instruction"]
    return f"""
Si expert na slovenský dabing technických videí.

Prelož vetu z angličtiny do prirodzenej slovenskej hovorenej formy.

HLASOVÝ PROFIL:
{VOICE_PERSONA["description"]}

PROSODY:
{prosody_instruction}

KRITICKÉ PRAVIDLÁ:
- Zachovaj význam.
- Zachovaj presne tieto výrazy:
  SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, TRUE, FALSE, BOOLEAN
- Nikdy nepíš:
  ajznul, koalesk, esíkjúel
- Nepoužívaj doslovné preklady.
- Nepíš neprirodzené výrazy ako:
  "hlboký ponor", "nášho stola", "odíde a nahradí"
- Výstup musí znieť prirodzene pri čítaní nahlas.
- Ak veta obsahuje SQL funkciu, uprednostni krátku a priamu formuláciu.
- Výstup musí rešpektovať slot.

ŠTÝL:
- krátke vety
- jasné
- vhodné pre voiceover

REŽIM: {mode}
SLOT: {slot:.2f} sekundy

TEXT:
"{text_en}"

Vráť iba finálnu vetu.
""".strip()


def ollama_generate(prompt: str, *, ollama_url: str, ollama_model: str, timeout: int = 120) -> str:
    response = requests.post(
        ollama_url,
        json={
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        return str(payload["response"]).strip()
    except (KeyError, TypeError):
        return ""


def rewrite_for_tts(
    seg: dict[str, Any],
    slot: float,
    *,
    target_cps: float,
    hard_cps_limit: float,
    enable_ollama_rewrite: bool,
    ollama_url: str,
    ollama_model: str,
) -> tuple[str, str, str]:
    source_text = clean_spaces(str(seg.get("text_src") or seg.get("text") or seg.get("tts_input") or ""))
    fallback_text = clean_spaces(str(seg.get("text") or seg.get("tts_input") or source_text or ""))
    seg_type = classify_segment_type(source_text or fallback_text)
    effective_target_cps, effective_hard_cps = effective_cps_limits(
        seg_type,
        target_cps=target_cps,
        hard_cps_limit=hard_cps_limit,
    )
    mode = choose_rewrite_mode(
        slot,
        source_text or fallback_text,
        target_cps=effective_target_cps,
        hard_cps_limit=effective_hard_cps,
    )

    if enable_ollama_rewrite:
        prompt = build_rewrite_prompt(source_text or fallback_text, slot, mode, seg_type)
        try:
            out = ollama_generate(prompt, ollama_url=ollama_url, ollama_model=ollama_model)
            out = fix_sql_terms(out)
            out = clean_punctuation(out)
            out = enforce_sql_terms(out)
            out = enforce_spoken_safety(out)
            return out, mode, "ollama"
        except Exception as exc:
            print(f"[WARN] Ollama rewrite failed: {exc}", flush=True)

    fallback_source = fallback_text or source_text
    fallback = simple_rewrite(fallback_source, slot, aggressive=(mode == "aggressive"))
    fallback = enforce_spoken_safety(enforce_sql_terms(fix_sql_terms(fallback)))
    return fallback, mode, "fallback"


def write_stub_wav(out_wav: Path, duration_s: float, sr: int = 24000) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(sr * max(duration_s, 0.15)))
    silence = b"\x00\x00" * frames
    with wave.open(str(out_wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(silence)


def resolve_reference(
    *,
    audio_path: Path | None,
    segments: list[dict[str, Any]],
    reference_audio: Path | None,
    reference_text: str,
    output_dir: Path,
    tts_mode: str,
) -> tuple[str | None, str | None]:
    if tts_mode == "stub":
        return None, None

    if reference_audio is not None and reference_audio.exists() and reference_text.strip():
        return str(reference_audio), reference_text.strip()

    if audio_path is None or not audio_path.exists():
        return None, None

    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import tts  # type: ignore

    auto_ref_path = output_dir / "reference_auto.wav"
    prepared = tts.prepare_s2_pro_reference(
        audio_path=audio_path,
        output_path=auto_ref_path,
        segments=segments,
        auto=True,
    )
    if prepared is None:
        return None, None
    return prepared


def fish_tts(
    *,
    text: str,
    out_wav: Path,
    checkpoint_path: Path,
    reference_audio_path: str | None,
    reference_text: str | None,
    seg_type: str,
    server_python: Path,
    server_port: int,
    mode: str,
    slot: float,
) -> Path:
    if mode == "stub":
        estimated_duration = max(0.2, min(slot * 0.98, len(text) / max(8.0, estimate_cps(text, max(slot, 0.01)))))
        write_stub_wav(out_wav, estimated_duration)
        return out_wav

    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import tts  # type: ignore

    prosody_instruction = PROSODY_PRESETS.get(seg_type, PROSODY_PRESETS["default"])["instruction"]
    base_reference_text = clean_spaces(reference_text or DEFAULT_REFERENCE_TEXT)
    enriched_reference_text = clean_spaces(
        f"{base_reference_text} {VOICE_PERSONA['description']}. {prosody_instruction}"
    )

    return tts.s2_pro_tts(
        text=text,
        output_path=out_wav,
        checkpoint_path=str(checkpoint_path),
        reference_audio_path=reference_audio_path,
        reference_text=enriched_reference_text,
        server_python=str(server_python),
        server_port=server_port,
    )


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return frames / max(1, rate)


def asr_backcheck(
    audio_path: Path,
    *,
    model_size: str,
    device: str,
    compute_type: str,
    language: str,
) -> str:
    model = load_faster_whisper_model(model_size, device, compute_type)
    segments, _info = model.transcribe(str(audio_path), beam_size=5, language=language)
    return clean_spaces(" ".join(seg.text.strip() for seg in segments))


def asr_match_score(expected: str, got: str) -> float:
    expected_words = set(clean_spaces(expected).lower().split())
    got_words = set(clean_spaces(got).lower().split())
    if not expected_words:
        return 0.0
    overlap = len(expected_words & got_words) / len(expected_words)
    return round(overlap, 3)


def spoken_score(text: str) -> float:
    low = text.lower()
    score = 1.0
    for bad in ["hlboký ponor", "našho stola", "nášho stola", "odíde a nahradí", "go a"]:
        if bad in low:
            score -= 0.3
    return max(0.0, round(score, 3))


def term_score(text: str) -> float:
    low = text.lower()
    score = 1.0
    for bad in ["ajznul", "koalesk", "esikjuel", "esikjul", "iz not null", "izn ot null"]:
        if bad in low:
            score -= 0.4
    return max(0.0, round(score, 3))


def duration_score(delta: float) -> float:
    absolute_delta = abs(delta)
    if absolute_delta <= 0.20:
        return 1.0
    if absolute_delta <= 0.40:
        return 0.8
    if absolute_delta <= 0.70:
        return 0.5
    return 0.2


def cps_score(cps: float, *, target: float, hard_limit: float) -> float:
    if cps <= target:
        return 1.0
    if cps <= hard_limit:
        return 0.8
    return 0.5


def choose_retry_mode(seg: dict[str, Any], *, hard_cps_limit: float) -> str | None:
    if float(seg["term_score"]) < 1.0:
        return "locked_terms"
    if float(seg["cps"]) > hard_cps_limit:
        return "aggressive_shorten"
    if float(seg["asr_match_score"]) < 0.75:
        return "pronunciation_safe"
    if float(seg["duration_score"]) < 0.8:
        return "aggressive_shorten"
    return None


def final_score(seg: dict[str, Any]) -> float:
    return round(
        0.35 * float(seg["term_score"])
        + 0.25 * float(seg["duration_score"])
        + 0.20 * float(seg["cps_score"])
        + 0.10 * float(seg["spoken_score"])
        + 0.10 * float(seg["asr_match_score"]),
        3,
    )


def load_segments_from_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read segments from {path}: {exc}") from exc
    if isinstance(data, dict) and "segments" in data:
        return list(data["segments"]), dict(data)
    if isinstance(data, list):
        return list(data), None
    raise ValueError(f"Unsupported JSON format: {path}")


def save_segments(path: Path, segments: list[dict[str, Any]], wrapper: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if wrapper is not None:
        payload = dict(wrapper)
        payload["segments"] = segments
    else:
        payload = segments
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_segments(
    segments: list[dict[str, Any]],
    *,
    audio_path: Path | None,
    output_dir: Path,
    tts_mode: str,
    checkpoint_path: Path,
    reference_audio: Path | None,
    reference_text: str,
    server_python: Path,
    server_port: int,
    target_cps: float,
    hard_cps_limit: float,
    enable_ollama_rewrite: bool,
    ollama_url: str,
    ollama_model: str,
    enable_asr_backcheck: bool,
    enable_retry: bool,
    backcheck_model_size: str,
    device: str,
    compute_type: str,
    target_language: str,
    limit_segments: int,
) -> list[dict[str, Any]]:
    wav_dir = output_dir / "wav_segments"
    wav_dir.mkdir(parents=True, exist_ok=True)

    if limit_segments > 0:
        segments = segments[:limit_segments]

    ref_audio_path, ref_text = resolve_reference(
        audio_path=audio_path,
        segments=segments,
        reference_audio=reference_audio,
        reference_text=reference_text,
        output_dir=output_dir,
        tts_mode=tts_mode,
    )

    results: list[dict[str, Any]] = []
    for index, seg in enumerate(segments):
        slot = max(0.01, float(seg.get("end", 0.0) or 0.0) - float(seg.get("start", 0.0) or 0.0))
        source_for_type = clean_spaces(str(seg.get("text_src") or seg.get("text") or ""))
        source_seg_type = classify_segment_type(source_for_type)
        effective_target_cps, effective_hard_cps = effective_cps_limits(
            source_seg_type,
            target_cps=target_cps,
            hard_cps_limit=hard_cps_limit,
        )
        tts_input, rewrite_mode, rewrite_engine = rewrite_for_tts(
            seg,
            slot,
            target_cps=effective_target_cps,
            hard_cps_limit=effective_hard_cps,
            enable_ollama_rewrite=enable_ollama_rewrite,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )
        seg_type = classify_segment_type(tts_input)
        cps = round(estimate_cps(tts_input, slot), 2)

        out_wav = wav_dir / f"seg_{index:04d}.wav"
        fish_tts(
            text=tts_input,
            out_wav=out_wav,
            checkpoint_path=checkpoint_path,
            reference_audio_path=ref_audio_path,
            reference_text=ref_text,
            seg_type=seg_type,
            server_python=server_python,
            server_port=server_port,
            mode=tts_mode,
            slot=slot,
        )

        wav_dur = wav_duration_seconds(out_wav)
        delta = round(wav_dur - slot, 3)
        backcheck = ""
        backcheck_score = 1.0
        if enable_asr_backcheck and tts_mode != "stub":
            backcheck = asr_backcheck(
                out_wav,
                model_size=backcheck_model_size,
                device=device,
                compute_type=compute_type,
                language=target_language,
            )
            backcheck_score = asr_match_score(tts_input, backcheck)

        row = {
            **seg,
            "slot": round(slot, 3),
            "segment_type": seg_type,
            "tts_input": tts_input,
            "rewrite_mode": rewrite_mode,
            "rewrite_engine": rewrite_engine,
            "wav": str(out_wav),
            "wav_duration": round(wav_dur, 3),
            "delta": delta,
            "cps": cps,
            "duration_score": duration_score(delta),
            "cps_score": cps_score(cps, target=effective_target_cps, hard_limit=effective_hard_cps),
            "target_cps": round(effective_target_cps, 2),
            "hard_cps_limit": round(effective_hard_cps, 2),
            "spoken_score": spoken_score(tts_input),
            "term_score": term_score(tts_input),
            "asr_backcheck": backcheck,
            "asr_match_score": backcheck_score,
            "retry_mode": None,
            "accepted": False,
        }
        row["score"] = final_score(row)

        if enable_retry:
            retry_mode = choose_retry_mode(row, hard_cps_limit=effective_hard_cps)
            if retry_mode:
                row["retry_mode"] = retry_mode
                retry_text = tts_input
                retry_engine = row["rewrite_engine"]

                if enable_ollama_rewrite:
                    retry_prompt = f"""
Si expert na slovenský technický dabing.

Prepíš vetu tak, aby opravila tento problém:
{retry_mode}

Pravidlá:
- Zachovaj význam.
- Zachovaj presne: SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, TRUE, FALSE, BOOLEAN
- Nepíš ich foneticky.
- Nikdy nepíš: ajznul, koalesk, esíkjúel.
- Nepíš „nášho stola“ ani „hlboký ponor“.
- Ak veta obsahuje SQL funkciu, uprednostni krátku a priamu formuláciu.
- Veta musí byť vhodná pre hlasné čítanie.
- Musí sa zmestiť do slotu {slot:.2f} sekundy.

Veta:
"{tts_input}"

Vráť iba finálnu slovenskú vetu.
""".strip()
                    try:
                        retry_text = ollama_generate(
                            retry_prompt,
                            ollama_url=ollama_url,
                            ollama_model=ollama_model,
                        )
                        retry_text = fix_sql_terms(retry_text)
                        retry_text = clean_punctuation(retry_text)
                        retry_text = enforce_sql_terms(retry_text)
                        retry_text = enforce_spoken_safety(retry_text)
                        retry_engine = "ollama_retry"
                    except Exception as exc:
                        print(f"[WARN] Ollama retry failed: {exc}", flush=True)
                        if retry_mode in {"aggressive_shorten", "pronunciation_safe"}:
                            retry_text = simple_rewrite(tts_input, slot, aggressive=True)
                            retry_engine = "fallback_retry"
                        elif retry_mode == "locked_terms":
                            retry_text = fix_sql_terms(tts_input)
                            retry_engine = "fallback_retry"
                else:
                    if retry_mode in {"aggressive_shorten", "pronunciation_safe"}:
                        retry_text = simple_rewrite(tts_input, slot, aggressive=True)
                        retry_engine = "fallback_retry"
                    elif retry_mode == "locked_terms":
                        retry_text = fix_sql_terms(tts_input)
                        retry_engine = "fallback_retry"
                retry_text = enforce_spoken_safety(enforce_sql_terms(fix_sql_terms(clean_punctuation(retry_text))))

                retry_wav = wav_dir / f"seg_{index:04d}_retry.wav"
                retry_seg_type = classify_segment_type(retry_text)
                fish_tts(
                    text=retry_text,
                    out_wav=retry_wav,
                    checkpoint_path=checkpoint_path,
                    reference_audio_path=ref_audio_path,
                    reference_text=ref_text,
                    seg_type=retry_seg_type,
                    server_python=server_python,
                    server_port=server_port,
                    mode=tts_mode,
                    slot=slot,
                )

                retry_wav_dur = wav_duration_seconds(retry_wav)
                retry_delta = round(retry_wav_dur - slot, 3)
                retry_cps = round(estimate_cps(retry_text, slot), 2)
                retry_target_cps, retry_hard_cps = effective_cps_limits(
                    retry_seg_type,
                    target_cps=target_cps,
                    hard_cps_limit=hard_cps_limit,
                )
                retry_backcheck = ""
                retry_backcheck_score = 1.0
                if enable_asr_backcheck and tts_mode != "stub":
                    retry_backcheck = asr_backcheck(
                        retry_wav,
                        model_size=backcheck_model_size,
                        device=device,
                        compute_type=compute_type,
                        language=target_language,
                    )
                    retry_backcheck_score = asr_match_score(retry_text, retry_backcheck)

                retry_row = {
                    **row,
                    "segment_type": retry_seg_type,
                    "tts_input": retry_text,
                    "rewrite_mode": retry_mode,
                    "rewrite_engine": retry_engine,
                    "wav": str(retry_wav),
                    "wav_duration": round(retry_wav_dur, 3),
                    "delta": retry_delta,
                    "cps": retry_cps,
                    "duration_score": duration_score(retry_delta),
                    "cps_score": cps_score(
                        retry_cps,
                        target=retry_target_cps,
                        hard_limit=retry_hard_cps,
                    ),
                    "target_cps": round(retry_target_cps, 2),
                    "hard_cps_limit": round(retry_hard_cps, 2),
                    "spoken_score": spoken_score(retry_text),
                    "term_score": term_score(retry_text),
                    "asr_backcheck": retry_backcheck,
                    "asr_match_score": retry_backcheck_score,
                }
                retry_row["score"] = final_score(retry_row)
                if float(retry_row["score"]) > float(row["score"]):
                    row = retry_row

        row["accepted"] = bool(float(row["score"]) >= 0.82)
        results.append(row)
        print(
            f"Segment {index + 1}/{len(segments)} | score={row['score']} | "
            f"accepted={row['accepted']} | wav={Path(row['wav']).name}",
            flush=True,
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="faster-whisper + WhisperX + Fish S2 retry skeleton")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO), help="Input audio path for ASR")
    parser.add_argument("--input-json", default="", help="Optional pre-segmented JSON to skip ASR")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--device", default="cuda", help="ASR device")
    parser.add_argument("--fw-model", default="large-v3", help="faster-whisper model size")
    parser.add_argument("--fw-compute-type", default="float16", help="faster-whisper compute type")
    parser.add_argument("--whisperx-model", default="large-v2", help="WhisperX model")
    parser.add_argument("--whisperx-compute-type", default="float16", help="WhisperX compute type")
    parser.add_argument("--whisperx-batch-size", type=int, default=16, help="WhisperX batch size")
    parser.add_argument("--source-language", default="en", help="Source ASR language")
    parser.add_argument("--target-language", default="sk", help="Target language for back-check")
    parser.add_argument("--disable-alignment", action="store_true", help="Skip WhisperX alignment")
    parser.add_argument("--disable-asr-backcheck", action="store_true", help="Skip TTS ASR back-check")
    parser.add_argument("--disable-retry", action="store_true", help="Disable retry loop")
    parser.add_argument("--tts-mode", choices=["real", "stub"], default="real", help="Use real S2-Pro or stub WAVs")
    parser.add_argument("--checkpoint", default=str(DEFAULT_FISH_CHECKPOINT), help="Fish S2-Pro checkpoint path")
    parser.add_argument("--server-python", default=str(DEFAULT_FISH_SERVER_PYTHON), help="Fish server python path")
    parser.add_argument("--server-port", type=int, default=8091, help="Fish server port")
    parser.add_argument("--reference-audio", default=str(DEFAULT_REFERENCE_WAV), help="Reference WAV path")
    parser.add_argument("--reference-text", default="", help="Reference transcript for voice cloning")
    parser.add_argument("--target-cps", type=float, default=DEFAULT_TARGET_CPS, help="Target CPS")
    parser.add_argument("--hard-cps-limit", type=float, default=DEFAULT_HARD_CPS_LIMIT, help="Hard CPS limit")
    parser.add_argument("--backcheck-model", default="large-v3", help="faster-whisper model for back-check")
    parser.add_argument("--limit-segments", type=int, default=0, help="Limit processed segments for debugging")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama generate endpoint")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help="Ollama model for EN->SK rewrite")
    parser.add_argument(
        "--disable-ollama-rewrite",
        action="store_true",
        help="Disable Ollama rewrite and use heuristic fallback only",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists() and args.tts_mode == "real":
        raise FileNotFoundError(f"Fish S2-Pro checkpoint not found: {checkpoint}")

    reference_audio = Path(args.reference_audio).expanduser().resolve() if args.reference_audio else None
    audio_path = Path(args.audio).expanduser().resolve() if args.audio else None
    input_json = Path(args.input_json).expanduser().resolve() if args.input_json else None

    if input_json is not None:
        print(f"Loading segments from JSON: {input_json}", flush=True)
        segments, wrapper = load_segments_from_json(input_json)
    else:
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError("Provide --audio or --input-json")
        print("1/4 ASR faster-whisper...", flush=True)
        segments = transcribe_faster_whisper(
            audio_path,
            model_size=args.fw_model,
            device=args.device,
            compute_type=args.fw_compute_type,
            language=args.source_language,
        )
        if args.disable_alignment:
            print("2/4 WhisperX alignment skipped", flush=True)
        else:
            print("2/4 WhisperX alignment...", flush=True)
            segments = align_whisperx(
                audio_path,
                segments,
                model_name=args.whisperx_model,
                device=args.device,
                compute_type=args.whisperx_compute_type,
                language=args.source_language,
                batch_size=args.whisperx_batch_size,
            )
        wrapper = {"segments": segments}

    print("3/4 Rewrite + TTS + audit...", flush=True)
    results = process_segments(
        segments,
        audio_path=audio_path if audio_path and audio_path.exists() else None,
        output_dir=output_dir,
        tts_mode=args.tts_mode,
        checkpoint_path=checkpoint,
        reference_audio=reference_audio if reference_audio and reference_audio.exists() else None,
        reference_text=args.reference_text or DEFAULT_REFERENCE_TEXT,
        server_python=Path(args.server_python).expanduser().resolve(),
        server_port=args.server_port,
        target_cps=args.target_cps,
        hard_cps_limit=args.hard_cps_limit,
        enable_ollama_rewrite=ENABLE_OLLAMA_REWRITE and not args.disable_ollama_rewrite,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        enable_asr_backcheck=not args.disable_asr_backcheck,
        enable_retry=not args.disable_retry,
        backcheck_model_size=args.backcheck_model,
        device=args.device,
        compute_type=args.fw_compute_type,
        target_language=args.target_language,
        limit_segments=args.limit_segments,
    )

    accepted = sum(1 for row in results if row.get("accepted"))
    summary = {
        "segments": len(results),
        "accepted": accepted,
        "needs_retry": len(results) - accepted,
        "tts_mode": args.tts_mode,
        "alignment_enabled": not args.disable_alignment,
        "asr_backcheck_enabled": not args.disable_asr_backcheck,
        "ollama_rewrite_enabled": ENABLE_OLLAMA_REWRITE and not args.disable_ollama_rewrite,
        "ollama_model": args.ollama_model,
    }
    payload = dict(wrapper or {})
    payload["segments"] = results
    payload["summary"] = summary
    output_json = output_dir / "result_whisperx_fish_retry.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("4/4 Done", flush=True)
    print(f"Accepted: {accepted}/{len(results)}", flush=True)
    print(f"Hotovo: {output_json}", flush=True)


if __name__ == "__main__":
    main()
