#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for entry in (ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from src.utils_text import normalize_text  # noqa: E402


SOURCES: dict[str, dict[str, str]] = {
    "miso_sql_01": {
        "url": "https://www.youtube.com/watch?v=_AIQF-ij-2M",
        "title": "Kurz SQL (PostgreSQL) | Lekcia 1 | Uvod",
        "channel": "Informatika s Misom",
        "default_bucket": "core_clean",
    },
    "itacademy_sql_01": {
        "url": "https://www.youtube.com/watch?v=cpHhb-Y6GMc",
        "title": "Online Kurz SQL a MySQL I. Zaciatocnik",
        "channel": "IT Academy",
        "default_bucket": "hard_examples",
    },
    "itacademy_db_analyst": {
        "url": "https://www.youtube.com/watch?v=87nZQczlUFo",
        "title": "Online Kurz Databazy a SQL - Junior DB Analyst",
        "channel": "IT Academy",
        "default_bucket": "technical_sql",
    },
}

TECHNICAL_TERMS = (
    "sql",
    "postgresql",
    "mysql",
    "databaz",
    "tabulk",
    "stlpec",
    "riadok",
    "null",
    "is null",
    "is not null",
    "isnull",
    "coalesce",
    "nullif",
    "select",
    "where",
    "join",
    "dotaz",
)

PROMO_PATTERNS = (
    "vitajte na kanali",
    "tento kanal je urceny",
    "na tomto kanali najdete",
    "okrem toho tu najdete",
    "kurzy programovania",
    "absolutnych zaciatocnikov",
    "objektov orientovaneho programovania",
    "napiste ich pod video",
    "kontaktujte ma",
    "socialnych sietach",
    "financne podporit",
    "odber",
    "predplatne",
    "kurz najdes tu",
)

ADMIN_PATTERNS = (
    "prezenc",
    "prezenck",
    "heslo na tabuli",
    "na tabuli",
    "vypnite mikrofon",
    "mikrofon",
    "kamera",
    "chat",
    "domena",
    "lomitko",
    "otvorte prehliadac",
    "pracovnu zmluvu",
    "sedite na stolicke",
)

BAD_PATTERNS = PROMO_PATTERNS + ADMIN_PATTERNS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ingest Slovak YouTube sources for S2 dataset curation.")
    ap.add_argument("--out-dir", default="level21_ingest", help="Output directory")
    ap.add_argument(
        "--source-ids",
        nargs="*",
        default=None,
        help=f"Subset of source ids. Available: {', '.join(SOURCES)}",
    )
    ap.add_argument("--list-sources", action="store_true", help="Print bundled sources and exit")
    ap.add_argument("--urls", nargs="*", default=[], help="Arbitrary YouTube URLs to ingest")
    ap.add_argument("--local-files", nargs="*", default=[], help="Local video/audio files to ingest")
    ap.add_argument("--download-sections", default="", help="Optional yt-dlp download sections, e.g. '*0-35'")
    ap.add_argument("--reuse-audio", action="store_true", help="Reuse existing downloaded wav if present")
    ap.add_argument("--whisper-model", default="large-v3-turbo", help="faster-whisper model name")
    ap.add_argument("--download-root", default="", help="Optional faster-whisper download root")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="ASR device")
    ap.add_argument("--min-duration", type=float, default=1.5, help="Minimum segment duration")
    ap.add_argument("--max-duration", type=float, default=12.0, help="Maximum segment duration")
    ap.add_argument("--min-chars", type=int, default=20, help="Minimum cleaned text length")
    ap.add_argument("--vad-threshold", type=float, default=0.35, help="Silero VAD threshold")
    ap.add_argument("--min-silence-ms", type=int, default=350, help="Min silence for VAD")
    ap.add_argument("--min-speech-ms", type=int, default=250, help="Min speech duration for VAD")
    ap.add_argument("--speech-pad-ms", type=int, default=120, help="Speech pad for VAD")
    ap.add_argument("--max-speech-s", type=float, default=25.0, help="Max speech split length for VAD")
    return ap


def _asr_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _normalize_for_rules(text: str) -> str:
    out = normalize_text(text).lower()
    replacements = {
        "á": "a",
        "ä": "a",
        "č": "c",
        "ď": "d",
        "é": "e",
        "ě": "e",
        "í": "i",
        "ĺ": "l",
        "ľ": "l",
        "ň": "n",
        "ó": "o",
        "ô": "o",
        "ŕ": "r",
        "ř": "r",
        "š": "s",
        "ť": "t",
        "ú": "u",
        "ů": "u",
        "ý": "y",
        "ž": "z",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def download_audio(source_id: str, *, out_dir: Path, section: str, reuse_audio: bool) -> Path:
    source = SOURCES[source_id]
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{source_id}.wav"
    if reuse_audio and wav_path.exists():
        return wav_path

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-x",
        "--audio-format",
        "wav",
        "-o",
        str(audio_dir / f"{source_id}.%(ext)s"),
    ]
    if section:
        cmd.extend(["--download-sections", section])
    cmd.append(source["url"])
    subprocess.run(cmd, check=True, timeout=600)
    if not wav_path.exists():
        raise FileNotFoundError(f"yt-dlp did not create expected wav: {wav_path}")
    return wav_path


def download_url(url: str, source_id: str, *, out_dir: Path, section: str, reuse_audio: bool) -> Path:
    """Download an arbitrary YouTube URL (not in the SOURCES dict)."""
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{source_id}.wav"
    if reuse_audio and wav_path.exists():
        return wav_path
    cmd = [
        "yt-dlp", "--no-playlist", "-x", "--audio-format", "wav",
        "-o", str(audio_dir / f"{source_id}.%(ext)s"),
    ]
    if section:
        cmd.extend(["--download-sections", section])
    cmd.append(url)
    subprocess.run(cmd, check=True, timeout=600)
    if not wav_path.exists():
        raise FileNotFoundError(f"yt-dlp did not create expected wav: {wav_path}")
    return wav_path


def convert_to_wav(src: Path, out_dir: Path) -> Path:
    """Convert any video/audio file to 16-kHz mono WAV using ffmpeg."""
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{src.stem}.wav"
    if wav_path.exists():
        return wav_path
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(wav_path)],
        check=True, timeout=600, capture_output=True,
    )
    return wav_path


def _url_source_id(url: str) -> str:
    """Derive a stable source_id from a YouTube URL."""
    import re, hashlib
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    if m:
        return f"yt_{m.group(1)}"
    return "yt_" + hashlib.md5(url.encode()).hexdigest()[:8]


def looks_technical(clean_text: str) -> bool:
    return any(term in clean_text for term in TECHNICAL_TERMS)


def classify_bucket(source_id: str, clean_text: str) -> str:
    if source_id not in SOURCES:
        # Local file or arbitrary URL — always core_clean (or technical if detected)
        return "technical_sql" if looks_technical(clean_text) else "core_clean"
    source_bucket = SOURCES[source_id]["default_bucket"]
    if source_id == "miso_sql_01":
        return "technical_sql" if looks_technical(clean_text) else "core_clean"
    if source_id == "itacademy_db_analyst":
        return "technical_sql" if looks_technical(clean_text) else "hard_examples"
    return source_bucket


def classify_segment(
    seg: dict[str, Any],
    *,
    source_id: str,
    min_duration: float,
    max_duration: float,
    min_chars: int,
) -> tuple[bool, str, str]:
    text = normalize_text(str(seg.get("text") or ""))
    if not text:
        return False, "empty_text", ""

    try:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False, "invalid_timing", text

    duration = max(0.0, end - start)
    if duration < min_duration:
        return False, "too_short", text
    if duration > max_duration:
        return False, "too_long", text
    if len(text) < min_chars:
        return False, "short_text", text

    rule_text = _normalize_for_rules(text)
    if any(pattern in rule_text for pattern in PROMO_PATTERNS):
        return False, "promo", text
    if any(pattern in rule_text for pattern in ADMIN_PATTERNS):
        return False, "admin", text
    if source_id in SOURCES:
        if source_id == "itacademy_sql_01" and not looks_technical(rule_text):
            return False, "non_technical_hard", text
        if source_id == "itacademy_db_analyst" and ("junior" in rule_text or "pracovna zmluva" in rule_text):
            return False, "career_talk", text
    return True, "ok", text


def transcribe_source(
    wav_path: Path,
    *,
    whisper_model: str,
    download_root: str,
    device: str,
    vad_threshold: float,
    min_silence_ms: int,
    min_speech_ms: int,
    speech_pad_ms: int,
    max_speech_s: float,
) -> dict[str, Any]:
    from stt import transcribe_whisper_vad

    return transcribe_whisper_vad(
        str(wav_path),
        whisper_model,
        download_root or str((ROOT / "models" / "faster_whisper").resolve()),
        _asr_device(device),
        vad_threshold=vad_threshold,
        min_silence_ms=min_silence_ms,
        min_speech_ms=min_speech_ms,
        speech_pad_ms=speech_pad_ms,
        max_speech_s=max_speech_s,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def persist_outputs(
    *,
    manifests_dir: Path,
    kept_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    source_stats: dict[str, Any],
    args: argparse.Namespace,
    status: str,
    interrupted_source: str = "",
) -> dict[str, str]:
    candidates_json = manifests_dir / "candidate_segments.json"
    review_json = manifests_dir / "review_rejected.json"
    candidates_jsonl = manifests_dir / "candidate_manifest.jsonl"
    core_jsonl = manifests_dir / "core_clean.jsonl"
    technical_jsonl = manifests_dir / "technical_sql.jsonl"
    hard_jsonl = manifests_dir / "hard_examples.jsonl"
    stats_json = manifests_dir / "stats.json"

    candidates_json.write_text(json.dumps(kept_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    review_json.write_text(json.dumps(rejected_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(candidates_jsonl, kept_rows)
    write_jsonl(core_jsonl, [row for row in kept_rows if row["bucket"] == "core_clean"])
    write_jsonl(technical_jsonl, [row for row in kept_rows if row["bucket"] == "technical_sql"])
    write_jsonl(hard_jsonl, [row for row in kept_rows if row["bucket"] == "hard_examples"])

    rejected_counts: dict[str, int] = {}
    for row in rejected_rows:
        reason = str(row.get("reject_reason") or "unknown")
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    bucket_counts: dict[str, int] = {}
    for row in kept_rows:
        bucket = str(row["bucket"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    outputs = {
        "candidate_segments_json": str(candidates_json),
        "candidate_manifest_jsonl": str(candidates_jsonl),
        "review_rejected_json": str(review_json),
        "core_clean_jsonl": str(core_jsonl),
        "technical_sql_jsonl": str(technical_jsonl),
        "hard_examples_jsonl": str(hard_jsonl),
    }
    stats = {
        "status": status,
        "sources": args.source_ids,
        "completed_sources": sorted(source_stats.keys()),
        "interrupted_source": interrupted_source or "",
        "download_sections": args.download_sections,
        "whisper_model": args.whisper_model,
        "device": _asr_device(args.device),
        "total_kept": len(kept_rows),
        "total_rejected": len(rejected_rows),
        "bucket_counts": bucket_counts,
        "rejected_counts": rejected_counts,
        "source_stats": source_stats,
        "outputs": outputs,
    }
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["stats_json"] = str(stats_json)
    return outputs


def _process_source(
    source_id: str,
    wav_path: Path,
    source_meta: dict[str, str],
    *,
    args,
    raw_dir: Path,
    kept_rows: list,
    rejected_rows: list,
    source_stats: dict,
) -> None:
    """Shared segment processing after audio is ready."""
    transcript = transcribe_source(
        wav_path,
        whisper_model=args.whisper_model,
        download_root=args.download_root,
        device=args.device,
        vad_threshold=args.vad_threshold,
        min_silence_ms=args.min_silence_ms,
        min_speech_ms=args.min_speech_ms,
        speech_pad_ms=args.speech_pad_ms,
        max_speech_s=args.max_speech_s,
    )
    raw_path = raw_dir / f"{source_id}_segments_raw.json"
    raw_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    kept = 0
    rejected = 0
    segments = transcript.get("segments") or []
    for idx, seg in enumerate(segments, start=1):
        keep, reason, text = classify_segment(
            seg,
            source_id=source_id,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_chars=args.min_chars,
        )
        row = {
            "source_id": source_id,
            "source_title": source_meta.get("title", source_id),
            "source_channel": source_meta.get("channel", ""),
            "source_url": source_meta.get("url", ""),
            "bucket": classify_bucket(source_id, _normalize_for_rules(text)),
            "segment_index": idx,
            "start": round(float(seg.get("start", 0.0) or 0.0), 3),
            "end": round(float(seg.get("end", 0.0) or 0.0), 3),
            "duration": round(max(0.0, float(seg.get("end", 0.0) or 0.0) - float(seg.get("start", 0.0) or 0.0)), 3),
            "text": text,
            "audio": str(wav_path),
        }
        if keep:
            kept += 1
            kept_rows.append(row)
        else:
            rejected += 1
            row["reject_reason"] = reason
            rejected_rows.append(row)

    source_stats[source_id] = {
        "title": source_meta.get("title", source_id),
        "channel": source_meta.get("channel", ""),
        "audio": str(wav_path),
        "raw_segments": len(segments),
        "kept": kept,
        "rejected": rejected,
        "raw_json": str(raw_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.list_sources:
        print(json.dumps(SOURCES, ensure_ascii=False, indent=2))
        return

    # Determine which builtin source-ids to process
    if args.source_ids is None:
        # Default: process builtins only if no custom URLs or local files provided
        if not args.urls and not args.local_files:
            source_ids = list(SOURCES.keys())
        else:
            source_ids = []
    else:
        source_ids = args.source_ids

    unknown = [sid for sid in source_ids if sid not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown source ids: {', '.join(unknown)}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    interrupted_source = ""
    final_status = "completed"
    current_source = ""

    try:
        # ── 1. Builtin YouTube sources ─────────────────────────────────────
        for source_id in source_ids:
            current_source = source_id
            source = SOURCES[source_id]
            print(f"[INGEST] {source_id} :: download", flush=True)
            wav_path = download_audio(
                source_id,
                out_dir=out_dir,
                section=args.download_sections,
                reuse_audio=args.reuse_audio,
            )
            print(f"[INGEST] {source_id} :: transcribe", flush=True)
            _process_source(source_id, wav_path, source,
                            args=args, raw_dir=raw_dir,
                            kept_rows=kept_rows, rejected_rows=rejected_rows,
                            source_stats=source_stats)
            persist_outputs(manifests_dir=manifests_dir, kept_rows=kept_rows,
                            rejected_rows=rejected_rows, source_stats=source_stats,
                            args=args, status="running")
            print(f"[INGEST] {source_id} :: saved partial manifests", flush=True)

        # ── 2. Arbitrary YouTube URLs (--urls) ────────────────────────────
        for url in (args.urls or []):
            source_id = _url_source_id(url)
            current_source = source_id
            print(f"[INGEST] {source_id} ({url}) :: download", flush=True)
            wav_path = download_url(url, source_id,
                                    out_dir=out_dir, section=args.download_sections,
                                    reuse_audio=args.reuse_audio)
            print(f"[INGEST] {source_id} :: transcribe", flush=True)
            _process_source(source_id, wav_path,
                            {"title": source_id, "channel": "", "url": url},
                            args=args, raw_dir=raw_dir,
                            kept_rows=kept_rows, rejected_rows=rejected_rows,
                            source_stats=source_stats)
            persist_outputs(manifests_dir=manifests_dir, kept_rows=kept_rows,
                            rejected_rows=rejected_rows, source_stats=source_stats,
                            args=args, status="running")
            print(f"[INGEST] {source_id} :: saved partial manifests", flush=True)

        # ── 3. Local files (--local-files) ────────────────────────────────
        for fpath in (args.local_files or []):
            src = Path(fpath).expanduser().resolve()
            if not src.exists():
                print(f"[WARN] Súbor neexistuje: {src}", flush=True)
                continue
            source_id = f"local_{src.stem}"
            current_source = source_id
            print(f"[INGEST] {source_id} ({src.name}) :: convert to wav", flush=True)
            if src.suffix.lower() == ".wav":
                wav_path = src
            else:
                wav_path = convert_to_wav(src, out_dir)
            print(f"[INGEST] {source_id} :: transcribe", flush=True)
            _process_source(source_id, wav_path,
                            {"title": src.name, "channel": "local", "url": str(src)},
                            args=args, raw_dir=raw_dir,
                            kept_rows=kept_rows, rejected_rows=rejected_rows,
                            source_stats=source_stats)
            persist_outputs(manifests_dir=manifests_dir, kept_rows=kept_rows,
                            rejected_rows=rejected_rows, source_stats=source_stats,
                            args=args, status="running")
            print(f"[INGEST] {source_id} :: saved partial manifests", flush=True)

    except KeyboardInterrupt:
        interrupted_source = current_source
        final_status = "interrupted"
        print(f"[INGEST] Interrupted during source: {interrupted_source or 'unknown'}", flush=True)
    finally:
        outputs = persist_outputs(
            manifests_dir=manifests_dir,
            kept_rows=kept_rows,
            rejected_rows=rejected_rows,
            source_stats=source_stats,
            args=args,
            status=final_status,
            interrupted_source=interrupted_source,
        )

    print("HOTOVO:" if final_status == "completed" else "PARTIAL:")
    print(f" - {outputs['candidate_segments_json']}")
    print(f" - {outputs['candidate_manifest_jsonl']}")
    print(f" - {outputs['review_rejected_json']}")
    print(f" - {outputs['stats_json']}")


if __name__ == "__main__":
    main()
