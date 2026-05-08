#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Level 20 orchestration runner over current JSON/TTS assets.")
    ap.add_argument("--input", required=True, help="Input segment JSON for Level 14.")
    ap.add_argument("--out-dir", default="level20_engine_out", help="Output directory")
    ap.add_argument("--provider", choices=["ollama", "disabled"], default="ollama", help="Level 14 rewrite provider")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama URL")
    ap.add_argument("--ollama-model", default="gemma3:27b", help="Ollama model")
    ap.add_argument("--max-ollama-segments", type=int, default=8, help="Max segments for Level 14 multipass rewrite")
    ap.add_argument("--skip-tts", action="store_true", help="Stop after Level 15 export pack")
    ap.add_argument("--checkpoint", default="", help="Fish S2-Pro checkpoint for Level 16")
    ap.add_argument("--reference-wav", default="", help="Optional reference wav for Level 16")
    ap.add_argument("--reference-text", default="", help="Optional reference text for Level 16")
    ap.add_argument("--server-python", default="", help="Optional Fish server python for Level 16")
    ap.add_argument("--server-port", type=int, default=8091, help="Fish server port for Level 16")
    ap.add_argument("--run-level19", action="store_true", help="Run Level 19 after Level 16")
    ap.add_argument("--asr-backend", choices=["disabled", "faster_whisper"], default="disabled", help="Level 19 ASR backend")
    return ap


def _run(cmd: list[str], *, timeout: int = 300) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, timeout=timeout)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    level14_json = out_dir / "level14.json"
    level14_report = out_dir / "level14_report.json"
    level15_dir = out_dir / "level15_pack"
    level16_dir = out_dir / "level16_out"
    level19_report = out_dir / "level19_report.json"
    level19_summary = out_dir / "level19_summary.json"
    manifest_path = out_dir / "engine_manifest.json"

    _run(
        [
            sys.executable,
            str(THIS_DIR / "level14_full_pipeline.py"),
            "--input",
            str(input_path),
            "--output",
            str(level14_json),
            "--report",
            str(level14_report),
            "--provider",
            args.provider,
            "--ollama-url",
            args.ollama_url,
            "--ollama-model",
            args.ollama_model,
            "--max-ollama-segments",
            str(args.max_ollama_segments),
        ]
    )

    _run(
        [
            sys.executable,
            str(THIS_DIR / "level15_export_pack.py"),
            "--input",
            str(level14_json),
            "--out-dir",
            str(level15_dir),
        ]
    )

    manifest: dict[str, object] = {
        "level14_json": str(level14_json),
        "level14_report": str(level14_report),
        "level15_pack": str(level15_dir),
    }

    if not args.skip_tts:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --skip-tts is set.")
        _run(
            [
                sys.executable,
                str(THIS_DIR / "level16_tts_inference.py"),
                "--input",
                str(level15_dir / "tts_ready.json"),
                "--out-dir",
                str(level16_dir),
                "--checkpoint",
                args.checkpoint,
                "--server-port",
                str(args.server_port),
                *(["--reference-wav", args.reference_wav] if args.reference_wav else []),
                *(["--reference-text", args.reference_text] if args.reference_text else []),
                *(["--server-python", args.server_python] if args.server_python else []),
            ]
        )
        manifest["level16_manifest"] = str(level16_dir / "tts_manifest.json")
        manifest["level16_failed"] = str(level16_dir / "tts_failed.json")

        if args.run_level19:
            _run(
                [
                    sys.executable,
                    str(THIS_DIR / "level19_auto_evaluate.py"),
                    "--input",
                    str(level16_dir / "tts_manifest.json"),
                    "--output",
                    str(level19_report),
                    "--summary",
                    str(level19_summary),
                    "--asr-backend",
                    args.asr_backend,
                ]
            )
            manifest["level19_report"] = str(level19_report)
            manifest["level19_summary"] = str(level19_summary)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HOTOVO: {manifest_path}")


if __name__ == "__main__":
    main()
