#!/usr/bin/env python3
"""
fix_short_translations.py
=========================
Opraví segmenty s príliš krátkym prekladom (nízke CPS) v sk_segments.json.
Zavolá gemma4_hf_worker iba na problematické segmenty a aktualizuje JSON.

Použitie:
    python fix_short_translations.py /cesta/k/sk_segments.json
    python fix_short_translations.py /cesta/k/sk_segments.json --cps_threshold 9.0 --dry_run
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
GEMMA4_WORKER = SCRIPT_DIR / "gemma4_hf_worker.py"
DOMAIN_CONTEXT = SCRIPT_DIR / "domain_context.json"

DEFAULT_FISH_PYTHON = "/mnt/tts_data/miniforge3/envs/fish_env/bin/python3"
DEFAULT_GEMMA4_MODEL = "/mnt/tts_data/VideoTranslator_studio/models/gemma-4-E4B-it"
CPS_THRESHOLD_DEFAULT = 9.0


def load_segments(path: Path) -> tuple:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        segs = data.get("segments", list(data.values())[0])
        return segs, data
    return data, None


def save_segments(path: Path, segs: list, wrapper) -> None:
    if wrapper is not None:
        wrapper["segments"] = segs
        out = wrapper
    else:
        out = segs
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def run_gemma4(segments_to_fix: list, fish_python: str, gemma4_model: str) -> list:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f_in:
        json.dump(segments_to_fix, f_in, ensure_ascii=False)
        seg_path = f_in.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_out:
        out_path = f_out.name

    try:
        cmd = [
            fish_python, str(GEMMA4_WORKER),
            "--segments_json", seg_path,
            "--model_id", gemma4_model,
            "--output_json", out_path,
            "--max_new_tokens", "250",
        ]
        if DOMAIN_CONTEXT.exists():
            cmd += ["--extra_context", str(DOMAIN_CONTEXT)]
            print(f"[FIX] Domain context: {DOMAIN_CONTEXT.name}", flush=True)
        print(f"[FIX] Spúšťam gemma4 na {len(segments_to_fix)} segmentov...", flush=True)
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"gemma4_hf_worker zlyhal (returncode={proc.returncode})")
        return json.loads(Path(out_path).read_text(encoding="utf-8"))
    finally:
        Path(seg_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("segments_json", help="Cesta k sk_segments.json")
    ap.add_argument("--cps_threshold", type=float, default=CPS_THRESHOLD_DEFAULT,
                    help=f"Segmenty s CPS < N sa preložia znovu (default={CPS_THRESHOLD_DEFAULT})")
    ap.add_argument("--dry_run", action="store_true",
                    help="Len zobraz problematické segmenty, nič nemeň")
    ap.add_argument("--model", default=DEFAULT_GEMMA4_MODEL, help="Cesta k gemma4 modelu")
    ap.add_argument("--python", default=DEFAULT_FISH_PYTHON, help="fish_env python")
    args = ap.parse_args()

    seg_path = Path(args.segments_json)
    if not seg_path.exists():
        sys.exit(f"[ERROR] Súbor neexistuje: {seg_path}")

    segs, wrapper = load_segments(seg_path)
    print(f"[FIX] Načítaných {len(segs)} segmentov z {seg_path.name}")

    # Nájdi problematické segmenty
    low_cps_indices = []
    for i, seg in enumerate(segs):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        slot = max(0.01, end - start)
        text = (seg.get("text") or "").strip()
        cps = len(text) / slot
        if cps < args.cps_threshold:
            src = seg.get("text_src") or seg.get("source_text") or ""
            target_chars = int(slot * 12.5)
            low_cps_indices.append(i)
            print(
                f"  [{i+1:02d}] t={start:.1f}-{end:.1f}s slot={slot:.1f}s "
                f"cps={cps:.1f} chars={len(text)} target={target_chars}"
            )
            if src:
                print(f"       EN: {src[:80]}")
            print(f"       SK: {text}")
            print()

    if not low_cps_indices:
        print("[FIX] Žiadne segmenty s nízkym CPS — nič na opravu.")
        return

    print(f"[FIX] Celkovo {len(low_cps_indices)} segmentov na opravu (CPS < {args.cps_threshold})")

    if args.dry_run:
        print("[FIX] --dry_run: koniec (nič sa nezmenilo)")
        return

    # Priprav segmenty pre gemma4
    segs_to_fix = []
    for i in low_cps_indices:
        seg = segs[i]
        src = seg.get("text_src") or seg.get("source_text") or seg.get("text", "")
        segs_to_fix.append({
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": src,
            "text_src": src,
            "_orig_idx": i,
        })

    # Spusti gemma4
    fixed = run_gemma4(segs_to_fix, args.python, args.model)

    # Aplikuj opravené preklady
    fixed_count = 0
    for orig_seg_info, fixed_seg in zip(segs_to_fix, fixed):
        orig_idx = orig_seg_info["_orig_idx"]
        new_text = (fixed_seg.get("text") or "").strip()
        old_text = (segs[orig_idx].get("text") or "").strip()
        slot = max(0.01, float(segs[orig_idx].get("end", 0)) - float(segs[orig_idx].get("start", 0)))

        if new_text and len(new_text) > len(old_text):
            segs[orig_idx]["text"] = new_text
            new_cps = len(new_text) / slot
            print(
                f"  [{orig_idx+1:02d}] opravený: {len(old_text)}→{len(new_text)}ch cps→{new_cps:.1f}"
            )
            print(f"       {new_text}")
            fixed_count += 1
        else:
            print(f"  [{orig_idx+1:02d}] nezmenený (gemma4 nevrátil dlhší text): {new_text[:60]}")

    # Záloha + uloženie
    backup_path = seg_path.with_suffix(".json.bak")
    shutil.copy2(seg_path, backup_path)
    print(f"\n[FIX] Záloha: {backup_path}")

    save_segments(seg_path, segs, wrapper)
    print(f"[FIX] Uložené: {seg_path} ({fixed_count}/{len(low_cps_indices)} segmentov opravených)")


if __name__ == "__main__":
    main()
