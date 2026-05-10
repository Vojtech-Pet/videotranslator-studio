#!/usr/bin/env python3
"""
Translation quality A/B test — porovnanie ollama variantov + llama.cpp pre 26B SK preklad.

Použitie:
  python translation_quality_test.py \\
      --segments_json /mnt/tts_data/VideoTranslator_studio/temp/<video>/<video>_sk_segments.json \\
      --n 10 \\
      --out_dir /tmp/tr_quality_test

Spustí každý zdrojový EN segment cez:
  - ollama 26b-translator:latest          (Q4_K_M)
  - ollama 26b-translator-q5:latest       (Q5_K_M)
  - ollama 26b-translator-q5l:latest      (Q5_K_L)
  - ollama 26b-translator-q6:latest       (Q6_K)
  - llama-cpp-python  26b_translator_q5km.gguf  (priame, pre referenciu)

Výstup:
  - <out_dir>/comparison.md   — Markdown tabuľka EN | Q4 | Q5 | Q5L | Q6 | llama.cpp
  - <out_dir>/comparison.csv  — to isté ako CSV
  - <out_dir>/timings.json    — sec/segment per model
"""
from __future__ import annotations
import argparse, json, sys, time, csv
from pathlib import Path
from typing import List, Dict, Tuple

import urllib.request, urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODELS = [
    ("q4",  "26b-translator:latest"),
    ("q5",  "26b-translator-q5:latest"),
    ("q5l", "26b-translator-q5l:latest"),
    ("q6",  "26b-translator-q6:latest"),
]

try:
    from paths import PATHS
    LLAMA_CPP_GGUF = str(PATHS._26b_q5km_ft)
except ImportError:
    LLAMA_CPP_GGUF = "/mnt/tts_data/knihy/26b_translator_q5km.gguf"  # fallback
LLAMA_CPP_CTX = 8192
LLAMA_CPP_GPU_LAYERS = -1

PROMPT_TEMPLATE = (
    "Si profesionálny prekladateľ. Prelož nasledujúci anglický text do slovenčiny.\n"
    "Dodržuj prirodzený hovorený slovenský jazyk. Zachovaj tón a dĺžku vety.\n"
    "Technické výrazy 'engine', 'engines', 'shader', 'rendering', 'framework' NEPRELEKÁŠ.\n"
    "Slovo 'era'/'eras' prekladaj ako 'éra'/'éry'. Slovo 'went from' prekladaj ako 'prešiel od'.\n"
    "Odpoveď: IBA preložený text, bez vysvetlení.\n\n"
    "EN: {src}\nSK:"
)


def load_segments(path: Path, n: int) -> List[Dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "segments" in raw:
        segs = raw["segments"]
    elif isinstance(raw, list):
        segs = raw
    else:
        raise SystemExit(f"Neznámy formát JSON: {path}")
    out = []
    for s in segs:
        en = (s.get("text_src") or "").strip()
        if not en:
            continue
        out.append({"start": s.get("start"), "end": s.get("end"), "text_src": en})
        if len(out) >= n:
            break
    return out


def translate_ollama(model: str, src: str, timeout: int = 120, warmup: bool = False) -> Tuple[str, float]:
    body = json.dumps({
        "model": model,
        "prompt": PROMPT_TEMPLATE.format(src=src),
        "stream": False,
        "options": {
            "temperature": 0.0,         # deterministic — žiadne halucinácie/loops
            "num_predict": 256,          # max 256 tokenov výstupu (prevent runaway)
            "num_ctx": 2048,             # kontextové okno (segment ~500 tok stačí)
            "repeat_penalty": 1.15,      # mierne penalizovať opakovania
            "stop": ["EN:", "\nEN:", "\nSK:", "\n\n"],
        },
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode("utf-8")).get("response", "").strip()
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return f"[ERROR: {e}]", time.time() - t0
    return out, time.time() - t0


def translate_llama_cpp(llm, src: str) -> Tuple[str, float]:
    t0 = time.time()
    try:
        out = llm.create_completion(
            prompt=PROMPT_TEMPLATE.format(src=src),
            max_tokens=512,
            temperature=0.0,
            stop=["\nEN:", "\n\n"],
        )
        text = out["choices"][0]["text"].strip()
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - t0
    return text, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments_json", required=True, help="JSON s 'segments' (použije text_src)")
    ap.add_argument("--n", type=int, default=10, help="Počet segmentov (default 10)")
    ap.add_argument("--out_dir", default="/tmp/tr_quality_test", help="Výstupný priečinok")
    ap.add_argument("--skip_llama_cpp", action="store_true", help="Preskoč llama-cpp-python (rýchlejšie)")
    ap.add_argument("--skip_ollama", nargs="*", default=[], help="Mená variantov (q4/q5/q5l/q6) na preskočenie")
    args = ap.parse_args()

    seg_path = Path(args.segments_json)
    if not seg_path.exists():
        raise SystemExit(f"Neexistuje: {seg_path}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Načítavam segmenty z: {seg_path.name}", flush=True)
    segs = load_segments(seg_path, args.n)
    print(f"[INFO] Otestuje sa {len(segs)} segmentov", flush=True)
    if not segs:
        raise SystemExit("Žiadne segmenty s text_src.")

    results = [{"start": s["start"], "end": s["end"], "src": s["text_src"]} for s in segs]
    timings: Dict[str, float] = {}

    # Ollama varianty
    active_ollama = [(t, m) for t, m in OLLAMA_MODELS if t not in set(args.skip_ollama)]
    for tag, model in active_ollama:
        print(f"[OLLAMA-{tag.upper()}] model={model}", flush=True)
        # WARM-UP: prvý request len pre cold-load (model sa nahraje do VRAM)
        print(f"  warming up...", flush=True)
        warmup_text, warmup_dt = translate_ollama(model, "Hello world.", timeout=180)
        print(f"  warmup: {warmup_dt:.1f}s — {warmup_text[:60]}", flush=True)
        # Steady-state translations
        total = 0.0
        for i, r in enumerate(results):
            text, dt = translate_ollama(model, r["src"], timeout=120)
            r[f"ollama_{tag}"] = text
            total += dt
            print(f"  [{i+1}/{len(results)}] {dt:.1f}s — {text[:80]}", flush=True)
        timings[f"ollama_{tag}"] = total / len(results)
        timings[f"ollama_{tag}_warmup"] = warmup_dt
        print(f"  → priemer (po warm-up) {timings[f'ollama_{tag}']:.2f}s/seg", flush=True)

    # llama-cpp-python (Q5_K_M ako referencia)
    if not args.skip_llama_cpp:
        try:
            from llama_cpp import Llama
            print(f"[LLAMA-CPP] Načítavam {LLAMA_CPP_GGUF} (ctx={LLAMA_CPP_CTX})", flush=True)
            llm = Llama(
                model_path=LLAMA_CPP_GGUF,
                n_ctx=LLAMA_CPP_CTX,
                n_gpu_layers=LLAMA_CPP_GPU_LAYERS,
                verbose=False,
            )
            total = 0.0
            for i, r in enumerate(results):
                text, dt = translate_llama_cpp(llm, r["src"])
                r["llama_cpp_q5km"] = text
                total += dt
                print(f"  [{i+1}/{len(results)}] {dt:.1f}s — {text[:80]}", flush=True)
            timings["llama_cpp_q5km"] = total / len(results)
            print(f"  → priemer {timings['llama_cpp_q5km']:.2f}s/seg", flush=True)
            del llm
        except Exception as e:
            print(f"[LLAMA-CPP] Skipping — {e}", flush=True)

    # Markdown tabuľka
    md_path = out_dir / "comparison.md"
    cols = ["src"] + [f"ollama_{t}" for t, _ in active_ollama]
    if not args.skip_llama_cpp and "llama_cpp_q5km" in results[0]:
        cols.append("llama_cpp_q5km")
    headers = ["EN"] + [c.replace("ollama_", "OL ").replace("llama_cpp_q5km", "llama.cpp Q5") for c in cols[1:]]
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Translation quality comparison\n\n")
        f.write(f"Source: `{seg_path.name}`  ·  segments: {len(results)}\n\n")
        f.write(f"## Average sec/segment\n\n")
        for k, v in timings.items():
            f.write(f"- **{k}**: {v:.2f}s/seg\n")
        f.write("\n## Side-by-side\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in results:
            row = []
            for c in cols:
                cell = (r.get(c, "") or "").replace("|", "\\|").replace("\n", " ")
                row.append(cell)
            f.write("| " + " | ".join(row) + " |\n")
    print(f"[OK] Markdown: {md_path}", flush=True)

    # CSV
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end"] + cols)
        for r in results:
            w.writerow([r["start"], r["end"]] + [r.get(c, "") for c in cols])
    print(f"[OK] CSV: {csv_path}", flush=True)

    # Timings
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"[OK] Timings: {out_dir / 'timings.json'}", flush=True)
    print(f"\n[DONE] Výstupy v: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
