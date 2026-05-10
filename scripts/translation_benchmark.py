#!/usr/bin/env python3
"""
Translation benchmark: EN → SK
Porovnáva MADLAD, NLLB, Google, (voliteľne LLM) na kategorizovanom datasete.
Výstup: JSON + CSV na manuálne hodnotenie.

Spustenie:
  /mnt/tts_data/miniforge3/envs/chatterbox_env/bin/python3 \
    /home/vojtech/VideoTranslator/scripts/translation_benchmark.py \
    [--engines madlad,nllb,google] [--out /tmp/benchmark_out]
"""
import sys, json, csv, time, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# ─── dataset ──────────────────────────────────────────────────────────────────
DATASET = [
    # ── A. Bežné vety ─────────────────────────────────────────────────────────
    {"id": "A01", "cat": "normal", "en": "This is how the system works."},
    {"id": "A02", "cat": "normal", "en": "We can now see the result on the screen."},
    {"id": "A03", "cat": "normal", "en": "Let's move on to the next example."},
    {"id": "A04", "cat": "normal", "en": "As you can see, the output is correct."},
    {"id": "A05", "cat": "normal", "en": "Now let's take a look at the second approach."},
    {"id": "A06", "cat": "normal", "en": "Great, so now we have our result."},
    {"id": "A07", "cat": "normal", "en": "Let me show you what happens when we do this."},
    {"id": "A08", "cat": "normal", "en": "This is one of the most common mistakes people make."},

    # ── B. Technické IT ────────────────────────────────────────────────────────
    {"id": "B01", "cat": "technical", "en": "The server processes requests asynchronously."},
    {"id": "B02", "cat": "technical", "en": "This function returns a list of objects."},
    {"id": "B03", "cat": "technical", "en": "We use Docker containers for deployment."},
    {"id": "B04", "cat": "technical", "en": "The application runs in a virtual environment."},
    {"id": "B05", "cat": "technical", "en": "We need to handle the exception properly."},
    {"id": "B06", "cat": "technical", "en": "This method accepts a string and returns a boolean."},
    {"id": "B07", "cat": "technical", "en": "The pipeline consists of three main stages."},
    {"id": "B08", "cat": "technical", "en": "We define a class that inherits from the base class."},

    # ── C. Čísla / jednotky ────────────────────────────────────────────────────
    {"id": "C01", "cat": "numbers", "en": "The model achieved 95% accuracy."},
    {"id": "C02", "cat": "numbers", "en": "The file size is 256 MB."},
    {"id": "C03", "cat": "numbers", "en": "The query returned 1234 rows."},
    {"id": "C04", "cat": "numbers", "en": "We have 3 tables with over 50,000 records each."},
    {"id": "C05", "cat": "numbers", "en": "The timeout is set to 30 seconds."},
    {"id": "C06", "cat": "numbers", "en": "The price increased by 12.5 percent last year."},

    # ── D. Skratky / abbreviations ─────────────────────────────────────────────
    {"id": "D01", "cat": "abbreviations", "en": "The API uses HTTP and JSON."},
    {"id": "D02", "cat": "abbreviations", "en": "This runs on CPU and GPU."},
    {"id": "D03", "cat": "abbreviations", "en": "We use CI/CD pipelines."},
    {"id": "D04", "cat": "abbreviations", "en": "The URL points to the REST endpoint."},
    {"id": "D05", "cat": "abbreviations", "en": "The HTML file contains embedded CSS and JavaScript."},
    {"id": "D06", "cat": "abbreviations", "en": "The RAM is limited to 16 GB on this instance."},

    # ── E. SQL / databázy ──────────────────────────────────────────────────────
    {"id": "E01", "cat": "sql", "en": "We use SELECT to retrieve data from a table."},
    {"id": "E02", "cat": "sql", "en": "The query uses a LEFT JOIN to combine the tables."},
    {"id": "E03", "cat": "sql", "en": "We apply COALESCE to handle NULL values."},
    {"id": "E04", "cat": "sql", "en": "The WHERE clause filters the rows before grouping."},
    {"id": "E05", "cat": "sql", "en": "ISNULL returns true if the value is null."},
    {"id": "E06", "cat": "sql", "en": "We use GROUP BY to aggregate the results."},
    {"id": "E07", "cat": "sql", "en": "The HAVING clause works like WHERE but after GROUP BY."},
    {"id": "E08", "cat": "sql", "en": "This is what null means in SQL."},

    # ── F. Dlhé hovorové vety ─────────────────────────────────────────────────
    {"id": "F01", "cat": "spoken_long", "en": "So what we're going to do now is take a look at how this actually behaves in a real-world scenario."},
    {"id": "F02", "cat": "spoken_long", "en": "This is something that you'll encounter quite often when working with real data."},
    {"id": "F03", "cat": "spoken_long", "en": "Now I know this might look a little bit confusing at first, but once you understand the logic, it actually makes a lot of sense."},
    {"id": "F04", "cat": "spoken_long", "en": "So the idea here is that we want to replace any missing value with a default value that we define ourselves."},
    {"id": "F05", "cat": "spoken_long", "en": "Let's go ahead and write a query that demonstrates exactly how this works in practice."},
    {"id": "F06", "cat": "spoken_long", "en": "And that's basically all there is to it — pretty straightforward once you see it in action."},
]

# ─── engines ──────────────────────────────────────────────────────────────────

_GOOGLE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Android 10; Mobile) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
_GOOGLE_LANG_MAP = {"sk": "sk", "cs": "cs", "en": "en", "slk": "sk", "ces": "cs"}

def _google_raw(text: str, tgt: str = "sk", src: str = "en") -> str:
    """Raw Google Translate — no phonetic_map, no preprocessing."""
    import html, requests, warnings
    warnings.filterwarnings("ignore")
    tgt_code = _GOOGLE_LANG_MAP.get(tgt.lower(), tgt[:2])
    src_code = _GOOGLE_LANG_MAP.get(src.lower(), src[:2])
    url = (f"https://translate.google.com/m"
           f"?sl={src_code}&tl={tgt_code}&hl={tgt_code}"
           f"&q={requests.utils.quote(text)}")
    resp = requests.get(url, headers=_GOOGLE_HEADERS, timeout=15, verify=False)
    resp.raise_for_status()
    import re
    m = re.search(r'class="(?:result-container|t0)">([^<]+)<', resp.text)
    if not m:
        raise RuntimeError("Google: no result in response")
    return html.unescape(m.group(1).strip())


def run_google(texts: list[str], tgt: str = "sk") -> list[str]:
    results = []
    for t in texts:
        try:
            r = _google_raw(t, tgt=tgt)
            print(f"  [google] '{t[:45]}' → '{r[:60]}'")
            results.append(r)
        except Exception as e:
            results.append(f"[ERROR] {e}")
        time.sleep(0.3)  # rate limit
    return results


def run_madlad(texts: list[str], tgt: str = "sk") -> list[str]:
    from translation import translate_segment_with_madlad
    results = []
    for t in texts:
        try:
            r = translate_segment_with_madlad(t, tgt_lang=tgt)
            results.append(r or "")
        except Exception as e:
            results.append(f"[ERROR] {e}")
    return results


def run_nllb(texts: list[str], tgt: str = "sk") -> list[str]:
    from translation import translate_segment_with_nllb
    results = []
    for t in texts:
        try:
            r = translate_segment_with_nllb(t, tgt_lang=tgt)
            results.append(r or "")
        except Exception as e:
            results.append(f"[ERROR] {e}")
    return results


ENGINE_MAP = {
    "google": run_google,
    "madlad": run_madlad,
    "nllb":   run_nllb,
}

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="google,madlad", help="Comma-separated: google,madlad,nllb")
    ap.add_argument("--tgt", default="sk", help="Target language code (default: sk)")
    ap.add_argument("--out", default="/tmp/translation_benchmark", help="Output directory")
    ap.add_argument("--cats", default="", help="Filter categories (comma-sep, e.g. sql,numbers)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    cats    = [c.strip() for c in args.cats.split(",") if c.strip()] if args.cats else []

    dataset = DATASET
    if cats:
        dataset = [d for d in dataset if d["cat"] in cats]

    print(f"[BENCH] Engines: {engines}")
    print(f"[BENCH] Sentences: {len(dataset)} | Target: {args.tgt}")
    print(f"[BENCH] Output: {out_dir}")

    texts = [d["en"] for d in dataset]
    results: dict[str, list[str]] = {}

    for eng in engines:
        if eng not in ENGINE_MAP:
            print(f"[BENCH] Unknown engine: {eng}, skip")
            continue
        print(f"\n[BENCH] Running: {eng} ...", flush=True)
        t0 = time.time()
        try:
            translations = ENGINE_MAP[eng](texts, tgt=args.tgt)
        except Exception as e:
            print(f"[BENCH] Engine {eng} failed: {e}")
            translations = [f"[ENGINE_ERROR] {e}"] * len(texts)
        elapsed = time.time() - t0
        results[eng] = translations
        print(f"[BENCH] {eng}: done in {elapsed:.1f}s")

        # Free GPU memory between heavy models
        if eng in ("madlad", "nllb"):
            try:
                if eng == "madlad":
                    from translation import free_madlad_model
                    free_madlad_model()
                elif eng == "nllb":
                    from translation import free_nllb_model
                    free_nllb_model()
            except Exception:
                pass
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()

    # ── build output ────────────────────────────────────────────────────────
    rows = []
    for i, seg in enumerate(dataset):
        row = {
            "id":  seg["id"],
            "cat": seg["cat"],
            "en":  seg["en"],
        }
        for eng in engines:
            row[eng] = results.get(eng, [""] * len(dataset))[i]
        # score columns (empty — fill manually)
        for eng in engines:
            for metric in ("faith", "natural", "terms", "dub"):
                row[f"{eng}_{metric}"] = ""
        rows.append(row)

    # JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = out_dir / f"benchmark_{ts}.json"
    json_path.write_text(json.dumps({"engines": engines, "tgt": args.tgt, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[BENCH] JSON saved: {json_path}")

    # CSV
    csv_path = out_dir / f"benchmark_{ts}.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[BENCH] CSV saved: {csv_path}")

    # ── quick preview ────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"{'ID':<5} {'CAT':<14} {'EN':<45}")
    for eng in engines:
        print(f"       {'':14} [{eng}]")
    print("─" * 80)
    for row in rows[:5]:
        print(f"{row['id']:<5} {row['cat']:<14} {row['en'][:45]}")
        for eng in engines:
            val = row.get(eng, "")
            print(f"       {'':14}  → {val[:60]}")
        print()
    if len(rows) > 5:
        print(f"  ... ({len(rows) - 5} more rows in output files)")


if __name__ == "__main__":
    main()
