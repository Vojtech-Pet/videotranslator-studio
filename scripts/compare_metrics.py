"""
compare_metrics.py — Porovná metriky pred/po z *_sk_metrics.json

Použitie:
  python3 compare_metrics.py path/to/video_sk_metrics.json
  python3 compare_metrics.py path/to/video_sk_metrics.json --all    # všetky behy
"""
import json
import sys
import argparse
from pathlib import Path


def fmt_diff(a, b, invert=False):
    """Zobrazí rozdiel a → b so šipkou a farbou."""
    d = b - a
    if d == 0:
        return f"{b} (=)"
    arrow = "↓" if d < 0 else "↑"
    # Pre metriky kde nižšie = lepšie, ↓ = green, ↑ = red
    good = d < 0 if not invert else d > 0
    color = "\033[92m" if good else "\033[91m"
    reset = "\033[0m"
    return f"{color}{b} ({arrow}{abs(d)}){reset}"


def print_comparison(runs: list[dict]) -> None:
    if len(runs) < 2:
        print(f"Len {len(runs)} beh(y) — potrebné aspoň 2 na porovnanie.")
        if runs:
            r = runs[0]
            print(f"\nBeh {r.get('timestamp','')} — {r.get('video','')}:")
            print(f"  Segmenty: {r['total_segments']}")
            print(f"  Flagged:  {r['flagged_count']} ({r['flagged_pct']}%)")
            print(f"  bad_term: {r['bad_term_count']}")
            print(f"  haluc.:   {r['hallucination_count']}")
            print(f"  drift:    {r['alignment_drift_count']}")
            print(f"  too_long: {r['too_long_for_slot_count']}")
            print(f"  repaired: {r['gemma_repaired_count']}")
        return

    a, b = runs[-2], runs[-1]
    print(f"\n{'='*60}")
    print(f"POROVNANIE: pred → po")
    print(f"  pred: {a.get('timestamp','')}  ({a.get('video','')})")
    print(f"  po:   {b.get('timestamp','')}  ({b.get('video','')})")
    print(f"{'='*60}")

    metrics = [
        ("Flagged segmenty",       "flagged_count"),
        ("Flagged %",              "flagged_pct"),
        ("bad_term celkom",        "bad_term_count"),
        ("Halucinácie",            "hallucination_count"),
        ("Alignment drift",        "alignment_drift_count"),
        ("Too long for slot",      "too_long_for_slot_count"),
        ("Prázdne segmenty",       "empty_count"),
        ("Gemma opravených",       "gemma_repaired_count"),
    ]
    for label, key in metrics:
        av, bv = a.get(key, 0), b.get(key, 0)
        # gemma_repaired: viac = lepšie (viac opravených)
        invert = key == "gemma_repaired_count"
        print(f"  {label:<26} {av:>5} → {fmt_diff(av, bv, invert=invert)}")

    print(f"\nFlag breakdown (po):")
    for flag, cnt in sorted(b.get("flag_breakdown", {}).items(), key=lambda x: -x[1]):
        prev_cnt = a.get("flag_breakdown", {}).get(flag, 0)
        diff = cnt - prev_cnt
        marker = f"  (↑{diff})" if diff > 0 else (f"  (↓{abs(diff)})" if diff < 0 else "")
        print(f"  {cnt:3d}x  {flag}{marker}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics_file", help="Cesta k *_sk_metrics.json")
    p.add_argument("--all", action="store_true", help="Zobraz všetky behy")
    args = p.parse_args()

    path = Path(args.metrics_file)
    if not path.exists():
        print(f"Súbor neexistuje: {path}"); sys.exit(1)

    runs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(runs, list):
        runs = [runs]

    if args.all:
        for i, r in enumerate(runs):
            print(f"\nBeh {i+1}: {r.get('timestamp','')} — flagged={r['flagged_count']} bad_terms={r['bad_term_count']} repaired={r['gemma_repaired_count']}")
    else:
        print_comparison(runs)


if __name__ == "__main__":
    main()
