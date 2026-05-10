#!/usr/bin/env python3
"""Analýza post-processing metrík — porovnanie fallback rate medzi dvoma runmi.

Použitie:
    # jeden run:
    python scripts/analyze_postproc.py temp/video_A/

    # A/B porovnanie:
    python scripts/analyze_postproc.py temp/video_A/ temp/video_B/ --labels "blend_old" "blend_new"
"""
import argparse
import json
import sys
from pathlib import Path


# ── Rozdelenie segmentov do bucketov ────────────────────────────────────────
def _bucket(dur_s: float) -> str:
    if dur_s < 1.5:
        return "short (<1.5s)"
    if dur_s < 4.0:
        return "medium (1.5–4s)"
    return "long (>4s)"


def _load_metrics(run_dir: Path) -> list[dict]:
    """Načíta všetky *_postproc_metrics.json z priečinka (aj vnorených)."""
    records = []
    for f in sorted(run_dir.rglob("*_postproc_metrics.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Môže byť dict (jeden súbor = jeden run) alebo list segmentov
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                segs = data.get("segments", [data])
                records.extend(segs)
        except Exception as e:
            print(f"  [warn] {f}: {e}", file=sys.stderr)
    return records


# ── Štatistiky pre jeden run ─────────────────────────────────────────────────
def _stats(records: list[dict]) -> dict:
    buckets = {
        "short (<1.5s)": {"total": 0, "raw": 0, "pause_only": 0, "pause_f0": 0},
        "medium (1.5–4s)": {"total": 0, "raw": 0, "pause_only": 0, "pause_f0": 0},
        "long (>4s)": {"total": 0, "raw": 0, "pause_only": 0, "pause_f0": 0},
    }
    dur_deltas = []
    f0_var_deltas = []

    for r in records:
        dur_s = r.get("seg_dur_s", r.get("src_dur", 0.0))
        b = _bucket(float(dur_s))
        decision = r.get("decision", r.get("pp_decision", "raw"))
        buckets[b]["total"] += 1
        if "f0" in decision.lower():
            buckets[b]["pause_f0"] += 1
        elif "pause" in decision.lower():
            buckets[b]["pause_only"] += 1
        else:
            buckets[b]["raw"] += 1

        # dur delta
        dd = r.get("dur_delta", r.get("dur_delta_s"))
        if dd is not None:
            dur_deltas.append(abs(float(dd)))

        # f0 variance delta
        f0b = r.get("f0_var_before")
        f0a = r.get("f0_var_after")
        if f0b is not None and f0a is not None:
            f0_var_deltas.append(float(f0a) - float(f0b))

    total_all = sum(v["total"] for v in buckets.values())
    fallback_total = sum(v["raw"] + v["pause_only"] for v in buckets.values())

    return {
        "buckets": buckets,
        "total": total_all,
        "fallback_total": fallback_total,
        "avg_dur_delta": (sum(dur_deltas) / len(dur_deltas)) if dur_deltas else None,
        "avg_f0_var_delta": (sum(f0_var_deltas) / len(f0_var_deltas)) if f0_var_deltas else None,
    }


# ── Formátovaný výpis ────────────────────────────────────────────────────────
BUCKET_ORDER = ["short (<1.5s)", "medium (1.5–4s)", "long (>4s)"]
COLS = ["total", "pause_f0", "pause_only", "raw", "fallback%"]


def _pct(n, total):
    return f"{100*n/total:.1f}%" if total else "—"


def _print_table(label: str, s: dict):
    print(f"\n{'─'*60}")
    print(f"  Run: {label}   (celkom segmentov: {s['total']})")
    print(f"{'─'*60}")
    hdr = f"  {'Bucket':<22}  {'total':>5}  {'pause+f0':>8}  {'pause_only':>10}  {'raw':>5}  {'fallback%':>9}"
    print(hdr)
    print(f"  {'-'*56}")
    for b in BUCKET_ORDER:
        bk = s["buckets"][b]
        t = bk["total"]
        fb = bk["raw"] + bk["pause_only"]
        print(
            f"  {b:<22}  {t:>5}  {bk['pause_f0']:>8}  {bk['pause_only']:>10}  {bk['raw']:>5}  {_pct(fb,t):>9}"
        )
    print(f"  {'-'*56}")
    fb_all = s["fallback_total"]
    print(f"  {'SPOLU':<22}  {s['total']:>5}  {'':>8}  {'':>10}  {'':>5}  {_pct(fb_all, s['total']):>9}")
    if s["avg_dur_delta"] is not None:
        print(f"\n  Priemerný |dur_delta|: {s['avg_dur_delta']*1000:.1f} ms")
    if s["avg_f0_var_delta"] is not None:
        sign = "+" if s["avg_f0_var_delta"] >= 0 else ""
        print(f"  Priemerná Δf0_var:      {sign}{s['avg_f0_var_delta']:.2f} Hz  (+ = väčšia variabilita)")


def _print_diff(label_a: str, sa: dict, label_b: str, sb: dict):
    print(f"\n{'═'*70}")
    print(f"  A/B DIFF:  {label_a}  →  {label_b}")
    print(f"{'═'*70}")
    print(f"  {'Bucket':<22}  {'fallback% A':>11}  {'fallback% B':>11}  {'Δ':>8}")
    print(f"  {'-'*58}")
    for b in BUCKET_ORDER:
        ba = sa["buckets"][b]
        bb = sb["buckets"][b]
        fba = ba["raw"] + ba["pause_only"]
        fbb = bb["raw"] + bb["pause_only"]
        pct_a = 100 * fba / ba["total"] if ba["total"] else float("nan")
        pct_b = 100 * fbb / bb["total"] if bb["total"] else float("nan")
        delta = pct_b - pct_a
        sign = "↓" if delta < -1 else ("↑" if delta > 1 else "≈")
        print(f"  {b:<22}  {pct_a:>10.1f}%  {pct_b:>10.1f}%  {delta:>+6.1f}pp {sign}")
    print(f"  {'-'*58}")
    tot_a = 100 * sa["fallback_total"] / sa["total"] if sa["total"] else float("nan")
    tot_b = 100 * sb["fallback_total"] / sb["total"] if sb["total"] else float("nan")
    delta_tot = tot_b - tot_a
    sign = "↓" if delta_tot < -1 else ("↑" if delta_tot > 1 else "≈")
    print(f"  {'SPOLU':<22}  {tot_a:>10.1f}%  {tot_b:>10.1f}%  {delta_tot:>+6.1f}pp {sign}")

    if sa["avg_dur_delta"] is not None and sb["avg_dur_delta"] is not None:
        d_a = sa["avg_dur_delta"] * 1000
        d_b = sb["avg_dur_delta"] * 1000
        print(f"\n  Avg |dur_delta|:  A={d_a:.1f}ms  B={d_b:.1f}ms  Δ={d_b-d_a:+.1f}ms")
    if sa["avg_f0_var_delta"] is not None and sb["avg_f0_var_delta"] is not None:
        f_a = sa["avg_f0_var_delta"]
        f_b = sb["avg_f0_var_delta"]
        print(f"  Avg Δf0_var:      A={f_a:+.2f}Hz  B={f_b:+.2f}Hz  Δ={f_b-f_a:+.2f}Hz")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Analýza postproc metrík (fallback rate, dur drift, f0 variabilita)")
    ap.add_argument("runs", nargs="+", metavar="RUN_DIR",
                    help="Jeden alebo dva priečinky s *_postproc_metrics.json súbormi")
    ap.add_argument("--labels", nargs="+", metavar="LABEL",
                    help="Menovky pre runy (napr. 'blend_old' 'blend_new')")
    args = ap.parse_args()

    if len(args.runs) > 2:
        ap.error("Maximálne 2 runy na porovnanie.")

    labels = args.labels or [f"run_{i+1}" for i in range(len(args.runs))]
    if len(labels) < len(args.runs):
        labels += [f"run_{i+1}" for i in range(len(labels), len(args.runs))]

    stats_list = []
    for i, run_dir in enumerate(args.runs):
        p = Path(run_dir)
        if not p.exists():
            print(f"[error] Priečinok neexistuje: {p}", file=sys.stderr)
            sys.exit(1)
        records = _load_metrics(p)
        if not records:
            print(f"[warn] Žiadne *_postproc_metrics.json v {p}", file=sys.stderr)
        s = _stats(records)
        stats_list.append(s)
        _print_table(labels[i], s)

    if len(stats_list) == 2:
        _print_diff(labels[0], stats_list[0], labels[1], stats_list[1])

    print()


if __name__ == "__main__":
    main()
