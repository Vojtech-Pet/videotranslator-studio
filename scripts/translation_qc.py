"""
translation_qc.py
=================
Kontrola kvality prekladu pred a po preklade.
Detekuje problémové segmenty a generuje report.

Vstup:  WhisperX JSON výstup
Výstup: qc_reports/ adresár so samostatnými súbormi per typ problému

Použitie:
    python translation_qc.py --input whisperx_output.json --analyze-only
    python translation_qc.py --input whisperx_output.json --translate
    python translation_qc.py --input whisperx_output.json --translate --model facebook/nllb-200-1.3B
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ── Detektory problémov ───────────────────────────────────────────────────────

def detect_short_segment(text: str, min_words: int = 3) -> dict | None:
    words = text.strip().split()
    if len(words) < min_words:
        return {
            "issue":    "short_segment",
            "detail":   f"{len(words)} slov (min {min_words})",
            "severity": "high" if len(words) == 1 else "medium",
        }
    return None


def detect_incomplete_sentence(text: str) -> dict | None:
    t = text.strip()
    issues = []
    if t and t[0].islower():
        issues.append("začína malým písmenom")
    if t and t[-1] not in ".?!…,;:":
        issues.append("nekončí interpunkciou")
    first_word = t.split()[0].lower() if t.split() else ""
    if first_word in {"and", "but", "or", "so", "because", "that", "which",
                      "who", "when", "if", "though", "although", "however"}:
        issues.append(f"začína spojkou '{first_word}'")
    if issues:
        return {
            "issue":    "incomplete_sentence",
            "detail":   ", ".join(issues),
            "severity": "high" if len(issues) >= 2 else "medium",
        }
    return None


def detect_number_date(text: str) -> dict | None:
    patterns = {
        "date_dmy":   r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b",
        "date_text":  r"\b(january|february|march|april|may|june|july|august|"
                      r"september|october|november|december)\b",
        "time":       r"\b\d{1,2}:\d{2}\b",
        "percentage": r"\b\d+(\.\d+)?\s?%",
        "currency":   r"\$\d+|\d+\s?(dollars?|euros?|usd|eur)",
        "large_num":  r"\b\d{4,}\b",
        "ordinal":    r"\b\d+(st|nd|rd|th)\b",
    }
    found = [name for name, pat in patterns.items()
             if re.search(pat, text, re.IGNORECASE)]
    if found:
        return {"issue": "numbers_dates", "detail": ", ".join(found), "severity": "medium"}
    return None


def detect_nonsense(text: str) -> dict | None:
    t = text.strip()
    issues = []
    words = t.lower().split()
    if len(words) >= 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            issues.append(f"opakujúce sa slová (unique ratio {unique_ratio:.2f})")
    tag_count = len(re.findall(r"\[.*?\]|\(.*?\)", t))
    if tag_count >= 2:
        issues.append(f"{tag_count} tagov/zátvorkiek")
    if not re.search(r"[a-zA-Z]", t):
        issues.append("žiadne písmená")
    if len(t.replace(" ", "")) < 3:
        issues.append("príliš krátky obsah")
    if issues:
        return {"issue": "nonsense", "detail": ", ".join(issues), "severity": "high"}
    return None


def detect_proper_names(text: str) -> dict | None:
    abbr = re.findall(r"\b[A-Z]{2,}\b", text)
    words = text.split()
    capitalized = [w for i, w in enumerate(words)
                   if i > 0 and w and w[0].isupper() and w.isalpha()]
    found = []
    if abbr:
        found.append(f"skratky: {', '.join(set(abbr))}")
    if len(capitalized) >= 2:
        found.append(f"vlastné mená: {', '.join(set(capitalized[:5]))}")
    if found:
        return {"issue": "proper_names", "detail": ", ".join(found), "severity": "low"}
    return None


# ── Analýza segmentov ─────────────────────────────────────────────────────────

def analyze_segment(seg: dict, idx: int) -> dict:
    text   = seg.get("text", "").strip()
    issues = []
    for detector in [detect_short_segment, detect_incomplete_sentence,
                     detect_number_date, detect_nonsense, detect_proper_names]:
        result = detector(text)
        if result:
            issues.append(result)

    severity = "ok"
    if any(i["severity"] == "high"   for i in issues): severity = "high"
    elif any(i["severity"] == "medium" for i in issues): severity = "medium"
    elif issues:                                          severity = "low"

    return {"idx": idx, "start": seg.get("start", 0), "end": seg.get("end", 0),
            "text": text, "issues": issues, "severity": severity}


def analyze_all(segments: list[dict]) -> dict:
    results         = []
    severity_counts = defaultdict(int)
    issue_counts    = defaultdict(int)
    for i, seg in enumerate(segments):
        r = analyze_segment(seg, i)
        results.append(r)
        severity_counts[r["severity"]] += 1
        for iss in r["issues"]:
            issue_counts[iss["issue"]] += 1
    return {
        "timestamp":       datetime.utcnow().isoformat(),
        "total_segments":  len(segments),
        "severity_counts": dict(severity_counts),
        "issue_counts":    dict(issue_counts),
        "segments":        results,
    }


# ── QC po preklade ────────────────────────────────────────────────────────────

def compare_translation(original: str, translated: str,
                        src_lang: str = "eng_Latn") -> list[dict]:
    issues = []
    if not translated.strip():
        issues.append({"issue": "empty_translation", "severity": "high"})
        return issues

    orig_words  = len(original.strip().split())
    trans_words = len(translated.strip().split())

    if orig_words > 3 and trans_words < orig_words * 0.4:
        issues.append({"issue": "translation_too_short",
                       "detail": f"original {orig_words} → preklad {trans_words} slov",
                       "severity": "high"})
    if trans_words > orig_words * 2.5:
        issues.append({"issue": "translation_too_long",
                       "detail": f"original {orig_words} → preklad {trans_words} slov",
                       "severity": "medium"})

    if src_lang == "eng_Latn":
        common_en = {"the", "is", "are", "was", "were", "this", "that",
                     "with", "from", "have", "has", "will", "would", "could"}
        en_leaks = set(translated.lower().split()) & common_en
        if len(en_leaks) >= 2:
            issues.append({"issue": "english_leakage",
                           "detail": f"anglické slová: {', '.join(en_leaks)}",
                           "severity": "medium"})

    orig_nums  = set(re.findall(r"\b\d+[\d.,]*\b", original))
    trans_nums = set(re.findall(r"\b\d+[\d.,]*\b", translated))
    missing    = orig_nums - trans_nums
    if missing:
        issues.append({"issue": "missing_numbers",
                       "detail": f"chýbajú čísla: {', '.join(missing)}",
                       "severity": "high"})
    return issues


# ── Preklad (voliteľný) ───────────────────────────────────────────────────────

def translate_segments(segments: list[dict], model_name: str,
                       src_lang: str) -> list[dict]:
    from transformers import pipeline
    import torch

    print(f"Načítavam prekladový model: {model_name}")
    translator = pipeline(
        "translation", model=model_name,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    texts = [s.get("text", "").strip() for s in segments]
    print(f"Prekladám {len(texts)} segmentov...")
    translated = translator(texts, src_lang=src_lang, tgt_lang="slk_Latn",
                            max_length=256, batch_size=16)

    result = []
    for seg, t in zip(segments, translated):
        sk_text     = t["translation_text"]
        post_issues = compare_translation(seg.get("text", ""), sk_text, src_lang)
        result.append({**seg, "sk_text": sk_text, "post_qc_issues": post_issues})
    return result


# ── Report ────────────────────────────────────────────────────────────────────

def print_summary(report: dict):
    total = report["total_segments"]
    sc    = report["severity_counts"]
    print("\n" + "=" * 60)
    print("QC REPORT – ZHRNUTIE")
    print("=" * 60)
    print(f"Celkom segmentov : {total}")
    for sev in ["ok", "low", "medium", "high"]:
        n = sc.get(sev, 0)
        print(f"  {sev:<8}        : {n} ({n/max(total,1):.1%})")
    print("\nTypy problémov:")
    for issue, count in sorted(report["issue_counts"].items(), key=lambda x: -x[1]):
        print(f"  {issue:<30} {count}×")
    high = [s for s in report["segments"] if s["severity"] == "high"][:5]
    if high:
        print(f"\nUkážka HIGH problémov (prvých 5):")
        for s in high:
            print(f"\n  [{s['idx']:4d}] {s['start']:.1f}s–{s['end']:.1f}s")
            print(f"         TEXT: {s['text'][:80]}")
            for iss in s["issues"]:
                print(f"         ⚠  {iss['issue']}: {iss.get('detail', '')}")


def save_separate_reports(report: dict, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for seg in report["segments"]:
        if seg["severity"] == "ok":
            continue
        for iss in seg.get("issues", []):
            key = f"qc_{iss['issue']}"
            buckets.setdefault(key, []).append((seg, iss))
        for iss in seg.get("post_qc_issues", []):
            key = f"qc_post_{iss['issue']}"
            buckets.setdefault(key, []).append((seg, iss))

    for bucket_name, items in sorted(buckets.items()):
        path  = out / f"{bucket_name}.txt"
        lines = [bucket_name.upper().replace("_", " "),
                 f"Celkom: {len(items)} segmentov", "=" * 60, ""]
        for seg, iss in items:
            lines.append(f"[{seg['idx']:4d}] {seg['start']:.2f}s – {seg['end']:.2f}s  [{seg['severity'].upper()}]")
            lines.append(f"  TEXT: {seg['text']}")
            if seg.get("sk_text"):
                lines.append(f"  SK:   {seg['sk_text']}")
            lines.append(f"  ⚠   {iss.get('detail', '')}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  ✓ {path}  ({len(items)} segmentov)")

    # súhrnný súbor
    summary_path  = out / "qc_summary.txt"
    total_issues  = sum(len(v) for v in buckets.values())
    sc            = report["severity_counts"]
    total         = report["total_segments"]
    lines = ["QC REPORT – SÚHRN PROBLÉMOV",
             f"Vygenerované: {report['timestamp']}",
             f"Celkom segmentov: {total}",
             f"Celkom issues: {total_issues}", "=" * 60, "", "TYPY PROBLÉMOV:"]
    for bucket_name, items in sorted(buckets.items(), key=lambda x: -len(x[1])):
        lines.append(f"  {len(items):4d}×  {bucket_name}")
    lines += ["", "SEVERITY:"]
    for sev in ["high", "medium", "low", "ok"]:
        n = sc.get(sev, 0)
        lines.append(f"  {sev:<8} {n:4d}  ({n/max(total,1):.1%})")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ {summary_path}  (súhrn)")

    # kompletný JSON
    json_path = out / "qc_full_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {json_path}  (kompletný JSON)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QC kontrola prekladu WhisperX segmentov.")
    parser.add_argument("--input",        required=True, help="WhisperX JSON súbor")
    parser.add_argument("--output-dir",   default="qc_reports")
    parser.add_argument("--translate",    action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--model",        default="facebook/nllb-200-1.3B")
    parser.add_argument("--src-lang",     default="eng_Latn")
    args = parser.parse_args()

    print(f"Načítavam: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", []) if isinstance(data, dict) else data
    print(f"Segmentov: {len(segments)}")

    print("Analyzujem segmenty...")
    report = analyze_all(segments)
    print_summary(report)

    if args.translate and not args.analyze_only:
        translated_segs = translate_segments(segments, args.model, args.src_lang)
        for rep_seg, trans_seg in zip(report["segments"], translated_segs):
            rep_seg["sk_text"]        = trans_seg.get("sk_text", "")
            rep_seg["post_qc_issues"] = trans_seg.get("post_qc_issues", [])
            if any(p["severity"] == "high" for p in rep_seg["post_qc_issues"]):
                rep_seg["severity"] = "high"

    print(f"\nUkladám reporty do: {args.output_dir}/")
    save_separate_reports(report, args.output_dir)


if __name__ == "__main__":
    main()
