#!/usr/bin/env python3
"""
Unified EN -> SK translation benchmark for local/cloud engines.

Focus:
- direct LLMs
- MT engines
- hybrid MT + LLM pipelines

Outputs:
- markdown summary with ranking
- json with full row-level results

Example:
  /mnt/tts_data/miniforge3/envs/musetalk_env/bin/python \
    /home/vojtech/VideoTranslator/scripts/translation_llm_benchmark.py \
    --engines google,translategemma,gemma12b,gemma27b,eurollm,madlad,madlad_gemma_qa
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys_path_added = str(SCRIPT_DIR)
if sys_path_added not in os.sys.path:
    os.sys.path.insert(0, sys_path_added)

from translation import (
    free_madlad_model,
    free_nllb_model,
    refine_fulldoc_with_llama,
    translate_batch_madlad_gemma_qa,
    translate_full_document_with_translategemma,
    translate_segment_with_google,
    translate_segment_with_llama,
    translate_segment_with_madlad,
    translate_segment_with_nllb,
)


DATASET = [
    ("f01", "film", "I just don't know what to do anymore.", "Jednoducho neviem, čo mám ďalej robiť."),
    ("f02", "film", "She looked at him and smiled, but her eyes told a different story.", "Pozrela sa naňho a usmiala sa, ale jej oči hovorili niečo iné."),
    ("f03", "film", "That's not what I meant and you know it.", "To som tým nemyslel a ty to vieš."),
    ("f04", "film", "We've been through worse than this together.", "Prešli sme spolu aj horšími vecami."),
    ("d01", "doc", "The human brain contains approximately 86 billion neurons.", "Ľudský mozog obsahuje približne 86 miliárd neurónov."),
    ("d02", "doc", "Climate change is accelerating at an unprecedented rate.", "Klimatická zmena sa zrýchľuje bezprecedentným tempom."),
    ("d03", "doc", "This phenomenon was first documented in the early 20th century.", "Tento jav bol prvýkrát zdokumentovaný na začiatku 20. storočia."),
    ("t01", "tech", "The server returns a 404 error when the resource is not found.", "Server vráti chybu 404, keď sa zdroj nenájde."),
    ("t02", "tech", "Make sure to commit your changes before pushing to the repository.", "Pred odoslaním do repozitára nezabudni potvrdiť svoje zmeny."),
    ("t03", "tech", "The API rate limit is 1000 requests per minute.", "Limit API je 1000 požiadaviek za minútu."),
    ("i01", "idiom", "It's not rocket science, you just have to practice.", "To nie je nič zložité, len treba cvičiť."),
    ("i02", "idiom", "He bit off more than he could chew.", "Zahryzol sa do viac, než zvládol."),
    (
        "l01",
        "long",
        "While the initial results seemed promising, further analysis revealed significant methodological flaws that undermined the validity of the entire study.",
        "Hoci počiatočné výsledky vyzerali sľubne, ďalšia analýza odhalila závažné metodologické nedostatky, ktoré podkopali platnosť celej štúdie.",
    ),
    ("s01", "sql", "We apply COALESCE to handle NULL values.", "Používame COALESCE na spracovanie hodnôt NULL."),
    ("s02", "sql", "The WHERE clause filters the rows before grouping.", "Klauzula WHERE filtruje riadky pred zoskupením."),
    ("a01", "abbrev", "The API uses HTTP and JSON.", "API používa HTTP a JSON."),
]


MODEL_PATHS = {
    "translategemma": ROOT_DIR / "scripts" / "models" / "translategemma" / "translategemma-12b-it-Q8_0.gguf",
    "eurollm": ROOT_DIR / "scripts" / "models" / "eurollm" / "EuroLLM-9B-Instruct-Q6_K_L.gguf",
    "gemma12b": Path("/mnt/tts_data/VideoTranslator_studio/models/gemma-3-12b-it/google_gemma-3-12b-it-Q8_0.gguf"),
    "gemma27b": Path("/mnt/tts_data/VideoTranslator_studio/models/gemma-3-27b-it/google_gemma-3-27b-it-Q6_K.gguf"),
    "qa8b": Path("/mnt/tts_data/VideoTranslator_studio/models/llama-3-8b-abliterated-v3/Meta-Llama-3-8B-Instruct-Q8_0.gguf"),
}


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_for_score(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(hyp: str, ref: str) -> float:
    hyp_tokens = TOKEN_RE.findall(normalize_for_score(hyp))
    ref_tokens = TOKEN_RE.findall(normalize_for_score(ref))
    if not hyp_tokens or not ref_tokens:
        return 0.0
    hyp_counts = Counter(hyp_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(hyp_counts[t], ref_counts[t]) for t in hyp_counts.keys() & ref_counts.keys())
    precision = overlap / max(sum(hyp_counts.values()), 1)
    recall = overlap / max(sum(ref_counts.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def seq_ratio(hyp: str, ref: str) -> float:
    return difflib.SequenceMatcher(None, normalize_for_score(hyp), normalize_for_score(ref)).ratio()


def length_ratio(hyp: str, ref: str) -> float:
    hyp_len = len(TOKEN_RE.findall(normalize_for_score(hyp)))
    ref_len = len(TOKEN_RE.findall(normalize_for_score(ref)))
    if hyp_len == 0 or ref_len == 0:
        return 0.0
    return min(hyp_len, ref_len) / max(hyp_len, ref_len)


def score_translation(hyp: str, ref: str) -> dict[str, float]:
    tf1 = token_f1(hyp, ref)
    sr = seq_ratio(hyp, ref)
    lr = length_ratio(hyp, ref)
    score = (0.55 * tf1) + (0.35 * sr) + (0.10 * lr)
    return {
        "token_f1": round(tf1, 4),
        "seq_ratio": round(sr, 4),
        "length_ratio": round(lr, 4),
        "score": round(score * 100.0, 2),
    }


def rows_to_segments(dataset: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    return [
        {"id": sid, "cat": cat, "text": en, "start": float(i), "end": float(i + 3)}
        for i, (sid, cat, en, _ref) in enumerate(dataset)
    ]


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def require_model(alias: str) -> str:
    path = MODEL_PATHS[alias]
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return str(path)


def run_google(dataset: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    out = []
    for sid, cat, en, ref in dataset:
        t0 = time.time()
        try:
            sk = translate_segment_with_google(en, tgt_lang="sk", src_lang="en")
        except Exception as exc:
            sk = f"[ERROR] {exc}"
        out.append({"id": sid, "cat": cat, "en": en, "ref": ref, "out": sk, "sec": round(time.time() - t0, 3)})
        time.sleep(0.2)
    return out


def run_madlad(dataset: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    out = []
    for sid, cat, en, ref in dataset:
        t0 = time.time()
        try:
            sk = translate_segment_with_madlad(en, tgt_lang="sk", model_name="google/madlad400-7b-mt")
        except Exception as exc:
            sk = f"[ERROR] {exc}"
        out.append({"id": sid, "cat": cat, "en": en, "ref": ref, "out": sk, "sec": round(time.time() - t0, 3)})
    free_madlad_model()
    release_memory()
    return out


def run_nllb(dataset: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    out = []
    device = "cuda" if _torch_cuda_available() else "cpu"
    for sid, cat, en, ref in dataset:
        t0 = time.time()
        try:
            sk = translate_segment_with_nllb(en, tgt_lang="sk", model_name="facebook/nllb-200-distilled-1.3B", device=device)
        except Exception as exc:
            sk = f"[ERROR] {exc}"
        out.append({"id": sid, "cat": cat, "en": en, "ref": ref, "out": sk, "sec": round(time.time() - t0, 3)})
    free_nllb_model()
    release_memory()
    return out


def run_translategemma(dataset: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    segs = rows_to_segments(dataset)
    t0 = time.time()
    translated = translate_full_document_with_translategemma(
        segs,
        "sk",
        require_model("translategemma"),
        n_gpu_layers=-1,
        n_ctx=8192,
    )
    total = max(time.time() - t0, 0.001)
    out = []
    for i, (sid, cat, en, ref) in enumerate(dataset):
        sk = translated[i].get("text", "") if i < len(translated) else "[MISSING]"
        out.append({"id": sid, "cat": cat, "en": en, "ref": ref, "out": sk, "sec": round(total / len(dataset), 3)})
    release_memory()
    return out


def run_multislav5lang(dataset: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer, MarianMTModel

    model_name = "allegro/multislav-5lang"
    device = "cuda" if _torch_cuda_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    out = []
    try:
        for sid, cat, en, ref in dataset:
            t0 = time.time()
            try:
                batch = tokenizer(
                    [f">>slk<< {en}"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                with torch.no_grad():
                    generated = model.generate(**batch, max_new_tokens=196)
                sk = tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
            except Exception as exc:
                sk = f"[ERROR] {exc}"
            out.append({"id": sid, "cat": cat, "en": en, "ref": ref, "out": sk, "sec": round(time.time() - t0, 3)})
    finally:
        del model
        del tokenizer
        release_memory()
    return out


def run_llama_batch(
    dataset: list[tuple[str, str, str, str]],
    model_alias: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    temperature: float = 0.05,
) -> list[dict[str, Any]]:
    from llama_cpp import Llama

    model_path = require_model(model_alias)
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    out = []
    prev_translation = None
    try:
        for sid, cat, en, ref in dataset:
            t0 = time.time()
            try:
                sk = translate_segment_with_llama(
                    text=en,
                    duration_sec=max(1.0, len(en.split()) / 2.6),
                    llama_model=model_path,
                    tgt_lang="sk",
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    temperature=temperature,
                    llm_instance=llm,
                    prev_translation=prev_translation,
                    content_type="general",
                )
            except Exception as exc:
                sk = f"[ERROR] {exc}"
            prev_translation = sk[-200:] if sk and not sk.startswith("[ERROR]") else prev_translation
            out.append({"id": sid, "cat": cat, "en": en, "ref": ref, "out": sk, "sec": round(time.time() - t0, 3)})
    finally:
        del llm
        release_memory()
    return out


def run_hybrid_madlad_gemma_qa(
    dataset: list[tuple[str, str, str, str]],
    use_google_phase1: bool = False,
    use_qa: bool = True,
) -> list[dict[str, Any]]:
    segs = rows_to_segments(dataset)
    t0 = time.time()
    results = translate_batch_madlad_gemma_qa(
        segs,
        tgt_lang="sk",
        madlad_model="google/madlad400-7b-mt",
        gemma_model=require_model("gemma27b"),
        gemma_n_ctx=4096,
        gemma_n_gpu_layers=56,
        qa_model=require_model("qa8b") if use_qa else "",
        qa_n_ctx=2048,
        qa_n_gpu_layers=32,
        use_google_phase1=use_google_phase1,
        src_lang="en",
    )
    total = max(time.time() - t0, 0.001)
    out = []
    for i, (sid, cat, en, ref) in enumerate(dataset):
        row = results[i] if i < len(results) else {}
        sk = row.get("text_final", row.get("text_polished", row.get("text_translated", "[MISSING]")))
        out.append(
            {
                "id": sid,
                "cat": cat,
                "en": en,
                "ref": ref,
                "out": sk,
                "sec": round(total / len(dataset), 3),
                "qa_ok": row.get("qa_ok", True),
                "qa_feedback": row.get("qa_feedback", "skipped"),
            }
        )
    release_memory()
    return out


def run_refine_pass(
    rows: list[dict[str, Any]],
    model_alias: str = "gemma27b",
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
) -> list[dict[str, Any]]:
    segs = [{"text": row["out"], "start": float(i), "end": float(i + 3)} for i, row in enumerate(rows)]
    refined = refine_fulldoc_with_llama(segs, "sk", require_model(model_alias), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    out = []
    for i, row in enumerate(rows):
        updated = dict(row)
        updated["out"] = refined[i].get("text", row["out"]) if i < len(refined) else row["out"]
        out.append(updated)
    release_memory()
    return out


def run_multislav5lang_gemma12b(dataset: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    t0 = time.time()
    rows = run_multislav5lang(dataset)
    rows = run_refine_pass(rows, model_alias="gemma12b", n_ctx=2048, n_gpu_layers=-1)
    avg_sec = round(max(time.time() - t0, 0.001) / max(len(dataset), 1), 3)
    for row in rows:
        row["sec"] = avg_sec
    return rows


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


ENGINE_RUNNERS = {
    "google": lambda ds: run_google(ds),
    "madlad": lambda ds: run_madlad(ds),
    "nllb": lambda ds: run_nllb(ds),
    "multislav5lang": lambda ds: run_multislav5lang(ds),
    "multislav5lang_gemma12b": lambda ds: run_multislav5lang_gemma12b(ds),
    "translategemma": lambda ds: run_translategemma(ds),
    "eurollm": lambda ds: run_llama_batch(ds, "eurollm", n_ctx=3072),
    "gemma12b": lambda ds: run_llama_batch(ds, "gemma12b", n_ctx=4096),
    "gemma27b": lambda ds: run_llama_batch(ds, "gemma27b", n_ctx=4096),
    "madlad_gemma_qa": lambda ds: run_hybrid_madlad_gemma_qa(ds, use_google_phase1=False, use_qa=True),
    "google_gemma_qa": lambda ds: run_hybrid_madlad_gemma_qa(ds, use_google_phase1=True, use_qa=True),
}


def summarize_engine(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "avg_score": 0.0,
        "avg_token_f1": 0.0,
        "avg_seq_ratio": 0.0,
        "avg_len_ratio": 0.0,
        "avg_sec": 0.0,
        "categories": {},
        "qa_warnings": 0,
    }
    if not rows:
        return summary

    score_sum = 0.0
    f1_sum = 0.0
    seq_sum = 0.0
    len_sum = 0.0
    sec_sum = 0.0
    by_cat: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        metrics = row["metrics"]
        score_sum += metrics["score"]
        f1_sum += metrics["token_f1"]
        seq_sum += metrics["seq_ratio"]
        len_sum += metrics["length_ratio"]
        sec_sum += row["sec"]
        by_cat[row["cat"]].append(metrics["score"])
        if row.get("qa_ok") is False:
            summary["qa_warnings"] += 1

    count = len(rows)
    summary["avg_score"] = round(score_sum / count, 2)
    summary["avg_token_f1"] = round(f1_sum / count, 4)
    summary["avg_seq_ratio"] = round(seq_sum / count, 4)
    summary["avg_len_ratio"] = round(len_sum / count, 4)
    summary["avg_sec"] = round(sec_sum / count, 3)
    summary["categories"] = {cat: round(sum(vals) / len(vals), 2) for cat, vals in sorted(by_cat.items())}
    return summary


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Translation LLM Benchmark")
    lines.append("")
    lines.append(f"Dátum: {report['timestamp']}")
    lines.append(f"Enginy: {', '.join(report['engines'])}")
    lines.append(f"Viet: {len(report['dataset'])}")
    lines.append("")
    lines.append("## Rebríček")
    lines.append("")
    lines.append("| Rank | Engine | Avg score | Avg sec/seg | QA warnings |")
    lines.append("|---|---|---:|---:|---:|")
    for idx, (engine, summary) in enumerate(report["ranking"], 1):
        lines.append(f"| {idx} | {engine} | {summary['avg_score']:.2f} | {summary['avg_sec']:.3f} | {summary['qa_warnings']} |")

    lines.append("")
    lines.append("## Kategórie")
    lines.append("")
    for engine, summary in report["ranking"]:
        lines.append(f"### {engine}")
        lines.append("")
        lines.append(f"- Priemer: `{summary['avg_score']:.2f}`")
        lines.append(f"- Rýchlosť: `{summary['avg_sec']:.3f}s/segment`")
        if summary["qa_warnings"]:
            lines.append(f"- QA warnings: `{summary['qa_warnings']}`")
        for cat, val in summary["categories"].items():
            lines.append(f"- {cat}: `{val:.2f}`")
        lines.append("")

    lines.append("## Detail")
    lines.append("")
    for engine in report["engines"]:
        lines.append(f"### {engine}")
        lines.append("")
        lines.append("| ID | Cat | Score | EN | OUT |")
        lines.append("|---|---|---:|---|---|")
        for row in report["results"][engine]:
            en_short = row["en"].replace("|", "\\|")
            out_short = row["out"].replace("|", "\\|")
            lines.append(f"| {row['id']} | {row['cat']} | {row['metrics']['score']:.2f} | {en_short} | {out_short} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engines",
        default="google,translategemma,gemma12b,gemma27b,eurollm,madlad,madlad_gemma_qa",
        help="Comma-separated engines",
    )
    ap.add_argument("--cats", default="", help="Comma-separated category filter")
    ap.add_argument("--refine", action="store_true", help="Add Gemma-27B refine pass to each engine result")
    ap.add_argument("--out", default=str(ROOT_DIR / "output" / "translation_llm_benchmark"), help="Output directory")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    cats = {c.strip() for c in args.cats.split(",") if c.strip()}
    dataset = [row for row in DATASET if not cats or row[1] in cats]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BENCH] Engines: {engines}", flush=True)
    print(f"[BENCH] Rows: {len(dataset)}", flush=True)
    print(f"[BENCH] Output dir: {out_dir}", flush=True)

    results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for engine in engines:
        runner = ENGINE_RUNNERS.get(engine)
        if runner is None:
            print(f"[BENCH] Unknown engine: {engine} — skip", flush=True)
            continue

        print(f"\n[BENCH] Running {engine} ...", flush=True)
        rows = runner(dataset)
        if args.refine:
            print(f"[BENCH] Refine pass for {engine} ...", flush=True)
            rows = run_refine_pass(rows)

        for row in rows:
            row["metrics"] = score_translation(row["out"], row["ref"])

        results[engine] = rows
        summaries[engine] = summarize_engine(rows)
        print(
            f"[BENCH] {engine}: score={summaries[engine]['avg_score']:.2f} "
            f"sec/seg={summaries[engine]['avg_sec']:.3f}",
            flush=True,
        )
        release_memory()

    ranking = sorted(summaries.items(), key=lambda item: item[1]["avg_score"], reverse=True)
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "engines": list(results.keys()),
        "dataset": [{"id": sid, "cat": cat, "en": en, "ref": ref} for sid, cat, en, ref in dataset],
        "results": results,
        "summaries": summaries,
        "ranking": ranking,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"translation_llm_benchmark_{ts}.json"
    md_path = out_dir / f"translation_llm_benchmark_{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")

    print("\n[BENCH] Ranking:", flush=True)
    for idx, (engine, summary) in enumerate(ranking, 1):
        print(f"  {idx}. {engine:<20} score={summary['avg_score']:.2f} sec/seg={summary['avg_sec']:.3f}", flush=True)
    print(f"[BENCH] JSON: {json_path}", flush=True)
    print(f"[BENCH] MD:   {md_path}", flush=True)


if __name__ == "__main__":
    main()
