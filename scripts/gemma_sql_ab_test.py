#!/usr/bin/env python3
"""
Gemma 3 vs Gemma 4 E4B A/B benchmark for SQL dubbing segments.

Purpose:
- compare local Gemma 3 GGUF vs Hugging Face Gemma 4 E4B on the same EN->SK SQL prompts
- score outputs for SQL term correctness, soft similarity to current SK reference,
  and timing friendliness (CPS for the original slot)
- export side-by-side JSON + Markdown for manual review

Typical usage:
  /mnt/tts_data/miniforge3/envs/chatterbox_env/bin/python3 \
    /mnt/tts_data/VideoTranslator_studio/scripts/gemma_sql_ab_test.py

  /mnt/tts_data/miniforge3/envs/chatterbox_env/bin/python3 \
    /mnt/tts_data/VideoTranslator_studio/scripts/gemma_sql_ab_test.py \
    --input_json /mnt/tts_data/VideoTranslator_studio/temp/sql_part_007_part001/sql_part_007_part001_sk_segments.json \
    --segments 5,8,20-22,26-28,35,39
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()

import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sql_logic_guard import (
    SQL_PHONETIC_PATTERNS,
    SQL_PROTECTED_TERMS,
    clean_punctuation,
    fix_sql_terms,
    guard_sql_text,
    protected_sql_terms_in,
)


DEFAULT_INPUT_JSON = ROOT_DIR / "temp" / "sql_part_007_part001" / "sql_part_007_part001_sk_segments.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "temp" / "gemma_sql_ab"
DEFAULT_GEMMA3_MODEL = Path("/mnt/tts_data/VideoTranslator_studio/models/gemma-3-12b-it/google_gemma-3-12b-it-Q8_0.gguf")
DEFAULT_GEMMA4_LOCAL_MODEL = Path("/mnt/tts_data/VideoTranslator_studio/models/gemma-4-E4B-it")
DEFAULT_GEMMA4_MODEL_ID = (
    str(DEFAULT_GEMMA4_LOCAL_MODEL) if DEFAULT_GEMMA4_LOCAL_MODEL.exists() else "google/gemma-4-E4B-it"
)
DEFAULT_SEGMENTS = "1-3,5,8,12-14,20-22,26-28,35,39"
TARGET_CPS = 16.0
HARD_CPS = 17.5

BAD_SPOKEN_PATTERNS = (
    "hlboký ponor",
    "nášho stola",
    "dostávajú to",
    "splynutie splynutia",
    "go and replace",
    "the isnull",
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_segments_spec(spec: str) -> list[int]:
    out: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return sorted(out)


def seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_spaces(a).lower(), normalize_spaces(b).lower()).ratio()


def expected_sql_terms(source_en: str, reference_sk: str) -> set[str]:
    low = normalize_spaces(source_en).lower()
    expected = set(protected_sql_terms_in(reference_sk or ""))
    if "sql" in low:
        expected.add("SQL")
    if "null" in low:
        expected.add("NULL")
    if "coalesce" in low:
        expected.add("COALESCE")
    if "nullif" in low or "null if" in low:
        expected.add("NULLIF")
    if "is not null" in low:
        expected.add("IS NOT NULL")
    if re.search(r"\bisnull\b", low):
        expected.add("ISNULL")
    if re.search(r"\bis null\b", low):
        expected.add("IS NULL")
    if "true" in low:
        expected.add("TRUE")
    if "false" in low:
        expected.add("FALSE")
    if "boolean" in low:
        expected.add("BOOLEAN")
    return expected


def phonetic_violations(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in SQL_PHONETIC_PATTERNS:
        if re.search(pattern, text or "", flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def cps_value(text: str, slot: float) -> float:
    return len(text or "") / max(slot, 0.01)


def cps_score(value: float) -> float:
    if value <= TARGET_CPS:
        return 1.0
    if value <= HARD_CPS:
        return 0.75
    if value <= HARD_CPS + 2.0:
        return 0.45
    return 0.20


def spoken_score(text: str) -> float:
    low = normalize_spaces(text).lower()
    score = 1.0
    for bad in BAD_SPOKEN_PATTERNS:
        if bad in low:
            score -= 0.25
    return max(0.0, round(score, 4))


def sql_term_score(text: str, expected_terms: set[str]) -> float:
    candidate = protected_sql_terms_in(text or "")
    if not expected_terms:
        base = 1.0
    else:
        matches = len(expected_terms & candidate)
        base = matches / max(len(expected_terms), 1)
    if phonetic_violations(text):
        base -= 0.35
    low = normalize_spaces(text).lower()
    if "true" in low and "false" in low and "isnull" in low and "is null" not in low:
        base -= 0.20
    return max(0.0, round(base, 4))


def soft_ref_score(candidate: str, reference: str) -> float:
    if not reference:
        return 0.0
    return round(seq_ratio(candidate, reference), 4)


def overall_score(*, sql_score: float, ref_score: float, cps_score_value: float, spoken_score_value: float) -> float:
    return round(
        (0.40 * sql_score) +
        (0.25 * ref_score) +
        (0.20 * cps_score_value) +
        (0.15 * spoken_score_value),
        4,
    )


def postprocess_candidate(text: str) -> str:
    out = normalize_spaces(text)
    for marker in ("<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>"):
        if marker in out:
            out = out.split(marker, 1)[0].strip()
    out = re.sub(r"^\s*assistant\s*:?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"^\s*[\"“„']|[\"”‟']\s*$", "", out)
    out = fix_sql_terms(out)
    out = clean_punctuation(out)
    return out


def build_system_prompt() -> str:
    return (
        "Si expert na slovenský technický dabing. "
        "Prelož alebo oprav anglický transcript do prirodzenej technickej slovenčiny vhodnej pre voiceover. "
        "Zachovaj význam, ale ak je anglický transcript mierne chybný alebo ASR-rozbitý, oprav ho podľa SQL kontextu. "
        "Zachovaj presne tieto výrazy: SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, NULLIF, TRUE, FALSE, BOOLEAN. "
        "Rozlišuj: ISNULL je funkcia na náhradu NULL, IS NULL je podmienka, NULLIF je samostatná funkcia. "
        "Nepíš fonetické tvary SQL výrazov. "
        "Používaj krátke, čisté, prirodzene vysloviteľné vety. "
        "Vráť iba finálnu slovenskú vetu."
    )


def build_user_prompt(source_en: str, slot: float) -> str:
    return (
        f"SLOT: {slot:.2f} sekundy\n"
        f"TEXT:\n\"{source_en.strip()}\"\n"
    )


def build_refine_system_prompt() -> str:
    return (
        "Si expert na slovenský technický dabing a SQL terminológiu. "
        "Dostaneš anglický zdroj a slovenský draft. "
        "Tvojou úlohou NIE JE prepísať všetko od nuly. "
        "Iba jemne oprav draft tak, aby bol vecne správny, prirodzený a vhodný pre TTS. "
        "Zachovaj význam draftu, ak je správny. "
        "Zachovaj presne tieto výrazy: SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, NULLIF, TRUE, FALSE, BOOLEAN. "
        "Rozlišuj: ISNULL je funkcia na náhradu NULL, IS NULL je podmienka, NULLIF je samostatná funkcia. "
        "Oprav iba technické chyby, zlé SQL názvy, neprirodzené formulácie a prípadne vetu jemne skráť, ak je zbytočne dlhá. "
        "Nepoužívaj fonetické tvary SQL termínov. "
        "Vráť iba finálnu slovenskú vetu."
    )


def build_refine_user_prompt(source_en: str, draft_sk: str, slot: float) -> str:
    return (
        f"SLOT: {slot:.2f} sekundy\n"
        f"EN SOURCE:\n\"{source_en.strip()}\"\n\n"
        f"SK DRAFT:\n\"{draft_sk.strip()}\"\n"
    )


@dataclass
class SegmentCase:
    seg: int
    start: float
    end: float
    slot: float
    source_en: str
    reference_sk: str


class Gemma3GGUFRunner:
    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int, max_tokens: int, temperature: float):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Missing llama_cpp. Use chatterbox_env or install llama-cpp-python.") from exc
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate_with_messages(self, messages: list[dict[str, str]]) -> str:
        resp = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return postprocess_candidate(text)

    def generate(self, source_en: str, slot: float) -> str:
        return self.generate_with_messages(
            [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(source_en, slot)},
            ]
        )

    def refine(self, source_en: str, draft_sk: str, slot: float) -> str:
        return self.generate_with_messages(
            [
                {"role": "system", "content": build_refine_system_prompt()},
                {"role": "user", "content": build_refine_user_prompt(source_en, draft_sk, slot)},
            ]
        )

    def close(self) -> None:
        del self.llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class Gemma4HFRunner:
    def __init__(self, model_id: str, max_new_tokens: int, device_map: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Missing transformers/torch. Use chatterbox_env or install transformers.") from exc

        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        self.processor = None
        self.tokenizer = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            self.processor = None
        if self.processor is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        template_owner = self.processor or self.tokenizer
        if hasattr(template_owner, "apply_chat_template"):
            try:
                return template_owner.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return template_owner.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        combined = []
        for msg in messages:
            combined.append(f"{msg['role'].upper()}:\n{msg['content']}")
        combined.append("ASSISTANT:\n")
        return "\n\n".join(combined)

    def _encode(self, prompt: str):
        owner = self.processor or self.tokenizer
        try:
            inputs = owner(text=prompt, return_tensors="pt")
        except TypeError:
            inputs = owner(prompt, return_tensors="pt")
        return {k: v.to(self.input_device) for k, v in inputs.items()}

    def _decode(self, token_ids) -> str:
        owner = self.processor or self.tokenizer
        if hasattr(owner, "decode"):
            return owner.decode(token_ids, skip_special_tokens=True)
        if getattr(owner, "tokenizer", None) is not None:
            return owner.tokenizer.decode(token_ids, skip_special_tokens=True)
        raise RuntimeError("Cannot decode Gemma 4 output.")

    def generate_with_messages(self, messages: list[dict[str, str]]) -> str:
        prompt = self._apply_chat_template(messages)
        inputs = self._encode(prompt)
        input_len = int(inputs["input_ids"].shape[-1])
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = outputs[0][input_len:]
        return postprocess_candidate(self._decode(new_tokens))

    def generate(self, source_en: str, slot: float) -> str:
        return self.generate_with_messages(
            [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(source_en, slot)},
            ]
        )

    def close(self) -> None:
        del self.model
        if self.processor is not None:
            del self.processor
        if self.tokenizer is not None:
            del self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def select_cases(input_json: Path, segments_spec: str, limit: int | None) -> list[SegmentCase]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    segments = data["segments"] if isinstance(data, dict) and "segments" in data else data
    selected_ids = parse_segments_spec(segments_spec) if segments_spec else list(range(1, len(segments) + 1))
    cases: list[SegmentCase] = []
    for seg_id in selected_ids:
        if seg_id < 1 or seg_id > len(segments):
            continue
        seg = segments[seg_id - 1]
        source_en = normalize_spaces(seg.get("text_src") or "")
        reference_sk = normalize_spaces(seg.get("text") or "")
        if not source_en:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        cases.append(
            SegmentCase(
                seg=seg_id,
                start=start,
                end=end,
                slot=max(0.01, end - start),
                source_en=source_en,
                reference_sk=reference_sk,
            )
        )
    if limit is not None:
        return cases[:limit]
    return cases


def score_candidate(case: SegmentCase, candidate: str) -> dict[str, Any]:
    expected_terms = expected_sql_terms(case.source_en, case.reference_sk)
    sql_score = sql_term_score(candidate, expected_terms)
    ref_score = soft_ref_score(candidate, case.reference_sk)
    cps = round(cps_value(candidate, case.slot), 2)
    cps_sc = cps_score(cps)
    spoken_sc = spoken_score(candidate)
    return {
        "text": candidate,
        "expected_terms": sorted(expected_terms),
        "found_terms": sorted(protected_sql_terms_in(candidate)),
        "phonetic_violations": phonetic_violations(candidate),
        "sql_term_score": sql_score,
        "soft_ref_score": ref_score,
        "cps": cps,
        "cps_score": round(cps_sc, 4),
        "spoken_score": spoken_sc,
        "overall_score": overall_score(
            sql_score=sql_score,
            ref_score=ref_score,
            cps_score_value=cps_sc,
            spoken_score_value=spoken_sc,
        ),
    }


def summarize(results: list[dict[str, Any]], engines: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"segments": len(results), "engines": {}}
    for engine in engines:
        totals = {
            "avg_overall_score": 0.0,
            "avg_sql_term_score": 0.0,
            "avg_soft_ref_score": 0.0,
            "avg_cps": 0.0,
            "too_long": 0,
            "phonetic_fail_segments": 0,
            "wins": 0,
            "errors": 0,
        }
        count = 0
        for row in results:
            item = row.get(engine) or {}
            if item.get("error"):
                totals["errors"] += 1
                continue
            count += 1
            totals["avg_overall_score"] += item.get("overall_score", 0.0)
            totals["avg_sql_term_score"] += item.get("sql_term_score", 0.0)
            totals["avg_soft_ref_score"] += item.get("soft_ref_score", 0.0)
            totals["avg_cps"] += item.get("cps", 0.0)
            if item.get("cps", 0.0) > HARD_CPS:
                totals["too_long"] += 1
            if item.get("phonetic_violations"):
                totals["phonetic_fail_segments"] += 1
        if count:
            totals["avg_overall_score"] = round(totals["avg_overall_score"] / count, 4)
            totals["avg_sql_term_score"] = round(totals["avg_sql_term_score"] / count, 4)
            totals["avg_soft_ref_score"] = round(totals["avg_soft_ref_score"] / count, 4)
            totals["avg_cps"] = round(totals["avg_cps"] / count, 2)
        summary["engines"][engine] = totals

    for row in results:
        best_engine = row.get("winner")
        if best_engine in summary["engines"]:
            summary["engines"][best_engine]["wins"] += 1
    return summary


def write_markdown(out_path: Path, args: argparse.Namespace, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Gemma SQL A/B Report")
    lines.append("")
    lines.append(f"- Input: `{args.input_json}`")
    lines.append(f"- Segments: `{args.segments or 'all'}`")
    lines.append(f"- Engines: `{','.join(args.engines)}`")
    lines.append(f"- Gemma 3 GGUF: `{args.gemma3_model}`")
    lines.append(f"- Gemma 4 HF: `{args.gemma4_model}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for engine in args.engines:
        s = summary["engines"].get(engine, {})
        lines.append(f"### {engine}")
        lines.append(f"- avg overall: `{s.get('avg_overall_score', 0.0)}`")
        lines.append(f"- avg SQL score: `{s.get('avg_sql_term_score', 0.0)}`")
        lines.append(f"- avg soft-ref score: `{s.get('avg_soft_ref_score', 0.0)}`")
        lines.append(f"- avg CPS: `{s.get('avg_cps', 0.0)}`")
        lines.append(f"- wins: `{s.get('wins', 0)}`")
        lines.append(f"- too long: `{s.get('too_long', 0)}`")
        lines.append(f"- phonetic fails: `{s.get('phonetic_fail_segments', 0)}`")
        lines.append("")
    lines.append("## Per Segment")
    lines.append("")
    for row in results:
        lines.append(f"### Seg {row['seg']} | winner: `{row['winner']}`")
        lines.append(f"- slot: `{row['slot']:.2f}s`")
        lines.append(f"- source: `{row['source_en']}`")
        if row.get("reference_sk"):
            lines.append(f"- reference: `{row['reference_sk']}`")
        lines.append("")
        for engine in args.engines:
            item = row[engine]
            if item.get("error"):
                lines.append(f"- {engine}: `[ERROR] {item['error']}`")
                continue
            lines.append(
                f"- {engine}: score=`{item['overall_score']}` sql=`{item['sql_term_score']}` "
                f"ref=`{item['soft_ref_score']}` cps=`{item['cps']}` text=`{item['text']}`"
            )
            if item.get("draft_text"):
                lines.append(f"- {engine} draft: `{item['draft_text']}`")
        lines.append("")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", default=str(DEFAULT_INPUT_JSON))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--segments", default=DEFAULT_SEGMENTS, help="Examples: 5,8,20-22")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--engines", default="gemma3,gemma4", help="Comma-separated subset: gemma3,gemma4,hybrid26b")
    ap.add_argument("--gemma3_model", default=str(DEFAULT_GEMMA3_MODEL))
    ap.add_argument("--gemma3_refine_model", default=str(DEFAULT_GEMMA3_MODEL), help="GGUF model used as refine step in hybrid26b")
    ap.add_argument("--gemma4_model", default=DEFAULT_GEMMA4_MODEL_ID)
    ap.add_argument("--llama_ctx", type=int, default=4096)
    ap.add_argument("--llama_gpu_layers", type=int, default=-1)
    ap.add_argument("--max_tokens", type=int, default=192)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    args.engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = select_cases(input_json, args.segments, args.limit)
    if not cases:
        raise RuntimeError("No benchmark cases selected.")

    if args.dry_run:
        print(json.dumps(
            {
                "selected_segments": [c.seg for c in cases],
                "count": len(cases),
                "input_json": str(input_json),
                "output_dir": str(output_dir),
                "engines": args.engines,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    results: list[dict[str, Any]] = [
        {
            "seg": case.seg,
            "start": case.start,
            "end": case.end,
            "slot": round(case.slot, 3),
            "source_en": case.source_en,
            "reference_sk": case.reference_sk,
        }
        for case in cases
    ]

    hybrid_drafts: list[str | None] = [None] * len(cases)
    hybrid26b_drafts: list[str | None] = [None] * len(cases)

    for engine_name in args.engines:
        if engine_name == "hybrid26b":
            draft_runner = Gemma3GGUFRunner(
                model_path=args.gemma3_model,
                n_ctx=args.llama_ctx,
                n_gpu_layers=args.llama_gpu_layers,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            try:
                for idx, case in enumerate(cases, start=1):
                    print(f"[hybrid26b:draft {idx}/{len(cases)}] seg {case.seg}", flush=True)
                    try:
                        hybrid26b_drafts[idx - 1] = draft_runner.generate(case.source_en, case.slot)
                    except Exception as exc:
                        results[idx - 1][engine_name] = {"error": f"draft_stage: {exc}"}
            finally:
                try:
                    draft_runner.close()
                except Exception:
                    pass
            continue
        elif engine_name == "hybrid":
            draft_runner = Gemma4HFRunner(
                model_id=args.gemma4_model,
                max_new_tokens=args.max_new_tokens,
            )
            try:
                for idx, case in enumerate(cases, start=1):
                    print(f"[hybrid:draft {idx}/{len(cases)}] seg {case.seg}", flush=True)
                    try:
                        hybrid_drafts[idx - 1] = draft_runner.generate(case.source_en, case.slot)
                    except Exception as exc:
                        results[idx - 1][engine_name] = {"error": f"draft_stage: {exc}"}
            finally:
                try:
                    draft_runner.close()
                except Exception:
                    pass
            continue
        elif engine_name == "gemma3":
            runner = Gemma3GGUFRunner(
                model_path=args.gemma3_model,
                n_ctx=args.llama_ctx,
                n_gpu_layers=args.llama_gpu_layers,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elif engine_name == "gemma4":
            runner = Gemma4HFRunner(
                model_id=args.gemma4_model,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            raise RuntimeError(f"Unknown engine: {engine_name}")

        try:
            for idx, case in enumerate(cases, start=1):
                print(f"[{engine_name} {idx}/{len(cases)}] seg {case.seg}", flush=True)
                t0 = time.time()
                try:
                    candidate = runner.generate(case.source_en, case.slot)
                    scored = score_candidate(case, candidate)
                    scored["elapsed_s"] = round(time.time() - t0, 3)
                    results[idx - 1][engine_name] = scored
                except Exception as exc:
                    results[idx - 1][engine_name] = {"error": str(exc)}
        finally:
            try:
                runner.close()
            except Exception:
                pass

    if "hybrid26b" in args.engines:
        refine_runner = Gemma3GGUFRunner(
            model_path=args.gemma3_refine_model,
            n_ctx=args.llama_ctx,
            n_gpu_layers=args.llama_gpu_layers,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        try:
            for idx, case in enumerate(cases, start=1):
                current = results[idx - 1].get("hybrid26b") or {}
                if current.get("error"):
                    continue
                draft_text = hybrid26b_drafts[idx - 1]
                if not draft_text:
                    results[idx - 1]["hybrid26b"] = {"error": "draft_stage: empty draft"}
                    continue
                print(f"[hybrid26b:refine {idx}/{len(cases)}] seg {case.seg}", flush=True)
                t0 = time.time()
                try:
                    candidate = refine_runner.refine(case.source_en, draft_text, case.slot)
                    candidate = guard_sql_text(candidate, source_text=case.source_en, fallback_text=draft_text)
                    scored = score_candidate(case, candidate)
                    scored["elapsed_s"] = round(time.time() - t0, 3)
                    scored["draft_text"] = draft_text
                    results[idx - 1]["hybrid26b"] = scored
                except Exception as exc:
                    results[idx - 1]["hybrid26b"] = {"error": f"refine_stage: {exc}"}
        finally:
            try:
                refine_runner.close()
            except Exception:
                pass

    if "hybrid" in args.engines:
        refine_runner = Gemma3GGUFRunner(
            model_path=args.gemma3_model,
            n_ctx=args.llama_ctx,
            n_gpu_layers=args.llama_gpu_layers,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        try:
            for idx, case in enumerate(cases, start=1):
                current = results[idx - 1].get("hybrid") or {}
                if current.get("error"):
                    continue
                draft_text = hybrid_drafts[idx - 1]
                if not draft_text:
                    results[idx - 1]["hybrid"] = {"error": "draft_stage: empty draft"}
                    continue
                print(f"[hybrid:refine {idx}/{len(cases)}] seg {case.seg}", flush=True)
                t0 = time.time()
                try:
                    candidate = refine_runner.refine(case.source_en, draft_text, case.slot)
                    candidate = guard_sql_text(candidate, source_text=case.source_en, fallback_text=draft_text)
                    scored = score_candidate(case, candidate)
                    scored["elapsed_s"] = round(time.time() - t0, 3)
                    scored["draft_text"] = draft_text
                    results[idx - 1]["hybrid"] = scored
                except Exception as exc:
                    results[idx - 1]["hybrid"] = {"error": f"refine_stage: {exc}"}
        finally:
            try:
                refine_runner.close()
            except Exception:
                pass

    for row in results:
        best_engine = None
        best_score = -1.0
        for engine_name in args.engines:
            item = row.get(engine_name) or {}
            if item.get("error"):
                continue
            score = item.get("overall_score", -1.0)
            if score > best_score:
                best_score = score
                best_engine = engine_name
        row["winner"] = best_engine

    summary = summarize(results, args.engines)
    payload = {
        "args": {
            "input_json": str(input_json),
            "segments": args.segments,
            "engines": args.engines,
            "gemma3_model": args.gemma3_model,
            "gemma4_model": args.gemma4_model,
        },
        "summary": summary,
        "results": results,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_out = output_dir / f"gemma_sql_ab_{stamp}.json"
    md_out = output_dir / f"gemma_sql_ab_{stamp}.md"
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_out, args, results, summary)

    print(f"[OK] JSON: {json_out}")
    print(f"[OK] Markdown: {md_out}")


if __name__ == "__main__":
    main()
