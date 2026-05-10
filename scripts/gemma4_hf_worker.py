#!/usr/bin/env python3
"""
Standalone worker pre Gemma 4 E4B preklad (HuggingFace transformers).
Spúšťa sa ako subprocess cez fish_env (transformers 5.x).

Vstup:  --segments_json  cesta k JSON so segmentmi
        --model_id       cesta k HF modelu
        --output_json    kam zapísať preložené segmenty
        --max_new_tokens (voliteľné, default 200)
"""

import argparse
import gc
import json
import re
import sys
from pathlib import Path

_DOMAIN_CONTEXT_DEFAULT = Path(__file__).parent / "domain_context.json"


def _load_extra_context(path: str | None) -> str:
    """Načíta domain_context.json a sformátuje ho do system prompt bloku."""
    ctx_path = Path(path) if path else _DOMAIN_CONTEXT_DEFAULT
    if not ctx_path.exists():
        return ""
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[GEMMA4-WORKER] Nepodarilo sa načítať kontext {ctx_path}: {e}", flush=True)
        return ""

    lines = ["\n\n--- DOMAIN KNOWLEDGE (MemPalace) ---"]

    if ctx.get("domain_context"):
        lines.append(f"KONTEXT: {ctx['domain_context']}")

    if ctx.get("sql_term_rules"):
        lines.append("\nSQL VÝRAZY — presné pravidlá:")
        for term, rule in ctx["sql_term_rules"].items():
            lines.append(f"  {term}: {rule}")

    if ctx.get("phonetic_avoid"):
        bad = ", ".join(ctx["phonetic_avoid"][:8])
        lines.append(f"\nFONETICKÉ TVARY ZAKÁZANÉ: {bad}")

    if ctx.get("translation_rules"):
        lines.append("\nPRAVIDLÁ PREKLADU:")
        for rule in ctx["translation_rules"]:
            lines.append(f"  • {rule}")

    if ctx.get("known_issues"):
        lines.append("\nZNÁME PROBLÉMY:")
        for issue in ctx["known_issues"]:
            lines.append(f"  ⚠ {issue}")

    lines.append("--- KONIEC DOMAIN KNOWLEDGE ---")
    result = "\n".join(lines)
    print(f"[GEMMA4-WORKER] Domain context načítaný z {ctx_path.name} ({len(result)} znakov)", flush=True)
    return result

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sql_logic_guard import fix_sql_terms


_SYS_PROMPT = (
    "Si expert na slovenský technický dabing. "
    "Prelož anglický transcript do prirodzenej technickej slovenčiny vhodnej pre voiceover. "
    "Zachovaj presne tieto výrazy: SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, NULLIF, TRUE, FALSE, BOOLEAN. "
    "Rozlišuj: ISNULL je funkcia na náhradu NULL, IS NULL je podmienka, NULLIF je samostatná funkcia. "
    "Nepíš fonetické tvary SQL výrazov. "
    "Používaj prirodzene vysloviteľné vety. "
    "DÔLEŽITÉ: Prelož tak, aby výsledný text mal PRESNE toľko znakov koľko je uvedené v TARGET_CHARS (±10%). "
    "Ak je EN text dlhší ako slot dovoľuje, skráť menej dôležité časti ale zachovaj kľúčové pojmy. "
    "Ak je EN text kratší, môžeš mierne rozvinúť pre prirodzejší prejav. "
    "Vráť iba finálnu slovenskú vetu bez komentára."
)


def postprocess(text: str) -> str:
    out = text.strip()
    for marker in ("<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>"):
        if marker in out:
            out = out.split(marker, 1)[0].strip()
    out = re.sub(r"^\s*assistant\s*:?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r'^\s*["\u201e\u201c\u2018\u2019\u201f\']|["\u201c\u201e\u2018\u2019\u201f\']\s*$', "", out)
    out = fix_sql_terms(out)
    return out.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_json", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--extra_context", default=None,
                    help="Cesta k domain_context.json (default: scripts/domain_context.json)")
    args = ap.parse_args()

    segments = json.loads(Path(args.segments_json).read_text(encoding="utf-8"))

    _extra_ctx = _load_extra_context(args.extra_context)
    _sys_prompt_full = _SYS_PROMPT + _extra_ctx

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    print(f"[GEMMA4-WORKER] Loading model: {args.model_id}", flush=True)
    processor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(args.model_id)
    except Exception:
        processor = None
    if processor is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    input_device = next(model.parameters()).device
    template_owner = processor or tokenizer

    _CPS_SK = 12.5  # Chatterbox SK chars/second

    def translate_one(source_en: str, slot: float) -> str:
        target_chars = max(8, int(slot * _CPS_SK))
        target_min   = max(5, int(target_chars * 0.90))
        target_max   = int(target_chars * 1.10)
        messages = [
            {"role": "system", "content": _sys_prompt_full},
            {
                "role": "user",
                "content": (
                    f"SLOT: {slot:.2f}s | TARGET_CHARS: {target_chars} (min={target_min}, max={target_max})\n"
                    f"TEXT:\n\"{source_en.strip()}\""
                ),
            },
        ]
        if hasattr(template_owner, "apply_chat_template"):
            try:
                prompt = template_owner.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt = template_owner.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:\n"

        try:
            inputs = template_owner(text=prompt, return_tensors="pt")
        except TypeError:
            inputs = template_owner(prompt, return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        new_tokens = outputs[0][input_len:]

        if hasattr(template_owner, "decode"):
            raw = template_owner.decode(new_tokens, skip_special_tokens=True)
        elif getattr(template_owner, "tokenizer", None) is not None:
            raw = template_owner.tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            raw = str(new_tokens)

        return postprocess(raw)

    # Meta-response patterns: Gemma4 refusals / commentary instead of translation
    _META_PREFIXES_W = (
        "toto je technický text", "tento text je v rozpore", "tento text je v rozporu",
        "preklad nebol poskytnutý", "(preklad nebol", "preklad nie je možný",
        "nie je možné preložiť", "text neobsahuje", "text je v rozpore",
        "nie je reálny", "nie reálny text", "prosím, poskytnite",
        "neobsahuje žiadny", "neobsahuje žiadnu",
        "toto nie je technický", "toto nie je reálny",
        "tu je skrátená", "tu je opravená", "tu je preklad",
        "prepis:", "vysvetlenie:", "odpoveď:", "preklad:",
        "subtituly urobil", "titulky urobil", "subtitles by",
    )

    def _is_meta_response(t: str) -> bool:
        tl = t.lower().strip()
        return any(tl.startswith(p) for p in _META_PREFIXES_W)

    result = []
    for i, seg in enumerate(segments):
        src = (seg.get("text") or "").strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        slot = max(0.01, end - start)
        seg_copy = dict(seg)
        seg_copy["text_src"] = src
        if src:
            try:
                translated_text = translate_one(src, slot)
                if _is_meta_response(translated_text):
                    print(f"[GEMMA4-WORKER] [{i}] META-RESPONSE dropped: {translated_text[:80]!r}", flush=True)
                    seg_copy["text"] = ""
                else:
                    seg_copy["text"] = translated_text
                    print(f"[GEMMA4-WORKER] [{i}] {translated_text[:60]}", flush=True)
            except Exception as exc:
                print(f"[GEMMA4-WORKER] [{i}] FAIL ({exc}) — keeping original", flush=True)
        result.append(seg_copy)

    del model
    if processor is not None:
        del processor
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[GEMMA4-WORKER] Model released from memory", flush=True)

    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GEMMA4-WORKER] Saved {len(result)} segments → {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
