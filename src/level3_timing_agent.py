from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.text_adaptation import analyze_segment, choose_rewrite_mode

from src.level2_rewrite_agent import call_llm, is_valid_rewrite, read_prompt_text
from src.utils_sql import contains_sql_terms, fix_sql_terms
from src.utils_text import normalize_text

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLOT_PROMPT = _ROOT / "prompts" / "rewrite_slot_system.txt"

TARGET_CPS = 16.0
HARD_CPS_LIMIT = 17.0


def estimate_cps(text: str, slot: float) -> float:
    if slot <= 0.0:
        return 999.0
    return len(normalize_text(text)) / max(slot, 0.01)


def build_timing_prompt(
    *,
    text: str,
    source_text: str,
    slot: float,
    target_cps: float,
    rewrite_mode: str,
    analysis: dict,
) -> str:
    source = normalize_text(source_text)
    lines = [
        f"Rezim: {rewrite_mode}",
        f"Slot: {slot:.2f} sekundy",
        f"Max chars per second: {target_cps:.1f}",
        f"Styl segmentu: {analysis.get('style', 'general_spoken')}",
        f"Treba split: {'ano' if analysis.get('needs_split') else 'nie'}",
    ]
    if source:
        lines += [f'Zdrojovy anglicky text:\n"{source}"']
    lines += [f'Text:\n"{normalize_text(text)}"']
    return "\n\n".join(lines)


def timing_rewrite(
    text: str,
    slot: float,
    *,
    source_text: str = "",
    llm: Any = None,
    system_prompt_path: str | Path = DEFAULT_SLOT_PROMPT,
    target_cps: float | None = None,
    hard_cps_limit: float | None = None,
    max_tokens: int = 256,
) -> str:
    clean = normalize_text(text)
    preserve_sql_terms = contains_sql_terms(clean) or contains_sql_terms(source_text)
    if preserve_sql_terms:
        clean = fix_sql_terms(clean)
    analysis = analyze_segment(clean, source_text, slot)
    effective_target_cps = float(target_cps if target_cps is not None else getattr(analysis, "target_cps", TARGET_CPS))
    effective_hard_cps = float(
        hard_cps_limit if hard_cps_limit is not None else getattr(analysis, "hard_cps_limit", HARD_CPS_LIMIT)
    )
    cps_before = estimate_cps(clean, slot)
    rewrite_mode = choose_rewrite_mode(cps_before, effective_target_cps, effective_hard_cps)

    if llm is None or cps_before <= effective_target_cps:
        return clean

    system_prompt = read_prompt_text(system_prompt_path)
    user_prompt = build_timing_prompt(
        text=clean,
        source_text=source_text,
        slot=slot,
        target_cps=effective_target_cps,
        rewrite_mode=rewrite_mode,
        analysis=getattr(analysis, "__dict__", {}),
    )
    rewritten = call_llm(
        llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=0.05,
    )
    rewritten = normalize_text(rewritten)
    if preserve_sql_terms:
        rewritten = fix_sql_terms(rewritten)
    if not is_valid_rewrite(clean, rewritten, preserve_sql_terms=preserve_sql_terms):
        return clean
    if estimate_cps(rewritten, slot) >= cps_before:
        return clean
    return rewritten


def process_timing_segment(
    seg: dict,
    *,
    llm: Any = None,
    target_cps: float = TARGET_CPS,
    hard_cps_limit: float = HARD_CPS_LIMIT,
) -> dict:
    result = dict(seg)
    slot = max(0.0, float(result.get("end", 0.0)) - float(result.get("start", 0.0)))
    source_text = normalize_text(result.get("text_src"))
    base_text = normalize_text(result.get("level2_text") or result.get("level1_text") or result.get("text"))
    base_text = fix_sql_terms(base_text) if contains_sql_terms(base_text) or contains_sql_terms(source_text) else base_text
    cps_before = estimate_cps(base_text, slot)
    analysis = analyze_segment(base_text, source_text, slot)

    if llm is not None and getattr(analysis, "needs_rewrite", False):
        level3_text = timing_rewrite(
            base_text,
            slot,
            source_text=source_text,
            llm=llm,
            target_cps=target_cps or getattr(analysis, "target_cps", TARGET_CPS),
            hard_cps_limit=hard_cps_limit or getattr(analysis, "hard_cps_limit", HARD_CPS_LIMIT),
        )
        rewrite_mode = "timing_rewrite" if level3_text != base_text else "keep"
    else:
        level3_text = base_text
        rewrite_mode = "keep"

    result["slot"] = round(slot, 3)
    result["cps_before"] = round(cps_before, 2)
    result["cps_after"] = round(estimate_cps(level3_text, slot), 2)
    result["rewrite_mode"] = rewrite_mode
    result["level3_text"] = level3_text
    result["tts_input"] = level3_text
    result["level3_analysis"] = getattr(analysis, "__dict__", {})
    return result

