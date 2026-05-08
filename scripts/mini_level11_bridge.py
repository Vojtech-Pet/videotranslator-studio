"""
mini_level11_bridge.py
======================
Lightweight Studio integration of the mini_level11 rewrite pass.

Runs before TTS and uses:
- English source (text_src) as primary rewrite source
- Ollama when available
- SQL logic guard to preserve ISNULL / IS NULL / NULLIF semantics
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from linux_logic_guard import guard_linux_text
from sql_logic_guard import guard_sql_text, source_guided_template
from text_adaptation import estimate_chars_per_second


@dataclass
class MiniLevel11Stats:
    total: int = 0
    changed: int = 0
    ollama: int = 0
    source_template: int = 0
    fallback: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_spaces(text: str) -> str:
    return " ".join((text or "").split())


def _clean_punctuation(text: str) -> str:
    import re

    out = re.sub(r"\s+([,.:;!?])", r"\1", text or "")
    out = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", out)
    return _normalize_spaces(out)


def _classify_segment_type(text: str) -> str:
    low = _normalize_spaces(text).lower()
    if any(x in low for x in ["syntax", "isnull", "coalesce", "nullif", "is null", "is not null"]):
        return "definition"
    if any(x in low for x in ["example", "for example", "scenario", "address", "príklad", "pozrime sa"]):
        return "example"
    if any(x in low for x in ["summary", "overview", "prehľad", "zhrnutie"]):
        return "summary"
    return "default"


def _choose_mode(slot: float, current_text: str, *, target_cps: float, hard_cps_limit: float) -> str:
    cps = estimate_chars_per_second(current_text, slot)
    if cps <= target_cps:
        return "balanced"
    if cps <= hard_cps_limit:
        return "dub_friendly"
    return "aggressive"


def _build_prompt(text_en: str, slot: float, mode: str, seg_type: str) -> str:
    return f"""
Si expert na slovensky technicky dabing.

Uloha:
Preloz vetu z anglictiny do prirodzenej technickej slovenciny vhodnej pre voiceover.

KRITICKE PRAVIDLA:
- Zachovaj vyznam.
- Zachovaj presne tieto vyrazy:
  SQL, NULL, IS NULL, IS NOT NULL, ISNULL, COALESCE, NULLIF, TRUE, FALSE, BOOLEAN
- Nikdy nepis:
  ajznul, koalesk, esikjuel
- Neprekladaj SQL keywordy.
- Rozlisuj presne:
  ISNULL = funkcia na nahradenie NULL hodnoty
  IS NULL = podmienka na kontrolu NULL
  NULLIF = funkcia, ktora pri zhode vracia NULL
- Nepouzivaj doslovne anglicke formulacie.
- Vystup musi byt vhodny na hlasne citanie.
- Pouzi kratke, ciste vety vhodne pre TTS.

REZIM:
{mode}

TYP SEGMENTU:
{seg_type}

SLOT:
{slot:.2f} sekundy

TEXT:
"{text_en}"

Vrat iba finalnu slovensku vetu.
""".strip()


def _call_ollama(prompt: str, *, url: str, model: str, timeout: int) -> str:
    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["response"]).strip()


def _ollama_available(url: str, timeout: int) -> tuple[bool, str]:
    try:
        parts = urlsplit(url)
        path = parts.path or ""
        if path.endswith("/api/generate"):
            path = path[: -len("/api/generate")] + "/api/tags"
        elif not path.endswith("/api/tags"):
            path = path.rstrip("/") + "/api/tags"
        probe_url = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
        response = requests.get(probe_url, timeout=min(timeout, 5))
        response.raise_for_status()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _simple_fallback(text: str) -> str:
    import re

    t = _normalize_spaces(text)
    replacements = [
        (r"\bv poriadku,?\s*priatelia,?\s*", ""),
        (r"\bTakže,\s*", ""),
        (r"\bTakže\b", ""),
        (r"\bDobre, priatelia,\s*", ""),
        (r"\bponoríme sa do\b", "pozrieme sa na"),
        (r"\burobíme prehľad do\b", "pozrieme sa na"),
        (r"\bhlboký ponor\b", "prehľad"),
        (r"\bnášho stola\b", "tabuľky"),
        (r"\bnáš stôl\b", "našu tabuľku"),
        (r"\bhodnotu hodnotou\b", "hodnotu"),
        (r"\bnovou hodnotou, napríklad 40 a in\b", "novou hodnotou, napríklad 40"),
        (r"\bmôžete ísť a skontrolovať\b", "môžete skontrolovať"),
        (r"\bpoďme si dať príklad\b", "pozrime sa na príklad"),
        (r"\bMôžeme použiť funkciu\b", "Môžeme použiť"),
    ]
    for pattern, replacement in replacements:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    return _clean_punctuation(t)


def rewrite_segments_batch(
    segments: list[dict[str, Any]],
    *,
    use_ollama: bool,
    ollama_url: str,
    ollama_model: str,
    ollama_timeout: int = 120,
    target_cps: float = 16.0,
    hard_cps_limit: float = 17.0,
    verbose: bool = False,
    stats_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], MiniLevel11Stats]:
    stats = MiniLevel11Stats()
    out_segments: list[dict[str, Any]] = []
    ollama_enabled = use_ollama

    if use_ollama:
        ok, err = _ollama_available(ollama_url, ollama_timeout)
        if not ok:
            ollama_enabled = False
            if verbose:
                print(f"[MINI-L11] Ollama unavailable ({err}) -> fallback/template only", flush=True)

    for idx, seg in enumerate(segments, start=1):
        seg_out = dict(seg)
        source_text = _normalize_spaces(str(seg_out.get("text_src") or ""))
        current_text = _normalize_spaces(str(seg_out.get("text") or seg_out.get("tts_input") or ""))
        slot = max(0.01, float(seg_out.get("end", 0.0) or 0.0) - float(seg_out.get("start", 0.0) or 0.0))

        stats.total += 1
        mode = _choose_mode(
            slot,
            current_text or source_text,
            target_cps=target_cps,
            hard_cps_limit=hard_cps_limit,
        )
        seg_type = _classify_segment_type(source_text or current_text)
        provider = "fallback"
        rewritten = current_text

        template = source_guided_template(source_text)
        if template:
            rewritten = template
            provider = "source_template"
        elif ollama_enabled and source_text:
            try:
                prompt = _build_prompt(source_text, slot, mode, seg_type)
                rewritten = _call_ollama(
                    prompt,
                    url=ollama_url,
                    model=ollama_model,
                    timeout=ollama_timeout,
                )
                provider = "ollama"
            except Exception as exc:
                provider = "fallback"
                stats.errors += 1
                seg_out["mini_level11_error"] = str(exc)
                rewritten = _simple_fallback(current_text or source_text)
        else:
            rewritten = _simple_fallback(current_text or source_text)

        rewritten = guard_sql_text(rewritten, source_text=source_text, fallback_text=current_text)
        rewritten = guard_linux_text(rewritten, source_text=source_text, fallback_text=current_text)
        seg_out["text"] = rewritten
        seg_out["mini_level11_provider"] = provider
        seg_out["mini_level11_mode"] = mode
        seg_out["mini_level11_segment_type"] = seg_type

        if rewritten != current_text and rewritten:
            stats.changed += 1
        if provider == "ollama":
            stats.ollama += 1
        elif provider == "source_template":
            stats.source_template += 1
        else:
            stats.fallback += 1

        if verbose:
            print(
                f"[MINI-L11] seg {idx} provider={provider} mode={mode} "
                f"text={rewritten[:80]!r}",
                flush=True,
            )
        out_segments.append(seg_out)

    if stats_path:
        stats_file = Path(stats_path)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return out_segments, stats
