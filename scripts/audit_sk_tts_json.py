"""
audit_sk_tts_json.py
--------------------
Vrstva 1: scoring + flagging SK segmentov pred TTS.
Vrstva 2: bezpečné regex fixy (úvodzovky, glossary, whitespace).

Výstup:
  <stem>_sk_segments_audited.json  — opravené + oflagnúté segmenty
  <stem>_sk_audit_report.txt       — ľudsky čitateľný report

Použitie:
  python3 audit_sk_tts_json.py input_sk_segments.json
  python3 audit_sk_tts_json.py input_sk_segments.json --fix   # aj aplikuje bezpečné fixy
  python3 audit_sk_tts_json.py input_sk_segments.json --fix --report
"""

import re
import json
import sys
import argparse
from pathlib import Path
from difflib import SequenceMatcher

# Centrálny glossary — jeden zdroj pravdy
try:
    from sk_glossary import postfix_rules as _sk_postfix_rules, audit_bad_terms as _sk_bad_terms
    _USE_CENTRAL_GLOSSARY = True
except ImportError:
    _USE_CENTRAL_GLOSSARY = False

_WEIRD_QUOTE_FUNC = re.compile(
    r'[„"](na|v|z|do|od|po|pre|nie|a|ale|že|s|zo|za|ku|k|pri|vo|nad|pod|o|si)[""]',
    re.IGNORECASE,
)
_BARE_QUOTE = re.compile(r'[„"]([A-Za-zÁ-Žá-ž0-9._+\- ]{1,30})[""]')


def apply_safe_fixes(text: str) -> tuple[str, list[str]]:
    """Aplikuje bezpečné regex fixy. Vracia (opravený text, zoznam aplikovaných fixov)."""
    applied = []

    # 1) osamotené quoted function words
    new = _WEIRD_QUOTE_FUNC.sub(lambda m: m.group(1), text)
    if new != text:
        applied.append("removed_quoted_function_words")
        text = new

    # 2) úvodzovky okolo bežných slov
    new = _BARE_QUOTE.sub(lambda m: m.group(1), text)
    if new != text:
        applied.append("stripped_bare_quotes")
        text = new

    # 3) glossary (centrálny zdroj ak dostupný)
    if _USE_CENTRAL_GLOSSARY:
        for compiled_re, repl in _sk_postfix_rules():
            new = compiled_re.sub(repl, text)
            if new != text:
                applied.append(f"glossary:{repl[:30]}")
                text = new
    else:
        # fallback inline pravidlá
        _inline = [
            (r'\bčasový kód\b', 'časová značka'),
            (r'\bčasovým kódom\b', 'časovou značkou'),
            (r'\bnázov\s+hostiteľa\b', 'hostname'),
            (r'\bPrize\s+Repository\b', 'repozitár bez predplatného'),
        ]
        for pat, repl in _inline:
            new = re.sub(pat, repl, text, flags=re.IGNORECASE)
            if new != text:
                applied.append(f"glossary:{repl[:30]}")
                text = new

    # 4) whitespace cleanup
    new = re.sub(r'\s+', ' ', text).strip()
    new = re.sub(r'\s+([,.;:!?])', r'\1', new)
    if new != text:
        applied.append("whitespace_cleanup")
        text = new

    return text, applied


# ---------------------------------------------------------------------------
# VRSTVA 1 — scoring + flagging
# ---------------------------------------------------------------------------

_SK_CHARS = set('áäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ')
def _get_bad_terms():
    if _USE_CENTRAL_GLOSSARY:
        return _sk_bad_terms()
    # fallback
    return [
        (re.compile(r'\bzásob[au]?\s+energie\b', re.IGNORECASE), 'bad_term:power_supply'),
        (re.compile(r'\bPrize\s+Repository\b',   re.IGNORECASE), 'bad_term:prize_repository'),
        (re.compile(r'\bčasový kód\b'),                          'bad_term:time_code'),
        (re.compile(r'\bčasovým kódom\b'),                       'bad_term:time_code'),
        (re.compile(r'\bnázov\s+hostiteľa\b',    re.IGNORECASE), 'bad_term:hostname'),
        (re.compile(r'\bflash\w*\b',             re.IGNORECASE), 'bad_term:flash_untranslated'),
    ]

_BAD_TERMS_CACHE = None

def _bad_terms():
    global _BAD_TERMS_CACHE
    if _BAD_TERMS_CACHE is None:
        _BAD_TERMS_CACHE = _get_bad_terms()
    return _BAD_TERMS_CACHE

_REPETITION_RE = re.compile(
    r'(.{15,})\1',
)

_HALLUCINATION_PHRASES = [
    "môžete ich použiť. Ak máte niečo",
    "Ak chcete vidieť, môžete ich použiť",
]

_SUSPICIOUS_NAME_DECLENSION = [
    re.compile(
        r'\b(?:oslovil(?:i|a|o)?|spoznal(?:i|a|o)?|poznal(?:i|a|o)?|'
        r'našiel(?:i|a|o)?|hľadal(?:i|a|o)?|videl(?:i|a|o)?|'
        r'miloval(?:i|a|o)?|nenávidel(?:i|a|o)?|zabil(?:i|a|o)?|'
        r'porazil(?:i|a|o)?|zničil(?:i|a|o)?|porezal(?:i|a|o)?|'
        r'privolal(?:i|a|o)?|poslal(?:i|a|o)?|vtiahol(?:i|a|o)?|'
        r'ponúkol(?:i|a|o)?|vytiahol(?:i|a|o)?|všiml(?:o|a|i)?(?:\s+si)?)\s+'
        r'(?:Kratos|Jason|Freddy|Shinnok|Predator|Terminator|Nightwolf|RoboCop|Raiden|Fujin)\b'
    ),
    re.compile(r'\btíto\s+bohovia\s+Kratos\s+nikdy\s+nemilovali\b', re.IGNORECASE),
]


def _topic_words(text: str) -> set:
    """Extrahuje podstatné slová pre tematickú zhodu."""
    words = re.findall(r'\b[A-Za-zÁ-Žá-ž]{4,}\b', text.lower())
    return set(words)


def _topic_similarity(src: str, tgt: str) -> float:
    """Jednoduchá miera tematickej podobnosti cez SequenceMatcher."""
    return SequenceMatcher(None, src.lower()[:120], tgt.lower()[:120]).ratio()


def score_segment(seg: dict, idx: int, prev_seg: dict | None) -> dict:
    """Ohodnotí jeden segment. Vracia dict s flags, score, reasons."""
    text = (seg.get("text") or "").strip()
    src  = (seg.get("text_src") or "").strip()
    start = seg.get("start", 0)
    end   = seg.get("end", 0)
    duration = end - start

    flags   = []
    score   = 0  # vyšší = viac problémov

    # --- prázdny segment ---
    if not text:
        flags.append("empty_text")
        score += 5

    # --- chýbajúce SK znaky (podozrenie na nepreložené) ---
    if text and len(text) > 10:
        sk_ratio = sum(1 for c in text if c in _SK_CHARS) / len(text)
        en_words = re.findall(r'\b[a-z]{4,}\b', text)
        if sk_ratio < 0.02 and len(en_words) > 2:
            flags.append("possibly_untranslated")
            score += 4

    # --- rozbité úvodzovky ---
    if re.search(r'[„"](?:na|v|z|do|od|po|pre|nie)[""]', text, re.IGNORECASE):
        flags.append("weird_quotes")
        score += 2
    elif re.search(r'[„"][^"„]{1,30}[""]', text):
        flags.append("bare_quotes")
        score += 1

    # --- zlé termíny ---
    for compiled_re, flag in _bad_terms():
        if compiled_re.search(text):
            flags.append(flag)
            score += 3

    # --- repetícia/fragment ---
    if _REPETITION_RE.search(text):
        flags.append("repetition")
        score += 4

    for phrase in _HALLUCINATION_PHRASES:
        if phrase in text:
            flags.append("possible_hallucination")
            score += 5

    for pat in _SUSPICIOUS_NAME_DECLENSION:
        if pat.search(text):
            flags.append("possible_name_declension")
            score += 3
            break

    # --- alignment drift (tematická nesúvislosť) ---
    if src and text:
        sim = _topic_similarity(src, text)
        if sim < 0.15:
            flags.append("possible_alignment_drift")
            score += 3

    # --- fragmentácia (veľmi krátky text, ale dlhý slot) ---
    if text and len(text.split()) <= 2 and duration > 2.0:
        flags.append("fragmented")
        score += 2

    # --- extrémne dlhý preklad voči slotu ---
    slot_chars = duration * 14.5  # SK_CHARS_PER_SEC
    if text and len(text) > slot_chars * 1.5:
        flags.append("too_long_for_slot")
        score += 2

    return {
        "idx": idx,
        "start": start,
        "end": end,
        "score": score,
        "flags": flags,
        "text_src": src,
        "text": text,
    }


def audit_json(path: Path, apply_fixes: bool = False) -> tuple[list[dict], list[dict]]:
    """Načíta JSON, ohodnotí segmenty, voliteľne aplikuje fixy."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    segs = data["segments"] if isinstance(data, dict) else data
    audited = []
    audit_results = []

    prev = None
    for i, seg in enumerate(segs):
        result = score_segment(seg, i + 1, prev)
        audit_results.append(result)

        new_seg = dict(seg)
        fixes_applied = []

        if apply_fixes and new_seg.get("text"):
            fixed_text, fixes_applied = apply_safe_fixes(new_seg["text"])
            new_seg["text"] = fixed_text

        new_seg["_audit"] = {
            "score": result["score"],
            "flags": result["flags"],
            "fixes": fixes_applied,
        }
        audited.append(new_seg)
        prev = seg

    return audited, audit_results


def compute_metrics(audit_results: list[dict], repaired_count: int = 0) -> dict:
    """Vypočíta metriky pre jeden beh. Ukladá sa do *_sk_metrics.json."""
    from collections import Counter
    import datetime
    all_flags = [f for r in audit_results for f in r["flags"]]
    flag_counts = dict(Counter(all_flags))
    flagged = [r for r in audit_results if r["score"] > 0]
    bad_term_count = sum(1 for f in all_flags if f.startswith("bad_term:"))
    hallucination_count = sum(1 for f in all_flags if "hallucination" in f)
    drift_count = sum(1 for f in all_flags if "alignment_drift" in f)
    empty_count = sum(1 for f in all_flags if f == "empty_text")
    too_long_count = sum(1 for f in all_flags if f == "too_long_for_slot")
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "total_segments": len(audit_results),
        "flagged_count": len(flagged),
        "clean_count": len(audit_results) - len(flagged),
        "flagged_pct": round(100 * len(flagged) / max(len(audit_results), 1), 1),
        "bad_term_count": bad_term_count,
        "hallucination_count": hallucination_count,
        "alignment_drift_count": drift_count,
        "empty_count": empty_count,
        "too_long_for_slot_count": too_long_count,
        "gemma_repaired_count": repaired_count,
        "flag_breakdown": flag_counts,
    }


def write_metrics(metrics: dict, out_path: Path) -> None:
    """Uloží metriky do JSON súboru (append do histórie)."""
    history = []
    if out_path.exists():
        try:
            history = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(metrics)
    out_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(audit_results: list[dict], out_path: Path) -> None:
    flagged = [r for r in audit_results if r["score"] > 0]
    flagged.sort(key=lambda r: -r["score"])

    lines = [
        "=" * 70,
        f"SK TTS AUDIT REPORT — {len(audit_results)} segmentov",
        f"Flagged: {len(flagged)}  |  Clean: {len(audit_results) - len(flagged)}",
        "=" * 70,
        "",
    ]

    # Súhrn flagov
    from collections import Counter
    all_flags = [f for r in audit_results for f in r["flags"]]
    counts = Counter(all_flags)
    lines.append("SÚHRN FLAGOV:")
    for flag, cnt in counts.most_common():
        lines.append(f"  {cnt:3d}x  {flag}")
    lines.append("")

    # Detail — len skóre > 2
    lines.append("DETAIL (skóre > 2, zoradené od najhoršieho):")
    lines.append("-" * 70)
    for r in flagged:
        if r["score"] <= 2:
            continue
        lines.append(
            f"[seg {r['idx']:3d}]  score={r['score']}  {r['start']:.1f}s–{r['end']:.1f}s"
        )
        lines.append(f"  FLAGS : {', '.join(r['flags'])}")
        lines.append(f"  SRC   : {r['text_src'][:90]}")
        lines.append(f"  SK    : {r['text'][:90]}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[AUDIT] Report: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Audit SK TTS segmentov")
    p.add_argument("input", help="Cesta k _sk_segments.json")
    p.add_argument("--fix",    action="store_true", help="Aplikuj bezpečné fixy")
    p.add_argument("--report", action="store_true", help="Vygeneruj textový report")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[CHYBA] Súbor neexistuje: {in_path}"); sys.exit(1)

    print(f"[AUDIT] Načítavam: {in_path}")
    audited, audit_results = audit_json(in_path, apply_fixes=args.fix)

    flagged = [r for r in audit_results if r["score"] > 0]
    print(f"[AUDIT] {len(audit_results)} segmentov | flagged: {len(flagged)}")

    from collections import Counter
    all_flags = [f for r in audit_results for f in r["flags"]]
    for flag, cnt in Counter(all_flags).most_common():
        print(f"  {cnt:3d}x  {flag}")

    # Uložiť opravený JSON
    stem = in_path.stem.replace("_sk_segments", "")
    out_json = in_path.parent / f"{stem}_sk_segments_audited.json"
    with open(out_json, "w", encoding="utf-8") as f:
        payload = {"segments": audited} if isinstance(json.loads(in_path.read_text()), dict) else audited
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[AUDIT] Výstup JSON: {out_json}")

    if args.report:
        out_report = in_path.parent / f"{stem}_sk_audit_report.txt"
        write_report(audit_results, out_report)


if __name__ == "__main__":
    main()
