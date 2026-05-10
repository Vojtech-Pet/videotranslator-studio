"""
phonetic_guard.py — ochrana technických termínov pred phonetic vrstvou.

Logika:
- PROTECTED_EXACT / PROTECTED_REGEX: SQL, NULL, COALESCE, TRUE/FALSE → NEDOTÝKAŤ SA
- should_skip_phonetic(text): ak text obsahuje protected term → preskočiť celú phonetic vrstvu
- replace_outside_protected(): nahradí len mimo chránených oblastí
- process_segment(): drop-in pre pipeline debug/test
"""
import re
from typing import Dict, Any, List, Tuple

PROTECTED_EXACT = [
    "SQL",
    "NULL",
    "IS NULL",
    "IS NOT NULL",
    "ISNULL",
    "COALESCE",
    "TRUE",
    "FALSE",
    "BOOLEAN",
]

PROTECTED_REGEX = [
    r"\bSQL\b",
    r"\bNULL\b",
    r"\bIS\s+NULL\b",
    r"\bIS\s+NOT\s+NULL\b",
    r"\bISNULL\b",
    r"\bCOALESCE\b",
    r"\bTRUE\b",
    r"\bFALSE\b",
    r"\bBOOLEAN\b",
]

NUMBER_PATTERNS = [
    r"\b\d+\b",
    r"\b\d+[.,]\d+\b",
]

PHONETIC_MAP: Dict[str, str] = {
    # Sem dávaj IBA veci ktoré TTS nezvláda — NIE SQL keywordy
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains_protected_term(text: str) -> bool:
    for pattern in PROTECTED_REGEX:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def find_protected_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for pattern in PROTECTED_REGEX:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged: List[List[int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(s, e) for s, e in merged]


def is_inside_spans(index: int, spans: List[Tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def replace_outside_protected(text: str, replacements: Dict[str, str]) -> str:
    """Nahradí text podľa `replacements` ale preskočí chránené oblasti."""
    spans = find_protected_spans(text)
    result = []
    i = 0
    while i < len(text):
        if is_inside_spans(i, spans):
            for start, end in spans:
                if start <= i < end:
                    result.append(text[i:end])
                    i = end
                    break
            continue

        replaced = False
        for src, dst in replacements.items():
            if text[i:].startswith(src):
                overlap = any(i < end and i + len(src) > start for start, end in spans)
                if not overlap:
                    result.append(dst)
                    i += len(src)
                    replaced = True
                    break

        if not replaced:
            result.append(text[i])
            i += 1

    return "".join(result)


def should_skip_phonetic(text: str) -> bool:
    """True ak text obsahuje SQL/technické termíny → preskočiť phonetic vrstvu."""
    return contains_protected_term(text)


def safe_number_normalization(text: str) -> str:
    """Príklad: jednoduchá číselná normalizácia mimo protected oblastí."""
    replacements = {
        " 40 ": " štyridsať ",
        "(40)": "(štyridsať)",
        " 0 ": " nula ",
    }
    padded = f" {text} "
    for src, dst in replacements.items():
        padded = padded.replace(src, dst)
    return normalize_spaces(padded)


def apply_phonetic_layer(text: str) -> str:
    """
    Aplikuje phonetic vrstvu s ochranou technických termínov.
    Ak text obsahuje SQL/protected → vráti text bez zmeny.
    """
    if should_skip_phonetic(text):
        return text
    return normalize_spaces(replace_outside_protected(text, PHONETIC_MAP))


MAX_ADAPT_RATIO:   float = 1.40
MAX_CHARS_PER_SEC: float = 18.5

# Generické filler-slová / nadbytočné frázy — bezpečné odstrániť v SK/CS
_SHORTEN_RULES = [
    (r"\bV poriadku,\s*",              ""),
    (r"\bpriatelia,?\s*",              ""),
    (r"\btakže\b",                     ""),
    (r"\bteraz\b",                     ""),
    (r"\bskutočne\b",                  ""),
    (r"\bveľmi\b",                     ""),
    (r"\bnaozaj\b",                    ""),
    (r"\bv podstate\b",                ""),
    (r"\bpomocou toho\b",              ""),
    (r"\bpoďme si\b",                  "poďme"),
    (r"\bMôžeme ísť a\b",             "Môžeme"),
    (r"\bIdeme a\b",                   ""),
    (r"\burobíme hlboký ponor do\b",   "sa pozrieme na"),
    (r"\bhlboký ponor\b",              "pohľad"),
    (r"\bv našich údajoch\b",          "v dátach"),
    (r"\bv našej databáze\b",          "v databáze"),
    (r"\bnášho stola\b",               "tabuľky"),
    (r"\bv podstate\b",                ""),
    # CS variants
    (r"\bvlastně\b",                   ""),
    (r"\bprostě\b",                    ""),
    (r"\btakže\b",                     ""),
    (r"\bteď\b",                       ""),
    (r"\bno takže\b",                  ""),
]


def needs_rewrite(
    adapt_ratio: float,
    chars_per_sec: float,
    *,
    max_adapt_ratio: float = MAX_ADAPT_RATIO,
    max_chars_per_sec: float = MAX_CHARS_PER_SEC,
) -> bool:
    """True ak segment je príliš dlhý pre TTS slot."""
    return adapt_ratio >= max_adapt_ratio or chars_per_sec >= max_chars_per_sec


def shorten_text(text: str) -> str:
    """
    Generické skrátenie textu odstraňovaním filler slov.
    Zachováva chránené termíny (SQL, NULL, atď.).
    Používa sa ako fallback keď adapt_for_timing() už prebehol ale text je stále dlhý.
    """
    out = text
    for pattern, replacement in _SHORTEN_RULES:
        candidate = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        candidate = normalize_spaces(candidate)
        if len(candidate) < len(out):
            out = candidate
    # prvé písmeno veľké
    if out:
        out = out[0].upper() + out[1:]
    # oprav zdvojené medzery / interpunkciu
    out = re.sub(r"\s+([,.:;!?])", r"\1", out)
    out = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", out)
    return normalize_spaces(out)


def process_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop-in pre pipeline: spracuje jeden segment cez phonetic guard.
    Vráti seg s debug poľami: after_normalize, after_phonetic, phonetic_skipped.
    """
    translated = normalize_spaces(seg.get("translated", ""))

    after_normalize = safe_number_normalization(translated)
    after_phonetic = apply_phonetic_layer(after_normalize)

    seg["after_normalize"] = after_normalize
    seg["after_phonetic"] = after_phonetic
    seg["after_sk_fixes"] = after_phonetic
    seg["tts_input"] = after_phonetic
    seg["phonetic_skipped"] = should_skip_phonetic(translated)

    return seg


# ── Integrácia do pipeline.py ──────────────────────────────────────────────
# Pridaj PRED after_phonetic krok v pipeline.py:
#
#   from phonetic_guard import should_skip_phonetic
#
#   if should_skip_phonetic(text):
#       _step["after_phonetic"] = text  # bez zmeny
#       _step["phonetic_skipped"] = True
#   else:
#       text = apply_phonetic_respelling(text, skip_sql_phonetic=use_s2)
#       _step["after_phonetic"] = text
