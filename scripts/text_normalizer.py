"""
NeMo-inspired text normalizer for Czech and Slovak TTS.

Implements a staged semiotic-class verbalization pipeline:
  1. Dates       (DD.MM.YYYY, YYYY-MM-DD → spoken ordinals)
  2. Times       (HH:MM, HH:MM:SS → spoken)
  3. Percentages (50% → padesát/päťdesiat procent)
  4. Currency    ($50, 100 Kč → verbalized)
  5. Units       (100 km, 5 MB → verbalized)
  6. Ordinals    (1. bod → první/prvý bod)
  7. Cardinals   (1234 → tisíc dvě stě třicet čtyři / tisíc dvestotridsaťštyri)

Architecture mirrors NeMo's Tagger→Verbalizer pipeline but uses
regex-based rules instead of WFST (Weighted Finite-State Transducers),
making it zero-dependency and fast.

Usage:
    from text_normalizer import normalize_for_tts
    text = normalize_for_tts("Dotaz vrátil 1234 řádků.", lang="cs")
    text = normalize_for_tts("Dotaz vrátil 1234 riadkov.", lang="sk")
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Czech cardinal number verbalization (nominative case)
# ---------------------------------------------------------------------------

_ONES_CS = (
    "nula", "jedna", "dva", "tři", "čtyři", "pět", "šest", "sedm", "osm", "devět",
    "deset", "jedenáct", "dvanáct", "třináct", "čtrnáct", "patnáct", "šestnáct",
    "sedmnáct", "osmnáct", "devatenáct",
)
_TENS_CS = (
    "", "", "dvacet", "třicet", "čtyřicet", "padesát",
    "šedesát", "sedmdesát", "osmdesát", "devadesát",
)
_HUNDREDS_CS = (
    "", "sto", "dvě stě", "tři sta", "čtyři sta",
    "pět set", "šest set", "sedm set", "osm set", "devět set",
)

# backward-compat aliases (keep existing internal callers working)
_ONES = _ONES_CS
_TENS = _TENS_CS
_HUNDREDS = _HUNDREDS_CS

# ---------------------------------------------------------------------------
# Slovak cardinal number verbalization (nominative case)
# ---------------------------------------------------------------------------

_ONES_SK = (
    "nula", "jeden", "dva", "tri", "štyri", "päť", "šesť", "sedem", "osem", "deväť",
    "desať", "jedenásť", "dvanásť", "trinásť", "štrnásť", "pätnásť", "šestnásť",
    "sedemnásť", "osemnásť", "devätnásť",
)
_TENS_SK = (
    "", "", "dvadsať", "tridsať", "štyridsať", "päťdesiat",
    "šesťdesiat", "sedemdesiat", "osemdesiat", "deväťdesiat",
)
_HUNDREDS_SK = (
    "", "sto", "dvesto", "tristo", "štyri sto",
    "päťsto", "šesťsto", "sedemsto", "osemsto", "deväťsto",
)


def _verb_int_sk(n: int) -> str:
    """Convert non-negative integer to Slovak cardinal words (nominative)."""
    if n < 0:
        return "mínus " + _verb_int_sk(-n)
    if n < 20:
        return _ONES_SK[n]
    if n < 100:
        t, o = divmod(n, 10)
        return (_TENS_SK[t] + (" " + _ONES_SK[o] if o else "")).strip()
    if n < 1_000:
        h, r = divmod(n, 100)
        return (_HUNDREDS_SK[h] + (" " + _verb_int_sk(r) if r else "")).strip()
    if n < 1_000_000:
        th, r = divmod(n, 1_000)
        if th == 1:
            t_word = "tisíc"
        elif th == 2:
            t_word = "dvetisíc"   # natural Slovak: "dvetisíc devätnásť", nie "dva tisíce"
        elif th in (3, 4):
            t_word = _verb_int_sk(th) + " tisíce"
        else:
            t_word = _verb_int_sk(th) + " tisíc"
        return (t_word + (" " + _verb_int_sk(r) if r else "")).strip()
    if n < 1_000_000_000:
        m, r = divmod(n, 1_000_000)
        if m == 1:
            m_word = "milión"
        elif m in (2, 3, 4):
            m_word = _verb_int_sk(m) + " milióny"
        else:
            m_word = _verb_int_sk(m) + " miliónov"
        return (m_word + (" " + _verb_int_sk(r) if r else "")).strip()
    if n < 1_000_000_000_000:
        b, r = divmod(n, 1_000_000_000)
        if b == 1:
            b_word = "miliarda"
        elif b in (2, 3, 4):
            b_word = _verb_int_sk(b) + " miliardy"
        else:
            b_word = _verb_int_sk(b) + " miliárd"
        return (b_word + (" " + _verb_int_sk(r) if r else "")).strip()
    return str(n)


def _verb_int(n: int, lang: str = "cs") -> str:
    """Convert non-negative integer to Czech or Slovak cardinal words (nominative)."""
    if lang == "sk":
        return _verb_int_sk(n)
    # Czech (default)
    if n < 0:
        return "minus " + _verb_int(-n)
    if n < 20:
        return _ONES_CS[n]
    if n < 100:
        t, o = divmod(n, 10)
        return (_TENS_CS[t] + (" " + _ONES_CS[o] if o else "")).strip()
    if n < 1_000:
        h, r = divmod(n, 100)
        return (_HUNDREDS_CS[h] + (" " + _verb_int(r) if r else "")).strip()
    if n < 1_000_000:
        th, r = divmod(n, 1_000)
        if th == 1:
            t_word = "tisíc"
        elif th in (2, 3, 4):
            t_word = _verb_int(th) + " tisíce"
        else:
            t_word = _verb_int(th) + " tisíc"
        return (t_word + (" " + _verb_int(r) if r else "")).strip()
    if n < 1_000_000_000:
        m, r = divmod(n, 1_000_000)
        if m == 1:
            m_word = "milión"
        elif m in (2, 3, 4):
            m_word = _verb_int(m) + " milióny"
        else:
            m_word = _verb_int(m) + " miliónů"
        return (m_word + (" " + _verb_int(r) if r else "")).strip()
    if n < 1_000_000_000_000:
        b, r = divmod(n, 1_000_000_000)
        if b == 1:
            b_word = "miliarda"
        elif b in (2, 3, 4):
            b_word = _verb_int(b) + " miliardy"
        else:
            b_word = _verb_int(b) + " miliard"
        return (b_word + (" " + _verb_int(r) if r else "")).strip()
    return str(n)  # too large — leave as-is


# ---------------------------------------------------------------------------
# Czech ordinal verbalization (nominative, masculine)
# NeMo ordinal class: "1." → "první", "2." → "druhý", etc.
# ---------------------------------------------------------------------------

_ORD_BASE_CS = {
    1: "první",    2: "druhý",    3: "třetí",    4: "čtvrtý",   5: "pátý",
    6: "šestý",   7: "sedmý",   8: "osmý",    9: "devátý",  10: "desátý",
    11: "jedenáctý", 12: "dvanáctý", 13: "třináctý", 14: "čtrnáctý",
    15: "patnáctý",  16: "šestnáctý", 17: "sedmnáctý", 18: "osmnáctý",
    19: "devatenáctý",
}
_ORD_TENS_CS = {
    2: "dvacátý", 3: "třicátý", 4: "čtyřicátý", 5: "padesátý",
    6: "šedesátý", 7: "sedmdesátý", 8: "osmdesátý", 9: "devadesátý",
}
# backward-compat aliases
_ORD_BASE = _ORD_BASE_CS
_ORD_TENS = _ORD_TENS_CS

# Slovak ordinals
_ORD_BASE_SK = {
    1: "prvý",    2: "druhý",    3: "tretí",    4: "štvrtý",   5: "piaty",
    6: "šiesty",  7: "siedmy",  8: "ôsmy",    9: "deviaty",  10: "desiaty",
    11: "jedenásty", 12: "dvanásty", 13: "trinásty", 14: "štrnásty",
    15: "pätnásty",  16: "šestnásty", 17: "sedemnásty", 18: "osemnásty",
    19: "devätnásty",
}
_ORD_TENS_SK = {
    2: "dvadsiaty", 3: "tridsiaty", 4: "štyridsiaty", 5: "päťdesiaty",  # 40. = štyridsiatý
    6: "šesťdesiaty", 7: "sedemdesiaty", 8: "osemdesiaty", 9: "deväťdesiaty",
}


def _verb_ordinal(n: int, lang: str = "cs") -> str:
    """1-based integer → Czech or Slovak ordinal (masculine, nominative)."""
    if n <= 0:
        return str(n)
    base = _ORD_BASE_SK if lang == "sk" else _ORD_BASE_CS
    tens = _ORD_TENS_SK if lang == "sk" else _ORD_TENS_CS
    if n in base:
        return base[n]
    if n < 100:
        t, o = divmod(n, 10)
        tens_word = tens.get(t, "")
        if o == 0:
            return tens_word
        return (tens_word + " " + base.get(o, _verb_int(o, lang))).strip()
    if n == 100:
        return "stý"
    if n == 1000:
        return "tisíci" if lang == "cs" else "tisíci"
    return _verb_int(n, lang) + "."  # fallback for large ordinals


# ---------------------------------------------------------------------------
# Genitive ordinals for date verbalization (NeMo "date" class)
# Czech dates use genitive: "prvního ledna" (of the first of January)
# ---------------------------------------------------------------------------

_ORD_GEN_CS = {
    1: "prvního",    2: "druhého",   3: "třetího",   4: "čtvrtého",
    5: "pátého",     6: "šestého",   7: "sedmého",   8: "osmého",
    9: "devátého",  10: "desátého", 11: "jedenáctého", 12: "dvanáctého",
    13: "třináctého", 14: "čtrnáctého", 15: "patnáctého", 16: "šestnáctého",
    17: "sedmnáctého", 18: "osmnáctého", 19: "devatenáctého", 20: "dvacátého",
    21: "dvacátého prvního",  22: "dvacátého druhého",  23: "dvacátého třetího",
    24: "dvacátého čtvrtého", 25: "dvacátého pátého",   26: "dvacátého šestého",
    27: "dvacátého sedmého",  28: "dvacátého osmého",   29: "dvacátého devátého",
    30: "třicátého", 31: "třicátého prvního",
}
# backward-compat alias
_ORD_GEN = _ORD_GEN_CS

_ORD_GEN_SK = {
    1: "prvého",    2: "druhého",   3: "tretieho",  4: "štvrtého",
    5: "piateho",   6: "šiesteho",  7: "siedmeho",  8: "ôsmeho",
    9: "deviateho", 10: "desiateho", 11: "jedenásteho", 12: "dvanásteho",
    13: "trinásteho", 14: "štrnásteho", 15: "pätnásteho", 16: "šestnásteho",
    17: "sedemnásteho", 18: "osemnásteho", 19: "devätnásteho", 20: "dvadsiateho",
    21: "dvadsiateho prvého",   22: "dvadsiateho druhého",  23: "dvadsiateho tretieho",
    24: "dvadsiateho štvrtého", 25: "dvadsiateho piateho",  26: "dvadsiateho šiesteho",
    27: "dvadsiateho siedmeho", 28: "dvadsiateho ôsmeho",   29: "dvadsiateho deviateho",
    30: "tridsiateho", 31: "tridsiateho prvého",
}


def _verb_date_part(n: int, lang: str = "cs") -> str:
    """Day/month number → Czech or Slovak genitive ordinal."""
    table = _ORD_GEN_SK if lang == "sk" else _ORD_GEN_CS
    return table.get(n, _verb_ordinal(n, lang) + "ho")


# ---------------------------------------------------------------------------
# 1. DATES (NeMo "date" semiotic class)
# Must run before ordinals and cardinals to prevent partial matching.
# ---------------------------------------------------------------------------

# DD.MM.YYYY or DD. MM. YYYY (with optional spaces around dots)
_DATE_DMY_FULL = re.compile(r'(?<!\d)(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})(?!\d)')
# DD.MM. (partial date — month + day only)
_DATE_DMY_SHORT = re.compile(r'(?<!\d)(\d{1,2})\.\s*(\d{1,2})\.(?!\s*\d)')
# ISO: YYYY-MM-DD
_DATE_ISO = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')


def normalize_dates(text: str, lang: str = "cs") -> str:
    """Convert date strings to Czech or Slovak spoken form (genitive ordinals)."""
    def _dmyf(m: re.Match) -> str:
        d, mo, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= d <= 31 and 1 <= mo <= 12 and 1600 <= yr <= 2100):
            return m.group(0)
        return f"{_verb_date_part(d, lang)} {_verb_date_part(mo, lang)} {_verb_int(yr, lang)}"

    def _dmy(m: re.Match) -> str:
        d, mo = int(m.group(1)), int(m.group(2))
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        return f"{_verb_date_part(d, lang)} {_verb_date_part(mo, lang)}"

    def _iso(m: re.Match) -> str:
        yr, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= d <= 31 and 1 <= mo <= 12 and 1600 <= yr <= 2100):
            return m.group(0)
        return f"{_verb_date_part(d, lang)} {_verb_date_part(mo, lang)} {_verb_int(yr, lang)}"

    text = _DATE_DMY_FULL.sub(_dmyf, text)
    text = _DATE_DMY_SHORT.sub(_dmy, text)
    text = _DATE_ISO.sub(_iso, text)
    return text


# ---------------------------------------------------------------------------
# 2. TIMES (NeMo "time" semiotic class)
# ---------------------------------------------------------------------------

# Match HH:MM or HH:MM:SS — but only when it looks like a time (H ≤ 23, M ≤ 59)
_TIME_HMS = re.compile(r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b')


def normalize_times(text: str, lang: str = "cs") -> str:
    """Convert HH:MM or HH:MM:SS to spoken Czech or Slovak."""
    def _t(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            return m.group(0)
        h_word = _verb_int(h, lang)
        mi_word = ("nula " + _verb_int(mi, lang)) if mi < 10 else _verb_int(mi, lang)
        sec_str = ""
        if m.group(3) is not None:
            s = int(m.group(3))
            if not (0 <= s <= 59):
                return m.group(0)
            sec_str = " " + (("nula " + _verb_int(s, lang)) if s < 10 else _verb_int(s, lang))
        return f"{h_word} {mi_word}{sec_str}"
    return _TIME_HMS.sub(_t, text)


# ---------------------------------------------------------------------------
# 3. PERCENTAGES (NeMo "fraction/percentage" semiotic class)
# ---------------------------------------------------------------------------

# Matches: optional space, integer or decimal, then %
_PCT_PAT = re.compile(r'(\d+(?:[,.]\d+)?)\s*%')


def _verb_digit_sequence(raw: str, lang: str = "cs") -> str:
    table = _ONES_SK if lang == "sk" else _ONES_CS
    return " ".join(table[int(ch)] for ch in raw if ch.isdigit())


def normalize_percentages(text: str, lang: str = "cs") -> str:
    """Convert N% to a spoken percentage, including simple decimals."""
    def _p(m: re.Match) -> str:
        raw = m.group(1).replace(',', '.')
        try:
            f = float(raw)
            n = int(f)
            if float(n) == f:
                return _verb_int(n, lang) + (" percent" if lang == "sk" else " procent")
            int_part, frac_part = raw.split(".", 1)
            frac_part = frac_part.rstrip("0")
            if not frac_part:
                return _verb_int(int(int_part), lang) + (" percent" if lang == "sk" else " procent")
            whole = _verb_int(int(int_part), lang)
            if len(frac_part) == 1:
                frac_words = _verb_int(int(frac_part), lang)
            else:
                frac_words = _verb_digit_sequence(frac_part, lang)
            pct_word = "percent" if lang == "sk" else "procent"
            return f"{whole} celých {frac_words} {pct_word}"
        except ValueError:
            return m.group(0)
    return _PCT_PAT.sub(_p, text)


# ---------------------------------------------------------------------------
# 4. CURRENCY (NeMo "money" semiotic class)
# ---------------------------------------------------------------------------

def normalize_currency(text: str, lang: str = "cs") -> str:
    """Verbalize common currency symbols + amounts."""
    def _verb_amount(raw: str) -> str:
        raw = raw.strip().replace('\xa0', '').replace(' ', '').replace(',', '.')
        try:
            f = float(raw)
            n = int(f)
            if float(n) == f:
                return _verb_int(n, lang)
            return raw
        except ValueError:
            return raw

    _dol = "dolárov" if lang == "sk" else "dolarů"
    _lib = "libier"  if lang == "sk" else "liber"
    _kor = "korún"   if lang == "sk" else "korun"
    text = re.sub(r'\$\s*(\d[\d,.]*)', lambda m: _verb_amount(m.group(1)) + f" {_dol}", text)
    text = re.sub(r'€\s*(\d[\d,.]*)',  lambda m: _verb_amount(m.group(1)) + " eur", text)
    text = re.sub(r'£\s*(\d[\d,.]*)',  lambda m: _verb_amount(m.group(1)) + f" {_lib}", text)
    text = re.sub(r'(\d[\d,.]*)\s*Kč\b',  lambda m: _verb_amount(m.group(1)) + f" {_kor}", text)
    text = re.sub(r'(\d[\d,.]*)\s*CZK\b', lambda m: _verb_amount(m.group(1)) + f" {_kor}", text)
    text = re.sub(r'(\d[\d,.]*)\s*EUR\b', lambda m: _verb_amount(m.group(1)) + " eur", text)
    text = re.sub(r'(\d[\d,.]*)\s*USD\b', lambda m: _verb_amount(m.group(1)) + f" {_dol}", text)
    return text


# ---------------------------------------------------------------------------
# 5. MEASUREMENT UNITS (NeMo "measure" semiotic class)
# Pattern: NUMBER + optional space + UNIT
# Must run before cardinal normalization so numbers are still digits.
# ---------------------------------------------------------------------------

# (unit_regex, spoken_form)
# Longer/more specific patterns first to avoid partial matches
_UNIT_PATTERNS_CS = [
    # Data transfer rates
    (r'[Gg][Bb]/s\b',    "gigabajtů za sekundu"),
    (r'[Mm][Bb]/s\b',    "megabajtů za sekundu"),
    (r'[Kk][Bb]/s\b',    "kilobajtů za sekundu"),
    # Data sizes
    (r'[Tt][Bb]\b',      "terabajtů"),
    (r'[Gg][Bb]\b',      "gigabajtů"),
    (r'[Mm][Bb]\b',      "megabajtů"),
    (r'[Kk][Bb]\b',      "kilobajtů"),
    (r'[Bb]\b',          "bajtů"),
    # Frequency
    (r'[Gg][Hh][Zz]\b',  "gigahertzů"),
    (r'[Mm][Hh][Zz]\b',  "megahertzů"),
    (r'[Kk][Hh][Zz]\b',  "kilohertzů"),
    (r'[Hh][Zz]\b',      "hertzů"),
    # Speed
    (r'[Kk][Mm]/h\b',    "kilometrů za hodinu"),
    (r'[Kk][Mm]/s\b',    "kilometrů za sekundu"),
    (r'm/s\b',           "metrů za sekundu"),
    # Distance
    (r'[Kk][Mm]\b',      "kilometrů"),
    (r'[Cc][Mm]\b',      "centimetrů"),
    (r'[Mm][Mm]\b',      "milimetrů"),
    (r'(?<![a-zA-Z])m\b', "metrů"),
    # Mass
    (r'[Kk][Gg]\b',      "kilogramů"),
    (r'[Dd][Gg]\b',      "dekagramů"),
    (r'[Mm][Gg]\b',      "miligramů"),
    (r'(?<![a-zA-Z])g\b', "gramů"),
    # Time (careful: s/ms AFTER distance m to avoid conflicts)
    (r'[Mm][Ss]\b',      "milisekund"),
    (r'[Mm][Ii][Nn]\b',  "minut"),
    (r'(?<![a-zA-Z])s\b', "sekund"),
    # Power
    (r'[Kk][Ww]\b',      "kilowattů"),
    (r'[Mm][Ww]\b',      "megawattů"),
    (r'(?<![a-zA-Z])W\b', "wattů"),
    # Temperature
    (r'°[Cc]\b',         "stupňů Celsia"),
    (r'°[Ff]\b',         "stupňů Fahrenheita"),
    (r'°[Kk]\b',         "kelvinů"),
    # Resolution / pixels
    (r'[Mm][Pp]\b',      "megapixelů"),
    (r'[Kk]\b',          "tisíc"),  # e.g. "10K followers" — handled last
]

_UNIT_PATTERNS_SK = [
    # Data transfer rates
    (r'[Gg][Bb]/s\b',    "gigabajtov za sekundu"),
    (r'[Mm][Bb]/s\b',    "megabajtov za sekundu"),
    (r'[Kk][Bb]/s\b',    "kilobajtov za sekundu"),
    # Data sizes
    (r'[Tt][Bb]\b',      "terabajtov"),
    (r'[Gg][Bb]\b',      "gigabajtov"),
    (r'[Mm][Bb]\b',      "megabajtov"),
    (r'[Kk][Bb]\b',      "kilobajtov"),
    (r'[Bb]\b',          "bajtov"),
    # Frequency
    (r'[Gg][Hh][Zz]\b',  "gigahertzov"),
    (r'[Mm][Hh][Zz]\b',  "megahertzov"),
    (r'[Kk][Hh][Zz]\b',  "kilohertzov"),
    (r'[Hh][Zz]\b',      "hertzov"),
    # Speed
    (r'[Kk][Mm]/h\b',    "kilometrov za hodinu"),
    (r'[Kk][Mm]/s\b',    "kilometrov za sekundu"),
    (r'm/s\b',           "metrov za sekundu"),
    # Distance
    (r'[Kk][Mm]\b',      "kilometrov"),
    (r'[Cc][Mm]\b',      "centimetrov"),
    (r'[Mm][Mm]\b',      "milimetrov"),
    (r'(?<![a-zA-Z])m\b', "metrov"),
    # Mass
    (r'[Kk][Gg]\b',      "kilogramov"),
    (r'[Dd][Gg]\b',      "dekagramov"),
    (r'[Mm][Gg]\b',      "miligramov"),
    (r'(?<![a-zA-Z])g\b', "gramov"),
    # Time
    (r'[Mm][Ss]\b',      "milisekúnd"),
    (r'[Mm][Ii][Nn]\b',  "minút"),
    (r'(?<![a-zA-Z])s\b', "sekúnd"),
    # Power
    (r'[Kk][Ww]\b',      "kilowattov"),
    (r'[Mm][Ww]\b',      "megawattov"),
    (r'(?<![a-zA-Z])W\b', "wattov"),
    # Temperature
    (r'°[Cc]\b',         "stupňov Celzia"),
    (r'°[Ff]\b',         "stupňov Fahrenheita"),
    (r'°[Kk]\b',         "kelvinov"),
    # Resolution / pixels
    (r'[Mm][Pp]\b',      "megapixelov"),
    (r'[Kk]\b',          "tisíc"),
]

# backward-compat alias
_UNIT_PATTERNS = _UNIT_PATTERNS_CS


def normalize_units(text: str, lang: str = "cs") -> str:
    """Convert number+unit combinations to spoken Czech or Slovak."""
    patterns = _UNIT_PATTERNS_SK if lang == "sk" else _UNIT_PATTERNS_CS

    def _make_repl(unit_word: str):
        def _repl(m: re.Match) -> str:
            raw = m.group(1).replace('\xa0', '').replace(' ', '').replace(',', '.')
            try:
                f = float(raw)
                n = int(f)
                n_str = _verb_int(n, lang) if float(n) == f else raw
                return n_str + " " + unit_word
            except ValueError:
                return m.group(0)
        return _repl

    for unit_pat, unit_word in patterns:
        full_pat = r'(\d+(?:[,.]\d+)?)\s*' + unit_pat
        text = re.sub(full_pat, _make_repl(unit_word), text)

    return text


# ---------------------------------------------------------------------------
# 6. ORDINALS (NeMo "ordinal" semiotic class)
# Pattern: isolated digit(s) followed by period + space + non-digit word
# Must run AFTER dates (to avoid consuming date parts) and
# BEFORE cardinals (to avoid converting the digit first).
# ---------------------------------------------------------------------------

# Match: 1-3 digit number + "." + whitespace + letter (not digit)
# Not preceded by another digit or dot (avoids matching "3.14" or "1.2.")
# Match: digit(s) + "." not preceded by digit/dot (avoids "3.14", "1.2.")
# Lookahead REQUIRES whitespace then a non-digit character (Czech word follows).
# We do NOT match at end-of-string to avoid converting sentence-ending numbers
# like "id > 100." where 100 is a cardinal, not an ordinal.
_ORDINAL_PAT = re.compile(
    r'(?<![.\d])(\d{1,3})\.(?=\s+[^\d\s])'
)


def normalize_ordinals(text: str, lang: str = "cs") -> str:
    """Convert written ordinals ('1.', '2.') to Czech or Slovak spoken ordinals."""
    def _o(m: re.Match) -> str:
        n = int(m.group(1))
        if n < 1 or n > 999:
            return m.group(0)
        return _verb_ordinal(n, lang)

    return _ORDINAL_PAT.sub(_o, text)


# ---------------------------------------------------------------------------
# 7. CARDINALS (NeMo "cardinal" semiotic class)
# Convert standalone integers to Czech words.
# Skips: numbers inside identifiers, decimal numbers, version strings.
# ---------------------------------------------------------------------------

# Pre-pass: numbers with space/NBSP thousands separators, e.g. "1 234 567" → 1234567
# Pattern: 1-3 digits, followed by one or more groups of (space/NBSP + 3 digits)
_THOUSANDS_SEP_PAT = re.compile(r'\b(\d{1,3})((?:[\s\xa0]\d{3})+)\b')


def _collapse_thousands(text: str) -> str:
    """Combine '1 234 567' style numbers into a single token for verbalization."""
    def _repl(m: re.Match) -> str:
        combined = m.group(1) + m.group(2).replace('\xa0', '').replace(' ', '')
        return combined
    return _THOUSANDS_SEP_PAT.sub(_repl, text)


# Match standalone integers.
# Lookbehind: not preceded by digit+dot (avoids "14" in "3.14") or slash/backslash.
# Lookahead: not followed by .\d (avoids "3" in "3.14"), or slash/backslash.
# Sentence-ending periods (digit followed by "." then non-digit) ARE allowed.
_CARDINAL_PAT = re.compile(r'(?<!\d\.)(?<![/\\])\b(\d+)\b(?![/\\])(?!\.\d)')


def normalize_cardinals(text: str, lang: str = "cs") -> str:
    """Convert standalone integers to Czech or Slovak cardinal words."""
    text = _collapse_thousands(text)

    def _c(m: re.Match) -> str:
        n = int(m.group(1))
        if n > 999_999_999_999:
            return m.group(0)
        return _verb_int(n, lang)

    return _CARDINAL_PAT.sub(_c, text)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def normalize_for_tts(
    text: str,
    content_type: str | None = None,
    lang: str = "cs",
    mode: str = "full",
) -> str:
    """
    NeMo-inspired staged text normalization for Czech (lang='cs') or Slovak (lang='sk') TTS.

    mode="full"  — all stages (legacy behaviour)
    mode="light" — only dates, times, percentages, cardinals
                   (safe for production dubbing — no aggressive unit/currency/ordinal rewrites)

    Pipeline (order matters — each stage expects previous stages done):
    1. Dates      — must run before ordinals/cardinals (date patterns use digits+dots)
    2. Times      — after dates (time HH:MM must not match date partial)
    3. Percentages — before cardinals (N% → 'N procent')
    4. Currency   — before cardinals       [full only]
    5. Units      — before cardinals       [full only]
    6. Ordinals   — before cardinals       [full only]
    7. Cardinals  — last number pass (remaining standalone digits)
    8. Cleanup    — collapse extra whitespace
    """
    if not text:
        return text

    text = normalize_dates(text, lang)
    text = normalize_times(text, lang)
    text = normalize_percentages(text, lang)
    if mode != "light":
        text = normalize_currency(text, lang)
        text = normalize_units(text, lang)
        text = normalize_ordinals(text, lang)
    text = normalize_cardinals(text, lang)
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()

    return text
