"""
linux_logic_guard.py
===================
Shared guard for Linux / CLI / Proxmox terminology in Slovak technical dubbing.

Goal:
- preserve exact product and command names such as Proxmox, SSH, SMTP
- keep literal commands and paths like apt update, unattended-upgrades, /etc/apt
- repair common broken phonetic outputs created by translation/TTS prep layers
"""

from __future__ import annotations

import re

LINUX_PROTECTED_TERMS = (
    "Proxmox",
    "Proxmox VE",
    "Debian",
    "Ubuntu",
    "SSH",
    "SMTP",
    "e-mail",
    "email",
    "apt",
    "apt update",
    "apt upgrade",
    "apt install",
    "apt full-upgrade",
    "apt-get",
    "apt-get update",
    "apt-get upgrade",
    "unattended-upgrades",
    "dpkg-reconfigure",
    "systemctl",
    "systemd",
    "apt.conf.d",
    "root",
    "sudo",
    "/etc/apt",
    "/etc/apt/apt.conf.d",
)

_LINUX_MIXED_TERM_REPLACEMENTS = (
    (r"\bnainštalovan\w*\s+bezobslužn\w*\s+(?:upgrade\w*|aktualizáci\w*|inováci\w*)\b", "nainštalovaný unattended-upgrades"),
    (r"\bnastavenie\s+bezobslužn\w*\s+(?:upgrade\w*|aktualizáci\w*|inováci\w*)\b", "nastavenie unattended-upgrades"),
    (r"\bnastaviť\s+bezobslužn\w*\s+(?:upgrade\w*|aktualizáci\w*|inováci\w*)\b", "nastaviť unattended-upgrades"),
    (r"\bbezobslužn\w*\s+(?:upgrade\w*|aktualizáci\w*|inováci\w*)\b", "unattended-upgrades"),
    (r"\bnastaviť\s+unattended-upgrades\b", "nastaviť unattended-upgrades"),
    (r"\bnastavenie\s+unattended-upgrades\b", "nastavenie unattended-upgrades"),
    (r"\bnainštalované\s+unattended-upgrades\b", "nainštalovaný unattended-upgrades"),
    (r"\bnainštalovan[ýé]\s+bal[ií]k\s+unattended-upgrades\b", "nainštalovaný balík unattended-upgrades"),
)

_LINUX_DYNAMIC_PATTERNS = (
    r"/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+",
    r"\bapt(?:-get)?(?:\s+(?:update|upgrade|install|full-upgrade|dist-upgrade|autoremove|remove|purge))?\b",
    r"\bsystemctl(?:\s+[A-Za-z0-9_.@:/-]+){0,3}\b",
    r"\bdpkg-reconfigure(?:\s+[A-Za-z0-9_.@:/-]+){0,2}\b",
    r"\bunattended-upgrades?\b",
    r"\b[A-Za-z0-9_.-]+\.(?:service|timer)\b",
)

_LINUX_FIX_REPLACEMENTS = (
    (r"\bproks?moks\b", "Proxmox"),
    (r"\bes[\s-]*es[\s-]*h[aá]\b", "SSH"),
    (r"\bes[\s-]*em[\s-]*t[eé][\s-]*p[eé]\b", "SMTP"),
    (r"\b[íi]mejl\b", "e-mail"),
    (r"\b[íi]mejly\b", "e-maily"),
    (r"\bapt\s+apdejt\b", "apt update"),
    (r"\bapt\s+apgrejd\b", "apt upgrade"),
    (r"\bapt-get\s+apdejt\b", "apt-get update"),
    (r"\bapt-get\s+apgrejd\b", "apt-get upgrade"),
    (r"\bunattended\s+upgrades?\b", "unattended-upgrades"),
    (r"\bapt\s*\.\s*conf\s*\.\s*d\s*\.?", "apt.conf.d"),
    (r"\bslash\s+etc(?:y)?\s+slash\s+apt\s+slash\s+apt\.?conf\.?d\b", "/etc/apt/apt.conf.d"),
    (r"\bslash\s+etc(?:y)?\s+slash\s+apt\b", "/etc/apt"),
    (r"\blomka\s+etc(?:y)?\s+lomka\s+apt\s+lomka\s+apt\.?conf\.?d\b", "/etc/apt/apt.conf.d"),
    (r"\blomka\s+etc(?:y)?\s+lomka\s+apt\b", "/etc/apt"),
)

_LINUX_META_REPLACEMENTS = (
    (r"Prepáč,\s*neviem\.", "Bohužiaľ nepoznám odpoveď na túto otázku."),
    (r"Pozrite(?:\s+si)?\s+na\s+oficiálnu\s+stránku\.", "Pozrite si oficiálnu komunitu."),
)

_LINUX_SQL_DRIFT_PATTERNS = (
    r"\bSQL\b",
    r"\bNULL\b",
    r"\bIS NULL\b",
    r"\bIS NOT NULL\b",
    r"\bISNULL\b",
    r"\bCOALESCE\b",
    r"\bNULLIF\b",
    r"\bBOOLEAN\b",
    r"\bTRUE\b",
    r"\bFALSE\b",
    r"\bkontrolu nul\b",
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


_DOMAIN_TLD_RE = re.compile(r"(\w)\.(org|com|net|io|gov|edu|ai|dev|app|sk|cz|eu)\b", re.IGNORECASE)


def clean_punctuation(text: str) -> str:
    out = re.sub(r"\s+([,.:;!?])", r"\1", text or "")
    # Protect patterns that should NOT get a space inserted:
    #   - digit.digit (e.g. "2.0", "verzia 3.0")
    #   - word.TLD (e.g. "brilliant.org")
    out = re.sub(r"(\d)\.(\d)", lambda m: f"{m.group(1)}\x00{m.group(2)}", out)
    out = _DOMAIN_TLD_RE.sub(lambda m: f"{m.group(1)}\x00{m.group(2)}", out)
    # Add space po interpunkcii ALE iba ak nasleduje normálny znak — NIE iná interpunkcia.
    # Inak by sa "..." stalo ". ..", "?!" stalo "? !" atď.
    out = re.sub(r"([,.:;!?])([^\s,.:;!?])", r"\1 \2", out)
    out = out.replace("\x00", ".")
    return normalize_spaces(out)


def soften_proxmox_brand_declension(text: str) -> str:
    out = normalize_spaces(text)
    if not out:
        return out

    replacements = (
        (r"\bo Proxmoxe\b", "o platforme Proxmox"),
        (r"\bv Proxmoxe\b", "v systéme Proxmox"),
        (r"\bvo Proxmoxe\b", "v systéme Proxmox"),
        (r"\bna Proxmoxe\b", "na platforme Proxmox"),
        (r"\bs Proxmoxom\b", "s platformou Proxmox"),
        (r"\bz Proxmoxu\b", "z platformy Proxmox"),
        (r"\bna Proxmox stránkach\b", "na stránkach Proxmox"),
        (r"\bwiki Proxmoxu\b", "wiki Proxmox"),
        (r"\bdokumentáci[aeu]\s+Proxmoxu\b", "dokumentáciu Proxmox"),
        (r"\bpodpora Proxmoxu\b", "podpora Proxmox"),
    )

    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    return clean_punctuation(out)


def linux_source_guided_template(source_text: str) -> str | None:
    src = normalize_spaces(source_text)
    low = src.lower()
    if not src:
        return None

    if "official community" in low and "question" in low and "proxmox" in low:
        return "Bohužiaľ nepoznám odpoveď na vašu otázku. Skúste oficiálnu komunitu Proxmox."
    if "find your answers there" in low and "proxmox" in low:
        return "Pre Proxmox tam možno nájdete odpovede na svoje otázky."
    if "also has its own" in low and "proxmox" in low:
        return "Proxmox má aj vlastné riešenia."
    if "proxmox" in low and "server" in low and "running" in low:
        return "Váš Proxmox server je spustený."
    return None


def looks_like_linux_sql_drift(text: str, *, source_text: str = "") -> bool:
    low_src = normalize_spaces(source_text).lower()
    low_txt = normalize_spaces(text).lower()
    if not low_txt:
        return False

    linux_context = any(term in low_src for term in ("proxmox", "debian", "ubuntu", "ssh", "smtp", "apt "))
    sql_in_source = any(term in low_src for term in (" sql", "is null", "isnull", "coalesce", "nullif", "boolean"))
    if not linux_context or sql_in_source:
        return False

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _LINUX_SQL_DRIFT_PATTERNS)


def looks_like_linux_meta_phrase(text: str) -> bool:
    low = normalize_spaces(text).lower()
    return any(
        phrase in low
        for phrase in (
            "bohužiaľ nepoznám odpoveď",
            "prepáč, neviem",
            "pozrite si oficiálnu komunitu",
            "pozrite na oficiálnu stránku",
        )
    )


def protect_linux_literals(text: str, prefix: str = "__LX") -> tuple[str, dict[str, str]]:
    """
    Protect Linux/CLI literals before machine translation.

    Returns protected_text, token_map.
    """
    if not text:
        return text, {}

    protected = text
    token_map: dict[str, str] = {}
    token_idx = 0

    def _reserve(match_text: str) -> str:
        nonlocal token_idx
        token = f"{prefix}{token_idx}__"
        token_map[token] = match_text
        token_idx += 1
        return token

    for pattern in _LINUX_DYNAMIC_PATTERNS:
        matches = list(re.finditer(pattern, protected, flags=re.IGNORECASE))
        for match in reversed(matches):
            token = _reserve(match.group(0))
            protected = protected[: match.start()] + token + protected[match.end() :]

    for term in sorted(LINUX_PROTECTED_TERMS, key=len, reverse=True):
        pattern = re.escape(term)
        flags = 0 if any(ch.isupper() for ch in term if ch.isalpha()) else re.IGNORECASE
        protected = re.sub(pattern, lambda m: _reserve(m.group(0)), protected, flags=flags)

    return protected, token_map


def restore_linux_literals(text: str, token_map: dict[str, str]) -> str:
    if not text or not token_map:
        return text

    restored = text
    for token, original in token_map.items():
        restored = restored.replace(token, original)
        restored = restored.replace(token.lower(), original)
    return restored


def guard_linux_text(text: str, *, source_text: str = "", fallback_text: str = "") -> str:
    out = normalize_spaces(text)
    if not out:
        return normalize_spaces(fallback_text)

    for pattern, replacement in _LINUX_META_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    for pattern, replacement in _LINUX_FIX_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    src_low = normalize_spaces(source_text).lower()

    if "proxmox" in src_low:
        out = re.sub(r"\bv rámci Proxmox\b", "v rámci platformy Proxmox", out, flags=re.IGNORECASE)
        out = re.sub(r"\bv Proxmox\b", "v systéme Proxmox", out, flags=re.IGNORECASE)
        out = re.sub(r"\bz Proxmox\b", "z platformy Proxmox", out, flags=re.IGNORECASE)
        out = re.sub(r"\bs Proxmox\b", "s platformou Proxmox", out, flags=re.IGNORECASE)
        out = soften_proxmox_brand_declension(out)

    if "unattended-upgrades" in src_low:
        out = re.sub(
            r"\bbezobslužn\w*\s+(?:aktualiz[áa]cie|inov[áa]cie)(?:\s+spojovn[ií]ka)?\b",
            "unattended-upgrades",
            out,
            flags=re.IGNORECASE,
        )

    if "apt update" in src_low:
        out = re.sub(r"\bapt\s+update\b", "apt update", out, flags=re.IGNORECASE)
        out = re.sub(r"\bapt\s+apdejt\b", "apt update", out, flags=re.IGNORECASE)
        out = re.sub(r"\bapt\s+update\s+and\b", "apt update a", out, flags=re.IGNORECASE)
    if "apt upgrade" in src_low:
        out = re.sub(r"\bapt\s+upgrade\b", "apt upgrade", out, flags=re.IGNORECASE)
        out = re.sub(r"\bapt\s+apgrejd\b", "apt upgrade", out, flags=re.IGNORECASE)
    if "apt-get update" in src_low:
        out = re.sub(r"\bapt-get\s+apdejt\b", "apt-get update", out, flags=re.IGNORECASE)
    if "apt-get upgrade" in src_low:
        out = re.sub(r"\bapt-get\s+apgrejd\b", "apt-get upgrade", out, flags=re.IGNORECASE)
    if "/etc/apt/apt.conf.d" in src_low:
        out = re.sub(r"\bapt\s*\.\s*conf\s*\.\s*d\s*\.?", "apt.conf.d", out, flags=re.IGNORECASE)
        out = re.sub(r"/etc/apt\b", "/etc/apt/apt.conf.d", out, count=1)
    if "apt.conf.d" in src_low:
        out = re.sub(r"\bapt\s*\.\s*conf\s*\.\s*d\s*\.?", "apt.conf.d", out, flags=re.IGNORECASE)
    if "/etc/apt" in src_low:
        out = re.sub(r"\bslash\s+etc(?:y)?\s+slash\s+apt\b", "/etc/apt", out, flags=re.IGNORECASE)

    template = linux_source_guided_template(source_text)
    if template and (
        looks_like_linux_sql_drift(out, source_text=source_text)
        or looks_like_linux_meta_phrase(out)
    ):
        out = template

    out = clean_punctuation(out)

    # Re-apply literal Linux/CLI fixes after punctuation cleanup.
    for pattern, replacement in _LINUX_FIX_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    if "apt.conf.d" in src_low:
        out = re.sub(r"\bapt\s*\.\s*conf\s*\.\s*d\s*\.?", "apt.conf.d", out, flags=re.IGNORECASE)
    if "/etc/apt/apt.conf.d" in src_low:
        out = re.sub(r"/etc/apt\b", "/etc/apt/apt.conf.d", out, count=1)

    return normalize_spaces(out)


def apply_linux_mixed_mode(text: str, *, source_text: str = "", fallback_text: str = "") -> str:
    """
    Mixed SK+EN technical mode.

    Keeps Slovak sentence structure, but preserves Linux / CLI literals in English
    when they are important technical anchors in the source.
    """
    out = guard_linux_text(text, source_text=source_text, fallback_text=fallback_text)
    src_low = normalize_spaces(source_text).lower()

    if "proxmox" in src_low:
        out = soften_proxmox_brand_declension(out)
        out = re.sub(r"\bvo vašej inštancii Proxmox\b", "vo vašej inštancii Proxmox", out, flags=re.IGNORECASE)

    if "unattended upgrades" in src_low or "unattended-upgrades" in src_low:
        for pattern, replacement in _LINUX_MIXED_TERM_REPLACEMENTS:
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    if "ssh" in src_low:
        out = re.sub(r"\bSSH\b", "SSH", out)
    if "smtp" in src_low:
        out = re.sub(r"\bSMTP\b", "SMTP", out)
    if "email" in src_low or "e-mail" in src_low:
        out = re.sub(r"\be-?mail\b", "e-mail", out, flags=re.IGNORECASE)

    if "apt update" in src_low:
        out = re.sub(r"\bapt\s+update\b", "apt update", out, flags=re.IGNORECASE)
    if "apt upgrade" in src_low:
        out = re.sub(r"\bapt\s+upgrade\b", "apt upgrade", out, flags=re.IGNORECASE)
    if "apt install" in src_low:
        out = re.sub(r"\bapt\s+install\b", "apt install", out, flags=re.IGNORECASE)
    if "apt full-upgrade" in src_low:
        out = re.sub(r"\bapt\s+full-upgrade\b", "apt full-upgrade", out, flags=re.IGNORECASE)
    if "/etc/apt" in src_low:
        out = re.sub(r"/etc/apt\b", "/etc/apt", out)
    if "apt.conf.d" in src_low:
        out = re.sub(r"\bapt\s*\.\s*conf\s*\.\s*d\b", "apt.conf.d", out, flags=re.IGNORECASE)

    return normalize_spaces(out)


def normalize_linux_for_tts(text: str) -> str:
    """
    TTS-only pronunciation helper for mixed technical Slovak text.

    Protects exact commands/paths first, then lightly phoneticizes standalone
    English nouns that Slovak models often read poorly.
    """
    if not text:
        return text

    protected, token_map = protect_linux_literals(text, prefix="__LXTTS")
    out = protected

    replacements = (
        (r"\bupdate\b", "apdejt"),
        (r"\bUpdate\b", "Apdejt"),
        (r"\bupdates\b", "apdejty"),
        (r"\bupgrade\b", "apgrejd"),
        (r"\bUpgrade\b", "Apgrejd"),
        (r"\bupgrades\b", "apgrejdy"),
        (r"\be-?mail\b", "ímejl"),
        (r"\bE-?mail\b", "Ímejl"),
    )

    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out)

    for token, original in token_map.items():
        out = out.replace(token, original)
        out = out.replace(token.lower(), original)

    return normalize_spaces(out)
