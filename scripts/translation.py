"""
Translation module: LLM/MADLAD/API translation, phonetic respelling,
text utilities, profile presets, SRT/JSON export.
"""

import gc
import json
import re
from pathlib import Path

import numpy as np
import requests
import torch
from llama_cpp import Llama
# T5 imports are lazy — transformers 5.x removed them from top-level namespace

from stt import split_text
from config import SCRIPT_DIR, target_lang_name
from paths import PATHS
from linux_logic_guard import (
    apply_linux_mixed_mode,
    guard_linux_text,
    linux_source_guided_template,
    looks_like_linux_sql_drift,
    protect_linux_literals,
)
from sql_logic_guard import guard_sql_text, source_guided_template

# G3 12B v1 wrappers — opt-in cez --use_g3_v1_translator / --use_g3_v1_length_adjuster
try:
    from llm_v1 import translate_g3_v1, adjust_length_g3_v1, adjust_length_for_duration
    _G3_V1_AVAILABLE = True
except ImportError:
    _G3_V1_AVAILABLE = False


def translate_fulldoc_with_g3_v1(segments, tgt_lang="sk", src_lang="en",
                                  content_type: str = "general",
                                  batch_size: int = 0):
    """Preloží zoznam segmentov cez Gemma 3 12B v1 translator (ollama).

    content_type: "technical"/"programming"/"sql" → model zachová EN tech termy.
    batch_size:
        0 = per-segment mode (legacy, problémy s halucináciou na krátkych vetách)
        5-10 = batch mode — pošle viacero viet v 1 prompt-e s [n] markermi.
               Model dostane kontext susedných viet, halucinácia výrazne klesne.

    Returns:
        list[dict]: každý segment má pridanú kľúčovú dvojicu 'text' (SK preklad).
    """
    if not _G3_V1_AVAILABLE:
        raise ImportError("llm_v1 module nedostupný — chýba scripts/llm_v1.py")
    if batch_size and batch_size > 1:
        return _translate_g3_v1_batch(segments, batch_size=batch_size,
                                       content_type=content_type)
    # Legacy per-segment mode
    out = []
    for seg in segments:
        src = (seg.get("text") or "").strip()
        if not src:
            out.append({**seg, "text": ""})
            continue
        sk = translate_g3_v1(src, fallback=src, content_type=content_type)
        out.append({**seg, "text": sk})
    return out


def _translate_g3_v1_batch(segments, batch_size: int = 6,
                           content_type: str = "general"):
    """Batch translate via G3-12B — viacero viet v 1 prompt-e cez [n] markery.

    Strategy:
        Vstup: "[1] sentence one [2] sentence two [3] ..."
        Výstup parser hľadá [1] [2] [3] ... a mapuje na pôvodné segmenty.
        Ak parser zlyhá (model nevrátil markers) → per-segment fallback pre celý batch.
    """
    import re as _re
    from llm_v1 import _ollama_generate as _ol_gen, _strip_format_artifacts

    # Build prompt for batch
    def _build_prompt(en_segs: list[str]) -> str:
        glossary = (
            "\nGLOSSARY (zachovaj v EN bez slovenských deklinácií): "
            "C++, shader, engine, framework, OpenGL, Vulkan, DirectX, voxel, mesh, "
            "GPU, CPU, Pacman, Minecraft, compute shader, pipeline, chunk, rigid body, "
            "collider, physics, rendering, vertex, fragment.\n"
        ) if content_type in ("technical", "programming", "sql") else ""
        numbered = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(en_segs))
        return (
            f"Si profesionálny EN→SK prekladateľ. Prelož každú očíslovanú vetu "
            f"do prirodzenej slovenčiny. Zachovaj presne rovnaké [n] čísla.{glossary}"
            f"Žiadny komentár, žiadne markdown bloky. Iba [n] SK preklad pre každú vetu.\n\n"
            f"EN:\n{numbered}\n\nSK:"
        )

    # Parse output — find [n] markers
    def _parse_batch(raw: str, expected_n: int) -> list[str] | None:
        text = _strip_format_artifacts(raw or "")
        # Match [N] text up to next [N+1] or end
        pattern = _re.compile(r"\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)", _re.DOTALL)
        matches = pattern.findall(text)
        if not matches:
            return None
        result = [""] * expected_n
        for num_str, content in matches:
            try:
                idx = int(num_str) - 1
                if 0 <= idx < expected_n:
                    result[idx] = content.strip()
            except ValueError:
                continue
        # Sanity: at least 80% slots filled
        filled = sum(1 for s in result if s)
        if filled < int(expected_n * 0.8):
            return None
        return result

    out = []
    n = len(segments)
    print(f"[G3-V1-BATCH] {n} segmentov v batch_size={batch_size}", flush=True)

    i = 0
    while i < n:
        batch = segments[i:i + batch_size]
        en_texts = [(seg.get("text") or "").strip() for seg in batch]
        # Skip empty batch
        non_empty = [(j, t) for j, t in enumerate(en_texts) if t]
        if not non_empty:
            for seg in batch:
                out.append({**seg, "text": ""})
            i += batch_size
            continue

        # Build prompt with only non-empty texts
        non_empty_texts = [t for _, t in non_empty]
        prompt = _build_prompt(non_empty_texts)
        # Token budget: ~2× input chars
        n_predict = max(512, sum(len(t) for t in non_empty_texts) * 3)

        raw = _ol_gen("g3-12b-translator-v1:latest", prompt,
                      num_predict=n_predict, temperature=0.2)
        parsed = _parse_batch(raw, len(non_empty))

        if parsed is None:
            # Batch parse zlyhal — fallback per-segment pre celý batch
            print(f"[G3-V1-BATCH] Batch {i//batch_size + 1} parse zlyhal — per-segment fallback", flush=True)
            for j, seg in enumerate(batch):
                src = en_texts[j]
                if not src:
                    out.append({**seg, "text": ""})
                else:
                    sk = translate_g3_v1(src, fallback=src, content_type=content_type)
                    out.append({**seg, "text": sk})
        else:
            # Map parsed back to original segments (some were empty)
            parse_idx = 0
            for j, seg in enumerate(batch):
                if not en_texts[j]:
                    out.append({**seg, "text": ""})
                else:
                    sk = parsed[parse_idx] if parse_idx < len(parsed) else en_texts[j]
                    out.append({**seg, "text": sk or en_texts[j]})  # fallback to EN if empty
                    parse_idx += 1
        i += batch_size

    print(f"[G3-V1-BATCH] Hotovo, {len(out)} segmentov preložených.", flush=True)
    return out


def compress_with_g3_v1(text: str, target_len: int) -> tuple[str, str]:
    """Compress slovenský text cez Gemma 3 12B length adjuster v1.

    Returns:
        (compressed_text, method) — method = 'g3_v1' alebo 'noop' (fallback)
    """
    if not _G3_V1_AVAILABLE:
        return text, "noop"
    out = adjust_length_g3_v1(text, target_chars=target_len)
    method = "g3_v1" if out != text else "noop"
    return out, method


# ---------------------------------------------------------------------------
# Phonetic respelling for TTS
# ---------------------------------------------------------------------------

PHONETIC_MAP = {
    # SQL keywords — identity mapping: protect from translation AND phonetic respelling
    # protect_phonetic_terms() wraps these in placeholders before LLM translation
    "SELECT": "SELECT",
    "FROM": "FROM",
    "WHERE": "WHERE",
    "JOIN": "JOIN",
    "LEFT JOIN": "LEFT JOIN",
    "RIGHT JOIN": "RIGHT JOIN",
    "INNER JOIN": "INNER JOIN",
    "FULL JOIN": "FULL JOIN",
    "CROSS JOIN": "CROSS JOIN",
    "GROUP BY": "GROUP BY",
    "ORDER BY": "ORDER BY",
    "HAVING": "HAVING",
    "INSERT": "INSERT",
    "UPDATE": "UPDATE",
    "DELETE": "DELETE",
    "CREATE": "CREATE",
    "ALTER": "ALTER",
    "DROP": "DROP",
    "TRUNCATE": "TRUNCATE",
    "INDEX": "INDEX",
    "CONSTRAINT": "CONSTRAINT",
    "PRIMARY KEY": "PRIMARY KEY",
    "FOREIGN KEY": "FOREIGN KEY",
    "REFERENCES": "REFERENCES",
    "NOT NULL": "NOT NULL",
    "IS NOT NULL": "IS NOT NULL",
    "IS NULL": "IS NULL",
    "NULL": "NULL",
    "UNIQUE": "UNIQUE",
    "DEFAULT": "DEFAULT",
    "TRIGGER": "TRIGGER",
    "VIEW": "VIEW",
    "PROCEDURE": "PROCEDURE",
    "FUNCTION": "FUNCTION",
    "DECLARE": "DECLARE",
    "BEGIN": "BEGIN",
    "END": "END",
    "UNION": "UNION",
    "UNION ALL": "UNION ALL",
    "INTERSECT": "INTERSECT",
    "EXCEPT": "EXCEPT",
    "DISTINCT": "DISTINCT",
    "TOP": "TOP",
    "LIMIT": "LIMIT",
    "OFFSET": "OFFSET",
    "WITH": "WITH",
    "CASE": "CASE",
    "WHEN": "WHEN",
    "THEN": "THEN",
    "ELSE": "ELSE",
    "LIKE": "LIKE",
    "BETWEEN": "BETWEEN",
    "EXISTS": "EXISTS",
    "PARTITION BY": "PARTITION BY",
    "OVER": "OVER",
    "PIVOT": "PIVOT",
    "UNPIVOT": "UNPIVOT",
    "ROLLBACK": "ROLLBACK",
    "COMMIT": "COMMIT",
    "TRANSACTION": "TRANSACTION",
    "CURSOR": "CURSOR",
    # SQL built-in functions — identity mapping
    "SQL": "SQL",
    "NULLIF": "NULLIF",
    "ISNULL": "ISNULL",
    "COALESCE": "COALESCE",
    "NVL": "NVL",
    "CAST": "CAST",
    "CONVERT": "CONVERT",
    "TRY_CAST": "TRY_CAST",
    "TRY_CONVERT": "TRY_CONVERT",
    "COUNT": "COUNT",
    "SUM": "SUM",
    "AVG": "AVG",
    "MIN": "MIN",
    "MAX": "MAX",
    "UPPER": "UPPER",
    "LOWER": "LOWER",
    "LEN": "LEN",
    "LENGTH": "LENGTH",
    "SUBSTRING": "SUBSTRING",
    "SUBSTR": "SUBSTR",
    "CHARINDEX": "CHARINDEX",
    "PATINDEX": "PATINDEX",
    "REPLACE": "REPLACE",
    "REPLICATE": "REPLICATE",
    "REVERSE": "REVERSE",
    "LTRIM": "LTRIM",
    "RTRIM": "RTRIM",
    "TRIM": "TRIM",
    "CONCAT": "CONCAT",
    "STRING_AGG": "STRING_AGG",
    "DATEPART": "DATEPART",
    "DATEDIFF": "DATEDIFF",
    "DATEADD": "DATEADD",
    "GETDATE": "GETDATE",
    "GETUTCDATE": "GETUTCDATE",
    "SYSDATETIME": "SYSDATETIME",
    "FORMAT": "FORMAT",
    "ROW_NUMBER": "ROW_NUMBER",
    "RANK": "RANK",
    "DENSE_RANK": "DENSE_RANK",
    "NTILE": "NTILE",
    "LEAD": "LEAD",
    "LAG": "LAG",
    "FIRST_VALUE": "FIRST_VALUE",
    "LAST_VALUE": "LAST_VALUE",
    "IIF": "IIF",
    "CHOOSE": "CHOOSE",
    "ABS": "ABS",
    "ROUND": "ROUND",
    "CEILING": "CEILING",
    "FLOOR": "FLOOR",
    "POWER": "POWER",
    "SQRT": "SQRT",
    # Tech terms
    "Java": "Džava",
    "java": "džava",
    "JavaScript": "Džava-skript",
    "javascript": "džava-skript",
    "Python": "Pajton",
    "python": "pajton",
    "Docker": "Doker",
    "docker": "doker",
    "Kubernetes": "Kubernétýs",
    "kubernetes": "kubernétýs",
    "cloud": "klaud",
    "Cloud": "Klaud",
    "framework": "frejm-work",
    "Framework": "Frejm-work",
    "software": "softvér",
    "Software": "Softvér",
    "hardware": "hárdvér",
    "Hardware": "Hárdvér",
    "update": "update",
    "Update": "Update",
    "upgrade": "upgrade",
    "download": "daunloud",
    "upload": "aploud",
    "online": "onlajn",
    "Online": "Onlajn",
    "offline": "oflajn",
    "Offline": "Oflajn",
    "streaming": "strýming",
    "Streaming": "Strýming",
    "machine learning": "mašín lerning",
    "deep learning": "díp lerning",
    "tutorial": "tutoriál",
    "Tutorial": "Tutoriál",
    "feature": "fíčura",
    "Feature": "Fíčura",
    "features": "fíčury",
    "bug": "bag",
    "debug": "dýbag",
    "deploy": "diploj",
    "deployment": "diplojment",
    "pipeline": "pajplajn",
    "Pipeline": "Pajplajn",
    "server": "server",
    "cache": "keš",
    "Cache": "Keš",
    "router": "rauter",
    "Router": "Rauter",
    "switch": "svič",
    "Switch": "Svič",
    "interface": "interfejs",
    "Wi-Fi": "Váj-fáj",
    "wifi": "váj-fáj",
    "Bluetooth": "Blútůs",
    "bluetooth": "blútůs",
    "YouTube": "JůTjůb",
    "youtube": "jůtjůb",
    "Google": "Gůgl",
    "google": "gůgl",
    "Apple": "Epl",
    "apple": "epl",
    "Windows": "Vindous",
    "windows": "vindous",
    "Linux": "Linux",
    "container": "kontejner",
    "Container": "Kontejner",
    "virtual": "virtuál",
    "Virtual": "Virtuál",
    "Proxmox": "Proxmox",
    "proxmox": "proxmox",
    # Common English words
    "cool": "kúl",
    "Cool": "Kúl",
    "nice": "najs",
    "like": "lajk",
    "okay": "okej",
    "OK": "okej",
    "awesome": "ósam",
    "amazing": "amejzing",
    "performance": "performens",
    "Performance": "Performens",
    "review": "rivjů",
    "Review": "Rivjů",
    "game": "gejm",
    "Game": "Gejm",
    "gaming": "gejming",
    "Gaming": "Gejming",
    "design": "dizajn",
    "Design": "Dizajn",
    "display": "displej",
    "Display": "Displej",
    "smartphone": "smartfón",
    "Smartphone": "Smartfón",
    "laptop": "laptop",
    "desktop": "desktop",
    "email": "email",
    "Email": "Email",
    "e-mail": "e-mail",
    "feedback": "fídbek",
    "Feedback": "Fídbek",
    "startup": "startap",
    "Startup": "Startap",
    "website": "vebsajt",
    "Website": "Vebsajt",
    "layout": "lejaut",
    "Layout": "Lejaut",
    "plugin": "plagin",
    "Plugin": "Plagin",
    "thumbnail": "tambneil",
    "Thumbnail": "Tambneil",
    "screenshot": "skrínšot",
    "Screenshot": "Skrínšot",
    # Web & standards acronyms (letter-by-letter spelling in Czech)
    "HTML": "há-té-em-el",
    "CSS": "cé-es-es",
    "XML": "iks-em-el",
    "CSV": "cé-es-vé",
    "JSON": "džejsón",
    "YAML": "jáml",
    "API": "á-pé-í",
    "URL": "ú-er-el",
    "URI": "ú-er-í",
    "HTTP": "há-té-té-pé",
    "HTTPS": "há-té-té-pé-es",
    "DNS": "dé-en-es",
    "TCP": "té-cé-pé",
    "UDP": "ú-dé-pé",
    "VPN": "vé-pé-en",
    "LAN": "lan",
    "WAN": "van",
    "SSH": "SSH",
    "SSL": "es-es-el",
    "FTP": "ef-té-pé",
    "SMTP": "SMTP",
    # Hardware acronyms
    "CPU": "cé-pé-ú",
    "GPU": "gé-pé-ú",
    "RAM": "ram",
    "ROM": "rom",
    "SSD": "es-es-dé",
    "HDD": "há-dé-dé",
    "NVMe": "en-vé-mé",
    "USB": "ú-es-bé",
    "HDMI": "há-dé-em-í",
    "PCIe": "pé-cé-í-í",
    # Dev & ops
    "CI": "sí-áj",
    "CD": "sí-dí",
    "CLI": "cé-el-í",
    "IDE": "í-dé-é",
    "ORM": "ó-er-em",
    "ETL": "é-té-el",
    "CRUD": "krůd",
    "REST": "rest",
    "SOAP": "sóp",
    "PDF": "PDF",
    "AI": "á-í",
    "ML": "em-el",
    "UI": "jů-áj",
    "UX": "jů-iks",
    "DevOps": "dev-ops",
    "SaaS": "sás",
    "PaaS": "pás",
    "IaaS": "íás",
    # Anatomy — protect from translation (same word in Czech; Whisper mishears as "pennies")
    "penis": "penis",
    "vagina": "vagína",
    "clitoris": "klitoris",
    "transgender": "transgender",
    "cisgender": "cisgender",
    # Gaming / graphics — identity mapping: protect from translation.
    # LLM sometimes translates "engine" → "motor", "shader" → "tieňovač".
    # TTS phonetic respelling (engine→enžin, shader→šejder) is handled by user phonetic_map.json
    # which overrides these identity values in apply_phonetic_respelling().
    "engine": "engine",
    "engines": "engines",
    "shader": "shader",
    "shaders": "shaders",
    "rendering": "rendering",
    "renderer": "renderer",
    "voxel": "voxel",
    "voxels": "voxels",
    "framerate": "framerate",
    "pathfinding": "pathfinding",
    "raycasting": "raycasting",
    "raytracing": "raytracing",
    # "era" LLM prekladá ako "chyba" (error) — ochrana + TTS respelling v phonetic_map.json
    "era": "era",
    "eras": "eras",
}


def _load_user_phonetic_map() -> dict:
    """Load user-defined phonetic overrides from phonetic_map.json if it exists."""
    user_file = SCRIPT_DIR / "data" / "phonetic_map.json"
    if user_file.exists():
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                user_map = json.load(f)
            print(f"[PHONETIC] Loaded {len(user_map)} entries from phonetic_map.json", flush=True)
            return user_map
        except Exception as e:
            print(f"[PHONETIC] Failed to load phonetic_map.json: {e}", flush=True)
    return {}


def apply_phonetic_respelling(text: str, skip_sql_phonetic: bool = False) -> str:
    """Replace English words with phonetic spelling for better TTS pronunciation.
    User overrides from phonetic_map.json take priority over built-in map.
    Also handles Czech/Slovak declension suffixes (e.g. Javu, Javě, Javy).

    skip_sql_phonetic=True: preskočí SQL TTS konverzie (IS NULL→ajznul atď.)
    Použiť pre Fish Speech S2 Pro ktorý číta SQL natívne bez konverzií.
    """
    import re
    # Czech/Slovak declension suffixes added to consonant-ending words (Docker->Dockeru)
    _CONS_SUFFIXES = ["u", "y", "em", "e", "i", "a", "ů", "ech", "ům", "ami"]
    # Czech/Slovak declension endings replacing final 'a' (Java->Javy, Javu, Javě)
    _A_REPLACE_ENDINGS = ["y", "ě", "u", "ou", "o", "ám", "ách", "ami"]
    # Adjective-forming suffixes (Docker->Dockerový, Java->Javový)
    _ADJ_SUFFIXES = ["ový", "ová", "ové", "ového", "ovém", "ovým", "ových",
                     "ovému", "ový", "ovskou", "ovský", "ovská", "ovské"]

    # Merge: user overrides take priority
    merged = dict(PHONETIC_MAP)
    merged.update(_load_user_phonetic_map())
    merged.pop("_info", None)

    # Sort by length descending so longer phrases match first
    sorted_entries = sorted(merged.items(), key=lambda x: len(x[0]), reverse=True)

    def _match_case(src: str, template: str) -> str:
        if not src:
            return template
        if src.isupper():
            return template.upper()
        if src[0].isupper():
            return template[:1].upper() + template[1:]
        return template

    for eng, phonetic in sorted_entries:
        # 1) Exact term match with safe boundaries
        pattern_exact = r'(?<!\w)' + re.escape(eng) + r'(?!\w)'
        # All-uppercase keys (e.g. "IDE", "API") match case-sensitively
        # to avoid replacing common Slovak/Czech lowercase words like "ide", "log"
        _flags = 0 if eng.isupper() else re.IGNORECASE
        text = re.sub(
            pattern_exact,
            lambda m, p=phonetic: _match_case(m.group(0), p),
            text,
            flags=_flags,
        )

        # 2) Handle declined forms
        if len(eng) < 3:
            continue

        if eng[-1].lower() == 'a' and len(eng) > 3:
            eng_stem = eng[:-1]
            pho_stem = phonetic[:-1] if phonetic[-1].lower() == 'a' else phonetic
            for ending in _A_REPLACE_ENDINGS:
                declined = eng_stem + ending
                pho_declined = pho_stem + ending
                text = re.sub(r'\b' + re.escape(declined) + r'\b', pho_declined, text)
                if declined[0].isupper():
                    pass
                else:
                    cap_declined = declined[0].upper() + declined[1:]
                    cap_pho = pho_declined[0].upper() + pho_declined[1:]
                    text = re.sub(r'\b' + re.escape(cap_declined) + r'\b', cap_pho, text)
        else:
            for suffix in _CONS_SUFFIXES:
                declined = eng + suffix
                pho_declined = phonetic + suffix
                text = re.sub(r'\b' + re.escape(declined) + r'\b', pho_declined, text)

        # 3) Adjective forms
        if eng[-1].lower() == 'a' and len(eng) > 3:
            adj_stem = eng[:-1]
            pho_adj_stem = phonetic[:-1] if phonetic[-1].lower() == 'a' else phonetic
        else:
            adj_stem = eng
            pho_adj_stem = phonetic
        for adj_suf in _ADJ_SUFFIXES:
            adj_form = adj_stem + adj_suf
            pho_adj = pho_adj_stem + adj_suf
            text = re.sub(r'\b' + re.escape(adj_form) + r'\b', pho_adj, text)

    # SQL function TTS phonetics — regex patterns, runs after PHONETIC_MAP.
    # Catches both correctly-preserved SQL names AND common LLM mistranslations.
    _SQL_TTS_REGEX = [
        # COALESCE — catch English form AND all Czech LLM mistranslations
        (r'(?<!\w)COALESCE(?!\w)',          'koalesk'),  # correct form kept by LLM
        (r'(?<!\w)koalov\w*',               'koalesk'),  # koalové, koalového, koalovou …
        (r'(?<!\w)koal[yi]\w*',             'koalesk'),  # koaly, koali
        (r'(?<!\w)koala(?!\w)',             'koalesk'),  # singular koala
        (r'(?<!\w)koalesc\w*',              'koalesk'),  # koalesc — LLM phonetic hint from translation prompt
        # IS NOT NULL / IS NULL / NOT NULL — two-word SQL operators (must come before ISNULL)
        (r'(?<!\w)IS\s+NOT\s+NULL(?!\w)',   'íz not nul'),
        (r'(?<!\w)IS\s+NULL(?!\w)',         'ajznul'),
        (r'(?<!\w)NOT\s+NULL(?!\w)',        'not nul'),
        # ISNULL (one word) — catch English form AND LLM mistranslations
        (r'(?<!\w)ISNULL(?!\w)',            'ajznul'),
        (r'(?i)vol[áa]\s+null',             'ajznul'),   # "volá NULL" = LLM mistranslation
        (r'(?<!\w)je\s+null(?!\w)',         'ajznul'),   # "je null"
        # NULLIF
        (r'(?<!\w)NULLIF(?!\w)',            'nulíf'),
        # Other SQL functions
        (r'(?<!\w)CAST(?!\w)',              'kast'),
        (r'(?<!\w)CHARINDEX(?!\w)',         'char-index'),
        (r'(?<!\w)SUBSTRING(?!\w)',         'sub-string'),
        (r'(?<!\w)DATEPART(?!\w)',          'dejt-párt'),
        (r'(?<!\w)DATEDIFF(?!\w)',          'dejt-dif'),
        (r'(?<!\w)DATEADD(?!\w)',           'dejt-ad'),
        (r'(?<!\w)GETDATE(?!\w)',           'get-dejt'),
        (r'(?<!\w)ROW_NUMBER(?!\w)',        'ro-nambr'),
        (r'(?<!\w)DENSE_RANK(?!\w)',        'dens-rank'),
        (r'(?<!\w)STRING_AGG(?!\w)',        'string-ag'),
        # Comparison operators → spoken form (safe for all content types)
        (r'(?<![<>!])<=(?!=)',              'menší nebo rovno'),
        (r'(?<![<>!])>=(?!=)',              'větší nebo rovno'),
        (r'<>',                             'nerovná se'),
        (r'(?<![!<>])!=',                   'nerovná se'),
        # SELECT * → select vše (must come before bare * replacement)
        (r'(?i)\bSELECT\s+\*',             'SELECT vše'),
    ]
    if not skip_sql_phonetic:
        for _pat, _pho in _SQL_TTS_REGEX:
            text = re.sub(_pat, _pho, text, flags=re.IGNORECASE)

    return guard_linux_text(text)


def guard_technical_text(text: str, *, source_text: str = "", fallback_text: str = "") -> str:
    out = guard_sql_text(text, source_text=source_text, fallback_text=fallback_text)
    out = guard_linux_text(out, source_text=source_text, fallback_text=fallback_text)
    return out


def _looks_like_sql_query(text: str) -> bool:
    """True if text contains an actual SQL query (not just explanation with SQL terms)."""
    return bool(re.search(
        r'\b(?:SELECT|UPDATE|DELETE|INSERT\s+INTO|WITH\s+\w+\s+AS)\b',
        text, re.IGNORECASE,
    ))


def is_sql_heavy(text: str) -> bool:
    """
    Auto-detect if a text segment contains actual SQL code (not just vocabulary).

    Returns True only when the segment contains executable SQL syntax:
      - SQL DML/DDL statements (SELECT ... FROM, INSERT, UPDATE, DELETE, WITH ... AS)
      - SQL function calls with parentheses: ISNULL(...), COALESCE(...)
      - SQL operators/clauses in combination (WHERE, JOIN, GROUP BY, ORDER BY)
      - High symbol density (= < > ! ( ) _ *) → likely code paste

    Returns False for natural-language explanations that merely mention SQL terms
    like "null", "SQL", "isNull" — these should still go through text adaptation.
    """
    if not text:
        return False

    # Actual SQL statements (DML/DDL)
    has_sql_stmt = bool(re.search(
        r'\b(?:SELECT\s+\S|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|'
        r'WITH\s+\w+\s+AS\s*\(|CREATE\s+(?:TABLE|INDEX|VIEW)|DROP\s+(?:TABLE|VIEW))\b',
        text, re.IGNORECASE,
    ))

    # SQL function calls with parentheses
    has_sql_func = bool(re.search(
        r'\b(?:ISNULL|COALESCE|NULLIF|IFNULL|NVL|IIF|CASE\s+WHEN)\s*\(',
        text, re.IGNORECASE,
    ))

    # SQL clause keywords (WHERE, JOIN, etc.) — only meaningful in code context
    has_sql_clause = bool(re.search(
        r'\b(?:WHERE|(?:INNER|LEFT|RIGHT|FULL)\s+(?:OUTER\s+)?JOIN|'
        r'GROUP\s+BY|ORDER\s+BY|HAVING|UNION\s+(?:ALL)?)\b',
        text, re.IGNORECASE,
    ))

    # Backtick-wrapped identifiers → code
    has_backticks = bool(re.search(r'`[A-Z_]{2,}`', text, re.IGNORECASE))

    # High symbol density → likely code paste
    symbol_density = len(re.findall(r'[=<>!()*_]', text)) / max(len(text), 1)
    has_high_symbols = symbol_density > 0.10

    return has_sql_stmt or has_sql_func or has_sql_clause or has_backticks or has_high_symbols


# SQL clause boundary keywords — used for breath-splitting.
# No bare JOIN: "INNER JOIN" would split into "INNER" + "JOIN..." without this exclusion.
_SQL_CLAUSE_KWS = (
    r'(?:FULL\s+OUTER\s+JOIN|INNER\s+JOIN|LEFT\s+OUTER\s+JOIN|LEFT\s+JOIN|'
    r'RIGHT\s+OUTER\s+JOIN|RIGHT\s+JOIN|CROSS\s+JOIN|'
    r'GROUP\s+BY|ORDER\s+BY|HAVING|WHERE|FROM|UNION\s+ALL|UNION|EXCEPT|INTERSECT)'
)


def split_sql_clauses(text: str) -> list[str]:
    """
    Split an actual SQL query into clause-level chunks for natural TTS pacing.
    Only applied when text contains SELECT/UPDATE/DELETE/INSERT.
    Each clause is a natural breathing unit.
    """
    if not _looks_like_sql_query(text):
        return [text]
    # Split before each clause keyword (lookahead — keeps keyword with its clause)
    parts = re.split(r'(?i)(?=\b' + _SQL_CLAUSE_KWS + r'\b)', text)
    result = [p.strip() for p in parts if p.strip()]
    return result if len(result) > 1 else [text]


def normalize_sql_for_tts(text: str) -> str:
    """
    Generic SQL/code-aware text normalization for better TTS pronunciation.
    Generic rules, NOT a per-term map.
    Called automatically for sql content_type OR when is_sql_heavy() detects code.
    Does NOT split clauses — use split_sql_clauses() for that.
    """
    if not text:
        return text

    # 0) Comparison operators — must run BEFORE the plain = rule below.
    #    Order: longer operators first (<=, >=, !=, <>) before plain < > =
    text = re.sub(r'<=',  ' menší nebo rovno ',  text)
    text = re.sub(r'>=',  ' větší nebo rovno ',   text)
    text = re.sub(r'!=',  ' nerovná se ',          text)
    text = re.sub(r'<>',  ' nerovná se ',          text)
    text = re.sub(r'(?<![<>!=])<(?!=)',  ' menší než ',  text)
    text = re.sub(r'(?<![<>!=])>(?!=)',  ' větší než ',   text)

    # 0b) SELECT * wildcard → "hvězdička"
    text = re.sub(
        r'(?i)(?<=SELECT\s)\s*\*',
        ' hvězdička',
        text,
    )
    # General standalone * (not inside words like Python **kwargs)
    text = re.sub(r'(?<!\w)\*(?!\w)', ' krát ', text)

    # 0c) % in LIKE patterns → "procent" (if after quotes or already processed)
    #     Also catches bare % used as wildcard
    text = re.sub(r'(?<!\d)%(?!\d)', ' procent ', text)

    text = re.sub(r' {2,}', ' ', text).strip()

    # 1) schema.table notation → "schema bodka table"
    text = re.sub(
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b',
        lambda m: m.group(1) + ' bodka ' + m.group(2),
        text,
    )

    # 2) Remove/flatten parentheses in function call context: func(a, b) → func a, b
    text = re.sub(r'\(', ' ', text)
    text = re.sub(r'\)', ' ', text)
    text = re.sub(r' {2,}', ' ', text).strip()

    # 3) snake_case identifiers → split on underscore (e.g. customer_id → customer id)
    text = re.sub(
        r'\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b',
        lambda m: m.group(0).replace('_', ' '),
        text,
    )

    # 4) CamelCase → insert space before uppercase after lowercase (min 5 chars)
    #    e.g. CustomerOrders → Customer Orders, OrderNumber → Order Number
    text = re.sub(
        r'\b[A-Z][a-zA-Z]{4,}\b',
        lambda m: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', m.group(0)),
        text,
    )

    # 5) Standalone = (SQL equality / assignment) → "rovná se"
    #    Avoid double-replacement: skip if already part of <=, >=, !=, <>
    text = re.sub(r'(?<![<>!=])\s*=\s*(?!=)', ' rovná se ', text)
    text = re.sub(r' {2,}', ' ', text).strip()

    # 6) Micro-pauses: ONLY after FROM, WHERE, JOIN types — not GROUP BY/ORDER BY
    #    (too many pauses sound robotic; FROM/WHERE/JOIN are the main breath points)
    _PAUSE_KWS = [
        r'\bINNER\s+JOIN\b', r'\bLEFT\s+JOIN\b', r'\bRIGHT\s+JOIN\b',
        r'\bFULL\s+OUTER\s+JOIN\b', r'\bJOIN\b',
        r'\bWHERE\b', r'\bFROM\b',
    ]
    for kw_pat in _PAUSE_KWS:
        text = re.sub(
            kw_pat + r'(?!\s*[,.])',
            lambda m: m.group(0) + ',',
            text,
            flags=re.IGNORECASE,
        )

    # 7) Common SQL type abbreviations
    text = re.sub(r'(?<!\w)VARCHAR2?\b', 'var-char', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\w)NVARCHAR\b',  'en-var-char', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\w)BIGINT\b',    'big-int', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\w)DATETIME\b',  'dejt-tajm', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\w)NCHAR\b',     'en-char', text, flags=re.IGNORECASE)

    # 8) SQL function names — phonetic Slovak pronunciation (TTS only, not stored in text)
    # Strip backticks and ALL unicode typographic/curly quotes
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'[„\u201e\u201c\u201d\u2018\u2019""]', '', text)  # strip all curly/low quotes
    # Slovak translated forms (Gemma/Google adaptation may produce these lowercase variants)
    text = re.sub(r'\biz\s+not\s+nul\b',  'íz not nul', text, flags=re.IGNORECASE)
    text = re.sub(r'\biz\s+nul\b',        'ajznul',     text, flags=re.IGNORECASE)  # iz nul → ajznul
    text = re.sub(r'\bis\s+not\s+nul\b',  'íz not nul', text, flags=re.IGNORECASE)  # is not nul (1L)
    text = re.sub(r'\bis\s+nul\b',        'ajznul',     text, flags=re.IGNORECASE)  # is nul (1L) → ajznul
    text = re.sub(r'\bje\s+nul\b',        'ajznul',     text, flags=re.IGNORECASE)  # je nul (SK) → ajznul
    text = re.sub(r'\bNUL\s+if\b',        'nulíf',      text, flags=re.IGNORECASE)  # NULLIF→NUL if
    text = re.sub(r'\bkoales\w*\b',       'koalesk',    text, flags=re.IGNORECASE)  # COALESCE→koalesk
    # Standard SQL keyword forms (long patterns before short)
    text = re.sub(r'\bIS\s+NOT\s+NULL\b', 'íz not nul', text, flags=re.IGNORECASE)
    text = re.sub(r'\bIS\s+NULL\b',       'ajznul',     text, flags=re.IGNORECASE)
    text = re.sub(r'\bNOT\s+NULL\b',      'not nul',    text, flags=re.IGNORECASE)
    text = re.sub(r'\bISNULL\b',          'ajznul',     text, flags=re.IGNORECASE)
    text = re.sub(r'\bNULLIF\b',          'nulíf',      text, flags=re.IGNORECASE)
    text = re.sub(r'\bCOALESCE\b',        'koalesk',    text, flags=re.IGNORECASE)
    text = re.sub(r'\bNULL\b',            'nul',        text, flags=re.IGNORECASE)
    text = re.sub(r'\bNUL\b',             'nul',        text, flags=re.IGNORECASE)  # single-L Slovak form
    text = re.sub(r'\bSQL\b',             'esíkjúel',   text, flags=re.IGNORECASE)

    return text


def normalize_sql_for_chatterbox_tts(text: str) -> str:
    """
    Gentle SQL normalization for Chatterbox SK models.

    Unlike normalize_sql_for_tts(), this keeps function names such as ISNULL,
    IS NULL and IS NOT NULL untouched because aggressive phonetic rewrites
    ("ajznul") made Slovak fine-tuned Chatterbox models less stable in practice.
    We only normalize a narrow set of terms that consistently helped:
      - SQL    -> esíkjúel
      - NULL   -> nul
      - ISNULL -> íz nul
      - NULLIF -> nulíf
    """
    if not text:
        return text

    out = text

    # Strip code quotes around keywords before lightweight phonetic prep.
    out = re.sub(r'`([^`]+)`', r'\1', out)
    out = re.sub(r'[„\u201e\u201c\u201d\u2018\u2019""]', '', out)

    # Order matters: longer tokens first.
    out = re.sub(r'\bNULLIF\b', 'nulíf', out, flags=re.IGNORECASE)
    out = re.sub(r'\bISNULL\b', 'íz nul', out, flags=re.IGNORECASE)
    out = re.sub(r'\bSQL\b', 'esíkjúel', out, flags=re.IGNORECASE)
    out = re.sub(r'\bNULL\b', 'nul', out, flags=re.IGNORECASE)
    out = re.sub(r'\bNUL\b', 'nul', out, flags=re.IGNORECASE)

    out = re.sub(r' {2,}', ' ', out).strip()
    return out


def force_keep_cool(src: str, dst: str) -> str:
    if not src:
        return dst
    s = src.lower().strip()
    d = (dst or "").strip()
    if "cool" in s:
        if d.lower() in ("a", "and", ""):
            return "kúl"
        if "kúl" not in d.lower() and "cool" not in d.lower():
            return d + " kúl"
    return dst


def apply_tts_pronunciation_fixes(text: str) -> str:
    if not text:
        return text
    text = apply_phonetic_respelling(text)
    return text


def protect_phonetic_terms(text: str, use_guillemets: bool = False) -> tuple[str, dict[str, str]]:
    import re
    if not text:
        return text, {}
    terms: list[str] = []
    seen = set()
    # Use ONLY base PHONETIC_MAP values for translation protection.
    # User phonetic_map.json is for TTS phonetic respelling AFTER translation — it must not
    # interfere with translation protection (e.g. "engine"→"enžin" override must not suppress
    # the identity "engine"→"engine" protection from PHONETIC_MAP).
    for v in PHONETIC_MAP.values():
        if isinstance(v, str) and v and v not in seen:
            seen.add(v)
            terms.append(v)
    terms.sort(key=len, reverse=True)
    protected = text
    token_map: dict[str, str] = {}
    token_idx = 0
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        _flags = 0 if term.isupper() else re.IGNORECASE
        if re.search(pattern, protected, flags=_flags):
            if use_guillemets:
                def _repl_g(m):
                    return f"«{m.group(0)}»"
                protected = re.sub(pattern, _repl_g, protected, flags=_flags)
            else:
                token = f"__PHON_{token_idx}__"
                matched_text = term
                def _repl(m):
                    nonlocal matched_text
                    matched_text = m.group(0)
                    return token
                protected = re.sub(pattern, _repl, protected, flags=_flags)
                token_map[token] = matched_text
                token_idx += 1
    return protected, token_map


def restore_phonetic_terms(text: str, token_map: dict[str, str]) -> str:
    if not text or not token_map:
        return text
    # Normalize MADLAD/T5-corrupted PHON tokens. T5 tokenizer splits __PHON_0__ into
    # underscores and words with spaces, producing many variants:
    #   "__ PHON_1__", "__PHON_1 __", "_ _ PHON _ 1__", "__phon_0_", "__ PHON _0__",
    #   "__phone_4__" (extra 'e'), "__ phon _2__"
    # [Ee]? allows 4-char "PHON" or 5-char "phone"; [\s_]* allows spaces/underscores around digits
    text = re.sub(
        r'_+[\s_]*[Pp][Hh][Oo][Nn][Ee]?[\s_]*_?[\s_]*(\d+)[\s_]*_+',
        lambda m: f'__PHON_{m.group(1)}__',
        text
    )
    restored = text
    for token, term in token_map.items():
        restored = restored.replace(token, term)
    # Log ak ostali neobnovené tokeny — LLM ich deformoval nad rámec regex opravy
    remaining = re.findall(r'__PHON_\d+__', restored)
    if remaining:
        print(f"[RESTORE-PHON] WARNING: {len(remaining)} unrestored tokens: {remaining} in: {restored[:120]!r}", flush=True)
        # Vymaž zvyšné tokeny aby nešli do TTS
        restored = re.sub(r'__PHON_\d+__', '', restored).strip()
    return restored


def strip_guillemets(text: str) -> str:
    if not text:
        return text
    return text.replace("«", "").replace("»", "")


# ---------------------------------------------------------------------------
# Content type & profile presets
# ---------------------------------------------------------------------------

CONTENT_TYPE_HINTS = {
    "general": "Keep the same tone and similar sentence length.",
    "educational": "Translate in a clear lecture style: precise, structured, no slang. Use consistent terminology throughout. Preserve lists, steps, and definitions.",
    "podcast": "Translate as natural spoken dialogue. Keep it conversational and flowing. Use fillers only if present in the original.",
    "review": "Translate as a review: evaluative tone, clear pros/cons. Keep the same level of subjectivity; do not intensify or soften opinions.",
    "technical": "Translate with maximum technical accuracy. Do not change code, commands, parameters, file paths, logs, or identifiers. If a Slovak term is standard, use it; otherwise keep the English term (optionally add a one-time Slovak clarification in parentheses).",
    "news": "Translate in a neutral news style: factual, objective, no emotive language. Preserve quotes, names, places, and numbers exactly.",
    "entertainment": "Translate in an entertaining, lively tone. Adapt humor so it works in Slovak without changing the point or adding new jokes.",
    "programming": (
        "This is a programming or game development tutorial. "
        "Translate 'engine' in programming context as 'enžin' (NOT 'motor'): "
        "game engine = herný enžin, graphics engine = grafický enžin, physics engine = fyzikálny enžin. "
        "Translate 'shader' as 'šejder' for TTS readability. "
        "Translate 'era' as 'éra' (do NOT confuse with 'error' = 'chyba'). "
        "IMPORTANT: The word 'nothing' in everyday speech (e.g. 'knowing nothing', 'from nothing') "
        "means 'nič' or 'neznalec' — do NOT translate as 'null', 'NULL', or 'hodnota null'. "
        "'NULL' is ONLY used for database/SQL NULL, never for 'knowing nothing'. "
        "'I went from knowing nothing' = 'prešiel som od absolútnej nuly' or 'od úplného neznalca', NOT 'od hodnoty null'. "
        "If using 'nula' as idiom, use correct feminine: 'od absolútnej nuly' (NOT 'od absolútneho nula'). "
        "Keep unchanged: C++, DirectX, OpenGL, SDL, pathfinding, voxel, framework, Pac-Man. "
        "Use 'potravinový reťazec' for 'food chain'. "
        "Use 'nízkoúrovňové programovanie' for 'low level programming'. "
        "Prefer natural spoken Slovak over literal translation."
    ),
    "sql": (
        "This is a SQL/database tutorial. "
        "Keep ALL SQL keywords UNCHANGED (do NOT translate): "
        "SELECT, FROM, WHERE, JOIN, LEFT JOIN, RIGHT JOIN, INNER JOIN, FULL JOIN, CROSS JOIN, "
        "GROUP BY, ORDER BY, HAVING, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, TRUNCATE, "
        "INDEX, CONSTRAINT, PRIMARY KEY, FOREIGN KEY, UNIQUE, NOT NULL, DEFAULT, REFERENCES, "
        "TRIGGER, VIEW, PROCEDURE, FUNCTION, DECLARE, SET, BEGIN, END, "
        "UNION, UNION ALL, INTERSECT, EXCEPT, WITH, AS, DISTINCT, TOP, LIMIT, OFFSET, "
        "IS NULL, IS NOT NULL, CASE, WHEN, THEN, ELSE, END, LIKE, BETWEEN, IN, EXISTS, "
        "PARTITION BY, OVER, PIVOT, UNPIVOT, ROLLBACK, COMMIT, TRANSACTION, CURSOR. "
        "Keep ALL SQL built-in functions UNCHANGED: "
        "COALESCE, ISNULL, NULLIF, NVL, COUNT, SUM, AVG, MIN, MAX, "
        "CAST, CONVERT, TRY_CAST, TRY_CONVERT, "
        "UPPER, LOWER, LEN, LENGTH, SUBSTRING, SUBSTR, CHARINDEX, PATINDEX, "
        "REPLACE, REPLICATE, REVERSE, LTRIM, RTRIM, TRIM, CONCAT, STRING_AGG, "
        "DATEPART, DATEDIFF, DATEADD, GETDATE, GETUTCDATE, SYSDATETIME, FORMAT, "
        "ROW_NUMBER, RANK, DENSE_RANK, NTILE, LEAD, LAG, FIRST_VALUE, LAST_VALUE, "
        "IIF, CHOOSE, ABS, ROUND, CEILING, FLOOR, POWER, SQRT. "
        "Keep database terms in English: query, schema, table, column, row, record, index, "
        "stored procedure, transaction, subquery, aggregate, join, NULL. "
        "Do NOT translate table names, column names, alias names, or SQL code snippets."
    ),
}

PROFILE_PRESETS = {
    "general": {
        "vad_llama_min_speech_ms": 3500,
        "vad_llama_min_silence_ms": 4500,
        "vad_madlad_min_speech_ms": 2500,
        "vad_madlad_min_silence_ms": 3000,
        "merge_llama_max_gap_s": 1.2,
        "merge_llama_max_dur_s": 18.0,
        "merge_madlad_max_gap_s": 1.2,
        "merge_madlad_max_dur_s": 18.0,
        "stretch_min": 0.80,
        "stretch_max": 1.25,
        "pass1": (
            "Translate EN→CZ naturally (neutral spoken Czech).\n"
            "Keep meaning, names, numbers.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
        "pass2": (
            "Polish Czech translation to sound fluent and natural.\n"
            "Fix literal phrasing, keep meaning exactly.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
    },
    "educational": {
        "vad_llama_min_speech_ms": 5000,
        "vad_llama_min_silence_ms": 6000,
        "vad_madlad_min_speech_ms": 3500,
        "vad_madlad_min_silence_ms": 4000,
        "merge_llama_max_gap_s": 2.0,
        "merge_llama_max_dur_s": 25.0,
        "merge_madlad_max_gap_s": 1.5,
        "merge_madlad_max_dur_s": 20.0,
        "stretch_min": 0.85,
        "stretch_max": 1.20,
        "pass1": (
            "Translate EN→CZ for an educational YouTube voiceover.\n"
            "Style: clear, explanatory, friendly spoken Czech.\n"
            "Use informal spoken Czech (tykání).\n"
            "Keep terminology consistent.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
        "pass2": (
            "You are a Czech editor for educational narration.\n"
            "Make the Czech fluent, easy to follow, natural spoken style.\n"
            "Use informal spoken Czech (tykání).\n"
            "Keep meaning exactly, keep terminology consistent.\n"
            "Pay extra attention to negations (not, never, without). Never flip polarity.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
    },
    "podcast": {
        "vad_llama_min_speech_ms": 2500,
        "vad_llama_min_silence_ms": 2500,
        "vad_madlad_min_speech_ms": 2000,
        "vad_madlad_min_silence_ms": 2000,
        "merge_llama_max_gap_s": 0.8,
        "merge_llama_max_dur_s": 12.0,
        "merge_madlad_max_gap_s": 0.8,
        "merge_madlad_max_dur_s": 12.0,
        "stretch_min": 0.75,
        "stretch_max": 1.30,
        "pass1": (
            "Translate EN→CZ as natural spoken conversation/podcast.\n"
            "Keep informal flow, preserve questions and reactions.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
        "pass2": (
            "Polish Czech text to sound like real conversation.\n"
            "Make it relaxed, spoken, natural.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
    },
    "review": {
        "vad_llama_min_speech_ms": 4000,
        "vad_llama_min_silence_ms": 4500,
        "vad_madlad_min_speech_ms": 3000,
        "vad_madlad_min_silence_ms": 3500,
        "merge_llama_max_gap_s": 1.5,
        "merge_llama_max_dur_s": 20.0,
        "merge_madlad_max_gap_s": 1.5,
        "merge_madlad_max_dur_s": 20.0,
        "stretch_min": 0.80,
        "stretch_max": 1.25,
        "pass1": (
            "Translate EN→CZ for a review voiceover.\n"
            "Keep evaluative tone and opinions.\n"
            "Keep product names unchanged.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
        "pass2": (
            "Polish Czech review narration to sound fluent and natural.\n"
            "Keep opinions, keep names/numbers.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
    },
    "technical": {
        "vad_llama_min_speech_ms": 4500,
        "vad_llama_min_silence_ms": 5000,
        "vad_madlad_min_speech_ms": 3500,
        "vad_madlad_min_silence_ms": 4000,
        "merge_llama_max_gap_s": 1.8,
        "merge_llama_max_dur_s": 22.0,
        "merge_madlad_max_gap_s": 1.8,
        "merge_madlad_max_dur_s": 22.0,
        "stretch_min": 0.90,
        "stretch_max": 1.15,
        "pass1": (
            "Translate EN→CZ in technical/scientific style.\n"
            "Priority: accuracy and consistent terminology.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
        "pass2": (
            "Polish technical Czech translation for clarity,\n"
            "without changing meaning or terminology.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
    },
    "news": {
        "vad_llama_min_speech_ms": 3000,
        "vad_llama_min_silence_ms": 3500,
        "vad_madlad_min_speech_ms": 3000,
        "vad_madlad_min_silence_ms": 3500,
        "merge_llama_max_gap_s": 1.0,
        "merge_llama_max_dur_s": 15.0,
        "merge_madlad_max_gap_s": 1.0,
        "merge_madlad_max_dur_s": 15.0,
        "stretch_min": 0.90,
        "stretch_max": 1.15,
        "pass1": (
            "Translate EN→CZ in news style: concise, neutral, factual.\n"
            "Keep names and numbers exact.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
        "pass2": (
            "Polish Czech news narration: clean, neutral, concise.\n"
            "Keep meaning exactly.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
    },
    "entertainment": {
        "vad_llama_min_speech_ms": 2500,
        "vad_llama_min_silence_ms": 3000,
        "vad_madlad_min_speech_ms": 2500,
        "vad_madlad_min_silence_ms": 3000,
        "merge_llama_max_gap_s": 1.0,
        "merge_llama_max_dur_s": 15.0,
        "merge_madlad_max_gap_s": 1.0,
        "merge_madlad_max_dur_s": 15.0,
        "stretch_min": 0.75,
        "stretch_max": 1.35,
        "pass1": (
            "Translate EN→CZ for entertainment content.\n"
            "Style: lively, engaging, spoken Czech.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
        "pass2": (
            "Polish Czech text to sound energetic and fun,\n"
            "while keeping meaning.\n"
            "Keep [SEG###] tags unchanged.\n"
            "Output only Czech text."
        ),
    },
}


def get_content_type_instruction(content_type: str) -> str:
    return CONTENT_TYPE_HINTS.get(content_type, "")


def relax_translation_hint_for_rewrite(hint: str) -> str:
    """Remove timing/length pressure from translation hints.

    Translation should preserve meaning and natural phrasing first.
    Timing/CPS is handled later in the adaptation stage.
    """
    if not hint:
        return ""
    out = hint
    out = re.sub(
        r"Keep the same tone and similar sentence length\.?",
        "Keep the same tone.",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"Keep it concise for dubbing timing\.?",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s+", " ", out).strip()
    return out


def apply_profile_defaults(args) -> None:
    """Apply preset profile values when CLI values are still defaults."""
    profile = PROFILE_PRESETS.get(args.content_type)
    if not profile:
        args.pass1_prompt = ""
        args.pass2_prompt = ""
        return

    is_madlad = bool(args.use_madlad)
    # These must match argparse defaults in config.build_parser() exactly,
    # otherwise the "has user changed this?" check never triggers.
    default_vad_speech = 3500
    default_vad_silence = 4500
    default_stretch_min = 0.85
    default_stretch_max = 1.30
    default_merge_gap = 0.0
    default_merge_dur = 0.0

    if args.vad_min_speech_ms == default_vad_speech:
        args.vad_min_speech_ms = (
            profile["vad_madlad_min_speech_ms"] if is_madlad else profile["vad_llama_min_speech_ms"]
        )
    if args.vad_min_silence_ms == default_vad_silence:
        args.vad_min_silence_ms = (
            profile["vad_madlad_min_silence_ms"] if is_madlad else profile["vad_llama_min_silence_ms"]
        )
    if args.stretch_min == default_stretch_min:
        args.stretch_min = profile["stretch_min"]
    if args.stretch_max == default_stretch_max:
        args.stretch_max = profile["stretch_max"]
    if args.merge_max_gap_s == default_merge_gap:
        args.merge_max_gap_s = (
            profile["merge_madlad_max_gap_s"] if is_madlad else profile["merge_llama_max_gap_s"]
        )
    if args.merge_max_dur_s == default_merge_dur:
        args.merge_max_dur_s = (
            profile["merge_madlad_max_dur_s"] if is_madlad else profile["merge_llama_max_dur_s"]
        )
    if args.merge_max_gap_s > 0 and args.merge_max_dur_s > 0:
        args.merge_segments = True

    pass1_prompt = profile["pass1"]
    pass2_prompt = profile["pass2"]

    # Adapt CZ prompts for SK target language
    tgt = getattr(args, "tgt_lang", "cs").lower()
    if tgt in ("sk", "slk"):
        def _adapt_sk(p):
            return (p
                .replace("EN→CZ", "EN→SK")
                .replace("Czech", "Slovak")
                .replace("czechtinu", "slovenčinu")
                .replace("češtinu", "slovenčinu")
                .replace("spoken Czech (tykání)", "spoken Slovak (tykanie)")
                .replace("informal spoken Czech", "informal spoken Slovak"))
        pass1_prompt = _adapt_sk(pass1_prompt)
        pass2_prompt = _adapt_sk(pass2_prompt)

    args.pass1_prompt = pass1_prompt
    args.pass2_prompt = pass2_prompt


# ---------------------------------------------------------------------------
# Tagged segment text utilities
# ---------------------------------------------------------------------------

def build_tagged_segment_text(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        seg_text = (seg.get("text") or "").strip()
        lines.append(f"[SEG{i:03d}] {seg_text}")
    return "\n".join(lines).strip()


def parse_tagged_segment_text(tagged_text: str, expected_count: int) -> dict[int, str]:
    if not tagged_text:
        return {}
    pattern = re.compile(
        r"^\[SEG(\d{3,6})\]\s*(.*?)(?=^\[SEG\d{3,6}\]\s*|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    out: dict[int, str] = {}
    for m in pattern.finditer(tagged_text):
        idx = int(m.group(1))
        if 1 <= idx <= expected_count:
            out[idx] = (m.group(2) or "").strip()
    return out


def extract_seg_tags_in_order(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\[SEG\d{3,6}\]", text)


def tags_match_source(source_tagged: str, candidate_tagged: str) -> bool:
    return extract_seg_tags_in_order(source_tagged) == extract_seg_tags_in_order(candidate_tagged)


# ---------------------------------------------------------------------------
# Translation functions
# ---------------------------------------------------------------------------

def create_glossary_with_llama(
    full_text: str,
    llama_model: str,
    tgt_lang: str,
    n_ctx: int,
    n_gpu_layers: int,
    llm_instance=None,
    content_type: str = "general"
) -> str:
    if llm_instance is None:
        llm = Llama(model_path=llama_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    else:
        llm = llm_instance
    lang_name = target_lang_name(tgt_lang)
    content_hint = get_content_type_instruction(content_type)
    max_text = 6000
    text_for_analysis = full_text[:max_text] if len(full_text) > max_text else full_text
    content_context = f"\nContent type: {content_hint}" if content_hint else ""
    system = f"""You are a translation preparation assistant. Analyze the following text and create a glossary for consistent translation into {lang_name}.{content_context}

Extract and translate:
1. NAMES: People names, company names, product names (keep original or transliterate appropriately)
2. TECHNICAL TERMS: Technical/specialized vocabulary with correct {lang_name} translations
3. KEY PHRASES: Recurring phrases that should be translated consistently

Format your response as a simple list:
NAMES:
- Original -> Translation
TECHNICAL:
- Original -> Translation
KEY PHRASES:
- Original -> Translation

Be concise. Only include terms that appear in the text and need consistent translation."""

    print("[GLOSSARY] Analyzing text for consistent translation...", flush=True)
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Analyze this text:\n\n{text_for_analysis}"},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    glossary = resp["choices"][0]["message"]["content"].strip()
    print(f"[GLOSSARY] Created glossary with context", flush=True)
    return glossary


def translate_with_llama(text: str, llama_model: str, tgt_lang: str, max_chars: int, n_ctx: int, n_gpu_layers: int, temperature: float, content_type: str = "general") -> str:
    llm = Llama(model_path=llama_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    try:
        lang_name = target_lang_name(tgt_lang)
        content_hint = get_content_type_instruction(content_type)
        content_instruction = f" {content_hint}" if content_hint else ""
        system = (
            f"You are a natural dubbing translation engine. Translate the following text into {lang_name}.{content_instruction} "
            "Preserve the original meaning, tone, slang, and intensity, but prefer fluent spoken phrasing over literal word-by-word calques. "
            "Do not add warnings, do not omit important information, and do not provide explanations. "
            f"IMPORTANT: When you keep English words (brand names, tech terms), write them phonetically in {lang_name} spelling so TTS reads them correctly. "
            + (
                "Examples: Java→Džáva, YouTube→Júťjúb, Python→Pajton, cloud→klaud, nice→najs, update→apdejt, feature→fíčura. "
                if lang_name == "Czech" else
                "Examples: Java→Džava, YouTube→Júťjúb, Python→Pajton, cloud→klaud, nice→najs, update→apdejt, feature→fíčura. "
                if lang_name == "Slovak" else
                "Examples: Java→Džáva, YouTube→Júťjúb, Python→Pajton. "
            ) +
            "Output only the final translation."
        )
        chunks = split_text(text, max_chars=max_chars)
        out = []
        total = len(chunks)
        for i, chunk in enumerate(chunks, 1):
            print(f"[LLAMA] Chunk {i}/{total}", flush=True)
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": chunk},
                ],
                temperature=temperature,
                max_tokens=1024,
            )
            translated = resp["choices"][0]["message"]["content"].strip()
            out.append(translated)
        return "\n".join(out).strip()
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        print("[LLAMA] Model released from memory", flush=True)


def api_chat_complete(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout_s: int = 120,
) -> str:
    base = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    is_gpt5_family = (model or "").startswith("gpt-5")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if is_gpt5_family:
        # GPT-5 chat models can burn the whole completion budget on reasoning
        # unless we force minimal reasoning and use max_completion_tokens.
        payload["reasoning_effort"] = "minimal"
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if not r.ok:
        raise RuntimeError(f"API translate HTTP {r.status_code}: {r.text[:400]}")
    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"API translate invalid response: {e}; body={str(data)[:400]}")
    return (content or "").strip()


def translate_segment_with_api(
    text: str,
    duration_sec: float,
    tgt_lang: str,
    api_key: str,
    api_base_url: str,
    api_model: str,
    provider: str,
    content_type: str = "general",
    glossary: str = None,
    prev_translation: str = None,
    custom_system_prompt: str = "",
    topic_hint: str = "",
) -> str:
    lang_name = target_lang_name(tgt_lang)
    protected_text, token_map = protect_phonetic_terms(text)
    # Custom prompt overrides default system prompt if provided
    if custom_system_prompt:
        word_count = len(text.split())
        system_prompt = custom_system_prompt.replace("{lang}", lang_name).replace("{words}", str(word_count)).replace("{duration}", f"{duration_sec:.1f}")
        if token_map:
            system_prompt += " Preserve placeholders like __PHON_0__ exactly."
    else:
        content_hint = relax_translation_hint_for_rewrite(get_content_type_instruction(content_type))
        system_parts = [
            f"You are an expert dubbing translator. Translate into {lang_name}.",
            "Return ONLY translated text, no explanations.",
            "Preserve meaning, tone, intent, and important nuance faithfully.",
            "Prefer natural spoken phrasing over literal word-by-word calques.",
            "Keep names, numbers, units, commands, paths, and code unchanged.",
            "Do not add information, do not omit important information, and do not summarize.",
            "Do not optimize for timing here. Timing, shortening, and CPS control will happen in a later pass.",
            "Do not force the translation to match the original length. Correctness and naturalness come first.",
        ]
        if token_map:
            system_parts.append("Preserve placeholders like __PHON_0__ exactly.")
        if content_hint:
            system_parts.append(content_hint)
        if topic_hint:
            system_parts.append(f"Topic context: {topic_hint}.")
        if glossary:
            system_parts.append(f"\nGlossary for consistency:\n{glossary}")
        if prev_translation:
            system_parts.append(f"\nPrevious translated context:\n{prev_translation}")
        system_prompt = " ".join(system_parts)
    translated = api_chat_complete(
        api_key=api_key,
        base_url=api_base_url,
        model=api_model,
        system_prompt=system_prompt,
        user_prompt=protected_text,
        temperature=0.15,
        max_tokens=512,
    )
    translated = translated.replace(">>>", "").replace("<<<", "").strip()
    translated = restore_phonetic_terms(translated, token_map)
    return translated


def translate_with_api(
    text: str,
    tgt_lang: str,
    max_chars: int,
    api_key: str,
    api_base_url: str,
    api_model: str,
    provider: str,
    content_type: str = "general",
) -> str:
    chunks = split_text(text, max_chars=max_chars)
    out = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        print(f"[API-TRANS] Chunk {i}/{total} provider={provider}", flush=True)
        translated = translate_segment_with_api(
            text=chunk,
            duration_sec=max(1.0, len(chunk.split()) / 2.5),
            tgt_lang=tgt_lang,
            api_key=api_key,
            api_base_url=api_base_url,
            api_model=api_model,
            provider=provider,
            content_type=content_type,
            glossary=None,
            prev_translation=out[-1][-200:] if out else None,
        )
        out.append(translated)
    return "\n".join(out).strip()


def translate_segment_with_context(
    segments: list[dict],
    current_idx: int,
    llama_model: str,
    tgt_lang: str,
    n_ctx: int,
    n_gpu_layers: int,
    temperature: float,
    llm_instance=None,
    glossary: str = None,
    window_size: int = 2,
    content_type: str = "general",
) -> str:
    from stt import get_sliding_window_context
    if llm_instance is None:
        llm = Llama(model_path=llama_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    else:
        llm = llm_instance
    seg = segments[current_idx]
    current_text = (seg.get("text") or "").strip()
    duration = seg["end"] - seg["start"]
    before_context, after_context = get_sliding_window_context(segments, current_idx, window_size)
    lang_name = target_lang_name(tgt_lang)
    word_count = len(current_text.split())
    content_hint = relax_translation_hint_for_rewrite(get_content_type_instruction(content_type))
    system_parts = [f"You are a translation engine. Translate into {lang_name}."]
    if content_hint:
        system_parts.append(content_hint)
    system_parts.extend([
        "Keep meaning and tone faithful. Do not invent information.",
        "Do not optimize for timing here. Timing and shortening happen later.",
        "Do not force the translation to match the original length.",
        "Output ONLY translation of the CURRENT segment (marked with >>> <<<).",
        "Do NOT output context segments.",
    ])
    if glossary:
        system_parts.append(f"\nGlossary for consistency:\n{glossary}")
    system = " ".join(system_parts)
    user_parts = []
    if before_context:
        user_parts.append(f"[Previous context: {before_context}]")
    user_parts.append(f">>> {current_text} <<<")
    if after_context:
        user_parts.append(f"[Following context: {after_context}]")
    user_message = "\n".join(user_parts)
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        temperature=min(temperature, 0.10),
        max_tokens=512,
    )
    translated = resp["choices"][0]["message"]["content"].strip()
    translated = force_keep_cool(current_text, translated)
    translated = translated.replace(">>>", "").replace("<<<", "").strip()
    trans_words = len(translated.split())
    print(f"    Original: {word_count} words, Translation: {trans_words} words", flush=True)
    return translated


def translate_segment_with_llama(
    text: str,
    duration_sec: float,
    llama_model: str,
    tgt_lang: str,
    n_ctx: int,
    n_gpu_layers: int,
    temperature: float,
    llm_instance=None,
    glossary: str = None,
    prev_translation: str = None,
    custom_system_prompt: str = "",
    content_type: str = "general",
    topic_hint: str = "",
) -> str:
    if llm_instance is None:
        llm = Llama(model_path=llama_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    else:
        llm = llm_instance
    lang_name = target_lang_name(tgt_lang)
    word_count = len(text.split())
    # Custom prompt overrides everything if provided
    if custom_system_prompt:
        system = custom_system_prompt.replace("{lang}", lang_name).replace("{words}", str(word_count)).replace("{duration}", f"{duration_sec:.1f}")
    else:
        content_hint = relax_translation_hint_for_rewrite(get_content_type_instruction(content_type))
        system_parts = [
            f"You are a translation engine. Translate into {lang_name}.",
            "Keep meaning and tone faithful. Do not invent information.",
            "Return only the translated text.",
            "Do not optimize for timing here. Timing and shortening happen later.",
            "Do not force the translation to match the original length.",
        ]
        if content_hint:
            system_parts.append(content_hint)
        if topic_hint:
            system_parts.append(f"Topic context: {topic_hint}.")
        if glossary:
            system_parts.append(f"\nUse this glossary for consistent translation:\n{glossary}")
        if prev_translation:
            system_parts.append(f"\nPrevious translation for context (maintain consistency):\n\"{prev_translation}\"")
        system_parts.append("\nOutput ONLY the direct translation text. No notes, no explanations.")
        system = " ".join(system_parts)
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=min(temperature, 0.10),
        max_tokens=512,
    )
    translated = resp["choices"][0]["message"]["content"].strip()
    translated = force_keep_cool(text, translated)
    lines = translated.split('\n')
    clean_lines = []
    skip_glossary = True
    for line in lines:
        line_stripped = line.strip()
        if skip_glossary:
            is_glossary_line = (
                line_stripped.startswith(('NAMES:', 'TECHNICAL:', 'KEY PHRASES:', '-')) or
                '->' in line_stripped or
                (line_stripped.isupper() and len(line_stripped.split()) <= 3) or
                not line_stripped
            )
            if not is_glossary_line and line_stripped:
                skip_glossary = False
                clean_lines.append(line)
        else:
            clean_lines.append(line)
    if clean_lines:
        translated = '\n'.join(clean_lines).strip()

    # Detect meta-response: LLM acknowledges task instead of translating.
    # Return empty string so the caller can fall back (TTS will use silence or
    # the pipeline guard will replace with a silence chunk).
    _LLAMA_META_PREFIX = (
        "Rozumím", "Jasně,", "Jasně.", "Dobře,", "Dobře ", "OK,", "Samozřejmě",
        "Ano,", "Ano.", "Dobre,", "Dobre.", "Porozumiem",
        "Dostanu", "Získám", "Bude ", "Pošli", "Pošlete", "Mluvte",
        "Prosím, vlož", "Jsem připraven", "Pojďme na to",
        "Toto je preklad", "Tu je preklad", "Nasleduje preklad",
        "Preklad:", "Poznámka:", "Note:", "Výsledok:",
    )
    _LLAMA_META_ANYWHERE = (
        "halucin", "nesprávny preklad", "incorrect translation",
        "skrátená verzia", "lepší preklad", "pre dabing by",
        "tu je opravená verzia", "tu je skrátená",
    )
    _tl = translated.lower()
    if translated.startswith(_LLAMA_META_PREFIX) or any(p in _tl for p in _LLAMA_META_ANYWHERE):
        print(
            f"    [LLAMA-META] meta-response detected — returning empty: '{translated[:80]}'",
            flush=True,
        )
        return ""
    # Príliš dlhý výstup = LLM komentoval namiesto prekladu
    if len(translated) > len(text) * 2.5 + 100:
        print(f"    [LLAMA-META] output too long ({len(translated)} vs {len(text)}) — returning empty", flush=True)
        return ""

    # Strip markdown artifacts (**, *, _) that LLMs sometimes add
    import re as _re
    translated = _re.sub(r'\*{1,3}', '', translated)
    translated = _re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', translated)
    translated = _re.sub(r' {2,}', ' ', translated).strip()

    trans_words = len(translated.split())
    print(f"    Original: {word_count} words, Translation: {trans_words} words", flush=True)
    return translated


# Global MADLAD model cache
_madlad_model = None
_madlad_tokenizer = None
_multislav_model = None
_multislav_tokenizer = None
_multislav_model_name = ""
_multislav_device = ""


def free_madlad_model() -> None:
    """Release MADLAD model from memory (call before loading Gemma/QA)."""
    global _madlad_model, _madlad_tokenizer
    if _madlad_model is not None:
        del _madlad_model
        _madlad_model = None
    if _madlad_tokenizer is not None:
        del _madlad_tokenizer
        _madlad_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[MADLAD] Model released from memory", flush=True)


def free_multislav5lang_model() -> None:
    """Release MultiSlav model from memory."""
    global _multislav_model, _multislav_tokenizer, _multislav_model_name, _multislav_device
    had_any = (
        _multislav_model is not None
        or _multislav_tokenizer is not None
        or bool(_multislav_model_name)
        or bool(_multislav_device)
    )
    if _multislav_model is not None:
        del _multislav_model
        _multislav_model = None
    if _multislav_tokenizer is not None:
        del _multislav_tokenizer
        _multislav_tokenizer = None
    _multislav_model_name = ""
    _multislav_device = ""
    gc.collect()
    torch.cuda.empty_cache()
    if had_any:
        print("[MULTISLAV] Model released from memory", flush=True)


def translate_segment_with_madlad(
    text: str,
    tgt_lang: str,
    model_name: str = "google/madlad400-3b-mt",
    glossary: str = None,
    prev_translation: str = None,
    device: str = None
) -> str:
    global _madlad_model, _madlad_tokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _madlad_model is None:
        print(f"[MADLAD] Loading {model_name} on {device}...", flush=True)
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
        except ImportError:
            from transformers import AutoTokenizer as T5Tokenizer, AutoModelForSeq2SeqLM as T5ForConditionalGeneration
        _madlad_tokenizer = T5Tokenizer.from_pretrained(model_name)
        _madlad_model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            _madlad_model = _madlad_model.to(device)
        print("[MADLAD] Model loaded", flush=True)
    lang_map = {
        "ces": "cs", "cz": "cs", "cs": "cs",
        "slk": "sk", "sk": "sk",
        "deu": "de", "de": "de",
        "fra": "fr", "fr": "fr",
        "eng": "en", "en": "en",
        "spa": "es", "es": "es",
        "ita": "it", "it": "it",
        "rus": "ru", "ru": "ru",
        "pol": "pl", "pl": "pl",
    }
    tgt_code = lang_map.get(tgt_lang.lower(), tgt_lang.lower()[:2])
    protected_text, token_map = protect_phonetic_terms(text)
    input_text = f"<2{tgt_code}> {protected_text}"
    inputs = _madlad_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _madlad_model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    translated = _madlad_tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated = restore_phonetic_terms(translated, token_map)
    del inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    # Fix MADLAD English caps leakage (e.g. "WHEN", "WHERE", "ELSE" left in all-caps)
    _EN_CAPS_CS = {
        r'\bWHEN\b': 'když', r'\bWHERE\b': 'kde', r'\bELSE\b': 'jinak',
        r'\bWITH\b': 's', r'\bTHEN\b': 'pak', r'\bIF\b': 'pokud',
        r'\bAND\b': 'a', r'\bBUT\b': 'ale', r'\bOR\b': 'nebo',
        r'\bFOR\b': 'pro', r'\bOF\b': 'z', r'\bIN\b': 'v',
        r'\bON\b': 'na', r'\bTO\b': 'na',
    }
    _EN_CAPS_SK = {
        r'\bWHEN\b': 'keď', r'\bWHERE\b': 'kde', r'\bELSE\b': 'inak',
        r'\bWITH\b': 's', r'\bTHEN\b': 'potom', r'\bIF\b': 'ak',
        r'\bAND\b': 'a', r'\bBUT\b': 'ale', r'\bOR\b': 'alebo',
        r'\bFOR\b': 'pre', r'\bOF\b': 'z', r'\bIN\b': 'v',
        r'\bON\b': 'na', r'\bTO\b': 'na',
    }
    caps_map = _EN_CAPS_SK if tgt_code in ('sk',) else _EN_CAPS_CS
    for pattern, replacement in caps_map.items():
        new = re.sub(pattern, replacement, translated)
        if new != translated:
            print(f"    [MADLAD-CAPS] {pattern} → '{replacement}'", flush=True)
            translated = new
    word_count_orig = len(text.split())
    word_count_trans = len(translated.split())
    print(f"    Original: {word_count_orig} words, Translation: {word_count_trans} words", flush=True)
    return translated


def translate_segment_with_multislav5lang(
    text: str,
    tgt_lang: str,
    model_name: str = "allegro/multislav-5lang",
    device: str = None,
) -> str:
    global _multislav_model, _multislav_tokenizer, _multislav_model_name, _multislav_device

    from transformers import AutoTokenizer, MarianMTModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tgt_map = {
        "cs": ">>ces<<",
        "ces": ">>ces<<",
        "cz": ">>ces<<",
        "en": ">>eng<<",
        "eng": ">>eng<<",
        "pl": ">>pol<<",
        "pol": ">>pol<<",
        "sk": ">>slk<<",
        "slk": ">>slk<<",
        "sl": ">>slv<<",
        "slv": ">>slv<<",
    }
    tgt_tag = tgt_map.get((tgt_lang or "").lower())
    if not tgt_tag:
        raise ValueError("MultiSlav-5lang supports only cs/en/pl/sk/sl target languages")

    if (
        _multislav_model is None
        or _multislav_tokenizer is None
        or _multislav_model_name != model_name
        or _multislav_device != device
    ):
        free_multislav5lang_model()
        print(f"[MULTISLAV] Loading {model_name} on {device}...", flush=True)
        _multislav_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _multislav_model = MarianMTModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        _multislav_model.eval()
        _multislav_model_name = model_name
        _multislav_device = device
        print("[MULTISLAV] Model loaded", flush=True)

    protected_text, token_map = protect_phonetic_terms(text)
    inputs = _multislav_tokenizer(
        [f"{tgt_tag} {protected_text}"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _multislav_model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    translated = _multislav_tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    translated = restore_phonetic_terms(translated, token_map)
    del inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    word_count_orig = len(text.split())
    word_count_trans = len(translated.split())
    print(f"    Original: {word_count_orig} words, Translation: {word_count_trans} words", flush=True)
    return translated


# ---------------------------------------------------------------------------
# Google Translate (free mobile endpoint — inspired by pyvideotrans/_google.py)
# ---------------------------------------------------------------------------

_GOOGLE_LANG_MAP = {
    "ces": "cs", "cs": "cs",
    "slk": "sk", "sk": "sk",
    "deu": "de", "de": "de",
    "fra": "fr", "fr": "fr",
    "eng": "en", "en": "en",
    "spa": "es", "es": "es",
    "ita": "it", "it": "it",
    "rus": "ru", "ru": "ru",
    "pol": "pl", "pl": "pl",
    "hun": "hu", "hu": "hu",
    "ron": "ro", "ro": "ro",
    "ukr": "uk", "uk": "uk",
}

_GOOGLE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    )
}


def translate_full_document_with_translategemma(
    segments: list,
    tgt_lang: str,
    model_path: str,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    context_hint: str = "",
) -> list:
    """
    Preloží celý dokument naraz (nie segment po segmente).
    Vloží markéry §N§ pred každý segment, preloží celý text,
    potom rozdelí výstup späť podľa markérov.
    Výhoda: model vidí celý kontext → správny rod, konzistentná terminológia.
    """
    from translategemma import get_translategemma, free_translategemma

    # Zostaviť celý text s markérmi
    lines = []
    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"§{i}§ {text}")

    full_text = "\n".join(lines)

    # TranslateGemma je čistý prekladový model — neprijíma inštrukcie.
    # Pošleme iba surový text s §N§ markérmi. Model ich zachová (§ nie je slovné znaky).
    # context_hint ignorujeme — kontext poskytuje samotný celý dokument.
    if context_hint:
        print(f"[FULLDOC] context_hint ignored (TranslateGemma is a pure translation model)", flush=True)

    # Odhadni počet tokenov vstupu (1 token ≈ 3 znaky)
    estimated_input_tokens = max(256, int(len(full_text) / 3))
    # Bezpečná hranica: 50 % kontextu pre vstup, 50 % pre výstup
    max_input_tokens = int(n_ctx * 0.5)

    print(f"[FULLDOC] Loading TranslateGemma (n_ctx={n_ctx})...", flush=True)
    tg = get_translategemma(model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)

    def _translate_chunk(chunk_text: str) -> str:
        out_tokens = max(512, int(len(chunk_text) / 2.5))
        return tg.translate(chunk_text, src="en", tgt=tgt_lang, max_tokens=out_tokens)

    if estimated_input_tokens <= max_input_tokens:
        # Celý dokument sa zmestí — preložíme naraz
        out_tokens = max(2048, int(len(full_text) / 2.5))
        print(f"[FULLDOC] Input chars: {len(full_text)} (~{estimated_input_tokens} tok), max_tokens={out_tokens}", flush=True)
        result_text = tg.translate(full_text, src="en", tgt=tgt_lang, max_tokens=out_tokens)
    else:
        # Dokument je príliš dlhý — dávkový preklad po blokoch
        batch_chars = max(1000, max_input_tokens * 3)
        print(f"[FULLDOC] Document too large (~{estimated_input_tokens} tok > {max_input_tokens}), batching (batch≈{batch_chars} chars)", flush=True)
        current_lines: list[str] = []
        current_chars = 0
        batch_results: list[str] = []
        for line in lines:
            if current_chars + len(line) > batch_chars and current_lines:
                bt = "\n".join(current_lines)
                print(f"[FULLDOC] Batch {len(current_lines)} segs ({len(bt)} chars)", flush=True)
                batch_results.append(_translate_chunk(bt))
                current_lines = []
                current_chars = 0
            current_lines.append(line)
            current_chars += len(line) + 1
        if current_lines:
            bt = "\n".join(current_lines)
            print(f"[FULLDOC] Batch {len(current_lines)} segs ({len(bt)} chars)", flush=True)
            batch_results.append(_translate_chunk(bt))
        result_text = "\n".join(batch_results)

    free_translategemma()
    gc.collect()

    # Parsovanie výstupu — hľadáme §N§ markéry
    import re as _re

    def _strip_editorial(text: str) -> str:
        """Odstráni redakčné komentáre ktoré translategemma pridáva k neúplným vetám."""
        # Markdown artefakty: **bold**, *italic*, bullet points (* text, - text)
        text = _re.sub(r'\*{1,3}', '', text)
        text = _re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        text = _re.sub(r'^\s*[-•]\s+', '', text, flags=_re.MULTILINE)  # bullet points
        text = _re.sub(r'^\s*#+\s+', '', text, flags=_re.MULTILINE)    # headings
        # Newlines → space (FULLDOC often returns multi-line for single segment)
        text = _re.sub(r'\n+', ' ', text)
        text = _re.sub(r' {2,}', ' ', text)
        # (tu chýba...), (zvyšok vety chýba), (here something is missing), atď.
        text = _re.sub(r'\s*\([^)]{0,80}chýba[^)]*\)', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\s*\([^)]{0,80}missing[^)]*\)', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\s*\([^)]{0,80}neúplný[^)]*\)', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\s*\(pozn\.[^)]*\)', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\s*\(pozn\. prekladateľa[^)]*\)', '', text, flags=_re.IGNORECASE)
        # Trailing "ako ste navrhli", "podľa kontextu" — voľné LLM doplnky
        text = _re.sub(r',?\s*ako ste navrhli\.?$', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r',?\s*podľa kontextu\.?$', '', text, flags=_re.IGNORECASE)
        # Trailing "..." — translategemma pridáva keď veta je neukončená
        text = _re.sub(r'\s*\.\.\.\s*$', '', text)
        return text.strip()

    translation_map: dict[int, str] = {}
    for match in _re.finditer(r'§(\d+)§\s*(.*?)(?=§\d+§|$)', result_text, _re.DOTALL):
        idx = int(match.group(1))
        translated = _strip_editorial(match.group(2).strip())
        if translated:
            translation_map[idx] = translated

    print(f"[FULLDOC] Parsed {len(translation_map)}/{len(segments)} segments from full-doc translation.", flush=True)

    # Zostaviť výsledok — ak markér chýba, ponechaj pôvodný text
    result = []
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        if i in translation_map:
            seg_copy["text"] = translation_map[i]
            print(f"[FULLDOC] [{i}] {seg_copy['text'][:60]}", flush=True)
        else:
            print(f"[FULLDOC] [{i}] MISSING — keeping original", flush=True)
        result.append(seg_copy)
    return result


def translate_fulldoc_with_lmstudio(
    segments: list,
    tgt_lang: str,
    lmstudio_url: str = "http://localhost:1234/v1",
    lmstudio_model: str = "",
    lmstudio_timeout: int = 300,
    content_type: str = "general",
    src_lang: str = "en",
    batch_size: int = 50,
    speaker_gender: str = "",  # "feminine" / "masculine" / "" (default = masculine)
) -> list:
    """Preloží dokument cez LM Studio API v dávkach po batch_size segmentov.

    Dávky zaručujú že model nepreťaží kontext a zachová 1:1 mapovanie segmentov.
    Fallback na sliding window ak dávka zlyhá alebo má < 50% zhodu.
    """
    import sys
    from pathlib import Path as _P
    _scripts_dir = str(_P(__file__).parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from sk_glossary import get_llm_prompt_block

    lang_name = target_lang_name(tgt_lang)
    content_hint = relax_translation_hint_for_rewrite(get_content_type_instruction(content_type))
    glossary_block = get_llm_prompt_block(tgt_lang) if tgt_lang in ("sk", "slk") else ""

    # Auto-detect model ID
    _effective_model = lmstudio_model
    if not _effective_model:
        try:
            import requests as _req_m
            _mresp = _req_m.get(
                (lmstudio_url or "http://localhost:1234/v1").rstrip("/") + "/models",
                timeout=5,
            ).json()
            _llm_models = [
                m["id"] for m in _mresp.get("data", [])
                if "embed" not in m.get("id", "").lower()
            ]
            if _llm_models:
                _effective_model = _llm_models[0]
                print(f"[LMSTUDIO-FULLDOC] Auto model: {_effective_model}", flush=True)
        except Exception:
            _effective_model = "local-model"

    def _ts(sec: float) -> str:
        h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    system_prompt = (
        f"You are a professional subtitle translator specialized in {lang_name} dubbing. "
        "Translate all content accurately and completely. "
        "Never refuse, censor, summarize, or omit any part of the source text. "
        "CRITICAL: Output EXACTLY the same number of subtitle blocks as the input. "
        "Do NOT merge, split, or skip any subtitle. Each input block = one output block. "
        "Preserve subtitle numbering and timestamps EXACTLY unchanged. "
        "Output valid SRT format only — no notes, no markdown, no explanations."
    )

    rules_base = [
        f"Translate the following subtitles from {src_lang.upper()} to {lang_name}.",
        "Rules:",
        "- Copy subtitle numbers and timestamps EXACTLY — do not change them.",
        "- Translate EACH subtitle block separately — do NOT combine or split blocks.",
        "- Output EXACTLY the same number of blocks as the input.",
        "- Translate into natural, fluent spoken Slovak — prioritize meaning and natural phrasing.",
        "- Use correct Slovak grammar: proper declension (pád, rod, číslo) and verb conjugation.",
        "- Keep foreign proper nouns and code identifiers as-is (C++, DirectX, SDL, Sea of Thieves).",
        # Game-dev / programming pojmy — POVINNE zachovaj v EN bez deklinácií, kvôli TTS výslovnosti.
        # Napr. 'physics engine' NIE 'fyzikálny motor', 'shader' NIE 'tieňovač', 'voxel engine' NIE 'voxelový motor'.
        "- ALWAYS keep these tech terms in English (no Slovak translation, no inflection):",
        "  engine, shader, voxel, mesh, vertex, fragment, framework, pipeline, chunk,",
        "  rigid body, collider, OpenGL, Vulkan, DirectX, GPU, CPU, RAM, VRAM, API, SDK, NDA.",
        "- Pre 'physics engine' použi 'fyzikálny engine', NIE 'fyzikálny motor'.",
        "- Pre 'graphics engine' použi 'grafický engine', NIE 'grafický motor'.",
        "- Pre 'game engine' použi 'herný engine', NIE 'herný motor'.",
        "- Slovo 'engine' sa NIKDY neprekladá ako 'motor' v programátorskom kontexte.",
    ]
    # Gender hint pre 1. osobu — keď user vybral ženský voice design, použijeme ženské tvary
    # slovies a prídavných mien v 1. osobe singuláru (napr. "Začala som" namiesto "Začal som").
    if speaker_gender == "feminine":
        rules_base.append(
            "- DÔLEŽITÉ: Hovoriaca osoba je ŽENA. Pre 1. osobu jednotného čísla používaj "
            "ŽENSKÉ tvary slovies a prídavných mien v minulom čase: "
            "'Začala som' (NIE 'Začal som'), 'Naučila som sa' (NIE 'Naučil som sa'), "
            "'bola som rada' (NIE 'bol som rád'), 'urobila som' (NIE 'urobil som'). "
            "Toto platí pre celé video — všetky 1. os. sg. minulého času."
        )
        system_prompt += " Speaker is FEMALE — use feminine forms for 1st person singular."
    elif speaker_gender == "masculine":
        # Default už je masculine, ale explicit aby model nezmenil
        rules_base.append(
            "- Hovoriaca osoba je MUŽ. Pre 1. osobu jednotného čísla používaj mužské tvary: "
            "'Začal som', 'Naučil som sa', 'bol som rád'."
        )

    results = [dict(s) for s in segments]
    total_matched = 0

    # Rozdelíme na dávky
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    print(f"[LMSTUDIO-FULLDOC] {len(segments)} segmentov v {len(batches)} dávkach po {batch_size}", flush=True)

    for batch_idx, batch in enumerate(batches):
        offset = batch_idx * batch_size
        # Číslujeme segmenty 1-based v rámci dávky
        srt_lines = []
        for local_i, seg in enumerate(batch):
            text = (seg.get("text") or "").strip() or "..."
            srt_lines.append(f"{local_i+1}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{text}")
        batch_srt = "\n\n".join(srt_lines)

        # max_tokens cap: dynamicky podľa veľkosti batch (whole-doc môže potrebovať veľa).
        # 4 chars ≈ 1 token, output ~ rovnako dlhý ako vstup; cap 32000 pre LM Studio Gemma 26B
        _est_out_tokens = len(batch_srt) // 3
        max_out = min(32000, max(2048, _est_out_tokens * 2))
        prompt_parts = ["\n".join(rules_base)]
        prompt_parts.append(f"Subtitles to translate ({len(batch)} blocks):\n{batch_srt}")
        user_prompt = "\n\n".join(prompt_parts)

        print(f"[LMSTUDIO-FULLDOC] Dávka {batch_idx+1}/{len(batches)}: seg {offset+1}–{offset+len(batch)}, "
              f"~{len(user_prompt)} znakov, max_tokens={max_out}", flush=True)

        try:
            raw = api_chat_complete(
                api_key="lmstudio",
                base_url=lmstudio_url or "http://localhost:1234/v1",
                model=_effective_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.05,
                max_tokens=max_out,
                timeout_s=lmstudio_timeout,
            )
        except Exception as e:
            print(f"[LMSTUDIO-FULLDOC] Dávka {batch_idx+1} zlyhala: {e}. Použijem sliding window pre túto dávku.", flush=True)
            partial = translate_segments_sliding_window(
                segments=batch, tgt_lang=tgt_lang, engine="lmstudio",
                lmstudio_url=lmstudio_url, lmstudio_model=lmstudio_model,
                lmstudio_timeout=lmstudio_timeout, content_type=content_type,
                glossary=glossary_block, batch_size=20, src_lang=src_lang,
            )
            for local_i, seg in enumerate(partial):
                results[offset + local_i] = seg
            continue

        # Parsuj SRT odpoveď — lokálne indexy 1..len(batch)
        blocks = re.split(r'\n\s*\n', raw.strip())
        batch_matched = 0
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines_b = block.splitlines()
            if not lines_b:
                continue
            try:
                local_idx = int(lines_b[0].strip()) - 1
            except ValueError:
                continue
            if local_idx < 0 or local_idx >= len(batch):
                continue
            text_lines = lines_b[2:] if len(lines_b) > 2 else []
            text = " ".join(t.strip() for t in text_lines if t.strip())
            if text and text != "...":
                global_idx = offset + local_idx
                results[global_idx]["text"] = text
                results[global_idx]["text_src"] = (segments[global_idx].get("text") or "").strip()
                batch_matched += 1

        print(f"[LMSTUDIO-FULLDOC] Dávka {batch_idx+1}: preložených {batch_matched}/{len(batch)}", flush=True)

        if batch_matched < len(batch) * 0.5:
            print(f"[LMSTUDIO-FULLDOC] Dávka {batch_idx+1} má príliš málo zhôd → sliding window.", flush=True)
            partial = translate_segments_sliding_window(
                segments=batch, tgt_lang=tgt_lang, engine="lmstudio",
                lmstudio_url=lmstudio_url, lmstudio_model=lmstudio_model,
                lmstudio_timeout=lmstudio_timeout, content_type=content_type,
                glossary=glossary_block, batch_size=20, src_lang=src_lang,
            )
            for local_i, seg in enumerate(partial):
                results[offset + local_i] = seg
            batch_matched = len(batch)

        total_matched += batch_matched

    print(f"[LMSTUDIO-FULLDOC] Celkom preložených {total_matched}/{len(segments)} segmentov.", flush=True)

    # Post-translate fix: LM Studio občas prekladá "engine" → "motor" v tech kontexte aj
    # napriek glossary pravidlu. Deterministický cleanup zachytí tieto prípady.
    # POZN.: results[i]["text"] obsahuje SK preklad (prepísaný v batch loop). Source EN text
    # je v segments[i]["text"] (input parameter). Cleanup potrebuje OBOJE pre context-aware fix.
    _fixed_engine = 0
    for _i, _seg in enumerate(results):
        _txt = _seg.get("text", "")
        if not _txt:
            continue
        _src_en = (segments[_i].get("text") or "") if _i < len(segments) else ""
        _new = _post_translate_fix_tech_terms(_txt, src_text=_src_en)
        if _new != _txt:
            _seg["text"] = _new
            _fixed_engine += 1
    if _fixed_engine:
        print(f"[LMSTUDIO-FULLDOC] Tech-term cleanup: {_fixed_engine} segmentov opravených (engine/shader/voxel).", flush=True)
    return results


_TECH_TERM_FIXUPS = [
    # (regex, replacement) — case-insensitive, slovo-hraničné
    # "fyzikálny motor" / "fyzikálne motory" / "fyzikálnych motorov" → engine
    (re.compile(r"\bfyzikáln(y|eho|emu|om|ým|e|é|ych|ymi|a|ou)\s+motor(y|ov|om|och|mi|u|a|ovi|e)?\b", re.IGNORECASE),
     lambda m: f"fyzikáln{m.group(1)} engine"),
    (re.compile(r"\bgrafick(ý|ého|ému|om|ým|é|ej|ých|ymi|a|ou)\s+motor(y|ov|om|och|mi|u|a|ovi|e)?\b", re.IGNORECASE),
     lambda m: f"grafick{m.group(1)} engine"),
    (re.compile(r"\bhern(ý|ého|ému|om|ým|é|ej|ých|ymi|a|ou)\s+motor(y|ov|om|och|mi|u|a|ovi|e)?\b", re.IGNORECASE),
     lambda m: f"hern{m.group(1)} engine"),
    (re.compile(r"\bvoxelov(ý|ého|ému|om|ým|é|ej|ých|ymi|a|ou)\s+motor(y|ov|om|och|mi|u|a|ovi|e)?\b", re.IGNORECASE),
     lambda m: f"voxelov{m.group(1)} engine"),
    # "tieňovač" → shader (rare LLM mistake)
    (re.compile(r"\btieňovač(e|a|ov|om|och|mi|u)?\b", re.IGNORECASE), "shader"),
]

# Context-aware: ak EN text obsahuje "engine" a SK má samostatne stojaci "motor"
# (nielen "fyzikálny motor" atď.), nahradíme — v tech kontexte je "engine" vždy správne.
_STANDALONE_MOTOR = re.compile(
    r"\bmotor(y|ov|om|och|mi|u|a|ovi|e)?\b",
    re.IGNORECASE,
)


def _post_translate_fix_tech_terms(text: str, src_text: str = "") -> str:
    """Deterministický cleanup tech termov po LLM translate. Zachytáva idiomatické chyby
    ako 'fyzikálny motor' (engine), 'tieňovač' (shader) — kde glossary v prompt-e
    nezabralo. Spustené po každom batch-i LM Studio translation.

    Args:
        text: SK preklad
        src_text: pôvodný EN text (pre context-aware "motor"→"engine" rozhodnutie)
    """
    if not text:
        return text
    out = text
    for _re_pat, _repl in _TECH_TERM_FIXUPS:
        out = _re_pat.sub(_repl, out)
    # Context-aware standalone "motor" → "engine" iba ak EN má "engine"
    src_lower = (src_text or "").lower()
    if "engine" in src_lower:
        # Nahradíme všetky tvary "motor" (motor/motory/motorov/motore...) za "engine"
        # Pozn.: zámerne strácame slovenskú deklináciu — "engine" je foreign noun bez deklinácie
        out = _STANDALONE_MOTOR.sub("engine", out)
    return out


def _merge_fragments_to_sentences(segments: list) -> list:
    """Spojí segmenty ktoré tvoria neúplnú vetu (Whisper splits) do "sentence units".
    Vráti list dictov: {"text": "celá veta", "indices": [idx z pôvodných segmentov]}
    Sentence terminator: . ! ? alebo "..." na konci.
    """
    units = []
    current_text = ""
    current_indices = []
    for i, seg in enumerate(segments):
        t = (seg.get("text") or "").strip()
        if not t:
            if current_text:
                units.append({"text": current_text.strip(), "indices": current_indices})
                current_text = ""
                current_indices = []
            units.append({"text": "", "indices": [i]})
            continue
        current_text = (current_text + " " + t).strip() if current_text else t
        current_indices.append(i)
        # Sentence terminator?
        if t.endswith(('.', '!', '?', '."', '!"', '?"', ').', ')!')) or t.endswith('...'):
            units.append({"text": current_text.strip(), "indices": current_indices})
            current_text = ""
            current_indices = []
    if current_text:
        units.append({"text": current_text.strip(), "indices": current_indices})
    return units


def _distribute_sk_to_segments(sk_text: str, en_segments: list) -> list:
    """Rozdelí SK preklad medzi N pôvodných segmentov.
    Stratégia:
    1. Ak počet SK viet (split . ! ?) == počet EN segmentov → 1:1
    2. Inak: proporcionálne podľa DURATION (nie EN length) v slovách
    """
    import re as _re
    if not sk_text or not en_segments:
        return ["" for _ in en_segments]
    if len(en_segments) == 1:
        return [sk_text]

    # 1) Rozdelenie SK na vety
    sk_sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', sk_text) if s.strip()]
    if len(sk_sentences) == len(en_segments):
        return sk_sentences

    # 2) Proporcionálne podľa duration
    durations = [max(0.1, float(s.get("end", 0)) - float(s.get("start", 0))) for s in en_segments]
    total_dur = sum(durations)
    sk_words = sk_text.split()
    n_words = len(sk_words)

    result = []
    word_idx = 0
    for i, dur in enumerate(en_segments):
        if i == len(en_segments) - 1:
            result.append(" ".join(sk_words[word_idx:]))
        else:
            ratio = durations[i] / total_dur
            target_words = max(1, round(n_words * ratio))
            result.append(" ".join(sk_words[word_idx:word_idx + target_words]))
            word_idx += target_words
    return result


def translate_fulldoc_with_ollama_translator(
    segments: list,
    tgt_lang: str,
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "26b-translator",
    timeout: int = 120,
    src_lang: str = "en",
) -> list:
    """Preklad cez fine-tuned 26B translator v ollama.

    Používa raw=True API s presným chat template z trénovania.
    PRE-PROCESS: spojí segmenty do plných viet (Whisper často rozdelí v polovici).
    POST-PROCESS: SK preklad rozdelí späť na pôvodné segmenty proporcionálne.
    """
    SYSTEM = (
        "Si profesionálny prekladateľ titulkov. Prekladáš z angličtiny do slovenčiny. "
        "Dodržuj správne slovenské skloňovanie, slovosled a prirodzený hovorový štýl. "
        "Zachovaj mená hier a značiek v originále. Neprekladaj technické pojmy doslovne."
    )

    results = [dict(s) for s in segments]
    for r in results:
        r["text_src"] = (r.get("text") or "").strip()
        r["text"] = ""

    # Pre-process: spoji fragmenty do sentence units
    units = _merge_fragments_to_sentences(segments)
    n_units = len([u for u in units if u["text"]])
    print(f"[OLLAMA-TRANSLATOR] {len(segments)} segmentov → {n_units} viet, model={model}", flush=True)

    matched = 0
    for unit_idx, unit in enumerate(units):
        en_text = unit["text"].strip()
        indices = unit["indices"]

        if not en_text:
            for i in indices:
                results[i]["text"] = ""
            continue

        raw_prompt = (
            f"<bos><start_of_turn>user\n"
            f"{SYSTEM}\n\nPrelož do slovenčiny:\n{en_text}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

        try:
            r = requests.post(ollama_url, json={
                "model": model,
                "prompt": raw_prompt,
                "raw": True,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max(200, int(len(en_text) * 1.8)),
                    "stop": ["<end_of_turn>", "<eos>"],
                },
            }, timeout=timeout)
            r.raise_for_status()
            sk_text = r.json().get("response", "").strip()
            # Cleanup template fragments
            for tok in ["<end_of_turn>", "<end_of_", "<end_of", "<eos>", "<bos>"]:
                sk_text = sk_text.replace(tok, "")
            sk_text = sk_text.strip().rstrip("<").strip()

            if sk_text:
                # Distribute SK preklad na pôvodné segmenty
                en_segs_in_unit = [segments[i] for i in indices]
                sk_parts = _distribute_sk_to_segments(sk_text, en_segs_in_unit)
                for local_i, idx in enumerate(indices):
                    results[idx]["text"] = sk_parts[local_i].strip()
                matched += len(indices)
                if (unit_idx + 1) % 5 == 0 or unit_idx == len(units) - 1:
                    print(f"[OLLAMA-TRANSLATOR] unit {unit_idx+1}/{len(units)}: {sk_text[:80]}", flush=True)
            else:
                # Fallback: zachovaj EN v segmentoch
                for idx in indices:
                    results[idx]["text"] = (segments[idx].get("text") or "").strip()
        except Exception as e:
            print(f"[OLLAMA-TRANSLATOR] unit {unit_idx} chyba: {e}", flush=True)
            for idx in indices:
                results[idx]["text"] = (segments[idx].get("text") or "").strip()

    print(f"[OLLAMA-TRANSLATOR] Preložených {matched}/{len(segments)} segmentov ({n_units} viet).", flush=True)
    return results


def translate_fulldoc_with_gemma4_hf(
    segments: list,
    tgt_lang: str,
    model_id: str = str(PATHS.gemma4_e4b_hf),
    max_new_tokens: int = 200,
    fish_env_python: str = str(PATHS.python_fish),
    extra_context: str | None = None,
) -> list:
    """
    Preloží segmenty pomocou Gemma 4 E4B cez subprocess v fish_env.
    chatterbox_env má transformers 4.46 (nepodporuje gemma4); fish_env má 5.x.
    Worker skript: scripts/gemma4_hf_worker.py
    """
    import json as _json
    import subprocess
    import tempfile

    worker = Path(__file__).parent / "gemma4_hf_worker.py"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f_in:
        _json.dump(segments, f_in, ensure_ascii=False)
        segments_path = f_in.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_out:
        output_path = f_out.name

    try:
        print(f"[GEMMA4-HF] Spúšťam worker cez fish_env: {fish_env_python}", flush=True)
        _default_ctx = Path(__file__).parent / "domain_context.json"
        _ctx_path = extra_context or (str(_default_ctx) if _default_ctx.exists() else None)
        cmd = [
            fish_env_python, str(worker),
            "--segments_json", segments_path,
            "--model_id", model_id,
            "--output_json", output_path,
            "--max_new_tokens", str(max_new_tokens),
        ]
        if _ctx_path:
            cmd += ["--extra_context", _ctx_path]
        proc = subprocess.run(cmd, text=True, capture_output=False)
        if proc.returncode != 0:
            raise RuntimeError(f"gemma4_hf_worker zlyhal (returncode={proc.returncode})")
        result = _json.loads(Path(output_path).read_text(encoding="utf-8"))
        print(f"[GEMMA4-HF] Preložených segmentov: {len(result)}", flush=True)
        return result
    finally:
        Path(segments_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def refine_translation_with_llama(
    segments: list,
    tgt_lang: str,
    llama_model: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = 20,
    context_hint: str = "",
) -> list:
    """Second-pass refinement: fix grammar, declension and naturalness using LLM."""
    from llama_cpp import Llama
    lang_name = target_lang_name(tgt_lang)
    context_line = f" Kontext: {context_hint}." if context_hint else ""
    system = (
        f"Si natívny hovorca jazyka {lang_name}.{context_line} "
        f"Dostaneš preložený text. Oprav skloňovanie, časovanie, rod a prirodzenosť viet. "
        f"Pri cudzích vlastných menách použi prirodzené slovenské skloňovanie podľa syntaktickej roly. "
        f"Zachovaj presný zmysel a podobnú dĺžku. Vráť IBA opravený text bez akýchkoľvek vysvetlení."
    )
    print(f"[REFINE] Loading model: {llama_model}", flush=True)
    llm = Llama(model_path=llama_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    refined = []
    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text:
            refined.append(seg)
            continue
        print(f"[REFINE] {i+1}/{len(segments)}: {text[:60]}", flush=True)
        result = _safe_llama_chat_content(
            llm,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=0.05,
            max_tokens=256,
            label="REFINE",
        )
        if not result or len(result) > len(text) * 2.5:
            refined.append(seg)
            continue
        print(f"[REFINE]  → {result[:60]}", flush=True)
        seg_copy = dict(seg)
        seg_copy["text"] = result
        refined.append(seg_copy)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("[REFINE] Done, model released.", flush=True)
    return refined


_SK_DECLENSION_RULES = """\
SLOVENSKÉ PREDLOŽKOVÉ VÄZBY (musíš dodržať):
• LOKÁL (6. pád, kde?):  v/vo, na, po, pri, o, nad(spoč.), pod(spoč.), za(spoč.), medzi(spoč.)
  Príklady: v systéme, na serveri, po reštartovaní, pri okne, o nastavení
• GENITÍV (2. pád, koho?/čoho?): do/zo, z, od, bez, okolo, pre, kvôli, namiesto, popri, mimo
  Príklady: do servera, z priečinka, od používateľa, bez hesla, pre správcu
• DATÍV (3. pád, komu?/čomu?): k/ku, oproti, naproti, vďaka, napriek
  Príklady: k nastaveniu, ku klientovi, vďaka firewallu
• AKUZATÍV (4. pád, pohyb kam?): cez, za, pod, nad, medzi, na, v (pohyb)
  Príklady: cez sieť, za bránu, na server (ísť), do systému
• INŠTRUMENTÁL (7. pád, s kým?/čím?): s/so, za(v zmysle 'ako'), medzi(spoč.), pred(spoč.)
  Príklady: so serverom, s konfiguráciou, pred spustením
POZOR: "v systém" je CHYBA → správne "v systéme"; "do server" je CHYBA → "do servera"\
"""

_SK_REPAIR_GLOSSARY = """Terminology:
- power supply = napájací zdroj
- no subscription repository = repozitár bez predplatného
- timecode / time code = časová značka
- hostname = hostname
- flash / flashing = nahranie obrazu
- build = zostava
- ISNULL = funkcia na nahradenie NULL hodnoty
- IS NULL = podmienka na kontrolu NULL, vracia TRUE/FALSE
- NULLIF = funkcia, ktorá pri zhode vracia NULL
- COALESCE = funkcia, ktorá vracia prvú nenulovú hodnotu"""

_SK_FOREIGN_NAME_HINTS = """\
- Foreign proper names and character names must use natural Slovak declension when syntax requires it.
- Examples: Kratos -> Kratosa, Freddy -> Freddyho, Jason -> Jasona.
- If the current Slovak sentence has awkward word order around a proper name, rebuild it from SOURCE_EN instead of keeping wrong declension."""

_SK_REPAIR_PROMPT_LIGHT = """\
You are a Slovak dubbing editor preparing subtitles for TTS voice synthesis.

Task: lightly repair the Slovak subtitle segment for natural spoken delivery.

Inputs:
SOURCE_EN: {src}
CURRENT_SK: {text}
SLOT_SECONDS: {slot:.1f}
TARGET_CHARS: {target_chars}
CURRENT_CHARS: {current_chars}

Instructions:
- Improve fluency and naturalness for spoken delivery (TTS will read this aloud).
- Fix grammar, declension, conjugation, and awkward word order.
- KRITICKÉ: Skontroluj a oprav predložkové väzby podľa tabuľky nižšie.
- Fix Slovak declension of foreign proper names when needed.
- Remove weird quotation marks around normal words.
- Fix wrong terminology according to the glossary below.
- Respect SQL semantics exactly: ISNULL is a function, IS NULL is a condition, NULLIF is a different function.
- Prefer colloquial spoken Slovak over formal written style.
- Keep meaning faithful to SOURCE_EN.
- {length_hint}
- Output ONLY the repaired Slovak segment, nothing else. No explanations.

{name_hints}

{declension_rules}

{glossary}"""

_SK_REPAIR_PROMPT_AGGRESSIVE = """\
You are a Slovak dubbing editor preparing subtitles for TTS voice synthesis.

Task: aggressively repair or fully rewrite the Slovak subtitle segment for natural spoken delivery.

Inputs:
SOURCE_EN: {src}
CURRENT_SK: {text}
SLOT_SECONDS: {slot:.1f}
TARGET_CHARS: {target_chars}
CURRENT_CHARS: {current_chars}
AUDIT_FLAGS: {flags}

Instructions:
- SOURCE_EN is the ground truth meaning.
- CURRENT_SK may be mistranslated, fragmented, too literal, or hallucinated.
- If CURRENT_SK is unreliable, IGNORE it and rebuild the sentence from SOURCE_EN.
- You MAY fully restructure the sentence.
- Use natural spoken colloquial Slovak — not a word-for-word translation from English.
- Remove repetition, broken fragments, weird quotation marks, and unnatural phrasing.
- KRITICKÉ: Skontroluj a oprav predložkové väzby podľa tabuľky nižšie.
- Fix Slovak declension of foreign proper names when needed.
- Fix wrong terminology according to the glossary below.
- Respect SQL semantics exactly: ISNULL is a function, IS NULL is a condition, NULLIF is a different function.
- {length_hint}
- Do NOT output notes, explanations, or alternatives.
- Output EXACTLY one repaired Slovak segment, nothing else.

Dubbing constraints:
- natural spoken colloquial Slovak, no literal calques from English
- no repeated or unfinished fragments
- no explanatory additions
- no unnatural word order
- suitable for TTS voice synthesis (no symbols, abbreviations, or markdown)

{name_hints}

{glossary}"""

_AGGRESSIVE_FLAGS = {
    "possible_alignment_drift", "possible_hallucination",
    "possibly_untranslated", "bad_term:prize_repository",
    "possible_name_declension",
}


def _safe_llama_chat_content(
    llm,
    *,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    label: str,
) -> str | None:
    """Best-effort llama.cpp chat wrapper.

    Some local llama_cpp builds throw runtime AttributeError inside chat generation
    (for example missing `sampler`). We never want that to break the whole pipeline.
    """
    try:
        resp = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        print(f"[{label}] llama.cpp chat failed: {e}", flush=True)
        return None


_TARGET_CPS_SK = 13.0  # priemerná SK rečová hustota znakov/s


def _slot_length_hint(text: str, slot: float) -> tuple[str, int, int]:
    """
    Vráti (hint_str, target_chars, current_chars) pre slot-aware prompt.
    - Ak je text príliš krátky pre slot → inštrukcia expandovať
    - Ak je text príliš dlhý → inštrukcia skrátiť
    - Ak je v pohode → inštrukcia zachovať dĺžku
    """
    current_chars = len(text.strip())
    target_chars = max(10, int(slot * _TARGET_CPS_SK))
    ratio = current_chars / target_chars if target_chars > 0 else 1.0

    if ratio < 0.6:
        # Príliš krátky — expandovať
        hint = (
            f"The text is TOO SHORT for the slot ({current_chars} chars, target ~{target_chars}). "
            f"EXPAND naturally to ~{target_chars} chars using SOURCE_EN content. "
            f"Add natural spoken elaboration, context, or restate clearly — stay faithful to SOURCE_EN meaning."
        )
    elif ratio > 1.3:
        # Príliš dlhý — skrátiť
        hint = (
            f"The text is TOO LONG for the slot ({current_chars} chars, target ~{target_chars}). "
            f"Shorten to ~{target_chars} chars — prefer shorter, cleaner sentence over literal translation."
        )
    else:
        hint = f"Text length is appropriate (~{target_chars} chars target). Keep similar length."

    return hint, target_chars, current_chars


def expand_text_single_ollama(
    text: str,
    src_text: str,
    slot_dur: float,
    content_type: str = "general",
    ollama_url: str = "http://localhost:11434/api/generate",
    ollama_model: str = "gemma4:26b",
    timeout: int = 45,
) -> str:
    """
    Expanduje SK text pre jeden segment pomocou ollama (slot-aware).
    Vráti rozšírený text, alebo pôvodný text ak expand zlyhal / nebol lepší.
    Určené pre Duration Planner REWRITE_EXPAND akciu.
    """
    import requests as _req

    _hint, target_chars, current_chars = _slot_length_hint(text, slot_dur)

    prompt = (
        f"You are a Slovak dubbing editor. Expand the Slovak dubbed text to fill a {slot_dur:.1f}s audio slot.\n"
        f"SOURCE_EN: {src_text}\n"
        f"CURRENT_SK: {text}\n"
        f"Target: ~{target_chars} characters (current: {current_chars} chars).\n"
        f"Rules:\n"
        f"- Expand naturally using SOURCE_EN meaning — add context, restate clearly, or elaborate\n"
        f"- Keep proper Slovak grammar and natural spoken style\n"
        f"- Do NOT add empty filler words ('teda', 'vlastne', 'takže' alone)\n"
        f"- Output ONLY the expanded Slovak text, nothing else\n"
    )

    try:
        _resp = _req.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if _resp.ok:
            _result = _resp.json().get("response", "").strip()
            # Sanity check: must be longer than original, not absurdly longer than target
            if _result and len(_result) >= len(text) and len(_result) <= int(target_chars * 1.5):
                return _result
            elif _result and len(_result) > int(target_chars * 1.5):
                print(
                    f"[EXPAND] ollama result too long ({len(_result)} > {int(target_chars * 1.5)}), skipping",
                    flush=True,
                )
    except Exception as _e:
        print(f"[EXPAND] ollama expand failed: {_e}", flush=True)

    return text  # fallback: pôvodný text


def detect_emotion_levels_ollama(
    segments: list[dict],
    ollama_url: str = "http://localhost:11434/api/generate",
    ollama_model: str = "gemma4:26b",
    timeout: int = 60,
) -> list[str]:
    """
    Batch detekcia emocionálnej intenzity segmentov pomocou Ollama.
    Vráti list rovnakej dĺžky ako segments, každý prvok: "low" | "medium" | "high".

    low    → neutrálna inštrukcia, popis                    → exaggeration 0.50
    medium → dôraz, upozornenie, otázka, vysvetlenie        → exaggeration 0.65
    high   → zvolanie, nadšenie, varovanie, prekvapenie     → exaggeration 0.78
    """
    import requests as _req
    import json as _json

    texts = [(seg.get("text") or "").strip() for seg in segments]
    n = len(texts)
    if n == 0:
        return []

    # Batch prompt — číslovanie 1..n
    numbered = "\n".join(f"{idx+1}. {t}" for idx, t in enumerate(texts))
    prompt = (
        "Classify the emotional intensity of each Slovak dubbed sentence.\n"
        "Reply ONLY with a JSON array of strings, one per line, values: low / medium / high.\n"
        "low = neutral instruction or description\n"
        "medium = emphasis, question, explanation, warning\n"
        "high = exclamation, excitement, strong warning, surprise\n\n"
        f"Sentences:\n{numbered}\n\n"
        f"Reply with exactly {n} values as JSON array, e.g. [\"low\",\"medium\",\"high\",...]"
    )

    try:
        _resp = _req.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if _resp.ok:
            raw = _resp.json().get("response", "").strip()
            # Extrahuj JSON array z odpovede
            _start = raw.find("[")
            _end = raw.rfind("]")
            if _start != -1 and _end != -1:
                arr = _json.loads(raw[_start:_end+1])
                if isinstance(arr, list) and len(arr) == n:
                    valid = {"low", "medium", "high"}
                    result = [v if v in valid else "low" for v in arr]
                    return result
    except Exception as _e:
        print(f"[EMOTION] Ollama batch detekcia zlyhala: {_e}", flush=True)

    return ["low"] * n  # fallback: všetky neutrálne


_EMOTION_EXAG: dict[str, float] = {
    "low":    0.50,
    "medium": 0.65,
    "high":   0.78,
}


def fix_grammar_ollama(
    segments: list[dict],
    ollama_url: str = "http://localhost:11434/api/generate",
    ollama_model: str = "llama3.1:8b",
    timeout: int = 30,
    batch_size: int = 10,
) -> list[dict]:
    """Opraví predložkové väzby (skloňovanie) vo všetkých SK segmentoch pomocou Ollama.

    Spracováva segmenty v dávkach. Každá dávka = jeden Ollama request.
    Vracia nový list segmentov s opravenými textami.
    """
    import requests as _req
    import json as _json

    if not segments:
        return segments

    system_prompt = (
        "Si slovenský jazykový korektor. Dostaneš číslované slovenské vety. "
        "Oprav VÝHRADNE chybné predložkové väzby (pády po predložkách). "
        "Nemeň zmysel, neskracuj, nerozvíjaj. "
        "Ak je veta správna, vráť ju nezmenene.\n\n"
        + _SK_DECLENSION_RULES
    )

    repaired = list(segments)
    total = len(segments)
    fixed_count = 0

    for batch_start in range(0, total, batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        texts = [(seg.get("text") or "").strip() for seg in batch]

        # Preskočiť prázdne / príliš krátke segmenty
        if all(len(t) < 4 for t in texts):
            continue

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = (
            f"{system_prompt}\n\n"
            f"Vety:\n{numbered}\n\n"
            f"Odpovedz IBA JSON poľom {len(batch)} opravených reťazcov v rovnakom poradí. "
            f"Príklad: [\"opravená veta 1\",\"opravená veta 2\",...]"
        )

        try:
            _resp = _req.post(
                ollama_url,
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            if not _resp.ok:
                continue
            raw = _resp.json().get("response", "").strip()
            _s = raw.find("[")
            _e = raw.rfind("]")
            if _s == -1 or _e == -1:
                continue
            arr = _json.loads(raw[_s:_e + 1])
            if not isinstance(arr, list) or len(arr) != len(batch):
                continue

            for j, (seg, fixed_text) in enumerate(zip(batch, arr)):
                fixed_text = str(fixed_text).strip()
                orig_text  = texts[j]
                # Sanity: nesmie byť prázdne, nesmie byť drasticky iná dĺžka
                if (fixed_text
                        and fixed_text != orig_text
                        and 0.5 < len(fixed_text) / max(len(orig_text), 1) < 2.0):
                    idx = batch_start + j
                    repaired[idx] = dict(seg)
                    repaired[idx]["text"] = fixed_text
                    fixed_count += 1
                    print(f"[GRAMMAR] Seg {idx+1}: '{orig_text[:50]}' → '{fixed_text[:50]}'", flush=True)

        except Exception as _e:
            print(f"[GRAMMAR] Dávka {batch_start//batch_size + 1} zlyhala: {_e}", flush=True)
            continue

    print(f"[GRAMMAR] Opravených {fixed_count}/{total} segmentov", flush=True)
    return repaired


def shrink_text_single_ollama(
    text: str,
    src_text: str,
    slot_dur: float,
    content_type: str = "general",
    ollama_url: str = "http://localhost:11434/api/generate",
    ollama_model: str = "gemma4:26b",
    timeout: int = 45,
) -> str:
    """
    Skráti SK text pre jeden segment pomocou ollama (slot-aware).
    Vráti skrátený text, alebo pôvodný text ak shrink zlyhal / nebol kratší.
    Určené pre Duration Planner REWRITE_SHRINK akciu pri Chatterbox TTS.
    """
    import requests as _req

    _hint, target_chars, current_chars = _slot_length_hint(text, slot_dur)
    # target_chars môže byť väčší ako current — vypočítame priamo
    from duration_planner import _CPS, _CPS_DEFAULT
    _cps = _CPS.get(content_type, _CPS_DEFAULT)
    target_chars = max(5, int(slot_dur * _cps))

    if current_chars <= target_chars:
        return text  # už sa zmestí, nič nerobiť

    prompt = (
        f"You are a Slovak dubbing editor. Shorten the Slovak dubbed text to fit a {slot_dur:.1f}s audio slot.\n"
        f"SOURCE_EN: {src_text}\n"
        f"CURRENT_SK: {text}\n"
        f"Target: ~{target_chars} characters (current: {current_chars} chars).\n"
        f"Rules:\n"
        f"- Shorten by removing less important details, keeping the core meaning\n"
        f"- Keep technical terms exact (SQL, NULL, IS NULL, ISNULL, COALESCE etc.)\n"
        f"- Keep proper Slovak grammar and natural spoken style\n"
        f"- Output ONLY the shortened Slovak text, nothing else\n"
    )

    try:
        _resp = _req.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if _resp.ok:
            _result = _resp.json().get("response", "").strip()
            # Sanity: must be shorter than original, not too short (>40% of target)
            if _result and len(_result) < len(text) and len(_result) >= int(target_chars * 0.4):
                return _result
            elif _result:
                print(
                    f"[SHRINK] ollama result rejected (len={len(_result)}, target={target_chars}), skipping",
                    flush=True,
                )
    except Exception as _e:
        print(f"[SHRINK] ollama shrink failed: {_e}", flush=True)

    return text  # fallback: pôvodný text


def repair_sk_flagged_with_gemma(
    segments: list,
    audit_results: list,
    llama_model: str,
    n_gpu_layers: int = -1,
    min_score: int = 1,
) -> list:
    """
    Opraví SK segmenty oflagnúté auditom pomocou Gemma LLM.
    - score < min_score → preskočiť
    - aggressive flags alebo score > 5 → agresívny prompt (rebuild from source)
    - inak → light repair prompt
    Vracia nový zoznam segmentov s opravenými textami.
    """
    from llama_cpp import Llama

    # Indexovanie audit výsledkov podľa idx (1-based = pozícia v liste)
    audit_map: dict[int, dict] = {r["idx"]: r for r in audit_results}

    to_repair = [
        (i, seg, audit_map.get(i + 1, {}))
        for i, seg in enumerate(segments)
        if audit_map.get(i + 1, {}).get("score", 0) >= min_score
    ]

    if not to_repair:
        print("[REPAIR] Žiadne segmenty na opravu.", flush=True)
        return segments

    print(f"[REPAIR] Načítavam model: {llama_model}", flush=True)
    # n_ctx=4096 (zvýšené z 1024) — pre dlhé segmenty s prompt+response
    # Pôvodné 1024 padalo "Requested tokens (1038) exceed context window of 1024"
    llm = Llama(model_path=llama_model, n_ctx=4096, n_gpu_layers=n_gpu_layers, verbose=False)

    repaired = list(segments)
    for i, seg, audit in to_repair:
        text = (seg.get("text") or "").strip()
        src  = (seg.get("text_src") or "").strip()
        slot = float(seg.get("end", 0)) - float(seg.get("start", 0))
        flags = set(audit.get("flags", []))
        score = audit.get("score", 0)

        if not text and not src:
            continue

        _length_hint, _target_chars, _current_chars = _slot_length_hint(text or src, slot)
        aggressive = bool(flags & _AGGRESSIVE_FLAGS) or score > 5
        if aggressive:
            prompt = _SK_REPAIR_PROMPT_AGGRESSIVE.format(
                src=src or text, text=text or src,
                slot=slot, flags=", ".join(sorted(flags)),
                target_chars=_target_chars, current_chars=_current_chars,
                length_hint=_length_hint,
                name_hints=_SK_FOREIGN_NAME_HINTS,
                declension_rules=_SK_DECLENSION_RULES,
                glossary=_SK_REPAIR_GLOSSARY,
            )
            mode = "AGGRESSIVE"
        else:
            prompt = _SK_REPAIR_PROMPT_LIGHT.format(
                src=src or text, text=text,
                slot=slot,
                target_chars=_target_chars, current_chars=_current_chars,
                length_hint=_length_hint,
                name_hints=_SK_FOREIGN_NAME_HINTS,
                declension_rules=_SK_DECLENSION_RULES,
                glossary=_SK_REPAIR_GLOSSARY,
            )
            mode = "LIGHT"

        print(f"[REPAIR/{mode}] seg {i+1} score={score} flags={sorted(flags)}", flush=True)
        print(f"[REPAIR]  SK : {text[:70]}", flush=True)

        result = _safe_llama_chat_content(
            llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            label=f"REPAIR/{mode}",
        )
        if not result:
            result = source_guided_template(src) or ""
            if result:
                print(f"[REPAIR]  → source-template fallback", flush=True)
            else:
                print(f"[REPAIR]  ⚠ preskočené po chybe llama.cpp", flush=True)
                continue
        result = guard_technical_text(result, source_text=src, fallback_text=text)

        # Guard: ak výsledok je príliš dlhý alebo prázdny → ponechaj pôvodný
        if not result or len(result) > max(len(text or src) * 1.6, 20):
            print(f"[REPAIR]  ⚠ zamietnutý (prázdny alebo príliš dlhý)", flush=True)
            continue

        print(f"[REPAIR]  → {result[:70]}", flush=True)
        seg_copy = dict(seg)
        seg_copy["text"] = result
        if "_audit" in seg_copy:
            seg_copy["_audit"]["repaired"] = True
            seg_copy["_audit"]["repair_mode"] = mode
        repaired[i] = seg_copy

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("[REPAIR] Hotovo, model uvoľnený.", flush=True)
    return repaired


def repair_sk_flagged_with_openai(
    segments: list,
    audit_results: list,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    min_score: int = 1,
) -> list:
    """
    OpenAI verzia repair_sk_flagged — volá GPT-4o / GPT-4o-mini namiesto lokálneho LLM.
    Rovnaká logika: light repair pre nízke skóre, aggressive pre vysoké/flagged.
    """
    import openai

    client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    audit_map: dict[int, dict] = {r["idx"]: r for r in audit_results}
    to_repair = [
        (i, seg, audit_map.get(i + 1, {}))
        for i, seg in enumerate(segments)
        if audit_map.get(i + 1, {}).get("score", 0) >= min_score
    ]

    if not to_repair:
        print("[REPAIR/OAI] Žiadne segmenty na opravu.", flush=True)
        return segments

    repaired = list(segments)
    for i, seg, audit in to_repair:
        text = (seg.get("text") or "").strip()
        src  = (seg.get("text_src") or "").strip()
        slot = float(seg.get("end", 0)) - float(seg.get("start", 0))
        flags = set(audit.get("flags", []))
        score = audit.get("score", 0)

        if not text and not src:
            continue

        _length_hint, _target_chars, _current_chars = _slot_length_hint(text or src, slot)
        aggressive = bool(flags & _AGGRESSIVE_FLAGS) or score > 5
        if aggressive:
            prompt = _SK_REPAIR_PROMPT_AGGRESSIVE.format(
                src=src or text, text=text or src,
                slot=slot, flags=", ".join(sorted(flags)),
                target_chars=_target_chars, current_chars=_current_chars,
                length_hint=_length_hint,
                name_hints=_SK_FOREIGN_NAME_HINTS,
                declension_rules=_SK_DECLENSION_RULES,
                glossary=_SK_REPAIR_GLOSSARY,
            )
            mode = "AGGRESSIVE"
        else:
            prompt = _SK_REPAIR_PROMPT_LIGHT.format(
                src=src or text, text=text,
                slot=slot,
                target_chars=_target_chars, current_chars=_current_chars,
                length_hint=_length_hint,
                name_hints=_SK_FOREIGN_NAME_HINTS,
                declension_rules=_SK_DECLENSION_RULES,
                glossary=_SK_REPAIR_GLOSSARY,
            )
            mode = "LIGHT"

        print(f"[REPAIR/OAI/{mode}] seg {i+1} score={score}", flush=True)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            result = (resp.choices[0].message.content or "").strip()
            result = guard_technical_text(result, source_text=src, fallback_text=text)
        except Exception as e:
            print(f"[REPAIR/OAI] ⚠ chyba API: {e}", flush=True)
            continue

        if not result or len(result) > max(len(text or src) * 1.6, 20):
            print(f"[REPAIR/OAI] ⚠ zamietnutý (prázdny alebo príliš dlhý)", flush=True)
            continue

        print(f"[REPAIR/OAI]  → {result[:70]}", flush=True)
        seg_copy = dict(seg)
        seg_copy["text"] = result
        if "_audit" in seg_copy:
            seg_copy["_audit"]["repaired"] = True
            seg_copy["_audit"]["repair_mode"] = f"OAI_{mode}"
        repaired[i] = seg_copy

    print("[REPAIR/OAI] Hotovo.", flush=True)
    return repaired


def repair_sk_flagged_with_ollama(
    segments: list,
    audit_results: list,
    *,
    url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:26b",
    timeout: int = 120,
    min_score: int = 1,
) -> list:
    """
    Ollama verzia repair_sk_flagged.
    Používa rovnaké prompty ako lokálny Gemma repair, ale beží cez Ollama HTTP API.
    """
    audit_map: dict[int, dict] = {r["idx"]: r for r in audit_results}
    to_repair = [
        (i, seg, audit_map.get(i + 1, {}))
        for i, seg in enumerate(segments)
        if audit_map.get(i + 1, {}).get("score", 0) >= min_score
    ]

    if not to_repair:
        print("[REPAIR/OLLAMA] Žiadne segmenty na opravu.", flush=True)
        return segments

    repaired = list(segments)
    for i, seg, audit in to_repair:
        text = (seg.get("text") or "").strip()
        src = (seg.get("text_src") or "").strip()
        slot = float(seg.get("end", 0)) - float(seg.get("start", 0))
        flags = set(audit.get("flags", []))
        score = audit.get("score", 0)

        if not text and not src:
            continue

        _length_hint, _target_chars, _current_chars = _slot_length_hint(text or src, slot)
        aggressive = bool(flags & _AGGRESSIVE_FLAGS) or score > 5
        if aggressive:
            prompt = _SK_REPAIR_PROMPT_AGGRESSIVE.format(
                src=src or text,
                text=text or src,
                slot=slot,
                flags=", ".join(sorted(flags)),
                target_chars=_target_chars, current_chars=_current_chars,
                length_hint=_length_hint,
                name_hints=_SK_FOREIGN_NAME_HINTS,
                declension_rules=_SK_DECLENSION_RULES,
                glossary=_SK_REPAIR_GLOSSARY,
            )
            mode = "AGGRESSIVE"
        else:
            prompt = _SK_REPAIR_PROMPT_LIGHT.format(
                src=src or text,
                text=text,
                slot=slot,
                target_chars=_target_chars, current_chars=_current_chars,
                length_hint=_length_hint,
                name_hints=_SK_FOREIGN_NAME_HINTS,
                declension_rules=_SK_DECLENSION_RULES,
                glossary=_SK_REPAIR_GLOSSARY,
            )
            mode = "LIGHT"

        print(f"[REPAIR/OLLAMA/{mode}] seg {i+1} score={score} model={model}", flush=True)
        print(f"[REPAIR/OLLAMA]  SK : {text[:70]}", flush=True)
        try:
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
            result = str(payload.get("response") or "").strip()
        except Exception as e:
            print(f"[REPAIR/OLLAMA] ⚠ chyba API: {e}", flush=True)
            result = source_guided_template(src) or ""
            if result:
                print("[REPAIR/OLLAMA]  → source-template fallback", flush=True)
            else:
                continue

        result = guard_technical_text(result, source_text=src, fallback_text=text)

        if not result or len(result) > max(len(text or src) * 1.6, 20):
            print(f"[REPAIR/OLLAMA] ⚠ zamietnutý (prázdny alebo príliš dlhý)", flush=True)
            continue

        print(f"[REPAIR/OLLAMA]  → {result[:70]}", flush=True)
        seg_copy = dict(seg)
        seg_copy["text"] = result
        if "_audit" in seg_copy:
            seg_copy["_audit"]["repaired"] = True
            seg_copy["_audit"]["repair_mode"] = f"OLLAMA_{mode}"
        repaired[i] = seg_copy

    print("[REPAIR/OLLAMA] Hotovo.", flush=True)
    return repaired


def apply_gender_fix_sk_feminine(segments: list) -> list:
    """
    Deterministická oprava rodu pre prvú osobu ženského rodu v slovenčine.
    Vhodné ak prekladový model používa mužský rod ako default.

    Pravidlá:
      som Xl    → som Xla     (som videl → som videla, som zažil → som zažila, ...)
      [Bb]ol som → [Bb]ola som
      [Bb]ol by som → [Bb]ola by som
      Xl by som  → Xla by som  (zaujímal by som → zaujímala by som)
    """
    import re as _re

    def _fix(text: str) -> str:
        # som [verb]l → som [verb]la  (past tense m → f, covers: som videl, som bol, som vyrastal ...)
        text = _re.sub(r'\bsom (\w+l)\b', r'som \1a', text)
        # [Bb]ol som (word order reversed)
        text = _re.sub(r'\b([Bb])ol som\b', r'\1ola som', text)
        # [Bb]ol by som
        text = _re.sub(r'\b([Bb])ol by som\b', r'\1ola by som', text)
        # [verb]l by som  (conditional: zaujímal by som → zaujímala by som)
        text = _re.sub(r'\b(\w+l) by som\b', r'\1a by som', text)
        # by som [particle] [verb]l  (conditional reversed: by som sa zaujímal → by som sa zaujímala)
        text = _re.sub(r'\bby som (\w+ )(\w+l)\b', r'by som \1\2a', text)
        return text

    result = []
    for seg in segments:
        seg_copy = dict(seg)
        orig = (seg.get("text") or "").strip()
        fixed = _fix(orig)
        if fixed != orig:
            print(f"[GENDER-FIX] {orig[:70]}", flush=True)
            print(f"[GENDER-FIX] → {fixed[:70]}", flush=True)
        seg_copy["text"] = fixed
        result.append(seg_copy)
    return result


def fix_sk_conditional_grammar(segments: list) -> list:
    """
    Opraví typickú chybu TranslateGemma v slovenčine:
    'radi by sme/ste/by [infinitív]' → 'radi by sme/ste/by [l-forma]'

    Pravidlo: po 'radi by sme/ste' nasleduje infinitív (končí na -ť).
    Slovenská l-forma množného čísla = inf. kmeň + -li.
      odstrán-i-ť → odstrán-i-li
      nahrad-i-ť  → nahrad-i-li
      poved-a-ť   → poved-a-li
      rob-i-ť     → rob-i-li

    Výnimky s nepravidelnou l-formou sú ošetrené explicitne:
      ísť → išli, byť → boli, prísť → prišli, nájsť → našli, atď.
    """
    import re as _re

    _IRREG = {
        "ísť": "išli", "byť": "boli", "prísť": "prišli", "nájsť": "našli",
        "vziať": "vzali", "jesť": "jedli", "viesť": "viedli", "rásť": "rástli",
        "niesť": "niesli", "triesť": "triasli", "kviesť": "kvitli",
        "môcť": "mohli", "chcieť": "chceli", "vedieť": "vedeli",
        "vidieť": "videli", "sedieť": "sedeli", "letieť": "leteli",
        "stáť": "stáli", "dať": "dali", "ísť": "išli",
    }

    # pattern: radi by sme/ste (to/sa/si)? [infinitív] [a/alebo infinitív]*
    _PAT = _re.compile(
        r'(radi by (?:sme|ste)\s+(?:(?:to|sa|si|ich|ho|jej|im)\s+)?)'
        r'(\w+ť\b(?:\s+(?:a|alebo)\s+\w+ť\b)*)',
        _re.IGNORECASE,
    )

    def _to_lform(inf: str) -> str:
        low = inf.lower()
        if low in _IRREG:
            return _IRREG[low]
        if low.endswith("ť"):
            return inf[:-1] + "li"
        return inf

    def _fix_verb_chain(chain: str) -> str:
        """Convert 'odstrániť a nahradiť' → 'odstránili a nahradili'."""
        parts = _re.split(r'(\s+(?:a|alebo)\s+)', chain, flags=_re.IGNORECASE)
        result = []
        for part in parts:
            if _re.match(r'\w+ť$', part.strip(), _re.IGNORECASE):
                result.append(_to_lform(part.strip()))
            else:
                result.append(part)
        return "".join(result)

    def _fix(text: str) -> str:
        def _repl(m: re.Match) -> str:
            return m.group(1) + _fix_verb_chain(m.group(2))
        return _PAT.sub(_repl, text)

    import re
    result = []
    for seg in segments:
        seg_copy = dict(seg)
        orig = (seg.get("text") or "").strip()
        fixed = _fix(orig)
        if fixed != orig:
            print(f"[SK-GRAMMAR] {orig[:80]}", flush=True)
            print(f"[SK-GRAMMAR] → {fixed[:80]}", flush=True)
        seg_copy["text"] = fixed
        result.append(seg_copy)
    return result


def normalize_camelcase_for_tts(text: str) -> str:
    """
    Targeted CamelCase → TTS-readable normalization.

    Beží VŽDY pred TTS (aj keď je phonetic_respelling OFF).
    Rieši len konkrétne SQL/tech slová ktoré inak spôsobujú artefakty v TTS.

    C#          → cé šarp
    c#          → cé šarp
    .NET        → dot net
    . NET       → dot net
    isNull      → is nul
    isNotNull   → is not nul
    nullIf      → nul if
    coalesce    → koalesk   (SK TTS nevie "coalesce")
    ISNULL      → is nul    (ALL-CAPS variant)
    COALESCE    → koalesk
    NULLIF      → nul if
    """
    import re as _r
    _CAMEL_MAP = [
        # Programming language names / symbols
        (r'(?<!\w)C#(?!\w)',        'cé šarp'),
        (r'(?<!\w)c#(?!\w)',        'cé šarp'),
        (r'\.\s*NET\b',             ' dotnet'),
        (r'\.\s*net\b',             ' dotnet'),
        # CamelCase forms
        (r'\bisNotNull\b',          'íz not nul'),
        (r'\bisNull\b',             'ajznul'),
        (r'\bnullIf\b',             'nulíf'),
        (r'\bifNull\b',             'ajznul'),
        (r'\bISNOTNULL\b',          'íz not nul'),
        # ALL-CAPS + IS NULL / IS NOT NULL variants (long before short)
        (r'\bIS\s+NOT\s+NULL\b',    'íz not nul'),
        (r'\bIS\s+NULL\b',          'ajznul'),
        (r'\bNOT\s+NULL\b',         'not nul'),
        (r'\bISNULL\b',             'ajznul'),
        (r'\bIFNULL\b',             'ajznul'),
        (r'\bNULLIF\b',             'nulíf'),
        (r'\bCOALESCE\b',           'koalesk'),
        (r'(?i)\bcoalesce\b',       'koalesk'),
        # Gemma/Google lowercase 1-L variants ("is nul", "je nul", "iz nul")
        (r'(?i)\bis\s+not\s+nul\b', 'íz not nul'),
        (r'(?i)\bis\s+nul\b',       'ajznul'),
        (r'(?i)\biz\s+nul\b',       'ajznul'),
        (r'(?i)\bje\s+nul\b',       'ajznul'),
        # NULL → nul (single-L Slovak pronunciation)
        (r'\bNULL\b',               'nul'),
        (r'\bSQL\b',                'esíkjúel'),
    ]
    for pat, repl in _CAMEL_MAP:
        text = _r.sub(pat, repl, text)
    text = _r.sub(r'\s{2,}', ' ', text)
    return text.strip()


def fix_sk_sql_function_names(segments: list) -> list:
    """
    Opraví typickú chybu prekladu SQL názvov funkcií v slovenčine.

    TranslateGemma/Google prekladá SQL kľúčové slová ako vety:
      "the first one called IS NULL" → "prvá volaná je nul"  ← ZLEA
    Správne: "prvá volaná íz nul" (IS NULL = meno funkcie, nie stav hodnoty)

    Pravidlo: "je nul" v kontexte pomenovania funkcie → "íz nul"
    Kontext "hodnota je nul" (stav) sa NEMENÍ.
    """
    import re as _re

    # (vzor, náhrada) — len kontexty kde "je nul" = názov funkcie
    _RULES = [
        # volaná/volaný/nazvaná je nul → íz nul
        (r'\b(volaný|volaná|nazvaný|nazvaná|nazýva sa|volá sa|s názvom|menom)\s+je\s+nul\b',
         lambda m: m.group(0).replace(" je nul", " is nul")),
        # funkcia/metóda/príkaz "je nul" → funkcia "is nul"
        (r'\b(funkci[ae]|metód[ay]|príkaz[uy]?|klauzul[ay]?)\s+je\s+nul\b',
         lambda m: m.group(0).replace(" je nul", " is nul")),
        # "prvá je nul" alebo "druhá je nul" v SQL kontexte
        (r'\b(prvá|druhá|tretia)\s+(?:funkci[ae]\s+)?je\s+nul\b',
         lambda m: m.group(0).replace(" je nul", " is nul")),
        # is a nul → is nul (anglický zvyšok "is a null")
        (r'\bje\s+a\s+nul\b', "is nul"),
        # "je not nul" → "is not nul" (IS NOT NULL ako meno/podmienka)
        (r'\b(volaný|volaná|nazvaná|funkci[ae])\s+je\s+not\s+nul\b',
         lambda m: m.group(0).replace(" je not nul", " is not nul")),
        # priame "íz nul" → "is nul" (zvyšok zo starých prekladov)
        (r'\bíz\s+nul\b', "is nul"),
        (r'\bíz\s+not\s+nul\b', "is not nul"),
    ]

    def _fix(text: str) -> str:
        for pat, repl in _RULES:
            if callable(repl):
                text = _re.sub(pat, repl, text, flags=_re.IGNORECASE)
            else:
                text = _re.sub(pat, repl, text, flags=_re.IGNORECASE)
        return text

    result = []
    for seg in segments:
        seg_copy = dict(seg)
        orig = (seg.get("text") or "").strip()
        fixed = _fix(orig)
        if fixed != orig:
            print(f"[SK-SQL] {orig[:100]}", flush=True)
            print(f"[SK-SQL] → {fixed[:100]}", flush=True)
        seg_copy["text"] = fixed
        result.append(seg_copy)
    return result


def apply_sql_source_guided_cleanup(segments: list) -> list:
    """
    Pre SQL tutorial vety aplikuje source-guided template alebo aspoň SQL guard ešte
    pred zápisom základného *_sk_segments.json.
    """
    result = []
    for seg in segments:
        seg_copy = dict(seg)
        text = (seg.get("text") or "").strip()
        src = (seg.get("text_src") or "").strip()
        if not text:
            result.append(seg_copy)
            continue

        template = source_guided_template(src) if src else None
        fixed = guard_technical_text(text, source_text=src, fallback_text=text)
        preferred = guard_technical_text(template, source_text=src, fallback_text=text) if template else fixed

        if preferred != text:
            print(f"[SQL-SRC] {text[:100]}", flush=True)
            print(f"[SQL-SRC] → {preferred[:100]}", flush=True)

        seg_copy["text"] = preferred
        result.append(seg_copy)
    return result


def apply_linux_source_guided_cleanup(segments: list, *, mixed_mode: bool = False) -> list:
    """
    Deterministic Linux / Proxmox cleanup before JSON export.

    mixed_mode=True keeps Slovak narration, but preserves important Linux/CLI
    literals in English.
    """
    result = []
    for seg in segments:
        seg_copy = dict(seg)
        text = (seg.get("text") or "").strip()
        src = (seg.get("text_src") or "").strip()
        if not text:
            result.append(seg_copy)
            continue

        fixed = guard_linux_text(text, source_text=src, fallback_text=text)
        template = linux_source_guided_template(src) if src else None
        if template and looks_like_linux_sql_drift(fixed, source_text=src):
            fixed = template
        if mixed_mode:
            fixed = apply_linux_mixed_mode(fixed, source_text=src, fallback_text=text)

        if fixed != text:
            print(f"[LINUX-SRC] {text[:100]}", flush=True)
            print(f"[LINUX-SRC] → {fixed[:100]}", flush=True)

        seg_copy["text"] = fixed
        result.append(seg_copy)
    return result


def fix_sk_noun_declension(segments: list) -> list:
    """
    Opraví typické chyby skloňovania podstatných mien po predložkách.

    Zachytáva: predložka + NOMINATÍV → správny pád
      - po v/vo/na/pri/po/o  (lokál)  : tabuľka → tabuľke
      - po z/zo/od/bez/do/okolo       (genitív): tabuľka → tabuľky
    """
    import re as _re

    _LOC = r'(?:vo?|na|pri|po|o)'
    _GEN = r'(?:zo?|od|bez|do|okolo|namiesto|počas|kvôli)'

    # (nominatív, genitív_sg, lokál_sg)
    _FORMS = [
        # Vzor žena — tvrdá spoluhláska + -a
        ("tabuľka",    "tabuľky",    "tabuľke"),
        ("databáza",   "databázy",   "databáze"),
        ("hodnota",    "hodnoty",    "hodnote"),
        ("podmienka",  "podmienky",  "podmienke"),
        ("položka",    "položky",    "položke"),
        ("štruktúra",  "štruktúry",  "štruktúre"),
        ("schéma",     "schémy",     "schéme"),
        ("premenná",   "premennej",  "premennej"),
        ("chyba",      "chyby",      "chybe"),
        ("správa",     "správy",     "správe"),
        ("objednávka", "objednávky", "objednávke"),
        ("transakcia", "transakcie", "transakcii"),
        ("operácia",   "operácie",   "operácii"),
        ("funkcia",    "funkcie",    "funkcii"),
        ("závislosť",  "závislosti", "závislosti"),
        ("vlastnosť",  "vlastnosti", "vlastnosti"),
        ("možnosť",    "možnosti",   "možnosti"),
        ("časť",       "časti",      "časti"),
        ("odpoveď",    "odpovede",   "odpovedi"),
        ("skupina",    "skupiny",    "skupine"),
        ("tabuľka",    "tabuľky",    "tabuľke"),
        # Vzor dub — mužský rod neživotný
        ("stĺpec",     "stĺpca",     "stĺpci"),
        ("riadok",     "riadku",     "riadku"),
        ("príkaz",     "príkazu",    "príkaze"),
        ("výsledok",   "výsledku",   "výsledku"),
        ("dotaz",      "dotazu",     "dotaze"),
        ("index",      "indexu",     "indexe"),
        ("súbor",      "súboru",     "súbore"),
        ("zoznam",     "zoznamu",    "zozname"),
        ("parameter",  "parametra",  "parametri"),
        ("skript",     "skriptu",    "skripte"),
        ("server",     "servera",    "serveri"),
        ("výraz",      "výrazu",     "výraze"),
        ("formát",     "formátu",    "formáte"),
        ("atribút",    "atribútu",   "atribúte"),
        ("operátor",   "operátora",  "operátore"),
        ("kľúč",       "kľúča",      "kľúči"),
        ("vzťah",      "vzťahu",     "vzťahu"),
        ("záznam",     "záznamu",    "zázname"),
        ("blok",       "bloku",      "bloku"),
        ("prípad",     "prípadu",    "prípade"),
        ("postup",     "postupu",    "postupe"),
        ("výpis",      "výpisu",     "výpise"),
        ("krok",       "kroku",      "kroku"),
    ]

    # Zostroj pravidlá — len slová s diakritikou (slovenské), nie anglické homografy
    _rules_loc: list = []
    _rules_gen: list = []
    seen_nom: set = set()
    for nom, gen, loc in _FORMS:
        if nom in seen_nom:
            continue
        seen_nom.add(nom)
        # Ak sa nominatív = lokál (žiadna zmena), netlač prázdne pravidlo
        if loc == nom and gen == nom:
            continue
        # Regex: predložka + medzera + slovo — bez IGNORECASE aby sme odfiltrovali
        # SQL kľúčové slová veľkými písmenami (FROM, WHERE...) cez presný lowercase match
        pat_loc = _re.compile(rf'\b({_LOC})\s+({_re.escape(nom)})\b')
        pat_gen = _re.compile(rf'\b({_GEN})\s+({_re.escape(nom)})\b')
        if loc != nom:
            _rules_loc.append((pat_loc, loc))
        if gen != nom:
            _rules_gen.append((pat_gen, gen))

    def _fix(text: str) -> str:
        for pat, correct in _rules_loc:
            text = pat.sub(lambda m, c=correct: m.group(1) + " " + c, text)
        for pat, correct in _rules_gen:
            text = pat.sub(lambda m, c=correct: m.group(1) + " " + c, text)
        return text

    result = []
    for seg in segments:
        seg_copy = dict(seg)
        orig = (seg.get("text") or "").strip()
        fixed = _fix(orig)
        if fixed != orig:
            print(f"[SK-DECL] {orig[:100]}", flush=True)
            print(f"[SK-DECL] → {fixed[:100]}", flush=True)
        seg_copy["text"] = fixed
        result.append(seg_copy)
    return result


def fix_sk_foreign_name_declension(segments: list) -> list:
    """
    Opraví bezpečné a časté chyby pri skloňovaní cudzích mužských mien v slovenčine.

    Zámerne rieši len konzervatívne prípady:
    - sloveso + meno v priamom predmete (oslovili Kratos -> oslovili Kratosa)
    - zopár známych rozbitých naratívnych konštrukcií
    """
    import re as _re

    _name_forms = {
        "Kratos": "Kratosa",
        "Jason": "Jasona",
        "Freddy": "Freddyho",
        "Shinnok": "Shinnoka",
        "Predator": "Predátora",
        "Terminator": "Terminátora",
        "Nightwolf": "Nightwolfa",
        "RoboCop": "RoboCopa",
        "Raiden": "Raidena",
        "Fujin": "Fujina",
    }

    _acc_trigger = (
        r"(?:oslovil(?:i|a|o)?|spoznal(?:i|a|o)?|poznal(?:i|a|o)?|"
        r"našiel(?:i|a|o)?|hľadal(?:i|a|o)?|videl(?:i|a|o)?|"
        r"miloval(?:i|a|o)?|nenávidel(?:i|a|o)?|zabil(?:i|a|o)?|"
        r"porazil(?:i|a|o)?|zničil(?:i|a|o)?|porezal(?:i|a|o)?|"
        r"privolal(?:i|a|o)?|poslal(?:i|a|o)?|vtiahol(?:i|a|o)?|"
        r"ponúkol(?:i|a|o)?|vytiahol(?:i|a|o)?|všiml(?:o|a|i)?(?:\s+si)?)"
    )

    _direct_object_rules = [
        (
            _re.compile(
                rf"\b({_acc_trigger})\s+({_re.escape(name)})\b(?!\s+a\s+[A-ZÁ-Ž])"
            ),
            acc,
        )
        for name, acc in _name_forms.items()
    ]

    _phrase_rules = [
        (
            _re.compile(r"\btíto\s+bohovia\s+Kratos\s+nikdy\s+nemilovali\b", _re.IGNORECASE),
            "týchto bohov Kratos nikdy nemiloval",
        ),
    ]

    def _fix(text: str) -> str:
        for pat, acc in _direct_object_rules:
            text = pat.sub(lambda m, c=acc: f"{m.group(1)} {c}", text)
        for pat, repl in _phrase_rules:
            text = pat.sub(repl, text)
        return text

    result = []
    for seg in segments:
        seg_copy = dict(seg)
        orig = (seg.get("text") or "").strip()
        fixed = _fix(orig)
        if fixed != orig:
            print(f"[SK-NAMES] {orig[:100]}", flush=True)
            print(f"[SK-NAMES] → {fixed[:100]}", flush=True)
        seg_copy["text"] = fixed
        result.append(seg_copy)
    return result


def refine_fulldoc_with_llama(
    segments: list,
    tgt_lang: str,
    llama_model: str,
    n_ctx: int = 8192,
    n_gpu_layers: int = 20,
    context_hint: str = "",
    speaker_gender: str = "",
) -> list:
    """
    Full-document second-pass refinement using LLM.
    Posiela celý dokument naraz s §N§ markérmi — model vidí plný kontext.
    Vhodné na opravu rodu, sklonenia, konzistencie.
    """
    import re as _re
    from llama_cpp import Llama

    lang_name = target_lang_name(tgt_lang)

    lines = []
    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        slot = float(seg.get("end", 0)) - float(seg.get("start", 0))
        # Include slot duration so model knows timing constraint
        lines.append(f"§{i}§[{slot:.1f}s] {text}" if text else f"§{i}§[{slot:.1f}s]")
    full_text = "\n".join(lines)

    context_line = f"\nKontext: {context_hint}" if context_hint else ""
    _decl_rules = (
        "Skloňovanie (POVINNÉ): po predložkách v/vo/na/pri/po/o používaj lokál, "
        "po z/zo/od/bez/do používaj genitív. "
        "Po 'radi/chceli/mohli by sme/ste' musí nasledovať l-forma, nie infinitív. "
        "Cudzie vlastné mená skloňuj prirodzene po slovensky podľa syntaktickej roly "
        "(napr. Kratos -> Kratosa, Freddy -> Freddyho, Jason -> Jasona).\n"
    ) if tgt_lang in ("sk", "slk") else ""
    _gender_rules = ""
    if tgt_lang in ("sk", "slk", "cs", "ces") and speaker_gender == "feminine":
        _gender_rules = (
            "ROD HOVORIACEJ (POVINNÉ): Hovoriaca osoba je ŽENA — pre 1. osobu jednotného "
            "čísla v MINULOM čase POUŽÍVAJ ŽENSKÉ tvary slovies aj prídavných mien. "
            "Príklady: 'Začala som' (NIE 'Začal som'), 'Naučila som sa' (NIE 'Naučil som sa'), "
            "'bola som rada' (NIE 'bol som rád'), 'urobila som' (NIE 'urobil som'), "
            "'dokončila som' (NIE 'dokončil som'). Toto platí pre CELÝ dokument konzistentne.\n"
        )
    elif tgt_lang in ("sk", "slk", "cs", "ces") and speaker_gender == "masculine":
        _gender_rules = (
            "ROD HOVORIACEHO: Hovoriaci je MUŽ — pre 1. osobu jednotného čísla minulého času "
            "používaj mužské tvary ('Začal som', 'Naučil som sa', 'bol som rád').\n"
        )
    system = (
        f"Si natívny hovorca jazyka {lang_name}.{context_line}\n"
        f"Dostaneš preložený text rozdelený na riadky pomocou markérov §N§[Xs].\n"
        f"Číslo v zátvorkách [Xs] je časový slot v sekundách — koľko má rečník čas na daný riadok.\n"
        f"Oprav KAŽDÝ riadok — skloňovanie, časovanie, rod a prirodzenosť.\n"
        f"{_decl_rules}"
        f"{_gender_rules}"
        f"DÔLEŽITÉ: Opravený text NESMIE byť dlhší ako originál. Ak je veta príliš dlhá na daný slot, skráť ju.\n"
        f"Zachovaj markéry §N§ na začiatku každého riadku PRESNE TAK AKO SÚ (bez [Xs]).\n"
        f"Nevynechaj žiadny riadok. Vráť IBA opravený text s markérmi, bez vysvetlení."
    )

    # Odhadneme tokeny — realisticky: 1 token ≈ 3 znaky, výstup ≈ 1.1× vstup
    sys_tokens  = int(len(system) / 3) + 50
    text_tokens = max(256, int(len(full_text) / 3))
    out_tokens  = max(512, int(len(full_text) / 2.5))
    # Cap n_ctx na 4096 — batching zabezpečí dlhé dokumenty
    actual_ctx  = n_ctx  # použijeme len to čo prišlo (default 8192), bez auto-scale nahor
    max_input_tokens = int(actual_ctx * 0.45) - sys_tokens  # ~45 % kontextu pre vstup

    print(f"[REFINE-FULL] Loading model: {llama_model} (n_ctx={actual_ctx})", flush=True)
    llm = Llama(model_path=llama_model, n_ctx=actual_ctx, n_gpu_layers=n_gpu_layers, verbose=False)

    def _refine_chunk(chunk_lines: list[str]) -> str:
        chunk_text = "\n".join(chunk_lines)
        chunk_in_tok = int(len(chunk_text) / 3) + 10
        available_out = actual_ctx - sys_tokens - chunk_in_tok - 64  # 64 safety margin
        chunk_out = max(256, min(available_out, int(len(chunk_text) / 2.2)))
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": chunk_text},
            ],
            temperature=0.05,
            max_tokens=chunk_out,
        )
        if resp["choices"][0].get("finish_reason") == "length":
            print("[REFINE-FULL] WARNING: chunk output truncated", flush=True)
        return resp["choices"][0]["message"]["content"].strip()

    print(f"[REFINE-FULL] Input chars: {len(full_text)}, segments: {len(segments)}, ctx={actual_ctx}", flush=True)

    if text_tokens <= max_input_tokens:
        result_text = _refine_chunk(lines)
    else:
        # Dávkový refinement — batch musí mať dosť miesta pre vstup aj výstup (~1:1)
        # max_input_tokens = 45% kontextu; výstup ≈ rovnaký → batch ≤ 22% kontextu * 3 znaky/token
        batch_chars = max(500, int(actual_ctx * 0.22) * 3)
        print(f"[REFINE-FULL] Batching (batch≈{batch_chars} chars, ctx={actual_ctx})", flush=True)
        current: list[str] = []
        current_chars = 0
        batch_results: list[str] = []
        for line in lines:
            if current_chars + len(line) > batch_chars and current:
                batch_results.append(_refine_chunk(current))
                current = []
                current_chars = 0
            current.append(line)
            current_chars += len(line) + 1
        if current:
            batch_results.append(_refine_chunk(current))
        result_text = "\n".join(batch_results)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    def _strip_refine_artifacts(text: str) -> str:
        """Odstráni artefakty ktoré refine model pridáva k neúplným vetám."""
        # Markdown artefakty
        text = _re.sub(r'\*+', '', text)
        text = _re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        text = _re.sub(r'^\s*[-•]\s+', '', text, flags=_re.MULTILINE)
        text = _re.sub(r'^\s*#+\s+', '', text, flags=_re.MULTILINE)
        text = _re.sub(r'\n+', ' ', text)
        text = _re.sub(r' {2,}', ' ', text)
        text = _re.sub(r'\s*\([^)]{0,80}chýba[^)]*\)', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\s*\([^)]{0,80}missing[^)]*\)', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\s*\([^)]{0,80}neúplný[^)]*\)', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r',?\s*ako ste navrhli\.?$', '', text, flags=_re.IGNORECASE)
        text = _re.sub(r',?\s*podľa kontextu\.?$', '', text, flags=_re.IGNORECASE)
        # Trailing "..." zvyšok vety od translategemma → odstrán
        text = _re.sub(r'\s*\.\.\.\s*$', '', text)
        return text.strip()

    # Parsuj §N§ markéry (výstup môže mať §N§ alebo §N§[Xs] — strip [Xs] ak tam ostalo)
    refined_map: dict[int, str] = {}
    for match in _re.finditer(r'§(\d+)§(?:\[\d+(?:\.\d+)?s\])?\s*(.*?)(?=§\d+§|$)', result_text, _re.DOTALL):
        idx = int(match.group(1))
        text = _strip_refine_artifacts(match.group(2).strip())
        if text:
            refined_map[idx] = text

    missing = [i for i in range(len(segments)) if i not in refined_map]
    print(f"[REFINE-FULL] Parsed {len(refined_map)}/{len(segments)} segments.", flush=True)
    if missing:
        print(f"[REFINE-FULL] Missing segment indices: {missing}", flush=True)

    result = []
    longer_count = 0
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        if i in refined_map:
            orig = (seg.get("text") or "").strip()
            refined = refined_map[i]
            # Reject if refined is significantly longer than original (>10% chars)
            if len(refined) > len(orig) * 1.10 and len(orig) > 10:
                seg_copy["text"] = orig
                longer_count += 1
                print(f"[REFINE-FULL] [{i}] REJECTED (longer: {len(orig)}→{len(refined)} chars)", flush=True)
            else:
                seg_copy["text"] = refined
                if orig != refined:
                    print(f"[REFINE-FULL] [{i}] {orig[:50]} → {refined[:50]}", flush=True)
        result.append(seg_copy)

    if longer_count:
        print(f"[REFINE-FULL] Rejected {longer_count} segments (too long) — kept originals.", flush=True)
    print("[REFINE-FULL] Done.", flush=True)
    return result


def refine_fulldoc_with_ollama(
    segments: list,
    tgt_lang: str,
    *,
    url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:26b",
    timeout: int = 120,
    n_ctx: int = 8192,
    context_hint: str = "",
    speaker_gender: str = "",
) -> list:
    """
    Full-document refinement cez Ollama HTTP API.
    Zachováva rovnaký marker-based workflow ako llama.cpp verzia, ale nevyžaduje GGUF model.
    """
    import re as _re

    lang_name = target_lang_name(tgt_lang)

    lines = []
    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        slot = float(seg.get("end", 0)) - float(seg.get("start", 0))
        lines.append(f"§{i}§[{slot:.1f}s] {text}" if text else f"§{i}§[{slot:.1f}s]")
    full_text = "\n".join(lines)

    context_line = f"\nKontext: {context_hint}" if context_hint else ""
    _decl_rules = (
        "Skloňovanie (POVINNÉ): po predložkách v/vo/na/pri/po/o používaj lokál, "
        "po z/zo/od/bez/do používaj genitív. "
        "Po 'radi/chceli/mohli by sme/ste' musí nasledovať l-forma, nie infinitív. "
        "Cudzie vlastné mená skloňuj prirodzene po slovensky podľa syntaktickej roly "
        "(napr. Kratos -> Kratosa, Freddy -> Freddyho, Jason -> Jasona).\n"
    ) if tgt_lang in ("sk", "slk") else ""
    _gender_rules = ""
    if tgt_lang in ("sk", "slk", "cs", "ces") and speaker_gender == "feminine":
        _gender_rules = (
            "ROD HOVORIACEJ (POVINNÉ): Hovoriaca osoba je ŽENA — pre 1. osobu jednotného "
            "čísla v MINULOM čase POUŽÍVAJ ŽENSKÉ tvary slovies aj prídavných mien. "
            "Príklady: 'Začala som' (NIE 'Začal som'), 'Naučila som sa' (NIE 'Naučil som sa'), "
            "'bola som rada' (NIE 'bol som rád'), 'dokončila som' (NIE 'dokončil som'). "
            "Toto platí pre CELÝ dokument konzistentne.\n"
        )
    elif tgt_lang in ("sk", "slk", "cs", "ces") and speaker_gender == "masculine":
        _gender_rules = (
            "ROD HOVORIACEHO: Hovoriaci je MUŽ — pre 1. osobu jednotného čísla minulého času "
            "používaj mužské tvary ('Začal som', 'Naučil som sa', 'bol som rád').\n"
        )
    system = (
        f"Si natívny hovorca jazyka {lang_name}.{context_line}\n"
        f"Dostaneš preložený text rozdelený na riadky pomocou markérov §N§[Xs].\n"
        f"Číslo v zátvorkách [Xs] je časový slot v sekundách — koľko má rečník čas na daný riadok.\n"
        f"Oprav KAŽDÝ riadok — skloňovanie, časovanie, rod a prirodzenosť.\n"
        f"{_decl_rules}"
        f"{_gender_rules}"
        f"DÔLEŽITÉ: Opravený text NESMIE byť dlhší ako originál. Ak je veta príliš dlhá na daný slot, skráť ju.\n"
        f"Zachovaj markéry §N§ na začiatku každého riadku PRESNE TAK AKO SÚ (bez [Xs]).\n"
        f"Nevynechaj žiadny riadok. Vráť IBA opravený text s markérmi, bez vysvetlení."
    )

    sys_tokens = int(len(system) / 3) + 50
    text_tokens = max(256, int(len(full_text) / 3))
    actual_ctx = n_ctx
    max_input_tokens = int(actual_ctx * 0.45) - sys_tokens

    print(f"[REFINE-FULL/OLLAMA] Using model: {model} url={url}", flush=True)

    def _refine_chunk(chunk_lines: list[str]) -> str:
        chunk_text = "\n".join(chunk_lines)
        prompt = f"{system}\n\n{chunk_text}".strip()
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
        return str(payload.get("response") or "").strip()

    print(f"[REFINE-FULL/OLLAMA] Input chars: {len(full_text)}, segments: {len(segments)}, ctx={actual_ctx}", flush=True)

    if text_tokens <= max_input_tokens:
        result_text = _refine_chunk(lines)
    else:
        batch_chars = max(500, int(actual_ctx * 0.22) * 3)
        print(f"[REFINE-FULL/OLLAMA] Batching (batch≈{batch_chars} chars, ctx={actual_ctx})", flush=True)
        current: list[str] = []
        current_chars = 0
        batch_results: list[str] = []
        for line in lines:
            if current_chars + len(line) > batch_chars and current:
                batch_results.append(_refine_chunk(current))
                current = []
                current_chars = 0
            current.append(line)
            current_chars += len(line) + 1
        if current:
            batch_results.append(_refine_chunk(current))
        result_text = "\n".join(batch_results)

    def _strip_refine_artifacts(text: str) -> str:
        text = _re.sub(r"\*+", "", text)
        text = _re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
        text = _re.sub(r"^\s*[-•]\s+", "", text, flags=_re.MULTILINE)
        text = _re.sub(r"^\s*#+\s+", "", text, flags=_re.MULTILINE)
        text = _re.sub(r"\n+", " ", text)
        text = _re.sub(r" {2,}", " ", text)
        text = _re.sub(r"\s*\([^)]{0,80}chýba[^)]*\)", "", text, flags=_re.IGNORECASE)
        text = _re.sub(r"\s*\([^)]{0,80}missing[^)]*\)", "", text, flags=_re.IGNORECASE)
        text = _re.sub(r"\s*\([^)]{0,80}neúplný[^)]*\)", "", text, flags=_re.IGNORECASE)
        text = _re.sub(r",?\s*ako ste navrhli\.?$", "", text, flags=_re.IGNORECASE)
        text = _re.sub(r",?\s*podľa kontextu\.?$", "", text, flags=_re.IGNORECASE)
        text = _re.sub(r"\s*\.\.\.\s*$", "", text)
        return text.strip()

    refined_map: dict[int, str] = {}
    for match in _re.finditer(r'§(\d+)§(?:\[\d+(?:\.\d+)?s\])?\s*(.*?)(?=§\d+§|$)', result_text, _re.DOTALL):
        idx = int(match.group(1))
        text = _strip_refine_artifacts(match.group(2).strip())
        if text:
            refined_map[idx] = text

    missing = [i for i in range(len(segments)) if i not in refined_map]
    print(f"[REFINE-FULL/OLLAMA] Parsed {len(refined_map)}/{len(segments)} segments.", flush=True)
    if missing:
        print(f"[REFINE-FULL/OLLAMA] Missing segment indices: {missing}", flush=True)

    result = []
    longer_count = 0
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        if i in refined_map:
            orig = (seg.get("text") or "").strip()
            refined = guard_technical_text(refined_map[i], source_text=(seg.get("text_src") or ""), fallback_text=orig)
            if len(refined) > len(orig) * 1.10 and len(orig) > 10:
                seg_copy["text"] = orig
                longer_count += 1
                print(f"[REFINE-FULL/OLLAMA] [{i}] REJECTED (longer: {len(orig)}→{len(refined)} chars)", flush=True)
            else:
                seg_copy["text"] = refined
                if orig != refined:
                    print(f"[REFINE-FULL/OLLAMA] [{i}] {orig[:50]} → {refined[:50]}", flush=True)
        result.append(seg_copy)

    if longer_count:
        print(f"[REFINE-FULL/OLLAMA] Rejected {longer_count} segments (too long) — kept originals.", flush=True)
    print("[REFINE-FULL/OLLAMA] Done.", flush=True)
    return result


# Tech terms that Google Translate must NOT transliterate / translate.
# Google Translate fonetizuje napr. "Proxmox" → "Proksmoks", "server" → "servr".
_GOOGLE_PROTECT_TERMS = [
    # Produkt / brand mená
    "Proxmox", "VMware", "XCP-NG", "XCPNG", "Kubernetes", "Terraform",
    "Ubuntu", "Debian", "CentOS", "AlmaLinux", "Rocky Linux",
    "Ansible", "Docker", "Portainer", "TrueNAS", "pfSense",
    # IT pojmy ktoré Google skracuje / fonetizuje
    "server", "Server", "cluster", "Cluster",
    "hypervisor", "Hypervisor", "firewall", "Firewall",
    "snapshot", "Snapshot", "backup", "Backup",
    # Game-dev / programming pojmy
    # Google Translate idiomatically renders "engines" → "kravatový motor" (tie engines=kravatový)
    # alebo "physics engine" → "fyzikálny motor". Chceme zachovať "engine" v EN.
    "engine", "Engine", "engines", "Engines",
    "physics engine", "graphics engine", "game engine",
    "shader", "Shader", "shaders", "Shaders",
    "voxel", "Voxel", "voxels", "Voxels",
    "mesh", "meshes", "vertex", "vertices",
    "GPU", "CPU", "RAM", "VRAM", "API", "SDK",
    "C++", "Python", "Rust", "OpenGL", "Vulkan", "DirectX",
    "shader code", "compute shader", "vertex shader", "fragment shader",
    # Linux / CLI literals
    "SSH", "SMTP", "email", "e-mail",
    "apt", "apt update", "apt upgrade", "apt install", "apt full-upgrade",
    "apt-get", "apt-get update", "apt-get upgrade",
    "dpkg-reconfigure", "systemctl", "systemd",
    "unattended-upgrades", "root", "sudo",
    "/etc/apt", "/etc/apt/apt.conf.d",
]


def translate_segment_with_google(
    text: str,
    tgt_lang: str,
    src_lang: str = "en",
    retries: int = 3,
    apply_phonetic: bool = True,
) -> str:
    """Free Google Translate via mobile web endpoint (no API key needed).
    Inspired by pyvideotrans/videotrans/translator/_google.py.
    Tech terms in _GOOGLE_PROTECT_TERMS are protected via placeholders to prevent
    Google from transliterating product names (Proxmox → Proksmoks, server → servr)."""
    import time as _time
    import html
    tgt_code = _GOOGLE_LANG_MAP.get(tgt_lang.lower(), tgt_lang.lower()[:2])
    src_code = _GOOGLE_LANG_MAP.get(src_lang.lower(), src_lang.lower()[:2])

    # Protect terms before translation
    _protected = {}
    _text_to_translate = text
    # Protect video timestamps like 2:30, 1:05:30 (dynamic, regex-based)
    import re as _re_g
    _ts_matches = list(_re_g.finditer(r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b', _text_to_translate))
    for _tsi, _tsm in enumerate(reversed(_ts_matches)):
        _ph = f"__TS{_tsi}__"
        _protected[_ph] = _tsm.group(1)
        _text_to_translate = _text_to_translate[:_tsm.start()] + _ph + _text_to_translate[_tsm.end():]
    _text_to_translate, _linux_protected = protect_linux_literals(_text_to_translate)
    _protected.update(_linux_protected)
    for _ti, _term in enumerate(_GOOGLE_PROTECT_TERMS):
        if _term in _text_to_translate:
            _ph = f"__GT{_ti}__"
            _text_to_translate = _text_to_translate.replace(_term, _ph)
            _protected[_ph] = _term

    for attempt in range(retries):
        try:
            url = (
                f"https://translate.google.com/m"
                f"?sl={src_code}&tl={tgt_code}&hl={tgt_code}"
                f"&q={requests.utils.quote(_text_to_translate)}"
            )
            resp = requests.get(url, headers=_GOOGLE_HEADERS, timeout=15, verify=False)
            resp.raise_for_status()
            m = re.search(r'class="(?:result-container|t0)">([^<]+)<', resp.text)
            if not m:
                raise RuntimeError("Google Translate: no result found in response")
            translated = html.unescape(m.group(1).strip())
            # Restore protected terms
            for _ph, _orig in _protected.items():
                translated = translated.replace(_ph, _orig)
                # Also handle if Google lowercased the placeholder
                translated = translated.replace(_ph.lower(), _orig)
            # Phonetic respelling — respektuj `apply_phonetic` flag
            # (OmniVoice nepotrebuje, multilingual zero-shot)
            if apply_phonetic:
                translated = apply_phonetic_respelling(translated)
            translated = guard_linux_text(translated, source_text=text, fallback_text=text)
            print(f"[GOOGLE] {tgt_code}: '{text[:50]}' → '{translated[:50]}'", flush=True)
            return translated
        except Exception as e:
            print(f"  [GOOGLE] attempt {attempt+1} failed: {e}", flush=True)
            if attempt < retries - 1:
                _time.sleep(2)
    print(f"  [GOOGLE] all retries failed, returning empty", flush=True)
    return ""


# ---------------------------------------------------------------------------
# NLLB-200 → Gemma cleanup → QA pipeline
# ---------------------------------------------------------------------------

# NLLB-200 language codes (BCP-47 + script tag)
NLLB_LANG_CODES: dict[str, str] = {
    "ces": "ces_Latn", "cs": "ces_Latn",
    "slk": "slk_Latn", "sk": "slk_Latn",
    "eng": "eng_Latn", "en": "eng_Latn",
    "deu": "deu_Latn", "de": "deu_Latn",
    "fra": "fra_Latn", "fr": "fra_Latn",
    "spa": "spa_Latn", "es": "spa_Latn",
    "ita": "ita_Latn", "it": "ita_Latn",
    "pol": "pol_Latn", "pl": "pol_Latn",
    "rus": "rus_Cyrl", "ru": "rus_Cyrl",
    "ukr": "ukr_Cyrl", "uk": "ukr_Cyrl",
    "por": "por_Latn", "pt": "por_Latn",
    "nld": "nld_Latn", "nl": "nld_Latn",
    "zho": "zho_Hans", "zh": "zho_Hans",
    "jpn": "jpn_Jpan", "ja": "jpn_Jpan",
    "kor": "kor_Hang", "ko": "kor_Hang",
}

_nllb_model = None
_nllb_tokenizer = None


def _nllb_lang_code(tgt_lang: str) -> str:
    return NLLB_LANG_CODES.get(tgt_lang.lower(), f"{tgt_lang[:3]}_Latn")


def translate_segment_with_nllb(
    text: str,
    tgt_lang: str,
    model_name: str = "facebook/nllb-200-distilled-1.3B",
    src_lang: str = "eng_Latn",
    device: str = "cpu",
) -> str:
    """Stage A: NLLB-200 mechanical translation — no refusals, no filtering."""
    global _nllb_model, _nllb_tokenizer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tgt_code = _nllb_lang_code(tgt_lang)

    if _nllb_model is None or _nllb_tokenizer is None:
        print(f"[NLLB] Loading {model_name} on {device}...", flush=True)
        _nllb_tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        print(f"[NLLB] Model loaded ({device})", flush=True)

    _nllb_tokenizer.src_lang = src_lang
    inputs = _nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    try:
        forced_bos_token_id = _nllb_tokenizer.lang_code_to_id[tgt_code]
    except (AttributeError, KeyError):
        forced_bos_token_id = _nllb_tokenizer.convert_tokens_to_ids(tgt_code)

    with torch.no_grad():
        outputs = _nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    translated = _nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)
    del inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[NLLB] {len(text.split())}w → {len(translated.split())}w", flush=True)
    return translated


def free_nllb_model() -> None:
    """Release NLLB model from memory (call before loading large GPU models)."""
    global _nllb_model, _nllb_tokenizer
    if _nllb_model is not None:
        del _nllb_model
        _nllb_model = None
    if _nllb_tokenizer is not None:
        del _nllb_tokenizer
        _nllb_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[NLLB] Model released from memory", flush=True)


def cleanup_with_gemma(
    text: str,
    llama_model: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = 56,
    llm_instance=None,
    tgt_lang_name: str = "češtinu",
) -> str:
    """Stage B: Gemma language polish — fix grammar/fluency, NEVER soften tone."""
    if llm_instance is None:
        llm = Llama(model_path=llama_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    else:
        llm = llm_instance
    system = (
        f"Jsi jazykový editor. Dostaneš strojový překlad do {tgt_lang_name}. "
        "Tvůj úkol: POUZE oprav gramatiku, přirozené slovní spojení a plynulost. "
        "NEZMÍRŇUJ explicitnost. ZACHOVEJ původní tón a intenzitu přesně. "
        "Pokud je originál vulgární, silný nebo přímý — nechej to tak. "
        "NEMEŇ smysl, NEODSTRAŇUJ ani NEPŘIDÁVEJ informace. "
        "Výstup: POUZE upravený text, bez vysvětlení."
    )
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.05,
        max_tokens=512,
    )
    cleaned = resp["choices"][0]["message"]["content"].strip()
    cleaned = cleaned.replace(">>>", "").replace("<<<", "").strip()

    # Detect meta-response: model describes the task instead of performing it.
    # This happens when the LLM acknowledges the instruction instead of executing it,
    # producing a self-description like "Rozumím. Dostanu překlad..." instead of the
    # actual cleaned translation.
    _META_STARTERS = (
        "Rozumím", "Jasně,", "Jasně.", "Dobře,", "Dobře ", "OK,", "Samozřejmě",
        "Ano,", "Ano.", "Dobre,", "Dobre.", "Porozumiem", "Porozumiem,",
        "Získám", "Dostanu", "Bude ", "Pošli", "Pošlete", "Mluvte",
        "Prosím, vlož", "Jsem připraven", "Pojďme na to",
        "Toto je preklad", "Tu je preklad", "Nasleduje preklad",
        "Preklad:", "Poznámka:", "Note:", "Výsledok:",
    )
    _META_ANYWHERE = (
        "halucin", "nesprávny preklad", "incorrect translation",
        "skrátená verzia", "lepší preklad", "pre dabing by",
    )
    _cl = cleaned.lower()
    _is_meta_start = cleaned.startswith(_META_STARTERS) or any(p in _cl for p in _META_ANYWHERE)
    _is_too_long = len(cleaned) > max(2 * len(text) + 80, 400)
    if _is_meta_start or _is_too_long:
        print(
            f"[GEMMA-CLEAN] WARNING: meta-response detected "
            f"(start={_is_meta_start}, ratio={len(cleaned)}/{len(text)}={len(cleaned)/max(len(text),1):.1f}x) "
            f"— using MADLAD output unchanged",
            flush=True,
        )
        return text

    print(f"[GEMMA-CLEAN] {len(text.split())}w → {len(cleaned.split())}w", flush=True)
    return cleaned


def qa_check_tone(
    original_en: str,
    translated: str,
    llama_model: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = 32,
    llm_instance=None,
    tgt_lang: str = "cs",
) -> tuple[bool, str]:
    """Stage C: QA — verify tone/explicitness was not softened. Returns (is_ok, feedback)."""
    if llm_instance is None:
        llm = Llama(model_path=llama_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    else:
        llm = llm_instance
    lang_name = target_lang_name(tgt_lang)
    system = (
        "You are a translation quality checker. "
        f"Verify that the {lang_name} translation preserves the tone, explicitness, and intensity of the English original. "
        "Reply ONLY: OK  — if tone is fully preserved, or "
        "SOFTENED: <one-line reason>  — if tone was weakened, sanitized, or intent changed."
    )
    tgt_code = tgt_lang.upper()
    user = f"EN: {original_en}\n{tgt_code}: {translated}"
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=80,
    )
    result = resp["choices"][0]["message"]["content"].strip()
    is_ok = result.upper().startswith("OK")
    return is_ok, result


def translate_segment_nllb_gemma_qa(
    text: str,
    tgt_lang: str,
    nllb_model: str = "facebook/nllb-200-distilled-1.3B",
    nllb_device: str = "cpu",
    gemma_model: str = "",
    gemma_n_ctx: int = 2048,
    gemma_n_gpu_layers: int = 56,
    qa_model: str = "",
    qa_n_ctx: int = 2048,
    qa_n_gpu_layers: int = 32,
    gemma_instance=None,
    qa_instance=None,
) -> tuple[str, bool, str]:
    """
    3-stage translation pipeline:
      A) NLLB-200 → mechanical EN→TGT (no refusals/filtering)
      B) Gemma 27B → language polish (preserve tone exactly)
      C) Llama 8B QA → verify explicitness not lost

    Returns (translated_text, qa_ok, qa_feedback).
    """
    # Stage A: NLLB
    print(f"[NGQ] Stage A: NLLB ({nllb_model})...", flush=True)
    nllb_result = translate_segment_with_nllb(
        text, tgt_lang=tgt_lang, model_name=nllb_model, device=nllb_device
    )

    # Stage B: Gemma cleanup (free NLLB first if on same device)
    if gemma_model:
        if nllb_device == "cuda":
            free_nllb_model()
        print(f"[NGQ] Stage B: Gemma cleanup...", flush=True)
        _lang_cs_map = {
            "Czech": "češtinu", "Slovak": "slovenštinu", "German": "němčinu",
            "French": "francouzštinu", "Spanish": "španělštinu",
            "Polish": "polštinu", "Russian": "ruštinu",
        }
        _tgt_lang_name_cs = _lang_cs_map.get(target_lang_name(tgt_lang), target_lang_name(tgt_lang))
        cleaned = cleanup_with_gemma(
            nllb_result,
            gemma_model,
            n_ctx=gemma_n_ctx,
            n_gpu_layers=gemma_n_gpu_layers,
            llm_instance=gemma_instance,
            tgt_lang_name=_tgt_lang_name_cs,
        )
    else:
        cleaned = nllb_result

    # Stage C: QA
    qa_ok, qa_feedback = True, "skipped"
    if qa_model:
        print(f"[NGQ] Stage C: QA check...", flush=True)
        qa_ok, qa_feedback = qa_check_tone(
            text,
            cleaned,
            qa_model,
            n_ctx=qa_n_ctx,
            n_gpu_layers=qa_n_gpu_layers,
            llm_instance=qa_instance,
            tgt_lang=tgt_lang,
        )
        level = "OK" if qa_ok else "WARNING"
        print(f"[NGQ] QA {level}: {qa_feedback}", flush=True)

    return cleaned, qa_ok, qa_feedback


# ---------------------------------------------------------------------------
# MADLAD → Gemma polish → QA  (3-phase batch pipeline)
# ---------------------------------------------------------------------------

def translate_batch_madlad_gemma_qa(
    segments: list[dict],
    tgt_lang: str,
    madlad_model: str = "google/madlad400-7b-mt",
    gemma_model: str = "",
    gemma_n_ctx: int = 4096,
    gemma_n_gpu_layers: int = 56,
    qa_model: str = "",
    qa_n_ctx: int = 2048,
    qa_n_gpu_layers: int = 32,
    use_google_phase1: bool = False,
    src_lang: str = "en",
) -> list[dict]:
    """
    3-phase batch pipeline (memory-efficient: each model loaded → all segments → freed):

      Phase 1: MADLAD (or Google Translate) translates all segments EN→TGT
      Phase 2: Gemma 27B polishes all translations → free Gemma
      Phase 3: Llama 8B QA checks EN vs final translation → free QA

    Args:
        segments: list of dicts with at least {"text": "...", "start": ..., "end": ...}
        tgt_lang: target language code (e.g. "ces", "slk")
        madlad_model: HuggingFace model name or local path (used when use_google_phase1=False)
        gemma_model: path to Gemma GGUF (empty = skip polish)
        qa_model: path to QA Llama GGUF (empty = skip QA)
        use_google_phase1: if True, use Google Translate for Phase 1 instead of MADLAD

    Returns:
        list of dicts: original segment + added keys:
          "text_translated"  — Phase-1 raw translation
          "text_polished"    — Gemma-polished (= text_translated if gemma skipped)
          "text_final"       — final text to use (= text_polished)
          "qa_ok"            — True / False (True if qa skipped)
          "qa_feedback"      — QA response string ("skipped" if not run)
    """
    results = []

    # ── Phase 1: translate all segments ─────────────────────────────────────
    if use_google_phase1:
        print(f"[MGQ] Phase 1: Google Translate — {len(segments)} segments...", flush=True)
    else:
        print(f"[MGQ] Phase 1: MADLAD ({madlad_model}) — {len(segments)} segments...", flush=True)
    madlad_texts: list[str] = []
    for i, seg in enumerate(segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            madlad_texts.append("")
            continue
        print(f"[MGQ] P1 [{i}/{len(segments)}]", flush=True)
        if use_google_phase1:
            translated = translate_segment_with_google(text, tgt_lang=tgt_lang, src_lang=src_lang)
            if not translated:
                print(f"  [MGQ] Google failed seg {i}, falling back to MADLAD", flush=True)
                translated = translate_segment_with_madlad(text, tgt_lang=tgt_lang, model_name=madlad_model)
        else:
            translated = translate_segment_with_madlad(text, tgt_lang=tgt_lang, model_name=madlad_model)
        madlad_texts.append(translated)

    if not use_google_phase1:
        free_madlad_model()

    # ── Phase 2: Gemma polishes all ──────────────────────────────────────────
    polished_texts: list[str]
    if gemma_model:
        print(f"[MGQ] Phase 2: Gemma polish — {len(segments)} segments...", flush=True)
        gemma_llm = Llama(
            model_path=gemma_model,
            n_ctx=gemma_n_ctx,
            n_gpu_layers=gemma_n_gpu_layers,
            verbose=False,
        )
        polished_texts = []
        lang_name = target_lang_name(tgt_lang)
        # Map lang code to localized name for Czech system prompt
        lang_name_cs_map = {
            "Czech": "češtinu", "Slovak": "slovenštinu", "German": "němčinu",
            "French": "francouzštinu", "Spanish": "španělštinu",
            "Polish": "polštinu", "Russian": "ruštinu",
        }
        tgt_lang_name_cs = lang_name_cs_map.get(lang_name, lang_name)
        for i, text in enumerate(madlad_texts, 1):
            if not text:
                polished_texts.append("")
                continue
            print(f"[MGQ] P2 [{i}/{len(segments)}]", flush=True)
            polished = cleanup_with_gemma(
                text,
                gemma_model,
                n_ctx=gemma_n_ctx,
                n_gpu_layers=gemma_n_gpu_layers,
                llm_instance=gemma_llm,
                tgt_lang_name=tgt_lang_name_cs,
            )
            polished_texts.append(polished)
        del gemma_llm
        gc.collect()
        torch.cuda.empty_cache()
        print("[MGQ] Gemma released", flush=True)
    else:
        print("[MGQ] Phase 2: skipped (no gemma_model)", flush=True)
        polished_texts = list(madlad_texts)

    # ── Phase 3: QA checks all ───────────────────────────────────────────────
    if qa_model:
        print(f"[MGQ] Phase 3: QA — {len(segments)} segments...", flush=True)
        qa_llm = Llama(
            model_path=qa_model,
            n_ctx=qa_n_ctx,
            n_gpu_layers=qa_n_gpu_layers,
            verbose=False,
        )
        for i, (seg, madlad_t, polished_t) in enumerate(zip(segments, madlad_texts, polished_texts)):
            orig = (seg.get("text") or "").strip()
            print(f"[MGQ] P3 [{i + 1}/{len(segments)}]", flush=True)
            if orig and polished_t:
                qa_ok, qa_feedback = qa_check_tone(orig, polished_t, "", llm_instance=qa_llm, tgt_lang=tgt_lang)
                level = "OK" if qa_ok else "SOFTENED ⚠"
                print(f"[MGQ] QA {level}: {qa_feedback}", flush=True)
            else:
                qa_ok, qa_feedback = True, "skipped (empty)"
            results.append({
                **seg,
                "text_translated": madlad_t,
                "text_polished": polished_t,
                "text_final": polished_t,
                "qa_ok": qa_ok,
                "qa_feedback": qa_feedback,
            })
        del qa_llm
        gc.collect()
        torch.cuda.empty_cache()
        print("[MGQ] QA released", flush=True)
    else:
        print("[MGQ] Phase 3: skipped (no qa_model)", flush=True)
        for seg, madlad_t, polished_t in zip(segments, madlad_texts, polished_texts):
            results.append({
                **seg,
                "text_translated": madlad_t,
                "text_polished": polished_t,
                "text_final": polished_t,
                "qa_ok": True,
                "qa_feedback": "skipped",
            })

    softened_count = sum(1 for r in results if not r["qa_ok"])
    print(f"[MGQ] Done. {len(results)} segments, {softened_count} QA warnings.", flush=True)
    return results


# ---------------------------------------------------------------------------
# Text duration utilities
# ---------------------------------------------------------------------------

def estimate_speech_duration(text: str, chars_per_second: float = 11.0) -> float:
    return len(text) / chars_per_second


def fit_text_to_duration(text: str, target_duration: float, llm=None, tgt_lang: str = "sk", stretch_max: float = 1.60) -> str:
    estimated_dur = estimate_speech_duration(text)
    if estimated_dur <= target_duration * stretch_max:
        return text
    ratio = target_duration / estimated_dur
    print(f"    [FIT] keeping original text (hard text cut disabled)", flush=True)
    return text


def cleanup_segment_leading_conjunction(text: str, prev_text: str = "") -> str:
    if not text:
        return text
    stripped = text.strip()
    if stripped == "a":
        if prev_text and prev_text.rstrip().endswith((",", ";", ":")):
            return text
        return ""
    stripped2 = text.lstrip()
    if not stripped2.startswith("a "):
        return text
    if prev_text and prev_text.rstrip().endswith((",", ";", ":")):
        return text
    return stripped2[2:].lstrip()


def cleanup_segment_trailing_conjunction(text: str) -> str:
    if not text:
        return text
    stripped = text.rstrip()
    if re.search(r"(?:^|\s)[Aa]$", stripped):
        stripped = re.sub(r"\s+[Aa]$", "", stripped).rstrip()
    return stripped


# ---------------------------------------------------------------------------
# SRT / JSON export
# ---------------------------------------------------------------------------

def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_srt(segments: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            text = seg.get("text", "").strip()
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n")
            f.write("\n")
    print(f"[SRT] Saved subtitles with timing: {output_path}", flush=True)


def translate_segments_sliding_window(
    segments: list[dict],
    tgt_lang: str,
    engine: str,
    # llama/llm params
    llm_instance=None,
    llama_model: str = "",
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    temperature: float = 0.1,
    # translategemma params
    tg_model_path: str = "",
    # google params (no extra params needed)
    src_lang: str = "en",
    # api params
    api_key: str = "",
    api_base_url: str = "",
    api_model: str = "",
    api_provider: str = "chatgpt",
    # lmstudio params
    lmstudio_url: str = "http://localhost:1234/v1",
    lmstudio_model: str = "",
    lmstudio_timeout: int = 180,
    # shared
    content_type: str = "general",
    glossary: str = "",
    batch_size: int = 200,
    memory_size: int = 20,
    retry_count: int = 3,
) -> list[dict]:
    """
    Preloží segmenty v dávkach s klzavou pamäťou (sliding window).

    Každá dávka obsahuje batch_size segmentov v SRT formáte + posledných
    memory_size preložených segmentov ako pamäť + auto terminology block.

    Validácia: počet segmentov, indexy, timestamps — ak nesedí → retry
    s polovičným batchom (200 → 100 → 50).

    Engines: 'llama', 'translategemma', 'google', 'api'
    """
    import hashlib
    from pathlib import Path as _Path

    lang_name = target_lang_name(tgt_lang)
    results = [dict(s) for s in segments]  # deep-ish copy

    # ── SRT helpers ──────────────────────────────────────────────────────────
    def _ts(sec: float) -> str:
        h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    def _build_srt_block(idx: int, seg: dict, use_translated: bool = False) -> str:
        text = (seg.get("text") or "").strip()
        return f"{idx}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{text}"

    def _parse_srt_response(raw: str) -> list[dict]:
        """Parsuje SRT odpoveď → list[{idx, start_str, end_str, text}]"""
        out = []
        blocks = re.split(r'\n\s*\n', raw.strip())
        for block in blocks:
            lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
            if len(lines) < 3:
                continue
            try:
                idx = int(lines[0])
            except ValueError:
                continue
            ts_m = re.match(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', lines[1])
            if not ts_m:
                continue
            text = " ".join(lines[2:])
            out.append({"idx": idx, "start_str": ts_m.group(1), "end_str": ts_m.group(2), "text": text})
        return out

    def _validate(source_batch: list[dict], translated: list[dict], global_indices: list[int]) -> bool:
        if len(translated) != len(source_batch):
            return False
        for src, trg, gidx in zip(source_batch, translated, global_indices):
            if src["idx"] != trg["idx"]:
                return False
            # Timestamps musia sedieť (tolerancia 1ms)
            if src["start_str"].replace(".", ",") != trg["start_str"].replace(".", ","):
                return False
            if src["end_str"].replace(".", ",") != trg["end_str"].replace(".", ","):
                return False
        return True

    # ── Translation cache ─────────────────────────────────────────────────────
    _CACHE_DIR = _Path(SCRIPT_DIR) / "temp" / "translate_cache"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_key(text: str) -> str:
        cache_scope = "|".join([
            engine or "",
            src_lang or "",
            tgt_lang or "",
            api_provider or "",
            api_base_url or "",
            api_model or "",
            content_type or "",
            glossary or "",
        ])
        return hashlib.md5(f"{cache_scope}|{text}".encode()).hexdigest()

    def _cache_get(text: str):
        p = _CACHE_DIR / (_cache_key(text) + ".txt")
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _cache_set(text: str, translation: str):
        if translation.strip():
            (_CACHE_DIR / (_cache_key(text) + ".txt")).write_text(translation, encoding="utf-8")

    # ── Auto terminology extraction ───────────────────────────────────────────
    term_memory: dict[str, str] = {}  # EN term → preložený ekvivalent

    def _update_terms(src_segs: list[dict], tgt_segs: list[dict]):
        """Extrahuje vlastné mená a opakujúce sa termíny z párov src→tgt."""
        for src, tgt in zip(src_segs, tgt_segs):
            src_words = src.get("text", "").split()
            tgt_words = tgt.get("text", "").split()
            for w in src_words:
                # Vlastné meno = veľké písmeno, nie začiatok vety, nie v slovníku
                if len(w) > 2 and w[0].isupper() and not re.match(r'^[A-Z]{2,}$', w) and w not in term_memory:
                    # Hľadaj rovnaké slovo v cieľovom texte
                    clean = re.sub(r'[^a-zA-Z]', '', w)
                    if clean in tgt.get("text", ""):
                        term_memory[clean] = clean  # zachované
                    elif len(tgt_words) == len(src_words):
                        idx2 = src_words.index(w)
                        if idx2 < len(tgt_words):
                            term_memory[clean] = re.sub(r'[^a-záéíóúäěšžýřčůňľščťžýĺŕóúäé]', '', tgt_words[idx2].lower())

    def _build_term_block() -> str:
        if not term_memory:
            return ""
        lines = [f"- {k} = {v}" for k, v in list(term_memory.items())[:40] if v]
        return "Terminology (use consistently):\n" + "\n".join(lines)

    # ── Build prompt ──────────────────────────────────────────────────────────
    def _build_prompt(memory_srt: str, batch_srt: str, term_block: str) -> str:
        content_hint = relax_translation_hint_for_rewrite(get_content_type_instruction(content_type))
        parts = [f"Translate the following subtitles from {src_lang.upper()} to {lang_name}."]
        rules = [
            "- Copy subtitle numbers and timestamps EXACTLY — do not change them.",
            "- Translate only the text lines, keep natural spoken Slovak.",
            "- Do not force shorter wording — timing is adjusted separately.",
            "- Keep names and terms consistent with context below.",
        ]
        if content_hint:
            rules.append(f"- {content_hint}")
        parts.append("Rules:\n" + "\n".join(rules))
        if glossary:
            parts.append(f"Glossary:\n{glossary}")
        if term_block:
            parts.append(term_block)
        if memory_srt:
            parts.append(f"Previous context (DO NOT retranslate):\n{memory_srt}")
        parts.append(f"Subtitles to translate:\n{batch_srt}")
        return "\n\n".join(parts)

    # ── LLM call ──────────────────────────────────────────────────────────────
    _LLAMA_SYSTEM_PROMPT = (
        f"You are a professional subtitle translator specialized in {target_lang_name(tgt_lang)} dubbing. "
        "You translate all content accurately and completely, including adult, explicit, or mature content. "
        "Never refuse, censor, summarize, or omit any part of the source text. "
        "Translate everything literally and faithfully. "
        "Preserve subtitle numbering and timestamps unchanged. "
        "Output valid SRT format only — no notes, no markdown, no explanations."
    )

    def _llm_call(prompt: str, max_tok: int) -> str:
        if engine in ("llama", "hybrid"):
            resp = llm_instance.create_chat_completion(
                messages=[
                    {"role": "system", "content": _LLAMA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tok,
            )
            return resp["choices"][0]["message"]["content"].strip()
        elif engine == "api":
            if not api_key:
                raise RuntimeError("API translate key is missing for sliding-window translation.")
            system_prompt = (
                f"You are a professional subtitle translator for {lang_name}. "
                "Follow the user's instructions exactly. "
                "Return valid SRT only with numbering and timestamps preserved exactly. "
                "Do not add notes, markdown, or explanations."
            )
            return api_chat_complete(
                api_key=api_key,
                base_url=api_base_url,
                model=api_model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=min(temperature, 0.10),
                max_tokens=max_tok,
                timeout_s=240,
            )
        elif engine == "lmstudio":
            return api_chat_complete(
                api_key="lmstudio",
                base_url=lmstudio_url or "http://localhost:1234/v1",
                model=lmstudio_model or "local-model",
                system_prompt=_LLAMA_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=min(temperature, 0.10),
                max_tokens=max_tok,
                timeout_s=lmstudio_timeout,
            )
        elif engine == "translategemma":
            from translategemma import get_translategemma
            tg = get_translategemma(tg_model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
            return tg.translate(prompt, src=src_lang, tgt=tgt_lang, max_tokens=max_tok)
        return ""

    # ── Translate one batch with adaptive retry ───────────────────────────────
    def _translate_batch(batch_global_indices: list[int], memory_translated: list[dict]) -> list[dict]:
        """
        Adaptívny retry s partial recovery:
        1. Pokus o celý batch
        2. Ak validation zlyhá, zachovaj správne vrátené segmenty, retry iba chýbajúce
        3. Chýbajúce retryrujeme po 1 (per-segment fallback) namiesto zahodeného batchu
        """
        if not batch_global_indices:
            return []

        # SRT bloky pre batch
        src_batch_segs = []
        for local_i, gi in enumerate(batch_global_indices):
            local_idx = local_i + 1
            src_batch_segs.append({
                "idx": local_idx,
                "start_str": _ts(results[gi]["start"]),
                "end_str": _ts(results[gi]["end"]),
                "text": (results[gi].get("text") or "").strip(),
                "global_idx": gi,
            })

        # Memory SRT
        mem_srt = "\n\n".join(
            f"{s['local_idx']}\n{s['start_str']} --> {s['end_str']}\n{s['text']}"
            for s in memory_translated[-memory_size:]
        ) if memory_translated else ""

        batch_srt = "\n\n".join(
            f"{s['idx']}\n{s['start_str']} --> {s['end_str']}\n{s['text']}"
            for s in src_batch_segs
        )

        term_block = _build_term_block()
        prompt = _build_prompt(mem_srt, batch_srt, term_block)
        max_tok = min(8192, len(batch_global_indices) * 100)

        for attempt in range(retry_count):
            try:
                raw = _llm_call(prompt, max_tok)
            except Exception as e:
                print(f"[SW-TRANS] LLM error attempt {attempt}: {e}", flush=True)
                continue

            parsed = _parse_srt_response(raw)

            if _validate(src_batch_segs, parsed, batch_global_indices):
                for seg_out, src_seg in zip(parsed, src_batch_segs):
                    seg_out["global_idx"] = src_seg["global_idx"]
                    seg_out["local_idx"] = src_seg["idx"]
                return parsed

            got = len(parsed)
            exp = len(src_batch_segs)
            print(f"[SW-TRANS] Validation failed attempt {attempt}: got {got}/{exp} segs", flush=True)

            # Partial recovery: zachovaj segmenty so správnym idx+timestamp
            parsed_by_idx = {p["idx"]: p for p in parsed}
            recovered: list[dict] = []
            missing_globals: list[int] = []
            for src_seg in src_batch_segs:
                candidate = parsed_by_idx.get(src_seg["idx"])
                if (candidate and
                        candidate["start_str"].replace(".", ",") == src_seg["start_str"].replace(".", ",") and
                        candidate["end_str"].replace(".", ",") == src_seg["end_str"].replace(".", ",")):
                    candidate["global_idx"] = src_seg["global_idx"]
                    candidate["local_idx"] = src_seg["idx"]
                    recovered.append(candidate)
                else:
                    missing_globals.append(src_seg["global_idx"])

            if missing_globals and len(missing_globals) < len(batch_global_indices):
                print(f"[SW-TRANS] Partial recovery: {len(recovered)} ok, {len(missing_globals)} missing → retry individually", flush=True)
                # Retry chýbajúcich po 1 (per-segment)
                for gi in missing_globals:
                    retry_result = _translate_batch([gi], memory_translated + recovered)
                    recovered.extend(retry_result)
                # Zoraď podľa pôvodného poradia
                gi_order = {gi: i for i, gi in enumerate(batch_global_indices)}
                recovered.sort(key=lambda s: gi_order.get(s["global_idx"], 9999))
                return recovered

        # Fallback: vráť originály pre celý batch
        print(f"[SW-TRANS] Fallback to original for {len(batch_global_indices)} segs", flush=True)
        return [
            {"idx": s["idx"], "start_str": s["start_str"], "end_str": s["end_str"],
             "text": s["text"], "global_idx": s["global_idx"], "local_idx": s["idx"]}
            for s in src_batch_segs
        ]

    # ── Google: per-segment s cache ───────────────────────────────────────────
    def _translate_google_batch(batch_global_indices: list[int]) -> list[dict]:
        out = []
        for gi in batch_global_indices:
            orig = (results[gi].get("text") or "").strip()
            cached = _cache_get(orig)
            if cached:
                out.append({"global_idx": gi, "text": cached})
                continue
            tr = orig
            _protected, _tok_map = protect_phonetic_terms(orig, use_guillemets=True)
            for attempt in range(retry_count):
                try:
                    _tr_raw = translate_segment_with_google(_protected, tgt_lang, src_lang)
                    tr = strip_guillemets(restore_phonetic_terms(_tr_raw, _tok_map))
                    break
                except Exception as e:
                    if attempt == retry_count - 1:
                        print(f"[SW-TRANS] Google fail seg {gi}: {e}", flush=True)
            _cache_set(orig, tr)
            out.append({"global_idx": gi, "text": tr})
        return out

    # ── Pre-cache check ───────────────────────────────────────────────────────
    uncached = []
    for i, seg in enumerate(results):
        orig = (seg.get("text") or "").strip()
        if not orig:
            continue
        cached = _cache_get(orig)
        if cached:
            results[i]["text"] = cached
        else:
            uncached.append(i)

    cached_count = len(results) - len(uncached)
    print(f"[SW-TRANS] Total={len(results)}, cache_hits={cached_count}, to_translate={len(uncached)}", flush=True)

    if not uncached:
        return results

    # ── Main loop ─────────────────────────────────────────────────────────────
    batches = [uncached[i:i+batch_size] for i in range(0, len(uncached), batch_size)]
    memory_translated: list[dict] = []  # akumulované preložené segmenty pre pamäť

    for batch_num, batch_indices in enumerate(batches):
        print(f"[SW-TRANS] Batch {batch_num+1}/{len(batches)}: "
              f"segs {batch_indices[0]}–{batch_indices[-1]} "
              f"(n={len(batch_indices)}, memory={len(memory_translated)})", flush=True)

        if engine == "google":
            batch_out = _translate_google_batch(batch_indices)
            for item in batch_out:
                gi = item["global_idx"]
                results[gi]["text"] = item["text"]
                memory_translated.append({
                    "local_idx": len(memory_translated) + 1,
                    "start_str": _ts(results[gi]["start"]),
                    "end_str": _ts(results[gi]["end"]),
                    "text": item["text"],
                })
        else:
            batch_out = _translate_batch(batch_indices, memory_translated)
            src_segs_for_terms = []
            tgt_segs_for_terms = []
            for item in batch_out:
                gi = item.get("global_idx", batch_indices[batch_out.index(item)])
                tr = item.get("text", "").strip()
                orig = (results[gi].get("text") or "").strip()
                results[gi]["text"] = tr
                _cache_set(orig, tr)
                src_segs_for_terms.append({"text": orig})
                tgt_segs_for_terms.append({"text": tr})
                memory_translated.append({
                    "local_idx": len(memory_translated) + 1,
                    "start_str": item.get("start_str", _ts(results[gi]["start"])),
                    "end_str": item.get("end_str", _ts(results[gi]["end"])),
                    "text": tr,
                })
            _update_terms(src_segs_for_terms, tgt_segs_for_terms)

        if term_memory:
            print(f"[SW-TRANS] Terminology memory: {len(term_memory)} terms", flush=True)

    return results


def save_json_segments(segments: list[dict], output_path: Path) -> None:
    output_data = {
        "num_segments": len(segments),
        "segments": [
            {
                "index": i + 1,
                "start": seg["start"],
                "end": seg["end"],
                "start_formatted": format_srt_time(seg["start"]),
                "end_formatted": format_srt_time(seg["end"]),
                "duration": seg["end"] - seg["start"],
                "text": seg.get("text", "").strip()
            }
            for i, seg in enumerate(segments)
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def _is_valid_corrector_output(corrected: str, original: str) -> bool:
    """Validate corrector output: real correction vs hallucinated explanation.

    Pravidlá:
    1. Nesmie byť prázdny / príliš krátky
    2. Nesmie obsahovať markdown / explanácie (* _ # • bullet markers)
    3. Output dĺžka: 50%-150% pôvodnej (mierne kratšie OK, výrazne dlhšie = explanácia)
    4. Word overlap >= 40% (zachováva obsah, neha lucinuje nový)
    5. Nesmie obsahovať explicit "Tu je", "Správne je", "Možnosť" headers
    """
    if not corrected or len(corrected) < 3:
        return False
    if not original:
        return False

    # Reject markdown / structured explanations
    bad_starts = ("*", "_", "#", "•", "- ", "1.", "2.", "3.")
    if corrected.lstrip().startswith(bad_starts):
        return False

    # Reject explicit explanation phrases (anywhere in output)
    bad_phrases = ("Tu je správna", "Správne by malo", "Možnosť 1", "Možnosť 2",
                   "(opravený", "(oprava", "Najlepšie je", "Praví sa, že")
    for phrase in bad_phrases:
        if phrase.lower() in corrected.lower():
            return False

    # Length sanity: 50%-150% of original
    ratio = len(corrected) / len(original)
    if ratio < 0.5 or ratio > 1.5:
        return False

    # Char-level similarity check (Levenshtein-based, vhodné pre declension)
    # Stem comparison: porovná prvé 4 znaky každého slova (zachycuje tvary jedného koreňa)
    import re as _re
    def _stem_set(s):
        return set(w.lower()[:4] for w in _re.findall(r'\w+', s) if len(w) >= 2)
    orig_stems = _stem_set(original)
    corr_stems = _stem_set(corrected)
    if not orig_stems:
        return True
    common = orig_stems & corr_stems
    overlap = len(common) / len(orig_stems)
    # Pre krátky text (≤4 slová) povoľ menšie overlap (slová sa môžu meniť drasticky pri skloňovaní)
    threshold = 0.30 if len(orig_stems) <= 4 else 0.50
    return overlap >= threshold


def refine_with_sk_corrector(
    segments: list,
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "sk-corrector-v1",
    timeout: int = 30,
) -> list:
    """
    SK→SK grammar corrector pass cez ollama (sk-corrector-v1).
    Opraví skloňovanie/časovanie v každom segmente jednotlivo.
    """
    result = [dict(s) for s in segments]
    changed = 0
    for i, seg in enumerate(result):
        text = (seg.get("text") or "").strip()
        if not text or text == "...":
            continue
        try:
            resp = requests.post(
                ollama_url,
                json={"model": model, "prompt": f"Oprav slovenský titulok: {text}", "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            corrected = resp.json().get("response", "").strip()
            # Validácia: chceme zachovať **skutočné opravy** ale odmietnuť explanácie/halucinácie
            if _is_valid_corrector_output(corrected, text):
                if corrected != text:
                    changed += 1
                result[i]["text"] = corrected
        except Exception as _e:
            print(f"[SK-CORRECTOR] seg {i} chyba: {_e}", flush=True)
    print(f"[SK-CORRECTOR] Opravených: {changed}/{len(segments)} segmentov.", flush=True)
    return result


# ── Compression: D (deterministic filler removal) ──────────────────────────────
_FILLER_PATTERNS = [
    # Filler frázy — bezpečné odstrániť, nestrácajú význam
    (r',?\s+úprimne\s+povedané,?\s*',          ' '),
    (r',?\s+naozaj,?\s*',                      ' '),
    (r',?\s+vlastne,?\s*',                     ' '),
    (r',?\s+teda,?\s*',                        ' '),
    (r',?\s+v\s+skutočnosti,?\s*',             ' '),
    (r',?\s+povedzme\s+si,?\s*',               ' '),
    (r',?\s+dalo\s+by\s+sa\s+povedať,?\s*',    ' '),
    (r',?\s+v\s+podstate,?\s*',                ' '),
    (r',?\s+samozrejme,?\s*',                  ' '),
    (r',?\s+akoby,?\s*',                       ' '),
    # Redundantné intenzifikátory
    (r'\bveľmi\s+veľmi\b',                     'veľmi'),
    (r'\bnaozaj\s+naozaj\b',                   'naozaj'),
    # Zdvojené spojky
    (r'\ba\s+a\b',                             'a'),
    (r'\bale\s+ale\b',                         'ale'),
    # "tak potom" → "potom"
    (r'\btak\s+potom\b',                       'potom'),
    # "no tak" na začiatku vety
    (r'^\s*no\s+tak,?\s*',                     ''),
    # whitespace cleanup
    (r'\s{2,}',                                ' '),
    (r'\s+([,.!?])',                           r'\1'),
]

def compress_deterministic(text: str, target_len: int = None) -> tuple[str, int]:
    """Odstráni filler slová. Vráti (shortened_text, chars_saved).
    Ak target_len zadané a text už sedí, nič neurobí."""
    import re as _re
    if not text or not text.strip():
        return text, 0
    if target_len is not None and len(text) <= target_len:
        return text, 0
    orig_len = len(text)
    out = text
    for pat, repl in _FILLER_PATTERNS:
        out = _re.sub(pat, repl, out, flags=_re.IGNORECASE)
    out = out.strip()
    return out, orig_len - len(out)


# ── Compression: C (26B LM Studio for overflow) ───────────────────────────────
def compress_with_lmstudio(
    text: str,
    target_len: int,
    lmstudio_url: str = "http://localhost:1234/v1",
    lmstudio_model: str = "",
    timeout: int = 60,
) -> str:
    """Skráti text na target_len znakov cez 26B LM Studio.
    Volať len pre overflow segmenty — je to drahá operácia."""
    if not text or len(text) <= target_len:
        return text

    _effective_model = lmstudio_model
    if not _effective_model:
        try:
            _mresp = requests.get(
                lmstudio_url.rstrip("/") + "/models", timeout=5,
            ).json()
            _llm_models = [
                m["id"] for m in _mresp.get("data", [])
                if "embed" not in m.get("id", "").lower()
            ]
            if _llm_models:
                _effective_model = _llm_models[0]
        except Exception:
            _effective_model = "local-model"

    system_prompt = (
        "Si profesionálny editor slovenských dabingových titulkov. "
        "Tvoja úloha je skrátiť text na presnú cieľovú dĺžku pri zachovaní kľúčového zmyslu. "
        "Zachovaj všetky mená, čísla a technické pojmy. "
        "Odstráň filler frázy, zdvojenia, nadbytočné príklady. "
        "Vráť IBA skrátený text, bez vysvetlení, bez úvodzoviek."
    )
    user_prompt = (
        f"Skráť tento slovenský titulok na {target_len} znakov (tolerancia ±10%). "
        f"Originál má {len(text)} znakov.\n\n"
        f"Titulok:\n{text}"
    )

    try:
        resp = requests.post(
            lmstudio_url.rstrip("/") + "/chat/completions",
            json={
                "model": _effective_model or "local-model",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max(64, int(target_len * 1.5)),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"].strip()
        # Odstráň prípadné úvodzovky
        out = out.strip('"\'«»""')
        # Odmietni ak LLM vrátil explanations
        if out.startswith(("Samozrejme,", "Tu je", "Skrátený", "Výsledný", "Titulok:")):
            return text  # nezhoršuj
        # Odmietni ak je príliš krátky (stratil by zmysel)
        if len(out) < target_len * 0.5:
            return text
        # Odmietni ak je dlhší než originál (LLM nekrátil)
        if len(out) >= len(text):
            return text
        return out
    except Exception as _e:
        print(f"[COMPRESS-26B] chyba: {_e}", flush=True)
        return text


def compress_text_pipeline(
    text: str,
    target_len: int,
    use_lmstudio_fallback: bool = True,
    lmstudio_url: str = "http://localhost:1234/v1",
    lmstudio_model: str = "",
) -> tuple[str, str]:
    """Dvojfázová kompresia:
    1. Deterministické odstránenie filler slov (D)
    2. Ak stále overflow a use_lmstudio_fallback=True → 26B (C)

    Vráti (compressed_text, method_used)."""
    if len(text) <= target_len:
        return text, "noop"

    # Fáza D
    out, saved = compress_deterministic(text, target_len)
    if len(out) <= target_len:
        return out, "deterministic"

    # Fáza C (voliteľná)
    if use_lmstudio_fallback:
        out2 = compress_with_lmstudio(out, target_len,
                                       lmstudio_url=lmstudio_url,
                                       lmstudio_model=lmstudio_model)
        if len(out2) < len(out):
            return out2, "lmstudio"

    return out, "deterministic_partial"
