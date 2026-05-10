"""
sk_glossary.py — Centrálny SK glossary (jediný zdroj pravdy).

Tri výstupy:
  1. get_llm_prompt_block()  → vloží sa do systém promptu EuroLLM / Gemma
  2. get_postfix_rules()     → (regex, replacement, flags) zoznam pre post-fix vrstvu
  3. get_audit_bad_terms()   → (regex, flag_name) zoznam pre audit_sk_tts_json.py
"""

import re

# ---------------------------------------------------------------------------
# MASTER GLOSSARY
# Formát: (en_term, sk_term, regex_pattern_sk_bad, is_case_insensitive)
#
#  en_term         — anglický zdrojový termín (pre LLM prompt)
#  sk_term         — správny slovenský preklad
#  regex_bad_sk    — regex ktorý detekuje ZLÝ preklad v SK texte
#                    None = odvodiť automaticky z en_term (verbatim match)
#  case_insensitive — True/False pre regex
# ---------------------------------------------------------------------------

_GLOSSARY_ENTRIES = [
    # (en_term, sk_term, regex_bad_sk, case_insensitive)

    # Hardware
    ("power supply",            "napájací zdroj",
     r'\b(napájan[ie]+|zásob[au]?\s+energie|power\s+supply)\b', True),

    ("motherboard",             "základná doska",
     r'\bmother\s*board\b', True),

    ("CPU",                     "CPU",
     None, False),

    ("RAM",                     "RAM",
     None, False),

    ("SFX power supply",        "SFX napájací zdroj",
     r'\bSFX\s+napájan\w*\b', True),

    # Proxmox / virtualizácia
    ("no subscription repository",  "repozitár bez predplatného",
     r'\bno[- ]?subscription\s+repositor\w*|Prize\s+Repository\b', True),

    ("repository",              "repozitár",
     r'\brepository\b', True),

    ("cluster",                 "klaster",
     None, True),

    ("snapshot",                "snímka",
     None, True),

    ("backup",                  "záloha",
     None, True),

    ("virtual machine",         "virtuálny stroj",
     r'\bvirtual\s+machine\b', True),

    ("node",                    "uzol",
     None, True),

    ("datacenter",              "dátové centrum",
     r'\bdatacenter\b', True),

    # Sieť
    ("firewall",                "firewall",
     None, False),

    ("hostname",                "hostname",
     r'\bnázov\s+hostiteľa\b', True),

    ("IP address",              "IP adresa",
     r'\bIP\s+addres[sa]\b', True),

    ("gateway",                 "brána",
     None, True),

    # Média / inštalácia
    ("flash",                   "nahrať obraz",
     r'\bflashovani[ea]\b', True),

    ("ISO image",               "ISO obraz",
     r'\bISO\s+image\b', True),

    ("USB drive",               "USB kľúč",
     r'\bUSB\s+driv[e]?\b', True),

    # Video timestamps
    ("time code",               "časová značka",
     r'\bčasov[ýého]+\s+kód\w*|timecod\w*\b', True),

    ("timecode",                "časová značka",
     r'\btimecod\w*\b', True),

    # Iné technické
    ("build",                   "zostava",
     None, True),

    ("server management utility", "nástroj na správu servera",
     r'\bserver\s+management\s+utility\b', True),

    # C++ / game development
    ("game engine",             "herný engine",
     r'\bherný\s+motor\b', True),

    ("graphics engine",         "grafický engine",
     r'\bgrafický\s+motor\b', True),

    ("physics engine",          "fyzikálny engine",
     r'\bfyzikálny\s+motor\b', True),

    ("voxel engine",            "voxel engine",
     r'\bvoxel\w*\s+motor\b', True),

    ("rendering engine",        "renderovací engine",
     r'\brenderovac[íi]\s+motor\b', True),

    ("shader",                  "shader",
     r'\bshader\b', False),

    ("shaders",                 "shadery",
     r'\bshaders\b', False),

    ("pathfinding",             "pathfinding",
     None, False),

    ("voxel",                   "voxel",
     None, False),

    ("food chain",              "potravinový reťazec",
     r'\breťaz\w{0,3}\s+potravy\b', True),

    ("low level programming",   "nízkoúrovňové programovanie",
     r'\bprogramovanie\s+na\s+nízkej\s+úrovni\b', True),

    ("console application",     "konzoliová aplikácia",
     None, True),

    ("graphical application",   "grafická aplikácia",
     r'\bgrafick\w+\s+aplikáci\w+\b', True),

    ("OpenGL",                  "OpenGL",
     None, False),

    ("DirectX",                 "DirectX",
     None, False),

    ("SDL",                     "SDL",
     None, False),
]


def get_llm_prompt_block(tgt_lang: str = "sk") -> str:
    """Vráti len glossary block (pre backward-compat)."""
    if tgt_lang not in ("sk", "slk"):
        return ""
    lines = ["PROJECT GLOSSARY (MANDATORY):"]
    for en, sk, _, _ in _GLOSSARY_ENTRIES:
        lines.append(f"- {en} = {sk}")
    return "\n".join(lines)


_CONTENT_TYPE_CONTEXTS = {
    "general": (
        "You are translating segments from a continuous video.\n"
        "Keep style and terminology consistent across segments.\n"
        "Prefer natural spoken Slovak."
    ),
    "technical": (
        "You are translating segments from a technical tutorial video.\n"
        "Keep style and terminology consistent. Do not translate code, commands, or identifiers.\n"
        "If ambiguous, prefer the interpretation consistent with hardware, Linux, "
        "Proxmox, installation, and server setup."
    ),
    "programming": (
        "You are translating segments from a programming/game development tutorial.\n"
        "Key terminology rules:\n"
        "- 'engine' in programming context MUST stay as 'engine' (NOT 'motor', NOT 'enžin')\n"
        "- 'game engine' = 'herný engine', 'graphics engine' = 'grafický engine', "
        "'physics engine' = 'fyzikálny engine'\n"
        "- 'shader' = 'shader' (NOT 'tieňovač', NOT 'šablóna')\n"
        "- 'era' = 'éra' (NOT 'chyba' which means error)\n"
        "- Keep in English: DirectX, OpenGL, SDL, C++, pathfinding, voxel, framework, mesh, vertex, collider, fragment\n"
        "- 'food chain' = 'potravinový reťazec'"
    ),
    "educational": (
        "You are translating segments from an educational video.\n"
        "Keep style clear and structured. Preserve lists, steps, and definitions.\n"
        "Keep terminology consistent throughout the video."
    ),
    "sql": (
        "You are translating segments from a SQL/database tutorial.\n"
        "Keep ALL SQL keywords and function names in English exactly as-is.\n"
        "Do not translate table names, column names, or SQL identifiers."
    ),
}


def build_translation_system_prompt(
    tgt_lang: str = "sk",
    context_hint: str = "",
    slot_seconds: float = 0.0,
    content_type: str = "general",
) -> str:
    """
    Generuje kompletný systém prompt pre EuroLLM/Gemma prekladový engine.
    Obsahuje: pravidlá, glossary, forbidden variants, dubbing constraints, kontext.
    Vráti prázdny string ak tgt_lang nie je SK/SLK.
    """
    if tgt_lang not in ("sk", "slk"):
        return ""

    # --- Glossary block ---
    glossary_lines = ["PROJECT GLOSSARY (MANDATORY):"]
    for en, sk, _, _ in _GLOSSARY_ENTRIES:
        glossary_lines.append(f"- {en} = {sk}")
    glossary_block = "\n".join(glossary_lines)

    # --- Forbidden variants ---
    forbidden_block = (
        "Forbidden wrong variants:\n"
        '- "zásoba energie" for "power supply" is forbidden\n'
        '- "časový kód" for "timecode" / "time code" is forbidden\n'
        '- "Prize Repository" is forbidden\n'
        '- "nazov hostitela" for "hostname" is forbidden'
    )

    # --- Context block ---
    if context_hint:
        context_block = f"CONTEXT:\n{context_hint}"
    else:
        _default_ctx = _CONTENT_TYPE_CONTEXTS.get(content_type, _CONTENT_TYPE_CONTEXTS["general"])
        context_block = f"CONTEXT:\n{_default_ctx}"

    # --- Slot constraint (voliteľné) ---
    slot_line = (
        f"\nSLOT CONSTRAINT: Keep the translation concise enough to fit ~{slot_seconds:.1f}s when spoken naturally."
        if slot_seconds > 0 else ""
    )

    return f"""\
You are a professional English-to-Slovak translation engine for video dubbing.

Translate the English source text into natural, fluent, spoken Slovak suitable for TTS dubbing.

Rules:
- Preserve the original meaning.
- Prefer natural spoken Slovak over literal translation.
- Keep the translation concise and easy to speak aloud.
- Do not add explanations, notes, or extra information.
- Do not use quotation marks around ordinary words unless they are clearly necessary.
- If a required glossary term appears in the source, you MUST use the exact Slovak glossary equivalent.
- Do not invent alternative translations for glossary terms.
- Keep technical terminology consistent across segments.
- Output only the Slovak translation.{slot_line}

{glossary_block}

{forbidden_block}

DUBBING CONSTRAINTS:
- Prefer shorter phrasing when possible.
- Avoid unnatural literal calques from English.
- Avoid repeated fragments.
- Avoid unfinished clauses.
- Avoid awkward Slovak word order.
- The text should sound natural when spoken aloud by a Slovak voice.

{context_block}"""


def get_postfix_rules() -> list[tuple]:
    """
    Vráti zoznam (compiled_regex, replacement) pre post-fix vrstvu.
    Aplikuj na SK text PO preklade.
    Pokrýva nominatív — ostatné pády treba doplniť ručne kde je to potrebné.
    """
    rules = []

    # Špeciálne pádové pravidlá pre časový kód
    _timecode_cases = [
        (r'\bčasový kód\b',    'časová značka'),
        (r'\bčasového kódu\b', 'časovej značky'),
        (r'\bčasovému kódu\b', 'časovej značke'),
        (r'\bčasovom kóde\b',  'časovej značke'),
        (r'\bčasovým kódom\b', 'časovou značkou'),
        (r'\btimecod\w+\b',    'časová značka'),
    ]
    for pat, repl in _timecode_cases:
        rules.append((re.compile(pat, re.IGNORECASE), repl))

    # Pádové pravidlá pre napájací zdroj
    _power_cases = [
        (r'\bzásob[au]?\s+energie\b',      'napájací zdroj'),
        (r'\bnapájani[ae]\b(?!\s+zdroj)',   'napájací zdroj'),
    ]
    for pat, repl in _power_cases:
        rules.append((re.compile(pat, re.IGNORECASE), repl))

    # Jednoduché preklady z glossary
    _simple = [
        (r'\bPrize\s+Repository\b',         'repozitár bez predplatného'),
        (r'\bno[- ]?subscription\s+repositor\w+', 'repozitár bez predplatného'),
        (r'\bnázov\s+hostiteľa\b',           'hostname'),
        (r'\bserver\s+management\s+utility\b', 'nástroj na správu servera'),
        (r'\bflashovani[ae]\b',              'nahrávaní obrazu'),
        (r'\bflashovať\b',                   'nahrať obraz'),
        (r'\brepository\b',                  'repozitár'),
        (r'\bdatacenter\b',                  'dátové centrum'),
        # C++ / game dev — engine zostáva v EN (NIE motor, NIE enžin)
        # Preferujeme anglický termín "engine" — pôvodný odborný výraz, lepšie pre TTS
        (r'\bherné\s+motor\w*\b',            'herné enginy'),
        (r'\bherný\s+motor\b',               'herný engine'),
        (r'\bherného\s+motora?\b',           'herného engineu'),
        (r'\bherným\s+motorom\b',            'herným engineom'),
        (r'\bgrafický\s+motor\b',            'grafický engine'),
        (r'\bgrafického\s+motora?\b',        'grafického engineu'),
        (r'\bgrafické\s+motor\w*\b',         'grafické enginy'),
        (r'\bfyzikálny\s+motor\b',           'fyzikálny engine'),
        (r'\bfyzikálneho\s+motora?\b',       'fyzikálneho engineu'),
        (r'\bfyzikálne\s+motor\w*\b',        'fyzikálne enginy'),
        (r'\bvoxel\w*\s+motor\b',            'voxel engine'),
        (r'\bvoxel\w*\s+motora?\b',          'voxel engineu'),
        (r'\brenderovac[íi]\s+motor\b',      'renderovací engine'),
        # Force-replace zvyšný "motor" → "engine" v jednotlivých prípadoch
        # (LLM ho dáva sólo: "Motor nebežal", "Tieto dva motory", "začíname s grafickým motorom")
        # Konzervatívne: iba keď nie je v aute kontextu (auto/auta/automobil)
        (r'\bMotor\s+nebež',                 'Engine nebež'),
        (r'\bgrafickým\s+motorom\b',         'grafickým engineom'),
        (r'\bTieto\s+dva\s+motory\b',        'Tieto dva enginy'),
        (r'\bvlastné\s+motor[yi]\b',         'vlastné enginy'),
        (r'\bvlastné\s+programy,\s+motory\b','vlastné programy, enginy'),
        # Phonetic enžin → engine (legacy z predošlých postfix pravidiel)
        (r'\benžin\b',                       'engine'),
        (r'\benžinu\b',                      'engineu'),
        (r'\benžiny\b',                      'enginy'),
        (r'\benžinom\b',                     'engineom'),
        (r'\benžine\b',                      'engineu'),
        # food chain — iba striktný nominatív, iné pády necháme LLM-u
        (r'\breťaz\s+potravy\b',              'potravinový reťazec'),
        # shader — anglický tvar ostáva keď ho TTS neporozumie; inak shader
        (r'\bshader\b',                      'shader'),
        (r'\bshaders\b',                     'shadery'),
        # Whisper STT chyba: SDL sa počuje ako SDN
        (r'\bSDN\b(?=\s*[.,]|\s+čo|\s+ktorý|\s+mi|\s+je|\s+sa)', 'SDL'),
        # "nothing" preložené ako programátorský null — oprava na prirodzené "nič"
        # Vzor: "hodnoty null", "hodnota null", "hodnotu null", "hodnotou null"
        # Toto je VŽDY chyba — v SQL sa nikdy nevyskytuje "hodnot* null" ako spojenie
        (r'\bhodnot\w+\s+[Nn][Uu][Ll][Ll]\b', 'nič'),
        # "null value" ak ostalo v angličtine (prirodzený jazyk, nie SQL)
        (r'\bnull\s+value\b',                'žiadna hodnota'),
        # Gramatická oprava: "nula" je ženský rod → genitív "nuly", nie "nula"
        # "od absolútneho nula" → "od absolútnej nuly"  (mužský rod → ženský)
        # "od úplného nula" → "od absolútnej nuly"
        (r'\bod\s+absolútneho\s+nula\b', 'od absolútnej nuly'),
        (r'\bod\s+úplného\s+nula\b',     'od absolútnej nuly'),
        (r'\bod\s+doslova\s+nula\b',     'od absolútnej nuly'),
        # Filler frazy — prekladové artefakty, v dabingu znejú neprirodzene
        (r',?\s*práve\s+tu([,.]?)$',         r'\1'),
        (r',?\s*práve\s+tu([,.]?)\s',        r'\1 '),
        (r',?\s*práve\s+here([,.]?)$',       r'\1'),
        # "Určite sa pýtate" — TTS model ho mispronuncuje, nahradiť prirodzenejšou frázou
        (r'\bUrčite\s+sa\s+pýtate\b',        'Možno sa pýtate'),
        (r'\bUrčite\s+sa\s+spýtate\b',       'Možno sa pýtate'),
        (r'\bUrčite\s+si\s+pozrite\b',       'Pozrite si'),
        (r'\bUrčite\s+pozrite\b',            'Pozrite'),
        (r'^\s*Určite,\s*',                  'Samozrejme, '),
        (r'^\s*Určite\s+',                   'Samozrejme '),
        # ── Chyby špecifické pre 26B/27B preklad (zistené z C++ videa) ──────────────
        # "absolútnej NULL hodnoty" — model kombinuje "od nuly" s "NULL hodnota"
        # Rozšírené 2026-05-08: case-insensitive + všetky pády + "od X NULL hodnoty"
        (r'(?i)\babsolútnej\s+null\s+hodnot\w*\b',         'absolútnej nuly'),
        (r'(?i)\bod\s+null\s+hodnot\w+\b',                 'od nuly'),
        (r'(?i)\bod\s+(?:absolútneho|úplného)\s+null\b',   'od absolútnej nuly'),
        (r'(?i)\bnull\s+hodnot\w+\b',                       'nuly'),
        (r'(?i)\bhodnot\w+\s+null\b',                       'nič'),
        # "touched" v programming kontexte = "učil/učila", nie literálne "dotkol/dotkla"
        # ("never touched C++" → "nikdy som sa ho NEUČIL/NEUČILA", nie "nedotkol")
        (r'\bnikdy\s+som\s+sa\s+(ho|ju|toho|tomu)\s+nedotkla\b',
            r'nikdy som sa \1 neučila'),
        (r'\bnikdy\s+som\s+sa\s+(ho|ju|toho|tomu)\s+nedotkol\b',
            r'nikdy som sa \1 neučil'),
        (r'\bnikdy\s+som\s+sa\s+(ho|ju|toho|tomu)\s+nedotklo\b',
            r'nikdy som sa \1 neučilo'),
        # bez "som": "som sa toho nedotkla" → "som sa tomu neučila"
        (r'\bnikdy\s+som\s+sa\s+toho\s+nedotkla\b', 'nikdy som sa tomu neučila'),
        (r'\bnikdy\s+som\s+sa\s+toho\s+nedotkol\b', 'nikdy som sa tomu neučil'),
        # Sea of Thieves názvy hier (Rare)
        (r'\bTool[\s-]?[Tt]ale\b',             'Tall Tales'),
        (r'\bTool[\s-]?[Tt]ales\b',            'Tall Tales'),
        # Typos v 26B prekladoch
        (r'\bvoksl\b',                         'voxel'),
        (r'\bvoksly\b',                        'voxely'),
        # "per your request" 26B prekladá ako "kvôli" — správne je "podľa"
        (r'\bkvôli\s+vašej\s+žiadosti\b',      'podľa vašej žiadosti'),
        (r'\bkvôli\s+vašim\s+žiadostiam\b',    'podľa vašich žiadostí'),
        # "blow up your PC" → 26B halucinuje "biť o svoj život"
        (r'\bnemusel\s+biť\s+o\s+svoj\b',      'nepraskol o váš'),
        (r'\bbiť\s+o\s+svoj\s+život\b',        'praskol'),
        # "upome TRUE" — 26B halucinácia pri zdrojovom "Wait, hold up, TRUE"
        (r'\btrochu\s+sa\s+upome\s+TRUE\b',    'počkaj, spomaľme'),
        (r'\bupome\s+TRUE\b',                  'moment'),
        # "drums please / drum roll please" — má byť "bubny prosím"
        (r'\bRolníčky\s+prosím\b',             'Bubny prosím'),
        (r'\brolníčky,\s+prosím\b',            'bubny, prosím'),
        (r'\s{2,}',                           ' '),  # whitespace cleanup po odstránení
    ]
    for pat, repl in _simple:
        rules.append((re.compile(pat, re.IGNORECASE), repl))

    return rules


def get_audit_bad_terms() -> list[tuple]:
    """
    Vráti zoznam (compiled_regex, flag_name) pre audit_sk_tts_json.py.
    Detekuje zlé preklady v SK texte.
    Termíny kde sk_term == en_term (napr. RAM, CPU, hostname) sa NEflagujú —
    sú korektné aj v SK texte.
    """
    result = []
    for en, sk, bad_regex, ci in _GLOSSARY_ENTRIES:
        # Ak je sk_term == en_term, termín je korektný v SK texte — neflaguj
        if en.lower() == sk.lower() and bad_regex is None:
            continue
        if bad_regex is None:
            # Verbatim anglický termín v SK texte = nepreložené
            pattern = r'\b' + re.escape(en) + r'\b'
            flag = f"bad_term:{en.replace(' ', '_').lower()[:30]}"
        else:
            pattern = bad_regex
            flag = f"bad_term:{en.replace(' ', '_').lower()[:30]}"
        flags_val = re.IGNORECASE if ci else 0
        result.append((re.compile(pattern, flags_val), flag))
    return result


# Pre-kompilované výstupy (lazy cache)
_postfix_rules_cache = None
_audit_terms_cache = None


def postfix_rules() -> list[tuple]:
    global _postfix_rules_cache
    if _postfix_rules_cache is None:
        _postfix_rules_cache = get_postfix_rules()
    return _postfix_rules_cache


def audit_bad_terms() -> list[tuple]:
    global _audit_terms_cache
    if _audit_terms_cache is None:
        _audit_terms_cache = get_audit_bad_terms()
    return _audit_terms_cache


def apply_postfix(text: str) -> tuple[str, list[str]]:
    """Aplikuje post-fix pravidlá. Vracia (opravený text, zoznam aplikovaných pravidiel)."""
    applied = []
    for compiled_re, repl in postfix_rules():
        new = compiled_re.sub(repl, text)
        if new != text:
            applied.append(repl)
            text = new
    return text, applied


if __name__ == "__main__":
    print("=== LLM PROMPT BLOCK ===")
    print(get_llm_prompt_block())
    print("\n=== POST-FIX RULES ===")
    for r, repl in get_postfix_rules():
        print(f"  {r.pattern!r} → {repl!r}")
    print("\n=== AUDIT BAD TERMS ===")
    for r, flag in get_audit_bad_terms():
        print(f"  {r.pattern!r} → {flag}")
