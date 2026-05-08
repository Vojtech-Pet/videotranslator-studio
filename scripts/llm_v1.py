"""LLM v1 wrappery — Gemma 3 12B IT translator + length adjuster cez ollama.

Použitie z pipeline:
    from llm_v1 import translate_g3_v1, adjust_length_g3_v1

    sk = translate_g3_v1("Tom went to the cinema.")
    sk_short = adjust_length_g3_v1(sk, target_chars=30)

Modely:
    g3-12b-translator-v1       — EN→SK preklad
    g3-12b-length-adjuster-v1  — SK length adjustment (compress + expand)

Pri každom volaní:
    1. Strict prompt s explicit zákazom alternatív
    2. Output stripping (Alebo:, **, Vysvetlivky, atď.)
    3. Leak detection (asistent waiting prompt) → fallback na originál
"""
from __future__ import annotations
import json
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
TRANSLATOR_MODEL = "g3-12b-translator-v1"
ADJUSTER_MODEL = "g3-12b-length-adjuster-v1"

# Stop tokens / leak markery (signál že model spadol do asistent módu)
ASSISTANT_LEAK_PATTERNS = [
    "prosím poskytnite",
    "pošlite mi",
    "poskytnite mi",
    "som pripravený",
    "rád pomôžem",
    "rád ti pomôžem",
    "je tu nejaký",
    "máte nejaký konkrétny",
    # Prompt-leak: model namiesto prekladu reprodukuje vlastný prompt template
    "pošlem ti anglický text",
    "pošlem vám anglický text",
    "ja ho preložím",
    "preložím ho do slovenčiny",
    "tu je anglický text",
    "tu máš anglický text",
    "akonáhle mi ho pošleš",
    "as soon as you send",
    "send me the english",
]

# Format markery ktoré treba odstrániť z čistého výstupu
FORMAT_STRIP_MARKERS = [
    "\n\n",
    "**Vysvetlivky",
    "**Vysvetlenie",
    "Alebo:",
    "Alebo,",
    "Alebo môžete",
    "alebo:",
    "Vysvetlenie:",
    "Poznámka:",
    "Tu je preklad",
    "Tu je verzia",
    "Tu je niekoľko",
    "preformulujme si",
    "v závislosti od nuansy",
    "v závislosti od kontextu",
]

# Prefix patterns ktoré treba odstrániť ak začínajú výstup
FORMAT_PREFIX_PATTERNS = [
    r"^Okej[,.]?\s+",
    r"^Dobre[,.]?\s+",
    r"^OK[,.]?\s+",
    r"^Tu je preklad[^:]*:\s*",
    r"^Toto je preklad[^:]*:\s*",
    r"^Preklad[^:]*:\s*",
]

# Inštrukcie MUSIA presne zodpovedať tréningovým templatom (v translator + length_adjuster
# tréningu) — inak model je confusing a generuje nepredvídateľne.

TRANSLATE_INSTRUCTION = "Prelož anglický text do prirodzenej slovenčiny."

# Pri technickom obsahu — explicitné zachovanie EN technických termov
# (inak by "engine" → "motor" čo je vhodné pre auto-videá ale NIE pre tech)
TRANSLATE_INSTRUCTION_TECHNICAL = (
    "Prelož anglický text do prirodzenej slovenčiny. "
    "DÔLEŽITÉ: technické anglické termíny ako engine, engines, shader, shaders, "
    "framework, frameworks, rendering, pipeline, GPU, CPU, API, SDK, ECS "
    "PONECHAJ v angličtine — neprekladaj ich."
)

# Content typy ktoré tlačia model preserved EN tech terms
TECHNICAL_CONTENT_TYPES = {"technical", "programming", "sql"}

ADJUST_INSTRUCTION_FMT = (
    "Uprav slovenský text na cieľovú dĺžku približne {n} znakov. "
    "Ak je text dlhší — skráť (vyhoď výplňové slová ako však, tak, akože, vlastne). "
    "Ak je kratší — rozšír (pridaj prirodzené výplňové slová alebo kontext). "
    "Zachovaj význam, mená a čísla."
)


def _ollama_generate(model: str, prompt: str, num_predict: int = 512,
                      timeout: int = 90, temperature: float = 0.0) -> str:
    """Volanie ollama /api/generate s deterministickými parametrami.

    num_predict default 512 (zvýšené z 200 — predtým sa odsekávali dlhé vety).
    timeout 90s aj pre dlhšie segmenty na cold-loaded modeli.
    temperature default 0.0 (deterministic). Vyššia hodnota (0.5-0.8) sa použije
    pri retry pre EN-leak — preruší determined "stuck on EN" pattern.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": 1.05,
            "num_ctx": 4096,
            # Stop tokens — odstránené "EN:" a "\nSK:" (boli dôvodom predčasného cutoff
            # ak model náhodou vyslovil "SK:" v rámci výstupu)
            "stop": ["<end_of_turn>", "<start_of_turn>", "</end_of_turn>"],
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return f"[ERROR: {e}]"


def _has_assistant_leak(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in ASSISTANT_LEAK_PATTERNS)


def _strip_format_artifacts(text: str) -> str:
    """Odstráni asistent-style formátovanie a alternatívy."""
    out = text.strip()
    # End-of-turn / start-of-turn leaks (variant so / aj bez)
    for tok in ("</end_of_turn>", "<end_of_turn>", "</start_of_turn>", "<start_of_turn>",
                "<eos>", "</eos>"):
        out = out.replace(tok, "").strip()
    # Strip prefix patterns ("Okej, tu je preklad...", "Tu je niekoľko možností...")
    for pat in FORMAT_PREFIX_PATTERNS:
        out = re.sub(pat, "", out, count=1, flags=re.I).strip()
    # Cut at first known marker
    for marker in FORMAT_STRIP_MARKERS:
        if marker in out:
            out = out.split(marker, 1)[0].strip()
    # Markdown bold/italic
    out = re.sub(r"\*+", "", out)
    # "verzia / verzia2 / verzia3" → vezmi len prvú
    if " / " in out and len(out.split(" / ")[0]) >= 5:
        out = out.split(" / ")[0].strip()
    # Strip prefix "SK:" alebo "Preklad:" ak model echuje
    out = re.sub(r"^(SK|EN|Preklad|Translation):\s*", "", out, flags=re.I)
    # Trailing punctuation cleanup
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _output_suspicious(en: str, sk: str) -> bool:
    """Heuristika: výstup je príliš krátky alebo nedopovedaný."""
    if not sk: return True
    en_len, sk_len = len(en), len(sk)
    if en_len > 40 and sk_len < en_len * 0.25:
        return True
    if en_len > 60 and sk.rstrip()[-1:] not in ".!?…\"'»“”’":
        return True
    return False


def _output_hallucinated(en: str, sk: str) -> bool:
    """Heuristika: model halucinuje (SK output je výrazne dlhší než EN input)."""
    if not sk or not en:
        return False
    en_len = len(en.strip())
    sk_len = len(sk.strip())
    if en_len < 15:
        return False  # veľmi krátke vety môžu byť legitímne expanded
    # SK preklad EN má typicky 0.85-1.20× dĺžku.
    # Nad 2.0× pri >=30 chars EN, alebo nad 3.0× pri 15-29 chars EN = hallucination.
    if en_len >= 30 and sk_len > en_len * 2.0:
        return True
    if en_len >= 15 and sk_len > en_len * 3.0:
        return True
    return False


def _strip_bilingual_alternatives(sk: str) -> str:
    """Odstráň 'X/Y' bilingválne alternatívy → ponechaj prvý variant.

    Príklady:
      'zaobstaral/získal' → 'zaobstaral'
      'mohol/mohla' → 'mohol'
      'podpísal/a' → 'podpísal'
    """
    if not sk:
        return sk
    # word/word patterns (alphanumerics with diacritics, len 2-15 each side)
    sk = re.sub(
        r"\b(\w{2,15})/(\w{1,15})\b",
        lambda m: m.group(1),
        sk,
        flags=re.UNICODE,
    )
    return sk


# Detekcia EN výstupu (model zlyhal a dal späť angličtinu)
_EN_FUNCTION_WORDS = re.compile(
    r"\b(I|the|and|that|with|from|this|have|will|been|were|like|going|just|to|of|"
    r"for|in|on|is|was|are|am|my|me|you|he|she|it|we|they|them|us|our|its|but|so|"
    r"because|when|where|what|how|who|why|then|than|now|here|there|over|about|"
    r"into|out|down|up|after|before)\b", re.I)
_SK_FUNCTION_WORDS = re.compile(
    r"\b(je|som|si|sme|sú|na|do|od|po|cez|pre|ako|lebo|preto|tiež|aj|tu|tam|teraz|"
    r"včera|dnes|čo|kto|kde|kedy|prečo|môj|mňa|tvoj|náš|váš|že|aby|keď|kým|síce|"
    r"sa|so|v|vo|s|by|veľmi|tak|už|ešte|alebo|ale|však|treba|musí|môžem|mali|"
    r"môže|môžeš|robiť|byť|mať|nemám)\b", re.I)
_SK_DIACRITICS = re.compile(r"[áäčďéíĺľňóôŕšťúýž]", re.I)


def _is_english_output(sk: str) -> bool:
    """True ak SK output vyzerá ako EN (preklad zlyhal)."""
    if not sk or len(sk) < 15:
        return False
    # Strip EN apostrof patterns (it's, I'm, don't, ...) — inak short SK regex
    # match-uje "s" z "it's" ako SK slovo a pokazí detekciu
    sk_clean = re.sub(r"\b\w+'\w+\b", "", sk)
    en_count = len(_EN_FUNCTION_WORDS.findall(sk_clean))
    sk_count = len(_SK_FUNCTION_WORDS.findall(sk_clean))
    diac_count = len(_SK_DIACRITICS.findall(sk_clean))
    if en_count >= 3 and sk_count == 0 and diac_count <= 2:
        return True
    if en_count >= 5 and sk_count <= 1:
        return True
    # Ratio rule: vysoký EN count + EN dominuje SK aspoň 4× a žiadne diakritika
    if en_count >= 8 and en_count > sk_count * 4 and diac_count <= 2:
        return True
    return False


def translate_g3_v1(en_text: str, *, fallback: str | None = None,
                     content_type: str = "general") -> str:
    """EN→SK preklad cez Gemma 3 12B v1.

    Args:
        en_text: anglický text (1 veta odporúčané)
        fallback: čo vrátiť ak model padne; default = en_text
        content_type: "general" (default), "technical"/"programming"/"sql" → tlačí
                      model zachovať EN tech termy (engine, shader, framework, ...)

    Returns:
        slovenský preklad alebo fallback
    """
    if not en_text or not en_text.strip():
        return fallback if fallback is not None else en_text
    # Inštrukcia podľa content_type
    if content_type in TECHNICAL_CONTENT_TYPES:
        instr = TRANSLATE_INSTRUCTION_TECHNICAL
    else:
        instr = TRANSLATE_INSTRUCTION
    # Tréningový formát: instruction\n\n{en_text} (žiadne EN:/SK: prefixy)
    prompt = f"{instr}\n\n{en_text.strip()}"
    # Adaptívne num_predict — krátke vety nepotrebujú 512, dlhé treba viac
    n_words = len(en_text.split())
    n_predict = max(200, min(800, n_words * 4 + 100))
    raw = _ollama_generate(TRANSLATOR_MODEL, prompt, num_predict=n_predict)
    if raw.startswith("[ERROR"):
        return fallback if fallback is not None else en_text
    cleaned = _strip_format_artifacts(raw)
    if _has_assistant_leak(cleaned) or not cleaned:
        return fallback if fallback is not None else en_text
    # Retry s vyšším num_predict ak výstup vyzerá truncated
    if _output_suspicious(en_text, cleaned):
        retry_predict = min(1200, n_predict * 2)
        raw2 = _ollama_generate(TRANSLATOR_MODEL, prompt, num_predict=retry_predict)
        if not raw2.startswith("[ERROR"):
            cleaned2 = _strip_format_artifacts(raw2)
            if cleaned2 and not _has_assistant_leak(cleaned2):
                if len(cleaned2) > len(cleaned):
                    cleaned = cleaned2
    # Reject hallucination — model expandol >2× (typicky generic EN content)
    if _output_hallucinated(en_text, cleaned):
        print(f"[G3-V1] Hallucination detected ({len(en_text)}→{len(cleaned)} chars, ratio "
              f"{len(cleaned)/max(1,len(en_text)):.1f}×) — retry", flush=True)
        cleaned = ""  # vynúti EN-leak retry path nižšie
    # Retry ak výstup je EN alebo hallucinated — model zlyhal
    if not cleaned or _is_english_output(cleaned):
        # Retry 1 — strict prompt s temperature=0
        retry_prompt = (
            f"Si profesionálny prekladateľ. Tento text MUSÍŠ preložiť do slovenčiny. "
            f"Odpoveď MUSÍ byť po slovensky, NIE po anglicky.\n\n"
            f"{en_text.strip()}"
        )
        raw3 = _ollama_generate(TRANSLATOR_MODEL, retry_prompt, num_predict=n_predict)
        cleaned3 = _strip_format_artifacts(raw3) if not raw3.startswith("[ERROR") else ""
        if (cleaned3 and not _has_assistant_leak(cleaned3)
                and not _is_english_output(cleaned3)
                and not _output_hallucinated(en_text, cleaned3)):
            cleaned = cleaned3
        else:
            # Retry 2 — vyššia temperature 0.7 (preruší stuck-on-EN pattern + halucinácie)
            raw4 = _ollama_generate(TRANSLATOR_MODEL, retry_prompt,
                                     num_predict=n_predict, temperature=0.7)
            cleaned4 = _strip_format_artifacts(raw4) if not raw4.startswith("[ERROR") else ""
            if (cleaned4 and not _has_assistant_leak(cleaned4)
                    and not _is_english_output(cleaned4)
                    and not _output_hallucinated(en_text, cleaned4)):
                cleaned = cleaned4
                print(f"[G3-V1] Recovered cez temp=0.7 retry ({len(cleaned4)} chars)", flush=True)
            else:
                # G3 v1 zlyhal aj po 3 retry — vráť empty (pipeline urobí ticho)
                print(f"[G3-V1] EN-leak/hallucination po 3 retry — empty pre "
                      f"{len(en_text)}-char input", flush=True)
                return ""
    # Strip bilingválne alternatívy "X/Y" → "X"
    cleaned = _strip_bilingual_alternatives(cleaned)
    return cleaned


def adjust_length_g3_v1(sk_text: str, target_chars: int, *,
                         max_runaway_factor: float = 1.6) -> str:
    """SK length adjustment (compress alebo expand) cez Gemma 3 12B length adjuster v1.

    Args:
        sk_text: slovenský text na úpravu
        target_chars: cieľová dĺžka (podľa audio dur × CPS)
        max_runaway_factor: ak output > target × factor → fallback (model "rozprával sa")

    Returns:
        upravený text alebo originál (fallback)
    """
    if not sk_text or not sk_text.strip() or target_chars <= 0:
        return sk_text
    instr = ADJUST_INSTRUCTION_FMT.format(n=target_chars)
    prompt = f"{instr}\n\n{sk_text.strip()}"
    raw = _ollama_generate(ADJUSTER_MODEL, prompt,
                            num_predict=int(target_chars * 2.5))
    if raw.startswith("[ERROR"):
        return sk_text
    cleaned = _strip_format_artifacts(raw)

    # Detection: leak (asistent waiting for input)
    if _has_assistant_leak(cleaned):
        return sk_text  # fallback
    # Detection: empty
    if not cleaned:
        return sk_text
    # Detection: runaway length (model rozprávkuje)
    if len(cleaned) > target_chars * max_runaway_factor:
        return sk_text
    # Detection: zostal originál (no change)
    if cleaned == sk_text.strip():
        return sk_text

    return cleaned


def adjust_length_for_duration(sk_text: str, audio_duration_sec: float,
                                target_cps: float = 13.0) -> str:
    """Convenience: výpočet target_chars z audio dĺžky + adjust.

    Args:
        sk_text: slovenský text
        audio_duration_sec: dĺžka cieľového audio v sekundách
        target_cps: znaky/sec pri TTS rýchlosti (default 13.0 pre S2 Pro SK)

    Returns:
        text upravený na ~ duration × cps znakov
    """
    target_chars = max(1, int(audio_duration_sec * target_cps))
    return adjust_length_g3_v1(sk_text, target_chars)
