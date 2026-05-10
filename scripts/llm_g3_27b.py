"""LLM wrapper pre Gemma 3 27B multitask v1 — single-step EN→SK + length-aware.

Použitie z pipeline:
    from llm_g3_27b import translate_g3_27b

    # Unconstrained (čistý preklad)
    sk = translate_g3_27b("Hello world.")

    # Constrained (target length pre dabing)
    sk = translate_g3_27b("Hello world.", target_seconds=2.5, target_cps=15.0)

Model: g3-27b-multitask-v2 (Ollama, GGUF Q4_K_M ~17 GB)

Tréning:
    /mnt/tts_data/knihy/train_gemma3_27b_multitask_v2.py
    Train data: 42 880 vzoriek (v1 + 250 negation augmentation)
    LoRA r=32, α=64, 2 epochs, FA2, bf16, train_loss 0.1268
    Negation transfer: "I knew zero X" → "Nevedel som o X vôbec nič" (FIX vs v1 bug)

Pri tomto modeli netreba separátny length adjuster — single-step robí oba úlohy.
"""
from __future__ import annotations
import json
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "g3-27b-multitask-v2"

# Stop tokens / leak markery (rovnaké ako llm_v1)
ASSISTANT_LEAK_PATTERNS = [
    "prosím poskytnite", "pošlite mi", "poskytnite mi", "som pripravený",
    "rád pomôžem", "rád ti pomôžem", "je tu nejaký", "máte nejaký konkrétny",
    "pošlem ti anglický text", "pošlem vám anglický text", "ja ho preložím",
    "preložím ho do slovenčiny", "tu je anglický text", "tu máš anglický text",
    "akonáhle mi ho pošleš", "as soon as you send", "send me the english",
]

FORMAT_STRIP_MARKERS = [
    "\n\n", "**Vysvetlivky", "**Vysvetlenie",
    "Alebo:", "Alebo,", "alebo:", "Vysvetlenie:", "Poznámka:",
    "Tu je preklad", "Tu je verzia", "Tu je niekoľko",
    "preformulujme si", "v závislosti od nuansy", "v závislosti od kontextu",
]

FORMAT_PREFIX_PATTERNS = [
    r"^Okej[,.]?\s+", r"^Dobre[,.]?\s+", r"^OK[,.]?\s+",
    r"^Tu je preklad[^:]*:\s*", r"^Toto je preklad[^:]*:\s*",
    r"^Preklad[^:]*:\s*",
]

# EN-leak detection — rovnaké ako llm_v1
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


def _ollama_generate(prompt: str, num_predict: int = 512, timeout: int = 120,
                      temperature: float = 0.0) -> str:
    body = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": 1.05,
            "num_ctx": 2048,
            "stop": ["<end_of_turn>", "<start_of_turn>"],
        },
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                  headers={"Content-Type": "application/json"})
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
    out = text.strip()
    for marker in FORMAT_STRIP_MARKERS:
        idx = out.find(marker)
        if idx > 5:
            out = out[:idx].strip()
    for pat in FORMAT_PREFIX_PATTERNS:
        out = re.sub(pat, "", out, flags=re.I)
    if " / " in out and len(out.split(" / ")[0]) >= 5:
        out = out.split(" / ")[0].strip()
    out = re.sub(r"^(SK|EN|Preklad|Translation):\s*", "", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _is_english_output(sk: str) -> bool:
    if not sk or len(sk) < 15:
        return False
    sk_clean = re.sub(r"\b\w+'\w+\b", "", sk)
    en = len(_EN_FUNCTION_WORDS.findall(sk_clean))
    skk = len(_SK_FUNCTION_WORDS.findall(sk_clean))
    diac = len(_SK_DIACRITICS.findall(sk_clean))
    if en >= 3 and skk == 0 and diac <= 2:
        return True
    if en >= 5 and skk <= 1:
        return True
    if en >= 8 and en > skk * 4 and diac <= 2:
        return True
    return False


def _output_hallucinated(en: str, sk: str) -> bool:
    if not sk or not en:
        return False
    en_len = len(en.strip())
    sk_len = len(sk.strip())
    if en_len < 15:
        return False
    if en_len >= 30 and sk_len > en_len * 2.0:
        return True
    if en_len >= 15 and sk_len > en_len * 3.0:
        return True
    return False


def _strip_bilingual_alternatives(sk: str) -> str:
    if not sk:
        return sk
    sk = re.sub(r"\b(\w{2,15})/(\w{1,15})\b",
                lambda m: m.group(1), sk, flags=re.UNICODE)
    return sk


def translate_g3_27b(en_text: str, *,
                      target_seconds: float | None = None,
                      target_cps: float | None = None,
                      fallback: str | None = None) -> str:
    """EN→SK preklad cez Gemma 3 27B multitask v1.

    Args:
        en_text: anglický text
        target_seconds: ak nie None, model dostane constraint pre dĺžku audio
        target_cps: chars per second (typicky 13-17 pre slovenčinu)
        fallback: čo vrátiť pri zlyhaní (default = en_text)

    Returns:
        slovenský preklad (constrained alebo natural podľa parametrov)
    """
    if not en_text or not en_text.strip():
        return fallback if fallback is not None else en_text

    # Constrained vs unconstrained prompt (ZHODNÝ s tréningovým formátom)
    if target_seconds is not None and target_cps is not None:
        user_msg = (
            f"Translate to Slovak. "
            f"Duration: {target_seconds:.1f}s, target CPS: {target_cps:.1f}.\n"
            f"{en_text.strip()}"
        )
    else:
        user_msg = f"Translate to Slovak.\n{en_text.strip()}"

    n_words = len(en_text.split())
    n_predict = max(200, min(800, n_words * 4 + 100))

    raw = _ollama_generate(user_msg, num_predict=n_predict)
    if raw.startswith("[ERROR"):
        return fallback if fallback is not None else en_text

    cleaned = _strip_format_artifacts(raw)
    if _has_assistant_leak(cleaned) or not cleaned:
        return fallback if fallback is not None else en_text

    # Hallucination guard
    if _output_hallucinated(en_text, cleaned):
        print(f"[G3-27B] Hallucination detected ({len(en_text)}→{len(cleaned)}, "
              f"ratio {len(cleaned)/max(1,len(en_text)):.1f}×) — retry temp=0.5", flush=True)
        raw2 = _ollama_generate(user_msg, num_predict=n_predict, temperature=0.5)
        cleaned2 = _strip_format_artifacts(raw2) if not raw2.startswith("[ERROR") else ""
        if (cleaned2 and not _has_assistant_leak(cleaned2)
                and not _is_english_output(cleaned2)
                and not _output_hallucinated(en_text, cleaned2)):
            cleaned = cleaned2
        else:
            return ""

    # EN-leak retry (rovnaký pattern ako llm_v1)
    if _is_english_output(cleaned):
        retry_prompt = (
            f"Si profesionálny prekladateľ. Tento text MUSÍŠ preložiť do slovenčiny. "
            f"Odpoveď MUSÍ byť po slovensky, NIE po anglicky.\n\n"
            f"{en_text.strip()}"
        )
        raw3 = _ollama_generate(retry_prompt, num_predict=n_predict)
        cleaned3 = _strip_format_artifacts(raw3) if not raw3.startswith("[ERROR") else ""
        if (cleaned3 and not _has_assistant_leak(cleaned3)
                and not _is_english_output(cleaned3)
                and not _output_hallucinated(en_text, cleaned3)):
            cleaned = cleaned3
        else:
            raw4 = _ollama_generate(retry_prompt, num_predict=n_predict, temperature=0.7)
            cleaned4 = _strip_format_artifacts(raw4) if not raw4.startswith("[ERROR") else ""
            if (cleaned4 and not _has_assistant_leak(cleaned4)
                    and not _is_english_output(cleaned4)
                    and not _output_hallucinated(en_text, cleaned4)):
                cleaned = cleaned4
                print(f"[G3-27B] Recovered cez temp=0.7 retry ({len(cleaned4)} chars)", flush=True)
            else:
                print(f"[G3-27B] EN-leak/hallucination po 3 retry — empty pre "
                      f"{len(en_text)}-char input", flush=True)
                return ""

    cleaned = _strip_bilingual_alternatives(cleaned)
    return cleaned


def is_g3_27b_available() -> bool:
    """Skontroluj či ollama beží a g3-27b-multitask-v2 je nahraný."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
        names = [m.get("name", "") for m in data.get("models", [])]
        return any(MODEL_NAME in n for n in names)
    except Exception:
        return False
