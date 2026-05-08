"""
TranslateGemma drop-in wrapper pre llama-cpp-python pipeline.

Workaround pre chýbajúcu podporu chat_template_kwargs v llama-cpp-python:
- Načíta Jinja2 template priamo z GGUF metadát
- Manuálne renderuje prompt so source_lang_code / target_lang_code
- Posiela hotový prompt cez nízkoúrovňové create_completion() API
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jinja2 import Environment, StrictUndefined
from llama_cpp import Llama


@dataclass
class TranslateGemmaPipelineConfig:
    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    n_threads: Optional[int] = None
    verbose: bool = False
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    seed: int = 0


SUPPORTED_LANGS = {
    "af", "sq", "am", "ar", "hy", "az", "eu", "be", "bn", "bs",
    "bg", "ca", "zh", "hr", "cs", "da", "nl", "en", "et", "fi",
    "fr", "gl", "ka", "de", "el", "gu", "ht", "he", "hi", "hu",
    "is", "id", "ga", "it", "ja", "kn", "kk", "km", "ko", "ku",
    "ky", "lo", "lv", "lt", "mk", "ms", "ml", "mt", "mr", "mn",
    "my", "ne", "nb", "fa", "pl", "pt", "ro", "ru", "sr", "si",
    "sk", "sl", "so", "es", "sw", "sv", "tl", "tg", "ta", "te",
    "th", "tr", "tk", "uk", "ur", "uz", "vi", "cy", "xh", "yo", "zu",
}

# Mapovanie dlhých kódov na ISO 639-1
_LANG_MAP = {
    "slk": "sk", "ces": "cs", "eng": "en", "deu": "de",
    "fra": "fr", "pol": "pl", "hun": "hu", "ukr": "uk",
    "rus": "ru", "spa": "es", "ita": "it",
}


class TranslateGemmaPipeline:
    """
    Drop-in TranslateGemma wrapper pre llama-cpp-python.

    Použitie:
        translator = TranslateGemmaPipeline(
            TranslateGemmaPipelineConfig(
                model_path="/models/translategemma-12b-it-Q8_0.gguf",
                n_gpu_layers=-1,
            )
        )
        result = translator.translate("Hello world.", src="en", tgt="sk")
    """

    def __init__(self, config: TranslateGemmaPipelineConfig):
        self.config = config
        self.llm = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.n_threads,
            verbose=config.verbose,
            chat_format=None,
        )
        self.chat_template = self._load_chat_template()
        self.jinja_env = Environment(
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.jinja_env.from_string(self.chat_template)

    def _load_chat_template(self) -> str:
        metadata = getattr(self.llm, "metadata", None)
        if not metadata:
            raise RuntimeError("GGUF metadata nie sú dostupné.")
        template = metadata.get("tokenizer.chat_template")
        if template:
            return template
        candidates = {
            k: v for k, v in metadata.items()
            if "chat_template" in k and isinstance(v, str)
        }
        if "tokenizer.chat_template.default" in candidates:
            return candidates["tokenizer.chat_template.default"]
        if candidates:
            return next(iter(candidates.values()))
        raise RuntimeError("V GGUF metadata sa nenašiel chat template.")

    _SK_TRANSLATE_SYSTEM = (
        "Si profesionálny prekladateľ. Prekladáš text do slovenčiny. "
        "Dodržuj správne slovenské skloňovanie, slovosled a jazykové konvencie. "
        "Prekladaj prirodzene, nie doslovne. "
        "Technické výrazy 'engine', 'engines', 'shader', 'shaders', 'rendering', 'voxel' "
        "NEPRELEKÁŠ — zachovaj ich v angličtine (napr. 'engine' NIE 'motor'). "
        "Slovo 'era'/'eras' prekladaj ako 'éra'/'éry', nie 'chyba' ani 'error'. "
        "Slovo 'went from' prekladaj ako 'prešiel od', nie 'prestal'."
    )

    def _is_sk_translate_model(self) -> bool:
        return "translate-e4b" in self.config.model_path or "translate_e4b" in self.config.model_path

    def _build_messages(self, text: str, src: str, tgt: str) -> List[Dict[str, Any]]:
        if self._is_sk_translate_model():
            return [
                {
                    "role": "user",
                    "content": f"{self._SK_TRANSLATE_SYSTEM}\n\nPrelož do slovenčiny:\n{text}",
                    "tool_calls": None,
                }
            ]
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src,
                        "target_lang_code": tgt,
                        "text": text,
                    }
                ],
                "tool_calls": None,
            }
        ]

    def _render_prompt(
        self,
        text: str,
        src: str,
        tgt: str,
        *,
        add_generation_prompt: bool = True,
        bos_token: str = "",
        eos_token: str = "",
    ) -> str:
        messages = self._build_messages(text=text, src=src, tgt=tgt)
        prompt = self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=bos_token,
            eos_token=eos_token,
            tools=None,
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("Renderovaný prompt je prázdny.")
        return prompt

    def _cleanup_output(self, text: str) -> str:
        text = text.strip()
        for marker in ("<end_of_turn>", "<start_of_turn>", "<eos>"):
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        return text

    def _validate_lang(self, code: str) -> None:
        if code not in SUPPORTED_LANGS:
            raise ValueError(f"Nepodporovaný language code: '{code}'. Podporované: {sorted(SUPPORTED_LANGS)}")

    def translate(
        self,
        text: str,
        src: str = "en",
        tgt: str = "sk",
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        src = _LANG_MAP.get(src, src)
        tgt = _LANG_MAP.get(tgt, tgt)
        self._validate_lang(src)
        self._validate_lang(tgt)

        prompt = self._render_prompt(text=text, src=src, tgt=tgt)
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            top_p=top_p if top_p is not None else self.config.top_p,
            top_k=top_k if top_k is not None else self.config.top_k,
            min_p=min_p if min_p is not None else self.config.min_p,
            repeat_penalty=repeat_penalty if repeat_penalty is not None else self.config.repeat_penalty,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=seed if seed is not None else self.config.seed,
            echo=False,
            stream=False,
            stop=stop or [],
        )
        try:
            raw = response["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Neočakávaný response formát: {response}") from exc
        return self._cleanup_output(raw)

    def __call__(self, text: str, src: str = "en", tgt: str = "sk") -> str:
        return self.translate(text=text, src=src, tgt=tgt)

    def free(self):
        del self.llm
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Singleton cache pre pipeline.py
# ---------------------------------------------------------------------------

_tg_instance: Optional[TranslateGemmaPipeline] = None
_tg_model_path: str = ""


def get_translategemma(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096) -> TranslateGemmaPipeline:
    global _tg_instance, _tg_model_path
    if _tg_instance is None or _tg_model_path != model_path:
        if _tg_instance is not None:
            _tg_instance.free()
        _tg_instance = TranslateGemmaPipeline(
            TranslateGemmaPipelineConfig(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )
        )
        _tg_model_path = model_path
    return _tg_instance


def free_translategemma():
    global _tg_instance, _tg_model_path
    if _tg_instance is not None:
        _tg_instance.free()
        _tg_instance = None
        _tg_model_path = ""


def translate_segment_with_translategemma(
    text: str,
    tgt_lang: str,
    model_path: str,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    tg_instance: Optional[TranslateGemmaPipeline] = None,
) -> str:
    """Preloží jeden segment. Kompatibilné s pipeline.py."""
    tg = tg_instance or get_translategemma(model_path, n_gpu_layers, n_ctx)
    try:
        return tg.translate(text, src="en", tgt=tgt_lang)
    except Exception as e:
        print(f"[TRANSLATEGEMMA] Chyba: {e}", flush=True)
        return ""
