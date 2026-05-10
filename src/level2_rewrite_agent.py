from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils_sql import contains_sql_terms, fix_sql_terms, protected_terms_in
from src.utils_text import normalize_text

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = _ROOT / "prompts" / "rewrite_system.txt"
_META_STARTERS = (
    "Rozumiem",
    "Dobre,",
    "Dobre.",
    "Tu je",
    "Prepis:",
    "Vysvetlenie:",
    "Odpoved:",
    "Skratena verzia",
)


def read_prompt_text(path: str | Path = DEFAULT_SYSTEM_PROMPT) -> str:
    try:
        return Path(path).expanduser().resolve().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def load_llm(
    model_path: str,
    *,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    verbose: bool = False,
):
    if not (model_path or "").strip():
        return None
    from llama_cpp import Llama

    return Llama(
        model_path=str(Path(model_path).expanduser().resolve()),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )


def build_user_prompt(text: str, source_text: str = "") -> str:
    clean = normalize_text(text)
    source = normalize_text(source_text)
    if source:
        return f"Zdrojovy anglicky text:\n\"{source}\"\n\nText:\n\"{clean}\""
    return f"Text:\n\"{clean}\""


def call_llm(
    llm: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.08,
) -> str | None:
    if llm is None:
        return None
    if callable(llm) and not hasattr(llm, "create_chat_completion"):
        result = llm(system_prompt, user_prompt)
        return normalize_text(result)
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return normalize_text(str(content).strip('"\' '))


def is_valid_rewrite(original: str, rewritten: str, *, preserve_sql_terms: bool = False) -> bool:
    if not rewritten:
        return False
    if rewritten.startswith(_META_STARTERS):
        return False
    if len(rewritten) < 4:
        return False
    if len(rewritten) > max(int(len(original) * 1.10), len(original) + 20):
        return False
    if preserve_sql_terms:
        original_terms = protected_terms_in(original)
        if not original_terms.issubset(protected_terms_in(rewritten)):
            return False
    return True


def rewrite_text(
    text: str,
    *,
    source_text: str = "",
    llm: Any = None,
    system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 256,
    temperature: float = 0.08,
) -> str:
    original = normalize_text(text)
    preserve_sql_terms = contains_sql_terms(original) or contains_sql_terms(source_text)
    if preserve_sql_terms:
        original = fix_sql_terms(original)
    if llm is None or not original:
        return original

    system_prompt = read_prompt_text(system_prompt_path)
    user_prompt = build_user_prompt(original, source_text)
    rewritten = call_llm(
        llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    rewritten = normalize_text(rewritten)
    if preserve_sql_terms:
        rewritten = fix_sql_terms(rewritten)
    if not is_valid_rewrite(original, rewritten, preserve_sql_terms=preserve_sql_terms):
        return original
    return rewritten

