"""
video_translate.py
==================
Preloží video do slovenčiny. Automatická detekcia jazyka.
Výstup: video.sk.srt vedľa pôvodného súboru.

Použitie:
    python video_translate.py --input video.mp4
    python video_translate.py --input video.mp4 --device cpu
    python video_translate.py --input video.mp4 --model facebook/nllb-200-3.3B
"""

import argparse
import torch
from pathlib import Path
from langdetect import detect
from transformers import pipeline

LANG_MAP = {
    "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn",
    "cs": "ces_Latn", "pl": "pol_Latn", "hu": "hun_Latn",
    "uk": "ukr_Cyrl", "ru": "rus_Cyrl", "es": "spa_Latn", "it": "ita_Latn",
}
TARGET_LANG = "slk_Latn"


def load_models(device: str = "cuda", model_name: str = "facebook/nllb-200-1.3B"):
    import whisperx
    print("Načítavam WhisperX large-v2...")
    asr = whisperx.load_model("large-v2", device=device,
                               compute_type="float16" if device == "cuda" else "int8")
    print(f"Načítavam NLLB prekladač ({model_name})...")
    translator = pipeline(
        "translation", model=model_name,
        device=0 if device == "cuda" else -1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    return asr, translator


def detect_src_lang(segments: list[dict]) -> str:
    sample_text = " ".join(s["text"] for s in segments[:10])
    try:
        detected  = detect(sample_text)
        nllb_code = LANG_MAP.get(detected)
        if nllb_code:
            print(f"Detekovaný jazyk: {detected} → {nllb_code}")
            return nllb_code
        print(f"Neznámy jazyk '{detected}', fallback → angličtina")
    except Exception:
        print("Detekcia jazyka zlyhala, fallback → angličtina")
    return "eng_Latn"


def seconds_to_srt_time(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = int(s % 60)
    ms  = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def export_srt(segments: list[dict], output_path: str):
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seconds_to_srt_time(seg["start"])
        end   = seconds_to_srt_time(seg["end"])
        text  = seg.get("sk_text", seg["text"])
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"SRT uložený: {output_path}")


def translate_video(
    video_path:  str,
    device:      str = "cuda",
    model_name:  str = "facebook/nllb-200-1.3B",
    protect_terms: bool = True,
    out_srt:     str = None,
) -> str:
    import whisperx
    from scripts.term_protection import TermProtector, translate_with_protection

    video      = Path(video_path)
    output_srt = out_srt or str(video.with_suffix(".sk.srt"))

    asr, translator = load_models(device, model_name)
    tp = TermProtector() if protect_terms else None

    print(f"Transkripcia: {video_path}")
    result   = asr.transcribe(str(video), batch_size=16)
    segments = result["segments"]
    print(f"Segmentov: {len(segments)}")

    src_lang = detect_src_lang(segments)

    if protect_terms:
        segments = translate_with_protection(segments, translator, src_lang, tp)
    else:
        texts      = [s["text"].strip() for s in segments]
        translated = translator(texts, src_lang=src_lang, tgt_lang=TARGET_LANG,
                                max_length=256, batch_size=16)
        for seg, t in zip(segments, translated):
            seg["sk_text"] = t["translation_text"]

    export_srt(segments, output_srt)
    return output_srt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",          required=True, help="Video súbor")
    parser.add_argument("--device",         default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model",          default="facebook/nllb-200-1.3B",
                        help="NLLB model (1.3B / 3.3B)")
    parser.add_argument("--no-protection",  action="store_true",
                        help="Vypni ochranu termínov")
    parser.add_argument("--out-srt",        default=None,
                        help="Výstupný SRT súbor (default: video.sk.srt)")
    args = parser.parse_args()

    out = translate_video(
        video_path    = args.input,
        device        = args.device,
        model_name    = args.model,
        protect_terms = not args.no_protection,
        out_srt       = args.out_srt,
    )
    print(f"\n✓ Hotovo: {out}")


if __name__ == "__main__":
    main()
