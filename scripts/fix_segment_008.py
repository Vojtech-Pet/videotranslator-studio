"""
fix_segment_008.py — Oprava prvého segmentu sql_part_008_sk_tts.mp4

Problém: slot 1.322–31.032s (29.71s), TTS reč skončí skoro → ticho na ~20s
Riešenie: rozbiť text na vety, vygenerovať TTS znova, vložiť do videa

Spustenie:
    conda run -n chatterbox_env python scripts/fix_segment_008.py

Výstup: /mnt/tts_data/SQL/sk/sql_part_008_sk_tts_fixed.mp4
"""
from __future__ import annotations
import subprocess, sys, tempfile, json
from pathlib import Path

# ── Konfigurácia ─────────────────────────────────────────────────────────────
VIDEO_IN   = Path("/mnt/tts_data/SQL/sk/sql_part_008_sk_tts.mp4")
VIDEO_OUT  = Path("/mnt/tts_data/SQL/sk/sql_part_008_sk_tts_fixed.mp4")
TTS_WAV    = Path("/mnt/tts_data/VideoTranslator_studio/temp/sql_part_008/sql_part_008_sk_tts.wav")
VOICE_WAV  = Path("/mnt/tts_data/VideoTranslator_studio/voices/tomas_sk.wav")
CB_MODEL   = Path("/mnt/tts_data/VideoTranslator_studio/models/chatterbox/t3_sk_v2.2.safetensors")
SR         = 24000

# Slot prvého segmentu
SEG_START  = 1.322
SEG_END    = 31.032
SEG_DUR    = SEG_END - SEG_START  # 29.71s

# Text rozbitý na vety (prirodzené pauzy)
SENTENCES = [
    "Takto to urobíme a to vykonáme.",
    "Ak sa pozriete na výsledku, vidíte, že tu máme znova dĺžku dva, pretože máme dve medzery.",
    "Ale pri politike jeden máme nulu.",
    "Po aplikácii funkcie trim majú tieto dve hodnoty dĺžku nula.",
    "Týmto nemáme žiadne medzery.",
    "To znamená, že sme si istí, že po aplikácii trim máme buď NULL, alebo prázdny riadok.",
    "si len odstránim tieto informácie. som si istý, že obe sú.",
]

# ── Pomocné funkcie ───────────────────────────────────────────────────────────
def run(cmd: list, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"Príkaz zlyhal: {cmd[0]}")
    return r


_tts_model = None
_tts_device = None

def _load_model(device: str):
    global _tts_model, _tts_device
    if _tts_model is not None and _tts_device == device:
        return _tts_model

    import sys
    chatterbox_path = Path("/home/vojtech/Ai/chatterbox_git/src")
    if not chatterbox_path.exists():
        chatterbox_path = Path("/home/vojtech/chatterbox_git/src")
    if str(chatterbox_path) not in sys.path:
        sys.path.insert(0, str(chatterbox_path))

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from safetensors.torch import load_file as load_safetensors
    import torch

    print(f"  [TTS] Načítavam model na {device}...", flush=True)
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if CB_MODEL.exists():
        print(f"  [TTS] Načítavam SK váhy: {CB_MODEL.name}", flush=True)
        t3_state = load_safetensors(str(CB_MODEL), device="cpu")
        try:
            text_emb = t3_state.get("text_emb.weight")
            text_head = t3_state.get("text_head.weight")
            if text_emb is not None and text_head is not None:
                target_vocab = model.t3.text_emb.weight.shape[0]
                src_vocab = text_emb.shape[0]
                if src_vocab != target_vocab:
                    if src_vocab > target_vocab:
                        t3_state["text_emb.weight"] = text_emb[:target_vocab, :]
                        t3_state["text_head.weight"] = text_head[:target_vocab, :]
                    else:
                        pad_rows = target_vocab - src_vocab
                        import torch as _t
                        emb_pad = text_emb.mean(dim=0, keepdim=True).repeat(pad_rows, 1)
                        head_pad = text_head.mean(dim=0, keepdim=True).repeat(pad_rows, 1)
                        t3_state["text_emb.weight"] = _t.cat([text_emb, emb_pad], dim=0)
                        t3_state["text_head.weight"] = _t.cat([text_head, head_pad], dim=0)
        except Exception as e:
            print(f"  [TTS] WARN vocab resize: {e}", flush=True)
        model.t3.load_state_dict(t3_state, strict=True)
        if device != "cpu":
            model.t3.to(device)
        model.t3.eval()

    _tts_model = model
    _tts_device = device
    return model


def generate_tts(text: str, out_wav: Path) -> None:
    """Vygeneruje TTS pre jeden text pomocou Chatterbox."""
    import torch
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [TTS] '{text[:70]}' → {out_wav.name}", flush=True)

    model = _load_model(device)
    with torch.inference_mode():
        wav = model.generate(
            text,
            language_id="sk",
            audio_prompt_path=str(VOICE_WAV),
            exaggeration=0.50,
            cfg_weight=0.60,
            temperature=0.78,
        )
    torchaudio.save(str(out_wav), wav.cpu(), SR)


def get_wav_duration(wav: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav)],
        capture_output=True, text=True
    )
    return float(r.stdout.strip())


# ── Hlavný skript ─────────────────────────────────────────────────────────────
def main():
    print("=== fix_segment_008.py ===", flush=True)
    print(f"Slot: {SEG_START:.3f}s – {SEG_END:.3f}s ({SEG_DUR:.3f}s)", flush=True)

    with tempfile.TemporaryDirectory(prefix="fix_seg008_") as tmp:
        tmp = Path(tmp)

        # 1. Vygeneruj TTS pre každú vetu
        print("\n[1] Generovanie TTS pre každú vetu...", flush=True)
        chunk_wavs = []
        for i, sentence in enumerate(SENTENCES):
            out = tmp / f"chunk_{i:02d}.wav"
            generate_tts(sentence, out)
            dur = get_wav_duration(out)
            print(f"     chunk_{i:02d}: {dur:.2f}s", flush=True)
            chunk_wavs.append(out)

        # 2. Concatenuj všetky chunky
        print("\n[2] Concatenovanie chunkov...", flush=True)
        concat_list = tmp / "concat.txt"
        concat_list.write_text("\n".join(f"file '{w}'" for w in chunk_wavs))
        raw_wav = tmp / "raw_combined.wav"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), str(raw_wav)])
        raw_dur = get_wav_duration(raw_wav)
        print(f"  Celková dĺžka TTS: {raw_dur:.2f}s (slot: {SEG_DUR:.2f}s)", flush=True)

        # 3. Natiahnuť/skrátiť na presný slot
        print("\n[3] Fit do slotu...", flush=True)
        stretch_ratio = raw_dur / SEG_DUR
        print(f"  stretch ratio: {stretch_ratio:.3f}", flush=True)
        fitted_wav = tmp / "fitted.wav"
        if abs(stretch_ratio - 1.0) < 0.02:
            # Takmer presné — len skopíruj
            fitted_wav = raw_wav
            print("  Ratio ≈ 1.0 — bez stretchu", flush=True)
        elif stretch_ratio > 1.5 or stretch_ratio < 0.5:
            print(f"  WARN: ratio {stretch_ratio:.2f} je extrémny — slot je príliš dlhý pre tento text.", flush=True)
            print("  Použijem raw audio + ticho na konci.", flush=True)
            # Pridaj ticho na koniec aby sme vyplnili slot
            silence_dur = SEG_DUR - raw_dur
            run(["ffmpeg", "-y",
                 "-i", str(raw_wav),
                 "-af", f"apad=pad_dur={silence_dur:.3f}",
                 "-t", str(SEG_DUR),
                 str(fitted_wav)])
        else:
            # atempo: max 2.0 per filter, reťazíme ak treba
            tempo = 1.0 / stretch_ratio  # > 1 = zrýchlenie, < 1 = spomalenie
            if tempo > 2.0:
                af = f"atempo=2.0,atempo={tempo/2.0:.4f}"
            elif tempo < 0.5:
                af = f"atempo=0.5,atempo={tempo/0.5:.4f}"
            else:
                af = f"atempo={tempo:.4f}"
            run(["ffmpeg", "-y", "-i", str(raw_wav), "-af", af, str(fitted_wav)])

        fitted_dur = get_wav_duration(fitted_wav)
        print(f"  Fitted dĺžka: {fitted_dur:.2f}s", flush=True)

        # 4. Záloha pôvodného TTS wav
        backup = TTS_WAV.with_suffix(".wav.bak")
        if not backup.exists():
            print(f"\n[4] Záloha: {TTS_WAV.name} → {backup.name}", flush=True)
            import shutil
            shutil.copy2(TTS_WAV, backup)

        # 5. Nahraď slot v pôvodnom TTS wav
        print(f"\n[5] Nahrádzanie slotu {SEG_START}s–{SEG_END}s v TTS wav...", flush=True)
        new_tts_wav = tmp / "new_tts.wav"

        # Čas pred segmentom (0 → SEG_START)
        before = tmp / "before.wav"
        run(["ffmpeg", "-y", "-i", str(TTS_WAV),
             "-ss", "0", "-to", str(SEG_START),
             "-c", "copy", str(before)])

        # Čas po segmente (SEG_END → koniec)
        after = tmp / "after.wav"
        run(["ffmpeg", "-y", "-i", str(TTS_WAV),
             "-ss", str(SEG_END),
             "-c", "copy", str(after)])

        # Spoj: before + fitted + after
        concat2 = tmp / "concat2.txt"
        concat2.write_text(f"file '{before}'\nfile '{fitted_wav}'\nfile '{after}'")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat2), str(new_tts_wav)])

        new_tts_dur = get_wav_duration(new_tts_wav)
        orig_tts_dur = get_wav_duration(TTS_WAV)
        print(f"  Pôvodné TTS: {orig_tts_dur:.2f}s → nové: {new_tts_dur:.2f}s", flush=True)

        # 6. Zmiešaj nové TTS audio s videom
        print(f"\n[6] Finálne video: {VIDEO_OUT}", flush=True)
        run(["ffmpeg", "-y",
             "-i", str(VIDEO_IN),
             "-i", str(new_tts_wav),
             "-map", "0:v",
             "-map", "1:a",
             "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k",
             "-shortest",
             str(VIDEO_OUT)])

    print(f"\n✓ Hotovo: {VIDEO_OUT}", flush=True)
    print(f"  Skontroluj slot {SEG_START}s–{SEG_END}s v opravenom videu.", flush=True)


if __name__ == "__main__":
    main()
