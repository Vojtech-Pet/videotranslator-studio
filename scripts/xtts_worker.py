"""
Standalone XTTS v2 worker — volá sa ako subprocess z hlavného pipeline.

Použitie:
    python xtts_worker.py --text "Text na syntetizovanie" --output out.wav
                          [--lang cs] [--speaker_wav ref.wav] [--device cuda]
                          [--speaker "Damien Black"] [--pitch_shift -2]

--speaker       : meno built-in hlasu (pozri zoznam nižšie)
--pitch_shift   : semitóny (záporné = nižší hlas, kladné = vyšší); 0 = bez zmeny
                  Implementované cez asetrate+atempo (FFmpeg musí byť v PATH)

Výstup: WAV súbor na --output ceste, 24000 Hz mono.
"""

import argparse
import sys
import os
import subprocess
import shutil
from pathlib import Path

import torch
import soundfile as sf
import numpy as np

# Mužské built-in hlasy (XTTS v2):
# Damien Black, Craig Gutsy, Torcull Diarmuid, Viktor Menelaos,
# Luis Moray, Marcos Rudaski, Wulf Carlevaro, Aaron Dreschner,
# Kumar Dahl, Eugenio Mataracı, Ferran Simen, Xavier Hayasaka,
# Badr Odhiambo, Dionisio Schuyler, Royston Min, Viktor Eka,
# Abrahan Mack, Adde Michal, Baldur Sanjin, Gilberto Mathias,
# Ilkin Urbano, Kazuhiko Atallah, Ludvig Milivoj, Suad Qasim,
# Zacharie Aimilios, Filip Traverse, Damjan Chapman

DEFAULT_SPEAKER = "Damien Black"


def _apply_pitch_shift(wav_path: Path, semitones: float, sr: int = 24000) -> None:
    """Shift pitch by `semitones` without changing duration via asetrate+atempo."""
    if semitones == 0:
        return
    factor = 2 ** (semitones / 12.0)
    new_rate = int(sr * factor)
    # atempo must be in [0.5, 2.0]; chain two filters if needed
    tempo = 1.0 / factor
    if 0.5 <= tempo <= 2.0:
        tempo_chain = f"atempo={tempo:.6f}"
    elif tempo < 0.5:
        tempo_chain = f"atempo=0.5,atempo={tempo/0.5:.6f}"
    else:
        tempo_chain = f"atempo=2.0,atempo={tempo/2.0:.6f}"

    tmp = wav_path.with_name(wav_path.stem + "_ps.wav")
    af = f"asetrate={new_rate},{tempo_chain},aresample={sr}"
    cmd = ["ffmpeg", "-y", "-i", str(wav_path), "-af", af,
           "-ar", str(sr), "-c:a", "pcm_s16le", str(tmp)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 500:
        shutil.move(str(tmp), str(wav_path))
        print(f"[XTTS] Pitch shift {semitones:+.1f} st applied", flush=True)
    else:
        print(f"[XTTS] Pitch shift failed: {result.stderr[:200]}", flush=True)
        if tmp.exists():
            tmp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--output", required=True, help="Output WAV path")
    ap.add_argument("--lang", default="cs", help="Language code (cs, sk, en...)")
    ap.add_argument("--speaker_wav", default="", help="Reference voice WAV for cloning")
    ap.add_argument("--speaker", default=DEFAULT_SPEAKER,
                    help=f"Built-in speaker name (default: {DEFAULT_SPEAKER})")
    ap.add_argument("--pitch_shift", type=float, default=0.0,
                    help="Pitch shift in semitones (negative = lower, e.g. -3)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_name", default="tts_models/multilingual/multi-dataset/xtts_v2")
    args = ap.parse_args()

    os.environ["COQUI_TOS_AGREED"] = "1"

    from TTS.api import TTS

    print(f"[XTTS] Loading model {args.model_name} on {args.device}...", flush=True)
    tts = TTS(model_name=args.model_name, progress_bar=False).to(args.device)
    print("[XTTS] Model loaded", flush=True)

    text = args.text.strip()
    if not text:
        print("[XTTS] Empty text — writing silence", flush=True)
        silence = np.zeros(int(24000 * 0.5), dtype=np.float32)
        sf.write(args.output, silence, 24000)
        return

    speaker_wav = args.speaker_wav.strip() if args.speaker_wav else None
    if speaker_wav and not Path(speaker_wav).exists():
        print(f"[XTTS] WARNING: speaker_wav not found: {speaker_wav}, using built-in voice", flush=True)
        speaker_wav = None

    print(f"[XTTS] Synthesizing ({len(text)} chars, lang={args.lang})...", flush=True)

    if speaker_wav:
        wav = tts.tts(text=text, speaker_wav=speaker_wav, language=args.lang)
    else:
        # Resolve requested speaker name; fall back to first available
        try:
            sm = tts.synthesizer.tts_model.speaker_manager
            available = list(sm.name_to_id)
        except Exception:
            available = []

        speaker = args.speaker if args.speaker in available else (available[0] if available else None)
        if speaker != args.speaker and args.speaker:
            print(f"[XTTS] Speaker '{args.speaker}' not found, using '{speaker}'", flush=True)

        if speaker:
            print(f"[XTTS] Using built-in speaker: {speaker}", flush=True)
            wav = tts.tts(text=text, speaker=speaker, language=args.lang)
        else:
            wav = tts.tts(text=text, language=args.lang)

    wav = np.array(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Normalize
    peak = float(np.max(np.abs(wav)))
    if peak > 0.01:
        wav = wav / peak * 0.92

    sf.write(args.output, wav, 24000, subtype="PCM_16")
    dur = len(wav) / 24000
    print(f"[XTTS] Done — {dur:.2f}s → {args.output}", flush=True)

    # Optional pitch shift (post-processing via FFmpeg)
    if args.pitch_shift != 0.0:
        _apply_pitch_shift(Path(args.output), args.pitch_shift)


if __name__ == "__main__":
    main()
