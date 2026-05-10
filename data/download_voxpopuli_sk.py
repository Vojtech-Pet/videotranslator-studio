#!/usr/bin/env python3
"""
Download VoxPopuli Slovak dataset and save WAV files + metadata.

Dataset:  facebook/voxpopuli, config "sk"
License:  CC0 1.0 (Public Domain)
Audio:    WAV, 16 kHz
Hours:    ~35 hours (transcribed)
Splits:   train, validation, test
Fields:   audio_id, audio, raw_text, normalized_text, speaker_id, gender, ...

Saves WAVs to  : /mnt/tts_data/sk_tts/all_wavs/
Saves metadata : /mnt/tts_data/sk_tts/metadata_voxpopuli.json
"""

import json
import os
import re
import soundfile as sf
import numpy as np
from datasets import load_dataset
from pathlib import Path

OUTPUT_WAV_DIR = Path("/mnt/tts_data/sk_tts/all_wavs")
METADATA_PATH  = Path("/mnt/tts_data/sk_tts/metadata_voxpopuli.json")

# Use normalized_text (punctuation-stripped, lowercased) for TTS training.
# Switch to "raw_text" if you prefer the original casing/punctuation.
TRANSCRIPT_FIELD = "normalized_text"

OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_id(audio_id: str) -> str:
    """Replace any characters that are unsafe in filenames."""
    return re.sub(r"[^\w\-]", "_", audio_id)


def process_split(dataset_split, split_name: str, metadata: list):
    skipped = 0
    for i, sample in enumerate(dataset_split):
        text = sample.get(TRANSCRIPT_FIELD, "").strip()
        if not text:
            skipped += 1
            continue

        raw_id   = sample["audio_id"]          # e.g. "20100601-0900-PLENARY-sk_0"
        file_id  = f"voxpopuli_{sanitize_id(raw_id)}"
        wav_path = OUTPUT_WAV_DIR / f"{file_id}.wav"

        # audio field: {"array": np.ndarray, "sampling_rate": int, "path": str}
        audio_array  = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]   # should be 16000

        # Ensure float32 → int16 PCM for broad compatibility
        if audio_array.dtype != np.int16:
            audio_int16 = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)
        else:
            audio_int16 = audio_array

        sf.write(str(wav_path), audio_int16, sampling_rate, subtype="PCM_16")

        metadata.append({"id": file_id, "text": text})

        if (i + 1) % 500 == 0:
            print(f"  [{split_name}] processed {i + 1} samples …")

    print(f"  [{split_name}] done — {len(metadata)} saved, {skipped} skipped (no text)")


def main():
    print("Loading facebook/voxpopuli (sk) …")
    # trust_remote_code not needed; standard HF dataset
    ds = load_dataset("facebook/voxpopuli", "sk", trust_remote_code=False)

    metadata: list = []
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            print(f"  Split '{split_name}' not found, skipping.")
            continue
        print(f"\nProcessing split: {split_name} ({len(ds[split_name])} samples)")
        process_split(ds[split_name], split_name, metadata)

    print(f"\nWriting metadata → {METADATA_PATH}")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDone.  Total entries in metadata: {len(metadata)}")
    print(f"WAV files in: {OUTPUT_WAV_DIR}")
    print(f"Metadata at : {METADATA_PATH}")


if __name__ == "__main__":
    main()
