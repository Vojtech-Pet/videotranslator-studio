#!/usr/bin/env python3
"""
Download FLEURS Slovak dataset and save WAV files + metadata.

Dataset:  google/fleurs, config "sk_sk"
License:  CC-BY-4.0
Audio:    WAV, 16 kHz
Hours:    ~10 hours
Splits:   train, validation, test
Fields:   id (int32), audio, transcription, raw_transcription, gender, ...

Saves WAVs to  : /mnt/tts_data/sk_tts/all_wavs/
Saves metadata : /mnt/tts_data/sk_tts/metadata_fleurs.json
"""

import json
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset
from pathlib import Path

OUTPUT_WAV_DIR = Path("/mnt/tts_data/sk_tts/all_wavs")
METADATA_PATH  = Path("/mnt/tts_data/sk_tts/metadata_fleurs.json")

# "transcription" is the cleaned/normalized form; use "raw_transcription" for original.
TRANSCRIPT_FIELD = "transcription"

OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)


def process_split(dataset_split, split_name: str, metadata: list):
    skipped = 0
    for i, sample in enumerate(dataset_split):
        text = sample.get(TRANSCRIPT_FIELD, "").strip()
        if not text:
            skipped += 1
            continue

        # id is an int32 in FLEURS
        numeric_id = sample["id"]
        file_id    = f"fleurs_sk_{split_name}_{numeric_id:05d}"
        wav_path   = OUTPUT_WAV_DIR / f"{file_id}.wav"

        # audio field: {"array": np.ndarray float32, "sampling_rate": int, "path": str}
        audio_array   = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]   # 16000

        # Convert float32 → int16 PCM
        if audio_array.dtype != np.int16:
            audio_int16 = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)
        else:
            audio_int16 = audio_array

        sf.write(str(wav_path), audio_int16, sampling_rate, subtype="PCM_16")

        metadata.append({"id": file_id, "text": text})

        if (i + 1) % 200 == 0:
            print(f"  [{split_name}] processed {i + 1} samples …")

    print(f"  [{split_name}] done — written so far: {len(metadata)}, skipped: {skipped}")


def main():
    print("Loading google/fleurs (sk_sk) …")
    ds = load_dataset("google/fleurs", "sk_sk", trust_remote_code=True)

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
