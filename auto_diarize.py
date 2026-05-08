#!/usr/bin/env python3
"""
Speaker diarization cez Resemblyzer + KMeans clustering.

Inou cestou ako pyannote (ktorá má dep conflict s torch 2.0):
  1. Pre každý segment → extract audio chunk → Resemblyzer embedding (256-dim)
  2. KMeans cluster všetky embeddings (n_clusters auto-pick alebo --n-speakers)
  3. Mapuj cluster_id → SPEAKER_00, SPEAKER_01, ... (relabel by chronological order)
  4. Output: JSON s pridaným speaker_id per segment

Kvalita: ~85-90% accuracy pre 2-3 speakers (pyannote má 95%, ale stačí pre dabbing).

Použitie:
  ./auto_diarize.py video.mp4 --segments seg.json
    # výstup: <seg.json>.diarized.json + tlač zhrnutia

  ./auto_diarize.py video.mp4 --segments seg.json --n-speakers 2
    # explicit 2 speakers
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


def extract_segment_audio(video: Path, start: float, dur: float, out_wav: Path):
    """Extract chunk z videa do mono 16kHz WAV (Resemblyzer requirement)."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(dur),
         "-i", str(video), "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out_wav)],
        capture_output=True, check=True, timeout=60,
    )


def auto_n_speakers(embeddings: np.ndarray, max_k: int = 5) -> int:
    """Vypočítaj optimálne n_speakers cez silhouette score (k=2..max_k).
    Ak najlepší score < 0.15, default 1 speaker (homogeneous voice).
    """
    if len(embeddings) < 4:
        return 1
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k = 1
    best_score = -1.0
    for k in range(2, min(max_k + 1, len(embeddings))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
        labels = km.labels_
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
    if best_score < 0.15:
        return 1  # homogeneous → single speaker
    return best_k


def cluster_speakers(embeddings: np.ndarray, n_speakers: int) -> np.ndarray:
    """KMeans clustering. Returns array of cluster labels (0..n_speakers-1)."""
    if n_speakers <= 1:
        return np.zeros(len(embeddings), dtype=int)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_speakers, random_state=42, n_init=10).fit(embeddings)
    return km.labels_


def relabel_chronological(labels: np.ndarray, segments: list) -> np.ndarray:
    """Premenuj clustre tak aby SPEAKER_00 = first speaker chronologicky.
    Idea: prvý segment → SPEAKER_00, druhý nový speaker → SPEAKER_01, atď.
    """
    seen = {}
    counter = 0
    new_labels = np.zeros_like(labels)
    for i, lbl in enumerate(labels):
        if lbl not in seen:
            seen[lbl] = counter
            counter += 1
        new_labels[i] = seen[lbl]
    return new_labels


def main():
    ap = argparse.ArgumentParser(description="Resemblyzer-based speaker diarization")
    ap.add_argument("video", help="Input video MP4")
    ap.add_argument("--segments", required=True, help="Segments JSON (with start/end)")
    ap.add_argument("--out", default=None,
                    help="Output JSON (default: <segments>.diarized.json)")
    ap.add_argument("--n-speakers", type=int, default=0,
                    help="N speakers (0 = auto-detect cez silhouette score)")
    ap.add_argument("--min-segment-dur", type=float, default=1.0,
                    help="Min segment duration v sekundách (default 1.0; "
                         "kratšie segmenty sa pre embedding skipujú a label sa interpoluje)")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    seg_path = Path(args.segments).expanduser().resolve()
    if not video.exists() or not seg_path.exists():
        sys.exit(f"Vstup neexistuje: video={video.exists()}, segments={seg_path.exists()}")

    out_path = Path(args.out) if args.out else seg_path.with_suffix(".diarized.json")

    print(f"==> Auto-diarize")
    print(f"   video:    {video.name}")
    print(f"   segments: {seg_path.name}")
    print(f"   out:      {out_path.name}")
    print()

    data = json.loads(seg_path.read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    print(f"   Total segments: {len(segs)}")

    # 1) Compute Resemblyzer embedding per segment
    print(f"   [1/3] Extracting audio chunks + computing embeddings...", flush=True)
    from resemblyzer import VoiceEncoder, preprocess_wav
    enc = VoiceEncoder(verbose=False)
    embeddings = []
    valid_idx = []  # segment indices that got valid embedding
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, seg in enumerate(segs):
            start = float(seg.get("start", 0))
            end = float(seg.get("end", start))
            dur = end - start
            if dur < args.min_segment_dur:
                continue
            chunk_wav = tmp_dir / f"seg_{i:04d}.wav"
            try:
                extract_segment_audio(video, start, dur, chunk_wav)
                wav = preprocess_wav(str(chunk_wav))
                if len(wav) < 8000:  # < 0.5s after preprocess (silero VAD strips silences)
                    continue
                emb = enc.embed_utterance(wav)
                embeddings.append(emb)
                valid_idx.append(i)
            except Exception as e:
                print(f"   seg {i}: skip ({e})")
                continue
            if (i + 1) % 20 == 0:
                print(f"   ... {i+1}/{len(segs)} processed", flush=True)

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"   ✓ {len(embeddings)} embeddings (skipped {len(segs) - len(embeddings)} short/invalid)")

    if len(embeddings) < 2:
        print(f"   ⚠ Príliš málo embeddings, nedá sa robiť clustering — všetkým assignujem SPEAKER_00")
        for s in segs:
            s["speaker_id"] = "SPEAKER_00"
        out_path.write_text(json.dumps({"segments": segs}, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        return

    # 2) Determine n_speakers
    if args.n_speakers > 0:
        n_spk = args.n_speakers
        print(f"   [2/3] N speakers: {n_spk} (from --n-speakers)")
    else:
        n_spk = auto_n_speakers(embeddings)
        print(f"   [2/3] N speakers: {n_spk} (auto-detected via silhouette)")

    # 3) Cluster + relabel chronologically
    print(f"   [3/3] KMeans clustering + chronological relabel...")
    labels = cluster_speakers(embeddings, n_spk)
    # Build mapping valid_idx[i] → label[i]
    valid_to_label = dict(zip(valid_idx, labels))
    # Relabel chronologically (first appearance → SPEAKER_00)
    relabeled = relabel_chronological(labels, [segs[i] for i in valid_idx])
    valid_to_label = dict(zip(valid_idx, relabeled))

    # Assign labels to ALL segments (interpolate skipped ones from nearest neighbor)
    last_lbl = 0
    for i, seg in enumerate(segs):
        if i in valid_to_label:
            last_lbl = int(valid_to_label[i])
        seg["speaker_id"] = f"SPEAKER_{last_lbl:02d}"

    # Summary
    counts = Counter(s["speaker_id"] for s in segs)
    print()
    print(f"   ✓ Diarization done. Speakers detected: {n_spk}")
    for spk_id, cnt in sorted(counts.items()):
        pct = 100 * cnt / len(segs)
        print(f"     {spk_id}: {cnt} segments ({pct:.0f}%)")

    out_data = {"segments": segs} if isinstance(data, dict) else segs
    if isinstance(data, dict):
        # preserve other top-level keys
        for k in data:
            if k != "segments":
                out_data[k] = data[k]
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Output: {out_path}")


if __name__ == "__main__":
    main()
