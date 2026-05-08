#!/usr/bin/env python3
"""
Voice fingerprint cache pre RVC voice models.

Použitie:
  ./voice_fingerprint.py compute <wav>           # vypíše 256-dim embedding ako JSON
  ./voice_fingerprint.py find <wav>              # nájde najpodobnejší cached model
  ./voice_fingerprint.py register <wav> <name>   # uloží embedding do cache
  ./voice_fingerprint.py list                    # vypíše všetky cached modely

Cache layout:
  /mnt/tts_data/rvc/voice_models/
    <name>/
      model.pth         # RVC weights (uložené trénovaním)
      index.bin         # RVC search index
      embedding.npy     # Resemblyzer 256-dim
      meta.json         # source_video, created, train_epochs, ...
    cache.json          # zoznam všetkých modelov + ich embeddings (rýchle lookup)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

VOICE_MODELS_DIR = Path("/mnt/tts_data/rvc/voice_models")
CACHE_FILE = VOICE_MODELS_DIR / "cache.json"
SIMILARITY_THRESHOLD = 0.85  # cosine — over which we consider "same speaker"


def _ensure_dirs():
    VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_embedding(wav_path: Path) -> np.ndarray:
    """Vypočíta 256-dim Resemblyzer embedding z WAV súboru.
    Audio sa interne resampluje na 16kHz mono.
    """
    from resemblyzer import VoiceEncoder, preprocess_wav
    enc = VoiceEncoder(verbose=False)
    wav = preprocess_wav(str(wav_path))
    return enc.embed_utterance(wav).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def save_cache(cache: dict):
    _ensure_dirs()
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def find_matching_model(wav_path: Path, threshold: float = SIMILARITY_THRESHOLD) -> tuple:
    """Vráti (name, similarity, model_path) najpodobnejšieho cached modelu,
    alebo (None, best_sim, None) ak žiadny pod threshold."""
    cache = load_cache()
    if not cache:
        return (None, 0.0, None)
    new_emb = compute_embedding(wav_path)
    best = (None, -1.0, None)
    for name, entry in cache.items():
        cached_emb = np.array(entry["embedding"], dtype=np.float32)
        sim = cosine_sim(new_emb, cached_emb)
        if sim > best[1]:
            model_pth = VOICE_MODELS_DIR / name / "model.pth"
            best = (name, sim, model_pth if model_pth.exists() else None)
    if best[1] >= threshold and best[2] is not None:
        return best
    return (None, best[1], None)


def register_model(wav_path: Path, name: str, source_video: str = "",
                   train_epochs: int = 0):
    """Vypočíta embedding + uloží do cache + meta.json."""
    _ensure_dirs()
    model_dir = VOICE_MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    emb = compute_embedding(wav_path)
    np.save(model_dir / "embedding.npy", emb)
    meta = {
        "name": name,
        "source_video": source_video,
        "source_audio": str(wav_path),
        "created": datetime.utcnow().isoformat() + "Z",
        "train_epochs": train_epochs,
        "embedding_dim": int(emb.shape[0]),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cache = load_cache()
    cache[name] = {
        "embedding": emb.tolist(),
        "source_video": source_video,
        "created": meta["created"],
    }
    save_cache(cache)
    return meta


def list_models() -> list:
    cache = load_cache()
    rows = []
    for name, entry in cache.items():
        model_pth = VOICE_MODELS_DIR / name / "model.pth"
        rows.append({
            "name": name,
            "model_exists": model_pth.exists(),
            "source_video": entry.get("source_video", ""),
            "created": entry.get("created", ""),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Voice fingerprint cache pre RVC modely")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_compute = sub.add_parser("compute", help="Vypíše embedding z WAV")
    p_compute.add_argument("wav")

    p_find = sub.add_parser("find", help="Nájde najpodobnejší cached model")
    p_find.add_argument("wav")
    p_find.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)

    p_register = sub.add_parser("register",
                                 help="Vypočíta embedding + uloží do cache (model.pth predpokladá uložený)")
    p_register.add_argument("wav")
    p_register.add_argument("name", help="Názov modelu (= meno priečinku)")
    p_register.add_argument("--source-video", default="")
    p_register.add_argument("--train-epochs", type=int, default=0)

    p_list = sub.add_parser("list", help="Vypíše všetky cached modely")

    args = ap.parse_args()
    _ensure_dirs()

    if args.cmd == "compute":
        emb = compute_embedding(Path(args.wav).expanduser().resolve())
        print(json.dumps({"embedding": emb.tolist(), "dim": int(emb.shape[0])}))
    elif args.cmd == "find":
        name, sim, path = find_matching_model(Path(args.wav).expanduser().resolve(),
                                              threshold=args.threshold)
        result = {"matched": name is not None, "best_match": name,
                  "similarity": round(sim, 4),
                  "model_path": str(path) if path else None,
                  "threshold": args.threshold}
        print(json.dumps(result, indent=2))
    elif args.cmd == "register":
        meta = register_model(Path(args.wav).expanduser().resolve(), args.name,
                              source_video=args.source_video,
                              train_epochs=args.train_epochs)
        print(json.dumps(meta, indent=2))
    elif args.cmd == "list":
        rows = list_models()
        for r in rows:
            tag = "✓" if r["model_exists"] else "✗"
            print(f"  {tag} {r['name']:30s} created={r['created'][:10]} "
                  f"source={r['source_video']}")
        if not rows:
            print("  (žiadne modely)")


if __name__ == "__main__":
    main()
