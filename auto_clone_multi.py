#!/usr/bin/env python3
"""
Multi-voice cloning — per-speaker SK-flavored ref generation.

Pipeline:
  1. Načíta diarized segments JSON (so speaker_id per segment)
  2. Pre každého unikátneho speakera:
     a. Nájde najdlhší / najčistejší segment od tohto speakera
     b. Extract audio chunk z videa
     c. Cleanup + European EQ (ako auto_clone_ref)
     d. Chatterbox SK ref-gen (PRO config) → SK-flavored ref
     e. Save: voices/<video_stem>/<SPEAKER_XX>_clone.wav
  3. Output: speaker_refs JSON
     {"SPEAKER_00": "/path/SPEAKER_00_clone.wav",
      "SPEAKER_01": "/path/SPEAKER_01_clone.wav", ...}

Použitie:
  ./auto_clone_multi.py video.mp4 --diarized seg.json
    # output: voices/<stem>/_speaker_refs.json + per-speaker WAV files

  ./auto_clone_multi.py video.mp4 --diarized seg.json --no-via-chatterbox
    # raw video clones bez Chatterbox (pre OmniVoice ako zero-shot)
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
VOICES_DIR = REPO / "voices"
CACHE_REF_TEXT = VOICES_DIR / ".ref_text_cache"

# Seed text pre Chatterbox SK ref-gen (musí matchovať čo Chatterbox vyrobí)
_CHATTERBOX_SEED_TEXT = (
    "Vitajte, dnes si spolu vysvetlíme niekoľko zaujímavých vecí. "
    "Začneme od základov a postupne sa dostaneme aj k zložitejším témam."
)


def cleanup_speaker_ref(in_wav: Path, out_wav: Path, european: bool = True) -> bool:
    """Lightweight cleanup pre per-speaker concat audio.
    Žiadny demucs (audio je už čisté), žiadny find_clean_section (krátke clipy).
    Iba: highpass + denoise + loudnorm + voliteľne European EQ.
    """
    af_parts = [
        "highpass=f=70",
        "afftdn=nr=8:nt=w:om=o",
    ]
    if european:
        af_parts += [
            "equalizer=f=200:t=q:w=0.9:g=+2.5",
            "equalizer=f=400:t=q:w=1.0:g=-1.0",
            "equalizer=f=1000:t=q:w=1.0:g=-1.5",
            "equalizer=f=2200:t=q:w=1.4:g=-3.5",
            "equalizer=f=3500:t=q:w=1.2:g=-3.0",
            "equalizer=f=5500:t=q:w=1.0:g=-2.0",
            "lowpass=f=7500",
        ]
    af_parts.append("loudnorm=I=-20:TP=-2:LRA=7")
    af = ",".join(af_parts)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_wav), "-af", af,
             "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(out_wav)],
            capture_output=True, check=True, timeout=60,
        )
        return out_wav.exists() and out_wav.stat().st_size > 1000
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-300:]
        print(f"   ⚠ cleanup_speaker_ref ffmpeg err: {err}")
        return False


def chatterbox_sk_ref_gen(video_clone_path: Path, out_path: Path) -> bool:
    """Run Chatterbox SK on video_clone audio_prompt → SK-flavored ref.
    PRO config (CONFIG_ANTICUT) + retry 3× + safe_mode fallback (rovnaká logika ako auto_clone_ref).
    """
    cb_model_paths = [
        REPO / "models" / "chatterbox" / "t3_sk_v2.2.safetensors",
        Path("/mnt/tts_data/VideoTranslator_studio/models/chatterbox/t3_sk_v2.2.safetensors"),
    ]
    cb_src_dirs = [
        REPO.parent / "Ai" / "chatterbox_git" / "src",
        REPO.parent / "chatterbox_git" / "src",
    ]
    cb_model = next((p for p in cb_model_paths if p.exists()), None)
    cb_src = next((d for d in cb_src_dirs if d.exists()), None)
    if not cb_model or not cb_src:
        print(f"   ⚠ Chatterbox model alebo source dir chýba — skipujem ref-gen")
        return False

    py = "/mnt/tts_data/miniforge3/envs/musetalk_env/bin/python"
    scripts_dir = REPO / "scripts"

    def _make_code(safe: bool) -> str:
        if safe:
            params = "    safe_mode=True,\n    exaggeration=0.45,\n    cfg_weight=0.50,\n    temperature=0.55,\n"
        else:
            params = "    safe_mode=False,\n    exaggeration=0.50,\n    cfg_weight=0.50,\n    temperature=0.60,\n"
        return (
            f"import sys\n"
            f"sys.path.insert(0, '{scripts_dir}')\n"
            f"sys.path.insert(0, '{cb_src}')\n"
            f"from tts import chatterbox_tts\n"
            f"from pathlib import Path\n"
            f"chatterbox_tts(\n"
            f"    text={_CHATTERBOX_SEED_TEXT!r},\n"
            f"    output_path=Path('{out_path}'),\n"
            f"    model_path='{cb_model}',\n"
            f"    audio_prompt_path='{video_clone_path}',\n"
            f"    device='cuda',\n"
            f"    ultra_safe=False,\n"
            f"{params}"
            f"    use_torch_compile=True,\n"
            f")\n"
            f"print('OK', flush=True)\n"
        )

    for attempt in range(3):
        out_path.unlink(missing_ok=True)
        r = subprocess.run([py, "-c", _make_code(safe=False)],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"   ✓ Chatterbox SK ref (PRO, attempt {attempt+1}/3, "
                  f"{out_path.stat().st_size//1024} KB)")
            return True
    # Fallback: safe_mode
    for attempt in range(2):
        out_path.unlink(missing_ok=True)
        r = subprocess.run([py, "-c", _make_code(safe=True)],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"   ✓ Chatterbox SK ref (safe fallback, "
                  f"{out_path.stat().st_size//1024} KB)")
            return True
    print(f"   ⚠ Chatterbox SK ref-gen úplne zlyhalo")
    return False


def save_ref_text_cache(ref_wav: Path, text: str):
    """Save seed text as ref_text cache do oboch lokácií (kde test_omnivoice_batch hľadá)."""
    # Global cache
    CACHE_REF_TEXT.mkdir(parents=True, exist_ok=True)
    (CACHE_REF_TEXT / f"{ref_wav.stem}.txt").write_text(text, encoding="utf-8")
    # Local cache (test_omnivoice_batch hľadá tu)
    local = ref_wav.parent / ".ref_text_cache"
    local.mkdir(parents=True, exist_ok=True)
    (local / f"{ref_wav.stem}.txt").write_text(text, encoding="utf-8")


def get_speaker_segments(segments: list, speaker_id: str) -> list:
    """Vráti list (start, end) tuple-ov pre daného speakera, sorted chronologicky."""
    return sorted(
        [(float(s["start"]), float(s["end"]))
         for s in segments if s.get("speaker_id") == speaker_id],
        key=lambda x: x[0]
    )


def find_best_segment_for_speaker(segments: list, speaker_id: str,
                                   min_dur: float = 5.0) -> tuple:
    """Vráti (start, end) najdlhšieho jediného segmentu pre daného speakera.
    Ak žiadny segment nie je >= min_dur, vráti None — caller použije concat fallback.
    """
    spk_segs = get_speaker_segments(segments, speaker_id)
    if not spk_segs:
        return None
    # Vráť iba single longest segment ≥ min_dur. Inak None — caller nech robí concat.
    longest = max(spk_segs, key=lambda x: x[1] - x[0])
    if longest[1] - longest[0] >= min_dur:
        return longest
    return None


def main():
    ap = argparse.ArgumentParser(description="Multi-voice per-speaker SK ref generator")
    ap.add_argument("video", help="Input video MP4")
    ap.add_argument("--diarized", required=True,
                    help="Diarized segments JSON (output z auto_diarize.py)")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir pre per-speaker refs (default: voices/<video_stem>/)")
    ap.add_argument("--via-chatterbox", action=argparse.BooleanOptionalAction, default=True,
                    help="Použi Chatterbox SK ref-gen (default ON)")
    ap.add_argument("--target-dur", type=float, default=5.5,
                    help="Target ref dur per speaker (default 5.5s)")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    diar_path = Path(args.diarized).expanduser().resolve()
    if not video.exists():
        sys.exit(f"Video neexistuje: {video}")
    if not diar_path.exists():
        sys.exit(f"Diarized JSON neexistuje: {diar_path}")

    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else VOICES_DIR / video.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(diar_path.read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    speakers = sorted(set(s.get("speaker_id", "SPEAKER_00") for s in segs))
    print(f"==> Multi-voice ref-gen")
    print(f"   video:    {video.name}")
    print(f"   speakers: {len(speakers)} ({', '.join(speakers)})")
    print(f"   out_dir:  {out_dir}")
    print()

    # Pre single-speaker video padneme na štandardný auto_clone_ref tok
    if len(speakers) == 1:
        print(f"   Single speaker — pre-multi-voice nepotrebné. Volaj auto_clone_ref.py priamo.")
        return

    speaker_refs = {}
    auto_clone_script = REPO / "auto_clone_ref.py"
    py = "/mnt/tts_data/miniforge3/envs/musetalk_env/bin/python"

    for spk_id in speakers:
        print(f"\n--- {spk_id} ---")
        # 1) Try single longest segment ≥ target_dur
        seg_range = find_best_segment_for_speaker(segs, spk_id, min_dur=args.target_dur)
        # 2) Build extraction (single range OR concat speaker-only segments)
        temp_clip = out_dir / f"{spk_id}_temp.wav"
        spk_segs = get_speaker_segments(segs, spk_id)
        total_spk_dur = sum(e - s for s, e in spk_segs)
        if total_spk_dur < 1.5:
            print(f"   ⚠ Speaker {spk_id} má iba {total_spk_dur:.1f}s celkom — skipujem (príliš krátky)")
            continue

        if seg_range:
            start, end = seg_range
            print(f"   Best single segment: {start:.1f}-{end:.1f}s ({end-start:.1f}s)")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(start), "-t", str(end - start),
                     "-i", str(video), "-ac", "1", "-ar", "16000",
                     "-c:a", "pcm_s16le", str(temp_clip)],
                    capture_output=True, check=True, timeout=60,
                )
            except subprocess.CalledProcessError as e:
                print(f"   ⚠ ffmpeg extract zlyhal: {e}")
                continue
        else:
            # Concat fallback: extract iba segmenty tohto speakera + spojí ich
            # do jedného WAVu (žiadne audio iných speakerov)
            print(f"   No single segment ≥ {args.target_dur}s; concat {len(spk_segs)} speaker segments "
                  f"(total {total_spk_dur:.1f}s)")
            chunks = []
            for i, (s, e) in enumerate(spk_segs):
                chunk_path = out_dir / f"{spk_id}_chunk_{i:02d}.wav"
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-ss", str(s), "-t", str(e - s),
                         "-i", str(video), "-ac", "1", "-ar", "16000",
                         "-c:a", "pcm_s16le", str(chunk_path)],
                        capture_output=True, check=True, timeout=30,
                    )
                    chunks.append(chunk_path)
                except subprocess.CalledProcessError:
                    continue
                # Cap pri ~8s spolu (Chatterbox ref nemusí byť dlhšie)
                if sum(_e - _s for _s, _e in spk_segs[:len(chunks)]) >= 8.0:
                    break
            if not chunks:
                print(f"   ⚠ Žiadne chunky extrahované pre {spk_id}, skipujem")
                continue
            # Concat list file
            concat_list = out_dir / f"{spk_id}_concat.txt"
            concat_list.write_text("\n".join(f"file '{c}'" for c in chunks))
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(concat_list), "-ac", "1", "-ar", "16000",
                     "-c:a", "pcm_s16le", str(temp_clip)],
                    capture_output=True, check=True, timeout=60,
                )
            except subprocess.CalledProcessError as e:
                print(f"   ⚠ ffmpeg concat zlyhal: {e}")
                for c in chunks:
                    c.unlink(missing_ok=True)
                concat_list.unlink(missing_ok=True)
                continue
            # Cleanup chunks
            for c in chunks:
                c.unlink(missing_ok=True)
            concat_list.unlink(missing_ok=True)
            print(f"   ✓ Concat ref: {temp_clip.name}")

        # 3) Lightweight cleanup (highpass + denoise + European EQ + loudnorm)
        # Žiadny demucs (audio je už speaker-only), žiadny find_clean_section
        # (pre krátke concated clipy zlyháva).
        cleaned_clip = out_dir / f"{spk_id}_cleaned.wav"
        if not cleanup_speaker_ref(temp_clip, cleaned_clip, european=True):
            print(f"   ⚠ cleanup zlyhal pre {spk_id}, skipujem")
            temp_clip.unlink(missing_ok=True)
            continue
        temp_clip.unlink(missing_ok=True)
        out_wav = out_dir / f"{spk_id}_clone.wav"

        if args.via_chatterbox:
            # Chatterbox SK ref-gen → SK-flavored ref s týmto speaker timbre
            print(f"   Running Chatterbox SK ref-gen (PRO config)...")
            ok = chatterbox_sk_ref_gen(cleaned_clip, out_wav)
            if not ok:
                # Fallback: použi raw cleaned clip ako ref (US accent ale aspoň funguje)
                shutil.move(cleaned_clip, out_wav)
                print(f"   Fallback: raw cleaned ref ({out_wav.stat().st_size//1024} KB)")
                # Bez Chatterbox ref-textu — Whisper transcribe v omnivoice runner urobí neskôr
            else:
                cleaned_clip.unlink(missing_ok=True)
                # Save SK seed text ako ref_text (Chatterbox to povedalo)
                save_ref_text_cache(out_wav, _CHATTERBOX_SEED_TEXT)
        else:
            # No Chatterbox — použi raw cleaned ako ref (zachová original speaker accent)
            shutil.move(cleaned_clip, out_wav)
            print(f"   ✓ Raw cleaned ref ({out_wav.stat().st_size//1024} KB)")

        if out_wav.exists() and out_wav.stat().st_size > 1000:
            speaker_refs[spk_id] = str(out_wav)
            print(f"   ✓ Speaker {spk_id} ref: {out_wav.name}")

    # 4) Save speaker_refs JSON
    refs_json = out_dir / "_speaker_refs.json"
    refs_json.write_text(json.dumps(speaker_refs, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\n✓ Speaker refs: {refs_json}")
    print(f"  Použij: ./test_omnivoice_batch.py --segments {diar_path} --speaker-refs {refs_json}")


if __name__ == "__main__":
    main()
