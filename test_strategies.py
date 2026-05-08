#!/usr/bin/env python3
"""
Compare 3 strategies for fixing OmniVoice trailing silence:
  A) Trailing comma → period
  B) A + post-trim trailing silence v audio
  C) A + padding krátkych textov (<15 znakov)
"""
import json, subprocess, re, shutil
from pathlib import Path
import numpy as np
import soundfile as sf

REPO = Path("/mnt/tts_data/VideoTranslator_studio_v2")
SEG_PATH = REPO / "temp" / "2 Years of C++ Programming" / "2 Years of C++ Programming_sk_segments.json"
OUT_DIR = REPO / "input" / "sk"
WORK = Path("/tmp/strategies")
WORK.mkdir(exist_ok=True, parents=True)

def fix_trailing_comma(text: str) -> str:
    """A: trailing čiarka → bodka."""
    t = text.strip()
    if t.endswith(','):
        return t[:-1] + '.'
    return t

def pad_short_text(text: str, min_chars: int = 15) -> str:
    """C: pad krátke texty bezpečným suffixom — diffusion model potrebuje min ~1.5s.
    Bezpečné padding: tichá frázová častica, ktorú TTS nenahlas vysloví ale dá modelu kontext."""
    t = text.strip()
    if len(t) >= min_chars:
        return t
    # Konzervatívne — duplikuj alebo pridaj prirodzený suffix
    if t.endswith('.') or t.endswith('!') or t.endswith('?'):
        return t + " " + t  # opakovanie — diffusion vidí dlhší kontext, ale výstup môže byť krátky
    return t + "."

def trim_trailing_silence(in_wav: Path, out_wav: Path, threshold_db: float = -45) -> float:
    """B: trim trailing silence pomocou ffmpeg silenceremove (reverse trick)."""
    # silenceremove na koniec cez areverse + cez minimal duration
    cmd = ['ffmpeg', '-y', '-i', str(in_wav),
           '-af', f'areverse,silenceremove=start_periods=1:start_duration=0.05:start_threshold={threshold_db}dB,areverse',
           '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', str(out_wav)]
    subprocess.run(cmd, capture_output=True, timeout=30)
    if not out_wav.exists():
        shutil.copy(in_wav, out_wav)
    dur = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(out_wav)],capture_output=True,text=True).stdout.strip())
    return dur

# 1) Načítaj segmenty
data = json.loads(SEG_PATH.read_text())
segs = data['segments']
texts = [(s.get('text') or '').strip() for s in segs]
print(f"Segments: {len(texts)}")

# 2) Vytvor variant A a C
texts_A = [fix_trailing_comma(t) for t in texts]
texts_C = [pad_short_text(fix_trailing_comma(t)) for t in texts]

# Save modified segments JSONs
for variant, txts in [('A', texts_A), ('C', texts_C)]:
    new_segs = [dict(s, text=t) for s, t in zip(segs, txts)]
    new_data = dict(data, segments=new_segs)
    p = WORK / f"segments_{variant}.json"
    p.write_text(json.dumps(new_data, ensure_ascii=False, indent=2))
    print(f"Saved {p.name}")

# 3) Spusti batch pre A a C s --keep-chunks
for variant in ['A', 'C']:
    chunks_dir = WORK / f"chunks_{variant}"
    chunks_dir.mkdir(exist_ok=True)
    out_wav = WORK / f"out_{variant}.wav"
    print(f"\n==> Running OmniVoice batch for variant {variant}...")
    cmd = [str(REPO / "test_omnivoice_batch.py"),
           "--segments", str(WORK / f"segments_{variant}.json"),
           "--out", str(out_wav),
           "--chunks-dir", str(chunks_dir),
           "--keep-chunks"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"FAIL variant {variant}: {r.stderr[-500:]}")
        continue
    print(f"  out: {out_wav.stat().st_size//1024//1024}MB")

# 4) Vyrob B z A chunks: post-trim trailing silence
print("\n==> Building variant B (post-trim trailing silence from A chunks)")
chunks_A = sorted((WORK / "chunks_A").glob("chunk_*.wav"))
chunks_B = WORK / "chunks_B"
chunks_B.mkdir(exist_ok=True)
for c in chunks_B.glob("chunk_*.wav"): c.unlink()

for c in chunks_A:
    out = chunks_B / c.name
    trim_trailing_silence(c, out)

# Concat B
list_B = WORK / "concat_B.txt"
list_B.write_text("\n".join(f"file '{c.absolute()}'" for c in sorted(chunks_B.glob("chunk_*.wav"))) + "\n")
out_B = WORK / "out_B.wav"
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(list_B),
                "-ac","1","-ar","24000","-c:a","pcm_s16le",str(out_B)], capture_output=True)

# 5) Comparison report
print("\n" + "="*80)
print("COMPARISON REPORT")
print("="*80)
for variant in ['A', 'B', 'C']:
    out = WORK / f"out_{variant}.wav"
    if not out.exists():
        print(f"\n{variant}: MISSING")
        continue
    audio, sr = sf.read(str(out))
    chunks_d = WORK / f"chunks_{variant}"
    chunks = sorted(chunks_d.glob("chunk_*.wav"))

    silence_ratios = []
    for c in chunks:
        a, _ = sf.read(str(c))
        sil = float(np.mean(np.abs(a) < 10**(-50/20)))
        silence_ratios.append(sil)

    high_silence = sum(1 for s in silence_ratios if s > 0.3)
    print(f"\n  Variant {variant}:")
    print(f"    Total dur: {len(audio)/sr:.1f}s")
    print(f"    # chunks with >30% silence: {high_silence}/{len(silence_ratios)}")
    print(f"    Avg silence ratio: {np.mean(silence_ratios)*100:.1f}%")
    print(f"    Max silence ratio: {max(silence_ratios)*100:.1f}%")

# 6) Copy finals to input/sk/
print("\n==> Copying outputs to input/sk/")
for variant in ['A', 'B', 'C']:
    src = WORK / f"out_{variant}.wav"
    if src.exists():
        dst = OUT_DIR / f"2 Years of C++ Programming_STRATEGY_{variant}.wav"
        shutil.copy(src, dst)
        # Aj MP3
        mp3 = dst.with_suffix(".mp3")
        subprocess.run(["ffmpeg","-y","-i",str(dst),"-codec:a","libmp3lame","-b:a","192k",str(mp3)], capture_output=True)
        print(f"  {dst.name}: {dst.stat().st_size//1024//1024}MB + .mp3")

print("\n✓ Done. Files in input/sk/:")
print("    _STRATEGY_A.{wav,mp3} — trailing comma → period")
print("    _STRATEGY_B.{wav,mp3} — A + post-trim trailing silence")
print("    _STRATEGY_C.{wav,mp3} — A + padding short texts (<15 chars)")
