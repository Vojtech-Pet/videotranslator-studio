#!/usr/bin/env python3
"""
Standalone OmniVoice tester — žiadny pipeline, žiadny post-processing.

Použitie:
  ./test_omnivoice.py "Tvoj text na vyslovenie."
  ./test_omnivoice.py --text "Toto je dva roky v C++." --out custom.wav
  ./test_omnivoice.py --file text.txt --ref voices/iny_hlas.wav

Defaults:
  ref_audio:  voices/chatterbox_sk_greeting_studio.wav
  language:   sk
  num_step:   64
  guidance:   2.5
  expressive: False
  output:     out_omnivoice.wav (v current dir)

Žiadny silence trim, žiadne EQ, žiadne timeline. Iba čisté OmniVoice audio.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def omnivoice_speak_symbols(text: str) -> str:
    """Inline kópia z tts.py — pre-process text pre OmniVoice."""
    out = text
    out = re.sub(r'\bC\+\+', 'C plus plus', out)
    out = re.sub(r'\bc\+\+', 'C plus plus', out)
    out = re.sub(r'\bC#', 'C sharp', out)
    out = re.sub(r'\bF#', 'F sharp', out)
    out = re.sub(r'\.NET\b', 'dot NET', out)
    out = re.sub(r'[„""""\'\']', '', out)
    out = out.replace('—', ',').replace('–', ',')
    out = out.replace('…', '.')
    out = re.sub(r'\.{3,}\s*([A-Za-zÁ-žÀ-ÿ])', r', \1', out)
    out = re.sub(r'(?:\.\s*){2,}\.', '.', out)
    out = re.sub(r'\.{2,}', '.', out)
    out = re.sub(r'\s+', ' ', out).strip()
    if out and out[-1] == ',':
        out = out[:-1] + '.'
    return out

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REF = REPO_ROOT / "voices" / "chatterbox_sk_greeting_studio.wav"
DEFAULT_PY = "/mnt/tts_data/miniforge3/envs/f5tts_env/bin/python"
DEFAULT_REPO = "k2-fsa/OmniVoice"


# Subprocess runner — beží v f5tts_env (kde je nainštalovaný omnivoice)
RUNNER = r'''
import json, sys, torch, soundfile as sf
from pathlib import Path
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

p = json.loads(sys.stdin.read())

# Auto-Whisper transcribe ref_audio ak nie je zadaný ref_text
ref_audio = p["ref_audio"]
ref_text = p.get("ref_text") or ""
if not ref_text and ref_audio:
    print(f"[REF] Auto Whisper transcribe {ref_audio}...", flush=True)
    try:
        from faster_whisper import WhisperModel
        wm = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
        segs, _ = wm.transcribe(ref_audio, language=p.get("ref_lang", "sk"), beam_size=1)
        ref_text = " ".join(s.text.strip() for s in segs).strip() or "Tu je tichý hlas."
        del wm
        print(f"[REF] ref_text: {ref_text!r}", flush=True)
    except Exception as e:
        print(f"[REF] Whisper failed: {e}, použijem fallback ref_text", flush=True)
        ref_text = "Toto je vzor slovenského hlasu."

print(f"[OMNIVOICE] Loading model {p['repo']}...", flush=True)
model = OmniVoice.from_pretrained(p['repo'], device_map='cuda:0', dtype=torch.float16)

cfg = OmniVoiceGenerationConfig(
    num_step=p['num_step'],
    guidance_scale=p['guidance_scale'],
    position_temperature=p.get('position_temperature', 3.0),
    postprocess_output=p.get('postprocess', False),  # default False = no internal trim
)

text = p['text']
print(f"[OMNIVOICE] Generating {len(text)} chars: {text!r}", flush=True)

audio = model.generate(
    text=text,
    language=p['language'],
    ref_audio=ref_audio,
    ref_text=ref_text,
    generation_config=cfg,
)

sf.write(p['output_path'], audio[0], 24000)
out_dur = len(audio[0]) / 24000.0
print(f"[OMNIVOICE] Saved: {p['output_path']} ({out_dur:.2f}s)", flush=True)
'''


def main():
    ap = argparse.ArgumentParser(description="Standalone OmniVoice TTS tester")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("text", nargs="?", help="Text na TTS (alebo použi --text/--file)")
    g.add_argument("--text", dest="text_flag", help="Text na TTS")
    g.add_argument("--file", help="Cesta k textovému súboru")

    ap.add_argument("--out", "-o", default="out_omnivoice.wav", help="Output WAV (default: out_omnivoice.wav)")
    ap.add_argument("--ref", default=str(DEFAULT_REF), help=f"Reference audio (default: {DEFAULT_REF.name})")
    ap.add_argument("--ref-text", default="", help="Reference transcript (default: auto-Whisper)")
    ap.add_argument("--lang", default="sk", help="Output language (default: sk)")
    ap.add_argument("--num-step", type=int, default=64, help="Diffusion steps (default: 64)")
    ap.add_argument("--guidance", type=float, default=2.5, help="Guidance scale (default: 2.5)")
    ap.add_argument("--postprocess", action="store_true", help="Zapnúť OmniVoice interný silence trim (default: False)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="OmniVoice model repo")
    ap.add_argument("--python", default=DEFAULT_PY, help="Python executable s omnivoice balíčkom")

    args = ap.parse_args()

    # Resolve text
    if args.text:
        text = args.text
    elif args.text_flag:
        text = args.text_flag
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        sys.exit("Chyba: zadaj text alebo --file")

    if not text:
        sys.exit("Chyba: prázdny text")

    # Aplikuj symbol expansion (C++ → C plus plus, .NET → dot NET, ...)
    expanded = omnivoice_speak_symbols(text)
    if expanded != text:
        print(f"   symbol expansion: {text!r} → {expanded!r}")
    text = expanded

    out_path = Path(args.out).expanduser().resolve()
    ref_path = Path(args.ref).expanduser().resolve()

    if not ref_path.exists():
        sys.exit(f"Chyba: ref audio neexistuje: {ref_path}")
    if not Path(args.python).exists():
        sys.exit(f"Chyba: python neexistuje: {args.python}")

    # Auto-load cached ref_text z voices/.ref_text_cache/<stem>.txt
    ref_text = args.ref_text
    if not ref_text:
        cache_path = ref_path.parent / ".ref_text_cache" / f"{ref_path.stem}.txt"
        if cache_path.exists():
            ref_text = cache_path.read_text(encoding="utf-8").strip()
            print(f"   ref_text: cached: {ref_text!r}")

    payload = {
        "text": text,
        "output_path": str(out_path),
        "repo": args.repo,
        "ref_audio": str(ref_path),
        "ref_text": ref_text,
        "ref_lang": args.lang,
        "language": args.lang,
        "num_step": args.num_step,
        "guidance_scale": args.guidance,
        "postprocess": args.postprocess,
    }

    print(f"==> OmniVoice test")
    print(f"   text:     {text!r}")
    print(f"   ref:      {ref_path}")
    print(f"   out:      {out_path}")
    print(f"   lang:     {args.lang}")
    print(f"   num_step: {args.num_step}")
    print(f"   guidance: {args.guidance}")
    print(f"   postproc: {args.postprocess}")
    print()

    proc = subprocess.run(
        [args.python, "-c", RUNNER],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=False,
        timeout=600,
    )

    if proc.returncode != 0:
        sys.exit(f"OmniVoice failed (rc={proc.returncode})")
    if not out_path.exists() or out_path.stat().st_size < 1000:
        sys.exit(f"Output empty/missing: {out_path}")

    print(f"\n✓ Hotové: {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
