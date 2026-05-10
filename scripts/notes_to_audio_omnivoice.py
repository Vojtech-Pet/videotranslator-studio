"""
notes_to_audio_omnivoice.py — markdown → audio (OmniVoice SK) s iteratívnym QC.

OmniVoice: k2-fsa, Apache 2.0, 600+ jazykov, zero-shot voice cloning.
Reference audio: juraj_sk_clean_3s.wav (3s clean clip, vyladený pre OmniVoice).

Použitie:
    python notes_to_audio_omnivoice.py vstup.md vystup.wav [--pilot]
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice

DEFAULT_REF_WAV = "/mnt/tts_data/VideoTranslator_studio/voices/lubo_sk_studio.wav"
DEFAULT_REF_TEXT = "Podrobne som zmapoval financovanie jej prvej kampane v roku 2002, ktoré už vtedy vzbudzovalo vážne pochybnosti. Počas volebnej noci som bol aj v centrále strany,"
DEFAULT_INSTRUCT = "male, young adult, moderate pitch"  # OmniVoice akceptuje len whitelist EN pojmy
DEFAULT_SPEED = 1.00  # prirodzené tempo, lepšie na učenie

# Retry seeds — keď QC detekuje issues, skúsi iné seedy. Znížené na 2 (predtým 5)
# lebo pri stabilných issue-och (napr. "too_short" pri clone WAV s trailing silence)
# sa všetky retry produkujú rovnaký výsledok → zbytočné mrhanie času.
QC_RETRY_SEEDS = [42, 0]
MAX_CHARS_PER_CHUNK = 130
PAUSE_SHORT_SEC = 0.20
PAUSE_LONG_SEC = 0.45
SAMPLE_RATE = 24000
CROSSFADE_MS = 25  # crossfade na hraniciach chunkov (eliminuje clicky)


# ─────────────────────────────────────────────────────────────────────────────
# QC analyzátor (rovnaký ako v Chatterbox skripte)
# ─────────────────────────────────────────────────────────────────────────────
def qc_analyze(x: np.ndarray, sr: int, expected_chars: int) -> dict:
    n = x.shape[0]
    dur = n / sr
    issues = []

    # Adaptívny min duration:
    # - Pre krátke vstupy (<80 chars) tolerovať kompaktnejší output (28 chars/s)
    # - Pre dlhšie texty štandard (22 chars/s)
    cps_min = 28.0 if expected_chars < 80 else 22.0
    expected_dur_min = max(0.4, expected_chars / cps_min)
    expected_dur_max = expected_chars / 7.0
    # Tolerancia 0.3s pre short text — OmniVoice na krátkych textoch má prirodzene
    # menšiu varianciu output dur, takže striktné limity vyvolávajú falošné alarmy.
    if dur < expected_dur_min - 0.3:
        issues.append(f"too_short({dur:.1f}s<{expected_dur_min:.1f}s)")
    if dur > expected_dur_max + 1.5:
        issues.append(f"too_long({dur:.1f}s>{expected_dur_max:.1f}s)")

    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 0.005:
        issues.append(f"silent_chunk(rms={rms:.4f})")
    if rms > 0.6:
        issues.append(f"too_loud(rms={rms:.4f})")

    clip_ratio = float(np.mean(np.abs(x) > 0.98))
    if clip_ratio > 0.005:
        issues.append(f"clipping({clip_ratio*100:.1f}%)")

    win = max(1, int(sr * 0.05))
    nwin = n // win
    if nwin >= 4:
        env = np.array([np.sqrt(np.mean(x[i*win:(i+1)*win] ** 2)) for i in range(nwin)])
        peak = float(env.max() + 1e-9)
        thresh = peak * 0.05
        silent = env < thresh
        inner = silent[2:-2] if nwin > 4 else silent
        run = 0; max_run = 0
        for s in inner:
            if s:
                run += 1; max_run = max(max_run, run)
            else:
                run = 0
        gap_sec = max_run * (win / sr)
        if gap_sec > 0.6:
            issues.append(f"long_internal_silence({gap_sec:.2f}s)")

    if n > sr * 3:
        tail = x[-int(sr * 1.5):]
        prev = x[-int(sr * 3):-int(sr * 1.5)]
        min_len = min(len(tail), len(prev))
        if min_len > 0:
            corr = float(np.corrcoef(tail[:min_len], prev[:min_len])[0, 1])
            if corr > 0.9:
                issues.append(f"loop_repeat(r={corr:.2f})")

    score = (
        len(issues) * 10
        + max(0, expected_dur_min - dur) * 5
        + max(0, dur - expected_dur_max - 1.5) * 2
    )
    return {"ok": len(issues) == 0, "issues": issues, "score": score,
            "metrics": {"dur": dur, "rms": rms, "clip": clip_ratio}}


# ─────────────────────────────────────────────────────────────────────────────
# Markdown → speech (rovnaké ako predtým)
# ─────────────────────────────────────────────────────────────────────────────
def markdown_to_speech(md: str) -> list[tuple[str, str]]:
    lines = md.splitlines()
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    in_table = False
    in_code = False

    def flush():
        if not buf:
            return
        text = " ".join(buf).strip()
        if text:
            blocks.append(("para", text))
        buf.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush(); in_code = not in_code; continue
        if in_code:
            continue
        if "|" in line and not line.startswith("    "):
            flush(); in_table = True; continue
        if in_table and line.strip() == "":
            in_table = False
        if in_table:
            continue
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            flush()
            heading = strip_md_inline(m.group(1)).strip(" .:")
            if heading:
                blocks.append(("heading", heading))
            continue
        if line.strip() in ("---", "***", "___"):
            flush(); continue
        if line.strip().startswith("> "):
            flush(); blocks.append(("para", strip_md_inline(line.strip()[2:]))); continue
        m = re.match(r"^\s*[-*+]\s+(.*)", line)
        if m:
            flush(); blocks.append(("para", strip_md_inline(m.group(1)))); continue
        m = re.match(r"^\s*\d+\.\s+(.*)", line)
        if m:
            flush(); blocks.append(("para", strip_md_inline(m.group(1)))); continue
        if line.strip() == "":
            flush(); continue
        buf.append(strip_md_inline(line))
    flush()
    return blocks


def strip_md_inline(s: str) -> str:
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"~~([^~]+)~~", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"^[✦◼❑▪✓]+\s*", "", s)
    return s.strip()


def crossfade_concat(parts: list[np.ndarray], sr: int, fade_ms: int = CROSSFADE_MS) -> np.ndarray:
    """Spojí audio chunky s krátkym crossfade-om (eliminuje clicky na hraniciach)."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    fade_n = int(sr * fade_ms / 1000)
    if fade_n <= 0 or len(parts) == 1:
        return np.concatenate(parts).astype(np.float32)

    fade_in = np.linspace(0, 1, fade_n, dtype=np.float32)
    fade_out = 1 - fade_in
    out = parts[0].astype(np.float32).copy()
    for nxt in parts[1:]:
        nxt = nxt.astype(np.float32)
        # Ak niektorý chunk je príliš krátky (napr. silence < fade), len pripoj
        if len(out) < fade_n or len(nxt) < fade_n:
            out = np.concatenate([out, nxt])
            continue
        # Crossfade: posledných fade_n vzoriek out × fade_out + prvých fade_n nxt × fade_in
        head = out[:-fade_n]
        tail_mix = out[-fade_n:] * fade_out + nxt[:fade_n] * fade_in
        rest = nxt[fade_n:]
        out = np.concatenate([head, tail_mix, rest])
    return out


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            while len(s) > max_chars:
                cut = s.rfind(" ", 0, max_chars)
                if cut == -1:
                    cut = max_chars
                chunks.append(s[:cut].strip())
                s = s[cut:].strip()
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def preprocess_for_tts(text: str) -> str:
    """Fonetické úpravy aby OmniVoice neprečítal anglické skratky zle."""
    repl = [
        (r"\bMFA\b",  "Em-ef-Á"),
        (r"\bIDS\b",  "Aj-Dý-eS"),
        (r"\bIPS\b",  "Aj-Pý-eS"),
        (r"\bEDR\b",  "Í-Dý-eR"),
        (r"\bXDR\b",  "Iks-Dý-eR"),
        (r"\bNDR\b",  "eN-Dý-eR"),
        (r"\bSIEM\b", "siem"),
        (r"\bSOC\b",  "sók"),
        (r"\bSOAR\b", "sóár"),
        (r"\bNGFW\b", "en-Gé-ef-W"),
        (r"\bDoS\b",  "Dí-O-eS"),
        (r"\bDDoS\b", "Dý-Dý-O-eS"),
        (r"\bMITM\b", "Em-aj-tý-em"),
        (r"\bIB\b",   "Í-Bé"),
        (r"\bIS\b",   "informačný systém"),
        (r"\bIKT\b",  "Í-Ká-Té"),
        (r"\bSW\b",   "softvér"),
        (r"\bHW\b",   "hardvér"),
        (r"\bUPS\b",  "Ú-Pé-eS"),
        (r"\bDB\b",   "databáza"),
        (r"\bPC\b",   "Pé-Cé"),
        (r"\bLAN\b",  "lokálna sieť"),
        (r"\bWAN\b",  "rozľahlá sieť"),
        (r"\bSR\b",   "Slovenskej republiky"),
        (r"\bEÚ\b",   "Európskej únie"),
        (r"\bNIS\b",  "en-aj-eS"),
        (r"\bNIS2\b", "en-aj-eS dva"),
        (r"§\s*(\d+)", r"paragraf \1"),
        (r"\b2FA\b", "dvojfaktorová autentifikácia"),
        (r"\bOSINT\b", "ó-eS-aj-eN-tý"),
        (r"\bRCE\b", "eR-Cé-Í"),
        (r"\bLPE\b", "eL-Pé-Í"),
        (r"\bAES\b", "Á-Í-eS"),
        (r"\bDES\b", "Dí-Í-eS"),
        (r"\bRSA\b", "eR-eS-Á"),
        (r"\bECC\b", "Í-Cé-Cé"),
        (r"\bECDSA\b", "Í-Cé-Dí-eS-Á"),
        (r"\bDH\b", "Dí-Há"),
        (r"\bECDH\b", "Í-Cé-Dí-Há"),
        (r"\bSHA\b", "šejá"),
        (r"\bMAC\b", "em-ej-cé"),
        (r"\bHMAC\b", "Há-em-ej-cé"),
        (r"\bMD5\b", "em-Dý päť"),
        (r"\bSHA-1\b", "šejá jedna"),
        (r"\bSHA-2\b", "šejá dva"),
        (r"\bSHA-3\b", "šejá tri"),
        (r"\bSHA-256\b", "šejá-dvesto-päťdesiat-šesť"),
        (r"\bGCM\b", "Gé-Cé-em"),
        (r"\bECB\b", "Í-Cé-Bé"),
        (r"\bCBC\b", "Cé-Bé-Cé"),
        (r"\bCTR\b", "Cé-Té-eR"),
        (r"\bX\.509\b", "iks bodka päťstodeväť"),
        (r"\bCRL\b", "Cé-eR-eL"),
        (r"\bOCSP\b", "Ó-Cé-eS-Pé"),
        (r"\bTLS\b", "Té-eL-eS"),
        (r"\bPFS\b", "Pé-ef-eS"),
        (r"\bPKI\b", "Pé-Ká-aj"),
        (r"\bCA\b",  "Cé-Á"),
        (r"\bHSM\b", "Há-eS-em"),
        (r"\bTPM\b", "Té-Pé-em"),
        (r"\bSQL\b", "eS-Kvé-eL"),
        (r"\bSQLi\b", "eS-Kvé-eL aj"),
        (r"\bXSS\b", "iks-eS-eS"),
        (r"\bCSRF\b", "Cé-eS-eR-ef"),
        (r"\bCSP\b", "Cé-eS-Pé"),
        (r"\bDOM\b", "dóm"),
        (r"\bAPI\b", "Á-Pé-aj"),
        (r"\bURL\b", "Ú-eR-eL"),
    ]
    for pat, rep in repl:
        text = re.sub(pat, rep, text)
    text = text.replace("„", "").replace("\"", "").replace("«", "").replace("»", "")
    text = text.replace(" — ", ". ").replace("—", ".")
    text = text.replace(" – ", ", ").replace("–", "-")
    text = re.sub(r"[•●▪■◼◻]", "", text)
    # Pridaj prirodzené pauzy pred enumeráciami a po kľúčových slovách (živší prednes)
    text = re.sub(r"\s*=\s*", ", teda, ", text)
    text = re.sub(r"\s+\b(napríklad|napr\.)\s+", ", napríklad, ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\bteda\b\s+", ", teda, ", text)
    text = re.sub(r"\s+\bale\b\s+", ", ale ", text)
    text = re.sub(r"\s+\bzatiaľ čo\b\s+", ", zatiaľ čo ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# OmniVoice generator s QC retry
# ─────────────────────────────────────────────────────────────────────────────
def generate_chunk_with_qc(model, voice_clone_prompt, text: str, sr: int,
                            log_prefix: str = "",
                            instruct: str | None = None,
                            speed: float | None = None) -> tuple[np.ndarray | None, dict]:
    best_audio = None
    best_score = float("inf")
    best_qc = None

    gen_kwargs = {
        "text": text,
        "language": "sk",
        "voice_clone_prompt": voice_clone_prompt,
    }
    if instruct:
        gen_kwargs["instruct"] = instruct
    if speed:
        gen_kwargs["speed"] = speed

    for attempt, seed in enumerate(QC_RETRY_SEEDS):
        torch.manual_seed(seed)
        try:
            with torch.inference_mode():
                audio_list = model.generate(**gen_kwargs)
        except Exception as e:
            print(f"{log_prefix}  attempt {attempt+1}/{len(QC_RETRY_SEEDS)} seed={seed}: ERROR {e}")
            continue

        x = audio_list[0] if isinstance(audio_list, list) else audio_list
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if x.ndim > 1:
            x = x.squeeze()

        qc = qc_analyze(x, sr, expected_chars=len(text))
        flag = "✓" if qc["ok"] else "✗"
        issues_str = ",".join(qc["issues"]) if qc["issues"] else "none"
        print(f"{log_prefix}  attempt {attempt+1} seed={seed}: {flag} dur={qc['metrics']['dur']:.1f}s "
              f"rms={qc['metrics']['rms']:.3f} score={qc['score']:.0f} issues={issues_str}")

        if qc["score"] < best_score:
            best_score = qc["score"]
            best_audio = x
            best_qc = qc

        if qc["ok"]:
            break

    return best_audio, best_qc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("md", type=str)
    ap.add_argument("out_wav", type=str)
    ap.add_argument("--ref-wav", type=str, default=DEFAULT_REF_WAV)
    ap.add_argument("--ref-text", type=str, default=DEFAULT_REF_TEXT)
    ap.add_argument("--instruct", type=str, default=DEFAULT_INSTRUCT,
                    help="Voice design hint (energický, dynamický, …). Empty string = vypnuté.")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help="Tempo: 1.0 normálne, 1.1 živšie, 0.9 pomalšie")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS_PER_CHUNK)
    ap.add_argument("--pilot", action="store_true",
                    help="Iba prvé ~2 sekcie")
    ap.add_argument("--no-cleanup", action="store_true")
    args = ap.parse_args()

    md = Path(args.md).read_text(encoding="utf-8")
    blocks = markdown_to_speech(md)
    print(f"[md] parsed {len(blocks)} blokov z {args.md}")

    if args.pilot:
        kept = []
        sections_seen = 0
        for kind, text in blocks:
            if kind == "heading":
                sections_seen += 1
            kept.append((kind, text))
            if sections_seen >= 4:
                break
        blocks = kept
        print(f"[pilot] redukované na {len(blocks)} blokov")

    print("[model] loading OmniVoice from k2-fsa/OmniVoice...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
    )
    sr = SAMPLE_RATE
    print(f"[model] device={model.device}, sr={sr}")

    print(f"[ref] preparing voice clone prompt from {args.ref_wav}")
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_wav,
        ref_text=args.ref_text,
        preprocess_prompt=True,
    )
    print("[ref] voice clone prompt ready")

    all_audio: list[np.ndarray] = []
    qc_summary = {"chunks": 0, "remaining_issues": 0}

    for bi, (kind, text) in enumerate(blocks, 1):
        if not text.strip():
            continue
        text_t = preprocess_for_tts(text)
        chunks = chunk_text(text_t, max_chars=args.max_chars)
        print(f"\n[{bi}/{len(blocks)}] {kind:>7} | {len(chunks)} chunk(s) | {text[:60]!r}")

        for ci, chunk in enumerate(chunks):
            qc_summary["chunks"] += 1
            audio, qc = generate_chunk_with_qc(
                model, voice_clone_prompt, chunk, sr,
                log_prefix=f"  [chunk {ci+1}/{len(chunks)}]",
                instruct=args.instruct or None,
                speed=args.speed,
            )
            if audio is None:
                continue
            all_audio.append(audio)

            pause_sec = PAUSE_LONG_SEC if kind == "heading" else PAUSE_SHORT_SEC
            all_audio.append(np.zeros(int(pause_sec * sr), dtype=np.float32))

            if qc and not qc["ok"]:
                qc_summary["remaining_issues"] += 1

    if not all_audio:
        print("[error] No audio generated!")
        return

    out = crossfade_concat(all_audio, sr)
    out_path = Path(args.out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_raw.wav")
    sf.write(str(raw_path), out, sr)
    print(f"\n[raw]  {raw_path}  ({out.shape[0] / sr:.1f}s, crossfade={CROSSFADE_MS}ms)")

    if args.no_cleanup:
        raw_path.replace(out_path)
    else:
        # Mild post-cleanup ako pri Chatterbox skripte
        # silenceremove: trim dlhé interné ticho > 0.5s na 0.5s (drží prirodzený rytmus)
        af_clean = ",".join([
            "silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-30dB",
            "highpass=f=60",
            "afftdn=nr=7:nt=w:om=o",
            "equalizer=f=90:t=q:w=0.7:g=+5.0",
            "equalizer=f=160:t=q:w=1.0:g=+2.5",
            "lowpass=f=11500",
        ])
        print(f"[post] cleanup: {raw_path} -> {out_path}")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(raw_path),
             "-af", af_clean,
             "-ac", "1", "-ar", str(sr),
             "-c:a", "pcm_s16le",
             str(out_path)],
            check=True,
        )
        raw_path.unlink(missing_ok=True)

    print(f"[done] {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    mp3 = out_path.with_suffix(".mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_path),
             "-c:a", "libmp3lame", "-b:a", "96k", str(mp3)],
            check=True,
        )
        print(f"[done] {mp3}  ({mp3.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"[warn] mp3 skip: {e}")

    print("\n" + "═" * 50)
    print("QC SÚHRN")
    print("═" * 50)
    print(f"  Spolu chunkov: {qc_summary['chunks']}")
    print(f"  Zostávajúce problémy: {qc_summary['remaining_issues']} "
          f"({qc_summary['remaining_issues']*100//max(1,qc_summary['chunks'])} %)")


if __name__ == "__main__":
    main()
