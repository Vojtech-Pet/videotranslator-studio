"""
notes_to_audio.py — markdown → audio (Chatterbox SK) s iteratívnym QC.

Per-chunk loop:
    1. generuj kandidát (CONFIG_ANTICUT params)
    2. analyzuj (dĺžka, RMS, dropout, silence ratio, klipping)
    3. ak detekovaný defekt → retry s iným seedom (max 5 pokusov)
    4. zober najlepší kandidát (najvyššie skóre QC)

Použitie:
    python notes_to_audio.py vstup.md vystup.wav [--ref-wav PATH] [--pilot]
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file as load_safetensors

CHATTERBOX_SRC = "/home/vojtech/Ai/chatterbox_git/src"
sys.path.insert(0, CHATTERBOX_SRC)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

WEIGHTS = "/mnt/tts_data/VideoTranslator_studio/models/chatterbox/t3_sk_v2.2.safetensors"
DEFAULT_REF = "/mnt/tts_data/VideoTranslator_studio/voices/juraj_sk.wav"

QC_RETRY_SEEDS = [42, 0, 123, 7, 99, 31415, 271828]
MAX_CHARS_PER_CHUNK = 130
PAUSE_SHORT_SEC = 0.30
PAUSE_LONG_SEC = 0.80

# CONFIG_ANTICUT z VideoTranslator_studio/scripts/tts.py
GEN_PARAMS = {
    "exaggeration": 0.50,
    "cfg_weight":   0.50,
    "temperature":  0.60,
    "top_p":        0.92,
    "repetition_penalty": 1.25,
}


# ─────────────────────────────────────────────────────────────────────────────
# QC analyzátor — vráti dict s metrikami a issues
# ─────────────────────────────────────────────────────────────────────────────
def qc_analyze(wav: torch.Tensor, sr: int, expected_chars: int) -> dict:
    """Analyzuj wav. Vráti {ok: bool, issues: [...], score: float, metrics: {...}}."""
    if wav.dim() == 2:
        x = wav.squeeze(0).cpu().numpy()
    else:
        x = wav.cpu().numpy()
    n = x.shape[0]
    dur = n / sr
    issues = []

    # 1) Cieľová dĺžka — slovenčina ~14-18 char/sec
    expected_dur_min = max(0.4, expected_chars / 22.0)   # rýchle čítanie
    expected_dur_max = expected_chars / 7.0              # pomalé
    if dur < expected_dur_min:
        issues.append(f"too_short({dur:.1f}s<{expected_dur_min:.1f}s)")
    if dur > expected_dur_max + 1.5:
        issues.append(f"too_long({dur:.1f}s>{expected_dur_max:.1f}s)")

    # 2) RMS úroveň — chunk nesmie byť úplne tichý alebo extrémne hlasný
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 0.005:
        issues.append(f"silent_chunk(rms={rms:.4f})")
    if rms > 0.6:
        issues.append(f"too_loud(rms={rms:.4f})")

    # 3) Klipping (vzorky >= ±0.99)
    clip_ratio = float(np.mean(np.abs(x) > 0.98))
    if clip_ratio > 0.005:
        issues.append(f"clipping({clip_ratio*100:.1f}%)")

    # 4) Dropout / dlhá tichá medzera vo vnútri chunku (signál rozpadu)
    win = max(1, int(sr * 0.05))                 # 50 ms okno
    nwin = n // win
    if nwin >= 4:
        env = np.array([np.sqrt(np.mean(x[i*win:(i+1)*win] ** 2)) for i in range(nwin)])
        peak = float(env.max() + 1e-9)
        thresh = peak * 0.05                     # 5 % špičky = "ticho"
        silent = env < thresh
        # ignoruj prvé/posledné 2 okná (prirodzený nábeh/dobeh)
        inner = silent[2:-2] if nwin > 4 else silent
        # najdlhšia séria tíh vo vnútri
        run = 0; max_run = 0
        for s in inner:
            if s:
                run += 1; max_run = max(max_run, run)
            else:
                run = 0
        gap_sec = max_run * (win / sr)
        if gap_sec > 0.6:
            issues.append(f"long_internal_silence({gap_sec:.2f}s)")

    # 5) Náhly skok energie (artefakt, pukanec)
    if nwin >= 4:
        diff = np.abs(np.diff(env))
        if diff.size > 0 and diff.max() > peak * 0.9:
            issues.append("energy_spike")

    # 6) Repetition / loop — autocorrelation v poslednej tretine
    if n > sr * 2:
        tail = x[-int(sr * 1.5):]
        # ak je posledných 1.5s skoro identických ako predchádzajúcich 1.5s → loop
        if n > sr * 3:
            prev = x[-int(sr * 3):-int(sr * 1.5)]
            min_len = min(len(tail), len(prev))
            if min_len > 0:
                corr = float(np.corrcoef(tail[:min_len], prev[:min_len])[0, 1])
                if corr > 0.9:
                    issues.append(f"loop_repeat(r={corr:.2f})")

    # Skóre — nižšie = lepšie
    score = (
        len(issues) * 10
        + max(0, expected_dur_min - dur) * 5
        + max(0, dur - expected_dur_max - 1.5) * 2
    )

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "score": score,
        "metrics": {"dur": dur, "rms": rms, "clip": clip_ratio},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown → speech (rovnaké ako predtým + nadpisy)
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
        (r"\bNAC\b",  "en-ej-cé"),
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
        (r"\bWAV\b",  "vejv"),
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
        (r"\bSHA-512\b", "šejá-päťsto-dvanásť"),
        (r"\bGCM\b", "Gé-Cé-em"),
        (r"\bECB\b", "Í-Cé-Bé"),
        (r"\bCBC\b", "Cé-Bé-Cé"),
        (r"\bCTR\b", "Cé-Té-eR"),
        (r"\bOFB\b", "Ó-ef-Bé"),
        (r"\bCFB\b", "Cé-ef-Bé"),
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
        (r"\bXSS\b", "iks-eS-eS"),
        (r"\bCSRF\b", "Cé-eS-eR-ef"),
        (r"\bCSP\b", "Cé-eS-Pé"),
        (r"\bDOM\b", "dóm"),
        (r"\bAPI\b", "Á-Pé-aj"),
        (r"\bURL\b", "Ú-eR-eL"),
        (r"\bSAST\b", "saast"),
        (r"\bDAST\b", "daast"),
    ]
    for pat, rep in repl:
        text = re.sub(pat, rep, text)
    text = text.replace("„", "").replace("\"", "").replace("«", "").replace("»", "")
    text = text.replace(" — ", ". ").replace("—", ".")
    text = text.replace(" – ", ", ").replace("–", "-")
    text = re.sub(r"[•●▪■◼◻]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def patch(model: ChatterboxMultilingualTTS, weights_path: str, device: str) -> None:
    state = load_safetensors(weights_path, device="cpu")
    target_vocab = model.t3.text_emb.weight.shape[0]
    src_vocab = state["text_emb.weight"].shape[0]
    if src_vocab > target_vocab:
        state["text_emb.weight"] = state["text_emb.weight"][:target_vocab, :]
        state["text_head.weight"] = state["text_head.weight"][:target_vocab, :]
    elif src_vocab < target_vocab:
        pad = target_vocab - src_vocab
        emb_pad = state["text_emb.weight"].mean(dim=0, keepdim=True).repeat(pad, 1)
        head_pad = state["text_head.weight"].mean(dim=0, keepdim=True).repeat(pad, 1)
        state["text_emb.weight"] = torch.cat([state["text_emb.weight"], emb_pad], dim=0)
        state["text_head.weight"] = torch.cat([state["text_head.weight"], head_pad], dim=0)
    model.t3.load_state_dict(state, strict=True)
    model.t3.to(device).eval()


def cleanup_reference(src: str, dst: str) -> None:
    af = ",".join([
        "highpass=f=70",
        "afftdn=nr=12:nt=w:om=o",
        "lowpass=f=11000",
        "equalizer=f=6800:t=q:w=1.2:g=-1.5",
        "silenceremove=start_periods=1:start_silence=0.04:start_threshold=-50dB",
        "areverse",
        "silenceremove=start_periods=1:start_silence=0.06:start_threshold=-46dB",
        "areverse",
    ])
    print(f"[ref] cleanup: {src} -> {dst}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", af, dst],
        check=True,
    )


def generate_chunk_with_qc(model, text: str, sr: int, log_prefix: str = "") -> tuple[torch.Tensor | None, dict]:
    """Vygeneruj chunk s QC retry — vráti (najlepší wav, qc_dict)."""
    best_wav = None
    best_score = float("inf")
    best_qc = None

    for attempt, seed in enumerate(QC_RETRY_SEEDS):
        torch.manual_seed(seed)
        try:
            with torch.inference_mode():
                wav = model.generate(
                    text=text,
                    language_id="sk",
                    max_new_tokens=4096,
                    **GEN_PARAMS,
                )
        except Exception as e:
            print(f"{log_prefix}  attempt {attempt+1}/{len(QC_RETRY_SEEDS)} seed={seed}: ERROR {e}")
            continue

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        qc = qc_analyze(wav, sr, expected_chars=len(text))
        flag = "✓" if qc["ok"] else "✗"
        issues_str = ",".join(qc["issues"]) if qc["issues"] else "none"
        print(f"{log_prefix}  attempt {attempt+1} seed={seed}: {flag} dur={qc['metrics']['dur']:.1f}s "
              f"rms={qc['metrics']['rms']:.3f} score={qc['score']:.0f} issues={issues_str}")

        if qc["score"] < best_score:
            best_score = qc["score"]
            best_wav = wav.cpu()
            best_qc = qc

        if qc["ok"]:
            break

    return best_wav, best_qc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("md", type=str)
    ap.add_argument("out_wav", type=str)
    ap.add_argument("--ref-wav", type=str, default=DEFAULT_REF)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS_PER_CHUNK)
    ap.add_argument("--pilot", action="store_true",
                    help="Iba prvé ~2 sekcie (rýchly test kvality)")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Vynechaj post-cleanup ffmpeg filter")
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

    with tempfile.TemporaryDirectory() as td:
        cleaned = str(Path(td) / "ref_clean.wav")
        cleanup_reference(args.ref_wav, cleaned)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[model] loading on {device}")
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        patch(model, WEIGHTS, device)
        model.prepare_conditionals(cleaned)
        sr = model.sr
        print(f"[model] sr={sr}")

        # WARMUP — vygeneruj a zahod 1 dummy chunk aby sa model stabilizoval
        print("[warmup] generating throwaway chunk to stabilize model...")
        with torch.inference_mode():
            torch.manual_seed(42)
            _ = model.generate(text="Začíname touto poznámkou. Toto je úvodná veta.",
                               language_id="sk", max_new_tokens=1024, **GEN_PARAMS)
        print("[warmup] done")

        all_audio: list[torch.Tensor] = []
        qc_summary = {"chunks": 0, "ok_first_try": 0, "needed_retry": 0, "remaining_issues": 0}

        for bi, (kind, text) in enumerate(blocks, 1):
            if not text.strip():
                continue
            text_t = preprocess_for_tts(text)
            chunks = chunk_text(text_t, max_chars=args.max_chars)
            print(f"\n[{bi}/{len(blocks)}] {kind:>7} | {len(chunks)} chunk(s) | {text[:60]!r}")

            for ci, chunk in enumerate(chunks):
                qc_summary["chunks"] += 1
                wav, qc = generate_chunk_with_qc(
                    model, chunk, sr,
                    log_prefix=f"  [chunk {ci+1}/{len(chunks)}]"
                )
                if wav is None:
                    continue
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                all_audio.append(wav)

                pause_sec = PAUSE_LONG_SEC if kind == "heading" else PAUSE_SHORT_SEC
                all_audio.append(torch.zeros(1, int(pause_sec * sr)))

                if qc and qc["ok"]:
                    if qc_summary["chunks"] == qc_summary["ok_first_try"] + qc_summary["needed_retry"] + 1:
                        # was first try
                        pass
                else:
                    qc_summary["remaining_issues"] += 1

        out = torch.cat(all_audio, dim=-1)
        out_path = Path(args.out_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = out_path.with_name(out_path.stem + "_raw.wav")
        torchaudio.save(str(raw_path), out, sr)
        print(f"\n[raw]  {raw_path}  ({out.shape[-1] / sr:.1f}s)")

        if args.no_cleanup:
            raw_path.replace(out_path)
        else:
            af_clean = ",".join([
                "highpass=f=60",
                "afftdn=nr=7:nt=w:om=o",
                "equalizer=f=90:t=q:w=0.7:g=+5.0",
                "equalizer=f=160:t=q:w=1.0:g=+2.5",
                "lowpass=f=5500",
                "equalizer=f=3800:t=q:w=1.0:g=-3.5",
                "equalizer=f=5200:t=q:w=1.2:g=-5.5",
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

        # QC report
        print("\n" + "═" * 50)
        print("QC SÚHRN")
        print("═" * 50)
        print(f"  Spolu chunkov:          {qc_summary['chunks']}")
        print(f"  Zostávajúce problémy:   {qc_summary['remaining_issues']} ({qc_summary['remaining_issues']*100//max(1,qc_summary['chunks'])} %)")


if __name__ == "__main__":
    main()
