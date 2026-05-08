#!/usr/bin/env python3
"""
SRT → segments JSON konverzia pre VTS pipeline.

Použitie:
  ./srt_to_segments.py movie.srt
    # output: movie_sk_segments.json (default lang sk)

  ./srt_to_segments.py movie.srt --lang cs --out custom.json

  ./srt_to_segments.py movie.srt --merge-short 0.3
    # spojí titulky vzdialené menej ako 0.3s (merge consecutive cues)

Podporuje:
  - UTF-8, Windows-1250, ISO-8859-2 encoding (auto-detect cez chardet alebo fallback)
  - Multi-line SRT cues (zluč na jeden segment)
  - HTML tags (<i>, <b>, <font color>) — odstráni
  - Position tags ({\an8}, atď.) — odstráni
"""
import argparse
import json
import re
import sys
from pathlib import Path


SRT_TIMING_RE = re.compile(
    r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*'
    r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})'
)
HTML_TAG_RE = re.compile(r'<[^>]+>')
ASS_POS_RE = re.compile(r'\{\\[^}]+\}')


def srt_time_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def read_srt(path: Path) -> str:
    """Read SRT, try UTF-8 first then Windows-1250 fallback."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "windows-1250", "iso-8859-2", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_srt(content: str) -> list:
    """Parse SRT content into list of {index, start, end, text}."""
    cues = []
    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            continue
        # First line: number; second: timing; rest: text
        # But some SRTs skip the number — handle that
        if lines[0].isdigit():
            timing_idx = 1
        else:
            timing_idx = 0
        if timing_idx >= len(lines):
            continue
        m = SRT_TIMING_RE.search(lines[timing_idx])
        if not m:
            continue
        start = srt_time_to_seconds(*m.group(1, 2, 3, 4))
        end = srt_time_to_seconds(*m.group(5, 6, 7, 8))
        text_lines = lines[timing_idx + 1:]
        text = " ".join(text_lines).strip()
        # Strip HTML tags + ASS position tags
        text = HTML_TAG_RE.sub("", text)
        text = ASS_POS_RE.sub("", text)
        # Common SRT noise: speaker tags like "JOHN: text" — keep as is, TTS handles
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            cues.append({
                "start": start,
                "end": end,
                "text": text,
            })
    return cues


def merge_short_gaps(cues: list, gap_threshold: float = 0.0) -> list:
    """Merge consecutive cues if gap between them is < gap_threshold."""
    if gap_threshold <= 0 or len(cues) < 2:
        return cues
    merged = [dict(cues[0])]
    for c in cues[1:]:
        last = merged[-1]
        gap = c["start"] - last["end"]
        if gap < gap_threshold:
            # Merge: extend end + concatenate text
            last["end"] = c["end"]
            last["text"] = (last["text"].rstrip(".,;:! ") + " " + c["text"]).strip()
        else:
            merged.append(dict(c))
    return merged


def main():
    ap = argparse.ArgumentParser(description="SRT → VTS segments JSON")
    ap.add_argument("srt", help="Input SRT file")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: <stem>_<lang>_segments.json next to SRT)")
    ap.add_argument("--lang", default="sk",
                    help="Language code (sk, cs, en, ...). Used for filename. Default: sk")
    ap.add_argument("--merge-short", type=float, default=0.0,
                    help="Merge consecutive cues if gap < threshold seconds (default 0 = no merge)")
    ap.add_argument("--video-stem", default=None,
                    help="Video stem name (without extension) — pre default output naming")
    args = ap.parse_args()

    srt_path = Path(args.srt).expanduser().resolve()
    if not srt_path.exists():
        sys.exit(f"SRT neexistuje: {srt_path}")

    print(f"==> SRT → segments JSON")
    print(f"   input: {srt_path.name}")

    content = read_srt(srt_path)
    cues = parse_srt(content)
    print(f"   parsed: {len(cues)} cues")

    if args.merge_short > 0:
        before = len(cues)
        cues = merge_short_gaps(cues, args.merge_short)
        print(f"   merged: {before} → {len(cues)} (threshold {args.merge_short}s)")

    if not cues:
        sys.exit("Žiadne valid cues v SRT")

    total_dur = cues[-1]["end"] - cues[0]["start"]
    avg_seg = sum(c["end"] - c["start"] for c in cues) / len(cues)
    print(f"   total span: {total_dur:.1f}s, avg seg dur: {avg_seg:.2f}s")
    print(f"   first: [{cues[0]['start']:.2f}-{cues[0]['end']:.2f}] {cues[0]['text'][:60]}")
    print(f"   last:  [{cues[-1]['start']:.2f}-{cues[-1]['end']:.2f}] {cues[-1]['text'][:60]}")

    # Default output: pipeline-compatible location
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        stem = args.video_stem or srt_path.stem
        out_path = srt_path.parent / f"{stem}_{args.lang}_segments.json"

    out_data = {
        "segments": cues,
        "source": "srt",
        "source_file": str(srt_path),
        "language": args.lang,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Output: {out_path}")
    print(f"  Použij: cez GUI ako 'Existujúce titulky' alebo CLI:")
    print(f"  python main.py --input <video> --tgt_lang {args.lang} --use_existing_segments")


if __name__ == "__main__":
    main()
