"""
memory_monitor.py — RAM/VRAM monitoring skript pre VideoTranslator pipeline.

Spustenie vedľa main.py:
    python scripts/memory_monitor.py --log work/logs/memory_YYYYMMDD.log --interval 30

Alebo bez logu (len stdout):
    python scripts/memory_monitor.py
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("[memory_monitor] psutil nie je nainštalovaný. Spusti: pip install psutil", flush=True)
    sys.exit(1)

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ── Prahy pre varovania ──────────────────────────────────────────────────────
RAM_WARN_PCT   = 80.0   # % celkovej RAM
RAM_CRIT_PCT   = 90.0   # % celkovej RAM — kritické
SWAP_WARN_PCT  = 50.0   # % celkového swap
SWAP_CRIT_PCT  = 80.0   # % celkového swap
VRAM_WARN_PCT  = 85.0   # % VRAM
VRAM_CRIT_PCT  = 95.0   # % VRAM


def _fmt(val_gb: float, total_gb: float, pct: float) -> str:
    return f"{val_gb:.1f}/{total_gb:.1f}GB ({pct:.1f}%)"


def snapshot() -> dict:
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    ram_used_gb  = mem.used  / 1024**3
    ram_total_gb = mem.total / 1024**3
    ram_pct      = mem.percent

    swap_used_gb  = swap.used  / 1024**3
    swap_total_gb = swap.total / 1024**3
    swap_pct      = swap.percent

    vram_used_gb  = 0.0
    vram_total_gb = 0.0
    vram_pct      = 0.0
    if _TORCH and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = props.total_memory / 1024**3
        vram_used_gb  = torch.cuda.memory_reserved(0) / 1024**3
        vram_pct      = (vram_used_gb / vram_total_gb * 100) if vram_total_gb > 0 else 0.0

    return {
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_pct": ram_pct,
        "swap_used_gb": swap_used_gb,
        "swap_total_gb": swap_total_gb,
        "swap_pct": swap_pct,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "vram_pct": vram_pct,
    }


def level(pct: float, warn: float, crit: float) -> str:
    if pct >= crit:
        return "CRIT"
    if pct >= warn:
        return "WARN"
    return "OK"


def format_line(s: dict) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    ram_lvl  = level(s["ram_pct"],  RAM_WARN_PCT,  RAM_CRIT_PCT)
    swap_lvl = level(s["swap_pct"], SWAP_WARN_PCT, SWAP_CRIT_PCT)
    vram_lvl = level(s["vram_pct"], VRAM_WARN_PCT, VRAM_CRIT_PCT) if s["vram_total_gb"] > 0 else "N/A"

    ram_str  = _fmt(s["ram_used_gb"],  s["ram_total_gb"],  s["ram_pct"])
    swap_str = _fmt(s["swap_used_gb"], s["swap_total_gb"], s["swap_pct"])

    line = f"[{ts}] RAM {ram_lvl}: {ram_str} | SWAP {swap_lvl}: {swap_str}"
    if s["vram_total_gb"] > 0:
        vram_str = _fmt(s["vram_used_gb"], s["vram_total_gb"], s["vram_pct"])
        line += f" | VRAM {vram_lvl}: {vram_str}"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description="RAM/VRAM monitor pre VideoTranslator")
    parser.add_argument("--log",      type=str, default="", help="Cesta k log súboru")
    parser.add_argument("--interval", type=int, default=30, help="Interval v sekundách (default: 30)")
    args = parser.parse_args()

    log_file = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")

    def _write(msg: str) -> None:
        print(msg, flush=True)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    def _shutdown(sig, frame):  # noqa: ANN001
        _write(f"[memory_monitor] Ukončený ({datetime.now().strftime('%H:%M:%S')})")
        if log_file:
            log_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _write(f"[memory_monitor] Štart — interval {args.interval}s | prahy RAM {RAM_WARN_PCT}/{RAM_CRIT_PCT}% SWAP {SWAP_WARN_PCT}/{SWAP_CRIT_PCT}% VRAM {VRAM_WARN_PCT}/{VRAM_CRIT_PCT}%")

    while True:
        s = snapshot()
        line = format_line(s)
        _write(line)

        # Extra varovanie pri kritickom stave
        if s["ram_pct"] >= RAM_CRIT_PCT:
            _write(f"  !! KRITICKÁ RAM: {s['ram_pct']:.1f}% — pipeline môže zlyhať alebo swapovať!")
        if s["swap_pct"] >= SWAP_CRIT_PCT:
            _write(f"  !! KRITICKÝ SWAP: {s['swap_pct']:.1f}% — systém je veľmi pomalý!")
        if s["vram_total_gb"] > 0 and s["vram_pct"] >= VRAM_CRIT_PCT:
            _write(f"  !! KRITICKÁ VRAM: {s['vram_pct']:.1f}% — modely môžu spillovať do RAM!")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
