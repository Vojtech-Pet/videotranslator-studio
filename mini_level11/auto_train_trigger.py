#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_train_trigger.py — Automatický tréning Chatterbox SK modelu.

Flow:
  1. Skenuje Level 21 ingest adresáre → počíta nové (nevidené) vzorky
  2. Ak nových vzoriek >= --threshold (default 200) → spustí celý pipeline:
       Level 22  →  launch_chatterbox_sql_pilot.sh  →  model swap
  3. Model swap:
       - nájde najlepší checkpoint (trainer_state.json best_model_checkpoint)
       - skopíruje t3_finetuned.safetensors → Ai/models/chatterbox/t3_sk_vX.X.safetensors
       - updatuje def_chatterbox_model v musetalk_gui_config.json
  4. Označí vzorky ako spracované → ďalší run ich nepočíta znovu

Použitie:
  # Jedno spustenie (napr. z cronu)
  python auto_train_trigger.py --ingest-dirs level21_ingest

  # Dry-run — len zobraz počet nových vzoriek, nič nespúšťaj
  python auto_train_trigger.py --ingest-dirs level21_ingest --dry-run

  # Nižší threshold pre testovanie
  python auto_train_trigger.py --ingest-dirs level21_ingest --threshold 50
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cesty
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
VT_ROOT = ROOT.parent
AI_ROOT = Path.home() / "Ai"

DEFAULT_STATE_FILE = ROOT / "auto_train_state.json"
DEFAULT_OUT_DIR = ROOT / "auto_train_pack"
DEFAULT_GUI_CONFIG = VT_ROOT / "scripts" / "musetalk_gui_config.json"
DEFAULT_MODELS_DIR = AI_ROOT / "models" / "chatterbox"
DEFAULT_TRAIN_ROOT = AI_ROOT / "chatterbox-finetuning"
DEFAULT_PYTHON = Path.home() / "miniforge3" / "envs" / "chatterbox_env" / "bin" / "python3"


def _find_latest_model(models_dir: Path, turbo: bool = False) -> Path:
    """Vráti najnovší checkpoint pre daný typ, alebo fallback."""
    import re
    if turbo:
        candidates = sorted(models_dir.glob("t3_sk_turbo_v*.safetensors"))
        if not candidates:
            return models_dir / "t3_sk_turbo_v1.safetensors"
        def _vkey(p: Path):
            m = re.search(r"turbo_v(\d+)", p.name)
            return int(m.group(1)) if m else 0
        return max(candidates, key=_vkey)
    else:
        candidates = [p for p in sorted(models_dir.glob("t3_sk_v*.safetensors"))
                      if "turbo" not in p.name]
        if not candidates:
            return models_dir / "t3_sk_v2.2.safetensors"
        def _vkey(p: Path):
            m = re.search(r"t3_sk_v(\d+)\.(\d+)", p.name)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        return max(candidates, key=_vkey)


DEFAULT_RESUME_CHECKPOINT = _find_latest_model(DEFAULT_MODELS_DIR)

LEVEL22_SCRIPT = ROOT / "level22_prepare_chatterbox_training.py"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "auto_train_trigger.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("auto_train")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("State file poškodený — začínam od nuly: %s", path)
    return {"seen_ids": [], "runs": []}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _row_id(row: dict[str, Any]) -> str:
    """Jednoznačný identifikátor vzorky — source + časový rozsah."""
    src = str(row.get("source_id") or "")
    start = round(float(row.get("start", 0.0) or 0.0), 3)
    end = round(float(row.get("end", 0.0) or 0.0), 3)
    return f"{src}@{start:.3f}-{end:.3f}"


# ---------------------------------------------------------------------------
# Skenovanie Level 21 ingest adresárov
# ---------------------------------------------------------------------------

def _find_manifests(ingest_dirs: list[str]) -> list[Path]:
    """Nájde všetky candidate_segments.json súbory v zadaných adresároch."""
    found: list[Path] = []
    for d in ingest_dirs:
        base = Path(d).expanduser().resolve()
        if base.is_file() and base.name.endswith(".json"):
            found.append(base)
            continue
        for candidate in (
            base / "candidate_segments.json",
            base / "manifests" / "candidate_segments.json",
        ):
            if candidate.exists():
                found.append(candidate)
                break
        else:
            # Rekurzívne hľadaj
            for p in base.rglob("candidate_segments.json"):
                found.append(p)
    return list(dict.fromkeys(found))  # dedup, zachovaj poradie


def count_new_samples(
    ingest_dirs: list[str],
    seen_ids: set[str],
) -> tuple[int, list[dict[str, Any]]]:
    """Vráti (počet_nových, zoznam_nových_riadkov)."""
    manifests = _find_manifests(ingest_dirs)
    if not manifests:
        log.warning("Žiadne candidate_segments.json nenájdené v: %s", ingest_dirs)
        return 0, []

    new_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        try:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Nemôžem čítať manifest %s: %s", manifest, e)
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            rid = _row_id(row)
            if rid not in seen_ids:
                # Základná kvalita — audio musí existovať
                audio = Path(str(row.get("audio") or "")).expanduser()
                if audio.exists():
                    new_rows.append(row)

    log.info(
        "Manifesty: %d | Celkovo nových vzoriek: %d",
        len(manifests), len(new_rows),
    )
    return len(new_rows), new_rows


# ---------------------------------------------------------------------------
# Level 22 — príprava tréningových dát
# ---------------------------------------------------------------------------

def run_level22(
    ingest_dirs: list[str],
    out_dir: Path,
    python: Path,
    train_root: Path,
    resume_checkpoint: Path,
    prosody_filter: bool = True,
    prosody_rebucket: bool = True,
    is_turbo: bool = False,
) -> Path:
    """Spustí level22_prepare_chatterbox_training.py a vráti cestu k launch scriptu."""
    cmd = [
        str(python), str(LEVEL22_SCRIPT),
        "--manifest-dirs", *ingest_dirs,
        "--out-dir", str(out_dir),
        "--train-root", str(train_root),
        "--resume-checkpoint", str(resume_checkpoint),
        "--reuse-clips",
        "--epochs", "2.0",
        "--learning-rate", "3e-6",
        "--batch-size", "4",
        "--grad-accum", "4",
        "--save-steps", "100",
        "--warmup-steps", "25",
    ]
    if is_turbo:
        cmd.append("--turbo")
    if prosody_filter:
        cmd.append("--prosody-filter")
    if prosody_rebucket:
        cmd.append("--prosody-rebucket")

    log.info("Spúšťam Level 22: %s", " ".join(cmd[:6]) + " ...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Level 22 zlyhalo s kódom {result.returncode}")

    launch_script = out_dir / "launch_chatterbox_sql_pilot.sh"
    if not launch_script.exists():
        raise FileNotFoundError(f"Launch script nebol vygenerovaný: {launch_script}")

    log.info("Level 22 OK — launch script: %s", launch_script)
    return launch_script


# ---------------------------------------------------------------------------
# Tréning
# ---------------------------------------------------------------------------

def run_training(launch_script: Path, timeout_s: int = 14400) -> None:
    """Spustí launch_chatterbox_sql_pilot.sh a čaká na dokončenie."""
    log.info("Spúšťam tréning: %s", launch_script)
    start = time.time()
    result = subprocess.run(
        ["bash", str(launch_script)],
        check=False,
        timeout=timeout_s,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"Tréning zlyhalo s kódom {result.returncode} po {elapsed:.0f}s")
    log.info("Tréning dokončený za %.0fs", elapsed)


# ---------------------------------------------------------------------------
# Výber najlepšieho checkpointu
# ---------------------------------------------------------------------------

def find_best_checkpoint(output_dir: Path) -> Path | None:
    """Nájde najlepší checkpoint podľa trainer_state.json alebo posledný."""
    # 1. trainer_state.json v output_dir
    trainer_state = output_dir / "trainer_state.json"
    if trainer_state.exists():
        try:
            state = json.loads(trainer_state.read_text(encoding="utf-8"))
            best = state.get("best_model_checkpoint")
            if best:
                best_path = Path(best)
                ckpt_weights = best_path / "model.safetensors"
                if not ckpt_weights.exists():
                    ckpt_weights = best_path / "pytorch_model.bin"
                if ckpt_weights.exists():
                    log.info("Najlepší checkpoint (trainer_state): %s", ckpt_weights)
                    return ckpt_weights
        except Exception as e:
            log.warning("Nemôžem čítať trainer_state.json: %s", e)

    # 2. t3_finetuned.safetensors priamo v output_dir
    final = output_dir / "t3_finetuned.safetensors"
    if final.exists():
        log.info("Finálny model: %s", final)
        return final

    # 3. Posledný checkpoint podľa čísla
    ckpt_dirs = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    for ckpt_dir in reversed(ckpt_dirs):
        for name in ("model.safetensors", "pytorch_model.bin"):
            w = ckpt_dir / name
            if w.exists():
                log.info("Posledný checkpoint: %s", w)
                return w

    log.error("Žiadny checkpoint nenájdený v %s", output_dir)
    return None


# ---------------------------------------------------------------------------
# Model swap — nová verzia + update config
# ---------------------------------------------------------------------------

def _next_version(models_dir: Path, turbo: bool = False) -> str:
    """Nájde ďalšie číslo verzie pre daný typ modelu."""
    if turbo:
        prefix = "t3_sk_turbo_v"
        pattern = re.compile(r"t3_sk_turbo_v(\d+)\.safetensors")
        max_v = 1
        for p in models_dir.glob("t3_sk_turbo_v*.safetensors"):
            m = pattern.match(p.name)
            if m:
                v = int(m.group(1))
                if v > max_v:
                    max_v = v
        return f"{prefix}{max_v + 1}"
    else:
        pattern = re.compile(r"t3_sk_v(\d+)\.(\d+)\.safetensors")
        max_major, max_minor = 2, 5  # min verzia ak žiadna neexistuje
        for p in models_dir.glob("t3_sk_v*.safetensors"):
            m = pattern.match(p.name)
            if m:
                major, minor = int(m.group(1)), int(m.group(2))
                if (major, minor) > (max_major, max_minor):
                    max_major, max_minor = major, minor
        return f"t3_sk_v{max_major}.{max_minor + 1}"


def swap_model(
    checkpoint_weights: Path,
    models_dir: Path,
    gui_config_path: Path,
    turbo: bool = False,
) -> str:
    """Skopíruje checkpoint do models_dir, updatuje musetalk_gui_config.json.
    Vráti nový názov súboru (bez cesty).
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(models_dir, turbo=turbo)
    new_name = f"{version}.safetensors"
    dst = models_dir / new_name

    log.info("Kopírujem model: %s → %s", checkpoint_weights, dst)
    shutil.copy2(checkpoint_weights, dst)

    # Aktualizuj GUI config
    if gui_config_path.exists():
        try:
            cfg = json.loads(gui_config_path.read_text(encoding="utf-8"))
            old_model = cfg.get("def_chatterbox_model", "")
            cfg["def_chatterbox_model"] = new_name
            gui_config_path.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.info(
                "GUI config aktualizovaný: %s → %s",
                old_model, new_name,
            )
        except Exception as e:
            log.error("Nemôžem updatovať GUI config: %s", e)
    else:
        log.warning("GUI config neexistuje: %s", gui_config_path)

    return new_name


# ---------------------------------------------------------------------------
# Hlavný flow
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    state = _load_state(args.state_file)
    seen_ids: set[str] = set(state.get("seen_ids", []))

    n_new, new_rows = count_new_samples(args.ingest_dirs, seen_ids)

    if n_new < args.threshold:
        log.info(
            "Nových vzoriek: %d / %d — threshold nedosiahnutý, nič nespúšťam.",
            n_new, args.threshold,
        )
        return

    log.info("Threshold dosiahnutý: %d nových vzoriek ≥ %d → spúšťam pipeline", n_new, args.threshold)

    if args.dry_run:
        log.info("[DRY-RUN] Nič sa nespustí.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"run_{timestamp}"

    try:
        # Krok 1: Level 22
        launch_script = run_level22(
            ingest_dirs=args.ingest_dirs,
            out_dir=out_dir,
            python=Path(args.python),
            train_root=Path(args.train_root),
            resume_checkpoint=Path(args.resume_checkpoint),
            prosody_filter=not args.no_prosody_filter,
            prosody_rebucket=not args.no_prosody_rebucket,
            is_turbo=getattr(args, "turbo", False),
        )

        # Krok 2: Tréning
        run_training(launch_script, timeout_s=args.training_timeout)

        # Krok 3: Najlepší checkpoint
        training_output_dir = out_dir / "output_chatterbox_sql_pilot"
        checkpoint_weights = find_best_checkpoint(training_output_dir)
        if checkpoint_weights is None:
            raise RuntimeError("Žiadny checkpoint nebol nájdený po tréningu.")

        # Krok 4: Model swap
        new_model_name = swap_model(
            checkpoint_weights=checkpoint_weights,
            models_dir=Path(args.models_dir),
            gui_config_path=Path(args.gui_config),
            turbo=getattr(args, "turbo", False),
        )

        # Krok 5: Ulož stav — označ vzorky ako spracované
        new_ids = [_row_id(r) for r in new_rows]
        state["seen_ids"] = list(seen_ids | set(new_ids))
        state.setdefault("runs", []).append({
            "timestamp": timestamp,
            "new_samples": n_new,
            "out_dir": str(out_dir),
            "new_model": new_model_name,
            "checkpoint": str(checkpoint_weights),
        })
        _save_state(args.state_file, state)

        log.info("=" * 60)
        log.info("HOTOVO: nový model → %s", new_model_name)
        log.info("Celkovo spracovaných vzoriek: %d", len(state["seen_ids"]))
        log.info("=" * 60)

    except Exception as e:
        log.error("Pipeline zlyhala: %s", e, exc_info=True)
        # Ulož stav runu aj pri zlyhaní (bez označenia vzoriek)
        state.setdefault("runs", []).append({
            "timestamp": timestamp,
            "new_samples": n_new,
            "out_dir": str(out_dir),
            "error": str(e),
        })
        _save_state(args.state_file, state)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Automatický tréning Chatterbox SK — spustí sa keď je dosť nových vzoriek.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--ingest-dirs", nargs="+", required=True,
        help="Level 21 ingest adresáre alebo priame cesty k candidate_segments.json",
    )
    ap.add_argument(
        "--threshold", type=int, default=200,
        help="Minimálny počet nových vzoriek pre spustenie tréning pipeline",
    )
    ap.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help="Výstupný adresár pre Level 22 pack + launch script",
    )
    ap.add_argument(
        "--state-file", type=Path, default=DEFAULT_STATE_FILE,
        help="JSON súbor so stavom (spracované vzorky, história runov)",
    )
    ap.add_argument(
        "--python", default=str(DEFAULT_PYTHON),
        help="Python interpreter pre Level 22 a tréning",
    )
    ap.add_argument(
        "--train-root", default=str(DEFAULT_TRAIN_ROOT),
        help="Cesta k chatterbox-finetuning repozitáru",
    )
    ap.add_argument(
        "--resume-checkpoint", default=str(DEFAULT_RESUME_CHECKPOINT),
        help="Checkpoint z ktorého sa pokračuje (safetensors)",
    )
    ap.add_argument(
        "--models-dir", default=str(DEFAULT_MODELS_DIR),
        help="Adresár kde sa uloží nový model po tréningu",
    )
    ap.add_argument(
        "--gui-config", default=str(DEFAULT_GUI_CONFIG),
        help="Cesta k musetalk_gui_config.json (aktívny model sa tu updatuje)",
    )
    ap.add_argument(
        "--training-timeout", type=int, default=14400,
        help="Max čas čakania na tréning v sekundách (default 4h)",
    )
    ap.add_argument(
        "--no-prosody-filter", action="store_true",
        help="Vypni prosody filter (nevyhadzuj monotónne/tiché vzorky)",
    )
    ap.add_argument(
        "--no-prosody-rebucket", action="store_true",
        help="Vypni automatické rebucketovanie podľa prosody tieru",
    )
    ap.add_argument(
        "--turbo", action="store_true",
        help="Tréning Chatterbox Turbo (CHATTERBOX_IS_TURBO=1)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Len zobraz počet nových vzoriek, nič nespúšťaj",
    )
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
