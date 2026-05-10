"""
cps_calibrator.py
=================
Adaptívna kalibrácia chars-per-second pre TTS timing odhad.

Problém ktorý rieši:
  SK_CHARS_PER_SEC = 14.5 je fixná konštanta. Reálna rýchlosť Chatterbox SK
  závisí od konkrétneho modelu, hlasovej vzorky a obsahu (SQL text vs. bežný
  text). Výsledok: odhad timing adaptation je nepresný → WAV preteká alebo
  je príliš krátky.

Riešenie:
  Po každom rune pipeline uloží skutočné chars/sec z L4 timing auditov
  (kde je presne nameraná dĺžka každého WAV). Exponenciálny kĺzavý priemer
  (EMA) z týchto meraní sa použije v ďalšom rune ako kalibrácia.

Použitie v pipeline.py:
  # Pred adapt_segments_batch:
  from cps_calibrator import CPSCalibrator
  _cps_cal = CPSCalibrator()
  _cps_cal.load(out_dir / "cps_calibration.json")
  translated, _stats = adapt_segments_batch(
      translated, llm=_adapt_llm, chars_per_sec=_cps_cal.effective_cps, ...
  )

  # Po TTS slučke (keď máme _timing_audit_log):
  n = _cps_cal.update_from_audit_log(_timing_audit_log)
  if n > 0:
      _cps_cal.save(out_dir / "cps_calibration.json")
"""

from __future__ import annotations

import json
from pathlib import Path


class CPSCalibrator:
    """
    Exponenciálny kĺzavý priemer skutočného chars-per-second z L4 timing auditov.

    Parametre:
      base_cps    — fallback ak nemáme dosť meraní (default 14.5)
      alpha       — EMA faktor: 0.20 = pomalá adaptácia (história dostáva 80%)
      min_samples — koľko meraní treba pred použitím kalibrovanej hodnoty
      min_cps     — spodná hranica outlier filtra
      max_cps     — horná hranica outlier filtra
      min_chars   — ignoruj segmenty kratšie ako N znakov (nízka info. hodnota)
    """

    DEFAULT_BASE_CPS: float = 14.5
    MIN_CPS: float = 8.0
    MAX_CPS: float = 20.0
    MIN_CHARS: int = 15
    MIN_SAMPLES: int = 3
    ALPHA: float = 0.20

    def __init__(
        self,
        base_cps: float = DEFAULT_BASE_CPS,
        alpha: float = ALPHA,
        min_samples: int = MIN_SAMPLES,
    ) -> None:
        self.base_cps = base_cps
        self.alpha = alpha
        self.min_samples = min_samples
        self._ema: float = 0.0
        self._n: int = 0

    # ── Aktualizácia ──────────────────────────────────────────────────────────

    def update(self, actual_cps: float, text_chars: int = 0) -> None:
        """Pridaj jedno meranie z L4 auditu."""
        if text_chars > 0 and text_chars < self.MIN_CHARS:
            return
        cps = float(actual_cps)
        if not (self.MIN_CPS <= cps <= self.MAX_CPS):
            return
        if self._n == 0:
            self._ema = cps
        else:
            self._ema = self.alpha * cps + (1.0 - self.alpha) * self._ema
        self._n += 1

    def update_from_audit_log(self, timing_audit_log: list[dict]) -> int:
        """
        Hromadná aktualizácia z _timing_audit_log zo pipeline.py.

        Každá položka musí mať 'actual_cps' a voliteľne 'text_chars'.
        Vráti počet použitých meraní.
        """
        used = 0
        for entry in timing_audit_log:
            cps = float(entry.get("actual_cps") or 0)
            chars = int(entry.get("text_chars") or 0)
            if cps > 0:
                self.update(cps, chars)
                used += 1
        return used

    # ── Výstup ────────────────────────────────────────────────────────────────

    @property
    def effective_cps(self) -> float:
        """Kalibrovaný CPS. Ak nemáme dosť meraní, vráti base_cps (fallback)."""
        if self._n < self.min_samples:
            return self.base_cps
        return round(self._ema, 2)

    @property
    def confidence(self) -> str:
        """Sebahodnotenie presnosti kalibrácie."""
        if self._n < self.min_samples:
            return "low"
        if self._n < 15:
            return "medium"
        return "high"

    # ── Perzistencia ──────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Uloží stav kalibrátora do JSON pre ďalší run."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"ema": round(self._ema, 4), "n": self._n, "alpha": self.alpha},
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, path: Path | str) -> bool:
        """
        Načíta stav z predchádzajúceho runu.
        Vráti True ak úspešné, False ak súbor neexistuje alebo je poškodený.
        """
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._ema = float(data.get("ema", 0))
            self._n = int(data.get("n", 0))
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"CPSCalibrator(effective={self.effective_cps:.2f}, "
            f"n={self._n}, confidence={self.confidence})"
        )
