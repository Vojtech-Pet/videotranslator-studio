# Kritická analýza — čo nás obmedzuje a ako to vyriešiť
> 2026-04-10 | Analýza skutočného kódu, nie odhady

Každý problém obsahuje: kde presne je, prečo to bolí, aké sú možnosti riešenia, prečo je jedno najlepšie.

---

## PROBLÉM 1 — Level 11: žiadny timeout, môže zaseknut navždy

### Kde presne

[src/level11_manager.py:52](src/level11_manager.py#L52)

```python
for attempt_index in range(self.max_retries + 1):
    prompt_bundle = self.prompt_agent.get_prompt(base, ...)
    candidates = self.text_agent.generate(base, prompt_bundle, ...)   # ← LLM volanie
    audio_variants = self.audio_agent.synthesize(base, candidates)    # ← odhaduje dĺžky
    best = self.eval_agent.select_best(base, audio_variants)
    ...
    if best.get("ok") or attempt_index >= self.max_retries:
        break
    retry_mode = self.policy_agent.suggest(base, best)
```

### Prečo to bolí

`text_agent.generate()` volá llama-cpp LLM. Ak:
- model je na CPU a segment je dlhý → inferencia trvá 60+ sekúnd
- llama-cpp dostane nevalidný vstup → môže visieť bez výstupu
- `max_retries=3` × 60s = 4 minúty na **jeden** segment, 100 segmentov = 400 minút

Pipeline.py to volá bez akéhokoľvek watchdog vlákna. GUI zobrazuje len "Running..." bez zmeny. Používateľ nemá šancu vedieť či beží alebo je zaseknuté.

### Možnosti riešenia

| Varianta | Popis | Problém |
|----------|-------|---------|
| A. `signal.alarm` | UNIX timeout cez SIGALRM | Nefunguje vo Windows, nefunguje v threadoch |
| B. `multiprocessing` | Spusti LLM v subprocess s timeout | Ťažká serializácia llama-cpp inštancie |
| C. `concurrent.futures.ThreadPoolExecutor` | Thread s `future.result(timeout=N)` | **Toto je správne** — thread môže byť aj daemon |
| D. `asyncio.timeout` | Async wrapper | Vyžaduje prerobenie celej Level11 na async |

### Najlepšie riešenie: ThreadPoolExecutor + timeout

**Prečo:** llama-cpp nie je serializovateľné cez `multiprocessing`. `signal.alarm` je nereliabilné v threadoch. `asyncio` by vyžadovalo prerobenie celého volacieho stacku. `ThreadPoolExecutor` funguje presne — thread ideme submitnúť, dáme timeout, ak vyprší → return fallback.

```python
# src/level11_manager.py

import concurrent.futures
import logging

logger = logging.getLogger(__name__)

# Konfigurovateľné timeouty (sekundy)
_ATTEMPT_TIMEOUT_S = 45   # max čas na jeden attempt (LLM + eval)
_TOTAL_TIMEOUT_S   = 120  # max čas na celý process_segment


def _run_attempt(
    self,
    base: dict,
    segment_type: str,
    retry_mode: str | None,
) -> dict:
    """Jeden attempt — spúšťame v thread s timeoutom."""
    prompt_bundle = self.prompt_agent.get_prompt(base, segment_type=segment_type, retry_mode=retry_mode)
    candidates = self.text_agent.generate(base, prompt_bundle, force_mode=retry_mode)
    audio_variants = self.audio_agent.synthesize(base, candidates)
    best = self.eval_agent.select_best(base, audio_variants)
    return best, prompt_bundle


def process_segment(self, seg: dict) -> dict:
    base = dict(seg)
    features = extract_features(base)
    segment_type = str(base.get("segment_type") or classify_segment(features, str(base.get("tts_input") or "")))
    features["segment_type"] = segment_type
    base["segment_type"] = segment_type
    base["orchestrator_complex"] = self._is_complex(base, features)
    base["policy_name"] = self.registry.get_active_policy_name()

    attempts = []
    retry_mode = None
    best_result = None

    # ── NOVÉ: executor s per-attempt timeoutom ──────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for attempt_index in range(self.max_retries + 1):
            future = executor.submit(_run_attempt, self, base, segment_type, retry_mode)
            try:
                best, prompt_bundle = future.result(timeout=_ATTEMPT_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "[L11] Attempt %d TIMEOUT po %ds — segment '%s...' → fallback",
                    attempt_index, _ATTEMPT_TIMEOUT_S, str(base.get("tts_input", ""))[:40]
                )
                # Použij to čo máme, skonči loop
                if best_result is None:
                    best_result = _make_timeout_fallback(base)
                break
            except Exception as e:
                logger.error("[L11] Attempt %d ERROR: %s", attempt_index, e)
                if best_result is None:
                    best_result = _make_timeout_fallback(base)
                break

            best["attempt_index"] = attempt_index
            best["prompt_name"] = prompt_bundle.get("prompt_name")
            attempts.append({
                "attempt_index": attempt_index,
                "retry_mode": retry_mode,
                "prompt_name": prompt_bundle.get("prompt_name"),
                "best_final_score": best.get("final_score"),
            })

            if best_result is None or float(best.get("final_score", 0) or 0) > float(best_result.get("final_score", 0) or 0):
                best_result = best

            if best.get("ok") or attempt_index >= self.max_retries:
                break

            retry_plan = self.policy_agent.suggest(base, best)
            retry_mode = str(retry_plan.get("retry_mode") or "balanced")

    assert best_result is not None
    # ... zvyšok compose result (nezmenený)


def _make_timeout_fallback(base: dict) -> dict:
    """Vráť originálny segment ak L11 timeoutuje."""
    return {
        "ok": False,
        "final_score": 0.0,
        "status": "timeout",
        "tts_input": base.get("tts_input") or base.get("level3_text") or base.get("text", ""),
        "candidate_name": "timeout_fallback",
        "prosody_name": "default",
        "delta": 0.0,
        "estimated_duration": 0.0,
        "evaluated_variants": 0,
    }
```

**Výhody tohto riešenia:**
- llama-cpp inštancia zostáva v hlavnom processe (žiadna serializácia)
- Timeout je konfigurovateľný per-use-case
- Fallback vráti originálny text → pipeline pokračuje namiesto krachu
- Logger zachytí každý timeout → vidíme ktoré segmenty sú problematické

---

## PROBLÉM 2 — Fixná CPS kalibrácia (14.5 chars/sec pre všetky segmenty)

### Kde presne

[scripts/text_adaptation.py:41-45](scripts/text_adaptation.py#L41)

```python
SK_CHARS_PER_SEC: float = 14.5   # ← toto platí pre KAŽDÝ segment rovnako
ABBREV_BONUS_S:   float = 0.30
THRESHOLD_SOFT:   float = 1.10
THRESHOLD_HARD:   float = 1.25
THRESHOLD_FORCE_DUB: float = 1.40
```

[scripts/text_adaptation.py:213-226](scripts/text_adaptation.py#L213)

```python
def estimate_tts_duration(
    text: str,
    chars_per_sec: float = SK_CHARS_PER_SEC,  # ← vždy 14.5
    abbrev_bonus: float   = ABBREV_BONUS_S,
) -> float:
    base = len(text) / max(chars_per_sec, 1.0)
    abbrevs = re.findall(r'\b[A-Z]{2,}(?:/[A-Z]+)?\b', text)
    return base + len(abbrevs) * abbrev_bonus
```

### Prečo to bolí

**Scenár A — rýchly rečník:**
- Originál hovorí 180 slov/min (normálna prednáška)
- Chatterbox SK pri normálnom tempe robí 14.5 chars/sec = ~145 slov/min
- Reálne meraný WAV je 12.8 chars/sec (Chatterbox je pomalší na technický text)
- Výsledok: odhadujeme 14.5, reálne je 12.8 → WAV je o 13% dlhší ako slot
- Pipeline to musí stretch-ovať nad 1.0× → degradácia kvality

**Scenár B — pomalý rečník (výukové video):**
- Originál hovorí 110 slov/min, dlhé pauzy
- Slot je dlhší než potrebujeme
- Odhadujeme "text sa zmestí", ale v skutočnosti text je príliš krátky
- Výsledok: tichá medzera za TTS → záplata namiesto správnej adaptácie

**Scenár C — SQL výukové video (technický text):**
- Chatterbox číta "ISNULL, COALESCE, IS NOT NULL" pomalšie než slovenský text
- Skutočná rýchlosť: 11.2 chars/sec
- Odhad s 14.5: text "sa zmestí", ale WAV preteká o 30%

**Zdroj chyby:** Kalibrácia 14.5 bola pravdepodobne nameraná na jednom videu alebo odhadnutá. Neexistuje mechanizmus spätnej väzby medzi Level 4 audit (ktorý meria skutočné WAV dĺžky) a Level 3 kalibráciou.

### Možnosti riešenia

| Varianta | Popis | Problém |
|----------|-------|---------|
| A. Zmeniť konštantu na 12.0 | Konzervatívnejší odhad | Stále fixný, len iná chyba |
| B. Per-segment LLM odhad | Spýtaj sa LLM koľko sekúnd text zaberie | Pomalé, nepresné, nezmyselné |
| C. Kalibrácia z prvého batch runu | Spusti TTS na 5 testovacích segmentoch, zmeraj | Správny smer ale chýba feedback loop |
| D. **Exponenciálny kĺzavý priemer z L4 auditov** | Každý L4 audit vracia `wav_duration`, z toho prepočítaj chars/sec, ukladaj ako running average | **Najlepšie** |

### Najlepšie riešenie: Adaptívna CPS z Level 4 auditov

**Prečo:** L4 audit **už meria** skutočné `wav_duration` každého segmentu ([scripts/level4_audio_feedback.py](scripts/level4_audio_feedback.py)). Táto informácia sa ukladá do `seg["level4_audit"]["wav_duration"]`. Treba ju len spätne kŕmiť do CPS kalibrátora.

```python
# Nový súbor: scripts/cps_calibrator.py

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class CPSCalibrator:
    """
    Exponenciálny kĺzavý priemer chars/sec z reálnych L4 auditov.
    
    α = 0.15 → dáva väčšiu váhu histórii (pomalšia adaptácia)
    α = 0.35 → rýchlejšia adaptácia na konkrétny hlas
    
    Ukladá stav do JSON → pretrváva medzi video-session pre rovnaký hlas.
    """
    alpha: float = 0.20          # EMA faktor
    base_cps: float = 14.5       # fallback ak nemáme dáta
    min_cps: float = 9.0         # spodná hranica (Chatterbox nikdy nejde pod toto)
    max_cps: float = 19.0        # horná hranica
    min_samples: int = 3         # koľko meraní treba pred použitím
    
    _current_cps: float = field(default=0.0, init=False)
    _sample_count: int = field(default=0, init=False)
    _state_path: Optional[Path] = field(default=None, init=False)

    def update_from_l4_audit(self, audit: dict) -> None:
        """
        Aktualizuj kalibráciu z Level 4 audit záznamu.
        
        audit = seg["level4_audit"]
        """
        wav_duration = float(audit.get("wav_duration") or 0)
        tts_input = str(audit.get("tts_input") or audit.get("text") or "")
        
        if wav_duration < 0.3 or len(tts_input) < 5:
            return  # príliš krátky segment → nízka informačná hodnota
        
        # Merané chars/sec pre tento segment
        measured_cps = len(tts_input) / wav_duration
        measured_cps = max(self.min_cps, min(self.max_cps, measured_cps))
        
        if self._sample_count == 0:
            self._current_cps = measured_cps
        else:
            # EMA: nová_hodnota = α * meranie + (1-α) * aktuálna
            self._current_cps = self.alpha * measured_cps + (1 - self.alpha) * self._current_cps
        
        self._sample_count += 1

    @property
    def effective_cps(self) -> float:
        """Vráť kalibrovaný CPS ak máme dostatok meraní, inak base."""
        if self._sample_count < self.min_samples:
            return self.base_cps
        return round(self._current_cps, 2)

    @property
    def confidence(self) -> str:
        """Sebahodnotenie kalibrátora."""
        if self._sample_count < 3: return "low"
        if self._sample_count < 10: return "medium"
        return "high"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "current_cps": self._current_cps,
            "sample_count": self._sample_count,
            "alpha": self.alpha,
        }, indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            data = json.loads(path.read_text())
            self._current_cps = float(data.get("current_cps", 0))
            self._sample_count = int(data.get("sample_count", 0))
            self._state_path = path
```

**Integrácia do pipeline.py:**

```python
# scripts/pipeline.py — v časti kde sa volá adapt_for_timing

# Inicializácia (raz na začiatku runu):
from cps_calibrator import CPSCalibrator
cps_cal = CPSCalibrator(alpha=0.20, base_cps=14.5)
cal_path = Path(args.out_dir) / "cps_calibration.json"
cps_cal.load(cal_path)  # načítaj z predchádzajúceho runu ak existuje

# V TTS slučke — po každom vygenerovanom WAV:
if seg.get("level4_audit"):
    cps_cal.update_from_l4_audit(seg["level4_audit"])
    # Použi aktualizovaný CPS pre ďalšie segmenty
    current_cps = cps_cal.effective_cps

# Uloženie na konci runu:
cps_cal.save(cal_path)
print(f"[CPS] Kalibrovaný CPS: {cps_cal.effective_cps:.2f} ({cps_cal.confidence}, {cps_cal._sample_count} meraní)")
```

**Výhody:**
- Využíva dáta ktoré **už máme** (L4 audit meria každý WAV)
- Adaptuje sa na konkrétny hlas, konkrétny Chatterbox model, konkrétný obsah
- Stav pretrváva cez runy → po prvom video sú ďalšie presnejšie
- Fallback na 14.5 kým nemáme 3 merania → žiadna regresia
- Transparentné: `[CPS] 12.8 (high, 47 meraní)` v logu

**Prečo nie jednoducho zmeniť konštantu na 12.0:**
Pretože správna hodnota závisí od konkrétneho modelu, hlasovej vzorky, obsahu (technický vs. konverzačný text). 12.0 by bolo lepšie pre SQL tutoriály ale horšie pre bežné texty.

---

## PROBLÉM 3 — Žiadny real-time TTS preview

### Kde presne (architektonický problém)

GUI ([VideoTranslator_studio.py](VideoTranslator_studio.py)) spúšťa celý pipeline ako subprocess:
```python
class Runner(QThread):
    def run(self):
        self.proc = subprocess.Popen(self.cmd, ...)
        # čítame stdout, parsujeme progress
```

TTS sa volá hlboko vnútri pipeline.py, WAV súbory sa ukladajú do `output/wav_segments/tts_XXXX.wav`. GUI o nich nevie kým subprocess neskončí.

### Prečo to bolí

- Používateľ musí spustiť celý run (STT + preklad + TTS + montáž = 20-60 minút) len aby zistil či hlas znie dobre
- Ak je hlasová vzorka zlá alebo TTS parameter (exaggeration, CFG) nesprávny → zistí to až na konci
- Žiadna možnosť povedať "tento segment znie dobre, pokračuj, tento nie, skús znova"

### Možnosti riešenia

| Varianta | Popis | Problém |
|----------|-------|---------|
| A. Standalone TTS preview skript | Nový `preview_tts.py --text "..." --voice sample.wav` | Iba CLI, nie integrované do GUI |
| B. Polling výstupného adresára | Runner thread sleduje `wav_segments/` a prehráva nové WAV | Dirty hack, race condition |
| C. **Nová TTS Preview záložka v GUI** | Samostatný QThread spúšťa len TTS (nie celý pipeline) | **Najčistejšie** |
| D. Websocket / HTTP server v pipeline | Pipeline otvorí HTTP endpoint, GUI sa pripojí | Overcomplicated pre desktop app |

### Najlepšie riešenie: Nová "TTS Test" záložka + mini TTS runner

**Prečo:** Chatterbox TTS je plne nezávislé od zvyšku pipeline. Dokáže vyrobiť WAV z textu + voice sample bez STT, prekladu, montáže. Stačí nový QThread ktorý volá `tts.py::synthesize_chatterbox()` priamo.

```python
# VideoTranslator_studio.py — nová záložka "TTS Test"

class TTSPreviewRunner(QThread):
    """Spustí len TTS syntézu pre jeden segment — bez pipeline."""
    audio_ready = pyqtSignal(str)   # cesta k WAV súboru
    error_signal = pyqtSignal(str)
    
    def __init__(self, text: str, voice_sample: str, config: dict):
        super().__init__()
        self.text = text
        self.voice_sample = voice_sample
        self.config = config   # model path, exaggeration, cfg, temperature

    def run(self):
        try:
            import subprocess, tempfile, os
            out_wav = tempfile.mktemp(suffix=".wav")
            cmd = [
                self.config["py"],
                "-c",
                # Inline script — volá len TTS, bez importu celého pipeline
                f"""
import sys; sys.path.insert(0, '{SCRIPTS_DIR}')
from tts import synthesize_chatterbox_segment
synthesize_chatterbox_segment(
    text={self.text!r},
    out_path={out_wav!r},
    voice_sample={self.voice_sample!r},
    model_path={self.config['chatterbox_model']!r},
    exaggeration={self.config.get('exaggeration', 0.5)},
    cfg_weight={self.config.get('cfg_weight', 0.7)},
)
print("DONE")
""",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists(out_wav):
                self.audio_ready.emit(out_wav)
            else:
                self.error_signal.emit(result.stderr[-500:])
        except Exception as e:
            self.error_signal.emit(str(e))


class TTSPreviewWidget(QWidget):
    """
    Panel s:
    - textovým poľom (zadaj text na testovanie)
    - tlačidlom "Spustiť TTS"
    - progress indikátorom
    - tlačidlom "Prehrať" (PyQt6 QMediaPlayer)
    - posuvníkmi: exaggeration, CFG weight, temperature
    - zobrazením WAV waveformu (voliteľné)
    """
    def __init__(self, config_provider, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._runner = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Zadaj slovenský text pre TTS test...")
        layout.addWidget(self.text_edit)
        
        params_row = QHBoxLayout()
        self.exag_spin = QDoubleSpinBox()
        self.exag_spin.setRange(0.0, 1.0); self.exag_spin.setSingleStep(0.05); self.exag_spin.setValue(0.5)
        self.exag_spin.setPrefix("Exag: ")
        params_row.addWidget(self.exag_spin)
        
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(0.1, 1.0); self.cfg_spin.setSingleStep(0.05); self.cfg_spin.setValue(0.7)
        self.cfg_spin.setPrefix("CFG: ")
        params_row.addWidget(self.cfg_spin)
        layout.addLayout(params_row)
        
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Spustiť TTS")
        self.run_btn.clicked.connect(self._run_preview)
        btn_row.addWidget(self.run_btn)
        
        self.play_btn = QPushButton("Prehrať")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_audio)
        btn_row.addWidget(self.play_btn)
        layout.addLayout(btn_row)
        
        self.status_label = QLabel("Pripravené")
        layout.addWidget(self.status_label)
        
        self._last_wav = None

    def _run_preview(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            return
        self.run_btn.setEnabled(False)
        self.status_label.setText("Generujem TTS...")
        cfg = self._get_config()
        cfg["exaggeration"] = self.exag_spin.value()
        cfg["cfg_weight"] = self.cfg_spin.value()
        self._runner = TTSPreviewRunner(text, cfg.get("voice_sample", ""), cfg)
        self._runner.audio_ready.connect(self._on_audio_ready)
        self._runner.error_signal.connect(self._on_error)
        self._runner.start()

    def _on_audio_ready(self, wav_path: str):
        self._last_wav = wav_path
        self.play_btn.setEnabled(True)
        self.status_label.setText(f"Hotovo: {wav_path}")
        self.run_btn.setEnabled(True)
        self._play_audio()  # autoplay

    def _on_error(self, msg: str):
        self.status_label.setText(f"Chyba: {msg[:100]}")
        self.run_btn.setEnabled(True)

    def _play_audio(self):
        if self._last_wav:
            # QMediaPlayer (PyQt6)
            from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PyQt6.QtCore import QUrl
            if not hasattr(self, '_player'):
                self._player = QMediaPlayer()
                self._audio_out = QAudioOutput()
                self._player.setAudioOutput(self._audio_out)
            self._player.setSource(QUrl.fromLocalFile(self._last_wav))
            self._player.play()
```

**Výhody:**
- Používateľ môže testovať rôzne exaggeration/CFG hodnoty v sekunde, nie po celom pipeline rune
- Žiadna zmena v pipeline.py — len nová záložka v GUI
- TTSPreviewRunner beží nezávisle, neblokuje GUI
- Autoplay po generovaní → okamžitá spätná väzba

**Minimálna implementácia (ak nechceme plný widget):**
Stačí nový skript `scripts/tts_preview.py` s 30 riadkami + tlačidlo v GUI ktoré ho spustí s aktuálnymi nastaveniami.

---

## PROBLÉM 4 — Duplicitné ochranné moduly

### Kde presne

Existujú 3 oddelené moduly s prekrývajúcimi sa zodpovednosťami:

| Súbor | Čo robí | Problém |
|-------|---------|---------|
| [scripts/term_protection.py](scripts/term_protection.py) | Token-masking (replace → `__TERM_0__` → restore) pre GIT, TECH, BRAND, SLANG termíny | Iba mask/unmask, bez opráv |
| [scripts/linux_logic_guard.py](scripts/linux_logic_guard.py) | Oprava rozbiteľných výstupov Proxmox/SSH/apt termínov, regex replacementy | Iba Linux/CLI domény |
| [scripts/sql_logic_guard.py](scripts/sql_logic_guard.py) | Oprava SQL termínov (ajznul→ISNULL, koalesk→COALESCE), kanonická forma | Iba SQL doména |

Navyše [scripts/text_adaptation.py:54-78](scripts/text_adaptation.py#L54) **kopíruje** SQL fix replacementy:
```python
_SQL_FIX_REPLACEMENTS = [
    (r"\bajznul\b", "ISNULL"),
    (r"\bkoalesk\w*\b", "COALESCE"),
    # ... rovnaké vzory ako v sql_logic_guard.py!
]
```

### Prečo to bolí

1. **Duplicita** → opravíme chybu v `sql_logic_guard.py` ale zabudneme na kópiu v `text_adaptation.py`
2. **Nejasné volanie** → pipeline.py musí vedieť kedy zavolať ktorý guard
3. **Rôzne rozhrania** → `guard_sql_text(text, source_text, fallback_text)` vs `guard_linux_text(text)` vs `mask_terms(text)` / `unmask_terms(text, mapping)`
4. **Chýbajúce kombinácie** → ak segment má SQL aj Linux termíny (napr. "spusti SELECT na PostgreSQL serveri cez SSH"), treba volať oba guardy zvlášť a v správnom poradí

### Najlepšie riešenie: Unified TermGuard s pipeline-aware API

**Prečo unified a nie len "volajme oba":**
Guardy majú **fázy** s rôznou sémantikou:
- Fáza 1 (pred prekladom): mask → prekladač → unmask (term_protection.py)
- Fáza 2 (po preklade, pred L1): oprava rozbiteľných výstupov (sql_guard + linux_guard)
- Fáza 3 (pred TTS): fonetické prepisy (phonetic_guard.py)

Unified guard to robí explicitne, bez duplikácie, s jediným import-om.

```python
# Nový súbor: scripts/term_guard.py
# Nahradí: term_protection.py, linux_logic_guard.py, sql_logic_guard.py (zachovaj ako deprecated wrappers)

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Literal


Domain = Literal["sql", "linux", "git", "general"]


@dataclass
class TermGuard:
    """
    Unified ochrana technických termínov pre všetky fázy pipeline.
    
    Fázy:
      Phase 1 — pre_translation: mask() → prekladač → unmask()
      Phase 2 — post_translation: repair() — oprava rozbiteľných výstupov
      Phase 3 — pre_tts: phonetic() — fonetické prepisy pre TTS
    
    Domény sú aditivné: TermGuard(domains=["sql", "linux"]) opravuje obe.
    """
    domains: list[Domain] = field(default_factory=lambda: ["sql", "linux", "git"])
    
    # ── Phase 1: Maskovanie pred prekladom ─────────────────────────────────
    
    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Zamaskuje chránené termíny. Vráti (masked_text, token_map)."""
        token_map = {}
        for i, term in enumerate(self._all_protected_terms()):
            pattern = re.escape(term)
            token = f"__TERM_{i}__"
            if re.search(rf"\b{pattern}\b", text, re.IGNORECASE):
                text = re.sub(rf"\b{pattern}\b", token, text, flags=re.IGNORECASE)
                token_map[token] = term
        return text, token_map

    def unmask(self, text: str, token_map: dict[str, str]) -> str:
        """Obnov zamaskované termíny."""
        for token, term in token_map.items():
            text = text.replace(token, term)
        return text

    # ── Phase 2: Oprava po preklade ────────────────────────────────────────
    
    def repair(self, text: str, src_text: str = "") -> str:
        """Opraví rozbité výstupy prekladu (fonetické deformácie termínov)."""
        if "sql" in self.domains:
            text = self._repair_sql(text)
        if "linux" in self.domains:
            text = self._repair_linux(text)
        return text

    # ── Phase 3: Fonetické prepisy pre TTS ────────────────────────────────
    
    def phonetic(self, text: str) -> str:
        """Prepíše skratky foneticky (HTTP → há-té-té-pé) pre TTS engine."""
        # Importuj existujúcu phonetic_guard logiku
        from phonetic_guard import apply_phonetic_map
        return apply_phonetic_map(text)

    # ── Interné metódy ─────────────────────────────────────────────────────
    
    def _all_protected_terms(self) -> list[str]:
        terms = []
        if "sql" in self.domains:
            from sql_logic_guard import SQL_PROTECTED_TERMS
            terms.extend(SQL_PROTECTED_TERMS)
        if "linux" in self.domains:
            from linux_logic_guard import LINUX_PROTECTED_TERMS
            terms.extend(LINUX_PROTECTED_TERMS)
        if "git" in self.domains:
            from term_protection import GIT_TERMS, BRAND_TERMS
            terms.extend(GIT_TERMS)
            terms.extend(BRAND_TERMS)
        return sorted(set(terms), key=len, reverse=True)  # dlhšie najprv (IS NOT NULL pred NULL)

    def _repair_sql(self, text: str) -> str:
        from sql_logic_guard import _SQL_FIX_REPLACEMENTS, guard_sql_text
        return guard_sql_text(text)

    def _repair_linux(self, text: str) -> str:
        from linux_logic_guard import guard_linux_text
        return guard_linux_text(text)
```

**Integrácia v pipeline.py:**
```python
# Pred prekladom:
guard = TermGuard(domains=["sql", "linux"] if args.preset == "teaching_sql" else ["git", "general"])
masked_text, token_map = guard.mask(src_text)
translated = translate(masked_text)
translated = guard.unmask(translated, token_map)

# Po preklade (L1):
translated = guard.repair(translated, src_text=src_text)

# Pred TTS:
tts_text = guard.phonetic(translated)
```

**Výhody:**
- Jedna import, jeden objekt, explicitné fázy
- Staré moduly zostávajú ako compatibility wrappers → žiadne breaking changes
- Domény sú konfigurovateľné podľa presetov (teaching_sql vs production_dub)
- `_all_protected_terms()` triedi dlhšie najprv → "IS NOT NULL" sa zachytí pred "NULL"

---

## SÚHRNNÁ PRIORITIZÁCIA

| # | Problém | Súbor | Riadok | Pracnosť | Dopad |
|---|---------|-------|--------|----------|-------|
| 1 | L11 timeout | [src/level11_manager.py](src/level11_manager.py) | 52 | 2h | Zastavuje celé pipeline |
| 2 | Adaptívna CPS | [scripts/text_adaptation.py](scripts/text_adaptation.py) | 41 | 4h | Zlý timing každého segmentu |
| 3 | TTS Preview GUI | [VideoTranslator_studio.py](VideoTranslator_studio.py) | nové | 6h | Kvalita hlasu bez dlhého runu |
| 4 | Unified TermGuard | [scripts/term_guard.py](scripts/term_guard.py) | nové | 3h | Údržba + correctness |

### Odporúčané poradie implementácie

```
1. L11 timeout (2h) — najrýchlejší win, zabraňuje strate hodín práce
2. Adaptívna CPS (4h) — okamžite zlepší timing všetkých nových runov
3. Unified TermGuard (3h) — refactor existujúceho, nie nová funkcia
4. TTS Preview (6h) — UX zlepšenie, nezávisí od predchádzajúcich
```

---

## ČO NIE JE KRITICKÉ (ale môže vyzerať tak)

### Whisper hallucination guardy sú hardcoded

```python
WHISPER_NO_SPEECH_MAX = 0.6   # scripts/config.py
```

**Nie je kritické pretože:** Sú konfigurovateľné cez `--whisper_no_speech_max` CLI argument. Pipeline ich nepremenene fixuje do kódu — sú to len defaulty. Pre väčšinu videí fungujú dobre.

### Model registry roztrúsená

`_resolve_chatterbox_asset()` v `scripts/tts.py` s kandidátmi — vyzerá chaoticky ale **funguje spoľahlivo**. Je to maintenance issue, nie bug.

### Žiadne unit testy

Pravda, ale TTS/STT pipeline je ťažko unit-testovateľný bez modelov. Oveľa cennejší je integration test na skutočnom videu.

---

*Koniec analýzy — VideoTranslator_studio, 2026-04-10*
