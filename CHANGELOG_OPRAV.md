# Changelog opráv — VideoTranslator_studio
> 2026-04-10

---

## FIX 12 — API config split + secret hygiene + hash-safe reuse_tts

**Súbory:** [VideoTranslator_studio.py](/home/vojtech/VideoTranslator_studio/VideoTranslator_studio.py), [config.py](/home/vojtech/VideoTranslator_studio/scripts/config.py), [pipeline.py](/home/vojtech/VideoTranslator_studio/scripts/pipeline.py), [musetalk_gui.py](/home/vojtech/VideoTranslator_studio/scripts/musetalk_gui.py)

**Problém A — plaintext API kľúče:** GUI si ukladalo `openai_api_key`, `grok_api_key`, `gemini_api_key` a `hf_api_key` priamo do JSON configov.

**Oprava:**
- `AppSettings.save()` v Qt aj legacy Tk GUI teraz zapisuje config bez tajných kľúčov
- `AppSettings.load()` už nevracia staré uložené kľúče späť do runtime nastavení
- existujúce configy boli sanitizované: kľúče sú vymazané
- GUI polia ostali použiteľné pre aktuálnu session, ale perzistencia ide cez env premenné

**Problém B — miešanie prekladového modelu a TTS modelu:** pole `openai_tts_model` sa používalo aj ako model pre API preklad.

**Oprava:**
- nové samostatné polia:
  - `openai_translate_model` = `gpt-5-mini-2025-08-07`
  - `grok_translate_model`
  - `gemini_translate_model`
- UI teraz zobrazuje oddelene `Model (preklad)` a `Model (TTS)`
- builder pre `--use_api_translate` používa už len translate model
- OpenAI TTS default sa presunul na `gpt-4o-mini-tts`
- po live teste s project key z 2026-04-10 sa default upravil z nefunkčného `gpt-5.2-mini` na reálne dostupný `gpt-5-mini-2025-08-07`

**Problém C — nebezpečný `--reuse_tts`:** starý mechanizmus znovu použil WAV iba podľa toho, že súbor existoval a nebol prázdny.

**Oprava:**
- v `pipeline.py` pribudol semantic hash cache podľa:
  - vstupného textu
  - TTS engine
  - hlasovej/reference konfigurácie
  - dôležitých TTS prepínačov
- pri každom segmente sa ukladá `tts_XXXX.meta.json`
- `--reuse_tts` teraz znovu použije chunk iba keď sedí hash
- pri mismatch/missing meta sa segment bezpečne pregeneruje

**Dopad:** menej tichých regresií pri API preklade, nulová perzistencia kľúčov do configu a bezpečnejší reuse pri rerunoch.

**Poznámka k GPT-5 route:** v `translation.py` je doplnené `reasoning_effort="minimal"` + `max_completion_tokens` pre GPT-5 family, inak model vie spotrebovať celý budget na reasoning tokeny a vrátiť prázdny `message.content`.

---

## FIX 1 — Level 11: Timeout pre multi-agent loop

**Súbor:** [src/level11_manager.py](src/level11_manager.py)

**Problém:** `for attempt_index in range(self.max_retries + 1)` spúšťal LLM volania bez akéhokoľvek časového limitu. Ak llama-cpp zamrzlo alebo segment bol príliš dlhý → pipeline visela neobmedzene.

**Zmeny:**
- Pridané importy: `concurrent.futures`, `logging`
- Nová konštanta `_ATTEMPT_TIMEOUT_S = 45` (sekúnd na jeden attempt)
- Nová funkcia `_make_timeout_fallback(base)` — vráti originálny text ako fallback
- Nová metóda `Level11Manager._run_single_attempt()` — izoluje LLM + eval volania
- `process_segment()`: for loop obalený v `ThreadPoolExecutor`, každý attempt má `future.result(timeout=45)`
  - Pri `TimeoutError` → warning do logu + fallback, pipeline pokračuje
  - Pri inej `Exception` → error do logu + fallback, pipeline pokračuje

**Čo sa nezmenilo:** Logika výberu najlepšieho výsledku, všetky attempt záznamy, memory_agent.store() — všetko funguje rovnako. Zmenilo sa len to, že loop nemôže visieť.

---

## FIX 2 — Nový modul: scripts/cps_calibrator.py

**Súbor:** [scripts/cps_calibrator.py](scripts/cps_calibrator.py) *(nový)*

**Problém:** `SK_CHARS_PER_SEC = 14.5` v `text_adaptation.py` bola fixná konštanta pre každý segment a každý run. Reálna rýchlosť Chatterbox SK závisí od modelu, hlasovej vzorky a obsahu — pre SQL texty je to ~11-12 c/s, pre bežný text ~14-15 c/s.

**Čo robí:**
- `CPSCalibrator` trieda s EMA (exponenciálny kĺzavý priemer, alpha=0.20)
- `update(actual_cps, text_chars)` — aktualizuje stav z jedného merania
- `update_from_audit_log(timing_audit_log)` — hromadná aktualizácia z `_timing_audit_log` (pipeline.py)
- `effective_cps` property — vráti kalibrovaný CPS, alebo `base_cps=14.5` ak ešte nemáme ≥3 merania
- `save(path)` / `load(path)` — JSON perzistencia medzi runmi
- Outlier filter: ignoruje merania mimo [8.0, 20.0] c/s a segmenty kratšie ako 15 znakov

---

## FIX 3 — Adaptívna CPS kalibrácia v adapt_segments_batch

**Súbor:** [scripts/text_adaptation.py](scripts/text_adaptation.py)

**Problém:** `adapt_segments_batch` volala `adapt_for_timing` vždy s fixnou hodnotou `SK_CHARS_PER_SEC = 14.5`, bez možnosti externého override.

**Zmena:**
```python
# PRED:
def adapt_segments_batch(segments, llm=None, src_key="text_src", ...) -> ...:

# PO:
def adapt_segments_batch(segments, llm=None, src_key="text_src", ...,
                         chars_per_sec: float = SK_CHARS_PER_SEC) -> ...:
```
A v tele funkcie:
```python
# PRED:
result = adapt_for_timing(translated=text, src_text=src_text, slot_duration=slot_dur, llm=llm, verbose=verbose)

# PO:
result = adapt_for_timing(translated=text, src_text=src_text, slot_duration=slot_dur, llm=llm, verbose=verbose,
                          chars_per_sec=chars_per_sec)
```

**Backward compatibility:** Default hodnota `SK_CHARS_PER_SEC` — všetky existujúce volania bez parametra fungujú rovnako.

---

## FIX 4 — Integrácia CPS kalibrácie do pipeline.py

**Súbor:** [scripts/pipeline.py](scripts/pipeline.py)

**Zmena 1 — pred adapt_segments_batch** (v bloku `use_text_adaptation`):
```python
from cps_calibrator import CPSCalibrator
_cps_cal_path = out_dir / "cps_calibration.json"
_cps_cal = CPSCalibrator()
_cps_cal.load(_cps_cal_path)   # načíta z predchádzajúceho runu ak existuje
print(f"[CPS] Kalibrácia: {_cps_cal.effective_cps:.2f} c/s ...")
translated, _adapt_stats = adapt_segments_batch(
    ..., chars_per_sec=_cps_cal.effective_cps,
)
```

**Zmena 2 — po timing audit logu** (po TTS slučke):
```python
if "_cps_cal" in dir():
    _cps_updated = _cps_cal.update_from_audit_log(_timing_audit_log)
    if _cps_updated > 0:
        _cps_cal.save(_cps_cal_path)
        print(f"[CPS] Kalibrácia aktualizovaná → {_cps_cal.effective_cps:.2f} c/s ...")
```

**Efekt:**
- 1. run: `effective_cps = 14.5` (fallback, n=0)
- 2. run: `effective_cps = ~13.8` (z prvého runu, n=50+)
- 3. run: `effective_cps = ~13.5` (ďalej konverguje k reálnej hodnote)
- Log zobrazuje aktuálnu kalibráciu: `[CPS] Kalibrácia: 13.52 c/s (konfidencia=high, n=94)`

---

## FIX 5 — Odstránenie mŕtveho kódu _SQL_FIX_REPLACEMENTS

**Súbor:** [scripts/text_adaptation.py](scripts/text_adaptation.py)

**Problém:** `_SQL_FIX_REPLACEMENTS` (19-riadkový zoznam SQL regex opráv) bol definovaný v `text_adaptation.py` ale nikde v tomto súbore nepoužitý — identické vzory sú v `sql_logic_guard.py` a sú aplikované cez `guard_sql_text()` ktoré `fix_sql_terms()` volá priamo.

**Zmena:** Blok `_SQL_FIX_REPLACEMENTS = [...]` (19 riadkov) bol odstránený.

**Riziko:** Žiadne. Funkcia `fix_sql_terms()` deleguje na `guard_sql_text()` z `sql_logic_guard.py` — tá logika zostáva plne funkčná.

---

## Súhrn zmenených súborov

| Súbor | Typ zmeny |
|-------|-----------|
| [src/level11_manager.py](src/level11_manager.py) | Timeout ochrana, nový `_run_single_attempt`, `_make_timeout_fallback` |
| [scripts/cps_calibrator.py](scripts/cps_calibrator.py) | **Nový súbor** — adaptívna CPS kalibrácia |
| [scripts/text_adaptation.py](scripts/text_adaptation.py) | Nový param `chars_per_sec` v `adapt_segments_batch`, odstránený mŕtvy kód |
| [scripts/pipeline.py](scripts/pipeline.py) | Načítanie + ukladanie CPS kalibrácie, predanie do adapt |

---

## FIX 6 — Krehký `"_cps_cal" in dir()` check

**Súbor:** [scripts/pipeline.py](scripts/pipeline.py)

**Problém:** `if "_cps_cal" in dir()` je nespoľahlivý spôsob testovania existencie premennej. `dir()` vracia aj zdedené mená a môže dávať false positives.

**Zmena:** `_cps_cal = None` a `_cps_cal_path = None` pridané do inicializačného bloku premenných (riadok ~1863). Check zmenený na `if _cps_cal is not None:`.

---

## FIX 7 — Chýbajúce timeout na subprocess v pipeline.py

**Súbor:** [scripts/pipeline.py](scripts/pipeline.py)

`FFMPEG_TIMEOUT=300` a `FFPROBE_TIMEOUT=30` sú definované v `config.py` ale niektoré subprocess volania ich nepoužívali.

**Opravené volania:**
- `extract_emotion_audio_segment()` — ffmpeg trim bez timeout → pridaný `timeout=FFMPEG_TIMEOUT`
- `_wav_duration_seconds()` ffprobe fallback → pridaný `timeout=FFPROBE_TIMEOUT`
- MFCC speaker clustering ffmpeg → pridaný `timeout=FFMPEG_TIMEOUT`

---

## FIX 8 — Chýbajúce timeout na subprocess v tts.py a audio_pipeline.py

**Súbory:** [scripts/tts.py](scripts/tts.py), [scripts/audio_pipeline.py](scripts/audio_pipeline.py)

**tts.py — opravené volania:**
- `extract_voice_sample()` ffmpeg → `timeout=FFMPEG_TIMEOUT`
- `enhance_tts_audio()` ffmpeg → `timeout=FFMPEG_TIMEOUT`
- `piper_rs_tts()` subprocess → `timeout=FFMPEG_TIMEOUT`

**audio_pipeline.py — opravené volania:**
- `time_stretch_ffmpeg()` atempo → `timeout=FFMPEG_TIMEOUT`
- `trim_edge_silence_ffmpeg()` → `timeout=FFMPEG_TIMEOUT`
- Batch trim/fade per-segment (×2 volania) → `timeout=FFMPEG_TIMEOUT`
- Per-segment stretch → `timeout=FFMPEG_TIMEOUT`
- Timeline assembly `_assemble_with_filter_complex()` → `timeout=FFMPEG_TIMEOUT`

**Fix 9 — S2-Pro response check:** Preverením zistené že check **už existuje** (`if response.status_code != 200: raise RuntimeError`) — žiadna zmena potrebná.

---

---

## FIX 10 — level2_rewrite_agent.py: KeyError + FileNotFoundError

**Súbor:** [src/level2_rewrite_agent.py](src/level2_rewrite_agent.py)

**Bug A — riadok 75:** `content = response["choices"][0]["message"]["content"]` bez ochrany.
Ak llama-cpp vráti prázdnu alebo neúplnú odpoveď (napr. pri OOM alebo context overflow), padne s `KeyError`/`IndexError`.

```python
# PO:
try:
    content = response["choices"][0]["message"]["content"]
except (KeyError, IndexError, TypeError):
    return None   # caller (rewrite_text) spracuje None → vráti originál
```

**Bug B — riadok 24:** `read_text()` bez ochrany → `FileNotFoundError` ak prompt súbor chýba.

```python
# PO:
try:
    return Path(path).expanduser().resolve().read_text(encoding="utf-8").strip()
except OSError:
    return ""   # prázdny system prompt → LLM dostane len user message, nie crash
```

---

## FIX 11 — project_memory.py: JSONDecodeError bez ochrany

**Súbor:** [src/project_memory.py](src/project_memory.py)

**Bug:** `json.loads(self.path.read_text(...))` bez try/except. Ak bol súbor prerušene zapísaný (kill počas `save()`), načítanie havaruje a celý pipeline crashne.

```python
# PO:
try:
    self.data = json.loads(self.path.read_text(encoding="utf-8"))
except (json.JSONDecodeError, OSError):
    self.data = _default_memory_payload()   # začni s prázdnou pamäťou
```

---

## FIX 12 — policy_registry.py: JSONDecodeError bez ochrany

**Súbor:** [src/policy_registry.py](src/policy_registry.py)

**Bug:** Rovnaký problém ako `project_memory.py` — poškodený `policy_registry.json` crashne `Level11Manager.__init__` (volá `PolicyRegistry.__init__` → `load()`).

```python
# PO:
try:
    self.data = json.loads(self.path.read_text(encoding="utf-8"))
except (json.JSONDecodeError, OSError):
    pass   # zachovaj default {"active_policy": "policy_v1", "policies": {}}
```

---

## Falošné alarmy (agent hlásil, overením nepotvrdené)

| Pôvodný alarm | Skutočnosť |
|---------------|-----------|
| level6_manager.py:265 — division by zero | Chránené guard `if not segments: return summary` na riadku 259 |
| level6_manager.py:268 — IndexError | Chránené tým istým guard-om |
| level5_manager.py:40-41 — division by zero v tokenizácii | Chránené `if not reference_tokens` / `if not candidate_tokens` na riadkoch 35-37 |
| level3_timing_agent.py:90 — normalize_text(None) | `normalize_text` explicitne ošetruje None: `text or ""` |

---

---

## FIX 13 — mini_level11/agents/eval_agent.py: IndexError pri prázdnych kandidátoch

**Súbor:** [mini_level11/agents/eval_agent.py](mini_level11/agents/eval_agent.py)

**Bug:** `return scored[0]` v `choose_best()` — ak pipeline neposkytne žiadnych kandidátov, `scored` je prázdny list → `IndexError`.

```python
# PO:
if not scored:
    return {}
scored.sort(key=lambda item: item["final_score"], reverse=True)
return scored[0]
```

---

## FIX 14 — mini_level11/agents/memory_agent.py: JSONDecodeError bez ochrany

**Súbor:** [mini_level11/agents/memory_agent.py](mini_level11/agents/memory_agent.py)

**Bug:** `self.memory = json.loads(self.path.read_text(...))` — poškodený `pipeline_memory.json` crashne celý pipeline pri štarte.

```python
# PO:
try:
    self.memory = json.loads(self.path.read_text(encoding="utf-8"))
except (json.JSONDecodeError, OSError):
    self.memory = {"bad_terms": {}, "bad_patterns": {}, "good_prompts": {}, "failures": []}
```

---

## FIX 15 — mini_level11/pipeline_whisperx_fish_retry.py: KeyError + JSONDecodeError

**Súbor:** [mini_level11/pipeline_whisperx_fish_retry.py](mini_level11/pipeline_whisperx_fish_retry.py)

**Bug A — riadok 327:** `return str(payload["response"]).strip()` — ak Ollama zmení štruktúru odpovede (napr. model error), padne s `KeyError`.

```python
# PO:
try:
    return str(payload["response"]).strip()
except (KeyError, TypeError):
    return ""
```

**Bug B — riadok 549:** `data = json.loads(path.read_text(...))` v `load_segments_from_json()` — poškodený JSON súbor crashne pipeline.

```python
# PO:
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (json.JSONDecodeError, OSError) as exc:
    raise ValueError(f"Cannot read segments from {path}: {exc}") from exc
```

---

## FIX 16 — mini_level11/full_rewrite_sql_tts.py + fix_full_dataset.py: JSONDecodeError

**Súbory:** [mini_level11/full_rewrite_sql_tts.py](mini_level11/full_rewrite_sql_tts.py), [mini_level11/fix_full_dataset.py](mini_level11/fix_full_dataset.py)

**Bug:** `load_segments()` v oboch súboroch volá `json.loads(path.read_text(...))` bez try/except. Poškodený vstupný súbor → crash.

Oba súbory opravené rovnako:
```python
# PO:
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (json.JSONDecodeError, OSError) as exc:
    raise ValueError(f"Cannot read segments from {path}: {exc}") from exc
```

---

## FIX 17 — mini_level11/level22_prepare_chatterbox_training.py: JSONDecodeError + chýbajúci timeout

**Súbor:** [mini_level11/level22_prepare_chatterbox_training.py](mini_level11/level22_prepare_chatterbox_training.py)

**Bug A — riadok 100:** `json.loads(manifest_path.read_text(...))` v `load_rows()` — poškodený manifest crashne skript.

```python
# PO:
try:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
except (json.JSONDecodeError, OSError) as exc:
    raise ValueError(f"Cannot read manifest {manifest_path}: {exc}") from exc
```

**Bug B — riadok 185:** `subprocess.run(cmd, check=True, ...)` — ffmpeg resample bez `timeout` → môže visieť.

```python
# PO:
subprocess.run(cmd, check=True, stdout=DEVNULL, stderr=DEVNULL, timeout=300)
```

---

## FIX 18 — mini_level11: subprocess.run bez timeout (level16, level20, level21)

**Súbory:** [mini_level11/level16_tts_inference.py](mini_level11/level16_tts_inference.py), [mini_level11/level20_full_dubbing_engine.py](mini_level11/level20_full_dubbing_engine.py), [mini_level11/level21_ingest_slovak_sources.py](mini_level11/level21_ingest_slovak_sources.py)

- **level16** — ffmpeg loudnorm normalizácia: pridaný `timeout=300`
- **level20** — `_run()` helper (volá ffmpeg, ffprobe, yt-dlp): pridaný parameter `timeout: int = 300`, predávaný do `subprocess.run()`
- **level21** — `yt-dlp` sťahovanie: pridaný `timeout=600` (sieťový download môže trvať dlhšie)

---

## Súhrn mini_level11 opráv

| Súbor | Typ zmeny |
|-------|-----------|
| [mini_level11/agents/eval_agent.py](mini_level11/agents/eval_agent.py) | Guard pre prázdny `candidates` zoznam |
| [mini_level11/agents/memory_agent.py](mini_level11/agents/memory_agent.py) | JSONDecodeError ochrana pri načítaní pamäte |
| [mini_level11/pipeline_whisperx_fish_retry.py](mini_level11/pipeline_whisperx_fish_retry.py) | KeyError v Ollama response, JSONDecodeError v load_segments |
| [mini_level11/full_rewrite_sql_tts.py](mini_level11/full_rewrite_sql_tts.py) | JSONDecodeError v load_segments |
| [mini_level11/fix_full_dataset.py](mini_level11/fix_full_dataset.py) | JSONDecodeError v load_segments |
| [mini_level11/level22_prepare_chatterbox_training.py](mini_level11/level22_prepare_chatterbox_training.py) | JSONDecodeError v load_rows, timeout na ffmpeg |
| [mini_level11/level21_ingest_slovak_sources.py](mini_level11/level21_ingest_slovak_sources.py) | Timeout na yt-dlp subprocess |
| [mini_level11/level16_tts_inference.py](mini_level11/level16_tts_inference.py) | Timeout na ffmpeg subprocess |
| [mini_level11/level20_full_dubbing_engine.py](mini_level11/level20_full_dubbing_engine.py) | Timeout parameter v `_run()` helper |

---

## FIX 19 — Prosody vrstvy: Pause Map + Pitch Contour

**Súbor:** [scripts/pipeline.py](scripts/pipeline.py)

**Kontext:** Chatterbox čítal text sekvenčne bez vedomia kde v origináli boli pauzy alebo aká bola expresívnosť hlasu (pitch). `_derive_style_transfer_params` merala len RMS energiu → jeden skalár na segment.

### Nová funkcia: `_extract_pause_map()`

Extrahuje relatívne pozície páuz (0.0–1.0) z originálneho segmentu pomocou VAD cez RMS prah.

- Prahová hodnota: `percentile(rms, 20)` — adaptívna na segment
- Minimálna pauza: `0.28s` (kratšie sú prirodzené medzislovy, nie skutočné pauzy)
- Ignoruje okraje segmentu (prvých/posledných 8% = padding artefakty)

### Nová funkcia: `_inject_pauses(text, positions)`

Vkladá čiarky do preloženého textu na pozíciách zodpovedajúcich pauzám originálu.
Rešpektuje existujúcu interpunkciu — neprepíše `.!?` na konci slov.

### Nová funkcia: `_extract_pitch_stats()`

Extrahuje pitch štatistiky pomocou `librosa.pyin`:
- `pitch_median_hz` — stredná výška hlasu
- `pitch_range_hz` — rozsah (p90 − p10) — indikátor expresívnosti
- `voiced_ratio` — podiel znelých frames

### Rozšírenie `_derive_style_transfer_params()`

Pred výpočtom `exag/cfg/temp` sa teraz zavolá `_extract_pitch_stats()`:

```
# PRED:
combined = energy_score
exag = base_exag + (style_score - 0.5) * 0.30

# PO:
pitch_score = clamp((pitch_range_hz - 30) / 130, 0, 1)
combined_score = 0.50 * style_score + 0.30 * pitch_score + 0.20 * dynamics_score
exag = base_exag + (combined_score - 0.5) * 0.35
```

Log output rozšírený o `pitch_range_hz` a `pitch_score`.

### Integrácia do TTS loop

Pause map sa aktivuje ak `--emotion_clone` je zapnuté a nie je SQL mód:
```
[PAUSE-MAP] Segment 7: 2 pause(s) injected at ['0.32', '0.71']
```

Výsledok je zaznamenaný v `_step["pause_map"]` pre JSON metadata.

---

## FIX 20 — Prosody vrstva 4: Energy Contour per-chunk

**Súbory:** [scripts/pipeline.py](scripts/pipeline.py), [scripts/tts.py](scripts/tts.py)

**Nová funkcia `_extract_energy_contour(src_wav, start, end, n_chunks, ...)`**

Rozdelí originálny segment na `n_chunks` časových okien. Pre každé okno:
- RMS energia → `energy_score`
- RMS dynamika (std/mean) → `dynamics_score`
- `chunk_score = 0.65 * energy_score + 0.35 * dynamics_score`
- Výsledok: `{exaggeration, cfg_weight, temperature}` per chunk

Namiesto jedného skalára na celý segment má teraz každý text-chunk vlastné parametre odvodené z toho, čo sa v origináli dialo *v danej časti*.

**Rozšírenie `chatterbox_tts()`:**
```python
# Nové parametre:
chunk_overrides: list[dict] | None = None   # per-chunk exag/cfg/temp
chunk_prompts: list[str | None] | None = None  # per-chunk reference WAV
```
V chunk loope sa pre každý index `ci` vezme `chunk_overrides[ci]` namiesto globálnych hodnôt.

---

## FIX 21 — Prosody vrstva 5: Per-chunk Style Embedding

**Súbory:** [scripts/pipeline.py](scripts/pipeline.py), [scripts/tts.py](scripts/tts.py)

**Nová funkcia `_extract_chunk_references(src_wav, start, end, n_chunks, ...)`**

Pre každý chunk extrahuje zodpovedajúci časový úsek originálneho audia:
- Chunk 0 → originál 0%–(100/N)%
- Chunk 1 → originál (100/N)%–(200/N)%
- ...

Ak je úsek kratší ako `min_ref_s=1.5s`, rozšíri sa do susedných okien (max do hraníc segmentu). Ak stále príliš krátky → fallback na globálny audio_prompt.

**Integrácia v TTS loope:**
```
[PROSODY] Segment 7: 3 chunks | energy_contour=True | chunk_refs=3/3
[CHATTERBOX] Chunk 1/3 exag=0.62 cfg=0.47 temp=0.63 [custom_ref]: 'Dnes si pozrieme...'
[CHATTERBOX] Chunk 2/3 exag=0.71 cfg=0.44 temp=0.67 [custom_ref]: 'a ukážeme ako funguje...'
[CHATTERBOX] Chunk 3/3 exag=0.55 cfg=0.51 temp=0.61 [custom_ref]: 'výsledok je teda...'
```

Chunk refs sa vymažú po TTS syntéze (temp WAVy v `chunks_dir`).

**Podmienky aktivácie** (obe vrstvy):
- `--emotion_clone` zapnuté
- segment nie je SQL mód
- text sa reálne splituje na ≥2 chunky

**Metadata** sú uložené v `_step["prosody_layers"]` pre JSON výstup.

---

## Čo sa neopravovalo (dôvod)

| Problém | Dôvod odloženia |
|---------|----------------|
| TTS Preview GUI | Vyžaduje novú záložku v Qt6 GUI — väčší rozsah, nezávislý od ostatných opráv |
| Unified TermGuard | Refactor existujúcich modulov — rizikovejší, potrebuje testing pred mergom |
| Whisper hallucination guardy | Sú konfigurovateľné cez CLI args, nie kritické |
| Model registry | Maintenance issue, nie aktívny bug |
