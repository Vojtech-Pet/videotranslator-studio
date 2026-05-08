# VideoTranslator_studio — Kompletná technická analýza
> Vygenerované: 2026-04-10 | Model: claude-sonnet-4-6 | Stav: aktualizované po opravách FIX 1–26 a runtime zmenách 2026-04-10

---

## 1. PREHĽAD PROJEKTU

**VideoTranslator_studio** je desktopová aplikácia (Qt6) pre dabingový preklad videí (EN → SK/CS a ďalšie jazyky). Kombinuje:
- Whisper STT pre transkripciu originálu
- Viac prekladových engineov (Google, MADLAD, Llama, MultiSlav, Gemma)
- Timing-aware text adaptation (Character-per-Second fitting) s **adaptívnou CPS kalibráciou**
- Viacero TTS engineov (Chatterbox SK, S2-Pro/Fish Speech, API)
- Sofistikovaný multi-level spracovateľský framework (Levely 1–22)
- Komplexnú audio montáž s voliteľnou separáciou hudby

**Štatistiky:**
- ~114 Python súborov, ~34 500 riadkov kódu
- Hlavný jazyk výstupu: Slovak (sk), Czech (cs), EN, DE, FR, PL, RU...
- Primárny TTS: Chatterbox SK (lokálny finetuned model)
- Vstup: ľubovoľné video (MP4, MKV, MOV...)
- Výstup: dabovaný video + SRT titulky + JSON metadata

**Doplnkove orientacne dokumenty:**
- `ANALYZA_SLABYCH_MIEST.md` - prioritizovany risk audit systemu podla dopadu na runtime
- `ANALYZA_GAP_AUDIT.md` - potvrdene vynechane alebo poddokumentovane oblasti oproti realnemu kodu
- `BUG_MAP.md` - rychla mapa bugov, symptomov, hotspotov a invariantov
- `MAPA_KODU.md` - coverage index celeho aktualneho Python kodu po suboroch
- `ARCHITECTURE_CURRENT.puml` - aktualna PlantUML mapa routingu, hlasov a syncu

---

## 2. ADRESÁROVÁ ŠTRUKTÚRA

Poznamka:
Tato sekcia je architektonicky prehlad, nie uplny suborovy coverage index.
Kompletny index aktualnych Python suborov je v `MAPA_KODU.md`.

```
VideoTranslator_studio/
│
├── VideoTranslator_studio.py     # Hlavná Qt6 GUI aplikácia (3101 riadkov)
├── main.py                       # CLI vstupný bod → scripts/pipeline.py
├── run_pipeline.py               # Orchestrátor celého pipeline
├── run_level5.py … run_level11.py # Jednotlivé level runnery
│
├── scripts/                      # Jadro spracovania
│   ├── config.py                 # Konfigurácia, konštanty, argument parser (469 riadkov)
│   ├── pipeline.py               # Hlavný STT→Preklad→TTS→Montáž pipeline (2960 riadkov)
│   ├── stt.py                    # Speech-to-text (Whisper, VAD, extrakcia audia)
│   ├── translation.py            # Prekladové enginy (Google, MADLAD, Llama, MultiSlav...)
│   ├── tts.py                    # TTS syntéza (Chatterbox, S2-Pro, XTTS, API)
│   ├── audio_pipeline.py         # Montáž, time-stretch, Demucs, mastering
│   ├── text_normalizer.py        # NeMo-inšpirovaná verbalizácia čísel/dátumov/čas
│   ├── text_adaptation.py        # Timing-aware úprava textu (CPS fitting)
│   ├── cps_calibrator.py         # ★ NOVÝ: Adaptívna CPS kalibrácia (EMA, cross-run)
│   ├── phonetic_guard.py         # Fonetická ochrana SQL/tech termínov
│   ├── term_protection.py        # Systém ochrany termínov
│   ├── linux_logic_guard.py      # Ochrana Linux/CLI termínov
│   ├── sql_logic_guard.py        # Ochrana SQL kľúčových slov
│   ├── translategemma.py         # Google TranslateGemma integrácia
│   ├── level4_audio_feedback.py  # Audio QC (timing analýza)
│   ├── video_translate.py        # Standalone prekladový skript
│   ├── translation_qc.py         # Kontrola kvality prekladu
│   ├── sk_glossary.py            # Slovenské post-processing pravidlá
│   └── xtts_worker.py            # XTTS-špecifický worker
│
├── src/                          # Multi-level spracovateľský framework
│   ├── level1_text_hygiene.py    # Čistenie textu, normalizácia medzier
│   ├── level2_rewrite_agent.py   # LLM prepisovanie textu
│   ├── level3_timing_agent.py    # Timing adaptácia (CPS fitting)
│   ├── level4_audio_feedback.py  # Audio dĺžka audit
│   ├── level5_manager.py         # Generovanie variantov & skórovanie
│   ├── level6_manager.py         # Kvalitné metriky & evaluácia
│   ├── level7_manager.py         # Project memory management
│   ├── level8_manager.py         # Scene understanding
│   ├── level9_manager.py         # Learned decision framework
│   ├── level10_manager.py        # A/B testing framework
│   ├── level11_manager.py        # Multi-agent orchestrácia (s timeout ochranou)
│   ├── level12_manager.py        # Pokročilé filtrovanie
│   ├── level13_manager.py        # Dataset & curriculum building
│   ├── terminology_memory.py     # SQL/tech konzistencia termínov
│   ├── coherence_checker.py      # Sémantická koherencia
│   ├── text_agent.py             # Generovanie textu z promptov
│   ├── audio_agent.py            # Varianty audio syntézy
│   ├── eval_agent.py             # Multi-dimenzionálna evaluácia
│   ├── policy_agent.py           # Policy návrhy
│   ├── prompt_agent.py           # Výber/generovanie promptov
│   ├── memory_agent.py           # Session memory
│   ├── feature_extractor.py      # Feature extraction pre klasifikáciu
│   ├── segment_classifier.py     # Klasifikácia typov segmentov
│   ├── failure_memory.py         # Sledovanie zlyhaní & učenie
│   ├── failure_clusterer.py      # Klastrovanie zlyhaní
│   ├── project_memory.py         # Project-wide memory store
│   ├── policy_registry.py        # Registračná databáza politík
│   ├── persona_controller.py     # Konzistencia hovoriaceho (persona)
│   ├── scene_planner.py          # Scene-level plánovanie
│   ├── utils_io.py               # JSON I/O pomocníci
│   ├── utils_text.py             # Textové utility
│   ├── utils_sql.py              # SQL ochrana termínov
│   └── utils_audio.py            # Audio file utility
│
├── mini_level11/                 # Rozšírený framework (Levely 14–22)
│   ├── level14_full_pipeline.py  # Kompletný pipeline
│   ├── level15_export_pack.py    # Balenie výsledkov
│   ├── level16_tts_inference.py  # TTS inference runner
│   ├── level17_prepare_training_pack.py
│   ├── level18_prepare_curriculum_pack.py
│   ├── level19_auto_evaluate.py  # Automatická evaluácia
│   ├── level20_full_dubbing_engine.py
│   ├── level21_ingest_slovak_sources.py
│   ├── level22_prepare_chatterbox_training.py
│   ├── full_rewrite_sql_tts.py   # SQL-aware TTS rewrite
│   ├── fix_full_dataset.py       # Dataset fixup utility
│   ├── pipeline_whisperx_fish_retry.py  # WhisperX + Fish Speech pipeline
│   ├── agents/                   # Mini multi-agent komponenty
│   │   ├── eval_agent.py         # Skórovanie kandidátov
│   │   └── memory_agent.py       # Pipeline pamäť (JSON persistencia)
│   ├── core/                     # Jadro orchestrácie
│   └── memory/                   # Perzistentná pamäť
│
├── prompts/                      # LLM systémové prompty
│   ├── rewrite_system.txt        # Prepisovanie textu
│   └── rewrite_slot_system.txt   # Timing-aware prepisovanie
│
├── memory/                       # Perzistentný stav
│   ├── policy_registry.json      # Definície politík
│   └── project_memory.json       # Session pamäť
│
├── output/                       # Výsledky
│   ├── final/                    # Finálne výstupy
│   ├── wav_segments/             # Vygenerované audio chunky
│   └── cps_calibration.json      # ★ NOVÝ: Cross-run CPS kalibrácia
│
├── voices/                       # Vzorky hlasov pre klonovanie
├── languages/                    # i18n JSON súbory (UI preklady)
├── temp/                         # Dočasný pracovný adresár
├── musetalk_gui_config.json      # Konfigurácia TTS engineov, EQ profilov
└── VideoTranslator_studio.desktop # Launcher pre Linux desktop
```

---

## 3. END-TO-END DÁTOVÝ TOK

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VSTUPNÉ VIDEO (MP4/MKV/...)                        │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       STT FÁZA              │
                    │  ffmpeg → 16kHz WAV          │
                    │  Silero VAD segmentácia      │
                    │  Whisper large-v3            │
                    │  Hallucination guard         │
                    │  (no_speech_prob, logprob,   │
                    │   compression_ratio)         │
                    │  → segmenty {start,end,text} │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      PREKLADOVÁ FÁZA        │
                    │  Term protection            │
                    │  (SQL, Linux CLI termíny)   │
                    │  Engine selection:          │
                    │  Google / MADLAD /          │
                    │  Llama / MultiSlav /        │
                    │  TranslateGemma / NLLB /    │
                    │  OpenAI-compatible API      │
                    │  Hybrid mode + LLM QA       │
                    │  Sliding window context     │
                    │  ★ API-SW: previous context │
                    │    + term memory + batch SRT│
                    │  ★ prompty bez tvrdého CPS  │
                    │    a bez limitu dĺžky       │
                    │  Term restoration           │
                    │  → text_src + text (SK)     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   TEXT ADAPTÁCIA (L1–L4)    │
                    │  L1: Hygiene (cleanup)       │
                    │  L2: LLM rewrite             │
                    │  L3: CPS fitting             │
                    │      ★ adaptive CPS z        │
                    │        CPSCalibrator (EMA)   │
                    │      keep/concise/dub_friendly│
                    │  L4: Audio feedback audit    │
                    │  → tts_input text            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  MULTI-LEVEL REFINEMENT     │
                    │  (voliteľné, L5–L13)        │
                    │  L5: Variant generation     │
                    │  L6: Quality metrics         │
                    │  L7: Project memory         │
                    │  L8: Scene understanding    │
                    │  L9: Learned decisions      │
                    │  L10: A/B testing           │
                    │  L11: Multi-agent orch.     │
                    │       ★ s 45s timeout/pokus  │
                    │       PromptAgent →          │
                    │       TextAgent →            │
                    │       AudioAgent →           │
                    │       EvalAgent             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │         TTS FÁZA            │
                    │  Text normalizácia pre TTS  │
                    │  (skratky, SQL fonetika)    │
                    │  Voice cloning (voliteľné)  │
                    │  Engine per segment:        │
                    │  Chatterbox SK / S2-Pro /   │
                    │  Turbo / OpenAI / Grok /    │
                    │  Gemini                     │
                    │  ★ hlasové priority:        │
                    │    multi_voice > custom WAV │
                    │    > emotion/original       │
                    │  ★ custom WAV + emotion =   │
                    │    style-only transfer      │
                    │    (bez miešania identity)  │
                    │  Silence trimming           │
                    │  → wav_segments/tts_XXXX.wav│
                    │  ★ L4 audit → CPS update    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      AUDIO MONTÁŽ           │
                    │  Timeline / concat mode     │
                    │  ★ sync-safe default:       │
                    │    preserve original gaps   │
                    │    (compress len voliteľne) │
                    │  Time-stretch (librosa /    │
                    │   ffmpeg atempo)            │
                    │  Demucs (voliteľné)         │
                    │  hudba + speech mix         │
                    │  Mastering chain:           │
                    │  EQ → noise reduction →     │
                    │  speech gain → normalize    │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴──────────────────────┐
              │                                           │
     ┌────────▼────────┐                      ┌──────────▼──────────┐
     │  Dabované VIDEO  │                      │  SRT titulky +      │
     │  (MP4 + audio)   │                      │  JSON metadata      │
     └─────────────────┘                      └─────────────────────┘
```

---

## 3.1. ROUTING OPRÁV GRAMATIKY A GEMMA REFINE

Toto je dôležitá prevádzková pravda, ktorá musí byť čitateľná aj bez otvárania kódu:

V projekte neexistuje len jedna "oprava gramatiky", ale tri vrstvy:

1. **Deterministické SK post-fixy**
   - Bežia pre `sk/slk` výstup automaticky po preklade.
   - Zahŕňajú opravy podmieňovacej gramatiky, skloňovania, cudzích mien a source-guided cleanup.
   - Toto **nie je Gemma refine**.

2. **Full-document LLM refine pass**
   - Toto je to, čo GUI označuje ako `Opraviť gramatiku (2× LLM)`.
   - Spúšťa sa len vtedy, keď pipeline dostane `--refine_translation`.
   - Pri Ollama adaptácii ide cez Ollama.
   - Inak ide cez `llama.cpp`, ak existuje model a runtime je dostupný.

3. **Audit repair pre flagged segmenty**
   - Po exporte JSON sa pre `sk/slk` spúšťa audit.
   - Ak audit označí problematické segmenty, môže nasledovať selektívny repair.
   - Tento repair je podmienený tým, že je dostupný opravný model alebo Ollama route.

### Praktický význam checkboxu `Opraviť gramatiku (2× LLM)`

Tento checkbox **neznamená "či sa urobí akákoľvek oprava gramatiky"**.
Znamená len:

- pridať alebo nepridať **full-document Gemma/Ollama refine pass** nad už hotový preklad

Ak je checkbox vypnutý:

- deterministické SK fixy stále bežia
- audit stále beží
- flagged repair môže stále zasiahnuť, ak je dostupný repair backend
- ale hlavný full-document refine pass sa nespustí

### Engine matrix pre refine správanie

| Engine | Checkbox `chk_refine` | Full-document refine | Poznámka |
|---|---|---|---|
| Google / MADLAD / Llama / TranslateGemma / MultiSlav / NLLB | OFF | Nie | Bežia len základné SK fixy + audit |
| Google / MADLAD / Llama / TranslateGemma / MultiSlav / NLLB | ON | Áno | Pridá sa `--refine_translation` |
| Hybrid | OFF | Áno | Hybrid si Gemma refine zapína sám |
| Hybrid | ON | Áno | Checkbox tu nič zásadné nemení |
| ChatGPT / Grok / Gemini | OFF | Nie | Bežia len základné SK fixy + audit |
| ChatGPT / Grok / Gemini | ON | Nie cez tento checkbox | GUI momentálne refine pass explicitne nepridáva |

### Dôležitá interpretácia

Checkbox je dnes pomenovaný širšie, než aké má reálne správanie.

Presnejší mentálny model je:

- `chk_refine = full LLM post-refine`
- nie `chk_refine = všetky jazykové opravy`

To je dôležité pri debugovaní, lebo používateľ môže mať pocit, že pri vypnutom checkboxe sa "neopravuje nič", čo nie je pravda.

---

## 4. UML KOMPONENTOVÝ DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VideoTranslator_studio.py (GUI)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │DropZone  │  │ StepBar  │  │  Runner  │  │BatchItem │  │AppSett. │  │
│  │(drag&drop│  │(progress)│  │(QThread) │  │(state)   │  │(config) │  │
│  └──────────┘  └──────────┘  └────┬─────┘  └──────────┘  └─────────┘  │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │ subprocess.Popen
                                     │ args → stdout → Qt signals
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      scripts/pipeline.py (main)                         │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  STT Module  │  │ Translation  │  │  TTS Module  │                  │
│  │  stt.py      │  │ translation  │  │  tts.py      │                  │
│  │              │  │ .py          │  │              │                  │
│  │ ▸ Whisper    │  │ ▸ Google     │  │ ▸ Chatterbox │                  │
│  │ ▸ VAD        │  │ ▸ MADLAD     │  │ ▸ S2-Pro     │                  │
│  │ ▸ ffmpeg     │  │ ▸ Llama      │  │ ▸ Turbo      │                  │
│  │ ▸ halluc.    │  │ ▸ MultiSlav  │  │ ▸ XTTS       │                  │
│  │   guard      │  │ ▸ Gemma      │  │ ▸ OpenAI API │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                  │                          │
│         └────────┬────────┘                  │                          │
│                  ▼                           ▼                          │
│  ┌─────────────────────────┐  ┌─────────────────────────┐              │
│  │   Text Adaptation       │  │   Audio Pipeline        │              │
│  │   text_adaptation.py    │  │   audio_pipeline.py     │              │
│  │   text_normalizer.py    │  │                         │              │
│  │   phonetic_guard.py     │  │ ▸ time_stretch_librosa  │              │
│  │   ★ cps_calibrator.py   │  │ ▸ time_stretch_ffmpeg   │              │
│  │                         │  │ ▸ concat_wavs           │              │
│  │ ▸ adaptive CPS fitting  │  │ ▸ Demucs separation     │              │
│  │ ▸ keep/concise/dub      │  │ ▸ EQ mastering          │              │
│  │ ▸ term protection       │  │                         │              │
│  └─────────────────────────┘  └─────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              src/ — Multi-Level Processing Framework                    │
│                                                                         │
│  L1 TextHygiene → L2 RewriteAgent → L3 TimingAgent → L4 AudioFeedback  │
│       ↓                                                                 │
│  L5 VariantGen → L6 QualityMetrics → L7 ProjectMemory → L8 SceneUnder. │
│       ↓                                                                 │
│  L9 LearnedDecision → L10 ABTesting → L11 MultiAgentOrch.              │
│                                              │                          │
│                          ┌───────────────────┼──────────────────┐      │
│                          ▼                   ▼                  ▼      │
│                    PromptAgent         TextAgent           AudioAgent  │
│                          └───────────────────┬──────────────────┘      │
│                                              ▼                          │
│                                         EvalAgent                      │
│                                         PolicyAgent                    │
│       ↓                                                                 │
│  L12 AdvancedFilter → L13 DatasetBuilder                                │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              mini_level11/ — Extended Framework (L14–L22)               │
│  L14 FullPipeline → L15 ExportPack → L16 TTSInference                  │
│  L17 TrainingPrep → L18 CurriculumPrep → L19 AutoEval                  │
│  L20 FullDubbingEngine → L21 IngestSK → L22 ChatterboxTrainingPrep     │
│  + agents/ (EvalAgent, MemoryAgent)                                     │
│  + pipeline_whisperx_fish_retry.py                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1 UML — Aktuálny routing promptov, syncu a hlasov

```
                     ┌────────────────────────────────────┐
                     │ TRANSLATION ROUTING (2026-04)      │
                     └────────────────────────────────────┘

source segments
   │
   ├─ Google / MADLAD / TranslateGemma / Llama
   │
   └─ API translate
       │
       ├─ batch SRT
       ├─ Previous translated context
       ├─ terminology memory
       └─ relaxed prompt
            ├─ no hard "fit duration"
            ├─ no "similar sentence length"
            └─ timing/CPS až v text_adaptation.py


                     ┌────────────────────────────────────┐
                     │ CHATTERBOX VOICE PRIORITY          │
                     └────────────────────────────────────┘

multi_voice local cue
        │
        ▼
multi_voice cluster voice
        │
        ▼
explicit custom WAV
        │
        ├─ emotion OFF  → keep custom voice
        └─ emotion ON   → keep custom voice
                         + derive style from original segment
                         + map style → exag/cfg/temp
        │
        ▼
emotion clone from original segment
        │
        ▼
global voice sample / model default


                     ┌────────────────────────────────────┐
                     │ AUDIO ASSEMBLY DEFAULT             │
                     └────────────────────────────────────┘

original segment times
        │
        ├─ default      → preserve timeline gaps
        └─ optional     → --compress_timeline_gaps
```

---

## 5. UML SEKVENČNÝ DIAGRAM — LEVEL 11 MULTI-AGENT ORCHESTRÁCIA (s timeoutom)

```
GUI/CLI    Level11Manager     ThreadPoolExecutor   _run_single_attempt   EvalAgent
    │             │                   │                    │                  │
    │── process ─►│                   │                    │                  │
    │             │── submit(attempt)─►│                    │                  │
    │             │                   │── run ────────────►│                  │
    │             │                   │   PromptAgent.get_prompt()            │
    │             │                   │   TextAgent.generate()                │
    │             │                   │   AudioAgent.synthesize()             │
    │             │                   │   EvalAgent.select_best() ───────────►│
    │             │                   │◄──────────────────────────────────────│
    │             │◄─ result(t=45s) ──│                    │                  │
    │             │                   │                    │                  │
    │   [TimeoutError] ───────────────┘                    │                  │
    │             │ log WARNING                            │                  │
    │             │ fallback = _make_timeout_fallback()    │                  │
    │             │ break                                  │                  │
    │◄─ fallback ─│                   │                    │                  │
```

---

## 6. UML SEKVENČNÝ DIAGRAM — CPS KALIBRÁCIA (cross-run)

```
Run N                pipeline.py             CPSCalibrator         cps_calibration.json
    │                     │                       │                        │
    │ štart                │                       │                        │
    │────────────────────►│── load() ────────────►│◄── read() ─────────────│
    │                     │◄─ effective_cps ───────│    (n=94, ema=13.52)   │
    │                     │                       │                        │
    │                     │── adapt_segments_batch(chars_per_sec=13.52)    │
    │                     │                       │                        │
    │                     │   [TTS loop]           │                        │
    │                     │   _timing_audit_log    │                        │
    │                     │── update_from_audit() ►│                        │
    │                     │◄─ n_updated=47 ────────│                        │
    │                     │── save() ─────────────►│── write() ────────────►│
    │                     │                       │    (n=141, ema=13.47)   │
    │ koniec              │                       │                        │

Run N+1:  effective_cps = 13.47  (konverguje k reálnej hodnote)
```

---

## 7. UML STAVOVÝ DIAGRAM — SEGMENT ŽIVOTNÝ CYKLUS

```
                ┌──────────┐
                │  RAW STT │
                │ {text,   │
                │  start,  │
                │  end}    │
                └────┬─────┘
                     │ translation
                     ▼
                ┌──────────┐
                │TRANSLATED│
                │ + text_src│
                │ + text(SK)│
                └────┬─────┘
                     │ L1 hygiene
                     ▼
                ┌──────────┐
                │ L1 TEXT  │
                │ hygiene  │
                └────┬─────┘
                     │ L2 rewrite
                     ▼
                ┌──────────┐
                │ L2 TEXT  │◄── LLM rewrite
                └────┬─────┘    (ak CPS > 1.10×)
                     │ L3 timing
                     ▼
            ┌────────┴──────────────────────────────┐
            │         CPS CHECK (adaptive)           │
            │   CPS ≤ 1.10× → "keep"                 │
            │   CPS ≤ 1.25× → "concise"              │
            │   CPS > 1.25× → "dub_friendly"         │
            │   ★ target_cps z CPSCalibrator.ema     │
            └────────┬──────────────────────────────┘
                     │
                     ▼
                ┌──────────┐
                │ L3 TEXT  │ tts_input
                └────┬─────┘
                     │ TTS syntéza
                     ▼
                ┌──────────┐
                │  WAV     │ tts_XXXX.wav
                │  CHUNK   │
                └────┬─────┘
                     │ L4 audit
                     ▼
            ┌────────┴──────────────────┐
            │      DURATION CHECK       │
            │  delta ≤ 0.3s → "ok"      │
            │  delta > 0.3s → "too_long" │
            │  delta < -0.5s → "too_short"│
            └────────┬──────────────────┘
                     │ ★ → CPS kalibrácia update
                     ▼
                ┌──────────┐
                │  FINAL   │ → montáž
                │  SEGMENT │
                └──────────┘
```

---

## 8. KONFIGURAČNÝ SYSTÉM

### 8.1 musetalk_gui_config.json — GUI konfigurácia
```json
{
  "py": "/path/to/python",
  "ffmpeg": "ffmpeg",
  "def_tts_engine": "chatterbox | s2_pro | turbo | openai | grok | gemini",
  "def_trans_engine": "translategemma | madlad | google | multislav",
  "chatterbox_model": "cesta k modelu (.safetensors)",
  "s2_pro_model": "cesta k S2-Pro checkpointu",
  "chk_lipsync": false,
  "chk_keep_music": false,
  "chk_context": false,
  "chk_refine": true,
  "chk_voice_eq": true,
  "chk_multi_voice": false,
  "cb_emo_clone": false,
  "def_pause_gap_s": "5.00"
}
```

### 8.2 Argument Parser (scripts/config.py) — kľúčové parametre

| Parameter | Default | Popis |
|-----------|---------|-------|
| `--input` | — | vstupné video |
| `--tgt_lang` | `sk` | cieľový jazyk |
| `--whisper_model` | `large-v3` | veľkosť Whisper modelu |
| `--use_chatterbox` | False | aktivuje Chatterbox SK |
| `--use_madlad` | False | MADLAD prekladač |
| `--use_google_translate` | False | Google Translate |
| `--hybrid_translate` | False | Google + MADLAD hybrid |
| `--context_aware` | False | kontextový preklad |
| `--clone_voice` | False | klonovanie hlasu |
| `--emotion_clone` | False | prenos emócie/štýlu; pri custom WAV len style transfer, pri origináli full per-segment clone |
| `--keep_music` | False | zachovať hudbu (Demucs) |
| `--timeline_mode` | False | nový montážny režim |
| `--compress_timeline_gaps` | False | skomprimovať medzery na timeline (default OFF pre sync-safe výstup) |
| `--timed` | False | timing-sync mód |
| `--use_api_translate` | False | OpenAI-compatible LLM preklad so sliding-window batchingom |
| `--vad` | False | Silero VAD segmentácia |
| `--preset` | `default` | `production_dub` / `teaching_sql` |
| `--sql_tts_normalize` | False | SQL fonetika |
| `--pitch_shift` | 0 | zmena výšky hlasu (semitóny) |

### 8.3 EQ Profily (14 presetov)
- `Chatterbox Natural` — kompenzácia boxiness + nasality + harshness
- `Voice Clone` — jemná kompenzácia pre klonovaný hlas
- `Audiobook` — teplý, čistý hlas
- `Podcast` — speech clarity
- `Hegen`, `Warm Voice`, `Speech Clarity` — špeciálne profily

### 8.4 Prekladové Presets
- `production_dub` — čistý dabing (fonetika OFF, SQL-TTS OFF)
- `teaching_sql` — SQL/IT obsah (všetky normalizéry ON)

---

## 9. ZÁVISLOSTI A MODELY

### Python balíčky
| Balíček | Účel |
|---------|------|
| PyQt6 | GUI framework |
| torch, transformers | Deep learning |
| llama-cpp-python | Lokálny LLM (Gemma, GGUF) |
| faster-whisper | STT (CTranslate2 backend) |
| librosa | Time-stretch, MFCC |
| soundfile | WAV I/O |
| silero-vad | Voice Activity Detection |
| sklearn | MFCC clustering (multi-voice) |
| numpy, scipy | Numerické výpočty |
| requests | HTTP API volania |

### Externé modely
| Model | Umiestnenie | Účel |
|-------|-------------|------|
| Whisper large-v3 | Auto-download HF | STT |
| Chatterbox SK v2.2 | `~/Ai/models/chatterbox/` | TTS (standard) |
| Chatterbox Turbo v1 | `~/Ai/models/chatterbox/` | TTS (fast) |
| Chatterbox Edge v2.5 | `~/Ai/models/chatterbox/` | TTS (edge) |
| Fish Speech S2-Pro | `checkpoints/s2-pro/` | TTS (voice clone) |
| Gemma-3-12B Q8_0 | `~/Ai/models/` | LLM rewrite |
| MADLAD-400 | HF cache | Preklad |
| MultiSlav-5lang | HF cache | Slavic preklad |
| Demucs | pip/conda | Separácia hudby |

### Auto-migrácia ciest modelov
```
1. ~/Ai/models/chatterbox/           (primárny)
2. BASE_DIR/models/chatterbox/       (fallback 1)
3. ~/miniforge3/envs/fish_env/       (S2-Pro Python env)
```

---

## 10. DÁTOVÉ ŠTRUKTÚRY

### 10.1 Segment JSON (kumulovaný počas pipeline)
```json
{
  "start": 0.08,
  "end": 29.73,
  "text": "Preložený slovenský text",
  "text_src": "Original English text",
  "slot": 29.65,
  "level1_text": "text po hygiene",
  "level2_text": "text po rewrite",
  "level3_text": "text po timing adaptácii",
  "tts_input": "finálny text pre TTS",
  "cps_before": 12.55,
  "cps_after": 12.55,
  "rewrite_mode": "keep | concise | dub_friendly",
  "level4_audit": {
    "seg": 1,
    "slot": 29.65,
    "wav_duration": 28.5,
    "delta": -1.15,
    "status": "ok | too_long | too_short | missing_wav",
    "wav_path": "output/wav_segments/tts_0001.wav"
  },
  "level5_scores": {
    "meaning_score": 0.95,
    "term_score": 1.0,
    "cps_score": 0.9,
    "duration_score": 0.95,
    "spoken_score": 0.88,
    "asr_match_score": 0.92,
    "coherence_score": 0.87
  },
  "multi_agent_selected": {
    "candidate_name": "candidate_001",
    "prosody_name": "chatterbox",
    "prompt_name": "balanced",
    "final_score": 0.912,
    "status": "ok | timeout_fallback"
  }
}
```

### 10.2 CPS Kalibrácia (output/cps_calibration.json)
```json
{
  "base_cps": 14.5,
  "ema": 13.47,
  "n": 141,
  "alpha": 0.20,
  "min_cps": 8.0,
  "max_cps": 20.0,
  "min_chars": 15,
  "min_samples": 3
}
```
- `n < 3` → `effective_cps = base_cps = 14.5` (fallback)
- `n ≥ 3` → `effective_cps = ema` (naučená hodnota)

### 10.3 Výstupné súbory
```
output/
├── {stem}_{lang}_segments.json    # Plná metadata pre každý segment
├── {stem}_{lang}_metadata.json    # Sumárne štatistiky
├── {stem}_{lang}.srt              # Titulky s časovaním
├── cps_calibration.json           # ★ Cross-run CPS kalibrácia
├── wav_segments/
│   ├── tts_0001.wav               # Individual TTS chunky (24kHz, mono)
│   └── ...
└── final/
    └── {stem}_{lang}_dubbed.mp4   # Finálne dabované video
```

### 10.4 Checkpoint systém
```
scripts/batch_checkpoint.json      # Resume schopnosť
{
  "current_index": 5,
  "total": 12,
  "errors": ["file3.mp4: TTS timeout"],
  "completed": ["file1.mp4", "file2.mp4"]
}
```

---

## 11. KOMUNIKÁCIA GUI ↔ PIPELINE

```
GUI Thread (Qt main)
       │
       │  subprocess.Popen(args)
       ▼
Runner (QThread)
       │
       │  reads stdout line by line
       │  parses patterns:
       │    "[Segment N/M]" → progress signal
       │    "[STEP N/M]"    → step progress signal
       │    "tqdm lines"    → suppress (noise)
       │
       ├─ log_line_signal → GUI log panel
       ├─ progress_signal → StepBar update
       └─ finished_signal → batch next item

Pipeline subprocess (scripts/pipeline.py)
       │
       │  reads/writes:
       │    musetalk_gui_config.json (konfig)
       │    output/{stem}_segments.json
       │    output/wav_segments/*.wav
       │    output/{stem}.srt
       │    output/cps_calibration.json  ★ nové
       │    batch_checkpoint.json (resume)
```

---

## 12. OPRAVENÉ PROBLÉMY (FIX 1–26)

Prehľad všetkých opravených bugov — detaily v `CHANGELOG_OPRAV.md`.

| Fix | Súbor | Typ problému | Stav |
|-----|-------|-------------|------|
| 1 | `src/level11_manager.py` | L11 loop bez timeoutu → pipeline visí | ✅ ThreadPoolExecutor + 45s |
| 2 | `scripts/cps_calibrator.py` | Fixná CPS 14.5 pre všetky runy | ✅ Nový modul, EMA kalibrácia |
| 3 | `scripts/text_adaptation.py` | `adapt_segments_batch` bez `chars_per_sec` param | ✅ Nový parameter |
| 4 | `scripts/pipeline.py` | Žiadna integrácia CPS kalibrácie | ✅ load/save CPSCalibrator |
| 5 | `scripts/text_adaptation.py` | Mŕtvy kód `_SQL_FIX_REPLACEMENTS` (19 riadkov) | ✅ Odstránený |
| 6 | `scripts/pipeline.py` | `"_cps_cal" in dir()` — nespoľahlivý check | ✅ `is not None` |
| 7 | `scripts/pipeline.py` | 3× ffmpeg/ffprobe bez `timeout=` | ✅ FFMPEG_TIMEOUT |
| 8 | `scripts/tts.py`, `audio_pipeline.py` | 8× subprocess bez `timeout=` | ✅ FFMPEG_TIMEOUT |
| 9 | `scripts/tts.py` | S2-Pro response check chýba | N/A — už existoval |
| 10 | `src/level2_rewrite_agent.py` | `response["choices"][0]` KeyError | ✅ try/except |
| 11 | `src/project_memory.py` | `json.loads()` bez try/except | ✅ JSONDecodeError guard |
| 12 | `src/policy_registry.py` | `json.loads()` bez try/except | ✅ JSONDecodeError guard |
| 13 | `mini_level11/agents/eval_agent.py` | `scored[0]` IndexError pri prázdnych kandidátoch | ✅ guard `if not scored` |
| 14 | `mini_level11/agents/memory_agent.py` | `json.loads()` bez try/except | ✅ JSONDecodeError guard |
| 15 | `mini_level11/pipeline_whisperx_fish_retry.py` | `payload["response"]` KeyError + json.loads | ✅ try/except pre oba |
| 16 | `mini_level11/full_rewrite_sql_tts.py`, `fix_full_dataset.py` | `json.loads()` bez try/except | ✅ oba súbory |
| 17 | `mini_level11/level22_prepare_chatterbox_training.py` | json.loads + subprocess bez timeout | ✅ oba |
| 18 | `mini_level11/level16,20,21` | subprocess bez `timeout=` | ✅ 300s/600s |
| 19 | `scripts/pipeline.py`, `scripts/translation.py` | API preklad nešiel cez sliding-window kontext | ✅ `use_api_translate` → API-SW batch routing |
| 20 | `scripts/translation.py` | Prekladové prompty tlačili na timing/dĺžku príliš skoro | ✅ timing tlak odstránený; CPS až v adaptácii |
| 21 | `scripts/translation.py` | Cache prekladov miešala rôzne modely/providerov/prompty | ✅ cache scope rozšírený o engine/src/provider/model/content |
| 22 | `scripts/pipeline.py`, `scripts/config.py`, `VideoTranslator_studio.py`, `musetalk_gui_config.json` | Timeline gap compression rozbíjala sync s obrazom | ✅ default preserve gaps; `--compress_timeline_gaps` je voliteľný |
| 23 | `VideoTranslator_studio.py` | S2 GUI neukazovalo normálne WAV reference hlasy | ✅ zjednotený refresh a voice list |
| 24 | `VideoTranslator_studio.py`, `scripts/pipeline.py` | custom WAV + emotion clone/multi_voice miešali viac hlasov | ✅ konfliktné kombinácie blokované a priorita hlasov explicitná |
| 25 | `scripts/pipeline.py` | Pri custom WAV nebolo možné preniesť len štýl bez zmeny identity | ✅ nový style-only emotion transfer (`exag/cfg/temp`) |
| 26 | `scripts/pipeline.py` | `_extract_segment_audio()` používal `FFMPEG_TIMEOUT` bez importu | ✅ import z `config.py` doplnený |

---

## 13. AKTUÁLNE ZNÁME PROBLÉMY A LIMITÁCIE

### 13.1 Architektonické limitácie (zostatok po opravách)

#### STREDNÉ
- **Žiadny real-time preview** — nie je možné počúvať medzivýsledky TTS bez celého runu
- **Správa ciest modelov** — komplexná auto-migrácia; chýba unified model registry
- **S2-Pro checkpoint detekcia je krehká** — viacero kandidátov s nejasnou prioritou
- **Whisper hallucination guardy sú hardcoded** — pevné prahové hodnoty bez adaptácie na doménu
- **MFCC clustering pre multi-voice** — primitívny prístup; pyannote.audio by bol presnejší
- **Turbo fallback logika je krehká** — 80+ riadkov regexov
- **Style-only emotion transfer je heuristický** — používa energiu/aktivitu/dynamiku segmentu, nie disentangled emotion model
- **Prompt stack je stále rozvetvený** — `translation.py` má viac vetiev (API/Llama/context/full-doc), ktoré sú po zjednotení lepšie, ale stále zdieľajú logiku len čiastočne

#### MENŠIE
- **Time-stretch artefakty na krátkych segmentoch** — `n_fft=4096` môže byť príliš veľké pre <1s audio
- **Demucs bez timeout handling** — spomalenie bez upozornenia
- **Žiadne unit testy** — len smoke testy
- **Duplicitné ochranné moduly** — `term_protection.py`, `linux_logic_guard.py`, `sql_logic_guard.py` sa prekrývajú
- **Historické analytické súbory môžu driftovať** — tento súbor je aktuálnejší zdroj pravdy než staršie level reporty

### 13.2 Chýbajúce funkcie

| Funkcia | Dôvod pridania |
|---------|---------------|
| Segment editácia v GUI | Všetky úpravy sú cez JSON, nie UI-friendly |
| A/B porovnanie v GUI | Level 10 A/B existuje ale výsledky nie sú vizualizované |
| Real-time TTS preview | Rýchla kontrola bez celého runu |
| Rollback na predchádzajúce checkpointy | Mimo batch-level žiadny revert |
| Fallback chain pre prekladové enginy | Keď všetky enginy zlyhajú, žiadna graceful degradation |
| Paralelizácia Level 5-13 variantov | Sekvenčné generovanie je pomalé |
| Online policy learning | Statický policy registry bez adaptácie |
| Natívny prosody/style model | Heuristika pre custom WAV je užitočná, ale nie je to plný emotion disentanglement |
| Voice routing debug panel v GUI | Priorita hlasov už existuje, ale používateľ ju nevidí explicitne |

### 13.3 Výkonnostné bottlenecky

```
1. NAJPOMALŠIE: Whisper large-v3 — ~1-2× realtime na GPU, 5-10× na CPU
2. POMALY: LLM rewrite (Gemma Q8_0 na CPU) — bez ETA v GUI
3. POMALÉ: Demucs separácia — resource-intensive, bez progress
4. SEKVENČNÉ: Level 5-13 multi-variant generovanie — mohlo by byť paralelné
```

### 13.4 Code quality

- Inline komentáre sú mix SK/EN
- Niektoré moduly (term_protection, linux_guard, sql_guard) majú prekrývajúce sa zodpovednosti
- Chýbajú type annotations (zvlášť v starších moduloch)
- Chýba logging framework — väčšina výstupu cez print()

---

## 14. EXTERNÉ API INTEGRÁCIE

| Služba | Env premenná | Použitie |
|--------|-------------|---------|
| Google Translate | (žiadna — free mobile endpoint) | Preklad |
| OpenAI-compatible chat API | `API_TRANS_KEY` | Kontextový sliding-window preklad (ChatGPT/Grok/Gemini compatible) |
| OpenAI TTS | `OPENAI_API_KEY` | TTS syntéza |
| Grok TTS | `GROK_API_KEY` | TTS syntéza |
| Gemini TTS | `GEMINI_API_KEY` | TTS syntéza |
| Hugging Face | `HF_API_KEY` | Download modelov |
| Ollama (lokálny) | `http://localhost:11434` | LLM inference |

---

## 15. LEVEL 6 — QUALITY METRICS (22 komponentov)

```
Váhová distribúcia skóre:
  meaning_score     22%  ──── sémantická ekvivalencia (src vs. tgt)
  terminology_score 15%  ──── SQL/tech termíny zachované
  cps_score         12%  ──── chars-per-second v rámci slotu
  duration_score    12%  ──── WAV dĺžka vs. slot
  spoken_score      10%  ──── prirozená hovorená forma
  asr_match_score   10%  ──── ASR roundtrip match
  coherence_score    9%  ──── koherencia s okolím
  iné                10% ──── zvyšné metriky
```
