# VideoTranslator_studio — Mapa Kódu
> Vygenerované: 2026-04-10 | Stav: coverage index aktuálneho kódu

## 1. Účel

Tento súbor dopĺňa [ANALYZA_SYSTEMU.md](/home/vojtech/VideoTranslator_studio/ANALYZA_SYSTEMU.md).

Jeho cieľ nie je znovu vysvetliť architektúru, ale pokryť celý aktuálny Python kód na úrovni súborov:

- čo je runtime kritické
- čo je runner
- čo je pomocná utilita
- čo je experimentálna alebo tréningová vetva
- čo je dnes source of truth a čo je len sidecar tooling

## 2. Coverage contract

Táto mapa pokrýva všetkých **115 Python súborov** v projekte.

Rozdelenie:
- top-level entrypointy: 10
- `scripts/`: 30
- `src/`: 51
- `mini_level11/`: 24

Táto mapa je:
- úplná na úrovni súborov
- orientačná na úrovni zodpovednosti
- zámerne stručná pri experimental/tooling vetvách

Táto mapa nie je:
- plná dokumentácia každej funkcie
- náhrada za runtime bug analýzu
- náhrada za čítanie kritických hotspotov v `pipeline.py`, `translation.py`, `text_adaptation.py`, `tts.py`

## 3. Source Of Truth

Keď treba pochopiť projekt bez hádania, čítaj v tomto poradí:

1. [ANALYZA_SYSTEMU.md](/home/vojtech/VideoTranslator_studio/ANALYZA_SYSTEMU.md)
2. [ANALYZA_SLABYCH_MIEST.md](/home/vojtech/VideoTranslator_studio/ANALYZA_SLABYCH_MIEST.md)
3. [ANALYZA_GAP_AUDIT.md](/home/vojtech/VideoTranslator_studio/ANALYZA_GAP_AUDIT.md)
4. [MAPA_KODU.md](/home/vojtech/VideoTranslator_studio/MAPA_KODU.md)
5. [BUG_MAP.md](/home/vojtech/VideoTranslator_studio/BUG_MAP.md)
6. [ARCHITECTURE_CURRENT.puml](/home/vojtech/VideoTranslator_studio/ARCHITECTURE_CURRENT.puml)
7. [CHANGELOG_OPRAV.md](/home/vojtech/VideoTranslator_studio/CHANGELOG_OPRAV.md)

## 4. Top-Level Entry Pointy

| Súbor | Rola | Stav |
|---|---|---|
| `VideoTranslator_studio.py` | hlavná Qt GUI, prevod UI stavu do CLI flagov | runtime kritický |
| `main.py` | CLI vstupný bod, spúšťa pipeline | runtime kritický |
| `run_pipeline.py` | wrapper/orchestrátor pre pipeline spustenia | aktívny runner |
| `run_level5.py` | runner pre Level 5 | sidecar runner |
| `run_level6.py` | runner pre Level 6 | sidecar runner |
| `run_level7.py` | runner pre Level 7 | sidecar runner |
| `run_level8.py` | runner pre Level 8 | sidecar runner |
| `run_level9.py` | runner pre Level 9 | sidecar runner |
| `run_level10.py` | runner pre Level 10 | sidecar runner |
| `run_level11.py` | runner pre Level 11 | sidecar runner |

## 5. `scripts/` — Core Runtime Vrstva

Toto je hlavná prevádzková vrstva projektu. Keď sa niečo pokazí pri reálnom preklade videa, chyba býva najčastejšie tu.

| Súbor | Rola | Stav |
|---|---|---|
| `scripts/__init__.py` | package marker | neutrálne |
| `scripts/audio_pipeline.py` | finálna montáž audia, stretch, mix, mastering | runtime kritický |
| `scripts/audit_sk_tts_json.py` | audit slovenského TTS JSON a bezpečné opravy | runtime aktívny |
| `scripts/compare_metrics.py` | porovnanie metrík medzi behovmi | pomocná utilita |
| `scripts/config.py` | parser argumentov, defaulty, CLI konfigurácia | runtime kritický |
| `scripts/cps_calibrator.py` | adaptívna CPS kalibrácia medzi behovmi | runtime aktívny |
| `scripts/eval_text_adaptation.py` | evaluácia text adaptation vrstvy | eval utilita |
| `scripts/level4_audio_feedback.py` | audio QC a spätná väzba na timing | runtime sidecar |
| `scripts/linux_logic_guard.py` | ochrana Linux/CLI technických termínov | runtime aktívny |
| `scripts/mini_level11_bridge.py` | most medzi hlavnou pipeline a mini-level11 vetvou | integračný sidecar |
| `scripts/musetalk_gui.py` | podporná GUI vrstva okolo Musetalk workflow | sidecar GUI |
| `scripts/phonetic_guard.py` | ochrana výslovnosti/fonetických termínov | runtime aktívny |
| `scripts/phonetic_guard_fix.py` | doplnkové fonetické fixy | pomocná utilita |
| `scripts/phonetic_guard_rewrite.py` | prepisy pre fonetickú bezpečnosť | pomocná utilita |
| `scripts/pipeline.py` | hlavný STT → preklad → adaptácia → TTS → montáž pipeline | runtime kritický |
| `scripts/sk_glossary.py` | slovenské post-processing a glossary pravidlá | runtime aktívny |
| `scripts/sql_logic_guard.py` | ochrana SQL logiky a termínov | runtime aktívny |
| `scripts/stt.py` | extrakcia audia, VAD, Whisper transkripcia | runtime kritický |
| `scripts/term_protection.py` | ochrana zamknutých termínov pri preklade | runtime aktívny |
| `scripts/text_adaptation.py` | timing-aware skracovanie a CPS fitting | runtime kritický |
| `scripts/text_normalizer.py` | normalizácia textu pre TTS | runtime kritický |
| `scripts/translategemma.py` | integrácia TranslateGemma vetvy | runtime aktívny |
| `scripts/translation.py` | prekladové routovanie, prompty, cache, sliding window | runtime kritický |
| `scripts/translation_agent.py` | agentová/pomocná prekladová vrstva | sidecar utilita |
| `scripts/translation_benchmark.py` | benchmark prekladových engineov | benchmark utilita |
| `scripts/translation_llm_benchmark.py` | benchmark LLM prekladov | benchmark utilita |
| `scripts/translation_qc.py` | kontrola kvality prekladu | runtime sidecar |
| `scripts/tts.py` | TTS generovanie, routovanie engineov, hlasové parametre | runtime kritický |
| `scripts/video_translate.py` | standalone skript pre preklad videa | sidecar runner |
| `scripts/xtts_worker.py` | XTTS špecifický worker | voliteľná runtime vetva |

## 6. `src/` — Multi-Level Framework

Táto vetva obsahuje framework Level 1–13, policy/eval pamäť a výskumnejšie agentové vrstvy. Nie všetko z nej je na kritickej ceste každého behu, ale je dôležitá pre vyššie levely, experimenty a rozhodovanie.

| Súbor | Rola | Stav |
|---|---|---|
| `src/__init__.py` | package marker | neutrálne |
| `src/ab_tester.py` | A/B porovnanie variantov | framework aktívny |
| `src/adaptive_policy.py` | adaptívne policy rozhodovanie | framework aktívny |
| `src/audio_agent.py` | audio agent pre varianty | framework aktívny |
| `src/auto_promoter.py` | promotion/selection logika kandidátov | framework sidecar |
| `src/candidate_reranker.py` | reranking text/audio kandidátov | framework aktívny |
| `src/coherence_checker.py` | koherencia medzi segmentmi | framework aktívny |
| `src/dataset_builder.py` | príprava datasetov pre učenie/eval | dataset tooling |
| `src/eval_agent.py` | eval agent | framework aktívny |
| `src/evaluation_set.py` | definície eval sád | eval tooling |
| `src/evaluator_model.py` | model/metódy evaluácie | framework aktívny |
| `src/failure_clusterer.py` | klastrovanie zlyhaní | framework sidecar |
| `src/failure_memory.py` | pamäť zlyhaní | framework aktívny |
| `src/feature_extractor.py` | extrakcia feature pre rozhodovanie | framework aktívny |
| `src/learning_logger.py` | logovanie učenia a rozhodnutí | framework sidecar |
| `src/level1_text_hygiene.py` | Level 1 čistenie textu | framework aktívny |
| `src/level2_rewrite_agent.py` | Level 2 rewrite agent | framework aktívny |
| `src/level3_timing_agent.py` | Level 3 timing agent | framework aktívny |
| `src/level4_audio_feedback.py` | Level 4 audio spätná väzba | framework aktívny |
| `src/level5_manager.py` | Level 5 orchestrácia | framework aktívny |
| `src/level6_manager.py` | Level 6 orchestrácia | framework aktívny |
| `src/level7_manager.py` | Level 7 orchestrácia | framework aktívny |
| `src/level8_manager.py` | Level 8 orchestrácia | framework aktívny |
| `src/level9_manager.py` | Level 9 orchestrácia | framework aktívny |
| `src/level10_manager.py` | Level 10 orchestrácia | framework aktívny |
| `src/level11_manager.py` | Level 11 orchestrácia | framework aktívny |
| `src/level12_manager.py` | Level 12 orchestrácia | framework rozšírenie |
| `src/level13_manager.py` | Level 13 orchestrácia | framework rozšírenie |
| `src/memory_agent.py` | agent nad projektovou pamäťou | framework aktívny |
| `src/persona_controller.py` | kontrola konzistencie persony/hlasového štýlu | framework sidecar |
| `src/policy_agent.py` | policy návrhy a rozhodovanie | framework aktívny |
| `src/policy_learner.py` | učenie policy z výsledkov | framework sidecar |
| `src/policy_registry.py` | registry policy pravidiel | framework aktívny |
| `src/policy_updater.py` | aktualizácia policy registrov | framework sidecar |
| `src/project_memory.py` | projektová pamäť | framework aktívny |
| `src/prompt_agent.py` | výber/generovanie promptov | framework aktívny |
| `src/prompt_memory.py` | pamäť promptov | framework sidecar |
| `src/prompt_ranker.py` | ranking promptov | framework sidecar |
| `src/prompt_router.py` | routovanie promptov podľa prípadu | framework aktívny |
| `src/prompt_synthesizer.py` | skladanie promptov | framework sidecar |
| `src/rule_miner.py` | dolovanie pravidiel zo zlyhaní/výsledkov | framework sidecar |
| `src/scene_planner.py` | plánovanie na úrovni scény | framework aktívny |
| `src/segment_classifier.py` | klasifikácia segmentov | framework aktívny |
| `src/strategy_search.py` | hľadanie stratégie pre varianty | framework sidecar |
| `src/terminology_memory.py` | pamäť terminológie | framework aktívny |
| `src/text_agent.py` | text agent | framework aktívny |
| `src/threshold_tuner.py` | ladenie thresholdov | framework sidecar |
| `src/utils_audio.py` | audio utility | framework utilita |
| `src/utils_io.py` | I/O utility | framework utilita |
| `src/utils_sql.py` | SQL utility | framework utilita |
| `src/utils_text.py` | text utility | framework utilita |

## 7. `mini_level11/` — Rozšírená Vetva Level 14–22

Táto vetva je praktický advanced pack pre rewrite, training pack, evaluáciu a dubbing engine experimenty. Nie je to jediná source-of-truth runtime cesta, ale je to aktívna rozširujúca vetva projektu.

| Súbor | Rola | Stav |
|---|---|---|
| `mini_level11/agents/__init__.py` | package marker | neutrálne |
| `mini_level11/agents/asr_agent.py` | ASR agent | advanced pack |
| `mini_level11/agents/audio_agent.py` | audio agent | advanced pack |
| `mini_level11/agents/eval_agent.py` | eval agent | advanced pack |
| `mini_level11/agents/memory_agent.py` | memory agent | advanced pack |
| `mini_level11/agents/orchestrator.py` | orchestrácia mini agentov | advanced pack |
| `mini_level11/agents/text_agent.py` | text agent | advanced pack |
| `mini_level11/core/__init__.py` | package marker | neutrálne |
| `mini_level11/core/prompts.py` | prompty pre mini pipeline | advanced pack |
| `mini_level11/core/sql_fixer.py` | SQL-aware fixer | advanced pack |
| `mini_level11/core/utils.py` | helper utility | advanced pack |
| `mini_level11/fix_full_dataset.py` | full dataset fixer | dataset utilita |
| `mini_level11/full_rewrite_sql_tts.py` | full SQL-aware rewrite utility | dataset utilita |
| `mini_level11/level14_full_pipeline.py` | Level 14 full pipeline | advanced pack |
| `mini_level11/level15_export_pack.py` | Level 15 export pack | advanced pack |
| `mini_level11/level16_tts_inference.py` | Level 16 TTS inference | advanced pack |
| `mini_level11/level17_prepare_training_pack.py` | Level 17 training pack | advanced pack |
| `mini_level11/level18_prepare_curriculum_pack.py` | Level 18 curriculum/training prep | advanced pack |
| `mini_level11/level19_auto_evaluate.py` | Level 19 automatická evaluácia | advanced pack |
| `mini_level11/level20_full_dubbing_engine.py` | Level 20 full dubbing engine scaffold | advanced pack |
| `mini_level11/level21_ingest_slovak_sources.py` | ingest slovenských zdrojov | training/data tooling |
| `mini_level11/level22_prepare_chatterbox_training.py` | training pack pre Chatterbox | training/data tooling |
| `mini_level11/pipeline_whisperx_fish_retry.py` | alternatívna pipeline s retry | experimental runner |
| `mini_level11/run_mini_level11.py` | runner mini-level11 vetvy | advanced runner |

## 8. Čo Nie Je Runtime Source Of Truth

Tieto cesty sú dôležité, ale nemajú sa zamieňať za aktuálnu architektúru:

- `temp/`
- `output/`
- historické JSON reporty
- jednorazové smoke výstupy
- staré metrics snapshoty
- cache a checkpoint JSON súbory

Patria sem dôkazy z behov, nie definícia systému.

## 9. Ako Čítať Projekt Pri Debugu

Ak ide o:

- zlý preklad: `VideoTranslator_studio.py` → `scripts/config.py` → `scripts/translation.py` → `scripts/pipeline.py`
- zlý timing: `scripts/text_adaptation.py` → `scripts/cps_calibrator.py` → `scripts/audio_pipeline.py`
- miešanie hlasov: `VideoTranslator_studio.py` → `scripts/pipeline.py` → `scripts/tts.py`
- zlý sync: `scripts/pipeline.py` → `scripts/audio_pipeline.py`
- Level 11+ rozhodovanie: `src/level*.py` + `src/*agent.py`
- advanced dataset/training pack: `mini_level11/level14+`

## 10. Poznámka K Predchádzajúcej Analýze

Predchádzajúca analýza bola architektonicky užitočná, ale nebola dostatočne explicitná v tom, že:

- nepokrýva celý kód súbor-po-súbore
- niektoré runtime výnimky boli iba implicitné
- používateľ mohol oprávnene čakať, že aj UI label nuance budú zapísané priamo v dokumentácii

Táto mapa to narovnáva: odteraz je jasné, že `ANALYZA_SYSTEMU.md` je architektúra a `MAPA_KODU.md` je coverage index celého aktuálneho kódu.
