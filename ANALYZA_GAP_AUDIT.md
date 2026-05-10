# VideoTranslator_studio — Gap Audit Dokumentácie
> Vygenerované: 2026-04-10 | Typ: audit vynechaných alebo poddokumentovaných častí

## 1. Účel

Tento súbor nerieši ešte slabé miesta implementácie.

Rieši len jednu otázku:

**Čo v dokumentácii ešte chýbalo alebo bolo príliš zovšeobecnené oproti reálnemu kódu?**

Tým pádom je to most medzi:
- [ANALYZA_SYSTEMU.md](/home/vojtech/VideoTranslator_studio/ANALYZA_SYSTEMU.md)
- [MAPA_KODU.md](/home/vojtech/VideoTranslator_studio/MAPA_KODU.md)
- budúcou analýzou slabých miest

## 2. Metóda auditu

Audit bol robený proti aktuálnemu kódu v týchto vrstvách:

- top-level entrypointy
- `scripts/`
- `src/`
- `mini_level11/`
- GUI state a checkpoint logika v `VideoTranslator_studio.py`
- parser argumentov v `scripts/config.py`
- hlavný runtime flow v `scripts/pipeline.py`
- prekladová vrstva v `scripts/translation.py`
- timing vrstva v `scripts/text_adaptation.py`
- TTS vrstva v `scripts/tts.py`

Použité tvrdé počty:

- Python súbory v projekte: **115**
- CLI flagy v `scripts/config.py`: **175**
- explicitne rozpísané CLI flagy v `ANALYZA_SYSTEMU.md`: **19**
- polia v `AppSettings`: **73**

## 3. Čo už dokumentácia pokrýva dobre

Aby audit nebol nefér, toto už bolo pokryté dobre alebo aspoň použiteľne:

- hlavná architektúra STT → preklad → adaptácia → TTS → montáž
- základná adresárová štruktúra
- hlavné runtime súbory
- API sliding-window preklad
- sync-safe preserve gaps default
- custom WAV + style-only emotion transfer
- Gemma refine routing po posledných doplneniach
- bug hotspoty okolo hlasov, syncu a prompt pressure

## 4. Potvrdené vynechané oblasti

### 4.1 GUI stav a perzistencia boli poddokumentované

Dokumentácia doteraz nepokrývala dostatočne, že GUI nie je len “frontend”, ale aj **riadiaca a perzistentná vrstva**.

Chýbalo alebo bolo len naznačené:

- `AppSettings` má **73 polí**
- GUI perzistuje nielen všeobecné nastavenia, ale aj:
  - engine defaulty
  - API keys
  - API base URL
  - hlasové voľby pre Chatterbox/S2/Turbo
  - adapt provider
  - Ollama URL/model
  - VAD defaulty
  - EQ a denoise nastavenia
- existujú samostatné state súbory:
  - `musetalk_gui_config.json`
  - `batch_checkpoint.json`
  - `video_lists.json`
  - `saved_checkpoints.json`
- existuje migrácia starých model pathov:
  - `_migrate_chatterbox_model_path`
  - `_migrate_s2_pro_model_path`
  - `_migrate_turbo_model_path`

Praktický význam:

- runtime správanie vie “ožiť” zo starej konfigurácie aj bez zjavnej zmeny v GUI
- nie všetky problémy vznikajú v pipeline, časť vzniká v persisted state

### 4.2 `run_pipeline.py` bolo pomenované príliš všeobecne

V dokumentácii bolo vedené ako “orchestrátor celého pipeline”.

Reálne:

- `main.py` + `scripts/pipeline.py` sú hlavná runtime cesta pre produkčný beh
- `run_pipeline.py` je **Level 1 → Level 4 sidecar orchestrátor**, nie plná produkčná cesta

To je dôležité, lebo názov súboru znie širšie než jeho skutočná zodpovednosť.

### 4.3 CLI dokumentácia bola len výberová, nie coverage-level

To samo osebe nie je chyba, ale nebolo to explicitne pomenované.

Reálny stav:

- parser má **175 flagov**
- v analýze bolo explicitne rozpísaných iba **19**

Chýbajúce skupiny, ktoré neboli rozčlenené ako samostatné režimy:

- bypass a resume módy
  - `--use_existing_segments`
  - `--translation_only`
  - `--reuse_tts`
  - `--srt_align`
- advanced translation routes
  - `--use_translategemma_fulldoc`
  - `--use_madlad_gemma_qa`
  - `--hybrid_qa`
  - `--use_nllb_gemma`
  - `--use_qa`
- advanced pre-TTS stack
  - `--use_text_adaptation`
  - `--use_mini_level11`
  - `--use_level6_plus`
  - `--use_ollama_adaptation`
  - `--adapt_provider`
  - `--adapt_ollama_url`
  - `--adapt_ollama_model`
- TTS recovery a fallbacky
  - `--tts_timing_retry`
  - `--tts_timing_retry_max_attempts`
  - `--turbo_auto_fallback`
  - `--turbo_fallback_model`
  - `--turbo_fallback_device`
- TTS engine branches
  - `--use_s2_pro`
  - `--use_chatterbox_turbo`
  - `--use_xtts`
  - `--use_piper`
  - `--use_piper_rs`
  - `--use_chatterbox_onnx`
  - `--use_openai_tts`
  - `--use_api_tts`

### 4.4 Hlavné režimy v `pipeline.py` neboli rozdelené ako samostatné prevádzkové módy

Dokumentácia popisovala hlavný tok, ale neoddeľovala tieto runtime režimy:

- **`goto_tts` režim**
  - aktivovaný cez `--use_existing_segments`
  - preskočí STT, preklad aj adaptáciu
  - používa existujúci `{stem}_{lang}_segments.json`

- **`translation_only` režim**
  - nekončí hneď po preklade
  - prejde všetky pre-TTS textové vrstvy
  - končí až po JSON/SRT exporte bez TTS

- **pre-TTS text stack**
  - audit + selective repair
  - mini_level11
  - level12
  - level13
  - adaptácia
  - level6_plus sidecars

- **post-TTS timing recovery**
  - timing audit
  - retry pre S2 pri `too_long`
  - turbo fallback preroute
  - turbo QA fallback

### 4.5 Prekladová vrstva bola opísaná architektonicky, ale nie režimovo

V dokumentácii bolo zjavné, že existuje viac engineov.

Nebolo však dostatočne explicitné, že reálne existujú aj tieto samostatné cesty:

- Google sliding window
- MultiSlav route
- MADLAD route
- TranslateGemma sliding window
- TranslateGemma full-document route
- API translate route
- Llama sliding window route
- hybrid + hybrid QA
- MADLAD → Gemma → QA batch route
- NLLB route

Podobne nebolo dostatočne explicitné, že `translation.py` neobsahuje len preklad, ale aj:

- SQL TTS normalizáciu
- chatterbox-selective SQL normalizáciu
- phonetic term protection
- SK grammar cleanup
- SQL source-guided cleanup
- Linux source-guided cleanup
- full-document refine
- flagged-segment repair cez Gemma/Ollama/OpenAI-compatible route

### 4.6 Text adaptation vrstva bola poddokumentovaná na úrovni guardov

`text_adaptation.py` je kritická vrstva, ale doteraz nebolo dostatočne explicitné, že obsahuje:

- analýzu segmentu a štýlu
- SQL-aware protected rewrite guard
- validáciu rewrite výstupu
- estimate-only fallback, keď LLM nie je dostupné
- slot-aware prompting
- batch adaptáciu s CPS kalibráciou

To je dôležité, lebo kvalita výsledku už často nezávisí od samotného prekladu, ale od toho, ako táto vrstva rozhodne medzi:

- keep
- concise rewrite
- dub-friendly rewrite
- aggressive rewrite
- fallback stretch

### 4.7 TTS vrstva bola opísaná príliš hrubo

Architektúra hovorila “Chatterbox / Turbo / S2 / XTTS / API”.

Chýbali tieto praktické runtime detaily:

- S2-Pro má vlastný local server lifecycle
  - health check
  - auto start
  - cleanup
- existuje `inspect_tts_chunk()` QA vrstva
- existuje auto voice sample extraction
- existuje auto best-sample extraction
- existuje samostatná príprava S2 referencie
- existuje OpenAI-compatible API TTS branch
- existuje ONNX branch
- existuje `piper-rs` branch
- Turbo má vlastný model cache a prompt cache

### 4.8 TTS routing a fallbacky v `pipeline.py` neboli inventarizované

Chýbajúca dokumentácia sa týkala hlavne týchto behaviorálnych vetiev:

- `multi_voice` speaker clustering
- local clone subroute pre krátke cue texty
- explicit custom WAV prioritizácia
- style-only emotion transfer pri custom WAV
- Turbo preroute na MTL fallback
- QA retry fallback po zlom Turbo výstupe
- S2 timing retry s kratším textom
- per-segment silence fallback pri zlom TTS výstupe

### 4.9 Sidecar a debug artefakty neboli rozpísané

Dokumentácia síce spomínala JSON a SRT, ale nie celú rodinu debug a eval artefaktov, ktoré pipeline reálne generuje.

Podstatné chýbajúce výstupy:

- `*_tts_debug.json`
- `*_timing_audit.json`
- `*_tts_bad_segments.json`
- `*_tts_segments.json`
- `*_tts_input.srt`
- `*_speakers.json`
- `*_mini_level11_segments.json`
- `*_pre_tts_level12_smart_timing.json`
- `*_pre_tts_level13_multipass_rewrite.json`
- `*_pre_tts_level_sidecar_summary.json`

Praktický význam:

- bez týchto artefaktov sa ťažko rekonštruuje, čo presne TTS naozaj hovoril
- “base segment text” a “effective spoken text” nemusia byť totožné

### 4.10 VRAM/RAM lifecycle nebol opísaný ako samostatná časť

V kóde je viditeľné, že runtime aktívne rieši pamäť:

- unload Ollama na začiatku a po text stacku
- free MADLAD
- free MultiSlav
- release Chatterbox model
- release Turbo model
- GPU cache cleanup po jednotlivých fázach

To doteraz nebolo pomenované ako samostatná systémová vlastnosť, hoci priamo ovplyvňuje:

- stabilitu behov
- retry logiku
- súbehy modelov
- “prečo to raz ide a raz padá na VRAM”

## 5. Potvrdené dokumentačné drift body

Nie sú to ešte implementačné slabiny. Sú to miesta, kde názov alebo opis zvádza k zlej mentálnej mape:

- `run_pipeline.py` znie širšie než čo reálne robí
- `Opraviť gramatiku` znie širšie než čo reálne spúšťa
- “TTS engine selection” v dokumentácii nepokrývalo fallback routing
- “GUI config” bolo opísané ako krátky príklad, nie ako skutočný stavový kontrakt

## 6. Čo tento audit zámerne ešte nerobí

Tento súbor zatiaľ:

- nehodnotí, či sú tieto vetvy navrhnuté dobre alebo zle
- nehodnotí krehkosť regexov
- nehodnotí naming debt
- nehodnotí duplicitné alebo mätúce zodpovednosti
- nerobí prioritizáciu bug rizík

To patrí do ďalšej fázy:

**analýza slabých miest**

## 7. Záver

Po tomto audite je potvrdené:

- architektonická dokumentácia už je použiteľná
- coverage index kódu už existuje
- ale chýbala ešte tretia vrstva:
  - **čo bolo v dokumentácii vynechané alebo príliš zovšeobecnené**

Tento gap audit túto vrstvu dopĺňa.

Ďalší správny krok je už:

1. vziať tieto potvrdené omissions
2. nehádať sa, či “to tam asi stačí”
3. prejsť na **analýzu slabých miest podľa dopadu na runtime**
