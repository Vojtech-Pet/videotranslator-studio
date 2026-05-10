# VideoTranslator_studio — Analýza Slabých Miest
> Vygenerované: 2026-04-10 | Typ: prioritizovaný risk audit podľa dopadu na runtime

## Status update — 2026-04-10 večer

Od posledného auditu už boli implementované tieto mitigácie:

- plaintext API kľúče sa už neukladajú do configov v Qt ani legacy Tk GUI
- API prekladový model je oddelený od TTS modelu
- OpenAI API translate default je nastavený na `gpt-5.2-mini`
- `reuse_tts` už nepoužíva len existenciu WAV, ale semantic hash cache

To znamená, že položky **K1**, **K2** a časť **V1** už nie sú v pôvodnom kritickom stave. Audit nižšie je stále užitočný ako mapa rizík, ale tieto body treba čítať ako už rozpracované a čiastočne uzavreté.

## 1. Účel

Tento súbor už nehovorí o tom, čo v dokumentácii chýbalo.

Hovorí o tom:

- čo je dnes najslabšie miesto systému
- čo má najväčší praktický dopad
- čo najviac vysvetľuje nečakané správanie
- čo treba riešiť skôr než ďalšie jemné tuningy

## 2. Ako čítať tento audit

Priorita je podľa dopadu na:

1. bezpečnosť
2. správnosť runtime správania
3. kvalitu prekladu a TTS
4. debuggability
5. prenositeľnosť a údržbu

## 3. Kritické slabé miesta

### K1. API kľúče sa ukladajú v plain texte

Stav:

- GUI načíta a uloží všetky API kľúče priamo do `musetalk_gui_config.json`
- `AppSettings.save()` zapisuje celý objekt bez filtrovania citlivých polí
- v aktuálnom konfiguráku je prítomný reálny OpenAI API kľúč

Dopad:

- bezpečnostné riziko
- riziko náhodného commitnutia alebo zdieľania
- riziko úniku pri debug exporte alebo zálohe domovského adresára

Prečo je to kritické:

- toto nie je teoretická slabina, ale aktívny stav
- neohrozuje len kvalitu prekladu, ale priamo účet a API budget

Odporúčanie:

- okamžite presunúť kľúče do env premenných alebo OS keyring
- v GUI ukladať len label/provider, nie tajomstvo
- existujúce kľúče rotovať

### K2. ChatGPT/OpenAI model routing je mätúci a splitnutý medzi tri vrstvy

Stav:

- GUI pole pre **prekladový model** je uložené do `openai_tts_model`
- názov poľa teda hovorí TTS, ale runtime ho používa pre translate route
- parser má iný default pre `--api_translate_model`
- aktuálne uložený GUI stav stále ukazuje `gpt-4.1-mini`

Konkrétny problém:

Používateľ môže veriť, že používa napríklad GPT-5.2 mini, ale systém reálne beží na inom modeli, lebo:

- GUI názov poľa je mätúci
- uložený stav môže byť starý
- CLI default je iný než GUI default

Praktický dopad:

- ťažko sa vysvetľuje, prečo “ChatGPT web je lepší než API”
- ťažko sa spätne zistí, aký model vlastne preložil video
- dokumentácia a runtime sa rozchádzajú

Poznámka k aktuálnemu stavu:

K 2026-04-10 je v uloženom GUI configu:

- `trans_engine = chatgpt`
- `openai_tts_model = gpt-4.1-mini`

To znamená, že ak bol zámer používať GPT-5.2 mini, uložený stav mu dnes nezodpovedá.

Odporúčanie:

- rozdeliť config polia na:
  - `openai_translate_model`
  - `openai_tts_model`
- zapísať do JSON aj effective translate route
- zosúladiť GUI default, CLI default a uložený stav

### K3. GUI defaulty a CLI defaulty nie sú jednotné

Stav:

- `AppSettings.adapt_provider` defaultuje na `ollama`
- parser `--adapt_provider` defaultuje na `llama_cpp`

To znamená:

- rovnaký projekt sa môže správať inak podľa toho, či ide cez GUI alebo priamo cez CLI
- dokumentácia môže hovoriť pravdu len pre jednu cestu

Dopad:

- “u mňa to ide / u mňa to nejde” bugy
- ťažké porovnanie behov
- ťažké reprodukcie bez presného command line dumpu

Odporúčanie:

- mať jeden canonical default
- druhú vrstvu nechať len ako explicitný override

## 4. Vysoké slabé miesta

### V1. `reuse_tts` vie potichu recyklovať nesprávne audio

Stav:

- chunk sa znovu použije len podľa toho, že existuje a nie je prázdny
- nekontroluje sa:
  - text
  - TTS engine
  - voice routing
  - model
  - timing config

Dopad:

- systém môže hovoriť starý text
- používateľ si myslí, že sa použila nová verzia segmentu, ale počuje starý WAV
- vznikajú “nevysvetliteľné” regresie

Odporúčanie:

- viazať reuse na hash textu + engine + model + voice identity + timing mode
- alebo mať `reuse_tts` defaultne vypnuté pri zmene JSON/tts_input

### V2. `use_existing_segments` je viazané na naming convention, nie na explicitný vstup

Stav:

- `--use_existing_segments` nenačíta používateľom zadaný JSON path
- hľadá automaticky `{stem}_{tgt}_segments.json` v `out_dir`

Dopad:

- ľahko sa načíta starý alebo nesprávny súbor
- debug môže vyzerať ako chyba v TTS, hoci je to chyba vo výbere vstupu

Odporúčanie:

- pridať explicitný argument typu `--segments_json`
- `use_existing_segments` nechať len ako convenience skratku

### V3. Hlavná pipeline je príliš monolitická a vetvená

Stav:

- `scripts/pipeline.py` obsahuje veľmi veľa behaviorálnych vetiev v jednom súbore
- translation routing, pre-TTS stack, TTS fallbacky, timing retry, assembly a metrics sú spojené v jednej hlavnej procedúre

Dopad:

- vysoké riziko regresií
- ťažké izolovať zmeny
- rovnaké post-fixy sa musia strážiť na viacerých miestach

Odporúčanie:

- oddeliť:
  - translation stage orchestration
  - pre-TTS text stack
  - TTS routing
  - post-TTS QA/timing retry
  - assembly stage

### V4. Systém je príliš závislý na heuristikách a regexoch

Stav:

- Turbo fallback route je založený na heuristických regex rozhodnutiach
- style-only emotion transfer je heuristický
- SQL/Linux cleanup je rule-based
- timing adaptation validation je guard-based

Dopad:

- systém vie byť dobrý na známych prípadoch, ale krehký na hraničných
- oprava jedného edge-case vie rozbiť iný

Odporúčanie:

- per-segment route audit
- benchmark set pre regresie
- menej “implicitných čarov”, viac explicitných rozhodovacích záznamov

### V5. Prenositeľnosť projektu je slabá kvôli hardcoded cestám

Stav:

- v runtime sú stále natvrdo zakódované cesty pod `/home/vojtech/...`
- týka sa to modelov, env pythonov, cache ciest aj tréningových skriptov

Dopad:

- krehké pri presune na iný stroj
- ťažké zdieľať projekt bez lokálnych ručných zásahov
- ťažké robiť čisté reprodukcie

Odporúčanie:

- zaviesť central model registry / path resolver
- všetky absolútne cesty presunúť do config vrstvy alebo env

## 5. Stredné slabé miesta

### S1. Chýba canonical route audit na úrovni segmentu

Aj keď vzniká veľa sidecar JSONov, stále chýba jeden kompaktný záznam typu:

- ktorý translation engine spracoval segment
- či prešiel refine
- či prešiel audit repair
- či prešiel mini_level11 / level12 / level13
- ktorý TTS engine ho nakoniec vyrenderoval
- či bol timing retry
- či bol fallback

To by zásadne zlepšilo debug.

### S2. `translation_only` názvom zvádza k inému správaniu, než reálne robí

Stav:

- nekončí po čistom preklade
- ide cez celý pre-TTS text stack

To je možno rozumné, ale názov to nehovorí.

### S3. `run_pipeline.py` názvom zvádza k širšiemu významu

Názov naznačuje hlavný orchestrátor, ale reálne ide o Level 1–4 sidecar vetvu.

### S4. Test coverage je nízka vzhľadom na počet behaviorálnych vetiev

Projekt má veľa smoke výstupov a JSON sidecarov, ale chýba:

- regresný set pre translation routing
- regresný set pre voice routing
- regresný set pre `reuse_tts`
- regresný set pre `use_existing_segments`
- regresný set pre sync-safe assembly

## 6. Čo dnes najviac vysvetľuje používateľské zmätky

Ak by som mal vybrať iba 5 vecí, ktoré najviac vysvetľujú “prečo sa to správa divne”, sú to:

1. plaintext config + persisted state
2. ChatGPT/OpenAI model naming drift
3. split-brain GUI vs CLI defaulty
4. stale reuse cez `reuse_tts`
5. implicitné fallbacky v pipeline bez jednej route audit stopy

## 7. Čo riešiť ako prvé

Odporúčané poradie:

1. bezpečnosť kľúčov
2. rozdelenie translate/TTS model config polí pre API providery
3. zjednotenie GUI/CLI defaultov
4. hash-based `reuse_tts`
5. explicitný `--segments_json`
6. route audit JSON pre každý segment
7. až potom hlbšia refaktorizácia pipeline

## 8. Čo z toho je hneď akčné

Bez veľkého refaktoru sa dá hneď spraviť:

- rotácia a odstránenie plaintext API kľúčov
- premenovanie config polí pre OpenAI/Grok/Gemini translate modely
- záznam effective modelu do segment JSON
- ochrana `reuse_tts` hashom
- explicitný argument pre externý segments JSON

To by dalo veľký praktický zisk ešte pred “veľkou architektúrou”.
