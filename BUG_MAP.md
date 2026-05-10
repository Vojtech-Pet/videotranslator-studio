# VideoTranslator_studio - Bug Map

Generated: 2026-04-10
Status: current after fixes 1-26 and runtime changes from 2026-04-10

This file is the shortest path to understanding:
- what runs where
- which files own which behavior
- what must stay true
- where bugs usually come from
- what to inspect first

For the full long-form description, read `ANALYZA_SYSTEMU.md`.
For the visual map, open `ARCHITECTURE_CURRENT.puml`.

## 1. Current source of truth

Use these files in this order:
1. `ANALYZA_SYSTEMU.md`
2. `ANALYZA_SLABYCH_MIEST.md`
3. `ANALYZA_GAP_AUDIT.md`
4. `MAPA_KODU.md`
5. `BUG_MAP.md`
6. `ARCHITECTURE_CURRENT.puml`
7. `CHANGELOG_OPRAV.md`

Older level reports in `temp/` and `output/` are useful as evidence, not as current architecture truth.

## 2. Critical runtime path

The main execution path is:

1. `VideoTranslator_studio.py`
2. `main.py`
3. `scripts/pipeline.py`
4. stage modules:
   - `scripts/stt.py`
   - `scripts/translation.py`
   - `scripts/text_adaptation.py`
   - `scripts/tts.py`
   - `scripts/audio_pipeline.py`

If a run "looks wrong", it is almost always one of these four classes:
- translation bug
- timing/adaptation bug
- voice routing bug
- final assembly/sync bug

## 3. Ownership map

| Area | Main file | Why it matters |
|---|---|---|
| GUI state + CLI args | `VideoTranslator_studio.py` | User choices become pipeline flags here |
| Runtime orchestration | `scripts/pipeline.py` | Real routing logic lives here |
| Translation prompts + engines | `scripts/translation.py` | Meaning, context and term preservation start here |
| TTS synthesis | `scripts/tts.py` | Voice identity and generation params live here |
| Final sync + assembly | `scripts/audio_pipeline.py` | Audio/video sync can be fixed or broken here |
| Persistent GUI defaults | `musetalk_gui_config.json` | Old bad settings can revive old bugs |

## 4. Current invariants

These are the rules that should stay true. If one breaks, the output drifts fast.

### 4.1 Translation invariants

- Translation stage must preserve meaning first.
- Translation stage must not optimize for slot timing.
- Timing/CPS pressure belongs to `scripts/text_adaptation.py`, not to translation prompts.
- `text_src` is the source text.
- `text` is the translated text currently being refined.
- API translation must go through sliding-window batching with previous translated context.
- Translation cache must be scoped by engine + provider + model + content, not only text.

### 4.2 Voice invariants

- `multi_voice` has highest voice priority.
- Explicit custom WAV has higher priority than original-segment emotion clone.
- Custom WAV + emotion must keep the custom voice identity.
- Custom WAV + emotion now means style-only transfer, not full source voice cloning.
- `multi_voice` and explicit custom WAV should not be active together for Chatterbox.

### 4.3 Sync invariants

- Original timeline gaps are preserved by default.
- Gap compression is optional, not default.
- If audio drifts from video, first suspect gap compression or aggressive stretch.
- `def_pause_gap_s` and assembly mode have direct sync impact.

## 5. Highest-risk bug hotspots

### Hotspot A - Translation prompt pressure

Symptoms:
- weird short translations
- over-compressed wording
- lost nuance
- API output feels worse than ChatGPT web UI

Where:
- `scripts/translation.py`

What used to be wrong:
- prompt pressure existed too early
- rules like "fit duration", "similar sentence length", "keep concise for timing" distorted translation quality

Current rule:
- translation prompts are relaxed
- timing happens later

First checks:
- confirm `use_api_translate` route is active
- confirm sliding-window context is used
- confirm no old prompt override reintroduced hard timing constraints

### Hotspot B - Cache contamination

Symptoms:
- same segment behaves differently after model/provider change
- "old translation smell" after prompt updates
- impossible-to-explain stale outputs

Where:
- `scripts/translation.py`

Current fix:
- cache key includes engine, src lang, tgt lang, provider, model, content and glossary

Risk:
- any future cache simplification can silently reintroduce mixed results

### Hotspot C - Voice mixing

Symptoms:
- dubbed output sounds like two speakers blended
- custom WAV voice suddenly sounds like original speaker
- short segments use one voice, longer segments another

Where:
- `VideoTranslator_studio.py`
- `scripts/pipeline.py`
- `scripts/tts.py`
- `musetalk_gui_config.json`

Past root cause:
- explicit custom WAV + `emotion_clone`
- explicit custom WAV + `multi_voice`
- stale config flags re-enabled bad combinations

Current rule:
- custom WAV + emotion uses style-only transfer
- custom WAV + multi_voice is blocked in GUI/CLI routing

First checks:
- `cb_voice`
- `cb_emo_clone`
- `chk_multi_voice`
- final built CLI args

### Hotspot D - A/V sync drift

Symptoms:
- dubbed speech starts too early
- long silent areas collapse
- video mouth movement no longer matches pauses

Where:
- `scripts/pipeline.py`
- `scripts/audio_pipeline.py`
- `musetalk_gui_config.json`

Past root cause:
- timeline gaps were being compressed

Current rule:
- preserve original gaps by default
- compress only with explicit `--compress_timeline_gaps`

First checks:
- `def_pause_gap_s`
- `compress_timeline_gaps`
- final assembly path

### Hotspot E - Style-only emotion transfer limits

Symptoms:
- custom WAV keeps identity but emotion feels weaker than expected
- expressive source segment produces only mild style change

Where:
- `scripts/pipeline.py`

Current implementation:
- derive style from source segment energy/activity/dynamics
- map it into `exaggeration`, `cfg_weight`, `temperature`

Important:
- this is heuristic style transfer
- it is not a full prosody/emotion disentanglement model

### Hotspot F - "Opraviť gramatiku" does not mean all grammar fixing

Symptoms:
- user turns off `Opraviť gramatiku` and assumes no language cleanup runs
- user turns on `Opraviť gramatiku` for ChatGPT/Grok/Gemini and expects Gemma post-refine
- runtime behavior feels different from the UI label

Where:
- `VideoTranslator_studio.py`
- `scripts/pipeline.py`

Current rule:
- the checkbox controls only the full-document LLM refine pass
- deterministic Slovak fixes still run even when the checkbox is off
- SK audit still runs after JSON export
- selective flagged-segment repair may still run if a repair backend is available
- `hybrid` enables refine on its own
- `chatgpt`, `grok`, `gemini` are currently excluded from the generic checkbox-driven refine append

Fast mental model:
- checkbox OFF = no full Gemma/Ollama post-refine
- checkbox OFF != no correction at all

Debug implication:
- if output changed even with checkbox OFF, check deterministic SK fixes and audit repair before assuming a hidden Gemma pass

## 6. Fast diagnosis checklist

### If translation is bad

Check:
1. Which translation engine actually ran
2. Whether API translation used sliding-window route
3. Whether a custom prompt override reintroduced timing pressure
4. Whether old cached translations are still being read

### If voices are mixed

Check:
1. `cb_voice`
2. `cb_emo_clone`
3. `chk_multi_voice`
4. whether the selected TTS engine is Chatterbox or Turbo fallback
5. logs around voice routing in `pipeline.py`

### If sync is wrong

Check:
1. whether original gaps were preserved
2. whether `compress_timeline_gaps` was accidentally enabled
3. whether heavy stretch happened on short segments
4. whether the text adaptation made output much longer than slot

### If config "resurrects" old problems

Check:
1. `musetalk_gui_config.json`
2. saved GUI defaults after close/reopen
3. whether UI disable state also persists to config

### If grammar behavior feels inconsistent

Check:
1. which translation engine actually ran
2. whether `hybrid` was used
3. whether `chk_refine` only looked enabled in GUI or actually appended `--refine_translation`
4. whether deterministic SK fixes changed the text
5. whether SK audit + flagged repair rewrote some segments after JSON export

## 7. Current behavior that is easy to misunderstand

### "Emotion clone" does not mean pure emotion extraction

There are now two meanings depending on chosen voice:

- default/original voice:
  full source-segment prompt may be used

- custom WAV:
  no source voice cloning
  only speaking style is estimated from the source segment

This distinction is important. Without it, the name sounds more magical than the implementation really is.

### "Opraviť gramatiku" is a UI shortcut, not a full truth label

In current behavior, the label hides three separate mechanisms:

- deterministic Slovak cleanup
- optional full-document LLM refine
- audit-driven selective repair

If this is forgotten, debugging quickly turns into guesswork.

### "Predvoleny" is not the same as "Original video"

- `Predvoleny` means model/default reference path
- `Original video` means source-speaker driven path
- custom WAV means explicit user voice identity

## 8. Real constraints

These still limit quality even after the recent fixes:

- no full disentangled prosody model
- no pyannote-grade speaker diarization yet
- no native real-time preview
- no complete unit test coverage
- translation stack is still spread across several branches in `translation.py`

## 9. What to improve next

Best next improvements in order:

1. Add a GUI debug panel showing final voice routing per run
2. Add translation route audit block into segment JSON
3. Add a stronger prosody/style model for custom WAV emotion transfer
4. Split translation prompt builder into one canonical function reused by all engines
5. Add a sync regression smoke test with preserved vs compressed gaps

## 10. Practical "do not break this" list

- Do not reintroduce timing pressure into translation prompts.
- Do not let custom WAV and multi-voice run together for Chatterbox.
- Do not make gap compression the default again.
- Do not collapse translation cache scope back to text-only.
- Do not treat old `temp/` reports as architecture truth.
