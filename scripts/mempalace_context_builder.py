#!/usr/bin/env python3
"""
mempalace_context_builder.py
============================
Generátor domain_context.json z MemPalace vedomostnej bázy.

DÔLEŽITÉ: Tento skript NESPÚŠŤAJ priamo — MemPalace je MCP server,
dostupný cez MCP klientov ako Claude Code alebo Gemini CLI.

Postup regenerácie domain_context.json:
1. Otvor Claude Code alebo Gemini CLI v tomto projekte
2. Povedz: "Zregeneruj domain_context.json z MemPalace"
3. AI agent vykoná query do MemPalace a prepíše domain_context.json

Čo Claude dotazuje:
  - wing=videotranslator_studio, room=fixes  → SQL pravidlá, opravy
  - wing=videotranslator_studio, room=bugs   → známe problémy
  - wing=videotranslator_studio, room=tts_quality → TTS vzory

Výstup: scripts/domain_context.json
Používa: scripts/gemma4_hf_worker.py (--extra_context)
         scripts/fix_short_translations.py (automaticky)
         scripts/translation.py (translate_fulldoc_with_gemma4_hf)

Architektúra:
  Claude (MCP) → MemPalace query → domain_context.json
                                          ↓
  pipeline.py → translation.py → gemma4_hf_worker.py (subprocess)
                                          ↓ --extra_context
                                   Gemma 4 dostane SQL pravidlá
                                   + translation rules v system prompte
"""

# Tento súbor slúži ako dokumentácia workflow.
# Skutočná logika buildu beží cez Claude + MCP.

DOMAIN_CONTEXT_SCHEMA = {
    "_meta": "verzia, zdroj, dátum generovania",
    "sql_terms_preserve": "zoznam SQL termínov ktoré sa nesmú preložiť",
    "sql_term_rules": "dict: term → presné pravidlo použitia v SK dabingu",
    "phonetic_avoid": "fonetické prepisy ktoré Gemma 4 nesmie generovať",
    "cps_calibration": "kalibrovaná rýchlosť hlasu Chatterbox SK (chars/s)",
    "translation_rules": "zoznam pravidiel prekladu pre voiceover",
    "domain_context": "popis témy videa pre lepší kontext LLM",
    "known_issues": "aktuálne známe problémy pipeline",
}

if __name__ == "__main__":
    print(__doc__)
