"""
eval_text_adaptation.py
========================
Eval set pre text adaptation layer — 25 viet s overflow scenármi.

Pokrýva:
  - dlhé súvetia (clause overflow)
  - technické termíny + skratky
  - čísla + percentá
  - krátky slot (<2s)
  - veľký overflow (>2×)
  - OK segmenty (kontrola že sa nič nezmení)

Použitie:
    python eval_text_adaptation.py
    python eval_text_adaptation.py --llm models/gemma.gguf
    python eval_text_adaptation.py --llm models/gemma.gguf --verbose
"""

import argparse
from text_adaptation import adapt_for_timing, estimate_tts_duration, AdaptStats

# ── Eval set ─────────────────────────────────────────────────────────────────
# (src_en, translated_sk, slot_s, expected_mode_hint)

EVAL_SET = [
    # ── OK — nemá sa rewritovať ───────────────────────────────────────────
    ("Hello.",
     "Ahoj.",
     2.0, "ok_no_change"),

    ("Thank you.",
     "Ďakujem.",
     2.0, "ok_no_change"),

    ("The project is done.",
     "Projekt je hotový.",
     3.0, "ok_no_change"),

    ("This is a simple sentence.",
     "Toto je jednoduchá veta.",
     3.5, "ok_no_change"),

    # ── Mierne pretečenie — concise rewrite ──────────────────────────────
    ("This configuration allows the system to process requests more efficiently.",
     "Táto konfigurácia umožňuje systému spracovávať požiadavky efektívnejšie.",
     3.5, "concise_rewrite"),

    ("The meeting has been rescheduled to next Monday at ten in the morning.",
     "Stretnutie bolo presunuté na budúci pondelok o desiatej ráno.",
     3.0, "concise_rewrite"),

    ("We need to make sure that all the tests are passing before we deploy to production.",
     "Musíme sa uistiť, že všetky testy prechádzajú pred nasadením do produkcie.",
     4.0, "concise_rewrite"),

    ("The application uses a REST API to communicate with the backend server.",
     "Aplikácia používa REST API na komunikáciu s backendovým serverom.",
     3.5, "concise_rewrite"),

    # ── Veľké pretečenie — dub-friendly rewrite ───────────────────────────
    ("This configuration allows the system to process requests more efficiently and faster than before.",
     "Táto konfigurácia umožňuje systému spracovávať požiadavky efektívnejšie a rýchlejšie ako pred tým.",
     3.5, "dub_friendly_rewrite"),

    ("The new machine learning model was trained on a dataset containing over three hundred thousand samples.",
     "Nový model strojového učenia bol natrénovaný na datasete obsahujúcom vyše tristo tisíc vzoriek.",
     4.0, "dub_friendly_rewrite"),

    ("In this episode we will talk about the importance of continuous integration and continuous deployment in modern software development.",
     "V tejto epizóde si povieme o dôležitosti priebežnej integrácie a priebežného nasadenia v modernom vývoji softvéru.",
     5.0, "dub_friendly_rewrite"),

    # ── Technické termíny + skratky ───────────────────────────────────────
    ("Configure the HTTP headers, SSL certificate, and DNS records before deployment.",
     "Nakonfigurujte HTTP hlavičky, SSL certifikát a DNS záznamy pred nasadením.",
     3.5, "concise_rewrite"),

    ("The CI/CD pipeline automatically runs unit tests, builds the Docker image, and deploys to Kubernetes.",
     "CI/CD pipeline automaticky spúšťa unit testy, zostavuje Docker obraz a nasadzuje do Kubernetes.",
     5.0, "concise_rewrite"),

    # ── Čísla a percentá ──────────────────────────────────────────────────
    ("The model achieved an accuracy of ninety-four point seven percent on the test set.",
     "Model dosiahol presnosť deväťdesiatštyri celých sedem percent na testovacej sade.",
     4.0, "concise_rewrite"),

    ("In the year twenty twenty-six, artificial intelligence became part of everyday life.",
     "V roku dvetisícšesťadvadsaťšesť sa umelá inteligencia stala súčasťou každodenného života.",
     4.0, "dub_friendly_rewrite"),

    # ── Krátky slot (<2s) — extrémny tlak ────────────────────────────────
    ("This is a technical explanation.",
     "Toto je technické vysvetlenie, ktoré popisuje fungovanie systému.",
     1.5, "dub_friendly_rewrite"),

    ("The process failed.",
     "Proces zlyhal a musíme ho reštartovať.",
     1.5, "concise_rewrite"),

    # ── Veľký overflow (>2×) ──────────────────────────────────────────────
    ("Right.",
     "Áno, presne tak, máš úplnú pravdu, to je správna cesta ako to riešiť.",
     1.0, "dub_friendly_rewrite"),

    ("Exactly.",
     "Presne tak, to je ten správny prístup k tejto problematike, s ktorým plne súhlasím.",
     1.0, "dub_friendly_rewrite"),

    # ── Dlhé súvetia ─────────────────────────────────────────────────────
    ("When I woke up in the morning, it was raining outside, but by noon the weather improved and the sun came out.",
     "Keď som ráno vstal, vonku pršalo, ale do poludnia sa počasie zlepšilo a vyšlo slnko.",
     4.5, "ok_no_change"),

    ("The team completed the project on time, all customer requirements were met, and everyone did an excellent job.",
     "Tím dokončil projekt včas, všetky požiadavky zákazníka boli splnené a každý odviedol skvelú prácu.",
     5.5, "ok_no_change"),

    # ── IT podcast štýl (miso-like) ───────────────────────────────────────
    ("In today's episode we'll talk about how to get started with Python and what makes it such a popular language.",
     "V dnešnej epizóde si povieme o tom, ako začať s Pajtonom a čo ho robí tak populárnym jazykom.",
     5.5, "ok_no_change"),

    ("If you want to build a career in software testing, you should start by learning the basics of manual testing.",
     "Ak chceš budovať kariéru v testovaní softvéru, mal by si začať s učením základov manuálneho testovania.",
     6.0, "ok_no_change"),

    # ── SQL technické ─────────────────────────────────────────────────────
    ("When the value is NULL, the COALESCE function returns the first non-null expression.",
     "Keď je hodnota nul, funkcia koalesk vrátí prvý výraz ktorý nie je nul.",
     4.0, "ok_no_change"),

    ("The LEFT JOIN returns all rows from the left table even when there is no match.",
     "LEFT JOIN vráti všetky riadky z ľavej tabuľky aj keď neexistuje zhoda.",
     4.0, "ok_no_change"),
]


def run_eval(llm=None, verbose=False):
    stats = AdaptStats()
    results = []

    print(f"\n{'─'*80}")
    print(f"{'ID':<3} {'slot':>5} {'ratio_b':>7} {'ratio_a':>7}  {'mode':<22} {'hint':<22} {'match'}")
    print(f"{'─'*80}")

    for i, (src, sk, slot, hint) in enumerate(EVAL_SET, 1):
        result = adapt_for_timing(
            translated=sk, src_text=src,
            slot_duration=slot, llm=llm, verbose=verbose,
        )
        stats.record(result)

        match = "✓" if result.mode == hint else ("~" if hint in result.mode or result.mode in hint else "✗")
        print(
            f"{i:<3} {slot:>4.1f}s  "
            f"{result.ratio_before:>6.2f}×  {result.ratio:>6.2f}×  "
            f"{result.mode:<22}  {hint:<22}  {match}"
        )
        if verbose and result.mode != "ok_no_change":
            print(f"     ORIG: {sk[:70]}")
            print(f"     OUT:  {result.text[:70]}")

        results.append({
            "id": i, "src": src, "sk_orig": sk, "sk_out": result.text,
            "slot": slot, "ratio_before": round(result.ratio_before, 3),
            "ratio_after": round(result.ratio, 3),
            "mode": result.mode, "hint": hint,
            "match": match, "notes": result.notes,
        })

    print(f"{'─'*80}")
    print(stats.summary())

    ok_count    = sum(1 for r in results if r["match"] == "✓")
    approx_count = sum(1 for r in results if r["match"] == "~")
    fail_count  = sum(1 for r in results if r["match"] == "✗")
    print(f"\n✓ presné zhody: {ok_count}/{len(results)}  "
          f"~ čiastočné: {approx_count}  ✗ nezhody: {fail_count}")
    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",     default="", help="GGUF model path pre rewrite")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    llm_instance = None
    if args.llm:
        from llama_cpp import Llama
        llm_instance = Llama(model_path=args.llm, n_ctx=2048, n_gpu_layers=56, verbose=False)

    run_eval(llm=llm_instance, verbose=args.verbose)
