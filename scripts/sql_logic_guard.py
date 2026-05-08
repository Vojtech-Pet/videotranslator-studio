"""
sql_logic_guard.py
=================
Shared SQL terminology and logic guard for Slovak dubbing text.

Goal:
- normalize broken SQL spellings (ajznul, koalesk, ...)
- preserve SQL terms in canonical form
- distinguish ISNULL vs IS NULL vs NULLIF
- optionally prefer compact source-guided templates for common SQL tutorial lines
"""

from __future__ import annotations

import re

SQL_PROTECTED_TERMS = (
    "SQL",
    "NULL",
    "IS NULL",
    "IS NOT NULL",
    "ISNULL",
    "COALESCE",
    "NULLIF",
    "TRUE",
    "FALSE",
    "BOOLEAN",
)

SQL_PHONETIC_PATTERNS = (
    r"\bajznul\b",
    r"\bkoalesk\w*\b",
    r"\b(?:cowless|cowliss|cowlis|cowlitz|coulis|kawalis|cowlitzu)\w*\b",
    r"\bes[ií]k[jg]?[uú]el\b",
    r"\b[íi]z\s+not\s+nul\b",
    r"\bnot\s+nul\b",
    r"\bnul[ií]f\b",
)

_SQL_FIX_REPLACEMENTS = [
    (r"\bajznul\b", "ISNULL"),
    (r"\bkoalesk\w*\b", "COALESCE"),
    (r"\b(?:cowless|cowliss|cowlis|cowlitz|coulis|kawalis|cowlitzu)\w*\b", "COALESCE"),
    (r"\b[íi]z\s+not\s+null\b", "IS NOT NULL"),
    (r"\b[íi]z\s+not\s+nul\b", "IS NOT NULL"),
    (r"\biz\s+not\s+null\b", "IS NOT NULL"),
    (r"\biz\s+not\s+nul\b", "IS NOT NULL"),
    (r"\b[íi]z\s+null\b", "IS NULL"),
    (r"\b[íi]z\s+nul\b", "IS NULL"),
    (r"\bis\s+a\s+null\b", "IS NULL"),
    (r"\bis\s+a\s+nul\b", "IS NULL"),
    (r"\bis\s+null\b", "IS NULL"),
    (r"\bis\s+nul\b", "IS NULL"),
    (r"\bisnull\b", "ISNULL"),
    (r"\bcoalesce\b", "COALESCE"),
    (r"\bnull\s+if\b", "NULLIF"),
    (r"\bnullif\b", "NULLIF"),
    (r"\bisql\b", "SQL"),
    (r"\bboolean\b", "BOOLEAN"),
    (r"\btrue\b", "TRUE"),
    (r"\bfalse\b", "FALSE"),
    (r"\bsql\b", "SQL"),
    (r"\bnull\b", "NULL"),
]

_SUSPICIOUS_SQL_OUTPUT_PATTERNS = (
    r"\bajznul\b",
    r"\bkoalesk\w*\b",
    r"\b(?:cowless|cowliss|cowlis|cowlitz|coulis|kawalis|cowlitzu)\w*\b",
    r"\bes[ií]k[jg]?[uú]el\b",
    r"\bisql\b",
    r"\bnull if\b",
    r"\bfunction\b",
    r"\bdatabase\b",
    r"\bgo a\b",
    r"\bchoď(?:te)?\s+a\b",
    r"[„«]SQL[»”]",
    r"\bdva stoly\b",
    r"\bzdv[ií]hac[ií]\s+st[oô]l\b",
    r"\bsinteks\b",
    r"\bpull\b",
    r"\bnodes\b",
    r"\bz\.$",
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_punctuation(text: str) -> str:
    out = re.sub(r"\s+([,.:;!?])", r"\1", text or "")
    out = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", out)
    return normalize_spaces(out)


def fix_sql_terms(text: str) -> str:
    out = normalize_spaces(text)
    if not out:
        return out
    for pattern, replacement in _SQL_FIX_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return clean_punctuation(out)


def protected_sql_terms_in(text: str) -> set[str]:
    found: set[str] = set()
    haystack = fix_sql_terms(text)
    for term in SQL_PROTECTED_TERMS:
        pattern = re.escape(term).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", haystack, flags=re.IGNORECASE):
            found.add(term)
    return found


def source_guided_template(source_text: str) -> str | None:
    low = normalize_spaces(source_text).lower()

    if "what a null means in sql" in low and "deep dive" in low:
        return "Takto funguje hodnota NULL v SQL."
    if "special sql functions on how to handle the nulls inside our data" in low:
        return "Teraz sa pozrieme na špeciálne SQL funkcie."
    if "nulls inside our tables and we would like to go and remove it" in low:
        return "Ukážeme si, ako pracovať s NULL hodnotami v dátach."
    if "like for example 40 and in order to do that in sql we have two functions" in low:
        return "V tabuľkách sa často nachádzajú NULL hodnoty."
    if "the first one called" in low and "the second one called coalesce" in low:
        return "Tie môžeme nahradiť konkrétnou hodnotou, napríklad číslom 40."
    if "is a null and the second one called coalesce" in low:
        return "Na to slúžia funkcie ISNULL a COALESCE."
    if "what null means in sql" in low:
        return "To je to, čo znamená NULL v SQL."
    if "deep dive into special sql functions" in low:
        return "Teraz sa pozrieme na špeciálne funkcie v SQL."
    if "how to deal with nulls in our data" in low:
        return "Ako pracovať s NULL hodnotami v dátach."
    if "handle the nulls inside our data" in low and "some scenarios" in low:
        return "Ako pracovať s NULL hodnotami v dátach. Niekedy máme v tabuľkách NULL hodnoty."
    if "replace it with a new value" in low and "40" in low:
        return "Ak ju chceme nahradiť novou hodnotou, napríklad 40."
    if "do that in sql we have two functions" in low and ("called isnull" in low or "called is a null" in low):
        return "V SQL na to máme dve funkcie: ISNULL a COALESCE."
    if "in order to do that in sql we have two functions" in low and ("called isnull" in low or "called is a null" in low):
        return "V SQL na to máme dve funkcie: ISNULL a COALESCE."
    if "called coalesce" in low and "another scenario" in low:
        return "Druhá sa volá COALESCE. Teraz si ukážme ďalší scenár."
    if "called koalas" in low and "another scenario" in low:
        return "Druhá sa volá COALESCE. Teraz si ukážme ďalší scenár."
    if "doing the exact opposite" in low and "value with a null" in low:
        return "Existuje aj opačný scenár."
    if "another scenario" in low and "value 40" in low and ("make it null" in low or "make that null" in low):
        return "Máme hodnotu a chceme ju zmeniť na NULL."
    if "our table like the 40" in low and "make it as a null" in low:
        return "Máme hodnotu a chceme ju zmeniť na NULL."
    if ("replace the value with a null" in low or "replacing the value with a null" in low) and ("nullif" in low or "null if" in low):
        return "Na to použijeme funkciu NULLIF."
    if "function null if" in low and "two scenarios" in low:
        return "Na to použijeme funkciu NULLIF."
    if "through these two scenarios" in low and "null to value" in low:
        return "Takto vieme meniť NULL na hodnotu aj hodnotu na NULL."
    if "you can see with those two scenarios" in low and "from null to value" in low:
        return "Takto vieme meniť NULL na hodnotu aj hodnotu na NULL."
    if "from value to null so they are really helpful" in low:
        return "Takto vieme meniť NULL na hodnotu aj hodnotu na NULL."
    if "really useful" in low and "manipulating data" in low:
        return "Sú veľmi užitočné pri práci s údajmi v databáze."
    if "really helpful in order to manipulate the data inside our databases" in low:
        return "Sú veľmi užitočné pri práci s údajmi v databáze."
    if "inside our databases now moving on to another scenario" in low:
        return "Tieto funkcie sú veľmi užitočné pri práci s databázami."
    if ("don't want to manipulate anything" in low or "do not want to manipulate anything" in low) and "just want to check" in low:
        return "Niekedy však nechceme nič meniť, iba kontrolovať."
    if "just want to check" in low and ("replace or convert anything" in low or "replace or convert" in low):
        return "Niekedy však nechceme nič meniť, iba kontrolovať."
    if "just to check" in low and ("replace or convert anything" in low or "replace or convert" in low):
        return "Chceme zistiť, či je hodnota NULL."
    if "handle the nulls so now let's go and understand those functions one by one" in low:
        return "Teraz si tieto funkcie prejdeme jednu po druhej. Začnime prvou."
    if "between the is and null there is like space" in low:
        return "Na to použijeme podmienku IS NULL."
    if "if you apply is null" in low and ("true or false" in low or "pull in true or false" in low):
        return "Výsledok je TRUE alebo FALSE."
    if "scenario you will get true or the second option" in low:
        return "TRUE znamená, že hodnota je NULL."
    if "not null so we can use is not null" in low and "get false" in low:
        return "Ak hodnota nie je NULL, použijeme IS NOT NULL."
    if normalize_spaces(low) == "are getting":
        return "Takto vieme jednoducho kontrolovať NULL hodnoty."
    if "a boolean true or false" in low:
        return "Výsledkom je BOOLEAN hodnota TRUE alebo FALSE."
    if "big picture of all functions" in low and "nulls inside our data" in low:
        return "Takto spracujeme NULL hodnoty v SQL."
    if "the first function is null is now gonna go and replace a null" in low:
        return "Začnime funkciou ISNULL, ktorá nahradí NULL konkrétnou hodnotou."
    if "syntax of the is null is very simple" in low:
        return "Syntax funkcie ISNULL je veľmi jednoduchá."
    if "syntax of the isnull is very simple" in low:
        return "Syntax funkcie ISNULL je veľmi jednoduchá."
    if "accepts two arguments" in low and "replacement value" in low:
        return "Funkcia ISNULL prijíma dva argumenty: najprv hodnotu a potom náhradnú hodnotu."
    if "keyword isnull" in low and "accepts two arguments" in low:
        return "Použijeme funkciu ISNULL, ktorá prijíma dva argumenty."
    if "use the is null for the column called shipping address" in low:
        return "Funkciu ISNULL môžeme použiť napríklad na stĺpec s adresou doručenia."
    if "use the isnull for the column called shipping address" in low:
        return "Funkciu ISNULL môžeme použiť napríklad na stĺpec s adresou doručenia."
    if "if sql encounters any null" in low and "replace it with the" in low:
        return 'Ak SQL narazí na NULL, nahradí ju hodnotou "unknown".'
    if "default value" in low and "unknown" in low:
        return "Táto hodnota funguje ako predvolená náhrada za NULL."
    if "first value is the column and second value is like static" in low:
        return "Prvá hodnota je stĺpec a druhá je statická hodnota."
    if "don't want to have it always like the unknown" in low and "use another" in low:
        return 'V niektorých prípadoch však nechceme vždy použiť hodnotu "unknown".'
    if "column to help the first one" in low:
        return "Namiesto statickej hodnoty môžeme použiť iný stĺpec ako náhradu."
    if "whether we have a null value" in low:
        return "Takto skontrolujeme, či stĺpec obsahuje NULL hodnoty."
    if "shipping address is null" in low and "billing address" in low:
        return "Ak je hodnota NULL, použije sa hodnota z adresy fakturácie."
    if "replacing the nulls using the help of other column" in low:
        return "Takto nahrádzame NULL hodnoty pomocou iného stĺpca."
    if "replacement and if the value is not null then show the value itself" in low:
        return "Ak hodnota nie je NULL, zobrazíme ju. Pozrime sa na príklad."
    if "checking whether the value is null" in low and "get the value from the" in low:
        return "Overujeme, či je hodnota NULL. Ak áno, použijeme náhradnú hodnotu."
    if "very simple example" in low and "what we are doing we are" in low:
        return "Pozrime sa na jednoduchý príklad, aby bolo jasné, ako to funguje."
    if "have two orders the first order we are checking the shipment address" in low:
        return "Máme dve objednávky. Pri prvej kontrolujeme adresu zásielky."
    if "the value is null then we're gonna get the replacement value" in low:
        return "Ak je hodnota NULL, dostaneme náhradnú hodnotu."
    if "check the result what happens we're gonna get the addresses from the shipping address" in low:
        return "Pozrime sa na výsledok. Adresu prevezmeme z dodacej adresy, keď nie je NULL."
    if "if it's not null it's gonna return the same value" in low:
        return "Ak hodnota nie je NULL, vráti sa rovnaká hodnota."
    if "is null well no we have a value" in low and "return the same value" in low:
        return "Ak hodnota nie je NULL, SQL vráti rovnakú hodnotu."
    if "handle the nulls before doing any plus operator" in low:
        return "Pred operátorom plus musíme najprv ošetriť NULL hodnoty."
    if "again here we can go with the coulis" in low or "again here we can go with the coalesce" in low:
        return "Aj tu môžeme použiť COALESCE alebo ISNULL."
    if "create a new field using the coulis" in low or "create a new field using the coalesce" in low:
        return "Vytvorme nové pole pomocou COALESCE."
    if "define a new value if it's null" in low and "unknown" in low:
        return 'Ak je hodnota NULL, nastavíme náhradnú hodnotu, napríklad "unknown".'
    if "to the second order and here we have the shipment address as a null" in low:
        return "Pri druhej objednávke je adresa zásielky NULL. Čo sa stane v takom prípade?"
    if "value is the na" in low and "we will not get a null" in low:
        return 'Hodnota je "na", takže vo výstupe nedostaneme NULL, ale "na".'
    if "have a null we will get like default value" in low:
        return "Ak pole obsahuje NULL, dostaneme predvolenú hodnotu."
    if "default value in the output you will never get a null" in low:
        return "Ak používame predvolenú hodnotu, vo výstupe nikdy nedostaneme NULL."
    if "billing address so we have two columns" in low and "logic can be the same" in low:
        return "Máme teda dva stĺpce a logika zostáva rovnaká."
    if "first order is it null well no we have the value a" in low:
        return "Pri prvej objednávke adresa nie je NULL, takže obsahuje hodnotu A."
    if "can use more than two values with the cowless" in low:
        return "COALESCE môže pracovať s viac ako dvoma hodnotami."
    if "using the new kawalis" in low:
        return "Použijeme COALESCE. Najprv skontrolujeme prvú hodnotu."
    if "this is how the cowless works" in low:
        return "Takto funguje COALESCE."
    if "between the cowliss and is null" in low and "limited only to" in low:
        return "Rozdiel medzi COALESCE a ISNULL je v tom, že ISNULL pracuje len s dvoma hodnotami."
    if "two values where the cowliss is amazing" in low:
        return "ISNULL pracuje len s dvoma hodnotami, zatiaľ čo COALESCE vie pracovať s viacerými."
    if "in oracle they have different implementations" in low and "nvl" in low:
        return "V Oracle sa používa NVL a v iných databázach iné ekvivalenty."
    if "next use case for the cowlitz and is null" in low:
        return "Prejdime na ďalší príklad použitia COALESCE a ISNULL."
    if "replace the last name with the koalas" in low:
        return "Priezvisko nahradíme prázdnym reťazcom."
    if "make it a zero and afterward go and add a 10 points" in low:
        return "Najprv nastavíme hodnotu na nulu a potom pripočítame 10 bodov."
    if "very simple example where we have two tables and we want to combine them" in low:
        return "Máme dve tabuľky a chceme ich spojiť."
    if "use the equal operator in order to join tables" in low:
        return "Na spojenie tabuliek použijeme operátor =."
    if "type in null not an empty string" in low:
        return "V stĺpci type je NULL, nie prázdny reťazec."
    if "isql gonna use the scores in order to sort the data" in low:
        return "SQL použije skóre na zoradenie údajov."
    if "syntax of the null if it accepts" in low:
        return "Syntax funkcie NULLIF je jednoduchá."
    if "they are equal then sql gonna go and return a null" in low:
        return "Ak sa obe hodnoty rovnajú, SQL vráti NULL. Inak vráti prvú hodnotu."
    if "not null then we return a false" in low:
        return "Ak hodnota nie je NULL, vrátime FALSE."
    if "we have the full join" in low:
        return "Posledný typ je FULL JOIN."
    if "lift table gonna be the customers" in low:
        return "Keďže sa zameriavame na zákazníkov, ľavou tabuľkou budú zákazníci."
    if "what i do, i go" in low:
        return "V tomto scenári urobím toto."
    if "if you compare the results from null if and the price" in low:
        return "Ak porovnáte výsledky z NULLIF a ceny, uvidíte, že už tam nie sú chybné hodnoty."
    if "if we go and use the is not null" in low and "exact opposite" in low:
        return "Keď použijeme IS NOT NULL, bude to presne naopak."
    if "whether it's equal to minus one" in low and "price is equal to minus one" in low:
        return "Ak sa cena rovná mínus jedna, spracujeme ju ako špeciálny prípad."
    if normalize_spaces(low) == "we have orders.":
        return "Máme objednávky."
    if "so let's have an example" in low:
        return "Príklad."
    return None


def enforce_sql_logic(text: str, *, source_text: str = "", fallback_text: str = "") -> str:
    out = text or ""

    replacements = [
        (r"Syntax funkcie IS NULL", "Syntax funkcie ISNULL"),
        (r"Použijeme (?:kľúčové slovo |funkciu )IS NULL", "Použijeme funkciu ISNULL"),
        (r"IS NULL, ktorá prijíma dva argumenty", "ISNULL, ktorá prijíma dva argumenty"),
        (r"funkci[ao]u NULL\b", "funkciou NULLIF"),
        (r"\bSQL NULL\b", "NULLIF"),
        (r"\bprázdna hodnota\b", "NULL hodnota"),
        (r"\bprázdne hodnoty\b", "NULL hodnoty"),
    ]
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    for term in SQL_PROTECTED_TERMS:
        out = re.sub(rf"\b{re.escape(term)}\b", term, out, flags=re.IGNORECASE)

    low_src = normalize_spaces(source_text).lower()
    low_fallback = normalize_spaces(fallback_text).lower()
    joined = f"{low_src} {low_fallback}".strip()

    if "true or false" in joined or "pull in true or false" in joined or "boolean" in joined:
        out = re.sub(r"\bISNULL\b", "IS NULL", out, flags=re.IGNORECASE)

    if "two arguments" in joined or "replacement value" in joined or "default value" in joined:
        out = re.sub(r"Syntax funkcie IS NULL", "Syntax funkcie ISNULL", out, flags=re.IGNORECASE)
        out = re.sub(r"funkci[ao]u IS NULL", "funkciou ISNULL", out, flags=re.IGNORECASE)
        out = re.sub(r"\bIS NULL, ktorá prijíma dva argumenty", "ISNULL, ktorá prijíma dva argumenty", out, flags=re.IGNORECASE)

    if "nullif" in joined:
        out = re.sub(r"\bfunkci[ao]u\s+NULL\b", "funkciou NULLIF", out, flags=re.IGNORECASE)
        out = re.sub(r"\bfunkcia\s+NULL\b", "funkcia NULLIF", out, flags=re.IGNORECASE)
        out = re.sub(r"\bpoužijeme\s+NULL\b", "použijeme NULLIF", out, flags=re.IGNORECASE)
        out = re.sub(r"\bna to máme\s+NULL\b", "na to máme NULLIF", out, flags=re.IGNORECASE)
        out = re.sub(r"\bfunkcia ISNULL\b", "funkcia NULLIF", out, flags=re.IGNORECASE)

    if "coalesce" in joined:
        out = re.sub(r"\bkoalesk\b", "COALESCE", out, flags=re.IGNORECASE)

    if "isnull" in joined and "coalesce" in joined and "two functions" in joined:
        out = "V SQL na to máme dve funkcie: ISNULL a COALESCE."

    if "between the is and null there is like space" in joined:
        out = "Na to máme podmienku IS NULL, medzi IS a NULL je medzera."

    return clean_punctuation(out)


def cleanup_sql_tutorial_phrasing(text: str) -> str:
    out = text or ""

    replacements = [
        (r"[„“«»]SQL[„“«»]", "SQL"),
        (r"\bNULL\s+if\b", "NULLIF"),
        (r"\bISQL\b", "SQL"),
        (r"\bchoď(?:te)?\s+a(?:\s+|$)", ""),
        (r"\bpoďme\s+a\s+", "poďme "),
        (r"\bv poriadku,\s*priatelia,?\s*", ""),
        (r"\bvšetko v poriadku,\s*", ""),
        (r"\bTak poďme urobme to\b\.?", "Poďme to urobiť."),
        (r"\bponor[ií]me\s+sa(?:\s+do\s+hĺbky|\s+do)?", "pozrime sa"),
        (r"\bhodnota\s+nie\s+ISNULL\b", "hodnota nie je NULL"),
        (r"\bthe\s+ISNULL\b", "ISNULL"),
        (r"\bsplynutie\s+splynutia\b", "COALESCE"),
        (r"\bdostávajú\s+to\b", "výsledkom je"),
        (r"\bnulová\s+dodacia\s+adresa\b", "dodacia adresa je NULL"),
        (r"\bhodnota\s+c\s+vo\s+výstupe\b", "vo výstupe dostaneme hodnotu C"),
        (r"\bnuly\b", "NULL hodnoty"),
        (r"\bnejaké?\s+nuly\b", "NULL hodnoty"),
        (r"\bak\s+tam\s+sú\s+nuly\b", "ak tam sú NULL hodnoty"),
        (r"\bvidieť\s+nuly\b", "vidieť NULL hodnoty"),
        (r"\bnemal\s+žiadne\s+nuly\b", "nemal žiadne NULL hodnoty"),
        (r"\bspracovanie\s+nuly\b", "spracovanie NULL hodnôt"),
        (r"\bspracovať\s+nuly\b", "ošetriť NULL hodnoty"),
        (r"\bdva stoly\b", "dve tabuľky"),
        (r"\bprvý stôl\b", "prvá tabuľka"),
        (r"\bdruhý stôl\b", "druhá tabuľka"),
        (r"\bzdv[ií]hac[ií]\s+st[oô]l\b", "ľavá tabuľka"),
        (r"\búpln[ée]\s+pripojenie\b", "FULL JOIN"),
    ]

    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    return clean_punctuation(out)


def _should_prefer_source_template(current_text: str, template: str) -> bool:
    current_terms = protected_sql_terms_in(current_text)
    template_terms = protected_sql_terms_in(template)
    if template_terms and not template_terms.issubset(current_terms):
        return True
    if any(re.search(pattern, current_text, flags=re.IGNORECASE) for pattern in _SUSPICIOUS_SQL_OUTPUT_PATTERNS):
        return True
    return False


def guard_sql_text(text: str, *, source_text: str = "", fallback_text: str = "") -> str:
    out = fix_sql_terms(text or "")
    out = enforce_sql_logic(out, source_text=source_text, fallback_text=fallback_text)
    out = cleanup_sql_tutorial_phrasing(out)

    template = source_guided_template(source_text)
    if template and _should_prefer_source_template(out, template):
        out = fix_sql_terms(template)
        out = enforce_sql_logic(out, source_text=source_text, fallback_text=fallback_text)
        out = cleanup_sql_tutorial_phrasing(out)

    return clean_punctuation(out)
