"""
term_protection.py
==================
Ochrana výrazov pred prekladom.

Maskuje výrazy ktoré sa nemajú prekladať (git príkazy, technické termíny,
SQL kľúčové slová, značky, slangy). Pred prekladom ich zamaskuje tokenmi,
po preklade vráti späť.

Princíp:
  'Do a git commit and push to GitHub'
          → mask()
  'Do a git __TERM_0__ and __TERM_1__ to __TERM_2__'
          → prekladač
  'Urob git __TERM_0__ a __TERM_1__ na __TERM_2__'
          → unmask()
  'Urob git commit a push na GitHub'

Použitie:
  python term_protection.py --text 'Do a git commit and push to GitHub'
  python term_protection.py --input whisperx_output.json --output term_reports/
"""

import re
import json
import argparse
from pathlib import Path


# ── Slovníky chránených výrazov ───────────────────────────────────────────────

GIT_TERMS = {
    "commit", "push", "pull", "branch", "merge", "rebase", "clone",
    "fork", "checkout", "stash", "diff", "fetch", "tag", "release",
    "repository", "repo", "upstream", "downstream", "HEAD", "master",
    "main", "origin", "remote", "staging", "deploy", "deployment",
    "pipeline", "workflow", "action", "runner", "dockerfile",
    "pull request", "merge request", "code review", "hotfix",
}

TECH_TERMS = {
    "API", "REST", "GraphQL", "JSON", "XML", "YAML", "CSV",
    "token", "cache", "cookie", "session", "header", "payload",
    "endpoint", "webhook", "callback", "middleware", "proxy",
    "frontend", "backend", "fullstack", "devops", "CI/CD",
    "debug", "debugger", "breakpoint", "stack trace", "log",
    "runtime", "framework", "library", "package", "module",
    "class", "function", "method", "array", "object", "string",
    "boolean", "integer", "float", "null", "undefined",
    "async", "await", "promise", "callback", "thread", "process",
    "CPU", "GPU", "RAM", "SSD", "HDD", "OS", "CLI", "GUI",
    "HTTP", "HTTPS", "SSH", "SSL", "TLS", "DNS", "IP", "URL",
    "localhost", "server", "client", "database", "query", "schema",
    "Docker", "Kubernetes", "container", "image", "pod", "cluster",
    "Linux", "Ubuntu", "bash", "shell", "terminal", "script",
    "Python", "JavaScript", "TypeScript", "Rust", "Go", "Java",
    "React", "Vue", "Angular", "Node", "npm", "pip", "conda",
    "machine learning", "deep learning", "neural network",
    "model", "training", "inference", "dataset", "batch",
    "loss", "gradient", "optimizer", "epoch", "checkpoint",
    "LLM", "GPT", "transformer", "embedding", "fine-tune",
    "fine-tuning", "prompt", "tokenizer",
}

BRAND_TERMS = {
    "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Google Cloud",
    "Terraform", "Ansible", "Jenkins", "CircleCI", "Travis",
    "Slack", "Discord", "Zoom", "Teams", "Notion", "Linear",
    "Python", "JavaScript", "TypeScript", "Rust", "Go", "Java",
    "React", "Vue", "Angular", "Next.js", "FastAPI", "Django",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "TensorFlow", "PyTorch", "Keras", "Hugging Face", "OpenAI",
    "Anthropic", "Claude", "ChatGPT", "Mistral", "LLaMA",
    "Ubuntu", "Debian", "CentOS", "macOS", "Windows",
    "VSCode", "IntelliJ", "PyCharm", "Vim", "Neovim",
    "Stack Overflow", "npm", "pip", "conda", "brew",
}

SLANG_TERMS = {
    "cool", "okay", "ok", "vibe", "vibes", "hype", "hyped",
    "woke", "based", "cringe", "sus", "lowkey", "highkey",
    "legit", "lit", "fire", "goat", "flex", "grind", "hustle",
    "burnout", "ghosting", "networking", "onboarding", "offboarding",
    "upskill", "reskill", "pivot", "scale", "scaling", "bootstrap",
    "startup", "unicorn", "exit", "pitch", "deck", "roadmap",
    "sprint", "scrum", "agile", "kanban", "standup", "retrospective",
    "MVP", "POC", "KPI", "OKR", "SLA", "SLO",
    "async", "sync", "remote", "hybrid", "in-person",
    "deep dive", "bandwidth", "synergy", "leverage", "stakeholder",
    "no-brainer", "game changer", "pain point", "use case",
}

# SQL / databázové termíny — NEMAJÚ sa prekladať
# POZOR: nezahŕňame bežné anglické slová (AND, OR, NOT, IN, AS, ON, CASE, WHEN...),
# tie by rozbili Google Translate / MADLAD vety. Chránime len technicky špecifické výrazy.
SQL_TERMS = {
    # NULL funkcie — hlavná príčina CER problémov (NULL→NUL/nulová hodnota/prázdna hodnota)
    "NULL", "null", "ISNULL", "isNull", "IS NULL", "IS NOT NULL",
    "NULLIF", "COALESCE", "IFNULL", "NVL",
    # Viacslovné SQL klauzuly (bezpečné — nie bežné anglické slová)
    "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN", "FULL JOIN", "CROSS JOIN",
    "GROUP BY", "ORDER BY", "PRIMARY KEY", "FOREIGN KEY",
    "NOT IN", "NOT EXISTS", "UNION ALL",
    "NOT NULL", "AUTO_INCREMENT",
    # SQL typy ktoré sa zle prekladajú
    "VARCHAR", "DATETIME", "TIMESTAMP",
    "BOOLEAN", "Boolean",
    # SQL príkazy ktoré sa v texte objavujú ako technické termíny
    "TRUNCATE", "ROLLBACK", "TRANSACTION",
    "stored procedure",
    # hodnoty
    "N/A", "n/a", "True", "False", "true", "false",
}

CUSTOM_TERMS: set[str] = set()


def load_custom_terms(path: str):
    global CUSTOM_TERMS
    p = Path(path)
    if p.exists():
        terms = {
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        CUSTOM_TERMS.update(terms)
        print(f"[term_protection] Načítaných {len(terms)} vlastných výrazov z {path}")


# ── TermProtector ─────────────────────────────────────────────────────────────

class TermProtector:
    TOKEN_PATTERN = re.compile(r"__TERM_\d+__")

    def __init__(
        self,
        protect_git:    bool = True,
        protect_tech:   bool = True,
        protect_brands: bool = True,
        protect_slang:  bool = True,
        protect_sql:    bool = True,
        custom_terms:   set  = None,
        case_sensitive: bool = False,
    ):
        self.case_sensitive = case_sensitive
        self._build_terms(
            protect_git, protect_tech, protect_brands,
            protect_slang, protect_sql, custom_terms or set()
        )

    def _build_terms(self, git, tech, brands, slang, sql, custom):
        all_terms = set()
        if git:    all_terms.update(GIT_TERMS)
        if tech:   all_terms.update(TECH_TERMS)
        if brands: all_terms.update(BRAND_TERMS)
        if slang:  all_terms.update(SLANG_TERMS)
        if sql:    all_terms.update(SQL_TERMS)
        all_terms.update(custom)
        all_terms.update(CUSTOM_TERMS)
        # zoradiť od najdlhších po najkratšie
        # → "IS NOT NULL" sa nahradí pred "NULL"
        self.terms = sorted(all_terms, key=len, reverse=True)

    def mask(self, text: str) -> tuple[str, dict]:
        mapping = {}
        counter = [0]

        def replace(match):
            original = match.group(0)
            token = f"__TERM_{counter[0]}__"
            mapping[token] = original
            counter[0] += 1
            return token

        result = text
        for term in self.terms:
            if not term:
                continue
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
                result = re.sub(pattern, replace, result, flags=flags)
            except re.error:
                continue
        return result, mapping

    def unmask(self, text: str, mapping: dict) -> str:
        result = self._normalize_tokens(text, mapping)
        for token, original in mapping.items():
            result = result.replace(token, original)
        return result

    def _normalize_tokens(self, text: str, mapping: dict) -> str:
        result = text
        for token in mapping:
            lower = token.lower()
            if lower in result and token not in result:
                result = result.replace(lower, token)
        result = re.sub(r"__\s*TERM_(\d+)\s*__", r"__TERM_\1__", result)
        return result

    def mask_batch(self, texts: list[str]) -> tuple[list[str], list[dict]]:
        masked_texts, mappings = [], []
        for text in texts:
            m, mp = self.mask(text)
            masked_texts.append(m)
            mappings.append(mp)
        return masked_texts, mappings

    def unmask_batch(self, texts: list[str], mappings: list[dict]) -> list[str]:
        return [self.unmask(t, mp) for t, mp in zip(texts, mappings)]

    def analyze(self, text: str) -> list[dict]:
        found = []
        for term in self.terms:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
                for m in re.finditer(pattern, text, flags=flags):
                    found.append({"term": term, "match": m.group(0),
                                  "start": m.start(), "end": m.end()})
            except re.error:
                continue
        found.sort(key=lambda x: x["start"])
        return found


# ── Integrácia do translation pipeline ───────────────────────────────────────

def translate_with_protection(
    segments:  list[dict],
    translator,
    src_lang:  str,
    tp:        TermProtector = None,
    tgt_lang:  str = "slk_Latn",
) -> list[dict]:
    if tp is None:
        tp = TermProtector()

    texts = [s.get("text", "").strip() for s in segments]
    masked_texts, mappings = tp.mask_batch(texts)

    translated = translator(
        masked_texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=256,
        batch_size=16,
    )

    results = []
    for seg, t, mapping in zip(segments, translated, mappings):
        sk_raw   = t["translation_text"]
        sk_final = tp.unmask(sk_raw, mapping)
        results.append({
            **seg,
            "sk_text":         sk_final,
            "sk_text_raw":     sk_raw,
            "protected_terms": list(mapping.values()),
        })
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ochrana výrazov pred prekladom.")
    parser.add_argument("--text",         help="Jeden text na test")
    parser.add_argument("--input",        help="WhisperX JSON na analýzu")
    parser.add_argument("--output",       help="Výstupný adresár")
    parser.add_argument("--custom-terms", help="Súbor s vlastnými výrazmi")
    parser.add_argument("--no-git",       action="store_true")
    parser.add_argument("--no-tech",      action="store_true")
    parser.add_argument("--no-brands",    action="store_true")
    parser.add_argument("--no-slang",     action="store_true")
    parser.add_argument("--no-sql",       action="store_true")
    args = parser.parse_args()

    if args.custom_terms:
        load_custom_terms(args.custom_terms)

    tp = TermProtector(
        protect_git=    not args.no_git,
        protect_tech=   not args.no_tech,
        protect_brands= not args.no_brands,
        protect_slang=  not args.no_slang,
        protect_sql=    not args.no_sql,
    )

    if args.text:
        masked, mapping = tp.mask(args.text)
        print(f"\nOriginál : {args.text}")
        print(f"Maskovaný: {masked}")
        print(f"\nChránené výrazy ({len(mapping)}):")
        for token, original in mapping.items():
            print(f"  {token} → '{original}'")
        if not mapping:
            print("  (žiadne)")
        return

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", data) if isinstance(data, dict) else data
        print(f"Segmentov: {len(segments)}")

        category_map = {}
        for t in GIT_TERMS:    category_map[t.lower()] = "git"
        for t in TECH_TERMS:   category_map[t.lower()] = "tech"
        for t in BRAND_TERMS:  category_map[t.lower()] = "brands"
        for t in SLANG_TERMS:  category_map[t.lower()] = "slang"
        for t in SQL_TERMS:    category_map[t.lower()] = "sql"
        for t in CUSTOM_TERMS: category_map[t.lower()] = "custom"

        buckets   = {"git": [], "tech": [], "brands": [], "slang": [], "sql": [], "custom": []}
        term_freq = {}

        for i, seg in enumerate(segments):
            text  = seg.get("text", "")
            found = tp.analyze(text)
            for f in found:
                term_freq[f["term"]] = term_freq.get(f["term"], 0) + 1
                cat = category_map.get(f["term"].lower(), "custom")
                record = {"idx": i, "start": seg.get("start", 0),
                          "end": seg.get("end", 0), "text": text,
                          "term": f["term"], "match": f["match"]}
                if not any(r["idx"] == i and r["term"] == f["term"] for r in buckets[cat]):
                    buckets[cat].append(record)

        total_found = sum(len(v) for v in buckets.values())
        print(f"\nNájdených výskytov: {total_found}")
        for cat, items in buckets.items():
            if items:
                print(f"  {cat:<10} {len(items)} výskytov")
        print("\nNajčastejšie výrazy:")
        for term, count in sorted(term_freq.items(), key=lambda x: -x[1])[:20]:
            cat = category_map.get(term.lower(), "custom")
            print(f"  {count:4d}×  [{cat:<8}]  {term}")

        if args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            for cat, items in buckets.items():
                if not items:
                    continue
                path = out_dir / f"terms_{cat}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"  ✓ {path}  ({len(items)} záznamov)")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
