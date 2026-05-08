#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

random.seed(42)

SQL_TERMS = (
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare Level 18 curriculum/weighted training pack.")
    ap.add_argument("--train", required=True, help="Level 17 train.jsonl")
    ap.add_argument("--val", required=True, help="Level 17 val.jsonl")
    ap.add_argument("--out-dir", default="level18_curriculum_pack", help="Output directory")
    ap.add_argument("--core-weight", type=int, default=5, help="Oversampling weight for clean core")
    ap.add_argument("--sql-weight", type=int, default=3, help="Oversampling weight for technical SQL subset")
    ap.add_argument("--hard-weight", type=int, default=2, help="Oversampling weight for hard subset")
    return ap


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _contains_sql(text: str) -> bool:
    upper = text.upper()
    return any(term in upper for term in SQL_TERMS)


def _is_hard(text: str) -> bool:
    clean = (text or "").strip()
    return (
        len(clean) >= 90
        or clean.count(",") >= 2
        or clean.count(".") >= 2
        or any(token in clean.lower() for token in ("napríklad", "pozrime sa", "ak hodnota", "objednávk"))
    )


def _classify(row: dict) -> str:
    text = str(row.get("text") or "")
    if _contains_sql(text):
        if _is_hard(text):
            return "hard_examples"
        return "technical_sql"
    if _is_hard(text):
        return "hard_examples"
    return "clean_core"


def _oversample(rows: list[dict], weight: int) -> list[dict]:
    if weight <= 1:
        return list(rows)
    out: list[dict] = []
    for row in rows:
        for _ in range(weight):
            out.append(dict(row))
    return out


def main() -> None:
    args = build_parser().parse_args()
    train_path = Path(args.train).expanduser().resolve()
    val_path = Path(args.val).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_jsonl(train_path)
    val_rows = _load_jsonl(val_path)

    buckets = {"clean_core": [], "technical_sql": [], "hard_examples": []}
    for row in train_rows:
        buckets[_classify(row)].append(row)

    weighted_train = []
    weighted_train.extend(_oversample(buckets["clean_core"], args.core_weight))
    weighted_train.extend(_oversample(buckets["technical_sql"], args.sql_weight))
    weighted_train.extend(_oversample(buckets["hard_examples"], args.hard_weight))
    random.shuffle(weighted_train)

    _write_jsonl(out_dir / "train_clean_core.jsonl", buckets["clean_core"])
    _write_jsonl(out_dir / "train_technical_sql.jsonl", buckets["technical_sql"])
    _write_jsonl(out_dir / "train_hard_examples.jsonl", buckets["hard_examples"])
    _write_jsonl(out_dir / "train_weighted.jsonl", weighted_train)
    _write_jsonl(out_dir / "val.jsonl", val_rows)

    stats = {
        "input_train": len(train_rows),
        "input_val": len(val_rows),
        "clean_core": len(buckets["clean_core"]),
        "technical_sql": len(buckets["technical_sql"]),
        "hard_examples": len(buckets["hard_examples"]),
        "weighted_train": len(weighted_train),
        "weights": {
            "clean_core": args.core_weight,
            "technical_sql": args.sql_weight,
            "hard_examples": args.hard_weight,
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"HOTOVO: {out_dir}")


if __name__ == "__main__":
    main()
