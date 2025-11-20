#!/usr/bin/env python3
"""Fetch the first 10k AceCode records and store id/question pairs."""

from __future__ import annotations

import argparse
import csv
from itertools import islice
from pathlib import Path
from typing import Iterable, Mapping

from datasets import load_dataset

DATASET_ID = "TIGER-Lab/AceCode-87K"
SPLIT = "train"
FIELDNAMES = ["id", "question"]


def iter_rows(limit: int) -> Iterable[Mapping[str, str]]:
    """Stream rows from the dataset without downloading the full corpus."""
    dataset = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    return islice(dataset, limit)


def extract_question(row: Mapping[str, str]) -> str:
    """Return the best available problem statement field."""
    for key in ("question", "problem", "prompt"):
        text = row.get(key)
        if text:
            return text
    return ""


def export_rows(rows: Iterable[Mapping[str, str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        count = 0
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "id": row.get("id", f"sample_{idx}"),
                    "question": extract_question(row),
                }
            )
            if idx % 1000 == 0:
                print(f"Wrote {idx} rows...")
            count = idx
    print(f"Saved {count} total rows to {dest}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="number of rows to fetch (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("problems10k.csv"),
        help="output CSV path (default: problems10k.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise SystemExit("rows must be a positive integer")
    print(
        f"Fetching first {args.rows:,} rows from {DATASET_ID} ({SPLIT} split) into {args.output}"
    )
    rows = iter_rows(args.rows)
    export_rows(rows, args.output)


if __name__ == "__main__":
    main()
