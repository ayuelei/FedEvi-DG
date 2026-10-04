#!/usr/bin/env python
"""Attach within-domain BI-RADS ordinal levels to audited FedEvi-DG metadata.

Reads the raw sidecar files (BUSBRA ``bus_data.csv`` and the shared
``BIRADS&FOLD.xlsx`` for GDPH/SYSUCC), maps every category to a strictly
monotone ordinal scale, and writes the audited metadata plus a new
``birads_ordinal`` column to a fresh output CSV. Missing or unmappable
levels stay empty; the anchor loss treats them as absent.

BUSBRA's plain numeric categories are ranked within the domain (the anchor
only compares levels inside one client). GDPH/SYSUCC carry two readers on
a shared category scale; their mapped levels are averaged, and exact ties
produce no penalty.

The script never modifies the signed audit bundle in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CATEGORY_SCALE = {
    "2": 1.0,
    "3": 2.0,
    "4A": 3.0,
    "4B": 4.0,
    "4C": 5.0,
    "5": 6.0,
}


def _category_level(value: object) -> float | None:
    if pd.isna(value):
        return None
    token = str(value).strip().upper()
    if token in CATEGORY_SCALE:
        return CATEGORY_SCALE[token]
    try:
        numeric = float(token)
    except ValueError:
        return None
    if numeric == int(numeric) and 1 <= int(numeric) <= 5:
        return None  # plain "4" is ambiguous between 4A/4B/4C on the shared scale
    return None


def _numeric_level(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        numeric = float(str(value).strip())
    except ValueError:
        return None
    return numeric


def build_lookup(raw_root: Path) -> dict[str, dict[str, float]]:
    """Map filename-stem -> {dataset: ordinal level} where determinable."""
    by_stem: dict[str, dict[str, float]] = {}

    busbra_csv = raw_root / "BUSBRA" / "bus_data.csv"
    busbra = pd.read_csv(busbra_csv)
    numeric_levels: dict[str, float] = {}
    for identifier, birads in zip(busbra["ID"], busbra["BIRADS"], strict=True):
        level = _numeric_level(birads)
        if level is not None:
            numeric_levels[str(identifier).strip()] = level
    # within-domain ranks keep the anchor scale-free and monotone
    unique_sorted = sorted(set(numeric_levels.values()))
    rank_of = {value: float(index + 1) for index, value in enumerate(unique_sorted)}
    for identifier, level in numeric_levels.items():
        by_stem.setdefault(identifier, {})["BUSBRA"] = rank_of[level]

    birads_xlsx = raw_root / "GDPH&SYSUCC" / "BIRADS&FOLD.xlsx"
    readers = pd.read_excel(birads_xlsx, sheet_name="prediction")
    for identifier, reader1, reader2 in zip(
        readers["ID"], readers["BIRADS-reader1"], readers["BIRADS-reader2"], strict=True
    ):
        levels = [
            level
            for level in (
                _category_level(reader1),
                _category_level(reader2),
            )
            if level is not None
        ]
        if not levels:
            continue
        stem = str(identifier).strip()
        by_stem.setdefault(stem, {}).setdefault("GDPH", sum(levels) / len(levels))
        by_stem[stem].setdefault("SYSUCC", sum(levels) / len(levels))
    return by_stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args()

    for path in (args.output_csv, args.summary_json):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    lookup = build_lookup(args.raw_root.resolve())
    metadata = pd.read_csv(args.metadata_csv)

    levels: list[float] = []
    coverage = {
        domain: {"rows": 0, "matched": 0}
        for domain in sorted(metadata["dataset"].unique())
    }
    ambiguous = 0
    for row in metadata.itertuples(index=False):
        domain = str(row.dataset)
        stem = Path(str(row.image_path)).stem
        coverage[domain]["rows"] += 1
        domain_levels = lookup.get(stem, {})
        level = domain_levels.get(domain)
        if len(domain_levels) > 1 and len(set(domain_levels.values())) > 1:
            ambiguous += 1
            continue
        if level is None:
            levels.append(float("nan"))
            continue
        coverage[domain]["matched"] += 1
        levels.append(float(level))

    if ambiguous:
        raise SystemExit(
            f"{ambiguous} rows map to conflicting BI-RADS levels; resolve manually"
        )

    metadata = metadata.copy()
    metadata["birads_ordinal"] = levels
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_csv, index=False)

    summary = {
        "source_metadata": str(args.metadata_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "coverage": coverage,
        "birads_scale": {
            "GDPH_SYSUCC_category_scale": CATEGORY_SCALE,
            "BUSBRA": "within-domain ranks of numeric BIRADS",
        },
        "rows": int(len(metadata)),
        "matched_rows": int(sum(item["matched"] for item in coverage.values())),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["coverage"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
