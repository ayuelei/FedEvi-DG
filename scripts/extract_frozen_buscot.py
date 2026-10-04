#!/usr/bin/env python3
"""Materialize only the preregistered BUS-CoT benchmark images."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile

import pandas as pd


def _birads_ordinal(value: object) -> float:
    """Map BUS-CoT BI-RADS text to the normalized ordinal target used in training."""
    if pd.isna(value):
        return float("nan")
    normalized = str(value).strip().upper().replace("BI-RADS", "").strip()
    mapping = {
        "2": 0.0,
        "3": 0.2,
        "4": 0.5,
        "4A": 0.4,
        "4B": 0.6,
        "4C": 0.8,
        "5": 1.0,
    }
    return mapping.get(normalized, float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.frozen_manifest)
    required = {"buscot_key", "image_path", "sha256", "label", "patient_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frozen manifest is missing columns: {sorted(missing)}")
    if frame.empty or frame["label"].nunique() != 2:
        raise ValueError("frozen manifest must contain a non-empty binary benchmark")
    if frame["buscot_key"].duplicated().any():
        raise ValueError("buscot_key must be unique")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.", dir=args.output_root.parent
        )
    )
    try:
        image_dir = staging / "images"
        image_dir.mkdir()
        relative_paths = []
        with zipfile.ZipFile(args.archive) as archive:
            for row in frame.itertuples(index=False):
                value = archive.read(str(row.image_path))
                digest = hashlib.sha256(value).hexdigest()
                if digest != str(row.sha256):
                    raise ValueError(
                        f"BUS-CoT byte hash mismatch for {row.image_path}"
                    )
                relative = Path("images") / f"{row.buscot_key}.png"
                (staging / relative).write_bytes(value)
                relative_paths.append(str(relative))
        materialized = frame.copy()
        materialized["archive_image_path"] = materialized["image_path"]
        materialized["image_path"] = relative_paths
        if "birads" in materialized:
            materialized["birads_ordinal"] = materialized["birads"].map(
                _birads_ordinal
            )
        materialized.to_csv(staging / "manifest.csv", index=False)
        staging.rename(args.output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
