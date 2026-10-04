#!/usr/bin/env python3
"""Verify local split contents without publishing any per-record manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def frame_signature(frame):
    columns = [
        x
        for x in (
            "dataset",
            "sha256",
            "patient_id",
            "patient_id_source",
            "label",
            "birads_ordinal",
            "mask_sha256",
        )
        if x in frame
    ]
    canonical = (
        frame[columns].fillna("").astype(str).sort_values(columns, kind="stable")
    )
    return {
        "rows": len(frame),
        "columns": columns,
        "content_sha256": hashlib.sha256(
            canonical.to_csv(index=False).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reproducibility/split_fingerprints.json",
    )
    args = parser.parse_args()
    for target, splits in json.loads(args.expected.read_text()).items():
        for name, expected in splits.items():
            path = args.manifest_root / f"target_{target}" / f"{name}.csv"
            if frame_signature(pd.read_csv(path)) != expected:
                raise ValueError(
                    f"Split contents differ from the paper: {target}/{name}"
                )
    print("All 15 data-root-independent split fingerprints: PASS")


if __name__ == "__main__":
    main()
