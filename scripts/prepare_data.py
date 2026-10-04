#!/usr/bin/env python3
"""Rebuild the paper's local manifests from provider files, without derivatives."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.parse_source_data import (
    _authoritative_rows,
    _apply_exact_duplicate_policy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    frame = _authoritative_rows(raw)
    frame["raw_image_path"] = frame["image_path"]
    retained, decisions, excluded, _ = _apply_exact_duplicate_policy(frame)
    rules = json.loads((ROOT / "configs/development_exclusions.json").read_text())
    retained = retained[~retained.sha256.isin(rules["exclude_sha256"])].copy()
    output.mkdir(parents=True)
    retained.to_csv(output / "metadata.csv", index=False)
    decisions.to_csv(output / "exact_duplicate_decisions.csv", index=False)
    excluded.to_csv(output / "exact_duplicate_exclusions.csv", index=False)

    def run(script, *flags):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *map(str, flags)],
            cwd=ROOT,
            check=True,
        )

    run(
        "attach_birads_to_metadata.py",
        "--raw-root",
        raw,
        "--metadata-csv",
        output / "metadata.csv",
        "--output-csv",
        output / "metadata_birads.csv",
        "--summary-json",
        output / "birads_coverage.json",
    )
    run(
        "build_lodo_manifests.py",
        "--data-root",
        raw,
        "--metadata-csv",
        output / "metadata_birads.csv",
        "--output-root",
        output / "classification_splits",
    )
    run(
        "build_fedevi_manifests.py",
        "--source-root",
        output / "classification_splits",
        "--output-root",
        output / "lodo",
    )
    run("verify_splits.py", "--manifest-root", output / "lodo")
    print(f"Verified manifests: {output / 'lodo'}")


if __name__ == "__main__":
    main()
