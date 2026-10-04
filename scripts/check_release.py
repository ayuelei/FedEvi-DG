#!/usr/bin/env python3
"""Audit only the code-release allowlist; never walk local data/output trees."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (
    "fedevi_dg",
    "scripts",
    "configs",
    "tests",
    "docs",
    "reference_results",
    "reproducibility",
)
ROOT_FILES = (
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "LICENSE",
)
SUFFIXES = {".py", ".md", ".json", ".csv", ".toml", ".txt"}
INVENTORY = "reproducibility/release_files.json"


def collect_files():
    paths = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.is_symlink() or path.suffix not in SUFFIXES:
                raise ValueError(
                    f"Unexpected release artifact: {path.relative_to(ROOT)}"
                )
            # Generated manuscript figures are excluded even if text-based.
            if path.is_relative_to(ROOT / "docs/paper"):
                raise ValueError(
                    "Generated manuscript output must not be in the release"
                )
            paths.append(path)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-inventory", action="store_true")
    args = parser.parse_args()
    hashes = {}
    for path in collect_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative == INVENTORY:
            continue
        value = path.read_bytes()
        text = value.decode("utf-8")
        if re.search(
            r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,})",
            text,
        ):
            raise ValueError(f"Possible embedded credential: {relative}")
        if re.search(
            r"/(?:private/Pycharm_Project|home/Pycharm_Project|root/\.codex)/", text
        ):
            raise ValueError(f"Embedded private environment path: {relative}")
        if path.suffix == ".csv":
            if not path.is_relative_to(ROOT / "reference_results"):
                raise ValueError(f"Non-reference CSV not allowed: {relative}")
            columns = set(pd.read_csv(path, nrows=0).columns)
            if columns & {
                "patient_id",
                "image_path",
                "mask_path",
                "record_id",
                "score",
                "label",
            }:
                raise ValueError(f"Record-level CSV not allowed: {relative}")
        hashes[relative] = hashlib.sha256(value).hexdigest()
    inventory = ROOT / INVENTORY
    if args.write_inventory:
        inventory.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    else:
        if hashes != json.loads(inventory.read_text()):
            raise ValueError("Release contents differ from the recorded file inventory")
    print(
        f"Data-free release allowlist/integrity audit: PASS ({len(hashes)} files + inventory)"
    )


if __name__ == "__main__":
    main()
