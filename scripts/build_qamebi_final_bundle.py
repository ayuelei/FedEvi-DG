#!/usr/bin/env python3
"""Build a five-source, result-blind QAMEBI external-evaluation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


SOURCE_DOMAINS = ("BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT")
SPLIT_SEED = 20260728
VAL_FRACTION = 0.15
FRAME_NAMES = ("source_train", "source_val", "target_test")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("all labels must be binary")
    return {
        str(int(label)): int(count)
        for label, count in labels.value_counts().sort_index().items()
    }


def _stable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["dataset", "patient_id", "image_path"], kind="stable"
    ).reset_index(drop=True)


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _read_domain_inventory(
    lodo_root: Path, domain: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates: list[pd.DataFrame] = []
    provenance: list[str] = []
    for held_out in SOURCE_DOMAINS:
        if held_out == domain:
            continue
        bundle = lodo_root / f"target_{held_out}"
        paths = [bundle / "source_train.csv", bundle / "source_val.csv"]
        frame = pd.concat(
            [pd.read_csv(path, keep_default_na=False) for path in paths],
            ignore_index=True,
        )
        frame = frame[frame["dataset"].astype(str) == domain].copy()
        if frame.empty:
            raise RuntimeError(f"{bundle} contains no source rows for {domain}")
        candidates.append(_stable_frame(frame))
        provenance.append(str(bundle.resolve()))
    hashes = [_frame_hash(candidate) for candidate in candidates]
    if len(set(hashes)) != 1:
        raise RuntimeError(
            f"{domain} full inventory differs across frozen LODO bundles"
        )
    return candidates[0], {
        "inventory_frame_sha256": hashes[0],
        "candidates": [
            {
                "bundle": path,
                "frame_sha256": digest,
                "rows": len(candidate),
            }
            for path, digest, candidate in zip(
                provenance, hashes, candidates, strict=True
            )
        ],
    }


def _split_domain_inventory(
    inventory: pd.DataFrame, domain: str, domain_index: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = inventory.groupby("patient_id", sort=True)["label"].agg(
        ["first", "nunique"]
    )
    if (groups["nunique"] > 1).any():
        mixed = groups.index[groups["nunique"] > 1].tolist()
        raise RuntimeError(f"mixed-label patient groups in {domain}: {mixed}")
    if set(pd.to_numeric(groups["first"]).astype(int)) != {0, 1}:
        raise RuntimeError(f"{domain} inventory lacks one patient-level class")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_FRACTION,
        random_state=SPLIT_SEED + domain_index,
    )
    train_index, val_index = next(
        splitter.split(groups.index.to_numpy(), groups["first"].to_numpy())
    )
    train_ids = set(groups.index[train_index])
    val_ids = set(groups.index[val_index])
    if not train_ids.isdisjoint(val_ids):
        raise RuntimeError(f"patient leakage in new {domain} split")
    train = _stable_frame(inventory[inventory["patient_id"].isin(train_ids)])
    val = _stable_frame(inventory[inventory["patient_id"].isin(val_ids)])
    return train, val


def _canonical_patient_groups(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(dataset), " ".join(str(patient).split()).casefold())
        for dataset, patient in zip(
            frame["dataset"], frame["patient_id"], strict=True
        )
    }


def _build_target(manifest_path: Path) -> pd.DataFrame:
    target = pd.read_csv(manifest_path, keep_default_na=False)
    if not target["eligible"].astype(bool).all():
        raise ValueError("frozen QAMEBI manifest contains ineligible rows")
    if set(target["dataset"].astype(str)) != {"QAMEBI"}:
        raise ValueError("target manifest must contain only QAMEBI")
    if target["sha256"].duplicated().any():
        raise ValueError("target manifest contains duplicate image hashes")
    target["near_duplicate_resolution"] = "external_audit_passed"
    target["birads_ordinal"] = ""
    return _stable_frame(target)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lodo-root", type=Path, required=True)
    parser.add_argument("--qamebi-manifest", type=Path, required=True)
    parser.add_argument("--qamebi-audit-summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    destination = args.output_root / "target_QAMEBI"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if not args.lodo_root.is_dir():
        raise FileNotFoundError(args.lodo_root)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    source_provenance: dict[str, dict[str, list[str]]] = {}
    for domain_index, domain in enumerate(sorted(SOURCE_DOMAINS)):
        inventory, inventory_provenance = _read_domain_inventory(
            args.lodo_root, domain
        )
        train, val = _split_domain_inventory(inventory, domain, domain_index)
        if not _canonical_patient_groups(train).isdisjoint(
            _canonical_patient_groups(val)
        ):
            raise RuntimeError(f"patient leakage in frozen {domain} train/val split")
        if set(pd.to_numeric(train["label"]).astype(int)) != {0, 1}:
            raise RuntimeError(f"{domain} source_train lacks one class")
        if set(pd.to_numeric(val["label"]).astype(int)) != {0, 1}:
            raise RuntimeError(f"{domain} source_val lacks one class")
        train_parts.append(train)
        val_parts.append(val)
        source_provenance[domain] = {
            "inventory": inventory_provenance,
            "split_rule": {
                "kind": "per_domain_patient_group_stratified_shuffle_split",
                "split_seed": SPLIT_SEED,
                "domain_index": domain_index,
                "effective_random_state": SPLIT_SEED + domain_index,
                "val_fraction": VAL_FRACTION,
                "source_train_frame_sha256": _frame_hash(train),
                "source_val_frame_sha256": _frame_hash(val),
            },
        }

    frames = {
        "source_train": _stable_frame(pd.concat(train_parts, ignore_index=True)),
        "source_val": _stable_frame(pd.concat(val_parts, ignore_index=True)),
        "target_test": _build_target(args.qamebi_manifest),
    }
    source_hashes = set(frames["source_train"]["sha256"]) | set(
        frames["source_val"]["sha256"]
    )
    if source_hashes & set(frames["target_test"]["sha256"]):
        raise RuntimeError("exact source-target overlap in final external bundle")

    split_config = {
        "schema_version": 1,
        "protocol": "five_source_locked_public_external_validation",
        "domains": [*SOURCE_DOMAINS, "QAMEBI"],
        "source_domains": list(SOURCE_DOMAINS),
        "target_domain": "QAMEBI",
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "target_used_for_model_selection": False,
        "target_used_for_calibration": False,
        "target_used_for_threshold_selection": False,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=args.output_root, prefix=".target_QAMEBI.staging-")
    )
    try:
        files: dict[str, dict[str, object]] = {}
        for name in FRAME_NAMES:
            filename = f"{name}.csv"
            frames[name].to_csv(staging / filename, index=False)
            files[filename] = {
                "sha256": _sha256_file(staging / filename),
                "rows": len(frames[name]),
                "class_counts": _class_counts(frames[name]),
                "domain_counts": {
                    str(key): int(value)
                    for key, value in frames[name]["dataset"].value_counts().items()
                },
            }
        metadata = {
            "schema_version": 1,
            "target_domain": "QAMEBI",
            "source_domains": list(SOURCE_DOMAINS),
            "split_seed": SPLIT_SEED,
            "val_fraction": VAL_FRACTION,
            "split_config": split_config,
            "config_hash": _canonical_hash(split_config),
            "files": files,
            "source_split_provenance": source_provenance,
            "qamebi_manifest": str(args.qamebi_manifest.resolve()),
            "qamebi_manifest_sha256": _sha256_file(args.qamebi_manifest),
            "qamebi_audit_summary": str(args.qamebi_audit_summary.resolve()),
            "qamebi_audit_summary_sha256": _sha256_file(
                args.qamebi_audit_summary
            ),
            "frozen_protocol": str(args.protocol.resolve()),
            "frozen_protocol_sha256": _sha256_file(args.protocol),
            "model_predictions_seen": False,
        }
        _write_json(staging / "metadata.json", metadata)
        _write_json(
            staging / "COMPLETE",
            {"metadata_sha256": _sha256_file(staging / "metadata.json")},
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
