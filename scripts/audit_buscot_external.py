#!/usr/bin/env python3
"""Freeze a provenance-filtered BUS-CoT reasoning benchmark before inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image


KNOWN_DEVELOPMENT_SOURCES = frozenset({"BUSBRA", "BUSI", "BUDIAT"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dct_matrix(size: int = 32) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (positions + 0.5) * frequencies / size)
    matrix[0] *= np.sqrt(1.0 / size)
    matrix[1:] *= np.sqrt(2.0 / size)
    return matrix


DCT = _dct_matrix()
BIT_COUNTS = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def _phash_bytes(value: bytes) -> str:
    with Image.open(io.BytesIO(value)) as image:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    low = (DCT @ pixels @ DCT.T)[:8, :8]
    threshold = np.median(low.reshape(-1)[1:])
    bits = low.reshape(-1) > threshold
    integer = 0
    for bit in bits:
        integer = (integer << 1) | int(bit)
    return f"{integer:016x}"


def _minimum_hamming(value: str, references: np.ndarray) -> int:
    integer = np.asarray([int(value, 16)], dtype=">u8").view(np.uint8)
    xor = np.bitwise_xor(references, integer)
    return int(BIT_COUNTS[xor].sum(axis=1).min())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    development_group = parser.add_mutually_exclusive_group(required=True)
    development_group.add_argument("--development-manifest", type=Path)
    development_group.add_argument("--development-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    expert = json.loads(
        (args.metadata_dir / "BUS-Expert_dataset.json").read_text()
    )
    lesion = json.loads((args.metadata_dir / "lesion_dataset.json").read_text())
    if args.development_bundle is not None:
        development_paths = [
            args.development_bundle / f"{name}.csv"
            for name in ("source_train", "source_val", "target_test")
        ]
        development = pd.concat(
            [pd.read_csv(path) for path in development_paths], ignore_index=True
        )
    else:
        development_paths = [args.development_manifest]
        development = pd.read_csv(args.development_manifest)
    development_sha = set(development["sha256"].astype(str))
    development_phash_hex = development["phash"].astype(str).tolist()
    development_phashes = np.asarray(
        [int(value, 16) for value in development_phash_hex], dtype=">u8"
    ).view(np.uint8).reshape(-1, 8)

    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        for key, record in sorted(lesion.items()):
            if record.get("split") != "test":
                continue
            pathology = record.get("pathology_histology", {}).get("pathology")
            if pathology not in {"Benign", "Malignant"}:
                continue
            image_id = Path(record["image_path"]).name.split("@")[0]
            parent = expert[image_id]
            source = str(parent.get("dataset", ""))
            archive_path = "BUSCoT/" + record["image_path"]
            raw = archive.read(archive_path)
            sha256 = _sha256_bytes(raw)
            phash = _phash_bytes(raw)
            minimum_distance = _minimum_hamming(phash, development_phashes)
            known_source_overlap = source in KNOWN_DEVELOPMENT_SOURCES
            exact_overlap = sha256 in development_sha
            near_overlap = minimum_distance <= args.near_duplicate_distance
            eligible = not (known_source_overlap or exact_overlap or near_overlap)
            report = record.get("us_report", {})
            rows.append(
                {
                    "buscot_key": key,
                    "dataset": "BUSCoT_reasoning_external",
                    "source_dataset": source,
                    "patient_id": parent.get("patient_id"),
                    "image_path": archive_path,
                    "label": int(pathology == "Malignant"),
                    "pathology": pathology,
                    "birads": report.get("BIRADS"),
                    "lesion_edge": report.get("LesionEdge"),
                    "lesion_boundary": report.get("LesionBoundary"),
                    "calcification": report.get("LesionCalcificationFeatures"),
                    "echo": report.get("EchoCharacteristics"),
                    "sha256": sha256,
                    "phash": phash,
                    "minimum_development_phash_distance": minimum_distance,
                    "known_source_overlap": known_source_overlap,
                    "exact_sha256_overlap": exact_overlap,
                    "near_duplicate_overlap": near_overlap,
                    "eligible": eligible,
                }
            )

    audit = pd.DataFrame(rows)
    frozen = audit[audit["eligible"]].copy().reset_index(drop=True)
    if frozen.empty or frozen["label"].nunique() != 2:
        raise RuntimeError("provenance filtering left no valid binary benchmark")
    if frozen["patient_id"].isna().any():
        raise RuntimeError("eligible BUS-CoT records require patient IDs")
    args.output_dir.mkdir(parents=True)
    audit.to_csv(args.output_dir / "audit_all.csv", index=False)
    frozen.to_csv(args.output_dir / "frozen_manifest.csv", index=False)
    summary = {
        "role": "unseen_public_reasoning_and_explanation_benchmark",
        "not_a_clinical_external_cohort": True,
        "archive": str(args.archive.resolve()),
        "archive_sha256": _sha256_file(args.archive),
        "development_manifests": [str(path.resolve()) for path in development_paths],
        "development_manifest_sha256": {
            path.name: _sha256_file(path) for path in development_paths
        },
        "official_split": "test",
        "known_development_sources_excluded": sorted(KNOWN_DEVELOPMENT_SOURCES),
        "near_duplicate_hamming_threshold": args.near_duplicate_distance,
        "audited_rows": len(audit),
        "eligible_rows": len(frozen),
        "eligible_patients": int(frozen["patient_id"].nunique()),
        "eligible_class_counts": {
            str(key): int(value)
            for key, value in frozen["label"].value_counts().sort_index().items()
        },
        "eligible_source_counts": {
            str(key): int(value)
            for key, value in frozen["source_dataset"].value_counts().items()
        },
        "excluded_known_source": int(audit["known_source_overlap"].sum()),
        "excluded_exact_sha256": int(audit["exact_sha256_overlap"].sum()),
        "excluded_near_duplicate": int(audit["near_duplicate_overlap"].sum()),
        "frozen_manifest_sha256": _sha256_file(
            args.output_dir / "frozen_manifest.csv"
        ),
        "model_predictions_seen": False,
    }
    (args.output_dir / "freeze.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
