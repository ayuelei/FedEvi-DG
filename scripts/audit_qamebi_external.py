#!/usr/bin/env python3
"""Build and audit the result-blind QAMEBI public external manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


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


def _phash_file(path: Path) -> str:
    with Image.open(path) as image:
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


def _phash_bytes(values: list[str]) -> np.ndarray:
    return np.asarray([int(value, 16) for value in values], dtype=">u8").view(
        np.uint8
    ).reshape(-1, 8)


def _hamming_distances(value: str, references: np.ndarray) -> np.ndarray:
    integer = np.asarray([int(value, 16)], dtype=">u8").view(np.uint8)
    xor = np.bitwise_xor(references, integer)
    return BIT_COUNTS[xor].sum(axis=1)


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )


def _append_reference(
    records: list[dict[str, str]], *, source: str, path: Path
) -> None:
    with Image.open(path) as image:
        image.verify()
    records.append(
        {
            "source": source,
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "phash": _phash_file(path),
        }
    )


def _development_references(bundle: Path) -> list[dict[str, str]]:
    frame = pd.concat(
        [
            pd.read_csv(bundle / name)
            for name in ("source_train.csv", "source_val.csv", "target_test.csv")
        ],
        ignore_index=True,
    )
    required = {"dataset", "image_path", "sha256", "phash"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"development manifests lack {sorted(missing)}")
    if frame["image_path"].duplicated().any():
        raise ValueError("development bundle contains duplicate image paths")
    return [
        {
            "source": f"development:{row.dataset}",
            "path": str(row.image_path),
            "sha256": str(row.sha256),
            "phash": str(row.phash),
        }
        for row in frame.itertuples(index=False)
    ]


def _manifest_references(
    manifest: Path,
    *,
    source: str,
    root: Path | None,
    path_column: str,
) -> list[dict[str, str]]:
    frame = pd.read_csv(manifest)
    records: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        value = Path(str(getattr(row, path_column)))
        path = value if value.is_absolute() or root is None else root / value
        sha256 = str(getattr(row, "sha256", ""))
        phash = str(getattr(row, "phash", ""))
        if len(sha256) != 64 or len(phash) != 16:
            _append_reference(records, source=source, path=path)
        else:
            records.append(
                {
                    "source": source,
                    "path": str(path.resolve()),
                    "sha256": sha256,
                    "phash": phash,
                }
            )
    return records


def _scan_references(
    root: Path, *, source: str, required_path_component: str | None = None
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records: list[dict[str, str]] = []
    unreadable: list[dict[str, str]] = []
    for path in _image_files(root):
        if (
            required_path_component is not None
            and required_path_component not in path.relative_to(root).parts
        ):
            continue
        try:
            _append_reference(records, source=source, path=path)
        except (OSError, UnidentifiedImageError) as error:
            unreadable.append(
                {
                    "source": source,
                    "path": str(path.resolve()),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return records, unreadable


def _qamebi_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name, label in (("Benign", 0), ("Malignant", 1)):
        class_root = root / class_name
        originals = sorted(class_root.glob(f"* {class_name} Image.bmp"))
        expected = 109 if label == 0 else 123
        if len(originals) != expected:
            raise RuntimeError(
                f"{class_name} expected {expected} originals, found {len(originals)}"
            )
        for image_path in originals:
            prefix = image_path.name.removesuffix(f" {class_name} Image.bmp")
            lesion_path = class_root / f"{prefix} {class_name} Lesion.bmp"
            mask_path = class_root / f"{prefix} {class_name} Mask.tif"
            if not lesion_path.is_file() or not mask_path.is_file():
                raise RuntimeError(f"incomplete QAMEBI triplet for {image_path}")
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
            with Image.open(mask_path) as mask:
                mask_array = np.asarray(mask.convert("L"))
                mask_size = mask.size
            if image_size != mask_size:
                raise RuntimeError(
                    f"mask size mismatch for {image_path}: {image_size} vs {mask_size}"
                )
            if not np.any(mask_array > 0):
                raise RuntimeError(f"empty QAMEBI mask: {mask_path}")
            rows.append(
                {
                    "dataset": "QAMEBI",
                    "record_id": f"qamebi:{class_name.casefold()}:{int(prefix):03d}",
                    "patient_id": f"qamebi:record:{class_name.casefold()}:{int(prefix):03d}",
                    "patient_id_source": "dataset_record_not_verified_patient",
                    "patient_grouping_reliable": False,
                    "image_path": str(image_path.resolve()),
                    "label": label,
                    "label_source": "source_histopathology",
                    "is_original": True,
                    "sha256": _sha256_file(image_path),
                    "phash": _phash_file(image_path),
                    "mask_path": str(mask_path.resolve()),
                    "mask_sha256": _sha256_file(mask_path),
                    "mask_source": "source_radiologist_annotation",
                    "annotation_display_path": str(lesion_path.resolve()),
                    "width": image_size[0],
                    "height": image_size[1],
                    "mask_foreground_fraction": float(np.mean(mask_array > 0)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qamebi-root", type=Path, required=True)
    parser.add_argument("--benign-archive", type=Path, required=True)
    parser.add_argument("--malignant-archive", type=Path, required=True)
    parser.add_argument("--development-bundle", type=Path, required=True)
    parser.add_argument("--breast-manifest", type=Path, required=True)
    parser.add_argument("--breast-root", type=Path, required=True)
    parser.add_argument("--buscot-manifest", type=Path, required=True)
    parser.add_argument("--buscot-root", type=Path, required=True)
    parser.add_argument("--bcmid-root", type=Path, required=True)
    parser.add_argument("--bus-uclm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    parser.add_argument(
        "--keep-internal-near-record",
        action="append",
        default=[],
        help=(
            "Manually adjudicated record_id to retain from an internal pHash "
            "near-duplicate group. Repeat for multiple representatives."
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    references = _development_references(args.development_bundle)
    references.extend(
        _manifest_references(
            args.breast_manifest,
            source="external:BrEaST",
            root=args.breast_root,
            path_column="image_relpath",
        )
    )
    references.extend(
        _manifest_references(
            args.buscot_manifest,
            source="external:BUSCoT",
            root=args.buscot_root,
            path_column="image_path",
        )
    )
    unreadable_references: list[dict[str, str]] = []
    bcmid_references, bcmid_unreadable = _scan_references(
        args.bcmid_root,
        source="external:BCMID-ultrasound",
        required_path_component="Ultrasound",
    )
    references.extend(bcmid_references)
    unreadable_references.extend(bcmid_unreadable)
    bus_uclm_references, bus_uclm_unreadable = _scan_references(
        args.bus_uclm_root, source="external:BUS-UCLM"
    )
    references.extend(bus_uclm_references)
    unreadable_references.extend(bus_uclm_unreadable)
    reference_frame = pd.DataFrame(references).drop_duplicates(
        subset=["source", "path"], keep="first"
    )
    if reference_frame.empty:
        raise RuntimeError("no comparison images were found")
    reference_hashes = set(reference_frame["sha256"].astype(str))
    reference_phashes = _phash_bytes(reference_frame["phash"].astype(str).tolist())

    audit = pd.DataFrame(_qamebi_rows(args.qamebi_root))
    candidate_phashes = _phash_bytes(audit["phash"].astype(str).tolist())
    exact_flags: list[bool] = []
    near_flags: list[bool] = []
    minimum_distances: list[int] = []
    nearest_sources: list[str] = []
    nearest_paths: list[str] = []
    internal_exact: list[bool] = []
    internal_near: list[bool] = []
    for index, row in audit.iterrows():
        distances = _hamming_distances(str(row.phash), reference_phashes)
        nearest = int(np.argmin(distances))
        minimum = int(distances[nearest])
        exact_flags.append(str(row.sha256) in reference_hashes)
        near_flags.append(minimum <= args.near_duplicate_distance)
        minimum_distances.append(minimum)
        nearest_sources.append(str(reference_frame.iloc[nearest]["source"]))
        nearest_paths.append(str(reference_frame.iloc[nearest]["path"]))

        own_distances = _hamming_distances(str(row.phash), candidate_phashes)
        own_distances[index] = 255
        internal_exact.append(bool((audit["sha256"] == row.sha256).sum() > 1))
        internal_near.append(
            int(own_distances.min()) <= args.near_duplicate_distance
        )
    audit["exact_reference_overlap"] = exact_flags
    audit["near_reference_overlap"] = near_flags
    audit["minimum_reference_phash_distance"] = minimum_distances
    audit["nearest_reference_source"] = nearest_sources
    audit["nearest_reference_path"] = nearest_paths
    audit["internal_exact_duplicate"] = internal_exact
    audit["internal_near_duplicate"] = internal_near
    keep_internal_near = set(args.keep_internal_near_record)
    missing_kept_records = keep_internal_near - set(audit["record_id"])
    if missing_kept_records:
        raise ValueError(
            "unknown --keep-internal-near-record values: "
            f"{sorted(missing_kept_records)}"
        )
    non_near_kept_records = keep_internal_near - set(
        audit.loc[audit["internal_near_duplicate"], "record_id"]
    )
    if non_near_kept_records:
        raise ValueError(
            "manually retained records are not pHash near duplicates: "
            f"{sorted(non_near_kept_records)}"
        )
    audit["internal_near_duplicate_excluded"] = (
        audit["internal_near_duplicate"]
        & ~audit["record_id"].isin(keep_internal_near)
    )
    audit["eligible"] = ~(
        audit["exact_reference_overlap"]
        | audit["near_reference_overlap"]
        | audit["internal_exact_duplicate"]
        | audit["internal_near_duplicate_excluded"]
    )
    frozen = audit[audit["eligible"]].copy().reset_index(drop=True)
    if frozen.empty or frozen["label"].nunique() != 2:
        raise RuntimeError("overlap filtering left no valid binary benchmark")

    args.output_dir.mkdir(parents=True)
    audit.to_csv(args.output_dir / "audit_all.csv", index=False)
    frozen.to_csv(args.output_dir / "frozen_manifest.csv", index=False)
    reference_frame.to_csv(args.output_dir / "reference_inventory.csv", index=False)
    pd.DataFrame(
        unreadable_references, columns=["source", "path", "error"]
    ).to_csv(args.output_dir / "unreadable_references.csv", index=False)
    summary = {
        "dataset": "QAMEBI_BUID",
        "role": "locked_public_external_validation",
        "not_prospective_clinical_confirmation": True,
        "archives": {
            "benign": {
                "path": str(args.benign_archive.resolve()),
                "sha256": _sha256_file(args.benign_archive),
            },
            "malignant": {
                "path": str(args.malignant_archive.resolve()),
                "sha256": _sha256_file(args.malignant_archive),
            },
        },
        "near_duplicate_hamming_threshold": args.near_duplicate_distance,
        "audited_records": len(audit),
        "eligible_records": len(frozen),
        "eligible_class_counts": {
            str(key): int(value)
            for key, value in frozen["label"].value_counts().sort_index().items()
        },
        "patient_grouping_reliable": False,
        "analysis_unit": "lesion_record",
        "comparison_images": len(reference_frame),
        "unreadable_comparison_images": len(unreadable_references),
        "unreadable_comparison_image_paths": [
            record["path"] for record in unreadable_references
        ],
        "comparison_source_counts": {
            str(key): int(value)
            for key, value in reference_frame["source"].value_counts().items()
        },
        "excluded_exact_reference_overlap": int(
            audit["exact_reference_overlap"].sum()
        ),
        "excluded_near_reference_overlap": int(
            audit["near_reference_overlap"].sum()
        ),
        "excluded_internal_exact_duplicate": int(
            audit["internal_exact_duplicate"].sum()
        ),
        "excluded_internal_near_duplicate": int(
            audit["internal_near_duplicate_excluded"].sum()
        ),
        "internal_near_duplicate_members": int(
            audit["internal_near_duplicate"].sum()
        ),
        "manually_retained_internal_near_records": sorted(keep_internal_near),
        "minimum_reference_phash_distance": int(
            audit["minimum_reference_phash_distance"].min()
        ),
        "frozen_manifest_sha256": _sha256_file(
            args.output_dir / "frozen_manifest.csv"
        ),
        "model_predictions_seen": False,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
