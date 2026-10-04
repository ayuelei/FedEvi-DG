#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

import cv2
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fedusdg_mvp.manifest import (  # noqa: E402
    APPROVED_DOMAINS,
    REQUIRED_COLUMNS,
    audit_master_manifest,
    find_manifest_near_duplicates,
    resolve_path_beneath,
)


_WHU_DIRECTORY = "BUSI_WHU Breast Cancer Ultrasound Image  Dataset"
_WHU_LABEL = "Benign(0)\nmalignant(1)"
_UDIAT_DIRECTORY = Path("udiat/UDIAT_Dataset_B/UDIAT_Dataset_B")
_PUBLISHED_FILES = (
    "metadata_fedusdg_audited.csv",
    "source_mapping.csv",
    "exact_duplicate_decisions.csv",
    "excluded_rows.csv",
    "provenance_summary.json",
)
_DECISION_COLUMNS = (
    "sha256",
    "dataset",
    "raw_image_path",
    "label",
    "decision",
)
_EXCLUDED_COLUMNS = (
    "dataset",
    "raw_image_path",
    "patient_id",
    "label",
    "sha256",
    "reason",
)


def _format_identifier(value: object) -> str:
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("empty source identifier")
    return text


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{source} metadata missing columns: {missing}")
    null = [column for column in columns if frame[column].isna().any()]
    if null:
        raise ValueError(f"{source} metadata has missing values in: {null}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_row(
    *,
    dataset: str,
    image_path: Path,
    patient_id: str,
    patient_id_source: str,
    label: int,
    label_source: str,
    source_id: str,
    image_root: Path,
    path_name: str,
) -> dict[str, object]:
    image_path = resolve_path_beneath(
        image_root,
        image_path,
        path_name=path_name,
        root_name="images root",
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"authoritative raw image missing: {image_path}")
    return {
        "dataset": dataset,
        "image_path": str(image_path.resolve()),
        "patient_id": patient_id,
        "patient_id_source": patient_id_source,
        "label": int(label),
        "label_source": label_source,
        "is_original": True,
        "source_id": source_id,
    }


def _parse_busbra(raw_root: Path) -> list[dict[str, object]]:
    directory = resolve_path_beneath(
        raw_root,
        raw_root / "BUSBRA",
        path_name="BUSBRA directory",
        root_name="raw root",
    )
    metadata_path = resolve_path_beneath(
        raw_root,
        directory / "bus_data.csv",
        path_name="BUSBRA metadata path",
        root_name="raw root",
    )
    images_root = resolve_path_beneath(
        raw_root,
        directory / "Images",
        path_name="BUSBRA images directory",
        root_name="raw root",
    )
    frame = pd.read_csv(metadata_path)
    if "ID" not in frame:
        raise ValueError("BUSBRA metadata missing columns: ['ID']")
    _require_columns(frame, ("Case", "Pathology"), "BUSBRA")
    image_ids = frame["ID"].map(
        lambda value: "" if pd.isna(value) else str(value).strip()
    )
    if image_ids.eq("").any():
        raise ValueError("empty BUSBRA ID")
    if image_ids.duplicated().any():
        duplicates = sorted(image_ids[image_ids.duplicated(keep=False)].unique())
        raise ValueError(f"duplicate BUSBRA ID values: {duplicates}")
    for image_id in image_ids:
        resolve_path_beneath(
            images_root,
            images_root / f"{image_id}.png",
            path_name="BUSBRA image path",
            root_name="images root",
        )
    actual_png_names = {
        path.name
        for path in images_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    expected_png_names = {f"{image_id}.png" for image_id in image_ids}
    if actual_png_names != expected_png_names:
        raise ValueError(
            "BUSBRA PNG inventory mismatch; "
            f"missing={sorted(expected_png_names - actual_png_names)}, "
            f"extra={sorted(actual_png_names - expected_png_names)}"
        )
    rows = []
    for row_number, (_, source) in enumerate(frame.iterrows()):
        pathology = str(source.Pathology).strip().lower()
        if pathology not in {"benign", "malignant"}:
            raise ValueError(f"unsupported BUSBRA Pathology: {source.Pathology!r}")
        image_id = image_ids.iloc[row_number]
        case_id = _format_identifier(source.Case)
        rows.append(
            _base_row(
                dataset="BUSBRA",
                image_path=images_root / f"{image_id}.png",
                patient_id=f"busbra:case:{case_id}",
                patient_id_source="center_id",
                label=1 if pathology == "malignant" else 0,
                label_source="busbra_bus_data_pathology",
                source_id=image_id,
                image_root=images_root,
                path_name="BUSBRA image path",
            )
        )
    return rows


def _parse_busi_whu(raw_root: Path) -> list[dict[str, object]]:
    directory = resolve_path_beneath(
        raw_root,
        raw_root / _WHU_DIRECTORY,
        path_name="BUSI_WHU directory",
        root_name="raw root",
    )
    metadata_path = resolve_path_beneath(
        raw_root,
        directory / "Patient_infos_BUSI-WHU.xlsx",
        path_name="BUSI_WHU metadata path",
        root_name="raw root",
    )
    images_root = resolve_path_beneath(
        raw_root,
        directory / "img",
        path_name="BUSI_WHU images directory",
        root_name="raw root",
    )
    frame = pd.read_excel(metadata_path, sheet_name="Sheet1")
    _require_columns(frame, ("patient_index", _WHU_LABEL, "img_names"), "BUSI_WHU")
    by_name: dict[str, pd.Series] = {}
    for _, source in frame.iterrows():
        name = str(source.img_names).strip()
        if name in by_name:
            raise ValueError(f"duplicate BUSI_WHU img_names entry: {name}")
        by_name[name] = source
    raw_paths = sorted(images_root.glob("*.bmp"))
    if {path.name for path in raw_paths} != set(by_name):
        missing = sorted(set(by_name) - {path.name for path in raw_paths})
        extra = sorted({path.name for path in raw_paths} - set(by_name))
        raise ValueError(
            "BUSI_WHU image/metadata coverage mismatch; "
            f"missing={missing}, extra={extra}"
        )
    rows = []
    for image_path in raw_paths:
        source = by_name[image_path.name]
        label_value = source[_WHU_LABEL]
        if label_value not in (0, 1):
            raise ValueError(f"unsupported BUSI_WHU label: {label_value!r}")
        patient = _format_identifier(source.patient_index)
        rows.append(
            _base_row(
                dataset="BUSI_WHU",
                image_path=image_path,
                patient_id=f"busi_whu:patient:{patient}",
                patient_id_source="source_patient_id",
                label=int(label_value),
                label_source="busi_whu_patient_infos_label",
                source_id=Path(image_path.name).stem,
                image_root=images_root,
                path_name="BUSI_WHU image path",
            )
        )
    return rows


def _parse_gdph_sysucc(raw_root: Path) -> list[dict[str, object]]:
    directory = resolve_path_beneath(
        raw_root,
        raw_root / "GDPH&SYSUCC",
        path_name="GDPH_SYSUCC directory",
        root_name="raw root",
    )
    metadata_path = resolve_path_beneath(
        raw_root,
        directory / "BIRADS&FOLD.xlsx",
        path_name="GDPH_SYSUCC metadata path",
        root_name="raw root",
    )
    metadata = pd.read_excel(metadata_path)
    _require_columns(metadata, ("ID",), "GDPH_SYSUCC")
    metadata_ids = [str(value).strip() for value in metadata.ID]
    if len(metadata_ids) != len(set(metadata_ids)):
        raise ValueError("GDPH_SYSUCC metadata contains duplicate ID values")
    id_set = set(metadata_ids)
    encountered: set[str] = set()
    rows = []
    pattern = re.compile(r"^(benign|malignant)\((\d+)\)$", re.IGNORECASE)
    for dataset in ("GDPH", "SYSUCC"):
        images_root = resolve_path_beneath(
            raw_root,
            directory / dataset,
            path_name=f"{dataset} images directory",
            root_name="raw root",
        )
        for image_path in sorted(images_root.glob("*.png")):
            match = pattern.fullmatch(image_path.stem)
            if match is None:
                raise ValueError(
                    f"unsupported {dataset} filename label grammar: {image_path.name}"
                )
            source_id = image_path.stem
            if source_id not in id_set:
                raise ValueError(f"{dataset} raw ID absent from BIRADS&FOLD.xlsx: {source_id}")
            if source_id in encountered:
                raise ValueError(
                    f"GDPH_SYSUCC source ID appears in multiple raw paths: {source_id}"
                )
            encountered.add(source_id)
            label_text = match.group(1).lower()
            rows.append(
                _base_row(
                    dataset=dataset,
                    image_path=image_path,
                    patient_id=f"{dataset.lower()}:record:{source_id}",
                    patient_id_source="dataset_record",
                    label=1 if label_text == "malignant" else 0,
                    label_source="gdph_sysucc_filename_prefix",
                    source_id=source_id,
                    image_root=images_root,
                    path_name=f"{dataset} image path",
                )
            )
    if encountered != id_set:
        missing = sorted(id_set - encountered)
        raise ValueError(f"GDPH_SYSUCC XLSX IDs absent from raw directories: {missing}")
    return rows


def _parse_udiat(raw_root: Path) -> list[dict[str, object]]:
    directory = resolve_path_beneath(
        raw_root,
        raw_root / _UDIAT_DIRECTORY,
        path_name="UDIAT directory",
        root_name="raw root",
    )
    metadata_path = resolve_path_beneath(
        raw_root,
        directory / "DatasetB.xlsx",
        path_name="UDIAT metadata path",
        root_name="raw root",
    )
    images_root = resolve_path_beneath(
        raw_root,
        directory / "original",
        path_name="UDIAT images directory",
        root_name="raw root",
    )
    metadata = pd.read_excel(metadata_path, sheet_name="Hoja1")
    _require_columns(metadata, ("Image", "Type"), "UDIAT")
    rows = []
    names: set[str] = set()
    for _, source in metadata.iterrows():
        image_id = _format_identifier(source.Image)
        try:
            image_number = int(image_id)
        except ValueError as error:
            raise ValueError(f"unsupported UDIAT Image: {source.Image!r}") from error
        name = f"{image_number:06d}.png"
        if name in names:
            raise ValueError(f"duplicate UDIAT Image entry: {source.Image!r}")
        names.add(name)
        image_type = str(source.Type).strip().lower()
        if image_type not in {"benign", "malignant"}:
            raise ValueError(f"unsupported UDIAT Type: {source.Type!r}")
        rows.append(
            _base_row(
                dataset="UDIAT",
                image_path=images_root / name,
                patient_id=f"udiat:record:{image_id}",
                patient_id_source="dataset_record",
                label=1 if image_type == "malignant" else 0,
                label_source="udiat_datasetb_type",
                source_id=image_id,
                image_root=images_root,
                path_name="UDIAT image path",
            )
        )
    raw_names = {path.name for path in images_root.glob("*.png")}
    if raw_names != names:
        raise ValueError(
            "UDIAT image/metadata coverage mismatch; "
            f"missing={sorted(names - raw_names)}, extra={sorted(raw_names - names)}"
        )
    return sorted(rows, key=lambda row: str(row["image_path"]))


def _authoritative_rows(raw_root: Path) -> pd.DataFrame:
    rows = [
        *_parse_busbra(raw_root),
        *_parse_busi_whu(raw_root),
        *_parse_gdph_sysucc(raw_root),
        *_parse_udiat(raw_root),
    ]
    frame = pd.DataFrame(rows)
    if set(frame.dataset) != set(APPROVED_DOMAINS):
        raise ValueError("authoritative sources do not contain exactly the five approved domains")
    mixed = frame.groupby(["dataset", "patient_id"])["label"].nunique()
    if (mixed > 1).any():
        groups = [f"{dataset}/{patient}" for dataset, patient in mixed[mixed > 1].index]
        raise ValueError(f"mixed-label patient group in authoritative metadata: {groups}")
    return frame


def _processed_relative_path(row: pd.Series, index: int) -> Path:
    label = "malignant" if int(row.label) == 1 else "benign"
    client = {
        "BUSBRA": "client_busbra",
        "BUSI_WHU": "client_busi_whu",
        "GDPH": "client_gdph",
        "SYSUCC": "client_sysucc",
        "UDIAT": "client_udiat",
    }[row.dataset]
    prefix = {
        "BUSBRA": "busbra",
        "BUSI_WHU": "busiwhu",
        "GDPH": "gdph",
        "SYSUCC": "sysucc",
        "UDIAT": "udiat",
    }[row.dataset]
    return Path(client) / label / f"{prefix}_{index:04d}_{Path(row.image_path).stem}.png"


def _expected_legacy_mapping(authoritative: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset in APPROVED_DOMAINS:
        local = authoritative[authoritative.dataset.eq(dataset)]
        for index, (_, row) in enumerate(local.iterrows()):
            record = row.to_dict()
            record["legacy_file"] = str(_processed_relative_path(row, index))
            records.append(record)
    return pd.DataFrame(records)


def _verify_legacy(
    expected: pd.DataFrame,
    raw_root: Path,
    processed_root: Path,
    legacy_metadata: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    legacy = pd.read_csv(legacy_metadata)
    _require_columns(legacy, ("file", "client"), "legacy")
    legacy = legacy[legacy.client.astype(str).str.strip().isin(APPROVED_DOMAINS)].copy()
    if legacy.file.astype(str).duplicated().any():
        duplicates = legacy.loc[legacy.file.astype(str).duplicated(keep=False), "file"].tolist()
        raise ValueError(f"legacy mapping contains duplicate file entries: {duplicates}")
    expected_files = set(expected.legacy_file)
    actual_files = set(legacy.file.astype(str))
    if actual_files != expected_files:
        raise ValueError(
            "legacy mapping inventory mismatch; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    legacy_by_file = legacy.set_index(legacy.file.astype(str), drop=False)
    mappings = []
    non_square = 0

    def contained_candidate(root: Path, value: object) -> Path | None:
        try:
            return resolve_path_beneath(
                root,
                str(value),
                path_name="legacy source path",
                root_name=f"{root.name} root",
            )
        except ValueError:
            return None

    for _, row in expected.iterrows():
        evidence = legacy_by_file.loc[row.legacy_file]
        if str(evidence.client).strip() != row.dataset:
            raise ValueError(f"legacy mapping client mismatch for {row.legacy_file}")
        raw_path = Path(row.image_path).resolve()
        processed_path = resolve_path_beneath(
            processed_root,
            row.legacy_file,
            path_name="processed evidence path",
            root_name="processed root",
        )
        if not processed_path.is_file():
            raise ValueError(f"processed derivative missing: {processed_path}")
        legacy_source_status = "not_provided"
        if (
            "source" in evidence
            and pd.notna(evidence.source)
            and str(evidence.source).strip()
        ):
            raw_candidate = contained_candidate(raw_root, evidence.source)
            processed_candidate = contained_candidate(processed_root, evidence.source)
            if raw_candidate == raw_path:
                legacy_source_status = "raw_reference"
            elif processed_candidate == processed_path:
                legacy_source_status = "processed_self_reference"
            elif raw_candidate is not None:
                raise ValueError(
                    "legacy source path mismatches reconstructed raw path: "
                    f"{evidence.source}"
                )
            elif processed_candidate is not None:
                raise ValueError(
                    "legacy source path mismatches processed evidence path: "
                    f"{evidence.source}"
                )
            else:
                raise ValueError(
                    "legacy source path is outside raw and processed roots: "
                    f"{evidence.source}"
                )
        if "original_name" in evidence and pd.notna(evidence.original_name):
            expected_original_name = (
                processed_path.name
                if legacy_source_status == "processed_self_reference"
                else raw_path.name
            )
            if str(evidence.original_name) != expected_original_name:
                raise ValueError(f"legacy mapping original_name mismatch for {row.legacy_file}")
        source_pixels = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        processed_pixels = cv2.imread(str(processed_path), cv2.IMREAD_COLOR)
        if source_pixels is None:
            raise ValueError(f"undecodable raw image: {raw_path}")
        if processed_pixels is None:
            raise ValueError(f"processed derivative undecodable: {processed_path}")
        expected_pixels = cv2.cvtColor(source_pixels, cv2.COLOR_GRAY2RGB)
        expected_pixels = cv2.resize(
            expected_pixels, (512, 512), interpolation=cv2.INTER_LINEAR
        )
        expected_pixels = cv2.cvtColor(expected_pixels, cv2.COLOR_RGB2BGR)
        if processed_pixels.shape != (512, 512, 3) or not np.array_equal(
            processed_pixels, expected_pixels
        ):
            raise ValueError(f"processed derivative pixel mismatch: {processed_path}")
        height, width = source_pixels.shape
        non_square += int(width != height)
        mapping = row.to_dict()
        mapping.update(
            {
                "raw_image_path": str(raw_path),
                "processed_path": str(processed_path),
                "source_width": int(width),
                "source_height": int(height),
                "mapping_verified": True,
                "raw_path_verified": True,
                "label_verified": True,
                "pixel_verified": True,
                "legacy_source_status": legacy_source_status,
            }
        )
        if "near_duplicate_resolution" in evidence and pd.notna(
            evidence.near_duplicate_resolution
        ):
            mapping["near_duplicate_resolution"] = str(
                evidence.near_duplicate_resolution
            ).strip()
        mappings.append(mapping)
    mapping_frame = pd.DataFrame(mappings)
    return mapping_frame, {
        "non_square_source_rows": non_square,
        "square_source_rows": len(mapping_frame) - non_square,
        "stretched_to_square_rows": non_square,
    }


def _apply_exact_duplicate_policy(
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    work = mapping.copy()
    work["sha256"] = work.raw_image_path.map(lambda value: _sha256_file(Path(value)))
    excluded_indices: set[int] = set()
    decisions = []
    duplicate_groups = 0
    duplicate_rows = 0
    conflict_groups = 0
    for sha256, group in work.groupby("sha256", sort=True):
        if len(group) < 2:
            continue
        duplicate_groups += 1
        duplicate_rows += len(group)
        ordered = group.sort_values("raw_image_path", kind="stable")
        if ordered.label.nunique() == 1:
            for offset, (index, row) in enumerate(ordered.iterrows()):
                decision = (
                    "retain_same_label_duplicate"
                    if offset == 0
                    else "exclude_same_label_duplicate"
                )
                if offset:
                    excluded_indices.add(index)
                decisions.append(
                    {
                        "sha256": sha256,
                        "dataset": row.dataset,
                        "raw_image_path": row.raw_image_path,
                        "label": int(row.label),
                        "decision": decision,
                    }
                )
        else:
            conflict_groups += 1
            for index, row in ordered.iterrows():
                excluded_indices.add(index)
                decisions.append(
                    {
                        "sha256": sha256,
                        "dataset": row.dataset,
                        "raw_image_path": row.raw_image_path,
                        "label": int(row.label),
                        "decision": "exclude_conflicting_label_duplicate",
                    }
                )
    excluded_records = []
    decision_by_path = {
        record["raw_image_path"]: record["decision"] for record in decisions
    }
    for index in sorted(excluded_indices, key=lambda value: work.at[value, "raw_image_path"]):
        row = work.loc[index]
        excluded_records.append(
            {
                "dataset": row.dataset,
                "raw_image_path": row.raw_image_path,
                "patient_id": row.patient_id,
                "label": int(row.label),
                "sha256": row.sha256,
                "reason": decision_by_path[row.raw_image_path],
            }
        )
    retained = work.drop(index=list(excluded_indices)).copy()
    decisions_frame = pd.DataFrame(decisions, columns=_DECISION_COLUMNS)
    excluded_frame = pd.DataFrame(excluded_records, columns=_EXCLUDED_COLUMNS)
    return retained, decisions_frame, excluded_frame, {
        "groups": duplicate_groups,
        "rows": duplicate_rows,
        "label_conflict_groups": conflict_groups,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(
    mapping: pd.DataFrame,
    audited: pd.DataFrame,
    excluded: pd.DataFrame,
    geometry: dict[str, int],
    duplicates: dict[str, int],
) -> dict[str, object]:
    domains: dict[str, object] = {}
    for domain in APPROVED_DOMAINS:
        local = audited[audited.dataset.eq(domain)]
        domains[domain] = {
            "rows": int(len(local)),
            "class_counts": {
                str(int(label)): int(count)
                for label, count in local.label.value_counts().sort_index().items()
            },
            "patient_id_source": {
                str(source): int(count)
                for source, count in local.patient_id_source.value_counts().sort_index().items()
            },
            "unique_patient_groups": int(local.patient_id.nunique()),
        }
    candidates = find_manifest_near_duplicates(audited)
    return {
        "schema_version": 1,
        "input_rows": int(len(mapping)),
        "retained_rows": int(len(audited)),
        "excluded_rows": int(len(excluded)),
        "domains": domains,
        "validation": {
            "mapping_verified_rows": int(mapping.mapping_verified.sum()),
            "raw_path_verified_rows": int(mapping.raw_path_verified.sum()),
            "label_verified_rows": int(mapping.label_verified.sum()),
            "pixel_verified_rows": int(mapping.pixel_verified.sum()),
        },
        "geometry": geometry,
        "exact_duplicates": duplicates,
        "near_duplicate_candidate_count": int(len(candidates)),
        "legacy_source_status_counts": {
            str(status): int(count)
            for status, count in mapping.legacy_source_status.value_counts()
            .sort_index()
            .items()
        },
    }


def _validate_classes(frame: pd.DataFrame) -> None:
    for domain in APPROVED_DOMAINS:
        labels = set(frame.loc[frame.dataset.eq(domain), "label"])
        if labels != {0, 1}:
            raise ValueError(f"retained authoritative domain lacks both classes: {domain}")


def _publish_complete(staging: Path) -> None:
    current = {path.name for path in staging.iterdir()}
    if current != set(_PUBLISHED_FILES):
        raise RuntimeError(
            f"provenance staging inventory mismatch; expected={sorted(_PUBLISHED_FILES)}, "
            f"actual={sorted(current)}"
        )
    hashes = {name: _sha256_file(staging / name) for name in sorted(_PUBLISHED_FILES)}
    _write_json(staging / "COMPLETE", {"files": hashes})
    expected = {*_PUBLISHED_FILES, "COMPLETE"}
    if {path.name for path in staging.iterdir()} != expected:
        raise RuntimeError("provenance COMPLETE inventory mismatch")


def build_provenance_bundle(
    *,
    raw_root: str | Path,
    processed_root: str | Path,
    legacy_metadata: str | Path,
    output_root: str | Path,
) -> Path:
    """Reconstruct, verify, audit, and atomically publish source provenance."""
    raw_root = Path(raw_root).expanduser().resolve()
    processed_root = Path(processed_root).expanduser().resolve()
    legacy_input = Path(legacy_metadata).expanduser()
    if not legacy_input.is_absolute():
        legacy_input = Path.cwd() / legacy_input
    legacy_metadata = resolve_path_beneath(
        processed_root,
        legacy_input,
        path_name="legacy metadata path",
        root_name="processed root",
    )
    output_root_input = Path(output_root).expanduser()
    if not output_root_input.is_absolute():
        output_root_input = Path.cwd() / output_root_input
    output_root_lexical = output_root_input.absolute()
    if os.path.lexists(output_root_lexical):
        raise FileExistsError(f"output root already exists: {output_root_lexical}")
    output_root = output_root_input.resolve(strict=False)
    if os.path.lexists(output_root):
        raise FileExistsError(f"output root already exists: {output_root}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root does not exist: {raw_root}")
    if not processed_root.is_dir():
        raise FileNotFoundError(f"processed root does not exist: {processed_root}")
    if not legacy_metadata.is_file():
        raise FileNotFoundError(f"legacy metadata does not exist: {legacy_metadata}")
    for input_name, input_root in (
        ("raw root", raw_root),
        ("processed root", processed_root),
    ):
        if (
            output_root == input_root
            or output_root.is_relative_to(input_root)
            or input_root.is_relative_to(output_root)
        ):
            raise ValueError(f"output root overlaps {input_name}: {output_root}")
    if output_root == legacy_metadata:
        raise ValueError(f"output root overlaps legacy metadata: {output_root}")

    authoritative = _authoritative_rows(raw_root)
    expected = _expected_legacy_mapping(authoritative)
    mapping, geometry = _verify_legacy(
        expected, raw_root, processed_root, legacy_metadata
    )
    retained, decisions, excluded, duplicate_stats = _apply_exact_duplicate_policy(mapping)
    _validate_classes(retained)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        # SHA-256 is established from the raw bytes above; pHash is deliberately
        # left for the production manifest auditor to compute.
        candidate_columns = [column for column in REQUIRED_COLUMNS if column != "phash"]
        if "near_duplicate_resolution" in retained:
            candidate_columns.append("near_duplicate_resolution")
        candidate = retained
        candidate_path = staging / ".candidate.csv"
        candidate.loc[:, candidate_columns].to_csv(candidate_path, index=False)
        audited = audit_master_manifest(
            candidate_path,
            data_root=raw_root,
            domains=APPROVED_DOMAINS,
        )
        candidate_path.unlink()
        _validate_classes(audited)

        audited.to_csv(staging / "metadata_fedusdg_audited.csv", index=False)
        mapping_output = mapping.drop(columns=["image_path"], errors="ignore").sort_values(
            ["dataset", "raw_image_path"], kind="stable"
        )
        mapping_output.to_csv(staging / "source_mapping.csv", index=False)
        decisions.to_csv(staging / "exact_duplicate_decisions.csv", index=False)
        excluded.to_csv(staging / "excluded_rows.csv", index=False)
        _write_json(
            staging / "provenance_summary.json",
            _summary(mapping, audited, excluded, geometry, duplicate_stats),
        )
        _publish_complete(staging)
        if os.path.lexists(output_root):
            raise FileExistsError(f"output root already exists: {output_root}")
        staging.rename(output_root)
    except BaseException:
        if os.path.lexists(staging):
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging)
            else:
                staging.unlink()
        raise
    return output_root.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict original-source provenance bundle for FedUS-DG."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--legacy-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_provenance_bundle(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        legacy_metadata=args.legacy_metadata,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
