from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedShuffleSplit

from .config import ExperimentConfig
from .provenance import atomic_write_json, canonical_hash


APPROVED_DOMAINS = ("BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT")
FIXED_SPLIT_SEED = 20260728
FIXED_VAL_FRACTION = 0.15
REQUIRED_COLUMNS = (
    "dataset",
    "image_path",
    "patient_id",
    "patient_id_source",
    "label",
    "label_source",
    "is_original",
    "sha256",
    "phash",
)
APPROVED_PATIENT_ID_SOURCES = (
    "metadata",
    "source_patient_id",
    "original_source_id",
    "center_id",
    "dataset_record",
)
RELIABLE_PATIENT_ID_SOURCES = frozenset(
    {"metadata", "source_patient_id", "center_id"}
)
_RESOLUTION_COLUMNS = ("near_duplicate_resolution",)
_TRUE_VALUES = {"1", "true", "yes", "y"}
_FALSE_VALUES = {"0", "false", "no", "n"}
_BIT_COUNTS = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def _dct_matrix(size: int = 32) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (positions + 0.5) * frequencies / size)
    matrix[0] *= np.sqrt(1.0 / size)
    matrix[1:] *= np.sqrt(2.0 / size)
    return matrix


_DCT_32 = _dct_matrix()


def _rows_csv(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        columns = [column for column in columns if column in df.columns]
        df = df.loc[:, columns]
    return df.to_csv(index=False)


def _normalize_labels(df: pd.DataFrame) -> pd.Series:
    normalized = df["label"].map(
        lambda value: value.strip().lower() if isinstance(value, str) else value
    )
    labels = normalized.map(
        {"benign": 0, "malignant": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    )
    illegal = labels.isna()
    if illegal.any():
        raise ValueError(
            "illegal labels (expected benign/malignant or 0/1):\n"
            + _rows_csv(df.loc[illegal], ["dataset", "image_path", "label"])
        )
    return labels.astype("int64")


def _normalize_original_flags(df: pd.DataFrame) -> pd.Series:
    def normalize(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _TRUE_VALUES:
                return True
            if lowered in _FALSE_VALUES:
                return False
        return None

    flags = df["is_original"].map(normalize)
    illegal = flags.isna()
    if illegal.any():
        raise ValueError(
            "illegal is_original values:\n"
            + _rows_csv(
                df.loc[illegal], ["dataset", "image_path", "is_original"]
            )
        )
    return flags.astype(bool)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path_beneath(
    root: str | Path,
    candidate: str | Path,
    *,
    path_name: str,
    root_name: str,
) -> Path:
    """Resolve a path and reject traversal, symlink, or absolute escape."""
    resolved_root = Path(root).expanduser().resolve()
    candidate_path = Path(candidate).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = resolved_root / candidate_path
    resolved_candidate = candidate_path.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"{path_name} escapes {root_name}: {resolved_candidate}"
        ) from error
    return resolved_candidate


def _perceptual_hash(path: Path) -> str:
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L").resize(
                (32, 32), Image.Resampling.LANCZOS
            )
            pixels = np.asarray(grayscale, dtype=np.float64)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"undecodable image: {path}: {error}") from error
    coefficients = _DCT_32 @ pixels @ _DCT_32.T
    low_frequency = coefficients[:8, :8]
    threshold = np.median(low_frequency.reshape(-1)[1:])
    bits = low_frequency.reshape(-1) > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _canonical_patient_id(value) -> str:
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip()
    ).casefold()


def _validate_patient_metadata(df: pd.DataFrame) -> None:
    missing = df["patient_id"].isna() | df["patient_id"].astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError(
            "missing patient_id; provide a reliable patient or recoverable original-source ID:\n"
            + _rows_csv(
                df.loc[missing],
                ["dataset", "image_path", "patient_id", "patient_id_source"],
            )
        )
    canonical_ids = df["patient_id"].map(_canonical_patient_id)
    df["patient_id"] = canonical_ids
    missing_source = df["patient_id_source"].isna() | (
        df["patient_id_source"].astype(str).str.strip().eq("")
    )
    if missing_source.any():
        raise ValueError(
            "missing patient_id_source:\n"
            + _rows_csv(
                df.loc[missing_source],
                ["dataset", "image_path", "patient_id", "patient_id_source"],
            )
        )
    normalized_sources = (
        df["patient_id_source"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )
    unapproved = ~normalized_sources.isin(APPROVED_PATIENT_ID_SOURCES)
    if unapproved.any():
        raise ValueError(
            "unapproved patient_id_source; allowed sources are "
            f"{APPROVED_PATIENT_ID_SOURCES}:\n"
            + _rows_csv(
                df.loc[unapproved],
                ["dataset", "image_path", "patient_id", "patient_id_source"],
            )
        )
    df["patient_id_source"] = normalized_sources
    conflicting = (
        df.groupby(["dataset", "patient_id"])["patient_id_source"].nunique() > 1
    )
    if conflicting.any():
        bad_groups = set(conflicting[conflicting].index)
        bad_mask = [
            (dataset, patient_id) in bad_groups
            for dataset, patient_id in zip(df["dataset"], df["patient_id"])
        ]
        bad_rows = df.loc[bad_mask]
        raise ValueError(
            "conflicting patient_id_source for canonical patient group:\n"
            + _rows_csv(
                bad_rows,
                ["dataset", "image_path", "patient_id", "patient_id_source"],
            )
        )


def patient_id_source_is_reliable(value: object) -> bool:
    normalized = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    return normalized in RELIABLE_PATIENT_ID_SOURCES


def audit_master_manifest(
    config_or_metadata_csv: ExperimentConfig | str | Path,
    data_root: str | Path | None = None,
    domains: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Normalize and strictly audit one metadata table before any LODO split."""
    if isinstance(config_or_metadata_csv, ExperimentConfig):
        config = config_or_metadata_csv
        metadata_csv = config.metadata_csv
        data_root_path = config.data_root
        approved_domains = tuple(config.domains)
    else:
        metadata_csv = Path(config_or_metadata_csv)
        if data_root is None:
            raise TypeError("data_root is required when no ExperimentConfig is supplied")
        data_root_path = Path(data_root)
        approved_domains = tuple(domains or APPROVED_DOMAINS)
    if set(approved_domains) != set(APPROVED_DOMAINS) or len(approved_domains) != 5:
        raise ValueError(
            f"audit requires exactly the five approved domains: {APPROVED_DOMAINS}"
        )

    metadata_csv = Path(metadata_csv).resolve()
    data_root_path = Path(data_root_path).resolve()
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata CSV does not exist: {metadata_csv}")
    if not data_root_path.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root_path}")

    df = pd.read_csv(metadata_csv)
    legacy_renames = {}
    if "dataset" not in df and "client" in df:
        legacy_renames["client"] = "dataset"
    if "image_path" not in df and "file" in df:
        legacy_renames["file"] = "image_path"
    df = df.rename(columns=legacy_renames)
    missing_base = [
        column
        for column in (
            "dataset",
            "image_path",
            "patient_id",
            "patient_id_source",
            "label",
            "label_source",
            "is_original",
        )
        if column not in df
    ]
    if missing_base:
        raise ValueError(f"metadata missing required columns: {missing_base}")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df = df[df["dataset"].isin(approved_domains)].copy()
    if df.empty:
        raise ValueError(f"metadata contains no approved domains: {approved_domains}")
    missing_domains = sorted(set(approved_domains) - set(df["dataset"]))
    if missing_domains:
        raise ValueError(f"metadata missing approved domains: {missing_domains}")

    df["label"] = _normalize_labels(df)
    df["is_original"] = _normalize_original_flags(df)
    df = df[df["is_original"]].copy()
    if df.empty:
        raise ValueError("metadata contains no approved original files")
    remaining_domains = set(df["dataset"])
    if remaining_domains != set(APPROVED_DOMAINS):
        missing_after_filter = sorted(set(APPROVED_DOMAINS) - remaining_domains)
        raise ValueError(
            "after original-file filtering, missing approved domains: "
            f"{missing_after_filter}"
        )

    missing_label_source = df["label_source"].isna() | (
        df["label_source"].astype(str).str.strip().eq("")
    )
    if missing_label_source.any():
        raise ValueError(
            "missing label_source:\n"
            + _rows_csv(
                df.loc[missing_label_source],
                ["dataset", "image_path", "label", "label_source"],
            )
        )
    df["label_source"] = df["label_source"].astype(str).str.strip().str.casefold()
    _validate_patient_metadata(df)

    raw_paths = df["image_path"]
    missing_path_values = raw_paths.isna() | raw_paths.astype(str).str.strip().eq("")
    if missing_path_values.any():
        raise ValueError(
            "missing image paths:\n"
            + _rows_csv(df.loc[missing_path_values], ["dataset", "image_path"])
        )
    resolved_paths = []
    for raw_path in raw_paths.astype(str):
        resolved_paths.append(
            resolve_path_beneath(
                data_root_path,
                raw_path,
                path_name="image path",
                root_name="data root",
            )
        )
    df["image_path"] = [str(path) for path in resolved_paths]
    missing_files = pd.Series(
        [not path.is_file() for path in resolved_paths], index=df.index
    )
    if missing_files.any():
        raise ValueError(
            "missing image files:\n"
            + _rows_csv(df.loc[missing_files], ["dataset", "image_path"])
        )

    if "sha256" not in df:
        df["sha256"] = pd.NA
    if "phash" not in df:
        df["phash"] = pd.NA
    for row_index, path in zip(df.index, resolved_paths):
        sha256 = df.at[row_index, "sha256"]
        if pd.isna(sha256) or not str(sha256).strip():
            df.at[row_index, "sha256"] = _sha256_file(path)
        phash = df.at[row_index, "phash"]
        if pd.isna(phash) or not str(phash).strip():
            df.at[row_index, "phash"] = _perceptual_hash(path)
        else:
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as error:
                raise ValueError(f"undecodable image: {path}: {error}") from error

    df["sha256"] = df["sha256"].astype(str).str.lower()
    invalid_sha = ~df["sha256"].str.fullmatch(r"[0-9a-f]{64}")
    if invalid_sha.any():
        raise ValueError(
            "invalid sha256 values:\n"
            + _rows_csv(df.loc[invalid_sha], ["dataset", "image_path", "sha256"])
        )
    df["phash"] = df["phash"].astype(str).str.lower()
    invalid_phash = ~df["phash"].str.fullmatch(r"[0-9a-f]{16}")
    if invalid_phash.any():
        raise ValueError(
            "invalid 64-bit phash values:\n"
            + _rows_csv(df.loc[invalid_phash], ["dataset", "image_path", "phash"])
        )

    validate_no_cross_domain_duplicates(df)
    if "birads_ordinal" in df.columns:
        _validate_birads_ordinal(df["birads_ordinal"])
    retained_columns = list(REQUIRED_COLUMNS) + [
        column for column in (*_RESOLUTION_COLUMNS, "birads_ordinal") if column in df
    ]
    return (
        df.loc[:, retained_columns]
        .sort_values(["dataset", "image_path"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_birads_ordinal(column: pd.Series) -> None:
    """BI-RADS ordinal levels: positive floats or missing; NaN marks absent."""
    numeric = pd.to_numeric(column, errors="coerce")
    present = numeric.notna()
    original_present = column.notna() & (column.astype(str).str.strip() != "")
    unmappable = original_present & ~present
    if unmappable.any():
        raise ValueError(
            f"birads_ordinal contains non-numeric values: {sorted(column[unmappable].astype(str).unique())[:5]}"
        )
    if (numeric[present] <= 0).any():
        raise ValueError("birads_ordinal levels must be positive ranks")
    if not np.isfinite(numeric[present].to_numpy(dtype=float)).all():
        raise ValueError("birads_ordinal levels must be finite")


def split_protocol_config(target_domain: str) -> dict[str, object]:
    """Return the complete, canonical credential for the fixed LODO split."""
    if target_domain not in APPROVED_DOMAINS:
        raise ValueError(f"target domain is not approved: {target_domain}")
    return {
        "schema_version": 1,
        "protocol": "five_domain_lodo",
        "domains": list(APPROVED_DOMAINS),
        "target_domain": target_domain,
        "source_domains": sorted(set(APPROVED_DOMAINS) - {target_domain}),
        "split_seed": FIXED_SPLIT_SEED,
        "val_fraction": FIXED_VAL_FRACTION,
        "grouping_columns": ["dataset", "patient_id"],
        "stratification": "patient_label",
        "source_seed_rule": "split_seed_plus_sorted_source_domain_index",
    }


def validate_no_cross_domain_duplicates(df: pd.DataFrame) -> None:
    """Hard-stop on any repeated exact image bytes, within or across domains."""
    duplicates = df[df.duplicated("sha256", keep=False)].sort_values("sha256")
    if duplicates.empty:
        return
    cross_domain = (duplicates.groupby("sha256")["dataset"].nunique() > 1).any()
    prefix = "cross-domain exact duplicate" if cross_domain else "within-domain exact duplicate"
    raise ValueError(prefix + ":\n" + duplicates.to_csv(index=False))


def _near_duplicate_pair_id(left: pd.Series, right: pd.Series) -> str:
    endpoints = sorted(
        (
            {
                "sha256": str(left["sha256"]).strip().lower(),
                "image_path": str(left["image_path"]),
            },
            {
                "sha256": str(right["sha256"]).strip().lower(),
                "image_path": str(right["image_path"]),
            },
        ),
        key=lambda endpoint: (endpoint["sha256"], endpoint["image_path"]),
    )
    return canonical_hash({"near_duplicate_endpoints": endpoints})


def _approved_pair_ids(value) -> set[str]:
    if pd.isna(value):
        return set()
    approvals = set()
    for token in str(value).split(";"):
        token = token.strip().lower()
        if token.startswith("approved:"):
            pair_id = token.removeprefix("approved:")
            if re.fullmatch(r"[0-9a-f]{64}", pair_id):
                approvals.add(pair_id)
    return approvals


def _drop_pair_ids(value) -> set[str]:
    """Human drop marks: both ends of a pair carry ``drop:<pair_id>``."""
    if pd.isna(value):
        return set()
    drops = set()
    for token in str(value).split(";"):
        token = token.strip().lower()
        if token.startswith("drop:"):
            pair_id = token.removeprefix("drop:")
            if re.fullmatch(r"[0-9a-f]{64}", pair_id):
                drops.add(pair_id)
    return drops


def apply_near_duplicate_dedup(
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically keep one row per human-confirmed same-lesion group.

    Rows whose ``near_duplicate_resolution`` carries matching ``drop:<pair_id>``
    marks on both ends are grouped by union-find over those pair edges. Each
    connected component keeps the lexicographically smallest ``image_path``
    representative; the rest are excluded from the manifest (files stay on
    disk). Components spanning datasets or labels hard-fail for manual
    resolution. Every exclusion is reported with its reason.
    """
    frame = master.reset_index(drop=True)
    if "near_duplicate_resolution" not in frame.columns:
        return frame.copy(), pd.DataFrame(
            columns=[
                "dataset",
                "image_path",
                "label",
                "representative_image_path",
                "drop_pair_ids",
                "reason",
            ]
        )

    rows_by_pair: dict[str, list[int]] = {}
    for index, value in zip(frame.index, frame["near_duplicate_resolution"], strict=True):
        for pair_id in _drop_pair_ids(value):
            rows_by_pair.setdefault(pair_id, []).append(int(index))

    edges: list[tuple[int, int, str]] = []
    for pair_id, indices in sorted(rows_by_pair.items()):
        if len(indices) != 2:
            raise ValueError(
                f"drop mark must be written on exactly both ends of pair {pair_id}; "
                f"found {len(indices)} rows"
            )
        edges.append((indices[0], indices[1], pair_id))
    if not edges:
        return frame.copy(), pd.DataFrame(
            columns=[
                "dataset",
                "image_path",
                "label",
                "representative_image_path",
                "drop_pair_ids",
                "reason",
            ]
        )

    parent = list(range(len(frame)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right, _ in edges:
        parent[max(find(left), find(right))] = min(find(left), find(right))

    components: dict[int, list[int]] = {}
    for node in range(len(frame)):
        components.setdefault(find(node), []).append(node)

    excluded: list[dict] = []
    drop_indices: set[int] = set()
    for members in components.values():
        if len(members) < 2:
            continue
        paths = frame.loc[members, "image_path"].astype(str)
        datasets = frame.loc[members, "dataset"].astype(str)
        labels = frame.loc[members, "label"]
        if datasets.nunique() > 1:
            raise ValueError(
                "drop-marked group spans datasets; resolve manually: "
                + ", ".join(sorted(paths))
            )
        if labels.nunique() > 1:
            # mirrors the exact-duplicate precedent: a visually duplicated
            # group with conflicting labels must not inject an arbitrarily
            # labeled copy, so every member is excluded
            for member in members:
                drop_indices.add(member)
                excluded.append(
                    {
                        "dataset": str(frame.at[member, "dataset"]),
                        "image_path": str(frame.at[member, "image_path"]),
                        "label": int(frame.at[member, "label"]),
                        "representative_image_path": "",
                        "drop_pair_ids": ";".join(
                            sorted(
                                pair_id
                                for pair_id in _drop_pair_ids(
                                    frame.at[member, "near_duplicate_resolution"]
                                )
                            )
                        ),
                        "reason": "exclude_conflicting_label_near_duplicate",
                    }
                )
            continue
        representative = sorted(zip(paths, members, strict=True))[0][1]
        pair_ids_by_row = {
            member: ";".join(
                sorted(
                    pair_id
                    for pair_id in _drop_pair_ids(frame.at[member, "near_duplicate_resolution"])
                )
            )
            for member in members
        }
        for member in members:
            if member == representative:
                continue
            drop_indices.add(member)
            excluded.append(
                {
                    "dataset": str(frame.at[member, "dataset"]),
                    "image_path": str(frame.at[member, "image_path"]),
                    "label": int(frame.at[member, "label"]),
                    "representative_image_path": str(frame.at[representative, "image_path"]),
                    "drop_pair_ids": pair_ids_by_row[member],
                    "reason": "exclude_same_lesion_near_duplicate",
                }
            )

    reduced = frame.drop(index=sorted(drop_indices)).reset_index(drop=True)
    decisions = pd.DataFrame(excluded)
    if decisions.empty:
        decisions = pd.DataFrame(
            columns=[
                "dataset",
                "image_path",
                "label",
                "representative_image_path",
                "drop_pair_ids",
                "reason",
            ]
        )
    return reduced, decisions


def _find_near_duplicates(
    df: pd.DataFrame,
    max_hamming_distance: int = 4,
    output_path: str | Path | None = None,
    *,
    include_within_domain: bool,
) -> pd.DataFrame:
    if not 0 <= max_hamming_distance <= 64:
        raise ValueError("max_hamming_distance must be in [0, 64]")
    required = {"dataset", "image_path", "sha256", "phash"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manifest missing near-duplicate columns: {missing}")

    work = df.reset_index(drop=False).rename(columns={"index": "manifest_index"})
    try:
        hashes = np.array(
            [int(str(value), 16) for value in work["phash"]], dtype=np.uint64
        )
    except (TypeError, ValueError) as error:
        raise ValueError("phash values must be 16-character hexadecimal strings") from error
    if not work["phash"].astype(str).str.fullmatch(r"[0-9A-Fa-f]{16}").all():
        raise ValueError("phash values must be 16-character hexadecimal strings")

    approvals = (
        work["near_duplicate_resolution"].map(_approved_pair_ids).tolist()
        if "near_duplicate_resolution" in work
        else [set() for _ in range(len(work))]
    )
    records = []
    domains = sorted(work["dataset"].astype(str).unique())
    for left_domain_index, left_domain in enumerate(domains):
        left_positions = np.flatnonzero(work["dataset"].astype(str).eq(left_domain))
        first_right_domain = (
            left_domain_index if include_within_domain else left_domain_index + 1
        )
        for right_domain in domains[first_right_domain:]:
            right_positions = np.flatnonzero(
                work["dataset"].astype(str).eq(right_domain)
            )
            same_domain = left_domain == right_domain
            for left_offset, left_position in enumerate(left_positions):
                comparison_positions = (
                    right_positions[left_offset + 1 :]
                    if same_domain
                    else right_positions
                )
                if len(comparison_positions) == 0:
                    continue
                right_hashes = hashes[comparison_positions]
                xors = np.bitwise_xor(hashes[left_position], right_hashes)
                byte_values = xors.view(np.uint8).reshape(-1, 8)
                distances = _BIT_COUNTS[byte_values].sum(axis=1)
                for right_offset in np.flatnonzero(
                    distances <= max_hamming_distance
                ):
                    right_position = comparison_positions[right_offset]
                    pair_id = _near_duplicate_pair_id(
                        work.iloc[left_position], work.iloc[right_position]
                    )
                    records.append(
                        {
                            "pair_id": pair_id,
                            "left_index": work.at[left_position, "manifest_index"],
                            "right_index": work.at[right_position, "manifest_index"],
                            "left_dataset": left_domain,
                            "right_dataset": right_domain,
                            "left_image_path": work.at[left_position, "image_path"],
                            "right_image_path": work.at[right_position, "image_path"],
                            "left_phash": f"{int(hashes[left_position]):016x}",
                            "right_phash": f"{int(hashes[right_position]):016x}",
                            "hamming_distance": int(distances[right_offset]),
                            "resolved": bool(
                                pair_id in approvals[left_position]
                                and pair_id in approvals[right_position]
                            ),
                        }
                    )
    columns = [
        "pair_id",
        "left_index",
        "right_index",
        "left_dataset",
        "right_dataset",
        "left_image_path",
        "right_image_path",
        "left_phash",
        "right_phash",
        "hamming_distance",
        "resolved",
    ]
    report = pd.DataFrame.from_records(records, columns=columns)
    if not report.empty:
        report = report.sort_values(
            ["hamming_distance", "left_dataset", "right_dataset", "left_image_path"],
            kind="stable",
        ).reset_index(drop=True)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
    return report


def find_cross_domain_near_duplicates(
    df: pd.DataFrame,
    max_hamming_distance: int = 4,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Report perceptual-hash candidates between different datasets only."""
    return _find_near_duplicates(
        df,
        max_hamming_distance=max_hamming_distance,
        output_path=output_path,
        include_within_domain=False,
    )


def find_manifest_near_duplicates(
    df: pd.DataFrame,
    max_hamming_distance: int = 4,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Report all distinct perceptual-hash candidates, including same-domain pairs."""
    return _find_near_duplicates(
        df,
        max_hamming_distance=max_hamming_distance,
        output_path=output_path,
        include_within_domain=True,
    )


def build_lodo_frames(
    df: pd.DataFrame,
    target_domain: str,
    split_seed: int,
    val_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    dataset_set = set(df["dataset"])
    if dataset_set != set(APPROVED_DOMAINS):
        raise ValueError(
            "LODO split requires exactly five approved domains "
            f"{APPROVED_DOMAINS}; got {sorted(dataset_set)}"
        )
    if target_domain not in APPROVED_DOMAINS:
        raise ValueError(f"target domain is not approved: {target_domain}")
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    if split_seed != FIXED_SPLIT_SEED or val_fraction != FIXED_VAL_FRACTION:
        raise ValueError(
            "LODO split protocol requires split_seed=20260728 and val_fraction=0.15"
        )

    df = df.copy()
    _validate_patient_metadata(df)
    target = df[df.dataset == target_domain].copy()
    source = df[df.dataset != target_domain].copy()
    expected_source_domains = set(APPROVED_DOMAINS) - {target_domain}
    actual_source_domains = set(source["dataset"])
    if actual_source_domains != expected_source_domains or len(actual_source_domains) != 4:
        raise ValueError(
            "LODO split requires exactly four approved source domains; "
            f"got {sorted(actual_source_domains)}"
        )
    train_parts, val_parts = [], []
    for domain_index, domain in enumerate(sorted(source.dataset.unique())):
        local = source[source.dataset == domain]
        if local["patient_id"].isna().any():
            raise ValueError(f"missing patient_id in {domain}")
        groups = local.groupby("patient_id", sort=True)["label"].agg(
            ["first", "nunique"]
        )
        if (groups["nunique"] > 1).any():
            mixed_ids = groups.index[groups["nunique"] > 1].tolist()
            raise ValueError(f"mixed-label patient group in {domain}: {mixed_ids}")
        if set(groups["first"]) != {0, 1}:
            raise ValueError(f"source domain lacks both patient-level classes: {domain}")
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=split_seed + domain_index,
        )
        try:
            train_idx, val_idx = next(
                splitter.split(groups.index.to_numpy(), groups["first"].to_numpy())
            )
        except ValueError as error:
            raise ValueError(f"cannot stratify patient groups in {domain}: {error}") from error
        train_ids = set(groups.index[train_idx])
        val_ids = set(groups.index[val_idx])
        if not train_ids.isdisjoint(val_ids):
            raise RuntimeError(f"patient leakage between train and val in {domain}")
        train_part = local[local.patient_id.isin(train_ids)]
        val_part = local[local.patient_id.isin(val_ids)]
        if set(train_part["label"]) != {0, 1}:
            raise ValueError(f"source training split lacks both classes: {domain}")
        if set(val_part["label"]) != {0, 1}:
            raise ValueError(f"source validation split lacks both classes: {domain}")
        train_parts.append(train_part)
        val_parts.append(val_part)

    def sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(
            ["dataset", "image_path"], kind="stable"
        ).reset_index(drop=True)

    frames = {
        "source_train": sorted_frame(pd.concat(train_parts, ignore_index=True)),
        "source_val": sorted_frame(pd.concat(val_parts, ignore_index=True)),
        "target_test": sorted_frame(target),
    }
    source_datasets = set(frames["source_train"].dataset) | set(
        frames["source_val"].dataset
    )
    if target_domain in source_datasets:
        raise RuntimeError(f"target leakage into source frames: {target_domain}")
    return frames


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            newline="",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            frame.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


_BUNDLE_FRAME_NAMES = ("source_train", "source_val", "target_test")


def _validate_lodo_bundle_inputs(
    frames: Mapping[str, pd.DataFrame], config: ExperimentConfig
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    missing = [name for name in _BUNDLE_FRAME_NAMES if name not in frames]
    if missing:
        raise ValueError(f"LODO frames missing entries: {missing}")
    if config.target_domain not in APPROVED_DOMAINS:
        raise ValueError(f"target domain is not approved: {config.target_domain}")

    for name in _BUNDLE_FRAME_NAMES:
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frames[name].columns))
        if missing_columns:
            raise ValueError(f"{name} missing required columns: {missing_columns}")
    tagged = []
    for name in _BUNDLE_FRAME_NAMES:
        frame = frames[name].copy()
        frame["__bundle_frame"] = name
        tagged.append(frame)
    combined = pd.concat(tagged, ignore_index=True)
    _validate_patient_metadata(combined)
    normalized_frames = {
        name: combined.loc[combined["__bundle_frame"] == name]
        .drop(columns="__bundle_frame")
        .reset_index(drop=True)
        for name in _BUNDLE_FRAME_NAMES
    }

    target = normalized_frames["target_test"]
    train = normalized_frames["source_train"]
    val = normalized_frames["source_val"]
    expected_sources = set(APPROVED_DOMAINS) - {config.target_domain}
    if set(target["dataset"]) != {config.target_domain}:
        raise ValueError("target_test must contain only the configured target domain")
    if set(train["dataset"]) != expected_sources or set(val["dataset"]) != expected_sources:
        raise ValueError("both source frames must contain exactly four approved source domains")
    train_groups = set(
        zip(train["dataset"], train["patient_id"].map(_canonical_patient_id))
    )
    val_groups = set(
        zip(val["dataset"], val["patient_id"].map(_canonical_patient_id))
    )
    if not train_groups.isdisjoint(val_groups):
        raise ValueError("patient leakage between source_train and source_val")
    for domain in sorted(expected_sources):
        for frame_name, frame in (("source_train", train), ("source_val", val)):
            if set(frame.loc[frame.dataset == domain, "label"]) != {0, 1}:
                raise ValueError(f"{frame_name} lacks both classes in {domain}")

    combined = pd.concat(
        [normalized_frames[name] for name in _BUNDLE_FRAME_NAMES], ignore_index=True
    )
    if set(combined["dataset"]) != set(APPROVED_DOMAINS):
        raise ValueError("bundle must contain exactly five approved domains")
    validate_no_cross_domain_duplicates(combined)
    return normalized_frames, find_manifest_near_duplicates(combined)


def _complete_marker_payload(path: Path) -> dict[str, str]:
    return {"metadata_sha256": _sha256_file(path / "metadata.json")}


def _write_complete_marker(path: Path) -> None:
    atomic_write_json(path / "COMPLETE", _complete_marker_payload(path))


def _validate_bundle_directory(
    path: Path, config: ExperimentConfig, *, require_complete: bool = True
) -> None:
    expected_csvs = [path / f"{name}.csv" for name in _BUNDLE_FRAME_NAMES]
    metadata_path = path / "metadata.json"
    missing = [item.name for item in [*expected_csvs, metadata_path] if not item.is_file()]
    if missing:
        raise ValueError(f"bundle is incomplete; missing files: {missing}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid bundle metadata: {metadata_path}: {error}") from error
    if metadata.get("target_domain") != config.target_domain:
        raise ValueError("bundle metadata target_domain does not match config")
    expected_split_config = split_protocol_config(config.target_domain)
    if metadata.get("split_seed") != FIXED_SPLIT_SEED:
        raise ValueError("bundle metadata split_seed must equal 20260728")
    if metadata.get("val_fraction") != FIXED_VAL_FRACTION:
        raise ValueError("bundle metadata val_fraction must equal 0.15")
    if metadata.get("split_config") != expected_split_config:
        raise ValueError("bundle metadata split_config does not match fixed protocol")
    if metadata.get("config_hash") != canonical_hash(expected_split_config):
        raise ValueError("bundle metadata config_hash does not match split_config")
    file_metadata = metadata.get("files")
    if not isinstance(file_metadata, dict):
        raise ValueError("bundle metadata files entry must be an object")
    for csv_path in expected_csvs:
        details = file_metadata.get(csv_path.name)
        if not isinstance(details, dict):
            raise ValueError(f"bundle metadata missing entry for {csv_path.name}")
        actual_sha256 = _sha256_file(csv_path)
        if details.get("sha256") != actual_sha256:
            raise ValueError(f"bundle SHA-256 mismatch for {csv_path.name}")
        frame = pd.read_csv(csv_path)
        if details.get("rows") != len(frame):
            raise ValueError(f"bundle row-count mismatch for {csv_path.name}")
        if details.get("class_counts") != _class_counts(frame):
            raise ValueError(f"bundle class-count mismatch for {csv_path.name}")
    if require_complete:
        complete_path = path / "COMPLETE"
        if not complete_path.is_file():
            raise ValueError("bundle is incomplete; missing COMPLETE marker")
        try:
            complete_payload = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid COMPLETE marker: {complete_path}: {error}") from error
        if complete_payload != _complete_marker_payload(path):
            raise ValueError("COMPLETE marker does not match metadata.json")


def _destination_signature(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_dir():
        return canonical_hash({"type": "non-directory", "path": str(path)})
    entries = []
    for item in sorted(path.rglob("*")):
        relative = str(item.relative_to(path))
        if item.is_file():
            entries.append({"path": relative, "type": "file", "sha256": _sha256_file(item)})
        elif item.is_dir():
            entries.append({"path": relative, "type": "directory"})
        else:
            entries.append({"path": relative, "type": "other"})
    return canonical_hash(entries)


def _preflight_destination(
    output_dir: Path, config: ExperimentConfig, force: bool
) -> str | None:
    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise FileExistsError(f"bundle destination already exists: {output_dir}")
    try:
        _validate_bundle_directory(output_dir, config)
        complete = True
    except ValueError:
        complete = False
    if not force:
        if complete:
            raise FileExistsError(f"complete LODO bundle already exists: {output_dir}")
        raise FileExistsError(
            f"bundle destination already exists but is incomplete or corrupt: {output_dir}; "
            "use --force to replace it"
        )
    return _destination_signature(output_dir)


def preflight_lodo_bundle(
    frames: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    config: ExperimentConfig,
    force: bool = False,
) -> pd.DataFrame:
    """Validate all bundle contents and overwrite rules without writing anything."""
    _, candidates = _validate_lodo_bundle_inputs(frames, config)
    unresolved = candidates[~candidates["resolved"].astype(bool)]
    if not unresolved.empty:
        raise ValueError(
            "unresolved manifest near duplicate candidates:\n"
            + unresolved.to_csv(index=False)
        )
    _preflight_destination(Path(output_dir), config, force)
    return candidates


@dataclass
class _PreparedLodoBundle:
    output_dir: Path
    staging_dir: Path
    config: ExperimentConfig
    expected_destination_signature: str | None


def _discard_prepared_lodo_bundles(prepared: list[_PreparedLodoBundle]) -> None:
    for item in prepared:
        if item.staging_dir.exists():
            shutil.rmtree(item.staging_dir)


def _prepare_lodo_bundle(
    frames: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    config: ExperimentConfig,
    force: bool = False,
) -> _PreparedLodoBundle:
    output_dir = Path(output_dir)
    normalized_frames, candidates = _validate_lodo_bundle_inputs(frames, config)
    unresolved = candidates[~candidates["resolved"].astype(bool)]
    if not unresolved.empty:
        candidate_path = output_dir.parent / "near_duplicate_candidates.csv"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_path, index=False)
        raise ValueError(
            "unresolved manifest near duplicate candidates; inspect "
            f"{candidate_path}:\n{unresolved.to_csv(index=False)}"
        )
    expected_signature = _preflight_destination(output_dir, config, force)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent, prefix=f".{output_dir.name}.staging-"
        )
    )
    try:
        file_metadata = {}
        for name in _BUNDLE_FRAME_NAMES:
            path = staging_dir / f"{name}.csv"
            _atomic_write_csv(path, normalized_frames[name])
            file_metadata[path.name] = {
                "rows": int(len(normalized_frames[name])),
                "class_counts": _class_counts(normalized_frames[name]),
                "sha256": _sha256_file(path),
            }
        split_config = split_protocol_config(config.target_domain)
        if (
            config.split_seed != FIXED_SPLIT_SEED
            or config.val_fraction != FIXED_VAL_FRACTION
        ):
            raise ValueError(
                "LODO bundle protocol requires split_seed=20260728 and val_fraction=0.15"
            )
        metadata = {
            "target_domain": config.target_domain,
            "split_seed": FIXED_SPLIT_SEED,
            "val_fraction": FIXED_VAL_FRACTION,
            "split_config": split_config,
            "config_hash": canonical_hash(split_config),
            "patient_id_source_counts": {
                source: int(count)
                for source, count in pd.concat(normalized_frames.values())[
                    "patient_id_source"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
            "files": file_metadata,
        }
        atomic_write_json(staging_dir / "metadata.json", metadata)
        _validate_bundle_directory(staging_dir, config, require_complete=False)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return _PreparedLodoBundle(
        output_dir=output_dir,
        staging_dir=staging_dir,
        config=config,
        expected_destination_signature=expected_signature,
    )


def _unique_backup_path(output_dir: Path) -> Path:
    backup = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent, prefix=f".{output_dir.name}.backup-"
        )
    )
    backup.rmdir()
    return backup


def _publish_prepared_lodo_bundles(
    prepared: list[_PreparedLodoBundle],
) -> list[Path]:
    outputs = [item.output_dir.resolve() for item in prepared]
    if len(outputs) != len(set(outputs)):
        _discard_prepared_lodo_bundles(prepared)
        raise ValueError("prepared bundle destinations must be unique")

    records = []
    try:
        for item in prepared:
            current_signature = _destination_signature(item.output_dir)
            if current_signature != item.expected_destination_signature:
                raise RuntimeError(
                    f"bundle destination changed after preflight: {item.output_dir}"
                )
            record = {"item": item, "backup": None, "published": False}
            records.append(record)
            if current_signature is not None:
                backup = _unique_backup_path(item.output_dir)
                os.replace(item.output_dir, backup)
                record["backup"] = backup
            os.replace(item.staging_dir, item.output_dir)
            record["published"] = True
            _validate_bundle_directory(
                item.output_dir, item.config, require_complete=False
            )

        # Until every directory is published and revalidated, none has a COMPLETE
        # marker. Catchable failures therefore expose only visibly incomplete data.
        for record in records:
            item = record["item"]
            _write_complete_marker(item.output_dir)
        for record in records:
            item = record["item"]
            _validate_bundle_directory(item.output_dir, item.config)
    except BaseException as publish_error:
        recovery_errors = []
        preserved_backups = []
        for record in reversed(records):
            item = record["item"]
            backup = record["backup"]
            if record["published"] and item.output_dir.exists():
                try:
                    shutil.rmtree(item.output_dir)
                except OSError as error:
                    recovery_errors.append(
                        f"cannot remove failed new bundle {item.output_dir}: {error}"
                    )
            if backup is not None and backup.exists():
                try:
                    if item.output_dir.exists():
                        raise OSError("destination is still occupied")
                    os.replace(backup, item.output_dir)
                except OSError as error:
                    preserved_backups.append(str(backup))
                    recovery_errors.append(
                        f"cannot restore {item.output_dir} from {backup}: {error}"
                    )
        _discard_prepared_lodo_bundles(prepared)
        if recovery_errors:
            details = "; ".join(recovery_errors)
            backup_notice = ", ".join(preserved_backups) or "none"
            raise RuntimeError(
                f"bundle publication failed ({publish_error}); recovery incomplete: "
                f"{details}; preserved backup paths: {backup_notice}"
            ) from publish_error
        raise
    else:
        for record in records:
            backup = record["backup"]
            if backup is not None and backup.exists():
                shutil.rmtree(backup)
        _discard_prepared_lodo_bundles(prepared)
    return [item.output_dir for item in prepared]


def write_lodo_bundles(
    bundles: list[tuple[Mapping[str, pd.DataFrame], str | Path, ExperimentConfig]],
    force: bool = False,
) -> list[Path]:
    if len(bundles) != 5 or {config.target_domain for _, _, config in bundles} != set(
        APPROVED_DOMAINS
    ):
        raise ValueError("five-bundle transaction requires each approved target exactly once")
    prepared = []
    try:
        for frames, output_dir, config in bundles:
            prepared.append(
                _prepare_lodo_bundle(frames, output_dir, config, force=force)
            )
    except BaseException:
        _discard_prepared_lodo_bundles(prepared)
        raise
    return _publish_prepared_lodo_bundles(prepared)


def write_lodo_bundle(
    frames: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    config: ExperimentConfig,
    force: bool = False,
    near_duplicate_candidates: pd.DataFrame | None = None,
) -> Path:
    # A caller-provided report can be empty, stale, or from another manifest. Safety
    # therefore depends only on the freshly recomputed, manifest-bound report.
    del near_duplicate_candidates
    prepared = _prepare_lodo_bundle(frames, output_dir, config, force=force)
    return _publish_prepared_lodo_bundles([prepared])[0]
