from __future__ import annotations
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import re
import sys
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from fedevi_dg.manifest import (
    APPROVED_DOMAINS,
    resolve_path_beneath,
)

_WHU_DIRECTORY = "BUSI_WHU Breast Cancer Ultrasound Image  Dataset"
_WHU_LABEL = "Benign(0)\nmalignant(1)"
_UDIAT_DIRECTORY = Path("udiat/UDIAT_Dataset_B/UDIAT_Dataset_B")
_DECISION_COLUMNS = ("sha256", "dataset", "raw_image_path", "label", "decision")
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


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], source: str
) -> None:
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
        image_root, image_path, path_name=path_name, root_name="images root"
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
            f"BUSBRA PNG inventory mismatch; missing={sorted(expected_png_names - actual_png_names)}, extra={sorted(actual_png_names - expected_png_names)}"
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
            f"BUSI_WHU image/metadata coverage mismatch; missing={missing}, extra={extra}"
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
    pattern = re.compile("^(benign|malignant)\\((\\d+)\\)$", re.IGNORECASE)
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
                raise ValueError(
                    f"{dataset} raw ID absent from BIRADS&FOLD.xlsx: {source_id}"
                )
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
            f"UDIAT image/metadata coverage mismatch; missing={sorted(names - raw_names)}, extra={sorted(raw_names - names)}"
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
        raise ValueError(
            "authoritative sources do not contain exactly the five approved domains"
        )
    mixed = frame.groupby(["dataset", "patient_id"])["label"].nunique()
    if (mixed > 1).any():
        groups = [f"{dataset}/{patient}" for dataset, patient in mixed[mixed > 1].index]
        raise ValueError(
            f"mixed-label patient group in authoritative metadata: {groups}"
        )
    return frame


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
    for index in sorted(
        excluded_indices, key=lambda value: work.at[value, "raw_image_path"]
    ):
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
    return (
        retained,
        decisions_frame,
        excluded_frame,
        {
            "groups": duplicate_groups,
            "rows": duplicate_rows,
            "label_conflict_groups": conflict_groups,
        },
    )
