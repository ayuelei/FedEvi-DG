from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from fedevi_dg.config import ALL_METHODS, ExperimentConfig
from fedevi_dg.dataset import ManifestImageDataset, build_domain_loaders
from fedevi_dg.evidence import SourceNormalizedEvidenceResNet18
from fedevi_dg.manifest import (
    APPROVED_DOMAINS,
    FIXED_SPLIT_SEED,
    FIXED_VAL_FRACTION,
    REQUIRED_COLUMNS,
    find_manifest_near_duplicates,
    patient_id_source_is_reliable,
    split_protocol_config,
    validate_no_cross_domain_duplicates,
)
from fedevi_dg.preprocessing import SourceNormalization, normalize_to_three_channels
from fedevi_dg.provenance import atomic_write_json, canonical_hash
from fedevi_dg.trainer import FederatedTrainer, TrainingResult

_FRAME_NAMES = ("source_train", "source_val", "target_test")
_LOCKED_EXTERNAL_TARGETS = ("QAMEBI",)
_HEX_SHA256 = re.compile("[0-9a-f]{64}")
_RESULT_FILES = (
    "resolved_config.json",
    "manifest_hashes.json",
    "normalization.json",
    "signatures.json",
    "training_history.json",
    "source_val_metrics.json",
    "target_metrics.json",
    "target_predictions.csv",
    "best_global.pt",
    "implementation_snapshot.zip",
    "tracked_diff.patch",
    "git_status.txt",
    "run_metadata.json",
)
_IMPLEMENTATION_FILES = (
    *sorted((REPOSITORY_ROOT / "fedevi_dg").glob("*.py")),
    Path(__file__).resolve(),
)


class PrerequisiteError(RuntimeError):
    """A required, pre-training input is missing, incomplete, or unsafe."""


@dataclass(frozen=True)
class ValidatedLodoBundle:
    path: Path
    target_domain: str
    frames: dict[str, pd.DataFrame]
    metadata: dict
    manifest_hashes: dict[str, str]


class SourceNormalizedResNet18(nn.Module):
    """Accept raw grayscale batches and apply source-only normalization."""

    requires_content_mask = True

    def __init__(self, backbone: nn.Module, stats: SourceNormalization) -> None:
        super().__init__()
        if not np.isfinite([stats.mean, stats.std]).all() or stats.std <= 0:
            raise ValueError("source normalization must be finite with positive std")
        if not hasattr(backbone, "fc") or not hasattr(backbone.fc, "in_features"):
            raise TypeError("ResNet18 backbone must expose fc.in_features")
        backbone.fc = nn.Linear(int(backbone.fc.in_features), 2)
        self.backbone = backbone
        self.stats = stats

    def forward(
        self, raw_image: torch.Tensor, content_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.backbone(self._normalized_input(raw_image, content_mask))

    def _normalized_input(
        self, raw_image: torch.Tensor, content_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw_image = torch.as_tensor(raw_image)
        if raw_image.ndim != 4 or raw_image.shape[1] != 1:
            raise ValueError("model expects raw_image with shape (N, 1, H, W)")
        if content_mask is not None:
            content_mask = torch.as_tensor(content_mask, device=raw_image.device).bool()
            if content_mask.shape != raw_image.shape:
                raise ValueError("content_mask must match raw_image shape")
            masks = content_mask
        else:
            masks = [None] * len(raw_image)
        return torch.stack(
            [
                normalize_to_three_channels(image, self.stats, content_mask=mask)
                for image, mask in zip(raw_image, masks, strict=True)
            ]
        )


class SourceNormalizedOrdinalResNet18(SourceNormalizedResNet18):
    """ResNet18 with separate pathology and normalized BI-RADS risk heads."""

    def __init__(self, backbone: nn.Module, stats: SourceNormalization) -> None:
        nn.Module.__init__(self)
        if not np.isfinite([stats.mean, stats.std]).all() or stats.std <= 0:
            raise ValueError("source normalization must be finite with positive std")
        if not hasattr(backbone, "fc") or not hasattr(backbone.fc, "in_features"):
            raise TypeError("ResNet18 backbone must expose fc.in_features")
        feature_count = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.stats = stats
        self.binary_head = nn.Linear(feature_count, 2)
        self.ordinal_head = nn.Linear(feature_count, 1)

    def forward(
        self, raw_image: torch.Tensor, content_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        binary, _ = self.forward_with_ordinal(raw_image, content_mask=content_mask)
        return binary

    def forward_with_ordinal(
        self, raw_image: torch.Tensor, content_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(self._normalized_input(raw_image, content_mask))
        return (self.binary_head(features), self.ordinal_head(features).squeeze(1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one strict FedEvi-DG LODO or locked external-target experiment."
    )
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=(*APPROVED_DOMAINS, *_LOCKED_EXTERNAL_TARGETS),
        required=True,
    )
    parser.add_argument("--method", choices=ALL_METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/lodo"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Refuse to train unless the repository worktree is clean.",
    )
    parser.add_argument(
        "--birads-domains",
        default="auto",
        help="Comma-separated source clients allowed to use BI-RADS, or 'auto'.",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_implementation_snapshot() -> tuple[str, dict[str, str], dict[str, bytes]]:
    """Capture the exact implementation before training starts."""
    sources = {
        str(path.relative_to(REPOSITORY_ROOT)): path.read_bytes()
        for path in _IMPLEMENTATION_FILES
    }
    files = {
        name: hashlib.sha256(content).hexdigest() for name, content in sources.items()
    }
    return (canonical_hash(files), files, sources)


def _implementation_snapshot() -> tuple[str, dict[str, str]]:
    digest, files, _ = _capture_implementation_snapshot()
    return (digest, files)


def _runtime_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "pillow", "scikit-learn", "torch", "torchvision")
    versions = {"python": platform.python_version()}
    for package in packages:
        versions[package] = importlib_metadata.version(package)
    return versions


def _read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrerequisiteError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise PrerequisiteError(
            f"invalid {description}: expected a JSON object: {path}"
        )
    return value


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    try:
        labels = pd.to_numeric(frame["label"], errors="raise").astype("int64")
    except (KeyError, TypeError, ValueError) as error:
        raise PrerequisiteError(
            "manifest labels must be integer binary values"
        ) from error
    if not labels.isin([0, 1]).all():
        raise PrerequisiteError("manifest labels must be integer binary values")
    counts = labels.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _normalize_manifest_labels(frame: pd.DataFrame, filename: str) -> None:
    if "label" not in frame:
        raise PrerequisiteError(f"{filename} is missing required column: label")
    labels = frame["label"]
    invalid = ~labels.isin(["0", "1"])
    if invalid.any():
        rows = [int(index) for index in frame.index[invalid]]
        raise PrerequisiteError(
            f"{filename} labels must be exactly 0 or 1; invalid rows: {rows}"
        )
    frame["label"] = labels.astype("int64")


def _canonical_patient_groups(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(dataset), " ".join(str(patient).split()).casefold())
        for dataset, patient in zip(frame["dataset"], frame["patient_id"], strict=True)
    }


def _validate_frame_boundaries(
    frames: Mapping[str, pd.DataFrame], target_domain: str
) -> None:
    expected_sources = set(APPROVED_DOMAINS) - {target_domain}
    for name in _FRAME_NAMES:
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frames[name].columns))
        if missing_columns:
            raise PrerequisiteError(
                f"{name}.csv is missing required columns: {missing_columns}"
            )
        if frames[name].empty:
            raise PrerequisiteError(f"{name}.csv must be non-empty")
        _class_counts(frames[name])
    target_domains = set(frames["target_test"]["dataset"].astype(str))
    if target_domains != {target_domain}:
        raise PrerequisiteError(
            f"target_test must contain only the selected target domain {target_domain}; got {sorted(target_domains)}"
        )
    for name in ("source_train", "source_val"):
        actual = set(frames[name]["dataset"].astype(str))
        if actual != expected_sources:
            raise PrerequisiteError(
                f"{name} must contain exactly the four source domains {sorted(expected_sources)}; got {sorted(actual)}"
            )
        for domain in sorted(expected_sources):
            labels = set(
                pd.to_numeric(
                    frames[name].loc[frames[name]["dataset"] == domain, "label"]
                ).astype(int)
            )
            if labels != {0, 1}:
                raise PrerequisiteError(
                    f"{name} must contain both classes for source {domain}"
                )
    train_groups = _canonical_patient_groups(frames["source_train"])
    val_groups = _canonical_patient_groups(frames["source_val"])
    if not train_groups.isdisjoint(val_groups):
        raise PrerequisiteError("patient leakage between source_train and source_val")
    combined = pd.concat(frames.values(), ignore_index=True)
    try:
        validate_no_cross_domain_duplicates(combined)
    except ValueError as error:
        raise PrerequisiteError(str(error)) from error
    try:
        candidates = find_manifest_near_duplicates(combined)
    except ValueError as error:
        raise PrerequisiteError(str(error)) from error
    unresolved = candidates[~candidates["resolved"].astype(bool)]
    if not unresolved.empty:
        raise PrerequisiteError(
            "unresolved manifest near duplicate candidates:\n"
            + unresolved.to_csv(index=False)
        )


def _expected_split_protocol(target_domain: str) -> dict[str, object]:
    if target_domain in APPROVED_DOMAINS:
        return split_protocol_config(target_domain)
    if target_domain == "QAMEBI":
        return {
            "schema_version": 1,
            "protocol": "five_source_locked_public_external_validation",
            "domains": [*APPROVED_DOMAINS, "QAMEBI"],
            "source_domains": list(APPROVED_DOMAINS),
            "target_domain": "QAMEBI",
            "split_seed": FIXED_SPLIT_SEED,
            "val_fraction": FIXED_VAL_FRACTION,
            "target_used_for_model_selection": False,
            "target_used_for_calibration": False,
            "target_used_for_threshold_selection": False,
        }
    raise PrerequisiteError(f"unsupported target domain: {target_domain}")


def _resolve_data_path(data_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else data_root / path


def load_and_validate_lodo_bundle(
    manifest_root: str | Path, target_domain: str, data_root: str | Path
) -> ValidatedLodoBundle:
    """Validate all bytes and LODO boundaries before any image/model access."""
    manifest_root = Path(manifest_root)
    data_root = Path(data_root)
    bundle_path = manifest_root / f"target_{target_domain}"
    if not bundle_path.is_dir():
        raise PrerequisiteError(f"LODO bundle is missing: {bundle_path}")
    required = [
        *(bundle_path / f"{name}.csv" for name in _FRAME_NAMES),
        bundle_path / "metadata.json",
        bundle_path / "COMPLETE",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise PrerequisiteError(
            f"LODO bundle is incomplete at {bundle_path}; missing files: {missing}"
        )
    metadata_path = bundle_path / "metadata.json"
    metadata_sha256 = _sha256_file(metadata_path)
    complete = _read_json(bundle_path / "COMPLETE", "LODO COMPLETE marker")
    if complete != {"metadata_sha256": metadata_sha256}:
        raise PrerequisiteError("LODO COMPLETE marker does not match metadata.json")
    metadata = _read_json(metadata_path, "LODO bundle metadata")
    if metadata.get("target_domain") != target_domain:
        raise PrerequisiteError(
            "LODO metadata target_domain does not match selected target"
        )
    expected_split_config = _expected_split_protocol(target_domain)
    if metadata.get("split_seed") != FIXED_SPLIT_SEED:
        raise PrerequisiteError("LODO metadata split_seed must equal 20260728")
    if metadata.get("val_fraction") != FIXED_VAL_FRACTION:
        raise PrerequisiteError("LODO metadata val_fraction must equal 0.15")
    if metadata.get("split_config") != expected_split_config:
        raise PrerequisiteError(
            "LODO metadata split_config does not match fixed protocol"
        )
    if metadata.get("config_hash") != canonical_hash(expected_split_config):
        raise PrerequisiteError("LODO metadata config_hash does not match split_config")
    file_metadata = metadata.get("files")
    if not isinstance(file_metadata, dict):
        raise PrerequisiteError("LODO metadata files entry must be an object")
    manifest_hashes: dict[str, str] = {}
    for name in _FRAME_NAMES:
        filename = f"{name}.csv"
        details = file_metadata.get(filename)
        if not isinstance(details, dict):
            raise PrerequisiteError(f"LODO metadata is missing {filename}")
        actual_hash = _sha256_file(bundle_path / filename)
        if details.get("sha256") != actual_hash:
            raise PrerequisiteError(f"LODO SHA-256 mismatch for {filename}")
        manifest_hashes[filename] = actual_hash
    manifest_hashes["metadata.json"] = metadata_sha256
    manifest_hashes["COMPLETE"] = _sha256_file(bundle_path / "COMPLETE")
    frames: dict[str, pd.DataFrame] = {}
    for name in _FRAME_NAMES:
        filename = f"{name}.csv"
        try:
            frame = pd.read_csv(
                bundle_path / filename,
                dtype={column: "string" for column in REQUIRED_COLUMNS},
                keep_default_na=False,
            )
        except (OSError, pd.errors.ParserError, UnicodeError) as error:
            raise PrerequisiteError(
                f"invalid LODO manifest {filename}: {error}"
            ) from error
        _normalize_manifest_labels(frame, filename)
        details = file_metadata[filename]
        if details.get("rows") != len(frame):
            raise PrerequisiteError(f"LODO row-count mismatch for {filename}")
        if details.get("class_counts") != _class_counts(frame):
            raise PrerequisiteError(f"LODO class-count mismatch for {filename}")
        frames[name] = frame
    _validate_frame_boundaries(frames, target_domain)
    if not data_root.is_dir():
        raise PrerequisiteError(f"data root is missing: {data_root}")
    image_hashes: dict[Path, str] = {}
    decoded_images: set[Path] = set()
    for name in _FRAME_NAMES:
        for row_index, value in enumerate(frames[name]["image_path"]):
            image_path = _resolve_data_path(data_root, value).resolve()
            if not image_path.is_file():
                raise PrerequisiteError(
                    f"data file referenced by {name}.csv row {row_index} does not exist: {image_path}"
                )
            expected_hash = str(frames[name].at[row_index, "sha256"])
            if not _HEX_SHA256.fullmatch(expected_hash):
                raise PrerequisiteError(
                    f"{name}.csv row {row_index} has invalid image SHA-256: {expected_hash}"
                )
            if image_path not in image_hashes:
                image_hashes[image_path] = _sha256_file(image_path)
            actual_hash = image_hashes[image_path]
            if actual_hash != expected_hash:
                raise PrerequisiteError(
                    f"{name}.csv row {row_index} image SHA-256 mismatch: {image_path}; expected {expected_hash}, got {actual_hash}"
                )
            if image_path not in decoded_images:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        image.load()
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    raise PrerequisiteError(
                        f"undecodable image referenced by {name}.csv row {row_index}: {image_path}: {error}"
                    ) from error
                decoded_images.add(image_path)
    return ValidatedLodoBundle(
        path=bundle_path,
        target_domain=target_domain,
        frames=frames,
        metadata=metadata,
        manifest_hashes=manifest_hashes,
    )


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config != ":4096:8":
            raise RuntimeError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG to be ':4096:8' for this experiment"
            )
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_pretrained_model(
    stats: SourceNormalization,
    *,
    with_ordinal_head: bool = False,
    with_evidence_heads: bool = False,
    use_evidence_bottleneck: bool = True,
    monotonic_ordinal: bool = False,
) -> SourceNormalizedResNet18:
    if sum((with_ordinal_head, with_evidence_heads)) > 1:
        raise ValueError("auxiliary model variants are mutually exclusive")
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if with_evidence_heads:
        return SourceNormalizedEvidenceResNet18(
            backbone,
            stats,
            use_evidence_bottleneck=use_evidence_bottleneck,
            monotonic_ordinal=monotonic_ordinal,
        )
    if with_ordinal_head:
        return SourceNormalizedOrdinalResNet18(backbone, stats)
    return SourceNormalizedResNet18(backbone, stats)


def adapt_domain_loaders(loaders: Mapping[str, object]) -> dict[str, dict[str, object]]:
    source_train = loaders["source_train"]
    source_val = loaders["source_val"]
    if not isinstance(source_train, Mapping) or not isinstance(source_val, Mapping):
        raise TypeError("Source loaders must be domain mappings")
    if source_train.keys() != source_val.keys():
        raise ValueError("Source train/val domains must match")
    return {
        domain: {"train": source_train[domain], "val": source_val[domain]}
        for domain in sorted(source_train)
    }


def _signature_artifact(signatures: Mapping[str, object], metadata: dict) -> dict:
    if signatures:
        raise ValueError("FedEvi-DG does not use acquisition signatures")
    return {"metadata": metadata, "values": {}}


def build_target_predictions(
    target_frame: pd.DataFrame, target_metrics: Mapping[str, object]
) -> pd.DataFrame:
    count = len(target_frame)
    labels = list(target_metrics.get("labels", []))
    scores = list(target_metrics.get("scores", []))
    predictions = list(target_metrics.get("predictions", []))
    if not len(labels) == len(scores) == len(predictions) == count:
        raise ValueError("target metric arrays must match target manifest row count")
    manifest_labels = pd.to_numeric(target_frame["label"]).astype(int).tolist()
    if labels != manifest_labels:
        raise ValueError("target metric labels do not follow target manifest row order")
    if "patient_id_source" not in target_frame:
        raise ValueError("target manifest must include patient_id_source")
    patient_sources = target_frame["patient_id_source"].astype(str).tolist()
    values = {
        "manifest_row": range(count),
        "dataset": target_frame["dataset"].astype(str).tolist(),
        "image_path": target_frame["image_path"].astype(str).tolist(),
        "patient_id": target_frame["patient_id"].astype(str).tolist(),
        "patient_id_source": patient_sources,
        "patient_grouping_reliable": [
            patient_id_source_is_reliable(source) for source in patient_sources
        ],
        "label": labels,
        "score": scores,
        "prediction": predictions,
    }
    ordinal_scores = list(target_metrics.get("ordinal_scores", []))
    if ordinal_scores:
        if len(ordinal_scores) != count:
            raise ValueError("ordinal score count must match target manifest")
        values["ordinal_risk"] = ordinal_scores
    return pd.DataFrame(values)


def assert_output_writable(run_dir: str | Path, force: bool) -> None:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"run destination is not a directory: {run_dir}")
    if not force:
        state = "complete run" if (run_dir / "COMPLETE").is_file() else "incomplete run"
        raise FileExistsError(f"{state} already exists: {run_dir}")


def destination_signature(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    if not path.is_dir():
        return canonical_hash({"type": "non-directory", "path": str(path)})
    entries = []
    for item in sorted(path.rglob("*")):
        relative = str(item.relative_to(path))
        if item.is_file():
            entries.append(
                {"path": relative, "type": "file", "sha256": _sha256_file(item)}
            )
        elif item.is_dir():
            entries.append({"path": relative, "type": "directory"})
        else:
            entries.append({"path": relative, "type": "other"})
    return canonical_hash(entries)


def _unique_backup_path(run_dir: Path) -> Path:
    backup = Path(
        tempfile.mkdtemp(dir=run_dir.parent, prefix=f".{run_dir.name}.backup-")
    )
    backup.rmdir()
    return backup


def publish_staging_directory(
    staging_dir: str | Path,
    run_dir: str | Path,
    *,
    force: bool,
    expected_signature: str | None,
    validator: Callable[[Path], None] | None = None,
) -> Path:
    staging_dir = Path(staging_dir)
    run_dir = Path(run_dir)
    if staging_dir.parent.resolve() != run_dir.parent.resolve():
        raise ValueError("staging and run directories must have the same parent")
    if not (staging_dir / "COMPLETE").is_file():
        raise ValueError("staging result is incomplete; missing COMPLETE")
    validator = validator or (lambda path: None)
    validator(staging_dir)
    if destination_signature(run_dir) != expected_signature:
        raise RuntimeError(f"run destination changed after preflight: {run_dir}")
    if run_dir.exists() and (not force):
        raise FileExistsError(f"run destination already exists: {run_dir}")
    backup = None
    published = False
    try:
        if run_dir.exists():
            backup = _unique_backup_path(run_dir)
            os.replace(run_dir, backup)
        os.replace(staging_dir, run_dir)
        published = True
        validator(run_dir)
    except BaseException as publish_error:
        recovery_errors = []
        if published and run_dir.exists():
            try:
                shutil.rmtree(run_dir)
            except OSError as error:
                recovery_errors.append(
                    f"cannot remove failed new run {run_dir}: {error}"
                )
        if backup is not None and backup.exists():
            try:
                if run_dir.exists():
                    raise OSError("destination is occupied by the failed new run")
                os.replace(backup, run_dir)
            except OSError as error:
                recovery_errors.append(
                    f"cannot restore {run_dir} from {backup}: {error}"
                )
        if recovery_errors:
            backup_notice = (
                str(backup.resolve())
                if backup is not None and backup.exists()
                else "none"
            )
            raise RuntimeError(
                f"result publication failed ({publish_error}); recovery incomplete: {'; '.join(recovery_errors)}; recoverable backup path: {backup_notice}"
            ) from publish_error
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
    return run_dir


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            newline="",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_implementation_archive(path: Path, sources: Mapping[str, bytes]) -> None:
    """Write a deterministic archive containing the exact executed sources."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(sources):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 33188 << 16
                archive.writestr(info, sources[name])
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_complete_result(path: Path) -> None:
    missing = [
        name for name in (*_RESULT_FILES, "COMPLETE") if not (path / name).is_file()
    ]
    if missing:
        raise ValueError(f"result is incomplete; missing artifacts: {missing}")
    marker = json.loads((path / "COMPLETE").read_text(encoding="utf-8"))
    expected_hashes = marker.get("artifact_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        _RESULT_FILES
    ):
        raise ValueError("result COMPLETE marker has an invalid artifact inventory")
    for filename in _RESULT_FILES:
        if expected_hashes[filename] != _sha256_file(path / filename):
            raise ValueError(f"result artifact SHA-256 mismatch for {filename}")
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("target_evaluation_count") != 1:
        raise ValueError("complete result requires exactly one target evaluation")
    predictions = pd.read_csv(path / "target_predictions.csv")
    if predictions["manifest_row"].tolist() != list(range(len(predictions))):
        raise ValueError("target predictions are not in manifest row order")
    if metadata.get("target_prediction_count") != len(predictions):
        raise ValueError("target prediction count does not match run metadata")


def _git_provenance() -> tuple[dict[str, object], bytes, bytes]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return (
            {
                "git_commit": "unknown",
                "git_dirty": None,
                "tracked_diff_sha256": None,
                "git_status_sha256": None,
            },
            b"",
            b"",
        )
    status_bytes = status.stdout.encode("utf-8")
    return (
        {
            "git_commit": commit.stdout.strip(),
            "git_dirty": bool(status.stdout),
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            "git_status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        },
        tracked_diff,
        status_bytes,
    )


def _git_state() -> dict[str, object]:
    state, _, _ = _git_provenance()
    return state


def _write_result_staging(
    staging_dir: Path,
    *,
    resolved_config: dict,
    bundle: ValidatedLodoBundle,
    normalization: SourceNormalization,
    signatures: Mapping[str, object],
    signature_metadata: dict,
    result: TrainingResult,
    target_predictions: pd.DataFrame,
    wall_time_seconds: float,
    git_state: Mapping[str, object],
    tracked_diff: bytes,
    git_status: bytes,
    implementation_sources: Mapping[str, bytes],
) -> None:
    atomic_write_json(staging_dir / "resolved_config.json", resolved_config)
    atomic_write_json(staging_dir / "manifest_hashes.json", bundle.manifest_hashes)
    atomic_write_json(
        staging_dir / "normalization.json",
        {
            "mean": normalization.mean,
            "std": normalization.std,
            "source_split": "source_train",
            "target_excluded": True,
        },
    )
    atomic_write_json(
        staging_dir / "signatures.json",
        _signature_artifact(signatures, signature_metadata),
    )
    atomic_write_json(staging_dir / "training_history.json", result.training_history)
    atomic_write_json(
        staging_dir / "source_val_metrics.json", result.source_val_metrics
    )
    atomic_write_json(staging_dir / "target_metrics.json", result.target_metrics)
    _atomic_write_csv(staging_dir / "target_predictions.csv", target_predictions)
    torch.save(result.best_state_dict, staging_dir / "best_global.pt")
    _write_implementation_archive(
        staging_dir / "implementation_snapshot.zip", implementation_sources
    )
    _atomic_write_bytes(staging_dir / "tracked_diff.patch", tracked_diff)
    _atomic_write_bytes(staging_dir / "git_status.txt", git_status)
    hashes = {
        filename: _sha256_file(staging_dir / filename)
        for filename in _RESULT_FILES
        if filename != "run_metadata.json"
    }
    run_metadata = {
        **git_state,
        "hashes": hashes,
        "implementation_sha256": resolved_config["implementation_sha256"],
        "runtime_versions": resolved_config["runtime_versions"],
        "manifest_hashes": bundle.manifest_hashes,
        "best_round": result.best_round,
        "source_threshold": result.source_threshold,
        "wall_time_seconds": wall_time_seconds,
        "target_evaluation_count": result.target_evaluation_count,
        "target_prediction_count": len(target_predictions),
        "selected_checkpoint_kind": result.selected_checkpoint_kind,
        "selection_domains": list(result.selection_domains),
        "target_domain": result.target_domain,
        "method": resolved_config["method"],
        "seed": resolved_config["seed"],
        "split_protocol": bundle.metadata["split_config"],
        "split_protocol_hash": bundle.metadata["config_hash"],
        "method_metadata": result.method_metadata,
    }
    atomic_write_json(staging_dir / "run_metadata.json", run_metadata)
    artifact_hashes = {
        filename: _sha256_file(staging_dir / filename) for filename in _RESULT_FILES
    }
    atomic_write_json(staging_dir / "COMPLETE", {"artifact_sha256": artifact_hashes})
    _validate_complete_result(staging_dir)


def _resolved_configuration(
    args: argparse.Namespace,
    config: ExperimentConfig,
    num_workers: int,
    bundle: ValidatedLodoBundle,
    implementation_sha256: str,
    implementation_files: Mapping[str, str],
) -> dict:
    value = config.resolved()
    value.update(
        {
            "metadata_csv": None,
            "metadata_csv_status": "not_applicable_prebuilt_lodo",
            "lodo_bundle_metadata": str((bundle.path / "metadata.json").resolve()),
            "lodo_manifests": {
                f"{name}.csv": str((bundle.path / f"{name}.csv").resolve())
                for name in _FRAME_NAMES
            },
            "manifest_root": str(Path(args.manifest_root).resolve()),
            "output_root": str(Path(args.output_root).resolve()),
            "target": args.target,
            "method": args.method,
            "seed": args.seed,
            "device": args.device,
            "smoke": bool(args.smoke),
            "force": bool(args.force),
            "num_workers": num_workers,
            "model": "resnet18",
            "pretrained_weights": "IMAGENET1K_V1",
            "input_channels": 1,
            "classifier_classes": 2,
            "ordinal_risk_head": args.method
            in {
                "fedavg_ord",
                "fedevi_dg",
                "fedevi_mtl",
                "fedevi_shuffle",
                "fedevi_mask_shift",
                "fedevi_no_taskagg",
                "fedevi_monotonic",
            },
            "lesion_evidence_head": args.method
            in {
                "fedevi_dg",
                "fedevi_mtl",
                "fedevi_seg_only",
                "fedevi_shuffle",
                "fedevi_mask_shift",
                "fedevi_no_taskagg",
                "fedevi_monotonic",
            },
            "diagnostic_evidence_bottleneck": (
                args.method not in {"fedevi_mtl", "fedevi_seg_only"}
                if args.method.startswith("fedevi_")
                else None
            ),
            "active_auxiliary_tasks": (
                ["lesion_segmentation"] if args.method == "fedevi_seg_only" else None
            ),
            "monotonic_ordinal_evidence": args.method
            in {
                "fedevi_monotonic",
                "fedevi_shuffle",
                "fedevi_mask_shift",
                "fedevi_no_taskagg",
            },
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "implementation_sha256": implementation_sha256,
            "implementation_files": dict(implementation_files),
            "runtime_versions": _runtime_versions(),
            "split_protocol": bundle.metadata["split_config"],
            "split_protocol_hash": bundle.metadata["config_hash"],
        }
    )
    return value


def run_experiment(args: argparse.Namespace) -> Path:
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    run_dir = (
        Path(args.output_root)
        / f"target_{args.target}"
        / args.method
        / f"seed_{args.seed}"
    )
    assert_output_writable(run_dir, args.force)
    expected_signature = destination_signature(run_dir)
    bundle = load_and_validate_lodo_bundle(
        args.manifest_root, args.target, args.data_root
    )
    git_state, tracked_diff, git_status = _git_provenance()
    if (
        getattr(args, "require_clean_git", False)
        and git_state["git_dirty"] is not False
    ):
        raise PrerequisiteError(
            "--require-clean-git requires a clean, available Git worktree"
        )
    implementation_sha256, implementation_files, implementation_sources = (
        _capture_implementation_snapshot()
    )
    started = time.monotonic()
    set_global_seeds(args.seed)
    rounds = 2 if args.smoke else args.rounds
    num_workers = 0
    birads_domains = None
    if args.birads_domains.strip().casefold() != "auto":
        birads_domains = tuple(
            (token.strip() for token in args.birads_domains.split(",") if token.strip())
        )
        if not birads_domains:
            raise ValueError("--birads-domains must be 'auto' or a non-empty list")
    config = ExperimentConfig(
        data_root=Path(args.data_root),
        metadata_csv=bundle.path / "metadata.json",
        target_domain=args.target,
        domains=(
            APPROVED_DOMAINS
            if args.target in APPROVED_DOMAINS
            else (*APPROVED_DOMAINS, args.target)
        ),
        method=args.method,
        seed=args.seed,
        rounds=rounds,
        local_epochs=1,
        birads_domains=birads_domains,
        ordinal_supervised_weight=0.0 if args.method == "fedevi_seg_only" else 0.25,
    )
    loaders = build_domain_loaders(
        bundle.frames,
        data_root=config.data_root,
        input_size=config.input_size,
        batch_size=config.batch_size,
        num_workers=num_workers,
        seed=config.seed,
    )
    signatures, signature_metadata = ({}, {})
    if args.method in {
        "fedevi_dg",
        "fedevi_mtl",
        "fedevi_seg_only",
        "fedevi_shuffle",
        "fedevi_mask_shift",
        "fedevi_no_taskagg",
        "fedevi_monotonic",
    }:
        model = build_pretrained_model(
            loaders["source_normalization"],
            with_evidence_heads=True,
            use_evidence_bottleneck=args.method
            not in {"fedevi_mtl", "fedevi_seg_only"},
            monotonic_ordinal=args.method
            in {
                "fedevi_monotonic",
                "fedevi_shuffle",
                "fedevi_mask_shift",
                "fedevi_no_taskagg",
            },
        )
    elif args.method in {"fedavg_ord"}:
        model = build_pretrained_model(
            loaders["source_normalization"], with_ordinal_head=True
        )
    else:
        model = build_pretrained_model(loaders["source_normalization"])
    source_loaders = adapt_domain_loaders(loaders)
    trainer = FederatedTrainer(
        model,
        args.method,
        source_loaders,
        loaders["target_test"],
        config,
        signatures,
        torch.device(args.device),
    )
    result = trainer.train()
    expected_sources = tuple(sorted(set(APPROVED_DOMAINS) - {args.target}))
    if result.target_evaluation_count != 1:
        raise RuntimeError("trainer did not evaluate target exactly once")
    if tuple(result.selection_domains) != expected_sources:
        raise RuntimeError("trainer selection domains do not match exact LODO sources")
    if result.target_domain != args.target:
        raise RuntimeError("trainer target domain does not match requested target")
    predictions = build_target_predictions(
        bundle.frames["target_test"], result.target_metrics
    )
    resolved = _resolved_configuration(
        args, config, num_workers, bundle, implementation_sha256, implementation_files
    )
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(dir=run_dir.parent, prefix=f".{run_dir.name}.staging-")
    )
    try:
        _write_result_staging(
            staging_dir,
            resolved_config=resolved,
            bundle=bundle,
            normalization=loaders["source_normalization"],
            signatures=signatures,
            signature_metadata=signature_metadata,
            result=result,
            target_predictions=predictions,
            wall_time_seconds=time.monotonic() - started,
            git_state=git_state,
            tracked_diff=tracked_diff,
            git_status=git_status,
            implementation_sources=implementation_sources,
        )
        publish_staging_directory(
            staging_dir,
            run_dir,
            force=args.force,
            expected_signature=expected_signature,
            validator=_validate_complete_result,
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run_experiment(args)
    except PrerequisiteError as error:
        print(f"prerequisite error: {error}", file=sys.stderr)
        return 2
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
