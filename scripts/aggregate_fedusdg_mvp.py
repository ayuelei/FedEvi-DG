#!/usr/bin/env python3
"""Strictly validate and aggregate the phase-2/3 FedUS-DG result matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fedusdg_mvp.config import METHODS
from fedusdg_mvp.manifest import (
    FIXED_SPLIT_SEED,
    FIXED_VAL_FRACTION,
    patient_id_source_is_reliable,
    split_protocol_config,
)
from fedusdg_mvp.metrics import (
    binary_metrics,
    cluster_bootstrap_auc,
    patient_level_auc,
    select_source_threshold,
)
from fedusdg_mvp.provenance import canonical_hash


TARGETS = ("BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT")
RESULT_FILES = (
    "resolved_config.json",
    "manifest_hashes.json",
    "normalization.json",
    "signatures.json",
    "training_history.json",
    "source_val_metrics.json",
    "target_metrics.json",
    "target_predictions.csv",
    "best_global.pt",
    "run_metadata.json",
)
SUMMARY_FILES = (
    "per_target_metrics.csv",
    "patient_level_metrics.csv",
    "bootstrap_intervals.csv",
    "macro_worst_metrics.csv",
    "phase2_gate.json",
    "summary.md",
)
METRICS = (
    "auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "macro_f1",
    "loss",
)
BUDGET_KEYS = (
    "rounds",
    "local_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "fedprox_mu",
    "consistency_weight",
    "eval_interval",
    "num_frequency_bands",
    "lambda_min",
    "lambda_max",
    "input_size",
    "model",
    "pretrained_weights",
    "input_channels",
    "classifier_classes",
    "cublas_workspace_config",
    "smoke",
)
FIXED_BUDGET = {
    "local_epochs": 1,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "fedprox_mu": 0.01,
    "consistency_weight": 0.5,
    "eval_interval": 5,
    "num_frequency_bands": 8,
    "lambda_min": 0.1,
    "lambda_max": 0.5,
    "input_size": 256,
    "model": "resnet18",
    "pretrained_weights": "IMAGENET1K_V1",
    "input_channels": 1,
    "classifier_classes": 2,
    "cublas_workspace_config": ":4096:8",
    "smoke": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {description}: expected JSON object: {path}")
    return value


def _read_array(path: Path, description: str) -> list:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, list):
        raise ValueError(f"invalid {description}: expected JSON array: {path}")
    return value


def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def evaluate_phase2_gate(auc: Mapping[str, Sequence[float]]) -> dict[str, object]:
    """Evaluate the three pre-registered numerical phase-2 conditions."""
    required = ("fedavg", "whole", "depth", "fedus_dg_mvp")
    missing = [method for method in required if method not in auc]
    if missing:
        raise ValueError(f"phase-2 AUC is missing methods: {missing}")
    arrays: dict[str, np.ndarray] = {}
    for method in required:
        values = np.asarray(auc[method], dtype=float)
        if values.shape != (len(TARGETS),):
            raise ValueError("phase-2 gate requires exactly five target AUCs per method")
        if not np.isfinite(values).all():
            raise ValueError("phase-2 AUC values must be finite")
        arrays[method] = values

    depth_wins = int(np.count_nonzero(arrays["depth"] > arrays["whole"]))
    mvp_target_wins = int(
        np.count_nonzero(arrays["fedus_dg_mvp"] > arrays["fedavg"])
    )
    fedavg_macro = float(np.mean(arrays["fedavg"]))
    mvp_macro = float(np.mean(arrays["fedus_dg_mvp"]))
    macro_gate = bool(mvp_macro >= fedavg_macro)
    depth_gate = bool(depth_wins >= 3)
    multi_target_gate = bool(mvp_target_wins >= 2)
    return {
        "target_count": len(TARGETS),
        "depth_wins": depth_wins,
        "depth_wins_required": 3,
        "depth_gate": depth_gate,
        "mvp_target_wins": mvp_target_wins,
        "mvp_target_wins_required": 2,
        "multi_target_gate": multi_target_gate,
        "fedavg_macro_auc": fedavg_macro,
        "mvp_macro_auc": mvp_macro,
        "mvp_macro_not_below_fedavg": macro_gate,
        "pass": bool(depth_gate and macro_gate and multi_target_gate),
    }


def _validate_complete(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"run cell must be a real directory: {path}")
    complete = path / "COMPLETE"
    if not complete.is_file() or complete.is_symlink():
        raise ValueError(f"run cell is missing COMPLETE: {path}")
    marker = _read_object(complete, "COMPLETE marker")
    inventory = marker.get("artifact_sha256")
    if not isinstance(inventory, dict) or set(inventory) != set(RESULT_FILES):
        raise ValueError(f"COMPLETE has an invalid artifact inventory: {path}")
    for filename in RESULT_FILES:
        artifact = path / filename
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"COMPLETE artifact is missing: {artifact}")
        if inventory[filename] != _sha256(artifact):
            raise ValueError(f"artifact SHA-256 mismatch for {artifact}")


def _existing_cells(root: Path) -> list[tuple[str, str, int, Path]]:
    cells: list[tuple[str, str, int, Path]] = []
    if root.is_symlink():
        raise ValueError(f"output root must not be a symbolic link: {root}")
    if not root.is_dir():
        raise ValueError(f"output root is missing: {root}")
    if root.resolve() != root.absolute():
        raise ValueError(f"output root path contains a symbolic link: {root}")
    for target_path in root.iterdir():
        if target_path.name == "_summary" or target_path.name.startswith("._summary"):
            continue
        if target_path.is_symlink():
            raise ValueError(f"target entry must not be a symbolic link: {target_path}")
        if not target_path.is_dir() or not target_path.name.startswith("target_"):
            raise ValueError(f"unknown target entry: {target_path}")
        target = target_path.name.removeprefix("target_")
        if target not in TARGETS:
            raise ValueError(f"unknown target directory: {target_path}")
        for method_path in target_path.iterdir():
            if method_path.is_symlink():
                raise ValueError(f"method entry must not be a symbolic link: {method_path}")
            if not method_path.is_dir() or method_path.name not in METHODS:
                raise ValueError(f"unknown method entry: {method_path}")
            for seed_path in method_path.iterdir():
                if seed_path.is_symlink():
                    raise ValueError(f"seed entry must not be a symbolic link: {seed_path}")
                if not seed_path.is_dir() or not seed_path.name.startswith("seed_"):
                    raise ValueError(f"unknown seed entry: {seed_path}")
                try:
                    seed = int(seed_path.name.removeprefix("seed_"))
                except ValueError as error:
                    raise ValueError(f"unknown cell: {seed_path}") from error
                cells.append((target, method_path.name, seed, seed_path))
    return cells


def _validate_matrix(
    root: Path, required_seeds: tuple[int, ...]
) -> dict[tuple[str, str, int], Path]:
    cells = _existing_cells(root)
    for _, _, _, path in cells:
        _validate_complete(path)

    expected = {
        (target, method, seed)
        for target in TARGETS
        for method in METHODS
        for seed in required_seeds
    }
    actual: dict[tuple[str, str, int], Path] = {}
    for target, method, seed, path in cells:
        key = (target, method, seed)
        if key in actual:
            raise ValueError(f"duplicate cell: {key}")
        actual[key] = path
    unknown = sorted(set(actual) - expected)
    missing = sorted(expected - set(actual))
    if unknown:
        raise ValueError(f"unknown cell(s): {unknown}")
    if missing:
        raise ValueError(f"missing required cell(s): {missing}")
    return actual


def _validate_identity(
    path: Path,
    target: str,
    method: str,
    seed: int,
    resolved: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    expected = (target, method, seed)
    config_identity = (
        resolved.get("target_domain"),
        resolved.get("method"),
        resolved.get("seed"),
    )
    metadata_identity = (
        metadata.get("target_domain"),
        metadata.get("method"),
        metadata.get("seed"),
    )
    if config_identity != expected or metadata_identity != expected:
        raise ValueError(
            f"run identity mismatch at {path}: expected {expected}, "
            f"config={config_identity}, metadata={metadata_identity}"
        )
    if resolved.get("target") not in (None, target):
        raise ValueError(f"run identity target mismatch at {path}")
    if tuple(resolved.get("domains", ())) != TARGETS:
        raise ValueError(f"run identity has incorrect domain order at {path}")


def _validate_split_protocol(
    path: Path,
    target: str,
    resolved: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    expected = split_protocol_config(target)
    expected_hash = canonical_hash(expected)
    if resolved.get("split_seed") != FIXED_SPLIT_SEED:
        raise ValueError(f"split protocol split_seed mismatch at {path}")
    if resolved.get("val_fraction") != FIXED_VAL_FRACTION:
        raise ValueError(f"split protocol val_fraction mismatch at {path}")
    for artifact_name, artifact in (
        ("resolved configuration", resolved),
        ("run metadata", metadata),
    ):
        if artifact.get("split_protocol") != expected:
            raise ValueError(f"split protocol mismatch in {artifact_name} at {path}")
        if artifact.get("split_protocol_hash") != expected_hash:
            raise ValueError(
                f"split protocol digest mismatch in {artifact_name} at {path}"
            )


def _validate_leakage(
    path: Path,
    target: str,
    metadata: Mapping[str, object],
    signatures: Mapping[str, object],
    method: str,
) -> None:
    expected_sources = set(TARGETS) - {target}
    selection = metadata.get("selection_domains")
    if (
        not isinstance(selection, list)
        or len(selection) != 4
        or not all(isinstance(domain, str) for domain in selection)
        or len(set(selection)) != 4
        or set(selection) != expected_sources
    ):
        raise ValueError(
            f"selection_domains must be exactly four unique source domains at {path}"
        )
    signature_metadata = signatures.get("metadata")
    values = signatures.get("values")
    if not isinstance(signature_metadata, dict) or not isinstance(values, dict):
        raise ValueError(f"invalid signature artifact at {path}")
    domains = signature_metadata.get("domains", {})
    if not isinstance(domains, dict):
        raise ValueError(f"invalid signature metadata domains at {path}")
    if target in domains or target in values:
        raise ValueError(f"target leaked into signature metadata at {path}")
    if method in {
        "whole",
        "depth",
        "fedus_dg_mvp",
        "factor",
        "virtual",
        "fedphys",
        "control",
    }:
        if set(domains) != expected_sources or set(values) != expected_sources:
            raise ValueError(f"signature domains must exactly match sources at {path}")
        if signature_metadata.get("target_excluded") is not True:
            raise ValueError(f"signature metadata must attest target exclusion at {path}")
    elif domains or values:
        raise ValueError(f"non-spectral method has unexpected signature metadata at {path}")


def _validated_bool_series(series: pd.Series, description: str) -> np.ndarray:
    normalized = series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower()
    )
    mapped = normalized.map({True: True, False: False, "true": True, "false": False})
    if mapped.isna().any():
        raise ValueError(f"{description} must contain explicit booleans")
    return mapped.to_numpy(dtype=bool)


def _validated_predictions(
    path: Path,
    target: str,
    metrics: Mapping[str, object],
    source_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    try:
        frame = pd.read_csv(
            path / "target_predictions.csv",
            dtype={
                "dataset": "string",
                "patient_id": "string",
                "patient_id_source": "string",
                "image_path": "string",
            },
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"invalid target predictions at {path}: {error}") from error
    required = {
        "manifest_row",
        "dataset",
        "image_path",
        "patient_id",
        "patient_id_source",
        "patient_grouping_reliable",
        "label",
        "score",
        "prediction",
    }
    if set(frame.columns) != required:
        raise ValueError(f"target predictions have invalid columns at {path}")
    if frame.empty or frame["manifest_row"].tolist() != list(range(len(frame))):
        raise ValueError(f"target predictions are not in manifest order at {path}")
    if set(frame["dataset"].astype(str)) != {target}:
        raise ValueError(f"target predictions contain the wrong target at {path}")
    if frame["patient_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"target predictions contain empty patient IDs at {path}")
    if frame["patient_id_source"].astype(str).str.strip().eq("").any():
        raise ValueError(f"target predictions contain empty patient ID sources at {path}")
    reliability = _validated_bool_series(
        frame["patient_grouping_reliable"], "patient_grouping_reliable"
    )
    expected_reliability = np.asarray(
        [patient_id_source_is_reliable(value) for value in frame["patient_id_source"]],
        dtype=bool,
    )
    if not np.array_equal(reliability, expected_reliability):
        raise ValueError(f"patient grouping reliability disagrees with provenance at {path}")
    frame["patient_grouping_reliable"] = reliability
    try:
        labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=int)
        scores = pd.to_numeric(frame["score"], errors="raise").to_numpy(dtype=float)
        predictions = pd.to_numeric(
            frame["prediction"], errors="raise"
        ).to_numpy(dtype=int)
    except (TypeError, ValueError) as error:
        raise ValueError(f"target predictions contain invalid numeric data at {path}") from error
    if set(labels) != {0, 1} or not np.isin(predictions, [0, 1]).all():
        raise ValueError(f"target labels/predictions must be binary at {path}")
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError(f"target scores must be finite probabilities at {path}")
    expected_predictions = (scores >= source_threshold).astype(int)
    if not np.array_equal(predictions, expected_predictions):
        raise ValueError(
            f"target prediction disagrees with frozen source_threshold at {path}"
        )
    if reliability.all():
        patient_labels = (
            frame.assign(_label=labels).groupby("patient_id")["_label"].nunique()
        )
        if (patient_labels != 1).any():
            raise ValueError(f"target patient has mixed labels at {path}")

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    recomputed = {
        "auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "loss": float(log_loss(labels, scores, labels=[0, 1])),
    }
    for name, actual in recomputed.items():
        reported = _finite_number(metrics.get(name), f"target metric {name}")
        if not math.isclose(actual, reported, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"target metric {name} disagrees with predictions at {path}")
    if metrics.get("n") != len(frame):
        raise ValueError(f"target metric n disagrees with predictions at {path}")
    for column, key in ((labels, "labels"), (scores, "scores"), (predictions, "predictions")):
        reported = np.asarray(metrics.get(key))
        if reported.shape != column.shape or not np.allclose(reported, column, rtol=0, atol=0):
            raise ValueError(f"target metric {key} disagrees with predictions at {path}")
    return frame, recomputed


def _bootstrap_seed(target: str, method: str, seed: int) -> int:
    value = hashlib.sha256(f"{target}\0{method}\0{seed}".encode()).digest()
    return int.from_bytes(value[:4], "big")


def _grouping_for_inference(frame: pd.DataFrame) -> tuple[str, list[str]]:
    reliable = bool(frame["patient_grouping_reliable"].astype(bool).all())
    has_multi_image_patient = bool(frame["patient_id"].duplicated(keep=False).any())
    if reliable and has_multi_image_patient:
        return "patient", frame["patient_id"].astype(str).tolist()
    identities = [
        f"{row}:{path}"
        for row, path in zip(frame["manifest_row"], frame["image_path"], strict=True)
    ]
    return "image", identities


def _validate_source_metrics(
    path: Path,
    source_metrics: Mapping[str, object],
    expected_sources: set[str],
) -> float:
    if set(source_metrics) != expected_sources:
        raise ValueError(f"source metrics must exactly match selection domains at {path}")
    labels_by_source: dict[str, list[int]] = {}
    scores_by_source: dict[str, list[float]] = {}
    for domain, domain_metrics in source_metrics.items():
        if not isinstance(domain_metrics, dict):
            raise ValueError(f"source metrics for {domain} must be an object at {path}")
        try:
            recomputed = binary_metrics(
                domain_metrics.get("labels", []),
                domain_metrics.get("scores", []),
                threshold=0.5,
            )
        except ValueError as error:
            raise ValueError(
                f"source metric labels/scores for {domain} are invalid at {path}: {error}"
            ) from error
        for name in METRICS:
            reported = _finite_number(
                domain_metrics.get(name), f"source metric {name} for {domain}"
            )
            if not math.isclose(
                reported, float(recomputed[name]), rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    f"source metric {name} disagrees with labels/scores for {domain} at {path}"
                )
        if domain_metrics.get("n") != recomputed["n"]:
            raise ValueError(f"source metric n disagrees for {domain} at {path}")
        if domain_metrics.get("predictions") != recomputed["predictions"]:
            raise ValueError(f"source predictions disagree for {domain} at {path}")
        labels_by_source[domain] = recomputed["labels"]
        scores_by_source[domain] = recomputed["scores"]
    return select_source_threshold(labels_by_source, scores_by_source)


def _validate_training_history(
    path: Path,
    history: list,
    resolved: Mapping[str, object],
    metadata: Mapping[str, object],
    source_metrics: Mapping[str, object],
    expected_sources: set[str],
) -> None:
    rounds = resolved.get("rounds")
    eval_interval = resolved.get("eval_interval")
    if not isinstance(rounds, int) or not isinstance(eval_interval, int):
        raise ValueError(f"training history requires integer budget values at {path}")
    observed_rounds = [
        entry.get("round") if isinstance(entry, dict) else None for entry in history
    ]
    if observed_rounds != list(range(1, rounds + 1)):
        raise ValueError(f"training history rounds are not contiguous at {path}")
    evaluation_rounds = set(range(eval_interval, rounds + 1, eval_interval))
    evaluation_rounds.add(rounds)
    selected_rounds = []
    for entry in history:
        round_index = int(entry["round"])
        client_losses = entry.get("client_train_loss")
        if not isinstance(client_losses, dict) or set(client_losses) != expected_sources:
            raise ValueError(f"training history client domains mismatch at {path}")
        for domain, value in client_losses.items():
            _finite_number(value, f"training history client loss {domain}")

        source_validation = entry.get("source_validation")
        macro_auc = entry.get("macro_source_auc")
        macro_loss = entry.get("macro_source_loss")
        if round_index in evaluation_rounds:
            if (
                not isinstance(source_validation, dict)
                or set(source_validation) != expected_sources
            ):
                raise ValueError(f"training history source validation mismatch at {path}")
            for domain, metrics in source_validation.items():
                if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
                    raise ValueError(
                        f"training history source metrics mismatch for {domain} at {path}"
                    )
                for metric, value in metrics.items():
                    _finite_number(
                        value, f"training history {domain} source metric {metric}"
                    )
            reported_macro_auc = _finite_number(
                macro_auc, "training history macro source AUC"
            )
            reported_macro_loss = _finite_number(
                macro_loss, "training history macro source loss"
            )
            expected_macro_auc = float(
                np.mean(
                    [source_validation[domain]["auc"] for domain in expected_sources]
                )
            )
            expected_macro_loss = float(
                np.mean(
                    [source_validation[domain]["loss"] for domain in expected_sources]
                )
            )
            if not math.isclose(
                reported_macro_auc, expected_macro_auc, rel_tol=0, abs_tol=1e-12
            ) or not math.isclose(
                reported_macro_loss, expected_macro_loss, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"training history macro source metrics mismatch at {path}"
                )
        elif source_validation is not None or macro_auc is not None or macro_loss is not None:
            raise ValueError(f"training history has unscheduled validation at {path}")

        selected = entry.get("selected_checkpoint")
        if not isinstance(selected, bool):
            raise ValueError(f"training history selection flag must be boolean at {path}")
        if selected:
            selected_rounds.append(round_index)

    best_round = metadata.get("best_round")
    if isinstance(best_round, bool) or not isinstance(best_round, int):
        raise ValueError(f"training history best_round must be an integer at {path}")
    if selected_rounds != [best_round] or best_round not in evaluation_rounds:
        raise ValueError(f"training history selected checkpoint mismatch at {path}")
    selected_metrics = history[int(best_round) - 1]["source_validation"]
    for domain in expected_sources:
        for metric in METRICS:
            if not math.isclose(
                float(selected_metrics[domain][metric]),
                _finite_number(
                    source_metrics[domain].get(metric),
                    f"source metric {domain}.{metric}",
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"training history selected source metrics mismatch at {path}"
                )


def _validate_reproducibility_contract(
    path: Path,
    resolved: Mapping[str, object],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    implementation = resolved.get("implementation_sha256")
    implementation_files = resolved.get("implementation_files")
    if (
        not isinstance(implementation, str)
        or not re.fullmatch(r"[0-9a-f]{64}", implementation)
        or not isinstance(implementation_files, dict)
        or not implementation_files
        or not all(
            isinstance(name, str)
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            for name, digest in implementation_files.items()
        )
        or canonical_hash(implementation_files) != implementation
        or metadata.get("implementation_sha256") != implementation
    ):
        raise ValueError(f"implementation snapshot mismatch or invalid value at {path}")
    runtime_versions = resolved.get("runtime_versions")
    if (
        not isinstance(runtime_versions, dict)
        or not runtime_versions
        or not all(
            isinstance(name, str) and isinstance(version, str) and version
            for name, version in runtime_versions.items()
        )
        or metadata.get("runtime_versions") != runtime_versions
    ):
        raise ValueError(f"invalid runtime version snapshot at {path}")

    expected_hash_names = set(RESULT_FILES) - {"run_metadata.json"}
    metadata_hashes = metadata.get("hashes")
    if (
        not isinstance(metadata_hashes, dict)
        or set(metadata_hashes) != expected_hash_names
    ):
        raise ValueError(f"run metadata artifact hashes have invalid inventory at {path}")
    actual_hashes = {
        filename: _sha256(path / filename) for filename in expected_hash_names
    }
    if metadata_hashes != actual_hashes:
        raise ValueError(f"run metadata artifact hashes disagree at {path}")
    return {
        "implementation_sha256": implementation,
        "runtime_versions": runtime_versions,
    }


def _validate_run(
    path: Path, target: str, method: str, seed: int, n_bootstrap: int
) -> tuple[
    dict[str, object],
    dict[str, object] | None,
    dict[str, object],
    pd.DataFrame,
    dict[str, object],
]:
    resolved = _read_object(path / "resolved_config.json", "resolved configuration")
    manifests = _read_object(path / "manifest_hashes.json", "manifest hashes")
    normalization = _read_object(path / "normalization.json", "normalization")
    signatures = _read_object(path / "signatures.json", "signature artifact")
    history = _read_array(path / "training_history.json", "training history")
    source_metrics = _read_object(
        path / "source_val_metrics.json", "source validation metrics"
    )
    metrics = _read_object(path / "target_metrics.json", "target metrics")
    metadata = _read_object(path / "run_metadata.json", "run metadata")
    contract = _validate_reproducibility_contract(path, resolved, metadata)
    _validate_identity(path, target, method, seed, resolved, metadata)
    _validate_split_protocol(path, target, resolved, metadata)
    if metadata.get("target_evaluation_count") != 1:
        raise ValueError(f"target_evaluation_count must equal 1 at {path}")
    if metadata.get("selected_checkpoint_kind") != "global":
        raise ValueError(f"selected_checkpoint_kind must be global at {path}")
    source_threshold = _finite_number(
        metadata.get("source_threshold"), "source_threshold"
    )
    if not 0 <= source_threshold <= 1:
        raise ValueError(f"source_threshold must be in [0, 1] at {path}")
    if metadata.get("manifest_hashes") != manifests:
        raise ValueError(f"run metadata manifest hashes disagree at {path}")
    if normalization.get("target_excluded") is not True:
        raise ValueError(f"normalization must attest target exclusion at {path}")
    if normalization.get("source_split") != "source_train":
        raise ValueError(f"normalization source_split must be source_train at {path}")
    _finite_number(normalization.get("mean"), "normalization mean")
    normalization_std = _finite_number(
        normalization.get("std"), "normalization std"
    )
    if normalization_std <= 0:
        raise ValueError(f"normalization std must be positive at {path}")
    _validate_leakage(path, target, metadata, signatures, method)
    expected_sources = set(TARGETS) - {target}
    recomputed_threshold = _validate_source_metrics(path, source_metrics, expected_sources)
    _validate_training_history(
        path,
        history,
        resolved,
        metadata,
        source_metrics,
        expected_sources,
    )
    if not math.isclose(
        source_threshold, recomputed_threshold, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(
            f"source_threshold disagrees with source validation labels/scores at {path}"
        )
    frame, recomputed = _validated_predictions(
        path, target, metrics, source_threshold
    )
    if metadata.get("target_prediction_count") != len(frame):
        raise ValueError(f"target prediction count disagrees at {path}")

    grouping_unit, cluster_ids = _grouping_for_inference(frame)
    patient = None
    if grouping_unit == "patient":
        patient = patient_level_auc(
            frame["label"].astype(int).tolist(),
            frame["score"].astype(float).tolist(),
            cluster_ids,
        )
    interval = cluster_bootstrap_auc(
        frame["label"].astype(int).tolist(),
        frame["score"].astype(float).tolist(),
        cluster_ids,
        n_bootstrap=n_bootstrap,
        seed=_bootstrap_seed(target, method, seed),
    )
    run_record: dict[str, object] = {
        "target": target,
        "method": method,
        "seed": seed,
        **recomputed,
    }
    patient_record = None
    if patient is not None:
        patient_record = {
            "target": target,
            "method": method,
            "seed": seed,
            "aggregation": "single_seed",
            "patient_auc": patient["auc"],
            "n_patients": patient["n"],
            "patient_id_sources": ";".join(
                sorted(set(frame["patient_id_source"].astype(str)))
            ),
        }
    bootstrap_record = {
        "target": target,
        "method": method,
        "seed": seed,
        "aggregation": "single_seed",
        "seed_count": 1,
        "unit": grouping_unit,
        "level": (
            "patient_clustered_image_auc"
            if grouping_unit == "patient"
            else "image_singleton_cluster_auc"
        ),
        "point_auc": recomputed["auc"],
        "lower_95": interval["lower"],
        "upper_95": interval["upper"],
        "valid_replicates": interval["valid_replicates"],
        "requested_replicates": n_bootstrap,
    }
    contract.update(
        {
            "normalization": normalization,
            "signatures": signatures,
        }
    )
    return run_record, patient_record, bootstrap_record, frame, contract


def _per_target_table(run_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for method in METHODS:
            local = run_frame[
                (run_frame["target"] == target) & (run_frame["method"] == method)
            ]
            row: dict[str, object] = {
                "target": target,
                "method": method,
                "seed_count": len(local),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(local[metric].mean())
                row[f"{metric}_std"] = (
                    float(local[metric].std(ddof=1)) if len(local) > 1 else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _macro_table(per_target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        local = per_target[per_target["method"] == method]
        row: dict[str, object] = {"method": method, "target_count": len(local)}
        for metric in METRICS:
            values = local[f"{metric}_mean"].to_numpy(dtype=float)
            row[f"{metric}_macro_mean"] = float(np.mean(values))
            if metric == "loss":
                row[f"{metric}_worst_target_mean"] = float(np.max(values))
            else:
                row[f"{metric}_worst_target_mean"] = float(np.min(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _seed_mean_bootstrap_records(
    predictions: Mapping[tuple[str, str, int], pd.DataFrame],
    seeds: tuple[int, ...],
    n_bootstrap: int,
) -> list[dict[str, object]]:
    if len(seeds) <= 1:
        return []
    records: list[dict[str, object]] = []
    identity_columns = [
        "manifest_row",
        "dataset",
        "image_path",
        "patient_id",
        "patient_id_source",
        "patient_grouping_reliable",
        "label",
    ]
    for target in TARGETS:
        for method in METHODS:
            frames = [predictions[(target, method, seed)] for seed in seeds]
            reference = frames[0]
            reference_identity = reference.loc[:, identity_columns].reset_index(drop=True)
            for seed, frame in zip(seeds[1:], frames[1:], strict=True):
                if not reference_identity.equals(
                    frame.loc[:, identity_columns].reset_index(drop=True)
                ):
                    raise ValueError(
                        "seed prediction identity mismatch for "
                        f"target={target}, method={method}, seed={seed}"
                    )
            mean_scores = np.mean(
                np.stack([frame["score"].to_numpy(dtype=float) for frame in frames]),
                axis=0,
            )
            labels = reference["label"].to_numpy(dtype=int)
            grouping_unit, cluster_ids = _grouping_for_inference(reference)
            interval = cluster_bootstrap_auc(
                labels.tolist(),
                mean_scores.tolist(),
                cluster_ids,
                n_bootstrap=n_bootstrap,
                seed=_bootstrap_seed(target, method, 0x5EED),
            )
            records.append(
                {
                    "target": target,
                    "method": method,
                    "seed": None,
                    "aggregation": "seed_mean",
                    "seed_count": len(seeds),
                    "unit": grouping_unit,
                    "level": (
                        "patient_clustered_seed_mean_image_auc"
                        if grouping_unit == "patient"
                        else "image_singleton_clustered_seed_mean_auc"
                    ),
                    "point_auc": float(roc_auc_score(labels, mean_scores)),
                    "lower_95": interval["lower"],
                    "upper_95": interval["upper"],
                    "valid_replicates": interval["valid_replicates"],
                    "requested_replicates": n_bootstrap,
                }
            )
    return records


def _write_summary(
    staging: Path,
    per_target: pd.DataFrame,
    patient: pd.DataFrame,
    bootstrap: pd.DataFrame,
    macro: pd.DataFrame,
    gate: dict[str, object],
    required_seeds: tuple[int, ...],
) -> None:
    per_target.to_csv(staging / "per_target_metrics.csv", index=False)
    patient.to_csv(staging / "patient_level_metrics.csv", index=False)
    bootstrap.to_csv(staging / "bootstrap_intervals.csv", index=False)
    macro.to_csv(staging / "macro_worst_metrics.csv", index=False)
    (staging / "phase2_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# FedUS-DG MVP aggregation",
        "",
        f"- Phase: {gate['phase']}",
        f"- Required seeds: {', '.join(map(str, required_seeds))}",
        f"- Complete run cells: {len(TARGETS) * len(METHODS) * len(required_seeds)}",
        f"- Numerical phase-2 gate: {'PASS' if gate['pass'] else 'FAIL'}",
        f"- Strict Depth > Whole wins: {gate['depth_wins']}/5",
        f"- MVP > FedAvg targets: {gate['mvp_target_wins']}/5",
        f"- MVP macro AUC: {gate['mvp_macro_auc']:.6f}",
        f"- FedAvg macro AUC: {gate['fedavg_macro_auc']:.6f}",
        "",
        "The visual-quality condition is reviewed separately and is not inferred from metrics.",
        "Phase 3 statistics first aggregate seeds within each target (sample SD, ddof=1), then macro-average target means.",
        "",
    ]
    (staging / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    missing = [filename for filename in SUMMARY_FILES if not (staging / filename).is_file()]
    if missing:
        raise RuntimeError(f"summary staging is incomplete: {missing}")


def _publish(staging: Path, summary: Path, force: bool) -> None:
    backup: Path | None = None
    try:
        if os.path.lexists(summary):
            if not force:
                raise FileExistsError(f"summary already exists: {summary}")
            if summary.is_symlink() or not summary.is_dir():
                raise FileExistsError(f"summary destination is unsafe: {summary}")
            backup = Path(
                tempfile.mkdtemp(dir=summary.parent, prefix="._summary.backup-")
            )
            backup.rmdir()
            os.replace(summary, backup)
        os.replace(staging, summary)
    except BaseException:
        if backup is not None and backup.exists() and not os.path.lexists(summary):
            os.replace(backup, summary)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def aggregate_runs(
    output_root: str | Path,
    *,
    required_seeds: Sequence[int] = (42,),
    n_bootstrap: int = 2000,
    force: bool = False,
) -> Path:
    """Validate the complete result matrix and atomically publish six summaries."""
    root = Path(output_root)
    seeds = tuple(required_seeds)
    if seeds not in ((42,), (42, 43, 44)):
        raise ValueError("required_seeds must be phase 2 (42) or phase 3 (42,43,44)")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    summary = root / "_summary"
    if os.path.lexists(summary) and not force:
        raise FileExistsError(f"summary already exists: {summary}")

    matrix = _validate_matrix(root, seeds)
    run_records: list[dict[str, object]] = []
    patient_records: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []
    predictions_by_cell: dict[tuple[str, str, int], pd.DataFrame] = {}
    expected_budget: dict[str, object] | None = None
    expected_reproducibility: dict[str, object] | None = None
    manifest_by_target: dict[str, dict] = {}
    normalization_by_target: dict[str, dict] = {}
    signatures_by_target_group: dict[tuple[str, str], dict] = {}
    for target in TARGETS:
        for method in METHODS:
            for seed in seeds:
                path = matrix[(target, method, seed)]
                resolved = _read_object(path / "resolved_config.json", "resolved configuration")
                budget = {key: resolved.get(key) for key in BUDGET_KEYS}
                if budget.get("rounds") != 30 or any(
                    budget.get(key) != value for key, value in FIXED_BUDGET.items()
                ):
                    raise ValueError(f"run does not use an approved training budget at {path}")
                if expected_budget is None:
                    expected_budget = budget
                elif budget != expected_budget:
                    raise ValueError(f"training budget mismatch at {path}")
                manifests = _read_object(path / "manifest_hashes.json", "manifest hashes")
                incumbent = manifest_by_target.setdefault(target, manifests)
                if manifests != incumbent:
                    raise ValueError(f"manifest hash mismatch across methods at {path}")
                run, patient, interval, predictions, contract = _validate_run(
                    path, target, method, seed, n_bootstrap
                )
                reproducibility = {
                    "implementation_sha256": contract["implementation_sha256"],
                    "runtime_versions": contract["runtime_versions"],
                }
                if expected_reproducibility is None:
                    expected_reproducibility = reproducibility
                elif reproducibility != expected_reproducibility:
                    raise ValueError(f"implementation snapshot mismatch at {path}")

                normalization = contract["normalization"]
                incumbent_normalization = normalization_by_target.setdefault(
                    target, normalization
                )
                if normalization != incumbent_normalization:
                    raise ValueError(
                        f"normalization mismatch across methods/seeds at {path}"
                    )

                signature_group = (
                    "whole"
                    if method == "whole"
                    else "depth"
                    if method
                    in {"depth", "fedus_dg_mvp", "factor", "virtual", "fedphys", "control"}
                    else "non_spectral"
                )
                signature_key = (target, signature_group)
                signatures = contract["signatures"]
                incumbent_signatures = signatures_by_target_group.setdefault(
                    signature_key, signatures
                )
                if signatures != incumbent_signatures:
                    raise ValueError(
                        f"signature mismatch across equivalent runs at {path}"
                    )
                run_records.append(run)
                if patient is not None:
                    patient_records.append(patient)
                bootstrap_records.append(interval)
                predictions_by_cell[(target, method, seed)] = predictions

    bootstrap_records.extend(
        _seed_mean_bootstrap_records(predictions_by_cell, seeds, n_bootstrap)
    )

    run_frame = pd.DataFrame(run_records)
    per_target = _per_target_table(run_frame)
    macro = _macro_table(per_target)
    auc = {
        method: per_target.loc[
            per_target["method"] == method, "auc_mean"
        ].to_numpy(dtype=float)
        for method in METHODS
    }
    gate = evaluate_phase2_gate(auc)
    gate["phase"] = 2 if seeds == (42,) else 3
    gate["required_seeds"] = list(seeds)
    gate["aggregation_order"] = "seed_within_target_then_macro_across_targets"
    gate["seed_std_ddof"] = 1
    gate["visual_quality_gate"] = "separate_manual_review"

    staging = Path(tempfile.mkdtemp(dir=root, prefix="._summary.staging-"))
    try:
        _write_summary(
            staging,
            per_target,
            pd.DataFrame(patient_records),
            pd.DataFrame(bootstrap_records),
            macro,
            gate,
            seeds,
        )
        _publish(staging, summary, force)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/fedusdg_mvp")
    )
    parser.add_argument(
        "--phase", type=int, choices=(2, 3), default=2, help="phase determines seeds"
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = (42,) if args.phase == 2 else (42, 43, 44)
    try:
        output = aggregate_runs(
            args.output_root,
            required_seeds=seeds,
            n_bootstrap=args.n_bootstrap,
            force=args.force,
        )
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
