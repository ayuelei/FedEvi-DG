import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from fedusdg_mvp.manifest import split_protocol_config
from fedusdg_mvp.metrics import binary_metrics, select_source_threshold
from fedusdg_mvp.provenance import canonical_hash
from scripts.aggregate_fedusdg_mvp import aggregate_runs, evaluate_phase2_gate


TARGETS = ("BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT")
METHODS = (
    "fedavg",
    "fedprox",
    "whole",
    "depth",
    "fedus_dg_mvp",
    "factor",
    "virtual",
    "fedphys",
    "control",
    "virtual_cons",
    "control_full",
    "fedbn",
    "fedfat",
)
ARTIFACTS = (
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scores_for_auc(value: float) -> list[float]:
    choices = {
        1.0: [0.1, 0.2, 0.8, 0.9],
        0.75: [0.1, 0.7, 0.6, 0.9],
        0.5: [0.2, 0.8, 0.1, 0.9],
        0.25: [0.6, 0.8, 0.1, 0.7],
        0.0: [0.8, 0.9, 0.1, 0.2],
    }
    return choices[value]


def _write_run(root: Path, target: str, method: str, seed: int, *, auc=1.0) -> Path:
    path = root / f"target_{target}" / method / f"seed_{seed}"
    path.mkdir(parents=True)
    sources = sorted(set(TARGETS) - {target})
    labels = [0, 0, 1, 1]
    scores = _scores_for_auc(auc)
    predictions = [int(score >= 0.5) for score in scores]
    split_protocol = split_protocol_config(target)
    split_protocol_hash = canonical_hash(split_protocol)
    implementation_files = {"fedusdg_mvp/config.py": "a" * 64}
    resolved = {
        "target_domain": target,
        "target": target,
        "method": method,
        "seed": seed,
        "domains": list(TARGETS),
        "split_seed": 20260728,
        "val_fraction": 0.15,
        "rounds": 30,
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
        "implementation_sha256": canonical_hash(implementation_files),
        "implementation_files": implementation_files,
        "runtime_versions": {
            "python": "3.12.0",
            "torch": "2.4.1",
            "torchvision": "0.19.1",
        },
        "smoke": False,
        "split_protocol": split_protocol,
        "split_protocol_hash": split_protocol_hash,
    }
    manifests = {
        "source_train.csv": f"{target}-train",
        "source_val.csv": f"{target}-val",
        "target_test.csv": f"{target}-test",
        "metadata.json": f"{target}-metadata",
        "COMPLETE": f"{target}-complete",
    }
    signature_metadata = {}
    signature_values = {}
    if method in {
        "whole",
        "depth",
        "fedus_dg_mvp",
        "factor",
        "virtual",
        "fedphys",
        "control",
    }:
        signature_metadata = {
            "computed": True,
            "target_excluded": True,
            "source_split": "source_train",
            "domains": {domain: {"sample_count": 2} for domain in sources},
        }
        signature_values = {domain: {"mean": [[0.0]], "std": [[1.0]]} for domain in sources}
    target_metrics = binary_metrics(labels, scores, threshold=0.5)
    source_metrics = {
        domain: binary_metrics([0, 1], [0.2, 0.8], threshold=0.5)
        for domain in sources
    }
    source_threshold = select_source_threshold(
        {domain: value["labels"] for domain, value in source_metrics.items()},
        {domain: value["scores"] for domain, value in source_metrics.items()},
    )
    metadata = {
        "target_evaluation_count": 1,
        "target_prediction_count": 4,
        "selected_checkpoint_kind": "global",
        "source_threshold": source_threshold,
        "selection_domains": sources,
        "target_domain": target,
        "method": method,
        "seed": seed,
        "best_round": 5,
        "implementation_sha256": resolved["implementation_sha256"],
        "runtime_versions": resolved["runtime_versions"],
        "manifest_hashes": manifests,
        "split_protocol": split_protocol,
        "split_protocol_hash": split_protocol_hash,
    }
    json_values = {
        "resolved_config.json": resolved,
        "manifest_hashes.json": manifests,
        "normalization.json": {
            "mean": 0.5,
            "std": 0.2,
            "source_split": "source_train",
            "target_excluded": True,
        },
        "signatures.json": {"metadata": signature_metadata, "values": signature_values},
        "training_history.json": [
            {
                "round": round_index,
                "client_train_loss": {domain: 0.2 for domain in sources},
                "source_validation": (
                    {
                        domain: {
                            metric: source_metrics[domain][metric]
                            for metric in (
                                "auc",
                                "balanced_accuracy",
                                "sensitivity",
                                "specificity",
                                "macro_f1",
                                "loss",
                            )
                        }
                        for domain in sources
                    }
                    if round_index % 5 == 0
                    else None
                ),
                "macro_source_auc": 1.0 if round_index % 5 == 0 else None,
                "macro_source_loss": (
                    float(
                        np.mean(
                            [source_metrics[domain]["loss"] for domain in sources]
                        )
                    )
                    if round_index % 5 == 0
                    else None
                ),
                "selected_checkpoint": round_index == 5,
            }
            for round_index in range(1, 31)
        ],
        "source_val_metrics.json": source_metrics,
        "target_metrics.json": target_metrics,
        "run_metadata.json": metadata,
    }
    for filename, value in json_values.items():
        (path / filename).write_text(json.dumps(value), encoding="utf-8")
    pd.DataFrame(
        {
            "manifest_row": range(4),
            "dataset": [target] * 4,
            "image_path": [f"{target}/{index}.png" for index in range(4)],
            "patient_id": [f"{target}:benign", f"{target}:benign", f"{target}:malignant", f"{target}:malignant"],
            "patient_id_source": ["metadata"] * 4,
            "patient_grouping_reliable": [True] * 4,
            "label": labels,
            "score": scores,
            "prediction": predictions,
        }
    ).to_csv(path / "target_predictions.csv", index=False)
    (path / "best_global.pt").write_bytes(b"checkpoint")
    metadata["hashes"] = {
        filename: _sha256(path / filename)
        for filename in ARTIFACTS
        if filename != "run_metadata.json"
    }
    (path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    hashes = {filename: _sha256(path / filename) for filename in ARTIFACTS}
    (path / "COMPLETE").write_text(json.dumps({"artifact_sha256": hashes}), encoding="utf-8")
    return path


def _write_matrix(root: Path, seeds=(42,), auc_by_cell=None) -> None:
    for target in TARGETS:
        for method in METHODS:
            for seed in seeds:
                auc = 1.0 if auc_by_cell is None else auc_by_cell(target, method, seed)
                _write_run(root, target, method, seed, auc=auc)


def test_gate_requires_three_depth_wins_and_mvp_not_below_fedavg():
    auc = {
        "fedavg": [0.7] * 5,
        "whole": [0.71] * 5,
        "depth": [0.72, 0.72, 0.70, 0.72, 0.70],
        "fedus_dg_mvp": [0.73, 0.72, 0.71, 0.72, 0.70],
    }
    decision = evaluate_phase2_gate(auc)
    assert decision["depth_wins"] == 3
    assert decision["pass"] is True


def test_incomplete_run_is_rejected(tmp_path):
    (tmp_path / "target_UDIAT/fedavg/seed_42").mkdir(parents=True)
    with pytest.raises(ValueError, match="COMPLETE"):
        aggregate_runs(tmp_path, required_seeds=(42,))


def test_gate_is_strict_and_requires_two_mvp_target_wins():
    tied = {
        "fedavg": [0.7] * 5,
        "whole": [0.7] * 5,
        "depth": [0.7] * 5,
        "fedus_dg_mvp": [0.8, 0.7, 0.7, 0.7, 0.7],
    }
    decision = evaluate_phase2_gate(tied)
    assert decision["depth_wins"] == 0
    assert decision["mvp_target_wins"] == 1
    assert decision["mvp_macro_not_below_fedavg"] is True
    assert decision["pass"] is False


def test_complete_phase2_matrix_writes_six_artifacts_transactionally(tmp_path):
    _write_matrix(tmp_path)
    summary = aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=40)

    assert summary == tmp_path / "_summary"
    assert {path.name for path in summary.iterdir()} == {
        "per_target_metrics.csv",
        "patient_level_metrics.csv",
        "bootstrap_intervals.csv",
        "macro_worst_metrics.csv",
        "phase2_gate.json",
        "summary.md",
    }
    per_target = pd.read_csv(summary / "per_target_metrics.csv")
    assert len(per_target) == 65
    assert set(per_target["seed_count"]) == {1}
    assert per_target["auc_mean"].eq(1.0).all()
    assert per_target["auc_std"].isna().all()
    patient = pd.read_csv(summary / "patient_level_metrics.csv")
    bootstrap = pd.read_csv(summary / "bootstrap_intervals.csv")
    assert len(patient) == len(bootstrap) == 65
    assert set(bootstrap["level"]) == {"patient_clustered_image_auc"}
    assert set(bootstrap["unit"]) == {"patient"}
    gate = json.loads((summary / "phase2_gate.json").read_text(encoding="utf-8"))
    assert gate["target_count"] == 5

    with pytest.raises(FileExistsError, match="summary"):
        aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)


def test_prediction_scores_survive_csv_round_trip_exactly(tmp_path):
    _write_matrix(tmp_path)
    run = tmp_path / "target_UDIAT" / "fedavg" / "seed_42"
    prediction_path = run / "target_predictions.csv"
    frame = pd.read_csv(prediction_path)
    scores = [0.10000000000000002, 0.2, 0.8, 0.9]
    frame["score"] = scores
    frame.to_csv(prediction_path, index=False)

    metrics_path = run / "target_metrics.json"
    metrics = binary_metrics(frame["label"].tolist(), scores, threshold=0.5)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _rehash(run, "target_predictions.csv")
    _rehash(run, "target_metrics.json")

    summary = aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)
    assert summary.is_dir()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda run: (run / "COMPLETE").unlink(), "COMPLETE"),
        (
            lambda run: (run / "target_metrics.json").write_text("{}", encoding="utf-8"),
            "SHA-256",
        ),
        (
            lambda run: _mutate_json(run / "resolved_config.json", "rounds", 31, run),
            "budget",
        ),
        (
            lambda run: _mutate_json(run / "run_metadata.json", "target_evaluation_count", 2, run),
            "target_evaluation_count",
        ),
        (
            lambda run: _mutate_json(run / "run_metadata.json", "method", "depth", run),
            "identity",
        ),
    ],
)
def test_invalid_cell_is_rejected_without_partial_summary(tmp_path, mutate, message):
    _write_matrix(tmp_path)
    run = tmp_path / "target_UDIAT" / "fedavg" / "seed_42"
    mutate(run)
    with pytest.raises(ValueError, match=message):
        aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)
    assert not (tmp_path / "_summary").exists()


def test_reproducibility_contract_rejects_resigned_tampering(tmp_path):
    implementation = tmp_path / "implementation"
    _write_matrix(implementation)
    run = implementation / "target_UDIAT" / "fedavg" / "seed_42"
    _mutate_json(
        run / "resolved_config.json",
        "implementation_sha256",
        "2" * 64,
        run,
    )
    with pytest.raises(ValueError, match="implementation snapshot mismatch"):
        aggregate_runs(implementation, required_seeds=(42,), n_bootstrap=20)

    normalization = tmp_path / "normalization"
    _write_matrix(normalization)
    run = normalization / "target_UDIAT" / "whole" / "seed_42"
    _mutate_json(run / "normalization.json", "mean", 0.51, run)
    with pytest.raises(ValueError, match="normalization mismatch"):
        aggregate_runs(normalization, required_seeds=(42,), n_bootstrap=20)

    signatures = tmp_path / "signature-values"
    _write_matrix(signatures)
    run = signatures / "target_UDIAT" / "depth" / "seed_42"
    payload = json.loads((run / "signatures.json").read_text(encoding="utf-8"))
    first_source = next(iter(payload["values"]))
    payload["values"][first_source]["mean"] = [[0.1]]
    (run / "signatures.json").write_text(json.dumps(payload), encoding="utf-8")
    _rehash(run, "signatures.json")
    with pytest.raises(ValueError, match="signature mismatch"):
        aggregate_runs(signatures, required_seeds=(42,), n_bootstrap=20)

    algorithm = tmp_path / "algorithm"
    _write_matrix(algorithm)
    for config_path in algorithm.glob("target_*/*/seed_42/resolved_config.json"):
        _mutate_json(config_path, "lambda_max", 0.9, config_path.parent)
    with pytest.raises(ValueError, match="approved training budget"):
        aggregate_runs(algorithm, required_seeds=(42,), n_bootstrap=20)


def test_run_metadata_hashes_and_training_history_are_verified(tmp_path):
    hashes_root = tmp_path / "metadata-hashes"
    _write_matrix(hashes_root)
    run = hashes_root / "target_UDIAT" / "fedavg" / "seed_42"
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["hashes"]["target_metrics.json"] = "0" * 64
    (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _rehash(run, "run_metadata.json")
    with pytest.raises(ValueError, match="run metadata artifact hashes"):
        aggregate_runs(hashes_root, required_seeds=(42,), n_bootstrap=20)

    history_root = tmp_path / "history"
    _write_matrix(history_root)
    run = history_root / "target_UDIAT" / "fedavg" / "seed_42"
    history_path = run / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history[5]["round"] = 99
    history_path.write_text(json.dumps(history), encoding="utf-8")
    _rehash(run, "training_history.json")
    with pytest.raises(ValueError, match="training history rounds"):
        aggregate_runs(history_root, required_seeds=(42,), n_bootstrap=20)


def _mutate_json(path: Path, key: str, value, run: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    _rehash(run, path.name)


def test_rejects_manifest_mismatch_leakage_unknown_and_nonfinite_cells(tmp_path):
    _write_matrix(tmp_path)
    one = tmp_path / "target_UDIAT" / "whole" / "seed_42"
    _mutate_json(one / "manifest_hashes.json", "source_train.csv", "different", one)
    with pytest.raises(ValueError, match="manifest hash"):
        aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)

    # Restore a clean matrix in a separate root for each independent hard gate.
    leakage = tmp_path / "leakage"
    _write_matrix(leakage)
    run = leakage / "target_UDIAT" / "depth" / "seed_42"
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["selection_domains"].append("UDIAT")
    (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _rehash(run, "run_metadata.json")
    with pytest.raises(ValueError, match="selection_domains"):
        aggregate_runs(leakage, required_seeds=(42,), n_bootstrap=20)

    signatures = tmp_path / "signatures"
    _write_matrix(signatures)
    run = signatures / "target_UDIAT" / "depth" / "seed_42"
    payload = json.loads((run / "signatures.json").read_text(encoding="utf-8"))
    payload["metadata"]["domains"]["UDIAT"] = {"sample_count": 1}
    (run / "signatures.json").write_text(json.dumps(payload), encoding="utf-8")
    _rehash(run, "signatures.json")
    with pytest.raises(ValueError, match="signature"):
        aggregate_runs(signatures, required_seeds=(42,), n_bootstrap=20)

    nonfinite = tmp_path / "nonfinite"
    _write_matrix(nonfinite)
    run = nonfinite / "target_UDIAT" / "fedavg" / "seed_42"
    payload = json.loads((run / "target_metrics.json").read_text(encoding="utf-8"))
    payload["auc"] = float("nan")
    (run / "target_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    _rehash(run, "target_metrics.json")
    with pytest.raises(ValueError, match="finite"):
        aggregate_runs(nonfinite, required_seeds=(42,), n_bootstrap=20)

    source_nonfinite = tmp_path / "source-nonfinite"
    _write_matrix(source_nonfinite)
    run = source_nonfinite / "target_UDIAT" / "fedavg" / "seed_42"
    payload = json.loads((run / "source_val_metrics.json").read_text(encoding="utf-8"))
    first_source = next(iter(payload))
    payload[first_source]["auc"] = float("inf")
    (run / "source_val_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    _rehash(run, "source_val_metrics.json")
    with pytest.raises(ValueError, match="source metric.*finite"):
        aggregate_runs(source_nonfinite, required_seeds=(42,), n_bootstrap=20)

    unknown = tmp_path / "unknown"
    _write_matrix(unknown)
    _write_run(unknown, "UDIAT", "fedavg", 99)
    with pytest.raises(ValueError, match="unknown cell"):
        aggregate_runs(unknown, required_seeds=(42,), n_bootstrap=20)


def _rehash(run: Path, filename: str) -> None:
    marker_path = run / "COMPLETE"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_sha256"][filename] = _sha256(run / filename)
    if filename != "run_metadata.json":
        metadata_path = run / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "hashes" in metadata:
            metadata["hashes"][filename] = _sha256(run / filename)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            marker["artifact_sha256"]["run_metadata.json"] = _sha256(metadata_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def test_phase3_aggregates_seeds_inside_target_before_macro(tmp_path):
    def auc(target, method, seed):
        if method != "fedus_dg_mvp":
            return 0.5
        if target == "BUSBRA":
            return {42: 0.5, 43: 0.75, 44: 1.0}[seed]
        return 1.0

    _write_matrix(tmp_path, seeds=(42, 43, 44), auc_by_cell=auc)
    summary = aggregate_runs(
        tmp_path, required_seeds=(42, 43, 44), n_bootstrap=20
    )
    per_target = pd.read_csv(summary / "per_target_metrics.csv")
    row = per_target[
        (per_target.target == "BUSBRA")
        & (per_target.method == "fedus_dg_mvp")
    ].iloc[0]
    assert row.auc_mean == pytest.approx(0.75)
    assert row.auc_std == pytest.approx(0.25)  # sample std, ddof=1
    macro = pd.read_csv(summary / "macro_worst_metrics.csv")
    mvp = macro[macro.method == "fedus_dg_mvp"].iloc[0]
    assert mvp.auc_macro_mean == pytest.approx((0.75 + 4.0) / 5)
    assert mvp.auc_worst_target_mean == pytest.approx(0.75)
    assert len(pd.read_csv(summary / "patient_level_metrics.csv")) == 195
    bootstrap = pd.read_csv(summary / "bootstrap_intervals.csv")
    assert len(bootstrap) == 260
    seed_mean = bootstrap[bootstrap.aggregation == "seed_mean"]
    assert len(seed_mean) == 65
    assert set(seed_mean.seed_count) == {3}
    assert set(seed_mean.unit) == {"patient"}


def test_source_threshold_and_target_prediction_tampering_are_rejected(tmp_path):
    threshold_root = tmp_path / "threshold"
    _write_matrix(threshold_root)
    run = threshold_root / "target_UDIAT" / "fedavg" / "seed_42"
    _mutate_json(run / "run_metadata.json", "source_threshold", 0.49, run)
    with pytest.raises(ValueError, match="source_threshold.*source validation"):
        aggregate_runs(threshold_root, required_seeds=(42,), n_bootstrap=20)

    prediction_root = tmp_path / "prediction"
    _write_matrix(prediction_root)
    run = prediction_root / "target_UDIAT" / "fedavg" / "seed_42"
    prediction_path = run / "target_predictions.csv"
    frame = pd.read_csv(prediction_path)
    frame.loc[0, "prediction"] = 1
    frame.to_csv(prediction_path, index=False)
    metrics_path = run / "target_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["predictions"] = frame.prediction.astype(int).tolist()
    tn, fp, fn, tp = confusion_matrix(
        frame.label, frame.prediction, labels=[0, 1]
    ).ravel()
    metrics.update(
        {
            "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "macro_f1": f1_score(frame.label, frame.prediction, average="macro"),
        }
    )
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _rehash(run, "target_predictions.csv")
    _rehash(run, "target_metrics.json")
    with pytest.raises(ValueError, match="prediction.*source_threshold"):
        aggregate_runs(prediction_root, required_seeds=(42,), n_bootstrap=20)


@pytest.mark.parametrize(
    ("filename", "key", "value"),
    [
        ("resolved_config.json", "split_seed", 7),
        ("resolved_config.json", "val_fraction", 0.2),
        ("resolved_config.json", "split_protocol_hash", "0" * 64),
        ("run_metadata.json", "split_protocol", {"protocol": "tampered"}),
    ],
)
def test_aggregation_rejects_resigned_split_protocol_tampering(
    tmp_path, filename, key, value
):
    _write_matrix(tmp_path)
    run = tmp_path / "target_UDIAT" / "fedavg" / "seed_42"
    _mutate_json(run / filename, key, value, run)
    with pytest.raises(ValueError, match="split protocol"):
        aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)


def test_unreliable_grouping_uses_image_singletons_and_omits_patient_metric(tmp_path):
    _write_matrix(tmp_path)
    run = tmp_path / "target_UDIAT" / "fedavg" / "seed_42"
    path = run / "target_predictions.csv"
    frame = pd.read_csv(path)
    frame["patient_id"] = "unreliable-record"
    frame["patient_id_source"] = "original_source_id"
    frame["patient_grouping_reliable"] = False
    frame.to_csv(path, index=False)
    _rehash(run, "target_predictions.csv")

    summary = aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)
    patient = pd.read_csv(summary / "patient_level_metrics.csv")
    assert not (
        (patient.target == "UDIAT")
        & (patient.method == "fedavg")
        & (patient.seed == 42)
    ).any()
    bootstrap = pd.read_csv(summary / "bootstrap_intervals.csv")
    row = bootstrap[
        (bootstrap.target == "UDIAT")
        & (bootstrap.method == "fedavg")
        & (bootstrap.seed == 42)
    ].iloc[0]
    assert row.unit == "image"
    assert row.level == "image_singleton_cluster_auc"


def test_phase3_seed_mean_requires_aligned_target_sample_identity(tmp_path):
    _write_matrix(tmp_path, seeds=(42, 43, 44))
    run = tmp_path / "target_UDIAT" / "fedavg" / "seed_43"
    path = run / "target_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "image_path"] = "UDIAT/different.png"
    frame.to_csv(path, index=False)
    _rehash(run, "target_predictions.csv")

    with pytest.raises(ValueError, match="seed prediction identity mismatch"):
        aggregate_runs(
            tmp_path, required_seeds=(42, 43, 44), n_bootstrap=20
        )


def test_force_replaces_summary_only_after_new_summary_is_complete(tmp_path):
    _write_matrix(tmp_path)
    summary = aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)
    (summary / "old.txt").write_text("old", encoding="utf-8")
    replaced = aggregate_runs(
        tmp_path, required_seeds=(42,), n_bootstrap=20, force=True
    )
    assert replaced == summary
    assert not (summary / "old.txt").exists()
    assert list(tmp_path.glob("._summary.staging-*")) == []
    assert list(tmp_path.glob("._summary.backup-*")) == []


def test_uniform_wrong_budget_and_empty_unknown_domain_are_rejected(tmp_path):
    wrong_budget = tmp_path / "wrong-budget"
    _write_matrix(wrong_budget)
    for config_path in wrong_budget.glob("target_*/*/seed_42/resolved_config.json"):
        _mutate_json(config_path, "rounds", 31, config_path.parent)
    with pytest.raises(ValueError, match="approved training budget"):
        aggregate_runs(wrong_budget, required_seeds=(42,), n_bootstrap=20)

    unknown = tmp_path / "empty-unknown"
    _write_matrix(unknown)
    (unknown / "target_OASBUD").mkdir()
    with pytest.raises(ValueError, match="unknown target directory"):
        aggregate_runs(unknown, required_seeds=(42,), n_bootstrap=20)


def test_uniform_fifty_round_budget_is_not_phase2_or_phase3(tmp_path):
    _write_matrix(tmp_path)
    for config_path in tmp_path.glob("target_*/*/seed_42/resolved_config.json"):
        _mutate_json(config_path, "rounds", 50, config_path.parent)
    with pytest.raises(ValueError, match="approved training budget"):
        aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)


@pytest.mark.parametrize(
    ("filename", "key", "value", "message"),
    [
        ("normalization.json", "target_excluded", False, "normalization.*target"),
        ("normalization.json", "source_split", "all", "normalization.*source_train"),
        ("normalization.json", "mean", float("nan"), "normalization mean.*finite"),
        ("normalization.json", "std", 0.0, "normalization std.*positive"),
        ("run_metadata.json", "selected_checkpoint_kind", "local", "selected_checkpoint_kind"),
        ("run_metadata.json", "source_threshold", float("nan"), "source_threshold.*finite"),
        ("run_metadata.json", "source_threshold", 1.1, r"source_threshold.*\[0, 1\]"),
    ],
)
def test_protocol_credential_tampering_is_rejected(
    tmp_path, filename, key, value, message
):
    _write_matrix(tmp_path)
    run = tmp_path / "target_UDIAT" / "fedavg" / "seed_42"
    _mutate_json(run / filename, key, value, run)
    with pytest.raises(ValueError, match=message):
        aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)


def test_duplicate_selection_domain_and_unknown_hierarchy_entries_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate"
    _write_matrix(duplicate)
    run = duplicate / "target_UDIAT" / "fedavg" / "seed_42"
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["selection_domains"].append(metadata["selection_domains"][0])
    (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _rehash(run, "run_metadata.json")
    with pytest.raises(ValueError, match="four unique source domains"):
        aggregate_runs(duplicate, required_seeds=(42,), n_bootstrap=20)

    unknown_method_entry = tmp_path / "unknown-method-entry"
    _write_matrix(unknown_method_entry)
    (unknown_method_entry / "target_UDIAT" / "notes.txt").write_text("no")
    with pytest.raises(ValueError, match="unknown method entry"):
        aggregate_runs(unknown_method_entry, required_seeds=(42,), n_bootstrap=20)

    unknown_seed_entry = tmp_path / "unknown-seed-entry"
    _write_matrix(unknown_seed_entry)
    (unknown_seed_entry / "target_UDIAT" / "fedavg" / "latest").write_text("no")
    with pytest.raises(ValueError, match="unknown seed entry"):
        aggregate_runs(unknown_seed_entry, required_seeds=(42,), n_bootstrap=20)


def test_parent_symlink_in_run_hierarchy_is_rejected(tmp_path):
    _write_matrix(tmp_path)
    method = tmp_path / "target_UDIAT" / "fedavg"
    external = tmp_path.parent / f"{tmp_path.name}-external-fedavg"
    shutil.move(method, external)
    method.symlink_to(external, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="symbolic link"):
            aggregate_runs(tmp_path, required_seeds=(42,), n_bootstrap=20)
    finally:
        method.unlink(missing_ok=True)
        shutil.rmtree(external, ignore_errors=True)


def test_cli_help_runs_without_installed_package(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/aggregate_fedusdg_mvp.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--phase" in completed.stdout
