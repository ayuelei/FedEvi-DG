#!/usr/bin/env python3
"""Aggregate the frozen three-seed QAMEBI external validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fedusdg_mvp.metrics import (
    binary_metrics,
    clinical_reliability_metrics,
    select_source_threshold,
)


METHODS = ("fedavg", "fedevi_mtl", "fedevi_monotonic")
SEEDS = (42, 43, 44)
BOOTSTRAP_SEED = 20260925


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_run(run: Path) -> None:
    marker = json.loads((run / "COMPLETE").read_text(encoding="utf-8"))
    for filename, expected in marker["artifact_sha256"].items():
        actual = _sha256(run / filename)
        if actual != expected:
            raise RuntimeError(f"artifact hash mismatch: {run / filename}")


def _source_ensemble_threshold(runs: list[Path]) -> float:
    payloads = [
        json.loads((run / "source_val_metrics.json").read_text(encoding="utf-8"))
        for run in runs
    ]
    domains = tuple(sorted(payloads[0]))
    if any(tuple(sorted(payload)) != domains for payload in payloads[1:]):
        raise RuntimeError("source validation domains differ across seeds")
    labels_by_source: dict[str, list[int]] = {}
    scores_by_source: dict[str, list[float]] = {}
    for domain in domains:
        labels = payloads[0][domain]["labels"]
        if any(payload[domain]["labels"] != labels for payload in payloads[1:]):
            raise RuntimeError(f"source validation labels differ for {domain}")
        labels_by_source[domain] = labels
        scores_by_source[domain] = np.mean(
            [payload[domain]["scores"] for payload in payloads], axis=0
        ).tolist()
    return select_source_threshold(labels_by_source, scores_by_source)


def _bootstrap(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    replicates: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc_samples = {method: [] for method in METHODS}
    delta_samples = {
        "fedevi_monotonic_minus_fedavg": [],
        "fedevi_monotonic_minus_fedevi_mtl": [],
    }
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        values = {
            method: float(roc_auc_score(sampled_labels, score[indices]))
            for method, score in scores.items()
        }
        for method in METHODS:
            auc_samples[method].append(values[method])
        delta_samples["fedevi_monotonic_minus_fedavg"].append(
            values["fedevi_monotonic"] - values["fedavg"]
        )
        delta_samples["fedevi_monotonic_minus_fedevi_mtl"].append(
            values["fedevi_monotonic"] - values["fedevi_mtl"]
        )

    def summarize(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        lower, upper = np.percentile(array, [2.5, 97.5])
        return {
            "lower": float(lower),
            "upper": float(upper),
            "valid_replicates": int(len(array)),
        }

    auc_ci = {method: summarize(values) for method, values in auc_samples.items()}
    deltas = {}
    for comparison, values in delta_samples.items():
        summary = summarize(values)
        array = np.asarray(values, dtype=float)
        summary["probability_delta_gt_zero"] = float(np.mean(array > 0))
        deltas[comparison] = summary
    return auc_ci, deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    ensemble = None
    ensemble_scores: dict[str, np.ndarray] = {}
    method_reports: dict[str, dict[str, object]] = {}
    seed_rows: list[dict[str, object]] = []
    reference_identity = None
    labels = None

    for method in METHODS:
        runs = [args.result_root / method / f"seed_{seed}" for seed in SEEDS]
        for run in runs:
            _validate_run(run)
        predictions = [pd.read_csv(run / "target_predictions.csv") for run in runs]
        identity_columns = ["manifest_row", "image_path", "label"]
        identities = [frame[identity_columns].reset_index(drop=True) for frame in predictions]
        if any(not identities[0].equals(value) for value in identities[1:]):
            raise RuntimeError(f"target prediction rows differ across {method} seeds")
        if reference_identity is None:
            reference_identity = identities[0]
            labels = reference_identity["label"].to_numpy(dtype=int)
            ensemble = reference_identity.copy()
        elif not reference_identity.equals(identities[0]):
            raise RuntimeError(f"target prediction rows differ for {method}")

        scores = np.stack(
            [frame["score"].to_numpy(dtype=float) for frame in predictions]
        )
        mean_scores = scores.mean(axis=0)
        ensemble_scores[method] = mean_scores
        ensemble[f"score_{method}"] = mean_scores
        threshold = _source_ensemble_threshold(runs)
        metrics = binary_metrics(labels, mean_scores, threshold=threshold)
        metrics.update(clinical_reliability_metrics(labels, mean_scores))
        seed_aucs = []
        seed_dice = []
        for seed, run in zip(SEEDS, runs, strict=True):
            target = json.loads(
                (run / "target_metrics.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (run / "run_metadata.json").read_text(encoding="utf-8")
            )
            reliability = clinical_reliability_metrics(
                target["labels"], target["scores"]
            )
            seed_aucs.append(float(target["auc"]))
            if target.get("lesion_dice") is not None:
                seed_dice.append(float(target["lesion_dice"]))
            seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_round": metadata["best_round"],
                    "source_threshold": metadata["source_threshold"],
                    "auc": target["auc"],
                    "balanced_accuracy": target["balanced_accuracy"],
                    "sensitivity": target["sensitivity"],
                    "specificity": target["specificity"],
                    "macro_f1": target["macro_f1"],
                    "brier": target.get("brier", reliability["brier"]),
                    "ece": target.get("ece", reliability["ece"]),
                    "lesion_dice": target.get("lesion_dice"),
                    "lesion_iou": target.get("lesion_iou"),
                }
            )
        method_reports[method] = {
            "source_only_ensemble_threshold": threshold,
            "ensemble_metrics": {
                key: value
                for key, value in metrics.items()
                if key not in {"labels", "scores", "predictions"}
            },
            "seed_auc_mean": float(np.mean(seed_aucs)),
            "seed_auc_sample_std": float(np.std(seed_aucs, ddof=1)),
            "seed_lesion_dice_mean": (
                float(np.mean(seed_dice)) if seed_dice else None
            ),
        }

    assert labels is not None and ensemble is not None
    auc_ci, deltas = _bootstrap(
        labels, ensemble_scores, args.bootstrap_replicates
    )
    for method in METHODS:
        method_reports[method]["ensemble_auc_bootstrap_95_ci"] = auc_ci[method]
    point_deltas = {
        "fedevi_monotonic_minus_fedavg": float(
            method_reports["fedevi_monotonic"]["ensemble_metrics"]["auc"]
            - method_reports["fedavg"]["ensemble_metrics"]["auc"]
        ),
        "fedevi_monotonic_minus_fedevi_mtl": float(
            method_reports["fedevi_monotonic"]["ensemble_metrics"]["auc"]
            - method_reports["fedevi_mtl"]["ensemble_metrics"]["auc"]
        ),
    }
    for comparison, value in point_deltas.items():
        deltas[comparison]["point_estimate"] = value

    report = {
        "dataset": "QAMEBI_BUID",
        "role": "locked_public_external_validation",
        "analysis_unit": "lesion_record",
        "patient_grouping_reliable": False,
        "n": int(len(labels)),
        "class_counts": {
            "0": int(np.sum(labels == 0)),
            "1": int(np.sum(labels == 1)),
        },
        "primary_prediction": "arithmetic_mean_probability_across_seeds_42_43_44",
        "primary_endpoint": "AUROC",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "methods": method_reports,
        "paired_auc_differences": deltas,
        "not_prospective_clinical_confirmation": True,
    }
    args.output_dir.mkdir(parents=True)
    ensemble.to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
