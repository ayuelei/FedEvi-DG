#!/usr/bin/env python3
"""Evaluate a frozen target at patient level when grouping is reliable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def _auc_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = generator.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size == 2:
            values.append(roc_auc_score(sampled_labels, scores[indices]))
    if not values:
        raise RuntimeError("all bootstrap samples contained one class")
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    frame = pd.read_csv(args.run_dir / "target_predictions.csv")
    reliable = frame["patient_grouping_reliable"].map(
        lambda item: str(item).strip().casefold() == "true"
    )
    if not reliable.all():
        raise ValueError(
            "patient-level claims are forbidden because grouping is not reliable"
        )
    grouped = frame.groupby("patient_id", sort=True, observed=True)
    if int(grouped["label"].nunique().max()) != 1:
        raise ValueError("a patient has conflicting malignancy labels")
    labels = grouped["label"].first().to_numpy(dtype=int)
    mean_scores = grouped["score"].mean().to_numpy(dtype=float)
    max_scores = grouped["score"].max().to_numpy(dtype=float)
    report = {
        "aggregation_primary": "mean_probability_within_patient",
        "aggregation_sensitivity": "maximum_probability_within_patient",
        "image_count": len(frame),
        "patient_count": len(labels),
        "primary_auc": float(roc_auc_score(labels, mean_scores)),
        "primary_auc_bootstrap_95ci": _auc_interval(
            labels, mean_scores, seed=args.seed, replicates=args.bootstrap
        ),
        "primary_brier": float(brier_score_loss(labels, mean_scores)),
        "max_auc": float(roc_auc_score(labels, max_scores)),
        "max_auc_bootstrap_95ci": _auc_interval(
            labels, max_scores, seed=args.seed + 1, replicates=args.bootstrap
        ),
        "bootstrap_unit": "patient",
        "target_labels_used_for_model_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
