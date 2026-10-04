#!/usr/bin/env python3
"""Fit temperature on source validation clients, then evaluate a frozen target.

The held-out target labels never participate in fitting. Source domains receive
equal weight so that a large hospital cannot dominate the calibration map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


EPSILON = 1e-7


def _logits(scores: np.ndarray) -> np.ndarray:
    bounded = np.clip(scores.astype(float), EPSILON, 1.0 - EPSILON)
    return np.log(bounded) - np.log1p(-bounded)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1]
            if index == bins - 1
            else scores < edges[index + 1]
        )
        if selected.any():
            value += selected.mean() * abs(
                scores[selected].mean() - labels[selected].mean()
            )
    return float(value)


def _domain_nll(
    log_temperature: float,
    domains: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    temperature = float(np.exp(log_temperature))
    losses = []
    for labels, logits in domains:
        probabilities = _sigmoid(logits / temperature)
        losses.append(log_loss(labels, probabilities, labels=[0, 1]))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    source = json.loads(
        (args.run_dir / "source_val_metrics.json").read_text(encoding="utf-8")
    )
    target = json.loads(
        (args.run_dir / "target_metrics.json").read_text(encoding="utf-8")
    )
    domains: list[tuple[np.ndarray, np.ndarray]] = []
    source_counts: dict[str, int] = {}
    for name, metrics in sorted(source.items()):
        labels = np.asarray(metrics["labels"], dtype=int)
        logits = _logits(np.asarray(metrics["scores"], dtype=float))
        if len(labels) == 0 or np.unique(labels).size != 2:
            raise ValueError(f"source validation domain {name} is not binary")
        domains.append((labels, logits))
        source_counts[name] = len(labels)
    fit = minimize_scalar(
        lambda value: _domain_nll(value, domains),
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not fit.success:
        raise RuntimeError(f"temperature optimization failed: {fit.message}")
    temperature = float(np.exp(fit.x))

    labels = np.asarray(target["labels"], dtype=int)
    raw_scores = np.asarray(target["scores"], dtype=float)
    calibrated = _sigmoid(_logits(raw_scores) / temperature)
    old_threshold = float(
        json.loads(
            (args.run_dir / "run_metadata.json").read_text(encoding="utf-8")
        )["source_threshold"]
    )
    calibrated_threshold = float(
        _sigmoid(np.asarray([_logits(np.asarray([old_threshold]))[0] / temperature]))[0]
    )
    report = {
        "fit_scope": "source_validation_only",
        "source_domain_weighting": "equal_domain_nll",
        "source_counts": source_counts,
        "temperature": temperature,
        "source_equal_domain_nll": float(fit.fun),
        "target_n": len(labels),
        "target_auc_raw": float(roc_auc_score(labels, raw_scores)),
        "target_auc_calibrated": float(roc_auc_score(labels, calibrated)),
        "target_brier_raw": float(brier_score_loss(labels, raw_scores)),
        "target_brier_calibrated": float(
            brier_score_loss(labels, calibrated)
        ),
        "target_ece_raw": _ece(labels, raw_scores),
        "target_ece_calibrated": _ece(labels, calibrated),
        "raw_source_threshold": old_threshold,
        "calibrated_equivalent_threshold": calibrated_threshold,
        "target_labels_used_for_fitting": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "target_scores.npz",
        labels=labels,
        raw_scores=raw_scores,
        calibrated_scores=calibrated,
    )


if __name__ == "__main__":
    main()
