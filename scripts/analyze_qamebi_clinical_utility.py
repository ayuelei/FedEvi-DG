#!/usr/bin/env python3
"""Write descriptive decision-curve and reliability data for QAMEBI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("fedavg", "fedevi_mtl", "fedevi_monotonic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    frame = pd.read_csv(args.predictions)
    labels = frame["label"].to_numpy(dtype=int)
    n = len(labels)
    prevalence = float(np.mean(labels))
    decision_rows = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        odds = threshold / (1.0 - threshold)
        decision_rows.extend(
            [
                {
                    "method": "treat_none",
                    "threshold_probability": threshold,
                    "net_benefit": 0.0,
                },
                {
                    "method": "treat_all",
                    "threshold_probability": threshold,
                    "net_benefit": prevalence - (1.0 - prevalence) * odds,
                },
            ]
        )
        for method in METHODS:
            scores = frame[f"score_{method}"].to_numpy(dtype=float)
            predictions = scores >= threshold
            true_positive = int(np.sum(predictions & (labels == 1)))
            false_positive = int(np.sum(predictions & (labels == 0)))
            decision_rows.append(
                {
                    "method": method,
                    "threshold_probability": threshold,
                    "net_benefit": true_positive / n - false_positive / n * odds,
                }
            )

    reliability_rows = []
    edges = np.linspace(0.0, 1.0, 11)
    for method in METHODS:
        scores = frame[f"score_{method}"].to_numpy(dtype=float)
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            selected = (scores >= lower) & (
                scores <= upper if index == 9 else scores < upper
            )
            reliability_rows.append(
                {
                    "method": method,
                    "bin_index": index,
                    "lower": lower,
                    "upper": upper,
                    "n": int(np.sum(selected)),
                    "mean_predicted_probability": (
                        float(np.mean(scores[selected])) if np.any(selected) else np.nan
                    ),
                    "observed_malignant_fraction": (
                        float(np.mean(labels[selected])) if np.any(selected) else np.nan
                    ),
                }
            )

    args.output_dir.mkdir(parents=True)
    pd.DataFrame(decision_rows).to_csv(
        args.output_dir / "decision_curve.csv", index=False
    )
    pd.DataFrame(reliability_rows).to_csv(
        args.output_dir / "reliability.csv", index=False
    )


if __name__ == "__main__":
    main()
