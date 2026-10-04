#!/usr/bin/env python3
"""Portable classification analysis; uses the original statistical functions."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_fedevi_final as analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260926)
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Require source-fitted calibration for main and FedAvg.",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.bootstrap_iterations < 200:
        raise ValueError("bootstrap-iterations must be at least 200")
    os.chdir(ROOT)
    # Fresh runs all use one release implementation; historical replacement
    # folders must not silently replace any part of a new campaign.
    analysis.SEED42_RERUN_ROOTS = {}
    per_seed, ensemble, units = analysis._metric_rows()
    target, macro = analysis._bootstrap_rows(
        units, args.bootstrap_iterations, args.random_seed
    )
    patient = analysis._patient_aggregation_sensitivity_rows(
        args.bootstrap_iterations, args.random_seed
    )
    results = {
        "per_seed_auc": per_seed,
        "ensemble_metrics": ensemble,
        "paired_bootstrap_by_target": target,
        "paired_bootstrap_macro": macro,
        "patient_aggregation_sensitivity": patient,
        "clinical_operating_points": ensemble[
            [
                "method",
                "target",
                "evaluation_level",
                "n",
                "sensitivity_at_specificity_90",
            ]
        ],
    }
    if args.calibration:
        calibration, dca = analysis._calibration_and_dca()
        results.update(
            calibration=calibration,
            decision_curve=dca,
            decision_curve_summary=analysis._dca_summary(dca),
        )
    output.mkdir(parents=True)
    for name, frame in results.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    print(f"Classification outputs: {output}")
    print(
        "Ordinal/spatial explanation analyses are separate; no missing endpoint is imputed."
    )


if __name__ == "__main__":
    main()
