#!/usr/bin/env python3
"""Aggregate three-seed FedEvi QAMEBI explanation interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SEEDS = (42, 43, 44)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explanation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    frames = []
    seed_metrics = {}
    identity = None
    for seed in SEEDS:
        root = args.explanation_root / f"seed_{seed}"
        frame = pd.read_csv(root / "predictions.csv")
        current_identity = frame[["manifest_row", "image_path", "label"]]
        if identity is None:
            identity = current_identity
        elif not identity.equals(current_identity):
            raise RuntimeError("explanation rows differ across seeds")
        frames.append(frame)
        seed_metrics[str(seed)] = json.loads(
            (root / "metrics.json").read_text(encoding="utf-8")
        )

    labels = frames[0]["label"].to_numpy(dtype=int)
    columns = (
        "original_score",
        "lesion_only_score",
        "lesion_erased_score",
        "ordinal_clamp_low_score",
        "ordinal_clamp_high_score",
        "ordinal_clamp_low_log_odds",
        "ordinal_clamp_high_log_odds",
    )
    ensemble = {
        column: np.mean(
            [frame[column].to_numpy(dtype=float) for frame in frames], axis=0
        )
        for column in columns
    }
    original_auc = float(roc_auc_score(labels, ensemble["original_score"]))
    lesion_only_auc = float(roc_auc_score(labels, ensemble["lesion_only_score"]))
    lesion_erased_auc = float(
        roc_auc_score(labels, ensemble["lesion_erased_score"])
    )
    high = ensemble["ordinal_clamp_high_score"]
    low = ensemble["ordinal_clamp_low_score"]
    high_log_odds = ensemble["ordinal_clamp_high_log_odds"]
    low_log_odds = ensemble["ordinal_clamp_low_log_odds"]
    malignant = labels == 1
    report = {
        "dataset": "QAMEBI_BUID",
        "analysis_unit": "lesion_record",
        "n": int(len(labels)),
        "seeds": list(SEEDS),
        "ensemble_original_auc": original_auc,
        "ensemble_lesion_only_auc": lesion_only_auc,
        "ensemble_lesion_erased_auc": lesion_erased_auc,
        "ensemble_lesion_sufficiency_auc_gap": original_auc - lesion_only_auc,
        "ensemble_lesion_necessity_auc_drop": original_auc - lesion_erased_auc,
        "ensemble_malignant_score_drop_after_lesion_erasure": float(
            np.mean(
                ensemble["original_score"][malignant]
                - ensemble["lesion_erased_score"][malignant]
            )
        ),
        "ensemble_ordinal_high_minus_low_mean": float(np.mean(high - low)),
        "ensemble_ordinal_non_decreasing_fraction": float(np.mean(high >= low)),
        "ensemble_ordinal_negative_count": int(np.sum(high < low)),
        "ensemble_ordinal_log_odds_high_minus_low_mean": float(
            np.mean(high_log_odds - low_log_odds)
        ),
        "seed_mean_lesion_dice": float(
            np.mean([value["lesion_dice"] for value in seed_metrics.values()])
        ),
        "seed_mean_lesion_iou": float(
            np.mean([value["lesion_iou"] for value in seed_metrics.values()])
        ),
        "ordinal_birads_validation_available": False,
        "ordinal_birads_validation_reason": "QAMEBI provides no BI-RADS grades",
        "per_seed": seed_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
