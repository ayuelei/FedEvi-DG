#!/usr/bin/env python3
"""Create the final multi-seed FedEvi-DG statistical analysis artifacts.

The script never fits on target labels.  Reliable patient identifiers are
aggregated before metrics; otherwise the image/record remains the evaluation
unit.  AUROC differences use paired within-domain bootstrap resampling and
the five-domain macro interval averages one paired draw from every domain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    brier_score_loss,
    cohen_kappa_score,
    roc_auc_score,
)


TARGETS = ("BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT")
SEEDS = (42, 43, 44)
MASK_TARGETS = frozenset({"BUSBRA", "BUSI_WHU", "UDIAT"})
CANDIDATE = "fedevi_monotonic"
METHOD_ROOTS = {
    CANDIDATE: Path("outputs/fedevi_monotonic_gate"),
    "fedavg": Path("outputs/fedusdg_mvp_phase3"),
    "fedevi_mtl": Path("outputs/fedevi_controls_gate"),
    "fedevi_no_taskagg": Path("outputs/fedevi_controls_gate"),
    "fedevi_shuffle": Path("outputs/fedevi_controls_gate"),
    "fedevi_mask_shift": Path("outputs/fedevi_controls_gate"),
    "fedevi_dg": Path("outputs/fedevi_dg_gate"),
    "fedevi_seg_only": Path("outputs/fedevi_single_aux_controls"),
    "fedavg_ord": Path("outputs/fedevi_single_aux_controls"),
}
COMPARATORS = tuple(method for method in METHOD_ROOTS if method != CANDIDATE)
SEED42_RERUN_ROOTS = {
    CANDIDATE: Path("outputs/fedevi_monotonic_snapshot344_seed42_rerun"),
    "fedevi_shuffle": Path("outputs/fedevi_current_impl_reruns"),
    "fedevi_mask_shift": Path("outputs/fedevi_current_impl_reruns"),
    "fedevi_dg": Path("outputs/fedevi_current_impl_reruns"),
}


def _run_dir(method: str, target: str, seed: int) -> Path:
    # Seed-42 reruns remove implementation-snapshot mixing. Candidate reruns
    # use the recovered 344b61c9 snapshot shared by seeds 43/44; matched-control
    # reruns use their already-declared current implementation snapshot.
    rerun_root = SEED42_RERUN_ROOTS.get(method)
    rerun = (
        rerun_root
        / f"target_{target}"
        / method
        / f"seed_{seed}"
        if rerun_root is not None
        else Path("__no_rerun__")
    )
    if seed == 42 and (rerun / "COMPLETE").is_file():
        return rerun
    return (
        METHOD_ROOTS[method]
        / f"target_{target}"
        / method
        / f"seed_{seed}"
    )


def _truth(value: object) -> bool:
    return str(value).strip().casefold() == "true"


def _load_predictions(method: str, target: str, seed: int) -> pd.DataFrame:
    run = _run_dir(method, target, seed)
    if not (run / "COMPLETE").is_file():
        raise FileNotFoundError(f"incomplete run: {run}")
    frame = pd.read_csv(run / "target_predictions.csv")
    required = {
        "manifest_row",
        "image_path",
        "patient_id",
        "patient_grouping_reliable",
        "label",
        "score",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{run} lacks columns: {sorted(missing)}")
    return frame.sort_values("manifest_row").reset_index(drop=True)


def _ensemble_records(
    method: str,
    target: str,
    *,
    calibrated: bool = False,
) -> pd.DataFrame:
    frames = [_load_predictions(method, target, seed) for seed in SEEDS]
    reference = frames[0]
    for frame in frames[1:]:
        for column in ("manifest_row", "image_path", "patient_id", "label"):
            if not reference[column].equals(frame[column]):
                raise ValueError(f"unaligned {method}/{target}: {column}")
    scores = []
    for seed, frame in zip(SEEDS, frames, strict=True):
        value = frame["score"].to_numpy(dtype=float)
        if calibrated:
            calibration = _calibration_path(method, target, seed)
            metadata = json.loads(calibration.read_text(encoding="utf-8"))
            value = _temperature_scale(value, float(metadata["temperature"]))
        scores.append(value)
    result = reference[
        [
            "manifest_row",
            "image_path",
            "patient_id",
            "patient_grouping_reliable",
            "label",
        ]
    ].copy()
    result["score"] = np.mean(np.stack(scores), axis=0)
    return result


def _ensemble_records_without_rerun(method: str, target: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        run = (
            METHOD_ROOTS[method]
            / f"target_{target}"
            / method
            / f"seed_{seed}"
        )
        if not (run / "COMPLETE").is_file():
            raise FileNotFoundError(f"incomplete original run: {run}")
        frames.append(
            pd.read_csv(run / "target_predictions.csv")
            .sort_values("manifest_row")
            .reset_index(drop=True)
        )
    reference = frames[0]
    for frame in frames[1:]:
        for column in ("manifest_row", "image_path", "patient_id", "label"):
            if not reference[column].equals(frame[column]):
                raise ValueError(f"unaligned original {method}/{target}: {column}")
    result = reference[
        [
            "manifest_row",
            "image_path",
            "patient_id",
            "patient_grouping_reliable",
            "label",
        ]
    ].copy()
    result["score"] = np.mean(
        np.stack([frame["score"].to_numpy(dtype=float) for frame in frames]), axis=0
    )
    return result


def _evaluation_units(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    reliable = frame["patient_grouping_reliable"].map(_truth)
    if reliable.all():
        grouped = frame.groupby("patient_id", sort=True, observed=True)
        if int(grouped["label"].nunique().max()) != 1:
            raise ValueError("conflicting labels within reliable patient ID")
        return (
            grouped["label"].first().to_numpy(dtype=int),
            grouped["score"].mean().to_numpy(dtype=float),
            "patient_mean",
        )
    if reliable.any():
        raise ValueError("mixed reliable and unreliable patient grouping")
    return (
        frame["label"].to_numpy(dtype=int),
        frame["score"].to_numpy(dtype=float),
        "image_or_record",
    )


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
                float(scores[selected].mean()) - float(labels[selected].mean())
            )
    return float(value)


def _sensitivity_at_specificity(
    labels: np.ndarray, scores: np.ndarray, specificity: float = 0.90
) -> float:
    """Maximum descriptive sensitivity while meeting the specificity floor."""
    if not 0.0 < specificity < 1.0:
        raise ValueError("specificity must lie strictly between zero and one")
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if len(negatives) == 0 or len(positives) == 0:
        raise ValueError("both classes are required")
    allowed_false_positives = int(
        np.floor((1.0 - specificity) * len(negatives) + 1e-12)
    )
    thresholds = np.unique(
        np.concatenate(([float(np.nextafter(scores.max(), np.inf))], scores))
    )
    feasible = [
        float(np.mean(positives >= threshold))
        for threshold in thresholds
        if int(np.count_nonzero(negatives >= threshold)) <= allowed_false_positives
    ]
    return max(feasible, default=0.0)


def _paired_bootstrap_auc(
    labels: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not (len(labels) == len(candidate) == len(comparator)):
        raise ValueError("paired bootstrap arrays must align")
    values: list[float] = []
    while len(values) < iterations:
        selected = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[selected]
        if np.unique(sampled_labels).size < 2:
            continue
        values.append(
            float(roc_auc_score(sampled_labels, candidate[selected]))
            - float(roc_auc_score(sampled_labels, comparator[selected]))
        )
    return np.asarray(values)


def _calibration_path(method: str, target: str, seed: int) -> Path:
    roots = {
        CANDIDATE: Path("outputs/fedevi_monotonic_gate_calibration"),
        "fedavg": Path("outputs/fedavg_gate_calibration"),
    }
    return (
        roots[method]
        / f"target_{target}"
        / method
        / f"seed_{seed}"
        / "calibration.json"
    )


def _temperature_scale(scores: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def _net_benefit(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    return float(
        true_positive / len(labels)
        - false_positive / len(labels) * threshold / (1.0 - threshold)
    )


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _metric_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    per_seed = []
    ensemble = []
    units: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for method in METHOD_ROOTS:
        for target in TARGETS:
            for seed in SEEDS:
                labels, scores, level = _evaluation_units(
                    _load_predictions(method, target, seed)
                )
                per_seed.append(
                    {
                        "method": method,
                        "target": target,
                        "seed": seed,
                        "evaluation_level": level,
                        "n": len(labels),
                        "auc": roc_auc_score(labels, scores),
                    }
                )
            labels, scores, level = _evaluation_units(
                _ensemble_records(method, target)
            )
            units[(method, target)] = (labels, scores)
            ensemble.append(
                {
                    "method": method,
                    "target": target,
                    "evaluation_level": level,
                    "n": len(labels),
                    "auc": roc_auc_score(labels, scores),
                    "brier": brier_score_loss(labels, scores),
                    "ece": _ece(labels, scores),
                    "sensitivity_at_specificity_90": _sensitivity_at_specificity(
                        labels, scores
                    ),
                }
            )
    return pd.DataFrame(per_seed), pd.DataFrame(ensemble), units


def _patient_aggregation_sensitivity_rows(
    iterations: int, random_seed: int
) -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(METHOD_ROOTS):
        for target_index, target in enumerate(TARGETS):
            frame = _ensemble_records(method, target)
            reliable = frame["patient_grouping_reliable"].map(_truth)
            if not reliable.all():
                if reliable.any():
                    raise ValueError(
                        f"mixed reliable patient grouping: {method}/{target}"
                    )
                continue
            grouped = frame.groupby("patient_id", sort=True, observed=True)
            if int(grouped["label"].nunique().max()) != 1:
                raise ValueError(
                    f"conflicting labels within reliable patient ID: {method}/{target}"
                )
            labels = grouped["label"].first().to_numpy(dtype=int)
            mean_scores = grouped["score"].mean().to_numpy(dtype=float)
            max_scores = grouped["score"].max().to_numpy(dtype=float)
            mean_auc = float(roc_auc_score(labels, mean_scores))
            max_auc = float(roc_auc_score(labels, max_scores))
            boot = _paired_bootstrap_auc(
                labels,
                max_scores,
                mean_scores,
                iterations,
                np.random.default_rng(
                    random_seed + 20000 + 1000 * method_index + target_index
                ),
            )
            lower, upper = _mean_ci(boot)
            rows.append(
                {
                    "method": method,
                    "target": target,
                    "n_patients": len(labels),
                    "mean_probability_auc": mean_auc,
                    "maximum_probability_auc": max_auc,
                    "max_minus_mean_auc_delta": max_auc - mean_auc,
                    "delta_ci95_lower": lower,
                    "delta_ci95_upper": upper,
                    "bootstrap_unit": "patient",
                }
            )
    return pd.DataFrame(rows)


def _rerun_provenance_rows() -> pd.DataFrame:
    rows = []
    for method in (
        CANDIDATE,
        "fedevi_shuffle",
        "fedevi_mask_shift",
        "fedevi_dg",
    ):
        for target in TARGETS:
            rerun_root = SEED42_RERUN_ROOTS[method]
            rerun = (
                rerun_root
                / f"target_{target}"
                / method
                / "seed_42"
            )
            if not (rerun / "COMPLETE").is_file():
                continue
            original = (
                METHOD_ROOTS[method]
                / f"target_{target}"
                / method
                / "seed_42"
            )
            old_frame = pd.read_csv(original / "target_predictions.csv").sort_values(
                "manifest_row"
            )
            new_frame = pd.read_csv(rerun / "target_predictions.csv").sort_values(
                "manifest_row"
            )
            old_labels, old_scores, level = _evaluation_units(old_frame)
            new_labels, new_scores, new_level = _evaluation_units(new_frame)
            if level != new_level or not np.array_equal(old_labels, new_labels):
                raise ValueError(f"rerun alignment failure: {method}/{target}")
            old_config = json.loads(
                (original / "resolved_config.json").read_text(encoding="utf-8")
            )
            new_config = json.loads(
                (rerun / "resolved_config.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "method": method,
                    "target": target,
                    "seed": 42,
                    "evaluation_level": level,
                    "original_implementation_sha256": old_config[
                        "implementation_sha256"
                    ],
                    "replacement_implementation_sha256": new_config[
                        "implementation_sha256"
                    ],
                    "predictions_byte_identical": (
                        (original / "target_predictions.csv").read_bytes()
                        == (rerun / "target_predictions.csv").read_bytes()
                    ),
                    "maximum_absolute_score_difference": float(
                        np.max(np.abs(old_scores - new_scores))
                    ),
                    "original_auc": float(roc_auc_score(old_labels, old_scores)),
                    "replacement_auc": float(
                        roc_auc_score(new_labels, new_scores)
                    ),
                    "replacement_minus_original_auc": float(
                        roc_auc_score(new_labels, new_scores)
                        - roc_auc_score(old_labels, old_scores)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _r0_replacement_sensitivity(
    iterations: int, random_seed: int
) -> pd.DataFrame:
    variants = {
        "original_seed42": lambda target: _ensemble_records_without_rerun(
            "fedevi_dg", target
        ),
        "snapshot_consistent_replacement": lambda target: _ensemble_records(
            "fedevi_dg", target
        ),
    }
    rows = []
    for variant_index, (variant, loader) in enumerate(variants.items()):
        target_deltas = []
        target_draws = []
        positive = 0
        for target_index, target in enumerate(TARGETS):
            labels, candidate_scores, _ = _evaluation_units(
                _ensemble_records(CANDIDATE, target)
            )
            other_labels, comparator_scores, _ = _evaluation_units(loader(target))
            if not np.array_equal(labels, other_labels):
                raise ValueError(f"R0 sensitivity label mismatch: {variant}/{target}")
            delta = float(roc_auc_score(labels, candidate_scores)) - float(
                roc_auc_score(labels, comparator_scores)
            )
            target_deltas.append(delta)
            positive += int(delta > 0)
            target_draws.append(
                _paired_bootstrap_auc(
                    labels,
                    candidate_scores,
                    comparator_scores,
                    iterations,
                    np.random.default_rng(
                        random_seed
                        + 30000
                        + 1000 * variant_index
                        + target_index
                    ),
                )
            )
        macro_draws = np.mean(np.stack(target_draws), axis=0)
        lower, upper = _mean_ci(macro_draws)
        point = float(np.mean(target_deltas))
        rows.append(
            {
                "variant": variant,
                "candidate_minus_r0_macro_auc": point,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "positive_target_domains": positive,
                "directional_gate": bool(point >= 0 and positive >= 3),
                "ci_excludes_zero": bool(lower > 0 or upper < 0),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_rows(
    units: dict,
    iterations: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = []
    macro_rows = []
    for comparator_index, comparator in enumerate(COMPARATORS):
        draws = []
        positive_targets = 0
        for target_index, target in enumerate(TARGETS):
            labels, candidate_scores = units[(CANDIDATE, target)]
            other_labels, comparator_scores = units[(comparator, target)]
            if not np.array_equal(labels, other_labels):
                raise ValueError(f"label mismatch: {comparator}/{target}")
            point = float(roc_auc_score(labels, candidate_scores)) - float(
                roc_auc_score(labels, comparator_scores)
            )
            positive_targets += int(point > 0)
            rng = np.random.default_rng(
                random_seed + 1000 * comparator_index + target_index
            )
            boot = _paired_bootstrap_auc(
                labels, candidate_scores, comparator_scores, iterations, rng
            )
            draws.append(boot)
            lower, upper = _mean_ci(boot)
            target_rows.append(
                {
                    "comparator": comparator,
                    "target": target,
                    "auc_delta": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_probability_positive": float(np.mean(boot > 0)),
                }
            )
        macro_draws = np.mean(np.stack(draws), axis=0)
        macro_point = float(
            np.mean(
                [
                    row["auc_delta"]
                    for row in target_rows
                    if row["comparator"] == comparator
                ]
            )
        )
        lower, upper = _mean_ci(macro_draws)
        macro_rows.append(
            {
                "comparator": comparator,
                "macro_auc_delta": macro_point,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "positive_target_domains": positive_targets,
                "bootstrap_probability_positive": float(
                    np.mean(macro_draws > 0)
                ),
                "directional_gate": bool(
                    macro_point >= 0 and positive_targets >= 3
                ),
                "ci_excludes_zero": bool(lower > 0 or upper < 0),
            }
        )
    return pd.DataFrame(target_rows), pd.DataFrame(macro_rows)


def _calibration_and_dca() -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_rows = []
    dca_rows = []
    thresholds = np.arange(0.01, 0.51, 0.01)
    for target in TARGETS:
        calibrated_units = {}
        for method in (CANDIDATE, "fedavg"):
            raw_labels, raw_scores, level = _evaluation_units(
                _ensemble_records(method, target)
            )
            labels, scores, calibrated_level = _evaluation_units(
                _ensemble_records(method, target, calibrated=True)
            )
            if level != calibrated_level or not np.array_equal(raw_labels, labels):
                raise ValueError(f"calibration alignment failure: {method}/{target}")
            calibrated_units[method] = (labels, scores)
            calibration_rows.append(
                {
                    "method": method,
                    "target": target,
                    "evaluation_level": level,
                    "n": len(labels),
                    "raw_brier": brier_score_loss(labels, raw_scores),
                    "calibrated_brier": brier_score_loss(labels, scores),
                    "raw_ece": _ece(labels, raw_scores),
                    "calibrated_ece": _ece(labels, scores),
                }
            )
        labels = calibrated_units[CANDIDATE][0]
        prevalence = float(labels.mean())
        for threshold in thresholds:
            row = {
                "target": target,
                "threshold": threshold,
                "evaluation_level": level,
                "n": len(labels),
                "treat_none": 0.0,
                "treat_all": prevalence
                - (1.0 - prevalence) * threshold / (1.0 - threshold),
            }
            for method in (CANDIDATE, "fedavg"):
                method_labels, scores = calibrated_units[method]
                if not np.array_equal(labels, method_labels):
                    raise ValueError(f"DCA label mismatch: {target}")
                row[method] = _net_benefit(labels, scores, float(threshold))
            dca_rows.append(row)
    return pd.DataFrame(calibration_rows), pd.DataFrame(dca_rows)


def _dca_summary(dca: pd.DataFrame) -> pd.DataFrame:
    """Summarize a fixed clinically relevant probability range.

    The 0.05--0.30 range is declared here rather than selected per target;
    the complete 0.01--0.50 curves remain available in decision_curve.csv.
    """
    selected = dca[(dca["threshold"] >= 0.05) & (dca["threshold"] <= 0.30)]
    rows = []
    for target, frame in selected.groupby("target", sort=True, observed=True):
        delta = frame[CANDIDATE] - frame["fedavg"]
        best_default = frame[["treat_none", "treat_all"]].max(axis=1)
        rows.append(
            {
                "target": target,
                "threshold_min": 0.05,
                "threshold_max": 0.30,
                "threshold_count": len(frame),
                "candidate_mean_net_benefit": frame[CANDIDATE].mean(),
                "fedavg_mean_net_benefit": frame["fedavg"].mean(),
                "candidate_minus_fedavg_mean": delta.mean(),
                "candidate_better_than_fedavg_thresholds": int((delta > 0).sum()),
                "candidate_better_than_best_default_thresholds": int(
                    (frame[CANDIDATE] > best_default).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _ordinal_dataset(
    paths: Iterable[Path],
    observed_levels: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frames = [pd.read_csv(path).sort_values("manifest_row") for path in paths]
    reference = frames[0].reset_index(drop=True)
    for frame in frames[1:]:
        frame = frame.reset_index(drop=True)
        for column in ("manifest_row", "patient_id", "birads_ordinal"):
            left = reference[column].to_numpy()
            right = frame[column].to_numpy()
            aligned = (
                np.array_equal(left.astype(str), right.astype(str))
                if column == "patient_id"
                else np.allclose(left, right, equal_nan=True)
            )
            if not aligned:
                raise ValueError(f"ordinal prediction mismatch: {column}")
    predicted_risk = np.mean(
        np.stack([frame["ordinal_risk"].to_numpy(dtype=float) for frame in frames]),
        axis=0,
    )
    observed = reference["birads_ordinal"].to_numpy(dtype=float)
    finite = np.isfinite(observed)
    reference = reference.loc[finite].reset_index(drop=True)
    observed = observed[finite]
    predicted_risk = predicted_risk[finite]
    normalized_levels = np.linspace(0.0, 1.0, len(observed_levels))
    true_class = np.abs(
        observed[:, None] - observed_levels[None, :]
    ).argmin(axis=1)
    predicted_class = np.abs(
        predicted_risk[:, None] - normalized_levels[None, :]
    ).argmin(axis=1)
    return reference, true_class, predicted_class, predicted_risk


def _cluster_ordinal_bootstrap(
    frame: pd.DataFrame,
    true_class: np.ndarray,
    predicted_class: np.ndarray,
    predicted_risk: np.ndarray,
    levels: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    groups = {
        patient: np.flatnonzero(frame["patient_id"].to_numpy() == patient)
        for patient in sorted(frame["patient_id"].astype(str).unique())
    }
    patients = np.asarray(list(groups), dtype=object)
    values = {"qwk": [], "spearman": [], "mae": []}
    while len(values["qwk"]) < iterations:
        selected_patients = rng.choice(patients, size=len(patients), replace=True)
        selected = np.concatenate([groups[patient] for patient in selected_patients])
        if np.unique(true_class[selected]).size < 2:
            continue
        values["qwk"].append(
            cohen_kappa_score(
                true_class[selected], predicted_class[selected], weights="quadratic"
            )
        )
        values["spearman"].append(
            spearmanr(levels[true_class[selected]], predicted_risk[selected]).statistic
        )
        values["mae"].append(
            np.mean(np.abs(levels[true_class[selected]] - predicted_risk[selected]))
        )
    return {key: _mean_ci(np.asarray(value)) for key, value in values.items()}


def _ordinal_rows(
    iterations: int, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "BUSBRA": (
            [
                Path("outputs/fedevi_monotonic_gate_explanations_v2")
                / "target_BUSBRA"
                / f"seed_{seed}"
                / "predictions.csv"
                for seed in SEEDS
            ],
            np.linspace(1.0, 4.0, 4),
        ),
        "BUS-CoT": (
            [
                Path("outputs/fedevi_monotonic_gate_buscot_reasoning_v3")
                / f"seed_{seed}"
                / "predictions.csv"
                for seed in SEEDS
            ],
            np.linspace(0.0, 1.0, 6),
        ),
    }
    summary_rows = []
    calibration_rows = []
    for index, (dataset, (paths, raw_levels)) in enumerate(specs.items()):
        normalized_levels = np.linspace(0.0, 1.0, len(raw_levels))
        frame, true_class, predicted_class, predicted_risk = _ordinal_dataset(
            paths, raw_levels
        )
        observed_risk = normalized_levels[true_class]
        qwk = cohen_kappa_score(true_class, predicted_class, weights="quadratic")
        rho = spearmanr(observed_risk, predicted_risk).statistic
        mae = float(np.mean(np.abs(observed_risk - predicted_risk)))
        ci = _cluster_ordinal_bootstrap(
            frame,
            true_class,
            predicted_class,
            predicted_risk,
            normalized_levels,
            iterations,
            np.random.default_rng(random_seed + 10000 + index),
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "n_records": len(frame),
                "n_patients": frame["patient_id"].nunique(),
                "qwk": qwk,
                "qwk_ci95_lower": ci["qwk"][0],
                "qwk_ci95_upper": ci["qwk"][1],
                "spearman": rho,
                "spearman_ci95_lower": ci["spearman"][0],
                "spearman_ci95_upper": ci["spearman"][1],
                "ordinal_mae": mae,
                "ordinal_mae_ci95_lower": ci["mae"][0],
                "ordinal_mae_ci95_upper": ci["mae"][1],
                "bootstrap_unit": "patient_cluster",
            }
        )
        for class_index, level in enumerate(normalized_levels):
            selected = true_class == class_index
            calibration_rows.append(
                {
                    "dataset": dataset,
                    "ordinal_class": class_index,
                    "normalized_observed_level": level,
                    "n": int(selected.sum()),
                    "mean_predicted_risk": float(predicted_risk[selected].mean()),
                    "sd_predicted_risk": float(
                        predicted_risk[selected].std(ddof=1)
                        if selected.sum() > 1
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(calibration_rows)


def _spatial_gate() -> dict:
    values = {}
    for target in MASK_TARGETS:
        candidate = []
        shifted = []
        for seed in SEEDS:
            for method, destination in (
                (CANDIDATE, candidate),
                ("fedevi_mask_shift", shifted),
            ):
                metadata = json.loads(
                    (_run_dir(method, target, seed) / "target_metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
                destination.append(float(metadata["lesion_dice"]))
        values[target] = {
            "candidate_mean_dice": float(np.mean(candidate)),
            "shifted_mask_mean_dice": float(np.mean(shifted)),
            "delta": float(np.mean(candidate) - np.mean(shifted)),
        }
    values["all_mask_targets_positive"] = all(
        value["delta"] > 0
        for key, value in values.items()
        if key != "all_mask_targets_positive"
    )
    return values


def _write_report(
    output: Path,
    per_seed: pd.DataFrame,
    ensemble: pd.DataFrame,
    macro_bootstrap: pd.DataFrame,
    calibration: pd.DataFrame,
    dca_summary: pd.DataFrame,
    patient_sensitivity: pd.DataFrame,
    rerun_provenance: pd.DataFrame,
    r0_sensitivity: pd.DataFrame,
    ordinal: pd.DataFrame,
    spatial: dict,
) -> None:
    seed_macros = (
        per_seed.groupby(["method", "seed"], observed=True)["auc"]
        .mean()
        .reset_index()
    )
    seed_summary = (
        seed_macros.groupby("method", observed=True)["auc"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    ensemble_summary = (
        ensemble.groupby("method", observed=True)
        .agg(
            macro_auc=("auc", "mean"),
            worst_target_auc=("auc", "min"),
            macro_sensitivity_at_specificity_90=(
                "sensitivity_at_specificity_90",
                "mean",
            ),
            worst_sensitivity_at_specificity_90=(
                "sensitivity_at_specificity_90",
                "min",
            ),
        )
        .reset_index()
        .sort_values("macro_auc", ascending=False)
    )
    cal_summary = (
        calibration.groupby("method", observed=True)[
            ["raw_brier", "calibrated_brier", "raw_ece", "calibrated_ece"]
        ]
        .mean()
        .reset_index()
    )
    lines = [
        "# FedEvi-DG final non-clinical-external analysis",
        "",
        "## Three-seed five-domain performance",
        "",
        seed_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Three-seed ensemble performance",
        "",
        ensemble_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired hierarchical bootstrap: candidate minus comparator",
        "",
        macro_bootstrap.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Bootstrap resamples evaluation units within each target domain; "
        "the macro interval averages one paired draw from every domain. "
        "Intervals are effect-size intervals, not multiplicity-adjusted tests.",
        "",
        "## Source-validation-only temperature calibration",
        "",
        cal_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision-curve summary (fixed thresholds 0.05--0.30)",
        "",
        dca_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Patient aggregation sensitivity analysis",
        "",
        patient_sensitivity.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Implementation-rerun provenance",
        "",
        rerun_provenance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The replacement rule was implementation-snapshot consistency, not "
        "observed performance. Shuffle and shifted-mask reruns were byte-identical; "
        "R0 reruns were not, so both R0 analyses are retained below.",
        "",
        r0_sensitivity.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Ordinal evidence",
        "",
        ordinal.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Spatial corruption gate",
        "",
        f"- All three mask targets positive: {spatial['all_mask_targets_positive']}",
    ]
    for target in sorted(MASK_TARGETS):
        value = spatial[target]
        lines.append(
            f"- {target}: Dice delta candidate - shifted mask "
            f"{value['delta']:+.4f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Five-domain results are internal-external cross-validation, not "
            "a random five-fold split.",
            "- Calibration temperatures were fitted only on source validation "
            "domains; target labels were used only for final evaluation.",
            "- QAMEBI remains the locked public external validation. A new "
            "patient-level clinical confirmation cohort is still pending.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/fedevi_final_analysis")
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260926)
    args = parser.parse_args()
    if args.bootstrap_iterations < 200:
        raise ValueError("bootstrap-iterations must be at least 200")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    per_seed, ensemble, units = _metric_rows()
    target_bootstrap, macro_bootstrap = _bootstrap_rows(
        units, args.bootstrap_iterations, args.random_seed
    )
    calibration, dca = _calibration_and_dca()
    dca_summary = _dca_summary(dca)
    patient_sensitivity = _patient_aggregation_sensitivity_rows(
        args.bootstrap_iterations, args.random_seed
    )
    rerun_provenance = _rerun_provenance_rows()
    r0_sensitivity = _r0_replacement_sensitivity(
        args.bootstrap_iterations, args.random_seed
    )
    ordinal, ordinal_calibration = _ordinal_rows(
        args.bootstrap_iterations, args.random_seed
    )
    spatial = _spatial_gate()

    args.output_root.mkdir(parents=True)
    per_seed.to_csv(args.output_root / "per_seed_auc.csv", index=False)
    ensemble.to_csv(args.output_root / "ensemble_metrics.csv", index=False)
    ensemble[
        [
            "method",
            "target",
            "evaluation_level",
            "n",
            "sensitivity_at_specificity_90",
        ]
    ].to_csv(args.output_root / "clinical_operating_points.csv", index=False)
    target_bootstrap.to_csv(
        args.output_root / "paired_bootstrap_by_target.csv", index=False
    )
    macro_bootstrap.to_csv(
        args.output_root / "paired_bootstrap_macro.csv", index=False
    )
    calibration.to_csv(args.output_root / "calibration.csv", index=False)
    dca.to_csv(args.output_root / "decision_curve.csv", index=False)
    dca_summary.to_csv(
        args.output_root / "decision_curve_summary.csv", index=False
    )
    patient_sensitivity.to_csv(
        args.output_root / "patient_aggregation_sensitivity.csv", index=False
    )
    rerun_provenance.to_csv(
        args.output_root / "implementation_rerun_provenance.csv", index=False
    )
    r0_sensitivity.to_csv(
        args.output_root / "r0_replacement_sensitivity.csv", index=False
    )
    ordinal.to_csv(args.output_root / "ordinal_metrics.csv", index=False)
    ordinal_calibration.to_csv(
        args.output_root / "ordinal_calibration.csv", index=False
    )
    (args.output_root / "spatial_gate.json").write_text(
        json.dumps(spatial, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        args.output_root,
        per_seed,
        ensemble,
        macro_bootstrap,
        calibration,
        dca_summary,
        patient_sensitivity,
        rerun_provenance,
        r0_sensitivity,
        ordinal,
        spatial,
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
