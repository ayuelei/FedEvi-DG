from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class SourceScore:
    auc: float
    loss: float


def _validated_binary_arrays(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if labels_array.size == 0:
        raise ValueError("labels and scores must be non-empty")
    if labels_array.size != scores_array.size:
        raise ValueError("labels and scores must have equal length")
    if not np.all(np.isin(labels_array, [0, 1])):
        raise ValueError("labels must be binary 0 and 1")
    if np.unique(labels_array).size != 2:
        raise ValueError("labels must contain both classes")
    if not np.all(np.isfinite(scores_array)) or not np.all(
        (0 <= scores_array) & (scores_array <= 1)
    ):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return labels_array.astype(int, copy=False), scores_array


def _patient_groups(
    labels: np.ndarray,
    patient_ids: Sequence[Hashable],
) -> dict[Hashable, list[int]]:
    if len(patient_ids) != labels.size:
        raise ValueError("patient IDs and labels must have equal length")

    groups: dict[Hashable, list[int]] = {}
    patient_labels: dict[Hashable, int] = {}
    for index, (label, patient_id) in enumerate(zip(labels, patient_ids, strict=True)):
        label = int(label)
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError("expected one label per patient")
        patient_labels[patient_id] = label
        groups.setdefault(patient_id, []).append(index)
    return groups


def binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, object]:
    labels_array, scores_array = _validated_binary_arrays(labels, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    predictions = (scores_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels_array, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "auc": float(roc_auc_score(labels_array, scores_array)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "macro_f1": float(f1_score(labels_array, predictions, average="macro")),
        "loss": float(log_loss(labels_array, scores_array, labels=[0, 1])),
        "labels": labels_array.tolist(),
        "scores": scores_array.tolist(),
        "predictions": predictions.tolist(),
        "n": int(labels_array.size),
    }


def clinical_reliability_metrics(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> dict[str, float]:
    """Calibration and a source-independent high-specificity operating point."""
    if bins <= 1:
        raise ValueError("calibration requires at least two bins")
    labels_array, scores_array = _validated_binary_arrays(labels, scores)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    calibration_error = 0.0
    for bin_index in range(bins):
        lower, upper = bin_edges[bin_index : bin_index + 2]
        selected = (scores_array >= lower) & (
            scores_array <= upper if bin_index == bins - 1 else scores_array < upper
        )
        if np.any(selected):
            calibration_error += float(np.mean(selected)) * abs(
                float(np.mean(scores_array[selected]))
                - float(np.mean(labels_array[selected]))
            )

    negative_scores = scores_array[labels_array == 0]
    positive_scores = scores_array[labels_array == 1]
    # Choose the largest sensitivity attainable while permitting no more than
    # floor(10% * negatives) false positives. This is descriptive evaluation,
    # never a threshold used by training or checkpoint selection.
    allowed_false_positives = int(np.floor(0.10 * len(negative_scores)))
    candidates = np.unique(np.concatenate(([1.0 + np.finfo(float).eps], scores_array)))
    sensitivities = []
    for threshold in candidates:
        false_positives = int(np.count_nonzero(negative_scores >= threshold))
        if false_positives <= allowed_false_positives:
            sensitivities.append(float(np.mean(positive_scores >= threshold)))
    return {
        "brier": float(np.mean((scores_array - labels_array) ** 2)),
        "ece": calibration_error,
        "sensitivity_at_specificity_90": max(sensitivities, default=0.0),
    }


def macro_source_score(source_metrics: dict[str, dict[str, float]]) -> SourceScore:
    if not source_metrics:
        raise ValueError("macro score requires at least one source")
    aucs = [metrics["auc"] for metrics in source_metrics.values()]
    losses = [metrics["loss"] for metrics in source_metrics.values()]
    if not np.all(np.isfinite(aucs)) or not np.all(np.isfinite(losses)):
        raise ValueError("source AUC and loss must be finite")
    return SourceScore(
        auc=float(np.mean(aucs)),
        loss=float(np.mean(losses)),
    )


def is_better_checkpoint(
    candidate: tuple[float, float, int],
    incumbent: tuple[float, float, int],
) -> bool:
    candidate_auc, candidate_loss, candidate_round = candidate
    incumbent_auc, incumbent_loss, incumbent_round = incumbent
    if not np.all(
        np.isfinite(
            [candidate_auc, candidate_loss, incumbent_auc, incumbent_loss]
        )
    ):
        raise ValueError("checkpoint AUC and loss must be finite")
    return (-candidate_auc, candidate_loss, candidate_round) < (
        -incumbent_auc,
        incumbent_loss,
        incumbent_round,
    )


def select_source_threshold(
    labels_by_source: dict[str, Sequence[int]],
    scores_by_source: dict[str, Sequence[float]],
    step: float = 0.01,
) -> float:
    if step != 0.01:
        raise ValueError("threshold step must be 0.01")
    if not labels_by_source:
        raise ValueError("threshold selection requires at least one source")
    if labels_by_source.keys() != scores_by_source.keys():
        raise ValueError("label and score sources must match")
    validated_sources = {
        source: _validated_binary_arrays(
            labels_by_source[source], scores_by_source[source]
        )
        for source in labels_by_source
    }

    candidates: list[tuple[Fraction, int]] = []
    for percent in range(101):
        threshold = percent / 100
        source_balanced_accuracies = []
        for source_labels, source_scores in validated_sources.values():
            positive = source_labels == 1
            negative = ~positive
            true_positive = int(np.count_nonzero(source_scores[positive] >= threshold))
            true_negative = int(np.count_nonzero(source_scores[negative] < threshold))
            source_balanced_accuracies.append(
                (
                    Fraction(true_positive, int(np.count_nonzero(positive)))
                    + Fraction(true_negative, int(np.count_nonzero(negative)))
                )
                / 2
            )
        candidates.append(
            (
                sum(source_balanced_accuracies, start=Fraction())
                / len(source_balanced_accuracies),
                percent,
            )
        )

    best_score = max(score for score, _ in candidates)
    tied_percents = [percent for score, percent in candidates if score == best_score]
    return min(tied_percents, key=lambda percent: (abs(percent - 50), percent)) / 100


def patient_level_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    patient_ids: Sequence[Hashable],
) -> dict[str, object]:
    labels_array, scores_array = _validated_binary_arrays(labels, scores)
    grouped = _patient_groups(labels_array, patient_ids)
    patient_labels = [int(labels_array[indices[0]]) for indices in grouped.values()]
    patient_scores = [
        float(np.mean(scores_array[indices])) for indices in grouped.values()
    ]
    patient_labels_array, patient_scores_array = _validated_binary_arrays(
        patient_labels, patient_scores
    )
    return {
        "auc": float(roc_auc_score(patient_labels_array, patient_scores_array)),
        "patient_labels": patient_labels_array.tolist(),
        "patient_scores": patient_scores_array.tolist(),
        "patient_ids": list(grouped),
        "n": len(grouped),
    }


def cluster_bootstrap_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    patient_ids: Sequence[Hashable],
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    labels_array, scores_array = _validated_binary_arrays(labels, scores)
    groups = list(_patient_groups(labels_array, patient_ids).values())
    rng = np.random.default_rng(seed)
    bootstrap_aucs: list[float] = []

    for _ in range(n_bootstrap):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        sampled_indices = np.concatenate([groups[index] for index in sampled_groups])
        sampled_labels = labels_array[sampled_indices]
        if np.unique(sampled_labels).size < 2:
            continue
        bootstrap_aucs.append(
            float(roc_auc_score(sampled_labels, scores_array[sampled_indices]))
        )

    if not bootstrap_aucs:
        raise ValueError("no valid bootstrap replicates contained both classes")
    lower, upper = np.percentile(bootstrap_aucs, [2.5, 97.5])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": len(bootstrap_aucs),
    }
