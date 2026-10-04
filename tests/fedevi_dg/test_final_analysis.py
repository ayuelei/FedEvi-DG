import numpy as np
import pandas as pd

from scripts.analyze_fedevi_final import (
    CANDIDATE,
    _dca_summary,
    _evaluation_units,
    _sensitivity_at_specificity,
    _temperature_scale,
)


def test_temperature_one_preserves_probabilities():
    scores = np.array([0.01, 0.2, 0.5, 0.8, 0.99])

    assert np.allclose(_temperature_scale(scores, 1.0), scores)


def test_reliable_patient_ids_are_aggregated_before_metrics():
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b"],
            "patient_grouping_reliable": [True, True, True],
            "label": [0, 0, 1],
            "score": [0.1, 0.3, 0.9],
        }
    )

    labels, scores, level = _evaluation_units(frame)

    assert level == "patient_mean"
    assert labels.tolist() == [0, 1]
    assert np.allclose(scores, [0.2, 0.9])


def test_dca_summary_uses_fixed_threshold_window():
    frame = pd.DataFrame(
        {
            "target": ["A", "A", "A"],
            "threshold": [0.04, 0.05, 0.30],
            "treat_none": [0.0, 0.0, 0.0],
            "treat_all": [0.1, 0.1, 0.1],
            "fedavg": [0.1, 0.11, 0.12],
            CANDIDATE: [0.2, 0.13, 0.11],
        }
    )

    result = _dca_summary(frame).iloc[0]

    assert result["threshold_count"] == 2
    assert result["candidate_better_than_fedavg_thresholds"] == 1
    assert result["candidate_better_than_best_default_thresholds"] == 2


def test_sensitivity_at_ninety_specificity_uses_conservative_discrete_floor():
    labels = np.asarray([0] * 10 + [1] * 4)
    scores = np.asarray(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.90]
        + [0.95, 0.85, 0.50, 0.12]
    )

    value = _sensitivity_at_specificity(labels, scores, specificity=0.90)

    assert value == 0.75
