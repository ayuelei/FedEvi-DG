import pytest

from fedusdg_mvp.metrics import (
    binary_metrics,
    cluster_bootstrap_auc,
    is_better_checkpoint,
    macro_source_score,
    patient_level_auc,
    select_source_threshold,
)


def test_macro_auc_is_domain_balanced():
    score = macro_source_score(
        {
            "small": {"auc": 0.9, "loss": 0.7, "n": 10},
            "large": {"auc": 0.5, "loss": 0.3, "n": 1000},
        }
    )

    assert score.auc == 0.7
    assert score.loss == 0.5


def test_macro_source_score_rejects_an_empty_source_set():
    with pytest.raises(ValueError, match="at least one source"):
        macro_source_score({})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("auc", float("nan")),
        ("auc", float("inf")),
        ("loss", float("nan")),
        ("loss", float("-inf")),
    ],
)
def test_macro_source_score_rejects_nonfinite_auc_or_loss(field, value):
    metrics = {"source": {"auc": 0.8, "loss": 0.3}}
    metrics["source"][field] = value

    with pytest.raises(ValueError, match="finite"):
        macro_source_score(metrics)


def test_checkpoint_tie_breaks_loss_then_earlier_round():
    assert is_better_checkpoint((0.81, 9.0, 20), (0.80, 0.1, 1))
    assert not is_better_checkpoint((0.79, 0.1, 1), (0.80, 9.0, 20))
    assert is_better_checkpoint((0.8, 0.4, 10), (0.8, 0.5, 5))
    assert is_better_checkpoint((0.8, 0.4, 4), (0.8, 0.4, 5))
    assert not is_better_checkpoint((0.8, 0.4, 10), (0.8, 0.4, 5))


@pytest.mark.parametrize(
    ("candidate", "incumbent"),
    [
        ((float("nan"), 0.3, 1), (0.8, 0.3, 2)),
        ((0.8, float("inf"), 1), (0.8, 0.3, 2)),
        ((0.8, 0.3, 1), (float("nan"), 0.3, 2)),
        ((0.8, 0.3, 1), (0.8, float("-inf"), 2)),
    ],
)
def test_checkpoint_comparison_rejects_nonfinite_auc_or_loss(candidate, incumbent):
    with pytest.raises(ValueError, match="finite"):
        is_better_checkpoint(candidate, incumbent)


def test_threshold_maximizes_mean_source_balanced_accuracy():
    labels = {"A": [0, 1], "B": [0, 1]}
    scores = {"A": [0.1, 0.9], "B": [0.4, 0.6]}

    assert select_source_threshold(labels, scores, 0.01) == 0.5


def test_threshold_grid_classifies_a_score_at_exactly_seventy_percent_positive():
    assert select_source_threshold(
        {"source": [0, 1]},
        {"source": [0.69, 0.70]},
    ) == pytest.approx(0.70)


def test_threshold_distance_tie_uses_exact_integer_percent_distance():
    labels = {"left": [0, 1], "right": [0, 1]}
    scores = {"left": [0.28, 0.29], "right": [0.70, 0.71]}

    assert select_source_threshold(labels, scores) == pytest.approx(0.29)


def test_threshold_strictly_prefers_true_balanced_accuracy_maximum():
    labels = {"source": [0] * 100_000 + [1]}
    scores = {"source": [0.1] * 99_999 + [0.5, 0.9]}

    assert select_source_threshold(labels, scores) == pytest.approx(0.51)


def test_threshold_tie_prefers_nearest_half_then_smaller_value():
    labels = {"left": [0, 1], "right": [0, 1]}
    scores = {"left": [0.48, 0.49], "right": [0.50, 0.51]}

    assert select_source_threshold(labels, scores) == pytest.approx(0.49)


def test_threshold_balanced_accuracy_is_unweighted_across_source_domains():
    labels = {
        "small": [0, 1],
        "large": [0, 1] * 100,
    }
    scores = {
        "small": [0.3, 0.4],
        "large": [0.7, 0.8] * 100,
    }

    assert select_source_threshold(labels, scores) == pytest.approx(0.4)


def test_threshold_search_includes_one():
    assert select_source_threshold({"source": [0, 1]}, {"source": [0.99, 1.0]}) == 1.0


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ({}, {}, "at least one source"),
        ({"A": []}, {"A": []}, "non-empty"),
        ({"A": [0, 0]}, {"A": [0.1, 0.2]}, "both classes"),
    ],
)
def test_threshold_selection_hard_fails_on_empty_or_single_class_source_val(
    labels, scores, message
):
    with pytest.raises(ValueError, match=message):
        select_source_threshold(labels, scores)


def test_patient_auc_averages_images_before_scoring():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.3, 0.7, 0.9]
    patients = ["p0", "p0", "p1", "p1"]

    result = patient_level_auc(labels, scores, patients)

    assert result["patient_scores"] == [0.2, 0.8]
    assert result["auc"] == 1.0


def test_binary_metrics_returns_complete_fixed_threshold_results():
    result = binary_metrics(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.8, 0.4, 0.9],
        threshold=0.5,
    )

    assert set(result) == {
        "auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "macro_f1",
        "loss",
        "labels",
        "scores",
        "predictions",
        "n",
    }
    assert result["auc"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["sensitivity"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["loss"] == pytest.approx(0.684112418905977)
    assert result["labels"] == [0, 0, 1, 1]
    assert result["scores"] == [0.1, 0.8, 0.4, 0.9]
    assert result["predictions"] == [0, 1, 0, 1]
    assert result["n"] == 4


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([], [], "non-empty"),
        ([0, 0], [0.1, 0.2], "both classes"),
        ([1, 1], [0.8, 0.9], "both classes"),
    ],
)
def test_binary_metrics_hard_fails_on_empty_or_single_class_data(
    labels, scores, message
):
    with pytest.raises(ValueError, match=message):
        binary_metrics(labels, scores)


@pytest.mark.parametrize("scores", [[-0.1, 0.8], [0.1, 1.1], [0.1, float("nan")]])
def test_binary_metrics_rejects_non_probability_scores(scores):
    with pytest.raises(ValueError, match="finite probabilities"):
        binary_metrics([0, 1], scores)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_binary_metrics_rejects_an_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        binary_metrics([0, 1], [0.1, 0.9], threshold)


def test_patient_auc_rejects_a_patient_with_multiple_labels():
    with pytest.raises(ValueError, match="one label per patient"):
        patient_level_auc(
            labels=[0, 1, 1],
            scores=[0.1, 0.2, 0.9],
            patient_ids=["mixed", "mixed", "positive"],
        )


@pytest.mark.parametrize(
    ("labels", "scores", "patients", "message"),
    [
        ([], [], [], "non-empty"),
        ([0, 0], [0.1, 0.2], ["a", "b"], "both classes"),
    ],
)
def test_patient_auc_hard_fails_on_empty_or_single_class_patient_data(
    labels, scores, patients, message
):
    with pytest.raises(ValueError, match=message):
        patient_level_auc(labels, scores, patients)


def test_cluster_bootstrap_is_deterministic_and_discards_single_class_draws():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    patients = ["n0", "n0", "n1", "p0", "p0", "p1"]

    first = cluster_bootstrap_auc(labels, scores, patients, seed=17)
    second = cluster_bootstrap_auc(labels, scores, patients, seed=17)

    assert first == second
    assert first["lower"] == pytest.approx(1.0)
    assert first["upper"] == pytest.approx(1.0)
    assert first["valid_replicates"] == 1761


def test_cluster_bootstrap_rejects_a_patient_with_multiple_labels():
    with pytest.raises(ValueError, match="one label per patient"):
        cluster_bootstrap_auc(
            labels=[0, 1, 1],
            scores=[0.1, 0.2, 0.9],
            patient_ids=["mixed", "mixed", "positive"],
        )


def test_cluster_bootstrap_rejects_a_run_with_no_valid_replicates():
    with pytest.raises(ValueError, match="no valid bootstrap replicates"):
        cluster_bootstrap_auc(
            labels=[0, 1],
            scores=[0.1, 0.9],
            patient_ids=["negative", "positive"],
            n_bootstrap=1,
            seed=0,
        )


def test_cluster_bootstrap_rejects_a_nonpositive_replicate_count():
    with pytest.raises(ValueError, match="positive"):
        cluster_bootstrap_auc(
            labels=[0, 1],
            scores=[0.1, 0.9],
            patient_ids=["negative", "positive"],
            n_bootstrap=0,
        )
