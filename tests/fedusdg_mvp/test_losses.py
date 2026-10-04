import pytest
import torch
import torch.nn.functional as F

from fedusdg_mvp.losses import (
    compute_class_weights,
    compute_method_loss,
    fedprox_penalty,
)


def test_class_weights_are_n_over_two_nc():
    assert torch.allclose(
        compute_class_weights([0, 0, 0, 1]),
        torch.tensor([4 / 6, 4 / 2]),
    )


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([], "non-empty"),
        ([0, 0], "both binary classes"),
        ([1, 1], "both binary classes"),
        ([0, 1, 2], "binary labels"),
        ([0, 0.5, 1], "binary labels"),
        ([0, -1, 1], "binary labels"),
    ],
)
def test_class_weights_reject_invalid_or_degenerate_binary_labels(labels, message):
    with pytest.raises(ValueError, match=message):
        compute_class_weights(labels)


def test_fedprox_penalty_is_mu_over_two_times_squared_parameter_distance():
    local = [torch.tensor([1.0, 4.0]), torch.tensor([-1.0])]
    global_ = [torch.tensor([0.0, 2.0]), torch.tensor([2.0])]

    assert torch.allclose(fedprox_penalty(local, global_, mu=0.2), torch.tensor(1.4))


def test_fedprox_penalty_rejects_empty_parameter_lists():
    with pytest.raises(ValueError, match="non-empty"):
        fedprox_penalty([], [], mu=0.2)


def test_fedavg_is_weighted_original_cross_entropy_only():
    original = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([0.5, 2.0])

    result = compute_method_loss("fedavg", original, None, labels, weights)

    expected = F.cross_entropy(original, labels, weight=weights)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.original_ce, expected)
    assert result.augmented_ce.item() == 0
    assert result.consistency.item() == 0
    assert result.proximal.item() == 0


def test_loss_casts_cpu_class_weights_to_float64_logits():
    original = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    cpu_weights = compute_class_weights([0, 1])

    result = compute_method_loss("fedavg", original, None, labels, cpu_weights)

    expected = F.cross_entropy(
        original,
        labels,
        weight=torch.ones(2, dtype=torch.float64),
    )
    assert result.total.dtype == torch.float64
    assert torch.allclose(result.total, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_loss_moves_cpu_class_weights_to_cuda_logits():
    original = torch.tensor([[2.0, 0.0], [0.0, 1.0]], device="cuda")
    labels = torch.tensor([0, 1], device="cuda")
    cpu_weights = compute_class_weights([0, 1])

    result = compute_method_loss("fedavg", original, None, labels, cpu_weights)

    assert result.total.device.type == "cuda"


@pytest.mark.parametrize("method", ["fedavg", "fedprox"])
def test_single_view_methods_reject_augmented_logits(method):
    logits = torch.tensor([[2.0, 0.0]])

    with pytest.raises(ValueError, match=f"{method} must not receive augmented logits"):
        compute_method_loss(
            method,
            logits,
            logits,
            torch.tensor([0]),
            torch.ones(2),
        )


def test_fedprox_adds_the_supplied_canonical_penalty():
    original = torch.tensor([[2.0, 0.0]])
    proximal = fedprox_penalty(
        [torch.tensor([2.0, -1.0])],
        [torch.tensor([0.0, 1.0])],
        mu=0.5,
    )

    result = compute_method_loss(
        "fedprox",
        original,
        None,
        torch.tensor([0]),
        torch.ones(2),
        proximal=proximal,
    )

    expected_ce = F.cross_entropy(original, torch.tensor([0]), weight=torch.ones(2))
    assert torch.allclose(proximal, torch.tensor(2.0))
    assert torch.allclose(result.total, expected_ce + proximal)
    assert torch.allclose(result.proximal, proximal)


@pytest.mark.parametrize("method", ["whole", "depth"])
def test_whole_and_depth_use_equal_half_weighted_dual_view_objectives(method):
    original = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    augmented = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    weights = torch.ones(2)

    result = compute_method_loss(method, original, augmented, labels, weights)

    expected_original = F.cross_entropy(original, labels, weight=weights)
    expected_augmented = F.cross_entropy(augmented, labels, weight=weights)
    assert torch.allclose(result.total, 0.5 * expected_original + 0.5 * expected_augmented)
    assert result.consistency.item() == 0
    assert result.proximal.item() == 0


def test_non_mvp_consistency_weight_does_not_change_dual_view_objective():
    original = torch.tensor([[2.0, 0.0]])
    augmented = torch.tensor([[1.0, 0.0]])
    labels = torch.tensor([0])
    weights = torch.ones(2)

    default = compute_method_loss("whole", original, augmented, labels, weights)
    overridden = compute_method_loss(
        "whole",
        original,
        augmented,
        labels,
        weights,
        consistency_weight=9.0,
    )

    assert torch.allclose(overridden.total, default.total)


def test_mvp_adds_half_batchmean_forward_kl_at_temperature_one():
    original = torch.tensor([[1.5, -0.5], [-0.2, 0.8]])
    augmented = torch.tensor([[0.2, 0.1], [0.6, -0.4]])
    labels = torch.tensor([0, 1])
    weights = torch.ones(2)

    result = compute_method_loss(
        "fedus_dg_mvp", original, augmented, labels, weights
    )

    original_ce = F.cross_entropy(original, labels, weight=weights)
    augmented_ce = F.cross_entropy(augmented, labels, weight=weights)
    expected_kl = F.kl_div(
        F.log_softmax(augmented, dim=1),
        F.softmax(original, dim=1),
        reduction="batchmean",
    )
    assert torch.allclose(result.consistency, expected_kl)
    assert torch.allclose(
        result.total,
        0.5 * original_ce + 0.5 * augmented_ce + 0.5 * expected_kl,
    )


def test_mvp_rejects_noncanonical_consistency_weight():
    with pytest.raises(ValueError, match="consistency_weight must be 0.5"):
        compute_method_loss(
            "fedus_dg_mvp",
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.5, 0.0]]),
            torch.tensor([0]),
            torch.ones(2),
            consistency_weight=0.25,
        )


def test_dual_view_method_rejects_augmented_logits_with_a_different_dtype():
    with pytest.raises(ValueError, match="same dtype"):
        compute_method_loss(
            "whole",
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.5, 0.0]], dtype=torch.float64),
            torch.tensor([0]),
            torch.ones(2),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dual_view_method_rejects_augmented_logits_on_a_different_device():
    with pytest.raises(ValueError, match="same device"):
        compute_method_loss(
            "whole",
            torch.tensor([[1.0, 0.0]], device="cuda"),
            torch.tensor([[0.5, 0.0]]),
            torch.tensor([0], device="cuda"),
            torch.ones(2),
        )


def test_consistency_stops_original_branch_gradient():
    original = torch.tensor([[1.5, -0.5]], requires_grad=True)
    augmented = torch.tensor([[0.2, 0.1]], requires_grad=True)
    out = compute_method_loss(
        "fedus_dg_mvp",
        original,
        augmented,
        torch.tensor([0]),
        torch.ones(2),
        0.5,
    )

    grads = torch.autograd.grad(
        out.consistency, (original, augmented), allow_unused=True
    )
    assert grads[0] is None or torch.all(grads[0] == 0)
    assert grads[1] is not None


def test_unknown_method_is_rejected_instead_of_inheriting_an_objective():
    with pytest.raises(ValueError, match="unknown method"):
        compute_method_loss(
            "safa",
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            torch.ones(2),
        )
