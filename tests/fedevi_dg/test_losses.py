import pytest
import torch
import torch.nn.functional as F
from fedevi_dg.losses import compute_class_weights, compute_method_loss


def test_class_weights_are_n_over_two_nc():
    assert torch.allclose(
        compute_class_weights([0, 0, 0, 1]), torch.tensor([4 / 6, 4 / 2])
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


def test_fedavg_is_weighted_original_cross_entropy_only():
    original = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([0.5, 2.0])
    result = compute_method_loss("fedavg", original, None, labels, weights)
    expected = F.cross_entropy(original, labels, weight=weights)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.original_ce, expected)


def test_loss_casts_cpu_class_weights_to_float64_logits():
    original = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    cpu_weights = compute_class_weights([0, 1])
    result = compute_method_loss("fedavg", original, None, labels, cpu_weights)
    expected = F.cross_entropy(
        original, labels, weight=torch.ones(2, dtype=torch.float64)
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
