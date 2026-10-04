"""Class-balanced pathology loss used by all paper methods."""

from collections.abc import Iterable
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class LossResult:
    total: torch.Tensor
    original_ce: torch.Tensor


def compute_class_weights(labels: Iterable[int] | torch.Tensor) -> torch.Tensor:
    labels_tensor = torch.as_tensor(labels)
    if labels_tensor.numel() == 0:
        raise ValueError("labels must be non-empty")
    if not torch.all((labels_tensor == 0) | (labels_tensor == 1)).item():
        raise ValueError("class weights require binary labels 0 and 1")

    labels_tensor = labels_tensor.to(dtype=torch.long)
    counts = torch.bincount(labels_tensor, minlength=2).to(torch.get_default_dtype())
    if torch.any(counts == 0).item():
        raise ValueError("labels must contain both binary classes")
    return labels_tensor.numel() / (2 * counts)


def compute_method_loss(
    method: str,
    original_logits: torch.Tensor,
    augmented_logits: torch.Tensor | None,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> LossResult:
    if method != "fedavg":
        raise ValueError(f"unknown classification loss: {method}")
    if augmented_logits is not None:
        raise ValueError("fedavg must not receive augmented logits")
    class_weights = class_weights.to(
        device=original_logits.device,
        dtype=original_logits.dtype,
    )
    original_ce = F.cross_entropy(original_logits, labels, weight=class_weights)
    return LossResult(total=original_ce, original_ce=original_ce)
