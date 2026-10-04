from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossResult:
    total: torch.Tensor
    original_ce: torch.Tensor
    augmented_ce: torch.Tensor
    consistency: torch.Tensor
    proximal: torch.Tensor
    ordinal: torch.Tensor | None = None


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


def fedprox_penalty(
    local_parameters: Iterable[torch.Tensor],
    global_parameters: Iterable[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    squared_distance = None
    for local, global_ in zip(local_parameters, global_parameters, strict=True):
        term = (local - global_).square().sum()
        squared_distance = term if squared_distance is None else squared_distance + term

    if squared_distance is None:
        raise ValueError("parameter lists must be non-empty")
    return (mu / 2) * squared_distance


def birads_ordinal_penalty(
    logits: torch.Tensor,
    levels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise monotonicity penalty on the malignant logit.

    ``levels`` carries within-domain BI-RADS ranks as floats; NaN marks
    samples without a reliable ordinal level, which contribute nothing.
    Every strictly ordered pair (level_i < level_j) is penalized when the
    malignant logit does not clear ``margin`` in the same order.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("ordinal penalty requires two-class logits")
    if levels.shape != logits.shape[:1]:
        raise ValueError("levels must align with the batch dimension")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    malignant = logits[:, 1].to(torch.float64)
    finite = torch.isfinite(levels.to(torch.float64))
    order = levels.to(torch.float64)[finite]
    scores = malignant[finite]
    count = order.shape[0]
    if count < 2:
        return logits.new_zeros((), dtype=malignant.dtype)
    later, earlier = torch.tril_indices(count, count, offset=-1)
    level_difference = order[later] - order[earlier]
    ordered = level_difference != 0
    if not torch.any(ordered):
        return logits.new_zeros((), dtype=malignant.dtype)
    direction = torch.sign(level_difference)
    score_difference = scores[later] - scores[earlier]
    violations = F.relu(margin - direction * score_difference)[ordered]
    return violations.mean().to(dtype=malignant.dtype)


def compute_method_loss(
    method: str,
    original_logits: torch.Tensor,
    augmented_logits: torch.Tensor | None,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    consistency_weight: float = 0.5,
    proximal: torch.Tensor | None = None,
    birads_levels: torch.Tensor | None = None,
    birads_weight: float = 0.25,
    ordinal_margin: float = 1.0,
) -> LossResult:
    valid_methods = {
        "fedavg",
        "fedprox",
        "whole",
        "depth",
        "fedus_dg_mvp",
        "factor",
        "virtual",
        "fedphys",
        "control",
        "virtual_cons",
        "control_full",
        "control_cons",
        "fedbn",
    }
    if method not in valid_methods:
        raise ValueError(f"unknown method: {method}")

    if method == "fedus_dg_mvp" and consistency_weight != 0.5:
        raise ValueError("fedus_dg_mvp consistency_weight must be 0.5")
    if (
        method in {"fedphys", "virtual_cons", "control_full", "control_cons"}
        and consistency_weight != 0.5
    ):
        raise ValueError(f"{method} consistency_weight must be 0.5")

    spectral_methods = {
        "whole",
        "depth",
        "fedus_dg_mvp",
        "factor",
        "virtual",
        "fedphys",
        "control",
        "virtual_cons",
        "control_full",
        "control_cons",
        "fedbn",
    }

    if method in {"fedavg", "fedprox", "fedbn"}:
        if augmented_logits is not None:
            raise ValueError(f"{method} must not receive augmented logits")
    else:
        if augmented_logits is None:
            raise ValueError(f"{method} requires augmented logits")
        if augmented_logits.device != original_logits.device:
            raise ValueError("original and augmented logits must use the same device")
        if augmented_logits.dtype != original_logits.dtype:
            raise ValueError("original and augmented logits must use the same dtype")

    if method in {"fedphys", "control_full"} and birads_levels is None:
        raise ValueError(f"{method} requires birads_levels per batch")

    zero = original_logits.new_zeros(())
    class_weights = class_weights.to(
        device=original_logits.device,
        dtype=original_logits.dtype,
    )
    original_ce = F.cross_entropy(original_logits, labels, weight=class_weights)

    if method in {"fedavg", "fedprox", "fedbn"}:
        prox = proximal if method == "fedprox" and proximal is not None else zero
        return LossResult(original_ce + prox, original_ce, zero, zero, prox)

    augmented_ce = F.cross_entropy(augmented_logits, labels, weight=class_weights)
    base = 0.5 * (original_ce + augmented_ce)
    consistency = zero
    if method in {
        "fedus_dg_mvp",
        "fedphys",
        "virtual_cons",
        "control_full",
        "control_cons",
    }:
        consistency = F.kl_div(
            F.log_softmax(augmented_logits, dim=1),
            F.softmax(original_logits, dim=1).detach(),
            reduction="batchmean",
        )

    ordinal = zero
    if method in {"fedphys", "control_full"}:
        ordinal = birads_ordinal_penalty(
            original_logits, birads_levels, margin=ordinal_margin
        )

    return LossResult(
        base + consistency_weight * consistency + birads_weight * ordinal,
        original_ce,
        augmented_ce,
        consistency,
        zero,
        ordinal,
    )
