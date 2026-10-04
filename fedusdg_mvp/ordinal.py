from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DomainOrdinalScale:
    minimum: float
    maximum: float
    count: int

    def __post_init__(self) -> None:
        if not self.maximum > self.minimum:
            raise ValueError("ordinal scale requires at least two distinct levels")
        if self.count < 2:
            raise ValueError("ordinal scale requires at least two observations")

    def normalize(self, levels: torch.Tensor) -> torch.Tensor:
        levels = torch.as_tensor(levels, dtype=torch.float64)
        normalized = (levels - self.minimum) / (self.maximum - self.minimum)
        finite = torch.isfinite(levels)
        if torch.any((normalized[finite] < 0.0) | (normalized[finite] > 1.0)):
            raise ValueError("ordinal levels fall outside the fitted client scale")
        return normalized


@dataclass(frozen=True)
class OrdinalBridge:
    """Client-balanced pathology-to-ordinal sufficient statistics.

    BI-RADS encodings differ across source datasets. Each client is therefore
    normalized independently to [0, 1] before label-conditional moments are
    combined with equal client weight. Only counts and moments need to leave a
    client; no image or per-patient prediction is represented here.
    """

    means: torch.Tensor
    variances: torch.Tensor
    sample_counts: torch.Tensor
    client_counts: torch.Tensor
    scales: Mapping[str, DomainOrdinalScale]

    def __post_init__(self) -> None:
        for name, value in (
            ("means", self.means),
            ("variances", self.variances),
            ("sample_counts", self.sample_counts),
            ("client_counts", self.client_counts),
        ):
            if torch.as_tensor(value).shape != (2,):
                raise ValueError(f"{name} must contain one value per binary class")
        if not torch.isfinite(self.means).all() or not torch.isfinite(
            self.variances
        ).all():
            raise ValueError("ordinal bridge moments must be finite")
        if torch.any((self.means < 0.0) | (self.means > 1.0)):
            raise ValueError("ordinal bridge means must lie in [0, 1]")
        if torch.any(self.variances < 0.0):
            raise ValueError("ordinal bridge variances must be non-negative")
        if torch.any(self.sample_counts <= 0) or torch.any(self.client_counts <= 0):
            raise ValueError("both pathology classes require ordinal observations")

    def normalize(self, domain: str, levels: torch.Tensor) -> torch.Tensor:
        try:
            scale = self.scales[domain]
        except KeyError as error:
            return torch.full_like(
                torch.as_tensor(levels, dtype=torch.float64), float("nan")
            )
        return scale.normalize(levels)

    def priors(self, labels: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        labels = torch.as_tensor(labels, dtype=torch.long)
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("ordinal bridge requires binary pathology labels")
        return self.means.to(device=labels.device, dtype=dtype)[labels]

    def to_dict(self) -> dict[str, object]:
        return {
            "means": [float(value) for value in self.means],
            "variances": [float(value) for value in self.variances],
            "sample_counts": [int(value) for value in self.sample_counts],
            "client_counts": [int(value) for value in self.client_counts],
            "domain_scales": {
                domain: {
                    "minimum": scale.minimum,
                    "maximum": scale.maximum,
                    "count": scale.count,
                }
                for domain, scale in sorted(self.scales.items())
            },
            "aggregation": "equal_client_weight_label_conditional_moments",
        }


def estimate_ordinal_bridge(
    records: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> OrdinalBridge:
    """Estimate a bridge from local ``(pathology, ordinal level)`` records."""
    if not records:
        raise ValueError("ordinal bridge requires at least one annotated client")

    scales: dict[str, DomainOrdinalScale] = {}
    per_class: dict[int, list[tuple[float, float, int]]] = {0: [], 1: []}
    for domain, (labels_value, levels_value) in sorted(records.items()):
        labels = torch.as_tensor(labels_value, dtype=torch.long).flatten()
        levels = torch.as_tensor(levels_value, dtype=torch.float64).flatten()
        if labels.shape != levels.shape:
            raise ValueError(f"labels and ordinal levels must align for {domain}")
        finite = torch.isfinite(levels)
        if not torch.any(finite):
            continue
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError(f"pathology labels must be binary for {domain}")
        observed = levels[finite]
        minimum = float(observed.min())
        maximum = float(observed.max())
        if not maximum > minimum:
            continue
        scale = DomainOrdinalScale(minimum, maximum, int(finite.sum()))
        scales[domain] = scale
        normalized = scale.normalize(levels)
        for label in (0, 1):
            selected = normalized[finite & (labels == label)]
            if selected.numel() == 0:
                continue
            per_class[label].append(
                (
                    float(selected.mean()),
                    float(selected.var(unbiased=False)),
                    int(selected.numel()),
                )
            )

    if not scales:
        raise ValueError("no client has two distinct finite ordinal levels")

    means = []
    variances = []
    sample_counts = []
    client_counts = []
    for label in (0, 1):
        moments = per_class[label]
        if not moments:
            raise ValueError(
                f"ordinal bridge has no observations for pathology class {label}"
            )
        class_mean = sum(item[0] for item in moments) / len(moments)
        # Total variance under an equal-weight mixture of client distributions.
        class_variance = sum(
            variance + (mean - class_mean) ** 2
            for mean, variance, _ in moments
        ) / len(moments)
        means.append(class_mean)
        variances.append(class_variance)
        sample_counts.append(sum(item[2] for item in moments))
        client_counts.append(len(moments))

    return OrdinalBridge(
        means=torch.tensor(means, dtype=torch.float64),
        variances=torch.tensor(variances, dtype=torch.float64),
        sample_counts=torch.tensor(sample_counts, dtype=torch.long),
        client_counts=torch.tensor(client_counts, dtype=torch.long),
        scales=scales,
    )


def build_shuffled_level_maps(
    records: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> dict[str, dict[float, float]]:
    """Create fixed random derangements of each client's ordinal semantics."""
    mappings: dict[str, dict[float, float]] = {}
    for domain, (_, levels_value) in sorted(records.items()):
        levels = torch.as_tensor(levels_value, dtype=torch.float64).flatten()
        unique = sorted(
            float(value) for value in levels[torch.isfinite(levels)].unique()
        )
        if len(unique) < 2:
            continue
        shuffled = list(unique)
        # Sattolo's algorithm creates one cycle and therefore has no fixed
        # points. Hash-derived choices make the mapping reproducible without
        # depending on Python or torch PRNG implementation details.
        for index in range(len(shuffled) - 1, 0, -1):
            digest = hashlib.sha256(
                f"{seed}:{domain}:{index}".encode("utf-8")
            ).digest()
            swap_index = int.from_bytes(digest[:8], "big") % index
            shuffled[index], shuffled[swap_index] = (
                shuffled[swap_index],
                shuffled[index],
            )
        mappings[domain] = dict(zip(unique, shuffled, strict=True))
    return mappings


def apply_level_map(
    levels: torch.Tensor, mapping: Mapping[float, float]
) -> torch.Tensor:
    values = torch.as_tensor(levels, dtype=torch.float64)
    transformed = values.clone()
    finite = torch.isfinite(values)
    for source, destination in mapping.items():
        transformed[finite & (values == source)] = destination
    observed = {float(value) for value in values[finite].unique()}
    missing = observed - set(mapping)
    if missing:
        raise ValueError(f"ordinal permutation misses levels: {sorted(missing)}")
    return transformed


def categorical_birads_loss(
    original_logits: torch.Tensor,
    augmented_logits: torch.Tensor | None,
    normalized_levels: torch.Tensor,
) -> torch.Tensor:
    """Unordered fixed-bin BI-RADS auxiliary control."""
    original = torch.as_tensor(original_logits)
    if original.ndim != 2 or original.shape[1] < 2:
        raise ValueError("categorical BI-RADS head must return shape (N, C), C >= 2")
    levels = torch.as_tensor(
        normalized_levels, device=original.device, dtype=original.dtype
    ).flatten()
    if levels.shape != original.shape[:1]:
        raise ValueError("categorical logits and normalized levels must align")
    finite = torch.isfinite(levels)
    if not torch.any(finite):
        return original.sum() * 0.0
    targets = levels[finite]
    if torch.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("normalized categorical targets must lie in [0, 1]")
    bins = torch.round(targets * (original.shape[1] - 1)).to(torch.long)
    loss = F.cross_entropy(original[finite], bins)
    if augmented_logits is not None:
        augmented = torch.as_tensor(
            augmented_logits, device=original.device, dtype=original.dtype
        )
        if augmented.shape != original.shape:
            raise ValueError("augmented categorical logits must align with original")
        loss = 0.5 * (loss + F.cross_entropy(augmented[finite], bins))
    return loss


def supervised_ordinal_loss(
    original_risk_logits: torch.Tensor,
    augmented_risk_logits: torch.Tensor | None,
    normalized_levels: torch.Tensor,
    *,
    pairwise_weight: float = 0.25,
) -> torch.Tensor:
    """Soft ordinal calibration plus a smooth within-client ranking term."""
    original = torch.as_tensor(original_risk_logits).flatten()
    levels = torch.as_tensor(
        normalized_levels, device=original.device, dtype=original.dtype
    ).flatten()
    if original.shape != levels.shape:
        raise ValueError("risk logits and normalized levels must align")
    if pairwise_weight < 0.0:
        raise ValueError("pairwise_weight must be non-negative")
    finite = torch.isfinite(levels)
    if not torch.any(finite):
        return original.sum() * 0.0
    targets = levels[finite]
    if torch.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("normalized ordinal targets must lie in [0, 1]")

    logits = original[finite]
    calibration = F.binary_cross_entropy_with_logits(logits, targets)
    if augmented_risk_logits is not None:
        augmented = torch.as_tensor(
            augmented_risk_logits, device=original.device, dtype=original.dtype
        ).flatten()
        if augmented.shape != original.shape:
            raise ValueError("augmented risk logits must align with original logits")
        calibration = 0.5 * (
            calibration
            + F.binary_cross_entropy_with_logits(augmented[finite], targets)
        )

    count = targets.numel()
    if count < 2:
        return calibration
    later, earlier = torch.tril_indices(count, count, offset=-1, device=logits.device)
    target_difference = targets[later] - targets[earlier]
    ordered = target_difference != 0
    if not torch.any(ordered):
        return calibration
    direction = torch.sign(target_difference[ordered])
    score_difference = logits[later[ordered]] - logits[earlier[ordered]]
    ranking = F.softplus(-direction * score_difference).mean()
    return calibration + pairwise_weight * ranking


@dataclass(frozen=True)
class RelayLoss:
    loss: torch.Tensor
    mean_weight: torch.Tensor
    accepted_fraction: torch.Tensor


def ordinal_relay_loss(
    student_original_logits: torch.Tensor,
    student_augmented_logits: torch.Tensor,
    teacher_original_logits: torch.Tensor,
    teacher_augmented_logits: torch.Tensor,
    labels: torch.Tensor,
    bridge: OrdinalBridge,
    *,
    teacher_fraction: float = 0.7,
    minimum_weight: float = 0.05,
    variance_floor: float = 0.02,
) -> RelayLoss:
    """Relay global ordinal knowledge with pathology-compatible soft targets."""
    if not 0.0 <= teacher_fraction <= 1.0:
        raise ValueError("teacher_fraction must lie in [0, 1]")
    if not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("minimum_weight must lie in [0, 1]")
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive")

    student_original = torch.as_tensor(student_original_logits).flatten()
    tensors = [
        torch.as_tensor(
            value,
            device=student_original.device,
            dtype=student_original.dtype,
        ).flatten()
        for value in (
            student_augmented_logits,
            teacher_original_logits,
            teacher_augmented_logits,
        )
    ]
    if any(value.shape != student_original.shape for value in tensors):
        raise ValueError("all relay risk logits must share one batch shape")
    student_augmented, teacher_original, teacher_augmented = tensors
    labels = torch.as_tensor(labels, device=student_original.device, dtype=torch.long).flatten()
    if labels.shape != student_original.shape:
        raise ValueError("relay labels must align with risk logits")

    with torch.no_grad():
        teacher_original_probability = torch.sigmoid(teacher_original)
        teacher_augmented_probability = torch.sigmoid(teacher_augmented)
        teacher_probability = 0.5 * (
            teacher_original_probability + teacher_augmented_probability
        )
        prior = bridge.priors(labels, dtype=student_original.dtype)
        target = teacher_fraction * teacher_probability + (1.0 - teacher_fraction) * prior

        variances = bridge.variances.to(
            device=labels.device, dtype=student_original.dtype
        )[labels]
        compatibility = torch.exp(
            -0.5
            * (teacher_probability - prior).square()
            / (variances + variance_floor)
        )
        view_agreement = 1.0 - (
            teacher_original_probability - teacher_augmented_probability
        ).abs()
        weights = (compatibility * view_agreement).clamp(0.0, 1.0)
        accepted = weights >= minimum_weight

    per_sample = 0.5 * (
        F.binary_cross_entropy_with_logits(
            student_original, target, reduction="none"
        )
        + F.binary_cross_entropy_with_logits(
            student_augmented, target, reduction="none"
        )
    )
    effective = weights * accepted.to(weights.dtype)
    denominator = effective.sum()
    if not torch.any(accepted) or float(denominator) <= 0.0:
        loss = student_original.sum() * 0.0
    else:
        loss = (per_sample * effective).sum() / denominator
    return RelayLoss(
        loss=loss,
        mean_weight=weights.mean(),
        accepted_fraction=accepted.to(weights.dtype).mean(),
    )


def ranking_relay_loss(
    student_original: torch.Tensor,
    student_augmented: torch.Tensor,
    teacher_original: torch.Tensor,
    teacher_augmented: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Order-distillation relay: transmit the teacher's ranking, not values.

    The v3 value-based relay was pre-registered as ineffective; the shuffle
    control showed ordinal ORDER carries the signal (+1.05pt). Each sample's
    reference is the mean of its two views; teacher ordering defines pairwise
    hinge targets for the student on the relayed (unannotated) subset.
    """
    student = torch.as_tensor(student_original).flatten()
    teacher = torch.as_tensor(
        teacher_original, device=student.device, dtype=student.dtype
    ).flatten()
    student_augmented = torch.as_tensor(
        student_augmented, device=student.device, dtype=student.dtype
    ).flatten()
    teacher_augmented = torch.as_tensor(
        teacher_augmented, device=student.device, dtype=student.dtype
    ).flatten()
    shapes = {
        value.shape
        for value in (student, teacher, student_augmented, teacher_augmented)
    }
    if len(shapes) != 1:
        raise ValueError("ranking relay tensors must share one batch shape")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    student_reference = 0.5 * (student + student_augmented)
    with torch.no_grad():
        teacher_reference = 0.5 * (teacher + teacher_augmented)
    count = student_reference.numel()
    if count < 2:
        return student.new_zeros(())
    later, earlier = torch.tril_indices(count, count, offset=-1)
    order = teacher_reference[later] - teacher_reference[earlier]
    ordered = order != 0
    if not torch.any(ordered):
        return student.new_zeros(())
    direction = torch.sign(order)
    violations = torch.relu(
        margin
        - direction
        * (
            student_reference[later]
            - student_reference[earlier]
        )
    )[ordered]
    return violations.mean()


def head_coupling_loss(
    binary_logits: torch.Tensor,
    risk_logits: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Couple the binary malignancy head to the ordinal risk ranking.

    Gives ordinal supervision a direct path into the binary decision
    boundary: within a batch, samples ordered higher in ordinal risk must
    not receive lower malignancy probability.
    """
    if binary_logits.ndim != 2 or binary_logits.shape[1] != 2:
        raise ValueError("head coupling requires two-class logits")
    probability = torch.softmax(
        torch.as_tensor(binary_logits), dim=1
    )[:, 1].flatten()
    risk = torch.as_tensor(
        risk_logits, device=probability.device, dtype=probability.dtype
    ).flatten()
    if probability.shape != risk.shape:
        raise ValueError("head coupling tensors must share one batch shape")
    count = probability.numel()
    if count < 2:
        return probability.new_zeros(())
    later, earlier = torch.tril_indices(count, count, offset=-1)
    order = risk[later] - risk[earlier]
    ordered = order != 0
    if not torch.any(ordered):
        return probability.new_zeros(())
    direction = torch.sign(order)
    violations = torch.relu(
        margin
        - direction * (probability[later] - probability[earlier])
    )[ordered]
    return violations.mean()
