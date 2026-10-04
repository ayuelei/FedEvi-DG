from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .preprocessing import SourceNormalization, normalize_to_three_channels


@dataclass(frozen=True)
class EvidenceOutputs:
    """Intrinsic diagnostic evidence returned by the FedEvi-DG model."""

    binary_logits: torch.Tensor
    ordinal_logits: torch.Tensor
    lesion_logits: torch.Tensor
    lesion_features: torch.Tensor


class SourceNormalizedEvidenceResNet18(nn.Module):
    """ResNet18 whose diagnosis is mediated by lesion and ordinal evidence.

    The binary head cannot read the global pooled backbone feature directly.
    It receives only a soft lesion-gated feature and the predicted ordinal risk,
    making the two evidence channels part of the forward computation rather
    than post-hoc visualisations.
    """

    requires_content_mask = True

    def __init__(
        self,
        backbone: nn.Module,
        stats: SourceNormalization,
        *,
        use_evidence_bottleneck: bool = True,
        monotonic_ordinal: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "fc") or not hasattr(backbone.fc, "in_features"):
            raise TypeError("ResNet18 backbone must expose fc.in_features")
        if not torch.isfinite(torch.tensor([stats.mean, stats.std])).all() or stats.std <= 0:
            raise ValueError("source normalization must be finite with positive std")
        feature_count = int(backbone.fc.in_features)
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.stats = stats
        self.use_evidence_bottleneck = bool(use_evidence_bottleneck)
        self.monotonic_ordinal = bool(monotonic_ordinal)
        if self.monotonic_ordinal and not self.use_evidence_bottleneck:
            raise ValueError("monotonic ordinal evidence requires the bottleneck")
        self.lesion_head = nn.Sequential(
            nn.Conv2d(feature_count, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.ordinal_head = nn.Linear(feature_count, 1)
        diagnostic_feature_count = feature_count + int(
            self.use_evidence_bottleneck and not self.monotonic_ordinal
        )
        diagnostic_output_count = 1 if self.monotonic_ordinal else 2
        self.binary_head = nn.Sequential(
            nn.Linear(diagnostic_feature_count, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, diagnostic_output_count),
        )
        if self.monotonic_ordinal:
            # softplus(raw) is strictly positive, so raising clinical risk can
            # never lower the malignant log-odds for fixed lesion evidence.
            self.ordinal_scale_raw = nn.Parameter(torch.tensor(-1.0))

    def _normalized_input(
        self,
        raw_image: torch.Tensor,
        content_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        raw_image = torch.as_tensor(raw_image)
        if raw_image.ndim != 4 or raw_image.shape[1] != 1:
            raise ValueError("model expects raw_image with shape (N, 1, H, W)")
        if content_mask is None:
            masks = [None] * len(raw_image)
        else:
            content_mask = torch.as_tensor(
                content_mask, device=raw_image.device
            ).bool()
            if content_mask.shape != raw_image.shape:
                raise ValueError("content_mask must match raw_image shape")
            masks = content_mask
        return torch.stack(
            [
                normalize_to_three_channels(image, self.stats, content_mask=mask)
                for image, mask in zip(raw_image, masks, strict=True)
            ]
        )

    def _feature_map(self, image: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        image = backbone.conv1(image)
        image = backbone.bn1(image)
        image = backbone.relu(image)
        image = backbone.maxpool(image)
        image = backbone.layer1(image)
        image = backbone.layer2(image)
        image = backbone.layer3(image)
        return backbone.layer4(image)

    def forward_with_evidence(
        self,
        raw_image: torch.Tensor,
        content_mask: torch.Tensor | None = None,
    ) -> EvidenceOutputs:
        normalized = self._normalized_input(raw_image, content_mask)
        feature_map = self._feature_map(normalized)
        low_resolution_lesion_logits = self.lesion_head(feature_map)
        lesion_weights = torch.sigmoid(low_resolution_lesion_logits)
        if content_mask is not None:
            low_resolution_content = F.interpolate(
                torch.as_tensor(
                    content_mask,
                    device=feature_map.device,
                    dtype=feature_map.dtype,
                ),
                size=feature_map.shape[-2:],
                mode="nearest",
            )
            lesion_weights = lesion_weights * low_resolution_content
        # A small floor avoids a dead all-background state early in training.
        lesion_weights = lesion_weights + 1e-4 * (
            low_resolution_content if content_mask is not None else 1.0
        )
        lesion_features = (
            feature_map * lesion_weights
        ).flatten(2).sum(dim=2) / lesion_weights.flatten(2).sum(dim=2)
        global_features = feature_map.flatten(2).mean(dim=2)
        clinical_features = (
            lesion_features if self.use_evidence_bottleneck else global_features
        )
        ordinal_logits = self.ordinal_head(clinical_features).squeeze(1)
        if self.use_evidence_bottleneck:
            binary_logits = self.classify_from_evidence(
                lesion_features, torch.sigmoid(ordinal_logits)
            )
        else:
            binary_logits = self.binary_head(global_features)
        lesion_logits = F.interpolate(
            low_resolution_lesion_logits,
            size=raw_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return EvidenceOutputs(
            binary_logits=binary_logits,
            ordinal_logits=ordinal_logits,
            lesion_logits=lesion_logits,
            lesion_features=lesion_features,
        )

    def classify_from_evidence(
        self,
        lesion_features: torch.Tensor,
        ordinal_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Diagnose from explicit evidence, enabling causal fidelity probes."""
        lesion_features = torch.as_tensor(lesion_features)
        ordinal_probability = torch.as_tensor(
            ordinal_probability,
            device=lesion_features.device,
            dtype=lesion_features.dtype,
        ).flatten()
        if lesion_features.ndim != 2:
            raise ValueError("lesion features must have shape (N, F)")
        if ordinal_probability.shape != lesion_features.shape[:1]:
            raise ValueError("ordinal probability must contain one value per sample")
        if torch.any((ordinal_probability < 0.0) | (ordinal_probability > 1.0)):
            raise ValueError("ordinal probability must lie in [0, 1]")
        if not self.use_evidence_bottleneck:
            raise RuntimeError(
                "ordinary multitask control has no diagnostic evidence intervention"
            )
        if self.monotonic_ordinal:
            lesion_log_odds = self.binary_head(lesion_features).squeeze(1)
            malignant_log_odds = lesion_log_odds + F.softplus(
                self.ordinal_scale_raw
            ) * ordinal_probability
            return torch.stack(
                [torch.zeros_like(malignant_log_odds), malignant_log_odds], dim=1
            )
        diagnostic_evidence = torch.cat(
            [lesion_features, ordinal_probability.unsqueeze(1)], dim=1
        )
        return self.binary_head(diagnostic_evidence)

    def forward_with_ordinal(
        self,
        raw_image: torch.Tensor,
        content_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward_with_evidence(raw_image, content_mask)
        return outputs.binary_logits, outputs.ordinal_logits

    def forward(
        self,
        raw_image: torch.Tensor,
        content_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_evidence(raw_image, content_mask).binary_logits


def supervised_lesion_loss(
    lesion_logits: torch.Tensor,
    lesion_mask: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Balanced BCE + soft Dice on samples with a real lesion mask."""
    logits = torch.as_tensor(lesion_logits)
    target = torch.as_tensor(
        lesion_mask, device=logits.device, dtype=logits.dtype
    )
    available = torch.as_tensor(available, device=logits.device).bool().flatten()
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("lesion logits must have shape (N, 1, H, W)")
    if target.shape != logits.shape:
        raise ValueError("lesion mask must match lesion logits")
    if available.shape != logits.shape[:1]:
        raise ValueError("lesion availability must contain one value per sample")
    if not torch.any(available):
        return logits.sum() * 0.0

    logits = logits[available]
    target = target[available]
    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).flatten(1).sum(dim=1)
    denominator = probabilities.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_loss
