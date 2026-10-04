#!/usr/bin/env python3
"""Run a data-free forward and conditional-monotonicity demonstration."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchvision.models import resnet18
from fedusdg_mvp.evidence import SourceNormalizedEvidenceResNet18
from fedusdg_mvp.preprocessing import SourceNormalization


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = SourceNormalizedEvidenceResNet18(
        resnet18(weights=None), SourceNormalization(mean=0.5, std=0.2),
        monotonic_ordinal=True,
    ).eval()
    image = torch.rand(2, 1, 256, 256)
    valid_region = torch.ones_like(image, dtype=torch.bool)
    with torch.no_grad():
        evidence = model.forward_with_evidence(image, valid_region)
        low = model.classify_from_evidence(evidence.lesion_features, torch.zeros(2))
        high = model.classify_from_evidence(evidence.lesion_features, torch.ones(2))
        assert torch.all(high[:, 1] - high[:, 0] > low[:, 1] - low[:, 0])
    print("Synthetic forward pass and fixed-feature monotonicity: PASS")
    print("Binary logits:", tuple(evidence.binary_logits.shape))
    print("Lesion logits:", tuple(evidence.lesion_logits.shape))
    print("No medical data, pretrained downloads or clinical inference were used.")


if __name__ == "__main__":
    main()
