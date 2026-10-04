from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedevi_dg.config import ExperimentConfig
from fedevi_dg.evidence import EvidenceOutputs, supervised_lesion_loss
from fedevi_dg.trainer import (
    FederatedTrainer,
    _ClientPayload,
    _task_aware_weighted_average,
)


def test_supervised_lesion_loss_ignores_unannotated_samples():
    logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
    masks = torch.zeros_like(logits)
    masks[0, :, 1:3, 1:3] = 1
    loss = supervised_lesion_loss(logits, masks, torch.tensor([True, False]))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0]) > 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_evidence_classifier_rejects_invalid_interventions():
    from torchvision.models import resnet18

    from fedevi_dg.evidence import SourceNormalizedEvidenceResNet18
    from fedevi_dg.preprocessing import SourceNormalization

    model = SourceNormalizedEvidenceResNet18(
        resnet18(weights=None), SourceNormalization(mean=0.5, std=0.2)
    )
    features = torch.zeros(2, 512)
    logits = model.classify_from_evidence(features, torch.tensor([0.0, 1.0]))
    assert logits.shape == (2, 2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.classify_from_evidence(features, torch.tensor([-0.1, 1.0]))


def test_monotonic_evidence_classifier_cannot_reverse_clinical_risk():
    from torchvision.models import resnet18

    from fedevi_dg.evidence import SourceNormalizedEvidenceResNet18
    from fedevi_dg.preprocessing import SourceNormalization

    model = SourceNormalizedEvidenceResNet18(
        resnet18(weights=None),
        SourceNormalization(mean=0.5, std=0.2),
        monotonic_ordinal=True,
    ).eval()
    features = torch.randn(7, 512)
    low = torch.softmax(model.classify_from_evidence(features, torch.zeros(7)), dim=1)[
        :, 1
    ]
    high = torch.softmax(model.classify_from_evidence(features, torch.ones(7)), dim=1)[
        :, 1
    ]
    assert torch.all(high > low)


def test_monotonic_evidence_model_rejects_non_bottleneck_configuration():
    from torchvision.models import resnet18

    from fedevi_dg.evidence import SourceNormalizedEvidenceResNet18
    from fedevi_dg.preprocessing import SourceNormalization

    with pytest.raises(ValueError, match="requires the bottleneck"):
        SourceNormalizedEvidenceResNet18(
            resnet18(weights=None),
            SourceNormalization(mean=0.5, std=0.2),
            use_evidence_bottleneck=False,
            monotonic_ordinal=True,
        )


def test_task_aware_aggregation_uses_only_relevant_annotated_clients():
    first = _ClientPayload(
        state_dict={
            "body.weight": torch.tensor([2.0]),
            "ordinal_head.weight": torch.tensor([8.0]),
            "lesion_head.weight": torch.tensor([5.0]),
        },
        num_samples=2,
        mean_loss=0.0,
        ordinal_supervision_samples=2,
        lesion_supervision_samples=0,
    )
    second = _ClientPayload(
        state_dict={
            "body.weight": torch.tensor([4.0]),
            "ordinal_head.weight": torch.tensor([-8.0]),
            "lesion_head.weight": torch.tensor([9.0]),
        },
        num_samples=2,
        mean_loss=0.0,
        ordinal_supervision_samples=0,
        lesion_supervision_samples=2,
    )
    averaged = _task_aware_weighted_average([first, second])
    assert averaged["body.weight"].item() == pytest.approx(3.0)
    assert averaged["ordinal_head.weight"].item() == pytest.approx(8.0)
    assert averaged["lesion_head.weight"].item() == pytest.approx(9.0)


def test_segmentation_only_aggregation_ignores_ordinal_availability():
    masked = _ClientPayload(
        state_dict={
            "body.weight": torch.tensor([2.0]),
            "ordinal_head.weight": torch.tensor([8.0]),
            "lesion_head.weight": torch.tensor([5.0]),
        },
        num_samples=2,
        mean_loss=0.0,
        ordinal_supervision_samples=0,
        lesion_supervision_samples=2,
    )
    unmasked = _ClientPayload(
        state_dict={
            "body.weight": torch.tensor([4.0]),
            "ordinal_head.weight": torch.tensor([-8.0]),
            "lesion_head.weight": torch.tensor([9.0]),
        },
        num_samples=6,
        mean_loss=0.0,
        ordinal_supervision_samples=0,
        lesion_supervision_samples=0,
    )

    averaged = _task_aware_weighted_average(
        [masked, unmasked],
        aggregate_clinical=False,
        aggregate_lesion=True,
    )

    assert averaged["body.weight"].item() == pytest.approx(3.5)
    assert averaged["ordinal_head.weight"].item() == pytest.approx(-4.0)
    assert averaged["lesion_head.weight"].item() == pytest.approx(5.0)


class EvidenceDataset(Dataset):
    def __init__(self, domain: str, mask_available: bool):
        self.samples = []
        for index, label in enumerate((0, 1, 0, 1)):
            lesion = torch.zeros(1, 4, 4)
            lesion[:, 1:3, 1:3] = float(mask_available)
            self.samples.append(
                {
                    "raw_image": torch.full((1, 4, 4), 0.2 + 0.5 * label),
                    "content_mask": torch.ones(1, 4, 4, dtype=torch.bool),
                    "lesion_mask": lesion,
                    "lesion_mask_available": torch.tensor(mask_available),
                    "label": torch.tensor(label),
                    "birads_ordinal": torch.tensor(
                        2.0 + 3.0 * label, dtype=torch.float64
                    ),
                    "dataset": domain,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class TinyEvidenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Conv2d(1, 4, kernel_size=1)
        self.lesion_head = nn.Conv2d(4, 1, kernel_size=1)
        self.ordinal_head = nn.Linear(4, 1)
        self.binary_head = nn.Linear(5, 2)

    def forward_with_evidence(self, image, content_mask=None):
        features = torch.relu(self.body(image))
        lesion_logits = self.lesion_head(features)
        weights = torch.sigmoid(lesion_logits) + 1e-4
        pooled = (features * weights).flatten(2).sum(2) / weights.flatten(2).sum(2)
        ordinal = self.ordinal_head(pooled).squeeze(1)
        binary = self.binary_head(
            torch.cat([pooled, torch.sigmoid(ordinal).unsqueeze(1)], dim=1)
        )
        return EvidenceOutputs(binary, ordinal, lesion_logits, pooled)

    def forward(self, image, content_mask=None):
        return self.forward_with_evidence(image, content_mask).binary_logits


def test_fedevi_trainer_runs_with_heterogeneous_evidence(tmp_path):
    domains = ("MASKED", "UNMASKED", "TARGET")
    config = ExperimentConfig(
        data_root=tmp_path,
        metadata_csv=Path(tmp_path / "metadata.csv"),
        target_domain="TARGET",
        domains=domains,
        method="fedevi_dg",
        rounds=1,
        eval_interval=1,
        input_size=4,
        batch_size=2,
        lr=1e-2,
        weight_decay=0.0,
    )
    source_loaders = {
        domain: {
            "train": DataLoader(
                EvidenceDataset(domain, mask_available=domain == "MASKED"),
                batch_size=2,
                shuffle=False,
            ),
            "val": DataLoader(
                EvidenceDataset(domain, mask_available=domain == "MASKED"),
                batch_size=2,
                shuffle=False,
            ),
        }
        for domain in domains[:-1]
    }
    target = DataLoader(
        EvidenceDataset("TARGET", mask_available=True),
        batch_size=2,
        shuffle=False,
    )
    trainer = FederatedTrainer(
        TinyEvidenceNet(),
        "fedevi_dg",
        source_loaders,
        target,
        config,
        {},
        torch.device("cpu"),
    )
    result = trainer.train()
    assert result.target_evaluation_count == 1
    assert result.method_metadata["diagnostic_evidence_bottleneck"] is True
    assert result.training_history[0]["client_lesion_supervision_samples"] == {
        "MASKED": 4,
        "UNMASKED": 0,
    }
