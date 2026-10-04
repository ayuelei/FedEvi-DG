import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedusdg_mvp.config import ExperimentConfig
from fedusdg_mvp.ordinal import (
    apply_level_map,
    build_shuffled_level_maps,
    categorical_birads_loss,
    estimate_ordinal_bridge,
    ordinal_relay_loss,
    supervised_ordinal_loss,
)
from fedusdg_mvp.spectral import DomainSignature, SpectralTransfer
from fedusdg_mvp.trainer import (
    FederatedMVPTrainer,
    _ClientPayload,
    _task_aware_weighted_average,
)


def _bridge():
    return estimate_ordinal_bridge(
        {
            "A": (
                torch.tensor([0, 0, 1, 1]),
                torch.tensor([1.0, 2.0, 3.0, 4.0]),
            ),
            "B": (
                torch.tensor([0, 0, 1, 1]),
                torch.tensor([2.0, 3.0, 5.0, 6.0]),
            ),
        }
    )


def test_bridge_normalizes_each_client_before_equal_weight_aggregation():
    bridge = _bridge()

    assert bridge.means[0].item() == pytest.approx((1 / 6 + 1 / 8) / 2)
    assert bridge.means[1].item() == pytest.approx((5 / 6 + 7 / 8) / 2)
    assert bridge.client_counts.tolist() == [2, 2]
    assert bridge.normalize("A", torch.tensor([1.0, 4.0])).tolist() == [0.0, 1.0]


def test_bridge_rejects_missing_pathology_class():
    with pytest.raises(ValueError, match="class 1"):
        estimate_ordinal_bridge(
            {"A": (torch.tensor([0, 0]), torch.tensor([1.0, 2.0]))}
        )


def test_supervised_ordinal_loss_ignores_missing_and_rewards_correct_order():
    levels = torch.tensor([0.0, 0.5, 1.0, float("nan")])
    ordered = supervised_ordinal_loss(
        torch.tensor([-4.0, 0.0, 4.0, 9.0]), None, levels
    )
    reversed_ = supervised_ordinal_loss(
        torch.tensor([4.0, 0.0, -4.0, -9.0]), None, levels
    )

    assert ordered < reversed_


def test_relay_target_is_detached_and_updates_only_student():
    bridge = _bridge()
    student = torch.tensor([0.0, 0.0], requires_grad=True)
    student_augmented = torch.tensor([0.0, 0.0], requires_grad=True)
    teacher = torch.tensor([-1.0, 1.0], requires_grad=True)
    teacher_augmented = torch.tensor([-1.1, 1.1], requires_grad=True)

    result = ordinal_relay_loss(
        student,
        student_augmented,
        teacher,
        teacher_augmented,
        torch.tensor([0, 1]),
        bridge,
    )
    result.loss.backward()

    assert student.grad is not None
    assert student_augmented.grad is not None
    assert teacher.grad is None
    assert teacher_augmented.grad is None
    assert 0.0 < result.mean_weight.item() <= 1.0
    assert result.accepted_fraction.item() == 1.0


def test_relay_rejects_invalid_teacher_fraction():
    bridge = _bridge()
    values = torch.zeros(2)
    with pytest.raises(ValueError, match="teacher_fraction"):
        ordinal_relay_loss(
            values,
            values,
            values,
            values,
            torch.tensor([0, 1]),
            bridge,
            teacher_fraction=1.1,
        )


def test_shuffled_levels_are_deterministic_nonidentity_and_preserve_values():
    records = {
        "A": (
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
    }
    first = build_shuffled_level_maps(records, seed=17)
    second = build_shuffled_level_maps(records, seed=17)
    transformed = apply_level_map(records["A"][1], first["A"])

    assert first == second
    assert all(source != destination for source, destination in first["A"].items())
    assert sorted(transformed.tolist()) == [1.0, 2.0, 3.0, 4.0]
    assert not torch.equal(transformed, records["A"][1])


def test_categorical_control_uses_nearest_shared_bin_and_ignores_missing():
    levels = torch.tensor([0.0, 0.51, 1.0, float("nan")])
    logits = torch.full((4, 6), -4.0)
    logits[0, 0] = 4.0
    logits[1, 3] = 4.0
    logits[2, 5] = 4.0

    good = categorical_birads_loss(logits, None, levels)
    bad = categorical_birads_loss(logits.flip(1), None, levels)

    assert good < bad


def test_task_aware_aggregation_excludes_unannotated_ordinal_heads():
    annotated = _ClientPayload(
        {"body.weight": torch.tensor([2.0]), "ordinal_head.weight": torch.tensor([8.0])},
        num_samples=2,
        mean_loss=0.0,
        ordinal_supervision_samples=2,
    )
    unannotated = _ClientPayload(
        {"body.weight": torch.tensor([4.0]), "ordinal_head.weight": torch.tensor([-8.0])},
        num_samples=6,
        mean_loss=0.0,
        ordinal_supervision_samples=0,
    )

    averaged = _task_aware_weighted_average([annotated, unannotated])

    assert averaged["body.weight"].item() == pytest.approx(3.5)
    assert averaged["ordinal_head.weight"].item() == pytest.approx(8.0)


def test_task_aware_aggregation_excludes_unannotated_categorical_heads():
    annotated = _ClientPayload(
        {
            "body.weight": torch.tensor([2.0]),
            "categorical_head.weight": torch.tensor([8.0]),
        },
        num_samples=2,
        mean_loss=0.0,
        ordinal_supervision_samples=2,
    )
    unannotated = _ClientPayload(
        {
            "body.weight": torch.tensor([4.0]),
            "categorical_head.weight": torch.tensor([-8.0]),
        },
        num_samples=6,
        mean_loss=0.0,
        ordinal_supervision_samples=0,
    )

    averaged = _task_aware_weighted_average([annotated, unannotated])

    assert averaged["body.weight"].item() == pytest.approx(3.5)
    assert averaged["categorical_head.weight"].item() == pytest.approx(8.0)


class _OrdinalDataset(Dataset):
    def __init__(self, domain: str, annotated: bool):
        self.samples = []
        for index, (label, level) in enumerate(
            zip((0, 0, 1, 1), (1.0, 2.0, 3.0, 4.0), strict=True)
        ):
            image = torch.full((1, 4, 4), 0.1 + 0.7 * label)
            self.samples.append(
                {
                    "raw_image": image,
                    "content_mask": torch.ones_like(image, dtype=torch.bool),
                    "label": torch.tensor(label),
                    "birads_ordinal": torch.tensor(
                        level if annotated else float("nan"), dtype=torch.float64
                    ),
                    "dataset": domain,
                    "patient_id": f"{domain}-{index}",
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _OrdinalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU())
        self.binary_head = nn.Linear(8, 2)
        self.ordinal_head = nn.Linear(8, 1)

    def forward_with_ordinal(self, image, content_mask=None):
        features = self.body(image)
        return self.binary_head(features), self.ordinal_head(features).squeeze(1)

    def forward(self, image):
        return self.forward_with_ordinal(image)[0]


class _CategoricalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU())
        self.binary_head = nn.Linear(8, 2)
        self.categorical_head = nn.Linear(8, 6)

    def forward_with_categorical(self, image, content_mask=None):
        features = self.body(image)
        return self.binary_head(features), self.categorical_head(features)

    def forward(self, image):
        return self.forward_with_categorical(image)[0]


class _IdentityTransfer(SpectralTransfer):
    def apply(self, raw_image, content_mask, *args, **kwargs):
        return raw_image.clone()


@pytest.mark.parametrize("method", ["fedord_dg", "fedord_shuffle"])
def test_fedord_trainer_relays_to_client_without_birads(tmp_path, method):
    domains = ("ANNOTATED", "BINARY_ONLY", "TARGET")
    config = ExperimentConfig(
        data_root=tmp_path,
        metadata_csv=tmp_path / "metadata.csv",
        target_domain="TARGET",
        domains=domains,
        method=method,
        rounds=1,
        eval_interval=1,
        input_size=4,
        num_frequency_bands=2,
        batch_size=2,
        lr=1e-2,
        weight_decay=0.0,
        ordinal_relay_warmup_rounds=0,
    )
    source_loaders = {
        domain: {
            "train": DataLoader(
                _OrdinalDataset(domain, annotated), batch_size=2, shuffle=False
            ),
            "val": DataLoader(
                _OrdinalDataset(domain, annotated), batch_size=2, shuffle=False
            ),
        }
        for domain, annotated in (("ANNOTATED", True), ("BINARY_ONLY", False))
    }
    target_loader = DataLoader(
        _OrdinalDataset("TARGET", False), batch_size=2, shuffle=False
    )
    signatures = {
        domain: DomainSignature(
            mean=torch.zeros(3, 2, dtype=torch.float64),
            std=torch.ones(3, 2, dtype=torch.float64),
            class_total_weights={0: 0.5, 1: 0.5},
            clipped_std_cells=0,
        )
        for domain in domains[:-1]
    }

    trainer = FederatedMVPTrainer(
        _OrdinalNet(),
        method,
        source_loaders,
        target_loader,
        config,
        signatures,
        torch.device("cpu"),
        transfer=_IdentityTransfer(3, 2),
    )
    result = trainer.train()

    history = result.training_history[0]
    assert history["client_ordinal_supervision_samples"] == {
        "ANNOTATED": 4,
        "BINARY_ONLY": 0,
    }
    assert history["client_relay_loss"]["BINARY_ONLY"] > 0.0
    assert history["client_relay_weight"]["BINARY_ONLY"] > 0.0
    assert result.method_metadata["ordinal_domains"] == ["ANNOTATED"]
    assert "ordinal_head.weight" in result.best_state_dict
    if method == "fedord_shuffle":
        assert result.method_metadata["ordinal_level_maps"]["ANNOTATED"]


def test_categorical_control_trains_only_on_annotated_client(tmp_path):
    domains = ("ANNOTATED", "BINARY_ONLY", "TARGET")
    config = ExperimentConfig(
        data_root=tmp_path,
        metadata_csv=tmp_path / "metadata.csv",
        target_domain="TARGET",
        domains=domains,
        method="control_cat",
        rounds=1,
        eval_interval=1,
        input_size=4,
        num_frequency_bands=2,
        batch_size=2,
        lr=1e-2,
        weight_decay=0.0,
    )
    source_loaders = {
        domain: {
            "train": DataLoader(
                _OrdinalDataset(domain, annotated), batch_size=2, shuffle=False
            ),
            "val": DataLoader(
                _OrdinalDataset(domain, annotated), batch_size=2, shuffle=False
            ),
        }
        for domain, annotated in (("ANNOTATED", True), ("BINARY_ONLY", False))
    }
    target_loader = DataLoader(
        _OrdinalDataset("TARGET", False), batch_size=2, shuffle=False
    )
    signatures = {
        domain: DomainSignature(
            mean=torch.zeros(3, 2, dtype=torch.float64),
            std=torch.ones(3, 2, dtype=torch.float64),
            class_total_weights={0: 0.5, 1: 0.5},
            clipped_std_cells=0,
        )
        for domain in domains[:-1]
    }

    result = FederatedMVPTrainer(
        _CategoricalNet(),
        "control_cat",
        source_loaders,
        target_loader,
        config,
        signatures,
        torch.device("cpu"),
        transfer=_IdentityTransfer(3, 2),
    ).train()

    history = result.training_history[0]
    assert history["client_ordinal_supervision_samples"] == {
        "ANNOTATED": 4,
        "BINARY_ONLY": 0,
    }
    assert history["client_ordinal_loss"]["ANNOTATED"] > 0.0
    assert history["client_ordinal_loss"]["BINARY_ONLY"] == 0.0
    assert result.method_metadata["clinical_supervision"] == "unordered_categorical"
    assert "categorical_head.weight" in result.best_state_dict
