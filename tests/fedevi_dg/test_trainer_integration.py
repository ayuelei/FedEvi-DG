from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    Sampler,
    SequentialSampler,
    Subset,
    SubsetRandomSampler,
    TensorDataset,
    default_collate,
)

import fedevi_dg.trainer as trainer_module
from fedevi_dg.config import ExperimentConfig
from fedevi_dg.trainer import FederatedTrainer


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU())
        self.fc = nn.Linear(8, 2)

    def forward(self, image):
        return self.fc(self.body(image))


class BufferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.register_buffer("floating_seen", torch.tensor(0.0))
        self.register_buffer("integer_steps", torch.tensor(0, dtype=torch.long))

    def forward(self, image):
        if self.training:
            self.floating_seen.add_(image.detach().sum())
            self.integer_steps.add_(1)
        score = self.scale * image.flatten(1).mean(dim=1)
        return torch.stack((-score, score), dim=1)


class DictDataset(Dataset):
    def __init__(
        self,
        domain: str,
        offset: float,
        labels: tuple[int, ...] = (0, 1, 0, 1),
    ):
        self.samples = []
        for index, label in enumerate(labels):
            image = torch.full((1, 4, 4), 0.1 + 0.6 * label + offset)
            self.samples.append(
                {
                    "raw_image": image,
                    "content_mask": torch.ones_like(image, dtype=torch.bool),
                    "label": torch.tensor(label),
                    "dataset": domain,
                    "patient_id": f"{domain}-{index}",
                    "image_path": f"{domain}-{index}.png",
                }
            )

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class CountingDataset(DictDataset):
    def __init__(self, domain: str, offset: float):
        super().__init__(domain, offset)
        self.calls = 0

    def __getitem__(self, index):
        self.calls += 1
        return super().__getitem__(index)


class FailOnceDataset(DictDataset):
    def __init__(self, domain: str, offset: float):
        super().__init__(domain, offset)
        self.failed = False

    def __getitem__(self, index):
        if not self.failed:
            self.failed = True
            raise RuntimeError("target boom")
        return super().__getitem__(index)


class OpaqueDataset(Dataset):
    def __init__(self, domain: str):
        self.domain = domain

    def __len__(self):
        return 2

    def __getitem__(self, index):
        label = index
        image = torch.full((1, 4, 4), 0.2 + 0.5 * label)
        return {
            "raw_image": image,
            "content_mask": torch.ones_like(image, dtype=torch.bool),
            "label": torch.tensor(label),
            "dataset": self.domain,
        }


class ProbabilityNet(nn.Module):
    def forward(self, image):
        probability = image[:, 0, 0, 0]
        return torch.stack((torch.log1p(-probability), torch.log(probability)), dim=1)


@pytest.fixture
def synthetic_setup(tmp_path):
    domains = ("SOURCE_A", "SOURCE_B", "TARGET")
    config = ExperimentConfig(
        data_root=tmp_path,
        metadata_csv=Path(tmp_path / "metadata.csv"),
        target_domain="TARGET",
        domains=domains,
        rounds=2,
        local_epochs=1,
        eval_interval=1,
        input_size=4,
        batch_size=2,
        lr=1e-2,
        weight_decay=0.0,
        seed=31,
    )
    source_loaders = {}
    for index, domain in enumerate(domains[:-1]):
        source_loaders[domain] = {
            "train": DataLoader(
                DictDataset(domain, 0.02 * index),
                batch_size=2,
                shuffle=True,
                generator=torch.Generator().manual_seed(100 + index),
            ),
            "val": DataLoader(
                DictDataset(domain, 0.02 * index), batch_size=2, shuffle=False
            ),
        }
    target_loader = DataLoader(DictDataset("TARGET", 0.04), batch_size=2, shuffle=False)
    return config, source_loaders, target_loader


def _seeded_model(seed: int) -> TinyNet:
    torch.manual_seed(seed)
    return TinyNet()


def test_one_global_checkpoint_and_one_target_evaluation(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    result = trainer.train()

    assert result.best_round in {1, 2}
    assert result.target_evaluation_count == 1
    assert result.selected_checkpoint_kind == "global"
    assert result.selection_domains == ("SOURCE_A", "SOURCE_B")
    assert result.target_domain not in result.selection_domains
    assert [entry["round"] for entry in result.training_history] == [1, 2]
    assert (
        sum(bool(entry.get("selected_checkpoint")) for entry in result.training_history)
        == 1
    )
    assert (
        next(
            entry["round"]
            for entry in result.training_history
            if entry.get("selected_checkpoint")
        )
        == result.best_round
    )
    assert all(
        set(entry["client_train_loss"]) == {"SOURCE_A", "SOURCE_B"}
        for entry in result.training_history
    )
    assert all(
        tensor.device.type == "cpu" for tensor in result.best_state_dict.values()
    )
    with pytest.raises(RuntimeError, match="^target already evaluated$"):
        trainer.evaluate_target_once(result.source_threshold)


def test_trainer_preserves_reproducible_caller_initialization(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup
    first_model = _seeded_model(config.seed)
    second_model = _seeded_model(config.seed)
    caller_state = {
        name: value.clone() for name, value in first_model.state_dict().items()
    }

    first = FederatedTrainer(
        first_model,
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )
    second = FederatedTrainer(
        second_model,
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    for name, expected in caller_state.items():
        assert torch.equal(first.initial_state_dict[name], expected)
        assert torch.equal(
            first.initial_state_dict[name], second.initial_state_dict[name]
        )


def test_floating_state_uses_train_sample_weights_and_integer_buffers_are_safe(
    synthetic_setup,
):
    config, _, target_loader = synthetic_setup
    config = replace(config, rounds=1)
    source_a = DictDataset("SOURCE_A", 0.0)
    source_b = DictDataset("SOURCE_B", 0.02)
    source_loaders = {
        "SOURCE_A": {
            "train": DataLoader(Subset(source_a, [0, 1]), batch_size=2),
            "val": DataLoader(DictDataset("SOURCE_A", 0.0), batch_size=2),
        },
        "SOURCE_B": {
            "train": DataLoader(source_b, batch_size=2),
            "val": DataLoader(DictDataset("SOURCE_B", 0.02), batch_size=2),
        },
    }
    trainer = FederatedTrainer(
        BufferNet(),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    result = trainer.train()

    # A sees 12.8 intensity units over 2 samples; B sees 26.88 over 4.
    expected = (2 * 12.8 + 4 * 26.88) / 6
    assert result.best_state_dict["floating_seen"].item() == pytest.approx(expected)
    assert result.best_state_dict["integer_steps"].dtype == torch.long
    assert result.best_state_dict["integer_steps"].item() in {1, 2}


def test_target_evaluation_is_unavailable_before_training(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    assert trainer.target_evaluation_count == 0
    with pytest.raises(RuntimeError, match="target evaluation is not ready"):
        trainer.evaluate_target_once(0.5)
    with pytest.raises(AttributeError):
        trainer.target_evaluation_count = 0


def test_failed_target_evaluation_can_retry_only_with_frozen_threshold(
    synthetic_setup,
):
    config, source_loaders, _ = synthetic_setup
    target_loader = DataLoader(
        FailOnceDataset("TARGET", 0.04), batch_size=2, shuffle=False
    )
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="target boom"):
        trainer.train()
    assert trainer.target_evaluation_count == 0
    frozen_threshold = trainer.derive_source_threshold()
    wrong_threshold = (
        frozen_threshold + 0.01 if frozen_threshold < 1.0 else frozen_threshold - 0.01
    )
    with pytest.raises(ValueError, match="frozen source threshold"):
        trainer.evaluate_target_once(wrong_threshold)
    assert trainer.target_evaluation_count == 0

    target_metrics = trainer.evaluate_target_once(frozen_threshold)

    assert target_metrics["n"] == 4
    assert trainer.target_evaluation_count == 1
    with pytest.raises(AttributeError):
        trainer.target_evaluation_count = 0
    with pytest.raises(RuntimeError, match="^target already evaluated$"):
        trainer.evaluate_target_once(frozen_threshold)


def test_source_evaluation_path_rejects_target_loader(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="target loader"):
        trainer._evaluate_source_loader(target_loader, "SOURCE_A", 0.5)


def test_source_evaluation_rejects_wrapped_target_dataset_without_batch_domain(
    synthetic_setup,
):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    def collate_without_domain(samples):
        batch = default_collate(samples)
        del batch["dataset"]
        return batch

    wrapped_target = DataLoader(
        Subset(target_loader.dataset, [0, 1]),
        batch_size=2,
        collate_fn=collate_without_domain,
    )

    with pytest.raises(ValueError, match="target loader"):
        trainer._evaluate_source_loader(wrapped_target, "SOURCE_A", 0.5)


@pytest.mark.parametrize("alias_kind", ["loader", "dataset", "wrapped_dataset"])
def test_target_loader_and_dataset_must_not_alias_a_source(synthetic_setup, alias_kind):
    config, source_loaders, _ = synthetic_setup
    source_train = source_loaders["SOURCE_A"]["train"]
    if alias_kind == "loader":
        target_loader = source_train
    elif alias_kind == "dataset":
        target_loader = DataLoader(source_train.dataset, batch_size=2)
    else:
        target_loader = DataLoader(Subset(source_train.dataset, [0, 1]), batch_size=2)

    with pytest.raises(ValueError, match="target.*alias"):
        FederatedTrainer(
            _seeded_model(config.seed),
            "fedavg",
            source_loaders,
            target_loader,
            config,
            {},
            torch.device("cpu"),
        )


@pytest.mark.parametrize("alias_kind", ["loader", "dataset"])
def test_source_train_and_val_must_not_alias(synthetic_setup, alias_kind):
    config, source_loaders, target_loader = synthetic_setup
    train_loader = source_loaders["SOURCE_A"]["train"]
    source_loaders["SOURCE_A"]["val"] = (
        train_loader
        if alias_kind == "loader"
        else DataLoader(train_loader.dataset, batch_size=2)
    )

    with pytest.raises(ValueError, match="source loader alias"):
        FederatedTrainer(
            _seeded_model(config.seed),
            "fedavg",
            source_loaders,
            target_loader,
            config,
            {},
            torch.device("cpu"),
        )


@pytest.mark.parametrize("bad_keys", [("SOURCE_A",), ("SOURCE_A", "EXTRA")])
def test_source_keys_must_exactly_match_configured_leave_one_out_domains(
    synthetic_setup, bad_keys
):
    config, source_loaders, target_loader = synthetic_setup
    bad_loaders = {
        domain: source_loaders.get(
            domain,
            {
                "train": DataLoader(DictDataset(domain, 0.0), batch_size=2),
                "val": DataLoader(DictDataset(domain, 0.0), batch_size=2),
            },
        )
        for domain in bad_keys
    }

    with pytest.raises(ValueError, match="source loader domains"):
        FederatedTrainer(
            _seeded_model(config.seed),
            "fedavg",
            bad_loaders,
            target_loader,
            config,
            {},
            torch.device("cpu"),
        )


def test_dataset_domain_metadata_must_match_mapping_key(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup
    source_loaders["SOURCE_A"]["train"] = DataLoader(
        DictDataset("SOURCE_B", 0.0), batch_size=2
    )

    with pytest.raises(ValueError, match="dataset domain.*SOURCE_A"):
        FederatedTrainer(
            _seeded_model(config.seed),
            "fedavg",
            source_loaders,
            target_loader,
            config,
            {},
            torch.device("cpu"),
        )


def test_method_must_match_config(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup

    with pytest.raises(ValueError, match="method must match config.method"):
        FederatedTrainer(
            _seeded_model(config.seed),
            "fedevi_monotonic",
            source_loaders,
            target_loader,
            config,
            {},
            torch.device("cpu"),
        )


def test_subset_and_sampler_determine_class_weights_and_sample_count(
    synthetic_setup,
):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )
    base = DictDataset("SOURCE_A", 0.0)
    subset_loader = DataLoader(Subset(base, [0, 1, 3]), batch_size=3)
    sampler_loader = DataLoader(
        base,
        batch_size=2,
        sampler=SubsetRandomSampler([0, 1]),
    )

    subset_labels, subset_count = trainer._training_labels_and_count(subset_loader)
    sampler_labels, sampler_count = trainer._training_labels_and_count(sampler_loader)

    assert sorted(subset_labels) == [0, 1, 1]
    assert subset_count == 3
    torch.testing.assert_close(
        trainer._class_weights(subset_loader), torch.tensor([1.5, 0.75])
    )
    assert sorted(sampler_labels) == [0, 1]
    assert sampler_count == 2


def test_tensor_dataset_labels_are_supported(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )
    dataset = TensorDataset(torch.zeros(3, 1, 4, 4), torch.tensor([0, 1, 1]))

    labels, count = trainer._training_labels_and_count(
        DataLoader(dataset, batch_size=3)
    )

    assert labels == [0, 1, 1]
    assert count == 3


def test_custom_batch_sampler_is_rejected_even_with_standard_inner_sampler(
    synthetic_setup,
):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )
    dataset = DictDataset("SOURCE_A", 0.0)

    class CustomBatchSampler(BatchSampler):
        pass

    loader = DataLoader(
        dataset,
        batch_sampler=CustomBatchSampler(
            SequentialSampler(dataset), batch_size=2, drop_last=False
        ),
    )

    with pytest.raises(ValueError, match="standard BatchSampler"):
        trainer._training_labels_and_count(loader)


def test_unknown_sampler_cannot_bypass_validation_with_indices_attribute(
    synthetic_setup,
):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )
    dataset = DictDataset("SOURCE_A", 0.0)

    class FakeIndicesSampler(Sampler):
        indices = [0, 1]

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=FakeIndicesSampler(),
    )

    with pytest.raises(ValueError, match="unsupported training sampler"):
        trainer._training_labels_and_count(loader)


def test_unsupported_training_dataset_fails_instead_of_using_neutral_weights(
    synthetic_setup,
):
    config, source_loaders, target_loader = synthetic_setup
    source_loaders["SOURCE_A"]["train"] = DataLoader(
        OpaqueDataset("SOURCE_A"), batch_size=2
    )
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="cannot determine training labels"):
        trainer.train()


def test_aggregation_uses_actual_sampler_count(synthetic_setup):
    config, _, target_loader = synthetic_setup
    config = replace(config, rounds=1)
    source_a = DictDataset("SOURCE_A", 0.0)
    source_b = DictDataset("SOURCE_B", 0.02)
    source_loaders = {
        "SOURCE_A": {
            "train": DataLoader(
                source_a,
                batch_size=2,
                sampler=SubsetRandomSampler([0, 1]),
            ),
            "val": DataLoader(DictDataset("SOURCE_A", 0.0), batch_size=2),
        },
        "SOURCE_B": {
            "train": DataLoader(source_b, batch_size=2),
            "val": DataLoader(DictDataset("SOURCE_B", 0.02), batch_size=2),
        },
    }
    trainer = FederatedTrainer(
        BufferNet(),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    result = trainer.train()

    expected = (2 * 12.8 + 4 * 26.88) / 6
    assert result.best_state_dict["floating_seen"].item() == pytest.approx(expected)


def test_evaluation_loss_is_sample_weighted_across_unequal_batches(
    synthetic_setup,
):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        ProbabilityNet(),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )
    dataset = DictDataset("SOURCE_A", 0.0, labels=(0, 1, 1))
    probabilities = (0.2, 0.8, 0.6)
    for sample, probability in zip(dataset.samples, probabilities, strict=True):
        sample["raw_image"].fill_(probability)

    metrics = trainer._evaluate_source_loader(
        DataLoader(dataset, batch_size=2), "SOURCE_A", 0.5
    )

    expected_loss = -sum(math.log(0.8) for _ in range(2)) / 3
    expected_loss -= math.log(0.6) / 3
    assert metrics["loss"] == pytest.approx(expected_loss)


def test_best_checkpoint_is_loaded_before_threshold_derivation(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup

    class ObservingTrainer(FederatedTrainer):
        state_at_threshold = None

        def derive_source_threshold(self):
            self.state_at_threshold = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.global_model.state_dict().items()
            }
            return super().derive_source_threshold()

    trainer = ObservingTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )

    result = trainer.train()

    for name, expected in result.best_state_dict.items():
        assert torch.equal(trainer.state_at_threshold[name], expected)


def test_training_result_is_deep_copied_from_trainer_state(synthetic_setup):
    config, source_loaders, target_loader = synthetic_setup
    trainer = FederatedTrainer(
        _seeded_model(config.seed),
        "fedavg",
        source_loaders,
        target_loader,
        config,
        {},
        torch.device("cpu"),
    )
    result = trainer.train()
    expected_threshold = trainer.derive_source_threshold()
    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in trainer.global_model.state_dict().items()
    }

    first_name = next(iter(result.best_state_dict))
    result.best_state_dict[first_name].zero_()
    result.source_val_metrics["SOURCE_A"]["labels"].append(999)

    assert torch.equal(
        trainer.global_model.state_dict()[first_name], model_state[first_name]
    )
    assert trainer.derive_source_threshold() == expected_threshold


@pytest.mark.parametrize("source_count", [0, 1])
def test_federated_trainer_requires_at_least_two_sources(tmp_path, source_count):
    source_domains = tuple(f"SOURCE_{index}" for index in range(source_count))
    config = ExperimentConfig(
        data_root=tmp_path,
        metadata_csv=tmp_path / "metadata.csv",
        target_domain="TARGET",
        domains=source_domains + ("TARGET",),
        rounds=1,
        eval_interval=1,
        input_size=4,
        batch_size=2,
    )
    source_loaders = {
        domain: {
            "train": DataLoader(DictDataset(domain, 0.0), batch_size=2),
            "val": DataLoader(DictDataset(domain, 0.0), batch_size=2),
        }
        for domain in source_domains
    }

    with pytest.raises(ValueError, match="at least two source domains"):
        FederatedTrainer(
            _seeded_model(config.seed),
            "fedavg",
            source_loaders,
            DataLoader(DictDataset("TARGET", 0.0), batch_size=2),
            config,
            {},
            torch.device("cpu"),
        )
