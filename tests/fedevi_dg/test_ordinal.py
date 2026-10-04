import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedevi_dg.config import ExperimentConfig
from fedevi_dg.ordinal import (
    apply_level_map,
    build_shuffled_level_maps,
    estimate_ordinal_bridge,
    supervised_ordinal_loss,
)
from fedevi_dg.trainer import (
    FederatedTrainer,
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
        estimate_ordinal_bridge({"A": (torch.tensor([0, 0]), torch.tensor([1.0, 2.0]))})


def test_supervised_ordinal_loss_ignores_missing_and_rewards_correct_order():
    levels = torch.tensor([0.0, 0.5, 1.0, float("nan")])
    ordered = supervised_ordinal_loss(torch.tensor([-4.0, 0.0, 4.0, 9.0]), None, levels)
    reversed_ = supervised_ordinal_loss(
        torch.tensor([4.0, 0.0, -4.0, -9.0]), None, levels
    )

    assert ordered < reversed_


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


def test_task_aware_aggregation_excludes_unannotated_ordinal_heads():
    annotated = _ClientPayload(
        {
            "body.weight": torch.tensor([2.0]),
            "ordinal_head.weight": torch.tensor([8.0]),
        },
        num_samples=2,
        mean_loss=0.0,
        ordinal_supervision_samples=2,
    )
    unannotated = _ClientPayload(
        {
            "body.weight": torch.tensor([4.0]),
            "ordinal_head.weight": torch.tensor([-8.0]),
        },
        num_samples=6,
        mean_loss=0.0,
        ordinal_supervision_samples=0,
    )

    averaged = _task_aware_weighted_average([annotated, unannotated])

    assert averaged["body.weight"].item() == pytest.approx(3.5)
    assert averaged["ordinal_head.weight"].item() == pytest.approx(8.0)
