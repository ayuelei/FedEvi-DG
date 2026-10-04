from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


METHODS = (
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
    "fedbn",
    "fedfat",
)
UPGRADE_METHODS = (
    "fedavg_ord",
    "control_cons",
    "control_ord",
    "control_cat",
    "fedord_dg",
    "fedord_shuffle",
    "fedord_v2",
    "fedevi_dg",
    "fedevi_mtl",
    "fedevi_seg_only",
    "fedevi_shuffle",
    "fedevi_mask_shift",
    "fedevi_no_taskagg",
    "fedevi_monotonic",
)
ALL_METHODS = METHODS + UPGRADE_METHODS
MethodName = Literal[
    "fedavg",
    "fedprox",
    "fedbn",
    "fedfat",
    "whole",
    "depth",
    "fedus_dg_mvp",
    "factor",
    "virtual",
    "fedphys",
    "control",
    "virtual_cons",
    "control_full",
    "fedavg_ord",
    "control_cons",
    "control_ord",
    "control_cat",
    "fedord_dg",
    "fedord_shuffle",
    "fedord_v2",
    "fedevi_dg",
    "fedevi_mtl",
    "fedevi_seg_only",
    "fedevi_shuffle",
    "fedevi_mask_shift",
    "fedevi_no_taskagg",
    "fedevi_monotonic",
]


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    metadata_csv: Path
    target_domain: str = "UDIAT"
    method: MethodName = "fedavg"
    domains: tuple[str, ...] = ("BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT")
    split_seed: int = 20260728
    seed: int = 42
    input_size: int = 256
    val_fraction: float = 0.15
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    fedprox_mu: float = 0.01
    consistency_weight: float = 0.5
    eval_interval: int = 5
    num_frequency_bands: int = 8
    lambda_min: float = 0.1
    lambda_max: float = 0.5
    dirichlet_alpha: float = 1.0
    extrapolation_radius: float = 0.5
    extrapolation_probability: float = 0.3
    birads_weight: float = 0.25
    ordinal_margin: float = 1.0
    fedfat_perturbation: float = 0.05
    head_coupling_weight: float = 0.25
    birads_domains: tuple[str, ...] | None = None
    ordinal_supervised_weight: float = 0.25
    ordinal_pairwise_weight: float = 0.25
    ordinal_relay_weight: float = 0.25
    ordinal_relay_warmup_rounds: int = 5
    ordinal_teacher_fraction: float = 0.7
    ordinal_minimum_relay_weight: float = 0.05
    ordinal_variance_floor: float = 0.02
    ordinal_shuffle_seed: int = 20260913
    lesion_weight: float = 0.5

    def __post_init__(self):
        if self.target_domain not in self.domains:
            raise ValueError(f"target_domain must be one of {self.domains}")
        if self.method not in ALL_METHODS:
            raise ValueError(f"method must be one of {ALL_METHODS}")
        if not 0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be in (0, 0.5)")
        if self.dirichlet_alpha != 1.0:
            raise ValueError("dirichlet_alpha is fixed at 1.0 by protocol")
        if not 0.0 <= self.extrapolation_radius <= 1.0:
            raise ValueError("extrapolation_radius must be in [0, 1]")
        if not 0.0 <= self.extrapolation_probability <= 1.0:
            raise ValueError("extrapolation_probability must be in [0, 1]")
        if self.birads_weight < 0.0:
            raise ValueError("birads_weight must be non-negative")
        if self.birads_domains is not None:
            unknown = set(self.birads_domains) - set(self.domains)
            if unknown:
                raise ValueError(f"unknown birads_domains: {sorted(unknown)}")
            if len(set(self.birads_domains)) != len(self.birads_domains):
                raise ValueError("birads_domains must not contain duplicates")
        if self.ordinal_supervised_weight < 0.0:
            raise ValueError("ordinal_supervised_weight must be non-negative")
        if self.ordinal_pairwise_weight < 0.0:
            raise ValueError("ordinal_pairwise_weight must be non-negative")
        if self.ordinal_relay_weight < 0.0:
            raise ValueError("ordinal_relay_weight must be non-negative")
        if self.ordinal_relay_warmup_rounds < 0:
            raise ValueError("ordinal_relay_warmup_rounds must be non-negative")
        if not 0.0 <= self.ordinal_teacher_fraction <= 1.0:
            raise ValueError("ordinal_teacher_fraction must lie in [0, 1]")
        if not 0.0 <= self.ordinal_minimum_relay_weight <= 1.0:
            raise ValueError("ordinal_minimum_relay_weight must lie in [0, 1]")
        if self.ordinal_variance_floor <= 0.0:
            raise ValueError("ordinal_variance_floor must be positive")
        if self.ordinal_shuffle_seed < 0:
            raise ValueError("ordinal_shuffle_seed must be non-negative")
        if self.lesion_weight < 0.0:
            raise ValueError("lesion_weight must be non-negative")

    def resolved(self) -> dict:
        value = asdict(self)
        value["data_root"] = str(self.data_root.resolve())
        value["metadata_csv"] = str(self.metadata_csv.resolve())
        return value
