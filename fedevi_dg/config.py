from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

METHODS = (
    "fedavg",
    "fedavg_ord",
    "fedevi_dg",
    "fedevi_mtl",
    "fedevi_seg_only",
    "fedevi_shuffle",
    "fedevi_mask_shift",
    "fedevi_no_taskagg",
    "fedevi_monotonic",
)
ALL_METHODS = METHODS
MethodName = Literal[
    "fedavg",
    "fedavg_ord",
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
    eval_interval: int = 5
    head_coupling_weight: float = 0.25
    birads_domains: tuple[str, ...] | None = None
    ordinal_supervised_weight: float = 0.25
    ordinal_pairwise_weight: float = 0.25
    ordinal_shuffle_seed: int = 20260913
    lesion_weight: float = 0.5

    def __post_init__(self):
        if self.target_domain not in self.domains:
            raise ValueError(f"target_domain must be one of {self.domains}")
        if self.method not in ALL_METHODS:
            raise ValueError(f"method must be one of {ALL_METHODS}")
        if not 0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be in (0, 0.5)")
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
        if self.ordinal_shuffle_seed < 0:
            raise ValueError("ordinal_shuffle_seed must be non-negative")
        if self.lesion_weight < 0.0:
            raise ValueError("lesion_weight must be non-negative")

    def resolved(self) -> dict:
        value = asdict(self)
        value["data_root"] = str(self.data_root.resolve())
        value["metadata_csv"] = str(self.metadata_csv.resolve())
        return value
