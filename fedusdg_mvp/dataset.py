from __future__ import annotations

from functools import partial
import os
from pathlib import Path
import random
import re
import uuid
from typing import Mapping

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .preprocessing import (
    apply_train_geometry,
    apply_train_geometry_with_lesion,
    compute_source_normalization,
    letterbox_binary_mask,
    letterbox_gray,
)


def _letterbox_cache_root() -> Path | None:
    """Content-addressed letterbox cache location; falsy values disable it."""
    value = os.environ.get("FEDUSDG_LETTERBOX_CACHE", "").strip().lower()
    if value in {"0", "off", "false", "none", ""}:
        default = Path.home() / ".cache" / "fedusdg_letterbox"
        return default if value == "" else None
    return Path(value)


def _stable_seed(seed: int, domain_index: int, worker_id: int = 0) -> int:
    return int((seed + 1_000_003 * domain_index + 97 * worker_id) % (2**32))


def _seed_worker(worker_id: int, *, seed: int, domain_index: int) -> None:
    worker_seed = _stable_seed(seed, domain_index, worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_info = get_worker_info()
    if worker_info is not None:
        worker_info.dataset._generator.manual_seed(worker_seed)


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        input_size: int,
        train: bool,
        seed: int,
    ):
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.input_size = input_size
        self.train = train
        self.seed = seed
        self._generator = torch.Generator().manual_seed(seed)
        self._cache_root = _letterbox_cache_root()

    def _letterboxed(self, row) -> tuple[torch.Tensor, torch.Tensor]:
        """Letterbox an image, with a bit-exact content-addressed cache.

        The cache stores the uint8 letterbox canvas keyed by the manifest
        ``sha256`` and input size, so the float tensors reconstructed from it
        are identical to a direct ``letterbox_gray`` call. Cache misses decode
        through the original path and write atomically for concurrent readers.
        """
        digest = str(row.sha256) if "sha256" in self.frame.columns else ""
        cache_path = None
        if self._cache_root is not None and re.fullmatch(r"[0-9a-f]{64}", digest):
            cache_path = (
                self._cache_root
                / "v2_uncompressed"
                / digest[:2]
                / f"{digest}_{self.input_size}.npz"
            )
            if cache_path.exists():
                with np.load(cache_path) as cached:
                    raw = torch.from_numpy(cached["raw"].copy()).float() / 255.0
                    mask = torch.from_numpy(cached["mask"].copy()).bool()
                return raw, mask
        with Image.open(self.data_root / row.image_path) as image:
            raw, mask = letterbox_gray(image, self.input_size)
        if cache_path is not None:
            payload = {
                "raw": (raw * 255.0).round().to(torch.uint8).numpy(),
                "mask": mask.to(torch.uint8).numpy(),
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary.open("wb") as handle:
                    np.savez(handle, **payload)
                os.replace(temporary, cache_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return raw, mask

    def _lesion_mask(self, row) -> tuple[torch.Tensor, bool]:
        lesion_mask = torch.zeros(
            (1, self.input_size, self.input_size), dtype=torch.bool
        )
        if "mask_path" not in self.frame.columns:
            return lesion_mask, False
        value = row.mask_path
        if pd.isna(value) or str(value).strip() == "":
            return lesion_mask, False

        digest = (
            str(row.mask_sha256)
            if "mask_sha256" in self.frame.columns
            else ""
        )
        cache_path = None
        if self._cache_root is not None and re.fullmatch(r"[0-9a-f]{64}", digest):
            cache_path = (
                self._cache_root
                / "lesion_v1"
                / digest[:2]
                / f"{digest}_{self.input_size}.npy"
            )
            if cache_path.exists():
                return torch.from_numpy(np.load(cache_path).copy()).bool(), True

        mask_path = Path(str(value))
        if not mask_path.is_absolute():
            mask_path = self.data_root / mask_path
        image_path = Path(str(row.image_path))
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        with Image.open(image_path) as source_image, Image.open(
            mask_path
        ) as source_mask:
            if source_image.size != source_mask.size:
                raise ValueError(
                    "lesion mask size does not match image: "
                    f"{mask_path} vs {image_path}"
                )
            lesion_mask = letterbox_binary_mask(source_mask, self.input_size)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary.open("wb") as handle:
                    np.save(handle, lesion_mask.to(torch.uint8).numpy())
                os.replace(temporary, cache_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return lesion_mask, True

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        raw, mask = self._letterboxed(row)
        lesion_mask, lesion_mask_available = self._lesion_mask(row)
        if self.train:
            if lesion_mask_available:
                raw, mask, lesion_mask = apply_train_geometry_with_lesion(
                    raw, mask, lesion_mask, self._generator
                )
            else:
                raw, mask = apply_train_geometry(raw, mask, self._generator)
        sample = {
            "raw_image": raw,
            "content_mask": mask,
            "lesion_mask": lesion_mask.float(),
            "lesion_mask_available": torch.tensor(lesion_mask_available),
            "label": torch.tensor(int(row.label)),
            "dataset": row.dataset,
            "patient_id": row.patient_id,
            "image_path": row.image_path,
        }
        if "birads_ordinal" in self.frame.columns:
            value = row.birads_ordinal
            missing = pd.isna(value) or str(value).strip() == ""
            sample["birads_ordinal"] = torch.tensor(
                float("nan") if missing else float(value),
                dtype=torch.float64,
            )
        return sample

    def __len__(self) -> int:
        return len(self.frame)


def _make_loader(
    frame: pd.DataFrame,
    data_root: Path,
    input_size: int,
    train: bool,
    batch_size: int,
    num_workers: int,
    seed: int,
    domain_index: int,
) -> DataLoader:
    dataset_seed = _stable_seed(seed, domain_index)
    dataset = ManifestImageDataset(
        frame=frame,
        data_root=data_root,
        input_size=input_size,
        train=train,
        seed=dataset_seed,
    )
    loader_generator = torch.Generator().manual_seed(dataset_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        worker_init_fn=partial(
            _seed_worker, seed=seed, domain_index=domain_index
        ),
        generator=loader_generator,
        persistent_workers=num_workers > 0,
    )


def build_domain_loaders(
    frames: Mapping[str, pd.DataFrame],
    data_root: str | Path,
    input_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 0,
) -> dict[str, object]:
    """Build per-source loaders, one target loader, and source-train stats."""
    source_train_frame = frames["source_train"]
    source_val_frame = frames["source_val"]
    target_frame = frames["target_test"]
    source_domains = sorted(source_train_frame["dataset"].unique())
    domain_indices = {domain: index for index, domain in enumerate(source_domains)}
    data_root = Path(data_root)

    stats_dataset = ManifestImageDataset(
        source_train_frame,
        data_root,
        input_size=input_size,
        train=False,
        seed=seed,
    )
    source_normalization = compute_source_normalization(
        (sample["raw_image"], sample["content_mask"]) for sample in stats_dataset
    )

    source_train = {
        domain: _make_loader(
            source_train_frame[source_train_frame["dataset"] == domain],
            data_root,
            input_size,
            True,
            batch_size,
            num_workers,
            seed,
            domain_indices[domain],
        )
        for domain in source_domains
    }
    source_val = {
        domain: _make_loader(
            source_val_frame[source_val_frame["dataset"] == domain],
            data_root,
            input_size,
            False,
            batch_size,
            num_workers,
            seed,
            domain_indices[domain],
        )
        for domain in source_domains
    }
    target_index = len(source_domains)
    target_test = _make_loader(
        target_frame,
        data_root,
        input_size,
        False,
        batch_size,
        num_workers,
        seed,
        target_index,
    )
    return {
        "source_train": source_train,
        "source_val": source_val,
        "target_test": target_test,
        "source_normalization": source_normalization,
    }
