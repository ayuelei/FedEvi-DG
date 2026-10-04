from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from fedusdg_mvp.dataset import ManifestImageDataset, build_domain_loaders
from fedusdg_mvp.preprocessing import (
    apply_train_geometry,
    compute_source_normalization,
    letterbox_gray,
    normalize_to_three_channels,
)


def test_letterbox_preserves_aspect_and_content():
    image = Image.fromarray(np.full((100, 200), 128, np.uint8), mode="L")
    tensor, mask = letterbox_gray(image, 256)
    assert tensor.shape == mask.shape == (1, 256, 256)
    ys, xs = torch.where(mask[0])
    assert (int(xs.min()), int(xs.max())) == (0, 255)
    assert 126 <= int(ys.max() - ys.min() + 1) <= 129
    assert torch.all(tensor[~mask] == 0)


def test_letterbox_converts_rgb_to_single_channel():
    pixels = np.zeros((2, 4, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    tensor, mask = letterbox_gray(Image.fromarray(pixels, mode="RGB"), 8)
    assert tensor.shape == mask.shape == (1, 8, 8)
    assert torch.unique(tensor[mask]).numel() == 1


def test_normalization_ignores_padding():
    image = torch.zeros(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    image[:, 1:3, 1:3] = 0.5
    mask[:, 1:3, 1:3] = True
    stats = compute_source_normalization([(image, mask)])
    assert stats.mean == 0.5 and stats.std == 1e-6
    normalized = normalize_to_three_channels(image, stats, content_mask=mask)
    assert normalized.shape == (3, 4, 4)
    assert torch.all(normalized[:, ~mask[0]] == 0)


def test_train_geometry_is_reproducible_and_keeps_image_aligned_with_mask():
    image = torch.zeros(1, 32, 32)
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    image[:, 8:24, 11:21] = 1
    mask[:, 8:24, 11:21] = True
    first = apply_train_geometry(
        image, mask, generator=torch.Generator().manual_seed(17)
    )
    second = apply_train_geometry(
        image, mask, generator=torch.Generator().manual_seed(17)
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    _, peak_y, peak_x = np.unravel_index(
        first[0].argmax().item(), first[0].shape
    )
    assert first[1][0, peak_y, peak_x]


def test_train_geometry_rezeros_bilinear_halo_outside_nearest_content_mask():
    image = torch.zeros(1, 48, 48)
    mask = torch.zeros(1, 48, 48, dtype=torch.bool)
    image[:, 9:39, 12:36] = 0.75
    mask[:, 9:39, 12:36] = True

    transformed_image, transformed_mask = apply_train_geometry(
        image, mask, generator=torch.Generator().manual_seed(91)
    )

    assert torch.all(transformed_image[~transformed_mask] == 0)
    assert torch.all((transformed_image != 0) <= transformed_mask)


def _write_pattern(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:18, :30]
    pixels = ((xx * 9 + yy * 5 + offset) % 256).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _frame(records):
    return pd.DataFrame.from_records(records)


def _record(domain: str, filename: str, label: int, patient: int):
    return {
        "dataset": domain,
        "image_path": filename,
        "label": label,
        "patient_id": f"{domain}:p{patient}",
    }


def test_manifest_dataset_returns_raw_image_mask_and_identity(tmp_path):
    _write_pattern(tmp_path / "sample.png", 7)
    frame = _frame([_record("BUSBRA", "sample.png", 1, 3)])
    sample = ManifestImageDataset(frame, tmp_path, input_size=32, train=False, seed=9)[
        0
    ]
    assert sample["raw_image"].shape == sample["content_mask"].shape == (
        1,
        32,
        32,
    )
    assert sample["content_mask"].dtype == torch.bool
    assert sample["label"].item() == 1
    assert sample["dataset"] == "BUSBRA"
    assert sample["patient_id"] == "BUSBRA:p3"
    assert sample["image_path"] == "sample.png"
    assert sample["lesion_mask_available"].item() is False
    assert torch.count_nonzero(sample["lesion_mask"]) == 0


def test_manifest_dataset_loads_aligned_optional_lesion_mask(tmp_path):
    image = np.zeros((18, 30), dtype=np.uint8)
    image[5:14, 9:21] = 180
    lesion = np.zeros_like(image)
    lesion[5:14, 9:21] = 255
    Image.fromarray(image, mode="L").save(tmp_path / "sample.png")
    Image.fromarray(lesion, mode="L").save(tmp_path / "sample_mask.png")
    record = _record("BUSBRA", "sample.png", 1, 3)
    record["mask_path"] = "sample_mask.png"
    sample = ManifestImageDataset(
        _frame([record]), tmp_path, input_size=32, train=False, seed=9
    )[0]
    assert sample["lesion_mask_available"].item() is True
    assert sample["lesion_mask"].shape == sample["raw_image"].shape
    assert torch.all(sample["lesion_mask"].bool() <= sample["content_mask"])
    assert torch.count_nonzero(sample["lesion_mask"]) > 0


def test_build_domain_loaders_separates_domains_and_uses_only_source_train_stats(
    tmp_path,
):
    values = {
        "a.png": 64,
        "b.png": 192,
        "val_a.png": 0,
        "val_b.png": 255,
        "target.png": 255,
    }
    for filename, value in values.items():
        Image.new("L", (8, 8), value).save(tmp_path / filename)
    frames = {
        "source_train": _frame(
            [
                _record("BUSBRA", "a.png", 0, 0),
                _record("GDPH", "b.png", 1, 1),
            ]
        ),
        "source_val": _frame(
            [
                _record("BUSBRA", "val_a.png", 0, 2),
                _record("GDPH", "val_b.png", 1, 3),
            ]
        ),
        "target_test": _frame([_record("UDIAT", "target.png", 1, 4)]),
    }
    built = build_domain_loaders(
        frames, tmp_path, input_size=16, batch_size=2, num_workers=0, seed=23
    )
    assert set(built) == {
        "source_train",
        "source_val",
        "target_test",
        "source_normalization",
    }
    assert set(built["source_train"]) == {"BUSBRA", "GDPH"}
    assert set(built["source_val"]) == {"BUSBRA", "GDPH"}
    assert built["target_test"].dataset.train is False
    stats = built["source_normalization"]
    assert abs(stats.mean - (128 / 255)) < 1e-7
    assert abs(stats.std - (64 / 255)) < 1e-7


def _collect_epoch(loader):
    observed = {}
    for batch in loader:
        for index, image_path in enumerate(batch["image_path"]):
            observed[image_path] = batch["raw_image"][index].clone()
    return observed


def test_seeded_train_loader_repeats_runs_but_changes_augmentation_across_epochs(
    tmp_path,
):
    records = []
    for index in range(6):
        filename = f"train-{index}.png"
        _write_pattern(tmp_path / filename, index * 13)
        records.append(_record("BUSBRA", filename, index % 2, index))
    empty = pd.DataFrame(
        columns=["dataset", "image_path", "label", "patient_id"]
    )
    frames = {
        "source_train": _frame(records),
        "source_val": empty,
        "target_test": _frame([_record("UDIAT", "train-0.png", 0, 99)]),
    }
    first = build_domain_loaders(
        frames, tmp_path, input_size=32, batch_size=2, num_workers=0, seed=41
    )["source_train"]["BUSBRA"]
    second = build_domain_loaders(
        frames, tmp_path, input_size=32, batch_size=2, num_workers=0, seed=41
    )["source_train"]["BUSBRA"]

    first_epoch_a, first_epoch_b = _collect_epoch(first), _collect_epoch(second)
    second_epoch_a, second_epoch_b = _collect_epoch(first), _collect_epoch(second)
    assert all(
        torch.equal(first_epoch_a[path], first_epoch_b[path]) for path in first_epoch_a
    )
    assert all(
        torch.equal(second_epoch_a[path], second_epoch_b[path])
        for path in second_epoch_a
    )
    assert any(
        not torch.equal(first_epoch_a[path], second_epoch_a[path])
        for path in first_epoch_a
    )
