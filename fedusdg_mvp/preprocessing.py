from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


@dataclass(frozen=True)
class SourceNormalization:
    mean: float
    std: float


def letterbox_gray(
    image: Image.Image, size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an image to gray and letterbox it without distorting anatomy."""
    image = image.convert("L")
    scale = min(size / image.width, size / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.BILINEAR)
    resized_mask = Image.new("L", image.size, 255).resize(
        resized_size, Image.Resampling.NEAREST
    )

    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas = Image.new("L", (size, size), 0)
    mask_canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, (left, top))
    mask_canvas.paste(resized_mask, (left, top))

    tensor = transform_functional.pil_to_tensor(canvas).float() / 255
    mask = transform_functional.pil_to_tensor(mask_canvas).bool()
    return tensor, mask


def letterbox_binary_mask(image: Image.Image, size: int = 256) -> torch.Tensor:
    """Letterbox a lesion mask using nearest-neighbour interpolation."""
    image = image.convert("L")
    scale = min(size / image.width, size / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.NEAREST)
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, (left, top))
    return transform_functional.pil_to_tensor(canvas) > 0


def normalize_to_three_channels(
    image: torch.Tensor,
    stats: SourceNormalization,
    content_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("normalization expects image with shape (1, H, W)")
    normalized = ((image - stats.mean) / stats.std).repeat(3, 1, 1)
    if content_mask is None:
        return normalized
    content_mask = torch.as_tensor(content_mask, device=image.device).bool()
    if content_mask.shape != image.shape:
        raise ValueError("content_mask must have the same shape as image")
    return normalized.masked_fill(~content_mask.expand_as(normalized), 0.0)


def compute_source_normalization(
    samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> SourceNormalization:
    content_values = [image[mask] for image, mask in samples]
    if not content_values:
        raise ValueError("source normalization requires at least one sample")
    values = torch.cat(content_values).double()
    if values.numel() == 0:
        raise ValueError("source normalization requires non-empty content masks")
    return SourceNormalization(
        mean=float(values.mean()),
        std=max(float(values.std(correction=0)), 1e-6),
    )


def _uniform(
    lower: float, upper: float, generator: torch.Generator | None
) -> float:
    return lower + (upper - lower) * float(torch.rand((), generator=generator))


def apply_train_geometry(
    image: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one safe geometry and apply it to image and content mask."""
    if float(torch.rand((), generator=generator)) < 0.5:
        image = transform_functional.hflip(image)
        mask = transform_functional.hflip(mask)

    height, width = image.shape[-2:]
    angle = _uniform(-10.0, 10.0, generator)
    translate = [
        round(_uniform(-0.03, 0.03, generator) * width),
        round(_uniform(-0.03, 0.03, generator) * height),
    ]
    scale = _uniform(0.95, 1.0, generator)
    image = transform_functional.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    transformed_mask = transform_functional.affine(
        mask.float(),
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    ).bool()
    image = image.masked_fill(~transformed_mask, 0.0)
    return image, transformed_mask


def apply_train_geometry_with_lesion(
    image: torch.Tensor,
    content_mask: torch.Tensor,
    lesion_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one sampled geometry to image, content, and lesion masks."""
    if lesion_mask.shape != image.shape or content_mask.shape != image.shape:
        raise ValueError("image, content mask, and lesion mask must align")
    horizontal_flip = float(torch.rand((), generator=generator)) < 0.5
    if horizontal_flip:
        image = transform_functional.hflip(image)
        content_mask = transform_functional.hflip(content_mask)
        lesion_mask = transform_functional.hflip(lesion_mask)

    height, width = image.shape[-2:]
    angle = _uniform(-10.0, 10.0, generator)
    translate = [
        round(_uniform(-0.03, 0.03, generator) * width),
        round(_uniform(-0.03, 0.03, generator) * height),
    ]
    scale = _uniform(0.95, 1.0, generator)
    affine_arguments = {
        "angle": angle,
        "translate": translate,
        "scale": scale,
        "shear": [0.0, 0.0],
        "fill": 0.0,
    }
    image = transform_functional.affine(
        image, interpolation=InterpolationMode.BILINEAR, **affine_arguments
    )
    content_mask = transform_functional.affine(
        content_mask.float(),
        interpolation=InterpolationMode.NEAREST,
        **affine_arguments,
    ).bool()
    lesion_mask = transform_functional.affine(
        lesion_mask.float(),
        interpolation=InterpolationMode.NEAREST,
        **affine_arguments,
    ).bool()
    lesion_mask &= content_mask
    image = image.masked_fill(~content_mask, 0.0)
    return image, content_mask, lesion_mask
