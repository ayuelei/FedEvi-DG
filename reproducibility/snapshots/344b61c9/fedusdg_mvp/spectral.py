from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch


@dataclass(frozen=True)
class DomainSignature:
    mean: torch.Tensor
    std: torch.Tensor
    class_total_weights: dict[int, float]
    clipped_std_cells: int

    def as_vector(self) -> torch.Tensor:
        return torch.cat((self.mean.reshape(-1), self.std.reshape(-1)))


@dataclass(frozen=True)
class SpectralTransferDiagnostics:
    pre_clamp_content_finite: bool
    pre_clamp_content_min: float
    pre_clamp_content_max: float
    pre_clamp_content_shape: tuple[int, int]
    valid_content_pixel_count: int
    pre_clamp_out_of_range_pixel_count: int
    pre_clamp_out_of_range_pixel_fraction: float
    projection_applied: bool


def make_depth_windows(height: int, num_windows: int) -> torch.Tensor:
    if height <= 0:
        raise ValueError("height must be positive")
    if num_windows == 1:
        return torch.ones(1, height)
    if num_windows != 3:
        raise ValueError("relative-depth signatures require one or three windows")

    depth = torch.linspace(0.0, 1.0, height)
    windows = torch.zeros(3, height)

    windows[0, depth <= 0.25] = 1.0
    first_fall = (depth > 0.25) & (depth < 0.5)
    windows[0, first_fall] = 0.5 * (
        1.0 + torch.cos(torch.pi * (depth[first_fall] - 0.25) / 0.25)
    )

    middle_rise = (depth > 0.25) & (depth < 0.5)
    windows[1, middle_rise] = 0.5 * (
        1.0 - torch.cos(torch.pi * (depth[middle_rise] - 0.25) / 0.25)
    )
    windows[1, depth == 0.5] = 1.0
    middle_fall = (depth > 0.5) & (depth < 0.75)
    windows[1, middle_fall] = 0.5 * (
        1.0 + torch.cos(torch.pi * (depth[middle_fall] - 0.5) / 0.25)
    )

    last_rise = (depth > 0.5) & (depth < 0.75)
    windows[2, last_rise] = 0.5 * (
        1.0 - torch.cos(torch.pi * (depth[last_rise] - 0.5) / 0.25)
    )
    windows[2, depth >= 0.75] = 1.0

    return windows / windows.sum(dim=0, keepdim=True)


def make_radial_bands(
    height: int,
    width: int,
    num_bands: int,
    device: torch.device | str,
) -> torch.Tensor:
    if height <= 0 or width <= 0 or num_bands <= 0:
        raise ValueError("height, width, and num_bands must be positive")
    fy = torch.fft.fftfreq(height, device=device)[:, None]
    fx = torch.fft.rfftfreq(width, device=device)[None, :]
    radius = torch.sqrt(fy.square() + fx.square()) / (2**0.5 / 2)
    return torch.clamp(
        (radius * num_bands).floor().long(), max=num_bands - 1
    )


def balanced_sample_weights(labels: Iterable[int]) -> torch.Tensor:
    labels = torch.as_tensor(list(labels), dtype=torch.long)
    if torch.any((labels < 0) | (labels > 1)):
        raise ValueError("domain signatures require binary labels 0 and 1")
    counts = torch.bincount(labels, minlength=2)
    if torch.any(counts == 0):
        raise ValueError("both classes are required for a domain signature")
    return 1.0 / (2.0 * counts[labels].float())


def _as_spatial_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape (H, W) or (1, H, W)")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def compute_domain_signature(
    samples: Iterable[Mapping[str, torch.Tensor]],
    num_windows: int,
    num_bands: int,
) -> DomainSignature:
    sample_list = list(samples)
    labels: list[int] = []
    for sample in sample_list:
        label_tensor = torch.as_tensor(sample["label"])
        if label_tensor.numel() != 1:
            raise ValueError("each label must be a scalar")
        if label_tensor.is_floating_point() and not torch.isfinite(label_tensor):
            raise ValueError("labels must be finite")
        labels.append(int(label_tensor.item()))

    sample_weights = balanced_sample_weights(labels)
    per_sample_means = []
    per_sample_seconds = []
    signature_device: torch.device | None = None

    for sample in sample_list:
        image = _as_spatial_tensor(sample["raw_image"], "raw_image")
        mask = _as_spatial_tensor(sample["content_mask"], "content_mask")
        if image.shape != mask.shape:
            raise ValueError("raw_image and content_mask must have the same shape")
        if signature_device is None:
            signature_device = image.device
        elif image.device != signature_device:
            raise ValueError("all signature images must use the same device")

        mask = mask.to(dtype=torch.bool)
        content_rows, content_columns = torch.where(mask)
        if content_rows.numel() == 0:
            raise ValueError("signature requires a non-empty content mask")
        content_values = image[mask]
        if torch.any((content_values < 0) | (content_values > 1)):
            raise ValueError("raw_image content must be in [0, 1]")

        top, bottom = int(content_rows.min()), int(content_rows.max()) + 1
        left, right = int(content_columns.min()), int(content_columns.max()) + 1
        image = image[top:bottom, left:right]
        mask = mask[top:bottom, left:right]
        height, width = image.shape
        depth_windows = make_depth_windows(height, num_windows).to(
            device=image.device, dtype=torch.float64
        )
        valid_window_support = (
            (depth_windows[:, :, None] > 0) & mask[None, :, :]
        ).flatten(1).any(dim=1)
        unsupported = torch.where(~valid_window_support)[0]
        if unsupported.numel() > 0:
            raise ValueError(
                f"depth window {int(unsupported[0])} has no valid support"
            )
        radial_bands = make_radial_bands(
            height, width, num_bands, image.device
        )
        masked_image = torch.where(
            mask, image.to(torch.float64), torch.zeros((), device=image.device)
        )
        windowed_images = masked_image * depth_windows[:, :, None]
        log_magnitudes = torch.log(
            torch.abs(torch.fft.rfft2(windowed_images)) + 1e-6
        )
        if not torch.isfinite(log_magnitudes).all():
            raise ValueError("spectral signature contains non-finite values")

        image_means = torch.empty(
            (num_windows, num_bands),
            dtype=torch.float64,
            device=image.device,
        )
        image_seconds = torch.empty_like(image_means)
        for band in range(num_bands):
            values = log_magnitudes[:, radial_bands == band]
            if values.shape[1] == 0:
                raise ValueError(f"radial band {band} contains no coefficients")
            image_means[:, band] = values.mean(dim=1)
            image_seconds[:, band] = values.square().mean(dim=1)
        per_sample_means.append(image_means)
        per_sample_seconds.append(image_seconds)

    means = torch.stack(per_sample_means)
    seconds = torch.stack(per_sample_seconds)
    weights = sample_weights.to(device=means.device, dtype=means.dtype)[:, None, None]
    mean = (weights * means).sum(dim=0)
    second_moment = (weights * seconds).sum(dim=0)
    std = torch.sqrt((second_moment - mean.square()).clamp_min(0.0))
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("spectral signature contains non-finite moments")

    clipped = std < 1e-6
    std = std.clamp_min(1e-6)
    class_total_weights = {
        label: float(sample_weights[torch.as_tensor(labels) == label].sum())
        for label in (0, 1)
    }
    return DomainSignature(
        mean=mean,
        std=std,
        class_total_weights=class_total_weights,
        clipped_std_cells=int(clipped.sum()),
    )


def map_log_amplitude(
    log_amplitude: torch.Tensor,
    local_mean: torch.Tensor,
    local_std: torch.Tensor,
    donor_mean: torch.Tensor,
    donor_std: torch.Tensor,
    mix_lambda: float,
) -> torch.Tensor:
    mapped = donor_mean + donor_std * (
        log_amplitude - local_mean
    ) / local_std.clamp_min(1e-6)
    return (1.0 - mix_lambda) * log_amplitude + mix_lambda * mapped


def choose_donor(
    source_domains: Iterable[str],
    local_domain: str,
    target_domain: str,
    generator: torch.Generator,
) -> str:
    source_set = set(source_domains)
    if len(source_set) < 2:
        raise ValueError("at least two source domains are required")
    if local_domain not in source_set:
        raise ValueError("local domain must be a source domain")
    eligible = sorted(source_set - {local_domain, target_domain})
    if not eligible:
        raise ValueError("at least one eligible source donor is required")
    index = int(
        torch.randint(
            len(eligible),
            (),
            generator=generator,
            device=generator.device,
        ).item()
    )
    return eligible[index]


class SpectralTransfer:
    def __init__(self, num_depth_windows: int, num_bands: int) -> None:
        if num_depth_windows not in (1, 3):
            raise ValueError(
                "relative-depth transfer requires one or three windows"
            )
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        self.num_depth_windows = num_depth_windows
        self.num_bands = num_bands

    def _validate_signature(
        self, signature: DomainSignature, name: str
    ) -> None:
        expected_shape = (self.num_depth_windows, self.num_bands)
        if (
            signature.mean.shape != expected_shape
            or signature.std.shape != expected_shape
        ):
            raise ValueError(
                f"{name} signature must have shape {expected_shape}"
            )
        if not torch.isfinite(signature.mean).all() or not torch.isfinite(
            signature.std
        ).all():
            raise ValueError(f"{name} signature contains non-finite values")
        if torch.any(signature.std <= 0):
            raise ValueError(f"{name} signature standard deviations must be positive")

    def validate_bank(
        self,
        signature_bank: Mapping[str, DomainSignature],
        target_domain: str,
    ) -> None:
        if target_domain in signature_bank:
            raise ValueError("target signature must not be present in the bank")
        for domain, signature in signature_bank.items():
            self._validate_signature(signature, f"domain {domain}")

    def apply(
        self,
        raw_image: torch.Tensor,
        content_mask: torch.Tensor,
        local_signature: DomainSignature,
        donor_signature: DomainSignature,
        mix_lambda: float,
    ) -> torch.Tensor:
        output, _ = self._apply(
            raw_image,
            content_mask,
            local_signature,
            donor_signature,
            mix_lambda,
        )
        return output

    def apply_with_diagnostics(
        self,
        raw_image: torch.Tensor,
        content_mask: torch.Tensor,
        local_signature: DomainSignature,
        donor_signature: DomainSignature,
        mix_lambda: float,
    ) -> tuple[torch.Tensor, SpectralTransferDiagnostics]:
        return self._apply(
            raw_image,
            content_mask,
            local_signature,
            donor_signature,
            mix_lambda,
        )

    def _apply(
        self,
        raw_image: torch.Tensor,
        content_mask: torch.Tensor,
        local_signature: DomainSignature,
        donor_signature: DomainSignature,
        mix_lambda: float,
    ) -> tuple[torch.Tensor, SpectralTransferDiagnostics]:
        image_with_channel = torch.as_tensor(raw_image)
        mask_with_channel = torch.as_tensor(content_mask)
        image = _as_spatial_tensor(image_with_channel, "raw_image")
        mask = _as_spatial_tensor(mask_with_channel, "content_mask")
        if image.shape != mask.shape:
            raise ValueError("raw_image and content_mask must have the same shape")
        if not image.is_floating_point():
            raise ValueError("raw_image must be floating point")

        self._validate_signature(local_signature, "local")
        self._validate_signature(donor_signature, "donor")
        mix_lambda = float(mix_lambda)
        if not math.isfinite(mix_lambda) or not 0.0 <= mix_lambda <= 1.0:
            raise ValueError("mix_lambda must be finite and in [0, 1]")

        mask = mask.to(device=image.device, dtype=torch.bool)
        content_rows, content_columns = torch.where(mask)
        if content_rows.numel() == 0:
            raise ValueError("transfer requires a non-empty content mask")
        content_values = image[mask]
        if torch.any((content_values < 0) | (content_values > 1)):
            raise ValueError("raw_image content must be in [0, 1]")

        top, bottom = int(content_rows.min()), int(content_rows.max()) + 1
        left, right = int(content_columns.min()), int(content_columns.max()) + 1
        cropped_image = image[top:bottom, left:right]
        cropped_mask = mask[top:bottom, left:right]
        height, width = cropped_image.shape

        work_dtype = torch.float64
        depth_windows = make_depth_windows(
            height, self.num_depth_windows
        ).to(device=image.device, dtype=work_dtype)
        radial_bands = make_radial_bands(
            height, width, self.num_bands, image.device
        )
        masked_image = torch.where(
            cropped_mask,
            cropped_image.to(work_dtype),
            torch.zeros((), device=image.device, dtype=work_dtype),
        )
        windowed_images = masked_image * depth_windows[:, :, None]
        spectra = torch.fft.rfft2(windowed_images)
        log_amplitudes = torch.log(torch.abs(spectra) + 1e-6)
        mapped_log_amplitudes = log_amplitudes.clone()

        local_mean = local_signature.mean.to(
            device=image.device, dtype=work_dtype
        )
        local_std = local_signature.std.to(
            device=image.device, dtype=work_dtype
        )
        donor_mean = donor_signature.mean.to(
            device=image.device, dtype=work_dtype
        )
        donor_std = donor_signature.std.to(
            device=image.device, dtype=work_dtype
        )
        for window in range(self.num_depth_windows):
            for band in range(self.num_bands):
                coefficients = radial_bands == band
                mapped_log_amplitudes[window, coefficients] = map_log_amplitude(
                    log_amplitudes[window, coefficients],
                    local_mean[window, band],
                    local_std[window, band],
                    donor_mean[window, band],
                    donor_std[window, band],
                    mix_lambda,
                )

        mapped_amplitudes = (
            torch.exp(mapped_log_amplitudes) - 1e-6
        ).clamp_min(0.0)
        mapped_spectra = torch.polar(mapped_amplitudes, torch.angle(spectra))
        reconstructed_windows = torch.fft.irfft2(
            mapped_spectra, s=(height, width)
        )
        normalizer = depth_windows.square().sum(dim=0).clamp_min(1e-6)
        transferred_content = (
            reconstructed_windows * depth_windows[:, :, None]
        ).sum(dim=0) / normalizer[:, None]
        valid_pre_clamp_content = transferred_content[cropped_mask]
        out_of_range = (valid_pre_clamp_content < 0.0) | (
            valid_pre_clamp_content > 1.0
        )
        out_of_range_pixel_count = int(out_of_range.sum().item())
        valid_content_pixel_count = int(valid_pre_clamp_content.numel())
        diagnostics = SpectralTransferDiagnostics(
            pre_clamp_content_finite=bool(
                torch.isfinite(valid_pre_clamp_content).all()
            ),
            pre_clamp_content_min=float(valid_pre_clamp_content.min()),
            pre_clamp_content_max=float(valid_pre_clamp_content.max()),
            pre_clamp_content_shape=(height, width),
            valid_content_pixel_count=valid_content_pixel_count,
            pre_clamp_out_of_range_pixel_count=out_of_range_pixel_count,
            pre_clamp_out_of_range_pixel_fraction=(
                out_of_range_pixel_count / valid_content_pixel_count
            ),
            projection_applied=True,
        )
        projected_content = transferred_content.clamp(0.0, 1.0)

        canvas = image.clone()
        canvas[~mask] = 0
        cropped_canvas = canvas[top:bottom, left:right]
        cropped_canvas[cropped_mask] = projected_content[
            cropped_mask
        ].to(dtype=image.dtype)
        if image_with_channel.ndim == 3:
            return canvas.unsqueeze(0), diagnostics
        return canvas, diagnostics
