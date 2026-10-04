"""FedFAT baseline (Wang et al., Pattern Recognition 2026) adapted to the
strict five-domain LODO protocol.

Implements the published algorithm literally (Eqs. 3-12 and Algorithm 1);
interpretation choices for underspecified steps are pre-registered in the
2026-09-04 design document, section 11, before any run.

Per image: the full-spectrum amplitude is decomposed with ``fft2``/``fftshift``;
a donor amplitude map is drawn from the shared library of source-client
amplitude estimates; the fusion ratio and insertion-region size adapt to the
mean-squared deviation between donor and local amplitude (Eqs. 4-6); the
insertion mask keeps supra-threshold coefficients or defaults to the central
box (Eq. 7); amplitudes fuse per Eq. 8, the client's exponential moving
average of fused amplitudes is updated (Eq. 9) and used for reconstruction
with the local phase (Eq. 10).
"""

from __future__ import annotations

import os

import torch



class FedFATSpectrum:
    """Full-spectrum FFT helpers with an fftshift-centred layout."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size

    def _as_spatial(self, image: torch.Tensor) -> tuple[torch.Tensor, bool]:
        tensor = torch.as_tensor(image)
        squeezed = tensor.ndim == 3 and tensor.shape[0] == 1
        if squeezed:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError("image must have shape (H, W) or (1, H, W)")
        return tensor, squeezed

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tensor, _ = self._as_spatial(image)
        if tuple(tensor.shape) != (self.size, self.size):
            raise ValueError(f"image must be {self.size}x{self.size}")
        mask_tensor, _ = self._as_spatial(mask)
        if tensor.shape != mask_tensor.shape:
            raise ValueError("image and mask must share the shape")
        masked = torch.where(
            mask_tensor.to(torch.bool),
            tensor.to(torch.float64),
            torch.zeros((), dtype=torch.float64, device=tensor.device),
        )
        return torch.fft.fftshift(torch.fft.fft2(masked))

    def inverse(self, spectrum: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_tensor, _ = self._as_spatial(mask)
        spatial = torch.fft.ifft2(torch.fft.ifftshift(spectrum)).real
        clamped = spatial.clamp(0.0, 1.0)
        return torch.where(
            mask_tensor.to(torch.bool), clamped, torch.zeros_like(clamped)
        )


class FedFATAugmenter:
    """Per-client FedFAT augmentation state (library slot + amplitude EMA)."""

    def __init__(
        self,
        size: int = 256,
        lambda_default: float = 0.5,
        region_default: int = 32,
        ema_decay: float = 0.9,
        adaptive_threshold: float | None = None,
    ) -> None:
        if not 0.0 <= lambda_default <= 1.0:
            raise ValueError("lambda_default must be in [0, 1]")
        if not 0 < region_default <= size:
            raise ValueError("region_default must be in (0, size]")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("ema_decay must be in [0, 1]")
        if adaptive_threshold is not None and adaptive_threshold < 0:
            raise ValueError("adaptive_threshold must be non-negative")
        self.size = size
        self.lambda_default = lambda_default
        self.region_default = region_default
        self.ema_decay = ema_decay
        self.adaptive_threshold = adaptive_threshold
        self.ema_amplitude: torch.Tensor | None = None
        self._spectrum = FedFATSpectrum(size)

    def client_amplitude(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Mean amplitude spectrum over a client's images (library seed)."""
        amplitudes = [
            self._spectrum.forward(image, mask).abs()
            for image, mask in zip(images, masks, strict=True)
        ]
        return torch.stack(amplitudes).mean(dim=0)

    def augment_batch(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        donor_amplitude: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse donor amplitude into the batch and reconstruct with the EMA.

        Follows the paper's ordering: per-image fusion (Eq. 8), EMA update
        from the batch mean of fused amplitudes (Eq. 9), reconstruction with
        the EMA amplitude and each image's local phase (Eq. 10).
        """
        fused: list[torch.Tensor] = []
        phases: list[torch.Tensor] = []
        keep_channel: list[bool] = []
        for image, mask in zip(images, masks, strict=True):
            spectrum = self._spectrum.forward(image, mask)
            local_amplitude = spectrum.abs()
            phase = torch.angle(spectrum)
            _, squeezed = self._spectrum._as_spatial(image)
            keep_channel.append(squeezed)

            lam, region = self.lambda_default, self.region_default
            if self.adaptive_threshold is not None:
                deviation = torch.mean((donor_amplitude - local_amplitude) ** 2)
                if bool(deviation > self.adaptive_threshold):
                    lam = self.lambda_default / 2
                    region = max(1, self.region_default // 2)

            insertion = self._insertion_mask(local_amplitude, region)
            if os.environ.get("FEDUSDG_FEDFAT_SWAP_MASK", "").strip() in {"1", "true", "yes"}:
                fused.append(
                    (1.0 - lam) * local_amplitude * (~insertion)
                    + lam * donor_amplitude * insertion
                )
            else:
                fused.append(
                    (1.0 - lam) * donor_amplitude * (~insertion)
                    + lam * local_amplitude * insertion
                )
            phases.append(phase)

        batch_mean = torch.stack(fused).mean(dim=0)
        if self.ema_amplitude is None:
            self.ema_amplitude = batch_mean.clone()
        else:
            self.ema_amplitude = (1.0 - self.ema_decay) * self.ema_amplitude + (
                self.ema_decay
            ) * batch_mean

        reconstructed = []
        for mask, phase, restore_channel in zip(
            masks, phases, keep_channel, strict=True
        ):
            content = self._spectrum.inverse(
                torch.polar(self.ema_amplitude, phase), mask
            )
            if restore_channel:
                content = content.unsqueeze(0)
            reconstructed.append(content)
        stacked = torch.stack(reconstructed)
        if stacked.dtype != images.dtype:
            stacked = stacked.to(images.dtype)
        return stacked

    def _insertion_mask(self, amplitude: torch.Tensor, region: int) -> torch.Tensor:
        """Supra-threshold bounding box, or the central-region default."""
        supra = amplitude > 0.5 * amplitude.max()
        if bool(supra.any()):
            rows, columns = torch.where(supra)
            mask = torch.zeros_like(supra)
            mask[
                int(rows.min()) : int(rows.max()) + 1,
                int(columns.min()) : int(columns.max()) + 1,
            ] = True
            return mask
        mask = torch.zeros(
            (self.size, self.size), dtype=torch.bool, device=amplitude.device
        )
        center = self.size // 2
        half = max(1, region // 2)
        mask[center - half : center + half, center - half : center + half] = True
        return mask
