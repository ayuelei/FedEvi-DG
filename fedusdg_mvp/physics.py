"""Physics-structured signature factors and virtual-device synthesis.

Implements the 2026-09-04 method upgrade: each client signature's mean
table is decomposed into interpretable imaging factors (depth gain curve,
radial band-energy profile, monotone attenuation slope), and training-time
donor statistics can be synthesized by Dirichlet interpolation plus a
prior-fixed bounded extrapolation along those factors. A structure-destroyed
control keeps factor magnitudes while permuting their alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .spectral import DomainSignature


@dataclass(frozen=True)
class PhysicalFactors:
    """Additive factorization of a signature mean table M (D windows, B bands):

        M[d, r] ~= gain[d] + band_energy[r] + attenuation[d] * coordinate[r]

    ``attenuation`` carries the per-window slope of the band trend. It is
    estimated without a monotonicity constraint (three-window isotonic
    projections pool to a flat slope; see ``fit_physical_factors``), while
    virtual-device synthesis projects sampled slopes back onto the
    non-increasing cone. Total degrees of freedom: 2D + B.
    """

    gain: torch.Tensor
    band_energy: torch.Tensor
    attenuation: torch.Tensor

    def __post_init__(self) -> None:
        if self.gain.ndim != 1 or self.band_energy.ndim != 1 or self.attenuation.ndim != 1:
            raise ValueError("factor vectors must be one-dimensional")
        if self.gain.shape[0] != self.attenuation.shape[0]:
            raise ValueError("gain and attenuation must share the depth dimension")
        if not all(
            torch.isfinite(tensor).all() for tensor in (self.gain, self.band_energy, self.attenuation)
        ):
            raise ValueError("factors must be finite")


def band_coordinates(num_bands: int) -> torch.Tensor:
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")
    if num_bands == 1:
        return torch.zeros(1, dtype=torch.float64)
    return torch.linspace(-1.0, 1.0, num_bands, dtype=torch.float64)


def isotonic_non_increasing(values: torch.Tensor) -> torch.Tensor:
    """Pool-adjacent-violators projection onto non-increasing sequences."""
    if values.ndim != 1:
        raise ValueError("isotonic projection requires a one-dimensional tensor")
    if not torch.isfinite(values).all():
        raise ValueError("isotonic projection requires finite values")
    pooled: list[tuple[float, float]] = []  # (value, weight)
    for raw in values.to(torch.float64).tolist():
        block = (raw, 1.0)
        while pooled and pooled[-1][0] < block[0]:
            previous = pooled.pop()
            block = (
                (previous[0] * previous[1] + block[0] * block[1]) / (previous[1] + block[1]),
                previous[1] + block[1],
            )
        pooled.append(block)
    projected = torch.tensor([value for value, _ in pooled], dtype=torch.float64)
    return torch.repeat_interleave(
        projected,
        torch.tensor([int(weight) for _, weight in pooled], dtype=torch.long),
    ).to(values.dtype)


def _fit_additive(
    mean_table: torch.Tensor, attenuation: torch.Tensor, coordinates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Additive fit given attenuation slopes.

    The model is rank-deficient by one: a linear trend in ``band_energy``
    is indistinguishable from a uniform shift of ``attenuation``. The fit
    therefore orthogonalizes ``band_energy`` against the coordinates and
    returns the removed linear trend so the caller can shift
    ``attenuation`` accordingly, which makes the decomposition unique.
    """
    residual = mean_table - attenuation[:, None] * coordinates[None, :]
    gain = residual.mean(dim=1)
    band_energy = (residual - gain[:, None]).mean(dim=0)
    band_offset = band_energy.mean()
    gain = gain + band_offset
    band_energy = band_energy - band_offset
    trend = (band_energy * coordinates).sum() / coordinates.square().sum().clamp_min(1e-12)
    band_energy = band_energy - trend * coordinates
    return gain, band_energy, trend


def fit_physical_factors(mean_table: torch.Tensor, passes: int = 2) -> PhysicalFactors:
    """Least-squares physical fit of the signature mean table.

    Estimation is intentionally unconstrained: with only three depth
    windows, an isotonic projection on the attenuation slope collapses the
    per-window estimates to a common value and erases the depth
    differentiation the factorization is meant to capture (diagnosed
    2026-09-05). Physical plausibility is enforced downstream, in
    ``synthesize_virtual_signature``, where sampled devices are projected
    back onto the monotone cone before use — estimation unbiased,
    generation constrained.
    """
    table = torch.as_tensor(mean_table).to(torch.float64)
    if table.ndim != 2:
        raise ValueError("mean_table must have shape (windows, bands)")
    depth_count, band_count = table.shape
    if depth_count <= 0 or band_count <= 0:
        raise ValueError("mean_table must be non-empty")
    if not torch.isfinite(table).all():
        raise ValueError("mean_table must be finite")

    coordinates = band_coordinates(band_count)
    attenuation = torch.zeros(depth_count, dtype=torch.float64)
    denominator = coordinates.square().sum() + 1e-12
    for _ in range(max(1, passes)):
        gain, band_energy, trend = _fit_additive(table, attenuation, coordinates)
        attenuation = attenuation + trend
        if band_count == 1:
            break
        residual = table - gain[:, None] - band_energy[None, :]
        attenuation = torch.stack(
            [
                (residual[window] * coordinates).sum() / denominator
                for window in range(depth_count)
            ]
        )
    gain, band_energy, trend = _fit_additive(table, attenuation, coordinates)
    attenuation = attenuation + trend
    return PhysicalFactors(gain=gain, band_energy=band_energy, attenuation=attenuation)


def reconstruct_mean_table(factors: PhysicalFactors, num_bands: int) -> torch.Tensor:
    coordinates = band_coordinates(num_bands)
    if coordinates.shape[0] != factors.band_energy.shape[0]:
        raise ValueError("band count does not match the factorization")
    table = (
        factors.gain[:, None]
        + factors.band_energy[None, :]
        + factors.attenuation[:, None] * coordinates[None, :]
    )
    if not torch.isfinite(table).all():
        raise ValueError("reconstructed table must be finite")
    return table


def factorize_signature_bank(
    signatures: Mapping[str, DomainSignature],
) -> dict[str, PhysicalFactors]:
    return {domain: fit_physical_factors(signature.mean) for domain, signature in signatures.items()}


def reconstruct_donor_signature(
    factors: PhysicalFactors, template: DomainSignature
) -> DomainSignature:
    """Factor-reconstructed donor: low-DOF mean table, unchanged std table."""
    mean = reconstruct_mean_table(factors, template.mean.shape[1]).to(
        device=template.mean.device, dtype=template.mean.dtype
    )
    return DomainSignature(
        mean=mean,
        std=template.std,
        class_total_weights=dict(template.class_total_weights),
        clipped_std_cells=template.clipped_std_cells,
    )


def _spread(vector_by_domain: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(vector_by_domain)
    return (stacked[:, None, :] - stacked[None, :, :]).norm(dim=-1).max()


def _sample_dirichlet_uniform(count: int, generator: torch.Generator) -> torch.Tensor:
    """Dirichlet(1) via normalized exponentials, driven by the given generator."""
    uniforms = torch.rand(
        count,
        generator=generator,
        device=generator.device,
    ).clamp_min(torch.finfo(torch.float64).tiny)
    exponentials = -torch.log(uniforms.to(torch.float64))
    weights = exponentials / exponentials.sum()
    return weights


def _rand_like(shape: torch.Size, generator: torch.Generator) -> torch.Tensor:
    return 2.0 * torch.rand(
        shape, generator=generator, device=generator.device, dtype=torch.float64
    ).to(torch.float64) - 1.0


def _permutation(count: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(count, generator=generator).to(torch.int64)


def synthesize_virtual_signature(
    factor_bank: Mapping[str, PhysicalFactors],
    signatures: Mapping[str, DomainSignature],
    generator: torch.Generator,
    *,
    extrapolation_radius: float,
    extrapolation_probability: float,
    structure_destroyed: bool = False,
) -> DomainSignature:
    """Synthesize a virtual-device donor signature in factor space.

    Interpolation weights are Dirichlet(1); extrapolation perturbs each factor
    axis element-wise by U(-1, 1) scaled by the fixed radius times the maximum
    pairwise source spread on that axis. Unless ``structure_destroyed``, the
    attenuation slope is re-projected to non-increasing after perturbation and
    the standard-deviation table is interpolated without extrapolation.
    """
    if extrapolation_radius < 0.0:
        raise ValueError("extrapolation_radius must be non-negative")
    if not 0.0 <= extrapolation_probability <= 1.0:
        raise ValueError("extrapolation_probability must be in [0, 1]")
    domains = sorted(factor_bank)
    if len(domains) < 2:
        raise ValueError("virtual synthesis requires at least two source domains")
    if sorted(signatures) != domains:
        raise ValueError("factor bank and signature bank must cover the same domains")

    weights = _sample_dirichlet_uniform(len(domains), generator)
    factors_by_domain = [factor_bank[domain] for domain in domains]

    def _mixed(axis: str) -> torch.Tensor:
        stacked = torch.stack(
            [getattr(factors, axis).to(torch.float64) for factors in factors_by_domain]
        )
        return (weights[:, None] * stacked).sum(dim=0)

    mixed = {axis: _mixed(axis) for axis in ("gain", "band_energy", "attenuation")}

    extrapolate = (
        extrapolation_radius > 0.0
        and extrapolation_probability > 0.0
        and bool(
            torch.rand((), generator=generator, device=generator.device).item()
            < extrapolation_probability
        )
    )
    if extrapolate:
        for axis in ("gain", "band_energy", "attenuation"):
            spread = _spread(
                [getattr(factors, axis).to(torch.float64) for factors in factors_by_domain]
            )
            mixed[axis] = mixed[axis] + extrapolation_radius * spread * _rand_like(
                mixed[axis].shape, generator
            )

    if structure_destroyed:
        mixed["gain"] = mixed["gain"][_permutation(mixed["gain"].numel(), generator)]
        mixed["band_energy"] = mixed["band_energy"][
            _permutation(mixed["band_energy"].numel(), generator)
        ]
        mixed["attenuation"] = mixed["attenuation"][
            _permutation(mixed["attenuation"].numel(), generator)
        ]
    else:
        mixed["attenuation"] = isotonic_non_increasing(mixed["attenuation"])

    factors = PhysicalFactors(**mixed)
    num_bands = factors.band_energy.numel()
    mean_table = reconstruct_mean_table(factors, num_bands)

    std_stack = torch.stack(
        [signatures[domain].std.to(torch.float64) for domain in domains]
    )
    std_table = (weights[:, None, None] * std_stack).sum(dim=0).clamp_min(1e-6)
    if not torch.isfinite(std_table).all():
        raise ValueError("interpolated standard deviations must be finite")

    template = signatures[domains[0]]
    return DomainSignature(
        mean=mean_table.to(device=template.mean.device, dtype=template.mean.dtype),
        std=std_table.to(device=template.std.device, dtype=template.std.dtype),
        class_total_weights=dict(template.class_total_weights),
        clipped_std_cells=0,
    )
