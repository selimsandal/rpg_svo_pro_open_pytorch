"""Probabilistic inverse-depth seeds and direct epipolar matching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .camera import Camera
from .image import patches_in_bounds, prepare_image, sample_patches


@dataclass(slots=True)
class DepthSeeds:
    """Parallel inverse-depth Beta-Gaussian state ``[mu, sigma2, a, b]``."""

    mu: Tensor
    sigma2: Tensor
    a: Tensor
    b: Tensor

    def __post_init__(self) -> None:
        shape = self.mu.shape
        if any(value.shape != shape for value in (self.sigma2, self.a, self.b)):
            raise ValueError("mu, sigma2, a, and b must have equal shapes")
        if not self.mu.dtype.is_floating_point:
            raise TypeError("depth seed state must be floating point")

    @property
    def values(self) -> Tensor:
        return torch.stack((self.mu, self.sigma2, self.a, self.b), dim=-1)

    @property
    def depth(self) -> Tensor:
        return self.mu.reciprocal()

    @classmethod
    def from_tensor(cls, values: Tensor) -> DepthSeeds:
        if values.ndim < 1 or values.shape[-1] != 4:
            raise ValueError("seed tensor must end in [mu, sigma2, a, b]")
        return cls(*values.unbind(dim=-1))

    @classmethod
    def initialize(
        cls,
        count: int,
        *,
        depth_mean: float,
        depth_min: float,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        a: float = 10.0,
        b: float = 10.0,
    ) -> DepthSeeds:
        if count < 0 or depth_mean <= 0 or depth_min <= 0:
            raise ValueError("count must be non-negative and depths must be positive")
        mu = torch.full((count,), 1.0 / depth_mean, dtype=dtype, device=device)
        mu_range = 1.0 / depth_min
        sigma2 = torch.full_like(mu, mu_range * mu_range / 36.0)
        return cls(mu, sigma2, torch.full_like(mu, a), torch.full_like(mu, b))

    def to(self, *args: object, **kwargs: object) -> DepthSeeds:
        mu = self.mu.to(*args, **kwargs)
        return DepthSeeds(
            mu,
            self.sigma2.to(device=mu.device, dtype=mu.dtype),
            self.a.to(device=mu.device, dtype=mu.dtype),
            self.b.to(device=mu.device, dtype=mu.dtype),
        )

    def __len__(self) -> int:
        return self.mu.shape[0] if self.mu.ndim else 1

    def __getitem__(self, index: Tensor | slice | int) -> DepthSeeds:
        return DepthSeeds(self.mu[index], self.sigma2[index], self.a[index], self.b[index])

    def converged(self, mu_range: float | Tensor, threshold: float = 200.0) -> Tensor:
        bound = torch.as_tensor(mu_range, dtype=self.mu.dtype, device=self.mu.device) / threshold
        return self.sigma2 < bound.square()


def inverse_depth_variance(depth: Tensor, depth_sigma: Tensor, eps: float = 1e-12) -> Tensor:
    """SVO's symmetric depth interval mapped to inverse-depth variance."""

    near_inverse = (depth - depth_sigma).clamp_min(eps).reciprocal()
    far_inverse = (depth + depth_sigma).clamp_min(eps).reciprocal()
    return (0.5 * (near_inverse - far_inverse)).square()


def update_filter_gaussian(
    seeds: DepthSeeds,
    measurement: Tensor | float,
    tau2: Tensor | float,
) -> tuple[DepthSeeds, Tensor]:
    """Fuse an inverse-depth measurement with the exact Gaussian update."""

    z = torch.as_tensor(measurement, dtype=seeds.mu.dtype, device=seeds.mu.device)
    variance = torch.as_tensor(tau2, dtype=seeds.mu.dtype, device=seeds.mu.device)
    denominator = seeds.sigma2 + variance
    mu = (seeds.sigma2 * z + variance * seeds.mu) / denominator
    sigma2 = seeds.sigma2 * variance / denominator
    valid = torch.isfinite(mu) & torch.isfinite(sigma2) & (mu >= 0) & (sigma2 >= 0) & (variance > 0)
    result = DepthSeeds(
        torch.where(valid, mu, seeds.mu),
        torch.where(valid, sigma2, seeds.sigma2),
        seeds.a,
        seeds.b,
    )
    return result, valid


def update_filter_vogiatzis(
    seeds: DepthSeeds,
    measurement: Tensor | float,
    tau2: Tensor | float,
    mu_range: Tensor | float,
) -> tuple[DepthSeeds, Tensor]:
    """Exact Vogiatzis Beta-Gaussian inlier/outlier mixture update."""

    z = torch.as_tensor(measurement, dtype=seeds.mu.dtype, device=seeds.mu.device)
    variance = torch.as_tensor(tau2, dtype=seeds.mu.dtype, device=seeds.mu.device)
    value_range = torch.as_tensor(mu_range, dtype=seeds.mu.dtype, device=seeds.mu.device)
    norm_scale = torch.sqrt(seeds.sigma2 + variance)
    s2 = 1.0 / (1.0 / seeds.sigma2 + 1.0 / variance)
    m = s2 * (seeds.mu / seeds.sigma2 + z / variance)
    normal_pdf = torch.exp(-0.5 * ((z - seeds.mu) / norm_scale).square()) / (
        math.sqrt(2.0 * math.pi) * norm_scale
    )
    c1 = seeds.a / (seeds.a + seeds.b) * normal_pdf
    c2 = seeds.b / (seeds.a + seeds.b) / value_range
    normalization = c1 + c2
    c1 = c1 / normalization
    c2 = c2 / normalization
    denominator_1 = seeds.a + seeds.b + 1.0
    denominator_2 = seeds.a + seeds.b + 2.0
    f = c1 * (seeds.a + 1.0) / denominator_1 + c2 * seeds.a / denominator_1
    e = c1 * (seeds.a + 1.0) * (seeds.a + 2.0) / (denominator_1 * denominator_2) + c2 * seeds.a * (
        seeds.a + 1.0
    ) / (denominator_1 * denominator_2)

    mu = c1 * m + c2 * seeds.mu
    sigma2 = c1 * (s2 + m.square()) + c2 * (seeds.sigma2 + seeds.mu.square()) - mu.square()
    # Preserve SVO's fallback: a negative variance is discarded, while a
    # negative mean marks the seed invalid and resets its mean to one.
    sigma2 = torch.where(sigma2 < 0, seeds.sigma2, sigma2)
    beta_a = (e - f) / (f - e / f)
    beta_b = beta_a * (1.0 - f) / f
    negative_mu = mu < 0
    mu = torch.where(negative_mu, torch.ones_like(mu), mu)
    valid = (
        ~negative_mu
        & torch.isfinite(norm_scale)
        & torch.isfinite(mu)
        & torch.isfinite(sigma2)
        & torch.isfinite(beta_a)
        & torch.isfinite(beta_b)
        & (variance > 0)
        & (value_range > 0)
    )
    result = DepthSeeds(
        torch.where(valid | negative_mu, mu, seeds.mu),
        torch.where(valid, sigma2, seeds.sigma2),
        torch.where(valid, beta_a, seeds.a),
        torch.where(valid, beta_b, seeds.b),
    )
    return result, valid


# Concise aliases are convenient in tensor pipelines and retain the original
# C++ names above for source-to-source discoverability.
gaussian_update = update_filter_gaussian
vogiatzis_update = update_filter_vogiatzis


def compute_tau(
    T_ref_cur: Tensor,
    bearing_ref: Tensor,
    depth: Tensor | float,
    pixel_error_angle: Tensor | float,
) -> Tensor:
    """Depth uncertainty induced by angular pixel noise (law of sines)."""

    z = torch.as_tensor(depth, dtype=bearing_ref.dtype, device=bearing_ref.device)
    angle_error = torch.as_tensor(
        pixel_error_angle, dtype=bearing_ref.dtype, device=bearing_ref.device
    )
    translation = T_ref_cur[..., :3, 3]
    a = bearing_ref * z[..., None] - translation
    t_norm = torch.linalg.vector_norm(translation, dim=-1)
    a_norm = torch.linalg.vector_norm(a, dim=-1)
    tiny = torch.finfo(bearing_ref.dtype).eps
    alpha = torch.acos(
        ((bearing_ref * translation).sum(dim=-1) / t_norm.clamp_min(tiny)).clamp(-1.0, 1.0)
    )
    beta = torch.acos(
        ((a * -translation).sum(dim=-1) / (t_norm * a_norm).clamp_min(tiny)).clamp(-1.0, 1.0)
    )
    beta_plus = beta + angle_error
    gamma_plus = math.pi - alpha - beta_plus
    z_plus = t_norm * torch.sin(beta_plus) / torch.sin(gamma_plus)
    return z_plus - z


def zmssd_score(reference: Tensor, current: Tensor, *, mean: bool = False) -> Tensor:
    """Zero-mean SSD over the final two patch dimensions."""

    if reference.shape != current.shape or reference.ndim < 2:
        raise ValueError("reference and current patches must have equal [...,H,W] shape")
    difference = current - reference
    area = reference.shape[-2] * reference.shape[-1]
    score = difference.square().sum(dim=(-1, -2)) - difference.sum(dim=(-1, -2)).square() / area
    return score / area if mean else score


def triangulate_depth(
    T_cur_ref: Tensor,
    bearing_ref: Tensor,
    bearing_cur: Tensor,
    *,
    determinant_threshold: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Least-squares ray intersection, returning Euclidean depth in reference."""

    rotation = T_cur_ref[..., :3, :3]
    translation = T_cur_ref[..., :3, 3]
    rotated_ref = torch.einsum("...ij,...j->...i", rotation, bearing_ref)
    # R*f_ref*d_ref + t = f_cur*d_cur.
    system = torch.stack((rotated_ref, -bearing_cur), dim=-1)
    ata = system.transpose(-1, -2) @ system
    determinant = torch.linalg.det(ata)
    rhs = -(system.transpose(-1, -2) @ translation[..., None])
    solution, info = torch.linalg.solve_ex(ata, rhs)
    depth = solution[..., 0, 0]
    depth_cur = solution[..., 1, 0]
    valid = (
        (info == 0)
        & (determinant >= determinant_threshold)
        & torch.isfinite(depth)
        & torch.isfinite(depth_cur)
        & (depth > 0)
        & (depth_cur > 0)
    )
    return depth, valid


@dataclass(slots=True)
class EpipolarMatchResult:
    pixels: Tensor
    depth: Tensor
    inverse_depth: Tensor
    score: Tensor
    valid: Tensor


class EpipolarMatcher:
    """Sample an inverse-depth epipolar segment with normalized ZMSSD."""

    def __init__(
        self,
        camera: Camera,
        *,
        patch_size: int = 8,
        max_steps: int = 128,
        step: float = 0.7,
        max_score: float = 0.20,
    ) -> None:
        if patch_size < 2 or patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        if max_steps < 2 or step <= 0 or max_score <= 0:
            raise ValueError("invalid epipolar matcher options")
        self.camera = camera
        self.patch_size = patch_size
        self.max_steps = max_steps
        self.step = step
        self.max_score = max_score

    def match(
        self,
        ref_pyramid: list[Tensor],
        cur_pyramid: list[Tensor],
        pixels_ref: Tensor,
        T_cur_ref: Tensor,
        inverse_depth_min: Tensor | float,
        inverse_depth_max: Tensor | float,
        *,
        level: int = 0,
    ) -> EpipolarMatchResult:
        if not ref_pyramid or len(ref_pyramid) != len(cur_pyramid):
            raise ValueError("reference and current pyramids must have equal non-zero length")
        if not 0 <= level < len(ref_pyramid):
            raise ValueError("invalid pyramid level")
        if pixels_ref.ndim != 2 or pixels_ref.shape[1] != 2:
            raise ValueError("pixels_ref must have shape [N,2]")
        if T_cur_ref.shape != (4, 4):
            raise ValueError("T_cur_ref must have shape [4,4]")

        image_ref = prepare_image(
            ref_pyramid[level], device=ref_pyramid[level].device, dtype=ref_pyramid[level].dtype
        )
        image_cur = prepare_image(
            cur_pyramid[level], device=image_ref.device, dtype=image_ref.dtype
        )
        pixels_ref = pixels_ref.to(device=image_ref.device, dtype=image_ref.dtype)
        transform = T_cur_ref.to(device=image_ref.device, dtype=image_ref.dtype)
        count = pixels_ref.shape[0]
        if count == 0:
            empty = torch.empty(0, dtype=image_ref.dtype, device=image_ref.device)
            return EpipolarMatchResult(
                pixels=torch.empty((0, 2), dtype=image_ref.dtype, device=image_ref.device),
                depth=empty,
                inverse_depth=empty,
                score=empty,
                valid=torch.empty(0, dtype=torch.bool, device=image_ref.device),
            )
        rho_a = torch.as_tensor(
            inverse_depth_min, dtype=image_ref.dtype, device=image_ref.device
        ).expand(count)
        rho_b = torch.as_tensor(
            inverse_depth_max, dtype=image_ref.dtype, device=image_ref.device
        ).expand(count)
        bearings_ref = self.camera.unproject(pixels_ref).to(image_ref)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        rotated = torch.einsum("ij,nj->ni", rotation, bearings_ref)
        endpoint_a, visible_a = self.camera.project(rotated + translation * rho_a[:, None])
        endpoint_b, visible_b = self.camera.project(rotated + translation * rho_b[:, None])
        scale = float(1 << level)
        endpoint_a = endpoint_a / scale
        endpoint_b = endpoint_b / scale
        line_length = torch.linalg.vector_norm(endpoint_a - endpoint_b, dim=-1)
        n_steps = torch.ceil(line_length / self.step).to(torch.long).clamp(2, self.max_steps)
        fraction = torch.linspace(
            0.0,
            1.0,
            self.max_steps,
            device=image_ref.device,
            dtype=image_ref.dtype,
        )
        denominator = (n_steps - 1).clamp_min(1).to(image_ref.dtype)
        per_feature_fraction = fraction[None] * (self.max_steps - 1) / denominator[:, None]
        per_feature_fraction = per_feature_fraction.clamp_max(1.0)
        candidates = (
            endpoint_a[:, None]
            + per_feature_fraction[..., None] * (endpoint_b - endpoint_a)[:, None]
        )
        step_index = torch.arange(self.max_steps, device=image_ref.device)[None]
        candidate_enabled = step_index < n_steps[:, None]

        ref_centers = pixels_ref / scale
        reference = sample_patches(image_ref, ref_centers, self.patch_size)
        current = sample_patches(image_cur, candidates.reshape(-1, 2), self.patch_size).reshape(
            count, self.max_steps, self.patch_size, self.patch_size
        )
        scores = zmssd_score(reference[:, None].expand_as(current), current, mean=True)
        candidate_valid = patches_in_bounds(
            candidates.reshape(-1, 2), image_cur, self.patch_size
        ).reshape(count, self.max_steps)
        reference_valid = patches_in_bounds(ref_centers, image_ref, self.patch_size)
        candidate_valid &= candidate_enabled & reference_valid[:, None]
        candidate_valid &= (visible_a & visible_b)[:, None]
        scores = torch.where(candidate_valid, scores, torch.full_like(scores, float("inf")))
        best_score, best_index = scores.min(dim=-1)
        row = torch.arange(count, device=image_ref.device)
        best_level = candidates[row, best_index]
        best_pixel = best_level * scale
        bearing_cur = self.camera.unproject(best_pixel).to(image_ref)
        depth, triangulated = triangulate_depth(
            transform.expand(count, -1, -1), bearings_ref, bearing_cur
        )
        valid = triangulated & torch.isfinite(best_score) & (best_score <= self.max_score)
        inverse_depth = torch.where(valid, depth.reciprocal(), torch.full_like(depth, float("nan")))
        depth = torch.where(valid, depth, torch.full_like(depth, float("nan")))
        return EpipolarMatchResult(best_pixel, depth, inverse_depth, best_score, valid)
