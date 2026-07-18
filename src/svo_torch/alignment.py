"""Inverse-compositional patch tracking and sparse photometric pose alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .camera import Camera
from .geometry import se3_exp, transform_points
from .image import patches_in_bounds, prepare_image, sample_patches


@dataclass(slots=True)
class TrackingResult:
    """Result of independent pyramidal feature tracking."""

    pixels: Tensor
    status: Tensor
    error: Tensor
    iterations: Tensor

    @property
    def valid(self) -> Tensor:
        return self.status


@dataclass(slots=True)
class SparseAlignmentResult:
    """Joint sparse photometric pose estimate."""

    T_cur_ref: Tensor
    valid: Tensor
    error: float
    errors: Tensor
    iterations: int
    converged: bool

    @property
    def status(self) -> Tensor:
        return self.valid


def _validate_pyramids(ref_pyramid: list[Tensor], cur_pyramid: list[Tensor]) -> None:
    if not ref_pyramid or len(ref_pyramid) != len(cur_pyramid):
        raise ValueError("reference and current pyramids must have equal non-zero length")
    for ref, cur in zip(ref_pyramid, cur_pyramid, strict=True):
        if ref.shape[-2:] != cur.shape[-2:]:
            raise ValueError("corresponding pyramid levels must have equal image size")


def _reference_patch_jacobian(
    image: Tensor,
    pixels: Tensor,
    patch_size: int,
) -> tuple[Tensor, Tensor]:
    bordered = sample_patches(image, pixels, patch_size + 2)
    reference = bordered[:, 1:-1, 1:-1]
    dx = 0.5 * (bordered[:, 1:-1, 2:] - bordered[:, 1:-1, :-2])
    dy = 0.5 * (bordered[:, 2:, 1:-1] - bordered[:, :-2, 1:-1])
    jacobian = torch.stack((dx, dy), dim=-1).reshape(pixels.shape[0], -1, 2)
    return reference, jacobian


def _pose_pixel_projection(
    delta: Tensor,
    *,
    transform: Tensor,
    points_ref: Tensor,
    camera: Camera,
) -> Tensor:
    candidate = se3_exp(delta) @ transform
    points_cur = transform_points(candidate, points_ref)
    pixels_cur, _ = camera.project(points_cur)
    return pixels_cur


class PyramidalPatchTracker:
    """Batched 8x8 inverse-compositional Lucas--Kanade tracking."""

    def __init__(
        self,
        patch_size: int,
        max_level: int,
        min_level: int,
        max_iterations: int,
        min_update: float,
        *,
        damping: float = 1e-6,
    ) -> None:
        if patch_size < 2 or patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        if not 0 <= min_level <= max_level:
            raise ValueError("invalid pyramid-level range")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.patch_size = patch_size
        self.max_level = max_level
        self.min_level = min_level
        self.max_iterations = max_iterations
        self.min_update = min_update
        self.damping = damping

    def track(
        self,
        ref_pyramid: list[Tensor],
        cur_pyramid: list[Tensor],
        pixels_ref: Tensor,
        pixels_initial: Tensor | None = None,
    ) -> TrackingResult:
        _validate_pyramids(ref_pyramid, cur_pyramid)
        if pixels_ref.ndim != 2 or pixels_ref.shape[1] != 2:
            raise ValueError("pixels_ref must have shape [N, 2]")
        if pixels_initial is not None and pixels_initial.shape != pixels_ref.shape:
            raise ValueError("pixels_initial must have the same shape as pixels_ref")
        if self.max_level >= len(ref_pyramid):
            raise ValueError("tracker max_level exceeds the supplied pyramid")
        if not pixels_ref.dtype.is_floating_point:
            raise TypeError("pixel coordinates must be floating point")

        base = prepare_image(
            ref_pyramid[0],
            device=ref_pyramid[0].device,
            dtype=(
                ref_pyramid[0].dtype if ref_pyramid[0].dtype.is_floating_point else torch.float32
            ),
        )
        pixels_ref = pixels_ref.to(device=base.device, dtype=base.dtype)
        pixels = (
            pixels_ref.clone() if pixels_initial is None else pixels_initial.to(pixels_ref).clone()
        )
        count = pixels.shape[0]
        if count == 0:
            return TrackingResult(
                pixels=pixels,
                status=torch.empty(0, dtype=torch.bool, device=pixels.device),
                error=torch.empty(0, dtype=pixels.dtype, device=pixels.device),
                iterations=torch.empty(0, dtype=torch.long, device=pixels.device),
            )
        status = torch.zeros(count, dtype=torch.bool, device=pixels.device)
        error = torch.full((count,), float("inf"), dtype=pixels.dtype, device=pixels.device)
        iterations = torch.zeros(count, dtype=torch.long, device=pixels.device)
        identity = torch.eye(2, dtype=pixels.dtype, device=pixels.device)

        for level in range(self.max_level, self.min_level - 1, -1):
            ref_image = prepare_image(ref_pyramid[level], device=pixels.device, dtype=pixels.dtype)
            cur_image = prepare_image(cur_pyramid[level], device=pixels.device, dtype=pixels.dtype)
            scale = float(1 << level)
            reference_level = pixels_ref / scale
            current_level = pixels / scale
            reference, jacobian = _reference_patch_jacobian(
                ref_image, reference_level, self.patch_size
            )
            hessian = jacobian.transpose(1, 2) @ jacobian
            diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
            hessian = hessian + self.damping * diagonal[:, None, None] * identity
            reference_valid = patches_in_bounds(reference_level, ref_image, self.patch_size + 2)
            active = reference_valid.clone()
            converged = torch.zeros_like(active)
            level_error = torch.full_like(error, float("inf"))
            level_iterations = torch.zeros_like(iterations)

            for iteration in range(self.max_iterations):
                current_valid = patches_in_bounds(current_level, cur_image, self.patch_size)
                active &= current_valid & torch.isfinite(current_level).all(dim=-1)
                if not bool(active.any()):
                    break
                current = sample_patches(cur_image, current_level, self.patch_size)
                residual = (current - reference).reshape(count, -1)
                level_error = residual.square().mean(dim=-1).sqrt()
                gradient = -torch.einsum("npi,np->ni", jacobian, residual)
                update, info = torch.linalg.solve_ex(hessian, gradient.unsqueeze(-1))
                update = update.squeeze(-1)
                finite = (info == 0) & torch.isfinite(update).all(dim=-1)
                accepted = active & finite
                update = torch.where(accepted[:, None], update, torch.zeros_like(update))
                current_level = current_level + update
                level_iterations = torch.where(
                    accepted, torch.full_like(level_iterations, iteration + 1), level_iterations
                )
                just_converged = accepted & (
                    torch.linalg.vector_norm(update, dim=-1) < self.min_update
                )
                converged |= just_converged
                active &= ~just_converged & finite

            pixels = current_level * scale
            status = (
                reference_valid
                & patches_in_bounds(current_level, cur_image, self.patch_size)
                & converged
                & torch.isfinite(level_error)
            )
            error = level_error
            iterations += level_iterations

        error = torch.where(status, error, torch.full_like(error, float("inf")))
        return TrackingResult(pixels=pixels, status=status, error=error, iterations=iterations)


class SparseImageAligner:
    """Coarse-to-fine joint SE(3) alignment of sparse reference patches."""

    def __init__(
        self,
        camera: Camera,
        *,
        patch_size: int = 8,
        max_level: int = 4,
        min_level: int = 1,
        max_iterations: int = 10,
        min_update: float = 1e-5,
        huber_delta: float = 0.05,
        damping: float = 1e-5,
        min_features: int = 3,
    ) -> None:
        if patch_size < 2 or patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        if not 0 <= min_level <= max_level:
            raise ValueError("invalid pyramid-level range")
        self.camera = camera
        self.patch_size = patch_size
        self.max_level = max_level
        self.min_level = min_level
        self.max_iterations = max_iterations
        self.min_update = min_update
        self.huber_delta = huber_delta
        self.damping = damping
        self.min_features = min_features

    def align(
        self,
        ref_pyramid: list[Tensor],
        cur_pyramid: list[Tensor],
        pixels_ref: Tensor,
        depths_ref: Tensor,
        T_cur_ref_initial: Tensor,
    ) -> SparseAlignmentResult:
        _validate_pyramids(ref_pyramid, cur_pyramid)
        if pixels_ref.ndim != 2 or pixels_ref.shape[1] != 2:
            raise ValueError("pixels_ref must have shape [N, 2]")
        if depths_ref.shape != (pixels_ref.shape[0],):
            raise ValueError("depths_ref must have shape [N]")
        if T_cur_ref_initial.shape != (4, 4):
            raise ValueError("T_cur_ref_initial must have shape [4,4]")
        if self.max_level >= len(ref_pyramid):
            raise ValueError("aligner max_level exceeds the supplied pyramid")

        base = prepare_image(
            ref_pyramid[0],
            device=ref_pyramid[0].device,
            dtype=(
                ref_pyramid[0].dtype if ref_pyramid[0].dtype.is_floating_point else torch.float32
            ),
        )
        dtype, device = base.dtype, base.device
        pixels_ref = pixels_ref.to(device=device, dtype=dtype)
        depths_ref = depths_ref.to(device=device, dtype=dtype)
        transform = T_cur_ref_initial.to(device=device, dtype=dtype).clone()
        if pixels_ref.shape[0] == 0:
            return SparseAlignmentResult(
                T_cur_ref=transform,
                valid=torch.empty(0, dtype=torch.bool, device=device),
                error=float("inf"),
                errors=torch.empty(0, dtype=dtype, device=device),
                iterations=0,
                converged=False,
            )
        bearings = self.camera.unproject(pixels_ref).to(device=device, dtype=dtype)
        points_ref = bearings * depths_ref[:, None]
        depth_valid = torch.isfinite(depths_ref) & (depths_ref > 0)
        total_iterations = 0
        converged = False

        for level in range(self.max_level, self.min_level - 1, -1):
            ref_image = prepare_image(ref_pyramid[level], device=device, dtype=dtype)
            cur_image = prepare_image(cur_pyramid[level], device=device, dtype=dtype)
            scale = float(1 << level)
            camera_level = self.camera.scale(1.0 / scale)
            reference_centers = pixels_ref / scale
            reference = sample_patches(ref_image, reference_centers, self.patch_size)
            reference_valid = patches_in_bounds(reference_centers, ref_image, self.patch_size)

            for _ in range(self.max_iterations):
                points_cur = transform_points(transform, points_ref)
                pixels_cur, projected = camera_level.project(points_cur)
                valid = (
                    depth_valid
                    & projected
                    & reference_valid
                    & patches_in_bounds(pixels_cur, cur_image, self.patch_size)
                    & torch.isfinite(pixels_cur).all(dim=-1)
                )
                if int(valid.sum()) < self.min_features:
                    break

                def project_with_delta(
                    delta: Tensor,
                    base_transform: Tensor = transform,
                    level_camera: Camera = camera_level,
                ) -> Tensor:
                    return _pose_pixel_projection(
                        delta,
                        transform=base_transform,
                        points_ref=points_ref,
                        camera=level_camera,
                    )

                zero = torch.zeros(6, dtype=dtype, device=device, requires_grad=True)
                current, image_jacobian = _reference_patch_jacobian(
                    cur_image, pixels_cur, self.patch_size
                )
                residual = (current - reference).detach()
                try:
                    projection_jacobian = torch.autograd.functional.jacobian(
                        project_with_delta,
                        zero,
                        create_graph=False,
                        vectorize=True,
                        strategy="forward-mode",
                    )
                except RuntimeError:
                    # Some accelerator backends do not implement forward AD
                    # for every camera-model primitive.  Reverse AD is still
                    # bounded here because it differentiates only [N, 2]
                    # projections, never the [N, P, P] sampled image tensor.
                    projection_jacobian = torch.autograd.functional.jacobian(
                        project_with_delta,
                        zero,
                        create_graph=False,
                        vectorize=True,
                    )
                jacobian = torch.einsum(
                    "npi,niq->npq",
                    image_jacobian.detach(),
                    projection_jacobian.detach(),
                ).reshape(pixels_ref.shape[0], self.patch_size, self.patch_size, 6)
                flat_valid = valid[:, None, None].expand_as(residual).reshape(-1)
                r = residual.reshape(-1)[flat_valid]
                j = jacobian.reshape(-1, 6)[flat_valid]
                absolute = r.detach().abs()
                weights = torch.where(
                    absolute <= self.huber_delta,
                    torch.ones_like(absolute),
                    self.huber_delta / absolute.clamp_min(torch.finfo(dtype).eps),
                )
                sqrt_weight = weights.sqrt()
                weighted_j = j * sqrt_weight[:, None]
                weighted_r = r * sqrt_weight
                hessian = weighted_j.T @ weighted_j
                gradient = weighted_j.T @ weighted_r
                scale_h = torch.diagonal(hessian).mean().clamp_min(1.0)
                hessian = hessian + self.damping * scale_h * torch.eye(
                    6, dtype=dtype, device=device
                )
                update, info = torch.linalg.solve_ex(hessian, -gradient[:, None])
                update = update[:, 0]
                if int(info) != 0 or not bool(torch.isfinite(update).all()):
                    break
                transform = (se3_exp(update) @ transform).detach()
                total_iterations += 1
                if float(torch.linalg.vector_norm(update).detach()) < self.min_update:
                    converged = True
                    break

        final_level = self.min_level
        final_image = prepare_image(cur_pyramid[final_level], device=device, dtype=dtype)
        final_ref = prepare_image(ref_pyramid[final_level], device=device, dtype=dtype)
        final_camera = self.camera.scale(1.0 / float(1 << final_level))
        final_points = transform_points(transform, points_ref)
        final_pixels, projected = final_camera.project(final_points)
        ref_centers = pixels_ref / float(1 << final_level)
        valid = (
            depth_valid
            & projected
            & patches_in_bounds(ref_centers, final_ref, self.patch_size)
            & patches_in_bounds(final_pixels, final_image, self.patch_size)
        )
        ref_patch = sample_patches(final_ref, ref_centers, self.patch_size)
        cur_patch = sample_patches(final_image, final_pixels, self.patch_size)
        errors = (cur_patch - ref_patch).square().mean(dim=(-1, -2)).sqrt()
        errors = torch.where(valid, errors, torch.full_like(errors, float("inf")))
        finite = errors[torch.isfinite(errors)]
        error = float(finite.median().detach()) if finite.numel() else float("inf")
        return SparseAlignmentResult(
            T_cur_ref=transform,
            valid=valid,
            error=error,
            errors=errors,
            iterations=total_iterations,
            converged=converged or (bool(finite.numel()) and error < self.huber_delta),
        )
