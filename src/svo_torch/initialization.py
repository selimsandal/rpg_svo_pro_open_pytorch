"""Self-contained monocular two-view initialization in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, radians

import torch
from torch import Tensor

from .camera import Camera


@dataclass(slots=True)
class TwoViewGeometry:
    """Relative camera pose and triangulated reference-frame points."""

    T_cur_ref: Tensor
    points_ref: Tensor
    inliers: Tensor
    reprojection_errors: Tensor
    iterations: int

    @property
    def inlier_indices(self) -> Tensor:
        return torch.nonzero(self.inliers, as_tuple=False).squeeze(-1)


def _essential_from_bearings(bearings_ref: Tensor, bearings_cur: Tensor) -> Tensor:
    if bearings_ref.shape[0] < 8:
        raise ValueError("the eight-point algorithm needs at least 8 correspondences")
    design = (bearings_cur[:, :, None] * bearings_ref[:, None, :]).reshape(-1, 9)
    _, _, vh = torch.linalg.svd(design, full_matrices=True)
    essential = vh[-1].reshape(3, 3)
    u, singular, vh_e = torch.linalg.svd(essential)
    value = 0.5 * (singular[0] + singular[1])
    corrected = torch.stack((value, value, torch.zeros_like(value)))
    essential = u @ torch.diag(corrected) @ vh_e
    return essential / torch.linalg.vector_norm(essential).clamp_min(1e-12)


def _sampson_error(essential: Tensor, bearings_ref: Tensor, bearings_cur: Tensor) -> Tensor:
    """Symmetric angular epipolar-plane error for unit bearing vectors."""

    e_ref = torch.einsum("ij,nj->ni", essential, bearings_ref)
    et_cur = torch.einsum("ji,nj->ni", essential, bearings_cur)
    numerator = torch.einsum("ni,ni->n", bearings_cur, e_ref).abs()
    epsilon = torch.finfo(bearings_ref.dtype).eps
    sine_cur = numerator / torch.linalg.vector_norm(e_ref, dim=-1).clamp_min(epsilon)
    sine_ref = numerator / torch.linalg.vector_norm(et_cur, dim=-1).clamp_min(epsilon)
    angular = torch.asin(torch.maximum(sine_cur, sine_ref).clamp(0.0, 1.0))
    return angular.square()


def _triangulate_rays(
    bearings_ref: Tensor,
    bearings_cur: Tensor,
    rotation_cur_ref: Tensor,
    translation_cur_ref: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Closest-point two-ray triangulation with depths along unit bearings."""

    rotated_ref = torch.einsum("ij,nj->ni", rotation_cur_ref, bearings_ref)
    system = torch.stack((rotated_ref, -bearings_cur), dim=-1)
    rhs = -translation_cur_ref[None, :, None].expand(bearings_ref.shape[0], 3, 1)
    depths = torch.linalg.lstsq(system, rhs).solution.squeeze(-1)
    points_ref = bearings_ref * depths[:, :1]
    points_cur = torch.einsum("ij,nj->ni", rotation_cur_ref, points_ref) + translation_cur_ref
    return points_ref, points_cur, depths


def _decompose_and_select_pose(
    essential: Tensor,
    bearings_ref: Tensor,
    bearings_cur: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    u, _, vh = torch.linalg.svd(essential)
    if torch.linalg.det(u) < 0:
        u = u.clone()
        u[:, -1] *= -1
    if torch.linalg.det(vh) < 0:
        vh = vh.clone()
        vh[-1] *= -1
    w = essential.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotations = (u @ w @ vh, u @ w.T @ vh)
    translation = u[:, 2]

    best: tuple[int, Tensor, Tensor, Tensor, Tensor] | None = None
    for rotation in rotations:
        if torch.linalg.det(rotation) < 0:
            rotation = -rotation
        for sign in (1.0, -1.0):
            candidate_t = sign * translation
            points_ref, points_cur, depths = _triangulate_rays(
                bearings_ref, bearings_cur, rotation, candidate_t
            )
            cheirality = (depths[:, 0] > 1e-6) & (depths[:, 1] > 1e-6)
            count = int(cheirality.sum())
            if best is None or count > best[0]:
                best = (count, rotation, candidate_t, points_ref, points_cur)
    assert best is not None
    _, rotation, translation, points_ref, points_cur = best
    return rotation, translation, points_ref, points_cur


def estimate_two_view_geometry(
    camera: Camera,
    pixels_ref: Tensor,
    pixels_cur: Tensor,
    *,
    pixel_threshold: float = 2.0,
    probability: float = 0.999,
    max_iterations: int = 256,
    min_inliers: int = 20,
    scene_depth: float = 1.0,
    min_parallax_degrees: float = 0.5,
    random_seed: int | None = 7,
) -> TwoViewGeometry | None:
    """Estimate relative pose with eight-point RANSAC and triangulate.

    Translation and points are scaled so that the median point distance in the
    current camera equals ``scene_depth``, matching SVO's monocular scale prior.
    """

    if pixels_ref.shape != pixels_cur.shape or pixels_ref.ndim != 2 or pixels_ref.shape[1] != 2:
        raise ValueError("pixel correspondence tensors must both have shape [N, 2]")
    count = pixels_ref.shape[0]
    if count < max(8, min_inliers):
        return None
    if min_parallax_degrees < 0:
        raise ValueError("min_parallax_degrees must be non-negative")

    bearings_ref = camera.unproject(pixels_ref)
    bearings_cur = camera.unproject(pixels_cur)
    angle_threshold = camera.pixel_error_angle(pixel_threshold).to(pixels_ref)
    angular_threshold2 = angle_threshold.square()

    generator: torch.Generator | None = None
    if random_seed is not None:
        generator = torch.Generator(device=pixels_ref.device)
        generator.manual_seed(random_seed)
    best_inliers = torch.zeros(count, dtype=torch.bool, device=pixels_ref.device)
    best_essential: Tensor | None = None
    target_iterations = max_iterations
    completed = 0

    for iteration in range(max_iterations):
        sample = torch.randperm(count, generator=generator, device=pixels_ref.device)[:8]
        try:
            essential = _essential_from_bearings(bearings_ref[sample], bearings_cur[sample])
        except torch.linalg.LinAlgError:
            continue
        errors = _sampson_error(essential, bearings_ref, bearings_cur)
        inliers = torch.isfinite(errors) & (errors <= angular_threshold2)
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_essential = essential
            inlier_ratio = float(inliers.float().mean())
            all_inlier_probability = inlier_ratio**8
            if 0 < all_inlier_probability < 1:
                estimated = log(1 - probability) / log(1 - all_inlier_probability)
                target_iterations = min(target_iterations, max(1, ceil(estimated)))
        completed = iteration + 1
        if completed >= target_iterations:
            break

    if best_essential is None or int(best_inliers.sum()) < max(8, min_inliers):
        return None
    try:
        essential = _essential_from_bearings(bearings_ref[best_inliers], bearings_cur[best_inliers])
        errors = _sampson_error(essential, bearings_ref, bearings_cur)
        best_inliers = torch.isfinite(errors) & (errors <= angular_threshold2)
        if int(best_inliers.sum()) < max(8, min_inliers):
            return None
        rotation, translation, points_ref, points_cur = _decompose_and_select_pose(
            essential, bearings_ref[best_inliers], bearings_cur[best_inliers]
        )
    except torch.linalg.LinAlgError:
        return None

    # Reject points behind either camera and high angular triangulation errors.
    predicted_ref = points_ref / torch.linalg.vector_norm(
        points_ref, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    predicted_cur = points_cur / torch.linalg.vector_norm(
        points_cur, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    local_ref = bearings_ref[best_inliers]
    local_cur = bearings_cur[best_inliers]
    angular = torch.maximum(
        torch.acos((predicted_ref * local_ref).sum(-1).clamp(-1.0, 1.0)),
        torch.acos((predicted_cur * local_cur).sum(-1).clamp(-1.0, 1.0)),
    )
    depths = torch.linalg.vector_norm(points_ref, dim=-1)
    depths_cur = torch.linalg.vector_norm(points_cur, dim=-1)
    good_local = (
        torch.isfinite(angular)
        & (angular <= angle_threshold)
        & ((points_ref * local_ref).sum(-1) > 0)
        & ((points_cur * local_cur).sum(-1) > 0)
        & (depths > 1e-6)
        & (depths_cur > 1e-6)
    )
    rotated_ref = torch.einsum("ij,nj->ni", rotation, local_ref)
    parallax = torch.acos((rotated_ref * local_cur).sum(-1).clamp(-1.0, 1.0))
    finite_parallax = good_local & torch.isfinite(parallax)
    if not bool(finite_parallax.any()):
        return None
    if float(parallax[finite_parallax].median()) < radians(min_parallax_degrees):
        return None
    original_indices = torch.nonzero(best_inliers, as_tuple=False).squeeze(-1)
    final_inliers = torch.zeros_like(best_inliers)
    final_inliers[original_indices[good_local]] = True
    if int(final_inliers.sum()) < min_inliers:
        return None

    points_ref = points_ref[good_local]
    points_cur = points_cur[good_local]
    scale = scene_depth / torch.linalg.vector_norm(points_cur, dim=-1).median().clamp_min(1e-6)
    points_ref = points_ref * scale
    translation = translation * scale
    T_cur_ref = torch.eye(4, dtype=pixels_ref.dtype, device=pixels_ref.device)
    T_cur_ref[:3, :3] = rotation
    T_cur_ref[:3, 3] = translation
    radians_per_pixel = angle_threshold / max(pixel_threshold, 1e-12)
    reprojection_errors = torch.sqrt(errors.clamp_min(0)) / radians_per_pixel.clamp_min(1e-12)
    return TwoViewGeometry(
        T_cur_ref=T_cur_ref,
        points_ref=points_ref,
        inliers=final_inliers,
        reprojection_errors=reprojection_errors,
        iterations=completed,
    )
