"""Run MonoSVO on a deterministic, rendered non-planar point scene.

This is deliberately a geometry-driven smoke test rather than a prerecorded
asset: every grayscale frame is rendered with PyTorch from known camera poses
and depth-varying world points.

Run it from the project root with::

    uv run python examples/synthetic_sequence.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from svo_torch.camera import PinholeCamera
from svo_torch.config import SVOConfig
from svo_torch.geometry import invert_transform, so3_exp, transform_points
from svo_torch.odometry import MonoSVO, OdometryResult


@dataclass(slots=True)
class SyntheticSequence:
    camera: PinholeCamera
    images: Tensor
    T_world_camera: Tensor
    points_world: Tensor
    timestamp_step_ns: int = 10_000_000

    def timestamp(self, index: int) -> int:
        return index * self.timestamp_step_ns


def _scene_points(
    width: int,
    height: int,
    focal_length: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Make a regular image-space layout with deterministic varying depths."""

    pixel_rows: list[list[float]] = []
    depths: list[float] = []
    index = 0
    for y in range(14, height - 13, 13):
        for x in range(14, width - 13, 13):
            jitter_x = (((index * 7) % 5) - 2) * 0.45
            jitter_y = (((index * 11) % 5) - 2) * 0.45
            pixel_rows.append([x + jitter_x, y + jitter_y])
            depths.append(2.4 + ((index * 17) % 31) / 31.0 * 2.6)
            index += 1
    pixels = torch.tensor(pixel_rows, dtype=dtype, device=device)
    z = torch.tensor(depths, dtype=dtype, device=device)
    center = pixels.new_tensor([width / 2.0, height / 2.0])
    xy = (pixels - center) * (z / focal_length)[:, None]
    points = torch.cat((xy, z[:, None]), dim=-1)

    indices = torch.arange(points.shape[0], dtype=dtype, device=device)
    amplitudes = 0.35 + 0.55 * torch.remainder(indices * 23.0, 29.0) / 28.0
    return points, amplitudes


def _camera_poses(
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    poses: list[Tensor] = []
    for index in range(count):
        pose = torch.eye(4, dtype=dtype, device=device)
        pose[:3, :3] = so3_exp(pose.new_tensor([0.0, 0.002 * index, 0.0]))
        pose[:3, 3] = pose.new_tensor([0.045 * index, 0.004 * index, 0.0])
        poses.append(pose)
    return torch.stack(poses)


def _render(
    camera: PinholeCamera,
    points_world: Tensor,
    amplitudes: Tensor,
    T_world_camera: Tensor,
) -> Tensor:
    height, width = camera.height, camera.width
    points_camera = transform_points(invert_transform(T_world_camera), points_world)
    pixels, visible = camera.project(points_camera)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=points_world.dtype, device=points_world.device),
        torch.arange(width, dtype=points_world.dtype, device=points_world.device),
        indexing="ij",
    )
    dx = xx[None] - pixels[:, 0, None, None]
    dy = yy[None] - pixels[:, 1, None, None]
    main_blob = torch.exp(-(dx.square() + dy.square()) / (2.0 * 1.25**2))

    # A smaller offset lobe removes rotational symmetry and makes each local
    # Lucas--Kanade Hessian well-conditioned under subpixel translations.
    offset_dx = xx[None] - (pixels[:, 0, None, None] + 1.7)
    offset_dy = yy[None] - (pixels[:, 1, None, None] - 1.2)
    offset_blob = torch.exp(-(offset_dx.square() + offset_dy.square()) / (2.0 * 0.7**2))
    weights = (amplitudes * visible.to(amplitudes.dtype))[:, None, None]
    return (weights * (main_blob + 0.25 * offset_blob)).sum(dim=0).clamp(0.0, 1.0)


def make_synthetic_sequence(
    count: int = 6,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> SyntheticSequence:
    """Render a short deterministic sequence with metric ground-truth poses."""

    if count < 3:
        raise ValueError("the odometry demonstration requires at least three frames")
    device = torch.device(device)
    width, height, focal_length = 160, 120, 140.0
    camera = PinholeCamera(
        width,
        height,
        focal_length,
        focal_length,
        width / 2.0,
        height / 2.0,
        dtype=dtype,
        device=device,
    )
    points, amplitudes = _scene_points(
        width,
        height,
        focal_length,
        dtype=dtype,
        device=device,
    )
    poses = _camera_poses(count, dtype=dtype, device=device)
    images = torch.stack([_render(camera, points, amplitudes, pose) for pose in poses])
    return SyntheticSequence(camera, images, poses, points)


def demo_config() -> SVOConfig:
    """Small deterministic configuration used by the example and E2E test."""

    return SVOConfig(
        max_features=70,
        grid_size=12,
        n_pyr_levels=3,
        detector_threshold=0.002,
        detector_edgelet_ratio=0.0,
        feature_border=7,
        quality_min_features=12,
        quality_max_feature_drop=30,
        init_min_features=25,
        init_min_tracked=20,
        init_min_inliers=15,
        init_min_disparity=1.0,
        init_map_scale=3.5,
        init_ransac_iterations=64,
        pose_reprojection_threshold=2.0,
        pose_iterations=5,
        max_keyframes=4,
        keyframe_min_disparity=0.8,
        keyframe_min_frames=1,
        keyframe_min_tracked_ratio=0.5,
        patch_size=6,
        alignment_max_level=2,
        alignment_min_level=1,
        alignment_iterations=4,
        alignment_min_update=0.02,
        use_sparse_image_alignment=True,
        device="cpu",
        dtype="float32",
        deterministic=True,
        random_seed=7,
    )


def run_sequence(sequence: SyntheticSequence | None = None) -> tuple[MonoSVO, list[OdometryResult]]:
    """Run and return the frontend plus one result per rendered frame."""

    sequence = make_synthetic_sequence() if sequence is None else sequence
    frontend = MonoSVO(sequence.camera, demo_config())
    frontend.start()
    results = [
        frontend.process(image, sequence.timestamp(index))
        for index, image in enumerate(sequence.images)
    ]
    return frontend, results


def main() -> None:
    sequence = make_synthetic_sequence()
    frontend, results = run_sequence(sequence)
    print("frame  stage         update     observations  estimated position")
    for index, result in enumerate(results):
        position = (
            "-"
            if result.T_world_cam is None
            else " ".join(f"{value: .3f}" for value in result.T_world_cam[:3, 3].tolist())
        )
        print(
            f"{index:5d}  {result.stage.value:12s}  {result.update.value:9s}"
            f"  {result.num_observations:12d}  {position}"
        )
    print(f"keyframes={len(frontend.map)} landmarks={len(frontend.map.landmarks)}")


if __name__ == "__main__":
    main()
