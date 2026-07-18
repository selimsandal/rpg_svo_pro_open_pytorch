import pytest
import torch

from svo_torch.camera import PinholeCamera
from svo_torch.frame import CORNER, EDGELET
from svo_torch.geometry import invert_transform, se3_exp, transform_points
from svo_torch.optimization import PoseOptimizer, optimize_point


def test_pose_optimizer_recovers_small_perturbation() -> None:
    dtype = torch.float64
    camera = PinholeCamera(320, 240, 260.0, 258.0, 160.0, 120.0, dtype=dtype)
    generator = torch.Generator().manual_seed(4)
    points_world = torch.rand((40, 3), generator=generator, dtype=dtype)
    points_world[:, :2] = (points_world[:, :2] - 0.5) * 2.0
    points_world[:, 2] = points_world[:, 2] * 2.0 + 3.0
    true_pose = torch.eye(4, dtype=dtype)
    observed, visible = camera.project(transform_points(invert_transform(true_pose), points_world))
    initial_pose = invert_transform(
        se3_exp(torch.tensor([0.08, -0.04, 0.03, 0.01, -0.015, 0.008], dtype=dtype))
    )
    result = PoseOptimizer(camera, max_iterations=12).optimize(
        initial_pose, points_world, observed, visible
    )
    assert result.inliers.sum() >= 35
    assert result.final_error < 0.05
    assert torch.allclose(result.T_world_cam, true_pose, atol=2e-3, rtol=0)


def test_pose_optimizer_ignores_behind_camera_correspondences() -> None:
    dtype = torch.float64
    camera = PinholeCamera(320, 240, 260.0, 258.0, 160.0, 120.0, dtype=dtype)
    generator = torch.Generator().manual_seed(14)
    front = torch.rand((30, 3), generator=generator, dtype=dtype)
    front[:, :2] = (front[:, :2] - 0.5) * 1.5
    front[:, 2] = front[:, 2] * 2.0 + 3.0
    behind = torch.rand((120, 3), generator=generator, dtype=dtype)
    behind[:, :2] = (behind[:, :2] - 0.5) * 1.5
    behind[:, 2] = -(behind[:, 2] * 2.0 + 3.0)
    points = torch.cat((front, behind))
    observed, visibility = camera.project(points)
    assert visibility[:30].all() and not visibility[30:].any()
    initial_pose = invert_transform(
        se3_exp(torch.tensor([0.05, -0.03, 0.02, 0.005, -0.008, 0.004], dtype=dtype))
    )
    result = PoseOptimizer(camera, max_iterations=12).optimize(initial_pose, points, observed)
    assert result.inliers[:30].sum() >= 28
    assert not result.inliers[30:].any()
    assert torch.allclose(result.T_world_cam, torch.eye(4, dtype=dtype), atol=2e-3, rtol=0)


def test_pose_optimizer_uses_edgelet_normals_and_pyramid_levels() -> None:
    dtype = torch.float64
    camera = PinholeCamera(320, 240, 260.0, 258.0, 160.0, 120.0, dtype=dtype)
    points = torch.tensor(
        [
            [-0.8, -0.5, 4.0],
            [-0.4, 0.3, 4.5],
            [0.0, -0.2, 5.0],
            [0.3, 0.5, 4.2],
            [0.7, -0.4, 5.3],
            [-0.6, 0.6, 5.5],
            [0.5, 0.1, 4.8],
            [0.1, -0.7, 5.1],
        ],
        dtype=dtype,
    )
    observed, visible = camera.project(points)
    assert visible.all()
    observed = observed.clone()
    observed[0, 1] += 20.0  # tangent to an x-normal edgelet
    observed[1, 0] += 3.0  # 3 px normal error at level zero
    observed[2, 0] += 3.0  # 3 px normal error = 1.5 px at level one
    kinds = torch.full((points.shape[0],), CORNER, dtype=torch.long)
    kinds[:3] = EDGELET
    levels = torch.zeros(points.shape[0], dtype=torch.long)
    levels[2] = 1
    gradients = torch.zeros((points.shape[0], 2), dtype=dtype)
    gradients[:, 0] = 1.0

    # No update is needed here: this isolates the residual and inlier
    # conventions from pose observability and nonlinear convergence.
    result = PoseOptimizer(camera, max_iterations=0).optimize(
        torch.eye(4, dtype=dtype),
        points,
        observed,
        kinds=kinds,
        levels=levels,
        gradients=gradients,
    )

    torch.testing.assert_close(result.residuals[:3], torch.tensor([0.0, 3.0, 1.5], dtype=dtype))
    assert result.inliers[0]
    assert not result.inliers[1]
    assert result.inliers[2]


def test_pose_optimizer_requires_usable_edgelet_gradients() -> None:
    dtype = torch.float64
    camera = PinholeCamera(320, 240, 260.0, 258.0, 160.0, 120.0, dtype=dtype)
    points = torch.tensor(
        [
            [-0.4, -0.3, 4.0],
            [0.4, -0.3, 4.0],
            [-0.4, 0.3, 4.0],
            [0.4, 0.3, 4.0],
            [-0.2, 0.1, 5.0],
            [0.2, -0.1, 5.0],
        ],
        dtype=dtype,
    )
    observed, _ = camera.project(points)
    kinds = torch.full((points.shape[0],), EDGELET, dtype=torch.long)

    with pytest.raises(ValueError, match="gradients are required"):
        PoseOptimizer(camera).optimize(torch.eye(4, dtype=dtype), points, observed, kinds=kinds)
    with pytest.raises(ValueError, match="finite and non-zero"):
        PoseOptimizer(camera).optimize(
            torch.eye(4, dtype=dtype),
            points,
            observed,
            kinds=kinds,
            gradients=torch.zeros((points.shape[0], 2), dtype=dtype),
        )


def _point_observations(
    point: torch.Tensor, poses: torch.Tensor, camera: PinholeCamera
) -> torch.Tensor:
    projections = []
    for pose in poses:
        point_camera = transform_points(invert_transform(pose), point[None])
        pixel, visible = camera.project(point_camera)
        assert bool(visible[0])
        projections.append(pixel[0])
    return torch.stack(projections)


def test_optimize_point_ignores_behind_camera_and_nonfinite_observations() -> None:
    dtype = torch.float64
    camera = PinholeCamera(320, 240, 260.0, 258.0, 160.0, 120.0, dtype=dtype)
    true_point = torch.tensor([0.25, -0.12, 4.2], dtype=dtype)
    front_poses = torch.eye(4, dtype=dtype).repeat(4, 1, 1)
    front_poses[:, 0, 3] = torch.tensor([-0.6, -0.2, 0.25, 0.7], dtype=dtype)
    front_pixels = _point_observations(true_point, front_poses, camera)

    backward_pose = torch.eye(4, dtype=dtype)
    backward_pose[:3, :3] = torch.diag(torch.tensor([-1.0, 1.0, -1.0], dtype=dtype))
    behind_poses = backward_pose.repeat(12, 1, 1)
    behind_pixels = torch.tensor([20.0, 20.0], dtype=dtype).repeat(12, 1)
    invalid_poses = torch.stack(
        (torch.eye(4, dtype=dtype), torch.full((4, 4), float("nan"), dtype=dtype))
    )
    invalid_pixels = torch.stack(
        (
            torch.full((2,), float("nan"), dtype=dtype),
            torch.tensor([100.0, 100.0], dtype=dtype),
        )
    )

    poses = torch.cat((front_poses, behind_poses, invalid_poses))
    pixels = torch.cat((front_pixels, behind_pixels, invalid_pixels))
    cameras = [camera] * poses.shape[0]
    initial = true_point + torch.tensor([0.18, -0.1, 0.35], dtype=dtype)
    refined = optimize_point(initial, poses, pixels, cameras, max_iterations=12)

    assert torch.isfinite(refined).all()
    assert torch.allclose(refined, true_point, atol=2e-4, rtol=0)


def test_optimize_point_rejects_cost_increasing_step(monkeypatch) -> None:
    dtype = torch.float64
    camera = PinholeCamera(320, 240, 260.0, 258.0, 160.0, 120.0, dtype=dtype)
    point = torch.tensor([0.2, -0.1, 4.0], dtype=dtype)
    poses = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    poses[:, 0, 3] = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype)
    pixels = _point_observations(point, poses, camera)

    def increasing_update(hessian: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        del hessian, gradient
        return torch.tensor([1.0, 0.5, -0.25], dtype=dtype)

    monkeypatch.setattr(torch.linalg, "solve", increasing_update)
    refined = optimize_point(point, poses, pixels, [camera] * 3)

    assert torch.equal(refined, point)
