import torch

import svo_torch.initialization as initialization
from svo_torch.camera import PinholeCamera
from svo_torch.geometry import se3_exp, skew, so3_exp, transform_points
from svo_torch.initialization import _sampson_error, estimate_two_view_geometry


def test_two_view_geometry_recovers_pose_and_scale() -> None:
    dtype = torch.float64
    camera = PinholeCamera(640, 480, 450.0, 452.0, 320.0, 240.0, dtype=dtype)
    generator = torch.Generator().manual_seed(12)
    points_ref = torch.rand((120, 3), generator=generator, dtype=dtype)
    points_ref[:, :2] = (points_ref[:, :2] - 0.5) * 3.0
    points_ref[:, 2] = points_ref[:, 2] * 3.0 + 3.0
    truth = se3_exp(torch.tensor([0.35, 0.02, 0.04, 0.01, -0.025, 0.006], dtype=dtype))
    pixels_ref, valid_ref = camera.project(points_ref)
    points_cur = transform_points(truth, points_ref)
    pixels_cur, valid_cur = camera.project(points_cur)
    valid = valid_ref & valid_cur
    noise = torch.randn(pixels_cur.shape, generator=generator, dtype=dtype) * 0.05
    result = estimate_two_view_geometry(
        camera,
        pixels_ref[valid],
        pixels_cur[valid] + noise[valid],
        pixel_threshold=1.0,
        min_inliers=40,
        scene_depth=2.0,
        max_iterations=128,
    )
    assert result is not None
    assert result.inliers.sum() >= 80
    rotation_error = result.T_cur_ref[:3, :3] @ truth[:3, :3].T
    angle = torch.acos(((torch.trace(rotation_error) - 1) / 2).clamp(-1.0, 1.0))
    assert angle < 0.02
    direction_cosine = torch.dot(result.T_cur_ref[:3, 3], truth[:3, 3]) / (
        torch.linalg.vector_norm(result.T_cur_ref[:3, 3]) * torch.linalg.vector_norm(truth[:3, 3])
    )
    assert direction_cosine > 0.98
    assert torch.allclose(
        torch.linalg.vector_norm(
            transform_points(result.T_cur_ref, result.points_ref), dim=-1
        ).median(),
        torch.tensor(2.0, dtype=dtype),
        atol=1e-6,
    )


def test_two_view_geometry_rejects_pure_rotation() -> None:
    dtype = torch.float64
    camera = PinholeCamera(640, 480, 450.0, 452.0, 320.0, 240.0, dtype=dtype)
    generator = torch.Generator().manual_seed(23)
    points_ref = torch.rand((240, 3), generator=generator, dtype=dtype)
    points_ref[:, :2] = (points_ref[:, :2] - 0.5) * 3.0
    points_ref[:, 2] = points_ref[:, 2] * 3.0 + 3.0
    rotation = so3_exp(torch.tensor([0.02, -0.1, 0.01], dtype=dtype))
    pixels_ref, valid_ref = camera.project(points_ref)
    pixels_cur, valid_cur = camera.project(points_ref @ rotation.T)
    valid = valid_ref & valid_cur
    noise = 0.03 * torch.randn(pixels_cur.shape, generator=generator, dtype=dtype)
    result = estimate_two_view_geometry(
        camera,
        pixels_ref[valid],
        pixels_cur[valid] + noise[valid],
        pixel_threshold=1.0,
        min_inliers=40,
        min_parallax_degrees=0.5,
        max_iterations=128,
    )
    assert result is None


def test_angular_epipolar_error_is_coordinate_rotation_invariant() -> None:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(9)
    points = torch.rand((30, 3), generator=generator, dtype=dtype)
    points[:, :2] -= 0.5
    points[:, 2] += 2.0
    rotation = so3_exp(torch.tensor([0.04, -0.02, 0.03], dtype=dtype))
    translation = torch.tensor([0.2, -0.03, 0.04], dtype=dtype)
    bearings_ref = torch.nn.functional.normalize(points, dim=-1)
    bearings_cur = torch.nn.functional.normalize(points @ rotation.T + translation, dim=-1)
    essential = skew(translation) @ rotation
    errors = _sampson_error(essential, bearings_ref, bearings_cur)
    common_rotation = so3_exp(torch.tensor([-0.3, 0.2, 0.1], dtype=dtype))
    rotated_errors = _sampson_error(
        common_rotation @ essential @ common_rotation.T,
        bearings_ref @ common_rotation.T,
        bearings_cur @ common_rotation.T,
    )
    torch.testing.assert_close(rotated_errors, errors, atol=1e-14, rtol=1e-8)


def test_ransac_retains_a_minimal_model_when_consensus_refit_degrades(monkeypatch) -> None:
    """A bad non-minimal refit must neither replace nor stop a good model."""

    dtype = torch.float64
    camera = PinholeCamera(640, 480, 450.0, 452.0, 320.0, 240.0, dtype=dtype)
    generator = torch.Generator().manual_seed(31)
    points_ref = torch.rand((80, 3), generator=generator, dtype=dtype)
    points_ref[:, :2] = (points_ref[:, :2] - 0.5) * 2.0
    points_ref[:, 2] = points_ref[:, 2] * 2.0 + 4.0
    truth = se3_exp(torch.tensor([0.4, -0.02, 0.03, 0.01, -0.02, 0.005], dtype=dtype))
    pixels_ref, valid_ref = camera.project(points_ref)
    pixels_cur, valid_cur = camera.project(transform_points(truth, points_ref))
    valid = valid_ref & valid_cur

    good_essential = skew(truth[:3, 3]) @ truth[:3, :3]
    bad_essential = skew(torch.tensor([-0.1, 0.7, 0.2], dtype=dtype)) @ so3_exp(
        torch.tensor([0.4, -0.3, 0.2], dtype=dtype)
    )
    calls = {"minimal": 0, "refit": 0}

    def controlled_estimator(bearings_ref, bearings_cur):
        del bearings_cur
        if bearings_ref.shape[0] == 8:
            calls["minimal"] += 1
            return good_essential
        calls["refit"] += 1
        return bad_essential

    monkeypatch.setattr(initialization, "_essential_from_bearings", controlled_estimator)
    result = estimate_two_view_geometry(
        camera,
        pixels_ref[valid],
        pixels_cur[valid],
        pixel_threshold=1.0,
        min_inliers=40,
        max_iterations=4,
        random_seed=7,
    )

    assert result is not None
    assert int(result.inliers.sum()) == int(valid.sum())
    assert result.iterations == 4
    assert calls == {"minimal": 4, "refit": 2}
