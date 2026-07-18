import math

import pytest
import torch

from svo_torch.geometry import (
    compose_transforms,
    invert_transform,
    matrix_to_quaternion,
    quaternion_to_matrix,
    se3_exp,
    se3_log,
    skew,
    so3_exp,
    so3_log,
    transform_points,
    triangulate_points,
)

DTYPE = torch.float64


def test_skew_is_batched_cross_product() -> None:
    vectors = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.2, 4.0]], dtype=DTYPE)
    others = torch.tensor([[0.3, -2.0, 1.0], [2.0, 1.0, -0.5]], dtype=DTYPE)
    assert skew(vectors).shape == (2, 3, 3)
    torch.testing.assert_close(
        (skew(vectors) @ others[..., None]).squeeze(-1),
        torch.linalg.cross(vectors, others),
    )


@pytest.mark.parametrize(
    "omega",
    [
        [0.0, 0.0, 0.0],
        [1e-10, -2e-10, 3e-10],
        [0.2, -0.3, 0.1],
        [math.pi - 1e-7, 0.0, 0.0],
    ],
)
def test_so3_exp_log_stable_across_angle_range(omega: list[float]) -> None:
    value = torch.tensor(omega, dtype=DTYPE)
    rotation = so3_exp(value)
    torch.testing.assert_close(
        rotation.mT @ rotation, torch.eye(3, dtype=DTYPE), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0, dtype=DTYPE))
    torch.testing.assert_close(so3_exp(so3_log(rotation)), rotation, atol=2e-9, rtol=2e-9)


def test_so3_and_se3_are_batched_and_have_finite_origin_gradients() -> None:
    omega = torch.zeros((2, 3), dtype=DTYPE, requires_grad=True)
    loss = so3_exp(omega).square().sum()
    (gradient,) = torch.autograd.grad(loss, omega)
    assert torch.isfinite(gradient).all()

    twists = torch.tensor(
        [[0.1, -0.2, 0.3, 0.2, -0.1, 0.05], [-1.0, 0.4, 0.2, -0.3, 0.1, 0.2]],
        dtype=DTYPE,
    )
    transforms = se3_exp(twists)
    assert transforms.shape == (2, 4, 4)
    torch.testing.assert_close(se3_log(transforms), twists, atol=2e-12, rtol=2e-12)


def test_se3_translation_first_and_left_composition_convention() -> None:
    translation_only = torch.tensor([1.0, -2.0, 0.5, 0.0, 0.0, 0.0], dtype=DTYPE)
    transform = se3_exp(translation_only)
    torch.testing.assert_close(transform[:3, 3], translation_only[:3])

    current = se3_exp(torch.tensor([0.2, 0.0, 0.0, 0.0, 0.1, 0.0], dtype=DTYPE))
    delta = torch.tensor([0.0, 0.1, 0.0, 0.02, 0.0, 0.0], dtype=DTYPE)
    updated = compose_transforms(se3_exp(delta), current)
    torch.testing.assert_close(updated, se3_exp(delta) @ current)


def test_inverse_compose_and_transform_point_cloud() -> None:
    twists = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2], [0.0, -1.0, 0.5, 0.1, 0.2, 0.0]],
        dtype=DTYPE,
    )
    transforms = se3_exp(twists)
    identity = compose_transforms(transforms, invert_transform(transforms))
    torch.testing.assert_close(
        identity,
        torch.eye(4, dtype=DTYPE).expand(2, 4, 4),
        atol=2e-12,
        rtol=2e-12,
    )

    points = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]], [[-1.0, 0.0, 2.0], [0.2, 0.3, 1.0]]],
        dtype=DTYPE,
    )
    moved = transform_points(transforms, points)
    assert moved.shape == points.shape
    restored = transform_points(invert_transform(transforms), moved)
    torch.testing.assert_close(restored, points, atol=2e-12, rtol=2e-12)


def test_quaternion_conversions_are_scalar_first_and_stable_at_pi() -> None:
    quaternions = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, -0.5, 0.5, -0.5]],
        dtype=DTYPE,
    )
    rotations = quaternion_to_matrix(quaternions)
    recovered = matrix_to_quaternion(rotations)
    torch.testing.assert_close(quaternion_to_matrix(recovered), rotations, atol=1e-12, rtol=1e-12)
    assert torch.all(recovered[:, 0] >= 0.0)


def test_dlt_triangulation_supports_multiple_correspondences_and_gradients() -> None:
    projection_a = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=DTYPE,
    )
    projection_b = projection_a.clone()
    projection_b[0, 3] = -1.0
    expected = torch.tensor([[0.2, -0.1, 3.0], [1.4, 0.5, 5.0]], dtype=DTYPE)

    def project(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
        image = homogeneous @ matrix.mT
        return image[:, :2] / image[:, 2:]

    pixels_a = project(projection_a, expected).requires_grad_()
    pixels_b = project(projection_b, expected).requires_grad_()
    recovered = triangulate_points(projection_a, pixels_a, projection_b, pixels_b)
    torch.testing.assert_close(recovered, expected, atol=2e-12, rtol=2e-12)
    recovered.square().sum().backward()
    assert pixels_a.grad is not None and torch.isfinite(pixels_a.grad).all()
    assert pixels_b.grad is not None and torch.isfinite(pixels_b.grad).all()


def test_shape_validation_is_actionable() -> None:
    with pytest.raises(ValueError, match="omega"):
        so3_exp(torch.zeros(4))
    with pytest.raises(ValueError, match="projection_a"):
        triangulate_points(torch.zeros(4, 4), torch.zeros(2), torch.zeros(3, 4), torch.zeros(2))
