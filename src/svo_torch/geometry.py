"""Batched differentiable geometry primitives used by SVO Torch.

Transforms follow the :math:`T_a_b` convention: ``T_a_b`` maps coordinates
from frame ``b`` into frame ``a``.  Twists are translation first,
``[..., tx, ty, tz, rx, ry, rz]``.  Consequently, an increment used by the
optimizers is left-multiplicative: ``T_new = se3_exp(delta) @ T_old``.

Quaternion functions use scalar-first ``(w, x, y, z)`` ordering.  Every
operation accepts arbitrary leading batch dimensions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _require_floating(value: Tensor, name: str) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")


def _check_last_shape(value: Tensor, shape: tuple[int, ...], name: str) -> None:
    if value.shape[-len(shape) :] != shape:
        raise ValueError(f"{name} must end in shape {shape}, got {tuple(value.shape)}")


def _eye3(reference: Tensor) -> Tensor:
    return torch.eye(3, dtype=reference.dtype, device=reference.device).expand(
        reference.shape[:-1] + (3, 3)
    )


def skew(vector: Tensor) -> Tensor:
    """Return the skew-symmetric cross-product matrix of ``[..., 3]`` vectors."""

    _check_last_shape(vector, (3,), "vector")
    _require_floating(vector, "vector")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def _so3_coefficients(omega: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return Rodrigues ``A, B, C`` coefficients without a zero-angle branch."""

    theta2 = (omega * omega).sum(dim=-1)
    # vector_norm defines a zero subgradient at the origin; sqrt(sum(x*x))
    # would otherwise inject NaNs into the Jacobian of exp(0).
    theta = torch.linalg.vector_norm(omega, dim=-1)

    # torch.sinc(x) = sin(pi*x)/(pi*x), including its analytic value at zero.
    a = torch.sinc(theta / math.pi)
    half_sinc = torch.sinc(theta / (2.0 * math.pi))
    b = 0.5 * half_sinc.square()

    threshold = 1e-4 if omega.dtype == torch.float32 else 1e-8
    theta2_safe = theta2.clamp_min(torch.finfo(omega.dtype).tiny)
    c_regular = (1.0 - a) / theta2_safe
    c_series = 1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0
    c = torch.where(theta2 < threshold, c_series, c_regular)
    return a, b, c


def so3_exp(omega: Tensor) -> Tensor:
    """Exponentiate axis-angle vectors ``[..., 3]`` into rotation matrices."""

    _check_last_shape(omega, (3,), "omega")
    _require_floating(omega, "omega")
    a, b, _ = _so3_coefficients(omega)
    omega_hat = skew(omega)
    return (
        _eye3(omega) + a[..., None, None] * omega_hat + b[..., None, None] * (omega_hat @ omega_hat)
    )


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert scalar-first ``[..., (w, x, y, z)]`` quaternions to matrices.

    Input quaternions need not be normalized, but they must be non-zero.
    """

    _check_last_shape(quaternion, (4,), "quaternion")
    _require_floating(quaternion, "quaternion")
    norm2 = (quaternion * quaternion).sum(dim=-1)
    if bool(torch.any(norm2 <= torch.finfo(quaternion.dtype).tiny)):
        raise ValueError("quaternion must be non-zero")

    w, x, y, z = quaternion.unbind(dim=-1)
    scale = 2.0 / norm2
    xx, yy, zz = scale * x * x, scale * y * y, scale * z * z
    xy, xz, yz = scale * x * y, scale * x * z, scale * y * z
    wx, wy, wz = scale * w * x, scale * w * y, scale * w * z
    return torch.stack(
        (
            1.0 - yy - zz,
            xy - wz,
            xz + wy,
            xy + wz,
            1.0 - xx - zz,
            yz - wx,
            xz - wy,
            yz + wx,
            1.0 - xx - yy,
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to canonical scalar-first quaternions.

    The returned quaternion is normalized and has a non-negative scalar part.
    The largest-component construction remains stable for rotations close to
    180 degrees.
    """

    _check_last_shape(matrix, (3, 3), "matrix")
    _require_floating(matrix, "matrix")
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    squared = torch.stack(
        (
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ),
        dim=-1,
    )
    tiny = torch.finfo(matrix.dtype).tiny
    roots = torch.where(
        squared > 0.0,
        torch.sqrt(squared.clamp_min(tiny)),
        torch.zeros_like(squared),
    )

    candidates = torch.stack(
        (
            torch.stack((roots[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, roots[..., 1].square(), m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, roots[..., 2].square(), m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, roots[..., 3].square()), -1),
        ),
        dim=-2,
    )
    # Each row above is 4*q with its best-conditioned component as divisor.
    candidates = candidates / (2.0 * roots[..., :, None].clamp_min(0.1))
    choice = roots.argmax(dim=-1)
    gather_index = choice[..., None, None].expand(choice.shape + (1, 4))
    quaternion = candidates.gather(dim=-2, index=gather_index).squeeze(-2)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    return torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)


def so3_log(matrix: Tensor) -> Tensor:
    """Logarithm of rotation matrices as principal axis-angle vectors.

    The result has angle in ``[0, pi]`` and is stable at both zero and pi.
    """

    quaternion = matrix_to_quaternion(matrix)
    scalar = quaternion[..., 0]
    vector = quaternion[..., 1:]
    sin_half = torch.linalg.vector_norm(vector, dim=-1)
    angle = 2.0 * torch.atan2(sin_half, scalar.clamp_min(0.0))
    threshold = 1e-4 if matrix.dtype == torch.float32 else 1e-8
    safe = sin_half.clamp_min(torch.finfo(matrix.dtype).tiny)
    regular = angle / safe
    sin2 = sin_half.square()
    series = 2.0 + sin2 / 3.0 + 3.0 * sin2.square() / 20.0
    scale = torch.where(sin_half < threshold, series, regular)
    return scale[..., None] * vector


def se3_exp(twist: Tensor) -> Tensor:
    """Exponentiate translation-first twists into homogeneous transforms.

    ``twist[..., :3]`` is the translational tangent and ``twist[..., 3:]`` is
    the rotational tangent.  Optimization increments are intended to be
    applied on the left: ``T_new = se3_exp(delta) @ T_old``.
    """

    _check_last_shape(twist, (6,), "twist")
    _require_floating(twist, "twist")
    rho = twist[..., :3]
    omega = twist[..., 3:]
    a, b, c = _so3_coefficients(omega)
    omega_hat = skew(omega)
    omega_hat2 = omega_hat @ omega_hat
    identity = _eye3(omega)
    rotation = identity + a[..., None, None] * omega_hat + b[..., None, None] * omega_hat2
    v_matrix = identity + b[..., None, None] * omega_hat + c[..., None, None] * omega_hat2
    translation = (v_matrix @ rho[..., None]).squeeze(-1)

    top = torch.cat((rotation, translation[..., None]), dim=-1)
    bottom = torch.zeros(twist.shape[:-1] + (1, 4), dtype=twist.dtype, device=twist.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat((top, bottom), dim=-2)


def se3_log(transform: Tensor) -> Tensor:
    """Logarithm of ``T_a_b`` as a translation-first tangent vector."""

    _check_last_shape(transform, (4, 4), "transform")
    _require_floating(transform, "transform")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    omega = so3_log(rotation)
    theta2 = (omega * omega).sum(dim=-1)
    theta = torch.linalg.vector_norm(omega, dim=-1)
    omega_hat = skew(omega)

    threshold = 1e-4 if transform.dtype == torch.float32 else 1e-8
    half_theta = 0.5 * theta
    sin_half = torch.sin(half_theta)
    sin_half_safe = sin_half.abs().clamp_min(torch.finfo(transform.dtype).tiny)
    cot_half = torch.cos(half_theta) / sin_half_safe
    theta2_safe = theta2.clamp_min(torch.finfo(transform.dtype).tiny)
    d_regular = (1.0 - half_theta * cot_half) / theta2_safe
    d_series = 1.0 / 12.0 + theta2 / 720.0 + theta2.square() / 30240.0
    d = torch.where(theta2 < threshold, d_series, d_regular)

    v_inverse = _eye3(omega) - 0.5 * omega_hat + d[..., None, None] * (omega_hat @ omega_hat)
    rho = (v_inverse @ translation[..., None]).squeeze(-1)
    return torch.cat((rho, omega), dim=-1)


def invert_transform(transform: Tensor) -> Tensor:
    """Invert rigid transforms, mapping ``T_a_b`` to ``T_b_a``."""

    _check_last_shape(transform, (4, 4), "transform")
    _require_floating(transform, "transform")
    rotation_t = transform[..., :3, :3].transpose(-1, -2)
    translation = -(rotation_t @ transform[..., :3, 3, None]).squeeze(-1)
    top = torch.cat((rotation_t, translation[..., None]), dim=-1)
    bottom = torch.zeros(transform.shape[:-2] + (1, 4), **_factory_kwargs(transform))
    bottom[..., 0, 3] = 1.0
    return torch.cat((top, bottom), dim=-2)


def _factory_kwargs(reference: Tensor) -> dict[str, torch.dtype | torch.device]:
    return {"dtype": reference.dtype, "device": reference.device}


def compose_transforms(*transforms: Tensor) -> Tensor:
    """Compose transforms from left to right.

    For example, ``compose_transforms(T_a_b, T_b_c)`` returns ``T_a_c``.
    Leading dimensions use normal PyTorch broadcasting.
    """

    if not transforms:
        raise ValueError("at least one transform is required")
    result = transforms[0]
    _check_last_shape(result, (4, 4), "transform")
    for transform in transforms[1:]:
        _check_last_shape(transform, (4, 4), "transform")
        result = result @ transform
    return result


def transform_points(transform: Tensor, points: Tensor) -> Tensor:
    """Apply ``T_a_b`` to points expressed in frame ``b``.

    A transform batch prefix is matched to the leftmost point batch prefix;
    any additional point dimensions are transformed by the same transform.
    Thus ``[B, 4, 4]`` with ``[B, N, 3]`` produces ``[B, N, 3]``.  To apply a
    batched transform to a shared ``[N, 3]`` cloud, add a leading singleton to
    the cloud so the intended broadcast is explicit.
    """

    _check_last_shape(transform, (4, 4), "transform")
    _check_last_shape(points, (3,), "points")
    _require_floating(transform, "transform")
    _require_floating(points, "points")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    transform_batch_ndim = transform.ndim - 2
    point_batch_ndim = points.ndim - 1
    extra_point_dims = max(point_batch_ndim - transform_batch_ndim, 0)
    if extra_point_dims:
        shape = rotation.shape[:-2] + (1,) * extra_point_dims
        rotation = rotation.reshape(shape + (3, 3))
        translation = translation.reshape(shape + (3,))
    return (rotation @ points[..., None]).squeeze(-1) + translation


def triangulate_points(
    projection_a: Tensor,
    pixels_a: Tensor,
    projection_b: Tensor,
    pixels_b: Tensor,
) -> Tensor:
    """Triangulate corresponding ``(x, y)`` pixels with the linear DLT method.

    Projection matrices have shape ``[..., 3, 4]`` and pixels have shape
    ``[..., 2]`` or ``[..., N, 2]``.  Projection matrices conventionally map
    homogeneous coordinates in a common reference frame into each image.
    The returned Euclidean points are expressed in that reference frame.
    """

    _check_last_shape(projection_a, (3, 4), "projection_a")
    _check_last_shape(projection_b, (3, 4), "projection_b")
    _check_last_shape(pixels_a, (2,), "pixels_a")
    _check_last_shape(pixels_b, (2,), "pixels_b")
    _require_floating(projection_a, "projection_a")
    _require_floating(projection_b, "projection_b")
    _require_floating(pixels_a, "pixels_a")
    _require_floating(pixels_b, "pixels_b")

    if pixels_a.shape != pixels_b.shape:
        raise ValueError("pixel correspondence arrays must have the same shape")
    dtype = torch.promote_types(projection_a.dtype, projection_b.dtype)
    dtype = torch.promote_types(dtype, pixels_a.dtype)
    dtype = torch.promote_types(dtype, pixels_b.dtype)
    device = projection_a.device
    if projection_b.device != device or pixels_a.device != device or pixels_b.device != device:
        raise ValueError("projection matrices and pixels must be on the same device")
    projection_a = projection_a.to(dtype=dtype)
    projection_b = projection_b.to(dtype=dtype)
    pixels_a = pixels_a.to(dtype=dtype)
    pixels_b = pixels_b.to(dtype=dtype)

    projection_batch_ndim = max(projection_a.ndim, projection_b.ndim) - 2
    pixel_batch_ndim = pixels_a.ndim - 1
    extra_point_dims = max(pixel_batch_ndim - projection_batch_ndim, 0)
    if extra_point_dims:
        suffix = (1,) * extra_point_dims + (3, 4)
        projection_a = projection_a.reshape(projection_a.shape[:-2] + suffix)
        projection_b = projection_b.reshape(projection_b.shape[:-2] + suffix)

    xa, ya = pixels_a.unbind(dim=-1)
    xb, yb = pixels_b.unbind(dim=-1)
    rows = torch.stack(
        (
            xa[..., None] * projection_a[..., 2, :] - projection_a[..., 0, :],
            ya[..., None] * projection_a[..., 2, :] - projection_a[..., 1, :],
            xb[..., None] * projection_b[..., 2, :] - projection_b[..., 0, :],
            yb[..., None] * projection_b[..., 2, :] - projection_b[..., 1, :],
        ),
        dim=-2,
    )
    _, _, vh = torch.linalg.svd(rows, full_matrices=False)
    homogeneous = vh[..., -1, :]
    scale = homogeneous[..., 3:]
    eps = torch.finfo(homogeneous.dtype).eps
    scale_safe = torch.where(
        scale.abs() > eps,
        scale,
        torch.copysign(torch.full_like(scale, eps), scale),
    )
    return homogeneous[..., :3] / scale_safe


__all__ = [
    "compose_transforms",
    "invert_transform",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "se3_exp",
    "se3_log",
    "skew",
    "so3_exp",
    "so3_log",
    "transform_points",
    "triangulate_points",
]
