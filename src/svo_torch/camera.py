"""Tensor-native camera models and SVO camera-rig calibration loading.

Pixels are always ordered ``(x, y)``.  ``project`` maps camera-frame points to
pixels and returns an image-valid mask; ``unproject`` returns unit-length
camera-frame bearing vectors.  Camera calibration tensors follow their input
points to preserve dtype, device, broadcasting, and autograd behavior.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from .geometry import invert_transform


def _infer_dtype_device(
    values: Sequence[object],
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> tuple[torch.dtype, torch.device]:
    tensor = next((value for value in values if isinstance(value, Tensor)), None)
    if dtype is None:
        dtype = tensor.dtype if isinstance(tensor, Tensor) and tensor.is_floating_point() else None
        dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError("camera calibration dtype must be floating point")
    if device is None:
        device = tensor.device if isinstance(tensor, Tensor) else torch.device("cpu")
    return dtype, torch.device(device)


def _stack_scalars(
    values: Sequence[object],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    dtype, device = _infer_dtype_device(values, dtype, device)
    scalars = [torch.as_tensor(value, dtype=dtype, device=device).reshape(()) for value in values]
    return torch.stack(scalars)


def _as_float_vector(
    values: Tensor | Sequence[float],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    sample: list[object] = [values] if isinstance(values, Tensor) else list(values)
    dtype, device = _infer_dtype_device(sample, dtype, device)
    result = torch.as_tensor(values, dtype=dtype, device=device)
    if result.ndim != 1:
        raise ValueError("camera parameter vectors must be one-dimensional")
    return result


def _safe_signed(value: Tensor, minimum: float) -> Tensor:
    sign = torch.where(value < 0.0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() >= minimum, value, sign * minimum)


def _polynomial(coefficients: Tensor, value: Tensor) -> Tensor:
    result = torch.zeros_like(value) + coefficients[-1]
    for coefficient in coefficients[:-1].flip(0):
        result = result * value + coefficient
    return result


class Camera(ABC):
    """Abstract central camera API."""

    width: int
    height: int
    label: str

    def __init__(self, width: int, height: int, *, label: str = "") -> None:
        if width <= 0 or height <= 0:
            raise ValueError("camera width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self.label = label

    @property
    def image_size(self) -> tuple[int, int]:
        """Image size as ``(width, height)``."""

        return self.width, self.height

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Calibration tensor dtype."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Calibration tensor device."""

    @property
    @abstractmethod
    def principal_point(self) -> Tensor:
        """Optical center in image coordinates."""

    @abstractmethod
    def project(self, points_camera: Tensor) -> tuple[Tensor, Tensor]:
        """Project ``[..., 3]`` points, returning ``(pixels, valid)``."""

    @abstractmethod
    def unproject(self, pixels: Tensor) -> Tensor:
        """Back-project ``[..., 2]`` pixels to normalized bearing vectors."""

    @abstractmethod
    def scale(self, factor: float) -> Camera:
        """Return calibration for an isotropically rescaled image."""

    @abstractmethod
    def to(self, *args: object, **kwargs: object) -> Camera:
        """Return a camera whose calibration tensors use the requested placement."""

    def is_in_frame(self, pixels: Tensor, border: float = 0.0) -> Tensor:
        """Return whether ``(x, y)`` pixels are finite and inside the image."""

        if pixels.shape[-1:] != (2,):
            raise ValueError(f"pixels must end in shape (2,), got {tuple(pixels.shape)}")
        x, y = pixels.unbind(dim=-1)
        return (
            torch.isfinite(pixels).all(dim=-1)
            & (x >= border)
            & (y >= border)
            & (x < self.width - border)
            & (y < self.height - border)
        )

    def pixel_error_angle(self, pixel_error: float | Tensor) -> Tensor:
        """Convert an image-space error to a local angular error in radians."""

        error = torch.as_tensor(pixel_error, dtype=self.dtype, device=self.device)
        if bool(error < 0):
            raise ValueError("pixel_error must be non-negative")
        center = self.principal_point.to(dtype=self.dtype, device=self.device)
        pixels = torch.stack(
            (
                center,
                center + torch.stack((error, torch.zeros_like(error))),
                center + torch.stack((torch.zeros_like(error), error)),
            )
        )
        bearings = self.unproject(pixels)
        cosines = (bearings[1:] * bearings[:1]).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cosines).mean()


class PinholeCamera(Camera):
    """Pinhole projection with SVO-compatible distortion models.

    ``distortion`` accepts ``"none"``, ``"radial-tangential"`` (parameters
    ``k1, k2, p1, p2``), ``"equidistant"`` (Kannala-Brandt
    ``k1, k2, k3, k4``), or ``"fisheye"``/``"atan"`` (one FOV parameter).
    """

    _ALIASES = {
        "none": "none",
        "no": "none",
        "radtan": "radial-tangential",
        "radial_tangential": "radial-tangential",
        "radial-tangential": "radial-tangential",
        "equidistant": "equidistant",
        "kannala-brandt": "equidistant",
        "fisheye": "fisheye",
        "fisheye(atan)": "fisheye",
        "atan": "fisheye",
        "fov": "fisheye",
    }
    _PARAMETER_COUNTS = {"none": 0, "radial-tangential": 4, "equidistant": 4, "fisheye": 1}

    def __init__(
        self,
        width: int,
        height: int,
        fx: float | Tensor,
        fy: float | Tensor,
        cx: float | Tensor,
        cy: float | Tensor,
        distortion: str = "none",
        distortion_params: Tensor | Sequence[float] | None = None,
        *,
        label: str = "",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(width, height, label=label)
        try:
            self.distortion = self._ALIASES[distortion.lower()]
        except KeyError as error:
            raise ValueError(f"unsupported pinhole distortion: {distortion}") from error
        self._intrinsics = _stack_scalars((fx, fy, cx, cy), dtype=dtype, device=device)
        count = self._PARAMETER_COUNTS[self.distortion]
        if distortion_params is None:
            distortion_params = [0.0] * count
        self._distortion_params = _as_float_vector(
            distortion_params,
            dtype=self._intrinsics.dtype,
            device=self._intrinsics.device,
        )
        if self._distortion_params.numel() != count:
            raise ValueError(
                f"{self.distortion} distortion requires {count} parameters, "
                f"got {self._distortion_params.numel()}"
            )
        if bool(torch.any(self._intrinsics[:2] <= 0.0)):
            raise ValueError("focal lengths must be positive")

    @property
    def intrinsics(self) -> Tensor:
        return self._intrinsics

    @property
    def distortion_params(self) -> Tensor:
        return self._distortion_params

    @property
    def fx(self) -> Tensor:
        return self._intrinsics[0]

    @property
    def fy(self) -> Tensor:
        return self._intrinsics[1]

    @property
    def cx(self) -> Tensor:
        return self._intrinsics[2]

    @property
    def cy(self) -> Tensor:
        return self._intrinsics[3]

    @property
    def principal_point(self) -> Tensor:
        return self._intrinsics[2:4]

    @property
    def error_multiplier(self) -> Tensor:
        """Legacy SVO pixel/radian scale at the optical center."""

        return self.fx.abs()

    def pixel_error_angle(self, pixel_error: float | Tensor) -> Tensor:
        error = torch.as_tensor(pixel_error, dtype=self.dtype, device=self.device)
        if bool(error < 0):
            raise ValueError("pixel_error must be non-negative")
        return torch.atan(error / (2.0 * self.fx)) + torch.atan(error / (2.0 * self.fy))

    @property
    def dtype(self) -> torch.dtype:
        return self._intrinsics.dtype

    @property
    def device(self) -> torch.device:
        return self._intrinsics.device

    def _distort(self, normalized: Tensor, parameters: Tensor) -> Tensor:
        if self.distortion == "none":
            return normalized
        x, y = normalized.unbind(dim=-1)
        radius2 = x.square() + y.square()

        if self.distortion == "radial-tangential":
            k1, k2, p1, p2 = parameters
            radial = 1.0 + k1 * radius2 + k2 * radius2.square()
            xy2 = 2.0 * x * y
            return torch.stack(
                (
                    x * radial + p1 * xy2 + p2 * (radius2 + 2.0 * x.square()),
                    y * radial + p2 * xy2 + p1 * (radius2 + 2.0 * y.square()),
                ),
                dim=-1,
            )

        # vector_norm has a defined zero subgradient, which keeps projection
        # Jacobians finite on the optical axis.
        radius = torch.linalg.vector_norm(normalized, dim=-1)
        threshold = 1e-8
        if self.distortion == "equidistant":
            k1, k2, k3, k4 = parameters
            theta = torch.atan(radius)
            theta2 = theta.square()
            theta_distorted = theta * (
                1.0 + k1 * theta2 + k2 * theta2.square() + k3 * theta2.pow(3) + k4 * theta2.pow(4)
            )
            factor = theta_distorted / radius.clamp_min(threshold)
            factor = torch.where(radius < threshold, torch.ones_like(factor), factor)
            return normalized * factor[..., None]

        # Devernay-Faugeras FOV model, called AtanDistortion in SVO.
        (strength,) = parameters
        strength_safe = _safe_signed(strength, threshold)
        tangent = 2.0 * torch.tan(0.5 * strength_safe)
        factor = torch.atan(radius * tangent) / (strength_safe * radius.clamp_min(threshold))
        factor = torch.where(radius < 1e-3, torch.ones_like(factor), factor)
        factor = torch.where(strength.abs() < threshold, torch.ones_like(factor), factor)
        return normalized * factor[..., None]

    def _undistort(self, distorted: Tensor, parameters: Tensor) -> Tensor:
        if self.distortion == "none":
            return distorted
        x0, y0 = distorted.unbind(dim=-1)

        if self.distortion == "radial-tangential":
            k1, k2, p1, p2 = parameters
            x, y = x0, y0
            for _ in range(8):
                radius2 = x.square() + y.square()
                inverse_radial = 1.0 / (1.0 + k1 * radius2 + k2 * radius2.square())
                xy2 = 2.0 * x * y
                dx = p1 * xy2 + p2 * (radius2 + 2.0 * x.square())
                dy = p2 * xy2 + p1 * (radius2 + 2.0 * y.square())
                x = (x0 - dx) * inverse_radial
                y = (y0 - dy) * inverse_radial
            return torch.stack((x, y), dim=-1)

        radius_distorted = torch.linalg.vector_norm(distorted, dim=-1)
        threshold = 1e-8
        if self.distortion == "equidistant":
            k1, k2, k3, k4 = parameters
            theta = radius_distorted
            for _ in range(8):
                theta2 = theta.square()
                theta4 = theta2.square()
                theta6 = theta4 * theta2
                theta8 = theta4.square()
                residual = (
                    theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
                    - radius_distorted
                )
                derivative = (
                    1.0
                    + 3.0 * k1 * theta2
                    + 5.0 * k2 * theta4
                    + 7.0 * k3 * theta6
                    + 9.0 * k4 * theta8
                )
                theta = theta - residual / _safe_signed(derivative, threshold)
            factor = torch.tan(theta) / radius_distorted.clamp_min(threshold)
            factor = torch.where(radius_distorted < threshold, torch.ones_like(factor), factor)
            return distorted * factor[..., None]

        (strength,) = parameters
        strength_safe = _safe_signed(strength, threshold)
        tangent = 2.0 * torch.tan(0.5 * strength_safe)
        radius = torch.tan(radius_distorted * strength_safe) / tangent
        factor = radius / radius_distorted.clamp_min(threshold)
        factor = torch.where(radius_distorted <= 1e-2, torch.ones_like(factor), factor)
        factor = torch.where(strength.abs() < threshold, torch.ones_like(factor), factor)
        return distorted * factor[..., None]

    def project(self, points_camera: Tensor) -> tuple[Tensor, Tensor]:
        if points_camera.shape[-1:] != (3,):
            raise ValueError(
                f"points_camera must end in shape (3,), got {tuple(points_camera.shape)}"
            )
        if not points_camera.is_floating_point():
            raise TypeError("points_camera must be floating point")
        intrinsics = self._intrinsics.to(points_camera)
        parameters = self._distortion_params.to(points_camera)
        x, y, z = points_camera.unbind(dim=-1)
        eps = 10.0 * torch.finfo(points_camera.dtype).eps
        normalized = torch.stack((x, y), dim=-1) / _safe_signed(z, eps)[..., None]
        distorted = self._distort(normalized, parameters)
        pixels = distorted * intrinsics[:2] + intrinsics[2:]
        valid = torch.isfinite(points_camera).all(dim=-1) & (z > eps) & self.is_in_frame(pixels)
        return pixels, valid

    def unproject(self, pixels: Tensor) -> Tensor:
        if pixels.shape[-1:] != (2,):
            raise ValueError(f"pixels must end in shape (2,), got {tuple(pixels.shape)}")
        if not pixels.is_floating_point():
            raise TypeError("pixels must be floating point")
        intrinsics = self._intrinsics.to(pixels)
        parameters = self._distortion_params.to(pixels)
        distorted = (pixels - intrinsics[2:]) / intrinsics[:2]
        normalized = self._undistort(distorted, parameters)
        bearings = torch.cat((normalized, torch.ones_like(normalized[..., :1])), dim=-1)
        return torch.nn.functional.normalize(bearings, dim=-1)

    def scale(self, factor: float) -> PinholeCamera:
        if factor <= 0.0:
            raise ValueError("scale factor must be positive")
        scaled = self._intrinsics * self._intrinsics.new_tensor([factor, factor, factor, factor])
        return PinholeCamera(
            max(1, math.floor(self.width * factor)),
            max(1, math.floor(self.height * factor)),
            *scaled,
            distortion=self.distortion,
            distortion_params=self._distortion_params,
            label=self.label,
        )

    def to(self, *args: object, **kwargs: object) -> PinholeCamera:
        intrinsics = self._intrinsics.to(*args, **kwargs)
        if not intrinsics.is_floating_point():
            raise TypeError("camera calibration dtype must be floating point")
        return PinholeCamera(
            self.width,
            self.height,
            *intrinsics,
            distortion=self.distortion,
            distortion_params=self._distortion_params.to(
                device=intrinsics.device, dtype=intrinsics.dtype
            ),
            label=self.label,
        )


class OmniCamera(Camera):
    """Scaramuzza omnidirectional camera with SVO's 24-value layout.

    Parameters are ``poly[5], center[2], affine[3], inverse_poly[12],
    mask_relative_radii[2]``.  The affine tuple ``(c, d, e)`` forms
    ``[[1, e], [d, c]]``.  Mask radii ``(0, 0)`` disable the annular mask.
    """

    def __init__(
        self,
        width: int,
        height: int,
        parameters: Tensor | Sequence[float],
        *,
        label: str = "",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(width, height, label=label)
        self._parameters = _as_float_vector(parameters, dtype=dtype, device=device)
        if self._parameters.numel() != 24:
            raise ValueError(
                f"SVO omni calibration requires exactly 24 parameters, "
                f"got {self._parameters.numel()}"
            )
        low, high = self._parameters[22:24]
        mask_disabled = bool(low == 0.0 and high == 0.0)
        if not mask_disabled and not bool(0.0 <= low < high <= 1.0):
            raise ValueError("omni mask radii must be (0, 0) or satisfy 0 <= low < high <= 1")

    @property
    def parameters(self) -> Tensor:
        return self._parameters

    @property
    def intrinsics(self) -> Tensor:
        return self._parameters

    @property
    def principal_point(self) -> Tensor:
        return self._parameters[5:7]

    @property
    def dtype(self) -> torch.dtype:
        return self._parameters.dtype

    @property
    def device(self) -> torch.device:
        return self._parameters.device

    @staticmethod
    def _affine(parameters: Tensor) -> Tensor:
        c, d, e = parameters[7:10]
        one = torch.ones_like(c)
        return torch.stack((one, e, d, c)).reshape(2, 2)

    def _mask_valid(self, pixels: Tensor, parameters: Tensor) -> Tensor:
        low, high = parameters[22:24]
        disabled = (low == 0.0) & (high == 0.0)
        radius = torch.linalg.vector_norm(pixels - parameters[5:7], dim=-1)
        reference_radius = 0.5 * self.height
        inside = (radius >= reference_radius * low) & (radius <= reference_radius * high)
        return disabled | inside

    def project(self, points_camera: Tensor) -> tuple[Tensor, Tensor]:
        if points_camera.shape[-1:] != (3,):
            raise ValueError(
                f"points_camera must end in shape (3,), got {tuple(points_camera.shape)}"
            )
        if not points_camera.is_floating_point():
            raise TypeError("points_camera must be floating point")
        parameters = self._parameters.to(points_camera)
        x, y, z = points_camera.unbind(dim=-1)
        radial = torch.linalg.vector_norm(points_camera[..., :2], dim=-1)
        eps = 10.0 * torch.finfo(points_camera.dtype).eps
        theta = torch.atan2(-z, radial)
        rho = _polynomial(parameters[10:22], theta)
        direction = torch.stack((x, y), dim=-1) / radial.clamp_min(eps)[..., None]
        direction = torch.where((radial < eps)[..., None], torch.zeros_like(direction), direction)
        raw_pixels = direction * rho[..., None]
        pixels = raw_pixels @ self._affine(parameters).transpose(-1, -2) + parameters[5:7]
        norm = torch.linalg.vector_norm(points_camera, dim=-1)
        pole_is_defined = (radial >= eps) | (z > 0.0)
        valid = (
            torch.isfinite(points_camera).all(dim=-1)
            & (norm > eps)
            & pole_is_defined
            & self.is_in_frame(pixels)
            & self._mask_valid(pixels, parameters)
        )
        return pixels, valid

    def unproject(self, pixels: Tensor) -> Tensor:
        if pixels.shape[-1:] != (2,):
            raise ValueError(f"pixels must end in shape (2,), got {tuple(pixels.shape)}")
        if not pixels.is_floating_point():
            raise TypeError("pixels must be floating point")
        parameters = self._parameters.to(pixels)
        affine_inverse = torch.linalg.inv(self._affine(parameters))
        rectified = (pixels - parameters[5:7]) @ affine_inverse.transpose(-1, -2)
        radius = torch.linalg.vector_norm(rectified, dim=-1)
        z = -_polynomial(parameters[:5], radius)
        bearings = torch.cat((rectified, z[..., None]), dim=-1)
        return torch.nn.functional.normalize(bearings, dim=-1)

    def scale(self, factor: float) -> OmniCamera:
        if factor <= 0.0:
            raise ValueError("scale factor must be positive")
        parameters = self._parameters
        powers = torch.arange(5, dtype=parameters.dtype, device=parameters.device)
        polynomial = parameters[:5] * factor ** (1.0 - powers)
        scaled = torch.cat(
            (
                polynomial,
                parameters[5:7] * factor,
                parameters[7:10],
                parameters[10:22] * factor,
                parameters[22:24],
            )
        )
        return OmniCamera(
            max(1, math.floor(self.width * factor)),
            max(1, math.floor(self.height * factor)),
            scaled,
            label=self.label,
        )

    def to(self, *args: object, **kwargs: object) -> OmniCamera:
        parameters = self._parameters.to(*args, **kwargs)
        if not parameters.is_floating_point():
            raise TypeError("camera calibration dtype must be floating point")
        return OmniCamera(self.width, self.height, parameters, label=self.label)


class CameraRig:
    """A synchronized camera collection with camera-to-body transforms.

    ``T_body_camera[i]`` maps points from camera ``i`` into the body frame.
    """

    def __init__(
        self,
        cameras: Sequence[Camera],
        T_body_camera: Tensor | None = None,
        *,
        label: str = "",
    ) -> None:
        if not cameras:
            raise ValueError("a camera rig must contain at least one camera")
        self.cameras = tuple(cameras)
        self.label = label
        if T_body_camera is None:
            first = self.cameras[0]
            T_body_camera = (
                torch.eye(4, dtype=first.dtype, device=first.device)
                .expand(len(cameras), 4, 4)
                .clone()
            )
        if T_body_camera.shape != (len(cameras), 4, 4):
            raise ValueError(
                f"T_body_camera must have shape ({len(cameras)}, 4, 4), "
                f"got {tuple(T_body_camera.shape)}"
            )
        if not T_body_camera.is_floating_point():
            raise TypeError("T_body_camera must be floating point")
        if not torch.isfinite(T_body_camera).all():
            raise ValueError("T_body_camera must contain only finite values")
        rotation = T_body_camera[:, :3, :3]
        identity = torch.eye(3, dtype=T_body_camera.dtype, device=T_body_camera.device).expand_as(
            rotation
        )
        tolerance = 1e-4 if T_body_camera.dtype == torch.float32 else 1e-8
        if not torch.allclose(
            rotation.transpose(-1, -2) @ rotation,
            identity,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("T_body_camera rotations must be orthonormal")
        determinant = torch.linalg.det(rotation)
        if not torch.allclose(
            determinant,
            torch.ones_like(determinant),
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("T_body_camera rotations must have determinant +1")
        expected_bottom = T_body_camera.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(len(cameras), 4)
        if not torch.allclose(T_body_camera[:, 3], expected_bottom, atol=tolerance, rtol=0.0):
            raise ValueError("T_body_camera must have homogeneous bottom rows")
        if any(camera.device != T_body_camera.device for camera in self.cameras):
            raise ValueError("all cameras and rig transforms must share a device")
        if any(camera.dtype != T_body_camera.dtype for camera in self.cameras):
            raise ValueError("all cameras and rig transforms must share a dtype")
        self.T_body_camera = T_body_camera

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, index: int) -> Camera:
        return self.cameras[index]

    @property
    def T_camera_body(self) -> Tensor:
        return invert_transform(self.T_body_camera)

    @property
    def device(self) -> torch.device:
        return self.T_body_camera.device

    @property
    def dtype(self) -> torch.dtype:
        return self.T_body_camera.dtype

    def scale(self, factor: float) -> CameraRig:
        return CameraRig(
            [camera.scale(factor) for camera in self.cameras],
            self.T_body_camera,
            label=self.label,
        )

    def to(self, *args: object, **kwargs: object) -> CameraRig:
        transforms = self.T_body_camera.to(*args, **kwargs)
        if not transforms.is_floating_point():
            raise TypeError("camera rig dtype must be floating point")
        cameras = [
            camera.to(device=transforms.device, dtype=transforms.dtype) for camera in self.cameras
        ]
        return CameraRig(cameras, transforms, label=self.label)


def _yaml_vector(node: Any, name: str) -> list[float]:
    if isinstance(node, Mapping):
        if "data" not in node:
            raise ValueError(f"{name} matrix is missing 'data'")
        node = node["data"]
    if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
        raise ValueError(f"{name} must be a sequence or a matrix mapping")
    return [float(value) for value in node]


def _camera_from_yaml(
    node: Mapping[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> Camera:
    try:
        camera_type = str(node["type"]).lower()
        width = int(node["image_width"])
        height = int(node["image_height"])
        intrinsics = _yaml_vector(node["intrinsics"], "intrinsics")
    except KeyError as error:
        raise ValueError(f"camera entry is missing required key {error.args[0]!r}") from error
    label = str(node.get("label", ""))
    if camera_type == "pinhole":
        if len(intrinsics) != 4:
            raise ValueError("pinhole intrinsics must contain fx, fy, cx, cy")
        distortion_node = node.get("distortion", {"type": "none", "parameters": []})
        if not isinstance(distortion_node, Mapping):
            raise ValueError("distortion must be a mapping")
        distortion = str(distortion_node.get("type", "none"))
        distortion_params = _yaml_vector(
            distortion_node.get("parameters", []), "distortion parameters"
        )
        return PinholeCamera(
            width,
            height,
            *intrinsics,
            distortion=distortion,
            distortion_params=distortion_params,
            label=label,
            dtype=dtype,
            device=device,
        )
    if camera_type == "omni":
        return OmniCamera(
            width,
            height,
            intrinsics,
            label=label,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"unsupported SVO camera type: {camera_type}")


def _transform_from_yaml(
    node: Any,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> Tensor:
    values = _yaml_vector(node, "T_B_C")
    if len(values) != 16:
        raise ValueError(f"T_B_C must contain 16 values, got {len(values)}")
    transform = torch.tensor(values, dtype=dtype, device=device).reshape(4, 4)
    # The canonical SVO examples serialize the unused homogeneous row as zero.
    transform = torch.cat(
        (
            transform[:3],
            transform.new_tensor([[0.0, 0.0, 0.0, 1.0]]),
        ),
        dim=0,
    )
    return transform


def load_camera_rig(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> CameraRig:
    """Load canonical SVO camera or NCamera YAML calibration.

    Rig entries use canonical ``camera`` and ``T_B_C`` keys.  ``T_B_C`` is
    retained directly as ``rig.T_body_camera`` because it maps camera-frame
    coordinates into the body frame.  A single canonical camera mapping is
    also accepted, with an optional identity-defaulted ``T_B_C``.
    """

    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError("camera calibration dtype must be floating point")
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError(f"camera calibration root must be a mapping: {path}")

    cameras: list[Camera] = []
    transforms: list[Tensor] = []
    if "cameras" in document:
        entries = document["cameras"]
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
            raise ValueError("camera rig 'cameras' must be a non-empty sequence")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"camera rig entry {index} must be a mapping")
            camera_node = entry.get("camera")
            if not isinstance(camera_node, Mapping):
                raise ValueError(f"camera rig entry {index} is missing a camera mapping")
            if "T_B_C" not in entry:
                raise ValueError(f"camera rig entry {index} is missing T_B_C")
            cameras.append(_camera_from_yaml(camera_node, dtype=dtype, device=device))
            transforms.append(_transform_from_yaml(entry["T_B_C"], dtype=dtype, device=device))
    else:
        cameras.append(_camera_from_yaml(document, dtype=dtype, device=device))
        transform_node = document.get(
            "T_B_C", {"data": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}
        )
        transforms.append(_transform_from_yaml(transform_node, dtype=dtype, device=device))

    return CameraRig(cameras, torch.stack(transforms), label=str(document.get("label", "")))


__all__ = ["Camera", "CameraRig", "OmniCamera", "PinholeCamera", "load_camera_rig"]
