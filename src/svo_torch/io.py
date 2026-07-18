"""Trajectory serialization helpers.

The original SVO examples write one camera pose per line using the common TUM
layout::

    timestamp_seconds tx ty tz qx qy qz qw

This module accepts homogeneous ``T_world_cam`` matrices and keeps timestamp
conversion integer based so nanosecond precision is not lost through a float.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import AbstractContextManager
from os import PathLike
from pathlib import Path
from typing import Any, TextIO

import torch

from .datasets import TimestampError

_INT64_MAX = 2**63 - 1


def _validate_timestamp_ns(timestamp_ns: int) -> int:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        raise TimestampError("timestamp_ns must be an integer")
    if not 0 <= timestamp_ns <= _INT64_MAX:
        raise TimestampError(f"timestamp_ns {timestamp_ns} is outside the non-negative int64 range")
    return timestamp_ns


def format_timestamp_seconds(timestamp_ns: int) -> str:
    """Format integer nanoseconds as exact decimal seconds."""

    timestamp_ns = _validate_timestamp_ns(timestamp_ns)
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return f"{seconds}.{nanoseconds:09d}"


def _pose_matrix(pose: Any) -> torch.Tensor:
    if isinstance(pose, torch.Tensor):
        matrix = pose
    else:
        matrix = None
        for attribute in (
            "as_matrix",
            "matrix",
            "transformation_matrix",
            "tensor",
        ):
            if not hasattr(pose, attribute):
                continue
            candidate = getattr(pose, attribute)
            matrix = candidate() if callable(candidate) else candidate
            if matrix is not None:
                break
        if matrix is None:
            try:
                matrix = torch.as_tensor(pose)
            except (TypeError, ValueError, RuntimeError) as error:
                raise TypeError(
                    "pose must be a 3x4 or 4x4 tensor, or expose a matrix-like attribute"
                ) from error

    if not isinstance(matrix, torch.Tensor):
        try:
            matrix = torch.as_tensor(matrix)
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError("pose matrix could not be converted to a tensor") from error
    if matrix.shape not in {(3, 4), (4, 4)}:
        raise ValueError(f"pose must have shape (3, 4) or (4, 4), got {tuple(matrix.shape)}")
    if matrix.dtype.is_complex:
        raise TypeError("pose matrix must be real-valued")
    matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("pose matrix must contain only finite values")
    if matrix.shape == (4, 4):
        expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype)
        if not torch.allclose(matrix[3], expected_bottom, atol=1e-6, rtol=0.0):
            raise ValueError("pose matrix has an invalid homogeneous bottom row")
    return matrix[:3]


def rotation_matrix_to_quaternion_xyzw(rotation: torch.Tensor) -> tuple[float, ...]:
    """Convert a 3x3 rotation matrix to a normalized ``(x, y, z, w)`` quaternion."""

    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {tuple(rotation.shape)}")
    rotation = rotation.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(rotation).all()):
        raise ValueError("rotation must contain only finite values")

    # The branch on the largest diagonal term is stable around 180 degree
    # rotations where the trace-based expression has a small denominator.
    r00, r01, r02 = (float(value) for value in rotation[0])
    r10, r11, r12 = (float(value) for value in rotation[1])
    r20, r21, r22 = (float(value) for value in rotation[2])
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = 2.0 * math.sqrt(max(0.0, trace + 1.0))
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + r00 - r11 - r22))
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + r11 - r00 - r22))
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + r22 - r00 - r11))
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= torch.finfo(torch.float64).eps:
        raise ValueError("rotation matrix produced a zero-length quaternion")
    qx, qy, qz, qw = (component / norm for component in (qx, qy, qz, qw))

    # q and -q encode the same orientation.  A non-negative scalar component
    # gives deterministic files for most rotations.
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return qx, qy, qz, qw


def pose_to_translation_quaternion(
    T_world_cam: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Extract translation and an ``xyzw`` quaternion from ``T_world_cam``."""

    matrix = _pose_matrix(T_world_cam)
    translation = tuple(float(value) for value in matrix[:, 3])
    quaternion = rotation_matrix_to_quaternion_xyzw(matrix[:, :3])
    return translation, quaternion  # type: ignore[return-value]


def format_tum_pose(timestamp_ns: int, T_world_cam: Any) -> str:
    """Return one newline-terminated TUM trajectory record."""

    translation, quaternion = pose_to_translation_quaternion(T_world_cam)
    values = " ".join(f"{value:.17g}" for value in (*translation, *quaternion))
    return f"{format_timestamp_seconds(timestamp_ns)} {values}\n"


class TrajectoryWriter(AbstractContextManager["TrajectoryWriter"]):
    """Write strictly time-ordered ``T_world_cam`` poses in TUM format."""

    def __init__(
        self,
        destination: str | PathLike[str] | TextIO,
        *,
        flush: bool = False,
    ) -> None:
        self._owns_stream = not hasattr(destination, "write")
        if self._owns_stream:
            path = Path(destination)  # type: ignore[arg-type]
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream: TextIO = path.open("w", encoding="utf-8", newline="\n")
        else:
            self._stream = destination  # type: ignore[assignment]
        self.flush = flush
        self._last_timestamp_ns: int | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed or self._stream.closed

    def write(self, timestamp_ns: int, T_world_cam: Any) -> None:
        """Append a pose, rejecting duplicate or decreasing timestamps."""

        if self.closed:
            raise ValueError("cannot write to a closed trajectory")
        timestamp_ns = _validate_timestamp_ns(timestamp_ns)
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise TimestampError(
                f"timestamp_ns {timestamp_ns} must be greater than {self._last_timestamp_ns}"
            )
        self._stream.write(format_tum_pose(timestamp_ns, T_world_cam))
        self._last_timestamp_ns = timestamp_ns
        if self.flush:
            self._stream.flush()

    def write_result(self, timestamp_ns: int, result: Any) -> None:
        """Append ``result.T_world_cam`` from a frontend result object."""

        try:
            pose = result.T_world_cam
        except AttributeError as error:
            raise TypeError("result must expose T_world_cam") from error
        self.write(timestamp_ns, pose)

    def flush_stream(self) -> None:
        if not self.closed:
            self._stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_stream:
            self._stream.close()
        else:
            self._stream.flush()
        self._closed = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def write_tum_trajectory(
    destination: str | PathLike[str] | TextIO,
    poses: Iterable[tuple[int, Any]],
) -> None:
    """Write an iterable of ``(timestamp_ns, T_world_cam)`` pairs."""

    with TrajectoryWriter(destination) as writer:
        for timestamp_ns, pose in poses:
            writer.write(timestamp_ns, pose)


__all__ = [
    "TrajectoryWriter",
    "format_timestamp_seconds",
    "format_tum_pose",
    "pose_to_translation_quaternion",
    "rotation_matrix_to_quaternion_xyzw",
    "write_tum_trajectory",
]
