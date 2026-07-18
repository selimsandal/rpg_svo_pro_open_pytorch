from __future__ import annotations

import io
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from svo_torch.datasets import TimestampError
from svo_torch.io import (
    TrajectoryWriter,
    format_timestamp_seconds,
    format_tum_pose,
    pose_to_translation_quaternion,
    write_tum_trajectory,
)


def test_timestamp_format_keeps_all_nanoseconds() -> None:
    assert format_timestamp_seconds(0) == "0.000000000"
    assert format_timestamp_seconds(1_403_636_579_123_456_789) == ("1403636579.123456789")


def test_identity_pose_is_serialized_in_tum_xyzw_order() -> None:
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.25, -2.0, 3.5])

    fields = format_tum_pose(42, pose).split()

    assert fields[0] == "0.000000042"
    assert [float(value) for value in fields[1:]] == [1.25, -2.0, 3.5, 0, 0, 0, 1]


def test_rotation_matrix_converts_to_normalized_xyzw_quaternion() -> None:
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    translation, quaternion = pose_to_translation_quaternion(pose)

    assert translation == (0.0, 0.0, 0.0)
    assert quaternion == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))


def test_writer_requires_strictly_increasing_int64_timestamps() -> None:
    stream = io.StringIO()
    writer = TrajectoryWriter(stream)
    writer.write(10, torch.eye(4))

    with pytest.raises(TimestampError, match="greater"):
        writer.write(10, torch.eye(4))
    with pytest.raises(TimestampError, match="integer"):
        TrajectoryWriter(io.StringIO()).write(1.5, torch.eye(4))  # type: ignore[arg-type]

    writer.close()
    assert not stream.closed
    assert stream.getvalue().startswith("0.000000010 ")


def test_writer_accepts_result_and_creates_parent_directories(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "trajectory.txt"
    result = SimpleNamespace(T_world_cam=torch.eye(4))

    with TrajectoryWriter(output) as writer:
        writer.write_result(1, result)

    assert output.read_text(encoding="utf-8").split()[0] == "0.000000001"


def test_bulk_writer_and_pose_validation() -> None:
    stream = io.StringIO()
    pose = torch.eye(4)
    write_tum_trajectory(stream, [(1, pose), (2, pose)])
    assert len(stream.getvalue().splitlines()) == 2

    invalid = torch.eye(4)
    invalid[3, 3] = 0
    with pytest.raises(ValueError, match="bottom row"):
        format_tum_pose(3, invalid)
