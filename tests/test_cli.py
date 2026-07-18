from __future__ import annotations

import json
import math
import runpy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from svo_torch.cli import main

_EXAMPLE = runpy.run_path(Path(__file__).parents[1] / "examples" / "synthetic_sequence.py")
demo_config = _EXAMPLE["demo_config"]
make_synthetic_sequence = _EXAMPLE["make_synthetic_sequence"]


def _write_sequence_fixture(tmp_path: Path, count: int = 4) -> tuple[object, Path, Path, Path]:
    sequence = make_synthetic_sequence(count=count)
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    for index, image in enumerate(sequence.images):
        pixels = image.mul(255.0).round().clamp(0, 255).to(torch.uint8).contiguous()
        encoded = Image.frombytes(
            "L",
            (sequence.camera.width, sequence.camera.height),
            bytes(pixels.reshape(-1).tolist()),
        )
        encoded.save(image_directory / f"{sequence.timestamp(index)}.png")

    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(
        yaml.safe_dump(
            {
                "cameras": [
                    {
                        "camera": {
                            "label": "synthetic_cam0",
                            "image_height": sequence.camera.height,
                            "image_width": sequence.camera.width,
                            "type": "pinhole",
                            "intrinsics": {"data": sequence.camera.intrinsics.tolist()},
                            "distortion": {"type": "none", "parameters": {"data": []}},
                        },
                        "T_B_C": {
                            "data": [
                                1,
                                0,
                                0,
                                0,
                                0,
                                1,
                                0,
                                0,
                                0,
                                0,
                                1,
                                0,
                                0,
                                0,
                                0,
                                1,
                            ]
                        },
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(demo_config())), encoding="utf-8")
    return sequence, image_directory, calibration_path, config_path


def test_inspect_and_run_rendered_sequence_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sequence, image_directory, calibration_path, config_path = _write_sequence_fixture(tmp_path)

    assert main(["inspect-calib", str(calibration_path)]) == 0
    inspected_streams = capsys.readouterr()
    inspected = json.loads(inspected_streams.out)
    assert inspected_streams.err == ""
    assert inspected["num_cameras"] == 1
    assert inspected["cameras"] == [
        {
            "index": 0,
            "model": "PinholeCamera",
            "width": sequence.camera.width,
            "height": sequence.camera.height,
            "label": "synthetic_cam0",
        }
    ]
    assert inspected["T_body_camera"] == [
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ]

    trajectory_path = tmp_path / "result" / "trajectory.txt"
    assert (
        main(
            [
                "run",
                str(image_directory),
                "--calibration",
                str(calibration_path),
                "--config",
                str(config_path),
                "--format",
                "directory",
                "--trajectory",
                str(trajectory_path),
            ]
        )
        == 0
    )
    run_streams = capsys.readouterr()
    status_lines = run_streams.out.splitlines()
    assert len(status_lines) == len(sequence.images)
    assert "stage=initializing" in status_lines[0]
    assert any("stage=tracking" in line for line in status_lines[1:])
    assert f"processed {len(sequence.images)} frames" in run_streams.err
    assert f"wrote {len(sequence.images) - 1} poses" in run_streams.err

    rows = [line.split() for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(sequence.images) - 1
    assert [row[0] for row in rows] == [
        "0.010000000",
        "0.020000000",
        "0.030000000",
    ]
    assert all(len(row) == 8 for row in rows)
    poses = [[float(value) for value in row[1:]] for row in rows]
    assert max(math.dist(pose[:3], [0.0, 0.0, 0.0]) for pose in poses) > 0.01
    assert all(
        math.isclose(math.sqrt(sum(value * value for value in pose[3:])), 1.0) for pose in poses
    )
