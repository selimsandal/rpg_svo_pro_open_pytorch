from __future__ import annotations

import json
import math
import runpy
from pathlib import Path
from typing import Any

import pytest
import torch

EVALUATOR = runpy.run_path(
    Path(__file__).parents[1] / "examples" / "rpg_svo_pro_open" / "evaluate_trajectory.py"
)
TrajectoryFormatError = EVALUATOR["TrajectoryFormatError"]
evaluate_files = EVALUATOR["evaluate_files"]
main = EVALUATOR["main"]
read_tum_trajectory = EVALUATOR["read_tum_trajectory"]


def _rotation_z(angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def _write_tum(
    path: Path,
    timestamps: torch.Tensor,
    positions: torch.Tensor,
    quaternion_xyzw: tuple[float, float, float, float],
) -> None:
    lines = []
    for timestamp, position in zip(timestamps.tolist(), positions.tolist(), strict=True):
        values = [timestamp, *position, *quaternion_xyzw]
        lines.append(" ".join(f"{value:.17g}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _known_similarity_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    timestamps = torch.arange(15, dtype=torch.float64) * 0.1
    parameter = timestamps
    reference_positions = torch.stack(
        (
            0.4 * parameter + 0.03 * parameter.square(),
            -0.2 * parameter + 0.08 * parameter.pow(3),
            0.15 * parameter.square() + 0.02 * parameter.pow(4),
        ),
        dim=-1,
    )
    angle = 0.47
    rotation = _rotation_z(angle)
    scale = 2.4
    translation = torch.tensor([1.2, -0.7, 0.35], dtype=torch.float64)
    estimate_positions = ((reference_positions - translation) @ rotation) / scale

    reference_path = tmp_path / "reference.txt"
    estimate_path = tmp_path / "estimate.txt"
    half_angle = 0.5 * angle
    _write_tum(
        reference_path,
        timestamps,
        reference_positions,
        (0.0, 0.0, math.sin(half_angle), math.cos(half_angle)),
    )
    _write_tum(
        estimate_path,
        timestamps + 0.004,
        estimate_positions,
        (0.0, 0.0, 0.0, 1.0),
    )

    images_path = tmp_path / "images.txt"
    images_path.write_text(
        "# id seconds image\n"
        + "".join(f"{index} {index * 0.1:.9f} img/{index}.png\n" for index in range(20)),
        encoding="utf-8",
    )
    expected = {
        "rotation": rotation,
        "scale": scale,
        "translation": translation,
        "matches": len(timestamps),
    }
    return reference_path, estimate_path, images_path, expected


def test_known_sim3_timestamp_association_orientation_and_rpe(tmp_path: Path) -> None:
    reference_path, estimate_path, images_path, expected = _known_similarity_fixture(tmp_path)
    report = evaluate_files(
        reference_path,
        [("torch", estimate_path)],
        images_path=images_path,
        max_difference_seconds=0.01,
    )

    result = report["estimates"][0]
    assert result["matched_poses"] == expected["matches"]
    assert result["coverage"] == pytest.approx(0.75)
    assert result["association"]["max_offset_seconds"] == pytest.approx(0.004)
    assert result["sim3"]["scale"] == pytest.approx(expected["scale"], abs=1e-12)
    torch.testing.assert_close(
        torch.tensor(result["sim3"]["rotation"], dtype=torch.float64),
        expected["rotation"],
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        torch.tensor(result["sim3"]["translation"], dtype=torch.float64),
        expected["translation"],
        rtol=0.0,
        atol=1e-12,
    )
    assert result["ate_sim3"]["rmse"] < 1e-12
    assert result["ate_sim3"]["median"] < 1e-12
    assert result["ate_sim3"]["max"] < 1e-12
    assert result["rotation_error"]["rmse_deg"] < 1e-10
    assert result["rpe_1s"] is not None
    assert result["rpe_1s"]["pairs"] == 5
    assert result["rpe_1s"]["translation_rmse"] < 1e-12
    assert result["rpe_1s"]["rotation_rmse_deg"] < 1e-10


def test_cli_writes_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reference_path, estimate_path, images_path, _ = _known_similarity_fixture(tmp_path)
    output_path = tmp_path / "reports" / "comparison.json"

    assert (
        main(
            [
                str(reference_path),
                f"pytorch={estimate_path}",
                "--images",
                str(images_path),
                "--max-time-difference",
                "0.01",
                "--json",
                str(output_path),
            ]
        )
        == 0
    )
    streams = capsys.readouterr()
    assert "pytorch: matched=15 coverage=75.0%" in streams.out
    assert streams.err == ""
    serialized = json.loads(output_path.read_text(encoding="utf-8"))
    assert serialized["coverage_denominator"] == 20
    assert serialized["estimates"][0]["rpe_1s"]["pairs"] == 5


def test_rejects_nonfinite_duplicate_and_too_short_inputs(tmp_path: Path) -> None:
    nonfinite = tmp_path / "nonfinite.txt"
    nonfinite.write_text(
        "0 nan 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    with pytest.raises(TrajectoryFormatError, match="pose fields must be finite"):
        read_tum_trajectory(nonfinite)

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text(
        "0 0 0 0 0 0 0 1\n0 1 0 0 0 0 0 1\n1 2 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    with pytest.raises(TrajectoryFormatError, match="duplicate timestamp"):
        read_tum_trajectory(duplicate)

    too_short = tmp_path / "too_short.txt"
    too_short.write_text(
        "0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    with pytest.raises(TrajectoryFormatError, match="at least 3 poses"):
        read_tum_trajectory(too_short)
