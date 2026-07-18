"""Evaluate monocular trajectories against a TUM-format reference.

The evaluator uses a one-to-one nearest-timestamp association followed by an
Umeyama Sim(3) alignment from each estimate into the reference frame.  It has
no NumPy or SciPy dependency; all pose mathematics is performed with torch.

Example::

    uv run python examples/rpg_svo_pro_open/evaluate_trajectory.py \
      groundtruth.txt cpp=stamped_traj_estimate.txt torch=trajectory.txt \
      --images data/images.txt --json comparison.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class TrajectoryError(ValueError):
    """Base class for trajectory input and evaluation errors."""


class TrajectoryFormatError(TrajectoryError):
    """Raised when an input file is not a valid ordered TUM trajectory."""


class TrajectoryEvaluationError(TrajectoryError):
    """Raised when valid trajectories cannot produce a meaningful comparison."""


@dataclass(frozen=True, slots=True)
class PoseTrajectory:
    """An ordered trajectory using exact decimal timestamps and world poses."""

    path: Path
    timestamps: tuple[Decimal, ...]
    positions: Tensor
    rotations: Tensor

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True, slots=True)
class TimestampAssociation:
    """One-to-one timestamp matches ordered by reference timestamp."""

    reference_indices: tuple[int, ...]
    estimate_indices: tuple[int, ...]
    offsets_seconds: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.reference_indices)


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    """A mapping ``target = scale * rotation @ source + translation``."""

    scale: Tensor
    rotation: Tensor
    translation: Tensor

    def apply(self, points: Tensor) -> Tensor:
        return self.scale * (points @ self.rotation.transpose(-1, -2)) + self.translation


def _parse_decimal(token: str, source: str) -> Decimal:
    try:
        value = Decimal(token)
    except InvalidOperation as error:
        raise TrajectoryFormatError(f"{source}: invalid timestamp {token!r}") from error
    if not value.is_finite():
        raise TrajectoryFormatError(f"{source}: timestamp must be finite")
    if value < 0:
        raise TrajectoryFormatError(f"{source}: timestamp must be non-negative")
    return value


def _quaternion_xyzw_to_matrix(quaternions: Tensor) -> Tensor:
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape [N,4]")
    norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    if bool((norms <= torch.finfo(quaternions.dtype).eps).any()):
        raise TrajectoryFormatError("pose quaternion must have non-zero norm")
    x, y, z, w = (quaternions / norms).unbind(dim=-1)
    two = quaternions.new_tensor(2.0)
    return torch.stack(
        (
            1.0 - two * (y.square() + z.square()),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x.square() + z.square()),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def read_tum_trajectory(path: str | Path, *, minimum_poses: int = 3) -> PoseTrajectory:
    """Read ``timestamp tx ty tz qx qy qz qw`` records with strict validation."""

    trajectory_path = Path(path)
    timestamps: list[Decimal] = []
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    seen_timestamps: set[Decimal] = set()

    with trajectory_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            source = f"{trajectory_path}:{line_number}"
            fields = line.split()
            if len(fields) != 8:
                raise TrajectoryFormatError(
                    f"{source}: expected 8 fields "
                    "(timestamp tx ty tz qx qy qz qw), "
                    f"got {len(fields)}"
                )
            timestamp = _parse_decimal(fields[0], source)
            if timestamp in seen_timestamps:
                raise TrajectoryFormatError(f"{source}: duplicate timestamp {fields[0]}")
            if timestamps and timestamp < timestamps[-1]:
                raise TrajectoryFormatError(f"{source}: timestamps must be strictly increasing")
            seen_timestamps.add(timestamp)
            try:
                values = [float(token) for token in fields[1:]]
            except ValueError as error:
                raise TrajectoryFormatError(f"{source}: pose fields must be numeric") from error
            if not all(math.isfinite(value) for value in values):
                raise TrajectoryFormatError(f"{source}: pose fields must be finite")
            quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
            if quaternion_norm <= sys.float_info.epsilon:
                raise TrajectoryFormatError(f"{source}: pose quaternion must have non-zero norm")
            timestamps.append(timestamp)
            positions.append(values[:3])
            quaternions.append(values[3:])

    if len(timestamps) < minimum_poses:
        raise TrajectoryFormatError(
            f"{trajectory_path}: expected at least {minimum_poses} poses, got {len(timestamps)}"
        )
    position_tensor = torch.tensor(positions, dtype=torch.float64)
    quaternion_tensor = torch.tensor(quaternions, dtype=torch.float64)
    return PoseTrajectory(
        path=trajectory_path,
        timestamps=tuple(timestamps),
        positions=position_tensor,
        rotations=_quaternion_xyzw_to_matrix(quaternion_tensor),
    )


def read_benchmark_image_count(path: str | Path) -> int:
    """Count and validate records in an original SVO ``data/images.txt`` file."""

    manifest_path = Path(path)
    count = 0
    seen_ids: set[int] = set()
    seen_timestamps: set[Decimal] = set()
    previous_timestamp: Decimal | None = None
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            source = f"{manifest_path}:{line_number}"
            fields = line.split()
            if len(fields) < 3:
                raise TrajectoryFormatError(
                    f"{source}: expected id, timestamp, and at least one image path"
                )
            try:
                frame_id = int(fields[0])
            except ValueError as error:
                raise TrajectoryFormatError(f"{source}: invalid frame id {fields[0]!r}") from error
            timestamp = _parse_decimal(fields[1], source)
            if frame_id in seen_ids:
                raise TrajectoryFormatError(f"{source}: duplicate frame id {frame_id}")
            if timestamp in seen_timestamps:
                raise TrajectoryFormatError(f"{source}: duplicate timestamp {fields[1]}")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise TrajectoryFormatError(
                    f"{source}: image timestamps must be strictly increasing"
                )
            seen_ids.add(frame_id)
            seen_timestamps.add(timestamp)
            previous_timestamp = timestamp
            count += 1
    if count == 0:
        raise TrajectoryFormatError(f"{manifest_path}: manifest contains no image records")
    return count


def _association_tolerance(value: float | Decimal) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("association tolerance must be a non-negative finite number")
    try:
        tolerance = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("association tolerance must be a non-negative finite number") from error
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("association tolerance must be a non-negative finite number")
    return tolerance


def associate_trajectories(
    reference: PoseTrajectory,
    estimate: PoseTrajectory,
    max_difference_seconds: float | Decimal,
) -> TimestampAssociation:
    """Associate unique poses by the smallest timestamp difference within a bound."""

    tolerance = _association_tolerance(max_difference_seconds)
    candidates: list[tuple[Decimal, int, int]] = []
    for estimate_index, timestamp in enumerate(estimate.timestamps):
        first = bisect.bisect_left(reference.timestamps, timestamp - tolerance)
        stop = bisect.bisect_right(reference.timestamps, timestamp + tolerance)
        for reference_index in range(first, stop):
            difference = abs(reference.timestamps[reference_index] - timestamp)
            candidates.append((difference, estimate_index, reference_index))

    candidates.sort()
    used_reference: set[int] = set()
    used_estimate: set[int] = set()
    selected: list[tuple[int, int, Decimal]] = []
    for difference, estimate_index, reference_index in candidates:
        if reference_index in used_reference or estimate_index in used_estimate:
            continue
        used_reference.add(reference_index)
        used_estimate.add(estimate_index)
        selected.append((reference_index, estimate_index, difference))
    selected.sort()
    return TimestampAssociation(
        reference_indices=tuple(item[0] for item in selected),
        estimate_indices=tuple(item[1] for item in selected),
        offsets_seconds=tuple(float(item[2]) for item in selected),
    )


def estimate_similarity_umeyama(source: Tensor, target: Tensor) -> SimilarityTransform:
    """Estimate the least-squares Sim(3) mapping from ``source`` to ``target``."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target positions must have equal [N,3] shape")
    if source.shape[0] < 3:
        raise TrajectoryEvaluationError("Sim(3) alignment requires at least 3 matched poses")
    if not bool(torch.isfinite(source).all() and torch.isfinite(target).all()):
        raise TrajectoryEvaluationError("Sim(3) positions must be finite")

    source = source.to(dtype=torch.float64, device="cpu")
    target = target.to(dtype=torch.float64, device="cpu")
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = source_centered.square().sum() / source.shape[0]
    if float(source_variance) <= torch.finfo(source.dtype).eps:
        raise TrajectoryEvaluationError("estimate positions have zero variance")

    covariance = target_centered.transpose(0, 1) @ source_centered / source.shape[0]
    left, singular_values, right_h = torch.linalg.svd(covariance)
    rank_tolerance = 3.0 * torch.finfo(source.dtype).eps * singular_values[0]
    if int((singular_values > rank_tolerance).sum()) < 2:
        raise TrajectoryEvaluationError(
            "associated positions are degenerate; Sim(3) needs a non-collinear trajectory"
        )
    correction = torch.ones(3, dtype=source.dtype)
    if float(torch.linalg.det(left @ right_h)) < 0.0:
        correction[-1] = -1.0
    rotation = left @ torch.diag(correction) @ right_h
    scale = (singular_values * correction).sum() / source_variance
    translation = target_mean - scale * (rotation @ source_mean)
    if not bool(
        torch.isfinite(scale)
        and torch.isfinite(rotation).all()
        and torch.isfinite(translation).all()
        and scale > 0
    ):
        raise TrajectoryEvaluationError("Umeyama alignment produced an invalid transform")
    return SimilarityTransform(scale, rotation, translation)


def _rotation_angles(rotation_matrices: Tensor) -> Tensor:
    """Return stable principal rotation angles in radians."""

    cosine = ((torch.diagonal(rotation_matrices, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    skew_vector = torch.stack(
        (
            rotation_matrices[..., 2, 1] - rotation_matrices[..., 1, 2],
            rotation_matrices[..., 0, 2] - rotation_matrices[..., 2, 0],
            rotation_matrices[..., 1, 0] - rotation_matrices[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1)
    return torch.atan2(sine, cosine)


def _rmse(values: Tensor) -> float:
    return float(torch.sqrt(values.square().mean()))


def _one_second_pairs(
    timestamps: tuple[Decimal, ...], tolerance: Decimal
) -> tuple[tuple[int, int], ...]:
    target_delta = Decimal("1")
    pairs: list[tuple[int, int]] = []
    for first_index, timestamp in enumerate(timestamps[:-1]):
        target = timestamp + target_delta
        insertion = bisect.bisect_left(timestamps, target, lo=first_index + 1)
        candidates = [
            index for index in (insertion - 1, insertion) if first_index < index < len(timestamps)
        ]
        if not candidates:
            continue
        second_index = min(candidates, key=lambda index: (abs(timestamps[index] - target), index))
        if abs(timestamps[second_index] - target) <= tolerance:
            pairs.append((first_index, second_index))
    return tuple(pairs)


def _relative_pose_errors(
    reference_positions: Tensor,
    reference_rotations: Tensor,
    estimate_positions: Tensor,
    estimate_rotations: Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor]:
    translation_errors: list[Tensor] = []
    rotation_errors: list[Tensor] = []
    for first, second in pairs:
        reference_rotation_relative = (
            reference_rotations[first].transpose(0, 1) @ reference_rotations[second]
        )
        estimate_rotation_relative = (
            estimate_rotations[first].transpose(0, 1) @ estimate_rotations[second]
        )
        reference_translation_relative = reference_rotations[first].transpose(0, 1) @ (
            reference_positions[second] - reference_positions[first]
        )
        estimate_translation_relative = estimate_rotations[first].transpose(0, 1) @ (
            estimate_positions[second] - estimate_positions[first]
        )
        translation_errors.append(
            torch.linalg.vector_norm(estimate_translation_relative - reference_translation_relative)
        )
        rotation_errors.append(
            reference_rotation_relative.transpose(0, 1) @ estimate_rotation_relative
        )
    if not pairs:
        empty = torch.empty(0, dtype=torch.float64)
        return empty, empty
    return torch.stack(translation_errors), _rotation_angles(torch.stack(rotation_errors))


def evaluate_estimate(
    reference: PoseTrajectory,
    estimate: PoseTrajectory,
    *,
    name: str,
    max_difference_seconds: float | Decimal,
    coverage_denominator: int | None = None,
) -> dict[str, Any]:
    """Associate, Sim(3)-align, and evaluate one estimate trajectory."""

    tolerance = _association_tolerance(max_difference_seconds)
    association = associate_trajectories(reference, estimate, tolerance)
    if len(association) < 3:
        raise TrajectoryEvaluationError(
            f"{name}: only {len(association)} timestamp matches within {tolerance}s; "
            "at least 3 are required"
        )
    reference_indices = torch.tensor(association.reference_indices, dtype=torch.long)
    estimate_indices = torch.tensor(association.estimate_indices, dtype=torch.long)
    reference_positions = reference.positions[reference_indices]
    reference_rotations = reference.rotations[reference_indices]
    estimate_positions = estimate.positions[estimate_indices]
    estimate_rotations = estimate.rotations[estimate_indices]

    similarity = estimate_similarity_umeyama(estimate_positions, reference_positions)
    aligned_positions = similarity.apply(estimate_positions)
    aligned_rotations = similarity.rotation[None] @ estimate_rotations
    translation_errors = torch.linalg.vector_norm(aligned_positions - reference_positions, dim=-1)
    absolute_rotation_errors = _rotation_angles(
        reference_rotations.transpose(-1, -2) @ aligned_rotations
    )

    matched_timestamps = tuple(
        reference.timestamps[index] for index in association.reference_indices
    )
    rpe_pairs = _one_second_pairs(matched_timestamps, tolerance)
    rpe_translation, rpe_rotation = _relative_pose_errors(
        reference_positions,
        reference_rotations,
        aligned_positions,
        aligned_rotations,
        rpe_pairs,
    )
    denominator = len(reference) if coverage_denominator is None else coverage_denominator
    if denominator <= 0:
        raise ValueError("coverage denominator must be positive")
    result: dict[str, Any] = {
        "name": name,
        "path": str(estimate.path),
        "estimate_poses": len(estimate),
        "matched_poses": len(association),
        "coverage": len(association) / denominator,
        "association": {
            "max_offset_seconds": max(association.offsets_seconds),
            "median_offset_seconds": float(
                torch.quantile(torch.tensor(association.offsets_seconds, dtype=torch.float64), 0.5)
            ),
        },
        "sim3": {
            "scale": float(similarity.scale),
            "rotation": similarity.rotation.tolist(),
            "translation": similarity.translation.tolist(),
        },
        "ate_sim3": {
            "rmse": _rmse(translation_errors),
            "median": float(torch.quantile(translation_errors, 0.5)),
            "max": float(translation_errors.max()),
        },
        "rotation_error": {
            "rmse_deg": math.degrees(_rmse(absolute_rotation_errors)),
        },
        "rpe_1s": None,
    }
    if rpe_pairs:
        result["rpe_1s"] = {
            "pairs": len(rpe_pairs),
            "translation_rmse": _rmse(rpe_translation),
            "rotation_rmse_deg": math.degrees(_rmse(rpe_rotation)),
        }
    return result


def evaluate_files(
    reference_path: str | Path,
    estimates: Sequence[tuple[str, str | Path]],
    *,
    images_path: str | Path | None = None,
    max_difference_seconds: float | Decimal = 0.02,
) -> dict[str, Any]:
    """Evaluate named trajectory files and return a JSON-serializable report."""

    if not estimates:
        raise ValueError("at least one estimate trajectory is required")
    names = [name for name, _ in estimates]
    if any(not name for name in names):
        raise ValueError("estimate names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("estimate names must be unique")
    tolerance = _association_tolerance(max_difference_seconds)
    reference = read_tum_trajectory(reference_path)
    coverage_denominator = (
        read_benchmark_image_count(images_path) if images_path is not None else len(reference)
    )
    coverage_source = str(Path(images_path)) if images_path is not None else "reference poses"
    evaluated = [
        evaluate_estimate(
            reference,
            read_tum_trajectory(path),
            name=name,
            max_difference_seconds=tolerance,
            coverage_denominator=coverage_denominator,
        )
        for name, path in estimates
    ]
    return {
        "reference": str(reference.path),
        "reference_poses": len(reference),
        "coverage_denominator": coverage_denominator,
        "coverage_source": coverage_source,
        "max_timestamp_difference_seconds": float(tolerance),
        "estimates": evaluated,
    }


def _estimate_specification(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("estimate must use NAME=PATH syntax")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("estimate must have a non-empty NAME and PATH")
    return name, Path(path)


def _nonnegative_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sim(3)-align and evaluate TUM monocular trajectories"
    )
    parser.add_argument("reference", type=Path, help="TUM-format reference trajectory")
    parser.add_argument(
        "estimates",
        nargs="+",
        type=_estimate_specification,
        metavar="NAME=PATH",
        help="named TUM-format estimate trajectory (repeatable)",
    )
    parser.add_argument(
        "--images",
        "--benchmark-images",
        type=Path,
        help="optional original SVO data/images.txt used as the coverage denominator",
    )
    parser.add_argument(
        "--max-time-difference",
        "--max-association-seconds",
        type=_nonnegative_finite_float,
        default=0.02,
        metavar="SECONDS",
        help="maximum nearest-pose timestamp difference (default: 0.02)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        metavar="PATH",
        help="also write the complete report as JSON; use '-' for stdout only",
    )
    return parser


def _print_human_report(report: dict[str, Any]) -> None:
    print(
        f"reference: {report['reference']} ({report['reference_poses']} poses); "
        f"coverage denominator: {report['coverage_denominator']} "
        f"({report['coverage_source']})"
    )
    for estimate in report["estimates"]:
        ate = estimate["ate_sim3"]
        rotation = estimate["rotation_error"]
        print(
            f"{estimate['name']}: matched={estimate['matched_poses']} "
            f"coverage={estimate['coverage']:.1%} "
            f"ATE(Sim3) rmse={ate['rmse']:.6g} median={ate['median']:.6g} "
            f"max={ate['max']:.6g} rotation_rmse={rotation['rmse_deg']:.6g}deg"
        )
        rpe = estimate["rpe_1s"]
        if rpe is None:
            print("  RPE@1s unavailable (no timestamp pairs within tolerance)")
        else:
            print(
                f"  RPE@1s pairs={rpe['pairs']} "
                f"translation_rmse={rpe['translation_rmse']:.6g} "
                f"rotation_rmse={rpe['rotation_rmse_deg']:.6g}deg"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate_files(
            args.reference,
            args.estimates,
            images_path=args.images,
            max_difference_seconds=args.max_time_difference,
        )
    except (OSError, TrajectoryError, ValueError) as error:
        parser.error(str(error))

    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.json_output == Path("-"):
        print(serialized)
    else:
        _print_human_report(report)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
