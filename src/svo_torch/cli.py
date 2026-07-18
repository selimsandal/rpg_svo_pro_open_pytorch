"""Command-line entry points for calibration inspection and monocular runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .camera import load_camera_rig
from .config import SVOConfig
from .datasets import camera_image_size, open_image_dataset
from .io import TrajectoryWriter
from .odometry import MonoSVO, Stage, UpdateResult


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"unsupported torch dtype {name!r}") from error
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("dtype must be float32 or float64")
    return dtype


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _camera_summary(camera: Any, index: int) -> dict[str, Any]:
    width, height = camera_image_size(camera)
    summary: dict[str, Any] = {
        "index": index,
        "model": type(camera).__name__,
        "width": width,
        "height": height,
    }
    for attribute in ("label", "distortion_model"):
        if hasattr(camera, attribute):
            value = getattr(camera, attribute)
            if callable(value):
                value = value()
            if value is not None:
                summary[attribute] = str(value)
    return summary


def _load_rig(path: str | Path, *, device: torch.device, dtype: torch.dtype) -> Any:
    # Keyword arguments are the public API.  The fallback makes the CLI useful
    # with early ports and small third-party compatibility loaders that only
    # accepted the calibration path.
    try:
        return load_camera_rig(path, device=device, dtype=dtype)
    except TypeError as first_error:
        try:
            return load_camera_rig(path)
        except TypeError:
            raise first_error from None


def _rig_cameras(rig: Any) -> tuple[Any, ...]:
    try:
        cameras = tuple(rig.cameras)
    except (AttributeError, TypeError) as error:
        raise TypeError("camera loader returned a rig without an iterable cameras field") from error
    if not cameras:
        raise ValueError("camera rig contains no cameras")
    return cameras


def _config_from_args(args: argparse.Namespace) -> SVOConfig:
    config = SVOConfig.from_yaml_files(args.config) if args.config else SVOConfig()
    if args.device is not None:
        config.device = args.device
    if args.dtype is not None:
        config.dtype = args.dtype
    config.validate()
    return config


def _resolve_run_inputs(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Path, Path]:
    inputs = [Path(item) for item in args.inputs]
    calibration_option = Path(args.calibration) if args.calibration else None
    if calibration_option is not None:
        if len(inputs) != 1:
            parser.error("run with --calibration accepts exactly one dataset input")
        return inputs[0], calibration_option
    if len(inputs) != 2:
        parser.error("run requires DATASET --calibration RIG_YAML, or two positional paths")

    first, second = inputs
    yaml_suffixes = {".yaml", ".yml"}
    if first.suffix.lower() in yaml_suffixes and second.suffix.lower() not in yaml_suffixes:
        return second, first
    if second.suffix.lower() in yaml_suffixes:
        return first, second
    parser.error("could not identify calibration YAML in the two positional inputs")


def _inspect_calibration(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    rig = _load_rig(args.calibration, device=device, dtype=dtype)
    cameras = _rig_cameras(rig)
    output: dict[str, Any] = {
        "calibration": str(Path(args.calibration)),
        "num_cameras": len(cameras),
        "cameras": [_camera_summary(camera, index) for index, camera in enumerate(cameras)],
    }
    transforms = getattr(rig, "T_body_camera", None)
    if isinstance(transforms, torch.Tensor):
        output["T_body_camera"] = transforms.detach().cpu().tolist()
    print(json.dumps(output, indent=2))
    return 0


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    source, calibration = _resolve_run_inputs(args, parser)
    config = _config_from_args(args)
    device = config.torch_device()
    dtype = config.torch_dtype()
    rig = _load_rig(calibration, device=device, dtype=dtype)
    camera = _rig_cameras(rig)[0]
    dataset = open_image_dataset(
        source,
        dataset_format=args.format,
        camera=camera,
        period_ns=args.period_ns,
        start_timestamp_ns=args.start_timestamp_ns,
    )

    frontend = MonoSVO(camera, config)
    frontend.start()
    processed = 0
    poses_written = 0
    trajectory_path = Path(args.trajectory)
    with TrajectoryWriter(trajectory_path) as trajectory:
        for sample in dataset:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            result = frontend.process(sample.image.to(device=device), sample.timestamp_ns)
            processed += 1
            pose = result.T_world_cam
            pose_is_usable = (
                pose is not None
                and result.stage == Stage.TRACKING
                and result.update != UpdateResult.FAILURE
            )
            if pose_is_usable:
                trajectory.write(sample.timestamp_ns, pose)
                poses_written += 1
            if not args.quiet:
                stage = _enum_value(getattr(result, "stage", "unknown"))
                quality = _enum_value(getattr(result, "quality", "unknown"))
                observations = getattr(result, "num_observations", 0)
                keyframe = " keyframe" if getattr(result, "is_keyframe", False) else ""
                print(
                    f"{sample.timestamp_ns} stage={stage} quality={quality} "
                    f"observations={observations}{keyframe}"
                )

    print(
        f"processed {processed} frames; wrote {poses_written} poses to {trajectory_path}",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="svo-torch",
        description="Tensor-native semi-direct monocular visual odometry",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-calib", help="parse a camera rig and print a JSON summary"
    )
    inspect_parser.add_argument("calibration", help="SVO camera rig YAML")
    inspect_parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    inspect_parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    inspect_parser.set_defaults(handler=_inspect_calibration)

    run_parser = subparsers.add_parser("run", help="run monocular SVO on an image sequence")
    run_parser.add_argument(
        "inputs",
        nargs="+",
        metavar="PATH",
        help="dataset path, optionally followed or preceded by calibration YAML",
    )
    run_parser.add_argument(
        "-c", "--calibration", help="camera rig YAML (recommended over positional form)"
    )
    run_parser.add_argument(
        "--config",
        action="append",
        help="SVO runtime YAML; repeat to apply camera-specific overrides in order",
    )
    run_parser.add_argument(
        "--format",
        choices=("auto", "directory", "benchmark", "euroc"),
        default="auto",
        help="input layout (default: detect automatically)",
    )
    run_parser.add_argument("-o", "--trajectory", default="trajectory.txt", help="TUM output path")
    run_parser.add_argument("--period-ns", type=int, help="clock period for generic directories")
    run_parser.add_argument("--start-timestamp-ns", type=int, default=0)
    run_parser.add_argument("--max-frames", type=int, help="stop after this many images")
    run_parser.add_argument("--device", help="override config device, e.g. cpu or cuda")
    run_parser.add_argument("--dtype", choices=("float32", "float64"), help="override config dtype")
    run_parser.add_argument("--quiet", action="store_true", help="suppress per-frame status")
    run_parser.set_defaults(handler=_run, parser=run_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``svo-torch`` command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.max_frames is not None and args.max_frames < 0:
            parser.error("--max-frames must be non-negative")
        return args.handler(args, args.parser)
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover - console-script convenience
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
