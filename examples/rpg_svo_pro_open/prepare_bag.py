#!/usr/bin/env python3
"""Download and extract the two real bags documented by RPG SVO Pro.

The output uses SVO Pro's ``data/images.txt`` benchmark layout so the C++
reference and ``svo-torch`` can consume exactly the same decoded PNG files.
``rosbags`` is used only at this example-data boundary; the visual frontend
remains implemented solely with Python and PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True, slots=True)
class BagExample:
    name: str
    filename: str
    url: str
    size_bytes: int
    sha256: str
    image_topic: str
    image_width: int
    image_height: int
    calibration: str
    mask: str | None = None
    groundtruth_topic: str | None = None


EXAMPLES = {
    "pinhole": BagExample(
        name="pinhole",
        filename="airground_rig_s3_2013-03-18_21-38-48.bag",
        url=(
            "https://download.ifi.uzh.ch/rpg/web/datasets/airground_rig_s3_2013-03-18_21-38-48.bag"
        ),
        size_bytes=1_772_552_296,
        sha256="f84eb9c5931f6da9e3fe54197c761dcab7992fde61d6151b9513ca39e1b26f5f",
        image_topic="camera/image_raw",
        image_width=752,
        image_height=480,
        calibration="calib/svo_test_pinhole.yaml",
        groundtruth_topic="Rig",
    ),
    "fisheye": BagExample(
        name="fisheye",
        filename="test_fisheye.bag",
        url="https://download.ifi.uzh.ch/rpg/web/datasets/test_fisheye.bag",
        size_bytes=742_137_657,
        sha256="0b1d8b489309932a8bd12ac44ce41a44c1694ac5ff08c2a1a654327613968f45",
        image_topic="/camera/image_raw",
        image_width=752,
        image_height=480,
        calibration="calib/bluefox_25000826_fisheye.yaml",
        mask="calib/25000826_fisheye_mask.png",
    ),
}

_ROOT = Path(__file__).resolve().parent
_CHUNK_SIZE = 8 * 1024 * 1024


def _ros_timestamp_ns(message: object) -> int:
    try:
        stamp = message.header.stamp  # type: ignore[attr-defined]
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except AttributeError as error:
        raise ValueError("ROS message does not contain header.stamp") from error


def _format_seconds(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return f"{seconds}.{nanoseconds:09d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bag(path: Path, example: BagExample) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != example.size_bytes:
        raise ValueError(
            f"{path}: expected {example.size_bytes} bytes for the official bag, got {size}"
        )
    actual_hash = _sha256(path)
    if actual_hash != example.sha256:
        raise ValueError(f"{path}: SHA-256 mismatch; expected {example.sha256}, got {actual_hash}")


def _download(example: BagExample, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / example.filename
    if destination.is_file():
        _validate_bag(destination, example)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"downloading {example.url}", file=sys.stderr)
    with urllib.request.urlopen(example.url) as response, partial.open("wb") as output:
        copied = 0
        while chunk := response.read(_CHUNK_SIZE):
            output.write(chunk)
            copied += len(chunk)
            print(
                f"\r{example.name}: {copied / (1024**2):.0f} / "
                f"{example.size_bytes / (1024**2):.0f} MiB",
                end="",
                file=sys.stderr,
            )
    print(file=sys.stderr)
    partial.replace(destination)
    _validate_bag(destination, example)
    return destination


def _decode_mono8(message: object) -> Image.Image:
    try:
        width = int(message.width)  # type: ignore[attr-defined]
        height = int(message.height)  # type: ignore[attr-defined]
        step = int(message.step)  # type: ignore[attr-defined]
        encoding = str(message.encoding).lower()  # type: ignore[attr-defined]
        data = memoryview(message.data).cast("B")  # type: ignore[attr-defined]
    except AttributeError as error:
        raise ValueError("image topic did not contain sensor_msgs/Image messages") from error
    if encoding not in {"mono8", "8uc1"}:
        raise ValueError(f"expected mono8 image data, got encoding {encoding!r}")
    if width <= 0 or height <= 0 or step < width or len(data) < height * step:
        raise ValueError(
            f"invalid image storage: width={width}, height={height}, step={step}, bytes={len(data)}"
        )
    if step == width:
        pixels = bytes(data[: height * step])
    else:
        pixels = b"".join(bytes(data[row * step : row * step + width]) for row in range(height))
    return Image.frombytes("L", (width, height), pixels)


def _write_groundtruth(
    reader: object,
    example: BagExample,
    destination: Path,
    first_timestamp_ns: int,
    last_timestamp_ns: int,
) -> int:
    if example.groundtruth_topic is None:
        return 0
    connections = [
        connection
        for connection in reader.connections  # type: ignore[attr-defined]
        if connection.topic == example.groundtruth_topic
    ]
    if len(connections) != 1:
        raise ValueError(
            f"expected one {example.groundtruth_topic!r} ground-truth topic, "
            f"found {len(connections)}"
        )
    lines = ["# timestamp tx ty tz qx qy qz qw\n"]
    previous_timestamp = -1
    for connection, _, rawdata in reader.messages(connections=connections):  # type: ignore[attr-defined]
        message = reader.deserialize(rawdata, connection.msgtype)  # type: ignore[attr-defined]
        timestamp_ns = _ros_timestamp_ns(message)
        if timestamp_ns < first_timestamp_ns or timestamp_ns > last_timestamp_ns:
            continue
        if timestamp_ns <= previous_timestamp:
            raise ValueError("ground-truth header timestamps are not strictly increasing")
        previous_timestamp = timestamp_ns
        pose = message.pose
        position = pose.position
        orientation = pose.orientation
        values = (
            position.x,
            position.y,
            position.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        lines.append(
            f"{_format_seconds(timestamp_ns)} "
            + " ".join(f"{float(value):.17g}" for value in values)
            + "\n"
        )
    destination.write_text("".join(lines), encoding="utf-8")
    return len(lines) - 1


def extract_example(
    bag: Path,
    example: BagExample,
    output_root: Path,
    *,
    start_frame: int,
    max_frames: int | None,
    force: bool,
) -> Path:
    """Extract one official bag into a shared C++/PyTorch benchmark dataset."""

    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError(
            "the real-bag example requires the optional 'examples' dependencies; "
            "run it with `uv run --extra examples`"
        ) from error

    destination = output_root / example.name
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass --force to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{example.name}-", dir=destination.parent))
    try:
        data_directory = temporary / "data"
        image_directory = data_directory / "img"
        image_directory.mkdir(parents=True)
        calibration_source = _ROOT / example.calibration
        shutil.copy2(calibration_source, temporary / "calib.yaml")
        if example.mask is not None:
            mask_source = _ROOT / example.mask
            shutil.copy2(mask_source, temporary / mask_source.name)

        manifest = ["# id timestamp image_name\n"]
        timestamps: list[int] = []
        with AnyReader([bag]) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.topic == example.image_topic
            ]
            if len(connections) != 1:
                raise ValueError(
                    f"expected one {example.image_topic!r} image topic, found {len(connections)}"
                )
            for source_index, (connection, _, rawdata) in enumerate(
                reader.messages(connections=connections)
            ):
                if source_index < start_frame:
                    continue
                if max_frames is not None and len(timestamps) >= max_frames:
                    break
                message = reader.deserialize(rawdata, connection.msgtype)
                timestamp_ns = _ros_timestamp_ns(message)
                if timestamps and timestamp_ns <= timestamps[-1]:
                    raise ValueError("image header timestamps are not strictly increasing")
                image = _decode_mono8(message)
                expected_size = (example.image_width, example.image_height)
                if image.size != expected_size:
                    raise ValueError(
                        f"expected {expected_size[0]}x{expected_size[1]} images, "
                        f"got {image.width}x{image.height}"
                    )
                relative_path = Path("img") / f"frame_{source_index:06d}.png"
                image.save(data_directory / relative_path, compress_level=1)
                frame_id = source_index
                manifest.append(
                    f"{frame_id} {_format_seconds(timestamp_ns)} {relative_path.as_posix()}\n"
                )
                timestamps.append(timestamp_ns)
            if not timestamps:
                raise ValueError("the requested frame range contains no images")
            groundtruth_count = _write_groundtruth(
                reader,
                example,
                data_directory / "stamped_groundtruth.txt",
                timestamps[0],
                timestamps[-1],
            )

        (data_directory / "images.txt").write_text("".join(manifest), encoding="utf-8")
        metadata = {
            **asdict(example),
            "bag": str(bag.resolve()),
            "start_frame": start_frame,
            "frames": len(timestamps),
            "first_timestamp_ns": timestamps[0],
            "last_timestamp_ns": timestamps[-1],
            "groundtruth_poses": groundtruth_count,
        }
        (temporary / "example.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the two real ROS-bag examples documented by RPG SVO Pro"
    )
    parser.add_argument("example", choices=(*EXAMPLES, "all"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("real-data"),
        help="dataset parent directory (default: ./real-data)",
    )
    parser.add_argument("--bag", type=Path, help="existing official bag (single example only)")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "svo-torch",
        help="download cache",
    )
    parser.add_argument("--download", action="store_true", help="download a missing official bag")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="number of images to extract; use 0 for the complete bag (default: 500)",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing dataset")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.start_frame < 0 or args.max_frames < 0:
        raise SystemExit("--start-frame and --max-frames must be non-negative")
    if args.example == "all" and args.bag is not None:
        raise SystemExit("--bag can only be used with one example")
    names = list(EXAMPLES) if args.example == "all" else [args.example]
    max_frames = None if args.max_frames == 0 else args.max_frames

    for name in names:
        example = EXAMPLES[name]
        if args.bag is not None:
            bag = args.bag
            _validate_bag(bag, example)
        else:
            bag = args.cache / example.filename
            if not bag.is_file() and not args.download:
                raise SystemExit(
                    f"{bag} is missing; pass --bag PATH or --download (source: {example.url})"
                )
            bag = _download(example, args.cache)
        destination = extract_example(
            bag,
            example,
            args.output,
            start_frame=args.start_frame,
            max_frames=max_frames,
            force=args.force,
        )
        print(f"prepared {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
