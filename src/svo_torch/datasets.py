"""Portable monocular image-sequence readers.

The readers in this module deliberately keep the I/O boundary small: images are
decoded with Pillow and returned as CPU ``torch.uint8`` tensors with shape
``(height, width)``.  No NumPy dependency is needed.  Timestamp parsing and
ordering are validated before any image is processed so malformed datasets fail
early and with a useful error message.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

_INT64_MAX = 2**63 - 1
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".tif", ".tiff", ".webp"})
_TRAILING_INTEGER = re.compile(r"(-?\d+)$")
_NATURAL_PART = re.compile(r"(\d+)")


class DatasetFormatError(ValueError):
    """Raised when an image sequence manifest is malformed."""


class TimestampError(DatasetFormatError):
    """Raised when timestamps are invalid or not strictly increasing."""


class ImageDimensionError(ValueError):
    """Raised when an image does not match its calibrated dimensions."""


@dataclass(frozen=True, slots=True)
class ImageSample:
    """One decoded monocular frame.

    Iteration over a sample yields ``(timestamp_ns, image)`` for convenient use
    in simple processing loops, while ``path`` and ``frame_id`` remain available
    as named metadata.
    """

    timestamp_ns: int
    image: torch.Tensor
    path: Path
    frame_id: int | None = None

    def __iter__(self) -> Iterator[int | torch.Tensor]:
        yield self.timestamp_ns
        yield self.image


@dataclass(frozen=True, slots=True)
class _ImageEntry:
    timestamp_ns: int
    path: Path
    frame_id: int | None
    source: str


def _coerce_dimension(value: Any, name: str) -> int:
    if callable(value):
        value = value()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"camera {name} must be scalar, got shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool):
        raise TypeError(f"camera {name} must be an integer")
    try:
        dimension = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"camera {name} must be an integer, got {value!r}") from error
    if dimension <= 0:
        raise ValueError(f"camera {name} must be positive, got {dimension}")
    return dimension


def camera_image_size(camera: Any) -> tuple[int, int]:
    """Return ``(width, height)`` from a camera-like object.

    The native camera classes expose ``width`` and ``height``.  A few aliases
    are accepted here so the dataset readers are also useful with lightweight
    test doubles and compatibility wrappers.
    """

    if camera is None:
        raise TypeError("camera must not be None")

    if hasattr(camera, "image_size"):
        image_size = camera.image_size
        if callable(image_size):
            image_size = image_size()
        if len(image_size) != 2:
            raise TypeError("camera image_size must contain (width, height)")
        return (
            _coerce_dimension(image_size[0], "width"),
            _coerce_dimension(image_size[1], "height"),
        )

    width = None
    height = None
    for name in ("width", "image_width", "imageWidth"):
        if hasattr(camera, name):
            width = getattr(camera, name)
            break
    for name in ("height", "image_height", "imageHeight"):
        if hasattr(camera, name):
            height = getattr(camera, name)
            break
    if width is None or height is None:
        raise TypeError("camera must expose width/height or image_size=(width, height)")
    return _coerce_dimension(width, "width"), _coerce_dimension(height, "height")


def _resolve_expected_size(
    camera: Any | None, expected_size: tuple[int, int] | None
) -> tuple[int, int] | None:
    camera_size = camera_image_size(camera) if camera is not None else None
    if expected_size is None:
        return camera_size
    if len(expected_size) != 2:
        raise TypeError("expected_size must contain (width, height)")
    size = (
        _coerce_dimension(expected_size[0], "width"),
        _coerce_dimension(expected_size[1], "height"),
    )
    if camera_size is not None and camera_size != size:
        raise ValueError(f"camera dimensions {camera_size} disagree with expected_size {size}")
    return size


def _validate_timestamp_ns(value: int, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimestampError(f"{source}: timestamp must be an integer number of nanoseconds")
    if not 0 <= value <= _INT64_MAX:
        raise TimestampError(f"{source}: timestamp {value} is outside non-negative int64 range")
    return value


def seconds_to_nanoseconds(value: str, source: str = "timestamp") -> int:
    """Parse a decimal-seconds token into an integer nanosecond timestamp.

    ``Decimal`` avoids the precision loss that occurs when EuRoC-era timestamps
    are routed through a Python float.  Fractions below one nanosecond are
    truncated, matching the original benchmark reader's integer conversion.
    """

    try:
        seconds = Decimal(value)
    except InvalidOperation as error:
        raise TimestampError(f"{source}: invalid seconds timestamp {value!r}") from error
    if not seconds.is_finite():
        raise TimestampError(f"{source}: timestamp must be finite")
    if seconds < 0:
        raise TimestampError(f"{source}: timestamp must be non-negative")
    timestamp_ns = int(seconds * Decimal(1_000_000_000))
    return _validate_timestamp_ns(timestamp_ns, source)


def load_grayscale_image(
    path: str | Path, expected_size: tuple[int, int] | None = None
) -> torch.Tensor:
    """Decode an image as a contiguous ``torch.uint8`` ``(H, W)`` tensor."""

    image_path = Path(path)
    try:
        with Image.open(image_path) as encoded:
            grayscale = encoded.convert("L")
            width, height = grayscale.size
            if expected_size is not None and (width, height) != expected_size:
                raise ImageDimensionError(
                    f"{image_path}: image size {(width, height)} does not match "
                    f"calibration {expected_size}"
                )
            # bytearray supplies a writable buffer, avoiding the warning emitted
            # by torch.frombuffer(bytes(...)).  clone() gives the tensor its own
            # lifetime after the temporary buffer goes out of scope.
            pixels = bytearray(grayscale.tobytes())
    except (FileNotFoundError, IsADirectoryError):
        raise
    except UnidentifiedImageError as error:
        raise DatasetFormatError(f"{image_path}: unsupported or corrupt image") from error
    return torch.frombuffer(pixels, dtype=torch.uint8).reshape(height, width).clone()


class ImageSequence(Sequence[ImageSample]):
    """A repeatable, lazily decoded sequence of image entries."""

    def __init__(
        self,
        entries: Sequence[_ImageEntry],
        *,
        camera: Any | None = None,
        expected_size: tuple[int, int] | None = None,
    ) -> None:
        self._entries = tuple(entries)
        self.expected_size = _resolve_expected_size(camera, expected_size)
        self._validate_entries()

    def _validate_entries(self) -> None:
        previous: _ImageEntry | None = None
        for entry in self._entries:
            _validate_timestamp_ns(entry.timestamp_ns, entry.source)
            if previous is not None and entry.timestamp_ns <= previous.timestamp_ns:
                raise TimestampError(
                    f"{entry.source}: timestamp {entry.timestamp_ns} must be greater than "
                    f"{previous.timestamp_ns} from {previous.source}"
                )
            previous = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int | slice) -> ImageSample | list[ImageSample]:
        if isinstance(index, slice):
            return [self._decode(entry) for entry in self._entries[index]]
        return self._decode(self._entries[index])

    def __iter__(self) -> Iterator[ImageSample]:
        for entry in self._entries:
            yield self._decode(entry)

    def _decode(self, entry: _ImageEntry) -> ImageSample:
        image = load_grayscale_image(entry.path, self.expected_size)
        return ImageSample(entry.timestamp_ns, image, entry.path, entry.frame_id)


def _default_filename_timestamp(path: Path) -> int:
    match = _TRAILING_INTEGER.search(path.stem)
    if match is None:
        raise TimestampError(
            f"{path}: filename must end in an integer timestamp, or period_ns must be provided"
        )
    return _validate_timestamp_ns(int(match.group(1)), str(path))


def _natural_path_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    """Sort ``frame2`` before ``frame10`` without imposing a name pattern."""

    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in _NATURAL_PART.split(path.name)
    )


class ImageDirectoryDataset(ImageSequence):
    """Read all supported images in a directory.

    By default, the final integer in each filename stem is interpreted as an
    already-nanosecond timestamp.  Supplying ``period_ns`` instead generates a
    regular clock beginning at ``start_timestamp_ns``.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        camera: Any | None = None,
        expected_size: tuple[int, int] | None = None,
        period_ns: int | None = None,
        start_timestamp_ns: int = 0,
        timestamp_parser: Callable[[Path], int] | None = None,
        recursive: bool = False,
        suffixes: Sequence[str] = tuple(sorted(_IMAGE_SUFFIXES)),
    ) -> None:
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(root)
        allowed_suffixes = {suffix.lower() for suffix in suffixes}
        paths = [
            path
            for path in (root.rglob("*") if recursive else root.iterdir())
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        ]
        if not paths:
            raise DatasetFormatError(f"{root}: directory contains no supported images")

        if period_ns is not None:
            period_ns = _validate_timestamp_ns(period_ns, "period_ns")
            if period_ns == 0:
                raise TimestampError("period_ns must be greater than zero")
            start_timestamp_ns = _validate_timestamp_ns(start_timestamp_ns, "start_timestamp_ns")
            paths.sort(key=_natural_path_key)
            timestamps = [start_timestamp_ns + index * period_ns for index in range(len(paths))]
        else:
            parser = timestamp_parser or _default_filename_timestamp
            parsed = [(parser(path), path) for path in paths]
            parsed.sort(key=lambda item: (item[0], item[1].name))
            timestamps = [timestamp for timestamp, _ in parsed]
            paths = [path for _, path in parsed]

        entries = [
            _ImageEntry(timestamp, path, index, str(path))
            for index, (timestamp, path) in enumerate(zip(timestamps, paths, strict=True))
        ]
        super().__init__(entries, camera=camera, expected_size=expected_size)


def _benchmark_paths(root_or_manifest: str | Path) -> tuple[Path, Path]:
    supplied = Path(root_or_manifest)
    if supplied.is_file():
        return supplied, supplied.parent
    candidates = (
        (supplied / "data" / "images.txt", supplied / "data"),
        (supplied / "images.txt", supplied),
    )
    for manifest, data_root in candidates:
        if manifest.is_file():
            return manifest, data_root
    raise FileNotFoundError(f"could not find data/images.txt under {supplied}")


class BenchmarkDataset(ImageSequence):
    """Read the original SVO benchmark ``data/images.txt`` format."""

    def __init__(
        self,
        root_or_manifest: str | Path,
        *,
        camera: Any | None = None,
        expected_size: tuple[int, int] | None = None,
        camera_index: int = 0,
    ) -> None:
        if camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        manifest, data_root = _benchmark_paths(root_or_manifest)
        entries: list[_ImageEntry] = []
        with manifest.open(encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                source = f"{manifest}:{line_number}"
                minimum_fields = 3 + camera_index
                if len(fields) < minimum_fields:
                    raise DatasetFormatError(
                        f"{source}: expected id, seconds, and image path for camera {camera_index}"
                    )
                try:
                    frame_id = int(fields[0])
                except ValueError as error:
                    raise DatasetFormatError(f"{source}: invalid frame id {fields[0]!r}") from error
                timestamp_ns = seconds_to_nanoseconds(fields[1], source)
                image_path = data_root / fields[2 + camera_index]
                entries.append(_ImageEntry(timestamp_ns, image_path, frame_id, source))
        if not entries:
            raise DatasetFormatError(f"{manifest}: manifest contains no image entries")
        super().__init__(entries, camera=camera, expected_size=expected_size)


def _euroc_paths(root_or_csv: str | Path, camera_index: int) -> tuple[Path, Path]:
    supplied = Path(root_or_csv)
    if supplied.is_file():
        return supplied, supplied.parent / "data"
    camera_name = f"cam{camera_index}"
    candidates = (
        supplied / "mav0" / camera_name,
        supplied / camera_name,
        supplied,
    )
    for camera_root in candidates:
        csv_path = camera_root / "data.csv"
        if csv_path.is_file():
            return csv_path, camera_root / "data"
    raise FileNotFoundError(f"could not find {camera_name}/data.csv under {supplied}")


class EurocDataset(ImageSequence):
    """Read EuRoC's ``mav0/cam0/data.csv`` camera stream."""

    def __init__(
        self,
        root_or_csv: str | Path,
        *,
        camera: Any | None = None,
        expected_size: tuple[int, int] | None = None,
        camera_index: int = 0,
    ) -> None:
        if camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        csv_path, image_root = _euroc_paths(root_or_csv, camera_index)
        entries: list[_ImageEntry] = []
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for line_number, row in enumerate(csv.reader(stream), start=1):
                if not row or not any(field.strip() for field in row):
                    continue
                if row[0].lstrip().startswith("#"):
                    continue
                source = f"{csv_path}:{line_number}"
                if len(row) < 2:
                    raise DatasetFormatError(f"{source}: expected timestamp_ns,filename")
                timestamp_token = row[0].strip()
                try:
                    timestamp_ns = int(timestamp_token)
                except ValueError as error:
                    raise TimestampError(
                        f"{source}: invalid nanosecond timestamp {timestamp_token!r}"
                    ) from error
                timestamp_ns = _validate_timestamp_ns(timestamp_ns, source)
                filename = row[1].strip()
                if not filename:
                    raise DatasetFormatError(f"{source}: image filename is empty")
                entries.append(
                    _ImageEntry(timestamp_ns, image_root / filename, len(entries), source)
                )
        if not entries:
            raise DatasetFormatError(f"{csv_path}: manifest contains no image entries")
        super().__init__(entries, camera=camera, expected_size=expected_size)


# Readable aliases for callers that prefer the full names.
BenchmarkImageDataset = BenchmarkDataset
EuRoCDataset = EurocDataset


def open_image_dataset(
    source: str | Path,
    *,
    dataset_format: str = "auto",
    camera: Any | None = None,
    expected_size: tuple[int, int] | None = None,
    period_ns: int | None = None,
    start_timestamp_ns: int = 0,
) -> ImageSequence:
    """Open a generic, benchmark, or EuRoC image source."""

    source_path = Path(source)
    normalized_format = dataset_format.lower()
    if normalized_format == "auto":
        if source_path.is_file() and source_path.name == "images.txt":
            normalized_format = "benchmark"
        elif source_path.is_file() and source_path.name == "data.csv":
            normalized_format = "euroc"
        elif (source_path / "data" / "images.txt").is_file() or (
            source_path / "images.txt"
        ).is_file():
            normalized_format = "benchmark"
        elif any(
            candidate.is_file()
            for candidate in (
                source_path / "mav0" / "cam0" / "data.csv",
                source_path / "cam0" / "data.csv",
                source_path / "data.csv",
            )
        ):
            normalized_format = "euroc"
        else:
            normalized_format = "directory"

    common = {"camera": camera, "expected_size": expected_size}
    if normalized_format in {"directory", "images", "dir"}:
        return ImageDirectoryDataset(
            source_path,
            period_ns=period_ns,
            start_timestamp_ns=start_timestamp_ns,
            **common,
        )
    if normalized_format in {"benchmark", "svo"}:
        return BenchmarkDataset(source_path, **common)
    if normalized_format in {"euroc", "euroc-cam0"}:
        return EurocDataset(source_path, **common)
    raise ValueError(
        f"unknown dataset format {dataset_format!r}; choose auto, directory, benchmark, or euroc"
    )


__all__ = [
    "BenchmarkDataset",
    "BenchmarkImageDataset",
    "DatasetFormatError",
    "EuRoCDataset",
    "EurocDataset",
    "ImageDimensionError",
    "ImageDirectoryDataset",
    "ImageSample",
    "ImageSequence",
    "TimestampError",
    "camera_image_size",
    "load_grayscale_image",
    "open_image_dataset",
    "seconds_to_nanoseconds",
]
