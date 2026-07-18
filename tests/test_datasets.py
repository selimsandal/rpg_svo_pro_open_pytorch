from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from svo_torch.datasets import (
    BenchmarkDataset,
    DatasetFormatError,
    EurocDataset,
    ImageDimensionError,
    ImageDirectoryDataset,
    TimestampError,
    load_grayscale_image,
    open_image_dataset,
    seconds_to_nanoseconds,
)


class CameraStub:
    width = 3
    height = 2


def save_image(path: Path, values: list[int] | None = None, size: tuple[int, int] = (3, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", size)
    if values is not None:
        image.putdata(values)
    image.save(path)


def test_grayscale_decode_is_uint8_tensor_without_numpy(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    image = Image.new("RGB", (2, 1))
    image.putdata([(255, 0, 0), (0, 255, 0)])
    image.save(image_path)

    decoded = load_grayscale_image(image_path, (2, 1))

    assert decoded.dtype == torch.uint8
    assert decoded.shape == (1, 2)
    assert decoded.is_contiguous()
    assert decoded.tolist() == [[76, 150]]


def test_calibrated_image_dimensions_are_enforced(tmp_path: Path) -> None:
    image_path = tmp_path / "1.png"
    save_image(image_path, size=(4, 2))
    dataset = ImageDirectoryDataset(tmp_path, camera=CameraStub())

    with pytest.raises(ImageDimensionError, match="calibration"):
        _ = dataset[0]


def test_image_directory_uses_numeric_filename_timestamps(tmp_path: Path) -> None:
    save_image(tmp_path / "frame_20.png", [20] * 6)
    save_image(tmp_path / "frame_3.png", [3] * 6)
    dataset = ImageDirectoryDataset(tmp_path, camera=CameraStub())

    samples = list(dataset)

    assert [sample.timestamp_ns for sample in samples] == [3, 20]
    assert [sample.image[0, 0].item() for sample in samples] == [3, 20]
    timestamp_ns, image = samples[0]
    assert timestamp_ns == 3
    assert image.shape == (2, 3)


def test_image_directory_can_generate_a_regular_clock(tmp_path: Path) -> None:
    save_image(tmp_path / "image_b.png")
    save_image(tmp_path / "image_a.png")

    dataset = ImageDirectoryDataset(
        tmp_path, expected_size=(3, 2), period_ns=50, start_timestamp_ns=100
    )

    assert [sample.path.name for sample in dataset] == ["image_a.png", "image_b.png"]
    assert [sample.timestamp_ns for sample in dataset] == [100, 150]


def test_benchmark_manifest_parses_decimal_seconds_exactly(tmp_path: Path) -> None:
    save_image(tmp_path / "data" / "images" / "first.png", [1] * 6)
    save_image(tmp_path / "data" / "images" / "second.png", [2] * 6)
    manifest = tmp_path / "data" / "images.txt"
    manifest.write_text(
        "# id seconds image\n"
        "7 1403636579.123456789 images/first.png\n"
        "8 1403636579.123456790 images/second.png\n",
        encoding="utf-8",
    )

    dataset = BenchmarkDataset(tmp_path, camera=CameraStub())

    assert [sample.frame_id for sample in dataset] == [7, 8]
    assert [sample.timestamp_ns for sample in dataset] == [
        1_403_636_579_123_456_789,
        1_403_636_579_123_456_790,
    ]


def test_benchmark_rejects_duplicate_timestamps_before_decoding(tmp_path: Path) -> None:
    manifest = tmp_path / "images.txt"
    manifest.write_text("1 1.0 missing.png\n2 1.0 also-missing.png\n", encoding="utf-8")

    with pytest.raises(TimestampError, match="must be greater"):
        BenchmarkDataset(manifest)


def test_euroc_cam0_layout_and_auto_detection(tmp_path: Path) -> None:
    camera_root = tmp_path / "mav0" / "cam0"
    save_image(camera_root / "data" / "100.png", [10] * 6)
    save_image(camera_root / "data" / "200.png", [20] * 6)
    (camera_root / "data.csv").write_text(
        "#timestamp [ns],filename\n100,100.png\n200,200.png\n", encoding="utf-8"
    )

    explicit = EurocDataset(tmp_path, camera=CameraStub())
    detected = open_image_dataset(tmp_path, camera=CameraStub())

    assert [sample.timestamp_ns for sample in explicit] == [100, 200]
    assert [sample.path.name for sample in detected] == ["100.png", "200.png"]


def test_bad_euroc_timestamp_and_empty_directory_report_format_errors(tmp_path: Path) -> None:
    camera_root = tmp_path / "mav0" / "cam0"
    camera_root.mkdir(parents=True)
    (camera_root / "data.csv").write_text("not-an-int,image.png\n", encoding="utf-8")

    with pytest.raises(TimestampError, match="invalid nanosecond"):
        EurocDataset(tmp_path)
    with pytest.raises(DatasetFormatError, match="no supported images"):
        ImageDirectoryDataset(camera_root)


def test_seconds_parser_preserves_precision_and_checks_int64() -> None:
    assert seconds_to_nanoseconds("1.000000001") == 1_000_000_001
    assert seconds_to_nanoseconds("0.0000000019") == 1
    with pytest.raises(TimestampError, match="non-negative"):
        seconds_to_nanoseconds("-0.0000000001")
    with pytest.raises(TimestampError, match="int64"):
        seconds_to_nanoseconds("9223372036.854775808")
