from pathlib import Path

import pytest
import torch

from svo_torch.config import SVOConfig


def test_load_native_config() -> None:
    path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = SVOConfig.from_yaml(path)
    assert config.max_features == 180
    assert config.patch_size == 8
    assert config.torch_dtype() is torch.float32


def test_original_parameter_aliases() -> None:
    config = SVOConfig.from_mapping({"max_fts": 120, "max_n_kfs": 5, "img_align_max_level": 3})
    assert config.max_features == 120
    assert config.max_keyframes == 5
    assert config.alignment_max_level == 3


def test_detection_and_alignment_can_request_different_pyramid_depths() -> None:
    config = SVOConfig(n_pyr_levels=3, alignment_min_level=2, alignment_max_level=4)
    config.validate()
    assert config.alignment_max_level == 4
    assert config.image_pyramid_levels == 5


def test_original_quality_and_reprojector_aliases() -> None:
    config = SVOConfig.from_mapping(
        {
            "quality_min_fts": 20,
            "quality_max_drop_fts": 90,
            "reprojector_max_n_kfs": 4,
        }
    )
    assert config.quality_min_features == 20
    assert config.quality_max_feature_drop == 90
    assert config.reprojector_max_keyframes == 4


def test_float64_auto_avoids_mps_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert SVOConfig(device="auto", dtype="float64").torch_device() == torch.device("cpu")
    with pytest.raises(RuntimeError, match="does not support float64"):
        SVOConfig(device="mps", dtype="float64").torch_device()
