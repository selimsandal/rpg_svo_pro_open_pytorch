import torch

from svo_torch.features import GridFeatureDetector, shi_tomasi_score
from svo_torch.frame import CORNER, EDGELET
from svo_torch.image import build_image_pyramid


def _corner_scene(size: int = 96) -> torch.Tensor:
    image = torch.zeros((size, size), dtype=torch.float32)
    image[14:38, 14:38] = 1.0
    image[52:82, 50:84] = 0.8
    image[20:76, 44:48] = 0.5
    return image


def test_shi_tomasi_distinguishes_corner_from_flat_and_edge() -> None:
    image = _corner_scene(64)
    score, _, _ = shi_tomasi_score(image, window_size=7)
    corner = score[0, 0, 14, 14]
    edge = score[0, 0, 14, 25]
    flat = score[0, 0, 5, 5]
    assert corner > edge
    assert corner > flat
    assert flat == 0


def test_grid_detector_returns_distributed_corners_and_edgelets() -> None:
    pyramid = build_image_pyramid(_corner_scene(), 3)
    detector = GridFeatureDetector(
        max_features=24,
        grid_size=12,
        quality_level=0.02,
        border=5,
        edgelet_ratio=0.5,
        max_level=2,
    )
    features = detector.detect(pyramid)
    assert 4 <= len(features) <= 24
    assert features.pixels.shape == (len(features), 2)
    assert torch.isfinite(features.scores).all()
    assert torch.allclose(
        torch.linalg.vector_norm(features.gradients, dim=-1),
        torch.ones(len(features)),
    )
    assert (features.kinds == CORNER).any()
    assert (features.kinds == EDGELET).any()
    cells = torch.floor(features.pixels / 12).long()
    cell_ids = cells[:, 1] * 8 + cells[:, 0]
    assert torch.unique(cell_ids).numel() == len(features)


def test_detector_honors_level_zero_mask() -> None:
    pyramid = build_image_pyramid(_corner_scene(), 2)
    mask = torch.zeros((96, 96), dtype=torch.uint8)
    mask[:, :48] = 255
    detector = GridFeatureDetector(30, 10, 0.01, 5, 0.4, max_level=1)
    features = detector.detect(pyramid, mask)
    assert len(features) > 0
    assert (features.pixels[:, 0] < 48).all()


def test_uniform_image_has_no_features() -> None:
    pyramid = build_image_pyramid(torch.full((64, 64), 0.4), 2)
    detector = GridFeatureDetector(20, 12, 0.01, 5, 0.5, max_level=1)
    features = detector.detect(pyramid)
    assert len(features) == 0
