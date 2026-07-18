import torch
import torch.nn.functional as F

from svo_torch.alignment import PyramidalPatchTracker, SparseImageAligner
from svo_torch.camera import PinholeCamera
from svo_torch.image import build_image_pyramid, prepare_image


def _textured_image(height: int = 96, width: int = 112) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    image = (
        0.8 * torch.sin(0.13 * x)
        + 0.7 * torch.cos(0.17 * y)
        + 0.45 * torch.sin(0.09 * (x + y))
        + 0.2 * torch.cos(0.21 * (x - y))
    )
    return ((image - image.amin()) / (image.amax() - image.amin())).float()


def _translate(image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    source = prepare_image(image)
    height, width = image.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=image.dtype),
        torch.arange(width, dtype=image.dtype),
        indexing="ij",
    )
    grid_x = 2.0 * (x - dx) / (width - 1) - 1.0
    grid_y = 2.0 * (y - dy) / (height - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)[None]
    return F.grid_sample(source, grid, align_corners=True)[0, 0]


def test_pyramidal_tracker_is_exact_on_identical_images() -> None:
    image = _textured_image()
    pyramid = build_image_pyramid(image, 3)
    pixels = torch.tensor([[35.0, 30.0], [72.0, 42.0], [55.0, 70.0]])
    tracker = PyramidalPatchTracker(8, 2, 0, 15, 0.01)
    result = tracker.track(pyramid, pyramid, pixels)
    assert result.status.all()
    assert torch.allclose(result.pixels, pixels, atol=1e-5)
    assert torch.all(result.error < 1e-6)


def test_pyramidal_tracker_recovers_translation() -> None:
    image = _textured_image()
    shift = torch.tensor([1.75, -1.25])
    current = _translate(image, float(shift[0]), float(shift[1]))
    ref_pyramid = build_image_pyramid(image, 3)
    cur_pyramid = build_image_pyramid(current, 3)
    pixels = torch.tensor([[38.0, 34.0], [75.0, 44.0], [58.0, 72.0]])
    initial = pixels + shift + torch.tensor([0.35, -0.25])
    tracker = PyramidalPatchTracker(8, 2, 0, 25, 0.005)
    result = tracker.track(ref_pyramid, cur_pyramid, pixels, initial)
    assert result.status.all()
    assert torch.allclose(result.pixels, pixels + shift, atol=0.12)
    assert torch.all(result.error < 0.01)


def test_sparse_image_aligner_identity_pose() -> None:
    image = _textured_image(80, 96)
    pyramid = build_image_pyramid(image, 2)
    camera = PinholeCamera(96, 80, 80.0, 80.0, 47.5, 39.5)
    pixels = torch.tensor([[28.0, 25.0], [48.0, 25.0], [68.0, 28.0], [34.0, 55.0], [62.0, 54.0]])
    depths = torch.full((pixels.shape[0],), 4.0)
    aligner = SparseImageAligner(
        camera, patch_size=8, max_level=1, min_level=0, max_iterations=4, min_features=3
    )
    result = aligner.align(pyramid, pyramid, pixels, depths, torch.eye(4))
    assert result.valid.all()
    assert result.converged
    assert result.error < 1e-6
    assert torch.allclose(result.T_cur_ref, torch.eye(4), atol=1e-5)


def test_sparse_image_aligner_builds_a_bounded_projection_jacobian(monkeypatch) -> None:
    image = _textured_image(80, 96)
    pyramid = build_image_pyramid(image, 2)
    camera = PinholeCamera(96, 80, 80.0, 80.0, 47.5, 39.5)
    pixels = torch.tensor([[28.0, 25.0], [48.0, 25.0], [68.0, 28.0], [34.0, 55.0], [62.0, 54.0]])
    depths = torch.full((pixels.shape[0],), 4.0)
    calls: list[tuple[torch.Size, str | None]] = []
    original_jacobian = torch.autograd.functional.jacobian

    def recording_jacobian(function, inputs, *args, **kwargs):
        calls.append((function(inputs).shape, kwargs.get("strategy")))
        return original_jacobian(function, inputs, *args, **kwargs)

    monkeypatch.setattr(torch.autograd.functional, "jacobian", recording_jacobian)
    aligner = SparseImageAligner(
        camera, patch_size=8, max_level=1, min_level=0, max_iterations=1, min_features=3
    )
    result = aligner.align(pyramid, pyramid, pixels, depths, torch.eye(4))

    assert result.valid.all()
    assert calls
    assert all(shape == (pixels.shape[0], 2) for shape, _ in calls)
    assert calls[0][1] == "forward-mode"


def test_tracker_accepts_uint8_pyramids_and_empty_batches() -> None:
    image = (_textured_image() * 255).to(torch.uint8)
    pyramid = [image[None, None]]
    tracker = PyramidalPatchTracker(8, 0, 0, 5, 0.01)
    pixels = torch.tensor([[45.0, 45.0]])
    result = tracker.track(pyramid, pyramid, pixels)
    assert result.status.item()
    empty = tracker.track(pyramid, pyramid, torch.empty((0, 2)))
    assert empty.pixels.shape == (0, 2)
    assert empty.status.shape == (0,)
