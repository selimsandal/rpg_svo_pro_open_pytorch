import torch
import torch.nn.functional as F

from svo_torch.camera import PinholeCamera
from svo_torch.depth import (
    DepthSeeds,
    EpipolarMatcher,
    compute_tau,
    inverse_depth_variance,
    triangulate_depth,
    update_filter_gaussian,
    update_filter_vogiatzis,
    zmssd_score,
)
from svo_torch.image import build_image_pyramid, prepare_image


def _texture(size: int = 96) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    value = torch.sin(0.17 * x) + 0.7 * torch.cos(0.13 * y) + 0.4 * torch.sin(0.11 * (x + y))
    return (value - value.amin()) / (value.amax() - value.amin())


def _translate_x(image: torch.Tensor, dx: float) -> torch.Tensor:
    source = prepare_image(image)
    height, width = image.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=image.dtype),
        torch.arange(width, dtype=image.dtype),
        indexing="ij",
    )
    grid = torch.stack(
        (2.0 * (x - dx) / (width - 1) - 1.0, 2.0 * y / (height - 1) - 1.0),
        dim=-1,
    )[None]
    return F.grid_sample(source, grid, align_corners=True)[0, 0]


def test_depth_seed_initialization_and_gaussian_update() -> None:
    seeds = DepthSeeds.initialize(2, depth_mean=4.0, depth_min=2.0, dtype=torch.float64)
    assert torch.allclose(seeds.mu, torch.full((2,), 0.25, dtype=torch.float64))
    assert torch.allclose(seeds.sigma2, torch.full((2,), 0.25 / 36.0, dtype=torch.float64))
    updated, valid = update_filter_gaussian(seeds, 0.3, 0.0025)
    expected_mu = (seeds.sigma2 * 0.3 + 0.0025 * seeds.mu) / (seeds.sigma2 + 0.0025)
    expected_sigma2 = seeds.sigma2 * 0.0025 / (seeds.sigma2 + 0.0025)
    assert valid.all()
    assert torch.allclose(updated.mu, expected_mu)
    assert torch.allclose(updated.sigma2, expected_sigma2)
    assert torch.equal(updated.a, seeds.a)
    assert torch.equal(updated.b, seeds.b)


def test_vogiatzis_update_matches_reference_numbers() -> None:
    seeds = DepthSeeds(
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
    )
    updated, valid = update_filter_vogiatzis(seeds, 0.3, 0.0025, 0.5)
    assert valid.item()
    assert torch.allclose(updated.mu, torch.tensor([0.2746997844805311], dtype=torch.float64))
    assert torch.allclose(updated.sigma2, torch.tensor([0.005437955129730346], dtype=torch.float64))
    assert torch.allclose(updated.a, torch.tensor([10.138559182068283], dtype=torch.float64))
    assert torch.allclose(updated.b, torch.tensor([9.914169884780964], dtype=torch.float64))


def test_inverse_depth_variance_tau_and_zmssd() -> None:
    variance = inverse_depth_variance(torch.tensor(4.0), torch.tensor(0.2))
    expected = (0.5 * (1 / 3.8 - 1 / 4.2)) ** 2
    assert torch.allclose(variance, torch.tensor(expected))

    transform = torch.eye(4)
    transform[0, 3] = -0.2
    bearing = torch.tensor([0.0, 0.0, 1.0])
    tau = compute_tau(transform, bearing, 4.0, 0.002)
    assert torch.isfinite(tau)
    assert tau > 0

    reference = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64.0
    assert torch.allclose(zmssd_score(reference, reference + 0.2), torch.tensor(0.0), atol=1e-5)


def test_triangulation_recovers_reference_depth() -> None:
    camera = PinholeCamera(96, 96, 80.0, 80.0, 47.5, 47.5)
    pixel = torch.tensor([[42.0, 45.0]])
    bearing_ref = camera.unproject(pixel)
    depth_true = torch.tensor([4.0])
    point_ref = bearing_ref * depth_true[:, None]
    transform = torch.eye(4)
    transform[0, 3] = 0.1
    point_cur = point_ref + transform[:3, 3]
    bearing_cur = F.normalize(point_cur, dim=-1)
    depth, valid = triangulate_depth(transform[None], bearing_ref, bearing_cur)
    assert valid.item()
    assert torch.allclose(depth, depth_true, atol=1e-3)


def test_triangulation_rejects_points_behind_either_camera() -> None:
    transform = torch.eye(4)[None]
    transform[0, :3, 3] = torch.tensor([1.0, 0.0, 2.0])
    bearing_ref = torch.tensor([[0.0, 0.0, 1.0]])
    bearing_cur = F.normalize(torch.tensor([[1.0, 0.0, 1.0]]), dim=-1)
    depth, valid = triangulate_depth(transform, bearing_ref, bearing_cur)
    assert not valid.item()
    assert depth.item() < 0


def test_epipolar_matcher_finds_planar_translation() -> None:
    image = _texture()
    depth_true = 4.0
    camera = PinholeCamera(96, 96, 80.0, 80.0, 47.5, 47.5)
    transform = torch.eye(4)
    transform[0, 3] = 0.1
    disparity = 80.0 * 0.1 / depth_true
    current = _translate_x(image, disparity)
    ref_pyramid = build_image_pyramid(image, 2)
    cur_pyramid = build_image_pyramid(current, 2)
    pixels = torch.tensor([[36.0, 36.0], [60.0, 40.0], [48.0, 65.0]])
    matcher = EpipolarMatcher(camera, patch_size=8, max_steps=32, step=0.2, max_score=0.01)
    result = matcher.match(
        ref_pyramid,
        cur_pyramid,
        pixels,
        transform,
        inverse_depth_min=0.15,
        inverse_depth_max=0.35,
    )
    assert result.valid.all()
    assert torch.allclose(result.depth, torch.full_like(result.depth, depth_true), atol=0.3)
    assert torch.allclose(result.pixels[:, 0], pixels[:, 0] + disparity, atol=0.2)
