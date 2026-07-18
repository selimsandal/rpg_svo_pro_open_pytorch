import torch

from svo_torch.image import (
    build_image_pyramid,
    prepare_image,
    sample_image,
    sample_patches,
)


def test_prepare_image_accepts_uint8_and_float_255() -> None:
    integer = torch.tensor([[0, 64], [128, 255]], dtype=torch.uint8)
    expected = integer.float()[None, None] / 255.0
    assert torch.equal(prepare_image(integer), expected)
    assert torch.equal(prepare_image(integer.float()), expected)
    assert prepare_image(integer).shape == (1, 1, 2, 2)


def test_floor_sized_box_pyramid_has_exact_values() -> None:
    image = torch.arange(35, dtype=torch.float32).reshape(5, 7) / 34.0
    pyramid = build_image_pyramid(image, 3)
    assert [item.shape for item in pyramid] == [(1, 1, 5, 7), (1, 1, 2, 3), (1, 1, 1, 1)]
    expected_level_1 = (
        torch.tensor([[4.0, 6.0, 8.0], [18.0, 20.0, 22.0]], dtype=torch.float32) / 34.0
    )
    assert torch.allclose(pyramid[1][0, 0], expected_level_1)
    assert torch.allclose(pyramid[2][0, 0, 0, 0], expected_level_1[:2, :2].mean())


def test_bilinear_sampling_and_patch_sampling_are_differentiable() -> None:
    source = torch.arange(36, dtype=torch.float64).reshape(6, 6).requires_grad_()
    image = prepare_image(source, dtype=torch.float64, normalize=False)
    pixel = torch.tensor([[2.5, 1.5]], dtype=torch.float64)
    value = sample_image(image, pixel)
    assert torch.allclose(value, torch.tensor([11.5], dtype=torch.float64))
    patch = sample_patches(image, torch.tensor([[3.0, 3.0]], dtype=torch.float64), 4)
    assert patch.shape == (1, 4, 4)
    (value.sum() + patch.sum()).backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
