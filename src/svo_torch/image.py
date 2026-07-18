"""Tensor-only grayscale image preparation, pyramids, and bilinear sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def prepare_image(
    image: Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> Tensor:
    """Return a grayscale image as contiguous ``[1, 1, H, W]`` float data.

    Accepted layouts are ``[H, W]``, ``[1, H, W]``, ``[H, W, 1]``, and
    ``[1, 1, H, W]``.  Integer data is interpreted as its full dtype range;
    floating-point data above one is interpreted as 0--255 image data.  This
    deliberately keeps the public boundary permissive while giving all direct
    methods one unambiguous internal representation.
    """

    if not isinstance(image, Tensor):
        image = torch.as_tensor(image)
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        if image.shape[0] == 1:
            image = image[None]
        elif image.shape[-1] == 1:
            image = image.permute(2, 0, 1)[None]
        else:
            raise ValueError("a rank-3 image must have one grayscale channel")
    elif image.ndim == 4:
        if image.shape[:2] != (1, 1):
            raise ValueError("a rank-4 image must have shape [1, 1, H, W]")
    else:
        raise ValueError("image must have shape [H,W], [1,H,W], [H,W,1], or [1,1,H,W]")
    if image.shape[-2] < 1 or image.shape[-1] < 1:
        raise ValueError("image dimensions must be positive")
    if not dtype.is_floating_point:
        raise ValueError("internal image dtype must be floating point")

    source_dtype = image.dtype
    result = image.to(device=device, dtype=dtype)
    if normalize:
        if source_dtype == torch.bool:
            pass
        elif not source_dtype.is_floating_point:
            scale = float(torch.iinfo(source_dtype).max)
            result = result / scale
        elif result.numel() and bool((result.detach().amax() > 1.0).item()):
            result = result / 255.0
        result = result.clamp(0.0, 1.0)
    return result.contiguous()


def build_image_pyramid(image: Tensor, levels: int) -> list[Tensor]:
    """Build ``levels`` using floor-sized, non-overlapping 2x2 box averages."""

    if levels < 1:
        raise ValueError("levels must be positive")
    current = prepare_image(
        image,
        device=image.device if isinstance(image, Tensor) else None,
        dtype=(
            image.dtype
            if isinstance(image, Tensor) and image.dtype.is_floating_point
            else torch.float32
        ),
    )
    pyramid = [current]
    for _ in range(1, levels):
        height, width = current.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("requested pyramid has more levels than the image supports")
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        pyramid.append(current)
    return pyramid


def image_gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    """Central-difference gradients with one-sided replication at the border."""

    image = prepare_image(
        image,
        device=image.device,
        dtype=image.dtype if image.dtype.is_floating_point else torch.float32,
        normalize=not image.dtype.is_floating_point,
    )
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    dx = 0.5 * (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2])
    dy = 0.5 * (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1])
    return dx, dy


def _pixel_grid(pixels: Tensor, height: int, width: int) -> Tensor:
    if pixels.shape[-1] != 2:
        raise ValueError("pixels must end in an (x, y) coordinate")
    x = pixels[..., 0]
    y = pixels[..., 1]
    x = 2.0 * x / (width - 1) - 1.0 if width > 1 else torch.zeros_like(x)
    y = 2.0 * y / (height - 1) - 1.0 if height > 1 else torch.zeros_like(y)
    return torch.stack((x, y), dim=-1)


def sample_image(image: Tensor, pixels: Tensor, *, padding_mode: str = "zeros") -> Tensor:
    """Bilinearly sample an image at ``(x, y)`` coordinates.

    ``pixels`` may have shape ``[N, 2]`` or ``[N, Hs, Ws, 2]``.  The return
    shape is ``[N]`` or ``[N, Hs, Ws]`` respectively.  A single source image is
    expanded without copying for all sample batches.
    """

    image = prepare_image(
        image,
        device=image.device,
        dtype=image.dtype if image.dtype.is_floating_point else torch.float32,
        normalize=not image.dtype.is_floating_point,
    )
    if pixels.ndim == 2:
        grid_pixels = pixels[:, None, None, :]
        squeeze = True
    elif pixels.ndim == 4:
        grid_pixels = pixels
        squeeze = False
    else:
        raise ValueError("pixels must have shape [N,2] or [N,Hs,Ws,2]")
    n = grid_pixels.shape[0]
    height, width = image.shape[-2:]
    grid = _pixel_grid(grid_pixels.to(dtype=image.dtype, device=image.device), height, width)
    source = image.expand(n, -1, -1, -1)
    sampled = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )[:, 0]
    return sampled[:, 0, 0] if squeeze else sampled


def patch_offsets(
    patch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Return row-major SVO offsets ``[-half, half)`` as ``[P,P,2]``."""

    if patch_size < 2 or patch_size % 2:
        raise ValueError("patch_size must be a positive even number")
    half = patch_size // 2
    values = torch.arange(-half, half, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def sample_patches(
    image: Tensor,
    centers: Tensor,
    patch_size: int = 8,
    *,
    padding_mode: str = "zeros",
) -> Tensor:
    """Sample square patches centered at ``centers`` into ``[N, P, P]``."""

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape [N, 2]")
    image = prepare_image(
        image,
        device=image.device,
        dtype=image.dtype if image.dtype.is_floating_point else torch.float32,
        normalize=not image.dtype.is_floating_point,
    )
    centers = centers.to(device=image.device, dtype=image.dtype)
    offsets = patch_offsets(patch_size, device=image.device, dtype=image.dtype)
    grids = centers[:, None, None, :] + offsets[None]
    return sample_image(image, grids, padding_mode=padding_mode)


def patches_in_bounds(centers: Tensor, image: Tensor, patch_size: int = 8) -> Tensor:
    """Return whether all continuous patch coordinates lie inside the image."""

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape [N, 2]")
    half = patch_size // 2
    height, width = image.shape[-2:]
    return (
        torch.isfinite(centers).all(dim=-1)
        & (centers[:, 0] - half >= 0)
        & (centers[:, 1] - half >= 0)
        & (centers[:, 0] + half - 1 <= width - 1)
        & (centers[:, 1] + half - 1 <= height - 1)
    )
