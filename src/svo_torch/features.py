"""Grid-distributed Shi--Tomasi corner and gradient edgelet detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .frame import CORNER, EDGELET, FeatureSet
from .image import image_gradients, prepare_image


@dataclass(slots=True)
class _Candidates:
    pixels: Tensor
    scores: Tensor
    levels: Tensor
    gradients: Tensor
    kinds: Tensor

    @classmethod
    def empty(cls, image: Tensor) -> _Candidates:
        device, dtype = image.device, image.dtype
        return cls(
            torch.empty((0, 2), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty((0, 2), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.long),
        )

    @staticmethod
    def concatenate(parts: list[_Candidates], image: Tensor) -> _Candidates:
        parts = [part for part in parts if part.scores.numel()]
        if not parts:
            return _Candidates.empty(image)
        return _Candidates(
            pixels=torch.cat([part.pixels for part in parts]),
            scores=torch.cat([part.scores for part in parts]),
            levels=torch.cat([part.levels for part in parts]),
            gradients=torch.cat([part.gradients for part in parts]),
            kinds=torch.cat([part.kinds for part in parts]),
        )

    def take(self, indices: Tensor) -> _Candidates:
        return _Candidates(
            self.pixels[indices],
            self.scores[indices],
            self.levels[indices],
            self.gradients[indices],
            self.kinds[indices],
        )


def shi_tomasi_score(image: Tensor, window_size: int = 7) -> tuple[Tensor, Tensor, Tensor]:
    """Return the minimum structure-tensor eigenvalue and image gradients."""

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and at least three")
    image = prepare_image(
        image,
        device=image.device,
        dtype=image.dtype if image.dtype.is_floating_point else torch.float32,
    )
    dx, dy = image_gradients(image)
    padding = window_size // 2
    xx = F.avg_pool2d(dx.square(), window_size, stride=1, padding=padding)
    yy = F.avg_pool2d(dy.square(), window_size, stride=1, padding=padding)
    xy = F.avg_pool2d(dx * dy, window_size, stride=1, padding=padding)
    discriminant = ((xx - yy).square() + 4.0 * xy.square()).clamp_min(0.0).sqrt()
    score = 0.5 * (xx + yy - discriminant)
    return score, dx, dy


def _nms(score: Tensor, threshold: Tensor | float) -> Tensor:
    pooled = F.max_pool2d(score, kernel_size=3, stride=1, padding=1)
    # Equality deliberately gives deterministic raster-order tie breaking later.
    return (score >= pooled) & (score > threshold)


def _mask_at_pixels(mask: Tensor | None, pixels: Tensor) -> Tensor:
    if mask is None:
        return torch.ones(pixels.shape[0], dtype=torch.bool, device=pixels.device)
    mask_image = torch.as_tensor(mask, device=pixels.device)
    while mask_image.ndim > 2 and mask_image.shape[0] == 1:
        mask_image = mask_image[0]
    if mask_image.ndim != 2:
        raise ValueError("detector mask must have shape [H,W], [1,H,W], or [1,1,H,W]")

    # SVO's camera mask uses direct ``cv::Mat::at`` lookup after C++ integer
    # conversion.  Besides matching that behavior, indexing the bitmap once
    # avoids expanding a full-resolution mask for every feature candidate.
    height, width = mask_image.shape
    indices = pixels.to(dtype=torch.long)
    x, y = indices.unbind(dim=-1)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x.clamp(0, width - 1)
    y = y.clamp(0, height - 1)
    return inside & (mask_image[y, x] != 0)


def _level_candidates(
    image: Tensor,
    level: int,
    quality_level: float,
    border: int,
    window_size: int,
) -> tuple[_Candidates, _Candidates]:
    score, dx, dy = shi_tomasi_score(image, window_size)
    magnitude = torch.sqrt(dx.square() + dy.square())
    corner_threshold = score.detach().amax() * quality_level
    edge_threshold = magnitude.detach().amax() * quality_level
    corner_mask = _nms(score, corner_threshold)
    edge_mask = _nms(magnitude, edge_threshold)

    height, width = image.shape[-2:]
    effective_border = max(border, window_size // 2 + 1)
    valid_region = torch.zeros_like(corner_mask)
    if height > 2 * effective_border and width > 2 * effective_border:
        valid_region[
            ...,
            effective_border : height - effective_border,
            effective_border : width - effective_border,
        ] = True
    corner_mask &= valid_region
    edge_mask &= valid_region

    def gather(mask: Tensor, values: Tensor, kind: int) -> _Candidates:
        yx = torch.nonzero(mask[0, 0], as_tuple=False)
        if not yx.numel():
            return _Candidates.empty(image)
        y, x = yx[:, 0], yx[:, 1]
        scale = float(1 << level)
        pixels = torch.stack((x, y), dim=-1).to(image.dtype) * scale
        gradients = torch.stack((dx[0, 0, y, x], dy[0, 0, y, x]), dim=-1)
        norm = torch.linalg.vector_norm(gradients, dim=-1, keepdim=True)
        fallback = torch.zeros_like(gradients)
        fallback[:, 0] = 1.0
        gradients = torch.where(
            norm > torch.finfo(image.dtype).eps,
            gradients / norm.clamp_min(torch.finfo(image.dtype).eps),
            fallback,
        )
        n = x.shape[0]
        return _Candidates(
            pixels,
            values[0, 0, y, x],
            torch.full((n,), level, dtype=torch.long, device=image.device),
            gradients,
            torch.full((n,), kind, dtype=torch.long, device=image.device),
        )

    return gather(corner_mask, score, CORNER), gather(edge_mask, magnitude, EDGELET)


def _grid_winners(
    candidates: _Candidates,
    grid_size: int,
    image_width: int,
    image_height: int,
    unavailable_cells: Tensor | None = None,
) -> _Candidates:
    if not candidates.scores.numel():
        return candidates
    n_cols = (image_width + grid_size - 1) // grid_size
    n_rows = (image_height + grid_size - 1) // grid_size
    xy_cell = torch.floor(candidates.pixels / grid_size).to(torch.long)
    cells = xy_cell[:, 1] * n_cols + xy_cell[:, 0]
    inside = (
        (xy_cell[:, 0] >= 0)
        & (xy_cell[:, 0] < n_cols)
        & (xy_cell[:, 1] >= 0)
        & (xy_cell[:, 1] < n_rows)
    )
    if unavailable_cells is not None and unavailable_cells.numel():
        occupied = torch.zeros(n_cols * n_rows, dtype=torch.bool, device=cells.device)
        occupied[unavailable_cells] = True
        inside &= ~occupied[cells.clamp(0, n_cols * n_rows - 1)]
    source = torch.nonzero(inside, as_tuple=False).squeeze(-1)
    if not source.numel():
        return candidates.take(source)
    filtered = candidates.take(source)
    cells = cells[source]
    # Stable score order followed by first-rank scatter gives one deterministic
    # maximum per cell without a CPU loop or device synchronization.
    order = torch.argsort(filtered.scores, descending=True, stable=True)
    ordered_cells = cells[order]
    ranks = torch.arange(order.numel(), device=order.device, dtype=torch.long)
    best_rank = torch.full((n_cols * n_rows,), order.numel(), device=order.device, dtype=torch.long)
    best_rank.scatter_reduce_(0, ordered_cells, ranks, reduce="amin", include_self=True)
    keep_ordered = ranks == best_rank[ordered_cells]
    return filtered.take(order[keep_ordered])


def _candidate_cells(candidates: _Candidates, grid_size: int, image_width: int) -> Tensor:
    n_cols = (image_width + grid_size - 1) // grid_size
    cell_xy = torch.floor(candidates.pixels / grid_size).to(torch.long)
    return cell_xy[:, 1] * n_cols + cell_xy[:, 0]


class GridFeatureDetector:
    """Shi--Tomasi corners followed by gradient edgelets, one per grid cell."""

    def __init__(
        self,
        max_features: int,
        grid_size: int,
        quality_level: float,
        border: int,
        edgelet_ratio: float,
        *,
        min_level: int = 0,
        max_level: int | None = None,
        window_size: int = 7,
    ) -> None:
        if max_features < 1:
            raise ValueError("max_features must be positive")
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        if not 0.0 < quality_level <= 1.0:
            raise ValueError("quality_level must be in (0, 1]")
        if border < 0:
            raise ValueError("border must be non-negative")
        if not 0.0 <= edgelet_ratio <= 1.0:
            raise ValueError("edgelet_ratio must be in [0, 1]")
        self.max_features = max_features
        self.grid_size = grid_size
        self.quality_level = quality_level
        self.border = border
        self.edgelet_ratio = edgelet_ratio
        self.min_level = min_level
        self.max_level = max_level
        self.window_size = window_size

    def detect(self, pyramid: list[Tensor], mask: Tensor | None = None) -> FeatureSet:
        if not pyramid:
            raise ValueError("pyramid must contain at least one image")
        images = [
            prepare_image(
                image,
                device=image.device,
                dtype=image.dtype if image.dtype.is_floating_point else torch.float32,
            )
            for image in pyramid
        ]
        base = images[0]
        max_level = (
            len(images) - 1 if self.max_level is None else min(self.max_level, len(images) - 1)
        )
        if not 0 <= self.min_level <= max_level:
            raise ValueError("invalid detector pyramid-level range")

        corner_parts: list[_Candidates] = []
        edge_parts: list[_Candidates] = []
        for level in range(self.min_level, max_level + 1):
            corners, edges = _level_candidates(
                images[level], level, self.quality_level, self.border, self.window_size
            )
            corner_parts.append(corners)
            edge_parts.append(edges)
        corners = _Candidates.concatenate(corner_parts, base)
        edges = _Candidates.concatenate(edge_parts, base)
        if corners.scores.numel():
            keep = torch.nonzero(_mask_at_pixels(mask, corners.pixels), as_tuple=False).squeeze(-1)
            corners = corners.take(keep)
        if edges.scores.numel():
            keep = torch.nonzero(_mask_at_pixels(mask, edges.pixels), as_tuple=False).squeeze(-1)
            edges = edges.take(keep)

        height, width = base.shape[-2:]
        corners = _grid_winners(corners, self.grid_size, width, height)
        corner_budget = min(
            self.max_features,
            round(self.max_features * (1.0 - self.edgelet_ratio)),
        )
        corner_count = min(corner_budget, corners.scores.numel())
        selected_corners = corners.take(torch.arange(corner_count, device=base.device))
        occupied = _candidate_cells(selected_corners, self.grid_size, width)
        edges = _grid_winners(edges, self.grid_size, width, height, occupied)
        edge_budget = min(
            self.max_features - corner_count,
            round(self.max_features * self.edgelet_ratio),
        )
        edge_count = min(edge_budget, edges.scores.numel())
        selected_edges = edges.take(torch.arange(edge_count, device=base.device))

        # Fill unused quota with remaining corners while preserving one feature
        # per cell.  This matters on low-texture scenes with very few edgelets.
        chosen = _Candidates.concatenate([selected_corners, selected_edges], base)
        if chosen.scores.numel() < self.max_features and corners.scores.numel() > corner_count:
            occupied = _candidate_cells(chosen, self.grid_size, width)
            remaining = corners.take(
                torch.arange(corner_count, corners.scores.numel(), device=base.device)
            )
            remaining = _grid_winners(remaining, self.grid_size, width, height, occupied)
            n_extra = min(self.max_features - chosen.scores.numel(), remaining.scores.numel())
            chosen = _Candidates.concatenate(
                [chosen, remaining.take(torch.arange(n_extra, device=base.device))], base
            )

        if not chosen.scores.numel():
            return FeatureSet.empty(device=base.device, dtype=base.dtype)
        order = torch.argsort(chosen.scores, descending=True, stable=True)
        chosen = chosen.take(order[: self.max_features])
        return FeatureSet.from_detection(
            chosen.pixels,
            chosen.scores,
            chosen.levels,
            chosen.gradients,
            chosen.kinds,
        )
