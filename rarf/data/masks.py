"""Mask distributions for natural-image inpainting.

The Perlin distribution is independently implemented from the strategy
described by RAD: https://github.com/srk1995/RAD
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _fade(values: torch.Tensor) -> torch.Tensor:
    """Smoothly interpolate between neighboring Perlin grid cells."""

    return values**3 * (values * (values * 6.0 - 15.0) + 10.0)


def _perlin_noise(height: int, width: int, scale: float) -> torch.Tensor:
    """Generate one conventional two-dimensional Perlin noise field."""

    y = torch.arange(height, dtype=torch.float32) / scale
    x = torch.arange(width, dtype=torch.float32) / scale
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    y0 = y_grid.floor().long()
    x0 = x_grid.floor().long()
    y1 = y0 + 1
    x1 = x0 + 1

    angles = torch.rand(
        int(y1.max()) + 1,
        int(x1.max()) + 1,
    ) * (2.0 * math.pi)
    gradients = torch.stack((angles.cos(), angles.sin()), dim=-1)

    local_y = y_grid - y0
    local_x = x_grid - x0

    def ramp(grid_y: torch.Tensor, grid_x: torch.Tensor, dy: float, dx: float):
        gradient = gradients[grid_y, grid_x]
        offset = torch.stack((local_y - dy, local_x - dx), dim=-1)
        return (gradient * offset).sum(dim=-1)

    top_left = ramp(y0, x0, 0.0, 0.0)
    top_right = ramp(y0, x1, 0.0, 1.0)
    bottom_left = ramp(y1, x0, 1.0, 0.0)
    bottom_right = ramp(y1, x1, 1.0, 1.0)

    blend_y = _fade(local_y)
    blend_x = _fade(local_x)
    top = torch.lerp(top_left, top_right, blend_x)
    bottom = torch.lerp(bottom_left, bottom_right, blend_x)
    return torch.lerp(top, bottom, blend_y)


@dataclass(frozen=True)
class PerlinMaskGenerator:
    """Generate binary masks using RAD's training-mask strategy.

    A smooth Perlin field is sampled at a random spatial scale and thresholded
    at a random area fraction. Consequently, training sees both fine and coarse
    holes with widely varying sizes. The output follows RARF's convention:
    ``1`` is the region to generate and ``0`` is observed context.

    Args:
        min_scale: Smallest Perlin feature size in pixels. Smaller values create
            finer mask structures.
        max_scale: Largest feature size. ``None`` uses the shorter image side,
            matching RAD's full fine-to-coarse scale range.
        min_hole_fraction: Smallest sampled fraction of missing pixels.
        max_hole_fraction: Largest sampled fraction of missing pixels.
    """

    min_scale: float = 1.0
    max_scale: float | None = None
    min_hole_fraction: float = 0.0
    max_hole_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.min_scale <= 0.0:
            raise ValueError("min_scale must be positive.")
        if self.max_scale is not None and self.max_scale < self.min_scale:
            raise ValueError("max_scale cannot be smaller than min_scale.")
        if not 0.0 <= self.min_hole_fraction <= self.max_hole_fraction <= 1.0:
            raise ValueError("Hole fractions must satisfy 0 <= min <= max <= 1.")

    def __call__(self, height: int, width: int) -> torch.Tensor:
        """Return a newly sampled mask with shape ``[1, height, width]``."""

        if height < 1 or width < 1:
            raise ValueError("Mask dimensions must be positive.")

        maximum = float(self.max_scale or min(height, width))
        if maximum < self.min_scale:
            raise ValueError("Image dimensions are smaller than min_scale.")

        scale = torch.empty(()).uniform_(self.min_scale, maximum).item()
        noise = _perlin_noise(height, width, scale)

        pixel_count = noise.numel()
        minimum = round(self.min_hole_fraction * pixel_count)
        maximum = round(self.max_hole_fraction * pixel_count)
        hole_pixels = int(torch.randint(minimum, maximum + 1, ()).item())
        if hole_pixels == 0:
            mask = torch.zeros_like(noise)
        elif hole_pixels == pixel_count:
            mask = torch.ones_like(noise)
        else:
            threshold = noise.flatten().kthvalue(hole_pixels).values
            mask = noise <= threshold

        return mask.to(torch.float32).unsqueeze(0)
