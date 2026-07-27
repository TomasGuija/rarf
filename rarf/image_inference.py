"""Reusable natural-image preprocessing and inpainting helpers."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _resize_and_center_crop(
    image: Image.Image,
    resolution: int,
    resample: Image.Resampling,
) -> Image.Image:
    width, height = image.size
    scale = resolution / min(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)),
        resample=resample,
    )
    left = (resized.width - resolution) // 2
    top = (resized.height - resolution) // 2
    return resized.crop((left, top, left + resolution, top + resolution))


def prepare_image(image: Image.Image, resolution: int) -> torch.Tensor:
    """Convert an image to a centered RGB tensor in ``[-1, 1]``."""

    image = _resize_and_center_crop(
        image.convert("RGB"),
        resolution,
        Image.Resampling.BILINEAR,
    )
    pixels = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).div(127.5).sub(1.0)


def prepare_mask(mask: Image.Image, resolution: int) -> torch.Tensor:
    """Convert a drawn mask to a centered binary ``[1, H, W]`` tensor."""

    mask = _resize_and_center_crop(
        mask.convert("L"),
        resolution,
        Image.Resampling.NEAREST,
    )
    values = np.asarray(mask) > 0
    return torch.from_numpy(values).to(torch.float32).unsqueeze(0)


def sample_image(
    model,
    flow,
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    steps: int,
    num_samples: int = 1,
    strategy: str = "normal",
    seed: int | None = None,
) -> np.ndarray:
    """Generate one image or merge several samples on the CPU."""

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if strategy not in {"normal", "mean", "closest_to_mean"}:
        raise ValueError("Unknown sampling strategy.")
    if strategy == "normal" and num_samples != 1:
        raise ValueError("Normal sampling requires num_samples=1.")

    count = 1 if strategy == "normal" else num_samples
    predictions = []
    for index in range(count):
        initial_noise = None
        if seed is not None:
            generator = torch.Generator(device=image.device).manual_seed(seed + index)
            initial_noise = torch.randn(
                image.shape,
                generator=generator,
                device=image.device,
                dtype=image.dtype,
            )
        prediction = flow.sample_inpaint(
            model,
            image,
            mask,
            steps=steps,
            initial_noise=initial_noise,
        )
        predictions.append(prediction[0].cpu().numpy())

    if strategy == "normal":
        return predictions[0]

    stack = np.stack(predictions)
    mean = stack.mean(axis=0)
    if strategy == "mean":
        return mean

    region = mask[0, 0].bool().cpu().numpy()
    distances = ((stack[:, :, region] - mean[:, region]) ** 2).mean(axis=(1, 2))
    return stack[int(np.argmin(distances))]


def rgb_image(array: np.ndarray | torch.Tensor) -> Image.Image:
    """Convert a channel-first ``[-1, 1]`` image to RGB PIL format."""

    array = np.asarray(array, dtype=np.float32)
    array = np.clip((array + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    return Image.fromarray(np.moveaxis(array, 0, -1))


def mask_image(mask: np.ndarray | torch.Tensor) -> Image.Image:
    """Convert a channel-first binary mask to grayscale PIL format."""

    array = np.asarray(mask, dtype=np.float32).squeeze(0)
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def comparison_image(target, voided, mask, prediction) -> Image.Image:
    """Place input, voided input, mask, and prediction side by side."""

    panels = [
        rgb_image(target),
        rgb_image(voided),
        mask_image(mask).convert("RGB"),
        rgb_image(prediction),
    ]
    result = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
    left = 0
    for panel in panels:
        result.paste(panel, (left, 0))
        left += panel.width
    return result
