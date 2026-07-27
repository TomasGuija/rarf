"""Generic RGB image dataset for RARF training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rarf.data.masks import PerlinMaskGenerator

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def discover_images(directory: str | Path) -> list[Path]:
    """Recursively find supported image files in a directory."""

    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found under {root}")
    return paths


def _load_rgb(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    raise TypeError(f"Unsupported image value of type {type(value).__name__}.")


def _resize_shorter_side(image: Image.Image, resolution: int) -> Image.Image:
    width, height = image.size
    scale = resolution / min(width, height)
    size = (round(width * scale), round(height * scale))
    return image.resize(size, resample=Image.Resampling.BILINEAR)


def _crop_square(image: Image.Image, resolution: int, random_crop: bool) -> Image.Image:
    width, height = image.size
    max_left = width - resolution
    max_top = height - resolution
    if random_crop:
        left = int(torch.randint(max_left + 1, ()).item())
        top = int(torch.randint(max_top + 1, ()).item())
    else:
        left = max_left // 2
        top = max_top // 2
    return image.crop((left, top, left + resolution, top + resolution))


class NaturalImageDataset(torch.utils.data.Dataset):
    """Prepare RGB images and online inpainting masks.

    ``records`` may be a sequence of image paths or a Hugging Face dataset. A
    new mask is drawn on every access, so an image receives different regional
    paths over training epochs.
    """

    def __init__(
        self,
        records: Sequence,
        *,
        image_column: str = "image",
        resolution: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        mask_generator: PerlinMaskGenerator | None = None,
        seed: int | None = None,
    ) -> None:
        if resolution < 1:
            raise ValueError("resolution must be positive.")
        self.records = records
        self.image_column = image_column
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.mask_generator = mask_generator or PerlinMaskGenerator()
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def _image_value(self, index: int):
        record = self.records[index]
        if isinstance(record, dict):
            if self.image_column not in record:
                raise KeyError(f"Dataset record has no '{self.image_column}' column.")
            return record[self.image_column]
        return record

    def _make_sample(self, index: int) -> dict[str, torch.Tensor]:
        image = _load_rgb(self._image_value(index))
        image = _resize_shorter_side(image, self.resolution)
        image = _crop_square(image, self.resolution, self.random_crop)
        if self.random_flip and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        pixels = np.array(image, dtype=np.float32, copy=True)
        target = torch.from_numpy(pixels).permute(2, 0, 1)
        target = target.div(127.5).sub(1.0)
        mask = self.mask_generator(self.resolution, self.resolution)
        return {"target": target, "mask": mask}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.seed is None:
            return self._make_sample(index)

        # Validation should evaluate the same crop and mask every epoch without
        # modifying the worker's random state used by other samples.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + index)
            return self._make_sample(index)
