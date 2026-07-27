"""Run RARF inpainting on one natural image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rarf.data.image_dataset import NaturalImageDataset
from rarf.data.masks import PerlinMaskGenerator
from rarf.image_inference import comparison_image, rgb_image, sample_image
from rarf.training.checkpoints import load_rarf_checkpoint


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run RARF inpainting on one natural image."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--sampling-strategy",
        choices=("normal", "mean", "closest_to_mean"),
        default="normal",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    module, selected_weights, settings = load_rarf_checkpoint(
        args.checkpoint,
        device=device,
        weights="auto",
        return_data_hparams=True,
    )

    mask_generator = PerlinMaskGenerator(
        min_scale=settings["mask_min_scale"],
        max_scale=settings["mask_max_scale"],
        min_hole_fraction=settings["min_hole_fraction"],
        max_hole_fraction=settings["max_hole_fraction"],
    )
    dataset = NaturalImageDataset(
        [image_path],
        resolution=settings["resolution"],
        random_crop=False,
        random_flip=False,
        mask_generator=mask_generator,
        seed=args.seed,
    )
    sample = dataset[0]
    target = sample["target"]
    mask = sample["mask"]
    voided = target * (1.0 - mask)
    image = voided.unsqueeze(0).to(device)
    mask_tensor = mask.unsqueeze(0).to(device)

    print(f"Using {selected_weights} weights", flush=True)
    with torch.inference_mode():
        prediction = sample_image(
            module.model,
            module.flow,
            image,
            mask_tensor,
            steps=args.sample_steps or module.sample_steps,
            num_samples=args.num_samples,
            strategy=args.sampling_strategy,
            seed=args.seed,
        )

    target = target.numpy()
    mask = mask.numpy()
    voided = voided.numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}-rarf-inpainted.png"
    comparison_path = output_dir / f"{image_path.stem}-comparison.png"
    rgb_image(prediction).save(output_path)
    comparison_image(target, voided, mask, prediction).save(comparison_path)
    print(output_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
