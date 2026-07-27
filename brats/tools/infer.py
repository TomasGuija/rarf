"""Run RARF inpainting on one BraTS inference case."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from brats.data.dataset import (
    BraTSDataset,
    _anatomy_center,
    _crop_or_pad_centered,
    _denormalize_image,
    _load_nifti,
    _normalization_bounds,
    _normalize_image,
    _to_binary_mask,
)
from rarf.training.checkpoints import load_rarf_checkpoint


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run RARF inpainting on one BraTS case."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--index", type=int, default=0)
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


def _data_settings(checkpoint):
    settings = checkpoint["datamodule_hyper_parameters"]
    try:
        return (
            tuple(settings["crop_shape"]),
            float(settings["robust_percentile_lower"]),
            float(settings["robust_percentile"]),
        )
    except KeyError as error:
        raise ValueError(
            f"Checkpoint is missing the data setting {error.args[0]!r}."
        ) from error


def _select_case(dataset, case_id, index):
    if case_id is None:
        return dataset.cases[index]
    for case in dataset.cases:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"Case {case_id!r} not found.")


def _crop_slices(full_shape, crop_shape, center):
    source = []
    destination = []
    for size, target_size, midpoint in zip(full_shape, crop_shape, center):
        start = midpoint - target_size // 2
        stop = start + target_size
        source_start = max(start, 0)
        source_stop = min(stop, size)
        destination_start = source_start - start
        destination_stop = destination_start + source_stop - source_start
        source.append(slice(source_start, source_stop))
        destination.append(slice(destination_start, destination_stop))
    return tuple(source), tuple(destination)


def _sample(model, flow, image, mask, steps, num_samples, strategy, seed):
    """Generate one sample or merge multiple sequential samples on the CPU."""

    if num_samples < 1:
        raise ValueError("--num-samples must be at least 1.")
    if strategy == "normal" and num_samples != 1:
        raise ValueError("Normal sampling requires --num-samples 1.")
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
        predictions.append(prediction[0, 0].float().cpu().numpy())

    if strategy == "normal":
        return predictions[0]
    stack = np.stack(predictions)
    mean = stack.mean(axis=0, dtype=np.float32)
    if strategy == "mean":
        return mean

    region = mask[0, 0].bool().cpu().numpy()
    distances = ((stack[:, region] - mean[region]) ** 2).mean(axis=1)
    return stack[int(np.argmin(distances))]


def main():
    args = _parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    crop_shape, robust_percentile_lower, robust_percentile = _data_settings(
        checkpoint
    )
    del checkpoint

    dataset = BraTSDataset(
        args.dataset_root,
        test_flag=True,
        crop_shape=crop_shape,
        robust_percentile_lower=robust_percentile_lower,
        robust_percentile=robust_percentile,
    )
    case = _select_case(dataset, args.case_id, args.index)
    reference_image = nib.load(str(case["t1n_voided"]))
    voided = _load_nifti(case["t1n_voided"])
    mask = _to_binary_mask(_load_nifti(case["mask"]))
    voided = voided * (1.0 - mask)

    center = _anatomy_center(voided, mask)
    voided_crop = _crop_or_pad_centered(voided, crop_shape, center)
    mask_crop = _crop_or_pad_centered(mask, crop_shape, center)
    lower, upper = _normalization_bounds(
        voided,
        robust_percentile_lower,
        robust_percentile,
    )
    image = torch.from_numpy(
        _normalize_image(voided_crop, lower, upper)[None, None]
    ).float().to(device)
    mask_tensor = torch.from_numpy(mask_crop[None, None]).float().to(device)

    module, selected_weights = load_rarf_checkpoint(
        args.checkpoint,
        device=device,
        weights="auto",
    )
    print(f"Using {selected_weights} weights", flush=True)
    with torch.inference_mode():
        prediction = _sample(
            module.model,
            module.flow,
            image,
            mask_tensor,
            args.sample_steps or module.sample_steps,
            args.num_samples,
            args.sampling_strategy,
            args.seed,
        )
    prediction = _denormalize_image(
        np.clip(prediction, 0.0, 1.0),
        lower,
        upper,
    )

    source_slices, destination_slices = _crop_slices(
        voided.shape, crop_shape, center
    )
    result = voided.copy()
    result_region = result[source_slices]
    generated_region = prediction[destination_slices]
    regional_mask = mask_crop[destination_slices] > 0
    result_region[regional_mask] = generated_region[regional_mask]
    result[source_slices] = result_region

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case['case_id']}-rarf-inpainted.nii.gz"
    nib.save(
        nib.Nifti1Image(result, reference_image.affine, reference_image.header),
        output_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
