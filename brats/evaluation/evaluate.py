"""Quantitative evaluation of RARF checkpoints on healthy BraTS tissue."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import torch

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm


METRIC_KEYS = ("mse", "mae", "ssim", "psnr_01")
LOGGER = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a RARF checkpoint on healthy BraTS synthesis."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Validation folder. Defaults to val_data_dir from the checkpoint.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save generated and voided NIfTI volumes for visual inspection.",
    )
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--weights",
        choices=("auto", "ema", "raw"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "32", "bf16", "16"),
        default="auto",
    )
    return parser.parse_args()


def _checkpoint_hparams(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint["hyper_parameters"], checkpoint["datamodule_hyper_parameters"]


def _metric_tensor(array, *, boolean=False):
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).unsqueeze(0)
    return tensor.bool().contiguous() if boolean else tensor.contiguous()


def _to_float(value):
    """Convert a scalar metric tensor or number to a Python float."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _evaluation_cases(dataset):
    """Use every case in the selected validation folder with its first mask."""

    cases = []
    for case in dataset.cases:
        variant = case["mask_variants"][0]
        cases.append(
            {
                **case,
                "mask": variant["mask"],
                "mask_healthy": variant["mask_healthy"],
            }
        )
    return cases


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


def _evaluate_case(
    module,
    case,
    crop_shape,
    robust_percentile_lower,
    robust_percentile,
    sample_steps,
    sampling_seed,
    device,
    precision,
    generate_metrics,
    output_dir,
    save_outputs,
):
    import torch

    from brats.data.dataset import (
        _anatomy_center,
        _crop_or_pad_centered,
        _denormalize_image,
        _load_nifti,
        _normalization_bounds,
        _normalize_image,
        _to_binary_mask,
    )

    reference_image = nib.load(str(case["t1n"]))
    target = _load_nifti(case["t1n"])
    full_mask = _to_binary_mask(_load_nifti(case["mask"]))
    healthy_mask = _to_binary_mask(_load_nifti(case["mask_healthy"]))
    voided = target * (1.0 - full_mask)

    center = _anatomy_center(voided, full_mask)
    target_crop = _crop_or_pad_centered(target, crop_shape, center)
    mask_crop = _crop_or_pad_centered(full_mask, crop_shape, center)
    voided_crop = target_crop * (1.0 - mask_crop)
    lower, upper = _normalization_bounds(
        voided,
        robust_percentile_lower,
        robust_percentile,
    )
    normalized_target = _normalize_image(target_crop, lower, upper)
    normalized_voided = normalized_target * (1.0 - mask_crop)
    image = torch.from_numpy(normalized_voided[None, None]).float().to(device)
    mask = torch.from_numpy(mask_crop[None, None]).float().to(device)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    initial_noise = torch.randn(image.shape, generator=generator, device=device)

    autocast_dtype = (
        torch.bfloat16 if precision == "bf16" else torch.float16
    )

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=autocast_dtype,
        enabled=device.type == "cuda" and precision != "32",
    ):
        prediction_crop = module.flow.sample_inpaint(
            module.model,
            image,
            mask,
            steps=sample_steps,
            low_memory=True,
            initial_noise=initial_noise,
        )

    prediction_crop = _denormalize_image(
        prediction_crop[0, 0].float().cpu().numpy().clip(0.0, 1.0),
        lower,
        upper,
    )

    source_slices, destination_slices = _crop_slices(
        target.shape,
        crop_shape,
        center,
    )
    prediction = voided.copy()
    prediction_region = prediction[source_slices]
    generated_region = prediction_crop[destination_slices]
    regional_mask = mask_crop[destination_slices] > 0
    prediction_region[regional_mask] = generated_region[regional_mask]
    prediction[source_slices] = prediction_region

    all_metrics = generate_metrics(
        prediction=_metric_tensor(prediction),
        target=_metric_tensor(target),
        normalization_tensor=_metric_tensor(voided),
        mask=_metric_tensor(healthy_mask, boolean=True),
    )
    metrics = {key: _to_float(all_metrics[key]) for key in METRIC_KEYS}

    prediction_path = None
    if save_outputs:
        prediction_directory = output_dir / "predictions"
        voided_directory = output_dir / "voided"
        prediction_directory.mkdir(parents=True, exist_ok=True)
        voided_directory.mkdir(parents=True, exist_ok=True)
        prediction_path = prediction_directory / f"{case['case_id']}-prediction.nii.gz"
        voided_path = voided_directory / f"{case['case_id']}-voided.nii.gz"
        nib.save(
            nib.Nifti1Image(prediction, reference_image.affine, reference_image.header),
            prediction_path,
        )
        nib.save(
            nib.Nifti1Image(voided, reference_image.affine, reference_image.header),
            voided_path,
        )
    return metrics, prediction_path


def _write_results(rows, metadata, output_dir):
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("case_id", *METRIC_KEYS))
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in ("case_id", *METRIC_KEYS)} for row in rows
        )

    summary = {**metadata, "metrics": {}}
    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary["metrics"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return csv_path, summary_path, summary


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.sample_steps is not None and args.sample_steps < 2:
        raise ValueError("--sample-steps must be at least 2.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive.")

    output_dir = Path(
        args.output_dir
        or Path("outputs/brats-evaluation") / Path(args.checkpoint).stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib"))

    import torch
    from inpainting.challenge_metrics_2023 import generate_metrics

    from brats.data.dataset import BraTSDataset
    from rarf.training.checkpoints import load_rarf_checkpoint

    model_hparams, data_hparams = _checkpoint_hparams(args.checkpoint)
    data_root = args.data_root or data_hparams.get("val_data_dir")
    if not data_root:
        raise ValueError("The checkpoint has no val_data_dir; pass it with --data-root.")
    crop_shape = tuple(data_hparams["crop_shape"])
    robust_percentile_lower = float(data_hparams["robust_percentile_lower"])
    robust_percentile = float(data_hparams["robust_percentile"])
    sample_steps = int(
        args.sample_steps
        if args.sample_steps is not None
        else model_hparams.get("sample_steps", 16)
    )

    dataset = BraTSDataset(
        data_root,
        test_flag=False,
        crop_shape=crop_shape,
        robust_percentile_lower=robust_percentile_lower,
        robust_percentile=robust_percentile,
        random_mask_variant=False,
        image_augment=False,
    )
    cases = _evaluation_cases(dataset)
    if args.max_samples is not None:
        cases = cases[: args.max_samples]

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    precision = args.precision
    if device.type != "cuda" or precision == "32":
        precision = "32"
    elif precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "16"
    module, selected_weights = load_rarf_checkpoint(
        args.checkpoint,
        device=device,
        weights=args.weights,
    )
    LOGGER.info(
        "checkpoint=%s | validation_root=%s",
        args.checkpoint,
        data_root,
    )
    LOGGER.info(
        "cases=%d | sampling_points=%d | weights=%s | device=%s | precision=%s",
        len(cases),
        sample_steps,
        selected_weights,
        device,
        precision,
    )

    rows = []
    progress = tqdm(cases, desc="Evaluating BraTS", unit="case", dynamic_ncols=True)
    for index, case in enumerate(progress):
        progress.set_description(f"Evaluating {case['case_id']}")
        metrics, prediction_path = _evaluate_case(
            module=module,
            case=case,
            crop_shape=crop_shape,
            robust_percentile_lower=robust_percentile_lower,
            robust_percentile=robust_percentile,
            sample_steps=sample_steps,
            sampling_seed=args.sampling_seed + index,
            device=device,
            precision=precision,
            generate_metrics=generate_metrics,
            output_dir=output_dir,
            save_outputs=args.save_outputs,
        )
        rows.append({"case_id": case["case_id"], **metrics})
        progress.set_postfix(
            mse=f"{metrics['mse']:.4g}",
            ssim=f"{metrics['ssim']:.4g}",
            psnr=f"{metrics['psnr_01']:.4g}",
        )
        if prediction_path is not None:
            LOGGER.info("Saved prediction for %s to %s", case["case_id"], prediction_path)

    csv_path, summary_path, summary = _write_results(
        rows,
        {
            "num_samples": len(rows),
            "checkpoint": str(args.checkpoint),
            "data_root": str(data_root),
            "sampling_seed": args.sampling_seed,
            "sample_steps": sample_steps,
            "weights": selected_weights,
            "precision": precision,
            "healthy_tissue_only": True,
            "saved_outputs": args.save_outputs,
        },
        output_dir,
    )
    LOGGER.info("Wrote per-case metrics to %s", csv_path)
    LOGGER.info("Wrote metric summary to %s", summary_path)
    LOGGER.info("Summary metrics:\n%s", json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
