"""Minimal Docker inference for the BraTS inpainting submission format.

The container expects challenge cases below ``/input`` and writes one flat
``*-t1n-inference.nii.gz`` file per case to ``/output``. Configuration is kept
to four environment variables: ``SAMPLE_STEPS``, ``NUM_SAMPLES``, ``SEED``,
and ``MERGE_STRATEGY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from rarf.models.utils import create_model
from rarf.region_aware_flows import RARF


@dataclass(frozen=True)
class Settings:
    """Validated inference settings read from the environment and checkpoint."""

    sample_steps: int
    num_samples: int
    seed: int
    merge_strategy: str
    device: torch.device


def _load_checkpoint(path: Path, device: torch.device):
    """Restore the current RARF model, flow, and preprocessing metadata."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]

    model = create_model(
        image_channels=hparams["image_channels"],
        spatial_dims=hparams["spatial_dims"],
        conditioning_mode=hparams["conditioning_mode"],
        mask_conditioning=hparams["mask_conditioning"],
        model_channels=hparams["model_channels"],
        num_res_blocks=hparams["num_res_blocks"],
        channel_mult=hparams["channel_mult"],
        dropout=hparams["dropout"],
    )
    flow = RARF(
        path_mode=hparams["path_mode"],
        conditioning_mode=hparams["conditioning_mode"],
        time_sampling=hparams["time_sampling"],
        phase_boundary=hparams["phase_boundary"],
        num_steps=hparams["flow_steps"],
        loss_type=hparams["flow_loss"],
        recon_loss_type=hparams["recon_loss"],
        recon_loss_weight=hparams["recon_loss_weight"],
        ode_steps=hparams["sample_steps"],
    )

    # Prefer EMA weights when the training run saved them intentionally.
    ema_state = checkpoint.get("ema_state_dict")
    if hparams["use_ema"] and ema_state is not None:
        model_state = ema_state
        selected_weights = "EMA"
    else:
        model_state = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
        }
        selected_weights = "raw"
    model.load_state_dict(model_state, strict=True)

    data_hparams = checkpoint["datamodule_hyper_parameters"]
    del checkpoint

    model.to(device).eval()
    print(f"Loaded {selected_weights} weights on {device}.", flush=True)
    return model, flow, hparams, data_hparams


def _device() -> torch.device:
    """Return the requested device, failing clearly when CUDA is unavailable."""

    requested = os.environ.get("DEVICE", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _settings(hparams: dict, device: torch.device) -> Settings:
    """Build the small public inference interface from environment variables."""

    settings = Settings(
        sample_steps=int(
            os.environ.get("SAMPLE_STEPS", hparams["sample_steps"])
        ),
        num_samples=int(os.environ.get("NUM_SAMPLES", "1")),
        seed=int(os.environ.get("SEED", "0")),
        merge_strategy=os.environ.get("MERGE_STRATEGY", "mean"),
        device=device,
    )
    if settings.sample_steps < 2:
        raise ValueError("SAMPLE_STEPS must be at least 2.")
    if settings.num_samples < 1:
        raise ValueError("NUM_SAMPLES must be at least 1.")
    if settings.merge_strategy not in {"mean", "closest_to_mean"}:
        raise ValueError("MERGE_STRATEGY must be mean or closest_to_mean.")
    return settings


def _single_file(case_dir: Path, suffix: str) -> Path:
    """Return the unique case file ending in ``suffix``."""

    matches = sorted(case_dir.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Case {case_dir.name}: expected one *{suffix}, found {len(matches)}."
        )
    return matches[0]


def _anatomy_center(image: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Center the crop on the union of visible anatomy and missing region."""

    foreground = (image > 0) | (mask > 0)
    if not np.any(foreground):
        return tuple(size // 2 for size in image.shape)
    coordinates = np.argwhere(foreground)
    return tuple(
        ((coordinates.min(axis=0) + coordinates.max(axis=0) + 1) // 2).tolist()
    )


def _crop_slices(full_shape, crop_shape, center):
    """Return matching source and crop slices for restoring the full volume."""

    source_slices = []
    crop_slices = []
    for size, target_size, midpoint in zip(full_shape, crop_shape, center):
        start = midpoint - target_size // 2
        source_start = max(start, 0)
        source_stop = min(start + target_size, size)
        crop_start = source_start - start
        crop_stop = crop_start + source_stop - source_start
        source_slices.append(slice(source_start, source_stop))
        crop_slices.append(slice(crop_start, crop_stop))
    return tuple(source_slices), tuple(crop_slices)


def _crop_or_pad(
    volume: np.ndarray,
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> np.ndarray:
    """Return a centered crop, padding outside the source volume with zero."""

    source_slices, crop_slices = _crop_slices(volume.shape, shape, center)
    output = np.zeros(shape, dtype=volume.dtype)
    output[crop_slices] = volume[source_slices]
    return output


def _normalization_bounds(voided, lower_percentile, upper_percentile):
    """Return challenge-style intensity bounds from a full voided volume."""

    lower, upper = np.percentile(voided, [lower_percentile, upper_percentile])
    lower = max(float(lower), 0.0)
    upper = float(upper)
    if upper <= lower:
        raise RuntimeError("The normalization upper bound must exceed the lower bound.")
    return lower, upper


def _sample(
    model,
    flow,
    image: torch.Tensor,
    mask: torch.Tensor,
    settings: Settings,
    case_seed: int,
) -> np.ndarray:
    """Generate deterministic samples sequentially and merge them on the CPU."""

    predictions = []
    for sample_index in range(settings.num_samples):
        generator = torch.Generator(device=settings.device).manual_seed(
            case_seed + sample_index
        )
        noise = torch.randn(
            image.shape,
            generator=generator,
            device=settings.device,
            dtype=image.dtype,
        )
        prediction = flow.sample_inpaint(
            model,
            image,
            mask,
            steps=settings.sample_steps,
            low_memory=True,
            initial_noise=noise,
        )
        predictions.append(prediction[0, 0].cpu().numpy())
        print(
            f"Generated sample {sample_index + 1}/{settings.num_samples}.",
            flush=True,
        )

    if len(predictions) == 1:
        return predictions[0]
    stack = np.stack(predictions)
    mean = stack.mean(axis=0)
    if settings.merge_strategy == "mean":
        return mean

    region = mask[0, 0].bool().cpu().numpy()
    distances = ((stack[:, region] - mean[region]) ** 2).mean(axis=1)
    selected = int(np.argmin(distances))
    print(
        f"Selected sample {selected + 1}/{settings.num_samples} closest to mean.",
        flush=True,
    )
    return stack[selected]


def _inpaint_case(
    case_dir: Path,
    output_dir: Path,
    model,
    flow,
    settings: Settings,
    crop_shape: tuple[int, int, int],
    robust_percentile_lower: float,
    robust_percentile: float,
    case_index: int,
) -> str:
    """Load, inpaint, and save one challenge case."""

    image_path = _single_file(case_dir, "-t1n-voided.nii.gz")
    mask_path = _single_file(case_dir, "-mask.nii.gz")
    reference = nib.load(str(image_path))
    voided = np.asarray(reference.dataobj, dtype=np.float32)
    mask = (np.asarray(nib.load(str(mask_path)).dataobj) > 0).astype(np.float32)
    if voided.shape != mask.shape:
        raise RuntimeError(f"Case {case_dir.name}: image and mask shapes differ.")

    voided = voided * (1.0 - mask)
    center = _anatomy_center(voided, mask)
    voided_crop = _crop_or_pad(voided, crop_shape, center)
    mask_crop = _crop_or_pad(mask, crop_shape, center)
    lower, upper = _normalization_bounds(
        voided,
        robust_percentile_lower,
        robust_percentile,
    )

    # The challenge image should already be voided; enforcing the mask keeps
    # preprocessing identical even if residual values are present in the hole.
    normalized = np.clip(voided_crop, lower, upper)
    normalized = ((normalized - lower) / (upper - lower)) * (1.0 - mask_crop)
    image = torch.from_numpy(normalized[None, None]).to(settings.device)
    region_mask = torch.from_numpy(mask_crop[None, None]).to(settings.device)
    prediction = _sample(
        model,
        flow,
        image,
        region_mask,
        settings,
        settings.seed + case_index * 100_000,
    )
    prediction = np.clip(prediction, 0.0, 1.0)
    prediction = prediction * (upper - lower) + lower

    source_slices, crop_slices = _crop_slices(voided.shape, crop_shape, center)
    result = voided.copy()
    destination = result[source_slices]
    active = mask_crop[crop_slices] > 0
    destination[active] = prediction[crop_slices][active]
    result[source_slices] = destination

    output_name = image_path.name.replace("t1n-voided", "t1n-inference")
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(result, reference.affine, header),
        str(output_dir / output_name),
    )
    return output_name


def main() -> None:
    """Run inference over every case directory and validate the flat output."""

    input_dir = Path(os.environ.get("INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    checkpoint_path = Path(os.environ.get("CHECKPOINT_PATH", "/app/model.ckpt"))
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint does not exist: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not case_dirs:
        raise RuntimeError(f"No case directories found below {input_dir}.")

    device = _device()
    model, flow, hparams, data_hparams = _load_checkpoint(checkpoint_path, device)
    settings = _settings(hparams, device)
    try:
        crop_shape = tuple(int(value) for value in data_hparams["crop_shape"])
        robust_percentile_lower = float(data_hparams["robust_percentile_lower"])
        robust_percentile = float(data_hparams["robust_percentile"])
    except KeyError as error:
        raise RuntimeError(
            f"Checkpoint is missing preprocessing setting {error.args[0]!r}."
        ) from error
    if len(crop_shape) != 3 or any(size < 1 for size in crop_shape):
        raise RuntimeError("Checkpoint crop_shape must contain three positive sizes.")
    if not 0.0 <= robust_percentile_lower < robust_percentile <= 100.0:
        raise RuntimeError(
            "Checkpoint percentiles must satisfy 0 <= lower < upper <= 100."
        )

    print(
        f"Cases={len(case_dirs)} samples={settings.num_samples} "
        f"merge={settings.merge_strategy} steps={settings.sample_steps}",
        flush=True,
    )
    expected_names = set()
    for case_index, case_dir in enumerate(case_dirs):
        output_name = _inpaint_case(
            case_dir=case_dir,
            output_dir=output_dir,
            model=model,
            flow=flow,
            settings=settings,
            crop_shape=crop_shape,
            robust_percentile_lower=robust_percentile_lower,
            robust_percentile=robust_percentile,
            case_index=case_index,
        )
        expected_names.add(output_name)
        print(f"[{case_index + 1}/{len(case_dirs)}] {case_dir.name}", flush=True)

    outputs = sorted(output_dir.iterdir())
    if {path.name for path in outputs} != expected_names or not all(
        path.is_file() for path in outputs
    ):
        raise RuntimeError("Output must contain exactly one prediction file per case.")


if __name__ == "__main__":
    main()
