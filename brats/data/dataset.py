import csv
import hashlib
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from brats.data.augmentation import augment_case

CACHE_VERSION = 8

ROLE_SUFFIXES = {
    "t1n": "-t1n",
    "t1n_voided": "-t1n-voided",
    "mask": "-mask",
    "mask_healthy": "-mask-healthy",
}


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def _normalization_bounds(
    voided: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[float, float]:
    """Return challenge-style intensity bounds from a full voided volume."""

    lower, upper = np.percentile(voided, [lower_percentile, upper_percentile])
    lower = max(float(lower), 0.0)
    upper = float(upper)
    return lower, upper


def _normalize_image(
    volume: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Clip an image to its challenge bounds and map it to ``[0, 1]``."""

    volume = np.clip(volume, lower, upper)
    return ((volume - lower) / (upper - lower)).astype(np.float32)


def _denormalize_image(
    volume: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Map a normalized image back to its original intensity range."""

    return volume.astype(np.float32, copy=False) * (upper - lower) + lower


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32)


def _training_loss_mask(healthy_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Return the BraTS-specific weighting applied after RARF's active mask.

    ``healthy`` preserves the challenge submission objective. ``active`` adds
    no spatial restriction, allowing two-phase RARF to supervise context in
    the first phase and the hole in the second.
    """

    if mode == "healthy":
        return healthy_mask
    if mode == "active":
        return torch.ones_like(healthy_mask)
    raise ValueError("loss_mask_mode must be 'healthy' or 'active'.")


def _anatomy_center(volume: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int, int]:
    foreground = volume > 0
    if mask is not None:
        foreground = foreground | (mask > 0)
    if not np.any(foreground):
        return tuple(size // 2 for size in volume.shape)

    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return tuple(((mins + maxs + 1) // 2).astype(int).tolist())


def _crop_or_pad_centered(
    volume: np.ndarray,
    crop_shape: tuple[int, int, int] | None,
    center: tuple[int, int, int],
    fill_value=0.0,
) -> np.ndarray:
    if crop_shape is None:
        return volume
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    output = np.full(crop_shape, fill_value, dtype=volume.dtype)
    src_slices = []
    dst_slices = []

    for axis, target_size in enumerate(crop_shape):
        start = center[axis] - target_size // 2
        stop = start + target_size
        src_start = max(start, 0)
        src_stop = min(stop, volume.shape[axis])
        dst_start = src_start - start
        dst_stop = dst_start + (src_stop - src_start)
        src_slices.append(slice(src_start, src_stop))
        dst_slices.append(slice(dst_start, dst_stop))

    output[tuple(dst_slices)] = volume[tuple(src_slices)]
    return output


class BraTSDataset(torch.utils.data.Dataset):
    """BraTS-style NIfTI dataset for region-aware inpainting.

    The loader expects all volumes in a case to already share orientation,
    shape, and voxel grid. It crops or pads each case around the visible anatomy
    and mask center, normalizes intensities to [0, 1] using full-volume voided
    image percentiles, and returns tensors shaped as [C, H, W, D].

    Inference cases require `*-t1n-voided.nii.gz` and `*-mask.nii.gz`.
    Training pairs each healthy mask with its combined mask. Training may
    randomly select among numbered pairs, while validation uses the first pair.

    Args:
        directory: Dataset root. Relative paths in ``cases_csv`` are resolved
            against this directory.
        cases_csv: Optional CSV containing the dataset cases. Training rows
            contain ``case_id``, ``t1n``, ``variant_id``, ``mask``, and
            ``mask_healthy``. Inference rows contain ``case_id``,
            ``t1n_voided``, and ``mask``. When provided, the CSV is trusted
            and the dataset root is not inspected during case collection.
    """

    def __init__(
        self,
        directory,
        test_flag=True,
        cases_csv=None,
        crop_shape=(128, 128, 128),
        cache_dir=None,
        robust_percentile_lower=0.5,
        robust_percentile=99.5,
        random_mask_variant=False,
        image_augment=False,
        loss_mask_mode="healthy",
        flip_prob=0.5,
        affine_prob=0.25,
        intensity_prob=0.5,
        max_rotation_degrees=5.0,
        max_shift_voxels=4.0,
    ):
        super().__init__()
        self.directory = Path(directory).expanduser()
        self.source = os.path.abspath(os.fspath(self.directory))
        self.test_flag = test_flag
        self.cases_csv = Path(cases_csv).expanduser() if cases_csv else None
        self.crop_shape = tuple(crop_shape) if crop_shape is not None else None
        self.robust_percentile_lower = robust_percentile_lower
        self.robust_percentile = robust_percentile
        if not 0.0 <= self.robust_percentile_lower < self.robust_percentile <= 100.0:
            raise ValueError(
                "Normalization percentiles must satisfy 0 <= lower < upper <= 100."
            )
        self.random_mask_variant = random_mask_variant and not self.test_flag
        self.image_augment = image_augment and not self.test_flag
        if loss_mask_mode not in {"healthy", "active"}:
            raise ValueError("loss_mask_mode must be 'healthy' or 'active'.")
        self.loss_mask_mode = loss_mask_mode
        self.flip_prob = flip_prob
        self.affine_prob = affine_prob
        self.intensity_prob = intensity_prob
        self.max_rotation_degrees = max_rotation_degrees
        self.max_shift_voxels = max_shift_voxels

        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cases = (
            self._read_cases_csv()
            if self.cases_csv is not None
            else self._collect_cases()
        )
        if not self.cases:
            mode = "inference" if self.test_flag else "training"
            raise FileNotFoundError(
                f"No complete BraTS {mode} cases found under {self.directory}"
            )

    def _read_cases_csv(self):
        """Read trusted case paths without inspecting the data root."""
        cases = {}
        with self.cases_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                case_id = row["case_id"]

                def source_path(column):
                    return self.directory / row[column]

                if self.test_flag:
                    t1n_voided = source_path("t1n_voided")
                    cases[case_id] = {
                        "case_id": case_id,
                        "case_dir": t1n_voided.parent,
                        "t1n_voided": t1n_voided,
                        "mask": source_path("mask"),
                    }
                    continue

                if case_id not in cases:
                    t1n = source_path("t1n")
                    cases[case_id] = {
                        "case_id": case_id,
                        "case_dir": t1n.parent,
                        "t1n": t1n,
                        "mask_variants": [],
                    }
                cases[case_id]["mask_variants"].append(
                    {
                        "variant_id": row["variant_id"],
                        "mask": source_path("mask"),
                        "mask_healthy": source_path("mask_healthy"),
                    }
                )
        return list(cases.values())

    def _collect_cases(self):
        """Collect complete BraTS cases from the dataset directory."""
        if not self.directory.is_dir():
            return []

        case_dirs = [self.directory]
        case_dirs.extend(
            path for path in self.directory.iterdir()
            if path.is_dir()
        )

        required_roles = (
            {"t1n_voided", "mask"}
            if self.test_flag
            else {"t1n"}
        )

        cases = []

        for case_dir in sorted(case_dirs):
            files = {}

            for role, suffix in ROLE_SUFFIXES.items():
                for extension in (".nii.gz", ".nii"):
                    path = case_dir / f"{case_dir.name}{suffix}{extension}"
                    if path.is_file():
                        files[role] = path
                        break

            if not required_roles.issubset(files):
                continue

            if not self.test_flag:
                # Training masks may be either an unnumbered pair or one or
                # more numbered pairs such as mask-healthy-0000/mask-0000.
                mask_variants = self._collect_mask_variants(case_dir, files)
                if not mask_variants:
                    continue
                files["mask_variants"] = mask_variants

            cases.append(
                {
                    "case_id": case_dir.name,
                    "case_dir": case_dir,
                    **files,
                }
            )

        return cases

    def _collect_mask_variants(self, case_dir, files):
        variants = []
        pattern = re.compile(
            rf"^{re.escape(case_dir.name)}-mask-healthy-(\d{{4}})\.nii(?:\.gz)?$"
        )
        for healthy_path in sorted(case_dir.iterdir()):
            match = pattern.match(healthy_path.name)
            if match is None:
                continue
            variant_id = match.group(1)
            for extension in (".nii.gz", ".nii"):
                full_mask = case_dir / f"{case_dir.name}-mask-{variant_id}{extension}"
                if full_mask.is_file():
                    variants.append(
                        {
                            "variant_id": variant_id,
                            "mask_healthy": healthy_path,
                            "mask": full_mask,
                        }
                    )
                    break

        if not variants and {"mask_healthy", "mask"}.issubset(files):
            variants.append(
                {
                    "variant_id": "base",
                    "mask_healthy": files["mask_healthy"],
                    "mask": files["mask"],
                }
            )
        return variants

    def _select_training_variant(self, case):
        variants = case["mask_variants"]
        index = (
            int(torch.randint(len(variants), (1,)).item())
            if self.random_mask_variant and len(variants) > 1
            else 0
        )
        variant = variants[index]
        return {
            **case,
            "mask_variant_id": variant["variant_id"],
            "mask_healthy": variant["mask_healthy"],
            "mask": variant["mask"],
        }

    def _cache_path(self, case):
        if self.cache_dir is None:
            return None

        crop_key = "full" if self.crop_shape is None else "x".join(map(str, self.crop_shape))
        mode = "inference" if self.test_flag else "training"
        source_key = "|".join(
            [
                str(CACHE_VERSION),
                mode,
                crop_key,
                self.source,
                case["case_id"],
                str(case.get("mask_variant_id", "")),
                str(self.robust_percentile_lower),
                str(self.robust_percentile),
            ]
        )
        digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]
        variant = case.get("mask_variant_id")
        variant_key = f"_{variant}" if variant is not None else ""
        return self.cache_dir / (
            f"{case['case_id']}{variant_key}_{mode}_{crop_key}_{digest}.pt"
        )

    def _load_cached(self, case):
        cache_path = self._cache_path(case)
        if cache_path is None or not cache_path.is_file():
            return None
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    def _save_cached(self, case, sample):
        cache_path = self._cache_path(case)
        if cache_path is None:
            return
        tmp_path = cache_path.with_suffix(f".tmp-{os.getpid()}.pt")
        torch.save(sample, tmp_path)
        tmp_path.replace(cache_path)

    def _prepare_inference_case(self, case):
        voided_raw = _load_nifti(case["t1n_voided"])
        mask_raw = _to_binary_mask(_load_nifti(case["mask"]))
        voided_raw = voided_raw * (1.0 - mask_raw)
        center = _anatomy_center(voided_raw, mask_raw)
        lower, upper = _normalization_bounds(
            voided_raw,
            self.robust_percentile_lower,
            self.robust_percentile,
        )
        return {
            "image": torch.from_numpy(
                _crop_or_pad_centered(voided_raw, self.crop_shape, center)
                .astype(np.float32)
            ),
            "mask": torch.from_numpy(
                _crop_or_pad_centered(mask_raw, self.crop_shape, center)
                .astype(np.float32)
            ),
            "normalization_lower": lower,
            "normalization_upper": upper,
        }

    def _prepare_training_case(self, case):
        t1n_raw = _load_nifti(case["t1n"])
        full_mask_raw = _to_binary_mask(_load_nifti(case["mask"]))
        voided_raw = t1n_raw * (1.0 - full_mask_raw)
        center = _anatomy_center(voided_raw, full_mask_raw)
        lower, upper = _normalization_bounds(
            voided_raw,
            self.robust_percentile_lower,
            self.robust_percentile,
        )

        def crop(volume):
            return _crop_or_pad_centered(volume, self.crop_shape, center).astype(np.float32)

        return {
            "t1n": torch.from_numpy(crop(t1n_raw)),
            "anatomy_mask": torch.from_numpy(crop(_to_binary_mask(t1n_raw))),
            "healthy_mask": torch.from_numpy(
                crop(_to_binary_mask(_load_nifti(case["mask_healthy"])))
            ),
            "full_mask": torch.from_numpy(crop(full_mask_raw)),
            "normalization_lower": lower,
            "normalization_upper": upper,
        }

    def _get_prepared_case(self, case):
        cached = self._load_cached(case)
        if cached is not None:
            return cached

        sample = (
            self._prepare_inference_case(case)
            if self.test_flag
            else self._prepare_training_case(case)
        )
        self._save_cached(case, sample)
        return sample

    def __getitem__(self, idx):
        case = self.cases[idx]
        if not self.test_flag:
            case = self._select_training_variant(case)
        prepared = self._get_prepared_case(case)

        if self.test_flag:
            voided = prepared["image"]
            image = torch.from_numpy(
                _normalize_image(
                    voided.numpy(),
                    prepared["normalization_lower"],
                    prepared["normalization_upper"],
                )
            )
            return {
                "image": image[None, ...],
                "mask": prepared["mask"][None, ...],
                "case_id": case["case_id"],
            }

        t1n = prepared["t1n"]
        healthy_mask = prepared["healthy_mask"]
        full_mask = prepared["full_mask"]
        anatomy_mask = prepared["anatomy_mask"]

        target = torch.from_numpy(
            _normalize_image(
                t1n.numpy(),
                prepared["normalization_lower"],
                prepared["normalization_upper"],
            )
        )

        if self.image_augment:
            target, healthy_mask, full_mask, anatomy_mask = augment_case(
                image=target,
                healthy_mask=healthy_mask,
                full_mask=full_mask,
                anatomy_mask=anatomy_mask,
                flip_prob=self.flip_prob,
                affine_prob=self.affine_prob,
                intensity_prob=self.intensity_prob,
                max_rotation_degrees=self.max_rotation_degrees,
                max_shift_voxels=self.max_shift_voxels,
            )

        loss_mask = _training_loss_mask(healthy_mask, self.loss_mask_mode)[None, ...]
        return {
            "target": target[None, ...],
            "mask": full_mask[None, ...],
            "loss_mask": loss_mask,
            "recon_loss_mask": loss_mask,
            "case_id": case["case_id"],
        }

    def __len__(self):
        return len(self.cases)
