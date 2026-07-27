from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm


@dataclass
class Case:
    case_id: str
    t1n_path: Path
    unhealthy_path: Path
    healthy_path: Path | None


@dataclass
class Template:
    mask: np.ndarray
    brain_fraction: float


def _find_case_file(case_dir: Path, suffix: str) -> Path | None:
    for extension in (".nii.gz", ".nii"):
        path = case_dir / f"{case_dir.name}{suffix}{extension}"
        if path.is_file():
            return path
    return None


def discover_cases(root: Path) -> list[Case]:
    cases = []

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue

        t1n_path = _find_case_file(case_dir, "-t1n")
        unhealthy_path = _find_case_file(case_dir, "-mask-unhealthy")

        if t1n_path is None or unhealthy_path is None:
            continue

        cases.append(
            Case(
                case_id=case_dir.name,
                t1n_path=t1n_path,
                unhealthy_path=unhealthy_path,
                healthy_path=_find_case_file(case_dir, "-mask-healthy"),
            )
        )

    return cases


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj) > 0


def brain_mask(t1n: np.ndarray) -> np.ndarray:
    foreground = t1n > 0
    labeled, count = ndimage.label(foreground, np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return foreground
    sizes = ndimage.sum(foreground, labeled, range(1, count + 1))
    largest = labeled == int(np.argmax(sizes)) + 1
    return ndimage.binary_fill_holes(largest)


def extract_templates(
    cases: list[Case],
    min_component_voxels: int,
) -> list[Template]:
    templates = []
    structure = np.ones((3, 3, 3), dtype=bool)
    for case in tqdm(cases, desc="Extracting mask templates", unit="case"):
        t1n = np.asarray(nib.load(str(case.t1n_path)).dataobj)
        anatomy = brain_mask(t1n)
        anatomy_voxels = max(1, int(anatomy.sum()))
        unhealthy = ndimage.binary_fill_holes(load_mask(case.unhealthy_path))
        labeled, count = ndimage.label(unhealthy, structure)

        for component_id in range(1, count + 1):
            component = labeled == component_id
            voxel_count = int(component.sum())
            if voxel_count < min_component_voxels:
                continue
            slices = ndimage.find_objects(component.astype(np.uint8))[0]
            templates.append(
                Template(
                    mask=component[slices],
                    brain_fraction=voxel_count / anatomy_voxels,
                )
            )
    if not templates:
        raise RuntimeError("No valid unhealthy-mask components were found.")
    templates.sort(key=lambda item: item.brain_fraction)
    return templates


def transform_template(template: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    transformed = template.copy()
    for axis in range(3):
        if rng.random() < 0.5:
            transformed = np.flip(transformed, axis=axis)

    for axes in ((0, 1), (1, 2)):
        transformed = ndimage.rotate(
            transformed.astype(np.uint8),
            angle=float(rng.uniform(0.0, 360.0)),
            axes=axes,
            reshape=True,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ) > 0
    return transformed


def sample_template(
    templates: list[Template],
    target_tumor_fraction: float,
    tolerance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    fractions = np.asarray([template.brain_fraction for template in templates])
    insertion = int(np.searchsorted(fractions, target_tumor_fraction))
    inverse_index = len(templates) - 1 - min(insertion, len(templates) - 1)
    half_window = max(1, int(round(len(templates) * tolerance / 2)))
    low = max(0, inverse_index - half_window)
    high = min(len(templates), inverse_index + half_window + 1)
    selected = templates[int(rng.integers(low, high))].mask
    return transform_template(selected, rng)


def place_template(
    template: np.ndarray,
    center: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    starts = [center[axis] - template.shape[axis] // 2 for axis in range(3)]
    stops = [starts[axis] + template.shape[axis] for axis in range(3)]
    if any(start < 0 for start in starts):
        return None
    if any(stops[axis] > shape[axis] for axis in range(3)):
        return None

    placed = np.zeros(shape, dtype=bool)
    slices = tuple(slice(starts[axis], stops[axis]) for axis in range(3))
    placed[slices] = template
    return placed


def intersection_over_union(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def generate_mask(
    templates: list[Template],
    anatomy: np.ndarray,
    unhealthy: np.ndarray,
    target_tumor_fraction: float,
    existing_masks: list[np.ndarray],
    rng: np.random.Generator,
    tumor_dilation: float,
    min_distance: float,
    size_tolerance: float,
    random_points: int,
    min_brain_fraction: float,
    max_iou: float,
    max_attempts: int,
) -> np.ndarray:
    distance = ndimage.distance_transform_edt(~unhealthy)
    valid_centers = np.argwhere(anatomy & (distance > tumor_dilation + min_distance))
    if len(valid_centers) == 0:
        raise RuntimeError("No valid healthy-mask centers remain after distance filtering.")

    for _ in range(max_attempts):
        candidate_ids = rng.integers(0, len(valid_centers), size=max(1, random_points))
        centers = valid_centers[candidate_ids]
        center = tuple(centers[np.argmax(distance[tuple(centers.T)])].tolist())
        template = sample_template(
            templates,
            target_tumor_fraction,
            size_tolerance,
            rng,
        )
        candidate = place_template(template, center, anatomy.shape)
        if candidate is None or not candidate.any():
            continue
        if np.logical_and(candidate, unhealthy).any():
            continue
        if float(np.logical_and(candidate, anatomy).sum() / candidate.sum()) < min_brain_fraction:
            continue
        if float(distance[candidate].min()) < tumor_dilation + min_distance:
            continue
        if any(intersection_over_union(candidate, other) > max_iou for other in existing_masks):
            continue
        return candidate

    raise RuntimeError(f"Could not generate a valid mask after {max_attempts} attempts.")


def save_mask(mask: np.ndarray, reference: nib.Nifti1Image, path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header),
        str(path),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiple healthy inpainting masks per BRATS case using "
            "transformed connected components from the dataset's unhealthy masks."
        )
    )
    parser.add_argument("root", type=Path, help="BRATS challenge training-data root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to root and adds numbered masks beside existing files.",
    )
    parser.add_argument("--samples-per-case", type=int, default=5)
    parser.add_argument(
        "--include-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the existing unnumbered healthy mask as variant 0000.",
    )
    parser.add_argument("--min-component-voxels", type=int, default=800)
    parser.add_argument(
        "--tumor-dilation",
        type=float,
        default=0.0,
        help=(
            "Additional dilation applied through distance filtering. Keep 0 for "
            "challenge mask-unhealthy files, which are already dilated."
        ),
    )
    parser.add_argument("--min-distance", type=float, default=5.0)
    parser.add_argument("--size-tolerance", type=float, default=0.1)
    parser.add_argument("--random-points", type=int, default=2)
    parser.add_argument("--min-brain-fraction", type=float, default=0.75)
    parser.add_argument("--max-iou", type=float, default=0.8)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else root
    )
    cases = discover_cases(root)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise RuntimeError(f"No complete cases found under {root}")
    if args.samples_per_case <= 0:
        raise ValueError("--samples-per-case must be positive.")

    templates = extract_templates(cases, args.min_component_voxels)
    for case_index, case in enumerate(
        tqdm(cases, desc="Generating healthy masks", unit="case")
    ):
        rng = np.random.default_rng(args.seed + case_index)
        t1n_img = nib.load(str(case.t1n_path))
        t1n = np.asarray(t1n_img.dataobj)
        anatomy = brain_mask(t1n)
        unhealthy = load_mask(case.unhealthy_path)
        target_fraction = float(unhealthy.sum() / max(1, anatomy.sum()))
        generated = []
        if args.include_original:
            if case.healthy_path is None:
                raise FileNotFoundError(
                    f"Missing original healthy mask for {case.case_id}"
                )
            original = load_mask(case.healthy_path)
            generated.append(original)

        while len(generated) < args.samples_per_case:
            mask = generate_mask(
                templates=templates,
                anatomy=anatomy,
                unhealthy=unhealthy,
                target_tumor_fraction=target_fraction,
                existing_masks=generated,
                rng=rng,
                tumor_dilation=args.tumor_dilation,
                min_distance=args.min_distance,
                size_tolerance=args.size_tolerance,
                random_points=args.random_points,
                min_brain_fraction=args.min_brain_fraction,
                max_iou=args.max_iou,
                max_attempts=args.max_attempts,
            )
            generated.append(mask)

        output_case = output_root / case.case_id
        reference = nib.load(str(case.unhealthy_path))
        for index, healthy in enumerate(generated):
            full_mask = np.logical_or(unhealthy, healthy)
            save_mask(
                healthy,
                reference,
                output_case / f"{case.case_id}-mask-healthy-{index:04d}.nii.gz",
                args.overwrite,
            )
            save_mask(
                full_mask,
                reference,
                output_case / f"{case.case_id}-mask-{index:04d}.nii.gz",
                args.overwrite,
            )
    print(f"Wrote masks for {len(cases)} cases to {output_root}")


if __name__ == "__main__":
    main()
