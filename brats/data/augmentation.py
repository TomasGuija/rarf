import torch

from monai.transforms import Compose, RandAffined, RandFlipd


_KEYS = (
    "image",
    "healthy_mask",
    "full_mask",
    "anatomy_mask",
)


def augment_case(
    image: torch.Tensor,
    healthy_mask: torch.Tensor,
    full_mask: torch.Tensor,
    anatomy_mask: torch.Tensor,
    flip_prob: float = 0.5,
    affine_prob: float = 0.25,
    intensity_prob: float = 0.5,
    max_rotation_degrees: float = 5.0,
    max_shift_voxels: float = 4.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply synchronized spatial and intensity augmentation to one case."""

    data = {
        "image": image[None],
        "healthy_mask": healthy_mask[None],
        "full_mask": full_mask[None],
        "anatomy_mask": anatomy_mask[None],
    }

    spatial_transform = Compose(
        [
            RandFlipd(
                keys=_KEYS,
                prob=flip_prob,
                spatial_axis=0,
            ),
            RandAffined(
                keys=_KEYS,
                prob=affine_prob,
                rotate_range=tuple(
                    torch.deg2rad(
                        torch.tensor(max_rotation_degrees)
                    ).item()
                    for _ in range(3)
                ),
                translate_range=(
                    max_shift_voxels,
                    max_shift_voxels,
                    max_shift_voxels,
                ),
                mode=(
                    "bilinear",
                    "nearest",
                    "nearest",
                    "nearest",
                ),
                padding_mode="zeros",
            ),
        ]
    )

    data = spatial_transform(data)

    image = data["image"][0]
    healthy_mask = (data["healthy_mask"][0] > 0.5).float()
    full_mask = (data["full_mask"][0] > 0.5).float()
    anatomy_mask = (data["anatomy_mask"][0] > 0.5).float()

    if torch.rand(()) < intensity_prob:
        body = anatomy_mask > 0

        if torch.any(body):
            gain = torch.empty(()).uniform_(0.9, 1.1)
            bias = torch.empty(()).uniform_(-0.03, 0.03)
            gamma = torch.empty(()).uniform_(0.85, 1.15)

            augmented = torch.clamp(
                image * gain + bias,
                min=0.0,
            ).pow(gamma)

            image = torch.where(
                body,
                augmented,
                torch.zeros_like(image),
            )

    return (
        image,
        healthy_mask,
        full_mask,
        anatomy_mask,
    )
