"""Lightning data module for the BraTS challenge layout."""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from brats.data.dataset import BraTSDataset


class BraTSDataModule(pl.LightningDataModule):
    """Build BraTS train/validation datasets from one or two roots.

    With only ``data_dir``, cases are reproducibly split according to
    ``train_split``. When ``val_data_dir`` is provided, every case in
    ``data_dir`` is used for training and every case in ``val_data_dir`` is
    used for validation. Optional ``train_csv`` and ``val_csv`` case lists skip
    case discovery in their respective roots.
    """

    def __init__(
        self,
        data_dir="",
        val_data_dir=None,
        train_csv=None,
        val_csv=None,
        batch_size=1,
        num_workers=4,
        train_split=0.95,
        seed=10,
        shuffle=True,
        crop_shape=(128, 128, 128),
        cache_dir=None,
        robust_percentile_lower=0.5,
        robust_percentile=99.5,
        random_mask_variant=True,
        image_augment=True,
        loss_mask_mode="healthy",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["_instantiator"])
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.seed = seed
        self.shuffle = shuffle
        self.crop_shape = tuple(crop_shape) if crop_shape is not None else None
        self.cache_dir = cache_dir
        self.robust_percentile_lower = robust_percentile_lower
        self.robust_percentile = robust_percentile
        self.random_mask_variant = random_mask_variant
        self.image_augment = image_augment
        self.loss_mask_mode = loss_mask_mode
        self.train_dataset = None
        self.val_dataset = None

    def _make_dataset(
        self, root, *, cases_csv, random_mask_variant, image_augment
    ):
        return BraTSDataset(
            root,
            test_flag=False,
            cases_csv=cases_csv,
            crop_shape=self.crop_shape,
            cache_dir=self.cache_dir,
            robust_percentile_lower=self.robust_percentile_lower,
            robust_percentile=self.robust_percentile,
            random_mask_variant=random_mask_variant,
            image_augment=image_augment,
            loss_mask_mode=self.loss_mask_mode,
        )

    def setup(self, stage=None):
        if not self.data_dir:
            raise ValueError("data_dir is required for BraTS training.")
        if self.val_csv and not self.val_data_dir:
            raise ValueError("val_csv requires val_data_dir.")

        base_train = self._make_dataset(
            self.data_dir,
            cases_csv=self.train_csv,
            random_mask_variant=self.random_mask_variant,
            image_augment=self.image_augment,
        )
        if self.val_data_dir:
            self.train_dataset = base_train
            self.val_dataset = self._make_dataset(
                self.val_data_dir,
                cases_csv=self.val_csv,
                random_mask_variant=False,
                image_augment=False,
            )
        else:
            if not 0.0 < self.train_split < 1.0:
                raise ValueError("train_split must be in (0, 1) when val_data_dir is not provided.")

            base_val = self._make_dataset(
                self.data_dir,
                cases_csv=self.train_csv,
                random_mask_variant=False,
                image_augment=False,
            )
            train_size = max(1, min(int(len(base_train) * self.train_split), len(base_train) - 1))
            indices = torch.randperm(
                len(base_train), generator=torch.Generator().manual_seed(self.seed)
            ).tolist()
            self.train_dataset = torch.utils.data.Subset(
                base_train, indices[:train_size]
            )
            self.val_dataset = torch.utils.data.Subset(
                base_val, indices[train_size:]
            )
        print(f"[BraTSDataModule] train samples: {len(self.train_dataset)}")
        print(f"[BraTSDataModule] val samples: {len(self.val_dataset)}")

    def _loader(self, dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup(stage="fit")
        return self._loader(self.train_dataset, self.shuffle)

    def val_dataloader(self):
        if self.train_dataset is None:
            self.setup(stage="fit")
        return None if self.val_dataset is None else self._loader(self.val_dataset, False)
