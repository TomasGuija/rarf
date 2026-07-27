"""Training-only Lightning data module for natural images."""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from rarf.data.image_dataset import NaturalImageDataset, discover_images
from rarf.data.masks import PerlinMaskGenerator


class NaturalImageDataModule(pl.LightningDataModule):
    """Load and split a local directory or Hugging Face image dataset.

    Exactly one of ``dataset_name`` and ``data_dir`` must be provided. The
    source is reproducibly divided into training and validation records.

    ``dataset_config_name`` optionally selects one configuration from a
    Hugging Face dataset that publishes multiple variants. Its records must
    contain the image field named by ``image_column``. An existing
    ``validation`` split is preserved; otherwise, one is sampled from
    ``train`` according to ``validation_size``. For a local ``data_dir``,
    files are discovered recursively and both Hugging Face options are
    ignored.
    """

    def __init__(
        self,
        dataset_name: str | None = None,
        data_dir: str | None = None,
        dataset_config_name: str | None = None,
        image_column: str = "image",
        cache_dir: str | None = None,
        resolution: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = True,
        random_crop: bool = True,
        random_flip: bool = True,
        validation_size: float = 0.05,
        split_seed: int = 42,
        mask_min_scale: float = 1.0,
        mask_max_scale: float | None = None,
        min_hole_fraction: float = 0.1,
        max_hole_fraction: float = 0.9,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["_instantiator"])
        self.train_dataset = None
        self.val_dataset = None

    def _validation_count(self, dataset_size: int) -> int:
        validation_size = self.hparams.validation_size
        if isinstance(validation_size, float) and 0.0 < validation_size < 1.0:
            count = round(dataset_size * validation_size)
        else:
            raise ValueError(
                "validation_size must be a float in (0, 1)."
            )
        if not 0 < count < dataset_size:
            raise ValueError(
                "validation_size must leave at least one image in each split."
            )
        return count

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        if bool(self.hparams.dataset_name) == bool(self.hparams.data_dir):
            raise ValueError("Provide exactly one of dataset_name or data_dir.")

        if self.hparams.dataset_name:
            try:
                from datasets import load_dataset
            except ImportError as error:
                raise ImportError(
                    "Hugging Face datasets support requires `pip install -e '.[images]'`."
                ) from error
            splits = load_dataset(
                self.hparams.dataset_name,
                self.hparams.dataset_config_name,
                cache_dir=self.hparams.cache_dir,
            )
            if "train" not in splits:
                raise ValueError("The Hugging Face dataset has no 'train' split.")
            training_records = splits["train"]
            validation_records = splits.get("validation")
        else:
            training_records = discover_images(self.hparams.data_dir)
            validation_records = None

        if validation_records is None:
            validation_count = self._validation_count(len(training_records))
            indices = torch.randperm(
                len(training_records),
                generator=torch.Generator().manual_seed(self.hparams.split_seed),
            ).tolist()
            validation_indices = indices[:validation_count]
            training_indices = indices[validation_count:]
            validation_records = torch.utils.data.Subset(
                training_records,
                validation_indices,
            )
            training_records = torch.utils.data.Subset(
                training_records,
                training_indices,
            )

        mask_generator = PerlinMaskGenerator(
            min_scale=self.hparams.mask_min_scale,
            max_scale=self.hparams.mask_max_scale,
            min_hole_fraction=self.hparams.min_hole_fraction,
            max_hole_fraction=self.hparams.max_hole_fraction,
        )
        self.train_dataset = NaturalImageDataset(
            training_records,
            image_column=self.hparams.image_column,
            resolution=self.hparams.resolution,
            random_crop=self.hparams.random_crop,
            random_flip=self.hparams.random_flip,
            mask_generator=mask_generator,
        )
        self.val_dataset = NaturalImageDataset(
            validation_records,
            image_column=self.hparams.image_column,
            resolution=self.hparams.resolution,
            random_crop=False,
            random_flip=False,
            mask_generator=mask_generator,
            seed=self.hparams.split_seed,
        )
        print(f"[NaturalImageDataModule] train samples: {len(self.train_dataset)}")
        print(f"[NaturalImageDataModule] val samples: {len(self.val_dataset)}")

    def _dataloader(self, dataset, *, shuffle: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup(stage="fit")
        return self._dataloader(self.train_dataset, shuffle=self.hparams.shuffle)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup(stage="fit")
        return self._dataloader(self.val_dataset, shuffle=False)
