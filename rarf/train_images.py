"""Natural-image training entry point for RARF."""

import logging

import torch

from rarf.data import NaturalImageDataModule
from rarf.training.cli import RARFLightningCLI
from rarf.training.module import RARFModule


def main() -> None:
    torch.set_float32_matmul_precision("medium")
    RARFLightningCLI(
        model_class=RARFModule,
        datamodule_class=NaturalImageDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=333,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
