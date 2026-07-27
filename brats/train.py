"""BraTS training entry point using the generic RARF Lightning module."""

import logging

import torch

from brats.data.datamodule import BraTSDataModule
from rarf.training.cli import RARFLightningCLI
from rarf.training.module import RARFModule


def main():
    torch.set_float32_matmul_precision("medium")
    RARFLightningCLI(
        model_class=RARFModule,
        datamodule_class=BraTSDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=333,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
