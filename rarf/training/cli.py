"""Reusable Lightning CLI helpers for task-specific training entry points."""

import logging

import yaml
from lightning.pytorch.cli import LightningCLI


class RARFLightningCLI(LightningCLI):
    """Lightning CLI shared by RARF task packages."""

    def before_fit(self):
        self._dump_config()
        config = self.config_dump
        logging.info("Loaded configuration:\n%s", yaml.safe_dump(config, sort_keys=False))
