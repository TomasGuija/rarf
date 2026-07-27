"""Generic Lightning module for Region-Aware Rectified Flows."""

from __future__ import annotations

import math
from collections.abc import Mapping

import lightning.pytorch as pl
import torch

from rarf.region_aware_flows import RARF
from rarf.models.utils import create_model


class RARFModule(pl.LightningModule):
    """Train a RARF model from a task-independent batch contract.

    A training batch should be a mapping with a required ``target`` tensor and
    optional ``mask``, ``loss_mask``, and ``recon_loss_mask`` tensors.
    """

    def __init__(
        self,
        # Optimization.
        lr=1e-4,
        weight_decay=0.0,
        # RARF method.
        path_mode="two_phase",
        conditioning_mode="spatial",
        time_sampling="uniform",
        phase_boundary=0.5,
        flow_steps=1000,
        flow_loss="l2",
        recon_loss="mae",
        recon_loss_weight=0.0,
        sample_steps=16,
        # Model.
        image_channels=1,
        spatial_dims=3,
        mask_conditioning=False,
        model_channels=32,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        dropout=0.0,
        # Checkpoint behavior.
        weights_path="",
        use_ema=False,
        ema_decay=0.999,
    ):
        super().__init__()
        self.model = create_model(
            image_channels=image_channels,
            spatial_dims=spatial_dims,
            conditioning_mode=conditioning_mode,
            mask_conditioning=mask_conditioning,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            dropout=dropout,
        )
        self.flow = RARF(
            path_mode=path_mode,
            conditioning_mode=conditioning_mode,
            time_sampling=time_sampling,
            phase_boundary=phase_boundary,
            num_steps=flow_steps,
            loss_type=flow_loss,
            recon_loss_type=recon_loss,
            recon_loss_weight=recon_loss_weight,
            ode_steps=sample_steps,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.sample_steps = sample_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1).")
        self._ema_state = None
        self._ema_backup = None
        self.save_hyperparameters(ignore=["weights_path", "_instantiator"])
        if weights_path:
            self._load_pretrained_weights(weights_path)

    # TODO: Replace the manual EMA lifecycle below with
    # Lightning's EMAWeightAveraging after upgrading from 2.5.5 and migrating
    # existing checkpoints.
    def _load_pretrained_weights(self, path):
        """Warm-start the model and, when available, its EMA state."""

        print(f"[RARFModule] loading pretrained weights from {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=True)

        ema_state = checkpoint.get("ema_state_dict")
        if self.use_ema and ema_state is not None:
            self._load_ema_state(ema_state)

    def on_fit_start(self):
        if self.use_ema and self._ema_state is None:
            self._init_ema_state()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.use_ema:
            self._update_ema_state()

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            if self._ema_state is None:
                raise RuntimeError("EMA is enabled, but its state has not been initialized.")
            checkpoint["ema_state_dict"] = {
                key: value.detach().cpu() for key, value in self._ema_state.items()
            }

    def on_load_checkpoint(self, checkpoint):
        ema_state = checkpoint.get("ema_state_dict")
        if self.use_ema:
            if ema_state is None:
                raise RuntimeError(
                    "EMA is enabled, but the checkpoint has no 'ema_state_dict'."
                )
            self._load_ema_state(ema_state)

    def _init_ema_state(self):
        self._ema_state = {
            key: value.detach().clone() for key, value in self.model.state_dict().items()
        }

    def _load_ema_state(self, ema_state):
        model_state = self.model.state_dict()
        if set(ema_state) != set(model_state):
            raise RuntimeError("EMA state does not match the model state.")
        for key, value in model_state.items():
            if ema_state[key].shape != value.shape:
                raise RuntimeError(f"EMA tensor '{key}' has the wrong shape.")
        self._ema_state = {
            key: value.detach().clone() for key, value in ema_state.items()
        }

    def _update_ema_state(self):
        if self._ema_state is None:
            raise RuntimeError("EMA state has not been initialized.")
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                value = value.detach()
                ema_value = self._ema_state[key]
                if ema_value.device != value.device or ema_value.dtype != value.dtype:
                    ema_value = ema_value.to(device=value.device, dtype=value.dtype)
                    self._ema_state[key] = ema_value
                if value.dtype.is_floating_point:
                    ema_value.mul_(self.ema_decay).add_(value, alpha=1.0 - self.ema_decay)
                else:
                    ema_value.copy_(value)

    def apply_ema_weights(self):
        if self._ema_state is None:
            raise RuntimeError("EMA state has not been initialized.")
        self.model.load_state_dict(
            {
                key: self._ema_state[key].to(
                    device=value.device,
                    dtype=value.dtype,
                )
                for key, value in self.model.state_dict().items()
            },
            strict=True,
        )

    def _swap_to_ema(self):
        if not self.use_ema:
            return
        if self._ema_state is None:
            raise RuntimeError("EMA is enabled, but its state has not been initialized.")
        if self._ema_backup is not None:
            raise RuntimeError("Cannot apply EMA weights: a previous swap is still active.")
        self._ema_backup = {
            key: value.detach().clone() for key, value in self.model.state_dict().items()
        }
        try:
            self.apply_ema_weights()
        except Exception:
            self.model.load_state_dict(self._ema_backup, strict=True)
            self._ema_backup = None
            raise

    def _restore_from_ema(self):
        if not self.use_ema:
            return
        if self._ema_backup is None:
            raise RuntimeError("Cannot restore model weights: no EMA swap is active.")
        self.model.load_state_dict(self._ema_backup, strict=True)
        self._ema_backup = None

    def on_validation_epoch_start(self):
        self._swap_to_ema()

    def on_validation_epoch_end(self):
        self._restore_from_ema()

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, Mapping):
            raise TypeError("RARF training batches must be mappings.")
        if "target" not in batch:
            raise KeyError("RARF training batches require a 'target' tensor.")
        return {
            "target": batch["target"],
            "mask": batch.get("mask"),
            "loss_mask": batch.get("loss_mask"),
            "recon_loss_mask": batch.get("recon_loss_mask"),
        }

    def _shared_step(self, batch, stage, batch_idx):
        inputs = self._unpack_batch(batch)
        losses = self.flow.compute_loss(
            self.model,
            **inputs,
            return_components=True,
        )

        on_step = stage == "train"
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=name == "loss",
                sync_dist=stage == "val",
            )
        if stage == "train":
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1234 + batch_idx)
            return self._shared_step(batch, "val", batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(1000, max(1, total_steps - 1))
        min_lr_factor = 1e-6 / self.lr

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
