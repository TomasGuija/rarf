"""Checkpoint loading shared by task-specific inference applications."""

import torch

from rarf.training.module import RARFModule


def load_rarf_checkpoint(
    checkpoint_path,
    device="cpu",
    weights="auto",
    return_data_hparams=False,
):
    """Load raw or EMA inference weights without Lightning CLI state."""

    if weights not in {"auto", "ema", "raw"}:
        raise ValueError("weights must be 'auto', 'ema', or 'raw'.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    data_hparams = checkpoint["datamodule_hyper_parameters"]
    module = RARFModule(**hparams)
    module.load_state_dict(checkpoint["state_dict"], strict=True)

    ema_state = checkpoint.get("ema_state_dict")
    use_ema = weights == "ema" or (
        weights == "auto" and ema_state is not None and hparams["use_ema"]
    )
    if weights == "ema" and ema_state is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain EMA weights.")
    if use_ema:
        module.model.load_state_dict(ema_state, strict=True)

    module.use_ema = False
    del checkpoint, ema_state
    module.to(device)
    module.eval()
    selected_weights = "ema" if use_ema else "raw"
    if return_data_hparams:
        return module, selected_weights, data_hparams
    return module, selected_weights
