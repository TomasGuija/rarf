import math

import torch
import torch.nn as nn


def create_model(
    model_channels,
    num_res_blocks,
    image_channels=1,
    spatial_dims=3,
    conditioning_mode="scalar",
    mask_conditioning=False,
    in_channels=None,
    channel_mult=(1, 1, 2, 2, 4, 4),
    dropout=0.0,
):
    from .unet import UNetModel

    if in_channels is not None:
        if in_channels < image_channels:
            raise ValueError("in_channels cannot be smaller than image_channels.")
        mask_conditioning = in_channels == image_channels + 1
        if in_channels not in {image_channels, image_channels + 1}:
            raise ValueError(
                "Legacy in_channels must equal image_channels or image_channels + 1."
            )

    return UNetModel(
        image_channels=image_channels,
        model_channels=model_channels,
        out_channels=image_channels,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        channel_mult=channel_mult,
        spatial_dims=spatial_dims,
        conditioning_mode=conditioning_mode,
        mask_conditioning=mask_conditioning,
    )


def timestep_embedding(timesteps, dim, max_period=10000):
    if timesteps.ndim != 1:
        raise ValueError("Scalar timestep embedding expects a [B] tensor.")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def spatial_timestep_embedding(timesteps, dim, max_period=10000):
    """Apply sinusoidal timestep encoding independently at every position.

    Args:
        timesteps: Spatial values with shape ``[B, 1, *spatial]``.
        dim: Number of output embedding channels.
    """

    if timesteps.ndim not in (4, 5) or timesteps.shape[1] != 1:
        raise ValueError("Spatial timestep embedding expects [B, 1, H, W] or [B, 1, D, H, W].")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    frequency_shape = (1, half) + (1,) * (timesteps.ndim - 2)
    args = timesteps.float() * freqs.reshape(frequency_shape)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


def zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def normalization(channels):
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return GroupNorm32(groups, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(spatial_dims, *args, **kwargs):
    if spatial_dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if spatial_dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError("spatial_dims must be 2 or 3.")


def avg_pool_nd(spatial_dims, *args, **kwargs):
    if spatial_dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if spatial_dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError("spatial_dims must be 2 or 3.")
