import torch
import torch.nn as nn
import torch.nn.functional as F

from rarf.models.utils import (
    avg_pool_nd,
    conv_nd,
    normalization,
    spatial_timestep_embedding,
    timestep_embedding,
    zero_module,
)


class TimestepBlock(nn.Module):
    """Base class for modules that receive timestep embeddings."""

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential):
    """Sequential container that forwards timestep embeddings when required."""

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_embedding)
            else:
                x = layer(x)

        return x


class Upsample(nn.Module):
    """Upsample spatial dimensions by a factor of two."""

    def __init__(
        self,
        channels: int,
        spatial_dims: int,
        use_conv: bool = True,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.conv = (
            conv_nd(
                spatial_dims,
                channels,
                channels,
                kernel_size=3,
                padding=1,
            )
            if use_conv
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.shape[1]}."
            )

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        )

        if self.conv is not None:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    """Downsample spatial dimensions by a factor of two."""

    def __init__(
        self,
        channels: int,
        spatial_dims: int,
        use_conv: bool = True,
    ) -> None:
        super().__init__()

        self.channels = channels

        if use_conv:
            self.op = conv_nd(
                spatial_dims,
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            self.op = avg_pool_nd(
                spatial_dims,
                kernel_size=2,
                stride=2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.shape[1]}."
            )

        return self.op(x)


class ResBlock(TimestepBlock):
    """Residual block conditioned on scalar or spatial timestep embeddings."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        spatial_dims: int,
        conditioning_mode: str,
        out_channels: int | None = None,
        use_conv: bool = False,
        up: bool = False,
        down: bool = False,
    ) -> None:
        super().__init__()

        if up and down:
            raise ValueError("A residual block cannot both upsample and downsample.")

        if conditioning_mode not in {"scalar", "spatial"}:
            raise ValueError(
                "conditioning_mode must be either 'scalar' or 'spatial'."
            )

        self.channels = channels
        self.out_channels = out_channels or channels
        self.conditioning_mode = conditioning_mode
        self.updown = up or down

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(
                spatial_dims,
                channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        if up:
            self.h_upd = Upsample(
                channels,
                spatial_dims,
                use_conv=False,
            )
            self.x_upd = Upsample(
                channels,
                spatial_dims,
                use_conv=False,
            )
        elif down:
            self.h_upd = Downsample(
                channels,
                spatial_dims,
                use_conv=False,
            )
            self.x_upd = Downsample(
                channels,
                spatial_dims,
                use_conv=False,
            )
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        if conditioning_mode == "scalar":
            embedding_projection = nn.Linear(
                emb_channels,
                self.out_channels,
            )
        else:
            embedding_projection = conv_nd(
                spatial_dims,
                emb_channels,
                self.out_channels,
                kernel_size=1,
            )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            embedding_projection,
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    spatial_dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding=1,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                spatial_dims,
                channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self.skip_connection = conv_nd(
                spatial_dims,
                channels,
                self.out_channels,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.updown:
            input_rest = self.in_layers[:-1]
            input_conv = self.in_layers[-1]

            h = input_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = input_conv(h)
        else:
            h = self.in_layers(x)

        embedding_input = time_embedding
        if (
            self.conditioning_mode == "spatial"
            and embedding_input.shape[2:] != h.shape[2:]
        ):
            embedding_input = F.interpolate(
                embedding_input,
                size=h.shape[2:],
                mode="nearest",
            )

        embedding = self.emb_layers(embedding_input).to(dtype=h.dtype)
        if self.conditioning_mode == "scalar":
            while embedding.ndim < h.ndim:
                embedding = embedding[..., None]

        h = h + embedding
        h = self.out_layers(h)

        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    """U-Net conditioned on scalar or spatial timestep information.

    Args:
        image_channels:
            Number of channels in the input image.
        model_channels:
            Base number of feature channels.
        out_channels:
            Number of output channels.
        num_res_blocks:
            Number of residual blocks per resolution level.
        dropout:
            Dropout probability used in residual blocks.
        channel_mult:
            Channel multiplier for each resolution level.
        spatial_dims:
            Number of spatial dimensions. Must be 2 or 3.
        conditioning_mode:
            Timestep conditioning mode. Either ``"scalar"`` or ``"spatial"``.
        mask_conditioning:
            Whether to concatenate a single-channel mask to the input image.
    """

    def __init__(
        self,
        image_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        spatial_dims: int = 3,
        conditioning_mode: str = "scalar",
        mask_conditioning: bool = False,
    ) -> None:
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be either 2 or 3.")

        if conditioning_mode not in {"scalar", "spatial"}:
            raise ValueError(
                "conditioning_mode must be either 'scalar' or 'spatial'."
            )

        if image_channels <= 0:
            raise ValueError("image_channels must be greater than zero.")

        if model_channels <= 0:
            raise ValueError("model_channels must be greater than zero.")

        if out_channels <= 0:
            raise ValueError("out_channels must be greater than zero.")

        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least one.")

        if not channel_mult:
            raise ValueError("channel_mult must contain at least one value.")

        if any(mult <= 0 for mult in channel_mult):
            raise ValueError("All channel multipliers must be greater than zero.")

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        self.image_channels = image_channels
        self.in_channels = image_channels + int(mask_conditioning)
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = tuple(channel_mult)
        self.spatial_dims = spatial_dims
        self.conditioning_mode = conditioning_mode
        self.mask_conditioning = mask_conditioning

        time_embed_dim = model_channels * 4

        if conditioning_mode == "scalar":
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.time_embed = nn.Sequential(
                conv_nd(
                    spatial_dims,
                    model_channels,
                    time_embed_dim,
                    kernel_size=1,
                ),
                nn.SiLU(),
                conv_nd(
                    spatial_dims,
                    time_embed_dim,
                    time_embed_dim,
                    kernel_size=1,
                ),
            )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(
                        spatial_dims,
                        self.in_channels,
                        model_channels,
                        kernel_size=3,
                        padding=1,
                    )
                )
            ]
        )

        input_block_channels = [model_channels]
        channels = model_channels

        for level, multiplier in enumerate(self.channel_mult):
            for _ in range(num_res_blocks):
                block_channels = multiplier * model_channels

                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            channels=channels,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            spatial_dims=spatial_dims,
                            conditioning_mode=conditioning_mode,
                            out_channels=block_channels,
                        )
                    )
                )

                channels = block_channels
                input_block_channels.append(channels)

            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            channels,
                            spatial_dims,
                        )
                    )
                )
                input_block_channels.append(channels)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=channels,
                emb_channels=time_embed_dim,
                dropout=dropout,
                spatial_dims=spatial_dims,
                conditioning_mode=conditioning_mode,
            ),
            ResBlock(
                channels=channels,
                emb_channels=time_embed_dim,
                dropout=dropout,
                spatial_dims=spatial_dims,
                conditioning_mode=conditioning_mode,
            ),
        )

        self.output_blocks = nn.ModuleList()

        for level, multiplier in reversed(
            list(enumerate(self.channel_mult))
        ):
            for block_index in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                block_channels = model_channels * multiplier

                layers: list[nn.Module] = [
                    ResBlock(
                        channels=channels + skip_channels,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        spatial_dims=spatial_dims,
                        conditioning_mode=conditioning_mode,
                        out_channels=block_channels,
                    )
                ]

                channels = block_channels

                if level > 0 and block_index == num_res_blocks:
                    layers.append(
                        Upsample(
                            channels,
                            spatial_dims,
                        )
                    )

                self.output_blocks.append(
                    TimestepEmbedSequential(*layers)
                )

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            zero_module(
                conv_nd(
                    spatial_dims,
                    channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict an output with the same spatial shape as the input.

        Args:
            x:
                Input tensor with shape ``[B, C, *spatial]``.
            timesteps:
                Scalar or spatial timestep conditioning.
            mask:
                Optional tensor with shape ``[B, 1, *spatial]`` when mask
                conditioning is enabled.
        """
        if self.mask_conditioning:
            if mask is None:
                raise ValueError(
                    "A mask is required when mask_conditioning=True."
                )

            if mask.ndim != x.ndim:
                raise ValueError(
                    "The mask and input must have the same number of dimensions."
                )

            if mask.shape[0] != x.shape[0]:
                raise ValueError(
                    "The mask and input must have the same batch size."
                )

            if mask.shape[1] != 1:
                raise ValueError(
                    f"Expected a single-channel mask, got {mask.shape[1]} channels."
                )

            if mask.shape[2:] != x.shape[2:]:
                raise ValueError(
                    "The mask and input must have matching spatial dimensions."
                )

            x = torch.cat(
                [x, mask.to(dtype=x.dtype)],
                dim=1,
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )

        if self.conditioning_mode == "scalar":
            time_embedding = timestep_embedding(
                timesteps,
                self.model_channels,
            )
        else:
            time_embedding = spatial_timestep_embedding(
                timesteps,
                self.model_channels,
            )

        time_embedding = self.time_embed(time_embedding)

        skip_connections = []
        h = x

        for module in self.input_blocks:
            h = module(h, time_embedding)
            skip_connections.append(h)

        h = self.middle_block(h, time_embedding)

        for module in self.output_blocks:
            skip = skip_connections.pop()

            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(
                    h,
                    size=skip.shape[2:],
                    mode="nearest",
                )

            h = torch.cat([h, skip], dim=1)
            h = module(h, time_embedding)

        return self.out(h)
