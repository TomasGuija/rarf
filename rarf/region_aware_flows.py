"""Region-Aware Rectified Flow.

Mask convention
---------------
``mask == 1`` identifies the region to generate (the hole) and ``mask == 0``
identifies observed context.

Path modes
----------
``inpaint`` basic inpainting setup: context is fixed at the data
endpoint while the hole follows a straight noise-to-data path.

``two_phase`` follows RAD's regional ordering along one global path.  In the
noise-to-data direction, context is generated first while the hole remains at
noise; the hole is then generated while context remains at data.  Reversing
the path therefore noises the hole first and the context second, as in RAD.

``unconditional`` is an ordinary rectified flow in which every spatial
position shares the same progress value. Not really relevant for inpainting tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def append_dims(value: torch.Tensor, ndims: int) -> torch.Tensor:
    """Append singleton dimensions to a batch-shaped tensor."""

    return value.reshape(*value.shape, *((1,) * ndims))


def spatial_progress(times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the inpainting progress map ``mask * t + (1 - mask)``."""

    padded_times = append_dims(times, mask.ndim - 1)
    return mask * padded_times + (1.0 - mask)


@dataclass(frozen=True)
class RARFTrainingSample:
    """A fully specified regional flow-matching training sample."""

    state: torch.Tensor
    target_velocity: torch.Tensor
    progress: torch.Tensor
    time_condition: torch.Tensor
    active_mask: torch.Tensor
    region_mask: torch.Tensor | None
    times: torch.Tensor
    phase_ids: torch.Tensor
    phase_durations: torch.Tensor


class RARF:
    """Configurable Region-Aware Rectified Flow.

    Args:
        path_mode: Regional path to train. ``inpaint`` keeps context fixed and
            transports only the hole; ``two_phase`` transports context and
            hole consecutively; ``unconditional`` transports every position
            together without a mask.
        conditioning_mode: Give the model a batchwise ``scalar`` time or a
            ``spatial`` progress map.  Two-phase training requires spatial
            conditioning because one scalar cannot describe asynchronous
            regional progress.
        time_sampling: ``uniform`` or ``mixed``.  ``mixed`` does
            clean-biased sampling.
        phase_boundary: Global noise-to-data time at which two-phase training
            switches from generating context to generating the hole.
            Equally sized phases correspond to ``0.5``.
        num_steps: Scale applied to scalar times or spatial progress maps
            before passing them to the model. It does not control ODE
            integration; use ``ode_steps`` for that.
        loss_type: Flow-matching error, either absolute error (``l1``) or
            squared error (``l2``).
        recon_loss_type: One reconstruction loss or a sequence of losses.
            Supported values are ``mae``, ``mse``, and ``ssim``. An empty
            sequence disables reconstruction supervision.
        recon_loss_weight: One weight or a sequence matching
            ``recon_loss_type``. Set weights to ``0`` to disable them.
        ode_steps: Default number of integration points used by the sampling
            methods when ``steps`` is not supplied explicitly.
        odeint_kwargs: Keyword arguments forwarded to ``torchdiffeq.odeint``
            for full-trajectory sampling. The default is midpoint integration
            with absolute and relative tolerances of ``1e-5``.
    """

    PATH_MODES = {"inpaint", "two_phase", "unconditional"}
    CONDITIONING_MODES = {"scalar", "spatial"}
    TIME_SAMPLING_MODES = {"uniform", "mixed"}

    def __init__(
        self,
        *,
        path_mode="two_phase",
        conditioning_mode="spatial",
        time_sampling="uniform",
        phase_boundary=0.5,
        num_steps=1000,
        loss_type="l2",
        recon_loss_type="mae",
        recon_loss_weight=0.0,
        ode_steps=16,
        odeint_kwargs=None,
    ):
        if path_mode not in self.PATH_MODES:
            raise ValueError(f"path_mode must be one of {sorted(self.PATH_MODES)}.")
        if conditioning_mode not in self.CONDITIONING_MODES:
            raise ValueError(f"conditioning_mode must be one of {sorted(self.CONDITIONING_MODES)}.")
        if time_sampling not in self.TIME_SAMPLING_MODES:
            raise ValueError(f"time_sampling must be one of {sorted(self.TIME_SAMPLING_MODES)}.")
        if path_mode == "two_phase" and conditioning_mode != "spatial":
            raise ValueError("two_phase training requires spatial conditioning.")
        if not 0.0 < phase_boundary < 1.0:
            raise ValueError("phase_boundary must be strictly between 0 and 1.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if loss_type not in {"l1", "l2"}:
            raise ValueError("loss_type must be either 'l1' or 'l2'.")
        recon_loss_types = (
            [recon_loss_type]
            if isinstance(recon_loss_type, str)
            else list(recon_loss_type)
        )
        recon_loss_weights = (
            [recon_loss_weight]
            if isinstance(recon_loss_weight, (int, float))
            else list(recon_loss_weight)
        )
        if len(recon_loss_types) != len(recon_loss_weights):
            raise ValueError(
                "recon_loss_type and recon_loss_weight must have the same length."
            )
        unsupported = set(recon_loss_types) - {"mae", "mse", "ssim"}
        if unsupported:
            raise ValueError(
                f"Unsupported reconstruction losses: {sorted(unsupported)}."
            )
        if len(set(recon_loss_types)) != len(recon_loss_types):
            raise ValueError("Reconstruction loss names must be unique.")

        self.path_mode = path_mode
        self.conditioning_mode = conditioning_mode
        self.time_sampling = time_sampling
        self.phase_boundary = phase_boundary
        self.num_steps = num_steps
        self.loss_type = loss_type
        self.recon_loss_type = tuple(recon_loss_types)
        self.recon_loss_weight = tuple(recon_loss_weights)
        self.ode_steps = ode_steps
        self.odeint_kwargs = odeint_kwargs or {
            "atol": 1e-5,
            "rtol": 1e-5,
            "method": "midpoint",
        }

    def _sample_times(self, batch_size, device):
        uniform = torch.rand(batch_size, device=device)
        if self.time_sampling == "uniform":
            return uniform

        clean_biased = 1.0 - torch.rand(batch_size, device=device).square()
        use_clean_biased = torch.rand(batch_size, device=device) < 0.25
        return torch.where(use_clean_biased, clean_biased, uniform)

    @staticmethod
    def _validate_target(target):
        if target.ndim not in (4, 5):
            raise ValueError(
                "Expected a 2D or 3D batch [B, C, ...], "
                f"got shape {tuple(target.shape)}."
            )

    @staticmethod
    def _validate_mask(mask, target):
        if mask is None:
            raise ValueError("A region mask is required for regional path modes.")
        if mask.ndim != target.ndim or mask.shape[0] != target.shape[0]:
            raise ValueError("Mask and target must have matching batch and spatial dimensions.")
        if (
            mask.shape[1] not in (1, target.shape[1])
            or mask.shape[2:] != target.shape[2:]
        ):
            raise ValueError("Mask must have one channel (or one per image channel) and match target space.")
        return mask.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)

    def _time_condition(self, times, progress):
        value = times if self.conditioning_mode == "scalar" else progress
        return value * self.num_steps

    def _two_phase_path(self, times, mask):
        """Describe the two regional stages at each global time.

        Global time runs from ``0`` (all noise) to ``1`` (all clean data). Let
        ``b = phase_boundary``. For ``0 <= t < b``, context progresses from
        noise to data as ``t / b`` while the hole remains fully noisy. For
        ``b <= t <= 1``, context remains clean while the hole progresses as
        ``(t - b) / (1 - b)``.

        Besides the spatial progress map and active region, the method returns
        the duration of the active phase: ``b`` for context or ``1 - b`` for
        the hole. ``make_training_sample`` uses its reciprocal to convert the
        local phase velocity into velocity with respect to global time.
        """

        padded_times = append_dims(times, mask.ndim - 1)
        boundary = self.phase_boundary
        context_progress = (padded_times / boundary).clamp(0.0, 1.0)
        hole_progress = ((padded_times - boundary) / (1.0 - boundary)).clamp(0.0, 1.0)
        progress = (1.0 - mask) * context_progress + mask * hole_progress

        context_phase = times < boundary
        select_context = append_dims(context_phase, mask.ndim - 1)
        active_mask = torch.where(select_context, 1.0 - mask, mask)
        phase_ids = (~context_phase).long()
        phase_durations = torch.where(
            context_phase,
            times.new_full(times.shape, boundary),
            times.new_full(times.shape, 1.0 - boundary),
        )
        return progress, active_mask, phase_ids, phase_durations

    def _sampling_progress(self, times, mask):
        if self.path_mode == "two_phase":
            return self._two_phase_path(times, mask)[0]
        return spatial_progress(times, mask)

    def make_training_sample(
        self,
        target,
        mask=None,
        *,
        noise=None,
        times=None,
    ):
        """Create one rectified-flow training example.

        Starting from noise and a clean ``target``, this builds the state at
        ``times`` and the velocity the model should predict there. The chosen
        path controls which regions are being denoised; in ``two_phase``
        mode, ``times`` selects one point on its single context-then-hole
        trajectory.
        """

        self._validate_target(target)
        batch_size = target.shape[0]
        noise = torch.randn_like(target) if noise is None else noise.to(target)
        if noise.shape != target.shape:
            raise ValueError("Noise and target must have identical shapes.")

        if times is None:
            times = self._sample_times(batch_size, target.device)
        else:
            times = times.to(device=target.device, dtype=target.dtype)
        if times.shape != (batch_size,):
            raise ValueError(f"Expected times with shape {(batch_size,)}, got {times.shape}.")
        if torch.any((times < 0.0) | (times > 1.0)):
            raise ValueError("Training times must lie in [0, 1].")

        padded_times = append_dims(times, target.ndim - 1)

        if self.path_mode == "unconditional":
            progress = padded_times.expand(batch_size, 1, *target.shape[2:])
            active_mask = torch.ones_like(progress)
            region_mask = None
            phase_ids = torch.full((batch_size,), 2, device=target.device, dtype=torch.long)
            phase_durations = torch.ones_like(times)
        else:
            mask = self._validate_mask(mask, target)
            region_mask = mask
            if self.path_mode == "inpaint":
                progress = mask * padded_times + (1.0 - mask)
                active_mask = mask
                phase_ids = torch.ones(batch_size, device=target.device, dtype=torch.long)
                phase_durations = torch.ones_like(times)
            else:
                progress, active_mask, phase_ids, phase_durations = self._two_phase_path(
                    times, mask
                )

        state = noise.lerp(target, progress)
        velocity_scale = append_dims(phase_durations.reciprocal(), target.ndim - 1)
        target_velocity = (target - noise) * active_mask * velocity_scale
        return RARFTrainingSample(
            state=state,
            target_velocity=target_velocity,
            progress=progress,
            time_condition=self._time_condition(times, progress),
            active_mask=active_mask,
            region_mask=region_mask,
            times=times,
            phase_ids=phase_ids,
            phase_durations=phase_durations,
        )

    @staticmethod
    def _model_velocity(model, state, time_condition, region_mask):
        return model(state, time_condition, mask=region_mask)

    @staticmethod
    def _expanded_weight(weight, reference):
        weight = weight.to(device=reference.device, dtype=reference.dtype)
        return torch.broadcast_to(weight, reference.shape)

    @staticmethod
    def _ssim_reconstruction_map(prediction, target, weight):
        """Return the official BraTS-style masked SSIM loss map."""

        from torchmetrics.functional.image import (
            structural_similarity_index_measure,
        )

        region = weight > 0
        prediction = prediction.clamp(0.0, 1.0) * region
        target = target.clamp(0.0, 1.0) * region
        _, ssim_map = structural_similarity_index_measure(
            prediction[:, 0],
            target[:, 0],
            return_full_image=True,
        )
        return (1.0 - ssim_map).unsqueeze(1)

    def compute_loss(
        self,
        model,
        target,
        mask=None,
        loss_mask=None,
        recon_loss_mask=None,
        *,
        return_components=False,
        noise=None,
        times=None,
    ):
        """Compute regional flow matching and optional endpoint losses."""

        sample = self.make_training_sample(
            target,
            mask,
            noise=noise,
            times=times,
        )
        prediction = self._model_velocity(
            model,
            sample.state,
            sample.time_condition,
            sample.region_mask,
        )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Model returned {tuple(prediction.shape)}, expected {tuple(target.shape)}."
            )

        flow_weight = sample.active_mask
        if loss_mask is not None:
            flow_weight = flow_weight * loss_mask.to(flow_weight)
        flow_weight = self._expanded_weight(flow_weight, prediction)

        error = prediction - sample.target_velocity
        element_loss = error.abs() if self.loss_type == "l1" else error.square()
        flow_loss = (element_loss * flow_weight).sum() / flow_weight.sum().clamp_min(1.0)

        recon_weight_mask = flow_weight
        if recon_loss_mask is not None:
            recon_weight_mask = self._expanded_weight(
                sample.active_mask * recon_loss_mask.to(sample.active_mask),
                prediction,
            )
        phase_duration = append_dims(sample.phase_durations, target.ndim - 1)
        remaining_path = phase_duration * (1.0 - sample.progress)
        predicted_endpoint = sample.state + remaining_path * prediction
        remaining_path = remaining_path.clamp_min(1e-3)
        recon_error = (predicted_endpoint - target) / remaining_path
        time_weight = append_dims(
            1.0 + 0.5 * sample.times.square(),
            recon_error.ndim - 1,
        )
        recon_losses = {}
        total = flow_loss
        for loss_type, loss_weight in zip(
            self.recon_loss_type,
            self.recon_loss_weight,
        ):
            if loss_type == "mae":
                recon_element = recon_error.abs()
            elif loss_type == "mse":
                recon_element = recon_error.square()
            else:
                recon_element = self._ssim_reconstruction_map(
                    predicted_endpoint,
                    target,
                    recon_weight_mask,
                )
            recon_value = (recon_element * recon_weight_mask * time_weight).sum()
            recon_value = recon_value / recon_weight_mask.sum().clamp_min(1.0)
            recon_losses[loss_type] = recon_value
            total = total + loss_weight * recon_value

        if return_components:
            aggregate_recon_loss = (
                sum(recon_losses.values())
                if recon_losses
                else flow_loss.new_zeros(())
            )
            components = {
                "loss": total,
                "flow_loss": flow_loss,
                "recon_loss": aggregate_recon_loss,
            }
            if len(recon_losses) > 1:
                components.update(
                    {
                        f"recon_{name}_loss": value
                        for name, value in recon_losses.items()
                    }
                )
            return components
        return total

    @torch.no_grad()
    def sample_inpaint(
        self,
        model,
        image,
        mask,
        steps=None,
        low_memory=False,
        initial_noise=None,
    ):
        """Generate ``mask == 1`` while preserving the supplied context.

        For ``two_phase``, this starts exactly at the global phase boundary
        and integrates only the hole phase, matching RAD inpainting inference.
        """

        self._validate_target(image)
        mask = self._validate_mask(mask, image)
        noise = (
            torch.randn_like(image)
            if initial_noise is None
            else initial_noise.to(image)
        )
        if noise.shape != image.shape:
            raise ValueError("Initial noise and image must have identical shapes.")
        state = noise * mask + image * (1.0 - mask)

        def ode_fn(t, current):
            batch_times = t.expand(current.shape[0])
            progress = self._sampling_progress(batch_times, mask)
            current = current * mask + image * (1.0 - mask)
            condition = self._time_condition(batch_times, progress)
            velocity = self._model_velocity(model, current, condition, mask)
            return velocity * mask

        start_time = (self.phase_boundary if self.path_mode == "two_phase" else 0.0)
        integration_times = torch.linspace(
            start_time,
            1.0,
            steps or self.ode_steps,
            device=image.device,
            dtype=image.dtype,
        )
        if low_memory:
            if self.odeint_kwargs.get("method", "midpoint") != "midpoint":
                raise ValueError(
                    "Low-memory sampling supports only the midpoint solver."
                )
            for start, stop in zip(integration_times[:-1], integration_times[1:]):
                dt = stop - start
                velocity = ode_fn(start, state)
                midpoint = state + velocity * (0.5 * dt)
                state = state + ode_fn(start + 0.5 * dt, midpoint) * dt
        else:
            from torchdiffeq import odeint

            state = odeint(ode_fn, state, integration_times, **self.odeint_kwargs)[-1]
        return state * mask + image * (1.0 - mask)

    @torch.no_grad()
    def sample_two_phase(
        self,
        model,
        initial_noise,
        mask,
        steps=None,
        low_memory=False,
    ):
        """Run the complete RAD-ordered reverse path from noise to data.

        Context is generated over ``[0, phase_boundary]`` while the hole stays
        noisy.  The hole is then generated over ``[phase_boundary, 1]`` while
        context stays fixed.  ``initial_noise`` supplies the output shape.
        """

        if self.path_mode != "two_phase":
            raise ValueError("sample_two_phase requires path_mode='two_phase'.")
        self._validate_target(initial_noise)
        mask = self._validate_mask(mask, initial_noise)
        state = initial_noise.clone()

        def ode_fn(t, current):
            batch_times = t.expand(current.shape[0])
            progress, active_mask, _, _ = self._two_phase_path(batch_times, mask)
            condition = self._time_condition(batch_times, progress)
            velocity = self._model_velocity(model, current, condition, mask)
            return velocity * active_mask

        point_count = max(3, int(steps or self.ode_steps))
        interval_count = point_count - 1
        context_intervals = min(
            interval_count - 1,
            max(1, round(interval_count * self.phase_boundary)),
        )
        hole_intervals = interval_count - context_intervals
        context_times = torch.linspace(
            0.0,
            self.phase_boundary,
            context_intervals + 1,
            device=initial_noise.device,
            dtype=initial_noise.dtype,
        )
        hole_times = torch.linspace(
            self.phase_boundary,
            1.0,
            hole_intervals + 1,
            device=initial_noise.device,
            dtype=initial_noise.dtype,
        )
        integration_times = torch.cat((context_times, hole_times[1:]))

        if low_memory:
            if self.odeint_kwargs.get("method", "midpoint") != "midpoint":
                raise ValueError(
                    "Low-memory sampling supports only the midpoint solver."
                )
            for start, stop in zip(integration_times[:-1], integration_times[1:]):
                dt = stop - start
                velocity = ode_fn(start, state)
                midpoint = state + velocity * (0.5 * dt)
                state = state + ode_fn(start + 0.5 * dt, midpoint) * dt
        else:
            from torchdiffeq import odeint

            state = odeint(ode_fn, state, integration_times, **self.odeint_kwargs)[-1]
        return state
