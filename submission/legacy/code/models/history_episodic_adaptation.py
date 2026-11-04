"""History-episodic adaptation operator for RealPDE Track 1.

The model reuses the spatial encoder, causal trajectory mixer and transport
decomposition from :mod:`models.temporal_transport`, but makes two dimensions
dynamic at call time:

* an episode may contain 12, 16 or 20 observed frames;
* the same model may predict a four-frame support suffix or the twenty-frame
  competition horizon.

Adaptation is deliberately functional.  A caller creates a fresh per-sample
``[B, hidden_channels]`` fast vector, obtains its support-loss gradient, and
passes the result of one normalized fixed-radius step to the query forward.
No sample-dependent tensor is stored on the module, so calls and processes
cannot leak adaptation state into one another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_transport import (
    CausalTemporalMixer,
    HorizonQueryDecoder,
    SharedFrameEncoder,
    TemporalTransportConfig,
    TemporalTransportOutput,
)
from .unet2d import ResidualConvBlock


AdaptationMode = Literal["static", "self", "donor"]
ADAPTATION_MODES: tuple[str, ...] = ("static", "self", "donor")


@dataclass(frozen=True)
class HistoryEpisodicAdaptationConfig:
    """Architecture and functional-adaptation contract for HEAT."""

    max_context_steps: int = 20
    max_output_steps: int = 20
    allowed_context_steps: tuple[int, ...] = (12, 16, 20)
    allowed_output_steps: tuple[int, ...] = (4, 20)
    in_channels: int = 3
    out_channels: int = 3
    hidden_channels: int = 64
    spatial_blocks: int = 2
    spatial_downsample: int = 2
    temporal_blocks: int = 3
    temporal_kernel: int = 5
    mixer_mode: str = "causal_temporal"
    groups: int = 8
    use_coordinates: bool = True
    max_displacement: float = 4.0
    residual_limit: float = 4.0
    output_initialization_std: float = 1.0e-3
    fast_radius: float = 0.10
    fast_epsilon: float = 1.0e-8
    predict_uncertainty: bool = False

    def __post_init__(self) -> None:
        # Architecture dictionaries may have passed through JSON, where tuples
        # are represented as lists.  Canonicalize them for strict round trips.
        object.__setattr__(
            self, "allowed_context_steps",
            tuple(self.allowed_context_steps))
        object.__setattr__(
            self, "allowed_output_steps",
            tuple(self.allowed_output_steps))

    def validate(self) -> None:
        if self.max_context_steps <= 0 or self.max_output_steps <= 0:
            raise ValueError("maximum context/output steps must be positive")
        if not self.allowed_context_steps or not self.allowed_output_steps:
            raise ValueError("allowed step sets must be non-empty")
        if len(set(self.allowed_context_steps)) != len(
                self.allowed_context_steps):
            raise ValueError("allowed_context_steps must be unique")
        if len(set(self.allowed_output_steps)) != len(
                self.allowed_output_steps):
            raise ValueError("allowed_output_steps must be unique")
        if any(
                not isinstance(value, int) or value <= 0
                or value > self.max_context_steps
                for value in self.allowed_context_steps):
            raise ValueError(
                "allowed context steps must be positive integers no larger "
                "than max_context_steps")
        if any(
                not isinstance(value, int) or value <= 0
                or value > self.max_output_steps
                for value in self.allowed_output_steps):
            raise ValueError(
                "allowed output steps must be positive integers no larger "
                "than max_output_steps")
        if self.max_context_steps not in self.allowed_context_steps:
            raise ValueError(
                "max_context_steps must be one of allowed_context_steps")
        if self.max_output_steps not in self.allowed_output_steps:
            raise ValueError(
                "max_output_steps must be one of allowed_output_steps")
        if self.fast_radius <= 0.0:
            raise ValueError("fast_radius must be positive")
        if self.fast_epsilon <= 0.0:
            raise ValueError("fast_epsilon must be positive")
        self.temporal_transport_config().validate()

    def temporal_transport_config(self) -> TemporalTransportConfig:
        """Return the fixed-size component configuration used internally."""
        return TemporalTransportConfig(
            in_steps=self.max_context_steps,
            out_steps=self.max_output_steps,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            spatial_blocks=self.spatial_blocks,
            spatial_downsample=self.spatial_downsample,
            temporal_blocks=self.temporal_blocks,
            temporal_kernel=self.temporal_kernel,
            mixer_mode=self.mixer_mode,
            groups=self.groups,
            use_coordinates=self.use_coordinates,
            max_displacement=self.max_displacement,
            residual_limit=self.residual_limit,
            output_initialization_std=self.output_initialization_std,
            predict_uncertainty=self.predict_uncertainty,
        )


def fresh_fast_vector(
        batch_size: int,
        hidden_channels: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
        ) -> torch.Tensor:
    """Create a zero fast vector without reading or changing module state."""
    if batch_size <= 0 or hidden_channels <= 0:
        raise ValueError("batch_size and hidden_channels must be positive")
    return torch.zeros(
        batch_size,
        hidden_channels,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def per_sample_fast_gradients(
        per_sample_losses: torch.Tensor,
        fast_vector: torch.Tensor,
        *,
        create_graph: bool = False,
        retain_graph: bool | None = None,
        ) -> torch.Tensor:
    """Differentiate independent scalar episode losses w.r.t. fast vectors.

    Summing the losses is safe because every row of ``fast_vector`` belongs to
    exactly one sample.  The returned tensor retains the batch dimension so a
    trainer can permute rows for the donor control without mixing features or
    targets inside the model.
    """
    if fast_vector.ndim != 2:
        raise ValueError("fast_vector must have shape [B, hidden_channels]")
    if per_sample_losses.ndim != 1:
        raise ValueError("per_sample_losses must have shape [B]")
    if per_sample_losses.shape[0] != fast_vector.shape[0]:
        raise ValueError("loss and fast-vector batch dimensions must match")
    if not fast_vector.requires_grad:
        raise ValueError("fast_vector must require gradients")
    if retain_graph is None:
        retain_graph = create_graph
    gradient, = torch.autograd.grad(
        per_sample_losses.sum(),
        fast_vector,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    return gradient


def normalized_fixed_radius_update(
        fast_vector: torch.Tensor,
        gradient: torch.Tensor,
        *,
        radius: float,
        epsilon: float = 1.0e-8,
        ) -> torch.Tensor:
    """Take one per-sample gradient step with a fixed Euclidean radius.

    Every non-zero gradient row receives exactly ``radius`` displacement.
    A zero row remains unchanged rather than being divided by ``epsilon``.
    Inputs are never modified in place.
    """
    if fast_vector.ndim != 2 or gradient.shape != fast_vector.shape:
        raise ValueError(
            "fast_vector and gradient must have identical [B,C] shape")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not torch.isfinite(gradient).all():
        raise ValueError("gradient must be finite")
    norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
    safe_direction = gradient / norm.clamp_min(epsilon)
    step = torch.where(
        norm > epsilon,
        safe_direction * gradient.new_tensor(radius),
        torch.zeros_like(gradient),
    )
    return fast_vector - step


def _validate_donor_indices(
        donor_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    if donor_indices.ndim != 1 or donor_indices.shape[0] != batch_size:
        raise ValueError("donor_indices must have shape [B]")
    if donor_indices.dtype != torch.long:
        raise ValueError("donor_indices must use torch.long dtype")
    if donor_indices.numel() and (
            int(donor_indices.min()) < 0
            or int(donor_indices.max()) >= batch_size):
        raise ValueError("donor_indices contain an out-of-range row")
    return donor_indices


def one_step_fast_adaptation(
        fast_vector: torch.Tensor,
        self_gradient: torch.Tensor,
        *,
        mode: AdaptationMode,
        radius: float,
        epsilon: float = 1.0e-8,
        donor_indices: torch.Tensor | None = None,
        donor_gradient: torch.Tensor | None = None,
        ) -> torch.Tensor:
    """Apply the trainer-facing static/self/donor adaptation semantics.

    ``static`` returns the input exactly.  ``self`` uses the aligned support
    gradient.  ``donor`` accepts either an explicitly computed donor gradient
    or row indices into the current batch's support gradients.  The two donor
    specifications are mutually exclusive.
    """
    if mode not in ADAPTATION_MODES:
        raise ValueError(
            f"mode must be one of {ADAPTATION_MODES}, got {mode!r}")
    if self_gradient.shape != fast_vector.shape:
        raise ValueError("self_gradient must match fast_vector shape")
    if mode == "static":
        if donor_indices is not None or donor_gradient is not None:
            raise ValueError("static adaptation does not accept donor data")
        return fast_vector
    if mode == "self":
        if donor_indices is not None or donor_gradient is not None:
            raise ValueError("self adaptation does not accept donor data")
        selected_gradient = self_gradient
    else:
        if (donor_indices is None) == (donor_gradient is None):
            raise ValueError(
                "donor adaptation requires exactly one of donor_indices or "
                "donor_gradient")
        if donor_gradient is not None:
            if donor_gradient.shape != fast_vector.shape:
                raise ValueError("donor_gradient must match fast_vector shape")
            selected_gradient = donor_gradient
        else:
            assert donor_indices is not None
            donor_indices = _validate_donor_indices(
                donor_indices, fast_vector.shape[0])
            donor_indices = donor_indices.to(device=self_gradient.device)
            selected_gradient = self_gradient.index_select(0, donor_indices)
    return normalized_fixed_radius_update(
        fast_vector,
        selected_gradient,
        radius=radius,
        epsilon=epsilon,
    )


class VariableHorizonQueryDecoder(HorizonQueryDecoder):
    """A horizon decoder whose learned maximum query bank can be sliced."""

    def forward(
            self,
            trajectory: torch.Tensor,
            output_steps: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory.ndim != 5:
            raise ValueError("query decoder requires [B,T,C,H,W]")
        if output_steps <= 0 or output_steps > self.out_steps:
            raise ValueError("output_steps exceed the learned query bank")
        batch, _steps, channels, height, width = trajectory.shape
        keys = self._project_frames(self.key, trajectory)
        values = self._project_frames(self.value, trajectory)
        queries = self.query_projection(self.query[:output_steps]).to(
            trajectory)
        scores = torch.einsum(
            "btdhw,od->bothw", keys, queries) / self.scale
        attention = scores.float().softmax(dim=2).to(values)
        context = torch.einsum(
            "bothw,btdhw->bodhw", attention, values)
        horizon = queries.view(1, output_steps, channels, 1, 1)
        decoded = context + trajectory[:, -1:].expand(
            -1, output_steps, -1, -1, -1) + horizon
        decoded = self.decoder(decoded.reshape(
            batch * output_steps, channels, height, width))
        return decoded.reshape(
            batch, output_steps, channels, height, width), attention


class HistoryEpisodicForecaster(nn.Module):
    """Variable-context temporal transport with an explicit fast state."""

    def __init__(self, config: HistoryEpisodicAdaptationConfig):
        super().__init__()
        config.validate()
        self.config = config
        component_config = config.temporal_transport_config()
        self.frame_encoder = SharedFrameEncoder(component_config)
        self.temporal_mixer = CausalTemporalMixer(component_config)
        self.horizon_decoder = VariableHorizonQueryDecoder(component_config)
        channels = config.hidden_channels
        self.mean_decoder = ResidualConvBlock(
            channels, channels, config.groups)
        self.mean_head = nn.Conv2d(channels, config.out_channels, 1)
        self.fluctuation_head = nn.Conv2d(
            channels, config.out_channels, 1)
        self.transport_head = nn.Conv2d(channels, 2, 1)
        self._initialize_output_heads()
        self.register_buffer(
            "persistence_scale", torch.ones(config.out_channels))
        self.register_buffer(
            "persistence_bias", torch.zeros(config.out_channels))

    def _initialize_output_heads(self) -> None:
        std = self.config.output_initialization_std
        for head in (
                self.mean_head, self.fluctuation_head,
                self.transport_head):
            if std:
                nn.init.normal_(head.weight, std=std)
            else:
                nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def architecture_config(self) -> dict:
        return {"architecture": "history_episodic_adaptation", **asdict(
            self.config)}

    @torch.no_grad()
    def set_normalizer_statistics(
            self, mean_input: torch.Tensor, mean_target: torch.Tensor,
            std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        channels = self.config.out_channels
        device = self.persistence_scale.device
        mean_input = torch.as_tensor(mean_input, device=device)
        mean_target = torch.as_tensor(mean_target, device=device)
        std_input = torch.as_tensor(std_input, device=device)
        std_target = torch.as_tensor(std_target, device=device)
        safe_target = std_target[:channels].clamp_min(1.0e-8)
        self.persistence_scale.copy_(
            (std_input[:channels] / safe_target).to(self.persistence_scale))
        self.persistence_bias.copy_((
            (mean_input[:channels] - mean_target[:channels]) / safe_target
        ).to(self.persistence_bias))

    def new_fast_vector(
            self,
            batch_size: int,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
            requires_grad: bool = False,
            ) -> torch.Tensor:
        """Return a fresh state; the model never retains the returned tensor."""
        if device is None:
            device = self.persistence_scale.device
        if dtype is None:
            dtype = self.persistence_scale.dtype
        return fresh_fast_vector(
            batch_size,
            self.config.hidden_channels,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    def adapt_fast_state(
            self,
            fast_state: torch.Tensor,
            self_gradient: torch.Tensor,
            *,
            mode: AdaptationMode,
            donor_indices: torch.Tensor | None = None,
            donor_gradient: torch.Tensor | None = None,
            ) -> torch.Tensor:
        return one_step_fast_adaptation(
            fast_state,
            self_gradient,
            mode=mode,
            radius=self.config.fast_radius,
            epsilon=self.config.fast_epsilon,
            donor_indices=donor_indices,
            donor_gradient=donor_gradient,
        )

    def _validate_history(self, history: torch.Tensor) -> None:
        if (history.ndim != 5
                or history.shape[1] not in self.config.allowed_context_steps
                or history.shape[-1] != self.config.in_channels):
            raise ValueError(
                "expected history [B,T,H,W,C] with T in "
                f"{self.config.allowed_context_steps} and C="
                f"{self.config.in_channels}, got {tuple(history.shape)}")

    def _validate_output_steps(self, output_steps: int) -> None:
        if output_steps not in self.config.allowed_output_steps:
            raise ValueError(
                f"output_steps must be in {self.config.allowed_output_steps}, "
                f"got {output_steps}")

    def _validate_fast_state(
            self, fast_state: torch.Tensor, history: torch.Tensor) -> None:
        expected = (history.shape[0], self.config.hidden_channels)
        if fast_state.shape != expected:
            raise ValueError(
                f"fast_state must have shape {expected}, got "
                f"{tuple(fast_state.shape)}")
        if fast_state.device != history.device:
            raise ValueError("fast_state and history must share a device")
        if fast_state.dtype != history.dtype:
            raise ValueError("fast_state and history must share a dtype")

    def _persistence(self, history: torch.Tensor) -> torch.Tensor:
        channels = self.config.out_channels
        return (
            history[:, -1, ..., :channels]
            * self.persistence_scale.view(1, 1, 1, channels)
            + self.persistence_bias.view(1, 1, 1, channels))

    @staticmethod
    def _resize_horizon_field(
            value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        batch, steps, channels, height, width = value.shape
        if (height, width) != size:
            value = F.interpolate(
                value.reshape(batch * steps, channels, height, width),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, steps, channels, *size)
        return value.permute(0, 1, 3, 4, 2)

    def encode_frames(self, history: torch.Tensor) -> torch.Tensor:
        self._validate_history(history)
        return self.frame_encoder(history)

    def encode_history(
            self,
            history: torch.Tensor,
            *,
            fast_state: torch.Tensor | None = None,
            ) -> torch.Tensor:
        trajectory = self.temporal_mixer(self.encode_frames(history))
        if fast_state is None:
            return trajectory
        self._validate_fast_state(fast_state, history)
        # The episode state directly perturbs only the terminal causal state.
        # Concatenation avoids mutating either the trajectory or fast vector.
        adapted_terminal = trajectory[:, -1:] + fast_state.view(
            history.shape[0], 1, self.config.hidden_channels, 1, 1)
        return torch.cat((trajectory[:, :-1], adapted_terminal), dim=1)

    def forward_components(
            self,
            history: torch.Tensor,
            *,
            output_steps: int | None = None,
            fast_state: torch.Tensor | None = None,
            ) -> TemporalTransportOutput:
        self._validate_history(history)
        if output_steps is None:
            output_steps = self.config.max_output_steps
        self._validate_output_steps(output_steps)
        if fast_state is None:
            fast_state = self.new_fast_vector(
                history.shape[0],
                device=history.device,
                dtype=history.dtype,
            )
        self._validate_fast_state(fast_state, history)

        target_size = tuple(history.shape[2:4])
        trajectory = self.encode_history(
            history, fast_state=fast_state)
        decoded, attention = self.horizon_decoder(
            trajectory, output_steps=output_steps)
        batch, steps, channels, latent_height, latent_width = decoded.shape
        flat = decoded.reshape(
            batch * steps, channels, latent_height, latent_width)

        raw_displacement = self.transport_head(flat).reshape(
            batch, steps, 2, latent_height, latent_width)
        displacement = self._resize_horizon_field(
            raw_displacement, target_size)
        displacement = self.config.max_displacement * torch.tanh(
            displacement)
        transported = self._warp(self._persistence(history), displacement)

        raw_fluctuation = self.fluctuation_head(flat).reshape(
            batch, steps, self.config.out_channels,
            latent_height, latent_width)
        fluctuation = self._resize_horizon_field(
            raw_fluctuation, target_size)
        fluctuation = self.config.residual_limit * torch.tanh(
            fluctuation / self.config.residual_limit)
        fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)

        mean_feature = self.mean_decoder(trajectory[:, -1])
        mean_delta = self.mean_head(mean_feature)
        if mean_delta.shape[-2:] != target_size:
            mean_delta = F.interpolate(
                mean_delta,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        mean_delta = mean_delta.permute(0, 2, 3, 1)
        mean_delta = self.config.residual_limit * torch.tanh(
            mean_delta / self.config.residual_limit)

        prediction = transported + mean_delta.unsqueeze(1) + fluctuation
        return TemporalTransportOutput(
            prediction=prediction,
            transported=transported,
            mean_delta=mean_delta,
            fluctuation=fluctuation,
            displacement=displacement,
            attention=attention,
        )

    @staticmethod
    def _warp(
            source: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """Deterministic border-padded bilinear transport.

        This has the same pixel-space convention as
        :meth:`TemporalTransportOperator.warp`, but avoids CUDA
        ``grid_sample`` backward.  That kernel uses atomic accumulation and
        cannot satisfy the formal deterministic-algorithm contract.  In HEAT
        the source is observed data and does not require gradients; only the
        bilinear weights need to carry gradients to ``displacement``.
        """
        if source.ndim != 4:
            raise ValueError("transport source must be [B,H,W,C]")
        if (displacement.ndim != 5
                or displacement.shape[0] != source.shape[0]
                or displacement.shape[2:4] != source.shape[1:3]
                or displacement.shape[-1] != 2):
            raise ValueError(
                "displacement must be [B,T,H,W,2] aligned with source")
        batch, height, width, _channels = source.shape
        work_source = source.float()
        work_displacement = displacement.float()
        x_base = torch.arange(
            width, device=source.device, dtype=torch.float32).view(
                1, 1, 1, width)
        y_base = torch.arange(
            height, device=source.device, dtype=torch.float32).view(
                1, 1, height, 1)
        x = (x_base + work_displacement[..., 0]).clamp(
            0.0, float(max(width - 1, 0)))
        y = (y_base + work_displacement[..., 1]).clamp(
            0.0, float(max(height - 1, 0)))
        x0 = x.floor().to(torch.long)
        y0 = y.floor().to(torch.long)
        x1 = (x0 + 1).clamp_max(width - 1)
        y1 = (y0 + 1).clamp_max(height - 1)
        wx = (x - x0).unsqueeze(-1)
        wy = (y - y0).unsqueeze(-1)
        batch_index = torch.arange(
            batch, device=source.device).view(batch, 1, 1, 1)
        top_left = work_source[batch_index, y0, x0]
        top_right = work_source[batch_index, y0, x1]
        bottom_left = work_source[batch_index, y1, x0]
        bottom_right = work_source[batch_index, y1, x1]
        top = top_left + wx * (top_right - top_left)
        bottom = bottom_left + wx * (bottom_right - bottom_left)
        return (top + wy * (bottom - top)).to(source)

    def _history_in_target_coordinates(
            self, history: torch.Tensor) -> torch.Tensor:
        """Map normalized observations into normalized forecast coordinates."""
        channels = self.config.out_channels
        return (
            history[..., :channels]
            * self.persistence_scale.view(1, 1, 1, 1, channels)
            + self.persistence_bias.view(1, 1, 1, 1, channels))

    def support_loss(
            self,
            history: torch.Tensor,
            fast_state: torch.Tensor,
            ) -> torch.Tensor:
        """Return one support MSE per episode using only its 20 input frames.

        Two four-frame tasks emphasize recent dynamics while keeping every
        label inside the observed history::

            history[0:12] -> history[12:16]
            history[4:16] -> history[16:20]

        The model output lives in target-normalized coordinates, so observed
        suffixes are mapped with the same affine normalizer contract used by
        persistence before the error is formed.
        """
        self._validate_history(history)
        if history.shape[1] != self.config.max_context_steps:
            raise ValueError(
                "support_loss requires a complete max-context episode")
        if self.config.max_context_steps != 20:
            raise ValueError(
                "the frozen HEAT support split requires max_context_steps=20")
        if 12 not in self.config.allowed_context_steps:
            raise ValueError("support_loss requires 12-step contexts")
        if 4 not in self.config.allowed_output_steps:
            raise ValueError("support_loss requires four-step outputs")
        self._validate_fast_state(fast_state, history)

        target_coordinates = self._history_in_target_coordinates(history)
        first_prediction = self(
            history[:, :12], fast_state=fast_state, output_steps=4)
        second_prediction = self(
            history[:, 4:16], fast_state=fast_state, output_steps=4)
        # The preregistered inner objective identifies velocity dynamics only.
        # Channel 2 remains available to the point-query loss, but must not
        # change the sample-specific adaptation direction.
        first_error = (
            first_prediction[..., :2]
            - target_coordinates[:, 12:16, ..., :2])
        second_error = (
            second_prediction[..., :2]
            - target_coordinates[:, 16:20, ..., :2])
        reduction_dims = tuple(range(1, first_error.ndim))
        return 0.5 * (
            first_error.square().mean(dim=reduction_dims)
            + second_error.square().mean(dim=reduction_dims))

    def adapted_fast_state(
            self,
            history: torch.Tensor,
            *,
            radius: float = 0.10,
            create_graph: bool = False,
            mode: AdaptationMode = "self",
            donor_indices: torch.Tensor | None = None,
            donor_gradient: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        """Build and adapt a fresh state, returning detached diagnostics.

        ``mode`` implements the paired trainer semantics.  ``static`` still
        evaluates and differentiates the support tasks, but applies no step;
        ``self`` applies the aligned gradient; ``donor`` uses either an
        explicit donor gradient or a caller-provided row mapping.
        """
        self._validate_history(history)
        fast_state = self.new_fast_vector(
            history.shape[0],
            device=history.device,
            dtype=history.dtype,
            requires_grad=True,
        )
        losses = self.support_loss(history, fast_state)
        self_gradient = per_sample_fast_gradients(
            losses,
            fast_state,
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        adapted = one_step_fast_adaptation(
            fast_state,
            self_gradient,
            mode=mode,
            radius=radius,
            epsilon=self.config.fast_epsilon,
            donor_indices=donor_indices,
            donor_gradient=donor_gradient,
        )
        diagnostics: dict[str, torch.Tensor | str] = {
            "mode": mode,
            "per_sample_loss": losses.detach(),
            "self_gradient_norm": torch.linalg.vector_norm(
                self_gradient.detach(), dim=1),
            "step_norm": torch.linalg.vector_norm(
                (adapted - fast_state).detach(), dim=1),
        }
        return adapted, diagnostics

    def forward(
            self,
            history: torch.Tensor,
            fast_state: torch.Tensor | None = None,
            output_steps: int | None = None,
            ) -> torch.Tensor:
        return self.forward_components(
            history,
            output_steps=output_steps,
            fast_state=fast_state,
        ).prediction


def build_history_episodic_adaptation(
        **overrides) -> HistoryEpisodicForecaster:
    return HistoryEpisodicForecaster(
        HistoryEpisodicAdaptationConfig(**overrides))


# Descriptive compatibility alias for callers that imported the initial name
# while the mechanism was being developed.  Both names refer to one class and
# therefore have identical state dictionaries.
HistoryEpisodicAdaptationOperator = HistoryEpisodicForecaster
