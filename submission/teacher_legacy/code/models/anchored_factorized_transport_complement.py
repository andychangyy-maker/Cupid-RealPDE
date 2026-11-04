"""Anchored factorized transport complement for the mature B210/XCB model.

The legacy model remains a complete, independently trainable predictor.  A
second decoder is copied tensor-for-tensor from its mature decoder and learns
four disjoint velocity roles after the legacy physical-energy controller:

``flow``
    A smooth semi-Lagrangian displacement on a four-anchor 8x16 grid.
``source``
    A four-anchor, 3x5 additive fluctuation with exactly zero temporal mean.
``temporal_mean``
    One 3x5 u/v field shared by all twenty forecast horizons.
``log_energy``
    One 3x5 joint-u/v log-energy field shared by the complete horizon.

Every tensor from the legacy path is detached before candidate composition,
and the copied decoder receives detached encoder coordinates.  Consequently a
candidate-only loss cannot alter the mature predictor.  The legacy output is
also returned explicitly so a trainer can apply the unchanged base objective
with a separate optimizer.

All four role projections are zero initialized.  Loading a legacy checkpoint
therefore gives bitwise identity for every public output at update zero.  In
train mode the copied decoder replays the primary decoder's dropout state in a
forked RNG scope: both copies see the same masks while the caller observes the
same generator advance as one legacy decoder call.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
from .native_feature_lagrangian_phase import spatial_backtrace_warp
from .unet2d import ResidualConvBlock
from .wake_fluctuation_complementary import (
    _ExactIdentityAddSTE,
    _normalize_tuple_fields,
    WakeFluctuationComplementaryUNet2d,
)


DOWNSTREAM_X_START = 13
DOWNSTREAM_TAPER_COLUMNS = 8
FLOW_TEMPORAL_ANCHORS = 4
FLOW_CONTROL_HEIGHT = 8
FLOW_CONTROL_WIDTH = 16
ROLE_GRID_HEIGHT = 3
ROLE_GRID_WIDTH = 5
ROLE_HIDDEN_CHANNELS = 32
ROLE_GROUPS = 8
MAX_DISPLACEMENT_PIXELS = 1.5
SOURCE_LIMIT = 0.75
MEAN_UPDATE_LIMIT = 0.75
LOG_ENERGY_LIMIT = math.log(2.0)


class _FactorizedPrimaryFeatures(list[torch.Tensor]):
    """Primary feature list carrying its live copied-decoder coordinates."""

    def __init__(
            self, primary: list[torch.Tensor],
            role: list[torch.Tensor]) -> None:
        super().__init__(primary)
        self.role_features = role


@dataclass
class AnchoredFactorizedTransportFields:
    """Named tensors for role, identity and gradient-boundary audits."""

    primary_features: list[torch.Tensor]
    role_features: list[torch.Tensor]
    base_prediction: torch.Tensor
    base_members: torch.Tensor
    raw_flow_control: torch.Tensor
    raw_source_control: torch.Tensor
    raw_temporal_mean_control: torch.Tensor
    raw_log_energy_control: torch.Tensor
    downstream_taper: torch.Tensor
    displacement: torch.Tensor
    source_delta: torch.Tensor
    temporal_mean_delta: torch.Tensor
    log_energy_delta: torch.Tensor
    energy_amplitude: torch.Tensor
    warped_members_uv: torch.Tensor
    transport_source_fluctuation_uv: torch.Tensor
    candidate_members: torch.Tensor
    candidate_prediction: torch.Tensor
    identity_bypass: torch.Tensor


@dataclass
class AnchoredFactorizedTrainingOutput:
    """One-forward base/candidate pair for separately owned objectives."""

    base_outputs: tuple[Any, ...]
    fields: AnchoredFactorizedTransportFields

    @property
    def candidate_prediction(self) -> torch.Tensor:
        return self.fields.candidate_prediction

    @property
    def candidate_members(self) -> torch.Tensor:
        return self.fields.candidate_members

    def public_outputs(self) -> tuple[Any, ...]:
        """Replace point/members and expose native UQ as read-only context.

        Base training owns the live tensors in ``base_outputs``.  The public
        candidate tuple must not provide a second route from a generic
        quantile/SPS loss into the mature model, so every inherited UQ and
        feature tensor is detached here without changing its value.
        """
        return (
            self.candidate_prediction,
            self.candidate_members,
            *(_detach_tree(value) for value in self.base_outputs[2:]),
        )


def _detach_tree(value: Any) -> Any:
    """Detach tensors in the small, stable native-UQ output tree."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _detach_tree(item)) for key, item in value.items())
    return value


class AnchoredFactorizedTransportComplementUNet2d(EnsembleUNet2d):
    """Strict legacy XCB upgrade with a gradient-isolated four-role decoder."""

    ARCHITECTURE = "ensemble_unet_anchored_factorized_transport_complement_v1"
    ROLE_PREFIX = "factorized_role_"
    RESPONSIBILITY = (
        "detached_complete_decoder_flow_zero_mean_low_frequency_source_"
        "temporal_mean_and_joint_uv_log_energy_after_legacy_controller")

    def __init__(self, backbone_config: EnsembleUNet2dConfig) -> None:
        self._validate_backbone_contract(backbone_config)
        super().__init__(backbone_config)

        # deepcopy consumes no random numbers and starts from the exact primary
        # function.  Only decoder operators are copied; mature point/UQ heads
        # remain exclusively owned by the anchored base path.
        self.factorized_role_up = deepcopy(self.up)
        self.factorized_role_decoder = deepcopy(self.decoder)
        self.factorized_role_decoder_regime_film = deepcopy(
            self.decoder_regime_film)
        self.factorized_role_cross_conditioner = deepcopy(
            self.cross_conditioner)

        decoder_channels = tuple(
            backbone_config.base * 2 ** level
            for level in range(backbone_config.depth - 2, -1, -1))
        # New modules cannot perturb the constructor RNG stream seen by a
        # same-seed legacy control.  This matters for strict paired lifecycle
        # tests even though the role heads themselves begin at zero.
        with torch.random.fork_rng(devices=[]):
            self.factorized_role_feature_projections = nn.ModuleList(
                nn.Conv2d(channels, ROLE_HIDDEN_CHANNELS, 1)
                for channels in decoder_channels)
            self.factorized_role_body = nn.Sequential(
                ResidualConvBlock(
                    ROLE_HIDDEN_CHANNELS, ROLE_HIDDEN_CHANNELS, ROLE_GROUPS),
                ResidualConvBlock(
                    ROLE_HIDDEN_CHANNELS, ROLE_HIDDEN_CHANNELS, ROLE_GROUPS),
            )
            self.factorized_role_flow_head = nn.Conv2d(
                ROLE_HIDDEN_CHANNELS, FLOW_TEMPORAL_ANCHORS * 2, 1)
            self.factorized_role_source_head = nn.Conv2d(
                ROLE_HIDDEN_CHANNELS, FLOW_TEMPORAL_ANCHORS * 2, 1)
            self.factorized_role_temporal_mean_head = nn.Conv2d(
                ROLE_HIDDEN_CHANNELS, 2, 1)
            self.factorized_role_log_energy_head = nn.Conv2d(
                ROLE_HIDDEN_CHANNELS, 1, 1)
        self._zero_role_heads_()

    @staticmethod
    def _validate_backbone_contract(config: EnsembleUNet2dConfig) -> None:
        WakeFluctuationComplementaryUNet2d._validate_backbone_contract(config)
        if config.out_steps != 20:
            raise ValueError("factorized transport requires twenty outputs")
        if (not config.physical_energy_controller
                or not config.physical_energy_member_sync):
            raise ValueError(
                "factorized transport requires member-synchronous legacy "
                "physical energy control")
        if config.shared_member_uv_mean:
            raise ValueError(
                "factorized transport requires native member means")
        if config.decoupled_member_residual_channels:
            raise ValueError(
                "factorized transport excludes post-controller member heads")

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": self.ARCHITECTURE,
            "backbone": EnsembleUNet2d.architecture_config(self),
            "responsibility": self.RESPONSIBILITY,
            "support_axis": "width",
            "downstream_x_start": DOWNSTREAM_X_START,
            "taper_columns": DOWNSTREAM_TAPER_COLUMNS,
            "flow_control": [
                FLOW_TEMPORAL_ANCHORS,
                FLOW_CONTROL_HEIGHT,
                FLOW_CONTROL_WIDTH,
                2,
            ],
            "source_control": [
                FLOW_TEMPORAL_ANCHORS,
                ROLE_GRID_HEIGHT,
                ROLE_GRID_WIDTH,
                2,
            ],
            "temporal_mean_control": [
                1, ROLE_GRID_HEIGHT, ROLE_GRID_WIDTH, 2],
            "log_energy_control": [
                1, ROLE_GRID_HEIGHT, ROLE_GRID_WIDTH, 1],
            "role_hidden_channels": ROLE_HIDDEN_CHANNELS,
            "role_groups": ROLE_GROUPS,
            "max_displacement_pixels": MAX_DISPLACEMENT_PIXELS,
            "source_limit": SOURCE_LIMIT,
            "mean_update_limit": MEAN_UPDATE_LIMIT,
            "log_energy_limit": LOG_ENERGY_LIMIT,
            "controller_order": "legacy_controller_then_factorized_roles",
            "pressure_policy": "base_bitwise_passthrough",
            "base_context_gradient": "detached",
            "native_uq_policy": "detached_anchored_base_tuple",
            "update_zero": "all_public_outputs_bitwise_legacy_identity",
        }

    def role_contract(self) -> dict[str, object]:
        return {
            "flow": {
                "control_shape": [
                    FLOW_TEMPORAL_ANCHORS, FLOW_CONTROL_HEIGHT,
                    FLOW_CONTROL_WIDTH, 2],
                "bound_pixels": MAX_DISPLACEMENT_PIXELS,
                "owns": "semi_lagrangian_phase",
            },
            "source": {
                "control_shape": [
                    FLOW_TEMPORAL_ANCHORS, ROLE_GRID_HEIGHT,
                    ROLE_GRID_WIDTH, 2],
                "pre_center_bound_normalized_units": SOURCE_LIMIT,
                "post_center_absolute_bound_normalized_units": (
                    2.0 * SOURCE_LIMIT),
                "invariant": "exact_zero_full20_temporal_mean",
            },
            "temporal_mean": {
                "control_shape": [1, ROLE_GRID_HEIGHT, ROLE_GRID_WIDTH, 2],
                "bound_normalized_units": MEAN_UPDATE_LIMIT,
                "owns": "member_temporal_mean_shift",
            },
            "log_energy": {
                "control_shape": [1, ROLE_GRID_HEIGHT, ROLE_GRID_WIDTH, 1],
                "bound": LOG_ENERGY_LIMIT,
                "joint_uv": True,
                "amplitude": "exp(0.5*delta_log_energy)",
            },
            "gradient_boundary": {
                "base_members": "detached_before_composition",
                "encoder_skips_regime_history_to_copy": "detached",
                "candidate_loss_to_base": False,
            },
            "pressure_passthrough": "base_bitwise",
            "native_uq": "detached_base_prediction_and_base_members",
            "update_zero": "all_public_outputs_bitwise_legacy_identity",
        }

    @classmethod
    def from_architecture_config(
            cls, values: Mapping[str, object],
            ) -> "AnchoredFactorizedTransportComplementUNet2d":
        stored = dict(values)
        fixed = {
            "architecture": cls.ARCHITECTURE,
            "responsibility": cls.RESPONSIBILITY,
            "support_axis": "width",
            "downstream_x_start": DOWNSTREAM_X_START,
            "taper_columns": DOWNSTREAM_TAPER_COLUMNS,
            "flow_control": [
                FLOW_TEMPORAL_ANCHORS, FLOW_CONTROL_HEIGHT,
                FLOW_CONTROL_WIDTH, 2],
            "source_control": [
                FLOW_TEMPORAL_ANCHORS, ROLE_GRID_HEIGHT, ROLE_GRID_WIDTH, 2],
            "temporal_mean_control": [
                1, ROLE_GRID_HEIGHT, ROLE_GRID_WIDTH, 2],
            "log_energy_control": [
                1, ROLE_GRID_HEIGHT, ROLE_GRID_WIDTH, 1],
            "role_hidden_channels": ROLE_HIDDEN_CHANNELS,
            "role_groups": ROLE_GROUPS,
            "max_displacement_pixels": MAX_DISPLACEMENT_PIXELS,
            "source_limit": SOURCE_LIMIT,
            "mean_update_limit": MEAN_UPDATE_LIMIT,
            "log_energy_limit": LOG_ENERGY_LIMIT,
            "controller_order": "legacy_controller_then_factorized_roles",
            "pressure_policy": "base_bitwise_passthrough",
            "base_context_gradient": "detached",
            "native_uq_policy": "detached_anchored_base_tuple",
            "update_zero": "all_public_outputs_bitwise_legacy_identity",
        }
        for key, expected_value in fixed.items():
            if stored.pop(key, None) != expected_value:
                raise ValueError(f"factorized transport {key} changed")
        raw_backbone = stored.pop("backbone", None)
        if stored or not isinstance(raw_backbone, Mapping):
            raise ValueError("factorized transport config is incomplete")
        backbone = dict(raw_backbone)
        if backbone.pop("architecture", None) != "ensemble_unet":
            raise ValueError("factorized transport backbone changed")
        return cls(EnsembleUNet2dConfig(
            **_normalize_tuple_fields(backbone)))

    def backbone_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return exactly the legacy inventory accepted by strict upgrade."""
        return OrderedDict(
            (name, value) for name, value in self.state_dict().items()
            if not name.startswith(self.ROLE_PREFIX))

    def base_named_parameters(
            self) -> Iterator[tuple[str, nn.Parameter]]:
        return (
            (name, parameter) for name, parameter in self.named_parameters()
            if not name.startswith(self.ROLE_PREFIX))

    def role_named_parameters(
            self) -> Iterator[tuple[str, nn.Parameter]]:
        return (
            (name, parameter) for name, parameter in self.named_parameters()
            if name.startswith(self.ROLE_PREFIX))

    def base_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for _name, parameter in self.base_named_parameters())

    def role_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for _name, parameter in self.role_named_parameters())

    @torch.no_grad()
    def _zero_role_heads_(self) -> None:
        for head in (
                self.factorized_role_flow_head,
                self.factorized_role_source_head,
                self.factorized_role_temporal_mean_head,
                self.factorized_role_log_energy_head):
            head.weight.zero_()
            head.bias.zero_()

    @staticmethod
    def _copy_optional_module(
            primary: nn.Module | None, role: nn.Module | None,
            label: str) -> None:
        if (primary is None) != (role is None):
            raise RuntimeError(f"factorized role {label} topology differs")
        if primary is not None and role is not None:
            role.load_state_dict(primary.state_dict(), strict=True)

    @torch.no_grad()
    def copy_primary_into_factorized_role_(self) -> None:
        self.factorized_role_up.load_state_dict(
            self.up.state_dict(), strict=True)
        self.factorized_role_decoder.load_state_dict(
            self.decoder.state_dict(), strict=True)
        self._copy_optional_module(
            self.decoder_regime_film,
            self.factorized_role_decoder_regime_film,
            "decoder regime FiLM")
        self._copy_optional_module(
            self.cross_conditioner,
            self.factorized_role_cross_conditioner,
            "cross-conditioner")

    def role_decoder_copy_report(self) -> dict[str, object]:
        pairs = (
            ("up", self.up, self.factorized_role_up),
            ("decoder", self.decoder, self.factorized_role_decoder),
            ("decoder_regime_film", self.decoder_regime_film,
             self.factorized_role_decoder_regime_film),
            ("cross_conditioner", self.cross_conditioner,
             self.factorized_role_cross_conditioner),
        )
        reports: dict[str, object] = {}
        all_equal = True
        for label, primary, role in pairs:
            if primary is None or role is None:
                equal = primary is None and role is None
                reports[label] = {
                    "present": False, "tensor_count": 0,
                    "all_equal": equal, "mismatched": []}
                all_equal = all_equal and equal
                continue
            first = primary.state_dict()
            second = role.state_dict()
            mismatched = sorted(
                set(first).symmetric_difference(second)
                | {name for name in set(first).intersection(second)
                   if not torch.equal(first[name], second[name])})
            equal = not mismatched
            reports[label] = {
                "present": True, "tensor_count": len(first),
                "all_equal": equal, "mismatched": mismatched}
            all_equal = all_equal and equal
        return {"all_equal": all_equal, "pairs": reports}

    def load_legacy_state_dict(
            self, state: Mapping[str, torch.Tensor]) -> dict[str, object]:
        """Strictly load legacy XCB, copy decoder and reset identity heads."""
        expected_legacy = set(self.backbone_state_dict())
        supplied = set(state)
        if supplied != expected_legacy:
            missing = sorted(expected_legacy - supplied)
            unexpected = sorted(supplied - expected_legacy)
            raise RuntimeError(
                "legacy XCB state inventory changed: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}")
        incompatible = self.load_state_dict(state, strict=False)
        expected_new = set(self.state_dict()) - expected_legacy
        if (set(incompatible.missing_keys) != expected_new
                or incompatible.unexpected_keys
                or any(not name.startswith(self.ROLE_PREFIX)
                       for name in expected_new)):
            raise RuntimeError(
                "legacy XCB state is incompatible with factorized upgrade")
        self.copy_primary_into_factorized_role_()
        self._zero_role_heads_()
        report = self.role_decoder_copy_report()
        if not report["all_equal"]:
            raise RuntimeError("factorized copied decoder verification failed")
        return {
            "strict_legacy_upgrade": True,
            "legacy_keys": len(expected_legacy),
            "new_role_keys": len(expected_new),
            "decoder_copy": report,
            "update0_public_identity_expected": True,
        }

    def _decode_factorized_features(
            self, x: torch.Tensor,
            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if x.shape[3] <= DOWNSTREAM_X_START:
            raise ValueError(
                "factorized transport input does not reach downstream x=13")
        encoded, skips, regime, dynamics_state = self._encode_history(x)
        cpu_state = torch.get_rng_state() if self.training else None
        cuda_devices: list[int] = []
        cuda_states: dict[int, torch.Tensor] = {}
        if self.training and encoded.device.type == "cuda":
            index = encoded.device.index
            if index is None:
                index = torch.cuda.current_device()
            cuda_devices.append(index)
            cuda_states[index] = torch.cuda.get_rng_state(index)

        primary = self._decode_from_encoded(
            encoded, skips, regime, dynamics_state, x,
            self.up, self.decoder, self.decoder_regime_film,
            self.cross_conditioner)
        if self.training:
            # fork_rng snapshots the externally visible post-primary state.
            # Replaying the pre-primary state gives the copy identical masks,
            # and leaving the context restores the one-call legacy stream.
            with torch.random.fork_rng(devices=cuda_devices):
                if cpu_state is None:  # pragma: no cover - typing invariant
                    raise RuntimeError("factorized decoder RNG state is missing")
                torch.set_rng_state(cpu_state)
                for index, state in cuda_states.items():
                    torch.cuda.set_rng_state(state, device=index)
                role = self._decode_from_encoded(
                    encoded.detach(),
                    [value.detach() for value in skips],
                    None if regime is None else regime.detach(),
                    (None if dynamics_state is None
                     else dynamics_state.detach()),
                    x.detach(),
                    self.factorized_role_up,
                    self.factorized_role_decoder,
                    self.factorized_role_decoder_regime_film,
                    self.factorized_role_cross_conditioner)
        else:
            role = self._decode_from_encoded(
                encoded.detach(),
                [value.detach() for value in skips],
                None if regime is None else regime.detach(),
                None if dynamics_state is None else dynamics_state.detach(),
                x.detach(),
                self.factorized_role_up,
                self.factorized_role_decoder,
                self.factorized_role_decoder_regime_film,
                self.factorized_role_cross_conditioner)
        if len(primary) != len(role):
            raise RuntimeError("factorized decoder scale contract changed")
        return primary, role

    def _decode_role_features(
            self, x: torch.Tensor,
            ) -> tuple[list[torch.Tensor], None, None, None]:
        primary, role = self._decode_factorized_features(x)
        return _FactorizedPrimaryFeatures(primary, role), None, None, None

    @staticmethod
    def _downstream_taper(value: torch.Tensor) -> torch.Tensor:
        width = value.shape[-2]
        if width <= DOWNSTREAM_X_START:
            raise ValueError(
                "factorized transport input does not reach downstream x=13")
        coordinate = torch.arange(
            width, device=value.device, dtype=value.dtype)
        phase = (
            coordinate - float(DOWNSTREAM_X_START - 1)
        ) / float(DOWNSTREAM_TAPER_COLUMNS)
        phase = phase.clamp(0.0, 1.0)
        smoothstep = phase.square() * (3.0 - 2.0 * phase)
        return smoothstep.view(1, 1, 1, width, 1)

    def _role_hidden(
            self, role_features: list[torch.Tensor]) -> torch.Tensor:
        if (len(role_features)
                != len(self.factorized_role_feature_projections)):
            raise ValueError("factorized role feature pyramid depth changed")
        size = (FLOW_CONTROL_HEIGHT, FLOW_CONTROL_WIDTH)
        projected = []
        for projection, feature in zip(
                self.factorized_role_feature_projections, role_features):
            if feature.ndim != 4:
                raise ValueError("factorized role features must be BCHW")
            projected.append(F.adaptive_avg_pool2d(projection(feature), size))
        return self.factorized_role_body(F.silu(torch.stack(projected).sum(0)))

    @staticmethod
    def _reshape_anchor_control(
            raw: torch.Tensor, anchors: int,
            channels: int) -> torch.Tensor:
        batch, packed, height, width = raw.shape
        if packed != anchors * channels:
            raise ValueError("factorized temporal control channels changed")
        return raw.reshape(
            batch, anchors, channels, height, width
        ).permute(0, 1, 3, 4, 2)

    @staticmethod
    def _expand_anchor_control(
            control: torch.Tensor, out_steps: int,
            output_size: tuple[int, int]) -> torch.Tensor:
        if control.ndim != 5 or control.shape[1] < 2:
            raise ValueError("factorized anchor control must be BAHWC, A>=2")
        batch, anchors, height, width, channels = control.shape
        temporal = F.interpolate(
            control.permute(0, 2, 3, 4, 1).reshape(
                batch * height * width, channels, anchors),
            size=out_steps, mode="linear", align_corners=True,
        ).reshape(
            batch, height, width, channels, out_steps,
        ).permute(0, 4, 1, 2, 3)
        spatial = F.interpolate(
            temporal.permute(0, 1, 4, 2, 3).reshape(
                batch * out_steps, channels, height, width),
            size=output_size, mode="bilinear", align_corners=False)
        return spatial.reshape(
            batch, out_steps, channels, *output_size
        ).permute(0, 1, 3, 4, 2)

    @staticmethod
    def _expand_static_control(
            control: torch.Tensor,
            output_size: tuple[int, int]) -> torch.Tensor:
        if control.ndim != 5 or control.shape[1] != 1:
            raise ValueError("factorized static control must be B1HWC")
        batch, _one, height, width, channels = control.shape
        expanded = F.interpolate(
            control[:, 0].permute(0, 3, 1, 2),
            size=output_size, mode="bilinear", align_corners=False)
        return expanded.permute(0, 2, 3, 1).unsqueeze(1)

    @staticmethod
    def _warp_members_uv(
            member_uv: torch.Tensor,
            displacement: torch.Tensor) -> torch.Tensor:
        if (member_uv.ndim != 6 or member_uv.shape[-1] != 2
                or displacement.shape != (
                    member_uv.shape[0], member_uv.shape[2],
                    member_uv.shape[3], member_uv.shape[4], 2)):
            raise ValueError("factorized member/flow shapes changed")
        batch, members, steps, height, width, channels = member_uv.shape
        source = member_uv.reshape(
            batch * members, steps, height, width, channels)
        flow = displacement[:, None].expand(
            batch, members, steps, height, width, 2).reshape_as(source)
        warped = spatial_backtrace_warp(source, flow).to(member_uv)
        identity = spatial_backtrace_warp(
            source, torch.zeros_like(flow)).to(member_uv)
        delta = (warped - identity).reshape_as(member_uv)
        bypass = displacement.eq(0.0).all(
            dim=-1, keepdim=True)[:, None]
        return _ExactIdentityAddSTE.apply(member_uv, delta, bypass)

    def _factorized_fields_from_base(
            self, x: torch.Tensor,
            primary_features: list[torch.Tensor],
            role_features: list[torch.Tensor],
            base_prediction: torch.Tensor,
            base_members: torch.Tensor,
            ) -> AnchoredFactorizedTransportFields:
        if (base_members.ndim != 6 or base_members.shape[-1] < 3
                or base_prediction.shape != (
                    base_members.shape[0], base_members.shape[2],
                    base_members.shape[3], base_members.shape[4],
                    base_members.shape[5])):
            raise ValueError("factorized base point/member contract changed")
        batch, _members, steps, height, width, _channels = base_members.shape
        if steps != self.config.out_steps:
            raise ValueError("factorized base horizon changed")

        hidden = self._role_hidden(role_features)
        role_grid = F.adaptive_avg_pool2d(
            hidden, (ROLE_GRID_HEIGHT, ROLE_GRID_WIDTH))
        raw_flow = self._reshape_anchor_control(
            self.factorized_role_flow_head(hidden),
            FLOW_TEMPORAL_ANCHORS, 2)
        raw_source = self._reshape_anchor_control(
            self.factorized_role_source_head(role_grid),
            FLOW_TEMPORAL_ANCHORS, 2)
        raw_mean = self._reshape_anchor_control(
            self.factorized_role_temporal_mean_head(role_grid), 1, 2)
        raw_log_energy = self._reshape_anchor_control(
            self.factorized_role_log_energy_head(role_grid), 1, 1)

        output_size = (height, width)
        flow_full = self._expand_anchor_control(
            raw_flow, steps, output_size)
        source_full = self._expand_anchor_control(
            raw_source, steps, output_size)
        mean_full = self._expand_static_control(raw_mean, output_size)
        log_energy_full = self._expand_static_control(
            raw_log_energy, output_size)
        taper = self._downstream_taper(flow_full)
        displacement = (
            MAX_DISPLACEMENT_PIXELS * torch.tanh(flow_full) * taper)
        source_delta = SOURCE_LIMIT * torch.tanh(source_full) * taper
        source_delta = source_delta.float()
        source_delta = (
            source_delta - source_delta.mean(dim=1, keepdim=True)
        ).to(base_members)
        temporal_mean_delta = (
            MEAN_UPDATE_LIMIT * torch.tanh(mean_full) * taper).to(base_members)
        log_energy_delta = (
            LOG_ENERGY_LIMIT * torch.tanh(log_energy_full) * taper)
        energy_amplitude = torch.exp(0.5 * log_energy_delta.float())

        # This is the hard gradient boundary of the experiment.  A role loss
        # sees the live copied decoder and role heads, but never any mature
        # member, point, decoder, controller or encoder tensor.
        base_uv = base_members[..., :2].detach()
        warped_uv = self._warp_members_uv(base_uv, displacement)
        transported_source = warped_uv.float() + source_delta[:, None].float()
        fluctuation = (
            transported_source
            - transported_source.mean(dim=2, keepdim=True))
        base_member_mean = base_uv.float().mean(dim=2, keepdim=True)
        candidate_proposal = (
            base_member_mean
            + temporal_mean_delta[:, None].float()
            + energy_amplitude[:, None] * fluctuation).to(base_uv)

        # Select the legacy value at every inactive spatial pixel, not merely
        # when the whole sample is inactive.  Besides update-zero identity,
        # this makes the fixed upstream support bitwise immutable after the
        # role heads learn nonzero downstream fields.
        controls_are_zero = (
            displacement.eq(0.0).all(dim=(1, 4), keepdim=True)
            & source_delta.eq(0.0).all(dim=(1, 4), keepdim=True)
            & temporal_mean_delta.eq(0.0).all(dim=(1, 4), keepdim=True)
            & log_energy_delta.eq(0.0).all(dim=(1, 4), keepdim=True)
        )
        bypass = controls_are_zero[:, None]
        candidate_uv = _ExactIdentityAddSTE.apply(
            base_uv, candidate_proposal - base_uv, bypass)
        candidate_members = torch.cat((
            candidate_uv, base_members[..., 2:].detach(),
        ), dim=-1)
        candidate_prediction = candidate_members.mean(dim=1)
        return AnchoredFactorizedTransportFields(
            primary_features=primary_features,
            role_features=role_features,
            base_prediction=base_prediction,
            base_members=base_members,
            raw_flow_control=raw_flow,
            raw_source_control=raw_source,
            raw_temporal_mean_control=raw_mean,
            raw_log_energy_control=raw_log_energy,
            downstream_taper=taper,
            displacement=displacement,
            source_delta=source_delta,
            temporal_mean_delta=temporal_mean_delta,
            log_energy_delta=log_energy_delta,
            energy_amplitude=energy_amplitude,
            warped_members_uv=warped_uv,
            transport_source_fluctuation_uv=fluctuation,
            candidate_members=candidate_members,
            candidate_prediction=candidate_prediction,
            identity_bypass=bypass,
        )

    def forward_factorized_fields(
            self, x: torch.Tensor) -> AnchoredFactorizedTransportFields:
        """Efficient point-only forward exposing the deployed decomposition."""
        tagged, _fluctuation, _uq_role, _mean = self._decode_role_features(x)
        if not isinstance(tagged, _FactorizedPrimaryFeatures):
            raise RuntimeError("factorized feature carrier is missing")
        primary = list(tagged)
        base_members = EnsembleUNet2d._members_from_features(
            self, x, primary)
        base_members = EnsembleUNet2d._energy_controlled_members(
            self, base_members, primary, x)
        base_prediction = base_members.mean(dim=1)
        return self._factorized_fields_from_base(
            x, primary, tagged.role_features,
            base_prediction, base_members)

    def forward_factorized_training(
            self, x: torch.Tensor, *, return_mixture: bool = False,
            mixture_probability_shrinkage: float | None = None,
            apply_mixture_interval: bool = True,
            ) -> AnchoredFactorizedTrainingOutput:
        """Return live legacy outputs and detached-base candidate together."""
        base = EnsembleUNet2d.forward_uq_training(
            self, x, return_mixture=return_mixture,
            mixture_probability_shrinkage=mixture_probability_shrinkage,
            apply_mixture_interval=apply_mixture_interval)
        tagged = base[5]
        if not isinstance(tagged, _FactorizedPrimaryFeatures):
            raise RuntimeError("factorized feature carrier is missing")
        primary = list(tagged)
        base_outputs = (*base[:5], primary, *base[6:])
        fields = self._factorized_fields_from_base(
            x, primary, tagged.role_features,
            base[0], base[1])
        return AnchoredFactorizedTrainingOutput(
            base_outputs=base_outputs, fields=fields)

    def forward_uq_training(
            self, x: torch.Tensor, *, return_mixture: bool = False,
            mixture_probability_shrinkage: float | None = None,
            apply_mixture_interval: bool = True,
            ) -> tuple[Any, ...]:
        paired = self.forward_factorized_training(
            x, return_mixture=return_mixture,
            mixture_probability_shrinkage=mixture_probability_shrinkage,
            apply_mixture_interval=apply_mixture_interval)
        return paired.public_outputs()

    def forward_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fields = self.forward_factorized_fields(x)
        return fields.candidate_prediction, fields.primary_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_factorized_fields(x).candidate_prediction


def build_anchored_factorized_transport_complement(
        backbone_config: EnsembleUNet2dConfig,
        ) -> AnchoredFactorizedTransportComplementUNet2d:
    return AnchoredFactorizedTransportComplementUNet2d(backbone_config)
