"""Wake-local complementary decoder integrated before native energy control.

The released XCB model already has a mature encoder, decoder, ensemble point
readouts and UQ stack.  This module upgrades that exact state with a second
complete decoder.  Both decoders share the live encoder; the second decoder is
initialized by copying the primary decoder tensor-for-tensor after loading the
legacy checkpoint.

The complementary path is deliberately narrow in *responsibility*, not in
capacity.  It may change only the zero-time-mean joint-u/v fluctuation inside
the fixed wake strip ``x in [13, 38)``.  Its proposal is projected back onto
the primary point prediction's per-pixel physical u/v energy shell.  The same
resulting correction is then added to every primary ensemble member before the
existing physical-energy controller and native UQ stack execute.  Pressure is
always taken from the primary members by identity.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import torch

from .ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig


WAKE_X_START = 13
WAKE_X_STOP = 38


class _WakeDynamicsFeatures(list[torch.Tensor]):
    """Dynamics list carrying its paired primary controller coordinates."""

    def __init__(
            self, dynamics: list[torch.Tensor],
            primary: list[torch.Tensor]) -> None:
        super().__init__(dynamics)
        self.primary_features = primary


class _ExactIdentityAddSTE(torch.autograd.Function):
    """Return the legacy value exactly on a zero proposal, but train it.

    Reconstructing ``mean + fluctuation`` is not guaranteed bitwise equal to
    the original floating-point value.  The explicit identity branch keeps a
    copied decoder an exact evaluation-time upgrade while its backward pass
    still reaches both decoder paths from the first optimization step.
    """

    @staticmethod
    def forward(
            ctx: Any, legacy: torch.Tensor, delta: torch.Tensor,
            bypass: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.where(bypass, legacy, legacy + delta)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor):
        del ctx
        return gradient, gradient, None


@dataclass
class WakeFluctuationComplementaryFields:
    """Read-only decomposition for training/evaluation audits.

    ``wake_shell_*`` are the values immediately before the inherited physical
    energy controller.  ``controller_*`` are immediately after that
    controller.  These tensors are observations of the deployed graph, not an
    alternate prediction route.
    """

    primary_features: list[torch.Tensor]
    dynamics_features: list[torch.Tensor]
    primary_members_pre_shell: torch.Tensor
    dynamics_members_pre_shell: torch.Tensor
    primary_point_pre_shell: torch.Tensor
    dynamics_point_pre_shell: torch.Tensor
    centered_wake_delta_uv: torch.Tensor
    shell_correction_uv: torch.Tensor
    wake_shell_members_pre_controller: torch.Tensor
    wake_shell_point_pre_controller: torch.Tensor
    controller_members_post: torch.Tensor
    controller_point_post: torch.Tensor


def _normalize_tuple_fields(values: Mapping[str, object]) -> dict[str, object]:
    """Restore tuple-valued dataclass fields after a JSON round trip."""
    result = dict(values)
    defaults = EnsembleUNet2dConfig()
    for field in fields(EnsembleUNet2dConfig):
        if field.name in result and isinstance(getattr(defaults, field.name), tuple):
            result[field.name] = tuple(result[field.name])  # type: ignore[arg-type]
    return result


class WakeFluctuationComplementaryUNet2d(EnsembleUNet2d):
    """XCB-compatible dual decoder with a wake-only fluctuation role."""

    ARCHITECTURE = "ensemble_unet_wake_fluctuation_complementary_v1"
    DYNAMICS_PREFIX = "wake_dynamics_"
    RESPONSIBILITY = (
        "shared_encoder_complete_decoder_joint_uv_zero_time_mean_"
        "wake_energy_shell_before_native_energy_controller")

    def __init__(self, backbone_config: EnsembleUNet2dConfig) -> None:
        self._validate_backbone_contract(backbone_config)
        super().__init__(backbone_config)

        # Deep copies consume no random numbers.  They make construction an
        # identity when starting from scratch, while load_legacy_state_dict()
        # repeats the copy after the mature legacy tensors have been loaded.
        self.wake_dynamics_up = deepcopy(self.up)
        self.wake_dynamics_decoder = deepcopy(self.decoder)
        self.wake_dynamics_decoder_regime_film = deepcopy(
            self.decoder_regime_film)
        self.wake_dynamics_cross_conditioner = deepcopy(
            self.cross_conditioner)
        self.wake_dynamics_mean_profile_refiner = deepcopy(
            self.mean_profile_refiner)

    @staticmethod
    def _validate_backbone_contract(config: EnsembleUNet2dConfig) -> None:
        config.validate()
        if config.dual_decoder_mode != "none":
            raise ValueError(
                "wake complementary upgrade requires one legacy decoder")
        if not config.mean_fluctuation:
            raise ValueError(
                "wake complementary decoder requires mean_fluctuation=True")
        if config.out_channels < 3:
            raise ValueError(
                "wake complementary decoder requires u/v/pressure outputs")
        if config.deformable_history_memory_mode != "none":
            raise ValueError(
                "wake complementary XCB contract excludes deformable memory")
        if config.mean_preserving_member_adapters:
            raise ValueError(
                "wake complementary XCB contract excludes recursive "
                "mean-preserving adapters")
        if (config.history_conditioned_operator_basis
                or config.late_multiscale_sidecar_mode != "none"):
            raise ValueError(
                "wake complementary XCB contract requires native decoder "
                "features")
        # The audited fields and the normal forward must name the same three
        # stages.  The exact XCB topology has none of these later point
        # transforms; reject silently different graphs rather than mislabel an
        # audit tensor.
        if (config.hard_region_expert_channels
                or config.conservative_closure_experts
                or config.recurrent_latent_rollout_channels
                or config.scalar_potential_residual_mode != "none"
                or config.temporal_point_decoder
                or config.latent_ode_point_decoder
                or config.latent_ude_band_decoder
                or config.latent_ude_coarse_residual
                or config.latent_ude_exclusive_coarse_fluctuation
                or config.isoenergetic_temporal_transport
                or config.temporal_band_mixer
                or config.spatial_transport_warp
                or config.phase_delta_hidden_channels):
            raise ValueError(
                "wake complementary XCB contract excludes other point "
                "transforms")

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": self.ARCHITECTURE,
            "backbone": EnsembleUNet2d.architecture_config(self),
            "wake_axis": "width",
            "wake_x_half_open": [WAKE_X_START, WAKE_X_STOP],
            "responsibility": self.RESPONSIBILITY,
            "pressure_policy": "primary_bitwise_passthrough",
            "member_policy": "same_shared_uv_correction_for_every_member",
        }

    @classmethod
    def from_architecture_config(
            cls, values: Mapping[str, object],
            ) -> "WakeFluctuationComplementaryUNet2d":
        stored = dict(values)
        if stored.pop("architecture", None) != cls.ARCHITECTURE:
            raise ValueError("wake complementary architecture changed")
        if stored.pop("wake_axis", None) != "width":
            raise ValueError("wake complementary axis changed")
        if tuple(stored.pop("wake_x_half_open", ())) != (
                WAKE_X_START, WAKE_X_STOP):
            raise ValueError("wake complementary strip changed")
        if stored.pop("responsibility", None) != cls.RESPONSIBILITY:
            raise ValueError("wake complementary responsibility changed")
        if stored.pop("pressure_policy", None) != (
                "primary_bitwise_passthrough"):
            raise ValueError("wake complementary pressure policy changed")
        if stored.pop("member_policy", None) != (
                "same_shared_uv_correction_for_every_member"):
            raise ValueError("wake complementary member policy changed")
        raw_backbone = stored.pop("backbone", None)
        if stored or not isinstance(raw_backbone, Mapping):
            raise ValueError("wake complementary config is incomplete")
        backbone = dict(raw_backbone)
        if backbone.pop("architecture", None) != "ensemble_unet":
            raise ValueError("wake complementary backbone is not ensemble_unet")
        return cls(EnsembleUNet2dConfig(
            **_normalize_tuple_fields(backbone)))

    def backbone_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return the exact state inventory accepted by legacy XCB."""
        return OrderedDict(
            (name, value) for name, value in self.state_dict().items()
            if not name.startswith(self.DYNAMICS_PREFIX))

    @torch.no_grad()
    def copy_primary_into_wake_dynamics_(self) -> None:
        """Copy every complete decoder tensor from the primary path."""
        self.wake_dynamics_up.load_state_dict(
            self.up.state_dict(), strict=True)
        self.wake_dynamics_decoder.load_state_dict(
            self.decoder.state_dict(), strict=True)
        self._copy_optional_module(
            self.decoder_regime_film,
            self.wake_dynamics_decoder_regime_film,
            "decoder regime FiLM")
        self._copy_optional_module(
            self.cross_conditioner,
            self.wake_dynamics_cross_conditioner,
            "cross-conditioner")
        self._copy_optional_module(
            self.mean_profile_refiner,
            self.wake_dynamics_mean_profile_refiner,
            "mean-profile refiner")

    @staticmethod
    def _copy_optional_module(
            primary: torch.nn.Module | None,
            dynamics: torch.nn.Module | None,
            label: str) -> None:
        if (primary is None) != (dynamics is None):
            raise RuntimeError(f"wake {label} topology differs")
        if primary is not None and dynamics is not None:
            dynamics.load_state_dict(primary.state_dict(), strict=True)

    def wake_decoder_copy_report(self) -> dict[str, object]:
        """Report tensorwise equality of every duplicated decoder module."""
        pairs = (
            ("up", self.up, self.wake_dynamics_up),
            ("decoder", self.decoder, self.wake_dynamics_decoder),
            ("decoder_regime_film", self.decoder_regime_film,
             self.wake_dynamics_decoder_regime_film),
            ("cross_conditioner", self.cross_conditioner,
             self.wake_dynamics_cross_conditioner),
            ("mean_profile_refiner", self.mean_profile_refiner,
             self.wake_dynamics_mean_profile_refiner),
        )
        pair_reports: dict[str, object] = {}
        all_equal = True
        for label, primary, dynamics in pairs:
            if primary is None or dynamics is None:
                equal = primary is None and dynamics is None
                pair_reports[label] = {
                    "present": False, "tensor_count": 0,
                    "all_equal": equal, "mismatched": []}
                all_equal = all_equal and equal
                continue
            first = primary.state_dict()
            second = dynamics.state_dict()
            mismatched = sorted(
                set(first).symmetric_difference(second)
                | {name for name in set(first).intersection(second)
                   if not torch.equal(first[name], second[name])})
            equal = not mismatched
            pair_reports[label] = {
                "present": True, "tensor_count": len(first),
                "all_equal": equal, "mismatched": mismatched}
            all_equal = all_equal and equal
        return {"all_equal": all_equal, "pairs": pair_reports}

    def load_legacy_state_dict(
            self, state: Mapping[str, torch.Tensor]) -> dict[str, object]:
        """Strictly upgrade one legacy EnsembleUNet2d/XCB state.

        Strictness is checked against the complete legacy inventory first.
        Only the duplicated decoder keys may be absent from the candidate
        state load; after loading, those tensors are copied from the mature
        primary decoder and verified tensorwise.
        """
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
                or incompatible.unexpected_keys):
            raise RuntimeError(
                "legacy XCB state is incompatible with wake complementary "
                "upgrade")
        self.copy_primary_into_wake_dynamics_()
        copy_report = self.wake_decoder_copy_report()
        if not copy_report["all_equal"]:
            raise RuntimeError("wake complementary decoder copy failed")
        return {
            "strict_legacy_upgrade": True,
            "legacy_keys": len(expected_legacy),
            "new_decoder_keys": len(expected_new),
            "decoder_copy": copy_report,
            "update0_eval_identity_expected": True,
        }

    def _decode_wake_features(
            self, x: torch.Tensor,
            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if x.shape[3] < WAKE_X_STOP:
            raise ValueError(
                f"wake complementary input width must be >= {WAKE_X_STOP}")
        encoded, skips, regime, dynamics_state = self._encode_history(x)
        return self._decode_paired_from_encoded(
            x, encoded, skips, regime, dynamics_state)

    def _decode_paired_from_encoded(
            self, x: torch.Tensor, encoded: torch.Tensor,
            skips: list[torch.Tensor], regime: torch.Tensor | None,
            dynamics_state: torch.Tensor | None,
            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Decode both copies with one decoder call's RNG consumption.

        Residual decoder dropout is part of the training operator.  Letting
        the two otherwise identical copies draw consecutive masks creates a
        nonzero complementary proposal before either copy has learned a
        different function.  The primary path therefore advances the global
        generator normally; the dynamics path replays the state from just
        before that call inside a fork which restores the post-primary state
        on exit.  Consequently copied paths see bitwise-identical stochastic
        masks while the caller observes exactly one decoder call's RNG
        advance.  Evaluation avoids the state bookkeeping altogether.
        """
        cpu_state = torch.get_rng_state() if self.training else None
        cuda_devices: list[int] = []
        cuda_states: dict[int, torch.Tensor] = {}
        if self.training and encoded.device.type == "cuda":
            device_index = encoded.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_devices.append(device_index)
            cuda_states[device_index] = torch.cuda.get_rng_state(device_index)

        primary = self._decode_from_encoded(
            encoded, skips, regime, dynamics_state, x,
            self.up, self.decoder, self.decoder_regime_film,
            self.cross_conditioner)
        if self.training:
            # Enter after the primary call so fork_rng snapshots (and later
            # restores) the externally visible post-primary generator state.
            with torch.random.fork_rng(devices=cuda_devices):
                if cpu_state is None:  # pragma: no cover - typing invariant
                    raise RuntimeError("paired decoder CPU RNG state is missing")
                torch.set_rng_state(cpu_state)
                for device_index, state in cuda_states.items():
                    torch.cuda.set_rng_state(state, device=device_index)
                dynamics = self._decode_from_encoded(
                    encoded, skips, regime, dynamics_state, x,
                    self.wake_dynamics_up, self.wake_dynamics_decoder,
                    self.wake_dynamics_decoder_regime_film,
                    self.wake_dynamics_cross_conditioner)
        else:
            dynamics = self._decode_from_encoded(
                encoded, skips, regime, dynamics_state, x,
                self.wake_dynamics_up, self.wake_dynamics_decoder,
                self.wake_dynamics_decoder_regime_film,
                self.wake_dynamics_cross_conditioner)
        if len(primary) != len(dynamics):
            raise RuntimeError("wake decoder scale contract changed")
        return primary, dynamics

    def _decode_role_features(
            self, x: torch.Tensor,
            ) -> tuple[list[torch.Tensor], list[torch.Tensor], None, None]:
        primary, dynamics = self._decode_wake_features(x)
        # The point builder below consumes both paths explicitly.  The tagged
        # list lets the inherited outer energy controller recover the primary
        # feature coordinate.  Returning no UQ-role override likewise keeps
        # native UQ on primary features (while it sees corrected members), so
        # this experiment changes only the proposed wake fluctuation role.
        paired = _WakeDynamicsFeatures(dynamics, primary)
        return primary, paired, None, None

    def _physical_energy_delta(
            self, features: list[torch.Tensor],
            history: torch.Tensor | None = None) -> torch.Tensor:
        controller_features = (
            features.primary_features
            if isinstance(features, _WakeDynamicsFeatures) else features)
        return EnsembleUNet2d._physical_energy_delta(
            self, controller_features, history)

    def _plain_members(
            self, x: torch.Tensor, features: list[torch.Tensor], *,
            dynamics: bool) -> torch.Tensor:
        mean_correction = None
        if dynamics and self.wake_dynamics_mean_profile_refiner is not None:
            mean_correction = self.wake_dynamics_mean_profile_refiner(
                features[-1])
        elif not dynamics and self.mean_profile_refiner is not None:
            mean_correction = self.mean_profile_refiner(features[-1])
        return EnsembleUNet2d._members_from_features(
            self, x, features, fluctuation_features=None,
            use_member_adapters=True, base_members=None,
            mean_profile_correction=mean_correction)

    @staticmethod
    def _wake_mask(value: torch.Tensor) -> torch.Tensor:
        width = value.shape[3]
        if width < WAKE_X_STOP:
            raise ValueError(
                f"wake complementary input width must be >= {WAKE_X_STOP}")
        mask = torch.zeros(
            (1, 1, 1, width, 1), dtype=torch.bool,
            device=value.device)
        mask[..., WAKE_X_START:WAKE_X_STOP, :] = True
        return mask

    def _wake_shell_members(
            self, primary_members: torch.Tensor,
            dynamics_members: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (primary_members.shape != dynamics_members.shape
                or primary_members.ndim != 6
                or primary_members.shape[-1] < 3):
            raise ValueError(
                "wake members must share shape (B,M,T,H,W,C), C>=3")
        primary_point = primary_members.mean(dim=1)
        dynamics_point = dynamics_members.mean(dim=1)
        wake = self._wake_mask(primary_point)

        raw_delta = (
            dynamics_point[..., :2] - primary_point[..., :2])
        raw_delta = torch.where(wake, raw_delta, torch.zeros_like(raw_delta))
        centered_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)

        primary_uv = primary_point[..., :2]
        primary_mean = primary_uv.mean(dim=1, keepdim=True)
        primary_fluctuation = primary_uv - primary_mean
        proposed_fluctuation = primary_fluctuation + centered_delta

        # Official TKE lives in physical coordinates; target_std therefore
        # belongs inside the joint-u/v shell norm.  One scalar is used for the
        # whole horizon and both velocity components at each spatial pixel.
        physical_scale = self.target_std[:2].to(
            proposed_fluctuation).view(1, 1, 1, 1, 2)
        primary_energy = (
            primary_fluctuation * physical_scale
        ).square().sum(dim=(1, 4), keepdim=True)
        proposed_energy = (
            proposed_fluctuation * physical_scale
        ).square().sum(dim=(1, 4), keepdim=True)
        tiny = torch.finfo(proposed_energy.dtype).tiny
        relative_floor = (
            64.0 * torch.finfo(proposed_energy.dtype).eps
            * primary_energy.clamp_min(tiny))
        degenerate = (
            (primary_energy > 0.0)
            & (proposed_energy <= relative_floor))
        shell_scale = torch.sqrt(
            primary_energy / proposed_energy.clamp_min(tiny))
        shell_scale = torch.where(
            primary_energy > 0.0, shell_scale,
            torch.zeros_like(shell_scale))
        candidate_uv = primary_mean + proposed_fluctuation * shell_scale
        shell_delta = torch.where(
            wake, candidate_uv - primary_uv,
            torch.zeros_like(candidate_uv))
        shell_delta = torch.where(
            degenerate.expand_as(shell_delta),
            torch.zeros_like(shell_delta), shell_delta)

        # If both decoder predictions are bitwise equal at a pixel, bypass
        # every reconstructive floating-point operation at that pixel.  The
        # STE backward deliberately still trains the complementary decoder.
        bypass = centered_delta.eq(0.0).all(
            dim=(1, 4), keepdim=True) | degenerate
        member_uv = _ExactIdentityAddSTE.apply(
            primary_members[..., :2], shell_delta.unsqueeze(1),
            bypass.unsqueeze(1))
        corrected = torch.cat((
            member_uv,
            primary_members[..., 2:],
        ), dim=-1)
        applied_delta = member_uv[:, 0] - primary_members[:, 0, ..., :2]
        return corrected, centered_delta, applied_delta

    def _members_from_features(
            self, x: torch.Tensor,
            features: list[torch.Tensor],
            fluctuation_features: list[torch.Tensor] | None = None,
            use_member_adapters: bool = True,
            base_members: torch.Tensor | None = None,
            mean_profile_correction: torch.Tensor | None = None,
            ) -> torch.Tensor:
        if (fluctuation_features is None or not use_member_adapters
                or base_members is not None
                or mean_profile_correction is not None):
            raise ValueError(
                "wake complementary members require the paired native XCB "
                "decoder contract")
        primary_members = self._plain_members(x, features, dynamics=False)
        dynamics_members = self._plain_members(
            x, fluctuation_features, dynamics=True)
        corrected, _centered, _delta = self._wake_shell_members(
            primary_members, dynamics_members)
        return corrected

    def forward_wake_fields(
            self, x: torch.Tensor) -> WakeFluctuationComplementaryFields:
        """Run and expose the three point/member stages used in deployment."""
        primary_features, dynamics_features = self._decode_wake_features(x)
        primary_members = self._plain_members(
            x, primary_features, dynamics=False)
        dynamics_members = self._plain_members(
            x, dynamics_features, dynamics=True)
        shell_members, centered_delta, shell_correction = (
            self._wake_shell_members(primary_members, dynamics_members))
        shell_point = shell_members.mean(dim=1)
        if self.config.physical_energy_member_sync:
            controller_members = self._energy_controlled_members(
                shell_members, primary_features, x)
            controller_point = controller_members.mean(dim=1)
        else:
            controller_members = shell_members
            controller_point = self._energy_controlled_prediction(
                shell_point, primary_features, x)
        return WakeFluctuationComplementaryFields(
            primary_features=primary_features,
            dynamics_features=dynamics_features,
            primary_members_pre_shell=primary_members,
            dynamics_members_pre_shell=dynamics_members,
            primary_point_pre_shell=primary_members.mean(dim=1),
            dynamics_point_pre_shell=dynamics_members.mean(dim=1),
            centered_wake_delta_uv=centered_delta,
            shell_correction_uv=shell_correction,
            wake_shell_members_pre_controller=shell_members,
            wake_shell_point_pre_controller=shell_point,
            controller_members_post=controller_members,
            controller_point_post=controller_point,
        )


def build_wake_fluctuation_complementary(
        backbone_config: EnsembleUNet2dConfig,
        ) -> WakeFluctuationComplementaryUNet2d:
    return WakeFluctuationComplementaryUNet2d(backbone_config)
