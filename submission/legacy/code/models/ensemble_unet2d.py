"""Shared-backbone U-Net ensemble with a conditional residual distribution.

The model keeps inference to one encoder/decoder pass. Several lightweight
point heads provide sample-local epistemic disagreement, while a nonlinear
multi-scale decoder predicts monotone residual quantiles. Prediction intervals
are selected from those quantiles by the official SPS width utility instead of
using a fixed coverage target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import log

import torch
from torch import nn
from torch.nn import functional as F

from models.shared_point_feature_refiner import SharedPointFeatureRefiner
from models.unet2d import (
    HISTORY_AXIS_ADAPTER_MODES,
    LocalDynamicsCrossConditioner,
    ResidualConvBlock,
    UNet2d,
    UNet2dConfig,
    _group_norm,
    history_regime_summary,
)
SPS_SIGMA = 0.0563870259

# These modes reuse the existing explicit two-decoder boundary, but unlike the
# earlier mean/fluctuation experiment their second decoder is an exact copy of
# the first one.  Both execute the same history-only scalar router.  The fixed
# arm discards its value for point routing, making it an equal-capacity and
# equal-operator control for the learned-router arm.
REGIME_EXPERT_DECODER_MODES = (
    "regime_expert_fixed_average",
    "regime_expert_history_router",
)


@dataclass(frozen=True)
class EnsembleUNet2dConfig(UNet2dConfig):
    ensemble_members: int = 4
    uq_hidden_channels: int = 64
    member_adapter_channels: int = 0
    point_feature_refiner_channels: int = 0
    fluctuation_role_channels: int = 0
    history_conditioned_operator_basis: bool = False
    dual_decoder_mode: str = "none"
    mean_preserving_member_adapters: bool = False
    detach_member_adapter_context: bool = False
    detach_uq_context: bool = False
    budgeted_disagreement_allocation: bool = False
    disagreement_allocation_log_limit: float = 1.5
    mean_profile_channels: int = 0
    uq_flow_context: bool = False
    uq_full_history_context: bool = False
    uq_sparse_history_context: bool = False
    residual_quantile_intervals: bool = False
    utility_refinement: bool = False
    utility_refinement_log_limit: float = 0.75
    utility_refinement_kernel: int = 1
    utility_refinement_blocks: int = 0
    output_space_uq_channels: int = 0
    output_space_uq_blocks: int = 2
    output_space_uq_log_limit: float = 0.75
    output_space_uq_budget_channels: int = 0
    output_space_uq_budget_blocks: int = 1
    output_space_uq_budget_log_limit: float = 0.50
    temporal_uq_refinement: bool = False
    temporal_uq_channels: int = 24
    temporal_uq_blocks: int = 2
    temporal_uq_kernel: int = 3
    temporal_uq_logit_limit: float = 1.5
    coverage_calibration: bool = False
    coverage_calibration_kernel: int = 3
    coverage_calibration_blocks: int = 1
    coverage_calibration_logit_limit: float = 2.0
    coverage_calibration_mode: str = "independent"
    coverage_calibration_temperature_limit: float = 0.75
    quantile_levels: tuple[float, ...] = (
        0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)
    quantile_gap_bias: float = -4.0
    disagreement_scale_fraction: float = 0.0
    shared_member_uv_mean: bool = False
    decoupled_member_residual_channels: int = 0
    decoupled_member_residual_blocks: int = 2
    physical_energy_controller: bool = False
    physical_energy_member_sync: bool = False
    physical_energy_multiscale: bool = False
    physical_energy_horizon_local: bool = False
    physical_energy_history_conditioned: bool = False
    physical_energy_context_head: bool = False
    physical_energy_decoder_context: str = "none"
    physical_energy_history_steps: int = 5
    physical_energy_hidden_channels: int = 32
    physical_energy_log_limit: float = 1.0
    scalar_potential_residual_mode: str = "none"
    scalar_potential_residual_channels: int = 0
    scalar_potential_residual_blocks: int = 2
    scalar_potential_residual_limit: float = 0.05
    phase_delta_hidden_channels: int = 0
    phase_delta_steps: int = 5
    temporal_point_decoder: bool = False
    temporal_point_channels: int = 24
    temporal_point_blocks: int = 2
    temporal_point_kernel: int = 5
    temporal_point_amplitude_factorization: bool = False
    temporal_point_amplitude_limit: float = 3.0
    latent_ode_point_decoder: bool = False
    latent_ode_channels: int = 64
    latent_ode_solver_substeps: int = 1
    latent_ode_step_size: float = 0.25
    latent_ode_derivative_limit: float = 1.0
    latent_ode_residual_limit: float = 4.0
    latent_ude_band_decoder: bool = False
    latent_ude_closure_limit: float = 0.25
    latent_ude_coarse_residual: bool = False
    latent_ude_coarse_residual_limit: float = 0.75
    latent_ude_coarse_velocity_only: bool = False
    latent_ude_exclusive_coarse_fluctuation: bool = False
    isoenergetic_temporal_transport: bool = False
    temporal_transport_channels: int = 24
    temporal_transport_blocks: int = 2
    temporal_transport_kernel: int = 5
    temporal_transport_max_displacement: float = 1.5
    temporal_band_mixer: bool = False
    temporal_band_channels: int = 32
    temporal_band_count: int = 4
    temporal_band_log_gain_limit: float = 0.50
    temporal_band_phase_limit: float = 0.50
    spatial_transport_warp: bool = False
    spatial_transport_channels: int = 32
    spatial_transport_max_displacement: float = 1.5
    recurrent_latent_rollout_channels: int = 0
    recurrent_latent_rollout_limit: float = 0.25
    recurrent_latent_rollout_downsample: int = 4
    recurrent_latent_rollout_lowpass_modes: int = 0
    point_readout_adapter_rank: int = 0
    hard_region_expert_channels: int = 0
    hard_region_expert_blocks: int = 2
    hard_region_expert_limit: float = 0.75
    hard_region_expert_detach_context: bool = True
    hard_region_expert_feature_noise_std: float = 0.0
    conservative_closure_experts: int = 0
    conservative_closure_channels: int = 64
    conservative_closure_blocks: int = 2
    conservative_closure_flux_limit: float = 0.125
    conservative_closure_router_temperature: float = 1.0
    residual_mixture_components: int = 0
    residual_mixture_mean_limit: float = 4.0
    residual_mixture_scale_floor: float = 0.025
    residual_mixture_scale_init: float = 0.60
    residual_mixture_probability_shrinkage: float = 0.25
    residual_mixture_endpoint_mode: str = "native_bank"
    residual_mixture_endpoint_shrinkage: float = 0.25
    residual_mixture_bisection_steps: int = 8
    residual_mixture_geometry_projection: str = "sample_budget"
    late_multiscale_sidecar_mode: str = "none"
    late_multiscale_sidecar_channels: tuple[int, int] = (32, 48)
    late_multiscale_sidecar_blocks: int = 1
    late_multiscale_sidecar_kernel: int = 5

    def validate(self) -> None:
        super().validate()
        if self.late_multiscale_sidecar_mode not in HISTORY_AXIS_ADAPTER_MODES:
            raise ValueError(
                "late multiscale sidecar mode must be one of "
                f"{HISTORY_AXIS_ADAPTER_MODES}")
        if (len(self.late_multiscale_sidecar_channels) != 2
                or any(value <= 0
                       for value in self.late_multiscale_sidecar_channels)):
            raise ValueError(
                "late multiscale sidecar requires two positive widths")
        if self.late_multiscale_sidecar_blocks <= 0:
            raise ValueError(
                "late multiscale sidecar blocks must be positive")
        if self.late_multiscale_sidecar_kernel < 2:
            raise ValueError(
                "late multiscale sidecar kernel must be at least two")
        if (self.late_multiscale_sidecar_mode != "none"
                and self.depth != 4):
            raise ValueError(
                "late multiscale sidecar requires the depth-four pyramid")
        if (self.late_multiscale_sidecar_mode != "none"
                and self.dual_decoder_mode != "none"):
            raise ValueError(
                "late multiscale sidecar is screened with one decoder")
        if self.ensemble_members < 2:
            raise ValueError("ensemble_members must be at least two")
        if self.uq_hidden_channels <= 0:
            raise ValueError("uq_hidden_channels must be positive")
        if self.uq_full_history_context and (not self.uq_flow_context or self.in_steps != 20):
            raise ValueError("full UQ history requires the 20-frame flow context")
        if self.uq_sparse_history_context and (not self.uq_flow_context or self.in_steps != 20
                                               or self.uq_full_history_context):
            raise ValueError("sparse UQ history requires the original 20-frame flow-context layout")
        if self.point_feature_refiner_channels not in (0, 32):
            raise ValueError("point feature refiner is fixed to disabled or width32")
        if self.member_adapter_channels < 0:
            raise ValueError("member_adapter_channels must be non-negative")
        if self.fluctuation_role_channels < 0:
            raise ValueError(
                "fluctuation role channels must be non-negative")
        if (self.history_conditioned_operator_basis
                and (not self.mean_fluctuation or self.depth != 4)):
            raise ValueError(
                "HCOB-F requires mean_fluctuation and the depth-four "
                "B210 decoder pyramid")
        if (self.history_conditioned_operator_basis
                and (self.fluctuation_role_channels
                     or self.dual_decoder_mode != "none")):
            raise ValueError(
                "HCOB-F is the single-decoder fluctuation specialization")
        if self.dual_decoder_mode not in (
                "none", "shared_fused", "mean_fluctuation",
                "fused_mean_fluctuation", "learned_fluctuation_gate",
                "separate_point_uq_fluctuation_gates",
                *REGIME_EXPERT_DECODER_MODES):
            raise ValueError(
                "dual decoder mode must be none/shared_fused/"
                "mean_fluctuation/fused_mean_fluctuation/"
                "learned_fluctuation_gate/"
                "separate_point_uq_fluctuation_gates/"
                "regime_expert_fixed_average/"
                "regime_expert_history_router")
        if self.dual_decoder_mode != "none" and not self.mean_fluctuation:
            raise ValueError(
                "dual decoder requires mean_fluctuation=True")
        if (self.dual_decoder_mode != "none"
                and (self.fluctuation_role_channels
                     or self.temporal_point_decoder
                     or self.latent_ode_point_decoder
                     or self.latent_ude_band_decoder
                     or self.point_readout_adapter_rank)):
            raise ValueError(
                "dual decoder requires unspecialized direct point heads")
        if (self.dual_decoder_mode in REGIME_EXPERT_DECODER_MODES
                and self.deformable_history_memory_mode != "none"):
            raise ValueError(
                "regime expert decoders do not support deformable memory")
        if self.fluctuation_role_channels and not self.mean_fluctuation:
            raise ValueError(
                "fluctuation role tower requires mean_fluctuation=True")
        if (self.fluctuation_role_channels
                and (self.temporal_point_decoder
                     or self.latent_ode_point_decoder
                     or self.latent_ude_band_decoder
                     or self.point_readout_adapter_rank)):
            raise ValueError(
                "fluctuation role tower requires legacy direct point heads")
        if (self.mean_preserving_member_adapters
                and not self.member_adapter_channels):
            raise ValueError(
                "mean-preserving member adapters require adapter channels")
        if (self.detach_member_adapter_context
                and not self.mean_preserving_member_adapters):
            raise ValueError(
                "detached adapter context requires mean-preserving adapters")
        if (self.budgeted_disagreement_allocation
                and not self.mean_preserving_member_adapters):
            raise ValueError(
                "budgeted disagreement requires mean-preserving adapters")
        if not 0.0 < self.disagreement_allocation_log_limit < float("inf"):
            raise ValueError(
                "disagreement allocation log limit must be positive")
        if self.mean_profile_channels < 0:
            raise ValueError("mean_profile_channels must be non-negative")
        if self.shared_member_uv_mean and not self.mean_fluctuation:
            raise ValueError(
                "shared member u/v mean requires mean_fluctuation=True")
        if self.decoupled_member_residual_channels < 0:
            raise ValueError(
                "decoupled member residual channels must be non-negative")
        if self.decoupled_member_residual_blocks <= 0:
            raise ValueError(
                "decoupled member residual blocks must be positive")
        if (self.shared_member_uv_mean
                and self.decoupled_member_residual_channels):
            raise ValueError(
                "shared member mean and decoupled residual are separate arms")
        if self.physical_energy_controller and not self.mean_fluctuation:
            raise ValueError(
                "physical energy controller requires mean_fluctuation=True")
        if (self.physical_energy_member_sync
                and not self.physical_energy_controller):
            raise ValueError(
                "member-synchronous energy requires the physical energy "
                "controller")
        if (self.physical_energy_multiscale
                and not self.physical_energy_controller):
            raise ValueError(
                "multiscale physical energy requires the controller")
        if (self.physical_energy_horizon_local
                and not self.physical_energy_controller):
            raise ValueError(
                "horizon-local physical energy requires the controller")
        if (self.physical_energy_history_conditioned
                and not self.physical_energy_controller):
            raise ValueError(
                "history-conditioned physical energy requires the controller")
        if (self.physical_energy_horizon_local
                and (self.physical_energy_multiscale
                     or self.physical_energy_history_conditioned)):
            raise ValueError(
                "physical energy controller variants are separate arms")
        if not 2 <= self.physical_energy_history_steps < self.in_steps:
            raise ValueError(
                "physical energy history steps must be in [2, in_steps)")
        if self.physical_energy_hidden_channels <= 0:
            raise ValueError(
                "physical energy hidden channels must be positive")
        if not 0.0 < self.physical_energy_log_limit < float("inf"):
            raise ValueError("physical energy log limit must be positive")
        if self.scalar_potential_residual_mode not in (
                "none", "gradient", "curl"):
            raise ValueError(
                "scalar potential residual mode must be none/gradient/curl")
        if self.scalar_potential_residual_channels < 0:
            raise ValueError(
                "scalar potential residual channels must be non-negative")
        if self.scalar_potential_residual_blocks <= 0:
            raise ValueError("scalar potential residual blocks must be positive")
        if not 0.0 < self.scalar_potential_residual_limit < float("inf"):
            raise ValueError("scalar potential residual limit must be positive")
        if ((self.scalar_potential_residual_mode == "none")
                != (self.scalar_potential_residual_channels == 0)):
            raise ValueError(
                "scalar potential residual mode and channels must be enabled "
                "together")
        if (self.scalar_potential_residual_mode != "none"
                and self.out_channels < 2):
            raise ValueError("scalar potential residual requires u/v channels")
        if self.phase_delta_hidden_channels < 0:
            raise ValueError("phase delta hidden channels must be non-negative")
        if not 2 <= self.phase_delta_steps < self.in_steps:
            raise ValueError("phase delta steps must be in [2, in_steps)")
        if self.temporal_point_channels <= 0:
            raise ValueError("temporal point channels must be positive")
        if self.temporal_point_blocks <= 0:
            raise ValueError("temporal point blocks must be positive")
        if (self.temporal_point_kernel <= 0
                or self.temporal_point_kernel % 2 == 0):
            raise ValueError(
                "temporal point kernel must be positive and odd")
        if (self.temporal_point_amplitude_factorization
                and not self.temporal_point_decoder):
            raise ValueError(
                "temporal amplitude factorization requires its decoder")
        if not 0.0 < self.temporal_point_amplitude_limit < float("inf"):
            raise ValueError("temporal point amplitude limit must be positive")
        if self.latent_ode_channels <= 0:
            raise ValueError("latent ODE channels must be positive")
        if (self.latent_ode_point_decoder
                and self.latent_ode_channels % self.groups):
            raise ValueError(
                "latent ODE channels must be divisible by groups")
        if self.latent_ode_solver_substeps <= 0:
            raise ValueError("latent ODE solver substeps must be positive")
        if not 0.0 < self.latent_ode_step_size < float("inf"):
            raise ValueError("latent ODE step size must be positive")
        if not 0.0 < self.latent_ode_derivative_limit < float("inf"):
            raise ValueError("latent ODE derivative limit must be positive")
        if not 0.0 < self.latent_ode_residual_limit < float("inf"):
            raise ValueError("latent ODE residual limit must be positive")
        if not 0.0 < self.latent_ude_closure_limit < float("inf"):
            raise ValueError("latent UDE closure limit must be positive")
        if not 0.0 < self.latent_ude_coarse_residual_limit < float("inf"):
            raise ValueError(
                "latent UDE coarse residual limit must be positive")
        if sum((
                self.temporal_point_decoder,
                self.latent_ode_point_decoder,
                self.latent_ude_band_decoder)) > 1:
            raise ValueError(
                "temporal, latent ODE, and latent UDE decoders are separate arms")
        if (self.latent_ode_point_decoder
                and (self.laplacian_point_pyramid
                     or self.member_adapter_channels)):
            raise ValueError(
                "latent ODE does not use legacy point pyramids/adapters")
        if (self.latent_ude_band_decoder
                and (self.laplacian_point_pyramid
                     or self.member_adapter_channels)):
            raise ValueError(
                "latent UDE owns its point bands and does not use legacy "
                "pyramids/adapters")
        if ((self.latent_ude_coarse_residual
             or self.latent_ude_exclusive_coarse_fluctuation)
                and (self.temporal_point_decoder
                     or self.latent_ode_point_decoder
                     or self.latent_ude_band_decoder)):
            raise ValueError(
                "latent UDE coarse residual requires the legacy direct path")
        if (self.latent_ude_coarse_residual
                and self.latent_ude_exclusive_coarse_fluctuation):
            raise ValueError(
                "additive and exclusive coarse UDE are separate arms")
        if (self.latent_ude_coarse_velocity_only
                and not (self.latent_ude_coarse_residual
                         or self.latent_ude_exclusive_coarse_fluctuation)):
            raise ValueError(
                "velocity-only coarse UDE requires its residual branch")
        if self.temporal_transport_channels <= 0:
            raise ValueError("temporal transport channels must be positive")
        if self.temporal_transport_blocks <= 0:
            raise ValueError("temporal transport blocks must be positive")
        if (self.temporal_transport_kernel <= 0
                or self.temporal_transport_kernel % 2 == 0):
            raise ValueError(
                "temporal transport kernel must be positive and odd")
        if not 0.0 < self.temporal_transport_max_displacement < float("inf"):
            raise ValueError(
                "temporal transport displacement must be positive")
        if self.temporal_band_channels <= 0:
            raise ValueError("temporal band channels must be positive")
        if not 2 <= self.temporal_band_count <= self.out_steps // 2:
            raise ValueError(
                "temporal band count must lie in [2, out_steps // 2]")
        if not 0.0 < self.temporal_band_log_gain_limit < float("inf"):
            raise ValueError("temporal band log-gain limit must be positive")
        if not 0.0 < self.temporal_band_phase_limit < float("inf"):
            raise ValueError("temporal band phase limit must be positive")
        if (self.isoenergetic_temporal_transport
                and self.temporal_band_mixer):
            raise ValueError(
                "isoenergetic transport and temporal band mixing are "
                "separate arms")
        if self.spatial_transport_channels <= 0:
            raise ValueError("spatial transport channels must be positive")
        if not 0.0 < self.spatial_transport_max_displacement < float("inf"):
            raise ValueError(
                "spatial transport displacement must be positive")
        if self.spatial_transport_warp and (
                self.isoenergetic_temporal_transport
                or self.temporal_band_mixer):
            raise ValueError(
                "spatial, temporal-band, and isoenergetic transports are "
                "separate arms")
        if self.recurrent_latent_rollout_channels < 0:
            raise ValueError(
                "recurrent latent rollout channels must be non-negative")
        if (self.recurrent_latent_rollout_channels
                and self.recurrent_latent_rollout_channels % self.groups):
            raise ValueError(
                "recurrent latent rollout channels must be divisible by groups")
        if not 0.0 < self.recurrent_latent_rollout_limit < float("inf"):
            raise ValueError(
                "recurrent latent rollout limit must be positive")
        if self.recurrent_latent_rollout_downsample <= 0:
            raise ValueError(
                "recurrent latent rollout downsample must be positive")
        if not 0 <= self.recurrent_latent_rollout_lowpass_modes < self.out_steps:
            raise ValueError(
                "recurrent latent low-pass modes must lie in [0, out_steps)")
        if (self.recurrent_latent_rollout_lowpass_modes
                and not self.recurrent_latent_rollout_channels):
            raise ValueError(
                "recurrent latent low-pass split requires its rollout")
        if self.point_readout_adapter_rank < 0:
            raise ValueError("point readout adapter rank must be non-negative")
        if (self.point_readout_adapter_rank
                and (self.temporal_point_decoder
                     or self.latent_ode_point_decoder
                     or self.latent_ude_band_decoder)):
            raise ValueError(
                "point readout adapters require legacy multi-scale heads")
        if self.hard_region_expert_channels < 0:
            raise ValueError(
                "hard-region expert channels must be non-negative")
        if self.hard_region_expert_blocks <= 0:
            raise ValueError("hard-region expert blocks must be positive")
        if not 0.0 < self.hard_region_expert_limit < float("inf"):
            raise ValueError("hard-region expert limit must be positive")
        if (not 0.0 <= self.hard_region_expert_feature_noise_std
                < float("inf")):
            raise ValueError(
                "hard-region expert feature noise must be non-negative")
        if self.conservative_closure_experts not in (0, 2, 3, 4):
            raise ValueError(
                "conservative closure experts must be zero or lie in [2, 4]")
        if self.conservative_closure_channels <= 0:
            raise ValueError(
                "conservative closure channels must be positive")
        if self.conservative_closure_blocks <= 0:
            raise ValueError(
                "conservative closure blocks must be positive")
        if not 0.0 < self.conservative_closure_flux_limit < float("inf"):
            raise ValueError(
                "conservative closure flux limit must be positive")
        if not (
                0.0 < self.conservative_closure_router_temperature
                < float("inf")):
            raise ValueError(
                "conservative closure router temperature must be positive")
        if (self.conservative_closure_experts
                and self.hard_region_expert_channels):
            raise ValueError(
                "conservative closure and hard-region expert are separate arms")
        if (self.conservative_closure_experts
                and (self.temporal_point_decoder
                     or self.latent_ode_point_decoder
                     or self.latent_ude_band_decoder
                     or self.recurrent_latent_rollout_channels)):
            raise ValueError(
                "conservative closure requires the legacy direct point path")
        if self.residual_mixture_components not in (0, 2, 3, 4, 5):
            raise ValueError(
                "residual mixture components must be zero or lie in [2, 5]")
        if not 0.0 < self.residual_mixture_mean_limit < float("inf"):
            raise ValueError("residual mixture mean limit must be positive")
        if not 0.0 < self.residual_mixture_scale_floor < float("inf"):
            raise ValueError("residual mixture scale floor must be positive")
        if not (
                self.residual_mixture_scale_floor
                < self.residual_mixture_scale_init < float("inf")):
            raise ValueError(
                "residual mixture initial scale must exceed its floor")
        if not 0.0 <= self.residual_mixture_probability_shrinkage <= 1.0:
            raise ValueError(
                "residual mixture probability shrinkage must lie in [0, 1]")
        if self.residual_mixture_endpoint_mode not in (
                "native_bank", "mixture_quantiles", "mixture_projected"):
            raise ValueError(
                "residual mixture endpoint mode is not supported")
        if not 0.0 <= self.residual_mixture_endpoint_shrinkage <= 1.0:
            raise ValueError(
                "residual mixture endpoint shrinkage must lie in [0, 1]")
        if not 6 <= self.residual_mixture_bisection_steps <= 16:
            raise ValueError(
                "residual mixture bisection steps must lie in [6, 16]")
        if self.residual_mixture_geometry_projection not in (
                "sample_budget", "pixel_width"):
            raise ValueError(
                "residual mixture geometry projection is not supported")
        if self.uq_flow_context and self.in_steps < 6:
            raise ValueError("UQ flow context requires at least six input steps")
        levels = tuple(float(value) for value in self.quantile_levels)
        if len(levels) < 5 or len(levels) % 2 == 0:
            raise ValueError("quantile_levels must contain an odd count >= 5")
        if levels != tuple(sorted(set(levels))):
            raise ValueError("quantile_levels must be strictly increasing")
        if any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("quantile levels must lie in (0,1)")
        if levels[len(levels) // 2] != 0.5:
            raise ValueError("the middle quantile must be exactly 0.5")
        if not self.predict_uncertainty:
            raise ValueError("ensemble_unet requires predict_uncertainty=True")
        if not 0.0 <= self.disagreement_scale_fraction < float("inf"):
            raise ValueError("disagreement scale must be finite/non-negative")
        if not 0.0 < self.utility_refinement_log_limit < float("inf"):
            raise ValueError("utility refinement log limit must be positive")
        if (self.utility_refinement_kernel <= 0
                or self.utility_refinement_kernel % 2 == 0):
            raise ValueError("utility refinement kernel must be positive and odd")
        if self.utility_refinement_blocks < 0:
            raise ValueError("utility refinement blocks must be non-negative")
        if self.output_space_uq_channels < 0:
            raise ValueError(
                "output-space UQ channels must be non-negative")
        if self.output_space_uq_blocks < 0:
            raise ValueError(
                "output-space UQ blocks must be non-negative")
        if not 0.0 < self.output_space_uq_log_limit < float("inf"):
            raise ValueError(
                "output-space UQ log limit must be positive")
        if self.output_space_uq_channels and self.in_steps < 6:
            raise ValueError(
                "output-space UQ requires at least six input steps")
        if self.output_space_uq_budget_channels < 0:
            raise ValueError(
                "output-space UQ budget channels must be non-negative")
        if self.output_space_uq_budget_blocks < 0:
            raise ValueError(
                "output-space UQ budget blocks must be non-negative")
        if not 0.0 < self.output_space_uq_budget_log_limit < float("inf"):
            raise ValueError(
                "output-space UQ budget log limit must be positive")
        if self.output_space_uq_budget_channels and self.in_steps < 6:
            raise ValueError(
                "output-space UQ budget requires at least six input steps")
        if self.temporal_uq_channels <= 0:
            raise ValueError("temporal UQ channels must be positive")
        if self.temporal_uq_blocks <= 0:
            raise ValueError("temporal UQ blocks must be positive")
        if (self.temporal_uq_kernel <= 0
                or self.temporal_uq_kernel % 2 == 0):
            raise ValueError("temporal UQ kernel must be positive and odd")
        if not 0.0 < self.temporal_uq_logit_limit < float("inf"):
            raise ValueError("temporal UQ logit limit must be positive")
        if (self.coverage_calibration_kernel <= 0
                or self.coverage_calibration_kernel % 2 == 0):
            raise ValueError(
                "coverage calibration kernel must be positive and odd")
        if self.coverage_calibration_blocks < 0:
            raise ValueError(
                "coverage calibration blocks must be non-negative")
        if not 0.0 < self.coverage_calibration_logit_limit < float("inf"):
            raise ValueError(
                "coverage calibration logit limit must be positive")
        if self.coverage_calibration_mode not in (
                "independent", "temperature"):
            raise ValueError(
                "coverage calibration mode must be independent/temperature")
        if not 0.0 < self.coverage_calibration_temperature_limit < float("inf"):
            raise ValueError(
                "coverage calibration temperature limit must be positive")


def _flatten_forecast(value: torch.Tensor) -> torch.Tensor:
    batch, steps, height, width, channels = value.shape
    return value.permute(0, 1, 4, 2, 3).reshape(
        batch, steps * channels, height, width)


class _ExactIdentityAddSTE(torch.autograd.Function):
    """Keep an exact zero-controller forward while retaining its gradient."""

    @staticmethod
    def forward(ctx, legacy: torch.Tensor, delta: torch.Tensor,
                bypass: torch.Tensor) -> torch.Tensor:
        return torch.where(bypass, legacy, legacy + delta)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, grad_output, None


class MemberFeatureAdapter(nn.Module):
    """Zero-initialized spatial residual branch for one member and scale."""

    def __init__(self, width: int, bottleneck: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, bottleneck, 1),
            _group_norm(groups, bottleneck),
            nn.SiLU(),
            ResidualConvBlock(bottleneck, bottleneck, groups),
            nn.Conv2d(bottleneck, width, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class HistoryConditionedOperatorBasis(nn.Module):
    """One zero-output, three-operator fluctuation residual at one scale.

    The compact body is deliberately fixed to 64 channels.  Its three paths
    have point, local and low-mode spectral spatial responsibilities; only the
    final projection is zero initialized, so an update-zero model is exactly
    the legacy decoder.  Consequently only that projection receives a
    non-zero first-update gradient; the conditioner and basis paths begin to
    receive gradients after the projection has moved.
    """

    compact_channels = 64
    low_modes = 4

    def __init__(self, width: int, condition_channels: int = 16):
        super().__init__()
        self.input_projection = nn.Conv2d(width, self.compact_channels, 1)
        self.pointwise = nn.Conv2d(
            self.compact_channels, self.compact_channels, 1)
        self.local_depthwise = nn.Conv2d(
            self.compact_channels, self.compact_channels, 5, padding=2,
            groups=self.compact_channels)
        self.local_mixing = nn.Conv2d(
            self.compact_channels, self.compact_channels, 1)
        self.spectral_weight = nn.Parameter(torch.empty(
            self.compact_channels, self.compact_channels,
            self.low_modes, self.low_modes, dtype=torch.cfloat))
        nn.init.normal_(self.spectral_weight.real, std=0.02)
        nn.init.normal_(self.spectral_weight.imag, std=0.02)
        self.operator_logits = nn.Linear(
            condition_channels, 3 * self.compact_channels)
        self.output_projection = nn.Conv2d(self.compact_channels, width, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _global(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the fixed low-mode complex channel map at the available size."""
        height, width = value.shape[-2:]
        modes_y = min(self.low_modes, height)
        modes_x = min(self.low_modes, width // 2 + 1)
        # CUDA FFT does not consistently support autocast half/bfloat16
        # inputs (and some supported sizes have further restrictions).  Keep
        # this complex low-mode path in FP32, then return to the surrounding
        # decoder dtype so it composes with AMP convolutions.
        with torch.autocast(device_type=value.device.type, enabled=False):
            spectrum = torch.fft.rfft2(value.float(), norm="ortho")
            output_spectrum = torch.zeros_like(spectrum)
            output_spectrum[:, :, :modes_y, :modes_x] = torch.einsum(
                "bcyx,ocyx->boyx", spectrum[:, :, :modes_y, :modes_x],
                self.spectral_weight[:, :, :modes_y, :modes_x])
            output = torch.fft.irfft2(
                output_spectrum, s=(height, width), norm="ortho")
        return output.to(dtype=value.dtype)

    def forward(self, feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[0] != feature.shape[0]:
            raise ValueError("HCOB-F condition batch does not match features")
        compact = self.input_projection(feature)
        operators = torch.stack((
            self.pointwise(compact),
            self.local_mixing(self.local_depthwise(compact)),
            self._global(compact),
        ), dim=1)
        weights = self.operator_logits(condition).reshape(
            feature.shape[0], 3, self.compact_channels, 1, 1).softmax(dim=1)
        return feature + self.output_projection((operators * weights).sum(dim=1))


class FluctuationDynamicsBlock(nn.Module):
    """Wide/local gated decoder block for the fluctuation-only path.

    This deliberately differs from the ordinary pair of dense 3x3
    convolutions.  A cheap projection first resolves the skip and upsampled
    state, then depthwise local and dilated operators compete pixelwise.  The
    branch therefore has a distinct spatial-dynamics bias without duplicating
    the full dense decoder cost.
    """

    def __init__(self, input_channels: int, output_channels: int, groups: int):
        super().__init__()
        self.skip = nn.Conv2d(input_channels, output_channels, 1)
        self.pre = nn.Sequential(
            _group_norm(groups, input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, 1),
            _group_norm(groups, output_channels),
            nn.SiLU(),
        )
        self.local = nn.Conv2d(
            output_channels, output_channels, 3, padding=1,
            groups=output_channels)
        self.wide = nn.Conv2d(
            output_channels, output_channels, 3, padding=2, dilation=2,
            groups=output_channels)
        self.gate = nn.Conv2d(output_channels, output_channels, 1)
        self.output = nn.Sequential(
            _group_norm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        state = self.pre(value)
        gate = torch.sigmoid(self.gate(state))
        dynamics = gate * self.local(state) + (1.0 - gate) * self.wide(state)
        return self.skip(value) + self.output(state + dynamics)


class FusedMemberFeatureAdapterReadout(nn.Module):
    """Inference-only adapter whose final projection is fused with a head.

    A trained feature adapter is consumed only by a 1x1 member point head:
    ``head(x + last(prefix(x)))``.  The two consecutive 1x1 projections can
    therefore be multiplied once at model load, leaving
    ``head(x) + fused(prefix(x))`` and avoiding the wide intermediate tensor.
    """

    def __init__(self, prefix: nn.Sequential, output: nn.Conv2d):
        super().__init__()
        self.prefix = prefix
        self.output = output

    @classmethod
    def from_modules(
            cls, adapter: MemberFeatureAdapter,
            head: nn.Conv2d) -> "FusedMemberFeatureAdapterReadout":
        children = list(adapter.body.children())
        if not children or not isinstance(children[-1], nn.Conv2d):
            raise TypeError("member adapter has no final convolution")
        last = children[-1]
        if (last.kernel_size != (1, 1) or last.groups != 1
                or head.kernel_size != (1, 1) or head.groups != 1
                or last.out_channels != head.in_channels):
            raise ValueError("member adapter/readout cannot be fused")
        output = nn.Conv2d(
            last.in_channels, head.out_channels, 1, bias=True,
            device=last.weight.device, dtype=last.weight.dtype)
        with torch.no_grad():
            head_matrix = head.weight[:, :, 0, 0].float()
            last_matrix = last.weight[:, :, 0, 0].float()
            output.weight.copy_(
                (head_matrix @ last_matrix).to(output.weight.dtype)[
                    :, :, None, None])
            last_bias = (
                last.bias.float()
                if last.bias is not None
                else torch.zeros(
                    last.out_channels, device=last.weight.device,
                    dtype=torch.float32))
            output.bias.copy_(
                (head_matrix @ last_bias).to(output.bias.dtype))
        return cls(nn.Sequential(*children[:-1]), output)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.prefix(value))


class DecoupledMemberResidualDecoder(nn.Module):
    """Learn member diversity without a gradient path into point features.

    The legacy member predictions remain the update-zero forward baseline but
    are detached before this branch.  A small decoder reads a detached finest
    feature and predicts member residuals whose cross-member mean is exactly
    removed.  Member/UQ supervision can therefore train this module without
    directly rewriting the point backbone or the aggregate point forecast.
    """

    def __init__(
            self, input_channels: int, hidden_channels: int,
            members: int, out_steps: int, out_channels: int,
            blocks: int, groups: int):
        super().__init__()
        self.members = int(members)
        self.out_steps = int(out_steps)
        self.out_channels = int(out_channels)
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            *(ResidualConvBlock(
                hidden_channels, hidden_channels, groups)
              for _ in range(blocks)),
        )
        self.output = nn.Conv2d(
            hidden_channels,
            members * out_steps * out_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
            self, members: torch.Tensor,
            finest_feature: torch.Tensor) -> torch.Tensor:
        if (members.ndim != 6 or members.shape[1] != self.members
                or members.shape[2] != self.out_steps
                or members.shape[-1] != self.out_channels):
            raise ValueError("decoupled member residual shape changed")
        batch, _members, _steps, height, width, _channels = members.shape
        raw = self.output(self.body(finest_feature.detach()))
        if raw.shape[-2:] != (height, width):
            raw = F.interpolate(
                raw, size=(height, width), mode="bilinear",
                align_corners=False)
        residual = raw.reshape(
            batch, self.members, self.out_steps, self.out_channels,
            height, width).permute(0, 1, 2, 4, 5, 3)
        residual = residual - residual.mean(dim=1, keepdim=True)
        return members.detach() + residual


class ScalarPotentialResidual(nn.Module):
    """A zero-start shared u/v residual generated from one scalar potential.

    The mode changes only the fixed readout operator.  Consequently the
    gradient control and curl candidate have precisely the same learned body
    and scalar projection, which makes their checkpoints directly comparable.
    """

    def __init__(
            self, input_channels: int, hidden_channels: int,
            out_steps: int, blocks: int, groups: int, mode: str,
            limit: float):
        super().__init__()
        self.out_steps = int(out_steps)
        self.mode = mode
        self.limit = float(limit)
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            *(ResidualConvBlock(hidden_channels, hidden_channels, groups)
              for _ in range(blocks)),
        )
        self.output = nn.Conv2d(hidden_channels, out_steps, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _two_cell_support(value: torch.Tensor) -> torch.Tensor:
        """Keep psi zero on a fixed two-cell boundary support."""
        height, width = value.shape[-2:]
        support = value.new_zeros((1, 1, height, width))
        if height > 4 and width > 4:
            support[..., 2:-2, 2:-2] = 1.0
        return value * support

    @staticmethod
    def central_derivatives(
            value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tensor-product central Dx/Dy with a zero extension boundary."""
        if value.ndim != 4:
            raise ValueError("scalar potential must have shape (B,T,H,W)")
        extended = F.pad(value, (1, 1, 1, 1))
        dx = 0.5 * (extended[..., 1:-1, 2:] - extended[..., 1:-1, :-2])
        dy = 0.5 * (extended[..., 2:, 1:-1] - extended[..., :-2, 1:-1])
        return dx, dy

    @classmethod
    def correction_from_psi(
            cls, psi: torch.Tensor, mode: str) -> torch.Tensor:
        dx, dy = cls.central_derivatives(psi)
        if mode == "gradient":
            return torch.stack((dx, dy), dim=-1)
        if mode == "curl":
            return torch.stack((dy, -dx), dim=-1)
        raise ValueError("scalar potential mode must be gradient/curl")

    def forward(
            self, feature: torch.Tensor,
            size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.output(self.body(feature))
        if raw.shape[-2:] != size:
            raw = F.interpolate(raw, size=size, mode="bilinear",
                                align_corners=False)
        # Tanh bounds the learned potential before the fixed support and
        # derivative operator.  There is deliberately no post-curl mask.
        psi = self._two_cell_support(torch.tanh(raw))
        return psi, self.limit * self.correction_from_psi(psi, self.mode)


def _standardized_region_score(value: torch.Tensor) -> torch.Tensor:
    """Map a sample-local physical indicator to a smooth high-risk score."""
    if value.ndim != 3:
        raise ValueError("region indicator must have shape (B,H,W)")
    mean = value.mean(dim=(1, 2), keepdim=True)
    scale = value.sub(mean).square().mean(
        dim=(1, 2), keepdim=True).add(1.0e-6).sqrt()
    # z=0.8 is close to the hard top-quintile masks used by the audit, while
    # the finite temperature keeps useful gradients and avoids brittle class
    # boundaries under small Sim-to-Real changes.
    return torch.sigmoid((value - mean) / scale * 2.5 - 2.0)


def hard_region_conditioning(
        history: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-free soft flow-region maps and a conservative gate.

    The maps use only the supplied history: temporal variability, streamwise
    speed deficit, vorticity, strain, recent acceleration, and far-wake
    geometry.  They intentionally overlap.  The gate suppresses the inlet and
    stable field but never makes a hard binary decision, so a mild input
    perturbation cannot move a pixel discontinuously between experts.
    """
    if history.ndim != 5 or history.shape[1] < 3 or history.shape[-1] < 2:
        raise ValueError("hard-region history must be (B,T,H,W,C>=2)")
    uv = history[..., :2]
    last = uv[:, -1]
    variability = uv.std(dim=1, unbiased=False).square().sum(dim=-1).sqrt()
    acceleration = (
        last - 2.0 * uv[:, -2] + uv[:, -3]
    ).square().sum(dim=-1).add(1.0e-8).sqrt()

    # A robust upper streamwise reference makes the deficit insensitive to a
    # global gain while retaining the wake location. kthvalue avoids a CPU
    # synchronization and remains deterministic for the fixed grid.
    streamwise = last[..., 0]
    flat = streamwise.flatten(1)
    rank = max(1, int(round(0.90 * flat.shape[1])))
    reference = flat.kthvalue(rank, dim=1).values[:, None, None]
    deficit = (reference - streamwise).clamp_min(0.0)

    def dx(value: torch.Tensor) -> torch.Tensor:
        middle = 0.5 * (value[:, :, 2:] - value[:, :, :-2])
        return torch.cat((
            value[:, :, 1:2] - value[:, :, :1], middle,
            value[:, :, -1:] - value[:, :, -2:-1]), dim=2)

    def dy(value: torch.Tensor) -> torch.Tensor:
        middle = 0.5 * (value[:, 2:] - value[:, :-2])
        return torch.cat((
            value[:, 1:2] - value[:, :1], middle,
            value[:, -1:] - value[:, -2:-1]), dim=1)

    du_dx, dv_dx = dx(last[..., 0]), dx(last[..., 1])
    du_dy, dv_dy = dy(last[..., 0]), dy(last[..., 1])
    vorticity = (dv_dx - du_dy).abs()
    strain = torch.sqrt(
        (du_dx - dv_dy).square() + (du_dy + dv_dx).square() + 1.0e-8)

    batch, _steps, height, width, _channels = history.shape
    yy = torch.linspace(
        -1.0, 1.0, height, device=history.device, dtype=history.dtype)
    xx = torch.linspace(
        -1.0, 1.0, width, device=history.device, dtype=history.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    far_wake = (
        torch.sigmoid(8.0 * grid_x)
        * torch.sigmoid(8.0 * (0.65 - grid_y.abs()))
    ).unsqueeze(0).expand(batch, -1, -1)
    maps = torch.stack((
        _standardized_region_score(variability),
        _standardized_region_score(deficit),
        _standardized_region_score(vorticity),
        _standardized_region_score(strain),
        _standardized_region_score(acceleration),
        far_wake,
    ), dim=1)

    dynamic_union = maps[:, (0, 2, 3, 4)].amax(dim=1)
    # The audited hard errors were dominated by dynamic wake intersections.
    # Requiring both dynamics and either deficit/far-wake structure prevents
    # the expert from rewriting the broad stable field.
    wake_structure = 1.0 - (
        (1.0 - maps[:, 1]) * (1.0 - maps[:, 5]))
    inlet_keep = torch.sigmoid(10.0 * (grid_x + 0.72)).unsqueeze(0)
    gate = (dynamic_union * wake_structure * inlet_keep).clamp(0.0, 1.0)
    return maps, gate.unsqueeze(1)


class HardRegionResidualExpert(nn.Module):
    """Zero-initialized residual expert restricted to difficult wake pixels."""

    region_channels = 6

    def __init__(
            self, feature_channels: int, hidden_channels: int,
            out_steps: int, blocks: int, groups: int, limit: float,
            detach_context: bool, feature_noise_std: float):
        super().__init__()
        self.out_steps = int(out_steps)
        self.limit = float(limit)
        self.detach_context = bool(detach_context)
        self.feature_noise_std = float(feature_noise_std)
        self.body = nn.Sequential(
            nn.Conv2d(
                feature_channels + self.region_channels,
                hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            *(ResidualConvBlock(
                hidden_channels, hidden_channels, groups)
              for _ in range(blocks)),
        )
        self.output = nn.Conv2d(hidden_channels, out_steps * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
            self, feature: torch.Tensor, history: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        maps, gate = hard_region_conditioning(history.detach())
        if feature.shape[-2:] != maps.shape[-2:]:
            feature = F.interpolate(
                feature, size=maps.shape[-2:], mode="bilinear",
                align_corners=False)
        if self.detach_context:
            feature = feature.detach()
        if self.training and self.feature_noise_std > 0.0:
            noise = torch.randn_like(feature)
            noise = F.avg_pool2d(noise, 3, stride=1, padding=1)
            noise = noise / noise.square().mean(
                dim=(2, 3), keepdim=True).add(1.0e-6).sqrt()
            feature_scale = feature.detach().square().mean(
                dim=(2, 3), keepdim=True).add(1.0e-6).sqrt()
            feature = feature + (
                self.feature_noise_std * feature_scale * noise
                * gate)
        raw = self.output(self.body(torch.cat((feature, maps), dim=1)))
        batch, _channels, height, width = raw.shape
        residual = raw.reshape(
            batch, self.out_steps, 2, height, width).permute(0, 1, 3, 4, 2)
        residual = self.limit * torch.tanh(residual)
        return residual * gate.permute(0, 2, 3, 1).unsqueeze(1), gate


def conservative_history_features(history: torch.Tensor) -> torch.Tensor:
    """Build target-free local state features for a closure law.

    Unlike the hard-region mask above, these channels do not select pixels to
    rewrite.  They describe the latest state, temporal derivatives, spatial
    derivatives, Laplacians, audited flow-region scores, and coordinates.  A
    downstream closure must still express its correction as a flux divergence.
    """
    if history.ndim != 5 or history.shape[1] < 3 or history.shape[-1] < 2:
        raise ValueError("closure history must be (B,T,H,W,C>=2)")
    velocity = history[..., :2]
    last = velocity[:, -1]
    first_time = last - velocity[:, -2]
    second_time = last - 2.0 * velocity[:, -2] + velocity[:, -3]

    def dx(value: torch.Tensor) -> torch.Tensor:
        middle = 0.5 * (value[:, :, 2:] - value[:, :, :-2])
        return torch.cat((
            value[:, :, 1:2] - value[:, :, :1], middle,
            value[:, :, -1:] - value[:, :, -2:-1]), dim=2)

    def dy(value: torch.Tensor) -> torch.Tensor:
        middle = 0.5 * (value[:, 2:] - value[:, :-2])
        return torch.cat((
            value[:, 1:2] - value[:, :1], middle,
            value[:, -1:] - value[:, -2:-1]), dim=1)

    gradients = torch.stack((
        dx(last[..., 0]), dy(last[..., 0]),
        dx(last[..., 1]), dy(last[..., 1])), dim=-1)
    laplacian = torch.stack((
        dx(dx(last[..., 0])) + dy(dy(last[..., 0])),
        dx(dx(last[..., 1])) + dy(dy(last[..., 1]))), dim=-1)
    region_maps, _gate = hard_region_conditioning(history)

    batch, _steps, height, width, _channels = history.shape
    yy = torch.linspace(
        -1.0, 1.0, height, device=history.device, dtype=history.dtype)
    xx = torch.linspace(
        -1.0, 1.0, width, device=history.device, dtype=history.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    coordinates = torch.stack((grid_y, grid_x)).unsqueeze(0).expand(
        batch, -1, -1, -1)
    state = torch.cat((
        last.permute(0, 3, 1, 2),
        first_time.permute(0, 3, 1, 2),
        second_time.permute(0, 3, 1, 2),
        gradients.permute(0, 3, 1, 2),
        laplacian.permute(0, 3, 1, 2),
        region_maps,
        coordinates,
    ), dim=1)
    return state


class RegimeGatedConservativeClosure(nn.Module):
    """History-routed local experts whose correction is a flux divergence.

    Each expert predicts horizontal/vertical u/v fluxes plus learned inflow
    boundary fluxes.  The finite-volume difference telescopes exactly: the
    spatial sum of the correction is determined only by boundary flux.  A
    sample-level router uses the observed history, never target or metadata.
    """

    state_channels = 20

    def __init__(
            self, feature_channels: int, hidden_channels: int,
            out_steps: int, in_steps: int, experts: int, blocks: int,
            groups: int, flux_limit: float, router_temperature: float):
        super().__init__()
        self.out_steps = int(out_steps)
        self.experts = int(experts)
        self.flux_limit = float(flux_limit)
        self.router_temperature = float(router_temperature)
        self.body = nn.Sequential(
            nn.Conv2d(
                feature_channels + self.state_channels,
                hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            *(ResidualConvBlock(
                hidden_channels, hidden_channels, groups)
              for _ in range(blocks)),
        )
        self.flux_head = nn.Conv2d(
            hidden_channels, experts * out_steps * 4, 1)
        self.inflow_head = nn.Conv2d(
            hidden_channels, experts * out_steps * 4, 1)
        router_hidden = max(32, hidden_channels // 2)
        self.router = nn.Sequential(
            nn.Linear(in_steps * 4, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, experts),
        )
        # Exact identity at update zero comes from the two zero flux heads.
        # The router deliberately keeps its isolated random initialization:
        # sample-varying weights break expert symmetry from the first update,
        # while they still cannot change the zero correction.
        nn.init.zeros_(self.flux_head.weight)
        nn.init.zeros_(self.flux_head.bias)
        nn.init.zeros_(self.inflow_head.weight)
        nn.init.zeros_(self.inflow_head.bias)

    def route(self, history: torch.Tensor) -> torch.Tensor:
        descriptor = history_regime_summary(history.detach())
        return torch.softmax(
            self.router(descriptor) / self.router_temperature, dim=-1)

    @staticmethod
    def flux_divergence(
            flux: torch.Tensor, inflow: torch.Tensor) -> torch.Tensor:
        """Return -div(flux) for [B,T,4,H,W] tensors.

        Channels 0:2 are x-face u/v fluxes and 2:4 are y-face fluxes.  The
        matching inflow tensor supplies the left and top boundary faces; only
        its left/top edges are used.  Therefore summation over H,W telescopes.
        """
        if flux.ndim != 5 or flux.shape[2] != 4:
            raise ValueError("closure flux must have shape (B,T,4,H,W)")
        if inflow.shape != flux.shape:
            raise ValueError("closure inflow must match closure flux")
        flux_x, flux_y = flux[:, :, :2], flux[:, :, 2:]
        inlet_x = inflow[:, :, :2, :, :1]
        inlet_y = inflow[:, :, 2:, :1, :]
        previous_x = torch.cat((inlet_x, flux_x[..., :-1]), dim=-1)
        previous_y = torch.cat((inlet_y, flux_y[..., :-1, :]), dim=-2)
        return -(flux_x - previous_x + flux_y - previous_y)

    def forward(
            self, feature: torch.Tensor, history: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        state = conservative_history_features(history.detach())
        if feature.shape[-2:] != state.shape[-2:]:
            feature = F.interpolate(
                feature, size=state.shape[-2:], mode="bilinear",
                align_corners=False)
        hidden = self.body(torch.cat((feature, state), dim=1))
        batch, _channels, height, width = hidden.shape
        shape = (batch, self.experts, self.out_steps, 4, height, width)
        flux = self.flux_limit * torch.tanh(self.flux_head(hidden).reshape(
            shape))
        inflow = self.flux_limit * torch.tanh(
            self.inflow_head(hidden).reshape(shape))
        weights = self.route(history)
        weights_view = weights[:, :, None, None, None, None]
        mixed_flux = (flux * weights_view).sum(dim=1)
        mixed_inflow = (inflow * weights_view).sum(dim=1)
        residual = self.flux_divergence(mixed_flux, mixed_inflow)
        return residual.permute(0, 1, 3, 4, 2), weights


class PhysicalEnergyController(nn.Module):
    """Predict one target-free spatial log-energy correction per sample."""

    def __init__(self, input_channels: int, hidden_channels: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.body(feature)


class DecoderContextEnergyController(PhysicalEnergyController):
    """Retain the decoder controller and add zero-initialized context before GN."""
    def __init__(self, input_channels, hidden_channels, groups, recent_steps, mode):
        super().__init__(input_channels, hidden_channels, groups)
        if mode not in ("physical", "units"):
            raise ValueError("unknown decoder context mode")
        self.recent_steps = recent_steps
        self.mode = mode
        self.context_projection = nn.Conv2d(2 * recent_steps + 8, hidden_channels, 1, bias=False)
        nn.init.zeros_(self.context_projection.weight)

    def context_features(self, history, base_uv, input_std, target_std):
        # Input means are not needed: the common spatial/temporal offset is removed.
        velocity = history[..., :2] * input_std[:2].reshape(1, 1, 1, 1, 2)
        velocity = velocity - velocity.mean(dim=(1, 2, 3), keepdim=True)
        future = base_uv * target_std[:2].reshape(1, 1, 1, 1, 2)
        if self.mode == "units":
            scale = velocity.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
            velocity = velocity / scale
            future = future / scale
        recent = velocity[:, -(self.recent_steps + 1):]
        difference = recent[:, 1:] - recent[:, :-1]
        b, t, h, w, c = difference.shape
        local = difference.permute(0, 1, 4, 2, 3).reshape(b, t*c, h, w)
        mean = recent.mean(dim=1).permute(0, 3, 1, 2)
        spread = recent.std(dim=1, unbiased=False).permute(0, 3, 1, 2)
        return torch.cat((local, mean, spread, fluctuation_energy_context(future)), dim=1)

    def forward(self, feature, history, base_uv, input_std, target_std):
        context = self.context_features(history, base_uv, input_std, target_std)
        value = self.body[0](feature) + self.context_projection(context)
        for layer in self.body[1:]:
            value = layer(value)
        return value


class HorizonLocalPhysicalEnergyController(nn.Module):
    """Predict a separate spatial log-amplitude correction per horizon."""

    def __init__(
            self, input_channels: int, hidden_channels: int,
            out_steps: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_steps, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.body(feature)


class MultiScalePhysicalEnergyController(nn.Module):
    """Fuse every decoder scale into the deployed log-energy correction."""

    def __init__(
            self, input_channels: list[int], hidden_channels: int,
            groups: int):
        super().__init__()
        if not input_channels:
            raise ValueError("multiscale energy controller needs features")
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, hidden_channels, 1),
                _group_norm(groups, hidden_channels),
                nn.SiLU(),
            )
            for width in input_channels)
        # Keep the final projection at body.5 so all existing zero-init and
        # controller-gradient audits retain the same semantic key.
        self.body = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("multiscale energy feature contract changed")
        size = features[-1].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value, size=size, mode="bilinear", align_corners=False)
            projected.append(value)
        return self.body(torch.stack(projected).mean(dim=0))


class HistoryConditionedPhysicalEnergyController(nn.Module):
    """Predict energy correction from observed dynamics, not decoder state.

    The controller sees recent normalized u/v first differences plus their
    temporal mean and standard deviation.  Its input therefore stays attached
    to observations when a later Sim-to-Real stage adapts the point backbone.
    """

    def __init__(
            self, recent_steps: int, hidden_channels: int, groups: int):
        super().__init__()
        if recent_steps < 2:
            raise ValueError("history energy controller needs two recent steps")
        self.recent_steps = int(recent_steps)
        input_channels = 2 * recent_steps + 4
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if (history.ndim != 5 or history.shape[1] <= self.recent_steps
                or history.shape[-1] < 2):
            raise ValueError("history energy input must be [B,T,H,W,C>=2]")
        velocity = history[:, -(self.recent_steps + 1):, ..., :2]
        differences = velocity[:, 1:] - velocity[:, :-1]
        batch, steps, height, width, channels = differences.shape
        local = differences.permute(0, 1, 4, 2, 3).reshape(
            batch, steps * channels, height, width)
        mean = velocity.mean(dim=1).permute(0, 3, 1, 2)
        spread = velocity.std(dim=1, unbiased=False).permute(0, 3, 1, 2)
        return self.body(torch.cat((local, mean, spread), dim=1))


def fluctuation_energy_context(uv: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Per-channel fluctuation-energy context of a pre-controller prediction.

    ``uv`` has shape (..., T, H, W, 2).  Returns (..., 2 * C, H, W) holding
    ``log var`` and ``var / mean(var)`` per channel, all computed from the model's
    own prediction, so the context is available at inference time.
    """
    fluctuation = uv - uv.mean(dim=-4, keepdim=True)
    variance = fluctuation.square().mean(dim=-4).movedim(-1, -3)
    scale = variance.mean(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
    log_variance = torch.log(variance + epsilon).clamp(-12.0, 6.0)
    relative = (variance / scale).clamp(0.0, 50.0)
    return torch.cat((log_variance, relative), dim=-3)


class ContextConditionedEnergyHead(nn.Module):
    """Predict one log-energy correction per channel from dynamics plus context.

    Inputs
    ------
    history : (B, T, H, W, C>=2) normalised history; the last ``recent_steps``
        first differences plus their temporal mean and standard deviation are used.
    context : (B, 4, H, W) from :func:`fluctuation_energy_context` of the base
        prediction's u/v fluctuation.

    Output
    ------
    (B, 2, H, W) raw per-channel log corrections for u and v.  The last layer is
    zero initialised, so a fresh head is an exact identity.
    """

    def __init__(self, recent_steps: int, hidden_channels: int, groups: int):
        super().__init__()
        if recent_steps < 2:
            raise ValueError("context energy head needs two recent steps")
        self.recent_steps = int(recent_steps)
        input_channels = 2 * recent_steps + 4 + 4
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=2, dilation=2,
                groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, history: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if (history.ndim != 5 or history.shape[1] <= self.recent_steps
                or history.shape[-1] < 2):
            raise ValueError("history energy input must be [B,T,H,W,C>=2]")
        velocity = history[:, -(self.recent_steps + 1):, ..., :2]
        differences = velocity[:, 1:] - velocity[:, :-1]
        batch, steps, height, width, channels = differences.shape
        local = differences.permute(0, 1, 4, 2, 3).reshape(
            batch, steps * channels, height, width)
        mean = velocity.mean(dim=1).permute(0, 3, 1, 2)
        spread = velocity.std(dim=1, unbiased=False).permute(0, 3, 1, 2)
        return self.body(torch.cat((local, mean, spread, context), dim=1))


class HybridMultiScaleHistoryEnergyController(nn.Module):
    """Add decoder-multiscale and observation-history energy corrections."""

    def __init__(
            self, decoder_channels: list[int], recent_steps: int,
            hidden_channels: int, groups: int):
        super().__init__()
        decoder = MultiScalePhysicalEnergyController(
            decoder_channels, hidden_channels, groups)
        # Preserve top-level body.5/projections names used by all controller
        # gradient, update, checkpoint, and packaging audits.
        self.projections = decoder.projections
        self.body = decoder.body
        self.history = HistoryConditionedPhysicalEnergyController(
            recent_steps, hidden_channels, groups)

    def forward(
            self, features: list[torch.Tensor], history: torch.Tensor,
            ) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("hybrid energy feature contract changed")
        size = features[-1].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value, size=size, mode="bilinear", align_corners=False)
            projected.append(value)
        decoder_delta = self.body(torch.stack(projected).mean(dim=0))
        history_delta = self.history(history)
        if history_delta.shape[-2:] != size:
            history_delta = F.interpolate(
                history_delta, size=size, mode="bilinear",
                align_corners=False)
        return decoder_delta + history_delta


class PhaseConditionedLocalDelta(nn.Module):
    """Predict a shared high-resolution forecast delta from recent dynamics."""

    def __init__(
            self, in_steps: int, out_steps: int, hidden_channels: int,
            recent_steps: int, groups: int):
        super().__init__()
        self.out_steps = int(out_steps)
        self.recent_steps = int(recent_steps)
        # recent first differences plus their recent second differences
        input_channels = 2 * (2 * recent_steps - 1)
        self.local = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
        )
        self.condition = nn.Sequential(
            nn.Linear(in_steps * 4, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * hidden_channels),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self.output = nn.Conv2d(hidden_channels, out_steps * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        velocity = history[:, -(self.recent_steps + 1):, ..., :2]
        differences = velocity[:, 1:] - velocity[:, :-1]
        accelerations = differences[:, 1:] - differences[:, :-1]
        batch, _steps, height, width, _channels = differences.shape
        local = torch.cat((
            differences.reshape(batch, self.recent_steps, height, width, 2),
            accelerations.reshape(
                batch, self.recent_steps - 1, height, width, 2),
        ), dim=1).permute(0, 1, 4, 2, 3).reshape(
            batch, 2 * (2 * self.recent_steps - 1), height, width)
        feature = self.local(local)
        scale, shift = self.condition(
            history_regime_summary(history)).chunk(2, dim=-1)
        feature = (
            feature * (1.0 + 0.25 * torch.tanh(scale)[..., None, None])
            + 0.25 * shift[..., None, None])
        raw = self.output(feature).reshape(
            batch, self.out_steps, 2, height, width)
        return raw.permute(0, 1, 3, 4, 2)


class SpatioTemporalResidualBlock(nn.Module):
    """Residual 3-D convolution over forecast horizon and image space."""

    def __init__(self, channels: int, kernel: int, groups: int):
        super().__init__()
        padding = kernel // 2
        self.body = nn.Sequential(
            nn.Conv3d(
                channels, channels, kernel,
                padding=padding, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(
                channels, channels, kernel,
                padding=padding, bias=False),
            _group_norm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value))


class FactorizedTemporalPointBlock(nn.Module):
    """Cheap explicit forecast-time mixing followed by spatial mixing."""

    def __init__(self, channels: int, temporal_kernel: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(
                channels, channels, (temporal_kernel, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                groups=channels, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(
                channels, channels, (1, 3, 3), padding=(0, 1, 1),
                groups=channels, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 1, bias=False),
            _group_norm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value))


class TemporalPointDecoder(nn.Module):
    """Decode explicit horizon-local point trajectories at every U-Net scale.

    Legacy point heads flatten all forecast times into ordinary 2-D output
    channels.  This decoder keeps the time axis explicit, shares factorized
    temporal/spatial dynamics across ensemble members, and retains lightweight
    member-specific mean and fluctuation projections.  The direct variant
    zero-initializes its fluctuation outputs; the stable variant instead
    zero-initializes separate amplitude outputs.  Both make update zero exact
    persistence while the stable shape path starts non-degenerate.
    """

    def __init__(
            self, input_channels: list[int], hidden_channels: int,
            out_steps: int, out_channels: int, members: int,
            blocks: int, temporal_kernel: int, groups: int,
            amplitude_factorization: bool = False,
            amplitude_limit: float = 3.0):
        super().__init__()
        if not input_channels:
            raise ValueError("temporal point decoder needs decoder features")
        self.out_steps = int(out_steps)
        self.out_channels = int(out_channels)
        self.amplitude_factorization = bool(amplitude_factorization)
        self.amplitude_limit = float(amplitude_limit)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, hidden_channels, 1, bias=False),
                _group_norm(groups, hidden_channels),
                nn.SiLU(),
            )
            for width in input_channels)
        position = torch.arange(out_steps, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, hidden_channels, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0)) / hidden_channels))
        embedding = torch.zeros(out_steps, hidden_channels)
        embedding[:, 0::2] = torch.sin(position * frequency)
        embedding[:, 1::2] = torch.cos(
            position * frequency[:embedding[:, 1::2].shape[1]])
        self.horizon_embedding = nn.Parameter(
            embedding.transpose(0, 1).reshape(
                1, hidden_channels, out_steps, 1, 1))
        self.scale_blocks = nn.ModuleList(
            nn.Sequential(*(
                FactorizedTemporalPointBlock(
                    hidden_channels, temporal_kernel, groups)
                for _ in range(blocks)))
            for _ in input_channels)
        self.member_mean_heads = nn.ModuleList(
            nn.ModuleList(
                nn.Conv2d(width, out_channels, 1)
                for width in input_channels)
            for _ in range(members))
        self.member_fluctuation_heads = nn.ModuleList(
            nn.ModuleList(
                nn.Conv3d(hidden_channels, out_channels, 1)
                for _ in input_channels)
            for _ in range(members))
        self.member_amplitude_heads = (
            nn.ModuleList(
                nn.ModuleList(
                    nn.Conv2d(width, out_channels, 1)
                    for width in input_channels)
                for _ in range(members))
            if self.amplitude_factorization else None)
        zero_heads = list(self.member_mean_heads)
        if self.member_amplitude_heads is not None:
            zero_heads.extend(self.member_amplitude_heads)
        else:
            zero_heads.extend(self.member_fluctuation_heads)
        for member_heads in zero_heads:
            for head in member_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def temporal_features(
            self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError("temporal point feature contract changed")
        return [
            blocks(
                projection(feature).unsqueeze(2)
                + self.horizon_embedding)
            for feature, projection, blocks in zip(
                features, self.projections, self.scale_blocks)
        ]

    def member_prediction(
            self, persistence: torch.Tensor, features: list[torch.Tensor],
            temporal_features: list[torch.Tensor], member: int,
            ) -> torch.Tensor:
        if (len(features) != len(self.projections)
                or len(temporal_features) != len(self.projections)):
            raise ValueError("temporal point scale contract changed")
        batch, height, width, channels = persistence.shape
        if channels != self.out_channels:
            raise ValueError("temporal point persistence channels changed")
        means = []
        fluctuations = []
        for feature, temporal, mean_head, fluctuation_head in zip(
                features, temporal_features,
                self.member_mean_heads[member],
                self.member_fluctuation_heads[member]):
            mean = mean_head(feature)
            if mean.shape[-2:] != (height, width):
                mean = F.interpolate(
                    mean, size=(height, width), mode="bilinear",
                    align_corners=False)
            means.append(mean)
            fluctuation = fluctuation_head(temporal)
            if fluctuation.shape[-2:] != (height, width):
                fluctuation = F.interpolate(
                    fluctuation,
                    size=(self.out_steps, height, width),
                    mode="trilinear", align_corners=False)
            if self.member_amplitude_heads is not None:
                fluctuation = fluctuation - fluctuation.mean(
                    dim=2, keepdim=True)
                rms = fluctuation.float().square().mean(
                    dim=2, keepdim=True).add(1e-6).sqrt().to(fluctuation)
                temporal_shape = fluctuation / rms
                amplitude = self.member_amplitude_heads[member][
                    len(fluctuations)](feature)
                if amplitude.shape[-2:] != (height, width):
                    amplitude = F.interpolate(
                        amplitude, size=(height, width), mode="bilinear",
                        align_corners=False)
                amplitude = self.amplitude_limit * torch.tanh(
                    amplitude / self.amplitude_limit)
                fluctuation = temporal_shape * amplitude.unsqueeze(2)
            fluctuations.append(fluctuation)
        mean_delta = torch.stack(means).mean(dim=0).permute(
            0, 2, 3, 1).unsqueeze(1)
        fluctuation = torch.stack(fluctuations).mean(dim=0)
        fluctuation = fluctuation - fluctuation.mean(dim=2, keepdim=True)
        fluctuation = fluctuation.permute(0, 2, 3, 4, 1)
        residual = mean_delta + fluctuation
        return persistence.unsqueeze(1) + residual


class LatentODEVectorField(nn.Module):
    """Autonomous convolutional vector field conditioned on encoded history."""

    def __init__(
            self, channels: int, groups: int, derivative_limit: float):
        super().__init__()
        self.derivative_limit = float(derivative_limit)
        self.body = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels, channels, 3, padding=1,
                groups=channels, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(
            self, state: torch.Tensor,
            context: torch.Tensor) -> torch.Tensor:
        if state.shape != context.shape:
            raise ValueError("latent ODE state/context shapes changed")
        raw = self.body(torch.cat((state, context), dim=1))
        return self.derivative_limit * torch.tanh(
            raw / self.derivative_limit)


class LatentODEPointDecoder(nn.Module):
    """Encode one latent state, integrate it, then spatially decode all times.

    Unlike the optional recurrent residual branch, this module is the only
    point-forecast path when enabled: no flattened 20-horizon head can bypass
    the shared dynamics.  A fixed explicit-midpoint solver keeps packaging
    dependency-free and deterministic.  The latent solve happens at the
    coarsest decoder scale; static U-Net skips restore fine spatial structure
    after all horizon states have been stacked into the batch dimension.
    """

    def __init__(
            self, input_channels: list[int], hidden_channels: int,
            out_steps: int, out_channels: int, members: int,
            solver_substeps: int, step_size: float,
            derivative_limit: float, residual_limit: float, groups: int):
        super().__init__()
        if not input_channels:
            raise ValueError("latent ODE decoder needs decoder features")
        self.out_steps = int(out_steps)
        self.out_channels = int(out_channels)
        self.solver_substeps = int(solver_substeps)
        self.step_size = float(step_size)
        self.residual_limit = float(residual_limit)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels[0], hidden_channels, 1, bias=False),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            ResidualConvBlock(hidden_channels, hidden_channels, groups),
        )
        self.vector_field = LatentODEVectorField(
            hidden_channels, groups, derivative_limit)

        decoder_widths = []
        for level in range(len(input_channels) - 1):
            raw_width = max(groups, hidden_channels // (2 ** (level + 1)))
            width = max(groups, (raw_width // groups) * groups)
            decoder_widths.append(width)
        self.upsample_projections = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        current = hidden_channels
        for skip_channels, width in zip(input_channels[1:], decoder_widths):
            self.upsample_projections.append(
                nn.Conv2d(current, width, 1, bias=False))
            self.skip_projections.append(
                nn.Conv2d(skip_channels, width, 1, bias=False))
            self.spatial_blocks.append(
                ResidualConvBlock(width, width, groups))
            current = width
        self.member_heads = nn.ModuleList(
            nn.Conv2d(current, out_channels, 1)
            for _ in range(members))
        for head in self.member_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def latent_trajectory(
            self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.skip_projections) + 1:
            raise ValueError("latent ODE feature-scale contract changed")
        context = self.encoder(features[0])
        state = context
        states = []
        dt = self.step_size / self.solver_substeps
        for _horizon in range(self.out_steps):
            for _substep in range(self.solver_substeps):
                first = self.vector_field(state, context)
                midpoint = state + 0.5 * dt * first
                second = self.vector_field(midpoint, context)
                state = state + dt * second
            states.append(state)
        return torch.stack(states, dim=1)

    def spatial_decode(
            self, trajectory: torch.Tensor,
            features: list[torch.Tensor]) -> torch.Tensor:
        if (trajectory.ndim != 5
                or trajectory.shape[1] != self.out_steps):
            raise ValueError("latent ODE trajectory shape changed")
        batch, steps, channels, height, width = trajectory.shape
        value = trajectory.reshape(batch * steps, channels, height, width)
        for skip, up, project_skip, block in zip(
                features[1:], self.upsample_projections,
                self.skip_projections, self.spatial_blocks):
            target_size = skip.shape[-2:]
            value = F.interpolate(
                value, size=target_size, mode="bilinear",
                align_corners=False)
            value = up(value)
            static = project_skip(skip).unsqueeze(1).expand(
                batch, steps, -1, -1, -1).reshape(
                    batch * steps, -1, *target_size)
            value = block(value + static)
        return value.reshape(
            batch, steps, value.shape[1], *value.shape[-2:])

    def member_prediction(
            self, persistence: torch.Tensor,
            decoded: torch.Tensor, member: int) -> torch.Tensor:
        batch, steps, channels, height, width = decoded.shape
        if (steps != self.out_steps
                or persistence.shape != (
                    batch, height, width, self.out_channels)):
            raise ValueError("latent ODE persistence/output contract changed")
        raw = self.member_heads[member](decoded.reshape(
            batch * steps, channels, height, width))
        residual = raw.reshape(
            batch, steps, self.out_channels, height, width).permute(
                0, 1, 3, 4, 2)
        residual = self.residual_limit * torch.tanh(
            residual / self.residual_limit)
        return persistence.unsqueeze(1) + residual


class LatentUDEVectorField(nn.Module):
    """Shared resolved drift plus centered member-conditioned closure.

    The member coordinate is a deterministic quadrature coordinate rather
    than runtime noise.  Centering the learned closure at every derivative
    evaluation keeps it from directly translating the ensemble mean while
    still allowing unresolved dynamics to separate member trajectories.
    """

    def __init__(
            self, channels: int, members: int, out_steps: int, groups: int,
            derivative_limit: float, closure_limit: float):
        super().__init__()
        self.channels = int(channels)
        self.members = int(members)
        self.out_steps = int(out_steps)
        self.closure_limit = float(closure_limit)
        self.drift = LatentODEVectorField(
            channels, groups, derivative_limit)
        self.closure = nn.Sequential(
            nn.Conv2d(2 * channels + 2, channels, 1, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels, channels, 3, padding=1,
                groups=channels, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.register_buffer(
            "member_coordinate",
            torch.linspace(-1.0, 1.0, members).reshape(1, members, 1, 1, 1),
            persistent=True)

    def derivative(
            self, state: torch.Tensor, context: torch.Tensor,
            horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (state.ndim != 5 or state.shape[1] != self.members
                or context.shape != (
                    state.shape[0], self.channels,
                    state.shape[-2], state.shape[-1])):
            raise ValueError("latent UDE state/context shapes changed")
        if not 0 <= horizon < self.out_steps:
            raise ValueError("latent UDE horizon is out of range")
        batch, members, channels, height, width = state.shape
        flat_state = state.reshape(batch * members, channels, height, width)
        expanded_context = context.unsqueeze(1).expand(
            -1, members, -1, -1, -1)
        flat_context = expanded_context.reshape(
            batch * members, channels, height, width)
        drift = self.drift(flat_state, flat_context).reshape_as(state)

        time_value = float(horizon + 1) / self.out_steps
        time = state.new_full((batch, members, 1, height, width), time_value)
        member = self.member_coordinate.to(state).expand(
            batch, -1, -1, height, width)
        condition = torch.cat((state, expanded_context, time, member), dim=2)
        raw_closure = self.closure(condition.reshape(
            batch * members, 2 * channels + 2, height, width))
        closure = self.closure_limit * torch.tanh(
            raw_closure.reshape_as(state) / self.closure_limit)
        closure = closure - closure.mean(dim=1, keepdim=True)
        return drift + closure, closure

    def forward(
            self, state: torch.Tensor, context: torch.Tensor,
            horizon: int) -> torch.Tensor:
        return self.derivative(state, context, horizon)[0]


class LatentUDEBandPointDecoder(nn.Module):
    """Coarse latent UDE plus exclusive direct middle/fine spatial bands."""

    def __init__(
            self, input_channels: list[int], hidden_channels: int,
            out_steps: int, out_channels: int, members: int,
            solver_substeps: int, step_size: float,
            derivative_limit: float, closure_limit: float,
            residual_limit: float, groups: int, mean_fluctuation: bool):
        super().__init__()
        if len(input_channels) < 2:
            raise ValueError("latent UDE band decoder needs multiple scales")
        self.out_steps = int(out_steps)
        self.out_channels = int(out_channels)
        self.members = int(members)
        self.solver_substeps = int(solver_substeps)
        self.step_size = float(step_size)
        self.residual_limit = float(residual_limit)
        self.mean_fluctuation = bool(mean_fluctuation)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels[0], hidden_channels, 1, bias=False),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            ResidualConvBlock(hidden_channels, hidden_channels, groups),
        )
        self.vector_field = LatentUDEVectorField(
            hidden_channels, members, out_steps, groups,
            derivative_limit, closure_limit)
        self.member_coarse_heads = nn.ModuleList(
            nn.Conv2d(hidden_channels, out_channels, 1)
            for _ in range(members))
        point_channels = out_channels * (
            out_steps + 1 if mean_fluctuation else out_steps)
        self.member_band_heads = nn.ModuleList(
            nn.ModuleList(
                nn.Conv2d(channels, point_channels, 1)
                for channels in input_channels[1:])
            for _ in range(members))
        for head in self.member_coarse_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for heads in self.member_band_heads:
            for head in heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def latent_trajectory(
            self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.member_band_heads[0]) + 1:
            raise ValueError("latent UDE feature-scale contract changed")
        context = self.encoder(features[0])
        state = context.unsqueeze(1).expand(
            -1, self.members, -1, -1, -1).clone()
        states = []
        dt = self.step_size / self.solver_substeps
        for horizon in range(self.out_steps):
            for _substep in range(self.solver_substeps):
                first = self.vector_field(state, context, horizon)
                midpoint = state + 0.5 * dt * first
                second = self.vector_field(midpoint, context, horizon)
                state = state + dt * second
            states.append(state)
        return torch.stack(states, dim=2)

    def _raw_direct_residual(self, raw: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = raw.shape
        if self.mean_fluctuation:
            mean = raw[:, :self.out_channels].unsqueeze(1)
            fluctuation = raw[:, self.out_channels:].reshape(
                batch, self.out_steps, self.out_channels, height, width)
            fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)
            return mean + fluctuation
        return raw.reshape(
            batch, self.out_steps, self.out_channels, height, width)

    @staticmethod
    def _spatial_highpass(
            value: torch.Tensor,
            coarse_size: tuple[int, int]) -> torch.Tensor:
        batch, steps, channels, height, width = value.shape
        flat = value.reshape(batch * steps, channels, height, width)
        low = F.interpolate(flat, size=coarse_size, mode="area")
        low = F.interpolate(
            low, size=(height, width), mode="bilinear",
            align_corners=False)
        high = flat - low
        # Bilinear upsampling is not an exact right-inverse of area pooling.
        # Remove the remaining pooled leakage with a piecewise-constant lift
        # so the native middle/fine band has exactly zero coarse cell mean.
        leakage = F.interpolate(high, size=coarse_size, mode="area")
        leakage = F.interpolate(leakage, size=(height, width), mode="nearest")
        return (high - leakage).reshape_as(value)

    @staticmethod
    def _resize_trajectory(
            value: torch.Tensor,
            size: tuple[int, int]) -> torch.Tensor:
        if value.shape[-2:] == size:
            return value
        batch, steps, channels, height, width = value.shape
        resized = F.interpolate(
            value.reshape(batch * steps, channels, height, width),
            size=size, mode="bilinear", align_corners=False)
        return resized.reshape(batch, steps, channels, *size)

    def coarse_member_residual(
            self, trajectory: torch.Tensor, member: int) -> torch.Tensor:
        if (trajectory.ndim != 6 or trajectory.shape[1] != self.members
                or trajectory.shape[2] != self.out_steps):
            raise ValueError("latent UDE trajectory shape changed")
        value = trajectory[:, member]
        batch, steps, channels, height, width = value.shape
        raw = self.member_coarse_heads[member](value.reshape(
            batch * steps, channels, height, width))
        residual = raw.reshape(
            batch, steps, self.out_channels, height, width)
        return self.residual_limit * torch.tanh(
            residual / self.residual_limit)

    def member_prediction(
            self, persistence: torch.Tensor, trajectory: torch.Tensor,
            features: list[torch.Tensor], member: int) -> torch.Tensor:
        if len(features) != len(self.member_band_heads[member]) + 1:
            raise ValueError("latent UDE member band contract changed")
        target_size = persistence.shape[1:3]
        values = [self.coarse_member_residual(trajectory, member)]
        for index, (feature, head) in enumerate(zip(
                features[1:], self.member_band_heads[member]), start=1):
            direct = self._raw_direct_residual(head(feature))
            values.append(self._spatial_highpass(
                direct, features[index - 1].shape[-2:]))
        residual = torch.stack([
            self._resize_trajectory(value, target_size)
            for value in values], dim=0).mean(dim=0)
        residual = residual.permute(0, 1, 3, 4, 2)
        return persistence.unsqueeze(1) + residual


class LatentUDECoarseResidual(nn.Module):
    """Low-frequency dynamical correction for the strong direct decoder.

    The legacy point heads retain responsibility for the full forecast,
    including its resolved temporal mean and all middle/fine spatial bands.
    This branch can only add a coarse latent-UDE correction whose temporal
    mean at every pixel and spatial mean at every horizon are exactly zero.
    The fixed projection prevents a learned router from reallocating those
    responsibilities.  Member-specific corrections flow into the native UQ
    tower through the corrected ensemble trajectories.
    """

    def __init__(
            self, input_channels: int, hidden_channels: int,
            out_steps: int, out_channels: int, members: int,
            solver_substeps: int, step_size: float,
            derivative_limit: float, closure_limit: float,
            residual_limit: float, groups: int,
            residual_channels: int | None = None,
            zero_spatial_mean: bool = True):
        super().__init__()
        self.out_steps = int(out_steps)
        self.out_channels = int(out_channels)
        self.residual_channels = int(
            out_channels if residual_channels is None else residual_channels)
        if not 0 < self.residual_channels <= self.out_channels:
            raise ValueError("coarse UDE residual channels are invalid")
        self.members = int(members)
        self.solver_substeps = int(solver_substeps)
        self.step_size = float(step_size)
        self.residual_limit = float(residual_limit)
        self.zero_spatial_mean = bool(zero_spatial_mean)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            ResidualConvBlock(hidden_channels, hidden_channels, groups),
        )
        self.vector_field = LatentUDEVectorField(
            hidden_channels, members, out_steps, groups,
            derivative_limit, closure_limit)
        self.member_heads = nn.ModuleList(
            nn.Conv2d(hidden_channels, self.residual_channels, 1)
            for _ in range(members))
        for head in self.member_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def latent_trajectory(
            self, coarse_feature: torch.Tensor) -> torch.Tensor:
        context = self.encoder(coarse_feature)
        state = context.unsqueeze(1).expand(
            -1, self.members, -1, -1, -1).clone()
        states = []
        dt = self.step_size / self.solver_substeps
        for horizon in range(self.out_steps):
            for _substep in range(self.solver_substeps):
                first = self.vector_field(state, context, horizon)
                midpoint = state + 0.5 * dt * first
                second = self.vector_field(midpoint, context, horizon)
                state = state + dt * second
            states.append(state)
        return torch.stack(states, dim=2)

    def _fixed_responsibility_projection(
            self, correction: torch.Tensor) -> torch.Tensor:
        if correction.ndim != 6:
            raise ValueError("coarse UDE correction must be [B,M,T,H,W,C]")
        # These two orthogonal centering projections commute.  Applying both
        # makes the direct decoder the sole owner of the per-pixel temporal
        # mean and of every horizon/channel's spatial bulk mean.
        correction = correction - correction.mean(dim=2, keepdim=True)
        if self.zero_spatial_mean:
            correction = correction - correction.mean(
                dim=(3, 4), keepdim=True)
        return correction

    def forward(
            self, coarse_feature: torch.Tensor,
            target_size: tuple[int, int]) -> torch.Tensor:
        trajectory = self.latent_trajectory(coarse_feature)
        batch, members, steps, channels, height, width = trajectory.shape
        raw = []
        for member, head in enumerate(self.member_heads):
            value = head(trajectory[:, member].reshape(
                batch * steps, channels, height, width))
            raw.append(value.reshape(
                batch, steps, self.residual_channels, height, width))
        correction = torch.stack(raw, dim=1)
        correction = self.residual_limit * torch.tanh(
            correction / self.residual_limit)
        correction = F.interpolate(
            correction.reshape(
                batch * members * steps, self.residual_channels,
                height, width),
            size=target_size, mode="bilinear", align_corners=False)
        correction = correction.reshape(
            batch, members, steps, self.residual_channels,
            *target_size).permute(
                0, 1, 2, 4, 5, 3)
        if self.residual_channels < self.out_channels:
            correction = F.pad(
                correction, (0, self.out_channels - self.residual_channels))
        return self._fixed_responsibility_projection(correction)


class IsoenergeticTemporalTransport(nn.Module):
    """Redistribute a point forecast in time without changing its energy.

    The strong flattened point heads remain the primary predictor.  This
    module predicts a small, spatially varying displacement on the explicit
    forecast-time axis, linearly samples the existing u/v trajectory, and
    restores its original temporal mean and RMS at every pixel/channel.  Its
    output projection is exactly zero at construction, so the untrained
    transport is a bitwise identity while the straight-through branch keeps
    the displacement head trainable.
    """

    def __init__(
            self, feature_channels: int, hidden_channels: int,
            out_steps: int, blocks: int, temporal_kernel: int, groups: int,
            max_displacement: float):
        super().__init__()
        self.out_steps = int(out_steps)
        self.max_displacement = float(max_displacement)
        self.feature_projection = nn.Conv2d(
            feature_channels, hidden_channels, 1)
        self.trajectory_projection = nn.Conv3d(2, hidden_channels, 1)
        position = torch.arange(out_steps, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, hidden_channels, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0)) / hidden_channels))
        embedding = torch.zeros(out_steps, hidden_channels)
        embedding[:, 0::2] = torch.sin(position * frequency)
        embedding[:, 1::2] = torch.cos(
            position * frequency[:embedding[:, 1::2].shape[1]])
        self.horizon_embedding = nn.Parameter(
            embedding.transpose(0, 1).reshape(
                1, hidden_channels, out_steps, 1, 1))
        self.blocks = nn.Sequential(*(
            FactorizedTemporalPointBlock(
                hidden_channels, temporal_kernel, groups)
            for _ in range(blocks)))
        self.output = nn.Conv3d(hidden_channels, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _linear_time_warp(
            fluctuation: torch.Tensor,
            displacement: torch.Tensor) -> torch.Tensor:
        """Sample ``fluctuation`` at t + displacement with edge clamping."""
        if fluctuation.shape != displacement.shape:
            raise ValueError("temporal transport tensors must be aligned")
        if fluctuation.ndim != 5 or fluctuation.shape[-1] != 2:
            raise ValueError("temporal transport requires [B,T,H,W,2]")
        steps = fluctuation.shape[1]
        values = fluctuation.permute(0, 2, 3, 4, 1)
        delta = displacement.permute(0, 2, 3, 4, 1)
        timeline = torch.arange(
            steps, dtype=delta.dtype, device=delta.device).view(
                1, 1, 1, 1, steps)
        coordinate = (timeline + delta).clamp(0.0, float(steps - 1))
        lower = coordinate.floor()
        upper = (lower + 1.0).clamp_max(float(steps - 1))
        weight = coordinate - lower
        low = torch.gather(values, -1, lower.long())
        high = torch.gather(values, -1, upper.long())
        warped = low + (high - low) * weight
        return warped.permute(0, 4, 1, 2, 3)

    @staticmethod
    def _restore_mean_and_rms(
            warped: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Restore source temporal mean and RMS in float32."""
        source32 = source.float()
        warped32 = warped.float()
        source_mean = source32.mean(dim=1, keepdim=True)
        source_fluctuation = source32 - source_mean
        candidate = warped32 - warped32.mean(dim=1, keepdim=True)
        source_energy = source_fluctuation.square().mean(dim=1, keepdim=True)
        candidate_energy = candidate.square().mean(dim=1, keepdim=True)
        # Zero fluctuation is the exact update-0 state of the legacy heads.
        # Mask both operands *before* division: wrapping a computed 0/0-ish
        # ratio in torch.where still leaves a non-finite inactive derivative.
        epsilon = 1e-12
        active = (
            (source_energy > epsilon) & (candidate_energy > epsilon))
        safe_source = torch.where(
            active, source_energy, torch.ones_like(source_energy))
        safe_candidate = torch.where(
            active, candidate_energy, torch.ones_like(candidate_energy))
        scale = torch.sqrt(safe_source / safe_candidate)
        restored_fluctuation = torch.where(
            active, candidate * scale, source_fluctuation)
        return (source_mean + restored_fluctuation).to(source)

    def forward(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor,
            history: torch.Tensor | None = None) -> torch.Tensor:
        if (prediction.ndim != 5 or prediction.shape[1] != self.out_steps
                or prediction.shape[-1] < 2):
            raise ValueError(
                "temporal transport prediction contract changed")
        uv = prediction[..., :2]
        fluctuation = uv - uv.mean(dim=1, keepdim=True)
        context = fluctuation.detach().permute(0, 4, 1, 2, 3)
        feature = self.feature_projection(finest_feature)
        if feature.shape[-2:] != prediction.shape[2:4]:
            feature = F.interpolate(
                feature, size=prediction.shape[2:4], mode="bilinear",
                align_corners=False)
        value = (
            feature.unsqueeze(2)
            + self.trajectory_projection(context)
            + self.horizon_embedding)
        raw = self.output(self.blocks(F.silu(value))).permute(0, 2, 3, 4, 1)
        displacement = self.max_displacement * torch.tanh(raw)
        warped = self._linear_time_warp(fluctuation, displacement)
        candidate_uv = self._restore_mean_and_rms(warped, uv)
        delta = candidate_uv - uv
        bypass = raw.eq(0).flatten(1).all(dim=1).view(-1, 1, 1, 1, 1)
        transported_uv = _ExactIdentityAddSTE.apply(uv, delta, bypass)
        if prediction.shape[-1] == 2:
            return transported_uv
        return torch.cat((transported_uv, prediction[..., 2:]), dim=-1)


class HistoryConditionedTemporalBandMixer(nn.Module):
    """Modulate existing temporal fluctuation bands without replacing them.

    The direct flattened point heads remain responsible for the complete
    forecast.  This branch predicts bounded complex gains for a small bank of
    non-DC temporal Fourier bands.  The DC coefficient is never touched and
    the reconstructed u/v trajectory is explicitly recentered, so temporal
    means are preserved.  Unlike the scalar physical-energy controller, each
    band can change amplitude independently; unlike isoenergetic transport,
    it may adjust total fluctuation energy when the observed dynamics support
    that change.

    The controller reads both the finest decoder feature and local statistics
    from the most recent six observed velocity frames.  Its final projection
    is zero initialized, making update zero an exact identity through the
    straight-through identity branch.
    """

    def __init__(
            self, feature_channels: int, hidden_channels: int,
            out_steps: int, bands: int, groups: int,
            log_gain_limit: float, phase_limit: float):
        super().__init__()
        if out_steps < 4 or bands < 2 or bands > out_steps // 2:
            raise ValueError("temporal band dimensions are invalid")
        self.out_steps = int(out_steps)
        self.bands = int(bands)
        self.log_gain_limit = float(log_gain_limit)
        self.phase_limit = float(phase_limit)
        self.feature_projection = nn.Conv2d(
            feature_channels, hidden_channels, 1)
        # Five recent u/v differences plus local u/v mean and spread.
        self.history_projection = nn.Conv2d(14, hidden_channels, 1)
        self.body = nn.Sequential(
            ResidualConvBlock(hidden_channels, hidden_channels, groups),
            ResidualConvBlock(hidden_channels, hidden_channels, groups),
        )
        self.output = nn.Conv2d(hidden_channels, bands * 2 * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        frequencies = out_steps // 2 + 1
        assignment = torch.zeros(frequencies, bands)
        for frequency in range(1, frequencies):
            band = min(
                bands - 1,
                (frequency - 1) * bands // (frequencies - 1))
            assignment[frequency, band] = 1.0
        self.register_buffer("frequency_assignment", assignment)

    @staticmethod
    def history_features(history: torch.Tensor) -> torch.Tensor:
        if (history.ndim != 5 or history.shape[1] < 6
                or history.shape[-1] < 2):
            raise ValueError(
                "temporal band mixer requires history [B,T>=6,H,W,C>=2]")
        velocity = history[:, -6:, ..., :2]
        difference = velocity[:, 1:] - velocity[:, :-1]
        mean = velocity.mean(dim=1)
        spread = velocity.std(dim=1, unbiased=False)
        return torch.cat((
            difference.permute(0, 1, 4, 2, 3).flatten(1, 2),
            mean.permute(0, 3, 1, 2),
            spread.permute(0, 3, 1, 2),
        ), dim=1)

    def forward(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor,
            history: torch.Tensor | None = None) -> torch.Tensor:
        if (prediction.ndim != 5 or prediction.shape[1] != self.out_steps
                or prediction.shape[-1] < 2):
            raise ValueError("temporal band prediction contract changed")
        if history is None:
            raise ValueError("temporal band mixing requires observed history")
        target_size = prediction.shape[2:4]
        feature = self.feature_projection(finest_feature)
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(
                feature, size=target_size, mode="bilinear",
                align_corners=False)
        observed = self.history_projection(self.history_features(history))
        if observed.shape[-2:] != target_size:
            observed = F.interpolate(
                observed, size=target_size, mode="bilinear",
                align_corners=False)
        raw = self.output(self.body(F.silu(feature + observed)))
        batch, _channels, height, width = raw.shape
        raw = raw.reshape(
            batch, self.bands, 2, 2, height, width
        ).permute(0, 1, 4, 5, 2, 3)
        frequency_raw = torch.einsum(
            "fk,bkhwcp->bfhwcp",
            self.frequency_assignment.to(raw), raw)
        log_gain = self.log_gain_limit * torch.tanh(frequency_raw[..., 0])
        phase = self.phase_limit * torch.tanh(frequency_raw[..., 1])
        multiplier = torch.polar(torch.exp(log_gain.float()), phase.float())

        uv = prediction[..., :2]
        uv32 = uv.float()
        uv_mean = uv32.mean(dim=1, keepdim=True)
        fluctuation = uv32 - uv_mean
        spectrum = torch.fft.rfft(fluctuation, dim=1)
        mixed = torch.fft.irfft(
            spectrum * multiplier, n=self.out_steps, dim=1)
        mixed = mixed - mixed.mean(dim=1, keepdim=True)
        candidate_uv = (uv_mean + mixed).to(uv)
        delta = candidate_uv - uv
        bypass = raw.eq(0).flatten(1).all(dim=1).view(-1, 1, 1, 1, 1)
        mixed_uv = _ExactIdentityAddSTE.apply(uv, delta, bypass)
        if prediction.shape[-1] == 2:
            return mixed_uv
        return torch.cat((mixed_uv, prediction[..., 2:]), dim=-1)


class HistoryConditionedSpatialTransport(nn.Module):
    """Apply a bounded per-horizon spatial phase correction to direct u/v.

    The direct point heads still predict every value.  This branch reads the
    finest decoder field and recent observed dynamics, then predicts a small
    two-dimensional sampling displacement for each future horizon.  Warped
    fields are recentered to the source spatial mean at every horizon/channel;
    pressure bypasses the branch.  A zero output projection is exact identity
    through the same straight-through bypass used by other residual arms.
    """

    def __init__(
            self, feature_channels: int, hidden_channels: int,
            out_steps: int, groups: int, max_displacement: float):
        super().__init__()
        if out_steps < 1 or max_displacement <= 0.0:
            raise ValueError("spatial transport dimensions are invalid")
        self.out_steps = int(out_steps)
        self.max_displacement = float(max_displacement)
        self.feature_projection = nn.Conv2d(
            feature_channels, hidden_channels, 1)
        self.history_projection = nn.Conv2d(14, hidden_channels, 1)
        self.body = nn.Sequential(
            ResidualConvBlock(hidden_channels, hidden_channels, groups),
            ResidualConvBlock(hidden_channels, hidden_channels, groups),
        )
        self.output = nn.Conv2d(hidden_channels, out_steps * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _warp(uv: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        if uv.ndim != 5 or uv.shape[-1] != 2:
            raise ValueError("spatial transport requires u/v [B,T,H,W,2]")
        if displacement.shape != uv.shape:
            raise ValueError("spatial displacement must align with u/v")
        batch, steps, height, width, _channels = uv.shape
        dtype = displacement.dtype
        y = torch.linspace(-1.0, 1.0, height, device=uv.device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=uv.device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).view(
            1, 1, height, width, 2)
        scale = displacement.new_tensor((
            2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)))
        sample_grid = grid + displacement * scale
        values = uv.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, 2, height, width)
        sample_grid = sample_grid.expand(
            batch, steps, height, width, 2).reshape(
                batch * steps, height, width, 2)
        warped = F.grid_sample(
            values.float(), sample_grid.float(), mode="bilinear",
            padding_mode="border", align_corners=True)
        return warped.reshape(
            batch, steps, 2, height, width).permute(0, 1, 3, 4, 2)

    def forward(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor,
            history: torch.Tensor | None = None) -> torch.Tensor:
        if (prediction.ndim != 5 or prediction.shape[1] != self.out_steps
                or prediction.shape[-1] < 2):
            raise ValueError("spatial transport prediction contract changed")
        if history is None:
            raise ValueError("spatial transport requires observed history")
        target_size = prediction.shape[2:4]
        feature = self.feature_projection(finest_feature)
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(
                feature, size=target_size, mode="bilinear",
                align_corners=False)
        observed = self.history_projection(
            HistoryConditionedTemporalBandMixer.history_features(history))
        if observed.shape[-2:] != target_size:
            observed = F.interpolate(
                observed, size=target_size, mode="bilinear",
                align_corners=False)
        raw = self.output(self.body(F.silu(feature + observed)))
        batch, _channels, height, width = raw.shape
        raw = raw.reshape(
            batch, self.out_steps, 2, height, width
        ).permute(0, 1, 3, 4, 2)
        displacement = self.max_displacement * torch.tanh(raw)
        uv = prediction[..., :2]
        candidate_uv = self._warp(uv, displacement)
        source_mean = uv.float().mean(dim=(2, 3), keepdim=True)
        candidate_uv = (
            candidate_uv - candidate_uv.mean(dim=(2, 3), keepdim=True)
            + source_mean).to(uv)
        delta = candidate_uv - uv
        bypass = raw.eq(0).flatten(1).all(dim=1).view(-1, 1, 1, 1, 1)
        transported_uv = _ExactIdentityAddSTE.apply(uv, delta, bypass)
        if prediction.shape[-1] == 2:
            return transported_uv
        return torch.cat((transported_uv, prediction[..., 2:]), dim=-1)


class RecurrentLatentPointCorrector(nn.Module):
    """Roll out a bounded common point residual on a compact latent grid.

    The legacy flattened 2-D heads remain responsible for the absolute flow
    field and its amplitude.  This branch evolves a shared recurrent state for
    the forecast horizon and adds only a zero-time-mean u/v correction.  The
    zero-initialized output projection makes update zero an exact identity,
    while the shared transition forces the correction to represent dynamics
    instead of twenty unrelated output slices.
    """

    def __init__(
            self, feature_channels: int, hidden_channels: int,
            out_steps: int, groups: int, correction_limit: float,
            downsample: int, lowpass_modes: int = 0):
        super().__init__()
        self.out_steps = int(out_steps)
        self.correction_limit = float(correction_limit)
        self.downsample = int(downsample)
        self.lowpass_modes = int(lowpass_modes)
        timeline = torch.arange(out_steps, dtype=torch.float32).unsqueeze(1)
        modes = torch.arange(out_steps, dtype=torch.float32).unsqueeze(0)
        dct = torch.cos(
            torch.pi / out_steps * (timeline + 0.5) * modes)
        dct[:, 0] *= (1.0 / out_steps) ** 0.5
        dct[:, 1:] *= (2.0 / out_steps) ** 0.5
        retained = dct[:, :self.lowpass_modes + 1]
        self.register_buffer(
            "lowpass_projection", retained @ retained.transpose(0, 1))
        self.context_projection = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
        )
        position = torch.arange(out_steps, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, hidden_channels, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0)) / hidden_channels))
        embedding = torch.zeros(out_steps, hidden_channels)
        embedding[:, 0::2] = torch.sin(position * frequency)
        embedding[:, 1::2] = torch.cos(
            position * frequency[:embedding[:, 1::2].shape[1]])
        self.horizon_embedding = nn.Parameter(embedding)
        self.gates = nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3,
                               padding=1)
        self.candidate = nn.Conv2d(2 * hidden_channels, hidden_channels, 3,
                                   padding=1)
        self.output = nn.Conv2d(hidden_channels, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def temporal_lowpass(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.out_steps:
            raise ValueError(
                "temporal spectral split requires [B,T,H,W,C]")
        projection = self.lowpass_projection.to(dtype=value.dtype)
        return torch.einsum("st,bthwc->bshwc", projection, value)

    def temporal_highpass(self, value: torch.Tensor) -> torch.Tensor:
        return value - self.temporal_lowpass(value)

    def _bound(self, correction: torch.Tensor) -> torch.Tensor:
        peak = correction.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6)
        return correction * (
            self.correction_limit / peak).clamp_max(1.0)

    def forward(self, finest_feature: torch.Tensor) -> torch.Tensor:
        if finest_feature.ndim != 4:
            raise ValueError("recurrent latent rollout requires BCHW features")
        height, width = finest_feature.shape[-2:]
        latent_height = max(1, height // self.downsample)
        latent_width = max(1, width // self.downsample)
        context = self.context_projection(finest_feature)
        context = F.adaptive_avg_pool2d(
            context, output_size=(latent_height, latent_width))
        state = torch.tanh(context)
        outputs = []
        for step in range(self.out_steps):
            horizon = self.horizon_embedding[step].view(
                1, -1, 1, 1).to(dtype=context.dtype)
            transition_input = context + horizon
            reset, update = torch.sigmoid(self.gates(torch.cat(
                (transition_input, state), dim=1))).chunk(2, dim=1)
            candidate = torch.tanh(self.candidate(torch.cat(
                (transition_input, reset * state), dim=1)))
            state = (1.0 - update) * state + update * candidate
            outputs.append(self.output(state))
        raw = torch.stack(outputs, dim=2)
        raw = F.interpolate(
            raw, size=(self.out_steps, height, width), mode="trilinear",
            align_corners=False)
        correction = self.correction_limit * torch.tanh(
            raw / self.correction_limit)
        correction = correction - correction.mean(dim=2, keepdim=True)
        correction = correction.permute(0, 2, 3, 4, 1)
        correction = self._bound(correction)
        if self.lowpass_modes:
            correction = self.temporal_highpass(correction)
            correction = self._bound(correction)
        return correction


class PointReadoutAdapter(nn.Module):
    """A zero-output low-rank residual beside one frozen Sim point head."""

    def __init__(self, input_channels: int, output_channels: int, rank: int):
        super().__init__()
        self.down = nn.Conv2d(input_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, output_channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(value))


class SpatioTemporalQuantileRefiner(nn.Module):
    """Refine quantile logits with explicit horizon-local flow context.

    The existing UQ decoder compresses all forecast horizons into ordinary
    2-D channels. This branch restores the time axis, combines a shared
    multi-scale feature with horizon-specific point/disagreement fields, and
    convolves jointly over time and space. Its output projection is exactly
    zero at construction so old checkpoints retain bitwise-identical outputs
    until the branch is trained.
    """

    def __init__(
            self, fused_channels: int, hidden_channels: int,
            out_steps: int, quantiles: int, kernel: int,
            blocks: int, groups: int):
        super().__init__()
        self.out_steps = out_steps
        self.quantiles = quantiles
        self.shared_projection = nn.Conv2d(
            fused_channels, hidden_channels, 1)
        # Per horizon: point mean u/v, member std u/v, member range u/v.
        self.member_projection = nn.Conv3d(6, hidden_channels, 1)
        self.horizon_embedding = nn.Parameter(torch.zeros(
            1, hidden_channels, out_steps, 1, 1))
        self.blocks = nn.Sequential(*(
            SpatioTemporalResidualBlock(
                hidden_channels, kernel, groups)
            for _ in range(blocks)))
        self.output = nn.Conv3d(
            hidden_channels, 2 * quantiles, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
            self, fused: torch.Tensor,
            members: torch.Tensor) -> torch.Tensor:
        detached = members[..., :2].detach()
        mean = detached.mean(dim=1)
        std = detached.std(dim=1, unbiased=False)
        extent = detached.amax(dim=1) - detached.amin(dim=1)
        member_context = torch.cat((mean, std, extent), dim=-1).permute(
            0, 4, 1, 2, 3)
        value = (
            self.shared_projection(fused).unsqueeze(2)
            + self.member_projection(member_context)
            + self.horizon_embedding)
        value = self.blocks(F.silu(value))
        raw = self.output(value).reshape(
            members.shape[0], 2, self.quantiles,
            self.out_steps, members.shape[3], members.shape[4])
        return raw.permute(0, 3, 4, 5, 1, 2)


def _sidecar_sinusoidal_embedding(
        steps: int, channels: int) -> torch.Tensor:
    position = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float32)
        * (-log(10000.0) / max(channels, 2)))
    result = torch.zeros(steps, channels, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * frequency)
    if channels > 1:
        result[:, 1::2] = torch.cos(
            position * frequency[:result[:, 1::2].shape[1]])
    return result


class MatchedLateAxisMixer(nn.Module):
    """Use one tensor along causal time or left-padded spatial x."""

    def __init__(
            self, channels: int, kernel: int, groups: int, mode: str):
        super().__init__()
        if mode not in ("causal_temporal", "framewise_spatial_control"):
            raise ValueError(f"unsupported late sidecar axis mode {mode!r}")
        self.channels = int(channels)
        self.kernel = int(kernel)
        self.left_pad = self.kernel - 1
        self.right_pad = 0
        self.mode = str(mode)
        self.norm = _group_norm(groups, channels)
        self.depthwise = nn.Conv3d(
            channels, channels, (kernel, 1, 1),
            groups=channels, bias=False)
        self.projection = nn.Conv3d(channels, 2 * channels, 1)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 5 or trajectory.shape[2] != self.channels:
            raise ValueError(
                "late sidecar axis mixer requires [B,T,C,H,W]")
        batch, steps, channels, height, width = trajectory.shape
        normalized = self.norm(trajectory.reshape(
            batch * steps, channels, height, width))
        if self.mode == "causal_temporal":
            packed = normalized.reshape(
                batch, steps, channels, height, width
            ).permute(0, 2, 1, 3, 4)
            packed = F.pad(
                packed, (0, 0, 0, 0, self.left_pad, self.right_pad))
            filtered = self.depthwise(packed)
            value, gate = self.projection(filtered).chunk(2, dim=1)
            update = (value * F.silu(gate)).permute(0, 2, 1, 3, 4)
        else:
            # T is folded irreversibly into batch.  The exact same depthwise
            # tensor sees x on its convolution axis with the same left pad.
            packed = normalized.permute(0, 1, 3, 2).unsqueeze(-1)
            packed = F.pad(
                packed, (0, 0, 0, 0, self.left_pad, self.right_pad))
            filtered = self.depthwise(packed)
            value, gate = self.projection(filtered).chunk(2, dim=1)
            update = (value * F.silu(gate)).squeeze(-1).permute(
                0, 1, 3, 2).reshape(
                    batch, steps, channels, height, width)
        return trajectory + update


class LateMultiscaleHistorySidecar(nn.Module):
    """Preserve observed time until late fusion with two decoder scales.

    The branch deliberately leaves the mature full-resolution B210 path
    untouched.  It processes normalized input frames at half and quarter
    resolution, projects the observed trajectory into all future slots, and
    adds those dynamics features to the matching decoder coordinates.  The
    causal and framewise-spatial arms own exactly the same tensors and differ
    only in the axis used by :class:`MatchedLateAxisMixer`.
    """

    def __init__(
            self, config: EnsembleUNet2dConfig,
            decoder_channels: list[int]):
        super().__init__()
        if config.late_multiscale_sidecar_mode == "none":
            raise ValueError("late multiscale sidecar requires an active mode")
        if len(decoder_channels) != 3:
            raise ValueError("late multiscale sidecar requires three decoder scales")
        half_channels, quarter_channels = (
            int(value) for value in config.late_multiscale_sidecar_channels)
        self.mode = str(config.late_multiscale_sidecar_mode)
        self.in_steps = int(config.in_steps)
        self.out_steps = int(config.out_steps)
        self.in_channels = int(config.in_channels)
        self.half_channels = half_channels
        self.quarter_channels = quarter_channels
        self.half_input = nn.Conv2d(
            self.in_channels, half_channels, 3, stride=2, padding=1)
        self.half_axis = nn.ModuleList(
            MatchedLateAxisMixer(
                half_channels, config.late_multiscale_sidecar_kernel,
                config.groups, self.mode)
            for _ in range(config.late_multiscale_sidecar_blocks))
        self.half_spatial = ResidualConvBlock(
            half_channels, half_channels, config.groups)
        self.quarter_downsample = nn.Conv2d(
            half_channels, quarter_channels, 3, stride=2, padding=1)
        self.quarter_axis = nn.ModuleList(
            MatchedLateAxisMixer(
                quarter_channels, config.late_multiscale_sidecar_kernel,
                config.groups, self.mode)
            for _ in range(config.late_multiscale_sidecar_blocks))
        self.quarter_spatial = ResidualConvBlock(
            quarter_channels, quarter_channels, config.groups)

        # Explicit observed positions and future queries prevent the future
        # slots from collapsing to one repeated pooled state.  This mirrors
        # the successful PC3D time contract while keeping one projection map
        # shared between the two spatial scales.
        self.observed_time_half = nn.Parameter(
            _sidecar_sinusoidal_embedding(self.in_steps, half_channels))
        self.observed_time_quarter = nn.Parameter(
            _sidecar_sinusoidal_embedding(self.in_steps, quarter_channels))
        self.horizon_half = nn.Parameter(
            _sidecar_sinusoidal_embedding(self.out_steps, half_channels))
        self.horizon_quarter = nn.Parameter(
            _sidecar_sinusoidal_embedding(self.out_steps, quarter_channels))
        self.observed_to_future_logits = nn.Parameter(torch.empty(
            self.out_steps, self.in_steps))
        with torch.no_grad():
            observed = torch.arange(self.in_steps, dtype=torch.float32)
            self.observed_to_future_logits.copy_(
                -0.12 * (float(self.in_steps - 1) - observed).square().view(
                    1, self.in_steps).expand(self.out_steps, -1))
        self.quarter_fusion = nn.Conv2d(
            self.out_steps * quarter_channels, decoder_channels[0], 1)
        self.half_fusion = nn.Conv2d(
            self.out_steps * half_channels, decoder_channels[1], 1)
        # Exact legacy function at update zero.  Once these projections take
        # their first step, gradients reach the complete temporal sidecar.
        for projection in (self.quarter_fusion, self.half_fusion):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _framewise(
            module: nn.Module, trajectory: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = trajectory.shape
        value = module(trajectory.reshape(
            batch * steps, channels, height, width))
        return value.reshape(
            batch, steps, value.shape[1], value.shape[2], value.shape[3])

    @staticmethod
    def _add_time(
            trajectory: torch.Tensor,
            embedding: torch.Tensor) -> torch.Tensor:
        return trajectory + embedding.to(trajectory).view(
            1, trajectory.shape[1], -1, 1, 1)

    def _future_flatten(
            self, trajectory: torch.Tensor,
            horizon: torch.Tensor) -> torch.Tensor:
        weights = self.observed_to_future_logits.softmax(dim=-1)
        future = torch.einsum("ot,btchw->bochw", weights, trajectory)
        future = self._add_time(future, horizon)
        return future.reshape(
            future.shape[0], self.out_steps * future.shape[2],
            future.shape[3], future.shape[4])

    def forward(
            self, history: torch.Tensor,
            features: list[torch.Tensor]) -> list[torch.Tensor]:
        if (history.ndim != 5 or history.shape[1] != self.in_steps
                or history.shape[-1] != self.in_channels):
            raise ValueError(
                "late multiscale sidecar requires [B,in_steps,H,W,C]")
        if len(features) != 3:
            raise ValueError("late multiscale sidecar decoder scales changed")
        batch, steps, height, width, channels = history.shape
        frames = history.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, channels, height, width)
        half = F.silu(self.half_input(frames)).reshape(
            batch, steps, self.half_channels,
            features[1].shape[-2], features[1].shape[-1])
        half = self._add_time(half, self.observed_time_half)
        for block in self.half_axis:
            half = block(half)
        half = self._framewise(self.half_spatial, half)

        quarter = self._framewise(self.quarter_downsample, half)
        if quarter.shape[-2:] != features[0].shape[-2:]:
            raise ValueError("late sidecar quarter-resolution contract changed")
        quarter = self._add_time(
            quarter, self.observed_time_quarter)
        for block in self.quarter_axis:
            quarter = block(quarter)
        quarter = self._framewise(self.quarter_spatial, quarter)

        quarter_delta = self.quarter_fusion(
            self._future_flatten(quarter, self.horizon_quarter))
        half_delta = self.half_fusion(
            self._future_flatten(half, self.horizon_half))
        if (quarter_delta.shape != features[0].shape
                or half_delta.shape != features[1].shape):
            raise ValueError("late sidecar fusion feature contract changed")
        return [
            features[0] + quarter_delta,
            features[1] + half_delta,
            features[2],
        ]

    def execution_contract(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "scales": ("half", "quarter"),
            "channels": (self.half_channels, self.quarter_channels),
            "axis_blocks_per_scale": (
                len(self.half_axis), len(self.quarter_axis)),
            "future_projection_shape": tuple(
                self.observed_to_future_logits.shape),
            "axis_kernel": self.half_axis[0].kernel,
            "left_pad": self.half_axis[0].left_pad,
            "right_pad": self.half_axis[0].right_pad,
        }


class EnsembleUNet2d(UNet2d):
    """One U-Net backbone, bootstrap point heads, and a quantile UQ decoder."""

    def __init__(self, config: EnsembleUNet2dConfig):
        config.validate()
        # Build the shared U-Net without its legacy interval heads, then restore
        # the full ensemble config for inherited persistence helpers.
        base_config = UNet2dConfig(**{
            key: getattr(config, key)
            for key in UNet2dConfig.__dataclass_fields__
        })
        super().__init__(replace(base_config, predict_uncertainty=False))
        self.config = config
        decoder_channels = [
            config.base * 2 ** level
            for level in range(config.depth - 2, -1, -1)
        ]
        # This from-scratch structural arm is isolated from the mature B210
        # initialization stream.  Enabling its exact-zero late projections
        # therefore preserves every common state tensor and update-zero output.
        with torch.random.fork_rng(devices=[]):
            self.late_multiscale_sidecar = (
                LateMultiscaleHistorySidecar(config, decoder_channels)
                if config.late_multiscale_sidecar_mode != "none" else None)

        # A true second decoder begins at the shared bottleneck.  Both the
        # responsibility-split candidate and the shared-fused capacity control
        # instantiate exactly these modules, so their parameter count and
        # executed operator path are identical.  An isolated RNG scope keeps
        # every legacy tensor comparable to a model with this arm disabled.
        with torch.random.fork_rng(devices=[]):
            if config.dual_decoder_mode != "none":
                channels = [
                    config.base * 2 ** level
                    for level in range(config.depth)
                ]
                if config.dual_decoder_mode in REGIME_EXPERT_DECODER_MODES:
                    # ``deepcopy`` is intentional: the released-data Sim
                    # endpoint has one mature decoder, so a fair Real-only
                    # specialization must begin from that exact function, not
                    # from a differently initialized or differently shaped
                    # dynamics block.  A checkpoint upgrader calls the public
                    # copy method again after loading the mature first path.
                    from copy import deepcopy
                    self.fluctuation_up = deepcopy(self.up)
                    self.fluctuation_decoder = deepcopy(self.decoder)
                    self.fluctuation_decoder_regime_film = deepcopy(
                        self.decoder_regime_film)
                    self.fluctuation_cross_conditioner = deepcopy(
                        self.cross_conditioner)
                else:
                    self.fluctuation_up = nn.ModuleList()
                    self.fluctuation_decoder = nn.ModuleList()
                    for level in range(config.depth - 1, 0, -1):
                        self.fluctuation_up.append(nn.ConvTranspose2d(
                            channels[level], channels[level - 1], 2, stride=2))
                        self.fluctuation_decoder.append(
                            FluctuationDynamicsBlock(
                                2 * channels[level - 1], channels[level - 1],
                                config.groups))
                    self.fluctuation_decoder_regime_film = (
                        nn.ModuleList(
                            nn.Linear(
                                config.regime_summary_channels, 2 * width)
                            for width in decoder_channels)
                        if config.multi_scale_regime_film else None)
                    if self.fluctuation_decoder_regime_film is not None:
                        for film in self.fluctuation_decoder_regime_film:
                            nn.init.zeros_(film.weight)
                            nn.init.zeros_(film.bias)
                    self.fluctuation_cross_conditioner = (
                        LocalDynamicsCrossConditioner(
                            config.cross_conditioned_steps,
                            config.cross_conditioned_channels,
                            decoder_channels, config.groups,
                            config.dynamics_state_channels)
                        if config.cross_conditioned_backbone else None)
            else:
                self.fluctuation_up = None
                self.fluctuation_decoder = None
                self.fluctuation_decoder_regime_film = None
                self.fluctuation_cross_conditioner = None
        # A one-logit sample router reads only the observed twenty-frame
        # velocity mean/std descriptor.  Its input is detached explicitly, so
        # the optional AoA-cluster classification auxiliary can update only
        # the router and cannot leak labels into the shared encoder.
        with torch.random.fork_rng(devices=[]):
            self.regime_expert_router = (
                nn.Sequential(
                    nn.Linear(config.in_steps * 4, 16),
                    nn.Tanh(),
                    nn.Linear(16, 1),
                )
                if config.dual_decoder_mode in REGIME_EXPERT_DECODER_MODES
                else None)
            if self.regime_expert_router is not None:
                # The endpoint groups are non-monotone in AoA
                # ({0,20} versus {5,10,15}), so a single hyperplane is an
                # avoidable expressivity bottleneck.  Only the final map is
                # zero: update-zero routing remains exactly 0.5 while the
                # hidden history descriptor is ready to learn immediately.
                nn.init.zeros_(self.regime_expert_router[-1].weight)
                nn.init.zeros_(self.regime_expert_router[-1].bias)
        self.register_buffer(
            "regime_expert_router_mean",
            (torch.zeros(config.in_steps * 4)
             if self.regime_expert_router is not None else None))
        self.register_buffer(
            "regime_expert_router_std",
            (torch.ones(config.in_steps * 4)
             if self.regime_expert_router is not None else None))
        # A continuous responsibility gate starts at the exact shared-fused
        # control and can move each scale/channel toward the dynamics decoder:
        #   role = fused + tanh(gate) * (dynamics - fused).
        # Zero construction consumes no RNG, so every common state tensor and
        # every update-zero output remains exactly paired with shared_fused.
        self.fluctuation_role_gates = (
            nn.ParameterList(
                nn.Parameter(torch.zeros(1, width, 1, 1))
                for width in decoder_channels)
            if config.dual_decoder_mode in (
                "learned_fluctuation_gate",
                "separate_point_uq_fluctuation_gates")
            else None)
        self.uq_fluctuation_role_gates = (
            nn.ParameterList(
                nn.Parameter(torch.zeros(1, width, 1, 1))
                for width in decoder_channels)
            if (config.dual_decoder_mode
                == "separate_point_uq_fluctuation_gates")
            else None)

        first_member = self.point_heads
        del self.point_heads
        self.temporal_point_decoder = (
            TemporalPointDecoder(
                decoder_channels, config.temporal_point_channels,
                config.out_steps, config.out_channels,
                config.ensemble_members, config.temporal_point_blocks,
                config.temporal_point_kernel, config.groups,
                config.temporal_point_amplitude_factorization,
                config.temporal_point_amplitude_limit)
            if config.temporal_point_decoder else None)
        self.latent_ode_point_decoder = (
            LatentODEPointDecoder(
                decoder_channels, config.latent_ode_channels,
                config.out_steps, config.out_channels,
                config.ensemble_members, config.latent_ode_solver_substeps,
                config.latent_ode_step_size,
                config.latent_ode_derivative_limit,
                config.latent_ode_residual_limit, config.groups)
            if config.latent_ode_point_decoder else None)
        self.latent_ude_band_point_decoder = (
            LatentUDEBandPointDecoder(
                decoder_channels, config.latent_ode_channels,
                config.out_steps, config.out_channels,
                config.ensemble_members, config.latent_ode_solver_substeps,
                config.latent_ode_step_size,
                config.latent_ode_derivative_limit,
                config.latent_ude_closure_limit,
                config.latent_ode_residual_limit, config.groups,
                config.mean_fluctuation)
            if config.latent_ude_band_decoder else None)
        if (self.temporal_point_decoder is None
                and self.latent_ode_point_decoder is None
                and self.latent_ude_band_point_decoder is None):
            self.member_point_heads = nn.ModuleList([first_member])
            point_channels = config.out_channels * (
                config.out_steps + 1
                if config.mean_fluctuation else config.out_steps)
            for _ in range(1, config.ensemble_members):
                heads = nn.ModuleList(
                    nn.Conv2d(width, point_channels, 1)
                    for width in decoder_channels)
                for head in heads:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
                self.member_point_heads.append(heads)
        else:
            # Keep the public module name for code that enumerates point heads;
            # temporal-point checkpoints store their projections under the
            # explicit temporal_point_decoder namespace instead.
            self.member_point_heads = nn.ModuleList()
        # The residual UDE is initialized in an isolated RNG scope so merely
        # enabling the identity-at-update-zero branch cannot perturb any
        # common direct or UQ parameter.
        with torch.random.fork_rng(devices=[]):
            self.latent_ude_coarse_residual = (
                LatentUDECoarseResidual(
                    decoder_channels[0], config.latent_ode_channels,
                    config.out_steps, config.out_channels,
                    config.ensemble_members,
                    config.latent_ode_solver_substeps,
                    config.latent_ode_step_size,
                    config.latent_ode_derivative_limit,
                    config.latent_ude_closure_limit,
                    config.latent_ude_coarse_residual_limit,
                    config.groups,
                    (min(2, config.out_channels)
                     if config.latent_ude_coarse_velocity_only else None),
                    zero_spatial_mean=(
                        not config.latent_ude_exclusive_coarse_fluctuation))
                if (config.latent_ude_coarse_residual
                    or config.latent_ude_exclusive_coarse_fluctuation)
                else None)
        # Real-domain low-rank adapters are created in an isolated RNG scope:
        # their zero output leaves every legacy prediction exact at update 0,
        # and enabling them cannot perturb any common UQ initialization.
        with torch.random.fork_rng(devices=[]):
            self.member_point_head_adapters = (
                nn.ModuleList(
                    nn.ModuleList(
                        PointReadoutAdapter(
                            width, point_channels,
                            config.point_readout_adapter_rank)
                        for width in decoder_channels)
                    for _ in range(config.ensemble_members))
                if config.point_readout_adapter_rank else None)

        with torch.random.fork_rng(devices=[]):
            self.point_feature_refiner = (
                SharedPointFeatureRefiner(config.base)
                if config.point_feature_refiner_channels else None)

        # Candidate-only adapters must not move the initialization stream of
        # any legacy point/UQ parameter.  Their zero output makes update zero
        # an exact identity while the independent RNG scope keeps every common
        # state tensor identical to the no-adapter architecture.
        with torch.random.fork_rng(devices=[]):
            self.member_feature_adapters = (
                nn.ModuleList(
                    nn.ModuleList(
                        MemberFeatureAdapter(
                            width, config.member_adapter_channels,
                            config.groups)
                        for width in decoder_channels)
                    for _ in range(config.ensemble_members))
                if config.member_adapter_channels else None)
        # The point output is already parameterized as temporal mean plus a
        # strictly zero-time-mean fluctuation.  This optional tower gives the
        # fluctuation role its own nonlinear multiscale coordinates without
        # changing the mean feature path.  Its zero projections make the
        # complete model an exact update-zero identity and the isolated RNG
        # scope keeps every legacy/member/UQ tensor bitwise comparable.
        with torch.random.fork_rng(devices=[]):
            self.fluctuation_role_adapters = (
                nn.ModuleList(
                    MemberFeatureAdapter(
                        width, config.fluctuation_role_channels,
                        config.groups)
                    for width in decoder_channels)
                if config.fluctuation_role_channels else None)
        # Populated only after a strict checkpoint load by the optional
        # inference rewrite. ``None`` adds no state and leaves training exact.
        self.fused_member_feature_adapter_readouts = None
        # This candidate-only branch must not move the initialization stream of
        # any legacy point/UQ parameter, so paired update-zero models remain
        # exactly comparable even when the control keeps this module frozen.
        with torch.random.fork_rng(devices=[]):
            self.decoupled_member_residual = (
                DecoupledMemberResidualDecoder(
                    config.base, config.decoupled_member_residual_channels,
                    config.ensemble_members, config.out_steps,
                    config.out_channels,
                    config.decoupled_member_residual_blocks, config.groups)
                if config.decoupled_member_residual_channels else None)
        self.mean_profile_refiner = (
            nn.Sequential(
                nn.Conv2d(
                    config.base, config.mean_profile_channels, 1),
                _group_norm(config.groups, config.mean_profile_channels),
                nn.SiLU(),
                nn.Conv2d(
                    config.mean_profile_channels,
                    config.mean_profile_channels, (1, 9),
                    padding=(0, 4), groups=config.mean_profile_channels),
                _group_norm(config.groups, config.mean_profile_channels),
                nn.SiLU(),
                nn.Conv2d(
                    config.mean_profile_channels,
                    config.mean_profile_channels, (9, 1),
                    padding=(4, 0), groups=config.mean_profile_channels),
                _group_norm(config.groups, config.mean_profile_channels),
                nn.SiLU(),
                nn.Conv2d(config.mean_profile_channels, 2, 1),
            )
            if config.mean_profile_channels else None)
        if self.mean_profile_refiner is not None:
            nn.init.zeros_(self.mean_profile_refiner[-1].weight)
            nn.init.zeros_(self.mean_profile_refiner[-1].bias)
        # The Real AoA gradient audit included this readout in the conflicting
        # shared-decoder scope.  Regime experts therefore duplicate it too;
        # leaving it shared would retain a small but direct mean-flow conflict.
        with torch.random.fork_rng(devices=[]):
            if config.dual_decoder_mode in REGIME_EXPERT_DECODER_MODES:
                from copy import deepcopy
                self.regime_expert_mean_profile_refiner = deepcopy(
                    self.mean_profile_refiner)
            else:
                self.regime_expert_mean_profile_refiner = None
        self.energy_controller = (
            (DecoderContextEnergyController(
                config.base, config.physical_energy_hidden_channels, config.groups,
                config.physical_energy_history_steps, config.physical_energy_decoder_context)
             if config.physical_energy_decoder_context != "none" else
             ContextConditionedEnergyHead(
                config.physical_energy_history_steps,
                config.physical_energy_hidden_channels, config.groups)
             if getattr(config, "physical_energy_context_head", False) else
             (HybridMultiScaleHistoryEnergyController(
                decoder_channels, config.physical_energy_history_steps,
                config.physical_energy_hidden_channels, config.groups)
             if (config.physical_energy_history_conditioned
                 and config.physical_energy_multiscale) else
             (HistoryConditionedPhysicalEnergyController(
                config.physical_energy_history_steps,
                config.physical_energy_hidden_channels, config.groups)
             if config.physical_energy_history_conditioned else
             (MultiScalePhysicalEnergyController(
                decoder_channels, config.physical_energy_hidden_channels,
                config.groups)
             if config.physical_energy_multiscale else
             (HorizonLocalPhysicalEnergyController(
                 config.base, config.physical_energy_hidden_channels,
                 config.out_steps, config.groups)
              if config.physical_energy_horizon_local else
              PhysicalEnergyController(
                  config.base, config.physical_energy_hidden_channels,
                  config.groups))))))
            if config.physical_energy_controller else None)
        self.phase_delta = (
            PhaseConditionedLocalDelta(
                config.in_steps, config.out_steps,
                config.phase_delta_hidden_channels,
                config.phase_delta_steps, config.groups)
            if config.phase_delta_hidden_channels else None)
        if config.temporal_band_mixer or config.spatial_transport_warp:
            # This candidate-only branch must not move the legacy UQ
            # initialization stream.  Existing isoenergetic checkpoints keep
            # their historical construction order below.
            with torch.random.fork_rng(devices=[]):
                self.temporal_transport = (
                    HistoryConditionedSpatialTransport(
                        config.base, config.spatial_transport_channels,
                        config.out_steps, config.groups,
                        config.spatial_transport_max_displacement)
                    if config.spatial_transport_warp else
                    HistoryConditionedTemporalBandMixer(
                        config.base, config.temporal_band_channels,
                        config.out_steps, config.temporal_band_count,
                        config.groups, config.temporal_band_log_gain_limit,
                        config.temporal_band_phase_limit))
        else:
            self.temporal_transport = (
                IsoenergeticTemporalTransport(
                    config.base, config.temporal_transport_channels,
                    config.out_steps, config.temporal_transport_blocks,
                    config.temporal_transport_kernel, config.groups,
                    config.temporal_transport_max_displacement)
                if config.isoenergetic_temporal_transport else None)
        # This from-scratch research branch is isolated so enabling it cannot
        # perturb any legacy point/UQ initialization.  Its own zero projection
        # makes paired update-zero predictions exactly identical.
        with torch.random.fork_rng(devices=[]):
            self.recurrent_latent_rollout = (
                RecurrentLatentPointCorrector(
                    config.base, config.recurrent_latent_rollout_channels,
                    config.out_steps, config.groups,
                    config.recurrent_latent_rollout_limit,
                    config.recurrent_latent_rollout_downsample,
                    config.recurrent_latent_rollout_lowpass_modes)
                if config.recurrent_latent_rollout_channels else None)
        # Keep this candidate-only branch in an isolated RNG scope so enabling
        # it does not change the initialization of any legacy UQ parameter.
        with torch.random.fork_rng(devices=[]):
            self.hard_region_expert = (
                HardRegionResidualExpert(
                    config.base, config.hard_region_expert_channels,
                    config.out_steps, config.hard_region_expert_blocks,
                    config.groups, config.hard_region_expert_limit,
                    config.hard_region_expert_detach_context,
                    config.hard_region_expert_feature_noise_std)
                if config.hard_region_expert_channels else None)
        # This finite-volume closure is an independent architecture arm.  Its
        # isolated RNG and zero flux projections preserve the paired legacy
        # initialization and update-zero prediction exactly.
        with torch.random.fork_rng(devices=[]):
            self.conservative_closure = (
                RegimeGatedConservativeClosure(
                    config.base, config.conservative_closure_channels,
                    config.out_steps, config.in_steps,
                    config.conservative_closure_experts,
                    config.conservative_closure_blocks, config.groups,
                    config.conservative_closure_flux_limit,
                    config.conservative_closure_router_temperature)
                if config.conservative_closure_experts else None)
        # The gradient control and curl candidate differ only in a fixed
        # operator after this common scalar head.  Isolating construction keeps
        # all legacy point/UQ tensors bitwise paired when the branch is off.
        with torch.random.fork_rng(devices=[]):
            self.scalar_potential_residual = (
                ScalarPotentialResidual(
                    config.base, config.scalar_potential_residual_channels,
                    config.out_steps, config.scalar_potential_residual_blocks,
                    config.groups, config.scalar_potential_residual_mode,
                    config.scalar_potential_residual_limit)
                if config.scalar_potential_residual_mode != "none" else None)

        self.uq_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, config.uq_hidden_channels, 1),
                _group_norm(config.groups, config.uq_hidden_channels),
                nn.SiLU(),
                ResidualConvBlock(
                    config.uq_hidden_channels,
                    config.uq_hidden_channels,
                    config.groups),
            )
            for width in decoder_channels)
        with torch.random.fork_rng(devices=[]):
            self.fluctuation_uq_projections = (
                nn.ModuleList(
                    nn.Sequential(
                        nn.Conv2d(width, config.uq_hidden_channels, 1),
                        _group_norm(config.groups, config.uq_hidden_channels),
                        nn.SiLU(),
                        ResidualConvBlock(
                            config.uq_hidden_channels,
                            config.uq_hidden_channels,
                            config.groups),
                    )
                    for width in decoder_channels)
                if (config.dual_decoder_mode != "none"
                    and config.dual_decoder_mode
                    not in REGIME_EXPERT_DECODER_MODES) else None)
        disagreement_channels = 4 * config.out_steps
        fusion_channels = (
            len(decoder_channels) * config.uq_hidden_channels
            + (len(decoder_channels) * config.uq_hidden_channels
               if self.fluctuation_uq_projections is not None else 0)
            + disagreement_channels
            + ((100 if config.uq_full_history_context else 70) if config.uq_flow_context else 0))
        self.uq_fusion = nn.Sequential(
            nn.Conv2d(
                fusion_channels, config.uq_hidden_channels, 3, padding=1),
            ResidualConvBlock(
                config.uq_hidden_channels,
                config.uq_hidden_channels,
                config.groups),
            ResidualConvBlock(
                config.uq_hidden_channels,
                config.uq_hidden_channels,
                config.groups),
        )
        quantile_channels = (
            config.out_steps * 2 * len(config.quantile_levels))
        self.quantile_head = nn.Conv2d(
            config.uq_hidden_channels, quantile_channels, 1)
        nn.init.zeros_(self.quantile_head.weight)
        bias = torch.full(
            (config.out_steps, 2, len(config.quantile_levels)),
            float(config.quantile_gap_bias))
        bias[..., len(config.quantile_levels) // 2] = 0.0
        with torch.no_grad():
            self.quantile_head.bias.copy_(bias.reshape(-1))
        middle = len(config.quantile_levels) // 2
        self.interval_candidate_pairs = tuple(
            (lower_index, upper_index)
            for lower_index in range(middle + 1)
            for upper_index in range(
                max(middle, lower_index + 1),
                len(config.quantile_levels)))
        self.coverage_calibration_head = None
        if config.coverage_calibration:
            calibration_outputs = (
                len(self.interval_candidate_pairs)
                if config.coverage_calibration_mode == "independent"
                else 2)
            output = nn.Conv2d(
                config.uq_hidden_channels,
                config.out_steps * 2 * calibration_outputs,
                config.coverage_calibration_kernel,
                padding=config.coverage_calibration_kernel // 2)
            self.coverage_calibration_head = (
                nn.Sequential(
                    *(ResidualConvBlock(
                        config.uq_hidden_channels,
                        config.uq_hidden_channels, config.groups)
                      for _ in range(config.coverage_calibration_blocks)),
                    output)
                if config.coverage_calibration_blocks else output)
            final = (
                self.coverage_calibration_head[-1]
                if config.coverage_calibration_blocks
                else self.coverage_calibration_head)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.utility_refinement_head = None
        if config.utility_refinement:
            output = nn.Conv2d(
                config.uq_hidden_channels, config.out_steps * 2 * 2,
                config.utility_refinement_kernel,
                padding=config.utility_refinement_kernel // 2)
            self.utility_refinement_head = (
                nn.Sequential(
                    *(ResidualConvBlock(
                        config.uq_hidden_channels,
                        config.uq_hidden_channels, config.groups)
                      for _ in range(config.utility_refinement_blocks)),
                    output)
                if config.utility_refinement_blocks else output)
        if self.utility_refinement_head is not None:
            output = (
                self.utility_refinement_head[-1]
                if config.utility_refinement_blocks
                else self.utility_refinement_head)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        # A stable UQ responsibility boundary: unlike the legacy UQ tower,
        # this allocator never reads backbone feature coordinates.  It sees
        # only history-derived motion cues and the frozen point forecast, then
        # redistributes (but does not create) spatial interval-width budget.
        # The isolated RNG and zero final projection preserve every legacy
        # tensor and the update-zero interval function.
        with torch.random.fork_rng(devices=[]):
            if config.output_space_uq_channels:
                output_space_layers: list[nn.Module] = [
                    nn.Conv2d(
                        30 + 2 * config.out_steps,
                        config.output_space_uq_channels, 3, padding=1),
                    _group_norm(
                        config.groups, config.output_space_uq_channels),
                    nn.SiLU(),
                    *(ResidualConvBlock(
                        config.output_space_uq_channels,
                        config.output_space_uq_channels, config.groups)
                      for _ in range(config.output_space_uq_blocks)),
                    nn.Conv2d(
                        config.output_space_uq_channels,
                        config.out_steps * 2 * 2, 1),
                ]
                self.output_space_uq_allocator = nn.Sequential(
                    *output_space_layers)
                nn.init.zeros_(self.output_space_uq_allocator[-1].weight)
                nn.init.zeros_(self.output_space_uq_allocator[-1].bias)
            else:
                self.output_space_uq_allocator = None
        # The global budgeter owns only sample/horizon/channel coverage scale.
        # Keeping it separate from the spatial allocator makes their SPS
        # contributions independently testable while preserving the same
        # hidden-coordinate-free input contract.
        with torch.random.fork_rng(devices=[]):
            if config.output_space_uq_budget_channels:
                budget_layers: list[nn.Module] = [
                    nn.Conv2d(
                        30 + 2 * config.out_steps,
                        config.output_space_uq_budget_channels,
                        3, padding=1),
                    _group_norm(
                        config.groups,
                        config.output_space_uq_budget_channels),
                    nn.SiLU(),
                    *(ResidualConvBlock(
                        config.output_space_uq_budget_channels,
                        config.output_space_uq_budget_channels,
                        config.groups)
                      for _ in range(config.output_space_uq_budget_blocks)),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(
                        config.output_space_uq_budget_channels,
                        config.out_steps * 2 * 2),
                ]
                self.output_space_uq_budgeter = nn.Sequential(*budget_layers)
                nn.init.zeros_(self.output_space_uq_budgeter[-1].weight)
                nn.init.zeros_(self.output_space_uq_budgeter[-1].bias)
            else:
                self.output_space_uq_budgeter = None
        self.temporal_uq_refiner = (
            SpatioTemporalQuantileRefiner(
                config.uq_hidden_channels,
                config.temporal_uq_channels,
                config.out_steps,
                len(config.quantile_levels),
                config.temporal_uq_kernel,
                config.temporal_uq_blocks,
                config.groups)
            if config.temporal_uq_refinement else None)
        self.disagreement_allocation_strength = (
            nn.Parameter(torch.zeros(config.out_steps, 2))
            if config.budgeted_disagreement_allocation else None)
        self.disagreement_allocation_sharpness = (
            nn.Parameter(torch.zeros(config.out_steps, 2))
            if config.budgeted_disagreement_allocation else None)
        # The mixture head is an opt-in research branch.  Keeping it in an
        # isolated RNG scope preserves every legacy tensor and lets an old
        # checkpoint be upgraded without changing point or quantile outputs.
        with torch.random.fork_rng(devices=[]):
            self.residual_mixture_head = (
                nn.Conv2d(
                    config.uq_hidden_channels,
                    config.out_steps * 2
                    * config.residual_mixture_components * 3,
                    1)
                if config.residual_mixture_components else None)
        if self.residual_mixture_head is not None:
            nn.init.zeros_(self.residual_mixture_head.weight)
            components = config.residual_mixture_components
            bias = torch.zeros(config.out_steps, 2, components, 3)
            centers = torch.linspace(-0.75, 0.75, components)
            raw_centers = torch.atanh(
                (centers / config.residual_mixture_mean_limit).clamp(
                    -0.999, 0.999))
            scale_above_floor = (
                config.residual_mixture_scale_init
                - config.residual_mixture_scale_floor)
            raw_scale = torch.log(torch.expm1(torch.tensor(
                scale_above_floor)))
            bias[..., 1] = raw_centers
            bias[..., 2] = raw_scale
            with torch.no_grad():
                self.residual_mixture_head.bias.copy_(bias.reshape(-1))
        # HCOB-F is constructed after every legacy module, inside an isolated
        # RNG scope.  Thus enabling it neither changes old parameter values
        # nor consumes randomness from the native UQ construction stream.
        with torch.random.fork_rng(devices=[]):
            self.history_operator_conditioner = (
                nn.Sequential(
                    nn.Linear(config.in_steps * 4, 16),
                    nn.Tanh(),
                ) if config.history_conditioned_operator_basis else None)
            self.history_operator_basis = (
                nn.ModuleList(
                    HistoryConditionedOperatorBasis(width)
                    for width in decoder_channels)
                if config.history_conditioned_operator_basis else None)
        self._history_operator_condition_override: torch.Tensor | None = None

    def architecture_config(self) -> dict:
        return {"architecture": "ensemble_unet", **asdict(self.config)}

    @torch.no_grad()
    def copy_primary_into_regime_expert_(self) -> None:
        """Copy a loaded single-decoder endpoint into the second expert.

        Construction already starts both paths identically.  This explicit
        operation is required after the common mature Sim state has been
        loaded into the primary path of an upgraded model.
        """
        if self.config.dual_decoder_mode not in REGIME_EXPERT_DECODER_MODES:
            raise RuntimeError("model is not a regime-expert decoder")
        if self.fluctuation_up is None or self.fluctuation_decoder is None:
            raise RuntimeError("regime-expert decoder modules are missing")
        self.fluctuation_up.load_state_dict(self.up.state_dict(), strict=True)
        self.fluctuation_decoder.load_state_dict(
            self.decoder.state_dict(), strict=True)
        if ((self.decoder_regime_film is None)
                != (self.fluctuation_decoder_regime_film is None)):
            raise RuntimeError("regime-expert FiLM topology differs")
        if self.decoder_regime_film is not None:
            self.fluctuation_decoder_regime_film.load_state_dict(
                self.decoder_regime_film.state_dict(), strict=True)
        if ((self.cross_conditioner is None)
                != (self.fluctuation_cross_conditioner is None)):
            raise RuntimeError("regime-expert cross-conditioner differs")
        if self.cross_conditioner is not None:
            self.fluctuation_cross_conditioner.load_state_dict(
                self.cross_conditioner.state_dict(), strict=True)
        if ((self.mean_profile_refiner is None)
                != (self.regime_expert_mean_profile_refiner is None)):
            raise RuntimeError("regime-expert mean refiner topology differs")
        if self.mean_profile_refiner is not None:
            self.regime_expert_mean_profile_refiner.load_state_dict(
                self.mean_profile_refiner.state_dict(), strict=True)
        if self.regime_expert_router is None:
            raise RuntimeError("regime-expert router is missing")
        self.regime_expert_router[-1].weight.zero_()
        self.regime_expert_router[-1].bias.zero_()

    def regime_expert_router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return history-only regime logits without metadata or target."""
        if (self.regime_expert_router is None
                or self.regime_expert_router_mean is None
                or self.regime_expert_router_std is None):
            raise RuntimeError("regime-expert router is disabled")
        descriptor = history_regime_summary(x.detach())
        descriptor = (
            descriptor - self.regime_expert_router_mean.to(descriptor))
        descriptor = descriptor / self.regime_expert_router_std.to(
            descriptor).clamp_min(1.0e-6)
        return self.regime_expert_router(descriptor)

    @torch.no_grad()
    def set_regime_expert_router_standardization(
            self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Bind prefix-only descriptor statistics into the model state."""
        if (self.regime_expert_router_mean is None
                or self.regime_expert_router_std is None):
            raise RuntimeError("regime-expert router is disabled")
        mean = torch.as_tensor(mean, device=self.regime_expert_router_mean.device,
                               dtype=self.regime_expert_router_mean.dtype)
        std = torch.as_tensor(std, device=self.regime_expert_router_std.device,
                              dtype=self.regime_expert_router_std.dtype)
        if (mean.shape != self.regime_expert_router_mean.shape
                or std.shape != self.regime_expert_router_std.shape
                or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all()
                or not torch.all(std > 0.0)):
            raise ValueError("invalid regime-expert router standardization")
        self.regime_expert_router_mean.copy_(mean)
        self.regime_expert_router_std.copy_(std)

    def history_operator_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deployment-safe 80->16 history condition for HCOB-F."""
        if self.history_operator_conditioner is None:
            raise RuntimeError("HCOB-F is disabled")
        return self.history_operator_conditioner(history_regime_summary(x))

    def set_history_operator_condition_override(
            self, condition: torch.Tensor | None) -> None:
        """Install a transient donor condition for the sealed permutation arm.

        The override is intentionally runtime-only: aligned inference always
        derives its condition from the current sample, and a fixed-permuted
        experiment must explicitly supply its pre-sealed donor condition.
        """
        if self.history_operator_basis is None:
            raise RuntimeError("HCOB-F is disabled")
        if condition is not None and (
                condition.ndim != 2 or condition.shape[1] != 16):
            raise ValueError("HCOB-F override must have shape [B,16]")
        self._history_operator_condition_override = condition

    def _history_operator_features(
            self, x: torch.Tensor,
            features: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.history_operator_basis is None:
            return features
        if len(features) != len(self.history_operator_basis):
            raise RuntimeError("HCOB-F decoder scale contract changed")
        condition = self._history_operator_condition_override
        if condition is None:
            condition = self.history_operator_condition(x)
        if (condition.shape[0] != x.shape[0]
                or condition.device != x.device):
            raise ValueError("HCOB-F override does not match the current batch")
        condition = condition.to(dtype=features[0].dtype)
        return [basis(feature, condition)
                for basis, feature in zip(self.history_operator_basis, features)]

    def _decode_role_features(
            self, x: torch.Tensor,
            ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None,
                       list[torch.Tensor] | None, torch.Tensor | None]:
        """Return role features and an optional routed mean correction."""
        if self.config.dual_decoder_mode == "none":
            features = self._decode(x)
            if self.late_multiscale_sidecar is not None:
                features = self.late_multiscale_sidecar(x, features)
            if self.history_operator_basis is not None:
                fluctuation_features = self._history_operator_features(x, features)
                return (features, fluctuation_features,
                        fluctuation_features, None)
            return features, None, None, None
        if self.fluctuation_up is None or self.fluctuation_decoder is None:
            raise RuntimeError("dual decoder modules are missing")
        encoded, skips, regime, dynamics_state = self._encode_history(x)
        mean_features = self._decode_from_encoded(
            encoded, skips, regime, dynamics_state, x, self.up, self.decoder,
            self.decoder_regime_film, self.cross_conditioner)
        fluctuation_features = self._decode_from_encoded(
            encoded, skips, regime, dynamics_state, x,
            self.fluctuation_up, self.fluctuation_decoder,
            self.fluctuation_decoder_regime_film,
            self.fluctuation_cross_conditioner)
        if len(mean_features) != len(fluctuation_features):
            raise RuntimeError("dual decoder scale contract changed")
        if self.config.dual_decoder_mode in REGIME_EXPERT_DECODER_MODES:
            logits = self.regime_expert_router_logits(x)
            probability = torch.sigmoid(logits)
            if self.config.dual_decoder_mode == "regime_expert_fixed_average":
                # Keep the router in the executed graph for the same module
                # path as R2, while making the point routing coefficient
                # exactly one half.
                weight = probability * 0.0 + 0.5
            else:
                weight = probability
            weight = weight[..., None, None]
            fused = [
                first + weight * (second - first)
                for first, second in zip(
                    mean_features, fluctuation_features)
            ]
            mean_correction = None
            if self.mean_profile_refiner is not None:
                if self.regime_expert_mean_profile_refiner is None:
                    raise RuntimeError(
                        "regime-expert mean refiner is missing")
                first_mean = self.mean_profile_refiner(mean_features[-1])
                second_mean = self.regime_expert_mean_profile_refiner(
                    fluctuation_features[-1])
                mean_correction = first_mean + weight * (
                    second_mean - first_mean)
            # This is a complete-decoder mixture.  Point, member and the
            # unchanged native-UQ tower all consume the same routed features;
            # no extra role-specific UQ projection is introduced in round one.
            return fused, None, None, mean_correction
        if self.config.dual_decoder_mode in (
                "shared_fused", "fused_mean_fluctuation",
                "learned_fluctuation_gate",
                "separate_point_uq_fluctuation_gates"):
            # The discrete arms are an equal-parameter/equal-compute pair.
            # The continuous arm adds only one scalar gate per feature channel
            # and begins at the exact shared_fused coordinate contract.
            fused = [
                0.5 * (mean_value + fluctuation_value)
                for mean_value, fluctuation_value in zip(
                    mean_features, fluctuation_features)]
            if self.config.dual_decoder_mode == "shared_fused":
                return fused, fused, fused, None
            if self.config.dual_decoder_mode == "fused_mean_fluctuation":
                return fused, fluctuation_features, fluctuation_features, None
            if self.fluctuation_role_gates is None:
                raise RuntimeError("continuous fluctuation gates are missing")
            if len(self.fluctuation_role_gates) != len(fused):
                raise RuntimeError("continuous fluctuation gate scales changed")
            point_role_features = [
                fused_value + torch.tanh(gate) * (
                    fluctuation_value - fused_value)
                for fused_value, fluctuation_value, gate in zip(
                    fused, fluctuation_features,
                    self.fluctuation_role_gates)]
            if (self.config.dual_decoder_mode
                    == "learned_fluctuation_gate"):
                return fused, point_role_features, point_role_features, None
            if self.uq_fluctuation_role_gates is None:
                raise RuntimeError("continuous UQ fluctuation gates are missing")
            if len(self.uq_fluctuation_role_gates) != len(fused):
                raise RuntimeError("continuous UQ gate scales changed")
            uq_role_features = [
                fused_value + torch.tanh(gate) * (
                    fluctuation_value - fused_value)
                for fused_value, fluctuation_value, gate in zip(
                    fused, fluctuation_features,
                    self.uq_fluctuation_role_gates)]
            return fused, point_role_features, uq_role_features, None
        return mean_features, fluctuation_features, fluctuation_features, None

    def fuse_member_feature_adapters_for_inference(self) -> bool:
        """Fuse trained feature-adapter output projections into point heads.

        This is intentionally a post-load rewrite: checkpoints retain their
        complete training architecture and strict state-dict contract. It is
        idempotent and returns whether a rewrite was performed.
        """
        if self.fused_member_feature_adapter_readouts is not None:
            return False
        if (self.member_feature_adapters is None
                or self.fluctuation_role_adapters is not None
                or self.config.dual_decoder_mode != "none"):
            return False
        if (self.temporal_point_decoder is not None
                or self.latent_ode_point_decoder is not None
                or self.latent_ude_band_point_decoder is not None):
            raise RuntimeError(
                "member feature fusion requires direct multi-scale heads")
        if self.member_point_head_adapters is not None:
            raise RuntimeError(
                "member feature fusion does not stack readout adapters")
        fused = nn.ModuleList()
        for adapters, heads in zip(
                self.member_feature_adapters, self.member_point_heads):
            if len(adapters) != len(heads):
                raise RuntimeError("member adapter/head scale count changed")
            fused.append(nn.ModuleList(
                FusedMemberFeatureAdapterReadout.from_modules(adapter, head)
                for adapter, head in zip(adapters, heads)))
        # Register the compact modules before releasing the wide projections.
        self.fused_member_feature_adapter_readouts = fused
        self.member_feature_adapters = None
        return True

    def _member_prediction(
            self, x: torch.Tensor, features: list[torch.Tensor],
            heads: nn.ModuleList,
            adapters: nn.ModuleList | None = None,
            fluctuation_residuals: list[torch.Tensor] | None = None,
            fluctuation_features: list[torch.Tensor] | None = None,
            ) -> torch.Tensor:
        batch, _, height, width, _ = x.shape
        if (fluctuation_residuals is not None
                and fluctuation_features is not None):
            raise ValueError("two fluctuation feature paths were supplied")
        if fluctuation_residuals is not None or fluctuation_features is not None:
            role_features = (
                fluctuation_features
                if fluctuation_features is not None else [
                    feature + residual
                    for feature, residual in zip(
                        features, fluctuation_residuals)])
            if (not self.config.mean_fluctuation
                    or adapters is not None
                    or len(features) != len(heads)
                    or len(heads) != len(role_features)):
                raise ValueError(
                    "fluctuation role point-readout contract changed")
            channels = self.config.out_channels
            native = []
            for feature, head, role_feature in zip(
                    features, heads, role_features):
                if feature.shape != role_feature.shape:
                    raise ValueError(
                        "fluctuation role feature shape changed")
                mean_raw = F.conv2d(
                    feature, head.weight[:channels], head.bias[:channels])
                fluctuation_raw = F.conv2d(
                    role_feature, head.weight[channels:],
                    head.bias[channels:])
                # HCOB-F is a velocity-fluctuation specialization.  Pressure
                # (and any channels after u/v) keeps the exact legacy
                # fluctuation readout, so a history condition cannot change
                # pressure through this residual path.
                if self.history_operator_basis is not None and channels > 2:
                    legacy_fluctuation_raw = F.conv2d(
                        feature, head.weight[channels:],
                        head.bias[channels:])
                    fluctuation_raw = torch.cat((
                        fluctuation_raw.reshape(
                            feature.shape[0], self.config.out_steps,
                            channels, *feature.shape[-2:])[:, :, :2],
                        legacy_fluctuation_raw.reshape(
                            feature.shape[0], self.config.out_steps,
                            channels, *feature.shape[-2:])[:, :, 2:],
                    ), dim=2).reshape_as(fluctuation_raw)
                native.append(torch.cat((mean_raw, fluctuation_raw), dim=1))
            value_fusion = (
                self._laplacian_multi_scale_values
                if self.config.laplacian_point_pyramid
                else self._multi_scale_values)
            raw = value_fusion(native, (height, width))
        elif adapters is None:
            point_fusion = (
                self._laplacian_multi_scale
                if self.config.laplacian_point_pyramid else self._multi_scale)
            raw = point_fusion(features, heads, (height, width))
        else:
            if len(features) != len(heads) or len(heads) != len(adapters):
                raise ValueError("point readout adapter scales changed")
            native = [
                head(feature) + adapter(feature)
                for feature, head, adapter in zip(features, heads, adapters)]
            value_fusion = (
                self._laplacian_multi_scale_values
                if self.config.laplacian_point_pyramid
                else self._multi_scale_values)
            raw = value_fusion(native, (height, width))
        if self.config.mean_fluctuation:
            channels = self.config.out_channels
            mean_delta = raw[:, :channels].permute(0, 2, 3, 1).unsqueeze(1)
            fluctuation = raw[:, channels:].reshape(
                batch, self.config.out_steps, channels,
                height, width).permute(0, 1, 3, 4, 2)
            fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)
            residual = mean_delta + fluctuation
        else:
            residual = raw.reshape(
                batch, self.config.out_steps, self.config.out_channels,
                height, width).permute(0, 1, 3, 4, 2)
        return self._persistence(x).unsqueeze(1) + residual

    def _members_from_features(
            self, x: torch.Tensor,
            features: list[torch.Tensor],
            fluctuation_features: list[torch.Tensor] | None = None,
            use_member_adapters: bool = True,
            base_members: torch.Tensor | None = None,
            mean_profile_correction: torch.Tensor | None = None,
            ) -> torch.Tensor:
        if self.point_feature_refiner is not None:
            features = [*features[:-1], self.point_feature_refiner(features[-1])]
        members = []
        feature_adapters_enabled = bool(
            use_member_adapters and self.member_feature_adapters is not None)
        fused_readouts_enabled = bool(
            use_member_adapters
            and self.fused_member_feature_adapter_readouts is not None)
        adapters_enabled = feature_adapters_enabled or fused_readouts_enabled
        fluctuation_residuals = (
            [adapter(feature) - feature
             for adapter, feature in zip(
                 self.fluctuation_role_adapters, features)]
            if self.fluctuation_role_adapters is not None else None)

        adapter_source = (
            features if fluctuation_features is None else fluctuation_features)

        def adapted_features(index: int) -> list[torch.Tensor]:
            if not feature_adapters_enabled:
                return adapter_source
            values = []
            for adapter, feature in zip(
                    self.member_feature_adapters[index], adapter_source):
                if self.config.detach_member_adapter_context:
                    detached = feature.detach()
                    values.append(feature + (adapter(detached) - detached))
                else:
                    values.append(adapter(feature))
            return values

        if self.latent_ode_point_decoder is not None:
            persistence = self._persistence(x)
            trajectory = self.latent_ode_point_decoder.latent_trajectory(
                features)
            decoded = self.latent_ode_point_decoder.spatial_decode(
                trajectory, features)
            for index in range(self.config.ensemble_members):
                members.append(
                    self.latent_ode_point_decoder.member_prediction(
                        persistence, decoded, index))
        elif self.latent_ude_band_point_decoder is not None:
            persistence = self._persistence(x)
            trajectory = self.latent_ude_band_point_decoder.latent_trajectory(
                features)
            for index in range(self.config.ensemble_members):
                members.append(
                    self.latent_ude_band_point_decoder.member_prediction(
                        persistence, trajectory, features, index))
        elif self.temporal_point_decoder is not None:
            persistence = self._persistence(x)
            shared_temporal_features = (
                self.temporal_point_decoder.temporal_features(features)
                if not adapters_enabled else None)
            for index in range(self.config.ensemble_members):
                member_features = adapted_features(index)
                temporal_features = (
                    shared_temporal_features
                    if shared_temporal_features is not None
                    else self.temporal_point_decoder.temporal_features(
                        member_features))
                members.append(
                    self.temporal_point_decoder.member_prediction(
                        persistence, member_features,
                        temporal_features, index))
        else:
            for index, heads in enumerate(self.member_point_heads):
                member_features = adapted_features(index)
                if fused_readouts_enabled:
                    adapters = self.fused_member_feature_adapter_readouts[index]
                else:
                    adapters = (
                        None if self.member_point_head_adapters is None
                        else self.member_point_head_adapters[index])
                members.append(self._member_prediction(
                    x,
                    (features if fluctuation_features is not None
                     else member_features),
                    heads, adapters, fluctuation_residuals,
                    (member_features
                     if fluctuation_features is not None else None)))
        stacked = torch.stack(members, dim=1)
        if self.latent_ude_coarse_residual is not None:
            ude_correction = self.latent_ude_coarse_residual(
                features[0], (x.shape[2], x.shape[3]))
            if self.config.latent_ude_exclusive_coarse_fluctuation:
                uv = stacked[..., :2]
                uv_mean = uv.mean(dim=2, keepdim=True)
                fluctuation = uv - uv_mean
                batch, member_count, steps, height, width, channels = (
                    fluctuation.shape)
                native = fluctuation.reshape(
                    batch * member_count, steps, height, width,
                    channels).permute(0, 1, 4, 2, 3)
                highpass = LatentUDEBandPointDecoder._spatial_highpass(
                    native, features[0].shape[-2:]).permute(
                        0, 1, 3, 4, 2).reshape_as(fluctuation)
                fused_uv = uv_mean + highpass + ude_correction[..., :2]
                stacked = (
                    fused_uv if self.config.out_channels == 2
                    else torch.cat((fused_uv, stacked[..., 2:]), dim=-1))
            else:
                stacked = stacked + ude_correction
        if self.mean_profile_refiner is not None:
            mean_correction = (
                self.mean_profile_refiner(features[-1])
                if mean_profile_correction is None
                else mean_profile_correction)
            mean_correction = mean_correction.permute(0, 2, 3, 1)
            if self.config.out_channels > 2:
                mean_correction = F.pad(
                    mean_correction,
                    (0, self.config.out_channels - 2))
            stacked = (
                stacked
                + mean_correction[:, None, None])
        if self.phase_delta is not None:
            phase_delta = self.phase_delta(x)
            if self.config.out_channels > 2:
                phase_delta = F.pad(
                    phase_delta, (0, self.config.out_channels - 2))
            stacked = stacked + phase_delta[:, None]
        if (adapters_enabled
                and self.config.mean_preserving_member_adapters):
            if base_members is None:
                base_members = self._members_from_features(
                    x, features, fluctuation_features,
                    use_member_adapters=False,
                    mean_profile_correction=mean_profile_correction)
            adapter_residual = stacked - base_members
            adapter_residual = (
                adapter_residual
                - adapter_residual.mean(dim=1, keepdim=True))
            stacked = base_members + adapter_residual
        return stacked

    def _apply_hard_region_expert(
            self, members: torch.Tensor, features: list[torch.Tensor],
            history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.hard_region_expert is None:
            return members, None
        correction, gate = self.hard_region_expert(features[-1], history)
        if self.config.out_channels > 2:
            correction = F.pad(
                correction, (0, self.config.out_channels - 2))
        delta = correction[:, None].expand_as(members)
        bypass = delta.eq(0.0)
        return _ExactIdentityAddSTE.apply(members, delta, bypass), gate

    def _apply_conservative_closure(
            self, members: torch.Tensor, features: list[torch.Tensor],
            history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.conservative_closure is None:
            return members, None
        correction, router_weights = self.conservative_closure(
            features[-1], history)
        if self.config.out_channels > 2:
            correction = F.pad(
                correction, (0, self.config.out_channels - 2))
        delta = correction[:, None].expand_as(members)
        bypass = delta.eq(0.0)
        return (
            _ExactIdentityAddSTE.apply(members, delta, bypass),
            router_weights)

    def _apply_recurrent_latent_rollout(
            self, members: torch.Tensor,
            finest_feature: torch.Tensor) -> torch.Tensor:
        if self.recurrent_latent_rollout is None:
            return members
        if self.recurrent_latent_rollout.lowpass_modes:
            batch, member_count, steps, height, width, _channels = (
                members.shape)
            velocity = members[..., :2].reshape(
                batch * member_count, steps, height, width, 2)
            velocity = self.recurrent_latent_rollout.temporal_lowpass(
                velocity).reshape(
                    batch, member_count, steps, height, width, 2)
            members = (
                velocity if self.config.out_channels == 2
                else torch.cat((velocity, members[..., 2:]), dim=-1))
        correction = self.recurrent_latent_rollout(finest_feature)
        if self.config.out_channels > 2:
            correction = F.pad(
                correction, (0, self.config.out_channels - 2))
        delta = correction[:, None].expand_as(members)
        return _ExactIdentityAddSTE.apply(members, delta, delta.eq(0.0))

    def _energy_controlled_prediction(
            self, prediction: torch.Tensor,
            features: list[torch.Tensor],
            history: torch.Tensor | None = None) -> torch.Tensor:
        """Scale only physical u/v time fluctuations by exp(delta_logK/2).

        ``delta_logK`` is sample- and pixel-conditional but constant across the
        forecast horizon.  It is predicted only from the finest decoder
        feature.  The residual ``expm1`` form makes a zero controller an exact
        no-op while preserving temporal means mathematically.  Pressure and
        any later output channels bypass the controller bitwise.
        """
        if self.energy_controller is None:
            return prediction
        if prediction.ndim != 5 or prediction.shape[-1] < 2:
            raise ValueError(
                "prediction must have shape (B,T,H,W,C) with C >= 2")
        raw_delta = self._physical_energy_delta(
            features, history, prediction[..., :2])
        if raw_delta.shape[-2:] != prediction.shape[2:4]:
            raw_delta = F.interpolate(
                raw_delta, size=prediction.shape[2:4], mode="bilinear",
                align_corners=False)
        delta_logk = (
            self.config.physical_energy_log_limit * torch.tanh(raw_delta))
        uv = prediction[..., :2]
        uv_mean = uv.mean(dim=1, keepdim=True)
        fluctuation = uv - uv_mean
        if self.config.physical_energy_horizon_local:
            if raw_delta.shape[1] != prediction.shape[1]:
                raise ValueError("horizon-local energy length changed")
            relative_amplitude = torch.expm1(
                0.5 * delta_logk).permute(0, 1, 2, 3).unsqueeze(-1)
            candidate = fluctuation * (1.0 + relative_amplitude)
            candidate = candidate - candidate.mean(dim=1, keepdim=True)
            delta = candidate - fluctuation
            bypass = raw_delta.eq(0).unsqueeze(-1)
            controlled_uv = _ExactIdentityAddSTE.apply(uv, delta, bypass)
        else:
            relative_amplitude = torch.expm1(
                0.5 * delta_logk).permute(0, 2, 3, 1).unsqueeze(1)
            controlled_uv = uv + fluctuation * relative_amplitude
        if prediction.shape[-1] == 2:
            return controlled_uv
        return torch.cat((controlled_uv, prediction[..., 2:]), dim=-1)

    def _energy_controlled_members(
            self, members: torch.Tensor,
            features: list[torch.Tensor],
            history: torch.Tensor | None = None) -> torch.Tensor:
        """Apply one spatial energy state consistently to every member.

        Each member keeps its own temporal mean.  A zero controller takes an
        exact identity branch (including signed zero and other IEEE payloads),
        while the straight-through backward still trains the zero-initialized
        controller projection.
        """
        if self.energy_controller is None:
            return members
        if members.ndim != 6 or members.shape[-1] < 2:
            raise ValueError(
                "members must have shape (B,M,T,H,W,C) with C >= 2")
        raw_delta = self._physical_energy_delta(
            features, history, members[..., :2].mean(dim=1))
        if raw_delta.shape[-2:] != members.shape[3:5]:
            raw_delta = F.interpolate(
                raw_delta, size=members.shape[3:5], mode="bilinear",
                align_corners=False)
        delta_logk = (
            self.config.physical_energy_log_limit * torch.tanh(raw_delta))
        uv = members[..., :2]
        uv_mean = uv.mean(dim=2, keepdim=True)
        fluctuation = uv - uv_mean
        if self.config.physical_energy_horizon_local:
            if raw_delta.shape[1] != members.shape[2]:
                raise ValueError("horizon-local member energy length changed")
            relative_amplitude = torch.expm1(
                0.5 * delta_logk)[:, None, ..., None]
            candidate = fluctuation * (1.0 + relative_amplitude)
            candidate = candidate - candidate.mean(dim=2, keepdim=True)
            delta = candidate - fluctuation
            bypass = raw_delta.eq(0)[:, None, ..., None]
        else:
            relative_amplitude = torch.expm1(0.5 * delta_logk).permute(
                0, 2, 3, 1)[:, None, None]
            delta = fluctuation * relative_amplitude
            bypass = raw_delta.eq(0).permute(0, 2, 3, 1)[:, None, None]
        controlled_uv = _ExactIdentityAddSTE.apply(uv, delta, bypass)
        if members.shape[-1] == 2:
            return controlled_uv
        return torch.cat((controlled_uv, members[..., 2:]), dim=-1)

    def _physical_energy_delta(
            self, features: list[torch.Tensor],
            history: torch.Tensor | None = None,
            base_uv: torch.Tensor | None = None) -> torch.Tensor:
        if self.energy_controller is None:
            raise RuntimeError("physical energy controller is disabled")
        if self.config.physical_energy_decoder_context != "none":
            if history is None or base_uv is None:
                raise ValueError("decoder context requires input history and predicted UV")
            return self.energy_controller(features[-1], history, base_uv,
                                          self.persistence_scale * self.target_std, self.target_std)
        if getattr(self.config, "physical_energy_context_head", False):
            if history is None:
                raise ValueError("context energy head requires history")
            if base_uv is None:
                raise ValueError(
                    "context energy head requires the base prediction")
            return self.energy_controller(
                history, fluctuation_energy_context(base_uv))
        if self.config.physical_energy_history_conditioned:
            if history is None:
                raise ValueError("history-conditioned energy requires history")
            if self.config.physical_energy_multiscale:
                return self.energy_controller(features, history)
            return self.energy_controller(history)
        return self.energy_controller(
            features if self.config.physical_energy_multiscale
            else features[-1])

    @staticmethod
    def _members_with_shared_uv_mean(
            members: torch.Tensor) -> torch.Tensor:
        """Share each member's u/v temporal mean but preserve its fluctuation.

        Pressure and any later channels are returned by identity.  The
        transformation has no parameters and keeps gradients through the
        cross-member mean, so member losses jointly train the shared u/v mean
        while each member retains its own zero-time-mean role.
        """
        if members.ndim != 6 or members.shape[-1] < 2:
            raise ValueError(
                "members must have shape (B,M,T,H,W,C) with C >= 2")
        uv = members[..., :2]
        member_mean = uv.mean(dim=2, keepdim=True)
        shared_mean = member_mean.mean(dim=1, keepdim=True)
        shared_uv = uv - member_mean + shared_mean
        if members.shape[-1] == 2:
            return shared_uv
        return torch.cat((shared_uv, members[..., 2:]), dim=-1)

    def _scalar_potential_correction(
            self, feature: torch.Tensor,
            size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bounded potential and normalized u/v correction."""
        if self.scalar_potential_residual is None:
            raise RuntimeError("scalar potential residual is disabled")
        psi, physical = self.scalar_potential_residual(feature, size)
        normalized = physical / self.target_std[:2].to(
            dtype=physical.dtype).view(1, 1, 1, 1, 2)
        return psi, normalized

    def scalar_potential_residual_audit(
            self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute candidate diagnostics without retaining forward state.

        ``physical_correction`` is in target u/v units; the normalized value
        is the exact correction that is added to predictions and members.
        """
        features, _point_role, _uq_role, _mean_correction = (
            self._decode_role_features(x))
        psi, normalized = self._scalar_potential_correction(
            features[-1], (x.shape[2], x.shape[3]))
        physical = normalized * self.target_std[:2].to(
            dtype=normalized.dtype).view(1, 1, 1, 1, 2)
        return {
            "psi": psi,
            "physical_correction": physical,
            "normalized_correction": normalized,
        }

    def _apply_scalar_potential_residual(
            self, members: torch.Tensor, prediction: torch.Tensor,
            feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scalar_potential_residual is None:
            return members, prediction
        _psi, correction = self._scalar_potential_correction(
            feature, (members.shape[3], members.shape[4]))
        member_delta = correction[:, None].expand(
            -1, members.shape[1], -1, -1, -1, -1)
        member_delta = (
            F.pad(member_delta, (0, self.config.out_channels - 2))
            if self.config.out_channels > 2 else member_delta)
        members = _ExactIdentityAddSTE.apply(
            members, member_delta, member_delta.eq(0.0))
        prediction_delta = (
            F.pad(correction, (0, self.config.out_channels - 2))
            if self.config.out_channels > 2 else correction)
        prediction = _ExactIdentityAddSTE.apply(
            prediction, prediction_delta, prediction_delta.eq(0.0))
        return members, prediction

    @staticmethod
    def _flow_context(
            x: torch.Tensor, prediction: torch.Tensor, full_history: bool = False,
            sparse_history: bool = False) -> torch.Tensor:
        """Direct high-resolution motion cues for residual localization."""
        history = x[..., :2].detach()
        prediction = prediction[..., :2].detach()
        recent = history if full_history else (history[:, ::4] if sparse_history else history[:, -5:])
        differences = history[:, -6:][:, 1:] - history[:, -6:][:, :-1]
        history_stats = torch.cat((
            history.mean(dim=1),
            history.std(dim=1, unbiased=False)), dim=-1
        ).permute(0, 3, 1, 2)
        prediction_stats = torch.cat((
            prediction.mean(dim=1),
            prediction.std(dim=1, unbiased=False)), dim=-1
        ).permute(0, 3, 1, 2)
        batch, _, height, width, _ = history.shape
        yy = torch.linspace(
            -1.0, 1.0, height, device=x.device, dtype=x.dtype)
        xx = torch.linspace(
            -1.0, 1.0, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack((grid_y, grid_x)).unsqueeze(0).expand(
            batch, -1, -1, -1)
        return torch.cat((
            _flatten_forecast(recent),
            _flatten_forecast(differences),
            history_stats,
            _flatten_forecast(prediction),
            prediction_stats,
            coordinates,
        ), dim=1)

    def _fused_uq_features(
            self, features: list[torch.Tensor],
            members: torch.Tensor, x: torch.Tensor,
            prediction: torch.Tensor | None = None, *,
            detach_members: bool = True,
            fluctuation_features: list[torch.Tensor] | None = None,
            ) -> torch.Tensor:
        height, width = members.shape[3:5]
        projected = []
        for feature, projection in zip(features, self.uq_projections):
            value = projection(feature)
            if value.shape[-2:] != (height, width):
                value = F.interpolate(
                    value, size=(height, width), mode="bilinear",
                    align_corners=False)
            projected.append(value)
        if self.fluctuation_uq_projections is not None:
            if (fluctuation_features is None
                    or len(fluctuation_features)
                    != len(self.fluctuation_uq_projections)):
                raise ValueError("semantic fluctuation UQ features are missing")
            for feature, projection in zip(
                    fluctuation_features,
                    self.fluctuation_uq_projections):
                value = projection(feature)
                if value.shape[-2:] != (height, width):
                    value = F.interpolate(
                        value, size=(height, width), mode="bilinear",
                        align_corners=False)
                projected.append(value)

        member_uv = members[..., :2]
        if detach_members:
            member_uv = member_uv.detach()
        member_std = member_uv.std(dim=1, unbiased=False)
        member_range = member_uv.amax(dim=1) - member_uv.amin(dim=1)
        if self.config.disagreement_scale_fraction:
            target_std = self.target_std[:2].view(1, 1, 1, 1, 2)
            reference = member_uv.new_tensor(
                self.config.uncertainty_base_widths).view(
                    1, 1, 1, 1, 2)
            reference = (
                reference * self.config.disagreement_scale_fraction)
            member_std = torch.log1p(
                member_std * target_std / reference)
            member_range = torch.log1p(
                member_range * target_std / reference)
        disagreement = torch.cat((member_std, member_range), dim=-1)
        fusion_inputs = [*projected, _flatten_forecast(disagreement)]
        if self.config.uq_flow_context:
            fusion_inputs.append(self._flow_context(
                x, (member_uv.mean(dim=1)
                    if prediction is None else prediction), self.config.uq_full_history_context,
                self.config.uq_sparse_history_context))
        return self.uq_fusion(torch.cat(fusion_inputs, dim=1))

    def _quantiles_from_fused(
            self, fused: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        batch = members.shape[0]
        height, width = members.shape[3:5]
        raw = self.quantile_head(fused).reshape(
            batch, self.config.out_steps, 2,
            len(self.config.quantile_levels), height, width
        ).permute(0, 1, 4, 5, 2, 3)
        if self.temporal_uq_refiner is not None:
            correction = self.temporal_uq_refiner(fused, members)
            raw = raw + self.config.temporal_uq_logit_limit * torch.tanh(
                correction)

        middle = len(self.config.quantile_levels) // 2
        center = raw[..., middle:middle + 1]
        lower_gaps = F.softplus(raw[..., :middle])
        lower_distance = torch.cumsum(
            lower_gaps.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
        lower = center - lower_distance
        upper_gaps = F.softplus(raw[..., middle + 1:])
        upper = center + torch.cumsum(upper_gaps, dim=-1)
        return torch.cat((lower, center, upper), dim=-1)

    def _quantiles_from_features(
            self, features: list[torch.Tensor],
            members: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        fused = self._fused_uq_features(features, members, x)
        return self._quantiles_from_fused(fused, members)

    def _refine_utility_interval(
            self, lower: torch.Tensor, upper: torch.Tensor,
            fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.utility_refinement_head is None:
            return lower, upper
        batch, steps, height, width, channels = lower.shape
        raw = self.utility_refinement_head(fused).reshape(
            batch, steps, channels, 2, height, width
        ).permute(0, 1, 4, 5, 2, 3)
        log_scale = (
            self.config.utility_refinement_log_limit * torch.tanh(raw))
        return (
            lower * torch.exp(log_scale[..., 0]),
            upper * torch.exp(log_scale[..., 1]),
        )

    def _allocate_output_space_interval(
            self, lower: torch.Tensor, upper: torch.Tensor,
            x: torch.Tensor, prediction: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reallocate width using a hidden-coordinate-free UQ interface.

        The arithmetic mean width is preserved independently for every
        sample, horizon and velocity channel.  Consequently this module owns
        spatial/asymmetric allocation only; the upstream calibrated quantile
        tower remains responsible for the total width budget.
        """
        if self.output_space_uq_allocator is None:
            return lower, upper
        context = self._flow_context(x.detach(), prediction.detach())
        batch, steps, height, width, channels = lower.shape
        raw = self.output_space_uq_allocator(context).reshape(
            batch, steps, channels, 2, height, width
        ).permute(0, 1, 4, 5, 2, 3)
        log_scale = (
            self.config.output_space_uq_log_limit * torch.tanh(raw))
        candidate_lower = lower * torch.exp(log_scale[..., 0])
        candidate_upper = upper * torch.exp(log_scale[..., 1])
        original_budget = (lower + upper).mean(
            dim=(2, 3), keepdim=True)
        candidate_budget = (candidate_lower + candidate_upper).mean(
            dim=(2, 3), keepdim=True)
        correction = original_budget / candidate_budget.clamp_min(1.0e-8)
        return candidate_lower * correction, candidate_upper * correction

    def _calibrate_output_space_budget(
            self, lower: torch.Tensor, upper: torch.Tensor,
            x: torch.Tensor, prediction: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calibrate sample/horizon/channel width without hidden features."""
        if self.output_space_uq_budgeter is None:
            return lower, upper
        context = self._flow_context(x.detach(), prediction.detach())
        batch, steps, _height, _width, channels = lower.shape
        raw = self.output_space_uq_budgeter(context).reshape(
            batch, steps, channels, 2)
        log_scale = (
            self.config.output_space_uq_budget_log_limit
            * torch.tanh(raw)).unsqueeze(2).unsqueeze(2)
        return (
            lower * torch.exp(log_scale[..., 0]),
            upper * torch.exp(log_scale[..., 1]),
        )

    def _budgeted_disagreement_interval(
            self, lower: torch.Tensor, upper: torch.Tensor,
            members: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reallocate, but never create, mean interval-width budget.

        The base UQ tower is conditioned on collapsed members.  This final
        branch receives the real detached member spread, ranks it within each
        sample/horizon/channel, and moves the already-predicted width toward
        difficult pixels.  Arithmetic mean width over H/W is preserved for
        every sample, horizon and velocity channel.
        """
        if self.disagreement_allocation_strength is None:
            return lower, upper
        if members.ndim != 6 or members.shape[-1] < 2:
            raise ValueError(
                "members must have shape (B,M,T,H,W,C) with C >= 2")
        member_uv = members[..., :2].detach()
        target_std = self.target_std[:2].view(1, 1, 1, 1, 2)
        reference = member_uv.new_tensor(
            self.config.uncertainty_base_widths).view(1, 1, 1, 1, 2)
        physical_std = member_uv.std(dim=1, unbiased=False) * target_std
        physical_range = (
            member_uv.amax(dim=1) - member_uv.amin(dim=1)) * target_std
        signal = 0.5 * (
            torch.log1p(physical_std / reference)
            + torch.log1p(physical_range / (2.0 * reference)))
        centered = signal - signal.mean(dim=(2, 3), keepdim=True)
        variance = centered.square().mean(dim=(2, 3), keepdim=True)
        standardized = centered / torch.sqrt(variance + 1.0e-4)
        sharpness = torch.exp(
            self.disagreement_allocation_sharpness.clamp(-2.0, 2.0)
        ).view(1, self.config.out_steps, 1, 1, 2)
        localized = torch.tanh(sharpness * standardized)
        strength = (
            self.config.disagreement_allocation_log_limit
            * torch.tanh(self.disagreement_allocation_strength)
        ).view(1, self.config.out_steps, 1, 1, 2)
        raw_scale = torch.exp(strength * localized)
        base_width = (lower + upper).clamp_min(0.0)
        base_budget = base_width.mean(dim=(2, 3), keepdim=True)
        proposed_budget = (
            base_width * raw_scale).mean(dim=(2, 3), keepdim=True)
        budget_correction = torch.where(
            base_budget > 0.0,
            base_budget / proposed_budget.clamp_min(
                torch.finfo(proposed_budget.dtype).tiny),
            torch.ones_like(base_budget))
        scale = raw_scale * budget_correction
        bypass = strength.eq(0.0).expand_as(lower)
        return (
            _ExactIdentityAddSTE.apply(
                lower, lower * (scale - 1.0), bypass),
            _ExactIdentityAddSTE.apply(
                upper, upper * (scale - 1.0), bypass),
        )

    @staticmethod
    def _ordered_interval_extents(
            lower: torch.Tensor,
            upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Order residual endpoints without discarding interval asymmetry.

        Residual quantiles intentionally allow signed extents. Independent
        positive rescaling of both sides can therefore very rarely move the
        residual lower endpoint ``-lower`` above the upper endpoint
        ``upper``. Swapping those endpoints maps ``(lower, upper)`` to
        ``(-upper, -lower)`` and guarantees a non-negative total width while
        leaving every already-valid element exactly unchanged.
        """
        invalid = lower + upper < 0.0
        return (
            torch.where(invalid, -upper, lower),
            torch.where(invalid, -lower, upper),
        )

    def _interval_candidates(
            self, quantiles: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor]:
        levels = quantiles.new_tensor(self.config.quantile_levels)
        lower_candidates = []
        upper_candidates = []
        valid_candidates = []
        masses = []
        for lower_index, upper_index in self.interval_candidate_pairs:
            lower_quantile = quantiles[..., lower_index]
            upper_quantile = quantiles[..., upper_index]
            if self.config.residual_quantile_intervals:
                # Extents may be signed: prediction - lower_extent is the
                # learned lower residual quantile, and prediction +
                # upper_extent is the learned upper residual quantile.
                # Their sum remains the strictly positive interval width.
                lower_width = -lower_quantile
                upper_width = upper_quantile
                valid = torch.ones_like(lower_width, dtype=torch.bool)
            else:
                lower_width = (-lower_quantile).clamp_min(1e-6)
                upper_width = upper_quantile.clamp_min(1e-6)
                valid = (
                    (lower_quantile <= 0.0) & (upper_quantile >= 0.0))
            lower_candidates.append(lower_width)
            upper_candidates.append(upper_width)
            valid_candidates.append(valid)
            masses.append(levels[upper_index] - levels[lower_index])
        return (
            torch.stack(lower_candidates, dim=-1),
            torch.stack(upper_candidates, dim=-1),
            torch.stack(valid_candidates, dim=-1),
            torch.stack(masses))

    def _coverage_logits(self, fused: torch.Tensor) -> torch.Tensor | None:
        if self.coverage_calibration_head is None:
            return None
        batch, _, height, width = fused.shape
        raw = self.coverage_calibration_head(fused)
        levels = raw.new_tensor(self.config.quantile_levels)
        masses = torch.stack([
            levels[upper] - levels[lower]
            for lower, upper in self.interval_candidate_pairs])
        base_logits = torch.logit(masses.clamp(1e-5, 1.0 - 1e-5))
        if self.config.coverage_calibration_mode == "temperature":
            correction = raw.reshape(
                batch, self.config.out_steps, 2, 2, height, width
            ).permute(0, 1, 4, 5, 2, 3)
            bias = (
                self.config.coverage_calibration_logit_limit
                * torch.tanh(correction[..., 0:1]))
            slope = torch.exp(
                self.config.coverage_calibration_temperature_limit
                * torch.tanh(correction[..., 1:2]))
            return base_logits * slope + bias
        correction = raw.reshape(
            batch, self.config.out_steps, 2,
            len(self.interval_candidate_pairs), height, width
        ).permute(0, 1, 4, 5, 2, 3)
        return (
            base_logits
            + self.config.coverage_calibration_logit_limit
            * torch.tanh(correction))

    def _residual_mixture_from_fused(
            self, fused: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Predict a Gaussian mixture in channel-specific physical units.

        Means and scales are expressed relative to the fixed u/v SPS
        references.  This makes one proper-scoring objective usable for both
        velocity channels without letting the larger u scale dominate v.
        """
        if self.residual_mixture_head is None:
            return None
        batch, _channels, height, width = fused.shape
        components = self.config.residual_mixture_components
        raw = self.residual_mixture_head(fused).reshape(
            batch, self.config.out_steps, 2, components, 3, height, width
        ).permute(0, 1, 5, 6, 2, 3, 4)
        logits = raw[..., 0]
        means = (
            self.config.residual_mixture_mean_limit
            * torch.tanh(raw[..., 1]))
        scales = (
            self.config.residual_mixture_scale_floor
            + F.softplus(raw[..., 2]))
        return logits, means, scales

    @staticmethod
    def _gaussian_mixture_cdf(
            value: torch.Tensor, logits: torch.Tensor,
            means: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Evaluate a diagonal scalar Gaussian mixture CDF."""
        standardized = (
            value.unsqueeze(-1) - means) / scales.clamp_min(1.0e-6)
        component_cdf = 0.5 * (
            1.0 + torch.erf(standardized / (2.0 ** 0.5)))
        return (torch.softmax(logits, dim=-1) * component_cdf).sum(dim=-1)

    def _mixture_utility_interval(
            self, quantiles: torch.Tensor,
            mixture: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            probability_shrinkage: float | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select existing endpoints using learned conditional probability.

        The endpoint bank remains the stable monotone quantile bank.  The
        mixture replaces its unconditional nominal mass with a coherent
        conditional residual probability, shrunk toward nominal mass to avoid
        fitting a target-visible per-sample oracle.
        """
        shrinkage = (
            self.config.residual_mixture_probability_shrinkage
            if probability_shrinkage is None else float(probability_shrinkage))
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("mixture probability shrinkage must lie in [0, 1]")
        if shrinkage == 0.0:
            return self._utility_interval(quantiles)
        logits, means, scales = mixture
        target_std = self.target_std[:2].view(1, 1, 1, 1, 2, 1)
        reference = quantiles.new_tensor([0.01075, 0.00550]).view(
            1, 1, 1, 1, 2, 1)
        lower_stack, upper_stack, valid_stack, masses = (
            self._interval_candidates(quantiles))
        # Evaluate a few candidate intervals at a time.  Materializing
        # (B,T,H,W,C,candidates,components) for the full bank raises the 4090
        # peak enough to kill an otherwise valid B210 evaluation batch.
        best_utility = lower_stack[..., 0].new_full((), -1.0).expand_as(
            lower_stack[..., 0]).clone()
        best_lower = lower_stack[..., 0]
        best_upper = upper_stack[..., 0]
        mixture_logits = logits.unsqueeze(-2)
        mixture_means = means.unsqueeze(-2)
        mixture_scales = scales.unsqueeze(-2)
        for start in range(0, lower_stack.shape[-1], 3):
            stop = min(start + 3, lower_stack.shape[-1])
            lower_chunk = lower_stack[..., start:stop]
            upper_chunk = upper_stack[..., start:stop]
            lower_residual = -lower_chunk * target_std / reference
            upper_residual = upper_chunk * target_std / reference
            conditional_mass = (
                self._gaussian_mixture_cdf(
                    upper_residual, mixture_logits,
                    mixture_means, mixture_scales)
                - self._gaussian_mixture_cdf(
                    lower_residual, mixture_logits,
                    mixture_means, mixture_scales)
            ).clamp(0.0, 1.0)
            nominal_mass = masses[start:stop].view(1, 1, 1, 1, 1, -1)
            probability = nominal_mass.lerp(conditional_mass, shrinkage)
            physical_width = (lower_chunk + upper_chunk) * target_std
            utility = probability * torch.exp(-physical_width / SPS_SIGMA)
            utility = torch.where(
                valid_stack[..., start:stop], utility,
                utility.new_full((), -1.0))
            chunk_utility, chunk_index = utility.max(dim=-1)
            gather_index = chunk_index.unsqueeze(-1)
            chunk_lower = lower_chunk.gather(-1, gather_index).squeeze(-1)
            chunk_upper = upper_chunk.gather(-1, gather_index).squeeze(-1)
            improve = chunk_utility > best_utility
            best_utility = torch.where(improve, chunk_utility, best_utility)
            best_lower = torch.where(improve, chunk_lower, best_lower)
            best_upper = torch.where(improve, chunk_upper, best_upper)
        return best_lower, best_upper

    def _residual_mixture_quantiles(
            self,
            mixture: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> torch.Tensor:
        """Invert the learned mixture CDF at the native quantile levels.

        Levels are processed one at a time to keep the B210 evaluation peak
        below 4090 memory.  This path is a research oracle-geometry probe; a
        successful endpoint bank should later be distilled into a direct head
        before packaging.
        """
        logits, means, scales = mixture
        lower_bound = (means - 6.0 * scales).amin(dim=-1)
        upper_bound = (means + 6.0 * scales).amax(dim=-1)
        quantiles = []
        for level in self.config.quantile_levels:
            low = lower_bound
            high = upper_bound
            target = float(level)
            for _iteration in range(
                    self.config.residual_mixture_bisection_steps):
                middle = 0.5 * (low + high)
                probability = self._gaussian_mixture_cdf(
                    middle, logits, means, scales)
                low = torch.where(probability < target, middle, low)
                high = torch.where(probability < target, high, middle)
            quantiles.append(0.5 * (low + high))
        reference = logits.new_tensor([0.01075, 0.00550]).view(
            1, 1, 1, 1, 2, 1)
        target_std = self.target_std[:2].view(1, 1, 1, 1, 2, 1)
        return torch.stack(quantiles, dim=-1) * reference / target_std

    def _utility_interval(
            self, quantiles: torch.Tensor,
            coverage_logits: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        target_std = self.target_std[:2].view(1, 1, 1, 1, 2, 1)
        lower_stack, upper_stack, valid_stack, masses = (
            self._interval_candidates(quantiles))
        probability = (
            masses.view(1, 1, 1, 1, 1, -1)
            if coverage_logits is None
            else torch.sigmoid(coverage_logits))
        physical_width = (lower_stack + upper_stack) * target_std
        utility_stack = probability * torch.exp(
            -physical_width / SPS_SIGMA)
        utility_stack = torch.where(
            valid_stack, utility_stack, utility_stack.new_full((), -1.0))
        selected = utility_stack.argmax(dim=-1, keepdim=True)
        lower = lower_stack.gather(-1, selected).squeeze(-1)
        upper = upper_stack.gather(-1, selected).squeeze(-1)
        return lower, upper

    def forward_uq_training(
            self, x: torch.Tensor, *, return_mixture: bool = False,
            mixture_probability_shrinkage: float | None = None,
            apply_mixture_interval: bool = True,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor, list[torch.Tensor],
                       torch.Tensor | None]:
        (features, fluctuation_features, uq_role_features,
         mean_profile_correction) = (
            self._decode_role_features(x))
        if self.config.mean_preserving_member_adapters:
            point_members = self._members_from_features(
                x, features, fluctuation_features,
                use_member_adapters=False,
                mean_profile_correction=mean_profile_correction)
            raw_members = self._members_from_features(
                x, features, fluctuation_features,
                base_members=point_members,
                mean_profile_correction=mean_profile_correction)
        else:
            raw_members = self._members_from_features(
                x, features, fluctuation_features,
                mean_profile_correction=mean_profile_correction)
            point_members = raw_members
        if raw_members is point_members:
            point_members, _hard_gate = self._apply_hard_region_expert(
                point_members, features, x)
            raw_members = point_members
        else:
            point_members, _hard_gate = self._apply_hard_region_expert(
                point_members, features, x)
            raw_members, _raw_hard_gate = self._apply_hard_region_expert(
                raw_members, features, x)
        if raw_members is point_members:
            point_members, _closure_weights = self._apply_conservative_closure(
                point_members, features, x)
            raw_members = point_members
        else:
            point_members, _closure_weights = self._apply_conservative_closure(
                point_members, features, x)
            raw_members, _raw_closure_weights = (
                self._apply_conservative_closure(raw_members, features, x))
        if raw_members is point_members:
            point_members = self._apply_recurrent_latent_rollout(
                point_members, features[-1])
            raw_members = point_members
        else:
            point_members = self._apply_recurrent_latent_rollout(
                point_members, features[-1])
            raw_members = self._apply_recurrent_latent_rollout(
                raw_members, features[-1])
        dynamics_features = (
            features if fluctuation_features is None
            else fluctuation_features)
        if self.config.physical_energy_member_sync:
            raw_members = self._energy_controlled_members(
                raw_members, dynamics_features, x)
            point_members = self._energy_controlled_members(
                point_members, dynamics_features, x)
            prediction = point_members.mean(dim=1)
        else:
            prediction = self._energy_controlled_prediction(
                point_members.mean(dim=1), dynamics_features, x)
        if self.temporal_transport is not None:
            prediction = self.temporal_transport(
                prediction, features[-1], x)
        members = (
            self._members_with_shared_uv_mean(raw_members)
            if self.config.shared_member_uv_mean else raw_members)
        if self.decoupled_member_residual is not None:
            members = self.decoupled_member_residual(
                members, features[-1])
        members, prediction = self._apply_scalar_potential_residual(
            members, prediction, features[-1])
        uq_features = (
            [feature.detach() for feature in features]
            if self.config.detach_uq_context else features)
        uq_fluctuation_features = (
            None if uq_role_features is None else (
                [feature.detach() for feature in uq_role_features]
                if self.config.detach_uq_context
                else uq_role_features))
        uq_members = (
            members.detach() if self.config.detach_uq_context else members)
        uq_conditioning_members = uq_members
        if self.config.budgeted_disagreement_allocation:
            uq_conditioning_members = (
                uq_members.mean(dim=1, keepdim=True).expand_as(uq_members))
        uq_history = x.detach() if self.config.detach_uq_context else x
        fused = self._fused_uq_features(
            uq_features, uq_conditioning_members, uq_history,
            prediction=(
                prediction.detach()
                if (self.config.detach_uq_context
                    and self.config.shared_member_uv_mean) else (
                    prediction
                    if self.config.shared_member_uv_mean else None)),
            fluctuation_features=uq_fluctuation_features)
        quantiles = self._quantiles_from_fused(
            fused, uq_conditioning_members)
        coverage_logits = self._coverage_logits(fused)
        residual_mixture = self._residual_mixture_from_fused(fused)
        if residual_mixture is None or not apply_mixture_interval:
            lower, upper = self._utility_interval(
                quantiles, coverage_logits)
        elif self.config.residual_mixture_endpoint_mode in (
                "mixture_quantiles", "mixture_projected"):
            endpoint_shrinkage = (
                self.config.residual_mixture_endpoint_shrinkage)
            if endpoint_shrinkage == 0.0:
                lower, upper = self._utility_interval(quantiles)
            else:
                mixture_quantiles = self._residual_mixture_quantiles(
                    residual_mixture)
                if self.config.residual_mixture_endpoint_mode == "mixture_quantiles":
                    selection_quantiles = quantiles.lerp(
                        mixture_quantiles, endpoint_shrinkage)
                    lower, upper = self._utility_interval(selection_quantiles)
                else:
                    native_lower, native_upper = self._utility_interval(
                        quantiles)
                    mixture_lower, mixture_upper = self._utility_interval(
                        mixture_quantiles)
                    native_width = native_lower + native_upper
                    mixture_width = mixture_lower + mixture_upper
                    if (self.config.residual_mixture_geometry_projection
                            == "pixel_width"):
                        scale = native_width / mixture_width.clamp_min(1.0e-6)
                    else:
                        native_budget = native_width.mean(
                            dim=(2, 3), keepdim=True)
                        mixture_budget = mixture_width.mean(
                            dim=(2, 3), keepdim=True)
                        scale = native_budget / mixture_budget.clamp_min(1.0e-6)
                    projected_lower = mixture_lower * scale
                    projected_upper = mixture_upper * scale
                    lower = native_lower.lerp(
                        projected_lower, endpoint_shrinkage)
                    upper = native_upper.lerp(
                        projected_upper, endpoint_shrinkage)
        else:
            lower, upper = self._mixture_utility_interval(
                quantiles, residual_mixture,
                probability_shrinkage=mixture_probability_shrinkage)
        lower, upper = self._refine_utility_interval(
            lower, upper, fused)
        lower, upper = self._ordered_interval_extents(lower, upper)
        lower, upper = self._allocate_output_space_interval(
            lower, upper, uq_history, prediction)
        lower, upper = self._calibrate_output_space_budget(
            lower, upper, uq_history, prediction)
        # Signed residual extents can individually be negative.  Independent
        # positive allocation scales may therefore invert an otherwise valid
        # endpoint pair at isolated pixels; reorder after allocation as well.
        lower, upper = self._ordered_interval_extents(lower, upper)
        lower, upper = self._budgeted_disagreement_interval(
            lower, upper, uq_members)
        values = (prediction, members, quantiles, lower, upper, features,
                  coverage_logits)
        return values + (residual_mixture,) if return_mixture else values

    def forward_quantiles_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        return self.forward_uq_training(x)[:6]

    def forward_quantiles(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor]:
        values = self.forward_quantiles_with_features(x)
        return values[:5]

    def forward_distribution_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       list[torch.Tensor]]:
        prediction, _members, _quantiles, lower, upper, features = (
            self.forward_quantiles_with_features(x))
        return prediction, lower, upper, features

    def forward_distribution(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, lower, upper, _features = (
            self.forward_distribution_with_features(x))
        return prediction, lower, upper

    def forward_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        (features, fluctuation_features, _uq_role_features,
         mean_profile_correction) = (
            self._decode_role_features(x))
        members = self._members_from_features(
            x, features, fluctuation_features,
            mean_profile_correction=mean_profile_correction)
        members, _hard_gate = self._apply_hard_region_expert(
            members, features, x)
        members, _closure_weights = self._apply_conservative_closure(
            members, features, x)
        members = self._apply_recurrent_latent_rollout(
            members, features[-1])
        dynamics_features = (
            features if fluctuation_features is None
            else fluctuation_features)
        if self.config.physical_energy_member_sync:
            members = self._energy_controlled_members(
                members, dynamics_features, x)
            prediction = members.mean(dim=1)
        else:
            prediction = self._energy_controlled_prediction(
                members.mean(dim=1), dynamics_features, x)
        if self.temporal_transport is not None:
            prediction = self.temporal_transport(
                prediction, features[-1], x)
        members, prediction = self._apply_scalar_potential_residual(
            members, prediction, features[-1])
        return prediction, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _features = self.forward_with_features(x)
        return prediction


def build_ensemble_unet2d(**overrides) -> EnsembleUNet2d:
    return EnsembleUNet2d(EnsembleUNet2dConfig(**overrides))
