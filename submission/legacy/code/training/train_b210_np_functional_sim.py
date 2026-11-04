#!/usr/bin/env python3
"""Train the bounded B210 np-user functional-approximation Sim bundle.

This is a train-from-random-initialization Sim experiment.  It deliberately
does not read a C208 checkpoint or any Real held labels.  A paired update-200
control checks the mechanics; only the candidate can continue to update 10000.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

import train_c208_real_sim_rehearsal as rehearsal
import train_c208_sim_hgrad_pretrain as sim_hgrad
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
from physics.closure_augmentation import ClosureResidualAugmentor
from physics.parameter_mix_augmentation import (
    ParameterMixContract,
    PhaseAlignedParameterMixAugmentor,
)
from probabilistic_objectives import (
    conditional_distribution_targets, functional_energy_score,
    pointwise_ensemble_crps)


METHOD = "b210_np_functional_sim_v3"
REAL_CLOSURE_METHOD = "b210_np_functional_sim_realclosure_v1"
REAL_CLOSURE_EQUIVARIANCE_METHOD = (
    "b210_np_functional_sim_realclosure_equivariance_v1")
REAL_CLOSURE_ACCELERATION_METHOD = (
    "b210_np_functional_sim_realclosure_acceleration_v1")
REAL_CLOSURE_PHASE_DELTA_METHOD = (
    "b210_np_functional_sim_realclosure_phase_delta_v1")
REAL_CLOSURE_SEED1_METHOD = (
    "b210_np_functional_sim_realclosure_seed1_soup_member_v1")
PHASE_ALIGNED_RE_MIX_METHOD = (
    "b210_np_functional_sim_phase_aligned_re_mix_v1")
PHASE_ALIGNED_GRID_MIX_METHOD = (
    "b210_np_functional_sim_phase_aligned_parameter_grid_mix_v1")
MULTISCALE_TKE_METHOD = (
    "b210_np_functional_sim_realclosure_multiscale_tke_v1")
MULTISCALE_ENERGY_METHOD = (
    "b210_np_functional_sim_realclosure_multiscale_energy_v1")
HORIZON_LOCAL_ENERGY_METHOD = (
    "b210_np_functional_sim_realclosure_horizon_local_energy_v1")
HISTORY_ENERGY_METHOD = (
    "b210_np_functional_sim_realclosure_history_energy_v1")
DIRECT_HISTORY_TKE_METHOD = (
    "b210_np_functional_sim_realclosure_direct_history_tke_v1")
HYBRID_HISTORY_MULTISCALE_ENERGY_METHOD = (
    "b210_np_functional_sim_realclosure_hybrid_history_multiscale_energy_v1")
CROSS_CONDITIONED_BACKBONE_METHOD = (
    "b210_np_functional_sim_realclosure_cross_conditioned_backbone_v1")
TEMPORAL_POINT_DECODER_METHOD = (
    "b210_np_functional_sim_realclosure_temporal_point_decoder_v1")
TEMPORAL_AMPLITUDE_FACTORIZATION_METHOD = (
    "b210_np_functional_sim_realclosure_temporal_amplitude_factorization_v1")
ISOENERGETIC_TEMPORAL_TRANSPORT_METHOD = (
    "b210_np_functional_sim_realclosure_isoenergetic_temporal_transport_v1")
TEMPORAL_BAND_MIXER_METHOD = (
    "b210_np_functional_sim_standard_xcb_member_temporal_band_mixer_v1")
OFFICIAL_POINT_TEMPORAL_TRANSPORT_METHOD = (
    "b210_np_functional_sim_realclosure_isoenergetic_transport_"
    "officialpoint_v1")
OFFICIAL_POINT_TKE_FLOOR_METHOD = (
    "b210_np_functional_sim_realclosure_isoenergetic_transport_"
    "officialpoint_controller_tkefloor_v1")
PARETO_TKE_PROJECTION_METHOD = (
    "b210_np_functional_sim_realclosure_officialpoint_"
    "output_pareto_tke_v1")
LEGACY_TKE_GRADIENT_FLOOR_METHOD = (
    "b210_np_functional_sim_realclosure_officialpoint_"
    "legacy_tke_gradient_floor_v1")
MEMBER_INTEGRATED_TKE_GRADIENT_FLOOR_METHOD = (
    "b210_np_functional_sim_realclosure_officialpoint_"
    "member_integrated_legacy_tke_gradient_floor_v1")
MEMBER_OFFICIAL_MEAN_TKE_FLOOR_METHOD = (
    "b210_np_functional_sim_realclosure_officialpoint_"
    "member_official_mean_legacy_tke_floor_v1")
DECOUPLED_MEMBER_RESIDUAL_METHOD = (
    "b210_np_functional_sim_realclosure_officialpoint_"
    "decoupled_zero_mean_member_residual_v1")
MEMBER_FEATURE_ADAPTER_METHOD = (
    "b210_np_functional_sim_realclosure_officialpoint_"
    "multiscale_member_feature_adapters_v1")
CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD = (
    "b210_np_functional_sim_realclosure_officialpoint_"
    "cross_conditioned_multiscale_member_adapters_v1")
STANDARD_CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD = (
    "b210_np_functional_sim_realclosure_standardobjective_"
    "cross_conditioned_multiscale_member_adapters_v2")
STANDARD_XCB_MEMBER_MULTISCALE_ENERGY_METHOD = (
    "b210_np_functional_sim_realclosure_standardobjective_"
    "cross_conditioned_member_adapters_multiscale_energy_v3")
STANDARD_XCB_MEMBER_FLUCTUATION_ROLE_METHOD = (
    "b210_np_functional_sim_realclosure_standardobjective_"
    "xcb_member_fluctuation_role_tower_v1")
CONDITIONAL_FUTURE_ENERGY_METHOD = (
    "b210_np_functional_sim_realclosure_conditional_future_energy_v1")
CONDITIONAL_FUTURE_MEMBER_ONLY_METHOD = (
    "b210_np_functional_sim_realclosure_conditional_future_member_only_v2")
CONDITIONAL_FUTURE_ADAPTER_ONLY_METHOD = (
    "b210_np_functional_sim_realclosure_conditional_future_adapter_only_v3")
CONDITIONAL_FUTURE_CRPS_MEAN_PRESERVING_METHOD = (
    "b210_np_functional_sim_realclosure_mean_preserving_pointwise_crps_v4")
CONDITIONAL_FUTURE_CRPS_DETACHED_CONTEXT_METHOD = (
    "b210_np_functional_sim_realclosure_mean_preserving_pointwise_crps_"
    "detached_adapter_context_v5")
ISOLATED_DISTRIBUTION_TOWER_METHOD = (
    "b210_np_functional_sim_realclosure_fully_isolated_distribution_tower_v6")
BUDGETED_DISAGREEMENT_ALLOCATION_METHOD = (
    "b210_np_functional_sim_realclosure_budgeted_disagreement_allocator_v7")
PASS_STATUS = "PASS_B210_NP_SIM10000_FOR_REAL_FT12"
HEALTH_PASS_STATUS = "PASS_B210_NP_SIM_U200_HEALTH_CONTINUE"
HEALTH_STOP_STATUS = "STOP_B210_NP_SIM_U200_HEALTH"
RUNTIME_STOP_STATUS = "STOP_B210_NP_SIM_RUNTIME"
WALL_STOP_STATUS = "STOP_B210_NP_SIM_WALL_BUDGET"

SEED = 1
CONTROL_UPDATES = 200
TERMINAL_UPDATE = 10_000
WALL_BUDGET_SECONDS = 6 * 60 * 60
ANCHOR_SIZE = 128
BATCH_SIZE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
LOG_INTERVAL = 500

BACKBONE_DROPOUT = 0.20
SOFT_SPS_WEIGHT = 0.05
SOFT_SPS_WARMUP_UPDATES = 1500
P1_DISTILL_WEIGHT = rehearsal.FORMAL_P1_DISTILL_WEIGHT
HGRAD_WEIGHT = sim_hgrad.HGRAD_WEIGHT
HGRAD_RAMP_UPDATES = sim_hgrad.HGRAD_RAMP_UPDATES
TKE_AUX_WEIGHT = 0.020
TKE_AUX_UPDATES = 1500
TKE_AUX_PERSISTENT_FLOOR_WEIGHT = 0.004
TKE_GRADIENT_RATIO_RANGE = (0.05, 0.50)
TKE_AUX_CALIBRATION_SOURCE_MEDIAN = 0.04955256027147056
TKE_AUX_CALIBRATION_TARGET = 0.10
MULTISCALE_TKE_WEIGHT = 0.005
MULTISCALE_TKE_PHYSICAL_SCALE = 0.010
MULTISCALE_TKE_LEVEL_WEIGHTS = (0.25, 0.50, 1.00)
MULTISCALE_TKE_GRADIENT_RATIO_RANGE = (0.005, 0.25)
MULTISCALE_TKE_HIDDEN_CHANNELS = 32
MULTISCALE_TKE_GRADIENT_AUDIT_UPDATE = 2
DIRECT_HISTORY_TKE_WEIGHT = 0.010
DIRECT_HISTORY_TKE_LOG_EPSILON = 1e-8

EXPECTED_PARAMETERS = 178_705_575
EXPECTED_CONTROLLER_PARAMETERS = 7_169
EXPECTED_MULTISCALE_ENERGY_PARAMETERS = 178_747_207
EXPECTED_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS = 48_801
EXPECTED_HORIZON_LOCAL_ENERGY_PARAMETERS = 178_706_202
EXPECTED_HORIZON_LOCAL_ENERGY_CONTROLLER_PARAMETERS = 7_796
EXPECTED_HISTORY_ENERGY_PARAMETERS = 178_699_303
EXPECTED_HISTORY_ENERGY_CONTROLLER_PARAMETERS = 897
EXPECTED_HYBRID_HISTORY_MULTISCALE_ENERGY_PARAMETERS = 178_748_104
EXPECTED_HYBRID_HISTORY_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS = 49_698
EXPECTED_CROSS_CONDITIONED_BACKBONE_PARAMETERS = 178_903_203
EXPECTED_CROSS_CONDITIONER_PARAMETERS = 197_628
EXPECTED_TEMPORAL_POINT_DECODER_PARAMETERS = 178_395_195
EXPECTED_TEMPORAL_POINT_PARAMETERS = 60_816
EXPECTED_TEMPORAL_AMPLITUDE_FACTORIZATION_PARAMETERS = 178_412_871
EXPECTED_TEMPORAL_AMPLITUDE_PARAMETERS = 78_492
EXPECTED_TEMPORAL_TRANSPORT_PARAMETERS = 178_713_353
EXPECTED_TEMPORAL_TRANSPORT_MODULE_PARAMETERS = 7_778
EXPECTED_DECOUPLED_MEMBER_RESIDUAL_PARAMETERS = 178_883_031
EXPECTED_DECOUPLED_MEMBER_RESIDUAL_MODULE_PARAMETERS = 177_456
EXPECTED_MEMBER_FEATURE_ADAPTER_PARAMETERS = 180_355_743
EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS = 1_650_168
EXPECTED_BUDGETED_DISAGREEMENT_PARAMETERS = 180_355_823
EXPECTED_BUDGETED_DISAGREEMENT_MODULE_PARAMETERS = 80
EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS = 180_553_371
EXPECTED_FLUCTUATION_ROLE_MODULE_PARAMETERS = 412_542
EXPECTED_XCB_MEMBER_FLUCTUATION_ROLE_PARAMETERS = 180_965_913
EXPECTED_XCB_MEMBER_MULTISCALE_ENERGY_PARAMETERS = 180_595_003
EXPECTED_XCB_MEMBER_TEMPORAL_BAND_PARAMETERS = 180_598_379
EXPECTED_TEMPORAL_BAND_MODULE_PARAMETERS = 45_008
EXPECTED_SPATIAL_TRANSPORT_MODULE_PARAMETERS = 45_800
TEACHER_CACHE_SHA256 = rehearsal.FORMAL_P1_CACHE_SHA256
REAL_STATS_SHA256 = rehearsal.FORMAL_REAL_STATS_SHA256
RESEARCH_CACHE_ACKNOWLEDGEMENT = "B210_EQUIVALENT_P1_CACHE_RESEARCH_ONLY"

CLEAN_REAL_MANIFEST_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
REAL_CLOSURE_PROBABILITY = 0.50
REAL_CLOSURE_AMPLITUDE = 0.05
REAL_CLOSURE_DECAY = 0.85
REAL_CLOSURE_PERTURBATION_DECAY = 0.50
REAL_CLOSURE_ADVECTION_CELLS = 0.50
REAL_CLOSURE_CANDIDATE_COUNT = 3
REAL_CLOSURE_EQUIVARIANCE_WEIGHT = 1.0
CLOSURE_ACCELERATION_STEPS = 0
PHASE_DELTA_HIDDEN_CHANNELS = 0
PHASE_DELTA_STEPS = 5
CROSS_CONDITIONED_STEPS = 5
CROSS_CONDITIONED_CHANNELS = 64
TEMPORAL_POINT_CHANNELS = 24
TEMPORAL_POINT_BLOCKS = 2
TEMPORAL_POINT_KERNEL = 5
TEMPORAL_POINT_AMPLITUDE_LIMIT = 3.0
TEMPORAL_TRANSPORT_CHANNELS = 24
TEMPORAL_TRANSPORT_BLOCKS = 2
TEMPORAL_TRANSPORT_KERNEL = 5
TEMPORAL_TRANSPORT_MAX_DISPLACEMENT = 1.5
TEMPORAL_BAND_CHANNELS = 32
TEMPORAL_BAND_COUNT = 4
TEMPORAL_BAND_LOG_GAIN_LIMIT = 0.50
TEMPORAL_BAND_PHASE_LIMIT = 0.50
DECOUPLED_MEMBER_RESIDUAL_CHANNELS = 64
DECOUPLED_MEMBER_RESIDUAL_BLOCKS = 2
MEMBER_FEATURE_ADAPTER_CHANNELS = 64
FLUCTUATION_ROLE_CHANNELS = 64
OFFICIAL_POINT_ACCURACY_WEIGHTS = (46.0, 10.0, 9.0)
CONDITIONAL_FUTURE_PROBABILITY = 1.0
CONDITIONAL_FUTURE_AMPLITUDE = 0.05
CONDITIONAL_FUTURE_ENERGY_WEIGHT = 0.05
CONDITIONAL_FUTURE_CRPS_WEIGHT = 0.05
CONDITIONAL_FUTURE_MEMBER_MSE_WEIGHT = 0.05
CONDITIONAL_FUTURE_RNG_OFFSET = 1_000_003

# The directional diagnostic is intentionally not a hard u200 stop: dropout,
# hgrad, and the early TKE controller can need more than 200 updates to settle.
DIRECTIONAL_TKE_DELTA_MIN = 0.00
DIRECTIONAL_HGRAD_ADVANTAGE_MIN = 0.01
DIRECTIONAL_POINT_DELTA_FLOOR = -0.10
DIRECTIONAL_COMPONENT_DELTA_FLOOR = -0.15
DIRECTIONAL_SPS_DELTA_FLOOR = -0.01
DIRECTIONAL_P1_RATIO_MAX = 1.02

# Health floors only catch a broken run, not a weak short-prefix trend.
CATASTROPHIC_POINT_DELTA_FLOOR = -1.00
CATASTROPHIC_COMPONENT_DELTA_FLOOR = -1.50
CATASTROPHIC_SPS_DELTA_FLOOR = -1.00

POINT_COMPONENT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")


class MultiScaleTKEDeepSupervisor(nn.Module):
    """Training-only decoder heads for future physical TKE maps."""

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        if len(channels) != len(MULTISCALE_TKE_LEVEL_WEIGHTS):
            raise ValueError("multiscale TKE channel/level contract changed")
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, MULTISCALE_TKE_HIDDEN_CHANNELS, 1),
                nn.SiLU(),
                nn.Conv2d(
                    MULTISCALE_TKE_HIDDEN_CHANNELS,
                    MULTISCALE_TKE_HIDDEN_CHANNELS, 3, padding=1,
                    groups=MULTISCALE_TKE_HIDDEN_CHANNELS),
                nn.SiLU(),
                nn.Conv2d(MULTISCALE_TKE_HIDDEN_CHANNELS, 1, 1),
            )
            for width in channels)

    @staticmethod
    def physical_tke_map(target_physical_uv: torch.Tensor) -> torch.Tensor:
        if target_physical_uv.ndim != 5 or target_physical_uv.shape[-1] != 2:
            raise ValueError("multiscale TKE target must be [B,T,H,W,2]")
        velocity = target_physical_uv.float()
        fluctuation = velocity - velocity.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)

    def forward(
            self, features: Sequence[torch.Tensor],
            target_physical_uv: torch.Tensor,
            ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(features) != len(self.heads):
            raise ValueError("multiscale TKE feature/head contract changed")
        target_tke = self.physical_tke_map(target_physical_uv).unsqueeze(1)
        losses = []
        for feature, head in zip(features, self.heads):
            target = F.adaptive_avg_pool2d(target_tke, feature.shape[-2:])
            target = torch.log1p(target / MULTISCALE_TKE_PHYSICAL_SCALE)
            losses.append(F.smooth_l1_loss(head(feature).float(), target))
        weights = losses[0].new_tensor(MULTISCALE_TKE_LEVEL_WEIGHTS)
        total = torch.sum(torch.stack(losses) * weights) / weights.sum()
        return total, losses


class WallBudgetExceeded(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--real-closure-bank", type=Path,
        help=("clean81 training-prefix closure bank; when supplied, only the "
              "candidate Sim arm receives the frozen domain augmentation"))
    parser.add_argument("--real-closure-bank-sha256")
    parser.add_argument(
        "--phase-aligned-re-mix", action="store_true",
        help=("compose Real-closure with fixed, phase-aligned adjacent-Re "
              "interpolation; training only and deployment-cost neutral"))
    parser.add_argument(
        "--phase-aligned-parameter-grid-mix", action="store_true",
        help=("compose Real-closure with fixed, phase-aligned adjacent-Re/AoA "
              "interpolation; training only and deployment-cost neutral"))
    parser.add_argument(
        "--real-closure-equivariance", action="store_true",
        help=("add the frozen paired clean/closure point-equivariance loss; "
              "requires --real-closure-bank"))
    parser.add_argument(
        "--multiscale-tke-deep-supervision", action="store_true",
        help=("train three disposable decoder heads against future physical "
              "TKE maps; requires --real-closure-bank"))
    parser.add_argument(
        "--multiscale-energy-controller", action="store_true",
        help=("deploy a zero-initialized all-decoder-scale physical-energy "
              "controller; requires --real-closure-bank"))
    parser.add_argument(
        "--horizon-local-energy-controller", action="store_true",
        help=("deploy a zero-initialized 20-channel physical-energy "
              "controller; requires --real-closure-bank"))
    parser.add_argument(
        "--history-energy-controller", action="store_true",
        help=("deploy a zero-initialized observation-dynamics physical-energy "
              "controller; requires --real-closure-bank"))
    parser.add_argument(
        "--physical-energy-context-head", action="store_true",
        help=("TKE head v2: history dynamics plus the base prediction's "
              "own per-channel energy context; two per-channel gains"))
    parser.add_argument(
        "--direct-history-tke-regression", action="store_true",
        help=("train the observation-history controller against the direct "
              "truth/base log-TKE ratio for all Sim updates"))
    parser.add_argument(
        "--hybrid-history-multiscale-energy-controller", action="store_true",
        help=("deploy additive observation-history and all-decoder-scale "
              "physical-energy branches; requires --real-closure-bank"))
    parser.add_argument(
        "--cross-conditioned-backbone", action="store_true",
        help=("spatially FiLM-condition every decoder scale from a separate "
              "recent-dynamics stream; requires --real-closure-bank"))
    parser.add_argument(
        "--temporal-point-decoder", action="store_true",
        help=("replace flattened-horizon point heads with a factorized "
              "explicit-time decoder; requires --real-closure-bank"))
    parser.add_argument(
        "--temporal-point-amplitude-factorization", action="store_true",
        help=("RMS-normalize temporal shapes and predict their bounded "
              "spatial amplitudes separately; implies --temporal-point-decoder"))
    parser.add_argument(
        "--isoenergetic-temporal-transport", action="store_true",
        help=("retain the flattened point heads and learn a bounded explicit-"
              "time warp that exactly restores their temporal mean and RMS; "
              "requires --real-closure-bank"))
    parser.add_argument(
        "--temporal-band-mixer", action="store_true",
        help=("retain the flattened point heads and modulate four non-DC "
              "temporal frequency bands using decoder/history context; "
              "requires the standard XCB/member architecture"))
    parser.add_argument(
        "--official-point-objective", action="store_true",
        help=("replace the legacy Sim (1,4,1) Rel/TKE/MVPE objective by "
              "the published point-score weights (46,10,9); this arm "
              "requires --isoenergetic-temporal-transport"))
    parser.add_argument(
        "--official-point-controller-tke-floor", action="store_true",
        help=("retain a calibrated 0.004 physical-TKE auxiliary floor after "
              "update 1500, applied only to the shared energy controller; "
              "requires --official-point-objective"))
    parser.add_argument(
        "--pareto-tke-projection", action="store_true",
        help=("use the published (46,10,9) point objective on the unchanged "
              "B210 architecture, while minimally projecting destructive "
              "output-space gradients away from the official TKE gradient; "
              "requires --real-closure-bank"))
    parser.add_argument(
        "--legacy-tke-gradient-floor", action="store_true",
        help=("use the published (46,10,9) point objective while minimally "
              "restoring, per sample, any TKE-parallel output-gradient "
              "magnitude missing relative to the legacy (1,4,1) objective; "
              "requires --real-closure-bank"))
    parser.add_argument(
        "--member-integrated-legacy-tke-gradient-floor",
        action="store_true",
        help=("extend --legacy-tke-gradient-floor to the complete ensemble: "
              "bootstrap members use the published (46,10,9) point loss, "
              "the same-state reference swaps mean and members to (1,4,1), "
              "and the sample-local floor acts on the full member tensor"))
    parser.add_argument(
        "--member-official-objective", action="store_true",
        help=("train bootstrap members with the published (46,10,9) point "
              "loss while retaining the legacy TKE gradient floor only on "
              "the ensemble mean; requires --real-closure-bank"))
    parser.add_argument(
        "--decoupled-member-residual", action="store_true",
        help=("add a detached-feature, cross-member-zero-mean residual "
              "decoder so member/UQ supervision cannot directly rewrite "
              "the point backbone; implies --member-official-objective"))
    parser.add_argument(
        "--member-feature-adapters", action="store_true",
        help=("add zero-initialized member-specific residual adapters at "
              "every decoder scale while preserving the legacy bootstrap "
              "member path; implies --legacy-tke-gradient-floor"))
    parser.add_argument(
        "--standard-objective-member-feature-adapters", action="store_true",
        help=("add the same zero-initialized multiscale member adapters but "
              "retain the standard Sim (1,4,1) objective; intended for the "
              "from-scratch cross-conditioned Standard24k architecture arm"))
    parser.add_argument(
        "--fluctuation-role-tower", action="store_true",
        help=("give the zero-time-mean point fluctuation a shared nonlinear "
              "multiscale residual tower while its temporal-mean path stays "
              "unchanged; requires the standard XCB/member architecture"))
    parser.add_argument(
        "--conditional-future-energy", action="store_true",
        help=("train all-scale member adapters as functional distribution "
              "samples with an energy score against target-only Real-closure "
              "branches; implies --member-feature-adapters"))
    parser.add_argument(
        "--conditional-future-energy-members-only", action="store_true",
        help=("route target-only closure branches only to member energy; "
              "quantile and soft-SPS remain supervised by the base truth; "
              "implies --member-feature-adapters"))
    parser.add_argument(
        "--conditional-future-energy-adapter-only", action="store_true",
        help=("route target-only future energy gradients only to member-"
              "specific adapters; all shared/point/UQ paths retain base "
              "supervision; implies --member-feature-adapters"))
    parser.add_argument(
        "--conditional-future-crps-mean-preserving", action="store_true",
        help=("train exact mean-preserving multiscale member residuals with "
              "pointwise empirical CRPS on target-only future branches; "
              "point and direct UQ supervision retain base truth"))
    parser.add_argument(
        "--detach-mean-preserving-adapter-context", action="store_true",
        help=("stop adapter-body context gradients at decoder features while "
              "retaining the live legacy feature/member path; requires the "
              "mean-preserving pointwise-CRPS mode"))
    parser.add_argument(
        "--isolate-distribution-tower", action="store_true",
        help=("train legacy members only from the base member path, train "
              "stochastic residual adapters only from future CRPS, and train "
              "the UQ tower only from detached point/member/backbone context; "
              "requires the mean-preserving pointwise-CRPS mode"))
    parser.add_argument(
        "--budgeted-disagreement-allocation", action="store_true",
        help=("collapse members in the base UQ tower, then let detached real "
              "member spread only redistribute the existing spatial width "
              "budget; requires --isolate-distribution-tower"))
    parser.add_argument(
        "--research-equivalent-cache-sha256",
        help=("permit a semantically validated rebuild of the frozen P1 cache; "
              "this changes artifact identity and is never formal evidence"))
    parser.add_argument("--acknowledge-research-cache", default="")
    return parser.parse_args()


def expected_teacher_cache_sha256(args: argparse.Namespace) -> str:
    research = getattr(args, "research_equivalent_cache_sha256", None)
    acknowledgement = getattr(args, "acknowledge_research_cache", "")
    if research is None:
        if acknowledgement:
            raise ValueError(
                "research-cache acknowledgement requires a research cache SHA")
        return TEACHER_CACHE_SHA256
    research = rehearsal.normalized_sha256(
        research, "research-equivalent-cache-sha256")
    if acknowledgement != RESEARCH_CACHE_ACKNOWLEDGEMENT:
        raise ValueError("research-equivalent cache requires acknowledgement")
    if research == TEACHER_CACHE_SHA256:
        raise ValueError("formal cache must use the formal cache path")
    return research


def contract_parameters(args: argparse.Namespace) -> int:
    """Return the architecture count printed before any GPU training."""
    if bool(getattr(args, "fluctuation_role_tower", False)):
        return EXPECTED_XCB_MEMBER_FLUCTUATION_ROLE_PARAMETERS
    if bool(getattr(args, "temporal_band_mixer", False)):
        return EXPECTED_XCB_MEMBER_TEMPORAL_BAND_PARAMETERS
    if (bool(getattr(args, "cross_conditioned_backbone", False))
            and bool(getattr(args, "member_feature_adapters", False))
            and bool(getattr(args, "multiscale_energy_controller", False))):
        return EXPECTED_XCB_MEMBER_MULTISCALE_ENERGY_PARAMETERS
    if (bool(getattr(args, "cross_conditioned_backbone", False))
            and bool(getattr(args, "member_feature_adapters", False))):
        return EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS
    if bool(getattr(args, "decoupled_member_residual", False)):
        return EXPECTED_DECOUPLED_MEMBER_RESIDUAL_PARAMETERS
    if bool(getattr(args, "budgeted_disagreement_allocation", False)):
        return EXPECTED_BUDGETED_DISAGREEMENT_PARAMETERS
    if bool(getattr(args, "member_feature_adapters", False)):
        return EXPECTED_MEMBER_FEATURE_ADAPTER_PARAMETERS
    if bool(getattr(
            args, "temporal_point_amplitude_factorization", False)):
        return EXPECTED_TEMPORAL_AMPLITUDE_FACTORIZATION_PARAMETERS
    if bool(getattr(args, "isoenergetic_temporal_transport", False)):
        return EXPECTED_TEMPORAL_TRANSPORT_PARAMETERS
    if bool(getattr(args, "temporal_point_decoder", False)):
        return EXPECTED_TEMPORAL_POINT_DECODER_PARAMETERS
    if bool(getattr(args, "cross_conditioned_backbone", False)):
        return EXPECTED_CROSS_CONDITIONED_BACKBONE_PARAMETERS
    if bool(getattr(
            args, "hybrid_history_multiscale_energy_controller", False)):
        return EXPECTED_HYBRID_HISTORY_MULTISCALE_ENERGY_PARAMETERS
    if bool(getattr(args, "history_energy_controller", False)):
        return EXPECTED_HISTORY_ENERGY_PARAMETERS
    if bool(getattr(args, "multiscale_energy_controller", False)):
        return EXPECTED_MULTISCALE_ENERGY_PARAMETERS
    if bool(getattr(args, "horizon_local_energy_controller", False)):
        return EXPECTED_HORIZON_LOCAL_ENERGY_PARAMETERS
    return EXPECTED_PARAMETERS


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers != 0:
        raise ValueError("formal B210 Sim bundle freezes num-workers=0")
    bank = getattr(args, "real_closure_bank", None)
    bank_sha = getattr(args, "real_closure_bank_sha256", None)
    future_modes = sum(bool(getattr(args, name, False)) for name in (
        "conditional_future_energy",
        "conditional_future_energy_members_only",
        "conditional_future_energy_adapter_only",
        "conditional_future_crps_mean_preserving"))
    if future_modes > 1:
        raise ValueError("conditional future routing modes are mutually exclusive")
    standard_member_adapters = bool(getattr(
        args, "standard_objective_member_feature_adapters", False))
    if standard_member_adapters and bool(getattr(
            args, "member_feature_adapters", False)):
        raise ValueError(
            "select only one member-feature-adapter objective contract")
    if (standard_member_adapters
            and not bool(getattr(args, "cross_conditioned_backbone", False))):
        raise ValueError(
            "standard-objective adapters require cross conditioning")
    if (standard_member_adapters
            and bool(getattr(args, "legacy_tke_gradient_floor", False))):
        raise ValueError(
            "standard-objective adapters cannot use the legacy TKE floor")
    if standard_member_adapters and future_modes:
        raise ValueError(
            "standard-objective adapters cannot use conditional-future modes")
    if standard_member_adapters:
        args.member_feature_adapters = True
    fluctuation_role_tower = bool(getattr(
        args, "fluctuation_role_tower", False))
    if (fluctuation_role_tower
            and not (standard_member_adapters
                     and bool(getattr(
                         args, "cross_conditioned_backbone", False)))):
        raise ValueError(
            "fluctuation role tower requires standard XCB/member")
    if (fluctuation_role_tower
            and bool(getattr(args, "multiscale_energy_controller", False))):
        raise ValueError(
            "fluctuation role tower and multiscale energy are separate arms")
    xcb_member_multiscale = bool(
        standard_member_adapters
        and getattr(args, "cross_conditioned_backbone", False)
        and getattr(args, "multiscale_energy_controller", False))
    if future_modes:
        args.member_feature_adapters = True
    args.mean_preserving_member_adapters = bool(getattr(
        args, "conditional_future_crps_mean_preserving", False))
    if (getattr(args, "isolate_distribution_tower", False)
            and not args.mean_preserving_member_adapters):
        raise ValueError(
            "isolated distribution tower requires mean-preserving CRPS")
    if (getattr(args, "budgeted_disagreement_allocation", False)
            and not getattr(args, "isolate_distribution_tower", False)):
        raise ValueError(
            "budgeted disagreement requires isolated distribution tower")
    if (getattr(args, "detach_mean_preserving_adapter_context", False)
            and not args.mean_preserving_member_adapters):
        raise ValueError(
            "detached adapter context requires mean-preserving CRPS")
    args.detach_member_adapter_context = bool(
        getattr(args, "detach_mean_preserving_adapter_context", False)
        or getattr(args, "isolate_distribution_tower", False))
    args.detach_uq_context = bool(getattr(
        args, "isolate_distribution_tower", False))
    if getattr(
            args, "member_integrated_legacy_tke_gradient_floor", False):
        args.legacy_tke_gradient_floor = True
    if getattr(args, "decoupled_member_residual", False):
        args.member_official_objective = True
    if (getattr(args, "member_feature_adapters", False)
            and not standard_member_adapters):
        args.legacy_tke_gradient_floor = True
    if getattr(args, "member_official_objective", False):
        args.legacy_tke_gradient_floor = True
    if getattr(args, "temporal_point_amplitude_factorization", False):
        args.temporal_point_decoder = True
    if getattr(args, "direct_history_tke_regression", False):
        args.history_energy_controller = True
    if getattr(args, "hybrid_history_multiscale_energy_controller", False):
        args.history_energy_controller = True
        args.multiscale_energy_controller = True
    if (bank is None) != (bank_sha is None):
        raise ValueError(
            "Real-closure experiment requires both bank and bank SHA-256")
    if bank_sha is not None:
        args.real_closure_bank_sha256 = rehearsal.normalized_sha256(
            bank_sha, "real-closure-bank-sha256")
    if getattr(args, "real_closure_equivariance", False) and bank is None:
        raise ValueError("Real-closure equivariance requires a closure bank")
    if getattr(args, "phase_aligned_re_mix", False) and bank is None:
        raise ValueError("phase-aligned Re mix requires a closure bank")
    if (getattr(args, "phase_aligned_parameter_grid_mix", False)
            and bank is None):
        raise ValueError("phase-aligned parameter-grid mix requires a closure bank")
    if (getattr(args, "phase_aligned_re_mix", False)
            and getattr(args, "phase_aligned_parameter_grid_mix", False)):
        raise ValueError("select only one phase-aligned parameter-mix arm")
    if (getattr(args, "phase_aligned_re_mix", False)
            and getattr(args, "real_closure_equivariance", False)):
        raise ValueError(
            "phase-aligned Re mix and closure equivariance are separate arms")
    if (getattr(args, "phase_aligned_parameter_grid_mix", False)
            and getattr(args, "real_closure_equivariance", False)):
        raise ValueError(
            "phase-aligned grid mix and closure equivariance are separate arms")
    if (getattr(args, "multiscale_tke_deep_supervision", False)
            and bank is None):
        raise ValueError(
            "multiscale TKE deep supervision requires a closure bank")
    if (getattr(args, "multiscale_tke_deep_supervision", False)
            and getattr(args, "real_closure_equivariance", False)):
        raise ValueError(
            "multiscale TKE and closure equivariance are separate arms")
    if (getattr(args, "multiscale_energy_controller", False)
            and bank is None):
        raise ValueError("multiscale energy controller requires a closure bank")
    if (getattr(args, "multiscale_energy_controller", False)
            and (getattr(args, "multiscale_tke_deep_supervision", False)
                 or getattr(args, "real_closure_equivariance", False))):
        raise ValueError(
            "multiscale energy is a separate architecture arm")
    if (getattr(args, "horizon_local_energy_controller", False)
            and bank is None):
        raise ValueError("horizon-local energy controller requires a closure bank")
    if (getattr(args, "horizon_local_energy_controller", False)
            and (getattr(args, "multiscale_energy_controller", False)
                 or getattr(args, "multiscale_tke_deep_supervision", False)
                 or getattr(args, "real_closure_equivariance", False))):
        raise ValueError("horizon-local energy is a separate architecture arm")
    if (getattr(args, "history_energy_controller", False)
            and bank is None):
        raise ValueError("history energy controller requires a closure bank")
    if (getattr(args, "history_energy_controller", False)
            and (getattr(args, "multiscale_energy_controller", False)
                 and not getattr(
                     args, "hybrid_history_multiscale_energy_controller", False)
                 or getattr(args, "horizon_local_energy_controller", False)
                 or getattr(args, "multiscale_tke_deep_supervision", False)
                 or getattr(args, "real_closure_equivariance", False))):
        raise ValueError("history energy is a separate architecture arm")
    if (getattr(args, "hybrid_history_multiscale_energy_controller", False)
            and getattr(args, "direct_history_tke_regression", False)):
        raise ValueError("hybrid energy and direct history TKE are separate arms")
    if getattr(args, "cross_conditioned_backbone", False):
        if bank is None:
            raise ValueError(
                "cross-conditioned backbone requires a closure bank")
        if (getattr(args, "real_closure_equivariance", False)
                or getattr(args, "multiscale_tke_deep_supervision", False)
                or (getattr(args, "multiscale_energy_controller", False)
                    and not xcb_member_multiscale)
                or getattr(args, "horizon_local_energy_controller", False)
                or getattr(args, "history_energy_controller", False)
                or getattr(args, "direct_history_tke_regression", False)
                or getattr(
                    args, "hybrid_history_multiscale_energy_controller", False)):
            raise ValueError(
                "cross-conditioned backbone is a separate architecture arm")
    if getattr(args, "temporal_point_decoder", False):
        if bank is None:
            raise ValueError("temporal point decoder requires a closure bank")
        if (getattr(args, "real_closure_equivariance", False)
                or getattr(args, "multiscale_tke_deep_supervision", False)
                or getattr(args, "multiscale_energy_controller", False)
                or getattr(args, "horizon_local_energy_controller", False)
                or getattr(args, "history_energy_controller", False)
                or getattr(args, "direct_history_tke_regression", False)
                or getattr(
                    args, "hybrid_history_multiscale_energy_controller", False)
                or getattr(args, "cross_conditioned_backbone", False)):
            raise ValueError(
                "temporal point decoder is a separate architecture arm")
    if getattr(args, "isoenergetic_temporal_transport", False):
        if bank is None:
            raise ValueError(
                "isoenergetic temporal transport requires a closure bank")
        if (getattr(args, "real_closure_equivariance", False)
                or getattr(args, "multiscale_tke_deep_supervision", False)
                or getattr(args, "multiscale_energy_controller", False)
                or getattr(args, "horizon_local_energy_controller", False)
                or getattr(args, "history_energy_controller", False)
                or getattr(args, "direct_history_tke_regression", False)
                or getattr(
                    args, "hybrid_history_multiscale_energy_controller", False)
                or getattr(args, "cross_conditioned_backbone", False)
                or getattr(args, "temporal_point_decoder", False)):
            raise ValueError(
                "isoenergetic temporal transport is a separate architecture "
                "arm")
    if getattr(args, "temporal_band_mixer", False):
        if bank is None:
            raise ValueError("temporal band mixer requires a closure bank")
        if getattr(args, "isoenergetic_temporal_transport", False):
            raise ValueError(
                "temporal band mixer and isoenergetic transport are separate")
        if not (
                standard_member_adapters
                and getattr(args, "cross_conditioned_backbone", False)
                and getattr(args, "member_feature_adapters", False)):
            raise ValueError(
                "temporal band mixer requires standard XCB/member adapters")
        if (getattr(args, "multiscale_energy_controller", False)
                or getattr(args, "horizon_local_energy_controller", False)
                or getattr(args, "history_energy_controller", False)
                or getattr(args, "temporal_point_decoder", False)):
            raise ValueError(
                "temporal band mixer is a separate point-correction arm")
    if (getattr(args, "official_point_objective", False)
            and not getattr(args, "isoenergetic_temporal_transport", False)):
        raise ValueError(
            "official point objective requires isoenergetic temporal transport")
    if (getattr(args, "official_point_controller_tke_floor", False)
            and not getattr(args, "official_point_objective", False)):
        raise ValueError(
            "controller TKE floor requires official point objective")
    if getattr(args, "pareto_tke_projection", False):
        if bank is None:
            raise ValueError("Pareto TKE projection requires a closure bank")
        if (getattr(args, "official_point_objective", False)
                or getattr(
                    args, "official_point_controller_tke_floor", False)
                or getattr(args, "isoenergetic_temporal_transport", False)
                or getattr(args, "cross_conditioned_backbone", False)
                or getattr(args, "temporal_point_decoder", False)
                or getattr(args, "multiscale_tke_deep_supervision", False)
                or getattr(args, "multiscale_energy_controller", False)
                or getattr(args, "horizon_local_energy_controller", False)
                or getattr(args, "history_energy_controller", False)
                or getattr(args, "real_closure_equivariance", False)):
            raise ValueError(
                "Pareto TKE projection is a separate training-mechanism arm")
    if getattr(args, "legacy_tke_gradient_floor", False):
        if bank is None:
            raise ValueError(
                "legacy TKE gradient floor requires a closure bank")
        if (getattr(args, "pareto_tke_projection", False)
                or getattr(args, "official_point_objective", False)
                or getattr(
                    args, "official_point_controller_tke_floor", False)
                or getattr(args, "isoenergetic_temporal_transport", False)
                or (getattr(args, "cross_conditioned_backbone", False)
                    and not getattr(args, "member_feature_adapters", False))
                or getattr(args, "temporal_point_decoder", False)
                or getattr(args, "multiscale_tke_deep_supervision", False)
                or getattr(args, "multiscale_energy_controller", False)
                or getattr(args, "horizon_local_energy_controller", False)
                or getattr(args, "history_energy_controller", False)
                or getattr(args, "real_closure_equivariance", False)):
            raise ValueError(
                "legacy TKE gradient floor is a separate training arm")
    if getattr(args, "decoupled_member_residual", False):
        if bank is None:
            raise ValueError(
                "decoupled member residual requires a closure bank")
        if (getattr(
                    args, "member_integrated_legacy_tke_gradient_floor",
                    False)
                or getattr(args, "pareto_tke_projection", False)
                or getattr(args, "official_point_objective", False)
                or getattr(args, "isoenergetic_temporal_transport", False)
                or getattr(args, "cross_conditioned_backbone", False)
                or getattr(args, "temporal_point_decoder", False)
                or getattr(args, "multiscale_tke_deep_supervision", False)
                or getattr(args, "multiscale_energy_controller", False)
                or getattr(args, "horizon_local_energy_controller", False)
                or getattr(args, "history_energy_controller", False)
                or getattr(args, "real_closure_equivariance", False)):
            raise ValueError(
                "decoupled member residual is a separate architecture arm")
    if getattr(args, "member_feature_adapters", False):
        if bank is None:
            raise ValueError("member feature adapters require a closure bank")
        if (getattr(args, "decoupled_member_residual", False)
                or getattr(
                    args, "member_integrated_legacy_tke_gradient_floor",
                    False)
                or getattr(args, "member_official_objective", False)
                or getattr(args, "pareto_tke_projection", False)
                or getattr(args, "official_point_objective", False)
                or getattr(args, "isoenergetic_temporal_transport", False)
                or getattr(args, "temporal_point_decoder", False)
                or getattr(args, "multiscale_tke_deep_supervision", False)
                or (getattr(args, "multiscale_energy_controller", False)
                    and not xcb_member_multiscale)
                or getattr(args, "horizon_local_energy_controller", False)
                or getattr(args, "history_energy_controller", False)
                or getattr(args, "real_closure_equivariance", False)):
            raise ValueError(
                "member feature adapters are a separate architecture arm")
    if ((getattr(args, "conditional_future_energy", False)
         or getattr(args, "conditional_future_energy_members_only", False)
         or getattr(args, "conditional_future_energy_adapter_only", False)
         or getattr(args, "conditional_future_crps_mean_preserving", False))
            and bank is None):
        raise ValueError("conditional future energy requires a closure bank")
    args.teacher_cache_sha256 = rehearsal.normalized_sha256(
        args.teacher_cache_sha256, "teacher-cache-sha256")
    selected_cache_sha = expected_teacher_cache_sha256(args)
    if args.teacher_cache_sha256 != selected_cache_sha:
        raise ValueError("teacher cache SHA is not the frozen current-P1 cache")
    protected = (
        "summary.json", "model_10000.pth", "hgrad_scale_contract.json",
        "multiscale_tke_supervisor_10000.pth", "stop.json")
    if any((args.output_dir / name).exists() for name in protected):
        raise FileExistsError(
            f"refusing to overwrite a prior B210 run in {args.output_dir}")


def b210_config(
        arm: str, multiscale_energy_controller: bool = False,
        horizon_local_energy_controller: bool = False,
        history_energy_controller: bool = False,
        hybrid_history_multiscale_energy_controller: bool = False,
        cross_conditioned_backbone: bool = False,
        temporal_point_decoder: bool = False,
        temporal_point_amplitude_factorization: bool = False,
        isoenergetic_temporal_transport: bool = False,
        temporal_band_mixer: bool = False,
        decoupled_member_residual: bool = False,
        member_feature_adapters: bool = False,
        fluctuation_role_tower: bool = False,
        mean_preserving_member_adapters: bool = False,
        detach_member_adapter_context: bool = False,
        detach_uq_context: bool = False,
        budgeted_disagreement_allocation: bool = False,
        physical_energy_context_head: bool = False,
        ) -> EnsembleUNet2dConfig:
    if arm not in {"control", "candidate"}:
        raise ValueError(f"unknown B210 Sim arm {arm}")
    return EnsembleUNet2dConfig(
        base=210,
        depth=4,
        groups=8,
        difference_steps=5,
        acceleration_steps=CLOSURE_ACCELERATION_STEPS,
        use_coordinates=True,
        use_regime_film=True,
        multi_scale_regime_film=True,
        cross_conditioned_backbone=cross_conditioned_backbone,
        cross_conditioned_steps=CROSS_CONDITIONED_STEPS,
        cross_conditioned_channels=CROSS_CONDITIONED_CHANNELS,
        temporal_point_decoder=temporal_point_decoder,
        temporal_point_channels=TEMPORAL_POINT_CHANNELS,
        temporal_point_blocks=TEMPORAL_POINT_BLOCKS,
        temporal_point_kernel=TEMPORAL_POINT_KERNEL,
        temporal_point_amplitude_factorization=(
            temporal_point_amplitude_factorization),
        temporal_point_amplitude_limit=TEMPORAL_POINT_AMPLITUDE_LIMIT,
        isoenergetic_temporal_transport=isoenergetic_temporal_transport,
        temporal_transport_channels=TEMPORAL_TRANSPORT_CHANNELS,
        temporal_transport_blocks=TEMPORAL_TRANSPORT_BLOCKS,
        temporal_transport_kernel=TEMPORAL_TRANSPORT_KERNEL,
        temporal_transport_max_displacement=(
            TEMPORAL_TRANSPORT_MAX_DISPLACEMENT),
        temporal_band_mixer=temporal_band_mixer,
        temporal_band_channels=TEMPORAL_BAND_CHANNELS,
        temporal_band_count=TEMPORAL_BAND_COUNT,
        temporal_band_log_gain_limit=TEMPORAL_BAND_LOG_GAIN_LIMIT,
        temporal_band_phase_limit=TEMPORAL_BAND_PHASE_LIMIT,
        mean_fluctuation=True,
        ensemble_members=4,
        uq_hidden_channels=160,
        member_adapter_channels=(
            MEMBER_FEATURE_ADAPTER_CHANNELS
            if member_feature_adapters else 0),
        fluctuation_role_channels=(
            FLUCTUATION_ROLE_CHANNELS if fluctuation_role_tower else 0),
        mean_preserving_member_adapters=mean_preserving_member_adapters,
        detach_member_adapter_context=detach_member_adapter_context,
        detach_uq_context=detach_uq_context,
        budgeted_disagreement_allocation=(
            budgeted_disagreement_allocation),
        decoupled_member_residual_channels=(
            DECOUPLED_MEMBER_RESIDUAL_CHANNELS
            if decoupled_member_residual else 0),
        decoupled_member_residual_blocks=(
            DECOUPLED_MEMBER_RESIDUAL_BLOCKS),
        mean_profile_channels=104,
        uq_flow_context=True,
        residual_quantile_intervals=True,
        utility_refinement=True,
        utility_refinement_kernel=3,
        utility_refinement_blocks=0,
        temporal_uq_refinement=True,
        temporal_uq_channels=24,
        temporal_uq_blocks=2,
        temporal_uq_kernel=3,
        coverage_calibration=False,
        disagreement_scale_fraction=0.01,
        physical_energy_controller=True,
        physical_energy_member_sync=True,
        physical_energy_multiscale=(
            multiscale_energy_controller
            or hybrid_history_multiscale_energy_controller),
        physical_energy_horizon_local=horizon_local_energy_controller,
        physical_energy_history_conditioned=(
            history_energy_controller
            or hybrid_history_multiscale_energy_controller),
        physical_energy_context_head=bool(
            physical_energy_context_head
            or __import__("os").environ.get(
                "B210_ENERGY_CONTEXT_HEAD") == "1"),
        physical_energy_history_steps=5,
        physical_energy_hidden_channels=32,
        physical_energy_log_limit=1.0,
        phase_delta_hidden_channels=PHASE_DELTA_HIDDEN_CHANNELS,
        phase_delta_steps=PHASE_DELTA_STEPS,
        backbone_residual_dropout=(
            BACKBONE_DROPOUT if arm == "candidate" else 0.0),
    )


def training_args(args: argparse.Namespace) -> argparse.Namespace:
    """Freeze the native C208 Sim objective while changing only the bundle."""
    return argparse.Namespace(
        phase="sim",
        updates=TERMINAL_UPDATE,
        start_checkpoint=None,
        output_dir=args.output_dir,
        teacher_cache=args.teacher_cache,
        sim_dir=args.sim_dir,
        real_stats=args.real_stats,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=BATCH_SIZE,
        eval_batch_size=16,
        eval_interval=LOG_INTERVAL,
        checkpoint_interval=TERMINAL_UPDATE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        seed=SEED,
        max_files=0,
        window_stride=20,
        input_noise_std=0.0,
        input_gain_std=0.0,
        input_blur_probability=0.0,
        input_blur_strength=0.0,
        phase_aligned_re_mix=bool(getattr(args, "phase_aligned_re_mix", False)),
        phase_aligned_parameter_grid_mix=bool(getattr(
            args, "phase_aligned_parameter_grid_mix", False)),
        temporal_jitter_max=0,
        point_loss_mode="official_accuracy",
        mse_weight=0.25,
        mse_warmup_updates=0,
        mean_loss_weight=1.0,
        member_loss_weight=(
            CONDITIONAL_FUTURE_MEMBER_MSE_WEIGHT
            if (bool(getattr(args, "conditional_future_energy", False))
                or bool(getattr(
                    args, "conditional_future_energy_members_only", False))
                or bool(getattr(
                    args, "conditional_future_energy_adapter_only", False))
                or bool(getattr(
                    args, "conditional_future_crps_mean_preserving", False)))
            else 0.25),
        member_loss_mode=(
            "official_accuracy" if (
                bool(getattr(args, "member_official_objective", False))
                or bool(getattr(
                    args, "member_integrated_legacy_tke_gradient_floor",
                    False)))
            else "mse"),
        bootstrap_probability=0.75,
        quantile_loss_weight=0.05,
        soft_sps_loss_weight=SOFT_SPS_WEIGHT,
        soft_sps_tau_fraction=0.05,
        soft_sps_warmup_updates=SOFT_SPS_WARMUP_UPDATES,
        soft_sps_uq_only=False,
        soft_sps_proxy_real_scale=True,
        coverage_bce_weight=0.0,
        profile_loss_weight=0.0,
        accuracy_weights=(
            OFFICIAL_POINT_ACCURACY_WEIGHTS
            if (bool(getattr(args, "official_point_objective", False))
                or bool(getattr(args, "pareto_tke_projection", False))
                or bool(getattr(
                    args, "legacy_tke_gradient_floor", False)))
            else rehearsal.SIM_ACCURACY_WEIGHTS),
        distill_weight=P1_DISTILL_WEIGHT,
        trainable_scope="all",
        method=METHOD,
        multiscale_tke_deep_supervision=bool(getattr(
            args, "multiscale_tke_deep_supervision", False)),
        multiscale_energy_controller=bool(getattr(
            args, "multiscale_energy_controller", False)),
        horizon_local_energy_controller=bool(getattr(
            args, "horizon_local_energy_controller", False)),
        history_energy_controller=bool(getattr(
            args, "history_energy_controller", False)),
        physical_energy_context_head=bool(getattr(
            args, "physical_energy_context_head", False)),
        direct_history_tke_regression=bool(getattr(
            args, "direct_history_tke_regression", False)),
        hybrid_history_multiscale_energy_controller=bool(getattr(
            args, "hybrid_history_multiscale_energy_controller", False)),
        cross_conditioned_backbone=bool(getattr(
            args, "cross_conditioned_backbone", False)),
        temporal_point_decoder=bool(getattr(
            args, "temporal_point_decoder", False)),
        temporal_point_amplitude_factorization=bool(getattr(
            args, "temporal_point_amplitude_factorization", False)),
        isoenergetic_temporal_transport=bool(getattr(
            args, "isoenergetic_temporal_transport", False)),
        temporal_band_mixer=bool(getattr(
            args, "temporal_band_mixer", False)),
        official_point_objective=bool(getattr(
            args, "official_point_objective", False)),
        official_point_controller_tke_floor=bool(getattr(
            args, "official_point_controller_tke_floor", False)),
        pareto_tke_projection=bool(getattr(
            args, "pareto_tke_projection", False)),
        legacy_tke_gradient_floor=bool(getattr(
            args, "legacy_tke_gradient_floor", False)),
        member_integrated_legacy_tke_gradient_floor=bool(getattr(
            args, "member_integrated_legacy_tke_gradient_floor", False)),
        member_official_objective=bool(getattr(
            args, "member_official_objective", False)),
        decoupled_member_residual=bool(getattr(
            args, "decoupled_member_residual", False)),
        member_feature_adapters=bool(getattr(
            args, "member_feature_adapters", False)),
        standard_objective_member_feature_adapters=bool(getattr(
            args, "standard_objective_member_feature_adapters", False)),
        fluctuation_role_tower=bool(getattr(
            args, "fluctuation_role_tower", False)),
        mean_preserving_member_adapters=bool(getattr(
            args, "mean_preserving_member_adapters", False)),
        detach_member_adapter_context=bool(getattr(
            args, "detach_member_adapter_context", False)),
        detach_uq_context=bool(getattr(args, "detach_uq_context", False)),
        isolate_distribution_tower=bool(getattr(
            args, "isolate_distribution_tower", False)),
        budgeted_disagreement_allocation=bool(getattr(
            args, "budgeted_disagreement_allocation", False)),
        conditional_future_energy=bool(getattr(
            args, "conditional_future_energy", False)),
        conditional_future_energy_members_only=bool(getattr(
            args, "conditional_future_energy_members_only", False)),
        conditional_future_energy_adapter_only=bool(getattr(
            args, "conditional_future_energy_adapter_only", False)),
        conditional_future_crps_mean_preserving=bool(getattr(
            args, "conditional_future_crps_mean_preserving", False)),
    )


def build_arm_model(
        arm: str, device: torch.device,
        multiscale_energy_controller: bool = False,
        horizon_local_energy_controller: bool = False,
        history_energy_controller: bool = False,
        hybrid_history_multiscale_energy_controller: bool = False,
        cross_conditioned_backbone: bool = False,
        temporal_point_decoder: bool = False,
        temporal_point_amplitude_factorization: bool = False,
        isoenergetic_temporal_transport: bool = False,
        temporal_band_mixer: bool = False,
        decoupled_member_residual: bool = False,
        member_feature_adapters: bool = False,
        fluctuation_role_tower: bool = False,
        mean_preserving_member_adapters: bool = False,
        detach_member_adapter_context: bool = False,
        detach_uq_context: bool = False,
        budgeted_disagreement_allocation: bool = False,
        ) -> EnsembleUNet2d:
    rehearsal.ensemble_train.seed_everything(SEED)
    model = EnsembleUNet2d(b210_config(
        arm, multiscale_energy_controller,
        horizon_local_energy_controller,
        history_energy_controller,
        hybrid_history_multiscale_energy_controller,
        cross_conditioned_backbone,
        temporal_point_decoder,
        temporal_point_amplitude_factorization,
        isoenergetic_temporal_transport,
        temporal_band_mixer,
        decoupled_member_residual,
        member_feature_adapters,
        fluctuation_role_tower,
        mean_preserving_member_adapters,
        detach_member_adapter_context,
        detach_uq_context,
        budgeted_disagreement_allocation)).float()
    total = sum(parameter.numel() for parameter in model.parameters())
    controller = sum(
        parameter.numel() for parameter in model.energy_controller.parameters())
    expected_total = (
        EXPECTED_MULTISCALE_ENERGY_PARAMETERS
        if multiscale_energy_controller else (
            EXPECTED_HORIZON_LOCAL_ENERGY_PARAMETERS
            if horizon_local_energy_controller else EXPECTED_PARAMETERS))
    if history_energy_controller:
        expected_total = EXPECTED_HISTORY_ENERGY_PARAMETERS
    if hybrid_history_multiscale_energy_controller:
        expected_total = EXPECTED_HYBRID_HISTORY_MULTISCALE_ENERGY_PARAMETERS
    if cross_conditioned_backbone:
        expected_total = EXPECTED_CROSS_CONDITIONED_BACKBONE_PARAMETERS
    if temporal_point_decoder:
        expected_total = EXPECTED_TEMPORAL_POINT_DECODER_PARAMETERS
    if temporal_point_amplitude_factorization:
        expected_total = EXPECTED_TEMPORAL_AMPLITUDE_FACTORIZATION_PARAMETERS
    if isoenergetic_temporal_transport:
        expected_total = EXPECTED_TEMPORAL_TRANSPORT_PARAMETERS
    if decoupled_member_residual:
        expected_total = EXPECTED_DECOUPLED_MEMBER_RESIDUAL_PARAMETERS
    if member_feature_adapters:
        expected_total = (
            EXPECTED_BUDGETED_DISAGREEMENT_PARAMETERS
            if budgeted_disagreement_allocation
            else EXPECTED_MEMBER_FEATURE_ADAPTER_PARAMETERS)
    if cross_conditioned_backbone and member_feature_adapters:
        expected_total = (
            EXPECTED_XCB_MEMBER_MULTISCALE_ENERGY_PARAMETERS
            if multiscale_energy_controller
            else EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS)
    if fluctuation_role_tower:
        expected_total = EXPECTED_XCB_MEMBER_FLUCTUATION_ROLE_PARAMETERS
    if temporal_band_mixer:
        expected_total = EXPECTED_XCB_MEMBER_TEMPORAL_BAND_PARAMETERS
    expected_controller = (
        EXPECTED_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS
        if multiscale_energy_controller else (
            EXPECTED_HORIZON_LOCAL_ENERGY_CONTROLLER_PARAMETERS
            if horizon_local_energy_controller
            else EXPECTED_CONTROLLER_PARAMETERS))
    if history_energy_controller:
        expected_controller = EXPECTED_HISTORY_ENERGY_CONTROLLER_PARAMETERS
    if hybrid_history_multiscale_energy_controller:
        expected_controller = (
            EXPECTED_HYBRID_HISTORY_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS)
    if total != expected_total or controller != expected_controller:
        raise RuntimeError(
            f"B210 parameter contract changed: total={total}, "
            f"controller={controller}")
    if cross_conditioned_backbone:
        conditioner = sum(
            parameter.numel()
            for parameter in model.cross_conditioner.parameters())
        if conditioner != EXPECTED_CROSS_CONDITIONER_PARAMETERS:
            raise RuntimeError(
                "B210 cross-conditioner parameter contract changed: "
                f"{conditioner}")
    if temporal_point_decoder:
        point_decoder = sum(
            parameter.numel()
            for parameter in model.temporal_point_decoder.parameters())
        expected_point_decoder = (
            EXPECTED_TEMPORAL_AMPLITUDE_PARAMETERS
            if temporal_point_amplitude_factorization
            else EXPECTED_TEMPORAL_POINT_PARAMETERS)
        if point_decoder != expected_point_decoder:
            raise RuntimeError(
                "B210 temporal point parameter contract changed: "
                f"{point_decoder}")
    if isoenergetic_temporal_transport or temporal_band_mixer:
        transport = sum(
            parameter.numel()
            for parameter in model.temporal_transport.parameters())
        expected_transport = (
            EXPECTED_TEMPORAL_BAND_MODULE_PARAMETERS
            if temporal_band_mixer
            else EXPECTED_TEMPORAL_TRANSPORT_MODULE_PARAMETERS)
        if transport != expected_transport:
            raise RuntimeError(
                "B210 temporal transport parameter contract changed: "
                f"{transport}")
    if decoupled_member_residual:
        residual = sum(
            parameter.numel()
            for parameter in model.decoupled_member_residual.parameters())
        if residual != EXPECTED_DECOUPLED_MEMBER_RESIDUAL_MODULE_PARAMETERS:
            raise RuntimeError(
                "B210 decoupled residual parameter contract changed: "
                f"{residual}")
    if member_feature_adapters:
        adapters = sum(
            parameter.numel()
            for parameter in model.member_feature_adapters.parameters())
        if adapters != EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS:
            raise RuntimeError(
                "B210 member adapter parameter contract changed: "
                f"{adapters}")
    if fluctuation_role_tower:
        role_parameters = sum(
            parameter.numel()
            for parameter in model.fluctuation_role_adapters.parameters())
        if role_parameters != EXPECTED_FLUCTUATION_ROLE_MODULE_PARAMETERS:
            raise RuntimeError(
                "B210 fluctuation role parameter contract changed: "
                f"{role_parameters}")
    if not all(
            torch.count_nonzero(parameter) == 0
            for parameter in model.energy_controller.body[-1].parameters()):
        raise RuntimeError("B210 controller output projection is not zero")
    return model.to(device)


def build_multiscale_tke_supervisor(
        model: EnsembleUNet2d, device: torch.device,
        ) -> MultiScaleTKEDeepSupervisor:
    channels = tuple(
        model.config.base * 2 ** level
        for level in range(model.config.depth - 2, -1, -1))
    # Head initialization must not move the paired model/dropout RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 21_000)
        supervisor = MultiScaleTKEDeepSupervisor(channels)
    return supervisor.to(device)


def configure_trainable(
        model: EnsembleUNet2d, arm: str,
        ) -> tuple[torch.nn.Parameter, ...]:
    if arm not in {"control", "candidate"}:
        raise ValueError(f"unknown B210 Sim arm {arm}")
    for name, parameter in model.named_parameters():
        is_controller = name.startswith("energy_controller.")
        is_cross_conditioner = name.startswith("cross_conditioner.")
        is_temporal_transport = name.startswith("temporal_transport.")
        is_decoupled_residual = name.startswith(
            "decoupled_member_residual.")
        is_member_adapter = name.startswith("member_feature_adapters.")
        is_fluctuation_role = name.startswith("fluctuation_role_adapters.")
        candidate_only = (
            is_controller or is_cross_conditioner or is_temporal_transport
            or is_decoupled_residual or is_member_adapter
            or is_fluctuation_role)
        parameter.requires_grad_(arm == "candidate" or not candidate_only)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    controller = sum(
        parameter.numel() for parameter in model.energy_controller.parameters())
    cross_conditioner = (
        0 if model.cross_conditioner is None else sum(
            parameter.numel()
            for parameter in model.cross_conditioner.parameters()))
    temporal_transport = (
        0 if model.temporal_transport is None else sum(
            parameter.numel()
            for parameter in model.temporal_transport.parameters()))
    decoupled_residual = (
        0 if model.decoupled_member_residual is None else sum(
            parameter.numel()
            for parameter in model.decoupled_member_residual.parameters()))
    member_adapters = (
        0 if model.member_feature_adapters is None else sum(
            parameter.numel()
            for parameter in model.member_feature_adapters.parameters()))
    fluctuation_role = (
        0 if model.fluctuation_role_adapters is None else sum(
            parameter.numel()
            for parameter in model.fluctuation_role_adapters.parameters()))
    expected = (
        total if arm == "candidate"
        else total - controller - cross_conditioner - temporal_transport
        - decoupled_residual - member_adapters - fluctuation_role)
    if sum(parameter.numel() for parameter in trainable) != expected:
        raise RuntimeError(f"B210 {arm} trainability contract changed")
    return trainable


def controller_state(
        model: EnsembleUNet2d,
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.energy_controller.state_dict().items())


def cross_conditioner_state(
        model: EnsembleUNet2d,
        ) -> OrderedDict[str, torch.Tensor]:
    if model.cross_conditioner is None:
        return OrderedDict()
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.cross_conditioner.state_dict().items())


def temporal_point_state(
        model: EnsembleUNet2d,
        ) -> OrderedDict[str, torch.Tensor]:
    if model.temporal_point_decoder is None:
        return OrderedDict()
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.temporal_point_decoder.state_dict().items())


def temporal_transport_state(
        model: EnsembleUNet2d,
        ) -> OrderedDict[str, torch.Tensor]:
    if model.temporal_transport is None:
        return OrderedDict()
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.temporal_transport.state_dict().items())


def decoupled_member_residual_state(
        model: EnsembleUNet2d,
        ) -> OrderedDict[str, torch.Tensor]:
    if model.decoupled_member_residual is None:
        return OrderedDict()
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in
        model.decoupled_member_residual.state_dict().items())


def decoupled_residual_output_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if name.startswith("output."))


def decoupled_residual_body_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if name.startswith("body."))


def member_feature_adapter_state(
        model: EnsembleUNet2d,
        ) -> OrderedDict[str, torch.Tensor]:
    if model.member_feature_adapters is None:
        return OrderedDict()
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.member_feature_adapters.state_dict().items())


def member_adapter_output_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if ".body.4." in f".{name}")


def member_adapter_body_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if ".body.4." not in f".{name}")


def fluctuation_role_state(
        model: EnsembleUNet2d,
        ) -> OrderedDict[str, torch.Tensor]:
    if model.fluctuation_role_adapters is None:
        return OrderedDict()
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in
        model.fluctuation_role_adapters.state_dict().items())


def temporal_transport_output_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if name.startswith("output."))


def temporal_transport_body_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if not name.startswith("output."))


def temporal_point_output_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if (name.startswith("member_mean_heads.")
            or name.startswith("member_fluctuation_heads.")
            or name.startswith("member_amplitude_heads.")))


def temporal_point_body_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if not (name.startswith("member_mean_heads.")
                or name.startswith("member_fluctuation_heads.")
                or name.startswith("member_amplitude_heads.")))


def cross_conditioner_projection_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if name.startswith("projections."))


def cross_conditioner_body_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if name.startswith("body."))


def controller_projection_names(
        state: Mapping[str, torch.Tensor],
        ) -> set[str]:
    """Names of the controller's final projection, found dynamically.

    v1 bodies end at `body.5`; the TKE head v2 body ends at `body.7`.  Resolving the
    last indexed module keeps the audit identical for both.
    """
    indices = sorted({
        int(name.split(".")[1]) for name in state
        if name.startswith("body.") and name.split(".")[1].isdigit()})
    if not indices:
        return set()
    last = indices[-1]
    return {f"body.{last}.weight", f"body.{last}.bias"}


def controller_projection_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    names = controller_projection_names(state)
    return OrderedDict(
        (name, value) for name, value in state.items() if name in names)


def controller_front_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    names = controller_projection_names(state)
    return OrderedDict(
        (name, value) for name, value in state.items() if name not in names)


def tensor_state_equal(first: Mapping[str, torch.Tensor],
                       second: Mapping[str, torch.Tensor]) -> bool:
    return list(first) == list(second) and all(
        torch.equal(first[name], second[name]) for name in first)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return rehearsal.ensemble_train.model_state_sha256(state)


def gradient_l2_from_parameters(
        parameters: Sequence[torch.nn.Parameter]) -> float:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt())


def gradient_l2_from_tensors(
        gradients: Sequence[torch.Tensor | None]) -> float:
    squared = None
    for gradient in gradients:
        if gradient is None:
            continue
        value = gradient.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt())


def multiscale_tke_feature_gradient_audit(
        native_loss: torch.Tensor, weighted_auxiliary: torch.Tensor,
        features: Sequence[torch.Tensor],
        ) -> dict[str, float | bool]:
    native = torch.autograd.grad(
        native_loss, tuple(features), retain_graph=True, allow_unused=False)
    auxiliary = torch.autograd.grad(
        weighted_auxiliary, tuple(features), retain_graph=True,
        allow_unused=False)
    native_norm = gradient_l2_from_tensors(native)
    auxiliary_norm = gradient_l2_from_tensors(auxiliary)
    ratio = auxiliary_norm / max(
        native_norm, torch.finfo(torch.float32).tiny)
    finite_nonzero = bool(
        math.isfinite(native_norm) and math.isfinite(auxiliary_norm)
        and math.isfinite(ratio) and native_norm > 0.0
        and auxiliary_norm > 0.0)
    return {
        "native_feature_gradient_norm": native_norm,
        "multiscale_tke_feature_gradient_norm": auxiliary_norm,
        "multiscale_tke_to_native_feature_gradient_ratio": ratio,
        "finite_nonzero": finite_nonzero,
    }


def add_parameter_gradients(
        parameters: Sequence[torch.nn.Parameter],
        gradients: Sequence[torch.Tensor | None]) -> None:
    if len(parameters) != len(gradients):
        raise ValueError("B210 controller parameter/gradient length mismatch")
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            parameter.grad.add_(gradient)


def direct_history_tke_regression_loss(
        model: EnsembleUNet2d, base_prediction_normalized: torch.Tensor,
        target_physical: torch.Tensor, history_normalized: torch.Tensor,
        features: Sequence[torch.Tensor], normalizer,
        ) -> tuple[torch.Tensor, dict[str, float]]:
    """Regress the exact bounded log-energy correction required by truth.

    The base prediction is detached: this auxiliary is a direct deployed-head
    target and must not rewrite the point backbone through its denominator.
    """
    if not model.config.physical_energy_history_conditioned:
        raise ValueError("direct history TKE requires the history controller")
    base_physical = rehearsal.ensemble_train.common.to_physical_target(
        base_prediction_normalized.detach(), normalizer)[..., :2].float()
    truth_physical = target_physical[..., :2].float()

    def energy(value: torch.Tensor) -> torch.Tensor:
        fluctuation = value - value.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)

    base_tke = energy(base_physical)
    truth_tke = energy(truth_physical)
    limit = float(model.config.physical_energy_log_limit)
    if bool(getattr(model.config, "physical_energy_context_head", False)):
        def per_channel_variance(value: torch.Tensor) -> torch.Tensor:
            fluctuation = value - value.mean(dim=1, keepdim=True)
            return fluctuation.square().mean(dim=1).movedim(-1, -3)

        base_variance = per_channel_variance(base_physical)
        truth_variance = per_channel_variance(truth_physical)
        target_delta = torch.log(
            (truth_variance + DIRECT_HISTORY_TKE_LOG_EPSILON)
            / (base_variance + DIRECT_HISTORY_TKE_LOG_EPSILON)
        ).clamp(-limit, limit)
        raw_delta = model._physical_energy_delta(
            list(features), history_normalized, base_physical)
        if raw_delta.shape[-2:] != target_delta.shape[-2:]:
            raw_delta = F.interpolate(
                raw_delta, size=target_delta.shape[-2:], mode="bilinear",
                align_corners=False)
        predicted_delta = limit * torch.tanh(raw_delta)
    else:
        target_delta = torch.log(
            (truth_tke + DIRECT_HISTORY_TKE_LOG_EPSILON)
            / (base_tke + DIRECT_HISTORY_TKE_LOG_EPSILON)
        ).clamp(-limit, limit)
        raw_delta = model._physical_energy_delta(
            list(features), history_normalized)
        if raw_delta.shape[-2:] != target_delta.shape[-2:]:
            raw_delta = F.interpolate(
                raw_delta, size=target_delta.shape[-2:], mode="bilinear",
                align_corners=False)
        predicted_delta = limit * torch.tanh(raw_delta[:, 0])
    loss = F.smooth_l1_loss(predicted_delta.float(), target_delta.detach())
    return loss, {
        "direct_history_tke_target_abs_mean": float(
            target_delta.detach().abs().mean()),
        "direct_history_tke_prediction_abs_mean": float(
            predicted_delta.detach().abs().mean()),
        "direct_history_tke_target_saturation_fraction": float(
            target_delta.detach().abs().eq(limit).float().mean()),
    }


def official_tke_score_loss(
        prediction_normalized: torch.Tensor,
        target_normalized: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Differentiable official TKE score loss, averaged per sample.

    This deliberately matches ``accuracy_objective``: temporal fluctuations
    are converted to physical u/v, reduced to one spatial kinetic-energy map,
    and transformed by ``1 - 1 / (1 + 0.5 * relative_error)``.
    """
    prediction = rehearsal.ensemble_train.common.to_physical_target(
        prediction_normalized, normalizer)[..., :2]
    target = rehearsal.ensemble_train.common.to_physical_target(
        target_normalized, normalizer)[..., :2]
    prediction_fluctuation = (
        prediction - prediction.mean(dim=1, keepdim=True))
    target_fluctuation = target - target.mean(dim=1, keepdim=True)
    prediction_tke = 0.5 * prediction_fluctuation.square().mean(
        dim=1).sum(dim=-1)
    target_tke = 0.5 * target_fluctuation.square().mean(
        dim=1).sum(dim=-1)
    batch = prediction_tke.shape[0]
    relative_error = torch.linalg.vector_norm(
        (prediction_tke - target_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        target_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)
    return (1.0 - 1.0 / (1.0 + 0.5 * relative_error)).mean()


def output_tke_projection(
        primary_loss: torch.Tensor, tke_loss: torch.Tensor,
        prediction: torch.Tensor,
        ) -> tuple[torch.Tensor | None, dict[str, float | bool]]:
    """Return the minimum output-gradient correction protecting TKE.

    If the primary update has a negative inner product with the TKE descent
    direction, remove exactly that conflicting component.  The returned
    tensor is an *additional* gradient for ``prediction``; it is ``None``
    when the primary gradient is already non-destructive.
    """
    primary_gradient = torch.autograd.grad(
        primary_loss, prediction, retain_graph=True, allow_unused=False)[0]
    tke_gradient = torch.autograd.grad(
        tke_loss, prediction, retain_graph=True, allow_unused=False)[0]
    primary_squared = primary_gradient.float().square().sum()
    tke_squared = tke_gradient.float().square().sum()
    dot = (primary_gradient.float() * tke_gradient.float()).sum()
    denominator = (primary_squared.sqrt() * tke_squared.sqrt()).clamp_min(
        torch.finfo(torch.float32).tiny)
    cosine = dot / denominator
    finite = bool(
        torch.isfinite(primary_squared)
        and torch.isfinite(tke_squared)
        and torch.isfinite(dot)
        and float(tke_squared) > 0.0)
    if not finite:
        raise FloatingPointError("non-finite B210 Pareto gradient audit")
    applied = bool(float(dot) < 0.0)
    correction = None
    correction_ratio = 0.0
    coefficient = 0.0
    if applied:
        coefficient_tensor = -dot / tke_squared.clamp_min(
            torch.finfo(torch.float32).tiny)
        correction = (
            coefficient_tensor.to(tke_gradient.dtype) * tke_gradient)
        coefficient = float(coefficient_tensor)
        correction_ratio = float(
            correction.float().square().sum().sqrt()
            / primary_squared.sqrt().clamp_min(
                torch.finfo(torch.float32).tiny))
    return correction, {
        "pareto_projection_applied": applied,
        "pareto_primary_tke_cosine": float(cosine),
        "pareto_primary_gradient_norm": float(primary_squared.sqrt()),
        "pareto_tke_gradient_norm": float(tke_squared.sqrt()),
        "pareto_projection_coefficient": coefficient,
        "pareto_correction_to_primary_ratio": correction_ratio,
        "pareto_projection_finite": finite,
    }


def output_legacy_tke_gradient_floor(
        primary_loss: torch.Tensor, legacy_reference_loss: torch.Tensor,
        tke_loss: torch.Tensor, prediction: torch.Tensor,
        ) -> tuple[torch.Tensor | None, dict[str, float | bool]]:
    """Restore only TKE-parallel strength missing from the legacy objective.

    Gradients are compared per sample.  The official primary keeps every
    orthogonal component; a correction along the unit TKE gradient is added
    only where its TKE-parallel component is smaller than the hypothetical
    legacy ``(1,4,1)`` non-auxiliary gradient on the same state and batch.
    """
    primary_gradient = torch.autograd.grad(
        primary_loss, prediction, retain_graph=True, allow_unused=False)[0]
    legacy_gradient = torch.autograd.grad(
        legacy_reference_loss, prediction,
        retain_graph=True, allow_unused=False)[0]
    tke_gradient = torch.autograd.grad(
        tke_loss, prediction, retain_graph=True, allow_unused=False)[0]
    shape = primary_gradient.shape
    batch = shape[0]
    primary_flat = primary_gradient.float().reshape(batch, -1)
    legacy_flat = legacy_gradient.float().reshape(batch, -1)
    tke_flat = tke_gradient.float().reshape(batch, -1)
    tke_norm = torch.linalg.vector_norm(tke_flat, dim=1).clamp_min(
        torch.finfo(torch.float32).tiny)
    tke_unit = tke_flat / tke_norm[:, None]
    primary_parallel = (primary_flat * tke_unit).sum(dim=1)
    legacy_parallel = (legacy_flat * tke_unit).sum(dim=1).clamp_min(0.0)
    missing = (legacy_parallel - primary_parallel).clamp_min(0.0)
    primary_norm = torch.linalg.vector_norm(primary_flat, dim=1).clamp_min(
        torch.finfo(torch.float32).tiny)
    ratios = missing / primary_norm
    finite = bool(all(torch.isfinite(value).all() for value in (
        primary_parallel, legacy_parallel, missing, ratios)))
    if not finite:
        raise FloatingPointError("non-finite legacy TKE gradient floor")
    applied = missing > 0.0
    correction = None
    if bool(applied.any()):
        correction = (
            missing[:, None] * tke_unit
        ).reshape(shape).to(tke_gradient.dtype)
    return correction, {
        "legacy_tke_floor_finite": finite,
        "legacy_tke_floor_applied_fraction": float(applied.float().mean()),
        "legacy_tke_floor_primary_parallel_median": float(
            primary_parallel.median()),
        "legacy_tke_floor_reference_parallel_median": float(
            legacy_parallel.median()),
        "legacy_tke_floor_missing_parallel_median": float(missing.median()),
        "legacy_tke_floor_correction_ratio_median": float(ratios.median()),
        "legacy_tke_floor_correction_ratio_max": float(ratios.max()),
    }


def backward_with_controller_only_auxiliary(
        non_auxiliary_loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        controller_parameters: Sequence[torch.nn.Parameter],
        auxiliary_weight: float,
        prediction: torch.Tensor | None = None,
        tke_protection_loss: torch.Tensor | None = None,
        legacy_reference_loss: torch.Tensor | None = None,
        ) -> dict[str, float]:
    """Backpropagate native loss, optional Pareto correction, and head aux."""
    if auxiliary_weight < 0.0 or not math.isfinite(auxiliary_weight):
        raise ValueError("B210 TKE auxiliary weight must be finite/non-negative")
    controller_parameters = tuple(
        parameter for parameter in controller_parameters
        if parameter.requires_grad)
    pareto_enabled = prediction is not None or tke_protection_loss is not None
    if pareto_enabled and (
            prediction is None or tke_protection_loss is None):
        raise ValueError(
            "Pareto projection requires prediction and TKE protection loss")
    legacy_floor_enabled = legacy_reference_loss is not None
    if legacy_floor_enabled and not pareto_enabled:
        raise ValueError(
            "legacy TKE floor requires prediction and TKE protection loss")
    pareto = {
        "pareto_projection_applied": False,
        "pareto_primary_tke_cosine": 0.0,
        "pareto_primary_gradient_norm": 0.0,
        "pareto_tke_gradient_norm": 0.0,
        "pareto_projection_coefficient": 0.0,
        "pareto_correction_to_primary_ratio": 0.0,
        "pareto_projection_finite": True,
    }
    correction = None
    legacy_floor = {
        "legacy_tke_floor_finite": True,
        "legacy_tke_floor_applied_fraction": 0.0,
        "legacy_tke_floor_primary_parallel_median": 0.0,
        "legacy_tke_floor_reference_parallel_median": 0.0,
        "legacy_tke_floor_missing_parallel_median": 0.0,
        "legacy_tke_floor_correction_ratio_median": 0.0,
        "legacy_tke_floor_correction_ratio_max": 0.0,
    }
    if legacy_floor_enabled:
        correction, legacy_floor = output_legacy_tke_gradient_floor(
            non_auxiliary_loss, legacy_reference_loss,
            tke_protection_loss, prediction)
    elif pareto_enabled:
        correction, pareto = output_tke_projection(
            non_auxiliary_loss, tke_protection_loss, prediction)
    non_auxiliary_loss.backward(
        retain_graph=auxiliary_weight > 0.0 or correction is not None)
    if correction is not None:
        prediction.backward(
            correction, retain_graph=auxiliary_weight > 0.0)
    native_norm = gradient_l2_from_parameters(controller_parameters)
    auxiliary_norm = 0.0
    ratio = 0.0
    if auxiliary_weight > 0.0:
        if not controller_parameters:
            raise RuntimeError("B210 TKE auxiliary has no trainable controller")
        gradients = torch.autograd.grad(
            auxiliary_weight * auxiliary_loss,
            controller_parameters, allow_unused=False)
        auxiliary_norm = gradient_l2_from_tensors(gradients)
        ratio = auxiliary_norm / max(
            native_norm, torch.finfo(torch.float32).tiny)
        add_parameter_gradients(controller_parameters, gradients)
    return {
        "native_controller_gradient_norm": native_norm,
        "tke_aux_gradient_norm": auxiliary_norm,
        "tke_aux_to_native_gradient_ratio": ratio,
        **pareto,
        **legacy_floor,
    }


def tke_aux_weight(iteration: int, persistent_floor: bool = False) -> float:
    """Cosine-decay the calibrated scale to zero or a fixed controller floor."""
    if iteration < 1:
        raise ValueError("B210 Sim iteration must be positive")
    floor = TKE_AUX_PERSISTENT_FLOOR_WEIGHT if persistent_floor else 0.0
    if iteration > TKE_AUX_UPDATES:
        return floor
    phase = (iteration - 1) / (TKE_AUX_UPDATES - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
    return floor + (TKE_AUX_WEIGHT - floor) * cosine


def hgrad_weight(iteration: int) -> float:
    return HGRAD_WEIGHT * min(1.0, iteration / HGRAD_RAMP_UPDATES)


def _controller_gradient_flags(model: EnsembleUNet2d) -> dict[str, bool]:
    result = {
        "projection_nonzero": False,
        "front_nonzero": False,
        "cross_projection_nonzero": False,
        "cross_body_nonzero": False,
        "temporal_mean_nonzero": False,
        "temporal_fluctuation_nonzero": False,
        "temporal_amplitude_nonzero": False,
        "temporal_projection_nonzero": False,
        "temporal_body_nonzero": False,
        "transport_output_nonzero": False,
        "transport_body_nonzero": False,
    }
    for name, parameter in model.named_parameters():
        if not name.startswith("energy_controller.") or parameter.grad is None:
            continue
        nonzero = bool(torch.count_nonzero(parameter.grad))
        if name.startswith("energy_controller.body.5."):
            result["projection_nonzero"] |= nonzero
        else:
            result["front_nonzero"] |= nonzero
    for name, parameter in model.named_parameters():
        if not name.startswith("cross_conditioner.") or parameter.grad is None:
            continue
        nonzero = bool(torch.count_nonzero(parameter.grad))
        if name.startswith("cross_conditioner.projections."):
            result["cross_projection_nonzero"] |= nonzero
        elif name.startswith("cross_conditioner.body."):
            result["cross_body_nonzero"] |= nonzero
    for name, parameter in model.named_parameters():
        if (not name.startswith("temporal_point_decoder.")
                or parameter.grad is None):
            continue
        nonzero = bool(torch.count_nonzero(parameter.grad))
        if name.startswith("temporal_point_decoder.member_mean_heads."):
            result["temporal_mean_nonzero"] |= nonzero
        elif name.startswith(
                "temporal_point_decoder.member_fluctuation_heads."):
            result["temporal_fluctuation_nonzero"] |= nonzero
        elif name.startswith(
                "temporal_point_decoder.member_amplitude_heads."):
            result["temporal_amplitude_nonzero"] |= nonzero
        elif name.startswith("temporal_point_decoder.projections."):
            result["temporal_projection_nonzero"] |= nonzero
        else:
            result["temporal_body_nonzero"] |= nonzero
    for name, parameter in model.named_parameters():
        if (not name.startswith("temporal_transport.")
                or parameter.grad is None):
            continue
        nonzero = bool(torch.count_nonzero(parameter.grad))
        if name.startswith("temporal_transport.output."):
            result["transport_output_nonzero"] |= nonzero
        else:
            result["transport_body_nonzero"] |= nonzero
    return result


def transform_teacher_for_real_closure(
        teacher_physical_uv: torch.Tensor,
        target_delta_physical_uv: torch.Tensor | None,
        ) -> torch.Tensor:
    """Apply the identical physical target displacement to a P1 teacher."""
    if target_delta_physical_uv is None:
        return teacher_physical_uv
    if teacher_physical_uv.shape != target_delta_physical_uv.shape:
        raise ValueError("closure target displacement is not teacher-aligned")
    if teacher_physical_uv.shape[-1] != 2:
        raise ValueError("closure teacher transformation is restricted to u/v")
    return teacher_physical_uv.float() + target_delta_physical_uv.float()


def closure_equivariance_target(
        clean_prediction_normalized: torch.Tensor,
        target_delta_physical_uv: torch.Tensor,
        normalizer,
        ) -> torch.Tensor:
    """Translate a detached clean prediction by the physical closure target."""
    if (clean_prediction_normalized.shape[:-1]
            != target_delta_physical_uv.shape[:-1]
            or clean_prediction_normalized.shape[-1] != 3
            or target_delta_physical_uv.shape[-1] != 2):
        raise ValueError("closure equivariance tensors are not aligned")
    delta = torch.zeros_like(clean_prediction_normalized)
    delta[..., :2] = (
        target_delta_physical_uv.float() / normalizer.std_target[:2])
    return clean_prediction_normalized.detach().float() + delta


def sim_update(
        model: EnsembleUNet2d, batch, normalizer, soft_sps_normalizer,
        teacher_cache: torch.Tensor | None, scale_contract: dict,
        args: argparse.Namespace, device: torch.device, iteration: int,
        optimizer, scheduler, arm: str,
        closure_augmentor: ClosureResidualAugmentor | None = None,
        closure_equivariance: bool = False,
        multiscale_tke_supervisor: MultiScaleTKEDeepSupervisor | None = None,
        ) -> tuple[dict[str, float], dict | None, str, dict[str, bool]]:
    if teacher_cache is None and args.distill_weight != 0:
        raise ValueError("absent teacher requires zero distillation weight")
    model.train()
    history, target, trajectory_ids, sample_indices = batch
    key = sim_hgrad.batch_key(trajectory_ids, sample_indices)
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    closure_selected = torch.zeros(
        history.shape[0], device=device, dtype=torch.bool)
    conditional_future_selected = torch.zeros(
        history.shape[0], device=device, dtype=torch.bool)
    closure_target_delta_uv = None
    original_history = None
    if closure_augmentor is not None:
        if arm != "candidate":
            raise ValueError("Real closure is restricted to the candidate arm")
        original_history = history.clone() if closure_equivariance else None
        original_target_uv = target[..., :2].clone()
        if hasattr(closure_augmentor, "set_batch_context"):
            closure_augmentor.set_batch_context(sample_indices, iteration)
        with torch.no_grad():
            history, target, closure_selected = closure_augmentor.augment(
                history, target, trajectory_ids,
                probability=REAL_CLOSURE_PROBABILITY,
                amplitude=REAL_CLOSURE_AMPLITUDE,
                closure_decay=REAL_CLOSURE_DECAY,
                perturbation_decay=REAL_CLOSURE_PERTURBATION_DECAY,
                advection_cells=REAL_CLOSURE_ADVECTION_CELLS)
        closure_target_delta_uv = target[..., :2] - original_target_uv
    history_norm, target_norm = normalizer.preprocess(history, target)
    distribution_target = target
    conditional_future_crps_mean_preserving = bool(
        getattr(args, "conditional_future_crps_mean_preserving", False))
    isolate_distribution_tower = bool(getattr(
        args, "isolate_distribution_tower", False))
    budgeted_disagreement_allocation = bool(getattr(
        args, "budgeted_disagreement_allocation", False))
    conditional_future_adapter_only = bool(
        getattr(args, "conditional_future_energy_adapter_only", False)
        or conditional_future_crps_mean_preserving)
    conditional_future_members_only = bool(
        getattr(args, "conditional_future_energy_members_only", False)
        or conditional_future_adapter_only)
    conditional_future_enabled = bool(
        arm == "candidate" and (
            getattr(args, "conditional_future_energy", False)
            or conditional_future_members_only
            or conditional_future_adapter_only))
    if conditional_future_enabled:
        if closure_augmentor is None:
            raise RuntimeError("conditional future energy has no closure bank")
        # Augmentation sampling is isolated from the model/dropout stream.  A
        # fixed iteration seed makes every branch replayable without changing
        # any random draw consumed by the baseline training pipeline.
        cuda_devices = [
            device.index
            if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            branch_seed = SEED + CONDITIONAL_FUTURE_RNG_OFFSET * iteration
            torch.manual_seed(branch_seed)
            torch.cuda.manual_seed_all(branch_seed)
            with torch.no_grad():
                distribution_target, conditional_future_selected = (
                    closure_augmentor.augment_future(
                        history, target, trajectory_ids,
                        probability=CONDITIONAL_FUTURE_PROBABILITY,
                        amplitude=CONDITIONAL_FUTURE_AMPLITUDE,
                        closure_decay=REAL_CLOSURE_DECAY,
                        perturbation_decay=REAL_CLOSURE_PERTURBATION_DECAY,
                        advection_cells=REAL_CLOSURE_ADVECTION_CELLS))
    _, distribution_target_norm = normalizer.preprocess(
        history, distribution_target)
    member_target_norm, uq_target_norm = conditional_distribution_targets(
        target_norm, distribution_target_norm,
        members_only=conditional_future_members_only)
    clean_prediction = None
    if closure_equivariance:
        if closure_augmentor is None or original_history is None:
            raise RuntimeError("closure equivariance has no paired clean input")
        clean_history_norm, _unused_target = normalizer.preprocess(
            original_history, target)
        cuda_rng = torch.cuda.get_rng_state(device)
        with torch.no_grad():
            clean_prediction = model(clean_history_norm)
        # The augmented forward consumes the exact dropout mask used by the
        # clean partner; the clean diagnostic itself does not advance RNG.
        torch.cuda.set_rng_state(cuda_rng, device)
    optimizer.zero_grad(set_to_none=True)
    (prediction, members, quantiles, lower, upper, features,
     coverage_logits) = model.forward_uq_training(history_norm)
    point_loss, point_components = rehearsal.ensemble_train.point_loss(
        args, prediction, target_norm, normalizer,
        tuple(float(value) for value in args.accuracy_weights), iteration)
    member_integrated_floor = bool(
        arm == "candidate" and getattr(
            args, "member_integrated_legacy_tke_gradient_floor", False))
    member_official_objective = bool(
        arm == "candidate" and (
            getattr(args, "member_official_objective", False)
            or member_integrated_floor))
    member_supervision_members = members
    if isolate_distribution_tower:
        member_supervision_members = model._members_from_features(
            history_norm, features, use_member_adapters=False)
        if model.config.physical_energy_member_sync:
            member_supervision_members = model._energy_controlled_members(
                member_supervision_members, features, history_norm)
    if member_official_objective:
        member_loss, bootstrap_fraction = (
            rehearsal.ensemble_train.bootstrap_member_point_loss(
                args, member_supervision_members, target_norm,
                trajectory_ids, normalizer,
                OFFICIAL_POINT_ACCURACY_WEIGHTS, iteration))
    else:
        member_loss, bootstrap_fraction = (
            rehearsal.ensemble_train.bootstrap_member_mse(
                member_supervision_members, target_norm, trajectory_ids,
                args.bootstrap_probability, args.seed))
    energy_score = prediction.new_zeros(())
    energy_attraction = prediction.new_zeros(())
    energy_repulsion = prediction.new_zeros(())
    crps_score = prediction.new_zeros(())
    crps_attraction = prediction.new_zeros(())
    crps_repulsion = prediction.new_zeros(())
    future_score_weight = 0.0
    if conditional_future_enabled:
        energy_score, energy_attraction, energy_repulsion = (
            functional_energy_score(members, member_target_norm))
        crps_score, crps_attraction, crps_repulsion = (
            pointwise_ensemble_crps(members, member_target_norm))
        future_score_weight = (
            CONDITIONAL_FUTURE_CRPS_WEIGHT
            if conditional_future_crps_mean_preserving
            else CONDITIONAL_FUTURE_ENERGY_WEIGHT)
    future_distribution_score = (
        crps_score if conditional_future_crps_mean_preserving
        else energy_score)
    uq_center = prediction.detach() if isolate_distribution_tower else prediction
    quantile_loss = rehearsal.ensemble_train.residual_quantile_loss(
        model, uq_center, quantiles, uq_target_norm, normalizer)
    soft_sps_loss, soft_sps_score = (
        rehearsal.ensemble_train.soft_native_sps_objective(
            uq_center, uq_target_norm, lower, upper,
            soft_sps_normalizer,
            args.soft_sps_tau_fraction, metric_normalizer=normalizer))
    coverage_bce = (
        rehearsal.ensemble_train.coverage_calibration_objective(
            model, uq_center, quantiles, coverage_logits,
            uq_target_norm, normalizer)
        if coverage_logits is not None else prediction.new_zeros(()))
    base_loss = (
        args.mean_loss_weight * point_loss
        + args.member_loss_weight * member_loss
        + (0.0 if conditional_future_adapter_only else future_score_weight)
        * future_distribution_score
        + args.quantile_loss_weight * quantile_loss
        + args.coverage_bce_weight * coverage_bce)
    if teacher_cache is None:
        distillation, distillation_components = prediction.new_zeros(()), {}
    else:
        teacher = teacher_cache[sample_indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        if closure_target_delta_uv is not None:
            # Preserve the frozen teacher's residual relative to the transformed
            # Sim target.  Distilling to the untransformed teacher would oppose the
            # domain-aligned counterfactual on exactly the augmented samples.
            teacher = transform_teacher_for_real_closure(
                teacher, closure_target_delta_uv)
        distillation, distillation_components = (
            rehearsal.ensemble_train.component_point_distillation_loss(
                prediction, teacher, normalizer))
    supervised_native = base_loss + args.distill_weight * distillation
    soft_sps_weight = args.soft_sps_loss_weight * min(
        1.0, iteration / args.soft_sps_warmup_updates)
    native_total = supervised_native + soft_sps_weight * soft_sps_loss

    equivariance_loss = prediction.new_zeros(())
    equivariance_weight = (
        REAL_CLOSURE_EQUIVARIANCE_WEIGHT if closure_equivariance else 0.0)
    if closure_equivariance and bool(closure_selected.any()):
        expected = closure_equivariance_target(
            clean_prediction, closure_target_delta_uv, normalizer)
        equivariance_loss = F.smooth_l1_loss(
            prediction[closure_selected].float(),
            expected[closure_selected])
    native_total = native_total + equivariance_weight * equivariance_loss

    spatial_loss = sim_hgrad.hgrad.hgrad_loss(
        prediction, target_norm, normalizer, scale_contract)
    spatial_weight = hgrad_weight(iteration) if arm == "candidate" else 0.0
    weighted_hgrad = spatial_weight * spatial_loss
    multiscale_tke_loss = prediction.new_zeros(())
    multiscale_tke_level_losses: list[torch.Tensor] = []
    multiscale_tke_weight = 0.0
    multiscale_tke_audit = None
    if multiscale_tke_supervisor is not None:
        multiscale_tke_loss, multiscale_tke_level_losses = (
            multiscale_tke_supervisor(features, target[..., :2]))
        multiscale_tke_weight = MULTISCALE_TKE_WEIGHT
    weighted_multiscale_tke = multiscale_tke_weight * multiscale_tke_loss
    non_auxiliary_loss = (
        native_total + weighted_hgrad + weighted_multiscale_tke)
    legacy_floor_enabled = bool(
        arm == "candidate"
        and getattr(args, "legacy_tke_gradient_floor", False))
    legacy_reference_loss = None
    if legacy_floor_enabled:
        official_accuracy, _official_details = (
            rehearsal.ensemble_train.common.accuracy_objective(
                prediction, target_norm, normalizer,
                OFFICIAL_POINT_ACCURACY_WEIGHTS))
        legacy_accuracy, _legacy_details = (
            rehearsal.ensemble_train.common.accuracy_objective(
                prediction, target_norm, normalizer,
                rehearsal.SIM_ACCURACY_WEIGHTS))
        legacy_reference_loss = (
            non_auxiliary_loss
            + args.mean_loss_weight * (
                legacy_accuracy - official_accuracy))
        if member_integrated_floor:
            legacy_member_loss, legacy_bootstrap_fraction = (
                rehearsal.ensemble_train.bootstrap_member_point_loss(
                    args, members, target_norm, trajectory_ids, normalizer,
                    rehearsal.SIM_ACCURACY_WEIGHTS, iteration))
            if not torch.equal(
                    bootstrap_fraction, legacy_bootstrap_fraction):
                raise RuntimeError(
                    "official and legacy member bootstrap fractions differ")
            legacy_reference_loss = (
                legacy_reference_loss
                + args.member_loss_weight * (
                    legacy_member_loss - member_loss))
    if (multiscale_tke_supervisor is not None
            and iteration == MULTISCALE_TKE_GRADIENT_AUDIT_UPDATE):
        multiscale_tke_audit = multiscale_tke_feature_gradient_audit(
            native_total + weighted_hgrad, weighted_multiscale_tke, features)

    auxiliary_loss = prediction.new_zeros(())
    direct_history_tke = bool(getattr(
        args, "direct_history_tke_regression", False))
    direct_history_metrics = {
        "direct_history_tke_target_abs_mean": 0.0,
        "direct_history_tke_prediction_abs_mean": 0.0,
        "direct_history_tke_target_saturation_fraction": 0.0,
    }
    train_only_temporal_band_residual = bool(getattr(
        args, "train_only_temporal_band_residual", False))
    auxiliary_weight = (
        0.0 if train_only_temporal_band_residual else (
            DIRECT_HISTORY_TKE_WEIGHT
            if arm == "candidate" and direct_history_tke else (
                tke_aux_weight(
                    iteration,
                    bool(getattr(
                        args, "official_point_controller_tke_floor", False)))
                if arm == "candidate" else 0.0)))
    if auxiliary_weight > 0.0:
        if direct_history_tke:
            with torch.no_grad():
                base_members = model._members_from_features(
                    history_norm, [feature.detach() for feature in features])
                base_prediction = base_members.mean(dim=1)
            auxiliary_loss, direct_history_metrics = (
                direct_history_tke_regression_loss(
                    model, base_prediction, target, history_norm,
                    features, normalizer))
        else:
            auxiliary_loss = (
                rehearsal.ensemble_train.physical_tke_map_distillation_loss(
                    prediction, target[..., :2], normalizer))
    adapter_energy_loss = (
        future_score_weight * future_distribution_score
        if conditional_future_enabled and conditional_future_adapter_only
        else prediction.new_zeros(()))
    total = (
        non_auxiliary_loss + auxiliary_weight * auxiliary_loss
        + adapter_energy_loss)
    if not torch.isfinite(total):
        raise FloatingPointError(
            f"non-finite B210 Sim loss at update {iteration}")

    hgrad_audit = None
    if arm == "candidate" and iteration in sim_hgrad.AUDIT_UPDATES:
        point_path = (
            args.mean_loss_weight * point_loss
            + args.distill_weight * distillation)
        hgrad_audit = sim_hgrad.hgrad.output_gradient_audit(
            point_path, weighted_hgrad, supervised_native, prediction)
        hgrad_audit["iteration"] = iteration

    pareto_enabled = bool(
        arm == "candidate"
        and (getattr(args, "pareto_tke_projection", False)
             or getattr(args, "legacy_tke_gradient_floor", False)))
    tke_protection_loss = (
        official_tke_score_loss(
            prediction, target_norm, normalizer)
        if pareto_enabled else None)
    adapter_energy_parameters: tuple[torch.nn.Parameter, ...] = ()
    adapter_energy_gradients: tuple[torch.Tensor | None, ...] = ()
    adapter_energy_gradient_norm = 0.0
    if conditional_future_enabled and conditional_future_adapter_only:
        if model.member_feature_adapters is None:
            raise RuntimeError("adapter-only future energy has no adapters")
        adapter_energy_parameters = tuple(
            parameter for parameter in model.member_feature_adapters.parameters()
            if parameter.requires_grad)
        adapter_energy_gradients = torch.autograd.grad(
            adapter_energy_loss, adapter_energy_parameters,
            retain_graph=True, allow_unused=False)
        adapter_energy_gradient_norm = gradient_l2_from_tensors(
            adapter_energy_gradients)
        # Legacy member projections are exact-zero at update zero, so the
        # adapters behind them intentionally have a zero gradient on the first
        # step.  The projections become live through base supervision; u200
        # separately requires this path to be finite and nonzero.
        if not math.isfinite(adapter_energy_gradient_norm):
            raise FloatingPointError(
                "adapter-only future energy gradient is invalid")
    gradient_diagnostic = backward_with_controller_only_auxiliary(
        non_auxiliary_loss, auxiliary_loss,
        tuple(model.energy_controller.parameters()), auxiliary_weight,
        prediction=(
            members if pareto_enabled and member_integrated_floor else (
                prediction if pareto_enabled else None)),
        tke_protection_loss=tke_protection_loss,
        legacy_reference_loss=legacy_reference_loss)
    native_adapter_gradient_norm = gradient_l2_from_parameters(
        adapter_energy_parameters)
    adapter_energy_to_native_ratio = (
        adapter_energy_gradient_norm / max(
            native_adapter_gradient_norm,
            torch.finfo(torch.float32).tiny)
        if adapter_energy_parameters else 0.0)
    if adapter_energy_parameters:
        add_parameter_gradients(
            adapter_energy_parameters, adapter_energy_gradients)
    controller_flags = _controller_gradient_flags(model)
    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(
            tuple(parameter for parameter in model.parameters()
                  if parameter.requires_grad), args.grad_clip)
    optimizer.step()
    scheduler.step()

    values = {
        "loss": float(total.detach()),
        "native_total_loss": float(native_total.detach()),
        "supervised_native_loss": float(supervised_native.detach()),
        "point_loss": float(point_loss.detach()),
        "member_loss": float(member_loss.detach()),
        "functional_energy_score": float(energy_score.detach()),
        "functional_energy_attraction": float(energy_attraction.detach()),
        "functional_energy_repulsion": float(energy_repulsion.detach()),
        "functional_energy_weight": (
            0.0 if conditional_future_crps_mean_preserving
            else future_score_weight),
        "pointwise_crps_score": float(crps_score.detach()),
        "pointwise_crps_attraction": float(crps_attraction.detach()),
        "pointwise_crps_repulsion": float(crps_repulsion.detach()),
        "pointwise_crps_weight": (
            future_score_weight
            if conditional_future_crps_mean_preserving else 0.0),
        "conditional_future_crps_mean_preserving": float(
            conditional_future_crps_mean_preserving),
        "detach_member_adapter_context": float(getattr(
            args, "detach_member_adapter_context", False)),
        "detach_uq_context": float(getattr(
            args, "detach_uq_context", False)),
        "isolate_distribution_tower": float(isolate_distribution_tower),
        "member_supervision_uses_base_members": float(
            isolate_distribution_tower),
        "budgeted_disagreement_allocation": float(
            budgeted_disagreement_allocation),
        "disagreement_allocation_strength_abs_max": (
            0.0 if model.disagreement_allocation_strength is None else float(
                model.disagreement_allocation_strength.detach().abs().max())),
        "disagreement_allocation_sharpness_abs_max": (
            0.0 if model.disagreement_allocation_sharpness is None else float(
                model.disagreement_allocation_sharpness.detach().abs().max())),
        "functional_energy_adapter_only": float(
            conditional_future_adapter_only),
        "functional_energy_adapter_gradient_norm": (
            adapter_energy_gradient_norm),
        "native_adapter_gradient_norm": native_adapter_gradient_norm,
        "functional_energy_to_native_adapter_gradient_ratio": (
            adapter_energy_to_native_ratio),
        "functional_energy_shared_gradient_by_construction": 0.0,
        "member_uv_normalized_spread": float(
            members[..., :2].float().std(
                dim=1, unbiased=False).mean().detach()),
        "quantile_loss": float(quantile_loss.detach()),
        "soft_sps_loss": float(soft_sps_loss.detach()),
        "soft_sps_score": float(soft_sps_score.detach()),
        "soft_sps_weight": soft_sps_weight,
        "p1_distill_loss": float(distillation.detach()),
        "hgrad_loss": float(spatial_loss.detach()),
        "hgrad_weight": spatial_weight,
        "weighted_hgrad_loss": float(weighted_hgrad.detach()),
        "multiscale_tke_loss": float(multiscale_tke_loss.detach()),
        "multiscale_tke_weight": multiscale_tke_weight,
        "weighted_multiscale_tke_loss": float(
            weighted_multiscale_tke.detach()),
        "multiscale_tke_level_losses": [
            float(value.detach()) for value in multiscale_tke_level_losses],
        "multiscale_tke_gradient_audit": multiscale_tke_audit,
        "tke_aux_loss": float(auxiliary_loss.detach()),
        "tke_aux_weight": auxiliary_weight,
        "pareto_tke_protection_loss": (
            0.0 if tke_protection_loss is None
            else float(tke_protection_loss.detach())),
        "direct_history_tke_regression": float(direct_history_tke),
        **direct_history_metrics,
        "bootstrap_fraction": float(bootstrap_fraction.detach()),
        "real_closure_selected_fraction": float(
            closure_selected.float().mean()),
        "real_closure_target_delta_uv_rms": (
            0.0 if closure_target_delta_uv is None else float(
                closure_target_delta_uv.float().square().mean().sqrt())),
        "real_closure_equivariance_loss": float(equivariance_loss.detach()),
        "real_closure_equivariance_weight": equivariance_weight,
        "conditional_future_selected_fraction": float(
            conditional_future_selected.float().mean()),
        "conditional_future_target_delta_uv_rms": float(
            (distribution_target[..., :2] - target[..., :2]).float()
            .square().mean().sqrt()),
        "conditional_future_quantile_uses_base": float(
            conditional_future_members_only),
        **gradient_diagnostic,
        **{key: float(value) for key, value in point_components.items()},
        **{key: float(value) for key, value in distillation_components.items()},
    }
    if closure_augmentor is not None and hasattr(
            closure_augmentor, "last_audit"):
        values.update({
            name: float(value)
            for name, value in closure_augmentor.last_audit.items()
        })
    return values, hgrad_audit, key, controller_flags


@torch.inference_mode()
def capture_update0(
        model: EnsembleUNet2d, dataset, normalizer,
        device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    history, target = dataset[0]
    history = history.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    history_norm, _target_norm = normalizer.preprocess(history, target)
    point, members, quantiles, lower, upper, _features, _coverage = (
        model.forward_uq_training(history_norm))
    return {
        "point": point.detach().cpu(),
        "members": members.detach().cpu(),
        "quantiles": quantiles.detach().cpu(),
        "lower": lower.detach().cpu(),
        "upper": upper.detach().cpu(),
    }


def mean_preserving_member_adapter_audit(
        model: EnsembleUNet2d, dataset, normalizer,
        device: torch.device) -> dict[str, float | bool]:
    """Verify that deployed point output has no member-adapter graph path."""
    if (model.member_feature_adapters is None
            or not model.config.mean_preserving_member_adapters):
        raise ValueError("mean-preserving adapter audit has no such adapters")
    was_training = model.training
    model.eval()
    history, target = dataset[0]
    history = history.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    history_norm, _ = normalizer.preprocess(history, target)
    prediction, members, *_rest = model.forward_uq_training(history_norm)
    parameters = tuple(
        parameter for parameter in model.member_feature_adapters.parameters()
        if parameter.requires_grad)
    gradients = torch.autograd.grad(
        prediction.float().sum(), parameters,
        allow_unused=True, retain_graph=False)
    point_adapter_gradient_none = all(value is None for value in gradients)
    with torch.no_grad():
        member_mean_gap = float(
            (members.mean(dim=1) - prediction).abs().max())
    model.train(was_training)
    return {
        "point_adapter_gradient_none": point_adapter_gradient_none,
        "member_mean_prediction_max_abs_gap": member_mean_gap,
        "member_mean_matches_prediction": member_mean_gap <= 2.0e-6,
        "pass": bool(
            point_adapter_gradient_none and member_mean_gap <= 2.0e-6),
    }


@torch.no_grad()
def temporal_transport_invariant_audit(
        model: EnsembleUNet2d, dataset, normalizer,
        device: torch.device) -> dict[str, float | bool]:
    """Check the deployed transport against its exact physical contract."""
    if model.temporal_transport is None:
        raise ValueError("temporal transport invariant audit has no module")
    model.eval()
    history, target = dataset[0]
    history = history.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    history_norm, _target_norm = normalizer.preprocess(history, target)
    features = model._decode(history_norm)
    members = model._members_from_features(history_norm, features)
    if model.config.physical_energy_member_sync:
        members = model._energy_controlled_members(
            members, features, history_norm)
        source = members.mean(dim=1)
    else:
        source = model._energy_controlled_prediction(
            members.mean(dim=1), features, history_norm)
    transported = model.temporal_transport(
        source, features[-1], history_norm)
    source_uv = source[..., :2].float()
    transported_uv = transported[..., :2].float()
    source_fluctuation = source_uv - source_uv.mean(dim=1, keepdim=True)
    transported_fluctuation = (
        transported_uv - transported_uv.mean(dim=1, keepdim=True))
    mean_error = float((
        transported_uv.mean(dim=1)
        - source_uv.mean(dim=1)).abs().max())
    energy_error = float((
        transported_fluctuation.square().mean(dim=1)
        - source_fluctuation.square().mean(dim=1)).abs().max())
    pressure_exact = bool(torch.equal(transported[..., 2:], source[..., 2:]))
    finite = bool(
        torch.isfinite(transported).all()
        and math.isfinite(mean_error) and math.isfinite(energy_error))
    preserve_energy = bool(model.config.isoenergetic_temporal_transport)
    return {
        "finite": finite,
        "mean_max_abs_error": mean_error,
        "energy_max_abs_error": energy_error,
        "energy_is_invariant": preserve_energy,
        "pressure_bitwise_exact": pressure_exact,
        "pass": bool(
            finite and mean_error <= 2e-5
            and (not preserve_energy or energy_error <= 2e-5)
            and pressure_exact),
    }


def update0_identity(first: Mapping[str, torch.Tensor],
                     second: Mapping[str, torch.Tensor]) -> dict:
    checks = {name: torch.equal(first[name], second[name]) for name in first}
    return {"pass": all(checks.values()), "checks": checks}


def ratio_diagnostic(ratios: Sequence[float]) -> dict:
    if len(ratios) != CONTROL_UPDATES:
        return {
            "pass": False, "samples": len(ratios), "median": None,
            "p95_diagnostic_only": None,
            "range": list(TKE_GRADIENT_RATIO_RANGE)}
    ordered = sorted(float(value) for value in ratios)
    median = float(statistics.median(ordered))
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    finite_nonnegative = all(
        math.isfinite(value) and value >= 0.0 for value in ordered)
    positive_samples = sum(value > 0.0 for value in ordered)
    passed = bool(
        finite_nonnegative
        and TKE_GRADIENT_RATIO_RANGE[0] <= median
        <= TKE_GRADIENT_RATIO_RANGE[1])
    return {
        "pass": passed,
        "samples": len(ordered),
        "median": median,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "positive_samples": positive_samples,
        "zero_samples_diagnostic_only": len(ordered) - positive_samples,
        "individual_zero_is_hard_gate": False,
        "p95_diagnostic_only": p95,
        "p95_is_hard_gate": False,
        "range": list(TKE_GRADIENT_RATIO_RANGE),
    }


def pareto_projection_summary(rows: Sequence[Mapping[str, float]]) -> dict:
    """Summarize the fixed u1--u200 output-gradient conflict audit."""
    expected = CONTROL_UPDATES
    finite = bool(
        len(rows) == expected
        and all(bool(row["pareto_projection_finite"]) for row in rows)
        and all(math.isfinite(float(row[key])) for row in rows for key in (
            "pareto_primary_tke_cosine",
            "pareto_primary_gradient_norm",
            "pareto_tke_gradient_norm",
            "pareto_correction_to_primary_ratio")))
    cosines = [float(row["pareto_primary_tke_cosine"]) for row in rows]
    ratios = [
        float(row["pareto_correction_to_primary_ratio"]) for row in rows]
    applied = sum(bool(row["pareto_projection_applied"]) for row in rows)
    conflict_fraction = applied / max(len(rows), 1)
    conflict_observed = bool(conflict_fraction >= 0.05)
    return {
        "samples": len(rows),
        "expected_samples": expected,
        "finite": finite,
        "projection_applied_updates": applied,
        "conflict_fraction": conflict_fraction,
        "conflict_observed": conflict_observed,
        "median_cosine": (
            float(statistics.median(cosines)) if cosines else None),
        "minimum_cosine": min(cosines) if cosines else None,
        "median_correction_to_primary_ratio": (
            float(statistics.median(ratios)) if ratios else None),
        "maximum_correction_to_primary_ratio": (
            max(ratios) if ratios else None),
        "continuation_rule": (
            "all 200 finite and at least 5% destructive updates"),
        "pass": bool(finite and conflict_observed),
    }


def legacy_tke_floor_summary(rows: Sequence[Mapping[str, float]]) -> dict:
    """Summarize the fixed u1--u200 legacy TKE magnitude-floor audit."""
    finite = bool(
        len(rows) == CONTROL_UPDATES
        and all(bool(row["legacy_tke_floor_finite"]) for row in rows)
        and all(math.isfinite(float(row[key])) for row in rows for key in (
            "legacy_tke_floor_applied_fraction",
            "legacy_tke_floor_primary_parallel_median",
            "legacy_tke_floor_reference_parallel_median",
            "legacy_tke_floor_missing_parallel_median",
            "legacy_tke_floor_correction_ratio_median",
            "legacy_tke_floor_correction_ratio_max")))
    applied_fraction = (
        statistics.mean(float(row["legacy_tke_floor_applied_fraction"])
                        for row in rows) if rows else 0.0)
    correction_ratios = [
        float(row["legacy_tke_floor_correction_ratio_median"])
        for row in rows]
    median_ratio = (
        float(statistics.median(correction_ratios))
        if correction_ratios else None)
    mechanically_active = bool(
        applied_fraction >= 0.50
        and median_ratio is not None
        and 0.10 <= median_ratio <= 5.00)
    return {
        "samples": len(rows),
        "expected_samples": CONTROL_UPDATES,
        "finite": finite,
        "mean_applied_fraction": applied_fraction,
        "median_correction_to_primary_ratio": median_ratio,
        "maximum_correction_to_primary_ratio": (
            max(float(row["legacy_tke_floor_correction_ratio_max"])
                for row in rows) if rows else None),
        "mechanically_active": mechanically_active,
        "continuation_rule": (
            "200 finite updates, >=50% sample activation, and median "
            "correction/main ratio in [0.10,5.00]"),
        "pass": bool(finite and mechanically_active),
    }


def prefix_gate(
        initial: dict, control: dict, candidate: dict,
        gradient_ratios: Sequence[float], update0: dict,
        controller_audit: Mapping[str, bool], hgrad_audits: list[dict],
        ) -> dict:
    ratio = ratio_diagnostic(gradient_ratios)
    component_delta = {
        key: candidate[key] - control[key] for key in POINT_COMPONENT_KEYS}
    point_delta = candidate["point_score"] - control["point_score"]
    sps_delta = (
        candidate["native_soft_sps_proxy"]
        - control["native_soft_sps_proxy"])
    hgrad_advantage = (
        (control["hgrad_loss"] - candidate["hgrad_loss"])
        / max(abs(control["hgrad_loss"]), 1e-12))
    p1_ratio = (
        candidate["p1_distill_loss"]
        / max(abs(control["p1_distill_loss"]), 1e-12))
    finite_metrics = all(math.isfinite(float(value)) for value in (
        *component_delta.values(), point_delta, sps_delta,
        hgrad_advantage, p1_ratio,
        candidate["hgrad_loss"], candidate["p1_distill_loss"]))
    hgrad_screen = sim_hgrad.hgrad.evaluate_preflight(hgrad_audits)

    health_checks = {
        "finite_metrics": finite_metrics,
        "update0_bitwise_identity": bool(update0.get("pass")),
        "tke_controller_gradient_ratio": bool(ratio["pass"]),
        "controller_state_updated": bool(
            controller_audit.get("state_updated")),
        "controller_projection_updated": bool(
            controller_audit.get("projection_updated")),
        "controller_front_updated": bool(controller_audit.get("front_updated")),
        "controller_projection_gradient_seen": bool(
            controller_audit.get("projection_gradient_seen")),
        "controller_front_gradient_seen": bool(
            controller_audit.get("front_gradient_seen")),
        "point_not_catastrophic": (
            point_delta >= CATASTROPHIC_POINT_DELTA_FLOOR),
        "components_not_catastrophic": all(
            value >= CATASTROPHIC_COMPONENT_DELTA_FLOOR
            for value in component_delta.values()),
        "sps_not_catastrophic": (
            sps_delta >= CATASTROPHIC_SPS_DELTA_FLOOR),
        "hgrad_audits_finite": (
            [row.get("iteration") for row in hgrad_audits]
            == list(sim_hgrad.AUDIT_UPDATES)
            and all(row.get("finite_nonzero") is True
                    for row in hgrad_audits)),
    }
    directional_checks = {
        "tke_nonnegative": (
            component_delta["tke_rel_l2_score"]
            >= DIRECTIONAL_TKE_DELTA_MIN),
        "hgrad_advantage": (
            hgrad_advantage >= DIRECTIONAL_HGRAD_ADVANTAGE_MIN),
        "point_near_control": point_delta >= DIRECTIONAL_POINT_DELTA_FLOOR,
        "components_near_control": all(
            value >= DIRECTIONAL_COMPONENT_DELTA_FLOOR
            for value in component_delta.values()),
        "sps_near_control": sps_delta >= DIRECTIONAL_SPS_DELTA_FLOOR,
        "p1_near_control": p1_ratio <= DIRECTIONAL_P1_RATIO_MAX,
        "legacy_hgrad_screen": bool(hgrad_screen["pass"]),
    }
    health_pass = all(health_checks.values())
    return {
        "status": (
            HEALTH_PASS_STATUS if health_pass else HEALTH_STOP_STATUS),
        "health_pass": health_pass,
        "health_checks": health_checks,
        "directional_pass": all(directional_checks.values()),
        "directional_checks": directional_checks,
        "directional_is_hard_gate": False,
        "continuation_rule": "continue_if_and_only_if_health_pass",
        "gradient_ratio": ratio,
        "hgrad_screen_diagnostic": hgrad_screen,
        "initial_anchor": initial,
        "control_anchor": control,
        "candidate_anchor": candidate,
        "component_delta": component_delta,
        "point_delta": point_delta,
        "soft_sps_proxy_delta": sps_delta,
        "hgrad_relative_advantage": hgrad_advantage,
        "p1_distill_ratio": p1_ratio,
    }


def make_optimizer(
        model: EnsembleUNet2d, args: argparse.Namespace, arm: str,
        multiscale_tke_supervisor: MultiScaleTKEDeepSupervisor | None = None,
        ):
    trainable = configure_trainable(model, arm)
    if multiscale_tke_supervisor is not None:
        if arm != "candidate":
            raise ValueError("multiscale TKE supervisor is candidate-only")
        trainable = trainable + tuple(
            multiscale_tke_supervisor.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TERMINAL_UPDATE)
    return optimizer, scheduler


def _deadline_check(deadline: float, iteration: int) -> None:
    if time.monotonic() > deadline:
        raise WallBudgetExceeded(
            f"B210 Sim wall budget exceeded before update {iteration}")


def save_terminal_checkpoint(
        output_dir: Path, model: EnsembleUNet2d,
        args: argparse.Namespace, history: list[dict], normalizer,
        ) -> tuple[str, str]:
    args.backbone_residual_dropout = BACKBONE_DROPOUT
    args.physical_energy_controller = True
    args.physical_energy_member_sync = True
    args.physical_energy_multiscale = bool(getattr(
        args, "multiscale_energy_controller", False))
    args.physical_energy_horizon_local = bool(getattr(
        args, "horizon_local_energy_controller", False))
    args.physical_energy_history_conditioned = bool(getattr(
        args, "history_energy_controller", False))
    args.physical_energy_history_steps = 5
    args.direct_history_tke_regression = bool(getattr(
        args, "direct_history_tke_regression", False))
    args.hybrid_history_multiscale_energy_controller = bool(getattr(
        args, "hybrid_history_multiscale_energy_controller", False))
    args.cross_conditioned_backbone = bool(getattr(
        args, "cross_conditioned_backbone", False))
    args.temporal_point_decoder = bool(getattr(
        args, "temporal_point_decoder", False))
    args.temporal_point_amplitude_factorization = bool(getattr(
        args, "temporal_point_amplitude_factorization", False))
    args.isoenergetic_temporal_transport = bool(getattr(
        args, "isoenergetic_temporal_transport", False))
    args.temporal_band_mixer = bool(getattr(
        args, "temporal_band_mixer", False))
    args.temporal_band_channels = TEMPORAL_BAND_CHANNELS
    args.temporal_band_count = TEMPORAL_BAND_COUNT
    args.temporal_band_log_gain_limit = TEMPORAL_BAND_LOG_GAIN_LIMIT
    args.temporal_band_phase_limit = TEMPORAL_BAND_PHASE_LIMIT
    args.decoupled_member_residual = bool(getattr(
        args, "decoupled_member_residual", False))
    args.decoupled_member_residual_channels = (
        DECOUPLED_MEMBER_RESIDUAL_CHANNELS
        if args.decoupled_member_residual else 0)
    args.decoupled_member_residual_blocks = (
        DECOUPLED_MEMBER_RESIDUAL_BLOCKS)
    args.member_feature_adapters = bool(getattr(
        args, "member_feature_adapters", False))
    args.standard_objective_member_feature_adapters = bool(getattr(
        args, "standard_objective_member_feature_adapters", False))
    args.fluctuation_role_tower = bool(getattr(
        args, "fluctuation_role_tower", False))
    args.fluctuation_role_channels = (
        FLUCTUATION_ROLE_CHANNELS if args.fluctuation_role_tower else 0)
    args.member_adapter_channels = (
        MEMBER_FEATURE_ADAPTER_CHANNELS
        if args.member_feature_adapters else 0)
    args.conditional_future_energy = bool(getattr(
        args, "conditional_future_energy", False))
    args.conditional_future_energy_members_only = bool(getattr(
        args, "conditional_future_energy_members_only", False))
    args.conditional_future_energy_adapter_only = bool(getattr(
        args, "conditional_future_energy_adapter_only", False))
    args.conditional_future_crps_mean_preserving = bool(getattr(
        args, "conditional_future_crps_mean_preserving", False))
    args.mean_preserving_member_adapters = bool(getattr(
        args, "mean_preserving_member_adapters", False))
    args.detach_member_adapter_context = bool(getattr(
        args, "detach_member_adapter_context", False))
    args.detach_uq_context = bool(getattr(
        args, "detach_uq_context", False))
    args.isolate_distribution_tower = bool(getattr(
        args, "isolate_distribution_tower", False))
    args.budgeted_disagreement_allocation = bool(getattr(
        args, "budgeted_disagreement_allocation", False))
    args.hgrad_weight = HGRAD_WEIGHT
    args.hgrad_ramp_updates = HGRAD_RAMP_UPDATES
    args.tke_aux_weight = TKE_AUX_WEIGHT
    args.tke_aux_updates = TKE_AUX_UPDATES
    args.multiscale_tke_deep_supervision = bool(getattr(
        args, "multiscale_tke_deep_supervision", False))
    args.multiscale_tke_weight = (
        MULTISCALE_TKE_WEIGHT
        if args.multiscale_tke_deep_supervision else 0.0)
    args.multiscale_tke_heads_saved = False
    args.u200_health_pass = True
    path = output_dir / "model_10000.pth"
    rehearsal.ensemble_train.common.save_checkpoint(
        path, model, args, TERMINAL_UPDATE, history,
        rehearsal.ensemble_train.common.normalizer_dict(normalizer),
        {"status": PASS_STATUS})
    return str(path), sim_hgrad.sha256(path)


def run_experiment(
        args: argparse.Namespace, sim_args: argparse.Namespace,
        raw_loader, normalizer, soft_sps_normalizer,
        teacher_cache: torch.Tensor, teacher_metadata: dict,
        scale_contract: dict, device: torch.device,
        closure_augmentor: ClosureResidualAugmentor | None = None,
        closure_equivariance: bool = False,
        ) -> dict:
    started = time.monotonic()
    deadline = started + WALL_BUDGET_SECONDS
    dataset = raw_loader.dataset
    multiscale_energy_controller = bool(getattr(
        sim_args, "multiscale_energy_controller", False))
    horizon_local_energy_controller = bool(getattr(
        sim_args, "horizon_local_energy_controller", False))
    history_energy_controller = bool(getattr(
        sim_args, "history_energy_controller", False))
    hybrid_history_multiscale_energy_controller = bool(getattr(
        sim_args, "hybrid_history_multiscale_energy_controller", False))
    cross_conditioned_backbone = bool(getattr(
        sim_args, "cross_conditioned_backbone", False))
    temporal_point_decoder = bool(getattr(
        sim_args, "temporal_point_decoder", False))
    temporal_point_amplitude_factorization = bool(getattr(
        sim_args, "temporal_point_amplitude_factorization", False))
    isoenergetic_temporal_transport = bool(getattr(
        sim_args, "isoenergetic_temporal_transport", False))
    temporal_band_mixer = bool(getattr(
        sim_args, "temporal_band_mixer", False))
    temporal_module_enabled = bool(
        isoenergetic_temporal_transport or temporal_band_mixer)
    official_point_objective = bool(getattr(
        sim_args, "official_point_objective", False))
    official_point_controller_tke_floor = bool(getattr(
        sim_args, "official_point_controller_tke_floor", False))
    pareto_tke_projection = bool(getattr(
        sim_args, "pareto_tke_projection", False))
    legacy_tke_gradient_floor = bool(getattr(
        sim_args, "legacy_tke_gradient_floor", False))
    member_integrated_legacy_tke_gradient_floor = bool(getattr(
        sim_args, "member_integrated_legacy_tke_gradient_floor", False))
    member_official_objective = bool(getattr(
        sim_args, "member_official_objective", False))
    decoupled_member_residual = bool(getattr(
        sim_args, "decoupled_member_residual", False))
    member_feature_adapters = bool(getattr(
        sim_args, "member_feature_adapters", False))
    standard_objective_member_feature_adapters = bool(getattr(
        sim_args, "standard_objective_member_feature_adapters", False))
    fluctuation_role_tower = bool(getattr(
        sim_args, "fluctuation_role_tower", False))
    mean_preserving_member_adapters = bool(getattr(
        sim_args, "mean_preserving_member_adapters", False))
    detach_member_adapter_context = bool(getattr(
        sim_args, "detach_member_adapter_context", False))
    detach_uq_context = bool(getattr(
        sim_args, "detach_uq_context", False))
    isolate_distribution_tower = bool(getattr(
        sim_args, "isolate_distribution_tower", False))
    budgeted_disagreement_allocation = bool(getattr(
        sim_args, "budgeted_disagreement_allocation", False))
    conditional_future_energy = bool(getattr(
        sim_args, "conditional_future_energy", False))
    conditional_future_energy_members_only = bool(getattr(
        sim_args, "conditional_future_energy_members_only", False))
    conditional_future_energy_adapter_only = bool(getattr(
        sim_args, "conditional_future_energy_adapter_only", False))
    conditional_future_crps_mean_preserving = bool(getattr(
        sim_args, "conditional_future_crps_mean_preserving", False))
    conditional_future_any = bool(
        conditional_future_energy or conditional_future_energy_members_only
        or conditional_future_energy_adapter_only
        or conditional_future_crps_mean_preserving)
    deploy_parameters = (
        EXPECTED_MULTISCALE_ENERGY_PARAMETERS
        if multiscale_energy_controller else (
            EXPECTED_HORIZON_LOCAL_ENERGY_PARAMETERS
            if horizon_local_energy_controller else EXPECTED_PARAMETERS))
    if history_energy_controller:
        deploy_parameters = EXPECTED_HISTORY_ENERGY_PARAMETERS
    if hybrid_history_multiscale_energy_controller:
        deploy_parameters = EXPECTED_HYBRID_HISTORY_MULTISCALE_ENERGY_PARAMETERS
    if cross_conditioned_backbone:
        deploy_parameters = EXPECTED_CROSS_CONDITIONED_BACKBONE_PARAMETERS
    if temporal_point_decoder:
        deploy_parameters = EXPECTED_TEMPORAL_POINT_DECODER_PARAMETERS
    if temporal_point_amplitude_factorization:
        deploy_parameters = EXPECTED_TEMPORAL_AMPLITUDE_FACTORIZATION_PARAMETERS
    if isoenergetic_temporal_transport:
        deploy_parameters = EXPECTED_TEMPORAL_TRANSPORT_PARAMETERS
    if decoupled_member_residual:
        deploy_parameters = EXPECTED_DECOUPLED_MEMBER_RESIDUAL_PARAMETERS
    if member_feature_adapters:
        deploy_parameters = (
            EXPECTED_BUDGETED_DISAGREEMENT_PARAMETERS
            if budgeted_disagreement_allocation
            else EXPECTED_MEMBER_FEATURE_ADAPTER_PARAMETERS)
    if cross_conditioned_backbone and member_feature_adapters:
        deploy_parameters = (
            EXPECTED_XCB_MEMBER_MULTISCALE_ENERGY_PARAMETERS
            if multiscale_energy_controller
            else EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS)
    if fluctuation_role_tower:
        deploy_parameters = EXPECTED_XCB_MEMBER_FLUCTUATION_ROLE_PARAMETERS
    if temporal_band_mixer:
        deploy_parameters = EXPECTED_XCB_MEMBER_TEMPORAL_BAND_PARAMETERS
    anchor_indices = list(range(ANCHOR_SIZE))
    if len(dataset) < ANCHOR_SIZE:
        raise RuntimeError(
            f"Sim dataset has {len(dataset)} windows, needs {ANCHOR_SIZE}")

    control_model = build_arm_model(
        "control", device, multiscale_energy_controller,
        horizon_local_energy_controller, history_energy_controller,
        hybrid_history_multiscale_energy_controller,
        cross_conditioned_backbone, temporal_point_decoder,
        temporal_point_amplitude_factorization,
        isoenergetic_temporal_transport,
        temporal_band_mixer,
        decoupled_member_residual,
        member_feature_adapters,
        fluctuation_role_tower,
        mean_preserving_member_adapters,
        detach_member_adapter_context,
        detach_uq_context,
        budgeted_disagreement_allocation)
    rehearsal.ensemble_train.common.set_model_stats(control_model, normalizer)
    control_state_hash = state_sha256(control_model.state_dict())
    initial_controller = controller_state(control_model)
    initial_cross_conditioner = cross_conditioner_state(control_model)
    initial_control_temporal_point = temporal_point_state(control_model)
    initial_control_temporal_transport = temporal_transport_state(control_model)
    initial_control_decoupled_residual = decoupled_member_residual_state(
        control_model)
    initial_control_member_adapters = member_feature_adapter_state(
        control_model)
    initial_control_fluctuation_role = fluctuation_role_state(control_model)
    control_update0 = capture_update0(
        control_model, dataset, normalizer, device)
    initial_anchor = sim_hgrad.evaluate_anchor(
        control_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract,
        sim_args, device)
    control_loader = rehearsal.ensemble_train.tag_training_loader(
        raw_loader, sim_args)
    control_iterator = rehearsal.ensemble_train.common.repro.cycling(
        control_loader)
    control_optimizer, control_scheduler = make_optimizer(
        control_model, sim_args, "control")
    control_keys = []
    control_last = None
    for iteration in range(1, CONTROL_UPDATES + 1):
        _deadline_check(deadline, iteration)
        control_last, audit, key, _flags = sim_update(
            control_model, next(control_iterator), normalizer,
            soft_sps_normalizer, teacher_cache, scale_contract,
            sim_args, device, iteration, control_optimizer,
            control_scheduler, "control", None, False)
        if audit is not None:
            raise RuntimeError("B210 control unexpectedly produced hgrad audit")
        control_keys.append(key)
    control_anchor = sim_hgrad.evaluate_anchor(
        control_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract,
        sim_args, device)
    control_final_controller = controller_state(control_model)
    if not tensor_state_equal(initial_controller, control_final_controller):
        raise RuntimeError("B210 frozen control controller changed")
    control_final_cross_conditioner = cross_conditioner_state(control_model)
    if not tensor_state_equal(
            initial_cross_conditioner, control_final_cross_conditioner):
        raise RuntimeError("B210 frozen control cross-conditioner changed")
    control_final_temporal_point = temporal_point_state(control_model)
    if (temporal_point_decoder and tensor_state_equal(
            initial_control_temporal_point, control_final_temporal_point)):
        raise RuntimeError("B210 control temporal point decoder did not train")
    control_final_temporal_transport = temporal_transport_state(control_model)
    if (temporal_module_enabled and not tensor_state_equal(
            initial_control_temporal_transport,
            control_final_temporal_transport)):
        raise RuntimeError("B210 frozen control temporal transport changed")
    control_final_decoupled_residual = decoupled_member_residual_state(
        control_model)
    if (decoupled_member_residual and not tensor_state_equal(
            initial_control_decoupled_residual,
            control_final_decoupled_residual)):
        raise RuntimeError("B210 frozen control member residual changed")
    control_final_member_adapters = member_feature_adapter_state(control_model)
    if (member_feature_adapters and not tensor_state_equal(
            initial_control_member_adapters,
            control_final_member_adapters)):
        raise RuntimeError("B210 frozen control member adapters changed")
    control_final_fluctuation_role = fluctuation_role_state(control_model)
    if (fluctuation_role_tower and not tensor_state_equal(
            initial_control_fluctuation_role,
            control_final_fluctuation_role)):
        raise RuntimeError("B210 frozen control fluctuation role changed")
    del (control_model, control_optimizer, control_scheduler,
         control_iterator, control_loader)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidate_model = build_arm_model(
        "candidate", device, multiscale_energy_controller,
        horizon_local_energy_controller, history_energy_controller,
        hybrid_history_multiscale_energy_controller,
        cross_conditioned_backbone, temporal_point_decoder,
        temporal_point_amplitude_factorization,
        isoenergetic_temporal_transport,
        temporal_band_mixer,
        decoupled_member_residual,
        member_feature_adapters,
        fluctuation_role_tower,
        mean_preserving_member_adapters,
        detach_member_adapter_context,
        detach_uq_context,
        budgeted_disagreement_allocation)
    rehearsal.ensemble_train.common.set_model_stats(candidate_model, normalizer)
    candidate_state_hash = state_sha256(candidate_model.state_dict())
    if candidate_state_hash != control_state_hash:
        raise RuntimeError("paired B210 initial model states differ")
    candidate_initial_controller = controller_state(candidate_model)
    candidate_initial_cross_conditioner = cross_conditioner_state(
        candidate_model)
    candidate_initial_temporal_point = temporal_point_state(candidate_model)
    candidate_initial_temporal_transport = temporal_transport_state(
        candidate_model)
    candidate_initial_decoupled_residual = (
        decoupled_member_residual_state(candidate_model))
    candidate_initial_member_adapters = member_feature_adapter_state(
        candidate_model)
    candidate_initial_fluctuation_role = fluctuation_role_state(
        candidate_model)
    if not tensor_state_equal(initial_controller, candidate_initial_controller):
        raise RuntimeError("paired B210 initial controller states differ")
    if not tensor_state_equal(
            initial_cross_conditioner, candidate_initial_cross_conditioner):
        raise RuntimeError(
            "paired B210 initial cross-conditioner states differ")
    if not tensor_state_equal(
            initial_control_temporal_point,
            candidate_initial_temporal_point):
        raise RuntimeError(
            "paired B210 initial temporal point states differ")
    if not tensor_state_equal(
            initial_control_temporal_transport,
            candidate_initial_temporal_transport):
        raise RuntimeError(
            "paired B210 initial temporal transport states differ")
    if not tensor_state_equal(
            initial_control_decoupled_residual,
            candidate_initial_decoupled_residual):
        raise RuntimeError(
            "paired B210 initial member residual states differ")
    if not tensor_state_equal(
            initial_control_member_adapters,
            candidate_initial_member_adapters):
        raise RuntimeError(
            "paired B210 initial member adapter states differ")
    if not tensor_state_equal(
            initial_control_fluctuation_role,
            candidate_initial_fluctuation_role):
        raise RuntimeError(
            "paired B210 initial fluctuation role states differ")
    candidate_update0 = capture_update0(
        candidate_model, dataset, normalizer, device)
    update0 = update0_identity(control_update0, candidate_update0)
    if not update0["pass"]:
        raise RuntimeError("paired B210 update0 outputs are not bitwise identical")
    candidate_initial_anchor = sim_hgrad.evaluate_anchor(
        candidate_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract,
        sim_args, device)
    multiscale_tke_supervisor = (
        build_multiscale_tke_supervisor(candidate_model, device)
        if bool(getattr(sim_args, "multiscale_tke_deep_supervision", False))
        else None)
    multiscale_tke_parameters = (
        0 if multiscale_tke_supervisor is None else sum(
            parameter.numel()
            for parameter in multiscale_tke_supervisor.parameters()))
    candidate_loader = rehearsal.ensemble_train.tag_training_loader(
        raw_loader, sim_args)
    candidate_iterator = rehearsal.ensemble_train.common.repro.cycling(
        candidate_loader)
    optimizer, scheduler = make_optimizer(
        candidate_model, sim_args, "candidate", multiscale_tke_supervisor)
    candidate_keys = []
    gradient_ratios = []
    pareto_projection_rows = []
    legacy_tke_floor_rows = []
    hgrad_audits = []
    projection_gradient_seen = False
    front_gradient_seen = False
    cross_projection_gradient_seen = False
    cross_body_gradient_seen = False
    temporal_mean_gradient_seen = False
    temporal_fluctuation_gradient_seen = False
    temporal_amplitude_gradient_seen = False
    temporal_projection_gradient_seen = False
    temporal_body_gradient_seen = False
    transport_output_gradient_seen = False
    transport_body_gradient_seen = False
    controller_step1 = None
    history = []
    values = None
    multiscale_tke_gradient_audit = None
    for iteration in range(1, CONTROL_UPDATES + 1):
        _deadline_check(deadline, iteration)
        values, audit, key, flags = sim_update(
            candidate_model, next(candidate_iterator), normalizer,
            soft_sps_normalizer, teacher_cache, scale_contract,
            sim_args, device, iteration, optimizer, scheduler, "candidate",
            closure_augmentor, closure_equivariance,
            multiscale_tke_supervisor)
        sim_hgrad.require_paired_batch(control_keys[iteration - 1], key, iteration)
        candidate_keys.append(key)
        gradient_ratios.append(
            values["tke_aux_to_native_gradient_ratio"])
        if bool(getattr(sim_args, "pareto_tke_projection", False)):
            pareto_projection_rows.append({
                key: values[key] for key in (
                    "pareto_projection_applied",
                    "pareto_primary_tke_cosine",
                    "pareto_primary_gradient_norm",
                    "pareto_tke_gradient_norm",
                    "pareto_correction_to_primary_ratio",
                    "pareto_projection_finite")
            })
        if bool(getattr(sim_args, "legacy_tke_gradient_floor", False)):
            legacy_tke_floor_rows.append({
                key: values[key] for key in (
                    "legacy_tke_floor_finite",
                    "legacy_tke_floor_applied_fraction",
                    "legacy_tke_floor_primary_parallel_median",
                    "legacy_tke_floor_reference_parallel_median",
                    "legacy_tke_floor_missing_parallel_median",
                    "legacy_tke_floor_correction_ratio_median",
                    "legacy_tke_floor_correction_ratio_max")
            })
        projection_gradient_seen |= flags["projection_nonzero"]
        front_gradient_seen |= flags["front_nonzero"]
        cross_projection_gradient_seen |= flags["cross_projection_nonzero"]
        cross_body_gradient_seen |= flags["cross_body_nonzero"]
        temporal_mean_gradient_seen |= flags["temporal_mean_nonzero"]
        temporal_fluctuation_gradient_seen |= flags[
            "temporal_fluctuation_nonzero"]
        temporal_amplitude_gradient_seen |= flags[
            "temporal_amplitude_nonzero"]
        temporal_projection_gradient_seen |= flags[
            "temporal_projection_nonzero"]
        temporal_body_gradient_seen |= flags["temporal_body_nonzero"]
        transport_output_gradient_seen |= flags[
            "transport_output_nonzero"]
        transport_body_gradient_seen |= flags["transport_body_nonzero"]
        if iteration == 1:
            controller_step1 = controller_state(candidate_model)
        if values.get("multiscale_tke_gradient_audit") is not None:
            multiscale_tke_gradient_audit = values.get(
                "multiscale_tke_gradient_audit")
        if audit is not None:
            hgrad_audits.append(audit)
            print(json.dumps({
                "event": "b210_np_sim_hgrad_audit", **audit,
            }), flush=True)
    candidate_anchor_200 = sim_hgrad.evaluate_anchor(
        candidate_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract,
        sim_args, device)
    mean_preserving_audit = (
        mean_preserving_member_adapter_audit(
            candidate_model, dataset, normalizer, device)
        if mean_preserving_member_adapters else None)
    candidate_controller_200 = controller_state(candidate_model)
    candidate_cross_conditioner_200 = cross_conditioner_state(candidate_model)
    candidate_temporal_point_200 = temporal_point_state(candidate_model)
    candidate_temporal_transport_200 = temporal_transport_state(
        candidate_model)
    candidate_decoupled_residual_200 = decoupled_member_residual_state(
        candidate_model)
    candidate_member_adapters_200 = member_feature_adapter_state(
        candidate_model)
    candidate_fluctuation_role_200 = fluctuation_role_state(candidate_model)
    if controller_step1 is None:
        raise RuntimeError("B210 candidate has no update-1 controller state")
    controller_audit = {
        "projection_updated": not tensor_state_equal(
            controller_projection_state(candidate_initial_controller),
            controller_projection_state(controller_step1)),
        "front_updated": not tensor_state_equal(
            controller_front_state(candidate_initial_controller),
            controller_front_state(candidate_controller_200)),
        "projection_gradient_seen": projection_gradient_seen,
        "front_gradient_seen": front_gradient_seen,
        "state_updated": not tensor_state_equal(
            candidate_initial_controller, candidate_controller_200),
        "initial_state_sha256": state_sha256(candidate_initial_controller),
        "update200_state_sha256": state_sha256(candidate_controller_200),
    }
    if hybrid_history_multiscale_energy_controller:
        initial_history = OrderedDict(
            (name, value) for name, value in candidate_initial_controller.items()
            if name.startswith("history."))
        final_history = OrderedDict(
            (name, value) for name, value in candidate_controller_200.items()
            if name.startswith("history."))
        initial_decoder = OrderedDict(
            (name, value) for name, value in candidate_initial_controller.items()
            if not name.startswith("history."))
        final_decoder = OrderedDict(
            (name, value) for name, value in candidate_controller_200.items()
            if not name.startswith("history."))
        controller_audit["hybrid_history_updated"] = not tensor_state_equal(
            initial_history, final_history)
        controller_audit["hybrid_decoder_updated"] = not tensor_state_equal(
            initial_decoder, final_decoder)
    if cross_conditioned_backbone:
        controller_audit.update({
            "cross_projection_updated": not tensor_state_equal(
                cross_conditioner_projection_state(
                    candidate_initial_cross_conditioner),
                cross_conditioner_projection_state(
                    candidate_cross_conditioner_200)),
            "cross_body_updated": not tensor_state_equal(
                cross_conditioner_body_state(
                    candidate_initial_cross_conditioner),
                cross_conditioner_body_state(
                    candidate_cross_conditioner_200)),
            "cross_projection_gradient_seen": (
                cross_projection_gradient_seen),
            "cross_body_gradient_seen": cross_body_gradient_seen,
            "cross_initial_state_sha256": state_sha256(
                candidate_initial_cross_conditioner),
            "cross_update200_state_sha256": state_sha256(
                candidate_cross_conditioner_200),
        })
    if temporal_point_decoder:
        controller_audit.update({
            "temporal_output_updated": not tensor_state_equal(
                temporal_point_output_state(candidate_initial_temporal_point),
                temporal_point_output_state(candidate_temporal_point_200)),
            "temporal_body_updated": not tensor_state_equal(
                temporal_point_body_state(candidate_initial_temporal_point),
                temporal_point_body_state(candidate_temporal_point_200)),
            "temporal_mean_gradient_seen": temporal_mean_gradient_seen,
            "temporal_fluctuation_gradient_seen": (
                temporal_fluctuation_gradient_seen),
            "temporal_amplitude_gradient_seen": (
                temporal_amplitude_gradient_seen),
            "temporal_projection_gradient_seen": (
                temporal_projection_gradient_seen),
            "temporal_body_gradient_seen": temporal_body_gradient_seen,
            "temporal_initial_state_sha256": state_sha256(
                candidate_initial_temporal_point),
            "temporal_update200_state_sha256": state_sha256(
                candidate_temporal_point_200),
        })
    if temporal_module_enabled:
        invariant_audit = temporal_transport_invariant_audit(
            candidate_model, dataset, normalizer, device)
        controller_audit.update({
            "transport_output_updated": not tensor_state_equal(
                temporal_transport_output_state(
                    candidate_initial_temporal_transport),
                temporal_transport_output_state(
                    candidate_temporal_transport_200)),
            "transport_body_updated": not tensor_state_equal(
                temporal_transport_body_state(
                    candidate_initial_temporal_transport),
                temporal_transport_body_state(
                    candidate_temporal_transport_200)),
            "transport_output_gradient_seen": (
                transport_output_gradient_seen),
            "transport_body_gradient_seen": transport_body_gradient_seen,
            "transport_invariant_audit": invariant_audit,
            "transport_initial_state_sha256": state_sha256(
                candidate_initial_temporal_transport),
            "transport_update200_state_sha256": state_sha256(
                candidate_temporal_transport_200),
        })
    if decoupled_member_residual:
        controller_audit.update({
            "decoupled_residual_output_updated": not tensor_state_equal(
                decoupled_residual_output_state(
                    candidate_initial_decoupled_residual),
                decoupled_residual_output_state(
                    candidate_decoupled_residual_200)),
            "decoupled_residual_body_updated": not tensor_state_equal(
                decoupled_residual_body_state(
                    candidate_initial_decoupled_residual),
                decoupled_residual_body_state(
                    candidate_decoupled_residual_200)),
            "decoupled_residual_initial_state_sha256": state_sha256(
                candidate_initial_decoupled_residual),
            "decoupled_residual_update200_state_sha256": state_sha256(
                candidate_decoupled_residual_200),
        })
    if member_feature_adapters:
        controller_audit.update({
            "member_adapter_output_updated": not tensor_state_equal(
                member_adapter_output_state(
                    candidate_initial_member_adapters),
                member_adapter_output_state(
                    candidate_member_adapters_200)),
            "member_adapter_body_updated": not tensor_state_equal(
                member_adapter_body_state(
                    candidate_initial_member_adapters),
                member_adapter_body_state(candidate_member_adapters_200)),
            "member_adapter_initial_state_sha256": state_sha256(
                candidate_initial_member_adapters),
            "member_adapter_update200_state_sha256": state_sha256(
                candidate_member_adapters_200),
        })
    if fluctuation_role_tower:
        controller_audit.update({
            "fluctuation_role_output_updated": not tensor_state_equal(
                member_adapter_output_state(
                    candidate_initial_fluctuation_role),
                member_adapter_output_state(
                    candidate_fluctuation_role_200)),
            "fluctuation_role_body_updated": not tensor_state_equal(
                member_adapter_body_state(
                    candidate_initial_fluctuation_role),
                member_adapter_body_state(
                    candidate_fluctuation_role_200)),
            "fluctuation_role_initial_state_sha256": state_sha256(
                candidate_initial_fluctuation_role),
            "fluctuation_role_update200_state_sha256": state_sha256(
                candidate_fluctuation_role_200),
        })
    if not controller_audit["state_updated"]:
        controller_audit["projection_updated"] = False
        controller_audit["front_updated"] = False
    prefix = prefix_gate(
        candidate_initial_anchor, control_anchor, candidate_anchor_200,
        gradient_ratios, update0, controller_audit, hgrad_audits)
    if bool(getattr(sim_args, "pareto_tke_projection", False)):
        pareto_summary = pareto_projection_summary(pareto_projection_rows)
        prefix["pareto_output_gradient_audit"] = pareto_summary
        prefix["health_checks"]["pareto_projection_mechanically_valid"] = (
            bool(pareto_summary["pass"]))
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if fluctuation_role_tower:
        for key in (
                "fluctuation_role_output_updated",
                "fluctuation_role_body_updated"):
            prefix["health_checks"][key] = bool(controller_audit[key])
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if conditional_future_any:
        future_checks = {
            "conditional_future_all_selected": math.isclose(
                float(values["conditional_future_selected_fraction"]),
                CONDITIONAL_FUTURE_PROBABILITY, abs_tol=1.0e-7),
            "conditional_future_nonzero": (
                float(values["conditional_future_target_delta_uv_rms"])
                > 0.0),
            "functional_energy_finite": all(math.isfinite(float(values[key]))
                for key in (
                    "functional_energy_score",
                    "functional_energy_attraction",
                    "functional_energy_repulsion")),
            "functional_members_noncollapsed": (
                float(values["functional_energy_repulsion"]) > 1.0e-6),
        }
        if conditional_future_crps_mean_preserving:
            future_checks.update({
                "pointwise_crps_finite": all(math.isfinite(float(values[key]))
                    for key in (
                        "pointwise_crps_score",
                        "pointwise_crps_attraction",
                        "pointwise_crps_repulsion")),
                "pointwise_crps_members_noncollapsed": (
                    float(values["pointwise_crps_repulsion"]) > 1.0e-6),
                "mean_preserving_member_adapters": bool(
                    mean_preserving_member_adapters),
            })
            if detach_member_adapter_context:
                future_checks["detached_member_adapter_context"] = bool(
                    values["detach_member_adapter_context"] == 1.0)
            if isolate_distribution_tower:
                future_checks.update({
                    "detached_uq_context": bool(
                        values["detach_uq_context"] == 1.0),
                    "base_member_supervision": bool(
                        values["member_supervision_uses_base_members"] == 1.0),
                })
            if budgeted_disagreement_allocation:
                future_checks.update({
                    "budgeted_disagreement_allocation": bool(
                        values["budgeted_disagreement_allocation"] == 1.0),
                    "disagreement_allocator_strength_updated": bool(
                        math.isfinite(float(values[
                            "disagreement_allocation_strength_abs_max"]))
                        and float(values[
                            "disagreement_allocation_strength_abs_max"])
                        > 0.0),
                })
        prefix["health_checks"].update(future_checks)
        prefix["conditional_future_energy_audit"] = {
            "values_at_update200": {
                key: values[key] for key in (
                    "functional_energy_score",
                    "functional_energy_attraction",
                    "functional_energy_repulsion",
                    "member_uv_normalized_spread",
                    "conditional_future_selected_fraction",
                    "conditional_future_target_delta_uv_rms",
                    "conditional_future_quantile_uses_base",
                    "pointwise_crps_score",
                    "pointwise_crps_attraction",
                    "pointwise_crps_repulsion",
                    "disagreement_allocation_strength_abs_max",
                    "disagreement_allocation_sharpness_abs_max")},
            "checks": future_checks,
        }
        if (conditional_future_energy_adapter_only
                or conditional_future_crps_mean_preserving):
            adapter_ratio = float(values[
                "functional_energy_to_native_adapter_gradient_ratio"])
            adapter_checks = {
                "future_energy_adapter_only": bool(
                    values["functional_energy_adapter_only"] == 1.0),
                "future_energy_adapter_gradient_finite_nonzero": bool(
                    math.isfinite(float(values[
                        "functional_energy_adapter_gradient_norm"]))
                    and float(values[
                        "functional_energy_adapter_gradient_norm"]) > 0.0),
                "future_energy_shared_gradient_zero": bool(
                    values[
                        "functional_energy_shared_gradient_by_construction"]
                    == 0.0),
            }
            if isolate_distribution_tower:
                adapter_checks["native_adapter_gradient_zero"] = bool(
                    float(values["native_adapter_gradient_norm"]) == 0.0)
            else:
                adapter_checks["future_energy_adapter_gradient_ratio"] = bool(
                    math.isfinite(adapter_ratio)
                    and 0.05 <= adapter_ratio <= 5.0)
            prefix["health_checks"].update(adapter_checks)
            prefix["conditional_future_energy_audit"].update({
                "adapter_only_values_at_update200": {
                    key: values[key] for key in (
                        "functional_energy_adapter_gradient_norm",
                        "native_adapter_gradient_norm",
                        "functional_energy_to_native_adapter_gradient_ratio",
                        "functional_energy_shared_gradient_by_construction")},
                "adapter_only_checks": adapter_checks,
            })
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if mean_preserving_member_adapters:
        prefix["mean_preserving_member_adapter_audit"] = (
            mean_preserving_audit)
        prefix["health_checks"][
            "mean_preserving_member_adapter_isolation"] = bool(
                mean_preserving_audit["pass"])
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if bool(getattr(sim_args, "legacy_tke_gradient_floor", False)):
        floor_summary = legacy_tke_floor_summary(legacy_tke_floor_rows)
        weighted_delta = (
            candidate_anchor_200["weighted_accuracy_score"]
            - control_anchor["weighted_accuracy_score"])
        directional = {
            "tke_delta_at_least_minus_0p35": (
                prefix["component_delta"]["tke_rel_l2_score"] >= -0.35),
            "official_weighted_delta_at_least_minus_0p05": (
                weighted_delta >= -0.05),
        }
        prefix["legacy_tke_gradient_floor_audit"] = floor_summary
        prefix["legacy_tke_floor_directional_gate"] = {
            "checks": directional,
            "official_weighted_delta": weighted_delta,
            "is_hard_gate": not (
                member_integrated_legacy_tke_gradient_floor
                or member_official_objective),
            "pass": all(directional.values()),
        }
        prefix["health_checks"]["legacy_tke_floor_mechanically_valid"] = (
            bool(floor_summary["pass"]))
        if not (member_integrated_legacy_tke_gradient_floor
                or member_official_objective):
            prefix["health_checks"]["legacy_tke_floor_directional_gate"] = (
                all(directional.values()))
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if hybrid_history_multiscale_energy_controller:
        prefix["health_checks"]["hybrid_history_updated"] = bool(
            controller_audit["hybrid_history_updated"])
        prefix["health_checks"]["hybrid_decoder_updated"] = bool(
            controller_audit["hybrid_decoder_updated"])
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if cross_conditioned_backbone:
        for key in (
                "cross_projection_updated", "cross_body_updated",
                "cross_projection_gradient_seen", "cross_body_gradient_seen"):
            prefix["health_checks"][key] = bool(controller_audit[key])
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if temporal_point_decoder:
        mechanics = [
                "temporal_output_updated", "temporal_body_updated",
                "temporal_mean_gradient_seen",
                "temporal_fluctuation_gradient_seen",
                "temporal_projection_gradient_seen",
                "temporal_body_gradient_seen"]
        if temporal_point_amplitude_factorization:
            mechanics.append("temporal_amplitude_gradient_seen")
        for key in mechanics:
            prefix["health_checks"][key] = bool(controller_audit[key])
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if temporal_module_enabled:
        for key in (
                "transport_output_updated", "transport_body_updated",
                "transport_output_gradient_seen",
                "transport_body_gradient_seen"):
            prefix["health_checks"][key] = bool(controller_audit[key])
        prefix["health_checks"]["transport_invariants"] = bool(
            controller_audit["transport_invariant_audit"]["pass"])
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if decoupled_member_residual:
        for key in (
                "decoupled_residual_output_updated",
                "decoupled_residual_body_updated"):
            prefix["health_checks"][key] = bool(controller_audit[key])
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if member_feature_adapters:
        for key in (
                "member_adapter_output_updated",
                "member_adapter_body_updated"):
            prefix["health_checks"][key] = bool(controller_audit[key])
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    if multiscale_tke_supervisor is not None:
        ratio = (
            float(multiscale_tke_gradient_audit[
                "multiscale_tke_to_native_feature_gradient_ratio"])
            if multiscale_tke_gradient_audit is not None else float("nan"))
        multiscale_health = bool(
            multiscale_tke_gradient_audit is not None
            and multiscale_tke_gradient_audit["finite_nonzero"]
            and MULTISCALE_TKE_GRADIENT_RATIO_RANGE[0] <= ratio
            <= MULTISCALE_TKE_GRADIENT_RATIO_RANGE[1])
        prefix["health_checks"][
            "multiscale_tke_feature_gradient"] = multiscale_health
        prefix["multiscale_tke_gradient_audit"] = (
            multiscale_tke_gradient_audit)
        prefix["multiscale_tke_gradient_ratio_range"] = list(
            MULTISCALE_TKE_GRADIENT_RATIO_RANGE)
        prefix["health_pass"] = all(prefix["health_checks"].values())
        prefix["status"] = (
            HEALTH_PASS_STATUS if prefix["health_pass"]
            else HEALTH_STOP_STATUS)
    print(json.dumps({
        "event": "b210_np_sim_u200", **prefix,
    }), flush=True)

    completed_updates = CONTROL_UPDATES
    checkpoint = checkpoint_sha = None
    multiscale_tke_checkpoint = multiscale_tke_checkpoint_sha = None
    terminal_anchor = None
    if prefix["health_pass"]:
        history.append({"iteration": CONTROL_UPDATES, **values})
        for iteration in range(CONTROL_UPDATES + 1, TERMINAL_UPDATE + 1):
            _deadline_check(deadline, iteration)
            values, audit, _key, _flags = sim_update(
                candidate_model, next(candidate_iterator), normalizer,
                soft_sps_normalizer, teacher_cache, scale_contract,
                sim_args, device, iteration, optimizer, scheduler, "candidate",
                closure_augmentor, closure_equivariance,
                multiscale_tke_supervisor)
            if audit is not None:
                raise RuntimeError("B210 hgrad audit occurred after u200")
            if iteration % LOG_INTERVAL == 0 or iteration == TERMINAL_UPDATE:
                row = {"iteration": iteration, **values}
                history.append(row)
                print(json.dumps({
                    "event": "b210_np_sim_progress", **row,
                }), flush=True)
        completed_updates = TERMINAL_UPDATE
        terminal_anchor = sim_hgrad.evaluate_anchor(
            candidate_model, dataset, anchor_indices, normalizer,
            soft_sps_normalizer, teacher_cache, scale_contract,
            sim_args, device)
        print(json.dumps({
            "event": "b210_np_sim_terminal_anchor", **terminal_anchor,
        }), flush=True)
        checkpoint, checkpoint_sha = save_terminal_checkpoint(
            args.output_dir, candidate_model, sim_args, history, normalizer)
        if multiscale_tke_supervisor is not None:
            supervisor_path = (
                args.output_dir / "multiscale_tke_supervisor_10000.pth")
            torch.save({
                "method": MULTISCALE_TKE_METHOD,
                "model_checkpoint_sha256": checkpoint_sha,
                "state_dict": multiscale_tke_supervisor.state_dict(),
                "channels": [
                    int(head[0].in_channels)
                    for head in multiscale_tke_supervisor.heads],
                "physical_scale": MULTISCALE_TKE_PHYSICAL_SCALE,
                "level_weights": list(MULTISCALE_TKE_LEVEL_WEIGHTS),
                "weight": MULTISCALE_TKE_WEIGHT,
                "deployable": False,
            }, supervisor_path)
            multiscale_tke_checkpoint = str(supervisor_path)
            multiscale_tke_checkpoint_sha = sim_hgrad.sha256(supervisor_path)

    status = PASS_STATUS if completed_updates == TERMINAL_UPDATE else prefix["status"]
    method = METHOD
    if closure_augmentor is not None:
        method = REAL_CLOSURE_SEED1_METHOD
    if bool(getattr(sim_args, "phase_aligned_re_mix", False)):
        method = PHASE_ALIGNED_RE_MIX_METHOD
    if bool(getattr(sim_args, "phase_aligned_parameter_grid_mix", False)):
        method = PHASE_ALIGNED_GRID_MIX_METHOD
    if closure_equivariance:
        method = REAL_CLOSURE_EQUIVARIANCE_METHOD
    if multiscale_tke_supervisor is not None:
        method = MULTISCALE_TKE_METHOD
    if multiscale_energy_controller:
        method = MULTISCALE_ENERGY_METHOD
    if horizon_local_energy_controller:
        method = HORIZON_LOCAL_ENERGY_METHOD
    if history_energy_controller:
        method = HISTORY_ENERGY_METHOD
    if bool(getattr(sim_args, "direct_history_tke_regression", False)):
        method = DIRECT_HISTORY_TKE_METHOD
    if hybrid_history_multiscale_energy_controller:
        method = HYBRID_HISTORY_MULTISCALE_ENERGY_METHOD
    if cross_conditioned_backbone:
        method = CROSS_CONDITIONED_BACKBONE_METHOD
    if temporal_point_decoder:
        method = TEMPORAL_POINT_DECODER_METHOD
    if temporal_point_amplitude_factorization:
        method = TEMPORAL_AMPLITUDE_FACTORIZATION_METHOD
    if isoenergetic_temporal_transport:
        method = ISOENERGETIC_TEMPORAL_TRANSPORT_METHOD
    if official_point_objective:
        method = OFFICIAL_POINT_TEMPORAL_TRANSPORT_METHOD
    if official_point_controller_tke_floor:
        method = OFFICIAL_POINT_TKE_FLOOR_METHOD
    if pareto_tke_projection:
        method = PARETO_TKE_PROJECTION_METHOD
    if legacy_tke_gradient_floor:
        method = LEGACY_TKE_GRADIENT_FLOOR_METHOD
    if member_official_objective:
        method = MEMBER_OFFICIAL_MEAN_TKE_FLOOR_METHOD
    if member_integrated_legacy_tke_gradient_floor:
        method = MEMBER_INTEGRATED_TKE_GRADIENT_FLOOR_METHOD
    if decoupled_member_residual:
        method = DECOUPLED_MEMBER_RESIDUAL_METHOD
    if member_feature_adapters:
        method = MEMBER_FEATURE_ADAPTER_METHOD
    if cross_conditioned_backbone and member_feature_adapters:
        method = (
            STANDARD_XCB_MEMBER_FLUCTUATION_ROLE_METHOD
            if fluctuation_role_tower
            else STANDARD_XCB_MEMBER_MULTISCALE_ENERGY_METHOD
            if (standard_objective_member_feature_adapters
                and multiscale_energy_controller)
            else STANDARD_CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD
            if standard_objective_member_feature_adapters
            else CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD)
    # Architecture-combination labels must follow their constituent labels.
    # Otherwise the generic XCB/member arm above masks the band-mixer identity
    # even though the serialized checkpoint training_config is correct.
    if temporal_band_mixer:
        method = TEMPORAL_BAND_MIXER_METHOD
    if conditional_future_energy:
        method = CONDITIONAL_FUTURE_ENERGY_METHOD
    if conditional_future_energy_members_only:
        method = CONDITIONAL_FUTURE_MEMBER_ONLY_METHOD
    if conditional_future_energy_adapter_only:
        method = CONDITIONAL_FUTURE_ADAPTER_ONLY_METHOD
    if conditional_future_crps_mean_preserving:
        method = CONDITIONAL_FUTURE_CRPS_MEAN_PRESERVING_METHOD
    if detach_member_adapter_context:
        method = CONDITIONAL_FUTURE_CRPS_DETACHED_CONTEXT_METHOD
    if isolate_distribution_tower:
        method = ISOLATED_DISTRIBUTION_TOWER_METHOD
    if budgeted_disagreement_allocation:
        method = BUDGETED_DISAGREEMENT_ALLOCATION_METHOD
    if budgeted_disagreement_allocation:
        future_contract = (
            "base UQ receives collapsed members; actual detached member "
            "spread can only reallocate, never change, each sample/horizon/"
            "channel spatial mean width budget")
    elif isolate_distribution_tower:
        future_contract = (
            "point/member/hgrad/distillation train the legacy point path; "
            "legacy member MSE uses adapter-free members; pointwise future "
            "CRPS trains only centered stochastic adapters; quantile and "
            "soft-SPS losses read detached point/member/backbone context and "
            "train only the UQ tower")
    elif detach_member_adapter_context:
        future_contract = (
            "point prediction is computed from legacy members; adapter bodies "
            "read detached decoder context while their residuals are added to "
            "the live legacy feature path; output residuals are member-"
            "centered; pointwise CRPS is adapter-only")
    elif conditional_future_crps_mean_preserving:
        future_contract = (
            "point prediction is computed from the legacy members and is "
            "structurally independent of adapters; adapter output residuals "
            "are centered across members; pointwise CRPS future gradients are "
            "adapter-only; direct UQ uses base truth")
    elif conditional_future_energy_adapter_only:
        future_contract = (
            "point/hgrad/distillation/quantile/soft-SPS and legacy member MSE "
            "use base target; future energy gradients are added only to "
            "member-specific adapters; shared gradient is zero by autograd "
            "parameter selection")
    elif conditional_future_energy_members_only:
        future_contract = (
            "point/hgrad/distillation/quantile/soft-SPS use base target; only "
            "member energy uses target-only branch; history is bitwise "
            "preserved; branch RNG is forked")
    elif conditional_future_energy:
        future_contract = (
            "point/hgrad/distillation use base target; member energy, quantile "
            "pinball and soft-SPS use target-only branch; history is bitwise "
            "preserved; branch RNG is forked")
    else:
        future_contract = None
    return {
        "method": method,
        "status": status,
        "completed_updates": completed_updates,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "multiscale_tke_checkpoint": multiscale_tke_checkpoint,
        "multiscale_tke_checkpoint_sha256": (
            multiscale_tke_checkpoint_sha),
        "model_config": candidate_model.architecture_config(),
        "parameters": deploy_parameters,
        "train_only_multiscale_tke_parameters": multiscale_tke_parameters,
        "teacher_cache": teacher_metadata,
        "hgrad_scale_contract": scale_contract,
        "paired_control": {
            "updates": CONTROL_UPDATES,
            "initial_model_state_sha256": control_state_hash,
            "initial_anchor": initial_anchor,
            "terminal_anchor": control_anchor,
            "terminal_train_values": control_last,
            "batch_sequence_sha256": rehearsal.canonical_sha256(control_keys),
            "controller_frozen_exact": True,
            "writes_artifacts": False,
        },
        "paired_candidate": {
            "initial_model_state_sha256": candidate_state_hash,
            "initial_anchor": candidate_initial_anchor,
            "batch_sequence_sha256_update200": rehearsal.canonical_sha256(
                candidate_keys),
            "controller_audit": controller_audit,
            "terminal_anchor": terminal_anchor,
        },
        "u200_gate": prefix,
        "history": history,
        "training_contract": {
            "hypothesis": (
                "B210 capacity plus train-only dp02, Sim-integrated hgrad, "
                "member-synchronous controller-only truth TKE, native soft-SPS, "
                "and current-P1 distillation produces a stronger Sim prior "
                "for the separately gated Real ft12 stage"),
            "exact_baseline": (
                "same-seed B210 controller-enabled dropout0 native Sim arm"),
            "seed": SEED,
            "control_updates": CONTROL_UPDATES,
            "terminal_update": TERMINAL_UPDATE,
            "closure_acceleration_steps": CLOSURE_ACCELERATION_STEPS,
            "phase_delta_hidden_channels": PHASE_DELTA_HIDDEN_CHANNELS,
            "phase_delta_steps": PHASE_DELTA_STEPS,
            "wall_budget_seconds": WALL_BUDGET_SECONDS,
            "hard_stop": (
                "runtime/OOM/nonfinite/wall budget or u200 health failure"),
            "directional_u200_is_diagnostic_only": True,
            "no_real_data_or_held_labels": True,
            "no_c208_checkpoint_loaded": True,
            "backbone_residual_dropout": BACKBONE_DROPOUT,
            "hgrad_weight": HGRAD_WEIGHT,
            "hgrad_ramp_updates": HGRAD_RAMP_UPDATES,
            "tke_auxiliary_scope": "energy_controller_parameters_only",
            "tke_aux_weight": TKE_AUX_WEIGHT,
            "tke_aux_updates": (
                TERMINAL_UPDATE if official_point_controller_tke_floor
                else TKE_AUX_UPDATES),
            "tke_aux_persistent_floor": (
                TKE_AUX_PERSISTENT_FLOOR_WEIGHT
                if official_point_controller_tke_floor else 0.0),
            "tke_aux_schedule": (
                "cosine_0.020_to_0.004_by_u1500_then_constant_to_u10000"
                if official_point_controller_tke_floor
                else "cosine_0.020_to_zero_by_u1500"),
            "tke_aux_calibration": {
                "source": "b210_np_functional_sim_v1_u200",
                "source_weight": 0.010,
                "source_median_gradient_ratio": (
                    TKE_AUX_CALIBRATION_SOURCE_MEDIAN),
                "target_gradient_ratio": TKE_AUX_CALIBRATION_TARGET,
                "rule": "round(0.010 * 0.10 / source_median, 3)",
                "recalibrations_allowed_after_v2": 0,
            },
            "tke_gradient_ratio_median_range": list(
                TKE_GRADIENT_RATIO_RANGE),
            "tke_gradient_ratio_p95": "diagnostic_only",
            "multiscale_tke_deep_supervision": (
                multiscale_tke_supervisor is not None),
            "multiscale_tke_weight": (
                MULTISCALE_TKE_WEIGHT
                if multiscale_tke_supervisor is not None else 0.0),
            "multiscale_tke_physical_scale": (
                MULTISCALE_TKE_PHYSICAL_SCALE),
            "multiscale_tke_level_weights": list(
                MULTISCALE_TKE_LEVEL_WEIGHTS),
            "multiscale_tke_gradient_audit": (
                multiscale_tke_gradient_audit),
            "multiscale_tke_heads_saved": False,
            "multiscale_energy_controller": multiscale_energy_controller,
            "horizon_local_energy_controller": (
                horizon_local_energy_controller),
            "history_energy_controller": history_energy_controller,
            "direct_history_tke_regression": bool(getattr(
                sim_args, "direct_history_tke_regression", False)),
            "direct_history_tke_weight": (
                DIRECT_HISTORY_TKE_WEIGHT if bool(getattr(
                    sim_args, "direct_history_tke_regression", False)) else 0.0),
            "hybrid_history_multiscale_energy_controller": (
                hybrid_history_multiscale_energy_controller),
            "cross_conditioned_backbone": cross_conditioned_backbone,
            "cross_conditioned_steps": CROSS_CONDITIONED_STEPS,
            "cross_conditioned_channels": CROSS_CONDITIONED_CHANNELS,
            "temporal_point_decoder": temporal_point_decoder,
            "temporal_point_channels": TEMPORAL_POINT_CHANNELS,
            "temporal_point_blocks": TEMPORAL_POINT_BLOCKS,
            "temporal_point_kernel": TEMPORAL_POINT_KERNEL,
            "temporal_point_amplitude_factorization": (
                temporal_point_amplitude_factorization),
            "temporal_point_amplitude_limit": TEMPORAL_POINT_AMPLITUDE_LIMIT,
            "isoenergetic_temporal_transport": (
                isoenergetic_temporal_transport),
            "temporal_band_mixer": temporal_band_mixer,
            "temporal_band_channels": TEMPORAL_BAND_CHANNELS,
            "temporal_band_count": TEMPORAL_BAND_COUNT,
            "temporal_band_log_gain_limit": TEMPORAL_BAND_LOG_GAIN_LIMIT,
            "temporal_band_phase_limit": TEMPORAL_BAND_PHASE_LIMIT,
            "temporal_transport_channels": TEMPORAL_TRANSPORT_CHANNELS,
            "temporal_transport_blocks": TEMPORAL_TRANSPORT_BLOCKS,
            "temporal_transport_kernel": TEMPORAL_TRANSPORT_KERNEL,
            "temporal_transport_max_displacement": (
                TEMPORAL_TRANSPORT_MAX_DISPLACEMENT),
            "official_point_objective": official_point_objective,
            "official_point_controller_tke_floor": (
                official_point_controller_tke_floor),
            "pareto_tke_projection": pareto_tke_projection,
            "pareto_projection_space": (
                "normalized_point_output" if pareto_tke_projection
                else None),
            "pareto_protected_metric": (
                "official_tke_score_loss" if pareto_tke_projection
                else None),
            "pareto_rule": (
                "project whole non-auxiliary output gradient only when "
                "dot(primary,tke)<0" if pareto_tke_projection else None),
            "legacy_tke_gradient_floor": legacy_tke_gradient_floor,
            "member_integrated_legacy_tke_gradient_floor": (
                member_integrated_legacy_tke_gradient_floor),
            "member_official_objective": member_official_objective,
            "decoupled_member_residual": decoupled_member_residual,
            "decoupled_member_residual_channels": (
                DECOUPLED_MEMBER_RESIDUAL_CHANNELS
                if decoupled_member_residual else 0),
            "decoupled_member_residual_blocks": (
                DECOUPLED_MEMBER_RESIDUAL_BLOCKS),
            "decoupled_member_residual_parameters": (
                EXPECTED_DECOUPLED_MEMBER_RESIDUAL_MODULE_PARAMETERS
                if decoupled_member_residual else 0),
            "decoupled_member_residual_isolation": (
                "detached_finest_feature_and_detached_legacy_members; "
                "cross_member_zero_mean_residual; point_prediction_uses_"
                "pre_residual_member_mean" if decoupled_member_residual
                else None),
            "member_feature_adapters": member_feature_adapters,
            "standard_objective_member_feature_adapters": (
                standard_objective_member_feature_adapters),
            "fluctuation_role_tower": fluctuation_role_tower,
            "fluctuation_role_channels": (
                FLUCTUATION_ROLE_CHANNELS
                if fluctuation_role_tower else 0),
            "fluctuation_role_parameters": (
                EXPECTED_FLUCTUATION_ROLE_MODULE_PARAMETERS
                if fluctuation_role_tower else 0),
            "fluctuation_role_contract": (
                "shared multiscale nonlinear residual coordinates are read "
                "only by zero-time-mean point channels; temporal-mean "
                "readouts retain the unmodified member features"
                if fluctuation_role_tower else None),
            "mean_preserving_member_adapters": (
                mean_preserving_member_adapters),
            "detach_member_adapter_context": (
                detach_member_adapter_context),
            "detach_uq_context": detach_uq_context,
            "isolate_distribution_tower": isolate_distribution_tower,
            "budgeted_disagreement_allocation": (
                budgeted_disagreement_allocation),
            "budgeted_disagreement_parameters": (
                EXPECTED_BUDGETED_DISAGREEMENT_MODULE_PARAMETERS
                if budgeted_disagreement_allocation else 0),
            "member_feature_adapter_channels": (
                MEMBER_FEATURE_ADAPTER_CHANNELS
                if member_feature_adapters else 0),
            "member_feature_adapter_parameters": (
                EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS
                if member_feature_adapters else 0),
            "member_feature_adapter_scope": (
                "four_members_x_all_three_decoder_scales; legacy_member_"
                "bootstrap_and_backbone_gradient_preserved"
                if member_feature_adapters else None),
            "cross_conditioned_member_adapters": bool(
                cross_conditioned_backbone and member_feature_adapters),
            "conditional_future_energy": conditional_future_energy,
            "conditional_future_energy_members_only": (
                conditional_future_energy_members_only),
            "conditional_future_energy_adapter_only": (
                conditional_future_energy_adapter_only),
            "conditional_future_crps_mean_preserving": (
                conditional_future_crps_mean_preserving),
            "conditional_future_probability": (
                CONDITIONAL_FUTURE_PROBABILITY
                if conditional_future_any else 0.0),
            "conditional_future_amplitude": (
                CONDITIONAL_FUTURE_AMPLITUDE
                if conditional_future_any else 0.0),
            "conditional_future_energy_weight": (
                CONDITIONAL_FUTURE_ENERGY_WEIGHT
                if conditional_future_any
                and not conditional_future_crps_mean_preserving else 0.0),
            "conditional_future_crps_weight": (
                CONDITIONAL_FUTURE_CRPS_WEIGHT
                if conditional_future_crps_mean_preserving else 0.0),
            "conditional_future_member_mse_weight": (
                CONDITIONAL_FUTURE_MEMBER_MSE_WEIGHT
                if conditional_future_any else 0.25),
            "conditional_future_contract": future_contract,
            "legacy_tke_gradient_floor_scope": (
                ("per_sample_normalized_member_tensor" if
                 member_integrated_legacy_tke_gradient_floor else
                 "per_sample_normalized_point_output") if
                legacy_tke_gradient_floor else None),
            "legacy_tke_gradient_floor_reference": (
                ("same_state_same_batch_hypothetical_(1,4,1)_mean_and_"
                 "bootstrap_member_nonaux_gradient" if
                 member_integrated_legacy_tke_gradient_floor else
                 "same_state_same_batch_hypothetical_(1,4,1)_nonaux_gradient")
                if legacy_tke_gradient_floor else None),
            "legacy_tke_gradient_floor_rule": (
                "retain official orthogonal gradient and add only missing "
                "nonnegative TKE-parallel component" if
                legacy_tke_gradient_floor else None),
            "accuracy_weights_rel_tke_mvpe": list(
                sim_args.accuracy_weights),
            "soft_sps_weight": SOFT_SPS_WEIGHT,
            "soft_sps_warmup_updates": SOFT_SPS_WARMUP_UPDATES,
            "p1_distill_weight": P1_DISTILL_WEIGHT,
            "real_closure": (
                None if closure_augmentor is None else {
                    "source_manifest_sha256": CLEAN_REAL_MANIFEST_SHA256,
                    "probability": REAL_CLOSURE_PROBABILITY,
                    "amplitude": REAL_CLOSURE_AMPLITUDE,
                    "closure_decay": REAL_CLOSURE_DECAY,
                    "perturbation_decay": REAL_CLOSURE_PERTURBATION_DECAY,
                    "advection_cells": REAL_CLOSURE_ADVECTION_CELLS,
                    "candidate_count": REAL_CLOSURE_CANDIDATE_COUNT,
                    "teacher_transformation": (
                        "teacher_physical_uv += augmented_target_uv "
                        "- original_target_uv"),
                    "source_contract": closure_augmentor.source_contract,
                    "paired_clean_equivariance": closure_equivariance,
                    "equivariance_weight": (
                        REAL_CLOSURE_EQUIVARIANCE_WEIGHT
                        if closure_equivariance else 0.0),
                    "dropout_pairing": (
                        "same CUDA RNG state for clean and closure forward"),
                }),
            "phase_aligned_re_mix": bool(getattr(
                sim_args, "phase_aligned_re_mix", False)),
            "phase_aligned_re_mix_contract": (
                vars(ParameterMixContract(seed=SEED))
                if bool(getattr(sim_args, "phase_aligned_re_mix", False))
                else None),
            "phase_aligned_parameter_grid_mix": bool(getattr(
                sim_args, "phase_aligned_parameter_grid_mix", False)),
            "phase_aligned_parameter_grid_mix_contract": (
                vars(ParameterMixContract(
                    seed=SEED, include_aoa_neighbors=True))
                if bool(getattr(
                    sim_args, "phase_aligned_parameter_grid_mix", False))
                else None),
            "checkpoint_policy": (
                "candidate_only_after_u200_health_and_sim10000_completion"),
        },
        "evidence_class": (
            "sim_train_only_mechanism_and_initialization_artifact_not_official"),
        "wall_seconds": time.monotonic() - started,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def write_stop(
        output_dir: Path, status: str, error: BaseException,
        method: str = METHOD) -> None:
    path = output_dir / "stop.json"
    if path.exists():
        return
    write_json(path, {
        "method": method,
        "status": status,
        "error_type": type(error).__name__,
        "error": str(error),
    })


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal B210 functional Sim training requires CUDA")
    if sim_hgrad.sha256(args.real_stats) != REAL_STATS_SHA256:
        raise RuntimeError("Real stats SHA differs from frozen SPS proxy scale")
    selected_cache_sha = expected_teacher_cache_sha256(args)
    if sim_hgrad.sha256(args.teacher_cache) != selected_cache_sha:
        raise RuntimeError("current-P1 teacher cache SHA-256 changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim_args = training_args(args)
    closure_enabled = getattr(args, "real_closure_bank", None) is not None
    phase_aligned_re_mix = bool(getattr(args, "phase_aligned_re_mix", False))
    phase_aligned_parameter_grid_mix = bool(getattr(
        args, "phase_aligned_parameter_grid_mix", False))
    closure_equivariance = bool(getattr(
        args, "real_closure_equivariance", False))
    if phase_aligned_re_mix and not closure_enabled:
        raise ValueError("phase-aligned Re mix requires --real-closure-bank")
    if phase_aligned_re_mix and closure_equivariance:
        raise ValueError("phase-aligned Re mix is separate from equivariance")
    if phase_aligned_parameter_grid_mix and not closure_enabled:
        raise ValueError("phase-aligned grid mix requires --real-closure-bank")
    if phase_aligned_parameter_grid_mix and closure_equivariance:
        raise ValueError("phase-aligned grid mix is separate from equivariance")
    if bool(getattr(args, "budgeted_disagreement_allocation", False)):
        sim_args.method = BUDGETED_DISAGREEMENT_ALLOCATION_METHOD
    elif bool(getattr(args, "isolate_distribution_tower", False)):
        sim_args.method = ISOLATED_DISTRIBUTION_TOWER_METHOD
    elif bool(getattr(
            args, "conditional_future_crps_mean_preserving", False)):
        sim_args.method = (
            CONDITIONAL_FUTURE_CRPS_DETACHED_CONTEXT_METHOD
            if bool(getattr(
                args, "detach_mean_preserving_adapter_context", False))
            else CONDITIONAL_FUTURE_CRPS_MEAN_PRESERVING_METHOD)
    elif bool(getattr(args, "conditional_future_energy_adapter_only", False)):
        sim_args.method = CONDITIONAL_FUTURE_ADAPTER_ONLY_METHOD
    elif bool(getattr(args, "conditional_future_energy_members_only", False)):
        sim_args.method = CONDITIONAL_FUTURE_MEMBER_ONLY_METHOD
    elif bool(getattr(args, "conditional_future_energy", False)):
        sim_args.method = CONDITIONAL_FUTURE_ENERGY_METHOD
    elif closure_equivariance:
        sim_args.method = REAL_CLOSURE_EQUIVARIANCE_METHOD
    elif bool(getattr(args, "temporal_band_mixer", False)):
        sim_args.method = TEMPORAL_BAND_MIXER_METHOD
    elif (bool(getattr(args, "cross_conditioned_backbone", False))
          and bool(getattr(args, "member_feature_adapters", False))):
        sim_args.method = (
            STANDARD_XCB_MEMBER_FLUCTUATION_ROLE_METHOD
            if bool(getattr(args, "fluctuation_role_tower", False))
            else STANDARD_XCB_MEMBER_MULTISCALE_ENERGY_METHOD
            if (bool(getattr(
                args, "standard_objective_member_feature_adapters", False))
                and bool(getattr(
                    args, "multiscale_energy_controller", False)))
            else STANDARD_CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD
            if bool(getattr(
                args, "standard_objective_member_feature_adapters", False))
            else CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD)
    elif bool(getattr(args, "member_feature_adapters", False)):
        sim_args.method = MEMBER_FEATURE_ADAPTER_METHOD
    elif bool(getattr(args, "decoupled_member_residual", False)):
        sim_args.method = DECOUPLED_MEMBER_RESIDUAL_METHOD
    elif bool(getattr(
            args, "member_integrated_legacy_tke_gradient_floor", False)):
        sim_args.method = MEMBER_INTEGRATED_TKE_GRADIENT_FLOOR_METHOD
    elif bool(getattr(args, "member_official_objective", False)):
        sim_args.method = MEMBER_OFFICIAL_MEAN_TKE_FLOOR_METHOD
    elif bool(getattr(args, "legacy_tke_gradient_floor", False)):
        sim_args.method = LEGACY_TKE_GRADIENT_FLOOR_METHOD
    elif bool(getattr(args, "pareto_tke_projection", False)):
        sim_args.method = PARETO_TKE_PROJECTION_METHOD
    elif bool(getattr(
            args, "official_point_controller_tke_floor", False)):
        sim_args.method = OFFICIAL_POINT_TKE_FLOOR_METHOD
    elif bool(getattr(args, "official_point_objective", False)):
        sim_args.method = OFFICIAL_POINT_TEMPORAL_TRANSPORT_METHOD
    elif bool(getattr(args, "isoenergetic_temporal_transport", False)):
        sim_args.method = ISOENERGETIC_TEMPORAL_TRANSPORT_METHOD
    elif bool(getattr(args, "cross_conditioned_backbone", False)):
        sim_args.method = CROSS_CONDITIONED_BACKBONE_METHOD
    elif bool(getattr(
            args, "temporal_point_amplitude_factorization", False)):
        sim_args.method = TEMPORAL_AMPLITUDE_FACTORIZATION_METHOD
    elif bool(getattr(args, "temporal_point_decoder", False)):
        sim_args.method = TEMPORAL_POINT_DECODER_METHOD
    elif bool(getattr(
            args, "hybrid_history_multiscale_energy_controller", False)):
        sim_args.method = HYBRID_HISTORY_MULTISCALE_ENERGY_METHOD
    elif bool(getattr(args, "direct_history_tke_regression", False)):
        sim_args.method = DIRECT_HISTORY_TKE_METHOD
    elif bool(getattr(args, "history_energy_controller", False)):
        sim_args.method = HISTORY_ENERGY_METHOD
    elif bool(getattr(args, "multiscale_energy_controller", False)):
        sim_args.method = MULTISCALE_ENERGY_METHOD
    elif bool(getattr(args, "horizon_local_energy_controller", False)):
        sim_args.method = HORIZON_LOCAL_ENERGY_METHOD
    elif bool(getattr(args, "multiscale_tke_deep_supervision", False)):
        sim_args.method = MULTISCALE_TKE_METHOD
    elif phase_aligned_re_mix:
        sim_args.method = PHASE_ALIGNED_RE_MIX_METHOD
    elif phase_aligned_parameter_grid_mix:
        sim_args.method = PHASE_ALIGNED_GRID_MIX_METHOD
    elif closure_enabled:
        sim_args.method = REAL_CLOSURE_SEED1_METHOD
    rehearsal.ensemble_train.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")
    raw_loader, normalizer, train_windows, trajectories = (
        rehearsal.ensemble_train.common.prepare_sim(sim_args, device))
    if trajectories != rehearsal.FORMAL_SIM_TRAJECTORIES:
        raise RuntimeError(
            f"Sim trajectories={trajectories}, expected "
            f"{rehearsal.FORMAL_SIM_TRAJECTORIES}")
    manifest = json.loads(
        (args.output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    closure_augmentor = None
    if closure_enabled:
        actual_closure_sha = sim_hgrad.sha256(args.real_closure_bank)
        if actual_closure_sha != args.real_closure_bank_sha256:
            raise RuntimeError(
                "Real-closure bank SHA-256 changed: "
                f"actual={actual_closure_sha!r}/{len(actual_closure_sha)}, "
                f"expected={args.real_closure_bank_sha256!r}/"
                f"{len(args.real_closure_bank_sha256)}")
        closure_augmentor = ClosureResidualAugmentor.load(
            args.real_closure_bank, manifest,
            candidate_count=REAL_CLOSURE_CANDIDATE_COUNT)
        if closure_augmentor.source_contract.get(
                "manifest_sha256") != CLEAN_REAL_MANIFEST_SHA256:
            raise RuntimeError(
                "Real-closure bank is not tied to the frozen clean81 manifest")
        if closure_augmentor.source_contract.get("records") != 81:
            raise RuntimeError("Real-closure bank does not contain clean81")
        if not closure_augmentor.source_contract.get(
                "known_invalid_trajectory_excluded"):
            raise RuntimeError("Real-closure bank exclusion audit is absent")
        closure_augmentor.to(device)
        if phase_aligned_re_mix or phase_aligned_parameter_grid_mix:
            closure_augmentor = PhaseAlignedParameterMixAugmentor(
                closure_augmentor, raw_loader.dataset, manifest,
                ParameterMixContract(
                    seed=SEED,
                    include_aoa_neighbors=phase_aligned_parameter_grid_mix,
                )).to(device)
    teacher_cache, teacher_metadata = rehearsal.load_and_validate_cache(
        args.teacher_cache, manifest, raw_loader.dataset)
    if teacher_metadata["sha256"] != selected_cache_sha:
        raise RuntimeError("validated teacher cache differs from frozen P1")
    soft_sps_normalizer = rehearsal.ensemble_train.common.repro.Normalizer(
        args.real_stats, device)
    normalizer_values = rehearsal.ensemble_train.common.normalizer_dict(normalizer)
    normalizer_sha = rehearsal.canonical_sha256(normalizer_values)
    scale_contract = sim_hgrad.compute_sim_hgrad_scale_contract(
        raw_loader.dataset, rehearsal.canonical_sha256(manifest), normalizer_sha)
    write_json(args.output_dir / "hgrad_scale_contract.json", scale_contract)
    print(json.dumps({
        "event": "b210_np_sim_contract",
        "method": sim_args.method,
        "train_windows": train_windows,
        "trajectories": trajectories,
        "teacher_cache_sha256": selected_cache_sha,
        "formal_teacher_cache_sha256": TEACHER_CACHE_SHA256,
        "formal_cache_byte_identity": selected_cache_sha == TEACHER_CACHE_SHA256,
        "real_stats_sha256": REAL_STATS_SHA256,
        "parameters": contract_parameters(args),
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "real_closure_enabled": closure_enabled,
        "real_closure_equivariance": closure_equivariance,
        "phase_aligned_re_mix": phase_aligned_re_mix,
        "phase_aligned_parameter_grid_mix": phase_aligned_parameter_grid_mix,
        "multiscale_tke_deep_supervision": bool(getattr(
            args, "multiscale_tke_deep_supervision", False)),
        "multiscale_energy_controller": bool(getattr(
            args, "multiscale_energy_controller", False)),
        "horizon_local_energy_controller": bool(getattr(
            args, "horizon_local_energy_controller", False)),
        "history_energy_controller": bool(getattr(
            args, "history_energy_controller", False)),
        "direct_history_tke_regression": bool(getattr(
            args, "direct_history_tke_regression", False)),
        "direct_history_tke_weight": (
            DIRECT_HISTORY_TKE_WEIGHT
            if bool(getattr(args, "direct_history_tke_regression", False))
            else 0.0),
        "hybrid_history_multiscale_energy_controller": bool(getattr(
            args, "hybrid_history_multiscale_energy_controller", False)),
        "cross_conditioned_backbone": bool(getattr(
            args, "cross_conditioned_backbone", False)),
        "temporal_point_decoder": bool(getattr(
            args, "temporal_point_decoder", False)),
        "temporal_point_channels": TEMPORAL_POINT_CHANNELS,
        "temporal_point_blocks": TEMPORAL_POINT_BLOCKS,
        "temporal_point_kernel": TEMPORAL_POINT_KERNEL,
        "temporal_point_amplitude_factorization": bool(getattr(
            args, "temporal_point_amplitude_factorization", False)),
        "temporal_point_amplitude_limit": TEMPORAL_POINT_AMPLITUDE_LIMIT,
        "isoenergetic_temporal_transport": bool(getattr(
            args, "isoenergetic_temporal_transport", False)),
        "temporal_band_mixer": bool(getattr(
            args, "temporal_band_mixer", False)),
        "temporal_band_channels": TEMPORAL_BAND_CHANNELS,
        "temporal_band_count": TEMPORAL_BAND_COUNT,
        "temporal_band_log_gain_limit": TEMPORAL_BAND_LOG_GAIN_LIMIT,
        "temporal_band_phase_limit": TEMPORAL_BAND_PHASE_LIMIT,
        "temporal_transport_channels": TEMPORAL_TRANSPORT_CHANNELS,
        "temporal_transport_blocks": TEMPORAL_TRANSPORT_BLOCKS,
        "temporal_transport_kernel": TEMPORAL_TRANSPORT_KERNEL,
        "temporal_transport_max_displacement": (
            TEMPORAL_TRANSPORT_MAX_DISPLACEMENT),
        "official_point_objective": bool(getattr(
            args, "official_point_objective", False)),
        "official_point_controller_tke_floor": bool(getattr(
            args, "official_point_controller_tke_floor", False)),
        "pareto_tke_projection": bool(getattr(
            args, "pareto_tke_projection", False)),
        "pareto_projection_space": (
            "normalized_point_output" if bool(getattr(
                args, "pareto_tke_projection", False)) else None),
        "legacy_tke_gradient_floor": bool(getattr(
            args, "legacy_tke_gradient_floor", False)),
        "member_integrated_legacy_tke_gradient_floor": bool(getattr(
            args, "member_integrated_legacy_tke_gradient_floor", False)),
        "member_official_objective": bool(getattr(
            args, "member_official_objective", False)),
        "decoupled_member_residual": bool(getattr(
            args, "decoupled_member_residual", False)),
        "decoupled_member_residual_channels": (
            DECOUPLED_MEMBER_RESIDUAL_CHANNELS if bool(getattr(
                args, "decoupled_member_residual", False)) else 0),
        "decoupled_member_residual_blocks": (
            DECOUPLED_MEMBER_RESIDUAL_BLOCKS),
        "decoupled_member_residual_parameters": (
            EXPECTED_DECOUPLED_MEMBER_RESIDUAL_MODULE_PARAMETERS
            if bool(getattr(args, "decoupled_member_residual", False))
            else 0),
        "member_feature_adapters": bool(getattr(
            args, "member_feature_adapters", False)),
        "standard_objective_member_feature_adapters": bool(getattr(
            args, "standard_objective_member_feature_adapters", False)),
        "fluctuation_role_tower": bool(getattr(
            args, "fluctuation_role_tower", False)),
        "fluctuation_role_channels": (
            FLUCTUATION_ROLE_CHANNELS if bool(getattr(
                args, "fluctuation_role_tower", False)) else 0),
        "member_feature_adapter_channels": (
            MEMBER_FEATURE_ADAPTER_CHANNELS if bool(getattr(
                args, "member_feature_adapters", False)) else 0),
        "member_feature_adapter_parameters": (
            EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS
            if bool(getattr(args, "member_feature_adapters", False))
            else 0),
        "cross_conditioned_member_adapters": bool(
            getattr(args, "cross_conditioned_backbone", False)
            and getattr(args, "member_feature_adapters", False)),
        "conditional_future_energy": bool(getattr(
            args, "conditional_future_energy", False)),
        "conditional_future_energy_members_only": bool(getattr(
            args, "conditional_future_energy_members_only", False)),
        "conditional_future_energy_adapter_only": bool(getattr(
            args, "conditional_future_energy_adapter_only", False)),
        "conditional_future_crps_mean_preserving": bool(getattr(
            args, "conditional_future_crps_mean_preserving", False)),
        "mean_preserving_member_adapters": bool(getattr(
            args, "mean_preserving_member_adapters", False)),
        "detach_member_adapter_context": bool(getattr(
            args, "detach_member_adapter_context", False)),
        "detach_uq_context": bool(getattr(
            args, "detach_uq_context", False)),
        "isolate_distribution_tower": bool(getattr(
            args, "isolate_distribution_tower", False)),
        "budgeted_disagreement_allocation": bool(getattr(
            args, "budgeted_disagreement_allocation", False)),
        "conditional_future_probability": (
            CONDITIONAL_FUTURE_PROBABILITY if bool(
                getattr(args, "conditional_future_energy", False)
                or getattr(
                    args, "conditional_future_energy_members_only", False)
                or getattr(
                    args, "conditional_future_energy_adapter_only", False)
                or getattr(
                    args, "conditional_future_crps_mean_preserving", False))
            else 0.0),
        "conditional_future_amplitude": (
            CONDITIONAL_FUTURE_AMPLITUDE if bool(
                getattr(args, "conditional_future_energy", False)
                or getattr(
                    args, "conditional_future_energy_members_only", False)
                or getattr(
                    args, "conditional_future_energy_adapter_only", False)
                or getattr(
                    args, "conditional_future_crps_mean_preserving", False))
            else 0.0),
        "conditional_future_energy_weight": (
            CONDITIONAL_FUTURE_ENERGY_WEIGHT if bool(
                (getattr(args, "conditional_future_energy", False)
                 or getattr(
                    args, "conditional_future_energy_members_only", False)
                 or getattr(
                    args, "conditional_future_energy_adapter_only", False))
                and not getattr(
                    args, "conditional_future_crps_mean_preserving", False))
            else 0.0),
        "conditional_future_crps_weight": (
            CONDITIONAL_FUTURE_CRPS_WEIGHT if bool(getattr(
                args, "conditional_future_crps_mean_preserving", False))
            else 0.0),
        "conditional_future_member_mse_weight": (
            CONDITIONAL_FUTURE_MEMBER_MSE_WEIGHT if bool(
                getattr(args, "conditional_future_energy", False)
                or getattr(
                    args, "conditional_future_energy_members_only", False)
                or getattr(
                    args, "conditional_future_energy_adapter_only", False)
                or getattr(
                    args, "conditional_future_crps_mean_preserving", False))
            else 0.25),
        "legacy_tke_gradient_floor_scope": (
            ("per_sample_normalized_member_tensor" if bool(getattr(
                args, "member_integrated_legacy_tke_gradient_floor", False))
             else "per_sample_normalized_point_output") if bool(getattr(
                args, "legacy_tke_gradient_floor", False)) else None),
        "tke_aux_persistent_floor": (
            TKE_AUX_PERSISTENT_FLOOR_WEIGHT if bool(getattr(
                args, "official_point_controller_tke_floor", False)) else 0.0),
        "accuracy_weights_rel_tke_mvpe": list(sim_args.accuracy_weights),
        "real_closure_bank_sha256": (
            args.real_closure_bank_sha256 if closure_enabled else None),
        "phase_aligned_re_mix": phase_aligned_re_mix,
        "phase_aligned_re_mix_contract": (
            vars(ParameterMixContract(seed=SEED))
            if phase_aligned_re_mix else None),
        "phase_aligned_parameter_grid_mix": phase_aligned_parameter_grid_mix,
        "phase_aligned_parameter_grid_mix_contract": (
            vars(ParameterMixContract(
                seed=SEED, include_aoa_neighbors=True))
            if phase_aligned_parameter_grid_mix else None),
    }), flush=True)
    try:
        summary = run_experiment(
            args, sim_args, raw_loader, normalizer, soft_sps_normalizer,
            teacher_cache, teacher_metadata, scale_contract, device,
            closure_augmentor, closure_equivariance)
    except WallBudgetExceeded as error:
        write_stop(args.output_dir, WALL_STOP_STATUS, error, sim_args.method)
        raise
    except (torch.cuda.OutOfMemoryError, FloatingPointError, RuntimeError) as error:
        write_stop(args.output_dir, RUNTIME_STOP_STATUS, error, sim_args.method)
        raise
    if selected_cache_sha != TEACHER_CACHE_SHA256:
        if summary.get("checkpoint") is not None:
            summary["status"] = "RESEARCH_B210_SIM_COMPLETE_EQUIVALENT_CACHE"
        summary["evidence_class"] = (
            "research reproduction from random initialization; equivalent "
            "teacher semantics but non-formal cache byte identity")
        summary["teacher_cache_identity"] = {
            "actual_sha256": selected_cache_sha,
            "formal_sha256": TEACHER_CACHE_SHA256,
            "byte_exact": False,
            "semantic_validation": teacher_metadata,
        }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "event": "b210_np_sim_complete",
        "status": summary["status"],
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
    }), flush=True)


if __name__ == "__main__":
    main()
