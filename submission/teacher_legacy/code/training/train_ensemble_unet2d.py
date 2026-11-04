#!/usr/bin/env python3
"""Train the shared-backbone ensemble U-Net and residual quantile decoder."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (CODE_ROOT, Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from build_sim_p1_point_teacher_cache import (  # noqa: E402
    load_and_validate_cache,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sim", "real"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument(
        "--teacher-cache", type=Path,
        help="Current-P1 A/B/C point cache for optional Sim distillation.")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--real-all-windows", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--input-noise-std", type=float, default=0.0,
        help=("Training-only Gaussian noise in normalized u/v units; "
              "targets, pressure, and solid-body pixels remain unchanged."))

    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--unet-depth", type=int, default=4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--difference-steps", type=int, default=5)
    parser.add_argument(
        "--use-coordinates", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--use-regime-film", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--multi-scale-regime-film",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mean-fluctuation", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--ensemble-members", type=int, default=4)
    parser.add_argument("--uq-hidden-channels", type=int, default=64)
    parser.add_argument("--member-adapter-channels", type=int, default=0)
    parser.add_argument("--mean-profile-channels", type=int, default=0)
    parser.add_argument(
        "--uq-flow-context", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--residual-quantile-intervals",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--utility-refinement",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--utility-refinement-log-limit", type=float, default=0.75)
    parser.add_argument(
        "--utility-refinement-kernel", type=int, default=1)
    parser.add_argument(
        "--utility-refinement-blocks", type=int, default=0)
    parser.add_argument(
        "--temporal-uq-refinement",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-uq-channels", type=int, default=24)
    parser.add_argument("--temporal-uq-blocks", type=int, default=2)
    parser.add_argument("--temporal-uq-kernel", type=int, default=3)
    parser.add_argument(
        "--temporal-uq-logit-limit", type=float, default=1.5)
    parser.add_argument(
        "--coverage-calibration",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--coverage-calibration-kernel", type=int, default=3)
    parser.add_argument(
        "--coverage-calibration-blocks", type=int, default=1)
    parser.add_argument(
        "--coverage-calibration-logit-limit", type=float, default=2.0)
    parser.add_argument(
        "--coverage-calibration-mode",
        choices=("independent", "temperature"), default="independent")
    parser.add_argument(
        "--coverage-calibration-temperature-limit",
        type=float, default=0.75)
    parser.add_argument("--quantile-gap-bias", type=float, default=-4.0)
    parser.add_argument(
        "--disagreement-scale-fraction", type=float, default=0.01)

    parser.add_argument(
        "--point-loss-mode", choices=("mse", "official_accuracy"),
        default="official_accuracy")
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--mse-warmup-updates", type=int, default=0)
    parser.add_argument("--mean-loss-weight", type=float, default=1.0)
    parser.add_argument("--member-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--member-loss-mode", choices=("mse", "point"), default="mse")
    parser.add_argument("--bootstrap-probability", type=float, default=0.75)
    parser.add_argument("--quantile-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--soft-sps-loss-weight", type=float, default=0.0,
        help=("Jointly train native interval extents with a differentiable "
              "approximation of the official SPS utility."))
    parser.add_argument(
        "--soft-sps-tau-fraction", type=float, default=0.05)
    parser.add_argument(
        "--soft-sps-warmup-updates", type=int, default=0,
        help="Linearly ramp the joint SPS loss over this many updates.")
    parser.add_argument(
        "--soft-sps-uq-only",
        action=argparse.BooleanOptionalAction, default=False,
        help=("Backpropagate the SPS utility only through the distribution "
              "decoder, preserving the point/backbone optimization path."))
    parser.add_argument(
        "--soft-sps-proxy-real-scale",
        action=argparse.BooleanOptionalAction, default=True,
        help=("During Sim pretraining, map normalized Sim residuals onto the "
              "released Real u/v scale before applying the SPS utility."))
    parser.add_argument("--coverage-bce-weight", type=float, default=0.0)
    parser.add_argument("--profile-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--allow-real-start", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--trainable-scope", choices=("all", "mean_profile_refiner"),
        default="all")
    parser.add_argument(
        "--accuracy-weights", type=float, nargs=3,
        metavar=("REL", "TKE", "MVPE"), default=(5.0, 3.0, 2.0))
    parser.add_argument(
        "--selection-metric",
        choices=(
            "point_score", "native_sps_score", "mvpe_rel_l2_score"),
        default="native_sps_score")
    return parser.parse_args()


def model_config(args: argparse.Namespace) -> EnsembleUNet2dConfig:
    return EnsembleUNet2dConfig(
        base=args.base,
        depth=args.unet_depth,
        groups=args.groups,
        difference_steps=args.difference_steps,
        use_coordinates=args.use_coordinates,
        use_regime_film=args.use_regime_film,
        multi_scale_regime_film=args.multi_scale_regime_film,
        mean_fluctuation=args.mean_fluctuation,
        ensemble_members=args.ensemble_members,
        uq_hidden_channels=args.uq_hidden_channels,
        member_adapter_channels=args.member_adapter_channels,
        mean_profile_channels=args.mean_profile_channels,
        uq_flow_context=args.uq_flow_context,
        residual_quantile_intervals=args.residual_quantile_intervals,
        utility_refinement=args.utility_refinement,
        utility_refinement_log_limit=args.utility_refinement_log_limit,
        utility_refinement_kernel=args.utility_refinement_kernel,
        utility_refinement_blocks=args.utility_refinement_blocks,
        temporal_uq_refinement=args.temporal_uq_refinement,
        temporal_uq_channels=args.temporal_uq_channels,
        temporal_uq_blocks=args.temporal_uq_blocks,
        temporal_uq_kernel=args.temporal_uq_kernel,
        temporal_uq_logit_limit=args.temporal_uq_logit_limit,
        coverage_calibration=args.coverage_calibration,
        coverage_calibration_kernel=args.coverage_calibration_kernel,
        coverage_calibration_blocks=args.coverage_calibration_blocks,
        coverage_calibration_logit_limit=(
            args.coverage_calibration_logit_limit),
        coverage_calibration_mode=args.coverage_calibration_mode,
        coverage_calibration_temperature_limit=(
            args.coverage_calibration_temperature_limit),
        quantile_gap_bias=args.quantile_gap_bias,
        disagreement_scale_fraction=args.disagreement_scale_fraction,
    )


def validate(args: argparse.Namespace) -> tuple[float, float, float]:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.checkpoint_interval, args.lr,
        args.window_stride)
    if any(value <= 0 for value in positive):
        raise ValueError("training counts, learning rate and stride must be positive")
    nonnegative = (
        args.weight_decay, args.grad_clip, args.mse_weight,
        args.mean_loss_weight,
        args.member_loss_weight, args.quantile_loss_weight,
        args.profile_loss_weight, args.soft_sps_loss_weight,
        args.coverage_bce_weight, args.distill_weight,
        args.input_noise_std)
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("loss and optimizer values must be finite/non-negative")
    if not 0.0 < args.bootstrap_probability <= 1.0:
        raise ValueError("bootstrap_probability must lie in (0,1]")
    if (not math.isfinite(args.soft_sps_tau_fraction)
            or args.soft_sps_tau_fraction <= 0):
        raise ValueError("soft SPS tau fraction must be finite/positive")
    if args.coverage_bce_weight and not args.coverage_calibration:
        raise ValueError(
            "coverage BCE weight requires --coverage-calibration")
    if not 0 <= args.mse_warmup_updates <= args.updates:
        raise ValueError("MSE warmup must lie within the run")
    if not 0 <= args.soft_sps_warmup_updates <= args.updates:
        raise ValueError("soft SPS warmup must lie within the run")
    if args.phase == "sim" and args.start_checkpoint is not None:
        raise ValueError("Sim phase must start from random initialization")
    if args.phase == "real" and args.start_checkpoint is None:
        raise ValueError("Real phase requires a Sim checkpoint")
    if args.phase != "real" and args.real_all_windows:
        raise ValueError("real_all_windows is only valid in the Real phase")
    if args.phase != "real" and args.allow_real_start:
        raise ValueError("allow_real_start is only valid in the Real phase")
    if args.phase != "real" and args.trainable_scope != "all":
        raise ValueError("restricted trainable scope is only valid for Real")
    if (args.teacher_cache is None) != (args.distill_weight == 0.0):
        raise ValueError(
            "teacher cache and positive distill weight must be used together")
    if args.teacher_cache is not None and args.phase != "sim":
        raise ValueError("point-cache distillation is only supported in Sim")
    if (args.trainable_scope == "mean_profile_refiner"
            and args.mean_profile_channels <= 0):
        raise ValueError(
            "mean_profile_refiner scope requires mean-profile-channels > 0")
    weights = tuple(float(value) for value in args.accuracy_weights)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("accuracy weights must be finite/non-negative")
    if not any(weights):
        raise ValueError("at least one accuracy weight must be positive")
    model_config(args).validate()
    return weights


def build_or_load(
        args: argparse.Namespace, device: torch.device) -> EnsembleUNet2d:
    requested = model_config(args)
    if args.phase == "sim":
        return EnsembleUNet2d(requested).to(device)
    payload = torch.load(
        args.start_checkpoint, map_location="cpu", weights_only=False)
    start_phase = payload.get("phase")
    if start_phase != "sim" and not (
            args.allow_real_start and start_phase == "real"):
        raise ValueError(
            "Real phase must load a Sim checkpoint unless "
            "--allow-real-start is set for a Real continuation")
    stored = dict(payload["model_config"])
    if stored.pop("architecture") != "ensemble_unet":
        raise ValueError("start checkpoint is not an ensemble_unet")
    config = EnsembleUNet2dConfig(**stored)
    if config != requested:
        raise ValueError(
            "Real architecture flags must exactly match the Sim checkpoint")
    model = EnsembleUNet2d(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_training_input_noise(
        history_normalized: torch.Tensor,
        history_physical: torch.Tensor, noise_std: float,
        ) -> torch.Tensor:
    """Perturb normalized fluid u/v only; return by identity when disabled."""
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("input noise std must be finite and non-negative")
    if noise_std == 0.0:
        return history_normalized
    if (history_normalized.shape != history_physical.shape
            or history_normalized.shape[-1] < 2):
        raise ValueError("normalized and physical histories are not aligned")
    fluid = history_physical[..., :2].abs().sum(
        dim=-1, keepdim=True).ne(0.0)
    perturbed = history_normalized.clone()
    perturbed[..., :2].add_(
        noise_std * torch.randn_like(perturbed[..., :2])
        * fluid.to(perturbed.dtype))
    return perturbed


class TrajectoryTaggedDataset(Dataset):
    def __init__(self, base_dataset):
        if not hasattr(base_dataset, "windows"):
            raise TypeError("training dataset must expose window trajectory IDs")
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        history, target = self.base_dataset[index]
        trajectory = int(self.base_dataset.windows[index].trajectory)
        return history, target, trajectory, index


def tag_training_loader(loader, args: argparse.Namespace):
    tagged = TrajectoryTaggedDataset(loader.dataset)
    return DataLoader(
        tagged, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))


def trajectory_member_mask(
        trajectory_ids: torch.Tensor, probability: float, seed: int,
        member_count: int) -> torch.Tensor:
    """Return a deterministic trajectory-level member assignment mask."""
    trajectory_ids = trajectory_ids.to(dtype=torch.long, device="cpu").view(-1, 1)
    member_ids = torch.arange(member_count, dtype=torch.long).view(1, -1)
    modulus = 10007
    codes = (
        (trajectory_ids + 1) * 7919
        + (member_ids + 1) * 104729
        + (seed + 1) * 15485863
    ) % modulus
    threshold = int(round(probability * modulus))
    mask = codes < threshold
    empty_rows = ~mask.any(dim=1)
    if bool(empty_rows.any()):
        rows = torch.nonzero(empty_rows, as_tuple=False).flatten()
        fallback = (trajectory_ids[rows, 0] + seed) % member_count
        mask[rows, fallback] = True
    return mask


def bootstrap_member_mse(
        members: torch.Tensor, target: torch.Tensor,
        trajectory_ids: torch.Tensor, probability: float, seed: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample = (members - target[:, None]).square().mean(
        dim=(2, 3, 4, 5))
    mask = trajectory_member_mask(
        trajectory_ids, probability, seed, members.shape[1])
    mask = mask.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * mask).sum() / mask.sum(), mask.mean()


def residual_quantile_loss(
        model: EnsembleUNet2d, prediction: torch.Tensor,
        quantiles: torch.Tensor, target: torch.Tensor,
        normalizer,
        ) -> torch.Tensor:
    target_std = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    residual = (
        target[..., :2] - prediction[..., :2].detach()) * target_std
    quantiles_physical = quantiles * target_std.unsqueeze(-1)
    error = residual.unsqueeze(-1) - quantiles_physical
    levels = error.new_tensor(model.config.quantile_levels).view(
        1, 1, 1, 1, 1, -1)
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    truth_physical = common.to_physical_target(target, normalizer)[..., :2]
    scored = truth_physical != 0.0
    reference = pinball.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2, 1)
    normalized = pinball / reference
    expanded = scored.unsqueeze(-1).expand_as(normalized)
    return normalized[expanded].mean()


def soft_native_sps_objective(
        prediction: torch.Tensor, target: torch.Tensor,
        lower_extent: torch.Tensor, upper_extent: torch.Tensor,
        normalizer, tau_fraction: float, metric_normalizer=None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable official-SPS proxy for joint representation learning."""
    prediction_physical = common.to_physical_target(
        prediction, normalizer)[..., :2]
    target_physical = common.to_physical_target(
        target, normalizer)[..., :2]
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower = lower_extent * scale
    upper = upper_extent * scale
    residual = target_physical - prediction_physical.detach()
    # Sim pretraining may deliberately evaluate interval widths on the Real
    # u/v scale.  Absolute point accuracy and the solid-body mask must still
    # come from the dataset's own physical normalization; interpreting a Sim
    # normalized zero with Real mean/std would turn masked cells into flow.
    metric_normalizer = metric_normalizer or normalizer
    prediction_metric = common.to_physical_target(
        prediction.detach(), metric_normalizer)[..., :2]
    target_metric = common.to_physical_target(
        target, metric_normalizer)[..., :2]
    scored = target_metric != 0.0
    reference = lower.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2)
    tau = tau_fraction * reference
    soft_inside = (
        torch.sigmoid((residual + lower) / tau)
        * torch.sigmoid((upper - residual) / tau))
    efficiency = soft_inside * torch.exp(
        -(lower + upper) / common.SPS_SIGMA)
    count = scored.sum(dim=(1, 2, 3, 4)).clamp_min(1)
    per_sample = (
        (efficiency * scored).sum(dim=(1, 2, 3, 4)) / count)
    factor = common.sample_sps_factor(
        prediction_metric, target_metric).detach()
    soft_score = (factor * per_sample).mean()
    return -soft_score, soft_score.detach()


def coverage_calibration_objective(
        model: EnsembleUNet2d, prediction: torch.Tensor,
        quantiles: torch.Tensor, coverage_logits: torch.Tensor,
        target: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Proper binary loss for candidate-specific conditional hit rates."""
    lower, upper, valid, _masses = model._interval_candidates(quantiles)
    residual = (
        target[..., :2] - prediction[..., :2].detach()).unsqueeze(-1)
    inside = (residual >= -lower) & (residual <= upper)
    truth_physical = common.to_physical_target(
        target, normalizer)[..., :2]
    mask = truth_physical.ne(0.0).unsqueeze(-1).expand_as(inside) & valid
    return torch.nn.functional.binary_cross_entropy_with_logits(
        coverage_logits[mask], inside[mask].to(coverage_logits.dtype))


def uq_trainable_parameters(model: EnsembleUNet2d) -> list[torch.nn.Parameter]:
    """Return de-duplicated parameters that affect only interval prediction."""
    modules = [
        model.uq_projections,
        model.uq_fusion,
        model.quantile_head,
        model.utility_refinement_head,
        model.temporal_uq_refiner,
        model.coverage_calibration_head,
    ]
    result = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def point_loss(
        args: argparse.Namespace, prediction: torch.Tensor,
        target: torch.Tensor, normalizer,
        weights: tuple[float, float, float], iteration: int,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mse = (prediction - target).square().mean()
    components = {"mse": mse.detach()}
    warmup = iteration <= args.mse_warmup_updates
    if args.point_loss_mode == "mse" or warmup:
        loss = mse
    else:
        accuracy, details = common.accuracy_objective(
            prediction, target, normalizer, weights)
        loss = accuracy + args.mse_weight * mse
        components.update(details)
    components["mse_warmup"] = mse.new_tensor(float(warmup))
    return loss, components


def mean_profile_objective(
        prediction: torch.Tensor, target: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Exact differentiable v9 MVPE error on the four probe profiles."""
    pred = common.to_physical_target(prediction, normalizer)[..., :2]
    truth = common.to_physical_target(target, normalizer)[..., :2]
    batch = pred.shape[0]
    center_y = pred.shape[2] // 2
    probe_y = [center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < pred.shape[2]]
    errors = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if not 0 <= probe_x < pred.shape[3]:
            continue
        pred_profile = pred[:, :, probe_y, probe_x].mean(
            dim=1).reshape(batch, -1)
        truth_profile = truth[:, :, probe_y, probe_x].mean(
            dim=1).reshape(batch, -1)
        errors.append(
            torch.linalg.vector_norm(pred_profile - truth_profile, dim=1)
            / torch.linalg.vector_norm(
                truth_profile, dim=1).clamp_min(1e-8))
    if not errors:
        raise ValueError("no valid MVPE probes for this spatial resolution")
    return torch.stack(errors).mean()


POINT_DISTILLATION_COMPONENT_WEIGHTS = {
    "temporal_mean": 2.0,
    "zero_mean_fluctuation": 1.0,
    "mean_profile": 2.0,
    "physical_tke_map": 0.25,
}
POINT_DISTILLATION_SCALES = {
    "temporal_mean_normalized": 0.025,
    "zero_mean_fluctuation_normalized": 0.020,
    "mean_profile_normalized": 0.025,
}


def mean_profile_distillation_loss(
        student_mean: torch.Tensor, teacher_mean: torch.Tensor,
        scale: float,
        ) -> torch.Tensor:
    center_y = student_mean.shape[2] // 2
    probe_y = [center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < student_mean.shape[2]]
    losses = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if 0 <= probe_x < student_mean.shape[3]:
            losses.append(F.smooth_l1_loss(
                student_mean[:, :, probe_y, probe_x] / scale,
                teacher_mean[:, :, probe_y, probe_x] / scale))
    if not losses:
        raise ValueError("no valid mean-profile probes")
    return torch.stack(losses).mean()


def physical_tke_map_distillation_loss(
        student_normalized: torch.Tensor,
        teacher_physical: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    student_physical = common.to_physical_target(
        student_normalized, normalizer)[..., :2].float()
    teacher_physical = teacher_physical.float()

    def kinetic_energy(value: torch.Tensor) -> torch.Tensor:
        fluctuation = value - value.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)

    student_tke = kinetic_energy(student_physical)
    teacher_tke = kinetic_energy(teacher_physical)
    batch = student_tke.shape[0]
    relative = torch.linalg.vector_norm(
        (student_tke - teacher_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        teacher_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)
    return relative.mean()


def component_point_distillation_loss(
        student_normalized: torch.Tensor,
        teacher_physical: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match the formal point teacher without changing truth or UQ losses."""
    if (teacher_physical.ndim != student_normalized.ndim
            or teacher_physical.shape[:-1] != student_normalized.shape[:-1]
            or teacher_physical.shape[-1] != 2):
        raise ValueError("teacher point batch is not aligned with the student")
    teacher_normalized = (
        (teacher_physical.float() - normalizer.mean_target[:2])
        / normalizer.std_target[:2])
    student_uv = student_normalized[..., :2].float()
    student_mean = student_uv.mean(dim=1, keepdim=True)
    teacher_mean = teacher_normalized.mean(dim=1, keepdim=True)
    student_fluctuation = student_uv - student_mean
    teacher_fluctuation = teacher_normalized - teacher_mean

    mean_loss = F.smooth_l1_loss(
        student_mean / POINT_DISTILLATION_SCALES[
            "temporal_mean_normalized"],
        teacher_mean / POINT_DISTILLATION_SCALES[
            "temporal_mean_normalized"])
    fluctuation_loss = F.smooth_l1_loss(
        student_fluctuation / POINT_DISTILLATION_SCALES[
            "zero_mean_fluctuation_normalized"],
        teacher_fluctuation / POINT_DISTILLATION_SCALES[
            "zero_mean_fluctuation_normalized"])
    profile_loss = mean_profile_distillation_loss(
        student_mean, teacher_mean,
        POINT_DISTILLATION_SCALES["mean_profile_normalized"])
    tke_loss = physical_tke_map_distillation_loss(
        student_normalized, teacher_physical, normalizer)
    loss = (
        POINT_DISTILLATION_COMPONENT_WEIGHTS["temporal_mean"] * mean_loss
        + POINT_DISTILLATION_COMPONENT_WEIGHTS[
            "zero_mean_fluctuation"] * fluctuation_loss
        + POINT_DISTILLATION_COMPONENT_WEIGHTS["mean_profile"] * profile_loss
        + POINT_DISTILLATION_COMPONENT_WEIGHTS[
            "physical_tke_map"] * tke_loss)
    return loss, {
        "distill_temporal_mean": mean_loss.detach(),
        "distill_zero_mean_fluctuation": fluctuation_loss.detach(),
        "distill_mean_profile": profile_loss.detach(),
        "distill_physical_tke_map": tke_loss.detach(),
    }


def add_point_distillation_loss(
        base_loss: torch.Tensor, student_normalized: torch.Tensor,
        teacher_physical: torch.Tensor | None, normalizer,
        distill_weight: float,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``base_loss`` by identity when distillation is disabled."""
    if teacher_physical is None:
        if distill_weight != 0.0:
            raise ValueError("positive distill weight requires a teacher batch")
        return base_loss, {}
    if not math.isfinite(distill_weight) or distill_weight <= 0.0:
        raise ValueError("teacher batch requires a finite positive distill weight")
    distillation, components = component_point_distillation_loss(
        student_normalized, teacher_physical, normalizer)
    return base_loss + distill_weight * distillation, {
        "distillation_loss": distillation.detach(),
        "distillation_weight": distillation.new_tensor(distill_weight),
        **components,
    }


def bootstrap_member_point_loss(
        args: argparse.Namespace, members: torch.Tensor,
        target: torch.Tensor, trajectory_ids: torch.Tensor,
        normalizer, weights: tuple[float, float, float], iteration: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Train each member only on its fixed trajectory bootstrap subset."""
    mask = trajectory_member_mask(
        trajectory_ids, args.bootstrap_probability, args.seed,
        members.shape[1])
    losses = []
    for member_index in range(members.shape[1]):
        selected = mask[:, member_index].to(device=members.device)
        if bool(selected.any()):
            member_loss, _ = point_loss(
                args, members[selected, member_index], target[selected],
                normalizer, weights, iteration)
            losses.append(member_loss)
    if not losses:
        raise RuntimeError("trajectory bootstrap selected no ensemble members")
    fraction = mask.to(device=members.device, dtype=members.dtype).mean()
    return torch.stack(losses).mean(), fraction


def evaluate(
        model: EnsembleUNet2d, loader, normalizer,
        device: torch.device,
        weights: tuple[float, float, float]) -> dict[str, float]:
    metrics = common.repro.evaluate(model, loader, normalizer, device)
    common.add_weighted(metrics, weights)
    metrics.update(common.evaluate_native_sps(
        model, loader, normalizer, device))
    return metrics


def train(
        args: argparse.Namespace, model: EnsembleUNet2d,
        train_loader, normalizer, device: torch.device,
        weights: tuple[float, float, float],
        train_windows: int, trajectories: int,
        val_loader=None, val_windows: int = 0,
        soft_sps_normalizer=None,
        teacher_cache: torch.Tensor | None = None,
        ) -> dict:
    if args.trainable_scope == "mean_profile_refiner":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if model.mean_profile_refiner is None:
            raise ValueError("checkpoint has no mean-profile refiner")
        for parameter in model.mean_profile_refiner.parameters():
            parameter.requires_grad_(True)
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    if not trainable:
        raise ValueError("selected trainable scope contains no parameters")
    iterator = common.repro.cycling(train_loader)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    stats = common.normalizer_dict(normalizer)
    history = []
    sums: dict[str, float] = {}
    count = 0
    best_value = float("inf") if args.phase == "sim" else -float("inf")
    best_iteration = 0
    started = time.time()

    if val_loader is not None:
        initial = evaluate(model, val_loader, normalizer, device, weights)
        print(json.dumps({"event": "real_initial", **initial}), flush=True)
    else:
        initial = None

    for iteration in range(1, args.updates + 1):
        model.train()
        history_batch, target, trajectory_ids, sample_indices = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history_batch, target)
        if args.input_noise_std:
            history_norm = add_training_input_noise(
                history_norm, history_batch, args.input_noise_std)
        optimizer.zero_grad(set_to_none=True)
        (prediction, members, quantiles, lower, upper, _features,
         coverage_logits) = model.forward_uq_training(history_norm)
        mean_loss, components = point_loss(
            args, prediction, target_norm, normalizer, weights, iteration)
        if args.member_loss_mode == "point":
            member_loss, bootstrap_fraction = bootstrap_member_point_loss(
                args, members, target_norm, trajectory_ids,
                normalizer, weights, iteration)
        else:
            member_loss, bootstrap_fraction = bootstrap_member_mse(
                members, target_norm, trajectory_ids,
                args.bootstrap_probability, args.seed)
        quantile_loss = residual_quantile_loss(
            model, prediction, quantiles, target_norm, normalizer)
        if args.soft_sps_loss_weight:
            soft_sps_loss, soft_sps_score = soft_native_sps_objective(
                prediction, target_norm, lower, upper,
                soft_sps_normalizer or normalizer,
                args.soft_sps_tau_fraction,
                metric_normalizer=normalizer)
        else:
            soft_sps_loss = prediction.new_zeros(())
            soft_sps_score = prediction.new_zeros(())
        coverage_bce = (
            coverage_calibration_objective(
                model, prediction, quantiles, coverage_logits,
                target_norm, normalizer)
            if coverage_logits is not None
            else prediction.new_zeros(()))
        profile_loss = (
            mean_profile_objective(
                prediction, target_norm, normalizer)
            if args.profile_loss_weight else prediction.new_zeros(()))
        soft_sps_weight = args.soft_sps_loss_weight * min(
            1.0,
            iteration / max(args.soft_sps_warmup_updates, 1))
        base_loss = (
            args.mean_loss_weight * mean_loss
            + args.member_loss_weight * member_loss
            + args.quantile_loss_weight * quantile_loss
            + args.coverage_bce_weight * coverage_bce
            + args.profile_loss_weight * profile_loss)
        teacher_batch = (
            teacher_cache[sample_indices].to(
                device=device, dtype=torch.float32, non_blocking=True)
            if teacher_cache is not None else None)
        supervised_loss, distillation_components = add_point_distillation_loss(
            base_loss, prediction, teacher_batch, normalizer,
            args.distill_weight)
        loss = supervised_loss + soft_sps_weight * soft_sps_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite ensemble loss at {iteration}")
        if args.soft_sps_uq_only and soft_sps_weight:
            supervised_loss.backward(retain_graph=True)
            interval_parameters = uq_trainable_parameters(model)
            interval_gradients = torch.autograd.grad(
                soft_sps_weight * soft_sps_loss,
                interval_parameters, allow_unused=True)
            for parameter, gradient in zip(
                    interval_parameters, interval_gradients):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
        else:
            loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        physical_spread = (
            members[..., :2].std(dim=1, unbiased=False)
            * normalizer.std_target[:2].view(1, 1, 1, 1, 2))
        values = {
            "loss": loss.detach(),
            "member_loss": member_loss.detach(),
            "bootstrap_fraction": bootstrap_fraction.detach(),
            "quantile_pinball": quantile_loss.detach(),
            "soft_sps_loss": soft_sps_loss.detach(),
            "soft_sps_score": soft_sps_score,
            "soft_sps_weight": prediction.new_tensor(soft_sps_weight),
            "coverage_bce": coverage_bce.detach(),
            "profile_loss": profile_loss.detach(),
            "member_spread_physical": physical_spread.mean().detach(),
            **components,
            **distillation_components,
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

        if iteration % args.eval_interval == 0 or iteration == args.updates:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
            }
            if val_loader is not None:
                metrics = evaluate(model, val_loader, normalizer, device, weights)
                row.update(metrics)
                selected = metrics[args.selection_metric]
                improved = selected > best_value
            else:
                selected = row["train_loss_mean"]
                improved = selected < best_value
            if improved:
                best_value = selected
                best_iteration = iteration
                common.save_checkpoint(
                    args.output_dir / "best.pth", model, args, iteration,
                    history + [row], stats, row)
            history.append(row)
            print(json.dumps({"event": f"{args.phase}_eval", **row}), flush=True)
            sums.clear()
            count = 0
        if iteration % args.checkpoint_interval == 0:
            common.save_checkpoint(
                args.output_dir / f"model_{iteration:05d}.pth",
                model, args, iteration, history, stats)

    final_metrics = (
        evaluate(model, val_loader, normalizer, device, weights)
        if val_loader is not None else history[-1])
    common.save_checkpoint(
        args.output_dir / "final.pth", model, args, args.updates,
        history, stats, final_metrics)
    return {
        "phase": args.phase,
        "updates": args.updates,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "initial_metrics": initial,
        "final_metrics": final_metrics,
        "best_iteration": best_iteration,
        "best_metric_value": best_value,
        "selection_metric": (
            "train_loss_mean" if args.phase == "sim"
            else args.selection_metric),
        "history": history,
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    weights = validate(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    model = build_or_load(args, device)
    parameter_count = sum(value.numel() for value in model.parameters())
    print(json.dumps({
        "event": "model", "phase": args.phase,
        "config": model.architecture_config(),
        "parameters": parameter_count,
    }), flush=True)

    if args.phase == "sim":
        train_loader, normalizer, train_windows, trajectories = (
            common.prepare_sim(args, device))
        val_loader = None
        val_windows = 0
    else:
        (train_loader, val_loader, normalizer, train_windows,
         val_windows, trajectories) = common.prepare_real(args, device)
    teacher_cache = None
    teacher_cache_metadata = None
    if args.teacher_cache is not None:
        manifest = json.loads(
            (args.output_dir / "split_manifest.json").read_text(
                encoding="utf-8"))
        teacher_cache, teacher_cache_metadata = load_and_validate_cache(
            args.teacher_cache, manifest, train_loader.dataset)
        print(json.dumps({
            "event": "teacher_cache", **teacher_cache_metadata,
        }), flush=True)
    train_loader = tag_training_loader(train_loader, args)
    common.set_model_stats(model, normalizer)
    soft_sps_normalizer = normalizer
    if (args.phase == "sim" and args.soft_sps_loss_weight
            and args.soft_sps_proxy_real_scale):
        soft_sps_normalizer = common.repro.Normalizer(
            args.real_stats, device)
    summary = train(
        args, model, train_loader, normalizer, device, weights,
        train_windows, trajectories, val_loader, val_windows,
        soft_sps_normalizer, teacher_cache)
    summary.update({
        "model_config": model.architecture_config(),
        "parameters": parameter_count,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    })
    if teacher_cache_metadata is not None:
        summary["point_distillation"] = {
            "distill_weight": args.distill_weight,
            "component_weights": POINT_DISTILLATION_COMPONENT_WEIGHTS,
            "component_scales": POINT_DISTILLATION_SCALES,
            "teacher_cache": teacher_cache_metadata,
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if summary["history"]:
        with (args.output_dir / "learning_curve.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            fields = list(dict.fromkeys(
                key for row in summary["history"] for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary["history"])
    print(json.dumps({
        "event": "complete", "phase": args.phase,
        "wall_seconds": summary["wall_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
