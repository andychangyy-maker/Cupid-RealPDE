#!/usr/bin/env python3
"""Condition-fold OOF training for a frozen ensemble residual decoder.

The point backbone and every ensemble member head remain frozen. Only the
multi-scale UQ projections, fusion decoder, and monotone quantile head see
temporal-tail residuals that were not used for point-model optimization.
Optionally, blended lower/upper extents from two frozen teacher forecasters
regularize the native quantile decoder during training only. Released
simulation windows can also rehearse standardized residual structure while
Real condition folds remain strictly out of fold.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ensemble_unet2d as ensemble_train  # noqa: E402
import train_fno_sim_from_scratch as sim_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d  # noqa: E402
from train_distilled_flow_corrector import load_forecaster  # noqa: E402
from train_flow_backbone import construct_model  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


UQ_PREFIXES = (
    "uq_projections.", "uq_fusion.", "quantile_head.",
    "utility_refinement_head.", "temporal_uq_refiner.",
    "coverage_calibration_head.", "residual_mixture_head.",
    "output_space_uq_allocator.", "output_space_uq_budgeter.")

BEST_HELD_FOLD_RULE = "best_held_fold_sps"
FIXED_TERMINAL_RULE = "fixed_terminal_no_held_fold_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--fold-updates", type=int, default=750)
    parser.add_argument(
        "--fit-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--final-updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=150)
    parser.add_argument(
        "--fixed-terminal-folds",
        action=argparse.BooleanOptionalAction, default=False,
        help=("Retain the terminal fold checkpoint without selecting on "
              "held-fold SPS. Default behavior keeps the best held-fold "
              "checkpoint."))
    parser.add_argument(
        "--frozen-point-eval-mode",
        action=argparse.BooleanOptionalAction, default=False,
        help=("Keep the frozen point/backbone path in eval mode while only "
              "the trainable UQ modules enter train mode. This is required "
              "for point checkpoints containing residual dropout."))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pinball-weight", type=float, default=0.25)
    parser.add_argument("--soft-sps-weight", type=float, default=1.0)
    parser.add_argument("--interval-score-weight", type=float, default=0.01)
    parser.add_argument("--cvar-weight", type=float, default=0.10)
    parser.add_argument("--cvar-fraction", type=float, default=0.25)
    parser.add_argument("--coverage-target", type=float, default=0.85)
    parser.add_argument("--soft-tau-fraction", type=float, default=0.10)
    parser.add_argument(
        "--robust-sps-weight", type=float, default=0.0,
        help=("Additional SPS utility weight for antithetic residual jitter "
              "and a mild outward residual-tail stress."))
    parser.add_argument(
        "--robust-residual-jitter-fraction", type=float, default=0.0,
        help="Antithetic residual offset relative to the u/v width reference.")
    parser.add_argument(
        "--robust-residual-tail-scale", type=float, default=1.0,
        help="Outward multiplicative residual stress; must be at least one.")
    parser.add_argument(
        "--peer-sps-weight", type=float, default=0.0,
        help=("Additional SPS utility weight for antithetic residual fields "
              "borrowed from the nearest forecast-dynamics peer in a batch."))
    parser.add_argument(
        "--peer-residual-mix-fraction", type=float, default=0.0,
        help=("Fraction of the nearest-peer residual difference used for the "
              "structured antithetic perturbation; must lie in [0, 1]."))
    parser.add_argument("--extent-teacher-a", type=Path)
    parser.add_argument("--extent-teacher-b", type=Path)
    parser.add_argument("--extent-teacher-weight-b", type=float, default=0.25)
    parser.add_argument("--extent-distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--train-projections", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--add-flow-context", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--residual-quantile-intervals",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--utility-refinement",
        action=argparse.BooleanOptionalAction, default=False,
        help=("Add a zero-initialized continuous lower/upper refinement after "
              "discrete quantile utility selection."))
    parser.add_argument(
        "--utility-refinement-log-limit", type=float, default=0.75)
    parser.add_argument(
        "--utility-refinement-kernel", type=int, default=1)
    parser.add_argument(
        "--utility-refinement-blocks", type=int, default=0)
    parser.add_argument(
        "--temporal-uq-refinement",
        action=argparse.BooleanOptionalAction, default=False,
        help=("Add a zero-initialized 3-D residual-quantile decoder over "
              "forecast horizon and image space."))
    parser.add_argument("--temporal-uq-channels", type=int, default=24)
    parser.add_argument("--temporal-uq-blocks", type=int, default=2)
    parser.add_argument("--temporal-uq-kernel", type=int, default=3)
    parser.add_argument(
        "--temporal-uq-logit-limit", type=float, default=1.5)
    parser.add_argument(
        "--coverage-calibration",
        action=argparse.BooleanOptionalAction, default=False,
        help=("Learn conditional hit probabilities for every quantile-pair "
              "candidate instead of assuming nominal quantile mass."))
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
    parser.add_argument("--coverage-bce-weight", type=float, default=0.10)
    parser.add_argument("--real-input-gain-std", type=float, default=0.0)
    parser.add_argument("--real-input-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--real-input-blur-probability", type=float, default=0.0)
    parser.add_argument("--real-input-blur-strength", type=float, default=0.0)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--sim-rehearsal-weight", type=float, default=0.0,
        help=("Weight of released-Sim standardized residual rehearsal. "
              "Zero preserves Real-only OOF training."))
    parser.add_argument(
        "--sim-rehearsal-mode", choices=("official", "allocation"),
        default="official",
        help=("official reuses the complete SPS proxy; allocation transfers "
              "only within-window log residual structure."))
    parser.add_argument(
        "--sim-allocation-floor-fraction", type=float, default=1e-3,
        help="Numerical log floor relative to the per-channel Real reference.")
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument("--sim-stats", type=Path)
    parser.add_argument("--sim-window-stride", type=int, default=40)
    parser.add_argument("--sim-max-files", type=int, default=0)
    parser.add_argument(
        "--sim-batch-size", type=int, default=0,
        help="Sim rehearsal batch size; zero reuses --batch-size.")
    parser.add_argument(
        "--zero-sim-pressure", action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove pressure after Sim normalization to match Real PIV input.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def fold_selection_contract(fixed_terminal: bool) -> dict[str, object]:
    if fixed_terminal:
        return {
            "select_best": False,
            "selection_rule": FIXED_TERMINAL_RULE,
            "held_fold_checkpoint_selection": False,
        }
    return {
        "select_best": True,
        "selection_rule": BEST_HELD_FOLD_RULE,
        "held_fold_checkpoint_selection": True,
    }


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.folds, args.fold_updates, args.batch_size,
        args.eval_batch_size, args.eval_interval, args.lr,
        args.window_stride, args.sim_window_stride)
    if any(value <= 0 for value in positive):
        raise ValueError("counts, learning rate and stride must be positive")
    if args.folds < 2:
        raise ValueError("at least two condition folds are required")
    if args.fold_limit < 0 or args.fold_limit > args.folds:
        raise ValueError("fold_limit must lie in [0, folds]")
    finite_nonnegative = (
        args.weight_decay, args.grad_clip, args.pinball_weight,
        args.soft_sps_weight, args.interval_score_weight,
        args.cvar_weight, args.extent_distill_weight,
        args.sim_rehearsal_weight, args.coverage_bce_weight,
        args.real_input_gain_std, args.real_input_noise_std,
        args.real_input_blur_strength, args.robust_sps_weight,
        args.robust_residual_jitter_fraction, args.peer_sps_weight,
        args.peer_residual_mix_fraction)
    if any(not math.isfinite(value) or value < 0
           for value in finite_nonnegative):
        raise ValueError("optimizer values must be finite/non-negative")
    if args.fit_final and args.final_updates <= 0:
        raise ValueError("final_updates must be positive when refitting")
    if args.sim_batch_size < 0:
        raise ValueError("sim_batch_size must be non-negative")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val_fraction must lie in (0, 0.5)")
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("cvar_fraction must lie in (0, 1]")
    if not 0.0 < args.coverage_target < 1.0:
        raise ValueError("coverage_target must lie in (0, 1)")
    if not 0.0 <= args.real_input_blur_probability <= 1.0:
        raise ValueError("real input blur probability must lie in [0, 1]")
    if not math.isfinite(args.soft_tau_fraction) or args.soft_tau_fraction <= 0:
        raise ValueError("soft_tau_fraction must be finite/positive")
    if (not math.isfinite(args.robust_residual_tail_scale)
            or args.robust_residual_tail_scale < 1.0):
        raise ValueError("robust residual tail scale must be finite and >= 1")
    if (args.robust_sps_weight > 0.0
            and args.robust_residual_jitter_fraction <= 0.0):
        raise ValueError("robust SPS requires positive residual jitter")
    if args.peer_residual_mix_fraction > 1.0:
        raise ValueError("peer residual mix fraction must not exceed one")
    if args.peer_sps_weight > 0.0 and args.peer_residual_mix_fraction <= 0.0:
        raise ValueError("peer SPS requires a positive residual mix")
    if (not math.isfinite(args.sim_allocation_floor_fraction)
            or args.sim_allocation_floor_fraction <= 0):
        raise ValueError("sim allocation floor must be finite/positive")
    if (not math.isfinite(args.utility_refinement_log_limit)
            or args.utility_refinement_log_limit <= 0):
        raise ValueError("utility refinement log limit must be finite/positive")
    if (args.utility_refinement_kernel <= 0
            or args.utility_refinement_kernel % 2 == 0):
        raise ValueError("utility refinement kernel must be positive and odd")
    if args.utility_refinement_blocks < 0:
        raise ValueError("utility refinement blocks must be non-negative")
    if args.temporal_uq_channels <= 0 or args.temporal_uq_blocks <= 0:
        raise ValueError("temporal UQ channels/blocks must be positive")
    if (args.temporal_uq_kernel <= 0
            or args.temporal_uq_kernel % 2 == 0):
        raise ValueError("temporal UQ kernel must be positive and odd")
    if (not math.isfinite(args.temporal_uq_logit_limit)
            or args.temporal_uq_logit_limit <= 0):
        raise ValueError("temporal UQ logit limit must be finite/positive")
    if (args.coverage_calibration_kernel <= 0
            or args.coverage_calibration_kernel % 2 == 0):
        raise ValueError(
            "coverage calibration kernel must be positive and odd")
    if args.coverage_calibration_blocks < 0:
        raise ValueError(
            "coverage calibration blocks must be non-negative")
    if (not math.isfinite(args.coverage_calibration_logit_limit)
            or args.coverage_calibration_logit_limit <= 0):
        raise ValueError(
            "coverage calibration logit limit must be finite/positive")
    if (not math.isfinite(args.coverage_calibration_temperature_limit)
            or args.coverage_calibration_temperature_limit <= 0):
        raise ValueError(
            "coverage calibration temperature limit must be finite/positive")
    if not 0.0 <= args.extent_teacher_weight_b <= 1.0:
        raise ValueError("extent teacher weight-b must lie in [0, 1]")
    teachers = (args.extent_teacher_a, args.extent_teacher_b)
    if (args.extent_distill_weight > 0) != all(teachers):
        raise ValueError(
            "positive extent distillation requires both teacher checkpoints")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(payload: dict, normalizer, device: torch.device,
               train_projections: bool,
               add_flow_context: bool = False,
               residual_quantile_intervals: bool = False,
               utility_refinement: bool = False,
               utility_refinement_log_limit: float = 0.75,
               utility_refinement_kernel: int = 1,
               utility_refinement_blocks: int = 0,
               temporal_uq_refinement: bool = False,
               temporal_uq_channels: int = 24,
               temporal_uq_blocks: int = 2,
               temporal_uq_kernel: int = 3,
               temporal_uq_logit_limit: float = 1.5,
               coverage_calibration: bool = False,
               coverage_calibration_kernel: int = 3,
               coverage_calibration_blocks: int = 1,
               coverage_calibration_logit_limit: float = 2.0,
               coverage_calibration_mode: str = "independent",
               coverage_calibration_temperature_limit: float = 0.75,
               ) -> EnsembleUNet2d:
    architecture = payload["model_config"]["architecture"]
    if architecture != "ensemble_unet":
        raise ValueError("point checkpoint must use ensemble_unet")
    model_config = dict(payload["model_config"])
    if add_flow_context:
        model_config["uq_flow_context"] = True
    if residual_quantile_intervals:
        model_config["residual_quantile_intervals"] = True
    if utility_refinement:
        model_config["utility_refinement"] = True
        model_config["utility_refinement_log_limit"] = (
            utility_refinement_log_limit)
        model_config["utility_refinement_kernel"] = utility_refinement_kernel
        model_config["utility_refinement_blocks"] = utility_refinement_blocks
    if temporal_uq_refinement:
        model_config["temporal_uq_refinement"] = True
        model_config["temporal_uq_channels"] = temporal_uq_channels
        model_config["temporal_uq_blocks"] = temporal_uq_blocks
        model_config["temporal_uq_kernel"] = temporal_uq_kernel
        model_config["temporal_uq_logit_limit"] = temporal_uq_logit_limit
    if coverage_calibration:
        model_config["coverage_calibration"] = True
        model_config["coverage_calibration_kernel"] = (
            coverage_calibration_kernel)
        model_config["coverage_calibration_blocks"] = (
            coverage_calibration_blocks)
        model_config["coverage_calibration_logit_limit"] = (
            coverage_calibration_logit_limit)
        model_config["coverage_calibration_mode"] = (
            coverage_calibration_mode)
        model_config["coverage_calibration_temperature_limit"] = (
            coverage_calibration_temperature_limit)
    model = construct_model(architecture, model_config).to(device)
    source = payload["model_state_dict"]
    destination = model.state_dict()
    for key, value in source.items():
        if key not in destination:
            raise KeyError(f"checkpoint key missing in expanded model: {key}")
        if destination[key].shape == value.shape:
            destination[key] = value
        elif (add_flow_context and key == "uq_fusion.0.weight"
              and destination[key].ndim == 4
              and destination[key].shape[0] == value.shape[0]
              and destination[key].shape[1] > value.shape[1]
              and destination[key].shape[2:] == value.shape[2:]):
            destination[key].zero_()
            destination[key][:, :value.shape[1]].copy_(value)
        else:
            raise ValueError(
                f"cannot migrate {key}: checkpoint {tuple(value.shape)} "
                f"!= model {tuple(destination[key].shape)}")
    model.load_state_dict(destination, strict=True)
    common.set_model_stats(model, normalizer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [model.uq_fusion, model.quantile_head]
    if model.utility_refinement_head is not None:
        modules.append(model.utility_refinement_head)
    if model.temporal_uq_refiner is not None:
        modules.append(model.temporal_uq_refiner)
    if model.coverage_calibration_head is not None:
        modules.append(model.coverage_calibration_head)
    if model.residual_mixture_head is not None:
        modules.append(model.residual_mixture_head)
    if model.output_space_uq_allocator is not None:
        modules.append(model.output_space_uq_allocator)
    if model.output_space_uq_budgeter is not None:
        modules.append(model.output_space_uq_budgeter)
    if train_projections:
        modules.append(model.uq_projections)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return model


def uq_state_dict(model: EnsembleUNet2d) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(UQ_PREFIXES)
    }


def load_uq_state(model: EnsembleUNet2d,
                  state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)


def trainable_parameters(model: EnsembleUNet2d) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("OOF UQ training has no trainable parameters")
    return parameters


def set_training_mode(args: argparse.Namespace,
                      model: EnsembleUNet2d) -> None:
    if not args.frozen_point_eval_mode:
        model.train()
        return
    model.eval()
    modules = [
        model.uq_projections,
        model.uq_fusion,
        model.quantile_head,
        model.utility_refinement_head,
        model.temporal_uq_refiner,
        model.coverage_calibration_head,
        getattr(model, "residual_mixture_head", None),
        getattr(model, "output_space_uq_allocator", None),
        getattr(model, "output_space_uq_budgeter", None),
    ]
    for module in modules:
        if module is not None:
            module.train()


def preprocess_real_training_batch(args: argparse.Namespace,
                                   history: torch.Tensor,
                                   target: torch.Tensor,
                                   normalizer,
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply training-only observation perturbations before UQ inference."""
    history_input = ensemble_train.add_training_input_gain(
        history, args.real_input_gain_std)
    history_norm, target_norm = normalizer.preprocess(history_input, target)
    history_norm = ensemble_train.masked_velocity_blur(
        history_norm, history, args.real_input_blur_probability,
        args.real_input_blur_strength)
    history_norm = ensemble_train.add_training_input_noise(
        history_norm, history, args.real_input_noise_std)
    return history_norm, target_norm


def make_loader(dataset, indices: list[int], batch_size: int,
                shuffle: bool, args: argparse.Namespace, seed: int):
    if not indices:
        raise ValueError("condition fold produced an empty dataset")
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=True,
        generator=(
            torch.Generator().manual_seed(seed) if shuffle else None))


@torch.inference_mode()
def exact_sps(model: EnsembleUNet2d, loader, normalizer,
              device: torch.device,
              residual_scale: float = 1.0) -> dict[str, float]:
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("residual_scale must be finite and positive")
    numerator = 0.0
    denominator = 0
    covered = 0
    width_sum = 0.0
    model.eval()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _ = normalizer.preprocess(history, target)
        prediction_norm, lower_norm, upper_norm = (
            model.forward_distribution(history_norm))
        prediction = common.to_physical_target(
            prediction_norm, normalizer)[..., :2]
        truth = target[..., :2]
        lower = lower_norm * scale
        upper = upper_norm * scale
        residual = (truth - prediction) * residual_scale
        scored = truth != 0.0
        inside = (residual >= -lower) & (residual <= upper)
        factor = common.sample_sps_factor(prediction, truth).view(
            -1, 1, 1, 1, 1)
        value = factor * inside * torch.exp(
            -(lower + upper) / common.SPS_SIGMA)
        numerator += float(value[scored].sum(dtype=torch.float64))
        denominator += int(scored.sum())
        covered += int((inside & scored).sum())
        width_sum += float(
            (lower + upper)[scored].sum(dtype=torch.float64))
    return {
        "sps_score": 100.0 * numerator / denominator,
        "coverage": covered / denominator,
        "mean_width": width_sum / denominator,
        "scored_elements": denominator,
    }


def aggregate(rows: list[dict], key: str) -> dict[str, float]:
    total = sum(row[key]["scored_elements"] for row in rows)
    if not total:
        raise RuntimeError("cannot aggregate empty fold metrics")
    return {
        metric: sum(
            row[key][metric] * row[key]["scored_elements"]
            for row in rows) / total
        for metric in ("sps_score", "coverage", "mean_width")
    } | {"scored_elements": total}


def quantile_forward(model: EnsembleUNet2d, history_norm: torch.Tensor):
    values = model.forward_uq_training(history_norm)
    (prediction, _members, quantiles, lower, upper, _features,
     coverage_logits) = values
    return prediction, quantiles, (lower, upper), coverage_logits


@torch.inference_mode()
def blended_teacher_extents(
        history_norm: torch.Tensor,
        teachers: tuple[torch.nn.Module, torch.nn.Module],
        weight_b: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    history_half = history_norm.to(dtype=torch.float16)
    _prediction_a, lower_a, upper_a = teachers[0].forward_distribution(
        history_half)
    _prediction_b, lower_b, upper_b = teachers[1].forward_distribution(
        history_half)
    return (
        lower_a.lerp(lower_b, weight_b).float(),
        upper_a.lerp(upper_b, weight_b).float(),
    )


def nearest_forecast_dynamics_peer(
        prediction: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Match each sample to a target-free, dynamically similar batch peer."""
    batch = prediction.shape[0]
    if batch < 2:
        return (torch.zeros(batch, dtype=torch.long, device=prediction.device),
                prediction.new_zeros(()))
    uv = prediction.detach()[..., :2]
    temporal = (uv[:, 1:] - uv[:, :-1]).abs().mean(dim=(1, 2, 3))
    horizontal = (uv[:, :, :, 1:] - uv[:, :, :, :-1]).abs().mean(
        dim=(1, 2, 3))
    vertical = (uv[:, :, 1:] - uv[:, :, :-1]).abs().mean(dim=(1, 2, 3))
    mean = uv.mean(dim=(1, 2, 3))
    standard_deviation = uv.std(dim=(1, 2, 3), unbiased=False)
    features = torch.cat((mean, standard_deviation, temporal,
                          horizontal, vertical), dim=1)
    features = (features - features.mean(dim=0, keepdim=True)) / (
        features.std(dim=0, keepdim=True, unbiased=False) + 1.0e-6)
    distance = torch.cdist(features.float(), features.float()).to(uv.dtype)
    distance.fill_diagonal_(torch.inf)
    peer = distance.argmin(dim=1)
    selected = distance.gather(1, peer[:, None]).mean()
    return peer, selected


def residual_mixture_proper_loss(
        prediction: torch.Tensor,
        mixture: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor, normalizer,
        crps_weight: float = 0.20,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score a conditional Gaussian residual mixture with NLL and CRPS.

    Residuals are normalized by the fixed channel-specific SPS references,
    matching ``EnsembleUNet2d._residual_mixture_from_fused``.  Both terms are
    proper distributional scores: unlike a target-visible oracle endpoint,
    their expectation is minimized by the conditional residual distribution.
    """
    if not math.isfinite(crps_weight) or crps_weight < 0.0:
        raise ValueError("mixture CRPS weight must be finite/non-negative")
    logits, means, scales = mixture
    if not (logits.shape == means.shape == scales.shape):
        raise ValueError("mixture parameter shapes differ")
    if logits.ndim != 6 or logits.shape[-1] < 2:
        raise ValueError(
            "mixture parameters must have shape (B,T,H,W,2,K>=2)")
    prediction_physical = common.to_physical_target(
        prediction, normalizer)[..., :2]
    target_physical = common.to_physical_target(target, normalizer)[..., :2]
    reference = prediction_physical.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2)
    residual = (
        target_physical - prediction_physical.detach()) / reference
    scored = target_physical != 0.0

    # Evaluate proper scores in FP32 even when deployment uses FP16.  Tail
    # likelihood and erf are otherwise unnecessarily fragile near tiny scales.
    residual_f = residual.float()
    logits_f = logits.float()
    means_f = means.float()
    scales_f = scales.float().clamp_min(1.0e-6)
    log_weights = torch.log_softmax(logits_f, dim=-1)
    standardized = (
        residual_f.unsqueeze(-1) - means_f) / scales_f
    component_log_probability = (
        log_weights - torch.log(scales_f)
        - 0.5 * standardized.square()
        - 0.5 * math.log(2.0 * math.pi))
    nll_element = -torch.logsumexp(component_log_probability, dim=-1)
    nll = nll_element[scored].mean()

    def gaussian_absolute_expectation(
            difference: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        z = difference / scale.clamp_min(1.0e-6)
        density = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return 2.0 * scale * density + difference * (2.0 * cdf - 1.0)

    weights = torch.softmax(logits_f, dim=-1)
    observation_term = (
        weights * gaussian_absolute_expectation(
            residual_f.unsqueeze(-1) - means_f, scales_f)
    ).sum(dim=-1)
    pair_difference = means_f.unsqueeze(-1) - means_f.unsqueeze(-2)
    pair_scale = torch.sqrt(
        scales_f.square().unsqueeze(-1)
        + scales_f.square().unsqueeze(-2))
    pair_weight = weights.unsqueeze(-1) * weights.unsqueeze(-2)
    pair_term = (
        pair_weight * gaussian_absolute_expectation(
            pair_difference, pair_scale)
    ).sum(dim=(-1, -2))
    crps_element = observation_term - 0.5 * pair_term
    crps = crps_element[scored].mean()
    loss = nll + crps_weight * crps
    return loss, {
        "mixture_loss": loss.detach(),
        "mixture_nll": nll.detach(),
        "mixture_crps": crps.detach(),
        "mixture_scale_mean": scales_f[scored.unsqueeze(-1).expand_as(
            scales_f)].mean().detach(),
        "mixture_weight_entropy": (
            -(weights * log_weights).sum(dim=-1)[scored].mean().detach()),
    }


def uq_objective(
        args: argparse.Namespace, model: EnsembleUNet2d,
        prediction: torch.Tensor, quantiles: torch.Tensor,
        target: torch.Tensor, normalizer,
        teacher_extents: tuple[torch.Tensor, torch.Tensor] | None = None,
        interval_extents: tuple[torch.Tensor, torch.Tensor] | None = None,
        coverage_logits: torch.Tensor | None = None,
        residual_physical_override: torch.Tensor | None = None,
        point_factor_override: torch.Tensor | None = None,
        scored_override: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    lower_norm, upper_norm = (
        interval_extents
        if interval_extents is not None
        else model._utility_interval(quantiles))
    prediction_physical = common.to_physical_target(
        prediction, normalizer)[..., :2]
    target_physical = common.to_physical_target(
        target, normalizer)[..., :2]
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower = lower_norm * scale
    upper = upper_norm * scale
    override_active = residual_physical_override is not None
    if override_active != (point_factor_override is not None):
        raise ValueError(
            "OOF residual and point-factor overrides must be supplied together")
    if override_active:
        residual = residual_physical_override.to(
            device=prediction.device, dtype=prediction.dtype)
        factor = point_factor_override.to(
            device=prediction.device, dtype=prediction.dtype)
        if (residual.shape != prediction[..., :2].shape
                or factor.shape != (prediction.shape[0],)
                or not bool(torch.isfinite(residual).all())
                or not bool(torch.isfinite(factor).all())):
            raise ValueError("OOF residual/factor override is invalid")
        # Pinball operates in normalized residual coordinates.  Construct only
        # the algebraic target needed by that proper loss; the physical truth
        # remains the source of the released-data scored mask below.
        quantile_target = target.clone()
        quantile_target[..., :2] = (
            prediction[..., :2].detach() + residual / scale)
    else:
        residual = target_physical - prediction_physical.detach()
        factor = common.sample_sps_factor(
            prediction_physical.detach(), target_physical).detach()
        quantile_target = target
    if scored_override is None:
        scored = target_physical != 0.0
    else:
        scored = scored_override.to(device=prediction.device, dtype=torch.bool)
        if scored.shape != prediction[..., :2].shape:
            raise ValueError("OOF scored-mask override is not aligned")
    pinball = ensemble_train.residual_quantile_loss(
        model, prediction, quantiles, quantile_target, normalizer,
        physical_target=target_physical)
    reference = lower.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2)
    tau = args.soft_tau_fraction * reference
    soft_inside = (
        torch.sigmoid((residual + lower) / tau)
        * torch.sigmoid((upper - residual) / tau))
    efficiency_element = soft_inside * torch.exp(
        -(lower + upper) / common.SPS_SIGMA)
    count = scored.sum(dim=(1, 2, 3, 4)).clamp_min(1)
    efficiency = (
        (efficiency_element * scored).sum(dim=(1, 2, 3, 4)) / count)
    soft_official = factor * efficiency
    worst_count = max(
        1, math.ceil(efficiency.numel() * args.cvar_fraction))
    worst_efficiency = torch.topk(
        efficiency, k=worst_count, largest=False).values.mean()

    # Optional antithetic residual stress.  This is deliberately not an
    # oracle endpoint target: one predicted interval is scored against the
    # observed residual and nearby plausible residuals.  The zero-mean pair
    # protects against small measurement/domain shifts, while a mild outward
    # scale guards the tail.  Pinball and interval score below remain proper
    # distributional anchors, preventing this proxy from widening without
    # limit or memorizing per-label endpoints.
    robust_sps_weight = float(getattr(args, "robust_sps_weight", 0.0))
    robust_jitter_fraction = float(getattr(
        args, "robust_residual_jitter_fraction", 0.0))
    robust_tail_scale = float(getattr(args, "robust_residual_tail_scale", 1.0))
    if (robust_sps_weight < 0.0 or robust_jitter_fraction < 0.0
            or robust_tail_scale < 1.0):
        raise ValueError("robust SPS controls are outside their valid range")
    robust_expected = lower.new_zeros(())
    robust_worst = lower.new_zeros(())
    if robust_sps_weight > 0.0:
        if robust_jitter_fraction <= 0.0:
            raise ValueError("robust SPS requires positive residual jitter")
        jitter = robust_jitter_fraction * reference
        stressed_residual = torch.stack((
            residual + jitter,
            residual - jitter,
            residual * robust_tail_scale,
        ), dim=0)
        stressed_inside = (
            torch.sigmoid(
                (stressed_residual + lower.unsqueeze(0)) / tau.unsqueeze(0))
            * torch.sigmoid(
                (upper.unsqueeze(0) - stressed_residual) / tau.unsqueeze(0)))
        stressed_element = stressed_inside * torch.exp(
            -(lower + upper).unsqueeze(0) / common.SPS_SIGMA)
        stressed_efficiency = (
            (stressed_element * scored.unsqueeze(0)).sum(
                dim=(2, 3, 4, 5)) / count.unsqueeze(0))
        stressed_official = factor.unsqueeze(0) * stressed_efficiency
        robust_expected = (
            0.25 * stressed_official[0]
            + 0.25 * stressed_official[1]
            + 0.50 * stressed_official[2]
        ).mean()
        robust_worst = stressed_official.amin(dim=0).mean()

    # Optional distribution-level stress using another released-data sample.
    # A target-free dynamics vector selects the nearest peer; only its residual
    # *difference* is mixed in, in a symmetric pair around the observed
    # residual.  This retains spatial/wake/temporal correlations that fixed
    # pixel noise cannot represent and avoids treating a single target-visible
    # minimal interval as a deployable label.
    peer_sps_weight = float(getattr(args, "peer_sps_weight", 0.0))
    peer_mix_fraction = float(getattr(
        args, "peer_residual_mix_fraction", 0.0))
    if (peer_sps_weight < 0.0 or not 0.0 <= peer_mix_fraction <= 1.0):
        raise ValueError("peer SPS controls are outside their valid range")
    peer_expected = lower.new_zeros(())
    peer_worst = lower.new_zeros(())
    peer_distance = lower.new_zeros(())
    if peer_sps_weight > 0.0:
        if peer_mix_fraction <= 0.0:
            raise ValueError("peer SPS requires a positive residual mix")
        peer, peer_distance = nearest_forecast_dynamics_peer(
            prediction_physical)
        if residual.shape[0] > 1:
            peer_valid = scored & scored[peer]
            difference = torch.where(
                peer_valid, residual[peer] - residual, torch.zeros_like(residual))
            offset = peer_mix_fraction * difference
            peer_residual = torch.stack((residual + offset, residual - offset))
            peer_inside = (
                torch.sigmoid(
                    (peer_residual + lower.unsqueeze(0)) / tau.unsqueeze(0))
                * torch.sigmoid(
                    (upper.unsqueeze(0) - peer_residual) / tau.unsqueeze(0)))
            peer_element = peer_inside * torch.exp(
                -(lower + upper).unsqueeze(0) / common.SPS_SIGMA)
            peer_efficiency = (
                (peer_element * scored.unsqueeze(0)).sum(dim=(2, 3, 4, 5))
                / count.unsqueeze(0))
            peer_official = factor.unsqueeze(0) * peer_efficiency
            peer_expected = peer_official.mean()
            peer_worst = peer_official.amin(dim=0).mean()

    alpha = 1.0 - args.coverage_target
    miss_lower = (-lower - residual).clamp_min(0.0)
    miss_upper = (residual - upper).clamp_min(0.0)
    interval_score = (
        lower + upper + (2.0 / alpha) * (miss_lower + miss_upper)
    ) / common.SPS_SIGMA
    interval_score = interval_score[scored].mean()
    extent_distill = lower.new_zeros(())
    if teacher_extents is not None:
        teacher_lower, teacher_upper = teacher_extents
        teacher_lower = teacher_lower * scale
        teacher_upper = teacher_upper * scale
        extent_distill = 0.5 * (
            F.smooth_l1_loss(lower / reference, teacher_lower / reference)
            + F.smooth_l1_loss(
                upper / reference, teacher_upper / reference))
    coverage_bce = lower.new_zeros(())
    if coverage_logits is not None:
        candidate_lower, candidate_upper, candidate_valid, _masses = (
            model._interval_candidates(quantiles))
        residual_normalized = residual / scale
        inside_candidates = (
            residual_normalized.unsqueeze(-1) >= -candidate_lower
        ) & (
            residual_normalized.unsqueeze(-1) <= candidate_upper)
        candidate_mask = (
            scored.unsqueeze(-1).expand_as(inside_candidates)
            & candidate_valid)
        coverage_bce = F.binary_cross_entropy_with_logits(
            coverage_logits[candidate_mask],
            inside_candidates[candidate_mask].to(coverage_logits.dtype))
    loss = (
        args.pinball_weight * pinball
        - args.soft_sps_weight * soft_official.mean()
        - args.cvar_weight * worst_efficiency
        - robust_sps_weight * (
            0.75 * robust_expected + 0.25 * robust_worst)
        - peer_sps_weight * (
            0.75 * peer_expected + 0.25 * peer_worst)
        + args.interval_score_weight * interval_score
        + args.extent_distill_weight * extent_distill
        + args.coverage_bce_weight * coverage_bce)
    return loss, {
        "loss": loss.detach(),
        "pinball": pinball.detach(),
        "soft_official": soft_official.mean().detach(),
        "worst_efficiency": worst_efficiency.detach(),
        "robust_expected": robust_expected.detach(),
        "robust_worst": robust_worst.detach(),
        "peer_expected": peer_expected.detach(),
        "peer_worst": peer_worst.detach(),
        "peer_distance": peer_distance.detach(),
        "interval_score": interval_score.detach(),
        "extent_distill": extent_distill.detach(),
        "coverage_bce": coverage_bce.detach(),
    }


def sim_allocation_objective(
        args: argparse.Namespace, model: EnsembleUNet2d,
        prediction: torch.Tensor, quantiles: torch.Tensor,
        target: torch.Tensor, normalizer, scored: torch.Tensor,
        interval_extents: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Transfer only Sim spatial residual allocation, not absolute width.

    Both log maps are centered independently for every sample, horizon and
    velocity channel. A uniform interval multiplier therefore has zero effect
    on this objective (apart from an inactive numerical floor), leaving Real
    OOF residuals to determine interval scale and asymmetry.
    """
    lower_norm, upper_norm = (
        interval_extents
        if interval_extents is not None
        else model._utility_interval(quantiles))
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    width = (lower_norm + upper_norm) * scale
    prediction_physical = common.to_physical_target(
        prediction, normalizer)[..., :2]
    target_physical = common.to_physical_target(
        target, normalizer)[..., :2]
    residual = (target_physical - prediction_physical.detach()).abs()
    scored = scored.to(device=width.device, dtype=torch.bool)
    reference = width.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2)
    floor = args.sim_allocation_floor_fraction * reference
    log_width = torch.log(width.clamp_min(floor))
    log_residual = torch.log(residual.clamp_min(floor))
    mask = scored.to(dtype=width.dtype)
    count = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    width_center = (log_width * mask).sum(
        dim=(2, 3), keepdim=True) / count
    residual_center = (log_residual * mask).sum(
        dim=(2, 3), keepdim=True) / count
    allocation = F.smooth_l1_loss(
        log_width - width_center,
        (log_residual - residual_center).detach(), reduction="none")
    loss = allocation[scored].mean()
    return loss, {
        "loss": loss.detach(),
        "log_width_std": (
            ((log_width - width_center).square() * mask).sum()
            / mask.sum().clamp_min(1.0)).sqrt().detach(),
        "log_residual_std": (
            ((log_residual - residual_center).square() * mask).sum()
            / mask.sum().clamp_min(1.0)).sqrt().detach(),
    }

def fit_fold(
        args: argparse.Namespace, payload: dict, normalizer,
        train_loader, eval_loader, device: torch.device, fold: int,
        extent_teachers: tuple[torch.nn.Module, torch.nn.Module] | None = None,
        sim_loader=None, sim_normalizer=None,
        select_best: bool = True,
        ) -> tuple[EnsembleUNet2d, dict]:
    seed_everything(args.seed + fold + 1)
    model = load_model(
        payload, normalizer, device, args.train_projections,
        args.add_flow_context, args.residual_quantile_intervals,
        args.utility_refinement, args.utility_refinement_log_limit,
        args.utility_refinement_kernel, args.utility_refinement_blocks,
        args.temporal_uq_refinement, args.temporal_uq_channels,
        args.temporal_uq_blocks, args.temporal_uq_kernel,
        args.temporal_uq_logit_limit,
        args.coverage_calibration, args.coverage_calibration_kernel,
        args.coverage_calibration_blocks,
        args.coverage_calibration_logit_limit,
        args.coverage_calibration_mode,
        args.coverage_calibration_temperature_limit)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.fold_updates)
    iterator = common.repro.cycling(train_loader)
    if (sim_loader is None) != (sim_normalizer is None):
        raise ValueError("sim_loader and sim_normalizer must be supplied together")
    if (args.sim_rehearsal_weight > 0) != (sim_loader is not None):
        raise ValueError(
            "positive Sim rehearsal weight must match prepared Sim data")
    sim_iterator = (
        common.repro.cycling(sim_loader) if sim_loader is not None else None)
    baseline = exact_sps(model, eval_loader, normalizer, device)
    best_metrics = baseline
    final_metrics = baseline
    best_iteration = 0
    best_state = uq_state_dict(model)
    history = []
    sums: dict[str, float] = {}
    count = 0
    started = time.time()

    for iteration in range(1, args.fold_updates + 1):
        set_training_mode(args, model)
        history_batch, target = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = preprocess_real_training_batch(
            args, history_batch, target, normalizer)
        optimizer.zero_grad(set_to_none=True)
        (prediction, quantiles, interval_extents,
         coverage_logits) = quantile_forward(model, history_norm)
        teacher_extents = None
        if extent_teachers is not None:
            teacher_extents = blended_teacher_extents(
                history_norm, extent_teachers,
                args.extent_teacher_weight_b)
        real_loss, real_components = uq_objective(
            args, model, prediction, quantiles, target_norm, normalizer,
            teacher_extents, interval_extents, coverage_logits)
        loss = real_loss
        components = real_components
        if sim_iterator is not None:
            sim_history, sim_target = next(sim_iterator)
            sim_history = sim_history.to(device, non_blocking=True)
            sim_target = sim_target.to(device, non_blocking=True)
            sim_history_norm, sim_target_norm = sim_normalizer.preprocess(
                sim_history, sim_target)
            if args.zero_sim_pressure:
                sim_history_norm[..., 2] = 0.0
                sim_target_norm[..., 2] = 0.0
            (sim_prediction, sim_quantiles, sim_interval_extents,
             sim_coverage_logits) = (
                quantile_forward(model, sim_history_norm))
            if args.sim_rehearsal_mode == "official":
                sim_loss, sim_components = uq_objective(
                    args, model, sim_prediction, sim_quantiles,
                    sim_target_norm, normalizer,
                    interval_extents=sim_interval_extents,
                    coverage_logits=sim_coverage_logits)
            else:
                sim_loss, sim_components = sim_allocation_objective(
                    args, model, sim_prediction, sim_quantiles,
                    sim_target_norm, normalizer,
                    scored=sim_target[..., :2] != 0.0,
                    interval_extents=sim_interval_extents)
            weight = args.sim_rehearsal_weight
            loss = (real_loss + weight * sim_loss) / (1.0 + weight)
            components = {
                "loss": loss.detach(),
                **{f"real_{key}": value
                   for key, value in real_components.items()},
                **{f"sim_{key}": value
                   for key, value in sim_components.items()},
            }
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite fold {fold} UQ loss at {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        scheduler.step()
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

        if (iteration % args.eval_interval == 0
                or iteration == args.fold_updates):
            metrics = exact_sps(
                model, eval_loader, normalizer, device)
            final_metrics = metrics
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                **{f"eval_{key}": value
                   for key, value in metrics.items()},
            }
            if metrics["sps_score"] > best_metrics["sps_score"]:
                best_metrics = metrics
                best_iteration = iteration
                best_state = uq_state_dict(model)
            history.append(row)
            print(json.dumps({
                "event": "oof_uq_eval", "fold": fold, **row,
            }), flush=True)
            sums.clear()
            count = 0

    if select_best:
        load_uq_state(model, best_state)
        selected_metrics = exact_sps(
            model, eval_loader, normalizer, device)
    else:
        selected_metrics = final_metrics
    return model, {
        "baseline": baseline,
        "final": final_metrics,
        "selected": selected_metrics,
        "best_iteration": best_iteration,
        "history": history,
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters),
    }


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    payload = torch.load(
        str(args.point_checkpoint), map_location="cpu", weights_only=False,
        mmap=True)
    if payload.get("phase") != "real":
        raise ValueError("OOF UQ training requires a Real-phase checkpoint")
    trajectories, metadata = common.repro.load_trajectories(
        args.real_dir, args.max_files)
    _point_windows, tail_windows, manifest = common.repro.make_split(
        metadata, args.window_stride, args.val_fraction)
    dataset = common.repro.WindowDataset(trajectories, tail_windows)
    sample_folds, fold_manifest = trajectory_folds(
        manifest, args.folds)
    if len(sample_folds) != len(dataset):
        raise RuntimeError("fold labels do not align with tail windows")
    normalizer = common.repro.Normalizer(args.real_stats, device)
    sim_dataset = None
    sim_normalizer = None
    sim_manifest = None
    sim_stats_source = None
    if args.sim_rehearsal_weight > 0:
        sim_trajectories, sim_windows, sim_manifest = (
            sim_train.load_simulations(
                args.sim_dir, args.sim_max_files,
                args.sim_window_stride, True))
        if args.sim_stats is not None:
            sim_stats = torch.load(
                args.sim_stats, map_location="cpu", weights_only=False)
            sim_stats_source = str(args.sim_stats)
        else:
            sim_stats = sim_train.window_statistics(
                sim_trajectories, sim_windows)
            sim_stats_source = "computed from rehearsal windows"
        if not isinstance(sim_stats, (tuple, list)) or len(sim_stats) != 4:
            raise ValueError("Sim statistics must contain mean/std for x and y")
        sim_normalizer = sim_train.ChannelNormalizer(
            *sim_stats, device=device)
        sim_dataset = sim_train.SimWindowDataset(
            sim_trajectories, sim_windows,
            mask_probability=0.0, noise_scale=0.0)
        torch.save(tuple(value.detach().cpu() for value in sim_stats),
                   args.output_dir / "sim_rehearsal_stats.pt")
        (args.output_dir / "sim_rehearsal_manifest.json").write_text(
            json.dumps(sim_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps({
            "event": "sim_rehearsal_dataset",
            "trajectories": len(sim_trajectories),
            "windows": len(sim_dataset),
            "weight": args.sim_rehearsal_weight,
            "mode": args.sim_rehearsal_mode,
            "stats_source": sim_stats_source,
            "zero_pressure": args.zero_sim_pressure,
        }), flush=True)
    extent_teachers = None
    if args.extent_distill_weight > 0:
        teacher_a, _teacher_a_metadata = load_forecaster(
            args.extent_teacher_a, normalizer, device)
        teacher_b, _teacher_b_metadata = load_forecaster(
            args.extent_teacher_b, normalizer, device)
        extent_teachers = (teacher_a, teacher_b)
    active_folds = list(range(args.folds))
    if args.fold_limit:
        active_folds = active_folds[:args.fold_limit]
    selection_contract = fold_selection_contract(args.fixed_terminal_folds)
    fold_rows = []
    all_indices = list(range(len(dataset)))

    for fold in active_folds:
        train_indices = [
            index for index in all_indices
            if sample_folds[index] != fold]
        eval_indices = [
            index for index in all_indices
            if sample_folds[index] == fold]
        train_loader = make_loader(
            dataset, train_indices, args.batch_size, True,
            args, args.seed + fold + 1)
        eval_loader = make_loader(
            dataset, eval_indices, args.eval_batch_size, False,
            args, args.seed)
        sim_loader = None
        if sim_dataset is not None:
            sim_loader = make_loader(
                sim_dataset, list(range(len(sim_dataset))),
                args.sim_batch_size or args.batch_size, True,
                args, args.seed + 10_000 + fold)
        model, row = fit_fold(
            args, payload, normalizer, train_loader,
            eval_loader, device, fold,
            extent_teachers=extent_teachers,
            sim_loader=sim_loader, sim_normalizer=sim_normalizer,
            select_best=selection_contract["select_best"])
        row.update({
            "fold": fold,
            "train_windows": len(train_indices),
            "eval_windows": len(eval_indices),
            "selection_rule": selection_contract["selection_rule"],
            "selected_iteration": (
                args.fold_updates if args.fixed_terminal_folds
                else row["best_iteration"]),
            "held_fold_checkpoint_selection": (
                selection_contract["held_fold_checkpoint_selection"]),
            "selected_delta": (
                row["selected"]["sps_score"]
                - row["baseline"]["sps_score"]),
            "final_delta": (
                row["final"]["sps_score"]
                - row["baseline"]["sps_score"]),
        })
        torch.save({
            "architecture": "ensemble_unet_uq_oof",
            "base_checkpoint": str(args.point_checkpoint),
            "fold": fold,
            "model_config": model.architecture_config(),
            "uq_state_dict": uq_state_dict(model),
            "selection_rule": row["selection_rule"],
            "selected_iteration": row["selected_iteration"],
            "held_fold_checkpoint_selection": (
                row["held_fold_checkpoint_selection"]),
            "metrics": row,
        }, args.output_dir / f"fold_{fold:02d}_best.pth")
        fold_rows.append(row)
        print(json.dumps({
            "event": "oof_fold_complete", **row,
        }), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "point_checkpoint": str(args.point_checkpoint),
        "folds_requested": args.folds,
        "folds_completed": len(fold_rows),
        "tail_windows": len(dataset),
        "sim_rehearsal_windows": (
            len(sim_dataset) if sim_dataset is not None else 0),
        "sim_rehearsal_stats_source": sim_stats_source,
        "fold_manifest": fold_manifest,
        "folds": fold_rows,
        "fold_selection_rule": selection_contract["selection_rule"],
        "held_fold_checkpoint_selection": (
            selection_contract["held_fold_checkpoint_selection"]),
        "baseline_oof": aggregate(fold_rows, "baseline"),
        "selected_oof": aggregate(fold_rows, "selected"),
        "final_oof": aggregate(fold_rows, "final"),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    summary["selected_delta"] = (
        summary["selected_oof"]["sps_score"]
        - summary["baseline_oof"]["sps_score"])
    summary["final_delta"] = (
        summary["final_oof"]["sps_score"]
        - summary["baseline_oof"]["sps_score"])

    if args.fit_final:
        final_args = copy.copy(args)
        final_args.fold_updates = args.final_updates
        final_train_loader = make_loader(
            dataset, all_indices, args.batch_size, True,
            args, args.seed + args.folds + 1)
        final_eval_loader = make_loader(
            dataset, all_indices, args.eval_batch_size, False,
            args, args.seed)
        final_sim_loader = None
        if sim_dataset is not None:
            final_sim_loader = make_loader(
                sim_dataset, list(range(len(sim_dataset))),
                args.sim_batch_size or args.batch_size, True,
                args, args.seed + 20_000)
        final_model, final_refit = fit_fold(
            final_args, payload, normalizer, final_train_loader,
            final_eval_loader, device, fold=-1,
            extent_teachers=extent_teachers,
            sim_loader=final_sim_loader, sim_normalizer=sim_normalizer,
            select_best=False)
        final_refit.update({
            "train_windows": len(dataset),
            "selected_iteration": args.final_updates,
            "selection_rule": "fixed final iteration from OOF protocol",
            "evaluation_scope": (
                "all-tail training diagnostic; not an OOF estimate"),
        })
        torch.save({
            "model_state_dict": common.checkpoint_state(final_model),
            "model_config": final_model.architecture_config(),
            "phase": "real",
            "iteration": args.final_updates,
            "history": final_refit["history"],
            "normalizer": common.normalizer_dict(normalizer),
            "metrics": final_refit["final"],
            "training_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "uq_refit": {
                "source": "all temporal-tail residuals",
                "evaluation_scope": final_refit["evaluation_scope"],
                "oof_selected_delta": summary["selected_delta"],
            },
        }, args.output_dir / "all_tail_refit.pth")
        summary["final_refit"] = final_refit
        del final_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "oof_complete",
        "baseline_oof": summary["baseline_oof"],
        "selected_oof": summary["selected_oof"],
        "selected_delta": summary["selected_delta"],
        "final_oof": summary["final_oof"],
        "final_delta": summary["final_delta"],
        "fit_final": args.fit_final,
    }), flush=True)


if __name__ == "__main__":
    main()
