#!/usr/bin/env python3
"""Train the high-frequency-aware HybridFNO2d on released RealPDE data.

The ``sim`` phase always starts from random initialization.  The ``real``
phase loads a checkpoint produced by the sim phase, swaps to the released real
normalizer, and evaluates on the same non-overlapping temporal-tail split used
by the existing reproduction work.  No released model checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.hybrid_fno2d import HybridFNO2d, HybridFNO2dConfig


SPS_SIGMA = 0.0563870259


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repro = load_script(
    "realpde_reproduce_hybrid", "code/training/reproduce_finetune.py")
scratch = load_script(
    "realpde_scratch_hybrid", "code/training/train_fno_sim_from_scratch.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sim", "real"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Debug only; 0 uses the complete released set.")
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes-y", type=int, default=14)
    parser.add_argument("--modes-x", type=int, default=24)
    parser.add_argument("--local-kernel", type=int, default=3)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument(
        "--use-local-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-velocity-differences", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--predict-uncertainty", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument("--uncertainty-hidden", type=int, default=24)
    parser.add_argument("--temporal-decoder-width", type=int, default=0)
    parser.add_argument("--temporal-decoder-layers", type=int, default=3)
    parser.add_argument("--temporal-decoder-groups", type=int, default=4)
    parser.add_argument(
        "--predict-energy-gain", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument("--energy-gain-hidden", type=int, default=16)

    parser.add_argument(
        "--sim-loss-mode", choices=("mse", "official_accuracy"), default="mse")
    parser.add_argument("--sim-mse-weight", type=float, default=0.25)
    parser.add_argument(
        "--loss-mode", choices=("mse", "official_accuracy"), default="mse")
    parser.add_argument(
        "--mse-warmup-updates", type=int, default=0,
        help=("In a real-phase official_accuracy run, use plain MSE for this "
              "many initial updates before switching to the requested loss."))
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--spectrum-weight", type=float, default=0.0)
    parser.add_argument(
        "--component-tke-weight", type=float, default=0.0,
        help=("Weight for a per-u/v temporal-variance field objective.  This "
              "prevents the summed official TKE term from hiding errors that "
              "cancel between velocity components."))
    parser.add_argument(
        "--component-tke-decay-updates", type=int, default=0,
        help=("Cosine-decay the component-TKE weight to zero over this many "
              "updates; 0 keeps it constant."))
    parser.add_argument(
        "--interval-loss-mode", choices=("laplace", "soft_sps"),
        default="soft_sps")
    parser.add_argument("--interval-loss-weight", type=float, default=0.05)
    parser.add_argument("--interval-nll-weight", type=float, default=0.10)
    parser.add_argument(
        "--interval-detach-mean", action=argparse.BooleanOptionalAction,
        default=True,
        help=("Stop interval residual gradients at the mean decoder while "
              "still training the shared latent through predicted widths."))
    parser.add_argument(
        "--teacher-checkpoint", type=Path,
        help="Optional frozen official FNO teacher for real-phase distillation.")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--accuracy-weights", type=float, nargs=3,
        metavar=("REL", "TKE", "MVPE"), default=(1.0, 1.0, 1.0))
    parser.add_argument(
        "--selection-metric",
        choices=("rmse", "accuracy_score_mean", "weighted_accuracy_score",
                 "point_score", "tke_rel_l2_score"),
        default="point_score")
    return parser.parse_args()


def architecture_from_args(args: argparse.Namespace) -> HybridFNO2dConfig:
    return HybridFNO2dConfig(
        width=args.width, layers=args.layers, modes_y=args.modes_y,
        modes_x=args.modes_x, local_kernel=args.local_kernel,
        groups=args.groups, expansion=args.expansion, padding=args.padding,
        use_local_branch=args.use_local_branch,
        use_velocity_differences=args.use_velocity_differences,
        predict_uncertainty=args.predict_uncertainty,
        uncertainty_hidden=args.uncertainty_hidden,
        temporal_decoder_width=args.temporal_decoder_width,
        temporal_decoder_layers=args.temporal_decoder_layers,
        temporal_decoder_groups=args.temporal_decoder_groups,
        predict_energy_gain=args.predict_energy_gain,
        energy_gain_hidden=args.energy_gain_hidden)


def checkpoint_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def save_checkpoint(path: Path, model: HybridFNO2d, args: argparse.Namespace,
                    iteration: int, history: list[dict],
                    normalizer_stats: dict[str, list[float]],
                    metrics: dict | None = None) -> None:
    torch.save({
        "model_state_dict": checkpoint_state(model),
        "model_config": model.architecture_config(),
        "phase": args.phase,
        "iteration": iteration,
        "history": history,
        "normalizer": normalizer_stats,
        "metrics": metrics,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }, path)


def normalizer_dict(normalizer) -> dict[str, list[float]]:
    return {
        key: getattr(normalizer, key).detach().cpu().tolist()
        for key in ("mean_input", "mean_target", "std_input", "std_target")
    }


def set_model_stats(model: HybridFNO2d, normalizer) -> None:
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)


def build_or_load_model(args: argparse.Namespace,
                        device: torch.device) -> HybridFNO2d:
    if args.start_checkpoint is None:
        if args.phase != "sim":
            raise ValueError("real phase requires --start-checkpoint from sim training")
        config = architecture_from_args(args)
        return HybridFNO2d(config).to(device)
    if args.phase != "real":
        raise ValueError("--start-checkpoint is only valid for the real phase")
    payload = torch.load(args.start_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("phase") != "sim":
        raise ValueError("real phase must start from a sim-phase checkpoint")
    config = HybridFNO2dConfig(**payload["model_config"])
    requested = architecture_from_args(args)
    if requested != config:
        raise ValueError(
            "architecture flags do not match the sim checkpoint: "
            f"requested={requested.to_dict()} checkpoint={config.to_dict()}")
    model = HybridFNO2d(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def validate_args(args: argparse.Namespace) -> tuple[float, float, float]:
    if (args.updates <= 0 or args.batch_size <= 0 or args.eval_interval <= 0
            or args.checkpoint_interval <= 0 or args.lr <= 0
            or args.weight_decay < 0 or args.grad_clip < 0
            or args.window_stride <= 0
            or args.component_tke_decay_updates < 0):
        raise ValueError("training arguments must be positive and finite")
    if not all(math.isfinite(value) for value in (
            args.lr, args.weight_decay, args.grad_clip,
            args.sim_mse_weight, args.mse_weight, args.spectrum_weight,
            args.component_tke_weight,
            args.interval_loss_weight, args.interval_nll_weight,
            args.distill_weight)):
        raise ValueError("loss/optimizer arguments must be finite")
    if (args.sim_mse_weight < 0 or args.mse_weight < 0
            or args.spectrum_weight < 0 or args.component_tke_weight < 0
            or args.interval_loss_weight < 0
            or args.interval_nll_weight < 0 or args.distill_weight < 0):
        raise ValueError("loss weights must be non-negative")
    if (args.teacher_checkpoint is None) != (args.distill_weight == 0.0):
        raise ValueError(
            "teacher checkpoint and positive distill weight must be used together")
    if args.phase == "sim" and args.teacher_checkpoint is not None:
        raise ValueError("teacher distillation is only supported in the real phase")
    if args.interval_loss_weight and not args.predict_uncertainty:
        # Keep point-only commands concise: the default interval weight is
        # dormant unless the architecture actually contains the native head.
        args.interval_loss_weight = 0.0
    if not 0 <= args.mse_warmup_updates <= args.updates:
        raise ValueError("mse warmup must be between zero and total updates")
    if args.loss_mode == "mse" and args.mse_warmup_updates:
        raise ValueError("mse warmup is only meaningful for official_accuracy")
    weights = tuple(float(value) for value in args.accuracy_weights)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("accuracy weights must be finite and non-negative")
    if not any(value > 0 for value in weights):
        raise ValueError("at least one accuracy weight must be positive")
    architecture_from_args(args).validate()
    return weights


def prepare_sim(args: argparse.Namespace, device: torch.device):
    trajectories, windows, manifest = scratch.load_simulations(
        args.sim_dir, args.max_files, args.window_stride, True)
    scratch_stats = scratch.window_statistics(trajectories, windows)
    dataset = scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    normalizer = scratch.ChannelNormalizer(*scratch_stats, device=device)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return loader, normalizer, len(windows), len(trajectories)


def prepare_real(args: argparse.Namespace, device: torch.device):
    trajectories, metadata = repro.load_trajectories(args.real_dir, args.max_files)
    train_windows, val_windows, manifest = repro.make_split(
        metadata, args.window_stride, args.val_fraction)
    optimization_windows = (
        train_windows + val_windows
        if getattr(args, "real_all_windows", False) else train_windows)
    train_dataset = repro.WindowDataset(trajectories, optimization_windows)
    val_dataset = repro.WindowDataset(trajectories, val_windows)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    normalizer = repro.Normalizer(args.real_stats, device)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return (train_loader, val_loader, normalizer, len(optimization_windows),
            len(val_windows), len(trajectories))


def add_weighted(metrics: dict[str, float],
                 weights: tuple[float, float, float]) -> None:
    repro.add_weighted_accuracy_score(metrics, weights)
    # Preserve the historical normalized 50/30/20 point_score below while
    # also exposing the competition's published, unnormalized point
    # contribution.  Time and SPS are intentionally absent here because a
    # point-only training checkpoint cannot affect or estimate them yet.
    metrics["official_point_contribution"] = (
        0.46 * metrics["rel_l2_score"]
        + 0.10 * metrics["tke_rel_l2_score"]
        + 0.09 * metrics["mvpe_rel_l2_score"])
    metrics["point_score"] = (
        0.5 * metrics["rel_l2_score"]
        + 0.3 * metrics["tke_rel_l2_score"]
        + 0.2 * metrics["mvpe_rel_l2_score"])


def to_physical_target(values: torch.Tensor, normalizer) -> torch.Tensor:
    return values * normalizer.std_target + normalizer.mean_target


def accuracy_objective(
        prediction: torch.Tensor, target: torch.Tensor, normalizer,
        weights: tuple[float, float, float],
        *, batch_mean_errors: bool = False,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable v9 point objective usable with sim or real stats."""
    pred = to_physical_target(prediction, normalizer)[..., :2]
    truth = to_physical_target(target, normalizer)[..., :2]
    batch = pred.shape[0]
    rel = torch.linalg.vector_norm(
        (pred - truth).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        truth.reshape(batch, -1), dim=1).clamp_min(1e-8)

    pred_fluctuation = pred - pred.mean(dim=1, keepdim=True)
    truth_fluctuation = truth - truth.mean(dim=1, keepdim=True)
    pred_tke = 0.5 * pred_fluctuation.square().mean(dim=1).sum(dim=-1)
    truth_tke = 0.5 * truth_fluctuation.square().mean(dim=1).sum(dim=-1)
    tke = torch.linalg.vector_norm(
        (pred_tke - truth_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        truth_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)

    center_y = pred.shape[2] // 2
    probe_y = [center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < pred.shape[2]]
    mvpe_parts = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if not 0 <= probe_x < pred.shape[3]:
            continue
        pred_profile = pred[:, :, probe_y, probe_x].mean(dim=1).reshape(batch, -1)
        truth_profile = truth[:, :, probe_y, probe_x].mean(dim=1).reshape(batch, -1)
        mvpe_parts.append(
            torch.linalg.vector_norm(pred_profile - truth_profile, dim=1)
            / torch.linalg.vector_norm(truth_profile, dim=1).clamp_min(1e-8))
    mvpe = torch.stack(mvpe_parts).mean(dim=0)
    errors = torch.stack((rel, tke, mvpe), dim=1)
    scored_errors = errors.mean(dim=0, keepdim=True) if batch_mean_errors else errors
    scores = 1.0 / (1.0 + 0.5 * scored_errors)
    weight_tensor = scores.new_tensor(weights)
    per_sample = 1.0 - (scores * weight_tensor).sum(dim=1) / weight_tensor.sum()
    return per_sample.mean(), {
        "rel_l2": rel.mean().detach(),
        "tke_rel_l2": tke.mean().detach(),
        "mvpe_rel_l2": mvpe.mean().detach(),
        "accuracy_objective": per_sample.mean().detach(),
    }


def component_tke_objective(
        prediction: torch.Tensor, target: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Relative spatial L2 of separate u/v temporal variance fields.

    The official TKE first sums the u and v variance fields.  This auxiliary
    keeps both components identifiable, so an energy-gain decoder cannot
    improve their sum by moving error from one velocity component to the
    other.
    """
    pred = to_physical_target(prediction, normalizer)[..., :2]
    truth = to_physical_target(target, normalizer)[..., :2]
    pred_variance = (pred - pred.mean(dim=1, keepdim=True)).square().mean(dim=1)
    truth_variance = (
        truth - truth.mean(dim=1, keepdim=True)).square().mean(dim=1)
    batch = pred.shape[0]
    pred_flat = pred_variance.reshape(batch, -1, 2)
    truth_flat = truth_variance.reshape(batch, -1, 2)
    error = torch.linalg.vector_norm(pred_flat - truth_flat, dim=1)
    denominator = torch.linalg.vector_norm(truth_flat, dim=1).clamp_min(1e-8)
    return (error / denominator).mean()


def component_tke_weight(args: argparse.Namespace, iteration: int) -> float:
    """Curriculum weight that smoothly hands control back to official TKE."""
    weight = float(args.component_tke_weight)
    decay = int(args.component_tke_decay_updates)
    if not weight or not decay:
        return weight
    progress = min(max(iteration / decay, 0.0), 1.0)
    return weight * 0.5 * (1.0 + math.cos(math.pi * progress))


def sample_sps_factor(prediction: torch.Tensor,
                      target: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
    """Per-sample point-accuracy factor used by the v9 SPS formula."""
    batch = prediction.shape[0]
    rel = torch.linalg.vector_norm(
        (prediction - target).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        target.reshape(batch, -1), dim=1).clamp_min(1e-8)
    pred_fluctuation = prediction - prediction.mean(dim=1, keepdim=True)
    target_fluctuation = target - target.mean(dim=1, keepdim=True)
    pred_tke = 0.5 * pred_fluctuation.square().mean(dim=1).sum(dim=-1)
    target_tke = 0.5 * target_fluctuation.square().mean(dim=1).sum(dim=-1)
    tke = torch.linalg.vector_norm(
        (pred_tke - target_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        target_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)
    center_y = prediction.shape[2] // 2
    probe_y = [center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < prediction.shape[2]]
    mvpe_parts = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if not 0 <= probe_x < prediction.shape[3]:
            continue
        pred_profile = prediction[:, :, probe_y, probe_x].mean(
            dim=1).reshape(batch, -1)
        target_profile = target[:, :, probe_y, probe_x].mean(
            dim=1).reshape(batch, -1)
        mvpe_parts.append(
            torch.linalg.vector_norm(pred_profile - target_profile, dim=1)
            / torch.linalg.vector_norm(target_profile, dim=1).clamp_min(1e-8))
    mvpe = torch.stack(mvpe_parts).mean(dim=0)
    factor = lambda error: 0.5 / (0.5 + error)
    value = 0.5 * factor(rel) + 0.3 * factor(tke) + 0.2 * factor(mvpe)
    return value.detach() if detach else value


def interval_objective(
        args: argparse.Namespace, prediction: torch.Tensor,
        target: torch.Tensor, lower_norm: torch.Tensor,
        upper_norm: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Asymmetric heteroscedastic loss in physical u/v units."""
    interval_prediction = (
        prediction.detach() if args.interval_detach_mean else prediction)
    pred = to_physical_target(interval_prediction, normalizer)[..., :2]
    truth = to_physical_target(target, normalizer)[..., :2]
    target_std = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower = lower_norm * target_std
    upper = upper_norm * target_std
    residual = truth - pred
    scored = truth != 0.0
    reference = pred.new_tensor([0.01075, 0.00550]).view(1, 1, 1, 1, 2)
    negative = (-residual).clamp_min(0.0) / lower.clamp_min(1e-7)
    positive = residual.clamp_min(0.0) / upper.clamp_min(1e-7)
    normalized_width = ((lower + upper) / (2.0 * reference)).clamp_min(1e-6)
    nll = (negative + positive + torch.log(normalized_width))[scored].mean()
    if args.interval_loss_mode == "laplace":
        loss = nll
        soft_score = nll.new_zeros(())
    else:
        tau = 0.10 * reference
        soft_inside = (
            torch.sigmoid((residual + lower) / tau)
            * torch.sigmoid((upper - residual) / tau))
        factor = sample_sps_factor(pred, truth).view(-1, 1, 1, 1, 1)
        penalty = torch.exp(-(lower + upper) / SPS_SIGMA)
        soft_score = (factor * soft_inside * penalty)[scored].mean()
        loss = -soft_score + args.interval_nll_weight * nll
    return loss, {
        "interval_nll": nll.detach(),
        "interval_soft_sps": soft_score.detach(),
        "interval_mean_width": (lower + upper)[scored].mean().detach(),
    }


def evaluate_native_sps(model: HybridFNO2d, loader: DataLoader,
                        normalizer, device: torch.device) -> dict[str, float]:
    """Stream the exact native-head SPS terms without caching full fields."""
    if not model.config.predict_uncertainty:
        return {}
    numerator = 0.0
    denominator = 0
    covered = 0
    width_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            x_norm, target_norm = normalizer.preprocess(x, target)
            pred_norm, lower_norm, upper_norm = model.forward_distribution(x_norm)
            prediction = to_physical_target(pred_norm, normalizer)[..., :2]
            truth = target[..., :2]
            scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
            lower = lower_norm * scale
            upper = upper_norm * scale
            residual = truth - prediction
            scored = truth != 0.0
            inside = (residual >= -lower) & (residual <= upper)
            factor = sample_sps_factor(prediction, truth).view(-1, 1, 1, 1, 1)
            value = factor * inside * torch.exp(-(lower + upper) / SPS_SIGMA)
            numerator += float(value[scored].sum(dtype=torch.float64))
            denominator += int(scored.sum())
            covered += int((inside & scored).sum())
            width_sum += float((lower + upper)[scored].sum(dtype=torch.float64))
    return {
        "native_sps_score": 100.0 * numerator / denominator,
        "native_sps_coverage": covered / denominator,
        "native_sps_mean_width": width_sum / denominator,
    }


def train_sim(args: argparse.Namespace, model: HybridFNO2d,
              loader: DataLoader, normalizer, device: torch.device,
              dataset_size: int, trajectories: int,
              weights: tuple[float, float, float]) -> dict:
    iterator = repro.cycling(loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    stats = normalizer_dict(normalizer)
    history = []
    interval_losses: list[float] = []
    interval_mses: list[float] = []
    interval_sums: dict[str, float] = {}
    interval_count = 0
    best_loss = float("inf")
    best_iteration = 0
    started = time.time()
    for iteration in range(1, args.updates + 1):
        model.train()
        x, y = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_norm, y_norm = normalizer.preprocess(x, y)
        optimizer.zero_grad(set_to_none=True)
        if model.config.predict_uncertainty:
            prediction, lower_width, upper_width = model.forward_distribution(x_norm)
        else:
            prediction = model(x_norm)
            lower_width = upper_width = None
        mse = torch.mean((prediction - y_norm) ** 2)
        if args.sim_loss_mode == "mse":
            point_loss = mse
            point_components = {}
        else:
            accuracy, point_components = accuracy_objective(
                prediction, y_norm, normalizer, weights)
            point_loss = accuracy + args.sim_mse_weight * mse
        effective_component_weight = component_tke_weight(args, iteration)
        if effective_component_weight:
            component_tke = component_tke_objective(
                prediction, y_norm, normalizer)
            point_loss = point_loss + effective_component_weight * component_tke
            point_components["component_tke"] = component_tke.detach()
            point_components["component_tke_weight"] = prediction.new_tensor(
                effective_component_weight)
        for key, value in point_components.items():
            interval_sums[key] = interval_sums.get(key, 0.0) + float(value)
        loss = point_loss
        if lower_width is not None and args.interval_loss_weight:
            uncertainty_loss, uncertainty_components = interval_objective(
                args, prediction, y_norm, lower_width, upper_width, normalizer)
            loss = loss + args.interval_loss_weight * uncertainty_loss
            for key, value in uncertainty_components.items():
                interval_sums[key] = interval_sums.get(key, 0.0) + float(value)
        interval_count += 1
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite sim loss at {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        interval_losses.append(float(loss.detach()))
        interval_mses.append(float(mse.detach()))
        if iteration % args.eval_interval == 0 or iteration == args.updates:
            row = {
                "iteration": iteration,
                "train_loss_mean": float(np.mean(interval_losses)),
                "train_loss_median": float(np.median(interval_losses)),
                "train_mse_mean": float(np.mean(interval_mses)),
                "train_mse_median": float(np.median(interval_mses)),
                **({f"train_{key}_mean": value / interval_count
                    for key, value in interval_sums.items()}
                   if interval_count else {}),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps({"event": "sim_eval", **row}), flush=True)
            if row["train_loss_median"] < best_loss:
                best_loss = row["train_loss_median"]
                best_iteration = iteration
                save_checkpoint(
                    args.output_dir / "best.pth", model, args, iteration,
                    history, stats, row)
            interval_losses.clear()
            interval_mses.clear()
            interval_sums.clear()
            interval_count = 0
        if iteration % args.checkpoint_interval == 0:
            save_checkpoint(
                args.output_dir / f"model_{iteration:05d}.pth", model, args,
                iteration, history, stats)
    save_checkpoint(
        args.output_dir / "final.pth", model, args, args.updates,
        history, stats, history[-1])
    return {
        "phase": "sim", "updates": args.updates,
        "train_windows": dataset_size, "trajectories": trajectories,
        "best_iteration": best_iteration, "best_train_loss": best_loss,
        "history": history, "wall_seconds": time.time() - started,
    }


def real_loss(args: argparse.Namespace, prediction: torch.Tensor,
              target: torch.Tensor, normalizer,
              weights: tuple[float, float, float],
              iteration: int,
              force_mse: bool = False):
    mse = torch.mean((prediction - target) ** 2)
    components = {"mse": mse.detach()}
    if args.loss_mode == "mse" or force_mse:
        loss = mse
    else:
        accuracy, detail = repro.official_accuracy_objective(
            prediction, target, normalizer, weights)
        loss = accuracy + args.mse_weight * mse
        components.update(detail)
    if args.spectrum_weight:
        spectrum = repro.temporal_spectrum_objective(
            prediction, target, normalizer)
        loss = loss + args.spectrum_weight * spectrum
        components["spectrum_objective"] = spectrum.detach()
    effective_component_weight = component_tke_weight(args, iteration)
    if effective_component_weight:
        component_tke = component_tke_objective(
            prediction, target, normalizer)
        loss = loss + effective_component_weight * component_tke
        components["component_tke"] = component_tke.detach()
        components["component_tke_weight"] = prediction.new_tensor(
            effective_component_weight)
    return loss, components


def train_real(args: argparse.Namespace, model: HybridFNO2d,
               train_loader: DataLoader, val_loader: DataLoader, normalizer,
               device: torch.device, train_windows: int, val_windows: int,
               trajectories: int,
               weights: tuple[float, float, float],
               teacher: torch.nn.Module | None = None) -> dict:
    iterator = repro.cycling(train_loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    stats = normalizer_dict(normalizer)
    initial_metrics = repro.evaluate(model, val_loader, normalizer, device)
    add_weighted(initial_metrics, weights)
    initial_metrics.update(
        evaluate_native_sps(model, val_loader, normalizer, device))
    print(json.dumps({"event": "real_initial", **initial_metrics}), flush=True)
    history = []
    sums: dict[str, float] = {}
    interval_count = 0
    best_value = (float("inf") if args.selection_metric == "rmse"
                  else -float("inf"))
    best_iteration = 0
    started = time.time()
    for iteration in range(1, args.updates + 1):
        model.train()
        x, y = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_norm, y_norm = normalizer.preprocess(x, y)
        optimizer.zero_grad(set_to_none=True)
        if model.config.predict_uncertainty:
            prediction, lower_width, upper_width = model.forward_distribution(x_norm)
        else:
            prediction = model(x_norm)
            lower_width = upper_width = None
        in_mse_warmup = iteration <= args.mse_warmup_updates
        loss, components = real_loss(
            args, prediction, y_norm, normalizer, weights,
            iteration=iteration,
            force_mse=in_mse_warmup)
        if teacher is not None:
            with torch.no_grad():
                teacher_prediction = teacher(x_norm)
            distillation = torch.mean(
                (prediction[..., :2] - teacher_prediction[..., :2]) ** 2)
            loss = loss + args.distill_weight * distillation
            components["distillation_mse"] = distillation.detach()
        if lower_width is not None and args.interval_loss_weight:
            uncertainty_loss, uncertainty_components = interval_objective(
                args, prediction, y_norm, lower_width, upper_width, normalizer)
            loss = loss + args.interval_loss_weight * uncertainty_loss
            components.update(uncertainty_components)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite real loss at {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        sums["loss"] = sums.get("loss", 0.0) + float(loss.detach())
        sums["mse_warmup_fraction"] = (
            sums.get("mse_warmup_fraction", 0.0) + float(in_mse_warmup))
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        interval_count += 1
        if iteration % args.eval_interval == 0 or iteration == args.updates:
            metrics = repro.evaluate(model, val_loader, normalizer, device)
            add_weighted(metrics, weights)
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / interval_count
                   for key, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                **metrics,
            }
            history.append(row)
            print(json.dumps({"event": "real_eval", **row}), flush=True)
            selected = metrics[args.selection_metric]
            improved = (selected < best_value if args.selection_metric == "rmse"
                        else selected > best_value)
            if improved:
                best_value = selected
                best_iteration = iteration
                save_checkpoint(
                    args.output_dir / "best.pth", model, args, iteration,
                    history, stats, metrics)
            sums.clear()
            interval_count = 0
        if iteration % args.checkpoint_interval == 0:
            save_checkpoint(
                args.output_dir / f"model_{iteration:05d}.pth", model, args,
                iteration, history, stats)
    final_metrics = repro.evaluate(model, val_loader, normalizer, device)
    add_weighted(final_metrics, weights)
    final_metrics.update(
        evaluate_native_sps(model, val_loader, normalizer, device))
    save_checkpoint(
        args.output_dir / "final.pth", model, args, args.updates,
        history, stats, final_metrics)
    best_payload = torch.load(
        args.output_dir / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"])
    best_metrics = repro.evaluate(model, val_loader, normalizer, device)
    add_weighted(best_metrics, weights)
    best_metrics.update(
        evaluate_native_sps(model, val_loader, normalizer, device))
    return {
        "phase": "real", "updates": args.updates,
        "train_windows": train_windows, "val_windows": val_windows,
        "trajectories": trajectories, "best_iteration": best_iteration,
        "selection_metric": args.selection_metric,
        "best_metric_value": best_value,
        "metrics": {"sim_pretrain": initial_metrics,
                    "real_best": best_metrics, "real_final": final_metrics},
        "history": history, "wall_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    weights = validate_args(args)
    if args.phase == "sim" and args.start_checkpoint is not None:
        raise ValueError("sim phase must use random initialization")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repro.set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = build_or_load_model(args, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters())
    print(json.dumps({
        "event": "model", "phase": args.phase,
        "config": model.architecture_config(),
        "parameters": parameter_count, "parameter_bytes": parameter_bytes,
    }), flush=True)
    if args.phase == "sim":
        loader, normalizer, windows, trajectories = prepare_sim(args, device)
        set_model_stats(model, normalizer)
        summary = train_sim(
            args, model, loader, normalizer, device, windows, trajectories,
            weights)
    else:
        (train_loader, val_loader, normalizer, train_windows,
         val_windows, trajectories) = prepare_real(args, device)
        set_model_stats(model, normalizer)
        teacher = None
        if args.teacher_checkpoint is not None:
            baseline_loader = repro.load_baseline_module(args.kit_dir)
            teacher, _teacher_metadata = baseline_loader.load_baseline(
                "fno", str(args.teacher_checkpoint), device=str(device))
            teacher = teacher.to(device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        summary = train_real(
            args, model, train_loader, val_loader, normalizer, device,
            train_windows, val_windows, trajectories, weights, teacher)
    summary.update({
        "model_config": model.architecture_config(),
        "parameters": parameter_count,
        "parameter_bytes": parameter_bytes,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    })
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if summary["history"]:
        with (args.output_dir / "learning_curve.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(dict.fromkeys(
                key for row in summary["history"] for key in row))
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary["history"])
    print(json.dumps({
        "event": "complete", "phase": args.phase,
        "wall_seconds": summary["wall_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
