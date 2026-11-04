#!/usr/bin/env python3
"""Train spatial and/or asymmetric uncertainty heads on a frozen Track 1 FNO."""

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
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/uncertainty"))

from structured_interval_head import StructuredIntervalHead
from conditioned_residual_corrector import ConditionedResidualCorrector

_base_spec = importlib.util.spec_from_file_location(
    "realpde_interval_base", REPO_ROOT / "code/training/train_fno_interval_head.py")
if _base_spec is None or _base_spec.loader is None:
    raise ImportError("Cannot import train_fno_interval_head.py")
_base = importlib.util.module_from_spec(_base_spec)
sys.modules[_base_spec.name] = _base
_base_spec.loader.exec_module(_base)
_repro = _base._repro


SIGMA_GLOBAL = _base.SIGMA_GLOBAL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--correction-checkpoint", type=Path,
        help="Optional frozen ConditionedResidualCorrector checkpoint.")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("pointwise", "spatial", "dilated", "deep"),
                        default="spatial")
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument(
        "--global-context", action="store_true",
        help="Add metadata-free frame/global flow statistics to the local features.")
    parser.add_argument(
        "--condition-context", action="store_true",
        help="Condition widths on Re/AoA estimates from --correction-checkpoint.")
    parser.add_argument(
        "--correction-scale", type=float, default=1.0,
        help=("Scale the frozen point correction while retaining its inferred "
              "condition context; 0 is a condition-only ablation."))
    parser.add_argument(
        "--continuity-weight", type=float, default=0.0,
        help="Causal first-frame boundary correction weight; 0 disables it.")
    parser.add_argument(
        "--continuity-power", type=float, default=4.0,
        help="Horizon decay power for --continuity-weight.")
    parser.add_argument("--variant", choices=("soft_sps", "soft_sps_reg"),
                        default="soft_sps_reg")
    parser.add_argument(
        "--nll-weight", type=float,
        help="Override the variant's NLL regularization weight.")
    parser.add_argument(
        "--temperature-fraction", type=float, default=0.10,
        help="Sigmoid coverage temperature as a fraction of channel reference width.")
    parser.add_argument(
        "--train-residual-scale-min", type=float, default=1.0,
        help="Minimum per-sample residual scale used during robust training.")
    parser.add_argument(
        "--train-residual-scale-max", type=float, default=1.0,
        help="Maximum per-sample residual scale used during robust training.")
    parser.add_argument(
        "--group-robust-weight", type=float, default=0.0,
        help="Blend mean interval loss with smooth worst Re/AoA-group loss.")
    parser.add_argument(
        "--group-temperature", type=float, default=0.05,
        help="Log-sum-exp temperature for marginal group-robust interval loss.")
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--holdout-re", type=int, nargs="*", default=(),
        help=("Optional Reynolds values reserved completely for validation. "
              "This is an OOD diagnostic; default uses the temporal-tail split."))
    parser.add_argument(
        "--holdout-aoa", type=int, nargs="*", default=(),
        help=("Optional angles of attack reserved completely for validation. "
              "Cannot be combined with --holdout-re."))
    parser.add_argument(
        "--refit-all-windows", action="store_true",
        help=("Train on the union of manifest train/validation windows and select "
              "the final iterate. Requires fixed robust multipliers because the "
              "diagnostic validation windows are no longer held out."))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--stats", type=Path,
        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--robust-residual-scale", type=float, default=1.65)
    parser.add_argument(
        "--fixed-robust-multipliers", type=float, nargs=4,
        metavar=("LOWER_U", "LOWER_V", "UPPER_U", "UPPER_V"),
        help="Override robust calibration, e.g. 1.4 1.5 1.4 1.5.")
    parser.add_argument("--skip-calibration", action="store_true")
    return parser.parse_args()


class GroupedWindowDataset(Dataset):
    """Attach coarse Reynolds/AoA groups for training-time robustness."""

    def __init__(self, trajectories, windows, metadata):
        self.base = _repro.WindowDataset(trajectories, windows)
        self.windows = windows
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, target = self.base[index]
        item = self.metadata[self.windows[index].trajectory]
        re_group = 0 if item["re"] <= 9000 else (1 if item["re"] < 17000 else 2)
        aoa_group = int(round(float(item["aoa"]) / 5.0))
        return x, target, re_group, aoa_group


def experiment_windows(manifest_path: Path, metadata: list[dict],
                       holdout_re: tuple[int, ...] | list[int],
                       holdout_aoa: tuple[int, ...] | list[int],
                       refit_all_windows: bool = False):
    """Build either the standard temporal split or a full-condition holdout."""
    if holdout_re and holdout_aoa:
        raise ValueError("Use only one of --holdout-re and --holdout-aoa")
    if refit_all_windows and (holdout_re or holdout_aoa):
        raise ValueError("--refit-all-windows cannot be combined with a holdout")
    filenames = [entry["file"] for entry in metadata]
    if not holdout_re and not holdout_aoa:
        train, val = _base.validation_windows(manifest_path, filenames)
        if refit_all_windows:
            all_windows = list(train) + list(val)
            return all_windows, val, {
                "kind": "refit_all_windows",
                "values": [],
                "diagnostic_val_is_in_training": True,
            }
        return train, val, {"kind": "temporal_tail", "values": []}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_file = {entry["file"]: entry for entry in manifest}
    if not set(filenames).issubset(by_file):
        raise ValueError("Split manifest does not cover every loaded data file")
    kind = "re" if holdout_re else "aoa"
    values = {int(value) for value in (holdout_re or holdout_aoa)}
    available = {int(entry[kind]) for entry in metadata}
    missing = values - available
    if missing:
        raise ValueError(f"Unknown {kind} holdout values: {sorted(missing)}")

    train = []
    val = []
    heldout_files = []
    for trajectory, item in enumerate(metadata):
        entry = by_file[item["file"]]
        if int(item[kind]) in values:
            heldout_files.append(item["file"])
            starts = sorted(set(entry["train_starts"] + entry["val_starts"]))
            val.extend(_repro.Window(trajectory, int(start)) for start in starts)
        else:
            train.extend(
                _repro.Window(trajectory, int(start))
                for start in entry["train_starts"])
    if not train or not val:
        raise ValueError("Condition holdout must leave both train and validation windows")
    return train, val, {
        "kind": f"holdout_{kind}",
        "values": sorted(values),
        "heldout_files": heldout_files,
    }


def interval_losses(prediction: torch.Tensor, target: torch.Tensor,
                    lower_width: torch.Tensor, upper_width: torch.Tensor,
                    variant: str, nll_weight: float | None = None,
                    temperature_fraction: float = 0.10,
                    residual_scale: float | torch.Tensor = 1.0,
                    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = prediction[..., :2]
    target = target[..., :2]
    residual = (target - pred) * residual_scale
    scored = target != 0.0
    factor = _base.sample_accuracy_factor(pred, target).reshape(-1, 1, 1, 1, 1)
    reference = pred.new_tensor([0.01075, 0.00550]).reshape(1, 1, 1, 1, 2)
    tau = temperature_fraction * reference

    negative_cost = (-residual).clamp_min(0.0) / lower_width.clamp_min(1e-7)
    positive_cost = residual.clamp_min(0.0) / upper_width.clamp_min(1e-7)
    normalized_total_width = (
        (lower_width + upper_width) / (2.0 * reference)).clamp_min(1e-6)
    nll_elements = negative_cost + positive_cost + torch.log(normalized_total_width)
    nll = nll_elements[scored].mean()

    soft_lower = torch.sigmoid((residual + lower_width) / tau)
    soft_upper = torch.sigmoid((upper_width - residual) / tau)
    soft_inside = soft_lower * soft_upper
    penalty = torch.exp(-(lower_width + upper_width) / SIGMA_GLOBAL)
    soft_score = (factor * soft_inside * penalty)[scored].mean()
    selected_nll_weight = (
        (0.02 if variant == "soft_sps" else 0.10)
        if nll_weight is None else nll_weight
    )
    loss = -soft_score + selected_nll_weight * nll
    return loss, {"nll": nll.detach(), "soft_sps": soft_score.detach()}


def group_robust_interval_loss(
        prediction: torch.Tensor, target: torch.Tensor,
        lower_width: torch.Tensor, upper_width: torch.Tensor,
        re_group: torch.Tensor, aoa_group: torch.Tensor,
        variant: str, nll_weight: float | None, temperature_fraction: float,
        residual_scale: float | torch.Tensor, temperature: float,
        ) -> torch.Tensor:
    """Return a smooth maximum over marginal Re and AoA group losses."""
    losses = []
    for groups in (re_group, aoa_group):
        for group in torch.unique(groups):
            selected = groups == group
            selected_scale = (
                residual_scale[selected]
                if isinstance(residual_scale, torch.Tensor)
                else residual_scale)
            value, _components = interval_losses(
                prediction[selected], target[selected], lower_width[selected],
                upper_width[selected], variant, nll_weight=nll_weight,
                temperature_fraction=temperature_fraction,
                residual_scale=selected_scale)
            losses.append(value)
    stacked = torch.stack(losses)
    return temperature * (
        torch.logsumexp(stacked / temperature, dim=0)
        - math.log(len(losses)))


def corrected_point_prediction(point_model: nn.Module, corrector: nn.Module | None,
                               x: torch.Tensor, normalizer,
                               correction_scale: float = 1.0,
                               continuity_weight: float = 0.0,
                               continuity_power: float = 4.0,
                               ) -> tuple[torch.Tensor, torch.Tensor | None]:
    prediction = _base.point_prediction(point_model, x, normalizer)
    result = prediction.clone()
    condition = None
    if corrector is not None:
        corrected_uv, condition = corrector(prediction[..., :2], x)
        result[..., :2] = (
            prediction[..., :2]
            + correction_scale * (corrected_uv - prediction[..., :2]))
    if continuity_weight > 0.0:
        phase = torch.linspace(
            1.0, 0.0, result.shape[1], dtype=result.dtype,
            device=result.device).pow(continuity_power)
        ramp = phase.view(1, -1, 1, 1, 1)
        jump = x[:, -1:, ..., :2] - result[:, :1, ..., :2]
        result[..., :2] = result[..., :2] + continuity_weight * ramp * jump
    result[..., 2] = 0.0
    return result, condition


def exact_sps_for_loader(point_model: nn.Module, corrector: nn.Module | None,
                         head: nn.Module, loader: DataLoader,
                         normalizer, device: torch.device,
                         correction_scale: float = 1.0,
                         continuity_weight: float = 0.0,
                         continuity_power: float = 4.0) -> dict[str, float]:
    numerator = 0.0
    denominator = 0
    inside_count = 0
    width_sum = 0.0
    head.eval()
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)[..., :2]
            full_pred, condition = corrected_point_prediction(
                point_model, corrector, x, normalizer, correction_scale,
                continuity_weight, continuity_power)
            pred = full_pred[..., :2]
            lower_width, upper_width = head(pred, x, condition)
            residual = target - pred
            scored = target != 0.0
            inside = (residual >= -lower_width) & (residual <= upper_width)
            factor = _base.sample_accuracy_factor(pred, target).reshape(-1, 1, 1, 1, 1)
            elements = factor * inside * torch.exp(
                -(lower_width + upper_width) / SIGMA_GLOBAL)
            numerator += float(elements[scored].sum())
            denominator += int(scored.sum())
            inside_count += int((inside & scored).sum())
            width_sum += float((lower_width + upper_width)[scored].sum())
    return {
        "sps_score": 100.0 * numerator / denominator,
        "coverage": inside_count / denominator,
        "mean_width": width_sum / denominator,
    }


def collect_interval_cache(point_model: nn.Module, corrector: nn.Module | None,
                           head: nn.Module, loader: DataLoader,
                           normalizer, device: torch.device,
                           correction_scale: float = 1.0,
                           continuity_weight: float = 0.0,
                           continuity_power: float = 4.0,
                           ) -> list[dict[str, torch.Tensor]]:
    parts = [
        {"residual": [], "lower": [], "upper": [], "factor": []},
        {"residual": [], "lower": [], "upper": [], "factor": []},
    ]
    head.eval()
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)[..., :2]
            full_pred, condition = corrected_point_prediction(
                point_model, corrector, x, normalizer, correction_scale,
                continuity_weight, continuity_power)
            pred = full_pred[..., :2]
            lower, upper = head(pred, x, condition)
            factor = _base.sample_accuracy_factor(pred, target).reshape(-1, 1, 1, 1)
            for channel in range(2):
                scored = target[..., channel] != 0.0
                expanded_factor = factor.expand_as(target[..., channel])
                parts[channel]["residual"].append(
                    (target[..., channel] - pred[..., channel])[scored])
                parts[channel]["lower"].append(lower[..., channel][scored])
                parts[channel]["upper"].append(upper[..., channel][scored])
                parts[channel]["factor"].append(expanded_factor[scored])
    return [
        {name: torch.cat(tensors) for name, tensors in channel.items()}
        for channel in parts
    ]


def channel_objective(channel: dict[str, torch.Tensor], lower_multiplier: float,
                      upper_multiplier: float, residual_scale: float) -> float:
    residual = channel["residual"] * residual_scale
    lower = channel["lower"] * lower_multiplier
    upper = channel["upper"] * upper_multiplier
    inside = (residual >= -lower) & (residual <= upper)
    value = channel["factor"] * inside * torch.exp(
        -(lower + upper) / SIGMA_GLOBAL)
    return float(torch.sum(value, dtype=torch.float64))


def exact_sps_for_cache(cache: list[dict[str, torch.Tensor]],
                        lower_multipliers: list[float],
                        upper_multipliers: list[float],
                        residual_scale: float = 1.0) -> dict[str, float]:
    numerator = 0.0
    denominator = 0
    inside_count = 0
    width_sum = 0.0
    for index, channel in enumerate(cache):
        residual = channel["residual"] * residual_scale
        lower = channel["lower"] * lower_multipliers[index]
        upper = channel["upper"] * upper_multipliers[index]
        inside = (residual >= -lower) & (residual <= upper)
        numerator += float(torch.sum(
            (channel["factor"] * torch.exp(
                -(lower + upper) / SIGMA_GLOBAL))[inside], dtype=torch.float64))
        denominator += int(residual.numel())
        inside_count += int(inside.sum())
        width_sum += float(torch.sum(lower + upper, dtype=torch.float64))
    return {
        "sps_score": 100.0 * numerator / denominator,
        "coverage": inside_count / denominator,
        "mean_width": width_sum / denominator,
    }


def calibrate_multipliers(cache: list[dict[str, torch.Tensor]], asymmetric: bool,
                          residual_scale: float) -> tuple[list[float], list[float], dict]:
    candidates = np.arange(0.50, 4.01, 0.10)
    lower_selected = [1.0, 1.0]
    upper_selected = [1.0, 1.0]
    traces: list[dict] = []
    for channel_index, channel in enumerate(cache):
        if not asymmetric:
            scores = [
                channel_objective(channel, float(value), float(value), residual_scale)
                for value in candidates
            ]
            selected = float(candidates[int(np.argmax(scores))])
            lower_selected[channel_index] = selected
            upper_selected[channel_index] = selected
            traces.append({"channel": channel_index, "symmetric_scores": scores})
            continue

        lower_multiplier = 1.0
        upper_multiplier = 1.0
        coordinate_trace = []
        for _iteration in range(5):
            lower_scores = [
                channel_objective(
                    channel, float(value), upper_multiplier, residual_scale)
                for value in candidates
            ]
            next_lower = float(candidates[int(np.argmax(lower_scores))])
            upper_scores = [
                channel_objective(
                    channel, next_lower, float(value), residual_scale)
                for value in candidates
            ]
            next_upper = float(candidates[int(np.argmax(upper_scores))])
            coordinate_trace.append({
                "lower_multiplier": next_lower,
                "upper_multiplier": next_upper,
                "lower_scores": lower_scores,
                "upper_scores": upper_scores,
            })
            if (next_lower == lower_multiplier and next_upper == upper_multiplier):
                break
            lower_multiplier, upper_multiplier = next_lower, next_upper
        lower_selected[channel_index] = lower_multiplier
        upper_selected[channel_index] = upper_multiplier
        traces.append({"channel": channel_index, "coordinate_trace": coordinate_trace})
    metrics = exact_sps_for_cache(
        cache, lower_selected, upper_selected, residual_scale)
    return lower_selected, upper_selected, {
        "candidates": candidates.tolist(), "traces": traces, "selected_metrics": metrics}


def save_head(path: Path, head: nn.Module, args: argparse.Namespace,
              iteration: int, metrics: dict[str, float]) -> None:
    torch.save({
        "head_state_dict": {
            key: value.detach().cpu() for key, value in head.state_dict().items()},
        "head_hidden": args.hidden,
        "architecture": args.architecture,
        "asymmetric": args.asymmetric,
        "global_context": args.global_context,
        "condition_context": args.condition_context,
        "correction_scale": args.correction_scale,
        "continuity_weight": args.continuity_weight,
        "continuity_power": args.continuity_power,
        "variant": args.variant,
        "nll_weight": args.nll_weight,
        "temperature_fraction": args.temperature_fraction,
        "train_residual_scale_min": args.train_residual_scale_min,
        "train_residual_scale_max": args.train_residual_scale_max,
        "group_robust_weight": args.group_robust_weight,
        "group_temperature": args.group_temperature,
        "iteration": iteration,
        "validation": metrics,
    }, path)


def main() -> None:
    args = parse_args()
    if (args.updates <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0
            or args.eval_interval <= 0 or args.lr <= 0 or args.hidden <= 0
            or not math.isfinite(args.temperature_fraction)
            or args.temperature_fraction <= 0
            or (args.nll_weight is not None
                and (not math.isfinite(args.nll_weight)
                     or args.nll_weight < 0))
            or not math.isfinite(args.train_residual_scale_min)
            or not math.isfinite(args.train_residual_scale_max)
            or args.train_residual_scale_min <= 0
            or args.train_residual_scale_max < args.train_residual_scale_min
            or not math.isfinite(args.robust_residual_scale)
            or args.robust_residual_scale <= 0
            or not math.isfinite(args.correction_scale)
            or not 0.0 <= args.correction_scale <= 2.0
            or not math.isfinite(args.continuity_weight)
            or not 0.0 <= args.continuity_weight <= 1.0
            or not math.isfinite(args.continuity_power)
            or args.continuity_power <= 0.0
            or not math.isfinite(args.group_robust_weight)
            or not 0.0 <= args.group_robust_weight <= 1.0
            or not math.isfinite(args.group_temperature)
            or args.group_temperature <= 0):
        raise ValueError("Training/calibration values must be positive")
    if args.refit_all_windows and args.fixed_robust_multipliers is None:
        raise ValueError(
            "--refit-all-windows requires --fixed-robust-multipliers")
    if args.condition_context and args.correction_checkpoint is None:
        raise ValueError(
            "--condition-context requires --correction-checkpoint")
    if (args.fixed_robust_multipliers is not None
            and any(not math.isfinite(value) or value <= 0
                    for value in args.fixed_robust_multipliers)):
        raise ValueError("Fixed robust multipliers must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _repro.set_seed(args.seed)
    device = torch.device(args.device)
    trajectories, metadata = _repro.load_trajectories(args.real_dir, args.max_files)
    train_windows, val_windows, validation_scheme = experiment_windows(
        args.split_manifest, metadata, args.holdout_re, args.holdout_aoa,
        args.refit_all_windows)
    train_dataset = (
        GroupedWindowDataset(trajectories, train_windows, metadata)
        if args.group_robust_weight > 0
        else _repro.WindowDataset(trajectories, train_windows))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(
        _repro.WindowDataset(trajectories, val_windows), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    train_iterator = _repro.cycling(train_loader)
    normalizer = _repro.Normalizer(args.stats, device)
    baseline = _repro.load_baseline_module(args.kit_dir)
    point_model, point_metadata = baseline.load_baseline(
        "fno", str(args.point_checkpoint), device=str(device))
    point_model.eval()
    for parameter in point_model.parameters():
        parameter.requires_grad_(False)

    corrector = None
    correction_metadata = None
    if args.correction_checkpoint is not None:
        correction_metadata = torch.load(
            args.correction_checkpoint, map_location=device, weights_only=False)
        corrector = ConditionedResidualCorrector(
            hidden=int(correction_metadata["hidden"]),
            context_hidden=int(correction_metadata["context_hidden"]),
            architecture=correction_metadata["architecture"],
        ).to(device)
        corrector.load_state_dict(correction_metadata["corrector_state_dict"])
        corrector.eval()
        for parameter in corrector.parameters():
            parameter.requires_grad_(False)

    head = StructuredIntervalHead(
        hidden=args.hidden, architecture=args.architecture,
        asymmetric=args.asymmetric,
        global_context=args.global_context,
        condition_context=args.condition_context).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates)
    best_sps = -float("inf")
    best_iteration = 0
    history = []
    interval_loss = interval_nll = interval_soft = 0.0
    interval_count = 0
    started = time.time()

    for iteration in range(1, args.updates + 1):
        head.train()
        batch = next(train_iterator)
        x, target = batch[:2]
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred, condition = corrected_point_prediction(
            point_model, corrector, x, normalizer, args.correction_scale,
            args.continuity_weight, args.continuity_power)
        lower_width, upper_width = head(pred[..., :2], x, condition)
        if args.train_residual_scale_max == args.train_residual_scale_min:
            train_residual_scale: float | torch.Tensor = (
                args.train_residual_scale_min)
        else:
            train_residual_scale = torch.empty(
                (x.shape[0], 1, 1, 1, 1), device=device,
                dtype=pred.dtype).uniform_(
                    args.train_residual_scale_min,
                    args.train_residual_scale_max)
        mean_loss, components = interval_losses(
            pred, target, lower_width, upper_width, args.variant,
            nll_weight=args.nll_weight,
            temperature_fraction=args.temperature_fraction,
            residual_scale=train_residual_scale)
        if args.group_robust_weight > 0:
            re_group = batch[2].to(device, non_blocking=True)
            aoa_group = batch[3].to(device, non_blocking=True)
            robust_loss = group_robust_interval_loss(
                pred, target, lower_width, upper_width,
                re_group, aoa_group, args.variant, args.nll_weight,
                args.temperature_fraction, train_residual_scale,
                args.group_temperature)
            loss = (
                (1.0 - args.group_robust_weight) * mean_loss
                + args.group_robust_weight * robust_loss)
        else:
            loss = mean_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        interval_loss += float(loss)
        interval_nll += float(components["nll"])
        interval_soft += float(components["soft_sps"])
        interval_count += 1

        if iteration % args.eval_interval == 0 or iteration == args.updates:
            validation = exact_sps_for_loader(
                point_model, corrector, head, val_loader, normalizer, device,
                args.correction_scale, args.continuity_weight,
                args.continuity_power)
            row = {
                "iteration": iteration,
                "train_loss": interval_loss / interval_count,
                "train_nll": interval_nll / interval_count,
                "train_soft_sps": interval_soft / interval_count,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                **validation,
            }
            history.append(row)
            print(json.dumps({"event": "eval", **row}), flush=True)
            interval_loss = interval_nll = interval_soft = 0.0
            interval_count = 0
            select_checkpoint = (
                iteration == args.updates if args.refit_all_windows
                else validation["sps_score"] > best_sps
            )
            if select_checkpoint:
                best_sps = validation["sps_score"]
                best_iteration = iteration
                save_head(args.output_dir / "best.pth", head, args, iteration, validation)

    best_payload = torch.load(
        args.output_dir / "best.pth", map_location=device, weights_only=False)
    head.load_state_dict(best_payload["head_state_dict"])
    cache = collect_interval_cache(
        point_model, corrector, head, val_loader, normalizer, device,
        args.correction_scale, args.continuity_weight,
        args.continuity_power)
    if args.skip_calibration:
        local_lower = local_upper = robust_lower = robust_upper = [1.0, 1.0]
        local_grid = robust_grid = {"skipped": True}
    else:
        local_lower, local_upper, local_grid = calibrate_multipliers(
            cache, args.asymmetric, 1.0)
        robust_lower, robust_upper, robust_grid = calibrate_multipliers(
            cache, args.asymmetric, args.robust_residual_scale)
    if args.fixed_robust_multipliers is not None:
        robust_lower = list(args.fixed_robust_multipliers[:2])
        robust_upper = list(args.fixed_robust_multipliers[2:])
        robust_grid = {
            "override": True,
            "source": "fixed_robust_multipliers",
            "values": list(args.fixed_robust_multipliers),
        }
    local_metrics = exact_sps_for_cache(cache, local_lower, local_upper)
    robust_on_local = exact_sps_for_cache(cache, robust_lower, robust_upper)
    robust_stress = exact_sps_for_cache(
        cache, robust_lower, robust_upper, args.robust_residual_scale)
    final_payload = {
        **best_payload,
        "point_checkpoint": str(args.point_checkpoint),
        "point_checkpoint_metadata": point_metadata,
        "correction_checkpoint": (
            str(args.correction_checkpoint)
            if args.correction_checkpoint is not None else None),
        "correction_scale": args.correction_scale,
        "continuity_weight": args.continuity_weight,
        "continuity_power": args.continuity_power,
        "local_lower_multiplier_uv": local_lower,
        "local_upper_multiplier_uv": local_upper,
        "robust_lower_multiplier_uv": robust_lower,
        "robust_upper_multiplier_uv": robust_upper,
        "robust_residual_scale": args.robust_residual_scale,
        "local_metrics": local_metrics,
        "robust_multiplier_on_local_metrics": robust_on_local,
        "robust_stress_metrics": robust_stress,
    }
    torch.save(final_payload, args.output_dir / "interval_head.pth")
    summary = {
        "point_checkpoint": str(args.point_checkpoint),
        "point_checkpoint_metadata": point_metadata,
        "correction_checkpoint": (
            str(args.correction_checkpoint)
            if args.correction_checkpoint is not None else None),
        "correction_iteration": (
            int(correction_metadata["iteration"])
            if correction_metadata is not None else None),
        "architecture": args.architecture,
        "asymmetric": args.asymmetric,
        "global_context": args.global_context,
        "condition_context": args.condition_context,
        "correction_scale": args.correction_scale,
        "continuity_weight": args.continuity_weight,
        "continuity_power": args.continuity_power,
        "variant": args.variant,
        "nll_weight": args.nll_weight,
        "temperature_fraction": args.temperature_fraction,
        "train_residual_scale_min": args.train_residual_scale_min,
        "train_residual_scale_max": args.train_residual_scale_max,
        "group_robust_weight": args.group_robust_weight,
        "group_temperature": args.group_temperature,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "refit_all_windows": args.refit_all_windows,
        "validation_scheme": validation_scheme,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "best_iteration": best_iteration,
        "best_uncalibrated_sps": best_sps,
        "local_lower_multiplier_uv": local_lower,
        "local_upper_multiplier_uv": local_upper,
        "robust_lower_multiplier_uv": robust_lower,
        "robust_upper_multiplier_uv": robust_upper,
        "local_metrics": local_metrics,
        "robust_multiplier_on_local_metrics": robust_on_local,
        "robust_stress_metrics": robust_stress,
        "wall_seconds": time.time() - started,
        "history": history,
        "local_grid": local_grid,
        "robust_grid": robust_grid,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with (args.output_dir / "learning_curve.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({
        "event": "complete", "output_dir": str(args.output_dir),
        "best_iteration": best_iteration, "best_sps": best_sps,
        "local_metrics": local_metrics,
        "robust_lower_multiplier_uv": robust_lower,
        "robust_upper_multiplier_uv": robust_upper,
        "robust_on_local": robust_on_local,
        "robust_stress": robust_stress,
        "wall_seconds": summary["wall_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
