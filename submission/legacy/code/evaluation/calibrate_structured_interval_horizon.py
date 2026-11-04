#!/usr/bin/env python3
"""Calibrate structured SPS bounds by forecast-horizon groups."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/uncertainty"))

from structured_interval_head import StructuredIntervalHead

_train_spec = importlib.util.spec_from_file_location(
    "realpde_structured_train",
    REPO_ROOT / "code/training/train_structured_interval_head.py")
if _train_spec is None or _train_spec.loader is None:
    raise ImportError("Cannot import train_structured_interval_head.py")
_train = importlib.util.module_from_spec(_train_spec)
sys.modules[_train_spec.name] = _train
_train_spec.loader.exec_module(_train)
_base = _train._base
_repro = _train._repro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bin-sizes", type=int, nargs="+", default=(20, 5, 2, 1),
        help="Contiguous horizon-group sizes to audit; 20 is global calibration.")
    parser.add_argument("--selected-bin-size", type=int, default=0,
                        help="Write a deployable checkpoint for this bin size; 0 selects best stress SPS.")
    parser.add_argument("--robust-residual-scale", type=float, default=1.65)
    parser.add_argument(
        "--power-candidates", type=float, nargs="+",
        default=(0.6, 0.8, 1.0, 1.2, 1.4),
        help="Candidate powers for each channel's predicted width contrast.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--stats", type=Path,
        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    return parser.parse_args()


def load_head(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    head = StructuredIntervalHead(
        hidden=int(payload["head_hidden"]),
        architecture=payload["architecture"],
        asymmetric=bool(payload["asymmetric"]),
        global_context=bool(payload.get("global_context", False)),
    ).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head, payload


def metrics(arrays: dict[str, torch.Tensor], lower_multiplier: torch.Tensor,
            upper_multiplier: torch.Tensor, residual_scale: float,
            sample_mask: torch.Tensor | None = None) -> dict[str, float]:
    selected = slice(None) if sample_mask is None else sample_mask
    residual = arrays["residual"][selected] * residual_scale
    lower = arrays["lower"][selected] * lower_multiplier.reshape(1, -1, 1, 1, 2)
    upper = arrays["upper"][selected] * upper_multiplier.reshape(1, -1, 1, 1, 2)
    scored = arrays["scored"][selected]
    inside = (residual >= -lower) & (residual <= upper)
    factor = arrays["factor"][selected].reshape(-1, 1, 1, 1, 1)
    value = factor * inside * torch.exp(
        -(lower + upper) / _train.SIGMA_GLOBAL)
    denominator = int(scored.sum())
    return {
        "sps_score": 100.0 * float(value[scored].sum(dtype=torch.float64)) / denominator,
        "coverage": int((inside & scored).sum()) / denominator,
        "mean_width": float((lower + upper)[scored].sum(dtype=torch.float64)) / denominator,
        "scored_elements": denominator,
    }


def grouped_calibration(arrays: dict[str, torch.Tensor], bin_size: int,
                        residual_scale: float,
                        sample_mask: torch.Tensor | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    timesteps = arrays["residual"].shape[1]
    if bin_size <= 0 or timesteps % bin_size != 0:
        raise ValueError(f"bin size {bin_size} must divide horizon {timesteps}")
    lower = torch.ones((timesteps, 2), device=arrays["residual"].device)
    upper = torch.ones_like(lower)
    groups = []
    selected = slice(None) if sample_mask is None else sample_mask
    for start in range(0, timesteps, bin_size):
        stop = start + bin_size
        cache = []
        for channel in range(2):
            scored = arrays["scored"][selected, start:stop, ..., channel]
            factor = arrays["factor"][selected].reshape(
                -1, 1, 1, 1).expand_as(scored)
            cache.append({
                "residual": arrays["residual"][selected, start:stop, ..., channel][scored],
                "lower": arrays["lower"][selected, start:stop, ..., channel][scored],
                "upper": arrays["upper"][selected, start:stop, ..., channel][scored],
                "factor": factor[scored],
            })
        group_lower, group_upper, _trace = _train.calibrate_multipliers(
            cache, asymmetric=True, residual_scale=residual_scale)
        lower[start:stop] = lower.new_tensor(group_lower)
        upper[start:stop] = upper.new_tensor(group_upper)
        groups.append({
            "start": start,
            "stop": stop,
            "lower_uv": group_lower,
            "upper_uv": group_upper,
        })
    return lower, upper, groups


def channel_cache(arrays: dict[str, torch.Tensor], channel: int,
                  power: float, sample_mask: torch.Tensor | None = None):
    selected = slice(None) if sample_mask is None else sample_mask
    scored = arrays["scored"][selected, ..., channel]
    factor = arrays["factor"][selected].reshape(
        -1, 1, 1, 1).expand_as(scored)
    reference = (0.01075, 0.00550)[channel]
    lower = reference * (
        arrays["lower"][selected, ..., channel] / reference).pow(power)
    upper = reference * (
        arrays["upper"][selected, ..., channel] / reference).pow(power)
    return {
        "residual": arrays["residual"][selected, ..., channel][scored],
        "lower": lower[scored],
        "upper": upper[scored],
        "factor": factor[scored],
    }


def select_width_powers(arrays: dict[str, torch.Tensor], candidates: list[float],
                        residual_scale: float):
    selected_powers = []
    traces = []
    for channel in range(2):
        channel_trace = []
        best_score = -float("inf")
        best_power = 1.0
        for power in candidates:
            cache = channel_cache(arrays, channel, power)
            # The calibration helper expects u/v caches. Duplicating this
            # channel preserves its optimum and keeps one exact implementation
            # of the asymmetric multiplier search.
            lower, upper, _grid = _train.calibrate_multipliers(
                [cache, cache], asymmetric=True,
                residual_scale=residual_scale)
            score = _train.channel_objective(
                cache, lower[0], upper[0], residual_scale)
            channel_trace.append({
                "power": power,
                "lower_multiplier": lower[0],
                "upper_multiplier": upper[0],
                "objective_sum": score,
            })
            if score > best_score:
                best_score = score
                best_power = power
        selected_powers.append(best_power)
        traces.append({"channel": channel, "candidates": channel_trace})
    return selected_powers, traces


def apply_width_powers(arrays: dict[str, torch.Tensor], powers: list[float]) -> None:
    for channel, reference in enumerate((0.01075, 0.00550)):
        power = powers[channel]
        arrays["lower"][..., channel] = reference * (
            arrays["lower"][..., channel] / reference).pow(power)
        arrays["upper"][..., channel] = reference * (
            arrays["upper"][..., channel] / reference).pow(power)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories, metadata = _repro.load_trajectories(args.real_dir, 0)
    filenames = [entry["file"] for entry in metadata]
    _train_windows, val_windows = _base.validation_windows(
        args.split_manifest, filenames)
    loader = DataLoader(
        _repro.WindowDataset(trajectories, val_windows),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
        pin_memory=True)
    normalizer = _repro.Normalizer(args.stats, device)
    baseline = _repro.load_baseline_module(args.kit_dir)
    point_model, _metadata = baseline.load_baseline(
        "fno", str(args.point_checkpoint), device=str(device))
    point_model.eval()
    head, payload = load_head(args.head_checkpoint, device)

    parts = {key: [] for key in ("residual", "lower", "upper", "scored", "factor")}
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)[..., :2]
            prediction = _base.point_prediction(
                point_model, x, normalizer)[..., :2]
            lower, upper = head(prediction, x)
            parts["residual"].append(target - prediction)
            parts["lower"].append(lower)
            parts["upper"].append(upper)
            parts["scored"].append(target != 0.0)
            parts["factor"].append(_base.sample_accuracy_factor(prediction, target))
    arrays = {key: torch.cat(value) for key, value in parts.items()}
    trajectory_fold = torch.tensor(
        [window.trajectory % 2 for window in val_windows], device=device)
    power_candidates = [float(value) for value in args.power_candidates]
    if any(value <= 0 for value in power_candidates):
        raise ValueError("Width-power candidates must be positive")
    selected_powers, power_trace = select_width_powers(
        arrays, power_candidates, args.robust_residual_scale)
    apply_width_powers(arrays, selected_powers)
    print(json.dumps({"event": "power_selection",
                      "selected_width_power_uv": selected_powers}), flush=True)

    results = {}
    calibrations = {}
    for bin_size in args.bin_sizes:
        lower_multiplier, upper_multiplier, groups = grouped_calibration(
            arrays, bin_size, args.robust_residual_scale)
        calibrations[bin_size] = (lower_multiplier, upper_multiplier)
        results[str(bin_size)] = {
            "groups": groups,
            "robust_on_local_metrics": metrics(
                arrays, lower_multiplier, upper_multiplier, 1.0),
            "robust_stress_metrics": metrics(
                arrays, lower_multiplier, upper_multiplier,
                args.robust_residual_scale),
        }
        heldout = []
        for calibration_fold in (0, 1):
            calibration_mask = trajectory_fold == calibration_fold
            evaluation_mask = ~calibration_mask
            fold_lower, fold_upper, _fold_groups = grouped_calibration(
                arrays, bin_size, args.robust_residual_scale,
                sample_mask=calibration_mask)
            heldout.append({
                "calibration_fold": calibration_fold,
                "evaluation_fold": 1 - calibration_fold,
                "samples": int(evaluation_mask.sum()),
                "robust_stress_metrics": metrics(
                    arrays, fold_lower, fold_upper,
                    args.robust_residual_scale, sample_mask=evaluation_mask),
            })
        total_elements = sum(
            item["robust_stress_metrics"]["scored_elements"] for item in heldout)
        results[str(bin_size)]["trajectory_parity_cross_validation"] = heldout
        results[str(bin_size)]["cross_validated_stress_sps"] = sum(
            item["robust_stress_metrics"]["scored_elements"]
            * item["robust_stress_metrics"]["sps_score"]
            for item in heldout) / total_elements
        print(json.dumps({"event": "bin_result", "bin_size": bin_size,
                          **results[str(bin_size)]}, ensure_ascii=False), flush=True)

    selected = args.selected_bin_size
    if selected == 0:
        selected = max(
            args.bin_sizes,
            key=lambda size: results[str(size)]["cross_validated_stress_sps"])
    if selected not in calibrations:
        raise ValueError("Selected bin size was not evaluated")
    selected_lower, selected_upper = calibrations[selected]
    output_payload = {
        **payload,
        "robust_lower_multiplier_uv": selected_lower.detach().cpu().tolist(),
        "robust_upper_multiplier_uv": selected_upper.detach().cpu().tolist(),
        "lower_width_power_uv": selected_powers,
        "upper_width_power_uv": selected_powers,
        "horizon_calibration": {
            "bin_size": selected,
            "residual_scale": args.robust_residual_scale,
            "validation_windows": len(val_windows),
        },
        "robust_multiplier_on_local_metrics": results[str(selected)]["robust_on_local_metrics"],
        "robust_stress_metrics": results[str(selected)]["robust_stress_metrics"],
    }
    checkpoint_path = args.output_dir / "interval_head_horizon_calibrated.pth"
    torch.save(output_payload, checkpoint_path)
    report = {
        "point_checkpoint": str(args.point_checkpoint),
        "head_checkpoint": str(args.head_checkpoint),
        "validation_windows": len(val_windows),
        "robust_residual_scale": args.robust_residual_scale,
        "selected_bin_size": selected,
        "selected_width_power_uv": selected_powers,
        "power_trace": power_trace,
        "checkpoint": str(checkpoint_path),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"event": "complete", **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
