#!/usr/bin/env python3
"""Audit and robustly calibrate native U-Net intervals on the Real tail split.

The native head predicts an asymmetric width for every element.  This script
keeps that structure and only searches multiplicative corrections split by
four horizon blocks, velocity channel, and lower/upper side.  Residual scaling
is used as an explicit Sim2Real stress test; it is not reported as validation
performance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from train_flow_backbone import construct_model  # noqa: E402


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--residual-scales", type=float, nargs="+",
        default=(1.0, 1.25, 1.4, 1.5, 1.65))
    parser.add_argument("--grid-min", type=float, default=0.50)
    parser.add_argument("--grid-max", type=float, default=2.50)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument(
        "--sample-stride", type=int, default=29,
        help="Tune on every Nth flattened element, then score chosen values exactly.")
    return parser.parse_args()


def sample_factor(official, prediction: np.ndarray,
                  target: np.ndarray) -> np.ndarray:
    errors = (
        official.rel_l2_per_sample(prediction, target, 2),
        official.tke_rel_l2_per_sample(prediction, target, 2),
        official.mvpe_rel_l2_per_sample(prediction, target),
    )
    factors = [0.5 / (0.5 + np.maximum(error, 0.0)) for error in errors]
    return 0.5 * factors[0] + 0.3 * factors[1] + 0.2 * factors[2]


def collect(model, loader, normalizer, device, official) -> dict[str, np.ndarray]:
    predictions = []
    targets = []
    lowers = []
    uppers = []
    model.eval()
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            x_norm, _target_norm = normalizer.preprocess(x, target)
            pred_norm, lower_norm, upper_norm = model.forward_distribution(x_norm)
            prediction = normalizer.postprocess_prediction(pred_norm)[..., :2]
            scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
            predictions.append(prediction.float().cpu().numpy())
            targets.append(target[..., :2].float().cpu().numpy())
            lowers.append((lower_norm * scale).float().cpu().numpy())
            uppers.append((upper_norm * scale).float().cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float32, copy=False)
    target = np.concatenate(targets).astype(np.float32, copy=False)
    lower = np.concatenate(lowers).astype(np.float32, copy=False)
    upper = np.concatenate(uppers).astype(np.float32, copy=False)
    return {
        "residual": target - prediction,
        "lower": lower,
        "upper": upper,
        "scored": target != 0.0,
        "factor": sample_factor(official, prediction, target).astype(np.float32),
        "prediction": prediction,
        "target": target,
    }


def calibration_array(values: np.ndarray) -> np.ndarray:
    """Expand [4 horizon blocks,2 channels,2 sides] to [20,2,2]."""
    if values.shape != (4, 2, 2):
        raise ValueError(f"unexpected calibration shape {values.shape}")
    return np.repeat(values, 5, axis=0)


def metrics_for(cache: dict[str, np.ndarray], calibration: np.ndarray,
                residual_scale: float = 1.0,
                sample_indices: np.ndarray | None = None) -> dict[str, float]:
    residual = cache["residual"]
    lower = cache["lower"] * calibration[None, ..., 0][..., None, None, :]
    upper = cache["upper"] * calibration[None, ..., 1][..., None, None, :]
    scored = cache["scored"]
    factor = cache["factor"]
    if sample_indices is not None:
        residual = residual[sample_indices]
        lower = lower[sample_indices]
        upper = upper[sample_indices]
        scored = scored[sample_indices]
        factor = factor[sample_indices]
    inside = ((residual * residual_scale >= -lower)
              & (residual * residual_scale <= upper))
    penalty = np.exp(-(lower + upper) / common.SPS_SIGMA)
    shaped_factor = factor.reshape((-1, 1, 1, 1, 1))
    count = int(np.count_nonzero(scored))
    score = np.sum(
        shaped_factor * penalty * inside, where=scored,
        dtype=np.float64) / count
    return {
        "sps_score": float(100.0 * score),
        "coverage": float(np.count_nonzero(inside & scored) / count),
        "mean_width": float(np.sum(
            lower + upper, where=scored, dtype=np.float64) / count),
    }


def tune(cache: dict[str, np.ndarray], residual_scale: float,
         grid: np.ndarray, sample_stride: int) -> tuple[np.ndarray, dict]:
    """Coordinate-descent 16 structured multipliers on a deterministic subset."""
    subset = {
        key: (value[:, ::2, ::2, ::2, ...]
              if key in {"residual", "lower", "upper", "scored"} else value)
        for key, value in cache.items()
        if key in {"residual", "lower", "upper", "scored", "factor"}
    }
    # Preserve whole samples so their SPS accuracy factors remain correctly aligned.
    if sample_stride > 1:
        subset = {
            key: (value[:, :, :, ::sample_stride, ...]
                  if key in {"residual", "lower", "upper", "scored"} else value)
            for key, value in subset.items()
        }
    values = np.ones((4, 2, 2), dtype=np.float32)
    trace = []
    sampled_horizons = np.arange(0, 20, 2)
    for round_index in range(2):
        for horizon_group in range(4):
            selected_horizons = np.flatnonzero(
                sampled_horizons // 5 == horizon_group)
            original_horizons = sampled_horizons[selected_horizons]
            for channel in range(2):
                for side in range(2):
                    best_value = float(values[horizon_group, channel, side])
                    best_score = -np.inf
                    for candidate in grid:
                        trial = values.copy()
                        trial[horizon_group, channel, side] = candidate
                        expanded = calibration_array(trial)[:, channel, side]
                        residual = subset["residual"][
                            ..., channel][:, selected_horizons]
                        lower = subset["lower"][
                            ..., channel][:, selected_horizons]
                        upper = subset["upper"][
                            ..., channel][:, selected_horizons]
                        scored = subset["scored"][
                            ..., channel][:, selected_horizons]
                        lower = lower * expanded[original_horizons][
                            None, :, None, None]
                        upper = upper * expanded[original_horizons][
                            None, :, None, None]
                        inside = ((residual * residual_scale >= -lower)
                                  & (residual * residual_scale <= upper))
                        penalty = np.exp(-(lower + upper) / common.SPS_SIGMA)
                        factor = subset["factor"].reshape((-1, 1, 1, 1))
                        count = int(np.count_nonzero(scored))
                        value = np.sum(
                            factor * penalty * inside, where=scored,
                            dtype=np.float64) / max(count, 1)
                        if value > best_score:
                            best_score = value
                            best_value = float(candidate)
                    values[horizon_group, channel, side] = best_value
                    trace.append({
                        "round": round_index + 1,
                        "horizon_group": horizon_group,
                        "channel": "uv"[channel],
                        "side": ("lower", "upper")[side],
                        "multiplier": best_value,
                        "subset_channel_score": float(100.0 * best_score),
                    })
    return calibration_array(values), {"block_values": values.tolist(), "trace": trace}


def grouped_metrics(cache: dict[str, np.ndarray], calibration: np.ndarray,
                    labels: list[dict], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(labels):
        groups.setdefault(str(item[key]), []).append(index)
    return {
        value: metrics_for(cache, calibration, sample_indices=np.asarray(indices))
        for value, indices in sorted(groups.items(), key=lambda pair: float(pair[0]))
    }


def horizon_metrics(cache: dict[str, np.ndarray],
                    calibration: np.ndarray) -> list[dict[str, float]]:
    result = []
    for group in range(4):
        selected = {
            key: (value[:, group * 5:(group + 1) * 5]
                  if key in {"residual", "lower", "upper", "scored"} else value)
            for key, value in cache.items()
            if key in {"residual", "lower", "upper", "scored", "factor"}
        }
        selected_calibration = calibration[group * 5:(group + 1) * 5]
        result.append({
            "horizon": f"{group * 5 + 1}-{(group + 1) * 5}",
            **metrics_for(selected, selected_calibration),
        })
    return result


def main() -> None:
    args = parse_args()
    if (args.eval_batch_size <= 0 or args.sample_stride <= 0
            or args.grid_step <= 0 or args.grid_min <= 0
            or args.grid_max < args.grid_min
            or any(value <= 0 for value in args.residual_scales)):
        raise ValueError("invalid positive calibration arguments")
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = construct_model("unet", payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    (train_loader, val_loader, normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)
    common.set_model_stats(model, normalizer)
    official = import_module(args.kit_dir / "scoring.py", "unet_native_scoring")
    cache = collect(model, val_loader, normalizer, device, official)
    del model, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    manifest = json.loads(
        (args.output.parent / "split_manifest.json").read_text(encoding="utf-8"))
    labels = [entry for entry in manifest for _start in entry["val_starts"]]
    identity = np.ones((20, 2, 2), dtype=np.float32)
    grid = np.arange(
        args.grid_min, args.grid_max + 0.5 * args.grid_step,
        args.grid_step, dtype=np.float32)
    scenarios = {}
    calibrations = {}
    for scale in args.residual_scales:
        calibration, detail = tune(cache, float(scale), grid, args.sample_stride)
        key = f"{float(scale):.2f}"
        calibrations[key] = calibration.tolist()
        scenarios[key] = {
            "calibration": detail,
            "native_under_stress": metrics_for(cache, identity, float(scale)),
            "on_local": metrics_for(cache, calibration),
            "under_stress": metrics_for(cache, calibration, float(scale)),
        }
        print(json.dumps({"event": "calibration", "scale": scale,
                          **scenarios[key]["on_local"]}), flush=True)

    point = {
        "rel_l2_score": official.score_error(float(np.mean(
            official.rel_l2_per_sample(
                cache["prediction"], cache["target"], 2)))),
        "tke_score": official.score_error(float(np.mean(
            official.tke_rel_l2_per_sample(
                cache["prediction"], cache["target"], 2)))),
        "mvpe_score": official.score_error(float(np.mean(
            official.mvpe_rel_l2_per_sample(
                cache["prediction"], cache["target"])))),
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "point_scores": point,
        "native": {
            **metrics_for(cache, identity),
            "by_re": grouped_metrics(cache, identity, labels, "re"),
            "by_aoa": grouped_metrics(cache, identity, labels, "aoa"),
            "by_horizon": horizon_metrics(cache, identity),
        },
        "scenarios": scenarios,
        "calibrations": calibrations,
        "note": (
            "Residual scales above 1 are stress tests for hidden-domain error "
            "inflation. on_local is the released temporal-tail score; "
            "under_stress is not a validation score."),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output)}),
          flush=True)


if __name__ == "__main__":
    main()
