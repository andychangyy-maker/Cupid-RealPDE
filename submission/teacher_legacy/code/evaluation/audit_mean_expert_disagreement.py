#!/usr/bin/env python3
"""Audit whether mean-expert disagreement can improve native SPS intervals.

The deployed mean-expert uses A/B distribution heads and C only for a
temporal-mean point correction.  This audit keeps the deployed point forecast
fixed and tests label-free interval transforms driven by A/B/C disagreement.
Search uses a deterministic sparse grid; finalists are scored exactly on the
full released real-tail split and under explicit residual-inflation stress.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
from evaluate_flow_backbone_blend import load_model  # noqa: E402


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OFFICIAL = import_module(
    REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py",
    "realpde_track1_v9_mean_expert_disagreement_scoring")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs=3, required=True,
                        metavar=("A", "B", "C"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--weight-b", type=float, default=0.25)
    parser.add_argument("--mean-expert-weight", type=float, default=0.5)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-stride", type=int, default=3)
    parser.add_argument("--finalists", type=int, default=16)
    parser.add_argument(
        "--residual-scales", type=float, nargs="+",
        default=(1.0, 1.15, 1.3, 1.5))
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.eval_batch_size <= 0 or args.search_stride <= 0 or args.finalists <= 0:
        raise ValueError("batch size, search stride and finalists must be positive")
    for name in ("weight_b", "mean_expert_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
    if not args.residual_scales or any(
            not math.isfinite(value) or value <= 0
            for value in args.residual_scales):
        raise ValueError("residual scales must be finite and positive")


def sample_factor(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    errors = (
        OFFICIAL.rel_l2_per_sample(prediction, target, 2),
        OFFICIAL.tke_rel_l2_per_sample(prediction, target, 2),
        OFFICIAL.mvpe_rel_l2_per_sample(prediction, target),
    )
    factors = [0.5 / (0.5 + np.maximum(value, 0.0)) for value in errors]
    return (0.5 * factors[0] + 0.3 * factors[1] + 0.2 * factors[2]
            ).astype(np.float32)


@torch.inference_mode()
def collect_one(model, loader, normalizer, device: torch.device,
                distribution: bool) -> tuple[np.ndarray, ...]:
    predictions: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    model.eval()
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _ = normalizer.preprocess(history, target)
        if distribution:
            point_norm, lower_norm, upper_norm = model.forward_distribution(
                history_norm)
        else:
            point_norm = model(history_norm)
        prediction = common.to_physical_target(
            point_norm, normalizer)[..., :2]
        predictions.append(prediction.float().cpu().numpy())
        targets.append(target[..., :2].float().cpu().numpy())
        masks.append((history[..., :2].abs().amax(dim=(1, 4)) > 1e-8
                      ).cpu().numpy())
        if distribution:
            lowers.append((lower_norm * scale).float().cpu().numpy())
            uppers.append((upper_norm * scale).float().cpu().numpy())
    result = [np.concatenate(predictions).astype(np.float32, copy=False)]
    if distribution:
        result.extend((
            np.concatenate(lowers).astype(np.float32, copy=False),
            np.concatenate(uppers).astype(np.float32, copy=False),
        ))
    result.extend((
        np.concatenate(targets).astype(np.float32, copy=False),
        np.concatenate(masks).astype(bool, copy=False),
    ))
    return tuple(result)


def collect_cache(args: argparse.Namespace, val_loader, normalizer,
                  device: torch.device) -> dict[str, np.ndarray]:
    arrays = []
    references = []
    metadata = []
    for index, checkpoint in enumerate(args.checkpoints):
        model, item = load_model(checkpoint, normalizer, device)
        collected = collect_one(
            model, val_loader, normalizer, device, distribution=index < 2)
        arrays.append(collected)
        references.append(collected[-2:])
        metadata.append(item)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for target, mask in references[1:]:
        if not np.array_equal(references[0][0], target):
            raise RuntimeError("target order changed between model evaluations")
        if not np.array_equal(references[0][1], mask):
            raise RuntimeError("valid mask order changed between evaluations")
    prediction_a, lower_a, upper_a, target, mask = arrays[0]
    prediction_b, lower_b, upper_b, _, _ = arrays[1]
    prediction_c, _, _ = arrays[2]
    result = {
        "prediction_a": prediction_a,
        "prediction_b": prediction_b,
        "prediction_c": prediction_c,
        "lower_a": lower_a,
        "upper_a": upper_a,
        "lower_b": lower_b,
        "upper_b": upper_b,
        "target": target,
        "valid_mask": mask,
        "model_metadata_json": np.asarray(json.dumps(metadata)),
    }
    if args.cache is not None:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, **result)
    return result


def load_or_collect(args, val_loader, normalizer, device) -> dict[str, np.ndarray]:
    if args.cache is not None and args.cache.is_file():
        with np.load(args.cache, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    return collect_cache(args, val_loader, normalizer, device)


def base_arrays(cache: dict[str, np.ndarray], weight_b: float,
                mean_weight: float) -> dict[str, np.ndarray]:
    prediction_a = cache["prediction_a"]
    prediction_b = cache["prediction_b"]
    prediction_c = cache["prediction_c"]
    base = prediction_a + np.float32(weight_b) * (prediction_b - prediction_a)
    mean_delta = (
        prediction_c.mean(axis=1, keepdims=True)
        - base.mean(axis=1, keepdims=True))
    prediction = base + np.float32(mean_weight) * mean_delta
    # ``collect_one`` stores positive physical extents, not absolute bounds.
    lower_a = cache["lower_a"]
    upper_a = cache["upper_a"]
    lower_b = cache["lower_b"]
    upper_b = cache["upper_b"]
    return {
        "prediction": prediction.astype(np.float32, copy=False),
        "lower": (lower_a + np.float32(weight_b) * (lower_b - lower_a)
                  ).astype(np.float32, copy=False),
        "upper": (upper_a + np.float32(weight_b) * (upper_b - upper_a)
                  ).astype(np.float32, copy=False),
        "target": cache["target"],
        "valid_mask": cache["valid_mask"],
        "prediction_a": prediction_a,
        "prediction_b": prediction_b,
        "prediction_c": prediction_c,
    }


def spatial_reallocate(native: np.ndarray, signal: np.ndarray,
                       valid_mask: np.ndarray, strength: float) -> np.ndarray:
    """Move a fixed per-sample/horizon/channel budget toward disagreement."""
    mask = valid_mask[:, None, :, :, None]
    masked_count = mask.sum(axis=(2, 3), keepdims=True).clip(min=1)
    mean = np.sum(signal, axis=(2, 3), keepdims=True, where=mask) / masked_count
    relative = signal / np.maximum(mean, np.float32(1e-6))
    logits = np.clip(np.float32(strength) * (relative - 1.0), -2.5, 2.5)
    allocation = np.exp(logits).astype(np.float32)
    denominator = np.sum(native, axis=(2, 3), keepdims=True, where=mask)
    changed = np.sum(
        native * allocation, axis=(2, 3), keepdims=True, where=mask)
    correction = changed / np.maximum(denominator, np.float32(1e-8))
    return native * allocation / np.maximum(correction, np.float32(1e-8))


def interval_for(base: dict[str, np.ndarray], method: str,
                 strength: float, c_weight: float) -> tuple[np.ndarray, np.ndarray]:
    prediction = base["prediction"]
    native_lower = base["lower"]
    native_upper = base["upper"]
    delta_a = base["prediction_a"] - prediction
    delta_b = base["prediction_b"] - prediction
    delta_c = base["prediction_c"] - prediction
    if method == "native":
        return native_lower, native_upper
    if method == "symmetric_reallocate":
        signal = (
            np.abs(delta_a - delta_b)
            + np.float32(c_weight) * np.abs(delta_c))
        total = spatial_reallocate(
            native_lower + native_upper, signal,
            base["valid_mask"], strength)
        ratio = native_lower / np.maximum(
            native_lower + native_upper, np.float32(1e-8))
        return total * ratio, total * (1.0 - ratio)
    low_signal = np.maximum.reduce((
        -delta_a, -delta_b, np.float32(c_weight) * -delta_c,
        np.zeros_like(delta_a)))
    up_signal = np.maximum.reduce((
        delta_a, delta_b, np.float32(c_weight) * delta_c,
        np.zeros_like(delta_a)))
    if method == "directional_reallocate":
        lower = spatial_reallocate(
            native_lower, low_signal + np.float32(1e-6),
            base["valid_mask"], strength)
        upper = spatial_reallocate(
            native_upper, up_signal + np.float32(1e-6),
            base["valid_mask"], strength)
        return lower, upper
    if method == "directional_add":
        return (
            native_lower + np.float32(strength) * low_signal,
            native_upper + np.float32(strength) * up_signal,
        )
    raise ValueError(f"unknown interval method {method}")


def subset(base: dict[str, np.ndarray], stride: int) -> dict[str, np.ndarray]:
    result = {}
    for key, value in base.items():
        if value.ndim == 5:
            result[key] = value[:, ::2, ::2, ::stride]
        elif key == "valid_mask":
            result[key] = value[:, ::2, ::stride]
        else:
            result[key] = value
    return result


def metrics(base: dict[str, np.ndarray], lower: np.ndarray,
            upper: np.ndarray, residual_scale: float = 1.0,
            sample_indices: np.ndarray | None = None) -> dict[str, float]:
    prediction = base["prediction"]
    target = base["target"]
    if sample_indices is not None:
        prediction = prediction[sample_indices]
        target = target[sample_indices]
        lower = lower[sample_indices]
        upper = upper[sample_indices]
    residual = (target - prediction) * np.float32(residual_scale)
    scored = target != 0.0
    inside = (residual >= -lower) & (residual <= upper)
    factor = sample_factor(prediction, target).reshape((-1, 1, 1, 1, 1))
    penalty = np.exp(-(lower + upper) / np.float32(OFFICIAL.SIGMA_GLOBAL))
    count = int(np.count_nonzero(scored))
    value = np.sum(factor * penalty * inside, where=scored,
                   dtype=np.float64) / max(count, 1)
    return {
        "sps_score": float(100.0 * value),
        "coverage": float(np.count_nonzero(inside & scored) / max(count, 1)),
        "mean_width": float(np.sum(
            lower + upper, where=scored, dtype=np.float64) / max(count, 1)),
        "scored_elements": count,
    }


def signal_diagnostics(base: dict[str, np.ndarray]) -> dict[str, float]:
    residual = np.abs(base["target"] - base["prediction"])
    missed = ((base["target"] - base["prediction"] < -base["lower"])
              | (base["target"] - base["prediction"] > base["upper"]))
    scored = base["target"] != 0.0
    signals = {
        "ab": np.abs(base["prediction_a"] - base["prediction_b"]),
        "c": np.abs(base["prediction_c"] - base["prediction"]),
    }
    result = {}
    sample = np.s_[::2, ::2, ::2, ::3, :]
    sampled_scored = scored[sample]
    sampled_residual = residual[sample][sampled_scored]
    sampled_missed = missed[sample][sampled_scored]
    for name, signal in signals.items():
        values = signal[sample][sampled_scored]
        result[f"{name}_residual_pearson"] = float(np.corrcoef(
            values, sampled_residual)[0, 1])
        result[f"{name}_mean_if_missed"] = float(values[sampled_missed].mean())
        result[f"{name}_mean_if_covered"] = float(values[~sampled_missed].mean())
        result[f"{name}_miss_ratio"] = (
            result[f"{name}_mean_if_missed"]
            / max(result[f"{name}_mean_if_covered"], 1e-12))
    return result


def grouped_metrics(base, lower, upper, labels, key: str):
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(labels):
        groups.setdefault(str(item[key]), []).append(index)
    return {
        value: metrics(
            base, lower, upper, sample_indices=np.asarray(indices, dtype=np.int64))
        for value, indices in sorted(groups.items(), key=lambda item: float(item[0]))
    }


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    train_loader, val_loader, normalizer, train_windows, val_windows, trajectories = (
        common.prepare_real(data_args, device))
    cache = load_or_collect(args, val_loader, normalizer, device)
    del train_loader, val_loader
    base = base_arrays(cache, args.weight_b, args.mean_expert_weight)
    sparse = subset(base, args.search_stride)

    candidates = [{"method": "native", "strength": 0.0, "c_weight": 0.0}]
    for method in ("symmetric_reallocate", "directional_reallocate"):
        for c_weight in (0.0, 0.25, 0.5, 1.0):
            for strength in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):
                candidates.append({
                    "method": method, "strength": strength,
                    "c_weight": c_weight})
    for c_weight in (0.0, 0.25, 0.5, 1.0):
        for strength in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            candidates.append({
                "method": "directional_add", "strength": strength,
                "c_weight": c_weight})

    search_rows = []
    for candidate in candidates:
        lower, upper = interval_for(sparse, **candidate)
        local = metrics(sparse, lower, upper)
        stress = metrics(sparse, lower, upper, residual_scale=1.3)
        row = {**candidate, "local": local, "stress_1.3": stress,
               "selection_score": local["sps_score"] + 0.35 * stress["sps_score"]}
        search_rows.append(row)
        print(json.dumps({"event": "search", **row}), flush=True)
    selected = sorted(
        search_rows, key=lambda row: row["selection_score"], reverse=True
    )[:args.finalists]
    if not any(row["method"] == "native" for row in selected):
        selected.append(search_rows[0])

    manifest = json.loads(
        (args.output.parent / "split_manifest.json").read_text(encoding="utf-8"))
    labels = [entry for entry in manifest for _ in entry["val_starts"]]
    if len(labels) != base["prediction"].shape[0]:
        raise RuntimeError("condition labels do not align with validation cache")
    exact_rows = []
    for candidate in selected:
        spec = {key: candidate[key] for key in ("method", "strength", "c_weight")}
        lower, upper = interval_for(base, **spec)
        scenarios = {
            f"residual_scale_{float(scale):.2f}": metrics(
                base, lower, upper, residual_scale=float(scale))
            for scale in args.residual_scales
        }
        row = {
            **spec,
            "scenarios": scenarios,
            "by_re": grouped_metrics(base, lower, upper, labels, "re"),
            "by_aoa": grouped_metrics(base, lower, upper, labels, "aoa"),
        }
        exact_rows.append(row)
        print(json.dumps({"event": "exact", **spec, **scenarios}), flush=True)

    baseline = next(row for row in exact_rows if row["method"] == "native")
    best_local = max(
        exact_rows,
        key=lambda row: row["scenarios"]["residual_scale_1.00"]["sps_score"])
    report = {
        "method": "mean_expert_disagreement_interval_audit",
        "checkpoints": [str(path) for path in args.checkpoints],
        "weight_b": args.weight_b,
        "mean_expert_weight": args.mean_expert_weight,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "signal_diagnostics": signal_diagnostics(base),
        "baseline": baseline,
        "best_local": best_local,
        "best_local_delta": (
            best_local["scenarios"]["residual_scale_1.00"]["sps_score"]
            - baseline["scenarios"]["residual_scale_1.00"]["sps_score"]),
        "search_rows": search_rows,
        "exact_finalists": exact_rows,
        "config": json.loads(json.dumps(vars(args), default=str)),
        "warning": (
            "The real-tail labels are used for this structural audit. Prefer "
            "stable gains across Re/AoA groups and residual-stress scenarios; "
            "do not interpret the best local row as hidden-set performance."),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "best_local_delta": report["best_local_delta"],
        "best_local": best_local, "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
