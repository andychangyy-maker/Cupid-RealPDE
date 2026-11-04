#!/usr/bin/env python3
"""Explore causal, label-free Track 1 post-processing on public validation windows.

The transforms in this file use only the 20 input frames and the model's 20
predicted frames.  Targets are used exclusively for evaluation.  This makes the
same implementation eligible for the hidden-input submission wrapper and keeps
the experiment reproducible for final organizer retraining.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_repro_spec = importlib.util.spec_from_file_location(
    "realpde_reproduce_finetune_postprocess",
    REPO_ROOT / "code/training/reproduce_finetune.py",
)
if _repro_spec is None or _repro_spec.loader is None:
    raise ImportError("Cannot import code/training/reproduce_finetune.py")
_repro = importlib.util.module_from_spec(_repro_spec)
sys.modules[_repro_spec.name] = _repro
_repro_spec.loader.exec_module(_repro)
Normalizer = _repro.Normalizer
WindowDataset = _repro.WindowDataset
load_baseline_module = _repro.load_baseline_module
load_trajectories = _repro.load_trajectories

_score_spec = importlib.util.spec_from_file_location(
    "realpde_official_scoring_postprocess",
    REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py",
)
if _score_spec is None or _score_spec.loader is None:
    raise ImportError("Cannot import Track 1 scoring.py")
_scoring = importlib.util.module_from_spec(_score_spec)
_score_spec.loader.exec_module(_scoring)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/sim_real_ft/sim_real_fno_fp16.pth"))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--stats", type=Path, default=Path(
        "kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument("--kit-dir", type=Path,
                        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true",
                        help="Evaluate only the most promising fluctuation/continuity grid.")
    return parser.parse_args()


def validation_windows(manifest_path: Path, filenames: list[str]):
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_file = {entry["file"]: entry for entry in manifest}
    if set(by_file) != set(filenames):
        raise ValueError("Split manifest and real-data filenames do not match")
    windows = [
        _repro.Window(index, int(start))
        for index, filename in enumerate(filenames)
        for start in by_file[filename]["val_starts"]
    ]
    return windows


def predict_fno(model, loader, normalizer: Normalizer, device: torch.device):
    inputs: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader, start=1):
            x_device = x.to(device, non_blocking=True)
            x_norm = (x_device - normalizer.mean_input) / normalizer.std_input
            pred = normalizer.postprocess_prediction(model(x_norm)).float().cpu().numpy()
            pred[..., 2] = 0.0
            inputs.append(x.numpy())
            predictions.append(pred)
            targets.append(y.numpy())
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(json.dumps({"event": "inference", "batch": batch_index,
                                  "total_batches": len(loader)}), flush=True)
    return tuple(np.concatenate(items).astype(np.float32, copy=False)
                 for items in (inputs, predictions, targets))


def score_accuracy(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    channels = _scoring.measured_channels(target)
    rel = float(np.mean(_scoring.rel_l2_per_sample(prediction, target, channels)))
    tke = float(np.mean(_scoring.tke_rel_l2_per_sample(prediction, target, channels)))
    mvpe = float(_scoring.mvpe_rel_l2(prediction, target))
    scores = {
        "rel_l2_score": _scoring.score_error(rel),
        "tke_score": _scoring.score_error(tke),
        "mvpe_score": _scoring.score_error(mvpe),
    }
    scores["accuracy_mean"] = float(np.mean(list(scores.values())))
    return {
        "raw_rel_l2": rel,
        "raw_tke_rel_l2": tke,
        "raw_mvpe_rel_l2": mvpe,
        **scores,
    }


def repeat_last(x: np.ndarray) -> np.ndarray:
    result = np.repeat(x[:, -1:], 20, axis=1).copy()
    result[..., 2] = 0.0
    return result


def repeat_window(x: np.ndarray) -> np.ndarray:
    result = x.copy()
    result[..., 2] = 0.0
    return result


def harmonic_forecast(x: np.ndarray, per_channel: bool) -> np.ndarray:
    """Stable sinusoidal extrapolation with frequency estimated from input only."""
    uv = x[..., :2].astype(np.float64, copy=False)
    mean = np.mean(uv, axis=1, keepdims=True)
    centered = uv - mean
    middle = centered[:, 1:-1]
    numerator = np.sum((centered[:, 2:] + centered[:, :-2]) * middle,
                       axis=(1, 2, 3), keepdims=True)
    denominator = 2.0 * np.sum(middle * middle, axis=(1, 2, 3), keepdims=True)
    cosine = numerator / np.maximum(denominator, 1e-12)
    if not per_channel:
        cosine = np.mean(cosine, axis=4, keepdims=True)
    cosine = np.clip(cosine, -0.999, 0.999)
    history = [centered[:, index] for index in range(centered.shape[1])]
    future = []
    coefficient = 2.0 * cosine[:, 0]
    for _ in range(20):
        next_frame = coefficient * history[-1] - history[-2]
        history.append(next_frame)
        future.append(next_frame)
    uv_prediction = np.stack(future, axis=1) + mean
    result = np.zeros((*uv_prediction.shape[:-1], 3), dtype=np.float32)
    result[..., :2] = uv_prediction.astype(np.float32)
    return result


def moment_match(x: np.ndarray, prediction: np.ndarray, mean_weight: float,
                 energy_weight: float, max_scale: float) -> np.ndarray:
    """Match observed temporal mean/TKE without reading future targets."""
    observed = x[..., :2].astype(np.float64, copy=False)
    predicted = prediction[..., :2].astype(np.float64, copy=False)
    observed_mean = np.mean(observed, axis=1, keepdims=True)
    predicted_mean = np.mean(predicted, axis=1, keepdims=True)
    observed_centered = observed - observed_mean
    predicted_centered = predicted - predicted_mean

    observed_energy = np.mean(np.sum(observed_centered ** 2, axis=-1),
                              axis=1, keepdims=True)[..., None]
    predicted_energy = np.mean(np.sum(predicted_centered ** 2, axis=-1),
                               axis=1, keepdims=True)[..., None]
    scale = np.sqrt((observed_energy + 1e-12) / (predicted_energy + 1e-12))
    scale = np.clip(scale, 1.0 / max_scale, max_scale)
    applied_scale = 1.0 + energy_weight * (scale - 1.0)
    adjusted_mean = predicted_mean + mean_weight * (observed_mean - predicted_mean)
    uv = adjusted_mean + applied_scale * predicted_centered
    result = prediction.copy()
    result[..., :2] = uv.astype(np.float32)
    result[..., 2] = 0.0
    return result


def continuity_correct(x: np.ndarray, prediction: np.ndarray, weight: float,
                       decay_power: float) -> np.ndarray:
    """Reduce the input/output boundary jump with a decaying causal offset."""
    result = prediction.copy()
    jump = x[:, -1:, ..., :2] - prediction[:, :1, ..., :2]
    phase = np.linspace(1.0, 0.0, prediction.shape[1], dtype=np.float32)
    ramp = np.power(phase, decay_power).reshape(1, -1, 1, 1, 1)
    result[..., :2] += weight * ramp * jump
    result[..., 2] = 0.0
    return result


def scale_predicted_fluctuations(prediction: np.ndarray, scale: float) -> np.ndarray:
    """Apply a constant scale to temporal fluctuations, preserving their mean."""
    result = prediction.copy()
    mean = np.mean(result[..., :2], axis=1, keepdims=True)
    result[..., :2] = mean + scale * (result[..., :2] - mean)
    result[..., 2] = 0.0
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = torch.device(args.device)
    trajectories, metadata = load_trajectories(args.real_dir, 0)
    filenames = [item["file"] for item in metadata]
    windows = validation_windows(args.split_manifest, filenames)
    dataset = WindowDataset(trajectories, windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    normalizer = Normalizer(args.stats, device)
    baseline_loader = load_baseline_module(args.kit_dir)
    model, checkpoint_metadata = baseline_loader.load_baseline(
        "fno", str(args.checkpoint), device=str(device))
    x, fno, target = predict_fno(model, loader, normalizer, device)
    del model, trajectories
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Store factories rather than all candidate tensors. One 725-window tensor is
    # roughly 350 MiB, so eager materialization wastes tens of GiB without helping
    # score throughput.
    candidates: list[tuple[str, object]] = [("fno", lambda: fno)]
    if not args.quick:
        candidates.extend([
            ("persistence", lambda: repeat_last(x)),
            ("repeat_input_window", lambda: repeat_window(x)),
            ("harmonic_shared", lambda: harmonic_forecast(x, per_channel=False)),
            ("harmonic_per_channel", lambda: harmonic_forecast(x, per_channel=True)),
        ])
        for copy_weight in (0.1, 0.25, 0.5, 0.75):
            candidates.append((
                f"fno_repeat_blend_{copy_weight:g}",
                lambda weight=copy_weight: (
                    (1.0 - weight) * fno + weight * repeat_window(x)
                ).astype(np.float32),
            ))
    scales = ((1.0, 1.05, 1.1, 1.15, 1.2) if args.quick
              else (0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3))
    for scale in scales:
        if scale == 1.0:
            continue
        candidates.append((
            f"fno_fluctuation_scale_{scale:g}",
            lambda value=scale: scale_predicted_fluctuations(fno, value),
        ))
    decay_powers = (4.0,) if args.quick else (0.5, 1.0, 2.0, 4.0)
    for weight in (0.25, 0.5, 0.75, 1.0):
        for decay_power in decay_powers:
            candidates.append((
                f"fno_continuity_weight{weight:g}_power{decay_power:g}",
                lambda value=weight, power=decay_power: continuity_correct(
                    x, fno, value, power),
            ))
    for scale in (1.05, 1.1, 1.15):
        for weight in (0.25, 0.5, 0.75):
            name = f"fno_scale{scale:g}_continuity{weight:g}_power4"
            candidates.append((
                name,
                lambda scale_value=scale, weight_value=weight: continuity_correct(
                    x, scale_predicted_fluctuations(fno, scale_value),
                    weight_value, 4.0),
            ))
    if not args.quick:
        for mean_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            for energy_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                if mean_weight == 0.0 and energy_weight == 0.0:
                    continue
                name = f"fno_moment_mean{mean_weight:g}_energy{energy_weight:g}"
                candidates.append((
                    name,
                    lambda mean_value=mean_weight, energy_value=energy_weight: moment_match(
                        x, fno, mean_weight=mean_value, energy_weight=energy_value,
                        max_scale=2.0),
                ))

    results = {}
    started = time.time()
    for index, (name, factory) in enumerate(candidates, start=1):
        prediction = factory()
        metrics = score_accuracy(prediction, target)
        results[name] = metrics
        print(json.dumps({"event": "score", "candidate": name, **metrics}), flush=True)
        if prediction is not fno:
            del prediction
        print(json.dumps({"event": "progress", "candidate": index,
                          "total_candidates": len(candidates)}), flush=True)

    ranked = sorted(results, key=lambda name: results[name]["accuracy_mean"], reverse=True)
    factories = dict(candidates)
    sample_aoa = np.asarray([metadata[item.trajectory]["aoa"] for item in windows])
    grouped_by_aoa = {}
    for candidate_name in dict.fromkeys(("fno", ranked[0])):
        prediction = factories[candidate_name]()
        grouped_by_aoa[candidate_name] = {
            str(aoa): {
                "samples": int(np.sum(sample_aoa == aoa)),
                **score_accuracy(prediction[sample_aoa == aoa],
                                 target[sample_aoa == aoa]),
            }
            for aoa in sorted(set(sample_aoa.tolist()))
        }
        if prediction is not fno:
            del prediction
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "split_manifest": str(args.split_manifest),
        "samples": int(target.shape[0]),
        "selection_note": "Targets are used only by score_accuracy, never by a transform.",
        "ranked_candidates": ranked,
        "results": results,
        "grouped_by_aoa": grouped_by_aoa,
        "score_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"event": "complete", "output": str(args.output),
                      "top": [{"name": name, **results[name]} for name in ranked[:10]]}),
          flush=True)


if __name__ == "__main__":
    main()
