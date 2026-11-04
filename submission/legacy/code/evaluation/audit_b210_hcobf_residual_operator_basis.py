#!/usr/bin/env python3
"""CPU-only structural audit for HCOB-F's point/local/global operator basis.

This tool never imports a model or opens a checkpoint.  It consumes aligned,
already materialized [N,T,H,W,2] prediction/target arrays, and optionally
the corresponding [N,20,H,W,>=2] histories plus trajectory IDs.  Its output
is a target-visible diagnosis, *not* an endpoint or model-selection metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--history", type=Path,
                        help="Aligned cached histories; enables history audit.")
    parser.add_argument("--trajectory", type=Path,
                        help="Aligned trajectory IDs; required with --history.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1.0,
                        help="Fixed standardized ridge penalty; never tuned.")
    return parser.parse_args()


def bands(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return non-overlapping radial low/mid/high masks and rFFT weights."""
    fy = np.fft.fftfreq(height)[:, None] / 0.5
    fx = np.fft.rfftfreq(width)[None, :] / 0.5
    radius = np.sqrt(fy * fy + fx * fx)
    # Fixed relative-to-Nyquist cutoffs; they are not fitted on the audit set.
    masks = np.stack((radius <= 1.0 / 3.0,
                      (radius > 1.0 / 3.0) & (radius <= 2.0 / 3.0),
                      radius > 2.0 / 3.0))
    weights = np.ones_like(radius)
    if width > 2:
        weights[:, 1:-1 if width % 2 == 0 else None] = 2.0
    return masks, weights


def spectrum_energy(value: np.ndarray, masks: np.ndarray,
                    weights: np.ndarray) -> np.ndarray:
    """Per-sample spatial spectral energy [N,3], Parseval-correct for rFFT."""
    # value: [N,T,H,W,C] or [N,H,W,C], orthonormal FFT preserves L2 energy.
    if value.ndim == 4:
        value = value[:, None]
    coeff = np.fft.rfft2(value, axes=(-3, -2), norm="ortho")
    power = (coeff.real * coeff.real + coeff.imag * coeff.imag)
    weighted = power * weights[None, None, :, :, None]
    return np.asarray([weighted[:, :, mask, :].sum(axis=(1, 2, 3))
                       for mask in masks], dtype=np.float64).T


def region_masks(history: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed, exclusive 32x64 taxonomy used in earlier B210 region audits."""
    last = history[:, -1, ..., :2]
    valid = np.max(np.abs(last), axis=-1) > 1.0e-8
    n, height, width = valid.shape
    yy, xx = np.ogrid[:height, :width]
    boundary = ((yy == 0) | (yy == height - 1) | (xx == 0) | (xx == width - 1))
    inlet = (xx < 8) & ~boundary
    near_body_wake = ((xx >= 8) & (xx < 32) & (yy >= 8) & (yy < 24)
                      & ~boundary)
    tail_wake = ((xx >= 32) & (yy >= 8) & (yy < 24) & ~boundary)
    masks = {
        "实体/固体": ~valid,
        "外边界皮层": valid & boundary,
        "入流带(x<8，去边界)": valid & inlet,
        "主体/近尾迹带": valid & near_body_wake,
        "尾迹带": valid & tail_wake,
    }
    allocated = np.zeros((n, height, width), dtype=bool)
    for mask in masks.values():
        allocated |= mask
    masks["其余自由流"] = valid & ~allocated
    if not np.all(sum(mask.astype(np.int8) for mask in masks.values()) == 1):
        raise RuntimeError("region masks are not exclusive/exhaustive")
    return masks


def ridge_loto(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
               penalty: float) -> tuple[np.ndarray, dict[str, float]]:
    """Leave-one-trajectory-out fixed-ridge probe; no split is selected."""
    prediction = np.empty_like(y, dtype=np.float64)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        mean, std = x[train].mean(0), x[train].std(0)
        std = np.maximum(std, 1.0e-8)
        train_x = (x[train] - mean) / std
        test_x = (x[test] - mean) / std
        design = np.c_[np.ones(train_x.shape[0]), train_x]
        test_design = np.c_[np.ones(test_x.shape[0]), test_x]
        regularizer = np.eye(design.shape[1]) * penalty
        regularizer[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + regularizer, design.T @ y[train])
        prediction[test] = test_design @ beta
    centered = y - y.mean(0)
    denominator = np.sum(centered * centered, axis=0)
    r2 = 1.0 - np.sum((prediction - y) ** 2, axis=0) / np.maximum(denominator, 1e-30)
    corr = []
    for column in range(y.shape[1]):
        corr.append(float(np.corrcoef(prediction[:, column], y[:, column])[0, 1])
                    if np.std(prediction[:, column]) > 0 and np.std(y[:, column]) > 0
                    else float("nan"))
    return prediction, {"r2": [float(v) for v in r2],
                        "pearson_r": corr}


def main() -> None:
    cfg = args()
    if cfg.chunk_size <= 0 or cfg.ridge < 0:
        raise ValueError("chunk-size must be positive and ridge non-negative")
    if (cfg.history is None) != (cfg.trajectory is None):
        raise ValueError("--history and --trajectory must be provided together")
    pred = np.load(cfg.prediction, mmap_mode="r")
    target = np.load(cfg.target, mmap_mode="r")
    if pred.shape != target.shape or pred.ndim != 5 or pred.shape[-1] != 2:
        raise ValueError("prediction/target must be matching [N,T,H,W,2] arrays")
    history = np.load(cfg.history, mmap_mode="r") if cfg.history else None
    groups = np.load(cfg.trajectory, mmap_mode="r") if cfg.trajectory else None
    if history is not None and (history.shape[0] != pred.shape[0] or history.ndim != 5
                                or history.shape[-1] < 2 or groups.shape != (pred.shape[0],)):
        raise ValueError("history/trajectory are not aligned with prediction")
    n, steps, height, width, _ = pred.shape
    masks, fft_weights = bands(height, width)
    names = ("low", "mid", "high")
    totals = {"total": 0.0, "temporal_mean": 0.0, "fluctuation": 0.0}
    spectral = {"total": np.zeros(3), "fluctuation": np.zeros(3)}
    region_total: dict[str, float] = {}
    sample_mean = np.zeros(n)
    sample_fluct = np.zeros(n)
    sample_band = np.zeros((n, 3))
    sample_region: dict[str, np.ndarray] = {}
    summary = np.zeros((n, 80)) if history is not None else None
    for start in range(0, n, cfg.chunk_size):
        stop = min(start + cfg.chunk_size, n)
        residual = np.asarray(target[start:stop], dtype=np.float64) - np.asarray(pred[start:stop], dtype=np.float64)
        mean = residual.mean(axis=1, keepdims=True)
        fluct = residual - mean
        total_energy = np.sum(residual * residual, axis=(1, 2, 3, 4))
        mean_energy = np.sum(mean * mean, axis=(1, 2, 3, 4)) * steps
        fluct_energy = np.sum(fluct * fluct, axis=(1, 2, 3, 4))
        totals["total"] += float(total_energy.sum())
        totals["temporal_mean"] += float(mean_energy.sum())
        totals["fluctuation"] += float(fluct_energy.sum())
        sample_mean[start:stop], sample_fluct[start:stop] = mean_energy, fluct_energy
        total_spec = spectrum_energy(residual, masks, fft_weights)
        fluct_spec = spectrum_energy(fluct, masks, fft_weights)
        spectral["total"] += total_spec.sum(axis=0)
        spectral["fluctuation"] += fluct_spec.sum(axis=0)
        sample_band[start:stop] = fluct_spec
        if history is not None:
            history_chunk = np.asarray(history[start:stop, ..., :2], dtype=np.float64)
            summary[start:stop] = np.concatenate((
                history_chunk.mean(axis=(2, 3)),
                history_chunk.std(axis=(2, 3))), axis=-1).reshape(stop - start, -1)
            for name, mask in region_masks(history_chunk).items():
                energy = np.sum(residual * residual * mask[:, None, :, :, None], axis=(1, 2, 3, 4))
                region_total[name] = region_total.get(name, 0.0) + float(energy.sum())
                sample_region.setdefault(name, np.zeros(n))[start:stop] = energy
    out: dict[str, object] = {
        "schema": "b210_hcobf_residual_operator_basis_audit_v1",
        "evidence_class": "target-visible clean716 structural audit only; no endpoint, hyperparameter, or generalization claim",
        "source": {"prediction": str(cfg.prediction), "target": str(cfg.target),
                   "shape": list(pred.shape), "cpu_only": True},
        "definitions": {
            "temporal": "r=target-prediction; mean is each sample/pixel/channel's 20-step mean, fluct=r-mean",
            "spatial_frequency": "orthonormal rFFT2, radial normalized-Nyquist bands: low<=1/3, mid=(1/3,2/3], high>2/3",
            "regions": "fixed 32x64 B210 geometry taxonomy; exclusive; not a learned segmentation",
            "operator_proxy": "argmax of fluctuation high/mid/low energy maps to pointwise/local/global only as a necessary spectral proxy, not an observed optimal HCOB-F gate",
        },
        "energy_share": {key: float(value / totals["total"]) for key, value in totals.items()},
        "frequency_energy_share": {kind: {name: float(value / values.sum())
                                           for name, value in zip(names, values)}
                                  for kind, values in spectral.items()},
    }
    if history is not None:
        out["source"]["history"] = str(cfg.history)
        out["source"]["trajectory"] = str(cfg.trajectory)
        out["region_energy_share"] = {name: float(value / totals["total"])
                                      for name, value in region_total.items()}
        # The stipulated 80-D HCOB-F summary: per time-step, u/v spatial mean/std.
        targets = np.c_[sample_mean / np.maximum(sample_mean + sample_fluct, 1e-30),
                        sample_band / np.maximum(sample_band.sum(1, keepdims=True), 1e-30),
                        np.stack([sample_region[name] for name in region_total], axis=1)
                        / np.maximum(sample_mean[:, None] + sample_fluct[:, None], 1e-30)]
        prediction, probe = ridge_loto(summary, targets, np.asarray(groups), cfg.ridge)
        labels = ("pointwise(high)", "local(mid)", "global(low)")
        # Reorder frequency columns [low,mid,high] into the declared proxy order.
        proxy_truth = np.argmax(sample_band[:, ::-1], axis=1)
        proxy_pred = np.argmax(prediction[:, 1:4][:, ::-1], axis=1)
        recalls = [float(np.mean(proxy_pred[proxy_truth == i] == i)) if np.any(proxy_truth == i) else float("nan")
                   for i in range(3)]
        out["trajectory_held_out_history_probe"] = {
            "protocol": "leave-one-trajectory-out; fixed ridge=1.0 on standardized 80-D mean/std history; no clean716 selection",
            "target_columns": ["temporal_mean_share", "fluct_low_share", "fluct_mid_share", "fluct_high_share"]
                              + ["region_" + name for name in region_total],
            **probe,
            "spectral_proxy_operator": {
                "labels": labels, "majority_fraction": float(np.max(np.bincount(proxy_truth, minlength=3)) / n),
                "accuracy": float(np.mean(proxy_pred == proxy_truth)),
                "class_recall": dict(zip(labels, recalls)),
                "warning": "This tests only whether history predicts a residual spectral category. It cannot establish an optimal HCOB-F operator without trained aligned/permuted gates and outcomes.",
            },
        }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
