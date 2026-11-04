#!/usr/bin/env python3
"""Target-visible smooth local-flow headroom audit on the frozen clean716 cache.

This is deliberately an oracle: the target chooses one spatially varying flow
per sample and five-frame horizon block.  It neither trains nor loads a model
and never writes a candidate prediction array.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
_INTEGER_PATH = REPO_ROOT / "code/evaluation/audit_b210_clean716_integer_transport_oracle.py"
_SPEC = importlib.util.spec_from_file_location("_integer_transport_audit", _INTEGER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_integer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_integer)

SCHEMA = "b210_clean716_smooth_local_flow_oracle_v1"
EVIDENCE_CLASS = "oracle"
EXPECTED_SHAPE = _integer.EXPECTED_SHAPE
HORIZON_BLOCKS = _integer.HORIZON_BLOCKS
WAKE_X_SLICE = _integer.WAKE_X_SLICE
POINT_WEIGHTS = _integer.POINT_WEIGHTS
CONTROL_GRID = (4, 8)
MAX_DISPLACEMENT_PIXELS = 1.5
TV_WEIGHT = 0.02
ADAM_STEPS = 200
ADAM_LR = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def displacement_from_raw(raw: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Upsample B,2,4,8 raw controls to bounded B,H,W,2 pixel displacement."""
    if raw.ndim != 4 or raw.shape[1:] != (2, *CONTROL_GRID):
        raise ValueError("raw flow must have shape (B,2,4,8)")
    if height < 2 or width < 2:
        raise ValueError("grid_sample requires height and width >= 2")
    flow = F.interpolate(raw, size=(height, width), mode="bilinear", align_corners=True)
    return (MAX_DISPLACEMENT_PIXELS * torch.tanh(flow)).permute(0, 2, 3, 1)


def smooth_flow_warp(source: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Warp B,T,H,W,2 using pixel displacement and border/bilinear sampling."""
    if source.ndim != 5 or source.shape[-1] != 2:
        raise ValueError("source must have shape (B,T,H,W,2)")
    batch, steps, height, width, channels = source.shape
    if displacement.shape != (batch, height, width, 2):
        raise ValueError("displacement must align with source spatial dimensions")
    # Preserve an exact identity path for non-differentiable callers/tests.  In
    # optimization displacement requires gradients, so the zero initialization
    # still traverses grid_sample and receives a displacement gradient.
    if not displacement.requires_grad and not bool(torch.count_nonzero(displacement)):
        return source.clone()
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype), indexing="ij")
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    # grid_sample maps output coordinates to input coordinates: positive flow
    # samples to the right/down.  This sign convention is fixed in config.
    grid_pixels = base + displacement.to(dtype=source.dtype)
    grid = torch.empty_like(grid_pixels)
    grid[..., 0] = 2.0 * grid_pixels[..., 0] / (width - 1) - 1.0
    grid[..., 1] = 2.0 * grid_pixels[..., 1] / (height - 1) - 1.0
    images = source.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width)
    grids = grid[:, None].expand(batch, steps, height, width, 2).reshape(
        batch * steps, height, width, 2)
    warped = F.grid_sample(images, grids, mode="bilinear", padding_mode="border",
                           align_corners=True)
    return warped.reshape(batch, steps, channels, height, width).permute(0, 1, 3, 4, 2)


def control_tv(raw: torch.Tensor) -> torch.Tensor:
    """Per-sample neighboring-control squared-difference penalty."""
    return ((raw[:, :, 1:, :] - raw[:, :, :-1, :]).square().mean((1, 2, 3))
            + (raw[:, :, :, 1:] - raw[:, :, :, :-1]).square().mean((1, 2, 3)))


def optimize_block_oracle(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize a bounded smooth flow, then select it only if it beats identity."""
    if source.shape != target.shape or source.ndim != 5 or source.shape[-1] != 2:
        raise ValueError("source and target must be aligned B,T,H,W,2 UV tensors")
    batch, _steps, height, width, _channels = source.shape
    raw = torch.zeros((batch, 2, *CONTROL_GRID), device=source.device,
                      dtype=source.dtype, requires_grad=True)
    identity_sse_raw = (source - target).square().sum((1, 2, 3, 4)).detach()
    identity_sse = identity_sse_raw.clamp_min(1.0e-12)
    optimizer = torch.optim.Adam((raw,), lr=ADAM_LR)
    for _ in range(ADAM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        displacement = displacement_from_raw(raw, height, width)
        warped = smooth_flow_warp(source, displacement)
        residual = (warped - target).square().sum((1, 2, 3, 4)) / identity_sse
        loss = (residual + TV_WEIGHT * control_tv(raw)).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        displacement = displacement_from_raw(raw, height, width)
        warped = smooth_flow_warp(source, displacement)
        flow_sse = (warped - target).square().sum((1, 2, 3, 4))
        # Compare against the un-clamped diagnostic SSE; the clamp is only a
        # safe denominator for the optimization objective.
        choose_flow = flow_sse < identity_sse_raw
        selected = torch.where(choose_flow[:, None, None, None, None], warped, source)
        selected_flow = torch.where(choose_flow[:, None, None, None], displacement,
                                   torch.zeros_like(displacement))
    return selected, choose_flow, selected_flow


def _flow_summary(flow: np.ndarray, chosen: np.ndarray) -> dict[str, float | int]:
    magnitude = np.linalg.norm(flow.astype(np.float64), axis=-1)
    return {
        "blocks": int(chosen.size),
        "identity_choices": int((~chosen).sum()),
        "identity_choice_fraction": float((~chosen).mean()),
        "selected_flow_mean_magnitude_pixels": float(magnitude.mean()),
        "selected_flow_max_magnitude_pixels": float(magnitude.max()),
    }


def _group_sse(baseline: np.ndarray, oracle: np.ndarray, rows: np.ndarray) -> dict:
    return _integer.sse_comparison(baseline[rows], oracle[rows])


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    if args.batch_size != 16:
        raise ValueError("the frozen oracle contract requires batch-size=16")
    output_names = ("config.json", "command.txt", "environment.txt", "result.json", "sha256.txt")
    if args.output_dir.exists() and any((args.output_dir / name).exists() for name in output_names):
        raise FileExistsError("refusing to overwrite an existing oracle audit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, metadata, _manifest = _integer.validate_inputs(args.cache_dir)
    device = _integer.resolve_device(args.device)
    prediction = np.load(paths["prediction"], mmap_mode="r")
    target = np.load(paths["target"], mmap_mode="r")
    aoa = np.load(paths["aoa"], mmap_mode="r")
    if (tuple(prediction.shape) != EXPECTED_SHAPE or tuple(target.shape) != EXPECTED_SHAPE
            or prediction.dtype != np.float32 or target.dtype != np.float32):
        raise ValueError("prediction/target arrays violate frozen clean716 contract")
    if tuple(aoa.shape) != (EXPECTED_SHAPE[0],):
        raise ValueError("AoA labels do not align with clean716")
    config = {
        "schema": SCHEMA, "evidence_class": EVIDENCE_CLASS, "target_visible": True,
        "deployable": False, "training_performed": False,
        "per_sample_target_optimization": True, "checkpoint_loaded": False,
        "large_prediction_written": False, "shape": list(EXPECTED_SHAPE),
        "cache_dir": str(args.cache_dir.resolve()), "batch_size": args.batch_size,
        "resolved_device": str(device), "horizon_blocks_1_based": [[a + 1, b] for a, b in HORIZON_BLOCKS],
        "control_grid": list(CONTROL_GRID), "max_displacement_pixels": MAX_DISPLACEMENT_PIXELS,
        "optimizer": {"name": "Adam", "steps": ADAM_STEPS, "lr": ADAM_LR},
        "objective": "per-sample/block UV SSE divided by detached identity SSE clamp(1e-12) + 0.02 control-grid TV",
        "tv_weight": TV_WEIGHT, "warp": {"mode": "bilinear", "padding_mode": "border", "align_corners": True},
        "selection": "per-sample/block minimum full UV SSE between optimized flow and exact identity",
        "wake_region": {"x_zero_based_half_open": list(WAKE_X_SLICE)}, "point_weights": POINT_WEIGHTS,
        "go_gate": "delta_point >= .20 OR wake_sse_reduction >= 15%; >=4/5 AoA positive; worst point >= -.03",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (args.output_dir / "environment.txt").write_text(_integer.environment_report(device))
    started = time.perf_counter(); samples = EXPECTED_SHAPE[0]
    base_errors = np.empty((samples, 3)); oracle_errors = np.empty_like(base_errors)
    base_sse = np.empty(samples); oracle_sse = np.empty(samples)
    base_wake = np.empty(samples); oracle_wake = np.empty(samples)
    base_h_errors = np.empty((samples, 4, 3)); oracle_h_errors = np.empty_like(base_h_errors)
    base_h_sse = np.empty((samples, 4)); oracle_h_sse = np.empty_like(base_h_sse)
    chosen = np.empty((samples, 4), dtype=bool); flows = np.empty((samples, 4, 32, 64, 2), dtype=np.float32)
    for start in range(0, samples, args.batch_size):
        stop = min(start + args.batch_size, samples)
        base = np.array(prediction[start:stop], copy=True); truth = np.array(target[start:stop], copy=True)
        if not (np.isfinite(base).all() and np.isfinite(truth).all()): raise FloatingPointError("non-finite cache")
        base_errors[start:stop] = _integer.official_error_triples(base, truth)
        candidate = base.copy()
        for block, (left, right) in enumerate(HORIZON_BLOCKS):
            selected, selected_choice, selected_flow = optimize_block_oracle(
                torch.from_numpy(base[:, left:right]).to(device), torch.from_numpy(truth[:, left:right]).to(device))
            candidate[:, left:right] = selected.cpu().numpy(); chosen[start:stop, block] = selected_choice.cpu().numpy()
            flows[start:stop, block] = selected_flow.cpu().numpy()
        oracle_errors[start:stop] = _integer.official_error_triples(candidate, truth)
        base_sse[start:stop] = _integer.per_sample_sse(base, truth); oracle_sse[start:stop] = _integer.per_sample_sse(candidate, truth)
        base_wake[start:stop] = _integer.per_sample_sse(base, truth, WAKE_X_SLICE); oracle_wake[start:stop] = _integer.per_sample_sse(candidate, truth, WAKE_X_SLICE)
        for block, (left, right) in enumerate(HORIZON_BLOCKS):
            base_h_errors[start:stop, block] = _integer.official_error_triples(base[:, left:right], truth[:, left:right])
            oracle_h_errors[start:stop, block] = _integer.official_error_triples(candidate[:, left:right], truth[:, left:right])
            base_h_sse[start:stop, block] = _integer.per_sample_sse(base[:, left:right], truth[:, left:right])
            oracle_h_sse[start:stop, block] = _integer.per_sample_sse(candidate[:, left:right], truth[:, left:right])
        if start == 0 or stop == samples or stop % 160 == 0:
            print(json.dumps({"event": "b210_smooth_flow_oracle_progress",
                              "samples_complete": stop,
                              "samples_total": samples,
                              "elapsed_seconds": time.perf_counter() - started}),
                  flush=True)
    tolerance = np.maximum(1.0, base_sse) * 1.0e-7
    if np.any(oracle_sse > base_sse + tolerance):
        raise RuntimeError("identity fallback failed to preserve per-sample UV SSE")
    comparison = _integer.metric_comparison(base_errors, oracle_errors)
    by_aoa = {}
    for value in sorted(np.unique(aoa).tolist()):
        rows = np.flatnonzero(aoa == value)
        by_aoa[str(int(value))] = {"official_point_metrics": _integer.metric_comparison(base_errors[rows], oracle_errors[rows]),
            "overall_sse": _group_sse(base_sse, oracle_sse, rows), "wake_sse": _group_sse(base_wake, oracle_wake, rows),
            "flow": _flow_summary(flows[rows], chosen[rows])}
    by_horizon = {f"{left + 1:02d}_{right:02d}": {"official_point_metrics": _integer.metric_comparison(base_h_errors[:, i], oracle_h_errors[:, i]), "sse": _integer.sse_comparison(base_h_sse[:, i], oracle_h_sse[:, i]), "flow": _flow_summary(flows[:, i:i+1], chosen[:, i:i+1])} for i, (left, right) in enumerate(HORIZON_BLOCKS)}
    delta = comparison["delta_oracle_minus_baseline"]["point_final_contribution_46_10_9"]
    deltas = [entry["official_point_metrics"]["delta_oracle_minus_baseline"]["point_final_contribution_46_10_9"] for entry in by_aoa.values()]
    wake = _integer.sse_comparison(base_wake, oracle_wake)["uv_sse_reduction_fraction"]
    go = (delta >= .20 or wake >= .15) and sum(item > 0 for item in deltas) >= 4 and min(deltas) >= -.03
    input_paths = {name: path for name, path in paths.items()
                   if name != "audit_script"}
    input_paths.update({"integer_audit_script": _INTEGER_PATH,
                        "smooth_flow_audit_script": Path(__file__).resolve()})
    result = {"schema": SCHEMA, "evidence_class": EVIDENCE_CLASS, "target_visible": True, "deployable": False,
        "training_performed": False, "per_sample_target_optimization": True,
        "checkpoint_loaded": False, "large_prediction_written": False,
        "config_canonical_sha256": _integer.canonical_sha256(config),
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                           "sha256": _hash(path)} for name, path in input_paths.items()},
        "source_archive": metadata.get("archive"), "overall": {"official_point_metrics": comparison, "overall_sse": _integer.sse_comparison(base_sse, oracle_sse), "wake_sse": _integer.sse_comparison(base_wake, oracle_wake), "flow": _flow_summary(flows, chosen)}, "by_aoa": by_aoa, "by_horizon": by_horizon,
        "decision": "GO_DIAGNOSTIC_SMOOTH_LOCAL_WARP" if go else "STOP_LEARNED_LOCAL_WARP", "go_checks": {"point_delta": delta, "wake_sse_reduction_fraction": wake, "positive_aoa": sum(item > 0 for item in deltas), "worst_aoa_point_delta": min(deltas)}}
    result_path = args.output_dir / "result.json"; result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    checksum_lines = [f"{_hash(path)}  input:{name}"
                      for name, path in input_paths.items()]
    checksum_lines += [f"{_hash(args.output_dir / name)}  output:{name}" for name in output_names[:-1]]
    (args.output_dir / "sha256.txt").write_text("\n".join(checksum_lines) + "\n")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"event": "b210_smooth_flow_oracle_complete", "decision": result["decision"], "point_delta": result["go_checks"]["point_delta"]}))


if __name__ == "__main__": main()
