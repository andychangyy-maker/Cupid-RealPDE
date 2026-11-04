#!/usr/bin/env python3
"""Attribute S2b archive tail latency to exact inference stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_track1_submission_archive import (  # noqa: E402
    load_submission,
    safe_extract,
)


STAGES = (
    "h2d",
    "normalize",
    "decode",
    "fused_point",
    "ftq_context",
    "ftq_stem",
    "ftq_blocks",
    "quantiles",
    "utility",
    "endpoint",
    "physical_output",
    "d2h",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--example-h5", type=Path, required=True)
    parser.add_argument("--expected-h5-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-repeats", type=int, default=20)
    parser.add_argument("--control-repeats", type=int, default=20)
    parser.add_argument("--profile-repeats", type=int, default=100)
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    parser.add_argument("--tail-factor", type=float, default=1.5)
    parser.add_argument("--tail-mad-multiplier", type=float, default=6.0)
    parser.add_argument("--attribution-fraction", type=float, default=0.70)
    parser.add_argument("--stage-slowdown", type=float, default=1.5)
    parser.add_argument("--max-profiler-median-drift", type=float, default=0.05)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def locked_identity(path: Path, expected: str, label: str) -> dict[str, object]:
    identity = file_identity(path)
    if identity["sha256"] != expected.lower():
        raise ValueError(f"{label} SHA-256 differs from the frozen contract")
    return identity


def load_input(path: Path, batch_size: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][:, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle["v"][:, ::2, ::2], dtype=np.float32)
    starts = [20 * index for index in range(batch_size)]
    if starts[-1] + 20 > len(u):
        raise ValueError("example H5 has too few frames")
    uv = np.stack((u, v), axis=-1)
    uv_batch = np.stack([uv[start:start + 20] for start in starts])
    pressure = np.zeros((*uv_batch.shape[:-1], 1), dtype=np.float32)
    return np.concatenate((uv_batch, pressure), axis=-1)


def timing_summary(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "values_ms": values,
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "minimum_ms": min(values),
        "p90_ms": ordered[p90_index],
        "maximum_ms": max(values),
    }


def median_abs_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def fast_mode_threshold(values: list[float], factor: float,
                        mad_multiplier: float) -> tuple[float, float, float]:
    ordered = sorted(values)
    lower = ordered[:max(1, len(ordered) // 2)]
    baseline = statistics.median(lower)
    mad = median_abs_deviation(lower)
    return max(factor * baseline, baseline + mad_multiplier * mad), baseline, mad


def control_call(submission, input_array: np.ndarray) -> tuple[np.ndarray, ...]:
    result = submission.predict(input_array, metadata={})
    return (
        np.asarray(result["prediction"]),
        np.asarray(result["lower"]),
        np.asarray(result["upper"]),
    )


def control_timings(submission, input_array: np.ndarray,
                    repeats: int) -> tuple[list[float], tuple[np.ndarray, ...]]:
    values = []
    outputs = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = control_call(submission, input_array)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    assert outputs is not None
    return values, outputs


@torch.inference_mode()
def staged_call(submission, state: dict[str, object], input_array: np.ndarray,
                events: list[torch.cuda.Event],
                ) -> tuple[dict[str, float], tuple[np.ndarray, ...]]:
    torch.cuda.synchronize()
    wall_started = time.perf_counter()
    events[0].record()
    submit_boundaries = [time.perf_counter_ns()]

    raw_input = np.asarray(input_array, dtype=np.float32)
    raw = torch.from_numpy(np.ascontiguousarray(raw_input)).to(
        device=state["device"], dtype=state["dtype"])
    events[1].record()
    submit_boundaries.append(time.perf_counter_ns())
    normalized = (raw - state["mean_input"]) / state["std_input"]
    events[2].record()
    submit_boundaries.append(time.perf_counter_ns())
    features = state["model"]._decode(normalized)
    events[3].record()
    submit_boundaries.append(time.perf_counter_ns())
    prediction_norm = submission.fused_mean_prediction(
        state["model"], normalized, features, state["fused_heads"])
    events[4].record()
    submit_boundaries.append(time.perf_counter_ns())
    context = state["head"]._project_context(
        normalized, prediction_norm, features)
    events[5].record()
    submit_boundaries.append(time.perf_counter_ns())
    hidden = (
        state["head"].stem(context)
        + state["head"].horizon_embedding.to(context.dtype))
    events[6].record()
    submit_boundaries.append(time.perf_counter_ns())
    hidden = state["head"].blocks(hidden)
    events[7].record()
    submit_boundaries.append(time.perf_counter_ns())
    quantiles = state["head"]._monotone_quantiles(hidden)
    events[8].record()
    submit_boundaries.append(time.perf_counter_ns())
    lower_extent, upper_extent = state["head"]._utility_interval(
        quantiles, state["model"].target_std)
    events[9].record()
    submit_boundaries.append(time.perf_counter_ns())
    lower_extent, upper_extent = state["head"].endpoint_refinement(
        lower_extent, upper_extent)
    lower_extent, upper_extent = state["head"]._ordered_extents(
        lower_extent, upper_extent)
    events[10].record()
    submit_boundaries.append(time.perf_counter_ns())

    prediction = prediction_norm * state["std_target"] + state["mean_target"]
    prediction[..., 2] = 0.0
    lower = prediction.clone()
    upper = prediction.clone()
    lower[..., :2] = (
        prediction[..., :2] - lower_extent * state["std_target"][:2])
    upper[..., :2] = (
        prediction[..., :2] + upper_extent * state["std_target"][:2])
    bundle = torch.stack((prediction, lower, upper)).float()
    events[11].record()
    submit_boundaries.append(time.perf_counter_ns())
    host = bundle.cpu().numpy()
    events[12].record()
    submit_boundaries.append(time.perf_counter_ns())
    events[12].synchronize()
    finalize_started = time.perf_counter()
    outputs = tuple(
        np.concatenate([host[index]]).astype(np.float32, copy=False)
        for index in range(3))
    host_finalize_ms = (time.perf_counter() - finalize_started) * 1000.0
    wall_ms = (time.perf_counter() - wall_started) * 1000.0

    values = {
        name: float(events[index].elapsed_time(events[index + 1]))
        for index, name in enumerate(STAGES)
    }
    values.update({
        f"host_submit_{name}": (
            submit_boundaries[index + 1] - submit_boundaries[index]) / 1e6
        for index, name in enumerate(STAGES)
    })
    values["cuda_sum"] = sum(values[name] for name in STAGES)
    values["host_residual"] = wall_ms - values["cuda_sum"]
    values["host_finalize"] = host_finalize_ms
    values["wall_total"] = wall_ms
    return values, outputs


def output_contract(outputs: tuple[np.ndarray, ...]) -> dict[str, object]:
    prediction, lower, upper = outputs
    return {
        "shapes": [list(value.shape) for value in outputs],
        "dtypes": [str(value.dtype) for value in outputs],
        "finite": bool(all(np.isfinite(value).all() for value in outputs)),
        "bounds_ordered": bool(np.all(lower <= upper)),
        "pressure_zero": bool(all(np.all(value[..., 2] == 0.0) for value in outputs)),
    }


def attribution(rows: list[dict[str, float]], args: argparse.Namespace,
                control_values: list[float]) -> dict[str, object]:
    wall_values = [row["wall_total"] for row in rows]
    cuda_values = [row["cuda_sum"] for row in rows]
    wall_median = statistics.median(wall_values)
    cuda_median = statistics.median(cuda_values)
    wall_threshold, wall_fast_baseline, wall_fast_mad = fast_mode_threshold(
        wall_values, args.tail_factor, args.tail_mad_multiplier)
    cuda_threshold, cuda_fast_baseline, cuda_fast_mad = fast_mode_threshold(
        cuda_values, args.tail_factor, args.tail_mad_multiplier)
    wall_slow_indices = {
        index for index, row in enumerate(rows)
        if row["wall_total"] > wall_threshold
    }
    cuda_slow_indices = {
        index for index, row in enumerate(rows)
        if row["cuda_sum"] > cuda_threshold
    }
    cuda_slow_pairs = [
        (index, rows[index]) for index in sorted(cuda_slow_indices)]
    cuda_slow = [row for _index, row in cuda_slow_pairs]
    cuda_fast = [
        row for index, row in enumerate(rows) if index not in cuda_slow_indices]
    control_median = statistics.median(control_values)
    profiler_drift = abs(wall_median / control_median - 1.0)
    profiler_valid = profiler_drift <= args.max_profiler_median_drift

    stage_summaries = {}
    aggregate_excess = {name: 0.0 for name in STAGES}
    cuda_excess = 0.0
    tail_culprits = []
    if cuda_fast:
        baselines = {
            name: statistics.median([row[name] for row in cuda_fast])
            for name in STAGES
        }
        fast_cuda = statistics.median([row["cuda_sum"] for row in cuda_fast])
        for profile_index, row in cuda_slow_pairs:
            cuda_excess += max(0.0, row["cuda_sum"] - fast_cuda)
            row_excess = {
                name: max(0.0, row[name] - baselines[name])
                for name in STAGES
            }
            for name in STAGES:
                aggregate_excess[name] += row_excess[name]
            culprit = max(STAGES, key=row_excess.__getitem__)
            total = sum(row_excess.values())
            tail_culprits.append({
                "profile_index": profile_index,
                "cuda_sum_ms": row["cuda_sum"],
                "wall_total_ms": row["wall_total"],
                "culprit": culprit,
                "culprit_excess_fraction": (
                    0.0 if total <= 0.0 else row_excess[culprit] / total),
            })
        for name in STAGES:
            fast_values = [row[name] for row in cuda_fast]
            slow_values = [row[name] for row in cuda_slow]
            slow_median = statistics.median(slow_values) if slow_values else None
            fast_median = statistics.median(fast_values)
            stage_summaries[name] = {
                "fast_median_ms": fast_median,
                "slow_median_ms": slow_median,
                "slow_over_fast": (
                    None if slow_median is None or fast_median <= 0.0
                    else slow_median / fast_median),
                "aggregate_cuda_slow_excess_ms": aggregate_excess[name],
                "cuda_excess_fraction": (
                    0.0 if cuda_excess <= 0.0
                    else aggregate_excess[name] / cuda_excess),
                "host_submit_fast_median_ms": statistics.median([
                    row[f"host_submit_{name}"] for row in cuda_fast]),
                "host_submit_slow_median_ms": (
                    None if not cuda_slow else statistics.median([
                        row[f"host_submit_{name}"] for row in cuda_slow])),
            }
    dominant = (
        None if not cuda_slow
        else max(STAGES, key=aggregate_excess.__getitem__))
    dominant_fraction = (
        0.0 if dominant is None or cuda_excess <= 0.0
        else aggregate_excess[dominant] / cuda_excess)
    dominant_ratio = (
        None if dominant is None
        else stage_summaries[dominant]["slow_over_fast"])
    dominant_tail_count = sum(
        item["culprit"] == dominant for item in tail_culprits)
    dominant_summary = (
        None if dominant is None else stage_summaries[dominant])
    dominant_host_submit_tail = bool(
        dominant_summary is not None and dominant != "d2h"
        and dominant_summary["host_submit_slow_median_ms"] is not None
        and dominant_summary["host_submit_fast_median_ms"] > 0.0
        and dominant_summary["host_submit_slow_median_ms"]
        / dominant_summary["host_submit_fast_median_ms"] >= 1.5
        and dominant_summary["host_submit_slow_median_ms"]
        - dominant_summary["host_submit_fast_median_ms"] >= 1.0)
    globally_slow_stages = [
        name for name, summary in stage_summaries.items()
        if summary["fast_median_ms"] >= 0.05
        and summary["slow_over_fast"] is not None
        and summary["slow_over_fast"] >= 1.5
    ]
    global_cuda_slowdown = len(globally_slow_stages) >= 3
    gate_pass = bool(
        profiler_valid and len(cuda_slow) >= 2 and dominant_tail_count >= 2
        and dominant_fraction >= args.attribution_fraction
        and dominant_ratio is not None
        and dominant_ratio >= args.stage_slowdown
        and not dominant_host_submit_tail and not global_cuda_slowdown)
    host_only_tails = wall_slow_indices - cuda_slow_indices
    if dominant_host_submit_tail:
        classification = "host_submit_or_runtime_stall"
    elif global_cuda_slowdown:
        classification = "global_cuda_slowdown"
    elif gate_pass:
        classification = "dominant_cuda_stage"
    elif wall_slow_indices and not cuda_slow_indices:
        classification = "host_runtime_or_scheduling"
    elif cuda_slow_indices:
        classification = "global_or_mixed_cuda_slowdown"
    else:
        classification = "no_tail_observed"
    return {
        "wall_tail_threshold_ms": wall_threshold,
        "cuda_tail_threshold_ms": cuda_threshold,
        "wall_fast_baseline_ms": wall_fast_baseline,
        "wall_fast_mad_ms": wall_fast_mad,
        "cuda_fast_baseline_ms": cuda_fast_baseline,
        "cuda_fast_mad_ms": cuda_fast_mad,
        "wall_slow_calls": len(wall_slow_indices),
        "cuda_slow_calls": len(cuda_slow_indices),
        "host_only_slow_calls": len(host_only_tails),
        "wall_slow_fraction": len(wall_slow_indices) / len(rows),
        "cuda_slow_fraction": len(cuda_slow_indices) / len(rows),
        "profile_wall": timing_summary(wall_values),
        "profile_cuda_sum": timing_summary(cuda_values),
        "host_residual": timing_summary([
            row["host_residual"] for row in rows]),
        "control_median_ms": control_median,
        "profiler_median_drift_fraction": profiler_drift,
        "profiler_valid": profiler_valid,
        "cuda_slow_excess_ms": cuda_excess,
        "dominant_stage": dominant,
        "dominant_cuda_excess_fraction": dominant_fraction,
        "dominant_slow_over_fast": dominant_ratio,
        "dominant_tail_count": dominant_tail_count,
        "dominant_host_submit_tail": dominant_host_submit_tail,
        "globally_slow_stages": globally_slow_stages,
        "tail_culprits": tail_culprits,
        "classification": classification,
        "stage_summaries": stage_summaries,
        "gate_pass": gate_pass,
    }


def main() -> None:
    args = parse_args()
    if (args.batch_size <= 0 or args.warmup_repeats < 0
            or args.control_repeats <= 0 or args.profile_repeats <= 0
            or args.batch_size > 32):
        raise ValueError("repeat and batch counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("stage attribution requires CUDA")
    archive_identity = locked_identity(
        args.archive, args.expected_archive_sha256, "archive")
    h5_identity = locked_identity(
        args.example_h5, args.expected_h5_sha256, "example H5")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    input_array = np.ascontiguousarray(load_input(args.example_h5, args.batch_size))
    input_sha_before = hashlib.sha256(memoryview(input_array).cast("B")).hexdigest()

    with tempfile.TemporaryDirectory(
            prefix="s2b-stage-", dir=args.temp_root) as temporary:
        package = Path(temporary)
        safe_extract(args.archive, package)
        for name in tuple(sys.modules):
            if name == "models" or name.startswith("models."):
                sys.modules.pop(name, None)
        submission = load_submission(package)
        state = submission._get_model()
        for _ in range(args.warmup_repeats):
            control_call(submission, input_array)
        before, outputs = control_timings(
            submission, input_array, args.control_repeats)
        events = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(len(STAGES) + 1)]
        _discard_values, _discard_outputs = staged_call(
            submission, state, input_array, events)
        rows = []
        profile_outputs = None
        for _ in range(args.profile_repeats):
            values, profile_outputs = staged_call(
                submission, state, input_array, events)
            rows.append(values)
        after, outputs = control_timings(
            submission, input_array, args.control_repeats)
        assert profile_outputs is not None

    control_values = before + after
    parity = bool(all(
        np.array_equal(profile, control)
        for profile, control in zip(profile_outputs, outputs)))
    attribution_result = attribution(rows, args, control_values)
    attribution_result["gate_pass"] = bool(
        attribution_result["gate_pass"] and parity)
    result = {
        "evidence_class": "s2b_stage_tail_attribution",
        "archive": archive_identity,
        "example_h5": h5_identity,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "profiler_source": file_identity(Path(__file__)),
        "protocol": {
            "batch_size": args.batch_size,
            "warmup_repeats": args.warmup_repeats,
            "control_repeats_before": args.control_repeats,
            "profile_repeats": args.profile_repeats,
            "control_repeats_after": args.control_repeats,
            "tail_factor": args.tail_factor,
            "tail_mad_multiplier": args.tail_mad_multiplier,
            "attribution_fraction": args.attribution_fraction,
            "stage_slowdown": args.stage_slowdown,
            "max_profiler_median_drift": args.max_profiler_median_drift,
        },
        "control": {
            "before": timing_summary(before),
            "after": timing_summary(after),
            "pooled": timing_summary(control_values),
        },
        "profile_rows": rows,
        "attribution": attribution_result,
        "output_contract": output_contract(outputs),
        "profile_matches_control": parity,
        "input_unchanged": input_sha_before == hashlib.sha256(
            memoryview(input_array).cast("B")).hexdigest(),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "s2b_stage_attribution_complete",
        "output": file_identity(args.output),
        "wall_slow_calls": result["attribution"]["wall_slow_calls"],
        "cuda_slow_calls": result["attribution"]["cuda_slow_calls"],
        "dominant_stage": result["attribution"]["dominant_stage"],
        "dominant_cuda_excess_fraction": result["attribution"][
            "dominant_cuda_excess_fraction"],
        "gate_pass": result["attribution"]["gate_pass"],
    }), flush=True)


if __name__ == "__main__":
    main()
