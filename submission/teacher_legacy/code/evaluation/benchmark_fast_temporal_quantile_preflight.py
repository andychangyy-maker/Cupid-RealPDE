#!/usr/bin/env python3
"""Benchmark the fixed S2 point, H96, direct-UQ, fused FTQ, or archive path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

PROCESS_STARTED = time.perf_counter()

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from evaluate_track1_submission_archive import (  # noqa: E402
    load_submission,
    normalized_result,
    safe_extract,
)
from models.distilled_interval_head import (  # noqa: E402
    DistilledIntervalHead,
    DistilledIntervalHeadConfig,
)
from models.fast_temporal_quantile_head import (  # noqa: E402
    FastTemporalQuantileHead,
    FastTemporalQuantileHeadConfig,
    fused_mean_point_heads,
    fused_mean_prediction,
)


MODEL_MODES = (
    "point-members", "point-fused", "h96", "direct", "ftq-fused")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODEL_MODES, "archive"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--example-h5", type=Path)
    parser.add_argument("--real-stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-repeats", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def locked_identity(path: Path, expected: str | None,
                    label: str) -> dict[str, object]:
    if expected is None:
        raise ValueError(f"{label} requires an expected SHA-256")
    identity = file_identity(path)
    if identity["sha256"] != expected.lower():
        raise ValueError(f"{label} SHA-256 differs from the frozen contract")
    return identity


def distribution_outputs(prediction: torch.Tensor, lower: torch.Tensor,
                         upper: torch.Tensor) -> tuple[torch.Tensor, ...]:
    prediction = prediction[..., :2]
    return prediction, prediction - lower, prediction + upper


def load_archive_input(path: Path, batch_size: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][:, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle["v"][:, ::2, ::2], dtype=np.float32)
    starts = [20 * index for index in range(batch_size)]
    if starts[-1] + 20 > len(u):
        raise ValueError(f"{path} has too few frames for batch size {batch_size}")
    uv = np.stack((u, v), axis=-1)
    uv_batch = np.stack([uv[start:start + 20] for start in starts])
    pressure = np.zeros((*uv_batch.shape[:-1], 1), dtype=np.float32)
    return np.concatenate((uv_batch, pressure), axis=-1)


def load_isolated_submission(package_dir: Path):
    """Load packaged model modules instead of cached repository modules."""
    for name in tuple(sys.modules):
        if name == "models" or name.startswith("models."):
            sys.modules.pop(name, None)
    return load_submission(package_dir)


def describe_outputs(outputs: tuple[torch.Tensor, ...]) -> dict[str, object]:
    names = ("prediction", "lower", "upper")[:len(outputs)]
    result = {}
    for name, value in zip(names, outputs):
        contiguous = value.detach().float().cpu().contiguous()
        result[name] = {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.numpy().dtype),
            "finite": bool(torch.isfinite(contiguous).all()),
            "sha256": hashlib.sha256(
                memoryview(contiguous.numpy()).cast("B")).hexdigest(),
        }
    if len(outputs) == 3:
        result["bounds_ordered"] = bool(torch.all(outputs[1] <= outputs[2]))
    return result


def timing_summary(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "values_ms": values,
        "median_ms": statistics.median(values),
        "minimum_ms": min(values),
        "p90_ms": ordered[p90_index],
    }


def cuda_event_timings(forward, repeats: int) -> tuple[list[float], tuple]:
    values = []
    outputs = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = forward()
        stop.record()
        stop.synchronize()
        values.append(float(start.elapsed_time(stop)))
    assert outputs is not None
    return values, outputs


def transfer_timings(outputs: tuple[torch.Tensor, ...],
                     repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        _host = [value.detach().float().cpu().numpy() for value in outputs]
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def end_to_end_timings(forward, repeats: int) -> tuple[list[float], tuple]:
    values = []
    outputs = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = forward()
        _host = [value.detach().float().cpu().numpy() for value in outputs]
        values.append((time.perf_counter() - started) * 1000.0)
    assert outputs is not None
    return values, outputs


def model_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint is None or args.real_stats is None:
        raise ValueError("model modes require checkpoint and real-stats")
    checkpoint_identity = locked_identity(
        args.checkpoint, args.expected_checkpoint_sha256, "checkpoint")
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=False).eval().half()
    fused_heads = fused_mean_point_heads(model)
    extra_module = None
    if args.mode == "direct":
        extra_module = DistilledIntervalHead(
            DistilledIntervalHeadConfig()).to(device).eval().half()
    elif args.mode == "ftq-fused":
        extra_module = FastTemporalQuantileHead(
            FastTemporalQuantileHeadConfig()).to(device).eval().half()

    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        args.batch_size, 20, 32, 64, 3, generator=generator,
        device=device, dtype=torch.float16)
    x[..., 2].zero_()

    def point_fused():
        features = model._decode(x)
        prediction = fused_mean_prediction(model, x, features, fused_heads)
        return (prediction[..., :2],)

    def point_members():
        return (model(x)[..., :2],)

    def h96():
        return distribution_outputs(*model.forward_distribution(x))

    def direct():
        features = model._decode(x)
        prediction = fused_mean_prediction(model, x, features, fused_heads)
        lower, upper = extra_module(
            x, prediction, features, model.target_std)
        return distribution_outputs(prediction, lower, upper)

    def ftq_fused():
        features = model._decode(x)
        prediction = fused_mean_prediction(model, x, features, fused_heads)
        _quantiles, lower, upper = extra_module(
            x, prediction, features, model.target_std)
        return distribution_outputs(prediction, lower, upper)

    forwards = {
        "point-fused": point_fused,
        "point-members": point_members,
        "h96": h96,
        "direct": direct,
        "ftq-fused": ftq_fused,
    }
    forward = forwards[args.mode]
    setup_seconds = time.perf_counter() - PROCESS_STARTED
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        torch.cuda.synchronize()
        first_started = time.perf_counter()
        first_outputs = forward()
        _first_host = [
            value.detach().float().cpu().numpy() for value in first_outputs]
        first_call_seconds = time.perf_counter() - first_started
        for _ in range(args.warmup_repeats):
            forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        cuda_values, outputs = cuda_event_timings(forward, args.repeats)
        transfer_values = transfer_timings(outputs, args.repeats)
        total_values, outputs = end_to_end_timings(forward, args.repeats)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    return {
        "evidence_class": "s2_structure_latency_preflight",
        "mode": args.mode,
        "checkpoint": checkpoint_identity,
        "real_stats": file_identity(args.real_stats),
        "model_config": payload["model_config"],
        "extra_config": (
            None if extra_module is None
            else extra_module.architecture_config()),
        "parameters": {
            "model": sum(parameter.numel() for parameter in model.parameters()),
            "extra": (
                0 if extra_module is None else sum(
                    parameter.numel() for parameter in extra_module.parameters())),
        },
        "startup_to_ready_seconds": setup_seconds,
        "first_call_seconds": first_call_seconds,
        "startup_to_first_output_seconds": setup_seconds + first_call_seconds,
        "forward_cuda": timing_summary(cuda_values),
        "device_to_host": timing_summary(transfer_values),
        "end_to_end": timing_summary(total_values),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "outputs": describe_outputs(outputs),
    }


def archive_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.archive is None or args.example_h5 is None:
        raise ValueError("archive mode requires --archive and --example-h5")
    archive_identity = locked_identity(
        args.archive, args.expected_archive_sha256, "archive")
    args.temp_root.mkdir(parents=True, exist_ok=True)
    x = load_archive_input(args.example_h5, args.batch_size)
    frozen_x = x.copy()
    with tempfile.TemporaryDirectory(
            prefix="s2-archive-", dir=args.temp_root) as temporary:
        package_dir = Path(temporary)
        safe_extract(args.archive, package_dir)
        submission = load_isolated_submission(package_dir)
        torch_module = getattr(submission, "torch", None)
        if torch_module is not None:
            torch_module.backends.cudnn.benchmark = False

        def synchronize() -> None:
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.synchronize()

        def forward():
            return tuple(normalized_result(
                submission.predict(x, metadata={})))

        setup_seconds = time.perf_counter() - PROCESS_STARTED
        synchronize()
        first_started = time.perf_counter()
        first_outputs = forward()
        synchronize()
        first_call_seconds = time.perf_counter() - first_started
        for _ in range(args.warmup_repeats):
            forward()
            synchronize()
        timings = []
        outputs = None
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            synchronize()
            started = time.perf_counter()
            outputs = forward()
            synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)
        assert outputs is not None
        arrays = tuple(torch.from_numpy(np.asarray(value)) for value in outputs)
        peak_allocated = (
            int(torch_module.cuda.max_memory_allocated())
            if torch_module is not None and torch_module.cuda.is_available()
            else 0)
        peak_reserved = (
            int(torch_module.cuda.max_memory_reserved())
            if torch_module is not None and torch_module.cuda.is_available()
            else 0)
    return {
        "evidence_class": "s2_deployment_archive_latency",
        "mode": "archive",
        "archive": archive_identity,
        "example_h5": file_identity(args.example_h5),
        "startup_to_ready_seconds": setup_seconds,
        "first_call_seconds": first_call_seconds,
        "startup_to_first_output_seconds": setup_seconds + first_call_seconds,
        "end_to_end": timing_summary(timings),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "execution_contract": {
            "execution_mode": "eager",
            "torch_compile_requested": False,
            "channels_last_requested": False,
            "cudnn_benchmark": (
                None if torch_module is None
                else bool(torch_module.backends.cudnn.benchmark)),
            "input_c_contiguous": bool(x.flags.c_contiguous),
            "input_pressure_zero": bool(np.count_nonzero(x[..., 2]) == 0),
            "input_bitwise_unchanged": bool(np.array_equal(x, frozen_x)),
        },
        "outputs": describe_outputs(arrays),
    }


def main() -> None:
    args = parse_args()
    if (args.batch_size <= 0 or args.warmup_repeats < 0
            or args.repeats <= 0):
        raise ValueError("batch/repeats must be positive and warmup non-negative")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("S2 latency preflight requires CUDA")
    torch.backends.cudnn.benchmark = False
    if args.mode == "archive":
        result = archive_benchmark(args)
    else:
        result = model_benchmark(args)
    result.update({
        "batch_size": args.batch_size,
        "warmup_repeats": args.warmup_repeats,
        "repeats": args.repeats,
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "torch_version": torch.__version__,
        "benchmark_source": file_identity(Path(__file__)),
        "ftq_source": file_identity(
            CODE_ROOT / "models/fast_temporal_quantile_head.py"),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "s2_latency_complete",
        "mode": args.mode,
        "end_to_end_median_ms": result["end_to_end"]["median_ms"],
        "output": file_identity(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
