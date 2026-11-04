#!/usr/bin/env python3
"""Probe which low-precision inference paths the Track 1 FNO actually supports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("models/sim_real_ft/sim_real_fno_fp16.pth"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("realpde_precision_loader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def short_error(exc: Exception) -> str:
    return str(exc).splitlines()[0]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_forward(model: torch.nn.Module, x: torch.Tensor, device: torch.device,
                 warmup: int, iterations: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        synchronize(device)
        times = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(x)
            synchronize(device)
            times.append(time.perf_counter() - started)
    return times


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("batch-size/iterations must be positive and warmup non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This probe is intended for CUDA inference")

    kit_dir = args.kit_dir.resolve()
    sys.path.insert(0, str(kit_dir))
    baseline = load_module(kit_dir / "load_baseline.py")
    checkpoint = args.checkpoint.resolve()
    model, _ = baseline.load_baseline(checkpoint, device=str(device))
    model.eval()

    total_parameters = sum(p.numel() for p in model.parameters())
    linear_parameters = sum(
        p.numel()
        for module in model.modules() if isinstance(module, torch.nn.Linear)
        for p in module.parameters(recurse=False))
    complex_parameters = sum(p.numel() for p in model.parameters() if p.is_complex())
    result: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "checkpoint": str(checkpoint),
        "batch_size": args.batch_size,
        "shape": [args.batch_size, 20, 32, 64, 3],
        "parameters": {
            "total": total_parameters,
            "complex": complex_parameters,
            "linear": linear_parameters,
            "linear_percent": 100.0 * linear_parameters / total_parameters,
        },
        "optional_backends": {
            package: importlib.util.find_spec(package) is not None
            for package in ("torchao", "bitsandbytes", "torch_tensorrt", "tensorrt")
        },
    }

    x = torch.randn(args.batch_size, 20, 32, 64, 3, device=device)
    fp32_times = time_forward(model, x, device, args.warmup, args.iterations)
    result["fp32"] = {
        "status": "ok",
        "times_ms": [1000.0 * value for value in fp32_times],
        "median_ms": 1000.0 * sorted(fp32_times)[len(fp32_times) // 2],
    }

    try:
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16):
            model(x)
        synchronize(device)
        result["cuda_amp_fp16"] = {"status": "ok"}
    except Exception as exc:  # compatibility probe: preserve the backend error
        result["cuda_amp_fp16"] = {
            "status": "unsupported", "error_type": type(exc).__name__,
            "error": short_error(exc),
        }

    del model, x
    torch.cuda.empty_cache()

    try:
        cpu_model, _ = baseline.load_baseline(checkpoint, device="cpu")
        quantized = torch.ao.quantization.quantize_dynamic(
            cpu_model, {torch.nn.Linear}, dtype=torch.qint8)
        quantized_modules = [
            name for name, module in quantized.named_modules()
            if "quantized.dynamic" in type(module).__module__
        ]
        quantized = quantized.to(device)
        q_input = torch.randn(args.batch_size, 20, 32, 64, 3, device=device)
        with torch.inference_mode():
            quantized(q_input)
        synchronize(device)
        result["dynamic_int8_cuda"] = {
            "status": "ok", "quantized_modules": quantized_modules}
    except Exception as exc:  # compatibility probe: preserve the backend error
        result["dynamic_int8_cuda"] = {
            "status": "unsupported", "error_type": type(exc).__name__,
            "error": short_error(exc),
            "quantized_modules": locals().get("quantized_modules", []),
        }

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
