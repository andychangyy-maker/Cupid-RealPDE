#!/usr/bin/env python3
"""Compare FP32 and FP16 inference on the standard Real temporal-tail split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from train_flow_backbone import construct_model  # noqa: E402


class InputCast(nn.Module):
    """Accept FP32 normalized inputs while running the wrapped model in FP16."""

    def __init__(self, model: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.config = model.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.dtype)).float()

    def forward_distribution(self, x: torch.Tensor):
        prediction, lower, upper = self.model.forward_distribution(
            x.to(self.dtype))
        return prediction.float(), lower.float(), upper.float()


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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    return parser.parse_args()


def evaluate(model: nn.Module, loader, normalizer, device) -> dict:
    metrics = common.repro.evaluate(model, loader, normalizer, device)
    common.add_weighted(metrics, (1.0, 4.0, 1.0))
    metrics.update(common.evaluate_native_sps(
        model, loader, normalizer, device))
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FP16 comparison requires CUDA")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = payload["model_config"].get("architecture")
    if architecture not in {"unet", "dit", "ensemble_unet"}:
        raise ValueError(f"unsupported architecture: {architecture}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    (_train_loader, val_loader, normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)

    fp32 = construct_model(architecture, payload["model_config"]).to(device)
    fp32.load_state_dict(payload["model_state_dict"], strict=True)
    common.set_model_stats(fp32, normalizer)
    fp32.eval()
    fp32_metrics = evaluate(fp32, val_loader, normalizer, device)
    del fp32
    torch.cuda.empty_cache()

    fp16_core = construct_model(architecture, payload["model_config"]).to(
        device=device, dtype=torch.float16)
    fp16_core.load_state_dict(payload["model_state_dict"], strict=True)
    common.set_model_stats(fp16_core, normalizer)
    fp16 = InputCast(fp16_core.eval(), torch.float16)
    fp16_metrics = evaluate(fp16, val_loader, normalizer, device)

    keys = (
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "point_score", "native_sps_score", "native_sps_coverage",
        "native_sps_mean_width", "eval_seconds")
    report = {
        "checkpoint": str(args.checkpoint),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "fp32": {key: fp32_metrics[key] for key in keys},
        "fp16": {key: fp16_metrics[key] for key in keys},
        "fp16_minus_fp32": {
            key: fp16_metrics[key] - fp32_metrics[key] for key in keys},
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
