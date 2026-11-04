#!/usr/bin/env python3
"""Evaluate same-architecture U-Net/DiT checkpoint interpolations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from train_flow_backbone import construct_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def mixed_state(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor],
                alpha: float) -> dict[str, torch.Tensor]:
    if a.keys() != b.keys():
        raise ValueError("checkpoint state dictionaries have different keys")
    result = {}
    for name in a:
        left, right = a[name], b[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"checkpoint tensor mismatch for {name}")
        if torch.is_floating_point(left) or torch.is_complex(left):
            result[name] = left + alpha * (right - left)
        else:
            result[name] = right if alpha >= 0.5 else left
    return result


def main() -> None:
    args = parse_args()
    if (args.eval_batch_size <= 0 or not args.alphas
            or any(not math.isfinite(value) or not 0 <= value <= 1
                   for value in args.alphas)):
        raise ValueError("batch size and alpha values must be valid")
    device = torch.device(args.device)
    payload_a = torch.load(
        args.checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.checkpoint_b, map_location="cpu", weights_only=False)
    if payload_a["model_config"] != payload_b["model_config"]:
        raise ValueError("checkpoint architecture configs differ")
    architecture = payload_a["model_config"]["architecture"]
    state_a = payload_a["model_state_dict"]
    state_b = payload_b["model_state_dict"]
    model = construct_model(architecture, payload_a["model_config"]).to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output_dir, seed=0)
    (_train_loader, val_loader, normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)
    common.set_model_stats(model, normalizer)

    results = {}
    best_alpha = None
    best_point = -float("inf")
    for alpha in args.alphas:
        state = mixed_state(state_a, state_b, float(alpha))
        model.load_state_dict(state, strict=True)
        metrics = common.repro.evaluate(model, val_loader, normalizer, device)
        common.add_weighted(metrics, (5.0, 3.0, 2.0))
        metrics.update(common.evaluate_native_sps(
            model, val_loader, normalizer, device))
        key = f"{float(alpha):.3f}"
        results[key] = metrics
        print(json.dumps({"event": "soup", "alpha": alpha,
                          **metrics}), flush=True)
        if metrics["point_score"] > best_point:
            best_point = metrics["point_score"]
            best_alpha = float(alpha)
            torch.save({
                **payload_b,
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in state.items()},
                "phase": "real",
                "soup": {
                    "checkpoint_a": str(args.checkpoint_a),
                    "checkpoint_b": str(args.checkpoint_b),
                    "alpha": best_alpha,
                },
            }, args.output_dir / "best.pth")
    report = {
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "alpha_definition": "state=(1-alpha)*a+alpha*b",
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "best_alpha": best_alpha,
        "best_point_score": best_point,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"event": "complete", "best_alpha": best_alpha,
                      "best_point_score": best_point}), flush=True)


if __name__ == "__main__":
    main()
