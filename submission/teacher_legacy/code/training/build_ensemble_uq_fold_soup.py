#!/usr/bin/env python3
"""Average condition-fold UQ decoder weights onto one frozen point model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_paths = sorted(args.fold_dir.glob("fold_*_best.pth"))
    if len(fold_paths) < 2:
        raise FileNotFoundError(
            f"need at least two fold checkpoints in {args.fold_dir}")
    base = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    states = []
    fold_metrics = []
    for path in fold_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        states.append(payload["uq_state_dict"])
        fold_metrics.append(payload.get("metrics", {}))
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise RuntimeError("fold UQ state dictionaries have different keys")
    averaged = {}
    for key in keys:
        values = [state[key] for state in states]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise RuntimeError(f"fold tensor shape changed for {key}")
        if torch.is_floating_point(values[0]):
            averaged[key] = torch.stack(
                [value.float() for value in values]).mean(dim=0).to(
                    dtype=values[0].dtype)
        else:
            if any(not torch.equal(value, values[0]) for value in values[1:]):
                raise RuntimeError(f"non-floating fold tensor changed for {key}")
            averaged[key] = values[0].clone()
    model_state = {
        key: value.detach().cpu().clone()
        for key, value in base["model_state_dict"].items()
    }
    model_state.update(averaged)
    output = {
        "model_state_dict": model_state,
        "model_config": base["model_config"],
        "phase": "real",
        "iteration": 0,
        "normalizer": base.get("normalizer", {}),
        "metrics": {},
        "training_config": {
            "method": "condition_fold_uq_weight_soup",
            "point_checkpoint": str(args.point_checkpoint),
            "fold_checkpoints": [str(path) for path in fold_paths],
            "fold_metrics": fold_metrics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({
        "event": "complete", "output": str(args.output),
        "folds": len(fold_paths), "uq_tensors": len(averaged),
        "parameters": sum(value.numel() for value in averaged.values()),
        "size_bytes": args.output.stat().st_size,
    }), flush=True)


if __name__ == "__main__":
    main()
