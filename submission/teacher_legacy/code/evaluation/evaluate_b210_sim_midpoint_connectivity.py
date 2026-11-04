#!/usr/bin/env python3
"""Check a fixed 50/50 B210 weight midpoint on the frozen Sim anchor."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for source in (CODE_ROOT, TRAINING_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import train_b210_np_functional_sim as b210  # noqa: E402
import train_c208_sim_hgrad_pretrain as sim_hgrad  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


SCHEMA = "b210_sim_midpoint_connectivity_v1"
ALPHA_B = 0.50
ANCHOR_SIZE = 128
POINT_FLOOR = -0.20
COMPONENT_FLOOR = -0.35
COMPONENTS = ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")


def mixed_state(first: Mapping[str, torch.Tensor],
                second: Mapping[str, torch.Tensor],
                alpha_b: float = ALPHA_B,
                ) -> OrderedDict[str, torch.Tensor]:
    if not math.isfinite(alpha_b) or not 0.0 <= alpha_b <= 1.0:
        raise ValueError("midpoint alpha must be finite and lie in [0,1]")
    if tuple(first) != tuple(second):
        raise ValueError("checkpoint model-state keys or order differ")
    result = OrderedDict()
    for name in first:
        left, right = first[name], second[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"model state {name} is not a tensor")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"checkpoint tensor mismatch for {name}")
        if torch.is_floating_point(left) or torch.is_complex(left):
            result[name] = torch.lerp(left.float(), right.float(), alpha_b).to(
                dtype=left.dtype)
        elif torch.equal(left, right):
            result[name] = left.detach().clone()
        else:
            raise ValueError(f"non-floating model state differs for {name}")
    return result


def connectivity_gate(first: Mapping[str, float],
                      second: Mapping[str, float],
                      midpoint: Mapping[str, float]) -> dict:
    point_reference = min(float(first["point_score"]),
                          float(second["point_score"]))
    component_references = {
        key: min(float(first[key]), float(second[key])) for key in COMPONENTS}
    checks = {
        "finite": all(math.isfinite(float(report[key]))
                      for report in (first, second, midpoint)
                      for key in (*COMPONENTS, "point_score")),
        "point_connected": (
            float(midpoint["point_score"]) - point_reference >= POINT_FLOOR),
        "components_connected": all(
            float(midpoint[key]) - component_references[key]
            >= COMPONENT_FLOOR for key in COMPONENTS),
    }
    return {
        "pass": all(checks.values()),
        "decision": (
            "PASS_B210_SIM_MIDPOINT_FOR_REAL4K"
            if all(checks.values())
            else "STOP_B210_SIM_MIDPOINT_DISCONNECTED"),
        "alpha_b": ALPHA_B,
        "thresholds": {
            "point_delta_from_worse_endpoint": POINT_FLOOR,
            "component_delta_from_worse_endpoint": COMPONENT_FLOOR,
        },
        "references": {
            "point_score": point_reference, **component_references},
        "midpoint_delta_from_worse_endpoint": {
            key: float(midpoint[key]) - (
                point_reference if key == "point_score"
                else component_references[key])
            for key in (*COMPONENTS, "point_score")
        },
        "checks": checks,
    }


def load_payload(path: Path) -> dict:
    payload = torch.load(str(path), map_location="cpu", weights_only=False,
                         mmap=True)
    if (not isinstance(payload, dict) or payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != b210.TERMINAL_UPDATE
            or payload.get("metrics", {}).get("status") != b210.PASS_STATUS
            or not isinstance(payload.get("model_state_dict"), Mapping)
            or not isinstance(payload.get("model_config"), Mapping)):
        raise ValueError(f"{path} is not a passed B210 Sim10k checkpoint")
    return payload


def parse_config(payload: Mapping[str, object]) -> EnsembleUNet2dConfig:
    stored = dict(payload["model_config"])
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError("checkpoint architecture is not ensemble_unet")
    return EnsembleUNet2dConfig(**stored)


def normalizer_matches(stored: Mapping[str, object], runtime) -> bool:
    runtime_values = common.normalizer_dict(runtime)
    if set(stored) != set(runtime_values):
        return False
    return all(torch.allclose(
        torch.as_tensor(stored[key], dtype=torch.float64),
        torch.as_tensor(runtime_values[key], dtype=torch.float64),
        rtol=0.0, atol=1e-7) for key in stored)


@torch.inference_mode()
def evaluate_state(config: EnsembleUNet2dConfig,
                   state: Mapping[str, torch.Tensor], loader,
                   normalizer, device: torch.device) -> dict[str, float]:
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    common.set_model_stats(model, normalizer)
    model.to(device)
    metrics = b210.rehearsal.ensemble_train.evaluate(
        model, loader, sim_hgrad.SimEvaluationNormalizer(normalizer), device,
        b210.rehearsal.SIM_ACCURACY_WEIGHTS)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {key: float(value) for key, value in metrics.items()}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    first = load_payload(args.checkpoint_a)
    second = load_payload(args.checkpoint_b)
    if first["normalizer"] != second["normalizer"]:
        raise ValueError("checkpoint Sim normalizers differ")
    config = parse_config(first)
    second_config = parse_config(second)
    if config != second_config:
        raise ValueError("checkpoint model configs differ after defaults")
    device = torch.device(args.device)
    trajectories, windows, _manifest = common.scratch.load_simulations(
        args.sim_dir, 0, 20, True)
    statistics = common.scratch.window_statistics(trajectories, windows)
    dataset = common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    normalizer = common.scratch.ChannelNormalizer(*statistics, device=device)
    if not normalizer_matches(first["normalizer"], normalizer):
        raise RuntimeError("runtime Sim normalizer differs from checkpoints")
    loader = DataLoader(
        Subset(dataset, range(ANCHOR_SIZE)), batch_size=16, shuffle=False,
        num_workers=0)
    state_a = first["model_state_dict"]
    state_b = second["model_state_dict"]
    midpoint_state = mixed_state(state_a, state_b)
    reports = {
        "seed0": evaluate_state(config, state_a, loader, normalizer, device),
        "seed1": evaluate_state(config, state_b, loader, normalizer, device),
        "midpoint": evaluate_state(
            config, midpoint_state, loader, normalizer, device),
    }
    gate = connectivity_gate(
        reports["seed0"], reports["seed1"], reports["midpoint"])
    result = {
        "schema": SCHEMA,
        "checkpoints": {
            "seed0": {"path": str(args.checkpoint_a.resolve()),
                      "sha256": b210.sim_hgrad.sha256(args.checkpoint_a)},
            "seed1": {"path": str(args.checkpoint_b.resolve()),
                      "sha256": b210.sim_hgrad.sha256(args.checkpoint_b)},
        },
        "anchor_samples": ANCHOR_SIZE,
        "reports": reports,
        "gate": gate,
        "real_labels_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "event": "b210_sim_midpoint_connectivity_complete",
        "reports": result["reports"], "gate": result["gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
