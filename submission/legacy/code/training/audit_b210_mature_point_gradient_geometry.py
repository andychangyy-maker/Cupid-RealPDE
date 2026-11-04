#!/usr/bin/env python3
"""Audit mature B210 Rel/TKE/MVPE output-gradient geometry read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_fno_sim_from_scratch as scratch
import train_hybrid_fno2d as common
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig


ANCHOR_SAMPLES = 128
OFFICIAL_WEIGHTS = (46.0, 10.0, 9.0)
LEGACY_WEIGHTS = (1.0, 4.0, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_sample_cosine(
        left: torch.Tensor, right: torch.Tensor,
        ) -> torch.Tensor:
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("gradient tensors must have equal batched shapes")
    left = left.float().reshape(left.shape[0], -1)
    right = right.float().reshape(right.shape[0], -1)
    numerator = (left * right).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(left, dim=1)
        * torch.linalg.vector_norm(right, dim=1)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    result = numerator / denominator
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite gradient cosine")
    return result


def legacy_tke_floor_geometry(
        official: torch.Tensor, legacy: torch.Tensor,
        tke: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
    """Measure the exact TKE-parallel correction missing from official."""
    if official.shape != legacy.shape or official.shape != tke.shape:
        raise ValueError("floor gradients must have equal shapes")
    shape = official.shape
    official = official.float().reshape(shape[0], -1)
    legacy = legacy.float().reshape(shape[0], -1)
    tke = tke.float().reshape(shape[0], -1)
    tke_norm = torch.linalg.vector_norm(tke, dim=1).clamp_min(
        torch.finfo(torch.float32).tiny)
    unit = tke / tke_norm[:, None]
    official_parallel = (official * unit).sum(dim=1)
    legacy_parallel = (legacy * unit).sum(dim=1).clamp_min(0.0)
    missing = (legacy_parallel - official_parallel).clamp_min(0.0)
    official_norm = torch.linalg.vector_norm(official, dim=1).clamp_min(
        torch.finfo(torch.float32).tiny)
    return {
        "official_parallel": official_parallel,
        "legacy_parallel": legacy_parallel,
        "missing_parallel": missing,
        "correction_to_official_norm": missing / official_norm,
        "applied": missing > 0.0,
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("gradient summary requires finite observations")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "samples": len(values),
        "negative": sum(value < 0.0 for value in values),
        "negative_fraction": sum(value < 0.0 for value in values) / len(values),
        "minimum": min(values),
        "q10": float(torch.quantile(tensor, 0.10)),
        "median": float(torch.quantile(tensor, 0.50)),
        "q90": float(torch.quantile(tensor, 0.90)),
        "maximum": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_model_and_normalizer(
        checkpoint: Path, device: torch.device,
        ) -> tuple[EnsembleUNet2d, scratch.ChannelNormalizer, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("phase") != "sim":
        raise ValueError("mature gradient audit requires a Sim checkpoint")
    stored = dict(payload["model_config"])
    if stored.pop("architecture") != "ensemble_unet":
        raise ValueError("checkpoint is not an ensemble U-Net")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    stats = payload["normalizer"]
    normalizer = scratch.ChannelNormalizer(
        torch.as_tensor(stats["mean_input"]),
        torch.as_tensor(stats["mean_target"]),
        torch.as_tensor(stats["std_input"]),
        torch.as_tensor(stats["std_target"]), device)
    return model, normalizer, payload


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = args.checkpoint_sha256.lower()
    observed = sha256(args.checkpoint)
    if observed != expected:
        raise RuntimeError("checkpoint SHA-256 differs")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("mature gradient audit requires CUDA")
    torch.set_float32_matmul_precision("high")
    model, normalizer, payload = load_model_and_normalizer(
        args.checkpoint, device)
    trajectories, windows, manifest = scratch.load_simulations(
        args.sim_dir, max_files=0, stride=20, exclude_final=True)
    dataset = scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    if len(dataset) < ANCHOR_SAMPLES:
        raise RuntimeError("Sim dataset is smaller than fixed anchor")
    loader = DataLoader(
        Subset(dataset, range(ANCHOR_SAMPLES)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=True)
    geometry: dict[str, list[float]] = {
        "rel_vs_tke": [],
        "mvpe_vs_tke": [],
        "legacy_main_vs_tke": [],
        "official_main_vs_tke": [],
    }
    floor_geometry: dict[str, list[float]] = {
        "official_parallel": [],
        "legacy_parallel": [],
        "missing_parallel": [],
        "correction_to_official_norm": [],
        "applied": [],
    }
    component_losses = {"rel": 0.0, "tke": 0.0, "mvpe": 0.0}
    seen = 0
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        prediction = model(history_norm)
        losses = []
        for weights in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0)):
            loss, _details = common.accuracy_objective(
                prediction, target_norm, normalizer, weights)
            losses.append(loss)
        gradients = [
            torch.autograd.grad(
                loss, prediction, retain_graph=index < 2,
                allow_unused=False)[0]
            for index, loss in enumerate(losses)
        ]
        rel_gradient, tke_gradient, mvpe_gradient = gradients
        legacy_gradient = (
            LEGACY_WEIGHTS[0] * rel_gradient
            + LEGACY_WEIGHTS[1] * tke_gradient
            + LEGACY_WEIGHTS[2] * mvpe_gradient
        ) / sum(LEGACY_WEIGHTS)
        official_gradient = (
            OFFICIAL_WEIGHTS[0] * rel_gradient
            + OFFICIAL_WEIGHTS[1] * tke_gradient
            + OFFICIAL_WEIGHTS[2] * mvpe_gradient
        ) / sum(OFFICIAL_WEIGHTS)
        pairs = {
            "rel_vs_tke": (rel_gradient, tke_gradient),
            "mvpe_vs_tke": (mvpe_gradient, tke_gradient),
            "legacy_main_vs_tke": (legacy_gradient, tke_gradient),
            "official_main_vs_tke": (official_gradient, tke_gradient),
        }
        for name, (left, right) in pairs.items():
            geometry[name].extend(
                float(value) for value in per_sample_cosine(left, right).cpu())
        floor = legacy_tke_floor_geometry(
            official_gradient, legacy_gradient, tke_gradient)
        for name, values in floor.items():
            floor_geometry[name].extend(
                float(value) for value in values.cpu())
        batch = history.shape[0]
        for name, loss in zip(component_losses, losses):
            component_losses[name] += float(loss.detach()) * batch
        seen += batch
    result = {
        "schema": "b210_mature_point_gradient_geometry_v1",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": observed,
            "iteration": payload.get("iteration"),
            "training_accuracy_weights": payload.get(
                "training_config", {}).get("accuracy_weights"),
        },
        "data": {
            "sim_dir": str(args.sim_dir.resolve()),
            "trajectories": len(trajectories),
            "windows": len(windows),
            "anchor_samples": seen,
            "anchor_rule": "first 128 stride20 released Sim windows",
            "manifest_records": len(manifest),
        },
        "mean_component_score_losses": {
            name: value / seen for name, value in component_losses.items()
        },
        "gradient_cosines": {
            name: summarize(values) for name, values in geometry.items()
        },
        "legacy_tke_floor_geometry": {
            "applied_samples": int(sum(floor_geometry["applied"])),
            "applied_fraction": (
                sum(floor_geometry["applied"])
                / len(floor_geometry["applied"])),
            **{
                name: summarize(values)
                for name, values in floor_geometry.items()
                if name != "applied"
            },
        },
        "read_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
