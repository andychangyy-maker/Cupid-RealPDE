#!/usr/bin/env python3
"""Train a lightweight dynamic SPS interval head on a frozen Track 1 FNO."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "code/uncertainty"))

from fno_interval_head import FNOIntervalHead

_repro_spec = importlib.util.spec_from_file_location(
    "realpde_interval_reproduce", REPO_ROOT / "code/training/reproduce_finetune.py")
if _repro_spec is None or _repro_spec.loader is None:
    raise ImportError("Cannot import reproduce_finetune.py")
_repro = importlib.util.module_from_spec(_repro_spec)
sys.modules[_repro_spec.name] = _repro
_repro_spec.loader.exec_module(_repro)


SIGMA_GLOBAL = 0.0563870259


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("laplace", "soft_sps", "soft_sps_reg"),
                        default="soft_sps_reg")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit trajectories for smoke tests; zero uses all files.")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--stats", type=Path,
        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--robust-residual-scale", type=float, default=1.65)
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip multiplier grids; intended only for smoke tests.")
    return parser.parse_args()


def validation_windows(manifest_path: Path, filenames: list[str]):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_file = {entry["file"]: entry for entry in manifest}
    if not set(filenames).issubset(by_file):
        raise ValueError("Split manifest does not cover every loaded data file")
    train = [
        _repro.Window(index, int(start))
        for index, filename in enumerate(filenames)
        for start in by_file[filename]["train_starts"]
    ]
    val = [
        _repro.Window(index, int(start))
        for index, filename in enumerate(filenames)
        for start in by_file[filename]["val_starts"]
    ]
    return train, val


def sample_accuracy_factor(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample v9 SPS accuracy multiplier, detached from point prediction."""
    pred = pred[..., :2]
    target = target[..., :2]
    batch = pred.shape[0]
    rel = torch.linalg.vector_norm((pred - target).reshape(batch, -1), dim=1)
    rel /= torch.linalg.vector_norm(target.reshape(batch, -1), dim=1).clamp_min(1e-8)

    pred_fluctuation = pred - pred.mean(dim=1, keepdim=True)
    target_fluctuation = target - target.mean(dim=1, keepdim=True)
    pred_tke = 0.5 * torch.sum(torch.mean(pred_fluctuation.square(), dim=1), dim=-1)
    target_tke = 0.5 * torch.sum(torch.mean(target_fluctuation.square(), dim=1), dim=-1)
    tke = torch.linalg.vector_norm((pred_tke - target_tke).reshape(batch, -1), dim=1)
    tke /= torch.linalg.vector_norm(target_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)

    center_y = pred.shape[2] // 2
    probe_y = [center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < pred.shape[2]]
    mvpe_parts = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if not 0 <= probe_x < pred.shape[3]:
            continue
        pred_probe = pred[:, :, probe_y, probe_x, :].mean(dim=1).reshape(batch, -1)
        target_probe = target[:, :, probe_y, probe_x, :].mean(dim=1).reshape(batch, -1)
        error = torch.linalg.vector_norm(pred_probe - target_probe, dim=1)
        error /= torch.linalg.vector_norm(target_probe, dim=1).clamp_min(1e-8)
        mvpe_parts.append(error)
    mvpe = torch.stack(mvpe_parts).mean(dim=0)

    def factor(error: torch.Tensor) -> torch.Tensor:
        return 0.5 / (0.5 + error)

    return (0.5 * factor(rel) + 0.3 * factor(tke) + 0.2 * factor(mvpe)).detach()


def interval_losses(pred: torch.Tensor, target: torch.Tensor, half_width: torch.Tensor,
                    variant: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = pred[..., :2]
    target = target[..., :2]
    error = torch.abs(target - pred)
    scored = target != 0.0
    factor = sample_accuracy_factor(pred, target).reshape(-1, 1, 1, 1, 1)
    reference = pred.new_tensor([0.01075, 0.00550]).reshape(1, 1, 1, 1, 2)
    tau = 0.10 * reference

    scaled_width = (half_width / reference).clamp_min(1e-6)
    nll_elements = error / half_width.clamp_min(1e-7) + torch.log(scaled_width)
    nll = nll_elements[scored].mean()
    soft_inside = torch.sigmoid((half_width - error) / tau)
    soft_elements = factor * soft_inside * torch.exp(-2.0 * half_width / SIGMA_GLOBAL)
    soft_score = soft_elements[scored].mean()

    if variant == "laplace":
        loss = nll
    elif variant == "soft_sps":
        loss = -soft_score + 0.02 * nll
    else:
        loss = -soft_score + 0.10 * nll
    return loss, {"nll": nll.detach(), "soft_sps": soft_score.detach()}


def point_prediction(model: nn.Module, x: torch.Tensor, normalizer) -> torch.Tensor:
    with torch.inference_mode():
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        pred_norm = model(x_norm)
        pred = normalizer.postprocess_prediction(pred_norm)
    # A normal detached tensor can be saved by the head's autograd operations;
    # an inference tensor cannot be saved for backward even when the FNO is frozen.
    return pred.detach().clone()


def exact_sps_for_loader(point_model: nn.Module, head: nn.Module, loader: DataLoader,
                         normalizer, device: torch.device,
                         multiplier_uv: tuple[float, float] = (1.0, 1.0),
                         residual_scale: float = 1.0) -> dict[str, float]:
    numerator = 0.0
    denominator = 0
    inside_count = 0
    width_sum = 0.0
    multiplier = torch.tensor(multiplier_uv, device=device).reshape(1, 1, 1, 1, 2)
    head.eval()
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)[..., :2]
            pred = point_prediction(point_model, x, normalizer)[..., :2]
            half_width = head(pred, x[:, -1, ..., :2]) * multiplier
            residual = (target - pred) * residual_scale
            scored = target != 0.0
            inside = torch.abs(residual) <= half_width
            factor = sample_accuracy_factor(pred, target).reshape(-1, 1, 1, 1, 1)
            elements = factor * inside * torch.exp(-2.0 * half_width / SIGMA_GLOBAL)
            numerator += float(elements[scored].sum())
            denominator += int(scored.sum())
            inside_count += int((inside & scored).sum())
            width_sum += float((2.0 * half_width)[scored].sum())
    return {
        "sps_score": 100.0 * numerator / denominator,
        "coverage": inside_count / denominator,
        "mean_width": width_sum / denominator,
    }


def collect_interval_cache(point_model: nn.Module, head: nn.Module, loader: DataLoader,
                           normalizer, device: torch.device) -> list[dict[str, torch.Tensor]]:
    """Run the frozen models once and retain only scored SPS element tensors."""
    parts = [
        {"error": [], "width": [], "factor": []},
        {"error": [], "width": [], "factor": []},
    ]
    head.eval()
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)[..., :2]
            pred = point_prediction(point_model, x, normalizer)[..., :2]
            width = head(pred, x[:, -1, ..., :2])
            factor = sample_accuracy_factor(pred, target).reshape(-1, 1, 1, 1)
            for channel in range(2):
                scored = target[..., channel] != 0.0
                expanded_factor = factor.expand_as(target[..., channel])
                parts[channel]["error"].append(
                    torch.abs(target[..., channel] - pred[..., channel])[scored])
                parts[channel]["width"].append(width[..., channel][scored])
                parts[channel]["factor"].append(expanded_factor[scored])
    return [
        {name: torch.cat(tensors) for name, tensors in channel.items()}
        for channel in parts
    ]


def exact_sps_for_cache(cache: list[dict[str, torch.Tensor]],
                        multiplier_uv: tuple[float, float] = (1.0, 1.0),
                        residual_scale: float = 1.0) -> dict[str, float]:
    numerator = torch.zeros((), dtype=torch.float64, device=cache[0]["error"].device)
    inside_count = 0
    width_sum = torch.zeros_like(numerator)
    denominator = sum(int(channel["error"].numel()) for channel in cache)
    for channel, multiplier in zip(cache, multiplier_uv):
        width = channel["width"] * float(multiplier)
        inside = channel["error"] * residual_scale <= width
        numerator += torch.sum(
            (channel["factor"] * torch.exp(-2.0 * width / SIGMA_GLOBAL))[inside],
            dtype=torch.float64)
        inside_count += int(inside.sum())
        width_sum += torch.sum(2.0 * width, dtype=torch.float64)
    return {
        "sps_score": 100.0 * float(numerator) / denominator,
        "coverage": inside_count / denominator,
        "mean_width": float(width_sum) / denominator,
    }


def calibrate_multipliers(cache: list[dict[str, torch.Tensor]],
                          residual_scale: float) -> tuple[list[float], dict[str, object]]:
    # The NLL variant can learn a deliberately narrow conditional scale, so its
    # hidden-set stress multiplier may legitimately exceed 2x.
    candidates = np.arange(0.50, 4.01, 0.10)
    channel_scores = [[], []]
    # Because the SPS sum is separable by channel, optimize u and v independently.
    for channel in range(2):
        for candidate in candidates:
            multiplier = [1.0, 1.0]
            multiplier[channel] = float(candidate)
            metrics = exact_sps_for_cache(
                cache, tuple(multiplier), residual_scale=residual_scale)
            channel_scores[channel].append(metrics)
    selected = [
        float(candidates[int(np.argmax([row["sps_score"] for row in scores]))])
        for scores in channel_scores
    ]
    selected_metrics = exact_sps_for_cache(
        cache, tuple(selected), residual_scale=residual_scale)
    return selected, {
        "candidates": candidates.tolist(),
        "channel_metrics": channel_scores,
        "selected_metrics": selected_metrics,
    }


def save_head(path: Path, head: nn.Module, args: argparse.Namespace,
              iteration: int, metrics: dict[str, float]) -> None:
    torch.save({
        "head_state_dict": {key: value.detach().cpu()
                            for key, value in head.state_dict().items()},
        "head_hidden": args.hidden,
        "variant": args.variant,
        "iteration": iteration,
        "validation": metrics,
    }, path)


def main() -> None:
    args = parse_args()
    if (args.updates <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0
            or args.eval_interval <= 0 or args.lr <= 0
            or not math.isfinite(args.robust_residual_scale)
            or args.robust_residual_scale <= 0):
        raise ValueError("Training counts/rates and robust scale must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _repro.set_seed(args.seed)
    device = torch.device(args.device)
    trajectories, metadata = _repro.load_trajectories(args.real_dir, args.max_files)
    filenames = [entry["file"] for entry in metadata]
    train_windows, val_windows = validation_windows(args.split_manifest, filenames)
    train_loader = DataLoader(
        _repro.WindowDataset(trajectories, train_windows), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(
        _repro.WindowDataset(trajectories, val_windows), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    train_iterator = _repro.cycling(train_loader)
    normalizer = _repro.Normalizer(args.stats, device)
    loader_module = _repro.load_baseline_module(args.kit_dir)
    point_model, point_metadata = loader_module.load_baseline(
        "fno", str(args.point_checkpoint), device=str(device))
    point_model.eval()
    for parameter in point_model.parameters():
        parameter.requires_grad_(False)

    head = FNOIntervalHead(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates)
    best_sps = -float("inf")
    best_iteration = 0
    history = []
    interval_loss = interval_nll = interval_soft = 0.0
    interval_count = 0
    started = time.time()

    for iteration in range(1, args.updates + 1):
        head.train()
        x, target = next(train_iterator)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = point_prediction(point_model, x, normalizer)
        half_width = head(pred, x[:, -1, ..., :2])
        loss, components = interval_losses(pred, target, half_width, args.variant)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        interval_loss += float(loss)
        interval_nll += float(components["nll"])
        interval_soft += float(components["soft_sps"])
        interval_count += 1

        if iteration % args.eval_interval == 0 or iteration == args.updates:
            validation = exact_sps_for_loader(
                point_model, head, val_loader, normalizer, device)
            row = {
                "iteration": iteration,
                "train_loss": interval_loss / interval_count,
                "train_nll": interval_nll / interval_count,
                "train_soft_sps": interval_soft / interval_count,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                **validation,
            }
            history.append(row)
            print(json.dumps({"event": "eval", **row}), flush=True)
            interval_loss = interval_nll = interval_soft = 0.0
            interval_count = 0
            if validation["sps_score"] > best_sps:
                best_sps = validation["sps_score"]
                best_iteration = iteration
                save_head(args.output_dir / "best.pth", head, args, iteration, validation)

    best_payload = torch.load(
        args.output_dir / "best.pth", map_location=device, weights_only=False)
    head.load_state_dict(best_payload["head_state_dict"])
    interval_cache = collect_interval_cache(
        point_model, head, val_loader, normalizer, device)
    if args.skip_calibration:
        local_multiplier = robust_multiplier = [1.0, 1.0]
        local_grid = robust_grid = {"skipped": True}
    else:
        local_multiplier, local_grid = calibrate_multipliers(
            interval_cache, residual_scale=1.0)
        robust_multiplier, robust_grid = calibrate_multipliers(
            interval_cache, residual_scale=args.robust_residual_scale)
    local_metrics = exact_sps_for_cache(
        interval_cache, tuple(local_multiplier))
    robust_on_local = exact_sps_for_cache(
        interval_cache, tuple(robust_multiplier))
    robust_stress = exact_sps_for_cache(
        interval_cache, tuple(robust_multiplier),
        residual_scale=args.robust_residual_scale)
    final_payload = {
        **best_payload,
        "point_checkpoint": str(args.point_checkpoint),
        "point_checkpoint_metadata": point_metadata,
        "local_multiplier_uv": local_multiplier,
        "robust_multiplier_uv": robust_multiplier,
        "robust_residual_scale": args.robust_residual_scale,
        "local_metrics": local_metrics,
        "robust_multiplier_on_local_metrics": robust_on_local,
        "robust_stress_metrics": robust_stress,
    }
    torch.save(final_payload, args.output_dir / "interval_head.pth")
    summary = {
        "variant": args.variant,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "best_iteration": best_iteration,
        "best_uncalibrated_sps": best_sps,
        "local_multiplier_uv": local_multiplier,
        "robust_multiplier_uv": robust_multiplier,
        "local_metrics": local_metrics,
        "robust_multiplier_on_local_metrics": robust_on_local,
        "robust_stress_metrics": robust_stress,
        "wall_seconds": time.time() - started,
        "history": history,
        "local_grid": local_grid,
        "robust_grid": robust_grid,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with (args.output_dir / "learning_curve.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({
        "event": "complete", "output_dir": str(args.output_dir),
        "best_iteration": best_iteration, "best_sps": best_sps,
        "local_metrics": local_metrics,
        "robust_multiplier_uv": robust_multiplier,
        "robust_on_local": robust_on_local,
        "robust_stress": robust_stress,
        "wall_seconds": summary["wall_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
