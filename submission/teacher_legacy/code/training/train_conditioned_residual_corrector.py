#!/usr/bin/env python3
"""Train a lightweight, condition-aware correction on a frozen Track 1 FNO."""

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
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/uncertainty"))

from conditioned_residual_corrector import ConditionedResidualCorrector

_base_spec = importlib.util.spec_from_file_location(
    "realpde_interval_base_corrector",
    REPO_ROOT / "code/training/train_fno_interval_head.py",
)
if _base_spec is None or _base_spec.loader is None:
    raise ImportError("Cannot import train_fno_interval_head.py")
_base = importlib.util.module_from_spec(_base_spec)
sys.modules[_base_spec.name] = _base
_base_spec.loader.exec_module(_base)
_repro = _base._repro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--context-hidden", type=int, default=16)
    parser.add_argument("--architecture", choices=("spatial", "dilated", "deep"),
                        default="deep")
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--spectrum-weight", type=float, default=0.05)
    parser.add_argument("--condition-weight", type=float, default=0.05)
    parser.add_argument("--correction-reg-weight", type=float, default=0.002)
    parser.add_argument("--group-robust-weight", type=float, default=0.50)
    parser.add_argument("--group-temperature", type=float, default=0.05)
    parser.add_argument("--accuracy-weights", type=float, nargs=3,
                        default=(1.0, 4.0, 1.0),
                        metavar=("REL", "TKE", "MVPE"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Debug only: use the first N trajectories.")
    parser.add_argument(
        "--holdout-re", type=int, nargs="*", default=(),
        help="Reserve complete Reynolds conditions for OOD validation.")
    parser.add_argument(
        "--holdout-aoa", type=int, nargs="*", default=(),
        help="Reserve complete angles of attack for OOD validation.")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--stats", type=Path, default=Path(
        "kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument("--kit-dir", type=Path, default=Path(
        "kits/track1/realpde_t1_starting_kit_v9"))
    return parser.parse_args()


class ConditionWindowDataset(Dataset):
    def __init__(self, trajectories: list[torch.Tensor], windows,
                 metadata: list[dict], re_min: float, re_max: float,
                 aoa_min: float, aoa_max: float):
        self.base = _repro.WindowDataset(trajectories, windows)
        self.windows = windows
        self.metadata = metadata
        self.log_re_min = math.log(re_min)
        self.log_re_max = math.log(re_max)
        self.aoa_min = aoa_min
        self.aoa_max = aoa_max

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        item = self.metadata[self.windows[index].trajectory]
        log_re = math.log(float(item["re"]))
        re_target = 2.0 * (
            (log_re - self.log_re_min)
            / max(self.log_re_max - self.log_re_min, 1e-12)) - 1.0
        aoa_target = 2.0 * (
            (float(item["aoa"]) - self.aoa_min)
            / max(self.aoa_max - self.aoa_min, 1e-12)) - 1.0
        re_group = 0 if item["re"] <= 9000 else (1 if item["re"] < 17000 else 2)
        aoa_group = int(round(float(item["aoa"]) / 5.0))
        return (
            x, y,
            torch.tensor((re_target, aoa_target), dtype=torch.float32),
            torch.tensor(re_group, dtype=torch.long),
            torch.tensor(aoa_group, dtype=torch.long),
        )


def split_windows(manifest_path: Path, filenames: list[str], metadata: list[dict],
                  holdout_re: tuple[int, ...] | list[int] = (),
                  holdout_aoa: tuple[int, ...] | list[int] = ()):
    if holdout_re and holdout_aoa:
        raise ValueError("Use only one of --holdout-re and --holdout-aoa")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_file = {entry["file"]: entry for entry in manifest}
    if not set(filenames).issubset(by_file):
        raise ValueError("Split manifest does not cover loaded filenames")
    train = []
    val = []
    kind = "re" if holdout_re else ("aoa" if holdout_aoa else None)
    heldout_values = {int(value) for value in (holdout_re or holdout_aoa)}
    if kind is not None:
        available = {int(item[kind]) for item in metadata}
        missing = heldout_values - available
        if missing:
            raise ValueError(f"Unknown {kind} holdout values: {sorted(missing)}")
    heldout_files = []
    for trajectory, filename in enumerate(filenames):
        entry = by_file[filename]
        if kind is not None and int(metadata[trajectory][kind]) in heldout_values:
            heldout_files.append(filename)
            starts = sorted(set(entry["train_starts"] + entry["val_starts"]))
            val.extend(
                _repro.Window(trajectory, int(start)) for start in starts)
        else:
            train.extend(
                _repro.Window(trajectory, int(start))
                for start in entry["train_starts"])
            if kind is None:
                val.extend(
                    _repro.Window(trajectory, int(start))
                    for start in entry["val_starts"])
    if not train or not val:
        raise ValueError("Split must leave both train and validation windows")
    scheme = (
        {"kind": "temporal_tail", "values": []}
        if kind is None else {
            "kind": f"holdout_{kind}",
            "values": sorted(heldout_values),
            "heldout_files": heldout_files,
        })
    return train, val, scheme


def per_sample_accuracy_loss(pred: torch.Tensor, target: torch.Tensor,
                             weights: tuple[float, float, float]):
    batch = pred.shape[0]
    rel_num = torch.linalg.vector_norm((pred - target).reshape(batch, -1), dim=1)
    rel_den = torch.linalg.vector_norm(target.reshape(batch, -1), dim=1).clamp_min(1e-8)
    rel_error = rel_num / rel_den

    pred_fluctuation = pred - pred.mean(dim=1, keepdim=True)
    target_fluctuation = target - target.mean(dim=1, keepdim=True)
    pred_tke = 0.5 * torch.sum(pred_fluctuation.square().mean(dim=1), dim=-1)
    target_tke = 0.5 * torch.sum(target_fluctuation.square().mean(dim=1), dim=-1)
    tke_error = torch.linalg.vector_norm(
        (pred_tke - target_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        target_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)

    probe_y = [pred.shape[2] // 2 + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < pred.shape[2]]
    mvpe_errors = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if not 0 <= probe_x < pred.shape[3]:
            continue
        pred_profile = pred[:, :, probe_y, probe_x].mean(dim=1).reshape(batch, -1)
        target_profile = target[:, :, probe_y, probe_x].mean(dim=1).reshape(batch, -1)
        mvpe_errors.append(
            torch.linalg.vector_norm(pred_profile - target_profile, dim=1)
            / torch.linalg.vector_norm(target_profile, dim=1).clamp_min(1e-8))
    mvpe_error = torch.stack(mvpe_errors).mean(dim=0)
    errors = torch.stack((rel_error, tke_error, mvpe_error), dim=1)
    scores = 1.0 / (1.0 + 0.5 * errors)
    weight_tensor = scores.new_tensor(weights)
    sample_loss = 1.0 - (
        scores * weight_tensor).sum(dim=1) / weight_tensor.sum()
    return sample_loss, errors.detach()


def marginal_group_robust_loss(sample_loss: torch.Tensor,
                               re_group: torch.Tensor,
                               aoa_group: torch.Tensor,
                               temperature: float) -> torch.Tensor:
    means = []
    for values in (re_group, aoa_group):
        for group in torch.unique(values):
            selected = values == group
            if torch.any(selected):
                means.append(sample_loss[selected].mean())
    stacked = torch.stack(means)
    return temperature * (
        torch.logsumexp(stacked / temperature, dim=0)
        - math.log(len(means)))


def spectrum_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    pred_spectrum = torch.abs(torch.fft.rfft(pred_centered, dim=1))
    target_spectrum = torch.abs(torch.fft.rfft(target_centered, dim=1))
    batch = pred.shape[0]
    return torch.mean(
        torch.linalg.vector_norm(
            (pred_spectrum - target_spectrum).reshape(batch, -1), dim=1)
        / torch.linalg.vector_norm(
            target_spectrum.reshape(batch, -1), dim=1).clamp_min(1e-8))


def score_arrays(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    metrics = _repro.regression_metrics(prediction, target)
    for name in ("rel_l2", "tke_rel_l2", "mvpe_rel_l2"):
        metrics[f"{name}_score"] = 100.0 / (1.0 + 0.5 * metrics[name])
    metrics["accuracy_score_mean"] = float(np.mean([
        metrics["rel_l2_score"], metrics["tke_rel_l2_score"],
        metrics["mvpe_rel_l2_score"],
    ]))
    return metrics


def evaluate(point_model: nn.Module, corrector: nn.Module, loader: DataLoader,
             normalizer, device: torch.device) -> dict:
    corrector.eval()
    predictions = []
    targets = []
    re_groups = []
    aoa_groups = []
    conditions = []
    with torch.inference_mode():
        for x, target, condition, re_group, aoa_group in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            point = _base.point_prediction(point_model, x, normalizer)[..., :2]
            corrected, estimated_condition = corrector(point, x)
            predictions.append(corrected.cpu().numpy())
            targets.append(target[..., :2].cpu().numpy())
            re_groups.append(re_group.numpy())
            aoa_groups.append(aoa_group.numpy())
            conditions.append((estimated_condition.cpu() - condition).square().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    re_group = np.concatenate(re_groups)
    aoa_group = np.concatenate(aoa_groups)
    overall = score_arrays(prediction, target)
    group_metrics = {}
    for prefix, values in (("re", re_group), ("aoa", aoa_group)):
        for value in sorted(set(values.tolist())):
            mask = values == value
            group_metrics[f"{prefix}_{value}"] = {
                "samples": int(mask.sum()),
                **score_arrays(prediction[mask], target[mask]),
            }
    group_scores = [
        value["accuracy_score_mean"] for value in group_metrics.values()]
    return {
        "overall": overall,
        "groups": group_metrics,
        "condition_mse": float(np.concatenate(conditions).mean()),
        "condition_accuracy_min": float(min(group_scores)),
        "robust_selection_score": float(
            overall["accuracy_score_mean"] + 0.25 * min(group_scores)),
    }


def save_checkpoint(path: Path, corrector: nn.Module, args: argparse.Namespace,
                    iteration: int, validation: dict, condition_ranges: dict) -> None:
    torch.save({
        "corrector_state_dict": {
            key: value.detach().cpu()
            for key, value in corrector.state_dict().items()},
        "hidden": args.hidden,
        "context_hidden": args.context_hidden,
        "architecture": args.architecture,
        "point_checkpoint": str(args.point_checkpoint),
        "iteration": iteration,
        "validation": validation,
        "condition_ranges": condition_ranges,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
    }, path)


def main() -> None:
    args = parse_args()
    scalar_values = (
        args.lr, args.mse_weight, args.spectrum_weight, args.condition_weight,
        args.correction_reg_weight, args.group_robust_weight,
        args.group_temperature,
    )
    if (args.updates <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0
            or args.eval_interval <= 0 or args.hidden <= 0
            or args.context_hidden <= 0 or any(
                not math.isfinite(value) or value < 0 for value in scalar_values)
            or args.lr <= 0 or args.group_temperature <= 0
            or not 0.0 <= args.group_robust_weight <= 1.0):
        raise ValueError("Invalid training arguments")
    weights = tuple(float(value) for value in args.accuracy_weights)
    if any(value < 0 or not math.isfinite(value) for value in weights) or not any(weights):
        raise ValueError("accuracy weights must be nonnegative and not all zero")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _repro.set_seed(args.seed)
    device = torch.device(args.device)
    trajectories, metadata = _repro.load_trajectories(
        args.real_dir, args.max_files)
    filenames = [item["file"] for item in metadata]
    train_windows, val_windows, validation_scheme = split_windows(
        args.split_manifest, filenames, metadata,
        args.holdout_re, args.holdout_aoa)
    re_values = [float(item["re"]) for item in metadata]
    aoa_values = [float(item["aoa"]) for item in metadata]
    condition_ranges = {
        "re_min": min(re_values), "re_max": max(re_values),
        "aoa_min": min(aoa_values), "aoa_max": max(aoa_values),
    }
    train_dataset = ConditionWindowDataset(
        trajectories, train_windows, metadata, **condition_ranges)
    val_dataset = ConditionWindowDataset(
        trajectories, val_windows, metadata, **condition_ranges)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    train_iterator = _repro.cycling(train_loader)
    normalizer = _repro.Normalizer(args.stats, device)
    baseline = _repro.load_baseline_module(args.kit_dir)
    point_model, point_metadata = baseline.load_baseline(
        "fno", str(args.point_checkpoint), device=str(device))
    point_model.eval()
    for parameter in point_model.parameters():
        parameter.requires_grad_(False)
    corrector = ConditionedResidualCorrector(
        hidden=args.hidden, context_hidden=args.context_hidden,
        architecture=args.architecture).to(device)
    optimizer = torch.optim.AdamW(
        corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)

    initial_validation = evaluate(
        point_model, corrector, val_loader, normalizer, device)
    best_score = initial_validation["robust_selection_score"]
    best_iteration = 0
    save_checkpoint(
        args.output_dir / "best.pth", corrector, args, 0,
        initial_validation, condition_ranges)
    history = []
    sums = {name: 0.0 for name in (
        "loss", "accuracy", "robust_accuracy", "mse", "spectrum",
        "condition", "correction_reg")}
    count = 0
    started = time.time()

    for iteration in range(1, args.updates + 1):
        corrector.train()
        x, target, condition, re_group, aoa_group = next(train_iterator)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)[..., :2]
        condition = condition.to(device, non_blocking=True)
        re_group = re_group.to(device, non_blocking=True)
        aoa_group = aoa_group.to(device, non_blocking=True)
        with torch.no_grad():
            point = _base.point_prediction(point_model, x, normalizer)[..., :2]
        corrected, estimated_condition = corrector(point, x)
        sample_accuracy, _errors = per_sample_accuracy_loss(
            corrected, target, weights)
        mean_accuracy = sample_accuracy.mean()
        robust_accuracy = marginal_group_robust_loss(
            sample_accuracy, re_group, aoa_group, args.group_temperature)
        accuracy = (
            (1.0 - args.group_robust_weight) * mean_accuracy
            + args.group_robust_weight * robust_accuracy)
        normalized_mse = torch.mean(
            ((corrected - target) / corrector.std).square())
        spectral = spectrum_loss(corrected, target)
        condition_loss = torch.mean((estimated_condition - condition).square())
        correction_reg = torch.mean(
            ((corrected - point) / corrector.max_correction_uv).square())
        loss = (
            accuracy
            + args.mse_weight * normalized_mse
            + args.spectrum_weight * spectral
            + args.condition_weight * condition_loss
            + args.correction_reg_weight * correction_reg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(corrector.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        values = {
            "loss": loss, "accuracy": mean_accuracy,
            "robust_accuracy": robust_accuracy, "mse": normalized_mse,
            "spectrum": spectral, "condition": condition_loss,
            "correction_reg": correction_reg,
        }
        for name, value in values.items():
            sums[name] += float(value.detach())
        count += 1

        if iteration % args.eval_interval == 0 or iteration == args.updates:
            validation = evaluate(
                point_model, corrector, val_loader, normalizer, device)
            row = {
                "iteration": iteration,
                **{f"train_{name}": value / count
                   for name, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                "val_accuracy_score_mean": validation["overall"]["accuracy_score_mean"],
                "val_rel_l2_score": validation["overall"]["rel_l2_score"],
                "val_tke_score": validation["overall"]["tke_rel_l2_score"],
                "val_mvpe_score": validation["overall"]["mvpe_rel_l2_score"],
                "val_condition_accuracy_min": validation["condition_accuracy_min"],
                "val_robust_selection_score": validation["robust_selection_score"],
                "val_condition_mse": validation["condition_mse"],
            }
            history.append(row)
            print(json.dumps({"event": "eval", **row}), flush=True)
            if validation["robust_selection_score"] > best_score:
                best_score = validation["robust_selection_score"]
                best_iteration = iteration
                save_checkpoint(
                    args.output_dir / "best.pth", corrector, args, iteration,
                    validation, condition_ranges)
            sums = {name: 0.0 for name in sums}
            count = 0

    final_validation = evaluate(
        point_model, corrector, val_loader, normalizer, device)
    best_payload = torch.load(
        args.output_dir / "best.pth", map_location=device, weights_only=False)
    corrector.load_state_dict(best_payload["corrector_state_dict"])
    best_validation = evaluate(
        point_model, corrector, val_loader, normalizer, device)
    summary = {
        "point_checkpoint": str(args.point_checkpoint),
        "point_checkpoint_metadata": point_metadata,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "validation_scheme": validation_scheme,
        "corrector_parameters": sum(
            parameter.numel() for parameter in corrector.parameters()),
        "best_iteration": best_iteration,
        "best_robust_selection_score": best_score,
        "initial_validation": initial_validation,
        "best_validation": best_validation,
        "final_validation": final_validation,
        "condition_ranges": condition_ranges,
        "history": history,
        "wall_seconds": time.time() - started,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
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
        "best_iteration": best_iteration,
        "best_validation": best_validation,
        "wall_seconds": summary["wall_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
