#!/usr/bin/env python3
"""Distill fold-held P0 absolute endpoints into the fast Direct UQ head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, Path(__file__).parent, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_p0_oof_absolute_endpoints as endpoint_cache  # noqa: E402
import evaluate_mean_expert_oof_blend as s1  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from models.distilled_interval_head import (  # noqa: E402
    DistilledIntervalHead,
    DistilledIntervalHeadConfig,
)
from train_distilled_flow_corrector import (  # noqa: E402
    load_forecaster,
    seed_everything,
)
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


P1_FOLD04_REFERENCE = {
    "sps_score": 53.4518883635,
    "coverage": 0.8991232113,
    "mean_width": 0.0164559502,
}
REFERENCE_WIDTH = (0.01075, 0.00550)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint-cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-head", type=Path)
    parser.add_argument("--expected-point-sha256")
    parser.add_argument("--expected-cache-sha256")
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--soft-sps-weight", type=float, default=0.05)
    parser.add_argument("--coverage-weight", type=float, default=0.10)
    parser.add_argument(
        "--coverage-target", type=float,
        default=P1_FOLD04_REFERENCE["coverage"])
    parser.add_argument("--soft-tau-fraction", type=float, default=0.10)
    parser.add_argument("--held-fold", type=int, default=4)
    parser.add_argument("--min-sps-score", type=float, default=53.20)
    parser.add_argument("--max-coverage-drop", type=float, default=0.012)
    parser.add_argument("--max-width-ratio", type=float, default=1.08)
    parser.add_argument("--time-budget-seconds", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size, args.lr,
        args.soft_tau_fraction, args.time_budget_seconds,
        args.min_sps_score, args.max_width_ratio,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("positive training and gate values must be finite")
    nonnegative = (
        args.weight_decay, args.grad_clip, args.distill_weight,
        args.soft_sps_weight, args.coverage_weight,
        args.max_coverage_drop,
    )
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("loss weights and tolerances must be non-negative")
    if not 0 < args.coverage_target < 1 or args.held_fold not in range(5):
        raise ValueError("coverage target or held fold is outside its contract")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    for value in (args.expected_point_sha256, args.expected_cache_sha256):
        if value is not None and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("expected SHA-256 values must be lowercase hex")


def select_fold_indices(
        folds: np.ndarray | torch.Tensor, held_fold: int,
        ) -> tuple[list[int], list[int]]:
    values = np.asarray(folds, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise ValueError("fold labels must be a non-empty vector")
    if held_fold not in range(5) or not set(values.tolist()).issubset(set(range(5))):
        raise ValueError("fold labels must lie in 0..4")
    train = np.flatnonzero(values != held_fold).tolist()
    evaluation = np.flatnonzero(values == held_fold).tolist()
    if (not train or not evaluation or set(train) & set(evaluation)
            or sorted(train + evaluation) != list(range(len(values)))):
        raise RuntimeError("held-fold partition is incomplete or overlapping")
    if any(values[index] == held_fold for index in train):
        raise RuntimeError("held fold leaked into optimization indices")
    return train, evaluation


def absolute_endpoints(
        prediction_normalized: torch.Tensor,
        lower_extent: torch.Tensor, upper_extent: torch.Tensor,
        target_mean: torch.Tensor, target_std: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (prediction_normalized.shape[:-1] != lower_extent.shape[:-1]
            or lower_extent.shape != upper_extent.shape
            or lower_extent.shape[-1] != 2
            or prediction_normalized.shape[-1] < 2):
        raise ValueError("point and extent tensors have incompatible shapes")
    mean = target_mean[:2].view(1, 1, 1, 1, 2)
    std = target_std[:2].view(1, 1, 1, 1, 2)
    point = prediction_normalized[..., :2] * std + mean
    return point, point - lower_extent * std, point + upper_extent * std


class IndexedTailDataset(Dataset):
    """Cache rows mapped explicitly to the original WindowDataset rows."""

    def __init__(self, windows, cache_payload: dict):
        self.windows = windows
        self.lower = cache_payload["lower"]
        self.upper = cache_payload["upper"]
        self.dataset_index = cache_payload["dataset_index"].to(torch.int64)

    def __len__(self) -> int:
        return len(self.dataset_index)

    def __getitem__(self, row: int):
        history, target = self.windows[int(self.dataset_index[row])]
        return history, target, self.lower[row], self.upper[row]


def load_and_validate_cache(
        path: Path, windows, sample_folds: list[int], fold_manifest: dict,
        raw_aoas: np.ndarray, expected_sha256: str | None,
        ) -> tuple[dict, dict]:
    identity = s1.file_identity(path)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError("endpoint cache SHA-256 differs from requested identity")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    expected_keys = {
        "lower", "upper", "target", "fold", "true_aoa",
        "dataset_index", "metadata",
    }
    if set(payload) != expected_keys:
        raise ValueError("endpoint cache keys differ from contract")
    metadata = payload["metadata"]
    if (metadata.get("format") != endpoint_cache.CACHE_FORMAT
            or metadata.get("composition")
            != {"weight_b": 0.625, "weight_c": 0.25}
            or metadata.get("fold_manifest") != fold_manifest):
        raise ValueError("endpoint cache provenance differs from contract")
    for side, expected in endpoint_cache.FULL_BOUNDS_SHA256.items():
        if metadata["full_float32_bounds_identity"][side]["sha256"] != expected:
            raise ValueError("endpoint cache full-bound identity is not frozen P0")
    shape = (725, 20, 32, 64, 2)
    if (tuple(payload["lower"].shape) != shape
            or tuple(payload["upper"].shape) != shape
            or tuple(payload["target"].shape) != shape
            or payload["lower"].dtype != torch.float32
            or payload["upper"].dtype != torch.float32
            or payload["target"].dtype != torch.float16):
        raise ValueError("endpoint cache tensor shape or dtype differs")
    for side in ("lower", "upper"):
        actual = s1.array_identity(payload[side].numpy())
        if actual != metadata["stored_uv_float32_identity"][side]:
            raise ValueError(f"stored {side} UV endpoints differ from metadata")
    target_identity = s1.array_identity(payload["target"].numpy())
    if target_identity != metadata["stored_target_uv_fp16_identity"]:
        raise ValueError("stored target differs from metadata")

    dataset_index = payload["dataset_index"].numpy().astype(np.int64)
    folds = payload["fold"].numpy().astype(np.int64)
    aoas = payload["true_aoa"].numpy().astype(np.int64)
    if (not np.array_equal(np.sort(dataset_index), np.arange(725))
            or not np.array_equal(folds, np.asarray(sample_folds)[dataset_index])
            or not np.array_equal(aoas, raw_aoas[dataset_index])):
        raise RuntimeError("cache index/fold/AoA alignment differs from split")
    for row, original_index in enumerate(dataset_index.tolist()):
        _history, target = windows[original_index]
        if not torch.equal(
                payload["target"][row], target[..., :2].to(torch.float16)):
            raise RuntimeError(
                f"cache target differs at row {row}, dataset {original_index}")
        lower = payload["lower"][row]
        upper = payload["upper"][row]
        if (not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
                or torch.any(lower > upper)):
            raise FloatingPointError(f"invalid endpoint cache row {row}")
    return payload, {"file": identity, "metadata": metadata}


def state_identity(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_head(start: Path | None, device: torch.device) -> tuple[DistilledIntervalHead, dict]:
    requested = DistilledIntervalHeadConfig()
    head = DistilledIntervalHead(requested)
    metadata: dict = {"initialization": "fresh_zero_output"}
    if start is not None:
        payload = torch.load(start, map_location="cpu", weights_only=False)
        config = dict(payload["model_config"])
        architecture = config.pop("architecture", "distilled_interval_head")
        if architecture != "distilled_interval_head":
            raise ValueError("start-head is not a distilled interval head")
        for key in ("feature_channels", "reference_total_widths"):
            if key in config:
                config[key] = tuple(config[key])
        if DistilledIntervalHeadConfig(**config) != requested:
            raise ValueError("start-head architecture differs from Direct contract")
        head.load_state_dict(payload["model_state_dict"], strict=True)
        metadata = {
            "initialization": "warm_start",
            "file": s1.file_identity(start),
            "source_iteration": payload.get("iteration"),
        }
    return head.to(device=device, dtype=torch.float32), metadata


def endpoint_objective(
        args: argparse.Namespace, point: torch.Tensor,
        lower: torch.Tensor, upper: torch.Tensor,
        teacher_lower: torch.Tensor, teacher_upper: torch.Tensor,
        truth: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reference = lower.new_tensor(REFERENCE_WIDTH).view(1, 1, 1, 1, 2)
    distill = 0.5 * (
        F.smooth_l1_loss((lower - teacher_lower) / reference,
                         torch.zeros_like(lower))
        + F.smooth_l1_loss((upper - teacher_upper) / reference,
                           torch.zeros_like(upper)))
    truth_uv = truth[..., :2]
    scored = truth_uv != 0.0
    tau = args.soft_tau_fraction * reference
    soft_inside = (
        torch.sigmoid((truth_uv - lower) / tau)
        * torch.sigmoid((upper - truth_uv) / tau))
    factor = common.sample_sps_factor(
        point.detach(), truth_uv).detach().view(-1, 1, 1, 1, 1)
    soft_sps = (
        factor * soft_inside
        * torch.exp(-(upper - lower) / common.SPS_SIGMA))[scored].mean()
    soft_coverage = soft_inside[scored].mean()
    coverage_penalty = F.relu(
        args.coverage_target - soft_coverage).square()
    loss = (
        args.distill_weight * distill
        - args.soft_sps_weight * soft_sps
        + args.coverage_weight * coverage_penalty)
    return loss, {
        "distill": distill.detach(),
        "soft_sps": soft_sps.detach(),
        "soft_coverage": soft_coverage.detach(),
        "coverage_penalty": coverage_penalty.detach(),
        "mean_width": (upper - lower)[scored].mean().detach(),
    }


def new_accumulator() -> dict[str, float | int]:
    return {"numerator": 0.0, "scored": 0, "covered": 0, "width": 0.0}


def update_accumulator(
        accumulator: dict, point: torch.Tensor, lower: torch.Tensor,
        upper: torch.Tensor, truth: torch.Tensor) -> None:
    scored = truth != 0.0
    inside = (truth >= lower) & (truth <= upper)
    factor = common.sample_sps_factor(point, truth).view(-1, 1, 1, 1, 1)
    value = factor * inside * torch.exp(
        -(upper - lower) / common.SPS_SIGMA)
    accumulator["numerator"] += float(value[scored].sum(dtype=torch.float64))
    accumulator["scored"] += int(scored.sum())
    accumulator["covered"] += int((inside & scored).sum())
    accumulator["width"] += float(
        (upper - lower)[scored].sum(dtype=torch.float64))


def finalize_accumulator(accumulator: dict) -> dict[str, float | int]:
    count = int(accumulator["scored"])
    if not count:
        raise RuntimeError("evaluation had no scored elements")
    return {
        "sps_score": 100.0 * float(accumulator["numerator"]) / count,
        "coverage": int(accumulator["covered"]) / count,
        "mean_width": float(accumulator["width"]) / count,
        "scored_elements": count,
    }


@torch.inference_mode()
def evaluate_once(base, head, loader, normalizer, device) -> dict:
    base.eval()
    head.eval()
    candidate = new_accumulator()
    teacher = new_accumulator()
    point_digest = hashlib.sha256()
    for history, truth, teacher_lower, teacher_upper in loader:
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True).float()
        teacher_lower = teacher_lower.to(device, non_blocking=True).float()
        teacher_upper = teacher_upper.to(device, non_blocking=True).float()
        history_norm, _ = normalizer.preprocess(history, truth)
        truth = truth[..., :2]
        prediction, features = base.forward_with_features(
            history_norm.to(dtype=torch.float16))
        prediction = prediction.float()
        lower_extent, upper_extent = head(
            history_norm.float(), prediction,
            [value.float() for value in features],
            base.target_std.float())
        point, lower, upper = absolute_endpoints(
            prediction, lower_extent, upper_extent,
            normalizer.mean_target, normalizer.std_target)
        update_accumulator(candidate, point, lower, upper, truth)
        update_accumulator(
            teacher, point, teacher_lower, teacher_upper, truth)
        point_digest.update(
            point.detach().cpu().contiguous().numpy().tobytes())
    return {
        "candidate": finalize_accumulator(candidate),
        "teacher_cache_with_current_point": finalize_accumulator(teacher),
        "point_output_sha256": point_digest.hexdigest(),
    }


def decide_gate(
        args: argparse.Namespace, completed_updates: int, wall_seconds: float,
        metrics: dict, point_state_equal: bool) -> dict:
    candidate = metrics["candidate"]
    width_ratio = (
        candidate["mean_width"] / P1_FOLD04_REFERENCE["mean_width"])
    coverage_drop = (
        P1_FOLD04_REFERENCE["coverage"] - candidate["coverage"])
    checks = {
        "formal_configuration": (
            args.updates == 800 and args.batch_size == 8
            and args.lr == 1e-4 and args.held_fold == 4
            and args.min_sps_score == 53.20
            and args.time_budget_seconds <= 600.0),
        "completed_800_updates": completed_updates == 800,
        "wall_time_within_budget": wall_seconds <= args.time_budget_seconds,
        "absolute_sps_ge_threshold": candidate["sps_score"] >= args.min_sps_score,
        "coverage_drop_le_threshold": coverage_drop <= args.max_coverage_drop,
        "width_ratio_le_threshold": width_ratio <= args.max_width_ratio,
        "point_base_state_bitwise_equal": point_state_equal,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "sps_score": candidate["sps_score"],
            "coverage_drop_vs_p1_fold04": coverage_drop,
            "width_ratio_vs_p1_fold04": width_ratio,
            "wall_seconds": wall_seconds,
        },
        "thresholds": {
            "min_absolute_sps_score": args.min_sps_score,
            "max_coverage_drop_vs_p1_fold04": args.max_coverage_drop,
            "max_width_ratio_vs_p1_fold04": args.max_width_ratio,
            "time_budget_seconds": args.time_budget_seconds,
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Direct absolute-endpoint training requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "final.pth"
    summary_path = args.output_dir / "summary.json"
    if final_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite final Direct artifacts")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    point_file = s1.file_identity(args.point_checkpoint)
    if (args.expected_point_sha256 is not None
            and point_file["sha256"] != args.expected_point_sha256):
        raise ValueError("point checkpoint SHA-256 differs from requested identity")
    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    sample_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    s1.validate_s1_split(
        runtime_manifest, frozen_manifest, len(tail_windows), sample_folds)
    windows = common.repro.WindowDataset(trajectories, tail_windows)
    raw_aoas = np.asarray([
        int(metadata[window.trajectory]["aoa"])
        for window in tail_windows
    ], dtype=np.int64)
    cache_payload, cache_metadata = load_and_validate_cache(
        args.endpoint_cache, windows, sample_folds, fold_manifest,
        raw_aoas, args.expected_cache_sha256)
    indexed = IndexedTailDataset(windows, cache_payload)
    train_indices, eval_indices = select_fold_indices(
        cache_payload["fold"], args.held_fold)
    train_loader = DataLoader(
        Subset(indexed, train_indices), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    eval_loader = DataLoader(
        Subset(indexed, eval_indices), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    base, base_metadata = load_forecaster(
        args.point_checkpoint, normalizer, device)
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise RuntimeError("point base is not frozen")
    point_state_before = state_identity(base)
    head, head_initialization = build_head(args.start_head, device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    iterator = common.repro.cycling(train_loader)
    history_rows = []
    sums: dict[str, float] = {}
    count = 0
    completed_updates = 0
    started = time.perf_counter()

    for iteration in range(1, args.updates + 1):
        if time.perf_counter() - started > args.time_budget_seconds:
            break
        head.train()
        history, truth, teacher_lower, teacher_upper = next(iterator)
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True).float()
        teacher_lower = teacher_lower.to(device, non_blocking=True).float()
        teacher_upper = teacher_upper.to(device, non_blocking=True).float()
        history_norm, _ = normalizer.preprocess(history, truth)
        with torch.no_grad():
            prediction, features = base.forward_with_features(
                history_norm.to(dtype=torch.float16))
        prediction = prediction.float()
        optimizer.zero_grad(set_to_none=True)
        lower_extent, upper_extent = head(
            history_norm.float(), prediction,
            [value.float() for value in features],
            base.target_std.float())
        point, lower, upper = absolute_endpoints(
            prediction, lower_extent, upper_extent,
            normalizer.mean_target, normalizer.std_target)
        loss, components = endpoint_objective(
            args, point, lower, upper,
            teacher_lower, teacher_upper, truth)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed_updates = iteration
        for key, value in {"loss": loss.detach(), **components}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % 100 == 0 or iteration == args.updates:
            row = {
                "iteration": iteration,
                "elapsed_seconds": time.perf_counter() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
            }
            history_rows.append(row)
            print(json.dumps({"event": "train", **row}), flush=True)
            sums.clear()
            count = 0

    evaluation = evaluate_once(base, head, eval_loader, normalizer, device)
    point_state_after = state_identity(base)
    wall_seconds = time.perf_counter() - started
    point_state_equal = point_state_before == point_state_after
    gate = decide_gate(
        args, completed_updates, wall_seconds, evaluation, point_state_equal)
    status = "PASS_FOLD04_GATE" if gate["passed"] else "STOP_FOLD04_GATE"
    summary = {
        "status": status,
        "evidence_class": (
            "fold04-held tail pilot; folds0-3 optimization; one terminal "
            "fold04 evaluation; no official submission evidence"),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "counts": {
            "point_windows_unused": len(point_windows),
            "tail_windows": len(indexed),
            "train_windows_folds0_to3": len(train_indices),
            "held_fold04_windows": len(eval_indices),
        },
        "completed_updates": completed_updates,
        "wall_seconds": wall_seconds,
        "fold_manifest": fold_manifest,
        "point_checkpoint": point_file,
        "endpoint_cache": cache_metadata,
        "base": base_metadata,
        "head_initialization": head_initialization,
        "head_config": head.architecture_config(),
        "head_parameters": sum(value.numel() for value in head.parameters()),
        "point_identity": {
            "state_sha256_before": point_state_before,
            "state_sha256_after": point_state_after,
            "bitwise_equal": point_state_equal,
            "held_output_sha256": evaluation["point_output_sha256"],
        },
        "p1_fold04_reference": P1_FOLD04_REFERENCE,
        "evaluation": evaluation,
        "gate": gate,
        "training_history": history_rows,
    }
    torch.save({
        "phase": "real_absolute_endpoint_distillation",
        "iteration": completed_updates,
        "model_config": head.architecture_config(),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in head.state_dict().items()},
        "point_checkpoint": point_file,
        "endpoint_cache": cache_metadata["file"],
        "fold": args.held_fold,
        "metrics": evaluation,
        "gate": gate,
    }, final_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "status": status,
        "completed_updates": completed_updates,
        "wall_seconds": wall_seconds,
        "metrics": evaluation["candidate"],
        "gate": gate,
    }), flush=True)


if __name__ == "__main__":
    main()
