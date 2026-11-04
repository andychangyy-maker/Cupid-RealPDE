#!/usr/bin/env python3
"""Train the S2c FTQ graph from exact fold-safe P0 endpoint anchors only.

Only non-held point histories and exact post-selection/refinement P0 absolute
endpoints are used for optimization.  Tail targets are not part of training.
After the fixed final update, exact P0 endpoints for the held tail fold are
reconstructed and that fold's truth is consumed by one terminal evaluation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for search_path in (CODE_ROOT, Path(__file__).parent, EVALUATION_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_mean_expert_oof_blend as s1  # noqa: E402
import evaluate_p1_observation_gated_point_blend as point_gate  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
import train_p1_direct_native_endpoint_teacher as endpoint_teacher  # noqa: E402
import train_p1_direct_native_quantile_teacher as quantile_teacher  # noqa: E402
from models.fast_temporal_quantile_head import (  # noqa: E402
    FastTemporalQuantileHead,
    FastTemporalQuantileHeadConfig,
    fused_mean_point_heads,
    fused_mean_prediction,
)
from train_distilled_flow_corrector import load_forecaster  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


ARCHITECTURE = "fast_temporal_quantile_head_fused_mean_member_free"
FORMAL_POINT_SHA256 = endpoint_teacher.FORMAL_POINT_SHA256
WEIGHT_B = 0.625
WEIGHT_C = 0.25
EXPECTED_POINT_WINDOWS = 2576
EXPECTED_TAIL_WINDOWS = 725
EXPECTED_HELD_POINT = {0: 487, 1: 512, 2: 512, 3: 521, 4: 544}
EXPECTED_HELD_TAIL = {0: 137, 1: 144, 2: 144, 3: 147, 4: 153}
REFERENCE_WIDTH = (0.01075, 0.00550)
FOLD04_MIN_SPS = 53.20
FOLD04_MIN_COVERAGE = 0.8941232113
FOLD04_MAX_WIDTH = 0.0172787459
MAX_EXACT_CEILING_REGRET = 0.10
MAX_COVERAGE_DROP = 0.005
MAX_WIDTH_RATIO = 1.05
POOLED_PROMOTION_SPS = 53.462071
ANCHOR_FORMAT = (
    "p0_exact_postselection_absolute_endpoints_fp16_dynamic_fold_ram_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-point-sha256", required=True)
    parser.add_argument("--point-checkpoint-a", type=Path, required=True)
    parser.add_argument("--oof-dir-a", type=Path, required=True)
    parser.add_argument("--point-checkpoint-b", type=Path, required=True)
    parser.add_argument("--oof-dir-b", type=Path, required=True)
    parser.add_argument("--mean-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-fold", type=int, default=4)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--reconstruction-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--time-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    quantile_teacher.validate_sha256(args.expected_point_sha256)
    if args.expected_point_sha256 != FORMAL_POINT_SHA256:
        raise ValueError("formal point SHA-256 must be the frozen SHA758b asset")
    if args.held_fold not in range(5):
        raise ValueError("held-fold must lie in 0..4")
    frozen = {
        "updates": 4000,
        "batch_size": 8,
        "eval_batch_size": 16,
        "reconstruction_batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "time_budget_seconds": 1800.0,
        "seed": 0,
        "num_workers": 0,
        "device": "cuda:0",
    }
    if any(getattr(args, key) != value for key, value in frozen.items()):
        raise ValueError("exact-anchor-only FTQ hyperparameters are frozen")
    numeric = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.reconstruction_batch_size, args.lr, args.weight_decay,
        args.grad_clip, args.time_budget_seconds,
    )
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("formal numeric arguments must be finite")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def history_from_window(trajectories, window) -> torch.Tensor:
    uv = trajectories[window.trajectory][window.start:window.start + 20]
    pressure = torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype)
    return torch.cat((uv, pressure), dim=-1)


class HistoryDummyWindows(Dataset):
    """Return histories plus a dummy value; no future target is read."""

    def __init__(self, trajectories, windows):
        self.trajectories = trajectories
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, row: int):
        return history_from_window(
            self.trajectories, self.windows[row]), torch.empty(0)


class HistoryOnlyWindows(Dataset):
    """Label-free full-stream point-output audit dataset."""

    def __init__(self, trajectories, windows):
        self.trajectories = trajectories
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, row: int) -> torch.Tensor:
        return history_from_window(self.trajectories, self.windows[row])


class AnchorTrainingDataset(Dataset):
    """Align selected point histories with exact physical endpoint anchors."""

    def __init__(self, histories: HistoryDummyWindows,
                 indices: list[int], anchors: torch.Tensor):
        if len(indices) != len(anchors):
            raise ValueError("point history and endpoint-anchor rows differ")
        self.histories = histories
        self.indices = indices
        self.anchors = anchors

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        history, _dummy = self.histories[self.indices[row]]
        return history, self.anchors[row]


class HeldEndpointEvaluationDataset(Dataset):
    """The sole consumer of held-tail truth, used by one terminal pass."""

    def __init__(self, windows, indices: list[int], endpoints: torch.Tensor):
        if len(indices) != len(endpoints):
            raise ValueError("held truth and exact endpoint rows differ")
        self.windows = windows
        self.indices = indices
        self.endpoints = endpoints

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        history, truth = self.windows[self.indices[row]]
        return (
            history, truth, self.endpoints[row, ..., 0],
            self.endpoints[row, ..., 1])


def formal_partitions(point_folds, tail_folds, held_fold: int) -> dict:
    point_train, point_held = quantile_teacher.fold_partition_indices(
        point_folds, held_fold)
    tail_train, tail_held = quantile_teacher.fold_partition_indices(
        tail_folds, held_fold)
    expected = (
        EXPECTED_POINT_WINDOWS - EXPECTED_HELD_POINT[held_fold],
        EXPECTED_HELD_POINT[held_fold],
        EXPECTED_TAIL_WINDOWS - EXPECTED_HELD_TAIL[held_fold],
        EXPECTED_HELD_TAIL[held_fold],
    )
    actual = (
        len(point_train), len(point_held), len(tail_train), len(tail_held))
    if actual != expected:
        raise RuntimeError(
            f"fold{held_fold:02d} counts {actual} differ from {expected}")
    if (any(point_folds[index] == held_fold for index in point_train)
            or any(tail_folds[index] == held_fold for index in tail_train)
            or any(point_folds[index] != held_fold for index in point_held)
            or any(tail_folds[index] != held_fold for index in tail_held)):
        raise RuntimeError("held fold leaked across the formal partition")
    return {
        "point_train": point_train,
        "point_held": point_held,
        "tail_train_unused": tail_train,
        "tail_held": tail_held,
    }


def model_state_identity(model: torch.nn.Module) -> str:
    return quantile_teacher.state_identity(model.state_dict().items())


def build_head(device: torch.device, seed: int) -> FastTemporalQuantileHead:
    # S2c resets the seed immediately before constructing this exact graph.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = FastTemporalQuantileHead(FastTemporalQuantileHeadConfig())
    if head.architecture_config()["architecture"] != ARCHITECTURE:
        raise RuntimeError("FTQ constructor is not the S2c graph")
    return head.to(device=device, dtype=torch.float32)


def prepare_deploy_head(
        head: FastTemporalQuantileHead) -> FastTemporalQuantileHead:
    deployed = copy.deepcopy(head).eval().half()
    state = deployed.state_dict()
    if (not state
            or any(value.dtype != torch.float16 for value in state.values())
            or any(not torch.isfinite(value).all() for value in state.values())):
        raise FloatingPointError("deployed FP16 FTQ head is invalid")
    return deployed


@torch.inference_mode()
def reconstruct_exact_endpoint_rows(
        args: argparse.Namespace, dataset: Dataset, fold_labels: list[int],
        selected_indices: list[int], fold_manifest: dict, normalizer,
        device: torch.device, partition_name: str) -> tuple[torch.Tensor, dict]:
    """Rebuild exact P0 endpoints for arbitrary selected condition folds."""
    if not selected_indices:
        raise ValueError("exact endpoint reconstruction selection is empty")
    selected_folds = sorted({fold_labels[index] for index in selected_indices})
    if any(fold not in range(5) for fold in selected_folds):
        raise ValueError("selected endpoint fold lies outside 0..4")
    payload_a = torch.load(
        args.point_checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.point_checkpoint_b, map_location="cpu", weights_only=False)
    row_of_index = {
        index: row for row, index in enumerate(selected_indices)}
    endpoints = torch.empty(
        (len(selected_indices), 20, 32, 64, 2, 2), dtype=torch.float16)
    assigned = torch.zeros(len(selected_indices), dtype=torch.bool)

    model_c, metadata_c = point_gate.load_model(
        args.mean_expert_checkpoint, normalizer, device)
    c_loader = quantile_teacher.make_loader(
        dataset, selected_indices, args.reconstruction_batch_size,
        args.num_workers)
    prediction_c = quantile_teacher.collect_point_predictions(
        model_c, c_loader, normalizer, device)
    del model_c
    gc.collect()
    torch.cuda.empty_cache()

    fold_rows = []
    for fold in selected_folds:
        indices = [
            index for index in selected_indices if fold_labels[index] == fold]
        loader = quantile_teacher.make_loader(
            dataset, indices, args.reconstruction_batch_size,
            args.num_workers)
        model_a, metadata_a = s1.load_fold_model(
            payload_a, args.oof_dir_a / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_a, normalizer, device)
        prediction_a, lower_a, upper_a = (
            endpoint_teacher.collect_absolute_endpoints(
                model_a, loader, normalizer, device))
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        model_b, metadata_b = s1.load_fold_model(
            payload_b, args.oof_dir_b / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_b, normalizer, device)
        prediction_b, lower_b, upper_b = (
            endpoint_teacher.collect_absolute_endpoints(
                model_b, loader, normalizer, device))
        del model_b
        rows = torch.as_tensor(
            [row_of_index[index] for index in indices], dtype=torch.long)
        c_rows = prediction_c[rows]
        lower, upper = endpoint_teacher.compose_p0_absolute_endpoints(
            prediction_a.float(), lower_a.float(), upper_a.float(),
            prediction_b.float(), lower_b.float(), upper_b.float(),
            c_rows.float(), weight_b=WEIGHT_B, weight_c=WEIGHT_C)
        endpoints[rows, ..., 0] = lower.to(torch.float16)
        endpoints[rows, ..., 1] = upper.to(torch.float16)
        assigned[rows] = True
        fold_rows.append({
            "fold": fold,
            "samples": len(indices),
            "model_a": metadata_a,
            "model_b": metadata_b,
        })
        print(json.dumps({
            "event": "direct_exact_ftq_endpoint_reconstruction",
            "partition": partition_name,
            "fold": fold,
            "samples": len(indices),
        }), flush=True)
        del (
            prediction_a, lower_a, upper_a, prediction_b, lower_b, upper_b,
            rows, c_rows, lower, upper)
        gc.collect()
        torch.cuda.empty_cache()

    del payload_a, payload_b, prediction_c
    if not assigned.all():
        raise RuntimeError("exact endpoint reconstruction left unassigned rows")
    if (not torch.isfinite(endpoints).all()
            or torch.any(endpoints[..., 0] > endpoints[..., 1])):
        raise FloatingPointError("reconstructed exact P0 endpoints are invalid")
    return endpoints, {
        "format": ANCHOR_FORMAT,
        "partition": partition_name,
        "samples": len(endpoints),
        "selected_folds": selected_folds,
        "targets_read": False,
        "weights": {"b": WEIGHT_B, "c": WEIGHT_C},
        "stored_bytes": endpoints.numel() * endpoints.element_size(),
        "mean_expert": metadata_c,
        "folds": fold_rows,
    }


def absolute_endpoints_from_ftq(
        prediction_normalized: torch.Tensor,
        lower_extent: torch.Tensor, upper_extent: torch.Tensor,
        normalizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (lower_extent.shape != upper_extent.shape
            or prediction_normalized.shape[:-1] != lower_extent.shape[:-1]
            or lower_extent.shape[-1] != 2
            or prediction_normalized.shape[-1] < 2):
        raise ValueError("FTQ point and extent shapes do not align")
    point = common.to_physical_target(
        prediction_normalized.float(), normalizer)[..., :2]
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower = point - lower_extent.float() * scale
    upper = point + upper_extent.float() * scale
    if (not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
            or torch.any(lower > upper)):
        raise FloatingPointError("FTQ absolute endpoints are invalid")
    return point, lower, upper


def physical_extents_from_absolute(
        point: torch.Tensor, lower: torch.Tensor,
        upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert absolute bounds to the signed physical extents used by SPS."""
    if point.shape != lower.shape or lower.shape != upper.shape:
        raise ValueError("point and absolute endpoint shapes do not align")
    lower_extent = point - lower
    upper_extent = upper - point
    if (not torch.isfinite(lower_extent).all()
            or not torch.isfinite(upper_extent).all()
            or torch.any(lower_extent + upper_extent < 0.0)):
        raise FloatingPointError("absolute endpoints do not form ordered bounds")
    return lower_extent, upper_extent


def exact_endpoint_objective(
        prediction_normalized: torch.Tensor,
        lower_extent: torch.Tensor, upper_extent: torch.Tensor,
        teacher_endpoints: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if teacher_endpoints.shape[-2:] != (2, 2):
        raise ValueError("teacher endpoints must have (...,2,2) trailing shape")
    _point, lower, upper = absolute_endpoints_from_ftq(
        prediction_normalized, lower_extent, upper_extent, normalizer)
    teacher_lower = teacher_endpoints[..., 0].to(lower.dtype)
    teacher_upper = teacher_endpoints[..., 1].to(upper.dtype)
    reference = lower.new_tensor(REFERENCE_WIDTH).view(1, 1, 1, 1, 2)
    lower_loss = F.smooth_l1_loss(
        (lower - teacher_lower) / reference, torch.zeros_like(lower))
    upper_loss = F.smooth_l1_loss(
        (upper - teacher_upper) / reference, torch.zeros_like(upper))
    loss = 0.5 * (lower_loss + upper_loss)
    return loss, {
        "endpoint_distill": loss.detach(),
        "mean_width": (upper - lower).mean().detach(),
    }


@torch.inference_mode()
def direct_point_stream_identity(
        windows: Dataset, model, fused_heads, normalizer,
        device: torch.device, batch_size: int) -> dict:
    loader = DataLoader(
        windows, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    digest = hashlib.sha256()
    rows = 0
    model.eval()
    for history in loader:
        history = history.to(device, non_blocking=True)
        normalized = (
            history - normalizer.mean_input) / normalizer.std_input
        x_half = normalized.to(dtype=torch.float16)
        features = model._decode(x_half)
        prediction = fused_mean_prediction(
            model, x_half, features, fused_heads)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Direct point audit is non-finite")
        value = prediction.detach().cpu().contiguous()
        digest.update(value.numpy().tobytes(order="C"))
        rows += len(value)
    if rows != len(windows):
        raise RuntimeError("Direct point audit ended early")
    return {
        "sha256": digest.hexdigest(),
        "rows": rows,
        "dtype": str(value.dtype),
        "ordering": "all 2576 point histories then all 725 tail histories",
        "targets_read": False,
    }


@torch.inference_mode()
def evaluate_terminal(
        model, fused_heads, head: FastTemporalQuantileHead,
        loader: DataLoader, normalizer, device: torch.device) -> dict:
    model.eval()
    head.eval()
    if {value.dtype for value in head.state_dict().values()} != {torch.float16}:
        raise ValueError("terminal evaluation requires deployed FP16 FTQ head")
    student_accumulator = quantile_teacher.new_accumulator()
    teacher_accumulator = quantile_teacher.new_accumulator()
    rows = 0
    point_digest = hashlib.sha256()
    for history, truth, teacher_lower, teacher_upper in loader:
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True).float()[..., :2]
        teacher_lower = teacher_lower.to(
            device, non_blocking=True).float()
        teacher_upper = teacher_upper.to(
            device, non_blocking=True).float()
        normalized = (
            history - normalizer.mean_input) / normalizer.std_input
        x_half = normalized.to(dtype=torch.float16)
        features = model._decode(x_half)
        prediction_norm = fused_mean_prediction(
            model, x_half, features, fused_heads)
        quantiles, lower_extent, upper_extent = head(
            x_half, prediction_norm, features, model.target_std)
        point, lower, upper = absolute_endpoints_from_ftq(
            prediction_norm, lower_extent, upper_extent, normalizer)
        if (not torch.isfinite(quantiles).all()
                or not torch.isfinite(teacher_lower).all()
                or not torch.isfinite(teacher_upper).all()
                or torch.any(teacher_lower > teacher_upper)):
            raise FloatingPointError("terminal endpoint tensors are invalid")
        quantile_teacher.update_accumulator(
            student_accumulator, point,
            *physical_extents_from_absolute(point, lower, upper), truth)
        quantile_teacher.update_accumulator(
            teacher_accumulator, point,
            *physical_extents_from_absolute(
                point, teacher_lower, teacher_upper), truth)
        point_digest.update(
            point.detach().cpu().contiguous().numpy().tobytes(order="C"))
        rows += len(history)
    if rows != len(loader.dataset):
        raise RuntimeError("held-tail terminal evaluation ended early")
    return {
        "student": quantile_teacher.finalize_accumulator(
            student_accumulator),
        "exact_p0_ceiling_with_direct_point": (
            quantile_teacher.finalize_accumulator(teacher_accumulator)),
        "held_rows": rows,
        "truth_passes": 1,
        "point_output_sha256": point_digest.hexdigest(),
    }


def student_gate(
        args: argparse.Namespace, evaluation: dict | None,
        point_state_equal: bool, point_output_equal: bool,
        completed_updates: int, training_seconds: float,
        deployed_head_dtype: str) -> dict:
    formal = {
        "updates": args.updates == 4000,
        "batch_size": args.batch_size == 8,
        "eval_batch_size": args.eval_batch_size == 16,
        "reconstruction_batch_size": args.reconstruction_batch_size == 16,
        "lr": args.lr == 1e-4,
        "weight_decay": args.weight_decay == 1e-5,
        "grad_clip": args.grad_clip == 1.0,
        "time_budget_seconds": args.time_budget_seconds == 1800.0,
        "seed": args.seed == 0,
        "num_workers": args.num_workers == 0,
        "device": args.device == "cuda:0",
        "held_fold": args.held_fold in range(5),
    }
    checks = {
        "formal_configuration": all(formal.values()),
        "completed_fixed_4000_updates": completed_updates == 4000,
        "training_within_1800_seconds": (
            training_seconds <= args.time_budget_seconds),
        "student_gate_uses_deployed_fp16": deployed_head_dtype == "float16",
        "point_state_bitwise_equal": point_state_equal,
        "full_2576_725_direct_point_output_bitwise_equal": point_output_equal,
        "terminal_evaluation_completed": evaluation is not None,
    }
    thresholds = {
        "max_exact_ceiling_regret": MAX_EXACT_CEILING_REGRET,
        "max_coverage_drop_vs_exact_ceiling": MAX_COVERAGE_DROP,
        "max_width_ratio_vs_exact_ceiling": MAX_WIDTH_RATIO,
        "fold04_min_sps": FOLD04_MIN_SPS,
        "fold04_min_coverage": FOLD04_MIN_COVERAGE,
        "fold04_max_width": FOLD04_MAX_WIDTH,
    }
    observed = {"training_seconds": training_seconds}
    if evaluation is not None:
        student = evaluation["student"]
        teacher = evaluation["exact_p0_ceiling_with_direct_point"]
        regret = teacher["sps_score"] - student["sps_score"]
        coverage_floor = teacher["coverage"] - MAX_COVERAGE_DROP
        width_ceiling = teacher["mean_width"] * MAX_WIDTH_RATIO
        checks.update({
            "exact_ceiling_regret_le_0_10": (
                regret <= MAX_EXACT_CEILING_REGRET),
            "coverage_drop_vs_exact_ceiling_le_0_005": (
                student["coverage"] >= coverage_floor),
            "width_le_1_05x_exact_ceiling": (
                student["mean_width"] <= width_ceiling),
        })
        if args.held_fold == 4:
            checks.update({
                "fold04_sps_ge_53_20": (
                    student["sps_score"] >= FOLD04_MIN_SPS),
                "fold04_coverage_ge_0_8941232113": (
                    student["coverage"] >= FOLD04_MIN_COVERAGE),
                "fold04_width_le_0_0172787459": (
                    student["mean_width"] <= FOLD04_MAX_WIDTH),
            })
        observed.update({
            "student": student,
            "exact_p0_ceiling_with_direct_point": teacher,
            "exact_ceiling_regret": regret,
            "relative_min_coverage": coverage_floor,
            "relative_max_mean_width": width_ceiling,
        })
    else:
        checks.update({
            "exact_ceiling_regret_le_0_10": False,
            "coverage_drop_vs_exact_ceiling_le_0_005": False,
            "width_le_1_05x_exact_ceiling": False,
        })
        if args.held_fold == 4:
            checks.update({
                "fold04_sps_ge_53_20": False,
                "fold04_coverage_ge_0_8941232113": False,
                "fold04_width_le_0_0172787459": False,
            })
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "formal_fields": formal,
        "observed": observed,
        "thresholds": thresholds,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal exact-anchor-only FTQ training requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    point_file = s1.file_identity(args.point_checkpoint)
    if (point_file["sha256"] != FORMAL_POINT_SHA256
            or point_file["sha256"] != args.expected_point_sha256):
        raise ValueError("point checkpoint is not frozen SHA758b")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    process_started = time.perf_counter()

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    if (len(point_windows) != EXPECTED_POINT_WINDOWS
            or len(tail_windows) != EXPECTED_TAIL_WINDOWS):
        raise RuntimeError("frozen 2576/725 split changed")
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    if runtime_manifest != frozen_manifest:
        raise RuntimeError("runtime split differs from frozen manifest")
    tail_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    point_folds = quantile_teacher.partition_fold_labels(
        runtime_manifest, fold_manifest, "train_starts")
    partitions = formal_partitions(point_folds, tail_folds, args.held_fold)
    point_train = partitions["point_train"]
    tail_held = partitions["tail_held"]

    formal_sources = endpoint_teacher.validate_formal_source_files(args)
    oof_contracts = endpoint_teacher.validate_formal_oof_contracts(
        args, fold_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    point_histories = HistoryDummyWindows(trajectories, point_windows)
    tail_histories = HistoryDummyWindows(trajectories, tail_windows)
    point_audit = HistoryOnlyWindows(
        trajectories, [*point_windows, *tail_windows])

    anchor, anchor_metadata = reconstruct_exact_endpoint_rows(
        args, point_histories, point_folds, point_train, fold_manifest,
        normalizer, device, "point_nonheld_training_anchors")
    anchor_dataset = AnchorTrainingDataset(
        point_histories, point_train, anchor)
    anchor_loader = DataLoader(
        anchor_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    anchor_iterator = common.repro.cycling(anchor_loader)

    model, model_metadata = load_forecaster(
        args.point_checkpoint, normalizer, device)
    fused_heads = fused_mean_point_heads(model).to(device).eval()
    fused_heads.requires_grad_(False)
    point_state_before = model_state_identity(model)
    point_output_before = direct_point_stream_identity(
        point_audit, model, fused_heads, normalizer,
        device, args.eval_batch_size)
    head = build_head(device, args.seed)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)

    history_rows = []
    sums: dict[str, float] = {}
    count = 0
    completed_updates = 0
    training_started = time.perf_counter()
    for iteration in range(1, args.updates + 1):
        if time.perf_counter() - training_started > args.time_budget_seconds:
            break
        history, teacher_endpoints = next(anchor_iterator)
        history = history.to(device, non_blocking=True)
        teacher_endpoints = teacher_endpoints.to(
            device, non_blocking=True).float()
        normalized = (
            history - normalizer.mean_input) / normalizer.std_input
        x_half = normalized.to(dtype=torch.float16)
        with torch.no_grad():
            features = model._decode(x_half)
            prediction_norm = fused_mean_prediction(
                model, x_half, features, fused_heads)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        _quantiles, lower_extent, upper_extent = head(
            x_half.float(), prediction_norm.float(),
            [feature.float() for feature in features],
            model.target_std.float())
        loss, components = exact_endpoint_objective(
            prediction_norm.float(), lower_extent, upper_extent,
            teacher_endpoints, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite exact endpoint loss at update {iteration}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed_updates = iteration
        for key, value in {"loss": loss.detach(), **components}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % 250 == 0 or iteration == args.updates:
            row = {
                "iteration": iteration,
                "elapsed_seconds": time.perf_counter() - training_started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{key}_mean": value / count
                    for key, value in sums.items()
                },
            }
            history_rows.append(row)
            print(json.dumps({
                "event": "direct_exact_ftq_train", **row}), flush=True)
            sums.clear()
            count = 0
    training_seconds = time.perf_counter() - training_started

    deploy_head = prepare_deploy_head(head).cpu()
    point_state_after = model_state_identity(model)
    point_output_after = direct_point_stream_identity(
        point_audit, model, fused_heads, normalizer,
        device, args.eval_batch_size)
    point_state_equal = point_state_before == point_state_after
    point_output_equal = point_output_before == point_output_after
    del (
        anchor_iterator, anchor_loader, anchor_dataset, anchor,
        optimizer, scheduler, head, fused_heads, model)
    gc.collect()
    torch.cuda.empty_cache()

    evaluation = None
    terminal_endpoint_metadata = None
    reloaded_point_state_equal = False
    if completed_updates == args.updates:
        held_endpoints, terminal_endpoint_metadata = (
            reconstruct_exact_endpoint_rows(
                args, tail_histories, tail_folds, tail_held, fold_manifest,
                normalizer, device, "held_tail_terminal_exact_ceiling"))
        tail_truth_windows = common.repro.WindowDataset(
            trajectories, tail_windows)
        terminal_dataset = HeldEndpointEvaluationDataset(
            tail_truth_windows, tail_held, held_endpoints)
        terminal_loader = DataLoader(
            terminal_dataset, batch_size=args.eval_batch_size,
            shuffle=False, num_workers=0, pin_memory=True)
        deploy_model, _deploy_metadata = load_forecaster(
            args.point_checkpoint, normalizer, device)
        deploy_fused_heads = fused_mean_point_heads(
            deploy_model).to(device).eval()
        deploy_fused_heads.requires_grad_(False)
        reloaded_point_state_equal = (
            model_state_identity(deploy_model) == point_state_before)
        deploy_head = deploy_head.to(device=device, dtype=torch.float16)
        evaluation = evaluate_terminal(
            deploy_model, deploy_fused_heads, deploy_head,
            terminal_loader, normalizer, device)
        deploy_head = deploy_head.cpu()
        del (
            deploy_model, deploy_fused_heads, terminal_loader,
            terminal_dataset, tail_truth_windows, held_endpoints)
        gc.collect()
        torch.cuda.empty_cache()

    point_state_equal = point_state_equal and reloaded_point_state_equal
    gate = student_gate(
        args, evaluation, point_state_equal, point_output_equal,
        completed_updates, training_seconds, "float16")
    status = (
        f"PASS_DIRECT_EXACT_ENDPOINT_FTQ_FOLD{args.held_fold:02d}"
        if gate["passed"]
        else f"STOP_DIRECT_EXACT_ENDPOINT_FTQ_FOLD{args.held_fold:02d}")
    checkpoint_path = args.output_dir / "direct_exact_endpoint_ftq_student.pth"
    torch.save({
        "architecture": ARCHITECTURE,
        "head_config": deploy_head.architecture_config(),
        "head_state_dict": {
            key: value.detach().cpu()
            for key, value in deploy_head.state_dict().items()
        },
        "deployed_head_dtype": "float16",
        "point_checkpoint": point_file,
        "held_fold": args.held_fold,
        "updates": completed_updates,
        "evaluation": evaluation,
        "gate": gate,
    }, checkpoint_path)
    summary = {
        "status": status,
        "evidence_class": (
            f"fold{args.held_fold:02d}-held exact-P0-anchor-only FTQ pilot; "
            "nonheld point histories only; one terminal held-tail truth pass; "
            "local diagnostic, not official evidence"),
        "mechanism": {
            "training_target": (
                "exact A/B post-selection/refinement absolute endpoints plus "
                "exact P0 B=.625/C=.25 composition"),
            "tail_training_truth_read": False,
            "ftq_graph": ARCHITECTURE,
            "deployment_dtype": "float16",
        },
        "point_checkpoint": point_file,
        "model": model_metadata,
        "formal_sources": formal_sources,
        "oof_contracts": oof_contracts,
        "held_fold": args.held_fold,
        "counts": {
            "point_anchor_train": len(point_train),
            "point_held_targets_unused": len(partitions["point_held"]),
            "tail_train_truth_unused": len(partitions["tail_train_unused"]),
            "tail_held_terminal_evaluation": len(tail_held),
        },
        "point_anchor": anchor_metadata,
        "terminal_exact_ceiling": terminal_endpoint_metadata,
        "held_fold_evaluations_during_training": 0,
        "held_tail_truth_passes": (
            0 if evaluation is None else evaluation["truth_passes"]),
        "completed_updates": completed_updates,
        "training_seconds": training_seconds,
        "wall_seconds": time.perf_counter() - process_started,
        "fold_manifest": fold_manifest,
        "point_state": {
            "before": point_state_before,
            "after": point_state_after,
            "reloaded_equal": reloaded_point_state_equal,
            "bitwise_equal": point_state_equal,
        },
        "full_direct_point_output_stream": {
            "before": point_output_before,
            "after": point_output_after,
            "bitwise_equal": point_output_equal,
        },
        "evaluation": evaluation,
        "gate": gate,
        "next_gate": {
            "five_distinct_held_folds_required": True,
            "pooled_fixed725_sps_min": POOLED_PROMOTION_SPS,
            "not_evaluated_by_this_single_fold_script": True,
        },
        "training_history": history_rows,
        "checkpoint": s1.file_identity(checkpoint_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "direct_exact_endpoint_ftq_complete",
        "status": status,
        "evaluation": evaluation,
        "gate": gate,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
