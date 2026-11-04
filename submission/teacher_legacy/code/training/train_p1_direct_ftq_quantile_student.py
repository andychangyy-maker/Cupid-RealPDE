#!/usr/bin/env python3
"""Distill a gated native 7q teacher into the latency-qualified FTQ graph.

The native teacher is materialized once, deterministically, on point folds
0--3 (2032 windows) plus tail folds 0--3 (572 windows).  Only one fused FP16
7q cache is created, inside a temporary directory, and it is removed when the
run exits.  Held point fold 04 targets (544) are never read; their histories
participate only in a label-free before/after point-output identity audit.  Held
tail fold 04 (153) is evaluated exactly once after the fixed final update.
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
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, Path(__file__).parent, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_mean_expert_oof_blend as s1  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
import train_p1_direct_native_quantile_teacher as teacher_train  # noqa: E402
from models.fast_temporal_quantile_head import (  # noqa: E402
    FastTemporalQuantileHead,
    FastTemporalQuantileHeadConfig,
    fused_mean_point_heads,
    fused_mean_prediction,
)
from train_distilled_flow_corrector import load_forecaster  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402
import train_ensemble_uq_oof as uq_train  # noqa: E402


HELD_FOLD = 4
EXPECTED_POINT_TRAIN = 2032
EXPECTED_POINT_HELD = 544
EXPECTED_TAIL_TRAIN = 572
EXPECTED_TAIL_HELD = 153
EXPECTED_DISTILL_ROWS = EXPECTED_POINT_TRAIN + EXPECTED_TAIL_TRAIN
LABEL_SHAPE = (20, 32, 64, 2, 7)
LABEL_FORMAT = "native_teacher_recentered_ftq_residual_7q_fp16_v1"
STUDENT_MIN_SPS = 53.20
MAX_TEACHER_REGRET = 0.10
MAX_COVERAGE_DROP = 0.005
MAX_WIDTH_RATIO = 1.05
POOLED_PROMOTION_SPS = 53.462071
REFERENCE_WIDTH = (0.01075, 0.00550)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-point-sha256", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--start-head", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--materialize-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--time-budget-seconds", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.materialize_batch_size, args.lr, args.time_budget_seconds,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("positive training arguments must be finite")
    if any(not math.isfinite(value) or value < 0 for value in (
            args.weight_decay, args.grad_clip)):
        raise ValueError("optimizer arguments must be non-negative")
    # The temporary mmap and WindowDataset references are deliberately local
    # to one process; multi-worker reopening is not part of this contract.
    if args.num_workers != 0:
        raise ValueError("deterministic FTQ distillation requires num-workers=0")
    teacher_train.validate_sha256(args.expected_point_sha256)
    teacher_train.validate_sha256(args.expected_teacher_sha256)
    if args.expected_point_sha256 != teacher_train.FORMAL_POINT_SHA256:
        raise ValueError("formal point SHA-256 must be the frozen SHA758b asset")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndexedWindows(Dataset):
    """One deterministic list spanning selected point then tail windows."""

    def __init__(self, point_dataset, point_indices: list[int],
                 tail_dataset, tail_indices: list[int]):
        self.rows = [
            (point_dataset, index) for index in point_indices
        ] + [
            (tail_dataset, index) for index in tail_indices
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, row: int):
        dataset, index = self.rows[row]
        history, _target = dataset[index]
        return history


class HistoryOnlyWindows(Dataset):
    """Read only the 20-step input prefix for a full point-stream audit."""

    def __init__(self, trajectories, windows):
        self.trajectories = trajectories
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, row: int) -> torch.Tensor:
        item = self.windows[row]
        uv = self.trajectories[
            item.trajectory][item.start:item.start + 20]
        pressure = torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype)
        return torch.cat((uv, pressure), dim=-1)


class DistillationDataset(Dataset):
    def __init__(self, windows: IndexedWindows, labels: np.ndarray):
        if len(windows) != len(labels):
            raise ValueError("teacher label rows do not align with windows")
        self.windows = windows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, row: int):
        # Copy one read-only mmap row so PyTorch never writes through it.
        label = torch.from_numpy(np.array(self.labels[row], copy=True))
        return self.windows[row], label


def build_head(start: Path | None, device: torch.device,
               seed: int = 0) -> tuple[FastTemporalQuantileHead, dict]:
    # Match the S2c package wrapper: reset immediately before constructing the
    # FTQ graph so unrelated point-model construction cannot consume its RNG.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    requested = FastTemporalQuantileHeadConfig()
    head = FastTemporalQuantileHead(requested)
    metadata = {
        "initialization": "fresh_exact_ftq_graph",
        "constructor_seed_reset": seed,
    }
    if start is not None:
        payload = torch.load(start, map_location="cpu", weights_only=False)
        config = dict(payload.get("head_config", payload.get(
            "model_config", {})))
        architecture = config.pop("architecture", None)
        if architecture not in {
                "fast_temporal_quantile_head_fused_mean_member_free",
                "fast_temporal_quantile_head"}:
            raise ValueError("start-head is not the latency-qualified FTQ graph")
        for key in ("feature_channels", "quantile_levels"):
            if key in config:
                config[key] = tuple(config[key])
        if FastTemporalQuantileHeadConfig(**config) != requested:
            raise ValueError("start-head FTQ config differs from exact graph")
        state = payload.get("head_state_dict", payload.get("model_state_dict"))
        if not isinstance(state, dict):
            raise ValueError("start-head has no state dict")
        head.load_state_dict(state, strict=True)
        metadata = {
            "initialization": "warm_start_exact_ftq_graph",
            "file": s1.file_identity(start),
            "constructor_seed_reset": seed,
        }
    return head.to(device=device, dtype=torch.float32), metadata


def prepare_deploy_head(
        head: FastTemporalQuantileHead) -> FastTemporalQuantileHead:
    deployed = copy.deepcopy(head).eval().half()
    state = deployed.state_dict()
    if (not state
            or any(value.dtype != torch.float16 for value in state.values())
            or any(not torch.isfinite(value).all() for value in state.values())):
        raise FloatingPointError("deployed FP16 FTQ head is invalid")
    return deployed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def materialize_teacher_labels(
        path: Path, windows: IndexedWindows, model,
        fused_heads, normalizer, device: torch.device,
        batch_size: int) -> dict:
    expected_shape = (len(windows), *LABEL_SHAPE)
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * 2
    if len(windows) != EXPECTED_DISTILL_ROWS:
        raise RuntimeError("distillation rows differ from 2032+572")
    free = shutil.disk_usage(path.parent).free
    if free < expected_bytes + 512 * 1024**2:
        raise OSError(
            "insufficient temporary disk for the single fused 7q cache")
    labels = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=expected_shape)
    loader = DataLoader(
        windows, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    point_digest = hashlib.sha256()
    offset = 0
    model.eval()
    for history in loader:
        history = history.to(device, non_blocking=True)
        normalized = (
            history - normalizer.mean_input) / normalizer.std_input
        x_half = normalized.to(dtype=torch.float16)
        values = model.forward_uq_training(x_half)
        native_prediction, quantiles, features = values[0], values[2], values[5]
        direct_prediction = fused_mean_prediction(
            model, x_half, features, fused_heads)
        # Preserve the native absolute quantiles while switching the center to
        # the exact Direct fused point path used by the deployed FTQ graph.
        recentered = quantiles + (
            native_prediction[..., :2] - direct_prediction[..., :2]
        ).unsqueeze(-1)
        batch = recentered.detach().cpu().numpy().astype(
            np.float16, copy=False)
        stop = offset + len(batch)
        labels[offset:stop] = batch
        point_digest.update(direct_prediction.detach().cpu().contiguous(
        ).numpy().tobytes(order="C"))
        offset = stop
        if offset % 256 == 0 or offset == len(windows):
            print(json.dumps({
                "event": "teacher_7q_materialize",
                "rows": offset, "total": len(windows),
            }), flush=True)
    labels.flush()
    del labels
    if offset != len(windows):
        raise RuntimeError("teacher materialization ended early")
    return {
        "format": LABEL_FORMAT,
        "shape": list(expected_shape),
        "dtype": "float16",
        "tensor_bytes": expected_bytes,
        "file_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "point_output_sha256": point_digest.hexdigest(),
        "ordering": "point folds0-3 in dataset order, then tail folds0-3",
        "held_point_rows": 0,
        "held_tail_rows": 0,
    }


def quantile_distillation_objective(
        student: torch.Tensor, teacher: torch.Tensor,
        target_std: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape or student.shape[-2:] != (2, 7):
        raise ValueError("student/teacher quantiles must share (...,2,7)")
    scale = target_std[:2].to(
        device=student.device, dtype=student.dtype).view(
            1, 1, 1, 1, 2, 1)
    reference = student.new_tensor(REFERENCE_WIDTH).view(
        1, 1, 1, 1, 2, 1)
    error = (student - teacher.to(student.dtype)) * scale / reference
    return F.smooth_l1_loss(error, torch.zeros_like(error))


@torch.inference_mode()
def direct_point_stream_identity(
        windows: Dataset, model, fused_heads, normalizer,
        device: torch.device, batch_size: int) -> dict:
    loader = DataLoader(
        windows, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    digest = hashlib.sha256()
    rows = 0
    dtype = None
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
            raise FloatingPointError("Direct point audit produced non-finite output")
        value = prediction.detach().cpu().contiguous()
        digest.update(value.numpy().tobytes(order="C"))
        rows += len(value)
        dtype = str(value.dtype)
    if rows != len(windows):
        raise RuntimeError("Direct point audit ended early")
    return {
        "sha256": digest.hexdigest(),
        "rows": rows,
        "dtype": dtype,
        "ordering": "all 2576 point histories then all 725 tail histories",
        "targets_read": False,
    }


def recenter_interval_extents(
        lower_native: torch.Tensor, upper_native: torch.Tensor,
        native_prediction: torch.Tensor, direct_prediction: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve native absolute endpoints under the Direct point center."""
    if (lower_native.shape != upper_native.shape
            or lower_native.shape[-1] != 2
            or native_prediction.shape != direct_prediction.shape
            or native_prediction.shape[:-1] != lower_native.shape[:-1]
            or native_prediction.shape[-1] < 2):
        raise ValueError("native/Direct interval tensors do not align")
    delta = direct_prediction[..., :2] - native_prediction[..., :2]
    lower_direct = lower_native + delta
    upper_direct = upper_native - delta
    if (not torch.isfinite(lower_direct).all()
            or not torch.isfinite(upper_direct).all()
            or torch.any(lower_direct + upper_direct < 0.0)):
        raise FloatingPointError("recentered Direct teacher interval is invalid")
    return lower_direct, upper_direct


@torch.inference_mode()
def evaluate_teacher_student(model, fused_heads, head, loader,
                             normalizer, device: torch.device) -> dict:
    model.eval()
    head.eval()
    if ({value.dtype for value in head.state_dict().values()}
            != {torch.float16}):
        raise ValueError("formal student evaluation requires deployed FP16 head")
    native_teacher_accumulator = teacher_train.new_accumulator()
    direct_teacher_accumulator = teacher_train.new_accumulator()
    student_accumulator = teacher_train.new_accumulator()
    point_digest = hashlib.sha256()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for history, truth in loader:
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True).float()[..., :2]
        normalized = (
            history - normalizer.mean_input) / normalizer.std_input
        x_half = normalized.to(dtype=torch.float16)
        values = model.forward_uq_training(x_half)
        native_prediction_norm, teacher_lower, teacher_upper, features = (
            values[0], values[3], values[4], values[5])
        prediction_norm = fused_mean_prediction(
            model, x_half, features, fused_heads)
        direct_teacher_lower, direct_teacher_upper = recenter_interval_extents(
            teacher_lower, teacher_upper,
            native_prediction_norm, prediction_norm)
        student_quantiles, student_lower, student_upper = head(
            x_half, prediction_norm, features, model.target_std)
        if any(not torch.isfinite(value).all() for value in (
                native_prediction_norm, prediction_norm,
                student_quantiles, teacher_lower, teacher_upper,
                student_lower, student_upper)):
            raise FloatingPointError("FP16 terminal evaluation is non-finite")
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        native_prediction = common.to_physical_target(
            native_prediction_norm.float(), normalizer)[..., :2]
        teacher_train.update_accumulator(
            native_teacher_accumulator, native_prediction,
            teacher_lower.float() * scale,
            teacher_upper.float() * scale, truth)
        teacher_train.update_accumulator(
            direct_teacher_accumulator, prediction,
            direct_teacher_lower.float() * scale,
            direct_teacher_upper.float() * scale, truth)
        teacher_train.update_accumulator(
            student_accumulator, prediction,
            student_lower.float() * scale,
            student_upper.float() * scale, truth)
        point_digest.update(
            prediction.cpu().contiguous().numpy().tobytes(order="C"))
    return {
        "teacher_native_formal_center": teacher_train.finalize_accumulator(
            native_teacher_accumulator),
        "teacher_recentered_to_direct": teacher_train.finalize_accumulator(
            direct_teacher_accumulator),
        "student": teacher_train.finalize_accumulator(student_accumulator),
        "point_output_sha256": point_digest.hexdigest(),
    }


def student_gate(args: argparse.Namespace, evaluation: dict,
                 point_state_equal: bool, point_output_equal: bool,
                 completed_updates: int,
                 deployed_head_dtype: str) -> dict:
    teacher = evaluation["teacher_recentered_to_direct"]
    student = evaluation["student"]
    reference = teacher_train.FOLD04_REFERENCE
    regret = teacher["sps_score"] - student["sps_score"]
    coverage_floor = reference["coverage"] - MAX_COVERAGE_DROP
    width_ceiling = reference["mean_width"] * MAX_WIDTH_RATIO
    checks = {
        "formal_configuration": (
            args.updates == 4000
            and args.batch_size == 8
            and args.eval_batch_size == 16
            and args.materialize_batch_size == 16
            and args.lr == 1e-4
            and args.weight_decay == 1e-5
            and args.grad_clip == 1.0
            and args.seed == 0
            and args.num_workers == 0
            and args.device == "cuda:0"
            and args.start_head is None),
        "completed_fixed_4000_updates": completed_updates == 4000,
        "student_gate_uses_deployed_fp16": deployed_head_dtype == "float16",
        "student_sps_ge_53_20": student["sps_score"] >= STUDENT_MIN_SPS,
        "teacher_regret_le_0_10": regret <= MAX_TEACHER_REGRET,
        "coverage_ge_reference_minus_0_005": (
            student["coverage"] >= coverage_floor),
        "width_le_1_05x_reference": (
            student["mean_width"] <= width_ceiling),
        "point_state_bitwise_equal": point_state_equal,
        "full_2576_725_direct_point_output_bitwise_equal": point_output_equal,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {"teacher_regret": regret},
        "thresholds": {
            "min_student_sps": STUDENT_MIN_SPS,
            "max_teacher_regret": MAX_TEACHER_REGRET,
            "min_coverage": coverage_floor,
            "max_mean_width": width_ceiling,
            "next_pooled_fixed725_sps": POOLED_PROMOTION_SPS,
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal FTQ distillation requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    args.temp_root.mkdir(parents=True, exist_ok=True)
    point_file = s1.file_identity(args.point_checkpoint)
    teacher_file = s1.file_identity(args.teacher_checkpoint)
    if (point_file["sha256"] != teacher_train.FORMAL_POINT_SHA256
            or point_file["sha256"] != args.expected_point_sha256):
        raise ValueError("point checkpoint SHA-256 differs from contract")
    if teacher_file["sha256"] != args.expected_teacher_sha256:
        raise ValueError("teacher checkpoint SHA-256 differs from contract")
    teacher_payload = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False)
    if (teacher_payload.get("architecture")
            != "p1_direct_native_7q_teacher"
            or tuple(teacher_payload.get("quantile_levels", ()))
            != teacher_train.QUANTILE_LEVELS
            or teacher_payload.get("point_checkpoint", {}).get("sha256")
            != point_file["sha256"]
            or not teacher_payload.get("gate", {}).get("passed")):
        raise ValueError("native teacher did not pass its exact input gate")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    process_started = time.perf_counter()

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    if runtime_manifest != frozen_manifest:
        raise RuntimeError("runtime split differs from frozen manifest")
    tail_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    point_folds = teacher_train.partition_fold_labels(
        runtime_manifest, fold_manifest, "train_starts")
    point_train, point_held = teacher_train.fold_partition_indices(point_folds)
    tail_train, tail_held = teacher_train.fold_partition_indices(tail_folds)
    if (len(point_train), len(point_held), len(tail_train), len(tail_held)) != (
            EXPECTED_POINT_TRAIN, EXPECTED_POINT_HELD,
            EXPECTED_TAIL_TRAIN, EXPECTED_TAIL_HELD):
        raise RuntimeError("fixed fold counts differ from 2032/544/572/153")
    point_dataset = common.repro.WindowDataset(trajectories, point_windows)
    tail_dataset = common.repro.WindowDataset(trajectories, tail_windows)
    distill_windows = IndexedWindows(
        point_dataset, point_train, tail_dataset, tail_train)
    point_audit_windows = HistoryOnlyWindows(
        trajectories, [*point_windows, *tail_windows])
    normalizer = common.repro.Normalizer(args.real_stats, device)
    model, model_metadata = load_forecaster(
        args.point_checkpoint, normalizer, device)
    uq_train.load_uq_state(model, teacher_payload["uq_state_dict"])
    model.requires_grad_(False)
    model.eval()
    fused_heads = fused_mean_point_heads(model).to(device).eval()
    fused_heads.requires_grad_(False)
    point_state_before = teacher_train.point_state_identity(model)
    point_output_before = direct_point_stream_identity(
        point_audit_windows, model, fused_heads, normalizer,
        device, args.eval_batch_size)
    head, head_initialization = build_head(
        args.start_head, device, seed=args.seed)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)

    cache_metadata = None
    history_rows = []
    completed_updates = 0
    training_seconds = 0.0
    with tempfile.TemporaryDirectory(
            prefix="p1-ftq-7q-", dir=args.temp_root) as temporary:
        cache_path = Path(temporary) / "teacher_7q_fp16.npy"
        cache_metadata = materialize_teacher_labels(
            cache_path, distill_windows, model, fused_heads,
            normalizer, device, args.materialize_batch_size)
        labels = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        train_dataset = DistillationDataset(distill_windows, labels)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed))
        iterator = common.repro.cycling(train_loader)
        sums: dict[str, float] = {}
        count = 0
        training_started = time.perf_counter()
        for iteration in range(1, args.updates + 1):
            if time.perf_counter() - training_started > args.time_budget_seconds:
                break
            history, teacher_quantiles = next(iterator)
            history = history.to(device, non_blocking=True)
            teacher_quantiles = teacher_quantiles.to(
                device, non_blocking=True).float()
            normalized = (
                history - normalizer.mean_input) / normalizer.std_input
            x_half = normalized.to(dtype=torch.float16)
            with torch.no_grad():
                features = model._decode(x_half)
                prediction = fused_mean_prediction(
                    model, x_half, features, fused_heads)
            optimizer.zero_grad(set_to_none=True)
            student_quantiles, _lower, _upper = head(
                x_half.float(), prediction.float(),
                [feature.float() for feature in features],
                model.target_std.float())
            loss = quantile_distillation_objective(
                student_quantiles, teacher_quantiles,
                normalizer.std_target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite FTQ distill loss at update {iteration}")
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            completed_updates = iteration
            sums["loss"] = sums.get("loss", 0.0) + float(loss)
            count += 1
            if iteration % 250 == 0 or iteration == args.updates:
                row = {
                    "iteration": iteration,
                    "elapsed_seconds": time.perf_counter() - training_started,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_distill_loss_mean": sums["loss"] / count,
                }
                history_rows.append(row)
                print(json.dumps({"event": "ftq_distill_train", **row}),
                      flush=True)
                sums.clear()
                count = 0
        training_seconds = time.perf_counter() - training_started
        del labels, train_dataset, train_loader, iterator
    # TemporaryDirectory has now removed the only 2.78-GiB pseudo-label file.

    deploy_head = prepare_deploy_head(head)
    held_loader = DataLoader(
        torch.utils.data.Subset(tail_dataset, tail_held),
        batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    evaluation = evaluate_teacher_student(
        model, fused_heads, deploy_head, held_loader, normalizer, device)
    point_state_after = teacher_train.point_state_identity(model)
    point_state_equal = point_state_before == point_state_after
    point_output_after = direct_point_stream_identity(
        point_audit_windows, model, fused_heads, normalizer,
        device, args.eval_batch_size)
    point_output_equal = point_output_before == point_output_after
    gate = student_gate(
        args, evaluation, point_state_equal, point_output_equal,
        completed_updates,
        deployed_head_dtype="float16")
    status = "PASS_FTQ_STUDENT_FOLD04" if gate["passed"] else (
        "STOP_FTQ_STUDENT_FOLD04")
    checkpoint_path = args.output_dir / "ftq_student.pth"
    torch.save({
        "architecture": "fast_temporal_quantile_head_fused_mean_member_free",
        "head_config": deploy_head.architecture_config(),
        "head_state_dict": {
            key: value.detach().cpu()
            for key, value in deploy_head.state_dict().items()},
        "deployed_head_dtype": "float16",
        "point_checkpoint": point_file,
        "teacher_checkpoint": teacher_file,
        "held_fold": HELD_FOLD,
        "updates": completed_updates,
        "evaluation": evaluation,
        "gate": gate,
    }, checkpoint_path)
    elapsed = time.perf_counter() - process_started
    summary = {
        "status": status,
        "evidence_class": (
            "fold04-held FTQ full-7q distillation; point2032+tail572 "
            "teacher pseudo-labels; one terminal tail evaluation; no official evidence"),
        "point_checkpoint": point_file,
        "teacher_checkpoint": teacher_file,
        "model": model_metadata,
        "head_initialization": head_initialization,
        "head_parameters": sum(
            value.numel() for value in deploy_head.parameters()),
        "deployed_head_dtype": "float16",
        "student_gate_uses_deployed_fp16": True,
        "counts": {
            "point_distill_folds0_to3": len(point_train),
            "point_held_fold04_targets_unused": len(point_held),
            "point_and_tail_histories_in_point_identity_audit": (
                len(point_audit_windows)),
            "tail_distill_folds0_to3": len(tail_train),
            "tail_held_fold04_terminal_eval": len(tail_held),
        },
        "held_fold_evaluations_during_training": 0,
        "completed_updates": completed_updates,
        "training_seconds": training_seconds,
        "wall_seconds": elapsed,
        "fold_manifest": fold_manifest,
        "temporary_teacher_cache": {
            **cache_metadata, "removed_after_training": True},
        "point_state": {
            "before": point_state_before,
            "after": point_state_after,
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
            "pooled_fixed725_sps_min": POOLED_PROMOTION_SPS,
            "not_evaluated_by_this_fold04_script": True,
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
        "event": "ftq_student_complete", "status": status,
        "evaluation": evaluation, "gate": gate,
        "checkpoint": summary["checkpoint"],
    }), flush=True)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
