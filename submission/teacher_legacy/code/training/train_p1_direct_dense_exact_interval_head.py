#!/usr/bin/env python3
"""One-shot fold-held pilot for the direct dense exact interval head.

The head is trained only on non-held point histories and fold-safe exact P0
absolute endpoints.  Tail truth is never used for optimization.  A held-tail
fold is read exactly once, after the fixed final update, for terminal scoring.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for search_path in (CODE_ROOT, Path(__file__).parent, EVALUATION_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import train_p1_direct_exact_endpoint_ftq_student as exact  # noqa: E402
from models.distilled_interval_head import (  # noqa: E402
    DistilledIntervalHead,
    DistilledIntervalHeadConfig,
)


# Reuse the already-audited split, exact-anchor, point-identity, and physical
# endpoint infrastructure without changing its frozen FTQ contract.
s1 = exact.s1
common = exact.common
endpoint_teacher = exact.endpoint_teacher
quantile_teacher = exact.quantile_teacher
load_forecaster = exact.load_forecaster
trajectory_folds = exact.trajectory_folds
fused_mean_point_heads = exact.fused_mean_point_heads
fused_mean_prediction = exact.fused_mean_prediction
HistoryDummyWindows = exact.HistoryDummyWindows
HistoryOnlyWindows = exact.HistoryOnlyWindows
AnchorTrainingDataset = exact.AnchorTrainingDataset
HeldEndpointEvaluationDataset = exact.HeldEndpointEvaluationDataset
formal_partitions = exact.formal_partitions
model_state_identity = exact.model_state_identity
absolute_endpoints_from_ftq = exact.absolute_endpoints_from_ftq
physical_extents_from_absolute = exact.physical_extents_from_absolute
exact_endpoint_objective = exact.exact_endpoint_objective
direct_point_stream_identity = exact.direct_point_stream_identity


ARCHITECTURE = "distilled_interval_head"
FORMAL_POINT_SHA256 = exact.FORMAL_POINT_SHA256
WEIGHT_B = exact.WEIGHT_B
WEIGHT_C = exact.WEIGHT_C
EXPECTED_POINT_WINDOWS = exact.EXPECTED_POINT_WINDOWS
EXPECTED_TAIL_WINDOWS = exact.EXPECTED_TAIL_WINDOWS
EXPECTED_HELD_POINT = exact.EXPECTED_HELD_POINT
EXPECTED_HELD_TAIL = exact.EXPECTED_HELD_TAIL
REFERENCE_WIDTH = exact.REFERENCE_WIDTH
FOLD04_MIN_SPS = exact.FOLD04_MIN_SPS
FOLD04_MIN_COVERAGE = exact.FOLD04_MIN_COVERAGE
FOLD04_MAX_WIDTH = exact.FOLD04_MAX_WIDTH
MAX_EXACT_CEILING_REGRET = exact.MAX_EXACT_CEILING_REGRET
MAX_COVERAGE_DROP = exact.MAX_COVERAGE_DROP
MAX_WIDTH_RATIO = exact.MAX_WIDTH_RATIO
POOLED_PROMOTION_SPS = exact.POOLED_PROMOTION_SPS

EXPECTED_HEAD_PARAMETERS = 45_568
MAX_UNREPRESENTABLE_FRACTION = 0.0
MAX_REPRESENTATION_SPS_LOSS_BOUND = 0.0
TRAINING_TIME_BUDGET_SECONDS = 1200.0
ANCHOR_FORMAT = (
    "p0_exact_postselection_absolute_endpoints_fp16_dynamic_fold_"
    "direct_dense_interval_v1")


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
    parser.add_argument(
        "--time-budget-seconds", type=float,
        default=TRAINING_TIME_BUDGET_SECONDS)
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
        "time_budget_seconds": TRAINING_TIME_BUDGET_SECONDS,
        "seed": 0,
        "num_workers": 0,
        "device": "cuda:0",
    }
    if any(getattr(args, key) != value for key, value in frozen.items()):
        raise ValueError("direct dense exact interval hyperparameters are frozen")
    numeric = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.reconstruction_batch_size, args.lr, args.weight_decay,
        args.grad_clip, args.time_budget_seconds,
    )
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("formal numeric arguments must be finite")


def build_head(device: torch.device, seed: int) -> DistilledIntervalHead:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = DistilledIntervalHead(DistilledIntervalHeadConfig())
    if head.architecture_config()["architecture"] != ARCHITECTURE:
        raise RuntimeError("constructor is not the frozen direct interval head")
    parameters = sum(value.numel() for value in head.parameters())
    if parameters != EXPECTED_HEAD_PARAMETERS:
        raise RuntimeError(
            f"direct interval head has {parameters} parameters, expected "
            f"{EXPECTED_HEAD_PARAMETERS}")
    return head.to(device=device, dtype=torch.float32)


def prepare_deploy_head(
        head: DistilledIntervalHead) -> DistilledIntervalHead:
    deployed = copy.deepcopy(head).eval().half()
    state = deployed.state_dict()
    if (not state
            or any(value.dtype != torch.float16 for value in state.values())
            or any(not torch.isfinite(value).all() for value in state.values())):
        raise FloatingPointError("deployed FP16 direct interval head is invalid")
    return deployed


def representability_from_physical(
        point: torch.Tensor, endpoints: torch.Tensor,
        config: DistilledIntervalHeadConfig | None = None) -> dict:
    """Measure exact-anchor support under the head's bounded parameterization."""
    config = config or DistilledIntervalHeadConfig()
    config.validate()
    if endpoints.shape[-2:] != (2, 2):
        raise ValueError("endpoints must have (...,2,2) trailing shape")
    if point.shape != endpoints.shape[:-1]:
        raise ValueError("point and endpoint shapes do not align")
    lower = endpoints[..., 0]
    upper = endpoints[..., 1]
    width = upper - lower
    finite = (
        torch.isfinite(point) & torch.isfinite(lower) & torch.isfinite(upper))
    positive_width = width > 0.0
    structurally_valid = finite & positive_width
    safe_width = torch.where(
        structurally_valid, width, torch.ones_like(width))
    reference = point.new_tensor(config.reference_total_widths).view(
        *((1,) * (point.ndim - 1)), 2)
    log_width_ratio = torch.log(safe_width / reference)
    center = 0.5 * (lower + upper) - point
    center_fraction = torch.abs(center) / safe_width
    # tanh cannot attain its endpoints: equality to either configured limit is
    # not exactly representable and therefore fails the strict preflight.
    width_beyond_limit = (
        structurally_valid
        & (torch.abs(log_width_ratio) >= config.max_log_width_ratio))
    center_beyond_limit = (
        structurally_valid
        & (center_fraction >= config.max_center_fraction))
    structurally_invalid = ~structurally_valid
    violating = (
        structurally_invalid | width_beyond_limit | center_beyond_limit)
    total = point.numel()
    violating_count = int(violating.sum().item())
    fraction = violating_count / total

    valid_log = torch.abs(log_width_ratio)[structurally_valid]
    valid_center = center_fraction[structurally_valid]
    max_abs_log_width_ratio = (
        float(valid_log.max().item()) if valid_log.numel() else None)
    max_center_fraction = (
        float(valid_center.max().item()) if valid_center.numel() else None)
    structurally_invalid_count = int(structurally_invalid.sum().item())
    return {
        "total_scalar_endpoints": total,
        "structurally_invalid": structurally_invalid_count,
        "width_beyond_limit": int(width_beyond_limit.sum().item()),
        "center_beyond_limit": int(center_beyond_limit.sum().item()),
        "unrepresentable_union": violating_count,
        "unrepresentable_fraction": fraction,
        "representable_fraction": 1.0 - fraction,
        "max_abs_log_width_ratio": max_abs_log_width_ratio,
        "max_center_fraction": max_center_fraction,
        "min_width_log_margin": (
            None if max_abs_log_width_ratio is None
            else config.max_log_width_ratio - max_abs_log_width_ratio),
        "min_center_fraction_margin": (
            None if max_center_fraction is None
            else config.max_center_fraction - max_center_fraction),
        "worst_case_sps_loss_bound": 100.0 * fraction,
        "passed": (
            structurally_invalid_count == 0
            and fraction <= MAX_UNREPRESENTABLE_FRACTION),
    }


@torch.inference_mode()
def representation_preflight(
        dataset: Dataset, model, fused_heads, normalizer,
        device: torch.device, batch_size: int) -> dict:
    """Read point histories and exact anchors only; never consume truth."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    totals = {
        "total_scalar_endpoints": 0,
        "structurally_invalid": 0,
        "width_beyond_limit": 0,
        "center_beyond_limit": 0,
        "unrepresentable_union": 0,
    }
    max_abs_log_width_ratio = None
    max_center_fraction = None
    expected_total = dataset.anchors.numel() // 2
    upper_tail_count = max(1, math.ceil(expected_total * 0.001))
    upper_log_values = torch.empty(0, device=device)
    upper_center_values = torch.empty(0, device=device)
    rows = 0
    model.eval()
    for history, endpoints in loader:
        history = history.to(device, non_blocking=True)
        endpoints = endpoints.to(device, non_blocking=True).float()
        normalized = (
            history - normalizer.mean_input) / normalizer.std_input
        x_half = normalized.to(dtype=torch.float16)
        features = model._decode(x_half)
        prediction_normalized = fused_mean_prediction(
            model, x_half, features, fused_heads)
        point = common.to_physical_target(
            prediction_normalized.float(), normalizer)[..., :2]
        row = representability_from_physical(point, endpoints)
        lower = endpoints[..., 0]
        upper = endpoints[..., 1]
        width = upper - lower
        valid = (
            torch.isfinite(point) & torch.isfinite(lower)
            & torch.isfinite(upper) & (width > 0.0))
        safe_width = torch.where(valid, width, torch.ones_like(width))
        reference = point.new_tensor(
            DistilledIntervalHeadConfig().reference_total_widths).view(
                1, 1, 1, 1, 2)
        abs_log = torch.where(
            valid, torch.abs(torch.log(safe_width / reference)),
            torch.full_like(width, math.inf)).reshape(-1)
        center_ratio = torch.where(
            valid, torch.abs(0.5 * (lower + upper) - point) / safe_width,
            torch.full_like(width, math.inf)).reshape(-1)
        upper_log_values = torch.topk(
            torch.cat((upper_log_values, abs_log)),
            min(upper_tail_count, upper_log_values.numel() + abs_log.numel()),
            sorted=False).values
        upper_center_values = torch.topk(
            torch.cat((upper_center_values, center_ratio)),
            min(
                upper_tail_count,
                upper_center_values.numel() + center_ratio.numel()),
            sorted=False).values
        for key in totals:
            totals[key] += row[key]
        if row["max_abs_log_width_ratio"] is not None:
            max_abs_log_width_ratio = max(
                max_abs_log_width_ratio or 0.0,
                row["max_abs_log_width_ratio"])
        if row["max_center_fraction"] is not None:
            max_center_fraction = max(
                max_center_fraction or 0.0, row["max_center_fraction"])
        rows += len(history)
    if rows != len(dataset):
        raise RuntimeError("representation preflight ended early")
    total = totals["total_scalar_endpoints"]
    if total != expected_total:
        raise RuntimeError(
            f"representation scalar count {total} differs from {expected_total}")
    fraction = totals["unrepresentable_union"] / total
    p99_9_abs_log_width_ratio = float(upper_log_values.min().item())
    p99_9_center_fraction = float(upper_center_values.min().item())
    if not math.isfinite(p99_9_abs_log_width_ratio):
        p99_9_abs_log_width_ratio = None
    if not math.isfinite(p99_9_center_fraction):
        p99_9_center_fraction = None
    result = {
        **totals,
        "rows": rows,
        "unrepresentable_fraction": fraction,
        "representable_fraction": 1.0 - fraction,
        "max_abs_log_width_ratio": max_abs_log_width_ratio,
        "max_center_fraction": max_center_fraction,
        "p99_9_abs_log_width_ratio_conservative": (
            p99_9_abs_log_width_ratio),
        "p99_9_center_fraction_conservative": p99_9_center_fraction,
        "min_width_log_margin": (
            None if max_abs_log_width_ratio is None else
            DistilledIntervalHeadConfig().max_log_width_ratio
            - max_abs_log_width_ratio),
        "min_center_fraction_margin": (
            None if max_center_fraction is None else
            DistilledIntervalHeadConfig().max_center_fraction
            - max_center_fraction),
        "worst_case_sps_loss_bound": 100.0 * fraction,
        "max_unrepresentable_fraction": MAX_UNREPRESENTABLE_FRACTION,
        "max_worst_case_sps_loss_bound": (
            MAX_REPRESENTATION_SPS_LOSS_BOUND),
        "targets_read": False,
    }
    result["passed"] = (
        result["structurally_invalid"] == 0
        and result["unrepresentable_union"] == 0
        and result["min_width_log_margin"] is not None
        and result["min_center_fraction_margin"] is not None
        and result["min_width_log_margin"] > 0.0
        and result["min_center_fraction_margin"] > 0.0)
    return result


@torch.inference_mode()
def evaluate_terminal(
        model, fused_heads, head: DistilledIntervalHead,
        loader: DataLoader, normalizer, device: torch.device) -> dict:
    """Score deployed absolute bounds with SPS's signed-extent accumulator."""
    model.eval()
    head.eval()
    if {value.dtype for value in head.state_dict().values()} != {torch.float16}:
        raise ValueError("terminal evaluation requires deployed FP16 head")
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
        prediction_normalized = fused_mean_prediction(
            model, x_half, features, fused_heads)
        lower_extent, upper_extent = head(
            x_half, prediction_normalized, features, model.target_std)
        point, lower, upper = absolute_endpoints_from_ftq(
            prediction_normalized, lower_extent, upper_extent, normalizer)
        if (not torch.isfinite(teacher_lower).all()
                or not torch.isfinite(teacher_upper).all()
                or torch.any(teacher_lower > teacher_upper)):
            raise FloatingPointError("terminal teacher endpoints are invalid")
        # The accumulator expects signed physical extents around `point`, not
        # absolute lower/upper tensors.  This conversion also handles bounds
        # lying on the same side of a non-zero point without clipping.
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
        "finite_ordered_absolute_bounds": True,
        "point_output_sha256": point_digest.hexdigest(),
    }


def student_gate(
        args: argparse.Namespace, representation: dict,
        evaluation: dict | None, point_state_equal: bool,
        point_output_equal: bool, completed_updates: int,
        training_seconds: float, scientific_wall_seconds: float,
        deployed_head_dtype: str) -> dict:
    formal = {
        "updates": args.updates == 4000,
        "batch_size": args.batch_size == 8,
        "eval_batch_size": args.eval_batch_size == 16,
        "reconstruction_batch_size": args.reconstruction_batch_size == 16,
        "lr": args.lr == 1e-4,
        "weight_decay": args.weight_decay == 1e-5,
        "grad_clip": args.grad_clip == 1.0,
        "time_budget_seconds": (
            args.time_budget_seconds == TRAINING_TIME_BUDGET_SECONDS),
        "seed": args.seed == 0,
        "num_workers": args.num_workers == 0,
        "device": args.device == "cuda:0",
        "held_fold": args.held_fold in range(5),
    }
    checks = {
        "formal_configuration": all(formal.values()),
        "representation_preflight_passed": representation["passed"],
        "all_training_anchor_elements_representable": (
            representation["unrepresentable_union"] == 0),
        "strict_representation_margins_positive": (
            representation["min_width_log_margin"] is not None
            and representation["min_center_fraction_margin"] is not None
            and representation["min_width_log_margin"] > 0.0
            and representation["min_center_fraction_margin"] > 0.0),
        "representation_loss_bound_eq_0": (
            representation["worst_case_sps_loss_bound"]
            <= MAX_REPRESENTATION_SPS_LOSS_BOUND),
        "completed_fixed_4000_updates": completed_updates == 4000,
        "training_within_1200_seconds": (
            training_seconds <= args.time_budget_seconds),
        "scientific_wall_through_terminal_within_1200_seconds": (
            scientific_wall_seconds <= args.time_budget_seconds),
        "student_gate_uses_deployed_fp16": deployed_head_dtype == "float16",
        "point_state_bitwise_equal": point_state_equal,
        "full_2576_725_direct_point_output_bitwise_equal": point_output_equal,
        "terminal_evaluation_completed": evaluation is not None,
    }
    thresholds = {
        "max_unrepresentable_fraction": MAX_UNREPRESENTABLE_FRACTION,
        "max_representation_sps_loss_bound": (
            MAX_REPRESENTATION_SPS_LOSS_BOUND),
        "max_exact_ceiling_regret": MAX_EXACT_CEILING_REGRET,
        "max_coverage_drop_vs_exact_ceiling": MAX_COVERAGE_DROP,
        "max_width_ratio_vs_exact_ceiling": MAX_WIDTH_RATIO,
        "fold04_min_sps": FOLD04_MIN_SPS,
        "fold04_min_coverage": FOLD04_MIN_COVERAGE,
        "fold04_max_width": FOLD04_MAX_WIDTH,
    }
    observed = {
        "training_seconds": training_seconds,
        "scientific_wall_seconds": scientific_wall_seconds,
        "representation": representation,
    }
    if evaluation is not None:
        student = evaluation["student"]
        teacher = evaluation["exact_p0_ceiling_with_direct_point"]
        regret = teacher["sps_score"] - student["sps_score"]
        coverage_floor = teacher["coverage"] - MAX_COVERAGE_DROP
        width_ceiling = teacher["mean_width"] * MAX_WIDTH_RATIO
        checks.update({
            "finite_ordered_absolute_bounds": (
                evaluation["finite_ordered_absolute_bounds"]),
            "held_truth_read_once": evaluation["truth_passes"] == 1,
            "held_row_count_exact": (
                evaluation["held_rows"] == EXPECTED_HELD_TAIL[args.held_fold]),
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
            "finite_ordered_absolute_bounds": False,
            "held_truth_read_once": False,
            "held_row_count_exact": False,
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
        raise RuntimeError("formal direct dense exact pilot requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    point_file = s1.file_identity(args.point_checkpoint)
    if (point_file["sha256"] != FORMAL_POINT_SHA256
            or point_file["sha256"] != args.expected_point_sha256):
        raise ValueError("point checkpoint is not frozen SHA758b")
    exact.seed_everything(args.seed)
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

    anchor, reused_anchor_metadata = exact.reconstruct_exact_endpoint_rows(
        args, point_histories, point_folds, point_train, fold_manifest,
        normalizer, device, "point_nonheld_training_anchors")
    anchor_metadata = {
        **reused_anchor_metadata,
        "format": ANCHOR_FORMAT,
        "reconstruction_engine": exact.ANCHOR_FORMAT,
    }
    anchor_dataset = AnchorTrainingDataset(
        point_histories, point_train, anchor)

    model, model_metadata = load_forecaster(
        args.point_checkpoint, normalizer, device)
    model.eval().requires_grad_(False)
    fused_heads = fused_mean_point_heads(model).to(device).eval()
    fused_heads.requires_grad_(False)
    point_state_before = model_state_identity(model)
    point_output_before = direct_point_stream_identity(
        point_audit, model, fused_heads, normalizer,
        device, args.eval_batch_size)
    head = build_head(device, args.seed)
    representation = representation_preflight(
        anchor_dataset, model, fused_heads, normalizer,
        device, args.eval_batch_size)
    print(json.dumps({
        "event": "direct_dense_exact_representation_preflight",
        **representation,
    }), flush=True)

    history_rows = []
    completed_updates = 0
    training_seconds = 0.0
    if representation["passed"]:
        anchor_loader = DataLoader(
            anchor_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed))
        anchor_iterator = common.repro.cycling(anchor_loader)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.updates)
        sums: dict[str, float] = {}
        count = 0
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
                prediction_normalized = fused_mean_prediction(
                    model, x_half, features, fused_heads)
            head.train()
            optimizer.zero_grad(set_to_none=True)
            lower_extent, upper_extent = head(
                x_half.float(), prediction_normalized.float(),
                [feature.float() for feature in features],
                model.target_std.float())
            loss, components = exact_endpoint_objective(
                prediction_normalized.float(), lower_extent, upper_extent,
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
                    "elapsed_seconds": (
                        time.perf_counter() - training_started),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **{
                        f"train_{key}_mean": value / count
                        for key, value in sums.items()
                    },
                }
                history_rows.append(row)
                print(json.dumps({
                    "event": "direct_dense_exact_interval_train", **row,
                }), flush=True)
                sums.clear()
                count = 0
        training_seconds = time.perf_counter() - training_started
        del anchor_iterator, anchor_loader, optimizer, scheduler

    deploy_head = prepare_deploy_head(head).cpu()
    point_state_after = model_state_identity(model)
    point_output_after = direct_point_stream_identity(
        point_audit, model, fused_heads, normalizer,
        device, args.eval_batch_size)
    point_state_equal = point_state_before == point_state_after
    point_output_equal = point_output_before == point_output_after
    del anchor_dataset, anchor, head, fused_heads, model
    gc.collect()
    torch.cuda.empty_cache()

    evaluation = None
    terminal_endpoint_metadata = None
    reloaded_point_state_equal = None
    if representation["passed"] and completed_updates == args.updates:
        held_endpoints, terminal_endpoint_metadata = (
            exact.reconstruct_exact_endpoint_rows(
                args, tail_histories, tail_folds, tail_held, fold_manifest,
                normalizer, device, "held_tail_terminal_exact_ceiling"))
        terminal_endpoint_metadata = {
            **terminal_endpoint_metadata,
            "format": ANCHOR_FORMAT,
            "reconstruction_engine": exact.ANCHOR_FORMAT,
        }
        tail_truth_windows = common.repro.WindowDataset(
            trajectories, tail_windows)
        terminal_dataset = HeldEndpointEvaluationDataset(
            tail_truth_windows, tail_held, held_endpoints)
        terminal_loader = DataLoader(
            terminal_dataset, batch_size=args.eval_batch_size,
            shuffle=False, num_workers=0, pin_memory=True)
        deploy_model, _deploy_metadata = load_forecaster(
            args.point_checkpoint, normalizer, device)
        deploy_model.eval().requires_grad_(False)
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

    scientific_wall_seconds = time.perf_counter() - process_started
    gate = student_gate(
        args, representation, evaluation, point_state_equal,
        point_output_equal, completed_updates, training_seconds,
        scientific_wall_seconds, "float16")
    status = (
        f"PASS_DIRECT_DENSE_EXACT_INTERVAL_FOLD{args.held_fold:02d}"
        if gate["passed"]
        else f"STOP_DIRECT_DENSE_EXACT_INTERVAL_FOLD{args.held_fold:02d}")
    checkpoint_path = (
        args.output_dir / "direct_dense_exact_interval_head.pth")
    deploy_state = {
        key: value.detach().cpu()
        for key, value in deploy_head.state_dict().items()
    }
    deploy_config = deploy_head.architecture_config()
    torch.save({
        "phase": "direct_dense_exact_interval_distillation",
        "iteration": completed_updates,
        "architecture": ARCHITECTURE,
        "model_config": deploy_config,
        "model_state_dict": deploy_state,
        "head_config": deploy_config,
        "head_state_dict": deploy_state,
        "deployed_head_dtype": "float16",
        "point_checkpoint": point_file,
        "held_fold": args.held_fold,
        "updates": completed_updates,
        "representation_preflight": representation,
        "evaluation": evaluation,
        "gate": gate,
    }, checkpoint_path)
    summary = {
        "status": status,
        "evidence_class": (
            f"fold{args.held_fold:02d}-held exact-P0-anchor-only direct "
            "interval pilot; nonheld point histories only; one terminal "
            "held-tail truth pass; local diagnostic, not official evidence"),
        "mechanism": {
            "training_target": (
                "pure physical SmoothL1 on exact A/B post-selection and "
                "refinement absolute endpoints after P0 B=.625/C=.25 "
                "composition"),
            "tail_training_truth_read": False,
            "interval_graph": ARCHITECTURE,
            "interval_parameter_count": EXPECTED_HEAD_PARAMETERS,
            "continuous_ordered_signed_extents": True,
            "quantile_bins_or_argmax": False,
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
        "representation_preflight": representation,
        "terminal_exact_ceiling": terminal_endpoint_metadata,
        "held_fold_evaluations_during_training": 0,
        "held_tail_truth_passes": (
            0 if evaluation is None else evaluation["truth_passes"]),
        "completed_updates": completed_updates,
        "training_seconds": training_seconds,
        "scientific_wall_seconds": scientific_wall_seconds,
        "wall_seconds_at_summary_write": time.perf_counter() - process_started,
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
            "only_after_fold04_pass": True,
            "five_distinct_held_folds_required": True,
            "pooled_fixed725_sps_min": POOLED_PROMOTION_SPS,
            "final_refit_uses_all_2576_exact_anchors_and_no_tail_truth": True,
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
        "event": "direct_dense_exact_interval_complete",
        "status": status,
        "evaluation": evaluation,
        "gate": gate,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
