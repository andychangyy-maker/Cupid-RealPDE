#!/usr/bin/env python3
"""RSM1b: calibrated train-only role specialization of the C208 backbone."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

import train_c208_real_sim_rehearsal as rehearsal
import train_c208_real_teacher_distillation as rtd1
import train_p1_direct_native_quantile_teacher as quantile_teacher
from train_uq_budget_reallocator import trajectory_folds


CACHE_SCHEMA_VERSION = 1
CACHE_METHOD = "c208_rsm1_real2576_foldaware_roles_physical_v1"
EXPECTED_WINDOWS = 2576
EVIDENCE_CLASS = (
    "fixed725_checkpoint_selection_contaminated_teacher_diagnostic")
FLUCTUATION_WEIGHTS = {"a": 0.5, "b": 0.5}
MEAN_WEIGHTS = {"a": 0.25, "b": 0.25, "c": 0.5}
TEACHER_TENSOR_SPECS = {
    "a_fluctuation_uv_physical": torch.float16,
    "b_fluctuation_uv_physical": torch.float16,
    "p1_temporal_mean_uv_physical": torch.float16,
}
PHYSICAL_UNITS = {
    name: "physical_velocity" for name in TEACHER_TENSOR_SPECS
}
FLUCTUATION_SCALE = 0.020
MEAN_SCALE = 0.025
ROLE_DISTILL_WEIGHT = 0.00125
MEAN_DISTILL_WEIGHT = 0.00125
DISTILL_RAMP_UPDATES = 50
PREFLIGHT_UPDATE = 200
PREFLIGHT_AUDIT_UPDATES = (50, 100, 150, 200)
PREFLIGHT_LOSS_RATIO_RANGE = (0.002, 0.10)
PREFLIGHT_GRAD_RATIO_RANGE = (0.01, 0.30)
PREFLIGHT_MEDIAN_COSINE_MIN = -0.10
PREFLIGHT_WORST_COSINE_MIN = -0.50
ANCHOR_SIZE = 128
ANCHOR_DECLINE_FRACTION = 0.10
TERMINAL_UPDATE = rehearsal.FORMAL_TERMINAL_UPDATE
GATE_POINT_DELTA_MIN = 0.10
GATE_COMPONENT_DELTA_FLOOR = -0.02
GATE_SPS_DELTA_MIN = 0.70
GATE_NON_TIME_SHADOW_MIN = 0.30
NATIVE_STEP4000_METRICS = {
    "rel_l2_score": 95.86352389426571,
    "tke_rel_l2_score": 80.95221179698406,
    "mvpe_rel_l2_score": 96.51432542559023,
    "point_score": 91.52029057134612,
    "native_sps_score": 53.07979215521265,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache-sha256", required=True)
    parser.add_argument("--identity-summary", type=Path, required=True)
    parser.add_argument("--identity-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--updates", type=int, default=rehearsal.FORMAL_REQUESTED_UPDATES)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.updates != rehearsal.FORMAL_REQUESTED_UPDATES:
        raise ValueError("formal RSM1 requires --updates 5000")
    args.start_sim_sha256 = rehearsal.normalized_sha256(
        args.start_sim_sha256, "start-sim-sha256")
    if args.start_sim_sha256 != rehearsal.FORMAL_START_SIM_SHA256:
        raise ValueError("start-sim-sha256 is not the frozen Sim10k start")
    args.teacher_cache_sha256 = rehearsal.normalized_sha256(
        args.teacher_cache_sha256, "teacher-cache-sha256")
    args.identity_summary_sha256 = rehearsal.normalized_sha256(
        args.identity_summary_sha256, "identity-summary-sha256")
    if args.identity_summary_sha256 != rehearsal.FORMAL_IDENTITY_SUMMARY_SHA256:
        raise ValueError("identity summary is not the frozen standard C208 control")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse RSM1 output: {args.output_dir}")


def expected_teacher_shapes(dataset) -> dict[str, tuple[int, ...]]:
    if not hasattr(dataset, "windows") or len(dataset) != EXPECTED_WINDOWS:
        raise ValueError("RSM1 cache requires the ordered fixed2576 dataset")
    _history, target = dataset[0]
    if target.ndim != 4 or target.shape[-1] < 2:
        raise ValueError("Real target must have shape (T,H,W,C>=2)")
    fluctuation = (len(dataset), *target.shape[:-1], 2)
    mean = (len(dataset), 1, *target.shape[1:-1], 2)
    return {
        "a_fluctuation_uv_physical": fluctuation,
        "b_fluctuation_uv_physical": fluctuation,
        "p1_temporal_mean_uv_physical": mean,
    }


def validate_teacher_cache_payload(
        payload: dict, manifest: list[dict], dataset,
        expected_folds: list[int], expected_fold_manifest: dict,
        real_stats_sha256: str,
        ) -> tuple[dict[str, torch.Tensor], dict]:
    if not isinstance(payload, dict):
        raise TypeError("teacher cache payload must be a mapping")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("teacher cache schema version mismatch")
    if payload.get("method") != CACHE_METHOD:
        raise ValueError("teacher cache method/unit contract mismatch")
    forbidden = (
        "prediction_uv_normalized", "point_uv_normalized",
        "lower_uv_physical", "upper_uv_physical", "target", "truth")
    if any(key in payload for key in forbidden):
        raise ValueError("normalized, endpoint, and target-bearing caches are forbidden")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("teacher cache has no metadata")
    if metadata.get("split_manifest_sha256") != rehearsal.canonical_sha256(
            manifest):
        raise ValueError("teacher cache split manifest SHA mismatch")
    if metadata.get("real_stats_sha256") != real_stats_sha256:
        raise ValueError("teacher cache Real stats/unit transform mismatch")
    keys = rtd1.ordered_train_keys(dataset.windows, manifest)
    if metadata.get("train_keys") != keys:
        raise ValueError("teacher cache train keys/natural index order mismatch")
    if metadata.get("fluctuation_weights") != FLUCTUATION_WEIGHTS:
        raise ValueError("teacher cache fluctuation formula mismatch")
    if metadata.get("mean_weights") != MEAN_WEIGHTS:
        raise ValueError("teacher cache P1 mean formula mismatch")
    if metadata.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError("teacher cache evidence provenance mismatch")
    if metadata.get("units") != PHYSICAL_UNITS:
        raise ValueError("teacher cache units are not physical velocity")
    if metadata.get("composition_precision") != (
            "float32_before_final_float16_storage"):
        raise ValueError("teacher cache was quantized before role composition")
    if metadata.get("fold_manifest") != expected_fold_manifest:
        raise ValueError("teacher cache fold manifest mismatch")
    reconstruction = metadata.get("reconstruction")
    if not isinstance(reconstruction, dict):
        raise ValueError("teacher cache has no reconstruction provenance")
    rows = reconstruction.get("folds")
    if not isinstance(rows, list):
        raise ValueError("teacher cache reconstruction has no fold rows")
    expected_counts = {
        fold: expected_folds.count(fold) for fold in sorted(set(expected_folds))}
    actual_counts = {
        int(row.get("fold", -1)): int(row.get("samples", -1))
        for row in rows if isinstance(row, dict)}
    if actual_counts != expected_counts:
        raise ValueError("teacher cache reconstruction fold counts mismatch")
    rtd1._validate_formal_source_contracts(metadata)

    shapes = expected_teacher_shapes(dataset)
    tensors = {}
    for name, dtype in TEACHER_TENSOR_SPECS.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"teacher cache {name} must be a tensor")
        if value.device.type != "cpu" or value.dtype != dtype:
            raise TypeError(f"teacher cache {name} has wrong physical dtype")
        if tuple(value.shape) != shapes[name]:
            raise ValueError(f"teacher cache {name} shape mismatch")
        if not all(bool(torch.isfinite(value[start:start + 16]).all())
                   for start in range(0, len(value), 16)):
            raise ValueError(f"teacher cache {name} contains non-finite values")
        if rtd1.tensor_sha256(value) != rtd1._recorded_tensor_sha(
                metadata, name):
            raise ValueError(f"teacher cache {name} tensor SHA mismatch")
        tensors[name] = value
    for name in (
            "a_fluctuation_uv_physical", "b_fluctuation_uv_physical"):
        maximum_mean = max(
            float(tensors[name][start:start + 16].float().mean(dim=1)
                  .abs().max())
            for start in range(0, len(tensors[name]), 16))
        if maximum_mean > 2e-3:
            raise ValueError(f"teacher cache {name} is not zero-time-mean")
    folds = payload.get("fold")
    if (not isinstance(folds, torch.Tensor) or folds.dtype != torch.int8
            or tuple(folds.shape) != (EXPECTED_WINDOWS,)
            or not torch.equal(
                folds, torch.as_tensor(expected_folds, dtype=torch.int8))):
        raise ValueError("teacher cache fold labels do not align")
    return tensors, metadata


def load_teacher_cache(
        path: Path, expected_sha256: str, manifest: list[dict], dataset,
        expected_folds: list[int], expected_fold_manifest: dict,
        real_stats_sha256: str,
        ) -> tuple[dict[str, torch.Tensor], dict]:
    actual = rehearsal.sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"teacher cache SHA-256 {actual} != expected {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensors, metadata = validate_teacher_cache_payload(
        payload, manifest, dataset, expected_folds, expected_fold_manifest,
        real_stats_sha256)
    return tensors, {**metadata, "path": str(path.resolve()), "sha256": actual}


def prepare_real_training(args: argparse.Namespace, device: torch.device) -> dict:
    real = rehearsal.prepare_real_training(args, device)
    manifest = json.loads(
        (args.output_dir / "real_split_manifest.json").read_text(
            encoding="utf-8"))
    dataset = real["loader"].dataset.base_dataset
    _tail_folds, fold_manifest = trajectory_folds(manifest, 5)
    point_folds = quantile_teacher.partition_fold_labels(
        manifest, fold_manifest, "train_starts")
    tensors, metadata = load_teacher_cache(
        args.teacher_cache, args.teacher_cache_sha256, manifest, dataset,
        point_folds, fold_manifest, real["stats_sha256"])
    real.update({
        "teacher": tensors,
        "teacher_metadata": metadata,
        "fold_manifest": fold_manifest,
    })
    return real


def load_identity_step4000(args: argparse.Namespace) -> dict:
    """Validate the frozen control and its unique exact update-4000 row."""
    proxy = argparse.Namespace(
        rehearsal_weight=rehearsal.FORMAL_REHEARSAL_WEIGHT,
        identity_summary=args.identity_summary,
        identity_summary_sha256=args.identity_summary_sha256,
    )
    identity = rehearsal.load_identity_step2000(proxy)
    if identity is None:
        raise RuntimeError("frozen identity summary unexpectedly resolved null")
    payload = json.loads(args.identity_summary.read_text(encoding="utf-8"))
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("identity summary has no history")
    rows = [
        row for row in history if isinstance(row, dict)
        and int(row.get("iteration", -1)) == TERMINAL_UPDATE]
    if len(rows) != 1:
        raise ValueError("identity summary must have exactly one update-4000 row")
    metrics = {}
    for key, expected in NATIVE_STEP4000_METRICS.items():
        value = rows[0].get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"identity update-4000 row is missing finite {key}")
        metrics[key] = float(value)
        if not math.isclose(
                metrics[key], expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"identity update-4000 {key}={metrics[key]}, expected {expected}")
    return {**identity, "metrics_update4000": metrics}


def ramp_weight(weight: float, iteration: int) -> float:
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    return weight * min(1.0, iteration / DISTILL_RAMP_UPDATES)


def role_parameters(model) -> tuple[list[torch.nn.Parameter],
                                    list[torch.nn.Parameter]]:
    if len(model.member_point_heads) != 4:
        raise RuntimeError("RSM1 requires exactly four member point heads")
    if model.mean_profile_refiner is None:
        raise RuntimeError("RSM1 requires the C208 mean_profile_refiner")
    member_parameters = [
        parameter for module in model.member_point_heads
        for parameter in module.parameters() if parameter.requires_grad]
    mean_parameters = [
        parameter for parameter in model.mean_profile_refiner.parameters()
        if parameter.requires_grad]
    if not member_parameters or not mean_parameters:
        raise RuntimeError("RSM1 role parameter sets must be non-empty")
    if {id(parameter) for parameter in member_parameters}.intersection(
            id(parameter) for parameter in mean_parameters):
        raise RuntimeError("RSM1 role parameter sets overlap")
    return member_parameters, mean_parameters


def role_distillation_objectives(
        prediction: torch.Tensor, members: torch.Tensor,
        teacher_a: torch.Tensor, teacher_b: torch.Tensor,
        teacher_mean: torch.Tensor, normalizer,
        ) -> dict[str, torch.Tensor]:
    if members.ndim != 6 or members.shape[1] != 4:
        raise ValueError("members must have shape [B,4,T,H,W,C]")
    std = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    mean = normalizer.mean_target[:2].view(1, 1, 1, 1, 2)
    teacher_a_norm = teacher_a.float() / std
    teacher_b_norm = teacher_b.float() / std
    teacher_mean_norm = (teacher_mean.float() - mean) / std
    student_members = members[..., :2]
    student_fluctuation = (
        student_members - student_members.mean(dim=2, keepdim=True))
    teacher_a_members = teacher_a_norm[:, None].expand_as(
        student_fluctuation[:, :2])
    teacher_b_members = teacher_b_norm[:, None].expand_as(
        student_fluctuation[:, 2:])
    a_loss = F.smooth_l1_loss(
        student_fluctuation[:, :2] / FLUCTUATION_SCALE,
        teacher_a_members / FLUCTUATION_SCALE)
    b_loss = F.smooth_l1_loss(
        student_fluctuation[:, 2:] / FLUCTUATION_SCALE,
        teacher_b_members / FLUCTUATION_SCALE)
    mean_loss = F.smooth_l1_loss(
        prediction[..., :2].mean(dim=1, keepdim=True) / MEAN_SCALE,
        teacher_mean_norm / MEAN_SCALE)
    aggregate_teacher_fluctuation = 0.5 * (
        teacher_a_norm + teacher_b_norm)
    aggregate_student_fluctuation = (
        prediction[..., :2]
        - prediction[..., :2].mean(dim=1, keepdim=True))
    aggregate_fluctuation_loss = F.smooth_l1_loss(
        aggregate_student_fluctuation / FLUCTUATION_SCALE,
        aggregate_teacher_fluctuation / FLUCTUATION_SCALE)
    return {
        "a_fluctuation": a_loss,
        "b_fluctuation": b_loss,
        "role": 0.5 * (a_loss + b_loss),
        "p1_mean": mean_loss,
        "p1_aggregate": 0.5 * (mean_loss + aggregate_fluctuation_loss),
    }


def _gradient_geometry(left_gradients, right_gradients) -> dict[str, float | bool]:
    return rtd1._gradient_geometry(left_gradients, right_gradients)


def parameter_gradient_audit(
        supervised_loss: torch.Tensor, weighted_role: torch.Tensor,
        weighted_mean: torch.Tensor, member_parameters,
        mean_parameters, iteration: int,
        ) -> dict:
    role_base = torch.autograd.grad(
        supervised_loss, member_parameters, retain_graph=True,
        allow_unused=True)
    role_aux = torch.autograd.grad(
        weighted_role, member_parameters, retain_graph=True,
        allow_unused=True)
    mean_base = torch.autograd.grad(
        supervised_loss, mean_parameters, retain_graph=True,
        allow_unused=True)
    mean_aux = torch.autograd.grad(
        weighted_mean, mean_parameters, retain_graph=True,
        allow_unused=True)
    denominator = float(supervised_loss.detach().abs())
    audit = {
        "iteration": iteration,
        "role_heads": _gradient_geometry(role_base, role_aux),
        "mean_profile_refiner": _gradient_geometry(mean_base, mean_aux),
    }
    for name, weighted in (
            ("role_heads", weighted_role),
            ("mean_profile_refiner", weighted_mean)):
        loss_ratio = (
            float(weighted.detach().abs()) / denominator
            if denominator > 0.0 else math.inf)
        audit[name]["loss_ratio"] = loss_ratio
        audit[name]["finite_nonzero"] = bool(
            audit[name]["finite_nonzero"] and math.isfinite(loss_ratio))
    return audit


def anchor_indices(length: int, count: int = ANCHOR_SIZE) -> list[int]:
    if length < count or count <= 1:
        raise ValueError("anchor requires at least 128 train samples")
    return list(range(count))


def make_anchor_loader(dataset, num_workers: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, anchor_indices(len(dataset))),
        batch_size=rehearsal.FORMAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(rehearsal.FORMAL_SEED))


@torch.inference_mode()
def evaluate_train_anchor(
        model, loader, normalizer, teacher: dict[str, torch.Tensor],
        device: torch.device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums = {name: 0.0 for name in (
        "a_fluctuation", "b_fluctuation", "p1_mean", "p1_aggregate")}
    samples = 0
    for history, _target, _trajectory_ids, sample_indices in loader:
        history = history.to(device, non_blocking=True)
        history_norm = (
            history - normalizer.mean_input) / normalizer.std_input
        prediction, members, *_rest = model.forward_uq_training(history_norm)
        batch_size = len(history)
        objectives = role_distillation_objectives(
            prediction, members,
            teacher["a_fluctuation_uv_physical"][sample_indices].to(
                device=device, dtype=torch.float32, non_blocking=True),
            teacher["b_fluctuation_uv_physical"][sample_indices].to(
                device=device, dtype=torch.float32, non_blocking=True),
            teacher["p1_temporal_mean_uv_physical"][sample_indices].to(
                device=device, dtype=torch.float32, non_blocking=True),
            normalizer)
        for name in sums:
            sums[name] += float(objectives[name]) * batch_size
        samples += batch_size
    model.train(was_training)
    if samples != ANCHOR_SIZE:
        raise RuntimeError(f"anchor evaluated {samples}, expected {ANCHOR_SIZE}")
    return {name: value / samples for name, value in sums.items()}


def evaluate_preflight(
        audits: list[dict], anchor_start: dict[str, float],
        anchor_update200: dict[str, float]) -> dict:
    if [row.get("iteration") for row in audits] != list(
            PREFLIGHT_AUDIT_UPDATES):
        return {
            "status": "STOP_RSM1B_PREFLIGHT", "pass": False,
            "reason": "missing_or_duplicate_audit_updates", "audits": audits,
        }
    checks = {}
    for branch in ("role_heads", "mean_profile_refiner"):
        rows = [audit[branch] for audit in audits]
        cosines = [row["cosine"] for row in rows]
        checks[branch] = {
            "finite_nonzero": all(row["finite_nonzero"] for row in rows),
            "loss_ratio_range": all(
                PREFLIGHT_LOSS_RATIO_RANGE[0] <= row["loss_ratio"]
                <= PREFLIGHT_LOSS_RATIO_RANGE[1] for row in rows),
            "gradient_ratio_range": all(
                PREFLIGHT_GRAD_RATIO_RANGE[0] <= row["gradient_ratio"]
                <= PREFLIGHT_GRAD_RATIO_RANGE[1] for row in rows),
            "median_cosine": median(cosines),
            "worst_cosine": min(cosines),
            "cosine_gate": (
                median(cosines) >= PREFLIGHT_MEDIAN_COSINE_MIN
                and min(cosines) >= PREFLIGHT_WORST_COSINE_MIN),
        }
    anchor_checks = {}
    for name in ("a_fluctuation", "b_fluctuation", "p1_mean", "p1_aggregate"):
        start = float(anchor_start[name])
        end = float(anchor_update200[name])
        finite = math.isfinite(start) and math.isfinite(end)
        if finite and start > 1e-12:
            decline = 1.0 - end / start
            anchor_pass = end <= (1.0 - ANCHOR_DECLINE_FRACTION) * start
        else:
            decline = None
            anchor_pass = bool(finite and end <= start + 1e-12)
        anchor_checks[name] = {
            "start": start, "update200": end, "decline_fraction": decline,
            "pass": anchor_pass,
        }
    branches_pass = all(
        row["finite_nonzero"] and row["loss_ratio_range"]
        and row["gradient_ratio_range"] and row["cosine_gate"]
        for row in checks.values())
    passed = bool(
        branches_pass and all(row["pass"] for row in anchor_checks.values()))
    return {
        "status": "GO_RSM1B_TO_UPDATE4000" if passed
        else "STOP_RSM1B_PREFLIGHT",
        "pass": passed, "checks": checks, "anchor": anchor_checks,
        "audits": audits,
    }


def _add_gradients(parameters, gradients) -> None:
    rtd1._add_gradients(parameters, gradients)


def backward_real_role_specialization(
        model, batch, normalizer, teacher: dict[str, torch.Tensor],
        device: torch.device, iteration: int,
        member_parameters=None, mean_parameters=None,
        ) -> tuple[dict[str, float], dict | None]:
    if member_parameters is None or mean_parameters is None:
        member_parameters, mean_parameters = role_parameters(model)
    history, target, trajectory_ids, sample_indices = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    history_norm, target_norm = normalizer.preprocess(history, target)
    (prediction, members, quantiles, lower, upper, _features,
     _coverage_logits) = model.forward_uq_training(history_norm)
    point_loss, point_components = rehearsal.ensemble_train.point_loss(
        rehearsal.real_loss_args(), prediction, target_norm, normalizer,
        rehearsal.REAL_ACCURACY_WEIGHTS, iteration)
    member_loss, bootstrap_fraction = (
        rehearsal.ensemble_train.bootstrap_member_point_loss(
            rehearsal.real_loss_args(), members, target_norm, trajectory_ids,
            normalizer, rehearsal.REAL_ACCURACY_WEIGHTS, iteration))
    quantile_loss = rehearsal.ensemble_train.residual_quantile_loss(
        model, prediction, quantiles, target_norm, normalizer)
    soft_sps_loss, soft_sps_score = (
        rehearsal.ensemble_train.soft_native_sps_objective(
            prediction, target_norm, lower, upper, normalizer,
            tau_fraction=rehearsal.REAL_SOFT_SPS_TAU_FRACTION))
    soft_sps_weight = rehearsal.REAL_SOFT_SPS_WEIGHT * min(
        1.0, iteration / rehearsal.REAL_SOFT_SPS_WARMUP_UPDATES)
    supervised_loss = 0.25 * point_loss + member_loss + 0.05 * quantile_loss
    total_standard_loss = supervised_loss + soft_sps_weight * soft_sps_loss

    teacher_a = teacher["a_fluctuation_uv_physical"][sample_indices].to(
        device=device, dtype=torch.float32, non_blocking=True)
    teacher_b = teacher["b_fluctuation_uv_physical"][sample_indices].to(
        device=device, dtype=torch.float32, non_blocking=True)
    teacher_mean = teacher["p1_temporal_mean_uv_physical"][sample_indices].to(
        device=device, dtype=torch.float32, non_blocking=True)
    objectives = role_distillation_objectives(
        prediction, members, teacher_a, teacher_b, teacher_mean, normalizer)
    role_weight = ramp_weight(ROLE_DISTILL_WEIGHT, iteration)
    mean_weight = ramp_weight(MEAN_DISTILL_WEIGHT, iteration)
    weighted_role = role_weight * objectives["role"]
    weighted_mean = mean_weight * objectives["p1_mean"]
    if not all(bool(torch.isfinite(value)) for value in (
            total_standard_loss, weighted_role, weighted_mean)):
        raise FloatingPointError(f"non-finite RSM1 loss at update {iteration}")

    audit = None
    if iteration in PREFLIGHT_AUDIT_UPDATES:
        audit = parameter_gradient_audit(
            supervised_loss, weighted_role, weighted_mean,
            member_parameters, mean_parameters, iteration)

    # Preserve the complete native Real update, including native UQ.  Add each
    # role teacher only to its frozen parameter scope.
    supervised_loss.backward(retain_graph=True)
    uq_parameters = rehearsal.ensemble_train.uq_trainable_parameters(model)
    soft_gradients = torch.autograd.grad(
        soft_sps_weight * soft_sps_loss, uq_parameters,
        retain_graph=True, allow_unused=True)
    _add_gradients(uq_parameters, soft_gradients)
    role_gradients = torch.autograd.grad(
        weighted_role, member_parameters, retain_graph=True,
        allow_unused=True)
    _add_gradients(member_parameters, role_gradients)
    mean_gradients = torch.autograd.grad(
        weighted_mean, mean_parameters, allow_unused=True)
    _add_gradients(mean_parameters, mean_gradients)

    values = {
        "real_loss": float(total_standard_loss.detach()),
        "real_supervised_loss": float(supervised_loss.detach()),
        "real_point_loss": float(point_loss.detach()),
        "real_member_loss": float(member_loss.detach()),
        "real_quantile_loss": float(quantile_loss.detach()),
        "real_soft_sps_loss": float(soft_sps_loss.detach()),
        "real_soft_sps_score": float(soft_sps_score.detach()),
        "real_soft_sps_weight": soft_sps_weight,
        "real_bootstrap_fraction": float(bootstrap_fraction.detach()),
        "role_a_fluctuation_loss": float(objectives["a_fluctuation"].detach()),
        "role_b_fluctuation_loss": float(objectives["b_fluctuation"].detach()),
        "role_distill_loss": float(objectives["role"].detach()),
        "role_distill_weight": role_weight,
        "mean_distill_loss": float(objectives["p1_mean"].detach()),
        "mean_distill_weight": mean_weight,
        **{f"real_{key}": float(value.detach())
           for key, value in point_components.items()},
    }
    return values, audit


def terminal_metric_gate(metrics: dict, baseline: dict | None = None) -> dict:
    if baseline is None:
        baseline = NATIVE_STEP4000_METRICS
    delta = {
        key: float(metrics[key]) - expected
        for key, expected in baseline.items()}
    shadow = sum(
        rehearsal.NON_TIME_COEFFICIENTS[key] * delta[key]
        for key in rehearsal.NON_TIME_COEFFICIENTS)
    component_floors = {
        key: delta[key] >= GATE_COMPONENT_DELTA_FLOOR - 1e-12
        for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")}
    passed = bool(
        delta["point_score"] >= GATE_POINT_DELTA_MIN - 1e-12
        and all(component_floors.values())
        and delta["native_sps_score"] >= GATE_SPS_DELTA_MIN - 1e-12
        and shadow >= GATE_NON_TIME_SHADOW_MIN - 1e-12)
    return {
        "status": "PROMOTE_RSM1B_MODEL_04000" if passed
        else "STOP_RSM1B_UPDATE4000_GATE",
        "pass": passed, "metrics": {key: float(value)
                                     for key, value in metrics.items()},
        "baseline": dict(baseline), "delta": delta,
        "component_floors": component_floors,
        "non_time_shadow": shadow,
        "evidence_class": EVIDENCE_CLASS,
    }


def evaluate_terminal_gate(
        model, real: dict, device: torch.device, identity: dict) -> dict:
    if real["validation_access_updates"]:
        raise RuntimeError("fixed725 was accessed before its unique update4000 gate")
    metrics = rehearsal.ensemble_train.evaluate(
        model, real["validation_loader"], real["normalizer"], device,
        rehearsal.REAL_ACCURACY_WEIGHTS)
    real["validation_access_updates"].append(TERMINAL_UPDATE)
    result = terminal_metric_gate(metrics, identity["metrics_update4000"])
    result["identity_summary"] = identity
    return result


def _stopped(status, iteration, history, preflight, gate, anchor, started):
    return {
        "status": status, "completed_updates": iteration,
        "history": history, "preflight": preflight, "gate": gate,
        "anchor": anchor, "checkpoint": None,
        "wall_seconds": time.time() - started,
    }


def train_endpoint(
        args, model, real: dict, identity: dict,
        device: torch.device) -> dict:
    member_parameters, mean_parameters = role_parameters(model)
    anchor_loader = make_anchor_loader(
        real["loader"].dataset, args.num_workers)
    anchor_start = evaluate_train_anchor(
        model, anchor_loader, real["normalizer"], real["teacher"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rehearsal.FORMAL_LR,
        weight_decay=rehearsal.FORMAL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    iterator = rehearsal.ensemble_train.common.repro.cycling(real["loader"])
    history, audits = [], []
    sums: dict[str, float] = {}
    count = 0
    preflight = gate = None
    anchor = {"start": anchor_start, "update200": None}
    started = time.time()
    for iteration in range(1, TERMINAL_UPDATE + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values, audit = backward_real_role_specialization(
            model, next(iterator), real["normalizer"], real["teacher"],
            device, iteration, member_parameters, mean_parameters)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), rehearsal.FORMAL_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        if audit is not None:
            audits.append(audit)
            print(json.dumps({"event": "c208_rsm1b_audit", **audit}), flush=True)
        values["learning_rate"] = optimizer.param_groups[0]["lr"]
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration == PREFLIGHT_UPDATE:
            anchor["update200"] = evaluate_train_anchor(
                model, anchor_loader, real["normalizer"], real["teacher"],
                device)
            preflight = evaluate_preflight(
                audits, anchor["start"], anchor["update200"])
            print(json.dumps({
                "event": "c208_rsm1b_preflight", **preflight,
            }), flush=True)
            if not preflight["pass"]:
                return _stopped(
                    preflight["status"], iteration, history, preflight,
                    gate, anchor, started)
        if iteration % rehearsal.FORMAL_LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps({"event": "c208_rsm1b_progress", **row}), flush=True)
            sums.clear()
            count = 0
        if iteration == TERMINAL_UPDATE:
            gate = evaluate_terminal_gate(model, real, device, identity)
            print(json.dumps({"event": "c208_rsm1b_gate", **gate}), flush=True)
            if not gate["pass"]:
                return _stopped(
                    gate["status"], iteration, history, preflight, gate,
                    anchor, started)

    checkpoint = args.output_dir / "model_04000.pth"
    stats = rehearsal.ensemble_train.common.normalizer_dict(real["normalizer"])
    rehearsal.ensemble_train.common.save_checkpoint(
        checkpoint, model, args, TERMINAL_UPDATE, history, stats, history[-1])
    return {
        "status": "COMPLETE_RSM1B_MODEL_04000_AFTER_GATES",
        "completed_updates": TERMINAL_UPDATE, "history": history,
        "preflight": preflight, "gate": gate, "anchor": anchor,
        "checkpoint": str(checkpoint), "wall_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    identity = load_identity_step4000(args)
    args.phase = "real"
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal C208 RSM1 training requires CUDA")
    rehearsal.ensemble_train.seed_everything(rehearsal.FORMAL_SEED)
    torch.set_float32_matmul_precision("high")
    payload = rehearsal.load_start_payload(args)
    model = rehearsal.load_model(payload, device)
    del payload
    args.output_dir.mkdir(parents=True, exist_ok=False)
    real = prepare_real_training(args, device)
    rehearsal.ensemble_train.common.set_model_stats(model, real["normalizer"])
    member_parameters, mean_parameters = role_parameters(model)
    print(json.dumps({
        "event": "c208_rsm1b_contract",
        "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
        "start_sim_sha256": args.start_sim_sha256,
        "teacher_cache_sha256": args.teacher_cache_sha256,
        "role_distill": {
            "weight": ROLE_DISTILL_WEIGHT, "ramp_updates": DISTILL_RAMP_UPDATES,
            "assignment": {"heads_0_1": "A_fluctuation",
                           "heads_2_3": "B_fluctuation"},
            "gradient_scope": "member_point_heads_only",
            "trainable_parameters": sum(p.numel() for p in member_parameters),
        },
        "mean_distill": {
            "weight": MEAN_DISTILL_WEIGHT, "ramp_updates": DISTILL_RAMP_UPDATES,
            "target": "formal_P1_temporal_mean",
            "gradient_scope": "mean_profile_refiner_only",
            "trainable_parameters": sum(p.numel() for p in mean_parameters),
        },
        "standard_real_loss_changed": False,
        "endpoint_teacher": None,
        "native_uq_objective_changed": False,
        "auxiliary_gradient_to_uq_parameters": False,
        "fixed725_access": "once_at_update4000_only",
    }), flush=True)
    result = train_endpoint(args, model, real, identity, device)
    summary = {
        "method": "c208_rsm1b_foldaware_role_specialization_calibrated",
        "evidence_class": EVIDENCE_CLASS,
        "independence_warning": (
            "A/B fold best iterations used fixed725 folds for checkpoint "
            "selection; fixed725 is contaminated diagnostic evidence only."),
        "start_sim": str(args.start_sim),
        "start_sim_sha256": args.start_sim_sha256,
        "teacher_cache": real["teacher_metadata"],
        "identity_summary": identity,
        "real_contract": {
            "counts": real["counts"],
            "split_manifest_sha256": real["manifest_sha256"],
            "real_stats_sha256": real["stats_sha256"],
            "validation_optimizer_access": False,
            "validation_access_updates": real["validation_access_updates"],
        },
        "training_contract": {
            "requested_updates": args.updates,
            "terminal_update": TERMINAL_UPDATE,
            "scheduler_t_max": rehearsal.FORMAL_SCHEDULER_T_MAX,
            "standard_real_loss_changed": False,
            "native_uq_objective_changed": False,
            "auxiliary_gradient_to_uq_parameters": False,
            "endpoint_teacher": None,
            "role_distill_weight": ROLE_DISTILL_WEIGHT,
            "mean_distill_weight": MEAN_DISTILL_WEIGHT,
            "distill_ramp_updates": DISTILL_RAMP_UPDATES,
            "role_gradient_scope": "member_point_heads_only_via_autograd_grad",
            "mean_gradient_scope": "mean_profile_refiner_only_via_autograd_grad",
            "checkpoints_saved": (
                ["model_04000.pth"] if result["checkpoint"] else []),
        },
        "preflight_contract": {
            "audit_updates": list(PREFLIGHT_AUDIT_UPDATES),
            "parameter_gradient_screen": [
                "member_point_heads", "mean_profile_refiner"],
            "anchor_source": "deterministic_fixed128_train_only",
            "anchor_decline_fraction_min": ANCHOR_DECLINE_FRACTION,
            "on_failure": "STOP_NO_FIXED725_NO_CHECKPOINT",
        },
        "gate_contract": {
            "update": TERMINAL_UPDATE,
            "baseline": NATIVE_STEP4000_METRICS,
            "point_score_delta_min": GATE_POINT_DELTA_MIN,
            "component_delta_floor": GATE_COMPONENT_DELTA_FLOOR,
            "native_sps_delta_min": GATE_SPS_DELTA_MIN,
            "non_time_shadow_min": GATE_NON_TIME_SHADOW_MIN,
            "single_fixed725_access": True,
            "on_failure": "STOP_NO_CHECKPOINT",
        },
        **result,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "status": result["status"],
        "summary": str(summary_path), "checkpoint": result["checkpoint"],
    }), flush=True)


if __name__ == "__main__":
    main()
