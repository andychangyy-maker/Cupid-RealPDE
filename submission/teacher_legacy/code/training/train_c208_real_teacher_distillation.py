#!/usr/bin/env python3
"""RTD1: distill the fold-aware current-P1 teacher during C208 Real training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys
import time

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_real_sim_rehearsal as rehearsal  # noqa: E402
import train_p1_direct_native_endpoint_teacher as endpoint_teacher  # noqa: E402
import train_p1_direct_native_quantile_teacher as quantile_teacher  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


CACHE_SCHEMA_VERSION = 1
CACHE_METHOD = "c208_rtd1_real2576_foldaware_physical_v1"
EXPECTED_WINDOWS = 2576
POINT_WEIGHTS = {"b": 0.5, "c": 0.5}
BOUNDS_WEIGHTS = {"b": 0.625, "c": 0.25}
EVIDENCE_CLASS = "fixed725_checkpoint_selection_contaminated_teacher_diagnostic"
PHYSICAL_UNITS = {
    "point_uv_physical": "physical_velocity",
    "lower_uv_physical": "physical_velocity",
    "upper_uv_physical": "physical_velocity",
}
TEACHER_TENSOR_SPECS = {
    "point_uv_physical": torch.float16,
    "lower_uv_physical": torch.float32,
    "upper_uv_physical": torch.float32,
}

POINT_DISTILL_WEIGHT = 4e-5
ENDPOINT_DISTILL_WEIGHT = 0.007
DISTILL_RAMP_UPDATES = 50
PREFLIGHT_UPDATE = 200
PREFLIGHT_AUDIT_UPDATES = (50, 100, 150, 200)
PREFLIGHT_LOSS_RATIO_RANGE = (0.002, 0.10)
PREFLIGHT_GRAD_RATIO_RANGE = (0.01, 0.30)
PREFLIGHT_MEDIAN_COSINE_MIN = -0.10
PREFLIGHT_WORST_COSINE_MIN = -0.50
GATE_UPDATE = rehearsal.FORMAL_GATE_UPDATE
TERMINAL_UPDATE = rehearsal.FORMAL_TERMINAL_UPDATE
GATE_POINT_DELTA_MIN = 0.05
GATE_COMPONENT_DELTA_FLOOR = -0.08
GATE_SPS_DELTA_FLOOR = 0.0
GATE_NON_TIME_SHADOW_MIN = 0.10
REFERENCE_WIDTH = endpoint_teacher.REFERENCE_WIDTH


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
        raise ValueError("formal RTD1 requires --updates 5000")
    args.start_sim_sha256 = rehearsal.normalized_sha256(
        args.start_sim_sha256, "start-sim-sha256")
    if args.start_sim_sha256 != rehearsal.FORMAL_START_SIM_SHA256:
        raise ValueError("start-sim-sha256 is not the frozen Sim10k start")
    args.teacher_cache_sha256 = rehearsal.normalized_sha256(
        args.teacher_cache_sha256, "teacher-cache-sha256")
    args.identity_summary_sha256 = rehearsal.normalized_sha256(
        args.identity_summary_sha256, "identity-summary-sha256")
    if args.identity_summary_sha256 != rehearsal.FORMAL_IDENTITY_SUMMARY_SHA256:
        raise ValueError("identity summary is not the frozen standard Real control")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse RTD1 output: {args.output_dir}")


def sha256(path: Path) -> str:
    return rehearsal.sha256(path)


def tensor_sha256(value: torch.Tensor, chunk_size: int = 16) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise TypeError("cache tensor hashing requires a CPU tensor")
    digest = hashlib.sha256()
    for start in range(0, len(value), chunk_size):
        chunk = value[start:start + chunk_size].contiguous()
        digest.update(chunk.numpy().tobytes(order="C"))
    return digest.hexdigest()


def ordered_train_keys(windows, manifest: list[dict]) -> list[list[object]]:
    keys = []
    for window in windows:
        trajectory = int(window.trajectory)
        if not 0 <= trajectory < len(manifest):
            raise ValueError("Real window has an invalid trajectory index")
        start = int(window.start)
        row = manifest[trajectory]
        if start not in row.get("train_starts", []):
            raise ValueError("Real window is absent from manifest train_starts")
        keys.append([str(row["file"]), start])
    return keys


def expected_teacher_shape(dataset) -> tuple[int, ...]:
    if not hasattr(dataset, "windows") or len(dataset) != EXPECTED_WINDOWS:
        raise ValueError("RTD1 cache requires the ordered fixed2576 dataset")
    _history, target = dataset[0]
    if target.ndim != 4 or target.shape[-1] < 2:
        raise ValueError("Real target must have shape (T,H,W,C>=2)")
    return (len(dataset), *target.shape[:-1], 2)


def _recorded_tensor_sha(metadata: dict, name: str) -> str:
    identities = metadata.get("tensor_sha256")
    if not isinstance(identities, dict):
        raise ValueError("teacher cache has no tensor SHA-256 contract")
    recorded = identities.get(name)
    if isinstance(recorded, dict):
        recorded = recorded.get("sha256")
    return rehearsal.normalized_sha256(recorded, f"tensor_sha256[{name}]")


def _identity_sha(value, name: str) -> str:
    if isinstance(value, dict):
        value = value.get("sha256")
    return rehearsal.normalized_sha256(value, name)


def _validate_formal_source_contracts(metadata: dict) -> None:
    sources = metadata.get("source_contracts")
    if not isinstance(sources, dict):
        raise ValueError("teacher cache has no source contracts")
    for name, expected in quantile_teacher.FORMAL_SOURCE_SHA256.items():
        actual = _identity_sha(sources.get(name), f"source_contracts[{name}]")
        if actual != expected:
            raise ValueError(f"teacher cache formal source SHA differs at {name}")

    oof = metadata.get("oof_contracts")
    if not isinstance(oof, dict) or set(oof) != {"a", "b"}:
        raise ValueError("teacher cache has no complete A/B OOF contracts")
    for source in ("a", "b"):
        contract = oof[source]
        if not isinstance(contract, dict):
            raise ValueError(f"teacher cache {source} OOF contract is invalid")
        expected = quantile_teacher.FORMAL_OOF_SHA256[source]
        summary = contract.get("summary_file", contract.get("summary.json"))
        if _identity_sha(summary, f"oof_contracts[{source}][summary]") != (
                expected["summary.json"]):
            raise ValueError(f"teacher cache {source} OOF summary SHA differs")
        rows = contract.get("folds")
        if isinstance(rows, list):
            by_fold = {int(row.get("fold", -1)): row for row in rows}
        else:
            by_fold = {}
        for fold in range(5):
            filename = f"fold_{fold:02d}_best.pth"
            value = (
                by_fold.get(fold, {}).get("checkpoint")
                if by_fold else contract.get(filename))
            if _identity_sha(
                    value, f"oof_contracts[{source}][{filename}]") != (
                    expected[filename]):
                raise ValueError(
                    f"teacher cache {source} OOF fold {fold} SHA differs")


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
    if any(key in payload for key in (
            "prediction_uv_normalized", "point_uv_normalized",
            "lower_uv_normalized", "upper_uv_normalized", "target", "truth")):
        raise ValueError("normalized teachers and target-bearing caches are forbidden")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("teacher cache has no metadata")
    manifest_sha = rehearsal.canonical_sha256(manifest)
    if metadata.get("split_manifest_sha256") != manifest_sha:
        raise ValueError("teacher cache split manifest SHA mismatch")
    if metadata.get("real_stats_sha256") != real_stats_sha256:
        raise ValueError("teacher cache Real stats/unit transform mismatch")
    keys = ordered_train_keys(dataset.windows, manifest)
    if metadata.get("train_keys") != keys:
        raise ValueError("teacher cache train keys/order mismatch")
    if metadata.get("point_weights") != POINT_WEIGHTS:
        raise ValueError("teacher cache is not current P1 B=.5/C=.5 point")
    if metadata.get("bounds_weights") != BOUNDS_WEIGHTS:
        raise ValueError("teacher cache is not fixed P0 B=.625/C=.25 bounds")
    if metadata.get("point_weights") == metadata.get("bounds_weights"):
        raise ValueError("point and bounds composition weights must differ")
    if metadata.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError("teacher cache evidence provenance mismatch")
    if metadata.get("units") != PHYSICAL_UNITS:
        raise ValueError("teacher cache units are not physical velocity")
    source_contracts = metadata.get("source_contracts")
    oof_contracts = metadata.get("oof_contracts")
    reconstruction = metadata.get("reconstruction")
    if not isinstance(source_contracts, dict) or not source_contracts:
        raise ValueError("teacher cache has no source contracts")
    if not isinstance(oof_contracts, dict) or set(oof_contracts) != {"a", "b"}:
        raise ValueError("teacher cache has no complete A/B OOF contracts")
    if not isinstance(reconstruction, dict) or not reconstruction:
        raise ValueError("teacher cache has no reconstruction provenance")
    if metadata.get("fold_manifest") != expected_fold_manifest:
        raise ValueError("teacher cache fold manifest mismatch")
    fold_rows = reconstruction.get("folds")
    if not isinstance(fold_rows, list):
        raise ValueError("teacher cache reconstruction has no fold rows")
    expected_counts = {
        fold: expected_folds.count(fold) for fold in sorted(set(expected_folds))}
    actual_counts = {
        int(row.get("fold", -1)): int(row.get("samples", -1))
        for row in fold_rows if isinstance(row, dict)}
    if actual_counts != expected_counts:
        raise ValueError("teacher cache reconstruction fold counts mismatch")
    _validate_formal_source_contracts(metadata)

    expected_shape = expected_teacher_shape(dataset)
    tensors = {}
    for name, dtype in TEACHER_TENSOR_SPECS.items():
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"teacher cache {name} must be a tensor")
        if value.device.type != "cpu" or value.dtype != dtype:
            raise TypeError(f"teacher cache {name} has wrong physical dtype")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"teacher cache {name} shape mismatch")
        if not all(bool(torch.isfinite(value[start:start + 16]).all())
                   for start in range(0, len(value), 16)):
            raise ValueError(f"teacher cache {name} contains non-finite values")
        actual_tensor_sha = tensor_sha256(value)
        if actual_tensor_sha != _recorded_tensor_sha(metadata, name):
            raise ValueError(f"teacher cache {name} tensor SHA mismatch")
        tensors[name] = value
    if any(bool((tensors["lower_uv_physical"][start:start + 16]
                 > tensors["upper_uv_physical"][start:start + 16]).any())
           for start in range(0, EXPECTED_WINDOWS, 16)):
        raise ValueError("teacher cache absolute lower exceeds upper")
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
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"teacher cache SHA-256 {actual} != expected {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensors, metadata = validate_teacher_cache_payload(
        payload, manifest, dataset, expected_folds, expected_fold_manifest,
        real_stats_sha256)
    return tensors, {
        **metadata, "path": str(path.resolve()), "sha256": actual,
    }


def load_identity_step2000(args: argparse.Namespace) -> dict:
    proxy = argparse.Namespace(
        rehearsal_weight=rehearsal.FORMAL_REHEARSAL_WEIGHT,
        identity_summary=args.identity_summary,
        identity_summary_sha256=args.identity_summary_sha256,
    )
    return rehearsal.load_identity_step2000(proxy)


def prepare_real_training(args: argparse.Namespace, device: torch.device) -> dict:
    real = rehearsal.prepare_real_training(args, device)
    manifest_path = args.output_dir / "real_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = real["loader"].dataset.base_dataset
    _tail_folds, fold_manifest = trajectory_folds(manifest, 5)
    point_folds = quantile_teacher.partition_fold_labels(
        manifest, fold_manifest, "train_starts")
    tensors, metadata = load_teacher_cache(
        args.teacher_cache, args.teacher_cache_sha256,
        manifest, dataset, point_folds, fold_manifest, real["stats_sha256"])
    real.update({
        "teacher": tensors,
        "teacher_metadata": metadata,
        "fold_manifest": fold_manifest,
    })
    return real


def ramp_weight(weight: float, iteration: int) -> float:
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    return weight * min(1.0, iteration / DISTILL_RAMP_UPDATES)


def endpoint_distillation_objective(
        prediction_norm: torch.Tensor, lower_extent: torch.Tensor,
        upper_extent: torch.Tensor, teacher_lower: torch.Tensor,
        teacher_upper: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    point = rehearsal.ensemble_train.common.to_physical_target(
        prediction_norm.float(), normalizer)[..., :2].detach()
    lower = point - lower_extent.float() * scale
    upper = point + upper_extent.float() * scale
    reference = lower.new_tensor(REFERENCE_WIDTH).view(1, 1, 1, 1, 2)
    lower_loss = F.smooth_l1_loss(
        (lower - teacher_lower.float()) / reference,
        torch.zeros_like(lower))
    upper_loss = F.smooth_l1_loss(
        (upper - teacher_upper.float()) / reference,
        torch.zeros_like(upper))
    return 0.5 * (lower_loss + upper_loss)


def _gradient_geometry(left_gradients, right_gradients) -> dict[str, float | bool]:
    left_sq = 0.0
    right_sq = 0.0
    dot = 0.0
    finite = True
    for left, right in zip(left_gradients, right_gradients):
        if left is None and right is None:
            continue
        if left is None:
            left = torch.zeros_like(right)
        if right is None:
            right = torch.zeros_like(left)
        finite = finite and bool(torch.isfinite(left).all()) and bool(
            torch.isfinite(right).all())
        left_float = left.detach().float()
        right_float = right.detach().float()
        left_sq += float(left_float.square().sum())
        right_sq += float(right_float.square().sum())
        dot += float((left_float * right_float).sum())
    left_norm = math.sqrt(left_sq)
    right_norm = math.sqrt(right_sq)
    nonzero = left_norm > 0.0 and right_norm > 0.0
    ratio = right_norm / left_norm if left_norm > 0.0 else math.inf
    cosine = dot / (left_norm * right_norm) if nonzero else math.nan
    return {
        "gradient_ratio": ratio,
        "cosine": cosine,
        "finite_nonzero": bool(
            finite and nonzero and math.isfinite(ratio)
            and math.isfinite(cosine)),
    }


def training_signal_audit(
        supervised_loss: torch.Tensor, native_uq_loss: torch.Tensor,
        weighted_point: torch.Tensor, weighted_endpoint: torch.Tensor,
        prediction: torch.Tensor, quantiles: torch.Tensor,
        iteration: int,
        ) -> dict:
    """Cheap output-gradient screen; not a full UQ-parameter audit."""
    point_base = torch.autograd.grad(
        supervised_loss, prediction, retain_graph=True, allow_unused=True)
    point_aux = torch.autograd.grad(
        weighted_point, prediction, retain_graph=True, allow_unused=True)
    endpoint_base = torch.autograd.grad(
        native_uq_loss, quantiles, retain_graph=True, allow_unused=True)
    endpoint_aux = torch.autograd.grad(
        weighted_endpoint, quantiles, retain_graph=True, allow_unused=True)
    denominator = float(supervised_loss.detach().abs())
    point = _gradient_geometry(point_base, point_aux)
    endpoint = _gradient_geometry(endpoint_base, endpoint_aux)
    point["loss_ratio"] = (
        float(weighted_point.detach().abs()) / denominator
        if denominator > 0.0 else math.inf)
    endpoint["loss_ratio"] = (
        float(weighted_endpoint.detach().abs()) / denominator
        if denominator > 0.0 else math.inf)
    point["finite_nonzero"] = bool(
        point["finite_nonzero"] and math.isfinite(point["loss_ratio"]))
    endpoint["finite_nonzero"] = bool(
        endpoint["finite_nonzero"] and math.isfinite(endpoint["loss_ratio"]))
    return {"iteration": iteration, "point": point, "endpoint": endpoint}


def evaluate_preflight(audits: list[dict]) -> dict:
    expected = list(PREFLIGHT_AUDIT_UPDATES)
    if [row.get("iteration") for row in audits] != expected:
        return {
            "status": "STOP_RTD1_PREFLIGHT", "pass": False,
            "reason": "missing_or_duplicate_audit_updates", "audits": audits,
        }
    checks = {}
    for branch in ("point", "endpoint"):
        rows = [audit[branch] for audit in audits]
        loss_ok = all(
            PREFLIGHT_LOSS_RATIO_RANGE[0] <= row["loss_ratio"]
            <= PREFLIGHT_LOSS_RATIO_RANGE[1] for row in rows)
        gradient_ok = all(
            PREFLIGHT_GRAD_RATIO_RANGE[0] <= row["gradient_ratio"]
            <= PREFLIGHT_GRAD_RATIO_RANGE[1] for row in rows)
        cosines = [row["cosine"] for row in rows]
        checks[branch] = {
            "finite_nonzero": all(row["finite_nonzero"] for row in rows),
            "loss_ratio_range": loss_ok,
            "gradient_ratio_range": gradient_ok,
            "median_cosine": median(cosines),
            "worst_cosine": min(cosines),
            "cosine_gate": (
                median(cosines) >= PREFLIGHT_MEDIAN_COSINE_MIN
                and min(cosines) >= PREFLIGHT_WORST_COSINE_MIN),
        }
    passed = all(
        row["finite_nonzero"] and row["loss_ratio_range"]
        and row["gradient_ratio_range"] and row["cosine_gate"]
        for row in checks.values())
    return {
        "status": "GO_RTD1_TO_UPDATE2000" if passed else "STOP_RTD1_PREFLIGHT",
        "pass": passed, "checks": checks, "audits": audits,
    }


def _add_gradients(parameters, gradients) -> None:
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            parameter.grad.add_(gradient)


def backward_real_teacher_distillation(
        model, batch, normalizer, teacher: dict[str, torch.Tensor],
        device: torch.device, iteration: int,
        ) -> tuple[dict[str, float], dict | None]:
    history, target, trajectory_ids, sample_indices = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    history_norm, target_norm = normalizer.preprocess(history, target)
    (prediction, members, quantiles, lower, upper, features,
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
    supervised_loss = (
        0.25 * point_loss + member_loss + 0.05 * quantile_loss)
    native_uq_loss = 0.05 * quantile_loss + soft_sps_weight * soft_sps_loss
    total_standard_loss = supervised_loss + soft_sps_weight * soft_sps_loss

    teacher_point = teacher["point_uv_physical"][sample_indices].to(
        device=device, dtype=torch.float32, non_blocking=True)
    teacher_lower = teacher["lower_uv_physical"][sample_indices].to(
        device=device, non_blocking=True)
    teacher_upper = teacher["upper_uv_physical"][sample_indices].to(
        device=device, non_blocking=True)
    point_distill, point_distill_components = (
        rehearsal.ensemble_train.component_point_distillation_loss(
            prediction, teacher_point, normalizer))
    endpoint_distill = endpoint_distillation_objective(
        prediction, lower, upper, teacher_lower, teacher_upper, normalizer)
    point_weight = ramp_weight(POINT_DISTILL_WEIGHT, iteration)
    endpoint_weight = ramp_weight(ENDPOINT_DISTILL_WEIGHT, iteration)
    weighted_point = point_weight * point_distill
    weighted_endpoint = endpoint_weight * endpoint_distill
    if not all(bool(torch.isfinite(value)) for value in (
            total_standard_loss, weighted_point, weighted_endpoint)):
        raise FloatingPointError(f"non-finite RTD1 loss at update {iteration}")

    audit = None
    if iteration in PREFLIGHT_AUDIT_UPDATES:
        audit = training_signal_audit(
            supervised_loss, native_uq_loss, weighted_point,
            weighted_endpoint, prediction, quantiles, iteration)

    # Preserve the standard Real update exactly, then add point distillation
    # through its full graph and endpoint distillation only to native UQ params.
    supervised_loss.backward(retain_graph=True)
    uq_parameters = rehearsal.ensemble_train.uq_trainable_parameters(model)
    soft_gradients = torch.autograd.grad(
        soft_sps_weight * soft_sps_loss, uq_parameters,
        retain_graph=True, allow_unused=True)
    _add_gradients(uq_parameters, soft_gradients)
    weighted_point.backward(retain_graph=True)
    endpoint_gradients = torch.autograd.grad(
        weighted_endpoint, uq_parameters, allow_unused=True)
    _add_gradients(uq_parameters, endpoint_gradients)

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
        "point_distill_loss": float(point_distill.detach()),
        "point_distill_weight": point_weight,
        "endpoint_distill_loss": float(endpoint_distill.detach()),
        "endpoint_distill_weight": endpoint_weight,
        **{
            f"real_{key}": float(value.detach())
            for key, value in point_components.items()
        },
        **{
            key: float(value.detach())
            for key, value in point_distill_components.items()
        },
    }
    return values, audit


def evaluate_step2000_gate(model, real: dict, device, identity: dict) -> dict:
    if real["validation_access_updates"]:
        raise RuntimeError("fixed725 was already accessed before its single gate")
    metrics = rehearsal.ensemble_train.evaluate(
        model, real["validation_loader"], real["normalizer"], device,
        rehearsal.REAL_ACCURACY_WEIGHTS)
    real["validation_access_updates"].append(GATE_UPDATE)
    metrics = {key: float(value) for key, value in metrics.items()}
    baseline = identity["metrics"]
    delta = {
        key: metrics[key] - baseline[key]
        for key in rehearsal.IDENTITY_METRIC_KEYS}
    shadow = sum(
        rehearsal.NON_TIME_COEFFICIENTS[key] * delta[key]
        for key in rehearsal.NON_TIME_COEFFICIENTS)
    component_floors = all(
        delta[key] >= GATE_COMPONENT_DELTA_FLOOR
        for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score"))
    passed = bool(
        delta["point_score"] >= GATE_POINT_DELTA_MIN
        and component_floors
        and delta["native_sps_score"] >= GATE_SPS_DELTA_FLOOR
        and shadow >= GATE_NON_TIME_SHADOW_MIN)
    return {
        "status": "GO_RTD1_TO_UPDATE4000" if passed
        else "STOP_RTD1_UPDATE2000_GATE",
        "evidence_class": EVIDENCE_CLASS,
        "independence_warning": (
            "A/B OOF best_iteration used the held fixed725 folds for checkpoint "
            "selection; this gate is contaminated diagnostic evidence only."),
        "metrics": metrics, "identity_summary": identity,
        "delta": delta, "component_floors_pass": component_floors,
        "non_time_shadow": shadow, "pass": passed,
    }


def train_endpoint(args, model, real: dict, identity: dict,
                   device: torch.device) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rehearsal.FORMAL_LR,
        weight_decay=rehearsal.FORMAL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    iterator = rehearsal.ensemble_train.common.repro.cycling(real["loader"])
    history = []
    audits = []
    sums: dict[str, float] = {}
    count = 0
    started = time.time()
    preflight = None
    gate = None
    for iteration in range(1, TERMINAL_UPDATE + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values, audit = backward_real_teacher_distillation(
            model, next(iterator), real["normalizer"], real["teacher"],
            device, iteration)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), rehearsal.FORMAL_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        if audit is not None:
            audits.append(audit)
            print(json.dumps({"event": "c208_rtd1_audit", **audit}), flush=True)
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % rehearsal.FORMAL_LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps({"event": "c208_rtd1_progress", **row}), flush=True)
            sums.clear()
            count = 0
        if iteration == PREFLIGHT_UPDATE:
            preflight = evaluate_preflight(audits)
            print(json.dumps({
                "event": "c208_rtd1_preflight", **preflight,
            }), flush=True)
            if not preflight["pass"]:
                return _stopped(
                    preflight["status"], iteration, history,
                    preflight, gate, started)
        if iteration == GATE_UPDATE:
            gate = evaluate_step2000_gate(model, real, device, identity)
            print(json.dumps({"event": "c208_rtd1_gate", **gate}), flush=True)
            if not gate["pass"]:
                return _stopped(
                    gate["status"], iteration, history,
                    preflight, gate, started)

    stats = rehearsal.ensemble_train.common.normalizer_dict(real["normalizer"])
    checkpoint = args.output_dir / "model_04000.pth"
    rehearsal.ensemble_train.common.save_checkpoint(
        checkpoint, model, args, TERMINAL_UPDATE,
        history, stats, history[-1])
    return {
        "status": "COMPLETE_RTD1_UPDATE4000_AFTER_GATES",
        "completed_updates": TERMINAL_UPDATE,
        "history": history, "preflight": preflight, "gate": gate,
        "checkpoint": str(checkpoint),
        "wall_seconds": time.time() - started,
    }


def _stopped(status, iteration, history, preflight, gate, started) -> dict:
    return {
        "status": status, "completed_updates": iteration,
        "history": history, "preflight": preflight, "gate": gate,
        "checkpoint": None, "wall_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.phase = "real"
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal C208 RTD1 training requires CUDA")
    rehearsal.ensemble_train.seed_everything(rehearsal.FORMAL_SEED)
    torch.set_float32_matmul_precision("high")
    payload = rehearsal.load_start_payload(args)
    model = rehearsal.load_model(payload, device)
    del payload
    args.output_dir.mkdir(parents=True, exist_ok=False)
    real = prepare_real_training(args, device)
    rehearsal.ensemble_train.common.set_model_stats(model, real["normalizer"])
    identity = load_identity_step2000(args)
    print(json.dumps({
        "event": "c208_rtd1_contract",
        "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
        "start_sim_sha256": args.start_sim_sha256,
        "teacher_cache_sha256": args.teacher_cache_sha256,
        "point_distill": {"weight": POINT_DISTILL_WEIGHT,
                          "ramp_updates": DISTILL_RAMP_UPDATES,
                          "gradient_scope": "full_point_graph"},
        "endpoint_distill": {"weight": ENDPOINT_DISTILL_WEIGHT,
                             "ramp_updates": DISTILL_RAMP_UPDATES,
                             "gradient_scope": "uq_parameters_only"},
        "preflight_audit_scope": (
            "output_gradient_screen_at_prediction_and_quantiles"),
        "fixed725_optimizer_access": False,
        "fixed725_evidence_class": EVIDENCE_CLASS,
    }), flush=True)
    result = train_endpoint(args, model, real, identity, device)
    summary = {
        "method": "c208_rtd1b_real_foldaware_teacher_distillation",
        "evidence_class": EVIDENCE_CLASS,
        "independence_warning": (
            "fixed725 gate is checkpoint-selection contaminated; it is not "
            "independent OOF/held-out evidence and cannot replace official hidden."),
        "start_sim": str(args.start_sim),
        "start_sim_sha256": args.start_sim_sha256,
        "teacher_cache": real["teacher_metadata"],
        "real_contract": {
            "counts": real["counts"],
            "split_manifest_sha256": real["manifest_sha256"],
            "real_stats_sha256": real["stats_sha256"],
            "validation_optimizer_access": False,
            "validation_access_updates": real["validation_access_updates"],
        },
        "training_contract": {
            "requested_updates": args.updates,
            "preflight_update": PREFLIGHT_UPDATE,
            "preflight_audit_updates": list(PREFLIGHT_AUDIT_UPDATES),
            "gate_update": GATE_UPDATE,
            "terminal_update": TERMINAL_UPDATE,
            "scheduler_t_max": rehearsal.FORMAL_SCHEDULER_T_MAX,
            "point_distill_weight": POINT_DISTILL_WEIGHT,
            "endpoint_distill_weight": ENDPOINT_DISTILL_WEIGHT,
            "distill_ramp_updates": DISTILL_RAMP_UPDATES,
            "standard_real_loss_changed": False,
            "endpoint_gradient_scope": "uq_parameters_only_via_autograd_grad",
            "preflight_audit_scope": (
                "output_gradient_screen_at_prediction_and_quantiles"),
            "preflight_audit_limit": (
                "screen does not cover utility-refiner direct fused-feature "
                "parameter gradients; it is not a full UQ-parameter audit"),
            "checkpoints_saved": (
                ["model_04000.pth"] if result["checkpoint"] else []),
        },
        "gate_contract": {
            "point_score_delta_min": GATE_POINT_DELTA_MIN,
            "component_delta_floor": GATE_COMPONENT_DELTA_FLOOR,
            "native_sps_delta_floor": GATE_SPS_DELTA_FLOOR,
            "non_time_shadow_min": GATE_NON_TIME_SHADOW_MIN,
            "single_fixed725_access": True,
            "on_failure": "STOP_NO_CHECKPOINT",
        },
        **result,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "status": result["status"],
        "summary": str(args.output_dir / "summary.json"),
        "checkpoint": result["checkpoint"],
    }), flush=True)


if __name__ == "__main__":
    main()
