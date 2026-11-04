#!/usr/bin/env python3
"""Run the frozen 200-update RN1 role-native C208 Sim preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import torch
from torch.utils.data import DataLoader, Subset

import build_sim_p1_role_teacher_cache as role_cache_builder
import c208_role_native_objectives as rn1
import train_c208_real_sim_rehearsal as rehearsal
import train_c208_sim_hgrad_pretrain as shg
import train_c208_sim_low_lr_refine as s4k
from models.ensemble_unet2d import EnsembleUNet2d


METHOD = "c208_rn1_role_native_sim_preflight"

CONTROL_UPDATES = 200
TERMINAL_UPDATE = 10_000
ANCHOR_SIZE = 128
AUDIT_UPDATES = rn1.AUDIT_UPDATES
ROLE_SCALE = rn1.ROLE_SCALE
ROLE_MEMBER_GROUPS = rn1.ROLE_MEMBER_GROUPS

BASELINE_SIM_SHA256 = rehearsal.FORMAL_START_SIM_SHA256
BASELINE_SUMMARY_SHA256 = shg.BASELINE_SUMMARY_SHA256
BASELINE_SPLIT_MANIFEST_SHA256 = s4k.BASELINE_SPLIT_MANIFEST_SHA256
AGGREGATE_TEACHER_CACHE_SHA256 = rehearsal.FORMAL_P1_CACHE_SHA256
REAL_STATS_SHA256 = rehearsal.FORMAL_REAL_STATS_SHA256
CONTROL_CHECKPOINT_SHA256 = (
    "cfa5332f1755e5ae6c8e6784a4f8f017016c8c2194e478d7df2f15015a94f5ab")
CONTROL_TRAINING_CONFIG_SHA256 = (
    "d4d80e02fdc0c70fbd0b96bc9cb30be9cddf522669640d22248dc3d59bed2ae7")
INITIAL_MODEL_STATE_SHA256 = (
    "16f3341b5352ec42ea0df326e1e797bcc088c551a3f76834bd34bba4ca8a70b1")
TRAINING_READY_MODEL_STATE_SHA256 = (
    "1f5292c972669fbd5ab40b53dcd0932f3c3c38cefe1bf9387e0f1391d9f2d113")
MODEL_CONFIG_SHA256 = s4k.BASELINE_MODEL_CONFIG_SHA256
NORMALIZER_SHA256 = s4k.BASELINE_NORMALIZER_SHA256
ROLE_TEACHER_CHECKPOINT_SHA256 = {
    "a": "e5498ba59db11287a7ee480c69d9a114ef09fe1ef431a378ca94e1662fb1d791",
    "b": "83e9f6364272360e2026987add03ac443238a933cb7dd674a99bf05245e37c9b",
    "c": "5d00dc6fef25a782b2ce9fe69d1c84b16b63e968024525c54d456378d781f9a3",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sim", type=Path, required=True)
    parser.add_argument("--baseline-sim-sha256", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary-sha256", required=True)
    parser.add_argument("--baseline-split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-split-manifest-sha256", required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint-sha256", required=True)
    parser.add_argument("--aggregate-teacher-cache", type=Path, required=True)
    parser.add_argument("--aggregate-teacher-cache-sha256", required=True)
    parser.add_argument("--role-teacher-cache", type=Path, required=True)
    parser.add_argument("--role-teacher-cache-sha256", required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--real-stats-sha256", required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers != 0:
        raise ValueError("formal RN1 freezes num-workers=0")
    frozen = {
        "baseline_sim_sha256": BASELINE_SIM_SHA256,
        "baseline_summary_sha256": BASELINE_SUMMARY_SHA256,
        "baseline_split_manifest_sha256": BASELINE_SPLIT_MANIFEST_SHA256,
        "control_checkpoint_sha256": CONTROL_CHECKPOINT_SHA256,
        "aggregate_teacher_cache_sha256": AGGREGATE_TEACHER_CACHE_SHA256,
        "real_stats_sha256": REAL_STATS_SHA256,
    }
    for name, expected in frozen.items():
        value = rehearsal.normalized_sha256(
            getattr(args, name), name.replace("_", "-"))
        setattr(args, name, value)
        if value != expected:
            raise ValueError(f"{name} is not the frozen RN1 artifact")
    args.role_teacher_cache_sha256 = rehearsal.normalized_sha256(
        args.role_teacher_cache_sha256, "role-teacher-cache-sha256")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"formal RN1 output directory must be empty: {args.output_dir}")


def _exact_float(actual, expected: float) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12)


NORMALIZER_BUFFER_KEYS = (
    "persistence_scale", "persistence_bias", "target_std")


def expected_normalizer_buffers(normalizer) -> dict[str, torch.Tensor]:
    def value(name: str) -> torch.Tensor:
        source = (
            normalizer[name] if isinstance(normalizer, dict)
            else getattr(normalizer, name))
        return torch.as_tensor(source, dtype=torch.float32).cpu()

    mean_input = value("mean_input")
    mean_target = value("mean_target")
    std_input = value("std_input")
    std_target = value("std_target")
    if not all(tensor.numel() >= 3 for tensor in (
            mean_input, mean_target, std_input, std_target)):
        raise ValueError("RN1 normalizer must contain three target channels")
    safe_target = std_target[:3].clamp_min(1e-8)
    return {
        "persistence_scale": std_input[:3] / safe_target,
        "persistence_bias": (
            mean_input[:3] - mean_target[:3]) / safe_target,
        "target_std": safe_target,
    }


def validate_normalizer_buffers(
        state: dict[str, torch.Tensor], normalizer) -> dict:
    expected = expected_normalizer_buffers(normalizer)
    actual = {}
    for key in NORMALIZER_BUFFER_KEYS:
        if key not in state:
            raise ValueError(f"RN1 model state misses normalizer buffer {key}")
        tensor = torch.as_tensor(state[key], dtype=torch.float32).detach().cpu()
        if not torch.equal(tensor, expected[key]):
            raise ValueError(f"RN1 model normalizer buffer differs at {key}")
        actual[key] = tensor.tolist()
    return {
        "keys": list(NORMALIZER_BUFFER_KEYS),
        "values": actual,
        "canonical_sha256": rehearsal.canonical_sha256(actual),
    }


def trainable_parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_training_ready_candidate(
        payload: dict, normalizer, device: torch.device,
        ) -> tuple[EnsembleUNet2d, dict]:
    """Reproduce the original trainer's constructor-to-training boundary."""
    rehearsal.ensemble_train.seed_everything(0)
    model = shg.build_model(payload, device, load_weights=False)
    raw_state_sha = shg.model_state_sha256(model)
    if raw_state_sha != INITIAL_MODEL_STATE_SHA256:
        raise RuntimeError(
            "RN1 raw scratch model state differs before normalizer buffers")
    raw_parameter_sha = trainable_parameter_sha256(model)
    raw_buffers = {
        key: torch.as_tensor(model.state_dict()[key]).detach().cpu().tolist()
        for key in NORMALIZER_BUFFER_KEYS}
    expected_raw = {
        "persistence_scale": [1.0, 1.0, 1.0],
        "persistence_bias": [0.0, 0.0, 0.0],
        "target_std": [1.0, 1.0, 1.0],
    }
    if raw_buffers != expected_raw:
        raise RuntimeError("RN1 raw scratch normalizer buffers differ")

    # This is the exact boundary used by train_ensemble_unet2d.main before
    # entering train(); the raw constructor hash is not a training initial hash.
    rehearsal.ensemble_train.common.set_model_stats(model, normalizer)
    ready_state_sha = shg.model_state_sha256(model)
    if ready_state_sha != TRAINING_READY_MODEL_STATE_SHA256:
        raise RuntimeError(
            "RN1 training-ready model state differs after normalizer buffers")
    ready_parameter_sha = trainable_parameter_sha256(model)
    if ready_parameter_sha != raw_parameter_sha:
        raise RuntimeError("set_model_stats changed trainable RN1 parameters")
    buffer_contract = validate_normalizer_buffers(model.state_dict(), normalizer)
    return model, {
        "raw_constructor_state_sha256": raw_state_sha,
        "training_ready_state_sha256": ready_state_sha,
        "raw_trainable_parameter_sha256": raw_parameter_sha,
        "training_ready_trainable_parameter_sha256": ready_parameter_sha,
        "trainable_parameters_unchanged": True,
        "state_boundary": (
            "seed0_constructor_then_set_model_stats_before_first_update"),
        "raw_normalizer_buffers": raw_buffers,
        "training_ready_normalizer_buffers": buffer_contract,
        "changed_state_keys": list(NORMALIZER_BUFFER_KEYS),
    }


def validate_frozen_training_config(training: dict) -> None:
    exact = {
        "phase": "sim",
        "updates": TERMINAL_UPDATE,
        "start_checkpoint": None,
        "trainable_scope": "all",
        "batch_size": 8,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "seed": 0,
        "num_workers": 0,
        "max_files": 0,
        "window_stride": 20,
        "input_noise_std": 0.0,
        "point_loss_mode": "official_accuracy",
        "mse_weight": 0.25,
        "mse_warmup_updates": 0,
        "mean_loss_weight": 0.25,
        "member_loss_weight": 1.0,
        "member_loss_mode": "point",
        "bootstrap_probability": 0.75,
        "quantile_loss_weight": 0.05,
        "soft_sps_loss_weight": 0.05,
        "soft_sps_tau_fraction": 0.05,
        "soft_sps_warmup_updates": 1500,
        "soft_sps_uq_only": False,
        "soft_sps_proxy_real_scale": True,
        "coverage_bce_weight": 0.0,
        "profile_loss_weight": 0.0,
        "distill_weight": rehearsal.FORMAL_P1_DISTILL_WEIGHT,
    }
    errors = []
    for key, expected in exact.items():
        actual = training.get(key)
        matches = (
            _exact_float(actual, expected)
            if isinstance(expected, float) else actual == expected)
        if not matches:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    accuracy = tuple(float(value) for value in training.get(
        "accuracy_weights", ()))
    if accuracy != rehearsal.SIM_ACCURACY_WEIGHTS:
        errors.append(
            f"accuracy_weights={accuracy!r}, expected "
            f"{rehearsal.SIM_ACCURACY_WEIGHTS!r}")
    if errors:
        raise ValueError("; ".join(errors))


def validate_control_payload(payload: dict, baseline_payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("RN1 control checkpoint must contain a mapping")
    if payload.get("phase") != "sim" or int(payload.get("iteration", -1)) \
            != CONTROL_UPDATES:
        raise ValueError("RN1 control must be the frozen Sim update-200 checkpoint")
    model_config = payload.get("model_config")
    training = payload.get("training_config")
    normalizer = payload.get("normalizer")
    state = payload.get("model_state_dict")
    if not all(isinstance(value, dict) for value in (
            model_config, training, normalizer, state)):
        raise ValueError("RN1 control checkpoint contract is incomplete")
    if not state:
        raise ValueError("RN1 control checkpoint model state is empty")
    identities = {
        "model_config_sha256": rehearsal.canonical_sha256(model_config),
        "training_config_sha256": rehearsal.canonical_sha256(training),
        "normalizer_sha256": rehearsal.canonical_sha256(normalizer),
    }
    expected = {
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "training_config_sha256": CONTROL_TRAINING_CONFIG_SHA256,
        "normalizer_sha256": NORMALIZER_SHA256,
    }
    for key, value in expected.items():
        if identities[key] != value:
            raise ValueError(f"RN1 control {key} differs from frozen contract")
    if model_config != baseline_payload.get("model_config"):
        raise ValueError("RN1 control/baseline model configs differ")
    if normalizer != baseline_payload.get("normalizer"):
        raise ValueError("RN1 control/baseline normalizers differ")
    normalizer_buffers = validate_normalizer_buffers(state, normalizer)
    validate_frozen_training_config(training)
    baseline_training = baseline_payload.get("training_config")
    if not isinstance(baseline_training, dict):
        raise ValueError("RN1 baseline training config is absent")
    drift = [
        key for key in shg.FROZEN_SIM_LOSS_KEYS
        if training.get(key) != baseline_training.get(key)]
    if drift:
        raise ValueError(
            f"RN1 control/baseline frozen training fields differ: {drift}")
    history = payload.get("history")
    if not isinstance(history, list) or not history \
            or int(history[-1].get("iteration", -1)) != CONTROL_UPDATES:
        raise ValueError("RN1 control history does not terminate at update 200")
    return {**identities, "normalizer_buffers": normalizer_buffers}


def load_control(args: argparse.Namespace, baseline_payload: dict):
    actual = shg.sha256(args.control_checkpoint)
    if actual != args.control_checkpoint_sha256 \
            or actual != CONTROL_CHECKPOINT_SHA256:
        raise RuntimeError("RN1 control checkpoint SHA-256 differs")
    payload = torch.load(
        args.control_checkpoint, map_location="cpu", weights_only=False)
    metadata = validate_control_payload(payload, baseline_payload)
    metadata.update({
        "path": str(args.control_checkpoint.resolve()),
        "sha256": actual,
        "phase": "sim",
        "iteration": CONTROL_UPDATES,
    })
    return payload, metadata


def validate_runtime_training_args(args: argparse.Namespace) -> None:
    validate_frozen_training_config(vars(args))
    if int(args.updates) != TERMINAL_UPDATE:
        raise ValueError("RN1 scheduler horizon must remain 10000")


def load_aggregate_cache(
        args: argparse.Namespace, manifest: list, dataset):
    actual = shg.sha256(args.aggregate_teacher_cache)
    if actual != AGGREGATE_TEACHER_CACHE_SHA256:
        raise RuntimeError("aggregate P1 cache SHA differs before deserialization")
    cache, metadata = rehearsal.load_and_validate_cache(
        args.aggregate_teacher_cache, manifest, dataset)
    if metadata["sha256"] != AGGREGATE_TEACHER_CACHE_SHA256:
        raise RuntimeError("validated aggregate P1 cache SHA differs")
    return cache, metadata


def load_role_cache(args: argparse.Namespace, manifest: list, dataset):
    actual = shg.sha256(args.role_teacher_cache)
    if actual != args.role_teacher_cache_sha256:
        raise RuntimeError("role teacher cache SHA differs before deserialization")
    tensors, metadata = role_cache_builder.load_and_validate_cache(
        args.role_teacher_cache, manifest, dataset,
        ROLE_TEACHER_CHECKPOINT_SHA256)
    if metadata["sha256"] != args.role_teacher_cache_sha256:
        raise RuntimeError("validated role teacher cache SHA differs")
    return tensors, metadata


def role_cache_consistency_audit(
        aggregate_cache: torch.Tensor,
        role_cache: dict[str, torch.Tensor], chunk_rows: int = 8) -> dict:
    if chunk_rows <= 0:
        raise ValueError("role cache audit chunk size must be positive")
    a = role_cache["a_fluctuation_uv_physical"]
    b = role_cache["b_fluctuation_uv_physical"]
    mean = role_cache["p1_temporal_mean_uv_physical"]
    if a.shape != aggregate_cache.shape or b.shape != aggregate_cache.shape:
        raise ValueError("role/aggregate cache prediction shapes differ")
    if mean.shape != (len(aggregate_cache), 1, *aggregate_cache.shape[2:]):
        raise ValueError("role cache temporal-mean shape differs")
    maxima = {
        "reconstruction_max_abs": 0.0,
        "temporal_mean_max_abs": 0.0,
        "a_fluctuation_mean_max_abs": 0.0,
        "b_fluctuation_mean_max_abs": 0.0,
        "reference_max_abs": 0.0,
    }
    absolute_sum = 0.0
    count = 0
    for start in range(0, len(aggregate_cache), chunk_rows):
        stop = min(start + chunk_rows, len(aggregate_cache))
        aggregate = aggregate_cache[start:stop].float()
        a_rows = a[start:stop].float()
        b_rows = b[start:stop].float()
        mean_rows = mean[start:stop].float()
        reconstructed = 0.5 * a_rows + 0.5 * b_rows + mean_rows
        difference = (reconstructed - aggregate).abs()
        temporal_difference = (
            mean_rows - aggregate.mean(dim=1, keepdim=True)).abs()
        values = {
            "reconstruction_max_abs": difference.max(),
            "temporal_mean_max_abs": temporal_difference.max(),
            "a_fluctuation_mean_max_abs": a_rows.mean(dim=1).abs().max(),
            "b_fluctuation_mean_max_abs": b_rows.mean(dim=1).abs().max(),
            "reference_max_abs": aggregate.abs().max(),
        }
        if not all(bool(torch.isfinite(value)) for value in values.values()):
            raise FloatingPointError("non-finite role cache consistency metric")
        for key, value in values.items():
            maxima[key] = max(maxima[key], float(value))
        absolute_sum += float(difference.sum(dtype=torch.float64))
        count += difference.numel()
    storage_bound = (
        8.0 * torch.finfo(torch.float16).eps
        * max(1.0, maxima["reference_max_abs"]))
    checks = {
        "stored_reconstruction_within_fp16_bound": (
            maxima["reconstruction_max_abs"] <= storage_bound),
        "stored_temporal_mean_within_fp16_bound": (
            maxima["temporal_mean_max_abs"] <= storage_bound),
        "stored_a_zero_mean_within_fp16_bound": (
            maxima["a_fluctuation_mean_max_abs"] <= storage_bound),
        "stored_b_zero_mean_within_fp16_bound": (
            maxima["b_fluctuation_mean_max_abs"] <= storage_bound),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"role cache failed stored-FP16 consistency audit: {checks}")
    return {
        **maxima,
        "reconstruction_mean_abs": absolute_sum / count,
        "derived_fp16_storage_bound": storage_bound,
        "checks": checks,
        "pass": True,
    }


def aggregate_distillation_loss(
        prediction: torch.Tensor, teacher: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, float]:
    control, components = (
        rehearsal.ensemble_train.component_point_distillation_loss(
            prediction, teacher, normalizer))
    aggregate_fluctuation = rn1.exact_zero_mean_fluctuation_loss(
        prediction, teacher, normalizer)
    error = rn1.validate_exact_component_replacement(
        aggregate_fluctuation,
        components["distill_zero_mean_fluctuation"])
    return control, components, aggregate_fluctuation, error


def role_native_distillation_loss(
        prediction: torch.Tensor, members: torch.Tensor,
        aggregate_teacher: torch.Tensor,
        role_batch: dict[str, torch.Tensor], normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor],
                   torch.Tensor, float]:
    control, components, aggregate_fluctuation, error = (
        aggregate_distillation_loss(prediction, aggregate_teacher, normalizer))
    roles = rn1.role_distillation_losses(members, role_batch, normalizer)
    role_native = rn1.compose_role_native_distillation(
        control, aggregate_fluctuation, roles)
    return role_native, components, roles, control, error


@torch.no_grad()
def evaluate_anchor(
        model: EnsembleUNet2d, dataset, indices: list[int], normalizer,
        soft_sps_normalizer, aggregate_cache: torch.Tensor,
        role_cache: dict[str, torch.Tensor], args: argparse.Namespace,
        device: torch.device) -> dict[str, float]:
    point_loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0))
    metrics = rehearsal.ensemble_train.common.repro.evaluate(
        model, point_loader, shg.SimEvaluationNormalizer(normalizer), device)
    rehearsal.ensemble_train.common.add_weighted(
        metrics, tuple(float(value) for value in args.accuracy_weights))
    indexed_loader = DataLoader(
        shg.IndexedDataset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0))
    sums: dict[str, float] = {}
    count = 0
    model.eval()
    for history, target, sample_indices in indexed_loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        prediction, members, _quantiles, lower, upper, _features, _coverage = (
            model.forward_uq_training(history_norm))
        aggregate = aggregate_cache[sample_indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        roles = {
            key: value[sample_indices].to(
                device=device, dtype=torch.float32, non_blocking=True)
            for key, value in role_cache.items()
        }
        role_native, components, role_values, control, error = (
            role_native_distillation_loss(
                prediction, members, aggregate, roles, normalizer))
        _soft_loss, soft_score = (
            rehearsal.ensemble_train.soft_native_sps_objective(
                prediction, target_norm, lower, upper,
                soft_sps_normalizer, args.soft_sps_tau_fraction,
                metric_normalizer=normalizer))
        batch = history.shape[0]
        values = {
            "aggregate_p1_distill_loss": float(control),
            "role_native_distill_loss": float(role_native),
            "a_role_loss": float(role_values["a"]),
            "b_role_loss": float(role_values["b"]),
            "role_loss": float(role_values["role"]),
            "swapped_role_loss": float(role_values["swapped"]),
            "specialization_margin": float(role_values["margin"]),
            "native_soft_sps_proxy": float(soft_score),
            "aggregate_fluctuation_recompute_error": error,
            **{key: float(value) for key, value in components.items()},
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + batch * value
        count += batch
    metrics.update({key: value / count for key, value in sums.items()})
    result = {key: float(value) for key, value in metrics.items()}
    if not result or not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite RN1 anchor metric")
    return result


def make_optimizer(model: EnsembleUNet2d, args: argparse.Namespace):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TERMINAL_UPDATE)
    return optimizer, scheduler


def sim_update(
        model: EnsembleUNet2d, batch, normalizer, soft_sps_normalizer,
        aggregate_cache: torch.Tensor,
        role_cache: dict[str, torch.Tensor], args: argparse.Namespace,
        device: torch.device, iteration: int, optimizer, scheduler,
        ) -> tuple[dict[str, float], dict | None, str, list[int]]:
    model.train()
    history, target, trajectory_ids, sample_indices = batch
    key = shg.batch_key(trajectory_ids, sample_indices)
    natural_indices = [int(value) for value in sample_indices.tolist()]
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    history_norm, target_norm = normalizer.preprocess(history, target)
    optimizer.zero_grad(set_to_none=True)
    (prediction, members, quantiles, lower, upper, features,
     coverage_logits) = model.forward_uq_training(history_norm)
    weights = tuple(float(value) for value in args.accuracy_weights)
    point_loss, point_components = rehearsal.ensemble_train.point_loss(
        args, prediction, target_norm, normalizer, weights, iteration)
    member_loss, bootstrap_fraction = (
        rehearsal.ensemble_train.bootstrap_member_point_loss(
            args, members, target_norm, trajectory_ids, normalizer,
            weights, iteration))
    quantile_loss = rehearsal.ensemble_train.residual_quantile_loss(
        model, prediction, quantiles, target_norm, normalizer)
    soft_sps_loss, soft_sps_score = (
        rehearsal.ensemble_train.soft_native_sps_objective(
            prediction, target_norm, lower, upper, soft_sps_normalizer,
            args.soft_sps_tau_fraction, metric_normalizer=normalizer))
    coverage_bce = (
        rehearsal.ensemble_train.coverage_calibration_objective(
            model, prediction, quantiles, coverage_logits,
            target_norm, normalizer)
        if coverage_logits is not None else prediction.new_zeros(()))
    profile_loss = (
        rehearsal.ensemble_train.mean_profile_objective(
            prediction, target_norm, normalizer)
        if args.profile_loss_weight else prediction.new_zeros(()))
    base_loss = (
        args.mean_loss_weight * point_loss
        + args.member_loss_weight * member_loss
        + args.quantile_loss_weight * quantile_loss
        + args.coverage_bce_weight * coverage_bce
        + args.profile_loss_weight * profile_loss)
    aggregate = aggregate_cache[sample_indices].to(
        device=device, dtype=torch.float32, non_blocking=True)
    roles = {
        name: value[sample_indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        for name, value in role_cache.items()
    }
    control, distillation_components, aggregate_fluctuation, error = (
        aggregate_distillation_loss(prediction, aggregate, normalizer))
    role_values = rn1.role_distillation_losses(members, roles, normalizer)
    role_native = rn1.compose_role_native_distillation(
        control, aggregate_fluctuation, role_values)
    supervised = base_loss + args.distill_weight * role_native
    soft_sps_weight = args.soft_sps_loss_weight * min(
        1.0, iteration / max(args.soft_sps_warmup_updates, 1))
    total = supervised + soft_sps_weight * soft_sps_loss
    if not bool(torch.isfinite(total)):
        raise FloatingPointError(f"non-finite RN1 loss at update {iteration}")
    audit = None
    if iteration in AUDIT_UPDATES:
        audit = rn1.gradient_audit(
            iteration, members, features, aggregate_fluctuation,
            role_values, model.parameters(), args.distill_weight)
    total.backward()
    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()
    values = {
        "iteration": iteration,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "loss": float(total.detach()),
        "supervised_loss": float(supervised.detach()),
        "point_loss": float(point_loss.detach()),
        "member_loss": float(member_loss.detach()),
        "quantile_loss": float(quantile_loss.detach()),
        "soft_sps_loss": float(soft_sps_loss.detach()),
        "soft_sps_score": float(soft_sps_score.detach()),
        "soft_sps_weight": float(soft_sps_weight),
        "aggregate_p1_distill_loss": float(control.detach()),
        "role_native_distill_loss": float(role_native.detach()),
        "a_role_loss": float(role_values["a"].detach()),
        "b_role_loss": float(role_values["b"].detach()),
        "role_loss": float(role_values["role"].detach()),
        "swapped_role_loss": float(role_values["swapped"].detach()),
        "specialization_margin": float(role_values["margin"].detach()),
        "aggregate_fluctuation_recompute_error": error,
        "bootstrap_fraction": float(bootstrap_fraction.detach()),
        **{key: float(value.detach())
           for key, value in point_components.items()},
        **{key: float(value.detach())
           for key, value in distillation_components.items()},
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError(f"non-finite RN1 diagnostic at update {iteration}")
    return values, audit, key, natural_indices


def run_experiment(
        args: argparse.Namespace, baseline_payload: dict,
        baseline_metadata: dict, baseline_summary_metadata: dict,
        control_payload: dict, control_metadata: dict,
        sim_args: argparse.Namespace, raw_loader, normalizer,
        soft_sps_normalizer, aggregate_cache: torch.Tensor,
        aggregate_metadata: dict, role_cache: dict[str, torch.Tensor],
        role_metadata: dict, consistency_audit: dict,
        normalizer_metadata: dict, manifest_metadata: dict,
        device: torch.device) -> dict:
    started = time.time()
    dataset = raw_loader.dataset
    prefix = s4k.derive_prefix_contract(dataset, sim_args)
    print(json.dumps({
        "event": "c208_rn1_prefix_contract",
        "canonical_sha256": prefix["canonical_sha256"],
        "batch_keys_sha256": prefix["batch_keys_sha256"],
        "seen_indices_sha256": prefix["seen_indices_sha256"],
        "unseen_indices_sha256": prefix["unseen_indices_sha256"],
    }), flush=True)

    control_model = shg.build_model(control_payload, device, load_weights=True)
    rehearsal.ensemble_train.common.set_model_stats(control_model, normalizer)
    control_state_sha = shg.model_state_sha256(control_model)
    seen_control = evaluate_anchor(
        control_model, dataset, prefix["seen_indices"], normalizer,
        soft_sps_normalizer, aggregate_cache, role_cache, sim_args, device)
    unseen_control = evaluate_anchor(
        control_model, dataset, prefix["unseen_indices"], normalizer,
        soft_sps_normalizer, aggregate_cache, role_cache, sim_args, device)
    del control_model, control_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidate, initialization = build_training_ready_candidate(
        baseline_payload, normalizer, device)
    seen_initial = evaluate_anchor(
        candidate, dataset, prefix["seen_indices"], normalizer,
        soft_sps_normalizer, aggregate_cache, role_cache, sim_args, device)
    unseen_initial = evaluate_anchor(
        candidate, dataset, prefix["unseen_indices"], normalizer,
        soft_sps_normalizer, aggregate_cache, role_cache, sim_args, device)

    training_loader = rehearsal.ensemble_train.tag_training_loader(
        raw_loader, sim_args)
    iterator = rehearsal.ensemble_train.common.repro.cycling(training_loader)
    optimizer, scheduler = make_optimizer(candidate, sim_args)
    audits = []
    batch_keys = []
    last_values = None
    for iteration in range(1, CONTROL_UPDATES + 1):
        last_values, audit, key, natural_indices = sim_update(
            candidate, next(iterator), normalizer, soft_sps_normalizer,
            aggregate_cache, role_cache, sim_args, device, iteration,
            optimizer, scheduler)
        if natural_indices != prefix["batch_sample_indices"][iteration - 1] \
                or key != prefix["batch_keys"][iteration - 1]:
            raise RuntimeError(
                f"actual RN1 batch differs at update {iteration}")
        batch_keys.append(key)
        if audit is not None:
            audits.append(audit)
            print(json.dumps({
                "event": "c208_rn1_gradient_audit", **audit,
            }), flush=True)

    terminal_state_sha = shg.model_state_sha256(candidate)
    seen_candidate = evaluate_anchor(
        candidate, dataset, prefix["seen_indices"], normalizer,
        soft_sps_normalizer, aggregate_cache, role_cache, sim_args, device)
    unseen_candidate = evaluate_anchor(
        candidate, dataset, prefix["unseen_indices"], normalizer,
        soft_sps_normalizer, aggregate_cache, role_cache, sim_args, device)
    batch_identity = (
        batch_keys == prefix["batch_keys"]
        and rehearsal.canonical_sha256(batch_keys)
        == prefix["batch_keys_sha256"])
    prefix_payload = dict(prefix)
    recorded_prefix_sha = prefix_payload.pop("canonical_sha256")
    prefix_identity = all((
        recorded_prefix_sha == rehearsal.canonical_sha256(prefix_payload),
        prefix["updates"] == CONTROL_UPDATES,
        prefix["batch_size"] == int(sim_args.batch_size),
        prefix["batch_sample_indices_sha256"]
        == rehearsal.canonical_sha256(prefix["batch_sample_indices"]),
        prefix["seen_indices_sha256"]
        == rehearsal.canonical_sha256(prefix["seen_indices"]),
        prefix["unseen_indices_sha256"]
        == rehearsal.canonical_sha256(prefix["unseen_indices"]),
        manifest_metadata["sha256"] == BASELINE_SPLIT_MANIFEST_SHA256,
    ))
    gate = rn1.prefix_gate(
        audits, seen_control, seen_candidate,
        unseen_control, unseen_candidate,
        batch_identity=batch_identity, prefix_identity=prefix_identity)
    print(json.dumps({
        "event": "c208_rn1_prefix_gate", **gate,
    }), flush=True)

    frozen_loss = {
        key: baseline_metadata["training_config"][key]
        for key in shg.FROZEN_SIM_LOSS_KEYS}
    frozen_loss["scheduler_t_max"] = TERMINAL_UPDATE
    return {
        "method": METHOD,
        "status": gate["status"],
        "completed_updates": CONTROL_UPDATES,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "checkpoints_saved": [],
        "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
        "model_config": baseline_payload["model_config"],
        "baseline_sim": {
            **{key: baseline_metadata[key] for key in (
                "path", "sha256", "model_config_sha256",
                "training_config_sha256", "normalizer_sha256")},
            "source_summary": baseline_summary_metadata,
        },
        "control_checkpoint": {
            **control_metadata,
            "model_state_sha256": control_state_sha,
            "seen_anchor": seen_control,
            "unseen_anchor": unseen_control,
        },
        "aggregate_teacher_cache": aggregate_metadata,
        "role_teacher_cache": role_metadata,
        "role_cache_consistency_audit": consistency_audit,
        "sim_manifest": manifest_metadata,
        "sim_normalizer": normalizer_metadata,
        "prefix_contract": prefix,
        "paired_candidate": {
            "initial_model_state_sha256": initialization[
                "raw_constructor_state_sha256"],
            "u0_evaluation_model_state_sha256": initialization[
                "training_ready_state_sha256"],
            "initialization_boundaries": initialization,
            "terminal_model_state_sha256": terminal_state_sha,
            "initial_seen_anchor": seen_initial,
            "initial_unseen_anchor": unseen_initial,
            "terminal_seen_anchor": seen_candidate,
            "terminal_unseen_anchor": unseen_candidate,
            "batch_sequence_sha256": rehearsal.canonical_sha256(batch_keys),
            "terminal_train_values": last_values,
        },
        "gradient_audits": audits,
        "prefix_gate": gate,
        "training_contract": {
            "only_scientific_delta": (
                "replace_aggregate_zero_mean_fluctuation_component_with_"
                "members01_to_A_and_members23_to_B_role_loss"),
            "seed": 0,
            "batch_size": 8,
            "optimizer": "AdamW",
            "lr": 3e-4,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": TERMINAL_UPDATE,
            "updates": CONTROL_UPDATES,
            "same_first_200_batches_as_control": True,
            "constructor_state_sha256": INITIAL_MODEL_STATE_SHA256,
            "training_ready_state_sha256": (
                TRAINING_READY_MODEL_STATE_SHA256),
            "initialization_boundary_order": (
                "seed0_build_then_set_model_stats_then_u0_eval_and_update1"),
            "future_sim10000_restart_must_match_both_boundaries": True,
            "role_scale": ROLE_SCALE,
            "role_member_groups": ROLE_MEMBER_GROUPS,
            "role_loss_formula": "0.5*(L_A_members01+L_B_members23)",
            "distill_formula": "D_control-L_aggregate_fluctuation+L_role",
            "distill_weight": rehearsal.FORMAL_P1_DISTILL_WEIGHT,
            "frozen_sim_loss": frozen_loss,
            "audit_updates": list(AUDIT_UPDATES),
            "cosine_gate": None,
            "checkpoint_policy": "never_save_in_preflight",
            "fixed725_windows_used": 0,
        },
        "evidence_class": "sim_train_only_formal_prefix_preflight",
        "wall_seconds": time.time() - started,
    }


def write_summary(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal C208-RN1 preflight requires CUDA")
    actual_files = {
        "baseline_split_manifest": shg.sha256(args.baseline_split_manifest),
        "real_stats": shg.sha256(args.real_stats),
    }
    if actual_files["baseline_split_manifest"] \
            != BASELINE_SPLIT_MANIFEST_SHA256:
        raise RuntimeError("baseline Sim split manifest SHA differs")
    if actual_files["real_stats"] != REAL_STATS_SHA256:
        raise RuntimeError("Real stats SHA differs from frozen SPS proxy scale")

    baseline_payload, baseline_metadata = shg.load_baseline(args)
    s4k.validate_baseline_contract(baseline_payload, baseline_metadata)
    baseline_summary_metadata = shg.load_baseline_summary(
        args, baseline_payload, baseline_metadata)
    s4k.validate_baseline_summary_contract(baseline_summary_metadata)
    control_payload, control_metadata = load_control(args, baseline_payload)

    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.teacher_cache = args.aggregate_teacher_cache
    sim_args = shg.training_args(
        runtime_args, baseline_metadata["training_config"])
    validate_runtime_training_args(sim_args)
    rehearsal.ensemble_train.seed_everything(0)
    torch.set_float32_matmul_precision("high")
    with tempfile.TemporaryDirectory(prefix="c208-rn1-manifest-") as temp:
        prepare_args = argparse.Namespace(**vars(sim_args))
        prepare_args.output_dir = Path(temp)
        raw_loader, normalizer, train_windows, trajectories = (
            rehearsal.ensemble_train.common.prepare_sim(prepare_args, device))
        runtime_manifest_path = Path(temp) / "split_manifest.json"
        runtime_manifest_sha = shg.sha256(runtime_manifest_path)
        manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    if train_windows != s4k.EXPECTED_SIM_WINDOWS \
            or trajectories != s4k.EXPECTED_SIM_CONDITIONS:
        raise RuntimeError("runtime Sim dataset shape differs from frozen RN1")
    if runtime_manifest_sha != BASELINE_SPLIT_MANIFEST_SHA256:
        raise RuntimeError("runtime Sim split manifest differs from baseline")
    baseline_manifest = json.loads(
        args.baseline_split_manifest.read_text(encoding="utf-8"))
    if manifest != baseline_manifest:
        raise RuntimeError("runtime/baseline Sim manifests differ structurally")
    manifest_metadata = {
        "baseline_path": str(args.baseline_split_manifest.resolve()),
        "sha256": runtime_manifest_sha,
        "canonical_sha256": rehearsal.canonical_sha256(manifest),
        "windows": train_windows,
        "conditions": trajectories,
    }
    normalizer_metadata = shg.validate_runtime_normalizer(
        baseline_payload, normalizer)
    if normalizer_metadata["baseline_sha256"] != NORMALIZER_SHA256 \
            or normalizer_metadata["runtime_sha256"] != NORMALIZER_SHA256:
        raise RuntimeError("runtime Sim normalizer differs from frozen RN1")
    aggregate_cache, aggregate_metadata = load_aggregate_cache(
        args, manifest, raw_loader.dataset)
    shg.validate_teacher_summary_binding(
        aggregate_metadata, baseline_summary_metadata)
    role_cache, role_metadata = load_role_cache(
        args, manifest, raw_loader.dataset)
    consistency_audit = role_cache_consistency_audit(
        aggregate_cache, role_cache)
    soft_sps_normalizer = rehearsal.ensemble_train.common.repro.Normalizer(
        args.real_stats, device)
    print(json.dumps({
        "event": "c208_rn1_contract",
        "method": METHOD,
        "baseline_sim_sha256": BASELINE_SIM_SHA256,
        "baseline_summary_sha256": BASELINE_SUMMARY_SHA256,
        "baseline_split_manifest_sha256": BASELINE_SPLIT_MANIFEST_SHA256,
        "control_checkpoint_sha256": CONTROL_CHECKPOINT_SHA256,
        "aggregate_teacher_cache_sha256": AGGREGATE_TEACHER_CACHE_SHA256,
        "role_teacher_cache_sha256": args.role_teacher_cache_sha256,
        "real_stats_sha256": REAL_STATS_SHA256,
        "role_cache_consistency": consistency_audit,
    }), flush=True)
    summary = run_experiment(
        args, baseline_payload, baseline_metadata, baseline_summary_metadata,
        control_payload, control_metadata, sim_args, raw_loader, normalizer,
        soft_sps_normalizer, aggregate_cache, aggregate_metadata,
        role_cache, role_metadata, consistency_audit,
        normalizer_metadata, manifest_metadata, device)
    write_summary(args.output_dir, summary)
    print(json.dumps({
        "event": "c208_rn1_complete",
        "status": summary["status"],
        "checkpoint": None,
        "checkpoints_saved": [],
    }), flush=True)


if __name__ == "__main__":
    main()
