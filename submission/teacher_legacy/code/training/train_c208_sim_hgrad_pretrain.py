#!/usr/bin/env python3
"""Run the single calibrated C208-SHG1b Sim pretraining experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import train_c208_real_hgrad as hgrad
import train_c208_real_sim_rehearsal as rehearsal
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig


METHOD = "c208_shg1b_sim_hgrad_pretrain"
PASS_STATUS = "PROMOTE_SHG1B_SIM_MODEL_10000"
PREFLIGHT_PASS_STATUS = "GO_SHG1B_TO_SIM10000"
PREFLIGHT_STOP_STATUS = "STOP_SHG1B_SIM_PREFLIGHT"
SIM_GATE_STOP_STATUS = "STOP_SHG1B_SIM10000_GATE"
BASELINE_SIM_SHA256 = rehearsal.FORMAL_START_SIM_SHA256
BASELINE_SUMMARY_SHA256 = (
    "9b57d155d914bbd3e152dbb5d82c7d1fa1b1d0b654c55cab41bb9df85e4296ef")
TEACHER_CACHE_SHA256 = rehearsal.FORMAL_P1_CACHE_SHA256
REAL_STATS_SHA256 = rehearsal.FORMAL_REAL_STATS_SHA256
CONTROL_UPDATES = 200
TERMINAL_UPDATE = 10_000
AUDIT_UPDATES = hgrad.PREFLIGHT_AUDIT_UPDATES
ANCHOR_SIZE = 128
HGRAD_WEIGHT = 0.020
HGRAD_RAMP_UPDATES = 200
CALIBRATION_SOURCE_SUMMARY_SHA256 = (
    "04fddf7fbf35820bc8e4a83aebd6483162119f9e8b26071ecf75cce6fbf3a7d3")

PREFLIGHT_HGRAD_SELF_DECLINE = 0.10
PREFLIGHT_HGRAD_CONTROL_ADVANTAGE = 0.05
PREFLIGHT_P1_RATIO_MAX = 1.01
PREFLIGHT_POINT_DELTA_FLOOR = -0.05
PREFLIGHT_COMPONENT_DELTA_FLOOR = -0.10
PREFLIGHT_SOFT_SPS_DELTA_FLOOR = -0.005
SIM_HGRAD_BASELINE_ADVANTAGE = 0.05
SIM_P1_RATIO_MAX = 1.01
SIM_POINT_DELTA_MIN = 0.03
SIM_COMPONENT_DELTA_FLOOR = -0.05
SIM_SOFT_SPS_DELTA_FLOOR = -0.005

POINT_COMPONENT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")
FROZEN_SIM_LOSS_KEYS = (
    "batch_size", "lr", "weight_decay", "grad_clip", "seed",
    "max_files", "window_stride", "input_noise_std", "point_loss_mode",
    "mse_weight", "mse_warmup_updates", "mean_loss_weight",
    "member_loss_weight", "member_loss_mode", "bootstrap_probability",
    "quantile_loss_weight", "soft_sps_loss_weight",
    "soft_sps_tau_fraction", "soft_sps_warmup_updates",
    "soft_sps_uq_only", "soft_sps_proxy_real_scale",
    "coverage_bce_weight", "profile_loss_weight", "accuracy_weights",
    "distill_weight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sim", type=Path, required=True)
    parser.add_argument("--baseline-sim-sha256", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary-sha256", required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers != 0:
        raise ValueError("formal SHG1 freezes num-workers=0")
    args.baseline_sim_sha256 = rehearsal.normalized_sha256(
        args.baseline_sim_sha256, "baseline-sim-sha256")
    args.teacher_cache_sha256 = rehearsal.normalized_sha256(
        args.teacher_cache_sha256, "teacher-cache-sha256")
    args.baseline_summary_sha256 = rehearsal.normalized_sha256(
        args.baseline_summary_sha256, "baseline-summary-sha256")
    if args.baseline_sim_sha256 != BASELINE_SIM_SHA256:
        raise ValueError("baseline Sim SHA is not the frozen C208 Sim10k")
    if args.teacher_cache_sha256 != TEACHER_CACHE_SHA256:
        raise ValueError("teacher cache SHA is not the frozen current-P1 cache")
    if args.baseline_summary_sha256 != BASELINE_SUMMARY_SHA256:
        raise ValueError("baseline summary SHA is not the frozen Sim10k summary")
    protected = (
        "summary.json", "model_10000.pth", "hgrad_scale_contract.json")
    if any((args.output_dir / name).exists() for name in protected):
        raise FileExistsError(
            f"refusing to overwrite a completed SHG1 run in {args.output_dir}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def rng_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(numpy_state[1].tobytes())
    digest.update(repr(numpy_state[2:]).encode("ascii"))
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def _exact_float(actual, expected: float) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12)


def calibrated_hgrad_weight(iteration: int) -> float:
    return HGRAD_WEIGHT * min(1.0, iteration / HGRAD_RAMP_UPDATES)


def validate_baseline_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("baseline Sim checkpoint must contain a mapping")
    if payload.get("phase") != "sim" or int(payload.get("iteration", -1)) != 10_000:
        raise ValueError("baseline must be the frozen Sim update-10000 checkpoint")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("baseline Sim checkpoint has no model_config")
    for key, expected in rehearsal.EXPECTED_MODEL_CONFIG.items():
        if model_config.get(key) != expected:
            raise ValueError(
                f"baseline model_config[{key}]={model_config.get(key)!r}, "
                f"expected {expected!r}")
    training = payload.get("training_config")
    if not isinstance(training, dict):
        raise ValueError("baseline Sim checkpoint has no training_config")
    missing = [key for key in FROZEN_SIM_LOSS_KEYS if key not in training]
    if missing:
        raise ValueError(f"baseline Sim training_config misses {missing}")
    exact = {
        "phase": "sim", "updates": TERMINAL_UPDATE, "seed": 0,
        "start_checkpoint": None, "trainable_scope": "all",
        "max_files": 0, "window_stride": 20, "input_noise_std": 0.0,
        "point_loss_mode": "official_accuracy", "mse_warmup_updates": 0,
        "soft_sps_loss_weight": 0.05,
        "soft_sps_warmup_updates": 1500,
        "soft_sps_proxy_real_scale": True, "soft_sps_uq_only": False,
        "distill_weight": rehearsal.FORMAL_P1_DISTILL_WEIGHT,
    }
    errors = []
    for key, expected in exact.items():
        actual = training.get(key)
        matches = (_exact_float(actual, expected)
                   if isinstance(expected, float) else actual == expected)
        if not matches:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    accuracy = tuple(float(value) for value in training["accuracy_weights"])
    if accuracy != rehearsal.SIM_ACCURACY_WEIGHTS:
        errors.append(
            f"accuracy_weights={accuracy!r}, expected "
            f"{rehearsal.SIM_ACCURACY_WEIGHTS!r}")
    if not training.get("teacher_cache"):
        errors.append("teacher_cache is absent")
    numeric_positive = (
        "batch_size", "lr", "grad_clip", "mse_weight",
        "mean_loss_weight", "member_loss_weight", "quantile_loss_weight",
        "soft_sps_tau_fraction")
    for key in numeric_positive:
        value = training.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value <= 0:
            errors.append(f"{key} must be finite/positive, got {value!r}")
    numeric_nonnegative = (
        "weight_decay", "coverage_bce_weight", "profile_loss_weight")
    for key in numeric_nonnegative:
        value = training.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value < 0:
            errors.append(f"{key} must be finite/nonnegative, got {value!r}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "model_config_sha256": rehearsal.canonical_sha256(model_config),
        "training_config_sha256": rehearsal.canonical_sha256(training),
        "training_config": training,
    }


def load_baseline(args: argparse.Namespace) -> tuple[dict, dict]:
    actual = sha256(args.baseline_sim)
    if actual != args.baseline_sim_sha256 or actual != BASELINE_SIM_SHA256:
        raise RuntimeError(
            f"baseline Sim SHA-256 {actual} != frozen {BASELINE_SIM_SHA256}")
    payload = torch.load(
        args.baseline_sim, map_location="cpu", weights_only=False)
    metadata = validate_baseline_payload(payload)
    metadata.update({"path": str(args.baseline_sim.resolve()), "sha256": actual})
    return payload, metadata


def validate_baseline_summary_payload(
        summary: dict, checkpoint: dict, checkpoint_metadata: dict) -> dict:
    if not isinstance(summary, dict):
        raise TypeError("baseline Sim summary must contain a mapping")
    if summary.get("phase") != "sim" or int(summary.get("updates", -1)) != 10_000:
        raise ValueError("baseline summary is not the Sim update-10000 run")
    if int(summary.get("parameters", -1)) != rehearsal.EXPECTED_PARAMETER_COUNT:
        raise ValueError("baseline summary C208 parameter count differs")
    model_config = summary.get("model_config")
    if (not isinstance(model_config, dict)
            or rehearsal.canonical_sha256(model_config)
            != checkpoint_metadata["model_config_sha256"]):
        raise ValueError("baseline summary/checkpoint model_config differs")
    training = summary.get("training_config")
    if (not isinstance(training, dict)
            or rehearsal.canonical_sha256(training)
            != checkpoint_metadata["training_config_sha256"]):
        raise ValueError("baseline summary/checkpoint training_config differs")
    summary_history = summary.get("history")
    checkpoint_history = checkpoint.get("history")
    if (not isinstance(summary_history, list) or not summary_history
            or not isinstance(checkpoint_history, list)
            or not checkpoint_history):
        raise ValueError("baseline summary/checkpoint history is absent")
    if rehearsal.canonical_sha256(summary_history) != \
            rehearsal.canonical_sha256(checkpoint_history):
        raise ValueError("baseline summary/checkpoint history differs")
    point = summary.get("point_distillation")
    if not isinstance(point, dict):
        raise ValueError("baseline summary has no point_distillation provenance")
    if not _exact_float(
            point.get("distill_weight"), rehearsal.FORMAL_P1_DISTILL_WEIGHT):
        raise ValueError("baseline summary P1 distill weight differs")
    cache = point.get("teacher_cache")
    if not isinstance(cache, dict) or cache.get("sha256") != TEACHER_CACHE_SHA256:
        raise ValueError("baseline summary teacher cache SHA differs")
    if cache.get("method") != "sim_current_p1_point_teacher":
        raise ValueError("baseline summary teacher cache method differs")
    return {
        "model_config_sha256": rehearsal.canonical_sha256(model_config),
        "training_config_sha256": rehearsal.canonical_sha256(training),
        "history_sha256": rehearsal.canonical_sha256(summary_history),
        "teacher_cache_sha256": cache["sha256"],
        "teacher_cache_method": cache["method"],
        "distill_weight": float(point["distill_weight"]),
    }


def validate_teacher_summary_binding(
        teacher_metadata: dict, baseline_summary_metadata: dict) -> None:
    if not isinstance(teacher_metadata, dict):
        raise TypeError("validated teacher metadata must be a mapping")
    if teacher_metadata.get("sha256") != baseline_summary_metadata.get(
            "teacher_cache_sha256"):
        raise RuntimeError("current teacher cache differs from baseline summary")
    if teacher_metadata.get("method") != baseline_summary_metadata.get(
            "teacher_cache_method"):
        raise RuntimeError("current teacher method differs from baseline summary")


def load_baseline_summary(
        args: argparse.Namespace, checkpoint: dict,
        checkpoint_metadata: dict) -> dict:
    actual = sha256(args.baseline_summary)
    if actual != args.baseline_summary_sha256 or actual != BASELINE_SUMMARY_SHA256:
        raise RuntimeError(
            f"baseline summary SHA-256 {actual} != frozen "
            f"{BASELINE_SUMMARY_SHA256}")
    summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    metadata = validate_baseline_summary_payload(
        summary, checkpoint, checkpoint_metadata)
    metadata.update({
        "path": str(args.baseline_summary.resolve()), "sha256": actual})
    return metadata


def training_args(args: argparse.Namespace, training: dict) -> argparse.Namespace:
    values = dict(training)
    values.update({
        "phase": "sim", "updates": TERMINAL_UPDATE,
        "start_checkpoint": None, "output_dir": args.output_dir,
        "sim_dir": args.sim_dir, "real_stats": args.real_stats,
        "teacher_cache": args.teacher_cache, "device": args.device,
        "num_workers": args.num_workers,
        "baseline_sim": args.baseline_sim,
        "baseline_summary": args.baseline_summary,
    })
    return argparse.Namespace(**values)


def build_model(payload: dict, device: torch.device,
                load_weights: bool) -> EnsembleUNet2d:
    stored = dict(payload["model_config"])
    if stored.pop("architecture") != "ensemble_unet":
        raise ValueError("baseline architecture is not ensemble_unet")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    if load_weights:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != rehearsal.EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"C208 parameters={count}, expected {rehearsal.EXPECTED_PARAMETER_COUNT}")
    return model.to(device)


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = self.indices[index]
        history, target = self.dataset[source]
        return history, target, source


class SimEvaluationNormalizer:
    """Give the Sim normalizer the physical-output API used by evaluators."""

    def __init__(self, normalizer):
        self._normalizer = normalizer
        for key in rehearsal.NORMALIZER_KEYS:
            setattr(self, key, getattr(normalizer, key))

    def preprocess(self, history: torch.Tensor, target: torch.Tensor):
        return self._normalizer.preprocess(history, target)

    def postprocess_prediction(self, value: torch.Tensor) -> torch.Tensor:
        return rehearsal.ensemble_train.common.to_physical_target(
            value, self._normalizer)

    def postprocess_target(self, value: torch.Tensor) -> torch.Tensor:
        return rehearsal.ensemble_train.common.to_physical_target(
            value, self._normalizer)


def compute_sim_hgrad_scale_contract(
        dataset, split_sha256: str, normalizer_sha256: str) -> dict:
    sums = {key: torch.zeros(2, dtype=torch.float64) for key in ("dx", "dy")}
    counts = {"dx": 0, "dy": 0}
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0))
    for _history, target in loader:
        velocity = target[..., :2].to(dtype=torch.float64)
        if not bool(torch.isfinite(velocity).all()):
            raise FloatingPointError("non-finite Sim target in hgrad scale")
        dx, dy = hgrad.forward_differences(velocity)
        sums["dx"].add_(dx.square().sum(dim=(0, 1, 2, 3)))
        sums["dy"].add_(dy.square().sum(dim=(0, 1, 2, 3)))
        counts["dx"] += dx.numel() // 2
        counts["dy"] += dy.numel() // 2
    scales = {
        key: torch.sqrt(sums[key] / counts[key]).clamp_min(
            hgrad.HGRAD_SCALE_FLOOR).tolist()
        for key in ("dx", "dy")}
    contract = {
        "method": "sim_train_physical_uv_forward_difference_rms",
        "train_windows": len(dataset),
        "split_manifest_sha256": split_sha256,
        "sim_normalizer_sha256": normalizer_sha256,
        "scale_floor": hgrad.HGRAD_SCALE_FLOOR,
        "counts_per_channel": counts,
        "sum_squares": {key: sums[key].tolist() for key in sums},
        "scales": scales,
        "real_windows_used": 0,
        "fixed725_windows_used": 0,
    }
    contract["canonical_sha256"] = rehearsal.canonical_sha256(contract)
    return contract


def batch_key(trajectory_ids: torch.Tensor,
              sample_indices: torch.Tensor) -> str:
    return rehearsal.canonical_sha256({
        "trajectory_ids": [int(value) for value in trajectory_ids.tolist()],
        "sample_indices": [int(value) for value in sample_indices.tolist()],
    })


def require_paired_identity(control_state_sha256: str,
                            candidate_state_sha256: str,
                            control_rng_sha256: str,
                            candidate_rng_sha256: str) -> None:
    if control_state_sha256 != candidate_state_sha256:
        raise RuntimeError("paired control/candidate initial model states differ")
    if control_rng_sha256 != candidate_rng_sha256:
        raise RuntimeError("paired control/candidate initial RNG states differ")


def require_paired_batch(control_key: str, candidate_key: str,
                         iteration: int) -> None:
    if control_key != candidate_key:
        raise RuntimeError(
            f"paired control/candidate batch differs at update {iteration}")


def validate_runtime_normalizer(payload: dict, normalizer) -> dict:
    stored = payload.get("normalizer")
    if not isinstance(stored, dict):
        raise ValueError("baseline Sim checkpoint has no normalizer")
    runtime = rehearsal.ensemble_train.common.normalizer_dict(normalizer)
    errors = []
    for key in rehearsal.NORMALIZER_KEYS:
        if key not in stored:
            errors.append(f"baseline normalizer misses {key}")
            continue
        left = torch.as_tensor(stored[key], dtype=torch.float64)
        right = torch.as_tensor(runtime[key], dtype=torch.float64)
        if left.shape != right.shape or not torch.allclose(
                left, right, rtol=0.0, atol=1e-7):
            errors.append(f"runtime Sim normalizer differs at {key}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "baseline_sha256": rehearsal.canonical_sha256(stored),
        "runtime_sha256": rehearsal.canonical_sha256(runtime),
        "values": runtime,
    }


@torch.no_grad()
def evaluate_anchor(
        model: EnsembleUNet2d, dataset, indices: list[int], normalizer,
        soft_sps_normalizer, teacher_cache: torch.Tensor,
        scale_contract: dict, args: argparse.Namespace,
        device: torch.device) -> dict[str, float]:
    point_loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    metrics = rehearsal.ensemble_train.evaluate(
        model, point_loader, SimEvaluationNormalizer(normalizer), device,
        tuple(float(value) for value in args.accuracy_weights))
    indexed_loader = DataLoader(
        IndexedDataset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    sums: dict[str, float] = {}
    count = 0
    model.eval()
    for history, target, sample_indices in indexed_loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        prediction, _members, _quantiles, lower, upper, _features, _coverage = (
            model.forward_uq_training(history_norm))
        teacher = teacher_cache[sample_indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        distillation, components = (
            rehearsal.ensemble_train.component_point_distillation_loss(
                prediction, teacher, normalizer))
        spatial = hgrad.hgrad_loss(
            prediction, target_norm, normalizer, scale_contract)
        _soft_loss, soft_score = (
            rehearsal.ensemble_train.soft_native_sps_objective(
                prediction, target_norm, lower, upper,
                soft_sps_normalizer, args.soft_sps_tau_fraction,
                metric_normalizer=normalizer))
        batch = history.shape[0]
        values = {
            "p1_distill_loss": float(distillation),
            "hgrad_loss": float(spatial),
            "native_soft_sps_proxy": float(soft_score),
            **{key: float(value) for key, value in components.items()},
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + batch * value
        count += batch
    metrics.update({key: value / count for key, value in sums.items()})
    return {key: float(value) for key, value in metrics.items()}


def make_optimizer(model: EnsembleUNet2d, args: argparse.Namespace):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TERMINAL_UPDATE)
    return optimizer, scheduler


def sim_update(
        model: EnsembleUNet2d, batch, normalizer, soft_sps_normalizer,
        teacher_cache: torch.Tensor, scale_contract: dict,
        args: argparse.Namespace, device: torch.device, iteration: int,
        optimizer, scheduler, add_hgrad: bool,
        ) -> tuple[dict[str, float], dict | None, str]:
    model.train()
    history, target, trajectory_ids, sample_indices = batch
    key = batch_key(trajectory_ids, sample_indices)
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    history_norm, target_norm = normalizer.preprocess(history, target)
    optimizer.zero_grad(set_to_none=True)
    (prediction, members, quantiles, lower, upper, _features,
     coverage_logits) = model.forward_uq_training(history_norm)
    point_loss, point_components = rehearsal.ensemble_train.point_loss(
        args, prediction, target_norm, normalizer,
        tuple(float(value) for value in args.accuracy_weights), iteration)
    if args.member_loss_mode == "point":
        member_loss, bootstrap_fraction = (
            rehearsal.ensemble_train.bootstrap_member_point_loss(
                args, members, target_norm, trajectory_ids, normalizer,
                tuple(float(value) for value in args.accuracy_weights),
                iteration))
    else:
        member_loss, bootstrap_fraction = (
            rehearsal.ensemble_train.bootstrap_member_mse(
                members, target_norm, trajectory_ids,
                args.bootstrap_probability, args.seed))
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
    teacher = teacher_cache[sample_indices].to(
        device=device, dtype=torch.float32, non_blocking=True)
    distillation, distillation_components = (
        rehearsal.ensemble_train.component_point_distillation_loss(
            prediction, teacher, normalizer))
    supervised_native = base_loss + args.distill_weight * distillation
    soft_sps_weight = args.soft_sps_loss_weight * min(
        1.0, iteration / max(args.soft_sps_warmup_updates, 1))
    native_total = supervised_native + soft_sps_weight * soft_sps_loss
    spatial_loss = hgrad.hgrad_loss(
        prediction, target_norm, normalizer, scale_contract)
    spatial_weight = calibrated_hgrad_weight(iteration) if add_hgrad else 0.0
    weighted_hgrad = spatial_weight * spatial_loss
    total = native_total + weighted_hgrad
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite Sim loss at update {iteration}")
    audit = None
    if add_hgrad and iteration in AUDIT_UPDATES:
        point_path = (
            args.mean_loss_weight * point_loss
            + args.distill_weight * distillation)
        audit = hgrad.output_gradient_audit(
            point_path, weighted_hgrad, supervised_native, prediction)
        audit["iteration"] = iteration
    total.backward()
    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()
    values = {
        "loss": float(total.detach()),
        "native_total_loss": float(native_total.detach()),
        "supervised_native_loss": float(supervised_native.detach()),
        "point_loss": float(point_loss.detach()),
        "member_loss": float(member_loss.detach()),
        "quantile_loss": float(quantile_loss.detach()),
        "soft_sps_loss": float(soft_sps_loss.detach()),
        "soft_sps_score": float(soft_sps_score.detach()),
        "soft_sps_weight": soft_sps_weight,
        "p1_distill_loss": float(distillation.detach()),
        "hgrad_loss": float(spatial_loss.detach()),
        "hgrad_weight": spatial_weight,
        "weighted_hgrad_loss": float(weighted_hgrad.detach()),
        "bootstrap_fraction": float(bootstrap_fraction.detach()),
        **{key: float(value) for key, value in point_components.items()},
        **{key: float(value) for key, value in distillation_components.items()},
    }
    return values, audit, key


def _relative_at_most(value: float, reference: float, multiplier: float) -> bool:
    if reference <= 1e-12:
        return value <= reference + 1e-12
    return value <= multiplier * reference + 1e-12


def paired_preflight_gate(audits: list[dict], initial: dict,
                          control: dict, candidate: dict) -> dict:
    screen = hgrad.evaluate_preflight(audits)
    component_delta = {
        key: candidate[key] - control[key] for key in POINT_COMPONENT_KEYS}
    checks = {
        "gradient_screen": bool(screen["pass"]),
        "hgrad_self_decline": _relative_at_most(
            candidate["hgrad_loss"], initial["hgrad_loss"],
            1.0 - PREFLIGHT_HGRAD_SELF_DECLINE),
        "hgrad_control_advantage": _relative_at_most(
            candidate["hgrad_loss"], control["hgrad_loss"],
            1.0 - PREFLIGHT_HGRAD_CONTROL_ADVANTAGE),
        "p1_non_regression": _relative_at_most(
            candidate["p1_distill_loss"], control["p1_distill_loss"],
            PREFLIGHT_P1_RATIO_MAX),
        "point_non_regression": (
            candidate["point_score"] - control["point_score"]
            >= PREFLIGHT_POINT_DELTA_FLOOR - 1e-12),
        "components_non_regression": all(
            value >= PREFLIGHT_COMPONENT_DELTA_FLOOR - 1e-12
            for value in component_delta.values()),
        "soft_sps_non_regression": (
            candidate["native_soft_sps_proxy"]
            - control["native_soft_sps_proxy"]
            >= PREFLIGHT_SOFT_SPS_DELTA_FLOOR - 1e-12),
    }
    passed = all(checks.values())
    return {
        "status": PREFLIGHT_PASS_STATUS if passed else PREFLIGHT_STOP_STATUS,
        "pass": passed, "checks": checks, "gradient_screen": screen,
        "initial_anchor": initial, "control_anchor": control,
        "candidate_anchor": candidate,
        "component_delta": component_delta,
        "point_delta": candidate["point_score"] - control["point_score"],
        "soft_sps_proxy_delta": (
            candidate["native_soft_sps_proxy"]
            - control["native_soft_sps_proxy"]),
    }


def terminal_sim_gate(baseline: dict, candidate: dict) -> dict:
    component_delta = {
        key: candidate[key] - baseline[key] for key in POINT_COMPONENT_KEYS}
    checks = {
        "hgrad_advantage": _relative_at_most(
            candidate["hgrad_loss"], baseline["hgrad_loss"],
            1.0 - SIM_HGRAD_BASELINE_ADVANTAGE),
        "p1_non_regression": _relative_at_most(
            candidate["p1_distill_loss"], baseline["p1_distill_loss"],
            SIM_P1_RATIO_MAX),
        "point_signal": (
            candidate["point_score"] - baseline["point_score"]
            >= SIM_POINT_DELTA_MIN - 1e-12),
        "components_non_regression": all(
            value >= SIM_COMPONENT_DELTA_FLOOR - 1e-12
            for value in component_delta.values()),
        "soft_sps_non_regression": (
            candidate["native_soft_sps_proxy"]
            - baseline["native_soft_sps_proxy"]
            >= SIM_SOFT_SPS_DELTA_FLOOR - 1e-12),
    }
    passed = all(checks.values())
    return {
        "status": PASS_STATUS if passed else SIM_GATE_STOP_STATUS,
        "pass": passed, "checks": checks,
        "baseline_anchor": baseline, "candidate_anchor": candidate,
        "component_delta": component_delta,
        "point_delta": candidate["point_score"] - baseline["point_score"],
        "soft_sps_proxy_delta": (
            candidate["native_soft_sps_proxy"]
            - baseline["native_soft_sps_proxy"]),
    }


def save_terminal_checkpoint(
        gate: dict, output_dir: Path, model: EnsembleUNet2d,
        args: argparse.Namespace, history: list[dict], normalizer,
        ) -> tuple[str | None, str | None]:
    if not gate.get("pass"):
        return None, None
    args.method = METHOD
    args.baseline_sim_sha256 = BASELINE_SIM_SHA256
    args.baseline_summary_sha256 = BASELINE_SUMMARY_SHA256
    args.calibration_source_summary_sha256 = (
        CALIBRATION_SOURCE_SUMMARY_SHA256)
    args.teacher_cache_sha256 = TEACHER_CACHE_SHA256
    args.hgrad_weight = HGRAD_WEIGHT
    args.hgrad_ramp_updates = HGRAD_RAMP_UPDATES
    args.preflight_pass = True
    args.sim_gate_pass = True
    checkpoint = output_dir / "model_10000.pth"
    rehearsal.ensemble_train.common.save_checkpoint(
        checkpoint, model, args, TERMINAL_UPDATE, history,
        rehearsal.ensemble_train.common.normalizer_dict(normalizer), gate)
    return str(checkpoint), sha256(checkpoint)


def write_summary(output_dir: Path, summary: dict) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def run_experiment(
        args: argparse.Namespace, payload: dict, baseline_metadata: dict,
        baseline_summary_metadata: dict,
        sim_args: argparse.Namespace, raw_loader, normalizer,
        soft_sps_normalizer, teacher_cache: torch.Tensor,
        teacher_metadata: dict, scale_contract: dict,
        normalizer_metadata: dict, device: torch.device,
        ) -> dict:
    started = time.time()
    dataset = raw_loader.dataset
    anchor_indices = list(range(ANCHOR_SIZE))
    if len(dataset) < ANCHOR_SIZE:
        raise RuntimeError(
            f"Sim dataset has {len(dataset)} windows, needs {ANCHOR_SIZE}")

    baseline_model = build_model(payload, device, load_weights=True)
    rehearsal.ensemble_train.common.set_model_stats(baseline_model, normalizer)
    baseline_anchor = evaluate_anchor(
        baseline_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract, sim_args, device)
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rehearsal.ensemble_train.seed_everything(0)
    control_model = build_model(payload, device, load_weights=False)
    rehearsal.ensemble_train.common.set_model_stats(control_model, normalizer)
    control_state_sha = model_state_sha256(control_model)
    control_rng_sha = rng_fingerprint()
    control_initial_anchor = evaluate_anchor(
        control_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract, sim_args, device)
    control_loader = rehearsal.ensemble_train.tag_training_loader(
        raw_loader, sim_args)
    control_iterator = rehearsal.ensemble_train.common.repro.cycling(
        control_loader)
    control_optimizer, control_scheduler = make_optimizer(
        control_model, sim_args)
    control_batch_keys = []
    control_last = None
    for iteration in range(1, CONTROL_UPDATES + 1):
        control_last, _audit, key = sim_update(
            control_model, next(control_iterator), normalizer,
            soft_sps_normalizer, teacher_cache, scale_contract,
            sim_args, device, iteration, control_optimizer,
            control_scheduler, add_hgrad=False)
        control_batch_keys.append(key)
    control_anchor = evaluate_anchor(
        control_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract, sim_args, device)
    del (control_model, control_optimizer, control_scheduler,
         control_iterator, control_loader)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rehearsal.ensemble_train.seed_everything(0)
    candidate_model = build_model(payload, device, load_weights=False)
    rehearsal.ensemble_train.common.set_model_stats(candidate_model, normalizer)
    candidate_state_sha = model_state_sha256(candidate_model)
    candidate_rng_sha = rng_fingerprint()
    require_paired_identity(
        control_state_sha, candidate_state_sha,
        control_rng_sha, candidate_rng_sha)
    candidate_initial_anchor = evaluate_anchor(
        candidate_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract, sim_args, device)
    candidate_loader = rehearsal.ensemble_train.tag_training_loader(
        raw_loader, sim_args)
    candidate_iterator = rehearsal.ensemble_train.common.repro.cycling(
        candidate_loader)
    optimizer, scheduler = make_optimizer(candidate_model, sim_args)
    audits = []
    history = []
    candidate_batch_keys = []
    log_interval = int(sim_args.eval_interval)
    for iteration in range(1, CONTROL_UPDATES + 1):
        values, audit, key = sim_update(
            candidate_model, next(candidate_iterator), normalizer,
            soft_sps_normalizer, teacher_cache, scale_contract,
            sim_args, device, iteration, optimizer, scheduler,
            add_hgrad=True)
        require_paired_batch(control_batch_keys[iteration - 1], key, iteration)
        candidate_batch_keys.append(key)
        if audit is not None:
            audits.append(audit)
            print(json.dumps({
                "event": "c208_shg1b_audit", **audit,
            }), flush=True)
        if iteration == CONTROL_UPDATES:
            history.append({"iteration": iteration, **values})
    candidate_anchor_200 = evaluate_anchor(
        candidate_model, dataset, anchor_indices, normalizer,
        soft_sps_normalizer, teacher_cache, scale_contract, sim_args, device)
    preflight = paired_preflight_gate(
        audits, candidate_initial_anchor, control_anchor,
        candidate_anchor_200)
    print(json.dumps({
        "event": "c208_shg1b_preflight", **preflight,
    }), flush=True)

    sim_gate = None
    checkpoint = checkpoint_sha = None
    completed_updates = CONTROL_UPDATES
    if preflight["pass"]:
        for iteration in range(CONTROL_UPDATES + 1, TERMINAL_UPDATE + 1):
            values, audit, _key = sim_update(
                candidate_model, next(candidate_iterator), normalizer,
                soft_sps_normalizer, teacher_cache, scale_contract,
                sim_args, device, iteration, optimizer, scheduler,
                add_hgrad=True)
            if audit is not None:
                raise RuntimeError("SHG1b audit occurred after frozen preflight")
            if iteration % log_interval == 0 or iteration == TERMINAL_UPDATE:
                row = {"iteration": iteration, **values}
                history.append(row)
                print(json.dumps({
                    "event": "c208_shg1b_progress", **row,
                }), flush=True)
        completed_updates = TERMINAL_UPDATE
        candidate_anchor_10000 = evaluate_anchor(
            candidate_model, dataset, anchor_indices, normalizer,
            soft_sps_normalizer, teacher_cache, scale_contract,
            sim_args, device)
        sim_gate = terminal_sim_gate(baseline_anchor, candidate_anchor_10000)
        print(json.dumps({
            "event": "c208_shg1b_sim_gate", **sim_gate,
        }), flush=True)
        checkpoint, checkpoint_sha = save_terminal_checkpoint(
            sim_gate, args.output_dir, candidate_model, sim_args,
            history, normalizer)

    status = (
        preflight["status"] if not preflight["pass"]
        else sim_gate["status"])
    frozen_sim_loss = {
        key: baseline_metadata["training_config"][key]
        for key in FROZEN_SIM_LOSS_KEYS}
    frozen_sim_loss["scheduler_t_max"] = TERMINAL_UPDATE
    summary = {
        "method": METHOD,
        "status": status,
        "completed_updates": completed_updates,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "model_config": payload["model_config"],
        "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
        "baseline_sim": {
            **{key: baseline_metadata[key] for key in (
                "path", "sha256", "model_config_sha256",
                "training_config_sha256")},
            "anchor_metrics": baseline_anchor,
            "source_summary": baseline_summary_metadata,
        },
        "teacher_cache": teacher_metadata,
        "sim_normalizer": normalizer_metadata,
        "hgrad_scale_contract": scale_contract,
        "paired_control": {
            "updates": CONTROL_UPDATES,
            "initial_model_state_sha256": control_state_sha,
            "initial_rng_sha256": control_rng_sha,
            "initial_anchor": control_initial_anchor,
            "terminal_anchor": control_anchor,
            "terminal_train_values": control_last,
            "batch_sequence_sha256": rehearsal.canonical_sha256(
                control_batch_keys),
            "writes_artifacts": False,
        },
        "paired_candidate": {
            "initial_model_state_sha256": candidate_state_sha,
            "initial_rng_sha256": candidate_rng_sha,
            "initial_anchor": candidate_initial_anchor,
            "batch_sequence_sha256_update200": rehearsal.canonical_sha256(
                candidate_batch_keys),
        },
        "preflight": preflight,
        "sim_gate": sim_gate,
        "history": history,
        "training_contract": {
            "seed": 0,
            "control_updates": CONTROL_UPDATES,
            "terminal_update": TERMINAL_UPDATE,
            "anchor_indices": anchor_indices,
            "frozen_sim_loss": frozen_sim_loss,
            "only_scientific_delta": (
                "physical_uv_wh_forward_difference_rms_hgrad_"
                "through_complete_aggregate_point_graph"),
            "hgrad_weight": HGRAD_WEIGHT,
            "hgrad_ramp_updates": HGRAD_RAMP_UPDATES,
            "calibration_source_summary_sha256": (
                CALIBRATION_SOURCE_SUMMARY_SHA256),
            "audit_updates": list(AUDIT_UPDATES),
            "audit_loss_ratio_range": list(
                hgrad.PREFLIGHT_LOSS_RATIO_RANGE),
            "audit_output_gradient_ratio_range": list(
                hgrad.PREFLIGHT_GRAD_RATIO_RANGE),
            "audit_median_cosine_min": hgrad.PREFLIGHT_MEDIAN_COSINE_MIN,
            "audit_worst_cosine_min": hgrad.PREFLIGHT_WORST_COSINE_MIN,
            "soft_sps_gradient_scope": "full_graph_uq_only_false",
            "control_writes_artifacts": False,
            "checkpoint_policy": "only_after_terminal_sim_gate_pass",
        },
        "evidence_class": (
            "released_sim_train_only_preflight_and_anchor_diagnostic"),
        "wall_seconds": time.time() - started,
    }
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal C208-SHG1b training requires CUDA")
    if sha256(args.real_stats) != REAL_STATS_SHA256:
        raise RuntimeError("Real stats SHA differs from frozen SPS proxy scale")
    payload, baseline_metadata = load_baseline(args)
    baseline_summary_metadata = load_baseline_summary(
        args, payload, baseline_metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim_args = training_args(args, baseline_metadata["training_config"])
    rehearsal.ensemble_train.seed_everything(0)
    torch.set_float32_matmul_precision("high")
    raw_loader, normalizer, train_windows, trajectories = (
        rehearsal.ensemble_train.common.prepare_sim(sim_args, device))
    if trajectories != rehearsal.FORMAL_SIM_TRAJECTORIES:
        raise RuntimeError(
            f"Sim trajectories={trajectories}, expected "
            f"{rehearsal.FORMAL_SIM_TRAJECTORIES}")
    normalizer_metadata = validate_runtime_normalizer(payload, normalizer)
    manifest = json.loads(
        (args.output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    teacher_cache, teacher_metadata = rehearsal.load_and_validate_cache(
        args.teacher_cache, manifest, raw_loader.dataset)
    if teacher_metadata["sha256"] != TEACHER_CACHE_SHA256:
        raise RuntimeError("validated teacher cache SHA differs from frozen P1")
    validate_teacher_summary_binding(
        teacher_metadata, baseline_summary_metadata)
    soft_sps_normalizer = rehearsal.ensemble_train.common.repro.Normalizer(
        args.real_stats, device)
    scale_contract = compute_sim_hgrad_scale_contract(
        raw_loader.dataset, rehearsal.canonical_sha256(manifest),
        normalizer_metadata["runtime_sha256"])
    (args.output_dir / "hgrad_scale_contract.json").write_text(
        json.dumps(scale_contract, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "c208_shg1b_contract", "method": METHOD,
        "train_windows": train_windows, "trajectories": trajectories,
        "baseline_sim_sha256": BASELINE_SIM_SHA256,
        "baseline_summary_sha256": BASELINE_SUMMARY_SHA256,
        "teacher_cache_sha256": TEACHER_CACHE_SHA256,
        "training_config_sha256": baseline_metadata[
            "training_config_sha256"],
        "hgrad_weight": HGRAD_WEIGHT,
        "hgrad_ramp_updates": HGRAD_RAMP_UPDATES,
    }), flush=True)
    summary = run_experiment(
        args, payload, baseline_metadata, baseline_summary_metadata,
        sim_args, raw_loader, normalizer, soft_sps_normalizer,
        teacher_cache, teacher_metadata, scale_contract,
        normalizer_metadata, device)
    write_summary(args.output_dir, summary)
    print(json.dumps({
        "event": "c208_shg1b_complete", "status": summary["status"],
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
    }), flush=True)


if __name__ == "__main__":
    main()
