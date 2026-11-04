#!/usr/bin/env python3
"""Run the single pre-registered C208 Real/Sim rehearsal endpoint.

Each update first builds and consumes the Real supervised graph, then builds
and consumes one Sim truth/P1-cache graph, and finally takes one optimizer
step.  Real temporal-tail windows never enter an optimizer loss or checkpoint
selection: they are read only at the pre-registered update-2000 GO/STOP gate;
the terminal update-4000 checkpoint performs no second tail audit.  A zero
rehearsal weight is the explicit
identity control and does not load Sim data, the P1 cache, or an extra
data-loader RNG.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ensemble_unet2d as ensemble_train  # noqa: E402
from build_sim_p1_point_teacher_cache import (  # noqa: E402
    load_and_validate_cache,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


FORMAL_REHEARSAL_WEIGHT = 0.05
IDENTITY_REHEARSAL_WEIGHT = 0.0
FORMAL_P1_DISTILL_WEIGHT = 0.005
FORMAL_REQUESTED_UPDATES = 5000
FORMAL_GATE_UPDATE = 2000
FORMAL_TERMINAL_UPDATE = 4000
FORMAL_SCHEDULER_T_MAX = FORMAL_REQUESTED_UPDATES
FORMAL_CHECKPOINT_INTERVAL = 4000
FORMAL_BATCH_SIZE = 8
FORMAL_LR = 2e-4
FORMAL_WEIGHT_DECAY = 1e-5
FORMAL_GRAD_CLIP = 1.0
FORMAL_SEED = 0
FORMAL_LOG_INTERVAL = 500
FORMAL_WINDOW_STRIDE = 20
FORMAL_VAL_FRACTION = 0.2
REAL_ACCURACY_WEIGHTS = (5.0, 3.0, 2.0)
SIM_ACCURACY_WEIGHTS = (1.0, 4.0, 1.0)
REAL_MSE_WARMUP_UPDATES = 1500
REAL_SOFT_SPS_WEIGHT = float(__import__("os").environ.get(
    "B210_REAL_SOFT_SPS_WEIGHT", "0.05"))
REAL_SOFT_SPS_WARMUP_UPDATES = 1500
REAL_SOFT_SPS_TAU_FRACTION = 0.05
FORMAL_REAL_COUNTS = {
    "train_windows": 2576,
    "val_windows": 725,
    "trajectories": 82,
}
FORMAL_REAL_SPLIT_SHA256 = (
    "b83811f908c8d55a9bd255d158f5f2648b3248a4271bca404c3085f36570dbbd")
FORMAL_REAL_STATS_SHA256 = (
    "bae5235f73431ac7d76e6bdcea480b8ca9fda0ac01e57ab26af66bad20dc3b40")
FORMAL_START_SIM_SHA256 = (
    "6674fcc6a1d777e6407b99bff1a14ee0e106c706a4b1cb67a9108e850ae809e3")
FORMAL_P1_CACHE_SHA256 = (
    "70cdc0d0d50c273bcd0abeaa706a21d4ef18f4709abcad62325f2c7111e1592a")
FORMAL_IDENTITY_SUMMARY_SHA256 = (
    "8775048ee9a176c82ef66a89accb827b000043a4fb36d1e8aab496bcdb29b975")
FORMAL_IDENTITY_STEP2000_METRICS = {
    "rel_l2_score": 95.76033380914909,
    "tke_rel_l2_score": 80.57366790342594,
    "mvpe_rel_l2_score": 96.45037244906473,
    "point_score": 91.34234176541528,
    "native_sps_score": 51.99563466623378,
    "native_sps_coverage": 0.8765575024,
    "native_sps_mean_width": 0.01595710049,
}
FORMAL_SIM_TRAJECTORIES = 100
EXPECTED_PARAMETER_COUNT = 175_387_412
EXPECTED_MODEL_CONFIG = {
    "architecture": "ensemble_unet",
    "base": 208,
    "depth": 4,
    "groups": 8,
    "difference_steps": 5,
    "use_coordinates": True,
    "use_regime_film": True,
    "multi_scale_regime_film": True,
    "mean_fluctuation": True,
    "ensemble_members": 4,
    "uq_hidden_channels": 160,
    "member_adapter_channels": 0,
    "mean_profile_channels": 104,
    "uq_flow_context": True,
    "residual_quantile_intervals": True,
    "utility_refinement": True,
    "utility_refinement_kernel": 3,
    "utility_refinement_blocks": 0,
    "temporal_uq_refinement": True,
    "temporal_uq_channels": 24,
    "temporal_uq_blocks": 2,
    "temporal_uq_kernel": 3,
    "coverage_calibration": False,
    "disagreement_scale_fraction": 0.01,
}
NORMALIZER_KEYS = (
    "mean_input", "mean_target", "std_input", "std_target")

# Frozen from the observed C208 losses, not tuned by this experiment.
OBSERVED_REAL_SUPERVISED_STEP2000 = 0.1340824
OBSERVED_SIM_TRUTH_PLUS_DISTILL_STEP10000 = 0.1627169
EXPECTED_REHEARSAL_TO_REAL_LOSS_RATIO = (
    FORMAL_REHEARSAL_WEIGHT * OBSERVED_SIM_TRUTH_PLUS_DISTILL_STEP10000
    / OBSERVED_REAL_SUPERVISED_STEP2000)

GATE_POINT_DELTA_MIN = 0.03
GATE_COMPONENT_DELTA_FLOOR = -0.10
GATE_SPS_DELTA_FLOOR = -0.10
NON_TIME_COEFFICIENTS = {
    "rel_l2_score": 0.47201126,
    "tke_rel_l2_score": 0.08586562,
    "mvpe_rel_l2_score": 0.09410214,
    "native_sps_score": 0.24699273,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--updates", type=int, default=FORMAL_REQUESTED_UPDATES,
        help="Frozen at 5000 to retain the standard Real LR schedule.")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument(
        "--rehearsal-weight", type=float,
        default=IDENTITY_REHEARSAL_WEIGHT,
        help=("Only 0 (identity control) or the pre-registered 0.05 is "
              "accepted; no weight sweep is supported."))
    parser.add_argument("--p1-cache", type=Path)
    parser.add_argument("--p1-cache-sha256")
    parser.add_argument("--identity-summary", type=Path)
    parser.add_argument("--identity-summary-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def normalized_sha256(value: str, name: str) -> str:
    result = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.updates != FORMAL_REQUESTED_UPDATES:
        raise ValueError("formal rehearsal requires --updates 5000")
    args.start_sim_sha256 = normalized_sha256(
        args.start_sim_sha256, "start-sim-sha256")
    if args.start_sim_sha256 != FORMAL_START_SIM_SHA256:
        raise ValueError("start-sim-sha256 is not the pre-registered Sim start")
    allowed = (IDENTITY_REHEARSAL_WEIGHT, FORMAL_REHEARSAL_WEIGHT)
    matched = next(
        (value for value in allowed
         if math.isclose(args.rehearsal_weight, value,
                         rel_tol=0.0, abs_tol=1e-12)), None)
    if matched is None:
        raise ValueError(
            "rehearsal-weight must be exactly 0 or the frozen 0.05")
    args.rehearsal_weight = matched
    supplied_cache = args.p1_cache is not None
    supplied_cache_sha = args.p1_cache_sha256 is not None
    if args.rehearsal_weight == IDENTITY_REHEARSAL_WEIGHT:
        if supplied_cache or supplied_cache_sha:
            raise ValueError(
                "identity control must not load a P1 cache or cache SHA")
        if (args.identity_summary is not None
                or args.identity_summary_sha256 is not None):
            raise ValueError(
                "identity control records its own metrics and takes no "
                "identity-summary artifact")
    else:
        if not supplied_cache or not supplied_cache_sha:
            raise ValueError(
                "formal rehearsal requires both --p1-cache and its SHA-256")
        args.p1_cache_sha256 = normalized_sha256(
            args.p1_cache_sha256, "p1-cache-sha256")
        if args.p1_cache_sha256 != FORMAL_P1_CACHE_SHA256:
            raise ValueError("p1-cache-sha256 is not the pre-registered P1 cache")
        if (args.identity_summary is None
                or args.identity_summary_sha256 is None):
            raise ValueError(
                "formal rehearsal requires the frozen identity summary")
        args.identity_summary_sha256 = normalized_sha256(
            args.identity_summary_sha256, "identity-summary-sha256")
        if args.identity_summary_sha256 != FORMAL_IDENTITY_SUMMARY_SHA256:
            raise ValueError(
                "identity-summary-sha256 is not the pre-registered control")
    if (args.output_dir / "summary.json").exists() or (
            args.output_dir / "model_04000.pth").exists():
        raise FileExistsError(
            f"refusing to overwrite a completed run in {args.output_dir}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


IDENTITY_TRAINING_CONTRACT = {
    "phase": "real",
    "updates": FORMAL_REQUESTED_UPDATES,
    "batch_size": FORMAL_BATCH_SIZE,
    "lr": FORMAL_LR,
    "weight_decay": FORMAL_WEIGHT_DECAY,
    "grad_clip": FORMAL_GRAD_CLIP,
    "seed": FORMAL_SEED,
    "window_stride": FORMAL_WINDOW_STRIDE,
    "val_fraction": FORMAL_VAL_FRACTION,
    "real_all_windows": False,
    "input_noise_std": 0.0,
    "point_loss_mode": "official_accuracy",
    "mse_weight": 0.25,
    "mse_warmup_updates": REAL_MSE_WARMUP_UPDATES,
    "mean_loss_weight": 0.25,
    "member_loss_weight": 1.0,
    "member_loss_mode": "point",
    "bootstrap_probability": 0.75,
    "quantile_loss_weight": 0.05,
    "soft_sps_loss_weight": REAL_SOFT_SPS_WEIGHT,
    "soft_sps_tau_fraction": REAL_SOFT_SPS_TAU_FRACTION,
    "soft_sps_warmup_updates": REAL_SOFT_SPS_WARMUP_UPDATES,
    "soft_sps_uq_only": True,
    "accuracy_weights": [5.0, 3.0, 2.0],
    "checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
}
IDENTITY_METRIC_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
    "native_sps_score", "native_sps_coverage", "native_sps_mean_width",
    "point_score")


def load_identity_step2000(args: argparse.Namespace) -> dict | None:
    """Validate the exact standard Direct summary used as the gate control."""
    if args.rehearsal_weight == IDENTITY_REHEARSAL_WEIGHT:
        return None
    actual = sha256(args.identity_summary)
    if actual != args.identity_summary_sha256:
        raise RuntimeError(
            f"identity summary SHA-256 {actual} != expected "
            f"{args.identity_summary_sha256}")
    if actual != FORMAL_IDENTITY_SUMMARY_SHA256:
        raise RuntimeError(
            "identity summary is not the pre-registered C208 control artifact")
    payload = json.loads(args.identity_summary.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("identity summary must contain a JSON mapping")
    training = payload.get("training_config")
    if not isinstance(training, dict):
        raise ValueError("identity summary has no training_config")
    errors = []
    for key, expected in IDENTITY_TRAINING_CONTRACT.items():
        actual_value = training.get(key)
        if isinstance(expected, float):
            matches = isinstance(actual_value, (int, float)) and math.isclose(
                float(actual_value), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = actual_value == expected
        if not matches:
            errors.append(
                f"training_config[{key}]={actual_value!r}, expected {expected!r}")
    history = payload.get("history")
    if not isinstance(history, list):
        errors.append("identity summary has no history")
        rows = []
    else:
        rows = [
            row for row in history if isinstance(row, dict)
            and int(row.get("iteration", -1)) == FORMAL_GATE_UPDATE]
    if len(rows) != 1:
        errors.append("identity summary must have exactly one update-2000 row")
    if errors:
        raise ValueError("; ".join(errors))
    missing_metrics = [key for key in IDENTITY_METRIC_KEYS if key not in rows[0]]
    if missing_metrics:
        raise ValueError(
            f"identity update-2000 row is missing metrics {missing_metrics}")
    metrics = {key: float(rows[0][key]) for key in IDENTITY_METRIC_KEYS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("identity update-2000 metrics are non-finite")
    for key, expected in FORMAL_IDENTITY_STEP2000_METRICS.items():
        if not math.isclose(
                metrics[key], expected, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"identity metric {key}={metrics[key]}, expected {expected}")
    return {
        "path": str(args.identity_summary.resolve()),
        "sha256": actual,
        "metrics": metrics,
    }


def load_start_payload(args: argparse.Namespace) -> dict:
    actual = sha256(args.start_sim)
    if actual != args.start_sim_sha256:
        raise RuntimeError(
            f"start Sim SHA-256 {actual} != expected {args.start_sim_sha256}")
    payload = torch.load(
        args.start_sim, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("start Sim checkpoint did not contain a mapping")
    if payload.get("phase") != "sim" or int(payload.get("iteration", -1)) != 10_000:
        raise ValueError("start must be the frozen Sim update-10000 checkpoint")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("start checkpoint has no model_config")
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(
                f"start model_config[{key!r}]={config.get(key)!r}, "
                f"expected {expected!r}")
    training = payload.get("training_config", {})
    if not math.isclose(
            float(training.get("distill_weight", -1.0)),
            FORMAL_P1_DISTILL_WEIGHT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("start is not the frozen P1-distilled Sim checkpoint")
    if not training.get("teacher_cache"):
        raise ValueError("start checkpoint does not record its P1 teacher cache")
    return payload


def load_model(payload: dict, device: torch.device) -> EnsembleUNet2d:
    stored = dict(payload["model_config"])
    if stored.pop("architecture") != "ensemble_unet":
        raise ValueError("start checkpoint architecture changed after validation")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"C208 parameter count {parameters} != {EXPECTED_PARAMETER_COUNT}")
    return model.to(device)


def frozen_sim_normalizer(payload: dict, device: torch.device):
    """Recreate the exact Sim normalizer persisted by the frozen start."""
    stored = payload.get("normalizer")
    if not isinstance(stored, dict) or any(
            key not in stored for key in NORMALIZER_KEYS):
        raise ValueError("start checkpoint has no complete Sim normalizer")
    tensors = {
        key: torch.as_tensor(stored[key], device=device, dtype=torch.float32)
        for key in NORMALIZER_KEYS
    }
    if any(value.shape != (3,) for value in tensors.values()):
        raise ValueError("start checkpoint has invalid Sim normalizer shapes")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("start checkpoint has non-finite Sim normalizer values")
    if bool((tensors["std_input"] <= 0).any()) or bool(
            (tensors["std_target"] <= 0).any()):
        raise ValueError("start checkpoint has non-positive Sim std")
    normalizer = ensemble_train.common.scratch.ChannelNormalizer(
        tensors["mean_input"], tensors["mean_target"],
        tensors["std_input"], tensors["std_target"], device=device)
    normalized = {
        key: tensors[key].detach().cpu().tolist() for key in NORMALIZER_KEYS}
    return normalizer, normalized, canonical_sha256(normalized)


def prepare_real_training(args: argparse.Namespace, device: torch.device) -> dict:
    trajectories, metadata = ensemble_train.common.repro.load_trajectories(
        args.real_dir, 0)
    train_windows, validation_windows, manifest = (
        ensemble_train.common.repro.make_split(
            metadata, FORMAL_WINDOW_STRIDE, FORMAL_VAL_FRACTION))
    counts = {
        "train_windows": len(train_windows),
        "val_windows": len(validation_windows),
        "trajectories": len(trajectories),
    }
    errors = [
        f"{key}={counts[key]}, expected {expected}"
        for key, expected in FORMAL_REAL_COUNTS.items()
        if counts[key] != expected]
    manifest_sha = canonical_sha256(manifest)
    stats_sha = sha256(args.real_stats)
    if manifest_sha != FORMAL_REAL_SPLIT_SHA256:
        errors.append("fixed Real split canonical hash mismatch")
    if stats_sha != FORMAL_REAL_STATS_SHA256:
        errors.append("Real stats SHA-256 mismatch")
    if errors:
        raise RuntimeError("; ".join(errors))

    # Only the fixed 2576 windows can enter the optimizer loader.  The tail
    # loader is held inert until the single update-2000 gate.
    dataset = ensemble_train.common.repro.WindowDataset(
        trajectories, train_windows)
    validation_dataset = ensemble_train.common.repro.WindowDataset(
        trajectories, validation_windows)
    tagged = ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_size=FORMAL_BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(FORMAL_SEED))
    validation_loader = DataLoader(
        validation_dataset, batch_size=16, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    normalizer = ensemble_train.common.repro.Normalizer(
        args.real_stats, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "real_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return {
        "loader": loader,
        "validation_loader": validation_loader,
        "normalizer": normalizer,
        "counts": counts,
        "manifest_sha256": manifest_sha,
        "stats_sha256": stats_sha,
        "validation_optimizer_access": False,
        "validation_access_updates": [],
    }


def prepare_sim_rehearsal(
        args: argparse.Namespace, device: torch.device,
        sim_normalizer=None,
        ) -> dict | None:
    """Return by identity before any Sim/cache/RNG work when weight is zero."""
    if args.rehearsal_weight == IDENTITY_REHEARSAL_WEIGHT:
        return None
    if sim_normalizer is None:
        raise ValueError("formal rehearsal requires the frozen Sim normalizer")
    actual_cache_sha = sha256(args.p1_cache)
    if actual_cache_sha != args.p1_cache_sha256:
        raise RuntimeError(
            f"P1 cache SHA-256 {actual_cache_sha} != expected "
            f"{args.p1_cache_sha256}")
    trajectories, windows, manifest = (
        ensemble_train.common.scratch.load_simulations(
            args.sim_dir, 0, FORMAL_WINDOW_STRIDE, True))
    if len(trajectories) != FORMAL_SIM_TRAJECTORIES:
        raise RuntimeError(
            f"Sim trajectories={len(trajectories)}, "
            f"expected {FORMAL_SIM_TRAJECTORIES}")
    dataset = ensemble_train.common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    cache, metadata = load_and_validate_cache(
        args.p1_cache, manifest, dataset)
    if metadata["sha256"] != args.p1_cache_sha256:
        raise RuntimeError("validated P1 cache SHA-256 changed during loading")
    tagged = ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_size=FORMAL_BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(FORMAL_SEED))
    (args.output_dir / "sim_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return {
        "loader": loader,
        "normalizer": sim_normalizer,
        "cache": cache,
        "cache_metadata": metadata,
        "trajectories": len(trajectories),
        "windows": len(windows),
        "manifest_sha256": canonical_sha256(manifest),
    }


def real_loss_args() -> argparse.Namespace:
    return argparse.Namespace(
        point_loss_mode="official_accuracy",
        mse_weight=0.25,
        mse_warmup_updates=REAL_MSE_WARMUP_UPDATES,
        bootstrap_probability=0.75,
        seed=FORMAL_SEED,
    )


def sim_loss_args() -> argparse.Namespace:
    return argparse.Namespace(
        point_loss_mode="official_accuracy",
        mse_weight=0.25,
        mse_warmup_updates=0,
    )


def backward_real_supervised(
        model: EnsembleUNet2d, batch, normalizer,
        device: torch.device, iteration: int, *,
        exact_physical_uq_target: bool = False,
        member_mask: torch.Tensor | None = None,
        ) -> dict[str, float]:
    history, target, trajectory_ids, _sample_indices = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    history_norm, target_norm = normalizer.preprocess(history, target)
    (prediction, members, quantiles, lower, upper, features,
     _coverage_logits) = model.forward_uq_training(history_norm)
    point_loss, point_components = ensemble_train.point_loss(
        real_loss_args(), prediction, target_norm, normalizer,
        REAL_ACCURACY_WEIGHTS, iteration)
    member_loss, bootstrap_fraction = (
        ensemble_train.bootstrap_member_point_loss(
            real_loss_args(), members, target_norm, trajectory_ids,
            normalizer, REAL_ACCURACY_WEIGHTS, iteration,
            member_mask=member_mask))
    quantile_loss = ensemble_train.residual_quantile_loss(
        model, prediction, quantiles, target_norm, normalizer,
        physical_target=(target if exact_physical_uq_target else None))
    soft_sps_loss, soft_sps_score = (
        ensemble_train.soft_native_sps_objective(
            prediction, target_norm, lower, upper, normalizer,
            tau_fraction=REAL_SOFT_SPS_TAU_FRACTION,
            physical_target=(target if exact_physical_uq_target else None)))
    soft_sps_weight = REAL_SOFT_SPS_WEIGHT * min(
        1.0, iteration / REAL_SOFT_SPS_WARMUP_UPDATES)
    supervised_loss = (
        0.25 * point_loss + 1.0 * member_loss + 0.05 * quantile_loss)
    total_loss = supervised_loss + soft_sps_weight * soft_sps_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError(
            f"non-finite Real loss at update {iteration}")

    # Consume the complete Real graph before a Sim batch is even requested.
    supervised_loss.backward(retain_graph=True)
    interval_parameters = ensemble_train.uq_trainable_parameters(model)
    interval_gradients = torch.autograd.grad(
        soft_sps_weight * soft_sps_loss,
        interval_parameters, allow_unused=True)
    for parameter, gradient in zip(interval_parameters, interval_gradients):
        if gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            parameter.grad.add_(gradient)
    values = {
        "real_loss": float(total_loss.detach()),
        "real_supervised_loss": float(supervised_loss.detach()),
        "real_point_loss": float(point_loss.detach()),
        "real_member_loss": float(member_loss.detach()),
        "real_quantile_loss": float(quantile_loss.detach()),
        "real_soft_sps_loss": float(soft_sps_loss.detach()),
        "real_soft_sps_score": float(soft_sps_score.detach()),
        "real_soft_sps_weight": soft_sps_weight,
        "real_bootstrap_fraction": float(bootstrap_fraction.detach()),
        **{
            f"real_{key}": float(value.detach())
            for key, value in point_components.items()
        },
    }
    del (history, target, history_norm, target_norm, prediction, members,
         quantiles, lower, upper, features, point_loss, member_loss,
         quantile_loss, soft_sps_loss, supervised_loss, total_loss,
         interval_gradients)
    return values


def backward_sim_rehearsal(
        model: EnsembleUNet2d, batch, normalizer,
        real_normalizer, cache: torch.Tensor, device: torch.device,
        outer_weight: float,
        ) -> dict[str, float]:
    history, target, _trajectory_ids, sample_indices = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    # Point predictions include a normalized persistence residual.  Switch its
    # non-trainable buffers to the frozen Sim statistics for this graph, then
    # restore Real statistics before the next Real update/checkpoint.
    ensemble_train.common.set_model_stats(model, normalizer)
    history_norm, target_norm = normalizer.preprocess(history, target)
    prediction = model(history_norm)
    truth_loss, truth_components = ensemble_train.point_loss(
        sim_loss_args(), prediction, target_norm, normalizer,
        SIM_ACCURACY_WEIGHTS, iteration=1)
    teacher = cache[sample_indices].to(
        device=device, dtype=torch.float32, non_blocking=True)
    distillation, distillation_components = (
        ensemble_train.component_point_distillation_loss(
            prediction, teacher, normalizer))
    raw_loss = truth_loss + FORMAL_P1_DISTILL_WEIGHT * distillation
    weighted_loss = outer_weight * raw_loss
    if not torch.isfinite(weighted_loss):
        raise FloatingPointError("non-finite Sim rehearsal loss")
    weighted_loss.backward()
    ensemble_train.common.set_model_stats(model, real_normalizer)
    values = {
        "rehearsal_raw_loss": float(raw_loss.detach()),
        "rehearsal_weighted_loss": float(weighted_loss.detach()),
        "rehearsal_truth_loss": float(truth_loss.detach()),
        "rehearsal_distillation_loss": float(distillation.detach()),
        "rehearsal_outer_weight": outer_weight,
        "rehearsal_inner_distill_weight": FORMAL_P1_DISTILL_WEIGHT,
        **{
            f"rehearsal_truth_{key}": float(value.detach())
            for key, value in truth_components.items()
        },
        **{
            f"rehearsal_{key}": float(value.detach())
            for key, value in distillation_components.items()
        },
    }
    del (history, target, history_norm, target_norm, prediction, teacher,
         truth_loss, distillation, raw_loss, weighted_loss)
    return values


def sequential_backward_update(
        optimizer: torch.optim.Optimizer, model: torch.nn.Module,
        real_backward: Callable[[], dict[str, float]],
        rehearsal_backward: Callable[[], dict[str, float]] | None,
        ) -> dict[str, float]:
    """Consume Real then Sim graphs and take exactly one optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    values = real_backward()
    if rehearsal_backward is not None:
        values.update(rehearsal_backward())
    torch.nn.utils.clip_grad_norm_(model.parameters(), FORMAL_GRAD_CLIP)
    optimizer.step()
    return values


def evaluate_step2000_gate(
        model: EnsembleUNet2d, real: dict, device: torch.device,
        identity: dict | None,
        ) -> dict:
    if real["validation_access_updates"]:
        raise RuntimeError("fixed725 was already accessed before its single gate")
    metrics = ensemble_train.evaluate(
        model, real["validation_loader"], real["normalizer"], device,
        REAL_ACCURACY_WEIGHTS)
    real["validation_access_updates"].append(FORMAL_GATE_UPDATE)
    metrics = {key: float(value) for key, value in metrics.items()}
    if identity is None:
        return {
            "status": "IDENTITY_STEP2000_RECORDED",
            "metrics": metrics,
            "identity_summary": None,
            "delta": None,
            "non_time_shadow": None,
            "pass": True,
        }
    control = identity["metrics"]
    delta = {
        key: metrics[key] - control[key] for key in IDENTITY_METRIC_KEYS}
    shadow = sum(
        NON_TIME_COEFFICIENTS[key] * delta[key]
        for key in NON_TIME_COEFFICIENTS)
    component_floors = all(
        delta[key] >= GATE_COMPONENT_DELTA_FLOOR
        for key in (
            "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score"))
    passed = bool(
        delta["point_score"] >= GATE_POINT_DELTA_MIN
        and delta["native_sps_score"] >= GATE_SPS_DELTA_FLOOR
        and component_floors and shadow > 0.0)
    return {
        "status": "GO_CONTINUE_TO_UPDATE4000" if passed
        else "STOP_SIM_REHEARSAL_UPDATE2000_GATE",
        "metrics": metrics,
        "identity_summary": identity,
        "delta": delta,
        "component_floors_pass": component_floors,
        "non_time_shadow": shadow,
        "pass": passed,
    }


def train_endpoint(
        args: argparse.Namespace, model: EnsembleUNet2d,
        real: dict, rehearsal: dict | None, identity: dict | None,
        device: torch.device,
        ) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FORMAL_LR,
        weight_decay=FORMAL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    real_iterator = ensemble_train.common.repro.cycling(real["loader"])
    rehearsal_iterator = (
        ensemble_train.common.repro.cycling(rehearsal["loader"])
        if rehearsal is not None else None)
    history = []
    sums: dict[str, float] = {}
    count = 0
    started = time.time()
    gate = None

    for iteration in range(1, FORMAL_TERMINAL_UPDATE + 1):
        model.train()
        real_batch = next(real_iterator)

        def real_step() -> dict[str, float]:
            return backward_real_supervised(
                model, real_batch, real["normalizer"], device, iteration)

        rehearsal_step = None
        if rehearsal_iterator is not None:
            # The Sim batch is requested only after real_step has returned.
            def rehearsal_step() -> dict[str, float]:
                return backward_sim_rehearsal(
                    model, next(rehearsal_iterator), rehearsal["normalizer"],
                    real["normalizer"], rehearsal["cache"], device,
                    args.rehearsal_weight)

        values = sequential_backward_update(
            optimizer, model, real_step, rehearsal_step)
        scheduler.step()
        values["learning_rate"] = optimizer.param_groups[0]["lr"]
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % FORMAL_LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps({
                "event": "c208_real_sim_rehearsal_progress", **row,
            }), flush=True)
            sums.clear()
            count = 0
        if iteration == FORMAL_GATE_UPDATE:
            gate = evaluate_step2000_gate(model, real, device, identity)
            print(json.dumps({
                "event": "c208_real_sim_rehearsal_gate", **gate,
            }), flush=True)
            if not gate["pass"]:
                return {
                    "status": gate["status"],
                    "completed_updates": iteration,
                    "history": history,
                    "gate": gate,
                    "checkpoint": None,
                    "wall_seconds": time.time() - started,
                }

    stats = ensemble_train.common.normalizer_dict(real["normalizer"])
    checkpoint = args.output_dir / "model_04000.pth"
    ensemble_train.common.save_checkpoint(
        checkpoint, model, args, FORMAL_TERMINAL_UPDATE,
        history, stats, history[-1])
    return {
        "status": "COMPLETE_UPDATE4000_AFTER_GATE",
        "completed_updates": FORMAL_TERMINAL_UPDATE,
        "history": history,
        "gate": gate,
        "checkpoint": str(checkpoint),
        "wall_seconds": time.time() - started,
    }


def gate_contract() -> dict:
    return {
        "gate_update": FORMAL_GATE_UPDATE,
        "terminal_update_after_pass": FORMAL_TERMINAL_UPDATE,
        "evaluation": "single_in_process_fixed725_gate_at_update2000",
        "comparator": (
            "same_start_same_real_batches_identity_weight0_update2000"),
        "point_score_delta_min": GATE_POINT_DELTA_MIN,
        "rel_l2_delta_floor": GATE_COMPONENT_DELTA_FLOOR,
        "tke_rel_l2_delta_floor": GATE_COMPONENT_DELTA_FLOOR,
        "mvpe_rel_l2_delta_floor": GATE_COMPONENT_DELTA_FLOOR,
        "native_sps_delta_floor": GATE_SPS_DELTA_FLOOR,
        "window_weighted_non_time_shadow_min_exclusive": 0.0,
        "non_time_coefficients": NON_TIME_COEFFICIENTS,
        "on_pass": "same_process_continue_with_optimizer_state_to_update4000",
        "on_failure": "STOP_SIM_REHEARSAL_NO_WEIGHT_SWEEP",
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.phase = "real"
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal C208 rehearsal training requires CUDA")
    ensemble_train.seed_everything(FORMAL_SEED)
    torch.set_float32_matmul_precision("high")
    payload = load_start_payload(args)
    model = load_model(payload, device)
    sim_normalizer = None
    sim_normalizer_values = None
    sim_normalizer_sha256 = None
    if args.rehearsal_weight != IDENTITY_REHEARSAL_WEIGHT:
        (sim_normalizer, sim_normalizer_values,
         sim_normalizer_sha256) = frozen_sim_normalizer(payload, device)
    del payload
    real = prepare_real_training(args, device)
    ensemble_train.common.set_model_stats(model, real["normalizer"])
    rehearsal = prepare_sim_rehearsal(args, device, sim_normalizer)
    identity = load_identity_step2000(args)
    print(json.dumps({
        "event": "c208_real_sim_rehearsal_contract",
        "parameters": EXPECTED_PARAMETER_COUNT,
        "requested_updates": args.updates,
        "gate_update": FORMAL_GATE_UPDATE,
        "terminal_update": FORMAL_TERMINAL_UPDATE,
        "scheduler_t_max": FORMAL_SCHEDULER_T_MAX,
        "rehearsal_weight": args.rehearsal_weight,
        "p1_distill_weight": FORMAL_P1_DISTILL_WEIGHT,
        "expected_rehearsal_to_real_loss_ratio": (
            EXPECTED_REHEARSAL_TO_REAL_LOSS_RATIO),
        "real_validation_optimizer_access": False,
    }), flush=True)
    result = train_endpoint(
        args, model, real, rehearsal, identity, device)
    summary = {
        "method": "c208_real_phase_sequential_sim_rehearsal",
        "evidence_class": (
            "fixed2576_real_optimizer_plus_sim_truth_p1_cache_"
            "single_fixed725_gate_update2000_terminal_update4000"),
        "start_sim": str(args.start_sim),
        "start_sim_sha256": args.start_sim_sha256,
        "real_contract": {
            "counts": real["counts"],
            "split_manifest_sha256": real["manifest_sha256"],
            "real_stats_sha256": real["stats_sha256"],
            "validation_optimizer_access": (
                real["validation_optimizer_access"]),
            "validation_access_updates": (
                real["validation_access_updates"]),
        },
        "rehearsal_contract": {
            "weight": args.rehearsal_weight,
            "formal_nonzero_weight": FORMAL_REHEARSAL_WEIGHT,
            "inner_p1_distill_weight": FORMAL_P1_DISTILL_WEIGHT,
            "sim_accuracy_weights": SIM_ACCURACY_WEIGHTS,
            "observed_real_supervised_step2000": (
                OBSERVED_REAL_SUPERVISED_STEP2000),
            "observed_sim_truth_plus_distill_step10000": (
                OBSERVED_SIM_TRUTH_PLUS_DISTILL_STEP10000),
            "expected_loss_ratio": EXPECTED_REHEARSAL_TO_REAL_LOSS_RATIO,
            "graph_order": (
                "real_forward_backward_then_sim_forward_backward_then_step"),
            "identity_zero_constructs_sim_resources": False,
            "model_buffer_normalizer_order": (
                "real_backward_then_sim_stats_sim_backward_then_real_stats"),
            "frozen_sim_normalizer": sim_normalizer_values,
            "frozen_sim_normalizer_sha256": sim_normalizer_sha256,
            "sim": (
                None if rehearsal is None else {
                    "trajectories": rehearsal["trajectories"],
                    "windows": rehearsal["windows"],
                    "split_manifest_sha256": rehearsal["manifest_sha256"],
                    "cache": rehearsal["cache_metadata"],
                }),
        },
        "training_contract": {
            "requested_updates": args.updates,
            "gate_update": FORMAL_GATE_UPDATE,
            "terminal_update": FORMAL_TERMINAL_UPDATE,
            "scheduler_t_max": FORMAL_SCHEDULER_T_MAX,
            "checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
            "batch_size": FORMAL_BATCH_SIZE,
            "lr": FORMAL_LR,
            "weight_decay": FORMAL_WEIGHT_DECAY,
            "grad_clip": FORMAL_GRAD_CLIP,
            "seed": FORMAL_SEED,
            "real_accuracy_weights": REAL_ACCURACY_WEIGHTS,
            "real_mse_warmup_updates": REAL_MSE_WARMUP_UPDATES,
            "real_soft_sps_weight": REAL_SOFT_SPS_WEIGHT,
            "real_soft_sps_warmup_updates": (
                REAL_SOFT_SPS_WARMUP_UPDATES),
            "checkpoints_saved": (
                ["model_04000.pth"] if result["checkpoint"] else []),
        },
        "gate_contract": gate_contract(),
        **result,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "status": summary["status"],
        "summary": str(args.output_dir / "summary.json"),
        "checkpoint": result["checkpoint"],
    }), flush=True)


if __name__ == "__main__":
    main()
