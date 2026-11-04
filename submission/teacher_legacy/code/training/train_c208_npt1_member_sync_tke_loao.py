#!/usr/bin/env python3
"""Run the two-fold C208 np-user functional-approximation pilot.

Both arms start from the same member-synchronous, zero-output physical-energy
controller.  The control trains every legacy parameter while keeping the
controller frozen.  The candidate also trains the controller and receives a
truth-native TKE relative-L2 gradient on that controller only during the first
1,500 updates.  Four raw/EMA endpoints are sealed before the held prefix is
opened once and evaluated in a fixed four-pass order.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for import_path in (CODE_ROOT, TRAINING_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_real_sim_rehearsal as joint_real  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


class GradientPreflightStop(RuntimeError):
    """A scientific STOP, not an infrastructure failure."""


EXPERIMENT = "c208_npuser_functional_pilot"
SCHEMA = "c208_npuser_functional_pilot_fold_v1"
AGGREGATE_SCHEMA = "c208_npuser_functional_pilot_aggregate_v1"
ENDPOINT_SCHEMA = "c208_npuser_functional_pilot_endpoint_v1"
FORMAL_BRANCH = "codex/npuser-functional-pilot"
FORMAL_ACKNOWLEDGEMENT = "NPUSER_FUNCTIONAL_PILOT"
FORMAL_START_SHA256 = (
    "6674fcc6a1d777e6407b99bff1a14ee0e106c706a4b1cb67a9108e850ae809e3")
FORMAL_MANIFEST_FILE_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
FORMAL_MANIFEST_SEMANTIC_SHA256 = dr1.FORMAL_MANIFEST_SHA256

FOLD_ORDER = (10, 0)
WAVE_ORDER = ((10, 0),)
WAVE_BY_HELD_AOA = {
    held_aoa: wave
    for wave in WAVE_ORDER
    for held_aoa in wave
}
ARM_ORDER = {
    10: ("candidate", "control"),
    0: ("control", "candidate"),
}
ENDPOINT_ORDER = (
    "control_raw", "candidate_raw", "control_ema", "candidate_ema")
UPDATES = 4000
BATCH_SIZE = 8
SEED = 4
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 5000
EMA_DECAY = 0.999
TKE_AUX_WEIGHT = 0.01
TKE_AUX_UPDATES = 1500
GRADIENT_PREFLIGHT_UPDATES = 200
GRADIENT_RATIO_MIN = 0.05
GRADIENT_RATIO_MAX = 0.50
LEGACY_PARAMETERS = 175_387_412
CONTROLLER_PARAMETERS = 7_105
TOTAL_PARAMETERS = 175_394_517
MIN_FREE_BYTES = 4 * 1024 ** 3
PAIR_WALL_BUDGET_SECONDS = {
    "rtx4090": 45.0 * 60.0,
    "rtx4080_super": 60.0 * 60.0,
}

DIRECT_COEFFICIENTS = {
    "rel_l2_score": 0.47201126,
    "tke_rel_l2_score": 0.08586562,
    "mvpe_rel_l2_score": 0.09410214,
}
SPS_COEFFICIENT = 0.24699273
FOLD_NON_TIME_FLOOR = -0.05
PILOT_TKE_DELTA_MIN = 0.20
PILOT_SPS_DELTA_MIN = 0.10
PILOT_NON_TIME_DELTA_MIN = 0.10
PILOT_COMPONENT_DELTA_FLOOR = -0.05
STRONG_TKE_DELTA_MIN = 1.00
STRONG_SPS_DELTA_MIN = 1.80
STRONG_NON_TIME_DELTA_MIN = 0.50
STRONG_REL_DELTA_FLOOR = -0.15
STRONG_MVPE_DELTA_FLOOR = -0.05
OFFICIAL_Q = 0.6854
OFFICIAL_C208 = {
    "rel_l2_score": 94.284314,
    "tke_rel_l2_score": 78.280202,
    "mvpe_rel_l2_score": 93.160448,
    "time_score": 90.344910,
    "native_sps_score": 41.774685,
}
CONSERVATIVE_BIAS = 0.197743
CONSERVATIVE_FINAL_MIN = 81.03

EXPECTED_MISSING_CONTROLLER_KEYS = {
    f"energy_controller.body.{index}.{name}"
    for index in (0, 1, 3, 5)
    for name in ("weight", "bias")
}

CONTRACT = {
    "evidence_class": (
        "paired clean81 prefix-only LOAO; not fixed725, clean716, release, "
        "latency, ZIP, or official evidence"),
    "start_sha256": FORMAL_START_SHA256,
    "manifest_file_sha256": FORMAL_MANIFEST_FILE_SHA256,
    "manifest_semantic_sha256": FORMAL_MANIFEST_SEMANTIC_SHA256,
    "fold_order": list(FOLD_ORDER),
    "wave_order": [list(wave) for wave in WAVE_ORDER],
    "arm_order": {str(key): list(value) for key, value in ARM_ORDER.items()},
    "endpoint_order": list(ENDPOINT_ORDER),
    "controller": {
        "enabled_both_arms": True,
        "member_synchronous": True,
        "hidden_channels": 32,
        "log_limit": 1.0,
        "zero_definition": "body.5 weight/bias and functional delta exact zero",
        "control": "complete update0 controller state frozen bitwise",
        "candidate": "trainable_with_legacy_and_controller_only_tke_aux",
    },
    "training": {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "updates": UPDATES,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "scheduler_eta_min": 0.0,
        "ema_decay": EMA_DECAY,
        "ema_scope": "full_model_parameters",
        "ema_update_order": ema_lib.UPDATE_ORDER_CONTRACT,
        "frozen_control_ema": (
            "canonicalize mathematically constant controller before seal"),
        "dtype": "float32",
        "amp": False,
        "compile": False,
        "loss_helper": "local backward_real_supervised_npt1",
        "exact_physical_uq_target": True,
        "teacher": None,
        "hgrad": None,
        "rehearsal": None,
        "additional_loss": {
            "name": "truth_native_tke_relative_l2",
            "scope": "candidate energy_controller parameters only",
            "weight_start": TKE_AUX_WEIGHT,
            "cosine_to_zero_at_update": TKE_AUX_UPDATES,
            "gradient_preflight_updates": GRADIENT_PREFLIGHT_UPDATES,
            "weighted_aux_to_native_controller_grad_ratio": [
                GRADIENT_RATIO_MIN, GRADIENT_RATIO_MAX],
            "weighted_aux_to_native_controller_grad_ratio_p95": "diagnostic",
        },
    },
    "promotion_gate": {
        "folds": list(FOLD_ORDER),
        "pooled_delta_tke_min": PILOT_TKE_DELTA_MIN,
        "pooled_delta_sps_min": PILOT_SPS_DELTA_MIN,
        "pooled_delta_non_time_min": PILOT_NON_TIME_DELTA_MIN,
        "pooled_delta_rel_floor": PILOT_COMPONENT_DELTA_FLOOR,
        "pooled_delta_mvpe_floor": PILOT_COMPONENT_DELTA_FLOOR,
        "worst_fold_non_time_floor": FOLD_NON_TIME_FLOOR,
        "pass_action": "train_npuser_inspired_B210_bundle",
        "strong_81_projection_diagnostic": {
            "pooled_delta_tke_min": STRONG_TKE_DELTA_MIN,
            "pooled_delta_sps_min": STRONG_SPS_DELTA_MIN,
            "pooled_delta_non_time_min": STRONG_NON_TIME_DELTA_MIN,
            "pooled_delta_rel_floor": STRONG_REL_DELTA_FLOOR,
            "pooled_delta_mvpe_floor": STRONG_MVPE_DELTA_FLOOR,
            "each_fold_tke_sps_non_time_positive": True,
            "conservative_official_projection_min": CONSERVATIVE_FINAL_MIN,
        },
    },
    "held_access": {
        "tail_arrays_read": 0,
        "excluded": dr1.EXCLUDED_BASENAME,
        "open_after_four_endpoint_seals": True,
        "loaded_once_evaluated_four_passes": True,
    },
    "submission_authorized": False,
}
CONTRACT_SHA256 = dr1.canonical_sha256(CONTRACT)


def normalizer_payload(normalizer: smu1.FoldNormalizer) -> dict:
    values = ensemble_train.common.normalizer_dict(normalizer)
    return {"values": values, "sha256": dr1.canonical_sha256(values)}


def controller_state(model: EnsembleUNet2d) -> OrderedDict[str, torch.Tensor]:
    if model.energy_controller is None:
        raise RuntimeError("enabled NPT1 model has no energy controller")
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.energy_controller.state_dict().items())


def controller_output_projection_is_zero(model_or_state) -> bool:
    state = (
        controller_state(model_or_state)
        if isinstance(model_or_state, EnsembleUNet2d) else model_or_state)
    expected = {"body.5.weight", "body.5.bias"}
    if not expected.issubset(state):
        return False
    return all(bool(torch.count_nonzero(state[name]) == 0)
               for name in expected)


def tensor_state_equal(first: Mapping[str, torch.Tensor],
                       second: Mapping[str, torch.Tensor]) -> bool:
    return (list(first) == list(second)
            and all(torch.equal(first[name].detach().cpu(),
                                second[name].detach().cpu())
                    for name in first))


def _controller_front_state(
        state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if name not in {"body.5.weight", "body.5.bias"})


@torch.no_grad()
def canonicalize_frozen_controller_ema(
        ema: ema_lib.FP32EMAShadow,
        model: EnsembleUNet2d,
        expected_controller_sha256: str,
        ) -> dict:
    """Restore the exact mathematical EMA of a frozen constant controller.

    Repeated FP32 ``decay*x + (1-decay)*x`` can drift by roundoff even though
    its real-number value is exactly ``x``.  The control controller is frozen,
    so copying its unchanged raw value into the EMA shadow is canonicalization,
    not endpoint selection.
    """
    raw = controller_state(model)
    raw_sha = ema_lib.tensor_state_sha256(raw)
    if raw_sha != expected_controller_sha256:
        raise RuntimeError("NPT1 frozen raw controller changed before EMA seal")
    before = OrderedDict()
    for name in raw:
        full_name = f"energy_controller.{name}"
        if full_name not in ema._shadow:
            raise RuntimeError("NPT1 EMA lacks frozen controller parameter")
        before[name] = ema._shadow[full_name].detach().cpu().clone()
    changed_elements = sum(
        int(torch.count_nonzero(before[name] != raw[name])) for name in raw)
    max_abs_drift = max(
        (float((before[name] - raw[name]).abs().max())
         if raw[name].numel() else 0.0)
        for name in raw)
    for name, parameter in model.energy_controller.named_parameters():
        ema._shadow[f"energy_controller.{name}"].copy_(
            parameter.detach().to(dtype=torch.float32))
    after = OrderedDict(
        (name, ema._shadow[f"energy_controller.{name}"].detach().cpu().clone())
        for name in raw)
    after_sha = ema_lib.tensor_state_sha256(after)
    if after_sha != raw_sha or not tensor_state_equal(after, raw):
        raise RuntimeError("NPT1 frozen controller EMA canonicalization failed")
    return {
        "applied": True,
        "reason": "fp32_roundoff_of_mathematically_constant_frozen_ema",
        "before_sha256": ema_lib.tensor_state_sha256(before),
        "raw_sha256": raw_sha,
        "after_sha256": after_sha,
        "changed_elements_before": changed_elements,
        "max_abs_drift_before": max_abs_drift,
        "exact_after": True,
    }


def _legacy_config(model_config: Mapping[str, object]) -> dict:
    stored = dict(model_config)
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError("NPT1 start is not an ensemble U-Net")
    optional = {
        "physical_energy_controller": False,
        "physical_energy_hidden_channels": 32,
        "physical_energy_log_limit": 1.0,
        "physical_energy_member_sync": False,
    }
    for key, expected in optional.items():
        value = stored.pop(key, expected)
        if value != expected:
            raise ValueError(f"legacy start has unexpected {key}={value!r}")
    return stored


def validate_start_checkpoint(
        path: Path, expected_sha256: str = FORMAL_START_SHA256,
        *, expected_legacy_parameters: int | None = LEGACY_PARAMETERS,
        ) -> tuple[dict, dict]:
    expected = dr1.normalized_sha256(expected_sha256, "start checkpoint SHA")
    actual = dr1.sha256(path)
    if actual != expected:
        raise RuntimeError("NPT1 start checkpoint SHA-256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict) or payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != 10_000):
        raise ValueError("NPT1 start must be the terminal Sim10k checkpoint")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("NPT1 start lacks model config/state")
    legacy = EnsembleUNet2d(EnsembleUNet2dConfig(**_legacy_config(config)))
    legacy.load_state_dict(state, strict=True)
    parameters = sum(value.numel() for value in legacy.parameters())
    if (expected_legacy_parameters is not None
            and parameters != expected_legacy_parameters):
        raise RuntimeError("NPT1 legacy parameter count changed")
    report = {
        "path": str(path.resolve()),
        "checkpoint_sha256": actual,
        "checkpoint_model_config_sha256": dr1.canonical_sha256(config),
        "checkpoint_model_state_sha256": dr1.model_state_sha256(state),
        "phase": "sim",
        "iteration": 10_000,
        "legacy_strict_load": True,
        "legacy_parameters": parameters,
    }
    del legacy
    gc.collect()
    return payload, report


def build_enabled_model(
        payload: Mapping[str, object], normalizer: smu1.FoldNormalizer,
        device: torch.device, *,
        expected_legacy_parameters: int | None = LEGACY_PARAMETERS,
        expected_total_parameters: int | None = TOTAL_PARAMETERS,
        expected_controller_parameters: int | None = CONTROLLER_PARAMETERS,
        ) -> tuple[EnsembleUNet2d, dict]:
    config = payload["model_config"]
    state = payload["model_state_dict"]
    stored = _legacy_config(config)
    stored.update({
        "physical_energy_controller": True,
        "physical_energy_hidden_channels": 32,
        "physical_energy_log_limit": 1.0,
        "physical_energy_member_sync": True,
    })
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored)).float()
    incompatible = model.load_state_dict(state, strict=False)
    if (set(incompatible.missing_keys) != EXPECTED_MISSING_CONTROLLER_KEYS
            or incompatible.unexpected_keys):
        raise RuntimeError(
            "legacy migration did not have controller-only missing keys")
    live = model.state_dict()
    for name, expected in state.items():
        if name not in live or not torch.equal(live[name].cpu(), expected.cpu()):
            raise RuntimeError(f"legacy tensor changed during migration: {name}")
    controller_parameters = sum(
        value.numel() for value in model.energy_controller.parameters())
    total_parameters = sum(value.numel() for value in model.parameters())
    legacy_parameters = total_parameters - controller_parameters
    expected_counts = (
        (legacy_parameters, expected_legacy_parameters, "legacy"),
        (controller_parameters, expected_controller_parameters, "controller"),
        (total_parameters, expected_total_parameters, "total"),
    )
    for actual, expected, label in expected_counts:
        if expected is not None and actual != expected:
            raise RuntimeError(f"NPT1 {label} parameter count changed")
    if any(value.dtype != torch.float32 for value in model.parameters()):
        raise TypeError("NPT1 model parameters must be FP32")
    if not controller_output_projection_is_zero(model):
        raise RuntimeError("NPT1 enabled migration is not zero-output")
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device), {
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "legacy_tensor_migration_bitwise": True,
        "legacy_parameters": legacy_parameters,
        "controller_parameters": controller_parameters,
        "total_parameters": total_parameters,
    }


def configure_trainable(
        model: EnsembleUNet2d, arm: str) -> tuple[torch.nn.Parameter, ...]:
    if arm not in {"control", "candidate"}:
        raise ValueError(f"unknown NPT1 arm {arm}")
    for name, parameter in model.named_parameters():
        controller = name.startswith("energy_controller.")
        parameter.requires_grad_(arm == "candidate" or not controller)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad)
    expected = TOTAL_PARAMETERS if arm == "candidate" else LEGACY_PARAMETERS
    if sum(value.numel() for value in trainable) != expected:
        raise RuntimeError(f"NPT1 {arm} trainability contract changed")
    if not controller_output_projection_is_zero(model):
        raise RuntimeError("NPT1 update0 controller projection is not zero")
    return trainable


@torch.inference_mode()
def capture_update0(model: EnsembleUNet2d, dataset, index: int,
                    normalizer: smu1.FoldNormalizer,
                    device: torch.device) -> dict[str, torch.Tensor]:
    history, target = dataset[index]
    history = history.unsqueeze(0).to(device=device, dtype=torch.float32)
    target = target.unsqueeze(0).to(device=device, dtype=torch.float32)
    history_norm, _target_norm = normalizer.preprocess(history, target)
    point, members, quantiles, lower, upper, _features, _coverage = (
        model.forward_uq_training(history_norm))
    return {
        "point": point.detach().cpu(),
        "members": members.detach().cpu(),
        "quantiles": quantiles.detach().cpu(),
        "lower": lower.detach().cpu(),
        "upper": upper.detach().cpu(),
    }


def update0_bitwise_identity(first: Mapping[str, torch.Tensor],
                             second: Mapping[str, torch.Tensor]) -> dict:
    checks = {
        key: torch.equal(first[key], second[key])
        for key in ("point", "members", "quantiles", "lower", "upper")
    }
    return {"pass": all(checks.values()), "checks": checks}


def _gradient_audit(model: EnsembleUNet2d) -> dict[str, bool]:
    result = {
        "legacy_nonzero": False,
        "controller_any_grad": False,
        "controller_all_grad_none": True,
        "controller_projection_nonzero": False,
        "controller_front_nonzero": False,
        "finite": True,
    }
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        controller = name.startswith("energy_controller.")
        if controller:
            result["controller_any_grad"] = True
            result["controller_all_grad_none"] = False
        result["finite"] &= bool(torch.isfinite(gradient).all())
        if bool(torch.count_nonzero(gradient)):
            if not controller:
                result["legacy_nonzero"] = True
            elif name.startswith("energy_controller.body.5."):
                result["controller_projection_nonzero"] = True
            else:
                result["controller_front_nonzero"] = True
    return result


def validate_schedule(schedule: Mapping[str, object]) -> None:
    updates = int(schedule["updates"])
    rows = schedule["rows"]
    batches = schedule["batches"]
    if updates != len(rows) or updates != len(batches):
        raise ValueError("NPT1 schedule rows/batches/update count differ")
    for index, (row, batch_indices) in enumerate(zip(rows, batches), start=1):
        if row["sample_indices"] != batch_indices:
            raise RuntimeError(f"NPT1 schedule indices differ at update {index}")
        size = len(row["sample_indices"])
        if size < 1 or size > BATCH_SIZE:
            raise RuntimeError(f"NPT1 schedule batch size changed at {index}")
        if len(row["trajectory_ids"]) != size:
            raise RuntimeError(f"NPT1 schedule trajectory count changed at {index}")
        mask = torch.as_tensor(row["bootstrap_mask"], dtype=torch.bool)
        if mask.device.type != "cpu" or mask.shape != (size, 4):
            raise RuntimeError(f"NPT1 bootstrap mask shape changed at {index}")
        if not bool(mask.any(dim=1).all()):
            raise RuntimeError(f"NPT1 empty bootstrap row at update {index}")


def tke_aux_weight(iteration: int) -> float:
    """Cosine-decay the controller-only TKE loss to exact zero at u1500."""
    if iteration < 1:
        raise ValueError("NPT1 iteration must be positive")
    if iteration > TKE_AUX_UPDATES:
        return 0.0
    phase = (iteration - 1) / (TKE_AUX_UPDATES - 1)
    return TKE_AUX_WEIGHT * 0.5 * (1.0 + math.cos(math.pi * phase))


def truth_native_tke_relative_l2(
        prediction: torch.Tensor, target_physical: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Return the official-shape physical TKE-map relative L2 objective."""
    if target_physical.shape != prediction.shape:
        raise ValueError("NPT1 physical target is not aligned with prediction")
    return ensemble_train.physical_tke_map_distillation_loss(
        prediction, target_physical[..., :2], normalizer)


def official_tke_score_loss(
        prediction_normalized: torch.Tensor,
        target_normalized: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Return the differentiable official TKE score loss per sample."""
    prediction = ensemble_train.common.to_physical_target(
        prediction_normalized, normalizer)[..., :2]
    target = ensemble_train.common.to_physical_target(
        target_normalized, normalizer)[..., :2]
    prediction_fluctuation = (
        prediction - prediction.mean(dim=1, keepdim=True))
    target_fluctuation = target - target.mean(dim=1, keepdim=True)
    prediction_tke = 0.5 * prediction_fluctuation.square().mean(
        dim=1).sum(dim=-1)
    target_tke = 0.5 * target_fluctuation.square().mean(
        dim=1).sum(dim=-1)
    batch = prediction_tke.shape[0]
    relative_error = torch.linalg.vector_norm(
        (prediction_tke - target_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        target_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)
    return (1.0 - 1.0 / (1.0 + 0.5 * relative_error)).mean()


def tke_floor_prediction(
        prediction: torch.Tensor, members: torch.Tensor,
        on_members: bool) -> torch.Tensor:
    """Bind the TKE direction to the tensor receiving the floor correction."""
    if not on_members:
        return prediction
    if (members.ndim != prediction.ndim + 1
            or members.shape[0] != prediction.shape[0]
            or members.shape[2:] != prediction.shape[1:]):
        raise ValueError("member tensor is not aligned with the point prediction")
    return members.mean(dim=1)


def output_legacy_tke_gradient_floor(
        primary_loss: torch.Tensor, legacy_reference_loss: torch.Tensor,
        tke_loss: torch.Tensor, prediction: torch.Tensor,
        ) -> tuple[torch.Tensor | None, dict[str, float | bool]]:
    """Add only the TKE-parallel strength missing from a legacy objective."""
    primary_gradient = torch.autograd.grad(
        primary_loss, prediction, retain_graph=True, allow_unused=False)[0]
    legacy_gradient = torch.autograd.grad(
        legacy_reference_loss, prediction,
        retain_graph=True, allow_unused=False)[0]
    tke_gradient = torch.autograd.grad(
        tke_loss, prediction, retain_graph=True, allow_unused=False)[0]
    shape = primary_gradient.shape
    batch = shape[0]
    primary_flat = primary_gradient.float().reshape(batch, -1)
    legacy_flat = legacy_gradient.float().reshape(batch, -1)
    tke_flat = tke_gradient.float().reshape(batch, -1)
    tke_norm = torch.linalg.vector_norm(tke_flat, dim=1).clamp_min(
        torch.finfo(torch.float32).tiny)
    tke_unit = tke_flat / tke_norm[:, None]
    primary_parallel = (primary_flat * tke_unit).sum(dim=1)
    legacy_parallel = (legacy_flat * tke_unit).sum(dim=1).clamp_min(0.0)
    missing = (legacy_parallel - primary_parallel).clamp_min(0.0)
    primary_norm = torch.linalg.vector_norm(primary_flat, dim=1).clamp_min(
        torch.finfo(torch.float32).tiny)
    ratios = missing / primary_norm
    finite = bool(all(torch.isfinite(value).all() for value in (
        primary_parallel, legacy_parallel, missing, ratios)))
    if not finite:
        raise FloatingPointError("non-finite Real legacy TKE gradient floor")
    applied = missing > 0.0
    correction = None
    if bool(applied.any()):
        correction = (
            missing[:, None] * tke_unit
        ).reshape(shape).to(tke_gradient.dtype)
    return correction, {
        "legacy_tke_floor_finite": finite,
        "legacy_tke_floor_applied_fraction": float(applied.float().mean()),
        "legacy_tke_floor_primary_parallel_median": float(
            primary_parallel.median()),
        "legacy_tke_floor_reference_parallel_median": float(
            legacy_parallel.median()),
        "legacy_tke_floor_missing_parallel_median": float(missing.median()),
        "legacy_tke_floor_correction_ratio_median": float(ratios.median()),
        "legacy_tke_floor_correction_ratio_max": float(ratios.max()),
    }


def gradient_l2_from_parameters(
        parameters: Sequence[torch.nn.Parameter]) -> float:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt())


def gradient_l2_from_tensors(
        gradients: Sequence[torch.Tensor | None]) -> float:
    squared = None
    for gradient in gradients:
        if gradient is None:
            continue
        value = gradient.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt())


def add_parameter_gradients(
        parameters: Sequence[torch.nn.Parameter],
        gradients: Sequence[torch.Tensor | None],
        ) -> None:
    if len(parameters) != len(gradients):
        raise ValueError("NPT1 parameter/gradient length mismatch")
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            parameter.grad.add_(gradient)


def backward_real_supervised_npt1(
        model: EnsembleUNet2d, batch, normalizer,
        device: torch.device, iteration: int, *,
        auxiliary_weight: float,
        member_mask: torch.Tensor,
        audit_controller_gradients: bool,
        input_augmentation: Mapping[str, float] | None = None,
        discard_input_corruption: bool = False,
        full_backbone_auxiliary: torch.nn.Module | None = None,
        full_backbone_auxiliary_weight: float = 0.0,
        full_backbone_gradient_audit_update: int = 0,
        point_accuracy_weights: tuple[float, float, float] | None = None,
        member_accuracy_weights: tuple[float, float, float] | None = None,
        legacy_tke_gradient_floor: bool = False,
        legacy_tke_gradient_floor_on_members: bool = False,
        legacy_point_accuracy_weights: (
            tuple[float, float, float] | None) = None,
        legacy_member_accuracy_weights: (
            tuple[float, float, float] | None) = None,
        input_region_augmentor: Callable | None = None,
        region_auxiliary: Callable | None = None,
        region_auxiliary_weight: float = 0.0,
        loss_scale: float = 1.0,
        ) -> dict[str, float]:
    """Backpropagate the legacy losses plus controller-only native TKE."""
    if not math.isfinite(loss_scale) or loss_scale < 0.0:
        raise ValueError("loss_scale must be finite and non-negative")
    history, target, trajectory_ids, _sample_indices = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    region_augmentation_metrics: dict[str, float] = {}
    if input_region_augmentor is not None:
        history, raw_metrics = input_region_augmentor(
            history, normalizer, iteration)
        if history.shape != batch[0].shape or history.device != device:
            raise ValueError("region augmentor changed history shape/device")
        region_augmentation_metrics = {
            str(key): float(value) for key, value in raw_metrics.items()}
        if any(not math.isfinite(value)
               for value in region_augmentation_metrics.values()):
            raise FloatingPointError("non-finite region augmentation metric")
    augmentation = {
        "input_gain_std": 0.0,
        "input_noise_std": 0.0,
        "input_blur_probability": 0.0,
        "input_blur_strength": 0.0,
    }
    if input_augmentation is not None:
        unknown = set(input_augmentation) - set(augmentation)
        if unknown:
            raise ValueError(
                f"unknown NPT1 input augmentation fields: {sorted(unknown)}")
        augmentation.update({
            key: float(value) for key, value in input_augmentation.items()
        })
    history_input = ensemble_train.add_training_input_gain(
        history, augmentation["input_gain_std"])
    history_norm, target_norm = normalizer.preprocess(history_input, target)
    history_norm = ensemble_train.masked_velocity_blur(
        history_norm, history,
        augmentation["input_blur_probability"],
        augmentation["input_blur_strength"])
    history_norm = ensemble_train.add_training_input_noise(
        history_norm, history, augmentation["input_noise_std"])
    if discard_input_corruption:
        if not any(augmentation.values()):
            raise ValueError('discard requires the matched original augmentation draws')
        history_norm, target_norm = normalizer.preprocess(history, target)
    (prediction, members, quantiles, lower, upper, features,
     _coverage_logits) = model.forward_uq_training(history_norm)
    resolved_point_weights = (
        joint_real.REAL_ACCURACY_WEIGHTS
        if point_accuracy_weights is None else tuple(
            float(value) for value in point_accuracy_weights))
    if (len(resolved_point_weights) != 3
            or any(not math.isfinite(value) or value <= 0.0
                   for value in resolved_point_weights)):
        raise ValueError("Real point accuracy weights must be positive/finite")
    resolved_member_weights = (
        joint_real.REAL_ACCURACY_WEIGHTS
        if member_accuracy_weights is None else tuple(
            float(value) for value in member_accuracy_weights))
    if (len(resolved_member_weights) != 3
            or any(not math.isfinite(value) or value <= 0.0
                   for value in resolved_member_weights)):
        raise ValueError("Real member accuracy weights must be positive/finite")
    if legacy_tke_gradient_floor_on_members and not legacy_tke_gradient_floor:
        raise ValueError("member TKE floor requires the Real TKE floor")
    if legacy_tke_gradient_floor:
        if legacy_point_accuracy_weights is None:
            raise ValueError("Real TKE floor requires legacy point weights")
        resolved_legacy_weights = tuple(
            float(value) for value in legacy_point_accuracy_weights)
        if (len(resolved_legacy_weights) != 3
                or any(not math.isfinite(value) or value <= 0.0
                       for value in resolved_legacy_weights)):
            raise ValueError(
                "Real legacy point weights must be positive/finite")
        if legacy_tke_gradient_floor_on_members:
            if legacy_member_accuracy_weights is None:
                raise ValueError(
                    "member TKE floor requires legacy member weights")
            resolved_legacy_member_weights = tuple(
                float(value) for value in legacy_member_accuracy_weights)
            if (len(resolved_legacy_member_weights) != 3
                    or any(not math.isfinite(value) or value <= 0.0
                           for value in resolved_legacy_member_weights)):
                raise ValueError(
                    "Real legacy member weights must be positive/finite")
        elif legacy_member_accuracy_weights is not None:
            raise ValueError(
                "legacy member weights require the member TKE floor")
        else:
            resolved_legacy_member_weights = None
    elif legacy_point_accuracy_weights is not None:
        raise ValueError("legacy point weights require the Real TKE floor")
    elif legacy_member_accuracy_weights is not None:
        raise ValueError("legacy member weights require the Real TKE floor")
    else:
        resolved_legacy_weights = None
        resolved_legacy_member_weights = None
    point_loss, point_components = ensemble_train.point_loss(
        joint_real.real_loss_args(), prediction, target_norm, normalizer,
        resolved_point_weights, iteration)
    member_loss, bootstrap_fraction = (
        ensemble_train.bootstrap_member_point_loss(
            joint_real.real_loss_args(), members, target_norm, trajectory_ids,
            normalizer, resolved_member_weights, iteration,
            member_mask=member_mask))
    quantile_loss = ensemble_train.residual_quantile_loss(
        model, prediction, quantiles, target_norm, normalizer,
        physical_target=target)
    soft_sps_loss, soft_sps_score = (
        ensemble_train.soft_native_sps_objective(
            prediction, target_norm, lower, upper, normalizer,
            tau_fraction=joint_real.REAL_SOFT_SPS_TAU_FRACTION,
            physical_target=target))
    soft_sps_weight = joint_real.REAL_SOFT_SPS_WEIGHT * min(
        1.0, iteration / joint_real.REAL_SOFT_SPS_WARMUP_UPDATES)
    native_supervised_loss = (
        0.25 * point_loss + 1.0 * member_loss + 0.05 * quantile_loss)
    region_auxiliary_loss = prediction.new_zeros(())
    region_auxiliary_metrics: dict[str, float] = {}
    if region_auxiliary is not None:
        if (not math.isfinite(region_auxiliary_weight)
                or region_auxiliary_weight <= 0.0):
            raise ValueError("region auxiliary weight must be positive/finite")
        region_auxiliary_loss, raw_metrics = region_auxiliary(
            history_input, prediction, target, target_norm, normalizer)
        if (region_auxiliary_loss.ndim != 0
                or not torch.isfinite(region_auxiliary_loss)):
            raise FloatingPointError("region auxiliary loss is invalid")
        region_auxiliary_metrics = {
            str(key): float(value) for key, value in raw_metrics.items()}
        if any(not math.isfinite(value)
               for value in region_auxiliary_metrics.values()):
            raise FloatingPointError("non-finite region auxiliary metric")
    elif region_auxiliary_weight != 0.0:
        raise ValueError("region auxiliary weight has no objective")
    weighted_region_auxiliary = (
        region_auxiliary_weight * region_auxiliary_loss)
    full_backbone_auxiliary_loss = prediction.new_zeros(())
    full_backbone_level_losses: list[torch.Tensor] = []
    if full_backbone_auxiliary is not None:
        if (not math.isfinite(full_backbone_auxiliary_weight)
                or full_backbone_auxiliary_weight <= 0.0):
            raise ValueError("full-backbone auxiliary weight must be positive")
        full_backbone_auxiliary_loss, full_backbone_level_losses = (
            full_backbone_auxiliary(features, target[..., :2]))
    elif full_backbone_auxiliary_weight != 0.0:
        raise ValueError("full-backbone auxiliary weight has no module")
    weighted_full_backbone_auxiliary = (
        full_backbone_auxiliary_weight * full_backbone_auxiliary_loss)
    supervised_loss = (
        native_supervised_loss + weighted_full_backbone_auxiliary
        + weighted_region_auxiliary)
    floor_metrics = {
        "legacy_tke_floor_finite": True,
        "legacy_tke_floor_applied_fraction": 0.0,
        "legacy_tke_floor_primary_parallel_median": 0.0,
        "legacy_tke_floor_reference_parallel_median": 0.0,
        "legacy_tke_floor_missing_parallel_median": 0.0,
        "legacy_tke_floor_correction_ratio_median": 0.0,
        "legacy_tke_floor_correction_ratio_max": 0.0,
    }
    floor_correction = None
    if legacy_tke_gradient_floor:
        primary_accuracy, _primary_accuracy_details = (
            ensemble_train.common.accuracy_objective(
                prediction, target_norm, normalizer,
                resolved_point_weights))
        legacy_accuracy, _legacy_accuracy_details = (
            ensemble_train.common.accuracy_objective(
                prediction, target_norm, normalizer,
                resolved_legacy_weights))
        legacy_reference_loss = (
            supervised_loss + 0.25 * (
                legacy_accuracy - primary_accuracy))
        if legacy_tke_gradient_floor_on_members:
            legacy_member_loss, _legacy_bootstrap_fraction = (
                ensemble_train.bootstrap_member_point_loss(
                    joint_real.real_loss_args(), members, target_norm,
                    trajectory_ids, normalizer,
                    resolved_legacy_member_weights, iteration,
                    member_mask=member_mask))
            legacy_reference_loss = (
                legacy_reference_loss + legacy_member_loss - member_loss)
        tke_protection_loss = official_tke_score_loss(
            tke_floor_prediction(
                prediction, members, legacy_tke_gradient_floor_on_members),
            target_norm, normalizer)
        floor_correction, floor_metrics = output_legacy_tke_gradient_floor(
            supervised_loss, legacy_reference_loss,
            tke_protection_loss,
            members if legacy_tke_gradient_floor_on_members else prediction)
    full_backbone_native_feature_gradient_norm = 0.0
    full_backbone_aux_feature_gradient_norm = 0.0
    full_backbone_aux_to_native_feature_gradient_ratio = 0.0
    full_backbone_gradient_finite_nonzero = 0.0
    if (full_backbone_auxiliary is not None
            and iteration == full_backbone_gradient_audit_update):
        native_feature_gradients = torch.autograd.grad(
            native_supervised_loss, tuple(features), retain_graph=True,
            allow_unused=False)
        auxiliary_feature_gradients = torch.autograd.grad(
            weighted_full_backbone_auxiliary, tuple(features),
            retain_graph=True, allow_unused=False)
        full_backbone_native_feature_gradient_norm = (
            gradient_l2_from_tensors(native_feature_gradients))
        full_backbone_aux_feature_gradient_norm = (
            gradient_l2_from_tensors(auxiliary_feature_gradients))
        full_backbone_aux_to_native_feature_gradient_ratio = (
            full_backbone_aux_feature_gradient_norm / max(
                full_backbone_native_feature_gradient_norm,
                torch.finfo(torch.float32).tiny))
        full_backbone_gradient_finite_nonzero = float(
            full_backbone_native_feature_gradient_norm > 0.0
            and full_backbone_aux_feature_gradient_norm > 0.0
            and math.isfinite(
                full_backbone_aux_to_native_feature_gradient_ratio))

    auxiliary_loss = prediction.new_zeros(())
    if auxiliary_weight > 0.0:
        auxiliary_loss = truth_native_tke_relative_l2(
            prediction, target, normalizer)
    total_loss = (
        supervised_loss + soft_sps_weight * soft_sps_loss
        + auxiliary_weight * auxiliary_loss)
    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"non-finite NPT1 loss at update {iteration}")

    # Callers that partition an already-fixed batch (for example by fit AoA)
    # accumulate several complete objectives before one optimizer step.  Keep
    # reported metrics in their native units, but scale every gradient source
    # consistently so this is an actual weighted objective rather than only a
    # weighted point loss.
    supervised_loss.backward(gradient=supervised_loss.new_tensor(loss_scale),
                             retain_graph=True)
    if floor_correction is not None:
        (members if legacy_tke_gradient_floor_on_members else prediction
         ).backward(floor_correction * loss_scale, retain_graph=True)
    controller_parameters = tuple(
        model.energy_controller.parameters()
        if model.energy_controller is not None else ())
    trainable_controller = tuple(
        parameter for parameter in controller_parameters
        if parameter.requires_grad)
    native_controller_gradient_norm = (
        gradient_l2_from_parameters(trainable_controller)
        if audit_controller_gradients else 0.0)
    auxiliary_gradient_norm = 0.0
    auxiliary_to_native_ratio = 0.0
    if auxiliary_weight > 0.0:
        if not trainable_controller:
            raise RuntimeError("NPT1 TKE auxiliary has no trainable controller")
        auxiliary_gradients = torch.autograd.grad(
            loss_scale * auxiliary_weight * auxiliary_loss,
            trainable_controller, retain_graph=True, allow_unused=False)
        if audit_controller_gradients:
            auxiliary_gradient_norm = gradient_l2_from_tensors(
                auxiliary_gradients)
            auxiliary_to_native_ratio = (
                auxiliary_gradient_norm / max(
                    native_controller_gradient_norm,
                    torch.finfo(torch.float32).tiny))
        add_parameter_gradients(trainable_controller, auxiliary_gradients)
    else:
        auxiliary_gradients = ()

    interval_parameters = ensemble_train.uq_trainable_parameters(model)
    interval_gradients = torch.autograd.grad(
        loss_scale * soft_sps_weight * soft_sps_loss,
        interval_parameters, allow_unused=True)
    add_parameter_gradients(interval_parameters, interval_gradients)
    values = {
        "real_loss": float(total_loss.detach()),
        "real_supervised_loss": float(supervised_loss.detach()),
        "real_native_supervised_loss": float(
            native_supervised_loss.detach()),
        "real_point_loss": float(point_loss.detach()),
        "real_member_loss": float(member_loss.detach()),
        "real_quantile_loss": float(quantile_loss.detach()),
        "real_soft_sps_loss": float(soft_sps_loss.detach()),
        "real_soft_sps_score": float(soft_sps_score.detach()),
        "real_soft_sps_weight": soft_sps_weight,
        "real_bootstrap_fraction": float(bootstrap_fraction.detach()),
        "region_auxiliary_loss": float(region_auxiliary_loss.detach()),
        "region_auxiliary_weight": float(region_auxiliary_weight),
        "region_weighted_auxiliary_loss": float(
            weighted_region_auxiliary.detach()),
        "region_augmentation_enabled": float(
            input_region_augmentor is not None),
        "npt1_tke_aux_loss": float(auxiliary_loss.detach()),
        "npt1_tke_aux_weight": auxiliary_weight,
        "npt1_native_controller_gradient_norm": (
            native_controller_gradient_norm),
        "npt1_tke_aux_gradient_norm": auxiliary_gradient_norm,
        "npt1_tke_aux_to_native_gradient_ratio": auxiliary_to_native_ratio,
        "npt1_input_augmentation_enabled": float(any(augmentation.values()) and not discard_input_corruption),
        "point_accuracy_weight_rel": resolved_point_weights[0],
        "point_accuracy_weight_tke": resolved_point_weights[1],
        "point_accuracy_weight_mvpe": resolved_point_weights[2],
        "member_accuracy_weight_rel": resolved_member_weights[0],
        "member_accuracy_weight_tke": resolved_member_weights[1],
        "member_accuracy_weight_mvpe": resolved_member_weights[2],
        "legacy_tke_gradient_floor_enabled": float(
            legacy_tke_gradient_floor),
        "legacy_tke_gradient_floor_on_members": float(
            legacy_tke_gradient_floor_on_members),
        **floor_metrics,
        "full_backbone_auxiliary_loss": float(
            full_backbone_auxiliary_loss.detach()),
        "full_backbone_auxiliary_weight": full_backbone_auxiliary_weight,
        "full_backbone_weighted_auxiliary_loss": float(
            weighted_full_backbone_auxiliary.detach()),
        "full_backbone_native_feature_gradient_norm": (
            full_backbone_native_feature_gradient_norm),
        "full_backbone_aux_feature_gradient_norm": (
            full_backbone_aux_feature_gradient_norm),
        "full_backbone_aux_to_native_feature_gradient_ratio": (
            full_backbone_aux_to_native_feature_gradient_ratio),
        "full_backbone_gradient_finite_nonzero": (
            full_backbone_gradient_finite_nonzero),
        **{
            f"full_backbone_level_{index}_loss": float(value.detach())
            for index, value in enumerate(full_backbone_level_losses)
        },
        **{
            f"real_{key}": float(value.detach())
            for key, value in point_components.items()
        },
        **{
            f"region_augmentation_{key}": value
            for key, value in region_augmentation_metrics.items()
        },
        **{
            f"region_auxiliary_{key}": value
            for key, value in region_auxiliary_metrics.items()
        },
    }
    del (history, history_input, target, history_norm, target_norm, prediction, members,
         quantiles, lower, upper, point_loss, member_loss, quantile_loss,
         soft_sps_loss, native_supervised_loss, supervised_loss, total_loss,
         auxiliary_loss, full_backbone_auxiliary_loss,
         weighted_full_backbone_auxiliary, region_auxiliary_loss,
         weighted_region_auxiliary,
         auxiliary_gradients, interval_gradients)
    return values


def train_arm(
        model: EnsembleUNet2d, dataset, schedule: Mapping[str, object],
        normalizer: smu1.FoldNormalizer, device: torch.device,
        deadline: float, arm: str,
        ) -> tuple[dict, ema_lib.FP32EMAShadow]:
    trainable = configure_trainable(model, arm)
    tagged = ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_sampler=schedule["batches"], num_workers=0,
        pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX, eta_min=0.0)
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    manifest_names = [row["name"] for row in ema.parameter_manifest()]
    if manifest_names != [name for name, _ in model.named_parameters()]:
        raise RuntimeError("NPT1 EMA does not cover all named parameters")
    if any(value.device != device for value in ema._shadow.values()):
        raise RuntimeError("NPT1 EMA shadow is not on the training device")
    initial_controller = controller_state(model)
    initial_controller_sha = ema_lib.tensor_state_sha256(initial_controller)
    initial_front_sha = ema_lib.tensor_state_sha256(
        _controller_front_state(initial_controller))
    actual_rows = []
    history = []
    gradient_seen = {
        "legacy_nonzero": False,
        "controller_projection_nonzero": False,
        "controller_front_nonzero": False,
    }
    legacy_parameter_updated = False
    candidate_projection_updated_at_step1 = False
    gradient_ratios = []
    gradient_preflight = None
    started = time.perf_counter()
    expected_updates = int(schedule["updates"])
    validate_schedule(schedule)
    for iteration, batch in enumerate(loader, start=1):
        if iteration > expected_updates:
            break
        if time.perf_counter() > deadline:
            raise TimeoutError(f"NPT1 {arm} pair wall budget expired")
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            raise RuntimeError("NPT1 training batch is not exact tuple4")
        history_value, target, trajectory_ids, sample_indices = batch
        expected = schedule["rows"][iteration - 1]
        actual = {
            "sample_indices": sample_indices.to(dtype=torch.long).tolist(),
            "trajectory_ids": trajectory_ids.to(dtype=torch.long).tolist(),
            "bootstrap_mask": expected["bootstrap_mask"],
        }
        if actual != expected:
            raise RuntimeError(f"NPT1 paired stream mismatch at {arm} {iteration}")
        actual_rows.append(actual)
        frozen_mask = torch.as_tensor(
            expected["bootstrap_mask"], dtype=torch.bool)
        if (frozen_mask.device.type != "cpu"
                or frozen_mask.shape != (len(sample_indices), 4)
                or not bool(frozen_mask.any(dim=1).all())):
            raise RuntimeError("NPT1 frozen bootstrap mask contract changed")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = backward_real_supervised_npt1(
            model,
            (history_value, target, trajectory_ids, sample_indices),
            normalizer, device, iteration,
            auxiliary_weight=(
                tke_aux_weight(iteration) if arm == "candidate" else 0.0),
            member_mask=frozen_mask,
            audit_controller_gradients=(
                arm == "candidate"
                and iteration <= GRADIENT_PREFLIGHT_UPDATES))
        audit = _gradient_audit(model)
        if not audit["finite"]:
            raise FloatingPointError(f"NPT1 {arm} gradient is non-finite")
        gradient_seen["legacy_nonzero"] |= audit["legacy_nonzero"]
        gradient_seen["controller_projection_nonzero"] |= (
            audit["controller_projection_nonzero"])
        gradient_seen["controller_front_nonzero"] |= (
            audit["controller_front_nonzero"])
        if arm == "control" and not audit["controller_all_grad_none"]:
            raise RuntimeError("NPT1 control controller received a gradient")
        if (arm == "candidate" and iteration == 1
                and not audit["controller_projection_nonzero"]):
            raise RuntimeError(
                "NPT1 candidate zero projection lacks update1 gradient")
        if (arm == "candidate" and iteration == 2
                and not audit["controller_front_nonzero"]):
            raise GradientPreflightStop(
                "STOP_NPT1_GRADIENT_PREFLIGHT_FRONT_NOT_ACTIVE_AT_U2")
        if arm == "candidate" and iteration <= GRADIENT_PREFLIGHT_UPDATES:
            ratio = float(values["npt1_tke_aux_to_native_gradient_ratio"])
            native_norm = float(
                values["npt1_native_controller_gradient_norm"])
            auxiliary_norm = float(values["npt1_tke_aux_gradient_norm"])
            if (not all(math.isfinite(value) for value in (
                    ratio, native_norm, auxiliary_norm))
                    or native_norm <= 0.0 or auxiliary_norm <= 0.0):
                raise GradientPreflightStop(
                    "STOP_NPT1_GRADIENT_PREFLIGHT_NONFINITE_OR_ZERO")
            gradient_ratios.append(ratio)
            if iteration == GRADIENT_PREFLIGHT_UPDATES:
                median_ratio = float(statistics.median(gradient_ratios))
                ordered_ratios = sorted(gradient_ratios)
                p95_ratio = float(ordered_ratios[
                    math.ceil(0.95 * len(ordered_ratios)) - 1])
                gradient_preflight = {
                    "updates": GRADIENT_PREFLIGHT_UPDATES,
                    "samples": len(gradient_ratios),
                    "weighted_aux_to_native_controller_grad_ratio_median": (
                        median_ratio),
                    "ratio_min": float(min(gradient_ratios)),
                    "ratio_max": float(max(gradient_ratios)),
                    "ratio_p95": p95_ratio,
                    "gate": [GRADIENT_RATIO_MIN, GRADIENT_RATIO_MAX],
                    "pass": (
                        GRADIENT_RATIO_MIN <= median_ratio
                        <= GRADIENT_RATIO_MAX),
                }
                if not gradient_preflight["pass"]:
                    raise GradientPreflightStop(
                        "STOP_NPT1_GRADIENT_PREFLIGHT_RATIO "
                        f"median={median_ratio:.8g}")
        legacy_probe = None
        if not legacy_parameter_updated:
            for name, parameter in model.named_parameters():
                if (not name.startswith("energy_controller.")
                        and parameter.grad is not None
                        and bool(torch.count_nonzero(parameter.grad))):
                    legacy_probe = (parameter, parameter.detach().clone())
                    break
        projection_before = None
        if arm == "candidate" and iteration == 1:
            projection_before = OrderedDict(
                (name, value.detach().clone())
                for name, value in model.energy_controller.state_dict().items()
                if name in {"body.5.weight", "body.5.bias"})
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"NPT1 {arm} gradient norm is non-finite")
        optimizer.step()
        if legacy_probe is not None:
            parameter, before = legacy_probe
            legacy_parameter_updated |= not torch.equal(parameter.detach(), before)
        if projection_before is not None:
            projection_after = model.energy_controller.state_dict()
            candidate_projection_updated_at_step1 = any(
                not torch.equal(value, projection_after[name])
                for name, value in projection_before.items())
            if not candidate_projection_updated_at_step1:
                raise RuntimeError(
                    "NPT1 candidate zero projection did not update at step1")
        ema.update_after_optimizer_step(model, optimizer_step=iteration)
        scheduler.step()
        if (arm == "control"
                and not tensor_state_equal(initial_controller,
                                           controller_state(model))):
            raise RuntimeError("NPT1 control controller state changed")
        if iteration == 1 or iteration % 500 == 0 or iteration == expected_updates:
            history.append({
                "iteration": iteration,
                **{key: float(value) for key, value in values.items()},
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
            })
    if len(actual_rows) != expected_updates or ema.updates != expected_updates:
        raise RuntimeError(f"NPT1 {arm} did not complete fixed updates")
    actual_sha = dr1.canonical_sha256(actual_rows)
    if actual_sha != schedule["sha256"]:
        raise RuntimeError(f"NPT1 {arm} observed stream digest changed")
    if not gradient_seen["legacy_nonzero"]:
        raise RuntimeError(f"NPT1 {arm} legacy parameters saw no gradient")
    if not legacy_parameter_updated:
        raise RuntimeError(f"NPT1 {arm} legacy parameters did not update")
    final_controller = controller_state(model)
    final_controller_sha = ema_lib.tensor_state_sha256(final_controller)
    final_front_sha = ema_lib.tensor_state_sha256(
        _controller_front_state(final_controller))
    if arm == "candidate":
        if gradient_preflight is None:
            raise RuntimeError("NPT1 candidate lacks u200 gradient preflight")
        if (not gradient_seen["controller_projection_nonzero"]
                or not candidate_projection_updated_at_step1):
            raise RuntimeError("NPT1 candidate projection did not train")
        if final_controller_sha == initial_controller_sha:
            raise RuntimeError("NPT1 candidate controller did not update")
        if final_front_sha == initial_front_sha:
            raise RuntimeError("NPT1 candidate controller front did not update")
    elif final_controller_sha != initial_controller_sha:
        raise RuntimeError("NPT1 control controller final state changed")
    return ({
        "arm": arm,
        "updates": expected_updates,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "scheduler_t_max": SCHEDULER_T_MAX,
        "ema_decay": EMA_DECAY,
        "ema_updates": ema.updates,
        "planned_stream_sha256": schedule["sha256"],
        "actual_stream_sha256": actual_sha,
        "gradient_seen": gradient_seen,
        "legacy_parameter_updated": legacy_parameter_updated,
        "candidate_projection_updated_at_step1": (
            candidate_projection_updated_at_step1 if arm == "candidate"
            else None),
        "tke_aux_gradient_preflight": gradient_preflight,
        "initial_controller_state_sha256": initial_controller_sha,
        "final_controller_state_sha256": final_controller_sha,
        "initial_controller_front_state_sha256": initial_front_sha,
        "final_controller_front_state_sha256": final_front_sha,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }, ema)


class FourEndpointAccessAudit:
    """Guard one prefix load and four held evaluation passes."""

    def __init__(self, held_aoa: int):
        if held_aoa not in FOLD_ORDER:
            raise ValueError(f"unexpected held AoA {held_aoa}")
        self.held_aoa = held_aoa
        self.seals: dict[str, dict] = {}
        self.events: list[dict] = []
        self.opened: set[str] = set()
        self.evaluations: list[str] = []

    @property
    def all_sealed(self) -> bool:
        return set(self.seals) == set(ENDPOINT_ORDER)

    def seal(self, endpoint: str, seal: Mapping[str, object]) -> None:
        if endpoint not in ENDPOINT_ORDER or endpoint in self.seals:
            raise RuntimeError(f"invalid or duplicate endpoint seal {endpoint}")
        self.seals[endpoint] = {
            "path": str(seal["path"]),
            "file_sha256": dr1.normalized_sha256(
                str(seal["file_sha256"]), f"{endpoint} file SHA"),
            "model_state_sha256": dr1.normalized_sha256(
                str(seal["model_state_sha256"]), f"{endpoint} state SHA"),
            "sealed_monotonic_ns": time.monotonic_ns(),
        }

    def before_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        aoa = int(row["aoa"])
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded H5 reached NPT1 opener")
        if phase == "fit":
            if aoa == self.held_aoa or self.seals:
                raise RuntimeError("invalid NPT1 fit-prefix open")
        elif phase == "held":
            if aoa != self.held_aoa or not self.all_sealed:
                raise RuntimeError("held prefix opened before four seals")
        else:
            raise ValueError(f"unknown NPT1 access phase {phase}")

    def record_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if filename in self.opened:
            raise RuntimeError(f"H5 file opened twice: {filename}")
        self.opened.add(filename)
        self.events.append({
            "file": filename,
            "aoa": int(row["aoa"]),
            "phase": phase,
            "slice": "u/v[:val_boundary,::2,::2]",
            "max_read_frame_exclusive": int(row["val_boundary"]),
            "four_endpoints_sealed_before_open": self.all_sealed,
        })

    def record_evaluation(self, endpoint: str) -> None:
        if not self.all_sealed:
            raise RuntimeError("held evaluation began before four seals")
        expected = ENDPOINT_ORDER[len(self.evaluations)]
        if endpoint != expected:
            raise RuntimeError(
                f"held evaluation order {endpoint} != expected {expected}")
        self.evaluations.append(endpoint)

    def report(self, expected_allowlist: Sequence[str]) -> dict:
        held_events = [row for row in self.events if row["phase"] == "held"]
        return {
            "opened_file_allowlist": sorted(self.opened),
            "expected_file_allowlist": sorted(expected_allowlist),
            "allowlist_exact": sorted(self.opened) == sorted(expected_allowlist),
            "tail_arrays_read": 0,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "four_endpoint_seals": self.seals,
            "held_first_open_after_four_seals": bool(
                held_events
                and held_events[0]["four_endpoints_sealed_before_open"]),
            "held_evaluation_order": list(self.evaluations),
            "held_complete_passes": len(self.evaluations),
            "events": self.events,
        }


def save_endpoint(
        path: Path, model: EnsembleUNet2d, state: Mapping[str, torch.Tensor],
        *, arm: str, endpoint_kind: str, held_aoa: int,
        start_report: Mapping[str, object],
        normalizer_report: Mapping[str, object], stream_sha256: str,
        ema_metadata: Mapping[str, object] | None,
        ) -> dict:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite NPT1 endpoint {path}")
    state = OrderedDict(
        (name, value.detach()) for name, value in state.items())
    if not state or any(value.device.type != "cpu" for value in state.values()):
        raise ValueError("NPT1 endpoint state must already be complete CPU state")
    state_sha = dr1.model_state_sha256(state)
    endpoint = f"{arm}_{endpoint_kind}"
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": UPDATES,
        "arm": arm,
        "endpoint_kind": endpoint_kind,
        "endpoint": endpoint,
        "held_aoa": held_aoa,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start": dict(start_report),
        "normalizer": dict(normalizer_report),
        "sample_stream_sha256": stream_sha256,
        "ema": None if ema_metadata is None else dict(ema_metadata),
        "optimizer_state_saved": False,
        "held_prefix_opened_before_seal": False,
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "endpoint": endpoint,
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": dr1.canonical_sha256(payload["model_config"]),
        "iteration": UPDATES,
        "optimizer_state_saved": False,
    }


def load_endpoint_model(
        seal: Mapping[str, object], normalizer: smu1.FoldNormalizer,
        device: torch.device) -> EnsembleUNet2d:
    path = Path(str(seal["path"]))
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError(f"NPT1 endpoint file changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENDPOINT_SCHEMA:
        raise ValueError(f"invalid NPT1 endpoint schema: {path}")
    state = payload.get("model_state_dict")
    if (not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]
            or payload.get("model_state_sha256") != seal["model_state_sha256"]):
        raise RuntimeError(f"NPT1 endpoint state changed: {path}")
    if dr1.canonical_sha256(payload["model_config"]) != (
            seal["model_config_sha256"]):
        raise RuntimeError(f"NPT1 endpoint config changed: {path}")
    if payload.get("normalizer") != normalizer_payload(normalizer):
        raise RuntimeError(f"NPT1 endpoint normalizer changed: {path}")
    stored = dict(payload["model_config"])
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError("NPT1 endpoint is not an ensemble U-Net")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored)).float()
    model.load_state_dict(state, strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def metric_comparison(candidate: Mapping[str, float],
                      control: Mapping[str, float]) -> dict:
    delta = smu1.metric_delta(dict(candidate), dict(control))
    direct = sum(
        DIRECT_COEFFICIENTS[key] * delta[key]
        for key in DIRECT_COEFFICIENTS)
    non_time = direct + SPS_COEFFICIENT * delta["native_sps_score"]
    return {
        "control": dict(control),
        "candidate": dict(candidate),
        "delta": delta,
        "direct_delta": float(direct),
        "non_time_delta": float(non_time),
    }


def fold_gate(primary: Mapping[str, object]) -> dict:
    non_time = float(primary["non_time_delta"])
    passed = non_time >= FOLD_NON_TIME_FLOOR
    return {
        "pass": passed,
        "condition": "DeltaNonTime >= -0.05",
        "decision": (
            "CONTINUE_NPT1_LOAO" if passed
            else "STOP_NPT1_FOLD_HARD_FLOOR"),
    }


def pooled_gate(pooled: Mapping[str, object], rows: Sequence[Mapping]) -> dict:
    delta = pooled["delta"]
    directional_conditions = {
        "tke": (
            float(delta["tke_rel_l2_score"]) >= PILOT_TKE_DELTA_MIN),
        "sps": float(delta["native_sps_score"]) >= PILOT_SPS_DELTA_MIN,
        "non_time": (
            float(pooled["non_time_delta"]) >= PILOT_NON_TIME_DELTA_MIN),
        "rel": (
            float(delta["rel_l2_score"]) >= PILOT_COMPONENT_DELTA_FLOOR),
        "mvpe": (
            float(delta["mvpe_rel_l2_score"]) >= PILOT_COMPONENT_DELTA_FLOOR),
        "all_fold_non_time_positive": min(
            float(row["primary"]["non_time_delta"]) for row in rows)
            > 0.0,
    }
    projection = official_projection(pooled)
    strong_conditions = {
        "tke": float(delta["tke_rel_l2_score"]) >= STRONG_TKE_DELTA_MIN,
        "sps": float(delta["native_sps_score"]) >= STRONG_SPS_DELTA_MIN,
        "non_time": (
            float(pooled["non_time_delta"]) >= STRONG_NON_TIME_DELTA_MIN),
        "rel": float(delta["rel_l2_score"]) >= STRONG_REL_DELTA_FLOOR,
        "mvpe": float(delta["mvpe_rel_l2_score"]) >= STRONG_MVPE_DELTA_FLOOR,
        "each_fold_tke_positive": all(
            float(row["primary"]["delta"]["tke_rel_l2_score"]) > 0.0
            for row in rows),
        "each_fold_sps_positive": all(
            float(row["primary"]["delta"]["native_sps_score"]) > 0.0
            for row in rows),
        "each_fold_non_time_positive": all(
            float(row["primary"]["non_time_delta"]) > 0.0
            for row in rows),
        "conservative_projection": projection["pass"],
    }
    return {
        "pass": all(directional_conditions.values()),
        "conditions": directional_conditions,
        "strong_81_projection_diagnostic_pass": all(
            strong_conditions.values()),
        "strong_81_projection_diagnostic_conditions": strong_conditions,
        "two_fold_projection_diagnostic": projection,
    }


def official_final(metrics: Mapping[str, float]) -> float:
    values = (
        (0.5, metrics["rel_l2_score"]),
        (0.1, metrics["tke_rel_l2_score"]),
        (0.1, metrics["mvpe_rel_l2_score"]),
        (0.1, metrics["time_score"]),
        (0.2, metrics["native_sps_score"]),
    )
    if any(not math.isfinite(float(value)) or value <= 0 for _weight, value in values):
        raise ValueError("official projection metrics must be finite/positive")
    return float(sum(
        weight * float(value) ** OFFICIAL_Q for weight, value in values
    ) ** (1.0 / OFFICIAL_Q))


def official_projection(pooled: Mapping[str, object]) -> dict:
    delta = pooled["delta"]
    projected = dict(OFFICIAL_C208)
    for key in (
            "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
            "native_sps_score"):
        projected[key] += float(delta[key])
    final = official_final(projected)
    conservative = final - CONSERVATIVE_BIAS
    return {
        "official_c208_anchor": OFFICIAL_C208,
        "projected_metrics": projected,
        "q": OFFICIAL_Q,
        "projected_final": final,
        "conservative_bias": CONSERVATIVE_BIAS,
        "conservative_final": conservative,
        "threshold": CONSERVATIVE_FINAL_MIN,
        "pass": conservative >= CONSERVATIVE_FINAL_MIN,
    }


def _validate_hardware(device: torch.device, hardware_class: str) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal NPT1 fold requires CUDA")
    name = torch.cuda.get_device_name(device).lower()
    expected = "4090" if hardware_class == "rtx4090" else "4080"
    if expected not in name or (
            hardware_class == "rtx4080_super" and "super" not in name):
        raise RuntimeError(
            f"NPT1 hardware class {hardware_class} does not match {name}")
    return PAIR_WALL_BUDGET_SECONDS[hardware_class]


def disk_preflight(output_dir: Path) -> dict:
    probe = output_dir.parent.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError("no existing parent for NPT1 output")
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"NPT1 output disk has {free} bytes free; requires {MIN_FREE_BYTES}")
    return {
        "probe": str(probe),
        "free_bytes_before_run": free,
        "minimum_free_bytes": MIN_FREE_BYTES,
    }


def validate_prior_authorization(
        held_aoa: int,
        prior_aggregate: Sequence[str] | None,
        ) -> dict:
    """Authorize only the preregistered concurrent two-fold pilot."""
    if held_aoa not in FOLD_ORDER:
        raise ValueError("NPT1 fold is outside the two-fold pilot")
    if prior_aggregate is not None:
        raise ValueError("NPT1 two-fold pilot must not receive prior aggregate")
    return {
        "kind": "preauthorized_wave1",
        "wave": list(WAVE_ORDER[0]),
        "completed_folds": 0,
    }


def run_fold(args: argparse.Namespace) -> dict:
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError("formal NPT1 fold is on the wrong branch")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    if args.held_aoa not in FOLD_ORDER:
        raise ValueError("NPT1 fold requires one frozen held AoA")
    prior_authorization = validate_prior_authorization(
        args.held_aoa, args.prior_aggregate)
    device = torch.device(args.device)
    budget = _validate_hardware(device, args.hardware_class)
    disk_report = disk_preflight(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + budget
    torch.set_float32_matmul_precision("high")

    if dr1.sha256(args.clean_manifest) != FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("NPT1 clean81 manifest file SHA changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    expected_fit = dr1.EXPECTED_FIT_COUNTS[args.held_aoa]
    expected_held = dr1.EXPECTED_AOA_COUNTS[args.held_aoa]
    if sum(len(row["train_starts"]) for row in fit_rows) != expected_fit:
        raise RuntimeError("NPT1 other-four fit count changed")
    if sum(len(row["train_starts"]) for row in held_rows) != expected_held:
        raise RuntimeError("NPT1 held count changed")

    access = FourEndpointAccessAudit(args.held_aoa)
    fit_trajectories, fit_metadata, fit_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", access)
    if len(fit_windows) != expected_fit:
        raise RuntimeError("NPT1 loaded fit count changed")
    fit_dataset = ensemble_train.common.repro.WindowDataset(
        fit_trajectories, fit_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, fit_windows)
    statistics = smu1.fold_statistics(sufficient, range(len(fit_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    normalizer_report = normalizer_payload(normalizer)
    schedule = smu1.paired_batch_schedule(
        range(len(fit_windows)), fit_windows, updates=UPDATES, seed=SEED)
    schedule["updates"] = UPDATES

    payload, start_report = validate_start_checkpoint(
        args.start_sim, args.start_sim_sha256)
    arm_reports: dict[str, dict] = {}
    update0_by_arm = {}
    start_state_sha = None
    for arm in ARM_ORDER[args.held_aoa]:
        ensemble_train.seed_everything(SEED)
        model, migration = build_enabled_model(
            payload, normalizer, device)
        current_start_sha = dr1.model_state_sha256(model.state_dict())
        if start_state_sha is None:
            start_state_sha = current_start_sha
        elif current_start_sha != start_state_sha:
            raise RuntimeError("NPT1 arms do not have the same enabled start")
        update0_by_arm[arm] = capture_update0(
            model, fit_dataset, 0, normalizer, device)
        training, ema = train_arm(
            model, fit_dataset, schedule, normalizer, device,
            deadline, arm)

        raw_state = ensemble_train.common.checkpoint_state(model)
        raw_path = args.output_dir / f"{arm}_raw_u4000.pth"
        raw_seal = save_endpoint(
            raw_path, model, raw_state, arm=arm, endpoint_kind="raw",
            held_aoa=args.held_aoa, start_report=start_report,
            normalizer_report=normalizer_report,
            stream_sha256=schedule["sha256"], ema_metadata=None)
        access.seal(f"{arm}_raw", raw_seal)
        del raw_state

        raw_sha_before_ema_seal = dr1.model_state_sha256(model.state_dict())
        if raw_seal["model_state_sha256"] != raw_sha_before_ema_seal:
            raise RuntimeError("NPT1 raw endpoint differs from live raw model")
        frozen_ema_canonicalization = None
        if arm == "control":
            frozen_ema_canonicalization = canonicalize_frozen_controller_ema(
                ema, model,
                training["initial_controller_state_sha256"])
        ema_seal_payload = ema.seal(model)
        raw_sha_after_ema_seal = dr1.model_state_sha256(model.state_dict())
        if raw_sha_after_ema_seal != raw_sha_before_ema_seal:
            raise RuntimeError("NPT1 EMA seal mutated the raw model")
        ema_state = ema_seal_payload.pop("model_state")
        if (int(ema_seal_payload["updates"]) != UPDATES
                or float(ema_seal_payload["decay"]) != EMA_DECAY):
            raise RuntimeError("NPT1 EMA seal count/decay changed")
        ema_path = args.output_dir / f"{arm}_ema999_u4000.pth"
        ema_seal = save_endpoint(
            ema_path, model, ema_state, arm=arm, endpoint_kind="ema",
            held_aoa=args.held_aoa, start_report=start_report,
            normalizer_report=normalizer_report,
            stream_sha256=schedule["sha256"],
            ema_metadata=ema_seal_payload)
        access.seal(f"{arm}_ema", ema_seal)
        if arm == "control":
            control_raw_controller = controller_state(model)
            ema_controller = OrderedDict(
                (name.removeprefix("energy_controller."), value)
                for name, value in ema_state.items()
                if name.startswith("energy_controller."))
            if (ema_lib.tensor_state_sha256(control_raw_controller)
                    != training["initial_controller_state_sha256"]):
                raise RuntimeError("NPT1 control raw controller changed")
            if (ema_lib.tensor_state_sha256(ema_controller)
                    != training["initial_controller_state_sha256"]):
                raise RuntimeError("NPT1 control EMA controller changed")
        arm_reports[arm] = {
            "migration": migration,
            "enabled_start_model_state_sha256": current_start_sha,
            "training": training,
            "raw_state_sha256_before_ema_seal": raw_sha_before_ema_seal,
            "raw_state_sha256_after_ema_seal": raw_sha_after_ema_seal,
            "frozen_controller_ema_canonicalization": (
                frozen_ema_canonicalization),
            "raw_seal": raw_seal,
            "ema_seal": ema_seal,
        }
        del model, ema, ema_state, ema_seal_payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del payload
    gc.collect()

    update0 = update0_bitwise_identity(
        update0_by_arm["control"], update0_by_arm["candidate"])
    if not update0["pass"]:
        raise RuntimeError(
            "NPT1 update0 point/member/quantile/bounds are not bitwise")
    controller_starts = {
        arm_reports[arm]["training"]["initial_controller_state_sha256"]
        for arm in ("control", "candidate")}
    if len(controller_starts) != 1:
        raise RuntimeError("NPT1 arm update0 controller states differ")
    if not access.all_sealed:
        raise RuntimeError("NPT1 held load began without four endpoint seals")

    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", access)
    if len(held_windows) != expected_held:
        raise RuntimeError("NPT1 loaded held count changed")
    held_dataset = ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    evaluate_indices = list(range(len(held_windows)))
    endpoint_reports = {}
    for endpoint in ENDPOINT_ORDER:
        access.record_evaluation(endpoint)
        arm, endpoint_kind = endpoint.split("_", maxsplit=1)
        seal = arm_reports[arm][f"{endpoint_kind}_seal"]
        model = load_endpoint_model(seal, normalizer, device)
        metrics, metric_state, held_audit = smu1.evaluate_model(
            model, held_dataset, evaluate_indices, normalizer,
            device, deadline)
        endpoint_reports[endpoint] = {
            "seal": seal,
            "metrics": metrics,
            "metric_state": metric_state,
            "held_target_audit": held_audit,
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    audits = [
        endpoint_reports[endpoint]["held_target_audit"]
        for endpoint in ENDPOINT_ORDER]
    if any(value != audits[0] for value in audits[1:]):
        raise RuntimeError("NPT1 four held target/mask audits differ")

    raw = metric_comparison(
        endpoint_reports["candidate_raw"]["metrics"],
        endpoint_reports["control_raw"]["metrics"])
    primary = metric_comparison(
        endpoint_reports["candidate_ema"]["metrics"],
        endpoint_reports["control_ema"]["metrics"])
    gate = fold_gate(primary)
    access_report = access.report([row["file"] for row in rows])
    access_ok = bool(
        access_report["allowlist_exact"]
        and access_report["tail_arrays_read"] == 0
        and not access_report["excluded_opened"]
        and access_report["held_first_open_after_four_seals"]
        and access_report["held_evaluation_order"] == list(ENDPOINT_ORDER)
        and access_report["held_complete_passes"] == 4)
    if not access_ok:
        raise RuntimeError("NPT1 prefix access contract failed")
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "mode": "fold_worker",
        "held_aoa": args.held_aoa,
        "prior_authorization": prior_authorization,
        "arm_order": list(ARM_ORDER[args.held_aoa]),
        "endpoint_order": list(ENDPOINT_ORDER),
        "fit_samples": len(fit_windows),
        "samples": len(held_windows),
        "manifest": manifest_report,
        "manifest_file_sha256": FORMAL_MANIFEST_FILE_SHA256,
        "normalizer": {
            **normalizer_report,
            "kind": "other-four optimizer-prefix windows only",
            "held_target_in_normalizer": False,
        },
        "sample_stream_sha256": schedule["sha256"],
        "fit_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(fit_windows)), fit_windows, fit_metadata),
        "held_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(held_windows)), held_windows, held_metadata),
        "start": start_report,
        "enabled_start_model_state_sha256": start_state_sha,
        "update0": update0,
        "arms": arm_reports,
        "endpoints": endpoint_reports,
        "raw_diagnostic": raw,
        "primary": primary,
        "fold_gate": gate,
        "prefix_access_audit": access_report,
        "git": identity,
        "hardware_class": args.hardware_class,
        "wall_budget_seconds": budget,
        "disk_preflight": disk_report,
        "elapsed_seconds": time.perf_counter() - started,
        "release_checkpoint_written": False,
        "submission_authorized": False,
        "decision": gate["decision"],
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    return result


def validate_fold_rows(rows: Sequence[Mapping]) -> None:
    observed = [int(row["held_aoa"]) for row in rows]
    if observed != list(FOLD_ORDER[:len(rows)]):
        raise ValueError("NPT1 fold results are not a fixed-order prefix")
    for row in rows:
        held = int(row["held_aoa"])
        if row.get("schema") != SCHEMA or row.get("experiment") != EXPERIMENT:
            raise ValueError("NPT1 fold result schema/experiment changed")
        if row.get("contract_sha256") != CONTRACT_SHA256:
            raise RuntimeError("NPT1 fold contract digest changed")
        if int(row["samples"]) != dr1.EXPECTED_AOA_COUNTS[held]:
            raise RuntimeError("NPT1 fold held sample count changed")
        if row.get("arm_order") != list(ARM_ORDER[held]):
            raise RuntimeError("NPT1 fold arm order changed")
        if row["endpoint_order"] != list(ENDPOINT_ORDER):
            raise RuntimeError("NPT1 endpoint order changed")
        if (row.get("manifest_file_sha256") != FORMAL_MANIFEST_FILE_SHA256
                or row["manifest"].get("semantic_sha256")
                != FORMAL_MANIFEST_SEMANTIC_SHA256
                or row["manifest"].get("windows") != dr1.EXPECTED_WINDOWS
                or row["manifest"].get("excluded") != dr1.EXCLUDED_BASENAME):
            raise RuntimeError("NPT1 fold manifest contract changed")
        if row["start"].get("checkpoint_sha256") != FORMAL_START_SHA256:
            raise RuntimeError("NPT1 fold start checkpoint changed")
        authorization = row.get("prior_authorization", {})
        if authorization != {
                "kind": "preauthorized_wave1",
                "wave": list(WAVE_ORDER[0]),
                "completed_folds": 0}:
            raise RuntimeError("NPT1 pilot authorization changed")
        update0 = row.get("update0", {})
        if (update0.get("pass") is not True
                or update0.get("checks") != {
                    "point": True, "members": True, "quantiles": True,
                    "lower": True, "upper": True}):
            raise RuntimeError("NPT1 fold update0 identity failed")
        access = row["prefix_access_audit"]
        if (access["tail_arrays_read"] != 0 or access["excluded_opened"]
                or access.get("allowlist_exact") is not True
                or access.get("held_first_open_after_four_seals") is not True
                or access.get("held_evaluation_order") != list(ENDPOINT_ORDER)
                or access.get("held_complete_passes") != 4
                or set(access.get("four_endpoint_seals", {}))
                != set(ENDPOINT_ORDER)):
            raise RuntimeError("NPT1 fold access audit failed")
        control = row["arms"]["control"]
        candidate = row["arms"]["candidate"]
        if (control["enabled_start_model_state_sha256"]
                != candidate["enabled_start_model_state_sha256"]
                or control["training"]["initial_controller_state_sha256"]
                != candidate["training"]["initial_controller_state_sha256"]):
            raise RuntimeError("NPT1 paired arm initialization changed")
        if (control["training"]["final_controller_state_sha256"]
                != control["training"]["initial_controller_state_sha256"]
                or candidate["training"]["final_controller_state_sha256"]
                == candidate["training"]["initial_controller_state_sha256"]
                or candidate["training"]["final_controller_front_state_sha256"]
                == candidate["training"]["initial_controller_front_state_sha256"]):
            raise RuntimeError("NPT1 controller trainability evidence failed")
        preflight = candidate["training"].get("tke_aux_gradient_preflight")
        if (not isinstance(preflight, Mapping)
                or preflight.get("pass") is not True
                or int(preflight.get("updates", -1))
                != GRADIENT_PREFLIGHT_UPDATES
                or int(preflight.get("samples", -1))
                != GRADIENT_PREFLIGHT_UPDATES
                or control["training"].get("tke_aux_gradient_preflight")
                is not None):
            raise RuntimeError("NPT1 gradient preflight evidence failed")
        canonicalization = control.get(
            "frozen_controller_ema_canonicalization", {})
        if (canonicalization.get("applied") is not True
                or canonicalization.get("exact_after") is not True
                or canonicalization.get("raw_sha256")
                != control["training"]["initial_controller_state_sha256"]
                or canonicalization.get("after_sha256")
                != canonicalization.get("raw_sha256")
                or candidate.get("frozen_controller_ema_canonicalization")
                is not None):
            raise RuntimeError(
                "NPT1 frozen controller EMA canonicalization changed")
        for arm in (control, candidate):
            if (arm.get("raw_state_sha256_before_ema_seal")
                    != arm.get("raw_state_sha256_after_ema_seal")
                    or arm["raw_seal"]["model_state_sha256"]
                    != arm["raw_state_sha256_before_ema_seal"]
                    or arm["training"].get("legacy_parameter_updated") is not True):
                raise RuntimeError("NPT1 raw/EMA or legacy update audit failed")
        streams = {
            row["sample_stream_sha256"],
            row["arms"]["control"]["training"]["actual_stream_sha256"],
            row["arms"]["candidate"]["training"]["actual_stream_sha256"],
        }
        if len(streams) != 1:
            raise RuntimeError("NPT1 paired fold streams differ")
        recomputed_raw = metric_comparison(
            row["endpoints"]["candidate_raw"]["metrics"],
            row["endpoints"]["control_raw"]["metrics"])
        recomputed_primary = metric_comparison(
            row["endpoints"]["candidate_ema"]["metrics"],
            row["endpoints"]["control_ema"]["metrics"])
        if (row.get("raw_diagnostic") != recomputed_raw
                or row.get("primary") != recomputed_primary
                or row.get("fold_gate") != fold_gate(recomputed_primary)):
            raise RuntimeError("NPT1 comparison endpoint binding changed")


def _stop_decision(rows: Sequence[Mapping]) -> dict | None:
    below_floor = [
        int(row["held_aoa"]) for row in rows
        if float(row["primary"]["non_time_delta"])
        < FOLD_NON_TIME_FLOOR]
    if below_floor:
        return {
            "decision": "STOP_NPT1_FOLD_HARD_FLOOR",
            "completed_folds": len(rows),
            "below_floor_aoas": below_floor,
        }
    if len(rows) >= 2 and all(
            float(row["primary"]["non_time_delta"]) <= 0.0
            for row in rows[:2]):
        return {
            "decision": "STOP_NPT1_FIRST_TWO_NONPOSITIVE",
            "completed_folds": len(rows),
        }
    return None


def aggregate_fold_rows(rows: Sequence[Mapping]) -> dict:
    if len(rows) != len(FOLD_ORDER):
        raise ValueError("NPT1 aggregator requires the complete two-fold pilot")
    validate_fold_rows(rows)
    stop = _stop_decision(rows)
    if stop is not None:
        return stop
    control_ema_state = smu1.merge_metric_states([
        row["endpoints"]["control_ema"]["metric_state"] for row in rows])
    candidate_ema_state = smu1.merge_metric_states([
        row["endpoints"]["candidate_ema"]["metric_state"] for row in rows])
    control_raw_state = smu1.merge_metric_states([
        row["endpoints"]["control_raw"]["metric_state"] for row in rows])
    candidate_raw_state = smu1.merge_metric_states([
        row["endpoints"]["candidate_raw"]["metric_state"] for row in rows])
    expected_samples = sum(int(row["samples"]) for row in rows)
    if any(int(state["samples"]) != expected_samples for state in (
            control_ema_state, candidate_ema_state,
            control_raw_state, candidate_raw_state)):
        raise RuntimeError("NPT1 pooled states do not cover both pilot folds")
    pooled = metric_comparison(
        smu1.finalize_metric_state(candidate_ema_state),
        smu1.finalize_metric_state(control_ema_state))
    pooled_raw = metric_comparison(
        smu1.finalize_metric_state(candidate_raw_state),
        smu1.finalize_metric_state(control_raw_state))
    gate = pooled_gate(pooled, rows)
    if not gate["pass"]:
        decision = "STOP_NPT1_TWO_FOLD_FUNCTIONAL_GATE"
    elif gate["strong_81_projection_diagnostic_pass"]:
        decision = (
            "GO_NPT1_STRONG_PROJECTED_TO_NPUSER_INSPIRED_B210_BUNDLE")
    else:
        decision = "GO_NPT1_DIRECTIONAL_TO_NPUSER_INSPIRED_B210_BUNDLE"
    return {
        "decision": decision,
        "completed_folds": len(rows),
        "pooled_control_ema_state": control_ema_state,
        "pooled_candidate_ema_state": candidate_ema_state,
        "pooled": pooled,
        "pooled_raw_diagnostic": pooled_raw,
        "gate": gate,
        "official_projection": gate["two_fold_projection_diagnostic"],
        "projection_reason": (
            "two-fold mechanism pilot does not authorize release projection"),
    }


def aggregate_from_files(pairs: Sequence[Sequence[str]]) -> dict:
    rows = []
    sources = []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected = dr1.normalized_sha256(raw_sha, "fold result SHA")
        actual = dr1.sha256(path)
        if actual != expected:
            raise RuntimeError(f"NPT1 fold result SHA changed: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("NPT1 fold result must be a JSON mapping")
        rows.append(value)
        sources.append({"path": str(path.resolve()), "sha256": actual})
    aggregation = aggregate_fold_rows(rows)
    return {
        "schema": AGGREGATE_SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "mode": "aggregate",
        "sources": sources,
        "rows": rows,
        **aggregation,
        "fixed725_accessed": False,
        "clean716_accessed": False,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold-worker", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--held-aoa", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256", default=FORMAL_START_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hardware-class", choices=tuple(PAIR_WALL_BUDGET_SECONDS))
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    parser.add_argument(
        "--prior-aggregate", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument(
        "--fold-result", action="append", nargs=2,
        metavar=("PATH", "SHA256"))
    parser.add_argument("--aggregate-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.start_sim_sha256 = dr1.normalized_sha256(
        args.start_sim_sha256, "start checkpoint SHA")
    if args.start_sim_sha256 != FORMAL_START_SHA256:
        raise ValueError("formal NPT1 requires frozen SHA6674")
    if args.fold_worker:
        required = (
            args.held_aoa is not None, args.output_dir is not None,
            args.clean_manifest is not None, args.start_sim is not None,
            args.hardware_class is not None, args.execute_formal_gpu,
            args.acknowledge_protocol == FORMAL_ACKNOWLEDGEMENT,
        )
        if not all(required):
            raise ValueError("formal NPT1 fold arguments are incomplete")
        if args.fold_result or args.aggregate_output:
            raise ValueError("fold worker cannot receive aggregate arguments")
        if args.held_aoa not in FOLD_ORDER:
            raise ValueError("NPT1 held AoA is outside the two-fold pilot")
        if args.prior_aggregate is not None:
            raise ValueError("NPT1 two-fold pilot cannot receive prior aggregate")
    elif args.aggregate_only:
        if not args.fold_result or args.aggregate_output is None:
            raise ValueError("NPT1 aggregate mode requires inputs and output")
        if args.aggregate_output.exists():
            raise FileExistsError(args.aggregate_output)
        if args.prior_aggregate is not None:
            raise ValueError("aggregate mode cannot receive prior aggregate")
    elif any((args.fold_result, args.aggregate_output,
              args.execute_formal_gpu, args.prior_aggregate)):
        raise ValueError("contract-only mode cannot execute work")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        result = {
            "experiment": EXPERIMENT,
            "contract": CONTRACT,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
        }
    elif args.aggregate_only:
        result = aggregate_from_files(args.fold_result)
        dr1.write_json_atomic(args.aggregate_output, result)
    else:
        try:
            result = run_fold(args)
        except GradientPreflightStop as error:
            failure = {
                "schema": SCHEMA,
                "experiment": EXPERIMENT,
                "mode": "fold_worker",
                "decision": "STOP_NPT1_GRADIENT_PREFLIGHT",
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry_started": False,
                "release_checkpoint_written": False,
                "submission_authorized": False,
            }
            if args.output_dir is not None and args.output_dir.is_dir():
                dr1.write_json_atomic(args.output_dir / "failure.json", failure)
            result = failure
        except Exception as error:
            failure = {
                "schema": SCHEMA,
                "experiment": EXPERIMENT,
                "mode": "fold_worker",
                "decision": "STOP_NPT1_INFRASTRUCTURE_OR_CONTRACT",
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry_started": False,
                "release_checkpoint_written": False,
                "submission_authorized": False,
            }
            if args.output_dir is not None and args.output_dir.is_dir():
                dr1.write_json_atomic(args.output_dir / "failure.json", failure)
            raise
    print(json.dumps({
        "event": "c208_npt1_complete",
        "decision": result["decision"],
    }), flush=True)


if __name__ == "__main__":
    main()
