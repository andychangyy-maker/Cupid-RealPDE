#!/usr/bin/env python3
"""Train the matched causal/spatial soft-residual pair on Sim then Real4k.

The frozen full-prefix native-UQ B210 is the exact identity anchor in both
stages.  Each arm trains only a 41,671-parameter history specialist and two
independent zero-initialized u/v gates.  Sim uses all released stride-20
windows after moment mapping into the Real coordinate system.  Real uses the
same clean81 stride-5/seed-4/augmentation stream as the mature Real4k recipe.
No clean716 path is accepted here and no score-based endpoint selection occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.causal_residual_soft_gate import (  # noqa: E402
    CausalResidualSoftGate,
    CausalResidualSoftGateConfig,
    FrozenB210CausalResidualSoftGate,
)
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_causal_phase_direction_real4k as phase_train  # noqa: E402
import train_b210_unified_temporal_state_sim_discovery as sim_data  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real4k  # noqa: E402


SCHEMA = "b210_causal_residual_soft_gate_lifecycle_v1"
ENDPOINT_SCHEMA = "b210_causal_residual_soft_gate_endpoint_v1"
SIM_ACKNOWLEDGEMENT = "RUN_B210_CAUSAL_RESIDUAL_SOFT_GATE_SIM4K_V1"
REAL_ACKNOWLEDGEMENT = "RUN_B210_CAUSAL_RESIDUAL_SOFT_GATE_REAL4K_V1"
ARMS = ("framewise_spatial_control", "causal_temporal")
SIM_UPDATES = 4_000
REAL_UPDATES = 4_000
BATCH_SIZE = 8
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 12_000
EMA_DECAY = 0.999
SOFT_SPS_WEIGHT = 0.05
SOFT_SPS_TAU_FRACTION = 0.05
EXPECTED_FUSION_PARAMETERS = 41_671
POINT_WEIGHTS = (46.0, 10.0, 9.0)
TRACE_INTERVAL = 500


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "sim-smoke", "sim", "real-smoke", "real"),
        default="contract")
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-checkpoint-sha256")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--sim-control-endpoint", type=Path)
    parser.add_argument("--sim-control-endpoint-sha256")
    parser.add_argument("--sim-causal-endpoint", type=Path)
    parser.add_argument("--sim-causal-endpoint-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args(argv)


def experiment_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "question": (
            "can a causal history specialist improve the frozen mature B210 "
            "when point and native-UQ choose its residual independently"),
        "architecture": {
            "anchor": "frozen B210 full-prefix native-UQ U3200",
            "specialist": asdict(CausalResidualSoftGateConfig()),
            "specialist_output": "strict per-pixel/channel zero-time-mean uv residual",
            "point_gate": "independent zero-init tanh per uv channel",
            "uq_gate": "independent zero-init tanh per uv channel and interval-centre only",
            "hard_replacement": False,
            "native_interval_width": "preserved; only centre may move",
            "pressure": "bitwise anchor passthrough",
            "trainable_parameters_per_arm": EXPECTED_FUSION_PARAMETERS,
        },
        "matched_pair": {
            "control": "framewise spatial-x mixer",
            "candidate": "strict left-causal observed-time mixer",
            "same_parameter_shapes": True,
            "same_initial_tensor_values": True,
            "same_sample_and_augmentation_stream": True,
            "only_execution_difference": "specialist mixer axis/mask",
        },
        "sim": {
            "data": "all released Sim100/4800 stride-20 windows",
            "updates_each_arm": SIM_UPDATES,
            "mapping": "Sim input/target z-score to frozen B210 Real moments; structural zeros retained",
            "augmentation": None,
        },
        "real": {
            "data": "clean81 optimizer-visible prefix, dense stride5/9933 windows",
            "updates_each_arm": REAL_UPDATES,
            "augmentation": dict(real4k.AUGMENTATION),
        },
        "optimization": {
            "point_loss": "published 46/10/9 differentiable accuracy objective",
            "uq_loss": "existing native soft-SPS proxy, point detached",
            "soft_sps_weight": SOFT_SPS_WEIGHT,
            "soft_sps_tau_fraction": SOFT_SPS_TAU_FRACTION,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": f"CosineAnnealingLR(T_max={SCHEDULER_T_MAX})",
            "ema_decay": EMA_DECAY,
        },
        "selection": None,
        "clean716_read": False,
        "submission_authorized": False,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str | None, role: str) -> str:
    if expected is None:
        raise ValueError(f"{role} requires an explicit SHA-256")
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{role} SHA-256 must be 64 lowercase hex digits")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{role} changed: {actual} != {expected}")
    return actual


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def normalizer_tensors(values) -> dict[str, torch.Tensor]:
    names = ("mean_input", "mean_target", "std_input", "std_target")
    result = {
        name: torch.as_tensor(
            values[name] if isinstance(values, Mapping) else getattr(values, name)
        ).detach().float().cpu().clone()
        for name in names
    }
    if any(tuple(value.shape) != (3,) for value in result.values()):
        raise ValueError("normalizer tensors must all have shape [3]")
    return result


def normalizer_from_values(values, device: torch.device):
    tensors = normalizer_tensors(values)
    return sim_data.ChannelNormalizer(
        tensors["mean_input"], tensors["mean_target"],
        tensors["std_input"], tensors["std_target"], device)


def normalizer_sha(values) -> str:
    return ema_lib.tensor_state_sha256(normalizer_tensors(values))


def load_base(
        path: Path, expected_sha: str, device: torch.device,
        ) -> tuple[Mapping[str, object], object, object, dict[str, object]]:
    payload, config, state, report = phase_train.load_base_payload(
        path, expected_sha)
    if (payload.get("schema") != "b210_full_prefix_native_uq_refit_v1"
            or payload.get("phase") != "real"
            or int(payload.get("uq_terminal_extension", {}).get("terminal", -1)) != 3_200
            or payload.get("point_state_bitwise") is not True):
        raise RuntimeError("anchor is not the sealed B210 native-UQ U3200 endpoint")
    values = payload.get("normalizer")
    if not isinstance(values, Mapping):
        raise RuntimeError("anchor lacks its Real normalizer")
    normalizer = normalizer_from_values(values, device)
    report.update({
        "schema": payload["schema"],
        "normalizer_sha256": normalizer_sha(values),
        "uq_terminal": 3_200,
        "point_state_bitwise": True,
    })
    return payload, config, state, {"report": report, "normalizer": normalizer}


def build_fusion(mode: str, device: torch.device) -> CausalResidualSoftGate:
    if mode not in ARMS:
        raise ValueError(f"unknown arm {mode!r}")
    torch.manual_seed(real4k.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(real4k.SEED)
    model = CausalResidualSoftGate(
        CausalResidualSoftGateConfig(mixer_mode=mode)).float()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_FUSION_PARAMETERS:
        raise RuntimeError(f"fusion parameter count changed: {count}")
    return model.to(device)


def parse_fusion_config(values: Mapping[str, object]) -> CausalResidualSoftGateConfig:
    raw = dict(values)
    metadata = {
        "architecture": "causal_residual_soft_gate",
        "anchor_responsibility": "B210 point, pressure, native interval width",
        "specialist_responsibility": "strict zero-time-mean future uv residual",
        "point_fusion": "tanh(point_gate) * specialist_residual",
        "uq_fusion": "tanh(uq_gate) * specialist_interval_center_shift",
        "hard_replacement_available": False,
        "pressure_passthrough": True,
        "native_interval_width_preserved": True,
    }
    for key, expected in metadata.items():
        if raw.pop(key, None) != expected:
            raise ValueError(f"fusion architecture metadata changed: {key}")
    config = CausalResidualSoftGateConfig(**raw)
    config.validate()
    return config


def common_initialization_report() -> dict[str, object]:
    states = {}
    for mode in ARMS:
        model = build_fusion(mode, torch.device("cpu"))
        states[mode] = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    if states[ARMS[0]].keys() != states[ARMS[1]].keys():
        raise RuntimeError("matched fusion state keys differ")
    equal = all(torch.equal(states[ARMS[0]][key], states[ARMS[1]][key])
                for key in states[ARMS[0]])
    if not equal:
        raise RuntimeError("matched fusion initial tensors differ")
    digest = ema_lib.tensor_state_sha256(states[ARMS[0]])
    return {
        "tensor_state_sha256": digest,
        "bitwise_equal": True,
        "state_tensors": len(states[ARMS[0]]),
        "parameters": EXPECTED_FUSION_PARAMETERS,
    }


def structural_zero_mask(values: torch.Tensor) -> torch.Tensor:
    return values[..., :2].eq(0.0)


def map_sim_to_real_coordinates(
        history: torch.Tensor, target: torch.Tensor,
        sim_normalizer, real_normalizer,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Moment-align released Sim while retaining exact solid-body zeros."""
    history_zero = structural_zero_mask(history)
    target_zero = structural_zero_mask(target)
    mapped_history = (
        (history - sim_normalizer.mean_input) / sim_normalizer.std_input
        * real_normalizer.std_input + real_normalizer.mean_input)
    mapped_target = (
        (target - sim_normalizer.mean_target) / sim_normalizer.std_target
        * real_normalizer.std_target + real_normalizer.mean_target)
    mapped_history[..., :2] = torch.where(
        history_zero, torch.zeros_like(mapped_history[..., :2]),
        mapped_history[..., :2])
    mapped_target[..., :2] = torch.where(
        target_zero, torch.zeros_like(mapped_target[..., :2]),
        mapped_target[..., :2])
    mapped_history[..., 2] = 0.0
    mapped_target[..., 2] = 0.0
    return mapped_history, mapped_target


def objective(
        values, target_normalized: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    point_loss, point = sim_data.accuracy_objective(
        values.prediction, target_normalized, normalizer, POINT_WEIGHTS)
    sps_loss, sps_score = ensemble_train.soft_native_sps_objective(
        values.prediction, target_normalized,
        values.lower_extent, values.upper_extent,
        normalizer, SOFT_SPS_TAU_FRACTION)
    loss = point_loss + SOFT_SPS_WEIGHT * sps_loss
    return loss, {
        **point,
        "soft_sps_loss": sps_loss.detach(),
        "soft_sps_score": sps_score,
        "total_objective": loss.detach(),
    }


def state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def gradient_nonzero(parameter: torch.Tensor | None) -> bool:
    return bool(parameter is not None and torch.isfinite(parameter).all()
                and torch.count_nonzero(parameter))


def train_arm(
        *, stage: str, arm: str, base_config, base_state,
        source: Mapping[str, object], normalizer, dataset,
        schedule: Sequence[Sequence[int]], output_dir: Path,
        device: torch.device, updates: int,
        initial_state: Mapping[str, torch.Tensor] | None = None,
        initial_lineage: Mapping[str, object] | None = None,
        sim_normalizer=None,
        ) -> dict[str, object]:
    base = phase_train.build_base(base_config, base_state, normalizer, device)
    fusion = build_fusion(arm, device)
    if initial_state is not None:
        fusion.load_state_dict(initial_state, strict=True)
    initial_state_sha = ema_lib.tensor_state_sha256(fusion.state_dict())
    wrapper = FrozenB210CausalResidualSoftGate(base, fusion).train()
    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX)
    ema = ema_lib.FP32EMAShadow(fusion, EMA_DECAY)
    real4k.seed_real_recipe(device)

    trace: list[dict[str, object]] = []
    mechanics = {
        "update0_prediction_bitwise_identity": False,
        "update0_lower_bitwise_identity": False,
        "update0_upper_bitwise_identity": False,
        "point_gate_gradient_u1": False,
        "uq_gate_gradient_u1": False,
        "specialist_gradient_u1_exact_zero": False,
        "specialist_direction_gradient_u2": False,
        "history_gradient_u2": False,
        "zero_time_mean_residual": False,
        "zero_time_mean_point_delta": False,
        "zero_time_mean_uq_delta": False,
        "pressure_bitwise": False,
    }
    started = time.perf_counter()
    for cursor in range(updates):
        update = cursor + 1
        histories, targets = zip(*(dataset[index] for index in schedule[cursor]))
        history_physical = torch.stack(histories).to(device)
        target_physical = torch.stack(targets).to(device)
        if stage == "sim":
            if sim_normalizer is None:
                raise RuntimeError("Sim stage requires the Sim normalizer")
            mapped_history, mapped_target = map_sim_to_real_coordinates(
                history_physical, target_physical,
                sim_normalizer, normalizer)
            base_history, target_normalized = normalizer.preprocess(
                mapped_history, mapped_target)
            adapter_history = base_history
        elif stage == "real":
            adapter_history = real4k.augment_real_history(
                history_physical, normalizer)
            base_history, target_normalized = normalizer.preprocess(
                history_physical, target_physical)
        else:
            raise ValueError(f"unknown stage {stage!r}")

        optimizer.zero_grad(set_to_none=True)
        values = wrapper.forward_components(
            base_history, adapter_history=adapter_history)
        if update == 1:
            if stage == "sim":
                mechanics.update({
                    "update0_prediction_bitwise_identity": torch.equal(
                        values.prediction, values.base_prediction),
                    "update0_lower_bitwise_identity": torch.equal(
                        values.lower_extent, values.base_lower_extent),
                    "update0_upper_bitwise_identity": torch.equal(
                        values.upper_extent, values.base_upper_extent),
                })
            else:
                inherited = (
                    initial_lineage.get("mechanics", {})
                    if isinstance(initial_lineage, Mapping) else {})
                for name in (
                        "update0_prediction_bitwise_identity",
                        "update0_lower_bitwise_identity",
                        "update0_upper_bitwise_identity"):
                    mechanics[name] = inherited.get(name) is True
            if not all(mechanics[key] for key in (
                    "update0_prediction_bitwise_identity",
                    "update0_lower_bitwise_identity",
                    "update0_upper_bitwise_identity")):
                raise RuntimeError("soft-gate update zero is not exact B210 identity")
        loss, details = objective(values, target_normalized, normalizer)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite {stage}/{arm} loss at u{update}")
        loss.backward()
        if update == 1:
            mechanics["point_gate_gradient_u1"] = gradient_nonzero(
                fusion.point_gate_logits.grad)
            mechanics["uq_gate_gradient_u1"] = gradient_nonzero(
                fusion.uq_gate_logits.grad)
            specialist_gradients = [
                parameter.grad for parameter in fusion.specialist.parameters()
                if parameter.grad is not None]
            if stage == "sim":
                mechanics["specialist_gradient_u1_exact_zero"] = bool(
                    specialist_gradients and all(
                        torch.count_nonzero(gradient) == 0
                        for gradient in specialist_gradients))
            else:
                inherited = (
                    initial_lineage.get("mechanics", {})
                    if isinstance(initial_lineage, Mapping) else {})
                mechanics["specialist_gradient_u1_exact_zero"] = (
                    inherited.get("specialist_gradient_u1_exact_zero") is True)
            if not (mechanics["point_gate_gradient_u1"]
                    and mechanics["uq_gate_gradient_u1"]
                    and mechanics["specialist_gradient_u1_exact_zero"]):
                raise RuntimeError(
                    f"soft-gate bootstrap failed at {stage}/{arm} u1: {mechanics}")
        if update == 2:
            mechanics["specialist_direction_gradient_u2"] = gradient_nonzero(
                fusion.specialist.direction_head.weight.grad)
            mechanics["history_gradient_u2"] = gradient_nonzero(
                fusion.specialist.frame_encoder.stem.weight.grad)
            if not (mechanics["specialist_direction_gradient_u2"]
                    and mechanics["history_gradient_u2"]):
                raise RuntimeError(
                    f"specialist did not receive u2 gradient: {mechanics}")
        gradient = nn.utils.clip_grad_norm_(fusion.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError(f"non-finite gradient at {stage}/{arm} u{update}")
        optimizer.step()
        ema.update_after_optimizer_step(fusion, optimizer_step=update)
        scheduler.step()

        if update in (1, 2, updates) or update % TRACE_INTERVAL == 0:
            residual_mean = values.specialist_residual.float().mean(dim=1)
            point_mean = values.point_delta.float().mean(dim=1)
            uq_mean = values.uq_center_delta.float().mean(dim=1)
            mechanics.update({
                "zero_time_mean_residual": bool(
                    residual_mean.abs().max() <= 1.0e-6),
                "zero_time_mean_point_delta": bool(
                    point_mean.abs().max() <= 1.0e-6),
                "zero_time_mean_uq_delta": bool(
                    uq_mean.abs().max() <= 1.0e-6),
                "pressure_bitwise": torch.equal(
                    values.prediction[..., 2:],
                    values.base_prediction[..., 2:]),
            })
            row = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "point_gate": values.point_gate.detach().cpu().flatten().tolist(),
                "uq_gate": values.uq_gate.detach().cpu().flatten().tolist(),
                "specialist_residual_rms": float(
                    values.specialist_residual.float().square().mean().sqrt()),
                "point_delta_rms": float(
                    values.point_delta.float().square().mean().sqrt()),
                "uq_center_delta_rms": float(
                    values.uq_center_delta.float().square().mean().sqrt()),
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({
                "event": "causal_residual_soft_gate_progress",
                "stage": stage, "arm": arm, **row,
            }, sort_keys=True), flush=True)

    if updates > 2 and not all(mechanics.values()):
        raise RuntimeError(f"complete {stage}/{arm} mechanics failed: {mechanics}")
    raw_state = state_snapshot(fusion)
    ema_state = ema.materialize_model_state(fusion)
    elapsed = time.perf_counter() - started
    endpoint = {
        "schema": ENDPOINT_SCHEMA,
        "experiment_schema": SCHEMA,
        "stage": stage,
        "arm": arm,
        "update": updates,
        "base_checkpoint": dict(source),
        "fusion_config": fusion.architecture_config(),
        "fusion_parameters": EXPECTED_FUSION_PARAMETERS,
        "initial_fusion_state_sha256": initial_state_sha,
        "initial_lineage": (
            None if initial_lineage is None else dict(initial_lineage)),
        "raw_fusion_state": raw_state,
        "raw_fusion_state_sha256": ema_lib.tensor_state_sha256(raw_state),
        "ema_fusion_state": ema_state,
        "ema_fusion_state_sha256": ema_lib.tensor_state_sha256(ema_state),
        "ema": {key: value for key, value in ema.seal(fusion).items()
                if key != "model_state"},
        "normalizer": normalizer_tensors(normalizer),
        "sample_schedule_sha256": real4k.schedule_sha(schedule[:updates]),
        "training": {
            "updates": updates,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler_t_max": SCHEDULER_T_MAX,
            "ema_decay": EMA_DECAY,
            "point_weights": list(POINT_WEIGHTS),
            "soft_sps_weight": SOFT_SPS_WEIGHT,
            "soft_sps_tau_fraction": SOFT_SPS_TAU_FRACTION,
            "augmentation": None if stage == "sim" else dict(real4k.AUGMENTATION),
            "trace": trace,
            "elapsed_seconds": elapsed,
        },
        "mechanics": mechanics,
        "clean716_opened": False,
        "checkpoint_selected": False,
        "submission_authorized": False,
    }
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    path = arm_dir / f"fusion_raw_ema_u{updates}.pt"
    real4k.atomic_torch_save(path, endpoint)
    report = {
        "arm": arm,
        "path": str(path.resolve()),
        "file_sha256": sha256(path),
        "raw_fusion_state_sha256": endpoint["raw_fusion_state_sha256"],
        "ema_fusion_state_sha256": endpoint["ema_fusion_state_sha256"],
        "initial_fusion_state_sha256": initial_state_sha,
        "updates": updates,
        "elapsed_seconds": elapsed,
        "mechanics": mechanics,
        "terminal_trace": trace[-1],
    }
    del wrapper, fusion, base, optimizer, scheduler, ema, endpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def validate_sim_endpoint(
        path: Path, expected_sha: str, expected_arm: str,
        expected_base_sha: str, expected_updates: int = SIM_UPDATES,
        ) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    actual = require_sha(path, expected_sha, f"Sim {expected_arm} endpoint")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, Mapping)
            or payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment_schema") != SCHEMA
            or payload.get("stage") != "sim"
            or payload.get("arm") != expected_arm
            or int(payload.get("update", -1)) != expected_updates):
        raise RuntimeError(
            f"{expected_arm} is not the required Sim u{expected_updates} endpoint")
    if payload.get("base_checkpoint", {}).get("sha256") != expected_base_sha:
        raise RuntimeError("Sim endpoint references another B210 anchor")
    config = parse_fusion_config(payload.get("fusion_config", {}))
    if config.mixer_mode != expected_arm:
        raise RuntimeError("Sim endpoint arm/config mismatch")
    state = payload.get("raw_fusion_state")
    if not isinstance(state, Mapping):
        raise RuntimeError("Sim endpoint lacks its raw fusion state")
    if ema_lib.tensor_state_sha256(state) != payload.get(
            "raw_fusion_state_sha256"):
        raise RuntimeError("Sim raw fusion state digest changed")
    probe = CausalResidualSoftGate(config).float()
    probe.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in probe.parameters()) != (
            EXPECTED_FUSION_PARAMETERS):
        raise RuntimeError("Sim endpoint parameter count changed")
    return {
        "path": str(path.resolve()),
        "file_sha256": actual,
        "raw_fusion_state_sha256": payload["raw_fusion_state_sha256"],
        "initial_fusion_state_sha256": payload["initial_fusion_state_sha256"],
        "sample_schedule_sha256": payload["sample_schedule_sha256"],
        "mechanics": dict(payload.get("mechanics", {})),
    }, dict(state)


def validate_cli(args: argparse.Namespace) -> tuple[str, int]:
    if args.mode in ("sim", "real"):
        expected_ack = (SIM_ACKNOWLEDGEMENT if args.mode == "sim"
                        else REAL_ACKNOWLEDGEMENT)
        if not args.execute or args.acknowledge_protocol != expected_ack:
            raise PermissionError(
                f"formal {args.mode} requires --execute and {expected_ack}")
        updates = SIM_UPDATES if args.mode == "sim" else REAL_UPDATES
    else:
        updates = 2
    if args.mode != "contract":
        if (args.base_checkpoint is None
                or args.base_checkpoint_sha256 is None
                or args.output_dir is None):
            raise ValueError("training requires base checkpoint/SHA and output dir")
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to reuse output {args.output_dir}")
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("lifecycle training requires CUDA")
    return ("sim" if args.mode.startswith("sim") else "real"), updates


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "contract":
        return experiment_contract()
    stage, updates = validate_cli(args)
    device = torch.device(args.device)
    payload, base_config, base_state, loaded = load_base(
        args.base_checkpoint, args.base_checkpoint_sha256, device)
    source = dict(loaded["report"])
    normalizer = loaded["normalizer"]
    initialization = common_initialization_report()

    if stage == "sim":
        if args.sim_dir is None:
            raise ValueError("Sim stage requires --sim-dir")
        rows = sim_data.inventory(args.sim_dir)
        trajectories, windows, files = sim_data.load_fields(args.sim_dir, rows)
        if len(trajectories) != 100 or len(windows) != 4_800:
            raise RuntimeError("released Sim100/4800 contract changed")
        dataset = sim_data.SimWindowDataset(trajectories, windows)
        statistics = sim_data.window_statistics(trajectories, windows)
        sim_normalizer = sim_data.ChannelNormalizer(*statistics, device)
        schedule = real4k.make_sample_schedule(
            len(dataset), SIM_UPDATES, BATCH_SIZE)
        dataset_report = {
            "domain": "released_Sim100",
            "trajectories": len(trajectories),
            "windows": len(dataset),
            "stride": 20,
            "window_file_sha256": real4k.canonical_sha(files),
            "sim_normalizer_sha256": normalizer_sha(sim_normalizer),
            "real_coordinate_normalizer_sha256": normalizer_sha(normalizer),
            "structural_zeros_retained": True,
        }
        initial_states = {arm: None for arm in ARMS}
        initial_lineages = {arm: None for arm in ARMS}
    else:
        required = (
            args.real_dir, args.clean_manifest,
            args.sim_control_endpoint, args.sim_control_endpoint_sha256,
            args.sim_causal_endpoint, args.sim_causal_endpoint_sha256,
        )
        if any(value is None for value in required):
            raise ValueError("Real stage requires Real data/manifest and both Sim seals")
        control_report, control_state = validate_sim_endpoint(
            args.sim_control_endpoint, args.sim_control_endpoint_sha256,
            ARMS[0], args.base_checkpoint_sha256,
            2 if args.mode == "real-smoke" else SIM_UPDATES)
        causal_report, causal_state = validate_sim_endpoint(
            args.sim_causal_endpoint, args.sim_causal_endpoint_sha256,
            ARMS[1], args.base_checkpoint_sha256,
            2 if args.mode == "real-smoke" else SIM_UPDATES)
        if (control_report["sample_schedule_sha256"]
                != causal_report["sample_schedule_sha256"]):
            raise RuntimeError("matched Sim endpoints used different sample streams")
        if (control_report["initial_fusion_state_sha256"]
                != causal_report["initial_fusion_state_sha256"]
                or control_report["initial_fusion_state_sha256"]
                != initialization["tensor_state_sha256"]):
            raise RuntimeError("matched Sim arms did not share initialization")
        rows, manifest_report = real4k.load_clean_manifest(args.clean_manifest)
        access = real4k.PrefixAccessAudit()
        trajectories, sparse_windows = real4k.load_real_rows(
            args.real_dir, rows, phase="fit", audit=access)
        derived_normalizer, normalizer_report = (
            real4k.normalizer_from_sparse_windows(
                trajectories, sparse_windows, device))
        if normalizer_sha(derived_normalizer) != normalizer_sha(normalizer):
            raise RuntimeError("base and clean81 Real normalizers differ")
        normalizer = derived_normalizer
        optimizer_windows = real4k.dense_windows(trajectories)
        if len(optimizer_windows) != real4k.EXPECTED_OPTIMIZER_WINDOWS:
            raise RuntimeError("dense Real optimizer window count changed")
        dataset = real4k.RealWindowDataset(trajectories, optimizer_windows)
        schedule = real4k.make_sample_schedule(
            len(dataset), REAL_UPDATES, BATCH_SIZE)
        sim_normalizer = None
        initial_states = {
            ARMS[0]: control_state,
            ARMS[1]: causal_state,
        }
        initial_lineages = {
            ARMS[0]: control_report,
            ARMS[1]: causal_report,
        }
        dataset_report = {
            "domain": "clean81_optimizer_visible_prefixes",
            "manifest": manifest_report,
            "normalizer": normalizer_report,
            "trajectories": len(trajectories),
            "windows": len(dataset),
            "stride": real4k.OPTIMIZER_STRIDE,
            "access_events": list(access.events),
            "sim_endpoints": {
                ARMS[0]: control_report,
                ARMS[1]: causal_report,
            },
        }

    args.output_dir.mkdir(parents=True)
    reports = []
    for arm in ARMS:
        reports.append(train_arm(
            stage=stage, arm=arm,
            base_config=base_config, base_state=base_state,
            source=source, normalizer=normalizer,
            dataset=dataset, schedule=schedule,
            output_dir=args.output_dir, device=device, updates=updates,
            initial_state=initial_states[arm],
            initial_lineage=initial_lineages[arm],
            sim_normalizer=sim_normalizer))
    if reports[0]["initial_fusion_state_sha256"] != reports[1][
            "initial_fusion_state_sha256"]:
        # In Real, each arm legitimately starts from its differently-trained
        # Sim state.  Sim itself must retain the exact matched initialization.
        if stage == "sim":
            raise RuntimeError("matched Sim arms started from different tensors")
    result = {
        "schema": SCHEMA,
        "mode": args.mode,
        "stage": stage,
        "contract": experiment_contract(),
        "base": source,
        "initialization": initialization,
        "dataset": dataset_report,
        "sample_schedule_sha256": real4k.schedule_sha(schedule[:updates]),
        "arms": reports,
        "clean716_opened": False,
        "checkpoint_selected": False,
        "submission_authorized": False,
    }
    atomic_json(args.output_dir / "result.json", result)
    del payload, base_state
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_causal_residual_soft_gate_lifecycle_complete",
        "mode": args.mode,
        "stage": result.get("stage"),
        "output_dir": None if args.output_dir is None else str(args.output_dir),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
