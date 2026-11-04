#!/usr/bin/env python3
"""Matched FFAR-ST adaptation from one sealed causal ST Sim10k endpoint.

The default policy preserves the original FFAR sealed Sim physical function.
The explicit ``clean81_real_affine`` policy instead installs the same clean81
Real affine normalizer on the sealed Sim weights before the matched pair is
constructed.  Under either policy, the control updates only the complete
source function while the candidate freezes it and updates an exact-zero-start
Real residual.  All other training and endpoint mechanics are shared.

Formal u4000/u12000 raw and EMA endpoints are fixed in advance.  clean716 is
not read until every endpoint for both arms has been atomically sealed and
re-hashed.  There is no score gate, early stopping, or checkpoint selection.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_ensemble_unet2d as mature_real  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real_long  # noqa: E402
import train_pyramid_causal3d_sim_long as sim_long  # noqa: E402
import train_st_b210_full_lifecycle as st_lifecycle  # noqa: E402
from models.full_function_anchored_spatiotemporal_b210 import (  # noqa: E402
    FULL_FUNCTION_ADAPTATION_MODES,
    SOURCE_NORMALIZER_POLICIES,
    FullFunctionAnchoredSTB210,
)
from models.spatiotemporal_b210 import (  # noqa: E402
    SpatiotemporalB210,
    SpatiotemporalB210Config,
)


SCHEMA = "st_b210_full_function_anchor_real_v1"
ENDPOINT_SCHEMA = "st_b210_full_function_anchor_endpoint_v1"
RESUME_SCHEMA = "st_b210_full_function_anchor_resume_v1"
ACKNOWLEDGEMENT = "RUN_ST_B210_FULL_FUNCTION_ANCHOR_MATCHED_REAL_V1"
REAL_AFFINE_SCHEMA = "st_b210_real_affine_aligned_anchor_real_v1"
REAL_AFFINE_ENDPOINT_SCHEMA = (
    "st_b210_real_affine_aligned_anchor_endpoint_v1")
REAL_AFFINE_RESUME_SCHEMA = "st_b210_real_affine_aligned_anchor_resume_v1"
REAL_AFFINE_ACKNOWLEDGEMENT = (
    "RUN_ST_B210_REAL_AFFINE_ALIGNED_ANCHOR_MATCHED_REAL_V1")
DEFAULT_SOURCE_NORMALIZER_POLICY = "sealed_sim"
REAL_AFFINE_SOURCE_NORMALIZER_POLICY = "clean81_real_affine"
ARMS = ("source_trainable_control", "full_function_anchor_candidate")
ARM_MODE = dict(zip(ARMS, FULL_FUNCTION_ADAPTATION_MODES, strict=True))
ENDPOINTS = ("raw", "ema")
REAL_UPDATE_CHOICES = (4_000, 12_000)
REAL_SEEDS = (4, 17)
EXPECTED_REAL_STREAM_SHA256 = {
    (4_000, 4): "9381a97fc114e4234a889d686158533d0088701676b683d268d12d4203f1be43",
    (4_000, 17): "608bf0f90b3beb535f385b7ae71e99501f698eac0545893725912c90c3b40bd6",
    (12_000, 4): "e853a2e0346947159fb41f8444b3ba872df2d786e8a471ae91a04c63a67a046d",
    (12_000, 17): "35afdae1cd5ad5b73c81745b3e4914dec4b146ea65c54388b7367d3ec09cb3e4",
}
DIAGNOSTIC_UPDATES = (1, 2, 500, 1_000, 2_000, 3_000)
EXPECTED_SINGLE_PARAMETERS = 29_187_608
EXPECTED_SINGLE_STATE_TENSORS = 238
EXPECTED_START_FILE_SHA256 = (
    "debb322723ea0f7b895c798c04825ec2899349ae9acbabd3018cd89d80e2f74c")
EXPECTED_START_MODEL_STATE_SHA256 = (
    "a474c1b535eed28349be3374ba6cc7ab66672cd2c366fd95cae176211008ccf3")
EXPECTED_START_NORMALIZER_SHA256 = (
    "90ea373cc0f38d3f576215c24d57491dd2dcb18058fea0965618f4781a36a69c")
EXPECTED_START_PHYSICAL_FUNCTION_SHA256 = (
    "4cc64fb26357e18fc9d8a0b35416961300f586fbb3259d4555f324ed3e835030")
EXPECTED_PARAMETERS = 58_375_219
EXPECTED_STATE_TENSORS = 485
EXPECTED_SOURCE_PARAMETERS = 29_187_608
EXPECTED_TRAINABLE_PARAMETERS = {
    ARMS[0]: EXPECTED_SINGLE_PARAMETERS,
    ARMS[1]: EXPECTED_SINGLE_PARAMETERS + 3,
}
OFFICIAL_POINT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")
MECHANICAL_EXPECTED = {
    "sealed_sim_physical_mse_all3": 3.894962,
    "real_affine_aligned_mse_all3": 0.164602,
    "sealed_sim_v_mse": 11.12856,
    "sealed_sim_prediction_v_std": 0.05193,
    "real_affine_prediction_v_std": 0.01490,
    "target_v_std": 0.01475,
}
MECHANICAL_REPORTED_DECIMALS = {
    "sealed_sim_physical_mse_all3": 6,
    "real_affine_aligned_mse_all3": 6,
    "sealed_sim_v_mse": 5,
    "sealed_sim_prediction_v_std": 5,
    "real_affine_prediction_v_std": 5,
    "target_v_std": 5,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "checkpoint-smoke", "formal"),
        default="contract")
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--real-updates", type=int, choices=REAL_UPDATE_CHOICES,
        default=4_000)
    parser.add_argument(
        "--real-seed", type=int, choices=REAL_SEEDS, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--source-normalizer-policy", choices=SOURCE_NORMALIZER_POLICIES,
        default=DEFAULT_SOURCE_NORMALIZER_POLICY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args(argv)


def milestone_updates(real_updates: int) -> tuple[int, ...]:
    if real_updates == 4_000:
        return (4_000,)
    if real_updates == 12_000:
        return (4_000, 12_000)
    raise ValueError(f"unsupported Real update count {real_updates}")


def configure_strict_fp32(device: torch.device) -> None:
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_real_recipe(seed: int, device: torch.device) -> None:
    configure_strict_fp32(device)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def experiment_schema(source_normalizer_policy: str) -> str:
    return (REAL_AFFINE_SCHEMA
            if source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
            else SCHEMA)


def endpoint_schema(source_normalizer_policy: str) -> str:
    return (REAL_AFFINE_ENDPOINT_SCHEMA
            if source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
            else ENDPOINT_SCHEMA)


def resume_schema(source_normalizer_policy: str) -> str:
    return (REAL_AFFINE_RESUME_SCHEMA
            if source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
            else RESUME_SCHEMA)


def acknowledgement(source_normalizer_policy: str) -> str:
    return (REAL_AFFINE_ACKNOWLEDGEMENT
            if source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
            else ACKNOWLEDGEMENT)


def make_sample_schedule(
        samples: int, updates: int, batch_size: int, seed: int,
        ) -> list[list[int]]:
    if samples <= 0 or updates <= 0 or batch_size <= 0:
        raise ValueError("sample count, updates, and batch size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    batches: list[list[int]] = []
    while len(batches) < updates:
        epoch = torch.randperm(samples, generator=generator).tolist()
        batches.extend([
            epoch[start:start + batch_size]
            for start in range(0, samples, batch_size)
        ])
    return batches[:updates]


def training_contract(
        *, real_updates: int, real_seed: int, schedule_sha256: str,
        source_normalizer_policy: str = DEFAULT_SOURCE_NORMALIZER_POLICY,
        ) -> dict[str, object]:
    if source_normalizer_policy not in SOURCE_NORMALIZER_POLICIES:
        raise ValueError("unsupported source normalizer policy")
    aligned = (
        source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY)
    return {
        "schema": experiment_schema(source_normalizer_policy),
        "question": (
            "after removing the confirmed Sim-to-Real affine scale mismatch, "
            "does preserving the complete aligned source function while "
            "learning a zero-start Real residual prevent destructive Real "
            "adaptation" if aligned else
            "does preserving the complete sealed Sim physical function while "
            "learning a zero-start Real residual prevent Real adaptation from "
            "erasing useful source dynamics"),
        "start": (
            "one_common_sealed_causal_ST_B210_Sim_raw_u10000_weight_state_"
            "with_clean81_Real_affine_installed_before_pair_construction"
            if aligned else
            "one_common_sealed_causal_ST_B210_Sim_raw_u10000"),
        "arms": {
            ARMS[0]: {
                "adaptation_mode": ARM_MODE[ARMS[0]],
                "complete_source_trainable": True,
                "real_residual_trainable": False,
            },
            ARMS[1]: {
                "adaptation_mode": ARM_MODE[ARMS[1]],
                "complete_source_trainable": False,
                "real_residual_trainable": True,
            },
        },
        "only_training_variable": (
            "Real_gradient_destination_complete_source_or_additive_residual"),
        "matched": {
            "same_topology_parameters_forward_flops": True,
            "same_initial_state_and_output": True,
            "mutually_exclusive_trainable_roles": True,
            "parameters_each": EXPECTED_PARAMETERS,
            "source_parameters_each": EXPECTED_SOURCE_PARAMETERS,
            "trainable_parameters": dict(EXPECTED_TRAINABLE_PARAMETERS),
        },
        "normalizer_semantics": {
            "fusion_space": "physical_units",
            **({"source_weight_state": "sealed_Sim_raw_u10000"}
               if aligned else {}),
            "source_input_and_output": (
                "clean81_Real_normalizer" if aligned
                else "sealed_Sim_normalizer_only"),
            "real_residual_input_and_output_scale": "clean81_Real_normalizer",
            **({
                "source_normalizer_policy": source_normalizer_policy,
                "source_and_residual_affine_bitwise_equal_at_update0": True,
                "generic_normalizer_mutation_after_pair_construction":
                    "forbidden",
            } if aligned else {
                "generic_common_normalizer_installation": "forbidden"}),
            "unmeasured_pressure": (
                "all3_compatibility_loss_then_exact_zero_in_official_evaluator"),
        },
        "residual": {
            "architecture": "complete_causal_ST_B210",
            "initialization": (
                "two_copies_of_sealed_Sim_weights_with_identical_clean81_Real_"
                "persistence_buffers" if aligned else
                "copy_of_sealed_source_except_Real_persistence_buffers"),
            "fusion": "source_physical_plus_gain_times_residual_physical",
            "gain": "per_channel_trainable_exact_zero_at_update0",
            "update0_body_gradient": "zero_by_chain_rule_then_live_after_gain_moves",
        },
        "real": {
            "data": "clean81_optimizer_visible_prefix",
            "normalizer_stride": real_long.NORMALIZER_STRIDE,
            "optimizer_stride": real_long.OPTIMIZER_STRIDE,
            "optimizer_windows": real_long.EXPECTED_OPTIMIZER_WINDOWS,
            "updates": int(real_updates),
            "seed": int(real_seed),
            "batch_size": real_long.BATCH_SIZE,
            "sample_stream_sha256": str(schedule_sha256),
            "sample_stream_continuous_u1_to_terminal": True,
            "optimizer": "AdamW",
            "lr": real_long.LR,
            "weight_decay": real_long.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": real_long.SCHEDULER_T_MAX,
            "loss": (
                "u1_u1500_all3_mse_then_accuracy_5_3_2_plus_.25_all3_mse"),
            "augmentation": dict(real_long.AUGMENTATION),
            "ema_decay": real_long.EMA_DECAY,
            "fixed_endpoint_updates": list(milestone_updates(real_updates)),
            "endpoint_kinds_each_update": list(ENDPOINTS),
            "rolling_resume_only": True,
        },
        "ema_semantics": {
            "fp32_named_parameter_shadow": True,
            "candidate_frozen_source_overlaid_bitwise_from_raw_at_endpoint": True,
            "control_frozen_residual_overlaid_bitwise_from_raw_at_endpoint": True,
            "buffers_copied_exactly_from_raw": True,
        },
        "precision": {
            "parameters_inputs_losses": "float32",
            "autocast": False,
            "float32_matmul_precision": "highest",
            "tf32": False,
        },
        "evaluation": {
            "clean716_first_read_after_all_arm_milestone_endpoint_seals": True,
            **({
                "sealed_Sim_weight_start_rehashed_at_clean716_barrier": True,
                "aligned_source_function_rehashed_from_each_endpoint": True,
            } if aligned else {
                "sealed_Sim_start_rehashed_at_clean716_barrier": True,
            }),
            "raw_and_ema": True,
            "by_aoa": [0, 5, 10, 15, 20],
            "same_target_order": True,
        },
        "score_gate": None,
        "early_stopping": False,
        "checkpoint_selection": False,
        "uq": False,
        "packaging": False,
        "submission_authorized": False,
    }


def contract_for_args(
        real_updates: int, real_seed: int,
        source_normalizer_policy: str = DEFAULT_SOURCE_NORMALIZER_POLICY,
        ) -> dict[str, object]:
    schedule = make_sample_schedule(
        real_long.EXPECTED_OPTIMIZER_WINDOWS, real_updates,
        real_long.BATCH_SIZE, real_seed)
    schedule_sha = real_long.schedule_sha(schedule)
    if schedule_sha != EXPECTED_REAL_STREAM_SHA256[(real_updates, real_seed)]:
        raise RuntimeError("pre-registered FFAR Real stream changed")
    return training_contract(
        real_updates=real_updates, real_seed=real_seed,
        schedule_sha256=schedule_sha,
        source_normalizer_policy=source_normalizer_policy)


def _state_snapshot(
        model_or_state: nn.Module | Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    state = (model_or_state.state_dict()
             if isinstance(model_or_state, nn.Module) else model_or_state)
    return OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in state.items())


def _normalizer_payload(normalizer) -> dict[str, torch.Tensor]:
    return {
        name: getattr(normalizer, name).detach().cpu().clone()
        for name in ("mean_input", "mean_target", "std_input", "std_target")
    }


def _normalizer_from_payload(
        values: Mapping[str, torch.Tensor], device: torch.device):
    required = {"mean_input", "mean_target", "std_input", "std_target"}
    if set(values) != required:
        raise RuntimeError("endpoint normalizer tensor inventory changed")
    return base.ChannelNormalizer(
        *(values[name].detach().cpu() for name in (
            "mean_input", "mean_target", "std_input", "std_target")),
        device=device)


def _source_model_state(
        model_or_state: FullFunctionAnchoredSTB210 | Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    if isinstance(model_or_state, FullFunctionAnchoredSTB210):
        return _state_snapshot(model_or_state.source_model_state())
    prefix = "source_model."
    result = OrderedDict(
        (name[len(prefix):], value.detach().cpu().clone())
        for name, value in model_or_state.items() if name.startswith(prefix))
    if not result:
        raise RuntimeError("FFAR state lacks the complete source function")
    return result


def _source_model_sha(
        model_or_state: FullFunctionAnchoredSTB210 | Mapping[str, torch.Tensor],
        ) -> str:
    return base.state_sha(_source_model_state(model_or_state))


def _single_parameter_state(
        model: SpatiotemporalB210,
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.named_parameters())


def _source_parameter_state(
        model: FullFunctionAnchoredSTB210,
        ) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.source_model.named_parameters())


def _source_function_state(
        model_or_state: FullFunctionAnchoredSTB210 | Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    state = (model_or_state.state_dict()
             if isinstance(model_or_state, FullFunctionAnchoredSTB210)
             else model_or_state)
    result = OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in state.items()
        if name.startswith("source_model.") or name.startswith("_sim_"))
    if not result or not any(name.startswith("_sim_") for name in result):
        raise RuntimeError("FFAR state lacks source function or Sim normalizer")
    return result


def _source_function_sha(
        model_or_state: FullFunctionAnchoredSTB210 | Mapping[str, torch.Tensor],
        ) -> str:
    return base.state_sha(_source_function_state(model_or_state))


def _sealed_source_function_sha(
        single_state: Mapping[str, torch.Tensor],
        sim_normalizer: Mapping[str, torch.Tensor],
        ) -> str:
    state = OrderedDict(
        (f"source_model.{name}", value.detach().cpu().clone())
        for name, value in single_state.items())
    state.update(
        (f"_sim_{name}", value.detach().cpu().clone())
        for name, value in sim_normalizer.items())
    return base.state_sha(state)


def _residual_branch_state(
        model_or_state: FullFunctionAnchoredSTB210 | Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    state = (model_or_state.state_dict()
             if isinstance(model_or_state, FullFunctionAnchoredSTB210)
             else model_or_state)
    prefix = "real_residual_model."
    result = OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in state.items()
        if (name.startswith(prefix) or name == "residual_gain"
            or name.startswith("_real_")))
    if not result:
        raise RuntimeError("FFAR state lacks the Real residual branch")
    return result


def _residual_branch_sha(
        model_or_state: FullFunctionAnchoredSTB210 | Mapping[str, torch.Tensor],
        ) -> str:
    return base.state_sha(_residual_branch_state(model_or_state))


def _assert_model(model: FullFunctionAnchoredSTB210, arm: str) -> None:
    if arm not in ARMS or model.adaptation_mode != ARM_MODE[arm]:
        raise RuntimeError("FFAR arm/adaptation-mode mismatch")
    parameters = sum(value.numel() for value in model.parameters())
    trainable = sum(
        value.numel() for value in model.parameters() if value.requires_grad)
    source = sum(value.numel() for value in model.source_model.parameters())
    if (parameters != EXPECTED_PARAMETERS
            or trainable != EXPECTED_TRAINABLE_PARAMETERS[arm]
            or source != EXPECTED_SOURCE_PARAMETERS
            or len(model.state_dict()) != EXPECTED_STATE_TENSORS):
        raise RuntimeError(
            "FFAR production structure changed: "
            f"parameters={parameters}, trainable={trainable}, "
            f"source={source}, state_tensors={len(model.state_dict())}")
    source_frozen = all(
        not parameter.requires_grad
        for parameter in model.source_model.parameters())
    residual_frozen = all(
        not parameter.requires_grad
        for parameter in model.real_residual_model.parameters())
    gain_frozen = not model.residual_gain.requires_grad
    if (source_frozen != (arm == ARMS[1])
            or residual_frozen != (arm == ARMS[0])
            or gain_frozen != (arm == ARMS[0])):
        raise RuntimeError("FFAR mutually exclusive gradient boundary changed")


def load_causal_sim_endpoint(
        path: Path, device: torch.device,
        ) -> tuple[SpatiotemporalB210, object, dict[str, object]]:
    """Load only a sealed causal ST Sim raw-u10000 physical endpoint."""
    file_sha = sim_long.file_sha256(path)
    if file_sha != EXPECTED_START_FILE_SHA256:
        raise RuntimeError("sealed causal ST Sim10k file SHA changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, Mapping)
            or payload.get("schema") != st_lifecycle.ENDPOINT_SCHEMA
            or payload.get("lifecycle_schema") != st_lifecycle.SCHEMA
            or payload.get("arm") != "temporal_candidate"
            or payload.get("interaction_mode") != "causal_temporal_candidate"
            or payload.get("phase") != "sim"
            or payload.get("endpoint") != "raw_u10000"
            or int(payload.get("update", -1)) != 10_000
            or payload.get("uq") is not False
            or payload.get("checkpoint_selected") is not False):
        raise RuntimeError(
            "start is not the sealed causal ST Sim raw-u10000 endpoint")
    raw_config = dict(payload.get("model_config", {}))
    if (raw_config.pop("architecture", None) != "spatiotemporal_b210"
            or raw_config.pop("interaction_mode", None)
            != "causal_temporal_candidate"
            or raw_config.pop("flattened_history_bypass", None) is not False):
        raise RuntimeError("causal ST endpoint architecture changed")
    model = SpatiotemporalB210(
        SpatiotemporalB210Config(**raw_config),
        "causal_temporal_candidate").float()
    state = payload.get("model_state_dict")
    normalizer_values = payload.get("normalizer")
    if not isinstance(state, Mapping) or not isinstance(normalizer_values, Mapping):
        raise RuntimeError("causal ST endpoint lacks state or normalizer")
    model.load_state_dict(state, strict=True)
    state_sha = base.state_sha(_state_snapshot(model))
    normalizer_sha = base.state_sha(normalizer_values)
    endpoint_source = payload.get("source")
    if (state_sha != payload.get("model_state_sha256")
            or state_sha != EXPECTED_START_MODEL_STATE_SHA256
            or normalizer_sha != payload.get("normalizer_sha256")
            or normalizer_sha != EXPECTED_START_NORMALIZER_SHA256
            or not isinstance(endpoint_source, Mapping)
            or endpoint_source.get("sample_stream_sha256")
            != st_lifecycle.SIM_SCHEDULE_SHA256
            or sum(value.numel() for value in model.parameters())
            != EXPECTED_SINGLE_PARAMETERS
            or len(model.state_dict()) != EXPECTED_SINGLE_STATE_TENSORS):
        raise RuntimeError("causal ST endpoint tensor seal or topology changed")
    normalizer = _normalizer_from_payload(normalizer_values, device)
    physical_function_sha = _sealed_source_function_sha(
        _state_snapshot(model), normalizer_values)
    if physical_function_sha != EXPECTED_START_PHYSICAL_FUNCTION_SHA256:
        raise RuntimeError("sealed causal ST physical-function SHA changed")
    report = {
        "path": str(path.resolve()),
        "file_sha256": file_sha,
        "model_state_sha256": state_sha,
        "normalizer_sha256": normalizer_sha,
        "physical_function_sha256": physical_function_sha,
        "parameters": EXPECTED_SINGLE_PARAMETERS,
        "state_tensors": EXPECTED_SINGLE_STATE_TENSORS,
        "model_config": dict(payload["model_config"]),
        "source": dict(endpoint_source),
        "stage": "Sim_raw_u10000_no_score_selection",
    }
    return model.to(device), normalizer, report


def construct_pair(
        single: SpatiotemporalB210, source_normalizer, real_normalizer,
        source_normalizer_policy: str = DEFAULT_SOURCE_NORMALIZER_POLICY,
        ) -> tuple[dict[str, FullFunctionAnchoredSTB210], dict[str, object]]:
    source_values = _normalizer_payload(source_normalizer)
    real_values = _normalizer_payload(real_normalizer)
    aligned = (
        source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY)
    if (aligned
            and (base.state_sha(source_values) != base.state_sha(real_values)
                 or any(not torch.equal(source_values[name], real_values[name])
                        for name in real_values))):
        raise RuntimeError(
            "Real-affine-aligned anchor requires one bitwise common clean81 "
            "normalizer for source and residual")
    models = {
        arm: FullFunctionAnchoredSTB210.from_single_model(
            single, ARM_MODE[arm], source_values, real_values,
            source_normalizer_policy=source_normalizer_policy).float()
        for arm in ARMS
    }
    states = {arm: _state_snapshot(model) for arm, model in models.items()}
    state_shas = {arm: base.state_sha(state) for arm, state in states.items()}
    source_model_shas = {
        arm: _source_model_sha(model) for arm, model in models.items()}
    source_function_shas = {
        arm: _source_function_sha(model) for arm, model in models.items()}
    residual_shas = {
        arm: _residual_branch_sha(model) for arm, model in models.items()}
    sealed_parameter_sha = base.state_sha(_single_parameter_state(single))
    source_parameter_shas = {
        arm: base.state_sha(_source_parameter_state(model))
        for arm, model in models.items()}
    if (states[ARMS[0]].keys() != states[ARMS[1]].keys()
            or state_shas[ARMS[0]] != state_shas[ARMS[1]]
            or len(set(source_model_shas.values())) != 1
            or len(set(source_function_shas.values())) != 1
            or len(set(residual_shas.values())) != 1
            or set(source_parameter_shas.values()) != {sealed_parameter_sha}):
        raise RuntimeError("FFAR control/candidate initial states differ")
    for arm, model in models.items():
        _assert_model(model, arm)
        if torch.count_nonzero(model.residual_gain).item() != 0:
            raise RuntimeError("FFAR Real residual is not exact-zero-start")
    identity = {
        "bitwise_identical_initial_states": True,
        "initial_state_sha256": state_shas[ARMS[0]],
        "initial_source_model_state_sha256": source_model_shas[ARMS[0]],
        "initial_complete_source_function_sha256":
            source_function_shas[ARMS[0]],
        "initial_real_residual_branch_state_sha256": residual_shas[ARMS[0]],
        "parameter_counts": {
            arm: sum(value.numel() for value in model.parameters())
            for arm, model in models.items()},
        "trainable_parameter_counts": {
            arm: sum(value.numel() for value in model.parameters()
                     if value.requires_grad)
            for arm, model in models.items()},
        "state_tensor_counts": {
            arm: len(model.state_dict()) for arm, model in models.items()},
        "residual_gain_exact_zero": True,
        "control_complete_source_trainable": True,
        "control_real_residual_frozen": True,
        "candidate_complete_source_frozen": True,
        "candidate_real_residual_trainable": True,
    }
    if aligned:
        identity.update({
            "sealed_sim_parameter_state_sha256": sealed_parameter_sha,
            "initial_source_parameter_state_sha256": sealed_parameter_sha,
            "source_parameters_bitwise_equal_sealed_sim": True,
            "source_normalizer_policy": source_normalizer_policy,
            "source_normalizer_sha256": base.state_sha(source_values),
            "source_and_residual_normalizers_bitwise_equal": True,
        })
    return models, identity


def _canonicalize_ema_frozen_role(
        model: FullFunctionAnchoredSTB210,
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    result = _state_snapshot(state)
    raw = model.state_dict()
    if model.adaptation_mode == "frozen_source_candidate":
        frozen = lambda name: (
            name.startswith("source_model.") or name.startswith("_sim_"))
    else:
        frozen = lambda name: (
            name.startswith("real_residual_model.")
            or name == "residual_gain" or name.startswith("_real_"))
    for name in result:
        if frozen(name):
            result[name] = raw[name].detach().cpu().clone()
    return result


def _endpoint_config(model: FullFunctionAnchoredSTB210) -> dict[str, object]:
    config = dict(model.architecture_config())
    aligned = (
        model.source_normalizer_policy
        == REAL_AFFINE_SOURCE_NORMALIZER_POLICY)
    if (config.get("architecture")
            != "full_function_anchored_spatiotemporal_b210"
            or config.get("adaptation_mode") != model.adaptation_mode
            or (aligned and config.get("source_normalizer_policy")
                != REAL_AFFINE_SOURCE_NORMALIZER_POLICY)
            or (not aligned and "source_normalizer_policy" in config)
            or config.get("physical_fusion") is not True
            or config.get("flattened_history_bypass") is not False):
        raise RuntimeError("FFAR architecture contract changed")
    return config


def seal_endpoint(
        path: Path, *, model: FullFunctionAnchoredSTB210,
        state: Mapping[str, torch.Tensor], real_normalizer, arm: str,
        endpoint: str, update: int, source: Mapping[str, object],
        ) -> dict[str, object]:
    _assert_model(model, arm)
    policy = model.source_normalizer_policy
    aligned = policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
    if endpoint not in ENDPOINTS or update not in (4_000, 12_000):
        raise ValueError("unsupported FFAR endpoint")
    materialized = _state_snapshot(state)
    if len(materialized) != EXPECTED_STATE_TENSORS:
        raise RuntimeError("FFAR endpoint state inventory changed")
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
           for value in materialized.values()):
        raise FloatingPointError("refusing a non-finite FFAR endpoint")
    model._assert_domain_normalizers_consistent()
    source_sha = _source_function_sha(materialized)
    residual_sha = _residual_branch_sha(materialized)
    if (arm == ARMS[1]
            and source_sha != source[
                "initial_complete_source_function_sha256"]):
        raise RuntimeError("candidate complete source changed before seal")
    if (arm == ARMS[0]
            and residual_sha
            != source["initial_real_residual_branch_state_sha256"]):
        raise RuntimeError("control frozen Real residual changed before seal")
    real_values = _normalizer_payload(real_normalizer)
    source_values = model.normalizer_payload("sim")
    model_real_values = model.normalizer_payload("real")
    expected_source_normalizer_sha = (
        source["source_normalizer_sha256"] if aligned else
        source["common_sim_endpoint"]["normalizer_sha256"])
    if (base.state_sha(source_values) != expected_source_normalizer_sha
            or base.state_sha(model_real_values) != base.state_sha(real_values)
            or (aligned
                and base.state_sha(source_values) != base.state_sha(real_values))
            or (aligned and any(
                not torch.equal(source_values[name], real_values[name])
                for name in real_values))
            or any(not torch.equal(model_real_values[name], real_values[name])
                   for name in real_values)):
        raise RuntimeError("FFAR model/top-level domain normalizers differ")
    payload = {
        "schema": endpoint_schema(policy),
        "experiment_schema": experiment_schema(policy),
        "arm": arm,
        "adaptation_mode": ARM_MODE[arm],
        "endpoint": endpoint,
        "update": int(update),
        "model_config": _endpoint_config(model),
        "model_state_dict": materialized,
        "model_state_sha256": base.state_sha(materialized),
        "complete_source_function_sha256": source_sha,
        "real_residual_branch_state_sha256": residual_sha,
        "real_normalizer": real_values,
        "real_normalizer_sha256": base.state_sha(real_values),
        "source": dict(source),
        "uq": False,
        "checkpoint_selected": False,
    }
    payload.update({
        ("source_normalizer" if aligned else "sim_normalizer"):
            source_values,
        ("source_normalizer_sha256" if aligned else
         "sim_normalizer_sha256"): base.state_sha(source_values),
    })
    sim_long.atomic_torch_save(payload, path)
    result = {
        "path": str(path.resolve()),
        "file_sha256": sim_long.file_sha256(path),
        "model_state_sha256": payload["model_state_sha256"],
        "complete_source_function_sha256": source_sha,
        "real_residual_branch_state_sha256": residual_sha,
        "real_normalizer_sha256": payload["real_normalizer_sha256"],
        "arm": arm,
        "adaptation_mode": ARM_MODE[arm],
        "endpoint": endpoint,
        "update": int(update),
        "schema": endpoint_schema(policy),
    }
    result.update({
        ("source_normalizer_sha256" if aligned else
         "sim_normalizer_sha256"): base.state_sha(source_values),
    })
    return result


def _config_from_endpoint(
        values: Mapping[str, object],
        source_normalizer_policy: str = DEFAULT_SOURCE_NORMALIZER_POLICY):
    raw = dict(values)
    aligned = (
        source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY)
    metadata = {
        "architecture": "full_function_anchored_spatiotemporal_b210",
        "source_function": (
            "complete_sealed_Sim_weight_ST_B210" if aligned else
            "complete_sealed_Sim_physical_ST_B210"),
        "real_residual": "complete_causal_ST_B210_times_zero_start_gain",
        "residual_gain_initialization": "exact_zero_per_output_channel",
        "source_internal_normalizer": (
            "clean81_Real_affine_installed_on_sealed_Sim_weights"
            if aligned else "immutable_Sim"),
        "residual_internal_normalizer": "immutable_Real",
        "input_output_api": "Real_normalized",
        "physical_fusion": True,
        "unmeasured_pressure": (
            "trained_all3_compatibility_then_zeroed_by_official_evaluator"),
        "flattened_history_bypass": False,
    }
    if aligned:
        metadata["source_normalizer_policy"] = source_normalizer_policy
    for key, expected in metadata.items():
        if raw.pop(key, None) != expected:
            raise RuntimeError(f"FFAR endpoint metadata changed: {key}")
    mode = raw.pop("adaptation_mode", None)
    if mode not in FULL_FUNCTION_ADAPTATION_MODES:
        raise RuntimeError("FFAR endpoint adaptation mode changed")
    return SpatiotemporalB210Config(**raw), str(mode)


def load_endpoint(
        seal: Mapping[str, object], device: torch.device,
        ) -> tuple[FullFunctionAnchoredSTB210, object, dict[str, object]]:
    path = Path(str(seal["path"]))
    if (not path.is_file()
            or sim_long.file_sha256(path) != seal.get("file_sha256")):
        raise RuntimeError("FFAR endpoint file seal changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise RuntimeError("not an FFAR endpoint")
    if payload.get("schema") == REAL_AFFINE_ENDPOINT_SCHEMA:
        policy = REAL_AFFINE_SOURCE_NORMALIZER_POLICY
    elif payload.get("schema") == ENDPOINT_SCHEMA:
        policy = DEFAULT_SOURCE_NORMALIZER_POLICY
    else:
        raise RuntimeError("not an FFAR endpoint")
    aligned = policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
    if payload.get("experiment_schema") != experiment_schema(policy):
        raise RuntimeError("not an FFAR endpoint")
    for key in ("arm", "adaptation_mode", "endpoint", "update"):
        if payload.get(key) != seal.get(key):
            raise RuntimeError(f"FFAR endpoint {key} seal changed")
    arm = str(payload["arm"])
    if arm not in ARMS or payload["adaptation_mode"] != ARM_MODE[arm]:
        raise RuntimeError("FFAR endpoint arm/mode changed")
    endpoint_source = payload.get("source")
    if (payload.get("endpoint") not in ENDPOINTS
            or int(payload.get("update", -1)) not in (4_000, 12_000)
            or payload.get("uq") is not False
            or payload.get("checkpoint_selected") is not False
            or not isinstance(endpoint_source, Mapping)
            or endpoint_source.get("arm") not in (None, arm)
            or endpoint_source.get("adaptation_mode") not in (
                None, ARM_MODE[arm])):
        raise RuntimeError("FFAR endpoint protocol identity changed")
    source_key = "source_normalizer" if aligned else "sim_normalizer"
    source_sha_key = (
        "source_normalizer_sha256" if aligned else "sim_normalizer_sha256")
    source_values = payload.get(source_key)
    real_values = payload.get("real_normalizer")
    if (not isinstance(source_values, Mapping)
            or not isinstance(real_values, Mapping)
            or base.state_sha(source_values)
            != payload.get(source_sha_key)
            or base.state_sha(real_values) != payload.get("real_normalizer_sha256")
            or (aligned
                and base.state_sha(source_values) != base.state_sha(real_values))
            or (aligned and any(
                not torch.equal(source_values[name], real_values[name])
                for name in real_values))):
        raise RuntimeError("FFAR endpoint normalizer seal changed")
    common_start = endpoint_source.get("common_sim_endpoint")
    if (not isinstance(common_start, Mapping)
            or (aligned and endpoint_source.get("source_normalizer_policy")
                != policy)
            or (aligned and endpoint_source.get("source_normalizer_sha256")
                != payload.get(source_sha_key))
            or (not aligned and common_start.get("normalizer_sha256")
                != payload.get(source_sha_key))
            or common_start.get("file_sha256")
            != EXPECTED_START_FILE_SHA256
            or common_start.get("model_state_sha256")
            != EXPECTED_START_MODEL_STATE_SHA256
            or common_start.get("normalizer_sha256")
            != EXPECTED_START_NORMALIZER_SHA256
            or common_start.get("physical_function_sha256")
            != EXPECTED_START_PHYSICAL_FUNCTION_SHA256
            or (aligned
                and endpoint_source.get("initial_source_parameter_state_sha256")
                != endpoint_source.get("sealed_sim_parameter_state_sha256"))
            or (not aligned and common_start.get("physical_function_sha256")
                != endpoint_source.get(
                    "initial_complete_source_function_sha256"))):
        raise RuntimeError("FFAR endpoint common Sim provenance changed")
    config, mode = _config_from_endpoint(payload["model_config"], policy)
    if mode != ARM_MODE[arm]:
        raise RuntimeError("FFAR config/payload adaptation mode differs")
    model = FullFunctionAnchoredSTB210(
        config, mode, source_values, real_values,
        source_normalizer_policy=policy).float()
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("FFAR endpoint lacks model state")
    model.load_state_dict(state, strict=True)
    model._apply_gradient_boundary()
    model._assert_domain_normalizers_consistent()
    if (base.state_sha(model.normalizer_payload("sim"))
            != payload.get(source_sha_key)
            or base.state_sha(model.normalizer_payload("real"))
            != payload.get("real_normalizer_sha256")):
        raise RuntimeError("FFAR wrapper/top-level normalizers differ")
    _assert_model(model, arm)
    actual_sha = base.state_sha(_state_snapshot(model))
    if (actual_sha != payload.get("model_state_sha256")
            or actual_sha != seal.get("model_state_sha256")
            or _source_function_sha(model)
            != payload.get("complete_source_function_sha256")
            or _residual_branch_sha(model)
            != payload.get("real_residual_branch_state_sha256")):
        raise RuntimeError("FFAR endpoint model-state seal changed")
    if (arm == ARMS[1]
            and _source_function_sha(model)
            != endpoint_source.get(
                "initial_complete_source_function_sha256")):
        raise RuntimeError("loaded candidate frozen source changed")
    if (aligned and arm == ARMS[1]
            and base.state_sha(_source_parameter_state(model))
            != endpoint_source.get("sealed_sim_parameter_state_sha256")):
        raise RuntimeError("loaded candidate frozen Sim weights changed")
    if (arm == ARMS[0]
            and _residual_branch_sha(model)
            != endpoint_source.get(
                "initial_real_residual_branch_state_sha256")):
        raise RuntimeError("loaded control frozen residual changed")
    return (
        model.to(device),
        _normalizer_from_payload(real_values, device),
        dict(payload),
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resume_payload(
        *, model: FullFunctionAnchoredSTB210, optimizer, scheduler,
        ema: ema_lib.FP32EMAShadow, update: int,
        schedule: Sequence[Sequence[int]], source: Mapping[str, object],
        real_normalizer, device: torch.device,
        trace: Sequence[Mapping[str, object]],
        endpoint_seals: Mapping[str, Mapping[str, Mapping[str, object]]],
        contract: Mapping[str, object],
        ) -> dict[str, object]:
    state = _state_snapshot(model)
    policy = model.source_normalizer_policy
    return {
        "schema": resume_schema(policy),
        "update": int(update),
        "sample_cursor": int(update),
        "schedule": [[int(index) for index in batch] for batch in schedule],
        "schedule_sha256": real_long.schedule_sha(schedule),
        "source": dict(source),
        "real_normalizer": _normalizer_payload(real_normalizer),
        "real_normalizer_sha256": real_long._normalizer_sha(real_normalizer),
        "model_state_dict": state,
        "model_state_sha256": base.state_sha(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_seal": ema.seal(model),
        "rng": real_long.capture_rng(device),
        "trace": [dict(row) for row in trace],
        "endpoint_seals": {
            str(key): {name: dict(value) for name, value in endpoints.items()}
            for key, endpoints in endpoint_seals.items()},
        "training_contract": dict(contract),
        "device_type": device.type,
    }


def _restore_resume(
        path: Path, *, model: FullFunctionAnchoredSTB210,
        optimizer, scheduler, expected_source: Mapping[str, object],
        real_normalizer, device: torch.device,
        schedule: Sequence[Sequence[int]], contract: Mapping[str, object],
        ) -> tuple[
            int, ema_lib.FP32EMAShadow, list[dict[str, object]],
            dict[str, dict[str, dict[str, object]]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = model.source_normalizer_policy
    aligned = policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
    schedule_values = [[int(index) for index in batch] for batch in schedule]
    stored_real_values = payload.get("real_normalizer")
    if (payload.get("schema") != resume_schema(policy)
            or payload.get("training_contract") != dict(contract)
            or payload.get("device_type") != device.type
            or payload.get("source") != dict(expected_source)
            or payload.get("real_normalizer_sha256")
            != real_long._normalizer_sha(real_normalizer)
            or not isinstance(stored_real_values, Mapping)
            or base.state_sha(stored_real_values)
            != payload.get("real_normalizer_sha256")
            or payload.get("schedule") != schedule_values
            or payload.get("schedule_sha256")
            != real_long.schedule_sha(schedule)):
        raise RuntimeError("FFAR resume contract, source, or stream changed")
    cursor = int(payload["sample_cursor"])
    if cursor != int(payload["update"]) or not 0 <= cursor <= len(schedule):
        raise RuntimeError("FFAR resume sample cursor is invalid")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model._apply_gradient_boundary()
    model._assert_domain_normalizers_consistent()
    expected_real_values = _normalizer_payload(real_normalizer)
    model_real_values = model.normalizer_payload("real")
    if (base.state_sha(model_real_values)
            != base.state_sha(expected_real_values)
            or base.state_sha(model.normalizer_payload("sim"))
            != (expected_source["source_normalizer_sha256"] if aligned else
                expected_source["common_sim_endpoint"]["normalizer_sha256"])
            or (aligned and base.state_sha(model.normalizer_payload("sim"))
                != base.state_sha(model_real_values))):
        raise RuntimeError("FFAR resume domain normalizers changed")
    if payload.get("model_state_sha256") != base.state_sha(_state_snapshot(model)):
        raise RuntimeError("FFAR resume raw state seal changed")
    if (model.adaptation_mode == "frozen_source_candidate"
            and _source_function_sha(model)
            != expected_source[
                "initial_complete_source_function_sha256"]):
        raise RuntimeError("candidate source changed in FFAR resume")
    if (aligned and model.adaptation_mode == "frozen_source_candidate"
            and base.state_sha(_source_parameter_state(model))
            != expected_source["sealed_sim_parameter_state_sha256"]):
        raise RuntimeError("candidate Sim weights changed in FFAR resume")
    if (model.adaptation_mode == "trainable_source_control"
            and _residual_branch_sha(model)
            != expected_source[
                "initial_real_residual_branch_state_sha256"]):
        raise RuntimeError("control residual changed in FFAR resume")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    ema = ema_lib.FP32EMAShadow.from_seal(model, payload["ema_seal"])
    if ema.updates != cursor or int(scheduler.last_epoch) != cursor:
        raise RuntimeError("FFAR resume EMA or scheduler cursor changed")
    real_long.restore_rng(payload["rng"], device)
    trace = payload.get("trace")
    raw_seals = payload.get("endpoint_seals")
    if not isinstance(trace, list) or not isinstance(raw_seals, Mapping):
        raise RuntimeError("FFAR resume trace or endpoints are malformed")
    seals = {
        str(key): {
            str(endpoint): dict(value)
            for endpoint, value in endpoints.items()}
        for key, endpoints in raw_seals.items()
    }
    expected_milestones = {
        str(value) for value in milestone_updates(len(schedule))
        if value <= cursor}
    if set(seals) != expected_milestones:
        raise RuntimeError("FFAR resume milestone endpoint set changed")
    for update_key, endpoints in seals.items():
        if set(endpoints) != set(ENDPOINTS):
            raise RuntimeError("FFAR resume endpoint inventory changed")
        for seal in endpoints.values():
            endpoint_path = Path(str(seal["path"]))
            if (not endpoint_path.is_file()
                    or sim_long.file_sha256(endpoint_path)
                    != seal.get("file_sha256")):
                raise RuntimeError("FFAR resume endpoint file changed")
    return cursor, ema, [dict(row) for row in trace], seals


def _assert_optimizer_inventory(model: nn.Module, optimizer) -> None:
    expected = {
        id(parameter) for parameter in model.parameters()
        if parameter.requires_grad}
    actual = {
        id(parameter) for group in optimizer.param_groups
        for parameter in group["params"]}
    if actual != expected or len(actual) != sum(
            len(group["params"]) for group in optimizer.param_groups):
        raise RuntimeError("FFAR optimizer inventory is not exactly trainable state")


def _real_source(
        *, arm: str, start: Mapping[str, object], initial_source_sha: str,
        initial_residual_sha: str, source_parameter_sha: str | None,
        manifest_report: Mapping[str, object], clean_manifest: Path,
        real_normalizer, git: Mapping[str, object], real_seed: int,
        real_updates: int, schedule_sha256: str,
        source_normalizer_policy: str,
        ) -> dict[str, object]:
    result = {
        "arm": arm,
        "adaptation_mode": ARM_MODE[arm],
        "common_sim_endpoint": dict(start),
        "initial_complete_source_function_sha256": str(initial_source_sha),
        "initial_real_residual_branch_state_sha256": str(initial_residual_sha),
        "clean81_manifest_semantic_sha256": manifest_report["semantic_sha256"],
        "clean81_manifest_file_sha256": sim_long.file_sha256(clean_manifest),
        "real_normalizer_sha256": real_long._normalizer_sha(real_normalizer),
        "optimizer_windows": real_long.EXPECTED_OPTIMIZER_WINDOWS,
        "optimizer_stride": real_long.OPTIMIZER_STRIDE,
        "real_seed": int(real_seed),
        "real_updates": int(real_updates),
        "sample_stream_sha256": str(schedule_sha256),
        "git": dict(git),
    }
    if source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY:
        if source_parameter_sha is None:
            raise ValueError("aligned source requires a sealed Sim parameter SHA")
        result.update({
            "source_normalizer_policy": source_normalizer_policy,
            "source_normalizer_sha256": real_long._normalizer_sha(
                real_normalizer),
            "sealed_sim_parameter_state_sha256": str(source_parameter_sha),
            "initial_source_parameter_state_sha256": str(source_parameter_sha),
        })
    return result


def _gradient_audit(
        model: FullFunctionAnchoredSTB210, arm: str, update: int,
        ) -> dict[str, object]:
    source = tuple(model.source_model.parameters())
    residual = tuple(model.real_residual_model.parameters())
    source_has_gradient = all(value.grad is not None for value in source)
    source_any_gradient = any(value.grad is not None for value in source)
    source_nonzero_gradient = any(
        value.grad is not None and bool(torch.count_nonzero(value.grad))
        for value in source)
    residual_graph_gradient = all(value.grad is not None for value in residual)
    residual_any_gradient = any(value.grad is not None for value in residual)
    residual_gradients_finite = all(
        value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in residual)
    residual_nonzero_gradient = any(
        value.grad is not None and bool(torch.count_nonzero(value.grad))
        for value in residual)
    gain_gradient = model.residual_gain.grad
    gain_nonzero = (
        gain_gradient is not None and bool(torch.count_nonzero(gain_gradient)))
    if arm == ARMS[0]:
        valid = (
            source_has_gradient
            and source_nonzero_gradient
            and not residual_any_gradient
            and gain_gradient is None)
    else:
        expected_residual_nonzero = update >= 2
        valid = (
            not source_any_gradient
            and residual_graph_gradient
            and residual_gradients_finite
            and gain_nonzero
            and residual_nonzero_gradient == expected_residual_nonzero)
    if not valid:
        raise RuntimeError("FFAR update gradient boundary changed")
    return {
        "gradient_audit_update": int(update),
        "complete_source_has_gradient": source_has_gradient,
        "complete_source_any_gradient": source_any_gradient,
        "complete_source_has_nonzero_gradient": source_nonzero_gradient,
        "real_residual_graph_has_gradient": residual_graph_gradient,
        "real_residual_any_gradient": residual_any_gradient,
        "real_residual_gradients_finite": residual_gradients_finite,
        "real_residual_has_nonzero_gradient": residual_nonzero_gradient,
        "residual_gain_has_nonzero_gradient": gain_nonzero,
        "gradient_boundary_valid": valid,
    }


def train_or_resume_arm(
        *, arm: str, single: SpatiotemporalB210, source_normalizer,
        start: Mapping[str, object], dataset, real_normalizer,
        manifest_report: Mapping[str, object], clean_manifest: Path,
        git: Mapping[str, object], output_dir: Path, device: torch.device,
        resume: bool, checkpoint_every: int, real_updates: int,
        real_seed: int, schedule: Sequence[Sequence[int]],
        contract: Mapping[str, object], source_normalizer_policy: str,
        ) -> dict[str, object]:
    arm_dir = output_dir / "real" / arm
    result_path = arm_dir / "stage_result.json"
    model = FullFunctionAnchoredSTB210.from_single_model(
        single, ARM_MODE[arm], _normalizer_payload(source_normalizer),
        _normalizer_payload(real_normalizer),
        source_normalizer_policy=source_normalizer_policy).float().to(device)
    model.train()
    if ((arm == ARMS[1] and model.source_model.training)
            or (arm == ARMS[0] and model.real_residual_model.training)):
        raise RuntimeError("FFAR frozen function is not held in eval mode")
    _assert_model(model, arm)
    initial_source_sha = _source_function_sha(model)
    initial_residual_sha = _residual_branch_sha(model)
    source_parameter_sha = base.state_sha(_source_parameter_state(model))
    if (source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
            and source_parameter_sha
            != base.state_sha(_single_parameter_state(single))):
        raise RuntimeError("aligned source no longer has sealed Sim weights")
    source = _real_source(
        arm=arm, start=start, initial_source_sha=initial_source_sha,
        initial_residual_sha=initial_residual_sha,
        source_parameter_sha=source_parameter_sha,
        manifest_report=manifest_report, clean_manifest=clean_manifest,
        real_normalizer=real_normalizer, git=git, real_seed=real_seed,
        real_updates=real_updates,
        schedule_sha256=real_long.schedule_sha(schedule),
        source_normalizer_policy=source_normalizer_policy)
    arm_existed = arm_dir.exists()
    if resume and result_path.is_file():
        result = _read_json(result_path)
        if result.get("source") != source:
            raise RuntimeError("completed FFAR arm does not match protocol")
        for endpoints in result["endpoints"].values():
            for seal in endpoints.values():
                loaded, _normalizer, payload = load_endpoint(seal, device)
                if payload.get("source") != source:
                    raise RuntimeError("completed FFAR endpoint source changed")
                del loaded
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result
    arm_dir.mkdir(parents=True, exist_ok=True)
    if not resume and any(arm_dir.iterdir()):
        raise FileExistsError(arm_dir)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad), lr=real_long.LR,
        weight_decay=real_long.WEIGHT_DECAY)
    _assert_optimizer_inventory(model, optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=real_long.SCHEDULER_T_MAX)
    ema = ema_lib.FP32EMAShadow(model, real_long.EMA_DECAY)
    cursor = 0
    trace: list[dict[str, object]] = []
    endpoint_seals: dict[str, dict[str, dict[str, object]]] = {}
    resume_path = arm_dir / "resume.pt"
    if resume and resume_path.is_file():
        cursor, ema, trace, endpoint_seals = _restore_resume(
            resume_path, model=model, optimizer=optimizer,
            scheduler=scheduler, expected_source=source,
            real_normalizer=real_normalizer, device=device,
            schedule=schedule, contract=contract)
        _assert_optimizer_inventory(model, optimizer)
    elif resume and arm_existed and any(arm_dir.iterdir()):
        raise FileNotFoundError(f"no resumable FFAR state for {arm}")
    else:
        seed_real_recipe(real_seed, device)
    loss_args = argparse.Namespace(
        point_loss_mode="official_accuracy",
        mse_weight=real_long.MSE_WEIGHT,
        mse_warmup_updates=real_long.MSE_WARMUP_UPDATES)
    started = time.perf_counter()
    progress_updates = {
        *DIAGNOSTIC_UPDATES, *milestone_updates(real_updates),
        *range(checkpoint_every, real_updates + 1, checkpoint_every),
    }
    for cursor_index in range(cursor, real_updates):
        indices = schedule[cursor_index]
        history, target = zip(*(dataset[index] for index in indices))
        history_physical = torch.stack(history).to(device)
        target_physical = torch.stack(target).to(device)
        history_normalized = real_long.augment_real_history(
            history_physical, real_normalizer)
        _unused, target_normalized = real_normalizer.preprocess(
            history_physical, target_physical)
        optimizer.zero_grad(set_to_none=True)
        components = model.forward_components(history_normalized)
        prediction = components["prediction"]
        update = cursor_index + 1
        loss, details = mature_real.point_loss(
            loss_args, prediction, target_normalized, real_normalizer,
            real_long.ACCURACY_WEIGHTS, update)
        mse = real_long.real_mse(prediction, target_normalized)
        accuracy = (
            loss if update <= real_long.MSE_WARMUP_UPDATES
            else loss - real_long.MSE_WEIGHT * mse)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite FFAR loss at u{update}")
        loss.backward()
        mechanical: dict[str, object] = {}
        if update in (1, 2):
            mechanical = _gradient_audit(model, arm, update)
            mechanical.update({
                "residual_gain": model.residual_gain.detach().cpu().tolist(),
                "prediction_minus_source_max_abs": float((
                    components["prediction_physical"]
                    - components["source_physical"]
                ).abs().max().detach()),
                "source_uses_sim_normalizer": (
                    source_normalizer_policy
                    == DEFAULT_SOURCE_NORMALIZER_POLICY),
                "source_function_sha256": _source_function_sha(model),
            })
            if (source_normalizer_policy
                    == REAL_AFFINE_SOURCE_NORMALIZER_POLICY):
                mechanical["source_normalizer_policy"] = (
                    source_normalizer_policy)
            if (update == 1
                    and mechanical["prediction_minus_source_max_abs"] != 0.0):
                raise RuntimeError("FFAR residual is not exact zero at update0")
        gradient = nn.utils.clip_grad_norm_(
            model.parameters(), real_long.GRAD_CLIP)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError(f"non-finite FFAR gradient at u{update}")
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=update)
        scheduler.step()
        if update in progress_updates:
            row = {
                "update": update,
                "loss": float(loss.detach()),
                "mse_all3": float(mse.detach()),
                "accuracy": float(accuracy.detach()),
                "grad_norm": float(gradient.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "diagnostic_only": update not in milestone_updates(real_updates),
                **mechanical,
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({
                "event": "ffar_st_real_progress", "arm": arm, **row,
            }, sort_keys=True), flush=True)
        if update in milestone_updates(real_updates):
            raw_state = _state_snapshot(model)
            ema_state = _canonicalize_ema_frozen_role(
                model, ema.materialize_model_state(model))
            endpoint_seals[str(update)] = {
                "raw": seal_endpoint(
                    arm_dir / f"terminal_raw_u{update:05d}.pt",
                    model=model, state=raw_state,
                    real_normalizer=real_normalizer, arm=arm,
                    endpoint="raw", update=update, source=source),
                "ema": seal_endpoint(
                    arm_dir / f"terminal_ema999_u{update:05d}.pt",
                    model=model, state=ema_state,
                    real_normalizer=real_normalizer, arm=arm,
                    endpoint="ema", update=update, source=source),
            }
        if update % checkpoint_every == 0 or update == real_updates:
            real_long.atomic_torch_save(
                resume_path,
                _resume_payload(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    ema=ema, update=update, schedule=schedule,
                    source=source, real_normalizer=real_normalizer,
                    device=device, trace=trace,
                    endpoint_seals=endpoint_seals, contract=contract))
    expected = {str(value) for value in milestone_updates(real_updates)}
    if set(endpoint_seals) != expected:
        raise RuntimeError("FFAR milestone endpoint inventory is incomplete")
    result = {
        "stage": "real",
        "arm": arm,
        "adaptation_mode": ARM_MODE[arm],
        "source": source,
        "training_contract": dict(contract),
        "training": {
            "updates": real_updates,
            "seed": real_seed,
            "batch_size": real_long.BATCH_SIZE,
            "sample_schedule_sha256": real_long.schedule_sha(schedule),
            "sample_cursor": real_updates,
            "trace": trace,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR_Tmax_12000",
            "ema_decay": real_long.EMA_DECAY,
        },
        "endpoints": endpoint_seals,
        "final_complete_source_function_sha256": _source_function_sha(model),
        "final_real_residual_branch_state_sha256":
            _residual_branch_sha(model),
        "complete_source_unchanged": (
            _source_function_sha(model) == initial_source_sha),
        "real_residual_unchanged": (
            _residual_branch_sha(model) == initial_residual_sha),
        "checkpoint_selection": False,
    }
    sim_long.atomic_json_save(result, result_path)
    del model, ema
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def prescribed_barrier_keys(milestones: Sequence[int]) -> set[str]:
    values = tuple(int(value) for value in milestones)
    if values not in ((4_000,), (4_000, 12_000)):
        raise ValueError("invalid FFAR barrier milestones")
    return {
        "start/sim_source",
        *(f"real/{arm}/u{update}/{endpoint}"
          for arm in ARMS for update in values for endpoint in ENDPOINTS),
    }


class EndpointAccessAudit:
    """Reject clean716 access until every fixed endpoint is sealed."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self._opened: dict[str, set[str]] = {"fit": set(), "clean716": set()}
        self.endpoint_seals: dict[str, dict[str, object]] | None = None

    def record_endpoint_seals(
            self, seals: Mapping[str, Mapping[str, object]],
            expected: set[str]) -> None:
        if self.endpoint_seals is not None:
            raise RuntimeError("FFAR endpoint seals were registered twice")
        if set(seals) != expected:
            raise RuntimeError("clean716 barrier endpoint inventory changed")
        verified: dict[str, dict[str, object]] = {}
        resolved_paths: set[Path] = set()
        for key, seal in seals.items():
            path = Path(str(seal["path"]))
            resolved = path.resolve()
            if resolved in resolved_paths:
                raise RuntimeError("FFAR barrier endpoint paths are not unique")
            resolved_paths.add(resolved)
            if (not path.is_file()
                    or sim_long.file_sha256(path) != seal.get("file_sha256")):
                raise RuntimeError(f"FFAR endpoint changed before barrier: {key}")
            if key != "start/sim_source":
                parts = key.split("/")
                if (len(parts) != 4 or parts[0] != "real"
                        or seal.get("arm") != parts[1]
                        or int(seal.get("update", -1))
                        != int(parts[2].removeprefix("u"))
                        or seal.get("endpoint") != parts[3]):
                    raise RuntimeError(
                        f"FFAR barrier endpoint identity differs: {key}")
            verified[key] = dict(seal)
        self.endpoint_seals = verified

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if filename == real_long.EXCLUDED_BASENAME:
            raise RuntimeError("excluded Real trajectory reached FFAR opener")
        if phase not in self._opened:
            raise ValueError(f"unsupported FFAR access phase {phase!r}")
        if filename in self._opened[phase]:
            raise RuntimeError(f"invalid repeated FFAR Real open: {filename}")
        if phase == "fit" and self.endpoint_seals is not None:
            raise RuntimeError("FFAR fit prefix opened after endpoint sealing")
        if phase == "clean716":
            self.assert_barrier_intact()

    def assert_barrier_intact(self) -> None:
        if self.endpoint_seals is None:
            raise RuntimeError(
                "clean716 opened before every FFAR endpoint seal")
        for key, seal in self.endpoint_seals.items():
            path = Path(str(seal["path"]))
            if (not path.is_file()
                    or sim_long.file_sha256(path)
                    != seal.get("file_sha256")):
                raise RuntimeError(
                    f"FFAR endpoint changed after barrier: {key}")

    def record_open(
            self, row: Mapping[str, object], phase: str, boundary: int) -> None:
        filename = str(row["file"])
        self._opened[phase].add(filename)
        self.events.append({
            "file": filename,
            "phase": phase,
            "channels_opened": ["u", "v"],
            "pressure_opened": False,
            "max_read_frame_exclusive": int(boundary),
            "all_endpoints_sealed_before_open": self.endpoint_seals is not None,
        })


def _score_delta(
        candidate: Mapping[str, float], control: Mapping[str, float],
        ) -> dict[str, float]:
    keys = (*OFFICIAL_POINT_KEYS, "public_point_contribution_46_10_9")
    return {key: float(candidate[key]) - float(control[key]) for key in keys}


def _compare_panel(
        endpoints: Mapping[str, Mapping[str, object]],
        control_key: str, candidate_key: str,
        ) -> dict[str, object]:
    control = endpoints[control_key]
    candidate = endpoints[candidate_key]
    return {
        "control": control_key,
        "candidate": candidate_key,
        "pooled_delta_candidate_minus_control": _score_delta(
            candidate["pooled"], control["pooled"]),
        "by_aoa_delta_candidate_minus_control": {
            aoa: _score_delta(
                candidate["by_aoa"][aoa], control["by_aoa"][aoa])
            for aoa in ("0", "5", "10", "15", "20")},
    }


def load_clean716_after_barrier(
        *, evaluation_manifest: Path,
        fit_rows: Sequence[Mapping[str, object]], real_dir: Path,
        access: EndpointAccessAudit,
        ) -> tuple[str, object, list[int]]:
    """Perform the first manifest-content and tail-field reads after barrier."""
    access.assert_barrier_intact()
    manifest_sha256 = sim_long.file_sha256(evaluation_manifest)
    tail_rows = real_long.load_clean716_rows(evaluation_manifest, fit_rows)
    tail_trajectories, tail_windows = real_long.load_real_rows(
        real_dir, tail_rows, phase="clean716", audit=access)
    tail_dataset = real_long.RealWindowDataset(
        tail_trajectories, tail_windows)
    tail_aoas = [
        int(tail_rows[window.trajectory]["aoa"]) for window in tail_windows]
    return manifest_sha256, tail_dataset, tail_aoas


def evaluate_after_all_seals(
        *, seals: Mapping[str, Mapping[str, object]],
        milestones: Sequence[int], tail_dataset,
        tail_aoas: Sequence[int], device: torch.device,
        ) -> dict[str, object]:
    order = [
        f"real/{arm}/u{update}/{endpoint}"
        for update in milestones for endpoint in ENDPOINTS for arm in ARMS
    ]
    endpoints: dict[str, dict[str, object]] = {}
    for key in order:
        model, normalizer, _payload = load_endpoint(seals[key], device)
        endpoints[key] = real_long.evaluate_clean716(
            model, tail_dataset, normalizer, tail_aoas, device)
        del model, normalizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    target_hashes = {value["target_sha256"] for value in endpoints.values()}
    if len(target_hashes) != 1:
        raise RuntimeError("FFAR endpoints did not use one clean716 target order")
    comparisons = {}
    for update in milestones:
        for endpoint in ENDPOINTS:
            comparisons[f"u{update}_{endpoint}"] = _compare_panel(
                endpoints,
                f"real/{ARMS[0]}/u{update}/{endpoint}",
                f"real/{ARMS[1]}/u{update}/{endpoint}")
    return {
        "endpoint_order": order,
        "endpoints": endpoints,
        "comparisons": comparisons,
        "common_target_sha256": next(iter(target_hashes)),
        "score_gate": None,
        "checkpoint_selection": False,
    }


def mechanical_alignment_metrics(
        *, single: SpatiotemporalB210, sim_normalizer, aligned_source,
        real_normalizer, history_physical: torch.Tensor,
        target_physical: torch.Tensor,
        ) -> dict[str, float]:
    """Measure only the source affine intervention on one fixed Real batch."""
    history_real, target_real = real_normalizer.preprocess(
        history_physical, target_physical)
    with torch.inference_mode():
        sim_history, _unused = sim_normalizer.preprocess(
            history_physical, target_physical)
        sealed_sim_physical = (
            single.eval()(sim_history) * sim_normalizer.std_target
            + sim_normalizer.mean_target)
        sealed_sim_real_normalized = (
            sealed_sim_physical - real_normalizer.mean_target
        ) / real_normalizer.std_target
        aligned_physical = aligned_source.eval().source_physical(
            history_physical)
        aligned_real_normalized = (
            aligned_physical - real_normalizer.mean_target
        ) / real_normalizer.std_target
    sealed_mse = real_long.real_mse(
        sealed_sim_real_normalized, target_real)
    sealed_v_mse = (
        sealed_sim_real_normalized[..., 1]
        - target_real[..., 1]).square().mean()
    return {
        "sealed_sim_physical_mse_all3": float(sealed_mse),
        "real_affine_aligned_mse_all3": float(
            real_long.real_mse(aligned_real_normalized, target_real)),
        "sealed_sim_v_mse": float(sealed_v_mse),
        "sealed_sim_v_share_of_total_squared_error_percent": float(
            100.0 * sealed_v_mse / (3.0 * sealed_mse)),
        "sealed_sim_prediction_v_std": float(
            sealed_sim_physical[..., 1].std()),
        "real_affine_prediction_v_std": float(aligned_physical[..., 1].std()),
        "target_v_std": float(target_physical[..., 1].std()),
    }


def checkpoint_smoke(
        start_sim: Path, output_dir: Path, device: torch.device,
        ) -> dict[str, object]:
    """Original FFAR no-data checkpoint smoke (sealed-Sim default)."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if device.type != "cpu":
        raise ValueError("checkpoint smoke is intentionally CPU-only")
    output_dir.mkdir(parents=True)
    single, sim_normalizer, start = load_causal_sim_endpoint(start_sim, device)
    real_normalizer = base.ChannelNormalizer(
        torch.tensor([.1, -.2, 0.]), torch.tensor([.3, -.4, 0.]),
        torch.tensor([1.2, .8, 1.]), torch.tensor([.9, 1.1, 1.]),
        device)
    models, identity = construct_pair(single, sim_normalizer, real_normalizer)
    physical = torch.zeros(1, 20, 8, 8, 3)
    with torch.inference_mode():
        expected = models[ARMS[0]].source_physical(physical)
        outputs = {
            arm: model.eval().forward_physical(physical)["prediction_physical"]
            for arm, model in models.items()}
    if any(not torch.equal(value, expected) for value in outputs.values()):
        raise RuntimeError("FFAR update-zero source function changed")
    expected_axis_calls = sum(single.config.stage_blocks)
    if any(len(model.execution_contract()["source"]) != expected_axis_calls
           or len(model.execution_contract()["real_residual"])
           != expected_axis_calls
           for model in models.values()):
        raise RuntimeError("FFAR did not execute both causal ST functions")
    arm = ARMS[1]
    source = {
        "common_sim_endpoint": start,
        "initial_complete_source_function_sha256":
            identity["initial_complete_source_function_sha256"],
        "initial_real_residual_branch_state_sha256":
            identity["initial_real_residual_branch_state_sha256"]}
    seal = seal_endpoint(
        output_dir / "production_candidate_roundtrip.pt",
        model=models[arm], state=models[arm].state_dict(),
        real_normalizer=real_normalizer, arm=arm,
        endpoint="raw", update=4_000, source=source)
    loaded, loaded_normalizer, payload = load_endpoint(seal, device)
    passed = (
        _source_function_sha(loaded) == start["physical_function_sha256"]
        and seal["model_state_sha256"] == base.state_sha(_state_snapshot(loaded))
        and payload["real_normalizer_sha256"]
        == base.state_sha(_normalizer_payload(loaded_normalizer))
        and payload["checkpoint_selected"] is False)
    result = {
        "schema": SCHEMA,
        "smoke": "PASS" if passed else "FAIL",
        "data_files_opened": 0,
        "start": start,
        "identity": identity,
        "update0_output_bitwise_equals_complete_source": True,
        "causal_axis_calls_each_function": expected_axis_calls,
        "endpoint": seal,
    }
    sim_long.atomic_json_save(result, output_dir / "result.json")
    return result


def real_affine_checkpoint_smoke(
        start_sim: Path, output_dir: Path, device: torch.device,
        *, real_dir: Path, clean_manifest: Path,
        ) -> dict[str, object]:
    """Production aligned-function, update-zero, and round-trip smoke.

    This reads only the optimizer-visible clean81 prefix.  No evaluation
    manifest is accepted by the function, so clean716 is structurally absent.
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if device.type != "cpu":
        raise ValueError("checkpoint smoke is intentionally CPU-only")
    output_dir.mkdir(parents=True)
    single, sim_normalizer, start = load_causal_sim_endpoint(start_sim, device)
    rows, manifest_report = real_long.load_clean_manifest(clean_manifest)
    access = EndpointAccessAudit()
    trajectories, sparse_windows = real_long.load_real_rows(
        real_dir, rows, phase="fit", audit=access)
    real_normalizer, real_normalizer_report = (
        real_long.normalizer_from_sparse_windows(
            trajectories, sparse_windows, device))
    optimizer_windows = real_long.dense_windows(trajectories)
    if len(optimizer_windows) != real_long.EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError("checkpoint smoke optimizer-window count changed")
    dataset = real_long.RealWindowDataset(trajectories, optimizer_windows)
    first_batch = make_sample_schedule(
        len(dataset), 4_000, real_long.BATCH_SIZE, 4)[0]
    history, target = zip(*(dataset[index] for index in first_batch))
    history_physical = torch.stack(history).to(device)
    target_physical = torch.stack(target).to(device)
    models, identity = construct_pair(
        single, real_normalizer, real_normalizer,
        REAL_AFFINE_SOURCE_NORMALIZER_POLICY)
    mechanical = {
        "sample_schedule_seed": 4,
        "first_batch_indices": [int(index) for index in first_batch],
        **mechanical_alignment_metrics(
            single=single, sim_normalizer=sim_normalizer,
            aligned_source=models[ARMS[0]], real_normalizer=real_normalizer,
            history_physical=history_physical,
            target_physical=target_physical),
    }
    reported = {
        key: round(mechanical[key], MECHANICAL_REPORTED_DECIMALS[key])
        for key in MECHANICAL_EXPECTED}
    if reported != MECHANICAL_EXPECTED:
        raise RuntimeError(
            f"Real-affine mechanical acceptance changed: {mechanical}")
    mechanical.update({
        "expected": dict(MECHANICAL_EXPECTED),
        "reported": reported,
        "reported_decimals": dict(MECHANICAL_REPORTED_DECIMALS),
        "acceptance": "PASS",
    })
    physical = torch.zeros(1, 20, 8, 8, 3)
    with torch.inference_mode():
        expected = models[ARMS[0]].source_physical(physical)
        outputs = {
            arm: model.eval().forward_physical(physical)["prediction_physical"]
            for arm, model in models.items()}
    if any(not torch.equal(value, expected) for value in outputs.values()):
        raise RuntimeError("FFAR update-zero source function changed")
    expected_axis_calls = sum(single.config.stage_blocks)
    if any(len(model.execution_contract()["source"]) != expected_axis_calls
           or len(model.execution_contract()["real_residual"])
           != expected_axis_calls
           for model in models.values()):
        raise RuntimeError("FFAR did not execute both causal ST functions")
    arm = ARMS[1]
    source = {
        "common_sim_endpoint": start,
        "initial_complete_source_function_sha256":
            identity["initial_complete_source_function_sha256"],
        "initial_real_residual_branch_state_sha256":
            identity["initial_real_residual_branch_state_sha256"],
        "source_normalizer_policy": REAL_AFFINE_SOURCE_NORMALIZER_POLICY,
        "source_normalizer_sha256": identity["source_normalizer_sha256"],
        "sealed_sim_parameter_state_sha256":
            identity["sealed_sim_parameter_state_sha256"],
        "initial_source_parameter_state_sha256":
            identity["initial_source_parameter_state_sha256"]}
    seal = seal_endpoint(
        output_dir / "production_candidate_roundtrip.pt",
        model=models[arm], state=models[arm].state_dict(),
        real_normalizer=real_normalizer, arm=arm,
        endpoint="raw", update=4_000, source=source)
    loaded, loaded_normalizer, payload = load_endpoint(seal, device)
    passed = (
        _source_function_sha(loaded)
        == identity["initial_complete_source_function_sha256"]
        and base.state_sha(_source_parameter_state(loaded))
        == identity["sealed_sim_parameter_state_sha256"]
        and payload["source_normalizer_sha256"]
        == payload["real_normalizer_sha256"]
        and seal["model_state_sha256"] == base.state_sha(_state_snapshot(loaded))
        and payload["real_normalizer_sha256"]
        == base.state_sha(_normalizer_payload(loaded_normalizer))
        and payload["checkpoint_selected"] is False)
    result = {
        "schema": REAL_AFFINE_SCHEMA,
        "smoke": "PASS" if passed else "FAIL",
        "data_files_opened": len(access.events),
        "clean81_files_opened": len(access.events),
        "clean716_files_opened": 0,
        "start": start,
        "clean81_manifest": manifest_report,
        "real_normalizer": real_normalizer_report,
        "identity": identity,
        "mechanical_acceptance": mechanical,
        "update0_output_bitwise_equals_complete_source": True,
        "causal_axis_calls_each_function": expected_axis_calls,
        "endpoint": seal,
    }
    sim_long.atomic_json_save(result, output_dir / "result.json")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    source_normalizer_policy = getattr(
        args, "source_normalizer_policy", DEFAULT_SOURCE_NORMALIZER_POLICY)
    if (not args.execute
            or args.acknowledge_protocol
            != acknowledgement(source_normalizer_policy)):
        raise PermissionError("formal FFAR run requires exact acknowledgement")
    required = (
        args.start_sim, args.real_dir, args.clean_manifest,
        args.evaluation_manifest, args.output_dir)
    if any(value is None for value in required):
        raise ValueError("formal FFAR run is missing a required path")
    if args.real_seed not in REAL_SEEDS or args.checkpoint_every <= 0:
        raise ValueError("invalid FFAR seed or checkpoint interval")
    device = torch.device(args.device)
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal FFAR run requires clean Git: {git['porcelain']}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal FFAR run requires CUDA")
    configure_strict_fp32(device)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    if not args.output_dir.exists() and args.resume:
        raise FileNotFoundError("--resume requires an existing output directory")
    # Stat only.  Its content and hash remain unavailable until every endpoint
    # has been sealed for both arms.
    if not args.evaluation_manifest.is_file():
        raise FileNotFoundError(args.evaluation_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    single, sim_normalizer, start = load_causal_sim_endpoint(
        args.start_sim, torch.device("cpu"))
    rows, manifest_report = real_long.load_clean_manifest(args.clean_manifest)
    access = EndpointAccessAudit()
    trajectories, sparse_windows = real_long.load_real_rows(
        args.real_dir, rows, phase="fit", audit=access)
    if any(tuple(value.shape[1:3]) != (32, 64) for value in trajectories):
        raise RuntimeError("formal FFAR Real grid must be exactly 32x64")
    real_normalizer, real_normalizer_report = (
        real_long.normalizer_from_sparse_windows(
            trajectories, sparse_windows, device))
    optimizer_windows = real_long.dense_windows(trajectories)
    if len(optimizer_windows) != real_long.EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError("clean81 stride-5 optimizer window count changed")
    dataset = real_long.RealWindowDataset(trajectories, optimizer_windows)
    schedule = make_sample_schedule(
        len(dataset), args.real_updates, real_long.BATCH_SIZE, args.real_seed)
    schedule_sha = real_long.schedule_sha(schedule)
    if schedule_sha != EXPECTED_REAL_STREAM_SHA256[
            (args.real_updates, args.real_seed)]:
        raise RuntimeError("formal FFAR Real sample stream changed")
    contract = training_contract(
        real_updates=args.real_updates, real_seed=args.real_seed,
        schedule_sha256=schedule_sha,
        source_normalizer_policy=source_normalizer_policy)

    source_normalizer = (
        real_normalizer
        if source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY
        else sim_normalizer)
    initial_models, initial_identity = construct_pair(
        single, source_normalizer, real_normalizer,
        source_normalizer_policy)
    if source_normalizer_policy == REAL_AFFINE_SOURCE_NORMALIZER_POLICY:
        if (initial_identity["initial_source_parameter_state_sha256"]
                != base.state_sha(_single_parameter_state(single))
                or initial_identity["source_normalizer_sha256"]
                != real_long._normalizer_sha(real_normalizer)
                or initial_identity["initial_complete_source_function_sha256"]
                == start["physical_function_sha256"]):
            raise RuntimeError(
                "Real-affine source does not preserve sealed Sim weights while "
                "changing the source coordinate system")
    elif (initial_identity["initial_source_model_state_sha256"]
            != start["model_state_sha256"]
            or initial_identity["initial_complete_source_function_sha256"]
            != start["physical_function_sha256"]):
        raise RuntimeError("FFAR source differs from sealed Sim start")
    del initial_models
    gc.collect()

    arm_results: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        arm_results[arm] = train_or_resume_arm(
            arm=arm, single=single, source_normalizer=source_normalizer,
            start=start, dataset=dataset, real_normalizer=real_normalizer,
            manifest_report=manifest_report,
            clean_manifest=args.clean_manifest, git=git,
            output_dir=args.output_dir, device=device, resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            real_updates=args.real_updates, real_seed=args.real_seed,
            schedule=schedule, contract=contract,
            source_normalizer_policy=source_normalizer_policy)

    milestones = milestone_updates(args.real_updates)
    all_seals = {
        "start/sim_source": {
            "path": start["path"],
            "file_sha256": start["file_sha256"],
            "physical_function_sha256": start["physical_function_sha256"],
        },
        **{
            f"real/{arm}/u{update}/{endpoint}":
                arm_results[arm]["endpoints"][str(update)][endpoint]
            for arm in ARMS for update in milestones for endpoint in ENDPOINTS
        },
    }
    access.record_endpoint_seals(
        all_seals, prescribed_barrier_keys(milestones))
    evaluation_manifest_sha256, tail_dataset, tail_aoas = (
        load_clean716_after_barrier(
            evaluation_manifest=args.evaluation_manifest,
            fit_rows=rows, real_dir=args.real_dir, access=access))
    evaluation = evaluate_after_all_seals(
        seals=all_seals, milestones=milestones,
        tail_dataset=tail_dataset, tail_aoas=tail_aoas, device=device)

    result = {
        "schema": experiment_schema(source_normalizer_policy),
        "contract": contract,
        "git": git,
        "start": start,
        "initial_identity": initial_identity,
        "clean81_manifest": manifest_report,
        "real_normalizer": real_normalizer_report,
        "real_optimizer_windows": len(optimizer_windows),
        "actual_real_seed": args.real_seed,
        "actual_sample_stream_sha256": schedule_sha,
        "arms": arm_results,
        "endpoint_seals": all_seals,
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "file_sha256": evaluation_manifest_sha256,
            "first_content_read_after_all_prescribed_endpoint_seals": True,
        },
        "clean716_evaluation": evaluation,
        "access_audit": access.events,
        "clean716_opened_before_all_endpoint_seals": False,
        "wall_seconds": time.perf_counter() - started,
        "score_gate": None,
        "early_stopping": False,
        "checkpoint_selection": False,
        "uq": False,
        "submission_authorized": False,
    }
    sim_long.atomic_json_save(result, args.output_dir / "result.json")
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps(
            contract_for_args(
                args.real_updates, args.real_seed,
                args.source_normalizer_policy),
            sort_keys=True))
        return
    if args.output_dir is None or args.start_sim is None:
        raise ValueError("this mode requires --start-sim and --output-dir")
    if args.mode == "checkpoint-smoke":
        if (args.source_normalizer_policy
                == REAL_AFFINE_SOURCE_NORMALIZER_POLICY):
            if args.real_dir is None or args.clean_manifest is None:
                raise ValueError(
                    "Real-affine checkpoint smoke requires --real-dir and "
                    "--clean-manifest")
            result = real_affine_checkpoint_smoke(
                args.start_sim, args.output_dir, torch.device(args.device),
                real_dir=args.real_dir, clean_manifest=args.clean_manifest)
        else:
            result = checkpoint_smoke(
                args.start_sim, args.output_dir, torch.device(args.device))
        print(json.dumps({
            "event": "ffar_st_checkpoint_smoke_complete",
            "smoke": result["smoke"], "endpoint": result["endpoint"],
        }, sort_keys=True))
        return
    result = run(args)
    print(json.dumps({
        "event": "ffar_st_matched_real_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "comparisons": result["clean716_evaluation"]["comparisons"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
