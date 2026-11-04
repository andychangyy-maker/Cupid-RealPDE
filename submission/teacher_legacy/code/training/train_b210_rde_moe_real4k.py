#!/usr/bin/env python3
"""Formal RDE-MoE Real4k comparison from one released-data Sim endpoint.

R0 keeps the mature single decoder.  R1 and R2 clone the complete audited
shared-decoder scope at the Sim->Real boundary.  Both execute the same frozen,
prefix-only history router; R1 fixes the blend to 0.5 and R2 uses its soft
probability.  No metadata is accepted by model inference.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1
from models.ensemble_unet2d import (
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
    REGIME_EXPERT_DECODER_MODES,
)
from models.unet2d import history_regime_summary


EXPERIMENT = "b210_rde_moe_real4k"
SCHEMA = "b210_rde_moe_real4k_v1"
ENDPOINT_SCHEMA = "b210_rde_moe_real4k_endpoint_v1"
ROUTER_SCHEMA = "b210_rde_moe_prefix_router_v1"
ACKNOWLEDGEMENT = "RUN_B210_RDE_MOE_REAL4K_V1"

ARM_ORDER = ("r0_single", "r1_fixed_average", "r2_history_router")
ARM_MODE = {
    "r0_single": "none",
    "r1_fixed_average": "regime_expert_fixed_average",
    "r2_history_router": "regime_expert_history_router",
}
ENDPOINT_ORDER = ("raw", "ema")

UPDATES = 4_000
SCHEDULER_T_MAX = 12_000
EMA_DECAY = 0.999
EXPECTED_START_ITERATION = 4_000
EXPECTED_START_STATUS = "PASS_B210_STANDARD24K_MATCHED_COMPUTE_SIM4000"
EXPECTED_START_METHOD = (
    "b210_standard_stride20_20k_then_standard_stride20_4k_v1")
EXPECTED_START_SHA256 = (
    "4b56454ce64e301a9898dc1c066d23180de587cbebc7c53786fd3ca126981e68")
EXPECTED_MANIFEST_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
EXPECTED_EVALUATION_MANIFEST_SHA256 = (
    "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
EXPECTED_BASE_PARAMETERS = 178_705_575
EXPECTED_AUDITED_SCOPE_PARAMETERS = 34_543_398
EXPECTED_ROUTER_PARAMETERS = 1_313
EXPECTED_EXPERT_PARAMETERS = (
    EXPECTED_BASE_PARAMETERS
    + EXPECTED_AUDITED_SCOPE_PARAMETERS
    + EXPECTED_ROUTER_PARAMETERS)
EXPECTED_CONTROLLER_PARAMETERS = ft12.EXPECTED_CONTROLLER_PARAMETERS
EXPECTED_DENSE_WINDOWS = 9_933
EXPECTED_ROUTER_NEGATIVES = 6_065
EXPECTED_ROUTER_POSITIVES = 3_868

ROUTER_NEGATIVE_AOAS = (5, 10, 15)
ROUTER_POSITIVE_AOAS = (0, 20)
ROUTER_SEED = 20_260_904
ROUTER_UPDATES = 400
ROUTER_LR = 3.0e-3
ROUTER_WEIGHT_DECAY = 1.0e-4
ROUTER_POS_WEIGHT = EXPECTED_ROUTER_NEGATIVES / EXPECTED_ROUTER_POSITIVES
ROUTER_AUDIT_SHA256 = (
    "38ae79f36a8f08af90be431b5f35b19d85c1e04d166f497472c25ab8198c199f")

POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--wall-budget-seconds", type=float, default=None,
        help=("optional operator safety timeout; omitted means no deadline and "
              "is the formal default"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def git_report() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]

    def run(*values: str) -> str:
        return subprocess.run(
            ["git", *values], cwd=root, check=True, capture_output=True,
            text=True).stdout.strip()

    porcelain = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "clean": not bool(porcelain),
        "porcelain": porcelain.splitlines(),
    }


def parse_start_payload(payload: Mapping[str, object]) -> tuple[
        EnsembleUNet2dConfig, Mapping[str, torch.Tensor], dict[str, object]]:
    if (payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != EXPECTED_START_ITERATION
            or not isinstance(payload.get("model_state_dict"), Mapping)
            or not isinstance(payload.get("model_config"), Mapping)):
        raise ValueError("start is not the frozen Standard24k Sim endpoint")
    metrics = payload.get("metrics")
    training = payload.get("training_config")
    if (not isinstance(metrics, Mapping)
            or metrics.get("status") != EXPECTED_START_STATUS
            or not isinstance(training, Mapping)
            or training.get("method") != EXPECTED_START_METHOD):
        raise ValueError("start status or method changed")
    raw = dict(payload["model_config"])
    if raw.pop("architecture", None) != "ensemble_unet":
        raise ValueError("start architecture changed")
    config = EnsembleUNet2dConfig(**raw)
    if (config.dual_decoder_mode != "none"
            or config.base != 210 or config.depth != 4
            or config.member_adapter_channels != 0
            or config.cross_conditioned_backbone
            or not config.mean_fluctuation
            or config.mean_profile_channels != 104
            or not config.physical_energy_controller
            or not config.physical_energy_member_sync):
        raise ValueError("start is not the ordinary B210 architecture")
    state = payload["model_state_dict"]
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in state.values() if isinstance(value, torch.Tensor)):
        raise TypeError("start floating state must be FP32")
    report = {
        "phase": "sim",
        "iteration": EXPECTED_START_ITERATION,
        "status": EXPECTED_START_STATUS,
        "method": EXPECTED_START_METHOD,
        "checkpoint_model_config_sha256": dr1.canonical_sha256(
            payload["model_config"]),
        "checkpoint_model_state_sha256": dr1.model_state_sha256(state),
    }
    return config, state, report


def arm_config(base: EnsembleUNet2dConfig, arm: str) -> EnsembleUNet2dConfig:
    if arm not in ARM_MODE:
        raise ValueError(f"unknown RDE-MoE arm {arm!r}")
    return replace(base, dual_decoder_mode=ARM_MODE[arm])


def expected_parameters(arm: str) -> int:
    if arm not in ARM_MODE:
        raise ValueError(arm)
    return (EXPECTED_BASE_PARAMETERS if arm == "r0_single"
            else EXPECTED_EXPERT_PARAMETERS)


def new_state_keys(model: EnsembleUNet2d,
                   base_state: Mapping[str, torch.Tensor]) -> set[str]:
    return set(model.state_dict()) - set(base_state)


def audited_scope_parameters(model: EnsembleUNet2d) -> int:
    prefixes = (
        "up.", "decoder.", "decoder_regime_film.",
        "mean_profile_refiner.",
    )
    return sum(parameter.numel() for name, parameter in model.named_parameters()
               if name.startswith(prefixes))


def expert_copy_is_exact(model: EnsembleUNet2d) -> bool:
    pairs = (
        (model.up, model.fluctuation_up),
        (model.decoder, model.fluctuation_decoder),
        (model.decoder_regime_film,
         model.fluctuation_decoder_regime_film),
        (model.mean_profile_refiner,
         model.regime_expert_mean_profile_refiner),
    )
    for first, second in pairs:
        if (first is None) != (second is None):
            return False
        if first is None:
            continue
        left, right = first.state_dict(), second.state_dict()
        if left.keys() != right.keys() or any(
                not torch.equal(left[name], right[name]) for name in left):
            return False
    return True


def router_state_sha256(model: EnsembleUNet2d) -> str:
    if model.regime_expert_router is None:
        raise RuntimeError("single decoder has no router")
    state = OrderedDict(
        (f"router.{name}", value.detach().cpu())
        for name, value in model.regime_expert_router.state_dict().items())
    state["descriptor_mean"] = (
        model.regime_expert_router_mean.detach().cpu())
    state["descriptor_std"] = (
        model.regime_expert_router_std.detach().cpu())
    return dr1.model_state_sha256(state)


def packaged_router_state(
        package: Mapping[str, object],
        ) -> OrderedDict[str, torch.Tensor]:
    """Return the strict-loadable router parameters and descriptor buffers.

    The prefix router itself is frozen during Real4k.  Its parameters are
    nevertheless present in :class:`FP32EMAShadow`, whose two FP32 arithmetic
    operations can move a mathematically constant value by a few ulps.  Keep
    parameters and buffers explicit so that post-training integrity checks do
    not confuse that deterministic numerical effect with optimizer training.
    """
    if package.get("schema") != ROUTER_SCHEMA:
        raise ValueError("router package schema changed")
    raw = package.get("router_state_dict")
    if not isinstance(raw, Mapping):
        raise ValueError("router package state is missing")
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in raw.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"router parameter is not floating point: {name}")
        state[f"regime_expert_router.{name}"] = (
            value.detach().cpu().contiguous().clone())
    for name, key in (
            ("regime_expert_router_mean", "descriptor_mean"),
            ("regime_expert_router_std", "descriptor_std")):
        value = package.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (80,):
            raise ValueError(f"router package {key} changed")
        state[name] = value.detach().cpu().contiguous().clone()
    return state


def endpoint_router_state(
        state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    """Extract a small CPU router state from a model or checkpoint state."""
    names = (
        "regime_expert_router.0.weight",
        "regime_expert_router.0.bias",
        "regime_expert_router.2.weight",
        "regime_expert_router.2.bias",
        "regime_expert_router_mean",
        "regime_expert_router_std",
    )
    if any(name not in state for name in names):
        raise ValueError("expert endpoint router state is incomplete")
    return OrderedDict(
        (name, state[name].detach().cpu().contiguous().clone())
        for name in names)


def deterministic_frozen_router_ema_state(
        package: Mapping[str, object], *, updates: int = UPDATES,
        decay: float = EMA_DECAY,
        ) -> OrderedDict[str, torch.Tensor]:
    """Replay the exact constant-source FP32 EMA arithmetic.

    This intentionally executes the same in-place ``mul_`` then ``add_`` as
    ``FP32EMAShadow.update_after_optimizer_step``.  Descriptor statistics are
    persistent buffers rather than named parameters, so EMA materialization
    must copy them bitwise without applying the recurrence.
    """
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("EMA updates must be a non-negative integer")
    if not math.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("EMA decay must lie in (0, 1)")
    source = packaged_router_state(package)
    expected: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in source.items():
        if name in {"regime_expert_router_mean", "regime_expert_router_std"}:
            expected[name] = value.clone()
            continue
        shadow = value.to(dtype=torch.float32).clone()
        current = value.to(dtype=torch.float32)
        for _ in range(updates):
            shadow.mul_(decay).add_(current, alpha=1.0 - decay)
        expected[name] = shadow
    return expected


def _assert_tensor_states_bitwise_equal(
        left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor],
        label: str) -> None:
    if tuple(left) != tuple(right):
        raise RuntimeError(f"{label}: tensor names differ")
    changed = [name for name in left
               if not torch.equal(left[name], right[name])]
    if changed:
        raise RuntimeError(f"{label}: bitwise mismatch in {changed}")


def audit_frozen_router_endpoints(
        package: Mapping[str, object],
        endpoint_states: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
        ) -> dict[str, object]:
    """Fail closed on raw router mutation and explain exact EMA roundoff.

    Required input contains R1/R2 and raw/EMA.  There is deliberately no
    tolerance: raw states equal the package bitwise, R1/R2 equal one another
    bitwise for each endpoint kind, and EMA states equal a bitwise replay of
    the constant-source FP32 recurrence.  Consequently a small but unrelated
    router update cannot pass merely because it lies below an ``allclose``
    threshold.
    """
    arms = ("r1_fixed_average", "r2_history_router")
    if set(endpoint_states) != set(arms) or any(
            set(endpoint_states[arm]) != set(ENDPOINT_ORDER) for arm in arms):
        raise ValueError("router audit requires exactly R1/R2 raw/EMA states")
    package_state = packaged_router_state(package)
    expected_ema = deterministic_frozen_router_ema_state(package)
    extracted = {
        arm: {kind: endpoint_router_state(endpoint_states[arm][kind])
              for kind in ENDPOINT_ORDER}
        for arm in arms
    }
    for arm in arms:
        _assert_tensor_states_bitwise_equal(
            extracted[arm]["raw"], package_state,
            f"{arm} raw router versus package")
        _assert_tensor_states_bitwise_equal(
            extracted[arm]["ema"], expected_ema,
            f"{arm} EMA router versus deterministic replay")
    for kind in ENDPOINT_ORDER:
        _assert_tensor_states_bitwise_equal(
            extracted[arms[0]][kind], extracted[arms[1]][kind],
            f"R1/R2 {kind} frozen router")

    parameter_names = tuple(
        name for name in package_state
        if name not in {"regime_expert_router_mean",
                        "regime_expert_router_std"})
    drift = {}
    changed_total = 0
    maximum = 0.0
    for name in parameter_names:
        difference = expected_ema[name] - package_state[name]
        changed = int(torch.count_nonzero(difference))
        max_abs = float(difference.abs().max())
        changed_total += changed
        maximum = max(maximum, max_abs)
        drift[name] = {
            "elements": int(difference.numel()),
            "changed_elements": changed,
            "max_abs": max_abs,
            "mean_abs": float(difference.abs().mean()),
        }
    return {
        "raw_equals_router_package_bitwise": True,
        "r1_r2_raw_equal_bitwise": True,
        "r1_r2_ema_equal_bitwise": True,
        "ema_equals_constant_source_fp32_replay_bitwise": True,
        "descriptor_buffers_equal_package_bitwise_raw_and_ema": True,
        "ema_replay": {
            "updates": UPDATES,
            "decay": EMA_DECAY,
            "operation": "shadow.mul_(decay).add_(source, alpha=1-decay)",
            "parameters_only": True,
            "changed_elements": changed_total,
            "max_abs": maximum,
            "by_parameter": drift,
        },
    }


def parameter_manifest(model: EnsembleUNet2d) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]


def install_router(model: EnsembleUNet2d,
                   package: Mapping[str, object]) -> None:
    if package.get("schema") != ROUTER_SCHEMA:
        raise ValueError("router package schema changed")
    state = package.get("router_state_dict")
    if not isinstance(state, Mapping) or model.regime_expert_router is None:
        raise ValueError("router package state is missing")
    model.regime_expert_router.load_state_dict(state, strict=True)
    model.set_regime_expert_router_standardization(
        package["descriptor_mean"], package["descriptor_std"])
    for parameter in model.regime_expert_router.parameters():
        parameter.requires_grad_(False)
    if router_state_sha256(model) != package["router_state_sha256"]:
        raise RuntimeError("installed router state digest changed")


def load_start(
        path: Path, arm: str, normalizer, device: torch.device,
        router_package: Mapping[str, object] | None,
        ) -> tuple[EnsembleUNet2d, EnsembleUNet2dConfig, dict[str, object]]:
    actual = dr1.sha256(path)
    if actual != EXPECTED_START_SHA256:
        raise RuntimeError("Standard24k start checkpoint SHA changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    base, state, report = parse_start_payload(payload)
    small4k.npt1.ensemble_train.seed_everything(small4k.SEED)
    config = arm_config(base, arm)
    model = EnsembleUNet2d(config).float()
    if arm == "r0_single":
        model.load_state_dict(state, strict=True)
        missing = set()
    else:
        incompatible = model.load_state_dict(state, strict=False)
        missing = new_state_keys(model, state)
        if (set(incompatible.missing_keys) != missing
                or incompatible.unexpected_keys):
            raise RuntimeError(
                "single->expert upgrade changed common keys: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}")
        allowed = (
            "fluctuation_up.", "fluctuation_decoder.",
            "fluctuation_decoder_regime_film.",
            "fluctuation_cross_conditioner.",
            "regime_expert_router.", "regime_expert_router_mean",
            "regime_expert_router_std",
            "regime_expert_mean_profile_refiner.",
        )
        if not missing or any(not name.startswith(allowed) for name in missing):
            raise RuntimeError("expert upgrade has an unauthorized new state")
        model.copy_primary_into_regime_expert_()
        if not expert_copy_is_exact(model):
            raise RuntimeError("expert is not a bitwise decoder/refiner copy")
        if router_package is None:
            raise ValueError("expert arm requires the frozen router package")
        install_router(model, router_package)
    total = sum(parameter.numel() for parameter in model.parameters())
    controller = sum(
        parameter.numel() for parameter in model.energy_controller.parameters())
    if (total != expected_parameters(arm)
            or controller != EXPECTED_CONTROLLER_PARAMETERS
            or audited_scope_parameters(model) !=
            EXPECTED_AUDITED_SCOPE_PARAMETERS):
        raise RuntimeError(
            f"RDE-MoE parameter contract changed for {arm}: "
            f"total={total}, scope={audited_scope_parameters(model)}, "
            f"controller={controller}")
    small4k.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    report.update({
        "path": str(path.resolve()),
        "checkpoint_sha256": actual,
        "arm": arm,
        "dual_decoder_mode": ARM_MODE[arm],
        "strict_common_state_load": True,
        "new_state_keys": sorted(missing),
        "expert_copy_exact": (None if arm == "r0_single" else True),
        "parameters": total,
        "audited_scope_parameters": audited_scope_parameters(model),
        "controller_parameters": controller,
        "router_state_sha256": (
            None if arm == "r0_single" else router_state_sha256(model)),
        "router_frozen": (
            None if arm == "r0_single" else not any(
                parameter.requires_grad
                for parameter in model.regime_expert_router.parameters())),
        "parameter_manifest_sha256": dr1.canonical_sha256(
            parameter_manifest(model)),
        "upgraded_model_state_sha256": dr1.model_state_sha256(
            model.state_dict()),
    })
    del payload, state
    gc.collect()
    return model.to(device), config, report


def router_label(aoa: int) -> int:
    if aoa in ROUTER_POSITIVE_AOAS:
        return 1
    if aoa in ROUTER_NEGATIVE_AOAS:
        return 0
    raise ValueError(f"unexpected AoA {aoa}")


def real_history_uvp(
        trajectories: Sequence[torch.Tensor], windows: Sequence,
        device: torch.device | str = "cpu") -> torch.Tensor:
    """Build the exact three-channel history seen by the deployed model.

    Released Real PIV files contain only ``u/v``.  ``WindowDataset`` appends a
    zero pressure channel before normalization, so router fitting and deployed
    routing must do the same rather than accidentally normalizing a two-channel
    tensor with three-channel statistics.
    """
    history_uv = torch.stack([
        trajectories[int(window.trajectory)][
            int(window.start):int(window.start) + 20]
        for window in windows
    ]).to(device=device, dtype=torch.float32)
    if history_uv.ndim != 5 or history_uv.shape[-1] != 2:
        raise ValueError("Real router histories must be Bx20xHxWx2")
    pressure = torch.zeros_like(history_uv[..., :1])
    return torch.cat((history_uv, pressure), dim=-1)


def normalized_router_descriptor(
        trajectories: Sequence[torch.Tensor], windows: Sequence,
        normalizer) -> torch.Tensor:
    history = real_history_uvp(trajectories, windows)
    mean = normalizer.mean_input.detach().cpu()
    std = normalizer.std_input.detach().cpu()
    if (mean.shape != (3,) or std.shape != (3,)
            or float(mean[2]) != 0.0 or float(std[2]) != 1.0):
        raise RuntimeError("Real zero-pressure normalizer contract changed")
    return history_regime_summary((history - mean) / std)


def prefix_router_tensors(
        trajectories: Sequence[torch.Tensor], windows: Sequence,
        metadata: Sequence[Mapping[str, object]], normalizer,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    descriptors = []
    labels = []
    aoas = []
    for offset in range(0, len(windows), 128):
        selected = windows[offset:offset + 128]
        descriptors.append(normalized_router_descriptor(
            trajectories, selected, normalizer))
        current_aoas = [
            int(metadata[int(window.trajectory)]["aoa"])
            for window in selected]
        labels.extend(router_label(value) for value in current_aoas)
        aoas.extend(current_aoas)
    descriptor = torch.cat(descriptors).float()
    label = torch.tensor(labels, dtype=torch.float32)
    aoa = torch.tensor(aoas, dtype=torch.long)
    if (descriptor.shape != (EXPECTED_DENSE_WINDOWS, 80)
            or label.shape != (EXPECTED_DENSE_WINDOWS,)
            or int(label.sum()) != EXPECTED_ROUTER_POSITIVES
            or int((1.0 - label).sum()) != EXPECTED_ROUTER_NEGATIVES):
        raise RuntimeError(
            "prefix router dense-window label contract changed")
    return descriptor, label, aoa


def router_metrics(labels: torch.Tensor, probabilities: torch.Tensor,
                   aoas: torch.Tensor) -> dict[str, object]:
    predicted = probabilities >= 0.5
    positive = labels.bool()
    negative = ~positive
    tpr = float((predicted & positive).sum() / positive.sum().clamp_min(1))
    tnr = float(((~predicted) & negative).sum()
                / negative.sum().clamp_min(1))
    entropy = -(probabilities.clamp(1.0e-7, 1.0 - 1.0e-7)
                * probabilities.clamp(1.0e-7, 1.0 - 1.0e-7).log()
                + (1.0 - probabilities).clamp(1.0e-7, 1.0 - 1.0e-7)
                * (1.0 - probabilities).clamp(
                    1.0e-7, 1.0 - 1.0e-7).log())
    per_aoa = {}
    for aoa in dr1.AOA_VALUES:
        selected = aoas == aoa
        per_aoa[str(aoa)] = {
            "windows": int(selected.sum()),
            "label": router_label(aoa),
            "mean_probability_expert1": float(probabilities[selected].mean()),
            "mean_entropy": float(entropy[selected].mean()),
            "accuracy": float(
                (predicted[selected] == positive[selected]).float().mean()),
        }
    return {
        "windows": int(labels.numel()),
        "accuracy": float((predicted == positive).float().mean()),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "mean_probability_expert1": float(probabilities.mean()),
        "mean_entropy": float(entropy.mean()),
        "per_aoa": per_aoa,
    }


def fit_prefix_router(
        trajectories: Sequence[torch.Tensor], windows: Sequence,
        metadata: Sequence[Mapping[str, object]], normalizer,
        output_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    descriptor, labels, aoas = prefix_router_tensors(
        trajectories, windows, metadata, normalizer)
    descriptor_mean = descriptor.mean(dim=0)
    descriptor_std = descriptor.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    standardized = (descriptor - descriptor_mean) / descriptor_std
    torch.manual_seed(ROUTER_SEED)
    router = nn.Sequential(
        nn.Linear(80, 16), nn.Tanh(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=ROUTER_LR,
        weight_decay=ROUTER_WEIGHT_DECAY)
    positive_weight = torch.tensor(ROUTER_POS_WEIGHT)
    history = []
    router.train()
    for update in range(1, ROUTER_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = router(standardized).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=positive_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("prefix router loss is non-finite")
        loss.backward()
        optimizer.step()
        if update in (1, 100, 200, 300, 400):
            history.append({"update": update, "loss": float(loss.detach())})
    router.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(
            router(standardized).squeeze(-1))
    state = OrderedDict(
        (name, value.detach().cpu())
        for name, value in router.state_dict().items())
    digest_state = OrderedDict(
        [(f"router.{name}", value) for name, value in state.items()]
        + [("descriptor_mean", descriptor_mean.detach().cpu()),
           ("descriptor_std", descriptor_std.detach().cpu())])
    state_sha = dr1.model_state_sha256(digest_state)
    package: dict[str, object] = {
        "schema": ROUTER_SCHEMA,
        "router_state_dict": state,
        "descriptor_mean": descriptor_mean.detach().cpu(),
        "descriptor_std": descriptor_std.detach().cpu(),
        "router_state_sha256": state_sha,
        "label_one_aoas": list(ROUTER_POSITIVE_AOAS),
        "label_zero_aoas": list(ROUTER_NEGATIVE_AOAS),
        "counts": {"positive": EXPECTED_ROUTER_POSITIVES,
                   "negative": EXPECTED_ROUTER_NEGATIVES},
    }
    temporary = output_path.with_name(output_path.name + ".tmp")
    if output_path.exists() or temporary.exists():
        raise FileExistsError(output_path)
    torch.save(package, temporary)
    temporary.replace(output_path)
    report = {
        "path": str(output_path.resolve()),
        "file_sha256": dr1.sha256(output_path),
        "router_state_sha256": state_sha,
        "architecture": "80->16 tanh->1",
        "descriptor": "history_regime_summary(normalized_history.detach())",
        "real_input_compatibility": (
            "append zero pressure exactly as WindowDataset, apply the same "
            "three-channel Real normalizer, then summarize normalized u/v"),
        "descriptor_standardization": "dense-prefix population mean/std",
        "windows": len(windows),
        "target_tensors_used": False,
        "tail_or_clean716_used": False,
        "label_one_aoas_expert1": list(ROUTER_POSITIVE_AOAS),
        "label_zero_aoas_expert0": list(ROUTER_NEGATIVE_AOAS),
        "positive_weight": ROUTER_POS_WEIGHT,
        "seed": ROUTER_SEED,
        "optimizer": "AdamW",
        "lr": ROUTER_LR,
        "weight_decay": ROUTER_WEIGHT_DECAY,
        "full_batch_updates": ROUTER_UPDATES,
        "scheduler": None,
        "augmentation": None,
        "early_stop_or_best_selection": False,
        "history": history,
        "fit_metrics_diagnostic_only": router_metrics(
            labels, probabilities, aoas),
        "independent_separability_audit_sha256": ROUTER_AUDIT_SHA256,
    }
    return package, report


def update0_signature(model: EnsembleUNet2d, dataset, normalizer,
                      device: torch.device) -> str:
    samples = [dataset[index] for index in range(min(2, len(dataset)))]
    history = torch.stack([sample[0] for sample in samples]).to(device)
    target = torch.stack([sample[1] for sample in samples]).to(device)
    history_norm, _target_norm = normalizer.preprocess(history, target)
    model.eval()
    with torch.inference_mode():
        values = model.forward_uq_training(history_norm)[:5]
    names = ("prediction", "members", "quantiles", "lower", "upper")
    return dr1.model_state_sha256(OrderedDict(
        (name, value.detach().float().cpu())
        for name, value in zip(names, values)))


@contextmanager
def endpoint_contract(arm: str) -> Iterator[None]:
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": f"ema_u4000_{arm}",
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": SCHEDULER_T_MAX,
        "EMA_DECAY": EMA_DECAY,
        "HELD_AOA": -1,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(small4k, name, value)


@contextmanager
def frozen_router_trainability(arm: str) -> Iterator[None]:
    original = small4k.ft12.configure_trainable

    def configure(model: EnsembleUNet2d,
                  expected: int | None = None) -> tuple[nn.Parameter, ...]:
        all_parameters = original(model, expected)
        if arm == "r0_single":
            return all_parameters
        if model.regime_expert_router is None:
            raise RuntimeError("expert arm lost its router")
        for parameter in model.regime_expert_router.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        trainable = tuple(
            parameter for parameter in all_parameters
            if parameter.requires_grad)
        if (sum(parameter.numel() for parameter in trainable)
                != expected_parameters(arm) - EXPECTED_ROUTER_PARAMETERS):
            raise RuntimeError("frozen-router trainable count changed")
        return trainable

    small4k.ft12.configure_trainable = configure
    try:
        yield
    finally:
        small4k.ft12.configure_trainable = original


class EndpointRegistry:
    def __init__(self, arm_order: Sequence[str] = ARM_ORDER) -> None:
        if not arm_order or len(set(arm_order)) != len(arm_order):
            raise ValueError("endpoint registry arms must be unique/non-empty")
        self.arm_order = tuple(arm_order)
        self.values: dict[str, dict[str, dict[str, object]]] = {
            arm: {} for arm in self.arm_order}

    def seal(self, arm: str, kind: str, value: Mapping[str, object]) -> None:
        if arm not in self.values or kind not in ENDPOINT_ORDER:
            raise ValueError("unknown endpoint seal")
        if kind in self.values[arm]:
            raise RuntimeError(f"{arm}/{kind} sealed twice")
        self.values[arm][kind] = dict(value)

    def complete(self) -> bool:
        return all(set(row) == set(ENDPOINT_ORDER)
                   for row in self.values.values())


class EmaSealAdapter:
    def __init__(self, registry: EndpointRegistry, arm: str) -> None:
        self.registry = registry
        self.arm = arm
        self.value: dict[str, object] | None = None

    def seal(self, value: Mapping[str, object]) -> None:
        if self.value is not None:
            raise RuntimeError("EMA endpoint sealed twice")
        self.value = dict(value)
        self.registry.seal(self.arm, "ema", value)


def save_raw_endpoint(
        path: Path, model: EnsembleUNet2d, arm: str,
        start_report: Mapping[str, object], normalizer,
        stream_sha256: str) -> dict[str, object]:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    state = OrderedDict(
        (name, value.detach().cpu())
        for name, value in model.state_dict().items())
    state_sha = dr1.model_state_sha256(state)
    normalizer_report = ft12.normalizer_payload(normalizer)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": UPDATES,
        "endpoint": f"raw_u4000_{arm}",
        "primary": False,
        "held_aoa": -1,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start": dict(start_report),
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": normalizer_report["sha256"],
        "sample_stream_sha256": stream_sha256,
        "ema": None,
        "optimizer_state_saved": False,
        "clean716_opened_before_seal": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "endpoint": payload["endpoint"],
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": dr1.canonical_sha256(
            payload["model_config"]),
        "sample_stream_sha256": stream_sha256,
        "iteration": UPDATES,
        "primary": False,
        "optimizer_state_saved": False,
    }


def train_arm(
        arm: str, args: argparse.Namespace, dataset,
        schedule: Mapping[str, object], normalizer,
        router_package: Mapping[str, object], registry: EndpointRegistry,
        expected_update0: str | None, expected_expert_initial: str | None,
        deadline: float,
        ) -> tuple[dict[str, object], str, str | None]:
    device = torch.device(args.device)
    model, config, start_report = load_start(
        args.start_sim, arm, normalizer, device,
        None if arm == "r0_single" else router_package)
    update0 = update0_signature(model, dataset, normalizer, device)
    if expected_update0 is not None and update0 != expected_update0:
        raise RuntimeError(
            f"{arm} update-zero eval function differs from R0")
    initial_sha = str(start_report["upgraded_model_state_sha256"])
    if arm != "r0_single" and expected_expert_initial is not None:
        if initial_sha != expected_expert_initial:
            raise RuntimeError("R1/R2 upgraded initial states differ")
    initial_router_sha = (
        None if arm == "r0_single" else router_state_sha256(model))
    arm_dir = args.output_dir / arm
    arm_dir.mkdir()
    seal_adapter = EmaSealAdapter(registry, arm)
    # Reset immediately before optimization.  R1/R2 execute the same two
    # decoder/refiner/router graph, so their dropout and input-augmentation RNG
    # call structure remains paired even after their values diverge.
    small4k.npt1.ensemble_train.seed_everything(small4k.SEED)
    with endpoint_contract(arm), frozen_router_trainability(arm):
        training, ema_seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, seal_adapter,
            input_augmentation=dense4k.AUGMENTATION,
            endpoint_filename=f"ema_u4000_{arm}.pth",
            expected_parameters=expected_parameters(arm),
            scheduler_t_max=SCHEDULER_T_MAX)
    if seal_adapter.value != ema_seal:
        raise RuntimeError("EMA endpoint was not bound to the registry")
    raw_seal = save_raw_endpoint(
        arm_dir / f"raw_u4000_{arm}.pth", model, arm,
        start_report, normalizer, schedule["sha256"])
    registry.seal(arm, "raw", raw_seal)
    final_router_sha = (
        None if arm == "r0_single" else router_state_sha256(model))
    router_grad_absent = (
        None if arm == "r0_single" else all(
            parameter.grad is None
            for parameter in model.regime_expert_router.parameters()))
    if (arm != "r0_single"
            and (initial_router_sha != final_router_sha
                 or not router_grad_absent)):
        raise RuntimeError("frozen prefix router changed during Real4k")
    report = {
        "mode": ARM_MODE[arm],
        "model_config": {"architecture": "ensemble_unet", **config.__dict__},
        "parameters": expected_parameters(arm),
        "parameter_manifest_sha256": start_report[
            "parameter_manifest_sha256"],
        "initial_state_sha256": initial_sha,
        "update0_eval_output_sha256": update0,
        "update0_identity_scope": (
            "eval/inference point,members,quantiles,lower,upper; training "
            "dropout intentionally gives the two experts independent masks"),
        "router_state_sha256_before_real": initial_router_sha,
        "router_state_sha256_after_real_raw": final_router_sha,
        "router_requires_grad_during_real": False if arm != "r0_single" else None,
        "router_gradients_all_none_after_real": router_grad_absent,
        "sample_stream_sha256": schedule["sha256"],
        "augmentation": dict(dense4k.AUGMENTATION),
        "training": training,
        "seals": {"raw": raw_seal, "ema": ema_seal},
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report, update0, initial_sha if arm != "r0_single" else None


class TailAudit:
    def __init__(self, registry: EndpointRegistry) -> None:
        if not registry.complete():
            raise RuntimeError(
                f"all {2 * len(registry.arm_order)} raw/EMA endpoints "
                "must be sealed")
        self.registry = registry
        self.opened: list[str] = []
        self.endpoint_snapshots: dict[str, tuple[int, int, int, int]] = {}
        # Hash each large endpoint once before any clean716 file is opened.
        # Per-row checks below use immutable stat snapshots, avoiding 486
        # repeated multi-GB hashes while preserving the access-order proof.
        for endpoints in self.registry.values.values():
            for seal in endpoints.values():
                path = Path(str(seal["path"]))
                if (not path.is_file()
                        or dr1.sha256(path) != seal["file_sha256"]):
                    raise RuntimeError(
                        "endpoint changed before clean716 audit started")
                self.endpoint_snapshots[str(path.resolve())] = (
                    path.stat().st_size, path.stat().st_mtime_ns,
                    path.stat().st_ctime_ns, path.stat().st_ino)

    @staticmethod
    def _stat_snapshot(path: Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        if phase != "held" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid clean716 open")
        for raw_path, snapshot in self.endpoint_snapshots.items():
            path = Path(raw_path)
            if (not path.is_file()
                    or self._stat_snapshot(path) != snapshot):
                raise RuntimeError("endpoint changed before clean716 open")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        if phase != "held" or row["file"] in self.opened:
            raise RuntimeError("invalid or duplicate clean716 record")
        self.opened.append(str(row["file"]))

    def after_load(self) -> None:
        """Re-hash all seals once after clean716 loading completes."""
        for endpoints in self.registry.values.values():
            for seal in endpoints.values():
                path = Path(str(seal["path"]))
                snapshot = self.endpoint_snapshots[str(path.resolve())]
                if (self._stat_snapshot(path) != snapshot
                        or dr1.sha256(path) != seal["file_sha256"]):
                    raise RuntimeError("endpoint changed during clean716 load")


def load_endpoint(
        seal: Mapping[str, object], arm: str, normalizer,
        device: torch.device, base_config: EnsembleUNet2dConfig,
        ) -> EnsembleUNet2d:
    path = Path(str(seal["path"]))
    if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("endpoint file changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment") != EXPERIMENT
            or payload.get("phase") != "real"
            or int(payload.get("iteration", -1)) != UPDATES
            or int(payload.get("held_aoa", 0)) != -1
            or payload.get("endpoint") != seal["endpoint"]
            or payload.get("primary") is not seal["primary"]
            or payload.get("sample_stream_sha256")
            != seal["sample_stream_sha256"]
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]
            or dr1.canonical_sha256(payload.get("model_config"))
            != seal["model_config_sha256"]):
        raise RuntimeError("endpoint payload changed")
    raw = dict(payload["model_config"])
    if raw.pop("architecture", None) != "ensemble_unet":
        raise ValueError("endpoint architecture changed")
    config = EnsembleUNet2dConfig(**raw)
    if config != arm_config(base_config, arm):
        raise RuntimeError("endpoint arm config changed")
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != (
            expected_parameters(arm)):
        raise RuntimeError("endpoint parameter count changed")
    small4k.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    if arm != "r0_single":
        for parameter in model.regime_expert_router.parameters():
            parameter.requires_grad_(False)
    del payload, state
    gc.collect()
    return model.to(device)


def expert_distance(model: EnsembleUNet2d) -> dict[str, float] | None:
    if model.config.dual_decoder_mode not in REGIME_EXPERT_DECODER_MODES:
        return None
    pairs = (
        ("up", model.up, model.fluctuation_up),
        ("decoder", model.decoder, model.fluctuation_decoder),
        ("decoder_regime_film", model.decoder_regime_film,
         model.fluctuation_decoder_regime_film),
        ("mean_profile_refiner", model.mean_profile_refiner,
         model.regime_expert_mean_profile_refiner),
    )
    report = {}
    total_square = 0.0
    reference_square = 0.0
    for label, first, second in pairs:
        local = 0.0
        local_reference = 0.0
        for left, right in zip(first.parameters(), second.parameters()):
            local += float((left.detach().float()
                            - right.detach().float()).square().sum())
            local_reference += float(left.detach().float().square().sum())
        report[f"{label}_l2"] = math.sqrt(local)
        report[f"{label}_relative_l2"] = math.sqrt(
            local / max(local_reference, torch.finfo(torch.float32).tiny))
        total_square += local
        reference_square += local_reference
    report["total_l2"] = math.sqrt(total_square)
    report["total_relative_l2"] = math.sqrt(
        total_square / max(reference_square, torch.finfo(torch.float32).tiny))
    return report


def held_router_report(
        model: EnsembleUNet2d, trajectories: Sequence[torch.Tensor],
        windows: Sequence, metadata: Sequence[Mapping[str, object]],
        normalizer) -> dict[str, object] | None:
    if model.regime_expert_router is None:
        return None
    descriptor, labels, aoas = prefix_router_tensors_general(
        trajectories, windows, metadata, normalizer)
    device = next(model.parameters()).device
    with torch.inference_mode():
        # Call the deployed module path, including its stored standardization.
        logits = []
        for offset in range(0, len(windows), 128):
            selected = windows[offset:offset + 128]
            history = real_history_uvp(
                trajectories, selected, device=device)
            normalized = (
                history - normalizer.mean_input) / normalizer.std_input
            logits.append(model.regime_expert_router_logits(
                normalized).squeeze(-1).cpu())
        probability = torch.sigmoid(torch.cat(logits))
    report = router_metrics(labels, probability, aoas)
    report["descriptor_recomputed_shape"] = list(descriptor.shape)
    report["effective_blend"] = (
        "fixed_0.5" if model.config.dual_decoder_mode
        == "regime_expert_fixed_average" else "soft_probability_expert1")
    report["metadata_used_by_inference"] = False
    return report


def prefix_router_tensors_general(
        trajectories: Sequence[torch.Tensor], windows: Sequence,
        metadata: Sequence[Mapping[str, object]], normalizer,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    descriptors = []
    labels = []
    aoas = []
    for offset in range(0, len(windows), 128):
        selected = windows[offset:offset + 128]
        descriptors.append(normalized_router_descriptor(
            trajectories, selected, normalizer))
        values = [int(metadata[int(window.trajectory)]["aoa"])
                  for window in selected]
        labels.extend(router_label(value) for value in values)
        aoas.extend(values)
    return (torch.cat(descriptors), torch.tensor(labels, dtype=torch.float32),
            torch.tensor(aoas, dtype=torch.long))


def metric_comparison(candidate: Mapping[str, float],
                      reference: Mapping[str, float]) -> dict[str, object]:
    keys = (
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "native_sps_score", "native_sps_coverage",
        "native_sps_mean_width")
    delta = {key: float(candidate[key]) - float(reference[key])
             for key in keys}
    point = sum(POINT_WEIGHTS[key] * delta[key] for key in POINT_WEIGHTS)
    return {
        "reference": dict(reference),
        "candidate": dict(candidate),
        "delta": delta,
        "official_point_contribution_delta": point,
        "official_non_time_contribution_delta": (
            point + 0.25 * delta["native_sps_score"]),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if (not args.execute
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise PermissionError("formal run requires protocol acknowledgement")
    required = (args.output_dir, args.real_dir, args.clean_manifest,
                args.evaluation_manifest, args.start_sim)
    if any(value is None for value in required):
        raise ValueError("formal run is missing a required path")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if dr1.sha256(args.clean_manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("clean81 manifest SHA changed")
    git = git_report()
    if not git["clean"]:
        raise RuntimeError(f"Git worktree must be clean: {git['porcelain']}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal RDE-MoE experiment requires CUDA")

    rows, manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    # Deliberately do not read or hash the fixed clean716 evaluation manifest
    # yet.  Its filenames and target-window indices become visible only after
    # every raw/EMA endpoint is immutable.
    if not args.evaluation_manifest.is_file():
        raise FileNotFoundError(args.evaluation_manifest)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = (
        math.inf if args.wall_budget_seconds is None
        else started + args.wall_budget_seconds)
    torch.set_float32_matmul_precision("high")

    prefix_audit = dense4k.PrefixAudit()
    trajectories, metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    if len(sparse_windows) != dr1.EXPECTED_WINDOWS:
        raise RuntimeError("sparse prefix windows changed")
    dense_windows = dense4k.dense_windows(
        trajectories, dense4k.CANDIDATE_STRIDE)
    if len(dense_windows) != EXPECTED_DENSE_WINDOWS:
        raise RuntimeError("dense prefix windows changed")
    sufficient = smu1.window_sufficient_statistics(
        trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "point_normalizer.pt")
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, dense_windows)
    schedule = smu1.paired_batch_schedule(
        range(len(dense_windows)), dense_windows,
        updates=UPDATES, seed=small4k.SEED)
    schedule["updates"] = UPDATES

    router_package, router_report = fit_prefix_router(
        trajectories, dense_windows, metadata, normalizer,
        args.output_dir / "router_prefix_only.pt")
    registry = EndpointRegistry()
    arms: dict[str, dict[str, object]] = {}
    common_update0 = None
    common_expert_initial = None
    base_config = None
    for arm in ARM_ORDER:
        report, update0, expert_initial = train_arm(
            arm, args, dataset, schedule, normalizer, router_package,
            registry, common_update0, common_expert_initial, deadline)
        common_update0 = update0
        if expert_initial is not None:
            if common_expert_initial is None:
                common_expert_initial = expert_initial
            elif expert_initial != common_expert_initial:
                raise RuntimeError("expert initial state mismatch")
        arms[arm] = report
        raw = dict(report["model_config"])
        raw.pop("architecture")
        current_base = replace(
            EnsembleUNet2dConfig(**raw), dual_decoder_mode="none")
        if base_config is None:
            base_config = current_base
        elif current_base != base_config:
            raise RuntimeError("arms do not share one base architecture")
    if not registry.complete():
        raise RuntimeError("endpoint registry is incomplete")
    if (arms["r1_fixed_average"]["router_state_sha256_after_real_raw"]
            != arms["r2_history_router"][
                "router_state_sha256_after_real_raw"]):
        raise RuntimeError("R1/R2 frozen router states differ")
    r1_config = dict(arms["r1_fixed_average"]["model_config"])
    r2_config = dict(arms["r2_history_router"]["model_config"])
    differing_config = {
        key: (r1_config.get(key), r2_config.get(key))
        for key in set(r1_config) | set(r2_config)
        if r1_config.get(key) != r2_config.get(key)
    }
    if differing_config != {
            "dual_decoder_mode": (
                "regime_expert_fixed_average",
                "regime_expert_history_router")}:
        raise RuntimeError(
            f"R1/R2 differ outside the routing rule: {differing_config}")
    if (arms["r1_fixed_average"]["parameter_manifest_sha256"]
            != arms["r2_history_router"]["parameter_manifest_sha256"]):
        raise RuntimeError("R1/R2 parameter manifests differ")

    del dataset
    gc.collect()
    tail_audit = TailAudit(registry)
    # Pin the fixed clean716 definition only after all six endpoints are
    # immutable.  Hashing here preserves the no-tail-before-seal access order
    # while preventing a different, merely shape-compatible set of 716
    # windows from being substituted at evaluation time.
    evaluation_manifest_sha256 = dr1.sha256(args.evaluation_manifest)
    if (evaluation_manifest_sha256
            != EXPECTED_EVALUATION_MANIFEST_SHA256):
        raise RuntimeError("fixed clean716 evaluation manifest SHA changed")
    evaluation_rows = dense4k.load_evaluation_rows(
        args.evaluation_manifest, rows)
    tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", tail_audit)
    tail_audit.after_load()
    tail_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)
    indices_by_aoa = dense4k.group_window_indices_by_aoa(
        tail_windows, tail_metadata)
    target_audit = None
    endpoint_router_states: dict[
        str, dict[str, OrderedDict[str, torch.Tensor]]] = {
            arm: {} for arm in ("r1_fixed_average", "r2_history_router")}
    for arm in ARM_ORDER:
        arms[arm]["endpoints"] = {}
        for kind in ENDPOINT_ORDER:
            seal = registry.values[arm][kind]
            model = load_endpoint(
                seal, arm, normalizer, device, base_config)
            metrics, state, current_target = smu1.evaluate_model(
                model, tail_dataset, list(range(len(tail_windows))),
                normalizer, device, deadline)
            if target_audit is None:
                target_audit = current_target
            elif target_audit != current_target:
                raise RuntimeError("clean716 target audit changed")
            by_aoa = {}
            for aoa, indices in indices_by_aoa.items():
                values, _state, _target = smu1.evaluate_model(
                    model, tail_dataset, indices, normalizer,
                    device, deadline)
                by_aoa[str(aoa)] = values
            arms[arm]["endpoints"][kind] = {
                "metrics": metrics,
                "metric_state": state,
                "metrics_by_aoa": by_aoa,
                "router": held_router_report(
                    model, tail_trajectories, tail_windows,
                    tail_metadata, normalizer),
                "router_state_sha256": (
                    None if arm == "r0_single"
                    else router_state_sha256(model)),
                "expert_distance": expert_distance(model),
                "seal": seal,
            }
            if arm in endpoint_router_states:
                endpoint_router_states[arm][kind] = endpoint_router_state(
                    model.state_dict())
            del model
            gc.collect()
            torch.cuda.empty_cache()

    router_integrity = audit_frozen_router_endpoints(
        router_package, endpoint_router_states)

    comparisons = {}
    for kind in ENDPOINT_ORDER:
        comparisons[kind] = {
            "r1_minus_r0": metric_comparison(
                arms["r1_fixed_average"]["endpoints"][kind]["metrics"],
                arms["r0_single"]["endpoints"][kind]["metrics"]),
            "r2_minus_r0": metric_comparison(
                arms["r2_history_router"]["endpoints"][kind]["metrics"],
                arms["r0_single"]["endpoints"][kind]["metrics"]),
            "r2_minus_r1": metric_comparison(
                arms["r2_history_router"]["endpoints"][kind]["metrics"],
                arms["r1_fixed_average"]["endpoints"][kind]["metrics"]),
        }
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "git": git,
        "start": {
            "checkpoint": str(args.start_sim.resolve()),
            "checkpoint_sha256": EXPECTED_START_SHA256,
            "released_data_only_standard_sim": True,
            "reuse_reason": (
                "avoid repeating identical Standard24k Sim training; any "
                "positive mechanism must later be reproduced fresh"),
        },
        "manifest": manifest,
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": evaluation_manifest_sha256,
            "windows": len(tail_windows),
        },
        "normalizer": {
            "kind": "sparse2544 clean81 prefix only",
            "statistics_sha256": smu1.statistics_digest(statistics),
        },
        "router": router_report,
        "router_endpoint_integrity": router_integrity,
        "cluster_interpretation": (
            "persistent expected-gradient cluster, not a per-sample physical "
            "regime; soft routing is mandatory"),
        "parameters": {
            "r0_single": EXPECTED_BASE_PARAMETERS,
            "r1_r2_each": EXPECTED_EXPERT_PARAMETERS,
            "duplicated_audited_scope": EXPECTED_AUDITED_SCOPE_PARAMETERS,
            "frozen_router": EXPECTED_ROUTER_PARAMETERS,
        },
        "training": {
            "windows": EXPECTED_DENSE_WINDOWS,
            "window_stride": 5,
            "sample_stream_sha256_all_arms": schedule["sha256"],
            "augmentation_all_arms": dict(dense4k.AUGMENTATION),
            "updates": UPDATES,
            "wall_budget_seconds": args.wall_budget_seconds,
            "formal_default_has_no_wall_deadline": (
                args.wall_budget_seconds is None),
            "scheduler_t_max": SCHEDULER_T_MAX,
            "native_loss": (
                "unchanged historical Real point/member/quantile/soft-SPS "
                "plus controller-only TKE auxiliary"),
            "expert_specific_uq_added": False,
            "quantile_gradient_reaches_decoder_features": True,
            "soft_sps_gradient_limited_to_uq_trainable_parameters": True,
            "r1_r2_only_config_difference": differing_config,
            "r1_r2_only_forward_rule_difference": (
                "R1 uses sigmoid(logit)*0+0.5; R2 uses sigmoid(logit) as "
                "the expert1 feature/mean-correction mix weight"),
        },
        "update0_eval_output_sha256_all_arms": common_update0,
        "expert_initial_state_sha256_r1_r2": common_expert_initial,
        "arms": arms,
        "comparisons": comparisons,
        "access_audit": {
            "prefix_opened": prefix_audit.opened,
            "router_targets_used": False,
            "clean716_opened_after_all_six_endpoint_seals": tail_audit.opened,
            "clean716_target_read_before_all_seals": False,
        },
        "decision": "RDE_MOE_REAL4K_COMPLETE_NO_SCORE_GATE",
        "score_gate": None,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    print(json.dumps({
        "event": "b210_rde_moe_real4k_complete",
        "comparisons": comparisons,
        "result": str(args.output_dir / "result.json"),
    }, sort_keys=True), flush=True)
    return result


def smoke(args: argparse.Namespace) -> None:
    if args.start_sim is None:
        raise ValueError("--smoke requires --start-sim")
    if dr1.sha256(args.start_sim) != EXPECTED_START_SHA256:
        raise RuntimeError("smoke start checkpoint SHA changed")
    payload = torch.load(
        str(args.start_sim), map_location="cpu", weights_only=False, mmap=True)
    base, state, source = parse_start_payload(payload)
    reports = {}
    expert_state_sha = None
    for arm in ARM_ORDER:
        small4k.npt1.ensemble_train.seed_everything(small4k.SEED)
        model = EnsembleUNet2d(arm_config(base, arm)).float()
        if arm == "r0_single":
            model.load_state_dict(state, strict=True)
        else:
            incompatible = model.load_state_dict(state, strict=False)
            if (set(incompatible.missing_keys) != new_state_keys(model, state)
                    or incompatible.unexpected_keys):
                raise RuntimeError("smoke partial-load contract failed")
            model.copy_primary_into_regime_expert_()
            if not expert_copy_is_exact(model):
                raise RuntimeError("smoke expert copy differs")
            current = dr1.model_state_sha256(model.state_dict())
            if expert_state_sha is None:
                expert_state_sha = current
            elif current != expert_state_sha:
                raise RuntimeError("smoke R1/R2 initial state differs")
        reports[arm] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "state_keys": len(model.state_dict()),
            "audited_scope_parameters": audited_scope_parameters(model),
            "expert_copy_exact": (
                None if arm == "r0_single" else expert_copy_is_exact(model)),
        }
        if reports[arm]["parameters"] != expected_parameters(arm):
            raise RuntimeError("smoke parameter contract changed")
        del model
        gc.collect()
    print(json.dumps({
        "smoke": "PASS",
        "source": source,
        "arms": reports,
        "r1_r2_initial_state_sha256": expert_state_sha,
    }, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
