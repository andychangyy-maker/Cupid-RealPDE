#!/usr/bin/env python3
"""Paired Real4k test of mergeable backbone NLT adaptation.

Both arms start from the same dense-curriculum Sim endpoint and use identical
stride-5 Real windows, input augmentation, loss, schedule, seed and EMA.  The
control fully fine-tunes the ordinary convolution weights.  The candidate
freezes convolution kernels in the spatial backbone and learns a neuron-linear
factor/offset for every input/output channel pair.  Heads, regime modulation,
native UQ and the physical-energy controller remain fully trainable.

NLT parametrizations are merged into ordinary convolutions when the EMA
endpoint is sealed, so held evaluation uses a strict-loadable legacy model and
the method adds no deployment-time branch.
"""

from __future__ import annotations

from collections import OrderedDict
import gc
from pathlib import Path
from typing import Mapping

import torch
from torch.nn.utils import parametrize

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_scratch_real_dense_aug as paired
from models.ensemble_unet2d import EnsembleUNet2d
from models.neuron_linear_transform import (
    InstalledNeuronLinearTransform,
    NeuronLinearKernelTransform,
    install_neuron_linear_transforms,
    merge_neuron_linear_transforms,
    neuron_linear_parameter_count,
)


EXPERIMENT = "b210_nlt_real_dense_aug"
SCHEMA = "b210_nlt_real_dense_aug_v1"
BACKBONE_PREFIXES = ("stem", "encoder", "bottleneck", "up", "decoder")
ORIGINAL_ARCHITECTURE_CONTRACT = paired.architecture_contract
ORIGINAL_VALIDATE = ft12.validate_start_checkpoint
ORIGINAL_CONFIGURE = ft12.configure_trainable
ORIGINAL_TRAIN = small4k.train_candidate
ORIGINAL_LOAD = small4k.load_primary_endpoint
ORIGINAL_EMA_UPDATE = small4k.ema_lib.FP32EMAShadow.update_after_optimizer_step
_CURRENT_ARM: str | None = None


def select_backbone_convolution(name: str, _module: torch.nn.Module) -> bool:
    return name == BACKBONE_PREFIXES[0] or name.startswith(tuple(
        f"{prefix}." for prefix in BACKBONE_PREFIXES[1:]))


def installed_transforms(
        model: torch.nn.Module,
        ) -> tuple[InstalledNeuronLinearTransform, ...]:
    rows = []
    for name, module in model.named_modules():
        if not select_backbone_convolution(name, module):
            continue
        if not parametrize.is_parametrized(module, "weight"):
            continue
        transforms = tuple(module.parametrizations.weight)
        if (len(transforms) != 1
                or not isinstance(transforms[0], NeuronLinearKernelTransform)):
            raise RuntimeError(f"unexpected backbone parametrization at {name}")
        rows.append(InstalledNeuronLinearTransform(
            name=name, module=module, transform=transforms[0]))
    if not rows:
        raise RuntimeError("model has no installed backbone NLT")
    return tuple(rows)


def install_backbone_nlt(
        model: torch.nn.Module,
        ) -> tuple[InstalledNeuronLinearTransform, ...]:
    return install_neuron_linear_transforms(
        model,
        select=select_backbone_convolution,
        freeze_source_weight=False,
        freeze_source_bias=False)


def standard_parameter_contract(config) -> tuple[int, int]:
    model = EnsembleUNet2d(config).float()
    total = sum(value.numel() for value in model.parameters())
    controller = sum(
        value.numel() for value in model.energy_controller.parameters())
    del model
    gc.collect()
    return total, controller


def nlt_architecture_contract(
        horizon_local_energy_controller: bool,
        multiscale_energy_controller: bool = False,
        ) -> tuple:
    config, standard_total, controller = ORIGINAL_ARCHITECTURE_CONTRACT(
        horizon_local_energy_controller, multiscale_energy_controller)
    model = EnsembleUNet2d(config).float()
    if sum(value.numel() for value in model.parameters()) != standard_total:
        raise RuntimeError("standard NLT source parameter contract changed")
    rows = install_backbone_nlt(model)
    total = sum(value.numel() for value in model.parameters())
    if total != standard_total + neuron_linear_parameter_count(rows):
        raise RuntimeError("expanded NLT parameter contract changed")
    del model
    gc.collect()
    return config, total, controller


def validate_nlt_start(
        path: Path, expected_sha256: str, normalizer, device: torch.device,
        expected_config=None, expected_parameters=None,
        expected_controller_parameters=None):
    if expected_config is None or expected_parameters is None:
        raise ValueError("NLT validation requires an expanded contract")
    standard_total, standard_controller = standard_parameter_contract(
        expected_config)
    if (expected_controller_parameters is not None
            and expected_controller_parameters != standard_controller):
        raise RuntimeError("NLT controller parameter contract changed")
    model, report = ORIGINAL_VALIDATE(
        path, expected_sha256, normalizer, device,
        expected_config=expected_config,
        expected_parameters=standard_total,
        expected_controller_parameters=standard_controller)
    rows = install_backbone_nlt(model)
    expanded = sum(value.numel() for value in model.parameters())
    nlt_parameters = neuron_linear_parameter_count(rows)
    if expanded != expected_parameters or expanded != standard_total + nlt_parameters:
        raise RuntimeError("validated NLT expansion differs from architecture contract")
    report = {
        **report,
        "nlt": {
            "scope": list(BACKBONE_PREFIXES),
            "convolutions": len(rows),
            "standard_parameters": standard_total,
            "nlt_parameters": nlt_parameters,
            "expanded_parameters": expanded,
            "identity_initialized": True,
        },
        "parameters": expanded,
    }
    return model, report


def configure_arm_trainable(
        model: torch.nn.Module, arm: str,
        expected_parameters: int | None = None,
        ) -> tuple[torch.nn.Parameter, ...]:
    if arm not in {"control", "candidate"}:
        raise ValueError(f"unknown NLT arm {arm!r}")
    rows = installed_transforms(model)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for row in rows:
        if arm == "control":
            row.transform.factor.requires_grad_(False)
            row.transform.offset.requires_grad_(False)
        else:
            row.source_weight.requires_grad_(False)
            if row.module.bias is not None:
                row.module.bias.requires_grad_(False)
    total = sum(value.numel() for value in model.parameters())
    if expected_parameters is not None and total != int(expected_parameters):
        raise RuntimeError("NLT trainability received the wrong total contract")
    trainable = tuple(
        value for value in model.parameters() if value.requires_grad)
    if not trainable or not all(
            value.requires_grad
            for value in model.energy_controller.parameters()):
        raise RuntimeError("NLT must leave the physical controller trainable")
    if arm == "control":
        if any(
                row.transform.factor.requires_grad
                or row.transform.offset.requires_grad for row in rows):
            raise RuntimeError("control NLT identity is not frozen")
    else:
        if any(
                row.source_weight.requires_grad
                or (row.module.bias is not None
                    and row.module.bias.requires_grad)
                for row in rows):
            raise RuntimeError("candidate source backbone convolution is trainable")
        if not all(
                row.transform.factor.requires_grad
                and row.transform.offset.requires_grad for row in rows):
            raise RuntimeError("candidate NLT parameters are frozen")
    return trainable


def configure_current_arm(
        model: torch.nn.Module,
        expected_parameters: int | None = None,
        ) -> tuple[torch.nn.Parameter, ...]:
    if _CURRENT_ARM is None:
        raise RuntimeError("NLT trainability requested outside a paired arm")
    return configure_arm_trainable(model, _CURRENT_ARM, expected_parameters)


def merged_standard_state(
        model: EnsembleUNet2d,
        parametrized_state: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    clone = EnsembleUNet2d(model.config).float()
    rows = install_backbone_nlt(clone)
    clone.load_state_dict(parametrized_state, strict=True)
    merge_neuron_linear_transforms(rows)
    result = OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in clone.state_dict().items())
    expected, _controller = standard_parameter_contract(model.config)
    if (sum(value.numel() for value in clone.parameters()) != expected
            or any("parametrizations.weight" in name for name in result)):
        raise RuntimeError("NLT endpoint did not merge to the standard model")
    del clone
    gc.collect()
    return result


def save_merged_endpoint(
        path: Path, model: EnsembleUNet2d,
        ema: small4k.ema_lib.FP32EMAShadow,
        *, start_report: Mapping[str, object],
        normalizer_report: Mapping[str, object], stream_sha256: str,
        ) -> tuple[dict, dict]:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite endpoint {path}")
    raw_before = paired.dr1.model_state_sha256(model.state_dict())
    nlt_ema_payload = ema.seal(model)
    raw_after = paired.dr1.model_state_sha256(model.state_dict())
    if raw_before != raw_after:
        raise RuntimeError("materializing NLT EMA mutated the raw model")
    parametrized = nlt_ema_payload.pop("model_state")
    if (int(nlt_ema_payload["updates"]) != small4k.UPDATES
            or float(nlt_ema_payload["decay"]) != small4k.EMA_DECAY
            or small4k.ema_lib.tensor_state_sha256(parametrized)
            != nlt_ema_payload["materialized_model_state_sha256"]):
        raise RuntimeError("NLT EMA seal contract changed")
    state = merged_standard_state(model, parametrized)
    state_sha = paired.dr1.model_state_sha256(state)
    model_config = model.architecture_config()
    # Bind the strict-loadable merged endpoint with a standard-model EMA seal.
    # The chronological expanded-parameter EMA evidence remains separately
    # preserved below as ``nlt_training_ema``; this second seal is an explicit
    # algebraic migration, not a claim that ordinary kernels were averaged at
    # every optimizer step.
    merged_model = EnsembleUNet2d(model.config).float()
    merged_model.load_state_dict(state, strict=True)
    endpoint_ema = small4k.ema_lib.FP32EMAShadow(
        merged_model, small4k.EMA_DECAY)
    endpoint_ema._updates = small4k.UPDATES
    endpoint_ema_payload = endpoint_ema.seal(merged_model)
    endpoint_ema_state = endpoint_ema_payload.pop("model_state")
    if (paired.dr1.model_state_sha256(endpoint_ema_state) != state_sha
            or endpoint_ema_payload["materialized_model_state_sha256"]
            != state_sha):
        raise RuntimeError("merged standard EMA seal does not bind endpoint")
    del endpoint_ema_state, endpoint_ema, merged_model
    gc.collect()
    payload = {
        "schema": small4k.ENDPOINT_SCHEMA,
        "experiment": small4k.EXPERIMENT,
        "phase": "real",
        "iteration": small4k.UPDATES,
        "endpoint": small4k.PRIMARY_ENDPOINT,
        "primary": True,
        "held_aoa": small4k.HELD_AOA,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model_config,
        "start": dict(start_report),
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": str(normalizer_report["sha256"]),
        "sample_stream_sha256": stream_sha256,
        "ema": dict(endpoint_ema_payload),
        "nlt_training_ema": dict(nlt_ema_payload),
        "nlt_merge": {
            "merged_to_standard_convolutions": True,
            "parametrized_ema_state_sha256": nlt_ema_payload[
                "materialized_model_state_sha256"],
            "merged_model_state_sha256": state_sha,
            "inference_branch_added": False,
        },
        "optimizer_state_saved": False,
        "held_prefix_opened_before_seal": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    seal = {
        "endpoint": small4k.PRIMARY_ENDPOINT,
        "path": str(path.resolve()),
        "file_sha256": paired.dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": paired.dr1.canonical_sha256(model_config),
        "sample_stream_sha256": stream_sha256,
        "iteration": small4k.UPDATES,
        "primary": True,
        "optimizer_state_saved": False,
    }
    del parametrized, state
    return seal, {
        **nlt_ema_payload,
        "raw_model_unchanged_by_seal": raw_before == raw_after,
        "nlt_merged_to_standard_convolutions": True,
        "merged_model_state_sha256": state_sha,
        "merged_endpoint_ema_materialized_model_state_sha256": (
            endpoint_ema_payload["materialized_model_state_sha256"]),
    }


def load_merged_endpoint(
        seal, normalizer, device, *, expected_model_config=None,
        expected_parameters=None, expected_controller_parameters=None):
    if expected_model_config is None:
        raise ValueError("NLT endpoint load requires the frozen model config")
    standard, controller = standard_parameter_contract(expected_model_config)
    if (expected_controller_parameters is not None
            and expected_controller_parameters != controller):
        raise RuntimeError("NLT endpoint controller contract changed")
    return ORIGINAL_LOAD(
        seal, normalizer, device,
        expected_model_config=expected_model_config,
        expected_parameters=standard,
        expected_controller_parameters=controller)


def train_arm(*args, **kwargs):
    global _CURRENT_ARM
    access = kwargs.get("access")
    if access is None and len(args) >= 9:
        access = args[8]
    arm = getattr(access, "arm", None)
    if arm not in {"control", "candidate"}:
        raise ValueError("NLT runner cannot identify paired arm")
    if _CURRENT_ARM is not None:
        raise RuntimeError("NLT paired arm state is already active")
    _CURRENT_ARM = arm
    try:
        training, seal = ORIGINAL_TRAIN(*args, **kwargs)
    finally:
        _CURRENT_ARM = None
    training["real_transfer_strategy"] = (
        "full_finetune_with_frozen_identity_nlt"
        if arm == "control" else "mergeable_backbone_nlt")
    training["nlt_scope"] = list(BACKBONE_PREFIXES)
    training["nlt_merged_endpoint"] = True
    return training, seal


@torch.no_grad()
def ema_update_skip_frozen(self, model, *, optimizer_step: int) -> None:
    if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int):
        raise TypeError("optimizer_step must be an integer")
    if optimizer_step != self._updates + 1:
        raise ValueError("NLT EMA optimizer step changed")
    named = self._validated_named_parameters(model)
    for name, parameter in named:
        source = parameter.detach()
        if source.dtype != torch.float32:
            source = source.to(dtype=torch.float32)
        if parameter.requires_grad:
            self._shadow[name].mul_(self.decay).add_(
                source, alpha=1.0 - self.decay)
        else:
            self._shadow[name].copy_(source)
    self._updates = optimizer_step


def identical_dense_augmentation() -> dict[str, dict[str, float]]:
    return {
        "control": dict(paired.AUGMENTATION),
        "candidate": dict(paired.AUGMENTATION),
    }


def identical_dense_windows(trajectories, _manifest_windows) -> dict[str, list]:
    control = paired.dense_windows(trajectories, 5)
    candidate = paired.dense_windows(trajectories, 5)
    if [(row.trajectory, row.start) for row in control] != [
            (row.trajectory, row.start) for row in candidate]:
        raise RuntimeError("NLT paired dense windows diverged")
    return {"control": control, "candidate": candidate}


def install_contract() -> None:
    if getattr(paired, "_nlt_real_contract_installed", False):
        raise RuntimeError("NLT Real contract installed twice")
    paired.EXPERIMENT = EXPERIMENT
    paired.SCHEMA = SCHEMA
    paired.CONTROL_STRIDE = 5
    paired.CANDIDATE_STRIDE = 5
    paired.architecture_contract = nlt_architecture_contract
    paired.arm_window_contract = identical_dense_windows
    paired.arm_augmentation_contract = identical_dense_augmentation
    paired.ft12.validate_start_checkpoint = validate_nlt_start
    paired.ft12.configure_trainable = configure_current_arm
    paired.small4k.train_candidate = train_arm
    paired.small4k.save_primary_endpoint = save_merged_endpoint
    paired.small4k.load_primary_endpoint = load_merged_endpoint
    paired.small4k.ema_lib.FP32EMAShadow.update_after_optimizer_step = (
        ema_update_skip_frozen)
    paired._nlt_real_contract_installed = True


def main() -> None:
    install_contract()
    paired.main()


if __name__ == "__main__":
    main()
