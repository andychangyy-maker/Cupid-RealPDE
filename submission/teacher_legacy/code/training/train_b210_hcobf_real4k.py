#!/usr/bin/env python3
"""Formal paired HCOB-F Real4k run with no score or endpoint gate.

Both arms safely upgrade one immutable released-data-only XCB/member Sim24k
endpoint.  They use the same mature dense Real4k optimizer path; the only
difference is whether its explicit history-condition override comes from the
recipient or a pre-sealed same-ordinal cross-trajectory donor.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Callable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_rde_moe_real4k as rde
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig


EXPERIMENT = "b210_hcobf_real4k"
SCHEMA = "b210_hcobf_real4k_v2"
ENDPOINT_SCHEMA = "b210_hcobf_real4k_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_B210_HCOBF_REAL4K_V1"
FORMAL_BRANCH = "codex/hcobf-formal-real4k"
ARM_ORDER = ("aligned", "fixed_permuted")
ENDPOINT_ORDER = ("raw", "ema")
CONDITION_MODES = ("native", "fixed_permuted")
PERMUTATION_SEED = 20_260_904
HCOBF_INITIALIZATION_SEED = 20_260_905

# Reuse the released-data Real4k execution contract rather than adding a new
# schedule, loss, augmentation or EMA implementation.
UPDATES = rde.UPDATES
SCHEDULER_T_MAX = rde.SCHEDULER_T_MAX
EMA_DECAY = rde.EMA_DECAY
AUGMENTATION = rde.dense4k.AUGMENTATION
BATCH_SIZE = small4k.BATCH_SIZE
EXPECTED_START_ITERATION = 4_000
EXPECTED_START_STATUS = "PASS_B210_XCB_MEMBER_STANDARD24K_SIM"
EXPECTED_START_METHOD = "b210_xcb_member_standard_stride20_20k_then_4k_v1"
EXPECTED_START_SHA256 = (
    "06ade419811d722e58235acdcb34e480dd6ae88555dd52037ebcb6bbeba2ce99")
EXPECTED_MANIFEST_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
EXPECTED_EVALUATION_MANIFEST_SHA256 = (
    "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
EXPECTED_DENSE_WINDOWS = 9_933
EXPECTED_TAIL_WINDOWS = 716
EXPECTED_BASE_PARAMETERS = 180_553_371
EXPECTED_HCOBF_PARAMETERS = 427_470
EXPECTED_PARAMETERS = EXPECTED_BASE_PARAMETERS + EXPECTED_HCOBF_PARAMETERS
EXPECTED_CONTROLLER_PARAMETERS = ft12.EXPECTED_CONTROLLER_PARAMETERS
OFFICIAL_POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
}
OFFICIAL_SPS_WEIGHT = 0.25
EXPECTED_HISTORICAL_XCB_RESULT_SHA256 = (
    "8f288149b0418fbc4f9d0744d4f5e6a43624bffb2eb7347a8d4a66ae3c328697")
EXPECTED_HISTORICAL_XCB_CHECKPOINT_SHA256 = (
    "095399c414f1193e6b281f287dcb31037632a01db55d765bf2c599b535f2b1f4")
EXPECTED_SAMPLE_STREAM_SHA256 = (
    "14331e0ff2d61c7f62a384061b47f629f6a71999fad6ba4e97f37d414f17a7fb")
EXPECTED_NORMALIZER_STATISTICS_SHA256 = (
    "3f2605d84ad8da57fedddfaeff292050f1173ce37f75defa9dd0de6e4b4e9492")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--historical-xcb-result", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-contract", action="store_true")
    return parser.parse_args(argv)


def dry_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "formal_branch": FORMAL_BRANCH,
        "acknowledgement": ACKNOWLEDGEMENT,
        "arms": list(ARM_ORDER),
        "endpoints_each": list(ENDPOINT_ORDER),
        "updates": UPDATES,
        "scheduler_t_max": SCHEDULER_T_MAX,
        "ema_decay": EMA_DECAY,
        "augmentation": dict(AUGMENTATION),
        "start_sha256": EXPECTED_START_SHA256,
        "clean_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "clean716_manifest_sha256": EXPECTED_EVALUATION_MANIFEST_SHA256,
        "historical_xcb_result_sha256": (
            EXPECTED_HISTORICAL_XCB_RESULT_SHA256),
        "condition_override_both_arms": True,
        "score_gate": None,
        "endpoint_selection": False,
        "submission_authorized": False,
    }


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
        raise ValueError("start is not the frozen XCB/member Sim24k endpoint")
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
    if (config.base != 210 or config.depth != 4
            or not config.cross_conditioned_backbone
            or config.member_adapter_channels != 64
            or not config.mean_fluctuation
            or not config.physical_energy_controller
            or not config.physical_energy_member_sync
            or config.history_conditioned_operator_basis
            or config.dual_decoder_mode != "none"):
        raise ValueError("start is not the frozen XCB/member architecture")
    state = payload["model_state_dict"]
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in state.values() if isinstance(value, torch.Tensor)):
        raise TypeError("start floating state must be FP32")
    return config, state, {
        "phase": "sim",
        "iteration": EXPECTED_START_ITERATION,
        "status": EXPECTED_START_STATUS,
        "method": EXPECTED_START_METHOD,
        "checkpoint_model_config_sha256": dr1.canonical_sha256(
            payload["model_config"]),
        "checkpoint_model_state_sha256": dr1.model_state_sha256(state),
        "checkpoint_normalizer_sha256": dr1.canonical_sha256(
            payload.get("normalizer")),
    }


def arm_config(base: EnsembleUNet2dConfig, arm: str) -> EnsembleUNet2dConfig:
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown HCOB-F arm {arm!r}")
    return replace(base, history_conditioned_operator_basis=True)


def new_state_keys(model: EnsembleUNet2d,
                   base_state: Mapping[str, torch.Tensor]) -> set[str]:
    return set(model.state_dict()) - set(base_state)


def parameter_manifest(model: EnsembleUNet2d) -> list[dict[str, object]]:
    return [{
        "name": name,
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "requires_grad": bool(parameter.requires_grad),
    } for name, parameter in model.named_parameters()]


def upgrade_released_start(
        base: EnsembleUNet2dConfig, state: Mapping[str, torch.Tensor],
        arm: str) -> EnsembleUNet2d:
    """Attach HCOB-F without perturbing caller RNG or any common tensor."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(HCOBF_INITIALIZATION_SEED)
        model = EnsembleUNet2d(arm_config(base, arm)).float()
    incompatible = model.load_state_dict(state, strict=False)
    missing = new_state_keys(model, state)
    if (set(incompatible.missing_keys) != missing
            or incompatible.unexpected_keys or not missing
            or any(not name.startswith((
                "history_operator_conditioner.",
                "history_operator_basis.")) for name in missing)):
        raise RuntimeError("released endpoint HCOB-F upgrade changed common state")
    if any(not torch.equal(value, model.state_dict()[name])
           for name, value in state.items()):
        raise RuntimeError("released common tensor changed during HCOB-F upgrade")
    return model


def load_start(path: Path, arm: str, normalizer,
               device: torch.device) -> tuple[
                   EnsembleUNet2d, EnsembleUNet2dConfig, dict[str, object]]:
    actual = dr1.sha256(path)
    if actual != EXPECTED_START_SHA256:
        raise RuntimeError("XCB/member Sim24k start checkpoint SHA changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    base, state, report = parse_start_payload(payload)
    model = upgrade_released_start(base, state, arm)
    total = sum(parameter.numel() for parameter in model.parameters())
    hcobf = sum(parameter.numel() for name, parameter in model.named_parameters()
                if name.startswith(("history_operator_conditioner.",
                                    "history_operator_basis.")))
    controller = sum(parameter.numel()
                     for parameter in model.energy_controller.parameters())
    if (total != EXPECTED_PARAMETERS or hcobf != EXPECTED_HCOBF_PARAMETERS
            or controller != EXPECTED_CONTROLLER_PARAMETERS):
        raise RuntimeError(
            f"HCOB-F parameter contract changed: total={total}, hcobf={hcobf}, "
            f"controller={controller}")
    small4k.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    report.update({
        "path": str(path.resolve()),
        "checkpoint_sha256": actual,
        "arm": arm,
        "strict_common_state_load": True,
        "new_state_keys": sorted(new_state_keys(model, state)),
        "parameters": total,
        "hcobf_parameters": hcobf,
        "controller_parameters": controller,
        "parameter_manifest_sha256": dr1.canonical_sha256(
            parameter_manifest(model)),
        "upgraded_model_state_sha256": dr1.model_state_sha256(
            model.state_dict()),
    })
    del payload, state
    gc.collect()
    return model.to(device), base, report


def make_fixed_cross_trajectory_permutation(
        window_keys: Sequence[Sequence[object]],
        seed: int = PERMUTATION_SEED) -> tuple[int, ...]:
    """Return a deterministic same-ordinal cross-trajectory bijection."""
    groups: dict[int, list[int]] = {}
    for index, key in enumerate(window_keys):
        if len(key) not in (2, 3):
            raise ValueError(
                "window keys must be (trajectory, ordinal[, start])")
        groups.setdefault(int(key[1]), []).append(index)
    result = [-1] * len(window_keys)
    for ordinal, indexes in sorted(groups.items()):
        ordered = sorted(indexes, key=lambda index: (
            str(window_keys[index][0]), index))
        trajectories = [str(window_keys[index][0]) for index in ordered]
        if len(ordered) < 2 or len(set(trajectories)) != len(trajectories):
            raise ValueError(
                "each ordinal needs distinct trajectories for permutation")
        digest = hashlib.sha256(f"{seed}:{ordinal}".encode()).digest()
        shift = 1 + int.from_bytes(digest[:8], "little") % (len(ordered) - 1)
        for recipient, donor in zip(ordered, ordered[shift:] + ordered[:shift]):
            result[recipient] = donor
    if sorted(result) != list(range(len(window_keys))):
        raise RuntimeError("fixed permutation is not a complete bijection")
    if any(window_keys[index][0] == window_keys[donor][0]
           or int(window_keys[index][1]) != int(window_keys[donor][1])
           for index, donor in enumerate(result)):
        raise RuntimeError("fixed permutation violated trajectory/ordinal")
    return tuple(result)


def permutation_record(
        window_keys: Sequence[Sequence[object]],
        seed: int = PERMUTATION_SEED) -> dict[str, object]:
    mapping = make_fixed_cross_trajectory_permutation(window_keys, seed)
    keys = [list(key) for key in window_keys]
    canonical = {"keys": keys, "mapping": list(mapping), "seed": seed}
    return {
        "seed": seed,
        "windows": len(keys),
        "window_keys_sha256": dr1.canonical_sha256(keys),
        "mapping": list(mapping),
        "mapping_sha256": dr1.canonical_sha256(canonical),
        "rule": "split-global same-ordinal cross-trajectory seeded bijection",
        "target_tensors_used": False,
    }


def dense_key_inventory(rows: Sequence[Mapping[str, object]]) -> list[list[object]]:
    result = []
    for row in rows:
        starts = range(0, int(row["val_boundary"]) - 40 + 1,
                       dense4k.CANDIDATE_STRIDE)
        result.extend([str(row["file"]), ordinal, int(start)]
                      for ordinal, start in enumerate(starts))
    return result


def tail_key_inventory(rows: Sequence[Mapping[str, object]]) -> list[list[object]]:
    return [[str(row["file"]), ordinal, int(start)]
            for row in rows
            for ordinal, start in enumerate(row["train_starts"])]


def loaded_window_keys(windows: Sequence,
                       metadata: Sequence[Mapping[str, object]]) -> list[list[object]]:
    ordinal: dict[int, int] = {}
    result = []
    for window in windows:
        trajectory = int(window.trajectory)
        current = ordinal.get(trajectory, 0)
        result.append([str(metadata[trajectory]["file"]), current,
                       int(window.start)])
        ordinal[trajectory] = current + 1
    return result


def normalized_unaugmented_history(history: torch.Tensor, normalizer,
                                   device: torch.device) -> torch.Tensor:
    history = history.to(device, non_blocking=True)
    if history.ndim != 5 or history.shape[-1] != 3:
        raise ValueError("history condition must be BxTxHxWx3")
    return ((history - normalizer.mean_input) / normalizer.std_input)


def donor_condition(model: EnsembleUNet2d,
                    donor_history_norm: torch.Tensor) -> torch.Tensor:
    if donor_history_norm.ndim != 5:
        raise ValueError("normalized donor history must be BxTxHxWxC")
    return model.history_operator_condition(donor_history_norm)


def officialize_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Remove the legacy 50/30/20 proxy and add exact published weights."""
    required = (*OFFICIAL_POINT_WEIGHTS, "native_sps_score",
                "native_sps_coverage", "native_sps_mean_width")
    if any(key not in metrics or not math.isfinite(float(metrics[key]))
           for key in required):
        raise ValueError("metric payload is incomplete or non-finite")
    result = {key: float(metrics[key]) for key in required}
    result["official_point_contribution"] = sum(
        OFFICIAL_POINT_WEIGHTS[key] * result[key]
        for key in OFFICIAL_POINT_WEIGHTS)
    result["official_non_time_contribution"] = (
        result["official_point_contribution"]
        + OFFICIAL_SPS_WEIGHT * result["native_sps_score"])
    return result


@contextmanager
def fixed_permuted_override(
        model: EnsembleUNet2d,
        donor_history_norm: torch.Tensor) -> Iterator[None]:
    model.set_history_operator_condition_override(
        donor_condition(model, donor_history_norm))
    try:
        yield
    finally:
        model.set_history_operator_condition_override(None)


class TrainingConditionCallback:
    """Resolve split-global donors and scope one differentiable override."""

    def __init__(self, arm: str, dataset, mapping: Sequence[int]) -> None:
        if arm not in ARM_ORDER or len(mapping) != len(dataset):
            raise ValueError("invalid training condition callback")
        self.arm = arm
        self.dataset = dataset
        self.mapping = tuple(int(value) for value in mapping)
        self.calls = 0
        self._binding_hasher = hashlib.sha256()
        self.gradient_audit: dict[str, bool] = {}
        self.rng_neutral = True

    def _histories(self, batch: Sequence[torch.Tensor]) -> tuple[
            torch.Tensor, list[int], list[int]]:
        recipients = batch[3].to(dtype=torch.long).tolist()
        if self.arm == "aligned":
            donors = list(recipients)
            histories = batch[0]
        else:
            donors = [self.mapping[index] for index in recipients]
            histories = torch.stack([self.dataset[index][0]
                                     for index in donors])
        return histories, recipients, donors

    @contextmanager
    def __call__(self, iteration: int, model: EnsembleUNet2d,
                 batch: Sequence[torch.Tensor], normalizer,
                 device: torch.device) -> Iterator[None]:
        histories, recipients, donors = self._histories(batch)
        if len(recipients) != len(donors):
            raise RuntimeError("recipient/donor batch length changed")
        cpu_rng = torch.get_rng_state() if self.calls == 0 else None
        cuda_rng = (torch.cuda.get_rng_state(device) if self.calls == 0
                    and device.type == "cuda" else None)
        normalized = normalized_unaugmented_history(
            histories, normalizer, device)
        condition = donor_condition(model, normalized)
        if cpu_rng is not None:
            self.rng_neutral &= torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            self.rng_neutral &= torch.equal(
                cuda_rng, torch.cuda.get_rng_state(device))
        self._binding_hasher.update(json.dumps({
            "iteration": int(iteration), "recipients": recipients,
            "donors": donors}, sort_keys=True, separators=(",", ":")).encode())
        self.calls += 1
        model.set_history_operator_condition_override(condition)
        try:
            yield
        finally:
            if iteration in (1, 2):
                projection = [basis.output_projection.weight.grad
                              for basis in model.history_operator_basis]
                condition_grads = [parameter.grad for parameter in
                                   model.history_operator_conditioner.parameters()]
                gate_grads = [basis.operator_logits.weight.grad
                              for basis in model.history_operator_basis]
                self.gradient_audit[f"u{iteration}_projection_nonzero"] = all(
                    value is not None and bool(torch.count_nonzero(value))
                    for value in projection)
                self.gradient_audit[f"u{iteration}_condition_nonzero"] = any(
                    value is not None and bool(torch.count_nonzero(value))
                    for value in condition_grads)
                self.gradient_audit[f"u{iteration}_gate_nonzero"] = all(
                    value is not None and bool(torch.count_nonzero(value))
                    for value in gate_grads)
            model.set_history_operator_condition_override(None)

    def report(self) -> dict[str, object]:
        required = {
            "u1_projection_nonzero": True,
            "u1_condition_nonzero": False,
            "u1_gate_nonzero": False,
            "u2_projection_nonzero": True,
            "u2_condition_nonzero": True,
            "u2_gate_nonzero": True,
        }
        if self.calls != UPDATES or self.gradient_audit != required:
            raise RuntimeError(
                "HCOB-F condition callback/gradient contract changed: "
                f"calls={self.calls}, audit={self.gradient_audit}")
        if not self.rng_neutral:
            raise RuntimeError("history condition callback consumed RNG")
        return {
            "calls": self.calls,
            "binding_stream_sha256": self._binding_hasher.hexdigest(),
            "condition_source": ("recipient unaugmented normalized history"
                                 if self.arm == "aligned" else
                                 "sealed donor unaugmented normalized history"),
            "global_mapping_never_used_as_batch_index": True,
            "rng_neutral": self.rng_neutral,
            "gradient_audit": dict(self.gradient_audit),
        }


def _five_outputs(model: EnsembleUNet2d,
                  history_norm: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    values = model.forward_uq_training(history_norm)[:5]
    return OrderedDict((name, value.detach().float().cpu()) for name, value in zip(
        ("prediction", "members", "quantiles", "lower", "upper"), values))


def update0_signature(model: EnsembleUNet2d, dataset, mapping: Sequence[int],
                      normalizer, device: torch.device,
                      condition_mode: str) -> tuple[str, OrderedDict]:
    indices = list(range(min(2, len(dataset))))
    histories = torch.stack([dataset[index][0] for index in indices])
    history_norm = normalized_unaugmented_history(histories, normalizer, device)
    if condition_mode == "native":
        condition_history = history_norm
    elif condition_mode == "fixed_permuted":
        donor = torch.stack([dataset[int(mapping[index])][0]
                             for index in indices])
        condition_history = normalized_unaugmented_history(
            donor, normalizer, device)
    else:
        raise ValueError(condition_mode)
    model.eval()
    with torch.inference_mode():
        model.set_history_operator_condition_override(
            donor_condition(model, condition_history))
        try:
            state = _five_outputs(model, history_norm)
        finally:
            model.set_history_operator_condition_override(None)
    return dr1.model_state_sha256(state), state


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


def save_raw_endpoint(path: Path, model: EnsembleUNet2d, arm: str,
                      start_report: Mapping[str, object], normalizer,
                      stream_sha256: str,
                      mapping_sha256: str) -> dict[str, object]:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    state = OrderedDict((name, value.detach().cpu())
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
        "arm": arm,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start": dict(start_report),
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": normalizer_report["sha256"],
        "sample_stream_sha256": stream_sha256,
        "fit_mapping_sha256": mapping_sha256,
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
        "model_config_sha256": dr1.canonical_sha256(payload["model_config"]),
        "sample_stream_sha256": stream_sha256,
        "fit_mapping_sha256": mapping_sha256,
        "iteration": UPDATES,
        "primary": False,
        "optimizer_state_saved": False,
    }


def train_arm(
        arm: str, args: argparse.Namespace, dataset,
        schedule: Mapping[str, object], normalizer,
        fit_mapping: Mapping[str, object], registry: rde.EndpointRegistry,
        expected_initial: str | None, expected_update0: str | None,
        deadline: float) -> tuple[dict[str, object], str, str,
                                  EnsembleUNet2dConfig]:
    device = torch.device(args.device)
    model, base_config, start_report = load_start(
        args.start_sim, arm, normalizer, device)
    initial = str(start_report["upgraded_model_state_sha256"])
    if expected_initial is not None and initial != expected_initial:
        raise RuntimeError("paired HCOB-F initial states differ")
    mapping = fit_mapping["mapping"]
    native_sha, native_outputs = update0_signature(
        model, dataset, mapping, normalizer, device, "native")
    permuted_sha, permuted_outputs = update0_signature(
        model, dataset, mapping, normalizer, device, "fixed_permuted")
    if native_sha != permuted_sha or any(
            not torch.equal(native_outputs[key], permuted_outputs[key])
            for key in native_outputs):
        raise RuntimeError("HCOB-F update-zero five-output identity changed")
    if expected_update0 is not None and native_sha != expected_update0:
        raise RuntimeError("paired HCOB-F update-zero outputs differ")
    callback = TrainingConditionCallback(arm, dataset, mapping)
    arm_dir = args.output_dir / arm
    arm_dir.mkdir()
    seal_adapter = rde.EmaSealAdapter(registry, arm)
    small4k.npt1.ensemble_train.seed_everything(small4k.SEED)
    with endpoint_contract(arm):
        training, ema_seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, seal_adapter,
            input_augmentation=AUGMENTATION,
            endpoint_filename=f"ema_u4000_{arm}.pth",
            expected_parameters=EXPECTED_PARAMETERS,
            scheduler_t_max=SCHEDULER_T_MAX,
            forward_context=callback)
    if seal_adapter.value != ema_seal:
        raise RuntimeError("EMA endpoint was not bound to registry")
    # The mature EMA payload does not know about the mapping; bind it in its
    # immutable external seal and in the final result contract.
    ema_seal["fit_mapping_sha256"] = fit_mapping["mapping_sha256"]
    registry.values[arm]["ema"]["fit_mapping_sha256"] = fit_mapping[
        "mapping_sha256"]
    raw_seal = save_raw_endpoint(
        arm_dir / f"raw_u4000_{arm}.pth", model, arm, start_report,
        normalizer, str(schedule["sha256"]),
        str(fit_mapping["mapping_sha256"]))
    registry.seal(arm, "raw", raw_seal)
    context_report = callback.report()
    report = {
        "condition_training": arm,
        "start": dict(start_report),
        "model_config": model.architecture_config(),
        "model_config_sha256": dr1.canonical_sha256(
            model.architecture_config()),
        "parameters": EXPECTED_PARAMETERS,
        "parameter_manifest_sha256": start_report[
            "parameter_manifest_sha256"],
        "initial_state_sha256": initial,
        "update0_five_output_sha256": native_sha,
        "update0_condition_modes_bitwise": True,
        "sample_stream_sha256": schedule["sha256"],
        "fit_mapping_sha256": fit_mapping["mapping_sha256"],
        "training_condition_callback": context_report,
        "training": training,
        "seals": {"raw": raw_seal, "ema": ema_seal},
    }
    del model, native_outputs, permuted_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return report, initial, native_sha, base_config


def load_endpoint(seal: Mapping[str, object], arm: str, normalizer,
                  device: torch.device,
                  base_config: EnsembleUNet2dConfig) -> EnsembleUNet2d:
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
        raise RuntimeError("endpoint HCOB-F config changed")
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != (
            EXPECTED_PARAMETERS):
        raise RuntimeError("endpoint parameter count changed")
    small4k.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    del payload, state
    gc.collect()
    return model.to(device)


class GateAccumulator:
    def __init__(self, scales: int) -> None:
        self.count = 0
        self.sum = [torch.zeros(3, 64, dtype=torch.float64)
                    for _ in range(scales)]
        self.square = [torch.zeros(3, 64, dtype=torch.float64)
                       for _ in range(scales)]
        self.entropy = [0.0] * scales
        self.hard = [torch.zeros(3, dtype=torch.long) for _ in range(scales)]

    def update(self, model: EnsembleUNet2d, condition: torch.Tensor) -> None:
        batch = condition.shape[0]
        self.count += batch
        for index, basis in enumerate(model.history_operator_basis):
            weight = basis.operator_logits(condition).reshape(
                batch, 3, 64).softmax(dim=1).detach().double().cpu()
            self.sum[index] += weight.sum(dim=0)
            self.square[index] += weight.square().sum(dim=0)
            self.entropy[index] += float((
                -(weight.clamp_min(1.0e-30).log() * weight).sum(dim=1)
            ).sum())
            self.hard[index] += torch.bincount(
                weight.argmax(dim=1).reshape(-1), minlength=3)

    def finalize(self) -> dict[str, object]:
        if self.count <= 0:
            raise RuntimeError("gate audit is empty")
        names = ("pointwise", "local", "global")
        scales = []
        for index, (total, square) in enumerate(zip(self.sum, self.square)):
            mean = total / self.count
            std = (square / self.count - mean.square()).clamp_min(0).sqrt()
            hard_total = int(self.hard[index].sum())
            scales.append({
                "scale": index,
                "mean_probability": {
                    name: float(mean[operator].mean())
                    for operator, name in enumerate(names)},
                "cross_sample_std_mean_over_channels": {
                    name: float(std[operator].mean())
                    for operator, name in enumerate(names)},
                "cross_sample_std_max_over_channels": {
                    name: float(std[operator].max())
                    for operator, name in enumerate(names)},
                "entropy_mean_over_samples_channels": (
                    self.entropy[index] / (self.count * 64)),
                "hard_usage_fraction": {
                    name: float(self.hard[index][operator]) / hard_total
                    for operator, name in enumerate(names)},
            })
        return {"samples": self.count, "scales": scales}


class ErrorAccumulator:
    def __init__(self, horizons: int = 20) -> None:
        self.samples = 0
        self.mean_rel_sum = 0.0
        self.fluct_rel_sum = 0.0
        self.fluct_cos_sum = 0.0
        self.fluct_rms_square_sum = 0.0
        self.horizon_rel_sum = torch.zeros(horizons, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction[..., :2].detach().float()
        target = target[..., :2].detach().float()
        batch = prediction.shape[0]
        tiny = torch.finfo(torch.float32).tiny
        pred_mean = prediction.mean(dim=1)
        true_mean = target.mean(dim=1)
        pred_fluct = prediction - pred_mean[:, None]
        true_fluct = target - true_mean[:, None]

        def rel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            dims = tuple(range(1, left.ndim))
            return ((left - right).square().sum(dim=dims).sqrt()
                    / right.square().sum(dim=dims).sqrt().clamp_min(tiny))

        mean_rel = rel(pred_mean, true_mean)
        fluct_rel = rel(pred_fluct, true_fluct)
        flat_pred = pred_fluct.flatten(1)
        flat_true = true_fluct.flatten(1)
        cosine = ((flat_pred * flat_true).sum(dim=1)
                  / (flat_pred.norm(dim=1) * flat_true.norm(dim=1)
                     ).clamp_min(tiny))
        rms_square = (pred_fluct - true_fluct).square().mean(dim=(1, 2, 3, 4))
        horizon_error = (prediction - target).square().flatten(2).sum(dim=2).sqrt()
        horizon_truth = target.square().flatten(2).sum(dim=2).sqrt().clamp_min(tiny)
        self.samples += batch
        self.mean_rel_sum += float(mean_rel.sum())
        self.fluct_rel_sum += float(fluct_rel.sum())
        self.fluct_cos_sum += float(cosine.sum())
        self.fluct_rms_square_sum += float(rms_square.sum())
        self.horizon_rel_sum += (horizon_error / horizon_truth).sum(
            dim=0).double().cpu()

    def finalize(self) -> dict[str, object]:
        if self.samples <= 0:
            raise RuntimeError("error audit is empty")
        return {
            "samples": self.samples,
            "temporal_mean_relative_l2_mean": self.mean_rel_sum / self.samples,
            "fluctuation_relative_l2_mean": self.fluct_rel_sum / self.samples,
            "fluctuation_cosine_mean": self.fluct_cos_sum / self.samples,
            "fluctuation_error_rms": math.sqrt(
                self.fluct_rms_square_sum / self.samples),
            "relative_l2_mean_by_horizon": [
                float(value / self.samples) for value in self.horizon_rel_sum],
        }


@torch.inference_mode()
def evaluate_endpoint(
        model: EnsembleUNet2d, dataset, windows: Sequence,
        metadata: Sequence[Mapping[str, object]], mapping: Sequence[int],
        normalizer, device: torch.device, deadline: float,
        condition_mode: str) -> dict[str, object]:
    if condition_mode not in CONDITION_MODES or len(mapping) != len(dataset):
        raise ValueError("invalid endpoint condition evaluation")
    indices = list(range(len(dataset)))
    batches = [indices[offset:offset + smu1.EVAL_BATCH_SIZE]
               for offset in range(0, len(indices), smu1.EVAL_BATCH_SIZE)]
    tagged = small4k.npt1.ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(tagged, batch_sampler=batches, num_workers=0,
                        pin_memory=device.type == "cuda")
    state = smu1.new_metric_state()
    by_aoa_state = {aoa: smu1.new_metric_state() for aoa in dr1.AOA_VALUES}
    error = ErrorAccumulator()
    gate = GateAccumulator(len(model.history_operator_basis))
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    target_samples = 0
    target_scored = 0
    model.eval()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for batch_indices, batch in zip(batches, loader):
        if time.perf_counter() > deadline:
            raise TimeoutError("HCOB-F clean716 evaluation deadline expired")
        history, target, trajectory_ids, sample_indices = batch
        if sample_indices.tolist() != batch_indices:
            raise RuntimeError("evaluation sample-index order changed")
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        target_hasher.update(str(tuple(target_cpu.shape)).encode("ascii"))
        target_hasher.update(target_cpu.numpy().tobytes(order="C"))
        mask_hasher.update(str(tuple(scored_cpu.shape)).encode("ascii"))
        mask_hasher.update(scored_cpu.numpy().tobytes(order="C"))
        target_samples += len(target_cpu)
        target_scored += int(scored_cpu.sum())
        history_device = history.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        history_norm = normalized_unaugmented_history(
            history_device, normalizer, device)
        if condition_mode == "native":
            condition_history = history_norm
        else:
            donors = [int(mapping[index]) for index in batch_indices]
            donor_history = torch.stack([dataset[index][0] for index in donors])
            condition_history = normalized_unaugmented_history(
                donor_history, normalizer, device)
        condition = donor_condition(model, condition_history)
        model.set_history_operator_condition_override(condition)
        try:
            prediction_norm, lower_norm, upper_norm = (
                model.forward_distribution(history_norm))
        finally:
            model.set_history_operator_condition_override(None)
        prediction = normalizer.postprocess_prediction(prediction_norm.float())
        truth = target_device.float()
        lower = lower_norm.float() * scale
        upper = upper_norm.float() * scale
        smu1.update_metric_state(state, prediction, truth, lower, upper)
        error.update(prediction, truth)
        gate.update(model, condition)
        aoas = torch.tensor([
            int(metadata[int(trajectory)]["aoa"])
            for trajectory in trajectory_ids.tolist()], device=device)
        for aoa in dr1.AOA_VALUES:
            selected = aoas == aoa
            if bool(selected.any()):
                smu1.update_metric_state(
                    by_aoa_state[aoa], prediction[selected], truth[selected],
                    lower[selected], upper[selected])
    if (int(state["samples"]) != len(dataset)
            or target_samples != len(dataset)
            or target_scored != int(state["sps_scored"])):
        raise RuntimeError("clean716 scorer did not consume exact target order")
    return {
        "condition_mode": condition_mode,
        "metrics": officialize_metrics(smu1.finalize_metric_state(state)),
        "metric_state": state,
        "metrics_by_aoa": {
            str(aoa): officialize_metrics(
                smu1.finalize_metric_state(by_aoa_state[aoa]))
            for aoa in dr1.AOA_VALUES},
        "mean_fluctuation_horizon": error.finalize(),
        "gate": gate.finalize(),
        "target_audit": {
            "ordered_raw_target_sha256": target_hasher.hexdigest(),
            "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
            "samples": target_samples,
            "scored_elements": target_scored,
        },
        "ordered_window_key_sha256": smu1.ordered_held_key_digest(
            indices, windows, metadata),
        "metadata_used_by_model_inference": False,
    }


def load_historical_xcb_ema_reference(
        path: Path | None, *, sample_stream_sha256: str,
        normalizer_statistics_sha256: str,
        target_audit: Mapping[str, object],
        ) -> dict[str, object] | None:
    """Load one SHA-bound same-recipe EMA context after current evaluations.

    This is context only: it is unavailable until all four current endpoints
    have been sealed and evaluated, and cannot select or stop an endpoint.
    The historical run did not retain a raw endpoint, so only EMA comparisons
    are scientifically supported.
    """
    if path is None:
        return None
    if dr1.sha256(path) != EXPECTED_HISTORICAL_XCB_RESULT_SHA256:
        raise RuntimeError("historical XCB Real4k result SHA changed")
    payload = json.loads(path.read_text())
    candidate = payload.get("arms", {}).get("candidate", {})
    training = candidate.get("training", {})
    seal = candidate.get("seal", {})
    if (payload.get("schema")
            != "b210_xcb_member_standard24k_real_dense_aug_v1"
            or payload.get("experiment")
            != "b210_xcb_member_standard24k_real_dense_aug"
            or payload.get("start", {}).get("checkpoint_sha256")
            != EXPECTED_START_SHA256
            or payload.get("evaluation_manifest", {}).get("sha256")
            != EXPECTED_EVALUATION_MANIFEST_SHA256
            or payload.get("normalizer", {}).get("statistics_sha256")
            != normalizer_statistics_sha256
            or normalizer_statistics_sha256
            != EXPECTED_NORMALIZER_STATISTICS_SHA256
            or int(candidate.get("fit_windows", -1)) != EXPECTED_DENSE_WINDOWS
            or candidate.get("input_augmentation") != dict(AUGMENTATION)
            or int(training.get("updates", -1)) != UPDATES
            or int(training.get("batch_size", -1)) != BATCH_SIZE
            or int(training.get("seed", -1)) != small4k.SEED
            or float(training.get("lr", math.nan)) != small4k.LR
            or float(training.get("weight_decay", math.nan))
            != small4k.WEIGHT_DECAY
            or int(training.get("scheduler_t_max", -1)) != SCHEDULER_T_MAX
            or float(training.get("ema_decay", math.nan)) != EMA_DECAY
            or training.get("input_augmentation") != dict(AUGMENTATION)
            or training.get("planned_stream_sha256") != sample_stream_sha256
            or training.get("actual_stream_sha256") != sample_stream_sha256
            or sample_stream_sha256 != EXPECTED_SAMPLE_STREAM_SHA256
            or candidate.get("target_audit") != dict(target_audit)
            or seal.get("file_sha256")
            != EXPECTED_HISTORICAL_XCB_CHECKPOINT_SHA256
            or seal.get("sample_stream_sha256") != sample_stream_sha256
            or seal.get("endpoint") != "ema_u4000"
            or seal.get("primary") is not True):
        raise RuntimeError("historical XCB EMA context contract changed")
    checkpoint = Path(str(seal.get("path")))
    if (not checkpoint.is_file()
            or dr1.sha256(checkpoint)
            != EXPECTED_HISTORICAL_XCB_CHECKPOINT_SHA256):
        raise RuntimeError("historical XCB EMA checkpoint changed")
    metric_state = candidate.get("metric_state")
    if not isinstance(metric_state, Mapping):
        raise RuntimeError("historical XCB metric state is missing")
    recomputed = smu1.finalize_metric_state(dict(metric_state))
    reported = candidate.get("metrics")
    if (not isinstance(reported, Mapping)
            or any(reported.get(key) != recomputed[key]
                   for key in recomputed)):
        raise RuntimeError("historical XCB metrics do not match metric state")
    return {
        "scope": "same-start/same-Real4k-recipe EMA context only",
        "result_path": str(path.resolve()),
        "result_sha256": EXPECTED_HISTORICAL_XCB_RESULT_SHA256,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": EXPECTED_HISTORICAL_XCB_CHECKPOINT_SHA256,
        "metrics": officialize_metrics(recomputed),
        "raw_endpoint_available": False,
        "used_for_gate_or_endpoint_selection": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute or args.acknowledge_protocol != ACKNOWLEDGEMENT:
        raise PermissionError("formal run requires exact protocol acknowledgement")
    required = (args.output_dir, args.real_dir, args.clean_manifest,
                args.evaluation_manifest, args.start_sim)
    if any(value is None for value in required):
        raise ValueError("formal run is missing a required path")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if dr1.sha256(args.clean_manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("clean81 prefix manifest SHA changed")
    git = git_report()
    if not git["clean"] or git["branch"] != FORMAL_BRANCH:
        raise RuntimeError(
            f"formal run requires clean {FORMAL_BRANCH}: {git}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal HCOB-F Real4k run requires CUDA")
    rows, manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    # Only stat the path before endpoint sealing.  Its content, names and tail
    # starts remain unread until all four endpoint files are immutable.
    if not args.evaluation_manifest.is_file():
        raise FileNotFoundError(args.evaluation_manifest)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = (math.inf if args.wall_budget_seconds is None
                else started + args.wall_budget_seconds)
    torch.set_float32_matmul_precision("high")

    # Seal from manifest keys before reading any Real H5 tensor.
    fit_keys = dense_key_inventory(rows)
    fit_mapping = permutation_record(fit_keys)
    if len(fit_keys) != EXPECTED_DENSE_WINDOWS:
        raise RuntimeError("dense fit key inventory changed")
    dr1.write_json_atomic(
        args.output_dir / "fit_mapping_seal.json", fit_mapping)
    prefix_audit = dense4k.PrefixAudit()
    trajectories, metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    dense_windows = dense4k.dense_windows(
        trajectories, dense4k.CANDIDATE_STRIDE)
    if loaded_window_keys(dense_windows, metadata) != fit_keys:
        raise RuntimeError("loaded dense windows differ from pre-H5 key seal")
    sufficient = smu1.window_sufficient_statistics(
        trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    normalizer_path = args.output_dir / "point_normalizer.pt"
    torch.save(tuple(value.detach().cpu() for value in statistics),
               normalizer_path)
    normalizer_report = ft12.normalizer_payload(normalizer)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, dense_windows)
    schedule = smu1.paired_batch_schedule(
        range(len(dense_windows)), dense_windows,
        updates=UPDATES, seed=small4k.SEED)
    schedule["updates"] = UPDATES
    if schedule["sha256"] != EXPECTED_SAMPLE_STREAM_SHA256:
        raise RuntimeError("mature Real4k paired sample stream changed")

    registry = rde.EndpointRegistry(ARM_ORDER)
    arms: dict[str, dict[str, object]] = {}
    common_initial = None
    common_update0 = None
    base_config = None
    for arm in ARM_ORDER:
        report, initial, update0, current_base = train_arm(
            arm, args, dataset, schedule, normalizer, fit_mapping,
            registry, common_initial, common_update0, deadline)
        common_initial = initial
        common_update0 = update0
        if base_config is None:
            base_config = current_base
        elif current_base != base_config:
            raise RuntimeError("paired HCOB-F base configs differ")
        arms[arm] = report
    if not registry.complete():
        raise RuntimeError("four HCOB-F raw/EMA endpoint seals are incomplete")
    if (arms["aligned"]["model_config_sha256"]
            != arms["fixed_permuted"]["model_config_sha256"]
            or arms["aligned"]["parameter_manifest_sha256"]
            != arms["fixed_permuted"]["parameter_manifest_sha256"]):
        raise RuntimeError("paired HCOB-F compute topology differs")

    del dataset, trajectories
    gc.collect()
    tail_audit = rde.TailAudit(registry)
    evaluation_manifest_sha256 = dr1.sha256(args.evaluation_manifest)
    if evaluation_manifest_sha256 != EXPECTED_EVALUATION_MANIFEST_SHA256:
        raise RuntimeError("fixed clean716 evaluation manifest SHA changed")
    evaluation_rows = dense4k.load_evaluation_rows(
        args.evaluation_manifest, rows)
    # The independent tail mapping is sealed after all endpoints and after the
    # tail definition becomes visible, but before any held H5 tensor is read.
    tail_keys = tail_key_inventory(evaluation_rows)
    tail_mapping = permutation_record(tail_keys)
    if len(tail_keys) != EXPECTED_TAIL_WINDOWS:
        raise RuntimeError("clean716 key inventory changed")
    dr1.write_json_atomic(
        args.output_dir / "tail_mapping_seal.json", tail_mapping)
    tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", tail_audit)
    tail_audit.after_load()
    if loaded_window_keys(tail_windows, tail_metadata) != tail_keys:
        raise RuntimeError("loaded clean716 windows differ from tail key seal")
    tail_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)

    common_target_audit = None
    common_window_key_sha = None
    for arm in ARM_ORDER:
        arms[arm]["endpoints"] = {}
        for kind in ENDPOINT_ORDER:
            model = load_endpoint(
                registry.values[arm][kind], arm, normalizer, device,
                base_config)
            evaluations = {}
            for condition_mode in CONDITION_MODES:
                current = evaluate_endpoint(
                    model, tail_dataset, tail_windows, tail_metadata,
                    tail_mapping["mapping"], normalizer, device, deadline,
                    condition_mode)
                if common_target_audit is None:
                    common_target_audit = current["target_audit"]
                    common_window_key_sha = current[
                        "ordered_window_key_sha256"]
                elif (current["target_audit"] != common_target_audit
                      or current["ordered_window_key_sha256"]
                      != common_window_key_sha):
                    raise RuntimeError(
                        "endpoint/override target or order audit changed")
                evaluations[condition_mode] = current
            primary_mode = ("native" if arm == "aligned"
                            else "fixed_permuted")
            arms[arm]["endpoints"][kind] = {
                "seal": registry.values[arm][kind],
                "primary_condition_mode": primary_mode,
                "primary": evaluations[primary_mode],
                "condition_overrides": evaluations,
                "fixed_permuted_minus_native": rde.metric_comparison(
                    evaluations["fixed_permuted"]["metrics"],
                    evaluations["native"]["metrics"]),
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()

    historical_xcb = load_historical_xcb_ema_reference(
        args.historical_xcb_result,
        sample_stream_sha256=str(schedule["sha256"]),
        normalizer_statistics_sha256=smu1.statistics_digest(statistics),
        target_audit=common_target_audit)
    comparisons = {}
    for kind in ENDPOINT_ORDER:
        aligned = arms["aligned"]["endpoints"][kind]["primary"]["metrics"]
        permuted = arms["fixed_permuted"]["endpoints"][kind]["primary"][
            "metrics"]
        comparisons[kind] = {
            "aligned_minus_fixed_permuted": rde.metric_comparison(
                aligned, permuted),
            "aligned_minus_historical_xcb": (
                None if kind == "raw" or historical_xcb is None else
                rde.metric_comparison(
                    aligned, historical_xcb["metrics"])),
            "historical_xcb_reason": (
                "historical run retained no raw endpoint" if kind == "raw"
                else ("no --historical-xcb-result was supplied"
                      if historical_xcb is None else None)),
        }

    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "git": git,
        "start": {
            "checkpoint": str(args.start_sim.resolve()),
            "checkpoint_sha256": EXPECTED_START_SHA256,
            "checkpoint_model_config_sha256": arms["aligned"]["start"][
                "checkpoint_model_config_sha256"],
            "checkpoint_model_state_sha256": arms["aligned"]["start"][
                "checkpoint_model_state_sha256"],
            "checkpoint_normalizer_sha256": arms["aligned"]["start"][
                "checkpoint_normalizer_sha256"],
            "released_data_only_xcb_member_standard24k": True,
            "no_legacy_baseline_retraining_arm": True,
        },
        "manifest": {
            **manifest,
            "file_path": str(args.clean_manifest.resolve()),
            "file_sha256": EXPECTED_MANIFEST_SHA256,
        },
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": evaluation_manifest_sha256,
            "windows": len(tail_windows),
        },
        "normalizer": {
            "kind": "sparse2544 clean81 prefix only",
            "values_sha256": normalizer_report["sha256"],
            "statistics_sha256": smu1.statistics_digest(statistics),
            "artifact_path": str(normalizer_path.resolve()),
            "artifact_sha256": dr1.sha256(normalizer_path),
        },
        "mappings": {"fit": fit_mapping, "tail": tail_mapping},
        "historical_xcb_ema_context": historical_xcb,
        "training": {
            "windows": len(dense_windows),
            "window_stride": dense4k.CANDIDATE_STRIDE,
            "sample_stream_sha256_both_arms": schedule["sha256"],
            "augmentation_both_arms": dict(AUGMENTATION),
            "updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "seed": small4k.SEED,
            "optimizer": "AdamW",
            "lr": small4k.LR,
            "weight_decay": small4k.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": SCHEDULER_T_MAX,
            "ema_decay": EMA_DECAY,
            "native_joint_loss_unchanged": True,
            "condition_history_is_unaugmented_same_normalizer": True,
            "wall_budget_seconds": args.wall_budget_seconds,
            "formal_default_has_no_deadline": args.wall_budget_seconds is None,
        },
        "compute_identity": {
            "parameters_each": EXPECTED_PARAMETERS,
            "hcobf_parameters_each": EXPECTED_HCOBF_PARAMETERS,
            "model_config_sha256_both_arms": arms["aligned"][
                "model_config_sha256"],
            "parameter_manifest_sha256_both_arms": arms["aligned"][
                "parameter_manifest_sha256"],
            "hcobf_operator_scales": 3,
            "operator_types_each_scale": ["pointwise", "local5x5", "global4x4"],
            "equal_flops_and_forward_path": True,
            "only_difference": "source index of explicit history condition",
        },
        "paired_initial_state_sha256": common_initial,
        "paired_update0_five_output_sha256": common_update0,
        "arms": arms,
        "comparisons": comparisons,
        "target_and_order_audit_all_evaluations": {
            "target": common_target_audit,
            "ordered_window_key_sha256": common_window_key_sha,
            "identical": True,
        },
        "access_audit": {
            "fit_mapping_sealed_before_any_real_h5": True,
            "fit_prefix_opened": prefix_audit.opened,
            "all_four_endpoint_files_sha256_sealed_before_clean716": True,
            "clean716_manifest_read_after_all_four_seals": True,
            "tail_mapping_sealed_before_tail_h5": True,
            "clean716_opened": tail_audit.opened,
            "clean716_target_read_before_all_seals": False,
        },
        "inference_contract": {
            "aligned_deployment_reads_current_history_only": True,
            "metadata_aoa_re_filename_target_other_samples_read_by_model": False,
            "fixed_permuted_is_non_deployable_negative_control": True,
            "fixed_permuted_must_not_be_packaged": True,
        },
        "reported_diagnostics": {
            "official_point_and_native_sps": True,
            "mean_fluctuation_and_horizon": True,
            "aoa": True,
            "gate_entropy_std_usage": True,
            "region_frequency": False,
            "scope_reason": "kept formal runner minimal; no target-derived tuning",
        },
        "decision": "HCOBF_REAL4K_COMPLETE_NO_SCORE_GATE",
        "score_gate": None,
        "endpoint_selected_by_result": False,
        "submission_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    print(json.dumps({
        "event": "b210_hcobf_real4k_complete_no_gate",
        "comparisons": comparisons,
        "result": str(args.output_dir / "result.json"),
    }, sort_keys=True), flush=True)
    return result


def smoke(args: argparse.Namespace) -> dict[str, object]:
    if args.start_sim is None:
        raise ValueError("--smoke requires --start-sim")
    if dr1.sha256(args.start_sim) != EXPECTED_START_SHA256:
        raise RuntimeError("smoke start checkpoint SHA changed")
    payload = torch.load(
        str(args.start_sim), map_location="cpu", weights_only=False, mmap=True)
    base, state, source = parse_start_payload(payload)
    reports = {}
    initial_sha = None
    for arm in ARM_ORDER:
        model = upgrade_released_start(base, state, arm)
        current = dr1.model_state_sha256(model.state_dict())
        if initial_sha is None:
            initial_sha = current
        elif current != initial_sha:
            raise RuntimeError("smoke paired initial state differs")
        restored = EnsembleUNet2d(arm_config(base, arm)).float()
        restored.load_state_dict(model.state_dict(), strict=True)
        reports[arm] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "state_keys": len(model.state_dict()),
            "strict_roundtrip": True,
        }
        if reports[arm]["parameters"] != EXPECTED_PARAMETERS:
            raise RuntimeError("smoke parameter contract changed")
        del model, restored
        gc.collect()
    result = {
        "smoke": "PASS",
        "source": source,
        "arms": reports,
        "paired_initial_state_sha256": initial_sha,
        "gpu_used": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    if args.dry_contract:
        print(json.dumps(dry_contract(), sort_keys=True))
    elif args.smoke:
        smoke(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
