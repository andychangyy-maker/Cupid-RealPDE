#!/usr/bin/env python3
"""Run the frozen B210 small4k held-AoA0 confirmation.

The held-AoA10 update-4000 result is discovery evidence only.  This runner
repeats the first 4000 updates of the already sealed FT12 recipe on the
independent held-AoA0 fold, fixes EMA999 at update 4000 as the sole primary,
and compares it with the SHA-frozen MEF2 ``control_ema`` anchor.  It does not
pool folds, build a release checkpoint, produce a ZIP, or authorize submission.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import time
from typing import Callable, ContextManager, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import compare_b210_np_to_frozen_c208 as frozen_compare
import fp32_ema_shadow as ema_lib
import train_b210_np_functional_real_ft12_loao as ft12
import train_c208_mef2_paired_joint_loao as mef2
from models.ensemble_unet2d import EnsembleUNet2d


dr1 = ft12.dr1
npt1 = ft12.npt1
smu1 = ft12.smu1

EXPERIMENT = "b210_np_functional_real_small4k_confirm_held0"
SCHEMA = "b210_np_functional_real_small4k_confirm_held0_v1"
ENDPOINT_SCHEMA = "b210_np_functional_real_small4k_endpoint_v1"
FORMAL_BRANCH = ft12.FORMAL_BRANCH
FORMAL_ACKNOWLEDGEMENT = "B210_NP_REAL_SMALL4K_CONFIRM_HELD0"

HELD_AOA = 0
PRIMARY_ENDPOINT = "ema_u4000"
UPDATES = 4_000
BATCH_SIZE = 8
SEED = 4
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 12_000
EMA_DECAY = 0.999
WALL_BUDGET_SECONDS = 60 * 60
MIN_FREE_BYTES = 3 * 1024 ** 3

DISCOVERY_SHA256 = (
    "4a5d533f5869618dde72d456ccc4c175a179b8218a7b4c6e5b652077e818736a")
ANCHOR_SHA256 = (
    "e4eb94eb2cf6a012a0afd468c148dedc5d0d4b5edd1bf6cdb49ac9d448b3c409")
DISCOVERY_DECISION = "STOP_B210_REAL_FT12_HELD10_CATASTROPHIC_INCREMENT"

CONFIRMATION_THRESHOLDS = {
    "rel_l2_score": 0.0,
    "tke_rel_l2_score": 0.20,
    "mvpe_rel_l2_score": 0.0,
    "native_sps_score": 0.30,
    "non_time_delta": 0.10,
}
DISCOVERY_EFFECT = {
    "rel_l2_score": 0.0652523,
    "tke_rel_l2_score": 0.2921284,
    "mvpe_rel_l2_score": 0.1069650,
    "native_sps_score": 0.5278943,
    "non_time_delta": 0.1963353,
}

CONTRACT = {
    "hypothesis": (
        "The post-hoc held10 EMA-u4000 discovery repeats on independent "
        "held0 when the exact first 4000 updates of the sealed FT12 recipe "
        "are rerun from the same Sim10k artifact."),
    "evidence_roles": {
        "held10": "discovery_only_post_hoc_not_confirmation",
        "held0": "single_independent_confirmation",
        "pooled": False,
        "official": False,
        "submission": False,
    },
    "discovery": {
        "fold_result_sha256": DISCOVERY_SHA256,
        "held_aoa": 10,
        "selected_before_confirmation": PRIMARY_ENDPOINT,
        "observed_relative_to_frozen_mef2_control_ema": DISCOVERY_EFFECT,
        "u12000_decision": DISCOVERY_DECISION,
    },
    "confirmation": {
        "held_aoa": HELD_AOA,
        "anchor_endpoint": "control_ema",
        "anchor_sha256": ANCHOR_SHA256,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "thresholds": CONFIRMATION_THRESHOLDS,
        "endpoint_selection_allowed": False,
    },
    "training": {
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "grad_clip": GRAD_CLIP,
        "backbone_residual_dropout": 0.20,
        "native_losses": "Real joint point/member/quantile/soft-SPS",
        "tke_aux": "controller-only weight .01 cosine u1-u1500",
        "ema": "FP32 EMA999 continuous u0-u4000",
    },
    "data": {
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "normalizer": "other-four fit prefix only",
        "excluded": dr1.EXCLUDED_BASENAME,
        "tail_arrays_read": 0,
    },
    "hard_stop": [
        "discovery_or_anchor_sha", "sim_sha_or_config", "manifest",
        "branch_or_hardware", "oom_or_nonfinite", "ema_or_held_leak",
        "gradient_preflight", "wall_budget_60min", "any_gate_failure",
    ],
    "release_checkpoint_written": False,
    "submission_authorized": False,
}
CONTRACT_SHA256 = dr1.canonical_sha256(CONTRACT)


class EndpointAccessAudit:
    """Keep held0 prefix reads behind the single frozen endpoint seal."""

    def __init__(self) -> None:
        self.seal_payload: dict | None = None
        self.events: list[dict] = []
        self.opened: set[str] = set()
        self.evaluated = False

    @property
    def sealed(self) -> bool:
        return self.seal_payload is not None

    def seal(self, value: Mapping[str, object]) -> None:
        if self.sealed or value.get("endpoint") != PRIMARY_ENDPOINT:
            raise RuntimeError("invalid or duplicate small4k endpoint seal")
        self.seal_payload = {
            "path": str(value["path"]),
            "file_sha256": dr1.normalized_sha256(
                str(value["file_sha256"]), "small4k endpoint file SHA"),
            "model_state_sha256": dr1.normalized_sha256(
                str(value["model_state_sha256"]), "small4k state SHA"),
            "sealed_monotonic_ns": time.monotonic_ns(),
        }

    def before_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        aoa = int(row["aoa"])
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded H5 reached small4k opener")
        if phase == "fit":
            if aoa == HELD_AOA or self.sealed:
                raise RuntimeError("invalid small4k fit-prefix open")
        elif phase == "held":
            if aoa != HELD_AOA or not self.sealed:
                raise RuntimeError("held0 prefix opened before endpoint seal")
        else:
            raise ValueError(f"unknown small4k access phase {phase}")

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
            "endpoint_sealed_before_open": self.sealed,
        })

    def record_evaluation(self) -> None:
        if not self.sealed or self.evaluated:
            raise RuntimeError("invalid small4k held evaluation")
        self.evaluated = True

    def report(self, expected_allowlist: Sequence[str]) -> dict:
        held_events = [row for row in self.events if row["phase"] == "held"]
        return {
            "opened_file_allowlist": sorted(self.opened),
            "expected_file_allowlist": sorted(expected_allowlist),
            "allowlist_exact": sorted(self.opened) == sorted(expected_allowlist),
            "tail_arrays_read": 0,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "endpoint_seal": self.seal_payload,
            "held_first_open_after_endpoint_seal": bool(
                held_events and held_events[0]["endpoint_sealed_before_open"]),
            "held_complete_passes": int(self.evaluated),
            "events": self.events,
        }


def _read_json_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON evidence must be a mapping: {path}")
    return value


def _validated_sha_file(path: Path, expected: str, label: str) -> dict:
    actual = dr1.sha256(path)
    if actual != dr1.normalized_sha256(expected, label):
        raise RuntimeError(f"{label} changed: {path}")
    return {"path": str(path.resolve()), "sha256": actual}


def validate_discovery_row(row: Mapping[str, object]) -> dict:
    """Validate the bound held10 result without promoting its post-hoc u4."""
    if (row.get("schema") != ft12.SCHEMA
            or row.get("experiment") != ft12.EXPERIMENT
            or row.get("contract_sha256") != ft12.CONTRACT_SHA256
            or int(row.get("held_aoa", -1)) != 10):
        raise RuntimeError("held10 discovery identity changed")
    if (row.get("primary_endpoint") != ft12.PRIMARY_ENDPOINT
            or row.get("decision") != DISCOVERY_DECISION
            or row.get("diagnostics_cannot_select_endpoint") is not True):
        raise RuntimeError("held10 discovery/STOP policy changed")
    samples = int(row.get("samples", -1))
    if samples != dr1.EXPECTED_AOA_COUNTS[10]:
        raise RuntimeError("held10 discovery sample count changed")

    u4_state, u4_metrics, u4_audit = frozen_compare._endpoint_payload(
        row, PRIMARY_ENDPOINT, "held10 discovery ema_u4000")
    u12_state, u12_metrics, u12_audit = frozen_compare._endpoint_payload(
        row, ft12.PRIMARY_ENDPOINT, "held10 stopped ema_u12000")
    if u4_audit != u12_audit:
        raise RuntimeError("held10 discovery endpoint target audits differ")
    primary = row.get("primary", {})
    if (primary.get("endpoint") != ft12.PRIMARY_ENDPOINT
            or primary.get("metrics") != u12_metrics
            or primary.get("metric_state") != u12_state):
        raise RuntimeError("held10 u12000 primary evidence changed")
    increment = ft12.metric_comparison(u12_metrics, u4_metrics)
    gate = ft12.fold10_disaster_gate(increment)
    if (row.get("ft12_increment_diagnostic") != increment
            or row.get("fold10_disaster_gate") != gate
            or gate["pass"] is not False):
        raise RuntimeError("held10 u12000 STOP is not reproducible")

    training = row.get("training", {})
    recipe = {
        "updates": 12_000,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "scheduler_t_max": SCHEDULER_T_MAX,
        "ema_decay": EMA_DECAY,
    }
    if any(training.get(key) != value for key, value in recipe.items()):
        raise RuntimeError("held10 FT12 training recipe changed")
    if (int(training.get("ema_updates", -1)) != 12_000
            or training.get("u4000_stream_sha256")
            != row["endpoints"][PRIMARY_ENDPOINT]["seal"].get(
                "sample_stream_sha256")):
        raise RuntimeError("held10 u4000 stream/EMA evidence changed")
    start_sha = dr1.normalized_sha256(
        str(row.get("start", {}).get("checkpoint_sha256")),
        "held10 discovery Sim SHA")
    return {
        "kind": "sha_sealed_held10_discovery_only_post_hoc",
        "held_aoa": 10,
        "decision": DISCOVERY_DECISION,
        "start_checkpoint_sha256": start_sha,
        "u4000_metrics": u4_metrics,
        "u4000_metric_state_sha256": dr1.canonical_sha256(u4_state),
        "u4000_target_audit": u4_audit,
        "u12000_metrics": u12_metrics,
        "u12000_metric_state_sha256": dr1.canonical_sha256(u12_state),
        "confirmation_role": False,
    }


def load_discovery(path: Path) -> dict:
    source = _validated_sha_file(path, DISCOVERY_SHA256, "discovery SHA")
    return {**source, **validate_discovery_row(_read_json_mapping(path))}


def confirmation_gate(comparison: Mapping[str, object]) -> dict:
    delta = comparison.get("delta", {})
    conditions = {
        "rel_l2_nonnegative": (
            float(delta["rel_l2_score"])
            >= CONFIRMATION_THRESHOLDS["rel_l2_score"]),
        "tke_at_least_plus_0p20": (
            float(delta["tke_rel_l2_score"])
            >= CONFIRMATION_THRESHOLDS["tke_rel_l2_score"]),
        "mvpe_nonnegative": (
            float(delta["mvpe_rel_l2_score"])
            >= CONFIRMATION_THRESHOLDS["mvpe_rel_l2_score"]),
        "sps_at_least_plus_0p30": (
            float(delta["native_sps_score"])
            >= CONFIRMATION_THRESHOLDS["native_sps_score"]),
        "non_time_at_least_plus_0p10": (
            float(comparison["non_time_delta"])
            >= CONFIRMATION_THRESHOLDS["non_time_delta"]),
    }
    passed = all(conditions.values())
    return {
        "pass": passed,
        "thresholds": dict(CONFIRMATION_THRESHOLDS),
        "conditions": conditions,
        "decision": (
            "PASS_B210_SMALL4K_HELD0_INDEPENDENT_CONFIRMATION"
            if passed else "STOP_B210_SMALL4K_HELD0_CONFIRMATION_GATE"),
    }


def save_primary_endpoint(
        path: Path, model: EnsembleUNet2d, ema: ema_lib.FP32EMAShadow,
        *, start_report: Mapping[str, object],
        normalizer_report: Mapping[str, object], stream_sha256: str,
        ) -> tuple[dict, dict]:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite endpoint {path}")
    raw_before = dr1.model_state_sha256(model.state_dict())
    ema_payload = ema.seal(model)
    raw_after = dr1.model_state_sha256(model.state_dict())
    if raw_before != raw_after:
        raise RuntimeError("materializing small4k EMA mutated raw model")
    state = ema_payload.pop("model_state")
    if (int(ema_payload["updates"]) != UPDATES
            or float(ema_payload["decay"]) != EMA_DECAY
            or ema_lib.tensor_state_sha256(state)
            != ema_payload["materialized_model_state_sha256"]):
        raise RuntimeError("small4k EMA seal contract changed")
    cpu_state = OrderedDict(
        (name, value.detach().cpu()) for name, value in state.items())
    state_sha = dr1.model_state_sha256(cpu_state)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": UPDATES,
        "endpoint": PRIMARY_ENDPOINT,
        "primary": True,
        "held_aoa": HELD_AOA,
        "model_state_dict": cpu_state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start": dict(start_report),
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": str(normalizer_report["sha256"]),
        "sample_stream_sha256": stream_sha256,
        "ema": dict(ema_payload),
        "optimizer_state_saved": False,
        "held_prefix_opened_before_seal": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    seal = {
        "endpoint": PRIMARY_ENDPOINT,
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": dr1.canonical_sha256(payload["model_config"]),
        "sample_stream_sha256": stream_sha256,
        "iteration": UPDATES,
        "primary": True,
        "optimizer_state_saved": False,
    }
    del state, cpu_state
    return seal, {
        **ema_payload,
        "raw_model_unchanged_by_seal": raw_before == raw_after,
    }


def load_primary_endpoint(
        seal: Mapping[str, object], normalizer: smu1.FoldNormalizer,
        device: torch.device, *,
        expected_model_config=None,
        expected_parameters: int | None = None,
        expected_controller_parameters: int | None = None,
        ) -> EnsembleUNet2d:
    path = Path(str(seal["path"]))
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("small4k endpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment") != EXPERIMENT
            or payload.get("endpoint") != PRIMARY_ENDPOINT
            or payload.get("primary") is not True
            or int(payload.get("iteration", -1)) != UPDATES
            or int(payload.get("held_aoa", -1)) != HELD_AOA
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("small4k endpoint payload changed")
    config = ft12.validate_b210_model_config(
        payload["model_config"], expected_model_config)
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    total = sum(parameter.numel() for parameter in model.parameters())
    controller = sum(
        parameter.numel() for parameter in model.energy_controller.parameters())
    expected_total = (
        ft12.EXPECTED_PARAMETERS
        if expected_parameters is None else expected_parameters)
    expected_controller = (
        ft12.EXPECTED_CONTROLLER_PARAMETERS
        if expected_controller_parameters is None
        else expected_controller_parameters)
    if total != expected_total or controller != expected_controller:
        raise RuntimeError("small4k endpoint parameter contract changed")
    npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def validate_ema_snapshot_contract(
        updates: Sequence[int], sink: Callable | None,
        terminal_update: int,
        ) -> tuple[int, ...]:
    """Validate optional in-memory EMA diagnostics without selecting a model."""
    values = tuple(int(value) for value in updates)
    if values != tuple(sorted(set(values))):
        raise ValueError("EMA snapshot updates must be unique and ordered")
    if any(value <= 0 or value >= terminal_update for value in values):
        raise ValueError("EMA snapshots must be intermediate training updates")
    if bool(values) != (sink is not None):
        raise ValueError("EMA snapshot updates and sink must be supplied together")
    return values


def batch_forward_context(
        callback: Callable[
            [int, EnsembleUNet2d, Sequence[torch.Tensor],
             smu1.FoldNormalizer, torch.device], ContextManager[None]] | None,
        iteration: int, model: EnsembleUNet2d,
        batch: Sequence[torch.Tensor], normalizer: smu1.FoldNormalizer,
        device: torch.device) -> ContextManager[None]:
    """Resolve an optional per-batch model context without touching RNG.

    The ``None`` route is the historical trainer path and returns only a
    no-op context.  Specialized runners may scope transient forward state
    around the complete native forward/backward call.
    """
    return (nullcontext() if callback is None
            else callback(iteration, model, batch, normalizer, device))


def validate_parameter_groups(trainable, parameter_groups):
    """Every trainable tensor must occur exactly once in named LR groups."""
    if parameter_groups is None:
        return trainable
    groups = [{**group, "params": tuple(group["params"])}
              for group in parameter_groups]
    names = [group.get("name") for group in groups]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    if (not groups or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or any(not group["params"] for group in groups)
            or len(ids) != len(set(ids))
            or set(ids) != {id(parameter) for parameter in trainable}
            or any(not math.isfinite(float(group.get("lr", LR)))
                   or float(group.get("lr", LR)) <= 0 for group in groups)):
        raise ValueError("parameter groups must name and cover all trainable tensors exactly once")
    return groups


def controller_tke_aux_weight(iteration: int, constant_controller_tke: bool = False) -> float:
    """Arm E: controller-only TKE objective at 0.010 for every Real update."""
    return .010


def train_candidate(
        model: EnsembleUNet2d, dataset, schedule: Mapping[str, object],
        normalizer: smu1.FoldNormalizer, device: torch.device,
        deadline: float, output_dir: Path, start_report: Mapping[str, object],
        access: EndpointAccessAudit,
        *, input_augmentation: Mapping[str, float] | None = None,
        discard_input_corruption: bool = False,
        sam_rho: float = 0.0,
        endpoint_filename: str = "ema999_u4000_primary.pth",
        full_backbone_auxiliary: torch.nn.Module | None = None,
        full_backbone_auxiliary_weight: float = 0.0,
        full_backbone_gradient_audit_update: int = 0,
        full_backbone_gradient_ratio_range: tuple[float, float] | None = None,
        gradient_ratio_diagnostics_only: bool = False,
        expected_parameters: int | None = None,
        extra_backward: Callable[
            [int, EnsembleUNet2d], Mapping[str, float]] | None = None,
        extra_backward_contract: Mapping[str, object] | None = None,
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
        scheduler_t_max: int = SCHEDULER_T_MAX,
        constant_controller_tke: bool = False,
        parameter_groups: Sequence[dict] | None = None,
        ema_snapshot_updates: Sequence[int] = (),
        ema_snapshot_sink: Callable[
            [int, Mapping[str, torch.Tensor]], None] | None = None,
        forward_context: Callable[
            [int, EnsembleUNet2d, Sequence[torch.Tensor],
             smu1.FoldNormalizer, torch.device], ContextManager[None]
        ] | None = None,
        expected_member_count: int = 4,
        ) -> tuple[dict, dict]:
    if model.config.ensemble_members != expected_member_count:
        raise ValueError("small4k model and training member counts differ")
    if scheduler_t_max < UPDATES:
        raise ValueError(
            "small4k scheduler_t_max cannot be shorter than the endpoint")
    snapshot_updates = validate_ema_snapshot_contract(
        ema_snapshot_updates, ema_snapshot_sink, UPDATES)
    trainable = ft12.configure_trainable(model, expected_parameters)
    if full_backbone_auxiliary is not None:
        trainable = trainable + tuple(full_backbone_auxiliary.parameters())
    tagged = npt1.ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_sampler=schedule["batches"], num_workers=0,
        pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        validate_parameter_groups(trainable, parameter_groups),
        lr=LR, weight_decay=WEIGHT_DECAY)
    initial_group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max, eta_min=0.0)
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    if [row["name"] for row in ema.parameter_manifest()] != [
            name for name, _ in model.named_parameters()]:
        raise RuntimeError("small4k EMA does not cover all parameters")

    initial_controller = npt1.controller_state(model)
    initial_controller_sha = ema_lib.tensor_state_sha256(initial_controller)
    initial_front_sha = ema_lib.tensor_state_sha256(
        npt1._controller_front_state(initial_controller))
    actual_rows: list[dict] = []
    ratios: list[float] = []
    history: list[dict] = []
    preflight = None
    projection_gradient_u1 = False
    front_gradient_u2 = False
    projection_updated_u1 = False
    legacy_gradient_seen = False
    legacy_parameter_updated = False
    full_backbone_gradient_audit = None
    legacy_tke_floor_rows: list[dict[str, float | bool]] = []
    legacy_tke_floor_preflight = None
    started = time.perf_counter()
    backward_real = npt1.backward_real_supervised_npt1
    if sam_rho:
        from functools import partial
        from matched_sam_backward import matched_sam_backward
        backward_real = partial(matched_sam_backward, backward_real, rho=sam_rho)

    for iteration, batch in enumerate(loader, start=1):
        if iteration > UPDATES:
            break
        if time.perf_counter() > deadline:
            raise TimeoutError("B210 small4k exceeded 60-minute budget")
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            raise RuntimeError("small4k training batch is not exact tuple4")
        history_value, target, trajectory_ids, sample_indices = batch
        expected = schedule["rows"][iteration - 1]
        actual = {
            "sample_indices": sample_indices.to(dtype=torch.long).tolist(),
            "trajectory_ids": trajectory_ids.to(dtype=torch.long).tolist(),
            "bootstrap_mask": expected["bootstrap_mask"],
        }
        if actual != expected:
            raise RuntimeError(f"small4k stream mismatch at update {iteration}")
        actual_rows.append(actual)
        member_mask = torch.as_tensor(expected["bootstrap_mask"], dtype=torch.bool)
        if (member_mask.device.type != "cpu"
                or member_mask.shape != (len(sample_indices), expected_member_count)
                or not bool(member_mask.any(dim=1).all())):
            raise RuntimeError("small4k frozen member mask changed")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_value = (history_value, target, trajectory_ids, sample_indices)
        context = batch_forward_context(
            forward_context, iteration, model, batch_value,
            normalizer, device)
        with context:
            values = backward_real(
                model, batch_value, normalizer, device, iteration,
                auxiliary_weight=controller_tke_aux_weight(iteration, constant_controller_tke),
                member_mask=member_mask,
                audit_controller_gradients=(
                    iteration <= ft12.GRADIENT_PREFLIGHT_UPDATES),
                input_augmentation=input_augmentation,
                discard_input_corruption=discard_input_corruption,
                full_backbone_auxiliary=full_backbone_auxiliary,
                full_backbone_auxiliary_weight=full_backbone_auxiliary_weight,
                full_backbone_gradient_audit_update=(
                full_backbone_gradient_audit_update),
                point_accuracy_weights=point_accuracy_weights,
                member_accuracy_weights=member_accuracy_weights,
                legacy_tke_gradient_floor=legacy_tke_gradient_floor,
                legacy_tke_gradient_floor_on_members=(
                    legacy_tke_gradient_floor_on_members),
                legacy_point_accuracy_weights=legacy_point_accuracy_weights,
                legacy_member_accuracy_weights=(
                    legacy_member_accuracy_weights),
                input_region_augmentor=input_region_augmentor,
                region_auxiliary=region_auxiliary,
                region_auxiliary_weight=region_auxiliary_weight)
        if legacy_tke_gradient_floor and iteration <= 200:
            legacy_tke_floor_rows.append({
                key: values[key] for key in (
                    "legacy_tke_floor_finite",
                    "legacy_tke_floor_applied_fraction",
                    "legacy_tke_floor_correction_ratio_median",
                    "legacy_tke_floor_correction_ratio_max")
            })
        if extra_backward is not None:
            extra_values = dict(extra_backward(iteration, model))
            overlap = set(values).intersection(extra_values)
            if overlap:
                raise RuntimeError(
                    f"extra backward metric collision: {sorted(overlap)}")
            if any(not torch.isfinite(torch.as_tensor(value))
                   for value in extra_values.values()):
                raise FloatingPointError(
                    f"non-finite extra backward metric at update {iteration}")
            values.update(extra_values)
        audit = npt1._gradient_audit(model)
        if not audit["finite"]:
            raise FloatingPointError("small4k gradient is non-finite")
        legacy_gradient_seen |= audit["legacy_nonzero"]
        if iteration == 1:
            projection_gradient_u1 = audit["controller_projection_nonzero"]
            if not projection_gradient_u1:
                raise ft12.GradientPreflightStop(
                    "STOP_B210_SMALL4K_CONTROLLER_PROJECTION_GRAD_U1")
        if iteration == 2:
            front_gradient_u2 = audit["controller_front_nonzero"]
            if not front_gradient_u2:
                raise ft12.GradientPreflightStop(
                    "STOP_B210_SMALL4K_CONTROLLER_FRONT_GRAD_U2")
        if iteration <= ft12.GRADIENT_PREFLIGHT_UPDATES:
            ratios.append(float(values[
                "npt1_tke_aux_to_native_gradient_ratio"]))
        if (full_backbone_auxiliary is not None
                and iteration == full_backbone_gradient_audit_update):
            ratio = float(values[
                "full_backbone_aux_to_native_feature_gradient_ratio"])
            finite_nonzero = bool(values[
                "full_backbone_gradient_finite_nonzero"])
            full_backbone_gradient_audit = {
                "iteration": iteration,
                "native_feature_gradient_norm": float(values[
                    "full_backbone_native_feature_gradient_norm"]),
                "auxiliary_feature_gradient_norm": float(values[
                    "full_backbone_aux_feature_gradient_norm"]),
                "auxiliary_to_native_feature_gradient_ratio": ratio,
                "finite_nonzero": finite_nonzero,
                "ratio_range": (
                    None if full_backbone_gradient_ratio_range is None
                    else list(full_backbone_gradient_ratio_range)),
            }
            if (not finite_nonzero
                    or (not gradient_ratio_diagnostics_only and (
                        full_backbone_gradient_ratio_range is None
                        or not full_backbone_gradient_ratio_range[0] <= ratio
                        <= full_backbone_gradient_ratio_range[1]))):
                raise ft12.GradientPreflightStop(
                    "STOP_B210_FULL_BACKBONE_AUXILIARY_GRADIENT "
                    f"ratio={ratio:.8g}")

        legacy_probe = None
        if not legacy_parameter_updated:
            for name, parameter in model.named_parameters():
                if (not name.startswith("energy_controller.")
                        and parameter.grad is not None
                        and bool(torch.count_nonzero(parameter.grad))):
                    legacy_probe = (parameter, parameter.detach().clone())
                    break
        projection_before = None
        if iteration == 1:
            projection_before = OrderedDict(
                (name, value.detach().clone())
                for name, value in model.energy_controller.state_dict().items()
                if name in {"body.5.weight", "body.5.bias"})
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("small4k clipped gradient is non-finite")
        optimizer.step()
        if legacy_probe is not None:
            parameter, before = legacy_probe
            legacy_parameter_updated |= not torch.equal(parameter.detach(), before)
        if projection_before is not None:
            after = model.energy_controller.state_dict()
            projection_updated_u1 = any(
                not torch.equal(value, after[name])
                for name, value in projection_before.items())
            if not projection_updated_u1:
                raise ft12.GradientPreflightStop(
                    "STOP_B210_SMALL4K_CONTROLLER_PROJECTION_UPDATE_U1")
        ema.update_after_optimizer_step(model, optimizer_step=iteration)
        if iteration in snapshot_updates:
            # Diagnostic callers retain these CPU clones in memory and may not
            # inspect a held target until every formal endpoint has been
            # sealed.  The default path supplies no sink and remains bitwise
            # identical to the historical trainer.
            ema_snapshot_sink(
                iteration, ema.materialize_model_state(model))
        scheduler.step()

        if iteration == ft12.GRADIENT_PREFLIGHT_UPDATES:
            preflight = ft12.ratio_diagnostic(
                ratios, allow_zero=gradient_ratio_diagnostics_only)
            if not gradient_ratio_diagnostics_only and not preflight["pass"]:
                raise ft12.GradientPreflightStop(
                    "STOP_B210_SMALL4K_GRADIENT_PREFLIGHT_RATIO "
                    f"median={preflight['weighted_aux_to_native_controller_grad_ratio_median']:.8g}")
            if legacy_tke_gradient_floor:
                activation = statistics.mean(float(row[
                    "legacy_tke_floor_applied_fraction"])
                    for row in legacy_tke_floor_rows)
                correction_ratio = statistics.median(float(row[
                    "legacy_tke_floor_correction_ratio_median"])
                    for row in legacy_tke_floor_rows)
                finite = bool(
                    len(legacy_tke_floor_rows) == 200
                    and all(bool(row["legacy_tke_floor_finite"])
                            for row in legacy_tke_floor_rows)
                    and math.isfinite(activation)
                    and math.isfinite(correction_ratio))
                passed = bool(
                    finite and activation >= 0.50
                    and 0.10 <= correction_ratio <= 5.00)
                legacy_tke_floor_preflight = {
                    "updates": len(legacy_tke_floor_rows),
                    "finite": finite,
                    "applied_fraction_mean": activation,
                    "correction_ratio_median": correction_ratio,
                    "correction_ratio_max": max(float(row[
                        "legacy_tke_floor_correction_ratio_max"])
                        for row in legacy_tke_floor_rows),
                    "pass": passed,
                }
                if not passed:
                    raise ft12.GradientPreflightStop(
                        "STOP_B210_REAL_LEGACY_TKE_FLOOR_PREFLIGHT "
                        f"activation={activation:.8g} "
                        f"correction_ratio={correction_ratio:.8g}")
        if iteration == 1 or iteration % 500 == 0:
            report = {
                "iteration": iteration,
                **{key: float(value) for key, value in values.items()},
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(report)
            print(json.dumps({
                "event": "b210_small4k_progress", **report,
            }), flush=True)

    if len(actual_rows) != UPDATES or ema.updates != UPDATES:
        raise RuntimeError("small4k did not complete exactly 4000 updates")
    actual_sha = dr1.canonical_sha256(actual_rows)
    if actual_sha != schedule["sha256"]:
        raise RuntimeError("small4k sample stream digest changed")
    if (preflight is None or not legacy_gradient_seen
            or not legacy_parameter_updated):
        raise RuntimeError("small4k gradient/update evidence is incomplete")
    final_controller = npt1.controller_state(model)
    final_controller_sha = ema_lib.tensor_state_sha256(final_controller)
    final_front_sha = ema_lib.tensor_state_sha256(
        npt1._controller_front_state(final_controller))
    if (final_controller_sha == initial_controller_sha
            or final_front_sha == initial_front_sha):
        raise RuntimeError("small4k controller did not train through its front")

    normalizer_report = ft12.normalizer_payload(normalizer)
    seal, ema_report = save_primary_endpoint(
        output_dir / endpoint_filename, model, ema,
        start_report=start_report, normalizer_report=normalizer_report,
        stream_sha256=schedule["sha256"])
    access.seal(seal)
    return ({
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "scheduler_t_max": scheduler_t_max,
        **({"controller_tke_schedule": "constant_0p01_4000"} if constant_controller_tke else {}),
        "parameter_groups": [{
            "name": group.get("name", "all"),
            "initial_lr": initial_group_lrs[index], "final_lr": float(group["lr"]),
            "tensor_count": len(group["params"]),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "parameter_names": group.get("param_names"),
        } for index, group in enumerate(optimizer.param_groups)],
        "ema_decay": EMA_DECAY,
        "input_augmentation": {
            key: float(value)
            for key, value in (input_augmentation or {}).items()
        },
        **({'input_augmentation_applied': False, 'augmentation_rng_draws_preserved': True}
           if discard_input_corruption else {}),
        **({'sam_rho': sam_rho, 'sam_matched_rng': True} if sam_rho else {}),
        "full_backbone_auxiliary": (
            full_backbone_auxiliary is not None),
        "full_backbone_auxiliary_weight": full_backbone_auxiliary_weight,
        "full_backbone_gradient_audit": full_backbone_gradient_audit,
        "gradient_ratio_diagnostics_only": gradient_ratio_diagnostics_only,
        "extra_backward": extra_backward is not None,
        "extra_backward_contract": (
            None if extra_backward_contract is None
            else dict(extra_backward_contract)),
        "point_accuracy_weights": list(
            npt1.joint_real.REAL_ACCURACY_WEIGHTS
            if point_accuracy_weights is None else point_accuracy_weights),
        "member_accuracy_weights": list(
            npt1.joint_real.REAL_ACCURACY_WEIGHTS
            if member_accuracy_weights is None else member_accuracy_weights),
        "legacy_tke_gradient_floor": legacy_tke_gradient_floor,
        "legacy_tke_gradient_floor_on_members": (
            legacy_tke_gradient_floor_on_members),
        "legacy_point_accuracy_weights": (
            None if legacy_point_accuracy_weights is None
            else list(legacy_point_accuracy_weights)),
        "legacy_member_accuracy_weights": (
            None if legacy_member_accuracy_weights is None
            else list(legacy_member_accuracy_weights)),
        "legacy_tke_floor_preflight": legacy_tke_floor_preflight,
        "ema_updates": ema.updates,
        "ema_snapshot_updates": list(snapshot_updates),
        "planned_stream_sha256": schedule["sha256"],
        "actual_stream_sha256": actual_sha,
        "gradient_preflight": preflight,
        "projection_gradient_u1": projection_gradient_u1,
        "front_gradient_u2": front_gradient_u2,
        "projection_updated_u1": projection_updated_u1,
        "legacy_gradient_seen": legacy_gradient_seen,
        "legacy_parameter_updated": legacy_parameter_updated,
        "initial_controller_state_sha256": initial_controller_sha,
        "final_controller_state_sha256": final_controller_sha,
        "initial_controller_front_state_sha256": initial_front_sha,
        "final_controller_front_state_sha256": final_front_sha,
        "ema_seal": ema_report,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }, seal)


def validate_anchor_row(candidate: Mapping[str, object],
                        anchor: Mapping[str, object]) -> tuple[dict, dict]:
    if (anchor.get("schema") != mef2.SCHEMA
            or anchor.get("experiment") != mef2.EXPERIMENT
            or int(anchor.get("held_aoa", -1)) != HELD_AOA
            or anchor.get("endpoint_order") != list(mef2.ENDPOINT_ORDER)):
        raise RuntimeError("frozen MEF2 held0 anchor identity changed")
    if int(anchor.get("samples", -1)) != int(candidate["samples"]):
        raise RuntimeError("candidate/anchor held0 sample counts differ")
    if anchor.get("manifest_file_sha256") != candidate.get(
            "manifest_file_sha256"):
        raise RuntimeError("candidate/anchor manifest file SHAs differ")
    for key in ("semantic_sha256", "windows", "excluded"):
        if anchor.get("manifest", {}).get(key) != candidate.get(
                "manifest", {}).get(key):
            raise RuntimeError(f"candidate/anchor manifest {key} differs")
    if anchor.get("normalizer", {}).get("sha256") != candidate.get(
            "normalizer", {}).get("sha256"):
        raise RuntimeError("candidate/anchor normalizer SHAs differ")
    if anchor.get("held_ordered_key_sha256") != candidate.get(
            "held_ordered_key_sha256"):
        raise RuntimeError("candidate/anchor held0 ordered keys differ")

    anchor_state, anchor_metrics, anchor_audit = frozen_compare._endpoint_payload(
        anchor, "control_ema", "frozen MEF2 held0 control_ema")
    candidate_report = candidate.get("endpoint", {})
    candidate_state = frozen_compare._validated_metric_state(
        candidate_report.get("metric_state"), int(candidate["samples"]),
        "B210 small4k held0 ema_u4000")
    candidate_metrics = smu1.finalize_metric_state(candidate_state)
    if candidate_report.get("metrics") != candidate_metrics:
        raise RuntimeError("small4k metrics are not finalized from state")
    candidate_audit = frozen_compare._validated_audit(
        candidate_report.get("held_target_audit"), int(candidate["samples"]),
        "B210 small4k held0 ema_u4000")
    if candidate_audit != anchor_audit:
        raise RuntimeError("candidate/anchor held0 target audits differ")
    return ({
        "metrics": candidate_metrics,
        "metric_state": candidate_state,
        "held_target_audit": candidate_audit,
    }, {
        "metrics": anchor_metrics,
        "metric_state": anchor_state,
        "held_target_audit": anchor_audit,
    })


def _disk_preflight(output_dir: Path) -> dict:
    probe = output_dir.parent.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError("no existing parent for small4k output")
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"small4k output disk has {free} bytes; requires {MIN_FREE_BYTES}")
    return {"probe": str(probe), "free_bytes_before_run": free,
            "minimum_free_bytes": MIN_FREE_BYTES}


def run(args: argparse.Namespace) -> dict:
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError("formal small4k confirmation is on the wrong branch")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    discovery = load_discovery(args.discovery_fold10)
    start_sha = dr1.normalized_sha256(args.start_sim_sha256, "start Sim SHA")
    if start_sha != discovery["start_checkpoint_sha256"]:
        raise RuntimeError("small4k Sim SHA differs from held10 discovery")
    anchor_source = _validated_sha_file(
        args.anchor_held0, ANCHOR_SHA256, "frozen held0 anchor SHA")
    device = torch.device(args.device)
    hardware = ft12.validate_hardware(device, args.hardware_class)
    disk = _disk_preflight(args.output_dir)
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("small4k clean2544 manifest file SHA changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != HELD_AOA]
    held_rows = [row for row in rows if int(row["aoa"]) == HELD_AOA]
    if sum(len(row["train_starts"]) for row in fit_rows) != (
            dr1.EXPECTED_FIT_COUNTS[HELD_AOA]):
        raise RuntimeError("small4k other-four fit count changed")
    if sum(len(row["train_starts"]) for row in held_rows) != (
            dr1.EXPECTED_AOA_COUNTS[HELD_AOA]):
        raise RuntimeError("small4k held0 count changed")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + WALL_BUDGET_SECONDS
    torch.set_float32_matmul_precision("high")
    access = EndpointAccessAudit()
    fit_trajectories, fit_metadata, fit_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", access)
    fit_dataset = npt1.ensemble_train.common.repro.WindowDataset(
        fit_trajectories, fit_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, fit_windows)
    statistics_values = smu1.fold_statistics(
        sufficient, range(len(fit_windows)))
    normalizer = smu1.FoldNormalizer(statistics_values, device)
    schedule = smu1.paired_batch_schedule(
        range(len(fit_windows)), fit_windows, updates=UPDATES, seed=SEED)
    schedule["updates"] = UPDATES

    npt1.ensemble_train.seed_everything(SEED)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device)
    training, seal = train_candidate(
        model, fit_dataset, schedule, normalizer, device, deadline,
        args.output_dir, start_report, access)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not access.sealed:
        raise RuntimeError("small4k held0 load began without endpoint seal")

    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", access)
    held_dataset = npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    access.record_evaluation()
    evaluated = load_primary_endpoint(seal, normalizer, device)
    metrics, metric_state, held_audit = smu1.evaluate_model(
        evaluated, held_dataset, list(range(len(held_windows))), normalizer,
        device, deadline)
    del evaluated
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    access_report = access.report([row["file"] for row in rows])
    if not (access_report["allowlist_exact"]
            and access_report["tail_arrays_read"] == 0
            and not access_report["excluded_opened"]
            and access_report["held_first_open_after_endpoint_seal"]
            and access_report["held_complete_passes"] == 1):
        raise RuntimeError("small4k prefix access contract failed")
    candidate_row = {
        "samples": len(held_windows),
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "normalizer": {
            **ft12.normalizer_payload(normalizer),
            "kind": "other-four fit prefix windows only",
            "held_target_in_normalizer": False,
        },
        "held_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(held_windows)), held_windows, held_metadata),
        "endpoint": {
            "seal": seal,
            "metrics": metrics,
            "metric_state": metric_state,
            "held_target_audit": held_audit,
        },
    }
    # Parse held0 anchor semantics only after the candidate endpoint is sealed
    # and held0 has been scored.  Its bytes were hash-checked before GPU work.
    anchor_row = _read_json_mapping(args.anchor_held0)
    candidate, anchor = validate_anchor_row(candidate_row, anchor_row)
    comparison = ft12.metric_comparison(
        candidate["metrics"], anchor["metrics"])
    gate = confirmation_gate(comparison)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "evidence_class": {
            "held10": "discovery_only_post_hoc_not_confirmation",
            "held0": "single_independent_confirmation",
            "pooled_official": False,
            "submission": False,
        },
        "held_aoa": HELD_AOA,
        "discovery": discovery,
        "anchor_source": {**anchor_source, "endpoint": "control_ema"},
        "primary_endpoint": PRIMARY_ENDPOINT,
        "endpoint_selection_allowed": False,
        "fit_samples": len(fit_windows),
        "samples": len(held_windows),
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "normalizer": candidate_row["normalizer"],
        "sample_stream_sha256": schedule["sha256"],
        "fit_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(fit_windows)), fit_windows, fit_metadata),
        "held_ordered_key_sha256": candidate_row[
            "held_ordered_key_sha256"],
        "start": start_report,
        "training": training,
        "endpoint": candidate_row["endpoint"],
        "anchor": anchor,
        "comparison": comparison,
        "confirmation_gate": gate,
        "prefix_access_audit": access_report,
        "git": identity,
        "hardware": hardware,
        "disk_preflight": disk,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "elapsed_seconds": time.perf_counter() - started,
        "release_checkpoint_written": False,
        "submission_authorized": False,
        "decision": gate["decision"],
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-held0", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--discovery-fold10", type=Path)
    parser.add_argument("--anchor-held0", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hardware-class", choices=("rtx4080_super",),
        default="rtx4080_super")
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    execution_values = (
        args.output_dir, args.clean_manifest, args.start_sim,
        args.start_sim_sha256, args.discovery_fold10, args.anchor_held0)
    if args.run_held0:
        if any(value is None for value in execution_values):
            raise ValueError("small4k held0 runner is missing required inputs")
        if (not args.execute_formal_gpu
                or args.acknowledge_protocol != FORMAL_ACKNOWLEDGEMENT):
            raise ValueError("small4k GPU execution requires acknowledgement")
        args.start_sim_sha256 = dr1.normalized_sha256(
            args.start_sim_sha256, "start-sim-sha256")
    elif any(value is not None for value in execution_values) or (
            args.execute_formal_gpu):
        raise ValueError("contract-only mode received execution inputs")


def write_stop(output_dir: Path, error: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stop.json"
    if path.exists():
        return
    if isinstance(error, torch.cuda.OutOfMemoryError):
        decision = "STOP_B210_SMALL4K_OOM"
    elif isinstance(error, TimeoutError):
        decision = "STOP_B210_SMALL4K_WALL_BUDGET"
    else:
        decision = "STOP_B210_SMALL4K_RUNTIME"
    dr1.write_json_atomic(path, {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "decision": decision,
        "error_type": type(error).__name__,
        "error": str(error),
        "release_checkpoint_written": False,
        "submission_authorized": False,
    })


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "contract": CONTRACT,
            "contract_sha256": CONTRACT_SHA256,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return
    try:
        result = run(args)
    except Exception as error:
        if args.output_dir is not None and not isinstance(error, FileExistsError):
            write_stop(args.output_dir, error)
        raise
    print(json.dumps({
        "event": "b210_small4k_held0_complete",
        "decision": result["decision"],
        "output": str(args.output_dir / "fold_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
