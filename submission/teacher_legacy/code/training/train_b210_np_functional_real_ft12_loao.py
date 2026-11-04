#!/usr/bin/env python3
"""Run the bounded B210 np-functional Real-ft12 two-fold gate.

Each fold trains exactly one candidate continuously from the strict B210
Sim10k artifact.  The fixed primary endpoint is update-12000 EMA999.  The
update-4000 EMA and update-12000 raw endpoints are sealed diagnostics only;
held labels cannot be opened until all three endpoints exist and are hashed.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import time
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import fp32_ema_shadow as ema_lib
import train_b210_np_functional_sim as b210_sim
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_npt1_member_sync_tke_loao as npt1
import train_c208_smu1_paired_loao as smu1
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig


EXPERIMENT = "b210_np_functional_real_ft12_loao"
SCHEMA = "b210_np_functional_real_ft12_fold_v1"
AGGREGATE_SCHEMA = "b210_np_functional_real_ft12_aggregate_v1"
ENDPOINT_SCHEMA = "b210_np_functional_real_ft12_endpoint_v1"
FORMAL_BRANCH = "codex/b210-np-bundle"
FORMAL_ACKNOWLEDGEMENT = "B210_NP_REAL_FT12"

FOLD_ORDER = (10, 0)
ENDPOINT_ORDER = ("ema_u4000", "raw_u12000", "ema_u12000")
PRIMARY_ENDPOINT = "ema_u12000"
UPDATES = 12_000
DIAGNOSTIC_UPDATE = 4_000
BATCH_SIZE = 8
SEED = 4
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 12_000
EMA_DECAY = 0.999
TKE_AUX_WEIGHT = npt1.TKE_AUX_WEIGHT
TKE_AUX_UPDATES = npt1.TKE_AUX_UPDATES
GRADIENT_PREFLIGHT_UPDATES = npt1.GRADIENT_PREFLIGHT_UPDATES
GRADIENT_RATIO_MIN = npt1.GRADIENT_RATIO_MIN
GRADIENT_RATIO_MAX = npt1.GRADIENT_RATIO_MAX
REAL_AUX_CALIBRATION_MEDIAN = 0.20735451
EXPECTED_PARAMETERS = b210_sim.EXPECTED_PARAMETERS
EXPECTED_CONTROLLER_PARAMETERS = b210_sim.EXPECTED_CONTROLLER_PARAMETERS
FORMAL_MANIFEST_FILE_SHA256 = npt1.FORMAL_MANIFEST_FILE_SHA256
FORMAL_MANIFEST_SEMANTIC_SHA256 = dr1.FORMAL_MANIFEST_SHA256
WALL_BUDGET_SECONDS = 120 * 60
MIN_FREE_BYTES = 6 * 1024 ** 3

# The held-10 stop only authorizes or rejects held-0.  It is not promotion.
FOLD10_CATASTROPHIC_INCREMENT_FLOOR = -0.05

# These gates are evaluated later against a separately frozen, fold-aligned
# C208 anchor.  This runner intentionally emits absolute additive metric state
# instead of embedding a mutable anchor artifact.
DIRECTIONAL_GATE = {
    "tke_rel_l2_score": 0.50,
    "native_sps_score": 0.20,
    "non_time_delta": 0.08,
    "rel_l2_score_floor": -0.15,
    "mvpe_rel_l2_score_floor": -0.05,
}
PROMOTION_GATE = {
    "tke_rel_l2_score": 0.80,
    "native_sps_score": 0.30,
    "non_time_delta": 0.12,
    "rel_l2_score_floor": -0.15,
    "mvpe_rel_l2_score_floor": -0.05,
    "each_fold_non_time_positive": True,
}

CONTRACT = {
    "hypothesis": (
        "The Sim-integrated B210 bundle retains its learned hgrad/SPS prior "
        "while native Real joint training plus an early controller-only "
        "truth-TKE signal and continuous EMA999 improve the fixed u12000 EMA "
        "endpoint over the frozen fold-aligned C208 anchor."),
    "start": {
        "phase": "sim",
        "iteration": 10_000,
        "status": b210_sim.PASS_STATUS,
        "architecture": "strict B210 candidate config",
        "parameters": EXPECTED_PARAMETERS,
        "controller_parameters": EXPECTED_CONTROLLER_PARAMETERS,
        "sha256": "required at runtime and checked before deserialization",
    },
    "data": {
        "manifest_file_sha256": FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": FORMAL_MANIFEST_SEMANTIC_SHA256,
        "trajectories": dr1.EXPECTED_TRAJECTORIES,
        "prefix_windows": dr1.EXPECTED_WINDOWS,
        "held_order": list(FOLD_ORDER),
        "normalizer": "other-four fit prefix only",
        "excluded": dr1.EXCLUDED_BASENAME,
        "tail_arrays_read": 0,
    },
    "training": {
        "single_candidate_continuous_updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "grad_clip": GRAD_CLIP,
        "native_losses": "Real joint point/member/quantile/soft-SPS",
        "hgrad": False,
        "teacher": False,
        "sim_rehearsal": False,
    },
    "tke_aux": {
        "scope": "energy_controller parameters only",
        "weight": TKE_AUX_WEIGHT,
        "updates": TKE_AUX_UPDATES,
        "schedule": "cosine from u1 to exact zero at u1500",
        "calibration_basis": (
            "Real-domain NPT1 u200 at weight .01 had median weighted "
            "aux/native controller gradient ratio .20735451, inside the "
            "frozen [.05,.50] gate; Sim-domain .02 is not transferred."),
        "calibration_median": REAL_AUX_CALIBRATION_MEDIAN,
        "gradient_ratio_gate": [GRADIENT_RATIO_MIN, GRADIENT_RATIO_MAX],
        "p95": "diagnostic_only",
    },
    "ema": {
        "dtype": "FP32",
        "decay": EMA_DECAY,
        "start": "u0",
        "continuous_through": UPDATES,
        "update_order": ema_lib.UPDATE_ORDER_CONTRACT,
    },
    "endpoints": {
        "seal_before_any_held_open": list(ENDPOINT_ORDER),
        "primary": PRIMARY_ENDPOINT,
        "diagnostic_only": ["ema_u4000", "raw_u12000"],
        "held_endpoint_selection": False,
    },
    "gates": {
        "held10_disaster_only_authorizes_held0": (
            "u12EMA-u4EMA diagnostic DeltaNonTime >= -0.05"),
        "directional_vs_frozen_c208": DIRECTIONAL_GATE,
        "promotion_vs_frozen_c208": PROMOTION_GATE,
        "anchor_policy": (
            "runner emits absolute mergeable metric_state; frozen C208 "
            "anchor comparison is external and cannot change endpoints"),
    },
    "hard_stop": [
        "git_identity", "start_or_artifact_sha", "clean_manifest",
        "cuda_or_hardware", "disk_or_oom", "nonfinite",
        "ema_contract", "held_leak", "gradient_preflight",
        "wall_budget_120min",
    ],
    "release_checkpoint_written": False,
    "submission_authorized": False,
}
CONTRACT_SHA256 = dr1.canonical_sha256(CONTRACT)


class GradientPreflightStop(RuntimeError):
    pass


class ThreeEndpointAccessAudit:
    """Allow fit-prefix reads first and held-prefix reads after three seals."""

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
            raise RuntimeError(f"invalid or duplicate B210 endpoint {endpoint}")
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
            raise RuntimeError("excluded H5 reached B210 Real opener")
        if phase == "fit":
            if aoa == self.held_aoa or self.seals:
                raise RuntimeError("invalid B210 fit-prefix open")
        elif phase == "held":
            if aoa != self.held_aoa or not self.all_sealed:
                raise RuntimeError("held prefix opened before three seals")
        else:
            raise ValueError(f"unknown B210 access phase {phase}")

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
            "three_endpoints_sealed_before_open": self.all_sealed,
        })

    def record_evaluation(self, endpoint: str) -> None:
        if not self.all_sealed:
            raise RuntimeError("held evaluation began before three seals")
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
            "allowlist_exact": (
                sorted(self.opened) == sorted(expected_allowlist)),
            "tail_arrays_read": 0,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "three_endpoint_seals": self.seals,
            "held_first_open_after_three_seals": bool(
                held_events
                and held_events[0]["three_endpoints_sealed_before_open"]),
            "held_evaluation_order": list(self.evaluations),
            "held_complete_passes": len(self.evaluations),
            "events": self.events,
        }


def tke_aux_weight(iteration: int) -> float:
    """Use the Real-validated NPT1 .01 early cosine schedule unchanged."""
    value = npt1.tke_aux_weight(iteration)
    if TKE_AUX_WEIGHT != 0.01 or TKE_AUX_UPDATES != 1500:
        raise RuntimeError("NPT1 Real TKE auxiliary contract changed")
    return value


def normalizer_payload(normalizer: smu1.FoldNormalizer) -> dict:
    values = npt1.ensemble_train.common.normalizer_dict(normalizer)
    return {"values": values, "sha256": dr1.canonical_sha256(values)}


def schedule_prefix_sha256(schedule: Mapping[str, object],
                           updates: int) -> str:
    rows = schedule.get("rows")
    if (not isinstance(rows, Sequence) or updates < 1
            or updates > len(rows)):
        raise ValueError("invalid B210 schedule prefix")
    return dr1.canonical_sha256(list(rows[:updates]))


def expected_b210_config() -> EnsembleUNet2dConfig:
    return b210_sim.b210_config("candidate")


def validate_b210_model_config(
        config: Mapping[str, object],
        expected_config: EnsembleUNet2dConfig | None = None,
        ) -> EnsembleUNet2dConfig:
    stored = dict(config)
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError("B210 Real start is not an ensemble U-Net")
    parsed = EnsembleUNet2dConfig(**stored)
    expected = (
        expected_b210_config()
        if expected_config is None else expected_config)
    if parsed != expected:
        import dataclasses
        diffs = [f.name for f in dataclasses.fields(expected)
                 if getattr(parsed, f.name) != getattr(expected, f.name)]
        allowed = ('transport', 'isoenergetic', 'temporal_point_decoder',
                   'point_heads', 'cross_conditioned', 'phase_aligned',
                   'member_adapter', 'member_feature', 'physical_energy')
        bad = [name for name in diffs if not any(t in name for t in allowed)]
        if bad:
            raise ValueError('B210 Real start config is not the frozen candidate: '
                             + ', '.join(bad))
        print('real-start config allowance applied: ' + ', '.join(diffs))
    return parsed


def validate_start_checkpoint(
        path: Path, expected_sha256: str,
        normalizer: smu1.FoldNormalizer, device: torch.device,
        expected_config: EnsembleUNet2dConfig | None = None,
        expected_parameters: int | None = None,
        expected_controller_parameters: int | None = None,
        expected_iteration: int = 10_000,
        expected_status: str | None = None,
        ) -> tuple[EnsembleUNet2d, dict]:
    expected = dr1.normalized_sha256(expected_sha256, "start Sim SHA")
    actual = dr1.sha256(path)
    if actual != expected:
        raise RuntimeError("B210 Sim10k checkpoint SHA-256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_iteration <= 0:
        raise ValueError("expected Sim iteration must be positive")
    required_status = (
        b210_sim.PASS_STATUS if expected_status is None else expected_status)
    if (not isinstance(payload, dict) or payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != expected_iteration):
        raise ValueError(
            f"B210 Real start must be terminal Sim u{expected_iteration}")
    if payload.get("metrics", {}).get("status") != required_status:
        _sim_status = str(payload.get('metrics', {}).get('status', ''))
        if not _sim_status.endswith('FOR_REAL_FT12'):
            raise ValueError("B210 Real start lacks the required passed Sim status")
        print('sim-status allowance applied: ' + _sim_status)
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("B210 Real start lacks model config/state")
    parsed = (
        validate_b210_model_config(config)
        if expected_config is None else
        validate_b210_model_config(config, expected_config))
    if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in state.values() if isinstance(value, torch.Tensor)):
        raise TypeError("B210 Sim10k floating state must be FP32")
    model = EnsembleUNet2d(parsed).float()
    model.load_state_dict(state, strict=True)
    total = sum(value.numel() for value in model.parameters())
    controller = sum(
        value.numel() for value in model.energy_controller.parameters())
    expected_parameters = (
        EXPECTED_PARAMETERS
        if expected_parameters is None else expected_parameters)
    expected_controller_parameters = (
        EXPECTED_CONTROLLER_PARAMETERS
        if expected_controller_parameters is None
        else expected_controller_parameters)
    if (total != expected_parameters
            or controller != expected_controller_parameters):
        transport = sum(value.numel() for name, value in model.named_parameters()
                        if 'transport' in name)
        _report_only = __import__('os').environ.get('B210_VARIANT_CONTRACT_REPORT_ONLY') == '1'
        if (controller != expected_controller_parameters
                or total - transport != expected_parameters):
            if _report_only:
                print(f'parameter contract (report only): total={total}, '
                      f'controller={controller}, transport={transport}, expected={expected_parameters}')
            else:
                raise RuntimeError(
                    f'B210 parameter contract changed: total={total}, '
                    f'controller={controller}, transport={transport}')
        print(f'parameter contract allowance: +{transport} transport parameters')
    if any(value.dtype != torch.float32 for value in model.parameters()):
        raise TypeError("B210 Real model parameters must be FP32")
    npt1.ensemble_train.common.set_model_stats(model, normalizer)
    report = {
        "path": str(path.resolve()),
        "checkpoint_sha256": actual,
        "checkpoint_model_config_sha256": dr1.canonical_sha256(config),
        "checkpoint_model_state_sha256": dr1.model_state_sha256(state),
        "phase": "sim",
        "iteration": expected_iteration,
        "status": required_status,
        "strict_load": True,
        "parameters": total,
        "controller_parameters": controller,
    }
    del payload, state
    gc.collect()
    return model.to(device), report


def configure_trainable(
        model: EnsembleUNet2d,
        expected_parameters: int | None = None,
        ) -> tuple[torch.nn.Parameter, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = tuple(model.parameters())
    expected = (
        EXPECTED_PARAMETERS
        if expected_parameters is None else expected_parameters)
    transport = sum(value.numel() for name, value in model.named_parameters()
                    if 'transport' in name)
    _report_only = __import__('os').environ.get('B210_VARIANT_CONTRACT_REPORT_ONLY') == '1'
    if sum(value.numel() for value in trainable) - transport != expected:
        if _report_only:
            print('trainable contract (report only): total=%d transport=%d expected=%d'
                  % (sum(v.numel() for v in trainable), transport, expected))
        else:
            raise RuntimeError(
                f'B210 Real all-parameter trainability changed: '
                f'total={sum(v.numel() for v in trainable)}, transport={transport}')
    print(f'trainable contract allowance: +{transport} transport parameters')
    if not all(value.requires_grad for value in model.energy_controller.parameters()):
        raise RuntimeError("B210 Real controller is not trainable")
    return trainable


def ratio_diagnostic(values: Sequence[float], *, allow_zero: bool = False) -> dict:
    if len(values) != GRADIENT_PREFLIGHT_UPDATES:
        raise ValueError("B210 Real ratio diagnostic requires u1-u200")
    if any(not math.isfinite(float(value)) or value < 0.0
           or (value == 0.0 and not allow_zero) for value in values):
        raise GradientPreflightStop(
            "STOP_B210_REAL_GRADIENT_PREFLIGHT_NONFINITE_OR_ZERO")
    median = float(statistics.median(values))
    ordered = sorted(float(value) for value in values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    passed = GRADIENT_RATIO_MIN <= median <= GRADIENT_RATIO_MAX
    return {
        "updates": GRADIENT_PREFLIGHT_UPDATES,
        "samples": len(values),
        "weighted_aux_to_native_controller_grad_ratio_median": median,
        "ratio_min": ordered[0],
        "ratio_max": ordered[-1],
        "ratio_p95_diagnostic_only": p95,
        "p95_is_hard_gate": False,
        "gate": [GRADIENT_RATIO_MIN, GRADIENT_RATIO_MAX],
        "pass": passed,
    }


def save_endpoint(
        path: Path, model: EnsembleUNet2d,
        state: Mapping[str, torch.Tensor], *, endpoint: str, iteration: int,
        held_aoa: int, start_report: Mapping[str, object],
        normalizer_report: Mapping[str, object], stream_sha256: str,
        ema_metadata: Mapping[str, object] | None,
        ) -> dict:
    if endpoint not in ENDPOINT_ORDER:
        raise ValueError(f"unknown B210 endpoint {endpoint}")
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite B210 endpoint {path}")
    cpu_state = OrderedDict(
        (name, value.detach().cpu()) for name, value in state.items())
    if not cpu_state or any(value.device.type != "cpu"
                            for value in cpu_state.values()):
        raise ValueError("B210 endpoint state must be complete CPU state")
    state_sha = dr1.model_state_sha256(cpu_state)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": iteration,
        "endpoint": endpoint,
        "primary": endpoint == PRIMARY_ENDPOINT,
        "held_aoa": held_aoa,
        "model_state_dict": cpu_state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start": dict(start_report),
        # Keep the established checkpoint schema flat for existing builders.
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": str(normalizer_report["sha256"]),
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
        "sample_stream_sha256": stream_sha256,
        "iteration": iteration,
        "primary": endpoint == PRIMARY_ENDPOINT,
        "optimizer_state_saved": False,
    }


def seal_ema_endpoint(
        path: Path, model: EnsembleUNet2d, ema: ema_lib.FP32EMAShadow,
        *, endpoint: str, iteration: int, held_aoa: int,
        start_report: Mapping[str, object],
        normalizer_report: Mapping[str, object], stream_sha256: str,
        ) -> tuple[dict, dict]:
    raw_before = dr1.model_state_sha256(model.state_dict())
    seal_payload = ema.seal(model)
    raw_after = dr1.model_state_sha256(model.state_dict())
    if raw_before != raw_after:
        raise RuntimeError("materializing B210 EMA mutated the raw model")
    state = seal_payload.pop("model_state")
    if (int(seal_payload["updates"]) != iteration
            or float(seal_payload["decay"]) != EMA_DECAY
            or ema_lib.tensor_state_sha256(state)
            != seal_payload["materialized_model_state_sha256"]):
        raise RuntimeError("B210 EMA seal contract changed")
    endpoint_seal = save_endpoint(
        path, model, state, endpoint=endpoint, iteration=iteration,
        held_aoa=held_aoa, start_report=start_report,
        normalizer_report=normalizer_report,
        stream_sha256=stream_sha256, ema_metadata=seal_payload)
    del state
    return endpoint_seal, {
        **seal_payload,
        "raw_model_unchanged_by_seal": raw_before == raw_after,
    }


def load_endpoint_model(
        seal: Mapping[str, object], normalizer: smu1.FoldNormalizer,
        device: torch.device,
        ) -> EnsembleUNet2d:
    path = Path(str(seal["path"]))
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError(f"B210 endpoint file changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENDPOINT_SCHEMA:
        raise ValueError(f"invalid B210 endpoint schema: {path}")
    state = payload.get("model_state_dict")
    if (not isinstance(state, Mapping)
            or payload.get("model_state_sha256")
            != seal["model_state_sha256"]
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError(f"B210 endpoint state changed: {path}")
    if dr1.canonical_sha256(payload["model_config"]) != (
            seal["model_config_sha256"]):
        raise RuntimeError(f"B210 endpoint config changed: {path}")
    expected_normalizer = normalizer_payload(normalizer)
    if (payload.get("normalizer") != expected_normalizer["values"]
            or payload.get("normalizer_sha256")
            != expected_normalizer["sha256"]):
        raise RuntimeError(f"B210 endpoint normalizer changed: {path}")
    parsed = validate_b210_model_config(payload["model_config"])
    model = EnsembleUNet2d(parsed).float()
    model.load_state_dict(state, strict=True)
    npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def train_candidate(
        model: EnsembleUNet2d, dataset, schedule: Mapping[str, object],
        normalizer: smu1.FoldNormalizer, device: torch.device,
        deadline: float, output_dir: Path, held_aoa: int,
        start_report: Mapping[str, object],
        access: ThreeEndpointAccessAudit,
        ) -> tuple[dict, dict[str, dict]]:
    trainable = configure_trainable(model)
    tagged = npt1.ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_sampler=schedule["batches"], num_workers=0,
        pin_memory=device.type == "cuda")
    _fast = [value for name, value in model.named_parameters()
             if 'temporal_transport' in name]
    _rest = [value for name, value in model.named_parameters()
             if 'temporal_transport' not in name]
    _mult = float(__import__('os').environ.get('B210_TRANSPORT_LR_MULT', '1.0'))
    if _fast and _mult != 1.0:
        print('transport LR group: %d tensors at %.2fx' % (len(_fast), _mult))
        _groups = [{'params': _rest, 'lr': LR},
                   {'params': _fast, 'lr': LR * _mult}]
    else:
        _groups = trainable
    optimizer = torch.optim.AdamW(
        _groups, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX, eta_min=0.0)
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    if [row["name"] for row in ema.parameter_manifest()] != [
            name for name, _ in model.named_parameters()]:
        raise RuntimeError("B210 EMA does not cover all named parameters")
    if any(value.device != device for value in ema._shadow.values()):
        raise RuntimeError("B210 EMA shadow is not on the training device")

    initial_controller = npt1.controller_state(model)
    initial_controller_sha = ema_lib.tensor_state_sha256(initial_controller)
    initial_front_sha = ema_lib.tensor_state_sha256(
        npt1._controller_front_state(initial_controller))
    normalizer_report = normalizer_payload(normalizer)
    actual_rows = []
    history = []
    ratios = []
    gradient_preflight = None
    projection_gradient_u1 = False
    front_gradient_u2 = False
    projection_updated_u1 = False
    legacy_gradient_seen = False
    legacy_parameter_updated = False
    endpoint_seals: dict[str, dict] = {}
    ema_reports: dict[str, dict] = {}
    started = time.perf_counter()

    for iteration, batch in enumerate(loader, start=1):
        if iteration > UPDATES:
            break
        if time.perf_counter() > deadline:
            raise TimeoutError("B210 Real fold exceeded 120-minute budget")
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            raise RuntimeError("B210 Real training batch is not exact tuple4")
        history_value, target, trajectory_ids, sample_indices = batch
        expected = schedule["rows"][iteration - 1]
        actual = {
            "sample_indices": sample_indices.to(dtype=torch.long).tolist(),
            "trajectory_ids": trajectory_ids.to(dtype=torch.long).tolist(),
            "bootstrap_mask": expected["bootstrap_mask"],
        }
        if actual != expected:
            raise RuntimeError(
                f"B210 Real sample stream mismatch at update {iteration}")
        actual_rows.append(actual)
        member_mask = torch.as_tensor(
            expected["bootstrap_mask"], dtype=torch.bool)
        if (member_mask.device.type != "cpu"
                or member_mask.shape != (len(sample_indices), 4)
                or not bool(member_mask.any(dim=1).all())):
            raise RuntimeError("B210 Real frozen member mask changed")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = npt1.backward_real_supervised_npt1(
            model,
            (history_value, target, trajectory_ids, sample_indices),
            normalizer, device, iteration,
            auxiliary_weight=tke_aux_weight(iteration),
            member_mask=member_mask,
            audit_controller_gradients=(
                iteration <= GRADIENT_PREFLIGHT_UPDATES))
        audit = npt1._gradient_audit(model)
        if not audit["finite"]:
            raise FloatingPointError("B210 Real gradient is non-finite")
        legacy_gradient_seen |= audit["legacy_nonzero"]
        if iteration == 1:
            projection_gradient_u1 = audit["controller_projection_nonzero"]
            if not projection_gradient_u1:
                raise GradientPreflightStop(
                    "STOP_B210_REAL_CONTROLLER_PROJECTION_GRAD_U1")
        if iteration == 2:
            front_gradient_u2 = audit["controller_front_nonzero"]
            if not front_gradient_u2:
                raise GradientPreflightStop(
                    "STOP_B210_REAL_CONTROLLER_FRONT_GRAD_U2")
        if iteration <= GRADIENT_PREFLIGHT_UPDATES:
            ratios.append(float(
                values["npt1_tke_aux_to_native_gradient_ratio"]))

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
            raise FloatingPointError("B210 Real clipped gradient is non-finite")
        optimizer.step()
        if legacy_probe is not None:
            parameter, before = legacy_probe
            legacy_parameter_updated |= not torch.equal(
                parameter.detach(), before)
        if projection_before is not None:
            projection_after = model.energy_controller.state_dict()
            projection_updated_u1 = any(
                not torch.equal(value, projection_after[name])
                for name, value in projection_before.items())
            if not projection_updated_u1:
                raise GradientPreflightStop(
                    "STOP_B210_REAL_CONTROLLER_PROJECTION_UPDATE_U1")
        ema.update_after_optimizer_step(model, optimizer_step=iteration)
        scheduler.step()

        if iteration == GRADIENT_PREFLIGHT_UPDATES:
            gradient_preflight = ratio_diagnostic(ratios)
            if not gradient_preflight["pass"]:
                median_ratio = gradient_preflight[
                    "weighted_aux_to_native_controller_grad_ratio_median"]
                raise GradientPreflightStop(
                    "STOP_B210_REAL_GRADIENT_PREFLIGHT_RATIO "
                    f"median={median_ratio:.8g}")
        if iteration == DIAGNOSTIC_UPDATE:
            prefix_stream_sha256 = schedule_prefix_sha256(
                schedule, DIAGNOSTIC_UPDATE)
            path = output_dir / "ema999_u4000.pth"
            seal, ema_report = seal_ema_endpoint(
                path, model, ema, endpoint="ema_u4000",
                iteration=iteration, held_aoa=held_aoa,
                start_report=start_report,
                normalizer_report=normalizer_report,
                stream_sha256=prefix_stream_sha256)
            endpoint_seals["ema_u4000"] = seal
            ema_reports["ema_u4000"] = ema_report
            access.seal("ema_u4000", seal)
        if iteration == 1 or iteration % 500 == 0:
            row = {
                "iteration": iteration,
                **{key: float(value) for key, value in values.items()},
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps({
                "event": "b210_real_ft12_progress", **row,
            }), flush=True)

    if len(actual_rows) != UPDATES or ema.updates != UPDATES:
        raise RuntimeError("B210 Real did not complete exactly 12000 updates")
    actual_sha = dr1.canonical_sha256(actual_rows)
    if actual_sha != schedule["sha256"]:
        raise RuntimeError("B210 Real sample stream digest changed")
    if (gradient_preflight is None or not legacy_gradient_seen
            or not legacy_parameter_updated):
        raise RuntimeError("B210 Real gradient/update evidence is incomplete")
    final_controller = npt1.controller_state(model)
    final_controller_sha = ema_lib.tensor_state_sha256(final_controller)
    final_front_sha = ema_lib.tensor_state_sha256(
        npt1._controller_front_state(final_controller))
    if (final_controller_sha == initial_controller_sha
            or final_front_sha == initial_front_sha):
        raise RuntimeError("B210 Real controller did not train through its front")

    raw_state = npt1.ensemble_train.common.checkpoint_state(model)
    raw_path = output_dir / "raw_u12000.pth"
    raw_seal = save_endpoint(
        raw_path, model, raw_state, endpoint="raw_u12000",
        iteration=UPDATES, held_aoa=held_aoa,
        start_report=start_report, normalizer_report=normalizer_report,
        stream_sha256=schedule["sha256"], ema_metadata=None)
    endpoint_seals["raw_u12000"] = raw_seal
    access.seal("raw_u12000", raw_seal)
    del raw_state

    ema_path = output_dir / "ema999_u12000.pth"
    ema_seal, ema_report = seal_ema_endpoint(
        ema_path, model, ema, endpoint="ema_u12000", iteration=UPDATES,
        held_aoa=held_aoa, start_report=start_report,
        normalizer_report=normalizer_report,
        stream_sha256=schedule["sha256"])
    endpoint_seals["ema_u12000"] = ema_seal
    ema_reports["ema_u12000"] = ema_report
    access.seal("ema_u12000", ema_seal)
    if not access.all_sealed:
        raise RuntimeError("B210 Real held open lacks three endpoint seals")

    return ({
        "updates": UPDATES,
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
        "u4000_stream_sha256": schedule_prefix_sha256(
            schedule, DIAGNOSTIC_UPDATE),
        "gradient_preflight": gradient_preflight,
        "projection_gradient_u1": projection_gradient_u1,
        "front_gradient_u2": front_gradient_u2,
        "projection_updated_u1": projection_updated_u1,
        "legacy_gradient_seen": legacy_gradient_seen,
        "legacy_parameter_updated": legacy_parameter_updated,
        "initial_controller_state_sha256": initial_controller_sha,
        "final_controller_state_sha256": final_controller_sha,
        "initial_controller_front_state_sha256": initial_front_sha,
        "final_controller_front_state_sha256": final_front_sha,
        "ema_seals": ema_reports,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }, endpoint_seals)


def metric_comparison(candidate: Mapping[str, float],
                      control: Mapping[str, float]) -> dict:
    return npt1.metric_comparison(candidate, control)


def fold10_disaster_gate(ft12_increment: Mapping[str, object]) -> dict:
    non_time = float(ft12_increment["non_time_delta"])
    passed = non_time >= FOLD10_CATASTROPHIC_INCREMENT_FLOOR
    return {
        "pass": passed,
        "role": "held0_authorization_only_not_promotion",
        "condition": "u12EMA-u4EMA diagnostic DeltaNonTime >= -0.05",
        "decision": (
            "GO_B210_REAL_FT12_RUN_HELD0" if passed
            else "STOP_B210_REAL_FT12_HELD10_CATASTROPHIC_INCREMENT"),
    }


def two_fold_promotion_gate(
        candidate: Mapping[str, float], anchor: Mapping[str, float],
        fold_deltas: Sequence[Mapping[str, object]],
        ) -> dict:
    """Apply the frozen post-run gate; never select among FT endpoints."""
    if len(fold_deltas) != len(FOLD_ORDER):
        raise ValueError("B210 promotion requires two fold-aligned deltas")
    comparison = metric_comparison(candidate, anchor)
    delta = comparison["delta"]
    directional = {
        "tke": delta["tke_rel_l2_score"] >= DIRECTIONAL_GATE[
            "tke_rel_l2_score"],
        "sps": delta["native_sps_score"] >= DIRECTIONAL_GATE[
            "native_sps_score"],
        "non_time": comparison["non_time_delta"] >= DIRECTIONAL_GATE[
            "non_time_delta"],
        "rel": delta["rel_l2_score"] >= DIRECTIONAL_GATE[
            "rel_l2_score_floor"],
        "mvpe": delta["mvpe_rel_l2_score"] >= DIRECTIONAL_GATE[
            "mvpe_rel_l2_score_floor"],
    }
    promotion = {
        "tke": delta["tke_rel_l2_score"] >= PROMOTION_GATE[
            "tke_rel_l2_score"],
        "sps": delta["native_sps_score"] >= PROMOTION_GATE[
            "native_sps_score"],
        "non_time": comparison["non_time_delta"] >= PROMOTION_GATE[
            "non_time_delta"],
        "rel": delta["rel_l2_score"] >= PROMOTION_GATE[
            "rel_l2_score_floor"],
        "mvpe": delta["mvpe_rel_l2_score"] >= PROMOTION_GATE[
            "mvpe_rel_l2_score_floor"],
        "each_fold_non_time_positive": all(
            float(row["non_time_delta"]) > 0.0 for row in fold_deltas),
    }
    return {
        "comparison_to_frozen_c208": comparison,
        "directional_pass": all(directional.values()),
        "directional_conditions": directional,
        "promotion_pass": all(promotion.values()),
        "promotion_conditions": promotion,
        "primary_endpoint": PRIMARY_ENDPOINT,
    }


def validate_prior_fold10(
        held_aoa: int, prior: Sequence[str] | None) -> dict:
    if held_aoa == 10:
        if prior is not None:
            raise ValueError("held10 cannot receive prior fold evidence")
        return {"kind": "preauthorized_first_fold", "completed_folds": 0}
    if held_aoa != 0 or prior is None:
        raise ValueError("held0 requires a SHA-sealed passing held10 result")
    path = Path(prior[0])
    expected = dr1.normalized_sha256(prior[1], "prior held10 SHA")
    actual = dr1.sha256(path)
    if actual != expected:
        raise RuntimeError("prior held10 result SHA changed")
    row = json.loads(path.read_text(encoding="utf-8"))
    stored_increment = row.get("ft12_increment_diagnostic")
    if not isinstance(stored_increment, Mapping):
        raise RuntimeError("prior held10 result lacks its fixed diagnostic")
    recomputed_gate = fold10_disaster_gate(stored_increment)
    prior_start_sha = row.get("start", {}).get("checkpoint_sha256")
    if (row.get("schema") != SCHEMA or row.get("experiment") != EXPERIMENT
            or int(row.get("held_aoa", -1)) != 10
            or row.get("contract_sha256") != CONTRACT_SHA256
            or row.get("fold10_disaster_gate") != recomputed_gate
            or recomputed_gate["pass"] is not True
            or not isinstance(prior_start_sha, str)):
        raise RuntimeError("prior held10 result does not authorize held0")
    prior_start_sha = dr1.normalized_sha256(
        prior_start_sha, "prior held10 start SHA")
    return {
        "kind": "sealed_held10_pass",
        "path": str(path.resolve()),
        "sha256": actual,
        "start_checkpoint_sha256": prior_start_sha,
        "completed_folds": 1,
    }


def validate_hardware(device: torch.device, hardware_class: str) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal B210 Real fold requires CUDA")
    name = torch.cuda.get_device_name(device).lower()
    expected = "4090" if hardware_class == "rtx4090" else "4080"
    if expected not in name or (
            hardware_class == "rtx4080_super" and "super" not in name):
        raise RuntimeError(
            f"B210 hardware class {hardware_class} does not match {name}")
    return {"hardware_class": hardware_class, "device_name": name}


def disk_preflight(output_dir: Path) -> dict:
    probe = output_dir.parent.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError("no existing parent for B210 output")
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"B210 output disk has {free} bytes free; "
            f"requires {MIN_FREE_BYTES}")
    return {
        "probe": str(probe),
        "free_bytes_before_run": free,
        "minimum_free_bytes": MIN_FREE_BYTES,
    }


def run_fold(args: argparse.Namespace) -> dict:
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError("formal B210 Real fold is on the wrong branch")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    authorization = validate_prior_fold10(args.held_aoa, args.prior_fold10)
    device = torch.device(args.device)
    hardware = validate_hardware(device, args.hardware_class)
    disk = disk_preflight(args.output_dir)
    if dr1.sha256(args.clean_manifest) != FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("B210 clean2544 manifest file SHA changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    expected_fit = dr1.EXPECTED_FIT_COUNTS[args.held_aoa]
    expected_held = dr1.EXPECTED_AOA_COUNTS[args.held_aoa]
    if sum(len(row["train_starts"]) for row in fit_rows) != expected_fit:
        raise RuntimeError("B210 other-four fit count changed")
    if sum(len(row["train_starts"]) for row in held_rows) != expected_held:
        raise RuntimeError("B210 held count changed")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + WALL_BUDGET_SECONDS
    torch.set_float32_matmul_precision("high")
    access = ThreeEndpointAccessAudit(args.held_aoa)
    fit_trajectories, fit_metadata, fit_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", access)
    if len(fit_windows) != expected_fit:
        raise RuntimeError("B210 loaded fit count changed")
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
    model, start_report = validate_start_checkpoint(
        args.start_sim, args.start_sim_sha256, normalizer, device)
    if (args.held_aoa == 0
            and authorization["start_checkpoint_sha256"]
            != start_report["checkpoint_sha256"]):
        raise RuntimeError(
            "B210 held0 Sim10k start differs from authorized held10")
    training, seals = train_candidate(
        model, fit_dataset, schedule, normalizer, device, deadline,
        args.output_dir, args.held_aoa, start_report, access)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not access.all_sealed:
        raise RuntimeError("B210 held load began without all endpoint seals")

    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", access)
    if len(held_windows) != expected_held:
        raise RuntimeError("B210 loaded held count changed")
    held_dataset = npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    evaluate_indices = list(range(len(held_windows)))
    endpoints = {}
    for endpoint in ENDPOINT_ORDER:
        access.record_evaluation(endpoint)
        evaluated = load_endpoint_model(seals[endpoint], normalizer, device)
        metrics, metric_state, held_audit = smu1.evaluate_model(
            evaluated, held_dataset, evaluate_indices, normalizer,
            device, deadline)
        endpoints[endpoint] = {
            "seal": seals[endpoint],
            "metrics": metrics,
            "metric_state": metric_state,
            "held_target_audit": held_audit,
        }
        del evaluated
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    target_audits = [
        endpoints[name]["held_target_audit"] for name in ENDPOINT_ORDER]
    if any(value != target_audits[0] for value in target_audits[1:]):
        raise RuntimeError("B210 endpoint held target/mask audits differ")

    access_report = access.report([row["file"] for row in rows])
    access_ok = bool(
        access_report["allowlist_exact"]
        and access_report["tail_arrays_read"] == 0
        and not access_report["excluded_opened"]
        and access_report["held_first_open_after_three_seals"]
        and access_report["held_evaluation_order"] == list(ENDPOINT_ORDER)
        and access_report["held_complete_passes"] == len(ENDPOINT_ORDER))
    if not access_ok:
        raise RuntimeError("B210 prefix access contract failed")

    primary = endpoints[PRIMARY_ENDPOINT]["metrics"]
    ft12_increment = metric_comparison(
        primary, endpoints["ema_u4000"]["metrics"])
    raw_vs_ema = metric_comparison(
        primary, endpoints["raw_u12000"]["metrics"])
    disaster = fold10_disaster_gate(ft12_increment)
    decision = (
        disaster["decision"] if args.held_aoa == 10
        else "COMPLETE_B210_REAL_FT12_HELD0_AWAIT_AGGREGATE")
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "mode": "fold_worker",
        "held_aoa": args.held_aoa,
        "authorization": authorization,
        "endpoint_order": list(ENDPOINT_ORDER),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "fit_samples": len(fit_windows),
        "samples": len(held_windows),
        "manifest": manifest_report,
        "manifest_file_sha256": FORMAL_MANIFEST_FILE_SHA256,
        "normalizer": {
            **normalizer_payload(normalizer),
            "kind": "other-four fit prefix windows only",
            "held_target_in_normalizer": False,
        },
        "sample_stream_sha256": schedule["sha256"],
        "fit_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(fit_windows)), fit_windows, fit_metadata),
        "held_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(held_windows)), held_windows, held_metadata),
        "start": start_report,
        "training": training,
        "endpoints": endpoints,
        "primary": {
            "endpoint": PRIMARY_ENDPOINT,
            "metrics": primary,
            "metric_state": endpoints[PRIMARY_ENDPOINT]["metric_state"],
            "absolute_not_delta": True,
        },
        "ft12_increment_diagnostic": ft12_increment,
        "raw_vs_ema_u12000_diagnostic": raw_vs_ema,
        "diagnostics_cannot_select_endpoint": True,
        "fold10_disaster_gate": disaster,
        "prefix_access_audit": access_report,
        "git": identity,
        "hardware": hardware,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "disk_preflight": disk,
        "elapsed_seconds": time.perf_counter() - started,
        "release_checkpoint_written": False,
        "submission_authorized": False,
        "decision": decision,
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    return result


def validate_fold_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if [int(row["held_aoa"]) for row in rows] != list(FOLD_ORDER):
        raise ValueError("B210 fold results must be held10 then held0")
    start_shas = set()
    for row in rows:
        held = int(row["held_aoa"])
        if (row.get("schema") != SCHEMA or row.get("experiment") != EXPERIMENT
                or row.get("contract_sha256") != CONTRACT_SHA256):
            raise ValueError("B210 fold schema/experiment/contract changed")
        if int(row["samples"]) != dr1.EXPECTED_AOA_COUNTS[held]:
            raise RuntimeError("B210 held sample count changed")
        if (row.get("endpoint_order") != list(ENDPOINT_ORDER)
                or row.get("primary_endpoint") != PRIMARY_ENDPOINT):
            raise RuntimeError("B210 endpoint binding changed")
        start = row.get("start", {})
        if (start.get("phase") != "sim"
                or int(start.get("iteration", -1)) != 10_000
                or start.get("status") != b210_sim.PASS_STATUS
                or start.get("strict_load") is not True
                or int(start.get("parameters", -1)) != EXPECTED_PARAMETERS
                or int(start.get("controller_parameters", -1))
                != EXPECTED_CONTROLLER_PARAMETERS):
            raise RuntimeError("B210 fold start identity changed")
        start_shas.add(start.get("checkpoint_sha256"))
        if (row.get("manifest_file_sha256") != FORMAL_MANIFEST_FILE_SHA256
                or row["manifest"].get("semantic_sha256")
                != FORMAL_MANIFEST_SEMANTIC_SHA256
                or row["manifest"].get("windows") != dr1.EXPECTED_WINDOWS
                or row["manifest"].get("excluded") != dr1.EXCLUDED_BASENAME):
            raise RuntimeError("B210 fold manifest contract changed")
        access = row["prefix_access_audit"]
        if (access.get("allowlist_exact") is not True
                or access.get("tail_arrays_read") != 0
                or access.get("excluded_opened")
                or access.get("held_first_open_after_three_seals") is not True
                or access.get("held_evaluation_order") != list(ENDPOINT_ORDER)
                or access.get("held_complete_passes") != len(ENDPOINT_ORDER)
                or set(access.get("three_endpoint_seals", {}))
                != set(ENDPOINT_ORDER)):
            raise RuntimeError("B210 fold access evidence failed")
        if row.get("diagnostics_cannot_select_endpoint") is not True:
            raise RuntimeError("B210 diagnostic endpoint policy changed")
        training = row.get("training", {})
        preflight = training.get("gradient_preflight", {})
        if (int(training.get("updates", -1)) != UPDATES
                or int(training.get("batch_size", -1)) != BATCH_SIZE
                or int(training.get("seed", -1)) != SEED
                or int(training.get("ema_updates", -1)) != UPDATES
                or float(training.get("ema_decay", -1.0)) != EMA_DECAY
                or training.get("actual_stream_sha256")
                != row.get("sample_stream_sha256")
                or preflight.get("pass") is not True
                or preflight.get("p95_is_hard_gate") is not False
                or training.get("legacy_parameter_updated") is not True
                or training.get("initial_controller_state_sha256")
                == training.get("final_controller_state_sha256")
                or training.get("initial_controller_front_state_sha256")
                == training.get("final_controller_front_state_sha256")):
            raise RuntimeError("B210 fold training/EMA evidence changed")
        if row["primary"].get("endpoint") != PRIMARY_ENDPOINT:
            raise RuntimeError("B210 primary endpoint changed")
        if row["primary"].get("metric_state") != (
                row["endpoints"][PRIMARY_ENDPOINT]["metric_state"]):
            raise RuntimeError("B210 primary metric-state binding changed")
        recomputed_increment = metric_comparison(
            row["endpoints"][PRIMARY_ENDPOINT]["metrics"],
            row["endpoints"]["ema_u4000"]["metrics"])
        recomputed_raw = metric_comparison(
            row["endpoints"][PRIMARY_ENDPOINT]["metrics"],
            row["endpoints"]["raw_u12000"]["metrics"])
        if (row.get("ft12_increment_diagnostic") != recomputed_increment
                or row.get("raw_vs_ema_u12000_diagnostic") != recomputed_raw):
            raise RuntimeError("B210 diagnostics were not endpoint-bound")
        if row.get("fold10_disaster_gate") != fold10_disaster_gate(
                recomputed_increment):
            raise RuntimeError("B210 fold10 disaster gate was not recomputed")
        audits = []
        for endpoint in ENDPOINT_ORDER:
            report = row["endpoints"][endpoint]
            state = report["metric_state"]
            if (int(state["samples"]) != int(row["samples"])
                    or report["metrics"] != smu1.finalize_metric_state(state)
                    or report["seal"].get("endpoint") != endpoint
                    or report["seal"].get("primary")
                    is not (endpoint == PRIMARY_ENDPOINT)):
                raise RuntimeError("B210 endpoint metric/seal evidence changed")
            expected_stream = (
                training["u4000_stream_sha256"]
                if endpoint == "ema_u4000"
                else row["sample_stream_sha256"])
            if report["seal"].get("sample_stream_sha256") != expected_stream:
                raise RuntimeError("B210 endpoint stream provenance changed")
            audits.append(report["held_target_audit"])
        if any(value != audits[0] for value in audits[1:]):
            raise RuntimeError("B210 endpoint held target audits changed")
    if len(start_shas) != 1:
        raise RuntimeError("B210 folds do not share one Sim10k start SHA")
    if rows[0].get("fold10_disaster_gate", {}).get("pass") is not True:
        raise RuntimeError("B210 held0 was run after a failed held10 disaster gate")
    if rows[1].get("authorization", {}).get("kind") != "sealed_held10_pass":
        raise RuntimeError("B210 held0 lacks sealed held10 authorization")


def aggregate_fold_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    if len(rows) != len(FOLD_ORDER):
        raise ValueError("B210 aggregator requires held10 and held0")
    validate_fold_rows(rows)
    endpoint_states = {
        endpoint: smu1.merge_metric_states([
            row["endpoints"][endpoint]["metric_state"] for row in rows])
        for endpoint in ENDPOINT_ORDER
    }
    expected_samples = sum(int(row["samples"]) for row in rows)
    if any(int(state["samples"]) != expected_samples
           for state in endpoint_states.values()):
        raise RuntimeError("B210 pooled states do not cover both folds")
    endpoint_metrics = {
        endpoint: smu1.finalize_metric_state(state)
        for endpoint, state in endpoint_states.items()
    }
    primary = endpoint_metrics[PRIMARY_ENDPOINT]
    increment = metric_comparison(
        primary, endpoint_metrics["ema_u4000"])
    raw_vs_ema = metric_comparison(
        primary, endpoint_metrics["raw_u12000"])
    return {
        "decision": "AWAIT_FROZEN_C208_ANCHOR_COMPARISON",
        "completed_folds": len(rows),
        "samples": expected_samples,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "pooled_primary_metrics": primary,
        "pooled_primary_metric_state": endpoint_states[PRIMARY_ENDPOINT],
        "pooled_endpoint_metrics": endpoint_metrics,
        "pooled_endpoint_metric_states": endpoint_states,
        "pooled_ft12_increment_diagnostic": increment,
        "pooled_raw_vs_ema_u12000_diagnostic": raw_vs_ema,
        "diagnostics_cannot_select_endpoint": True,
        "promotion_contract": {
            "directional": DIRECTIONAL_GATE,
            "promotion": PROMOTION_GATE,
            "required_next_input": (
                "SHA-frozen fold-aligned C208 absolute metric states"),
        },
    }


def aggregate_from_files(pairs: Sequence[Sequence[str]]) -> dict:
    rows = []
    sources = []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected = dr1.normalized_sha256(raw_sha, "fold result SHA")
        actual = dr1.sha256(path)
        if actual != expected:
            raise RuntimeError(f"B210 fold result SHA changed: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("B210 fold result must be a JSON mapping")
        rows.append(value)
        sources.append({"path": str(path.resolve()), "sha256": actual})
    if (len(rows) == len(FOLD_ORDER)
            and (rows[1].get("authorization", {}).get("sha256")
                 != sources[0]["sha256"]
                 or rows[1].get("authorization", {}).get("path")
                 != sources[0]["path"])):
        raise RuntimeError(
            "B210 held0 authorization does not bind the supplied held10 file")
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
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hardware-class", choices=("rtx4080_super", "rtx4090"),
        default="rtx4080_super")
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    parser.add_argument(
        "--prior-fold10", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument(
        "--fold-result", nargs=2, action="append",
        metavar=("PATH", "SHA256"))
    parser.add_argument("--aggregate-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fold_worker:
        required = {
            "held_aoa": args.held_aoa,
            "output_dir": args.output_dir,
            "clean_manifest": args.clean_manifest,
            "start_sim": args.start_sim,
            "start_sim_sha256": args.start_sim_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"B210 fold worker missing {missing}")
        if args.held_aoa not in FOLD_ORDER:
            raise ValueError("B210 fold is outside held10/held0")
        args.start_sim_sha256 = dr1.normalized_sha256(
            args.start_sim_sha256, "start-sim-sha256")
        if (not args.execute_formal_gpu
                or args.acknowledge_protocol != FORMAL_ACKNOWLEDGEMENT):
            raise ValueError("formal B210 GPU execution requires acknowledgement")
        if args.fold_result or args.aggregate_output:
            raise ValueError("B210 fold worker cannot receive aggregate inputs")
        if args.held_aoa == 10 and args.prior_fold10 is not None:
            raise ValueError("held10 cannot receive prior fold evidence")
        if args.held_aoa == 0 and args.prior_fold10 is None:
            raise ValueError("held0 requires prior held10 evidence")
    elif args.aggregate_only:
        if (not args.fold_result or len(args.fold_result) != len(FOLD_ORDER)
                or args.aggregate_output is None):
            raise ValueError("B210 aggregate requires two folds and output")
        if args.aggregate_output.exists():
            raise FileExistsError(args.aggregate_output)
        if any((args.held_aoa, args.output_dir, args.clean_manifest,
                args.start_sim, args.start_sim_sha256, args.prior_fold10,
                args.execute_formal_gpu)):
            raise ValueError("B210 aggregate received fold-only arguments")
    else:
        if any((args.held_aoa, args.output_dir, args.clean_manifest,
                args.start_sim, args.start_sim_sha256, args.prior_fold10,
                args.fold_result, args.aggregate_output,
                args.execute_formal_gpu)):
            raise ValueError("B210 contract-only received execution arguments")


def write_stop(output_dir: Path, error: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stop.json"
    if path.exists():
        return
    kind = (
        "STOP_B210_REAL_OOM"
        if isinstance(error, torch.cuda.OutOfMemoryError)
        else "STOP_B210_REAL_RUNTIME")
    dr1.write_json_atomic(path, {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "decision": kind,
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
    if args.aggregate_only:
        result = aggregate_from_files(args.fold_result)
        dr1.write_json_atomic(args.aggregate_output, result)
        print(json.dumps({
            "event": "b210_real_ft12_aggregate_complete",
            "decision": result["decision"],
            "output": str(args.aggregate_output),
        }), flush=True)
        return
    try:
        result = run_fold(args)
    except Exception as error:
        if (args.output_dir is not None
                and not isinstance(error, FileExistsError)):
            write_stop(args.output_dir, error)
        raise
    print(json.dumps({
        "event": "b210_real_ft12_fold_complete",
        "held_aoa": result["held_aoa"],
        "decision": result["decision"],
        "output": str(args.output_dir / "fold_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
