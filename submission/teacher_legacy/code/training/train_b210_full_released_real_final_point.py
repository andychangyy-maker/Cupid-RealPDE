#!/usr/bin/env python3
"""Train the final B210 point model on every released clean81 trajectory.

This is deliberately a thin, single-arm follow-up to
``train_b210_full_released_real_support.py``.  It keeps the frozen B210 Real4k
recipe and pure-Sim10k start, computes normalization from the historical
sparse prefix windows, and changes only optimizer support to every legal
stride-5 window in the complete released trajectories.

The program accepts no clean716 or evaluation-manifest input.  It writes raw
and FP32-EMA endpoints for later, fresh Native-UQ cross-fit/refit; neither
endpoint is itself a submission artifact.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import shutil
import time
from typing import Iterator

import torch

import train_b210_full_released_real_support as support


ft12 = support.ft12
small4k = support.small4k
sim_tke = support.sim_tke
dr1 = support.dr1
smu1 = support.smu1
npt1 = ft12.npt1

EXPERIMENT = "b210_full_released_real_final_point"
SCHEMA = "b210_full_released_real_final_point_v1"
ENDPOINT_SCHEMA = "b210_full_released_real_final_endpoint_v1"
MANIFEST_SEAL_SCHEMA = "b210_full_released_real_manifest_seal_v1"
ACCESS_SEAL_SCHEMA = "b210_full_released_real_access_seal_v1"
FORMAL_BRANCH = "codex/b210-full-released-real-final-point"
ACKNOWLEDGEMENT = "RUN_B210_FULL_RELEASED_REAL_FINAL_POINT_V1"
FORMAL_START_SHA256 = support.FORMAL_START_SHA256
SUPPORT_EVIDENCE_START_SHA256 = support.FORMAL_START_SHA256
MODEL_LINEAGE = "b210_pure_sim10k"
START_EXPECTED_ITERATION = 10_000
START_EXPECTED_STATUS = sim_tke.PASS_STATUS
START_MATCHES_SUPPORT_LINEAGE = True
COMPLETION_EVENT = "b210_full_released_real_final_point_complete"

PRIMARY_ENDPOINT = support.PRIMARY_ENDPOINT
DIAGNOSTIC_ENDPOINT = support.DIAGNOSTIC_ENDPOINT
ENDPOINT_ORDER = (DIAGNOSTIC_ENDPOINT, PRIMARY_ENDPOINT)
NO_HELD_AOA = -1
UPDATES = 4_000
BATCH_SIZE = 8
SEED = 4
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
OPTIMIZER_STRIDE = 5
NORMALIZER_STRIDE = 20
SCHEDULER_T_MAX = 12_000
EMA_DECAY = 0.999
DEFAULT_WALL_BUDGET_SECONDS = 2 * 60 * 60
MIN_FREE_BYTES = 3 * 1024**3

EXPECTED_TRAJECTORIES = 81
EXPECTED_NORMALIZER_WINDOWS = 2_544
EXPECTED_FULL_WINDOWS = 13_202
EXPECTED_TRAJECTORIES_BY_AOA = {0: 16, 5: 14, 10: 18, 15: 17, 20: 16}
EXPECTED_NORMALIZER_WINDOWS_BY_AOA = {
    0: 502, 5: 433, 10: 576, 15: 544, 20: 489}
EXPECTED_FULL_WINDOWS_BY_AOA = {
    0: 2_604, 5: 2_249, 10: 2_988, 15: 2_822, 20: 2_539}
PER_AOA_INVENTORY_FIELDS = (
    "trajectories_by_aoa",
    "normalizer_windows_by_aoa",
    "full_optimizer_windows_by_aoa",
)

AUGMENTATION = dict(support.AUGMENTATION)

CONTRACT = {
    "question": (
        "Train one final B210 point endpoint after the matched held-AoA "
        "support experiment, changing only Real optimizer support."),
    "method_evidence": {
        "schema": support.AGGREGATE_SCHEMA,
        "contract_sha256": support.CONTRACT_SHA256,
        "held_aoas": list(support.HELD_AOAS),
        "decision": "COMPLETE_TWO_FOLD_MATCHED_EVIDENCE_NO_SCORE_GATE",
        "score_gate": None,
        "adoption_decision": "manual; this runner applies no score threshold",
    },
    "start": {
        "kind": "same pure-Sim10k B210 used by matched support experiment",
        "checkpoint_sha256": FORMAL_START_SHA256,
        "caveat": (
            "The Sim soft-SPS scale used fixed Real statistics; no released "
            "Real H5 or Real-closure supervision entered the start."),
    },
    "data": {
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "trajectories": EXPECTED_TRAJECTORIES,
        "files_unique": True,
        "excluded": dr1.EXCLUDED_BASENAME,
        "optimizer": (
            "all legal complete-trajectory 20->20 windows, stride 5"),
        "optimizer_windows": EXPECTED_FULL_WINDOWS,
        "normalizer": "frozen sparse-prefix windows only, stride 20",
        "normalizer_windows": EXPECTED_NORMALIZER_WINDOWS,
        "clean716_manifest_accepted": False,
        "evaluation_manifest_accepted": False,
    },
    "training": {
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "ema": "FP32 EMA999 from update 0 through update 4000",
        "ema_decay": EMA_DECAY,
        "loss": (
            "unchanged native Real joint point/member/quantile/soft-SPS "
            "plus controller-only early TKE auxiliary"),
        "augmentation": AUGMENTATION,
        "endpoint_selection_allowed": False,
    },
    "outputs": {
        "primary": PRIMARY_ENDPOINT,
        "diagnostic_only": DIAGNOSTIC_ENDPOINT,
        "manifest_and_access_sealed": True,
        "fresh_native_uq_crossfit_refit_required": True,
        "submission_authorized": False,
    },
}
CONTRACT_SHA256 = dr1.canonical_sha256(CONTRACT)


def architecture_contract() -> tuple:
    """Return the exact deployable topology expected from the start artifact."""
    return (
        sim_tke.b210_config("candidate"),
        ft12.EXPECTED_PARAMETERS,
        ft12.EXPECTED_CONTROLLER_PARAMETERS,
    )


def _with_sha256(value: Mapping[str, object]) -> dict:
    body = dict(value)
    if "sha256" in body:
        raise ValueError("seal body already contains sha256")
    return {**body, "sha256": dr1.canonical_sha256(body)}


def _validate_self_seal(value: Mapping[str, object], label: str) -> None:
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != dr1.canonical_sha256(body):
        raise RuntimeError(f"{label} canonical SHA changed")


def released_inventory(rows: Sequence[Mapping[str, object]]) -> dict:
    """Derive the frozen counts without opening an H5 file."""
    files = [str(row["file"]) for row in rows]
    trajectories = Counter(int(row["aoa"]) for row in rows)
    normalizer = Counter()
    full = Counter()
    for row in rows:
        aoa = int(row["aoa"])
        frames = int(row["frames"])
        normalizer[aoa] += len(row["train_starts"])
        full[aoa] += len(range(0, frames - 40 + 1, OPTIMIZER_STRIDE))
    return {
        "trajectories": len(rows),
        "unique_files": len(set(files)),
        "excluded_present": dr1.EXCLUDED_BASENAME in files,
        "trajectories_by_aoa": dict(sorted(trajectories.items())),
        "normalizer_windows": sum(normalizer.values()),
        "normalizer_windows_by_aoa": dict(sorted(normalizer.items())),
        "full_optimizer_windows": sum(full.values()),
        "full_optimizer_windows_by_aoa": dict(sorted(full.items())),
    }


def validate_released_inventory(rows: Sequence[Mapping[str, object]]) -> dict:
    inventory = released_inventory(rows)
    expected = {
        "trajectories": EXPECTED_TRAJECTORIES,
        "unique_files": EXPECTED_TRAJECTORIES,
        "excluded_present": False,
        "trajectories_by_aoa": EXPECTED_TRAJECTORIES_BY_AOA,
        "normalizer_windows": EXPECTED_NORMALIZER_WINDOWS,
        "normalizer_windows_by_aoa": EXPECTED_NORMALIZER_WINDOWS_BY_AOA,
        "full_optimizer_windows": EXPECTED_FULL_WINDOWS,
        "full_optimizer_windows_by_aoa": EXPECTED_FULL_WINDOWS_BY_AOA,
    }
    if inventory != expected:
        raise RuntimeError(
            f"released clean81 full-trajectory inventory changed: {inventory}")
    return inventory


class FinalFullAccessAudit:
    """Allow one complete read of the exact clean81 manifest and then close it."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.events: list[dict] = []
        self.endpoints: dict[str, dict] = {}
        self.data_closed = False

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if phase != "fit":
            raise RuntimeError("final full point runner has no held/eval H5 phase")
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded duplicate reached final full opener")
        if self.data_closed or self.endpoints:
            raise RuntimeError("released Real H5 opened after data access closed")
        if filename in self.opened:
            raise RuntimeError(f"released Real H5 opened twice: {filename}")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if phase != "fit" or filename in self.opened:
            raise RuntimeError("invalid final full H5 open record")
        self.opened.append(filename)
        self.events.append({
            "file": filename,
            "aoa": int(row["aoa"]),
            "phase": "fit",
            "slice": "u/v[:,::2,::2]",
            "max_read_frame_exclusive": int(row["frames"]),
        })

    def close_data(self, rows: Sequence[Mapping[str, object]]) -> dict:
        expected = [str(row["file"]) for row in rows]
        if self.data_closed:
            raise RuntimeError("final full data access closed twice")
        if self.opened != expected or len(set(self.opened)) != len(expected):
            raise RuntimeError("final full H5 allowlist was not read exactly once")
        self.data_closed = True
        return _with_sha256({
            "schema": ACCESS_SEAL_SCHEMA,
            "opened_file_allowlist": list(self.opened),
            "opened_file_allowlist_sha256": dr1.canonical_sha256(self.opened),
            "events_sha256": dr1.canonical_sha256(self.events),
            "trajectories": len(self.opened),
            "allowlist_exact_and_ordered": True,
            "complete_trajectory_arrays_read": True,
            "slice": "u/v[:,::2,::2]",
            "excluded_opened": False,
            "held_or_evaluation_phase_opened": False,
            "clean716_manifest_read": False,
            "data_closed_before_optimizer": True,
        })

    def seal_endpoint(self, endpoint: str,
                      value: Mapping[str, object]) -> None:
        if not self.data_closed:
            raise RuntimeError("endpoint sealed before data access was closed")
        if endpoint not in ENDPOINT_ORDER or endpoint in self.endpoints:
            raise RuntimeError(f"invalid or repeated endpoint seal: {endpoint}")
        path = Path(str(value["path"]))
        if (not path.is_file()
                or dr1.sha256(path) != value.get("file_sha256")):
            raise RuntimeError(f"endpoint file was not SHA-sealed: {endpoint}")
        for key in (
                "model_state_sha256", "model_config_sha256",
                "normalizer_sha256", "sample_stream_sha256"):
            dr1.normalized_sha256(str(value[key]), f"{endpoint} {key}")
        self.endpoints[endpoint] = dict(value)

    def report(self, rows: Sequence[Mapping[str, object]],
               access_seal: Mapping[str, object]) -> dict:
        _validate_self_seal(access_seal, "access seal")
        expected = [str(row["file"]) for row in rows]
        return {
            "opened_file_allowlist": list(self.opened),
            "expected_file_allowlist": expected,
            "allowlist_exact_and_ordered": self.opened == expected,
            "files_unique": len(set(self.opened)) == len(self.opened),
            "events": list(self.events),
            "data_closed_before_optimizer": self.data_closed,
            "H5_open_after_data_close": False,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "clean716_manifest_read": False,
            "endpoint_seals": dict(self.endpoints),
            "endpoint_set_exact": tuple(self.endpoints) == ENDPOINT_ORDER,
            "access_seal": dict(access_seal),
        }


@contextmanager
def endpoint_contract() -> Iterator[None]:
    """Temporarily bind the reused Real4k loop to the final endpoint identity."""
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": PRIMARY_ENDPOINT,
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": SCHEDULER_T_MAX,
        "HELD_AOA": NO_HELD_AOA,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(small4k, name, value)


def validate_training_recipe() -> dict:
    observed = {
        "updates": small4k.UPDATES,
        "batch_size": small4k.BATCH_SIZE,
        "seed": small4k.SEED,
        "lr": small4k.LR,
        "weight_decay": small4k.WEIGHT_DECAY,
        "grad_clip": small4k.GRAD_CLIP,
        "scheduler_t_max": small4k.SCHEDULER_T_MAX,
        "ema_decay": small4k.EMA_DECAY,
    }
    expected = {
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "scheduler_t_max": SCHEDULER_T_MAX,
        "ema_decay": EMA_DECAY,
    }
    if observed != expected:
        raise RuntimeError(f"reused B210 Real4k recipe changed: {observed}")
    return observed


def validate_support_aggregate(path: Path, expected_sha256: str) -> dict:
    expected_sha = dr1.normalized_sha256(
        expected_sha256, "full-support aggregate SHA")
    actual_sha = dr1.sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError("full-support aggregate file SHA changed")
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, Mapping):
        raise TypeError("full-support aggregate must be a JSON mapping")
    required = {
        "schema": support.AGGREGATE_SCHEMA,
        "experiment": support.EXPERIMENT,
        "contract_sha256": support.CONTRACT_SHA256,
        "held_aoas": list(support.HELD_AOAS),
        "start_checkpoint_sha256": SUPPORT_EVIDENCE_START_SHA256,
        "held_full_windows": 1_399,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "raw_endpoint_role": "diagnostic_only_no_endpoint_selection",
        "score_gate": None,
        "decision": "COMPLETE_TWO_FOLD_MATCHED_EVIDENCE_NO_SCORE_GATE",
        "submission_authorized": False,
    }
    if any(row.get(key) != value for key, value in required.items()):
        raise RuntimeError("full-support aggregate identity changed")
    comparison = row.get("primary_comparison")
    if not isinstance(comparison, Mapping):
        raise RuntimeError("full-support aggregate lacks primary comparison")
    for key in (
            "point_final_contribution_delta",
            "non_time_final_contribution_delta"):
        value = comparison.get(key)
        if not isinstance(value, (int, float)) or not torch.isfinite(
                torch.as_tensor(value)):
            raise RuntimeError("full-support aggregate comparison is invalid")
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "schema": row["schema"],
        "contract_sha256": row["contract_sha256"],
        "held_aoas": list(row["held_aoas"]),
        "primary_endpoint": row["primary_endpoint"],
        "primary_comparison": dict(comparison),
        "score_gate": None,
        "method_adoption_was_manual": True,
    }


def make_manifest_seal(manifest_report: Mapping[str, object],
                       inventory: Mapping[str, object]) -> dict:
    if (manifest_report.get("semantic_sha256")
            != ft12.FORMAL_MANIFEST_SEMANTIC_SHA256
            or int(manifest_report.get("windows", -1))
            != EXPECTED_NORMALIZER_WINDOWS
            or manifest_report.get("excluded") != dr1.EXCLUDED_BASENAME):
        raise RuntimeError("clean81 manifest report changed")
    json_inventory = dict(inventory)
    for field in PER_AOA_INVENTORY_FIELDS:
        values = json_inventory.get(field)
        if not isinstance(values, Mapping):
            raise TypeError(f"manifest inventory {field} is not a mapping")
        json_inventory[field] = {
            str(int(aoa)): int(count)
            for aoa, count in sorted(
                values.items(), key=lambda item: int(item[0]))
        }
    return _with_sha256({
        "schema": MANIFEST_SEAL_SCHEMA,
        "file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "semantic_sha256": ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "excluded": dr1.EXCLUDED_BASENAME,
        # JSON object keys are strings.  Normalize before hashing so the seal
        # remains self-validating after point_stage_result.json round-trips.
        "inventory": json_inventory,
        "normalizer_support": "manifest sparse-prefix train_starts only",
        "optimizer_support": "complete released trajectories stride5",
    })


def endpoint_common(*, endpoint: str, primary: bool,
                    state: Mapping[str, torch.Tensor], model_config: Mapping,
                    start_report: Mapping[str, object],
                    normalizer_report: Mapping[str, object],
                    stream_sha256: str,
                    manifest_seal: Mapping[str, object],
                    access_seal: Mapping[str, object],
                    ema: Mapping[str, object] | None) -> dict:
    _validate_self_seal(manifest_seal, "manifest seal")
    _validate_self_seal(access_seal, "access seal")
    state_sha = dr1.model_state_sha256(state)
    return {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "training_scope": "all_clean81_full13202",
        "iteration": UPDATES,
        "endpoint": endpoint,
        "primary": primary,
        "held_aoa": None,
        "model_state_dict": OrderedDict(
            (name, value.detach().cpu()) for name, value in state.items()),
        "model_state_sha256": state_sha,
        "model_config": dict(model_config),
        "start": dict(start_report),
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": str(normalizer_report["sha256"]),
        "sample_stream_sha256": stream_sha256,
        "manifest_seal": dict(manifest_seal),
        "access_seal": dict(access_seal),
        "optimizer_windows": EXPECTED_FULL_WINDOWS,
        "optimizer_stride": OPTIMIZER_STRIDE,
        "normalizer_windows": EXPECTED_NORMALIZER_WINDOWS,
        "normalizer_stride": NORMALIZER_STRIDE,
        "ema": None if ema is None else dict(ema),
        "optimizer_state_saved": False,
        "endpoint_selection_allowed": False,
        "fresh_native_uq_crossfit_refit_required": True,
        "submission_authorized": False,
    }


def _write_endpoint(path: Path, payload: Mapping[str, object]) -> dict:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite endpoint: {path}")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return {
        "schema": ENDPOINT_SCHEMA,
        "endpoint": payload["endpoint"],
        "primary": payload["primary"],
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": payload["model_state_sha256"],
        "model_config_sha256": dr1.canonical_sha256(
            payload["model_config"]),
        "normalizer_sha256": payload["normalizer_sha256"],
        "sample_stream_sha256": payload["sample_stream_sha256"],
        "manifest_seal_sha256": payload["manifest_seal"]["sha256"],
        "access_seal_sha256": payload["access_seal"]["sha256"],
        "iteration": UPDATES,
        "optimizer_windows": EXPECTED_FULL_WINDOWS,
        "normalizer_windows": EXPECTED_NORMALIZER_WINDOWS,
    }


def save_raw_endpoint(path: Path, model, *,
                      start_report: Mapping[str, object],
                      normalizer_report: Mapping[str, object],
                      stream_sha256: str,
                      manifest_seal: Mapping[str, object],
                      access_seal: Mapping[str, object]) -> dict:
    state = npt1.ensemble_train.common.checkpoint_state(model)
    payload = endpoint_common(
        endpoint=DIAGNOSTIC_ENDPOINT, primary=False, state=state,
        model_config=model.architecture_config(), start_report=start_report,
        normalizer_report=normalizer_report, stream_sha256=stream_sha256,
        manifest_seal=manifest_seal, access_seal=access_seal, ema=None)
    seal = _write_endpoint(path, payload)
    del payload, state
    return seal


def finalize_ema_endpoint(precursor: Mapping[str, object], *,
                          start_report: Mapping[str, object],
                          normalizer_report: Mapping[str, object],
                          stream_sha256: str,
                          manifest_seal: Mapping[str, object],
                          access_seal: Mapping[str, object]) -> dict:
    """Rebind the generic trainer endpoint without changing its EMA tensors."""
    path = Path(str(precursor["path"]))
    precursor_file_sha = dr1.sha256(path)
    if precursor_file_sha != precursor.get("file_sha256"):
        raise RuntimeError("generic EMA precursor file changed")
    value = torch.load(path, map_location="cpu", weights_only=False)
    state = value.get("model_state_dict")
    if (value.get("schema") != ENDPOINT_SCHEMA
            or value.get("experiment") != EXPERIMENT
            or value.get("phase") != "real"
            or value.get("endpoint") != PRIMARY_ENDPOINT
            or value.get("primary") is not True
            or int(value.get("iteration", -1)) != UPDATES
            or int(value.get("held_aoa", 0)) != NO_HELD_AOA
            or not isinstance(state, Mapping)):
        raise RuntimeError("generic EMA precursor identity changed")
    state_sha = dr1.model_state_sha256(state)
    if (state_sha != precursor.get("model_state_sha256")
            or state_sha != value.get("model_state_sha256")):
        raise RuntimeError("generic EMA precursor tensor state changed")
    if (value.get("model_config") != dict(value["model_config"])
            or value.get("normalizer") != normalizer_report["values"]
            or value.get("normalizer_sha256") != normalizer_report["sha256"]
            or value.get("sample_stream_sha256") != stream_sha256
            or value.get("start") != dict(start_report)):
        raise RuntimeError("generic EMA precursor provenance changed")
    ema = value.get("ema")
    if (not isinstance(ema, Mapping)
            or ema.get("schema") != small4k.ema_lib.EMA_SEAL_SCHEMA
            or ema.get("update_order") != small4k.ema_lib.UPDATE_ORDER_CONTRACT
            or int(ema.get("updates", -1)) != UPDATES
            or float(ema.get("decay", -1.0)) != EMA_DECAY
            or ema.get("shadow_dtype") != "torch.float32"
            or ema.get("materialized_model_state_sha256") != state_sha):
        raise RuntimeError("generic EMA precursor EMA seal changed")
    payload = endpoint_common(
        endpoint=PRIMARY_ENDPOINT, primary=True, state=state,
        model_config=value["model_config"], start_report=start_report,
        normalizer_report=normalizer_report, stream_sha256=stream_sha256,
        manifest_seal=manifest_seal, access_seal=access_seal, ema=ema)
    payload["generic_precursor_file_sha256"] = precursor_file_sha
    temporary = path.with_name(path.name + ".final.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(payload, temporary)
    temporary.replace(path)
    seal = {
        "schema": ENDPOINT_SCHEMA,
        "endpoint": PRIMARY_ENDPOINT,
        "primary": True,
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": dr1.canonical_sha256(
            payload["model_config"]),
        "normalizer_sha256": payload["normalizer_sha256"],
        "sample_stream_sha256": stream_sha256,
        "manifest_seal_sha256": manifest_seal["sha256"],
        "access_seal_sha256": access_seal["sha256"],
        "iteration": UPDATES,
        "optimizer_windows": EXPECTED_FULL_WINDOWS,
        "normalizer_windows": EXPECTED_NORMALIZER_WINDOWS,
    }
    del value, state, payload
    return seal


def validate_endpoint(path: Path, seal: Mapping[str, object], *,
                      endpoint: str, primary: bool,
                      manifest_seal: Mapping[str, object],
                      access_seal: Mapping[str, object],
                      instantiate_model: bool = False) -> dict:
    if dr1.sha256(path) != seal.get("file_sha256"):
        raise RuntimeError(f"sealed endpoint file changed: {endpoint}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    required = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "training_scope": "all_clean81_full13202",
        "iteration": UPDATES,
        "endpoint": endpoint,
        "primary": primary,
        "held_aoa": None,
        "optimizer_windows": EXPECTED_FULL_WINDOWS,
        "optimizer_stride": OPTIMIZER_STRIDE,
        "normalizer_windows": EXPECTED_NORMALIZER_WINDOWS,
        "normalizer_stride": NORMALIZER_STRIDE,
        "optimizer_state_saved": False,
        "endpoint_selection_allowed": False,
        "fresh_native_uq_crossfit_refit_required": True,
        "submission_authorized": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"endpoint consumer identity changed: {endpoint}")
    if (not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]
            or payload.get("model_state_sha256")
            != seal["model_state_sha256"]):
        raise RuntimeError(f"endpoint state changed: {endpoint}")
    if any(tensor.is_floating_point() and tensor.dtype != torch.float32
           for tensor in state.values() if isinstance(tensor, torch.Tensor)):
        raise TypeError(f"endpoint state is not strict FP32: {endpoint}")
    expected_config, expected_parameters, expected_controller = (
        architecture_contract())
    config = ft12.validate_b210_model_config(
        payload["model_config"], expected_config)
    if (dr1.canonical_sha256(payload["model_config"])
            != seal["model_config_sha256"]
            or payload.get("normalizer_sha256")
            != seal["normalizer_sha256"]
            or dr1.canonical_sha256(payload.get("normalizer", {}))
            != seal["normalizer_sha256"]
            or payload.get("sample_stream_sha256")
            != seal["sample_stream_sha256"]
            or payload.get("start", {}).get("checkpoint_sha256")
            != FORMAL_START_SHA256
            or payload.get("manifest_seal") != dict(manifest_seal)
            or payload.get("access_seal") != dict(access_seal)):
        raise RuntimeError(f"endpoint provenance changed: {endpoint}")
    _validate_self_seal(payload["manifest_seal"], "endpoint manifest seal")
    _validate_self_seal(payload["access_seal"], "endpoint access seal")
    ema = payload.get("ema")
    if primary:
        if (not isinstance(ema, Mapping)
                or ema.get("schema") != small4k.ema_lib.EMA_SEAL_SCHEMA
                or ema.get("update_order")
                != small4k.ema_lib.UPDATE_ORDER_CONTRACT
                or int(ema.get("updates", -1)) != UPDATES
                or float(ema.get("decay", -1.0)) != EMA_DECAY
                or ema.get("shadow_dtype") != "torch.float32"
                or ema.get("materialized_model_state_sha256")
                != seal["model_state_sha256"]):
            raise RuntimeError("EMA endpoint seal changed")
    elif ema is not None:
        raise RuntimeError("raw diagnostic endpoint unexpectedly contains EMA")
    parameter_count = None
    if instantiate_model:
        model = small4k.EnsembleUNet2d(config).float()
        model.load_state_dict(state, strict=True)
        parameter_count = sum(value.numel() for value in model.parameters())
        controller_count = sum(
            value.numel() for value in model.energy_controller.parameters())
        if (parameter_count != expected_parameters
                or controller_count != expected_controller):
            raise RuntimeError("strict B210 endpoint parameter count changed")
        del model
    del payload, state
    return {
        "endpoint": endpoint,
        "file_sha256": seal["file_sha256"],
        "model_state_sha256": seal["model_state_sha256"],
        "strict_fp32_state": True,
        "strict_model_load": instantiate_model,
        "parameter_count": parameter_count,
    }


def _disk_preflight(output_dir: Path) -> dict:
    probe = output_dir.parent.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError("no existing parent for final point output")
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"final point output has {free} free bytes; requires {MIN_FREE_BYTES}")
    return {
        "probe": str(probe),
        "free_bytes_before_run": free,
        "minimum_free_bytes": MIN_FREE_BYTES,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal final full-trajectory point training requires CUDA")
    identity = dr1.git_identity()
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError(f"formal run requires branch {FORMAL_BRANCH}")
    if identity["status_porcelain"]:
        raise RuntimeError("formal run requires a clean scientific worktree")
    recipe = validate_training_recipe()
    evidence = validate_support_aggregate(
        args.full_support_aggregate,
        args.full_support_aggregate_sha256)
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    inventory = validate_released_inventory(rows)
    start_sha = dr1.sha256(args.start_sim)
    if (start_sha != FORMAL_START_SHA256
            or start_sha != dr1.normalized_sha256(
                args.start_sim_sha256, "start Sim10k SHA")):
        raise RuntimeError("final point frozen Sim start changed")
    disk = _disk_preflight(args.output_dir)

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    audit = FinalFullAccessAudit()
    trajectories, metadata = support.load_full_rows(
        args.real_dir, rows, "fit", audit)
    access_seal = audit.close_data(rows)

    normalizer_windows = support.manifest_prefix_windows(rows)
    full_windows = support.support_windows(
        trajectories, metadata, support="full", stride=OPTIMIZER_STRIDE)
    if (len(normalizer_windows) != EXPECTED_NORMALIZER_WINDOWS
            or len(full_windows) != EXPECTED_FULL_WINDOWS):
        raise RuntimeError("final point materialized window inventory changed")
    observed_full_by_aoa = Counter(
        int(metadata[window.trajectory]["aoa"]) for window in full_windows)
    if dict(sorted(observed_full_by_aoa.items())) != EXPECTED_FULL_WINDOWS_BY_AOA:
        raise RuntimeError("materialized full-window AoA inventory changed")
    manifest_seal = make_manifest_seal(manifest_report, inventory)

    sufficient = smu1.window_sufficient_statistics(
        trajectories, normalizer_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(EXPECTED_NORMALIZER_WINDOWS))
    normalizer = smu1.FoldNormalizer(statistics, device)
    normalizer_report = ft12.normalizer_payload(normalizer)
    dataset = npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, full_windows)
    schedule = support.uniform_schedule(full_windows)

    npt1.ensemble_train.seed_everything(SEED)
    expected_config, expected_parameters, expected_controller = (
        architecture_contract())
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=expected_config,
        expected_parameters=expected_parameters,
        expected_controller_parameters=expected_controller,
        expected_iteration=START_EXPECTED_ITERATION,
        expected_status=START_EXPECTED_STATUS)
    initial_state_sha = dr1.model_state_sha256(model.state_dict())
    capture = support.EmaSealCapture()
    with endpoint_contract():
        training, precursor = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            args.output_dir, start_report, capture,
            input_augmentation=AUGMENTATION,
            endpoint_filename=f"{PRIMARY_ENDPOINT}.pth",
            expected_parameters=expected_parameters,
            scheduler_t_max=SCHEDULER_T_MAX)
    if capture.value != precursor:
        raise RuntimeError("final point EMA precursor was not captured exactly")
    if training.get("ema_seal", {}).get(
            "raw_model_unchanged_by_seal") is not True:
        raise RuntimeError("EMA materialization changed the final raw model")

    raw_seal = save_raw_endpoint(
        args.output_dir / f"{DIAGNOSTIC_ENDPOINT}.pth", model,
        start_report=start_report, normalizer_report=normalizer_report,
        stream_sha256=schedule["sha256"], manifest_seal=manifest_seal,
        access_seal=access_seal)
    ema_seal = finalize_ema_endpoint(
        precursor, start_report=start_report,
        normalizer_report=normalizer_report,
        stream_sha256=schedule["sha256"], manifest_seal=manifest_seal,
        access_seal=access_seal)
    audit.seal_endpoint(DIAGNOSTIC_ENDPOINT, raw_seal)
    audit.seal_endpoint(PRIMARY_ENDPOINT, ema_seal)
    del model, dataset, sufficient, trajectories
    gc.collect()
    torch.cuda.empty_cache()

    validations = {
        DIAGNOSTIC_ENDPOINT: validate_endpoint(
            Path(raw_seal["path"]), raw_seal,
            endpoint=DIAGNOSTIC_ENDPOINT, primary=False,
            manifest_seal=manifest_seal, access_seal=access_seal,
            instantiate_model=True),
        PRIMARY_ENDPOINT: validate_endpoint(
            Path(ema_seal["path"]), ema_seal,
            endpoint=PRIMARY_ENDPOINT, primary=True,
            manifest_seal=manifest_seal, access_seal=access_seal,
            instantiate_model=True),
    }
    access_report = audit.report(rows, access_seal)
    if not (
            access_report["allowlist_exact_and_ordered"]
            and access_report["files_unique"]
            and access_report["data_closed_before_optimizer"]
            and not access_report["H5_open_after_data_close"]
            and not access_report["excluded_opened"]
            and not access_report["clean716_manifest_read"]
            and access_report["endpoint_set_exact"]):
        raise RuntimeError("final point data/access seal contract failed")
    if time.perf_counter() > deadline:
        raise TimeoutError("final full-trajectory point run exceeded wall budget")

    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "status": "POINT_STAGE_ONLY_NOT_SUBMITTABLE",
        "method_evidence": evidence,
        "manifest": {
            **manifest_report,
            "file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        },
        "manifest_seal": manifest_seal,
        "access_audit": access_report,
        "inventory": inventory,
        "normalizer": {
            **normalizer_report,
            "kind": "frozen clean81 sparse-prefix stride20",
            "windows": EXPECTED_NORMALIZER_WINDOWS,
            "full_tail_in_normalizer": False,
        },
        "optimizer_support": {
            "kind": "all clean81 complete-trajectory stride5 windows",
            "windows": EXPECTED_FULL_WINDOWS,
            "stride": OPTIMIZER_STRIDE,
            "sample_stream_sha256": schedule["sha256"],
            "ordered_key_sha256": smu1.ordered_held_key_digest(
                range(EXPECTED_FULL_WINDOWS), full_windows, metadata),
        },
        "start": {
            **start_report,
            "file_sha256": start_sha,
            "initial_model_state_sha256": initial_state_sha,
            "model_lineage": MODEL_LINEAGE,
            "same_start_lineage_as_matched_support": (
                START_MATCHES_SUPPORT_LINEAGE),
        },
        "recipe": recipe,
        "training": training,
        "endpoints": {
            DIAGNOSTIC_ENDPOINT: {
                "role": "diagnostic_only_no_endpoint_selection",
                "seal": raw_seal,
                "validation": validations[DIAGNOSTIC_ENDPOINT],
            },
            PRIMARY_ENDPOINT: {
                "role": "fixed_primary",
                "seal": ema_seal,
                "validation": validations[PRIMARY_ENDPOINT],
            },
        },
        "primary_endpoint": PRIMARY_ENDPOINT,
        "endpoint_selection_allowed": False,
        "fresh_native_uq_crossfit_refit_required": True,
        "submission_authorized": False,
        "git": identity,
        "disk_preflight": disk,
        "wall_budget_seconds": args.wall_budget_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": "COMPLETE_FINAL_FULL_REAL_POINT_REQUIRES_FRESH_NATIVE_UQ",
    }
    dr1.write_json_atomic(args.output_dir / "point_stage_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract-only", action="store_true")
    mode.add_argument("--run-final-full", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--full-support-aggregate", type=Path)
    parser.add_argument("--full-support-aggregate-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--wall-budget-seconds", type=float,
        default=DEFAULT_WALL_BUDGET_SECONDS)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    execution = (
        args.output_dir, args.real_dir, args.clean_manifest, args.start_sim,
        args.start_sim_sha256, args.full_support_aggregate,
        args.full_support_aggregate_sha256)
    if args.contract_only:
        if (any(value is not None for value in execution)
                or args.execute_formal_gpu or args.acknowledge_protocol):
            raise ValueError("contract-only mode received execution arguments")
        return
    if any(value is None for value in execution):
        raise ValueError("final full point runner is missing required inputs")
    if (not args.execute_formal_gpu
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise ValueError("final full point runner requires exact acknowledgement")
    if args.wall_budget_seconds <= 0:
        raise ValueError("wall budget must be positive")
    for path in (
            args.real_dir, args.clean_manifest, args.start_sim,
            args.full_support_aggregate):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.real_dir.is_dir():
        raise NotADirectoryError(args.real_dir)
    args.start_sim_sha256 = dr1.normalized_sha256(
        args.start_sim_sha256, "start Sim10k SHA")
    if args.start_sim_sha256 != FORMAL_START_SHA256:
        raise ValueError("final full point runner requires its frozen Sim start")
    args.full_support_aggregate_sha256 = dr1.normalized_sha256(
        args.full_support_aggregate_sha256, "full-support aggregate SHA")


def write_stop(output_dir: Path, error: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stop.json"
    if path.exists():
        return
    if isinstance(error, torch.cuda.OutOfMemoryError):
        decision = "STOP_FINAL_FULL_REAL_POINT_OOM"
    elif isinstance(error, TimeoutError):
        decision = "STOP_FINAL_FULL_REAL_POINT_WALL_BUDGET"
    else:
        decision = "STOP_FINAL_FULL_REAL_POINT_RUNTIME"
    dr1.write_json_atomic(path, {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "decision": decision,
        "error_type": type(error).__name__,
        "error": str(error),
        "submission_authorized": False,
    })


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "endpoint_schema": ENDPOINT_SCHEMA,
            "contract_sha256": CONTRACT_SHA256,
            "contract": CONTRACT,
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
        "event": COMPLETION_EVENT,
        "decision": result["decision"],
        "primary_endpoint": result["endpoints"][PRIMARY_ENDPOINT]["seal"],
        "result": str(args.output_dir / "point_stage_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
