#!/usr/bin/env python3
"""Paired ST-B210 Sim10k -> Real4k lifecycle without score selection.

The temporal-causal candidate and the framewise spatial-x control start from
bitwise-identical random states.  Both consume the same released Sim100
stride-20 stream and the same clean81 Real stride-5 stream.  The only model
configuration difference is ``interaction_mode``.  The two Sim terminals and
all four Real raw/EMA terminals are immutable before the clean716 manifest or
field values are read.

This runner intentionally has no UQ path, score gate, early stopping, best
checkpoint selection, packaging, or submission logic.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
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

import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real_long  # noqa: E402
import train_pyramid_causal3d_sim_long as sim_long  # noqa: E402
import train_st_b210_loao_screen as st_screen  # noqa: E402
from models.spatiotemporal_b210 import (  # noqa: E402
    SpatiotemporalB210,
    SpatiotemporalB210Config,
)


SCHEMA = "st_b210_paired_sim10k_real4k_lifecycle_v1"
ENDPOINT_SCHEMA = "st_b210_lifecycle_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_ST_B210_PAIRED_SIM10K_REAL4K_LIFECYCLE_V1"
ARMS = st_screen.ARMS
ARM_MODE = st_screen.ARM_MODE
ENDPOINTS = ("raw", "ema")

SIM_UPDATES = sim_long.UPDATES
SIM_BATCH_SIZE = sim_long.BATCH_SIZE
SIM_SCHEDULE_SHA256 = sim_long.SAMPLE_INDEX_SHA256
REAL_UPDATES = real_long.UPDATES
REAL_BATCH_SIZE = real_long.BATCH_SIZE
REAL_SCHEDULE_SHA256 = (
    "9381a97fc114e4234a889d686158533d0088701676b683d268d12d4203f1be43")
EXPECTED_PARAMETERS = st_screen.EXPECTED_PARAMETERS
EXPECTED_STATE_TENSORS = st_screen.EXPECTED_STATE_TENSORS
OFFICIAL_POINT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "checkpoint-smoke", "formal"),
        default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-checkpoint-every", type=int, default=2_000)
    parser.add_argument("--real-checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args(argv)


def lifecycle_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "acknowledgement": ACKNOWLEDGEMENT,
        "arms": {arm: ARM_MODE[arm] for arm in ARMS},
        "only_model_variable": "interaction_mode",
        "control_semantics": (
            "encoder_local_axis_mixers_have_no_cross_frame_interaction; "
            "the_shared_learned_20x20_observed_to_future_projection_still_reads_all_history"),
        "effective_grid": [32, 64],
        "initialization": {
            "seed": st_screen.SEED,
            "bitwise_identical_between_arms": True,
            "expected_parameters_each": EXPECTED_PARAMETERS,
            "expected_state_tensors_each": EXPECTED_STATE_TENSORS,
        },
        "sim": {
            "data": "competition_released_Sim100",
            "stride": sim_long.WINDOW_STRIDE,
            "windows": sim_long.EXPECTED_WINDOWS,
            "updates": SIM_UPDATES,
            "batch_size": SIM_BATCH_SIZE,
            "optimizer": "AdamW",
            "lr": sim_long.LR,
            "weight_decay": sim_long.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": SIM_UPDATES,
            "loss": "accuracy_1_4_1_plus_.25_all_channel_normalized_mse",
            "augmentation": "none",
            "ema": False,
            "sample_stream_sha256": SIM_SCHEDULE_SHA256,
            "endpoint": "raw_u10000_only",
        },
        "real": {
            "data": "clean81_optimizer_visible_prefix",
            "normalizer_stride": real_long.NORMALIZER_STRIDE,
            "optimizer_stride": real_long.OPTIMIZER_STRIDE,
            "optimizer_windows": real_long.EXPECTED_OPTIMIZER_WINDOWS,
            "updates": REAL_UPDATES,
            "batch_size": REAL_BATCH_SIZE,
            "optimizer": "AdamW",
            "lr": real_long.LR,
            "weight_decay": real_long.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": real_long.SCHEDULER_T_MAX,
            "loss": real_long.real_training_contract(
                updates=REAL_UPDATES, batch_size=REAL_BATCH_SIZE,
                scheduler_t_max=real_long.SCHEDULER_T_MAX)["loss"],
            "augmentation": dict(real_long.AUGMENTATION),
            "ema_decay": real_long.EMA_DECAY,
            "sample_stream_sha256": REAL_SCHEDULE_SHA256,
            "endpoints_each": list(ENDPOINTS),
            "each_arm_starts_from_its_own_sim_terminal": True,
            "unmeasured_pressure": (
                "exact_zero_in_dataset_and_after_input_augmentation"),
        },
        "evaluation": {
            "clean716_first_read_after_all_four_real_endpoint_seals": True,
            "endpoints": [
                "sim_raw_zero_shot", "real_raw_u4000", "real_ema_u4000"],
            "metrics": [*OFFICIAL_POINT_KEYS,
                        "official_point_contribution_46_10_9"],
            "by_aoa": [0, 5, 10, 15, 20],
            "candidate_control_survival": True,
        },
        "score_gate": None,
        "early_stopping": False,
        "checkpoint_selection": False,
        "uq": False,
        "packaging": False,
        "submission_authorized": False,
    }


def _state_snapshot(
        model_or_state: nn.Module | Mapping[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
    state = (model_or_state.state_dict()
             if isinstance(model_or_state, nn.Module) else model_or_state)
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _normalizer_payload(normalizer) -> dict[str, torch.Tensor]:
    return {name: getattr(normalizer, name).detach().cpu().clone()
            for name in ("mean_input", "mean_target", "std_input", "std_target")}


def _normalizer_from_payload(
        values: Mapping[str, torch.Tensor], device: torch.device):
    required = {"mean_input", "mean_target", "std_input", "std_target"}
    if set(values) != required:
        raise RuntimeError("endpoint normalizer tensor inventory changed")
    return base.ChannelNormalizer(
        *(values[name].detach().cpu() for name in (
            "mean_input", "mean_target", "std_input", "std_target")),
        device=device)


def _endpoint_config(model: nn.Module) -> dict[str, object]:
    value = model.architecture_config()
    if not isinstance(value, Mapping):
        raise TypeError("ST-B210 architecture_config must be a mapping")
    result = dict(value)
    if (result.get("architecture") != "spatiotemporal_b210"
            or result.get("flattened_history_bypass") is not False
            or result.get("interaction_mode") != model.interaction_mode):
        raise RuntimeError("ST-B210 endpoint architecture contract changed")
    return result


def seal_endpoint(
        path: Path, *, model: nn.Module,
        state: Mapping[str, torch.Tensor] | None,
        normalizer, arm: str, phase: str, endpoint: str, update: int,
        source: Mapping[str, object],
        ) -> dict[str, object]:
    if arm not in ARMS or model.interaction_mode != ARM_MODE[arm]:
        raise RuntimeError("endpoint arm/mode mismatch")
    if phase not in ("sim", "real", "checkpoint_smoke"):
        raise ValueError("unsupported endpoint phase")
    materialized = _state_snapshot(model if state is None else state)
    if (len(materialized) != EXPECTED_STATE_TENSORS
            or sum(parameter.numel() for parameter in model.parameters())
            != EXPECTED_PARAMETERS):
        raise RuntimeError("ST-B210 endpoint structure changed")
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
           for value in materialized.values()):
        raise FloatingPointError("refusing a non-finite endpoint")
    state_sha = base.state_sha(materialized)
    normalizer_values = _normalizer_payload(normalizer)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "lifecycle_schema": SCHEMA,
        "arm": arm,
        "interaction_mode": ARM_MODE[arm],
        "phase": phase,
        "endpoint": endpoint,
        "update": int(update),
        "model_config": _endpoint_config(model),
        "model_state_dict": materialized,
        "model_state_sha256": state_sha,
        "normalizer": normalizer_values,
        "normalizer_sha256": base.state_sha(normalizer_values),
        "source": dict(source),
        "uq": False,
        "checkpoint_selected": False,
    }
    sim_long.atomic_torch_save(payload, path)
    return {
        "path": str(path.resolve()),
        "file_sha256": sim_long.file_sha256(path),
        "model_state_sha256": state_sha,
        "normalizer_sha256": payload["normalizer_sha256"],
        "arm": arm,
        "interaction_mode": ARM_MODE[arm],
        "phase": phase,
        "endpoint": endpoint,
        "update": int(update),
        "schema": ENDPOINT_SCHEMA,
    }


def load_endpoint(
        seal: Mapping[str, object], device: torch.device,
        ) -> tuple[SpatiotemporalB210, object, dict[str, object]]:
    path = Path(str(seal["path"]))
    if (not path.is_file()
            or sim_long.file_sha256(path) != seal.get("file_sha256")):
        raise RuntimeError("ST-B210 endpoint file seal changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, Mapping)
            or payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("lifecycle_schema") != SCHEMA):
        raise RuntimeError("not an ST-B210 lifecycle endpoint")
    for key in ("arm", "interaction_mode", "phase", "endpoint", "update"):
        if payload.get(key) != seal.get(key):
            raise RuntimeError(f"endpoint {key} seal changed")
    arm = str(payload["arm"])
    if arm not in ARMS or payload["interaction_mode"] != ARM_MODE[arm]:
        raise RuntimeError("endpoint interaction mode changed")
    raw = dict(payload["model_config"])
    if (raw.pop("architecture", None) != "spatiotemporal_b210"
            or raw.pop("flattened_history_bypass", None) is not False
            or raw.pop("interaction_mode", None) != ARM_MODE[arm]):
        raise RuntimeError("endpoint architecture identity changed")
    config = SpatiotemporalB210Config(**raw)
    model = SpatiotemporalB210(config, ARM_MODE[arm]).float()
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("loaded ST-B210 parameter count changed")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("endpoint lacks model state")
    model.load_state_dict(state, strict=True)
    actual_state_sha = base.state_sha(_state_snapshot(model))
    if (actual_state_sha != payload.get("model_state_sha256")
            or actual_state_sha != seal.get("model_state_sha256")):
        raise RuntimeError("endpoint model-state seal changed")
    normalizer_values = payload.get("normalizer")
    if (not isinstance(normalizer_values, Mapping)
            or base.state_sha(normalizer_values) != payload.get("normalizer_sha256")
            or payload.get("normalizer_sha256") != seal.get("normalizer_sha256")):
        raise RuntimeError("endpoint normalizer seal changed")
    normalizer = _normalizer_from_payload(normalizer_values, device)
    return model.to(device), normalizer, dict(payload)


def _latest_checkpoint(directory: Path) -> Path | None:
    paths = sorted(directory.glob("checkpoint_u????????.pt"))
    return paths[-1] if paths else None


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sim_arm_contract(arm: str, model: nn.Module) -> dict[str, object]:
    value = dict(lifecycle_contract()["sim"])
    value.update({
        "schema": SCHEMA,
        "stage": "sim",
        "arm": arm,
        "interaction_mode": ARM_MODE[arm],
        "architecture": _endpoint_config(model),
        "only_model_variable": "interaction_mode",
    })
    return value


def train_or_resume_sim_arm(
        *, arm: str, model: nn.Module, dataset, normalizer,
        data_seal: Mapping[str, object], git: Mapping[str, object],
        output_dir: Path, device: torch.device, resume: bool,
        checkpoint_every: int,
        ) -> tuple[nn.Module, dict[str, object]]:
    arm_dir = output_dir / "sim" / arm
    result_path = arm_dir / "stage_result.json"
    if resume and result_path.is_file():
        result = _read_json(result_path)
        contract = _sim_arm_contract(arm, model)
        config_sha = sim_long.canonical_sha(contract)
        expected_source = {
            "git": dict(git),
            "data_seal_sha256": sim_long.canonical_sha(data_seal),
            "config_sha256": config_sha,
            "sample_stream_sha256": SIM_SCHEDULE_SHA256,
        }
        loaded, _stored_normalizer, payload = load_endpoint(
            result["endpoint"], device)
        if (result.get("training_contract") != contract
                or result.get("config_sha256") != config_sha
                or payload.get("source") != expected_source):
            raise RuntimeError("completed Sim arm does not match this lifecycle")
        return loaded, result
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = arm_dir / "checkpoints"
    latest = _latest_checkpoint(checkpoint_dir) if resume else None
    if not resume and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(checkpoint_dir)
    contract = _sim_arm_contract(arm, model)
    config_sha = sim_long.canonical_sha(contract)
    sim_long.seed_everything(st_screen.SEED, device)
    training = sim_long.train(
        model=model, dataset=dataset, normalizer=normalizer, device=device,
        updates=SIM_UPDATES, batch_size=SIM_BATCH_SIZE,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_every,
        resume=latest, config_seal=config_sha, data=data_seal, git=git,
        expected_schedule_sha256=SIM_SCHEDULE_SHA256)
    source = {
        "git": dict(git),
        "data_seal_sha256": sim_long.canonical_sha(data_seal),
        "config_sha256": config_sha,
        "sample_stream_sha256": training["sample_stream_sha256"],
    }
    endpoint = seal_endpoint(
        arm_dir / "terminal_raw_u10000.pt", model=model, state=None,
        normalizer=normalizer, arm=arm, phase="sim", endpoint="raw_u10000",
        update=SIM_UPDATES, source=source)
    if endpoint["model_state_sha256"] != training["endpoint_state_sha256"]:
        raise RuntimeError("Sim training and compact endpoint state seals differ")
    result = {
        "stage": "sim",
        "arm": arm,
        "interaction_mode": ARM_MODE[arm],
        "training_contract": contract,
        "config_sha256": config_sha,
        "training": training,
        "endpoint": endpoint,
        "checkpoint_selection": False,
    }
    sim_long.atomic_json_save(result, result_path)
    return model, result


def _score_delta(
        candidate: Mapping[str, float], control: Mapping[str, float],
        ) -> dict[str, float]:
    keys = (*OFFICIAL_POINT_KEYS, "public_point_contribution_46_10_9")
    return {key: float(candidate[key]) - float(control[key]) for key in keys}


@torch.inference_mode()
def evaluate_sim_terminals(
        models: Mapping[str, nn.Module], dataset, normalizer,
        device: torch.device,
        ) -> dict[str, object]:
    errors: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    digests = {arm: hashlib.sha256() for arm in ARMS}
    for start in range(0, len(dataset), 8):
        indices = range(start, min(start + 8, len(dataset)))
        samples = [dataset[index] for index in indices]
        history = torch.stack([sample[0] for sample in samples]).to(device)
        target = np.ascontiguousarray(torch.stack(
            [sample[1] for sample in samples]).numpy())
        history = (history - normalizer.mean_input) / normalizer.std_input
        for arm in ARMS:
            model = models[arm].eval()
            prediction = (
                model(history) * normalizer.std_target + normalizer.mean_target)
            array = np.ascontiguousarray(prediction.float().cpu().numpy())
            if not np.isfinite(array).all():
                raise FloatingPointError("non-finite Sim terminal prediction")
            digests[arm].update(array.tobytes())
            errors[arm].append(base.error_rows(array, target))
    aggregate = {}
    for arm in ARMS:
        score = dict(base.score(np.concatenate(errors[arm])))
        score["public_point_contribution_46_10_9"] = score.pop(
            "point_contribution_46_10_9")
        aggregate[arm] = score
    return {
        "scope": "released_Sim100_training_windows_reporting_only",
        "windows": len(dataset),
        "aggregate": aggregate,
        "delta_temporal_minus_spatial": _score_delta(
            aggregate[ARMS[1]], aggregate[ARMS[0]]),
        "prediction_sha256": {
            arm: digests[arm].hexdigest() for arm in ARMS},
        "used_for_checkpoint_selection": False,
    }


class LifecycleAccessAudit:
    """One prefix read and one tail read, with an all-endpoint barrier."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self._opened: dict[str, set[str]] = {"fit": set(), "clean716": set()}
        self.endpoint_seals: dict[str, dict[str, object]] | None = None

    def record_endpoint_seals(
            self, seals: Mapping[str, Mapping[str, object]]) -> None:
        if self.endpoint_seals is not None:
            raise RuntimeError("lifecycle endpoints were registered twice")
        expected = {
            *(f"sim/{arm}" for arm in ARMS),
            *(f"real/{arm}/{endpoint}" for arm in ARMS
              for endpoint in ENDPOINTS),
        }
        if set(seals) != expected:
            raise RuntimeError("tail barrier requires both Sim and all four Real seals")
        verified: dict[str, dict[str, object]] = {}
        for key, seal in seals.items():
            path = Path(str(seal["path"]))
            if (not path.is_file()
                    or sim_long.file_sha256(path) != seal.get("file_sha256")):
                raise RuntimeError(f"endpoint changed before tail barrier: {key}")
            verified[key] = dict(seal)
        self.endpoint_seals = verified

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if filename == real_long.EXCLUDED_BASENAME:
            raise RuntimeError("excluded Real trajectory reached lifecycle opener")
        if phase not in self._opened:
            raise ValueError(f"unsupported access phase {phase!r}")
        if filename in self._opened[phase]:
            raise RuntimeError(f"Real H5 opened twice in {phase}: {filename}")
        if phase == "clean716" and self.endpoint_seals is None:
            raise RuntimeError("clean716 opened before all lifecycle endpoints were sealed")

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
            "all_lifecycle_endpoints_sealed_before_open": (
                self.endpoint_seals is not None),
        })


def _real_source(
        arm: str, sim_seal: Mapping[str, object],
        manifest_report: Mapping[str, object], clean_manifest: Path,
        normalizer, git: Mapping[str, object],
        ) -> dict[str, object]:
    return {
        "arm": arm,
        "interaction_mode": ARM_MODE[arm],
        "sim_endpoint": dict(sim_seal),
        "clean81_manifest_semantic_sha256": manifest_report["semantic_sha256"],
        "clean81_manifest_file_sha256": sim_long.file_sha256(clean_manifest),
        "normalizer_sha256": real_long._normalizer_sha(normalizer),
        "optimizer_windows": real_long.EXPECTED_OPTIMIZER_WINDOWS,
        "optimizer_stride": real_long.OPTIMIZER_STRIDE,
        "sample_stream_sha256": REAL_SCHEDULE_SHA256,
        "git": dict(git),
    }


def _extract_internal_ema(
        seal: Mapping[str, object], *, model: nn.Module,
        source: Mapping[str, object],
        ) -> Mapping[str, torch.Tensor]:
    path = Path(str(seal["path"]))
    if (not path.is_file()
            or sim_long.file_sha256(path) != seal.get("file_sha256")):
        raise RuntimeError("internal Real EMA seal changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != real_long.ENDPOINT_SCHEMA
            or payload.get("source") != dict(source)
            or payload.get("interaction_mode") != model.interaction_mode
            or not isinstance(state, Mapping)
            or base.state_sha(state) != payload.get("model_state_sha256")):
        raise RuntimeError("internal Real EMA payload changed")
    return state


def train_or_resume_real_arm(
        *, arm: str, sim_seal: Mapping[str, object], dataset,
        normalizer, manifest_report: Mapping[str, object],
        clean_manifest: Path, git: Mapping[str, object], output_dir: Path,
        device: torch.device, resume: bool, checkpoint_every: int,
        ) -> dict[str, object]:
    arm_dir = output_dir / "real" / arm
    result_path = arm_dir / "stage_result.json"
    if resume and result_path.is_file():
        result = _read_json(result_path)
        expected_source = _real_source(
            arm, sim_seal, manifest_report, clean_manifest,
            normalizer, git)
        if result.get("source") != expected_source:
            raise RuntimeError("completed Real arm does not match this lifecycle")
        for endpoint in ENDPOINTS:
            loaded, _normalizer, payload = load_endpoint(
                result["endpoints"][endpoint], device)
            if payload.get("source") != expected_source:
                raise RuntimeError("completed Real endpoint source changed")
            del loaded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result
    arm_dir.mkdir(parents=True, exist_ok=True)
    model, _sim_normalizer, _sim_payload = load_endpoint(sim_seal, device)
    real_long.install_normalizer(model, normalizer)
    source = _real_source(
        arm, sim_seal, manifest_report, clean_manifest, normalizer, git)
    resume_path = arm_dir / "resume.pt"
    actual_resume = resume_path if resume and resume_path.is_file() else None
    real_long.seed_real_recipe(device)
    training, internal_ema_seal = real_long.train(
        model, dataset, normalizer, source, arm_dir,
        updates=REAL_UPDATES, batch_size=REAL_BATCH_SIZE,
        resume=actual_resume, checkpoint_every=checkpoint_every,
        device=device)
    if training["sample_schedule_sha256"] != REAL_SCHEDULE_SHA256:
        raise RuntimeError("mature Real sample stream changed")
    ema_state = _extract_internal_ema(
        internal_ema_seal, model=model, source=source)
    endpoints = {
        "raw": seal_endpoint(
            arm_dir / "terminal_raw_u4000.pt", model=model, state=None,
            normalizer=normalizer, arm=arm, phase="real",
            endpoint="raw_u4000", update=REAL_UPDATES, source=source),
        "ema": seal_endpoint(
            arm_dir / "terminal_ema999_u4000.pt", model=model,
            state=ema_state, normalizer=normalizer, arm=arm, phase="real",
            endpoint="ema999_u4000", update=REAL_UPDATES, source=source),
    }
    result = {
        "stage": "real",
        "arm": arm,
        "interaction_mode": ARM_MODE[arm],
        "source": source,
        "training": training,
        "internal_ema_seal": internal_ema_seal,
        "endpoints": endpoints,
        "checkpoint_selection": False,
    }
    sim_long.atomic_json_save(result, result_path)
    del model, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _compare_panel(
        endpoint_results: Mapping[str, Mapping[str, object]],
        control_key: str, candidate_key: str,
        ) -> dict[str, object]:
    control = endpoint_results[control_key]
    candidate = endpoint_results[candidate_key]
    pooled = _score_delta(candidate["pooled"], control["pooled"])
    by_aoa = {
        aoa: _score_delta(candidate["by_aoa"][aoa], control["by_aoa"][aoa])
        for aoa in ("0", "5", "10", "15", "20")}
    return {
        "control": control_key,
        "candidate": candidate_key,
        "pooled_delta_temporal_minus_spatial": pooled,
        "by_aoa_delta_temporal_minus_spatial": by_aoa,
    }


def _retention(real_delta: float, sim_zero_shot_delta: float) -> float | None:
    if sim_zero_shot_delta == 0.0:
        return None
    return real_delta / sim_zero_shot_delta


def evaluate_after_all_seals(
        *, seals: Mapping[str, Mapping[str, object]],
        tail_dataset, tail_aoas: Sequence[int], device: torch.device,
        ) -> dict[str, object]:
    order = [
        *(f"sim/{arm}" for arm in ARMS),
        *(f"real/{arm}/{endpoint}" for endpoint in ENDPOINTS for arm in ARMS),
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
        raise RuntimeError("endpoint evaluations did not consume one clean716 target order")
    comparisons = {
        "sim_raw_zero_shot": _compare_panel(
            endpoints, f"sim/{ARMS[0]}", f"sim/{ARMS[1]}"),
        "real_raw_u4000": _compare_panel(
            endpoints, f"real/{ARMS[0]}/raw", f"real/{ARMS[1]}/raw"),
        "real_ema_u4000": _compare_panel(
            endpoints, f"real/{ARMS[0]}/ema", f"real/{ARMS[1]}/ema"),
    }
    baseline = comparisons["sim_raw_zero_shot"][
        "pooled_delta_temporal_minus_spatial"]
    survival: dict[str, object] = {}
    for stage in ("real_raw_u4000", "real_ema_u4000"):
        current = comparisons[stage]["pooled_delta_temporal_minus_spatial"]
        survival[stage] = {
            "delta_before_real": dict(baseline),
            "delta_after_real": dict(current),
            "after_minus_before": {
                key: float(current[key]) - float(baseline[key])
                for key in baseline},
            "signed_retention_ratio": {
                key: _retention(float(current[key]), float(baseline[key]))
                for key in baseline},
        }
    return {
        "endpoint_order": order,
        "endpoints": endpoints,
        "comparisons": comparisons,
        "candidate_control_survival_same_clean716_panel": survival,
        "common_target_sha256": next(iter(target_hashes)),
        "score_gate": None,
        "checkpoint_selection": False,
    }


def checkpoint_smoke(output_dir: Path, device: torch.device) -> dict[str, object]:
    """Production-size CPU checkpoint round trip; no data file is opened."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if device.type != "cpu":
        raise ValueError("checkpoint smoke is intentionally CPU-only")
    output_dir.mkdir(parents=True)
    models, identity = st_screen.construct_pair(device)
    normalizer = base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    st_screen.set_pair_normalizer_statistics(models, normalizer)
    arm = ARMS[1]
    seal = seal_endpoint(
        output_dir / "production_state_roundtrip.pt", model=models[arm],
        state=None, normalizer=normalizer, arm=arm,
        phase="checkpoint_smoke", endpoint="untrained_roundtrip", update=0,
        source={"scope": "production_size_CPU_checkpoint_smoke_no_data"})
    loaded, loaded_normalizer, payload = load_endpoint(seal, device)
    passed = (
        identity["bitwise_identical"]
        and seal["model_state_sha256"] == base.state_sha(_state_snapshot(loaded))
        and seal["normalizer_sha256"]
        == base.state_sha(_normalizer_payload(loaded_normalizer))
        and payload["checkpoint_selected"] is False)
    result = {
        "schema": SCHEMA,
        "smoke": "PASS" if passed else "FAIL",
        "data_files_opened": 0,
        "production_parameters_each": identity["parameter_counts"],
        "initial_state_sha256": identity["state_sha256"],
        "endpoint": seal,
    }
    sim_long.atomic_json_save(result, output_dir / "result.json")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute or args.acknowledge_protocol != ACKNOWLEDGEMENT:
        raise PermissionError("formal lifecycle requires the exact acknowledgement")
    required = (
        args.sim_dir, args.real_dir, args.clean_manifest,
        args.evaluation_manifest, args.output_dir)
    if any(value is None for value in required):
        raise ValueError("formal lifecycle is missing a required path")
    if args.sim_checkpoint_every <= 0 or args.real_checkpoint_every <= 0:
        raise ValueError("checkpoint intervals must be positive")
    device = torch.device(args.device)
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal lifecycle requires clean Git: {git['porcelain']}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal lifecycle requires CUDA")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    if not args.output_dir.exists() and args.resume:
        raise FileNotFoundError("--resume requires an existing output directory")
    # Before all endpoints are sealed, only stat the evaluation-manifest path.
    if not args.evaluation_manifest.is_file():
        raise FileNotFoundError(args.evaluation_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    inventory = base.inventory(args.sim_dir)
    sim_trajectories, sim_windows, _sim_files = base.load_fields(
        args.sim_dir, inventory)
    if any(tuple(value.shape[1:3]) != (32, 64)
           for value in sim_trajectories):
        raise RuntimeError("formal ST-B210 Sim grid must be exactly 32x64")
    sim_data_seal = sim_long.data_seal(
        args.sim_dir, inventory, sim_trajectories, sim_windows)
    sim_dataset = base.SimWindowDataset(sim_trajectories, sim_windows)
    sim_normalizer = base.ChannelNormalizer(
        *base.window_statistics(sim_trajectories, sim_windows), device=device)
    if sim_long.validate_formal_schedule(
            sim_long.make_schedule(len(sim_dataset))) != SIM_SCHEDULE_SHA256:
        raise RuntimeError("Sim100 sample stream changed")

    models, initial_identity = st_screen.construct_pair(device)
    normalizer_identity = st_screen.set_pair_normalizer_statistics(
        models, sim_normalizer)
    sim_results: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        models[arm], sim_results[arm] = train_or_resume_sim_arm(
            arm=arm, model=models[arm], dataset=sim_dataset,
            normalizer=sim_normalizer, data_seal=sim_data_seal, git=git,
            output_dir=args.output_dir, device=device, resume=args.resume,
            checkpoint_every=args.sim_checkpoint_every)
        gc.collect()
        torch.cuda.empty_cache()
    sim_evaluation = evaluate_sim_terminals(
        models, sim_dataset, sim_normalizer, device)
    sim_long.atomic_json_save(
        sim_evaluation, args.output_dir / "sim_terminal_evaluation.json")
    sim_seals = {arm: sim_results[arm]["endpoint"] for arm in ARMS}
    del models, sim_dataset, sim_trajectories, sim_windows
    gc.collect()
    torch.cuda.empty_cache()

    rows, manifest_report = real_long.load_clean_manifest(args.clean_manifest)
    access = LifecycleAccessAudit()
    real_trajectories, sparse_windows = real_long.load_real_rows(
        args.real_dir, rows, phase="fit", audit=access)
    if any(tuple(value.shape[1:3]) != (32, 64)
           for value in real_trajectories):
        raise RuntimeError("formal ST-B210 Real grid must be exactly 32x64")
    real_normalizer, real_normalizer_report = (
        real_long.normalizer_from_sparse_windows(
            real_trajectories, sparse_windows, device))
    optimizer_windows = real_long.dense_windows(real_trajectories)
    if len(optimizer_windows) != real_long.EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError("clean81 dense stride-5 window count changed")
    real_dataset = real_long.RealWindowDataset(
        real_trajectories, optimizer_windows)
    real_schedule = real_long.make_sample_schedule(
        len(real_dataset), REAL_UPDATES, REAL_BATCH_SIZE)
    if real_long.schedule_sha(real_schedule) != REAL_SCHEDULE_SHA256:
        raise RuntimeError("clean81 mature Real sample stream changed")
    real_results: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        real_results[arm] = train_or_resume_real_arm(
            arm=arm, sim_seal=sim_seals[arm], dataset=real_dataset,
            normalizer=real_normalizer, manifest_report=manifest_report,
            clean_manifest=args.clean_manifest, git=git,
            output_dir=args.output_dir, device=device, resume=args.resume,
            checkpoint_every=args.real_checkpoint_every)

    all_seals = {
        **{f"sim/{arm}": sim_seals[arm] for arm in ARMS},
        **{f"real/{arm}/{endpoint}": real_results[arm]["endpoints"][endpoint]
           for arm in ARMS for endpoint in ENDPOINTS},
    }
    access.record_endpoint_seals(all_seals)
    # This is the first content read of the clean716 manifest.
    evaluation_manifest_sha256 = sim_long.file_sha256(
        args.evaluation_manifest)
    tail_rows = real_long.load_clean716_rows(args.evaluation_manifest, rows)
    tail_trajectories, tail_windows = real_long.load_real_rows(
        args.real_dir, tail_rows, phase="clean716", audit=access)
    tail_dataset = real_long.RealWindowDataset(
        tail_trajectories, tail_windows)
    tail_aoas = [int(tail_rows[window.trajectory]["aoa"])
                 for window in tail_windows]
    evaluation = evaluate_after_all_seals(
        seals=all_seals, tail_dataset=tail_dataset,
        tail_aoas=tail_aoas, device=device)

    result = {
        "schema": SCHEMA,
        "contract": lifecycle_contract(),
        "git": git,
        "initial_identity": initial_identity,
        "sim_normalizer_identity": normalizer_identity,
        "sim_data_seal": sim_data_seal,
        "sim_arms": sim_results,
        "sim_terminal_training_domain_evaluation": sim_evaluation,
        "clean81_manifest": manifest_report,
        "real_normalizer": real_normalizer_report,
        "real_optimizer_windows": len(optimizer_windows),
        "real_arms": real_results,
        "endpoint_seals": all_seals,
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "file_sha256": evaluation_manifest_sha256,
            "first_content_read_after_all_four_real_endpoint_seals": True,
        },
        "clean716_evaluation": evaluation,
        "access_audit": access.events,
        "clean716_opened_before_all_four_real_endpoint_seals": False,
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
        print(json.dumps(lifecycle_contract(), sort_keys=True))
        return
    if args.output_dir is None:
        raise ValueError("this mode requires --output-dir")
    if args.mode == "checkpoint-smoke":
        result = checkpoint_smoke(args.output_dir, torch.device(args.device))
        print(json.dumps({"event": "st_b210_checkpoint_smoke_complete",
                          "smoke": result["smoke"],
                          "endpoint": result["endpoint"]}, sort_keys=True))
        return
    result = run(args)
    print(json.dumps({
        "event": "st_b210_paired_lifecycle_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "survival": result["clean716_evaluation"][
            "candidate_control_survival_same_clean716_panel"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
