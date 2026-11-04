#!/usr/bin/env python3
"""Fit fresh B210 Native-UQ from 3,341 whole-AoA point-OOF residuals.

The deployed point checkpoint is the SHA-bound all-clean81/full-trajectory
EMA-u4000 model.  Its point/backbone/controller/member partition stays bitwise
frozen.  UQ-only parameters receive four predeclared 800-update cosine cycles
over every legal stride-20 released-Real window.  Supervision is the signed
residual and official point factor cached from a model that excluded the
sample's entire AoA, never an in-sample residual from the deployed point.

There is no clean716/evaluation-manifest input, no metric-based endpoint
selection, and no intermediate truth scoring.  The sole deployment endpoint
is the fixed fourth-cycle u3200 state.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import build_b210_full_real_point_oof_residual_cache as cache_builder
import train_b210_full_real_point_oof_fold as point_fold
import train_b210_small4k_uq_release as uq_release
import train_ensemble_uq_oof as uq_train


dr1 = point_fold.dr1
ft12 = point_fold.ft12
smu1 = point_fold.smu1

EXPERIMENT = "b210_full_real_oof_native_uq"
SCHEMA = "b210_full_real_oof_native_uq_refit_v1"
POINT_RESULT_SCHEMA = "b210_full_released_real_final_point_v1"
POINT_ENDPOINT_SCHEMA = "b210_full_released_real_final_endpoint_v1"
ENDPOINT = "fixed_full_oof_u3200"
ACKNOWLEDGEMENT = "RUN_B210_FULL_REAL_OOF_NATIVE_UQ_V1"
LOAO_AGGREGATE_SCHEMA = "b210_full_real_loao_native_uq_aggregate_v1"
EXPECTED_SAMPLES = cache_builder.EXPECTED_SAMPLES
CYCLES = 4
UPDATES_PER_CYCLE = 800
TOTAL_UPDATES = CYCLES * UPDATES_PER_CYCLE
BATCH_SIZE = 8
SEED = 18
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
LOG_INTERVAL = 100


class OOFResidualDataset(Dataset):
    def __init__(self, base, residuals: torch.Tensor, factors: torch.Tensor):
        if (len(base) != EXPECTED_SAMPLES
                or residuals.shape != (EXPECTED_SAMPLES, 20, 32, 64, 2)
                or factors.shape != (EXPECTED_SAMPLES,)):
            raise ValueError("OOF residual dataset shape changed")
        self.base = base
        self.residuals = residuals
        self.factors = factors

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        history, target = self.base[index]
        return history, target, self.residuals[index], self.factors[index]


def _sha_tensor_bytes(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def _point_seal(result: Mapping[str, object]) -> Mapping[str, object]:
    endpoints = result.get("endpoints")
    if not isinstance(endpoints, Mapping):
        raise RuntimeError("final point result has no endpoint registry")
    row = endpoints.get("ema_u4000")
    if not isinstance(row, Mapping) or not isinstance(row.get("seal"), Mapping):
        raise RuntimeError("final point result has no EMA-u4000 seal")
    return row["seal"]


def normalizer_from_payload(
        payload: Mapping[str, object], device: torch.device,
        ) -> tuple[smu1.FoldNormalizer, dict]:
    raw = payload.get("normalizer")
    keys = ("mean_input", "mean_target", "std_input", "std_target")
    if not isinstance(raw, Mapping) or set(raw) != set(keys):
        raise RuntimeError("final point normalizer schema changed")
    values = {key: raw[key] for key in keys}
    digest = dr1.canonical_sha256(values)
    if digest != payload.get("normalizer_sha256"):
        raise RuntimeError("final point normalizer SHA changed")
    tensors = tuple(torch.as_tensor(values[key], dtype=torch.float32) for key in keys)
    if (any(value.shape != (3,) or not bool(torch.isfinite(value).all())
            for value in tensors)
            or bool((tensors[2] <= 0).any())
            or bool((tensors[3] <= 0).any())):
        raise RuntimeError("final point normalizer values are invalid")
    return smu1.FoldNormalizer(tensors, device), {
        "values": values, "sha256": digest,
    }


def load_final_point(
        result_path: Path, expected_result_sha256: str,
        device: torch.device, *,
        expected_result_schema: str = POINT_RESULT_SCHEMA,
        expected_endpoint_schema: str = POINT_ENDPOINT_SCHEMA,
        expected_start_sha256: str | None = None,
        ) -> tuple[dict, torch.nn.Module, smu1.FoldNormalizer, dict]:
    expected_start_sha256 = (
        point_fold.START_SHA256
        if expected_start_sha256 is None else
        dr1.normalized_sha256(expected_start_sha256, "final point start SHA"))
    if (not result_path.is_file()
            or dr1.sha256(result_path) != expected_result_sha256):
        raise RuntimeError("final full point result SHA changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seal = _point_seal(result)
    if (result.get("schema") != expected_result_schema
            or int(result.get("inventory", {}).get(
                "full_optimizer_windows", -1)) != 13_202
            or int(result.get("inventory", {}).get(
                "normalizer_windows", -1)) != 2_544
            or int(result.get("optimizer_support", {}).get(
                "windows", -1)) != 13_202
            or int(result.get("optimizer_support", {}).get(
                "stride", -1)) != 5
            or int(result.get("normalizer", {}).get("windows", -1)) != 2_544
            or result.get("normalizer", {}).get(
                "full_tail_in_normalizer") is not False
            or result.get("primary_endpoint") != "ema_u4000"
            or result.get("fresh_native_uq_crossfit_refit_required") is not True
            or result.get("access_audit", {}).get(
                "clean716_manifest_read") is not False
            or result.get("access_audit", {}).get(
                "allowlist_exact_and_ordered") is not True
            or result.get("start", {}).get("checkpoint_sha256")
            != expected_start_sha256):
        raise RuntimeError("final full point result contract changed")
    checkpoint = Path(str(seal.get("path", "")))
    if (not checkpoint.is_file()
            or dr1.sha256(checkpoint) != seal.get("file_sha256")):
        raise RuntimeError("final full point endpoint file SHA changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != expected_endpoint_schema
            or payload.get("training_scope") != "all_clean81_full13202"
            or int(payload.get("optimizer_windows", -1)) != 13_202
            or int(payload.get("normalizer_windows", -1)) != 2_544
            or int(payload.get("optimizer_stride", -1)) != 5
            or int(payload.get("normalizer_stride", -1)) != 20
            or payload.get("phase") != "real"
            or int(payload.get("iteration", -1)) != 4_000
            or payload.get("endpoint") != "ema_u4000"
            or payload.get("primary") is not True
            or payload.get("held_aoa") is not None
            or payload.get("start", {}).get("checkpoint_sha256")
            != expected_start_sha256
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal.get("model_state_sha256")
            or payload.get("fresh_native_uq_crossfit_refit_required") is not True):
        raise RuntimeError("final full point endpoint contract changed")
    normalizer, normalizer_report = normalizer_from_payload(payload, device)
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=True)
    point_state = uq_release.partition_state(model, uq=False)
    point_state_sha = dr1.model_state_sha256(point_state)
    checkpoint_point_state_sha = dr1.model_state_sha256({
        key: value for key, value in state.items()
        if not key.startswith(uq_release.UQ_PREFIXES)})
    if point_state_sha != checkpoint_point_state_sha:
        raise RuntimeError(
            "loaded final point partition differs from checkpoint payload")
    return payload, model, normalizer, {
        "result": str(result_path.resolve()),
        "result_sha256": expected_result_sha256,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": seal["file_sha256"],
        "model_state_sha256": seal["model_state_sha256"],
        "point_partition_sha256": point_state_sha,
        "checkpoint_point_partition_sha256": checkpoint_point_state_sha,
        "normalizer": normalizer_report,
        "schema": expected_endpoint_schema,
        "training_scope": "all_clean81_full13202",
    }


def reset_uq_from_pure_sim(
        model: torch.nn.Module, path: Path, expected_sha256: str,
        expected_model_config: Mapping[str, object] | None = None,
        *, formal_start_sha256: str | None = None,
        expected_iteration: int = 10_000,
        expected_status: str | None = None,
        source_label: str = "pure_sim10k",
        ) -> dict:
    """Replace every existing UQ-only tensor with its pure-Sim10k value.

    Real point training is allowed to update the native interval modules, so
    inheriting them would expose the all-81 labels to every nominal OOF fold.
    The non-UQ partition is therefore kept from the Real point endpoint while
    the complete UQ partition is restored from the formal pure-Sim start.
    """
    expected_sha256 = dr1.normalized_sha256(
        expected_sha256, f"{source_label} UQ initialization SHA")
    formal_start_sha256 = dr1.normalized_sha256(
        point_fold.START_SHA256
        if formal_start_sha256 is None else formal_start_sha256,
        "formal UQ Sim source SHA")
    if expected_sha256 != formal_start_sha256:
        raise RuntimeError("fresh-UQ source is not the formal Sim-lineage start")
    if not path.is_file() or dr1.sha256(path) != expected_sha256:
        raise RuntimeError("fresh-UQ Sim-lineage checkpoint SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("model_state_dict")
    source_config = payload.get("model_config")
    if (expected_iteration <= 0
            or payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != expected_iteration
            or not isinstance(source, Mapping)
            or not isinstance(source_config, Mapping)):
        raise RuntimeError("fresh-UQ source is not the expected Sim endpoint")
    if (expected_status is not None
            and payload.get("metrics", {}).get("status") != expected_status):
        raise RuntimeError("fresh-UQ source lacks the expected Sim status")
    normalized_source_config = dict(source_config)
    normalized_expected_config = (
        None if expected_model_config is None else dict(expected_model_config))
    if (source_config.get("architecture") == "ensemble_unet"
            and expected_model_config is not None
            and expected_model_config.get("architecture") == "ensemble_unet"):
        expected_fields = dict(expected_model_config)
        expected_fields.pop("architecture")
        expected_config = ft12.EnsembleUNet2dConfig(**expected_fields)
        normalized_expected_config = {
            "architecture": "ensemble_unet", **asdict(expected_config)}
        normalized_source_config = {
            "architecture": "ensemble_unet", **asdict(
                ft12.validate_b210_model_config(
                    source_config, expected_config))}
    elif source_config.get("architecture") == "ensemble_unet":
        normalized_source_config = {
            "architecture": "ensemble_unet", **asdict(
                ft12.validate_b210_model_config(
                    source_config, point_fold.sim_tke.b210_config("candidate")))}
    if (normalized_expected_config is not None
            and dr1.canonical_sha256(normalized_source_config)
            != dr1.canonical_sha256(normalized_expected_config)):
        raise RuntimeError("fresh-UQ source architecture differs from point model")

    destination = model.state_dict()
    destination_uq = {
        key: value for key, value in destination.items()
        if key.startswith(uq_release.UQ_PREFIXES)}
    source_uq = {
        key: value for key, value in source.items()
        if key.startswith(uq_release.UQ_PREFIXES)}
    if not destination_uq or set(destination_uq) != set(source_uq):
        raise RuntimeError(
            "pure-Sim and final-point UQ state-key partitions differ")
    for key, value in source_uq.items():
        if (destination_uq[key].shape != value.shape
                or destination_uq[key].dtype != value.dtype):
            raise RuntimeError(f"fresh-UQ tensor contract changed at {key}")

    point_before = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    uq_before = dr1.model_state_sha256(destination_uq)
    source_uq_sha = dr1.model_state_sha256(source_uq)
    for key, value in source_uq.items():
        destination[key] = value
    model.load_state_dict(destination, strict=True)
    point_after = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    uq_after_state = uq_release.partition_state(model, uq=True)
    uq_after = dr1.model_state_sha256(uq_after_state)
    if point_after != point_before:
        raise RuntimeError("fresh-UQ reset changed the frozen point partition")
    if uq_after != source_uq_sha:
        raise RuntimeError("fresh-UQ reset does not equal pure-Sim UQ state")
    if uq_after == uq_before:
        raise RuntimeError(
            "final-point UQ state already equals pure-Sim; reset was not observable")
    prefixes = [
        prefix for prefix in uq_release.UQ_PREFIXES
        if any(key.startswith(prefix) for key in source_uq)]
    return {
        "policy": f"reset_all_existing_uq_only_tensors_from_{source_label}",
        "source_label": source_label,
        "source": str(path.resolve()),
        "source_file_sha256": expected_sha256,
        "source_phase": "sim",
        "source_iteration": expected_iteration,
        "source_status": expected_status,
        "source_model_config_sha256": dr1.canonical_sha256(source_config),
        "source_normalized_model_config_sha256": dr1.canonical_sha256(
            normalized_source_config),
        "target_normalized_model_config_sha256": dr1.canonical_sha256(
            normalized_expected_config),
        "reset_prefixes": prefixes,
        "reset_state_tensors": len(source_uq),
        "reset_state_numel": sum(value.numel() for value in source_uq.values()),
        "final_point_uq_sha256_before_reset": uq_before,
        "pure_sim_uq_sha256": source_uq_sha,
        "uq_sha256_after_reset": uq_after,
        "point_partition_sha256_before_reset": point_before,
        "point_partition_sha256_after_reset": point_after,
        "point_partition_bitwise": True,
        "full_real_uq_state_inherited": False,
        "non_uq_state_source": "final_all81_point_checkpoint",
        "uq_state_source": f"formal_{source_label}_checkpoint",
    }


def load_loao_uq_aggregate(
        path: Path, expected_sha256: str, cache_report: Mapping[str, object],
        uq_init_report: Mapping[str, object],
        ) -> dict:
    expected_sha256 = dr1.normalized_sha256(
        expected_sha256, "whole-AoA UQ aggregate SHA")
    if not path.is_file() or dr1.sha256(path) != expected_sha256:
        raise RuntimeError("whole-AoA UQ aggregate SHA changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    point_fold.verify_json_payload(payload)
    if (payload.get("schema") != LOAO_AGGREGATE_SCHEMA
            or payload.get("held_aoas") != list(point_fold.AOA_VALUES)
            or int(payload.get("folds", -1)) != point_fold.FOLDS
            or int(payload.get("updates_per_fold", -1)) != UPDATES_PER_CYCLE
            or payload.get("fixed_terminal_only") is not True
            or payload.get("checkpoint_selection") is not False
            or payload.get("clean716_manifest_read") is not False
            or payload.get("oof_residual_cache", {}).get("file_sha256")
            != cache_report["file_sha256"]
            or payload.get("uq_initialization", {}).get("source_file_sha256")
            != uq_init_report["source_file_sha256"]
            or payload.get("uq_initialization", {}).get(
                "full_real_uq_state_inherited") is not False
            or payload.get("reference_final_uq", {}).get(
                "real_in_sample_initialization") is not True
            or payload.get("reference_final_uq", {}).get(
                "reference_only") is not True
            or not isinstance(payload.get("pooled"), Mapping)
            or not isinstance(payload["pooled"].get("native_before"), Mapping)
            or not isinstance(payload["pooled"].get("native_after"), Mapping)):
        raise RuntimeError("whole-AoA UQ aggregate contract changed")
    return {
        "path": str(path.resolve()),
        "file_sha256": expected_sha256,
        "schema": LOAO_AGGREGATE_SCHEMA,
        "held_aoas": list(point_fold.AOA_VALUES),
        "folds": point_fold.FOLDS,
        "updates_per_fold": UPDATES_PER_CYCLE,
        "pooled": payload["pooled"],
        "per_fold_native_sps": payload["per_fold_native_sps"],
        "reference_final_uq": payload["reference_final_uq"],
        "fixed_terminal_only": True,
        "checkpoint_selection": False,
    }


def load_oof_cache(
        path: Path, expected_sha256: str,
        ) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
    if not path.is_file() or dr1.sha256(path) != expected_sha256:
        raise RuntimeError("whole-AoA OOF residual cache SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    residuals = payload.get("signed_residual_physical_uv")
    factors = payload.get("oof_point_factor")
    identities = payload.get("sample_identities")
    if (payload.get("schema") != cache_builder.SCHEMA
            or int(payload.get("folds", -1)) != point_fold.FOLDS
            or payload.get("aoa_values") != list(point_fold.AOA_VALUES)
            or int(payload.get("window_stride", -1)) != 20
            or int(payload.get("samples", -1)) != EXPECTED_SAMPLES
            or payload.get("manifest", {}).get("semantic_sha256")
            != ft12.FORMAL_MANIFEST_SEMANTIC_SHA256
            or payload.get("manifest_file_sha256")
            != ft12.FORMAL_MANIFEST_FILE_SHA256
            or payload.get("clean716_manifest_read") is not False
            or not isinstance(identities, list)
            or len(identities) != EXPECTED_SAMPLES
            or dr1.canonical_sha256(identities)
            != payload.get("sample_identity_sha256")
            or not isinstance(residuals, torch.Tensor)
            or not isinstance(factors, torch.Tensor)
            or residuals.shape != (EXPECTED_SAMPLES, 20, 32, 64, 2)
            or factors.shape != (EXPECTED_SAMPLES,)
            or residuals.dtype != torch.float16
            or not bool(torch.isfinite(residuals).all())
            or not bool(torch.isfinite(factors).all())
            or _sha_tensor_bytes(residuals)
            != payload.get("signed_residual_physical_uv_sha256")
            or _sha_tensor_bytes(factors)
            != payload.get("oof_point_factor_sha256")):
        raise RuntimeError("whole-AoA OOF residual cache contract changed")
    report = {
        "path": str(path.resolve()),
        "file_sha256": expected_sha256,
        "schema": cache_builder.SCHEMA,
        "samples": EXPECTED_SAMPLES,
        "sample_identity_sha256": payload["sample_identity_sha256"],
        "signed_residual_physical_uv_sha256": payload[
            "signed_residual_physical_uv_sha256"],
        "oof_point_factor_sha256": payload["oof_point_factor_sha256"],
        "ordered_history_stream_sha256": payload[
            "ordered_history_stream_sha256"],
        "ordered_target_stream_sha256": payload[
            "ordered_target_stream_sha256"],
        "fold_results": payload["fold_results"],
        "clean716_manifest_read": False,
    }
    return payload, residuals, factors, report


def build_dataset(
        rows: Sequence[Mapping[str, object]], real_dir: Path,
        cache: Mapping[str, object], residuals: torch.Tensor,
        factors: torch.Tensor,
        ) -> tuple[OOFResidualDataset, dict]:
    evaluation_rows = cache_builder.full_evaluation_rows(rows)
    identities = [
        {"file": str(row["file"]), "start": int(start),
         "re": int(row["re"]), "aoa": int(row["aoa"]),
         "fold": point_fold.aoa_fold(int(row["aoa"]))}
        for row in evaluation_rows for start in row["val_starts"]]
    if (identities != cache["sample_identities"]
            or dr1.canonical_sha256(identities)
            != cache["sample_identity_sha256"]):
        raise RuntimeError("released full-window order differs from OOF cache")
    trajectories = [
        cache_builder.read_trajectory(real_dir, row) for row in evaluation_rows]
    window_type = ft12.npt1.ensemble_train.common.repro.Window
    windows = [
        window_type(trajectory, int(start))
        for trajectory, row in enumerate(evaluation_rows)
        for start in row["val_starts"]]
    base = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    expected_history = cache["ordered_history_sample_sha256"]
    expected_target = cache["ordered_target_sample_sha256"]
    history_hashes, target_hashes = [], []
    for index in range(len(base)):
        history, target = base[index]
        history_hashes.append(_sha_tensor_bytes(history))
        target_hashes.append(_sha_tensor_bytes(target))
        if (history_hashes[-1] != expected_history[index]
                or target_hashes[-1] != expected_target[index]):
            raise RuntimeError(f"OOF cache sample content changed at {index}")
    if (dr1.canonical_sha256(history_hashes)
            != cache["ordered_history_stream_sha256"]
            or dr1.canonical_sha256(target_hashes)
            != cache["ordered_target_stream_sha256"]):
        raise RuntimeError("OOF cache sample stream digest changed")
    return OOFResidualDataset(base, residuals, factors), {
        "trajectories": len(trajectories),
        "windows": len(windows),
        "window_stride": 20,
        "ordered_history_stream_sha256": cache[
            "ordered_history_stream_sha256"],
        "ordered_target_stream_sha256": cache[
            "ordered_target_stream_sha256"],
        "all_complete_trajectory_frames_read": True,
        "clean716_manifest_read": False,
        "excluded_opened": False,
    }


def save_uq_cycle(
        path: Path, model: torch.nn.Module, cycle: int,
        point_partition_sha256: str, training: Mapping[str, object],
        ) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    state = {
        key: value.detach().cpu().clone()
        for key, value in uq_release.partition_state(model, uq=True).items()}
    state_sha = dr1.model_state_sha256(state)
    payload = {
        "schema": "b210_full_real_oof_native_uq_cycle_v1",
        "experiment": EXPERIMENT,
        "cycle": cycle,
        "iteration": cycle * UPDATES_PER_CYCLE,
        "endpoint": f"fixed_oof_u{cycle * UPDATES_PER_CYCLE}",
        "uq_state_dict": state,
        "uq_state_sha256": state_sha,
        "point_partition_sha256": point_partition_sha256,
        "training": dict(training),
        "checkpoint_selection": False,
        "clean716_manifest_read": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "uq_state_sha256": state_sha,
        "cycle": cycle,
        "iteration": cycle * UPDATES_PER_CYCLE,
        "endpoint": payload["endpoint"],
    }


def train_cycle(
        model: torch.nn.Module, dataset: OOFResidualDataset,
        normalizer: smu1.FoldNormalizer, device: torch.device,
        cycle: int, deadline: float, *,
        indices: Sequence[int] | None = None,
        seed: int = SEED,
        expected_point_partition_sha256: str | None = None,
        ) -> dict:
    optimizer_indices = (
        list(range(len(dataset))) if indices is None else list(indices))
    if (not optimizer_indices
            or len(optimizer_indices) != len(set(optimizer_indices))
            or min(optimizer_indices) < 0
            or max(optimizer_indices) >= len(dataset)):
        raise ValueError("Native-UQ optimizer indices are invalid")
    uq_train.seed_everything(seed)
    parameters = uq_release.configure_uq_only(model)
    frozen_before = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    if (expected_point_partition_sha256 is not None
            and frozen_before != expected_point_partition_sha256):
        raise RuntimeError("Native-UQ cycle did not start from frozen point SHA")
    loader = DataLoader(
        Subset(dataset, optimizer_indices),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed))
    iterator = ft12.npt1.ensemble_train.common.repro.cycling(loader)
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES_PER_CYCLE)
    objective_args = uq_release.sharp2_args()
    history_rows = []
    sums: dict[str, float] = {}
    count = 0
    gradient_preflight = None
    started = time.perf_counter()
    for update in range(1, UPDATES_PER_CYCLE + 1):
        if time.perf_counter() > deadline:
            raise TimeoutError("full-OOF Native-UQ wall budget expired")
        uq_release.set_uq_training_mode(model)
        history, target, residual, factor = next(iterator)
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        factor = factor.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        optimizer.zero_grad(set_to_none=True)
        prediction, quantiles, extents, logits = uq_train.quantile_forward(
            model, history_norm)
        loss, components = uq_train.uq_objective(
            objective_args, model, prediction, quantiles, target_norm,
            normalizer, interval_extents=extents, coverage_logits=logits,
            residual_physical_override=residual,
            point_factor_override=factor,
            scored_override=target[..., :2] != 0.0)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite full-OOF Native-UQ loss at cycle{cycle}/u{update}")
        loss.backward()
        gradients = [value.grad for value in parameters if value.grad is not None]
        if not gradients or any(
                not bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError("full-OOF Native-UQ gradient is invalid")
        if update == 1:
            squared = sum(float(value.float().square().sum()) for value in gradients)
            if not math.isfinite(squared) or squared <= 0.0:
                raise RuntimeError("full-OOF Native-UQ gradient preflight failed")
            gradient_preflight = {
                "iteration": 1, "finite": True, "nonzero": True,
                "l2_norm": math.sqrt(squared)}
        clipped = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        if not bool(torch.isfinite(clipped)):
            raise FloatingPointError("full-OOF Native-UQ clipped gradient is invalid")
        optimizer.step()
        scheduler.step()
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if update % LOG_INTERVAL == 0:
            row = {
                "cycle": cycle,
                "cycle_update": update,
                "total_update": (cycle - 1) * UPDATES_PER_CYCLE + update,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            history_rows.append(row)
            print(json.dumps({
                "event": "b210_full_oof_native_uq_progress", **row}), flush=True)
            sums.clear()
            count = 0
    frozen_after = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    if frozen_after != frozen_before:
        raise RuntimeError("Native-UQ cycle changed frozen point state")
    if (expected_point_partition_sha256 is not None
            and frozen_after != expected_point_partition_sha256):
        raise RuntimeError("Native-UQ cycle ended with unexpected point SHA")
    model.eval()
    return {
        "cycle": cycle,
        "updates": UPDATES_PER_CYCLE,
        "total_terminal_update": cycle * UPDATES_PER_CYCLE,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "optimizer_samples": len(optimizer_indices),
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": UPDATES_PER_CYCLE,
        "grad_clip": GRAD_CLIP,
        "objective": {
            key: value for key, value in vars(objective_args).items()
            if isinstance(value, (bool, int, float, str)) or value is None},
        "supervision": "whole-AoA point-OOF signed physical residual/factor",
        "gradient_preflight": gradient_preflight,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "frozen_state_bitwise": True,
        "history": history_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def save_final_endpoint(
        path: Path, model: torch.nn.Module, payload: Mapping[str, object],
        normalizer_report: Mapping[str, object], point_report: Mapping[str, object],
        cache_report: Mapping[str, object], dataset_report: Mapping[str, object],
        uq_initialization: Mapping[str, object],
        loao_validation: Mapping[str, object],
        cycles: Sequence[Mapping[str, object]], seals: Sequence[Mapping[str, object]],
        ) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    state_sha = dr1.model_state_sha256(state)
    frozen_sha = dr1.model_state_sha256({
        key: value for key, value in state.items()
        if not key.startswith(uq_release.UQ_PREFIXES)})
    if frozen_sha != point_report["point_partition_sha256"]:
        raise RuntimeError("final Native-UQ endpoint changed frozen point partition")
    output = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": TOTAL_UPDATES,
        "endpoint": ENDPOINT,
        "primary": True,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": normalizer_report["sha256"],
        "point_checkpoint": dict(point_report),
        "point_partition_sha256": frozen_sha,
        "point_state_bitwise": True,
        "uq_initialization": dict(uq_initialization),
        "whole_aoa_uq_oof_validation": dict(loao_validation),
        "oof_residual_cache": dict(cache_report),
        "dataset": dict(dataset_report),
        "training": {
            "cycles": [dict(row) for row in cycles],
            "cycle_seals": [dict(row) for row in seals],
            "cycles_fixed_before_training": CYCLES,
            "updates_per_cycle": UPDATES_PER_CYCLE,
            "total_updates": TOTAL_UPDATES,
            "checkpoint_selection": False,
        },
        "source_point_model_config_sha256": dr1.canonical_sha256(
            payload["model_config"]),
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "checkpoint_selection": False,
        "optimizer_state_saved": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(output, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "point_partition_sha256": frozen_sha,
        "schema": SCHEMA,
        "endpoint": ENDPOINT,
        "iteration": TOTAL_UPDATES,
    }


def verify_final_endpoint(path: Path, seal: Mapping[str, object]) -> dict:
    if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("final Native-UQ endpoint file SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != SCHEMA
            or payload.get("endpoint") != ENDPOINT
            or int(payload.get("iteration", -1)) != TOTAL_UPDATES
            or payload.get("point_state_bitwise") is not True
            or payload.get("uq_initialization", {}).get(
                "full_real_uq_state_inherited") is not False
            or payload.get("whole_aoa_uq_oof_validation", {}).get(
                "schema") != LOAO_AGGREGATE_SCHEMA
            or payload.get("clean716_manifest_read") is not False
            or payload.get("evaluation_manifest_read") is not False
            or payload.get("checkpoint_selection") is not False
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("final Native-UQ endpoint verification failed")
    frozen_sha = dr1.model_state_sha256({
        key: value for key, value in state.items()
        if not key.startswith(uq_release.UQ_PREFIXES)})
    if frozen_sha != seal["point_partition_sha256"]:
        raise RuntimeError("verified Native-UQ endpoint point state drifted")
    return {
        "exact": True,
        "file_sha256": seal["file_sha256"],
        "model_state_sha256": seal["model_state_sha256"],
        "point_partition_sha256": frozen_sha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-refit", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--point-result", type=Path)
    parser.add_argument("--point-result-sha256")
    parser.add_argument("--uq-init-sim", type=Path)
    parser.add_argument("--uq-init-sim-sha256")
    parser.add_argument("--oof-residual-cache", type=Path)
    parser.add_argument("--oof-residual-cache-sha256")
    parser.add_argument("--loao-uq-aggregate", type=Path)
    parser.add_argument("--loao-uq-aggregate-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 60 * 60)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = (
        args.output_dir, args.real_dir, args.clean_manifest, args.point_result,
        args.point_result_sha256, args.uq_init_sim, args.uq_init_sim_sha256,
        args.oof_residual_cache, args.oof_residual_cache_sha256,
        args.loao_uq_aggregate, args.loao_uq_aggregate_sha256)
    if args.contract_only:
        if any(value is not None for value in values):
            raise ValueError("contract-only mode received execution inputs")
        return
    if any(value is None for value in values):
        raise ValueError("full-OOF Native-UQ refit is missing required input")
    if (not args.execute_formal_gpu
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise PermissionError("formal Native-UQ refit needs acknowledgement")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.point_result_sha256 = dr1.normalized_sha256(
        args.point_result_sha256, "final point result SHA")
    args.uq_init_sim_sha256 = dr1.normalized_sha256(
        args.uq_init_sim_sha256, "pure-Sim10k UQ initialization SHA")
    args.oof_residual_cache_sha256 = dr1.normalized_sha256(
        args.oof_residual_cache_sha256, "OOF residual cache SHA")
    args.loao_uq_aggregate_sha256 = dr1.normalized_sha256(
        args.loao_uq_aggregate_sha256, "whole-AoA UQ aggregate SHA")


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal full-OOF Native-UQ refit requires CUDA")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    point_fold.validate_inventory(rows)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    point_payload, model, normalizer, point_report = load_final_point(
        args.point_result, args.point_result_sha256, device)
    uq_initialization = reset_uq_from_pure_sim(
        model, args.uq_init_sim, args.uq_init_sim_sha256,
        point_payload["model_config"])
    if (uq_initialization["point_partition_sha256_after_reset"]
            != point_report["point_partition_sha256"]):
        raise RuntimeError("fresh-UQ reset is not bound to final point SHA")
    cache, residuals, factors, cache_report = load_oof_cache(
        args.oof_residual_cache, args.oof_residual_cache_sha256)
    loao_validation = load_loao_uq_aggregate(
        args.loao_uq_aggregate, args.loao_uq_aggregate_sha256,
        cache_report, uq_initialization)
    dataset, dataset_report = build_dataset(
        rows, args.real_dir, cache, residuals, factors)
    if dataset_report["windows"] != EXPECTED_SAMPLES:
        raise RuntimeError("full-OOF Native-UQ dataset count changed")
    cycle_reports, cycle_seals = [], []
    for cycle in range(1, CYCLES + 1):
        report = train_cycle(
            model, dataset, normalizer, device, cycle, deadline,
            expected_point_partition_sha256=point_report[
                "point_partition_sha256"])
        seal = save_uq_cycle(
            args.output_dir / f"native_uq_u{cycle * UPDATES_PER_CYCLE}_uq_only.pth",
            model, cycle, point_report["point_partition_sha256"], report)
        cycle_reports.append(report)
        cycle_seals.append(seal)
        print(json.dumps({
            "event": "b210_full_oof_native_uq_cycle_sealed",
            "cycle": cycle, "seal": seal}), flush=True)
    endpoint = args.output_dir / "full_oof_native_uq_u3200.pth"
    seal = save_final_endpoint(
        endpoint, model, point_payload, point_report["normalizer"],
        point_report, cache_report, dataset_report, uq_initialization,
        loao_validation, cycle_reports, cycle_seals)
    verification = verify_final_endpoint(endpoint, seal)
    result = {
        "schema": "b210_full_real_oof_native_uq_run_v1",
        "experiment": EXPERIMENT,
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "point": point_report,
        "uq_initialization": uq_initialization,
        "whole_aoa_uq_oof_validation": loao_validation,
        "oof_residual_cache": cache_report,
        "dataset": dataset_report,
        "fixed_cycles": CYCLES,
        "updates_per_cycle": UPDATES_PER_CYCLE,
        "total_updates": TOTAL_UPDATES,
        "cycle_reports": cycle_reports,
        "cycle_seals": cycle_seals,
        "primary_endpoint": ENDPOINT,
        "endpoint": seal,
        "endpoint_verification": verification,
        "point_state_bitwise": True,
        "checkpoint_selection": False,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
        "decision": "COMPLETE_FIXED_FULL_OOF_NATIVE_UQ_U3200",
    }
    result = point_fold.seal_json_payload(result)
    result_path = args.output_dir / "result.json"
    dr1.write_json_atomic(result_path, result)
    point_fold.verify_json_payload(json.loads(
        result_path.read_text(encoding="utf-8")))
    del model, dataset, residuals, factors, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "experiment": EXPERIMENT,
            "point_result_schema": POINT_RESULT_SCHEMA,
            "point_endpoint_schema": POINT_ENDPOINT_SCHEMA,
            "oof_cache_schema": cache_builder.SCHEMA,
            "samples": EXPECTED_SAMPLES,
            "cycles": CYCLES,
            "updates_per_cycle": UPDATES_PER_CYCLE,
            "total_updates": TOTAL_UPDATES,
            "uq_initialization": (
                "all existing UQ-only tensors reset from pure-Sim10k"),
            "all81_real_uq_state_inherited": False,
            "loao_uq_validation_required_before_refit": True,
            "clean716_manifest_accepted": False,
            "endpoint_selection": False,
        }, indent=2, sort_keys=True))
        return
    result = run(args)
    print(json.dumps({
        "event": "b210_full_real_oof_native_uq_complete",
        "endpoint": result["endpoint"],
        "result": str((args.output_dir / "result.json").resolve()),
    }), flush=True)


if __name__ == "__main__":
    main()
