#!/usr/bin/env python3
"""Fixed-u800 native-UQ refit on every released Real training prefix.

This is the release-stage consequence of the frozen whole-AoA-held UQ gate.
It never updates point/backbone/controller/member state, never reads temporal
tails, and never selects a checkpoint from an evaluation metric.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Mapping

import torch

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_sim as sim_tke
import train_b210_small4k_uq_release as uq_release
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1
from models.ensemble_unet2d import EnsembleUNet2d


EXPERIMENT = "b210_full_prefix_native_uq_refit"
SCHEMA = "b210_full_prefix_native_uq_refit_v1"
SUMMARY_SCHEMA = "b210_full_prefix_native_uq_refit_summary_v1"
POINT_SCHEMA = "b210_np_functional_real_small4k_endpoint_v1"
VALIDATION_SCHEMA = "b210_condition_held_uq_only_aggregate_v1"
VALIDATION_DECISION = "PASS_CONDITION_HELD_UQ_ONLY_GATE"
UPDATES = 800
SEED = 18
EXPECTED_WINDOWS = dr1.EXPECTED_WINDOWS


class PrefixOnlyAudit:
    """Allow each released prefix once and reject every other data phase."""

    def __init__(self) -> None:
        self.opened: list[str] = []

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        name = str(row["file"])
        if phase != "fit" or name == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("full-prefix UQ received a non-fit trajectory")
        if name in self.opened:
            raise RuntimeError("full-prefix trajectory opened twice")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        self.before_open(row, phase)
        self.opened.append(str(row["file"]))

    def report(self, rows: list[dict]) -> dict:
        expected = [str(row["file"]) for row in rows]
        return {
            "opened": list(self.opened),
            "expected": expected,
            "allowlist_exact": self.opened == expected,
            "prefix_only": True,
            "tail_arrays_read": 0,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
        }


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be a mapping: {path}")
    return value


def validate_validation_gate(path: Path, expected_sha256: str) -> dict:
    actual = dr1.sha256(path)
    if actual != expected_sha256.lower():
        raise RuntimeError("condition-held UQ aggregate SHA changed")
    value = load_json(path)
    if (
        value.get("schema") != VALIDATION_SCHEMA
        or value.get("decision") != VALIDATION_DECISION
        or value.get("gate", {}).get("validation_gate_pass") is not True
        or value.get("held_aoas") != list(dr1.AOA_VALUES)
    ):
        raise RuntimeError("condition-held UQ promotion gate did not pass")
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "schema": VALIDATION_SCHEMA,
        "decision": VALIDATION_DECISION,
        "pooled_sps_delta": value["comparison"]["native_sps_delta"],
        "positive_folds": value["gate"]["positive_folds"],
        "worst_fold": value["gate"]["worst_fold"],
    }


def normalizer_from_point(
    payload: Mapping[str, object], device: torch.device
) -> tuple[smu1.FoldNormalizer, dict]:
    raw = payload.get("normalizer")
    keys = ("mean_input", "mean_target", "std_input", "std_target")
    if not isinstance(raw, Mapping) or set(raw) != set(keys):
        raise ValueError("point checkpoint normalizer schema changed")
    values = {key: raw[key] for key in keys}
    actual = dr1.canonical_sha256(values)
    if actual != str(payload.get("normalizer_sha256", "")).lower():
        raise RuntimeError("point checkpoint normalizer SHA changed")
    statistics = tuple(torch.as_tensor(values[key], dtype=torch.float32) for key in keys)
    if any(value.shape != (3,) for value in statistics):
        raise ValueError("point checkpoint normalizer shape changed")
    if any(not bool(torch.isfinite(value).all()) for value in statistics):
        raise ValueError("point checkpoint normalizer is non-finite")
    if bool((statistics[2] <= 0).any()) or bool((statistics[3] <= 0).any()):
        raise ValueError("point checkpoint normalizer std is non-positive")
    return smu1.FoldNormalizer(statistics, device), {
        "values": values,
        "sha256": actual,
    }


def load_point(
    path: Path, expected_sha256: str, device: torch.device
) -> tuple[dict, EnsembleUNet2d, smu1.FoldNormalizer, dict]:
    actual_file_sha = dr1.sha256(path)
    if actual_file_sha != expected_sha256.lower():
        raise RuntimeError("full-clean Dense4k point checkpoint SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("point checkpoint must contain a mapping")
    state = payload.get("model_state_dict")
    if (
        payload.get("schema") != POINT_SCHEMA
        or payload.get("phase") != "real"
        or int(payload.get("iteration", -1)) != 4000
        or payload.get("endpoint") != "ema_u4000"
        or not isinstance(state, Mapping)
    ):
        raise RuntimeError("full-clean Dense4k point endpoint contract changed")
    state_sha = dr1.model_state_sha256(state)
    if state_sha != str(payload.get("model_state_sha256", "")).lower():
        raise RuntimeError("full-clean Dense4k model-state SHA changed")
    normalizer, normalizer_report = normalizer_from_point(payload, device)
    config = ft12.validate_b210_model_config(
        payload["model_config"], sim_tke.b210_config("candidate")
    )
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    if dr1.model_state_sha256(model.state_dict()) != state_sha:
        raise RuntimeError("setting point normalizer changed the sealed point state")
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != ft12.EXPECTED_PARAMETERS:
        raise RuntimeError("full-clean Dense4k parameter count changed")
    report = {
        "path": str(path.resolve()),
        "file_sha256": actual_file_sha,
        "model_state_sha256": state_sha,
        "model_config_sha256": dr1.canonical_sha256(payload["model_config"]),
        "normalizer": normalizer_report,
        "schema": POINT_SCHEMA,
        "iteration": 4000,
        "endpoint": "ema_u4000",
        "parameters": parameters,
    }
    return payload, model.to(device), normalizer, report


def save_endpoint(
    path: Path,
    model: EnsembleUNet2d,
    point: Mapping[str, object],
    point_report: Mapping[str, object],
    normalizer_report: Mapping[str, object],
    validation: Mapping[str, object],
    manifest: Mapping[str, object],
    access: Mapping[str, object],
    training: Mapping[str, object],
) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    state_sha = dr1.model_state_sha256(state)
    frozen_sha = dr1.model_state_sha256(
        {key: value for key, value in state.items() if not key.startswith(uq_release.UQ_PREFIXES)}
    )
    if frozen_sha != training["frozen_state_sha256_before"]:
        raise RuntimeError("release endpoint changed frozen point state")
    payload = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "stage": "full_released_prefix_uq_refit",
        "phase": "real",
        "iteration": UPDATES,
        "endpoint": "fixed_terminal_full_prefix_u800",
        "primary": True,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": dict(point["model_config"]),
        "point_checkpoint": dict(point_report),
        "point_frozen_state_sha256": frozen_sha,
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": normalizer_report["sha256"],
        "validation_gate": dict(validation),
        "manifest": dict(manifest),
        "access_audit": dict(access),
        "training": dict(training),
        "optimizer_state_saved": False,
        "checkpoint_selection": False,
        "tail_arrays_read": 0,
        "point_state_bitwise": True,
        "zip_written": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "point_frozen_state_sha256": frozen_sha,
        "schema": SCHEMA,
        "iteration": UPDATES,
        "endpoint": "fixed_terminal_full_prefix_u800",
    }


def verify_endpoint(path: Path, seal: Mapping[str, object]) -> dict:
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("full-prefix UQ endpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("endpoint") != seal["endpoint"]
        or int(payload.get("iteration", -1)) != UPDATES
        or payload.get("point_state_bitwise") is not True
        or payload.get("checkpoint_selection") is not False
        or int(payload.get("tail_arrays_read", -1)) != 0
        or not isinstance(state, Mapping)
        or dr1.model_state_sha256(state) != seal["model_state_sha256"]
    ):
        raise RuntimeError("full-prefix UQ endpoint exact verification failed")
    frozen_sha = dr1.model_state_sha256(
        {key: value for key, value in state.items() if not key.startswith(uq_release.UQ_PREFIXES)}
    )
    if frozen_sha != seal["point_frozen_state_sha256"]:
        raise RuntimeError("verified endpoint point state drifted")
    return {
        "exact": True,
        "file_sha256": seal["file_sha256"],
        "model_state_sha256": seal["model_state_sha256"],
        "point_frozen_state_sha256": frozen_sha,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-prefix UQ refit requires CUDA")
    validation = validate_validation_gate(
        args.validation_aggregate, args.validation_aggregate_sha256
    )
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256
    )
    payload, model, normalizer, point_report = load_point(
        args.point_checkpoint, args.point_checkpoint_sha256, device
    )
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    audit = PrefixOnlyAudit()
    trajectories, _metadata, windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", audit
    )
    access_report = audit.report(rows)
    if (
        not access_report["allowlist_exact"]
        or access_report["tail_arrays_read"] != 0
        or access_report["excluded_opened"]
        or len(windows) != EXPECTED_WINDOWS
    ):
        raise RuntimeError("full-prefix UQ data access contract failed")
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows
    )
    training = uq_release.train_uq_terminal(
        model,
        dataset,
        list(range(len(windows))),
        normalizer,
        device,
        deadline,
        SEED,
        "full_prefix_refit",
    )
    if training["updates"] != UPDATES or not training["frozen_state_bitwise"]:
        raise RuntimeError("full-prefix UQ did not reach its fixed terminal")
    endpoint_path = args.output_dir / "full_prefix_native_uq_u800.pth"
    seal = save_endpoint(
        endpoint_path,
        model,
        payload,
        point_report,
        point_report["normalizer"],
        validation,
        manifest_report,
        access_report,
        training,
    )
    del model, dataset, trajectories
    gc.collect()
    torch.cuda.empty_cache()
    verification = verify_endpoint(endpoint_path, seal)
    result = {
        "schema": SUMMARY_SCHEMA,
        "experiment": EXPERIMENT,
        "decision": "FULL_PREFIX_UQ_REFIT_COMPLETE",
        "validation_gate": validation,
        "point": point_report,
        "manifest": manifest_report,
        "fit_windows": len(windows),
        "fit_trajectories": len(rows),
        "access_audit": access_report,
        "training": training,
        "endpoint": seal,
        "verification": verification,
        "elapsed_seconds": time.perf_counter() - started,
        "zip_written": False,
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "summary.json", result)
    print(json.dumps({
        "event": "b210_full_prefix_uq_refit_complete",
        "endpoint": seal,
        "elapsed_seconds": result["elapsed_seconds"],
    }), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint-sha256", required=True)
    parser.add_argument("--validation-aggregate", type=Path, required=True)
    parser.add_argument("--validation-aggregate-sha256", required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=1800.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
