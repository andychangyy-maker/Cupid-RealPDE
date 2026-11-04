#!/usr/bin/env python3
"""Paired clean81 Real transfer qualification for dense-connected MS-Closure.

This is deliberately a point-only bridge.  Both arms start from one sealed
released-Sim dense-connected MS endpoint and run the mature 4k Real recipe.
The control sees the attested stride-20 prefix windows without augmentation;
the candidate alone sees the mature stride-5/gain/noise/blur support.  No UQ,
package, hidden data, endpoint selection, or score threshold is involved.

Clean716 is opened only after *both* EMA-u4000 endpoint files are atomically
sealed and re-hashed.  The runner is independent of the PC3D entry point: the
latter is reused only for its already-audited released-Real loading,
normalization, augmentation, and metric definitions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

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
import train_multiple_shooting_closure_dense_stride5_connected_credit_long as ms_sim  # noqa: E402
import train_multiple_shooting_closure_scaled_long as ms_control  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real  # noqa: E402
from models.multiple_shooting_closure import (  # noqa: E402
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
)


SCHEMA = "msclosure_real_paired4k_clean716_point_v1"
ENDPOINT_SCHEMA = "msclosure_real_paired4k_ema_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_MSCLOSURE_REAL_PAIRED4K_CLEAN716"
EXPECTED_SIM_SCHEMA = "multiple_shooting_dense_stride5_connected_credit_held10_v1"
EXPECTED_SIM_UPDATES = 12_000
EXPECTED_CLEAN716_MANIFEST_SHA256 = (
    "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
UPDATES = 4_000
BATCH_SIZE = 8
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-5
EMA_DECAY = .999
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 12_000
SEED = 4
MSE_WARMUP_UPDATES = 1_500
ACCURACY_WEIGHTS = (5.0, 3.0, 2.0)
MSE_WEIGHT = .25
CONTROL_ARM = "control_stride20_no_augmentation"
CANDIDATE_ARM = "candidate_stride5_mature_augmentation"

# Exact released-clean716 FP32 B210 point reference reported alongside the
# existing PC3D Real qualification.  It is a local diagnostic, not an
# official/hidden leaderboard claim.
MATURE_B210_CLEAN716_REFERENCE = {
    "name": "mature_B210_FP32_clean716_point_anchor",
    "rel_l2_score": 96.0691669691,
    "tke_rel_l2_score": 81.2232378624,
    "mvpe_rel_l2_score": 96.6978105018,
    "point_contribution_46_10_9": 61.0169435372,
}


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def expected_model_config() -> dict[str, object]:
    return asdict(ms_control.MODEL_CONFIG)


def real_contract(*, arm: str, windows: int, stride: int,
                  augmentation: bool) -> dict[str, object]:
    if arm not in {CONTROL_ARM, CANDIDATE_ARM}:
        raise ValueError("unknown paired Real arm")
    if augmentation != (arm == CANDIDATE_ARM):
        raise ValueError("arm augmentation contract is inconsistent")
    return {
        "arm": arm,
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "sample_schedule": "seed4_epoch_randperm_with_short_final_batch",
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "grad_clip": GRAD_CLIP,
        "ema_decay": EMA_DECAY,
        "mse_warmup_updates": MSE_WARMUP_UPDATES,
        "accuracy_weights": list(ACCURACY_WEIGHTS),
        "mse_weight": MSE_WEIGHT,
        "loss": "u1_u1500_all3_mse_then_accuracy_5_3_2_plus_.25_all3_mse",
        "optimizer_windows": int(windows),
        "optimizer_stride": int(stride),
        "augmentation": (
            dict(real.AUGMENTATION) if augmentation
            else {"enabled": False}),
        "point_only": True,
    }


def _normalizer_state(normalizer) -> dict[str, torch.Tensor]:
    return {
        name: getattr(normalizer, name).detach().cpu().clone()
        for name in ("mean_input", "mean_target", "std_input", "std_target")
    }


def _normalizer_sha(normalizer) -> str:
    return base.state_sha(_normalizer_state(normalizer))


def _require_fp32_finite_state(state: Mapping[str, object]) -> None:
    if not state:
        raise ValueError("MS checkpoint model_state is empty")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"MS model state {name} is not a tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"MS model state {name} must be FP32, got {value.dtype}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"MS model state {name} is non-finite")


def validate_ms_checkpoint_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact dense-connected checkpoint contract before Real data."""
    required = {
        "schema", "endpoint_kind", "updates", "rollout_mode",
        "segment_gradient_mode", "model_config", "normalizer", "git",
        "model_state",
    }
    missing = required.difference(payload)
    if missing:
        raise KeyError("MS checkpoint lacks required fields: " + ", ".join(sorted(missing)))
    if payload["schema"] != EXPECTED_SIM_SCHEMA:
        raise ValueError("start checkpoint is not dense-connected MS-Closure")
    if payload["endpoint_kind"] != "ema999_u12000" or int(payload["updates"]) != EXPECTED_SIM_UPDATES:
        raise ValueError("Real bridge requires the sealed dense-connected EMA-u12000 endpoint")
    if (payload["rollout_mode"] != "carry_detached"
            or payload["segment_gradient_mode"] != "connected"):
        raise ValueError("MS start endpoint does not preserve dense-connected rollout semantics")
    if not isinstance(payload["model_config"], Mapping) or dict(payload["model_config"]) != expected_model_config():
        raise ValueError("MS start checkpoint model_config differs from the frozen 23M recipe")
    state = payload["model_state"]
    if not isinstance(state, Mapping):
        raise TypeError("MS start checkpoint model_state must be a mapping")
    _require_fp32_finite_state(state)
    normalizer = payload["normalizer"]
    required_normalizer = {"mean_input", "mean_target", "std_input", "std_target"}
    if not isinstance(normalizer, Mapping) or set(normalizer) != required_normalizer:
        raise ValueError("MS checkpoint normalizer record changed")
    for name in required_normalizer:
        value = torch.as_tensor(normalizer[name], dtype=torch.float32)
        if value.shape != (3,) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"MS checkpoint {name} is not a finite [3] vector")
        if name.startswith("std") and not bool((value > 0).all()):
            raise ValueError(f"MS checkpoint {name} must be positive")
        if name not in state or not torch.equal(state[name].cpu(), value):
            raise RuntimeError(f"MS state buffer {name} disagrees with checkpoint normalizer")
    return {
        "schema": str(payload["schema"]),
        "endpoint_kind": str(payload["endpoint_kind"]),
        "updates": int(payload["updates"]),
        "model_config_sha256": canonical_sha(dict(payload["model_config"])),
        "sim_normalizer_sha256": canonical_sha(dict(payload["normalizer"])),
        "model_state_sha256": base.state_sha(state),
    }


def load_ms_checkpoint(path: Path, expected_sha256: str,
                       device: torch.device) -> tuple[MultipleShootingClosure, dict[str, object]]:
    path = path.resolve()
    if len(expected_sha256) != 64 or any(value not in "0123456789abcdef" for value in expected_sha256.lower()):
        raise ValueError("--start-ms-sha256 must be a 64-character hexadecimal SHA-256")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = real.file_sha256(path)
    if observed_sha != expected_sha256.lower():
        raise RuntimeError("MS checkpoint file SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("MS checkpoint payload must be a mapping")
    report = validate_ms_checkpoint_payload(payload)
    config = MultipleShootingClosureConfig(**dict(payload["model_config"]))
    model = MultipleShootingClosure(
        config, rollout_mode=str(payload["rollout_mode"]),
        segment_gradient_mode=str(payload["segment_gradient_mode"])).float()
    state = payload["model_state"]
    model.load_state_dict(state, strict=True)
    state_sha = base.state_sha(model.state_dict())
    if state_sha != report["model_state_sha256"]:
        raise RuntimeError("MS checkpoint strict-load state identity mismatch")
    report.update({
        "path": str(path),
        "file_sha256": observed_sha,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_state": {name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()},
    })
    return model.to(device), report


def install_real_normalizer(model: MultipleShootingClosure, normalizer) -> None:
    """Rebind UV physical coordinates; pressure remains an unused zero channel."""
    state = _normalizer_state(normalizer)
    if any(value.shape != (3,) or value.dtype != torch.float32 for value in state.values()):
        raise ValueError("Real normalizer must expose FP32 [u,v,p] statistics")
    if not (bool(torch.equal(state["mean_input"][2], torch.zeros(())))
            and bool(torch.equal(state["mean_target"][2], torch.zeros(())))):
        raise ValueError("Real compatibility pressure means must remain exactly zero")
    if not (bool(torch.equal(state["std_input"][2], torch.ones(())))
            and bool(torch.equal(state["std_target"][2], torch.ones(())))):
        raise ValueError("Real compatibility pressure standard deviations must remain exactly one")
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)


def clone_start_model(start_model: MultipleShootingClosure,
                      start_state: Mapping[str, torch.Tensor], normalizer,
                      device: torch.device) -> MultipleShootingClosure:
    model = MultipleShootingClosure(
        start_model.config, rollout_mode=start_model.rollout_mode,
        segment_gradient_mode=start_model.segment_gradient_mode).float()
    model.load_state_dict(start_state, strict=True)
    install_real_normalizer(model, normalizer)
    return model.to(device)


class PairedPrefixAccessAudit:
    """Permit tail access only after immutable endpoint seals for both arms."""
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self._opened: dict[str, set[str]] = {"fit": set(), "clean716": set()}
        self.seals: dict[str, dict[str, object]] = {}

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if filename == real.EXCLUDED_BASENAME:
            raise RuntimeError("excluded trajectory reached MS Real H5 opener")
        if phase not in self._opened:
            raise ValueError(f"unsupported access phase {phase!r}")
        if filename in self._opened[phase]:
            raise RuntimeError(f"MS Real H5 opened twice in {phase}: {filename}")
        if phase == "fit" and self.seals:
            raise RuntimeError("fit prefix cannot be opened after an endpoint seal")
        if phase == "clean716":
            if set(self.seals) != {CONTROL_ARM, CANDIDATE_ARM}:
                raise RuntimeError("clean716 cannot open before both paired endpoint seals")
            for seal in self.seals.values():
                endpoint = Path(str(seal["path"]))
                if not endpoint.is_file() or real.file_sha256(endpoint) != seal["file_sha256"]:
                    raise RuntimeError("sealed MS Real endpoint changed before clean716")

    def record_open(self, row: Mapping[str, object], phase: str, boundary: int) -> None:
        self._opened[phase].add(str(row["file"]))
        self.events.append({
            "file": str(row["file"]), "phase": phase,
            "channels_opened": ["u", "v"], "pressure_opened": False,
            "max_read_frame_exclusive": int(boundary),
            "both_endpoints_sealed_before_open": (
                set(self.seals) == {CONTROL_ARM, CANDIDATE_ARM}),
        })

    def record_endpoint_seal(self, arm: str, seal: Mapping[str, object]) -> None:
        if arm not in {CONTROL_ARM, CANDIDATE_ARM} or arm in self.seals:
            raise RuntimeError("invalid or duplicate paired endpoint seal")
        self.seals[arm] = dict(seal)


def seal_endpoint(path: Path, *, arm: str, model: MultipleShootingClosure,
                  ema: ema_lib.FP32EMAShadow, start: Mapping[str, object],
                  normalizer, contract: Mapping[str, object],
                  schedule: Sequence[Sequence[int]]) -> dict[str, object]:
    if ema.updates != UPDATES:
        raise RuntimeError("MS Real endpoint is not EMA-u4000")
    ema_seal = ema.seal(model)
    state = ema_seal["model_state"]
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "arm": arm,
        "update": UPDATES,
        "model_config": asdict(model.config),
        "rollout_mode": model.rollout_mode,
        "segment_gradient_mode": model.segment_gradient_mode,
        "model_state": state,
        "model_state_sha256": base.state_sha(state),
        "normalizer": _normalizer_state(normalizer),
        "normalizer_sha256": _normalizer_sha(normalizer),
        "start_ms": {key: value for key, value in start.items()
                     if key != "model_state"},
        "contract": dict(contract),
        "sample_schedule_sha256": canonical_sha(
            [[int(index) for index in batch] for batch in schedule]),
        "ema": ema_seal,
        "real_pressure": "unmeasured_exact_zero_and_unscored",
    }
    real.atomic_torch_save(path, payload)
    return {
        "path": str(path.resolve()), "file_sha256": real.file_sha256(path),
        "model_state_sha256": payload["model_state_sha256"], "arm": arm,
        "schema": ENDPOINT_SCHEMA, "update": UPDATES,
    }


def load_sealed_endpoint(seal: Mapping[str, object], device: torch.device) -> MultipleShootingClosure:
    path = Path(str(seal["path"]))
    if not path.is_file() or real.file_sha256(path) != seal["file_sha256"]:
        raise RuntimeError("MS Real endpoint file seal mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("arm") != seal["arm"]
            or payload.get("update") != UPDATES):
        raise ValueError("invalid MS Real endpoint schema")
    model = MultipleShootingClosure(
        MultipleShootingClosureConfig(**payload["model_config"]),
        rollout_mode=payload["rollout_mode"],
        segment_gradient_mode=payload["segment_gradient_mode"]).float()
    model.load_state_dict(payload["model_state"], strict=True)
    if base.state_sha(model.state_dict()) != seal["model_state_sha256"]:
        raise RuntimeError("MS Real endpoint state seal mismatch")
    return model.to(device)


def train_arm(model: MultipleShootingClosure, dataset: real.RealWindowDataset,
              normalizer, *, arm: str, output_dir: Path,
              start: Mapping[str, object], device: torch.device,
              checkpoint_every: int) -> tuple[dict[str, object], dict[str, object]]:
    augmentation = arm == CANDIDATE_ARM
    stride = real.OPTIMIZER_STRIDE if augmentation else real.NORMALIZER_STRIDE
    contract = real_contract(arm=arm, windows=len(dataset), stride=stride,
                             augmentation=augmentation)
    schedule = real.make_sample_schedule(len(dataset), UPDATES, BATCH_SIZE)
    real.seed_real_recipe(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    trace = []
    started = time.perf_counter()
    for cursor, indices in enumerate(schedule, start=1):
        history_physical, target_physical = zip(*(dataset[index] for index in indices))
        history_physical = torch.stack(history_physical).to(device)
        target_physical = torch.stack(target_physical).to(device)
        history = (real.augment_real_history(history_physical, normalizer)
                   if augmentation else normalizer.preprocess(
                       history_physical, target_physical)[0])
        _unused, target = normalizer.preprocess(history_physical, target_physical)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        loss_args = argparse.Namespace(
            point_loss_mode="official_accuracy", mse_weight=MSE_WEIGHT,
            mse_warmup_updates=MSE_WARMUP_UPDATES)
        loss, details = mature_real.point_loss(
            loss_args, prediction, target, normalizer, ACCURACY_WEIGHTS, cursor)
        mse = real.real_mse(prediction, target)
        accuracy = loss if cursor <= MSE_WARMUP_UPDATES else loss - MSE_WEIGHT * mse
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite {arm} loss at update {cursor}")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError(f"non-finite {arm} gradient at update {cursor}")
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=cursor)
        scheduler.step()
        if cursor == 1 or cursor % checkpoint_every == 0 or cursor == UPDATES:
            row = {
                "update": cursor, "loss": float(loss.detach()),
                "mse_all3": float(mse.detach()), "accuracy": float(accuracy.detach()),
                "grad_norm": float(gradient.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({"event": "msclosure_real_paired4k_progress", "arm": arm, **row}, sort_keys=True), flush=True)
    endpoint = seal_endpoint(
        output_dir / arm / "ema999_u4000.pt", arm=arm, model=model, ema=ema,
        start=start, normalizer=normalizer, contract=contract, schedule=schedule)
    return {
        "contract": contract,
        "sample_schedule_sha256": canonical_sha([[int(index) for index in batch] for batch in schedule]),
        "trace": trace, "elapsed_seconds": time.perf_counter() - started,
    }, endpoint


def _metric_row(evaluation: Mapping[str, object]) -> dict[str, float]:
    pooled = evaluation["pooled"]
    return {
        "rel_l2_score": float(pooled["rel_l2_score"]),
        "tke_rel_l2_score": float(pooled["tke_rel_l2_score"]),
        "mvpe_rel_l2_score": float(pooled["mvpe_rel_l2_score"]),
        "point_contribution_46_10_9": float(pooled["public_point_contribution_46_10_9"]),
    }


def compare_metrics(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    keys = ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
            "point_contribution_46_10_9")
    return {key: float(candidate[key]) - float(reference[key]) for key in keys}


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError("formal MS Real bridge requires explicit acknowledgement")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal MS Real bridge requires CUDA")
    if args.updates != UPDATES or args.batch_size != BATCH_SIZE:
        raise ValueError("MS Real bridge freezes updates4000 and batch8")
    if args.clean716_manifest is None:
        raise ValueError("MS Real bridge requires frozen clean716 manifest")
    if any(value is None for value in (args.real_dir, args.clean_manifest, args.start_ms, args.output_dir)):
        raise ValueError("MS Real bridge requires data/manifests/start/output arguments")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing MS Real output directory")
    if (not args.clean716_manifest.is_file()
            or real.file_sha256(args.clean716_manifest)
            != EXPECTED_CLEAN716_MANIFEST_SHA256):
        raise RuntimeError("frozen clean716 manifest SHA-256 changed")
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError("formal MS Real bridge requires clean Git")

    # Validate the start checkpoint before opening released Real data.
    template, start = load_ms_checkpoint(args.start_ms, args.start_ms_sha256, device)
    start_state = start["model_state"]
    rows, manifest_report = real.load_clean_manifest(args.clean_manifest)
    audit = PairedPrefixAccessAudit()
    trajectories, sparse_windows = real.load_real_rows(
        args.real_dir, rows, phase="fit", audit=audit)
    normalizer, normalizer_report = real.normalizer_from_sparse_windows(
        trajectories, sparse_windows, device)
    dense_windows = real.dense_windows(trajectories)
    if len(sparse_windows) != real.EXPECTED_NORMALIZER_WINDOWS:
        raise RuntimeError("paired control window count changed")
    if len(dense_windows) != real.EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError("paired candidate dense window count changed")
    datasets = {
        CONTROL_ARM: real.RealWindowDataset(trajectories, sparse_windows),
        CANDIDATE_ARM: real.RealWindowDataset(trajectories, dense_windows),
    }
    template_state = clone_start_model(template, start_state, normalizer, device).state_dict()
    initial_state_sha = base.state_sha(template_state)
    del template, template_state

    args.output_dir.mkdir(parents=True)
    training, seals = {}, {}
    for arm in (CONTROL_ARM, CANDIDATE_ARM):
        model = clone_start_model(
            MultipleShootingClosure(
                template_config := MultipleShootingClosureConfig(**expected_model_config()),
                rollout_mode="carry_detached", segment_gradient_mode="connected"),
            start_state, normalizer, device)
        if base.state_sha(model.state_dict()) != initial_state_sha:
            raise RuntimeError("paired arms do not share the same Real-rebound initial state")
        training[arm], seals[arm] = train_arm(
            model, datasets[arm], normalizer, arm=arm, output_dir=args.output_dir,
            start=start, device=device, checkpoint_every=args.checkpoint_every)
        audit.record_endpoint_seal(arm, seals[arm])
        del model
        torch.cuda.empty_cache()

    tail_rows = real.load_clean716_rows(args.clean716_manifest, rows)
    tail_trajectories, tail_windows = real.load_real_rows(
        args.real_dir, tail_rows, phase="clean716", audit=audit)
    tail_dataset = real.RealWindowDataset(tail_trajectories, tail_windows)
    tail_aoas = [int(tail_rows[window.trajectory]["aoa"]) for window in tail_windows]
    evaluations = {}
    for arm in (CONTROL_ARM, CANDIDATE_ARM):
        endpoint_model = load_sealed_endpoint(seals[arm], device)
        evaluations[arm] = real.evaluate_clean716(
            endpoint_model, tail_dataset, normalizer, tail_aoas, device)
        del endpoint_model
        torch.cuda.empty_cache()
    control_row = _metric_row(evaluations[CONTROL_ARM])
    candidate_row = _metric_row(evaluations[CANDIDATE_ARM])
    result = {
        "schema": SCHEMA, "git": git,
        "start_ms": {key: value for key, value in start.items() if key != "model_state"},
        "manifest": manifest_report, "normalizer": normalizer_report,
        "paired_initial_state_sha256": initial_state_sha,
        "arms": {arm: {"training": training[arm], "endpoint": seals[arm],
                        "clean716": evaluations[arm]} for arm in (CONTROL_ARM, CANDIDATE_ARM)},
        "metrics": {CONTROL_ARM: control_row, CANDIDATE_ARM: candidate_row},
        "candidate_minus_control": compare_metrics(candidate_row, control_row),
        "control_minus_mature_B210_clean716": compare_metrics(control_row, MATURE_B210_CLEAN716_REFERENCE),
        "candidate_minus_mature_B210_clean716": compare_metrics(candidate_row, MATURE_B210_CLEAN716_REFERENCE),
        "mature_B210_clean716_reference": MATURE_B210_CLEAN716_REFERENCE,
        "access_audit": audit.events,
        "selection_gate": None,
        "selection": "none; both sealed EMA-u4000 endpoints are reported",
    }
    path = args.output_dir / "result.json"
    real.atomic_torch_save(args.output_dir / "protocol_seal.pt", {
        "schema": SCHEMA, "start_ms": result["start_ms"], "endpoints": seals,
        "paired_initial_state_sha256": initial_state_sha,
        "clean716_opened_after_both_seals": True,
    })
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "msclosure_real_paired4k_complete", "result": str(path.resolve()),
                      "candidate_minus_control": result["candidate_minus_control"],
                      "candidate_minus_mature_B210": result["candidate_minus_mature_B210_clean716"]}, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "formal"), default="contract")
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--clean716-manifest", type=Path)
    parser.add_argument("--start-ms", type=Path)
    parser.add_argument("--start-ms-sha256", default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps({"schema": SCHEMA, "acknowledgement": ACKNOWLEDGEMENT,
                          "control": real_contract(arm=CONTROL_ARM, windows=real.EXPECTED_NORMALIZER_WINDOWS,
                                                   stride=real.NORMALIZER_STRIDE, augmentation=False),
                          "candidate": real_contract(arm=CANDIDATE_ARM, windows=real.EXPECTED_OPTIMIZER_WINDOWS,
                                                     stride=real.OPTIMIZER_STRIDE, augmentation=True),
                          "selection_gate": None}, sort_keys=True))
        return
    result = run(args)
    print(json.dumps({"result": str((args.output_dir / "result.json").resolve()),
                      "candidate_minus_control": result["candidate_minus_control"]}, sort_keys=True))


if __name__ == "__main__":
    main()
