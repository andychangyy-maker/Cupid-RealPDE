#!/usr/bin/env python3
"""Fit a temporal-band residual while keeping a sealed B210 base exact.

This is a responsibility-isolation experiment, not an end-to-end continuation:
the released-data Standard24k base is loaded exactly and every pre-existing
parameter/buffer is frozen.  The zero-initialized temporal-band branch is the
only trainable module.  Two complete Reynolds conditions are excluded from
residual fitting and form the fixed gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

import torch
from torch.utils.data import DataLoader, Subset


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT, CODE_ROOT / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
from physics.closure_augmentation import ClosureResidualAugmentor  # noqa: E402
import train_b210_np_functional_sim as sim  # noqa: E402
import train_c208_real_sim_rehearsal as rehearsal  # noqa: E402
import train_c208_sim_hgrad_pretrain as sim_hgrad  # noqa: E402


TEMPORAL_BAND_METHOD = "b210_xcb_standard24k_frozen_temporal_band_residual_v1"
SPATIAL_WARP_METHOD = "b210_xcb_standard24k_frozen_spatial_transport_v1"
UPDATES = 2_000
LOG_INTERVAL = 200
HOLDOUT_RE = (10_125, 25_425)
OFFICIAL_ACCURACY_WEIGHTS = (46.0, 10.0, 9.0)
POINT_DELTA_MIN = 0.030
COMPONENT_DELTA_MIN = -0.030
SPS_DELTA_MIN = -0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache-sha256", required=True)
    parser.add_argument("--real-closure-bank", type=Path, required=True)
    parser.add_argument("--real-closure-bank-sha256", required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--spatial-warp", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    expected = expected.lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"invalid {label} SHA-256")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 changed: {actual} != {expected}")
    return actual


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def validate_base_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise TypeError("base checkpoint has no model_state_dict")
    config = payload.get("model_config")
    training = payload.get("training_config")
    if not isinstance(config, Mapping) or not isinstance(training, Mapping):
        raise TypeError("base checkpoint contracts are absent")
    if config.get("architecture") != "ensemble_unet":
        raise ValueError("base checkpoint is not an ensemble U-Net")
    if not bool(config.get("cross_conditioned_backbone")):
        raise ValueError("base checkpoint has no cross conditioner")
    if int(config.get("member_adapter_channels", 0)) != 64:
        raise ValueError("base checkpoint has no standard member adapters")
    if bool(config.get("temporal_band_mixer", False)):
        raise ValueError("base checkpoint already contains a temporal mixer")
    if training.get("method") != (
            "b210_xcb_member_standard_stride20_20k_then_4k_v1"):
        raise ValueError("base checkpoint is not the sealed Standard24k arm")
    if int(training.get("window_stride", -1)) != 20:
        raise ValueError("base checkpoint does not use stride 20")


def split_indices(dataset, manifest: list[dict]) -> tuple[list[int], list[int], dict[int, list[int]]]:
    if len(manifest) != 100 or len(dataset) != 4_800:
        raise ValueError("released Sim dataset cardinality changed")
    held_trajectories = {
        index for index, item in enumerate(manifest)
        if int(item["re"]) in HOLDOUT_RE
    }
    if len(held_trajectories) != 10:
        raise ValueError("frozen Reynolds holdout cardinality changed")
    train: list[int] = []
    held: list[int] = []
    by_re = {value: [] for value in HOLDOUT_RE}
    for index, window in enumerate(dataset.windows):
        trajectory = int(window.trajectory)
        if trajectory in held_trajectories:
            held.append(index)
            by_re[int(manifest[trajectory]["re"])].append(index)
        else:
            train.append(index)
    if len(train) != 4_320 or len(held) != 480:
        raise ValueError("frozen residual train/holdout split changed")
    if any(len(indices) != 240 for indices in by_re.values()):
        raise ValueError("each held Reynolds condition must have 240 windows")
    return train, held, by_re


def build_models(payload: Mapping[str, object], device: torch.device,
                 spatial_warp: bool = False,
                 ) -> tuple[EnsembleUNet2d, EnsembleUNet2d, list[str]]:
    raw = dict(payload["model_config"])
    if raw.pop("architecture") != "ensemble_unet":
        raise ValueError("base architecture identity changed")
    base = EnsembleUNet2d(EnsembleUNet2dConfig(**raw)).float()
    base.load_state_dict(payload["model_state_dict"], strict=True)

    candidate_config = dict(raw)
    if spatial_warp:
        candidate_config.update({
            "spatial_transport_warp": True,
            "spatial_transport_channels": 32,
            "spatial_transport_max_displacement": 1.5,
        })
    else:
        candidate_config.update({
            "temporal_band_mixer": True,
            "temporal_band_channels": sim.TEMPORAL_BAND_CHANNELS,
            "temporal_band_count": sim.TEMPORAL_BAND_COUNT,
            "temporal_band_log_gain_limit": sim.TEMPORAL_BAND_LOG_GAIN_LIMIT,
            "temporal_band_phase_limit": sim.TEMPORAL_BAND_PHASE_LIMIT,
        })
    candidate = EnsembleUNet2d(
        EnsembleUNet2dConfig(**candidate_config)).float()
    incompatible = candidate.load_state_dict(
        payload["model_state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"base state has unexpected candidate keys: {incompatible.unexpected_keys}")
    missing = sorted(incompatible.missing_keys)
    if not missing or any(
            not key.startswith("temporal_transport.") for key in missing):
        raise RuntimeError(f"candidate missing-key contract changed: {missing}")
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    if candidate.temporal_transport is None:
        raise RuntimeError("candidate temporal mixer was not constructed")
    for parameter in candidate.temporal_transport.parameters():
        parameter.requires_grad_(True)
    base.requires_grad_(False)
    return base.to(device).eval(), candidate.to(device).eval(), missing


@torch.no_grad()
def update0_identity(base: EnsembleUNet2d, candidate: EnsembleUNet2d,
                     dataset, normalizer, device: torch.device) -> dict:
    history, _target = dataset[0]
    history = history.unsqueeze(0).to(device)
    history_norm, _unused = normalizer.preprocess(history, history)
    base_outputs = base.forward_uq_training(history_norm)
    candidate_outputs = candidate.forward_uq_training(history_norm)
    names = ("prediction", "members", "quantiles", "lower", "upper")
    exact = {
        name: bool(torch.equal(base_outputs[index], candidate_outputs[index]))
        for index, name in enumerate(names)
    }
    return {"outputs": exact, "pass": all(exact.values())}


def freeze_contract(candidate: EnsembleUNet2d,
                    expected_parameters: int | None = None) -> dict:
    trainable = [
        name for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        name for name, parameter in candidate.named_parameters()
        if not parameter.requires_grad
    ]
    if not trainable or any(
            not name.startswith("temporal_transport.") for name in trainable):
        raise RuntimeError("non-temporal parameter is trainable")
    count = sum(
        parameter.numel() for parameter in candidate.parameters()
        if parameter.requires_grad)
    expected_parameters = (
        sim.EXPECTED_TEMPORAL_BAND_MODULE_PARAMETERS
        if expected_parameters is None else expected_parameters)
    if count != expected_parameters:
        raise RuntimeError(f"trainable temporal parameter count changed: {count}")
    return {
        "trainable_parameters": count,
        "trainable_tensor_names": trainable,
        "frozen_tensor_count": len(frozen),
    }


@torch.no_grad()
def base_state_exact(candidate: EnsembleUNet2d,
                     base_state: Mapping[str, torch.Tensor]) -> bool:
    current = candidate.state_dict()
    for name, expected in base_state.items():
        if name.startswith("temporal_transport."):
            raise RuntimeError("sealed base unexpectedly has temporal state")
        if name not in current or not torch.equal(current[name].cpu(), expected):
            return False
    return True


def metric_delta(candidate: Mapping[str, float],
                 reference: Mapping[str, float]) -> dict[str, float]:
    keys = (
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "native_sps_score")
    return {key: float(candidate[key]) - float(reference[key]) for key in keys}


def gate(reference: Mapping[str, float], candidate: Mapping[str, float],
         by_re_reference: Mapping[str, Mapping[str, float]],
         by_re_candidate: Mapping[str, Mapping[str, float]],
         mechanics: Mapping[str, bool]) -> dict:
    delta = metric_delta(candidate, reference)
    point = (
        0.46 * delta["rel_l2_score"]
        + 0.10 * delta["tke_rel_l2_score"]
        + 0.09 * delta["mvpe_rel_l2_score"])
    by_re_point = {}
    for key in sorted(by_re_reference):
        local = metric_delta(by_re_candidate[key], by_re_reference[key])
        by_re_point[key] = (
            0.46 * local["rel_l2_score"]
            + 0.10 * local["tke_rel_l2_score"]
            + 0.09 * local["mvpe_rel_l2_score"])
    checks = {
        "point_delta": point >= POINT_DELTA_MIN,
        "component_floors": all(
            delta[key] >= COMPONENT_DELTA_MIN
            for key in (
                "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")),
        "sps_floor": delta["native_sps_score"] >= SPS_DELTA_MIN,
        "both_reynolds_nonnegative": all(value >= 0.0 for value in by_re_point.values()),
        "mechanics": all(bool(value) for value in mechanics.values()),
        "finite": all(math.isfinite(value) for value in (*delta.values(), point, *by_re_point.values())),
    }
    return {
        "delta": delta,
        "point65_delta": point,
        "by_re_point65_delta": by_re_point,
        "checks": checks,
        "pass": all(checks.values()),
        "decision": "PASS_TO_REAL_RESPONSIBILITY_ISOLATION" if all(checks.values()) else "STOP_AFTER_FROZEN_RESIDUAL",
    }


def main() -> None:
    cli = parse_args()
    method = SPATIAL_WARP_METHOD if cli.spatial_warp else TEMPORAL_BAND_METHOD
    expected_residual_parameters = (
        sim.EXPECTED_SPATIAL_TRANSPORT_MODULE_PARAMETERS
        if cli.spatial_warp
        else sim.EXPECTED_TEMPORAL_BAND_MODULE_PARAMETERS)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    base_sha = require_sha(
        cli.base_checkpoint, cli.base_checkpoint_sha256, "base checkpoint")
    teacher_sha = require_sha(
        cli.teacher_cache, cli.teacher_cache_sha256, "teacher cache")
    closure_sha = require_sha(
        cli.real_closure_bank, cli.real_closure_bank_sha256, "closure bank")
    payload = torch.load(
        cli.base_checkpoint, map_location="cpu", weights_only=False)
    validate_base_payload(payload)
    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal frozen residual stage requires CUDA")

    training = copy.deepcopy(payload["training_config"])
    args = argparse.Namespace(**training)
    args.method = method
    args.output_dir = cli.output_dir
    args.sim_dir = cli.sim_dir
    args.real_stats = cli.real_stats
    args.teacher_cache = cli.teacher_cache
    args.device = cli.device
    args.num_workers = cli.num_workers
    args.accuracy_weights = OFFICIAL_ACCURACY_WEIGHTS
    args.temporal_band_mixer = not cli.spatial_warp
    args.spatial_transport_warp = cli.spatial_warp
    args.train_only_temporal_band_residual = True
    args.official_point_objective = False
    args.u200_health_pass = True

    rehearsal.ensemble_train.seed_everything(int(args.seed))
    torch.set_float32_matmul_precision("high")
    raw_loader, normalizer, train_windows, trajectories = (
        rehearsal.ensemble_train.common.prepare_sim(args, device))
    manifest = json.loads(
        (cli.output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train_indices, held_indices, held_by_re = split_indices(
        raw_loader.dataset, manifest)
    teacher_cache, teacher_metadata = rehearsal.load_and_validate_cache(
        cli.teacher_cache, manifest, raw_loader.dataset)
    soft_normalizer = rehearsal.ensemble_train.common.repro.Normalizer(
        cli.real_stats, device)
    normalizer_values = rehearsal.ensemble_train.common.normalizer_dict(normalizer)
    scale_contract = sim_hgrad.compute_sim_hgrad_scale_contract(
        raw_loader.dataset, rehearsal.canonical_sha256(manifest),
        rehearsal.canonical_sha256(normalizer_values))
    write_json(cli.output_dir / "hgrad_scale_contract.json", scale_contract)

    closure = ClosureResidualAugmentor.load(
        cli.real_closure_bank, manifest,
        candidate_count=sim.REAL_CLOSURE_CANDIDATE_COUNT).to(device)
    base, candidate, missing = build_models(
        payload, device, spatial_warp=cli.spatial_warp)
    identity = update0_identity(
        base, candidate, raw_loader.dataset, normalizer, device)
    if not identity["pass"]:
        raise RuntimeError("zero-initialized residual is not exact identity")
    frozen = freeze_contract(candidate, expected_residual_parameters)
    if not base_state_exact(candidate, payload["model_state_dict"]):
        raise RuntimeError("candidate did not load the sealed base exactly")

    def evaluate(model: EnsembleUNet2d, indices: list[int]) -> dict[str, float]:
        return sim_hgrad.evaluate_anchor(
            model, raw_loader.dataset, indices, normalizer,
            soft_normalizer, teacher_cache, scale_contract, args, device)

    reference = evaluate(base, held_indices)
    reference_by_re = {
        str(value): evaluate(base, indices)
        for value, indices in held_by_re.items()
    }
    del base
    torch.cuda.empty_cache()

    tagged = rehearsal.ensemble_train.TrajectoryTaggedDataset(
        raw_loader.dataset)
    loader = DataLoader(
        Subset(tagged, train_indices), batch_size=int(args.batch_size),
        shuffle=True, num_workers=int(args.num_workers), pin_memory=True,
        generator=torch.Generator().manual_seed(int(args.seed) + 24_000))
    iterator = rehearsal.ensemble_train.common.repro.cycling(loader)
    optimizer = torch.optim.AdamW(
        candidate.temporal_transport.parameters(), lr=float(args.lr),
        weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES)
    initial_temporal = sim.temporal_transport_state(candidate)
    output_gradient_seen = False
    body_gradient_seen = False
    history = []
    started = time.monotonic()
    for iteration in range(1, UPDATES + 1):
        values, _audit, _key, flags = sim.sim_update(
            candidate, next(iterator), normalizer, soft_normalizer,
            teacher_cache, scale_contract, args, device, iteration,
            optimizer, scheduler, "candidate", closure, False)
        output_gradient_seen |= flags["transport_output_nonzero"]
        body_gradient_seen |= flags["transport_body_nonzero"]
        if iteration % LOG_INTERVAL == 0 or iteration == UPDATES:
            row = {"iteration": iteration, **values}
            history.append(row)
            print(json.dumps({
                "event": "b210_frozen_structural_residual_progress", **row,
            }, sort_keys=True), flush=True)

    candidate_metrics = evaluate(candidate, held_indices)
    candidate_by_re = {
        str(value): evaluate(candidate, indices)
        for value, indices in held_by_re.items()
    }
    final_temporal = sim.temporal_transport_state(candidate)
    mechanics = {
        "update0_identity": identity["pass"],
        "output_gradient_seen": output_gradient_seen,
        "body_gradient_seen": body_gradient_seen,
        "temporal_state_updated": not sim.tensor_state_equal(
            initial_temporal, final_temporal),
        "sealed_base_exact_after_training": base_state_exact(
            candidate, payload["model_state_dict"]),
    }
    result_gate = gate(
        reference, candidate_metrics, reference_by_re,
        candidate_by_re, mechanics)
    checkpoint = None
    checkpoint_sha = None
    if result_gate["pass"]:
        checkpoint_path = cli.output_dir / "model_2000.pth"
        rehearsal.ensemble_train.common.save_checkpoint(
            checkpoint_path, candidate, args, UPDATES, history,
            normalizer_values, {"status": result_gate["decision"]})
        checkpoint = str(checkpoint_path.resolve())
        checkpoint_sha = sha256(checkpoint_path)
    summary = {
        "method": method,
        "status": result_gate["decision"],
        "completed_updates": UPDATES,
        "wall_seconds": time.monotonic() - started,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "base_checkpoint": str(cli.base_checkpoint.resolve()),
        "base_checkpoint_sha256": base_sha,
        "teacher_cache_sha256": teacher_sha,
        "closure_bank_sha256": closure_sha,
        "train_windows": len(train_indices),
        "held_windows": len(held_indices),
        "held_reynolds": list(HOLDOUT_RE),
        "total_released_windows": train_windows,
        "trajectories": trajectories,
        "missing_temporal_state_keys": missing,
        "freeze_contract": frozen,
        "mechanics": mechanics,
        "update0_identity": identity,
        "reference": reference,
        "candidate": candidate_metrics,
        "reference_by_re": reference_by_re,
        "candidate_by_re": candidate_by_re,
        "gate": result_gate,
        "history": history,
        "training_contract": {
            "only_temporal_transport_trainable": True,
            "residual_kind": (
                "spatial_transport" if cli.spatial_warp else "temporal_band"),
            "base_trained_from_random_on_released_data": True,
            "residual_training_excludes_complete_reynolds_conditions": True,
            "official_accuracy_weights": list(OFFICIAL_ACCURACY_WEIGHTS),
            "updates": UPDATES,
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "controller_only_tke_auxiliary_disabled": True,
            "real_closure_augmentation_retained": True,
            "external_data_or_model": False,
        },
        "model_config": candidate.architecture_config(),
    }
    write_json(cli.output_dir / "summary.json", summary)
    print(json.dumps({
        "event": "b210_frozen_structural_residual_complete",
        "decision": result_gate["decision"],
        "pass": result_gate["pass"],
        "point65_delta": result_gate["point65_delta"],
        "by_re_point65_delta": result_gate["by_re_point65_delta"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
