#!/usr/bin/env python3
"""Recover the RDE-MoE clean716 result from six immutable endpoints.

The original run completed all training and sealed raw/EMA endpoints for R0,
R1, and R2.  It then evaluated clean716 but stopped before writing result.json
because a post-evaluation assertion incorrectly required an EMA shadow of a
frozen FP32 router to remain bitwise equal to its source.  This program is
bound to those exact six files.  It validates all training-side evidence
before opening the evaluation manifest, then reruns evaluation only.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "code" / "training"
for directory in (REPO_ROOT / "code", TRAINING_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_rde_moe_real4k as rde  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2dConfig,
)


SCHEMA = "b210_rde_moe_real4k_sealed_recovery_v1"
ACKNOWLEDGEMENT = "RECOVER_B210_RDE_MOE_REAL4K_FROM_SIX_SEALS_V1"
ORIGINAL_RUN_COMMIT = "78e862fcc05a5abc4859b271106c3a23982dcd6a"
EXPECTED_FAILED_LOG_SHA256 = (
    "8d9e171142a653dcb77ab9cdc2733ca2faf09999cd113bb9f8a5629248b15167")
EXPECTED_ROUTER_FILE_SHA256 = (
    "956f4b3322807ed6ffff77b1aa7db37bf8127f1f55e2c5ca592d83bf350a6150")
EXPECTED_ROUTER_STATE_SHA256 = (
    "06fd59407d177e8bdd88275638abe11f5d696354c3131fae16bb0e2fa376c1d0")
EXPECTED_NORMALIZER_FILE_SHA256 = (
    "86b5feec17db74bca9dc909ed5baae93e4775c83f62db98bd46032a48f5bfc7f")
EXPECTED_NORMALIZER_SHA256 = (
    "c87f669fdf6cc6a67a610cae414c210e5927782dc252ce65f540c2241a454d3a")
EXPECTED_SAMPLE_STREAM_SHA256 = (
    "14331e0ff2d61c7f62a384061b47f629f6a71999fad6ba4e97f37d414f17a7fb")

SEALED_ARTIFACTS = {
    "r0_single": {
        "raw": {
            "name": "raw_u4000_r0_single.pth",
            "file_sha256": (
                "312013973373adc49593ada082293fc67e7cec08b4fcac3470175c3ea35ba4a2"),
            "model_state_sha256": (
                "c20fb27d48b7573db265b3bd816997c8e5f32d3105eab66bc9850c00126b4977"),
        },
        "ema": {
            "name": "ema_u4000_r0_single.pth",
            "file_sha256": (
                "6633a746e35a99a3f25cbfe4e9239d3aea08fad7259bb441188550fa550d957d"),
            "model_state_sha256": (
                "fa8ddfaaf4d539fd08ff7d833c5a21e3402238c0898c2e29dad9484ee966c08b"),
        },
    },
    "r1_fixed_average": {
        "raw": {
            "name": "raw_u4000_r1_fixed_average.pth",
            "file_sha256": (
                "bb6ff80ac5eae5ce2b4c9239da48410fee8f73cb9fb6c0660a9d07cf3aca7125"),
            "model_state_sha256": (
                "160ed1be52f98cb0bcf636d73fb71157aa1d43a74890291f29e55b7f33172b2f"),
        },
        "ema": {
            "name": "ema_u4000_r1_fixed_average.pth",
            "file_sha256": (
                "222354e25fc9d0d871c7de97343b2d8fb25b5fe7be8ed1d95bdbcbdda1ffbb7a"),
            "model_state_sha256": (
                "6b16e78df366d14c45312e2b0107d001c1db8608a8afeca12d8778b8197cd2a2"),
        },
    },
    "r2_history_router": {
        "raw": {
            "name": "raw_u4000_r2_history_router.pth",
            "file_sha256": (
                "646ec7a36f32f9872d1117407d3f81b8b3655a7361cf1b3e704b1c451930bec8"),
            "model_state_sha256": (
                "5dd3a3e0eff7de69b1a2fa0e0c49daf4b2f4f1bfc2809c147f53958182f236b7"),
        },
        "ema": {
            "name": "ema_u4000_r2_history_router.pth",
            "file_sha256": (
                "b7920520cb2c7038ff5f8ff8b0d103dda917c8442628fd2e27842da1ad7013e4"),
            "model_state_sha256": (
                "e61f455b3ecef8896c6e79f7c289156982939d16099a3f9c1edd32f87556faa3"),
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-dir", type=Path, required=True)
    parser.add_argument("--failed-log", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_failed_log(path: Path) -> dict[str, object]:
    if rde.dr1.sha256(path) != EXPECTED_FAILED_LOG_SHA256:
        raise RuntimeError("failed-run log SHA changed")
    text = path.read_text(encoding="utf-8")
    marker = "RuntimeError: R1/R2 raw/EMA router or standardization changed"
    if marker not in text or "line 1122, in run" not in text:
        raise RuntimeError("failed-run log no longer proves the terminal assertion")
    return {
        "path": str(path.resolve()),
        "sha256": EXPECTED_FAILED_LOG_SHA256,
        "terminal_exception": marker,
        "failure_stage": "after six clean716 endpoint evaluations",
    }


def validate_router_package(
        path: Path, trajectories, windows, metadata, normalizer,
        ) -> tuple[dict[str, object], dict[str, object]]:
    if rde.dr1.sha256(path) != EXPECTED_ROUTER_FILE_SHA256:
        raise RuntimeError("sealed prefix-router package SHA changed")
    package = torch.load(path, map_location="cpu", weights_only=False)
    if (package.get("schema") != rde.ROUTER_SCHEMA
            or package.get("router_state_sha256")
            != EXPECTED_ROUTER_STATE_SHA256
            or package.get("label_one_aoas")
            != list(rde.ROUTER_POSITIVE_AOAS)
            or package.get("label_zero_aoas")
            != list(rde.ROUTER_NEGATIVE_AOAS)
            or package.get("counts") != {
                "positive": rde.EXPECTED_ROUTER_POSITIVES,
                "negative": rde.EXPECTED_ROUTER_NEGATIVES}):
        raise RuntimeError("sealed prefix-router payload changed")
    state = package.get("router_state_dict")
    digest = OrderedDict(
        [(f"router.{name}", value) for name, value in state.items()]
        + [("descriptor_mean", package["descriptor_mean"]),
           ("descriptor_std", package["descriptor_std"])])
    if rde.dr1.model_state_sha256(digest) != EXPECTED_ROUTER_STATE_SHA256:
        raise RuntimeError("prefix-router tensor digest changed")
    descriptor, labels, aoas = rde.prefix_router_tensors(
        trajectories, windows, metadata, normalizer)
    mean = descriptor.mean(dim=0)
    std = descriptor.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    if (not torch.equal(mean, package["descriptor_mean"])
            or not torch.equal(std, package["descriptor_std"])):
        raise RuntimeError("router descriptor population statistics changed")
    router = nn.Sequential(nn.Linear(80, 16), nn.Tanh(), nn.Linear(16, 1))
    router.load_state_dict(state, strict=True)
    router.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(
            router((descriptor - mean) / std).squeeze(-1))
    report = {
        "path": str(path.resolve()),
        "file_sha256": EXPECTED_ROUTER_FILE_SHA256,
        "router_state_sha256": EXPECTED_ROUTER_STATE_SHA256,
        "architecture": "80->16 tanh->1",
        "descriptor": "history_regime_summary(normalized_history.detach())",
        "descriptor_standardization": "dense-prefix population mean/std",
        "windows": len(windows),
        "target_tensors_used": False,
        "tail_or_clean716_used": False,
        "fit_metrics_replayed_from_sealed_router": rde.router_metrics(
            labels, probability, aoas),
    }
    return package, report


def validate_ema_metadata(
        metadata: Mapping[str, object], state: Mapping[str, torch.Tensor],
        raw_model_state_sha256: str) -> None:
    if (metadata.get("schema") != ema_lib.EMA_SEAL_SCHEMA
            or metadata.get("update_order") != ema_lib.UPDATE_ORDER_CONTRACT
            or metadata.get("shadow_dtype") != str(torch.float32)
            or int(metadata.get("updates", -1)) != rde.UPDATES
            or float(metadata.get("decay", -1.0)) != rde.EMA_DECAY
            or metadata.get("raw_model_state_sha256")
            != raw_model_state_sha256
            or metadata.get("materialized_model_state_sha256")
            != rde.dr1.model_state_sha256(state)):
        raise RuntimeError("EMA endpoint metadata changed")
    manifest = metadata.get("parameter_manifest")
    buffers = metadata.get("buffer_names")
    if not isinstance(manifest, list) or not isinstance(buffers, list):
        raise RuntimeError("EMA manifest/buffer classification is missing")
    if ema_lib.canonical_sha256(manifest) != metadata.get(
            "parameter_manifest_sha256"):
        raise RuntimeError("EMA parameter manifest digest changed")
    parameter_names = [str(row["name"]) for row in manifest]
    parameter_state = OrderedDict((name, state[name]) for name in parameter_names)
    buffer_state = OrderedDict((name, state[name]) for name in buffers)
    if (ema_lib.tensor_state_sha256(parameter_state)
            != metadata.get("shadow_parameter_state_sha256")
            or ema_lib.tensor_state_sha256(buffer_state)
            != metadata.get("buffer_state_sha256")):
        raise RuntimeError("EMA parameter/buffer state digest changed")
    unsigned = dict(metadata)
    recorded = unsigned.pop("seal_metadata_sha256", None)
    if ema_lib.canonical_sha256(unsigned) != recorded:
        raise RuntimeError("EMA seal metadata digest changed")


def endpoint_seal(
        sealed_dir: Path, arm: str, kind: str,
        expected: Mapping[str, str], schedule_sha: str,
        normalizer_report: Mapping[str, object], base_config,
        ) -> tuple[dict[str, object], dict[str, object], OrderedDict | None]:
    path = sealed_dir / arm / expected["name"]
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model_state_dict")
    primary = kind == "ema"
    expected_endpoint = f"{kind}_u4000_{arm}"
    if (payload.get("schema") != rde.ENDPOINT_SCHEMA
            or payload.get("experiment") != rde.EXPERIMENT
            or payload.get("phase") != "real"
            or int(payload.get("iteration", -1)) != rde.UPDATES
            or int(payload.get("held_aoa", 0)) != -1
            or payload.get("endpoint") != expected_endpoint
            or payload.get("primary") is not primary
            or payload.get("sample_stream_sha256") != schedule_sha
            or payload.get("normalizer_sha256")
            != normalizer_report["sha256"]
            or payload.get("normalizer") != normalizer_report["values"]
            or payload.get("optimizer_state_saved") is not False
            or not isinstance(state, Mapping)
            or payload.get("model_state_sha256")
            != expected["model_state_sha256"]
            or rde.dr1.model_state_sha256(state)
            != expected["model_state_sha256"]):
        raise RuntimeError(f"sealed endpoint payload changed: {arm}/{kind}")
    start = payload.get("start")
    if (not isinstance(start, Mapping)
            or start.get("checkpoint_sha256") != rde.EXPECTED_START_SHA256
            or start.get("phase") != "sim"
            or int(start.get("iteration", -1)) != rde.EXPECTED_START_ITERATION
            or start.get("status") != rde.EXPECTED_START_STATUS
            or start.get("method") != rde.EXPECTED_START_METHOD
            or start.get("arm") != arm
            or start.get("dual_decoder_mode") != rde.ARM_MODE[arm]
            or int(start.get("parameters", -1)) != rde.expected_parameters(arm)):
        raise RuntimeError(f"endpoint start provenance changed: {arm}/{kind}")
    config_payload = dict(payload["model_config"])
    if config_payload.pop("architecture", None) != "ensemble_unet":
        raise RuntimeError("endpoint architecture changed")
    config = EnsembleUNet2dConfig(**config_payload)
    if config != rde.arm_config(base_config, arm):
        raise RuntimeError(f"endpoint model config changed: {arm}/{kind}")
    if kind == "raw":
        if (payload.get("ema") is not None
                or payload.get("clean716_opened_before_seal") is not False):
            raise RuntimeError("raw endpoint seal/access evidence changed")
    else:
        if payload.get("held_prefix_opened_before_seal") is not False:
            raise RuntimeError("EMA endpoint seal/access evidence changed")
        metadata = payload.get("ema")
        if not isinstance(metadata, Mapping):
            raise RuntimeError("EMA endpoint has no EMA metadata")
        raw_sha = SEALED_ARTIFACTS[arm]["raw"]["model_state_sha256"]
        validate_ema_metadata(metadata, state, raw_sha)
    seal = {
        "endpoint": expected_endpoint,
        "path": str(path.resolve()),
        "file_sha256": expected["file_sha256"],
        "model_state_sha256": expected["model_state_sha256"],
        "model_config_sha256": rde.dr1.canonical_sha256(
            payload["model_config"]),
        "sample_stream_sha256": schedule_sha,
        "iteration": rde.UPDATES,
        "primary": primary,
        "optimizer_state_saved": False,
    }
    router = (None if arm == "r0_single"
              else rde.endpoint_router_state(state))
    row = {
        "model_config": payload["model_config"],
        "start": dict(start),
        "seal": seal,
    }
    del payload, state
    gc.collect()
    return seal, row, router


def validate_pre_tail(args: argparse.Namespace) -> dict[str, object]:
    git = rde.git_report()
    if not git["clean"]:
        raise RuntimeError(
            f"recovery requires a clean Git worktree: {git['porcelain']}")
    if not args.sealed_dir.is_dir():
        raise FileNotFoundError(args.sealed_dir)
    if (args.sealed_dir / "result.json").exists():
        raise FileExistsError(args.sealed_dir / "result.json")
    failed_log = validate_failed_log(args.failed_log)
    if rde.dr1.sha256(args.start_sim) != rde.EXPECTED_START_SHA256:
        raise RuntimeError("Standard24k start checkpoint SHA changed")
    start_payload = torch.load(
        str(args.start_sim), map_location="cpu", weights_only=False, mmap=True)
    base_config, _start_state, start_report = rde.parse_start_payload(
        start_payload)
    del start_payload, _start_state
    if rde.dr1.sha256(args.clean_manifest) != rde.EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("clean81 manifest SHA changed")
    if not args.evaluation_manifest.is_file():
        raise FileNotFoundError(args.evaluation_manifest)
    rows, manifest = rde.dr1.load_manifest(
        args.clean_manifest, rde.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    prefix_audit = rde.dense4k.PrefixAudit()
    trajectories, metadata, sparse_windows = rde.dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    dense_windows = rde.dense4k.dense_windows(
        trajectories, rde.dense4k.CANDIDATE_STRIDE)
    if (len(sparse_windows) != rde.dr1.EXPECTED_WINDOWS
            or len(dense_windows) != rde.EXPECTED_DENSE_WINDOWS):
        raise RuntimeError("training-prefix window contract changed")
    sufficient = rde.smu1.window_sufficient_statistics(
        trajectories, sparse_windows)
    statistics = rde.smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = rde.smu1.FoldNormalizer(statistics, torch.device("cpu"))
    normalizer_report = rde.ft12.normalizer_payload(normalizer)
    if normalizer_report["sha256"] != EXPECTED_NORMALIZER_SHA256:
        raise RuntimeError("recomputed Real normalizer changed")
    normalizer_path = args.sealed_dir / "point_normalizer.pt"
    if rde.dr1.sha256(normalizer_path) != EXPECTED_NORMALIZER_FILE_SHA256:
        raise RuntimeError("sealed normalizer file SHA changed")
    saved_statistics = torch.load(
        normalizer_path, map_location="cpu", weights_only=False)
    if (len(saved_statistics) != len(statistics)
            or any(not torch.equal(left, right) for left, right in zip(
                saved_statistics, statistics))):
        raise RuntimeError("sealed normalizer tensors changed")
    schedule = rde.smu1.paired_batch_schedule(
        range(len(dense_windows)), dense_windows,
        updates=rde.UPDATES, seed=rde.small4k.SEED)
    schedule["updates"] = rde.UPDATES
    if schedule["sha256"] != EXPECTED_SAMPLE_STREAM_SHA256:
        raise RuntimeError("recomputed paired sample stream changed")
    router_package, router_report = validate_router_package(
        args.sealed_dir / "router_prefix_only.pt",
        trajectories, dense_windows, metadata, normalizer)

    registry = rde.EndpointRegistry()
    arms: dict[str, dict[str, object]] = {}
    router_states = {
        arm: {} for arm in ("r1_fixed_average", "r2_history_router")}
    for arm in rde.ARM_ORDER:
        arm_rows = {}
        for kind in rde.ENDPOINT_ORDER:
            seal, row, router = endpoint_seal(
                args.sealed_dir, arm, kind, SEALED_ARTIFACTS[arm][kind],
                schedule["sha256"], normalizer_report, base_config)
            registry.seal(arm, kind, seal)
            arm_rows[kind] = row
            if router is not None:
                router_states[arm][kind] = router
        if arm_rows["raw"]["model_config"] != arm_rows["ema"]["model_config"]:
            raise RuntimeError(f"raw/EMA config differs for {arm}")
        start = arm_rows["raw"]["start"]
        if start != arm_rows["ema"]["start"]:
            raise RuntimeError(f"raw/EMA start provenance differs for {arm}")
        arms[arm] = {
            "mode": rde.ARM_MODE[arm],
            "model_config": arm_rows["raw"]["model_config"],
            "parameters": rde.expected_parameters(arm),
            "parameter_manifest_sha256": start["parameter_manifest_sha256"],
            "initial_state_sha256": start["upgraded_model_state_sha256"],
            "router_state_sha256_before_real": start["router_state_sha256"],
            "router_state_sha256_after_real_raw": (
                None if arm == "r0_single"
                else rde.dr1.model_state_sha256(OrderedDict(
                    [(f"router.{name.removeprefix('regime_expert_router.')}", value)
                     for name, value in router_states[arm]["raw"].items()
                     if name.startswith("regime_expert_router.")]
                    + [("descriptor_mean", router_states[arm]["raw"][
                        "regime_expert_router_mean"]),
                       ("descriptor_std", router_states[arm]["raw"][
                        "regime_expert_router_std"])]))),
            "router_requires_grad_during_real": (
                None if arm == "r0_single" else False),
            "sample_stream_sha256": schedule["sha256"],
            "augmentation": dict(rde.dense4k.AUGMENTATION),
            "training": {
                "recovered_from_endpoint_evidence": True,
                "neural_training_reexecuted": False,
                "updates": rde.UPDATES,
                "batch_size": rde.small4k.BATCH_SIZE,
                "seed": rde.small4k.SEED,
                "optimizer": "AdamW",
                "lr": rde.small4k.LR,
                "weight_decay": rde.small4k.WEIGHT_DECAY,
                "grad_clip": rde.small4k.GRAD_CLIP,
                "scheduler_t_max": rde.SCHEDULER_T_MAX,
                "ema_decay": rde.EMA_DECAY,
                "planned_and_endpoint_stream_sha256": schedule["sha256"],
            },
            "seals": {kind: arm_rows[kind]["seal"]
                      for kind in rde.ENDPOINT_ORDER},
        }
    if not registry.complete():
        raise RuntimeError("six-endpoint recovery registry is incomplete")
    if (arms["r1_fixed_average"]["initial_state_sha256"]
            != arms["r2_history_router"]["initial_state_sha256"]
            or arms["r1_fixed_average"]["parameter_manifest_sha256"]
            != arms["r2_history_router"]["parameter_manifest_sha256"]):
        raise RuntimeError("R1/R2 initial-state or parameter manifest changed")
    router_integrity = rde.audit_frozen_router_endpoints(
        router_package, router_states)
    tail_audit = rde.TailAudit(registry)
    return {
        "git": git,
        "failed_log": failed_log,
        "start_report": start_report,
        "base_config": base_config,
        "manifest": manifest,
        "rows": rows,
        "prefix_opened": prefix_audit.opened,
        "statistics": statistics,
        "normalizer": normalizer,
        "normalizer_report": normalizer_report,
        "schedule": schedule,
        "router_package": router_package,
        "router_report": router_report,
        "router_integrity": router_integrity,
        "registry": registry,
        "tail_audit": tail_audit,
        "arms": arms,
    }


def recover(args: argparse.Namespace) -> dict[str, object]:
    if (not args.execute
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise PermissionError("sealed recovery requires protocol acknowledgement")
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    context = validate_pre_tail(args)
    audit_summary = {
        "schema": SCHEMA,
        "status": "PASS_SIX_SEALED_ENDPOINT_PREFLIGHT",
        "neural_training_reexecuted": False,
        "six_endpoint_files_sha256_validated": True,
        "six_endpoint_payload_and_state_sha256_validated": True,
        "start_checkpoint_and_payload_validated": True,
        "sample_stream_recomputed_and_validated": True,
        "normalizer_recomputed_and_validated": True,
        "router_endpoint_integrity": context["router_integrity"],
        "clean716_manifest_read_or_hashed": False,
        "clean716_target_accessed": False,
    }
    if args.audit_only:
        print(json.dumps(audit_summary, sort_keys=True))
        return audit_summary

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA recovery but CUDA is unavailable")
    deadline = (math.inf if args.wall_budget_seconds is None
                else started + args.wall_budget_seconds)
    # This is the first evaluation-manifest read in the recovery process.  The
    # six files, payloads, model-state hashes, sample stream, router, and an
    # immutable stat snapshot have already passed above.
    evaluation_sha = rde.dr1.sha256(args.evaluation_manifest)
    if evaluation_sha != rde.EXPECTED_EVALUATION_MANIFEST_SHA256:
        raise RuntimeError("fixed clean716 evaluation manifest SHA changed")
    evaluation_rows = rde.dense4k.load_evaluation_rows(
        args.evaluation_manifest, context["rows"])
    tail_trajectories, tail_metadata, tail_windows = rde.dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", context["tail_audit"])
    context["tail_audit"].after_load()
    tail_dataset = rde.ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)
    indices_by_aoa = rde.dense4k.group_window_indices_by_aoa(
        tail_windows, tail_metadata)
    normalizer = rde.smu1.FoldNormalizer(context["statistics"], device)
    target_audit = None
    for arm in rde.ARM_ORDER:
        context["arms"][arm]["endpoints"] = {}
        for kind in rde.ENDPOINT_ORDER:
            seal = context["registry"].values[arm][kind]
            model = rde.load_endpoint(
                seal, arm, normalizer, device, context["base_config"])
            metrics, metric_state, current_target = rde.smu1.evaluate_model(
                model, tail_dataset, list(range(len(tail_windows))),
                normalizer, device, deadline)
            if target_audit is None:
                target_audit = current_target
            elif target_audit != current_target:
                raise RuntimeError("clean716 target audit changed")
            by_aoa = {}
            for aoa, indices in indices_by_aoa.items():
                values, _state, _target = rde.smu1.evaluate_model(
                    model, tail_dataset, indices, normalizer, device, deadline)
                by_aoa[str(aoa)] = values
            context["arms"][arm]["endpoints"][kind] = {
                "metrics": metrics,
                "metric_state": metric_state,
                "metrics_by_aoa": by_aoa,
                "router": rde.held_router_report(
                    model, tail_trajectories, tail_windows,
                    tail_metadata, normalizer),
                "router_state_sha256": (
                    None if arm == "r0_single" else rde.router_state_sha256(model)),
                "expert_distance": rde.expert_distance(model),
                "seal": seal,
            }
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    comparisons = {}
    for kind in rde.ENDPOINT_ORDER:
        comparisons[kind] = {
            "r1_minus_r0": rde.metric_comparison(
                context["arms"]["r1_fixed_average"]["endpoints"][kind]["metrics"],
                context["arms"]["r0_single"]["endpoints"][kind]["metrics"]),
            "r2_minus_r0": rde.metric_comparison(
                context["arms"]["r2_history_router"]["endpoints"][kind]["metrics"],
                context["arms"]["r0_single"]["endpoints"][kind]["metrics"]),
            "r2_minus_r1": rde.metric_comparison(
                context["arms"]["r2_history_router"]["endpoints"][kind]["metrics"],
                context["arms"]["r1_fixed_average"]["endpoints"][kind]["metrics"]),
        }
    result = {
        "schema": rde.SCHEMA,
        "experiment": rde.EXPERIMENT,
        "git": context["git"],
        "recovery": {
            "schema": SCHEMA,
            "original_run_commit": ORIGINAL_RUN_COMMIT,
            "failed_log": context["failed_log"],
            "reason": (
                "post-evaluation assertion treated deterministic FP32 EMA "
                "roundoff of frozen named router parameters as training"),
            "neural_training_reexecuted": False,
            "evaluation_reexecuted_from_exact_six_seals": True,
            "preflight": audit_summary,
        },
        "start": {
            "checkpoint": str(args.start_sim.resolve()),
            "checkpoint_sha256": rde.EXPECTED_START_SHA256,
            "released_data_only_standard_sim": True,
        },
        "manifest": context["manifest"],
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": evaluation_sha,
            "windows": len(tail_windows),
        },
        "normalizer": {
            "kind": "sparse2544 clean81 prefix only",
            "statistics_sha256": rde.smu1.statistics_digest(
                context["statistics"]),
        },
        "router": context["router_report"],
        "router_endpoint_integrity": context["router_integrity"],
        "parameters": {
            "r0_single": rde.EXPECTED_BASE_PARAMETERS,
            "r1_r2_each": rde.EXPECTED_EXPERT_PARAMETERS,
            "duplicated_audited_scope": rde.EXPECTED_AUDITED_SCOPE_PARAMETERS,
            "frozen_router": rde.EXPECTED_ROUTER_PARAMETERS,
        },
        "training": {
            "recovered_not_reexecuted": True,
            "windows": rde.EXPECTED_DENSE_WINDOWS,
            "window_stride": 5,
            "sample_stream_sha256_all_arms": context["schedule"]["sha256"],
            "augmentation_all_arms": dict(rde.dense4k.AUGMENTATION),
            "updates": rde.UPDATES,
            "scheduler_t_max": rde.SCHEDULER_T_MAX,
            "ema_decay": rde.EMA_DECAY,
        },
        "expert_initial_state_sha256_r1_r2": context["arms"][
            "r1_fixed_average"]["initial_state_sha256"],
        "arms": context["arms"],
        "comparisons": comparisons,
        "access_audit": {
            "prefix_opened_during_recovery_preflight": context["prefix_opened"],
            "router_targets_used": False,
            "all_six_file_and_payload_seals_validated_before_evaluation_manifest_read": True,
            "clean716_recovery_opened_after_all_six_seal_validations": context[
                "tail_audit"].opened,
            "clean716_target_read_before_recovery_seal_validation": False,
            "original_endpoint_payloads_report_tail_open_before_seal": False,
        },
        "decision": "RDE_MOE_REAL4K_COMPLETE_NO_SCORE_GATE_SEALED_RECOVERY",
        "score_gate": None,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
    }
    rde.dr1.write_json_atomic(args.sealed_dir / "result.json", result)
    print(json.dumps({
        "event": "b210_rde_moe_sealed_recovery_complete",
        "comparisons": comparisons,
        "result": str(args.sealed_dir / "result.json"),
    }, sort_keys=True), flush=True)
    return result


def main() -> None:
    recover(parse_args())


if __name__ == "__main__":
    main()
