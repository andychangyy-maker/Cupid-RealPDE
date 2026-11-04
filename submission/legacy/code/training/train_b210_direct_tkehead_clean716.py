#!/usr/bin/env python3
"""Train one frozen-backbone direct-TKE head and gate it on clean716."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for value in (CODE_ROOT, TRAINING_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_np_functional_real_ft12_loao as ft12  # noqa: E402
import train_b210_scratch_real_dense_aug as scratch_real  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from models.direct_residual_tke_head import (  # noqa: E402
    DirectResidualTKEHead,
    FrozenB210DirectTKE,
)
from train_flow_backbone import construct_model  # noqa: E402


EXPERIMENT = "b210_direct_tkehead_clean716"
SCHEMA = "b210_direct_tkehead_clean716_v1"
ACKNOWLEDGEMENT = "B210_DIRECT_TKEHEAD_CLEAN716"
SOURCE_SHA256 = (
    "9012d1f163894e8f2f993e69a64426b6972fa608b434578cf109b06d6865273f")
NORMALIZER_SHA256 = (
    "86b5feec17db74bca9dc909ed5baae93e4775c83f62db98bd46032a48f5bfc7f")
PREFIX_MANIFEST_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
TAIL_MANIFEST_SHA256 = (
    "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")

UPDATES = 1_500
BATCH_SIZE = 8
SEED = 4
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
SOFT_SPS_WEIGHT = 0.05
SOFT_SPS_TAU = 0.10
PREFLIGHT_UPDATES = 200
GRADIENT_RATIO_GATE = (0.02, 0.20)
FIT_TKE_RELATIVE_IMPROVEMENT = 0.05
HEAD_PARAMETERS = 7_201


class PrefixAudit:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def before_open(self, row: dict, phase: str) -> None:
        if (phase != "fit" or row["file"] == dr1.EXCLUDED_BASENAME
                or row["file"] in self.opened):
            raise RuntimeError("invalid TKE-head prefix open")

    def record_open(self, row: dict, phase: str) -> None:
        if phase != "fit" or row["file"] in self.opened:
            raise RuntimeError("invalid TKE-head prefix audit")
        self.opened.append(str(row["file"]))


class TailAudit:
    def __init__(self, source: Path, source_sha: str,
                 endpoint: Path, endpoint_sha: str) -> None:
        self.sources = ((source, source_sha), (endpoint, endpoint_sha))
        self.opened: list[str] = []

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "held" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid TKE-head tail open")
        for path, expected in self.sources:
            if not path.is_file() or dr1.sha256(path) != expected:
                raise RuntimeError("sealed TKE-head source changed")

    def record_open(self, row: dict, phase: str) -> None:
        if phase != "held" or row["file"] in self.opened:
            raise RuntimeError("invalid TKE-head tail audit")
        self.opened.append(str(row["file"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600.0)
    parser.add_argument("--acknowledge-protocol", required=True)
    return parser.parse_args()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or dr1.sha256(path) != expected:
        raise RuntimeError(f"{label} identity changed: {path}")


def validate_args(args: argparse.Namespace) -> None:
    if args.acknowledge_protocol != ACKNOWLEDGEMENT:
        raise PermissionError("direct TKE-head protocol acknowledgement missing")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("direct TKE-head run requires CUDA")
    require_sha(args.source_checkpoint, SOURCE_SHA256, "source checkpoint")
    require_sha(args.normalizer, NORMALIZER_SHA256, "normalizer")
    require_sha(args.clean_manifest, PREFIX_MANIFEST_SHA256, "prefix manifest")
    require_sha(
        args.evaluation_manifest, TAIL_MANIFEST_SHA256, "tail manifest")


def load_base(path: Path, normalizer, device: torch.device):
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    model = construct_model(
        payload["model_config"]["architecture"], payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    common.set_model_stats(model, normalizer)
    model = model.to(device).eval()
    if not model.config.physical_energy_member_sync:
        raise RuntimeError("source B210 lacks member-synchronous energy")
    return model, payload


def stack_batch(dataset, indices: Sequence[int]):
    rows = [dataset[int(index)] for index in indices]
    return (torch.stack([row[0] for row in rows]),
            torch.stack([row[1] for row in rows]))


def objectives(model, batch, normalizer, device):
    history, target_physical = batch
    history = history.to(device, non_blocking=True)
    target_physical = target_physical.to(device, non_blocking=True)
    history_norm, target_norm = normalizer.preprocess(
        history, target_physical)
    prediction, _members, _quantiles, lower, upper = (
        model.forward_uq_training(history_norm)[:5])
    tke = ensemble_train.physical_tke_map_distillation_loss(
        prediction, target_physical[..., :2], normalizer)
    soft_sps, score = ensemble_train.soft_native_sps_objective(
        prediction, target_norm, lower, upper, normalizer, SOFT_SPS_TAU,
        physical_target=target_physical[..., :2])
    return tke + SOFT_SPS_WEIGHT * soft_sps, tke, soft_sps, score


def gradient_norm(loss, parameters, retain_graph: bool) -> float:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True)
    squares = [value.float().square().sum() for value in gradients
               if value is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


@torch.inference_mode()
def invariant_report(base, candidate, history_norm) -> dict:
    baseline = base.forward_uq_training(history_norm)
    values = candidate.forward_uq_training(history_norm)
    point_mean = float((
        values[0][..., :2].mean(dim=1)
        - baseline[0][..., :2].mean(dim=1)).abs().max())
    member_mean = float((
        values[1][..., :2].mean(dim=2)
        - baseline[1][..., :2].mean(dim=2)).abs().max())
    pressure = bool(
        torch.equal(values[0][..., 2:], baseline[0][..., 2:])
        and torch.equal(values[1][..., 2:], baseline[1][..., 2:]))
    exact = {
        name: bool(torch.equal(baseline[index], values[index]))
        for index, name in enumerate(
            ("point", "members", "quantiles", "lower", "upper"))
    }
    return {
        "exact_outputs": exact,
        "exact_identity": all(exact.values()),
        "point_temporal_mean_max_abs_delta": point_mean,
        "member_temporal_mean_max_abs_delta": member_mean,
        "temporal_mean_pass": max(point_mean, member_mean) <= 1e-6,
        "pressure_bitwise": pressure,
    }


def train_head(model, dataset, windows, normalizer, device, deadline,
               output_dir: Path, source_path: Path) -> tuple[Path, dict]:
    parameters = tuple(model.head.parameters())
    if (sum(value.numel() for value in parameters) != HEAD_PARAMETERS
            or any(value.requires_grad for value in model.base.parameters())):
        raise RuntimeError("direct TKE-head trainable scope changed")
    schedule = smu1.paired_batch_schedule(
        range(len(windows)), windows, updates=UPDATES, seed=SEED)
    optimizer = torch.optim.AdamW(
        parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES)
    ema = ema_lib.FP32EMAShadow(model.head, EMA_DECAY)
    fixed = stack_batch(dataset, schedule["batches"][0])
    fixed_history = normalizer.preprocess(
        fixed[0].to(device), fixed[1].to(device))[0]
    update0 = invariant_report(model.base, model, fixed_history)
    if not (update0["exact_identity"] and update0["temporal_mean_pass"]
            and update0["pressure_bitwise"]):
        raise RuntimeError("direct TKE-head update0 identity failed")
    with torch.no_grad():
        initial_tke = float(objectives(
            model, fixed, normalizer, device)[1])
    ratios: list[float] = []
    history: list[dict] = []
    started = time.perf_counter()
    preflight = None
    for iteration, indices in enumerate(schedule["batches"], start=1):
        if time.perf_counter() > deadline:
            raise TimeoutError("direct TKE-head training exceeded wall budget")
        batch = stack_batch(dataset, indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, tke, soft_sps, score = objectives(
            model, batch, normalizer, device)
        if not all(bool(torch.isfinite(value)) for value in
                   (total, tke, soft_sps, score)):
            raise FloatingPointError("direct TKE-head loss is non-finite")
        if iteration <= PREFLIGHT_UPDATES:
            tke_norm = gradient_norm(tke, parameters, retain_graph=True)
            sps_norm = gradient_norm(
                SOFT_SPS_WEIGHT * soft_sps, parameters, retain_graph=True)
            ratio = sps_norm / max(tke_norm, 1e-20)
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise RuntimeError("direct TKE-head gradient ratio invalid")
            ratios.append(ratio)
        total.backward()
        if any(value.grad is not None for value in model.base.parameters()):
            raise RuntimeError("frozen B210 received a gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("direct TKE-head gradient is non-finite")
        optimizer.step()
        ema.update_after_optimizer_step(model.head, optimizer_step=iteration)
        scheduler.step()
        if iteration == PREFLIGHT_UPDATES:
            median = float(statistics.median(ratios))
            with torch.no_grad():
                current_tke = float(objectives(
                    model, fixed, normalizer, device)[1])
            relative = (initial_tke - current_tke) / initial_tke
            preflight = {
                "gradient_ratio_median": median,
                "gradient_ratio_gate": list(GRADIENT_RATIO_GATE),
                "fixed_batch_initial_tke": initial_tke,
                "fixed_batch_u200_tke": current_tke,
                "fixed_batch_relative_improvement": relative,
                "pass": (GRADIENT_RATIO_GATE[0] <= median
                         <= GRADIENT_RATIO_GATE[1]
                         and relative >= FIT_TKE_RELATIVE_IMPROVEMENT),
            }
            if not preflight["pass"]:
                raise RuntimeError(f"direct TKE-head u200 gate failed: {preflight}")
        if iteration == 1 or iteration % 200 == 0:
            row = {
                "iteration": iteration,
                "tke_loss": float(tke.detach()),
                "soft_sps_loss": float(soft_sps.detach()),
                "soft_sps_score": float(score),
                "gradient_norm": float(norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps({"event": "tkehead_progress", **row}), flush=True)
    if ema.updates != UPDATES or preflight is None:
        raise RuntimeError("direct TKE-head did not complete fixed terminal")
    terminal = invariant_report(model.base, model, fixed_history)
    if not terminal["temporal_mean_pass"] or not terminal["pressure_bitwise"]:
        raise RuntimeError("direct TKE-head terminal invariant failed")
    seal = ema.seal(model.head)
    state = seal.pop("model_state")
    endpoint = output_dir / "ema999_tkehead_u1500.pth"
    payload = {
        "schema": "b210_direct_tkehead_endpoint_v1",
        "experiment": EXPERIMENT,
        "endpoint": "ema999_u1500",
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "head_config": model.head.architecture_config(),
        "head_state_dict": OrderedDict(
            (name, value.detach().cpu()) for name, value in state.items()),
        "head_state_sha256": ema_lib.tensor_state_sha256(state),
        "sample_stream_sha256": schedule["sha256"],
        "ema": seal,
    }
    torch.save(payload, endpoint)
    report = {
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "soft_sps_weight": SOFT_SPS_WEIGHT,
        "soft_sps_tau": SOFT_SPS_TAU,
        "head_parameters": HEAD_PARAMETERS,
        "sample_stream_sha256": schedule["sha256"],
        "update0": update0,
        "preflight": preflight,
        "terminal_invariants": terminal,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return endpoint, report


def comparison(candidate: Mapping[str, float],
               baseline: Mapping[str, float]) -> dict:
    keys = ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
            "point_score", "native_sps_score")
    delta = {key: float(candidate[key]) - float(baseline[key]) for key in keys}
    non_time = (0.5 * delta["rel_l2_score"]
                + 0.1 * delta["tke_rel_l2_score"]
                + 0.1 * delta["mvpe_rel_l2_score"]
                + 0.2 * delta["native_sps_score"])
    return {"baseline": dict(baseline), "candidate": dict(candidate),
            "delta": delta, "non_time_proxy_delta": non_time}


def gate(pooled: Mapping[str, object], by_aoa: Mapping[str, Mapping]) -> dict:
    delta = pooled["delta"]
    positive_aoas = sum(
        float(value["non_time_proxy_delta"]) > 0.0
        for value in by_aoa.values())
    worst_tke = min(
        float(value["delta"]["tke_rel_l2_score"])
        for value in by_aoa.values())
    conditions = {
        "tke_at_least_plus_0p20": delta["tke_rel_l2_score"] >= 0.20,
        "sps_nonnegative": delta["native_sps_score"] >= 0.0,
        "rel_at_least_minus_0p05": delta["rel_l2_score"] >= -0.05,
        "mvpe_abs_at_most_0p01": abs(delta["mvpe_rel_l2_score"]) <= 0.01,
        "non_time_positive": pooled["non_time_proxy_delta"] > 0.0,
        "at_least_four_positive_aoas": positive_aoas >= 4,
        "worst_aoa_tke_at_least_minus_0p05": worst_tke >= -0.05,
    }
    passed = all(conditions.values())
    return {
        "pass": passed, "conditions": conditions,
        "positive_aoas": positive_aoas, "worst_aoa_tke_delta": worst_tke,
        "decision": ("PASS_B210_DIRECT_TKEHEAD_CLEAN716"
                     if passed else "STOP_B210_DIRECT_TKEHEAD_CLEAN716"),
    }


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")

    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = scratch_real.load_evaluation_rows(
        args.evaluation_manifest, rows)
    prefix_audit = PrefixAudit()
    trajectories, _metadata, _stride20 = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    train_windows = scratch_real.dense_windows(trajectories, stride=5)
    dataset = common.repro.WindowDataset(trajectories, train_windows)
    normalizer = common.repro.Normalizer(args.normalizer, device)
    base, source_payload = load_base(
        args.source_checkpoint, normalizer, device)
    base_state_before = dr1.model_state_sha256(base.state_dict())
    model = FrozenB210DirectTKE(base, DirectResidualTKEHead()).to(device)
    endpoint, training = train_head(
        model, dataset, train_windows, normalizer, device, deadline,
        args.output_dir, args.source_checkpoint)
    endpoint_sha = dr1.sha256(endpoint)
    if dr1.model_state_sha256(base.state_dict()) != base_state_before:
        raise RuntimeError("direct TKE-head mutated the frozen source B210")

    del dataset, trajectories
    gc.collect()
    tail_audit = TailAudit(
        args.source_checkpoint, SOURCE_SHA256, endpoint, endpoint_sha)
    tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", tail_audit)
    tail_dataset = common.repro.WindowDataset(tail_trajectories, tail_windows)
    groups = scratch_real.group_window_indices_by_aoa(
        tail_windows, tail_metadata)

    endpoint_payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    model.head.load_state_dict(endpoint_payload["head_state_dict"], strict=True)
    model.eval()
    baseline_metrics, _baseline_state, baseline_audit = smu1.evaluate_model(
        base, tail_dataset, list(range(len(tail_windows))),
        normalizer, device, deadline)
    candidate_metrics, _candidate_state, candidate_audit = smu1.evaluate_model(
        model, tail_dataset, list(range(len(tail_windows))),
        normalizer, device, deadline)
    if baseline_audit != candidate_audit:
        raise RuntimeError("baseline/candidate tail target audit differs")
    pooled = comparison(candidate_metrics, baseline_metrics)
    by_aoa = {}
    for aoa, indices in groups.items():
        baseline_aoa = smu1.evaluate_model(
            base, tail_dataset, indices, normalizer, device, deadline)[0]
        candidate_aoa = smu1.evaluate_model(
            model, tail_dataset, indices, normalizer, device, deadline)[0]
        by_aoa[str(aoa)] = comparison(candidate_aoa, baseline_aoa)
    decision = gate(pooled, by_aoa)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "source": {
            "path": str(args.source_checkpoint.resolve()),
            "sha256": SOURCE_SHA256,
            "model_config": source_payload["model_config"],
            "model_state_sha256": base_state_before,
        },
        "endpoint": {
            "path": str(endpoint.resolve()),
            "sha256": endpoint_sha,
            "head_state_sha256": endpoint_payload["head_state_sha256"],
        },
        "manifest": manifest_report,
        "training": training,
        "clean716": {
            "samples": len(tail_windows),
            "comparison": pooled,
            "by_aoa": by_aoa,
            "target_audit": baseline_audit,
            "gate": decision,
        },
        "access_audit": {
            "prefix_opened_before_endpoint": prefix_audit.opened,
            "tail_opened_after_endpoint_seal": tail_audit.opened,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
        "decision": decision["decision"],
    }
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    return result


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except BaseException as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dr1.write_json_atomic(args.output_dir / "STOP.json", {
            "schema": "b210_direct_tkehead_clean716_stop_v1",
            "experiment": EXPERIMENT,
            "error_type": type(error).__name__,
            "error": str(error),
            "submission_authorized": False,
        })
        raise
    print(json.dumps({
        "event": "b210_direct_tkehead_complete",
        "decision": result["decision"],
        "comparison": result["clean716"]["comparison"],
        "result": str(args.output_dir / "result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
