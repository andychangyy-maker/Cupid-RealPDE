#!/usr/bin/env python3
"""One sealed B210 held-AoA discovery fold with fit-AoA risk weighting.

This is deliberately a narrow discovery runner: it keeps the condition-held
fit-only/held-after-seal protocol and the existing hierarchical batch stream,
but partitions each eight-sample batch into four two-sample AoA subbatches.
The four complete existing Real objectives accumulate before one optimizer
step.  Detached group EMA losses only set the *next* batch's weights.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_condition_held_real as condition  # noqa: E402
import train_b210_np_functional_real_ft12_loao as ft12  # noqa: E402
import train_b210_np_functional_real_small4k_confirm_held0 as small4k  # noqa: E402
import train_b210_np_functional_sim as sim_tke  # noqa: E402
import train_b210_scratch_real_dense_aug as dense4k  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402


EXPERIMENT = "b210_aoa_risk_held_fold"
SCHEMA = "b210_aoa_risk_held_fold_v1"
ENDPOINT = "ema_u4000_aoa_risk"
UPDATES = 4_000
EMA_BETA = 0.95
AOA_VALUES = condition.AOA_VALUES
DISCOVERY_GATE = {
    "point": 0.01,
    "non_time": 0.03,
    "component": -0.05,
}


class Seal:
    def __init__(self) -> None:
        self.value: dict | None = None

    def seal(self, value: Mapping[str, object]) -> None:
        if self.value is not None:
            raise RuntimeError("AoA-risk endpoint sealed twice")
        self.value = dict(value)


@contextmanager
def endpoint_contract(held_aoa: int):
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": "b210_aoa_risk_endpoint_v1",
        "PRIMARY_ENDPOINT": ENDPOINT,
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": 12_000,
        "HELD_AOA": held_aoa,
    }
    original = {key: getattr(small4k, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(small4k, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(small4k, key, value)


def update_group_weights(ema_loss: torch.Tensor, raw_loss: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Use only detached native supervised losses for the next batch q."""
    if ema_loss.shape != raw_loss.shape or ema_loss.ndim != 1:
        raise ValueError("AoA-risk losses must be matching vectors")
    if (not torch.isfinite(ema_loss).all() or not torch.isfinite(raw_loss).all()
            or bool((raw_loss <= 0.0).any()) or bool((ema_loss < 0.0).any())):
        raise FloatingPointError("AoA-risk losses are invalid")
    # All four groups are observed in every hierarchical batch.  An all-zero
    # state therefore unambiguously means "first observation" and lets the
    # risk scale be set by data rather than an arbitrary initial loss of one.
    ema = (raw_loss if bool((ema_loss == 0.0).all()) else
           EMA_BETA * ema_loss + (1.0 - EMA_BETA) * raw_loss)
    q = ema / ema.sum().clamp_min(torch.finfo(ema.dtype).tiny)
    if not torch.isfinite(q).all() or not torch.allclose(
            q.sum(), q.new_tensor(1.0), rtol=0.0, atol=1e-6):
        raise FloatingPointError("AoA-risk weights are invalid")
    return ema, q


def partition_batch(trajectory_ids: torch.Tensor, metadata: list[Mapping[str, object]],
                    fit_aoas: tuple[int, ...]) -> dict[int, torch.Tensor]:
    groups = {aoa: [] for aoa in fit_aoas}
    for index, trajectory in enumerate(trajectory_ids.tolist()):
        aoa = int(metadata[int(trajectory)]["aoa"])
        if aoa not in groups:
            raise RuntimeError("batch contains a held/non-fit AoA")
        groups[aoa].append(index)
    result = {
        aoa: torch.tensor(indices, dtype=torch.long)
        for aoa, indices in groups.items()
    }
    if any(len(indices) != 2 for indices in result.values()):
        raise RuntimeError("AoA-risk hierarchical batch is not exactly 2 per fit AoA")
    return result


def load_balanced_metrics(pair: list[str] | None, held_aoa: int) -> dict | None:
    if pair is None:
        return None
    path = Path(pair[0])
    if dr1.sha256(path) != dr1.normalized_sha256(pair[1], "balanced result SHA"):
        raise RuntimeError("balanced result SHA changed")
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("schema") != condition.BALANCED_SCHEMA
            or int(row.get("held_aoa", -1)) != held_aoa):
        raise ValueError("balanced result is not the requested held fold")
    return dict(row["arm"]["metrics"])


def load_risk_endpoint(seal: Mapping[str, object], normalizer,
                       device: torch.device, held_aoa: int):
    """Strict-load the private endpoint without pretending it is Dense4k."""
    path = Path(str(seal["path"]))
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("AoA-risk endpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != "b210_aoa_risk_endpoint_v1"
            or payload.get("experiment") != EXPERIMENT
            or payload.get("endpoint") != ENDPOINT
            or int(payload.get("held_aoa", -1)) != held_aoa
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("AoA-risk endpoint payload changed")
    config = ft12.validate_b210_model_config(
        payload["model_config"], sim_tke.b210_config("candidate"))
    model = small4k.EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def discovery_comparison(candidate: Mapping[str, float], control: Mapping[str, float]) -> dict:
    comparison = condition.metric_comparison(candidate, control)
    delta = comparison["delta"]
    checks = {
        "point": comparison["point_final_contribution_delta"] >= DISCOVERY_GATE["point"],
        "non_time": comparison["non_time_final_contribution_delta"] >= DISCOVERY_GATE["non_time"],
        "components": min(delta[key] for key in (
            "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")) >= DISCOVERY_GATE["component"],
        "native_sps": delta["native_sps_score"] >= DISCOVERY_GATE["component"],
    }
    return {**comparison, "gate": dict(DISCOVERY_GATE), "checks": checks,
            "decision": "PASS_AOA_RISK_DISCOVERY" if all(checks.values()) else "STOP_AOA_RISK_DISCOVERY"}


def train_risk_arm(model, dataset, schedule: Mapping[str, object], normalizer,
                   device: torch.device, deadline: float, output_dir: Path,
                   start_report: Mapping[str, object], held_aoa: int,
                   metadata: list[Mapping[str, object]]) -> tuple[dict, dict, list[dict]]:
    trainable = ft12.configure_trainable(model, ft12.EXPECTED_PARAMETERS)
    loader = DataLoader(ft12.npt1.ensemble_train.TrajectoryTaggedDataset(dataset),
                        batch_sampler=schedule["batches"], num_workers=0,
                        pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(trainable, lr=small4k.LR,
                                  weight_decay=small4k.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=12_000, eta_min=0.0)
    ema = small4k.ema_lib.FP32EMAShadow(model, small4k.EMA_DECAY)
    fit_aoas = tuple(aoa for aoa in AOA_VALUES if aoa != held_aoa)
    ema_loss = torch.zeros(len(fit_aoas), device=device)
    q = torch.full_like(ema_loss, 1.0 / len(fit_aoas))
    rows, risk_history = [], []
    started = time.perf_counter()
    for iteration, batch in enumerate(loader, start=1):
        if iteration > UPDATES:
            break
        if time.perf_counter() > deadline:
            raise TimeoutError("AoA-risk held fold exceeded wall budget")
        history, target, trajectory_ids, sample_indices = batch
        expected = schedule["rows"][iteration - 1]
        actual = {"sample_indices": sample_indices.long().tolist(),
                  "trajectory_ids": trajectory_ids.long().tolist(),
                  "bootstrap_mask": expected["bootstrap_mask"]}
        if actual != expected:
            raise RuntimeError("AoA-risk sample/bootstrap stream changed")
        rows.append(actual)
        partitions = partition_batch(trajectory_ids, metadata, fit_aoas)
        member_mask = torch.as_tensor(expected["bootstrap_mask"], dtype=torch.bool)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw = []
        for group_index, aoa in enumerate(fit_aoas):
            indices = partitions[aoa]
            subbatch = tuple(value.index_select(0, indices) for value in (
                history, target, trajectory_ids, sample_indices))
            values = ft12.npt1.backward_real_supervised_npt1(
                model, subbatch, normalizer, device, iteration,
                auxiliary_weight=ft12.tke_aux_weight(iteration),
                member_mask=member_mask.index_select(0, indices),
                audit_controller_gradients=False,
                input_augmentation=dense4k.AUGMENTATION,
                loss_scale=float(q[group_index]))
            raw.append(float(values["real_native_supervised_loss"]))
        raw_tensor = torch.tensor(raw, device=device)
        ema_loss, next_q = update_group_weights(ema_loss, raw_tensor)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, small4k.GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("AoA-risk clipped gradient is non-finite")
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=iteration)
        scheduler.step()
        risk_history.append({"iteration": iteration, "q_used": {
            str(aoa): float(q[index]) for index, aoa in enumerate(fit_aoas)},
            "ema_native_supervised_loss": {str(aoa): float(ema_loss[index])
                for index, aoa in enumerate(fit_aoas)},
            "raw_native_supervised_loss": {str(aoa): raw[index]
                for index, aoa in enumerate(fit_aoas)},
            "sample_count": {str(aoa): 2 for aoa in fit_aoas},
            "gradient_norm_before_clip": float(gradient_norm)})
        q = next_q
        if iteration == 1 or iteration % 500 == 0:
            print(json.dumps({"event": "b210_aoa_risk_progress", **risk_history[-1],
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    if len(rows) != UPDATES or ema.updates != UPDATES:
        raise RuntimeError("AoA-risk arm did not finish exactly 4000 updates")
    if dr1.canonical_sha256(rows) != schedule["sha256"]:
        raise RuntimeError("AoA-risk stream digest changed")
    seal_access = Seal()
    seal, ema_report = small4k.save_primary_endpoint(
        output_dir / "ema_u4000_aoa_risk.pth", model, ema,
        start_report=start_report, normalizer_report=ft12.normalizer_payload(normalizer),
        stream_sha256=schedule["sha256"])
    seal_access.seal(seal)
    return ({"updates": UPDATES, "ema_decay": small4k.EMA_DECAY,
             "scheduler_t_max": 12_000, "sample_stream_sha256": schedule["sha256"],
             "sampling": schedule["sampling"], "risk_ema_beta": EMA_BETA,
             "risk_history": risk_history, "ema": ema_report}, seal_access.value, rows)


def run(args: argparse.Namespace) -> dict:
    if args.held_aoa not in AOA_VALUES or args.output_dir.exists():
        raise ValueError("held AoA invalid or output already exists")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("AoA-risk discovery requires CUDA")
    git = dr1.git_identity()
    if git["status_porcelain"]:
        raise RuntimeError("AoA-risk discovery requires a clean worktree")
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    _control_row, dense_control = condition.load_frozen_control(
        args.dense_control_result, args.held_aoa)
    balanced_metrics = load_balanced_metrics(args.balanced_result, args.held_aoa)
    rows, manifest = dr1.load_manifest(args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    args.output_dir.mkdir(parents=True)
    deadline = time.perf_counter() + args.wall_budget_seconds
    audit = condition.PrefixAccessAudit(args.held_aoa, ["aoa_risk"])
    trajectories, metadata, sparse = dr1.load_prefix_rows(args.real_dir, fit_rows, "fit", audit)
    windows = dense4k.dense_windows(trajectories, dense4k.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    stats = smu1.fold_statistics(smu1.window_sufficient_statistics(trajectories, sparse), range(len(sparse)))
    normalizer = smu1.FoldNormalizer(stats, torch.device(args.device))
    if smu1.statistics_digest(stats) != dense_control["normalizer"]["statistics_sha256"]:
        raise RuntimeError("AoA-risk normalizer differs from frozen dense control")
    schedule = condition.hierarchical_batch_schedule(windows, metadata, UPDATES, small4k.SEED)
    start_sha = dr1.sha256(args.start_sim)
    if start_sha != dense_control["start"]["sha256"]:
        raise RuntimeError("AoA-risk Sim start differs from frozen dense control")
    ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, torch.device(args.device),
        expected_config=sim_tke.b210_config("candidate"),
        expected_parameters=ft12.EXPECTED_PARAMETERS,
        expected_controller_parameters=ft12.EXPECTED_CONTROLLER_PARAMETERS)
    with endpoint_contract(args.held_aoa):
        training, seal, _stream = train_risk_arm(model, dataset, schedule, normalizer,
            torch.device(args.device), deadline, args.output_dir, start_report,
            args.held_aoa, metadata)
    audit.seal("aoa_risk", seal)
    del model, dataset, trajectories
    gc.collect(); torch.cuda.empty_cache()
    held_trajectories, _held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", audit)
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(held_trajectories, held_windows)
    model = load_risk_endpoint(seal, normalizer, torch.device(args.device), args.held_aoa)
    metrics, metric_state, target_audit = smu1.evaluate_model(
        model, held_dataset, list(range(len(held_windows))), normalizer,
        torch.device(args.device), deadline)
    result = {"schema": SCHEMA, "experiment": EXPERIMENT, "held_aoa": args.held_aoa,
              "command": list(sys.argv), "git": git,
              "manifest": manifest, "control_dense4k": dense_control,
              "control_sources": {
                  "dense4k": {"path": str(Path(args.dense_control_result[0]).resolve()),
                               "sha256": args.dense_control_result[1].lower()},
                  "balanced4k": (None if args.balanced_result is None else
                      {"path": str(Path(args.balanced_result[0]).resolve()),
                       "sha256": args.balanced_result[1].lower()}),
              },
              "control_balanced4k_metrics": balanced_metrics, "training": training,
              "seal": seal, "metrics": metrics, "metric_state": metric_state,
              "target_audit": target_audit, "access_audit": audit.report(),
              "dense4k_comparison": discovery_comparison(metrics, dense_control["metrics"]),
              "balanced4k_comparison": (None if balanced_metrics is None else discovery_comparison(metrics, balanced_metrics)),
              "elapsed_seconds": time.perf_counter() - started,
              "submission_authorized": False}
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-aoa", type=int, required=True)
    parser.add_argument("--dense-control-result", nargs=2, required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--balanced-result", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"event": "b210_aoa_risk_held_fold_complete",
                      "decision": result["dense4k_comparison"]["decision"],
                      "output": str((args.output_dir / "result.json").resolve())}), flush=True)


if __name__ == "__main__":
    main()
