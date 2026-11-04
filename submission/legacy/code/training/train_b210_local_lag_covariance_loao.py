#!/usr/bin/env python3
"""Sealed Sim10k LOAO screen for a local-lag covariance auxiliary.

Only the point path is trainable.  Candidate and null control replay the
identical A-side stream; they differ solely in the history supplied to the
local-lag auxiliary.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
from itertools import permutations
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_domain_joint_loao as joint  # noqa: E402
import train_b210_fomldg_point_loao as m11  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402

EXPERIMENT = "b210_m13_local_lag_covariance_loao"
SCHEMA = "b210_m13_local_lag_covariance_loao_v1"
ENDPOINT_SCHEMA = "b210_m13_local_lag_covariance_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_M13_LOCAL_LAG_COVARIANCE_LOAO"
FORMAL_UPDATES = 1_920
FORMAL_BATCH_SIZE = 4
SMOKE_UPDATES = 24
SEED = 0
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
AUX_GRADIENT_FRACTION = 0.10
AUX_WARMUP_UPDATES = 12
AUX_CALIBRATION_PANEL_UPDATES = 12
AUX_CALIBRATION_QUANTILE = 0.90
FORMAL_SCHEDULE_SHA256 = (
    "17ee224e0d169d49273cb307c48ffc21f505586148670cbae1593458c7999082")
BASE_FORMAL_SCHEDULE_SHA256 = (
    "e69cf7cdc6ecaf5f80353b76e3221910bae6c01891c0d339db84c0d480e8dd4c")


def nonself_donor_positions(samples: Sequence[Mapping[str, object]]) -> list[int]:
    """Return the first deterministic permutation with no identity collision."""
    identities = [
        (str(sample["file"]), int(sample["start"]))
        for sample in samples
    ]
    for proposal in permutations(range(len(identities))):
        if all(identities[donor] != identities[recipient]
               for recipient, donor in enumerate(proposal)):
            return list(proposal)
    raise RuntimeError("batch has no donor permutation that breaks sample identity")


def build_m13_schedule(rows: Sequence[Mapping[str, object]], updates: int,
                       batch_size: int, seed: int) -> dict[str, object]:
    """Augment the frozen base stream with an audited nonself null mapping."""
    base = m11.build_cross_condition_schedule(rows, updates, batch_size, seed)
    augmented_rows = []
    for row in base["rows"]:
        copied = dict(row)
        copied["A_null_donor_positions"] = nonself_donor_positions(row["A"])
        augmented_rows.append(copied)
    donor_slots = sum(len(row["A_null_donor_positions"])
                      for row in augmented_rows)
    identity_collisions = sum(
        (str(row["A"][recipient]["file"]),
         int(row["A"][recipient]["start"]))
        == (str(row["A"][donor]["file"]),
            int(row["A"][donor]["start"]))
        for row in augmented_rows
        for recipient, donor in enumerate(row["A_null_donor_positions"])
    )
    if identity_collisions:
        raise RuntimeError("M13 null schedule retained a sample identity")
    schedule = dict(base)
    schedule.update({
        "schema": "b210_m13_local_lag_covariance_schedule_v1",
        "base_schedule_sha256": base["sha256"],
        "null_donor_audit": {
            "slots": donor_slots,
            "identity_collisions": identity_collisions,
            "all_donors_used_once_per_batch": True,
        },
        "rows": augmented_rows,
        "sha256": m11.canonical_sha256(augmented_rows),
    })
    return schedule


def arm_anchor(arm: str) -> str:
    if arm == "candidate":
        return "candidate"
    if arm == "control":
        return "null"
    raise ValueError(f"unknown arm {arm}")


def local_lag_loss(history: torch.Tensor, prediction: torch.Tensor,
                   target: torch.Tensor, arm: str,
                   donor_positions: Sequence[int]) -> torch.Tensor:
    """Late-bind the experimental loss so contract-only imports stay cheap."""
    module = importlib.import_module("local_lag_covariance")
    function = getattr(module, "local_lag_covariance_loss")
    anchor_mode = arm_anchor(arm)
    kwargs = ({"donor_positions": donor_positions}
              if arm == "control" else {})
    value = function(
        history, prediction, target, anchor=anchor_mode, **kwargs)
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise TypeError("local_lag_covariance_loss must return a scalar tensor")
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("non-finite local-lag covariance loss")
    return value


def prediction_gradient_norm(loss: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    gradient, = torch.autograd.grad(loss, prediction, retain_graph=True)
    return gradient.detach().float().square().sum().sqrt()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def nested_state_to_cpu(value):
    """Clone an optimizer state tree to CPU for exact branch replay."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: nested_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nested_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(nested_state_to_cpu(item) for item in value)
    return value


def nested_state_sha256(value) -> str:
    """Hash tensor/scalar containers without depending on torch serialization."""
    digest = hashlib.sha256()

    def update(item) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().contiguous().cpu().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            keys = sorted(
                item, key=lambda key: (type(key).__name__, repr(key)))
            for key in keys:
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(json.dumps(item, sort_keys=True).encode("utf-8"))
        else:
            raise TypeError(f"unsupported optimizer-state type {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _forward_main(view, history, target, trajectory_ids, member_mask,
                  normalizer, update: int, dropout_seed: int):
    with m11.rng_scope(dropout_seed, history.device):
        prediction, members = view(history)
    main, values = m11.point_member_objective(
        prediction, members, target, trajectory_ids, member_mask, normalizer, update)
    return prediction, main, values


def calibrate_auxiliary_weight(start: Mapping[str, object], dataset,
                               schedule: Mapping[str, object], normalizer,
                               device: torch.device,
                               ) -> tuple[dict[str, object], dict[str, object]]:
    """Calibrate once after a deterministic point-only domain warmup."""
    panel_end = AUX_WARMUP_UPDATES + AUX_CALIBRATION_PANEL_UPDATES
    if len(schedule["rows"]) < panel_end:
        raise ValueError("schedule is too short for M13 warmup calibration")
    ensemble_train.seed_everything(int(schedule["seed"]))
    model = joint.build_start_model(start, device)
    ensemble_train.common.set_model_stats(model, normalizer)
    trainable, _trainability = joint.configure_point_only_trainable(model)
    model.train()
    view = m11.PointTrainingView(model)
    initial_buffer = m11.buffer_state_sha256(model)
    uq_start = dr1.model_state_sha256(joint.uq_parameter_state(model))
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_losses = []
    for row in schedule["rows"][:AUX_WARMUP_UPDATES]:
        optimizer.zero_grad(set_to_none=True)
        history, target = m11.prepare_batch(
            dataset, row["A_indices"], normalizer, device, row["rng"]["A"])
        trajectory_ids = torch.as_tensor(
            [dataset.windows[index].trajectory for index in row["A_indices"]],
            dtype=torch.long, device=device)
        member_mask = torch.as_tensor(
            row["A_bootstrap_mask"], dtype=torch.bool, device=device)
        _prediction, main, values = _forward_main(
            view, history, target, trajectory_ids, member_mask, normalizer,
            int(row["update"]), int(row["rng"]["A"]["dropout"]))
        main.backward()
        clipped = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        if not bool(torch.isfinite(clipped)):
            raise FloatingPointError("non-finite M13 calibration warmup gradient")
        if any(parameter.grad is not None
               for _name, parameter in joint.frozen_named_parameters(model)):
            raise RuntimeError("calibration warmup reached a frozen UQ parameter")
        optimizer.step()
        warmup_losses.append(values["loss"])

    post_warmup_state = dr1.model_state_sha256(model.state_dict())
    post_warmup_optimizer_state = nested_state_sha256(
        optimizer.state_dict())
    uq_end = dr1.model_state_sha256(joint.uq_parameter_state(model))
    if m11.buffer_state_sha256(model) != initial_buffer or uq_end != uq_start:
        raise RuntimeError("calibration warmup changed buffers or frozen UQ state")
    optimizer.zero_grad(set_to_none=True)
    panel = []
    for row in schedule["rows"][AUX_WARMUP_UPDATES:panel_end]:
        history, target = m11.prepare_batch(
            dataset, row["A_indices"], normalizer, device, row["rng"]["A"])
        trajectory_ids = torch.as_tensor(
            [dataset.windows[index].trajectory for index in row["A_indices"]],
            dtype=torch.long, device=device)
        member_mask = torch.as_tensor(
            row["A_bootstrap_mask"], dtype=torch.bool, device=device)
        prediction, main, _values = _forward_main(
            view, history, target, trajectory_ids, member_mask, normalizer,
            int(row["update"]), int(row["rng"]["A"]["dropout"]))
        main_norm = float(prediction_gradient_norm(main, prediction))
        if main_norm <= 0.0:
            raise RuntimeError("point objective has zero calibration gradient")
        arm_reports = {}
        for arm in ("candidate", "control"):
            auxiliary = local_lag_loss(
                history, prediction, target, arm,
                row["A_null_donor_positions"])
            arm_reports[arm] = {
                "raw_aux_prediction_gradient_norm": float(
                    prediction_gradient_norm(auxiliary, prediction)),
                "raw_aux_loss": float(auxiliary.detach()),
            }
        aux_mean = sum(
            report["raw_aux_prediction_gradient_norm"]
            for report in arm_reports.values()) / 2
        if aux_mean <= 0.0:
            raise RuntimeError(
                "local-lag auxiliary has zero calibration gradient")
        panel.append({
            "update": int(row["update"]),
            "prediction_sha256": tensor_sha256(prediction),
            "main_prediction_gradient_norm": main_norm,
            "arms": arm_reports,
            "mean_aux_to_main_gradient_ratio": aux_mean / main_norm,
        })
        del history, target, prediction, main, auxiliary

    ratios = sorted(
        report["mean_aux_to_main_gradient_ratio"] for report in panel)
    rank = math.ceil(AUX_CALIBRATION_QUANTILE * len(ratios)) - 1
    selected_ratio = ratios[rank]
    weight = AUX_GRADIENT_FRACTION / selected_ratio
    if not torch.isfinite(torch.tensor(weight)) or weight <= 0.0:
        raise RuntimeError("local-lag auxiliary calibration is invalid")
    if (dr1.model_state_sha256(model.state_dict()) != post_warmup_state
            or nested_state_sha256(optimizer.state_dict())
            != post_warmup_optimizer_state):
        raise RuntimeError("read-only calibration panel changed branch state")
    branch_model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    branch_optimizer_state = nested_state_to_cpu(optimizer.state_dict())
    optimizer_state_sha = nested_state_sha256(branch_optimizer_state)
    if dr1.model_state_sha256(branch_model_state) != post_warmup_state:
        raise RuntimeError("CPU branch state changed the warmup model")
    if optimizer_state_sha != post_warmup_optimizer_state:
        raise RuntimeError("CPU branch state changed the warmup optimizer")
    result = {
        "fraction": AUX_GRADIENT_FRACTION,
        "weight": weight,
        "warmup_updates": AUX_WARMUP_UPDATES,
        "warmup_first_main_loss": warmup_losses[0],
        "warmup_last_main_loss": warmup_losses[-1],
        "post_warmup_model_state_sha256": post_warmup_state,
        "post_warmup_optimizer_state_sha256": optimizer_state_sha,
        "buffer_state_unchanged": True,
        "frozen_uq_state_unchanged": True,
        "panel_updates": AUX_CALIBRATION_PANEL_UPDATES,
        "panel_update_range": [panel[0]["update"], panel[-1]["update"]],
        "panel_quantile": AUX_CALIBRATION_QUANTILE,
        "panel_nearest_rank_index": rank,
        "selected_aux_to_main_gradient_ratio": selected_ratio,
        "panel": panel,
        "shared_prediction_per_panel_batch": True,
        "frozen_before_branch_update": AUX_WARMUP_UPDATES + 1,
    }
    del optimizer, view, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    branch_state = {
        "model_state_dict": branch_model_state,
        "optimizer_state_dict": branch_optimizer_state,
        "model_state_sha256": post_warmup_state,
        "optimizer_state_sha256": optimizer_state_sha,
    }
    return result, branch_state


def save_endpoint(path: Path, model, *, arm: str, start: Mapping[str, object],
                  normalizer, schedule_sha256: str, training: Mapping[str, object]) -> dict:
    """M13-native endpoint metadata; no M11 schema/experiment relabelling."""
    if path.exists():
        raise FileExistsError(path)
    ensemble_train.common.set_model_stats(model, normalizer)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state_sha = dr1.model_state_sha256(state)
    normalizer_values = joint.normalizer_payload(normalizer)
    payload = {"schema": ENDPOINT_SCHEMA, "experiment": EXPERIMENT,
               "phase": "real-local-lag-covariance", "arm": arm,
               "iteration": int(training["updates"]), "model_state_dict": state,
               "model_state_sha256": state_sha, "model_config": model.architecture_config(),
               "start": dict(start), "normalizer": normalizer_values,
               "normalizer_sha256": m11.canonical_sha256(normalizer_values),
               "schedule_sha256": schedule_sha256, "training": dict(training),
               "held_opened_before_seal": False, "optimizer_state_saved": False}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)
    return {"arm": arm, "path": str(path.resolve()), "file_sha256": m11.sha256(path),
            "model_state_sha256": state_sha,
            "model_config_sha256": m11.canonical_sha256(payload["model_config"]),
            "normalizer_sha256": payload["normalizer_sha256"],
            "schedule_sha256": schedule_sha256, "updates": int(training["updates"])}


def compare_evaluations(candidate: Mapping[str, object], control: Mapping[str, object]) -> dict:
    """M13 gate: same metrics, a stricter component floor, M13 decisions."""
    if candidate["target_audits"] != control["target_audits"]:
        raise RuntimeError("candidate/control held targets differ")
    keys = (*m11.POINT_WEIGHTS, "native_sps_score", "official_point_contribution_46_10_9")
    delta = {key: float(candidate["pooled"][key]) - float(control["pooled"][key]) for key in keys}
    trajectory = {name: float(candidate["per_trajectory"][name]["official_point_contribution_46_10_9"]) - float(control["per_trajectory"][name]["official_point_contribution_46_10_9"]) for name in candidate["per_trajectory"]}
    checks = {"pooled_point_delta": delta["official_point_contribution_46_10_9"] >= .04,
              "rel_floor": delta["rel_l2_score"] >= -.03,
              "tke_floor": delta["tke_rel_l2_score"] >= -.03,
              "mvpe_floor": delta["mvpe_rel_l2_score"] >= -.03,
              "positive_trajectories": sum(x > 0 for x in trajectory.values()) >= 12,
              "worst_trajectory": min(trajectory.values()) >= -.10}
    passed = all(checks.values())
    return {"candidate": candidate["pooled"], "control": control["pooled"], "delta": delta,
            "trajectory_point_delta": trajectory,
            "positive_trajectories": sum(x > 0 for x in trajectory.values()),
            "worst_trajectory_point_delta": min(trajectory.values()),
            "mean_equal_trajectory_point_delta": sum(trajectory.values()) / len(trajectory),
            "gate": {"pass": passed, "conditions": checks,
                     "decision": "GO_M13_LOCAL_LAG_COVARIANCE" if passed else "STOP_M13_LOCAL_LAG_COVARIANCE"}}


def load_endpoint_model(seal: Mapping[str, object], normalizer, device: torch.device):
    path = Path(str(seal["path"]))
    if m11.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("endpoint file changed after sealing")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENDPOINT_SCHEMA:
        raise ValueError("M13 endpoint schema changed")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or dr1.model_state_sha256(state) != seal["model_state_sha256"]:
        raise RuntimeError("endpoint model state changed")
    if m11.canonical_sha256(payload["model_config"]) != seal["model_config_sha256"]:
        raise RuntimeError("endpoint model config changed")
    values = joint.normalizer_payload(normalizer)
    if payload.get("normalizer") != values or payload.get("normalizer_sha256") != m11.canonical_sha256(values):
        raise RuntimeError("endpoint Real normalizer changed")
    config = dict(payload["model_config"])
    if config.pop("architecture", None) != "ensemble_unet":
        raise ValueError("endpoint is not an EnsembleUNet2d")
    model = m11.EnsembleUNet2d(m11.EnsembleUNet2dConfig(**config)).float()
    model.load_state_dict(state, strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def train_arm(arm: str, start: Mapping[str, object], dataset, schedule: Mapping,
              normalizer, device: torch.device, output_dir: Path,
              access: m11.PairAccessAudit, deadline: float, aux_weight: float,
              branch_state: Mapping[str, object]):
    ensemble_train.seed_everything(int(schedule["seed"]))
    model = joint.build_start_model(start, device)
    ensemble_train.common.set_model_stats(model, normalizer)
    trainable, trainability = joint.configure_point_only_trainable(model)
    if (sum(parameter.numel() for parameter in model.parameters())
            != m11.EXPECTED_PARAMETERS
            or trainability["trainable_parameters"]
            != m11.EXPECTED_TRAINABLE_PARAMETERS
            or trainability["frozen_uq_parameters"]
            != m11.EXPECTED_FROZEN_PARAMETERS):
        raise RuntimeError("B210 point/UQ parameter partition changed")
    model.train()
    view = m11.PointTrainingView(model)
    if tuple(map(id, trainable)) != tuple(
            id(p) for p in view.parameters() if p.requires_grad):
        raise RuntimeError("point view changed trainable order")
    model.load_state_dict(branch_state["model_state_dict"], strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    initial_state = dr1.model_state_sha256(model.state_dict())
    if initial_state != branch_state["model_state_sha256"]:
        raise RuntimeError("M13 arm did not load the shared warmup model")
    initial_buffer = m11.buffer_state_sha256(model)
    uq_start = dr1.model_state_sha256(joint.uq_parameter_state(model))
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    if (nested_state_sha256(branch_state["optimizer_state_dict"])
            != branch_state["optimizer_state_sha256"]):
        raise RuntimeError("shared warmup optimizer was mutated before branching")
    optimizer_state_copy = nested_state_to_cpu(
        branch_state["optimizer_state_dict"])
    optimizer.load_state_dict(optimizer_state_copy)
    del optimizer_state_copy
    initial_optimizer_state = nested_state_sha256(optimizer.state_dict())
    if initial_optimizer_state != branch_state["optimizer_state_sha256"]:
        raise RuntimeError("M13 arm did not load the shared warmup optimizer")
    actual, traces = [], []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for row in schedule["rows"][AUX_WARMUP_UPDATES:]:
        if time.perf_counter() > deadline:
            raise TimeoutError("M13 wall budget expired")
        update = int(row["update"])
        optimizer.zero_grad(set_to_none=True)
        history, target = m11.prepare_batch(dataset, row["A_indices"], normalizer,
                                            device, row["rng"]["A"])
        trajectory_ids = torch.as_tensor([dataset.windows[i].trajectory for i in row["A_indices"]],
                                         dtype=torch.long, device=device)
        mask = torch.as_tensor(row["A_bootstrap_mask"], dtype=torch.bool, device=device)
        prediction, main, values = _forward_main(
            view, history, target, trajectory_ids, mask, normalizer, update,
            int(row["rng"]["A"]["dropout"]))
        auxiliary = local_lag_loss(
            history, prediction, target, arm,
            row["A_null_donor_positions"])
        loss = main + aux_weight * auxiliary
        loss.backward()
        clipped = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        if not bool(torch.isfinite(clipped)):
            raise FloatingPointError("non-finite M13 gradient")
        if any(parameter.grad is not None
               for _name, parameter in joint.frozen_named_parameters(model)):
            raise RuntimeError("frozen UQ parameter received a gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        raw_aux_loss = float(auxiliary.detach())
        weighted_aux_loss = float((aux_weight * auxiliary).detach())
        traces.append({"update": update, "main_loss": values["loss"],
                       "raw_aux_loss": raw_aux_loss,
                       "weighted_aux_loss": weighted_aux_loss,
                       "gradient_norm": float(clipped.detach())})
        actual.append({"update": update, "A": row["A"],
                       "A_bootstrap_mask": row["A_bootstrap_mask"],
                       "A_null_donor_positions": row["A_null_donor_positions"],
                       "rng": row["rng"]["A"]})
        if update % 20 == 0 or update == len(schedule["rows"]):
            print(json.dumps({"event": "m13_progress", "arm": arm, "update": update,
                              "main_loss": values["loss"],
                              "raw_aux_loss": raw_aux_loss,
                              "gradient_norm": float(clipped.detach())}), flush=True)
    uq_end = dr1.model_state_sha256(joint.uq_parameter_state(model))
    if m11.buffer_state_sha256(model) != initial_buffer or uq_end != uq_start:
        raise RuntimeError("M13 changed buffers or frozen UQ state")
    stream = m11.canonical_sha256(actual)
    total_updates = AUX_WARMUP_UPDATES + len(actual)
    training = {"arm": arm, "updates": total_updates,
                "optimizer_steps": total_updates,
                "branch_updates": len(actual),
                "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                "aux_weight": aux_weight, "aux_trace": traces,
                "shared_warmup_updates": AUX_WARMUP_UPDATES,
                "auxiliary_optimizer_steps": len(actual),
                "post_warmup_model_state_sha256": initial_state,
                "post_warmup_optimizer_state_sha256": initial_optimizer_state,
                "stream_sha256": stream, "initial_model_state_sha256": initial_state,
                "initial_buffer_state_sha256": initial_buffer,
                "final_buffer_state_sha256": m11.buffer_state_sha256(model),
                "frozen_uq_start_sha256": uq_start,
                "frozen_uq_end_sha256": uq_end, "trainability": trainability,
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda" else 0)}
    seal = save_endpoint(output_dir / f"{arm}_raw.pth", model, arm=arm,
                            start=start, normalizer=normalizer,
                            schedule_sha256=str(schedule["sha256"]), training=training)
    access.seal(arm, seal)
    del optimizer, view, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return training, seal


def run(args: argparse.Namespace) -> dict:
    identity = m11.git_identity()
    if args.mode == "formal" and identity["status_porcelain"]:
        raise RuntimeError("formal M13 requires a clean Git worktree")
    if args.mode in {"formal", "smoke"}:
        if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
            raise RuntimeError("M13 execution requires explicit acknowledgement")
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
            raise RuntimeError("M13 execution requires CUDA")
    if args.mode == "formal" and (args.updates != FORMAL_UPDATES or args.batch_size != FORMAL_BATCH_SIZE):
        raise ValueError("formal M13 requires exactly 1920 updates and batch 4")
    if args.mode == "smoke" and (args.updates != SMOKE_UPDATES or args.batch_size != FORMAL_BATCH_SIZE):
        raise ValueError("smoke M13 requires exactly 24 updates and batch 4")
    if m11.sha256(args.runtime_manifest) != m11.RUNTIME_MANIFEST_FILE_SHA256:
        raise RuntimeError("runtime manifest file changed")
    rows, runtime_report = dr1.load_manifest(args.runtime_manifest, m11.RUNTIME_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [r for r in rows if int(r["aoa"]) in m11.FIT_AOAS]
    held_rows = [r for r in rows if int(r["aoa"]) == m11.HELD_AOA]
    schedule = build_m13_schedule(
        rows, args.updates, args.batch_size, args.seed)
    if (args.mode == "formal"
            and schedule["base_schedule_sha256"]
            != BASE_FORMAL_SCHEDULE_SHA256):
        raise RuntimeError("formal M13 base schedule digest changed")
    if args.mode == "formal" and schedule["sha256"] != FORMAL_SCHEDULE_SHA256:
        raise RuntimeError("formal M13 schedule digest changed")
    if args.mode == "contract":
        return {"schema": SCHEMA, "mode": "contract", "schedule": {k:v for k,v in schedule.items() if k != "rows"}}
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True)
    m11.write_json_atomic(args.output_dir / "schedule.json", schedule)
    access = m11.PairAccessAudit([r["file"] for r in fit_rows], [r["file"] for r in held_rows])
    trajectories, metadata, windows = dr1.load_prefix_rows(args.real_dir, fit_rows, "fit", access)
    statistics = joint.fit_only_real_statistics(trajectories, metadata, windows, m11.HELD_AOA,
                                                 allowed_files=[r["file"] for r in fit_rows])
    normalizer = smu1.FoldNormalizer(statistics, device)
    dataset = ensemble_train.common.repro.WindowDataset(trajectories, windows)
    resolved = m11.resolve_schedule_indices(schedule, windows, metadata)
    for raw, indices in zip(schedule["rows"], resolved["rows"]): raw["A_indices"] = indices["A_indices"]
    start = joint.inspect_start_checkpoint(args.start_checkpoint, m11.START_SHA256)
    config = {"schema": SCHEMA, "experiment": EXPERIMENT, "mode": args.mode,
              "git": identity, "seed": args.seed, "updates": args.updates,
              "batch_size": args.batch_size, "optimizer": {"name": "AdamW", "lr": LR,
              "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP},
              "auxiliary": {"loss": "raw_descriptor_mse",
                            "warmup_updates": AUX_WARMUP_UPDATES,
                            "calibration_panel_updates":
                                AUX_CALIBRATION_PANEL_UPDATES,
                            "calibration_quantile":
                                AUX_CALIBRATION_QUANTILE,
                            "gradient_fraction": AUX_GRADIENT_FRACTION},
              "runtime_manifest": runtime_report, "start": start,
              "schedule_sha256": schedule["sha256"], "aux_calibration": None,
              "held_opened_before_dual_seal": False}
    # Written before calibration/training so an interruption remains auditable.
    m11.write_json_atomic(args.output_dir / "config.json", config)
    calibration, branch_state = calibrate_auxiliary_weight(
        start, dataset, schedule, normalizer, device)
    config["aux_calibration"] = calibration
    m11.write_json_atomic(args.output_dir / "config.json", config)
    m11.write_json_atomic(args.output_dir / "calibration.json", calibration)
    deadline = time.perf_counter() + args.wall_budget_seconds
    candidate_training, candidate_seal = train_arm(
        "candidate", start, dataset, schedule, normalizer, device,
        args.output_dir, access, deadline, calibration["weight"],
        branch_state)
    control_training, control_seal = train_arm(
        "control", start, dataset, schedule, normalizer, device,
        args.output_dir, access, deadline, calibration["weight"],
        branch_state)
    del branch_state
    if candidate_training["initial_model_state_sha256"] != control_training["initial_model_state_sha256"] or candidate_training["stream_sha256"] != control_training["stream_sha256"]:
        raise RuntimeError("paired M13 arms diverged before auxiliary")
    seals = {"candidate": candidate_seal, "control": control_seal}
    if args.mode == "smoke":
        return {
            "schema": SCHEMA,
            "status": "MECHANICAL_SMOKE_COMPLETE_NO_HELD_OPEN",
            "candidate_training": candidate_training,
            "control_training": control_training,
            "seals": seals,
            "access": access.report(),
            "calibration": calibration,
        }
    held_rows = m11.validate_evaluation_manifest(args.evaluation_manifest, rows)
    held, groups = m11.load_held_tail(args.real_dir, held_rows, access)
    evaluations = {}
    for arm in ("candidate", "control"):
        model = load_endpoint_model(seals[arm], normalizer, device)
        evaluations[arm] = m11.evaluate_endpoint(model, held, groups, normalizer, device, deadline)
        del model
    comparison = compare_evaluations(evaluations["candidate"], evaluations["control"])
    return {"schema": SCHEMA, "status": comparison["gate"]["decision"], "config": config, "candidate_training": candidate_training, "control_training": control_training, "seals": seals, "evaluations": evaluations, "comparison": comparison, "access": access.report()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "smoke", "formal"), default="contract")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-checkpoint", type=Path, default=m11.DEFAULT_START)
    parser.add_argument("--runtime-manifest", type=Path, default=m11.DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--evaluation-manifest", type=Path, default=m11.DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--real-dir", type=Path, default=Path("/root/autodl-tmp/realpde/data/real"))
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--updates", type=int, default=FORMAL_UPDATES)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE); parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wall-budget-seconds", type=float, default=86400.0); parser.add_argument("--execute", action="store_true"); parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode != "contract" and args.output_dir is None: raise ValueError("execution modes require --output-dir")
    result = run(args)
    if args.mode != "contract": m11.write_json_atomic(args.output_dir / "result.json", result)
    print(json.dumps({"event": "m13_complete", "status": result.get("status", "CONTRACT_VALID")}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
