#!/usr/bin/env python3
"""Dense-stride5 MS-Closure with a training-only shifted-context teacher.

The completed dense-stride5 run is the exact control.  Each ordinary 20->20
training window beginning at ``t`` still receives its native supervised loss.
In addition, the same model reads the released trajectory at ``t + 5`` under
``torch.no_grad()``.  Its first fifteen predictions are a fixed training-time
teacher for the main prediction's final fifteen frames.  No model parameter,
inference path, optimizer sample exposure, normalizer, or held evaluator
changes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_multiple_shooting_closure_dense_stride5_connected_credit_long as dense_reference  # noqa: E402
import train_multiple_shooting_closure_dense_stride5_long as dense  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402


control = dense.control
SCHEMA = "multiple_shooting_dense_stride5_shifted_teacher_held10_v1"
ACKNOWLEDGEMENT = "RUN_MS_DENSE5_SHIFTED_TEACHER_HELD10"
TRAIN_STRIDE = dense.TRAIN_STRIDE
HELD_STRIDE = dense.HELD_STRIDE
SHIFT_FRAMES = 5
OVERLAP_FRAMES = 20 - SHIFT_FRAMES

# The formal reference is passed explicitly rather than being embedded as a
# private absolute path.  These identities prevent accidentally comparing to a
# different dense experiment while allowing the trainer to remain portable.
DENSE_CONTROL_RESULT_SHA256 = (
    "801071f4bc62bdf43ade3107280b80d1329984fb782c7d1b874e515c7c74828a")
DENSE_CONTROL_REFERENCE = dense_reference.DENSE_CONTROL_REFERENCE


def shifted_history_batch(dataset: base.SimWindowDataset,
                          indices: np.ndarray | list[int]) -> torch.Tensor:
    """Return the released 20-frame history beginning five frames later.

    The dense window convention leaves enough trajectory suffix for this at
    every legal start, including ``start=955`` of a 1000-frame trajectory.
    This helper deliberately uses only the source trajectory and does not
    synthesize a rollout or alter the optimizer's main window keys.
    """
    histories = []
    for raw_index in indices:
        index = int(raw_index)
        window = dataset.windows[index]
        start = int(window.start) + SHIFT_FRAMES
        history = dataset.trajectories[int(window.trajectory)][start:start + 20]
        if history.shape[0] != 20:
            raise ValueError(
                "shifted teacher history is outside its released trajectory: "
                f"index={index} trajectory={window.trajectory} start={window.start}")
        histories.append(history)
    if not histories:
        raise ValueError("shifted teacher batch is empty")
    return torch.stack(histories)


def validate_shifted_teacher_support(
        trajectories: list[torch.Tensor], windows: list[base.SimWindow]) -> dict[str, object]:
    """Prove that all dense keys admit the shifted 20-frame input."""
    if not windows:
        raise ValueError("shifted-teacher support is empty")
    invalid = [
        (int(window.trajectory), int(window.start))
        for window in windows
        if int(window.start) + SHIFT_FRAMES + 20
        > len(trajectories[int(window.trajectory)])]
    if invalid:
        raise ValueError(f"shifted-teacher support escapes trajectories: {invalid[:3]}")
    return {
        "shift_frames": SHIFT_FRAMES,
        "overlap_frames": OVERLAP_FRAMES,
        "windows_checked": len(windows),
        "all_shifted_histories_in_released_trajectories": True,
        "maximum_main_start": max(int(window.start) for window in windows),
    }


def shifted_teacher_accuracy(
        prediction: torch.Tensor, teacher_prediction: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official differentiable accuracy on the aligned 15-frame overlap.

    ``teacher_prediction`` is detached here, rather than trusting callers to
    do so, which makes the no-gradient teacher contract mechanically local.
    """
    if prediction.shape[1] != 20 or teacher_prediction.shape[1] != 20:
        raise ValueError("shifted teacher requires 20-frame predictions")
    if prediction.shape[0] != teacher_prediction.shape[0]:
        raise ValueError("shifted teacher batch sizes differ")
    if prediction.shape[2:] != teacher_prediction.shape[2:]:
        raise ValueError("shifted teacher prediction shapes differ")
    return base.accuracy_objective(
        prediction[:, SHIFT_FRAMES:],
        teacher_prediction[:, :OVERLAP_FRAMES].detach(),
        normalizer, control.LOSS_WEIGHTS)


def train(model, dataset: base.SimWindowDataset, normalizer,
          device: torch.device, *, updates: int, batch_size: int,
          trace_interval: int) -> tuple[dict[str, object], ema_lib.FP32EMAShadow]:
    """The dense control recipe plus one equal-weight overlap accuracy term."""
    schedule = control.make_schedule(len(dataset), updates, batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=control.LR, weight_decay=control.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=control.UPDATES if updates == control.UPDATES else updates)
    ema = ema_lib.FP32EMAShadow(model, control.EMA_DECAY)
    trace: list[dict[str, object]] = []
    interval: list[float] = []
    started = time.perf_counter()

    for update, indices in enumerate(schedule, start=1):
        items = [dataset[int(index)] for index in indices]
        history = torch.stack([item[0] for item in items]).to(device)
        target = torch.stack([item[1] for item in items]).to(device)
        shifted_history = shifted_history_batch(dataset, indices).to(device)
        history, target = normalizer.preprocess(history, target)
        shifted_history = (
            (shifted_history - normalizer.mean_input) / normalizer.std_input)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        # This forward deliberately shares the current parameter values but
        # creates no autograd graph and cannot itself change optimizer exposure.
        with torch.no_grad():
            teacher_prediction = model(shifted_history)
        main_accuracy, main_details = base.accuracy_objective(
            prediction, target, normalizer, control.LOSS_WEIGHTS)
        mse_uv = (prediction[..., :2] - target[..., :2]).square().mean()
        overlap_accuracy, overlap_details = shifted_teacher_accuracy(
            prediction, teacher_prediction, normalizer)
        main_loss = main_accuracy + control.MSE_WEIGHT * mse_uv
        loss = main_loss + overlap_accuracy
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), control.GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient at update {update}")
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=update)
        scheduler.step()
        interval.append(float(loss.detach()))

        if update == 1 or update % trace_interval == 0 or update == updates:
            row = {
                "update": update,
                "loss_mean_since_last_report": float(np.mean(interval)),
                "main_accuracy": float(main_accuracy.detach()),
                "main_mse_uv": float(mse_uv.detach()),
                "main_loss": float(main_loss.detach()),
                "overlap_accuracy_equal_weight": float(overlap_accuracy.detach()),
                "teacher_requires_grad": bool(teacher_prediction.requires_grad),
                "grad_norm_before_clip": float(grad_norm.detach()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "wall_seconds": time.perf_counter() - started,
                **{f"main_{name}": float(value) for name, value in main_details.items()},
                **{f"overlap_{name}": float(value) for name, value in overlap_details.items()},
            }
            trace.append(row)
            interval.clear()
            print(json.dumps(
                {"event": "ms_dense5_shifted_teacher_progress", **row},
                sort_keys=True), flush=True)

    if not all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in model.state_dict().values()):
        raise FloatingPointError("terminal model state is non-finite")
    return ({
        "updates": updates,
        "main_sample_stream_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "main_sample_exposure": int(updates * batch_size),
        "teacher": {
            "kind": "same_model_shifted_context_stop_gradient",
            "shift_frames": SHIFT_FRAMES,
            "overlap_frames": OVERLAP_FRAMES,
            "forward_autograd": False,
            "loss": "accuracy(1,4,1) on main[5:20] vs teacher[0:15]",
            "loss_weight": 1.0,
            "extra_optimizer_sample_exposure": 0,
        },
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
    }, ema)


def save_endpoint(
        path: Path, *, kind: str, state: dict[str, torch.Tensor],
        git: dict[str, object], normalizer, updates: int,
        manifest: dict[str, object], teacher_support: dict[str, object],
        ) -> dict[str, str]:
    atomic.atomic_torch_save({
        "schema": SCHEMA,
        "endpoint_kind": kind,
        "updates": updates,
        "rollout_mode": "carry_detached",
        "model_config": asdict(control.MODEL_CONFIG),
        "normalizer": control.normalizer_record(normalizer),
        "training_window_manifest": manifest,
        "shifted_teacher": teacher_support,
        "git": git,
        "model_state": state,
    }, path)
    return {
        "path": str(path.resolve()),
        "sha256": atomic.file_sha256(path),
        "model_state_sha256": base.state_sha(state),
    }


def dense_control_identity(
        manifest: dict[str, object], normalizer,
        training_window_count: int) -> dict[str, object]:
    normalizer_record = control.normalizer_record(normalizer)
    schedule = control.make_schedule(
        training_window_count, control.UPDATES, control.BATCH_SIZE)
    observed = {
        "training_window_manifest_sha256": base.canonical_sha(manifest),
        "normalizer_sha256": base.canonical_sha(normalizer_record),
        "sample_stream_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
    }
    expected = {
        name: DENSE_CONTROL_REFERENCE[name] for name in observed}
    if observed != expected:
        raise RuntimeError("shifted-teacher candidate is not dense-control-matched")
    return {"observed": observed, "expected": expected, "all_matched": True}


def _evaluation_comparison(candidate: dict[str, object],
                           reference: dict[str, object]) -> dict[str, object]:
    """Compare aggregate plus every fixed held trajectory and horizon."""
    result: dict[str, object] = {
        "aggregate": base.comparison(candidate["aggregate"], reference["aggregate"]),
    }
    for field in ("per_trajectory", "per_prefix_horizon"):
        candidate_rows = candidate[field]
        reference_rows = reference[field]
        if set(candidate_rows) != set(reference_rows):
            raise RuntimeError(f"dense reference {field} keys differ")
        result[field] = {
            str(key): base.comparison(candidate_rows[key], reference_rows[key])
            for key in sorted(candidate_rows, key=lambda value: str(value))
        }
    return result


def load_exact_dense_control(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Load the completed dense result and validate its identity before use."""
    actual_sha = atomic.file_sha256(path)
    if actual_sha != DENSE_CONTROL_RESULT_SHA256:
        raise RuntimeError(
            "dense control result SHA mismatch: "
            f"{actual_sha} != {DENSE_CONTROL_RESULT_SHA256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload.get("training", {})
    manifest = training.get("training_window_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("dense control result lacks training manifest")
    if base.canonical_sha(manifest) != DENSE_CONTROL_REFERENCE[
            "training_window_manifest_sha256"]:
        raise RuntimeError("dense control result manifest differs")
    if base.canonical_sha(payload.get("normalizer")) != DENSE_CONTROL_REFERENCE[
            "normalizer_sha256"]:
        raise RuntimeError("dense control result normalizer differs")
    training_result = payload.get("training_result", {})
    if training_result.get("sample_stream_sha256") != DENSE_CONTROL_REFERENCE[
            "sample_stream_sha256"]:
        raise RuntimeError("dense control result sample stream differs")
    held = payload.get("held_AoA10")
    if not isinstance(held, dict) or set(held) != {"raw", "ema999"}:
        raise RuntimeError("dense control result lacks raw/ema held evaluations")
    return payload, {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "training_window_manifest_sha256": base.canonical_sha(manifest),
        "normalizer_sha256": base.canonical_sha(payload["normalizer"]),
        "sample_stream_sha256": training_result["sample_stream_sha256"],
    }


def formal(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError(
            f"formal run requires --execute --acknowledge {ACKNOWLEDGEMENT}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    if args.sim_dir is None or args.output_dir is None:
        raise ValueError("formal training requires --sim-dir and --output-dir")
    if args.dense_control_result is None:
        raise ValueError("formal run requires --dense-control-result")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty output directory")
    dense_payload, dense_record = load_exact_dense_control(args.dense_control_result)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    git = base.git_report()
    inventory = base.inventory(args.sim_dir)
    fit_rows = [
        row for row in inventory if int(row["aoa"]) in control.FIT_AOAS]
    held_rows = [
        row for row in inventory if int(row["aoa"]) == control.HELD_AOA]
    fit_trajectories, sparse_fit_windows, _fit_files = base.load_fields(
        args.sim_dir, fit_rows)
    normalizer = base.ChannelNormalizer(
        *base.window_statistics(fit_trajectories, sparse_fit_windows), device=device)
    training_windows = dense.dense_windows(fit_trajectories)
    manifest = dense.window_manifest(training_windows)
    manifest["fit_inventory_sha256"] = base.canonical_sha(fit_rows)
    control_identity = dense_control_identity(
        manifest, normalizer, len(training_windows))
    teacher_support = validate_shifted_teacher_support(
        fit_trajectories, training_windows)
    dataset = base.SimWindowDataset(fit_trajectories, training_windows)

    model = control.construct_model(device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    training, ema = train(
        model, dataset, normalizer, device, updates=control.UPDATES,
        batch_size=control.BATCH_SIZE, trace_interval=control.TRACE_INTERVAL)
    if (training["main_sample_stream_sha256"]
            != control_identity["observed"]["sample_stream_sha256"]):
        raise RuntimeError("realized main sample stream changed from dense control")
    raw_state = control.snapshot(model)
    ema_state = dict(ema.materialize_model_state(model))
    checkpoints = {
        "raw": save_endpoint(
            args.output_dir / f"raw_u{control.UPDATES:05d}.pt",
            kind=f"raw_u{control.UPDATES:05d}", state=raw_state, git=git,
            normalizer=normalizer, updates=control.UPDATES, manifest=manifest,
            teacher_support=teacher_support),
        "ema": save_endpoint(
            args.output_dir / f"ema999_u{control.UPDATES:05d}.pt",
            kind=f"ema999_u{control.UPDATES:05d}", state=ema_state, git=git,
            normalizer=normalizer, updates=control.UPDATES, manifest=manifest,
            teacher_support=teacher_support),
    }

    del dataset, fit_trajectories
    gc.collect()
    torch.cuda.empty_cache()
    held_trajectories, held_windows, held_files = base.load_fields(
        args.sim_dir, held_rows)
    evaluations = {
        "raw": control.evaluate_endpoint(
            model, raw_state, held_trajectories, held_windows, held_files,
            normalizer, device),
        "ema999": control.evaluate_endpoint(
            model, ema_state, held_trajectories, held_windows, held_files,
            normalizer, device),
    }
    comparisons = {
        "raw_minus_dense_raw": _evaluation_comparison(
            evaluations["raw"], dense_payload["held_AoA10"]["raw"]),
        "ema999_minus_dense_ema999": _evaluation_comparison(
            evaluations["ema999"], dense_payload["held_AoA10"]["ema999"]),
        "ema999_minus_raw": _evaluation_comparison(
            evaluations["ema999"], evaluations["raw"]),
    }
    result = {
        "schema": SCHEMA,
        "git": git,
        "dense_control_reference": dense_record,
        "dense_control_identity": control_identity,
        "training": {
            "fit_aoas": list(control.FIT_AOAS),
            "held_aoa": control.HELD_AOA,
            "fresh_initialization": True,
            "training_stride": TRAIN_STRIDE,
            "held_stride": HELD_STRIDE,
            "normalizer_window_stride": HELD_STRIDE,
            "training_window_manifest": manifest,
            "sparse_control_windows": len(sparse_fit_windows),
            "released_fit_trajectories_only": True,
            "rollout_mode": "carry_detached",
            "model_config": asdict(control.MODEL_CONFIG),
            "architecture": model.architecture_config(),
            "parameters": sum(value.numel() for value in model.parameters()),
            "updates": control.UPDATES,
            "batch_size": control.BATCH_SIZE,
            "seed": control.SEED,
            "schedule_seed": control.SCHEDULE_SEED,
            "optimizer": "AdamW",
            "lr": control.LR,
            "weight_decay": control.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": control.UPDATES,
            "grad_clip": control.GRAD_CLIP,
            "main_loss": "accuracy(1,4,1)+0.25*normalized_uv_mse",
            "teacher_loss": (
                "1.0*accuracy(1,4,1) on main[5:20] vs "
                "stopgrad(shift5_teacher[0:15])"),
            "teacher_support": teacher_support,
            "ema_decay": control.EMA_DECAY,
            "trace_interval": control.TRACE_INTERVAL,
        },
        "normalizer": control.normalizer_record(normalizer),
        "training_result": training,
        "checkpoints": checkpoints,
        "held_AoA10": evaluations,
        "comparisons": comparisons,
        "selection_gate": None,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "ms_dense5_shifted_teacher_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "raw": evaluations["raw"]["aggregate"],
        "ema999": evaluations["ema999"]["aggregate"],
        "aggregate_comparisons": {
            name: value["aggregate"] for name, value in comparisons.items()},
        "selection_gate": None,
    }, sort_keys=True), flush=True)
    return result


def smoke(device: torch.device) -> dict[str, object]:
    model = control.construct_model(device, tiny=True)
    trajectory = torch.randn(60, 16, 16, 3)
    trajectories = [trajectory]
    windows = dense.dense_windows(trajectories)
    support = validate_shifted_teacher_support(trajectories, windows)
    dataset = base.SimWindowDataset(trajectories, windows)
    normalizer = base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    training, ema = train(
        model, dataset, normalizer, device,
        updates=2, batch_size=1, trace_interval=1)
    raw = model(dataset[0][0][None].to(device))
    model.load_state_dict(ema.materialize_model_state(model), strict=True)
    averaged = model(dataset[0][0][None].to(device))
    return {
        "schema": SCHEMA,
        "smoke": "PASS",
        "device": str(device),
        "training_window_manifest": dense.window_manifest(windows),
        "teacher_support": support,
        "training": training,
        "raw_finite": bool(torch.isfinite(raw).all()),
        "ema_finite": bool(torch.isfinite(averaged).all()),
        "output_shape": list(raw.shape),
        "selection_gate": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "smoke", "formal"), default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dense-control-result", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        model = control.construct_model(torch.device("cpu"))
        print(json.dumps({
            "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "training_stride": TRAIN_STRIDE,
            "held_stride": HELD_STRIDE,
            "shifted_teacher": {
                "shift_frames": SHIFT_FRAMES,
                "overlap_frames": OVERLAP_FRAMES,
                "forward_autograd": False,
                "loss_weight": 1.0,
                "main_sample_exposure_unchanged": True,
            },
            "model_config": asdict(control.MODEL_CONFIG),
            "parameters": sum(value.numel() for value in model.parameters()),
            "dense_control_result_sha256": DENSE_CONTROL_RESULT_SHA256,
            "selection_gate": None,
        }, sort_keys=True, indent=2))
        return
    device = torch.device(args.device)
    result = smoke(device) if args.mode == "smoke" else formal(args, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
