#!/usr/bin/env python3
"""Strict whole-AoA Sim screen for the temporal-axis transport operator.

The candidate and control have identical parameter tensors and training
streams.  Their only difference is whether each matched depthwise kernel acts
causally across observed time or independently along the spatial x axis.
Held flow fields are not loaded until both endpoint files have been sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Iterable

import h5py
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (CODE_ROOT, Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_flow_backbone as backbone  # noqa: E402
from models.temporal_transport import (  # noqa: E402
    TemporalTransportConfig,
    TemporalTransportOperator,
)


SCHEMA = "temporal_transport_whole_aoa_sim_screen_v1"
AOAS = (0, 5, 10, 15, 20)
SEED = 210
UPDATES = 2_000
BATCH_SIZE = 8
HIDDEN_CHANNELS = 128
EXPECTED_FILES = 100
EXPECTED_FILES_PER_AOA = 20
EXPECTED_PARAMETERS = 1_343_112
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}


def load_scoring():
    path = REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
    spec = importlib.util.spec_from_file_location("temporal_axis_loao_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORING = load_scoring()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-aoa", type=int, choices=AOAS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def git_report() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "clean": not bool(dirty),
        "porcelain": dirty.splitlines(),
    }


def scan_inventory(sim_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(sim_dir.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            for key in ("u", "v", "p", "aoa", "re"):
                if key not in handle:
                    raise ValueError(f"{path.name} misses {key}")
            shapes = [tuple(handle[key].shape) for key in ("u", "v", "p")]
            if not (shapes[0] == shapes[1] == shapes[2]):
                raise ValueError(f"field shapes differ in {path.name}")
            records.append({
                "file": path.name,
                "aoa": int(np.asarray(handle["aoa"]).reshape(-1)[0]),
                "re": int(np.asarray(handle["re"]).reshape(-1)[0]),
                "frames": int(shapes[0][0]),
                "shape": list(shapes[0]),
            })
    if len(records) != EXPECTED_FILES:
        raise ValueError(f"expected {EXPECTED_FILES} Sim files, got {len(records)}")
    counts = {aoa: sum(int(row["aoa"]) == aoa for row in records) for aoa in AOAS}
    if any(counts[aoa] != EXPECTED_FILES_PER_AOA for aoa in AOAS):
        raise ValueError(f"unexpected AoA inventory {counts}")
    return records


def load_fields(
        sim_dir: Path, records: Iterable[dict[str, object]],
        ) -> tuple[list[torch.Tensor], list[object], list[str]]:
    trajectories: list[torch.Tensor] = []
    windows: list[object] = []
    window_files: list[str] = []
    for trajectory_index, record in enumerate(records):
        path = sim_dir / str(record["file"])
        with h5py.File(path, "r") as handle:
            channels = [
                np.asarray(handle[key][:, ::2, ::2], dtype=np.float32)
                for key in ("u", "v", "p")]
        trajectory = torch.from_numpy(np.ascontiguousarray(
            np.stack(channels, axis=-1)))
        if len(trajectory) != int(record["frames"]):
            raise ValueError(f"frame count changed for {path.name}")
        trajectories.append(trajectory)
        starts = range(0, len(trajectory) - 40, 20)
        for start in starts:
            windows.append(backbone.common.scratch.SimWindow(
                trajectory_index, start))
            window_files.append(path.name)
    return trajectories, windows, window_files


def make_config(mode: str) -> TemporalTransportConfig:
    return TemporalTransportConfig(
        hidden_channels=HIDDEN_CHANNELS,
        spatial_blocks=2,
        spatial_downsample=2,
        temporal_blocks=3,
        temporal_kernel=5,
        mixer_mode=mode,
        groups=8,
        use_coordinates=True,
        max_displacement=4.0,
        residual_limit=4.0,
        output_initialization_std=1e-3,
        predict_uncertainty=False,
    )


def construct_pair(device: torch.device) -> tuple[
        TemporalTransportOperator, TemporalTransportOperator, dict[str, object]]:
    torch.manual_seed(SEED)
    control = TemporalTransportOperator(
        make_config("framewise_spatial_control")).to(device)
    control_initial = {
        key: value.detach().cpu().clone()
        for key, value in control.state_dict().items()}
    torch.manual_seed(SEED)
    candidate = TemporalTransportOperator(
        make_config("causal_temporal")).to(device)
    candidate_initial = {
        key: value.detach().cpu().clone()
        for key, value in candidate.state_dict().items()}
    if control_initial.keys() != candidate_initial.keys() or any(
            not torch.equal(control_initial[key], candidate_initial[key])
            for key in control_initial):
        raise RuntimeError("candidate/control initial states are not bitwise identical")
    counts = [sum(p.numel() for p in model.parameters())
              for model in (control, candidate)]
    if counts != [EXPECTED_PARAMETERS, EXPECTED_PARAMETERS]:
        raise RuntimeError(f"parameter contract changed: {counts}")
    return control, candidate, {
        "state_sha256": state_sha(control_initial),
        "parameters": counts[0],
        "state_tensors": len(control_initial),
        "bitwise_identical": True,
    }


def batches(dataset, schedule: np.ndarray):
    for row in schedule:
        samples = [dataset[int(index)] for index in row]
        yield (torch.stack([item[0] for item in samples]),
               torch.stack([item[1] for item in samples]))


def train_arm(
        name: str, model: TemporalTransportOperator, dataset,
        schedule: np.ndarray, normalizer, device: torch.device,
        output_dir: Path) -> dict[str, object]:
    backbone.common.set_model_stats(model, normalizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES)
    trace = []
    interval: list[float] = []
    started = time.perf_counter()
    for update, (x, y) in enumerate(batches(dataset, schedule), 1):
        model.train()
        x, y = x.to(device), y.to(device)
        x_norm, y_norm = normalizer.preprocess(x, y)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_norm)
        accuracy, details = backbone.common.accuracy_objective(
            prediction, y_norm, normalizer, (1.0, 4.0, 1.0))
        mse = (prediction - y_norm).square().mean()
        loss = accuracy + .25 * mse
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} non-finite at update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            row = {
                "arm": name, "update": update,
                "loss_mean": float(np.mean(interval)),
                "loss_median": float(np.median(interval)),
                "mse": float(mse.detach()),
                **{key: float(value) for key, value in details.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            trace.append(row)
            interval.clear()
            print(json.dumps(row, sort_keys=True), flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    endpoint = output_dir / f"{name}_endpoint.pth"
    torch.save({
        "schema": SCHEMA, "arm": name, "updates": UPDATES,
        "model_config": model.architecture_config(),
        "model_state_dict": state,
    }, endpoint)
    return {
        "path": str(endpoint), "sha256": sha256_file(endpoint),
        "model_state_sha256": state_sha(state), "trace": trace,
        "wall_seconds": time.perf_counter() - started,
    }


def error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((
        SCORING.rel_l2_per_sample(prediction, target, 2),
        SCORING.tke_rel_l2_per_sample(prediction, target, 2),
        SCORING.mvpe_rel_l2_per_sample(prediction, target),
    ), axis=1).astype(np.float64)


def score(errors: np.ndarray) -> dict[str, float | int]:
    means = errors.mean(0, dtype=np.float64)
    result: dict[str, float | int] = {
        "samples": int(len(errors)),
        "rel_l2_score": SCORING.score_error(float(means[0])),
        "tke_score": SCORING.score_error(float(means[1])),
        "mvpe_score": SCORING.score_error(float(means[2])),
    }
    result["point_contribution_46_10_9"] = sum(
        POINT_WEIGHTS[key] * float(result[key]) for key in POINT_WEIGHTS)
    return result


def evaluate_pair(
        control: nn.Module, candidate: nn.Module, trajectories, windows,
        window_files: list[str], normalizer, device: torch.device,
        ) -> dict[str, object]:
    dataset = backbone.common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    errors: dict[str, list[np.ndarray]] = {"control": [], "candidate": []}
    files = []
    for start in range(0, len(dataset), 64):
        rows = range(start, min(start + 64, len(dataset)))
        samples = [dataset[index] for index in rows]
        x = torch.stack([item[0] for item in samples]).to(device)
        target = torch.stack([item[1] for item in samples]).numpy()
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        for name, model in (("control", control), ("candidate", candidate)):
            model.eval()
            with torch.inference_mode():
                pred_norm = model(x_norm)
                prediction = (
                    pred_norm * normalizer.std_target
                    + normalizer.mean_target).float().cpu().numpy()
            errors[name].append(error_rows(prediction, target))
        files.extend(window_files[index] for index in rows)
    joined = {key: np.concatenate(value) for key, value in errors.items()}
    metrics = {key: score(value) for key, value in joined.items()}
    score_keys = (*POINT_WEIGHTS, "point_contribution_46_10_9")
    delta = {key: float(metrics["candidate"][key]) - float(metrics["control"][key])
             for key in score_keys}
    by_file = {}
    labels = np.asarray(files)
    for name in sorted(set(files)):
        rows = labels == name
        c0, c1 = score(joined["control"][rows]), score(joined["candidate"][rows])
        by_file[name] = {
            "control": c0, "candidate": c1,
            "delta_point": float(c1["point_contribution_46_10_9"])
            - float(c0["point_contribution_46_10_9"]),
        }
    points = [float(value["delta_point"]) for value in by_file.values()]
    return {
        "metrics": metrics, "delta_candidate_minus_control": delta,
        "per_trajectory_file": by_file,
        "positive_trajectory_files": sum(value > 0.0 for value in points),
        "worst_trajectory_delta_point": min(points),
    }


def smoke(device: torch.device) -> None:
    control, candidate, identity = construct_pair(device)
    x = torch.randn(2, 20, 16, 32, 3, device=device)
    for model in (control, candidate):
        y = model(x)
        assert y.shape == (2, 20, 16, 32, 3) and torch.isfinite(y).all()
    print(json.dumps({"smoke": "PASS", "identity": identity}, sort_keys=True))


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != "TEMPORAL_AXIS_LOAO_V1":
        raise PermissionError("formal run requires --execute and protocol acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = git_report()
    if not git["clean"]:
        raise RuntimeError(f"Git worktree must be clean: {git['porcelain']}")
    inventory = scan_inventory(args.sim_dir)
    fit_records = [row for row in inventory if int(row["aoa"]) != args.held_aoa]
    held_records = [row for row in inventory if int(row["aoa"]) == args.held_aoa]
    if len(fit_records) != 80 or len(held_records) != 20:
        raise RuntimeError("whole-AoA split count changed")
    config = {
        "schema": SCHEMA, "held_aoa": args.held_aoa,
        "fit_aoas": [aoa for aoa in AOAS if aoa != args.held_aoa],
        "updates": UPDATES, "batch_size": BATCH_SIZE,
        "seed": SEED, "hidden_channels": HIDDEN_CHANNELS,
        "fit_inventory_sha256": canonical_sha(fit_records),
        "held_inventory_sha256": canonical_sha(held_records),
        "held_field_values_loaded_before_both_endpoint_seals": False,
        "gate": {
            "delta_point_min": .08, "component_floor": -.05,
            "positive_trajectory_files_min": 15,
            "worst_trajectory_delta_point_min": -.05,
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8")
    (args.output_dir / "environment.txt").write_text("".join(
        f"{key}={value}\n" for key, value in {
            "python": sys.version.replace("\n", " "), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device), "pid": os.getpid(),
        }.items()), encoding="utf-8")

    # Only fit-AoA flow fields are loaded here.
    fit_trajectories, fit_windows, _ = load_fields(args.sim_dir, fit_records)
    stats = backbone.common.scratch.window_statistics(fit_trajectories, fit_windows)
    normalizer = backbone.common.scratch.ChannelNormalizer(*stats, device=device)
    fit_dataset = backbone.common.scratch.SimWindowDataset(
        fit_trajectories, fit_windows, mask_probability=0.0, noise_scale=0.0)
    rng = np.random.default_rng(SEED + args.held_aoa)
    schedule = rng.integers(
        0, len(fit_dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    control, candidate, initial = construct_pair(device)
    endpoints = {
        "control": train_arm(
            "control", control, fit_dataset, schedule, normalizer,
            device, args.output_dir),
        "candidate": train_arm(
            "candidate", candidate, fit_dataset, schedule, normalizer,
            device, args.output_dir),
    }
    # Endpoint files/hashes exist before this first held field load.
    held_trajectories, held_windows, held_window_files = load_fields(
        args.sim_dir, held_records)
    evaluation = evaluate_pair(
        control, candidate, held_trajectories, held_windows,
        held_window_files, normalizer, device)
    delta = evaluation["delta_candidate_minus_control"]
    component_pass = all(float(delta[key]) >= -.05 for key in POINT_WEIGHTS)
    passed = (
        float(delta["point_contribution_46_10_9"]) >= .08
        and component_pass
        and int(evaluation["positive_trajectory_files"]) >= 15
        and float(evaluation["worst_trajectory_delta_point"]) >= -.05)
    result = {
        "schema": SCHEMA, "evidence_class": "whole_AoA_held_Sim",
        "config": config, "git": git, "initial_identity": initial,
        "fit_trajectories": len(fit_records), "fit_windows": len(fit_windows),
        "held_trajectories": len(held_records), "held_windows": len(held_windows),
        "sample_stream_sha256_both_arms": schedule_sha,
        "endpoints": endpoints, "evaluation": evaluation,
        "decision": "GO_NEXT_HELD_AOA" if passed else "STOP_TEMPORAL_AXIS",
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"], "held_aoa": args.held_aoa,
        "delta": delta, "positive_files": evaluation["positive_trajectory_files"],
        "worst_file": evaluation["worst_trajectory_delta_point"],
        "result": str(result_path),
    }, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this screen requires CUDA")
    if args.smoke:
        smoke(device)
    else:
        run(args, device)


if __name__ == "__main__":
    main()
