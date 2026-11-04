#!/usr/bin/env python3
"""Strict whole-AoA Sim discovery screen for gap-conditioned forecasting.

Three *identical* small temporal-transport models are trained from the same
initial tensors and base history stream:

* ``fixed_lead0``: the released task, history ``[s,s+20)`` -> target
  ``[s+20,s+40)``;
* ``random_lead_blind``: the same histories but targets separated by a random
  released-data gap, without telling the model that gap;
* ``random_lead_conditioned``: exactly the blind arm's examples, additionally
  given the gap through a zero-preserving decoder condition.

The held AoA flow arrays are deliberately not read until all three endpoint
files and their hashes are sealed.  A held-20 execution additionally requires
a recorded held-10 GO, preventing informal selection across held conditions.
This is a discovery screen only: it does not manufacture trajectories and it
evaluates every arm on the ordinary 20->20 (gap zero) competition task.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
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
from models.gap_conditioned_temporal_transport import (  # noqa: E402
    GapConditionedTemporalTransportOperator,
)
from models.temporal_transport import TemporalTransportConfig  # noqa: E402


SCHEMA = "gap_conditioned_multilead_whole_aoa_sim_screen_v1"
AOAS = (0, 5, 10, 15, 20)
SEED = 210
UPDATES = 2_000
BATCH_SIZE = 8
HIDDEN_CHANNELS = 128
MAX_GAP_STEPS = 20
GAP_CHOICES = (0, 10, 20)
ARM_NAMES = (
    "fixed_lead0", "random_lead_blind", "random_lead_conditioned")
EXPECTED_FILES = 100
EXPECTED_FILES_PER_AOA = 20
EXPECTED_PARAMETERS = 1_343_496
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}


def load_scoring():
    path = REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
    spec = importlib.util.spec_from_file_location("multilead_loao_scoring", path)
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
    parser.add_argument("--held-aoa", choices=(10, 20), type=int, required=True)
    parser.add_argument("--prior-held10-result", type=Path)
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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
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
    """Read metadata/shapes only; never materialize a flow-field array."""
    records: list[dict[str, object]] = []
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
    files: list[str] = []
    for trajectory_index, record in enumerate(records):
        path = sim_dir / str(record["file"])
        with h5py.File(path, "r") as handle:
            channels = [np.asarray(handle[key][:, ::2, ::2], dtype=np.float32)
                        for key in ("u", "v", "p")]
        trajectory = torch.from_numpy(np.ascontiguousarray(
            np.stack(channels, axis=-1)))
        if len(trajectory) != int(record["frames"]):
            raise ValueError(f"frame count changed for {path.name}")
        if len(trajectory) < 40 + MAX_GAP_STEPS:
            raise ValueError(f"trajectory is too short for maximum gap: {path.name}")
        trajectories.append(trajectory)
        # This matches the official Sim window convention used by prior
        # screens; the maximum standard start is still valid for gap=20.
        for start in range(0, len(trajectory) - 40, 20):
            windows.append(backbone.common.scratch.SimWindow(trajectory_index, start))
            files.append(path.name)
    return trajectories, windows, files


def make_config() -> TemporalTransportConfig:
    return TemporalTransportConfig(
        hidden_channels=HIDDEN_CHANNELS, spatial_blocks=2, spatial_downsample=2,
        temporal_blocks=3, temporal_kernel=5, mixer_mode="causal_temporal",
        groups=8, use_coordinates=True, max_displacement=4.0,
        residual_limit=4.0, output_initialization_std=1e-3,
        predict_uncertainty=False)


def construct_arms(device: torch.device) -> tuple[
        dict[str, GapConditionedTemporalTransportOperator], dict[str, object]]:
    arms: dict[str, GapConditionedTemporalTransportOperator] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = GapConditionedTemporalTransportOperator(
            make_config(), max_gap_steps=MAX_GAP_STEPS).to(device)
        arms[name] = model
        states[name] = {key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()}
    reference = states[ARM_NAMES[0]]
    if any(reference.keys() != state.keys() or any(
            not torch.equal(reference[key], state[key]) for key in reference)
            for state in states.values()):
        raise RuntimeError("three arms do not share a bitwise initial state")
    parameters = sum(parameter.numel() for parameter in arms[ARM_NAMES[0]].parameters())
    if any(sum(parameter.numel() for parameter in model.parameters()) != parameters
           for model in arms.values()):
        raise RuntimeError("arm parameter counts differ")
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"gap-conditioned parameter contract changed: {parameters}")
    return arms, {
        "state_sha256": state_sha(reference), "parameters": parameters,
        "state_tensors": len(reference), "bitwise_identical": True,
    }


def build_schedule(
        windows: list[object], updates: int, batch_size: int,
        rng: np.random.Generator) -> np.ndarray:
    """Return [update,batch,(trajectory,start,random_gap)] int rows."""
    source = rng.integers(0, len(windows), size=(updates, batch_size))
    gaps = rng.choice(np.asarray(GAP_CHOICES, dtype=np.int64),
                      size=(updates, batch_size), replace=True)
    result = np.empty((updates, batch_size, 3), dtype=np.int64)
    for update in range(updates):
        for sample in range(batch_size):
            window = windows[int(source[update, sample])]
            result[update, sample] = (
                int(window.trajectory), int(window.start),
                int(gaps[update, sample]))
    return result


def materialize_batch(
        trajectories: list[torch.Tensor], rows: np.ndarray,
        *, random_lead: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    histories, targets, gaps = [], [], []
    for trajectory_index, start, sampled_gap in rows:
        trajectory = trajectories[int(trajectory_index)]
        gap = int(sampled_gap) if random_lead else 0
        histories.append(trajectory[int(start):int(start) + 20])
        targets.append(trajectory[
            int(start) + 20 + gap:int(start) + 40 + gap])
        gaps.append(gap)
    history = torch.stack(histories)
    target = torch.stack(targets)
    gap_tensor = torch.tensor(gaps, dtype=torch.long)
    if history.shape[1] != 20 or target.shape[1] != 20:
        raise RuntimeError("invalid temporal slice materialized")
    return history, target, gap_tensor


def arm_policy(name: str) -> tuple[bool, bool]:
    """Return ``(random_target_lead, reveal_lead_to_decoder)``."""
    if name == "fixed_lead0":
        return False, False
    if name == "random_lead_blind":
        return True, False
    if name == "random_lead_conditioned":
        return True, True
    raise ValueError(f"unknown arm {name!r}")


def train_arm(
        name: str, model: GapConditionedTemporalTransportOperator,
        trajectories: list[torch.Tensor], schedule: np.ndarray, normalizer,
        device: torch.device, output_dir: Path) -> dict[str, object]:
    random_lead, reveal_lead = arm_policy(name)
    backbone.common.set_model_stats(model, normalizer)
    # Re-seeding catches accidental future stochastic additions (e.g. dropout)
    # and keeps those additions paired across the three arms.
    torch.manual_seed(SEED + 17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 17)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    trace: list[dict[str, object]] = []
    interval: list[float] = []
    started = time.perf_counter()
    for update, rows in enumerate(schedule, 1):
        x, y, gaps = materialize_batch(
            trajectories, rows, random_lead=random_lead)
        model.train()
        x, y = x.to(device), y.to(device)
        x_norm, y_norm = normalizer.preprocess(x, y)
        condition = gaps.to(device) if reveal_lead else None
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_norm, condition)
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
    torch.save({"schema": SCHEMA, "arm": name, "updates": UPDATES,
                "model_config": model.architecture_config(),
                "model_state_dict": state}, endpoint)
    return {"path": str(endpoint), "sha256": sha256_file(endpoint),
            "model_state_sha256": state_sha(state), "trace": trace,
            "wall_seconds": time.perf_counter() - started}


def error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((
        SCORING.rel_l2_per_sample(prediction, target, 2),
        SCORING.tke_rel_l2_per_sample(prediction, target, 2),
        SCORING.mvpe_rel_l2_per_sample(prediction, target)), axis=1).astype(np.float64)


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


def evaluate_arms(
        arms: dict[str, GapConditionedTemporalTransportOperator], trajectories,
        windows, window_files: list[str], normalizer, device: torch.device,
        ) -> dict[str, object]:
    dataset = backbone.common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    errors: dict[str, list[np.ndarray]] = {name: [] for name in ARM_NAMES}
    files: list[str] = []
    for start in range(0, len(dataset), 64):
        indexes = range(start, min(start + 64, len(dataset)))
        samples = [dataset[index] for index in indexes]
        x = torch.stack([item[0] for item in samples]).to(device)
        target = torch.stack([item[1] for item in samples]).numpy()
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        for name, model in arms.items():
            model.eval()
            with torch.inference_mode():
                # gap=None is exactly the zero-gap competition deployment path.
                pred_norm = model(x_norm, None)
                prediction = (pred_norm * normalizer.std_target
                              + normalizer.mean_target).float().cpu().numpy()
            errors[name].append(error_rows(prediction, target))
        files.extend(window_files[index] for index in indexes)
    joined = {name: np.concatenate(value) for name, value in errors.items()}
    metrics = {name: score(value) for name, value in joined.items()}
    comparisons: dict[str, object] = {}
    labels = np.asarray(files)
    for baseline in ("fixed_lead0", "random_lead_blind"):
        delta = {key: float(metrics["random_lead_conditioned"][key])
                 - float(metrics[baseline][key])
                 for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}
        per_file = {}
        for file in sorted(set(files)):
            rows = labels == file
            base_score = score(joined[baseline][rows])
            cond_score = score(joined["random_lead_conditioned"][rows])
            per_file[file] = float(cond_score["point_contribution_46_10_9"]
                                  - base_score["point_contribution_46_10_9"])
        values = list(per_file.values())
        comparisons[f"conditioned_minus_{baseline}"] = {
            "delta": delta, "per_trajectory_file_delta_point": per_file,
            "positive_trajectory_files": sum(value > 0.0 for value in values),
            "worst_trajectory_delta_point": min(values),
        }
    return {"metrics": metrics, "conditioned_comparisons": comparisons}


def prior_held10_evidence(path: Path | None, held_aoa: int) -> dict[str, object] | None:
    if held_aoa == 10:
        if path is not None:
            raise ValueError("held10 does not take a prior held10 result")
        return None
    if path is None or not path.is_file():
        raise PermissionError("held20 requires --prior-held10-result from this screen")
    prior = json.loads(path.read_text(encoding="utf-8"))
    if (prior.get("schema") != SCHEMA
            or prior.get("config", {}).get("held_aoa") != 10
            or prior.get("decision") != "GO_HELD20"):
        raise PermissionError("prior file is not a held10 GO from this screen")
    return {"path": str(path), "sha256": sha256_file(path),
            "decision": prior["decision"]}


def gate(evaluation: dict[str, object]) -> tuple[bool, dict[str, object]]:
    fixed = evaluation["conditioned_comparisons"]["conditioned_minus_fixed_lead0"]
    fixed_delta = fixed["delta"]
    passed = (
        float(fixed_delta["point_contribution_46_10_9"]) >= .05
        and all(float(fixed_delta[key]) >= -.03 for key in POINT_WEIGHTS)
        and int(fixed["positive_trajectory_files"]) >= 15
        and float(fixed["worst_trajectory_delta_point"]) >= -.05)
    return passed, {
        "conditioned_minus_fixed_point_min": .05,
        "conditioned_minus_blind_is_diagnostic_only": True,
        "component_floor": -.03, "positive_trajectory_files_min": 15,
        "worst_trajectory_delta_point_min": -.05,
    }


def smoke(device: torch.device) -> None:
    arms, identity = construct_arms(device)
    x = torch.randn(2, 20, 16, 32, 3, device=device)
    zero = torch.zeros(2, dtype=torch.long, device=device)
    for model in arms.values():
        model.eval()
        with torch.inference_mode():
            ordinary = model(x)
            explicit_zero = model(x, zero)
            nonzero = model(x, torch.tensor([5, 20], device=device))
        assert ordinary.shape == (2, 20, 16, 32, 3)
        torch.testing.assert_close(ordinary, explicit_zero, rtol=0.0, atol=0.0)
        assert torch.isfinite(nonzero).all()
    print(json.dumps({"smoke": "PASS", "identity": identity}, sort_keys=True))


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != "GAP_MULTILEAD_LOAO_V1":
        raise PermissionError("formal run requires --execute and protocol acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    prior = prior_held10_evidence(args.prior_held10_result, args.held_aoa)
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
        "updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED,
        "hidden_channels": HIDDEN_CHANNELS, "max_gap_steps": MAX_GAP_STEPS,
        "gap_choices": list(GAP_CHOICES), "arms": list(ARM_NAMES),
        "fit_inventory_sha256": canonical_sha(fit_records),
        "held_inventory_sha256": canonical_sha(held_records),
        "held_field_values_loaded_before_all_endpoint_seals": False,
        "prior_held10_evidence": prior,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n",
        encoding="utf-8")
    (args.output_dir / "environment.txt").write_text("".join(
        f"{key}={value}\n" for key, value in {
            "python": sys.version.replace("\n", " "), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device), "pid": os.getpid(),
        }.items()), encoding="utf-8")

    # Fit values only: stats, examples, and all optimisation happen here.
    fit_trajectories, fit_windows, _ = load_fields(args.sim_dir, fit_records)
    if len(fit_windows) != 80 * 48:
        raise RuntimeError(f"expected 3840 fit windows, got {len(fit_windows)}")
    stats = backbone.common.scratch.window_statistics(fit_trajectories, fit_windows)
    normalizer = backbone.common.scratch.ChannelNormalizer(*stats, device=device)
    rng = np.random.default_rng(SEED + args.held_aoa)
    schedule = build_schedule(fit_windows, UPDATES, BATCH_SIZE, rng)
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    gap_histogram = {str(gap): int((schedule[..., 2] == gap).sum())
                     for gap in GAP_CHOICES}
    arms, initial = construct_arms(device)
    endpoints = {name: train_arm(name, arms[name], fit_trajectories, schedule,
                                 normalizer, device, args.output_dir)
                 for name in ARM_NAMES}
    seal = {"schema": SCHEMA, "endpoints": {
        name: {key: endpoints[name][key] for key in ("path", "sha256", "model_state_sha256")}
        for name in ARM_NAMES}}
    seal_path = args.output_dir / "all_endpoints_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    seal_sha = sha256_file(seal_path)

    # First held field array read occurs only after all endpoint seals above.
    held_trajectories, held_windows, held_files = load_fields(args.sim_dir, held_records)
    if len(held_windows) != 20 * 48:
        raise RuntimeError(f"expected 960 held windows, got {len(held_windows)}")
    evaluation = evaluate_arms(arms, held_trajectories, held_windows, held_files,
                               normalizer, device)
    passed, gate_config = gate(evaluation)
    decision = "GO_HELD20" if args.held_aoa == 10 and passed else (
        "COMPLETE_PROMISING" if passed else "STOP_MULTILEAD")
    result = {
        "schema": SCHEMA, "evidence_class": "whole_AoA_held_Sim",
        "config": config, "git": git, "initial_identity": initial,
        "fit_trajectories": len(fit_records), "fit_windows": len(fit_windows),
        "held_trajectories": len(held_records), "held_windows": len(held_windows),
        "sample_stream_sha256_all_arms": schedule_sha,
        "random_gap_histogram_all_arms": gap_histogram, "endpoints": endpoints,
        "endpoint_seal": {"path": str(seal_path), "sha256": seal_sha},
        "evaluation": evaluation, "gate": gate_config, "decision": decision,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")
    fixed = evaluation["conditioned_comparisons"]["conditioned_minus_fixed_lead0"]
    blind = evaluation["conditioned_comparisons"]["conditioned_minus_random_lead_blind"]
    print(json.dumps({"decision": decision, "held_aoa": args.held_aoa,
        "conditioned_minus_fixed": fixed["delta"],
        "conditioned_minus_blind": blind["delta"], "result": str(result_path)},
        sort_keys=True), flush=True)


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
