#!/usr/bin/env python3
"""Low-cost paired whole-AoA Sim discovery for B210 unified temporal state.

The candidate and control start from bitwise identical B210 XCB/member
parameters and consume an identical pre-generated fit-only batch stream.  The
sole experimental difference is the axis used by ``history_dynamics_state``:
causal observed-time mixing versus framewise spatial-x control mixing.  Held
flow fields are deliberately unopened until both in-memory endpoints have
been SHA-sealed; no large checkpoint is written and UQ is neither trained nor
executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for _path in (CODE_ROOT, Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig  # noqa: E402


SCHEMA = "b210_unified_temporal_state_sim_discovery_v1"
ACKNOWLEDGEMENT = "B210_UNIFIED_TEMPORAL_STATE_DISCOVERY_V1"
GO_DECISION = "GO_NEXT_HELD_AOA"
STOP_DECISION = "STOP_UNIFIED_TEMPORAL_STATE"
EVIDENCE_CLASS = "whole_AoA_held_Sim"
PAIR_DIFFERENCE = "history_dynamics_state_mode"
SAMPLE_STREAM_FIELD = "sample_stream_sha256_both_arms"
HELD_SEAL_FIELD = "held_field_values_loaded_before_both_endpoint_seals"
AOAS = (0, 5, 10, 15, 20)
HELD_AOAS = (10, 20)
ARMS = ("spatial_control", "causal_temporal")
ARM_MODE = {
    "spatial_control": "framewise_spatial_control",
    "causal_temporal": "causal_temporal",
}
SEED = 210
UPDATES = 2_000
BATCH_SIZE = 8
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
EXPECTED_FILES = 100
EXPECTED_FILES_PER_AOA = 20
EXPECTED_WINDOWS_PER_TRAJECTORY = 48
POINT_WEIGHTS = {
    "rel_l2_score": .46,
    "tke_rel_l2_score": .10,
    "mvpe_rel_l2_score": .09,
}
GATE = {
    "delta_point_min": .03,
    "component_floor": -.03,
    "positive_trajectories_min": 15,
    "worst_trajectory_delta_point_min": -.05,
}


@dataclass(frozen=True)
class SimWindow:
    trajectory: int
    start: int


class SimWindowDataset:
    """Deterministic 20-to-20 Sim windows; no augmentation in this screen."""
    def __init__(self, trajectories: list[torch.Tensor], windows: list[SimWindow]):
        self.trajectories, self.windows = trajectories, windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        data = self.trajectories[window.trajectory][window.start:window.start + 40]
        return data[:20], data[20:]


class ChannelNormalizer:
    def __init__(self, mean_input: torch.Tensor, mean_target: torch.Tensor,
                 std_input: torch.Tensor, std_target: torch.Tensor,
                 device: torch.device):
        self.mean_input = mean_input.to(device)
        self.mean_target = mean_target.to(device)
        self.std_input = std_input.clamp_min(1.0e-8).to(device)
        self.std_target = std_target.clamp_min(1.0e-8).to(device)

    def preprocess(self, history: torch.Tensor, target: torch.Tensor):
        return ((history - self.mean_input) / self.std_input,
                (target - self.mean_target) / self.std_target)


def window_statistics(trajectories: list[torch.Tensor], windows: list[SimWindow]):
    sums = [torch.zeros(3, dtype=torch.float64) for _ in range(2)]
    squares = [torch.zeros(3, dtype=torch.float64) for _ in range(2)]
    counts = [0, 0]
    for window in windows:
        values = trajectories[window.trajectory][window.start:window.start + 40]
        for index, part in enumerate((values[:20], values[20:])):
            part = part.to(torch.float64)
            sums[index] += part.sum(dim=(0, 1, 2))
            squares[index] += part.square().sum(dim=(0, 1, 2))
            counts[index] += part.numel() // 3
    means = [sums[index] / counts[index] for index in range(2)]
    stds = [(squares[index] / counts[index] - means[index].square()).clamp_min(0).sqrt()
            for index in range(2)]
    return tuple(value.float() for value in (*means, *stds))


def accuracy_objective(prediction: torch.Tensor, target: torch.Tensor,
                       normalizer: ChannelNormalizer,
                       weights: tuple[float, float, float]):
    """The mature differentiable point accuracy objective (1, 4, 1 here)."""
    pred = (prediction * normalizer.std_target + normalizer.mean_target)[..., :2]
    truth = (target * normalizer.std_target + normalizer.mean_target)[..., :2]
    batch = pred.shape[0]
    rel = torch.linalg.vector_norm((pred - truth).reshape(batch, -1), dim=1) / torch.linalg.vector_norm(
        truth.reshape(batch, -1), dim=1).clamp_min(1.0e-8)
    pred_tke = .5 * (pred - pred.mean(dim=1, keepdim=True)).square().mean(dim=1).sum(dim=-1)
    truth_tke = .5 * (truth - truth.mean(dim=1, keepdim=True)).square().mean(dim=1).sum(dim=-1)
    tke = torch.linalg.vector_norm((pred_tke - truth_tke).reshape(batch, -1), dim=1) / torch.linalg.vector_norm(
        truth_tke.reshape(batch, -1), dim=1).clamp_min(1.0e-8)
    center = pred.shape[2] // 2
    ys = [value for value in (center + 2 * offset for offset in range(-4, 5))
          if 0 <= value < pred.shape[2]]
    profiles = []
    for index in range(4):
        x = ((index + 1) * 16 + 10) // 2
        if x < pred.shape[3]:
            profiles.append(torch.linalg.vector_norm(
                (pred[:, :, ys, x].mean(dim=1) - truth[:, :, ys, x].mean(dim=1)).reshape(batch, -1), dim=1)
                / torch.linalg.vector_norm(truth[:, :, ys, x].mean(dim=1).reshape(batch, -1), dim=1).clamp_min(1.0e-8))
    mvpe = torch.stack(profiles).mean(dim=0)
    errors = torch.stack((rel, tke, mvpe), dim=1)
    score = 1.0 / (1.0 + .5 * errors)
    weight = score.new_tensor(weights)
    loss = 1.0 - (score * weight).sum(dim=1) / weight.sum()
    return loss.mean(), {"rel_l2": rel.mean().detach(), "tke_rel_l2": tke.mean().detach(),
                         "mvpe_rel_l2": mvpe.mean().detach(), "accuracy_objective": loss.mean().detach()}


def load_scoring():
    path = REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
    spec = importlib.util.spec_from_file_location("b210_unified_state_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORING = load_scoring()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--held-aoa", type=int, choices=HELD_AOAS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_sha(state: Mapping[str, torch.Tensor]) -> str:
    """Stable CPU state digest without serialising or retaining a checkpoint."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def git_report() -> dict[str, object]:
    def run(*values: str) -> str:
        return subprocess.run(
            ["git", *values], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
    porcelain = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "clean": not bool(porcelain),
        "porcelain": porcelain.splitlines(),
    }


def inventory(sim_dir: Path) -> list[dict[str, object]]:
    """Read metadata only; this never opens a u/v/p flow-field value."""
    try:
        import h5py
    except ImportError as error:  # Keep model-only smoke/unit tests lightweight.
        raise RuntimeError("HDF5 Sim inventory requires h5py") from error
    rows = []
    for path in sorted(sim_dir.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            for key in ("u", "v", "p", "aoa", "re"):
                if key not in handle:
                    raise ValueError(f"{path.name} misses {key}")
            shapes = [tuple(handle[key].shape) for key in ("u", "v", "p")]
            if not (shapes[0] == shapes[1] == shapes[2]):
                raise ValueError(f"field shapes differ in {path.name}")
            rows.append({
                "file": path.name,
                "aoa": int(np.asarray(handle["aoa"]).reshape(-1)[0]),
                "re": int(np.asarray(handle["re"]).reshape(-1)[0]),
                "frames": int(shapes[0][0]),
                "shape": list(shapes[0]),
            })
    if len(rows) != EXPECTED_FILES:
        raise ValueError(f"expected {EXPECTED_FILES} Sim files, got {len(rows)}")
    counts = {aoa: sum(int(row["aoa"]) == aoa for row in rows) for aoa in AOAS}
    if any(counts[aoa] != EXPECTED_FILES_PER_AOA for aoa in AOAS):
        raise ValueError(f"unexpected AoA inventory {counts}")
    return rows


def load_fields(sim_dir: Path, rows: Iterable[Mapping[str, object]]):
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("HDF5 Sim field loading requires h5py") from error
    trajectories: list[torch.Tensor] = []
    windows = []
    window_files: list[str] = []
    for trajectory_index, row in enumerate(rows):
        path = sim_dir / str(row["file"])
        with h5py.File(path, "r") as handle:
            channels = [np.asarray(handle[key][:, ::2, ::2], dtype=np.float32)
                        for key in ("u", "v", "p")]
        trajectory = torch.from_numpy(np.ascontiguousarray(np.stack(channels, axis=-1)))
        if len(trajectory) != int(row["frames"]):
            raise ValueError(f"frame count changed for {path.name}")
        trajectories.append(trajectory)
        for start in range(0, len(trajectory) - 40, 20):
            windows.append(SimWindow(trajectory_index, start))
            window_files.append(path.name)
    return trajectories, windows, window_files


def model_config(mode: str, *, tiny: bool = False) -> EnsembleUNet2dConfig:
    if mode not in ARM_MODE.values():
        raise ValueError(f"unknown state mixer mode {mode!r}")
    # This is the current mature B210 point/member topology.  UQ modules are
    # present as topology, but this harness calls only forward() and its point
    # loss; no UQ objective or UQ forward path is used.
    values = dict(
        base=210, depth=4, groups=8, difference_steps=5,
        acceleration_steps=0, use_coordinates=True, use_regime_film=True,
        multi_scale_regime_film=True, cross_conditioned_backbone=True,
        cross_conditioned_steps=5, cross_conditioned_channels=64,
        mean_fluctuation=True, ensemble_members=4, uq_hidden_channels=160,
        member_adapter_channels=64, mean_profile_channels=104,
        uq_flow_context=True, residual_quantile_intervals=True,
        utility_refinement=True, utility_refinement_kernel=3,
        utility_refinement_blocks=0, temporal_uq_refinement=True,
        temporal_uq_channels=24, temporal_uq_blocks=2, temporal_uq_kernel=3,
        coverage_calibration=False, disagreement_scale_fraction=.01,
        physical_energy_controller=True, physical_energy_member_sync=True,
        physical_energy_multiscale=False, physical_energy_horizon_local=False,
        physical_energy_history_conditioned=False, physical_energy_history_steps=5,
        physical_energy_hidden_channels=32, physical_energy_log_limit=1.0,
        phase_delta_hidden_channels=0, phase_delta_steps=5,
        backbone_residual_dropout=.20,
        hard_region_expert_feature_noise_std=0.0,
        history_dynamics_state_mode=mode, history_dynamics_state_channels=32,
        history_dynamics_state_blocks=2, history_dynamics_state_kernel=5,
    )
    if tiny:
        # Unit-test-only construct/train configuration; all paired semantics,
        # including the active mixer and standard control flags, are retained.
        values.update(
            base=8, depth=2, groups=4, cross_conditioned_channels=8,
            uq_hidden_channels=8, member_adapter_channels=8,
            mean_profile_channels=8, temporal_uq_channels=8,
            physical_energy_hidden_channels=8,
            history_dynamics_state_channels=8)
    return EnsembleUNet2dConfig(**values)


def model_metadata() -> dict[str, object]:
    """Describe the active architecture without affecting legacy execution."""
    return {
        "base": 210, "depth": 4, "difference_steps": 5,
        "backbone_residual_dropout": .20,
        "multi_scale_regime_film": True, "cross_conditioned": [64, 5],
        "member_adapter_channels": 64,
        "history_dynamics_state": [32, 2, 5],
    }


def construct_pair(device: torch.device, *, tiny: bool = False):
    models: dict[str, EnsembleUNet2d] = {}
    starts: dict[str, dict[str, torch.Tensor]] = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        model = EnsembleUNet2d(model_config(ARM_MODE[arm], tiny=tiny)).float().to(device)
        models[arm] = model
        starts[arm] = state_snapshot(model)
    first, second = starts[ARMS[0]], starts[ARMS[1]]
    equal = first.keys() == second.keys() and all(
        torch.equal(first[key], second[key]) for key in first)
    if not equal:
        raise RuntimeError("paired state mixer arms are not bitwise identical")
    count = sum(parameter.numel() for parameter in models[ARMS[0]].parameters())
    if count != sum(parameter.numel() for parameter in models[ARMS[1]].parameters()):
        raise RuntimeError("paired state mixer parameter counts differ")
    identity = {
        "state_sha256": state_sha(first), "parameters": count,
        "state_tensors": len(first), "bitwise_identical": True,
        "only_config_difference": PAIR_DIFFERENCE,
    }
    return models, identity


def batch_stream(dataset, schedule: np.ndarray):
    for row in schedule:
        samples = [dataset[int(index)] for index in row]
        yield torch.stack([item[0] for item in samples]), torch.stack([item[1] for item in samples])


def set_pair_normalizer_statistics(
        models: Mapping[str, EnsembleUNet2d],
        normalizer: ChannelNormalizer) -> dict[str, object]:
    """Install the fit-only physical persistence map identically in both arms."""
    values = (
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target,
    )
    for model in models.values():
        model.set_normalizer_statistics(*values)
    buffer_names = ("persistence_scale", "persistence_bias", "target_std")
    control = models[ARMS[0]]
    candidate = models[ARMS[1]]
    if not all(torch.equal(getattr(control, name), getattr(candidate, name))
               for name in buffer_names):
        raise RuntimeError("paired normalizer-dependent buffers differ")
    state = {name: getattr(control, name) for name in buffer_names}
    return {
        "state_sha256": state_sha(state),
        **{name: getattr(control, name).detach().cpu().tolist()
           for name in buffer_names},
    }


def train_arm(name: str, model: EnsembleUNet2d, dataset, schedule: np.ndarray,
              normalizer, device: torch.device, updates: int) -> dict[str, object]:
    # Reset the stochastic backbone stream so both sequential arms see the
    # same dropout masks as well as the same pre-generated samples.
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    trace = []
    interval: list[float] = []
    started = time.perf_counter()
    for update, (history, target) in enumerate(batch_stream(dataset, schedule), 1):
        model.train()
        history, target = history.to(device), target.to(device)
        history, target = normalizer.preprocess(history, target)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)  # Deliberately point-only: no forward_uq_training.
        accuracy, details = accuracy_objective(
            prediction, target, normalizer, (1.0, 4.0, 1.0))
        mse = (prediction - target).square().mean()
        loss = accuracy + .25 * mse
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} non-finite loss at update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0 or update == updates:
            row = {
                "update": update, "loss_mean": float(np.mean(interval)),
                "mse": float(mse.detach()),
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({"arm": name, **row,
                              "wall_seconds": time.perf_counter() - started},
                             sort_keys=True), flush=True)
            interval.clear()
    endpoint_sha = state_sha(state_snapshot(model))
    return {"endpoint_state_sha256": endpoint_sha, "trace": trace,
            "wall_seconds": time.perf_counter() - started}


def error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((
        SCORING.rel_l2_per_sample(prediction, target, 2),
        SCORING.tke_rel_l2_per_sample(prediction, target, 2),
        SCORING.mvpe_rel_l2_per_sample(prediction, target),
    ), axis=1).astype(np.float64)


def score(errors: np.ndarray) -> dict[str, float | int]:
    means = errors.mean(axis=0, dtype=np.float64)
    result: dict[str, float | int] = {
        "samples": int(len(errors)),
        "rel_l2_score": float(SCORING.score_error(float(means[0]))),
        "tke_rel_l2_score": float(SCORING.score_error(float(means[1]))),
        "mvpe_rel_l2_score": float(SCORING.score_error(float(means[2]))),
    }
    result["point_contribution_46_10_9"] = sum(
        POINT_WEIGHTS[key] * float(result[key]) for key in POINT_WEIGHTS)
    return result


def comparison(candidate: Mapping[str, float | int], control: Mapping[str, float | int]):
    keys = (*POINT_WEIGHTS, "point_contribution_46_10_9")
    return {key: float(candidate[key]) - float(control[key]) for key in keys}


def evaluate_pair(models: Mapping[str, EnsembleUNet2d], trajectories, windows,
                  window_files: list[str], normalizer, device: torch.device) -> dict[str, object]:
    file_counts = {
        name: window_files.count(name) for name in sorted(set(window_files))}
    if (len(file_counts) != EXPECTED_FILES_PER_AOA
            or set(file_counts.values()) != {EXPECTED_WINDOWS_PER_TRAJECTORY}):
        raise RuntimeError(
            "held evaluation is not a complete 20-trajectory/48-window panel: "
            f"{file_counts}")
    dataset = SimWindowDataset(trajectories, windows)
    errors = {arm: [] for arm in ARMS}
    labels: list[str] = []
    for start in range(0, len(dataset), 16):
        indices = range(start, min(start + 16, len(dataset)))
        samples = [dataset[index] for index in indices]
        history = torch.stack([item[0] for item in samples]).to(device)
        target = torch.stack([item[1] for item in samples]).numpy()
        history = (history - normalizer.mean_input) / normalizer.std_input
        for arm in ARMS:
            models[arm].eval()
            with torch.inference_mode():
                prediction = models[arm](history)
                prediction = (prediction * normalizer.std_target + normalizer.mean_target)
            errors[arm].append(error_rows(prediction.float().cpu().numpy(), target))
        labels.extend(window_files[index] for index in indices)
    joined = {arm: np.concatenate(parts) for arm, parts in errors.items()}
    aggregate = {arm: score(joined[arm]) for arm in ARMS}
    per_trajectory = {}
    label_array = np.asarray(labels)
    for file_name in sorted(set(labels)):
        selected = label_array == file_name
        control, candidate = score(joined["spatial_control"][selected]), score(joined["causal_temporal"][selected])
        per_trajectory[file_name] = {
            "spatial_control": control, "causal_temporal": candidate,
            "delta_temporal_minus_spatial": comparison(candidate, control),
        }
    deltas = [float(row["delta_temporal_minus_spatial"]["point_contribution_46_10_9"])
              for row in per_trajectory.values()]
    return {
        "aggregate": aggregate,
        "delta_temporal_minus_spatial": comparison(
            aggregate["causal_temporal"], aggregate["spatial_control"]),
        "per_trajectory": per_trajectory,
        "trajectory_count": len(per_trajectory),
        "windows_per_trajectory": EXPECTED_WINDOWS_PER_TRAJECTORY,
        "positive_trajectories": sum(value > 0.0 for value in deltas),
        "worst_trajectory_delta_point": min(deltas),
    }


def decide_gate(evaluation: Mapping[str, object]) -> dict[str, object]:
    delta = evaluation["delta_temporal_minus_spatial"]
    if not isinstance(delta, Mapping):
        raise ValueError("evaluation has no aggregate delta")
    component_pass = all(float(delta[key]) >= GATE["component_floor"] for key in POINT_WEIGHTS)
    complete_panel = (
        int(evaluation.get("trajectory_count", -1)) == EXPECTED_FILES_PER_AOA
        and int(evaluation.get("windows_per_trajectory", -1))
        == EXPECTED_WINDOWS_PER_TRAJECTORY)
    passed = (
        float(delta["point_contribution_46_10_9"]) >= GATE["delta_point_min"]
        and component_pass
        and complete_panel
        and int(evaluation["positive_trajectories"]) >= GATE["positive_trajectories_min"]
        and float(evaluation["worst_trajectory_delta_point"])
        >= GATE["worst_trajectory_delta_point_min"])
    return {**GATE, "component_pass": component_pass,
            "complete_panel": complete_panel, "pass": passed}


def smoke(device: torch.device) -> dict[str, object]:
    models, identity = construct_pair(device, tiny=True)
    normalizer = ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    normalizer_buffers = set_pair_normalizer_statistics(models, normalizer)
    trajectory = torch.randn(40, 16, 16, 3)
    dataset = SimWindowDataset([trajectory], [SimWindow(0, 0)])
    schedule = np.zeros((2, 1), dtype=np.int64)
    endpoints = {arm: train_arm(arm, models[arm], dataset, schedule, normalizer, device, 2)
                 for arm in ARMS}
    return {"smoke": "PASS", "identity": identity,
            "normalizer_buffers": normalizer_buffers, "endpoints": endpoints}


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge_protocol != ACKNOWLEDGEMENT:
        raise PermissionError("formal discovery requires --execute and acknowledgement")
    if args.sim_dir is None or args.output_dir is None or args.held_aoa is None:
        raise ValueError("formal discovery requires --sim-dir, --output-dir, and --held-aoa")
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("updates and batch-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty output directory")
    git = git_report()
    if not git["clean"]:
        raise RuntimeError(f"Git worktree must be clean: {git['porcelain']}")
    rows = inventory(args.sim_dir)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    if len(fit_rows) != 80 or len(held_rows) != 20:
        raise RuntimeError("whole-AoA Sim split count changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fit flow fields only.  This normalizer is never exposed to the held AoA.
    fit_trajectories, fit_windows, _unused = load_fields(args.sim_dir, fit_rows)
    statistics = window_statistics(fit_trajectories, fit_windows)
    normalizer = ChannelNormalizer(*statistics, device=device)
    fit_dataset = SimWindowDataset(fit_trajectories, fit_windows)
    rng = np.random.default_rng(SEED + int(args.held_aoa))
    schedule = rng.integers(0, len(fit_dataset), size=(args.updates, args.batch_size), dtype=np.int64)
    models, initial_identity = construct_pair(device)
    normalizer_buffers = set_pair_normalizer_statistics(models, normalizer)
    endpoints = {arm: train_arm(arm, models[arm], fit_dataset, schedule, normalizer, device, args.updates)
                 for arm in ARMS}

    # This is intentionally the first u/v/p held-field load, after both
    # endpoint SHA seals exist in memory.  No checkpoint is materialised.
    held_trajectories, held_windows, held_window_files = load_fields(args.sim_dir, held_rows)
    evaluation = evaluate_pair(models, held_trajectories, held_windows,
                               held_window_files, normalizer, device)
    gate = decide_gate(evaluation)
    result = {
        "schema": SCHEMA, "evidence_class": EVIDENCE_CLASS,
        "held_aoa": args.held_aoa,
        "fit_aoas": [aoa for aoa in AOAS if aoa != args.held_aoa],
        "git": git,
        "training": {"updates": args.updates, "batch_size": args.batch_size,
                     "seed": SEED, "point_loss": "accuracy_1_4_1_plus_.25_mse",
                     "paired_dropout_rng_reset": True,
                     "uq_executed": False},
        "model": model_metadata(),
        "fit_inventory_sha256": canonical_sha(fit_rows),
        "held_inventory_sha256": canonical_sha(held_rows),
        "fit_normalizer_sha256": state_sha({str(index): value for index, value in enumerate(statistics)}),
        HELD_SEAL_FIELD: False,
        "initial_identity": initial_identity,
        "normalizer_dependent_buffers": normalizer_buffers,
        SAMPLE_STREAM_FIELD: hashlib.sha256(schedule.tobytes()).hexdigest(),
        "endpoints": endpoints, "evaluation": evaluation, "gate": gate,
        "decision": GO_DECISION if gate["pass"] else STOP_DECISION,
    }
    path = args.output_dir / "result.json"
    path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    primary_delta = evaluation.get(
        "delta_temporal_minus_strongest_reference",
        evaluation["delta_temporal_minus_spatial"])
    print(json.dumps({
        "decision": result["decision"],
        "held_aoa": args.held_aoa,
        "delta": primary_delta,
        "strongest_reference_arm": evaluation.get(
            "strongest_reference_arm", "spatial_control"),
        "result": str(path),
    }, sort_keys=True), flush=True)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.smoke:
        print(json.dumps(smoke(device), sort_keys=True), flush=True)
    else:
        run(args, device)


if __name__ == "__main__":
    main()
