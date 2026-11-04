#!/usr/bin/env python3
"""Falsifiable direct-20 future-refiner screen on a whole-AoA Sim split.

This is deliberately a *small* target-reframing test.  A causal temporal
transport base is trained from scratch on fit AoAs for 2,000 updates, then
frozen and sealed.  Two bitwise-identical 3-D residual refiners are trained
for the same 2,000 updates over that frozen base prediction:

* both arms receive a full-weight direct ``z=p0 -> y`` loss at every update;
* ``control`` has a matched second direct pass, while ``future_noise`` uses
  its matched auxiliary pass for ``z=y+sigma*epsilon -> y``.

The only deployment operation is one candidate correction from ``p0`` with
noise level one (R=1), denoting one fit-residual RMS.  The noise scale is estimated from *fit-only* frozen
base residuals.  Held field values are not materialized until the frozen base
and both refiner endpoints have been sealed.  This is discovery infrastructure
only, never a submission training path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (CODE_ROOT, Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_flow_backbone as backbone  # noqa: E402
import train_gap_conditioned_multilead_loao_screen as common  # noqa: E402
from models.direct_future_refiner import (  # noqa: E402
    DirectFutureRefiner,
    DirectFutureRefinerConfig,
)
from models.temporal_transport import (  # noqa: E402
    TemporalTransportConfig,
    TemporalTransportOperator,
)


SCHEMA = "direct_future_refiner_whole_aoa_sim_screen_v1"
AOAS = (0, 5, 10, 15, 20)
SEED = 210
NOISE_SEED = SEED + 5_000
UPDATES = 2_000
BATCH_SIZE = 8
BASE_HIDDEN_CHANNELS = 128
REFINER_HIDDEN_CHANNELS = 32
REFINER_BLOCKS = 3
NOISE_LEVEL_MIN = 0.25
NOISE_LEVEL_MAX = 1.00
AUXILIARY_WEIGHT = 0.25
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}
ARM_NAMES = ("control", "future_noise")
EXPECTED_BASE_PARAMETERS = 1_343_112
EXPECTED_REFINER_PARAMETERS = 9_155


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-aoa", type=int, choices=(10, 20), required=True)
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


def state_sha(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_normalizer(normalizer, output_dir: Path) -> dict[str, object]:
    """Seal every statistic needed to replay normalized and physical scores."""
    state = {key: getattr(normalizer, key).detach().cpu().clone()
             for key in ("mean_input", "mean_target", "std_input", "std_target")}
    path = output_dir / "fit_normalizer.pth"
    torch.save({"schema": SCHEMA, "normalizer_state": state}, path)
    return {"path": str(path), "sha256": sha256_file(path),
            "state_sha256": state_sha(state)}


def git_report() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--untracked-files=all")
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "clean": not bool(dirty), "porcelain": dirty.splitlines()}


def make_base() -> TemporalTransportOperator:
    return TemporalTransportOperator(TemporalTransportConfig(
        hidden_channels=BASE_HIDDEN_CHANNELS, spatial_blocks=2,
        spatial_downsample=2, temporal_blocks=3, temporal_kernel=5,
        mixer_mode="causal_temporal", groups=8, use_coordinates=True,
        max_displacement=4.0, residual_limit=4.0,
        output_initialization_std=1e-3, predict_uncertainty=False))


def make_refiner() -> DirectFutureRefiner:
    return DirectFutureRefiner(DirectFutureRefinerConfig(
        steps=20, channels=3, hidden_channels=REFINER_HIDDEN_CHANNELS,
        blocks=REFINER_BLOCKS, groups=8, residual_limit=4.0,
        output_initialization_std=1e-3))


def construct_models(device: torch.device) -> tuple[
        TemporalTransportOperator, dict[str, DirectFutureRefiner], dict[str, object]]:
    torch.manual_seed(SEED)
    base = make_base().to(device)
    torch.manual_seed(SEED + 1)
    control = make_refiner().to(device)
    control_state = {key: value.detach().cpu().clone()
                     for key, value in control.state_dict().items()}
    torch.manual_seed(SEED + 1)
    candidate = make_refiner().to(device)
    candidate_state = {key: value.detach().cpu().clone()
                       for key, value in candidate.state_dict().items()}
    if control_state.keys() != candidate_state.keys() or any(
            not torch.equal(control_state[key], candidate_state[key])
            for key in control_state):
        raise RuntimeError("refiner arms do not share a bitwise initial state")
    refiners = {"control": control, "future_noise": candidate}
    base_parameters = sum(p.numel() for p in base.parameters())
    refiner_parameters = sum(p.numel() for p in control.parameters())
    if (base_parameters != EXPECTED_BASE_PARAMETERS
            or refiner_parameters != EXPECTED_REFINER_PARAMETERS):
        raise RuntimeError(
            "model parameter contract changed: "
            f"base={base_parameters}, refiner={refiner_parameters}")
    return base, refiners, {
        "base_parameters": base_parameters,
        "refiner_parameters": refiner_parameters,
        "refiner_state_sha256": state_sha(control_state),
        "refiner_state_tensors": len(control_state),
        "refiner_bitwise_identical": True,
    }


def build_window_schedule(windows: list[object], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(windows), size=(UPDATES, BATCH_SIZE), dtype=np.int64)


def build_refiner_policy(*, seed: int) -> np.ndarray:
    """Return candidate auxiliary noise levels in fit-residual-sigma units.

    Direct p0 passes always use level one, the actual deployment condition.
    Consequently half of the candidate's refiner forward calls are inference
    matched and its other half is a noise-conditioned future reconstruction
    task.  The matched control uses ``p0`` for both calls.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray((NOISE_LEVEL_MIN, NOISE_LEVEL_MAX), dtype=np.float32),
                      size=(UPDATES, BATCH_SIZE), replace=True)


def materialize_batch(trajectories: list[torch.Tensor], windows: list[object], indexes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    samples = []
    for index in indexes:
        window = windows[int(index)]
        trajectory = trajectories[int(window.trajectory)]
        start = int(window.start)
        samples.append((trajectory[start:start + 20], trajectory[start + 20:start + 40]))
    history, target = zip(*samples, strict=True)
    return torch.stack(history), torch.stack(target)


def prediction(base: nn.Module, x_norm: torch.Tensor) -> torch.Tensor:
    # ``inference_mode`` tensors cannot subsequently be saved by G's Conv3d
    # autograd graph.  ``no_grad`` still freezes F while returning an ordinary
    # tensor that is a legal constant input to refiner training.
    with torch.no_grad():
        return base(x_norm).detach()


def point_loss(prediction_norm: torch.Tensor, target_norm: torch.Tensor, normalizer) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    accuracy, details = backbone.common.accuracy_objective(
        prediction_norm, target_norm, normalizer, (1.0, 4.0, 1.0))
    mse = (prediction_norm - target_norm).square().mean()
    return accuracy + .25 * mse, {"mse": mse.detach(), **details}


def train_base(base: TemporalTransportOperator, trajectories, windows, schedule, normalizer, device: torch.device, output_dir: Path) -> dict[str, object]:
    backbone.common.set_model_stats(base, normalizer)
    optimizer = torch.optim.AdamW(base.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    trace, interval = [], []
    started = time.perf_counter()
    for update, indexes in enumerate(schedule, 1):
        x, y = materialize_batch(trajectories, windows, indexes)
        x, y = x.to(device), y.to(device)
        x_norm, y_norm = normalizer.preprocess(x, y)
        base.train()
        optimizer.zero_grad(set_to_none=True)
        loss, details = point_loss(base(x_norm), y_norm, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"base non-finite at update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        optimizer.step(); scheduler.step(); interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            row = {"stage": "base", "update": update,
                   "loss_mean": float(np.mean(interval)),
                   "loss_median": float(np.median(interval)),
                   **{key: float(value) for key, value in details.items()},
                   "elapsed_seconds": time.perf_counter() - started}
            trace.append(row); interval.clear(); print(json.dumps(row, sort_keys=True), flush=True)
    base.eval()
    state = {key: value.detach().cpu() for key, value in base.state_dict().items()}
    endpoint = output_dir / "base_endpoint.pth"
    torch.save({"schema": SCHEMA, "stage": "base", "updates": UPDATES,
                "model_config": base.architecture_config(), "model_state_dict": state}, endpoint)
    return {"path": str(endpoint), "sha256": sha256_file(endpoint),
            "model_state_sha256": state_sha(state), "trace": trace,
            "wall_seconds": time.perf_counter() - started}


def estimate_fit_residual_sigma(base: nn.Module, trajectories, windows, normalizer, device: torch.device) -> float:
    """Estimate the single noise unit from frozen base residuals on fit only."""
    squares = 0.0
    values = 0
    base.eval()
    for start in range(0, len(windows), 32):
        x, y = materialize_batch(trajectories, windows, np.arange(start, min(start + 32, len(windows))))
        x, y = x.to(device), y.to(device)
        x_norm, y_norm = normalizer.preprocess(x, y)
        residual = prediction(base, x_norm) - y_norm
        squares += float(residual.square().sum().cpu())
        values += int(residual.numel())
    sigma = float(np.sqrt(squares / max(values, 1)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise FloatingPointError("fit-only base residual sigma is invalid")
    return sigma


def noisy_refiner_input(*, target: torch.Tensor, noise_levels: np.ndarray,
                        sigma: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the candidate-only auxiliary denoising input from fit sigma."""
    level = torch.as_tensor(noise_levels, device=target.device, dtype=target.dtype)
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype,
                        generator=generator)
    z_noisy = target + level[:, None, None, None, None] * float(sigma) * noise
    return z_noisy, level


def train_refiner(arm: str, refiner: DirectFutureRefiner, base: nn.Module, trajectories, windows, schedule, noise_level_policy, sigma: float, normalizer, device: torch.device, output_dir: Path) -> dict[str, object]:
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    noise_generator = torch.Generator(device=device).manual_seed(NOISE_SEED)
    trace, interval = [], []
    started = time.perf_counter()
    for update, indexes in enumerate(schedule, 1):
        x, y = materialize_batch(trajectories, windows, indexes)
        x, y = x.to(device), y.to(device)
        x_norm, y_norm = normalizer.preprocess(x, y)
        p0 = prediction(base, x_norm)
        # Both arms first see the exact R=1 deployment operation.  ``level=1``
        # denotes the fit-only residual-sigma unit, not an extra inference
        # iteration.  This prevents the candidate's direct objective from
        # being diluted by its denoising auxiliary task.
        direct_level = p0.new_ones(p0.shape[0])
        refiner.train()
        optimizer.zero_grad(set_to_none=True)
        direct_loss, direct_details = point_loss(refiner(p0, direct_level), y_norm, normalizer)
        if arm == "control":
            auxiliary_input, auxiliary_level = p0, direct_level
        elif arm == "future_noise":
            auxiliary_input, auxiliary_level = noisy_refiner_input(
                target=y_norm, noise_levels=noise_level_policy[update - 1],
                sigma=sigma, generator=noise_generator)
        else:
            raise ValueError(arm)
        auxiliary_loss, auxiliary_details = point_loss(
            refiner(auxiliary_input, auxiliary_level), y_norm, normalizer)
        loss = direct_loss + AUXILIARY_WEIGHT * auxiliary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{arm} non-finite at update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        optimizer.step(); scheduler.step(); interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            row = {"stage": arm, "update": update,
                   "loss_mean": float(np.mean(interval)),
                   "loss_median": float(np.median(interval)),
                   "direct_loss": float(direct_loss.detach()),
                   "auxiliary_loss": float(auxiliary_loss.detach()),
                   "auxiliary_weight": AUXILIARY_WEIGHT,
                   "direct_mse": float(direct_details["mse"]),
                   "auxiliary_mse": float(auxiliary_details["mse"]),
                   "auxiliary_noise_level_mean": (float(np.mean(noise_level_policy[update - len(interval):update]))
                                                  if arm == "future_noise" else 1.0),
                   **{key: float(value) for key, value in direct_details.items()
                      if key != "mse"},
                   "elapsed_seconds": time.perf_counter() - started}
            trace.append(row); interval.clear(); print(json.dumps(row, sort_keys=True), flush=True)
    refiner.eval()
    state = {key: value.detach().cpu() for key, value in refiner.state_dict().items()}
    endpoint = output_dir / f"{arm}_endpoint.pth"
    torch.save({"schema": SCHEMA, "stage": arm, "updates": UPDATES,
                "model_config": refiner.architecture_config(), "model_state_dict": state}, endpoint)
    return {"path": str(endpoint), "sha256": sha256_file(endpoint),
            "model_state_sha256": state_sha(state), "trace": trace,
            "wall_seconds": time.perf_counter() - started}


def error_rows(prediction_value: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((
        common.SCORING.rel_l2_per_sample(prediction_value, target, 2),
        common.SCORING.tke_rel_l2_per_sample(prediction_value, target, 2),
        common.SCORING.mvpe_rel_l2_per_sample(prediction_value, target, 2),
    ), axis=1).astype(np.float64)


def score(errors: np.ndarray) -> dict[str, float | int]:
    means = errors.mean(0, dtype=np.float64)
    result: dict[str, float | int] = {"samples": int(len(errors)),
        "rel_l2_score": common.SCORING.score_error(float(means[0])),
        "tke_score": common.SCORING.score_error(float(means[1])),
        "mvpe_score": common.SCORING.score_error(float(means[2]))}
    result["point_contribution_46_10_9"] = sum(
        POINT_WEIGHTS[key] * float(result[key]) for key in POINT_WEIGHTS)
    return result


def evaluate(base: nn.Module, refiners: dict[str, nn.Module], trajectories, windows, window_files, normalizer, device: torch.device) -> dict[str, object]:
    errors: dict[str, list[np.ndarray]] = {"base": [], **{name: [] for name in ARM_NAMES}}
    files: list[str] = []
    for start in range(0, len(windows), 32):
        indexes = np.arange(start, min(start + 32, len(windows)))
        x, y = materialize_batch(trajectories, windows, indexes)
        x_device = x.to(device)
        x_norm = (x_device - normalizer.mean_input) / normalizer.std_input
        p0 = prediction(base, x_norm)
        outputs = {"base": p0}
        for name, refiner in refiners.items():
            refiner.eval()
            with torch.inference_mode():
                outputs[name] = refiner(p0, p0.new_ones(p0.shape[0]))
        target = y.numpy()
        for name, output in outputs.items():
            physical = (output * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
            errors[name].append(error_rows(physical, target))
        files.extend(window_files[int(index)] for index in indexes)
    joined = {name: np.concatenate(rows) for name, rows in errors.items()}
    metrics = {name: score(rows) for name, rows in joined.items()}
    labels = np.asarray(files)
    comparisons: dict[str, object] = {}
    for left, right in (("future_noise", "control"), ("future_noise", "base"), ("control", "base")):
        delta = {key: float(metrics[left][key]) - float(metrics[right][key])
                 for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}
        per_file = {}
        for file in sorted(set(files)):
            rows = labels == file
            per_file[file] = float(score(joined[left][rows])["point_contribution_46_10_9"] - score(joined[right][rows])["point_contribution_46_10_9"])
        values = list(per_file.values())
        comparisons[f"{left}_minus_{right}"] = {
            "delta": delta, "per_trajectory_file_delta_point": per_file,
            "positive_trajectory_files": sum(value > 0.0 for value in values),
            "worst_trajectory_delta_point": min(values)}
    return {"metrics": metrics, "comparisons": comparisons}


def prior_held10_evidence(path: Path | None, held_aoa: int) -> dict[str, object] | None:
    if held_aoa == 10:
        if path is not None:
            raise ValueError("held10 does not take a prior result")
        return None
    if path is None or not path.is_file():
        raise PermissionError("held20 requires --prior-held10-result from this screen")
    result = json.loads(path.read_text(encoding="utf-8"))
    config = result.get("config", {})
    fixed_contract = {
        "held_aoa": 10, "updates_base": UPDATES,
        "updates_per_refiner": UPDATES, "batch_size": BATCH_SIZE,
        "seed": SEED, "noise_seed": NOISE_SEED, "r_steps": 1,
        "base_hidden_channels": BASE_HIDDEN_CHANNELS,
        "refiner_hidden_channels": REFINER_HIDDEN_CHANNELS,
        "refiner_blocks": REFINER_BLOCKS, "arms": list(ARM_NAMES),
        "direct_p0_condition_level": 1.0,
        "auxiliary_weight": AUXILIARY_WEIGHT,
    }
    valid_gate = False
    try:
        valid_gate = gate(result["evaluation"])[0]
    except (KeyError, TypeError, ValueError):
        pass
    if (result.get("schema") != SCHEMA
            or any(config.get(key) != value
                   for key, value in fixed_contract.items())
            or result.get("decision") != "GO_HELD20"
            or result.get("git", {}).get("clean") is not True
            or not valid_gate):
        raise PermissionError("prior file is not a verified held10 GO")
    seal_record = result.get("endpoint_seal", {})
    seal_path = Path(str(seal_record.get("path", "")))
    if (not seal_path.is_file()
            or sha256_file(seal_path) != seal_record.get("sha256")):
        raise PermissionError("held10 endpoint seal is missing or changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    records = [seal.get("normalizer", {}), seal.get("base", {})]
    records.extend(seal.get("refiners", {}).get(name, {}) for name in ARM_NAMES)
    for record in records:
        artifact_path = Path(str(record.get("path", "")))
        if (not artifact_path.is_file()
                or sha256_file(artifact_path) != record.get("sha256")):
            raise PermissionError("a held10 sealed artifact is missing or changed")
    return {"path": str(path), "sha256": sha256_file(path), "decision": result["decision"]}


def gate(evaluation: dict[str, object]) -> tuple[bool, dict[str, object]]:
    paired = evaluation["comparisons"]["future_noise_minus_control"]
    base = evaluation["comparisons"]["future_noise_minus_base"]
    delta = paired["delta"]
    passed = (float(delta["point_contribution_46_10_9"]) >= .03
              and all(float(delta[key]) >= -.05 for key in POINT_WEIGHTS)
              and int(paired["positive_trajectory_files"]) >= 15
              and float(paired["worst_trajectory_delta_point"]) >= -.05
              and float(base["delta"]["point_contribution_46_10_9"]) >= 0.0)
    return passed, {"future_noise_minus_control_point_min": .03,
                    "component_floor": -.05,
                    "positive_trajectory_files_min": 15,
                    "worst_trajectory_delta_point_min": -.05,
                    "future_noise_minus_base_point_min": 0.0}


def smoke(device: torch.device) -> None:
    base, refiners, identity = construct_models(device)
    x = torch.randn(2, 20, 16, 32, 3, device=device)
    with torch.inference_mode():
        p0 = base(x)
        for refiner in refiners.values():
            ordinary = refiner(p0)
            explicit = refiner(p0, torch.zeros(2, device=device))
            nonzero = refiner(p0, torch.tensor([.25, 1.0], device=device))
            assert ordinary.shape == p0.shape
            torch.testing.assert_close(ordinary, explicit, rtol=0.0, atol=0.0)
            assert torch.isfinite(nonzero).all()
    levels = build_refiner_policy(seed=SEED)
    assert levels.shape == (UPDATES, BATCH_SIZE)
    assert set(np.unique(levels)).issubset({NOISE_LEVEL_MIN, NOISE_LEVEL_MAX})
    print(json.dumps({"smoke": "PASS", "identity": identity}, sort_keys=True))


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != "DIRECT_FUTURE_REFINER_LOAO_V1":
        raise PermissionError("formal run requires --execute and protocol acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    prior = prior_held10_evidence(args.prior_held10_result, args.held_aoa)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = git_report()
    if not git["clean"]:
        raise RuntimeError(f"Git worktree must be clean: {git['porcelain']}")
    inventory = common.scan_inventory(args.sim_dir)
    fit_records = [row for row in inventory if int(row["aoa"]) != args.held_aoa]
    held_records = [row for row in inventory if int(row["aoa"]) == args.held_aoa]
    if len(fit_records) != 80 or len(held_records) != 20:
        raise RuntimeError("whole-AoA split count changed")
    config = {"schema": SCHEMA, "held_aoa": args.held_aoa,
              "fit_aoas": [aoa for aoa in AOAS if aoa != args.held_aoa],
              "updates_base": UPDATES, "updates_per_refiner": UPDATES,
              "batch_size": BATCH_SIZE, "seed": SEED,
              "noise_seed": NOISE_SEED, "r_steps": 1,
              "base_hidden_channels": BASE_HIDDEN_CHANNELS,
              "refiner_hidden_channels": REFINER_HIDDEN_CHANNELS,
              "refiner_blocks": REFINER_BLOCKS, "arms": list(ARM_NAMES),
              "candidate_inference_matched_forward_fraction": 0.50,
              "direct_p0_condition_level": 1.0,
              "auxiliary_weight": AUXILIARY_WEIGHT,
              "noisy_level_range_fit_residual_sigma_units": [NOISE_LEVEL_MIN, NOISE_LEVEL_MAX],
              "fit_inventory_sha256": common.canonical_sha(fit_records),
              "held_inventory_sha256": common.canonical_sha(held_records),
              "held_field_values_loaded_before_all_endpoint_seals": False,
              "prior_held10_evidence": prior}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    (args.output_dir / "environment.txt").write_text("".join(
        f"{key}={value}\n" for key, value in {"python": sys.version.replace("\n", " "),
            "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device), "pid": os.getpid()}.items()), encoding="utf-8")

    fit_trajectories, fit_windows, _ = common.load_fields(args.sim_dir, fit_records)
    if len(fit_windows) != 80 * 48:
        raise RuntimeError(f"expected 3840 fit windows, got {len(fit_windows)}")
    normalizer = backbone.common.scratch.ChannelNormalizer(
        *backbone.common.scratch.window_statistics(fit_trajectories, fit_windows), device=device)
    normalizer_artifact = save_normalizer(normalizer, args.output_dir)
    base_schedule = build_window_schedule(fit_windows, seed=SEED + args.held_aoa)
    refiner_schedule = build_window_schedule(fit_windows, seed=SEED + args.held_aoa + 1)
    level_policy = build_refiner_policy(seed=SEED + args.held_aoa + 2)
    base, refiners, initial = construct_models(device)
    base_endpoint = train_base(base, fit_trajectories, fit_windows, base_schedule, normalizer, device, args.output_dir)
    # Fit-only calibration is intentionally measured only after the base has
    # frozen; it is a data-dependent part of the candidate's declared input.
    residual_sigma = estimate_fit_residual_sigma(base, fit_trajectories, fit_windows, normalizer, device)
    endpoints = {name: train_refiner(name, refiners[name], base, fit_trajectories, fit_windows,
                                     refiner_schedule, level_policy, residual_sigma,
                                     normalizer, device, args.output_dir) for name in ARM_NAMES}
    seal = {"schema": SCHEMA, "normalizer": normalizer_artifact,
            "base": {key: base_endpoint[key] for key in ("path", "sha256", "model_state_sha256")},
            "refiners": {name: {key: endpoints[name][key] for key in ("path", "sha256", "model_state_sha256")} for name in ARM_NAMES}}
    seal_path = args.output_dir / "all_endpoints_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    held_trajectories, held_windows, held_files = common.load_fields(args.sim_dir, held_records)
    if len(held_windows) != 20 * 48:
        raise RuntimeError(f"expected 960 held windows, got {len(held_windows)}")
    evaluation = evaluate(base, refiners, held_trajectories, held_windows, held_files, normalizer, device)
    passed, gate_config = gate(evaluation)
    decision = "GO_HELD20" if args.held_aoa == 10 and passed else ("COMPLETE_PROMISING" if passed else "STOP_DIRECT_FUTURE_REFINER")
    result = {"schema": SCHEMA, "evidence_class": "whole_AoA_held_Sim", "config": config,
              "git": git, "initial_identity": initial, "fit_trajectories": len(fit_records),
              "fit_windows": len(fit_windows), "held_trajectories": len(held_records), "held_windows": len(held_windows),
              "base_schedule_sha256": hashlib.sha256(base_schedule.tobytes()).hexdigest(),
              "refiner_schedule_sha256_all_arms": hashlib.sha256(refiner_schedule.tobytes()).hexdigest(),
              "candidate_noise_level_policy_sha256": hashlib.sha256(level_policy.tobytes()).hexdigest(),
              "candidate_noisy_examples": UPDATES * BATCH_SIZE, "fit_residual_sigma_normalized": residual_sigma,
              "normalizer_artifact": normalizer_artifact,
              "base_endpoint": base_endpoint, "refiner_endpoints": endpoints,
              "endpoint_seal": {"path": str(seal_path), "sha256": sha256_file(seal_path)},
              "evaluation": evaluation, "gate": gate_config, "decision": decision}
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "held_aoa": args.held_aoa,
                      "future_noise_minus_control": evaluation["comparisons"]["future_noise_minus_control"]["delta"],
                      "future_noise_minus_base": evaluation["comparisons"]["future_noise_minus_base"]["delta"],
                      "control_minus_base": evaluation["comparisons"]["control_minus_base"]["delta"],
                      "result": str(result_path)}, sort_keys=True), flush=True)


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
