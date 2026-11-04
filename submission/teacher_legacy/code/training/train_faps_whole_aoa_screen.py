#!/usr/bin/env python3
"""Strict whole-AoA Sim screen for future-aligned predictive states (FAPS).

This is deliberately a low-cost falsification screen.  Candidate and control
start from bitwise-identical ``hidden=64, latent=32`` models, receive the same
2,000 x batch-8 released-data stream, run the same two calls through one
shared decoder, and differ only in ``lambda_align`` (1e-4 versus 0).  The
future encoder sees future *residuals* during training only and is absent from
the deployment forecaster.

Held flow arrays are not read until both endpoints and their hashes are
sealed.  This screen evaluates the ordinary 20->20 task and does not create
new trajectories or use an autoregressive rollout.
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
import train_temporal_transport_loao_screen as temporal_screen  # noqa: E402
from models.future_aligned_predictive_state import (  # noqa: E402
    FAPSConfig,
    FutureAlignedPredictiveState,
)


SCHEMA = "faps_whole_aoa_sim_screen_v1"
AOAS = (0, 5, 10, 15, 20)
SEED = 210
UPDATES = 2_000
BATCH_SIZE = 8
HIDDEN_CHANNELS = 64
LATENT_CHANNELS = 32
ARM_NAMES = ("control", "candidate")
LAMBDA_ALIGN = {"control": 0.0, "candidate": 1.0e-4}
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}
EXPECTED_TRAINING_PARAMETERS = 575_875
EXPECTED_DEPLOYMENT_PARAMETERS = 373_411
INITIAL_ALIGNMENT_GRADIENT_RATIO_MAX = .25

# Reuse the long-standing inventory and released scoring implementation, not
# the temporal model itself.  This keeps window and score conventions exactly
# aligned with prior strict whole-AoA screens.
SCORING = temporal_screen.SCORING
scan_inventory = temporal_screen.scan_inventory
load_fields = temporal_screen.load_fields
sha256_file = temporal_screen.sha256_file
canonical_sha = temporal_screen.canonical_sha
state_sha = temporal_screen.state_sha


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


def git_report() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--untracked-files=all")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "clean": not bool(dirty), "porcelain": dirty.splitlines()}


def make_config() -> FAPSConfig:
    return FAPSConfig(hidden_channels=HIDDEN_CHANNELS,
                      latent_channels=LATENT_CHANNELS, groups=8,
                      encoder_blocks=2, decoder_blocks=2)


def construct_pair(device: torch.device) -> tuple[
        dict[str, FutureAlignedPredictiveState], dict[str, object]]:
    states: dict[str, dict[str, torch.Tensor]] = {}
    arms: dict[str, FutureAlignedPredictiveState] = {}
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = FutureAlignedPredictiveState(make_config()).to(device)
        arms[name] = model
        states[name] = {key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()}
    reference = states["control"]
    if any(reference.keys() != state.keys() or any(
            not torch.equal(reference[key], state[key]) for key in reference)
           for state in states.values()):
        raise RuntimeError("candidate/control initial states differ")
    parameters = {name: sum(value.numel() for value in model.parameters())
                  for name, model in arms.items()}
    if len(set(parameters.values())) != 1:
        raise RuntimeError(f"candidate/control parameter mismatch: {parameters}")
    deployment_parameters = sum(
        value.numel() for value in arms["control"].forecaster.parameters())
    if (parameters["control"] != EXPECTED_TRAINING_PARAMETERS
            or deployment_parameters != EXPECTED_DEPLOYMENT_PARAMETERS):
        raise RuntimeError(
            "FAPS parameter contract changed: "
            f"training={parameters['control']}, deployment={deployment_parameters}")
    return arms, {"state_sha256": state_sha(reference),
                  "training_parameters": parameters["control"],
                  "deployment_parameters": deployment_parameters,
                  "stripped_future_analysis_parameters": (
                      parameters["control"] - deployment_parameters),
                  "state_tensors": len(reference), "bitwise_identical": True,
                  "lambda_align": LAMBDA_ALIGN.copy()}


def make_schedule(windows: list[object], updates: int, batch_size: int,
                  rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, len(windows), size=(updates, batch_size), dtype=np.int64)


def batches(dataset, schedule: np.ndarray):
    for row in schedule:
        samples = [dataset[int(index)] for index in row]
        yield (torch.stack([item[0] for item in samples]),
               torch.stack([item[1] for item in samples]))


def loss_terms(model: FutureAlignedPredictiveState, x_norm: torch.Tensor,
               y_norm: torch.Tensor, normalizer, lambda_align: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Two shared-decoder passes in both arms; only the alignment scalar differs."""
    forecast = model.forecast_components(x_norm)
    analysis = model.analysis_components(y_norm, forecast["base"])
    accuracy, details = backbone.common.accuracy_objective(
        forecast["prediction"], y_norm, normalizer, (1.0, 4.0, 1.0))
    forecast_mse = (forecast["prediction"] - y_norm).square().mean()
    reconstruction_mse = (analysis["reconstructed_residual"]
                          - analysis["future_residual"]).square().mean()
    # zY is explicitly a target state.  No alignment gradient may train the
    # future encoder, which keeps this distinct from a hidden co-adaptation
    # bypass and makes the analysis state auditable.
    alignment = (forecast["z_history"] - analysis["z_future"].detach()).square().mean()
    loss = accuracy + .25 * forecast_mse + .25 * reconstruction_mse + lambda_align * alignment
    values = {"accuracy": accuracy, "forecast_mse": forecast_mse,
              "reconstruction_mse": reconstruction_mse, "alignment": alignment,
              "loss": loss, **details}
    return loss, values


def _gradient_norm(loss: torch.Tensor, parameters: tuple[torch.Tensor, ...], *,
                   retain_graph: bool) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                    allow_unused=True)
    squared = sum(float(gradient.detach().float().square().sum())
                  for gradient in gradients if gradient is not None)
    return float(np.sqrt(squared))


def initial_gradient_audit(model: FutureAlignedPredictiveState,
                           x_norm: torch.Tensor, y_norm: torch.Tensor,
                           normalizer) -> dict[str, float]:
    """Fit-only update-1 audit preventing latent alignment domination.

    Decoder output weights intentionally start near zero, so comparing scalar
    losses is misleading: the latent loss has a much shorter gradient path.
    The preregistered 1e-4 coefficient must contribute no more than 25% of the
    ordinary forecast gradient to the history encoder on the first fit batch.
    """
    model.train()
    forecast = model.forecast_components(x_norm)
    analysis = model.analysis_components(y_norm, forecast["base"])
    accuracy, _ = backbone.common.accuracy_objective(
        forecast["prediction"], y_norm, normalizer, (1.0, 4.0, 1.0))
    forecast_mse = (forecast["prediction"] - y_norm).square().mean()
    forecast_main = accuracy + .25 * forecast_mse
    alignment = (forecast["z_history"]
                 - analysis["z_future"].detach()).square().mean()
    parameters = tuple(model.forecaster.history_encoder.parameters())
    forecast_norm = _gradient_norm(forecast_main, parameters, retain_graph=True)
    alignment_norm = _gradient_norm(alignment, parameters, retain_graph=False)
    weighted_ratio = (LAMBDA_ALIGN["candidate"] * alignment_norm
                      / max(forecast_norm, 1.0e-12))
    return {"forecast_main_loss": float(forecast_main.detach()),
            "alignment_loss": float(alignment.detach()),
            "forecast_history_encoder_grad_norm": forecast_norm,
            "unweighted_alignment_history_encoder_grad_norm": alignment_norm,
            "weighted_alignment_to_forecast_grad_ratio": weighted_ratio,
            "maximum_allowed_ratio": INITIAL_ALIGNMENT_GRADIENT_RATIO_MAX}


def train_arm(name: str, model: FutureAlignedPredictiveState, dataset,
              schedule: np.ndarray, normalizer, device: torch.device,
              output_dir: Path) -> dict[str, object]:
    backbone.common.set_model_stats(model, normalizer)
    # Any later stochastic addition remains paired too.
    torch.manual_seed(SEED + 17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 17)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    trace: list[dict[str, object]] = []
    losses: list[float] = []
    started = time.perf_counter()
    for update, (x, y) in enumerate(batches(dataset, schedule), 1):
        model.train()
        x, y = x.to(device), y.to(device)
        x_norm, y_norm = normalizer.preprocess(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = loss_terms(model, x_norm, y_norm, normalizer, LAMBDA_ALIGN[name])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} non-finite at update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            trace.append({"arm": name, "update": update,
                          "loss_mean": float(np.mean(losses)),
                          "loss_median": float(np.median(losses)),
                          "elapsed_seconds": time.perf_counter() - started,
                          **{key: float(value.detach()) for key, value in terms.items()}})
            print(json.dumps(trace[-1], sort_keys=True), flush=True)
            losses.clear()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    endpoint = output_dir / f"{name}_endpoint.pth"
    torch.save({"schema": SCHEMA, "arm": name, "updates": UPDATES,
                "lambda_align": LAMBDA_ALIGN[name],
                "model_config": model.architecture_config(),
                "model_state_dict": state}, endpoint)
    return {"path": str(endpoint), "sha256": sha256_file(endpoint),
            "model_state_sha256": state_sha(state), "trace": trace,
            "wall_seconds": time.perf_counter() - started}


def error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((SCORING.rel_l2_per_sample(prediction, target, 2),
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


def latent_audit(model: FutureAlignedPredictiveState, x_norm: torch.Tensor,
                 y_norm: torch.Tensor) -> dict[str, float]:
    """Audit one window per held trajectory with cross-trajectory shuffling."""
    model.eval()
    with torch.inference_mode():
        forecast = model.forecast_components(x_norm)
        analysis = model.analysis_components(y_norm, forecast["base"])
        zh, zy = forecast["z_history"], analysis["z_future"]
        if zh.shape[0] < 2:
            raise ValueError("balanced latent audit needs at least two trajectories")
        shuffled_zy = torch.roll(zy, 1, dims=0)
        # Fixed non-identity permutation must visibly affect the shared decoder.
        permutation = torch.roll(torch.arange(zh.shape[1], device=zh.device), 1)
        permuted = model.forecaster.decoder(zh[:, permutation], forecast["base"])
        decoded = model.forecaster.decoder(zh, forecast["base"])
        reconstructed = analysis["reconstructed_residual"]
        reconstructed_shuffled = model.forecaster.decoder(
            shuffled_zy, forecast["base"])
        reconstructed_zero = model.forecaster.decoder(
            torch.zeros_like(zy), forecast["base"])
        residual = analysis["future_residual"]
    def effective_rank(value: torch.Tensor) -> float:
        rows = value.float().reshape(value.shape[0], -1)
        singular = torch.linalg.svdvals(rows - rows.mean(0, keepdim=True))
        probability = singular.square() / singular.square().sum().clamp_min(1.0e-12)
        return float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))
    matched = float((zh - zy).square().mean())
    shuffled = float((zh - shuffled_zy).square().mean())
    reconstruction = float((reconstructed - residual).square().mean())
    reconstruction_shuffled = float(
        (reconstructed_shuffled - residual).square().mean())
    reconstruction_zero = float((reconstructed_zero - residual).square().mean())
    return {"z_history_batch_std": float(zh.float().std(dim=0).mean()),
            "z_future_batch_std": float(zy.float().std(dim=0).mean()),
            "z_history_effective_rank": effective_rank(zh),
            "z_future_effective_rank": effective_rank(zy),
            "history_future_matched_mse": matched,
            "history_future_shuffled_mse": shuffled,
            "history_future_matched_to_shuffled_ratio": matched / max(shuffled, 1.0e-12),
            "future_reconstruction_mse": reconstruction,
            "shuffled_future_reconstruction_mse": reconstruction_shuffled,
            "zero_future_reconstruction_mse": reconstruction_zero,
            "future_reconstruction_to_best_baseline_ratio": reconstruction / max(
                min(reconstruction_shuffled, reconstruction_zero), 1.0e-12),
            "decoder_fixed_permutation_delta": float((decoded - permuted).abs().mean())}


def balanced_audit_batch(dataset, window_files: list[str], normalizer,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Select the middle standard window from every held trajectory."""
    indexes_by_file: dict[str, list[int]] = {}
    for index, file in enumerate(window_files):
        indexes_by_file.setdefault(file, []).append(index)
    indexes = [rows[len(rows) // 2] for _file, rows in sorted(indexes_by_file.items())]
    samples = [dataset[index] for index in indexes]
    x = torch.stack([item[0] for item in samples]).to(device)
    y = torch.stack([item[1] for item in samples]).to(device)
    x_norm = (x - normalizer.mean_input) / normalizer.std_input
    y_norm = (y - normalizer.mean_target) / normalizer.std_target
    return x_norm, y_norm, [window_files[index] for index in indexes]


def evaluate_pair(arms: dict[str, FutureAlignedPredictiveState], trajectories,
                  windows, window_files: list[str], normalizer,
                  device: torch.device) -> dict[str, object]:
    dataset = backbone.common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    errors = {name: [] for name in ARM_NAMES}
    files: list[str] = []
    for start in range(0, len(dataset), 64):
        indexes = range(start, min(start + 64, len(dataset)))
        samples = [dataset[index] for index in indexes]
        x = torch.stack([item[0] for item in samples]).to(device)
        target = torch.stack([item[1] for item in samples])
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        y_norm = (target.to(device) - normalizer.mean_target) / normalizer.std_target
        for name, model in arms.items():
            model.eval()
            with torch.inference_mode():
                prediction = (model(x_norm) * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
            errors[name].append(error_rows(prediction, target.numpy()))
        files.extend(window_files[index] for index in indexes)
    audit_x, audit_y, audit_files = balanced_audit_batch(
        dataset, window_files, normalizer, device)
    audits = {name: latent_audit(model, audit_x, audit_y)
              for name, model in arms.items()}
    joined = {name: np.concatenate(value) for name, value in errors.items()}
    metrics = {name: score(value) for name, value in joined.items()}
    labels = np.asarray(files)
    delta = {key: float(metrics["candidate"][key]) - float(metrics["control"][key])
             for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}
    per_file = {}
    for file in sorted(set(files)):
        mask = labels == file
        per_file[file] = float(score(joined["candidate"][mask])["point_contribution_46_10_9"]
                               - score(joined["control"][mask])["point_contribution_46_10_9"])
    return {"metrics": metrics, "candidate_minus_control": {
        "delta": delta, "per_trajectory_file_delta_point": per_file,
        "positive_trajectory_files": sum(value > 0.0 for value in per_file.values()),
        "worst_trajectory_delta_point": min(per_file.values())},
        "latent_audits": audits,
        "latent_audit_trajectory_files": audit_files}


def gate(evaluation: dict[str, object]) -> tuple[bool, dict[str, object]]:
    comparison = evaluation["candidate_minus_control"]
    delta = comparison["delta"]
    integrity = evaluation["latent_audits"]["candidate"]
    control_integrity = evaluation["latent_audits"]["control"]
    passed = (float(delta["point_contribution_46_10_9"]) >= .05
              and all(float(delta[key]) >= -.03 for key in POINT_WEIGHTS)
              and int(comparison["positive_trajectory_files"]) >= 15
              and float(comparison["worst_trajectory_delta_point"]) >= -.05
              and min(integrity["z_history_batch_std"], integrity["z_future_batch_std"]) > 1.0e-4
              and min(integrity["z_history_effective_rank"], integrity["z_future_effective_rank"]) >= 4.0
              and integrity["decoder_fixed_permutation_delta"] > 1.0e-7
              and integrity["history_future_matched_to_shuffled_ratio"] < 1.0
              and integrity["history_future_matched_to_shuffled_ratio"]
              < control_integrity["history_future_matched_to_shuffled_ratio"]
              and integrity["future_reconstruction_to_best_baseline_ratio"] < 1.0)
    return passed, {"candidate_minus_control_point_min": .05,
                    "component_floor": -.03, "positive_trajectory_files_min": 15,
                    "worst_trajectory_delta_point_min": -.05,
                    "noncollapse_batch_std_min": 1.0e-4,
                    "noncollapse_effective_rank_min": 4.0,
                    "decoder_fixed_permutation_delta_min": 1.0e-7,
                    "history_future_matched_to_shuffled_ratio_max": 1.0,
                    "candidate_alignment_ratio_must_beat_control": True,
                    "future_reconstruction_to_best_baseline_ratio_max": 1.0}


def prior_held10_evidence(path: Path | None, held_aoa: int) -> dict[str, object] | None:
    if held_aoa == 10:
        if path is not None:
            raise ValueError("held10 does not take a prior held10 result")
        return None
    if path is None or not path.is_file():
        raise PermissionError("held20 requires a held10 GO from this screen")
    prior = json.loads(path.read_text(encoding="utf-8"))
    config = prior.get("config", {})
    fixed_contract = {
        "held_aoa": 10, "updates": UPDATES, "batch_size": BATCH_SIZE,
        "seed": SEED, "hidden_channels": HIDDEN_CHANNELS,
        "latent_channels": LATENT_CHANNELS, "lambda_align": LAMBDA_ALIGN,
        "initial_alignment_gradient_ratio_max": INITIAL_ALIGNMENT_GRADIENT_RATIO_MAX,
    }
    valid_gate = False
    valid_gradient_audit = False
    try:
        valid_gate = gate(prior["evaluation"])[0]
        gradient_ratio = float(prior["initial_fit_only_gradient_audit"][
            "weighted_alignment_to_forecast_grad_ratio"])
        valid_gradient_audit = (
            np.isfinite(gradient_ratio)
            and gradient_ratio <= INITIAL_ALIGNMENT_GRADIENT_RATIO_MAX)
    except (KeyError, TypeError, ValueError):
        pass
    if (prior.get("schema") != SCHEMA
            or any(config.get(key) != value for key, value in fixed_contract.items())
            or prior.get("decision") != "GO_HELD20"
            or prior.get("git", {}).get("clean") is not True
            or not valid_gate or not valid_gradient_audit):
        raise PermissionError("prior file is not a verified held10 GO")
    seal_record = prior.get("endpoint_seal", {})
    seal_path = Path(str(seal_record.get("path", "")))
    if (not seal_path.is_file()
            or sha256_file(seal_path) != seal_record.get("sha256")):
        raise PermissionError("held10 endpoint seal is missing or changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    records = [seal.get("fit_only_normalizer", {})]
    records.extend(seal.get("endpoints", {}).get(name, {}) for name in ARM_NAMES)
    for record in records:
        artifact_path = Path(str(record.get("path", "")))
        if (not artifact_path.is_file()
                or sha256_file(artifact_path) != record.get("sha256")):
            raise PermissionError("a held10 sealed artifact is missing or changed")
    return {"path": str(path), "sha256": sha256_file(path), "decision": prior["decision"]}


def smoke(device: torch.device) -> None:
    arms, identity = construct_pair(device)
    x, y = (torch.randn(2, 20, 16, 32, 3, device=device),
            torch.randn(2, 20, 16, 32, 3, device=device))
    for name, model in arms.items():
        model.eval()
        with torch.inference_mode():
            forecast = model.forecast_components(x)
            analysis = model.analysis_components(y, forecast["base"])
            before = model(x)
            after = model.deployment().to(device).eval()(x)
        assert torch.isfinite(analysis["reconstructed_residual"]).all()
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
        assert name in LAMBDA_ALIGN
    print(json.dumps({"smoke": "PASS", "identity": identity}, sort_keys=True))


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != "FAPS_WHOLE_AOA_V1":
        raise PermissionError("formal run requires --execute and protocol acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    prior = prior_held10_evidence(args.prior_held10_result, args.held_aoa)
    git = git_report()
    if not git["clean"]:
        raise RuntimeError(f"Git worktree must be clean: {git['porcelain']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = scan_inventory(args.sim_dir)
    fit_records = [row for row in inventory if int(row["aoa"]) != args.held_aoa]
    held_records = [row for row in inventory if int(row["aoa"]) == args.held_aoa]
    if len(fit_records) != 80 or len(held_records) != 20:
        raise RuntimeError("whole-AoA inventory changed")
    config = {"schema": SCHEMA, "held_aoa": args.held_aoa,
              "fit_aoas": [aoa for aoa in AOAS if aoa != args.held_aoa],
              "updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED,
              "hidden_channels": HIDDEN_CHANNELS, "latent_channels": LATENT_CHANNELS,
              "lambda_align": LAMBDA_ALIGN,
              "initial_alignment_gradient_ratio_max": INITIAL_ALIGNMENT_GRADIENT_RATIO_MAX,
              "fit_inventory_sha256": canonical_sha(fit_records),
              "held_inventory_sha256": canonical_sha(held_records),
              "held_field_values_loaded_before_all_endpoint_seals": False,
              "prior_held10_evidence": prior}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    (args.output_dir / "environment.txt").write_text("".join(
        f"{key}={value}\n" for key, value in {"python": sys.version.replace("\n", " "),
            "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device), "pid": os.getpid()}.items()), encoding="utf-8")
    # Fit values only until this point.  Held fields do not enter memory until
    # after all endpoint files have been written and independently sealed.
    fit_trajectories, fit_windows, _ = load_fields(args.sim_dir, fit_records)
    if len(fit_windows) != 80 * 48:
        raise RuntimeError("unexpected fit window count")
    normalizer = backbone.common.scratch.ChannelNormalizer(
        *backbone.common.scratch.window_statistics(fit_trajectories, fit_windows), device=device)
    normalizer_path = args.output_dir / "fit_only_normalizer.pt"
    torch.save({"schema": SCHEMA, "mean_input": normalizer.mean_input.detach().cpu(),
                "mean_target": normalizer.mean_target.detach().cpu(),
                "std_input": normalizer.std_input.detach().cpu(),
                "std_target": normalizer.std_target.detach().cpu()}, normalizer_path)
    normalizer_state = {
        "mean_input": normalizer.mean_input.detach().cpu().clone(),
        "mean_target": normalizer.mean_target.detach().cpu().clone(),
        "std_input": normalizer.std_input.detach().cpu().clone(),
        "std_target": normalizer.std_target.detach().cpu().clone(),
    }
    normalizer_artifact = {"path": str(normalizer_path),
                           "sha256": sha256_file(normalizer_path),
                           "state_sha256": state_sha(normalizer_state)}
    dataset = backbone.common.scratch.SimWindowDataset(
        fit_trajectories, fit_windows, mask_probability=0.0, noise_scale=0.0)
    schedule = make_schedule(fit_windows, UPDATES, BATCH_SIZE,
                             np.random.default_rng(SEED + args.held_aoa))
    arms, identity = construct_pair(device)
    for model in arms.values():
        backbone.common.set_model_stats(model, normalizer)
    audit_samples = [dataset[int(index)] for index in schedule[0]]
    audit_x = torch.stack([item[0] for item in audit_samples]).to(device)
    audit_y = torch.stack([item[1] for item in audit_samples]).to(device)
    audit_x_norm, audit_y_norm = normalizer.preprocess(audit_x, audit_y)
    gradient_audit = initial_gradient_audit(
        arms["control"], audit_x_norm, audit_y_norm, normalizer)
    if gradient_audit["weighted_alignment_to_forecast_grad_ratio"] > INITIAL_ALIGNMENT_GRADIENT_RATIO_MAX:
        raise RuntimeError(f"alignment gradient contract failed: {gradient_audit}")
    endpoints = {name: train_arm(name, arms[name], dataset, schedule, normalizer,
                                 device, args.output_dir) for name in ARM_NAMES}
    seal = {"schema": SCHEMA, "fit_only_normalizer": normalizer_artifact,
            "endpoints": {name: {key: endpoints[name][key]
            for key in ("path", "sha256", "model_state_sha256")} for name in ARM_NAMES}}
    seal_path = args.output_dir / "all_endpoints_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_sha256 = sha256_file(seal_path)
    held_trajectories, held_windows, held_files = load_fields(args.sim_dir, held_records)
    if len(held_windows) != 20 * 48:
        raise RuntimeError("unexpected held window count")
    evaluation = evaluate_pair(arms, held_trajectories, held_windows, held_files, normalizer, device)
    passed, gate_config = gate(evaluation)
    decision = "GO_HELD20" if args.held_aoa == 10 and passed else (
        "COMPLETE_PROMISING" if passed else "STOP_FAPS")
    result = {"schema": SCHEMA, "evidence_class": "whole_AoA_held_Sim", "config": config,
              "git": git, "initial_identity": identity, "fit_trajectories": len(fit_records),
              "fit_windows": len(fit_windows), "held_trajectories": len(held_records),
              "held_windows": len(held_windows), "sample_stream_sha256_all_arms": hashlib.sha256(schedule.tobytes()).hexdigest(),
              "initial_fit_only_gradient_audit": gradient_audit,
              "fit_only_normalizer": normalizer_artifact, "endpoints": endpoints,
              "endpoint_seal": {"path": str(seal_path), "sha256": seal_sha256},
              "evaluation": evaluation, "gate": gate_config, "decision": decision}
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "held_aoa": args.held_aoa,
                      "candidate_minus_control": evaluation["candidate_minus_control"]["delta"],
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
