#!/usr/bin/env python3
"""Sealed whole-AoA screen for causal state accumulation versus direct states.

Both arms have identical tensors, optimizer groups, state initialization,
number of transition calls, decoder, and sample stream.  The only variable is
whether the state at horizon k reads state k-1 (candidate) or terminal z0
(control).  It is intentionally point-only and has no UQ or Real stage.
"""

from __future__ import annotations

import argparse
import hashlib
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
from models.causal_state_multiband import (  # noqa: E402
    CausalStateMultibandConfig,
    CausalStateMultibandForecaster,
)

SCHEMA = "causal_state_multiband_whole_aoa_sim_screen_v1"
AOAS = (0, 5, 10, 15, 20)
ARM_NAMES = ("independent_control", "cumulative_candidate")
ARM_MODES = {"independent_control": "independent", "cumulative_candidate": "cumulative"}
SEED, UPDATES, BATCH_SIZE = 210, 2_000, 8
STATE_CHANNELS, DECODER_CHANNELS, CONTEXT_CHANNELS, GROUPS = 96, 64, 96, 8
EXPECTED_PARAMETERS = 1_261_065
LEARNING_RATE, WEIGHT_DECAY, MSE_WEIGHT = 3e-4, 1e-5, .25
ACCURACY_CHANNEL_WEIGHTS = (1., 4., 1.)
STATE_RMS_MIN, PREDICTION_RMS_MIN = 1e-3, 1e-7
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}

SCORING = temporal_screen.SCORING
scan_inventory, load_fields = temporal_screen.scan_inventory, temporal_screen.load_fields
sha256_file, canonical_sha, state_sha = (temporal_screen.sha256_file,
                                         temporal_screen.canonical_sha,
                                         temporal_screen.state_sha)


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


def make_config() -> CausalStateMultibandConfig:
    return CausalStateMultibandConfig(state_channels=STATE_CHANNELS,
                                      decoder_channels=DECODER_CHANNELS,
                                      context_channels=CONTEXT_CHANNELS,
                                      groups=GROUPS)


def construct_arms(device: torch.device) -> tuple[dict[str, CausalStateMultibandForecaster], dict[str, object]]:
    arms, states = {}, {}
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = CausalStateMultibandForecaster(make_config(), ARM_MODES[name]).to(device)
        arms[name] = model
        states[name] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    reference = states[ARM_NAMES[0]]
    if any(reference.keys() != state.keys() or any(not torch.equal(reference[key], state[key])
           for key in reference) for state in states.values()):
        raise RuntimeError("arms do not have bitwise-identical initial state")
    parameters = {name: sum(value.numel() for value in model.parameters()) for name, model in arms.items()}
    if len(set(parameters.values())) != 1 or next(iter(parameters.values())) != EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameter contract changed: {parameters}")
    trainable = {name: sum(value.numel() for value in model.parameters() if value.requires_grad)
                 for name, model in arms.items()}
    if trainable != parameters:
        raise RuntimeError("trainable parameter contract mismatch")
    return arms, {"state_sha256": state_sha(reference), "parameters": parameters,
                  "trainable_parameters": trainable, "bitwise_identical": True,
                  "primary_comparison": "cumulative_candidate_minus_independent_control",
                  "only_variable": "future_state_accumulation"}


def make_schedule(windows: list[object]) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, len(windows), size=(UPDATES, BATCH_SIZE), dtype=np.int64)


def batches(dataset, schedule: np.ndarray):
    for row in schedule:
        values = [dataset[int(index)] for index in row]
        yield torch.stack([value[0] for value in values]), torch.stack([value[1] for value in values])


def loss_terms(model: CausalStateMultibandForecaster, x_norm: torch.Tensor,
               y_norm: torch.Tensor, normalizer) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = model(x_norm)
    accuracy, detail = backbone.common.accuracy_objective(
        prediction, y_norm, normalizer, ACCURACY_CHANNEL_WEIGHTS)
    mse = (prediction - y_norm).square().mean()
    loss = accuracy + MSE_WEIGHT * mse
    return loss, {"accuracy": accuracy, "mse": mse, "loss": loss, **detail}


def train_arm(name: str, model: CausalStateMultibandForecaster, dataset, schedule: np.ndarray,
              normalizer, device: torch.device, output_dir: Path) -> dict[str, object]:
    backbone.common.set_model_stats(model, normalizer)
    torch.manual_seed(SEED + 17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 17)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    trace, losses, started = [], [], time.perf_counter()
    for update, (x, y) in enumerate(batches(dataset, schedule), 1):
        model.train()
        x_norm, y_norm = normalizer.preprocess(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss, terms = loss_terms(model, x_norm, y_norm, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: non-finite loss at {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            row = {"arm": name, "update": update, "loss_mean": float(np.mean(losses)),
                   "elapsed_seconds": time.perf_counter() - started,
                   **{key: float(value.detach()) for key, value in terms.items()}}
            trace.append(row); losses.clear(); print(json.dumps(row, sort_keys=True), flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    endpoint = output_dir / f"{name}_endpoint.pth"
    torch.save({"schema": SCHEMA, "arm": name, "updates": UPDATES,
                "model_config": model.architecture_config(), "model_state_dict": state}, endpoint)
    return {"path": str(endpoint), "sha256": sha256_file(endpoint),
            "model_state_sha256": state_sha(state), "trace": trace,
            "wall_seconds": time.perf_counter() - started}


def error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((SCORING.rel_l2_per_sample(prediction, target, 2),
                     SCORING.tke_rel_l2_per_sample(prediction, target, 2),
                     SCORING.mvpe_rel_l2_per_sample(prediction, target, 2)), axis=1).astype(np.float64)


def score(errors: np.ndarray) -> dict[str, float | int]:
    mean = errors.mean(0, dtype=np.float64)
    result: dict[str, float | int] = {"samples": int(len(errors)),
        "rel_l2_score": SCORING.score_error(float(mean[0])),
        "tke_score": SCORING.score_error(float(mean[1])),
        "mvpe_score": SCORING.score_error(float(mean[2]))}
    result["point_contribution_46_10_9"] = sum(POINT_WEIGHTS[key] * float(result[key]) for key in POINT_WEIGHTS)
    return result


def metric_delta(candidate: dict[str, float | int], control: dict[str, float | int]) -> dict[str, float]:
    return {key: float(candidate[key]) - float(control[key])
            for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}


def comparison(candidate: str, control: str, errors: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, object]:
    delta = metric_delta(score(errors[candidate]), score(errors[control]))
    per_file = {name: float(score(errors[candidate][labels == name])["point_contribution_46_10_9"]
                            - score(errors[control][labels == name])["point_contribution_46_10_9"])
                for name in sorted(set(labels.tolist()))}
    return {"delta": delta, "per_trajectory_file_delta_point": per_file,
            "positive_trajectory_files": sum(value > 0 for value in per_file.values()),
            "worst_trajectory_delta_point": min(per_file.values())}


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().cpu())


def evaluate_arms(arms: dict[str, CausalStateMultibandForecaster], trajectories, windows,
                  window_files: list[str], normalizer, device: torch.device) -> dict[str, object]:
    dataset = backbone.common.scratch.SimWindowDataset(trajectories, windows, mask_probability=0., noise_scale=0.)
    errors = {name: [] for name in ARM_NAMES}
    ablation_errors = {"independent_same_checkpoint": [], "frozen_same_checkpoint": [],
                       "drop_low": [], "drop_mid": [], "drop_high": [],
                       "control_override_cumulative": []}
    labels, audits = [], {"native_vs_independent_state_rms": [],
                           "native_vs_independent_prediction_rms": [],
                           "pairwise_band_inner_products": []}
    candidate = arms["cumulative_candidate"]
    for start in range(0, len(dataset), 32):
        indexes = range(start, min(start + 32, len(dataset)))
        samples = [dataset[index] for index in indexes]
        x, target = torch.stack([sample[0] for sample in samples]).to(device), torch.stack([sample[1] for sample in samples])
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        with torch.inference_mode():
            for name, model in arms.items():
                model.eval()
                prediction = model(x_norm)
                physical = (prediction * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
                errors[name].append(error_rows(physical, target.numpy()))
            parts = candidate.forward_components(x_norm)
            independent_parts = candidate.forward_components(x_norm, state_mode="independent")
            frozen_parts = candidate.forward_components(x_norm, state_mode="frozen")
            alternatives = {
                "independent_same_checkpoint": independent_parts["prediction"],
                "frozen_same_checkpoint": frozen_parts["prediction"],
                "drop_low": parts["prediction"] - parts["low"],
                "drop_mid": parts["prediction"] - parts["mid"],
                "drop_high": parts["prediction"] - parts["high"],
                "control_override_cumulative": arms["independent_control"].forward_components(
                    x_norm, state_mode="cumulative")["prediction"],
            }
            for name, prediction in alternatives.items():
                physical = (prediction * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
                ablation_errors[name].append(error_rows(physical, target.numpy()))
            audits["native_vs_independent_state_rms"].append(_rms(
                parts["future_states"] - independent_parts["future_states"]))
            audits["native_vs_independent_prediction_rms"].append(_rms(
                parts["prediction"] - independent_parts["prediction"]))
            bands = [parts[name] for name in ("low", "mid", "high")]
            audits["pairwise_band_inner_products"].append([
                float((bands[left].float() * bands[right].float()).mean().cpu())
                for left, right in ((0, 1), (0, 2), (1, 2))])
        labels.extend(window_files[index] for index in indexes)
    joined = {name: np.concatenate(value) for name, value in errors.items()}
    joined_ablations = {name: np.concatenate(value) for name, value in ablation_errors.items()}
    labels_array = np.asarray(labels)
    candidate_score = score(joined["cumulative_candidate"])
    return {"metrics": {name: score(value) for name, value in joined.items()},
            "cumulative_candidate_minus_independent_control": comparison(
                "cumulative_candidate", "independent_control", joined, labels_array),
            "same_checkpoint_state_ablations": {
                "native": candidate_score,
                "independent": score(joined_ablations["independent_same_checkpoint"]),
                "frozen": score(joined_ablations["frozen_same_checkpoint"]),
                "native_minus_independent": metric_delta(
                    candidate_score, score(joined_ablations["independent_same_checkpoint"])),
                "native_minus_frozen": metric_delta(
                    candidate_score, score(joined_ablations["frozen_same_checkpoint"]))},
            "band_drop_diagnostics": {
                name: {
                    "dropped": score(joined_ablations[f"drop_{name}"]),
                    "native_minus_dropped": metric_delta(
                        candidate_score, score(joined_ablations[f"drop_{name}"])),
                }
                for name in ("low", "mid", "high")},
            "control_checkpoint_cumulative_override": score(joined_ablations["control_override_cumulative"]),
            "mechanical": {"native_vs_independent_state_rms": float(np.mean(audits["native_vs_independent_state_rms"])),
                           "native_vs_independent_prediction_rms": float(np.mean(audits["native_vs_independent_prediction_rms"])),
                           "pairwise_band_inner_product_mean": np.mean(audits["pairwise_band_inner_products"], axis=0).tolist()}}


def gate(evaluation: dict[str, object]) -> tuple[bool, dict[str, object]]:
    primary = evaluation["cumulative_candidate_minus_independent_control"]
    delta = primary["delta"]
    ablation = evaluation["same_checkpoint_state_ablations"]["native_minus_independent"]
    mechanical = evaluation["mechanical"]
    passed = (float(delta["point_contribution_46_10_9"]) >= .05
              and all(float(delta[key]) >= -.03 for key in POINT_WEIGHTS)
              and int(primary["positive_trajectory_files"]) >= 15
              and float(primary["worst_trajectory_delta_point"]) >= -.05
              and float(ablation["point_contribution_46_10_9"]) >= .01
              and float(mechanical["native_vs_independent_state_rms"]) > STATE_RMS_MIN
              and float(mechanical["native_vs_independent_prediction_rms"]) > PREDICTION_RMS_MIN)
    return passed, {"primary_point_min": .05, "component_floor": -.03,
                    "positive_trajectory_files_min": 15, "worst_trajectory_delta_point_min": -.05,
                    "same_checkpoint_native_minus_independent_point_min": .01,
                    "native_vs_independent_state_rms_min": STATE_RMS_MIN,
                    "native_vs_independent_prediction_rms_min": PREDICTION_RMS_MIN,
                    "frozen_and_band_drop": "diagnostic_only"}


def prior_held10_evidence(path: Path | None, held_aoa: int) -> dict[str, object] | None:
    if held_aoa == 10:
        if path is not None:
            raise ValueError("held10 takes no prior result")
        return None
    if path is None or not path.is_file():
        raise PermissionError("held20 requires a held10 GO result")
    prior = json.loads(path.read_text(encoding="utf-8"))
    fixed = {"schema": SCHEMA, "held_aoa": 10, "updates": UPDATES, "batch_size": BATCH_SIZE,
             "seed": SEED, "state_channels": STATE_CHANNELS, "decoder_channels": DECODER_CHANNELS,
             "context_channels": CONTEXT_CHANNELS, "groups": GROUPS, "arms": list(ARM_NAMES),
             "expected_parameters_per_arm": EXPECTED_PARAMETERS,
             "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
             "accuracy_channel_weights": list(ACCURACY_CHANNEL_WEIGHTS),
             "mse_weight": MSE_WEIGHT,
             "training_script_sha256": sha256_file(Path(__file__)),
             "model_source_sha256": sha256_file(
                 CODE_ROOT / "models" / "causal_state_multiband.py")}
    try:
        valid = (prior["schema"] == SCHEMA and prior["decision"] == "GO_HELD20"
                 and all(prior["config"].get(key) == value for key, value in fixed.items())
                 and gate(prior["evaluation"])[0])
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid or prior.get("git", {}).get("clean") is not True:
        raise PermissionError("prior file is not a verified held10 GO")
    seal_record = prior.get("endpoint_seal", {})
    seal_path = Path(str(seal_record.get("path", "")))
    if not seal_path.is_file() or sha256_file(seal_path) != seal_record.get("sha256"):
        raise PermissionError("held10 endpoint seal is missing or changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    artifacts = [seal.get("fit_only_normalizer", {})]
    artifacts.extend(seal.get("endpoints", {}).get(name, {}) for name in ARM_NAMES)
    for artifact in artifacts:
        artifact_path = Path(str(artifact.get("path", "")))
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256"):
            raise PermissionError("a held10 sealed artifact is missing or changed")
    return {"path": str(path), "sha256": sha256_file(path), "decision": "GO_HELD20"}


def git_report() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--untracked-files=all")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "clean": not bool(dirty), "porcelain": dirty.splitlines()}


def smoke(device: torch.device) -> None:
    arms, identity = construct_arms(device)
    x, y = torch.randn(2, 20, 32, 64, 3, device=device), torch.randn(2, 20, 32, 64, 3, device=device)
    for model in arms.values():
        model.train(); output = model(x); output.sub(y).square().mean().backward()
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite smoke output")
        model.zero_grad(set_to_none=True)
    print(json.dumps({"smoke": "PASS", "identity": identity}, sort_keys=True))


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != "CSMR_WHOLE_AOA_V1":
        raise PermissionError("formal run requires --execute and protocol acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    prior, git = prior_held10_evidence(args.prior_held10_result, args.held_aoa), git_report()
    if not git["clean"]:
        raise RuntimeError(f"worktree must be clean: {git['porcelain']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = scan_inventory(args.sim_dir)
    fit_records = [row for row in inventory if int(row["aoa"]) != args.held_aoa]
    held_records = [row for row in inventory if int(row["aoa"]) == args.held_aoa]
    if len(fit_records) != 80 or len(held_records) != 20:
        raise RuntimeError("unexpected whole-AoA inventory")
    config = {"schema": SCHEMA, "held_aoa": args.held_aoa, "fit_aoas": [aoa for aoa in AOAS if aoa != args.held_aoa],
              "updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED, "state_channels": STATE_CHANNELS,
              "decoder_channels": DECODER_CHANNELS, "context_channels": CONTEXT_CHANNELS, "groups": GROUPS,
              "arms": list(ARM_NAMES), "expected_parameters_per_arm": EXPECTED_PARAMETERS,
              "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
              "accuracy_channel_weights": list(ACCURACY_CHANNEL_WEIGHTS),
              "mse_weight": MSE_WEIGHT,
              "training_script_sha256": sha256_file(Path(__file__)),
              "model_source_sha256": sha256_file(
                  CODE_ROOT / "models" / "causal_state_multiband.py"),
              "fit_inventory_sha256": canonical_sha(fit_records),
              "held_inventory_sha256": canonical_sha(held_records), "prior_held10_evidence": prior,
              "held_field_values_loaded_before_all_endpoint_seals": False}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    (args.output_dir / "environment.txt").write_text("\n".join(f"{key}={value}" for key, value in {
        "python": sys.version.replace("\n", " "), "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "pid": os.getpid()}.items()) + "\n", encoding="utf-8")
    fit_trajectories, fit_windows, _ = load_fields(args.sim_dir, fit_records)
    if len(fit_windows) != 3_840:
        raise RuntimeError("unexpected fit window count")
    normalizer = backbone.common.scratch.ChannelNormalizer(*backbone.common.scratch.window_statistics(fit_trajectories, fit_windows), device=device)
    normalizer_state = {key: getattr(normalizer, key).detach().cpu() for key in ("mean_input", "mean_target", "std_input", "std_target")}
    normalizer_path = args.output_dir / "fit_only_normalizer.pt"; torch.save({"schema": SCHEMA, **normalizer_state}, normalizer_path)
    normalizer_artifact = {"path": str(normalizer_path), "sha256": sha256_file(normalizer_path), "state_sha256": state_sha(normalizer_state)}
    dataset = backbone.common.scratch.SimWindowDataset(fit_trajectories, fit_windows, mask_probability=0., noise_scale=0.)
    schedule, arms = make_schedule(fit_windows), construct_arms(device)
    models, identity = arms
    endpoints = {name: train_arm(name, models[name], dataset, schedule, normalizer, device, args.output_dir) for name in ARM_NAMES}
    seal = {"schema": SCHEMA, "fit_only_normalizer": normalizer_artifact,
            "endpoints": {name: {key: endpoints[name][key] for key in ("path", "sha256", "model_state_sha256")} for name in ARM_NAMES}}
    seal_path = args.output_dir / "all_endpoints_seal.json"; seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Materialize the seal digest before the first held field value is read.
    seal_sha256 = sha256_file(seal_path)
    held_trajectories, held_windows, held_files = load_fields(args.sim_dir, held_records)
    if len(held_windows) != 960:
        raise RuntimeError("unexpected held window count")
    evaluation = evaluate_arms(models, held_trajectories, held_windows, held_files, normalizer, device)
    passed, gate_config = gate(evaluation)
    decision = "GO_HELD20" if args.held_aoa == 10 and passed else ("COMPLETE_PROMISING" if passed else "STOP_CSMR")
    result = {"schema": SCHEMA, "evidence_class": "whole_AoA_held_Sim", "config": config, "git": git,
              "initial_identity": identity, "fit_trajectories": len(fit_records), "fit_windows": len(fit_windows),
              "held_trajectories": len(held_records), "held_windows": len(held_windows),
              "sample_stream_sha256_all_arms": hashlib.sha256(schedule.tobytes()).hexdigest(),
              "fit_only_normalizer": normalizer_artifact, "endpoints": endpoints,
              "endpoint_seal": {"path": str(seal_path), "sha256": seal_sha256},
              "evaluation": evaluation, "gate": gate_config, "decision": decision}
    result_path = args.output_dir / "result.json"; result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "result": str(result_path),
                      "candidate_minus_control": evaluation["cumulative_candidate_minus_independent_control"]["delta"]}, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this discovery screen requires CUDA")
    smoke(device) if args.smoke else run(args, device)


if __name__ == "__main__":
    main()
