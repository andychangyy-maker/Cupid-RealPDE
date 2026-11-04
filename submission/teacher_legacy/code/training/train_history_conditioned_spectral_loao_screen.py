#!/usr/bin/env python3
"""Sealed whole-AoA discovery screen for a history-conditioned operator.

Three arms share every parameter tensor, initial state, fit stream, optimizer,
weight decay, scheduler, and forward topology.  ``static_rank`` is the
primary matched control: its low-rank spectral correction is driven by a
learned input-independent gate.  ``conditional_rank`` differs only in that
the already-computed history conditioner contributes to that gate.  The
static-base arm is an auxiliary capacity attribution control.

This script is point-only.  It neither trains uncertainty nor reads held flow
arrays before fit-only normalizer and all three endpoints have been sealed.
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
from models.history_conditioned_spectral_operator import (  # noqa: E402
    ARM_MODES,
    HistoryConditionedSpectralConfig,
    HistoryConditionedSpectralOperator,
)

SCHEMA = "history_conditioned_spectral_whole_aoa_sim_screen_v1"
AOAS = (0, 5, 10, 15, 20)
ARM_NAMES = ARM_MODES
SEED, UPDATES, BATCH_SIZE = 210, 2_000, 8
WIDTH, LAYERS, MODES_Y, MODES_X, RANK = 32, 4, 8, 14, 4
CONDITION_HIDDEN, GROUPS = 64, 8
EXPECTED_PARAMETERS = 1_189_764
EXPECTED_COMPLEX_PARAMETERS = 1_146_880
EXPECTED_REAL_SCALAR_DEGREES = 2_336_644
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}
CONDITION_DISABLED_POINT_MIN = .01
CONDITION_SHUFFLED_POINT_MIN = .005

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


def make_config() -> HistoryConditionedSpectralConfig:
    return HistoryConditionedSpectralConfig(width=WIDTH, layers=LAYERS,
                                             modes_y=MODES_Y, modes_x=MODES_X,
                                             rank=RANK, condition_hidden=CONDITION_HIDDEN,
                                             groups=GROUPS)


def construct_arms(device: torch.device) -> tuple[dict[str, HistoryConditionedSpectralOperator], dict[str, object]]:
    arms: dict[str, HistoryConditionedSpectralOperator] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = HistoryConditionedSpectralOperator(make_config(), name).to(device)
        arms[name] = model
        states[name] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    reference = states["static_base"]
    if any(reference.keys() != state.keys() or any(not torch.equal(reference[key], state[key])
           for key in reference) for state in states.values()):
        raise RuntimeError("three arms do not have bitwise-identical initial state")
    counts = {name: sum(parameter.numel() for parameter in model.parameters())
              for name, model in arms.items()}
    if (len(set(counts.values())) != 1
            or any(value != EXPECTED_PARAMETERS for value in counts.values())):
        raise RuntimeError(f"parameter mismatch: {counts}")
    trainable = {name: sum(parameter.numel() for parameter in model.parameters()
                           if parameter.requires_grad)
                 for name, model in arms.items()}
    if trainable != counts:
        raise RuntimeError(f"requires-grad contract mismatch: {trainable}")
    complex_parameters = sum(parameter.numel() for parameter in arms[
        "static_base"].parameters() if parameter.is_complex())
    real_scalar_degrees = (2 * complex_parameters
                           + EXPECTED_PARAMETERS - complex_parameters)
    if (complex_parameters != EXPECTED_COMPLEX_PARAMETERS
            or real_scalar_degrees != EXPECTED_REAL_SCALAR_DEGREES):
        raise RuntimeError("complex parameter contract changed")
    return arms, {"state_sha256": state_sha(reference), "parameters": counts,
                  "trainable_parameters": trainable,
                  "complex_parameters": complex_parameters,
                  "real_scalar_degrees": real_scalar_degrees,
                  "state_tensors": len(reference), "bitwise_identical": True,
                  "primary_comparison": "conditional_rank_minus_static_rank",
                  "auxiliary_comparison": "conditional_rank_minus_static_base"}


def make_schedule(windows: list[object]) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, len(windows), size=(UPDATES, BATCH_SIZE), dtype=np.int64)


def batches(dataset, schedule: np.ndarray):
    for row in schedule:
        values = [dataset[int(index)] for index in row]
        yield torch.stack([value[0] for value in values]), torch.stack([value[1] for value in values])


def loss_terms(model: HistoryConditionedSpectralOperator, x_norm: torch.Tensor,
               y_norm: torch.Tensor, normalizer) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = model(x_norm)
    accuracy, detail = backbone.common.accuracy_objective(prediction, y_norm, normalizer, (1., 4., 1.))
    mse = (prediction - y_norm).square().mean()
    loss = accuracy + .25 * mse
    return loss, {"accuracy": accuracy, "mse": mse, "loss": loss, **detail}


def train_arm(name: str, model: HistoryConditionedSpectralOperator, dataset,
              schedule: np.ndarray, normalizer, device: torch.device,
              output_dir: Path) -> dict[str, object]:
    backbone.common.set_model_stats(model, normalizer)
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
        x_norm, y_norm = normalizer.preprocess(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss, terms = loss_terms(model, x_norm, y_norm, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} non-finite loss at update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            row = {"arm": name, "update": update, "loss_mean": float(np.mean(losses)),
                   "loss_median": float(np.median(losses)),
                   "elapsed_seconds": time.perf_counter() - started,
                   **{key: float(value.detach()) for key, value in terms.items()}}
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            losses.clear()
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
    result["point_contribution_46_10_9"] = sum(POINT_WEIGHTS[key] * float(result[key])
                                                  for key in POINT_WEIGHTS)
    return result


def _comparison(candidate: str, control: str, joined: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, object]:
    candidate_metrics, control_metrics = score(joined[candidate]), score(joined[control])
    delta = {key: float(candidate_metrics[key]) - float(control_metrics[key])
             for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}
    per_file = {file: float(score(joined[candidate][labels == file])["point_contribution_46_10_9"]
                            - score(joined[control][labels == file])["point_contribution_46_10_9"])
                for file in sorted(set(labels.tolist()))}
    return {"delta": delta, "per_trajectory_file_delta_point": per_file,
            "positive_trajectory_files": sum(value > 0 for value in per_file.values()),
            "worst_trajectory_delta_point": min(per_file.values())}


def metric_delta(candidate: dict[str, float | int],
                 control: dict[str, float | int]) -> dict[str, float]:
    return {key: float(candidate[key]) - float(control[key])
            for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}


def cross_trajectory_shuffle_index(window_files: list[str]) -> np.ndarray:
    """Map every window to the same ordinal window in the next trajectory."""
    indexes_by_file: dict[str, list[int]] = {}
    for index, file in enumerate(window_files):
        indexes_by_file.setdefault(file, []).append(index)
    files = sorted(indexes_by_file)
    if len(files) < 2 or len({len(indexes_by_file[file]) for file in files}) != 1:
        raise ValueError("cross-trajectory gate shuffle requires balanced trajectories")
    mapping = np.empty(len(window_files), dtype=np.int64)
    for position, file in enumerate(files):
        source = files[(position + 1) % len(files)]
        mapping[indexes_by_file[file]] = indexes_by_file[source]
    if any(window_files[index] == window_files[int(source)]
           for index, source in enumerate(mapping)):
        raise RuntimeError("gate shuffle did not cross trajectories")
    return mapping


def evaluate_arms(arms: dict[str, HistoryConditionedSpectralOperator], trajectories, windows,
                  window_files: list[str], normalizer, device: torch.device) -> dict[str, object]:
    dataset = backbone.common.scratch.SimWindowDataset(trajectories, windows, mask_probability=0., noise_scale=0.)
    errors = {name: [] for name in ARM_NAMES}
    condition_gates = []
    native_gates = []
    files: list[str] = []
    for start in range(0, len(dataset), 64):
        indexes = range(start, min(start + 64, len(dataset)))
        samples = [dataset[index] for index in indexes]
        x = torch.stack([sample[0] for sample in samples]).to(device)
        target = torch.stack([sample[1] for sample in samples])
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        for name, model in arms.items():
            model.eval()
            with torch.inference_mode():
                values = model.forward_components(x_norm)
                prediction = (values["prediction"] * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
                if name == "conditional_rank":
                    condition_gates.append(values["condition_gate"].float().cpu())
                    native_gates.append(values["gate"].float().cpu())
            errors[name].append(error_rows(prediction, target.numpy()))
        files.extend(window_files[index] for index in indexes)
    joined = {name: np.concatenate(value) for name, value in errors.items()}
    labels = np.asarray(files)
    condition_gate = torch.cat(condition_gates)
    native_gate = torch.cat(native_gates)
    shuffle_index = cross_trajectory_shuffle_index(window_files)
    shuffled_gate = native_gate[torch.from_numpy(shuffle_index)]
    ablation_errors: dict[str, list[np.ndarray]] = {
        "condition_disabled": [], "cross_trajectory_shuffled": []}
    candidate = arms["conditional_rank"]
    for start in range(0, len(dataset), 64):
        indexes = range(start, min(start + 64, len(dataset)))
        samples = [dataset[index] for index in indexes]
        x = torch.stack([sample[0] for sample in samples]).to(device)
        target = torch.stack([sample[1] for sample in samples])
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        count = len(samples)
        disabled_gate = torch.tanh(candidate.static_gate.detach()).view(
            1, -1).expand(count, -1)
        overrides = {
            "condition_disabled": disabled_gate,
            "cross_trajectory_shuffled": shuffled_gate[start:start + count].to(device),
        }
        candidate.eval()
        with torch.inference_mode():
            for name, override in overrides.items():
                prediction = candidate.forward_components(
                    x_norm, gate_override=override)["prediction"]
                physical = (prediction * normalizer.std_target
                            + normalizer.mean_target).float().cpu().numpy()
                ablation_errors[name].append(error_rows(physical, target.numpy()))
    ablation_scores = {name: score(np.concatenate(value))
                       for name, value in ablation_errors.items()}
    native_score = score(joined["conditional_rank"])
    return {"metrics": {name: score(value) for name, value in joined.items()},
            "conditional_rank_minus_static_rank": _comparison("conditional_rank", "static_rank", joined, labels),
            "conditional_rank_minus_static_base": _comparison("conditional_rank", "static_base", joined, labels),
            "conditional_gate": {"per_rank_std_mean": float(condition_gate.std(dim=0).mean()),
                                  "actual_gate_per_rank_std_mean": float(native_gate.std(dim=0).mean()),
                                  "per_rank_abs_mean": float(condition_gate.abs().mean()),
                                  "samples": int(len(condition_gate)),
                                  "cross_trajectory_shuffle_index_sha256": hashlib.sha256(
                                      shuffle_index.tobytes()).hexdigest()},
            "conditional_checkpoint_ablations": {
                "native": native_score, **ablation_scores,
                "native_minus_condition_disabled": metric_delta(
                    native_score, ablation_scores["condition_disabled"]),
                "native_minus_cross_trajectory_shuffled": metric_delta(
                    native_score, ablation_scores["cross_trajectory_shuffled"])}}


def gate(evaluation: dict[str, object]) -> tuple[bool, dict[str, object]]:
    primary = evaluation["conditional_rank_minus_static_rank"]
    auxiliary = evaluation["conditional_rank_minus_static_base"]
    ablations = evaluation["conditional_checkpoint_ablations"]
    delta = primary["delta"]
    passed = (float(delta["point_contribution_46_10_9"]) >= .05
              and all(float(delta[key]) >= -.03 for key in POINT_WEIGHTS)
              and int(primary["positive_trajectory_files"]) >= 15
              and float(primary["worst_trajectory_delta_point"]) >= -.05
              and float(auxiliary["delta"]["point_contribution_46_10_9"]) >= .03
              and float(evaluation["conditional_gate"]["actual_gate_per_rank_std_mean"]) > 1.0e-5
              and float(ablations["native_minus_condition_disabled"][
                  "point_contribution_46_10_9"]) >= CONDITION_DISABLED_POINT_MIN
              and float(ablations["native_minus_cross_trajectory_shuffled"][
                  "point_contribution_46_10_9"]) >= CONDITION_SHUFFLED_POINT_MIN)
    return passed, {"primary": "conditional_rank_minus_static_rank", "primary_point_min": .05,
                    "component_floor": -.03, "positive_trajectory_files_min": 15,
                    "worst_trajectory_delta_point_min": -.05,
                    "auxiliary_conditional_rank_minus_static_base_point_min": .03,
                    "conditional_gate_per_rank_std_mean_min": 1.0e-5,
                    "native_minus_condition_disabled_point_min": CONDITION_DISABLED_POINT_MIN,
                    "native_minus_cross_trajectory_shuffled_point_min": CONDITION_SHUFFLED_POINT_MIN}


def prior_held10_evidence(path: Path | None, held_aoa: int) -> dict[str, object] | None:
    if held_aoa == 10:
        if path is not None:
            raise ValueError("held10 takes no prior result")
        return None
    if path is None or not path.is_file():
        raise PermissionError("held20 requires a held10 GO result")
    prior = json.loads(path.read_text(encoding="utf-8"))
    fixed = {"held_aoa": 10, "updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED,
             "width": WIDTH, "layers": LAYERS, "modes_y": MODES_Y, "modes_x": MODES_X,
             "rank": RANK, "condition_hidden": CONDITION_HIDDEN,
             "condition_disabled_point_min": CONDITION_DISABLED_POINT_MIN,
             "condition_shuffled_point_min": CONDITION_SHUFFLED_POINT_MIN}
    valid_gate = False
    try:
        valid_gate = gate(prior["evaluation"])[0]
    except (KeyError, TypeError, ValueError):
        pass
    seal_record = prior.get("endpoint_seal", {})
    seal_path = Path(str(seal_record.get("path", "")))
    if (prior.get("schema") != SCHEMA or prior.get("decision") != "GO_HELD20"
            or prior.get("git", {}).get("clean") is not True
            or not all(prior.get("config", {}).get(key) == value
                       for key, value in fixed.items())
            or not valid_gate or not seal_path.is_file()
            or sha256_file(seal_path) != seal_record.get("sha256")):
        raise PermissionError("prior file is not a verified held10 GO")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    artifacts = [seal.get("fit_only_normalizer", {})]
    artifacts.extend(seal.get("endpoints", {}).get(name, {}) for name in ARM_NAMES)
    for artifact in artifacts:
        artifact_path = Path(str(artifact.get("path", "")))
        if (not artifact_path.is_file()
                or sha256_file(artifact_path) != artifact.get("sha256")):
            raise PermissionError("a held10 sealed artifact is missing or changed")
    return {"path": str(path), "sha256": sha256_file(path), "decision": "GO_HELD20"}


def smoke(device: torch.device) -> None:
    arms, identity = construct_arms(device)
    x, y = torch.randn(2, 20, 32, 64, 3, device=device), torch.randn(2, 20, 32, 64, 3, device=device)
    for model in arms.values():
        model.train()
        output = model(x)
        output.sub(y).square().mean().backward()
        assert torch.isfinite(output).all()
        model.zero_grad(set_to_none=True)
    print(json.dumps({"smoke": "PASS", "identity": identity}, sort_keys=True))


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != "HCSO_WHOLE_AOA_V1":
        raise PermissionError("formal run requires --execute and protocol acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    prior = prior_held10_evidence(args.prior_held10_result, args.held_aoa)
    git = git_report()
    if not git["clean"]:
        raise RuntimeError(f"worktree must be clean: {git['porcelain']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = scan_inventory(args.sim_dir)
    fit_records = [row for row in inventory if int(row["aoa"]) != args.held_aoa]
    held_records = [row for row in inventory if int(row["aoa"]) == args.held_aoa]
    if len(fit_records) != 80 or len(held_records) != 20:
        raise RuntimeError("unexpected whole-AoA inventory")
    config = {"held_aoa": args.held_aoa, "fit_aoas": [aoa for aoa in AOAS if aoa != args.held_aoa],
              "updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED, "width": WIDTH,
              "layers": LAYERS, "modes_y": MODES_Y, "modes_x": MODES_X, "rank": RANK,
              "condition_hidden": CONDITION_HIDDEN, "arms": list(ARM_NAMES),
              "condition_disabled_point_min": CONDITION_DISABLED_POINT_MIN,
              "condition_shuffled_point_min": CONDITION_SHUFFLED_POINT_MIN,
              "fit_inventory_sha256": canonical_sha(fit_records), "held_inventory_sha256": canonical_sha(held_records),
              "prior_held10_evidence": prior, "held_field_values_loaded_before_all_endpoint_seals": False}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    (args.output_dir / "environment.txt").write_text("\n".join(f"{key}={value}" for key, value in {
        "python": sys.version.replace("\n", " "), "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "pid": os.getpid()}.items()) + "\n", encoding="utf-8")
    fit_trajectories, fit_windows, _ = load_fields(args.sim_dir, fit_records)
    if len(fit_records) != 80 or len(fit_windows) != 3_840:
        raise RuntimeError("unexpected fit inventory")
    normalizer = backbone.common.scratch.ChannelNormalizer(*backbone.common.scratch.window_statistics(fit_trajectories, fit_windows), device=device)
    normalizer_path = args.output_dir / "fit_only_normalizer.pt"
    normalizer_state = {key: getattr(normalizer, key).detach().cpu() for key in ("mean_input", "mean_target", "std_input", "std_target")}
    torch.save({"schema": SCHEMA, **normalizer_state}, normalizer_path)
    normalizer_artifact = {"path": str(normalizer_path), "sha256": sha256_file(normalizer_path),
                           "state_sha256": state_sha(normalizer_state)}
    dataset = backbone.common.scratch.SimWindowDataset(fit_trajectories, fit_windows, mask_probability=0., noise_scale=0.)
    schedule = make_schedule(fit_windows)
    arms, identity = construct_arms(device)
    endpoints = {name: train_arm(name, arms[name], dataset, schedule, normalizer, device, args.output_dir)
                 for name in ARM_NAMES}
    seal = {"schema": SCHEMA, "fit_only_normalizer": normalizer_artifact,
            "endpoints": {name: {key: endpoints[name][key] for key in ("path", "sha256", "model_state_sha256")}
                          for name in ARM_NAMES}}
    seal_path = args.output_dir / "all_endpoints_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_sha256 = sha256_file(seal_path)
    held_trajectories, held_windows, held_files = load_fields(args.sim_dir, held_records)
    if len(held_records) != 20 or len(held_windows) != 960:
        raise RuntimeError("unexpected held inventory")
    evaluation = evaluate_arms(arms, held_trajectories, held_windows, held_files, normalizer, device)
    passed, gate_config = gate(evaluation)
    decision = "GO_HELD20" if args.held_aoa == 10 and passed else ("COMPLETE_PROMISING" if passed else "STOP_HCSO")
    result = {"schema": SCHEMA, "evidence_class": "whole_AoA_held_Sim", "config": config, "git": git,
              "initial_identity": identity, "fit_trajectories": len(fit_records), "fit_windows": len(fit_windows),
              "held_trajectories": len(held_records), "held_windows": len(held_windows),
              "sample_stream_sha256_all_arms": hashlib.sha256(schedule.tobytes()).hexdigest(),
              "fit_only_normalizer": normalizer_artifact, "endpoints": endpoints,
              "endpoint_seal": {"path": str(seal_path), "sha256": seal_sha256},
              "evaluation": evaluation, "gate": gate_config, "decision": decision}
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "result": str(result_path),
                      "conditional_minus_static_rank": evaluation["conditional_rank_minus_static_rank"]["delta"]}, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this discovery screen requires CUDA")
    smoke(device) if args.smoke else run(args, device)


if __name__ == "__main__":
    main()
