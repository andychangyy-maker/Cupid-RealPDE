#!/usr/bin/env python3
"""Sealed whole-AoA CVPS screen: signed local correspondence versus radial control.

The only arm variable is the fixed dense signed/radial bin map.  Both arms
form lag-4/8/16-minus-lag-1 delta volumes and execute the same K-by-K fixed
operation before their shared learned fusion.  Held fields are loaded only
after all fit-only artifacts and endpoint hashes seal.
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
from models.correlation_volume_predictive_state import (  # noqa: E402
    ARM_MODES, CorrelationVolumePredictiveState, CorrelationVolumePredictiveStateConfig)

SCHEMA = "correlation_volume_predictive_state_whole_aoa_sim_screen_v1"
AOAS, ARM_NAMES = (0, 5, 10, 15, 20), ("radial_control", "signed_candidate")
SEED, UPDATES, BATCH_SIZE = 210, 2_000, 8
LEARNING_RATE, WEIGHT_DECAY, MSE_WEIGHT = 3e-4, 1e-5, .25
ACCURACY_CHANNEL_WEIGHTS = (1., 4., 1.)
EXPECTED_PARAMETERS = 1_257_225
MODEL_CONTRACT = {"reference_lag": 1, "motion_lags": [4, 8, 16], "radius": 4,
                  "feature_channels": 48, "state_channels": 96,
                  "decoder_channels": 64, "groups": 8}
MATCHING_GRID, STATE_GRID = (16, 32), (8, 16)
GRAD_CLIP, TRAINING_RNG_SEED = 1., SEED + 17
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}
SCORING = temporal_screen.SCORING
scan_inventory, load_fields = temporal_screen.scan_inventory, temporal_screen.load_fields
sha256_file, canonical_sha, state_sha = (temporal_screen.sha256_file,
                                         temporal_screen.canonical_sha,
                                         temporal_screen.state_sha)
SOURCE_PATHS = {
    "trainer": Path(__file__),
    "model": CODE_ROOT / "models" / "correlation_volume_predictive_state.py",
    "band_decoder": CODE_ROOT / "models" / "causal_state_multiband.py",
    "flow_backbone_driver": CODE_ROOT / "training" / "train_flow_backbone.py",
    "loss_and_normalizer_driver": CODE_ROOT / "training" / "train_hybrid_fno2d.py",
    "scratch_dataset": CODE_ROOT / "training" / "train_fno_sim_from_scratch.py",
    "sealed_io_driver": CODE_ROOT / "training" / "train_temporal_transport_loao_screen.py",
    "official_v9_scoring": REPO_ROOT / "kits" / "track1" / "realpde_t1_starting_kit_v9" / "scoring.py",
}


def source_hashes() -> dict[str, str]:
    """Hash every local source that can change training or evaluation semantics."""
    return {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}


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


def make_config() -> CorrelationVolumePredictiveStateConfig:
    return CorrelationVolumePredictiveStateConfig()


def construct_arms(device: torch.device) -> tuple[dict[str, CorrelationVolumePredictiveState], dict[str, object]]:
    arms, snapshots = {}, {}
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = CorrelationVolumePredictiveState(make_config(), name).to(device)
        arms[name] = model
        snapshots[name] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    reference = snapshots[ARM_NAMES[0]]
    if any(reference.keys() != state.keys() or any(not torch.equal(reference[key], state[key])
           for key in reference) for state in snapshots.values()):
        raise RuntimeError("arms do not have bitwise-identical initial state")
    parameters = {name: sum(value.numel() for value in model.parameters()) for name, model in arms.items()}
    if set(parameters.values()) != {EXPECTED_PARAMETERS}:
        raise RuntimeError(f"parameter contract changed: {parameters}")
    return arms, {"parameters": parameters, "trainable_parameters": parameters,
                  "state_sha256": state_sha(reference), "bitwise_identical": True,
                  "primary_comparison": "signed_candidate_minus_radial_control",
                  "only_variable": "post_projection_bin_geometry",
                  "arm_modes": list(ARM_MODES)}


def make_schedule(windows: list[object]) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, len(windows), size=(UPDATES, BATCH_SIZE), dtype=np.int64)


def batches(dataset, schedule: np.ndarray):
    for row in schedule:
        values = [dataset[int(index)] for index in row]
        yield torch.stack([value[0] for value in values]), torch.stack([value[1] for value in values])


def loss_terms(model: CorrelationVolumePredictiveState, x_norm: torch.Tensor,
               y_norm: torch.Tensor, normalizer) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = model(x_norm)
    accuracy, detail = backbone.common.accuracy_objective(prediction, y_norm, normalizer, ACCURACY_CHANNEL_WEIGHTS)
    mse = (prediction - y_norm).square().mean()
    loss = accuracy + MSE_WEIGHT * mse
    return loss, {"accuracy": accuracy, "mse": mse, "loss": loss, **detail}


def train_arm(name: str, model: CorrelationVolumePredictiveState, dataset, schedule: np.ndarray,
              normalizer, device: torch.device, output_dir: Path) -> dict[str, object]:
    backbone.common.set_model_stats(model, normalizer)
    torch.manual_seed(TRAINING_RNG_SEED)
    if device.type == "cuda": torch.cuda.manual_seed_all(TRAINING_RNG_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    trace, losses, started = [], [], time.perf_counter()
    for update, (x, y) in enumerate(batches(dataset, schedule), 1):
        model.train(); x_norm, y_norm = normalizer.preprocess(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True); loss, terms = loss_terms(model, x_norm, y_norm, normalizer)
        if not torch.isfinite(loss): raise FloatingPointError(f"{name}: non-finite loss at {update}")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            row = {"arm": name, "update": update, "loss_mean": float(np.mean(losses)),
                   "elapsed_seconds": time.perf_counter() - started,
                   **{key: float(value.detach()) for key, value in terms.items()}}
            trace.append(row); losses.clear(); print(json.dumps(row, sort_keys=True), flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    endpoint = output_dir / f"{name}_endpoint.pth"
    torch.save({"schema": SCHEMA, "arm": name, "updates": UPDATES,
                "model_config": model.architecture_config(), "model_state_dict": state}, endpoint)
    return {"path": str(endpoint), "sha256": sha256_file(endpoint), "model_state_sha256": state_sha(state),
            "trace": trace, "wall_seconds": time.perf_counter() - started}


def reload_sealed_arm(name: str, endpoint: dict[str, object], device: torch.device) -> CorrelationVolumePredictiveState:
    """Strictly reconstruct the exact sealed endpoint used for held inference."""
    path = Path(str(endpoint["path"]))
    if sha256_file(path) != endpoint["sha256"]:
        raise RuntimeError(f"{name}: endpoint file changed before reload")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = dict(payload["model_config"])
    if model_config.pop("architecture") != "correlation_volume_predictive_state":
        raise RuntimeError(f"{name}: endpoint architecture mismatch")
    if (payload.get("schema") != SCHEMA or payload.get("arm") != name
            or payload.get("updates") != UPDATES or model_config.pop("arm_mode") != name):
        raise RuntimeError(f"{name}: endpoint identity mismatch")
    model = CorrelationVolumePredictiveState(
        CorrelationVolumePredictiveStateConfig(**model_config), name)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    loaded_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if state_sha(loaded_state) != endpoint["model_state_sha256"]:
        raise RuntimeError(f"{name}: reloaded model state hash mismatch")
    return model.to(device).eval()


def reload_sealed_normalizer(artifact: dict[str, object], device: torch.device):
    """Recreate preprocessing from the sealed fit-only artifact."""
    path = Path(str(artifact["path"]))
    if sha256_file(path) != artifact["sha256"]:
        raise RuntimeError("normalizer file changed before reload")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.pop("schema", None) != SCHEMA:
        raise RuntimeError("normalizer schema mismatch")
    keys = ("mean_input", "mean_target", "std_input", "std_target")
    if tuple(payload) != keys or state_sha(payload) != artifact["state_sha256"]:
        raise RuntimeError("normalizer state hash mismatch")
    return backbone.common.scratch.ChannelNormalizer(
        *(payload[key] for key in keys), device=device)


def error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((SCORING.rel_l2_per_sample(prediction, target, 2),
                     SCORING.tke_rel_l2_per_sample(prediction, target, 2),
                     SCORING.mvpe_rel_l2_per_sample(prediction, target, 2)), axis=1).astype(np.float64)


def score(errors: np.ndarray) -> dict[str, float | int]:
    mean = errors.mean(0, dtype=np.float64)
    result: dict[str, float | int] = {"samples": int(len(errors)),
        "rel_l2_score": SCORING.score_error(float(mean[0])), "tke_score": SCORING.score_error(float(mean[1])),
        "mvpe_score": SCORING.score_error(float(mean[2]))}
    result["point_contribution_46_10_9"] = sum(POINT_WEIGHTS[key] * float(result[key]) for key in POINT_WEIGHTS)
    return result


def metric_delta(candidate: dict[str, float | int], control: dict[str, float | int]) -> dict[str, float]:
    return {key: float(candidate[key]) - float(control[key]) for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}


def comparison(candidate: str, control: str, errors: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, object]:
    delta = metric_delta(score(errors[candidate]), score(errors[control]))
    per_file = {name: float(score(errors[candidate][labels == name])["point_contribution_46_10_9"] - score(errors[control][labels == name])["point_contribution_46_10_9"])
                for name in sorted(set(labels.tolist()))}
    return {"delta": delta, "per_trajectory_file_delta_point": per_file,
            "positive_trajectory_files": sum(value > 0 for value in per_file.values()),
            "worst_trajectory_delta_point": min(per_file.values())}


def raw_volume_diagnostics(raw: torch.Tensor) -> dict[str, object]:
    # These are delta-volume diagnostics; literal nonzero is uninformative for
    # floating correlations, so report directional (non-centre argmax) mass.
    batch, channels, height, width = raw.shape; bins = 81
    value = raw.reshape(batch, channels // bins, bins, height, width)
    probability = value.float().softmax(2)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(2)
    result: dict[str, object] = {"by_lag": {}}
    for lag_index, lag in enumerate(MODEL_CONTRACT["motion_lags"]):
        argmax = value[:, lag_index].argmax(1).reshape(-1)
        result["by_lag"][str(lag)] = {
            "entropy": float(entropy[:, lag_index].mean().cpu()),
            "argmax_histogram": torch.bincount(argmax.cpu(), minlength=bins).tolist(),
            "argmax_noncenter_fraction": float((argmax != bins // 2).float().mean().cpu()),
        }
    return result


def angular_donor_indexes(windows: list[object], number_of_trajectories: int) -> list[int]:
    """Map every window to the next trajectory at exactly the same start."""
    lookup = {(int(window.trajectory), int(window.start)): index
              for index, window in enumerate(windows)}
    try:
        return [lookup[((int(window.trajectory) + 1) % number_of_trajectories,
                        int(window.start))] for window in windows]
    except KeyError as error:
        raise RuntimeError("angular donor trajectory is missing the matched window start") from error


def angular_region_rms(raw: torch.Tensor, candidate: CorrelationVolumePredictiveState) -> dict[str, float]:
    """Separate true interior angular energy from padding-affected boundaries."""
    radial = candidate.cost_projection.radialize_raw(raw)
    energy = (raw - radial).float().square().mean(dim=1)
    radius = candidate.config.radius
    interior = torch.zeros_like(energy, dtype=torch.bool)
    interior[:, radius:energy.shape[-2] - radius, radius:energy.shape[-1] - radius] = True
    return {
        "all": float(energy.mean().sqrt().cpu()),
        "interior": float(energy[interior].mean().sqrt().cpu()),
        "boundary": float(energy[~interior].mean().sqrt().cpu()),
        "interior_pixel_fraction": float(interior.float().mean().cpu()),
    }


def evaluate_arms(arms, trajectories, windows, window_files: list[str], normalizer, device: torch.device) -> dict[str, object]:
    dataset = backbone.common.scratch.SimWindowDataset(trajectories, windows, mask_probability=0., noise_scale=0.)
    errors, labels = {name: [] for name in ARM_NAMES}, []
    override, shuffled, directional_regions, normalized_signed_radial_rms, raw_audits = [], [], [], [], []
    band_drop_errors, band_inner_products = {name: [] for name in ("low", "mid", "high")}, []
    normalized_rms = {name: [] for name in ARM_NAMES}
    horizons = {str(step): {name: [] for name in ARM_NAMES} for step in range(20)}
    block_ranges = {"horizons_1_4": (0, 4), "horizons_5_8": (4, 8),
                    "horizons_9_12": (8, 12), "horizons_13_16": (12, 16),
                    "horizons_17_20": (16, 20)}
    cumulative_ranges = {f"horizons_1_{end}": (0, end) for end in (4, 8, 12, 16, 20)}
    block_errors = {group: {name: [] for name in ARM_NAMES} for group in block_ranges}
    cumulative_errors = {group: {name: [] for name in ARM_NAMES} for group in cumulative_ranges}
    candidate = arms["signed_candidate"]
    donor_index_for_window = angular_donor_indexes(windows, len(trajectories))
    for start in range(0, len(dataset), 32):
        indexes = range(start, min(start + 32, len(dataset))); samples = [dataset[index] for index in indexes]
        x, target = torch.stack([s[0] for s in samples]).to(device), torch.stack([s[1] for s in samples])
        donor_indexes = [donor_index_for_window[index] for index in indexes]
        donor_x = torch.stack([dataset[index][0] for index in donor_indexes]).to(device)
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        with torch.inference_mode():
            for name, model in arms.items():
                physical = (model(x_norm) * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
                normalized_rms[name].append(float(model.correlation_components(x_norm)["projected_cost"].float().square().mean().sqrt().cpu()))
                whole = error_rows(physical, target.numpy()); errors[name].append(whole)
                for step in range(20): horizons[str(step)][name].append(error_rows(physical[:, step:step + 1], target[:, step:step + 1].numpy()))
                for group, (left, right) in block_ranges.items():
                    block_errors[group][name].append(error_rows(physical[:, left:right], target[:, left:right].numpy()))
                for group, (left, right) in cumulative_ranges.items():
                    cumulative_errors[group][name].append(error_rows(physical[:, left:right], target[:, left:right].numpy()))
            parts = candidate.forward_components(x_norm)
            radial = candidate.forward_components(x_norm, radial_override=True)
            donor_norm = (donor_x - normalizer.mean_input) / normalizer.std_input
            shuffled_parts = candidate.forward_components(x_norm, angular_donor_history=donor_norm)
            radial_physical = (radial["prediction"] * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
            override.append(error_rows(radial_physical, target.numpy()))
            shuffled_physical = (shuffled_parts["prediction"] * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
            shuffled.append(error_rows(shuffled_physical, target.numpy()))
            directional_regions.append(angular_region_rms(parts["raw_cost"], candidate))
            normalized_signed_radial_rms.append(float((parts["projected_cost"] - radial["projected_cost"]).float().square().mean().sqrt().cpu()))
            raw_audits.append(raw_volume_diagnostics(parts["raw_cost"]))
            bands = [parts[name] for name in ("low", "mid", "high")]
            band_inner_products.append([float((bands[left].float() * bands[right].float()).mean().cpu())
                                        for left, right in ((0, 1), (0, 2), (1, 2))])
            for name in ("low", "mid", "high"):
                dropped = ((parts["prediction"] - parts[name]) * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
                band_drop_errors[name].append(error_rows(dropped, target.numpy()))
        labels.extend(window_files[index] for index in indexes)
    joined = {name: np.concatenate(value) for name, value in errors.items()}; labels_array = np.asarray(labels)
    radial_errors, shuffled_errors = np.concatenate(override), np.concatenate(shuffled)
    candidate_score = score(joined["signed_candidate"])
    shuffle_comparison = comparison("native", "shuffle", {"native": joined["signed_candidate"], "shuffle": shuffled_errors}, labels_array)
    per_horizon_rel_mvpe = {}
    for step, row in horizons.items():
        delta = metric_delta(score(np.concatenate(row["signed_candidate"])),
                             score(np.concatenate(row["radial_control"])))
        # A one-frame TKE is identically zero after temporal de-meaning, so it
        # is deliberately omitted rather than reported as a spurious 100.
        per_horizon_rel_mvpe[f"horizon_{int(step) + 1}"] = {
            "rel_l2_score": delta["rel_l2_score"],
            "mvpe_score": delta["mvpe_score"],
        }
    aggregated_raw = {}
    for lag in MODEL_CONTRACT["motion_lags"]:
        rows = [audit["by_lag"][str(lag)] for audit in raw_audits]
        histogram = np.sum([row["argmax_histogram"] for row in rows], axis=0).astype(int)
        aggregated_raw[str(lag)] = {
            "entropy_mean": float(np.mean([row["entropy"] for row in rows])),
            "argmax_histogram": histogram.tolist(),
            "argmax_noncenter_fraction": float(1. - histogram[40] / histogram.sum()),
        }
    return {"metrics": {name: score(value) for name, value in joined.items()},
            "signed_candidate_minus_radial_control": comparison("signed_candidate", "radial_control", joined, labels_array),
            "same_checkpoint_signed_minus_radial_diagnostic": metric_delta(candidate_score, score(radial_errors)),
            "same_checkpoint_native_minus_angular_residual_shuffle": shuffle_comparison,
            "per_horizon_rel_mvpe_deltas": per_horizon_rel_mvpe,
            "horizon_block_deltas": {group: metric_delta(
                score(np.concatenate(rows["signed_candidate"])),
                score(np.concatenate(rows["radial_control"])))
                for group, rows in block_errors.items()},
            "cumulative_horizon_deltas": {group: metric_delta(
                score(np.concatenate(rows["signed_candidate"])),
                score(np.concatenate(rows["radial_control"])))
                for group, rows in cumulative_errors.items()},
            "raw_volume_diagnostics": {"by_lag": aggregated_raw},
            "mechanical": {"raw_vs_radial_directional_rms": float(np.mean([row["all"] for row in directional_regions])),
                "raw_vs_radial_directional_rms_interior": float(np.mean([row["interior"] for row in directional_regions])),
                "raw_vs_radial_directional_rms_boundary": float(np.mean([row["boundary"] for row in directional_regions])),
                "matching_interior_pixel_fraction": float(np.mean([row["interior_pixel_fraction"] for row in directional_regions])),
                "normalized_projected_rms": {name: float(np.mean(values)) for name, values in normalized_rms.items()},
                "normalized_signed_vs_radial_rms": float(np.mean(normalized_signed_radial_rms))},
            "band_diagnostics": {"drop": {name: metric_delta(candidate_score, score(np.concatenate(value)))
                                               for name, value in band_drop_errors.items()},
                                 "pairwise_inner_product_mean_normalized_target_space": dict(zip(
                                     ("low_mid", "low_high", "mid_high"),
                                     np.mean(band_inner_products, axis=0).tolist())),
                                 "role": "diagnostic_only"}}


def gate(evaluation: dict[str, object]) -> tuple[bool, dict[str, object]]:
    primary = evaluation["signed_candidate_minus_radial_control"]; delta = primary["delta"]
    shuffle = evaluation["same_checkpoint_native_minus_angular_residual_shuffle"]
    interior_rms = evaluation["mechanical"]["raw_vs_radial_directional_rms_interior"]
    passed = (float(delta["point_contribution_46_10_9"]) >= .08 and all(float(delta[key]) >= -.03 for key in POINT_WEIGHTS)
              and int(primary["positive_trajectory_files"]) >= 15 and float(primary["worst_trajectory_delta_point"]) >= -.05
              and float(shuffle["delta"]["point_contribution_46_10_9"]) >= .02
              and int(shuffle["positive_trajectory_files"]) >= 15 and float(interior_rms) > 1e-4)
    return passed, {"primary_point_min": .08, "component_floor": -.03, "positive_trajectory_files_min": 15,
                    "worst_trajectory_delta_point_min": -.05, "native_minus_angular_residual_shuffle_point_min": .02,
                    "angular_shuffle_positive_trajectory_files_min": 15,
                    "interior_raw_vs_radial_directional_rms_min": 1e-4,
                    "boundary_angular_energy": "diagnostic_only", "bands": "diagnostic_only"}


def prior_held10_evidence(path: Path | None, held_aoa: int) -> dict[str, object] | None:
    if held_aoa == 10:
        if path is not None: raise ValueError("held10 takes no prior result")
        return None
    if path is None or not path.is_file(): raise PermissionError("AoA20 robustness run requires an AoA10 GO result")
    try: prior = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error: raise PermissionError("prior file is not a verified held10 GO") from error
    fixed = {"schema": SCHEMA, "held_aoa": 10, "updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED,
             "arms": list(ARM_NAMES), "expected_parameters_per_arm": EXPECTED_PARAMETERS, **MODEL_CONTRACT,
             "source_sha256": source_hashes()}
    try: valid = (prior["schema"] == SCHEMA and prior["decision"] == "GO_AOA20_ROBUSTNESS" and prior.get("git", {}).get("clean") is True
                       and all(prior["config"].get(k) == v for k, v in fixed.items()) and gate(prior["evaluation"])[0])
    except (KeyError, TypeError, ValueError): valid = False
    if not valid: raise PermissionError("prior file is not a verified held10 GO")
    seal_record = prior.get("endpoint_seal", {}); seal_path = Path(str(seal_record.get("path", "")))
    if not seal_path.is_file() or sha256_file(seal_path) != seal_record.get("sha256"): raise PermissionError("held10 endpoint seal is missing or changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("source_sha256") != source_hashes():
        raise PermissionError("held10 sealed source contract is missing or changed")
    artifacts = [seal.get("config", {}), seal.get("fit_only_normalizer", {})] + [seal.get("endpoints", {}).get(name, {}) for name in ARM_NAMES]
    if any(not Path(str(a.get("path", ""))).is_file() or sha256_file(Path(str(a["path"]))) != a.get("sha256") for a in artifacts): raise PermissionError("a held10 sealed artifact is missing or changed")
    result_digest_path = path.with_suffix(path.suffix + ".sha256")
    if (not result_digest_path.is_file()
            or result_digest_path.read_text(encoding="utf-8").strip() != sha256_file(path)):
        raise PermissionError("held10 result detached digest is missing or changed")
    return {"path": str(path), "sha256": sha256_file(path), "decision": "GO_AOA20_ROBUSTNESS"}


def git_report() -> dict[str, object]:
    def run(*args: str) -> str: return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--untracked-files=all")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "clean": not bool(dirty), "porcelain": dirty.splitlines()}


def benchmark(device: torch.device) -> float:
    model = construct_arms(device)[0]["signed_candidate"].train(); x = torch.randn(8, 20, 32, 64, 3, device=device)
    # Exclude CUDA allocator/cuDNN one-time setup; this records steady-state
    # B8 forward+backward, the quantity relevant to the 0.3 s guardrail.
    for _ in range(2):
        model.zero_grad(set_to_none=True); model(x).square().mean().backward()
    if device.type == "cuda": torch.cuda.synchronize(device)
    model.zero_grad(set_to_none=True); started = time.perf_counter(); model(x).square().mean().backward()
    if device.type == "cuda": torch.cuda.synchronize(device)
    return time.perf_counter() - started


def smoke(device: torch.device) -> None:
    arms, identity = construct_arms(device); x, y = torch.randn(2, 20, 32, 64, 3, device=device), torch.randn(2, 20, 32, 64, 3, device=device)
    for model in arms.values():
        model.train(); output = model(x); output.sub(y).square().mean().backward()
        if not torch.isfinite(output).all(): raise FloatingPointError("non-finite smoke output")
        model.zero_grad(set_to_none=True)
    print(json.dumps({"smoke": "PASS", "identity": identity, "benchmark_b8_forward_backward_seconds": benchmark(device)}, sort_keys=True))


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != "CVPS_WHOLE_AOA_V1": raise PermissionError("formal run requires --execute and protocol acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError("refusing to overwrite non-empty output directory")
    prior, git = prior_held10_evidence(args.prior_held10_result, args.held_aoa), git_report()
    if not git["clean"]: raise RuntimeError(f"worktree must be clean: {git['porcelain']}")
    args.output_dir.mkdir(parents=True, exist_ok=True); inventory = scan_inventory(args.sim_dir)
    fit_records, held_records = [x for x in inventory if int(x["aoa"]) != args.held_aoa], [x for x in inventory if int(x["aoa"]) == args.held_aoa]
    if len(fit_records) != 80 or len(held_records) != 20: raise RuntimeError("unexpected whole-AoA inventory")
    sources = source_hashes()
    config = {"schema": SCHEMA, "held_aoa": args.held_aoa, "fit_aoas": [x for x in AOAS if x != args.held_aoa], "updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED, "training_rng_seed": TRAINING_RNG_SEED, "arms": list(ARM_NAMES), "expected_parameters_per_arm": EXPECTED_PARAMETERS, **MODEL_CONTRACT, "matching_grid": list(MATCHING_GRID), "state_grid": list(STATE_GRID), "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "scheduler": "CosineAnnealingLR", "scheduler_t_max": UPDATES, "grad_clip_norm": GRAD_CLIP, "accuracy_metric_weights_rel_tke_mvpe": list(ACCURACY_CHANNEL_WEIGHTS), "mse_weight": MSE_WEIGHT, "augmentation": {"mask_probability": 0., "noise_scale": 0.}, "sample_schedule": "seeded_with_replacement_updates_x_batch", "shuffle_donor_policy": "next_held_trajectory_modulo_20_same_window_start", "source_sha256": sources, "fit_inventory_sha256": canonical_sha(fit_records), "held_inventory_sha256": canonical_sha(held_records), "prior_held10_evidence": prior, "held_field_values_loaded_before_all_endpoint_seals": False}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n"); (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (args.output_dir / "environment.txt").write_text("\n".join(f"{k}={v}" for k, v in {"python": sys.version.replace("\n", " "), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "pid": os.getpid()}.items()) + "\n")
    fit_trajectories, fit_windows, _ = load_fields(args.sim_dir, fit_records)
    if len(fit_windows) != 3_840: raise RuntimeError("unexpected fit window count")
    normalizer = backbone.common.scratch.ChannelNormalizer(*backbone.common.scratch.window_statistics(fit_trajectories, fit_windows), device=device)
    normalizer_state = {k: getattr(normalizer, k).detach().cpu() for k in ("mean_input", "mean_target", "std_input", "std_target")}; normalizer_path = args.output_dir / "fit_only_normalizer.pt"; torch.save({"schema": SCHEMA, **normalizer_state}, normalizer_path)
    normalizer_artifact = {"path": str(normalizer_path), "sha256": sha256_file(normalizer_path), "state_sha256": state_sha(normalizer_state)}
    dataset = backbone.common.scratch.SimWindowDataset(fit_trajectories, fit_windows, mask_probability=0., noise_scale=0.); schedule, pair = make_schedule(fit_windows), construct_arms(device); models, identity = pair
    endpoints = {name: train_arm(name, models[name], dataset, schedule, normalizer, device, args.output_dir) for name in ARM_NAMES}
    if source_hashes() != sources:
        raise RuntimeError("source changed during fit; refusing to seal endpoints")
    config_path = args.output_dir / "config.json"
    seal = {"schema": SCHEMA, "source_sha256": sources,
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "fit_only_normalizer": normalizer_artifact,
            "endpoints": {name: {k: endpoints[name][k] for k in ("path", "sha256", "model_state_sha256")} for name in ARM_NAMES}}
    seal_path = args.output_dir / "all_endpoints_seal.json"; seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n"); seal_sha256 = sha256_file(seal_path)
    if source_hashes() != sources:
        raise RuntimeError("source changed after endpoint seal; refusing to load held fields")
    sealed_artifacts = [seal["config"], seal["fit_only_normalizer"], *seal["endpoints"].values()]
    if any(sha256_file(Path(str(artifact["path"]))) != artifact["sha256"] for artifact in sealed_artifacts):
        raise RuntimeError("sealed fit artifact changed before held-field load")
    # The held evaluator consumes freshly reconstructed checkpoint objects,
    # never the mutable models that happened to remain in training memory.
    normalizer = reload_sealed_normalizer(normalizer_artifact, device)
    models = {name: reload_sealed_arm(name, endpoints[name], device) for name in ARM_NAMES}
    held_trajectories, held_windows, held_files = load_fields(args.sim_dir, held_records)
    if len(held_windows) != 960: raise RuntimeError("unexpected held window count")
    evaluation = evaluate_arms(models, held_trajectories, held_windows, held_files, normalizer, device); passed, gate_config = gate(evaluation)
    decision = "GO_AOA20_ROBUSTNESS" if args.held_aoa == 10 and passed else ("COMPLETE_ROBUST" if passed else "STOP_CVPS")
    evidence_class = ("architecture_selection_unseen_whole_AoA_held_Sim" if args.held_aoa == 10
                      else "selection_seen_crossfit_robustness")
    result = {"schema": SCHEMA, "evidence_class": evidence_class, "config": config, "git": git, "initial_identity": identity, "fit_trajectories": len(fit_records), "fit_windows": len(fit_windows), "held_trajectories": len(held_records), "held_windows": len(held_windows), "sample_stream_sha256_all_arms": hashlib.sha256(schedule.tobytes()).hexdigest(), "fit_only_normalizer": normalizer_artifact, "endpoints": endpoints, "endpoint_seal": {"path": str(seal_path), "sha256": seal_sha256}, "evaluation": evaluation, "gate": gate_config, "decision": decision}
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    result_path.with_suffix(result_path.suffix + ".sha256").write_text(sha256_file(result_path) + "\n")
    print(json.dumps({"decision": decision, "result": str(result_path), "candidate_minus_control": evaluation["signed_candidate_minus_radial_control"]["delta"]}, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("this discovery screen requires CUDA")
    smoke(device) if args.smoke else run(args, device)


if __name__ == "__main__": main()
