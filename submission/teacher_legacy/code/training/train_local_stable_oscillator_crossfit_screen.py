#!/usr/bin/env python3
"""Fit-side, Reynolds-disjoint screen for local stable complex oscillator.

This is explicitly discovery evidence, not an unseen validation: all four
allowed Sim AoAs have informed earlier work.  Within a selected fold each of
the five held Reynolds numbers is withheld across *every* allowed AoA.
AoA10 is deliberately never selected for training or field loading.
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
import h5py

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for _path in (CODE_ROOT, Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_flow_backbone as backbone  # noqa: E402
import train_temporal_transport_loao_screen as io  # noqa: E402
from models.local_stable_oscillator import (  # noqa: E402
    LocalStableOscillatorConfig, LocalStableOscillatorForecaster,
)

SCHEMA = "local_stable_oscillator_fit_side_reynolds_disjoint_crossfit_v1"
ALLOWED_AOAS = (0, 5, 15, 20)
ARM_NAMES = ("real_control", "complex_candidate")
FAMILY_FOR_ARM = {"real_control": "real", "complex_candidate": "complex"}
SEED, TRAINING_RNG_SEED, UPDATES, BATCH_SIZE = 210, 227, 2_000, 8
LEARNING_RATE, WEIGHT_DECAY, MSE_WEIGHT, GRAD_CLIP = 3e-4, 1e-5, .25, 1.
EXPECTED_PARAMETERS = 1_250_609
ACCURACY_CHANNEL_WEIGHTS = (1., 4., 1.)
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}
SOURCE_PATHS = {
    "trainer": Path(__file__),
    "model": CODE_ROOT / "models" / "local_stable_oscillator.py",
    "band_decoder": CODE_ROOT / "models" / "causal_state_multiband.py",
    "flow_backbone_driver": CODE_ROOT / "training" / "train_flow_backbone.py",
    "loss_and_normalizer_driver": CODE_ROOT / "training" / "train_hybrid_fno2d.py",
    "scratch_dataset": CODE_ROOT / "training" / "train_fno_sim_from_scratch.py",
    "sealed_io_driver": CODE_ROOT / "training" / "train_temporal_transport_loao_screen.py",
    "official_v9_scoring": REPO_ROOT / "kits" / "track1" / "realpde_t1_starting_kit_v9" / "scoring.py",
}
SCORING = io.SCORING


def source_hashes() -> dict[str, str]:
    return {name: io.sha256_file(path) for name, path in SOURCE_PATHS.items()}


def scan_allowed_inventory(sim_dir: Path) -> list[dict[str, object]]:
    """Read metadata from every file, but touch field datasets only for AoA 0/5/15/20.

    In particular, AoA10's ``u/v/p`` datasets are never indexed, opened by
    name, or even queried for shape; its metadata only lets us reject it from
    the field-loading path below.
    """
    records = []
    for path in sorted(sim_dir.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            if "aoa" not in handle or "re" not in handle:
                raise ValueError(f"{path.name} misses AoA/Re metadata")
            aoa, re = int(np.asarray(handle["aoa"]).reshape(-1)[0]), int(np.asarray(handle["re"]).reshape(-1)[0])
            if aoa not in ALLOWED_AOAS:
                continue
            if any(key not in handle for key in ("u", "v", "p")):
                raise ValueError(f"{path.name} misses field dataset")
            shapes = [tuple(handle[key].shape) for key in ("u", "v", "p")]
            if not (shapes[0] == shapes[1] == shapes[2]):
                raise ValueError(f"field shapes differ in {path.name}")
            records.append({"file": path.name, "aoa": aoa, "re": re,
                            "frames": int(shapes[0][0]), "shape": list(shapes[0])})
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def select_fold(records: list[dict[str, object]], fold: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Return 60/20 trajectories, with a Re ordinal held across all AoAs."""
    unexpected = sorted({int(row["aoa"]) for row in records} - set(ALLOWED_AOAS) - {10})
    if unexpected:
        raise ValueError(f"unexpected AoA metadata {unexpected}")
    selected = [row for row in records if int(row["aoa"]) in ALLOWED_AOAS]
    if len(selected) != 80:
        raise ValueError(f"expected 80 allowed records, got {len(selected)}")
    by_aoa: dict[int, list[dict[str, object]]] = {}
    for aoa in ALLOWED_AOAS:
        rows = sorted((row for row in selected if int(row["aoa"]) == aoa), key=lambda row: (int(row["re"]), str(row["file"])))
        if len(rows) != 20 or len({int(row["re"]) for row in rows}) != 20:
            raise ValueError(f"AoA {aoa} must contain 20 unique Re trajectories")
        by_aoa[aoa] = rows
    held = [row for aoa in ALLOWED_AOAS for ordinal, row in enumerate(by_aoa[aoa]) if ordinal % 4 == fold]
    train = [row for aoa in ALLOWED_AOAS for ordinal, row in enumerate(by_aoa[aoa]) if ordinal % 4 != fold]
    held_res = {int(row["re"]) for row in held}
    if len(train) != 60 or len(held) != 20 or any(int(row["re"]) in held_res for row in train):
        raise RuntimeError("Re-disjoint 60/20 split contract failed")
    # This is intentionally asserted on records, before any selected field is opened.
    manifest = {"fold": fold, "allowed_aoas": list(ALLOWED_AOAS),
                "held_re_by_aoa": {str(aoa): [int(row["re"]) for ordinal, row in enumerate(by_aoa[aoa]) if ordinal % 4 == fold] for aoa in ALLOWED_AOAS},
                "train_re_by_aoa": {str(aoa): [int(row["re"]) for ordinal, row in enumerate(by_aoa[aoa]) if ordinal % 4 != fold] for aoa in ALLOWED_AOAS},
                "train_records_sha256": io.canonical_sha(train), "held_records_sha256": io.canonical_sha(held)}
    return train, held, manifest


def make_config() -> LocalStableOscillatorConfig:
    return LocalStableOscillatorConfig()


def construct_arms(device: torch.device) -> tuple[dict[str, LocalStableOscillatorForecaster], dict[str, object]]:
    snapshots, arms = {}, {}
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = LocalStableOscillatorForecaster(make_config(), FAMILY_FOR_ARM[name]).to(device)
        arms[name] = model
        snapshots[name] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    reference = snapshots[ARM_NAMES[0]]
    if any(reference.keys() != state.keys() or any(not torch.equal(reference[key], state[key]) for key in reference) for state in snapshots.values()):
        raise RuntimeError("arms do not have bitwise-identical initial state")
    counts = {name: sum(value.numel() for value in model.parameters()) for name, model in arms.items()}
    if set(counts.values()) != {EXPECTED_PARAMETERS}:
        raise RuntimeError(f"parameter contract changed {counts}")
    return arms, {"parameters": counts, "initial_state_sha256": io.state_sha(reference),
                  "bitwise_identical": True, "only_variable": "fixed_stable_temporal_family",
                  "families": dict(FAMILY_FOR_ARM)}


def make_schedule(windows: list[object]) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, len(windows), size=(UPDATES, BATCH_SIZE), dtype=np.int64)


def batches(dataset, schedule: np.ndarray):
    for row in schedule:
        values = [dataset[int(index)] for index in row]
        yield torch.stack([value[0] for value in values]), torch.stack([value[1] for value in values])


def loss_terms(model, x_norm: torch.Tensor, y_norm: torch.Tensor, normalizer):
    prediction = model(x_norm)
    accuracy, detail = backbone.common.accuracy_objective(prediction, y_norm, normalizer, ACCURACY_CHANNEL_WEIGHTS)
    mse = (prediction - y_norm).square().mean()
    loss = accuracy + MSE_WEIGHT * mse
    return loss, {"accuracy": accuracy, "mse": mse, "loss": loss, **detail}


def train_arm(name: str, model, dataset, schedule: np.ndarray, normalizer, device: torch.device, output_dir: Path) -> dict[str, object]:
    backbone.common.set_model_stats(model, normalizer)
    torch.manual_seed(TRAINING_RNG_SEED)
    if device.type == "cuda": torch.cuda.manual_seed_all(TRAINING_RNG_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    trace, bucket, started = [], [], time.perf_counter()
    for update, (x, y) in enumerate(batches(dataset, schedule), 1):
        model.train(); x_norm, y_norm = normalizer.preprocess(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True); loss, terms = loss_terms(model, x_norm, y_norm, normalizer)
        if not torch.isfinite(loss): raise FloatingPointError(f"{name}: non-finite loss at {update}")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step(); scheduler.step(); bucket.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            row = {"arm": name, "update": update, "loss_mean": float(np.mean(bucket)), "elapsed_seconds": time.perf_counter() - started,
                   **{key: float(value.detach()) for key, value in terms.items()}}
            trace.append(row); bucket.clear(); print(json.dumps(row, sort_keys=True), flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    path = output_dir / f"{name}_endpoint.pth"
    torch.save({"schema": SCHEMA, "arm": name, "updates": UPDATES, "model_config": model.architecture_config(), "model_state_dict": state}, path)
    return {"path": str(path), "sha256": io.sha256_file(path), "model_state_sha256": io.state_sha(state), "trace": trace, "wall_seconds": time.perf_counter() - started}


def reload_sealed_arm(name: str, endpoint: dict[str, object], device: torch.device):
    path = Path(str(endpoint["path"]))
    if io.sha256_file(path) != endpoint["sha256"]: raise RuntimeError(f"{name}: endpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False); config = dict(payload["model_config"])
    if (payload.get("schema") != SCHEMA or payload.get("arm") != name or payload.get("updates") != UPDATES
            or config.pop("architecture", None) != "local_stable_oscillator" or config.pop("temporal_family", None) != FAMILY_FOR_ARM[name]):
        raise RuntimeError(f"{name}: endpoint identity mismatch")
    model = LocalStableOscillatorForecaster(LocalStableOscillatorConfig(**config), FAMILY_FOR_ARM[name])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if io.state_sha(state) != endpoint["model_state_sha256"]: raise RuntimeError(f"{name}: state hash mismatch")
    return model.to(device).eval()


def save_normalizer(normalizer, output_dir: Path) -> dict[str, object]:
    state = {key: getattr(normalizer, key).detach().cpu() for key in ("mean_input", "mean_target", "std_input", "std_target")}
    path = output_dir / "fit_only_normalizer.pt"; torch.save({"schema": SCHEMA, **state}, path)
    return {"path": str(path), "sha256": io.sha256_file(path), "state_sha256": io.state_sha(state)}


def reload_sealed_normalizer(artifact: dict[str, object], device: torch.device):
    path = Path(str(artifact["path"]))
    if io.sha256_file(path) != artifact["sha256"]: raise RuntimeError("normalizer changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.pop("schema", None) != SCHEMA or tuple(payload) != ("mean_input", "mean_target", "std_input", "std_target") or io.state_sha(payload) != artifact["state_sha256"]:
        raise RuntimeError("normalizer identity mismatch")
    return backbone.common.scratch.ChannelNormalizer(*(payload[key] for key in payload), device=device)


def error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((SCORING.rel_l2_per_sample(prediction, target, 2), SCORING.tke_rel_l2_per_sample(prediction, target, 2), SCORING.mvpe_rel_l2_per_sample(prediction, target, 2)), axis=1).astype(np.float64)


def score(errors: np.ndarray) -> dict[str, float | int]:
    mean = errors.mean(0, dtype=np.float64)
    result: dict[str, float | int] = {"samples": int(len(errors)), "rel_l2_score": SCORING.score_error(float(mean[0])), "tke_score": SCORING.score_error(float(mean[1])), "mvpe_score": SCORING.score_error(float(mean[2]))}
    result["point_contribution_46_10_9"] = sum(POINT_WEIGHTS[key] * float(result[key]) for key in POINT_WEIGHTS)
    return result


def metric_delta(candidate: dict[str, float | int], control: dict[str, float | int]) -> dict[str, float]:
    return {key: float(candidate[key]) - float(control[key]) for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}


def _comparison(candidate: np.ndarray, control: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    per = {name: float(score(candidate[labels == name])["point_contribution_46_10_9"] - score(control[labels == name])["point_contribution_46_10_9"]) for name in sorted(set(labels.tolist()))}
    return {"delta": metric_delta(score(candidate), score(control)), "per_trajectory_file_delta_point": per,
            "positive_trajectory_files": sum(value > 0. for value in per.values()), "worst_trajectory_delta_point": min(per.values())}


def _diagnostics(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    rates = parts
    dynamic = rates["dynamic_states"].float()
    return {"gamma_mean": float(rates["gamma"].float().mean().cpu()), "gamma_min": float(rates["gamma"].float().min().cpu()),
            "gamma_max": float(rates["gamma"].float().max().cpu()),
            "omega_abs_mean": float(rates["omega"].float().abs().mean().cpu()), "omega_abs_max": float(rates["omega"].float().abs().max().cpu()),
            "terminal_state_rms": float(rates["terminal_state"].float().square().mean().sqrt().cpu()),
            "dynamic_state_rms": float(dynamic.square().mean().sqrt().cpu()),
            **{f"dynamic_state_rms_h{step + 1}": float(dynamic[:, step].square().mean().sqrt().cpu())
               for step in (0, 4, 11, 19)},
            "analytic_complex_gain_h40_max": float(torch.exp(-40. * rates["gamma"].float()).max().cpu()),
            "analytic_real_pole_gain_h40_max": float(torch.exp(
                40. * rates["real_log_rates"].float()).max().cpu())}


def reverse_preterminal_history(history: torch.Tensor) -> torch.Tensor:
    """Reverse the first 19 observations while preserving the terminal frame."""
    if history.ndim != 5 or history.shape[1] != 20:
        raise ValueError("expected a 20-frame history")
    return torch.cat((history[:, :19].flip(1), history[:, 19:20]), dim=1)


def evaluate_arms(arms, trajectories, windows, files: list[str], normalizer, device: torch.device) -> dict[str, object]:
    dataset = backbone.common.scratch.SimWindowDataset(trajectories, windows, mask_probability=0., noise_scale=0.)
    errors, labels = {name: [] for name in ARM_NAMES}, []
    reversed_errors = {name: [] for name in ARM_NAMES}
    reversed_middle_errors = {name: [] for name in ARM_NAMES}
    order_sensitivity = {name: {"prediction_rms": [], "gamma_rms": [], "temporal2_rms": []}
                         for name in ARM_NAMES}
    zero, zero_middle, rate_rows = [], [], []
    blocks = {name: {arm: [] for arm in ARM_NAMES} for name in ("horizons_1_4", "horizons_5_8", "horizons_9_12", "horizons_13_16", "horizons_17_20", "horizons_5_12")}
    ranges = {"horizons_1_4": (0, 4), "horizons_5_8": (4, 8), "horizons_9_12": (8, 12), "horizons_13_16": (12, 16), "horizons_17_20": (16, 20), "horizons_5_12": (4, 12)}
    for begin in range(0, len(dataset), 32):
        indexes = range(begin, min(begin + 32, len(dataset))); values = [dataset[index] for index in indexes]
        x, target = torch.stack([value[0] for value in values]).to(device), torch.stack([value[1] for value in values])
        x_norm = (x - normalizer.mean_input) / normalizer.std_input
        reversed_history = reverse_preterminal_history(x_norm)
        with torch.inference_mode():
            for name, model in arms.items():
                native_parts = model.forward_components(x_norm)
                reversed_parts = model.forward_components(reversed_history)
                physical = (native_parts["prediction"] * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
                reversed_physical = (reversed_parts["prediction"] * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
                errors[name].append(error_rows(physical, target.numpy()))
                reversed_errors[name].append(error_rows(reversed_physical, target.numpy()))
                reversed_middle_errors[name].append(error_rows(
                    reversed_physical[:, 4:12], target[:, 4:12].numpy()))
                order_sensitivity[name]["prediction_rms"].append(float(
                    (native_parts["prediction"] - reversed_parts["prediction"]).float().square().mean().sqrt().cpu()))
                order_sensitivity[name]["gamma_rms"].append(float(
                    (native_parts["gamma"] - reversed_parts["gamma"]).float().square().mean().sqrt().cpu()))
                order_sensitivity[name]["temporal2_rms"].append(float(
                    (native_parts["raw_temporal2"] - reversed_parts["raw_temporal2"]).float().square().mean().sqrt().cpu()))
                for block, (left, right) in ranges.items(): blocks[block][name].append(error_rows(physical[:, left:right], target[:, left:right].numpy()))
            parts = arms["complex_candidate"].forward_components(x_norm)
            zero_physical = (arms["complex_candidate"].forward_components(x_norm, temporal_family="zero_frequency")["prediction"] * normalizer.std_target + normalizer.mean_target).float().cpu().numpy()
            zero.append(error_rows(zero_physical, target.numpy()))
            zero_middle.append(error_rows(zero_physical[:, 4:12], target[:, 4:12].numpy()))
            rate_rows.append(_diagnostics(parts))
        labels.extend(files[index] for index in indexes)
    joined = {name: np.concatenate(rows) for name, rows in errors.items()}; labels_array = np.asarray(labels)
    joined_reversed = {name: np.concatenate(rows) for name, rows in reversed_errors.items()}
    return {"metrics": {name: score(value) for name, value in joined.items()},
            "complex_candidate_minus_real_control": _comparison(joined["complex_candidate"], joined["real_control"], labels_array),
            "same_checkpoint_native_minus_zero_frequency": metric_delta(score(joined["complex_candidate"]), score(np.concatenate(zero))),
            "same_checkpoint_native_minus_zero_frequency_horizons_5_12": metric_delta(
                score(np.concatenate(blocks["horizons_5_12"]["complex_candidate"])),
                score(np.concatenate(zero_middle))),
            "horizon_block_deltas": {name: metric_delta(score(np.concatenate(rows["complex_candidate"])), score(np.concatenate(rows["real_control"]))) for name, rows in blocks.items()},
            "preterminal_history_order_reversal_diagnostic": {
                name: {"native_minus_reversed": metric_delta(score(joined[name]), score(joined_reversed[name])),
                       "native_minus_reversed_horizons_5_12": metric_delta(
                           score(np.concatenate(blocks["horizons_5_12"][name])),
                           score(np.concatenate(reversed_middle_errors[name]))),
                       **{key: float(np.mean(values)) for key, values in order_sensitivity[name].items()}}
                for name in ARM_NAMES},
            "rate_and_state_diagnostics": {key: float(np.mean([row[key] for row in rate_rows])) for key in rate_rows[0]}}


def gate(evaluation: dict[str, object]) -> tuple[bool, dict[str, object]]:
    primary = evaluation["complex_candidate_minus_real_control"]
    point = primary["delta"]["point_contribution_46_10_9"]
    middle = evaluation["horizon_block_deltas"]["horizons_5_12"]["point_contribution_46_10_9"]
    middle_early = evaluation["horizon_block_deltas"]["horizons_5_8"]["point_contribution_46_10_9"]
    middle_late = evaluation["horizon_block_deltas"]["horizons_9_12"]["point_contribution_46_10_9"]
    zero = evaluation["same_checkpoint_native_minus_zero_frequency"]["point_contribution_46_10_9"]
    zero_middle = evaluation["same_checkpoint_native_minus_zero_frequency_horizons_5_12"]["point_contribution_46_10_9"]
    passed = (float(point) >= .08 and float(middle) >= .04
              and float(middle_early) >= 0. and float(middle_late) >= 0.
              and int(primary["positive_trajectory_files"]) >= 15
              and float(primary["worst_trajectory_delta_point"]) >= -.05
              and float(zero) >= .02 and float(zero_middle) >= .01)
    return passed, {"candidate_minus_control_point_min": .08, "middle_horizons_5_12_point_min": .04,
                    "horizons_5_8_and_9_12_point_min": 0., "positive_trajectory_files_min": 15,
                    "worst_trajectory_delta_point_min": -.05,
                    "native_minus_zero_frequency_point_min": .02,
                    "native_minus_zero_frequency_horizons_5_12_point_min": .01,
                    "component_scores": "diagnostic_only"}


def git_report() -> dict[str, object]:
    def run(*args): return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--untracked-files=all")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "clean": not bool(dirty), "porcelain": dirty.splitlines()}


def smoke(device: torch.device) -> None:
    arms, identity = construct_arms(device); x = torch.randn(2, 20, 32, 64, 3, device=device); y = torch.randn_like(x)
    for model in arms.values():
        output = model(x); output.sub(y).square().mean().backward(); model.zero_grad(set_to_none=True)
        if not torch.isfinite(output).all(): raise FloatingPointError("non-finite smoke output")
    print(json.dumps({"smoke": "PASS", "identity": identity}, sort_keys=True))


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != "M7_FIT_SIDE_REYNOLDS_DISJOINT_V1": raise PermissionError("formal run requires --execute and acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError("refusing to overwrite non-empty output directory")
    git = git_report()
    if not git["clean"]: raise RuntimeError(f"worktree must be clean: {git['porcelain']}")
    # AoA10 field datasets are never opened; the scanner itself only receives
    # allowed trajectories, and load_fields receives a subset of those.
    train_records, held_records, split = select_fold(scan_allowed_inventory(args.sim_dir), args.fold)
    sources = source_hashes(); args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config = make_config()
    config = {"schema": SCHEMA, "evidence_class": "fit_side_reynolds_disjoint_crossfit_discovery", "fold": args.fold, "allowed_aoas": list(ALLOWED_AOAS), "forbidden_aoa": 10, "updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED, "training_rng_seed": TRAINING_RNG_SEED, "arms": list(ARM_NAMES), "families": FAMILY_FOR_ARM, "expected_parameters_per_arm": EXPECTED_PARAMETERS, "model_contract": model_config.__dict__, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "scheduler": "CosineAnnealingLR", "scheduler_t_max": UPDATES, "grad_clip_norm": GRAD_CLIP, "accuracy_metric_weights_rel_tke_mvpe": list(ACCURACY_CHANNEL_WEIGHTS), "mse_weight": MSE_WEIGHT, "augmentation": {"mask_probability": 0., "noise_scale": 0.}, "sample_schedule": "seeded_with_replacement_updates_x_batch", "source_sha256": sources, "split": split, "held_field_values_loaded_before_all_endpoint_seals": False}
    config_path = args.output_dir / "config.json"; config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (args.output_dir / "environment.txt").write_text("\n".join(f"{key}={value}" for key, value in {"python": sys.version.replace("\n", " "), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "pid": os.getpid()}.items()) + "\n")
    train_trajectories, train_windows, _ = io.load_fields(args.sim_dir, train_records)
    if len(train_windows) != 2_880: raise RuntimeError("expected 2880 training windows")
    normalizer = backbone.common.scratch.ChannelNormalizer(*backbone.common.scratch.window_statistics(train_trajectories, train_windows), device=device)
    normalizer_artifact = save_normalizer(normalizer, args.output_dir)
    dataset = backbone.common.scratch.SimWindowDataset(train_trajectories, train_windows, mask_probability=0., noise_scale=0.)
    schedule = make_schedule(train_windows); models, identity = construct_arms(device)
    endpoints = {name: train_arm(name, models[name], dataset, schedule, normalizer, device, args.output_dir) for name in ARM_NAMES}
    if source_hashes() != sources: raise RuntimeError("source changed during fit")
    seal = {"schema": SCHEMA, "source_sha256": sources, "config": {"path": str(config_path), "sha256": io.sha256_file(config_path)}, "fit_only_normalizer": normalizer_artifact, "endpoints": {name: {key: endpoints[name][key] for key in ("path", "sha256", "model_state_sha256")} for name in ARM_NAMES}}
    seal_path = args.output_dir / "all_endpoints_seal.json"; seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    if source_hashes() != sources or any(io.sha256_file(Path(str(item["path"]))) != item["sha256"] for item in [seal["config"], normalizer_artifact, *seal["endpoints"].values()]): raise RuntimeError("sealed fit artifact changed before held load")
    normalizer = reload_sealed_normalizer(normalizer_artifact, device); models = {name: reload_sealed_arm(name, endpoints[name], device) for name in ARM_NAMES}
    held_trajectories, held_windows, held_files = io.load_fields(args.sim_dir, held_records)
    if len(held_windows) != 960: raise RuntimeError("expected 960 held windows")
    evaluation = evaluate_arms(models, held_trajectories, held_windows, held_files, normalizer, device); passed, gate_config = gate(evaluation)
    result = {"schema": SCHEMA, "evidence_class": "fit_side_reynolds_disjoint_crossfit_discovery", "config": config, "git": git, "initial_identity": identity, "fit_trajectories": len(train_records), "held_trajectories": len(held_records), "fit_windows": len(train_windows), "held_windows": len(held_windows), "sample_stream_sha256_all_arms": hashlib.sha256(schedule.tobytes()).hexdigest(), "fit_only_normalizer": normalizer_artifact, "endpoints": endpoints, "endpoint_seal": {"path": str(seal_path), "sha256": io.sha256_file(seal_path)}, "evaluation": evaluation, "gate": gate_config, "decision": "GO_NEXT_FOLDS" if passed else "STOP_M7_FOLD"}
    result_path = args.output_dir / "result.json"; result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"); result_path.with_suffix(".json.sha256").write_text(io.sha256_file(result_path) + "\n")
    print(json.dumps({"decision": result["decision"], "result": str(result_path), "candidate_minus_control": evaluation["complex_candidate_minus_real_control"]["delta"]}, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("this screen requires CUDA")
    smoke(device) if args.smoke else run(args, device)


if __name__ == "__main__": main()
