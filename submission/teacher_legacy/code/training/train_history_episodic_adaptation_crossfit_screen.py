#!/usr/bin/env python3
"""M9 HEAT fold-0, Re-disjoint, sealed Sim mechanism screen.

This is deliberately a small point-only experiment.  It trains three
bitwise-matched arms on exactly the same query windows.  Every arm computes
the two history-internal support gradients; only the source of the resulting
per-sample fast state differs.  Held fields are opened only after the fit
normalizer, donor map and all endpoints are sealed.
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
from typing import Iterable, Mapping

import h5py
import numpy as np
# Must precede torch import for deterministic cuBLAS kernels (M8 lesson).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for _path in (CODE_ROOT, Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_flow_backbone as backbone  # noqa: E402
import train_temporal_transport_loao_screen as io  # noqa: E402
from models.history_episodic_adaptation import (  # noqa: E402
    HistoryEpisodicAdaptationConfig,
    HistoryEpisodicAdaptationOperator,
)

SCHEMA = "history_episodic_adaptation_re_ordinal_crossfit_v1"
ALLOWED_AOAS = (0, 5, 15, 20)
ARM_NAMES = ("A_static", "B_self", "C_donor")
SEED, TRAINING_RNG_SEED, FOLD = 210, 227, 0
UPDATES, BATCH_SIZE = 2_000, 8
LEARNING_RATE, WEIGHT_DECAY, MSE_WEIGHT, GRAD_CLIP = 3e-4, 1e-5, .25, 1.0
ACCURACY_CHANNEL_WEIGHTS = (1.0, 4.0, 1.0)
POINT_WEIGHTS = {"rel_l2_score": .46, "tke_score": .10, "mvpe_score": .09}
FAST_RADIUS, FAST_EPSILON = .10, 1.0e-8
ACKNOWLEDGEMENT = "HEAT_SIM_FOLD0_V1"
SOURCE_PATHS = {
    "trainer": Path(__file__),
    "model": CODE_ROOT / "models" / "history_episodic_adaptation.py",
    "temporal_transport": CODE_ROOT / "models" / "temporal_transport.py",
    "flow_backbone_driver": CODE_ROOT / "training" / "train_flow_backbone.py",
    "sealed_io_driver": CODE_ROOT / "training" / "train_temporal_transport_loao_screen.py",
    "official_v9_scoring": REPO_ROOT / "kits" / "track1" / "realpde_t1_starting_kit_v9" / "scoring.py",
}
SCORING = io.SCORING


def source_hashes() -> dict[str, str]:
    return {name: io.sha256_file(path) for name, path in SOURCE_PATHS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), default=FOLD)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def git_report() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(("git", *args), cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--untracked-files=all")
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"), "clean": not bool(dirty),
            "porcelain": dirty.splitlines()}


def scan_allowed_inventory(sim_dir: Path) -> list[dict[str, object]]:
    """Read metadata only for forbidden AoAs; fields only for allowed AoAs."""
    records: list[dict[str, object]] = []
    for path in sorted(sim_dir.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            if "aoa" not in handle or "re" not in handle:
                raise ValueError(f"{path.name} misses AoA/Re metadata")
            aoa = int(np.asarray(handle["aoa"]).reshape(-1)[0])
            re = int(np.asarray(handle["re"]).reshape(-1)[0])
            if aoa not in ALLOWED_AOAS:
                continue
            if any(key not in handle for key in ("u", "v", "p")):
                raise ValueError(f"{path.name} misses field dataset")
            shapes = [tuple(handle[key].shape) for key in ("u", "v", "p")]
            if not (shapes[0] == shapes[1] == shapes[2]):
                raise ValueError(f"{path.name} has unequal field shapes")
            records.append({"file": path.name, "aoa": aoa, "re": re,
                            "frames": int(shapes[0][0]), "shape": list(shapes[0])})
    if len(records) != 80:
        raise ValueError(f"expected 80 allowed trajectories, got {len(records)}")
    return records


def select_fold(records: list[dict[str, object]], fold: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if fold not in range(4):
        raise ValueError("fold must be ordinal modulo four")
    by_aoa: dict[int, list[dict[str, object]]] = {}
    for aoa in ALLOWED_AOAS:
        rows = sorted((row for row in records if int(row["aoa"]) == aoa),
                      key=lambda row: (int(row["re"]), str(row["file"])))
        if len(rows) != 20 or len({int(row["re"]) for row in rows}) != 20:
            raise ValueError(f"AoA {aoa} must have 20 distinct Re trajectories")
        by_aoa[aoa] = rows
    held = [row for aoa in ALLOWED_AOAS for ordinal, row in enumerate(by_aoa[aoa]) if ordinal % 4 == fold]
    train = [row for aoa in ALLOWED_AOAS for ordinal, row in enumerate(by_aoa[aoa]) if ordinal % 4 != fold]
    held_re = {int(row["re"]) for row in held}
    if len(train) != 60 or len(held) != 20 or any(int(row["re"]) in held_re for row in train):
        raise RuntimeError("expected Re-disjoint 60/20 fold")
    manifest = {"fold": fold, "allowed_aoas": list(ALLOWED_AOAS),
                "train_records_sha256": io.canonical_sha(train),
                "held_records_sha256": io.canonical_sha(held),
                "held_re_by_aoa": {str(aoa): [int(row["re"]) for ordinal, row in enumerate(by_aoa[aoa]) if ordinal % 4 == fold] for aoa in ALLOWED_AOAS}}
    return train, held, manifest


def load_fields(sim_dir: Path, records: Iterable[dict[str, object]]) -> tuple[list[torch.Tensor], list[object], list[dict[str, object]]]:
    trajectories, windows, window_rows = [], [], []
    window_type = backbone.common.scratch.SimWindow
    for trajectory, row in enumerate(records):
        with h5py.File(sim_dir / str(row["file"]), "r") as handle:
            values = [np.asarray(handle[key][:, ::2, ::2], dtype=np.float32)
                      for key in ("u", "v", "p")]
        value = torch.from_numpy(np.ascontiguousarray(np.stack(values, axis=-1)))
        if len(value) != int(row["frames"]):
            raise RuntimeError("frame count changed after inventory")
        trajectories.append(value)
        for start in range(0, len(value) - 40, 20):
            windows.append(window_type(trajectory, start))
            window_rows.append({"file": str(row["file"]), "aoa": int(row["aoa"]),
                                "re": int(row["re"]), "trajectory": trajectory, "start": start})
    return trajectories, windows, window_rows


def donor_indexes(window_rows: list[Mapping[str, object]]) -> np.ndarray:
    """Frozen next-Re/file same-AoA/same-start donor mapping."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(window_rows):
        groups.setdefault((int(row["aoa"]), int(row["start"])), []).append(index)
    result = np.empty(len(window_rows), dtype=np.int64)
    for key, indexes in groups.items():
        ordered = sorted(indexes, key=lambda index: (int(window_rows[index]["re"]), str(window_rows[index]["file"])))
        if len(ordered) < 2:
            raise ValueError(f"donor group {key} has fewer than two trajectories")
        for position, index in enumerate(ordered):
            donor = ordered[(position + 1) % len(ordered)]
            source, target = window_rows[index], window_rows[donor]
            if (source["file"] == target["file"] or source["aoa"] != target["aoa"]
                    or source["start"] != target["start"]):
                raise RuntimeError("invalid same-AoA/start donor map")
            result[index] = donor
    return result


def donor_map_audit(
        window_rows: list[Mapping[str, object]],
        mapping: np.ndarray,
        ) -> dict[str, object]:
    """Return a serializable proof of the frozen donor invariants."""
    if mapping.shape != (len(window_rows),) or mapping.dtype != np.int64:
        raise ValueError("donor mapping has the wrong shape or dtype")
    valid_index = bool(
        len(mapping) == 0
        or (int(mapping.min()) >= 0 and int(mapping.max()) < len(window_rows)))
    different_trajectory = valid_index and all(
        str(window_rows[index]["file"])
        != str(window_rows[int(donor)]["file"])
        for index, donor in enumerate(mapping))
    same_aoa = valid_index and all(
        int(window_rows[index]["aoa"])
        == int(window_rows[int(donor)]["aoa"])
        for index, donor in enumerate(mapping))
    same_start = valid_index and all(
        int(window_rows[index]["start"])
        == int(window_rows[int(donor)]["start"])
        for index, donor in enumerate(mapping))
    passed = bool(
        valid_index and different_trajectory and same_aoa and same_start)
    return {
        "windows": len(window_rows),
        "unique_source_files": len({str(row["file"]) for row in window_rows}),
        "valid_index": valid_index,
        "different_trajectory": different_trajectory,
        "same_aoa": same_aoa,
        "same_start": same_start,
        "passed": passed,
    }


def make_schedule(windows: list[object]) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, len(windows), size=(UPDATES, BATCH_SIZE), dtype=np.int64)


def make_config() -> HistoryEpisodicAdaptationConfig:
    return HistoryEpisodicAdaptationConfig(hidden_channels=128, spatial_blocks=2,
        spatial_downsample=2, temporal_blocks=3, temporal_kernel=5, groups=8,
        fast_radius=FAST_RADIUS, fast_epsilon=FAST_EPSILON, predict_uncertainty=False)


def construct_arms(device: torch.device) -> tuple[dict[str, HistoryEpisodicAdaptationOperator], dict[str, object]]:
    arms, states = {}, {}
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = HistoryEpisodicAdaptationOperator(make_config()).to(device)
        arms[name] = model
        states[name] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    reference = states[ARM_NAMES[0]]
    if any(state.keys() != reference.keys() or any(not torch.equal(state[key], reference[key]) for key in reference) for state in states.values()):
        raise RuntimeError("HEAT arms are not bitwise-identical at initialization")
    counts = {name: sum(value.numel() for value in model.parameters()) for name, model in arms.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"HEAT parameter mismatch {counts}")
    return arms, {"initial_state_sha256": io.state_sha(reference), "parameters": counts,
                  "bitwise_identical": True, "primary_comparison": "B_self_minus_A_static"}


def _normalise_input(value: torch.Tensor, normalizer) -> torch.Tensor:
    return (value - normalizer.mean_input) / normalizer.std_input


def _normalise_target(value: torch.Tensor, normalizer) -> torch.Tensor:
    return (value - normalizer.mean_target) / normalizer.std_target


def support_per_sample_loss(model: HistoryEpisodicAdaptationOperator, history: torch.Tensor,
                            normalizer, fast: torch.Tensor) -> torch.Tensor:
    """Delegate the frozen S1/S2/UV definition to the model API."""
    return model.support_loss(_normalise_input(history, normalizer), fast)


def adapted_state(model: HistoryEpisodicAdaptationOperator, history: torch.Tensor, normalizer) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | str]]:
    state, diagnostics = model.adapted_fast_state(
        _normalise_input(history, normalizer), radius=FAST_RADIUS,
        create_graph=False, mode="self")
    return state, diagnostics


def arm_fast_state(name: str, model, query_history: torch.Tensor, donor_history: torch.Tensor, normalizer):
    if name == "A_static":
        # Static mode still computes/differentiates S1/S2, but discards it.
        static, diagnostics = model.adapted_fast_state(
            _normalise_input(query_history, normalizer), radius=FAST_RADIUS,
            create_graph=False, mode="static")
        return static, diagnostics
    self_state, self_diagnostics = adapted_state(model, query_history, normalizer)
    if name == "B_self":
        return self_state, self_diagnostics
    donor_state, _ = adapted_state(model, donor_history, normalizer)
    return donor_state, self_diagnostics


def query_loss(model, history: torch.Tensor, target: torch.Tensor, fast: torch.Tensor, normalizer):
    prediction = model(_normalise_input(history, normalizer), output_steps=20, fast_state=fast)
    target_norm = _normalise_target(target, normalizer)
    accuracy, detail = backbone.common.accuracy_objective(prediction, target_norm, normalizer, ACCURACY_CHANNEL_WEIGHTS)
    mse = (prediction - target_norm).square().mean()
    return accuracy + MSE_WEIGHT * mse, prediction, {"accuracy": accuracy, "mse": mse, "loss": accuracy + MSE_WEIGHT * mse, **detail}


def train_arm(name: str, model, dataset, schedule: np.ndarray, donor_map: np.ndarray, normalizer, device: torch.device, output_dir: Path) -> dict[str, object]:
    model.set_normalizer_statistics(normalizer.mean_input, normalizer.mean_target, normalizer.std_input, normalizer.std_target)
    torch.manual_seed(TRAINING_RNG_SEED)
    if device.type == "cuda": torch.cuda.manual_seed_all(TRAINING_RNG_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    trace, bucket, started = [], [], time.perf_counter()
    for update, indexes in enumerate(schedule, 1):
        samples, donor_samples = ([dataset[int(index)] for index in indexes], [dataset[int(donor_map[int(index)])] for index in indexes])
        history = torch.stack([sample[0] for sample in samples]).to(device)
        target = torch.stack([sample[1] for sample in samples]).to(device)
        donor_history = torch.stack([sample[0] for sample in donor_samples]).to(device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        fast, support = arm_fast_state(name, model, history, donor_history, normalizer)
        loss, _prediction, terms = query_loss(model, history, target, fast, normalizer)
        if not torch.isfinite(loss) or not torch.isfinite(fast).all():
            raise FloatingPointError(f"{name}: non-finite update {update}")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step(); scheduler.step()
        bucket.append(float(loss.detach()))
        if update == 1 or update % 250 == 0:
            row = {"arm": name, "update": update, "loss_mean": float(np.mean(bucket)),
                   "support_loss_mean": float(support["per_sample_loss"].mean()), "fast_norm_mean": float(torch.linalg.vector_norm(fast, dim=1).mean()),
                   "elapsed_seconds": time.perf_counter() - started,
                   **{key: float(value.detach()) for key, value in terms.items()}}
            trace.append(row); bucket.clear(); print(json.dumps(row, sort_keys=True), flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    path = output_dir / f"{name}_endpoint.pth"
    torch.save({"schema": SCHEMA, "arm": name, "updates": UPDATES, "model_config": model.architecture_config(), "model_state_dict": state}, path)
    return {"path": str(path), "sha256": io.sha256_file(path), "model_state_sha256": io.state_sha(state), "trace": trace, "wall_seconds": time.perf_counter() - started}


def reload_sealed_arm(name: str, endpoint: Mapping[str, object], normalizer, device: torch.device):
    path = Path(str(endpoint["path"]))
    if io.sha256_file(path) != endpoint["sha256"]: raise RuntimeError(f"{name} endpoint changed before held load")
    payload = torch.load(path, map_location="cpu", weights_only=False); config = dict(payload["model_config"])
    if payload.get("schema") != SCHEMA or payload.get("arm") != name or config.pop("architecture", None) != "history_episodic_adaptation": raise RuntimeError("sealed endpoint identity mismatch")
    model = HistoryEpisodicAdaptationOperator(HistoryEpisodicAdaptationConfig(**config))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if io.state_sha({key: value.detach().cpu() for key, value in model.state_dict().items()}) != endpoint["model_state_sha256"]: raise RuntimeError("sealed state mismatch")
    model = model.to(device).eval(); model.set_normalizer_statistics(normalizer.mean_input, normalizer.mean_target, normalizer.std_input, normalizer.std_target)
    return model


def error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack((SCORING.rel_l2_per_sample(prediction, target, 2), SCORING.tke_rel_l2_per_sample(prediction, target, 2), SCORING.mvpe_rel_l2_per_sample(prediction, target, 2)), axis=1).astype(np.float64)


def score(errors: np.ndarray) -> dict[str, float | int]:
    mean = errors.mean(0, dtype=np.float64)
    result: dict[str, float | int] = {"samples": int(len(errors)), "rel_l2_score": SCORING.score_error(float(mean[0])), "tke_score": SCORING.score_error(float(mean[1])), "mvpe_score": SCORING.score_error(float(mean[2]))}
    result["point_contribution_46_10_9"] = sum(POINT_WEIGHTS[key] * float(result[key]) for key in POINT_WEIGHTS)
    return result


def metric_delta(candidate: Mapping[str, float | int], control: Mapping[str, float | int]) -> dict[str, float]:
    return {key: float(candidate[key]) - float(control[key]) for key in (*POINT_WEIGHTS, "point_contribution_46_10_9")}


def comparison(candidate: np.ndarray, control: np.ndarray, rows: list[Mapping[str, object]]) -> dict[str, object]:
    files = np.asarray([str(row["file"]) for row in rows]); aoas = np.asarray([int(row["aoa"]) for row in rows])
    per_file = {file: float(score(candidate[files == file])["point_contribution_46_10_9"] - score(control[files == file])["point_contribution_46_10_9"]) for file in sorted(set(files.tolist()))}
    per_aoa = {str(aoa): float(score(candidate[aoas == aoa])["point_contribution_46_10_9"] - score(control[aoas == aoa])["point_contribution_46_10_9"]) for aoa in ALLOWED_AOAS}
    return {"delta": metric_delta(score(candidate), score(control)), "per_trajectory_file_delta_point": per_file,
            "positive_trajectory_files": sum(value > 0.0 for value in per_file.values()), "worst_trajectory_delta_point": min(per_file.values()),
            "per_aoa_delta_point": per_aoa, "positive_aoa_groups": sum(value > 0.0 for value in per_aoa.values())}


def evaluate(arms: Mapping[str, object], trajectories, windows, rows, donor_map, normalizer, device: torch.device) -> dict[str, object]:
    dataset = backbone.common.scratch.SimWindowDataset(trajectories, windows, mask_probability=0., noise_scale=0.)
    errors = {name: [] for name in ARM_NAMES}; late_errors = {name: [] for name in ARM_NAMES}; support = {"self": [], "static": [], "donor": []}; observed_rows = []
    self_gradient_norms: list[torch.Tensor] = []; self_step_norms: list[torch.Tensor] = []
    override_errors = {"self": [], "static": [], "donor": []}
    replay_audit: dict[str, dict[str, object]] = {}
    all_gradient_finite = True
    all_state_finite = True
    all_output_finite = True
    for begin in range(0, len(dataset), 32):
        idx = list(range(begin, min(begin + 32, len(dataset))))
        samples, donors = ([dataset[index] for index in idx], [dataset[int(donor_map[index])] for index in idx])
        history = torch.stack([item[0] for item in samples]).to(device); target = torch.stack([item[1] for item in samples]); donor_history = torch.stack([item[0] for item in donors]).to(device)
        for name, model in arms.items():
            with torch.enable_grad():
                self_state, self_diagnostics = adapted_state(model, history, normalizer)
                donor_state, donor_diagnostics = adapted_state(model, donor_history, normalizer)
                static_state = model.new_fast_vector(len(history), device=device, dtype=history.dtype)
                all_gradient_finite = all_gradient_finite and bool(
                    torch.isfinite(self_diagnostics["self_gradient_norm"]).all()
                    and torch.isfinite(donor_diagnostics["self_gradient_norm"]).all())
                all_state_finite = all_state_finite and all(
                    bool(torch.isfinite(value).all())
                    for value in (self_state, donor_state, static_state))
                # The support-mechanism gate is a property of the B endpoint.
                # Never average three independently trained checkpoints here.
                if name == "B_self":
                    support["static"].append(support_per_sample_loss(model, history, normalizer, static_state).detach().cpu())
                    support["self"].append(support_per_sample_loss(model, history, normalizer, self_state).detach().cpu())
                    support["donor"].append(support_per_sample_loss(model, history, normalizer, donor_state).detach().cpu())
                    self_gradient_norms.append(self_diagnostics["self_gradient_norm"].detach().cpu())
                    self_step_norms.append(self_diagnostics["step_norm"].detach().cpu())
                state = {"A_static": static_state, "B_self": self_state, "C_donor": donor_state}[name]
                # Query predictions are evaluation-only.  Keeping their full
                # graphs while constructing replay states multiplies memory
                # by the number of arms/modes and can exceed a 24 GB GPU.
                with torch.no_grad():
                    prediction = model(
                        _normalise_input(history, normalizer),
                        output_steps=20,
                        fast_state=state)
                all_output_finite = all_output_finite and bool(
                    torch.isfinite(prediction).all())
                if name not in replay_audit:
                    permutation = torch.arange(len(history) - 1, -1, -1, device=device)
                    inverse = torch.argsort(permutation)
                    permuted_state, _ = arm_fast_state(
                        name,
                        model,
                        history.index_select(0, permutation),
                        donor_history.index_select(0, permutation),
                        normalizer,
                    )
                    with torch.no_grad():
                        permuted = model(_normalise_input(history.index_select(0, permutation), normalizer), output_steps=20, fast_state=permuted_state).index_select(0, inverse)
                    repeated_state, _ = arm_fast_state(
                        name, model, history, donor_history, normalizer)
                    with torch.no_grad():
                        repeated = model(_normalise_input(history, normalizer), output_steps=20, fast_state=repeated_state)
                    replay_audit[name] = {"batch_permutation_restored_bitwise": bool(torch.equal(prediction.detach(), permuted.detach())), "batch_permutation_max_abs_error": float((prediction.detach() - permuted.detach()).abs().max()), "independent_call_bitwise": bool(torch.equal(prediction.detach(), repeated.detach())), "independent_call_max_abs_error": float((prediction.detach() - repeated.detach()).abs().max())}
                    all_state_finite = all_state_finite and bool(
                        torch.isfinite(permuted_state).all()
                        and torch.isfinite(repeated_state).all())
                    all_output_finite = all_output_finite and bool(
                        torch.isfinite(permuted).all()
                        and torch.isfinite(repeated).all())
                    if name == "B_self":
                        permuted_history = history.index_select(0, permutation)
                        permuted_donor_history = donor_history.index_select(
                            0, permutation)
                        permuted_self, _ = adapted_state(
                            model, permuted_history, normalizer)
                        permuted_donor, _ = adapted_state(
                            model, permuted_donor_history, normalizer)
                        permuted_static = model.new_fast_vector(
                            len(history), device=device, dtype=history.dtype)
                        repeated_self, _ = adapted_state(
                            model, history, normalizer)
                        repeated_donor, _ = adapted_state(
                            model, donor_history, normalizer)
                        repeated_static = model.new_fast_vector(
                            len(history), device=device, dtype=history.dtype)
                        mode_states = {
                            "self": (self_state, permuted_self, repeated_self),
                            "static": (
                                static_state, permuted_static,
                                repeated_static),
                            "donor": (
                                donor_state, permuted_donor,
                                repeated_donor),
                        }
                        for mode, (
                                original_mode_state,
                                permuted_mode_state,
                                repeated_mode_state) in mode_states.items():
                            with torch.no_grad():
                                original_mode_prediction = model(
                                    _normalise_input(history, normalizer),
                                    output_steps=20,
                                    fast_state=original_mode_state)
                                permuted_mode_prediction = model(
                                    _normalise_input(
                                        permuted_history, normalizer),
                                    output_steps=20,
                                    fast_state=permuted_mode_state,
                                ).index_select(0, inverse)
                                repeated_mode_prediction = model(
                                    _normalise_input(history, normalizer),
                                    output_steps=20,
                                    fast_state=repeated_mode_state)
                            replay_audit[f"B_override_{mode}"] = {
                                "batch_permutation_restored_bitwise": bool(
                                    torch.equal(
                                        original_mode_prediction.detach(),
                                        permuted_mode_prediction.detach())),
                                "batch_permutation_max_abs_error": float((
                                    original_mode_prediction.detach()
                                    - permuted_mode_prediction.detach()
                                ).abs().max()),
                                "independent_call_bitwise": bool(torch.equal(
                                    original_mode_prediction.detach(),
                                    repeated_mode_prediction.detach())),
                                "independent_call_max_abs_error": float((
                                    original_mode_prediction.detach()
                                    - repeated_mode_prediction.detach()
                                ).abs().max()),
                            }
                            all_state_finite = all_state_finite and all(
                                bool(torch.isfinite(value).all())
                                for value in (
                                    original_mode_state,
                                    permuted_mode_state,
                                    repeated_mode_state))
                            all_output_finite = all_output_finite and all(
                                bool(torch.isfinite(value).all())
                                for value in (
                                    original_mode_prediction,
                                    permuted_mode_prediction,
                                    repeated_mode_prediction))
            physical = (prediction * normalizer.std_target + normalizer.mean_target).detach().cpu().numpy()
            error = error_rows(physical, target.numpy()); errors[name].append(error); late_errors[name].append(error_rows(physical[:, 10:], target.numpy()[:, 10:]))
            if name == "B_self":
                for mode, value in (("self", self_state), ("static", static_state), ("donor", donor_state)):
                    with torch.no_grad():
                        p = model(_normalise_input(history, normalizer), output_steps=20, fast_state=value)
                    all_output_finite = all_output_finite and bool(
                        torch.isfinite(p).all())
                    override_errors[mode].append(error_rows((p * normalizer.std_target + normalizer.mean_target).cpu().numpy(), target.numpy()))
        observed_rows.extend(rows[index] for index in idx)
    joined = {name: np.concatenate(value) for name, value in errors.items()}; late = {name: np.concatenate(value) for name, value in late_errors.items()}; overrides = {name: score(np.concatenate(value)) for name, value in override_errors.items()}
    gradient_norm = torch.cat(self_gradient_norms); step_norm = torch.cat(self_step_norms)
    nonzero = gradient_norm > FAST_EPSILON
    expected_replays = set(ARM_NAMES) | {
        "B_override_self", "B_override_static", "B_override_donor"}
    if set(replay_audit) != expected_replays:
        raise RuntimeError("all arm and B-override replay audits were not executed")
    mechanical = {"samples": int(len(gradient_norm)), "all_gradient_finite": all_gradient_finite and bool(torch.isfinite(gradient_norm).all()), "all_state_finite": all_state_finite, "all_output_finite": all_output_finite, "all_step_finite": bool(torch.isfinite(step_norm).all()), "nonzero_gradient_samples": int(nonzero.sum()), "zero_gradient_fraction": float((~nonzero).float().mean()), "nonzero_step_radius": FAST_RADIUS, "nonzero_step_radius_max_abs_error": float((step_norm[nonzero] - FAST_RADIUS).abs().max()) if bool(nonzero.any()) else None, "nondegenerate_step_radius_exact": bool(nonzero.any()) and bool(torch.allclose(step_norm[nonzero], torch.full_like(step_norm[nonzero], FAST_RADIUS), rtol=0., atol=1e-6)), "replay": replay_audit, "donor_structure": donor_map_audit(rows, donor_map)}
    return {"metrics": {name: score(value) for name, value in joined.items()}, "horizons_11_20": {name: score(value) for name, value in late.items()},
            "B_self_minus_A_static": comparison(joined["B_self"], joined["A_static"], observed_rows),
            "B_self_minus_C_donor": comparison(joined["B_self"], joined["C_donor"], observed_rows),
            "B_same_checkpoint_overrides": {"metrics": overrides, "self_minus_donor": metric_delta(overrides["self"], overrides["donor"])},
            "support_mechanism": {key: float(torch.cat(value).mean()) for key, value in support.items()}, "mechanical_fast_state_audit": mechanical}


def gate(evaluation: Mapping[str, object]) -> tuple[bool, dict[str, object]]:
    support = evaluation["support_mechanism"]
    mechanics = evaluation["mechanical_fast_state_audit"]
    mechanism = (float(support["self"]) < float(support["static"]) and float(support["self"]) < float(support["donor"])
                 and bool(mechanics["all_gradient_finite"]) and bool(mechanics["all_state_finite"])
                 and bool(mechanics["all_output_finite"]) and bool(mechanics["all_step_finite"])
                 and bool(mechanics["nondegenerate_step_radius_exact"])
                 and bool(mechanics["donor_structure"]["passed"])
                 and all(bool(value["batch_permutation_restored_bitwise"])
                         and bool(value["independent_call_bitwise"])
                         for value in mechanics["replay"].values()))
    primary, donor = evaluation["B_self_minus_A_static"], evaluation["B_self_minus_C_donor"]
    delta = primary["delta"]
    passed = mechanism and float(delta["point_contribution_46_10_9"]) >= .05 and all(float(delta[key]) >= -.03 for key in POINT_WEIGHTS) and float(evaluation["horizons_11_20"]["B_self"]["point_contribution_46_10_9"] - evaluation["horizons_11_20"]["A_static"]["point_contribution_46_10_9"]) >= 0. and int(primary["positive_trajectory_files"]) >= 15 and float(primary["worst_trajectory_delta_point"]) >= -.06 and int(primary["positive_aoa_groups"]) >= 3 and float(donor["delta"]["point_contribution_46_10_9"]) >= .03 and float(evaluation["B_same_checkpoint_overrides"]["self_minus_donor"]["point_contribution_46_10_9"]) >= .02
    return passed, {"support_mechanism_pass": mechanism, "primary_point_min": .05, "component_floor": -.03, "late_horizon_point_min": 0., "positive_trajectories_min": 15, "worst_trajectory_min": -.06, "positive_aoas_min": 3, "donor_point_min": .03, "same_checkpoint_self_minus_donor_point_min": .02}


def smoke(device: torch.device) -> None:
    arms, identity = construct_arms(device); normalizer = type("N", (), {"mean_input": torch.zeros(3, device=device), "mean_target": torch.zeros(3, device=device), "std_input": torch.ones(3, device=device), "std_target": torch.ones(3, device=device)})()
    history, target = torch.randn(2, 20, 32, 64, 3, device=device), torch.randn(2, 20, 32, 64, 3, device=device)
    for name, model in arms.items():
        model.set_normalizer_statistics(normalizer.mean_input, normalizer.mean_target, normalizer.std_input, normalizer.std_target); fast, _ = arm_fast_state(name, model, history, history.flip(0), normalizer); loss, prediction, _ = query_loss(model, history, target, fast, normalizer); loss.backward(); assert torch.isfinite(prediction).all(); model.zero_grad(set_to_none=True)
    print(json.dumps({"smoke": "PASS", "identity": identity}, sort_keys=True))


def configure_determinism() -> dict[str, object]:
    """Apply and attest the complete deterministic-kernel contract."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    return {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def run(args: argparse.Namespace, device: torch.device) -> None:
    if not args.execute or args.acknowledge_protocol != ACKNOWLEDGEMENT: raise PermissionError("formal run requires --execute and fixed acknowledgement")
    if args.fold != FOLD: raise PermissionError("M9 preregistration permits only fold0")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError("refusing to overwrite output")
    determinism = configure_determinism()
    git = git_report()
    if not git["clean"]: raise RuntimeError(f"worktree must be clean: {git['porcelain']}")
    inventory = scan_allowed_inventory(args.sim_dir); fit_records, held_records, fold_manifest = select_fold(inventory, args.fold)
    args.output_dir.mkdir(parents=True); config = {"fold": FOLD, "updates": UPDATES, "batch_size": BATCH_SIZE, "model_rng": SEED, "training_rng": TRAINING_RNG_SEED, "arms": list(ARM_NAMES), "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "fast_radius": FAST_RADIUS, "fast_epsilon": FAST_EPSILON, "support": {"s1": "[0:12]->[12:16]", "s2": "[4:16]->[16:20]", "loss": "equal normalized uv mse"}, "source_sha256": source_hashes(), **fold_manifest, "held_fields_loaded_before_all_endpoint_seals": False}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    environment = {"python": sys.version.replace("\n", " "), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "pid": os.getpid()}; (args.output_dir / "environment.txt").write_text("\n".join(f"{key}={value}" for key, value in environment.items()) + "\n", encoding="utf-8")
    trajectories, windows, rows = load_fields(args.sim_dir, fit_records)
    if len(windows) != 2880: raise RuntimeError(f"expected 2880 fit windows, got {len(windows)}")
    normalizer = backbone.common.scratch.ChannelNormalizer(*backbone.common.scratch.window_statistics(trajectories, windows), device=device)
    nstate = {key: getattr(normalizer, key).detach().cpu() for key in ("mean_input", "mean_target", "std_input", "std_target")}; npath = args.output_dir / "fit_only_normalizer.pt"; torch.save({"schema": SCHEMA, **nstate}, npath)
    normalizer_artifact = {"path": str(npath), "sha256": io.sha256_file(npath), "state_sha256": io.state_sha(nstate)}
    donor_map, schedule = donor_indexes(rows), make_schedule(windows); donor_path = args.output_dir / "fit_donor_map.npy"; np.save(donor_path, donor_map)
    fit_donor_audit = donor_map_audit(rows, donor_map)
    if not fit_donor_audit["passed"]: raise RuntimeError("fit donor audit failed")
    donor_artifact = {"path": str(donor_path), "sha256": io.sha256_file(donor_path), "mapping_sha256": hashlib.sha256(donor_map.tobytes()).hexdigest()}
    dataset = backbone.common.scratch.SimWindowDataset(trajectories, windows, mask_probability=0., noise_scale=0.); arms, identity = construct_arms(device)
    endpoints = {name: train_arm(name, arms[name], dataset, schedule, donor_map, normalizer, device, args.output_dir) for name in ARM_NAMES}
    seal = {"schema": SCHEMA, "normalizer": normalizer_artifact, "donor_map": donor_artifact, "endpoints": {name: {key: endpoints[name][key] for key in ("path", "sha256", "model_state_sha256")} for name in ARM_NAMES}}; seal_path = args.output_dir / "all_endpoints_seal.json"; seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all(io.sha256_file(Path(str(value["path"]))) == value["sha256"] for value in seal["endpoints"].values()): raise RuntimeError("endpoint seal failed")
    held_trajectories, held_windows, held_rows = load_fields(args.sim_dir, held_records)
    if len(held_windows) != 960: raise RuntimeError(f"expected 960 held windows, got {len(held_windows)}")
    held_donor = donor_indexes(held_rows)
    held_donor_audit = donor_map_audit(held_rows, held_donor)
    if not held_donor_audit["passed"]: raise RuntimeError("held donor audit failed")
    fit_files = {str(row["file"]) for row in rows}; held_files = {str(row["file"]) for row in held_rows}
    donor_isolation = {"fit": fit_donor_audit, "held": held_donor_audit,
                       "cross_split_file_disjoint": fit_files.isdisjoint(held_files)}
    if not donor_isolation["cross_split_file_disjoint"]: raise RuntimeError("donor splits overlap")
    sealed_arms = {name: reload_sealed_arm(name, endpoints[name], normalizer, device) for name in ARM_NAMES}
    evaluation = evaluate(sealed_arms, held_trajectories, held_windows, held_rows, held_donor, normalizer, device); passed, gate_config = gate(evaluation)
    decision = "GO_FOLD1" if passed else ("STOP_M9_SUPPORT_MECHANISM" if not gate_config["support_mechanism_pass"] else "STOP_M9_HEAT")
    result = {"schema": SCHEMA, "evidence_class": "Sim_Re_ordinal_mod4_fold0", "config": config, "git": git, "determinism": determinism, "initial_identity": identity, "fit_trajectories": len(fit_records), "fit_windows": len(windows), "held_trajectories": len(held_records), "held_windows": len(held_windows), "sample_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(), "fit_donor_map": donor_artifact, "held_donor_map_sha256": hashlib.sha256(held_donor.tobytes()).hexdigest(), "donor_isolation": donor_isolation, "normalizer": normalizer_artifact, "endpoints": endpoints, "endpoint_seal": {"path": str(seal_path), "sha256": io.sha256_file(seal_path)}, "evaluation": evaluation, "gate": gate_config, "decision": decision}
    path = args.output_dir / "result.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"); print(json.dumps({"decision": decision, "result": str(path), "self_minus_static": evaluation["B_self_minus_A_static"]["delta"]}, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("HEAT formal screen requires CUDA")
    if args.smoke:
        configure_determinism()
        smoke(device)
    else:
        run(args, device)


if __name__ == "__main__": main()
