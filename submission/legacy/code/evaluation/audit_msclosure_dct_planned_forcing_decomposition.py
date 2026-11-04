#!/usr/bin/env python3
"""Audit realized DCT-plan and closure duties from one trained endpoint.

Each Sim window is passed through ``forward_components`` exactly once.  The
same output then supplies both the native prediction and a zero-parameter hard
K4 replacement which removes the realized non-plan K4 contribution.  The
replacement never reads the target; targets are used only for metrics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402
from models.msclosure_dct_planned_forcing import (  # noqa: E402
    DCTPlannedForcingMultipleShootingClosure,
    DCTPlannedForcingMultipleShootingConfig,
)


SCHEMA = "msclosure_dct_planned_forcing_decomposition_audit_v1"
ACKNOWLEDGEMENT = "RUN_MS_DCT_PLANNED_FORCING_DECOMPOSITION_AUDIT"
FIT_AOAS = (0, 5, 15, 20)
HELD_AOA = 10
# Finish every fit-side decomposition before opening the held panel.  This is
# an audit rather than checkpoint selection, but retaining the access order
# makes the evidence boundary explicit in the artifact.
AOAS = (*FIT_AOAS, HELD_AOA)
ARMS = ("current", "hard_k4_replacement")
DEFAULT_BATCH_SIZE = 16


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(
        path: Path, device: torch.device,
        ) -> tuple[DCTPlannedForcingMultipleShootingClosure,
                   base.ChannelNormalizer, dict[str, object]]:
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    required = {"model_config", "rollout_mode", "normalizer", "model_state"}
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        missing = required.difference(payload if isinstance(payload, Mapping)
                                      else ())
        raise ValueError(
            f"DCT planned-forcing checkpoint misses {sorted(missing)}")
    if payload["rollout_mode"] != "carry_detached":
        raise ValueError("DCT planned-forcing endpoint must use carry_detached")
    config = DCTPlannedForcingMultipleShootingConfig(
        **dict(payload["model_config"]))
    model = DCTPlannedForcingMultipleShootingClosure(
        config, rollout_mode="carry_detached").float()
    model.load_state_dict(payload["model_state"], strict=True)

    statistics = payload["normalizer"]
    normalizer = base.ChannelNormalizer(
        torch.tensor(statistics["mean_input"]),
        torch.tensor(statistics["mean_target"]),
        torch.tensor(statistics["std_input"]),
        torch.tensor(statistics["std_target"]), device)
    expected = {
        "mean_input": normalizer.mean_input.cpu(),
        "mean_target": normalizer.mean_target.cpu(),
        "std_input": normalizer.std_input.cpu(),
        "std_target": normalizer.std_target.cpu(),
    }
    if not all(torch.equal(getattr(model, name), value)
               for name, value in expected.items()):
        raise RuntimeError("checkpoint normalizer metadata and buffers disagree")
    return model.to(device).eval(), normalizer, dict(payload)


def _coefficient_mask(
        free_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if (free_mask.ndim != 4 or free_mask.shape[-1] != 1
            or free_mask.shape[0] != reference.shape[0]
            or free_mask.shape[1:3] != reference.shape[-2:]):
        raise ValueError("free mask must have shape [B,H,W,1]")
    return free_mask.permute(0, 3, 1, 2).unsqueeze(2).to(reference)


def _assert_numerically_close(
        actual: torch.Tensor, expected: torch.Tensor, name: str,
        ) -> tuple[float, float]:
    if actual.shape != expected.shape:
        raise RuntimeError(f"{name} shape mismatch")
    error = (actual - expected).abs().max()
    scale = torch.maximum(actual.abs().max(), expected.abs().max()).clamp_min(1)
    tolerance = 128 * torch.finfo(actual.dtype).eps * scale
    if not bool(torch.isfinite(error)) or bool(error > tolerance):
        raise RuntimeError(
            f"{name} failed: max_abs={float(error)} tolerance={float(tolerance)}")
    return float(error), float(tolerance)


@torch.no_grad()
def decompose_and_hard_replace(
        model: DCTPlannedForcingMultipleShootingClosure,
        prediction: torch.Tensor, applied_plan: torch.Tensor,
        free_mask: torch.Tensor,
        ) -> dict[str, object]:
    """Remove only the realized non-plan K4 component from scored UV."""
    if (prediction.ndim != 5 or prediction.shape[-1] != 3
            or applied_plan.shape != prediction[..., :2].shape):
        raise ValueError("prediction/plan shapes must be [B,20,H,W,3]/[...,2]")
    current_uv = prediction[..., :2]
    coefficient = model.plan_head.temporal_coefficients
    reconstruct = model.plan_head.reconstruct
    output_coefficients = coefficient(current_uv)
    plan_coefficients = coefficient(applied_plan)
    dynamic_coefficients = coefficient(current_uv - applied_plan)
    add_error, add_tolerance = _assert_numerically_close(
        output_coefficients, plan_coefficients + dynamic_coefficients,
        "coefficient additivity")

    dynamic_k4 = reconstruct(dynamic_coefficients)
    free_time = free_mask.unsqueeze(1)
    replacement_uv = torch.where(
        free_time, current_uv - dynamic_k4, current_uv)
    replacement = torch.cat((replacement_uv, prediction[..., 2:]), dim=-1)
    if not torch.equal(replacement[..., 2:], prediction[..., 2:]):
        raise RuntimeError("hard K4 replacement changed pressure")
    fixed_time = (~free_time).expand_as(current_uv)
    if not torch.equal(replacement_uv[fixed_time], current_uv[fixed_time]):
        raise RuntimeError("hard K4 replacement changed fixed geometry")

    replacement_coefficients = coefficient(replacement_uv)
    replacement_error, replacement_tolerance = _assert_numerically_close(
        replacement_coefficients * _coefficient_mask(
            free_mask, replacement_coefficients),
        plan_coefficients * _coefficient_mask(free_mask, plan_coefficients),
        "replacement-plan coefficient identity")
    mean_error, mean_tolerance = _assert_numerically_close(
        replacement_uv.mean(dim=1), current_uv.mean(dim=1),
        "replacement temporal mean")
    return {
        "current_prediction": prediction,
        "replacement_prediction": replacement,
        "output_coefficients": output_coefficients,
        "plan_coefficients": plan_coefficients,
        "dynamic_coefficients": dynamic_coefficients,
        "replacement_coefficients": replacement_coefficients,
        "invariants": {
            "coefficient_additivity_max_abs": add_error,
            "coefficient_additivity_tolerance": add_tolerance,
            "replacement_plan_coefficient_max_abs": replacement_error,
            "replacement_plan_coefficient_tolerance": replacement_tolerance,
            "temporal_mean_max_abs_delta": mean_error,
            "temporal_mean_tolerance": mean_tolerance,
            "pressure_bitwise_equal": True,
            "fixed_geometry_bitwise_equal": True,
        },
    }


@torch.no_grad()
def coefficient_diagnostic_rows(
        decomposition: Mapping[str, object], truth_coefficients: torch.Tensor,
        free_mask: torch.Tensor,
        ) -> dict[str, np.ndarray]:
    plan = decomposition["plan_coefficients"].float()
    dynamic = decomposition["dynamic_coefficients"].float()
    output = decomposition["output_coefficients"].float()
    truth = truth_coefficients.float()
    mask = _coefficient_mask(free_mask, truth).expand_as(truth)
    vectors = {
        "plan": (plan * mask).flatten(1),
        "dynamic": (dynamic * mask).flatten(1),
        "output": (output * mask).flatten(1),
        "truth": (truth * mask).flatten(1),
    }
    norms = {
        name: torch.linalg.vector_norm(value, dim=1)
        for name, value in vectors.items()}
    epsilon = torch.finfo(torch.float32).eps

    def relative(left: str, right: str) -> torch.Tensor:
        return torch.linalg.vector_norm(
            vectors[left] - vectors[right], dim=1
        ) / norms[right].clamp_min(epsilon)

    def cosine(left: str, right: str) -> torch.Tensor:
        denominator = norms[left] * norms[right]
        value = (vectors[left] * vectors[right]).sum(dim=1)
        return torch.where(
            denominator > epsilon, value / denominator.clamp_min(epsilon),
            torch.zeros_like(value))

    plan_square = norms["plan"].square()
    alignment = (vectors["dynamic"] * vectors["plan"]).sum(dim=1)
    alignment = torch.where(
        plan_square > epsilon, alignment / plan_square.clamp_min(epsilon),
        torch.zeros_like(alignment))
    coefficient_count = mask.flatten(1).sum(dim=1).clamp_min(1)
    rows = {
        "plan_truth_relative_frobenius": relative("plan", "truth"),
        "output_truth_relative_frobenius": relative("output", "truth"),
        "dynamic_truth_relative_frobenius": relative("dynamic", "truth"),
        "plan_truth_cosine": cosine("plan", "truth"),
        "dynamic_truth_cosine": cosine("dynamic", "truth"),
        "plan_dynamic_cosine": cosine("plan", "dynamic"),
        "plan_truth_norm_ratio": norms["plan"] / norms["truth"].clamp_min(epsilon),
        "dynamic_truth_norm_ratio": (
            norms["dynamic"] / norms["truth"].clamp_min(epsilon)),
        "output_truth_norm_ratio": (
            norms["output"] / norms["truth"].clamp_min(epsilon)),
        "dynamic_along_plan_coefficient": alignment,
        "plan_free_rms": norms["plan"] / coefficient_count.sqrt(),
        "dynamic_free_rms": norms["dynamic"] / coefficient_count.sqrt(),
        "output_free_rms": norms["output"] / coefficient_count.sqrt(),
        "truth_free_rms": norms["truth"] / coefficient_count.sqrt(),
        "plan_nonzero": (norms["plan"] > epsilon).to(torch.float32),
    }
    return {
        name: value.detach().double().cpu().numpy()
        for name, value in rows.items()}


def _score_with_errors(errors: np.ndarray) -> dict[str, object]:
    means = errors.mean(axis=0, dtype=np.float64)
    return {
        **base.score(errors),
        "mean_physical_errors": {
            "rel_l2_error": float(means[0]),
            "tke_rel_l2_error": float(means[1]),
            "mvpe_rel_l2_error": float(means[2]),
        },
    }


def _comparison(
        candidate: Mapping[str, object], control: Mapping[str, object],
        ) -> dict[str, object]:
    score_delta = base.comparison(candidate, control)
    candidate_error = candidate["mean_physical_errors"]
    control_error = control["mean_physical_errors"]
    return {
        **score_delta,
        "physical_error_delta_replacement_minus_current": {
            name: float(candidate_error[name]) - float(control_error[name])
            for name in candidate_error},
    }


def _mean_diagnostics(
        joined: Mapping[str, np.ndarray], selected: np.ndarray | None = None,
        ) -> dict[str, float]:
    return {
        name: float((value if selected is None else value[selected]).mean(
            dtype=np.float64))
        for name, value in joined.items()}


@torch.inference_mode()
def evaluate_dataset_once(
        model: DCTPlannedForcingMultipleShootingClosure, dataset,
        labels: list[str], normalizer: base.ChannelNormalizer,
        device: torch.device, *, batch_size: int = DEFAULT_BATCH_SIZE,
        ) -> dict[str, object]:
    """Evaluate both arms from one model forward per input batch."""
    if len(dataset) != len(labels) or len(dataset) == 0:
        raise ValueError("dataset and nonempty label list must have equal length")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    errors = {arm: [] for arm in ARMS}
    prefix_errors = {arm: [[] for _ in range(20)] for arm in ARMS}
    diagnostics: dict[str, list[np.ndarray]] = {}
    prediction_digests = {arm: hashlib.sha256() for arm in ARMS}
    ordered_labels: list[str] = []
    invariant_maxima: dict[str, float] = {}
    invariant_booleans = {
        "pressure_bitwise_equal": True,
        "fixed_geometry_bitwise_equal": True,
    }
    forward_calls = 0

    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[index] for index in indices]
        history_physical = torch.stack([item[0] for item in items]).to(device)
        target_physical = torch.stack([item[1] for item in items]).to(device)
        history, target = normalizer.preprocess(
            history_physical, target_physical)
        values = model.forward_components(history)
        forward_calls += 1
        decomposition = decompose_and_hard_replace(
            model, values["prediction"], values["applied_plan_normalized"],
            values["free_mask"])
        truth_coefficients = model.plan_head.temporal_coefficients(
            target[..., :2])
        rows = coefficient_diagnostic_rows(
            decomposition, truth_coefficients, values["free_mask"])
        for name, value in rows.items():
            diagnostics.setdefault(name, []).append(value)
        for name, value in decomposition["invariants"].items():
            if isinstance(value, bool):
                invariant_booleans[name] &= value
            elif name.endswith("max_abs") or name.endswith("delta"):
                invariant_maxima[name] = max(
                    invariant_maxima.get(name, 0.0), float(value))
            elif name.endswith("tolerance"):
                invariant_maxima[name] = max(
                    invariant_maxima.get(name, 0.0), float(value))

        target_array = np.ascontiguousarray(
            target_physical.float().cpu().numpy())
        for arm, key in zip(
                ARMS, ("current_prediction", "replacement_prediction")):
            normalized = decomposition[key]
            physical = (
                normalized * normalizer.std_target
                + normalizer.mean_target)
            array = np.ascontiguousarray(physical.float().cpu().numpy())
            if not np.isfinite(array).all():
                raise FloatingPointError(f"{arm} prediction is non-finite")
            prediction_digests[arm].update(array.tobytes())
            errors[arm].append(base.error_rows(array, target_array))
            for prefix in range(1, 21):
                prefix_errors[arm][prefix - 1].append(base.error_rows(
                    array[:, :prefix], target_array[:, :prefix]))
        ordered_labels.extend(labels[index] for index in indices)

    joined_errors = {
        arm: np.concatenate(parts) for arm, parts in errors.items()}
    joined_diagnostics = {
        name: np.concatenate(parts) for name, parts in diagnostics.items()}
    aggregate = {
        arm: _score_with_errors(value)
        for arm, value in joined_errors.items()}
    prefixes = {}
    for prefix in range(1, 21):
        scored = {
            arm: _score_with_errors(np.concatenate(
                prefix_errors[arm][prefix - 1]))
            for arm in ARMS}
        prefixes[str(prefix)] = {
            **scored,
            "delta_replacement_minus_current": _comparison(
                scored[ARMS[1]], scored[ARMS[0]]),
            "prefix_frames": prefix,
            "tke_degenerate_single_frame": prefix == 1,
        }

    label_array = np.asarray(ordered_labels)
    per_trajectory = {}
    trajectory_deltas = []
    for label in sorted(set(ordered_labels)):
        selected = label_array == label
        scored = {
            arm: _score_with_errors(joined_errors[arm][selected])
            for arm in ARMS}
        delta = _comparison(scored[ARMS[1]], scored[ARMS[0]])
        trajectory_deltas.append(float(delta["point_contribution_46_10_9"]))
        per_trajectory[label] = {
            **scored,
            "delta_replacement_minus_current": delta,
            "decomposition": _mean_diagnostics(
                joined_diagnostics, selected),
        }
    return {
        "aggregate": aggregate,
        "delta_replacement_minus_current": _comparison(
            aggregate[ARMS[1]], aggregate[ARMS[0]]),
        "decomposition": _mean_diagnostics(joined_diagnostics),
        "invariants": {
            **invariant_maxima,
            **invariant_booleans,
            "validated": all(invariant_booleans.values()),
        },
        "per_prefix_horizon": prefixes,
        "per_trajectory": per_trajectory,
        "prediction_sha256": {
            arm: digest.hexdigest()
            for arm, digest in prediction_digests.items()},
        "samples": len(dataset),
        "trajectory_count": len(per_trajectory),
        "windows_per_trajectory": base.EXPECTED_WINDOWS_PER_TRAJECTORY,
        "positive_trajectories": sum(
            delta > 0.0 for delta in trajectory_deltas),
        "positive_trajectories_definition": (
            "delta point_contribution_46_10_9 > 0.0"),
        "worst_trajectory_delta_point": min(trajectory_deltas),
        "forward_calls": forward_calls,
        "single_forward_supplies_both_arms": True,
    }


def _validate_complete_aoa_panel(labels: list[str]) -> None:
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    if (len(counts) != base.EXPECTED_FILES_PER_AOA
            or set(counts.values()) != {base.EXPECTED_WINDOWS_PER_TRAJECTORY}):
        raise RuntimeError(f"incomplete Sim AoA panel: {counts}")


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError(
            f"audit requires --execute --acknowledge {ACKNOWLEDGEMENT}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal all-AoA decomposition audit requires CUDA")
    if args.checkpoint is None or args.sim_dir is None or args.output is None:
        raise ValueError("audit requires --checkpoint, --sim-dir, and --output")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model, normalizer, payload = load_checkpoint(args.checkpoint, device)
    inventory = base.inventory(args.sim_dir)
    evaluations = {}
    inventory_sha256 = {}
    for aoa in AOAS:
        rows = [row for row in inventory if int(row["aoa"]) == aoa]
        inventory_sha256[str(aoa)] = base.canonical_sha(rows)
        trajectories, windows, labels = base.load_fields(args.sim_dir, rows)
        _validate_complete_aoa_panel(labels)
        dataset = base.SimWindowDataset(trajectories, windows)
        evaluation = evaluate_dataset_once(
            model, dataset, labels, normalizer, device,
            batch_size=args.batch_size)
        evaluation["role"] = "held" if aoa == HELD_AOA else "fit"
        evaluations[str(aoa)] = evaluation
        print(json.dumps({
            "event": "dct_planned_forcing_decomposition_aoa_complete",
            "aoa": aoa,
            "role": evaluation["role"],
            "delta": evaluation["delta_replacement_minus_current"],
            "decomposition": evaluation["decomposition"],
        }, sort_keys=True), flush=True)
        del dataset, trajectories
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "schema": SCHEMA,
        "git": base.git_report(),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": file_sha256(args.checkpoint),
            "schema": payload.get("schema"),
            "endpoint_kind": payload.get("endpoint_kind"),
            "updates": payload.get("updates"),
            "model_state_sha256": base.state_sha(payload["model_state"]),
        },
        "protocol": {
            "fit_aoas": list(FIT_AOAS),
            "held_aoa": HELD_AOA,
            "arms": list(ARMS),
            "batch_size": args.batch_size,
            "one_forward_per_batch_for_both_arms": True,
            "decomposition_coordinates": "target-normalized UV",
            "coefficient_scope": "per-sample free-fluid DCT-K4 Frobenius",
            "hard_replacement": "x-Q_K(x-p); pressure/fixed geometry unchanged",
            "replacement_target_access": False,
            "target_role": "metrics and decomposition truth only",
            "no_scale_scan": True,
            "selection_gate": None,
            "inventory_sha256_by_aoa": inventory_sha256,
        },
        "architecture": model.architecture_config(),
        "fit_by_aoa": {
            str(aoa): evaluations[str(aoa)] for aoa in FIT_AOAS},
        "held_AoA10": evaluations[str(HELD_AOA)],
        "wall_seconds": time.perf_counter() - started,
    }
    atomic.atomic_json_save(result, args.output)
    print(json.dumps({
        "event": "dct_planned_forcing_decomposition_complete",
        "output": str(args.output.resolve()),
        "held_delta": evaluations[str(HELD_AOA)][
            "delta_replacement_minus_current"],
    }, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "audit"), default="contract")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps({
            "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "fit_aoas": list(FIT_AOAS),
            "held_aoa": HELD_AOA,
            "arms": list(ARMS),
            "hard_replacement": "x-Q_K(x-p)",
            "single_forward_supplies_both_arms": True,
        }, sort_keys=True, indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
