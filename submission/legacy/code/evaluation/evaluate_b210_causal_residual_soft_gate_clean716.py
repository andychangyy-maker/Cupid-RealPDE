#!/usr/bin/env python3
"""Strict clean716 replay of the matched causal-residual soft-gate pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
        REPO_ROOT / "code", REPO_ROOT / "code" / "training",
        Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.causal_residual_soft_gate import CausalResidualSoftGate  # noqa: E402
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_causal_phase_direction_real4k as phase_train  # noqa: E402
import train_b210_causal_residual_soft_gate_lifecycle as lifecycle  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real4k  # noqa: E402


SCHEMA = "b210_causal_residual_soft_gate_clean716_replay_v1"
ACKNOWLEDGEMENT = "EVALUATE_B210_CAUSAL_RESIDUAL_SOFT_GATE_AFTER_REAL4K_SEALS_V1"
MODEL_NAMES = (
    "b210", "spatial_control_raw", "spatial_control_ema",
    "causal_raw", "causal_ema",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--spatial-control-endpoint", type=Path, required=True)
    parser.add_argument("--spatial-control-endpoint-sha256", required=True)
    parser.add_argument("--causal-endpoint", type=Path, required=True)
    parser.add_argument("--causal-endpoint-sha256", required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--clean716-manifest", type=Path, required=True)
    parser.add_argument("--clean716-manifest-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--acknowledge-target-access", default="")
    return parser.parse_args(argv)


def validate_endpoint(
        path: Path, expected_sha: str, expected_arm: str,
        expected_base_sha: str,
        ) -> tuple[dict[str, object], dict[str, Mapping[str, torch.Tensor]]]:
    actual = lifecycle.require_sha(
        path, expected_sha, f"Real {expected_arm} endpoint")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, Mapping)
            or payload.get("schema") != lifecycle.ENDPOINT_SCHEMA
            or payload.get("experiment_schema") != lifecycle.SCHEMA
            or payload.get("stage") != "real"
            or payload.get("arm") != expected_arm
            or int(payload.get("update", -1)) != lifecycle.REAL_UPDATES):
        raise RuntimeError(f"{expected_arm} is not a complete Real4k endpoint")
    source = payload.get("base_checkpoint")
    if not isinstance(source, Mapping) or source.get("sha256") != expected_base_sha:
        raise RuntimeError(f"{expected_arm} references another B210 anchor")
    config = lifecycle.parse_fusion_config(payload.get("fusion_config", {}))
    if config.mixer_mode != expected_arm:
        raise RuntimeError(f"{expected_arm} fusion mode changed")
    if int(payload.get("fusion_parameters", -1)) != (
            lifecycle.EXPECTED_FUSION_PARAMETERS):
        raise RuntimeError(f"{expected_arm} fusion parameter count changed")
    lineage = payload.get("initial_lineage")
    if (not isinstance(lineage, Mapping)
            or lineage.get("initial_fusion_state_sha256") is None
            or lineage.get("sample_schedule_sha256") is None):
        raise RuntimeError(f"{expected_arm} lacks its Sim4k lineage")
    mechanics = payload.get("mechanics")
    if not isinstance(mechanics, Mapping) or not all(
            value is True for value in mechanics.values()):
        raise RuntimeError(f"{expected_arm} mechanics evidence is incomplete")

    states = {}
    for kind in ("raw", "ema"):
        key = f"{kind}_fusion_state"
        sha_key = f"{kind}_fusion_state_sha256"
        state = payload.get(key)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"{expected_arm} lacks {kind} fusion state")
        if ema_lib.tensor_state_sha256(state) != payload.get(sha_key):
            raise RuntimeError(f"{expected_arm} {kind} tensor digest changed")
        probe = CausalResidualSoftGate(config).float()
        probe.load_state_dict(state, strict=True)
        if sum(parameter.numel() for parameter in probe.parameters()) != (
                lifecycle.EXPECTED_FUSION_PARAMETERS):
            raise RuntimeError(f"{expected_arm} {kind} strict-load size changed")
        if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
               for value in state.values()):
            raise FloatingPointError(f"{expected_arm} {kind} state is non-finite")
        states[kind] = state
    ema = payload.get("ema")
    if (not isinstance(ema, Mapping)
            or int(ema.get("updates", -1)) != lifecycle.REAL_UPDATES
            or ema.get("raw_model_state_sha256")
            != payload.get("raw_fusion_state_sha256")
            or ema.get("materialized_model_state_sha256")
            != payload.get("ema_fusion_state_sha256")):
        raise RuntimeError(f"{expected_arm} EMA seal changed")
    return dict(payload), states | {"report": {
        "path": str(path.resolve()),
        "file_sha256": actual,
        "arm": expected_arm,
        "raw_state_sha256": payload["raw_fusion_state_sha256"],
        "ema_state_sha256": payload["ema_fusion_state_sha256"],
        "sample_schedule_sha256": payload["sample_schedule_sha256"],
        "sim_lineage": dict(lineage),
    }}


def validate_pair(
        control: Mapping[str, object], causal: Mapping[str, object]) -> dict[str, object]:
    if control["sample_schedule_sha256"] != causal["sample_schedule_sha256"]:
        raise RuntimeError("Real4k arms used different sample schedules")
    control_config = lifecycle.parse_fusion_config(control["fusion_config"])
    causal_config = lifecycle.parse_fusion_config(causal["fusion_config"])
    left, right = vars(control_config).copy(), vars(causal_config).copy()
    left.pop("mixer_mode")
    right.pop("mixer_mode")
    if left != right:
        raise RuntimeError("matched pair differs outside mixer mode")
    for key in ("sha256", "model_state_sha256", "normalizer_sha256"):
        if control["base_checkpoint"].get(key) != causal[
                "base_checkpoint"].get(key):
            raise RuntimeError(f"matched pair base differs: {key}")
    control_lineage = control["initial_lineage"]
    causal_lineage = causal["initial_lineage"]
    if control_lineage["sample_schedule_sha256"] != causal_lineage[
            "sample_schedule_sha256"]:
        raise RuntimeError("Sim4k arms used different sample schedules")
    if control_lineage["initial_fusion_state_sha256"] != causal_lineage[
            "initial_fusion_state_sha256"]:
        raise RuntimeError("Sim4k arms did not share update-zero tensors")
    for name in ("mean_input", "mean_target", "std_input", "std_target"):
        if not torch.equal(
                torch.as_tensor(control["normalizer"][name]),
                torch.as_tensor(causal["normalizer"][name])):
            raise RuntimeError(f"matched pair normalizer differs: {name}")
    return {
        "sim_sample_schedule_sha256": control_lineage[
            "sample_schedule_sha256"],
        "sim_common_initialization_sha256": control_lineage[
            "initial_fusion_state_sha256"],
        "real_sample_schedule_sha256": control["sample_schedule_sha256"],
        "matched_outside_mixer_mode": True,
    }


def load_fusion(
        endpoint: Mapping[str, object], state: Mapping[str, torch.Tensor],
        device: torch.device) -> CausalResidualSoftGate:
    config = lifecycle.parse_fusion_config(endpoint["fusion_config"])
    model = CausalResidualSoftGate(config).float()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def decorate(metrics: Mapping[str, float]) -> dict[str, float]:
    result = {key: float(value) for key, value in metrics.items()}
    point = (
        .46 * result["rel_l2_score"]
        + .10 * result["tke_rel_l2_score"]
        + .09 * result["mvpe_rel_l2_score"])
    result["official_point_contribution_46_10_9"] = point
    result["official_normalized_point_46_10_9"] = point / .65
    result["non_time_contribution_46_10_9_25"] = (
        point + .25 * result["native_sps_score"])
    return result


def delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {key: float(left[key]) - float(right[key])
            for key in left.keys() & right.keys()
            if isinstance(left[key], (int, float))
            and isinstance(right[key], (int, float))}


def comparisons(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    return {
        "causal_minus_spatial_raw": delta(
            metrics["causal_raw"], metrics["spatial_control_raw"]),
        "causal_minus_spatial_ema": delta(
            metrics["causal_ema"], metrics["spatial_control_ema"]),
        "each_minus_b210": {
            name: delta(metrics[name], metrics["b210"])
            for name in MODEL_NAMES if name != "b210"
        },
    }


def finalize(states: Mapping[str, dict]) -> dict[str, dict[str, float]]:
    return {name: decorate(smu1.finalize_metric_state(state))
            for name, state in states.items()}


@torch.inference_mode()
def evaluate(
        base, fusions: Mapping[str, CausalResidualSoftGate],
        dataset, aoas: Sequence[int], labels: Sequence[str], normalizer,
        device: torch.device) -> dict[str, object]:
    pooled = {name: smu1.new_metric_state() for name in MODEL_NAMES}
    by_aoa = {
        str(aoa): {name: smu1.new_metric_state() for name in MODEL_NAMES}
        for aoa in sorted(set(aoas))}
    prediction_hashers = {name: hashlib.sha256() for name in MODEL_NAMES}
    target_hasher = hashlib.sha256()
    invariants = {
        name: {
            "point_delta_temporal_mean_max_abs": 0.0,
            "uq_center_delta_temporal_mean_max_abs": 0.0,
            "interval_width_delta_max_abs": 0.0,
            "pressure_bitwise": True,
        }
        for name in MODEL_NAMES if name != "b210"
    }
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for start in range(0, len(dataset), 8):
        stop = min(start + 8, len(dataset))
        histories, targets = zip(*(dataset[index] for index in range(start, stop)))
        history = torch.stack(histories).to(device)
        target = torch.stack(targets).to(device)
        history_normalized, _unused = normalizer.preprocess(history, target)
        base_prediction_norm, base_lower_norm, base_upper_norm = (
            base.forward_distribution(history_normalized))
        base_prediction = normalizer.postprocess_prediction(
            base_prediction_norm.float())
        base_prediction[..., 2] = 0.0
        outputs = {
            "b210": (base_prediction,
                     base_lower_norm.float() * scale,
                     base_upper_norm.float() * scale),
        }
        for name, fusion in fusions.items():
            values = fusion.forward_components(
                history_normalized, base_prediction_norm,
                base_lower_norm, base_upper_norm)
            prediction = normalizer.postprocess_prediction(
                values.prediction.float())
            prediction[..., 2] = 0.0
            lower = values.lower_extent.float() * scale
            upper = values.upper_extent.float() * scale
            outputs[name] = (prediction, lower, upper)
            audit = invariants[name]
            audit["point_delta_temporal_mean_max_abs"] = max(
                audit["point_delta_temporal_mean_max_abs"],
                float(values.point_delta.float().mean(dim=1).abs().max()))
            audit["uq_center_delta_temporal_mean_max_abs"] = max(
                audit["uq_center_delta_temporal_mean_max_abs"],
                float(values.uq_center_delta.float().mean(dim=1).abs().max()))
            audit["interval_width_delta_max_abs"] = max(
                audit["interval_width_delta_max_abs"],
                float(((values.lower_extent + values.upper_extent)
                       - (base_lower_norm + base_upper_norm)).float().abs().max()))
            audit["pressure_bitwise"] = bool(
                audit["pressure_bitwise"] and torch.equal(
                    values.prediction[..., 2:], base_prediction_norm[..., 2:]))

        target_hasher.update(target.detach().cpu().contiguous().numpy().tobytes())
        for name, (prediction, lower, upper) in outputs.items():
            prediction_hashers[name].update(
                prediction.detach().cpu().contiguous().numpy().tobytes())
            smu1.update_metric_state(pooled[name], prediction, target, lower, upper)
            for local, global_index in enumerate(range(start, stop)):
                selection = slice(local, local + 1)
                smu1.update_metric_state(
                    by_aoa[str(aoas[global_index])][name],
                    prediction[selection], target[selection],
                    lower[selection], upper[selection])

    pooled_metrics = finalize(pooled)
    aoa_metrics = {}
    for aoa, states in by_aoa.items():
        values = finalize(states)
        aoa_metrics[aoa] = {
            "models": values,
            "comparisons": comparisons(values),
        }
    invariant_pass = {
        name: {
            **values,
            "pass": (
                values["point_delta_temporal_mean_max_abs"] <= 1.0e-6
                and values["uq_center_delta_temporal_mean_max_abs"] <= 1.0e-6
                and values["interval_width_delta_max_abs"] <= 1.0e-6
                and values["pressure_bitwise"]),
        }
        for name, values in invariants.items()
    }
    if not all(row["pass"] for row in invariant_pass.values()):
        raise RuntimeError(f"soft-gate structural invariant failed: {invariant_pass}")
    return {
        "windows": len(dataset),
        "pooled": pooled_metrics,
        "comparisons": comparisons(pooled_metrics),
        "by_aoa": aoa_metrics,
        "invariants": invariant_pass,
        "prediction_sha256": {
            name: hasher.hexdigest() for name, hasher in prediction_hashers.items()},
        "target_sha256": target_hasher.hexdigest(),
        "trajectory_labels_sha256": real4k.canonical_sha(list(labels)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.acknowledge_target_access != ACKNOWLEDGEMENT:
        raise PermissionError("clean716 replay requires the exact acknowledgement")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    lifecycle.require_sha(
        args.clean716_manifest, args.clean716_manifest_sha256,
        "clean716 manifest")
    control, control_states = validate_endpoint(
        args.spatial_control_endpoint,
        args.spatial_control_endpoint_sha256,
        lifecycle.ARMS[0], args.base_checkpoint_sha256)
    causal, causal_states = validate_endpoint(
        args.causal_endpoint, args.causal_endpoint_sha256,
        lifecycle.ARMS[1], args.base_checkpoint_sha256)
    fairness = validate_pair(control, causal)
    # All four candidate tensor states are strict-loaded above before either
    # the clean716 manifest or any target H5 field is read.
    payload, base_config, base_state, base_report = phase_train.load_base_payload(
        args.base_checkpoint, args.base_checkpoint_sha256)
    if base_report["model_state_sha256"] != control[
            "base_checkpoint"]["model_state_sha256"]:
        raise RuntimeError("Real endpoints do not bind the supplied B210 state")
    device = torch.device(args.device)
    normalizer = smu1.FoldNormalizer(tuple(
        control["normalizer"][name] for name in (
            "mean_input", "mean_target", "std_input", "std_target")),
        device=device)
    base = phase_train.build_base(
        base_config, base_state, normalizer, device)
    fusions = {
        "spatial_control_raw": load_fusion(control, control_states["raw"], device),
        "spatial_control_ema": load_fusion(control, control_states["ema"], device),
        "causal_raw": load_fusion(causal, causal_states["raw"], device),
        "causal_ema": load_fusion(causal, causal_states["ema"], device),
    }

    fit_rows, fit_report = real4k.load_clean_manifest(args.clean_manifest)
    tail_rows = real4k.load_clean716_rows(args.clean716_manifest, fit_rows)
    clean716_report = {
        "path": str(args.clean716_manifest.resolve()),
        "file_sha256": args.clean716_manifest_sha256,
        "semantic_sha256": real4k.canonical_sha(json.loads(
            args.clean716_manifest.read_text(encoding="utf-8"))),
        "windows": sum(len(row["train_starts"]) for row in tail_rows),
    }
    access = real4k.PrefixAccessAudit()
    access.record_endpoint_seal({
        "path": str(args.causal_endpoint.resolve()),
        "file_sha256": causal_states["report"]["file_sha256"],
    })
    trajectories, windows = real4k.load_real_rows(
        args.real_dir, tail_rows, phase="clean716", audit=access)
    dataset = real4k.RealWindowDataset(trajectories, windows)
    aoas = [int(tail_rows[window.trajectory]["aoa"]) for window in windows]
    labels = [str(tail_rows[window.trajectory]["file"]) for window in windows]
    evaluation = evaluate(
        base, fusions, dataset, aoas, labels, normalizer, device)
    # Detect mutation during target replay for every immutable input, not only
    # the candidate file guarded by PrefixAccessAudit.
    lifecycle.require_sha(
        args.spatial_control_endpoint,
        control_states["report"]["file_sha256"], "spatial endpoint post-eval")
    lifecycle.require_sha(
        args.causal_endpoint,
        causal_states["report"]["file_sha256"], "causal endpoint post-eval")
    result = {
        "schema": SCHEMA,
        "evidence_class": "post_hoc_same_clean716_diagnostic",
        "endpoint_barrier": {
            "validated_before_clean716_manifest_or_target_open": True,
            "spatial_control": control_states["report"],
            "causal": causal_states["report"],
            "fairness": fairness,
            "states": [
                "spatial_control_raw", "spatial_control_ema",
                "causal_raw", "causal_ema"],
        },
        "base": base_report,
        "fit_manifest": fit_report,
        "clean716_manifest": clean716_report,
        "target_access_audit": access.events,
        "evaluation": evaluation,
        "checkpoint_selection": False,
        "submission_authorized": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.atomic_json(args.output_json, result)
    del payload, base_state
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_causal_residual_soft_gate_clean716_complete",
        "output_json": str(args.output_json.resolve()),
        "pooled": result["evaluation"]["pooled"],
        "comparisons": result["evaluation"]["comparisons"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
