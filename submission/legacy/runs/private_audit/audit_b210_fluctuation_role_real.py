#!/usr/bin/env python3
"""Ablate a trained fluctuation-role tower once on the sealed clean716 tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_scratch_real_dense_aug as scratch  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig  # noqa: E402


class ReadOnlyHeldAudit:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "held":
            raise RuntimeError("role audit may only open sealed held rows")

    def record_open(self, row: dict, phase: str) -> None:
        if phase != "held":
            raise RuntimeError("role audit phase changed")
        self.opened.append(str(row["file"]))


def new_component_state() -> dict[str, float | int]:
    return {
        "samples": 0,
        "elements": 0,
        "mean_error_square_sum": 0.0,
        "mean_target_square_sum": 0.0,
        "fluctuation_error_square_sum": 0.0,
        "fluctuation_target_square_sum": 0.0,
        "fluctuation_inner_sum": 0.0,
        "fluctuation_prediction_square_sum": 0.0,
    }


def update_component_state(
        state: dict[str, float | int], prediction: torch.Tensor,
        target: torch.Tensor) -> None:
    prediction = prediction.float()[..., :2]
    target = target.float()[..., :2]
    prediction_mean = prediction.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    prediction_fluctuation = prediction - prediction_mean
    target_fluctuation = target - target_mean
    state["samples"] += int(prediction.shape[0])
    state["elements"] += int(target_fluctuation.numel())
    state["mean_error_square_sum"] += float(
        (prediction_mean - target_mean).square().sum())
    state["mean_target_square_sum"] += float(target_mean.square().sum())
    state["fluctuation_error_square_sum"] += float(
        (prediction_fluctuation - target_fluctuation).square().sum())
    state["fluctuation_target_square_sum"] += float(
        target_fluctuation.square().sum())
    state["fluctuation_inner_sum"] += float(
        (prediction_fluctuation * target_fluctuation).sum())
    state["fluctuation_prediction_square_sum"] += float(
        prediction_fluctuation.square().sum())


def finalize_component_state(state: dict[str, float | int]) -> dict[str, float]:
    eps = 1.0e-20
    mean_error = float(state["mean_error_square_sum"])
    mean_target = float(state["mean_target_square_sum"])
    fluctuation_error = float(state["fluctuation_error_square_sum"])
    fluctuation_target = float(state["fluctuation_target_square_sum"])
    fluctuation_prediction = float(state["fluctuation_prediction_square_sum"])
    return {
        "temporal_mean_relative_l2": (mean_error / max(mean_target, eps)) ** 0.5,
        "fluctuation_relative_l2": (
            fluctuation_error / max(fluctuation_target, eps)) ** 0.5,
        "fluctuation_cosine": float(state["fluctuation_inner_sum"]) / max(
            (fluctuation_target * fluctuation_prediction) ** 0.5, eps),
        "prediction_to_target_fluctuation_rms_ratio": (
            fluctuation_prediction / max(fluctuation_target, eps)) ** 0.5,
    }


def delta(candidate: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    keys = (
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "point_score", "native_sps_score", "native_sps_coverage",
        "native_sps_mean_width",
    )
    values = {key: float(candidate[key]) - float(control[key]) for key in keys}
    values["official_point_contribution"] = (
        0.46 * values["rel_l2_score"]
        + 0.10 * values["tke_rel_l2_score"]
        + 0.09 * values["mvpe_rel_l2_score"])
    values["official_non_time_contribution"] = (
        values["official_point_contribution"]
        + 0.25 * values["native_sps_score"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_config = dict(payload["model_config"])
    if raw_config.pop("architecture") != "ensemble_unet":
        raise ValueError("checkpoint architecture changed")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**raw_config)).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model.fluctuation_role_adapters is None:
        raise ValueError("checkpoint has no fluctuation-role tower")
    normalizer_values = payload["normalizer"]
    statistics = tuple(torch.tensor(
        normalizer_values[key], dtype=torch.float32)
        for key in ("mean_input", "mean_target", "std_input", "std_target"))
    normalizer = scratch.smu1.FoldNormalizer(statistics, device)
    scratch.ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    model.to(device).eval()

    rows, _report = scratch.dr1.load_manifest(
        args.clean_manifest, scratch.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = scratch.load_evaluation_rows(
        args.evaluation_manifest, rows)
    audit = ReadOnlyHeldAudit()
    trajectories, metadata, windows = scratch.dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", audit)
    dataset = scratch.ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    if len(dataset) != scratch.TAIL_WINDOWS:
        raise RuntimeError("clean716 window contract changed")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    aoas = [int(metadata[int(window.trajectory)]["aoa"]) for window in windows]
    metric_states = {
        arm: scratch.smu1.new_metric_state() for arm in ("on", "off")}
    component_states = {
        arm: new_component_state() for arm in ("on", "off")}
    metric_states_by_aoa = {
        arm: {aoa: scratch.smu1.new_metric_state() for aoa in scratch.dr1.AOA_VALUES}
        for arm in ("on", "off")}
    component_states_by_aoa = {
        arm: {aoa: new_component_state() for aoa in scratch.dr1.AOA_VALUES}
        for arm in ("on", "off")}
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    temporal_mean_delta_abs_max = 0.0
    prediction_delta_square_sum = 0.0
    prediction_delta_count = 0
    interval_width_delta_square_sum = 0.0
    interval_width_delta_count = 0
    started = time.perf_counter()
    offset = 0
    role_tower = model.fluctuation_role_adapters
    with torch.no_grad():
        for history, target in loader:
            history = history.to(device)
            target = target.to(device)
            history_norm, _target_norm = normalizer.preprocess(history, target)
            on_prediction_norm, on_lower_norm, on_upper_norm = (
                model.forward_distribution(history_norm))
            model.fluctuation_role_adapters = None
            off_prediction_norm, off_lower_norm, off_upper_norm = (
                model.forward_distribution(history_norm))
            model.fluctuation_role_adapters = role_tower
            predictions = {
                "on": normalizer.postprocess_prediction(
                    on_prediction_norm.float()),
                "off": normalizer.postprocess_prediction(
                    off_prediction_norm.float()),
            }
            intervals = {
                "on": (on_lower_norm.float() * scale,
                       on_upper_norm.float() * scale),
                "off": (off_lower_norm.float() * scale,
                        off_upper_norm.float() * scale),
            }
            for arm in ("on", "off"):
                lower, upper = intervals[arm]
                scratch.smu1.update_metric_state(
                    metric_states[arm], predictions[arm], target.float(),
                    lower, upper)
                update_component_state(
                    component_states[arm], predictions[arm], target)
            count = int(history.shape[0])
            batch_aoas = aoas[offset:offset + count]
            for aoa in scratch.dr1.AOA_VALUES:
                indices = [index for index, value in enumerate(batch_aoas)
                           if value == aoa]
                if not indices:
                    continue
                selected = torch.as_tensor(indices, device=device)
                for arm in ("on", "off"):
                    lower, upper = intervals[arm]
                    scratch.smu1.update_metric_state(
                        metric_states_by_aoa[arm][aoa],
                        predictions[arm].index_select(0, selected),
                        target.float().index_select(0, selected),
                        lower.index_select(0, selected),
                        upper.index_select(0, selected))
                    update_component_state(
                        component_states_by_aoa[arm][aoa],
                        predictions[arm].index_select(0, selected),
                        target.index_select(0, selected))
            offset += count
            temporal_mean_delta_abs_max = max(
                temporal_mean_delta_abs_max,
                float((predictions["on"].mean(dim=1)
                       - predictions["off"].mean(dim=1)).abs().max()))
            prediction_delta = predictions["on"] - predictions["off"]
            prediction_delta_square_sum += float(
                prediction_delta.square().sum())
            prediction_delta_count += prediction_delta.numel()
            on_width = intervals["on"][1] - intervals["on"][0]
            off_width = intervals["off"][1] - intervals["off"][0]
            interval_delta = on_width - off_width
            interval_width_delta_square_sum += float(interval_delta.square().sum())
            interval_width_delta_count += interval_delta.numel()
    if offset != len(dataset) or model.fluctuation_role_adapters is not role_tower:
        raise RuntimeError("role audit did not restore the deployed model")

    metrics = {
        arm: scratch.smu1.finalize_metric_state(metric_states[arm])
        for arm in ("on", "off")}
    components = {
        arm: finalize_component_state(component_states[arm])
        for arm in ("on", "off")}
    by_aoa = {}
    for aoa in scratch.dr1.AOA_VALUES:
        aoa_metrics = {
            arm: scratch.smu1.finalize_metric_state(
                metric_states_by_aoa[arm][aoa])
            for arm in ("on", "off")}
        aoa_components = {
            arm: finalize_component_state(
                component_states_by_aoa[arm][aoa])
            for arm in ("on", "off")}
        by_aoa[str(aoa)] = {
            "metrics": aoa_metrics,
            "components": aoa_components,
            "on_minus_off": delta(aoa_metrics["on"], aoa_metrics["off"]),
        }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": scratch.dr1.sha256(args.checkpoint),
        "samples": len(dataset),
        "metrics": metrics,
        "components": components,
        "on_minus_off": delta(metrics["on"], metrics["off"]),
        "by_aoa": by_aoa,
        "positive_official_non_time_aoas": sum(
            row["on_minus_off"]["official_non_time_contribution"] > 0.0
            for row in by_aoa.values()),
        "temporal_mean_prediction_delta_abs_max": temporal_mean_delta_abs_max,
        "prediction_delta_rms": (
            prediction_delta_square_sum / prediction_delta_count) ** 0.5,
        "interval_width_delta_rms": (
            interval_width_delta_square_sum / interval_width_delta_count) ** 0.5,
        "interpretation_boundary": (
            "off is an in-checkpoint functional ablation of a co-adapted "
            "endpoint, not the separately trained no-role architecture; the "
            "historical Standard24k Real4k endpoint is the causal control"),
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "b210_fluctuation_role_real_ablation",
        "on_minus_off": result["on_minus_off"],
        "components": components,
        "positive_aoas": result["positive_official_non_time_aoas"],
        "temporal_mean_delta_abs_max": temporal_mean_delta_abs_max,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
