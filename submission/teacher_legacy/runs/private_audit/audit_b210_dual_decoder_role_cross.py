#!/usr/bin/env python3
"""Cross temporal-mean, fluctuation, and interval roles on fixed Sim held128.

The historical CLI calls the candidate ``mean-fluctuation``.  ``--candidate``
and ``--candidate-mode`` make the same frozen diagnostic usable for later
dual-decoder routing arms without changing its heldout/scoring contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as sim  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig  # noqa: E402


def load_model(path: Path, expected_mode: str) -> EnsembleUNet2d:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = dict(payload["model_config"])
    if raw.pop("architecture") != "ensemble_unet":
        raise ValueError(f"not an ensemble U-Net: {path}")
    config = EnsembleUNet2dConfig(**raw)
    if config.dual_decoder_mode != expected_mode:
        raise ValueError(
            f"expected {expected_mode}, got {config.dual_decoder_mode}")
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def official_contributions(metrics: dict[str, float]) -> dict[str, float]:
    point = (
        0.46 * metrics["rel_l2_score"]
        + 0.10 * metrics["tke_rel_l2_score"]
        + 0.09 * metrics["mvpe_rel_l2_score"])
    return {
        **metrics,
        "official_point_contribution": point,
        "official_non_time_contribution": (
            point + 0.25 * metrics["native_sps_score"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-fused", type=Path, required=True)
    candidate_paths = parser.add_mutually_exclusive_group(required=True)
    candidate_paths.add_argument("--candidate", type=Path)
    candidate_paths.add_argument("--mean-fluctuation", type=Path)
    parser.add_argument(
        "--candidate-mode", default="mean_fluctuation",
        choices=(
            "mean_fluctuation", "fused_mean_fluctuation",
            "learned_fluctuation_gate",
            "separate_point_uq_fluctuation_gates"))
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=sim.ANCHOR_SIZE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    candidate_path = args.candidate or args.mean_fluctuation
    candidate_name = args.candidate_mode
    models = {
        "shared_fused": load_model(args.shared_fused, "shared_fused"),
        candidate_name: load_model(candidate_path, candidate_name),
    }

    with tempfile.TemporaryDirectory(prefix="b210_dual_role_cross_") as root:
        prepare_args = argparse.Namespace(
            sim_dir=args.sim_dir, max_files=0, window_stride=20,
            batch_size=16, num_workers=0, seed=sim.SEED,
            output_dir=Path(root))
        raw_loader, normalizer, windows, trajectories = (
            sim.rehearsal.ensemble_train.common.prepare_sim(
                prepare_args, device))
        if windows != 4800 or trajectories != 100:
            raise RuntimeError("released stride-20 Sim contract changed")
        for model in models.values():
            sim.rehearsal.ensemble_train.common.set_model_stats(
                model, normalizer)
            model.to(device).eval()
        if args.count <= 0 or args.count > windows:
            raise ValueError("held slice size is invalid")
        loader = DataLoader(
            Subset(raw_loader.dataset, list(range(args.count))),
            batch_size=16, shuffle=False, num_workers=0)
        physical = sim.sim_hgrad.SimEvaluationNormalizer(normalizer)
        scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
        point_names = (
            "shared_fused", candidate_name,
            "shared_mean_candidate_fluctuation",
            "candidate_mean_shared_fluctuation",
        )
        states = {
            f"{point_name}__{interval_name}_interval": smu1.new_metric_state()
            for point_name in point_names
            for interval_name in ("shared_fused", candidate_name)
        }
        mean_difference_max = 0.0
        fluctuation_difference_square = 0.0
        fluctuation_difference_count = 0
        point_predictions: dict[str, list[np.ndarray]] = {
            name: [] for name in point_names}
        targets: list[np.ndarray] = []
        with torch.inference_mode():
            for history, target in loader:
                history = history.to(device)
                target = target.to(device)
                history_norm, target_norm = normalizer.preprocess(
                    history, target)
                outputs = {
                    name: model.forward_distribution(history_norm)
                    for name, model in models.items()
                }
                predictions = {
                    name: physical.postprocess_prediction(value[0].float())
                    for name, value in outputs.items()
                }
                target_physical = physical.postprocess_target(
                    target_norm.float())
                shared = predictions["shared_fused"]
                candidate = predictions[candidate_name]
                shared_mean = shared.mean(dim=1, keepdim=True)
                candidate_mean = candidate.mean(dim=1, keepdim=True)
                point_values = {
                    **predictions,
                    "shared_mean_candidate_fluctuation": (
                        shared_mean + candidate - candidate_mean),
                    "candidate_mean_shared_fluctuation": (
                        candidate_mean + shared - shared_mean),
                }
                intervals = {
                    name: (value[1].float() * scale,
                           value[2].float() * scale)
                    for name, value in outputs.items()
                }
                for point_name, prediction in point_values.items():
                    point_predictions[point_name].append(
                        prediction.cpu().numpy())
                    for interval_name, (lower, upper) in intervals.items():
                        smu1.update_metric_state(
                            states[f"{point_name}__{interval_name}_interval"],
                            prediction[..., :2], target_physical[..., :2],
                            lower, upper)
                targets.append(target_physical.cpu().numpy())
                mean_difference_max = max(
                    mean_difference_max,
                    float((shared_mean - candidate_mean).abs().max()))
                fluctuation_delta = (
                    (shared - shared_mean)
                    - (candidate - candidate_mean))
                fluctuation_difference_square += float(
                    fluctuation_delta.square().sum())
                fluctuation_difference_count += fluctuation_delta.numel()

    target_array = np.concatenate(targets)
    point_metrics = {}
    for name, chunks in point_predictions.items():
        metrics = sim.rehearsal.ensemble_train.common.repro.regression_metrics(
            np.concatenate(chunks), target_array)
        for key in ("rel_l2", "tke_rel_l2", "mvpe_rel_l2"):
            metrics[f"{key}_score"] = 100.0 / (1.0 + 0.5 * metrics[key])
        point_metrics[name] = metrics
    rows = {}
    for name, state in states.items():
        point_name = name.split("__", 1)[0]
        interval_metrics = smu1.finalize_metric_state(state)
        metrics = {
            key: float(point_metrics[point_name][key])
            for key in (
                "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")
        }
        metrics.update({
            key: float(interval_metrics[key])
            for key in (
                "native_sps_score", "native_sps_coverage",
                "native_sps_mean_width")
        })
        rows[name] = official_contributions(metrics)
    baseline = rows["shared_fused__shared_fused_interval"]
    for values in rows.values():
        values["delta_point_vs_shared_fused"] = (
            values["official_point_contribution"]
            - baseline["official_point_contribution"])
        values["delta_non_time_vs_shared_fused"] = (
            values["official_non_time_contribution"]
            - baseline["official_non_time_contribution"])
    result = {
        "shared_fused_checkpoint": str(args.shared_fused.resolve()),
        "shared_fused_sha256": sim.sim_hgrad.sha256(args.shared_fused),
        "candidate_mode": candidate_name,
        "candidate_checkpoint": str(candidate_path.resolve()),
        "candidate_sha256": sim.sim_hgrad.sha256(candidate_path),
        "samples": args.count,
        "rows": rows,
        "diagnostics": {
            "temporal_mean_cross_model_max_abs": mean_difference_max,
            "fluctuation_cross_model_rms": (
                fluctuation_difference_square
                / fluctuation_difference_count) ** 0.5,
        },
        "interpretation_boundary": (
            "post-hoc fixed held128 role crossing; diagnostic only and not a "
            "deployable two-checkpoint submission"),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "b210_dual_decoder_role_cross_audit",
        "rows": rows,
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
