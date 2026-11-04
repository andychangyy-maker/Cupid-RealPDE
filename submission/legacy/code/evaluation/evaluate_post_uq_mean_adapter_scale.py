#!/usr/bin/env python3
"""Test one closed-form scale for a frozen post-UQ mean adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_full_backbone_point_distillation as full  # noqa: E402
import train_post_uq_mean_residual_adapter as route  # noqa: E402


POINT_DELTA_GATE = 0.008
MAX_CORRECTION = 0.75
TIME_BUDGET_SECONDS = 300.0


class ScaledAdapter(nn.Module):
    def __init__(self, adapter: nn.Module, alpha: float):
        super().__init__()
        self.adapter = adapter
        self.alpha = float(alpha)
        self.max_abs_correction = 0.0

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        correction = self.alpha * self.adapter(feature)
        self.max_abs_correction = max(
            self.max_abs_correction, float(correction.abs().max()))
        return correction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def data_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        real_dir=args.real_dir,
        real_stats=args.real_stats,
        output_dir=args.output_dir,
        teacher_archive=args.teacher_archive,
        teacher_cache=args.teacher_cache,
        max_files=0,
        window_stride=20,
        val_fraction=0.2,
        batch_size=8,
        eval_batch_size=16,
        num_workers=0,
        seed=0,
    )


def load_adapter(
        path: Path, expected_base_sha256: str, device: torch.device,
        ) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("method") != "post_uq_temporal_mean_residual_adapter":
        raise ValueError("checkpoint is not a post-UQ mean adapter")
    if payload.get("base_checkpoint_sha256") != expected_base_sha256:
        raise ValueError("adapter was trained against a different base")
    if int(payload.get("iteration", -1)) != 250:
        raise ValueError("diagnostic requires the frozen step-250 adapter")
    if payload.get("status") != "STOP_STEP250_NO_SIGNAL":
        raise ValueError("adapter does not carry the expected STOP status")
    config = payload["adapter_config"]
    adapter = route.PostUQMeanResidualAdapter(
        input_channels=int(config["input_channels"]),
        hidden_channels=int(config["hidden_channels"]),
        groups=int(config["groups"])).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    adapter.requires_grad_(False)
    adapter.eval()
    return adapter, payload


@torch.inference_mode()
def fit_closed_form_scale(
        model: nn.Module, adapter: nn.Module, loader,
        teacher_cache: torch.Tensor, normalizer,
        device: torch.device, started: float,
        ) -> dict[str, float]:
    dot_c_r = 0.0
    dot_c_c = 0.0
    dot_r_r = 0.0
    zero_mean_sum = 0.0
    zero_profile_sum = 0.0
    trained_mean_sum = 0.0
    trained_profile_sum = 0.0
    max_abs_correction = 0.0
    windows = 0

    model.eval()
    adapter.eval()
    for indices, history, truth in loader:
        if time.time() - started > TIME_BUDGET_SECONDS:
            raise TimeoutError("closed-form diagnostic exceeded 300 seconds")
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        teacher = teacher_cache[indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        history_norm, _truth_norm = normalizer.preprocess(history, truth)
        base_prediction, features = model.forward_with_features(history_norm)
        correction = adapter(features[-1]).float()
        target_delta = (
            teacher.mean(dim=1, keepdim=True)
            - base_prediction[..., :2].float().mean(dim=1, keepdim=True))
        zeros = torch.zeros_like(correction)

        dot_c_r += float((correction * target_delta).sum(dtype=torch.float64))
        dot_c_c += float(correction.square().sum(dtype=torch.float64))
        dot_r_r += float(target_delta.square().sum(dtype=torch.float64))
        max_abs_correction = max(
            max_abs_correction, float(correction.abs().max()))

        batch = int(history.shape[0])
        windows += batch
        zero_mean_sum += batch * float(F.smooth_l1_loss(
            zeros / 0.025, target_delta / 0.025))
        trained_mean_sum += batch * float(F.smooth_l1_loss(
            correction / 0.025, target_delta / 0.025))
        zero_profile_sum += batch * float(route.profile_distill_loss(
            zeros, target_delta, 0.025))
        trained_profile_sum += batch * float(route.profile_distill_loss(
            correction, target_delta, 0.025))

    if windows != len(loader.dataset):
        raise RuntimeError("training loader did not cover each window once")
    if not dot_c_c > 0.0 or not dot_r_r > 0.0:
        raise RuntimeError("correction and residual norms must be positive")
    alpha = max(0.0, dot_c_r / dot_c_c)
    cosine = dot_c_r / math.sqrt(dot_c_c * dot_r_r)
    rms_ratio = math.sqrt(dot_c_c / dot_r_r)
    return {
        "alpha": alpha,
        "cosine": cosine,
        "correction_to_target_rms_ratio": rms_ratio,
        "unscaled_max_abs_correction": max_abs_correction,
        "train_windows": windows,
        "zero_mean_loss": zero_mean_sum / windows,
        "zero_profile_loss": zero_profile_sum / windows,
        "zero_total_loss": 2.0 * (
            zero_mean_sum + zero_profile_sum) / windows,
        "trained_mean_loss": trained_mean_sum / windows,
        "trained_profile_loss": trained_profile_sum / windows,
        "trained_total_loss": 2.0 * (
            trained_mean_sum + trained_profile_sum) / windows,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("closed-form diagnostic requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_sha256 = full.sha256(args.base_checkpoint)
    adapter_sha256 = full.sha256(args.adapter_checkpoint)
    if base_sha256 != args.expected_base_sha256.lower():
        raise ValueError("base checkpoint SHA-256 mismatch")
    if adapter_sha256 != args.expected_adapter_sha256.lower():
        raise ValueError("adapter checkpoint SHA-256 mismatch")

    prepared = data_args(args)
    (train_loader, cache_loader, val_loader, normalizer,
     train_keys, val_aoa, trajectories, split_manifest_sha256,
     real_stats_sha256, expected_teacher_shape) = full.prepare_data(
         prepared, device)
    if len(train_keys) != 2576 or len(val_aoa) != 725 or trajectories != 82:
        raise ValueError("fixed 2576/725 data contract mismatch")
    if split_manifest_sha256 != full.FORMAL_SPLIT_MANIFEST_SHA256:
        raise ValueError("fixed split SHA-256 mismatch")
    if real_stats_sha256 != full.FORMAL_REAL_STATS_SHA256:
        raise ValueError("real stats SHA-256 mismatch")
    if full.sha256(args.teacher_archive) != full.FORMAL_TEACHER_SHA256:
        raise ValueError("teacher archive SHA-256 mismatch")

    model, _base_payload = full.load_student(
        args.base_checkpoint, normalizer, device)
    model.requires_grad_(False)
    model.eval()
    adapter, adapter_payload = load_adapter(
        args.adapter_checkpoint, base_sha256, device)
    teacher_cache = full.build_or_load_teacher_cache(
        prepared, cache_loader, normalizer, train_keys,
        split_manifest_sha256, real_stats_sha256,
        expected_teacher_shape)

    train_fit = fit_closed_form_scale(
        model, adapter, train_loader, teacher_cache,
        normalizer, device, started)
    scaled_adapter = ScaledAdapter(adapter, train_fit["alpha"]).to(device)
    evaluation = route.evaluate(
        model, scaled_adapter, val_loader, normalizer,
        device, val_aoa, with_distribution=True)
    max_abs_correction = max(
        train_fit["alpha"] * train_fit["unscaled_max_abs_correction"],
        scaled_adapter.max_abs_correction)
    positive_aoa = sum(
        item["delta"]["point_score"] >= 0.0
        for item in evaluation["by_aoa"].values())
    distribution = evaluation["distribution"]
    gate = {
        "point_delta_pass": (
            evaluation["delta"]["point_score"] >= POINT_DELTA_GATE),
        "rel_positive": evaluation["delta"]["rel_l2_score"] > 0.0,
        "mvpe_positive": evaluation["delta"]["mvpe_rel_l2_score"] > 0.0,
        "positive_aoa_slices": positive_aoa,
        "aoa_pass": len(evaluation["by_aoa"]) == 5 and positive_aoa >= 4,
        "tke_invariant": abs(
            evaluation["delta"]["tke_rel_l2_score"]) <= 1e-4,
        "width_invariant": (
            distribution["max_abs_width_drift"] <= 2e-7
            and distribution["extents_bitwise_equal"]
            and distribution["max_abs_correction_temporal_span"] == 0.0),
        "empirical_observed_bound_pass": (
            max_abs_correction <= MAX_CORRECTION),
        "global_correction_bound_pass": (
            train_fit["alpha"] * route.MAX_POINT_DELTA
            <= MAX_CORRECTION),
    }
    diagnostic_gate_pass = all(
        value for key, value in gate.items()
        if key != "positive_aoa_slices")
    elapsed = time.time() - started
    if elapsed > TIME_BUDGET_SECONDS:
        status = "STOP_TIME_BUDGET"
        diagnostic_gate_pass = False
    elif diagnostic_gate_pass:
        status = "AMPLITUDE_DIAGNOSTIC_PASS"
    else:
        status = "STOP_SINGLE_FEATURE_MEAN_ADAPTER"

    result = {
        "method": "closed_form_post_uq_mean_adapter_scale",
        "status": status,
        "diagnostic_gate_pass": diagnostic_gate_pass,
        "promotion_gate_pass": False,
        "formula": (
            "pixel_l2_alpha=max(0,<correction,residual>/"
            "<correction,correction>); not the closed-form optimum of the "
            "training SmoothL1+profile objective"),
        "time_budget_seconds": TIME_BUDGET_SECONDS,
        "wall_seconds": elapsed,
        "base_checkpoint_sha256": base_sha256,
        "adapter_checkpoint_sha256": adapter_sha256,
        "adapter_source_status": adapter_payload["status"],
        "teacher_archive_sha256": full.sha256(args.teacher_archive),
        "teacher_cache_sha256": full.sha256(args.teacher_cache),
        "split_manifest_sha256": split_manifest_sha256,
        "real_stats_sha256": real_stats_sha256,
        "script_sha256": full.sha256(Path(__file__)),
        "train_fit": train_fit,
        "scaled_max_abs_correction": max_abs_correction,
        "evaluation": evaluation,
        "gate": gate,
        "uq_state_valid": False,
        "sps_actionable": False,
        "evidence_scope": (
            "fixed725 amplitude diagnostic; same-model AoA slices, "
            "not leave-one-AoA crossfit"),
    }
    output = args.output_dir / "amplitude_diagnostic.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "status": status,
        "diagnostic_gate_pass": diagnostic_gate_pass,
        "alpha": train_fit["alpha"],
        "point_delta": evaluation["delta"]["point_score"],
        "output": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
