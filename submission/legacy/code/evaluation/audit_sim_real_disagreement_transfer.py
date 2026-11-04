#!/usr/bin/env python3
"""Audit whether a Sim ensemble's difficulty map survives Real adaptation.

The audit evaluates two checkpoints on the same fixed Real windows.  Each
checkpoint uses the normalizer serialized with that checkpoint, so the Sim
model sees physical Real inputs through its Sim coordinate system while the
Real model uses its fitted Real coordinate system.  Only target-nonzero u/v
elements enter the rank diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
from scipy import stats
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
import train_b210_scratch_real_dense_aug as paired  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_b210_np_functional_real_ft12_loao as ft12  # noqa: E402


class ReadOnlyTailAudit:
    """Allow only the already frozen held-tail rows to be opened."""

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "held" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid disagreement-transfer tail open")

    def record_open(self, _row: dict, phase: str) -> None:
        if phase != "held":
            raise RuntimeError("invalid disagreement-transfer tail record")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("rank arrays must have the same nontrivial size")
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def spatial_zscore(value: torch.Tensor) -> torch.Tensor:
    """Standardize B,T,H,W,C maps within each B,T,C spatial field."""
    if value.ndim != 5:
        raise ValueError("spatial maps must have shape B,T,H,W,C")
    mean = value.mean(dim=(2, 3), keepdim=True)
    scale = value.sub(mean).square().mean(
        dim=(2, 3), keepdim=True).add(1.0e-12).sqrt()
    return (value - mean) / scale


def top_fraction_diagnostic(
        signal: np.ndarray, error: np.ndarray,
        fraction: float = 0.10) -> dict[str, float]:
    if signal.size != error.size or signal.size < 2:
        raise ValueError("signal/error arrays must have matching support")
    if not 0.0 < fraction < 1.0:
        raise ValueError("top fraction must be inside (0,1)")
    count = max(1, int(np.ceil(signal.size * fraction)))
    top_signal = np.argpartition(signal, -count)[-count:]
    top_error = np.argpartition(error, -count)[-count:]
    overlap = np.intersect1d(
        top_signal, top_error, assume_unique=False).size
    denominator = max(float(error.mean()), 1.0e-12)
    return {
        "top_fraction": fraction,
        "top_error_recall": float(overlap / count),
        "error_amplification": float(error[top_signal].mean() / denominator),
    }


def cross_map_diagnostic(
        sim: Mapping[str, np.ndarray],
        real: Mapping[str, np.ndarray]) -> dict:
    required = {"spread_z", "adapter_z", "error"}
    if set(sim) != required or set(real) != required:
        raise ValueError("checkpoint diagnostic fields changed")
    sizes = {value.size for value in (*sim.values(), *real.values())}
    if len(sizes) != 1:
        raise ValueError("checkpoint diagnostics do not share support")
    real_error = np.log1p(real["error"] / 1.0e-4)
    sim_error = np.log1p(sim["error"] / 1.0e-4)
    result = {
        "sampled_elements": int(real_error.size),
        "sim_real_total_spread_z_spearman": safe_spearman(
            sim["spread_z"], real["spread_z"]),
        "sim_real_adapter_spread_z_spearman": safe_spearman(
            sim["adapter_z"], real["adapter_z"]),
        "sim_total_spread_z_vs_real_error_spearman": safe_spearman(
            sim["spread_z"], real_error),
        "real_total_spread_z_vs_real_error_spearman": safe_spearman(
            real["spread_z"], real_error),
        "sim_adapter_spread_z_vs_real_error_spearman": safe_spearman(
            sim["adapter_z"], real_error),
        "real_adapter_spread_z_vs_real_error_spearman": safe_spearman(
            real["adapter_z"], real_error),
        "sim_total_spread_z_vs_sim_error_spearman": safe_spearman(
            sim["spread_z"], sim_error),
        "real_total_spread_z_vs_sim_error_spearman": safe_spearman(
            real["spread_z"], sim_error),
    }
    result["sim_total_for_real_error_top10"] = top_fraction_diagnostic(
        sim["spread_z"], real["error"])
    result["real_total_for_real_error_top10"] = top_fraction_diagnostic(
        real["spread_z"], real["error"])
    result["sim_adapter_for_real_error_top10"] = top_fraction_diagnostic(
        sim["adapter_z"], real["error"])
    result["real_adapter_for_real_error_top10"] = top_fraction_diagnostic(
        real["adapter_z"], real["error"])
    return result


def checkpoint_normalizer(
        payload: Mapping[str, object], device: torch.device,
        ) -> dict[str, torch.Tensor]:
    raw = payload.get("normalizer")
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint has no serialized normalizer")
    result = {}
    for key in ("mean_input", "std_input", "mean_target", "std_target"):
        value = torch.as_tensor(raw[key], device=device, dtype=torch.float32)
        if value.shape != (3,) or not torch.isfinite(value).all():
            raise ValueError(f"invalid checkpoint normalizer field {key}")
        result[key] = value
    return result


def load_model(payload: Mapping[str, object], device: torch.device,
               ) -> EnsembleUNet2d:
    raw = dict(payload["model_config"])
    if raw.pop("architecture", None) != "ensemble_unet":
        raise ValueError("audit checkpoint is not an ensemble U-Net")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**raw)).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def base_members(
        model: EnsembleUNet2d, history_norm: torch.Tensor,
        features: list[torch.Tensor]) -> torch.Tensor:
    members = model._members_from_features(
        history_norm, features, use_member_adapters=False)
    members, _ = model._apply_hard_region_expert(
        members, features, history_norm)
    members, _ = model._apply_conservative_closure(
        members, features, history_norm)
    members = model._apply_recurrent_latent_rollout(
        members, features[-1])
    if model.config.physical_energy_member_sync:
        members = model._energy_controlled_members(
            members, features, history_norm)
    return members


def evaluate_checkpoint(
        checkpoint: Path, loader: DataLoader,
        device: torch.device, element_stride: int) -> tuple[dict, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalizer = checkpoint_normalizer(payload, device)
    model = load_model(payload, device)
    spread_parts: list[np.ndarray] = []
    adapter_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    member_error_parts: list[np.ndarray] = []
    point_abs_sum = 0.0
    point_count = 0
    with torch.inference_mode():
        for batch in loader:
            history, target = batch[:2]
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            history_norm = (
                history - normalizer["mean_input"].view(1, 1, 1, 1, 3)
            ) / normalizer["std_input"].view(1, 1, 1, 1, 3)
            prediction_norm, members_norm, *_middle, features, _coverage = (
                model.forward_uq_training(history_norm))
            base_norm = base_members(model, history_norm, features)
            target_scale = normalizer["std_target"].view(1, 1, 1, 1, 3)
            target_mean = normalizer["mean_target"].view(1, 1, 1, 1, 3)
            member_scale = target_scale.unsqueeze(1)
            members = members_norm * member_scale + target_mean.unsqueeze(1)
            base = base_norm * member_scale + target_mean.unsqueeze(1)
            prediction = prediction_norm * target_scale + target_mean
            truth = target
            spread = members[..., :2].std(dim=1, unbiased=False)
            adapter_spread = (members - base)[..., :2].std(
                dim=1, unbiased=False)
            error = (truth - prediction)[..., :2].abs()
            scored = truth[..., :2] != 0.0
            flat = torch.nonzero(
                scored.reshape(-1), as_tuple=False).flatten()[::element_stride]
            spread_parts.append(
                spatial_zscore(spread).reshape(-1)[flat].float().cpu().numpy())
            adapter_parts.append(
                spatial_zscore(adapter_spread).reshape(-1)[flat]
                .float().cpu().numpy())
            error_parts.append(error.reshape(-1)[flat].float().cpu().numpy())
            member_errors = (truth[..., :2].unsqueeze(1) - members[..., :2])
            member_errors = member_errors.permute(
                0, 2, 3, 4, 5, 1).reshape(-1, members.shape[1])
            member_error_parts.append(
                member_errors[flat].float().cpu().numpy())
            point_abs_sum += float(error[scored].sum(dtype=torch.float64))
            point_count += int(scored.sum())
    arrays = {
        "spread_z": np.concatenate(spread_parts),
        "adapter_z": np.concatenate(adapter_parts),
        "error": np.concatenate(error_parts),
    }
    member_errors = np.concatenate(member_error_parts)
    pairwise = np.corrcoef(member_errors, rowvar=False)
    upper = pairwise[np.triu_indices(pairwise.shape[0], k=1)]
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": dr1.sha256(checkpoint),
        "sampled_elements": int(arrays["error"].size),
        "scored_elements": point_count,
        "mean_absolute_error": point_abs_sum / point_count,
        "member_error_pairwise_correlation_mean": float(upper.mean()),
        "total_spread_z_vs_own_error_spearman": safe_spearman(
            arrays["spread_z"], np.log1p(arrays["error"] / 1.0e-4)),
        "adapter_spread_z_vs_own_error_spearman": safe_spearman(
            arrays["adapter_z"], np.log1p(arrays["error"] / 1.0e-4)),
    }
    del model, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--real-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--element-stride", type=int, default=197)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0 or args.element_stride <= 0:
        raise ValueError("batch size and element stride must be positive")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 manifest byte identity changed")
    rows, _manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = paired.load_evaluation_rows(
        args.evaluation_manifest, rows)
    trajectories, _metadata, windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", ReadOnlyTailAudit())
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal disagreement-transfer audit requires CUDA")
    sim_report, sim_arrays = evaluate_checkpoint(
        args.sim_checkpoint, loader, device, args.element_stride)
    real_report, real_arrays = evaluate_checkpoint(
        args.real_checkpoint, loader, device, args.element_stride)
    result = {
        "schema": "b210_sim_real_disagreement_transfer_audit_v1",
        "support": {
            "windows": len(windows),
            "trajectories": len(trajectories),
            "element_stride": args.element_stride,
            "evaluation_manifest_sha256": dr1.sha256(
                args.evaluation_manifest),
        },
        "sim": sim_report,
        "real": real_report,
        "transfer": cross_map_diagnostic(sim_arrays, real_arrays),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dr1.write_json_atomic(args.output, result)
    print(json.dumps({
        "event": "b210_sim_real_disagreement_transfer_complete",
        "output": str(args.output), "transfer": result["transfer"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
