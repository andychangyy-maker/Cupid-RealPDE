#!/usr/bin/env python3
"""TRH1: test exact A/B fluctuation roles inside the frozen C208 UQ path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from train_flow_backbone import construct_model  # noqa: E402


METHOD = "c208_trh1_teacher_role_headroom"
ACKNOWLEDGEMENT = "TRH1_FIXED725_HEADROOM_ONLY"
EXPECTED_COUNTS = (2576, 725)
EXPECTED_MANIFEST_SHA256 = (
    "b83811f908c8d55a9bd255d158f5f2648b3248a4271bca404c3085f36570dbbd")
EXPECTED_REAL_STATS_SHA256 = (
    "bae5235f73431ac7d76e6bdcea480b8ca9fda0ac01e57ab26af66bad20dc3b40")
EXPECTED_SHA256 = {
    "c208": "0cb72f227f58b8dbd8bf51cb065c08c216383fa8611d8ca1621c3467e796c892",
    "a": "42f259ed60c9f275027fef3b27f1d969283a2286756c84e2c58c6a143b865334",
    "b": "1940b32afa42b2feadf223a2ef4cbdfb2b87af52604992cfe62d46fd7604884e",
}
AOAS = (0, 5, 10, 15, 20)
BASELINE_SPS = 54.024427
BASELINE_SPS_TOLERANCE = 0.03
FORMAL_BATCH_SIZE = 16
FORMAL_ELEMENT_STRIDE = 197
MEMBER_MEAN_TOLERANCE_PHYSICAL = 2e-4
ROLE_TIME_MEAN_TOLERANCE_PHYSICAL = 2e-6
WALL_SECONDS_MAX = 15 * 60
SPS_DELTA_MIN = 0.25
POSITIVE_AOAS_MIN = 4
WORST_AOA_SPS_DELTA_MIN = -0.05
ELEMENT_SPEARMAN_DELTA_MIN = 0.02
SAMPLE_SPEARMAN_DELTA_MIN = 0.05
TOP_ERROR_RECALL_DELTA_MIN = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c208-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-a", type=Path, required=True)
    parser.add_argument("--teacher-b", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--element-stride", type=int, default=197)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-fixed725-headroom-only")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.execute_formal_gpu:
        raise ValueError("formal TRH1 requires --execute-formal-gpu")
    if args.acknowledge_fixed725_headroom_only != ACKNOWLEDGEMENT:
        raise ValueError("formal TRH1 acknowledgement differs")
    if args.batch_size != FORMAL_BATCH_SIZE:
        raise ValueError("formal TRH1 freezes batch-size=16")
    if args.element_stride != FORMAL_ELEMENT_STRIDE:
        raise ValueError("formal TRH1 freezes element-stride=197")
    if args.num_workers != 0:
        raise ValueError("formal TRH1 freezes num-workers=0")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite TRH1 result: {args.output}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale TRH1 temporary result exists: {temporary}")
    if torch.device(args.device).type != "cuda":
        raise ValueError("formal TRH1 requires CUDA")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_identity() -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "branch": branch, "status_porcelain": status}


def load_model(checkpoint: Path, normalizer, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    architecture = config.get("architecture")
    if architecture != "ensemble_unet":
        raise ValueError(f"{checkpoint} is not ensemble_unet")
    model = construct_model(architecture, config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    common.set_model_stats(model, normalizer)
    model.to(dtype=torch.float16).eval()
    metadata = {
        "architecture": architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "phase": payload.get("phase"),
        "iteration": payload.get("iteration"),
    }
    del payload
    return model, metadata


@torch.inference_mode()
def collect_teacher(model, loader, normalizer,
                    device: torch.device) -> torch.Tensor:
    rows = []
    dtype = next(model.parameters()).dtype
    for history, _target in loader:
        history = history.to(device, non_blocking=True)
        history_norm = (
            (history - normalizer.mean_input) / normalizer.std_input)
        prediction_norm = model(history_norm.to(dtype=dtype))
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("teacher produced non-finite point")
        rows.append(prediction.float().cpu())
    return torch.cat(rows).contiguous()


def compose_role_members(
        prediction: torch.Tensor, members: torch.Tensor,
        teacher_a: torch.Tensor, teacher_b: torch.Tensor,
        target_std_uv: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, float]]:
    """Add exact zero-sum A/B teacher deviations to C208 members."""
    if members.ndim != 6 or members.shape[1] != 4 or members.shape[-1] < 2:
        raise ValueError("members must have shape [B,4,T,H,W,C>=2]")
    expected = (prediction.shape[0], *prediction.shape[1:4], 2)
    if prediction.ndim != 5 or tuple(teacher_a.shape) != expected \
            or tuple(teacher_b.shape) != expected:
        raise ValueError("prediction and A/B teacher shapes differ")
    if target_std_uv.shape != (2,) or bool((target_std_uv <= 0).any()):
        raise ValueError("target_std_uv must contain two positive values")
    if not all(bool(torch.isfinite(value).all()) for value in (
            prediction, members, teacher_a, teacher_b, target_std_uv)):
        raise FloatingPointError("non-finite TRH1 role input")

    a = teacher_a.float()
    b = teacher_b.float()
    a_fluct = a - a.mean(dim=1, keepdim=True)
    b_fluct = b - b.mean(dim=1, keepdim=True)
    role_delta = 0.5 * (a_fluct - b_fluct)
    delta_norm = role_delta / target_std_uv.float().view(1, 1, 1, 1, 2)

    original_uv = members[..., :2].float()
    candidate_uv = original_uv.clone()
    candidate_uv[:, :2] += delta_norm.unsqueeze(1)
    candidate_uv[:, 2:] -= delta_norm.unsqueeze(1)

    candidate = members.clone()
    candidate[..., :2] = candidate_uv.to(dtype=members.dtype)
    scale = target_std_uv.float().view(1, 1, 1, 1, 1, 2)
    member_mean_change = (
        (candidate[..., :2].float().mean(dim=1)
         - original_uv.mean(dim=1)).abs()
        * target_std_uv.float().view(1, 1, 1, 1, 2)).max()
    role_time_mean = torch.maximum(
        a_fluct.mean(dim=1).abs().max(),
        b_fluct.mean(dim=1).abs().max())
    role_spread = (candidate[..., :2].float() * scale).std(
        dim=1, unbiased=False)
    return candidate, {
        "member_mean_change_max_abs_physical": float(member_mean_change),
        "teacher_role_time_mean_max_abs_physical": float(role_time_mean),
        "candidate_role_spread_mean_physical": float(role_spread.mean()),
        "non_uv_members_reused_by_identity": bool(torch.equal(
            candidate[..., 2:], members[..., 2:])),
    }


class SPSAccumulator:
    def __init__(self) -> None:
        self.numerator = 0.0
        self.scored = 0
        self.covered = 0
        self.width = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor,
               lower: torch.Tensor, upper: torch.Tensor) -> None:
        residual = target - prediction
        scored = target != 0.0
        inside = (residual >= -lower) & (residual <= upper)
        factor = common.sample_sps_factor(prediction, target).view(
            -1, 1, 1, 1, 1)
        utility = factor * inside * torch.exp(
            -(lower + upper) / common.SPS_SIGMA)
        self.numerator += float(utility[scored].sum(dtype=torch.float64))
        self.scored += int(scored.sum())
        self.covered += int((inside & scored).sum())
        self.width += float(
            (lower + upper)[scored].sum(dtype=torch.float64))

    def result(self) -> dict[str, float | int]:
        if self.scored <= 0:
            raise RuntimeError("TRH1 SPS accumulator is empty")
        return {
            "sps_score": 100.0 * self.numerator / self.scored,
            "coverage": self.covered / self.scored,
            "mean_width": self.width / self.scored,
            "scored_elements": self.scored,
            "covered_elements": self.covered,
        }


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size != left.size:
        return 0.0
    if not bool(np.isfinite(left).all() and np.isfinite(right).all()):
        return 0.0

    def average_ranks(value: np.ndarray) -> np.ndarray:
        order = np.argsort(value, kind="mergesort")
        sorted_value = value[order]
        starts = np.concatenate((
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_value[1:] != sorted_value[:-1]) + 1))
        ends = np.concatenate((starts[1:], np.array([len(value)])))
        sorted_ranks = np.repeat(
            (starts + ends - 1).astype(np.float64) / 2.0,
            ends - starts)
        ranks = np.empty(len(value), dtype=np.float64)
        ranks[order] = sorted_ranks
        return ranks

    left_rank = average_ranks(np.asarray(left).reshape(-1))
    right_rank = average_ranks(np.asarray(right).reshape(-1))
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = math.sqrt(
        float(np.sum(left_rank * left_rank))
        * float(np.sum(right_rank * right_rank)))
    if denominator == 0.0:
        return 0.0
    value = float(np.sum(left_rank * right_rank)) / denominator
    return value if math.isfinite(value) else 0.0


def pad_zero_pressure(uv: torch.Tensor) -> torch.Tensor:
    if uv.ndim != 5 or uv.shape[-1] != 2:
        raise ValueError("pressure padding requires [B,T,H,W,2] UV fields")
    result = uv.new_zeros((*uv.shape[:-1], 3))
    result[..., :2] = uv
    return result


def recompute_uq_bounds(model, features: list[torch.Tensor],
                        members: torch.Tensor, model_input: torch.Tensor,
                        prediction: torch.Tensor,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    flow_prediction = (
        prediction if model.config.shared_member_uv_mean else None)
    fused = model._fused_uq_features(
        features, members, model_input, prediction=flow_prediction)
    quantiles = model._quantiles_from_fused(fused, members)
    coverage = model._coverage_logits(fused)
    lower, upper = model._utility_interval(quantiles, coverage)
    lower, upper = model._refine_utility_interval(lower, upper, fused)
    return model._ordered_interval_extents(lower, upper)


def rank_metrics(error: np.ndarray, spread: np.ndarray,
                 sample_error: np.ndarray,
                 sample_spread: np.ndarray) -> dict[str, float]:
    if error.size == 0 or sample_error.size == 0:
        raise ValueError("rank diagnostics require non-empty arrays")
    log_error = np.log1p(error / 1e-4)
    log_spread = np.log1p(spread / 1e-4)
    top_count = max(1, math.ceil(0.1 * len(error)))
    top_error = np.argsort(-error, kind="mergesort")[:top_count]
    top_spread = np.argsort(-spread, kind="mergesort")[:top_count]
    top_overlap = np.intersect1d(
        top_error, top_spread, assume_unique=True).size
    return {
        "element_log_spearman": safe_spearman(log_spread, log_error),
        "sample_spearman": safe_spearman(sample_spread, sample_error),
        "top_error_recall_at_top_spread": top_overlap / top_count,
        "top_fraction_actual": top_count / len(error),
        "mean_spread": float(spread.mean()),
        "spread_std": float(spread.astype(np.float64).std()),
    }


def gate_decision(report: dict) -> dict:
    hard = report["hard_guards"]
    pooled = report["pooled"]
    aoa_rows = report["aoa_rows"]
    rank_delta = report["rank_delta"]
    aoas_exact = (
        len(aoa_rows) == len(AOAS)
        and sorted(row["aoa"] for row in aoa_rows) == list(AOAS))
    sps_deltas = [row["sps_delta"] for row in aoa_rows]
    aoa_rows_finite = (
        aoas_exact
        and all(math.isfinite(float(value)) for value in sps_deltas))
    pooled_finite = math.isfinite(float(pooled["sps_delta"]))
    rank_finite = all(math.isfinite(float(rank_delta[key])) for key in (
        "element_log_spearman", "sample_spearman",
        "top_error_recall_at_top_spread"))
    checks = {
        "hard_guards": all(hard.values()),
        "aoa_rows_exact": aoas_exact,
        "aoa_rows_finite": aoa_rows_finite,
        "pooled_sps_delta": (
            pooled_finite and pooled["sps_delta"] >= SPS_DELTA_MIN),
        "positive_aoas": (
            aoa_rows_finite
            and sum(value > 0.0 for value in sps_deltas)
            >= POSITIVE_AOAS_MIN),
        "worst_aoa": (
            aoa_rows_finite
            and min(sps_deltas) >= WORST_AOA_SPS_DELTA_MIN),
        "element_spearman_delta": (
            rank_finite and rank_delta["element_log_spearman"]
            >= ELEMENT_SPEARMAN_DELTA_MIN),
        "sample_spearman_delta": (
            rank_finite
            and rank_delta["sample_spearman"] >= SAMPLE_SPEARMAN_DELTA_MIN),
        "top_error_recall_delta": (
            rank_finite and rank_delta["top_error_recall_at_top_spread"]
            >= TOP_ERROR_RECALL_DELTA_MIN),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "decision": (
            "GO_TRH1_TO_ZERO_SUM_ROLE_BRANCH"
            if passed else "STOP_TRH1_NO_ROLE_HEADROOM"),
    }


@torch.inference_mode()
def evaluate(model, loader, normalizer, device: torch.device,
             teacher_a: torch.Tensor, teacher_b: torch.Tensor,
             aoa_labels: list[int], element_stride: int) -> dict:
    baseline = SPSAccumulator()
    candidate = SPSAccumulator()
    baseline_by_aoa = {value: SPSAccumulator() for value in AOAS}
    candidate_by_aoa = {value: SPSAccumulator() for value in AOAS}
    error_parts = []
    baseline_spread_parts = []
    role_spread_parts = []
    sample_error_parts = []
    baseline_sample_spread_parts = []
    role_sample_spread_parts = []
    target_digest = hashlib.sha256()
    history_digest = hashlib.sha256()
    point_digest = hashlib.sha256()
    scored_mask_digest = hashlib.sha256()
    max_member_mean_error = 0.0
    max_role_time_mean = 0.0
    non_uv_members_reused = True
    pressure_output_contract = True
    alpha0_bounds_bitwise = True
    offset = 0
    scored_offset = 0
    dtype = next(model.parameters()).dtype
    uv_scale = normalizer.std_target[:2].float()
    extent_scale = uv_scale.view(1, 1, 1, 1, 2)
    member_scale = uv_scale.view(1, 1, 1, 1, 1, 2)

    for history, target in loader:
        batch = len(history)
        history_digest.update(history.contiguous().numpy().tobytes())
        target_digest.update(target.contiguous().numpy().tobytes())
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _ = normalizer.preprocess(history, target)
        model_input = history_norm.to(dtype=dtype)
        values = model.forward_uq_training(model_input)
        (prediction_norm, members_norm, _quantiles,
         baseline_lower_norm, baseline_upper_norm, features,
         _coverage) = values
        a = teacher_a[offset:offset + batch].to(device, non_blocking=True)
        b = teacher_b[offset:offset + batch].to(device, non_blocking=True)
        role_members, audit = compose_role_members(
            prediction_norm, members_norm, a, b, uv_scale.to(device))
        max_member_mean_error = max(
            max_member_mean_error,
            audit["member_mean_change_max_abs_physical"])
        max_role_time_mean = max(
            max_role_time_mean,
            audit["teacher_role_time_mean_max_abs_physical"])
        non_uv_members_reused = (
            non_uv_members_reused
            and audit["non_uv_members_reused_by_identity"])

        identity_lower_norm, identity_upper_norm = recompute_uq_bounds(
            model, features, members_norm, model_input, prediction_norm)
        alpha0_bounds_bitwise = (
            alpha0_bounds_bitwise
            and bool(torch.equal(identity_lower_norm, baseline_lower_norm))
            and bool(torch.equal(identity_upper_norm, baseline_upper_norm)))
        lower_norm, upper_norm = recompute_uq_bounds(
            model, features, role_members, model_input, prediction_norm)
        finite_tensors = (
            target, prediction_norm, members_norm, role_members,
            baseline_lower_norm, baseline_upper_norm,
            identity_lower_norm, identity_upper_norm, lower_norm, upper_norm)
        if not all(bool(torch.isfinite(value).all())
                   for value in finite_tensors):
            raise FloatingPointError("TRH1 C208 path produced non-finite value")
        if bool(((baseline_lower_norm + baseline_upper_norm) < 0.0).any()) \
                or bool(((lower_norm + upper_norm) < 0.0).any()):
            raise FloatingPointError("TRH1 interval ordering contract failed")

        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        deployment_point = pad_zero_pressure(prediction)
        pressure_output_contract = (
            pressure_output_contract
            and bool(torch.equal(deployment_point[..., :2], prediction))
            and int(torch.count_nonzero(deployment_point[..., 2])) == 0)
        del deployment_point
        truth = target[..., :2].float()
        baseline_lower = baseline_lower_norm.float() * extent_scale.to(device)
        baseline_upper = baseline_upper_norm.float() * extent_scale.to(device)
        role_lower = lower_norm.float() * extent_scale.to(device)
        role_upper = upper_norm.float() * extent_scale.to(device)
        baseline.update(
            prediction, truth, baseline_lower, baseline_upper)
        candidate.update(prediction, truth, role_lower, role_upper)

        labels = torch.as_tensor(
            aoa_labels[offset:offset + batch], device=device)
        for aoa in AOAS:
            selected = labels == aoa
            if bool(selected.any()):
                baseline_by_aoa[aoa].update(
                    prediction[selected], truth[selected],
                    baseline_lower[selected], baseline_upper[selected])
                candidate_by_aoa[aoa].update(
                    prediction[selected], truth[selected],
                    role_lower[selected], role_upper[selected])

        residual = (truth - prediction).abs()
        scored = truth != 0.0
        scored_mask_digest.update(
            scored.detach().cpu().contiguous().numpy().tobytes())
        baseline_spread = (
            members_norm[..., :2].float() * member_scale.to(device)
        ).std(dim=1, unbiased=False)
        role_spread = (
            role_members[..., :2].float() * member_scale.to(device)
        ).std(dim=1, unbiased=False)
        scored_error = residual.reshape(-1)[scored.reshape(-1)]
        scored_baseline_spread = baseline_spread.reshape(-1)[
            scored.reshape(-1)]
        scored_role_spread = role_spread.reshape(-1)[scored.reshape(-1)]
        phase = (-scored_offset) % element_stride
        indices = torch.arange(
            phase, len(scored_error), element_stride, device=device)
        error_parts.append(scored_error[indices].cpu().numpy())
        baseline_spread_parts.append(
            scored_baseline_spread[indices].cpu().numpy())
        role_spread_parts.append(
            scored_role_spread[indices].cpu().numpy())
        scored_offset += len(scored_error)
        scored_float = scored.to(residual.dtype)
        denominator = scored_float.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
        sample_error_parts.append((
            (residual * scored_float).sum(dim=(1, 2, 3, 4))
            / denominator).cpu().numpy())
        baseline_sample_spread_parts.append((
            (baseline_spread * scored_float).sum(dim=(1, 2, 3, 4))
            / denominator).cpu().numpy())
        role_sample_spread_parts.append((
            (role_spread * scored_float).sum(dim=(1, 2, 3, 4))
            / denominator).cpu().numpy())

        point_digest.update(
            prediction_norm.detach().cpu().contiguous().numpy().tobytes())
        offset += batch

    if offset != len(teacher_a) or offset != len(aoa_labels):
        raise RuntimeError("TRH1 stream length differs from teacher arrays")
    before = baseline.result()
    after = candidate.result()
    aoa_rows = []
    for aoa in AOAS:
        row_before = baseline_by_aoa[aoa].result()
        row_after = candidate_by_aoa[aoa].result()
        aoa_rows.append({
            "aoa": aoa,
            "baseline": row_before,
            "candidate": row_after,
            "sps_delta": row_after["sps_score"] - row_before["sps_score"],
        })
    error = np.concatenate(error_parts)
    sample_error = np.concatenate(sample_error_parts)
    rank_before = rank_metrics(
        error, np.concatenate(baseline_spread_parts), sample_error,
        np.concatenate(baseline_sample_spread_parts))
    rank_after = rank_metrics(
        error, np.concatenate(role_spread_parts), sample_error,
        np.concatenate(role_sample_spread_parts))
    rank_delta = {
        key: rank_after[key] - rank_before[key]
        for key in rank_before
    }
    return {
        "pooled": {
            "baseline": before,
            "candidate": after,
            "sps_delta": after["sps_score"] - before["sps_score"],
        },
        "aoa_rows": aoa_rows,
        "rank_baseline": rank_before,
        "rank_candidate": rank_after,
        "rank_delta": rank_delta,
        "stream": {
            "samples": offset,
            "history_sha256": history_digest.hexdigest(),
            "target_sha256": target_digest.hexdigest(),
            "point_sha256": point_digest.hexdigest(),
            "scored_mask_sha256": scored_mask_digest.hexdigest(),
            "scored_elements": scored_offset,
            "sampled_scored_elements": int(error.size),
        },
        "invariants": {
            "point_reused_by_identity": True,
            "alpha0_bounds_bitwise_replay": alpha0_bounds_bitwise,
            "non_uv_members_reused_by_identity": non_uv_members_reused,
            "pressure_constructed_and_verified_zero": pressure_output_contract,
            "member_mean_change_max_abs_physical": max_member_mean_error,
            "teacher_role_time_mean_max_abs_physical": max_role_time_mean,
            "global_stride_sample_count_exact": (
                int(error.size)
                == (scored_offset + element_stride - 1) // element_stride),
            "scored_mask_identity": (
                before["scored_elements"] == after["scored_elements"]
                == scored_offset),
            "aoa_scored_partition_exact": (
                sum(row["baseline"]["scored_elements"] for row in aoa_rows)
                == before["scored_elements"]
                and sum(row["candidate"]["scored_elements"]
                        for row in aoa_rows)
                == after["scored_elements"]),
        },
    }


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    git = git_identity()
    identities = {
        "c208": {"path": str(args.c208_checkpoint),
                  "sha256": file_sha256(args.c208_checkpoint)},
        "a": {"path": str(args.teacher_a),
              "sha256": file_sha256(args.teacher_a)},
        "b": {"path": str(args.teacher_b),
              "sha256": file_sha256(args.teacher_b)},
        "real_stats": {"path": str(args.real_stats),
                       "sha256": file_sha256(args.real_stats)},
    }
    identity_checks = {
        name: identities[name]["sha256"] == EXPECTED_SHA256[name]
        for name in EXPECTED_SHA256
    }
    identity_checks["real_stats"] = (
        identities["real_stats"]["sha256"] == EXPECTED_REAL_STATS_SHA256)
    if not all(identity_checks.values()) or git["status_porcelain"]:
        raise RuntimeError("TRH1 provenance or clean-Git guard failed")

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    train_windows, tail_windows, manifest = common.repro.make_split(
        metadata, 20, 0.2)
    if (len(train_windows), len(tail_windows)) != EXPECTED_COUNTS:
        raise RuntimeError("TRH1 fixed split counts changed")
    manifest_sha = canonical_sha256(manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("TRH1 fixed split manifest changed")
    dataset = common.repro.WindowDataset(trajectories, tail_windows)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    aoa_labels = [int(metadata[window.trajectory]["aoa"])
                  for window in tail_windows]
    if sorted(set(aoa_labels)) != list(AOAS):
        raise RuntimeError("TRH1 tail AoA labels changed")
    normalizer = common.repro.Normalizer(args.real_stats, device)

    teacher_a_model, teacher_a_metadata = load_model(
        args.teacher_a, normalizer, device)
    teacher_a = collect_teacher(
        teacher_a_model, loader, normalizer, device)
    del teacher_a_model
    gc.collect()
    torch.cuda.empty_cache()
    teacher_b_model, teacher_b_metadata = load_model(
        args.teacher_b, normalizer, device)
    teacher_b = collect_teacher(
        teacher_b_model, loader, normalizer, device)
    del teacher_b_model
    gc.collect()
    torch.cuda.empty_cache()

    c208_model, c208_metadata = load_model(
        args.c208_checkpoint, normalizer, device)
    evidence = evaluate(
        c208_model, loader, normalizer, device,
        teacher_a, teacher_b, aoa_labels, args.element_stride)
    elapsed = time.perf_counter() - started
    hard_guards = {
        "git_clean": git["status_porcelain"] == "",
        "checkpoint_identities": all(identity_checks.values()),
        "split_identity": manifest_sha == EXPECTED_MANIFEST_SHA256,
        "samples_exact": evidence["stream"]["samples"] == 725,
        "baseline_replay": abs(
            evidence["pooled"]["baseline"]["sps_score"] - BASELINE_SPS
        ) <= BASELINE_SPS_TOLERANCE,
        "point_identity": evidence["invariants"]["point_reused_by_identity"],
        "alpha0_bounds_replay": (
            evidence["invariants"]["alpha0_bounds_bitwise_replay"]),
        "non_uv_members_identity": (
            evidence["invariants"]["non_uv_members_reused_by_identity"]),
        "pressure_output_contract": (
            evidence["invariants"]["pressure_constructed_and_verified_zero"]),
        "member_mean": (
            evidence["invariants"]["member_mean_change_max_abs_physical"]
            <= MEMBER_MEAN_TOLERANCE_PHYSICAL),
        "teacher_role_time_mean": (
            evidence["invariants"]["teacher_role_time_mean_max_abs_physical"]
            <= ROLE_TIME_MEAN_TOLERANCE_PHYSICAL),
        "global_stride_sample_count": (
            evidence["invariants"]["global_stride_sample_count_exact"]),
        "scored_mask_identity": (
            evidence["invariants"]["scored_mask_identity"]),
        "aoa_scored_partition": (
            evidence["invariants"]["aoa_scored_partition_exact"]),
        "rank_spread_nonconstant": (
            evidence["rank_baseline"]["spread_std"] > 0.0
            and evidence["rank_candidate"]["spread_std"] > 0.0),
        "finite_metrics": all(math.isfinite(float(value)) for value in (
            evidence["pooled"]["baseline"]["sps_score"],
            evidence["pooled"]["candidate"]["sps_score"],
            *evidence["rank_baseline"].values(),
            *evidence["rank_candidate"].values())),
        "wall_budget": elapsed <= WALL_SECONDS_MAX,
        "no_training_checkpoint_or_zip": True,
    }
    report = {
        "method": METHOD,
        "evidence_class": (
            "fixed725 target-independent A/B teacher intervention with "
            "target-visible scoring; not official, LOAO, or a candidate"),
        "contract": {
            "intervention": (
                "alpha1 exact zero-sum A/B fluctuation deviations added to "
                "original C208 members; C208 point and all UQ weights frozen"),
            "no_sweep": True,
            "fixed725_reused": True,
            "batch_size": FORMAL_BATCH_SIZE,
            "global_scored_element_stride": FORMAL_ELEMENT_STRIDE,
            "gate": {
                "sps_delta_min": SPS_DELTA_MIN,
                "positive_aoas_min": POSITIVE_AOAS_MIN,
                "worst_aoa_sps_delta_min": WORST_AOA_SPS_DELTA_MIN,
                "element_spearman_delta_min": ELEMENT_SPEARMAN_DELTA_MIN,
                "sample_spearman_delta_min": SAMPLE_SPEARMAN_DELTA_MIN,
                "top_error_recall_delta_min": TOP_ERROR_RECALL_DELTA_MIN,
            },
        },
        "provenance": {
            "git": git,
            "identities": identities,
            "identity_checks": identity_checks,
            "manifest_sha256": manifest_sha,
            "c208_model": c208_metadata,
            "teacher_a_model": teacher_a_metadata,
            "teacher_b_model": teacher_b_metadata,
        },
        **evidence,
        "hard_guards": hard_guards,
        "wall_seconds": elapsed,
        "checkpoint": None,
        "zip": None,
    }
    report["gate"] = gate_decision(report)
    report["decision"] = report["gate"]["decision"]
    write_json_atomic(args.output, report)
    print(json.dumps({
        "event": "trh1_complete",
        "decision": report["decision"],
        "sps_delta": report["pooled"]["sps_delta"],
        "rank_delta": report["rank_delta"],
        "wall_seconds": elapsed,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
