#!/usr/bin/env python3
"""Train a frozen-point native 7q decoder from a fold-safe P0 anchor.

Only the uncertainty modules of ``--point-checkpoint`` are trainable.  The
P0 anchor is reconstructed in RAM, fold by fold, from the formal A/B OOF
checkpoints and the C temporal-mean expert.  The first phase sees only 2032
point-split pseudo-labels; the second phase sees only 572 tail truths.  Fold
04 (544 point + 153 tail) is never used for optimization and the tail is
evaluated exactly once after the fixed final update.

The reconstructed seven-quantile bank is P0-anchored (same A/B/C sources and
weights).  It is deliberately not called the exact P0 endpoint distribution:
the source models' utility-refinement modules act after interval selection and
therefore cannot be inverted into seven quantiles.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, Path(__file__).parent, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_mean_expert_oof_blend as s1  # noqa: E402
import evaluate_p1_observation_gated_point_blend as point_gate  # noqa: E402
import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from train_distilled_flow_corrector import load_forecaster  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


QUANTILE_LEVELS = (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)
WEIGHT_B = 0.625
WEIGHT_C = 0.25
HELD_FOLD = 4
EXPECTED_POINT_WINDOWS = 2576
EXPECTED_TAIL_WINDOWS = 725
EXPECTED_TRAIN_POINT = 2032
EXPECTED_HELD_POINT = 544
EXPECTED_TRAIN_TAIL = 572
EXPECTED_HELD_TAIL = 153
REFERENCE_WIDTH = (0.01075, 0.00550)
FOLD04_REFERENCE = {
    "sps_score": 53.297578817787475,
    "coverage": 0.8991232113472482,
    "mean_width": 0.016455948464344353,
}
TEACHER_MIN_SPS = 53.30
MAX_COVERAGE_DROP = 0.005
MAX_WIDTH_RATIO = 1.05
ANCHOR_FORMAT = "p0_abc_fold_safe_absolute_outcome_7q_fp16_ram_v1"
FORMAL_POINT_SHA256 = (
    "758b44b2ce124d9c1bf72657d1a39247481c4794a6f2837bac8ffc072836ce07")
FORMAL_SOURCE_SHA256 = {
    "point_checkpoint_a": (
        "42f259ed60c9f275027fef3b27f1d969283a2286756c84e2c58c6a143b865334"),
    "point_checkpoint_b": (
        "1940b32afa42b2feadf223a2ef4cbdfb2b87af52604992cfe62d46fd7604884e"),
    "mean_expert_checkpoint": (
        "3faea6c6aaa6fecf8543267dff451a665a9c523c5dc888c98287baaa6a326818"),
}
FORMAL_OOF_SHA256 = {
    "a": {
        "summary.json": (
            "8533d8913d9f3c84b0ed398ba04e3098684b21e73794ccd022bf39fb99f6472b"),
        "fold_00_best.pth": (
            "60a7e2fdf9cacd87c9d25aaf1499d0085c71e7a69b843d6c309cd0969e489383"),
        "fold_01_best.pth": (
            "0561a05c67cae8ae3d49444a94ba7fa451d17d2083a5ba66a905f8e776c699af"),
        "fold_02_best.pth": (
            "74ec6b6d89d2a4bc341ebadd08a9e9681ca022000124eef2917bb683996dd8a5"),
        "fold_03_best.pth": (
            "1cef6e69e2338f1b099c8a61a1d6a359743bd2704ff5859cb76215d8503feec4"),
        "fold_04_best.pth": (
            "05fbdf7080b5257d5a74615941557f7a1db19c025c182ca6a5426d9e8b0318cc"),
    },
    "b": {
        "summary.json": (
            "72c54235e24bdc237d27552b560d02bf7dacea0ea02d6324e5d99ded0f02ecca"),
        "fold_00_best.pth": (
            "9759500063db47371eb60d2fa990181ef0a3a20c6859313bb8995122694ffd5e"),
        "fold_01_best.pth": (
            "5799729db29d77bbd0f3eb461a6e00da4b76339be3f4e4b0a311b3514a3b8083"),
        "fold_02_best.pth": (
            "a0b43f70ed2cfe1e564f333c8bf934a9730ffcc4329728bf44b8bedaf05a67b0"),
        "fold_03_best.pth": (
            "79ee4b2a2fe282fadc756935281f39f36e46f131838b9bb283717c47c33f7dd6"),
        "fold_04_best.pth": (
            "f3fb10ba7464d348271cbe6161946b55829868892880cd82f03553cd349c6c8f"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-point-sha256", required=True)
    parser.add_argument("--point-checkpoint-a", type=Path, required=True)
    parser.add_argument("--oof-dir-a", type=Path, required=True)
    parser.add_argument("--point-checkpoint-b", type=Path, required=True)
    parser.add_argument("--oof-dir-b", type=Path, required=True)
    parser.add_argument("--mean-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--anchor-only-updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--reconstruction-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--truth-pinball-weight", type=float, default=0.25)
    parser.add_argument("--soft-sps-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=0.25)
    parser.add_argument("--coverage-target", type=float,
                        default=FOLD04_REFERENCE["coverage"])
    parser.add_argument("--soft-tau-fraction", type=float, default=0.10)
    parser.add_argument("--time-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.anchor_only_updates, args.batch_size,
        args.eval_batch_size, args.reconstruction_batch_size, args.lr,
        args.soft_tau_fraction, args.time_budget_seconds,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("positive training arguments must be finite")
    if args.anchor_only_updates >= args.updates:
        raise ValueError("anchor-only-updates must be smaller than updates")
    nonnegative = (
        args.weight_decay, args.grad_clip,
        args.truth_pinball_weight, args.soft_sps_weight,
        args.coverage_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("loss and optimizer arguments must be non-negative")
    if not 0.0 < args.coverage_target < 1.0:
        raise ValueError("coverage-target must lie in (0,1)")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    validate_sha256(args.expected_point_sha256)
    if args.expected_point_sha256 != FORMAL_POINT_SHA256:
        raise ValueError("formal point SHA-256 must be the frozen SHA758b asset")


def validate_sha256(value: str) -> None:
    if (len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("expected SHA-256 must be lowercase hexadecimal")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_identity(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def point_state_identity(model: torch.nn.Module) -> str:
    return state_identity(
        (name, value) for name, value in model.state_dict().items()
        if not name.startswith(uq_train.UQ_PREFIXES))


def fold_partition_indices(
        folds: list[int] | np.ndarray | torch.Tensor,
        held_fold: int = HELD_FOLD) -> tuple[list[int], list[int]]:
    values = np.asarray(folds, dtype=np.int64)
    if values.ndim != 1 or set(values.tolist()) != set(range(5)):
        raise ValueError("fold labels must be a non-empty 0..4 partition")
    train = np.flatnonzero(values != held_fold).tolist()
    held = np.flatnonzero(values == held_fold).tolist()
    if (set(train) & set(held)
            or sorted(train + held) != list(range(len(values)))):
        raise RuntimeError("fold partition overlaps or is incomplete")
    return train, held


def partition_fold_labels(manifest: list[dict], fold_manifest: dict,
                          starts_key: str) -> list[int]:
    """Expand condition folds over either train_starts or val_starts."""
    if starts_key not in {"train_starts", "val_starts"}:
        raise ValueError("starts_key must be train_starts or val_starts")
    assignments = fold_manifest.get("assignments", [])
    if len(assignments) != len(manifest):
        raise ValueError("fold manifest does not align with split manifest")
    labels = []
    for item, assignment in zip(manifest, assignments):
        if (assignment.get("file") != item.get("file")
                or assignment.get("re") != item.get("re")
                or assignment.get("aoa") != item.get("aoa")):
            raise ValueError("condition assignment order differs from split")
        fold = int(assignment["fold"])
        if fold not in range(5):
            raise ValueError("condition fold lies outside 0..4")
        labels.extend([fold] * len(item[starts_key]))
    return labels


def training_stage(iteration: int, anchor_only_updates: int,
                   total_updates: int) -> str:
    if (not 1 <= iteration <= total_updates
            or not 1 <= anchor_only_updates < total_updates):
        raise ValueError("invalid two-stage update boundary")
    return (
        "point2032_p0_7q_pretrain"
        if iteration <= anchor_only_updates
        else "tail572_truth_refine")


def compose_p0_absolute_quantiles(
        outcome_quantiles_a: torch.Tensor,
        outcome_quantiles_b: torch.Tensor,
        prediction_a: torch.Tensor,
        prediction_b: torch.Tensor,
        prediction_c: torch.Tensor,
        weight_b: float = WEIGHT_B,
        weight_c: float = WEIGHT_C) -> torch.Tensor:
    """Blend complete A/B outcome quantiles and apply the P0 C mean shift."""
    if (outcome_quantiles_a.shape != outcome_quantiles_b.shape
            or outcome_quantiles_a.ndim != 6
            or outcome_quantiles_a.shape[-2:] != (2, 7)):
        raise ValueError("A/B outcome quantiles must share (N,T,H,W,2,7)")
    expected = outcome_quantiles_a.shape[:-1]
    if any(value.shape != expected for value in (
            prediction_a, prediction_b, prediction_c)):
        raise ValueError("A/B/C point predictions do not align with quantiles")
    if not (0.0 <= weight_b <= 1.0 and 0.0 <= weight_c <= 1.0):
        raise ValueError("P0 blend weights must lie in [0,1]")
    base_prediction = prediction_a.lerp(prediction_b, weight_b)
    correction = weight_c * (
        prediction_c.mean(dim=1, keepdim=True)
        - base_prediction.mean(dim=1, keepdim=True))
    result = outcome_quantiles_a.lerp(
        outcome_quantiles_b, weight_b) + correction.unsqueeze(-1)
    if (not torch.isfinite(result).all()
            or not torch.all(result[..., 1:] > result[..., :-1])):
        raise FloatingPointError("P0-anchored quantiles are invalid/non-monotone")
    return result


def validate_oof_quantile_contract(
        point_checkpoint: Path, oof_dir: Path,
        fold_manifest: dict, source: str) -> dict:
    expected_files = FORMAL_OOF_SHA256[source]
    identities = {
        name: s1.file_identity(oof_dir / name)
        for name in expected_files
    }
    for name, expected in expected_files.items():
        if identities[name]["sha256"] != expected:
            raise ValueError(
                f"formal {source.upper()} OOF SHA-256 differs at {name}")
    summary = s1.validate_oof_summary(
        oof_dir / "summary.json", point_checkpoint, fold_manifest)
    rows = []
    for fold in range(5):
        path = oof_dir / f"fold_{fold:02d}_best.pth"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload.get("model_config", {})
        state = payload.get("uq_state_dict", {})
        if (payload.get("architecture") != "ensemble_unet_uq_oof"
                or payload.get("fold") != fold
                or tuple(config.get("quantile_levels", ()))
                != QUANTILE_LEVELS
                or "quantile_head.weight" not in state
                or "quantile_head.bias" not in state):
            raise ValueError(f"OOF fold lacks the complete 7q contract: {path}")
        rows.append({
            "fold": fold,
            "checkpoint": identities[path.name],
            "quantile_levels": list(QUANTILE_LEVELS),
        })
        del payload
    return {
        "summary": summary,
        "summary_file": identities["summary.json"],
        "folds": rows,
    }


@torch.inference_mode()
def collect_point_predictions(model, loader, normalizer,
                              device: torch.device) -> torch.Tensor:
    rows = []
    model.eval()
    for history, _unused_target in loader:
        history = history.to(device, non_blocking=True)
        history_norm = (
            history - normalizer.mean_input) / normalizer.std_input
        prediction_norm = model(history_norm.to(dtype=next(
            model.parameters()).dtype))
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        rows.append(prediction.cpu().to(torch.float16))
    return torch.cat(rows)


@torch.inference_mode()
def collect_outcome_quantiles(model, loader, normalizer,
                              device: torch.device) -> tuple[torch.Tensor,
                                                              torch.Tensor]:
    prediction_rows = []
    quantile_rows = []
    model.eval()
    dtype = next(model.parameters()).dtype
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2, 1)
    for history, _unused_target in loader:
        history = history.to(device, non_blocking=True)
        history_norm = (
            history - normalizer.mean_input) / normalizer.std_input
        prediction_norm, _members, quantiles, _lower, _upper = (
            model.forward_quantiles(history_norm.to(dtype=dtype)))
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        outcome = prediction.unsqueeze(-1) + quantiles.float() * scale
        prediction_rows.append(prediction.cpu().to(torch.float16))
        quantile_rows.append(outcome.cpu().to(torch.float16))
    return torch.cat(prediction_rows), torch.cat(quantile_rows)


def make_loader(dataset, indices: list[int], batch_size: int,
                num_workers: int, shuffle: bool = False,
                seed: int = 0) -> DataLoader:
    return uq_train.make_loader(
        dataset, indices, batch_size, shuffle,
        argparse.Namespace(num_workers=num_workers), seed)


def reconstruct_train_anchors(
        args: argparse.Namespace, dataset, sample_folds: list[int],
        fold_manifest: dict, normalizer, device: torch.device,
        train_indices: list[int]) -> tuple[torch.Tensor, dict]:
    """Reconstruct one fused FP16 bank in RAM; never persist A/B banks."""
    contracts = {
        "a": validate_oof_quantile_contract(
            args.point_checkpoint_a, args.oof_dir_a, fold_manifest, "a"),
        "b": validate_oof_quantile_contract(
            args.point_checkpoint_b, args.oof_dir_b, fold_manifest, "b"),
    }
    source_files = {
        "point_checkpoint_a": s1.file_identity(args.point_checkpoint_a),
        "point_checkpoint_b": s1.file_identity(args.point_checkpoint_b),
        "mean_expert_checkpoint": s1.file_identity(
            args.mean_expert_checkpoint),
    }
    for name, expected in FORMAL_SOURCE_SHA256.items():
        if source_files[name]["sha256"] != expected:
            raise ValueError(f"formal P0 source SHA-256 differs at {name}")
    payload_a = torch.load(
        args.point_checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.point_checkpoint_b, map_location="cpu", weights_only=False)
    row_of_index = {index: row for row, index in enumerate(train_indices)}
    anchor = torch.empty(
        (len(train_indices), 20, 32, 64, 2, 7), dtype=torch.float16)

    # C is small enough to collect once.  Only folds 0--3 are requested.
    model_c, metadata_c = point_gate.load_model(
        args.mean_expert_checkpoint, normalizer, device)
    c_loader = make_loader(
        dataset, train_indices, args.reconstruction_batch_size,
        args.num_workers)
    prediction_c = collect_point_predictions(
        model_c, c_loader, normalizer, device)
    del model_c
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fold_rows = []
    for fold in range(HELD_FOLD):
        indices = [index for index in train_indices
                   if sample_folds[index] == fold]
        loader = make_loader(
            dataset, indices, args.reconstruction_batch_size,
            args.num_workers)
        model_a, metadata_a = s1.load_fold_model(
            payload_a, args.oof_dir_a / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_a, normalizer, device)
        prediction_a, quantiles_a = collect_outcome_quantiles(
            model_a, loader, normalizer, device)
        del model_a
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model_b, metadata_b = s1.load_fold_model(
            payload_b, args.oof_dir_b / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_b, normalizer, device)
        prediction_b, quantiles_b = collect_outcome_quantiles(
            model_b, loader, normalizer, device)
        del model_b
        c_rows = torch.stack([
            prediction_c[row_of_index[index]] for index in indices])
        fused = compose_p0_absolute_quantiles(
            quantiles_a.float(), quantiles_b.float(),
            prediction_a.float(), prediction_b.float(), c_rows.float())
        output_rows = torch.as_tensor(
            [row_of_index[index] for index in indices], dtype=torch.long)
        anchor[output_rows] = fused.to(torch.float16)
        fold_rows.append({
            "fold": fold,
            "samples": len(indices),
            "model_a": metadata_a,
            "model_b": metadata_b,
        })
        print(json.dumps({
            "event": "p0_7q_anchor_fold", "fold": fold,
            "samples": len(indices),
        }), flush=True)
        del prediction_a, prediction_b, quantiles_a, quantiles_b
        del c_rows, fused, output_rows
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del payload_a, payload_b, prediction_c
    if not torch.isfinite(anchor).all():
        raise FloatingPointError("reconstructed P0 anchor has non-finite values")
    if not torch.all(anchor[..., 1:] > anchor[..., :-1]):
        raise FloatingPointError("reconstructed P0 anchor is not monotone")
    metadata = {
        "format": ANCHOR_FORMAT,
        "evidence": (
            "P0-source/weight anchored full outcome quantiles; not exact "
            "post-selection P0 endpoints"),
        "quantile_levels": list(QUANTILE_LEVELS),
        "weights": {"b": WEIGHT_B, "c": WEIGHT_C},
        "partition": "point train_starts pseudo-label pretraining only",
        "fold_safety": (
            "the selected OOF UQ head excluded this condition's tail truth; "
            "the frozen A/B point bases did train on point-split windows"),
        "train_folds": [0, 1, 2, 3],
        "held_fold_reconstructed": False,
        "samples": len(anchor),
        "stored_bytes": anchor.numel() * anchor.element_size(),
        "source_files": source_files,
        "oof_contracts": contracts,
        "mean_expert": metadata_c,
        "folds": fold_rows,
    }
    return anchor, metadata


class AnchoredPointDataset(Dataset):
    def __init__(self, windows, train_indices: list[int],
                 anchors: torch.Tensor):
        if len(train_indices) != len(anchors):
            raise ValueError("anchor rows do not align with training indices")
        self.windows = windows
        self.indices = train_indices
        self.anchors = anchors

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        history, target = self.windows[self.indices[row]]
        return history, target, self.anchors[row]


def absolute_outcome_quantiles(
        prediction_norm: torch.Tensor, quantiles_norm: torch.Tensor,
        normalizer) -> torch.Tensor:
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    point = common.to_physical_target(
        prediction_norm.float(), normalizer)[..., :2]
    return point.unsqueeze(-1) + quantiles_norm.float() * scale.unsqueeze(-1)


def anchor_distillation_objective(
        prediction_norm: torch.Tensor, quantiles_norm: torch.Tensor,
        anchor_absolute: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    quantiles_absolute = absolute_outcome_quantiles(
        prediction_norm, quantiles_norm, normalizer)
    reference = quantiles_absolute.new_tensor(REFERENCE_WIDTH).view(
        1, 1, 1, 1, 2, 1)
    anchor_error = (quantiles_absolute - anchor_absolute.float()) / reference
    anchor_loss = F.smooth_l1_loss(
        anchor_error, torch.zeros_like(anchor_error))
    return anchor_loss, {"anchor": anchor_loss.detach()}


def native_truth_objective(
        args: argparse.Namespace, prediction_norm: torch.Tensor,
        quantiles_norm: torch.Tensor, lower_norm: torch.Tensor,
        upper_norm: torch.Tensor, truth: torch.Tensor,
        normalizer) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    point = common.to_physical_target(
        prediction_norm.float(), normalizer)[..., :2]
    reference = point.new_tensor(REFERENCE_WIDTH).view(
        1, 1, 1, 1, 2, 1)
    truth_uv = truth[..., :2].float()
    residual = truth_uv - point.detach()
    scored = truth_uv != 0.0
    physical_quantiles = quantiles_norm.float() * scale.unsqueeze(-1)
    error = residual.unsqueeze(-1) - physical_quantiles
    levels = error.new_tensor(QUANTILE_LEVELS).view(1, 1, 1, 1, 1, -1)
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    pinball = (pinball / reference)[scored.unsqueeze(-1).expand_as(
        pinball)].mean()

    lower = lower_norm.float() * scale
    upper = upper_norm.float() * scale
    tau = args.soft_tau_fraction * reference.squeeze(-1)
    soft_inside = (
        torch.sigmoid((residual + lower) / tau)
        * torch.sigmoid((upper - residual) / tau))
    factor = common.sample_sps_factor(
        point.detach(), truth_uv).detach().view(-1, 1, 1, 1, 1)
    soft_sps = (
        factor * soft_inside
        * torch.exp(-(lower + upper) / common.SPS_SIGMA))[scored].mean()
    soft_coverage = soft_inside[scored].mean()
    coverage_penalty = F.relu(
        args.coverage_target - soft_coverage).square()
    loss = (
        args.truth_pinball_weight * pinball
        - args.soft_sps_weight * soft_sps
        + args.coverage_weight * coverage_penalty)
    return loss, {
        "pinball": pinball.detach(),
        "soft_sps": soft_sps.detach(),
        "soft_coverage": soft_coverage.detach(),
        "mean_width": (lower + upper)[scored].mean().detach(),
    }


def new_accumulator() -> dict[str, float | int]:
    return {"numerator": 0.0, "scored": 0, "covered": 0, "width": 0.0}


def update_accumulator(accumulator: dict, prediction: torch.Tensor,
                       lower: torch.Tensor, upper: torch.Tensor,
                       truth: torch.Tensor) -> None:
    scored = truth != 0.0
    residual = truth - prediction
    inside = (residual >= -lower) & (residual <= upper)
    factor = common.sample_sps_factor(prediction, truth).view(
        -1, 1, 1, 1, 1)
    value = factor * inside * torch.exp(
        -(lower + upper) / common.SPS_SIGMA)
    accumulator["numerator"] += float(
        value[scored].sum(dtype=torch.float64))
    accumulator["scored"] += int(scored.sum())
    accumulator["covered"] += int((inside & scored).sum())
    accumulator["width"] += float(
        (lower + upper)[scored].sum(dtype=torch.float64))


def finalize_accumulator(accumulator: dict) -> dict:
    count = int(accumulator["scored"])
    if not count:
        raise RuntimeError("terminal evaluation had no scored elements")
    return {
        "sps_score": 100.0 * float(accumulator["numerator"]) / count,
        "coverage": int(accumulator["covered"]) / count,
        "mean_width": float(accumulator["width"]) / count,
        "scored_elements": count,
    }


@torch.inference_mode()
def evaluate_native_teacher(model, loader, normalizer,
                            device: torch.device) -> dict:
    model.eval()
    accumulator = new_accumulator()
    point_digest = hashlib.sha256()
    dtype = next(model.parameters()).dtype
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for history, truth in loader:
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True).float()[..., :2]
        history_norm = (
            history - normalizer.mean_input) / normalizer.std_input
        prediction_norm, lower_norm, upper_norm = model.forward_distribution(
            history_norm.to(dtype=dtype))
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        lower = lower_norm.float() * scale
        upper = upper_norm.float() * scale
        update_accumulator(accumulator, prediction, lower, upper, truth)
        point_digest.update(
            prediction.cpu().contiguous().numpy().tobytes(order="C"))
    return {
        **finalize_accumulator(accumulator),
        "point_output_sha256": point_digest.hexdigest(),
    }


def teacher_gate(args: argparse.Namespace, metrics: dict,
                 point_state_equal: bool, completed_updates: int) -> dict:
    coverage_floor = FOLD04_REFERENCE["coverage"] - MAX_COVERAGE_DROP
    width_ceiling = FOLD04_REFERENCE["mean_width"] * MAX_WIDTH_RATIO
    checks = {
        "formal_configuration": (
            args.updates == 4000
            and args.anchor_only_updates == 1000
            and args.batch_size == 8
            and args.eval_batch_size == 16
            and args.reconstruction_batch_size == 16
            and args.lr == 1e-4
            and args.weight_decay == 1e-5
            and args.grad_clip == 1.0
            and args.truth_pinball_weight == 0.25
            and args.soft_sps_weight == 1.0
            and args.coverage_weight == 0.25
            and args.coverage_target == FOLD04_REFERENCE["coverage"]
            and args.soft_tau_fraction == 0.10
            and args.seed == 0
            and args.num_workers == 0
            and args.device == "cuda:0"),
        "completed_fixed_4000_updates": completed_updates == 4000,
        "sps_ge_53_30": metrics["sps_score"] >= TEACHER_MIN_SPS,
        "coverage_ge_reference_minus_0_005": (
            metrics["coverage"] >= coverage_floor),
        "width_le_1_05x_reference": (
            metrics["mean_width"] <= width_ceiling),
        "point_state_bitwise_equal": point_state_equal,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_sps_score": TEACHER_MIN_SPS,
            "min_coverage": coverage_floor,
            "max_mean_width": width_ceiling,
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal native teacher training requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    point_file = s1.file_identity(args.point_checkpoint)
    if (point_file["sha256"] != FORMAL_POINT_SHA256
            or point_file["sha256"] != args.expected_point_sha256):
        raise ValueError("point checkpoint SHA-256 differs from contract")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    process_started = time.perf_counter()

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    if (len(point_windows) != EXPECTED_POINT_WINDOWS
            or len(tail_windows) != EXPECTED_TAIL_WINDOWS):
        raise RuntimeError("frozen 2576/725 split changed")
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    if runtime_manifest != frozen_manifest:
        raise RuntimeError("runtime split manifest differs from frozen manifest")
    tail_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    point_folds = partition_fold_labels(
        runtime_manifest, fold_manifest, "train_starts")
    if len(point_folds) != len(point_windows):
        raise RuntimeError("point fold labels do not align with point windows")
    point_train_indices, point_held_indices = fold_partition_indices(
        point_folds)
    tail_train_indices, tail_held_indices = fold_partition_indices(tail_folds)
    if (len(point_train_indices) != EXPECTED_TRAIN_POINT
            or len(point_held_indices) != EXPECTED_HELD_POINT):
        raise RuntimeError("fold04 point counts differ from 2032/544")
    if (len(tail_train_indices) != EXPECTED_TRAIN_TAIL
            or len(tail_held_indices) != EXPECTED_HELD_TAIL):
        raise RuntimeError("fold04 tail counts differ from 572/153")
    point_dataset = common.repro.WindowDataset(trajectories, point_windows)
    tail_dataset = common.repro.WindowDataset(trajectories, tail_windows)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    anchors, anchor_metadata = reconstruct_train_anchors(
        args, point_dataset, point_folds, fold_manifest,
        normalizer, device, point_train_indices)
    anchor_dataset = AnchoredPointDataset(
        point_dataset, point_train_indices, anchors)
    anchor_loader = DataLoader(
        anchor_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    truth_loader = make_loader(
        tail_dataset, tail_train_indices, args.batch_size,
        args.num_workers, shuffle=True, seed=args.seed + 1)
    held_loader = make_loader(
        tail_dataset, tail_held_indices, args.eval_batch_size,
        args.num_workers)

    payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    if tuple(payload.get("model_config", {}).get(
            "quantile_levels", ())) != QUANTILE_LEVELS:
        raise ValueError("point checkpoint does not expose the required 7q head")
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=True)
    point_state_before = point_state_identity(model)
    parameters = uq_train.trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    anchor_iterator = common.repro.cycling(anchor_loader)
    truth_iterator = common.repro.cycling(truth_loader)
    history_rows = []
    sums: dict[str, float] = {}
    count = 0
    completed_updates = 0
    training_started = time.perf_counter()
    for iteration in range(1, args.updates + 1):
        if time.perf_counter() - training_started > args.time_budget_seconds:
            break
        stage = training_stage(
            iteration, args.anchor_only_updates, args.updates)
        if stage == "point2032_p0_7q_pretrain":
            history, _unused_truth, anchor = next(anchor_iterator)
            history = history.to(device, non_blocking=True)
            anchor = anchor.to(device, non_blocking=True).float()
            history_norm = (
                history - normalizer.mean_input) / normalizer.std_input
            truth = None
        else:
            stage = "tail572_truth_refine"
            history, truth = next(truth_iterator)
            history = history.to(device, non_blocking=True)
            truth = truth.to(device, non_blocking=True).float()
            history_norm, _ = normalizer.preprocess(history, truth)
            anchor = None
        optimizer.zero_grad(set_to_none=True)
        (prediction, quantiles, lower, upper, _features,
         _coverage_logits) = (lambda values: (
             values[0], values[2], values[3], values[4], values[5], values[6]
         ))(model.forward_uq_training(history_norm))
        if stage == "point2032_p0_7q_pretrain":
            loss, components = anchor_distillation_objective(
                prediction, quantiles, anchor, normalizer)
        else:
            loss, components = native_truth_objective(
                args, prediction, quantiles, lower, upper,
                truth, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed_updates = iteration
        if iteration == args.anchor_only_updates:
            # The 2.17-GiB point pseudo-label bank has served its only purpose.
            # Drop every reference before beginning tail-truth refinement.
            del anchor_iterator, anchor_loader, anchor_dataset, anchors
            gc.collect()
        for key, value in {"loss": loss.detach(), **components}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % 250 == 0 or iteration == args.updates:
            row = {
                "iteration": iteration,
                "stage": stage,
                "elapsed_seconds": time.perf_counter() - training_started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
            }
            history_rows.append(row)
            print(json.dumps({"event": "native_teacher_train", **row}),
                  flush=True)
            sums.clear()
            count = 0
    point_state_after = point_state_identity(model)
    point_state_equal = point_state_before == point_state_after
    if not point_state_equal:
        raise RuntimeError("frozen point/backbone state changed during training")
    uq_state = uq_train.uq_state_dict(model)
    del model, parameters, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    # Gate the deployable FP16 state, not the FP32 optimization copy.
    deploy_model, deploy_metadata = load_forecaster(
        args.point_checkpoint, normalizer, device)
    uq_train.load_uq_state(deploy_model, uq_state)
    evaluation = evaluate_native_teacher(
        deploy_model, held_loader, normalizer, device)
    gate = teacher_gate(
        args, evaluation, point_state_equal, completed_updates)
    status = "PASS_NATIVE_TEACHER_FOLD04" if gate["passed"] else (
        "STOP_NATIVE_TEACHER_FOLD04")
    elapsed = time.perf_counter() - process_started
    checkpoint_path = args.output_dir / "native_teacher.pth"
    torch.save({
        "architecture": "p1_direct_native_7q_teacher",
        "model_config": payload["model_config"],
        "uq_state_dict": uq_state,
        "point_checkpoint": point_file,
        "quantile_levels": QUANTILE_LEVELS,
        "held_fold": HELD_FOLD,
        "updates": completed_updates,
        "evaluation": evaluation,
        "gate": gate,
    }, checkpoint_path)
    summary = {
        "status": status,
        "evidence_class": (
            "fold04-held native 7q teacher; folds0-3 tail truth+P0 anchor; "
            "one terminal held evaluation; no official evidence"),
        "point_checkpoint": point_file,
        "deploy_model": deploy_metadata,
        "counts": {
            "point_windows": len(point_windows),
            "point_pseudolabel_train_folds0_to3": len(point_train_indices),
            "point_held_fold04_unused": len(point_held_indices),
            "tail_windows": len(tail_windows),
            "tail_truth_train_folds0_to3": len(tail_train_indices),
            "tail_held_fold04": len(tail_held_indices),
        },
        "held_fold_evaluations_during_training": 0,
        "completed_updates": completed_updates,
        "wall_seconds": elapsed,
        "fold_manifest": fold_manifest,
        "anchor": anchor_metadata,
        "point_state": {
            "before": point_state_before,
            "after": point_state_after,
            "bitwise_equal": point_state_equal,
        },
        "evaluation": evaluation,
        "reference": FOLD04_REFERENCE,
        "gate": gate,
        "training_history": history_rows,
        "checkpoint": s1.file_identity(checkpoint_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "native_teacher_complete", "status": status,
        "evaluation": evaluation, "gate": gate,
        "checkpoint": summary["checkpoint"],
    }), flush=True)


if __name__ == "__main__":
    main()
