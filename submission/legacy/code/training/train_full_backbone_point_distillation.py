#!/usr/bin/env python3
"""One bounded full-backbone distillation run for the C128 point student."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from evaluate_track1_submission_archive import (  # noqa: E402
    load_submission,
    normalized_result,
    safe_extract,
)
from train_distilled_flow_corrector import (  # noqa: E402
    profile_distill_loss,
    seed_everything,
)
from train_flow_backbone import construct_model  # noqa: E402


POINT_GATE = {
    "rel_l2_score": 95.9278,
    "tke_rel_l2_score": 80.9125,
    "mvpe_rel_l2_score": 96.6260,
    "point_score": 91.5629,
}

FORMAL_TEACHER_SHA256 = (
    "d832edc5d4db0827d5111a08fa14740431651b3670296816180a6023789ab5cf")
FORMAL_REAL_STATS_SHA256 = (
    "bae5235f73431ac7d76e6bdcea480b8ca9fda0ac01e57ab26af66bad20dc3b40")
FORMAL_SPLIT_MANIFEST_SHA256 = (
    "b83811f908c8d55a9bd255d158f5f2648b3248a4271bca404c3085f36570dbbd")
FORMAL_INITIAL_METRICS = {
    "rel_l2_score": 95.8901,
    "tke_rel_l2_score": 80.8787,
    "mvpe_rel_l2_score": 96.5725,
    "point_score": 91.5232,
}
FORMAL_INITIAL_TOLERANCE = 0.005
FORMAL_COUNTS = {
    "train_windows": 2576,
    "val_windows": 725,
    "trajectories": 82,
}
STEP400_SIGNAL_GATE = {
    "point_delta": 0.01,
    "tke_delta": -0.05,
}


def load_official_scoring():
    path = REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
    spec = importlib.util.spec_from_file_location(
        "realpde_track1_v9_full_distill_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official scoring from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = load_official_scoring()


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        history, target = self.base[index]
        return index, history, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-student", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=400)
    parser.add_argument("--encoder-lr", type=float, default=2e-6)
    parser.add_argument("--decoder-lr", type=float, default=1e-5)
    parser.add_argument("--point-head-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--time-budget-seconds", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--formal", action="store_true",
        help="enforce the frozen 2576/725 H96 experiment contract")
    parser.add_argument(
        "--expected-start-sha256",
        help="required with --formal; locks the exact H96 start checkpoint")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.encoder_lr, args.decoder_lr,
        args.point_head_lr, args.time_budget_seconds,
        args.window_stride)
    if any(value <= 0 for value in positive):
        raise ValueError("counts, learning rates, and budgets must be positive")
    if args.updates % args.eval_interval:
        raise ValueError("updates must be divisible by eval_interval")
    if args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("weight decay and grad clip must be non-negative")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val_fraction must lie in (0, 0.5)")


def sha256(path: Path) -> str:
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


def tensor_is_finite(value: torch.Tensor, chunk_size: int = 32) -> bool:
    return all(
        bool(torch.isfinite(value[start:start + chunk_size]).all())
        for start in range(0, value.shape[0], chunk_size))


def prepare_data(args: argparse.Namespace, device: torch.device):
    trajectories, metadata = common.repro.load_trajectories(
        args.real_dir, args.max_files)
    train_windows, val_windows, manifest = common.repro.make_split(
        metadata, args.window_stride, args.val_fraction)
    train_base = common.repro.WindowDataset(trajectories, train_windows)
    val_dataset = common.repro.WindowDataset(trajectories, val_windows)
    train_loader = DataLoader(
        IndexedDataset(train_base), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    cache_loader = DataLoader(
        train_base, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (args.output_dir / "split_manifest.json").write_text(
        manifest_text, encoding="utf-8")
    split_manifest_sha256 = canonical_sha256(manifest)
    train_keys = [
        (metadata[window.trajectory]["file"], int(window.start))
        for window in train_windows]
    example_target = train_base[0][1]
    expected_teacher_shape = (
        len(train_windows), *example_target.shape[:-1], 2)
    val_aoa = np.asarray([
        float(metadata[window.trajectory]["aoa"])
        for window in val_windows], dtype=np.float32)
    return (
        train_loader, cache_loader, val_loader, normalizer,
        train_keys, val_aoa, len(trajectories), split_manifest_sha256,
        sha256(args.real_stats), expected_teacher_shape)


def load_student(path: Path, normalizer, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    if config.get("architecture") != "ensemble_unet":
        raise ValueError("start student must be an ensemble_unet checkpoint")
    if int(config.get("base", -1)) != 128:
        raise ValueError("bounded run requires the C128 student")
    if int(config.get("uq_hidden_channels", -1)) != 96:
        raise ValueError("bounded run requires the official H96 student")
    if "point_head_distillation" not in payload:
        raise ValueError("start checkpoint is not the point-distilled baseline")
    model = construct_model("ensemble_unet", config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    common.set_model_stats(model, normalizer)
    return model, payload


def optimizer_groups(model) -> list[dict]:
    model.requires_grad_(False)
    encoder_modules = [model.stem, model.encoder]
    decoder_modules = [
        model.bottleneck, model.up, model.decoder,
        model.regime_film, model.decoder_regime_film,
        model.member_feature_adapters, model.mean_profile_refiner,
    ]
    point_modules = [model.member_point_heads]

    def parameters(modules) -> list[torch.nn.Parameter]:
        result = []
        for module in modules:
            if module is None:
                continue
            module.requires_grad_(True)
            result.extend(module.parameters())
        return list(dict.fromkeys(result))

    encoder = parameters(encoder_modules)
    decoder = parameters(decoder_modules)
    point = parameters(point_modules)
    all_parameters = encoder + decoder + point
    if len({id(value) for value in all_parameters}) != len(all_parameters):
        raise RuntimeError("optimizer parameter groups overlap")
    if not all_parameters:
        raise RuntimeError("full-backbone distillation has no trainable parameters")
    return [
        {"params": encoder, "lr": 2e-6, "name": "encoder"},
        {"params": decoder, "lr": 1e-5, "name": "decoder"},
        {"params": point, "lr": 3e-5, "name": "point_heads"},
    ]


def build_or_load_teacher_cache(
        args: argparse.Namespace, cache_loader: DataLoader,
        normalizer, train_keys: list[tuple[str, int]],
        split_manifest_sha256: str, real_stats_sha256: str,
        expected_shape: tuple[int, ...],
        ) -> torch.Tensor:
    archive_hash = sha256(args.teacher_archive)
    if args.teacher_cache.is_file():
        payload = torch.load(
            args.teacher_cache, map_location="cpu", weights_only=False)
        if payload.get("archive_sha256") != archive_hash:
            raise ValueError("teacher cache archive hash mismatch")
        if payload.get("split_manifest_sha256") != split_manifest_sha256:
            raise ValueError("teacher cache split manifest hash mismatch")
        if payload.get("real_stats_sha256") != real_stats_sha256:
            raise ValueError("teacher cache normalizer hash mismatch")
        if payload.get("train_keys") != train_keys:
            raise ValueError("teacher cache window order mismatch")
        values = payload.get("prediction_uv_normalized")
        if not isinstance(values, torch.Tensor):
            raise TypeError("teacher cache prediction must be a tensor")
        if tuple(values.shape) != expected_shape:
            raise ValueError("teacher cache shape mismatch")
        if not torch.is_floating_point(values):
            raise TypeError("teacher cache prediction must be floating point")
        if not tensor_is_finite(values):
            raise ValueError("teacher cache prediction contains non-finite values")
        return values

    args.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    args.output_dir.joinpath("tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="teacher-cache-", dir=args.output_dir / "tmp") as temporary:
        package_dir = Path(temporary)
        safe_extract(args.teacher_archive, package_dir)
        submission = load_submission(package_dir)
        for history, _target in cache_loader:
            result = submission.predict(history.numpy(), metadata={})
            prediction, _lower, _upper = normalized_result(result)
            predictions.append(np.asarray(prediction, dtype=np.float32)[..., :2])
        del submission
        if str(package_dir) in sys.path:
            sys.path.remove(str(package_dir))
    for name in ("models.ensemble_unet2d", "models.unet2d", "models"):
        sys.modules.pop(name, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    physical = torch.from_numpy(np.concatenate(predictions))
    mean = normalizer.mean_target[:2].detach().cpu()
    std = normalizer.std_target[:2].detach().cpu()
    normalized = ((physical - mean) / std).to(torch.float16)
    if tuple(normalized.shape) != expected_shape:
        raise ValueError(
            f"new teacher cache shape {tuple(normalized.shape)} != {expected_shape}")
    if not tensor_is_finite(normalized):
        raise ValueError("new teacher cache contains non-finite values")
    torch.save({
        "method": "official_mean_expert_point_teacher_cache",
        "archive": str(args.teacher_archive.resolve()),
        "archive_sha256": archive_hash,
        "split_manifest_sha256": split_manifest_sha256,
        "real_stats_sha256": real_stats_sha256,
        "train_keys": train_keys,
        "prediction_shape": list(expected_shape),
        "prediction_uv_normalized": normalized,
    }, args.teacher_cache)
    return normalized


def metric_bundle(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    channels = OFFICIAL.measured_channels(target)
    result = {
        "rel_l2_score": OFFICIAL.score_error(float(np.mean(
            OFFICIAL.rel_l2_per_sample(prediction, target, channels)))),
        "tke_rel_l2_score": OFFICIAL.score_error(float(np.mean(
            OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels)))),
        "mvpe_rel_l2_score": OFFICIAL.score_error(float(np.mean(
            OFFICIAL.mvpe_rel_l2_per_sample(prediction, target)))),
    }
    result["point_score"] = (
        0.5 * result["rel_l2_score"]
        + 0.3 * result["tke_rel_l2_score"]
        + 0.2 * result["mvpe_rel_l2_score"])
    return result


@torch.inference_mode()
def evaluate_point(model, loader, normalizer, device, val_aoa) -> dict:
    model.eval()
    predictions = []
    targets = []
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        prediction_norm = model(history_norm)
        predictions.append(
            normalizer.postprocess_prediction(prediction_norm).cpu().numpy())
        targets.append(
            normalizer.postprocess_target(target_norm).cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    if prediction.shape[0] != val_aoa.shape[0]:
        raise RuntimeError("AoA labels do not align with validation predictions")
    by_aoa = {}
    for aoa in sorted(np.unique(val_aoa)):
        selected = val_aoa == aoa
        by_aoa[f"{float(aoa):g}"] = metric_bundle(
            prediction[selected], target[selected])
    return {**metric_bundle(prediction, target), "by_aoa": by_aoa}


def tke_map_distill_loss(student, teacher, normalizer) -> torch.Tensor:
    mean = normalizer.mean_target[:2]
    std = normalizer.std_target[:2]
    student_physical = student[..., :2] * std + mean
    teacher_physical = teacher * std + mean

    def kinetic_energy(value):
        fluctuation = value - value.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)

    student_tke = kinetic_energy(student_physical)
    teacher_tke = kinetic_energy(teacher_physical)
    batch = student.shape[0]
    relative = torch.linalg.vector_norm(
        (student_tke - teacher_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        teacher_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)
    return relative.mean()


def distill_loss(student, teacher_uv, truth, normalizer):
    student_uv = student[..., :2]
    student_mean = student_uv.mean(dim=1, keepdim=True)
    teacher_mean = teacher_uv.mean(dim=1, keepdim=True)
    student_fluctuation = student_uv - student_mean
    teacher_fluctuation = teacher_uv - teacher_mean
    mean_loss = F.smooth_l1_loss(student_mean / 0.025, teacher_mean / 0.025)
    fluctuation_loss = F.smooth_l1_loss(
        student_fluctuation / 0.020,
        teacher_fluctuation / 0.020)
    profile_loss = profile_distill_loss(
        student_mean, teacher_mean, 0.025)
    tke_loss = tke_map_distill_loss(
        student_uv, teacher_uv, normalizer)
    truth_loss = common.accuracy_objective(
        student, truth, normalizer, (5.0, 3.0, 2.0))[0]
    loss = (
        2.0 * mean_loss
        + fluctuation_loss
        + 2.0 * profile_loss
        + 0.25 * tke_loss
        + 0.05 * truth_loss)
    return loss, {
        "mean_distill": mean_loss.detach(),
        "fluctuation_distill": fluctuation_loss.detach(),
        "profile_distill": profile_loss.detach(),
        "tke_map_distill": tke_loss.detach(),
        "truth_accuracy": truth_loss.detach(),
    }


def save_checkpoint(path, model, source_payload, args, iteration, metrics, status):
    payload = {
        "phase": "real",
        "iteration": iteration,
        "model_config": source_payload["model_config"],
        "normalizer": source_payload.get("normalizer", {}),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()},
        "metrics": metrics,
        "full_backbone_point_distillation": {
            "status": status,
            "teacher_archive": str(args.teacher_archive),
            "teacher_archive_sha256": sha256(args.teacher_archive),
            "metrics": metrics,
            "warning": "frozen UQ must be retrained after backbone changes",
            "uq_state_valid": False,
        },
    }
    if "point_head_distillation" in source_payload:
        payload["point_head_distillation"] = source_payload[
            "point_head_distillation"]
    torch.save(payload, path)


def aoa_slice_gate(initial: dict, final: dict) -> dict:
    groups = sorted(set(initial["by_aoa"]) & set(final["by_aoa"]))
    rows = {}
    positive = 0
    tke_safe = True
    for group in groups:
        point_delta = (
            final["by_aoa"][group]["point_score"]
            - initial["by_aoa"][group]["point_score"])
        tke_delta = (
            final["by_aoa"][group]["tke_rel_l2_score"]
            - initial["by_aoa"][group]["tke_rel_l2_score"])
        rows[group] = {"point_delta": point_delta, "tke_delta": tke_delta}
        positive += int(point_delta >= 0.0)
        tke_safe = tke_safe and tke_delta >= -0.10
    return {
        "kind": "same-model validation slices; not leave-one-AoA crossfit",
        "groups": rows,
        "positive_point_groups": positive,
        "required_positive_groups": 4,
        "tke_safe": tke_safe,
        "pass": len(groups) == 5 and positive >= 4 and tke_safe,
    }


def formal_contract_errors(
        args: argparse.Namespace, initial: dict,
        start_sha256: str, teacher_sha256: str,
        split_manifest_sha256: str, real_stats_sha256: str,
        train_windows: int, val_windows: int, trajectories: int,
        ) -> list[str]:
    if not args.formal:
        return ["--formal was not requested"]
    errors = []
    expected_args = {
        "updates": 800,
        "eval_interval": 400,
        "max_files": 0,
        "seed": 0,
        "window_stride": 20,
        "val_fraction": 0.2,
        "batch_size": 8,
        "eval_batch_size": 16,
        "encoder_lr": 2e-6,
        "decoder_lr": 1e-5,
        "point_head_lr": 3e-5,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
    }
    for key, expected in expected_args.items():
        actual = getattr(args, key)
        if isinstance(expected, float):
            matches = bool(np.isclose(actual, expected, rtol=0.0, atol=1e-12))
        else:
            matches = actual == expected
        if not matches:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    actual_counts = {
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
    }
    for key, expected in FORMAL_COUNTS.items():
        if actual_counts[key] != expected:
            errors.append(
                f"{key}={actual_counts[key]}, expected {expected}")
    if teacher_sha256 != FORMAL_TEACHER_SHA256:
        errors.append(
            f"teacher sha256 {teacher_sha256} != {FORMAL_TEACHER_SHA256}")
    if real_stats_sha256 != FORMAL_REAL_STATS_SHA256:
        errors.append(
            f"real stats sha256 {real_stats_sha256} "
            f"!= {FORMAL_REAL_STATS_SHA256}")
    if split_manifest_sha256 != FORMAL_SPLIT_MANIFEST_SHA256:
        errors.append(
            f"split manifest sha256 {split_manifest_sha256} "
            f"!= {FORMAL_SPLIT_MANIFEST_SHA256}")
    if not args.expected_start_sha256:
        errors.append("--expected-start-sha256 is required with --formal")
    elif start_sha256.lower() != args.expected_start_sha256.lower():
        errors.append(
            f"start sha256 {start_sha256} != {args.expected_start_sha256}")
    for key, expected in FORMAL_INITIAL_METRICS.items():
        actual = float(initial[key])
        if abs(actual - expected) > FORMAL_INITIAL_TOLERANCE:
            errors.append(
                f"initial {key}={actual:.8f}, expected {expected:.4f} "
                f"within {FORMAL_INITIAL_TOLERANCE}")
    return errors


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-backbone distillation requires CUDA")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (train_loader, cache_loader, val_loader, normalizer,
     train_keys, val_aoa, trajectories, split_manifest_sha256,
     real_stats_sha256, expected_teacher_shape) = prepare_data(args, device)
    start_sha256 = sha256(args.start_student)
    teacher_sha256 = sha256(args.teacher_archive)
    model, source_payload = load_student(
        args.start_student, normalizer, device)
    initial = evaluate_point(
        model, val_loader, normalizer, device, val_aoa)
    print(json.dumps({"event": "initial", **initial}), flush=True)
    contract_errors = formal_contract_errors(
        args, initial, start_sha256, teacher_sha256,
        split_manifest_sha256, real_stats_sha256,
        len(train_keys), int(val_aoa.shape[0]), trajectories)
    formal_contract_pass = bool(args.formal and not contract_errors)
    if args.formal and contract_errors:
        rejection = {
            "status": "REJECT_FORMAL_CONTRACT",
            "errors": contract_errors,
            "start_student_sha256": start_sha256,
            "teacher_archive_sha256": teacher_sha256,
            "initial_metrics": initial,
        }
        rejection_path = args.output_dir / "contract_rejection.json"
        rejection_path.write_text(
            json.dumps(rejection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        raise RuntimeError(json.dumps(rejection, ensure_ascii=False))

    teacher_cache = build_or_load_teacher_cache(
        args, cache_loader, normalizer, train_keys,
        split_manifest_sha256, real_stats_sha256,
        expected_teacher_shape)
    groups = optimizer_groups(model)
    for group, lr in zip(
            groups, (args.encoder_lr, args.decoder_lr, args.point_head_lr)):
        group["lr"] = lr
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    history = []
    iterator = common.repro.cycling(train_loader)
    sums: dict[str, float] = {}
    count = 0
    status = "RUNNING"
    started = time.time()
    completed_iteration = 0

    for iteration in range(1, args.updates + 1):
        if time.time() - started > args.time_budget_seconds:
            status = "STOP_TIME_BUDGET"
            break
        model.train()
        indices, history_batch, truth = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        teacher_uv = teacher_cache[indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        history_norm, truth_norm = normalizer.preprocess(history_batch, truth)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16):
            prediction = model(history_norm)
        loss, components = distill_loss(
            prediction.float(), teacher_uv, truth_norm, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in groups for parameter in group["params"]],
                args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed_iteration = iteration
        for key, value in {"loss": loss.detach(), **components}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

        if iteration % args.eval_interval == 0:
            metrics = evaluate_point(
                model, val_loader, normalizer, device, val_aoa)
            row = {
                "iteration": iteration,
                "elapsed_seconds": time.time() - started,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                **metrics,
            }
            history.append(row)
            print(json.dumps({"event": "eval", **row}), flush=True)
            save_checkpoint(
                args.output_dir / f"model_{iteration:05d}.pth",
                model, source_payload, args, iteration, metrics, "CHECKPOINT")
            sums.clear()
            count = 0
            if iteration == 400:
                point_delta = (
                    metrics["point_score"] - initial["point_score"])
                tke_delta = (
                    metrics["tke_rel_l2_score"]
                    - initial["tke_rel_l2_score"])
                if (point_delta < STEP400_SIGNAL_GATE["point_delta"]
                        or tke_delta < STEP400_SIGNAL_GATE["tke_delta"]):
                    status = "STOP_STEP400_NO_SIGNAL"
                    break

    final = evaluate_point(
        model, val_loader, normalizer, device, val_aoa)
    formal_gate_evaluated = bool(
        formal_contract_pass and completed_iteration == 800)
    pooled_pass = all(
        final[key] >= threshold for key, threshold in POINT_GATE.items())
    aoa_slice_result = aoa_slice_gate(initial, final)
    provisional_point_gate_pass = bool(
        formal_gate_evaluated and pooled_pass and aoa_slice_result["pass"])
    promotion_gate_pass = False
    if status == "RUNNING":
        if not args.formal:
            status = "NONFORMAL_COMPLETE_NO_GATE"
        elif provisional_point_gate_pass:
            status = "PROVISIONAL_GO_POINT_PENDING_LOAO"
        else:
            status = "STOP_FINAL_GATE"
    final_path = args.output_dir / f"final_{completed_iteration:05d}.pth"
    save_checkpoint(
        final_path, model, source_payload, args,
        completed_iteration, final, status)
    summary = {
        "method": "bounded_full_backbone_c128_point_distillation",
        "status": status,
        "formal_contract_pass": formal_contract_pass,
        "formal_gate_evaluated": formal_gate_evaluated,
        "provisional_point_gate_pass": provisional_point_gate_pass,
        "promotion_gate_pass": promotion_gate_pass,
        "required_next_gate": (
            "five-fold distillation-stage leave-one-AoA delta crossfit before "
            "any UQ retraining; start and teacher have seen all AoAs"),
        "point_gate": POINT_GATE,
        "pooled_gate_pass": pooled_pass,
        "aoa_slice_robustness": aoa_slice_result,
        "step400_signal_gate": STEP400_SIGNAL_GATE,
        "initial_metrics": initial,
        "final_metrics": final,
        "completed_iteration": completed_iteration,
        "train_windows": len(train_keys),
        "val_windows": int(val_aoa.shape[0]),
        "trajectories": trajectories,
        "start_student_sha256": start_sha256,
        "teacher_archive_sha256": teacher_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "real_stats_sha256": real_stats_sha256,
        "wall_seconds": time.time() - started,
        "history": history,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "loss_weights": {
            "temporal_mean": 2.0,
            "zero_mean_fluctuation": 1.0,
            "mean_profile": 2.0,
            "teacher_tke_map": 0.25,
            "real_truth_5_3_2": 0.05,
        },
        "final_checkpoint": str(final_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "status": status,
        "provisional_point_gate_pass": provisional_point_gate_pass,
        "promotion_gate_pass": promotion_gate_pass,
        "completed_iteration": completed_iteration,
        "summary": str(args.output_dir / "summary.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
