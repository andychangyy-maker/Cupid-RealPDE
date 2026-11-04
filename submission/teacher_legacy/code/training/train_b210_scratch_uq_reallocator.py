#!/usr/bin/env python3
"""Cross-fit a conditional uncertainty-budget reallocator on scratch B210.

The frozen point model and the five already-cross-fitted Native-UQ endpoints
define the baseline.  Each outer fold trains a small reallocator only on the
other four folds and evaluates one fixed terminal endpoint on its untouched
fold.  No held-fold checkpoint selection is allowed.  A plain dynamic arm is
compared with a Native-aware arm that also sees member disagreement and the
current asymmetric interval widths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ensemble_uq_oof as native_uq  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
import train_uq_budget_reallocator as reallocator  # noqa: E402
from train_flow_backbone import construct_model  # noqa: E402


EXPERIMENT = "b210_scratch_native_uq_reallocator"
SCHEMA = "b210_scratch_native_uq_reallocator_v1"
ARMS = ("dynamic", "native_aware")
NATIVE_AUXILIARY_CHANNELS = 160
EXPECTED_TAIL_WINDOWS = 716
EXCLUDED_FILE = "7575_0.h5"

FORMAL_IDENTITIES = {
    "point_checkpoint": (
        "d5d977a63bc665dfb4657ccd8a77f609dec1512a12d8083c758378edfa8e9103"),
    "real_stats": (
        "86b5feec17db74bca9dc909ed5baae93e4775c83f62db98bd46032a48f5bfc7f"),
    "native_summary": (
        "bb6af905a0314a9b9ff79bfef25bf2e7b580acb20da59d44ed66dd39e3df5a91"),
    "fold_00": (
        "1911f12fe5c5b78117a387979e6bd530dede1231bc50a8d1eb69bbc36cc7064d"),
    "fold_01": (
        "695df6f4b1e33ecfd0321ebe1a5bab19190986efa931da4e0796084f5ee07e76"),
    "fold_02": (
        "d233d72ee069a5643e629a9f5e03d8eb18ab1fa833fa5d27824c381dc5feeb24"),
    "fold_03": (
        "94efc6affeb631b1be76b03abd1a7d5b0b55ec2687db633133011f5f39a4d593"),
    "fold_04": (
        "d151082c9b041ccbc837816b2f37ab0a98dc0c845dd2b654e067ca1fb4b1821b"),
    "all_tail_refit": (
        "9012d1f163894e8f2f993e69a64426b6972fa608b434578cf109b06d6865273f"),
}

FORMAL_HYPERPARAMETERS = {
    "folds": 5,
    "fold_updates": 1500,
    "final_updates": 1500,
    "batch_size": 8,
    "eval_batch_size": 16,
    "eval_interval": 250,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "coverage_target": 0.86,
    "interval_score_weight": 0.02,
    "cvar_weight": 0.25,
    "cvar_fraction": 0.25,
    "hidden_channels": 48,
    "global_hidden": 160,
    "seed": 27,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--native-uq-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-updates", type=int, default=1500)
    parser.add_argument("--final-updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coverage-target", type=float, default=0.86)
    parser.add_argument("--interval-score-weight", type=float, default=0.02)
    parser.add_argument("--cvar-weight", type=float, default=0.25)
    parser.add_argument("--cvar-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--global-hidden", type=int, default=160)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wall-budget-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--acknowledge-protocol", required=True,
        choices=("B210_SCRATCH_NATIVE_UQ_REALLOCATOR",))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} identity changed: expected {expected}, got {actual}")


def validate_assets(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.folds != 5:
        raise ValueError("the formal protocol requires exactly five folds")
    changed = {
        key: (getattr(args, key), expected)
        for key, expected in FORMAL_HYPERPARAMETERS.items()
        if getattr(args, key) != expected
    }
    if changed:
        raise ValueError(f"formal hyperparameters changed: {changed}")
    positive = (
        args.fold_updates, args.final_updates, args.batch_size,
        args.eval_batch_size, args.eval_interval, args.lr,
        args.hidden_channels, args.global_hidden, args.wall_budget_seconds)
    if any(value <= 0 for value in positive):
        raise ValueError("training counts, dimensions and rates must be positive")
    finite_nonnegative = (
        args.weight_decay, args.grad_clip, args.interval_score_weight,
        args.cvar_weight)
    if any(not math.isfinite(value) or value < 0
           for value in finite_nonnegative):
        raise ValueError("optimizer and loss values must be finite/non-negative")
    if not 0.0 < args.coverage_target < 1.0:
        raise ValueError("coverage target must lie in (0,1)")
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("CVaR fraction must lie in (0,1]")
    if (args.real_dir / EXCLUDED_FILE).exists():
        raise RuntimeError(f"excluded Real trajectory is present: {EXCLUDED_FILE}")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("the formal reallocator experiment requires CUDA")

    files = {
        "point_checkpoint": args.point_checkpoint,
        "real_stats": args.real_stats,
        "native_summary": args.native_uq_dir / "summary.json",
        **{
            f"fold_{fold:02d}": (
                args.native_uq_dir / f"fold_{fold:02d}_best.pth")
            for fold in range(args.folds)
        },
        "all_tail_refit": args.native_uq_dir / "all_tail_refit.pth",
    }
    for label, path in files.items():
        _require_sha(path, FORMAL_IDENTITIES[label], label)

    summary = json.loads(files["native_summary"].read_text(encoding="utf-8"))
    if (summary.get("fold_selection_rule")
            != native_uq.FIXED_TERMINAL_RULE):
        raise RuntimeError("Native UQ source did not use fixed fold terminals")
    if summary.get("held_fold_checkpoint_selection") is not False:
        raise RuntimeError("Native UQ source selected on held folds")
    if summary.get("tail_windows") != EXPECTED_TAIL_WINDOWS:
        raise RuntimeError("Native UQ tail window count changed")
    return {
        "files": {label: str(path.resolve()) for label, path in files.items()},
        "sha256": dict(FORMAL_IDENTITIES),
        "formal_hyperparameters": dict(FORMAL_HYPERPARAMETERS),
        "native_summary": summary,
    }


def _flatten_time_channels(value: torch.Tensor) -> torch.Tensor:
    batch, steps, height, width, channels = value.shape
    return value.permute(0, 1, 4, 2, 3).reshape(
        batch, steps * channels, height, width)


def native_auxiliary(
        members: torch.Tensor, lower_physical: torch.Tensor,
        upper_physical: torch.Tensor, target_std: torch.Tensor,
        ) -> torch.Tensor:
    """Expose epistemic disagreement and current asymmetric widths locally."""
    if members.ndim != 6 or members.shape[-1] < 2:
        raise ValueError("members must have shape [B,M,T,H,W,C>=2]")
    if lower_physical.shape != upper_physical.shape:
        raise ValueError("native lower/upper shapes differ")
    if lower_physical.shape != members.shape[:1] + members.shape[2:5] + (2,):
        raise ValueError("native extents do not align with member forecasts")
    reference = lower_physical.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2)
    scale = target_std[:2].view(1, 1, 1, 1, 2)
    detached = members[..., :2].detach()
    disagreement = torch.cat((
        detached.std(dim=1, unbiased=False),
        detached.amax(dim=1) - detached.amin(dim=1),
    ), dim=-1)
    disagreement_scale = torch.cat((scale / reference, scale / reference), dim=-1)
    disagreement = torch.log1p(disagreement.clamp_min(0.0) * disagreement_scale)
    widths = torch.cat((
        torch.log1p(lower_physical.clamp_min(0.0) / reference),
        torch.log1p(upper_physical.clamp_min(0.0) / reference),
    ), dim=-1)
    auxiliary = torch.cat((
        _flatten_time_channels(disagreement),
        _flatten_time_channels(widths),
    ), dim=1)
    if auxiliary.shape[1] != NATIVE_AUXILIARY_CHANNELS:
        raise RuntimeError(
            f"native auxiliary has {auxiliary.shape[1]} channels, "
            f"expected {NATIVE_AUXILIARY_CHANNELS}")
    return auxiliary


@torch.inference_mode()
def collect_cache(model, loader, normalizer, device: torch.device) -> dict:
    keys = (
        "history", "prediction_norm", "prediction", "target",
        "native_lower", "native_upper", "valid_mask", "factor",
        "auxiliary")
    values: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    model.eval()
    target_std = normalizer.std_target
    physical_scale = target_std[:2].view(1, 1, 1, 1, 2)
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(history, target)
        (prediction_norm, members, _quantiles, lower_norm, upper_norm,
         _features, _coverage_logits) = model.forward_uq_training(history_norm)
        prediction = common.to_physical_target(
            prediction_norm, normalizer)[..., :2]
        lower = lower_norm * physical_scale
        upper = upper_norm * physical_scale
        batch = {
            "history": history_norm[..., :2],
            "prediction_norm": prediction_norm[..., :2],
            "prediction": prediction,
            "target": target[..., :2],
            "native_lower": lower,
            "native_upper": upper,
            "valid_mask": history[..., :2].abs().amax(dim=(1, 4)) > 1e-8,
            "factor": common.sample_sps_factor(prediction, target[..., :2]),
            "auxiliary": native_auxiliary(
                members, lower, upper, target_std),
        }
        for key, value in batch.items():
            if key == "valid_mask":
                dtype = torch.bool
            elif key in (
                    "prediction", "target", "native_lower", "native_upper",
                    "factor"):
                dtype = torch.float32
            else:
                dtype = torch.float16
            values[key].append(value.detach().to(device="cpu", dtype=dtype))
    return {key: torch.cat(items, dim=0) for key, items in values.items()}


def _load_point_with_fold_uq(
        point_payload: dict, fold_path: Path, normalizer,
        device: torch.device):
    model = construct_model(
        point_payload["model_config"]["architecture"],
        point_payload["model_config"]).to(device)
    model.load_state_dict(point_payload["model_state_dict"], strict=True)
    fold_payload = torch.load(
        fold_path, map_location="cpu", weights_only=False)
    if fold_payload.get("selection_rule") != native_uq.FIXED_TERMINAL_RULE:
        raise RuntimeError("fold UQ checkpoint is not a fixed terminal")
    if fold_payload.get("held_fold_checkpoint_selection") is not False:
        raise RuntimeError("fold UQ checkpoint used held-fold selection")
    if fold_payload.get("model_config") != model.architecture_config():
        raise RuntimeError("fold UQ architecture differs from point model")
    native_uq.load_uq_state(model, fold_payload["uq_state_dict"])
    common.set_model_stats(model, normalizer)
    model.eval()
    return model, fold_payload


def _load_refit_model(path: Path, normalizer, device: torch.device):
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    model = construct_model(
        payload["model_config"]["architecture"], payload["model_config"]
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    common.set_model_stats(model, normalizer)
    model.eval()
    return model, payload


def _arm_args(args: argparse.Namespace, arm: str) -> argparse.Namespace:
    result = copy.copy(args)
    result.use_dynamic_local = True
    result.use_static_template = False
    result.use_global_budget = False
    result.use_regime_film = False
    result.residual_stress = 1.0
    result.auxiliary_channels = (
        NATIVE_AUXILIARY_CHANNELS if arm == "native_aware" else 0)
    return result


def _arm_cache(cache: dict, arm: str) -> dict:
    if arm == "native_aware":
        return cache
    return {key: value for key, value in cache.items() if key != "auxiliary"}


def sample_attributes(manifest: list[dict]) -> tuple[list[int], list[int]]:
    folds, _fold_manifest = reallocator.trajectory_folds(manifest, 5)
    aoas: list[int] = []
    for item in manifest:
        aoas.extend([int(item["aoa"])] * len(item["val_starts"]))
    if len(folds) != len(aoas):
        raise RuntimeError("fold and AoA labels differ in length")
    return folds, aoas


def aggregate_aoa(
        fold_rows: list[dict], arm: str, aoa: int,
        ) -> dict[str, float]:
    rows = [row["by_aoa"][str(aoa)][arm] for row in fold_rows
            if str(aoa) in row["by_aoa"]]
    total = sum(row["scored_elements"] for row in rows)
    if total <= 0:
        raise RuntimeError(f"AoA {aoa} has no scored elements")
    return {
        metric: sum(row[metric] * row["scored_elements"] for row in rows) / total
        for metric in ("sps_score", "coverage", "mean_width")
    } | {"scored_elements": total}


def candidate_gate(summary: dict, arm: str) -> dict:
    baseline = summary["oof"]["baseline"]["sps_score"]
    score = summary["oof"][arm]["sps_score"]
    fold_deltas = [
        row[arm]["sps_score"] - row["baseline"]["sps_score"]
        for row in summary["folds"]]
    aoa_deltas = [
        summary["by_aoa"][key][arm]["sps_score"]
        - summary["by_aoa"][key]["baseline"]["sps_score"]
        for key in sorted(summary["by_aoa"], key=int)]
    checks = {
        "pooled_delta_ge_0p15": score - baseline >= 0.15,
        "positive_folds_ge_4": sum(value > 0.0 for value in fold_deltas) >= 4,
        "worst_fold_ge_minus_0p03": min(fold_deltas) >= -0.03,
        "positive_aoas_ge_4": sum(value > 0.0 for value in aoa_deltas) >= 4,
        "worst_aoa_ge_minus_0p05": min(aoa_deltas) >= -0.05,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "pooled_delta": score - baseline,
        "fold_deltas": fold_deltas,
        "aoa_deltas": aoa_deltas,
    }


def choose_arm(summary: dict) -> tuple[str | None, dict]:
    gates = {arm: candidate_gate(summary, arm) for arm in ARMS}
    passing = [arm for arm in ARMS if gates[arm]["pass"]]
    if not passing:
        return None, gates
    if "native_aware" in passing and "dynamic" in passing:
        native_gain = (
            summary["oof"]["native_aware"]["sps_score"]
            - summary["oof"]["dynamic"]["sps_score"])
        return ("native_aware" if native_gain >= 0.03 else "dynamic"), gates
    return passing[0], gates


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    assets = validate_assets(args)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    args.output_dir.mkdir(parents=True)
    torch.set_float32_matmul_precision("high")
    reallocator.seed_everything(args.seed)
    device = torch.device(args.device)

    data_args = argparse.Namespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, real_stats=args.real_stats,
        output_dir=args.output_dir, seed=args.seed)
    (train_loader, tail_loader, normalizer, train_windows,
     tail_windows, trajectories) = common.prepare_real(data_args, device)
    del train_loader
    if tail_windows != EXPECTED_TAIL_WINDOWS or trajectories != 81:
        raise RuntimeError(
            f"clean Real split changed: trajectories={trajectories}, "
            f"tail_windows={tail_windows}")
    manifest = json.loads(
        (args.output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    sample_folds, sample_aoas = sample_attributes(manifest)
    fold_tensor = torch.tensor(sample_folds, dtype=torch.long)
    aoa_tensor = torch.tensor(sample_aoas, dtype=torch.long)
    all_indices = torch.arange(len(sample_folds), dtype=torch.long)

    point_payload = torch.load(
        str(args.point_checkpoint), map_location="cpu", weights_only=False,
        mmap=True)
    fold_rows = []
    expected_folds = assets["native_summary"]["folds"]
    for fold in range(args.folds):
        if time.perf_counter() >= deadline:
            raise TimeoutError("wall budget expired before fold cache")
        model, fold_payload = _load_point_with_fold_uq(
            point_payload, args.native_uq_dir / f"fold_{fold:02d}_best.pth",
            normalizer, device)
        cache = collect_cache(model, tail_loader, normalizer, device)
        del model
        torch.cuda.empty_cache()
        train_indices = all_indices[fold_tensor != fold]
        eval_indices = all_indices[fold_tensor == fold]
        baseline = reallocator.exact_metrics(
            None, cache, eval_indices, device, args.eval_batch_size)
        expected = expected_folds[fold]["final"]
        if abs(baseline["sps_score"] - expected["sps_score"]) > 0.03:
            raise RuntimeError(
                f"fold {fold} Native replay drifted: "
                f"{baseline['sps_score']} vs {expected['sps_score']}")
        row = {
            "fold": fold,
            "train_samples": int(train_indices.numel()),
            "eval_samples": int(eval_indices.numel()),
            "baseline": baseline,
            "native_source_metrics": fold_payload["metrics"]["final"],
            "by_aoa": {},
        }
        heads = {}
        for arm_index, arm in enumerate(ARMS):
            arm_args = _arm_args(args, arm)
            head, history, held_selection = reallocator.train_head(
                arm_args, _arm_cache(cache, arm), train_indices,
                eval_indices=None, updates=args.fold_updates, device=device,
                seed=args.seed + 100 * fold + arm_index + 1)
            if held_selection is not None:
                raise RuntimeError("fixed-terminal arm unexpectedly selected held state")
            metrics = reallocator.exact_metrics(
                head, _arm_cache(cache, arm), eval_indices,
                device, args.eval_batch_size)
            row[arm] = metrics
            row[f"{arm}_delta"] = metrics["sps_score"] - baseline["sps_score"]
            row[f"{arm}_history"] = history
            heads[arm] = head
            torch.save({
                "architecture": "b210_native_uq_reallocator",
                "arm": arm,
                "fold": fold,
                "selection_rule": "fixed_terminal_no_held_fold_selection",
                "selected_iteration": args.fold_updates,
                "model_config": head.architecture_config(),
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in head.state_dict().items()},
                "source_native_uq": str(
                    args.native_uq_dir / f"fold_{fold:02d}_best.pth"),
                "metrics": metrics,
            }, args.output_dir / f"fold_{fold:02d}_{arm}.pth")

        for aoa in sorted(set(sample_aoas)):
            indices = all_indices[(fold_tensor == fold) & (aoa_tensor == aoa)]
            if not indices.numel():
                continue
            row["by_aoa"][str(aoa)] = {
                "baseline": reallocator.exact_metrics(
                    None, cache, indices, device, args.eval_batch_size),
                **{
                    arm: reallocator.exact_metrics(
                        heads[arm], _arm_cache(cache, arm), indices,
                        device, args.eval_batch_size)
                    for arm in ARMS
                },
            }
        fold_rows.append(row)
        print(json.dumps({
            "event": "b210_reallocator_fold_complete",
            "fold": fold,
            "baseline": baseline,
            **{arm: row[arm] for arm in ARMS},
        }), flush=True)
        del cache, heads
        torch.cuda.empty_cache()

    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "assets": assets,
        "protocol": {
            "held_fold_checkpoint_selection": False,
            "fold_updates": args.fold_updates,
            "final_updates": args.final_updates,
            "arms": list(ARMS),
            "native_auxiliary_channels": NATIVE_AUXILIARY_CHANNELS,
            "train_windows_point_model": train_windows,
            "tail_windows": tail_windows,
            "trajectories": trajectories,
            "excluded_file": EXCLUDED_FILE,
        },
        "folds": fold_rows,
        "oof": {
            key: reallocator.aggregate_fold_metrics(fold_rows, key)
            for key in ("baseline", *ARMS)
        },
        "by_aoa": {
            str(aoa): {
                key: aggregate_aoa(fold_rows, key, aoa)
                for key in ("baseline", *ARMS)
            }
            for aoa in sorted(set(sample_aoas))
        },
    }
    selected_arm, gates = choose_arm(result)
    result["gates"] = gates
    result["selected_arm"] = selected_arm

    if selected_arm is not None:
        if time.perf_counter() >= deadline:
            raise TimeoutError("wall budget expired before all-tail refit")
        refit_model, refit_payload = _load_refit_model(
            args.native_uq_dir / "all_tail_refit.pth", normalizer, device)
        refit_cache = collect_cache(
            refit_model, tail_loader, normalizer, device)
        del refit_model
        torch.cuda.empty_cache()
        baseline_refit = reallocator.exact_metrics(
            None, refit_cache, all_indices, device, args.eval_batch_size)
        expected_refit = assets["native_summary"]["final_refit"]["final"]
        if abs(baseline_refit["sps_score"] - expected_refit["sps_score"]) > 0.03:
            raise RuntimeError("all-tail Native refit replay drifted")
        final_args = _arm_args(args, selected_arm)
        final_head, final_history, held_selection = reallocator.train_head(
            final_args, _arm_cache(refit_cache, selected_arm), all_indices,
            eval_indices=None, updates=args.final_updates, device=device,
            seed=args.seed + 10_000)
        if held_selection is not None:
            raise RuntimeError("all-tail refit unexpectedly selected held state")
        final_metrics = reallocator.exact_metrics(
            final_head, _arm_cache(refit_cache, selected_arm), all_indices,
            device, args.eval_batch_size)
        refit_path = args.output_dir / "all_tail_reallocator_refit.pth"
        torch.save({
            "architecture": "b210_native_uq_reallocator",
            "arm": selected_arm,
            "selection_rule": "fixed final iteration from OOF protocol",
            "selected_iteration": args.final_updates,
            "model_config": final_head.architecture_config(),
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in final_head.state_dict().items()},
            "source_native_uq": str(
                args.native_uq_dir / "all_tail_refit.pth"),
            "source_native_uq_config": refit_payload["model_config"],
            "history": final_history,
            "baseline_fit_diagnostic": baseline_refit,
            "reallocated_fit_diagnostic": final_metrics,
            "oof_selected_arm": selected_arm,
            "oof_delta": gates[selected_arm]["pooled_delta"],
        }, refit_path)
        result["refit"] = {
            "checkpoint": str(refit_path.resolve()),
            "checkpoint_sha256": sha256(refit_path),
            "evaluation_scope": "all-tail training diagnostic; not OOF",
            "baseline": baseline_refit,
            "reallocated": final_metrics,
        }

    result["elapsed_seconds"] = time.perf_counter() - started
    result["submission_authorized"] = False
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({
        "event": "b210_scratch_uq_reallocator_complete",
        "selected_arm": selected_arm,
        "oof": result["oof"],
        "gates": gates,
        "result": str(args.output_dir / "result.json"),
    }), flush=True)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
