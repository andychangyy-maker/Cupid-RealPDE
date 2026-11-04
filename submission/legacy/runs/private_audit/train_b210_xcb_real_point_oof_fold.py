#!/usr/bin/env python3
"""Train one truly fold-excluded XCB Real point model and audit residuals."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys
import time
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (
        REPO_ROOT, PRIVATE, REPO_ROOT / "code",
        REPO_ROOT / "code" / "training", REPO_ROOT / "code" / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_sim_real_disagreement_transfer as transfer  # noqa: E402
import train_b210_self_describing_real_dense_aug as self_describing  # noqa: E402
import train_b210_xcb_member_standard24k_real_dense_aug as xcb  # noqa: E402
import train_b210_scratch_real_dense_aug as paired  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402


real = self_describing.real
ft12 = real.ft12
small4k = real.small4k
EXPERIMENT = "b210_xcb_real_point_oof_fold"
SCHEMA = "b210_xcb_real_point_oof_fold_v1"
ENDPOINT_SCHEMA = "b210_xcb_real_point_oof_endpoint_v1"
ENDPOINT = "ema_u4000"
UPDATES = 4_000
FOLDS = 5
GATE = {
    "error_map_spearman_min": 0.40,
    "error_map_spearman_max": 0.95,
    "deployed_spread_vs_oof_error_min": 0.20,
    "deployed_top_error_recall_min": 0.20,
    "oof_to_full_mae_ratio_max": 2.50,
}


def condition_fold_assignments(
        rows: Sequence[Mapping[str, object]], folds: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("point OOF needs at least two folds")
    re_values = sorted({int(row["re"]) for row in rows})
    aoa_values = sorted({int(row["aoa"]) for row in rows})
    re_rank = {value: index for index, value in enumerate(re_values)}
    aoa_rank = {value: index for index, value in enumerate(aoa_values)}
    result = {}
    for row in rows:
        name = str(row["file"])
        if name in result:
            raise ValueError("point OOF manifest contains duplicate files")
        result[name] = (
            re_rank[int(row["re"])] + 2 * aoa_rank[int(row["aoa"])]) % folds
    if set(result.values()) != set(range(folds)):
        raise RuntimeError("point OOF assignment has an empty fold")
    return result


class FoldAccessAudit:
    def __init__(self, held_files: set[str]) -> None:
        if not held_files:
            raise ValueError("point OOF held fold is empty")
        self.held_files = set(held_files)
        self.fit_opened: list[str] = []
        self.held_opened: list[str] = []
        self.endpoint: dict | None = None

    def before_open(self, row: dict, phase: str) -> None:
        name = str(row["file"])
        if name == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded trajectory entered point OOF")
        if phase == "fit":
            if name in self.held_files or self.endpoint is not None:
                raise RuntimeError("point OOF fit opened held or late data")
            return
        if phase != "held" or name not in self.held_files:
            raise RuntimeError("point OOF held open is outside the fold")
        if self.endpoint is None:
            raise RuntimeError("point OOF held target opened before endpoint seal")
        endpoint_path = Path(str(self.endpoint["path"]))
        if (not endpoint_path.is_file()
                or dr1.sha256(endpoint_path) != self.endpoint["file_sha256"]):
            raise RuntimeError("point OOF endpoint changed before held open")

    def record_open(self, row: dict, phase: str) -> None:
        target = self.fit_opened if phase == "fit" else self.held_opened
        name = str(row["file"])
        if name in target:
            raise RuntimeError("point OOF trajectory opened twice")
        target.append(name)

    def seal(self, value: Mapping[str, object]) -> None:
        if self.endpoint is not None:
            raise RuntimeError("point OOF endpoint sealed twice")
        self.endpoint = dict(value)


@contextmanager
def endpoint_contract(
        held_fold: int, experiment: str = EXPERIMENT) -> Iterator[None]:
    replacements = {
        "EXPERIMENT": experiment,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": ENDPOINT,
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": 12_000,
        "HELD_AOA": held_fold,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(small4k, name, value)


def official_comparison(
        candidate: Mapping[str, float],
        control: Mapping[str, float]) -> dict[str, object]:
    keys = (
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "native_sps_score")
    delta = {key: float(candidate[key]) - float(control[key]) for key in keys}
    point = (
        0.46 * delta["rel_l2_score"]
        + 0.10 * delta["tke_rel_l2_score"]
        + 0.09 * delta["mvpe_rel_l2_score"])
    return {
        "delta": delta,
        "point_contribution_delta": point,
        "non_time_contribution_delta": point + 0.25 * delta["native_sps_score"],
    }


def full_normalizer(path: Path, device: torch.device) -> smu1.FoldNormalizer:
    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ValueError("full Real normalizer is not the four-stat tuple")
    return smu1.FoldNormalizer(tuple(values), device)


def load_full_model(
        checkpoint: Path, normalizer: smu1.FoldNormalizer,
        device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config, parameters, controller = (
        self_describing.architecture_from_payload(payload))
    model = small4k.EnsembleUNet2d(config).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if (sum(value.numel() for value in model.parameters()) != parameters
            or sum(value.numel() for value in model.energy_controller.parameters())
            != controller):
        raise RuntimeError("full point model parameter contract changed")
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def member_integrated_lgf_contract(enabled: bool) -> dict[str, object]:
    """Return the parameter-free official-vs-legacy TKE gradient contract."""
    if not enabled:
        return {}
    legacy = tuple(
        float(value) for value in ft12.npt1.joint_real.REAL_ACCURACY_WEIGHTS)
    official = (46.0, 10.0, 9.0)
    return {
        "point_accuracy_weights": official,
        "member_accuracy_weights": official,
        "legacy_tke_gradient_floor": True,
        "legacy_tke_gradient_floor_on_members": True,
        "legacy_point_accuracy_weights": legacy,
        "legacy_member_accuracy_weights": legacy,
    }


def residual_transfer_gate(
        fold_arrays: Mapping[str, np.ndarray],
        full_arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    fold_error = np.log1p(fold_arrays["error"] / 1.0e-4)
    full_error = np.log1p(full_arrays["error"] / 1.0e-4)
    error_rank = transfer.safe_spearman(fold_error, full_error)
    deployed_vs_oof = transfer.safe_spearman(
        full_arrays["spread_z"], fold_error)
    top = transfer.top_fraction_diagnostic(
        full_arrays["spread_z"], fold_arrays["error"])
    mae_ratio = float(
        fold_arrays["error"].mean()
        / max(float(full_arrays["error"].mean()), 1.0e-12))
    conditions = {
        "oof_error_map_related_but_nonidentical": (
            GATE["error_map_spearman_min"] <= error_rank
            <= GATE["error_map_spearman_max"]),
        "deployed_spread_localizes_oof_error": (
            deployed_vs_oof >= GATE["deployed_spread_vs_oof_error_min"]),
        "deployed_top_spread_recalls_oof_error": (
            top["top_error_recall"]
            >= GATE["deployed_top_error_recall_min"]),
        "oof_error_is_not_catastrophic": (
            mae_ratio <= GATE["oof_to_full_mae_ratio_max"]),
    }
    passed = all(conditions.values())
    return {
        "thresholds": dict(GATE),
        "oof_full_error_log_spearman": error_rank,
        "deployed_spread_z_vs_oof_error_spearman": deployed_vs_oof,
        "deployed_spread_for_oof_error_top10": top,
        "oof_to_full_sampled_mae_ratio": mae_ratio,
        "conditions": conditions,
        "pass": passed,
        "decision": (
            "PASS_BUILD_FULL_POINT_OOF_RESIDUAL_CACHE"
            if passed else "STOP_POINT_OOF_RESIDUAL_ROUTE"),
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0 <= args.held_fold < args.folds:
        raise ValueError("held fold is outside point OOF range")
    if args.folds != FOLDS:
        raise ValueError("formal point OOF pilot fixes five folds")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("point OOF pilot requires CUDA")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 manifest byte identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    assignments = condition_fold_assignments(rows, args.folds)
    fit_rows = [
        row for row in rows
        if assignments[str(row["file"])] != args.held_fold]
    held_rows = [
        row for row in rows
        if assignments[str(row["file"])] == args.held_fold]
    held_files = {str(row["file"]) for row in held_rows}
    evaluation_rows = paired.load_evaluation_rows(
        args.evaluation_manifest, rows)
    held_evaluation_rows = [
        row for row in evaluation_rows if str(row["file"]) in held_files]
    if not fit_rows or not held_rows or not held_evaluation_rows:
        raise RuntimeError("point OOF fold partition is empty")
    start_sha = dr1.sha256(args.start_sim)
    if start_sha != args.start_sim_sha256.lower():
        raise RuntimeError("point OOF Sim start SHA changed")
    start_payload = torch.load(
        args.start_sim, map_location="cpu", weights_only=False)
    config, parameters, controller = (
        self_describing.architecture_from_payload(start_payload))
    del start_payload
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    audit = FoldAccessAudit(held_files)
    training_contract = member_integrated_lgf_contract(
        args.member_integrated_lgf)
    experiment = (
        "b210_xcb_member_integrated_lgf_real_point_oof_fold"
        if args.member_integrated_lgf else EXPERIMENT)
    fit_trajectories, fit_metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", audit)
    dense_windows = real.dense_windows(
        fit_trajectories, real.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        fit_trajectories, dense_windows)
    statistics = smu1.fold_statistics(
        smu1.window_sufficient_statistics(fit_trajectories, sparse_windows),
        range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "fit_normalizer.pt")
    validator = xcb.bind_validator(ft12.validate_start_checkpoint)
    ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
    model, start_report = validator(
        args.start_sim, start_sha, normalizer, device,
        expected_config=config, expected_parameters=parameters,
        expected_controller_parameters=controller)
    schedule = smu1.paired_batch_schedule(
        range(len(dense_windows)), dense_windows,
        updates=UPDATES, seed=small4k.SEED)
    schedule["updates"] = UPDATES
    arm_dir = args.output_dir / "fold_point"
    arm_dir.mkdir()
    with endpoint_contract(args.held_fold, experiment):
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, audit,
            input_augmentation=real.AUGMENTATION,
            expected_parameters=parameters,
            **training_contract)
        if audit.endpoint != seal:
            raise RuntimeError("point OOF endpoint was not bound to audit")
    del model, dataset, fit_trajectories
    gc.collect()
    torch.cuda.empty_cache()

    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_evaluation_rows, "held", audit)
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    held_indices = list(range(len(held_windows)))
    with endpoint_contract(args.held_fold, experiment):
        fold_model = small4k.load_primary_endpoint(
            seal, normalizer, device,
            expected_model_config=config,
            expected_parameters=parameters,
            expected_controller_parameters=controller)
    fold_metrics, fold_state, fold_target_audit = smu1.evaluate_model(
        fold_model, held_dataset, held_indices, normalizer, device, deadline)
    del fold_model
    gc.collect()
    torch.cuda.empty_cache()
    deployed_normalizer = full_normalizer(args.full_real_stats, device)
    full_model = load_full_model(
        args.full_point_checkpoint, deployed_normalizer, device)
    full_metrics, full_state, full_target_audit = smu1.evaluate_model(
        full_model, held_dataset, held_indices,
        deployed_normalizer, device, deadline)
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    if fold_target_audit != full_target_audit:
        raise RuntimeError("point OOF held target audit differs from full point")

    loader = DataLoader(
        held_dataset, batch_size=args.eval_batch_size,
        shuffle=False, num_workers=0)
    fold_report, fold_arrays = transfer.evaluate_checkpoint(
        Path(str(seal["path"])), loader, device, args.element_stride)
    full_report, full_arrays = transfer.evaluate_checkpoint(
        args.full_point_checkpoint, loader, device, args.element_stride)
    gate = residual_transfer_gate(fold_arrays, full_arrays)
    result = {
        "schema": SCHEMA,
        "experiment": experiment,
        "training_contract": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in training_contract.items()},
        "held_fold": args.held_fold,
        "folds": args.folds,
        "fold_formula": "(rank(re)+2*rank(aoa)) % folds",
        "fold_assignments": assignments,
        "manifest": manifest_report,
        "evaluation_manifest_sha256": dr1.sha256(args.evaluation_manifest),
        "fit": {
            "files": len(fit_rows), "sparse_windows": len(sparse_windows),
            "dense_windows": len(dense_windows),
            "normalizer_sha256": smu1.statistics_digest(statistics),
            "training": training, "seal": seal,
        },
        "held": {
            "files": len(held_rows), "windows": len(held_windows),
            "fold_metrics": fold_metrics, "full_metrics": full_metrics,
            "fold_metric_state": fold_state, "full_metric_state": full_state,
            "target_audit": fold_target_audit,
            "fold_minus_full": official_comparison(
                fold_metrics, full_metrics),
        },
        "residual_diagnostic": {
            "fold_excluded": fold_report,
            "full_point": full_report,
            "fold_as_sim_full_as_real_cross_fields": (
                transfer.cross_map_diagnostic(fold_arrays, full_arrays)),
        },
        "gate": gate,
        "access_audit": {
            "fit_opened": audit.fit_opened,
            "held_opened_after_seal": audit.held_opened,
            "held_target_opened_before_seal": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", required=True)
    parser.add_argument("--full-point-checkpoint", type=Path, required=True)
    parser.add_argument("--full-real-stats", type=Path, required=True)
    parser.add_argument("--held-fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--element-stride", type=int, default=197)
    parser.add_argument(
        "--member-integrated-lgf", action="store_true",
        help=("Use published point weights and the parameter-free legacy-TKE "
              "gradient floor on both mean and member prediction paths."))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_xcb_real_point_oof_fold_complete",
        "held_fold": args.held_fold,
        "gate": result["gate"],
        "result": str(args.output_dir / "result.json"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
