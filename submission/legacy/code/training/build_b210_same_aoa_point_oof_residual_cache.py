#!/usr/bin/env python3
"""Build 3,341 trajectory-cross-fit B210 residual labels.

Each complete released-Real trajectory is predicted by the Real4k point fold
that excluded that trajectory.  Folds follow the frozen same-AoA condition
formula and therefore each held partition contains all five AoAs.  The cache
stores physical signed UV residuals and point factors for fresh Native-UQ;
it never accepts the clean716 or any evaluation manifest.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import torch

import b210_same_aoa_trajectory_folds as fold_plan
import build_b210_full_real_point_oof_residual_cache as whole_cache
import train_b210_same_aoa_point_oof_fold as point_fold


dr1 = point_fold.dr1
ft12 = point_fold.ft12
SCHEMA = "b210_same_aoa_trajectory_point_oof_residual_cache_v1"
COMPLETION_EVENT = "b210_same_aoa_point_oof_residual_cache_complete"
EXPECTED_SAMPLES = fold_plan.EXPECTED_TOTAL["full_oof_windows"]


read_trajectory = whole_cache.read_trajectory
uvp_windows = whole_cache.uvp_windows
predict_fold = whole_cache.predict_fold
full_evaluation_rows = whole_cache.full_evaluation_rows


def _lineage_matches(payload: Mapping[str, object]) -> bool:
    observed = payload.get("model_lineage")
    return (observed == point_fold.MODEL_LINEAGE
            or (observed is None and not point_fold.LINEAGE_FIELD_REQUIRED))


def _sha_tensor_bytes(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def validate_fold_results(
        rows: Sequence[Mapping[str, object]], assignments: Mapping[str, int],
        result_pairs: Sequence[Sequence[str]],
        ) -> tuple[dict[int, Path], list[dict]]:
    if len(result_pairs) != fold_plan.FOLDS:
        raise ValueError("one SHA-bound point result is required per fold")
    expected_files = {
        fold: [str(row["file"]) for row in rows
               if assignments[str(row["file"])] == fold]
        for fold in fold_plan.FOLD_VALUES}
    checkpoints: dict[int, Path] = {}
    reports = []
    for raw_path, raw_sha in result_pairs:
        path = Path(raw_path)
        expected_sha = dr1.normalized_sha256(
            raw_sha, "same-AoA point-OOF result SHA")
        if not path.is_file() or dr1.sha256(path) != expected_sha:
            raise RuntimeError("same-AoA point fold result SHA changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        point_fold.whole_aoa.verify_json_payload(payload)
        fold = int(payload.get("held_fold", -1))
        if (fold not in fold_plan.FOLD_VALUES or fold in checkpoints
                or payload.get("schema") != point_fold.SCHEMA
                or int(payload.get("folds", -1)) != fold_plan.FOLDS
                or payload.get("fold_definition") != fold_plan.FORMULA
                or payload.get("fold_plan_assignment_sha256")
                != fold_plan.EXPECTED_ASSIGNMENT_SHA256
                or not _lineage_matches(payload)
                or payload.get("decision")
                != "COMPLETE_SAME_AOA_TRAJECTORY_POINT_OOF_FOLD"
                or payload.get("held_data_opened") is not False
                or payload.get("clean716_manifest_read") is not False
                or payload.get("evaluation_manifest_read") is not False
                or payload.get("start", {}).get("checkpoint_sha256")
                != point_fold.START_SHA256
                or payload.get("manifest_file_sha256")
                != ft12.FORMAL_MANIFEST_FILE_SHA256
                or payload.get("manifest", {}).get("semantic_sha256")
                != ft12.FORMAL_MANIFEST_SEMANTIC_SHA256
                or payload.get("held_files") != expected_files[fold]):
            raise RuntimeError("same-AoA point fold result identity changed")
        held = [row for row in payload.get("held_assignments", [])]
        if (set(int(row["aoa"]) for row in held)
                != set(fold_plan.AOA_VALUES)
                or dr1.canonical_sha256(held)
                != payload.get("held_assignment_sha256")):
            raise RuntimeError("same-AoA held fold lost physical AoA coverage")
        seal = payload.get("fit", {}).get("seal", {})
        checkpoint = Path(str(seal.get("path", "")))
        verified = point_fold.validate_endpoint(
            checkpoint, seal, held_fold=fold)
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint_payload.get("model_state_dict")
        if (not isinstance(state, Mapping)
                or dr1.model_state_sha256(state)
                != verified["model_state_sha256"]):
            raise RuntimeError("same-AoA point fold state changed")
        checkpoints[fold] = checkpoint.resolve()
        reports.append({
            "fold": fold,
            "held_files": expected_files[fold],
            "held_aoas": list(fold_plan.AOA_VALUES),
            "result": str(path.resolve()),
            "result_sha256": expected_sha,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": verified["file_sha256"],
            "model_state_sha256": verified["model_state_sha256"],
            "normalizer_sha256": verified["normalizer_sha256"],
            "model_lineage": point_fold.MODEL_LINEAGE,
        })
    if set(checkpoints) != set(fold_plan.FOLD_VALUES):
        raise RuntimeError("same-AoA point checkpoint coverage is incomplete")
    return checkpoints, sorted(reports, key=lambda row: int(row["fold"]))


def load_cache(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    """Validate a complete cache for downstream Native-UQ consumers."""
    expected_sha256 = dr1.normalized_sha256(
        expected_sha256, "same-AoA OOF residual cache SHA")
    if not path.is_file() or dr1.sha256(path) != expected_sha256:
        raise RuntimeError("same-AoA OOF residual cache SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    residuals = payload.get("signed_residual_physical_uv")
    factors = payload.get("oof_point_factor")
    identities = payload.get("sample_identities")
    if (payload.get("schema") != SCHEMA
            or int(payload.get("folds", -1)) != fold_plan.FOLDS
            or payload.get("fold_formula") != fold_plan.FORMULA
            or payload.get("fold_plan_assignment_sha256")
            != fold_plan.EXPECTED_ASSIGNMENT_SHA256
            or not _lineage_matches(payload)
            or int(payload.get("window_stride", -1)) != 20
            or int(payload.get("samples", -1)) != EXPECTED_SAMPLES
            or payload.get("manifest", {}).get("semantic_sha256")
            != ft12.FORMAL_MANIFEST_SEMANTIC_SHA256
            or payload.get("manifest_file_sha256")
            != ft12.FORMAL_MANIFEST_FILE_SHA256
            or payload.get("clean716_manifest_read") is not False
            or payload.get("evaluation_manifest_read") is not False
            or not isinstance(identities, list)
            or len(identities) != EXPECTED_SAMPLES
            or dr1.canonical_sha256(identities)
            != payload.get("sample_identity_sha256")
            or not isinstance(residuals, torch.Tensor)
            or residuals.shape != (EXPECTED_SAMPLES, 20, 32, 64, 2)
            or residuals.dtype != torch.float16
            or not bool(torch.isfinite(residuals).all())
            or _sha_tensor_bytes(residuals)
            != payload.get("signed_residual_physical_uv_sha256")
            or not isinstance(factors, torch.Tensor)
            or factors.shape != (EXPECTED_SAMPLES,)
            or not bool(torch.isfinite(factors).all())
            or _sha_tensor_bytes(factors)
            != payload.get("oof_point_factor_sha256")):
        raise RuntimeError("same-AoA OOF residual cache contract changed")
    return payload, {
        "path": str(path.resolve()),
        "file_sha256": expected_sha256,
        "schema": SCHEMA,
        "samples": EXPECTED_SAMPLES,
        "sample_identity_sha256": payload["sample_identity_sha256"],
        "signed_residual_physical_uv_sha256": payload[
            "signed_residual_physical_uv_sha256"],
        "oof_point_factor_sha256": payload["oof_point_factor_sha256"],
        "ordered_history_stream_sha256": payload[
            "ordered_history_stream_sha256"],
        "ordered_target_stream_sha256": payload[
            "ordered_target_stream_sha256"],
        "fold_plan_assignment_sha256": fold_plan.EXPECTED_ASSIGNMENT_SHA256,
        "model_lineage": point_fold.MODEL_LINEAGE,
        "fold_results": payload["fold_results"],
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--fold-result", nargs=2, action="append", required=True,
                        metavar=("PATH", "SHA256"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.output.exists() or temporary.exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    plan = fold_plan.build_plan(rows)
    assignments = {row["file"]: int(row["fold"])
                   for row in plan["assignments"]}
    evaluation_rows = full_evaluation_rows(rows)
    checkpoints, fold_reports = validate_fold_results(
        evaluation_rows, assignments, args.fold_result)
    identities = [
        {"file": str(row["file"]), "start": int(start),
         "re": int(row["re"]), "aoa": int(row["aoa"]),
         "fold": assignments[str(row["file"])]}
        for row in evaluation_rows for start in row["val_starts"]]
    positions = {
        (str(row["file"]), int(row["start"])): index
        for index, row in enumerate(identities)}
    if (len(identities) != EXPECTED_SAMPLES
            or len(positions) != EXPECTED_SAMPLES
            or set(row["fold"] for row in identities)
            != set(fold_plan.FOLD_VALUES)):
        raise RuntimeError("same-AoA OOF sample identities are incomplete")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    residuals = torch.empty(
        (EXPECTED_SAMPLES, 20, 32, 64, 2), dtype=torch.float16)
    factors = torch.full((EXPECTED_SAMPLES,), float("nan"), dtype=torch.float32)
    history_hashes: list[str | None] = [None] * EXPECTED_SAMPLES
    target_hashes: list[str | None] = [None] * EXPECTED_SAMPLES
    written = 0
    for fold in fold_plan.FOLD_VALUES:
        fold_rows = [row for row in evaluation_rows
                     if assignments[str(row["file"])] == fold]
        written += predict_fold(
            checkpoints[fold], fold_rows, positions, args.real_dir,
            residuals, factors, history_hashes, target_hashes,
            device, args.batch_size)
    if (written != EXPECTED_SAMPLES or not torch.isfinite(factors).all()
            or not torch.isfinite(residuals).all()
            or any(value is None for value in history_hashes + target_hashes)):
        raise RuntimeError("same-AoA OOF residual cache is incomplete/non-finite")
    payload = {
        "schema": SCHEMA,
        "folds": fold_plan.FOLDS,
        "fold_definition": (
            "complete trajectory groups; all five physical AoAs per fold"),
        "fold_formula": fold_plan.FORMULA,
        "fold_plan_assignment_sha256": plan["assignment_sha256"],
        "model_lineage": point_fold.MODEL_LINEAGE,
        "aoa_values": list(fold_plan.AOA_VALUES),
        "window_source": "all legal complete-trajectory 20-to-20 windows",
        "window_stride": 20,
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "sample_identities": identities,
        "sample_identity_sha256": dr1.canonical_sha256(identities),
        "ordered_history_sample_sha256": history_hashes,
        "ordered_target_sample_sha256": target_hashes,
        "ordered_history_stream_sha256": dr1.canonical_sha256(history_hashes),
        "ordered_target_stream_sha256": dr1.canonical_sha256(target_hashes),
        "fold_results": fold_reports,
        "signed_residual_physical_uv": residuals,
        "signed_residual_physical_uv_sha256": _sha_tensor_bytes(residuals),
        "oof_point_factor": factors,
        "oof_point_factor_sha256": _sha_tensor_bytes(factors),
        "mean_absolute_residual": float(residuals.float().abs().mean()),
        "samples": EXPECTED_SAMPLES,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    temporary.replace(args.output)
    report = {key: value for key, value in payload.items()
              if not isinstance(value, torch.Tensor)}
    report.update({
        "output": str(args.output.resolve()),
        "output_sha256": dr1.sha256(args.output),
        "cache_bytes": args.output.stat().st_size,
    })
    report = point_fold.whole_aoa.seal_json_payload(report)
    report_path = args.output.with_suffix(".json")
    dr1.write_json_atomic(report_path, report)
    point_fold.whole_aoa.verify_json_payload(json.loads(
        report_path.read_text(encoding="utf-8")))
    load_cache(args.output, dr1.sha256(args.output))
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "event": COMPLETION_EVENT,
        "samples": report["samples"],
        "mean_absolute_residual": report["mean_absolute_residual"],
        "output": report["output"],
    }), flush=True)


if __name__ == "__main__":
    main()
