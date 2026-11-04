#!/usr/bin/env python3
"""Build an identity-checked Real point-OOF residual cache.

Each fixed-tail window is predicted by the point model whose Real fit excluded
the window's entire Reynolds/AoA condition fold.  The resulting signed physical
u/v residuals are labels for a later UQ allocator; none of the fold checkpoints
is a submission candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation import audit_sim_real_disagreement_transfer as transfer  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402


SCHEMA = "b210_xcb_point_oof_residual_cache_v1"
FOLD_FORMULA = "(rank(re)+2*rank(aoa)) % folds"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def condition_fold_assignments(
        rows: Sequence[Mapping[str, object]], folds: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("point OOF needs at least two folds")
    re_values = sorted({int(row["re"]) for row in rows})
    aoa_values = sorted({int(row["aoa"]) for row in rows})
    re_rank = {value: index for index, value in enumerate(re_values)}
    aoa_rank = {value: index for index, value in enumerate(aoa_values)}
    assignments: dict[str, int] = {}
    for row in rows:
        filename = str(row["file"])
        if filename in assignments:
            raise ValueError("evaluation manifest contains duplicate files")
        assignments[filename] = (
            re_rank[int(row["re"])] + 2 * aoa_rank[int(row["aoa"])]) % folds
    if set(assignments.values()) != set(range(folds)):
        raise ValueError("point OOF assignment has an empty fold")
    return assignments


def validate_manifest(raw: object) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise TypeError("evaluation manifest must be a non-empty list")
    rows: list[dict] = []
    identities: set[tuple[str, int]] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("evaluation manifest row must be a mapping")
        required = {"file", "frames", "re", "aoa", "val_starts"}
        if not required.issubset(item):
            raise ValueError("evaluation manifest row is incomplete")
        filename = str(item["file"])
        if Path(filename).name != filename or not filename.endswith(".h5"):
            raise ValueError(f"unsafe evaluation filename {filename}")
        frames = int(item["frames"])
        starts = item["val_starts"]
        if (not isinstance(starts, list) or not starts
                or any(not isinstance(start, int) for start in starts)):
            raise ValueError(f"invalid val_starts for {filename}")
        for start in starts:
            identity = (filename, start)
            if start < 0 or start + 40 > frames or identity in identities:
                raise ValueError(f"invalid or duplicate window {identity}")
            identities.add(identity)
        rows.append(dict(item))
    return rows


def validate_fold_results(
        rows: Sequence[Mapping[str, object]], assignments: Mapping[str, int],
        result_paths: Sequence[Path], folds: int,
        ) -> tuple[dict[int, Path], list[dict]]:
    if len(result_paths) != folds:
        raise ValueError("exactly one point-OOF result is required per fold")
    checkpoints: dict[int, Path] = {}
    reports: list[dict] = []
    expected_assignments = dict(assignments)
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        fold = int(payload.get("held_fold", -1))
        if fold in checkpoints or not 0 <= fold < folds:
            raise ValueError("duplicate or invalid held fold result")
        if payload.get("folds") != folds:
            raise ValueError("point-OOF fold count changed")
        if payload.get("fold_formula") != FOLD_FORMULA:
            raise ValueError("point-OOF fold formula changed")
        observed = {
            str(key): int(value)
            for key, value in payload.get("fold_assignments", {}).items()}
        if observed != expected_assignments:
            raise ValueError("point-OOF result assignment differs from manifest")
        if not bool(payload.get("gate", {}).get("pass")):
            raise RuntimeError(f"point-OOF fold {fold} did not pass its gate")
        seal = payload.get("fit", {}).get("seal", {})
        checkpoint = Path(str(seal.get("path", ""))).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha = sha256(checkpoint)
        if checkpoint_sha != seal.get("file_sha256"):
            raise RuntimeError(f"point-OOF fold {fold} checkpoint SHA changed")
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False)
        if int(checkpoint_payload.get("held_aoa", -1)) != fold:
            raise ValueError("checkpoint held-fold identity changed")
        checkpoints[fold] = checkpoint
        reports.append({
            "fold": fold,
            "result": str(result_path.resolve()),
            "result_sha256": sha256(result_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
        })
    if set(checkpoints) != set(range(folds)):
        raise RuntimeError("point-OOF fold coverage is incomplete")
    held_files = {
        fold: sorted(name for name, value in assignments.items() if value == fold)
        for fold in range(folds)}
    for report in reports:
        report["held_files"] = held_files[int(report["fold"])]
    return checkpoints, sorted(reports, key=lambda item: int(item["fold"]))


def read_trajectory(real_dir: Path, row: Mapping[str, object]) -> torch.Tensor:
    root = real_dir.resolve()
    filename = str(row["file"])
    # The audited clean81 directory intentionally contains symlinks to the
    # immutable released H5 files.  Validate the basename before joining, but
    # do not reject that allowlisted symlink layout by resolving the file.
    if Path(filename).name != filename:
        raise ValueError(f"unsafe H5 filename {filename}")
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if handle["u"].shape != handle["v"].shape:
            raise ValueError(f"u/v shape mismatch in {filename}")
        if int(handle["u"].shape[0]) != int(row["frames"]):
            raise ValueError(f"frame count changed in {filename}")
        if int(np.asarray(handle["re"])) != int(row["re"]):
            raise ValueError(f"Re changed in {filename}")
        if int(np.asarray(handle["aoa"])) != int(row["aoa"]):
            raise ValueError(f"AoA changed in {filename}")
        u = np.asarray(handle["u"][:, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle["v"][:, ::2, ::2], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(np.stack((u, v), axis=-1)))


def uvp_windows(trajectory: torch.Tensor, starts: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    windows = torch.stack([trajectory[start:start + 40] for start in starts])
    pressure = torch.zeros((*windows.shape[:-1], 1), dtype=windows.dtype)
    values = torch.cat((windows, pressure), dim=-1)
    return values[:, :20], values[:, 20:]


def predict_fold(
        checkpoint: Path, rows: Sequence[Mapping[str, object]],
        positions: Mapping[tuple[str, int], int], real_dir: Path,
        residuals: torch.Tensor, factors: torch.Tensor,
        history_hashes: list[str | None], target_hashes: list[str | None],
        device: torch.device,
        batch_size: int) -> int:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalizer = transfer.checkpoint_normalizer(payload, device)
    model = transfer.load_model(payload, device)
    written = 0
    with torch.inference_mode():
        for row in rows:
            starts = [int(value) for value in row["val_starts"]]
            history, target = uvp_windows(
                read_trajectory(real_dir, row), starts)
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset:offset + batch_size]
                batch_history = history[offset:offset + batch_size].to(device)
                batch_target = target[offset:offset + batch_size].to(device)
                mean_input = normalizer["mean_input"].view(1, 1, 1, 1, 3)
                std_input = normalizer["std_input"].view(1, 1, 1, 1, 3)
                prediction_norm = model((batch_history - mean_input) / std_input)
                prediction = (
                    prediction_norm
                    * normalizer["std_target"].view(1, 1, 1, 1, 3)
                    + normalizer["mean_target"].view(1, 1, 1, 1, 3)
                )[..., :2]
                truth = batch_target[..., :2]
                signed = (truth - prediction).to(dtype=torch.float16).cpu()
                factor = common.sample_sps_factor(prediction, truth).float().cpu()
                for local, start in enumerate(batch_starts):
                    position = positions[(str(row["file"]), start)]
                    if torch.isfinite(factors[position]):
                        raise RuntimeError("point-OOF cache position written twice")
                    residuals[position].copy_(signed[local])
                    factors[position] = factor[local]
                    history_hashes[position] = hashlib.sha256(
                        history[offset + local].numpy().tobytes(order="C")
                    ).hexdigest()
                    target_hashes[position] = hashlib.sha256(
                        target[offset + local].numpy().tobytes(order="C")
                    ).hexdigest()
                    written += 1
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument(
        "--fold-result", type=Path, action="append", required=True,
        help="Repeat once for each fold result.json.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = validate_manifest(json.loads(
        args.evaluation_manifest.read_text(encoding="utf-8")))
    assignments = condition_fold_assignments(rows, args.folds)
    checkpoints, fold_reports = validate_fold_results(
        rows, assignments, args.fold_result, args.folds)
    identities = [
        {
            "file": str(row["file"]), "start": int(start),
            "re": int(row["re"]), "aoa": int(row["aoa"]),
            "fold": assignments[str(row["file"])],
        }
        for row in rows for start in row["val_starts"]]
    positions = {
        (str(item["file"]), int(item["start"])): index
        for index, item in enumerate(identities)}
    if len(positions) != len(identities):
        raise RuntimeError("point-OOF cache identities are not unique")
    shape = (len(identities), 20, 32, 64, 2)
    residuals = torch.empty(shape, dtype=torch.float16)
    factors = torch.full((len(identities),), float("nan"), dtype=torch.float32)
    history_hashes: list[str | None] = [None] * len(identities)
    target_hashes: list[str | None] = [None] * len(identities)
    written = 0
    for fold in range(args.folds):
        fold_rows = [
            row for row in rows
            if assignments[str(row["file"])] == fold]
        written += predict_fold(
            checkpoints[fold], fold_rows, positions, args.real_dir,
            residuals, factors, history_hashes, target_hashes,
            device, args.batch_size)
    if written != len(identities) or not torch.isfinite(factors).all():
        raise RuntimeError("point-OOF cache did not write every sample exactly once")
    if any(value is None for value in history_hashes + target_hashes):
        raise RuntimeError("point-OOF identity hashes are incomplete")
    if not torch.isfinite(residuals).all():
        raise FloatingPointError("point-OOF residual cache is non-finite")
    residual_sha = hashlib.sha256(
        residuals.numpy().tobytes(order="C")).hexdigest()
    payload = {
        "schema": SCHEMA,
        "folds": args.folds,
        "fold_formula": FOLD_FORMULA,
        "evaluation_manifest": str(args.evaluation_manifest.resolve()),
        "evaluation_manifest_sha256": sha256(args.evaluation_manifest),
        "sample_identities": identities,
        "sample_identity_sha256": canonical_sha256(identities),
        "ordered_history_sample_sha256": history_hashes,
        "ordered_target_sample_sha256": target_hashes,
        "ordered_history_stream_sha256": canonical_sha256(history_hashes),
        "ordered_target_stream_sha256": canonical_sha256(target_hashes),
        "fold_results": fold_reports,
        "signed_residual_physical_uv": residuals,
        "signed_residual_physical_uv_sha256": residual_sha,
        "oof_point_factor": factors,
        "mean_absolute_residual": float(residuals.float().abs().mean()),
        "samples": len(identities),
        "submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    report = {
        key: value for key, value in payload.items()
        if not isinstance(value, torch.Tensor)}
    report.update({
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "cache_bytes": args.output.stat().st_size,
    })
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "event": "b210_xcb_point_oof_residual_cache_complete",
        "samples": report["samples"],
        "mean_absolute_residual": report["mean_absolute_residual"],
        "output": report["output"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
