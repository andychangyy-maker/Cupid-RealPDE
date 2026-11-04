#!/usr/bin/env python3
"""Evaluate A/B/mean-expert blends using condition-fold UQ outputs.

Both interval models are reconstructed from the point checkpoint used by
their respective OOF run, then receive only the UQ state trained without the
current condition fold.  The mean expert is a point model trained on the
disjoint point-training split, so it can be shared across folds.

The stored ``fold_XX_best`` states excluded the held fold from gradient
training, but their checkpoint iteration was selected on that held fold.
They are therefore fold-excluded-training states, not strictly fold-safe
teachers for a future distillation claim.

When ``--fixed-bounds-candidate`` is set, the selected candidate supplies
absolute lower/upper bounds for every point candidate.  This is a
post-selection diagnostic on the same tail set, not a new independent OOF
estimate of blend-weight generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from evaluate_flow_backbone_blend import collect, load_model, score  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


S1_BASELINE_ZIP_SHA256 = (
    "4131dea445da0747dc1cdc49548d9afe7bc726620d6d4f62d925c79f66e09803")
S1_SPLIT_MANIFEST_SHA256 = (
    "f2637e1cc7cc0c32dddc00f53763abfd383ac73d822fd29d45caf11f8f73040f")
S1_REAL_STATS_SHA256 = (
    "bae5235f73431ac7d76e6bdcea480b8ca9fda0ac01e57ab26af66bad20dc3b40")
S1_SCORER_SHA256 = (
    "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39")
S1_CANDIDATES = (
    ("P0", 0.625, 0.25),
    ("P1", 0.500, 0.50),
    ("P2", 0.250, 0.50),
)
S1_FOLD_COUNTS = (137, 144, 144, 147, 153)
S1_SELECTION_ORDER = ("P1", "P2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint-a", type=Path, required=True)
    parser.add_argument("--oof-dir-a", type=Path, required=True)
    parser.add_argument("--point-checkpoint-b", type=Path, required=True)
    parser.add_argument("--oof-dir-b", type=Path, required=True)
    parser.add_argument("--mean-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights-b", type=float, nargs="+")
    parser.add_argument(
        "--mean-expert-weights", type=float, nargs="+")
    parser.add_argument(
        "--point-candidate", action="append", default=[], metavar="NAME:B:C",
        help=(
            "paired point candidate; repeat for each preregistered NAME, "
            "B blend weight and C mean-expert weight"))
    parser.add_argument(
        "--fixed-bounds-candidate",
        help="candidate name whose absolute lower/upper are reused verbatim")
    parser.add_argument("--baseline-zip", type=Path)
    parser.add_argument("--expected-baseline-zip-sha256")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--expected-split-manifest-sha256")
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--freeze-artifact-manifest", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_weights(values: list[float], name: str) -> None:
    if (not values or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values)):
        raise ValueError(f"{name} must contain finite values in [0,1]")


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def array_identity(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": value.nbytes,
        "sha256": digest,
    }


def read_blend_config(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        try:
            payload = archive.read("blend_config.json")
        except KeyError as error:
            raise ValueError("baseline ZIP has no blend_config.json") from error
    return json.loads(payload.decode("utf-8"))


def parse_point_candidate(value: str) -> dict[str, float | str]:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise ValueError(
            f"invalid point candidate {value!r}; expected NAME:B:C")
    try:
        weight_b, mean_weight = map(float, parts[1:])
    except ValueError as error:
        raise ValueError(
            f"invalid point candidate {value!r}; B and C must be floats"
        ) from error
    validate_weights([weight_b], f"{parts[0]} B")
    validate_weights([mean_weight], f"{parts[0]} C")
    return {
        "name": parts[0], "weight_b": weight_b,
        "mean_expert_weight_c": mean_weight,
    }


def resolve_candidates(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.point_candidate:
        if args.weights_b is not None or args.mean_expert_weights is not None:
            raise ValueError(
                "point-candidate cannot be combined with the Cartesian grid")
        candidates = [parse_point_candidate(value)
                      for value in args.point_candidate]
    else:
        if args.weights_b is None or args.mean_expert_weights is None:
            raise ValueError(
                "provide point-candidate entries or both grid weight lists")
        validate_weights(args.weights_b, "weights-b")
        validate_weights(args.mean_expert_weights, "mean-expert-weights")
        candidates = [
            {
                "name": f"b={weight_b:.5f}/mean_c={mean_weight:.5f}",
                "weight_b": weight_b,
                "mean_expert_weight_c": mean_weight,
            }
            for weight_b in args.weights_b
            for mean_weight in args.mean_expert_weights
        ]
    names = [candidate["name"] for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names must be unique: {names}")
    if (args.fixed_bounds_candidate is not None
            and args.fixed_bounds_candidate not in names):
        raise ValueError(
            "fixed-bounds-candidate must name one point candidate")
    return candidates


def validate_s1_contract(args: argparse.Namespace,
                         candidates: list[dict[str, object]]) -> None:
    if args.fixed_bounds_candidate is None:
        return
    actual = tuple(
        (candidate["name"], candidate["weight_b"],
         candidate["mean_expert_weight_c"])
        for candidate in candidates)
    if actual != S1_CANDIDATES:
        raise ValueError(
            f"fixed-bounds S1 requires exact candidates {S1_CANDIDATES}")
    if args.fixed_bounds_candidate != "P0":
        raise ValueError("fixed-bounds S1 control must be P0")
    if (args.folds != 5 or args.window_stride != 20
            or args.val_fraction != 0.2 or args.max_files != 0):
        raise ValueError(
            "fixed-bounds S1 requires folds=5, stride=20, "
            "val-fraction=0.2 and max-files=0")
    if not str(args.device).startswith("cuda"):
        if args.freeze_artifact_manifest is None:
            raise ValueError("fixed-bounds S1 scoring must run on CUDA")
    if (args.baseline_zip is None
            or args.expected_baseline_zip_sha256 is None
            or args.split_manifest is None
            or args.expected_split_manifest_sha256 is None):
        raise ValueError(
            "fixed-bounds S1 requires frozen baseline ZIP and split manifest")
    if args.expected_baseline_zip_sha256.lower() != S1_BASELINE_ZIP_SHA256:
        raise ValueError("unexpected S1 baseline ZIP SHA-256 contract")
    if (args.expected_split_manifest_sha256.lower()
            != S1_SPLIT_MANIFEST_SHA256):
        raise ValueError("unexpected S1 split manifest SHA-256 contract")
    if ((args.artifact_manifest is None)
            == (args.freeze_artifact_manifest is None)):
        raise ValueError(
            "choose exactly one of artifact-manifest for scoring or "
            "freeze-artifact-manifest for S0")


def validate_s1_split(runtime_manifest: list[dict],
                      frozen_manifest: list[dict], tail_windows: int,
                      sample_folds: list[int]) -> None:
    if runtime_manifest != frozen_manifest:
        raise RuntimeError("runtime split differs from the frozen manifest")
    fold_counts = tuple(
        sample_folds.count(fold) for fold in range(len(S1_FOLD_COUNTS)))
    if tail_windows != 725 or fold_counts != S1_FOLD_COUNTS:
        raise RuntimeError(
            f"unexpected S1 tail/fold counts: {tail_windows}, {fold_counts}")


def s1_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "baseline_zip": args.baseline_zip,
        "split_manifest": args.split_manifest,
        "real_stats": args.real_stats,
        "official_scorer": (
            REPO_ROOT
            / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"),
        "evaluator_source": Path(__file__),
        "point_checkpoint_a": args.point_checkpoint_a,
        "point_checkpoint_b": args.point_checkpoint_b,
        "mean_expert_checkpoint": args.mean_expert_checkpoint,
        "oof_summary_a": args.oof_dir_a / "summary.json",
        "oof_summary_b": args.oof_dir_b / "summary.json",
    }
    for fold in range(5):
        paths[f"oof_a_fold_{fold:02d}"] = (
            args.oof_dir_a / f"fold_{fold:02d}_best.pth")
        paths[f"oof_b_fold_{fold:02d}"] = (
            args.oof_dir_b / f"fold_{fold:02d}_best.pth")
    return paths


def artifact_identities(paths: dict[str, Path]) -> dict[str, dict]:
    missing = [f"{name}={path}" for name, path in paths.items()
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing S1 artifacts: {missing}")
    return {name: file_identity(path) for name, path in paths.items()}


def write_artifact_manifest(path: Path,
                            identities: dict[str, dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {path}")
    payload = {
        "schema_version": 1,
        "experiment": "track1_s1_point_bounds_decoupling",
        "candidate_contract": [list(row) for row in S1_CANDIDATES],
        "artifacts": identities,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")


def verify_artifact_manifest(path: Path,
                             identities: dict[str, dict]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen artifact manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != 1
            or payload.get("experiment")
            != "track1_s1_point_bounds_decoupling"
            or payload.get("candidate_contract")
            != [list(row) for row in S1_CANDIDATES]
            or payload.get("artifacts") != identities):
        raise ValueError("current artifacts differ from the frozen manifest")
    return file_identity(path)


def validate_oof_summary(path: Path, point_checkpoint: Path,
                         fold_manifest: dict) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold_ids = [row.get("fold") for row in payload.get("folds", [])]
    if (payload.get("folds_requested") != 5
            or payload.get("folds_completed") != 5
            or payload.get("tail_windows") != 725
            or payload.get("fold_manifest") != fold_manifest
            or fold_ids != list(range(5))
            or Path(str(payload.get("point_checkpoint", ""))).name
            != point_checkpoint.name):
        raise ValueError(f"OOF summary contract mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "point_checkpoint": payload["point_checkpoint"],
        "folds_requested": payload["folds_requested"],
        "folds_completed": payload["folds_completed"],
        "tail_windows": payload["tail_windows"],
    }


def blend_candidate(arrays_a: dict[str, np.ndarray],
                    arrays_b: dict[str, np.ndarray],
                    prediction_c: np.ndarray, weight_b: float,
                    mean_weight: float) -> dict[str, np.ndarray]:
    base = {
        key: arrays_a[key] + np.float32(weight_b) * (
            arrays_b[key] - arrays_a[key])
        for key in arrays_a
    }
    mean_delta = (
        prediction_c.mean(axis=1, keepdims=True)
        - base["prediction"].mean(axis=1, keepdims=True))
    correction = np.float32(mean_weight) * mean_delta
    return {
        "prediction": base["prediction"] + correction,
        "lower": base["lower"] + correction,
        "upper": base["upper"] + correction,
    }


def blend_point_candidate(prediction_a: np.ndarray,
                          prediction_b: np.ndarray,
                          prediction_c: np.ndarray, weight_b: float,
                          mean_weight: float) -> np.ndarray:
    base = prediction_a + np.float32(weight_b) * (
        prediction_b - prediction_a)
    mean_delta = (
        prediction_c.mean(axis=1, keepdims=True)
        - base.mean(axis=1, keepdims=True))
    return base + np.float32(mean_weight) * mean_delta


def point_bounds_diagnostic(results: dict[str, dict],
                            control_name: str) -> dict:
    control = results[control_name]
    candidate_gates = {}
    for name, result in results.items():
        point_delta = result["point_score"] - control["point_score"]
        sps_delta = result["sps_score"] - control["sps_score"]
        fold_deltas = [
            fold_result["point_score"]
            - control["folds"][fold]["point_score"]
            for fold, fold_result in enumerate(result["folds"])
        ]
        coverage_unchanged = result["coverage"] == control["coverage"]
        width_unchanged = result["mean_width"] == control["mean_width"]
        passed = (
            name != control_name
            and result["bounds_equal_to_control"]
            and point_delta >= 0.008
            and sps_delta >= 0.0
            and sum(delta >= 0.0 for delta in fold_deltas) >= 4
            and min(fold_deltas) >= -0.02
            and coverage_unchanged and width_unchanged)
        candidate_gates[name] = {
            "pooled_point_delta": point_delta,
            "pooled_sps_delta": sps_delta,
            "fold_point_deltas": fold_deltas,
            "nonnegative_folds": sum(
                delta >= 0.0 for delta in fold_deltas),
            "worst_fold_point_delta": min(fold_deltas),
            "coverage_unchanged": coverage_unchanged,
            "mean_width_unchanged": width_unchanged,
            "pass": passed,
        }
    passed_names = [
        name for name, gate in candidate_gates.items() if gate["pass"]]
    selected = None
    if passed_names:
        best_delta = max(
            candidate_gates[name]["pooled_point_delta"]
            for name in passed_names)
        near_best = [
            name for name in passed_names
            if best_delta - candidate_gates[name]["pooled_point_delta"] < 0.001
        ]
        order = {name: index for index, name in enumerate(S1_SELECTION_ORDER)}
        selected = max(
            near_best,
            key=lambda name: (
                candidate_gates[name]["pooled_sps_delta"],
                -order[name],
            ))
    return {
        "evidence_class": (
            "post-selection diagnostic on the same 725-tail set; "
            "not an independent OOF estimate of blend-weight selection"),
        "control": control_name,
        "gate": {
            "pooled_point_delta_min": 0.008,
            "pooled_sps_delta_min": 0.0,
            "nonnegative_folds_min": 4,
            "worst_fold_point_delta_min": -0.02,
            "bounds_array_equal_required": True,
            "coverage_and_mean_width_unchanged_required": True,
        },
        "selection_rule": (
            "highest pooled point delta; if candidates are within 0.001, "
            "highest pooled SPS delta; exact ties prefer P1 then P2"),
        "candidates": candidate_gates,
        "selected_point_teacher": selected,
        "decision": "PASS" if selected is not None else "STOP_KEEP_CONTROL",
    }


def load_fold_model(payload: dict, checkpoint_path: Path, expected_fold: int,
                    expected_base_checkpoint: Path, normalizer,
                    device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "ensemble_unet_uq_oof":
        raise ValueError(f"unexpected OOF architecture in {checkpoint_path}")
    if checkpoint.get("fold") != expected_fold:
        raise ValueError(
            f"{checkpoint_path} contains fold {checkpoint.get('fold')}, "
            f"expected {expected_fold}")
    stored_base = Path(str(checkpoint.get("base_checkpoint", ""))).name
    if stored_base != expected_base_checkpoint.name:
        raise ValueError(
            f"{checkpoint_path} base {stored_base!r} does not match "
            f"{expected_base_checkpoint.name!r}")
    config = checkpoint["model_config"]
    for key, value in payload["model_config"].items():
        if config.get(key) != value:
            raise ValueError(
                f"{checkpoint_path} base config differs at {key!r}")
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=False,
        add_flow_context=config.get("uq_flow_context", False),
        residual_quantile_intervals=config.get(
            "residual_quantile_intervals", False),
        utility_refinement=config.get("utility_refinement", False),
        utility_refinement_log_limit=config.get(
            "utility_refinement_log_limit", 0.75),
        utility_refinement_kernel=config.get(
            "utility_refinement_kernel", 1),
        utility_refinement_blocks=config.get(
            "utility_refinement_blocks", 0),
        temporal_uq_refinement=config.get(
            "temporal_uq_refinement", False),
        temporal_uq_channels=config.get("temporal_uq_channels", 24),
        temporal_uq_blocks=config.get("temporal_uq_blocks", 2),
        temporal_uq_kernel=config.get("temporal_uq_kernel", 3),
        temporal_uq_logit_limit=config.get(
            "temporal_uq_logit_limit", 1.5),
        coverage_calibration=config.get("coverage_calibration", False),
        coverage_calibration_kernel=config.get(
            "coverage_calibration_kernel", 3),
        coverage_calibration_blocks=config.get(
            "coverage_calibration_blocks", 1),
        coverage_calibration_logit_limit=config.get(
            "coverage_calibration_logit_limit", 2.0),
        coverage_calibration_mode=config.get(
            "coverage_calibration_mode", "independent"),
        coverage_calibration_temperature_limit=config.get(
            "coverage_calibration_temperature_limit", 0.75))
    uq_train.load_uq_state(model, checkpoint["uq_state_dict"])
    return model.eval(), {
        "checkpoint": str(checkpoint_path),
        "fold": checkpoint["fold"],
        "base_checkpoint": checkpoint["base_checkpoint"],
        "selected_iteration": checkpoint.get("selected_iteration"),
        "model_config": config,
    }


def append_distribution(storage: dict[str, list[np.ndarray]], values) -> None:
    for key, value in zip(("prediction", "lower", "upper"), values[:3]):
        storage[key].append(value)


def main() -> None:
    args = parse_args()
    candidates = resolve_candidates(args)
    validate_s1_contract(args, candidates)
    if args.freeze_artifact_manifest is None and args.output is None:
        raise ValueError("scoring requires --output")
    if args.folds < 2 or args.window_stride <= 0 or args.batch_size <= 0:
        raise ValueError("folds, stride and batch size must be positive")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val-fraction must lie in (0,0.5)")
    fold_inputs = (
        args.point_checkpoint_a, args.point_checkpoint_b,
        args.mean_expert_checkpoint,
        *(args.oof_dir_a / f"fold_{fold:02d}_best.pth"
          for fold in range(args.folds)),
        *(args.oof_dir_b / f"fold_{fold:02d}_best.pth"
          for fold in range(args.folds)),
    )
    baseline_identity = None
    split_identity = None
    artifact_manifest_identity = None
    oof_summary_validation = None
    if args.fixed_bounds_candidate is not None:
        paths = s1_artifact_paths(args)
        input_artifacts = artifact_identities(paths)
        baseline_identity = dict(input_artifacts["baseline_zip"])
        split_identity = input_artifacts["split_manifest"]
        if (baseline_identity["sha256"]
                != args.expected_baseline_zip_sha256.lower()):
            raise ValueError(
                "baseline ZIP SHA-256 does not match the frozen contract")
        baseline_config = read_blend_config(args.baseline_zip)
        if (baseline_config.get("weight_b") != S1_CANDIDATES[0][1]
                or baseline_config.get("mean_expert_weight")
                != S1_CANDIDATES[0][2]
                or baseline_config.get("checkpoints") != [
                    "model_a.pth", "model_b.pth", "model_c.pth"]):
            raise ValueError("baseline ZIP blend_config does not match P0")
        baseline_identity["blend_config"] = baseline_config
        split_identity = file_identity(args.split_manifest)
        if (split_identity["sha256"]
                != args.expected_split_manifest_sha256.lower()):
            raise ValueError(
                "split manifest SHA-256 does not match the frozen contract")
        if input_artifacts["real_stats"]["sha256"] != S1_REAL_STATS_SHA256:
            raise ValueError("real normalization stats differ from S1 contract")
        if input_artifacts["official_scorer"]["sha256"] != S1_SCORER_SHA256:
            raise ValueError("official scorer differs from S1 contract")
        frozen_manifest = json.loads(
            args.split_manifest.read_text(encoding="utf-8"))
        frozen_folds, frozen_fold_manifest = trajectory_folds(
            frozen_manifest, 5)
        validate_s1_split(
            frozen_manifest, frozen_manifest, 725, frozen_folds)
        oof_summary_validation = {
            "a": validate_oof_summary(
                args.oof_dir_a / "summary.json",
                args.point_checkpoint_a, frozen_fold_manifest),
            "b": validate_oof_summary(
                args.oof_dir_b / "summary.json",
                args.point_checkpoint_b, frozen_fold_manifest),
        }
        if args.freeze_artifact_manifest is not None:
            write_artifact_manifest(
                args.freeze_artifact_manifest, input_artifacts)
            print(json.dumps({
                "event": "s0_artifacts_frozen",
                "artifact_manifest": file_identity(
                    args.freeze_artifact_manifest),
                "artifacts": input_artifacts,
                "oof_summaries": oof_summary_validation,
            }), flush=True)
            return
        artifact_manifest_identity = verify_artifact_manifest(
            args.artifact_manifest, input_artifacts)
    else:
        required = (*fold_inputs, args.real_stats)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing OOF inputs: {missing}")
        input_artifacts = [file_identity(path) for path in required]

    device = torch.device(args.device)
    payload_a = torch.load(
        args.point_checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.point_checkpoint_b, map_location="cpu", weights_only=False)
    trajectories, metadata = common.repro.load_trajectories(
        args.real_dir, args.max_files)
    _point_windows, tail_windows, manifest = common.repro.make_split(
        metadata, args.window_stride, args.val_fraction)
    if args.fixed_bounds_candidate is not None:
        frozen_manifest = json.loads(
            args.split_manifest.read_text(encoding="utf-8"))
    dataset = common.repro.WindowDataset(trajectories, tail_windows)
    sample_folds, fold_manifest = trajectory_folds(manifest, args.folds)
    if args.fixed_bounds_candidate is not None:
        validate_s1_split(
            manifest, frozen_manifest, len(dataset), sample_folds)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    mean_expert, mean_metadata = load_model(
        args.mean_expert_checkpoint, normalizer, device)

    distributions_a = {key: [] for key in ("prediction", "lower", "upper")}
    distributions_b = {key: [] for key in ("prediction", "lower", "upper")}
    predictions_c = []
    targets = []
    fold_metadata = []
    fold_slices = []
    loader_args = argparse.Namespace(num_workers=args.num_workers)
    all_indices = list(range(len(dataset)))
    sample_offset = 0
    for fold in range(args.folds):
        indices = [index for index in all_indices if sample_folds[index] == fold]
        loader = uq_train.make_loader(
            dataset, indices, args.batch_size, False, loader_args, seed=0)
        path_a = args.oof_dir_a / f"fold_{fold:02d}_best.pth"
        path_b = args.oof_dir_b / f"fold_{fold:02d}_best.pth"
        model_a, metadata_a = load_fold_model(
            payload_a, path_a, fold, args.point_checkpoint_a,
            normalizer, device)
        values_a = collect(model_a, loader, normalizer, device)
        del model_a
        model_b, metadata_b = load_fold_model(
            payload_b, path_b, fold, args.point_checkpoint_b,
            normalizer, device)
        values_b = collect(model_b, loader, normalizer, device)
        del model_b
        values_c = collect(mean_expert, loader, normalizer, device)
        if (not np.array_equal(values_a[3], values_b[3])
                or not np.array_equal(values_a[3], values_c[3])):
            raise RuntimeError(f"target order changed in fold {fold}")
        append_distribution(distributions_a, values_a)
        append_distribution(distributions_b, values_b)
        predictions_c.append(values_c[0])
        targets.append(values_a[3])
        fold_samples = len(values_a[3])
        fold_slices.append((sample_offset, sample_offset + fold_samples))
        sample_offset += fold_samples
        fold_metadata.append({
            "fold": fold, "samples": fold_samples,
            "model_a": metadata_a, "model_b": metadata_b,
        })
        print(json.dumps({
            "event": "oof_collect", "fold": fold, "samples": len(indices),
        }), flush=True)
        del values_a, values_b, values_c
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del mean_expert, payload_a, payload_b, trajectories
    arrays_a = {key: np.concatenate(value)
                for key, value in distributions_a.items()}
    arrays_b = {key: np.concatenate(value)
                for key, value in distributions_b.items()}
    prediction_c = np.concatenate(predictions_c)
    target = np.concatenate(targets)
    fixed_bounds = None
    fixed_bounds_identity = None
    fixed_control_point_identity = None
    if args.fixed_bounds_candidate is not None:
        control_candidate = next(
            candidate for candidate in candidates
            if candidate["name"] == args.fixed_bounds_candidate)
        control_distribution = blend_candidate(
            arrays_a, arrays_b, prediction_c,
            control_candidate["weight_b"],
            control_candidate["mean_expert_weight_c"])
        fixed_bounds = {
            "lower": control_distribution["lower"],
            "upper": control_distribution["upper"],
        }
        fixed_control_point_identity = array_identity(
            control_distribution["prediction"])
        for value in fixed_bounds.values():
            value.setflags(write=False)
        fixed_bounds_identity = {
            key: array_identity(value) for key, value in fixed_bounds.items()
        }
        del control_distribution
    results = {}
    reference_tke = {}
    for candidate in candidates:
        name = candidate["name"]
        if fixed_bounds is None:
            values = blend_candidate(
                arrays_a, arrays_b, prediction_c,
                candidate["weight_b"],
                candidate["mean_expert_weight_c"])
            prediction = values["prediction"]
            lower, upper = values["lower"], values["upper"]
        else:
            prediction = blend_point_candidate(
                arrays_a["prediction"], arrays_b["prediction"], prediction_c,
                candidate["weight_b"],
                candidate["mean_expert_weight_c"])
            lower, upper = fixed_bounds["lower"], fixed_bounds["upper"]
            if (name == args.fixed_bounds_candidate
                    and array_identity(prediction)
                    != fixed_control_point_identity):
                raise RuntimeError(
                    "point-only P0 differs from the reconstructed control")
        metrics = score(prediction, lower, upper, target)
        metrics["non_time_mean"] = float(np.mean([
            metrics["rel_l2_score"], metrics["tke_score"],
            metrics["mvpe_score"], metrics["sps_score"],
        ]))
        weight_b = candidate["weight_b"]
        if weight_b in reference_tke:
            if abs(metrics["tke_score"] - reference_tke[weight_b]) > 2e-5:
                raise RuntimeError("mean-only correction changed OOF TKE")
        else:
            reference_tke[weight_b] = metrics["tke_score"]
        fold_results = []
        for fold, (start, stop) in enumerate(fold_slices):
            fold_results.append({
                "fold": fold,
                **score(
                    prediction[start:stop], lower[start:stop],
                    upper[start:stop], target[start:stop]),
            })
        results[name] = {
            "name": name,
            "weight_b": float(candidate["weight_b"]),
            "mean_expert_weight_c": float(
                candidate["mean_expert_weight_c"]),
            "bounds_equal_to_control": (
                None if fixed_bounds is None else bool(
                    np.array_equal(lower, fixed_bounds["lower"])
                    and np.array_equal(upper, fixed_bounds["upper"]))),
            **metrics,
            "folds": fold_results,
        }
        event = ("oof_blend" if fixed_bounds is None
                 else "point_bounds_candidate")
        print(json.dumps({"event": event, **results[name]}), flush=True)
        del prediction
        if fixed_bounds is None:
            del values

    diagnostic = None
    if fixed_bounds is not None:
        diagnostic = point_bounds_diagnostic(
            results, args.fixed_bounds_candidate)

    best_point = max(results, key=lambda key: results[key]["point_score"])
    best_non_time = max(
        results, key=lambda key: results[key]["non_time_mean"])
    report = {
        "protocol": (
            "Each A/B distribution uses the UQ state trained without its "
            "condition fold; C comes from the disjoint point-training split."),
        "tail_windows": len(dataset),
        "fold_manifest": fold_manifest,
        "folds": fold_metadata,
        "mean_expert": mean_metadata,
        "baseline_zip": baseline_identity,
        "split_manifest": split_identity,
        "artifact_manifest": artifact_manifest_identity,
        "oof_summary_validation": oof_summary_validation,
        "input_artifacts": input_artifacts,
        "evaluator_source": file_identity(Path(__file__)),
        "official_scorer": file_identity(
            REPO_ROOT
            / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"),
        "results": results,
    }
    if fixed_bounds is None:
        report.update({
            "evidence_class": "condition_fold_blend_grid",
            "best_point": results[best_point],
            "best_non_time": results[best_non_time],
        })
    else:
        report.update({
            "evidence_class": "post_selection_point_bounds_diagnostic",
            "protocol": (
                "A/B bounds are reconstructed with fold-held UQ states, but "
                "P0/P1/P2 weights were chosen after inspecting this same "
                "725-tail set. This is a post-selection diagnostic, not an "
                "independent estimate of blend-weight generalization."),
            "point_proxy_definition": (
                "0.5*rel_l2_score + 0.3*tke_score + 0.2*mvpe_score"),
            "bounds_source": "reconstructed_fold_excluded_training_P0",
            "checkpoint_selection_used_held_fold": True,
            "bounds_reuse": (
                "The exact same immutable P0 lower/upper arrays are passed "
                "to the official scorer for P0, P1 and P2."),
            "fixed_bounds_identity": fixed_bounds_identity,
            "fixed_control_point_identity": fixed_control_point_identity,
            "baseline_zip_role": "identity_only_not_bounds_source",
            "baseline_official_result": {
                "submission_id": 896362,
                "owner": "julius1",
                "rel_l2_score": 94.4944298385,
                "tke_score": 78.1610379906,
                "mvpe_score": 93.6908195850,
                "time_score": 88.3188958411,
                "sps_score": 42.2486244300,
                "final_score": 80.7522321802,
            },
            "decision": diagnostic,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if fixed_bounds is None:
        complete = {
            "event": "oof_complete", "best_point": report["best_point"],
            "best_non_time": report["best_non_time"],
        }
    else:
        complete = {
            "event": "point_bounds_complete",
            "decision": diagnostic["decision"],
            "selected_point_teacher": diagnostic["selected_point_teacher"],
        }
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
