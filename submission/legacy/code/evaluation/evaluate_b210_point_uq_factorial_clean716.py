#!/usr/bin/env python3
"""Replay a preregistered B210 point/UQ archive factorial on exact clean716."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (
    REPO_ROOT / "code",
    REPO_ROOT / "code/training",
    REPO_ROOT / "code/evaluation",
    REPO_ROOT / "code/submissions",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import verify_b210_clean716_int8_package as verifier  # noqa: E402
import build_track1_b210_strict_point_uq_factorial as strict_factorial  # noqa: E402


EXPECTED_ARMS = ("AA", "DD", "DA", "AD")
POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
}
NON_TIME_WEIGHTS = {**POINT_WEIGHTS, "native_sps_score": 0.25}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="label=/absolute/path.zip; provide AA, DD, DA and AD exactly once",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/"
            "runtime_train_manifest.json"
        ),
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=Path("/root/autodl-tmp/realpde/data/real"),
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("/root/autodl-tmp/b210_point_uq_factorial_extract"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_arms(rows: Sequence[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for row in rows:
        label, separator, value = row.partition("=")
        if separator != "=" or label not in EXPECTED_ARMS or not value:
            raise ValueError(f"invalid factorial arm: {row}")
        if label in arms:
            raise ValueError(f"duplicate factorial arm: {label}")
        arms[label] = Path(value).resolve()
    if tuple(sorted(arms)) != tuple(sorted(EXPECTED_ARMS)):
        raise ValueError(f"factorial requires exactly {EXPECTED_ARMS}")
    return arms


def runtime_source_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for name in verifier.EXPECTED_MEMBERS[:-1]:
            digest.update(name.encode())
            digest.update(archive.read(name))
    return digest.hexdigest()


def weighted(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return float(sum(weights[key] * float(metrics[key]) for key in weights))


def metric_delta(after: Mapping[str, float], before: Mapping[str, float]) -> dict:
    keys = tuple(NON_TIME_WEIGHTS)
    row = {key: float(after[key]) - float(before[key]) for key in keys}
    row["point65_delta"] = weighted(after, POINT_WEIGHTS) - weighted(
        before, POINT_WEIGHTS
    )
    row["non_time_delta"] = weighted(after, NON_TIME_WEIGHTS) - weighted(
        before, NON_TIME_WEIGHTS
    )
    return row


@torch.inference_mode()
def replay_arm(submission, dataset, metadata, device: torch.device) -> dict:
    tagged = verifier.ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged,
        batch_size=verifier.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    pooled = verifier.smu1.new_metric_state()
    by_aoa = {
        aoa: verifier.smu1.new_metric_state() for aoa in verifier.dr1.AOA_VALUES
    }
    point_digest = hashlib.sha256()
    lower_digest = hashlib.sha256()
    upper_digest = hashlib.sha256()
    samples = 0
    predict_seconds = 0.0
    for history, target, trajectory_ids, _sample_indices in loader:
        raw = np.ascontiguousarray(history.numpy())
        before = raw.copy()
        started = time.perf_counter()
        result = submission.predict(raw, metadata={})
        predict_seconds += time.perf_counter() - started
        if not np.array_equal(raw, before):
            raise RuntimeError("submission mutated clean716 input")
        if not isinstance(result, Mapping) or set(result) != {
            "prediction",
            "lower",
            "upper",
        }:
            raise ValueError("submission did not return point/lower/upper")
        prediction = np.asarray(result["prediction"])
        lower = np.asarray(result["lower"])
        upper = np.asarray(result["upper"])
        expected = tuple(target.shape)
        for label, value in (
            ("prediction", prediction),
            ("lower", lower),
            ("upper", upper),
        ):
            if (
                value.shape != expected
                or value.dtype != np.float32
                or not np.all(np.isfinite(value))
            ):
                raise FloatingPointError(f"invalid {label} output")
        if np.any(lower > upper):
            raise FloatingPointError("lower exceeds upper")
        if any(
            np.count_nonzero(value[..., 2])
            for value in (prediction, lower, upper)
        ):
            raise RuntimeError("pressure output is not exactly zero")

        prediction = np.ascontiguousarray(prediction)
        lower = np.ascontiguousarray(lower)
        upper = np.ascontiguousarray(upper)
        point_digest.update(prediction.tobytes())
        lower_digest.update(lower.tobytes())
        upper_digest.update(upper.tobytes())

        prediction_tensor = torch.from_numpy(prediction)
        truth = target.float()
        lower_extent = torch.from_numpy(
            np.ascontiguousarray(prediction[..., :2] - lower[..., :2])
        )
        upper_extent = torch.from_numpy(
            np.ascontiguousarray(upper[..., :2] - prediction[..., :2])
        )
        verifier.smu1.update_metric_state(
            pooled, prediction_tensor, truth, lower_extent, upper_extent
        )
        labels = [int(metadata[int(value)]["aoa"]) for value in trajectory_ids]
        for aoa in sorted(set(labels)):
            mask = torch.as_tensor([value == aoa for value in labels])
            verifier.smu1.update_metric_state(
                by_aoa[aoa],
                prediction_tensor[mask],
                truth[mask],
                lower_extent[mask],
                upper_extent[mask],
            )
        samples += len(prediction)

    if samples != verifier.uq_release.EXPECTED_TAIL_WINDOWS:
        raise RuntimeError(f"factorial replay consumed {samples} samples")
    metrics = verifier.smu1.finalize_metric_state(pooled)
    return {
        "metrics": metrics,
        "metric_state": pooled,
        "by_aoa": {
            str(aoa): verifier.smu1.finalize_metric_state(state)
            for aoa, state in by_aoa.items()
        },
        "point_prediction_sha256": point_digest.hexdigest(),
        "lower_sha256": lower_digest.hexdigest(),
        "upper_sha256": upper_digest.hexdigest(),
        "samples": samples,
        "predict_seconds": predict_seconds,
        "device": str(device),
    }


def compare_factorial(results: Mapping[str, Mapping[str, object]]) -> dict:
    dd = results["DD"]
    da = results["DA"]
    aa = results["AA"]
    ad = results["AD"]
    da_dd = metric_delta(da["metrics"], dd["metrics"])
    ad_aa = metric_delta(ad["metrics"], aa["metrics"])
    aoa_deltas = {
        aoa: float(da["by_aoa"][aoa]["native_sps_score"])
        - float(dd["by_aoa"][aoa]["native_sps_score"])
        for aoa in dd["by_aoa"]
    }
    point_identity = da["point_prediction_sha256"] == dd["point_prediction_sha256"]
    point_state_identity = verifier.uq_release.point_state_equal(
        dd["metric_state"], da["metric_state"]
    )
    runtime_identity = len(
        {results[arm]["runtime_source_sha256"] for arm in EXPECTED_ARMS}
    ) == 1
    conditions = {
        "da_point_bitwise_equal_dd": point_identity,
        "da_point_metric_state_equal_dd": point_state_identity,
        "da_sps_delta_ge_0_10": da_dd["native_sps_score"] >= 0.10,
        "da_sps_ge_aa": float(da["metrics"]["native_sps_score"])
        >= float(aa["metrics"]["native_sps_score"]),
        "da_positive_aoas_ge_4": sum(value >= 0.0 for value in aoa_deltas.values())
        >= 4,
        "runtime_sources_identical": runtime_identity,
    }
    # Explicit responsibility effects.  These names state the source change
    # directly and avoid interpreting DA-DD (old UQ minus fresh UQ) backwards.
    effects = {
        "new_point_given_old_uq": metric_delta(
            da["metrics"], aa["metrics"]
        ),
        "new_point_given_fresh_uq": metric_delta(
            dd["metrics"], ad["metrics"]
        ),
        "fresh_uq_given_old_point": metric_delta(
            ad["metrics"], aa["metrics"]
        ),
        "fresh_uq_given_new_point": metric_delta(
            dd["metrics"], da["metrics"]
        ),
    }
    interaction = {
        key: (
            float(dd["metrics"][key])
            - float(da["metrics"][key])
            - float(ad["metrics"][key])
            + float(aa["metrics"][key])
        )
        for key in NON_TIME_WEIGHTS
    }
    interaction["point65_delta"] = (
        effects["new_point_given_fresh_uq"]["point65_delta"]
        - effects["new_point_given_old_uq"]["point65_delta"]
    )
    interaction["non_time_delta"] = (
        effects["new_point_given_fresh_uq"]["non_time_delta"]
        - effects["new_point_given_old_uq"]["non_time_delta"]
    )
    return {
        "DA_minus_DD": da_dd,
        "AD_minus_AA": ad_aa,
        "DA_minus_DD_by_aoa_sps": aoa_deltas,
        "point_prediction_identity": {
            "DA_equals_DD": point_identity,
            "AD_equals_AA": ad["point_prediction_sha256"]
            == aa["point_prediction_sha256"],
        },
        "responsibility_effects": effects,
        "point_uq_interaction": interaction,
        "gate": {"conditions": conditions, "pass": all(conditions.values())},
    }


def main() -> None:
    args = parse_args()
    arms = parse_arms(args.arm)
    # This check happens before clean716 is opened.  In particular, passing the
    # old whole archive twice and the new whole archive twice is not a 2x2
    # factorial: it violates all four point/UQ partition identities.
    archive_factorial_audit = strict_factorial.audit_factorial_archives(arms)
    if not archive_factorial_audit["strict_factorial"]:
        raise RuntimeError(
            "archives are renamed whole models, not a strict point x UQ factorial: "
            + json.dumps(archive_factorial_audit, sort_keys=True)
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    dataset, metadata, windows, data_report = verifier.load_clean716(
        args.real_dir, args.clean_manifest
    )
    if len(windows) != verifier.uq_release.EXPECTED_TAIL_WINDOWS:
        raise RuntimeError("not exact clean716")

    results = {}
    for arm in EXPECTED_ARMS:
        archive = arms[arm]
        extraction = args.temp_root / arm
        if extraction.exists():
            shutil.rmtree(extraction)
        verifier.safe_extract_exact(archive, extraction)
        submission, module_sources = verifier.load_packaged_submission(extraction)
        row = replay_arm(submission, dataset, metadata, device)
        row.update(
            {
                "archive": {
                    "path": str(archive),
                    "sha256": verifier.dr1.sha256(archive),
                    "bytes": archive.stat().st_size,
                },
                "runtime_source_sha256": runtime_source_sha(archive),
                "module_sources": module_sources,
            }
        )
        results[arm] = row
        verifier.dr1.write_json_atomic(args.output_dir / f"{arm}.json", row)
        print(
            json.dumps(
                {"event": "factorial_arm_complete", "arm": arm, **row["metrics"]}
            ),
            flush=True,
        )
        verifier._purge_packaged_modules()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = compare_factorial(results)
    summary = {
        "schema": "b210_point_uq_factorial_clean716_v1",
        "evidence_class": "exact clean716 local diagnostic; not official score",
        "data": data_report,
        "archive_factorial_audit": archive_factorial_audit,
        "arms": results,
        "comparison": comparison,
        "decision": (
            "PASS_DIRECT_POINT_UQ_MODULAR_COMPOSITION"
            if comparison["gate"]["pass"]
            else "STOP_DIRECT_POINT_UQ_TENSOR_TRANSPLANT"
        ),
        "submission_authorized": False,
    }
    verifier.dr1.write_json_atomic(args.summary, summary)
    print(
        json.dumps(
            {
                "event": "factorial_complete",
                "decision": summary["decision"],
                "comparison": comparison,
                "summary": str(args.summary),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
