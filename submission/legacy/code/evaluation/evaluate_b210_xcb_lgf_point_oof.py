#!/usr/bin/env python3
"""Aggregate paired baseline/LGF XCB point-OOF fold results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_smu1_paired_loao as smu1  # noqa: E402


SCHEMA = "b210_xcb_member_integrated_lgf_point_oof_comparison_v1"
POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
}


def official_point_delta(delta: Mapping[str, float]) -> float:
    return sum(POINT_WEIGHTS[key] * float(delta[key]) for key in POINT_WEIGHTS)


def comparison(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict:
    keys = (*POINT_WEIGHTS, "native_sps_score")
    delta = {key: float(candidate[key]) - float(baseline[key]) for key in keys}
    point = official_point_delta(delta)
    return {
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "delta": delta,
        "official_point_contribution_delta": point,
        "non_time_contribution_delta": point + 0.25 * delta["native_sps_score"],
    }


def validate_candidate_contract(value: Mapping[str, object]) -> None:
    expected = {
        "point_accuracy_weights": [46.0, 10.0, 9.0],
        "member_accuracy_weights": [46.0, 10.0, 9.0],
        "legacy_tke_gradient_floor": True,
        "legacy_tke_gradient_floor_on_members": True,
        "legacy_point_accuracy_weights": [5.0, 3.0, 2.0],
        "legacy_member_accuracy_weights": [5.0, 3.0, 2.0],
    }
    if dict(value) != expected:
        raise ValueError("candidate is not the frozen member-integrated LGF contract")


def parse_folds(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(set(values)) != len(values) or any(not 0 <= value < 5 for value in values):
        raise ValueError("fold list must contain unique values in [0,4]")
    return values


def read_fold(template: str, fold: int) -> dict:
    path = Path(template.format(fold=fold))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("held_fold", -1)) != fold:
        raise ValueError(f"held-fold identity changed in {path}")
    return payload


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    folds = parse_folds(args.folds)
    rows = []
    baseline_states = []
    candidate_states = []
    for fold in folds:
        baseline = read_fold(args.baseline_template, fold)
        candidate = read_fold(args.candidate_template, fold)
        if baseline.get("training_contract") != {}:
            raise ValueError("baseline point-OOF contract changed")
        validate_candidate_contract(candidate.get("training_contract", {}))
        if (not baseline["gate"]["pass"] or not candidate["gate"]["pass"]):
            raise RuntimeError("a point-OOF residual gate did not pass")
        baseline_held = baseline["held"]
        candidate_held = candidate["held"]
        if baseline_held["target_audit"] != candidate_held["target_audit"]:
            raise RuntimeError("paired point-OOF folds saw different held targets")
        baseline_states.append(baseline_held["fold_metric_state"])
        candidate_states.append(candidate_held["fold_metric_state"])
        rows.append({
            "fold": fold,
            "comparison": comparison(
                candidate_held["fold_metrics"], baseline_held["fold_metrics"]),
        })
    baseline_metrics = smu1.finalize_metric_state(
        smu1.merge_metric_states(baseline_states))
    candidate_metrics = smu1.finalize_metric_state(
        smu1.merge_metric_states(candidate_states))
    aggregate = comparison(candidate_metrics, baseline_metrics)
    thresholds = {
        "official_point_contribution_delta_min": 0.005,
        "non_time_contribution_delta_min": 0.0 if args.stage == "screen" else 0.005,
        "rel_l2_delta_min": -0.03,
        "tke_delta_min": -0.05,
        "mvpe_delta_min": -0.03,
        "positive_folds_min": 1 if args.stage == "screen" else 4,
    }
    positive = sum(
        row["comparison"]["official_point_contribution_delta"] > 0.0
        for row in rows)
    delta = aggregate["delta"]
    conditions = {
        "point_improves": aggregate["official_point_contribution_delta"]
        >= thresholds["official_point_contribution_delta_min"],
        "non_time_improves": aggregate["non_time_contribution_delta"]
        >= thresholds["non_time_contribution_delta_min"],
        "rel_floor": delta["rel_l2_score"] >= thresholds["rel_l2_delta_min"],
        "tke_floor": delta["tke_rel_l2_score"] >= thresholds["tke_delta_min"],
        "mvpe_floor": delta["mvpe_rel_l2_score"] >= thresholds["mvpe_delta_min"],
        "fold_consistency": positive >= thresholds["positive_folds_min"],
    }
    if any(not math.isfinite(float(value)) for value in (
            *delta.values(), aggregate["official_point_contribution_delta"],
            aggregate["non_time_contribution_delta"])):
        raise FloatingPointError("LGF OOF comparison is non-finite")
    report = {
        "schema": SCHEMA,
        "stage": args.stage,
        "folds": folds,
        "fold_rows": rows,
        "aggregate": aggregate,
        "positive_folds": positive,
        "gate": {
            "thresholds": thresholds,
            "conditions": conditions,
            "pass": all(conditions.values()),
        },
        "decision": (
            "GO_LGF_REMAINING_FOLDS" if args.stage == "screen" and all(conditions.values())
            else "PASS_XCB_MEMBER_INTEGRATED_LGF_POINT_OOF" if all(conditions.values())
            else "STOP_XCB_MEMBER_INTEGRATED_LGF_POINT_OOF"),
        "submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-template", required=True)
    parser.add_argument("--candidate-template", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--stage", choices=("screen", "full"), default="full")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({
        "event": "b210_xcb_member_integrated_lgf_point_oof_comparison",
        "stage": report["stage"],
        "aggregate": report["aggregate"],
        "gate": report["gate"],
        "decision": report["decision"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
