#!/usr/bin/env python3
"""Frozen five-fold trajectory cross-fit plan for released RealPDE Real data.

The split is deliberately target-free and reuses the repository's established
condition-fold formula::

    (rank(Re) + 2 * rank(AoA)) % 5

Every trajectory is assigned as one indivisible group.  Unlike leave-one-AoA
out, every held fold and every fit partition contains all five released AoAs.
The module accepts only the SHA-bound clean81 training manifest and never opens
an H5 file or an evaluation manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import train_b210_full_real_point_oof_fold as whole_aoa
from train_uq_budget_reallocator import trajectory_folds


dr1 = whole_aoa.dr1
ft12 = whole_aoa.ft12

SCHEMA = "b210_same_aoa_trajectory_fold_plan_v1"
FOLDS = 5
FOLD_VALUES = tuple(range(FOLDS))
AOA_VALUES = whole_aoa.AOA_VALUES
FORMULA = "(rank(re)+2*rank(aoa)) % folds"
EXPECTED_ASSIGNMENT_SHA256 = (
    "4286ec7859334324b511878b7e4763433b9852ffccdfea3273c5ef658cf6497a")

EXPECTED_TOTAL = dict(whole_aoa.EXPECTED_TOTAL)
EXPECTED_FOLDS = {
    0: {
        "fit_trajectories": 65,
        "held_trajectories": 16,
        "normalizer_prefix_windows": 2_057,
        "full_optimizer_windows": 10_673,
        "fit_oof_windows": 2_701,
        "held_oof_windows": 640,
        "held_trajectories_by_aoa": {0: 3, 5: 3, 10: 4, 15: 3, 20: 3},
    },
    1: {
        "fit_trajectories": 65,
        "held_trajectories": 16,
        "normalizer_prefix_windows": 2_032,
        "full_optimizer_windows": 10_546,
        "fit_oof_windows": 2_669,
        "held_oof_windows": 672,
        "held_trajectories_by_aoa": {0: 4, 5: 2, 10: 4, 15: 3, 20: 3},
    },
    2: {
        "fit_trajectories": 65,
        "held_trajectories": 16,
        "normalizer_prefix_windows": 2_032,
        "full_optimizer_windows": 10_546,
        "fit_oof_windows": 2_669,
        "held_oof_windows": 672,
        "held_trajectories_by_aoa": {0: 4, 5: 3, 10: 3, 15: 4, 20: 2},
    },
    3: {
        "fit_trajectories": 65,
        "held_trajectories": 16,
        "normalizer_prefix_windows": 2_055,
        "full_optimizer_windows": 10_663,
        "fit_oof_windows": 2_698,
        "held_oof_windows": 643,
        "held_trajectories_by_aoa": {0: 2, 5: 3, 10: 3, 15: 4, 20: 4},
    },
    4: {
        "fit_trajectories": 64,
        "held_trajectories": 17,
        "normalizer_prefix_windows": 2_000,
        "full_optimizer_windows": 10_380,
        "fit_oof_windows": 2_627,
        "held_oof_windows": 714,
        "held_trajectories_by_aoa": {0: 3, 5: 3, 10: 4, 15: 3, 20: 4},
    },
}


def _all_legal_windows(row: Mapping[str, object], stride: int) -> int:
    frames = int(row["frames"])
    if frames < 40 or stride <= 0:
        raise ValueError("invalid trajectory/window stride")
    return len(range(0, frames - 40 + 1, stride))


def validate_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    inventory = whole_aoa.validate_inventory(rows)
    if inventory != EXPECTED_TOTAL:
        raise RuntimeError("same-AoA plan received a different clean81 inventory")
    return inventory


def assignment_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Return one canonical row per trajectory, ordered like the manifest."""
    validate_rows(rows)
    temporary = []
    for row in rows:
        temporary.append({
            "file": str(row["file"]),
            "re": int(row["re"]),
            "aoa": int(row["aoa"]),
            # trajectory_folds expands one label per val start; one dummy start
            # exposes its trajectory-level assignment without reading data.
            "val_starts": [0],
        })
    labels, report = trajectory_folds(temporary, FOLDS)
    if (report.get("formula") != FORMULA or len(labels) != len(rows)
            or len(report.get("assignments", [])) != len(rows)):
        raise RuntimeError("established condition-fold implementation changed")
    result = []
    for row, label, assigned in zip(rows, labels, report["assignments"]):
        if (int(label) != int(assigned["fold"])
                or str(assigned["file"]) != str(row["file"])
                or int(assigned["re"]) != int(row["re"])
                or int(assigned["aoa"]) != int(row["aoa"])):
            raise RuntimeError("trajectory fold labels lost manifest alignment")
        result.append({
            "file": str(row["file"]),
            "re": int(row["re"]),
            "aoa": int(row["aoa"]),
            "fold": int(label),
        })
    if len({row["file"] for row in result}) != len(result):
        raise RuntimeError("trajectory fold plan contains duplicate files")
    return result


def assignment_map(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {row["file"]: int(row["fold"])
            for row in assignment_rows(rows)}


def assignment_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    return dr1.canonical_sha256(assignment_rows(rows))


def fold_counts(
        rows: Sequence[Mapping[str, object]], assignments: Mapping[str, int],
        fold: int) -> dict:
    if fold not in FOLD_VALUES:
        raise ValueError(f"unsupported held fold {fold}")
    held = [row for row in rows
            if int(assignments[str(row["file"])]) == fold]
    fit = [row for row in rows
           if int(assignments[str(row["file"])]) != fold]
    held_by_aoa = dict(sorted(Counter(int(row["aoa"]) for row in held).items()))
    observed = {
        "fit_trajectories": len(fit),
        "held_trajectories": len(held),
        "normalizer_prefix_windows": sum(
            len(row["train_starts"]) for row in fit),
        "full_optimizer_windows": sum(
            _all_legal_windows(row, 5) for row in fit),
        "fit_oof_windows": sum(
            _all_legal_windows(row, 20) for row in fit),
        "held_oof_windows": sum(
            _all_legal_windows(row, 20) for row in held),
        "held_trajectories_by_aoa": held_by_aoa,
    }
    if observed != EXPECTED_FOLDS[fold]:
        raise RuntimeError(f"same-AoA fold {fold} counts changed: {observed}")
    if set(held_by_aoa) != set(AOA_VALUES):
        raise RuntimeError("a same-AoA held fold no longer covers all five AoAs")
    if ({str(row["file"]) for row in held}
            .intersection(str(row["file"]) for row in fit)):
        raise RuntimeError("fit/held trajectory groups overlap")
    return observed


def build_plan(rows: Sequence[Mapping[str, object]]) -> dict:
    inventory = validate_rows(rows)
    assignments = assignment_rows(rows)
    digest = dr1.canonical_sha256(assignments)
    if digest != EXPECTED_ASSIGNMENT_SHA256:
        raise RuntimeError("same-AoA trajectory assignment digest changed")
    mapping = {row["file"]: int(row["fold"]) for row in assignments}
    folds = []
    for fold in FOLD_VALUES:
        counts = fold_counts(rows, mapping, fold)
        held = [row for row in assignments if int(row["fold"]) == fold]
        folds.append({
            "fold": fold,
            "counts": counts,
            "held_files": [row["file"] for row in held],
            "held_assignment_sha256": dr1.canonical_sha256(held),
        })
    if sorted(row["file"] for row in assignments) != sorted(mapping):
        raise RuntimeError("same-AoA fold coverage is incomplete")
    return {
        "schema": SCHEMA,
        "folds": FOLDS,
        "fold_values": list(FOLD_VALUES),
        "aoa_values": list(AOA_VALUES),
        "formula": FORMULA,
        "group_unit": "complete released-Real trajectory",
        "stratification": "global Re rank plus AoA-rank offset",
        "every_fold_covers_all_aoas": True,
        "trajectory_disjoint": True,
        "target_values_used": False,
        "inventory": inventory,
        "assignments": assignments,
        "assignment_sha256": digest,
        "fold_reports": folds,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    plan = build_plan(rows)
    plan["manifest"] = manifest_report
    plan["manifest_file_sha256"] = ft12.FORMAL_MANIFEST_FILE_SHA256
    sealed = whole_aoa.seal_json_payload(plan)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        dr1.write_json_atomic(args.output, sealed)
        whole_aoa.verify_json_payload(json.loads(
            args.output.read_text(encoding="utf-8")))
    print(json.dumps({
        "event": "b210_same_aoa_trajectory_fold_plan",
        "assignment_sha256": plan["assignment_sha256"],
        "fold_counts": [row["counts"] for row in plan["fold_reports"]],
        "output": None if args.output is None else str(args.output.resolve()),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
