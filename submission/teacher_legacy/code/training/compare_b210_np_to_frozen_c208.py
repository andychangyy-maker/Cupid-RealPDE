#!/usr/bin/env python3
"""Compare fixed B210 Real-ft12 endpoints with frozen C208 MEF2 anchors.

This is a JSON-only, CPU-only evidence comparator.  It never loads a model,
selects an endpoint, accesses fixed725/clean716, or produces a submission.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from numbers import Real
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for import_path in (CODE_ROOT, TRAINING_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_b210_np_functional_real_ft12_loao as b210  # noqa: E402
import train_c208_mef2_paired_joint_loao as mef2  # noqa: E402


SCHEMA = "b210_np_vs_frozen_c208_two_fold_v1"
FOLD_ORDER = b210.FOLD_ORDER
CANDIDATE_ENDPOINT = b210.PRIMARY_ENDPOINT
ANCHOR_ENDPOINT = "control_ema"
FROZEN_ANCHOR_SHA256 = {
    10: "42368526e4380aeb20f02c175725d4d64acd7ec54ff0f240660436b438a1ef70",
    0: "e4eb94eb2cf6a012a0afd468c148dedc5d0d4b5edd1bf6cdb49ac9d448b33c409",
}
METRIC_STATE_KEYS = frozenset(b210.smu1.new_metric_state())
AUDIT_KEYS = frozenset({
    "ordered_raw_target_sha256",
    "ordered_scored_mask_sha256",
    "samples",
    "scored_elements",
})


def _read_json_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON evidence must be a mapping: {path}")
    return value


def _load_b210_rows(pairs: Sequence[Sequence[str]]) -> tuple[list[dict], list[dict]]:
    if len(pairs) != len(FOLD_ORDER):
        raise ValueError("comparison requires exactly held10 and held0 B210 rows")
    rows = []
    sources = []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected = b210.dr1.normalized_sha256(raw_sha, "B210 fold SHA")
        actual = b210.dr1.sha256(path)
        if actual != expected:
            raise RuntimeError(f"B210 fold result SHA changed: {path}")
        rows.append(_read_json_mapping(path))
        sources.append({"path": str(path.resolve()), "sha256": actual})
    return rows, sources


def _load_anchor_rows(paths: Sequence[str]) -> tuple[list[dict], list[dict]]:
    if len(paths) != len(FOLD_ORDER):
        raise ValueError("comparison requires exactly held10 and held0 C208 anchors")
    rows = []
    sources = []
    for raw_path, expected_held in zip(paths, FOLD_ORDER, strict=True):
        path = Path(raw_path)
        actual = b210.dr1.sha256(path)
        expected = FROZEN_ANCHOR_SHA256[expected_held]
        if actual != expected:
            raise RuntimeError(
                f"frozen C208 held{expected_held} anchor SHA changed: {path}")
        rows.append(_read_json_mapping(path))
        sources.append({
            "path": str(path.resolve()),
            "sha256": actual,
            "held_aoa": expected_held,
            "endpoint": ANCHOR_ENDPOINT,
        })
    return rows, sources


def _sha256_text(value: object, label: str) -> str:
    text = str(value)
    return b210.dr1.normalized_sha256(text, label)


def _validated_metric_state(value: object, samples: int, label: str) -> dict:
    if not isinstance(value, Mapping) or set(value) != METRIC_STATE_KEYS:
        raise RuntimeError(f"{label} metric_state schema changed")
    state = dict(value)
    if (any(isinstance(item, bool) or not isinstance(item, Real)
            for item in state.values())
            or any(not isinstance(state[key], int)
                   for key in ("samples", "sps_scored", "sps_covered"))):
        raise TypeError(f"{label} metric_state contains non-numeric values")
    if state["samples"] != samples:
        raise RuntimeError(f"{label} metric_state samples changed")
    if any(not math.isfinite(float(item)) for item in state.values()):
        raise FloatingPointError(f"{label} metric_state is non-finite")
    if (state["samples"] <= 0 or state["sps_scored"] <= 0
            or state["sps_covered"] < 0
            or state["sps_covered"] > state["sps_scored"]):
        raise RuntimeError(f"{label} metric_state counts are invalid")
    return state


def _validated_audit(value: object, samples: int, label: str) -> dict:
    if not isinstance(value, Mapping) or set(value) != AUDIT_KEYS:
        raise RuntimeError(f"{label} target/mask audit schema changed")
    audit = dict(value)
    if (not isinstance(audit["samples"], int)
            or isinstance(audit["samples"], bool)
            or not isinstance(audit["scored_elements"], int)
            or isinstance(audit["scored_elements"], bool)):
        raise TypeError(f"{label} target/mask audit counts are not integers")
    if audit["samples"] != samples:
        raise RuntimeError(f"{label} target/mask audit samples changed")
    _sha256_text(audit["ordered_raw_target_sha256"], f"{label} target SHA")
    _sha256_text(audit["ordered_scored_mask_sha256"], f"{label} mask SHA")
    if int(audit["scored_elements"]) <= 0:
        raise RuntimeError(f"{label} scored element count is empty")
    return audit


def _endpoint_payload(row: Mapping, endpoint: str, label: str) -> tuple[dict, dict, dict]:
    report = row.get("endpoints", {}).get(endpoint)
    if not isinstance(report, Mapping):
        raise RuntimeError(f"{label} fixed endpoint is missing")
    samples = int(row["samples"])
    state = _validated_metric_state(report.get("metric_state"), samples, label)
    metrics = b210.smu1.finalize_metric_state(state)
    if report.get("metrics") != metrics:
        raise RuntimeError(f"{label} metrics are not finalized from metric_state")
    audit = _validated_audit(report.get("held_target_audit"), samples, label)
    return state, metrics, audit


def _validate_alignment(b210_rows: Sequence[Mapping], anchor_rows: Sequence[Mapping]) -> None:
    b210.validate_fold_rows(b210_rows)
    mef2.validate_fold_rows(anchor_rows)
    if [int(row["held_aoa"]) for row in b210_rows] != list(FOLD_ORDER):
        raise ValueError("B210 held order changed")
    if [int(row["held_aoa"]) for row in anchor_rows] != list(FOLD_ORDER):
        raise ValueError("C208 anchor held order changed")
    for expected_held, candidate, anchor in zip(
            FOLD_ORDER, b210_rows, anchor_rows, strict=True):
        samples = int(candidate["samples"])
        if samples != int(anchor["samples"]):
            raise RuntimeError(f"held{expected_held} sample counts differ")
        candidate_manifest_file = _sha256_text(
            candidate.get("manifest_file_sha256"),
            f"B210 held{expected_held} manifest file SHA")
        anchor_manifest_file = _sha256_text(
            anchor.get("manifest_file_sha256"),
            f"C208 held{expected_held} manifest file SHA")
        if candidate_manifest_file != anchor_manifest_file:
            raise RuntimeError(f"held{expected_held} manifest file SHAs differ")
        candidate_manifest = candidate.get("manifest", {})
        anchor_manifest = anchor.get("manifest", {})
        for key in ("semantic_sha256", "windows", "excluded"):
            if candidate_manifest.get(key) != anchor_manifest.get(key):
                raise RuntimeError(
                    f"held{expected_held} manifest {key} differs")
        held_key = _sha256_text(
            candidate.get("held_ordered_key_sha256"),
            f"B210 held{expected_held} ordered-key SHA")
        anchor_key = _sha256_text(
            anchor.get("held_ordered_key_sha256"),
            f"C208 held{expected_held} ordered-key SHA")
        if held_key != anchor_key:
            raise RuntimeError(f"held{expected_held} ordered keys differ")
        candidate_normalizer = _sha256_text(
            candidate.get("normalizer", {}).get("sha256"),
            f"B210 held{expected_held} normalizer SHA")
        anchor_normalizer = _sha256_text(
            anchor.get("normalizer", {}).get("sha256"),
            f"C208 held{expected_held} normalizer SHA")
        if candidate_normalizer != anchor_normalizer:
            raise RuntimeError(f"held{expected_held} normalizer SHAs differ")
        _candidate_state, _candidate_metrics, candidate_audit = (
            _endpoint_payload(
                candidate, CANDIDATE_ENDPOINT,
                f"B210 held{expected_held} {CANDIDATE_ENDPOINT}"))
        _anchor_state, _anchor_metrics, anchor_audit = _endpoint_payload(
            anchor, ANCHOR_ENDPOINT,
            f"C208 held{expected_held} {ANCHOR_ENDPOINT}")
        if candidate_audit != anchor_audit:
            raise RuntimeError(
                f"held{expected_held} target/mask SHA audits differ")


def compare_rows(b210_rows: Sequence[Mapping],
                 anchor_rows: Sequence[Mapping]) -> dict:
    """Return the fixed two-fold comparison after strict alignment checks."""
    _validate_alignment(b210_rows, anchor_rows)
    fold_reports = []
    candidate_states = []
    anchor_states = []
    for held, candidate, anchor in zip(
            FOLD_ORDER, b210_rows, anchor_rows, strict=True):
        candidate_state, candidate_metrics, audit = _endpoint_payload(
            candidate, CANDIDATE_ENDPOINT,
            f"B210 held{held} {CANDIDATE_ENDPOINT}")
        anchor_state, anchor_metrics, _anchor_audit = _endpoint_payload(
            anchor, ANCHOR_ENDPOINT,
            f"C208 held{held} {ANCHOR_ENDPOINT}")
        comparison = b210.metric_comparison(candidate_metrics, anchor_metrics)
        candidate_states.append(candidate_state)
        anchor_states.append(anchor_state)
        fold_reports.append({
            "held_aoa": held,
            "samples": int(candidate["samples"]),
            "held_ordered_key_sha256": candidate["held_ordered_key_sha256"],
            "held_target_audit": audit,
            "candidate_endpoint": CANDIDATE_ENDPOINT,
            "anchor_endpoint": ANCHOR_ENDPOINT,
            "comparison": comparison,
        })
    candidate_pooled_state = b210.smu1.merge_metric_states(candidate_states)
    anchor_pooled_state = b210.smu1.merge_metric_states(anchor_states)
    expected_samples = sum(int(row["samples"]) for row in b210_rows)
    if (int(candidate_pooled_state["samples"]) != expected_samples
            or int(anchor_pooled_state["samples"]) != expected_samples):
        raise RuntimeError("pooled metric states do not cover both folds")
    candidate_pooled = b210.smu1.finalize_metric_state(
        candidate_pooled_state)
    anchor_pooled = b210.smu1.finalize_metric_state(anchor_pooled_state)
    fold_deltas = [row["comparison"] for row in fold_reports]
    gate = b210.two_fold_promotion_gate(
        candidate_pooled, anchor_pooled, fold_deltas)
    if gate["promotion_pass"]:
        decision = "PASS_LOCAL_TWO_FOLD_PROMOTION_GATE"
    elif gate["directional_pass"]:
        decision = "PASS_LOCAL_TWO_FOLD_DIRECTIONAL_GATE_ONLY"
    else:
        decision = "STOP_LOCAL_TWO_FOLD_NO_DIRECTIONAL_GATE"
    return {
        "schema": SCHEMA,
        "evidence_class": (
            "local clean2544 held10+held0 prefix-only comparison; not "
            "fixed725, clean716, release, ZIP, latency, or official evidence"),
        "fixed_endpoints": {
            "candidate": CANDIDATE_ENDPOINT,
            "anchor": ANCHOR_ENDPOINT,
            "endpoint_selection_allowed": False,
            "raw_endpoint_may_replace_primary": False,
        },
        "fold_order": list(FOLD_ORDER),
        "samples": expected_samples,
        "alignment": {
            "manifest": True,
            "normalizer": True,
            "held_ordered_keys": True,
            "ordered_raw_targets": True,
            "ordered_scored_masks": True,
            "metric_state_samples": True,
        },
        "folds": fold_reports,
        "pooled_candidate_metric_state": candidate_pooled_state,
        "pooled_anchor_metric_state": anchor_pooled_state,
        "pooled_candidate_metrics": candidate_pooled,
        "pooled_anchor_metrics": anchor_pooled,
        "gate": gate,
        "decision": decision,
        "local_held_evidence": True,
        "official_evidence": False,
        "submission_authorized": False,
        "release_checkpoint_written": False,
    }


def compare_from_files(b210_pairs: Sequence[Sequence[str]],
                       anchor_paths: Sequence[str]) -> dict:
    b210_rows, b210_sources = _load_b210_rows(b210_pairs)
    anchor_rows, anchor_sources = _load_anchor_rows(anchor_paths)
    return {
        "sources": {
            "b210": b210_sources,
            "frozen_c208": anchor_sources,
        },
        **compare_rows(b210_rows, anchor_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b210-fold", nargs=2, action="append", required=True,
        metavar=("PATH", "SHA256"),
        help="repeat in fixed held10, held0 order")
    parser.add_argument(
        "--frozen-c208-anchor", action="append", required=True,
        metavar="PATH", help="repeat in fixed held10, held0 order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_from_files(args.b210_fold, args.frozen_c208_anchor)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
