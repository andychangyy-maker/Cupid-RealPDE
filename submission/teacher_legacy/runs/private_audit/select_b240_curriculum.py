#!/usr/bin/env python3
"""Apply the frozen B240 curriculum decision to completed B210 summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
}
PHASE_MARGIN = 0.04


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal(summary: dict) -> dict[str, float]:
    anchor = summary["paired_candidate"]["terminal_anchor"]
    return {name: float(anchor[name]) for name in WEIGHTS}


def decide(dense_summary: dict, standard_summary: dict) -> dict:
    dense = terminal(dense_summary)
    standard = terminal(standard_summary)
    deltas = {name: dense[name] - standard[name] for name in WEIGHTS}
    phase_value = sum(WEIGHTS[name] * deltas[name] for name in WEIGHTS)
    safe = all(delta >= -0.05 for delta in deltas.values())
    selected = "dense_stride10_then_standard" if (
        phase_value >= PHASE_MARGIN and safe) else "standard_then_standard"
    return {
        "selected_curriculum": selected,
        "phase_value_point65": phase_value,
        "dense_minus_standard": deltas,
        "phase_margin": PHASE_MARGIN,
        "dense_component_floor": -0.05,
        "dense_component_floor_pass": safe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-summary", type=Path, required=True)
    parser.add_argument("--standard-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dense = json.loads(args.dense_summary.read_text(encoding="utf-8"))
    standard = json.loads(args.standard_summary.read_text(encoding="utf-8"))
    report = decide(dense, standard)
    report["dense_summary_sha256"] = sha256(args.dense_summary)
    report["standard_summary_sha256"] = sha256(args.standard_summary)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
