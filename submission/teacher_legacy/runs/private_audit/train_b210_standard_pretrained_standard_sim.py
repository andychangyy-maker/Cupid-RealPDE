#!/usr/bin/env python3
"""Matched-compute control for the dense-phase two-stage Sim experiment.

The experiment restores the historical seed-1 stride-20 u10000 endpoint into
both paired arms, restarts the optimizer/schedule, and runs the unchanged
stride-20 B210 recipe for another 10000 updates.  This isolates the value of
dense stride-10 phase pretraining from the value of simply training longer.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_dense_phase_pretrained_standard_sim as transfer  # noqa: E402


METHOD = "b210_standard_stride20_pretrain_then_standard_stride20_v1"


def validate_standard_pretrain_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("pretrain checkpoint has no model_state_dict")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("pretrain checkpoint is not a u10000 endpoint")
    config = payload.get("training_config")
    if not isinstance(config, dict):
        raise TypeError("pretrain checkpoint has no training_config")
    if int(config.get("window_stride", -1)) != 20:
        raise RuntimeError("matched-control checkpoint was not trained with stride 20")
    method = str(config.get("method", ""))
    if method != "b210_np_functional_sim_realclosure_seed1_soup_member_v1":
        raise RuntimeError(
            f"pretrain checkpoint is not the frozen standard seed-1 arm: {method}")


def annotate_standard_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["evidence_class"] = (
        "matched-compute control; standard stride-20 u10000 pretraining followed "
        "by a restarted standard stride-20 u10000 stage")
    summary["pretraining"] = {
        "checkpoint": str(transfer._PRETRAIN_PATH),
        "checkpoint_sha256": transfer._PRETRAIN_SHA256,
        "stage1_window_stride": 20,
        "stage1_updates": 10_000,
        "stage2_window_stride": 20,
        "stage2_updates": 10_000,
        "optimizer_and_schedule_restarted_for_stage2": True,
        "external_data_or_model": False,
        "both_paired_arms_loaded_identically": True,
    }
    contract = summary.get("training_contract")
    if isinstance(contract, dict):
        contract["no_c208_checkpoint_loaded"] = True
        contract["random_initialization_applies_to_stage1"] = True
        contract["stage2_initialization"] = "released-Sim stride20 u10000 endpoint"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    transfer.METHOD = METHOD
    transfer.validate_pretrain_payload = validate_standard_pretrain_payload
    transfer.annotate_summary = annotate_standard_summary
    transfer.main()


if __name__ == "__main__":
    main()
