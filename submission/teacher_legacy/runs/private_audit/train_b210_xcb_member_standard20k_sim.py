#!/usr/bin/env python3
"""Continue the standard-objective XCB/member model from Sim10k to Sim20k."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_dense_phase_pretrained_standard_sim as transfer  # noqa: E402


SOURCE_METHOD = (
    "b210_np_functional_sim_realclosure_standardobjective_"
    "cross_conditioned_multiscale_member_adapters_v2")
METHOD = "b210_xcb_member_standard_stride20_10k_then_10k_v1"
PASS_STATUS = "PASS_B210_XCB_MEMBER_STANDARD20K_SIM"


def validate_stage1_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("XCB/member Sim10k checkpoint has no model_state_dict")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("XCB/member stage1 is not a u10000 endpoint")
    config = payload.get("training_config")
    if not isinstance(config, dict):
        raise TypeError("XCB/member Sim10k checkpoint has no training_config")
    if config.get("method") != SOURCE_METHOD:
        raise RuntimeError("checkpoint is not standard-objective XCB/member stage1")
    if int(config.get("window_stride", -1)) != 20:
        raise RuntimeError("XCB/member stage1 was not trained at stride 20")
    if not bool(config.get("cross_conditioned_backbone")):
        raise RuntimeError("XCB/member stage1 has no cross conditioner")
    if not bool(config.get("standard_objective_member_feature_adapters")):
        raise RuntimeError("XCB/member stage1 has no standard-objective adapters")
    if bool(config.get("legacy_tke_gradient_floor")):
        raise RuntimeError("XCB/member stage1 unexpectedly used the TKE floor")


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["status"] = PASS_STATUS
    summary["evidence_class"] = (
        "from-random standard-objective cross-conditioned/member-adapter "
        "Sim10k followed by a restarted identical Sim10k stage")
    summary["pretraining"] = {
        "checkpoint": str(transfer._PRETRAIN_PATH),
        "checkpoint_sha256": transfer._PRETRAIN_SHA256,
        "stage1_updates": 10_000,
        "stage2_updates": 10_000,
        "cumulative_supervised_updates": 20_000,
        "optimizer_and_cosine_restarted": True,
        "window_stride": 20,
        "external_data_or_model": False,
        "both_paired_arms_loaded_identically": True,
    }
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    transfer.METHOD = METHOD
    transfer.validate_pretrain_payload = validate_stage1_payload
    transfer.annotate_summary = annotate_summary
    transfer.b210.STANDARD_CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD = METHOD
    transfer.b210.PASS_STATUS = PASS_STATUS
    transfer.main()


if __name__ == "__main__":
    main()
