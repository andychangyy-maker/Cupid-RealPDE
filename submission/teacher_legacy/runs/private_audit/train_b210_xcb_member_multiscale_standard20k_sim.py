#!/usr/bin/env python3
"""Continue from-random XCB/member/multiscale Sim10k to Sim20k."""

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
    "cross_conditioned_member_adapters_multiscale_energy_v3")
METHOD = "b210_xcb_member_multiscale_standard_stride20_10k_then_10k_v1"
PASS_STATUS = "PASS_B210_XCB_MEMBER_MULTISCALE_STANDARD20K_SIM"


def validate_stage1_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("XCB/member/multiscale Sim10k has no model state")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("XCB/member/multiscale stage1 is not u10000")
    config = payload.get("training_config")
    if not isinstance(config, dict) or config.get("method") != SOURCE_METHOD:
        raise RuntimeError("checkpoint is not the sealed composite stage1")
    expected_true = (
        "cross_conditioned_backbone",
        "standard_objective_member_feature_adapters",
        "multiscale_energy_controller",
    )
    if (int(config.get("window_stride", -1)) != 20
            or not all(bool(config.get(key)) for key in expected_true)
            or bool(config.get("legacy_tke_gradient_floor"))):
        raise RuntimeError("composite stage1 training contract changed")


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["status"] = PASS_STATUS
    summary["evidence_class"] = (
        "from-random standard-objective XCB/member/multiscale Sim10k "
        "followed by a restarted identical Sim10k stage")
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
    transfer.b210.STANDARD_XCB_MEMBER_MULTISCALE_ENERGY_METHOD = METHOD
    transfer.b210.PASS_STATUS = PASS_STATUS
    transfer.main()


if __name__ == "__main__":
    main()
