#!/usr/bin/env python3
"""Run XCB/member Standard Sim10k from TKE-aware causal pretraining."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_dense_phase_pretrained_standard_sim as transfer  # noqa: E402


METHOD = "b210_tke_causal_xcb_member_standard_stride20_10k_v1"
PRETEXT_METHOD = (
    "b210_tke_aware_causal_future_xcb_member_sim_real_prefix_pretrain_v1")
PASS_STATUS = "PASS_B210_TKE_CAUSAL_XCB_MEMBER_SIM10K"


def validate_pretrain_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("TKE-aware causal checkpoint has no model_state_dict")
    if payload.get("phase") != "sim_real_prefix_causal_pretrain":
        raise RuntimeError("checkpoint is not a causal future pretrain")
    if int(payload.get("iteration", -1)) != 4_000:
        raise RuntimeError("TKE-aware causal checkpoint is not u4000")
    config = payload.get("training_config")
    if not isinstance(config, dict) or config.get("method") != PRETEXT_METHOD:
        raise RuntimeError("TKE-aware causal method contract changed")
    if not bool(config.get("tke_aware_xcb_member")):
        raise RuntimeError("TKE-aware XCB/member flag is missing")
    calibration = config.get("tke_gradient_calibration")
    if (not isinstance(calibration, dict)
            or abs(float(calibration.get("calibrated_gradient_ratio", -1.0))
                   - 0.20) > 1.0e-6):
        raise RuntimeError("TKE-aware gradient calibration is invalid")
    architecture = payload.get("model_config")
    if (not isinstance(architecture, dict)
            or not bool(architecture.get("cross_conditioned_backbone"))
            or int(architecture.get("member_adapter_channels", 0)) <= 0):
        raise RuntimeError("pretext checkpoint is not the XCB/member architecture")
    audit = payload.get("pretraining_audit")
    if (not isinstance(audit, dict)
            or audit.get("changed_untouched_state_tensors") != []
            or audit.get("initial_non_backbone_sha256")
            != audit.get("final_non_backbone_sha256")):
        raise RuntimeError("pretext changed a deployed non-backbone state")


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["status"] = PASS_STATUS
    summary["evidence_class"] = (
        "from-random XCB/member; 4k causal Sim plus clean-Real-prefix future "
        "pretraining with gradient-calibrated physical multi-horizon TKE, "
        "then ordinary standard-objective stride20 Sim10k")
    summary["pretraining"] = {
        "checkpoint": str(transfer._PRETRAIN_PATH),
        "checkpoint_sha256": transfer._PRETRAIN_SHA256,
        "pretext_updates": 4_000,
        "supervised_updates": 10_000,
        "pretext_data": "released Sim + clean81 Real-prefix windows",
        "tke_horizons": [5, 10, 20],
        "tke_target_gradient_ratio": 0.20,
        "tail_arrays_read": 0,
        "external_data_or_model": False,
        "both_paired_arms_loaded_identically": True,
    }
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    transfer.METHOD = METHOD
    transfer.validate_pretrain_payload = validate_pretrain_payload
    transfer.annotate_summary = annotate_summary
    transfer.b210.STANDARD_CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD = METHOD
    transfer.b210.REAL_CLOSURE_SEED1_METHOD = METHOD
    transfer.b210.PASS_STATUS = PASS_STATUS
    transfer.main()


if __name__ == "__main__":
    main()
