#!/usr/bin/env python3
"""Continue TKE-aware causal XCB/member Sim20k for a final 4k stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_dense_phase_pretrained_standard_sim as transfer  # noqa: E402


SOURCE_METHOD = "b210_tke_causal_xcb_member_standard_stride20_20k_v1"
METHOD = "b210_tke_causal_xcb_member_standard_stride20_24k_v1"
PASS_STATUS = "PASS_B210_TKE_CAUSAL_XCB_MEMBER_SIM24K"
LOCAL_UPDATES = 4_000


def validate_stage2_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("TKE-aware XCB/member Sim20k checkpoint has no model state")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("TKE-aware XCB/member Sim20k is not local u10000")
    config = payload.get("training_config")
    if (not isinstance(config, dict) or config.get("method") != SOURCE_METHOD
            or int(config.get("window_stride", -1)) != 20
            or not bool(config.get("standard_objective_member_feature_adapters"))):
        raise RuntimeError("TKE-aware XCB/member Sim20k contract changed")


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["status"] = PASS_STATUS
    summary["evidence_class"] = (
        "TKE-aware causal XCB/member pretext plus ordinary restarted "
        "standard-objective Sim10k+10k+4k")
    summary["pretraining"] = {
        "checkpoint": str(transfer._PRETRAIN_PATH),
        "checkpoint_sha256": transfer._PRETRAIN_SHA256,
        "source_cumulative_supervised_updates": 20_000,
        "continuation_supervised_updates": LOCAL_UPDATES,
        "cumulative_supervised_updates": 24_000,
        "optimizer_and_cosine_restarted": True,
        "external_data_or_model": False,
        "both_paired_arms_loaded_identically": True,
    }
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def output_dir_from_argv(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--output-dir") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("--output-dir is required") from error


def finalize(output_dir: Path) -> None:
    old_path = output_dir / "model_10000.pth"
    new_path = output_dir / "model_4000.pth"
    if not old_path.is_file() or new_path.exists():
        raise RuntimeError("unexpected TKE-aware Sim24k artifact state")
    os.replace(old_path, new_path)
    checksum = transfer.sha256(new_path)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "checkpoint": str(new_path.resolve()),
        "checkpoint_sha256": checksum,
        "completed_updates": LOCAL_UPDATES,
        "cumulative_supervised_updates": 24_000,
    })
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, summary_path)


def main() -> None:
    output_dir = output_dir_from_argv(sys.argv)
    transfer.b210.TERMINAL_UPDATE = LOCAL_UPDATES
    transfer.METHOD = METHOD
    transfer.validate_pretrain_payload = validate_stage2_payload
    transfer.annotate_summary = annotate_summary
    transfer.b210.STANDARD_CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD = METHOD
    transfer.b210.PASS_STATUS = PASS_STATUS
    transfer.main()
    finalize(output_dir)


if __name__ == "__main__":
    main()
