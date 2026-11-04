#!/usr/bin/env python3
"""Continue sealed XCB/member/multiscale Sim20k for 4k updates."""

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


SOURCE_METHOD = "b210_xcb_member_multiscale_standard_stride20_10k_then_10k_v1"
METHOD = "b210_xcb_member_multiscale_standard_stride20_20k_then_4k_v1"
PASS_STATUS = "PASS_B210_XCB_MEMBER_MULTISCALE_STANDARD24K_SIM"
LOCAL_UPDATES = 4_000


def validate_stage2_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("XCB/member/multiscale Sim20k has no model state")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("XCB/member/multiscale stage2 is not local u10000")
    config = payload.get("training_config")
    if not isinstance(config, dict) or config.get("method") != SOURCE_METHOD:
        raise RuntimeError("checkpoint is not the sealed composite Sim20k")
    expected_true = (
        "cross_conditioned_backbone",
        "standard_objective_member_feature_adapters",
        "multiscale_energy_controller",
    )
    if (int(config.get("window_stride", -1)) != 20
            or not all(bool(config.get(key)) for key in expected_true)
            or bool(config.get("legacy_tke_gradient_floor"))):
        raise RuntimeError("composite stage2 training contract changed")


def configure_short_stage() -> None:
    transfer.b210.TERMINAL_UPDATE = LOCAL_UPDATES
    transfer.b210.PASS_STATUS = PASS_STATUS


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["status"] = PASS_STATUS
    summary["evidence_class"] = (
        "from-random standard-objective XCB/member/multiscale Sim20k "
        "followed by a restarted identical 4k matched-compute stage")
    summary["pretraining"] = {
        "checkpoint": str(transfer._PRETRAIN_PATH),
        "checkpoint_sha256": transfer._PRETRAIN_SHA256,
        "source_cumulative_supervised_updates": 20_000,
        "continuation_supervised_updates": LOCAL_UPDATES,
        "cumulative_supervised_updates": 24_000,
        "optimizer_and_cosine_restarted": True,
        "window_stride": 20,
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


def finalize_short_artifacts(output_dir: Path) -> tuple[Path, str]:
    old_path = output_dir / "model_10000.pth"
    new_path = output_dir / "model_4000.pth"
    if not old_path.is_file() or new_path.exists():
        raise RuntimeError("unexpected composite short-stage artifacts")
    os.replace(old_path, new_path)
    checksum = transfer.sha256(new_path)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint"] = str(new_path.resolve())
    summary["checkpoint_sha256"] = checksum
    summary["completed_updates"] = LOCAL_UPDATES
    summary["cumulative_supervised_updates"] = 24_000
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, summary_path)
    return new_path, checksum


def main() -> None:
    output_dir = output_dir_from_argv(sys.argv)
    configure_short_stage()
    transfer.METHOD = METHOD
    transfer.validate_pretrain_payload = validate_stage2_payload
    transfer.annotate_summary = annotate_summary
    transfer.b210.STANDARD_XCB_MEMBER_MULTISCALE_ENERGY_METHOD = METHOD
    transfer.main()
    checkpoint, checksum = finalize_short_artifacts(output_dir)
    print(json.dumps({
        "event": "b210_xcb_member_multiscale_standard24k_complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checksum,
        "local_updates": LOCAL_UPDATES,
        "cumulative_supervised_updates": 24_000,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
