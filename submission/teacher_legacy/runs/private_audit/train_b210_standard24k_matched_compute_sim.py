#!/usr/bin/env python3
"""Continue the sealed standard-20k B210 endpoint for 4k Sim updates.

This is the conservative matched-compute control for the 4k masked
reconstruction pretext experiment. It reuses the frozen standard recipe and
restarts its optimizer and cosine schedule for exactly 4000 updates. The base
runner has a historical ``model_10000.pth`` filename; after it returns we
atomically rename that artifact and repair the summary so the shorter
continuation is self-describing.
"""

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


METHOD = "b210_standard_stride20_20k_then_standard_stride20_4k_v1"
SOURCE_METHOD = "b210_standard_stride20_pretrain_then_standard_stride20_v1"
LOCAL_UPDATES = 4_000


def validate_standard20_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("standard-20k checkpoint has no model_state_dict")
    # Each restarted stage stores its local update count. The source method
    # and its summary establish the cumulative 20k history.
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("standard-20k checkpoint is not a local u10000 endpoint")
    config = payload.get("training_config")
    if not isinstance(config, dict):
        raise TypeError("standard-20k checkpoint has no training_config")
    if config.get("method") != SOURCE_METHOD:
        raise RuntimeError("checkpoint is not the sealed standard-20k endpoint")
    if int(config.get("window_stride", -1)) != 20:
        raise RuntimeError("standard-20k checkpoint was not trained at stride 20")


def configure_short_standard_stage() -> None:
    """Narrow the already frozen standard runner to the preregistered 4k."""
    transfer.b210.TERMINAL_UPDATE = LOCAL_UPDATES
    transfer.b210.PASS_STATUS = "PASS_B210_STANDARD24K_MATCHED_COMPUTE_SIM4000"


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["status"] = "PASS_B210_STANDARD24K_MATCHED_COMPUTE_SIM4000"
    summary["evidence_class"] = (
        "conservative matched-compute control; sealed standard stride-20 "
        "Sim20k followed by a restarted ordinary standard Sim4k stage")
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
    contract = summary.get("training_contract")
    if isinstance(contract, dict):
        contract["no_c208_checkpoint_loaded"] = True
        contract["random_initialization_applies_to_first_standard_stage"] = True
        contract["stage3_initialization"] = "sealed standard Sim20k endpoint"
        contract["matched_compute_role"] = (
            "control for 4000 masked Sim+Real-prefix reconstruction updates")
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def finalize_short_artifacts(output_dir: Path) -> tuple[Path, str]:
    old_path = output_dir / "model_10000.pth"
    new_path = output_dir / "model_4000.pth"
    if not old_path.is_file() or new_path.exists():
        raise RuntimeError("unexpected short-stage checkpoint artifact state")
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


def output_dir_from_argv(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--output-dir") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("--output-dir is required") from error


def main() -> None:
    output_dir = output_dir_from_argv(sys.argv)
    configure_short_standard_stage()
    transfer.METHOD = METHOD
    transfer.validate_pretrain_payload = validate_standard20_payload
    transfer.annotate_summary = annotate_summary
    transfer.main()
    checkpoint, checksum = finalize_short_artifacts(output_dir)
    print(json.dumps({
        "event": "b210_standard24k_matched_compute_complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checksum,
        "local_updates": LOCAL_UPDATES,
        "cumulative_supervised_updates": 24_000,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
