#!/usr/bin/env python3
"""Continue one sealed dual-decoder Sim20k arm to matched Sim24k."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
        REPO_ROOT / "code", REPO_ROOT / "code" / "training",
        REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_dense_phase_pretrained_standard_sim as transfer  # noqa: E402


LOCAL_UPDATES = 4_000
CONTRACTS = {
    "shared_fused": {
        "source": "b210_dualdecoder_sharedfused_standard20k_v1",
        "method": "b210_dualdecoder_sharedfused_standard24k_v1",
        "status": "PASS_B210_DUALDECODER_SHAREDFUSED_STANDARD24K_SIM",
        "parameters": 196_316_169,
    },
    "separate_point_uq_fluctuation_gates": {
        "source": "b210_dualdecoder_separatepointuqgate_standard20k_v1",
        "method": "b210_dualdecoder_separatepointuqgate_standard24k_v1",
        "status": "PASS_B210_DUALDECODER_SEPARATEPOINTUQGATE_STANDARD24K_SIM",
        "parameters": 196_319_109,
    },
}


def extract_mode(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dual-decoder-mode", choices=tuple(CONTRACTS))
    known, remaining = parser.parse_known_args(argv[1:])
    if known.dual_decoder_mode is None:
        raise ValueError("--dual-decoder-mode is required")
    argv[:] = [argv[0], *remaining]
    return known.dual_decoder_mode


def validate_stage2_payload(payload: dict, mode: str) -> None:
    contract = CONTRACTS[mode]
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("dual-decoder Sim20k checkpoint has no model state")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("dual-decoder Sim20k is not a local u10000 endpoint")
    training = payload.get("training_config")
    model = payload.get("model_config")
    if (not isinstance(training, dict)
            or training.get("method") != contract["source"]):
        raise RuntimeError("checkpoint method is not the sealed stage2 arm")
    if (not isinstance(model, dict)
            or model.get("dual_decoder_mode") != mode):
        raise RuntimeError("checkpoint dual-decoder mode changed")
    if (int(training.get("window_stride", -1)) != 20
            or not bool(training.get("cross_conditioned_backbone"))
            or not bool(training.get(
                "standard_objective_member_feature_adapters"))
            or bool(training.get("legacy_tke_gradient_floor"))):
        raise RuntimeError("dual-decoder stage2 training contract changed")


def finalize_short_artifacts(output_dir: Path) -> tuple[Path, str]:
    old_path = output_dir / "model_10000.pth"
    new_path = output_dir / "model_4000.pth"
    if not old_path.is_file() or new_path.exists():
        raise RuntimeError("unexpected dual-decoder short-stage artifacts")
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
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    return new_path, checksum


def install(mode: str) -> None:
    contract = CONTRACTS[mode]
    original_config = transfer.b210.b210_config

    def dual_config(*args, **kwargs):
        return replace(
            original_config(*args, **kwargs), dual_decoder_mode=mode)

    def annotate_summary(output_dir: Path) -> None:
        path = output_dir / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["method"] = contract["method"]
        summary["status"] = contract["status"]
        summary["evidence_class"] = (
            "matched dual-decoder Sim20k endpoint followed by restarted "
            "identical Sim4k continuation")
        summary["pretraining"] = {
            "checkpoint": str(transfer._PRETRAIN_PATH),
            "checkpoint_sha256": transfer._PRETRAIN_SHA256,
            "dual_decoder_mode": mode,
            "source_cumulative_supervised_updates": 20_000,
            "continuation_supervised_updates": LOCAL_UPDATES,
            "cumulative_supervised_updates": 24_000,
            "optimizer_and_cosine_restarted": True,
            "matched_control_candidate_protocol": True,
            "external_data_or_model": False,
        }
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")

    transfer.b210.b210_config = dual_config
    transfer.b210.EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS = (
        contract["parameters"])
    transfer.b210.STANDARD_CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD = (
        contract["method"])
    transfer.b210.TERMINAL_UPDATE = LOCAL_UPDATES
    transfer.b210.PASS_STATUS = contract["status"]
    transfer.METHOD = contract["method"]
    transfer.validate_pretrain_payload = (
        lambda payload: validate_stage2_payload(payload, mode))
    transfer.annotate_summary = annotate_summary


def main() -> None:
    mode = extract_mode(sys.argv)
    install(mode)
    output_dir = None
    for index, token in enumerate(sys.argv[:-1]):
        if token == "--output-dir":
            output_dir = Path(sys.argv[index + 1])
            break
    if output_dir is None:
        raise ValueError("--output-dir is required")
    transfer.main()
    checkpoint, checksum = finalize_short_artifacts(output_dir)
    print(json.dumps({
        "event": "b210_dual_decoder_standard24k_complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checksum,
        "local_updates": LOCAL_UPDATES,
        "cumulative_supervised_updates": 24_000,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

