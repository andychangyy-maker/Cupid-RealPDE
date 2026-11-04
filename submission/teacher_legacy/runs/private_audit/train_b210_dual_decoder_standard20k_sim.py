#!/usr/bin/env python3
"""Continue one sealed dual-decoder Sim10k arm to matched Sim20k."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
        REPO_ROOT / "code", REPO_ROOT / "code" / "training",
        REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_dense_phase_pretrained_standard_sim as transfer  # noqa: E402


CONTRACTS = {
    "shared_fused": {
        "source": "b210_xcb_member_dualdecoder_sharedfused_standard10k_v1",
        "method": "b210_dualdecoder_sharedfused_standard20k_v1",
        "status": "PASS_B210_DUALDECODER_SHAREDFUSED_STANDARD20K_SIM",
        "parameters": 196_316_169,
    },
    "separate_point_uq_fluctuation_gates": {
        "source": (
            "b210_xcb_member_dualdecoder_"
            "separatepointuqgate_standard10k_v4"),
        "method": "b210_dualdecoder_separatepointuqgate_standard20k_v1",
        "status": "PASS_B210_DUALDECODER_SEPARATEPOINTUQGATE_STANDARD20K_SIM",
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


def validate_stage1_payload(payload: dict, mode: str) -> None:
    contract = CONTRACTS[mode]
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("dual-decoder Sim10k checkpoint has no model state")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("dual-decoder stage1 is not a u10000 endpoint")
    training = payload.get("training_config")
    model = payload.get("model_config")
    if (not isinstance(training, dict)
            or training.get("method") != contract["source"]):
        raise RuntimeError("checkpoint method is not the sealed stage1 arm")
    if (not isinstance(model, dict)
            or model.get("dual_decoder_mode") != mode):
        raise RuntimeError("checkpoint dual-decoder mode changed")
    if (int(training.get("window_stride", -1)) != 20
            or not bool(training.get("cross_conditioned_backbone"))
            or not bool(training.get(
                "standard_objective_member_feature_adapters"))
            or bool(training.get("legacy_tke_gradient_floor"))):
        raise RuntimeError("dual-decoder stage1 training contract changed")


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
            "matched dual-decoder Sim10k endpoint followed by restarted "
            "identical Sim10k continuation")
        summary["pretraining"] = {
            "checkpoint": str(transfer._PRETRAIN_PATH),
            "checkpoint_sha256": transfer._PRETRAIN_SHA256,
            "dual_decoder_mode": mode,
            "stage1_updates": 10_000,
            "stage2_updates": 10_000,
            "cumulative_supervised_updates": 20_000,
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
    transfer.b210.PASS_STATUS = contract["status"]
    transfer.METHOD = contract["method"]
    transfer.validate_pretrain_payload = (
        lambda payload: validate_stage1_payload(payload, mode))
    transfer.annotate_summary = annotate_summary


def main() -> None:
    mode = extract_mode(sys.argv)
    install(mode)
    transfer.main()


if __name__ == "__main__":
    main()
