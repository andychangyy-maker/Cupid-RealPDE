#!/usr/bin/env python3
"""Run the second standard Sim10k after causal-pretrained Sim10k."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_standard_pretrained_standard_sim as standard  # noqa: E402


STAGE1_METHOD = "b210_causal_denoising_pretrain_then_standard_stride20_v1"
METHOD = "b210_causal_denoising_pretrain_then_standard_stride20x2_v1"


def validate_stage1(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("causal stage-one checkpoint has no model_state_dict")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("causal stage-one checkpoint is not u10000")
    config = payload.get("training_config")
    if not isinstance(config, dict) or config.get("method") != STAGE1_METHOD:
        raise RuntimeError("checkpoint is not causal-pretrained stage one")


def annotate(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["evidence_class"] = (
        "4k causal Sim+Real-prefix denoising future pretrain followed by two "
        "restarted ordinary standard stride-20 Sim10k stages")
    summary["pretraining"] = {
        "checkpoint": str(standard.transfer._PRETRAIN_PATH),
        "checkpoint_sha256": standard.transfer._PRETRAIN_SHA256,
        "causal_pretext_updates": 4_000,
        "stage1_supervised_updates": 10_000,
        "stage2_supervised_updates": 10_000,
        "external_data_or_model": False,
        "both_paired_arms_loaded_identically": True,
    }
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    standard.validate_standard_pretrain_payload = validate_stage1
    standard.annotate_standard_summary = annotate
    standard.METHOD = METHOD
    standard.main()


if __name__ == "__main__":
    main()
