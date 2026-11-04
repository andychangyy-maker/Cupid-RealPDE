#!/usr/bin/env python3
"""Run standard B210 Sim10k from the causal-denoising backbone."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_dense_phase_pretrained_standard_sim as transfer  # noqa: E402


METHOD = "b210_causal_denoising_pretrain_then_standard_stride20_v1"
PRETEXT_METHOD = "b210_causal_denoising_future_sim_real_prefix_pretrain_v1"


def validate_pretrain_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("causal pretrain checkpoint has no model_state_dict")
    if payload.get("phase") != "sim_real_prefix_causal_pretrain":
        raise RuntimeError("checkpoint is not a causal future pretrain")
    if int(payload.get("iteration", -1)) != 4_000:
        raise RuntimeError("causal pretrain checkpoint is not u4000")
    config = payload.get("training_config")
    if not isinstance(config, dict) or config.get("method") != PRETEXT_METHOD:
        raise RuntimeError("causal pretrain method contract changed")
    audit = payload.get("pretraining_audit")
    if (not isinstance(audit, dict)
            or audit.get("changed_untouched_state_tensors") != []
            or audit.get("initial_non_backbone_sha256")
            != audit.get("final_non_backbone_sha256")):
        raise RuntimeError("causal pretrain changed deployed non-backbone state")


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["evidence_class"] = (
        "from-random B210; 4k causal denoising future prediction on released "
        "Sim plus clean Real prefixes, then ordinary standard stride-20 Sim10k")
    summary["pretraining"] = {
        "checkpoint": str(transfer._PRETRAIN_PATH),
        "checkpoint_sha256": transfer._PRETRAIN_SHA256,
        "pretext_updates": 4_000,
        "supervised_updates": 10_000,
        "pretext_data": "released Sim windows + clean81 Real-prefix windows",
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
    transfer.main()


if __name__ == "__main__":
    main()
