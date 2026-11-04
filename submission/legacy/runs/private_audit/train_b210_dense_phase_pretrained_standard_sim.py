#!/usr/bin/env python3
"""Continue standard B210 Sim training from a dense-phase Sim checkpoint.

Stage one is the already-audited stride-10 B210 recipe.  This launcher is the
stage-two transfer test: it restores that point model into both paired arms,
then runs the unchanged stride-20/clean-closure B210 recipe.  Loading the same
state into both arms preserves the update-0 paired identity audit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as b210  # noqa: E402


METHOD = "b210_dense_phase_pretrain_then_standard_stride20_v1"
_PRETRAIN_PATH: Path | None = None
_PRETRAIN_SHA256: str | None = None
_PRETRAIN_PAYLOAD: dict | None = None
_ORIGINAL_BUILD_ARM_MODEL = b210.build_arm_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pop_private_args(argv: list[str]) -> tuple[list[str], Path, str]:
    remaining = [argv[0]]
    checkpoint = None
    expected_sha = None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in {"--pretrain-checkpoint", "--pretrain-checkpoint-sha256"}:
            if index + 1 >= len(argv):
                raise ValueError(f"{token} requires one value")
            value = argv[index + 1]
            if token == "--pretrain-checkpoint":
                checkpoint = Path(value)
            else:
                expected_sha = value.lower()
            index += 2
            continue
        remaining.append(token)
        index += 1
    if checkpoint is None or expected_sha is None:
        raise ValueError(
            "both --pretrain-checkpoint and --pretrain-checkpoint-sha256 are required")
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("pretrain checkpoint SHA-256 is invalid")
    return remaining, checkpoint, expected_sha


def validate_pretrain_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("pretrain checkpoint has no model_state_dict")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("pretrain checkpoint is not the stride-10 u10000 endpoint")
    config = payload.get("training_config")
    if not isinstance(config, dict):
        raise TypeError("pretrain checkpoint has no training_config")
    if int(config.get("window_stride", -1)) != 10:
        raise RuntimeError("pretrain checkpoint was not trained with stride 10")
    method = str(config.get("method", ""))
    if "phase_dense_stride10" not in method:
        raise RuntimeError(
            f"pretrain checkpoint method is not the dense-phase arm: {method}")


def build_pretrained_arm(*args, **kwargs):
    if _PRETRAIN_PAYLOAD is None:
        raise RuntimeError("dense-phase pretrain payload was not initialized")
    model = _ORIGINAL_BUILD_ARM_MODEL(*args, **kwargs)
    incompatible = model.load_state_dict(
        _PRETRAIN_PAYLOAD["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"pretrain state is incompatible: {incompatible}")
    return model


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["evidence_class"] = (
        "two-stage released-Sim transfer experiment; dense stride-10 u10000 "
        "pretraining followed by standard stride-20 u10000 training")
    summary["pretraining"] = {
        "checkpoint": str(_PRETRAIN_PATH),
        "checkpoint_sha256": _PRETRAIN_SHA256,
        "stage1_window_stride": 10,
        "stage1_updates": 10_000,
        "stage2_window_stride": 20,
        "stage2_updates": 10_000,
        "external_data_or_model": False,
        "both_paired_arms_loaded_identically": True,
    }
    contract = summary.get("training_contract")
    if isinstance(contract, dict):
        contract["no_c208_checkpoint_loaded"] = True
        contract["random_initialization_applies_to_stage1"] = True
        contract["stage2_initialization"] = "released-Sim stride10 u10000 endpoint"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    global _PRETRAIN_PATH, _PRETRAIN_SHA256, _PRETRAIN_PAYLOAD
    remaining, checkpoint, expected_sha = pop_private_args(sys.argv)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pretrain checkpoint SHA mismatch: {actual_sha} != {expected_sha}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_pretrain_payload(payload)
    _PRETRAIN_PATH = checkpoint.resolve()
    _PRETRAIN_SHA256 = actual_sha
    _PRETRAIN_PAYLOAD = payload
    sys.argv[:] = remaining

    b210.build_arm_model = build_pretrained_arm
    b210.REAL_CLOSURE_SEED1_METHOD = METHOD
    b210.main()

    output_dir = None
    for index, token in enumerate(remaining[:-1]):
        if token == "--output-dir":
            output_dir = Path(remaining[index + 1])
            break
    if output_dir is None:
        raise RuntimeError("base output directory was not found after training")
    annotate_summary(output_dir)


if __name__ == "__main__":
    main()
