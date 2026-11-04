#!/usr/bin/env python3
"""Derive a JSON-roundtrip-sealed point-OOF result without rewriting source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_b210_full_real_point_oof_fold as point_fold


dr1 = point_fold.dr1
ft12 = point_fold.ft12


def validate_unsealed_result(payload: dict) -> dict:
    if point_fold.JSON_SELF_SEAL_KEY in payload:
        raise ValueError("source point-OOF result is already self-sealed")
    held_aoa = int(payload.get("held_aoa", -1))
    fold = point_fold.aoa_fold(held_aoa)
    if (payload.get("schema") != point_fold.SCHEMA
            or int(payload.get("held_fold", -1)) != fold
            or int(payload.get("folds", -1)) != point_fold.FOLDS
            or payload.get("aoa_values") != list(point_fold.AOA_VALUES)
            or payload.get("decision") != "COMPLETE_FULL_REAL_POINT_OOF_FOLD"
            or payload.get("held_data_opened") is not False
            or payload.get("clean716_manifest_read") is not False
            or payload.get("start", {}).get("checkpoint_sha256")
            != point_fold.START_SHA256
            or payload.get("manifest_file_sha256")
            != ft12.FORMAL_MANIFEST_FILE_SHA256
            or payload.get("manifest", {}).get("semantic_sha256")
            != ft12.FORMAL_MANIFEST_SEMANTIC_SHA256):
        raise RuntimeError("unsealed point-OOF result contract changed")
    seal = payload.get("fit", {}).get("seal", {})
    checkpoint = Path(str(seal.get("path", "")))
    accepted_schemas = (
        point_fold.ENDPOINT_SCHEMA, point_fold.support.ENDPOINT_SCHEMA)
    accepted_experiments = (
        point_fold.EXPERIMENT, point_fold.support.EXPERIMENT)
    return point_fold.validate_endpoint(
        checkpoint, seal, held_aoa=held_aoa,
        accepted_schemas=accepted_schemas,
        accepted_experiments=accepted_experiments)


def derive(source: Path, expected_sha256: str, output: Path) -> dict:
    expected_sha256 = dr1.normalized_sha256(
        expected_sha256, "unsealed point-OOF result SHA")
    if source.resolve() == output.resolve():
        raise ValueError("sealed result must not overwrite its source")
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    if not source.is_file() or dr1.sha256(source) != expected_sha256:
        raise RuntimeError("unsealed point-OOF result SHA changed")
    payload = json.loads(source.read_text(encoding="utf-8"))
    endpoint = validate_unsealed_result(payload)
    payload["json_seal_derivation"] = {
        "source": str(source.resolve()),
        "source_file_sha256": expected_sha256,
        "transformation": (
            "JSON dump/load normalization plus provenance self-seal only"),
        "checkpoint_unchanged": True,
        "checkpoint_file_sha256": endpoint["file_sha256"],
        "checkpoint_model_state_sha256": endpoint["model_state_sha256"],
    }
    sealed = point_fold.seal_json_payload(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    dr1.write_json_atomic(output, sealed)
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    self_sha = point_fold.verify_json_payload(reloaded)
    return {
        "source": str(source.resolve()),
        "source_file_sha256": expected_sha256,
        "output": str(output.resolve()),
        "output_file_sha256": dr1.sha256(output),
        "payload_without_self_seal_sha256": self_sha,
        "held_aoa": int(reloaded["held_aoa"]),
        "checkpoint_file_sha256": endpoint["file_sha256"],
        "checkpoint_model_state_sha256": endpoint["model_state_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        derive(args.source, args.source_sha256, args.output),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
