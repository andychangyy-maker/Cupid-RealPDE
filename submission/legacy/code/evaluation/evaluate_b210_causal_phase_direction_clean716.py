#!/usr/bin/env python3
"""Replay all three sealed phase-adapter arms on one exact clean716 panel.

No target H5 is opened until the control, causal, and pooled-history endpoint
files, their raw and EMA tensor digests, initialization, common sample stream,
normalizer, and frozen B210 source have all been validated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_causal_phase_direction_real4k as phase_train  # noqa: E402
import train_b210_pooled_history_phase_real4k as pooled_train  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real4k  # noqa: E402


SCHEMA = "b210_phase_direction_three_arm_clean716_replay_v2"
ACKNOWLEDGEMENT = "EVALUATE_B210_PHASE_AFTER_SIX_ENDPOINT_SEALS_V2"
MODEL_NAMES = (
    "b210",
    "spatial_control_raw",
    "spatial_control_ema",
    "causal_raw",
    "causal_ema",
    "pooled_raw",
    "pooled_ema",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--spatial-control-endpoint", type=Path, required=True)
    parser.add_argument("--spatial-control-endpoint-sha256", required=True)
    parser.add_argument("--causal-endpoint", type=Path, required=True)
    parser.add_argument("--causal-endpoint-sha256", required=True)
    parser.add_argument("--pooled-endpoint", type=Path, required=True)
    parser.add_argument("--pooled-endpoint-sha256", required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--clean716-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--acknowledge-target-access", default="")
    return parser.parse_args(argv)


def _adapter_config(stored: Mapping[str, object]) -> CausalPhaseDirectionConfig:
    values = dict(stored)
    metadata = {
        "architecture": "causal_phase_direction_adapter",
        "base_responsibility": "temporal_mean_and_fluctuation_rms",
        "adapter_responsibility": "zero_mean_unit_rms_uv_direction",
        "pressure_passthrough": True,
        "flattened_history_bypass": False,
    }
    for key, expected in metadata.items():
        if values.pop(key, None) != expected:
            raise ValueError(f"adapter metadata changed: {key}")
    config = CausalPhaseDirectionConfig(**values)
    config.validate()
    return config


def validate_endpoint(
        path: Path, expected_sha: str, expected_arm: str,
        expected_base_sha: str,
        ) -> tuple[dict[str, object], dict[str, object]]:
    actual_sha = phase_train.require_sha(path, expected_sha)
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise TypeError("adapter endpoint must be a mapping")
    if (payload.get("schema") != phase_train.ENDPOINT_SCHEMA
            or payload.get("experiment_schema") != phase_train.SCHEMA
            or payload.get("arm") != expected_arm
            or int(payload.get("update", -1)) != phase_train.UPDATES):
        raise ValueError(f"{expected_arm} is not a complete u4000 endpoint")
    source = payload.get("base_checkpoint")
    if (not isinstance(source, Mapping)
            or source.get("sha256") != expected_base_sha):
        raise ValueError(f"{expected_arm} does not reference the frozen base")
    config = _adapter_config(payload.get("adapter_config", {}))
    if config.mixer_mode != expected_arm:
        raise ValueError(f"{expected_arm} mixer mode changed")
    if int(payload.get("adapter_parameters", -1)) != (
            phase_train.EXPECTED_ADAPTER_PARAMETERS):
        raise ValueError(f"{expected_arm} parameter count changed")
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    for endpoint_kind, state_key, sha_key in (
            ("raw", "raw_adapter_state", "raw_adapter_state_sha256"),
            ("ema", "ema_adapter_state", "ema_adapter_state_sha256")):
        state = payload.get(state_key)
        if not isinstance(state, Mapping):
            raise ValueError(f"{expected_arm} lacks {endpoint_kind} state")
        actual_state_sha = ema_lib.tensor_state_sha256(state)
        if actual_state_sha != payload.get(sha_key):
            raise RuntimeError(f"{expected_arm} {endpoint_kind} state SHA changed")
        probe = CausalPhaseDirectionAdapter(config).float()
        probe.load_state_dict(state, strict=True)
        if sum(parameter.numel() for parameter in probe.parameters()) != (
                phase_train.EXPECTED_ADAPTER_PARAMETERS):
            raise RuntimeError(f"{expected_arm} {endpoint_kind} strict-load size changed")
        if any(value.is_floating_point()
               and not bool(torch.isfinite(value).all())
               for value in state.values()):
            raise FloatingPointError(
                f"{expected_arm} {endpoint_kind} state is non-finite")
        states[endpoint_kind] = state
    ema = payload.get("ema")
    if not isinstance(ema, Mapping) or int(ema.get("updates", -1)) != phase_train.UPDATES:
        raise ValueError(f"{expected_arm} EMA is not u4000")
    ema_metadata = dict(ema)
    stored_ema_sha = ema_metadata.pop("seal_metadata_sha256", None)
    if stored_ema_sha != ema_lib.canonical_sha256(ema_metadata):
        raise RuntimeError(f"{expected_arm} EMA metadata seal changed")
    if (ema.get("raw_model_state_sha256")
            != payload.get("raw_adapter_state_sha256")
            or ema.get("materialized_model_state_sha256")
            != payload.get("ema_adapter_state_sha256")):
        raise RuntimeError(
            f"{expected_arm} EMA seal does not bind outer raw/EMA states")
    normalizer = payload.get("normalizer")
    if (not isinstance(normalizer, Mapping)
            or set(normalizer) != {
                "mean_input", "mean_target", "std_input", "std_target"}):
        raise ValueError(f"{expected_arm} normalizer changed")
    mechanics = payload.get("mechanics")
    required_mechanics = (
        "update0_bitwise_identity",
        "direction_head_gradient_u1",
        "history_encoder_gradient_u2",
        "direction_gate_gradient_u2",
    )
    if (not isinstance(mechanics, Mapping)
            or not all(mechanics.get(key) is True for key in required_mechanics)):
        raise ValueError(f"{expected_arm} mechanics evidence is incomplete")
    return dict(payload), {
        "path": str(path.resolve()),
        "file_sha256": actual_sha,
        "arm": expected_arm,
        "update": phase_train.UPDATES,
        "raw_state_sha256": payload["raw_adapter_state_sha256"],
        "ema_state_sha256": payload["ema_adapter_state_sha256"],
        "sample_schedule_sha256": payload["sample_schedule_sha256"],
        "initial_adapter_state_sha256": mechanics[
            "initial_adapter_state_sha256"],
    }


def _validate_pooled_file_seal(
        path: Path, actual_endpoint_sha: str,
        payload: Mapping[str, object],
        ) -> dict[str, object]:
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    if not seal_path.is_file():
        raise FileNotFoundError(f"pooled endpoint seal is missing: {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, Mapping):
        raise TypeError("pooled endpoint seal must be a mapping")
    expected = {
        "schema": pooled_train.SEAL_SCHEMA,
        "endpoint_schema": pooled_train.ENDPOINT_SCHEMA,
        "experiment_schema": pooled_train.SCHEMA,
        "arm": pooled_train.ARM,
        "update": pooled_train.UPDATES,
        "path": str(path.resolve()),
        "file_sha256": actual_endpoint_sha,
        "state_sha256": {
            "raw_adapter_state": payload["raw_adapter_state_sha256"],
            "ema_adapter_state": payload["ema_adapter_state_sha256"],
        },
        "base_checkpoint_sha256": payload["base_checkpoint"]["sha256"],
        "sample_schedule_sha256": payload["sample_schedule_sha256"],
        "strict_load": True,
        "finite": True,
        "temporal_mixer_absent": True,
    }
    if set(seal) != set(expected) | {"seal_metadata_sha256"}:
        raise RuntimeError("pooled endpoint seal fields changed")
    for key, value in expected.items():
        if seal.get(key) != value:
            raise RuntimeError(f"pooled endpoint seal changed: {key}")
    metadata = dict(seal)
    stored_metadata_sha = metadata.pop("seal_metadata_sha256", None)
    if stored_metadata_sha != real4k.canonical_sha(metadata):
        raise RuntimeError("pooled endpoint seal metadata SHA changed")
    return {
        "path": str(seal_path.resolve()),
        "file_sha256": phase_train.sha256(seal_path),
        "metadata_sha256": stored_metadata_sha,
    }


def validate_pooled_endpoint(
        path: Path, expected_sha: str, expected_base_sha: str,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the topology-changing endpoint before any target access."""
    actual_sha = phase_train.require_sha(path, expected_sha)
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise TypeError("pooled adapter endpoint must be a mapping")
    if (payload.get("schema") != pooled_train.ENDPOINT_SCHEMA
            or payload.get("experiment_schema") != pooled_train.SCHEMA
            or payload.get("arm") != pooled_train.ARM
            or int(payload.get("update", -1)) != pooled_train.UPDATES):
        raise ValueError("pooled endpoint is not a complete u4000 endpoint")
    if payload.get("sample_schedule_sha256") != (
            pooled_train.EXPECTED_SAMPLE_SCHEDULE_SHA256):
        raise ValueError("pooled endpoint sample stream changed")
    source = payload.get("base_checkpoint")
    if (not isinstance(source, Mapping)
            or source.get("sha256") != expected_base_sha
            or not isinstance(source.get("model_state_sha256"), str)):
        raise ValueError("pooled endpoint does not reference the frozen base")
    config = _adapter_config(payload.get("adapter_config", {}))
    if config.mixer_mode != pooled_train.ARM:
        raise ValueError("pooled endpoint mixer mode changed")
    if int(payload.get("adapter_parameters", -1)) != (
            pooled_train.POOLED_ADAPTER_PARAMETERS):
        raise ValueError("pooled endpoint parameter count changed")

    states: dict[str, Mapping[str, torch.Tensor]] = {}
    for endpoint_kind, state_key, sha_key in (
            ("raw", "raw_adapter_state", "raw_adapter_state_sha256"),
            ("ema", "ema_adapter_state", "ema_adapter_state_sha256")):
        state = payload.get(state_key)
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"pooled endpoint lacks {endpoint_kind} state")
        if any(str(name).startswith("temporal_mixer.") for name in state):
            raise RuntimeError("pooled endpoint retained temporal mixer state")
        actual_state_sha = ema_lib.tensor_state_sha256(state)
        if actual_state_sha != payload.get(sha_key):
            raise RuntimeError(
                f"pooled {endpoint_kind} state SHA changed")
        probe = CausalPhaseDirectionAdapter(config).float()
        probe.load_state_dict(state, strict=True)
        if sum(parameter.numel() for parameter in probe.parameters()) != (
                pooled_train.POOLED_ADAPTER_PARAMETERS):
            raise RuntimeError(
                f"pooled {endpoint_kind} strict-load size changed")
        if any(value.is_floating_point()
               and not bool(torch.isfinite(value).all())
               for value in state.values()):
            raise FloatingPointError(
                f"pooled {endpoint_kind} state is non-finite")
        states[endpoint_kind] = state

    ema = payload.get("ema")
    if (not isinstance(ema, Mapping)
            or int(ema.get("updates", -1)) != pooled_train.UPDATES):
        raise ValueError("pooled endpoint EMA is not u4000")
    ema_metadata = dict(ema)
    stored_ema_sha = ema_metadata.pop("seal_metadata_sha256", None)
    if stored_ema_sha != ema_lib.canonical_sha256(ema_metadata):
        raise RuntimeError("pooled endpoint EMA metadata seal changed")
    if (ema.get("raw_model_state_sha256")
            != payload.get("raw_adapter_state_sha256")
            or ema.get("materialized_model_state_sha256")
            != payload.get("ema_adapter_state_sha256")):
        raise RuntimeError(
            "pooled EMA seal does not bind outer raw/EMA states")

    normalizer = payload.get("normalizer")
    if (not isinstance(normalizer, Mapping)
            or set(normalizer) != {
                "mean_input", "mean_target", "std_input", "std_target"}
            or any(tuple(torch.as_tensor(value).shape) != (3,)
                   for value in normalizer.values())
            or any(not bool(torch.isfinite(torch.as_tensor(value)).all())
                   for value in normalizer.values())
            or any(not bool((torch.as_tensor(normalizer[name]) > 0).all())
                   for name in ("std_input", "std_target"))):
        raise ValueError("pooled endpoint normalizer changed")
    mechanics = payload.get("mechanics")
    required_mechanics = (
        "update0_bitwise_identity",
        "direction_head_gradient_u1",
        "frame_history_gradient_u2",
        "forecast_gradient_u2",
        "direction_gate_gradient_u2",
        "temporal_mixer_absent",
    )
    if (not isinstance(mechanics, Mapping)
            or not all(mechanics.get(key) is True for key in required_mechanics)
            or mechanics.get("temporal_mixer_state_keys") != []
            or not isinstance(
                mechanics.get("initial_common_state_sha256"), str)):
        raise ValueError("pooled endpoint mechanics evidence is incomplete")
    file_seal = _validate_pooled_file_seal(path, actual_sha, payload)
    return dict(payload), {
        "path": str(path.resolve()),
        "file_sha256": actual_sha,
        "file_seal": file_seal,
        "arm": pooled_train.ARM,
        "update": pooled_train.UPDATES,
        "raw_state_sha256": payload["raw_adapter_state_sha256"],
        "ema_state_sha256": payload["ema_adapter_state_sha256"],
        "sample_schedule_sha256": payload["sample_schedule_sha256"],
        "initial_common_state_sha256": mechanics[
            "initial_common_state_sha256"],
        "temporal_mixer_absent": True,
    }


def validate_pair(
        control: Mapping[str, object], causal: Mapping[str, object]) -> None:
    for key in ("sample_schedule_sha256",):
        if control.get(key) != causal.get(key):
            raise RuntimeError(f"matched endpoint {key} differs")
    control_mechanics = control["mechanics"]
    causal_mechanics = causal["mechanics"]
    if control_mechanics.get("initial_adapter_state_sha256") != (
            causal_mechanics.get("initial_adapter_state_sha256")):
        raise RuntimeError("matched arms did not start from one adapter state")
    control_config = asdict(_adapter_config(control["adapter_config"]))
    causal_config = asdict(_adapter_config(causal["adapter_config"]))
    control_config.pop("mixer_mode")
    causal_config.pop("mixer_mode")
    if control_config != causal_config:
        raise RuntimeError("matched arms differ outside mixer_mode")
    for name in ("mean_input", "mean_target", "std_input", "std_target"):
        if not torch.equal(control["normalizer"][name], causal["normalizer"][name]):
            raise RuntimeError(f"matched endpoint normalizer differs: {name}")


def validate_three_arm_fairness(
        control: Mapping[str, object], causal: Mapping[str, object],
        pooled: Mapping[str, object],
        ) -> dict[str, object]:
    """Bind the topology-changing arm by shared, never full-state, init."""
    validate_pair(control, causal)
    if pooled.get("sample_schedule_sha256") != (
            pooled_train.EXPECTED_SAMPLE_SCHEDULE_SHA256):
        raise RuntimeError("pooled endpoint does not use the sealed stream")
    for reference in (control, causal):
        if reference.get("sample_schedule_sha256") != pooled.get(
                "sample_schedule_sha256"):
            raise RuntimeError("pooled endpoint sample_schedule_sha256 differs")
        for key in ("sha256", "model_state_sha256"):
            left = reference["base_checkpoint"].get(key)
            right = pooled["base_checkpoint"].get(key)
            if left is not None and right is not None and left != right:
                raise RuntimeError(f"pooled endpoint base {key} differs")
        reference_config = asdict(_adapter_config(
            reference["adapter_config"]))
        pooled_config = asdict(_adapter_config(pooled["adapter_config"]))
        reference_config.pop("mixer_mode")
        pooled_config.pop("mixer_mode")
        if reference_config != pooled_config:
            raise RuntimeError("pooled endpoint differs outside mixer topology")
        for name in ("mean_input", "mean_target", "std_input", "std_target"):
            if not torch.equal(
                    torch.as_tensor(reference["normalizer"][name]),
                    torch.as_tensor(pooled["normalizer"][name])):
                raise RuntimeError(
                    f"pooled endpoint normalizer differs: {name}")

    initialization = pooled_train.common_initialization_report()
    actual_common_sha = pooled["mechanics"].get(
        "initial_common_state_sha256")
    if actual_common_sha != initialization["common_state_sha256"]:
        raise RuntimeError("pooled arm did not start from the sealed common state")
    return {
        "comparison_basis": "shared_non_mixer_tensors_only",
        "seed": initialization["seed"],
        "common_tensors": initialization["common_tensors"],
        "common_state_sha256": initialization["common_state_sha256"],
        "bitwise_equal_to_causal_initialization": initialization[
            "bitwise_equal_to_causal_initialization"],
        "full_state_compared_across_topologies": False,
    }


def _normalizer(payload: Mapping[str, object], device: torch.device):
    values = payload["normalizer"]
    return smu1.FoldNormalizer(tuple(
        values[name] for name in (
            "mean_input", "mean_target", "std_input", "std_target")),
        device=device)


def _load_adapter(
        endpoint: Mapping[str, object], endpoint_kind: str,
        device: torch.device) -> CausalPhaseDirectionAdapter:
    config = _adapter_config(endpoint["adapter_config"])
    adapter = CausalPhaseDirectionAdapter(config).float()
    adapter.load_state_dict(
        endpoint[f"{endpoint_kind}_adapter_state"], strict=True)
    return adapter.to(device).eval()


def _decorate(metrics: Mapping[str, float]) -> dict[str, float]:
    result = {key: float(value) for key, value in metrics.items()}
    result["official_point_contribution_46_10_9"] = (
        0.46 * result["rel_l2_score"]
        + 0.10 * result["tke_rel_l2_score"]
        + 0.09 * result["mvpe_rel_l2_score"])
    result["official_normalized_point_46_10_9"] = (
        result["official_point_contribution_46_10_9"] / 0.65)
    result["non_time_contribution_46_10_9_25"] = (
        result["official_point_contribution_46_10_9"]
        + 0.25 * result["native_sps_score"])
    return result


def _finalize_nested(states: Mapping[str, dict]) -> dict[str, dict[str, float]]:
    return {
        name: _decorate(smu1.finalize_metric_state(state))
        for name, state in states.items()
    }


def _delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {
        key: float(left[key]) - float(right[key])
        for key in left.keys() & right.keys()
        if isinstance(left[key], (int, float))
        and isinstance(right[key], (int, float))
    }


def _comparisons(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    """Build every comparison supported by the supplied fixed states.

    The formal three-arm replay still supplies all names in ``MODEL_NAMES``.
    Keeping this helper subset-safe lets narrower, independently sealed
    experiments reuse the metric loop without fabricating pooled endpoints.
    """
    result: dict[str, object] = {
        "each_minus_b210": {
            name: _delta(values, metrics["b210"])
            for name, values in metrics.items() if name != "b210"
        },
    }
    pairs = {
        "causal_minus_control_raw": ("causal_raw", "spatial_control_raw"),
        "causal_minus_control_ema": ("causal_ema", "spatial_control_ema"),
        "pooled_minus_control_raw": ("pooled_raw", "spatial_control_raw"),
        "pooled_minus_control_ema": ("pooled_ema", "spatial_control_ema"),
        "pooled_minus_causal_raw": ("pooled_raw", "causal_raw"),
        "pooled_minus_causal_ema": ("pooled_ema", "causal_ema"),
    }
    for label, (left, right) in pairs.items():
        if left in metrics and right in metrics:
            result[label] = _delta(metrics[left], metrics[right])
    return result


@torch.inference_mode()
def evaluate(
        base, adapters: Mapping[str, CausalPhaseDirectionAdapter],
        dataset, aoas: Sequence[int], trajectory_labels: Sequence[str],
        normalizer, device: torch.device,
        *, model_names: Sequence[str] = MODEL_NAMES,
        ) -> dict[str, object]:
    model_names = tuple(model_names)
    if (not model_names or model_names[0] != "b210"
            or len(model_names) != len(set(model_names))
            or not set(model_names).issubset(MODEL_NAMES)):
        raise ValueError("metric replay model names are invalid")
    expected_adapters = set(model_names) - {"b210"}
    if set(adapters) != expected_adapters:
        raise ValueError(
            f"metric replay requires adapters {sorted(expected_adapters)}")
    pooled = {name: smu1.new_metric_state() for name in model_names}
    by_aoa = {
        str(aoa): {name: smu1.new_metric_state() for name in model_names}
        for aoa in sorted(set(aoas))
    }
    by_trajectory = {
        label: {name: smu1.new_metric_state() for name in model_names}
        for label in sorted(set(trajectory_labels))
    }
    prediction_hashers = {name: hashlib.sha256() for name in model_names}
    target_hasher = hashlib.sha256()
    endpoint_max_delta = {name: 0.0 for name in model_names if name != "b210"}
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)

    for start in range(0, len(dataset), 8):
        stop = min(start + 8, len(dataset))
        histories, targets = zip(*(dataset[index] for index in range(start, stop)))
        history = torch.stack(histories).to(device)
        target = torch.stack(targets).to(device)
        history_normalized, _unused = normalizer.preprocess(history, target)
        base_prediction_norm, base_lower_norm, base_upper_norm = (
            base.forward_distribution(history_normalized))
        base_prediction = normalizer.postprocess_prediction(
            base_prediction_norm.float())
        base_prediction[..., 2] = 0.0
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
            "b210": (
                base_prediction,
                base_lower_norm.float() * scale,
                base_upper_norm.float() * scale,
            ),
        }
        base_lower_absolute = (
            base_prediction[..., :2] - base_lower_norm.float() * scale)
        base_upper_absolute = (
            base_prediction[..., :2] + base_upper_norm.float() * scale)
        for name, adapter in adapters.items():
            prediction_norm = adapter(
                history_normalized, base_prediction_norm.detach())
            point_delta = prediction_norm[..., :2] - base_prediction_norm[..., :2]
            lower_norm = base_lower_norm + point_delta
            upper_norm = base_upper_norm - point_delta
            prediction = normalizer.postprocess_prediction(
                prediction_norm.float())
            prediction[..., 2] = 0.0
            lower = lower_norm.float() * scale
            upper = upper_norm.float() * scale
            outputs[name] = (prediction, lower, upper)
            reconstructed_lower = prediction[..., :2] - lower
            reconstructed_upper = prediction[..., :2] + upper
            endpoint_max_delta[name] = max(
                endpoint_max_delta[name],
                float((reconstructed_lower - base_lower_absolute).abs().max()),
                float((reconstructed_upper - base_upper_absolute).abs().max()),
            )

        target_hasher.update(target.detach().cpu().contiguous().numpy().tobytes())
        for name, (prediction, lower, upper) in outputs.items():
            prediction_hashers[name].update(
                prediction.detach().cpu().contiguous().numpy().tobytes())
            smu1.update_metric_state(
                pooled[name], prediction, target, lower, upper)
            for local_index, global_index in enumerate(range(start, stop)):
                selection = slice(local_index, local_index + 1)
                smu1.update_metric_state(
                    by_aoa[str(aoas[global_index])][name],
                    prediction[selection], target[selection],
                    lower[selection], upper[selection])
                smu1.update_metric_state(
                    by_trajectory[trajectory_labels[global_index]][name],
                    prediction[selection], target[selection],
                    lower[selection], upper[selection])

    pooled_metrics = _finalize_nested(pooled)
    aoa_metrics = {}
    for label, states in by_aoa.items():
        metrics = _finalize_nested(states)
        aoa_metrics[label] = {
            "models": metrics,
            "comparisons": _comparisons(metrics),
        }
    trajectory_metrics = {}
    for label, states in by_trajectory.items():
        metrics = _finalize_nested(states)
        trajectory_metrics[label] = {
            "models": metrics,
            "comparisons": _comparisons(metrics),
        }
    return {
        "windows": len(dataset),
        "pooled": pooled_metrics,
        "comparisons": _comparisons(pooled_metrics),
        "by_aoa": aoa_metrics,
        "by_trajectory": trajectory_metrics,
        "absolute_endpoint_max_delta_from_b210": endpoint_max_delta,
        "prediction_sha256": {
            name: hasher.hexdigest()
            for name, hasher in prediction_hashers.items()
        },
        "target_sha256": target_hasher.hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.acknowledge_target_access != ACKNOWLEDGEMENT:
        raise PermissionError("clean716 replay requires the exact acknowledgement")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    expected_base_sha = args.base_checkpoint_sha256.strip().lower()
    # Barrier: validate all complete endpoint containers and all six tensor
    # states before reading either manifest or any Real H5 target.
    control, control_report = validate_endpoint(
        args.spatial_control_endpoint,
        args.spatial_control_endpoint_sha256,
        "framewise_spatial_control", expected_base_sha)
    causal, causal_report = validate_endpoint(
        args.causal_endpoint,
        args.causal_endpoint_sha256,
        "causal_temporal", expected_base_sha)
    pooled_endpoint, pooled_report = validate_pooled_endpoint(
        args.pooled_endpoint,
        args.pooled_endpoint_sha256,
        expected_base_sha)
    fairness = validate_three_arm_fairness(
        control, causal, pooled_endpoint)
    (_base_payload, base_config, base_state,
     base_report) = phase_train.load_base_payload(
        args.base_checkpoint, expected_base_sha)
    for endpoint in (control, causal, pooled_endpoint):
        source_state_sha = endpoint["base_checkpoint"].get(
            "model_state_sha256")
        if (source_state_sha is not None
                and source_state_sha != base_report["model_state_sha256"]):
            raise RuntimeError("endpoint base model-state SHA changed")

    device = torch.device(args.device)
    normalizer = _normalizer(control, device)
    base = phase_train.build_base(
        base_config, base_state, normalizer, device)
    adapters = {
        "spatial_control_raw": _load_adapter(control, "raw", device),
        "spatial_control_ema": _load_adapter(control, "ema", device),
        "causal_raw": _load_adapter(causal, "raw", device),
        "causal_ema": _load_adapter(causal, "ema", device),
        "pooled_raw": _load_adapter(pooled_endpoint, "raw", device),
        "pooled_ema": _load_adapter(pooled_endpoint, "ema", device),
    }

    fit_rows, fit_manifest_report = real4k.load_clean_manifest(
        args.clean_manifest)
    tail_rows = real4k.load_clean716_rows(
        args.clean716_manifest, fit_rows)
    clean716_manifest_report = {
        "path": str(args.clean716_manifest.resolve()),
        "file_sha256": phase_train.sha256(args.clean716_manifest),
        "semantic_sha256": real4k.canonical_sha(json.loads(
            args.clean716_manifest.read_text(encoding="utf-8"))),
        "windows": sum(len(row["train_starts"]) for row in tail_rows),
    }
    access = real4k.PrefixAccessAudit()
    # PrefixAccessAudit carries one immutable file guard.  The endpoint barrier
    # above already validated all three; guard the newly trained pooled file
    # against mutation between validation and every clean716 H5 open.
    access.record_endpoint_seal({
        "path": str(args.pooled_endpoint.resolve()),
        "file_sha256": pooled_report["file_sha256"],
    })
    trajectories, windows = real4k.load_real_rows(
        args.real_dir, tail_rows, phase="clean716", audit=access)
    dataset = real4k.RealWindowDataset(trajectories, windows)
    aoas = [int(tail_rows[window.trajectory]["aoa"]) for window in windows]
    trajectory_labels = [
        str(tail_rows[window.trajectory]["file"]) for window in windows]
    evaluation = evaluate(
        base, adapters, dataset, aoas, trajectory_labels,
        normalizer, device)
    result = {
        "schema": SCHEMA,
        "evidence_class": "post_hoc_same_clean716_diagnostic",
        "endpoint_barrier": {
            "validated_before_manifest_or_target_open": True,
            "spatial_control": control_report,
            "causal": causal_report,
            "pooled": pooled_report,
            "fairness": fairness,
            "six_states": [
                "spatial_control_raw", "spatial_control_ema",
                "causal_raw", "causal_ema",
                "pooled_raw", "pooled_ema",
            ],
        },
        "base": base_report,
        "fit_manifest": fit_manifest_report,
        "clean716_manifest": clean716_manifest_report,
        "target_access_audit": access.events,
        "evaluation": evaluation,
        "checkpoint_selection": False,
        "submission_authorized": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(args.output_json.name + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(args.output_json)
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_phase_three_arm_clean716_complete",
        "schema": result["schema"],
        "output_json": str(args.output_json.resolve()),
        "pooled": result["evaluation"]["pooled"],
        "comparisons": result["evaluation"]["comparisons"],
    }, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
