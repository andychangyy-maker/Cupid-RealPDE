#!/usr/bin/env python3
"""Fixed-endpoint Real4k survival test for the combined MS Sim start.

Only one model is trained: the sealed dense-stride5 + connected-credit +
shifted-teacher Sim EMA endpoint receives the already established mature
Real4k recipe.  Its raw-u4000 and EMA-u4000 states are sealed before clean716
is opened.  The comparison model is the existing connected-only Real4k EMA
endpoint; it is immutable and is never trained again.

This isolates the Sim start.  Real windows, augmentation, sample schedule,
normalizer, optimizer, loss, update count, evaluator, and model configuration
are required to match the frozen connected-only reference exactly.  There is
no endpoint selection, score gate, UQ, or package construction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_msclosure_real_paired4k as paired  # noqa: E402
from models.multiple_shooting_closure import (  # noqa: E402
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
)


SCHEMA = "msclosure_combined_sim_start_real4k_survival_clean716_v1"
RAW_ENDPOINT_SCHEMA = "msclosure_combined_sim_start_real4k_raw_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_MSCLOSURE_COMBINED_SIM_START_REAL4K_CLEAN716"

COMBINED_SIM_SCHEMA = (
    "multiple_shooting_dense_stride5_shifted_teacher_connected_held10_v1")
COMBINED_SIM_FILE_SHA256 = (
    "741470587e7d9791399673dd7cf194755fc282341443c7c1c5a5f4479afa6d1e")
COMBINED_SIM_STATE_SHA256 = (
    "47a64077134559aa9cc1bacdd6d8a3d581c66ca47dd7994f2296646e4df6f300")
SIM_NORMALIZER_SHA256 = (
    "bc8f8123745de6e1d8a1cc898283c48cf77e62c749ea0aaf2cfe4066690c99cf")
SIM_TRAINING_MANIFEST_SHA256 = (
    "ac3969636101f6363bd04380aee8c8e3e0abf1fd5af2ba7576000e62a2d9af89")
SIM_SHIFTED_TEACHER_SHA256 = (
    "abe5d1e3c21bbc2b688f6b878490ea046d4494c877b3288b03ba4d872d070920")

CONNECTED_SIM_FILE_SHA256 = (
    "b25fb2bd434aad445096ef92ae353ae994cafff9e34895e5cc782c8fff2debd8")
CONNECTED_SIM_STATE_SHA256 = (
    "e5c01ea2c0f0dceb44b3557eb358299aabc19b44f207e0d27cc9866856d37899")
CONNECTED_REAL_REFERENCE_FILE_SHA256 = (
    "cf1fca2971ec8bfd5d0dca5db326febc331b473bbf4617f6f41a1a5dd71fa4fd")
CONNECTED_REAL_REFERENCE_STATE_SHA256 = (
    "5d1fc0335098d349f82b3fe8b1e74883b60bfbb276b20f008f1c37dfb71475fc")
REAL_NORMALIZER_SHA256 = (
    "5cb147ec3d79e9fe0ff13090163c5e5d93c9230bc580bf21df3ab35fd3f6e572")
REAL_SAMPLE_SCHEDULE_SHA256 = (
    "9381a97fc114e4234a889d686158533d0088701676b683d268d12d4203f1be43")
REAL_FIT_MANIFEST_FILE_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")

REFERENCE = "connected_only_real4k_ema_frozen"
CANDIDATE_RAW = "combined_sim_start_real4k_raw_u4000"
CANDIDATE_EMA = "combined_sim_start_real4k_ema999_u4000"

PRIOR_REFERENCE_CLEAN716 = {
    "rel_l2_score": 95.939067,
    "tke_rel_l2_score": 80.260182,
    "mvpe_rel_l2_score": 96.644709,
    "point_contribution_46_10_9": 60.856013,
}


def _hex_sha(value: str, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def mature_candidate_contract() -> dict[str, object]:
    return paired.real_contract(
        arm=paired.CANDIDATE_ARM,
        windows=paired.real.EXPECTED_OPTIMIZER_WINDOWS,
        stride=paired.real.OPTIMIZER_STRIDE,
        augmentation=True,
    )


def expected_schedule_sha256() -> str:
    schedule = paired.real.make_sample_schedule(
        paired.real.EXPECTED_OPTIMIZER_WINDOWS,
        paired.UPDATES,
        paired.BATCH_SIZE,
    )
    return paired.canonical_sha(
        [[int(index) for index in batch] for batch in schedule])


def validate_fit_manifest_file(path: Path) -> str:
    """Require the tail-free runtime fit manifest, never split_manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = paired.real.file_sha256(path)
    if observed != REAL_FIT_MANIFEST_FILE_SHA256:
        raise RuntimeError(
            "Real fit manifest file identity changed; use the tail-free runtime_train_manifest.json")
    return observed


def validate_combined_sim_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact combined Sim endpoint, including its interventions."""
    required = {
        "schema", "endpoint_kind", "updates", "rollout_mode",
        "segment_gradient_mode", "model_config", "normalizer",
        "training_window_manifest", "shifted_teacher", "git", "model_state",
    }
    missing = required.difference(payload)
    if missing:
        raise KeyError("combined Sim checkpoint lacks: " + ", ".join(sorted(missing)))
    if payload["schema"] != COMBINED_SIM_SCHEMA:
        raise ValueError("start checkpoint is not the combined MS Sim endpoint")
    if payload["endpoint_kind"] != "ema999_u12000" or int(payload["updates"]) != 12_000:
        raise ValueError("combined MS start must be the sealed EMA999-u12000 endpoint")
    if (payload["rollout_mode"] != "carry_detached"
            or payload["segment_gradient_mode"] != "connected"):
        raise ValueError("combined MS start lost connected rollout semantics")
    if dict(payload["model_config"]) != paired.expected_model_config():
        raise ValueError("combined MS model_config differs from the frozen 23M recipe")
    if paired.canonical_sha(dict(payload["normalizer"])) != SIM_NORMALIZER_SHA256:
        raise RuntimeError("combined MS Sim normalizer identity changed")
    if paired.canonical_sha(dict(payload["training_window_manifest"])) != SIM_TRAINING_MANIFEST_SHA256:
        raise RuntimeError("combined MS dense training-window manifest changed")
    if paired.canonical_sha(dict(payload["shifted_teacher"])) != SIM_SHIFTED_TEACHER_SHA256:
        raise RuntimeError("combined MS shifted-teacher contract changed")
    shifted = payload["shifted_teacher"]
    if not isinstance(shifted, Mapping) or dict(shifted) != {
        "shift_frames": 5,
        "overlap_frames": 15,
        "windows_checked": 15_360,
        "all_shifted_histories_in_released_trajectories": True,
        "maximum_main_start": 955,
    }:
        raise RuntimeError("combined MS shifted-teacher semantics changed")
    state = payload["model_state"]
    if not isinstance(state, Mapping):
        raise TypeError("combined MS model_state must be a mapping")
    paired._require_fp32_finite_state(state)
    for name in ("mean_input", "mean_target", "std_input", "std_target"):
        value = torch.as_tensor(payload["normalizer"][name], dtype=torch.float32)
        if name not in state or not torch.equal(state[name].cpu(), value):
            raise RuntimeError(f"combined MS state buffer {name} disagrees with metadata")
    state_sha = base.state_sha(state)
    if state_sha != COMBINED_SIM_STATE_SHA256:
        raise RuntimeError("combined MS model state identity changed")
    return {
        "schema": str(payload["schema"]),
        "endpoint_kind": str(payload["endpoint_kind"]),
        "updates": int(payload["updates"]),
        "rollout_mode": str(payload["rollout_mode"]),
        "segment_gradient_mode": str(payload["segment_gradient_mode"]),
        "model_config_sha256": paired.canonical_sha(dict(payload["model_config"])),
        "sim_normalizer_sha256": SIM_NORMALIZER_SHA256,
        "training_window_manifest_sha256": SIM_TRAINING_MANIFEST_SHA256,
        "shifted_teacher_sha256": SIM_SHIFTED_TEACHER_SHA256,
        "model_state_sha256": state_sha,
    }


def load_combined_sim_checkpoint(
        path: Path, expected_sha256: str, device: torch.device,
        ) -> tuple[MultipleShootingClosure, dict[str, object]]:
    expected_sha256 = _hex_sha(expected_sha256, "--combined-start-sha256")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = paired.real.file_sha256(path)
    if observed_sha != expected_sha256 or observed_sha != COMBINED_SIM_FILE_SHA256:
        raise RuntimeError("combined MS checkpoint file SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("combined MS checkpoint payload must be a mapping")
    report = validate_combined_sim_payload(payload)
    model = MultipleShootingClosure(
        MultipleShootingClosureConfig(**dict(payload["model_config"])),
        rollout_mode=str(payload["rollout_mode"]),
        segment_gradient_mode=str(payload["segment_gradient_mode"]),
    ).float()
    model.load_state_dict(payload["model_state"], strict=True)
    if base.state_sha(model.state_dict()) != COMBINED_SIM_STATE_SHA256:
        raise RuntimeError("combined MS strict-load state identity mismatch")
    report.update({
        "path": str(path),
        "file_sha256": observed_sha,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    })
    return model.to(device), report


def _reference_start_contract(payload: Mapping[str, object]) -> dict[str, object]:
    start = payload.get("start_ms")
    if not isinstance(start, Mapping):
        raise ValueError("frozen Real reference lacks start_ms provenance")
    expected = {
        "schema": paired.EXPECTED_SIM_SCHEMA,
        "endpoint_kind": "ema999_u12000",
        "updates": 12_000,
        "model_config_sha256": paired.canonical_sha(paired.expected_model_config()),
        "sim_normalizer_sha256": SIM_NORMALIZER_SHA256,
        "model_state_sha256": CONNECTED_SIM_STATE_SHA256,
        "file_sha256": CONNECTED_SIM_FILE_SHA256,
        "parameters": 22_788_968,
    }
    observed = {key: start.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError("frozen Real reference is not descended from the connected-only Sim endpoint")
    return observed


def validate_reference_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate frozen Real endpoint provenance and the mature Real contract."""
    required = {
        "schema", "arm", "update", "model_config", "rollout_mode",
        "segment_gradient_mode", "model_state", "model_state_sha256",
        "normalizer", "normalizer_sha256", "start_ms", "contract",
        "sample_schedule_sha256", "ema",
    }
    missing = required.difference(payload)
    if missing:
        raise KeyError("frozen connected Real checkpoint lacks: " + ", ".join(sorted(missing)))
    if (payload["schema"] != paired.ENDPOINT_SCHEMA
            or payload["arm"] != paired.CANDIDATE_ARM
            or int(payload["update"]) != paired.UPDATES):
        raise ValueError("frozen reference is not the connected mature-augmentation EMA-u4000")
    if (payload["rollout_mode"] != "carry_detached"
            or payload["segment_gradient_mode"] != "connected"):
        raise ValueError("frozen Real reference rollout semantics changed")
    if dict(payload["model_config"]) != paired.expected_model_config():
        raise ValueError("frozen Real reference model_config changed")
    if dict(payload["contract"]) != mature_candidate_contract():
        raise RuntimeError("frozen Real reference did not use the mature Real4k recipe")
    if payload["sample_schedule_sha256"] != REAL_SAMPLE_SCHEDULE_SHA256:
        raise RuntimeError("frozen Real reference sample schedule changed")
    if payload["normalizer_sha256"] != REAL_NORMALIZER_SHA256:
        raise RuntimeError("frozen Real reference normalizer identity changed")
    normalizer = payload["normalizer"]
    expected_normalizer_keys = {"mean_input", "mean_target", "std_input", "std_target"}
    if not isinstance(normalizer, Mapping) or set(normalizer) != expected_normalizer_keys:
        raise ValueError("frozen Real reference normalizer record changed")
    if base.state_sha(normalizer) != REAL_NORMALIZER_SHA256:
        raise RuntimeError("frozen Real reference normalizer tensors changed")
    state = payload["model_state"]
    if not isinstance(state, Mapping):
        raise TypeError("frozen Real reference model_state must be a mapping")
    paired._require_fp32_finite_state(state)
    for name in expected_normalizer_keys:
        value = torch.as_tensor(normalizer[name], dtype=torch.float32)
        if name not in state or not torch.equal(state[name].cpu(), value):
            raise RuntimeError(f"frozen Real state buffer {name} disagrees with metadata")
    state_sha = base.state_sha(state)
    if (state_sha != CONNECTED_REAL_REFERENCE_STATE_SHA256
            or payload["model_state_sha256"] != state_sha):
        raise RuntimeError("frozen Real reference model state identity changed")
    ema = payload["ema"]
    if not isinstance(ema, Mapping) or int(ema.get("updates", -1)) != paired.UPDATES:
        raise RuntimeError("frozen Real reference is not an EMA-u4000 seal")
    start = _reference_start_contract(payload)
    return {
        "schema": str(payload["schema"]),
        "arm": str(payload["arm"]),
        "update": int(payload["update"]),
        "model_config_sha256": paired.canonical_sha(dict(payload["model_config"])),
        "model_state_sha256": state_sha,
        "normalizer_sha256": str(payload["normalizer_sha256"]),
        "sample_schedule_sha256": str(payload["sample_schedule_sha256"]),
        "start_ms": start,
    }


def validate_reference_checkpoint(path: Path, expected_sha256: str) -> dict[str, object]:
    expected_sha256 = _hex_sha(expected_sha256, "--connected-reference-sha256")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = paired.real.file_sha256(path)
    if observed_sha != expected_sha256 or observed_sha != CONNECTED_REAL_REFERENCE_FILE_SHA256:
        raise RuntimeError("frozen connected Real reference file SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("frozen connected Real reference must be a mapping")
    report = validate_reference_payload(payload)
    model = MultipleShootingClosure(
        MultipleShootingClosureConfig(**dict(payload["model_config"])),
        rollout_mode=str(payload["rollout_mode"]),
        segment_gradient_mode=str(payload["segment_gradient_mode"]),
    ).float()
    model.load_state_dict(payload["model_state"], strict=True)
    if base.state_sha(model.state_dict()) != CONNECTED_REAL_REFERENCE_STATE_SHA256:
        raise RuntimeError("frozen Real reference strict-load state identity mismatch")
    report.update({"path": str(path), "file_sha256": observed_sha})
    return report


def load_reference_model(path: Path, report: Mapping[str, object],
                         device: torch.device) -> MultipleShootingClosure:
    path = path.resolve()
    if paired.real.file_sha256(path) != report["file_sha256"]:
        raise RuntimeError("frozen Real reference changed after preflight")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MultipleShootingClosure(
        MultipleShootingClosureConfig(**dict(payload["model_config"])),
        rollout_mode=str(payload["rollout_mode"]),
        segment_gradient_mode=str(payload["segment_gradient_mode"]),
    ).float()
    model.load_state_dict(payload["model_state"], strict=True)
    if base.state_sha(model.state_dict()) != report["model_state_sha256"]:
        raise RuntimeError("frozen Real reference changed while loading")
    return model.to(device)


class CandidateAccessAudit:
    """Open clean716 only after raw/EMA candidate seals and frozen-ref check."""

    def __init__(self, reference: Mapping[str, object]) -> None:
        self.reference = dict(reference)
        self.events: list[dict[str, object]] = []
        self._opened: dict[str, set[str]] = {"fit": set(), "clean716": set()}
        self.candidate_seals: dict[str, dict[str, object]] = {}

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if filename == paired.real.EXCLUDED_BASENAME:
            raise RuntimeError("excluded trajectory reached combined-start Real audit")
        if phase not in self._opened:
            raise ValueError(f"unsupported access phase {phase!r}")
        if filename in self._opened[phase]:
            raise RuntimeError(f"Real H5 opened twice in {phase}: {filename}")
        if phase == "fit" and self.candidate_seals:
            raise RuntimeError("fit prefix cannot open after candidate endpoint sealing")
        if phase == "clean716":
            if set(self.candidate_seals) != {CANDIDATE_RAW, CANDIDATE_EMA}:
                raise RuntimeError("clean716 cannot open before raw and EMA candidate seals")
            for seal in self.candidate_seals.values():
                path = Path(str(seal["path"]))
                if (not path.is_file()
                        or paired.real.file_sha256(path) != seal["file_sha256"]):
                    raise RuntimeError("sealed combined-start candidate changed before clean716")
            reference_path = Path(str(self.reference["path"]))
            if (not reference_path.is_file()
                    or paired.real.file_sha256(reference_path) != self.reference["file_sha256"]):
                raise RuntimeError("frozen connected reference changed before clean716")

    def record_open(self, row: Mapping[str, object], phase: str, boundary: int) -> None:
        filename = str(row["file"])
        self._opened[phase].add(filename)
        self.events.append({
            "file": filename,
            "phase": phase,
            "channels_opened": ["u", "v"],
            "pressure_opened": False,
            "max_read_frame_exclusive": int(boundary),
            "candidate_raw_and_ema_sealed_before_open": (
                set(self.candidate_seals) == {CANDIDATE_RAW, CANDIDATE_EMA}),
            "reference_file_verified_before_open": True,
        })

    def record_candidate_seal(self, name: str, seal: Mapping[str, object]) -> None:
        if name not in {CANDIDATE_RAW, CANDIDATE_EMA} or name in self.candidate_seals:
            raise RuntimeError("invalid or duplicate combined-start candidate seal")
        self.candidate_seals[name] = dict(seal)


def _seal_raw_endpoint(
        path: Path, model: MultipleShootingClosure, *, start: Mapping[str, object],
        normalizer, contract: Mapping[str, object], schedule_sha256: str,
        ) -> dict[str, object]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    paired._require_fp32_finite_state(state)
    payload = {
        "schema": RAW_ENDPOINT_SCHEMA,
        "endpoint_kind": "raw_u4000",
        "update": paired.UPDATES,
        "model_config": asdict(model.config),
        "rollout_mode": model.rollout_mode,
        "segment_gradient_mode": model.segment_gradient_mode,
        "model_state": state,
        "model_state_sha256": base.state_sha(state),
        "normalizer": paired._normalizer_state(normalizer),
        "normalizer_sha256": paired._normalizer_sha(normalizer),
        "start_ms": {key: value for key, value in start.items() if key != "model_state"},
        "contract": dict(contract),
        "sample_schedule_sha256": schedule_sha256,
        "real_pressure": "unmeasured_exact_zero_and_unscored",
    }
    paired.real.atomic_torch_save(path, payload)
    return {
        "name": CANDIDATE_RAW,
        "path": str(path.resolve()),
        "file_sha256": paired.real.file_sha256(path),
        "model_state_sha256": payload["model_state_sha256"],
        "schema": RAW_ENDPOINT_SCHEMA,
        "update": paired.UPDATES,
    }


def load_raw_model(seal: Mapping[str, object], device: torch.device) -> MultipleShootingClosure:
    path = Path(str(seal["path"]))
    if not path.is_file() or paired.real.file_sha256(path) != seal["file_sha256"]:
        raise RuntimeError("combined-start raw endpoint seal mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != RAW_ENDPOINT_SCHEMA
            or payload.get("endpoint_kind") != "raw_u4000"
            or int(payload.get("update", -1)) != paired.UPDATES):
        raise ValueError("invalid combined-start raw endpoint schema")
    model = MultipleShootingClosure(
        MultipleShootingClosureConfig(**dict(payload["model_config"])),
        rollout_mode=str(payload["rollout_mode"]),
        segment_gradient_mode=str(payload["segment_gradient_mode"]),
    ).float()
    model.load_state_dict(payload["model_state"], strict=True)
    if base.state_sha(model.state_dict()) != seal["model_state_sha256"]:
        raise RuntimeError("combined-start raw endpoint state seal mismatch")
    return model.to(device)


def _metric_row(evaluation: Mapping[str, object]) -> dict[str, float]:
    return paired._metric_row(evaluation)


def _artifact_preflight(args: argparse.Namespace) -> dict[str, object]:
    required = (args.combined_start, args.connected_reference)
    if any(value is None for value in required):
        raise ValueError("preflight requires --combined-start and --connected-reference")
    model, start = load_combined_sim_checkpoint(
        args.combined_start, args.combined_start_sha256, torch.device("cpu"))
    del model
    reference = validate_reference_checkpoint(
        args.connected_reference, args.connected_reference_sha256)
    return {
        "schema": SCHEMA,
        "combined_start": {key: value for key, value in start.items() if key != "model_state"},
        "connected_reference": reference,
        "mature_real_contract": mature_candidate_contract(),
        "expected_sample_schedule_sha256": expected_schedule_sha256(),
        "real_h5_opened": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError("formal combined-start Real4k test requires explicit acknowledgement")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal combined-start Real4k test requires CUDA")
    if args.updates != paired.UPDATES or args.batch_size != paired.BATCH_SIZE:
        raise ValueError("combined-start Real4k test freezes updates4000 and batch8")
    required = (
        args.real_dir, args.clean_manifest, args.clean716_manifest,
        args.combined_start, args.connected_reference, args.output_dir,
    )
    if any(value is None for value in required):
        raise ValueError("formal run requires all data, checkpoint, manifest, and output arguments")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing combined-start Real output")
    # Do not read/hash the clean716-role manifest until candidate endpoints
    # are sealed.  Existence is the only pre-training filesystem check.
    if not args.clean716_manifest.is_file():
        raise FileNotFoundError(args.clean716_manifest)
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError("formal combined-start Real4k test requires clean Git")

    # Both immutable inputs are validated before any released Real H5 is opened.
    reference = validate_reference_checkpoint(
        args.connected_reference, args.connected_reference_sha256)
    template, start = load_combined_sim_checkpoint(
        args.combined_start, args.combined_start_sha256, device)
    start_state = start["model_state"]

    fit_manifest_file_sha256 = validate_fit_manifest_file(args.clean_manifest)
    rows, manifest_report = paired.real.load_clean_manifest(args.clean_manifest)
    audit = CandidateAccessAudit(reference)
    trajectories, sparse_windows = paired.real.load_real_rows(
        args.real_dir, rows, phase="fit", audit=audit)
    normalizer, normalizer_report = paired.real.normalizer_from_sparse_windows(
        trajectories, sparse_windows, device)
    if paired._normalizer_sha(normalizer) != REAL_NORMALIZER_SHA256:
        raise RuntimeError("computed Real normalizer does not match the frozen reference")
    dense_windows = paired.real.dense_windows(trajectories)
    if len(dense_windows) != paired.real.EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError("mature Real dense-window support changed")
    contract = mature_candidate_contract()
    if dict(reference["start_ms"])["file_sha256"] != CONNECTED_SIM_FILE_SHA256:
        raise RuntimeError("connected reference start provenance changed")
    schedule_sha = expected_schedule_sha256()
    if schedule_sha != REAL_SAMPLE_SCHEDULE_SHA256:
        raise RuntimeError("new candidate sample schedule differs from frozen reference")
    dataset = paired.real.RealWindowDataset(trajectories, dense_windows)

    model = paired.clone_start_model(template, start_state, normalizer, device)
    initial_state_sha = base.state_sha(model.state_dict())
    del template
    args.output_dir.mkdir(parents=True)
    training, ema_seal = paired.train_arm(
        model, dataset, normalizer, arm=paired.CANDIDATE_ARM,
        output_dir=args.output_dir, start=start, device=device,
        checkpoint_every=args.checkpoint_every,
    )
    if training["contract"] != contract:
        raise RuntimeError("executed candidate Real contract differs from frozen reference")
    if training["sample_schedule_sha256"] != schedule_sha:
        raise RuntimeError("executed candidate sample schedule differs from frozen reference")
    ema_named_seal = {"name": CANDIDATE_EMA, **ema_seal}
    raw_seal = _seal_raw_endpoint(
        args.output_dir / paired.CANDIDATE_ARM / "raw_u4000.pt",
        model, start=start, normalizer=normalizer, contract=contract,
        schedule_sha256=schedule_sha,
    )
    audit.record_candidate_seal(CANDIDATE_RAW, raw_seal)
    audit.record_candidate_seal(CANDIDATE_EMA, ema_named_seal)
    del model
    torch.cuda.empty_cache()

    # The evaluation set is opened once, only after the candidate endpoints
    # exist immutably.  Every model below sees this exact same dataset object.
    clean716_manifest_sha256 = paired.real.file_sha256(args.clean716_manifest)
    if clean716_manifest_sha256 != paired.EXPECTED_CLEAN716_MANIFEST_SHA256:
        raise RuntimeError("frozen clean716 manifest SHA-256 changed")
    tail_rows = paired.real.load_clean716_rows(args.clean716_manifest, rows)
    tail_trajectories, tail_windows = paired.real.load_real_rows(
        args.real_dir, tail_rows, phase="clean716", audit=audit)
    tail_dataset = paired.real.RealWindowDataset(tail_trajectories, tail_windows)
    tail_aoas = [int(tail_rows[window.trajectory]["aoa"]) for window in tail_windows]

    evaluations: dict[str, object] = {}
    model_loaders = (
        (REFERENCE, lambda: load_reference_model(
            args.connected_reference, reference, device)),
        (CANDIDATE_RAW, lambda: load_raw_model(raw_seal, device)),
        (CANDIDATE_EMA, lambda: paired.load_sealed_endpoint(ema_seal, device)),
    )
    for name, loader in model_loaders:
        endpoint_model = loader()
        evaluations[name] = paired.real.evaluate_clean716(
            endpoint_model, tail_dataset, normalizer, tail_aoas, device)
        del endpoint_model
        torch.cuda.empty_cache()

    metrics = {name: _metric_row(value) for name, value in evaluations.items()}
    result = {
        "schema": SCHEMA,
        "git": git,
        "causal_question": (
            "Does the combined Sim start survive the exact same mature Real4k endpoint "
            "better than the connected-only Sim start?"),
        "combined_start": {key: value for key, value in start.items() if key != "model_state"},
        "frozen_connected_reference": reference,
        "manifest": manifest_report,
        "fit_manifest_file_sha256": fit_manifest_file_sha256,
        "clean716_manifest_sha256": clean716_manifest_sha256,
        "normalizer": normalizer_report,
        "real_normalizer_sha256": paired._normalizer_sha(normalizer),
        "mature_real_contract": contract,
        "sample_schedule_sha256": schedule_sha,
        "candidate_initial_state_sha256_after_real_normalizer": initial_state_sha,
        "candidate_training": training,
        "candidate_endpoints": {CANDIDATE_RAW: raw_seal, CANDIDATE_EMA: ema_named_seal},
        "clean716": evaluations,
        "metrics": metrics,
        "combined_ema_minus_connected_reference": paired.compare_metrics(
            metrics[CANDIDATE_EMA], metrics[REFERENCE]),
        "combined_raw_minus_connected_reference": paired.compare_metrics(
            metrics[CANDIDATE_RAW], metrics[REFERENCE]),
        "combined_ema_minus_combined_raw": paired.compare_metrics(
            metrics[CANDIDATE_EMA], metrics[CANDIDATE_RAW]),
        "reference_rerun_minus_prior_rounded_report": paired.compare_metrics(
            metrics[REFERENCE], PRIOR_REFERENCE_CLEAN716),
        "access_audit": audit.events,
        "selection_gate": None,
        "selection": "none; fixed raw-u4000 and EMA999-u4000 candidate endpoints are both reported",
    }
    paired.real.atomic_torch_save(args.output_dir / "protocol_seal.pt", {
        "schema": SCHEMA,
        "combined_start": result["combined_start"],
        "frozen_connected_reference": reference,
        "candidate_endpoints": result["candidate_endpoints"],
        "candidate_initial_state_sha256_after_real_normalizer": initial_state_sha,
        "real_normalizer_sha256": result["real_normalizer_sha256"],
        "sample_schedule_sha256": schedule_sha,
        "fit_manifest_file_sha256": fit_manifest_file_sha256,
        "fit_manifest_semantic_sha256": manifest_report["semantic_sha256"],
        "clean716_manifest_sha256": clean716_manifest_sha256,
        "clean716_opened_after_candidate_raw_and_ema_seals": True,
    })
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "msclosure_combined_start_real4k_complete",
        "result": str(result_path.resolve()),
        "combined_ema_minus_connected_reference": result[
            "combined_ema_minus_connected_reference"],
    }, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "preflight", "formal"), default="contract")
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--clean716-manifest", type=Path)
    parser.add_argument("--combined-start", type=Path)
    parser.add_argument("--combined-start-sha256", default=COMBINED_SIM_FILE_SHA256)
    parser.add_argument("--connected-reference", type=Path)
    parser.add_argument(
        "--connected-reference-sha256", default=CONNECTED_REAL_REFERENCE_FILE_SHA256)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--updates", type=int, default=paired.UPDATES)
    parser.add_argument("--batch-size", type=int, default=paired.BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps({
            "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "combined_start_file_sha256": COMBINED_SIM_FILE_SHA256,
            "connected_reference_file_sha256": CONNECTED_REAL_REFERENCE_FILE_SHA256,
            "mature_real_contract": mature_candidate_contract(),
            "sample_schedule_sha256": expected_schedule_sha256(),
            "fit_manifest_file_sha256": REAL_FIT_MANIFEST_FILE_SHA256,
            "fit_manifest_semantic_sha256": paired.real.FORMAL_MANIFEST_SHA256,
            "clean716_manifest_sha256": paired.EXPECTED_CLEAN716_MANIFEST_SHA256,
            "selection_gate": None,
        }, sort_keys=True))
        return
    if args.mode == "preflight":
        print(json.dumps(_artifact_preflight(args), sort_keys=True))
        return
    result = run(args)
    print(json.dumps({
        "result": str((args.output_dir / "result.json").resolve()),
        "combined_ema_minus_connected_reference": result[
            "combined_ema_minus_connected_reference"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
