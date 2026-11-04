#!/usr/bin/env python3
"""Strict CPU clean716 replay for the fixed two-stage phase-KD pair.

Both Stage2 endpoint files, companion seals, raw/EMA tensor states, and the
matched-training contract are validated before the clean716 manifest or any
clean716 H5 target is opened.  This entry reports every fixed endpoint; it
cannot select a checkpoint or build a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
        REPO_ROOT / "code", REPO_ROOT / "code" / "training",
        REPO_ROOT / "code" / "evaluation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)
import evaluate_b210_causal_phase_direction_clean716 as phase_eval  # noqa: E402
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_causal_phase_direction_real4k as direct_phase  # noqa: E402
import train_b210_causal_phase_direction_two_stage_kd as stage_train  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real4k  # noqa: E402


SCHEMA = "b210_causal_phase_direction_two_stage_kd_clean716_replay_v1"
ACKNOWLEDGEMENT = "EVALUATE_B210_CAUSAL_PHASE_KD_AFTER_PAIR_SEALS_V1"
CLEAN716_MANIFEST_SHA256 = (
    "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
STAGE2_SCHEDULE_SHA256 = (
    "9381a97fc114e4234a889d686158533d0088701676b683d268d12d4203f1be43")
PUBLIC_MODEL_NAMES = (
    "b210", "direct_raw", "direct_ema", "kd_raw", "kd_ema")
INTERNAL_TO_PUBLIC = {
    "b210": "b210",
    "spatial_control_raw": "direct_raw",
    "spatial_control_ema": "direct_ema",
    "causal_raw": "kd_raw",
    "causal_ema": "kd_ema",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--direct-endpoint", type=Path, required=True)
    parser.add_argument("--direct-endpoint-sha256", required=True)
    parser.add_argument("--kd-endpoint", type=Path, required=True)
    parser.add_argument("--kd-endpoint-sha256", required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--clean716-manifest", type=Path, required=True)
    parser.add_argument("--clean716-manifest-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--acknowledge-target-access", default="")
    return parser.parse_args(argv)


def _expected_adapter_config() -> dict[str, object]:
    return CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig(
        mixer_mode="causal_temporal")).architecture_config()


def _finite_state(state: Mapping[str, torch.Tensor]) -> bool:
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in state.values())


def validate_endpoint(
        path: Path, expected_sha256: str, expected_arm: str,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """Validate one complete Stage2 container, sidecar, raw, and EMA state."""
    if expected_arm not in stage_train.STAGE2_ARMS:
        raise ValueError(f"unknown Stage2 arm {expected_arm!r}")
    file_sha = stage_train._require_file_sha(
        path, expected_sha256, f"Stage2 {expected_arm} endpoint")
    companion = stage_train.verify_companion_seal(path, file_sha)
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, Mapping)
            or payload.get("schema") != stage_train.STAGE2_ENDPOINT_SCHEMA
            or payload.get("experiment_schema") != stage_train.SCHEMA
            or payload.get("stage") != "stage2_matched_real4k"
            or payload.get("arm") != expected_arm
            or payload.get("endpoint") != "fixed_raw_and_ema_u4000"
            or int(payload.get("update", -1)) != stage_train.STAGE2_UPDATES
            or payload.get("reported_states") != ["raw_u4000", "ema_u4000"]
            or payload.get("checkpoint_selected") is not False
            or payload.get("score_gate") is not None
            or payload.get("clean716_opened") is not False
            or payload.get("submission_authorized") is not False):
        raise RuntimeError(f"{expected_arm} is not the fixed Stage2 endpoint")
    if (payload.get("adapter_config") != _expected_adapter_config()
            or int(payload.get("adapter_parameters", -1))
            != stage_train.EXPECTED_ADAPTER_PARAMETERS):
        raise RuntimeError(f"{expected_arm} adapter topology changed")
    if (companion.get("endpoint_schema") != payload["schema"]
            or companion.get("experiment_schema") != payload["experiment_schema"]
            or companion.get("stage") != payload["stage"]
            or companion.get("arm") != expected_arm
            or int(companion.get("update", -1)) != stage_train.STAGE2_UPDATES):
        raise RuntimeError(f"{expected_arm} companion identity changed")

    probes: dict[str, CausalPhaseDirectionAdapter] = {}
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    for kind in ("raw", "ema"):
        key = f"{kind}_adapter_state"
        sha_key = f"{key}_sha256"
        state = payload.get(key)
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"{expected_arm} lacks {kind} state")
        state_sha = ema_lib.tensor_state_sha256(state)
        if (state_sha != payload.get(sha_key)
                or companion.get("state_sha256", {}).get(key) != state_sha
                or not _finite_state(state)):
            raise RuntimeError(f"{expected_arm} {kind} tensor seal changed")
        probe = stage_train._adapter_from_config(payload["adapter_config"])
        probe.load_state_dict(state, strict=True)
        probes[kind] = probe
        states[kind] = state

    ema = payload.get("ema")
    if not isinstance(ema, Mapping):
        raise RuntimeError(f"{expected_arm} lacks EMA metadata")
    full_ema_seal = {**ema, "model_state": states["ema"]}
    restored_ema = ema_lib.FP32EMAShadow.from_seal(
        probes["raw"], full_ema_seal)
    if (restored_ema.updates != stage_train.STAGE2_UPDATES
            or restored_ema.decay != stage_train.EMA_DECAY
            or ema.get("raw_model_state_sha256")
            != payload["raw_adapter_state_sha256"]
            or ema.get("materialized_model_state_sha256")
            != payload["ema_adapter_state_sha256"]):
        raise RuntimeError(f"{expected_arm} EMA does not bind raw/EMA u4000")

    normalizer = payload.get("normalizer")
    if not isinstance(normalizer, Mapping):
        raise RuntimeError(f"{expected_arm} lacks its Real normalizer")
    normalizer_sha = stage_train._normalizer_sha(normalizer)
    if normalizer_sha != payload.get("normalizer_sha256"):
        raise RuntimeError(f"{expected_arm} normalizer tensor SHA changed")
    report = {
        "path": str(path.resolve()),
        "file_sha256": file_sha,
        "companion_seal": companion,
        "arm": expected_arm,
        "raw_state_sha256": payload["raw_adapter_state_sha256"],
        "ema_state_sha256": payload["ema_adapter_state_sha256"],
        "initial_state_sha256": payload.get("mechanics", {}).get(
            "initial_adapter_state_sha256"),
        "normalizer_sha256": normalizer_sha,
        "strict_raw_ema_load": True,
    }
    return dict(payload), report


def _training_contract(payload: Mapping[str, object]) -> dict[str, object]:
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise RuntimeError("Stage2 endpoint lacks training contract")
    keys = (
        "updates", "batch_size", "seed", "sample_schedule_sha256",
        "augmentation", "objective", "optimizer", "lr", "weight_decay",
        "grad_clip", "scheduler", "scheduler_t_max", "ema_decay",
        "ema_initialized_from_real_u0",
    )
    if set(training) != {*keys, "trace", "elapsed_seconds"}:
        raise RuntimeError("Stage2 training fields changed")
    return {key: training.get(key) for key in keys}


def validate_pair(
        direct: Mapping[str, object], kd: Mapping[str, object],
        *, expected_base_sha256: str, expected_clean_manifest_sha256: str,
        ) -> dict[str, object]:
    """Prove the Stage2 arms share one recipe and differ at Real u0 only."""
    if (direct.get("arm") != stage_train.STAGE2_ARMS[0]
            or kd.get("arm") != stage_train.STAGE2_ARMS[1]):
        raise RuntimeError("Stage2 pair arm identities changed")
    if direct.get("adapter_config") != kd.get("adapter_config"):
        raise RuntimeError("Stage2 pair adapter configs differ")
    for name in ("mean_input", "mean_target", "std_input", "std_target"):
        if not torch.equal(direct["normalizer"][name], kd["normalizer"][name]):
            raise RuntimeError(f"Stage2 pair normalizer differs: {name}")
    if direct.get("normalizer_sha256") != kd.get("normalizer_sha256"):
        raise RuntimeError("Stage2 pair normalizer SHA differs")

    direct_source, kd_source = direct.get("source"), kd.get("source")
    if (not isinstance(direct_source, Mapping)
            or not isinstance(kd_source, Mapping)
            or stage_train._canonical_sha(direct_source)
            != stage_train._canonical_sha(kd_source)):
        raise RuntimeError("Stage2 pair source lineage differs")
    base_source = direct_source.get("base")
    manifest_source = direct_source.get("clean_manifest")
    stage1_source = direct_source.get("stage1")
    if (not isinstance(base_source, Mapping)
            or base_source.get("sha256") != expected_base_sha256
            or not isinstance(manifest_source, Mapping)
            or manifest_source.get("file_sha256")
            != expected_clean_manifest_sha256
            or not isinstance(stage1_source, Mapping)
            or stage1_source.get("endpoint")
            != f"raw_u{stage_train.STAGE1_UPDATES}"
            or int(stage1_source.get("update", -1))
            != stage_train.STAGE1_UPDATES
            or int(direct_source.get("optimizer_windows", -1))
            != stage_train.REAL_HISTORIES
            or int(direct_source.get("optimizer_stride", -1))
            != real4k.OPTIMIZER_STRIDE):
        raise RuntimeError("Stage2 pair source contract changed")

    schedule = real4k.make_sample_schedule(
        stage_train.REAL_HISTORIES, stage_train.STAGE2_UPDATES,
        stage_train.BATCH_SIZE)
    expected_schedule = real4k.schedule_sha(schedule)
    if expected_schedule != STAGE2_SCHEDULE_SHA256:
        raise RuntimeError("fixed Stage2 schedule implementation changed")
    expected_training = {
        "updates": stage_train.STAGE2_UPDATES,
        "batch_size": stage_train.BATCH_SIZE,
        "seed": stage_train.SEED,
        "sample_schedule_sha256": expected_schedule,
        "augmentation": dict(real4k.AUGMENTATION),
        "objective": "physical_uv_official_Rel-L2_mapping_only",
        "optimizer": "AdamW",
        "lr": stage_train.LR,
        "weight_decay": stage_train.WEIGHT_DECAY,
        "grad_clip": stage_train.GRAD_CLIP,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": stage_train.SCHEDULER_T_MAX,
        "ema_decay": stage_train.EMA_DECAY,
        "ema_initialized_from_real_u0": True,
    }
    if (_training_contract(direct) != expected_training
            or _training_contract(kd) != expected_training):
        raise RuntimeError("Stage2 matched training contract changed")
    if direct_source.get("sample_schedule_sha256") != expected_schedule:
        raise RuntimeError("Stage2 source/training schedule seals disagree")

    direct_mechanics, kd_mechanics = (
        direct.get("mechanics"), kd.get("mechanics"))
    if not isinstance(direct_mechanics, Mapping) or not isinstance(
            kd_mechanics, Mapping):
        raise RuntimeError("Stage2 pair mechanics evidence is missing")
    mechanics_fields = {
        "initial_adapter_state_sha256", "ema_u0_state_sha256",
        "rng_u0_sha256", "augmentation_probes",
        "prediction_bitwise_B210_at_real_u0",
        "direction_head_gradient_u1", "history_encoder_gradient_u2",
    }
    if (set(direct_mechanics) != mechanics_fields
            or set(kd_mechanics) != mechanics_fields):
        raise RuntimeError("Stage2 mechanics fields changed")
    for mechanics in (direct_mechanics, kd_mechanics):
        if (mechanics.get("ema_u0_state_sha256")
                != mechanics.get("initial_adapter_state_sha256")
                or mechanics.get("direction_head_gradient_u1") is not True
                or mechanics.get("history_encoder_gradient_u2") is not True):
            raise RuntimeError("Stage2 u0/gradient mechanics changed")
    direct_initial = direct_mechanics.get("initial_adapter_state_sha256")
    kd_initial = kd_mechanics.get("initial_adapter_state_sha256")
    zero_adapter = direct_phase.build_adapter(
        "causal_temporal", torch.device("cpu"))
    zero_sha = ema_lib.tensor_state_sha256(zero_adapter.state_dict())
    if (direct_initial != zero_sha
            or kd_initial != stage1_source.get("raw_adapter_state_sha256")
            or direct_initial == kd_initial
            or direct_mechanics.get("prediction_bitwise_B210_at_real_u0")
            is not True):
        raise RuntimeError("Stage2 pair does not differ only by fixed u0 state")
    if (direct_mechanics.get("rng_u0_sha256")
            != kd_mechanics.get("rng_u0_sha256")
            or direct_mechanics.get("augmentation_probes")
            != kd_mechanics.get("augmentation_probes")):
        raise RuntimeError("Stage2 pair RNG/augmentation stream differs")
    probes = direct_mechanics.get("augmentation_probes")
    if not isinstance(probes, Mapping) or set(probes) != {"1", "2", "4000"}:
        raise RuntimeError("Stage2 augmentation probes are incomplete")
    for update in stage_train.AUGMENTATION_PROBE_UPDATES:
        probe = probes[str(update)]
        if (not isinstance(probe, Mapping)
                or probe.get("indices_sha256") != stage_train._canonical_sha([
                    int(index) for index in schedule[update - 1]])):
            raise RuntimeError("Stage2 augmentation probe indices changed")
    return {
        "only_initial_adapter_state_differs": True,
        "direct_initial_state_sha256": direct_initial,
        "kd_initial_state_sha256": kd_initial,
        "sample_schedule_sha256": expected_schedule,
        "augmentation_probes_bitwise_equal": True,
        "optimizer_loss_scheduler_ema_contract_equal": True,
    }


def validate_base_and_fit_manifest(
        direct: Mapping[str, object], kd: Mapping[str, object],
        *, base_report: Mapping[str, object], normalizer,
        clean_manifest_path: Path,
        ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    for endpoint in (direct, kd):
        source = endpoint["source"]
        if (source["base"].get("model_state_sha256")
                != base_report.get("model_state_sha256")
                or source["base"].get("normalizer", {}).get("sha256")
                != stage_train._normalizer_sha(normalizer)
                or not stage_train._normalizers_equal(
                    endpoint["normalizer"], normalizer)):
            raise RuntimeError("Stage2 endpoint differs from loaded B210")
    fit_rows, manifest_report = real4k.load_clean_manifest(clean_manifest_path)
    file_sha = direct_phase.sha256(clean_manifest_path)
    if any(endpoint["source"]["clean_manifest"].get("file_sha256") != file_sha
           for endpoint in (direct, kd)):
        raise RuntimeError("Stage2 endpoint differs from loaded clean81 manifest")
    embedded = direct["source"]["stage1"]
    _state, stage1_report = stage_train.load_stage1_endpoint(
        Path(str(embedded.get("path"))), embedded.get("file_sha256"),
        normalizer, base_report, file_sha)
    if (stage1_report["raw_adapter_state_sha256"]
            != embedded.get("raw_adapter_state_sha256")
            or stage1_report["raw_adapter_state_sha256"]
            != kd["mechanics"]["initial_adapter_state_sha256"]):
        raise RuntimeError("KD u0 does not bind the reloaded Stage1 endpoint")
    return (
        fit_rows,
        {**manifest_report, "path": str(clean_manifest_path.resolve()),
         "file_sha256": file_sha},
        stage1_report,
    )


def evaluation_normalizer(
        endpoint: Mapping[str, object], base_normalizer,
        device: torch.device):
    """Build the existing evaluator's postprocessing-capable normalizer."""
    normalizer = phase_eval._normalizer(endpoint, device)
    if not stage_train._normalizers_equal(normalizer, base_normalizer):
        raise RuntimeError("evaluator and B210 Real normalizers differ")
    return normalizer


class PairEndpointAccessAudit(real4k.PrefixAccessAudit):
    """Keep both already-validated Stage2 files immutable during H5 reads."""

    def __init__(self, seals: Sequence[Mapping[str, object]]) -> None:
        super().__init__()
        if len(seals) != 2:
            raise ValueError("clean716 replay requires exactly two endpoint seals")
        self.pair_seals = [dict(seal) for seal in seals]
        self.record_endpoint_seal(self.pair_seals[0])

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        super().before_open(row, phase)
        if phase == "clean716":
            for seal in self.pair_seals:
                path = Path(str(seal["path"]))
                if (not path.is_file()
                        or direct_phase.sha256(path) != seal["file_sha256"]):
                    raise RuntimeError(
                        "sealed Stage2 pair changed before clean716 evaluation")


def _public_models(
        models: Mapping[str, Mapping[str, float]],
        ) -> dict[str, dict[str, float]]:
    return {INTERNAL_TO_PUBLIC[name]: dict(values)
            for name, values in models.items()}


def _comparisons(
        models: Mapping[str, Mapping[str, float]],
        ) -> dict[str, object]:
    return {
        "kd_minus_direct_raw": phase_eval._delta(
            models["kd_raw"], models["direct_raw"]),
        "kd_minus_direct_ema": phase_eval._delta(
            models["kd_ema"], models["direct_ema"]),
        "each_minus_b210": {
            name: phase_eval._delta(models[name], models["b210"])
            for name in PUBLIC_MODEL_NAMES if name != "b210"
        },
    }


def evaluate(
        base, adapters: Mapping[str, CausalPhaseDirectionAdapter],
        dataset, aoas: Sequence[int], trajectory_labels: Sequence[str],
        normalizer, device: torch.device,
        ) -> dict[str, object]:
    """Reuse the existing phase evaluator, then relabel its fixed pair slots."""
    if set(adapters) != {"direct_raw", "direct_ema", "kd_raw", "kd_ema"}:
        raise ValueError("evaluation requires all four fixed Stage2 states")
    internal = {
        "spatial_control_raw": adapters["direct_raw"],
        "spatial_control_ema": adapters["direct_ema"],
        "causal_raw": adapters["kd_raw"],
        "causal_ema": adapters["kd_ema"],
    }
    raw = phase_eval.evaluate(
        base, internal, dataset, aoas, trajectory_labels, normalizer, device,
        model_names=("b210", *internal))
    pooled = _public_models(raw["pooled"])

    def relabel_groups(groups: Mapping[str, object]) -> dict[str, object]:
        result = {}
        for label, group in groups.items():
            models = _public_models(group["models"])
            result[label] = {"models": models, "comparisons": _comparisons(models)}
        return result

    return {
        "windows": raw["windows"],
        "pooled": pooled,
        "comparisons": _comparisons(pooled),
        "by_aoa": relabel_groups(raw["by_aoa"]),
        "by_trajectory": relabel_groups(raw["by_trajectory"]),
        "absolute_endpoint_max_delta_from_b210": {
            INTERNAL_TO_PUBLIC[name]: value
            for name, value in raw[
                "absolute_endpoint_max_delta_from_b210"].items()},
        "prediction_sha256": {
            INTERNAL_TO_PUBLIC[name]: value
            for name, value in raw["prediction_sha256"].items()},
        "target_sha256": raw["target_sha256"],
        "metric_implementation": phase_eval.SCHEMA,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.acknowledge_target_access != ACKNOWLEDGEMENT:
        raise PermissionError("clean716 replay requires the exact acknowledgement")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    if args.clean_manifest.resolve() == args.clean716_manifest.resolve():
        raise ValueError("clean81 and clean716 manifests must be distinct files")
    expected_base_sha = args.base_checkpoint_sha256.strip().lower()
    expected_manifest_sha = direct_phase.sha256(args.clean_manifest)

    # Hard barrier: nothing named clean716 is read before both endpoint
    # containers and the complete matched Stage2 contract pass validation.
    direct, direct_report = validate_endpoint(
        args.direct_endpoint, args.direct_endpoint_sha256,
        stage_train.STAGE2_ARMS[0])
    kd, kd_report = validate_endpoint(
        args.kd_endpoint, args.kd_endpoint_sha256,
        stage_train.STAGE2_ARMS[1])
    pair_report = validate_pair(
        direct, kd, expected_base_sha256=expected_base_sha,
        expected_clean_manifest_sha256=expected_manifest_sha)

    device = torch.device("cpu")
    base, base_normalizer, base_report = stage_train.load_b210_u3200(
        args.base_checkpoint, expected_base_sha, device)
    normalizer = evaluation_normalizer(direct, base_normalizer, device)
    fit_rows, fit_manifest_report, stage1_report = (
        validate_base_and_fit_manifest(
            direct, kd, base_report=base_report, normalizer=base_normalizer,
            clean_manifest_path=args.clean_manifest)
    )
    adapters = {
        "direct_raw": phase_eval._load_adapter(direct, "raw", device),
        "direct_ema": phase_eval._load_adapter(direct, "ema", device),
        "kd_raw": phase_eval._load_adapter(kd, "raw", device),
        "kd_ema": phase_eval._load_adapter(kd, "ema", device),
    }
    barrier = {
        "validated_before_clean716_manifest_or_h5_open": True,
        "direct": direct_report,
        "kd": kd_report,
        "matched_pair": pair_report,
        "base_and_clean81_bound": True,
        "stage1_lineage": stage1_report,
        "four_fixed_states": list(adapters),
    }

    for report in (direct_report, kd_report):
        fresh = stage_train.verify_companion_seal(
            Path(report["path"]), report["file_sha256"])
        if fresh["seal_file_sha256"] != report[
                "companion_seal"]["seal_file_sha256"]:
            raise RuntimeError("Stage2 companion changed before clean716 access")
    if (args.clean716_manifest_sha256.strip().lower()
            != CLEAN716_MANIFEST_SHA256):
        raise RuntimeError("clean716 manifest is not the fixed 716 panel")
    clean716_file_sha = stage_train._require_file_sha(
        args.clean716_manifest, args.clean716_manifest_sha256,
        "clean716 manifest")
    tail_rows = real4k.load_clean716_rows(
        args.clean716_manifest, fit_rows)
    clean716_report = {
        "path": str(args.clean716_manifest.resolve()),
        "file_sha256": clean716_file_sha,
        "semantic_sha256": real4k.canonical_sha(json.loads(
            args.clean716_manifest.read_text(encoding="utf-8"))),
        "windows": sum(len(row["train_starts"]) for row in tail_rows),
    }
    access = PairEndpointAccessAudit((direct_report, kd_report))
    trajectories, windows = real4k.load_real_rows(
        args.real_dir, tail_rows, phase="clean716", audit=access)
    dataset = real4k.RealWindowDataset(trajectories, windows)
    aoas = [int(tail_rows[window.trajectory]["aoa"]) for window in windows]
    labels = [str(tail_rows[window.trajectory]["file"]) for window in windows]
    evaluation = evaluate(
        base, adapters, dataset, aoas, labels, normalizer, device)
    result = {
        "schema": SCHEMA,
        "evidence_class": "post_hoc_fixed_clean716_pair_diagnostic",
        "device": "cpu",
        "endpoint_barrier": barrier,
        "base": base_report,
        "fit_manifest": fit_manifest_report,
        "clean716_manifest": clean716_report,
        "target_access_audit": access.events,
        "evaluation": evaluation,
        "reported_states": list(PUBLIC_MODEL_NAMES),
        "checkpoint_selection": False,
        "score_gate": None,
        "submission_authorized": False,
    }
    stage_train._atomic_json(args.output_json, result)
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_causal_phase_kd_clean716_complete",
        "schema": result["schema"],
        "output_json": str(args.output_json.resolve()),
        "pooled": result["evaluation"]["pooled"],
        "comparisons": result["evaluation"]["comparisons"],
    }, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
