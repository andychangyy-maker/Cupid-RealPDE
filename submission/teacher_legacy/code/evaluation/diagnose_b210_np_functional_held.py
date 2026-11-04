#!/usr/bin/env python3
"""Post-hoc held-fold attribution for the sealed B210 Real-ft12 run.

This tool is diagnostic only.  It cannot select an endpoint, train a model,
write a checkpoint, or authorize a submission.  It replays the already-opened
held partition to isolate the primary EMA energy controller and to measure the
existing Sim hgrad objective on Sim10k/u4EMA/u12EMA.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_real_ft12_loao as real_ft  # noqa: E402
import train_b210_np_functional_sim as b210_sim  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_real_hgrad as real_hgrad  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402


SCHEMA = "b210_np_functional_held_attribution_v1"
ACKNOWLEDGEMENT = "B210_NP_HELD_ATTRIBUTION_DIAGNOSTIC_ONLY"
HGRAD_METHOD = "sim_train_physical_uv_forward_difference_rms"
NORMALIZER_KEYS = ("mean_input", "mean_target", "std_input", "std_target")
METRIC_REPLAY_ATOL = 1e-6


def load_json(path: Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON mapping")
    return value


def validate_hgrad_scale_contract(contract: Mapping[str, object]) -> dict:
    value = dict(contract)
    digest = value.pop("canonical_sha256", None)
    if digest != dr1.canonical_sha256(value):
        raise RuntimeError("Sim hgrad scale canonical SHA-256 changed")
    if (value.get("method") != HGRAD_METHOD
            or int(value.get("train_windows", 0)) <= 0
            or value.get("real_windows_used") != 0
            or value.get("fixed725_windows_used") != 0):
        raise ValueError("Sim hgrad scale provenance changed")
    for direction in ("dx", "dy"):
        scales = value.get("scales", {}).get(direction)
        if (not isinstance(scales, list) or len(scales) != 2
                or any(not math.isfinite(float(item)) or float(item) <= 0.0
                       for item in scales)):
            raise ValueError(f"invalid Sim hgrad {direction} scales")
    return dict(contract)


def normalizer_from_result(row: Mapping[str, object],
                           device: torch.device) -> smu1.FoldNormalizer:
    report = row.get("normalizer")
    if not isinstance(report, Mapping):
        raise ValueError("fold result has no normalizer report")
    values = report.get("values")
    if not isinstance(values, Mapping) or set(values) != set(NORMALIZER_KEYS):
        raise ValueError("fold result normalizer values changed")
    if report.get("sha256") != dr1.canonical_sha256(dict(values)):
        raise RuntimeError("fold result normalizer SHA-256 changed")
    tensors = tuple(torch.as_tensor(
        values[key], dtype=torch.float32) for key in NORMALIZER_KEYS)
    if any(value.shape != (3,) or not bool(torch.isfinite(value).all())
           for value in tensors):
        raise ValueError("fold result normalizer tensors changed")
    return smu1.FoldNormalizer(tensors, device)


def validate_existing_diagnostics(row: Mapping[str, object]) -> dict:
    if (row.get("schema") != real_ft.SCHEMA
            or row.get("experiment") != real_ft.EXPERIMENT
            or row.get("contract_sha256") != real_ft.CONTRACT_SHA256):
        raise ValueError("fold result is not the frozen B210 Real-ft12 run")
    held_aoa = int(row.get("held_aoa", -1))
    if held_aoa not in real_ft.FOLD_ORDER:
        raise ValueError("fold result held AoA changed")
    if (row.get("endpoint_order") != list(real_ft.ENDPOINT_ORDER)
            or row.get("primary_endpoint") != real_ft.PRIMARY_ENDPOINT
            or row.get("diagnostics_cannot_select_endpoint") is not True):
        raise RuntimeError("fold result endpoint policy changed")
    endpoints = row.get("endpoints")
    if not isinstance(endpoints, Mapping) or set(endpoints) != set(
            real_ft.ENDPOINT_ORDER):
        raise ValueError("fold result endpoint set changed")
    audits = []
    for endpoint in real_ft.ENDPOINT_ORDER:
        report = endpoints[endpoint]
        if not isinstance(report, Mapping):
            raise TypeError(f"{endpoint} report is not a mapping")
        state = report.get("metric_state")
        metrics = report.get("metrics")
        seal = report.get("seal")
        if (not isinstance(state, dict) or not isinstance(metrics, dict)
                or not isinstance(seal, Mapping)):
            raise ValueError(f"{endpoint} report is incomplete")
        if smu1.finalize_metric_state(state) != metrics:
            raise RuntimeError(f"{endpoint} metric state changed")
        if (seal.get("endpoint") != endpoint
                or seal.get("primary") is not (
                    endpoint == real_ft.PRIMARY_ENDPOINT)):
            raise RuntimeError(f"{endpoint} seal binding changed")
        dr1.normalized_sha256(str(seal.get("file_sha256", "")),
                             f"{endpoint} file SHA")
        dr1.normalized_sha256(str(seal.get("model_state_sha256", "")),
                             f"{endpoint} state SHA")
        target_audit = report.get("held_target_audit")
        if (not isinstance(target_audit, Mapping)
                or set(target_audit) != {
                    "ordered_raw_target_sha256",
                    "ordered_scored_mask_sha256", "samples",
                    "scored_elements"}):
            raise ValueError(f"{endpoint} target/mask audit changed")
        dr1.normalized_sha256(
            str(target_audit["ordered_raw_target_sha256"]),
            f"{endpoint} target SHA")
        dr1.normalized_sha256(
            str(target_audit["ordered_scored_mask_sha256"]),
            f"{endpoint} mask SHA")
        audits.append(target_audit)
    if any(audit != audits[0] for audit in audits[1:]):
        raise RuntimeError("sealed endpoint target/mask audits differ")
    primary = endpoints[real_ft.PRIMARY_ENDPOINT]["metrics"]
    ft12 = real_ft.metric_comparison(
        primary, endpoints["ema_u4000"]["metrics"])
    raw_ema = real_ft.metric_comparison(
        primary, endpoints["raw_u12000"]["metrics"])
    if (ft12 != row.get("ft12_increment_diagnostic")
            or raw_ema != row.get("raw_vs_ema_u12000_diagnostic")):
        raise RuntimeError("stored B210 endpoint diagnostics changed")
    primary_report = row.get("primary")
    if (not isinstance(primary_report, Mapping)
            or primary_report.get("endpoint") != real_ft.PRIMARY_ENDPOINT
            or primary_report.get("metrics") != primary
            or primary_report.get("metric_state")
            != endpoints[real_ft.PRIMARY_ENDPOINT]["metric_state"]):
        raise RuntimeError("fold result primary binding changed")
    return {
        "ema_u4000_to_ema_u12000": ft12,
        "raw_u12000_to_ema_u12000": raw_ema,
        "recomputed_exactly_from_sealed_metric_states": True,
        "cannot_select_endpoint": True,
    }


def validate_sim_evidence(summary: Mapping[str, object],
                          scale_contract: Mapping[str, object],
                          start_sha256: str) -> None:
    if (summary.get("method") != b210_sim.METHOD
            or summary.get("status") != b210_sim.PASS_STATUS
            or int(summary.get("completed_updates", -1)) != 10_000
            or summary.get("checkpoint_sha256") != start_sha256):
        raise ValueError("Sim summary is not the passed B210 Sim10k source")
    if summary.get("hgrad_scale_contract") != dict(scale_contract):
        raise RuntimeError("Sim summary and hgrad scale artifact differ")


def validate_manifest_and_load_held(
        row: Mapping[str, object], manifest_path: Path, real_dir: Path,
        ) -> tuple[object, list[int], dict]:
    expected_file_sha = str(row.get("manifest_file_sha256", ""))
    dr1.normalized_sha256(expected_file_sha, "fold manifest file SHA")
    if (expected_file_sha != real_ft.FORMAL_MANIFEST_FILE_SHA256
            or dr1.sha256(manifest_path) != expected_file_sha):
        raise RuntimeError("clean2544 manifest file SHA-256 changed")
    rows, manifest_report = dr1.load_manifest(
        manifest_path, real_ft.FORMAL_MANIFEST_SEMANTIC_SHA256)
    if manifest_report != row.get("manifest"):
        raise RuntimeError("fold result manifest report changed")
    held_aoa = int(row["held_aoa"])
    held_rows = [item for item in rows if int(item["aoa"]) == held_aoa]
    access = real_ft.ThreeEndpointAccessAudit(held_aoa)
    for endpoint in real_ft.ENDPOINT_ORDER:
        access.seal(endpoint, row["endpoints"][endpoint]["seal"])
    trajectories, metadata, windows = dr1.load_prefix_rows(
        real_dir, held_rows, "held", access)
    expected_samples = dr1.EXPECTED_AOA_COUNTS[held_aoa]
    if len(windows) != expected_samples or int(row["samples"]) != expected_samples:
        raise RuntimeError("held fold sample count changed")
    indices = list(range(len(windows)))
    held_key = smu1.ordered_held_key_digest(indices, windows, metadata)
    if held_key != row.get("held_ordered_key_sha256"):
        raise RuntimeError("held ordered-key SHA-256 changed")
    held_files = [item["file"] for item in held_rows]
    access_report = access.report(held_files)
    if (not access_report["allowlist_exact"]
            or access_report["tail_arrays_read"] != 0
            or access_report["excluded_opened"]):
        raise RuntimeError("post-hoc held prefix access audit failed")
    dataset = smu1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    return dataset, indices, {
        "held_aoa": held_aoa,
        "samples": len(windows),
        "held_ordered_key_sha256": held_key,
        "opened_file_allowlist": access_report["opened_file_allowlist"],
        "tail_arrays_read": 0,
        "excluded_opened": False,
    }


def _target_audit_update(target: torch.Tensor, target_hasher,
                         mask_hasher) -> tuple[int, int]:
    target_cpu = target.contiguous()
    scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
    target_hasher.update(str(tuple(target_cpu.shape)).encode("ascii"))
    target_hasher.update(target_cpu.numpy().tobytes(order="C"))
    mask_hasher.update(str(tuple(scored_cpu.shape)).encode("ascii"))
    mask_hasher.update(scored_cpu.numpy().tobytes(order="C"))
    return len(target_cpu), int(scored_cpu.sum())


@torch.inference_mode()
def evaluate_physical_hgrad(
        model, dataset, indices: Sequence[int], normalizer,
        scale_contract: Mapping[str, object], device: torch.device,
        deadline: float,
        ) -> tuple[float, dict]:
    """Replay the existing Sim hgrad loss; no new hgrad metric is defined."""
    loader = DataLoader(
        dataset,
        batch_sampler=[
            list(indices[start:start + smu1.EVAL_BATCH_SIZE])
            for start in range(0, len(indices), smu1.EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=device.type == "cuda")
    model.eval()
    total = 0.0
    samples = 0
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    audited_scored = 0
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("B210 held attribution wall budget expired")
        batch_samples, batch_scored = _target_audit_update(
            target, target_hasher, mask_hasher)
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        prediction_norm, _lower, _upper = model.forward_distribution(
            history_norm)
        loss = real_hgrad.hgrad_loss(
            prediction_norm.float(), target_norm.float(), normalizer,
            dict(scale_contract))
        if not math.isfinite(float(loss)):
            raise FloatingPointError("non-finite held physical hgrad error")
        total += float(loss) * batch_samples
        samples += batch_samples
        audited_scored += batch_scored
    if samples != len(indices):
        raise RuntimeError("hgrad evaluator did not consume the held fold")
    return total / samples, {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": samples,
        "scored_elements": audited_scored,
    }


def _controller_projection(model) -> torch.nn.Module:
    controller = getattr(model, "energy_controller", None)
    if controller is None or not isinstance(controller.body[-1], torch.nn.Conv2d):
        raise RuntimeError("B210 primary has no expected energy-controller projection")
    projection = controller.body[-1]
    if projection.out_channels != 1:
        raise RuntimeError("B210 energy-controller projection shape changed")
    return projection


@contextmanager
def zero_controller_projection(model) -> Iterator[dict]:
    """Temporarily zero the controller output projection and restore exactly."""
    projection = _controller_projection(model)
    original = OrderedDict(
        (name, value.detach().clone())
        for name, value in projection.state_dict().items())
    model_before = dr1.model_state_sha256(model.state_dict())
    report = {"model_state_sha256_before": model_before}
    try:
        with torch.no_grad():
            projection.weight.zero_()
            if projection.bias is not None:
                projection.bias.zero_()
        if any(bool(value.ne(0).any()) for value in projection.state_dict().values()):
            raise RuntimeError("controller output projection did not become zero")
        report["zero_projection_active"] = True
        yield report
    finally:
        projection.load_state_dict(original, strict=True)
        restored = dr1.model_state_sha256(model.state_dict())
        report["model_state_sha256_after_restore"] = restored
        report["bitwise_restored"] = restored == model_before
        if restored != model_before:
            raise RuntimeError("controller ablation did not restore model exactly")


def assert_target_audit(actual: Mapping[str, object],
                        expected: Mapping[str, object], label: str) -> None:
    if dict(actual) != dict(expected):
        raise RuntimeError(f"{label} held target/mask audit changed")


def assert_metric_replay(actual: Mapping[str, float],
                         expected: Mapping[str, float]) -> dict:
    delta = {
        key: float(actual[key]) - float(expected[key]) for key in expected}
    maximum = max(abs(value) for value in delta.values())
    if maximum > METRIC_REPLAY_ATOL:
        raise RuntimeError(
            f"primary EMA metric replay drift {maximum} > {METRIC_REPLAY_ATOL}")
    return {"delta": delta, "max_abs_delta": maximum,
            "tolerance": METRIC_REPLAY_ATOL, "pass": True}


def relative_hgrad_advantage(value: float, reference: float) -> float:
    """Use the B210 Sim gate's existing lower-is-better comparison."""
    return float((reference - value) / max(abs(reference), 1e-12))


def artifact_paths(row: Mapping[str, object], start_sim: Path,
                   fold_result: Path, manifest: Path, sim_summary: Path,
                   scale_contract: Path) -> dict[str, Path]:
    result = {
        "fold_result": fold_result,
        "clean_manifest": manifest,
        "sim_summary": sim_summary,
        "hgrad_scale_contract": scale_contract,
        "sim10k": start_sim,
    }
    for endpoint in real_ft.ENDPOINT_ORDER:
        result[endpoint] = Path(str(
            row["endpoints"][endpoint]["seal"]["path"]))
    if len({path.resolve() for path in result.values()}) != len(result):
        raise ValueError("B210 attribution inputs unexpectedly alias")
    return result


def snapshot_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {label: dr1.sha256(path) for label, path in paths.items()}


def validate_artifact_hashes(
        row: Mapping[str, object], hashes: Mapping[str, str],
        fold_sha: str, summary_sha: str, scale_sha: str) -> None:
    expected = {
        "fold_result": fold_sha,
        "clean_manifest": row["manifest_file_sha256"],
        "sim_summary": summary_sha,
        "hgrad_scale_contract": scale_sha,
        "sim10k": row["start"]["checkpoint_sha256"],
        **{
            endpoint: row["endpoints"][endpoint]["seal"]["file_sha256"]
            for endpoint in real_ft.ENDPOINT_ORDER
        },
    }
    if dict(hashes) != expected:
        changed = [key for key in expected if hashes.get(key) != expected[key]]
        raise RuntimeError(f"B210 attribution artifact SHA mismatch: {changed}")


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite attribution {args.output}")
    fold_sha = dr1.normalized_sha256(
        args.fold_result_sha256, "fold-result SHA")
    summary_sha = dr1.normalized_sha256(
        args.sim_summary_sha256, "sim-summary SHA")
    scale_sha = dr1.normalized_sha256(
        args.hgrad_scale_contract_sha256, "hgrad-scale-contract SHA")
    row = load_json(args.fold_result, "fold result")
    existing = validate_existing_diagnostics(row)
    sim_summary = load_json(args.sim_summary, "Sim summary")
    scale_contract = validate_hgrad_scale_contract(load_json(
        args.hgrad_scale_contract, "hgrad scale contract"))
    validate_sim_evidence(
        sim_summary, scale_contract, row["start"]["checkpoint_sha256"])
    paths = artifact_paths(
        row, args.start_sim, args.fold_result, args.clean_manifest,
        args.sim_summary, args.hgrad_scale_contract)
    hashes_before = snapshot_hashes(paths)
    validate_artifact_hashes(
        row, hashes_before, fold_sha, summary_sha, scale_sha)

    device = torch.device(args.device)
    deadline = time.perf_counter() + args.wall_budget_seconds
    normalizer = normalizer_from_result(row, device)
    dataset, indices, held_audit = validate_manifest_and_load_held(
        row, args.clean_manifest, args.real_dir)
    target_audit = row["endpoints"][real_ft.PRIMARY_ENDPOINT][
        "held_target_audit"]

    # Strictly deserialize the third sealed endpoint even though it is used
    # only by the fold-result delta above.
    raw_model = real_ft.load_endpoint_model(
        row["endpoints"]["raw_u12000"]["seal"], normalizer, device)
    del raw_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sim_model, sim_report = real_ft.validate_start_checkpoint(
        args.start_sim, row["start"]["checkpoint_sha256"], normalizer,
        device)
    sim_hgrad, sim_target_audit = evaluate_physical_hgrad(
        sim_model, dataset, indices, normalizer, scale_contract, device,
        deadline)
    assert_target_audit(sim_target_audit, target_audit, "Sim10k hgrad")
    del sim_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    u4_model = real_ft.load_endpoint_model(
        row["endpoints"]["ema_u4000"]["seal"], normalizer, device)
    u4_hgrad, u4_target_audit = evaluate_physical_hgrad(
        u4_model, dataset, indices, normalizer, scale_contract, device,
        deadline)
    assert_target_audit(u4_target_audit, target_audit, "u4EMA hgrad")
    del u4_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    primary_model = real_ft.load_endpoint_model(
        row["endpoints"][real_ft.PRIMARY_ENDPOINT]["seal"], normalizer,
        device)
    active_metrics, _active_state, active_target_audit = smu1.evaluate_model(
        primary_model, dataset, indices, normalizer, device, deadline)
    assert_target_audit(active_target_audit, target_audit,
                        "primary controller-active")
    replay = assert_metric_replay(
        active_metrics,
        row["endpoints"][real_ft.PRIMARY_ENDPOINT]["metrics"])
    u12_hgrad, u12_target_audit = evaluate_physical_hgrad(
        primary_model, dataset, indices, normalizer, scale_contract, device,
        deadline)
    assert_target_audit(u12_target_audit, target_audit, "u12EMA hgrad")
    with zero_controller_projection(primary_model) as restore_report:
        zero_metrics, _zero_state, zero_target_audit = smu1.evaluate_model(
            primary_model, dataset, indices, normalizer, device, deadline)
        assert_target_audit(zero_target_audit, target_audit,
                            "primary controller-zero")
    controller_comparison = real_ft.metric_comparison(
        active_metrics, zero_metrics)
    del primary_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hashes_after = snapshot_hashes(paths)
    if hashes_after != hashes_before:
        changed = [key for key in hashes_before
                   if hashes_after[key] != hashes_before[key]]
        raise RuntimeError(f"diagnostic mutated input artifacts: {changed}")
    return {
        "schema": SCHEMA,
        "evidence_class": "posthoc_held_diagnostic_not_endpoint_selection",
        "diagnostic_only": True,
        "primary_endpoint": real_ft.PRIMARY_ENDPOINT,
        "endpoint_selection_authorized": False,
        "checkpoint_written": False,
        "submission_authorized": False,
        "inputs": {
            "sha256_before": hashes_before,
            "sha256_after": hashes_after,
            "all_inputs_unchanged": True,
            "sim_start_validation": sim_report,
        },
        "held_input_audit": {
            **held_audit,
            "target_and_scored_mask": target_audit,
            "all_replays_match_fold_result": True,
        },
        "existing_endpoint_diagnostics": existing,
        "controller_attribution": {
            "comparison": "ema_u12000_active_minus_same_model_projection_zero",
            "active": active_metrics,
            "projection_zero": zero_metrics,
            "active_minus_zero": controller_comparison,
            "active_metric_replay": replay,
            "zeroed_parameters": [
                "energy_controller.body.5.weight",
                "energy_controller.body.5.bias",
            ],
            "checkpoint_never_modified": True,
            "restore_audit": restore_report,
            "diagnostic_cannot_select_endpoint": True,
        },
        "physical_hgrad_retention": {
            "definition": (
                "existing Sim physical-u/v unpadded W/H forward-difference "
                "RMS, normalized by Sim-train target-gradient RMS"),
            "scale_contract": scale_contract,
            "lower_is_better": True,
            "same_held_partition": True,
            "errors": {
                "sim10k": sim_hgrad,
                "ema_u4000": u4_hgrad,
                "ema_u12000": u12_hgrad,
            },
            "comparisons": {
                "ema_u4000_vs_sim10k_relative_advantage":
                    relative_hgrad_advantage(u4_hgrad, sim_hgrad),
                "ema_u12000_vs_sim10k_relative_advantage":
                    relative_hgrad_advantage(u12_hgrad, sim_hgrad),
                "ema_u12000_vs_ema_u4000_relative_advantage":
                    relative_hgrad_advantage(u12_hgrad, u4_hgrad),
                "ema_u4000_error_ratio_to_sim10k": u4_hgrad / sim_hgrad,
                "ema_u12000_error_ratio_to_sim10k": u12_hgrad / sim_hgrad,
            },
            "comparison_formula": (
                "(reference_error - candidate_error) / "
                "max(abs(reference_error), 1e-12)"),
            "no_new_hgrad_definition": True,
            "diagnostic_cannot_select_endpoint": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-result", type=Path, required=True)
    parser.add_argument("--fold-result-sha256", required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--sim-summary", type=Path, required=True)
    parser.add_argument("--sim-summary-sha256", required=True)
    parser.add_argument("--hgrad-scale-contract", type=Path, required=True)
    parser.add_argument("--hgrad-scale-contract-sha256", required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=int, default=3600)
    parser.add_argument("--acknowledge-diagnostic", required=True)
    args = parser.parse_args()
    if args.acknowledge_diagnostic != ACKNOWLEDGEMENT:
        raise ValueError("explicit diagnostic-only acknowledgement is required")
    if args.wall_budget_seconds <= 0:
        raise ValueError("wall budget must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    real_ft.dr1.write_json_atomic(args.output, result)
    print(json.dumps({
        "event": "b210_np_held_attribution_complete",
        "output": str(args.output.resolve()),
        "diagnostic_only": True,
    }), flush=True)


if __name__ == "__main__":
    main()
