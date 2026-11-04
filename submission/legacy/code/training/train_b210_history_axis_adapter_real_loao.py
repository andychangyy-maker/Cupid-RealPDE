#!/usr/bin/env python3
"""Paired Real whole-AoA screen for the B210 history-axis adapter.

Both arms upgrade the same immutable XCB/member Sim24k checkpoint with an
identically initialized, zero-output pre-flatten adapter.  The control mixes
within each frame along spatial x; the candidate mixes causally across the
observed time axis.  Held-condition fields stay unopened until both fixed
EMA-u4000 endpoints have been sealed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
from pathlib import Path
import subprocess
import time
from typing import Iterator, Mapping

import torch

import train_b210_condition_held_real as held
import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig


EXPERIMENT = "b210_history_axis_adapter_real_loao"
SCHEMA = "b210_history_axis_adapter_real_loao_v1"
ENDPOINT_SCHEMA = "b210_history_axis_adapter_real_endpoint_v1"
ACKNOWLEDGEMENT = "B210_HISTORY_AXIS_ADAPTER_REAL_LOAO_V1"
ARM_ORDER = ("spatial_control", "causal_temporal")
ARM_MODE = {
    "spatial_control": "framewise_spatial_control",
    "causal_temporal": "causal_temporal",
}
UPDATES = 4_000
SCHEDULER_T_MAX = 12_000
EXPECTED_START_ITERATION = 4_000
EXPECTED_START_STATUS = "PASS_B210_XCB_MEMBER_STANDARD24K_SIM"
EXPECTED_START_METHOD = "b210_xcb_member_standard_stride20_20k_then_4k_v1"
EXPECTED_BASE_PARAMETERS = 180_553_371
EXPECTED_ADAPTER_PARAMETERS = 4_899
EXPECTED_PARAMETERS = EXPECTED_BASE_PARAMETERS + EXPECTED_ADAPTER_PARAMETERS
EXPECTED_CONTROLLER_PARAMETERS = ft12.EXPECTED_CONTROLLER_PARAMETERS
EXPECTED_START_SHA256 = (
    "06ade419811d722e58235acdcb34e480dd6ae88555dd52037ebcb6bbeba2ce99")
EXPECTED_MANIFEST_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
GATE = {
    "delta_point_min": 0.03,
    "component_floor": -0.03,
}
POINT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--held-aoa", type=int, choices=(10, 20))
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=2 * 3600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def git_report() -> dict[str, object]:
    def run(*values: str) -> str:
        return subprocess.run(
            ["git", *values], cwd=Path(__file__).resolve().parents[2],
            check=True, capture_output=True, text=True).stdout.strip()
    porcelain = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "clean": not bool(porcelain),
        "porcelain": porcelain.splitlines(),
    }


def parse_start_payload(payload: Mapping[str, object]) -> tuple[
        EnsembleUNet2dConfig, Mapping[str, torch.Tensor], dict[str, object]]:
    if (payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != EXPECTED_START_ITERATION
            or not isinstance(payload.get("model_state_dict"), Mapping)
            or not isinstance(payload.get("model_config"), Mapping)):
        raise ValueError("start checkpoint is not the frozen Sim24k endpoint")
    metrics = payload.get("metrics")
    training = payload.get("training_config")
    if (not isinstance(metrics, Mapping)
            or metrics.get("status") != EXPECTED_START_STATUS
            or not isinstance(training, Mapping)
            or training.get("method") != EXPECTED_START_METHOD):
        raise ValueError("start checkpoint status/method changed")
    raw_config = dict(payload["model_config"])
    if raw_config.pop("architecture", None) != "ensemble_unet":
        raise ValueError("start checkpoint architecture changed")
    config = EnsembleUNet2dConfig(**raw_config)
    if (config.history_axis_adapter_mode != "none"
            or not config.cross_conditioned_backbone
            or config.member_adapter_channels != 64
            or not config.physical_energy_controller
            or not config.physical_energy_member_sync):
        raise ValueError("start checkpoint is not the frozen XCB/member model")
    report = {
        "phase": "sim",
        "iteration": EXPECTED_START_ITERATION,
        "status": EXPECTED_START_STATUS,
        "method": EXPECTED_START_METHOD,
        "checkpoint_model_config_sha256": dr1.canonical_sha256(
            payload["model_config"]),
        "checkpoint_model_state_sha256": dr1.model_state_sha256(
            payload["model_state_dict"]),
    }
    return config, payload["model_state_dict"], report


def adapted_config(
        base: EnsembleUNet2dConfig, arm: str) -> EnsembleUNet2dConfig:
    if arm not in ARM_MODE:
        raise ValueError(f"unknown history-axis arm {arm!r}")
    return replace(base, history_axis_adapter_mode=ARM_MODE[arm])


def expected_adapter_keys(model: EnsembleUNet2d) -> set[str]:
    if model.history_axis_adapter is None:
        raise RuntimeError("adapted B210 has no history adapter")
    return {
        f"history_axis_adapter.{name}"
        for name in model.history_axis_adapter.state_dict()
    }


def update0_signature(model: EnsembleUNet2d, dataset, normalizer,
                      device: torch.device) -> str:
    samples = [dataset[index] for index in range(min(2, len(dataset)))]
    history = torch.stack([sample[0] for sample in samples]).to(device)
    target = torch.stack([sample[1] for sample in samples]).to(device)
    history_norm, _target_norm = normalizer.preprocess(history, target)
    model.eval()
    with torch.inference_mode():
        values = model.forward_uq_training(history_norm)[:5]
    names = ("prediction", "members", "quantiles", "lower", "upper")
    state = {
        name: value.detach().float().cpu()
        for name, value in zip(names, values)
    }
    return dr1.model_state_sha256(state)


def adapter_parts(model: EnsembleUNet2d) -> tuple[dict, dict]:
    state = model.history_axis_adapter.state_dict()
    output = {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if name.startswith("output_projection.")
    }
    body = {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if not name.startswith("output_projection.")
    }
    if not output or not body:
        raise RuntimeError("history adapter state partition changed")
    return output, body


def state_changed(before: Mapping[str, torch.Tensor],
                  after: Mapping[str, torch.Tensor]) -> bool:
    return (before.keys() == after.keys()
            and any(not torch.equal(before[key], after[key]) for key in before))


def load_upgraded_start(
        path: Path, arm: str, normalizer, device: torch.device,
        ) -> tuple[EnsembleUNet2d, EnsembleUNet2dConfig, dict[str, object]]:
    actual_sha = dr1.sha256(path)
    if actual_sha != EXPECTED_START_SHA256:
        raise RuntimeError("XCB/member Sim24k checkpoint SHA changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    base, state, report = parse_start_payload(payload)
    small4k.npt1.ensemble_train.seed_everything(small4k.SEED)
    config = adapted_config(base, arm)
    model = EnsembleUNet2d(config).float()
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = expected_adapter_keys(model)
    if (set(incompatible.missing_keys) != expected_missing
            or incompatible.unexpected_keys):
        raise RuntimeError(
            "checkpoint upgrade changed non-adapter keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}")
    total = sum(value.numel() for value in model.parameters())
    adapter_count = sum(
        value.numel() for value in model.history_axis_adapter.parameters())
    controller_count = sum(
        value.numel() for value in model.energy_controller.parameters())
    if (total != EXPECTED_PARAMETERS
            or adapter_count != EXPECTED_ADAPTER_PARAMETERS
            or controller_count != EXPECTED_CONTROLLER_PARAMETERS):
        raise RuntimeError(
            "upgraded B210 parameter contract changed: "
            f"total={total}, adapter={adapter_count}, "
            f"controller={controller_count}")
    small4k.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    report.update({
        "path": str(path.resolve()),
        "checkpoint_sha256": actual_sha,
        "strict_legacy_state_load": True,
        "missing_keys_exactly_new_adapter": sorted(expected_missing),
        "adapter_mode": ARM_MODE[arm],
        "adapter_parameters": adapter_count,
        "parameters": total,
        "controller_parameters": controller_count,
        "upgraded_model_state_sha256": dr1.model_state_sha256(
            model.state_dict()),
    })
    del payload, state
    gc.collect()
    return model.to(device), config, report


@contextmanager
def endpoint_contract(arm: str, held_aoa: int) -> Iterator[None]:
    if arm not in ARM_MODE:
        raise ValueError(arm)
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": f"ema_u4000_{arm}",
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": SCHEDULER_T_MAX,
        "HELD_AOA": held_aoa,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(small4k, name, value)


def train_arm(
        arm: str, args: argparse.Namespace, access: held.PrefixAccessAudit,
        dataset, schedule: Mapping[str, object], normalizer,
        expected_initial_sha: str | None, expected_update0_sha: str | None,
        deadline: float) -> tuple[dict[str, object], str, str]:
    device = torch.device(args.device)
    model, config, start_report = load_upgraded_start(
        args.start_sim, arm, normalizer, device)
    initial_sha = str(start_report["upgraded_model_state_sha256"])
    if expected_initial_sha is not None and initial_sha != expected_initial_sha:
        raise RuntimeError("paired adapter arms do not have identical tensors")
    current_update0_sha = update0_signature(
        model, dataset, normalizer, device)
    if (expected_update0_sha is not None
            and current_update0_sha != expected_update0_sha):
        raise RuntimeError("paired adapter update-zero outputs differ")
    initial_output, initial_body = adapter_parts(model)
    arm_dir = args.output_dir / arm
    arm_dir.mkdir()
    seal_adapter = held.ArmSealAdapter(access, arm)
    # Reset all stochastic training streams after construction/loading so the
    # two modes receive identical dropout and input-augmentation draws.
    small4k.npt1.ensemble_train.seed_everything(small4k.SEED)
    with endpoint_contract(arm, args.held_aoa):
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, seal_adapter,
            input_augmentation=dense4k.AUGMENTATION,
            endpoint_filename=f"ema_u4000_{arm}.pth",
            expected_parameters=EXPECTED_PARAMETERS,
            scheduler_t_max=SCHEDULER_T_MAX)
    if seal_adapter.value != seal:
        raise RuntimeError(f"{arm} endpoint was not bound to access audit")
    final_output, final_body = adapter_parts(model)
    output_updated = state_changed(initial_output, final_output)
    body_updated = state_changed(initial_body, final_body)
    if not output_updated or not body_updated:
        raise RuntimeError(
            "history adapter did not train through output and body")
    adapter_state = {
        name: value.detach().cpu()
        for name, value in model.history_axis_adapter.state_dict().items()
    }
    report = {
        "mode": ARM_MODE[arm],
        "model_config": {"architecture": "ensemble_unet", **config.__dict__},
        "initial_state_sha256": initial_sha,
        "update0_output_sha256": current_update0_sha,
        "adapter_output_updated": output_updated,
        "adapter_body_updated": body_updated,
        "adapter_raw_terminal_state_sha256": dr1.model_state_sha256(
            adapter_state),
        "sample_stream_sha256": schedule["sha256"],
        "training": training,
        "seal": seal,
    }
    del (adapter_state, initial_output, initial_body, final_output, final_body,
         model)
    gc.collect()
    torch.cuda.empty_cache()
    return report, initial_sha, current_update0_sha


def load_endpoint(
        arm: str, seal: Mapping[str, object], normalizer,
        device: torch.device, base_config: EnsembleUNet2dConfig,
        held_aoa: int,
        ) -> EnsembleUNet2d:
    with endpoint_contract(arm, held_aoa):
        return small4k.load_primary_endpoint(
            seal, normalizer, device,
            expected_model_config=adapted_config(base_config, arm),
            expected_parameters=EXPECTED_PARAMETERS,
            expected_controller_parameters=EXPECTED_CONTROLLER_PARAMETERS)


def run(args: argparse.Namespace) -> dict[str, object]:
    if (not args.execute
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise PermissionError("formal run requires protocol acknowledgement")
    required = (args.output_dir, args.held_aoa, args.real_dir,
                args.clean_manifest, args.start_sim)
    if any(value is None for value in required):
        raise ValueError("formal run is missing a required path/AoA")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if dr1.sha256(args.clean_manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("clean81 manifest file SHA changed")
    git = git_report()
    if not git["clean"]:
        raise RuntimeError(f"Git worktree must be clean: {git['porcelain']}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal B210 adapter screen requires CUDA")
    rows, manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    if not fit_rows or not held_rows:
        raise RuntimeError("whole-AoA split is empty")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    access = held.PrefixAccessAudit(args.held_aoa, ARM_ORDER)
    fit_trajectories, fit_metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", access)
    dense_windows = dense4k.dense_windows(
        fit_trajectories, dense4k.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        fit_trajectories, dense_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "fit_normalizer.pt")
    schedule = held.uniform_batch_schedule(dense_windows, UPDATES)

    arms: dict[str, dict[str, object]] = {}
    paired_initial_sha = None
    paired_update0_sha = None
    base_config = None
    for arm in ARM_ORDER:
        report, current_initial_sha, current_update0_sha = train_arm(
            arm, args, access, dataset, schedule, normalizer,
            paired_initial_sha, paired_update0_sha, deadline)
        paired_initial_sha = current_initial_sha
        paired_update0_sha = current_update0_sha
        arms[arm] = report
        raw = dict(report["model_config"])
        raw.pop("architecture")
        current_base = replace(
            EnsembleUNet2dConfig(**raw), history_axis_adapter_mode="none")
        if base_config is None:
            base_config = current_base
        elif current_base != base_config:
            raise RuntimeError("paired arms do not share a base config")

    del dataset, fit_trajectories
    gc.collect()
    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", access)
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    target_audit = None
    for arm in ARM_ORDER:
        model = load_endpoint(
            arm, arms[arm]["seal"], normalizer, device, base_config,
            args.held_aoa)
        metrics, metric_state, current_target_audit = smu1.evaluate_model(
            model, held_dataset, list(range(len(held_windows))),
            normalizer, device, deadline)
        if target_audit is None:
            target_audit = current_target_audit
        elif target_audit != current_target_audit:
            raise RuntimeError("paired held target audit changed")
        arms[arm]["metrics"] = metrics
        arms[arm]["metric_state"] = metric_state
        arms[arm]["target_audit"] = current_target_audit
        del model
        gc.collect()
        torch.cuda.empty_cache()

    comparison = held.metric_comparison(
        arms["causal_temporal"]["metrics"],
        arms["spatial_control"]["metrics"])
    component_pass = all(
        float(comparison["delta"][key]) >= GATE["component_floor"]
        for key in POINT_KEYS)
    passed = bool(
        comparison["point_final_contribution_delta"] >= GATE["delta_point_min"]
        and component_pass)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": args.held_aoa,
        "git": git,
        "fit_trajectories": len(fit_rows),
        "held_trajectories": len(held_rows),
        "fit_sparse_windows": len(sparse_windows),
        "fit_dense_windows": len(dense_windows),
        "held_prefix_windows": len(held_windows),
        "manifest": manifest,
        "normalizer": {
            "kind": "other-four AoA sparse Real prefixes only",
            "statistics_sha256": smu1.statistics_digest(statistics),
            "held_condition_in_normalizer": False,
        },
        "start_checkpoint_sha256": EXPECTED_START_SHA256,
        "parameters": EXPECTED_PARAMETERS,
        "adapter_parameters": EXPECTED_ADAPTER_PARAMETERS,
        "paired_initial_state_sha256": paired_initial_sha,
        "paired_update0_output_sha256": paired_update0_sha,
        "sample_stream_sha256_both_arms": schedule["sha256"],
        "arms": arms,
        "comparison_temporal_minus_spatial": comparison,
        "gate": {**GATE, "component_pass": component_pass, "pass": passed},
        "decision": (
            "GO_HELD20_CONFIRMATION" if passed and args.held_aoa == 10
            else "PASS_B210_HISTORY_AXIS_REAL_LOAO" if passed
            else "STOP_B210_HISTORY_AXIS_ADAPTER"),
        "access_audit": access.report(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_class": (
            "Real-stage whole-AoA held; shared released-Sim24k start contains "
            "all Sim conditions; no Real held-condition fields used before seals"),
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    print(json.dumps({
        "decision": result["decision"], "held_aoa": args.held_aoa,
        "delta": comparison["delta"],
        "delta_point": comparison["point_final_contribution_delta"],
        "delta_native_sps": comparison["delta"]["native_sps_score"],
        "result": str(args.output_dir / "result.json"),
    }, sort_keys=True), flush=True)
    return result


def smoke(args: argparse.Namespace) -> None:
    if args.start_sim is None:
        raise ValueError("--smoke requires --start-sim")
    if dr1.sha256(args.start_sim) != EXPECTED_START_SHA256:
        raise RuntimeError("smoke start checkpoint SHA changed")
    payload = torch.load(
        str(args.start_sim), map_location="cpu", weights_only=False, mmap=True)
    base, state, report = parse_start_payload(payload)
    initial = []
    for arm in ARM_ORDER:
        small4k.npt1.ensemble_train.seed_everything(small4k.SEED)
        model = EnsembleUNet2d(adapted_config(base, arm)).float()
        incompatible = model.load_state_dict(state, strict=False)
        if (set(incompatible.missing_keys) != expected_adapter_keys(model)
                or incompatible.unexpected_keys):
            raise RuntimeError("smoke partial-load contract failed")
        initial.append(dr1.model_state_sha256(model.state_dict()))
        del model
        gc.collect()
    if len(set(initial)) != 1:
        raise RuntimeError("smoke paired initial states differ")
    print(json.dumps({
        "smoke": "PASS", "source": report,
        "parameters": EXPECTED_PARAMETERS,
        "adapter_parameters": EXPECTED_ADAPTER_PARAMETERS,
        "paired_initial_state_sha256": initial[0],
    }, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
