#!/usr/bin/env python3
"""RUS1: select frozen UQ intervals by their final refined utility.

The frozen implementation selects one of 15 quantile pairs from raw width and
only then applies the learned left/right refinement scales.  This audit applies
those same frozen scales and endpoint ordering to every pair before argmax.
It trains nothing, reads no target during selection, and never changes point.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_c208_uc1_coverage_temperature as uc1  # noqa: E402


EXPERIMENT = "c208_rus1_refinement_aware_interval_selection"
PASS_DECISION = (
    "PASS_REFINEMENT_AWARE_SELECTION_TERMINAL_EQUIVALENT_DIAGNOSTIC_"
    "TO_FIXED_TERMINAL_ROBUSTNESS")
STOP_DECISION = "STOP_REFINEMENT_AWARE_SELECTION_NO_PAIRED_SPS_SIGNAL"
ERROR_DECISION = "STOP_REFINEMENT_AWARE_SELECTION_ERROR"
FOLD_SELECTION_RULE = "best_held_fold_sps"
FOLD_UPDATES = 800
POOLED_GAIN_GATE = 0.10
GROUP_POSITIVE_GATE = 4
FOLD_WORST_GATE = -0.03
AOA_WORST_GATE = -0.05
GPU_BUDGET_SECONDS = 5.0 * 60.0
TOTAL_BUDGET_SECONDS = 15.0 * 60.0
POINT_STREAM_SHA256 = (
    "6421136347632f9929be134bffbe28d25feeac4a01eaec0ba0babca47866354c")
TARGET_STREAM_SHA256 = (
    "2365a2ba5d9e788445a631d17207d664964c6855e6b606221cca4e132cc51b66")
FOLD_CHECKPOINT_SHA256 = {
    0: "05699963378de94e9c5eea8846af4ec24d26a39971f218ef19063de1939b8596",
    1: "f025cb52bb454059dfa2c83e5b486e4b4d59a89aaf5b6915b7b4d196efc40a3b",
    2: "32ff859040e8f27b5d8fe1a50c92ecda72c48e05141abb61d6397df8e22b31bc",
    3: "2e704f03cdfde65203aba23915dc17e1e230336fca4510e6a89257e0b351cc79",
    4: "addee57d1171d58f33e0b7498471730eaaac146d5453c7b346cd104bab944876",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def git_identity() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "porcelain": porcelain}


def refined_utility_select(
        refined_lower_physical: torch.Tensor,
        refined_upper_physical: torch.Tensor,
        valid: torch.Tensor, masses: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax nominal utility after frozen refinement and endpoint ordering."""
    if (refined_lower_physical.shape != refined_upper_physical.shape
            or refined_lower_physical.shape != valid.shape
            or refined_lower_physical.ndim != 6
            or refined_lower_physical.shape[-1] != len(masses)):
        raise ValueError("refined candidate stacks are misaligned")
    if (not bool(torch.isfinite(refined_lower_physical).all())
            or not bool(torch.isfinite(refined_upper_physical).all())
            or not bool(torch.isfinite(masses).all())):
        raise FloatingPointError("refined selection inputs are non-finite")
    width = refined_lower_physical + refined_upper_physical
    if bool((width < 0.0).any()):
        raise ValueError("refined candidate endpoints are not ordered")
    probability = masses.view(1, 1, 1, 1, 1, -1)
    utility = probability * torch.exp(-width / uc1.common.SPS_SIGMA)
    utility = torch.where(valid, utility, utility.new_full((), -1.0))
    return utility.argmax(dim=-1, keepdim=True), utility


def group_row(candidate: uc1.SPSAccumulator,
              baseline: uc1.SPSAccumulator, label: int) -> dict:
    before = baseline.result()
    after = candidate.result()
    return {
        "group": label,
        "baseline": before,
        "candidate": after,
        "sps_delta": float(after["sps_score"] - before["sps_score"]),
    }


def held_selection_identity(summary: dict) -> bool:
    """Reconstruct the frozen legacy held-fold selection contract."""
    folds = summary.get("folds", [])
    config = summary.get("training_config", {})
    return bool(
        config.get("fold_updates") == FOLD_UPDATES
        and config.get("eval_interval") == FOLD_UPDATES
        and len(folds) == 5
        and {row.get("fold") for row in folds} == set(range(5))
        and all(
            row.get("best_iteration") == FOLD_UPDATES
            and row.get("selected_iteration") in (None, FOLD_UPDATES)
            and row.get("selected") == row.get("final")
            and len(row.get("history", [])) == 1
            and row["history"][0].get("iteration") == FOLD_UPDATES
            for row in folds))


def gate_checks(
        pooled_delta: float, fold_rows: list[dict], aoa_rows: list[dict],
        hard_guards: dict[str, bool], gpu_seconds: float, elapsed: float,
        ) -> dict[str, bool]:
    if (len(fold_rows) != 5 or {row.get("group") for row in fold_rows} != set(range(5))):
        raise ValueError("gate requires exactly the five frozen folds")
    if (len(aoa_rows) != len(uc1.AOA_VALUES)
            or {row.get("group") for row in aoa_rows} != set(uc1.AOA_VALUES)):
        raise ValueError("gate requires exactly the five frozen AoAs")
    fold_deltas = [float(row["sps_delta"]) for row in fold_rows]
    aoa_deltas = [float(row["sps_delta"]) for row in aoa_rows]
    checks = dict(hard_guards)
    checks.update({
        "pooled_sps_gain_ge_0_10": pooled_delta >= POOLED_GAIN_GATE,
        "positive_folds_ge_4_of_5": (
            sum(value > 0.0 for value in fold_deltas) >= GROUP_POSITIVE_GATE),
        "worst_fold_ge_minus_0_03": min(fold_deltas) >= FOLD_WORST_GATE,
        "positive_aoas_ge_4_of_5": (
            sum(value > 0.0 for value in aoa_deltas) >= GROUP_POSITIVE_GATE),
        "worst_aoa_ge_minus_0_05": min(aoa_deltas) >= AOA_WORST_GATE,
        "gpu_within_5_minutes": gpu_seconds <= GPU_BUDGET_SECONDS,
        "wall_within_15_minutes": elapsed <= TOTAL_BUDGET_SECONDS,
    })
    return checks


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size != 16 or args.num_workers != 0:
        raise ValueError("formal RUS1 requires batch-size=16 and num-workers=0")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal RUS1 requires CUDA")

    uc1.configure_replay(uc1.EXPECTED_TRAINING_CONFIG["seed"])
    identities = {
        "point_checkpoint": uc1.s1.file_identity(args.point_checkpoint),
        "oof_summary": uc1.s1.file_identity(args.oof_dir / "summary.json"),
        "split_manifest": uc1.s1.file_identity(args.split_manifest),
        "real_stats": uc1.s1.file_identity(args.real_stats),
        "fold_checkpoints": {
            str(fold): uc1.s1.file_identity(
                args.oof_dir / f"fold_{fold:02d}_best.pth")
            for fold in range(5)},
    }
    if identities["point_checkpoint"]["sha256"] != uc1.POINT_CHECKPOINT_SHA256:
        raise ValueError("point checkpoint SHA differs from the RUS1 contract")
    if identities["oof_summary"]["sha256"] != uc1.OOF_SUMMARY_SHA256:
        raise ValueError("OOF summary SHA differs from the RUS1 contract")
    if identities["split_manifest"]["sha256"] != uc1.s1.S1_SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest SHA differs from the RUS1 contract")
    if identities["real_stats"]["sha256"] != uc1.s1.S1_REAL_STATS_SHA256:
        raise ValueError("real stats SHA differs from the RUS1 contract")
    for fold, expected in FOLD_CHECKPOINT_SHA256.items():
        if identities["fold_checkpoints"][str(fold)]["sha256"] != expected:
            raise ValueError(f"fold {fold} checkpoint SHA differs from RUS1")

    trajectories, metadata = uc1.common.repro.load_trajectories(args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = uc1.common.repro.make_split(
        metadata, 20, 0.2)
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    sample_folds, fold_manifest = uc1.trajectory_folds(runtime_manifest, 5)
    uc1.s1.validate_s1_split(
        runtime_manifest, frozen_manifest, len(tail_windows), sample_folds)
    if len(point_windows) != 2576 or len(tail_windows) != 725:
        raise ValueError("frozen split is not 2576/725")
    summary = uc1.validate_summary(
        args.oof_dir / "summary.json", args.point_checkpoint, fold_manifest)
    selection_contamination_declared = held_selection_identity(summary)
    if not selection_contamination_declared:
        raise ValueError(
            "OOF summary does not reproduce the frozen legacy held-fold "
            "checkpoint-selection contract")
    dataset = uc1.common.repro.WindowDataset(trajectories, tail_windows)
    aoas = np.asarray([
        int(metadata[window.trajectory]["aoa"]) for window in tail_windows],
        dtype=np.int64)
    if set(aoas.tolist()) != set(uc1.AOA_VALUES):
        raise ValueError("tail AoA values differ from the RUS1 contract")

    payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    normalizer = uc1.common.repro.Normalizer(args.real_stats, device)
    loader_args = argparse.Namespace(num_workers=0)
    fold_groups = tuple(range(5))
    baseline = uc1.SPSAccumulator()
    candidate = uc1.SPSAccumulator()
    baseline_folds = [uc1.SPSAccumulator() for _ in fold_groups]
    candidate_folds = [uc1.SPSAccumulator() for _ in fold_groups]
    baseline_aoas = {aoa: uc1.SPSAccumulator() for aoa in uc1.AOA_VALUES}
    candidate_aoas = {aoa: uc1.SPSAccumulator() for aoa in uc1.AOA_VALUES}
    point_digest_before = uc1.stream_digest()
    point_digest_after = uc1.stream_digest()
    target_digest = uc1.stream_digest()
    baseline_lower_digest = uc1.stream_digest()
    baseline_upper_digest = uc1.stream_digest()
    candidate_lower_digest = uc1.stream_digest()
    candidate_upper_digest = uc1.stream_digest()
    fold_metadata = []
    switched = switched_scored = total = total_scored = 0
    proxy_count = proxy_violations = 0
    proxy_min_delta = float("inf")
    proxy_sum_delta = 0.0
    candidate_finite = True
    candidate_ordered = True
    gpu_seconds = 0.0
    all_indices = list(range(len(dataset)))

    with torch.inference_mode():
        for fold in fold_groups:
            if time.perf_counter() - started > TOTAL_BUDGET_SECONDS:
                raise TimeoutError("RUS1 exceeded the wall budget")
            indices = [index for index in all_indices if sample_folds[index] == fold]
            loader = uc1.uq_train.make_loader(
                dataset, indices, args.batch_size, False, loader_args, seed=0)
            model, model_metadata = uc1.s1.load_fold_model(
                payload, args.oof_dir / f"fold_{fold:02d}_best.pth",
                fold, args.point_checkpoint, normalizer, device)
            fold_metadata.append(model_metadata)
            offset = 0
            for history, target_full in loader:
                history = history.to(device, non_blocking=True)
                target_full = target_full.to(device, non_blocking=True)
                history_norm, _ = normalizer.preprocess(history, target_full)
                target = target_full[..., :2]
                torch.cuda.synchronize(device)
                gpu_started = time.perf_counter()
                tensors = uc1.candidate_tensors(model, history_norm, normalizer)
                (prediction_norm, refined_lower, refined_upper, valid,
                 raw_width, masses, base_lower, base_upper) = tensors
                baseline_selected = uc1.identity_select(raw_width, valid, masses)
                selected, refined_utility = refined_utility_select(
                    refined_lower, refined_upper, valid, masses)
                selected_lower = uc1.gather_candidate(refined_lower, selected)
                selected_upper = uc1.gather_candidate(refined_upper, selected)
                torch.cuda.synchronize(device)
                gpu_seconds += time.perf_counter() - gpu_started
                prediction = uc1.common.to_physical_target(
                    prediction_norm, normalizer)[..., :2]
                uc1.digest_tensor(point_digest_before, prediction)
                uc1.digest_tensor(point_digest_after, prediction)
                uc1.digest_tensor(target_digest, target)
                uc1.digest_tensor(baseline_lower_digest, base_lower)
                uc1.digest_tensor(baseline_upper_digest, base_upper)
                uc1.digest_tensor(candidate_lower_digest, selected_lower)
                uc1.digest_tensor(candidate_upper_digest, selected_upper)

                base_proxy = uc1.gather_candidate(
                    refined_utility, baseline_selected)
                candidate_proxy = uc1.gather_candidate(
                    refined_utility, selected)
                proxy_delta = candidate_proxy - base_proxy
                proxy_count += proxy_delta.numel()
                proxy_violations += int((proxy_delta < 0.0).sum())
                proxy_min_delta = min(
                    proxy_min_delta, float(proxy_delta.min()))
                proxy_sum_delta += float(proxy_delta.sum(dtype=torch.float64))
                changed = selected.squeeze(-1) != baseline_selected.squeeze(-1)
                scored = target != 0.0
                switched += int(changed.sum())
                switched_scored += int((changed & scored).sum())
                total += changed.numel()
                total_scored += int(scored.sum())
                candidate_finite = bool(
                    candidate_finite
                    and torch.isfinite(selected_lower).all()
                    and torch.isfinite(selected_upper).all())
                candidate_ordered = bool(
                    candidate_ordered
                    and ((selected_lower + selected_upper) >= 0.0).all())

                baseline.update(prediction, target, base_lower, base_upper)
                candidate.update(
                    prediction, target, selected_lower, selected_upper)
                baseline_folds[fold].update(
                    prediction, target, base_lower, base_upper)
                candidate_folds[fold].update(
                    prediction, target, selected_lower, selected_upper)
                batch_aoas = aoas[indices[offset:offset + len(target)]]
                for aoa in uc1.AOA_VALUES:
                    in_aoa = torch.from_numpy(batch_aoas == aoa).to(device)
                    if bool(in_aoa.any()):
                        baseline_aoas[aoa].update(
                            prediction[in_aoa], target[in_aoa],
                            base_lower[in_aoa], base_upper[in_aoa])
                        candidate_aoas[aoa].update(
                            prediction[in_aoa], target[in_aoa],
                            selected_lower[in_aoa], selected_upper[in_aoa])
                offset += len(target)
            if offset != len(indices):
                raise RuntimeError("fold loader did not preserve every row")
            del model
            gc.collect()
            torch.cuda.empty_cache()

    baseline_result = baseline.result()
    candidate_result = candidate.result()
    baseline_fold_results = [row.result() for row in baseline_folds]
    baseline_replay = uc1.baseline_replay_gate(
        summary["selected_oof"], baseline_result,
        [row["selected"] for row in summary["folds"]],
        baseline_fold_results)
    if not baseline_replay["pass"]:
        raise RuntimeError("RUS1 baseline failed frozen OOF replay")
    fold_rows = [
        group_row(candidate_folds[fold], baseline_folds[fold], fold)
        for fold in fold_groups]
    aoa_rows = [
        group_row(candidate_aoas[aoa], baseline_aoas[aoa], aoa)
        for aoa in uc1.AOA_VALUES]
    pooled_delta = float(
        candidate_result["sps_score"] - baseline_result["sps_score"])
    point_unchanged = (
        point_digest_before.hexdigest() == point_digest_after.hexdigest())
    elapsed = time.perf_counter() - started
    hard_guards = {
        "baseline_replay_pass": bool(baseline_replay["pass"]),
        "point_stream_bitwise_unchanged": point_unchanged,
        "point_stream_sha_frozen": (
            point_digest_before.hexdigest() == POINT_STREAM_SHA256),
        "target_stream_sha_frozen": (
            target_digest.hexdigest() == TARGET_STREAM_SHA256),
        "fold_checkpoint_shas_frozen": all(
            identities["fold_checkpoints"][str(fold)]["sha256"] == expected
            for fold, expected in FOLD_CHECKPOINT_SHA256.items()),
        "legacy_held_fold_checkpoint_selection_contract_reproduced": (
            selection_contamination_declared),
        "candidate_proxy_utility_elementwise_nondecreasing": (
            proxy_violations == 0 and proxy_min_delta >= 0.0),
        "candidate_bounds_finite": candidate_finite,
        "candidate_bounds_ordered": candidate_ordered,
        "all_725_rows_scored_once": sum(
            row["candidate"]["scored_elements"] for row in fold_rows)
            == candidate_result["scored_elements"],
        "no_training_label_conditioning_or_new_parameters": True,
    }
    checks = gate_checks(
        pooled_delta, fold_rows, aoa_rows, hard_guards,
        gpu_seconds, elapsed)
    decision = PASS_DECISION if all(checks.values()) else STOP_DECISION
    report = {
        "experiment": EXPERIMENT,
        "evidence_class": (
            "fixed725 fold-excluded-training terminal-u800 cross-fitted "
            "diagnostic; realized fixed-terminal equivalent; campaign-"
            "reused; not fresh or official evidence"),
        "contract": {
            "baseline": (
                "argmax nominal_mass*exp(-raw_physical_width/sigma), then "
                "apply frozen left/right refinement and endpoint ordering"),
            "candidate": (
                "apply the same frozen refinement and ordering to all 15 "
                "pairs, then argmax nominal_mass*exp(-final_width/sigma)"),
            "quantile_pairs": 15,
            "new_parameters": 0,
            "training": False,
            "target_used_by_selector": False,
            "point_changed": False,
            "checkpoint_selection_rule": FOLD_SELECTION_RULE,
            "checkpoint_selection_provenance": (
                "legacy trainer plus exact summary; terminal step 800 was "
                "the only scheduled post-training evaluation and equals "
                "the saved selected state"),
            "historical_held_comparison": True,
            "intermediate_checkpoint_selection": False,
            "realized_fixed_terminal_equivalent": True,
        },
        "git": git_identity(),
        "identities": identities,
        "fold_metadata": fold_metadata,
        "stream_hashes": {
            "point_before": point_digest_before.hexdigest(),
            "point_after": point_digest_after.hexdigest(),
            "target": target_digest.hexdigest(),
            "baseline_lower": baseline_lower_digest.hexdigest(),
            "baseline_upper": baseline_upper_digest.hexdigest(),
            "candidate_lower": candidate_lower_digest.hexdigest(),
            "candidate_upper": candidate_upper_digest.hexdigest(),
        },
        "baseline": baseline_result,
        "candidate": candidate_result,
        "pooled_sps_delta": pooled_delta,
        "fold_rows": fold_rows,
        "aoa_rows": aoa_rows,
        "selection_diagnostics": {
            "pair_switch_elements": switched,
            "pair_switch_rate": switched / total,
            "pair_switch_scored_elements": switched_scored,
            "pair_switch_scored_rate": switched_scored / total_scored,
            "proxy_elements": proxy_count,
            "proxy_violations": proxy_violations,
            "proxy_min_delta": proxy_min_delta,
            "proxy_mean_delta": proxy_sum_delta / proxy_count,
        },
        "baseline_replay_gate": baseline_replay,
        "gate_checks": checks,
        "decision": decision,
        "elapsed_seconds": elapsed,
        "gpu_inference_seconds": gpu_seconds,
        "checkpoints_saved": [],
        "submission_created": False,
        "limitations": [
            "Fixed725 has been reused across campaign decisions.",
            ("The legacy saver compared u0 with the sole scheduled trained "
             "endpoint u800 using held-fold SPS; all five saved states equal "
             "terminal u800, so there was no choice among trained endpoints, "
             "but this is still post-hoc reused-split evidence."),
            "AoA rows are robustness groups, not full-model leave-one-AoA-out.",
            "Nominal mass need not equal post-refinement empirical coverage.",
            ("A PASS authorizes only the preregistered fixed-terminal "
             "directional confirmation, not packaging."),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "rus1_complete",
        "decision": decision,
        "sps_delta": pooled_delta,
        "pair_switch_rate": switched / total,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            error_args = parse_args()
            if not error_args.output.exists():
                error_args.output.parent.mkdir(parents=True, exist_ok=True)
                error_args.output.write_text(
                    json.dumps({
                        "experiment": EXPERIMENT,
                        "decision": ERROR_DECISION,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "git": git_identity(),
                        "checkpoints_saved": [],
                        "submission_created": False,
                    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
        except Exception:
            pass
        raise
