#!/usr/bin/env python3
"""Read-only B210 Real AoA gradient-geometry audit.

This opens only the clean81 training prefixes.  It never calls an optimizer,
does not save a model, and compares the native supervised gradients induced by
five AoA groups on a fixed, recorded two-window-per-round sample plan.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
from pathlib import Path
import sys
import time
from collections.abc import Mapping, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_npt1_member_sync_tke_loao as npt1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig  # noqa: E402


AOAS = (0, 5, 10, 15, 20)
MANIFEST_FILE_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
MANIFEST_SEMANTIC_SHA256 = (
    "d0f52e7ffdec0fad0cbd653ebb41a3d0d3c76dd2ee99a2b927cc9b0d14a76d09")
EXPECTED_TRAJECTORIES = 81
EXPECTED_WINDOWS = 2_544
EXPECTED_LABELS = {"sim10k", "real4k"}
SCOPE_PREFIXES = {
    "shared_decoder": ("up.", "decoder.", "decoder_regime_film.",
                       "mean_profile_refiner."),
    "point_heads": ("member_point_heads.", "energy_controller."),
    "uq": ("uq_projections.", "uq_fusion.", "quantile_head.",
           "temporal_uq_refiner.", "utility_refinement_head."),
}


def route_scope(parameter_name: str) -> str | None:
    """Return the one audit scope owning a state-dict parameter name."""
    found = [scope for scope, prefixes in SCOPE_PREFIXES.items()
             if parameter_name.startswith(prefixes)]
    if len(found) > 1:
        raise RuntimeError(f"ambiguous scope for {parameter_name}")
    return found[0] if found else None


def evenly_spread_indices(population: int, count: int) -> list[int]:
    if population < 1 or count < 1 or count > population:
        raise ValueError("invalid deterministic index-plan dimensions")
    if count == 1:
        return [population // 2]
    return [round(index * (population - 1) / (count - 1))
            for index in range(count)]


def make_aoa_index_plan(windows, metadata: Sequence[Mapping[str, object]],
                        rounds: int, per_round: int = 2) -> dict[int, list[list[int]]]:
    needed = rounds * per_round
    by_aoa = {aoa: [] for aoa in AOAS}
    for index, window in enumerate(windows):
        aoa = int(metadata[int(window.trajectory)]["aoa"])
        if aoa in by_aoa:
            by_aoa[aoa].append(index)
    plan = {}
    for aoa, population in by_aoa.items():
        local = evenly_spread_indices(len(population), needed)
        chosen = [population[index] for index in local]
        if len(chosen) != needed or len(set(chosen)) != needed:
            raise RuntimeError(f"AoA {aoa} index plan is not unique")
        plan[aoa] = [chosen[offset:offset + per_round]
                     for offset in range(0, needed, per_round)]
    return plan


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("cosine inputs must be same-shape vectors")
    left_norm = torch.linalg.vector_norm(left).item()
    right_norm = torch.linalg.vector_norm(right).item()
    if not math.isfinite(left_norm) or not math.isfinite(right_norm):
        raise FloatingPointError("non-finite gradient vector")
    if left_norm == 0.0 or right_norm == 0.0:
        raise FloatingPointError("zero gradient vector has undefined cosine")
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("cannot summarize empty/non-finite cosines")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2 else
              (ordered[middle - 1] + ordered[middle]) / 2.0)
    return {"negative_fraction": sum(value < 0.0 for value in ordered) / len(ordered),
            "min": ordered[0], "median": median}


def decide(scope_summaries: Mapping[str, Mapping[str, float]]) -> str:
    for scope in ("shared_decoder", "point_heads"):
        values = scope_summaries[scope]
        if (float(values["negative_fraction"]) >= 0.10
                and float(values["persistent_min"]) <= -0.02):
            return "PCGRAD_HEADROOM"
    return "STOP_GRADIENT_SURGERY_NO_HEADROOM"


class PrefixOnlyAudit:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.opened: set[str] = set()

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "fit" or str(row["file"]) == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("audit may only open included clean81 fit prefixes")

    def record_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if phase != "fit" or filename in self.opened:
            raise RuntimeError("invalid or duplicate prefix access")
        self.opened.add(filename)
        self.events.append({"file": filename, "aoa": int(row["aoa"]),
                            "phase": phase, "max_read_frame_exclusive": int(row["val_boundary"])})

    def report(self, expected_files: Sequence[str]) -> dict:
        expected = sorted(str(value) for value in expected_files)
        opened = sorted(self.opened)
        return {
            "opened_file_allowlist": opened,
            "expected_file_allowlist": expected,
            "allowlist_exact": opened == expected,
            "prefix_only": all(row["phase"] == "fit" for row in self.events),
            "tail_arrays_read": 0,
            "held_or_clean716_read": False,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "events": self.events,
        }


def normalized_checkpoint_stats(payload: Mapping[str, object]) -> dict[str, list[float]]:
    raw = payload.get("normalizer")
    keys = ("mean_input", "mean_target", "std_input", "std_target")
    if not isinstance(raw, Mapping) or any(key not in raw for key in keys):
        raise ValueError("checkpoint lacks normalizer statistics")
    return {
        key: torch.as_tensor(raw[key], dtype=torch.float32).reshape(-1).tolist()
        for key in keys
    }


def load_checkpoint(label: str, path: Path, expected_sha: str, normalizer,
                    device: torch.device) -> tuple[EnsembleUNet2d, dict]:
    expected = dr1.normalized_sha256(expected_sha, "checkpoint SHA")
    actual = dr1.sha256(path)
    if actual != expected:
        raise RuntimeError(f"checkpoint SHA mismatch: {path}")
    try:
        payload = torch.load(
            str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("checkpoint lacks model_config/model_state_dict")
    raw = dict(config)
    if raw.pop("architecture", None) != "ensemble_unet":
        raise ValueError("checkpoint is not an ensemble U-Net")
    parsed_config = EnsembleUNet2dConfig(**raw)
    parsed_config.validate()
    if parsed_config.base != 210:
        raise ValueError("gradient audit requires a B210 checkpoint")
    expected_phase, expected_iteration = (
        ("sim", 10_000) if label == "sim10k" else ("real", 4_000))
    if (payload.get("phase") != expected_phase
            or int(payload.get("iteration", -1)) != expected_iteration):
        raise ValueError(f"{label} checkpoint phase/iteration changed")
    stored_normalizer = normalized_checkpoint_stats(payload)
    audit_normalizer = npt1.ensemble_train.common.normalizer_dict(normalizer)
    normalizer_matches_audit = all(torch.allclose(
        torch.tensor(stored_normalizer[key]),
        torch.tensor(audit_normalizer[key]), rtol=0.0, atol=1e-7)
        for key in stored_normalizer)
    if label == "real4k" and not normalizer_matches_audit:
        raise ValueError("mature Real4k checkpoint normalizer differs from audit")
    model = EnsembleUNet2d(parsed_config).float()
    model.load_state_dict(state, strict=True)
    npt1.ensemble_train.common.set_model_stats(model, normalizer)
    normalized_config = model.architecture_config()
    report = {"path": str(path.resolve()), "checkpoint_sha256": actual,
              "checkpoint_model_config_sha256": dr1.canonical_sha256(config),
              "normalized_model_config_sha256": dr1.canonical_sha256(
                  normalized_config),
              "checkpoint_model_state_sha256": dr1.model_state_sha256(state),
              "iteration": payload.get("iteration"), "phase": payload.get("phase"),
              "stored_normalizer": stored_normalizer,
              "stored_normalizer_matches_audit": normalizer_matches_audit,
              "strict_load": True}
    return model.to(device).eval(), report


def collect_scope_gradients(model: EnsembleUNet2d) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    chunks = {scope: [] for scope in SCOPE_PREFIXES}
    audit = {scope: {"parameters": 0, "elements": 0, "none_grad_parameters": [],
                     "nonfinite_grad_parameters": []} for scope in SCOPE_PREFIXES}
    for name, parameter in model.named_parameters():
        scope = route_scope(name)
        if scope is None:
            continue
        record = audit[scope]
        record["parameters"] += 1
        record["elements"] += parameter.numel()
        gradient = parameter.grad
        if gradient is None:
            record["none_grad_parameters"].append(name)
            chunks[scope].append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            value = gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(value).all():
                record["nonfinite_grad_parameters"].append(name)
            chunks[scope].append(value)
    vectors = {scope: torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)
               for scope, parts in chunks.items()}
    for scope, record in audit.items():
        if not record["parameters"]:
            raise RuntimeError(f"checkpoint has no parameters in {scope}")
        record["none_grad_count"] = len(record["none_grad_parameters"])
        record["nonfinite_grad_count"] = len(record["nonfinite_grad_parameters"])
    return vectors, audit


def plan_report(plan: Mapping[int, Sequence[Sequence[int]]], windows, metadata) -> dict:
    rows = {str(aoa): [[{"dataset_index": index,
                         "file": metadata[int(windows[index].trajectory)]["file"],
                         "start": int(windows[index].start)} for index in group]
                        for group in groups] for aoa, groups in plan.items()}
    return {"rounds": len(next(iter(plan.values()))), "windows_per_group_per_round": 2,
            "groups": rows, "sha256": dr1.canonical_sha256(rows)}


def audit_checkpoint(model, dataset, plan, normalizer, device: torch.device,
                     label: str) -> dict:
    round_cosines = {scope: [] for scope in SCOPE_PREFIXES}
    gradient_sums = {
        scope: {aoa: None for aoa in AOAS} for scope in SCOPE_PREFIXES}
    finite_audit = []
    for round_index in range(len(plan[AOAS[0]])):
        vectors = {}
        for aoa in AOAS:
            model.zero_grad(set_to_none=True)
            indices = plan[aoa][round_index]
            tagged = npt1.ensemble_train.TrajectoryTaggedDataset(dataset)
            batch = default_collate([tagged[index] for index in indices])
            member_mask = torch.ones((len(indices), len(model.member_point_heads)),
                                     dtype=torch.bool, device=device)
            values = npt1.backward_real_supervised_npt1(
                model, batch, normalizer, device, 4000,
                auxiliary_weight=0.0, member_mask=member_mask,
                audit_controller_gradients=False, input_augmentation=None)
            if any(not math.isfinite(float(value)) for value in values.values()):
                raise FloatingPointError("non-finite NPT1 metric")
            vectors[aoa], per_scope = collect_scope_gradients(model)
            finite_audit.append({"round": round_index, "aoa": aoa, "scopes": per_scope})
            for scope in SCOPE_PREFIXES:
                if per_scope[scope]["nonfinite_grad_count"]:
                    raise FloatingPointError(f"non-finite {scope} gradient")
                if per_scope[scope]["none_grad_count"]:
                    raise RuntimeError(f"missing {scope} gradient")
                current = vectors[aoa][scope]
                total = gradient_sums[scope][aoa]
                if total is None:
                    gradient_sums[scope][aoa] = current.clone()
                else:
                    total.add_(current)
        for scope in SCOPE_PREFIXES:
            pairs = {f"{left}-{right}": cosine(vectors[left][scope], vectors[right][scope])
                     for left, right in itertools.combinations(AOAS, 2)}
            round_cosines[scope].append(pairs)
        print(json.dumps({
            "event": "b210_real_aoa_gradient_geometry_progress",
            "checkpoint": label,
            "round": round_index + 1,
            "rounds": len(plan[AOAS[0]]),
        }), flush=True)
    scope_report = {}
    for scope in SCOPE_PREFIXES:
        all_values = [value for row in round_cosines[scope] for value in row.values()]
        persistent = {f"{left}-{right}": cosine(gradient_sums[scope][left],
                                                  gradient_sums[scope][right])
                      for left, right in itertools.combinations(AOAS, 2)}
        scope_report[scope] = {"round_pair_cosines": round_cosines[scope],
                               **summarize(all_values), "persistent_pair_cosines": persistent,
                               "persistent_min": min(persistent.values())}
    return {"scopes": scope_report, "finite_none_gradient_audit": finite_audit,
            "decision": decide(scope_report)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", nargs=3, metavar=("LABEL", "PATH", "SHA256"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.rounds < 1 or len(args.checkpoint) != 2:
        raise ValueError("output must be new, rounds positive, and exactly two checkpoints supplied")
    labels = [row[0] for row in args.checkpoint]
    if set(labels) != EXPECTED_LABELS:
        raise ValueError("checkpoint labels must be exactly sim10k and real4k")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B210 gradient audit requires CUDA")
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    if dr1.sha256(args.clean_manifest) != MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 manifest file SHA changed")
    rows, manifest = dr1.load_manifest(
        args.clean_manifest, MANIFEST_SEMANTIC_SHA256)
    if (len(rows) != EXPECTED_TRAJECTORIES
            or any(str(row["file"]) == dr1.EXCLUDED_BASENAME for row in rows)):
        raise RuntimeError("clean81 manifest membership changed")
    access = PrefixOnlyAudit()
    trajectories, metadata, windows = dr1.load_prefix_rows(args.real_dir, rows, "fit", access)
    if len(windows) != EXPECTED_WINDOWS:
        raise RuntimeError("clean81 optimizer-window count changed")
    sufficient = smu1.window_sufficient_statistics(trajectories, windows)
    statistics = smu1.fold_statistics(sufficient, range(len(windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    normalizer_values = npt1.ensemble_train.common.normalizer_dict(normalizer)
    plan = make_aoa_index_plan(windows, metadata, args.rounds)
    plan_info = plan_report(plan, windows, metadata)
    dataset = npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    results = {}
    normalized_config_sha = None
    for label, path_text, expected_sha in args.checkpoint:
        model, checkpoint = load_checkpoint(
            label, Path(path_text), expected_sha, normalizer, device)
        if normalized_config_sha is None:
            normalized_config_sha = checkpoint["normalized_model_config_sha256"]
        elif checkpoint["normalized_model_config_sha256"] != normalized_config_sha:
            raise ValueError(
                "the two audit checkpoints do not share a normalized model config")
        results[label] = {"checkpoint": checkpoint,
                          "gradient_geometry": audit_checkpoint(
                              model, dataset, plan, normalizer, device, label)}
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    access_report = access.report([row["file"] for row in rows])
    if (not access_report["allowlist_exact"]
            or not access_report["prefix_only"]
            or access_report["excluded_opened"]):
        raise RuntimeError("prefix-only access audit failed")
    result = {"schema": "b210_real_aoa_gradient_geometry_audit_v1", "read_only": True,
              "git": dr1.git_identity(), "manifest": manifest,
              "manifest_file_sha256": dr1.sha256(args.clean_manifest),
              "normalizer": {"values": normalizer_values,
                             "sha256": dr1.canonical_sha256(normalizer_values),
                             "statistics_sha256": smu1.statistics_digest(statistics),
                             "kind": "full_clean81_optimizer_windows"},
              "sample_plan": plan_info,
              "prefix_access_audit": access_report,
              "checkpoints": results,
              "decision_source": "real4k",
              "decision": results["real4k"]["gradient_geometry"]["decision"],
              "elapsed_seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dr1.write_json_atomic(args.output, result)
    print(json.dumps({"event": "b210_real_aoa_gradient_geometry_complete",
                      "output": str(args.output),
                      "decisions": {label: value["gradient_geometry"]["decision"] for label, value in results.items()}}))


if __name__ == "__main__":
    main()
