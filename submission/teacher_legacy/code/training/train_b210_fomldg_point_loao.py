#!/usr/bin/env python3
"""Matched ERM2/FOMLDG Real-stage screen for the sealed Sim10k B210.

The candidate evaluates its query gradient after one detached support SGD
step.  The control executes the same stateless query call at an unchanged
parameter copy.  Deployment always uses the ordinary, fixed endpoint model.
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
import subprocess
import sys
import time
from typing import Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for _path in (CODE_ROOT, TRAINING_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_domain_joint_loao as joint  # noqa: E402
import train_b210_scratch_real_dense_aug as dense4k  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


EXPERIMENT = "b210_real_fomldg_point_loao"
SCHEMA = "b210_real_fomldg_point_loao_v1"
ENDPOINT_SCHEMA = "b210_real_fomldg_raw_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_M11_FOMLDG_MATCHED_ERM2"

HELD_AOA = 10
FIT_AOAS = (0, 5, 15, 20)
ORDERED_AOA_PAIRS = tuple(
    (source, query)
    for source in FIT_AOAS
    for query in FIT_AOAS
    if source != query)
FORMAL_UPDATES = 1_200
FORMAL_BATCH_SIZE = 4
SEED = 0
FORMAL_SEEDS = (0, 4, 8)
FORMAL_SCHEDULE_SHA256 = {
    0: "61e644fbe2b2c9b17853338c3cd1d7c27d65aa6577f5e6eb90fa8660a236b80b",
    4: "0613db780cd3baeac03a5c8770be70a9037a416d5220af2e1abfb4102d1315a6",
    8: "7dca9a83afa33091f91a1cedd7de446edbd358246c24c2770afb9ff2944645e9",
}
LR = 2e-4
INNER_LR = 2e-4
BETA = 1.0
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
EXPECTED_PARAMETERS = 178_705_575
EXPECTED_TRAINABLE_PARAMETERS = 175_025_153
EXPECTED_FROZEN_PARAMETERS = 3_680_422
EXPECTED_HELD_TRAJECTORIES = 18
EXPECTED_HELD_WINDOWS = 162
POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
}
GATE = {
    "point_delta_min": 0.04,
    "component_floor": -0.05,
    "positive_trajectories_min": 12,
    "worst_trajectory_point_floor": -0.10,
}

DEFAULT_START = Path(
    "/root/autodl-tmp/realpde/runs/b210_retrain_20260826/"
    "sim10k_repro/model_10000.pth")
START_SHA256 = (
    "8a6d9a1055269f95a308b42e73ee94b00b005edbf550430dfa096a610e2e0f29")
DEFAULT_RUNTIME_MANIFEST = Path(
    "/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/"
    "runtime_train_manifest.json")
RUNTIME_MANIFEST_FILE_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
RUNTIME_MANIFEST_SEMANTIC_SHA256 = (
    "d0f52e7ffdec0fad0cbd653ebb41a3d0d3c76dd2ee99a2b927cc9b0d14a76d09")
FIT_ROWS_SHA256 = (
    "266cb51bc3dacee621f2cf0f9628dbbdcae53fcfdf3f2d08cc36aeb7393c498e")
DEFAULT_EVALUATION_MANIFEST = Path(
    "/root/autodl-tmp/realpde/runs/b210_retrain_20260826/"
    "split_manifest.json")
EVALUATION_MANIFEST_SHA256 = (
    "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
HELD_IDENTITY_SHA256 = (
    "6f8ea8f6c7deb29b5ad564e9d8de59a48c006df49f24e0d2f5d6d20399fedd66")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def git_identity() -> dict[str, str]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments), cwd=REPO_ROOT, text=True).strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def derive_seed(base_seed: int, update: int, role: str,
                draw_kind: str, slot: int = -1) -> int:
    payload = [EXPERIMENT, int(base_seed), int(update), role, draw_kind, int(slot)]
    value = int.from_bytes(
        hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode(
            "utf-8")).digest()[:8], "big")
    return value % (2**31 - 1)


def keyed_index(size: int, *parts: object) -> int:
    if size <= 0:
        raise ValueError("cannot draw from an empty sequence")
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=False)
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8],
                           "big")
    return value % size


def bootstrap_mask(base_seed: int, update: int, role: str,
                   samples: Sequence[Mapping[str, object]],
                   member_count: int = 4) -> list[list[int]]:
    result: list[list[int]] = []
    for slot, sample in enumerate(samples):
        row = []
        for member in range(member_count):
            code = keyed_index(
                10_000, EXPERIMENT, base_seed, update, role, "bootstrap",
                slot, sample["file"], sample["start"], member)
            row.append(int(code < 7_500))
        if not any(row):
            fallback = keyed_index(
                member_count, EXPERIMENT, base_seed, update, role,
                "bootstrap_fallback", slot, sample["file"], sample["start"])
            row[fallback] = 1
        result.append(row)
    return result


def build_cross_condition_schedule(
        rows: Sequence[Mapping[str, object]], updates: int,
        batch_size: int, seed: int) -> dict[str, object]:
    if updates <= 0 or batch_size <= 0:
        raise ValueError("updates and batch size must be positive")
    if updates % len(ORDERED_AOA_PAIRS) != 0:
        raise ValueError("updates must balance all 12 directed AoA pairs")
    fit_rows = [dict(row) for row in rows if int(row["aoa"]) in FIT_AOAS]
    if canonical_sha256(fit_rows) != FIT_ROWS_SHA256:
        raise RuntimeError("fit-side Real rows changed")
    by_aoa: dict[int, dict[int, dict]] = {aoa: {} for aoa in FIT_AOAS}
    for row in fit_rows:
        aoa = int(row["aoa"])
        reynolds = int(row["re"])
        if reynolds in by_aoa[aoa]:
            raise ValueError("an AoA contains duplicate exact-Re trajectories")
        by_aoa[aoa][reynolds] = row

    schedule_rows = []
    for update in range(1, updates + 1):
        source_aoa, query_aoa = ORDERED_AOA_PAIRS[
            (update - 1) % len(ORDERED_AOA_PAIRS)]
        shared_re = sorted(set(by_aoa[source_aoa]) & set(by_aoa[query_aoa]))
        if not shared_re:
            raise RuntimeError("an AoA pair has no common exact Re")
        source_samples = []
        query_samples = []
        for slot in range(batch_size):
            reynolds = shared_re[keyed_index(
                len(shared_re), EXPERIMENT, seed, update, slot, "re",
                source_aoa, query_aoa)]
            source = by_aoa[source_aoa][reynolds]
            query = by_aoa[query_aoa][reynolds]
            starts = sorted(
                set(source["train_starts"]) & set(query["train_starts"]))
            if not starts:
                raise RuntimeError("matched exact-Re trajectories have no common start")
            start = starts[keyed_index(
                len(starts), EXPERIMENT, seed, update, slot, "start",
                source_aoa, query_aoa, reynolds)]
            source_samples.append({
                "file": str(source["file"]), "re": reynolds,
                "aoa": source_aoa, "start": int(start)})
            query_samples.append({
                "file": str(query["file"]), "re": reynolds,
                "aoa": query_aoa, "start": int(start)})
        rng = {
            role: {
                kind: derive_seed(seed, update, role, kind)
                for kind in ("gain", "blur", "noise", "dropout")
            }
            for role in ("A", "B")
        }
        schedule_rows.append({
            "update": update,
            "source_aoa": source_aoa,
            "query_aoa": query_aoa,
            "A": source_samples,
            "B": query_samples,
            "A_bootstrap_mask": bootstrap_mask(
                seed, update, "A", source_samples),
            "B_bootstrap_mask": bootstrap_mask(
                seed, update, "B", query_samples),
            "rng": rng,
        })
    pair_counts = {
        f"{source}->{query}": sum(
            row["source_aoa"] == source and row["query_aoa"] == query
            for row in schedule_rows)
        for source, query in ORDERED_AOA_PAIRS
    }
    role_counts = {
        str(aoa): {
            "A": sum(row["source_aoa"] == aoa for row in schedule_rows),
            "B": sum(row["query_aoa"] == aoa for row in schedule_rows),
        }
        for aoa in FIT_AOAS
    }
    if len(set(pair_counts.values())) != 1:
        raise RuntimeError("directed AoA pairs are not balanced")
    if len({value for counts in role_counts.values()
            for value in counts.values()}) != 1:
        raise RuntimeError("AoA support/query roles are not balanced")
    return {
        "schema": "b210_real_fomldg_exact_re_schedule_v1",
        "seed": seed,
        "updates": updates,
        "batch_size_per_role": batch_size,
        "ordered_aoa_pairs": [list(value) for value in ORDERED_AOA_PAIRS],
        "pair_counts": pair_counts,
        "role_counts": role_counts,
        "rows": schedule_rows,
        "sha256": canonical_sha256(schedule_rows),
    }


class PairAccessAudit:
    """Fail closed until both raw endpoints have been sealed."""

    def __init__(self, fit_files: Sequence[str], held_files: Sequence[str]):
        self.fit_files = set(map(str, fit_files))
        self.held_files = set(map(str, held_files))
        if not self.fit_files or not self.held_files or (
                self.fit_files & self.held_files):
            raise ValueError("fit/held file allowlists are invalid")
        self.fit_events: list[dict] = []
        self.held_events: list[dict] = []
        self.seals: dict[str, dict] = {}

    @property
    def all_sealed(self) -> bool:
        return set(self.seals) == {"candidate", "control"}

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if phase == "fit":
            if filename not in self.fit_files or self.all_sealed:
                raise RuntimeError("unauthorized fit-prefix open")
        elif phase == "held":
            if filename not in self.held_files or not self.all_sealed:
                raise RuntimeError("held tail opened before dual seal")
        else:
            raise ValueError(f"unknown access phase {phase}")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if phase != "fit":
            raise ValueError("load_prefix_rows may only record fit opens")
        if any(event["file"] == filename for event in self.fit_events):
            raise RuntimeError("fit file opened twice")
        self.fit_events.append({
            "file": filename,
            "aoa": int(row["aoa"]),
            "slice": f"u/v[:{int(row['val_boundary'])},::2,::2]",
            "max_frame_exclusive": int(row["val_boundary"]),
            "all_sealed_before_open": self.all_sealed,
        })

    def seal(self, arm: str, seal: Mapping[str, object]) -> None:
        if arm not in {"candidate", "control"} or arm in self.seals:
            raise ValueError("invalid or duplicate endpoint seal")
        path = Path(str(seal["path"]))
        if not path.is_file() or sha256(path) != seal["file_sha256"]:
            raise RuntimeError("cannot register an unverified endpoint")
        self.seals[arm] = dict(seal)

    def record_held(self, row: Mapping[str, object], starts: Sequence[int]) -> None:
        self.before_open(row, "held")
        filename = str(row["file"])
        if any(event["file"] == filename for event in self.held_events):
            raise RuntimeError("held file opened twice")
        self.held_events.append({
            "file": filename,
            "aoa": int(row["aoa"]),
            "starts": [int(value) for value in starts],
            "slices": [f"u/v[{int(value)}:{int(value)+40},::2,::2]"
                       for value in starts],
            "all_sealed_before_open": self.all_sealed,
        })

    def report(self) -> dict[str, object]:
        opened_fit = {event["file"] for event in self.fit_events}
        opened_held = {event["file"] for event in self.held_events}
        return {
            "fit_allowlist_exact": opened_fit == self.fit_files,
            "held_allowlist_exact": opened_held == self.held_files,
            "all_endpoints_sealed": self.all_sealed,
            "endpoint_seals": dict(self.seals),
            "fit_events": list(self.fit_events),
            "held_events": list(self.held_events),
            "held_first_open_after_both_seals": bool(
                self.held_events
                and self.held_events[0]["all_sealed_before_open"]),
            "held_windows_opened": sum(
                len(event["starts"]) for event in self.held_events),
        }


@contextmanager
def rng_scope(seed: int, device: torch.device) -> Iterator[None]:
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None
                   else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


class PointTrainingView(nn.Module):
    def __init__(self, model: EnsembleUNet2d):
        super().__init__()
        self.model = model

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return joint.forward_point_training(self.model, history)


def build_fast_parameter_mapping(
        module: nn.Module, gradients: Sequence[torch.Tensor | None],
        alpha: float) -> tuple[OrderedDict[str, torch.Tensor],
                               OrderedDict[str, torch.Tensor],
                               tuple[torch.Tensor, ...]]:
    trainable = [(name, parameter) for name, parameter in module.named_parameters()
                 if parameter.requires_grad]
    if len(trainable) != len(gradients):
        raise ValueError("gradient tuple does not match trainable parameters")
    by_name = dict(zip((name for name, _ in trainable), gradients))
    parameters: OrderedDict[str, torch.Tensor] = OrderedDict()
    fast_trainable = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            parameters[name] = parameter.detach()
            continue
        gradient = by_name[name]
        if gradient is None or alpha == 0.0:
            value = parameter.detach().clone()
        else:
            value = parameter.detach().add(gradient.detach(), alpha=-alpha)
        value.requires_grad_(True)
        parameters[name] = value
        fast_trainable.append(value)
    buffers = OrderedDict(
        (name, value.detach()) for name, value in module.named_buffers())
    return parameters, buffers, tuple(fast_trainable)


def resolve_schedule_indices(schedule: Mapping[str, object], windows: Sequence,
                             metadata: Sequence[Mapping[str, object]]) -> dict:
    index_by_key = {}
    for index, window in enumerate(windows):
        filename = str(metadata[int(window.trajectory)]["file"])
        key = (filename, int(window.start))
        if key in index_by_key:
            raise RuntimeError("duplicate fit window identity")
        index_by_key[key] = index
    resolved = []
    for row in schedule["rows"]:
        copied = dict(row)
        for role in ("A", "B"):
            copied[f"{role}_indices"] = [
                index_by_key[(str(sample["file"]), int(sample["start"]))]
                for sample in row[role]
            ]
        resolved.append(copied)
    return {
        "rows": resolved,
        "sha256": canonical_sha256([
            {
                "update": row["update"],
                "A_indices": row["A_indices"],
                "B_indices": row["B_indices"],
            }
            for row in resolved
        ]),
    }


def stack_batch(dataset, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    values = [dataset[int(index)] for index in indices]
    return (torch.stack([value[0] for value in values]),
            torch.stack([value[1] for value in values]))


def prepare_batch(dataset, indices: Sequence[int], normalizer,
                  device: torch.device, rng: Mapping[str, int],
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    history, target = stack_batch(dataset, indices)
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    with rng_scope(int(rng["gain"]), device):
        history_input = ensemble_train.add_training_input_gain(
            history, dense4k.AUGMENTATION["input_gain_std"])
    history_norm, target_norm = normalizer.preprocess(history_input, target)
    with rng_scope(int(rng["blur"]), device):
        history_norm = ensemble_train.masked_velocity_blur(
            history_norm, history,
            dense4k.AUGMENTATION["input_blur_probability"],
            dense4k.AUGMENTATION["input_blur_strength"])
    with rng_scope(int(rng["noise"]), device):
        history_norm = ensemble_train.add_training_input_noise(
            history_norm, history, dense4k.AUGMENTATION["input_noise_std"])
    return history_norm, target_norm


def point_member_objective(
        prediction: torch.Tensor, members: torch.Tensor,
        target: torch.Tensor, trajectory_ids: torch.Tensor,
        member_mask: torch.Tensor, normalizer, update: int,
        ) -> tuple[torch.Tensor, dict[str, float]]:
    contract = joint.DOMAIN_LOSS["real"]
    point_loss, components = ensemble_train.point_loss(
        joint._loss_args(), prediction, target, normalizer,
        contract["accuracy_weights"], update)
    member_loss, fraction = ensemble_train.bootstrap_member_point_loss(
        joint._loss_args(), members, target, trajectory_ids,
        normalizer, contract["accuracy_weights"], update,
        member_mask=member_mask)
    loss = (contract["point_weight"] * point_loss
            + contract["member_weight"] * member_loss)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite Real point/member objective")
    return loss, {
        "loss": float(loss.detach()),
        "point_loss": float(point_loss.detach()),
        "member_loss": float(member_loss.detach()),
        "bootstrap_fraction": float(fraction.detach()),
        **{key: float(value.detach()) for key, value in components.items()},
    }


def gradient_norm(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    values = [gradient.float().square().sum()
              for gradient in gradients if gradient is not None]
    if not values:
        raise RuntimeError("gradient estimator contains no gradients")
    return torch.stack(values).sum().sqrt()


def install_average_gradient(
        parameters: Sequence[torch.nn.Parameter],
        support: Sequence[torch.Tensor | None],
        query: Sequence[torch.Tensor | None]) -> None:
    if not (len(parameters) == len(support) == len(query)):
        raise ValueError("gradient estimator lengths differ")
    for parameter, first, second in zip(parameters, support, query):
        if first is None and second is None:
            parameter.grad = None
        elif first is None:
            second.mul_(0.5)
            parameter.grad = second
        elif second is None:
            first.mul_(0.5)
            parameter.grad = first
        else:
            first.add_(second).mul_(0.5)
            parameter.grad = first


def _gradient(
        view: PointTrainingView, trainable: Sequence[torch.Tensor],
        history: torch.Tensor, target: torch.Tensor,
        trajectory_ids: torch.Tensor, member_mask: torch.Tensor,
        normalizer, update: int, dropout_seed: int,
        parameter_and_buffer_dicts=None,
        capture_outputs: bool = False,
        ) -> tuple[tuple[torch.Tensor | None, ...], dict[str, float],
                   tuple[torch.Tensor, torch.Tensor] | None]:
    with rng_scope(dropout_seed, history.device):
        if parameter_and_buffer_dicts is None:
            prediction, members = view(history)
        else:
            prediction, members = torch.func.functional_call(
                view, parameter_and_buffer_dicts, (history,), strict=True)
        loss, values = point_member_objective(
            prediction, members, target, trajectory_ids, member_mask,
            normalizer, update)
        gradients = torch.autograd.grad(
            loss, tuple(trainable), create_graph=False, retain_graph=False,
            allow_unused=True)
    if any(gradient is not None and (
            gradient.requires_grad or gradient.grad_fn is not None)
            for gradient in gradients):
        raise RuntimeError("first-order gradient unexpectedly retained a graph")
    detached = tuple(
        None if gradient is None else gradient.detach()
        for gradient in gradients)
    outputs = ((prediction.detach(), members.detach())
               if capture_outputs else None)
    return detached, values, outputs


def buffer_state_sha256(module: nn.Module) -> str:
    return dr1.model_state_sha256(OrderedDict(
        (name, value.detach().cpu()) for name, value in module.named_buffers()))


def alpha_zero_equivalence_preflight(
        start: Mapping[str, object], dataset, schedule: Mapping[str, object],
        normalizer, device: torch.device) -> dict[str, object]:
    """Verify the stateless query is exactly ERM when inner alpha is zero."""
    ensemble_train.seed_everything(int(schedule["seed"]))
    model = joint.build_start_model(start, device)
    ensemble_train.common.set_model_stats(model, normalizer)
    trainable, _trainability = joint.configure_point_only_trainable(model)
    model.train()
    view = PointTrainingView(model)
    view_trainable = tuple(
        parameter for parameter in view.parameters() if parameter.requires_grad)
    if tuple(map(id, trainable)) != tuple(map(id, view_trainable)):
        raise RuntimeError("preflight point-view parameter order changed")
    buffer_before = buffer_state_sha256(model)
    row = schedule["rows"][0]
    role_reports = {}
    for role in ("A", "B"):
        history, target = prepare_batch(
            dataset, row[f"{role}_indices"], normalizer, device,
            row["rng"][role])
        trajectory_ids = torch.as_tensor([
            dataset.windows[index].trajectory
            for index in row[f"{role}_indices"]], dtype=torch.long)
        member_mask = torch.as_tensor(
            row[f"{role}_bootstrap_mask"], dtype=torch.bool)
        direct_gradients, direct_values, direct_outputs = _gradient(
            view, view_trainable, history, target, trajectory_ids, member_mask,
            normalizer, 1, int(row["rng"][role]["dropout"]),
            capture_outputs=True)
        fast_parameters, fast_buffers, fast_trainable = (
            build_fast_parameter_mapping(view, direct_gradients, 0.0))
        functional_gradients, functional_values, functional_outputs = _gradient(
            view, fast_trainable, history, target, trajectory_ids, member_mask,
            normalizer, 1, int(row["rng"][role]["dropout"]),
            (fast_parameters, fast_buffers), capture_outputs=True)
        if direct_outputs is None or functional_outputs is None:
            raise RuntimeError("preflight did not capture point/member outputs")
        none_pattern_equal = all(
            (direct is None) == (functional is None)
            for direct, functional in zip(
                direct_gradients, functional_gradients))
        tensor_equal = none_pattern_equal and all(
            direct is None or torch.equal(direct, functional)
            for direct, functional in zip(
                direct_gradients, functional_gradients))
        difference_square = sum(
            float((direct.float() - functional.float()).square().sum())
            for direct, functional in zip(
                direct_gradients, functional_gradients)
            if direct is not None and functional is not None)
        reference_square = sum(
            float(direct.float().square().sum())
            for direct in direct_gradients if direct is not None)
        relative_l2_difference = math.sqrt(difference_square) / max(
            math.sqrt(reference_square), 1e-30)
        maximum_difference = max(
            (float((direct - functional).abs().max())
             for direct, functional in zip(
                 direct_gradients, functional_gradients)
             if direct is not None and functional is not None),
            default=0.0)
        role_reports[role] = {
            "prediction_bitwise_equal": torch.equal(
                direct_outputs[0], functional_outputs[0]),
            "members_bitwise_equal": torch.equal(
                direct_outputs[1], functional_outputs[1]),
            "loss_bitwise_equal": (
                direct_values["loss"] == functional_values["loss"]),
            "all_gradient_none_patterns_equal": none_pattern_equal,
            "all_nonempty_gradients_bitwise_equal": tensor_equal,
            "maximum_gradient_absolute_difference": maximum_difference,
            "relative_gradient_l2_difference": relative_l2_difference,
            "all_nonempty_gradients_within_fp32_tolerance": (
                none_pattern_equal and maximum_difference <= 2e-5
                and relative_l2_difference <= 1e-4),
            "direct_loss": direct_values["loss"],
            "functional_loss": functional_values["loss"],
            "fast_parameters_are_detached_leaves": all(
                value.is_leaf and value.grad_fn is None
                for value in fast_trainable),
            "batch_size": len(row[f"{role}_indices"]),
        }
        del history, target, direct_gradients, functional_gradients
        del direct_outputs, functional_outputs
        del fast_parameters, fast_buffers, fast_trainable
    report = {
        "roles": role_reports,
        "gradient_fp32_tolerance": {
            "maximum_absolute": 2e-5,
            "relative_l2": 1e-4,
        },
        "raw_gradients_fail_closed_if_graph_retained": True,
        "buffer_state_unchanged": buffer_state_sha256(model) == buffer_before,
        "pass": all(
            role["prediction_bitwise_equal"]
            and role["members_bitwise_equal"]
            and role["loss_bitwise_equal"]
            and role["all_gradient_none_patterns_equal"]
            and role["all_nonempty_gradients_within_fp32_tolerance"]
            and role["fast_parameters_are_detached_leaves"]
            for role in role_reports.values()),
    }
    report["pass"] = bool(report["pass"] and report["buffer_state_unchanged"])
    if not report["pass"]:
        raise RuntimeError(f"alpha-zero functional preflight failed: {report}")
    del model, view, trainable, view_trainable
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def train_arm(
        arm: str, start: Mapping[str, object], dataset,
        schedule: Mapping[str, object], normalizer, device: torch.device,
        output_dir: Path, access: PairAccessAudit,
        deadline: float) -> tuple[dict, dict]:
    if arm not in {"candidate", "control"}:
        raise ValueError(f"unknown arm {arm}")
    ensemble_train.seed_everything(int(schedule["seed"]))
    model = joint.build_start_model(start, device)
    ensemble_train.common.set_model_stats(model, normalizer)
    total_parameters = sum(value.numel() for value in model.parameters())
    trainable, trainability = joint.configure_point_only_trainable(model)
    if (total_parameters != EXPECTED_PARAMETERS
            or trainability["trainable_parameters"]
            != EXPECTED_TRAINABLE_PARAMETERS
            or trainability["frozen_uq_parameters"]
            != EXPECTED_FROZEN_PARAMETERS):
        raise RuntimeError("B210 point/UQ parameter partition changed")
    model.train()
    view = PointTrainingView(model)
    view_trainable = tuple(
        parameter for parameter in view.parameters() if parameter.requires_grad)
    if tuple(id(value) for value in view_trainable) != tuple(
            id(value) for value in trainable):
        raise RuntimeError("point view changed trainable parameter order")
    initial_state_sha = dr1.model_state_sha256(model.state_dict())
    initial_buffer_sha = buffer_state_sha256(model)
    uq_start_sha = dr1.model_state_sha256(joint.uq_parameter_state(model))
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    if optimizer.state:
        raise RuntimeError("optimizer did not start empty")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    trace_path = output_dir / f"{arm}_fit_trace.jsonl"
    actual_rows = []
    started = time.perf_counter()
    first_mechanics = None
    for row in schedule["rows"]:
        if time.perf_counter() > deadline:
            raise TimeoutError("M11 wall budget expired")
        update = int(row["update"])
        optimizer.zero_grad(set_to_none=True)
        history_a, target_a = prepare_batch(
            dataset, row["A_indices"], normalizer, device,
            row["rng"]["A"])
        history_b, target_b = prepare_batch(
            dataset, row["B_indices"], normalizer, device,
            row["rng"]["B"])
        trajectory_a = torch.as_tensor(
            [dataset.windows[index].trajectory for index in row["A_indices"]],
            dtype=torch.long)
        trajectory_b = torch.as_tensor(
            [dataset.windows[index].trajectory for index in row["B_indices"]],
            dtype=torch.long)
        mask_a = torch.as_tensor(row["A_bootstrap_mask"], dtype=torch.bool)
        mask_b = torch.as_tensor(row["B_bootstrap_mask"], dtype=torch.bool)

        support_gradient, support_values, _support_outputs = _gradient(
            view, view_trainable, history_a, target_a, trajectory_a, mask_a,
            normalizer, update, int(row["rng"]["A"]["dropout"]))
        alpha = INNER_LR if arm == "candidate" else 0.0
        fast_parameters, fast_buffers, fast_trainable = (
            build_fast_parameter_mapping(view, support_gradient, alpha))
        query_gradient, query_values, _query_outputs = _gradient(
            view, fast_trainable, history_b, target_b, trajectory_b, mask_b,
            normalizer, update, int(row["rng"]["B"]["dropout"]),
            (fast_parameters, fast_buffers))
        support_norm = gradient_norm(support_gradient)
        query_norm = gradient_norm(query_gradient)
        inner_displacement_norm = alpha * support_norm
        install_average_gradient(trainable, support_gradient, query_gradient)
        clipped_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        if not bool(torch.isfinite(clipped_norm)):
            raise FloatingPointError("non-finite FOMLDG/ERM2 gradient")
        if any(parameter.grad is not None
               for _name, parameter in joint.frozen_named_parameters(model)):
            raise RuntimeError("frozen UQ parameter received a gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        trace = {
            "update": update,
            "source_aoa": row["source_aoa"],
            "query_aoa": row["query_aoa"],
            "support": support_values,
            "query": query_values,
            "support_gradient_norm": float(support_norm.detach()),
            "query_gradient_norm": float(query_norm.detach()),
            "inner_displacement_norm": float(inner_displacement_norm.detach()),
            "post_average_preclip_gradient_norm": float(clipped_norm.detach()),
            "alpha": alpha,
            "rng": row["rng"],
        }
        if first_mechanics is None:
            first_mechanics = dict(trace)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
        actual_rows.append({
            "update": update,
            "A": row["A"], "B": row["B"],
            "A_bootstrap_mask": row["A_bootstrap_mask"],
            "B_bootstrap_mask": row["B_bootstrap_mask"],
            "rng": row["rng"],
        })
        del history_a, target_a, history_b, target_b
        del fast_parameters, fast_buffers, fast_trainable
        del support_gradient, query_gradient
        if update % 20 == 0 or update == len(schedule["rows"]):
            print(json.dumps({
                "event": "m11_progress", "arm": arm, "update": update,
                "support_loss": support_values["loss"],
                "query_loss": query_values["loss"],
                "gradient_norm": float(clipped_norm.detach()),
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)

    final_buffer_sha = buffer_state_sha256(model)
    uq_end_sha = dr1.model_state_sha256(joint.uq_parameter_state(model))
    if final_buffer_sha != initial_buffer_sha or uq_end_sha != uq_start_sha:
        raise RuntimeError("functional training changed buffers or frozen UQ state")
    actual_stream_sha = canonical_sha256(actual_rows)
    expected_stream_sha = canonical_sha256([{
        "update": row["update"], "A": row["A"], "B": row["B"],
        "A_bootstrap_mask": row["A_bootstrap_mask"],
        "B_bootstrap_mask": row["B_bootstrap_mask"], "rng": row["rng"],
    } for row in schedule["rows"]])
    if actual_stream_sha != expected_stream_sha:
        raise RuntimeError("arm did not replay the frozen paired stream")
    training = {
        "arm": arm,
        "updates": len(actual_rows),
        "gradient_estimators_per_update": 2,
        "optimizer_steps": len(actual_rows),
        "alpha": INNER_LR if arm == "candidate" else 0.0,
        "beta": BETA,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "planned_stream_sha256": expected_stream_sha,
        "actual_stream_sha256": actual_stream_sha,
        "initial_model_state_sha256": initial_state_sha,
        "initial_buffer_state_sha256": initial_buffer_sha,
        "final_buffer_state_sha256": final_buffer_sha,
        "frozen_uq_start_sha256": uq_start_sha,
        "frozen_uq_end_sha256": uq_end_sha,
        "trainability": trainability,
        "first_update_mechanics": first_mechanics,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0),
    }
    endpoint_path = output_dir / f"{arm}_raw.pth"
    seal = save_endpoint(
        endpoint_path, model, arm=arm, start=start, normalizer=normalizer,
        schedule_sha256=str(schedule["source_schedule_sha256"]),
        training=training)
    access.seal(arm, seal)
    del optimizer, view, model, trainable, view_trainable
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return training, seal


def save_endpoint(path: Path, model: EnsembleUNet2d, *, arm: str,
                  start: Mapping[str, object], normalizer,
                  schedule_sha256: str, training: Mapping[str, object]) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    ensemble_train.common.set_model_stats(model, normalizer)
    state = OrderedDict(
        (name, value.detach().cpu()) for name, value in model.state_dict().items())
    state_sha = dr1.model_state_sha256(state)
    config = model.architecture_config()
    normalizer_values = joint.normalizer_payload(normalizer)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real-fomldg-screen",
        "arm": arm,
        "iteration": int(training["updates"]),
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": config,
        "model_config_sha256": canonical_sha256(config),
        "start": dict(start),
        "normalizer": normalizer_values,
        "normalizer_sha256": canonical_sha256(normalizer_values),
        "schedule_sha256": schedule_sha256,
        "training": dict(training),
        "held_opened_before_seal": False,
        "optimizer_state_saved": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "arm": arm,
        "path": str(path.resolve()),
        "file_sha256": sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": payload["model_config_sha256"],
        "normalizer_sha256": payload["normalizer_sha256"],
        "schedule_sha256": schedule_sha256,
        "updates": int(training["updates"]),
    }


def load_endpoint_model(seal: Mapping[str, object], normalizer,
                        device: torch.device) -> EnsembleUNet2d:
    path = Path(str(seal["path"]))
    if sha256(path) != seal["file_sha256"]:
        raise RuntimeError("endpoint file changed after sealing")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENDPOINT_SCHEMA:
        raise ValueError("endpoint schema changed")
    state = payload.get("model_state_dict")
    if (not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("endpoint model state changed")
    if canonical_sha256(payload["model_config"]) != seal["model_config_sha256"]:
        raise RuntimeError("endpoint model config changed")
    normalizer_values = joint.normalizer_payload(normalizer)
    if (payload.get("normalizer") != normalizer_values
            or payload.get("normalizer_sha256")
            != canonical_sha256(normalizer_values)):
        raise RuntimeError("endpoint Real normalizer changed")
    config = dict(payload["model_config"])
    if config.pop("architecture", None) != "ensemble_unet":
        raise ValueError("endpoint is not an EnsembleUNet2d")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
    model.load_state_dict(state, strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def validate_evaluation_manifest(path: Path, runtime_rows: Sequence[dict]
                                 ) -> list[dict]:
    if sha256(path) != EVALUATION_MANIFEST_SHA256:
        raise RuntimeError("clean716 evaluation manifest changed")
    rows = dense4k.load_evaluation_rows(path, runtime_rows)
    held = [row for row in rows if int(row["aoa"]) == HELD_AOA]
    identities = [{"file": row["file"], "starts": row["train_starts"]}
                  for row in held]
    if (len(held) != EXPECTED_HELD_TRAJECTORIES
            or sum(len(row["train_starts"]) for row in held)
            != EXPECTED_HELD_WINDOWS
            or canonical_sha256(identities) != HELD_IDENTITY_SHA256):
        raise RuntimeError("held AoA10 tail identity changed")
    return held


def load_held_tail(real_dir: Path, rows: Sequence[dict],
                   access: PairAccessAudit):
    trajectories = []
    windows = []
    groups: dict[str, list[int]] = {}
    root = real_dir.resolve()
    window_type = ensemble_train.common.repro.Window
    for row in rows:
        access.before_open(row, "held")
        filename = str(row["file"])
        path = (root / filename).resolve()
        if path.parent != root or path.name != filename or not path.is_file():
            raise FileNotFoundError(path)
        starts = [int(value) for value in row["train_starts"]]
        with h5py.File(path, "r") as handle:
            if (int(handle["u"].shape[0]) != int(row["frames"])
                    or handle["u"].shape != handle["v"].shape
                    or int(np.asarray(handle["re"])) != int(row["re"])
                    or int(np.asarray(handle["aoa"])) != HELD_AOA):
                raise RuntimeError(f"held metadata changed for {filename}")
            for start in starts:
                u = np.asarray(handle["u"][start:start + 40, ::2, ::2],
                               dtype=np.float32)
                v = np.asarray(handle["v"][start:start + 40, ::2, ::2],
                               dtype=np.float32)
                if u.shape != v.shape or u.shape[0] != 40:
                    raise RuntimeError("held tail window shape changed")
                value = torch.from_numpy(np.ascontiguousarray(
                    np.stack((u, v), axis=-1)))
                trajectory = len(trajectories)
                trajectories.append(value)
                windows.append(window_type(trajectory, 0))
                groups.setdefault(filename, []).append(len(windows) - 1)
        access.record_held(row, starts)
    if len(trajectories) != EXPECTED_HELD_WINDOWS or len(groups) != (
            EXPECTED_HELD_TRAJECTORIES):
        raise RuntimeError("held tail loader did not construct 18x9 windows")
    return ensemble_train.common.repro.WindowDataset(trajectories, windows), groups


def official_point(metrics: Mapping[str, float]) -> float:
    return float(sum(POINT_WEIGHTS[key] * float(metrics[key])
                     for key in POINT_WEIGHTS))


def add_official_point(metrics: Mapping[str, float]) -> dict[str, float]:
    result = {key: float(value) for key, value in metrics.items()}
    result["official_point_contribution_46_10_9"] = official_point(result)
    return result


def evaluate_endpoint(model: EnsembleUNet2d, dataset,
                      groups: Mapping[str, Sequence[int]], normalizer,
                      device: torch.device, deadline: float) -> dict:
    group_states = []
    per_trajectory = {}
    target_audits = {}
    for filename, indices in groups.items():
        metrics, state, target_audit = smu1.evaluate_model(
            model, dataset, indices, normalizer, device, deadline)
        group_states.append(state)
        per_trajectory[filename] = add_official_point(metrics)
        target_audits[filename] = target_audit
    pooled = add_official_point(
        smu1.finalize_metric_state(smu1.merge_metric_states(group_states)))
    return {
        "pooled": pooled,
        "per_trajectory": per_trajectory,
        "target_audits": target_audits,
        "target_audits_sha256": canonical_sha256(target_audits),
    }


def compare_evaluations(candidate: Mapping[str, object],
                        control: Mapping[str, object]) -> dict:
    if candidate["target_audits"] != control["target_audits"]:
        raise RuntimeError("candidate/control held targets differ")
    keys = (*POINT_WEIGHTS, "native_sps_score",
            "official_point_contribution_46_10_9")
    delta = {
        key: float(candidate["pooled"][key]) - float(control["pooled"][key])
        for key in keys
    }
    trajectory_delta = {
        filename: (
            float(candidate["per_trajectory"][filename][
                "official_point_contribution_46_10_9"])
            - float(control["per_trajectory"][filename][
                "official_point_contribution_46_10_9"]))
        for filename in candidate["per_trajectory"]
    }
    conditions = {
        "pooled_point_delta": (
            delta["official_point_contribution_46_10_9"]
            >= GATE["point_delta_min"]),
        "rel_floor": delta["rel_l2_score"] >= GATE["component_floor"],
        "tke_floor": delta["tke_rel_l2_score"] >= GATE["component_floor"],
        "mvpe_floor": delta["mvpe_rel_l2_score"] >= GATE["component_floor"],
        "positive_trajectories": sum(
            value > 0.0 for value in trajectory_delta.values())
            >= GATE["positive_trajectories_min"],
        "worst_trajectory": min(trajectory_delta.values())
            >= GATE["worst_trajectory_point_floor"],
    }
    passed = all(conditions.values())
    return {
        "candidate": candidate["pooled"],
        "control": control["pooled"],
        "delta": delta,
        "trajectory_point_delta": trajectory_delta,
        "positive_trajectories": sum(
            value > 0.0 for value in trajectory_delta.values()),
        "worst_trajectory_point_delta": min(trajectory_delta.values()),
        "mean_equal_trajectory_point_delta": float(np.mean(
            list(trajectory_delta.values()))),
        "gate": {
            "pass": passed,
            "conditions": conditions,
            "decision": (
                "GO_M11_SEEDS4_8" if passed else "STOP_M11_REAL_FOMLDG"),
        },
    }


def hardware_fingerprint(device: torch.device) -> dict[str, object]:
    result = {
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda":
        result.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                device).total_memory,
        })
    return result


def validate_seed0_authorization(path: Path,
                                 identity: Mapping[str, str]) -> dict[str, str]:
    result = json.loads(path.read_text(encoding="utf-8"))
    config = result.get("config", {})
    comparison = result.get("comparison", {})
    held = result.get("held", {})
    access = result.get("access", {})
    training = {
        arm: result.get(f"{arm}_training", {})
        for arm in ("candidate", "control")
    }
    seals = result.get("seals", {})
    candidate_target_sha = result.get("evaluations", {}).get(
        "candidate", {}).get("target_audits_sha256")
    control_target_sha = result.get("evaluations", {}).get(
        "control", {}).get("target_audits_sha256")
    expected_config = {
        "mode": "formal",
        "seed": SEED,
        "updates": FORMAL_UPDATES,
        "batch_size_per_role": FORMAL_BATCH_SIZE,
        "candidate_alpha": INNER_LR,
        "control_alpha": 0.0,
        "beta": BETA,
        "fit_rows_sha256": FIT_ROWS_SHA256,
        "schedule_sha256": FORMAL_SCHEDULE_SHA256[SEED],
    }
    conditions = {
        "schema": result.get("schema") == SCHEMA,
        "status": result.get("status") == "GO_M11_SEEDS4_8",
        "gate": bool(comparison.get("gate", {}).get("pass"))
            and comparison.get("gate", {}).get("decision")
            == "GO_M11_SEEDS4_8",
        "config": all(config.get(key) == value
                      for key, value in expected_config.items()),
        "commit": config.get("git", {}).get("commit") == identity["commit"],
        "start": config.get("start", {}).get("checkpoint_sha256")
            == START_SHA256,
        "runtime_manifest": config.get("runtime_manifest", {}).get(
            "semantic_sha256") == RUNTIME_MANIFEST_SEMANTIC_SHA256,
        "held": held.get("aoa") == HELD_AOA
            and held.get("trajectories") == EXPECTED_HELD_TRAJECTORIES
            and held.get("windows") == EXPECTED_HELD_WINDOWS
            and held.get("identity_sha256") == HELD_IDENTITY_SHA256,
        "access": bool(access.get("fit_allowlist_exact"))
            and bool(access.get("held_allowlist_exact"))
            and bool(access.get("all_endpoints_sealed"))
            and bool(access.get("held_first_open_after_both_seals"))
            and access.get("held_windows_opened") == EXPECTED_HELD_WINDOWS,
        "arms": set(training) == {"candidate", "control"}
            and all(value.get("updates") == FORMAL_UPDATES
                    and value.get("optimizer_steps") == FORMAL_UPDATES
                    and value.get("gradient_estimators_per_update") == 2
                    and value.get("planned_stream_sha256")
                    == value.get("actual_stream_sha256")
                    for value in training.values()),
        "paired_initial_state": training["candidate"].get(
            "initial_model_state_sha256") == training["control"].get(
                "initial_model_state_sha256"),
        "paired_stream": training["candidate"].get(
            "actual_stream_sha256") == training["control"].get(
                "actual_stream_sha256"),
        "seals": set(seals) == {"candidate", "control"}
            and access.get("endpoint_seals") == seals
            and all(seal.get("normalizer_sha256")
                    == config.get("normalizer_sha256")
                    for seal in seals.values()),
        "targets": isinstance(candidate_target_sha, str)
            and len(candidate_target_sha) == 64
            and candidate_target_sha == control_target_sha,
    }
    if not all(conditions.values()):
        failed = [key for key, value in conditions.items() if not value]
        raise RuntimeError(f"invalid seed0 GO authorization: {failed}")
    for arm, seal in seals.items():
        endpoint = Path(str(seal.get("path", "")))
        if (seal.get("arm") != arm or not endpoint.is_file()
                or sha256(endpoint) != seal.get("file_sha256")):
            raise RuntimeError(f"seed0 {arm} endpoint seal is no longer valid")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256(path),
        "status": str(result["status"]),
        "validated_seed0_seals_sha256": canonical_sha256(seals),
    }


def run(args: argparse.Namespace) -> dict:
    identity = git_identity()
    if args.mode == "formal" and identity["status_porcelain"]:
        raise RuntimeError("formal M11 requires a clean Git worktree")
    if args.mode in {"formal", "smoke"}:
        if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
            raise RuntimeError("M11 execution requires explicit acknowledgement")
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
            raise RuntimeError("M11 execution requires CUDA")
    if sha256(args.runtime_manifest) != RUNTIME_MANIFEST_FILE_SHA256:
        raise RuntimeError("runtime manifest file changed")
    runtime_rows, runtime_report = dr1.load_manifest(
        args.runtime_manifest, RUNTIME_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in runtime_rows if int(row["aoa"]) in FIT_AOAS]
    held_rows = [row for row in runtime_rows if int(row["aoa"]) == HELD_AOA]
    if canonical_sha256(fit_rows) != FIT_ROWS_SHA256:
        raise RuntimeError("fit row identity changed")
    schedule = build_cross_condition_schedule(
        runtime_rows, args.updates, args.batch_size, args.seed)
    if args.mode == "contract":
        return {
            "schema": SCHEMA,
            "mode": "contract",
            "runtime_manifest": runtime_report,
            "schedule": {key: value for key, value in schedule.items()
                         if key != "rows"},
        }
    if args.mode == "formal":
        if args.updates != FORMAL_UPDATES:
            raise ValueError("formal M11 requires exactly 1200 updates")
        if args.batch_size != FORMAL_BATCH_SIZE:
            raise ValueError("batch4 smoke passed; formal M11 requires batch 4")
        if args.seed not in FORMAL_SEEDS:
            raise ValueError("formal M11 seed must be one of 0, 4, 8")
        if schedule["sha256"] != FORMAL_SCHEDULE_SHA256[args.seed]:
            raise RuntimeError("formal M11 schedule digest changed")
    elif (args.updates != len(ORDERED_AOA_PAIRS)
          or args.batch_size not in {1, FORMAL_BATCH_SIZE}):
        raise ValueError("smoke mode requires 12 updates and batch size 1 or 4")

    seed_authorization = None
    if args.mode == "formal" and args.seed != SEED:
        if args.seed0_result is None or not args.seed0_result.is_file():
            raise RuntimeError("seed4/8 requires the sealed seed0 GO result")
        seed_authorization = validate_seed0_authorization(
            args.seed0_result, identity)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True)
    write_json_atomic(args.output_dir / "schedule.json", schedule)
    access = PairAccessAudit(
        [row["file"] for row in fit_rows],
        [row["file"] for row in held_rows])
    trajectories, metadata, windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", access)
    if len(trajectories) != 63 or len(windows) != 1_968:
        raise RuntimeError("fit prefix count changed")
    statistics = joint.fit_only_real_statistics(
        trajectories, metadata, windows, HELD_AOA,
        allowed_files=[row["file"] for row in fit_rows])
    normalizer = smu1.FoldNormalizer(statistics, device)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "real_normalizer.pt")
    dataset = ensemble_train.common.repro.WindowDataset(trajectories, windows)
    resolved = resolve_schedule_indices(schedule, windows, metadata)
    for original, row in zip(schedule["rows"], resolved["rows"]):
        original.update({
            "A_indices": row["A_indices"],
            "B_indices": row["B_indices"],
        })
    schedule["resolved_indices_sha256"] = resolved["sha256"]
    schedule["source_schedule_sha256"] = schedule["sha256"]
    write_json_atomic(args.output_dir / "resolved_schedule.json", schedule)
    start = joint.inspect_start_checkpoint(args.start_checkpoint, START_SHA256)
    write_json_atomic(args.output_dir / "bootstrap_seal.json", start)
    alpha_zero_preflight = alpha_zero_equivalence_preflight(
        start, dataset, schedule, normalizer, device)
    normalizer_values = joint.normalizer_payload(normalizer)
    config = {
        "schema": SCHEMA,
        "mode": args.mode,
        "git": identity,
        "hardware": hardware_fingerprint(device),
        "seed": args.seed,
        "updates": args.updates,
        "batch_size_per_role": args.batch_size,
        "optimizer": {"name": "AdamW", "lr": LR,
                      "weight_decay": WEIGHT_DECAY,
                      "grad_clip": GRAD_CLIP},
        "candidate_alpha": INNER_LR,
        "control_alpha": 0.0,
        "beta": BETA,
        "loss": joint.DOMAIN_LOSS["real"],
        "augmentation": dense4k.AUGMENTATION,
        "runtime_manifest": runtime_report,
        "fit_rows_sha256": FIT_ROWS_SHA256,
        "schedule_sha256": schedule["sha256"],
        "resolved_indices_sha256": resolved["sha256"],
        "normalizer": normalizer_values,
        "normalizer_sha256": canonical_sha256(normalizer_values),
        "start": start,
        "alpha_zero_preflight": alpha_zero_preflight,
        "arm_order": ["candidate", "control"],
        "seed0_authorization": seed_authorization,
        "held_manifest_not_parsed_before_dual_seal": True,
    }
    write_json_atomic(args.output_dir / "config.json", config)
    deadline = time.perf_counter() + args.wall_budget_seconds
    candidate_training, candidate_seal = train_arm(
        "candidate", start, dataset, schedule, normalizer, device,
        args.output_dir, access, deadline)
    control_training, control_seal = train_arm(
        "control", start, dataset, schedule, normalizer, device,
        args.output_dir, access, deadline)
    if (candidate_training["initial_model_state_sha256"]
            != control_training["initial_model_state_sha256"]
            or candidate_training["actual_stream_sha256"]
            != control_training["actual_stream_sha256"]):
        raise RuntimeError("paired arms changed initial state or sample/RNG stream")
    seals = {"candidate": candidate_seal, "control": control_seal}
    write_json_atomic(args.output_dir / "all_endpoints_seal.json", {
        "schema": "b210_real_fomldg_dual_raw_seal_v1",
        "seals": seals,
        "seals_sha256": canonical_sha256(seals),
        "held_opened_before_all_seals": False,
    })
    if args.mode == "smoke":
        result = {
            "schema": SCHEMA,
            "status": "MECHANICAL_SMOKE_COMPLETE_NO_HELD_OPEN",
            "config": config,
            "candidate_training": candidate_training,
            "control_training": control_training,
            "seals": seals,
            "access": access.report(),
        }
        write_json_atomic(args.output_dir / "result.json", result)
        return result

    # This is deliberately the first semantic access to val_starts.
    held_evaluation_rows = validate_evaluation_manifest(
        args.evaluation_manifest, runtime_rows)
    held_dataset, held_groups = load_held_tail(
        args.real_dir, held_evaluation_rows, access)
    evaluations = {}
    for arm in ("candidate", "control"):
        model = load_endpoint_model(seals[arm], normalizer, device)
        evaluations[arm] = evaluate_endpoint(
            model, held_dataset, held_groups, normalizer, device, deadline)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    comparison = compare_evaluations(
        evaluations["candidate"], evaluations["control"])
    access_report = access.report()
    if (not access_report["fit_allowlist_exact"]
            or not access_report["held_allowlist_exact"]
            or not access_report["held_first_open_after_both_seals"]
            or access_report["held_windows_opened"] != EXPECTED_HELD_WINDOWS):
        raise RuntimeError("M11 access contract failed")
    result = {
        "schema": SCHEMA,
        "status": comparison["gate"]["decision"],
        "evidence_class": (
            "AoA10 architecture-selection tail screen; 18 trajectory clusters; "
            "not independent confirmation"),
        "config": config,
        "candidate_training": candidate_training,
        "control_training": control_training,
        "seals": seals,
        "held": {
            "aoa": HELD_AOA,
            "trajectories": EXPECTED_HELD_TRAJECTORIES,
            "windows": EXPECTED_HELD_WINDOWS,
            "identity_sha256": HELD_IDENTITY_SHA256,
        },
        "evaluations": evaluations,
        "comparison": comparison,
        "access": access_report,
    }
    write_json_atomic(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "smoke", "formal"),
                        default="contract")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-checkpoint", type=Path, default=DEFAULT_START)
    parser.add_argument("--runtime-manifest", type=Path,
                        default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--evaluation-manifest", type=Path,
                        default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--real-dir", type=Path,
                        default=Path("/root/autodl-tmp/realpde/data/real"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=FORMAL_UPDATES)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seed0-result", type=Path)
    parser.add_argument("--wall-budget-seconds", type=float, default=86_400.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode != "contract" and args.output_dir is None:
        raise ValueError("execution modes require --output-dir")
    result = run(args)
    print(json.dumps({
        "event": "m11_complete",
        "mode": args.mode,
        "status": result.get("status", "CONTRACT_VALID"),
        "schedule_sha256": (
            result.get("config", {}).get("schedule_sha256")
            or result.get("schedule", {}).get("sha256")),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
