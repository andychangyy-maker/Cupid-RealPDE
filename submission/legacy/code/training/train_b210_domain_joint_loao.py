#!/usr/bin/env python3
"""Run the Stage-A pilot or Stage-B quick B210 domain-order experiment.

Both arms consume the same frozen Sim and fit-only Real event streams.  D0
executes all Sim events before all Real events; D1 interleaves the same events.
Every event owns one optimizer step.  The held Real condition is never opened.
Stage B evaluates only a trajectory-held inner validation split, after every
raw/EMA endpoint (including the Real-only absolute reference) is sealed.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for _path in (CODE_ROOT, TRAINING_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import domain_joint_events as domain_events  # noqa: E402
import train_b210_scratch_real_dense_aug as dense4k  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


EXPERIMENT = "b210_domain_joint_loao"
SCHEMA = "b210_domain_joint_stage_a_pilot_v1"
QUICK_SCHEMA = "b210_domain_joint_stage_b_quick_v1"
ENDPOINT_SCHEMA = "b210_domain_joint_point_endpoint_v1"
ARMS = domain_events.ARMS
DOMAINS = domain_events.DOMAINS
AOA_VALUES = (0, 5, 10, 15, 20)
REFERENCE_ARM = "dminus1_real_only"
QUICK_RUNS = (*ARMS, REFERENCE_ARM)
ENDPOINT_KINDS = ("raw", "ema")

PILOT_HELD_AOA = 10
PILOT_DOMAIN_UPDATES = 200
QUICK_HELD_AOA = 10
QUICK_DOMAIN_UPDATES = 1_000
FORMAL_DOMAIN_UPDATES = 4_000
BATCH_SIZE = 8
SEED = 0
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
EVAL_BATCH_SIZE = 16
EXPECTED_B210_PARAMETERS = 178_705_575

INNER_SPLIT_KEY = "TTDO-P2-stageB-inner-v1"
POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_score": 0.10,
    "mvpe_score": 0.09,
}
POINT_METRIC_KEYS = (*POINT_WEIGHTS, "point_final_contribution_46_10_9")
QUICK_GATE = {
    "raw_d1_minus_d0_point_min": 0.03,
    "raw_d1_minus_d0_component_floor": -0.20,
    "raw_d1_minus_dminus1_point_min": 0.0,
}

SIM_ACCURACY_WEIGHTS = (1.0, 4.0, 1.0)
REAL_ACCURACY_WEIGHTS = (5.0, 3.0, 2.0)
MSE_WEIGHT = 0.25
# Preserve the two domains' historical strong point/member mixtures.  The
# Real pure-MSE warmup is deliberately zero: R1 already uses accuracy 5/3/2.
DOMAIN_LOSS = {
    "sim": {"point_weight": 1.0, "member_weight": 0.25,
            "member_mode": "mse", "accuracy_weights": SIM_ACCURACY_WEIGHTS},
    "real": {"point_weight": 0.25, "member_weight": 1.0,
             "member_mode": "official_accuracy",
             "accuracy_weights": REAL_ACCURACY_WEIGHTS},
}

DEFAULT_SIM_MANIFEST = Path(
    "/root/autodl-tmp/realpde/runs/b210_retrain_20260826/"
    "sim10k_repro/split_manifest.json")
DEFAULT_SIM_MANIFEST_FILE_SHA256 = (
    "01c9ed5694832e0a486dd483711f7621a1078e38755de8f1edc8cbec0e9ad38e")
DEFAULT_SIM_MANIFEST_SEMANTIC_SHA256 = (
    "8b1650f3de3b3b92b8555335bc4de0a4b9ff4828613a2539d6591966cf99bd36")
DEFAULT_REAL_MANIFEST = Path(
    "/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/"
    "runtime_train_manifest.json")
DEFAULT_REAL_MANIFEST_FILE_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")

UQ_ONLY_MODULES = (
    "uq_fluctuation_role_gates", "uq_projections", "uq_fusion",
    "quantile_head", "utility_refinement_head", "output_space_uq_allocator",
    "output_space_uq_budgeter", "temporal_uq_refiner",
    "coverage_calibration_head", "residual_mixture_head",
)

SCORING_PATH = (
    REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py")


def load_official_scoring():
    """Load the released v9 scorer instead of maintaining a metric copy."""
    spec = importlib.util.spec_from_file_location(
        "realpde_track1_v9_domain_joint_scoring", SCORING_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official scorer from {SCORING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OFFICIAL = None


def official_scoring():
    """Lazily load scoring so the legacy Stage-A path gains no dependency."""
    global _OFFICIAL
    if _OFFICIAL is None:
        _OFFICIAL = load_official_scoring()
    return _OFFICIAL


def canonical_sha256(value: object) -> str:
    return domain_events.canonical_sha256(value)


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


def build_domain_event_orders(updates: int) -> dict[str, dict]:
    """Adapt the committed framework-neutral event contract for training."""
    contract = domain_events.build_paired_event_contract(updates, SEED)
    validation = domain_events.validate_paired_event_contract(contract)
    result: dict[str, dict] = {}
    for arm in ARMS:
        events = contract["arms"][arm]["events"]
        rows = [{
            **dict(event),
            "optimizer_step": optimizer_step,
            "domain_step": int(event["exposure_index"]) + 1,
        } for optimizer_step, event in enumerate(events, start=1)]
        result[arm] = {
            "rows": rows,
            "sha256": contract["arms"][arm]["order_sha256"],
            "optimizer_updates": len(rows),
            "domain_exposures": {domain: updates for domain in DOMAINS},
            "event_contract_sha256": contract["contract_sha256"],
            "event_contract_validation": validation,
        }
    validate_domain_event_orders(result, updates)
    return result


def build_quick_event_orders(updates: int) -> dict[str, dict]:
    """Add a same-start Real-only absolute reference to the causal pair.

    D-minus-1 deliberately receives only N optimizer steps, so it is not an
    equal-compute causal arm.  Its Real exposures, RNG keys, and sample stream
    are nevertheless identical to those in D0 and D1.
    """
    result = build_domain_event_orders(updates)
    real_events = [
        {**row, "optimizer_step": optimizer_step}
        for optimizer_step, row in enumerate(
            (row for row in result[ARMS[0]]["rows"]
             if row["domain"] == "real"), start=1)
    ]
    result[REFERENCE_ARM] = {
        "rows": real_events,
        "sha256": canonical_sha256(real_events),
        "optimizer_updates": updates,
        "domain_exposures": {"sim": 0, "real": updates},
        "event_contract_sha256": result[ARMS[0]]["event_contract_sha256"],
        "reference_only": True,
        "equal_compute_causal_arm": False,
    }
    if [int(row["domain_step"]) for row in real_events] != list(
            range(1, updates + 1)):
        raise RuntimeError("Real-only reference changed the Real exposure order")
    if any(row["domain"] != "real" for row in real_events):
        raise RuntimeError("Real-only reference contains a Sim event")
    return result


def _split_rank(aoa: int, filename: str) -> str:
    payload = f"{INNER_SPLIT_KEY}|{aoa}|{filename}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_stratified_trajectory_split(
        rows: Sequence[Mapping[str, object]], held_aoa: int,
        ) -> dict[str, object]:
    """Freeze the Stage-B trajectory split before any H5 file is opened."""
    if held_aoa not in AOA_VALUES:
        raise ValueError("held AoA is outside the released conditions")
    copied = [dict(row) for row in rows]
    filenames = [str(row.get("file", "")) for row in copied]
    if not copied or any(not name for name in filenames):
        raise ValueError("trajectory split requires non-empty filename rows")
    if len(set(filenames)) != len(filenames):
        raise ValueError("trajectory filenames must be unique")
    unknown = sorted(set(int(row["aoa"]) for row in copied) - set(AOA_VALUES))
    if unknown:
        raise ValueError(f"unexpected AoA values in trajectory split: {unknown}")

    held_rows = [row for row in copied if int(row["aoa"]) == held_aoa]
    if not held_rows:
        raise RuntimeError("trajectory split has no held-AoA rows")
    train_membership: set[str] = set()
    val_membership: set[str] = set()
    selection_sha256: dict[str, str] = {}
    by_aoa: dict[str, dict[str, object]] = {}

    def partition_summary(partition: Sequence[Mapping[str, object]]) -> dict:
        re_values = [int(row["re"]) for row in partition]
        return {
            "trajectories": len(partition),
            "sparse_windows": sum(
                len(row.get("train_starts", ())) for row in partition),
            "re_min": min(re_values),
            "re_max": max(re_values),
            "re_mean": float(sum(re_values) / len(re_values)),
        }

    for aoa in sorted(value for value in AOA_VALUES if value != held_aoa):
        group = [row for row in copied if int(row["aoa"]) == aoa]
        if not group:
            raise RuntimeError(f"trajectory split has no fit rows for AoA {aoa}")
        ranked = []
        for row in group:
            filename = str(row["file"])
            digest = _split_rank(aoa, filename)
            selection_sha256[filename] = digest
            ranked.append((digest, filename, row))
        ranked.sort(key=lambda item: (item[0], item[1]))
        n_val = math.ceil(len(ranked) / 5)
        aoa_val = [dict(item[2]) for item in ranked[:n_val]]
        aoa_train = [dict(item[2]) for item in ranked[n_val:]]
        if not aoa_val or not aoa_train:
            raise RuntimeError(
                f"AoA {aoa} cannot supply both inner-train and inner-val")
        val_membership.update(str(row["file"]) for row in aoa_val)
        train_membership.update(str(row["file"]) for row in aoa_train)
        by_aoa[str(aoa)] = {
            "trajectories": len(ranked),
            "inner_train_trajectories": len(aoa_train),
            "inner_val_trajectories": len(aoa_val),
            "inner_train_summary": partition_summary(aoa_train),
            "inner_val_summary": partition_summary(aoa_val),
            "ranked_files": [item[1] for item in ranked],
            "ranked_selection_sha256": [item[0] for item in ranked],
        }

    # Hash rank freezes membership only.  Preserve the source-manifest order so
    # trajectory IDs, bootstrap streams, and evaluation order do not change as
    # a side effect of the split implementation.
    train_rows = [row for row in copied if str(row["file"]) in train_membership]
    val_rows = [row for row in copied if str(row["file"]) in val_membership]
    train_files = {str(row["file"]) for row in train_rows}
    val_files = {str(row["file"]) for row in val_rows}
    held_files = {str(row["file"]) for row in held_rows}
    if train_files & val_files or (train_files | val_files) & held_files:
        raise RuntimeError("trajectory split partitions overlap")
    if train_files | val_files | held_files != set(filenames):
        raise RuntimeError("trajectory split did not partition the manifest")

    report: dict[str, object] = {
        "schema": "b210_domain_joint_stage_b_inner_split_v1",
        "selection_key": INNER_SPLIT_KEY,
        "selection_formula": (
            'SHA256("TTDO-P2-stageB-inner-v1|<aoa>|<filename>")'),
        "held_aoa": held_aoa,
        "source_rows": copied,
        "source_rows_sha256": canonical_sha256(copied),
        "inner_train_rows": train_rows,
        "inner_train_rows_sha256": canonical_sha256(train_rows),
        "inner_val_rows": val_rows,
        "inner_val_rows_sha256": canonical_sha256(val_rows),
        "held_rows": held_rows,
        "held_rows_sha256": canonical_sha256(held_rows),
        "row_selection_sha256": dict(sorted(selection_sha256.items())),
        "by_aoa": by_aoa,
        "partitions_disjoint": True,
        "every_fit_aoa_has_inner_val": True,
    }
    report["split_sha256"] = canonical_sha256(report)
    return report


def validate_domain_event_orders(orders: Mapping[str, Mapping], updates: int
                                 ) -> None:
    if set(orders) != set(ARMS):
        raise RuntimeError("domain-order arms changed")
    expected_stream = {
        domain: list(range(1, updates + 1)) for domain in DOMAINS}
    for arm in ARMS:
        rows = list(orders[arm]["rows"])
        if len(rows) != 2 * updates:
            raise RuntimeError(f"{arm} does not contain 2N optimizer events")
        for domain in DOMAINS:
            observed = [
                int(row["domain_step"]) for row in rows
                if row["domain"] == domain]
            if observed != expected_stream[domain]:
                raise RuntimeError(f"{arm} changed the {domain} event stream")
    d0 = [(row["domain"], row["domain_step"])
          for row in orders[ARMS[0]]["rows"]]
    d1 = [(row["domain"], row["domain_step"])
          for row in orders[ARMS[1]]["rows"]]
    if d0 == d1:
        raise RuntimeError("blocked and interleaved orders are identical")
    if sorted(d0) != sorted(d1):
        raise RuntimeError("arms do not have the same domain-event multiset")


def domain_learning_rate(domain: str, domain_step: int, updates: int,
                         base_lr: float = LR) -> float:
    """Frozen Stage-A domain-local schedule (constant in the first gate)."""
    if domain not in DOMAINS or not 1 <= domain_step <= updates:
        raise ValueError("invalid domain-local learning-rate coordinate")
    if not math.isfinite(base_lr) or base_lr <= 0.0:
        raise ValueError("base learning rate must be finite and positive")
    return float(base_lr)


def domain_event_seed(domain: str, domain_step: int, base_seed: int = SEED
                      ) -> int:
    if domain_step <= 0:
        raise ValueError("invalid domain RNG coordinate")
    return domain_events.derive_event_seed(
        base_seed, domain, domain_step - 1)


def rng_state_fingerprint(device: torch.device) -> str:
    digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if device.type == "cuda":
        digest.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def domain_event_rng(domain: str, domain_step: int, device: torch.device,
                     base_seed: int = SEED) -> Iterator[dict[str, object]]:
    """Replay an arm-independent RNG state and restore every host state."""
    seed = domain_event_seed(domain, domain_step, base_seed)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None
                   else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.set_rng_state(
                torch.Generator(device="cpu").manual_seed(seed).get_state())
            if device.type == "cuda":
                generator = torch.Generator(device=device).manual_seed(seed)
                torch.cuda.set_rng_state(generator.get_state(), device)
            yield {
                "seed": seed,
                "fingerprint": rng_state_fingerprint(device),
                "contract_fingerprint": domain_events.event_rng_fingerprint(
                    base_seed, domain, domain_step - 1),
            }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class HeldTargetGate:
    """Allow fit-prefix opens now and held-prefix opens only after both seals."""

    def __init__(self, held_aoa: int, arms: Sequence[str] = ARMS) -> None:
        if held_aoa not in AOA_VALUES:
            raise ValueError("held AoA is outside the released conditions")
        self.held_aoa = held_aoa
        self.arms = tuple(arms)
        self.seals: dict[str, dict] = {}
        self.fit_opened: list[str] = []
        self.held_opened: list[str] = []
        self._seals_verified = False

    @property
    def all_sealed(self) -> bool:
        return set(self.seals) == set(self.arms)

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        aoa = int(row["aoa"])
        filename = str(row["file"])
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded duplicate trajectory reached H5 open")
        if phase == "fit":
            if aoa == self.held_aoa or self.seals:
                raise RuntimeError("held/late trajectory reached fit H5 open")
            if filename in self.fit_opened:
                raise RuntimeError("fit trajectory opened twice")
            return
        if phase != "held" or aoa != self.held_aoa:
            raise RuntimeError("invalid held-prefix access")
        if not self.all_sealed:
            raise RuntimeError("held target opened before both endpoints sealed")
        if not self._seals_verified:
            for seal in self.seals.values():
                path = Path(str(seal["path"]))
                if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
                    raise RuntimeError("sealed endpoint changed before held open")
            self._seals_verified = True
        if filename in self.held_opened:
            raise RuntimeError("held trajectory opened twice")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        target = self.fit_opened if phase == "fit" else self.held_opened
        if filename in target:
            raise RuntimeError("trajectory access record duplicated")
        target.append(filename)

    def seal(self, arm: str, seal: Mapping[str, object]) -> None:
        if arm not in self.arms or arm in self.seals:
            raise RuntimeError(f"invalid or duplicate endpoint seal: {arm}")
        path = Path(str(seal["path"]))
        if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
            raise RuntimeError("cannot register an unverified endpoint")
        self.seals[arm] = dict(seal)

    def report(self) -> dict:
        return {
            "held_aoa": self.held_aoa,
            "fit_opened": list(self.fit_opened),
            "held_opened_after_all_seals": list(self.held_opened),
            "sealed_arms": sorted(self.seals),
            "held_target_opened_before_all_seals": False,
            "all_endpoints_sealed": self.all_sealed,
            "temporal_tail_constructed_or_read": False,
        }


class QuickAccessGate:
    """Fail-closed Stage-B access ledger for train/validation prefixes.

    The held AoA is forbidden for the complete quick experiment.  Inner-val
    H5 files may open only after raw and EMA endpoints for all three runs have
    been written and their file hashes verified.
    """

    def __init__(self, held_aoa: int, inner_train_rows: Sequence[Mapping],
                 inner_val_rows: Sequence[Mapping],
                 runs: Sequence[str] = QUICK_RUNS) -> None:
        if held_aoa not in AOA_VALUES:
            raise ValueError("held AoA is outside the released conditions")
        self.held_aoa = held_aoa
        self.runs = tuple(runs)
        if len(set(self.runs)) != len(self.runs):
            raise ValueError("quick-gate run names must be unique")
        self.expected_endpoint_keys = {
            f"{run}:{kind}" for run in self.runs for kind in ENDPOINT_KINDS}
        self.allowed = {
            "inner_train": {str(row["file"]) for row in inner_train_rows},
            "inner_val": {str(row["file"]) for row in inner_val_rows},
        }
        if (not all(self.allowed.values())
                or self.allowed["inner_train"] & self.allowed["inner_val"]):
            raise ValueError("quick-gate allowlists are empty or overlap")
        for row in (*inner_train_rows, *inner_val_rows):
            if int(row["aoa"]) == held_aoa:
                raise ValueError("held AoA entered a quick-gate allowlist")
        self.opened = {phase: [] for phase in self.allowed}
        self.seals: dict[str, dict] = {}
        self.evaluations: list[str] = []
        self.events: list[dict[str, object]] = []
        self._authorized_opens: set[tuple[str, str]] = set()
        self._seals_verified = False

    @property
    def all_sealed(self) -> bool:
        return set(self.seals) == self.expected_endpoint_keys

    def _verify_all_seals(self) -> None:
        if not self.all_sealed:
            raise RuntimeError("inner-val opened before every raw/EMA endpoint sealed")
        if self._seals_verified:
            return
        for seal in self.seals.values():
            path = Path(str(seal["path"]))
            if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
                raise RuntimeError("sealed endpoint changed before inner-val open")
        self._seals_verified = True

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        aoa = int(row["aoa"])
        filename = str(row["file"])
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded duplicate trajectory reached H5 open")
        if aoa == self.held_aoa or phase == "held":
            raise RuntimeError("held target is forbidden throughout Stage-B quick")
        if phase not in self.allowed or filename not in self.allowed[phase]:
            raise RuntimeError("trajectory is outside the phase access allowlist")
        if filename in self.opened[phase]:
            raise RuntimeError("trajectory opened twice in one quick phase")
        if phase == "inner_train":
            if self.seals:
                raise RuntimeError("inner-train trajectory opened after endpoint seal")
        else:
            self._verify_all_seals()
        key = (phase, filename)
        if key in self._authorized_opens:
            raise RuntimeError("trajectory open is already pending")
        self._authorized_opens.add(key)

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        key = (phase, filename)
        if (phase not in self.opened or filename in self.opened[phase]
                or key not in self._authorized_opens):
            raise RuntimeError("invalid or duplicate quick access record")
        self._authorized_opens.remove(key)
        self.opened[phase].append(filename)
        self.events.append({
            "phase": phase,
            "file": filename,
            "aoa": int(row["aoa"]),
            "max_read_frame_exclusive": int(row["val_boundary"]),
            "all_endpoints_sealed_before_open": self.all_sealed,
        })

    def seal(self, run: str, kind: str, seal: Mapping[str, object]) -> None:
        key = f"{run}:{kind}"
        if key not in self.expected_endpoint_keys or key in self.seals:
            raise RuntimeError(f"invalid or duplicate endpoint seal: {key}")
        if seal.get("arm") != run or seal.get("endpoint_kind") != kind:
            raise RuntimeError("endpoint seal identity differs from quick-gate key")
        path = Path(str(seal["path"]))
        if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
            raise RuntimeError("cannot register an unverified quick endpoint")
        self.seals[key] = dict(seal)

    def record_evaluation(self, run: str, kind: str) -> None:
        self._verify_all_seals()
        if set(self.opened["inner_val"]) != self.allowed["inner_val"]:
            raise RuntimeError("cannot evaluate before every inner-val file opened")
        key = f"{run}:{kind}"
        if key not in self.expected_endpoint_keys or key in self.evaluations:
            raise RuntimeError("invalid or duplicate quick endpoint evaluation")
        self.evaluations.append(key)

    def report(self) -> dict[str, object]:
        opened = {key: list(value) for key, value in self.opened.items()}
        result: dict[str, object] = {
            "schema": "b210_domain_joint_stage_b_access_v1",
            "held_aoa": self.held_aoa,
            "held_target_opened": False,
            "allowed_file_allowlists": {
                key: sorted(value) for key, value in self.allowed.items()},
            "opened_files": opened,
            "opened_exact": {
                key: set(opened[key]) == self.allowed[key] for key in self.allowed},
            "endpoint_seals": dict(self.seals),
            "expected_endpoint_keys": sorted(self.expected_endpoint_keys),
            "all_endpoints_sealed": self.all_sealed,
            "evaluation_order": list(self.evaluations),
            "events": list(self.events),
            "pending_authorized_opens": [list(value) for value in sorted(
                self._authorized_opens)],
            "temporal_tail_constructed_or_read": False,
        }
        result["allowed_rows_sha256"] = canonical_sha256(
            result["allowed_file_allowlists"])
        result["opened_rows_sha256"] = canonical_sha256(opened)
        result["access_events_sha256"] = canonical_sha256(self.events)
        result["endpoint_seals_sha256"] = canonical_sha256(self.seals)
        return result


def normalizer_payload(normalizer) -> dict[str, list[float]]:
    return ensemble_train.common.normalizer_dict(normalizer)


def fit_only_real_statistics(trajectories: Sequence[torch.Tensor],
                             metadata: Sequence[Mapping[str, object]],
                             sparse_windows: Sequence, held_aoa: int,
                             allowed_files: Sequence[str] | None = None,
                             ) -> tuple[torch.Tensor, ...]:
    if not metadata or any(int(row["aoa"]) == held_aoa for row in metadata):
        raise RuntimeError("held condition entered Real normalizer inputs")
    if allowed_files is not None:
        expected = sorted(str(value) for value in allowed_files)
        observed = sorted(str(row.get("file", "")) for row in metadata)
        if (not expected or len(expected) != len(set(expected))
                or observed != expected):
            raise RuntimeError(
                "Real normalizer inputs differ from inner-train allowlist")
    sufficient = smu1.window_sufficient_statistics(trajectories, sparse_windows)
    return smu1.fold_statistics(sufficient, range(len(sparse_windows)))


def _load_checkpoint(path: Path) -> dict:
    try:
        # torch 2.1 accepts mmap only when ``f`` is a string filename.
        return torch.load(
            str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(str(path), map_location="cpu", weights_only=False)


def inspect_start_checkpoint(path: Path, expected_sha256: str,
                             expected_iteration: int = 10_000) -> dict:
    expected = dr1.normalized_sha256(expected_sha256, "start checkpoint SHA")
    actual = dr1.sha256(path)
    if actual != expected:
        raise RuntimeError("start Sim checkpoint SHA-256 changed")
    payload = _load_checkpoint(path)
    if (not isinstance(payload, Mapping) or payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != expected_iteration):
        raise ValueError("start checkpoint is not the requested terminal Sim state")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    stored_normalizer = payload.get("normalizer")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("start checkpoint lacks model config/state")
    parsed = dict(config)
    if parsed.pop("architecture", None) != "ensemble_unet":
        raise ValueError("start checkpoint is not an ensemble U-Net")
    model_config = EnsembleUNet2dConfig(**parsed)
    model_config.validate()
    if model_config.base != 210:
        raise ValueError("P2a requires a B210 checkpoint")
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in state.values() if isinstance(value, torch.Tensor)):
        raise TypeError("start checkpoint floating state must be FP32")
    keys = ("mean_input", "mean_target", "std_input", "std_target")
    if not isinstance(stored_normalizer, Mapping) or any(
            key not in stored_normalizer for key in keys):
        raise ValueError("start checkpoint lacks its Sim normalizer")
    values = {
        key: torch.as_tensor(stored_normalizer[key], dtype=torch.float32)
        for key in keys}
    if any(value.shape != (3,) or not bool(torch.isfinite(value).all())
           for value in values.values()):
        raise ValueError("start Sim normalizer is invalid")
    if bool((values["std_input"] <= 0).any()) or bool(
            (values["std_target"] <= 0).any()):
        raise ValueError("start Sim normalizer has non-positive std")
    report = {
        "path": str(path.resolve()),
        "checkpoint_sha256": actual,
        "checkpoint_model_config_sha256": dr1.canonical_sha256(config),
        "checkpoint_model_state_sha256": dr1.model_state_sha256(state),
        "phase": "sim",
        "iteration": expected_iteration,
        "metrics_status": payload.get("metrics", {}).get("status"),
        "model_config": dict(config),
        "normalizer": {key: value.tolist() for key, value in values.items()},
        "normalizer_sha256": canonical_sha256(
            {key: value.tolist() for key, value in values.items()}),
    }
    del payload, state
    gc.collect()
    return report


def build_start_model(start: Mapping[str, object], device: torch.device
                      ) -> EnsembleUNet2d:
    path = Path(str(start["path"]))
    if dr1.sha256(path) != start["checkpoint_sha256"]:
        raise RuntimeError("start checkpoint changed between paired arms")
    payload = _load_checkpoint(path)
    if dr1.canonical_sha256(payload["model_config"]) != (
            start["checkpoint_model_config_sha256"]):
        raise RuntimeError("start model config changed between paired arms")
    if dr1.model_state_sha256(payload["model_state_dict"]) != (
            start["checkpoint_model_state_sha256"]):
        raise RuntimeError("start state changed between paired arms")
    stored = dict(payload["model_config"])
    stored.pop("architecture")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored)).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    gc.collect()
    return model.to(device)


def make_sim_normalizer(start: Mapping[str, object], device: torch.device):
    values = {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in start["normalizer"].items()}
    return ensemble_train.common.scratch.ChannelNormalizer(
        values["mean_input"], values["mean_target"],
        values["std_input"], values["std_target"], device)


def configure_point_only_trainable(model: EnsembleUNet2d) -> tuple[
        tuple[torch.nn.Parameter, ...], dict]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    frozen_ids: set[int] = set()
    for name in UQ_ONLY_MODULES:
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            frozen_ids.add(id(parameter))
    named = tuple(model.named_parameters())
    trainable = tuple(parameter for _name, parameter in named
                      if parameter.requires_grad)
    frozen = tuple((name, parameter) for name, parameter in named
                   if not parameter.requires_grad)
    if not trainable or not frozen or {id(value) for _name, value in frozen} != frozen_ids:
        raise RuntimeError("point/UQ trainability partition is incomplete")
    return trainable, {
        "trainable_parameters": sum(value.numel() for value in trainable),
        "frozen_uq_parameters": sum(value.numel() for _, value in frozen),
        "frozen_uq_names": [name for name, _ in frozen],
        "all_uq_gradients_absent": True,
    }


def uq_parameter_state(model: EnsembleUNet2d) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, parameter.detach()) for name, parameter in model.named_parameters()
        if not parameter.requires_grad)


def frozen_named_parameters(model: EnsembleUNet2d
                            ) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Return live frozen parameters; unlike a state snapshot these own grads."""
    return tuple(
        (name, parameter) for name, parameter in model.named_parameters()
        if not parameter.requires_grad)


def forward_point_training(model: EnsembleUNet2d, x: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the point/member prefix of ``forward_uq_training`` only."""
    (features, fluctuation_features, _uq_role_features,
     mean_profile_correction) = (
        model._decode_role_features(x))
    if model.config.mean_preserving_member_adapters:
        point_members = model._members_from_features(
            x, features, fluctuation_features, use_member_adapters=False,
            mean_profile_correction=mean_profile_correction)
        raw_members = model._members_from_features(
            x, features, fluctuation_features, base_members=point_members,
            mean_profile_correction=mean_profile_correction)
    else:
        raw_members = model._members_from_features(
            x, features, fluctuation_features,
            mean_profile_correction=mean_profile_correction)
        point_members = raw_members
    if raw_members is point_members:
        point_members, _ = model._apply_hard_region_expert(
            point_members, features, x)
        raw_members = point_members
    else:
        point_members, _ = model._apply_hard_region_expert(
            point_members, features, x)
        raw_members, _ = model._apply_hard_region_expert(
            raw_members, features, x)
    if raw_members is point_members:
        point_members, _ = model._apply_conservative_closure(
            point_members, features, x)
        raw_members = point_members
    else:
        point_members, _ = model._apply_conservative_closure(
            point_members, features, x)
        raw_members, _ = model._apply_conservative_closure(
            raw_members, features, x)
    if raw_members is point_members:
        point_members = model._apply_recurrent_latent_rollout(
            point_members, features[-1])
        raw_members = point_members
    else:
        point_members = model._apply_recurrent_latent_rollout(
            point_members, features[-1])
        raw_members = model._apply_recurrent_latent_rollout(
            raw_members, features[-1])
    dynamics = features if fluctuation_features is None else fluctuation_features
    if model.config.physical_energy_member_sync:
        raw_members = model._energy_controlled_members(raw_members, dynamics, x)
        point_members = model._energy_controlled_members(
            point_members, dynamics, x)
        prediction = point_members.mean(dim=1)
    else:
        prediction = model._energy_controlled_prediction(
            point_members.mean(dim=1), dynamics, x)
    if model.temporal_transport is not None:
        prediction = model.temporal_transport(prediction, features[-1], x)
    members = (
        model._members_with_shared_uv_mean(raw_members)
        if model.config.shared_member_uv_mean else raw_members)
    if model.decoupled_member_residual is not None:
        members = model.decoupled_member_residual(members, features[-1])
    return prediction, members


def _loss_args() -> argparse.Namespace:
    return argparse.Namespace(
        point_loss_mode="official_accuracy", mse_weight=MSE_WEIGHT,
        mse_warmup_updates=0, bootstrap_probability=0.75, seed=SEED)


def masked_member_mse(members: torch.Tensor, target: torch.Tensor,
                      member_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample = (members - target[:, None]).square().mean(
        dim=(2, 3, 4, 5))
    mask = member_mask.to(device=members.device, dtype=members.dtype)
    if tuple(mask.shape) != tuple(per_sample.shape) or not bool(mask.any(dim=1).all()):
        raise ValueError("member mask is not batch-by-member or has empty rows")
    return (per_sample * mask).sum() / mask.sum(), mask.mean()


def backward_point_domain(model: EnsembleUNet2d, batch, normalizer,
                          device: torch.device, domain: str,
                          domain_step: int, member_mask: torch.Tensor) -> dict:
    if domain not in DOMAIN_LOSS:
        raise ValueError(f"unknown domain {domain}")
    history, target, trajectory_ids, _sample_indices = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    if domain == "real":
        history_input = ensemble_train.add_training_input_gain(
            history, dense4k.AUGMENTATION["input_gain_std"])
    else:
        history_input = history
    history_norm, target_norm = normalizer.preprocess(history_input, target)
    if domain == "real":
        history_norm = ensemble_train.masked_velocity_blur(
            history_norm, history,
            dense4k.AUGMENTATION["input_blur_probability"],
            dense4k.AUGMENTATION["input_blur_strength"])
        history_norm = ensemble_train.add_training_input_noise(
            history_norm, history, dense4k.AUGMENTATION["input_noise_std"])
    prediction, members = forward_point_training(model, history_norm)
    contract = DOMAIN_LOSS[domain]
    point_loss, point_components = ensemble_train.point_loss(
        _loss_args(), prediction, target_norm, normalizer,
        contract["accuracy_weights"], domain_step)
    if contract["member_mode"] == "mse":
        member_loss, bootstrap_fraction = masked_member_mse(
            members, target_norm, member_mask)
    else:
        member_loss, bootstrap_fraction = (
            ensemble_train.bootstrap_member_point_loss(
                _loss_args(), members, target_norm, trajectory_ids,
                normalizer, contract["accuracy_weights"], domain_step,
                member_mask=member_mask))
    loss = (contract["point_weight"] * point_loss
            + contract["member_weight"] * member_loss)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite {domain} point-only loss")
    loss.backward()
    values = {
        "loss": float(loss.detach()),
        "point_loss": float(point_loss.detach()),
        "member_loss": float(member_loss.detach()),
        "bootstrap_fraction": float(bootstrap_fraction.detach()),
        **{key: float(value.detach()) for key, value in point_components.items()},
    }
    del history, target, history_input, history_norm, target_norm
    del prediction, members, point_loss, member_loss, loss
    return values


def validate_training_batch(batch, expected: Mapping[str, object]) -> torch.Tensor:
    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
        raise RuntimeError("domain event batch is not an exact tuple4")
    trajectory_ids = batch[2].to(dtype=torch.long, device="cpu")
    sample_indices = batch[3].to(dtype=torch.long, device="cpu")
    actual = {
        "sample_indices": sample_indices.tolist(),
        "trajectory_ids": trajectory_ids.tolist(),
        "bootstrap_mask": expected["bootstrap_mask"],
    }
    if actual != expected:
        raise RuntimeError("domain event batch differs from frozen stream")
    mask = torch.as_tensor(expected["bootstrap_mask"], dtype=torch.bool)
    if tuple(mask.shape) != (len(sample_indices), 4) or not bool(
            mask.any(dim=1).all()):
        raise RuntimeError("frozen member mask contract changed")
    return mask


def _cpu_model_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.state_dict().items())


def save_raw_endpoint(path: Path, model: EnsembleUNet2d, *, arm: str,
                      held_aoa: int, start: Mapping[str, object],
                      real_normalizer, order: Mapping[str, object],
                      training: Mapping[str, object],
                      uq_start_sha256: str) -> dict:
    """Seal the exact terminal optimizer state with exact Real buffers."""
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    ensemble_train.common.set_model_stats(model, real_normalizer)
    uq_end = dr1.model_state_sha256(uq_parameter_state(model))
    if uq_end != uq_start_sha256:
        raise RuntimeError("frozen UQ-only parameters changed")
    state = _cpu_model_state(model)
    state_sha = dr1.model_state_sha256(state)
    frozen_names = [name for name, _parameter in frozen_named_parameters(model)]
    frozen_sha = dr1.model_state_sha256(OrderedDict(
        (name, state[name]) for name in frozen_names))
    if frozen_sha != uq_start_sha256:
        raise RuntimeError("raw endpoint did not preserve exact frozen UQ state")
    config = model.architecture_config()
    normalizer = normalizer_payload(real_normalizer)
    expected_updates = int(order["optimizer_updates"])
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real-stage-domain-joint",
        "arm": arm,
        "endpoint_kind": "raw",
        "held_aoa": held_aoa,
        "optimizer_updates": expected_updates,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": config,
        "model_config_sha256": canonical_sha256(config),
        "start": dict(start),
        "real_normalizer": normalizer,
        "real_normalizer_sha256": canonical_sha256(normalizer),
        "event_order_sha256": order["sha256"],
        "training": dict(training),
        "endpoint_materialization": {
            "kind": "exact_raw_terminal_model_state",
            "trainable_parameters_from_raw": True,
            "frozen_uq_parameters_from_raw": True,
            "frozen_parameter_names": frozen_names,
            "final_raw_model_state_sha256": state_sha,
            "final_frozen_uq_state_sha256": frozen_sha,
        },
        "uq_only_parameters_frozen": True,
        "uq_parameter_start_sha256": uq_start_sha256,
        "uq_parameter_end_sha256": uq_end,
        "uq_parameter_endpoint_sha256": frozen_sha,
        "held_target_opened_before_seal": False,
        "optimizer_state_saved": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    seal = {
        "arm": arm,
        "endpoint_kind": "raw",
        "held_aoa": held_aoa,
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": payload["model_config_sha256"],
        "real_normalizer_sha256": payload["real_normalizer_sha256"],
        "optimizer_updates": expected_updates,
        "event_order_sha256": order["sha256"],
    }
    del state
    return seal


def save_endpoint(path: Path, model: EnsembleUNet2d,
                  ema: ema_lib.FP32EMAShadow, *, arm: str, held_aoa: int,
                  start: Mapping[str, object], real_normalizer,
                  order: Mapping[str, object], training: Mapping[str, object],
                  uq_start_sha256: str) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    ensemble_train.common.set_model_stats(model, real_normalizer)
    uq_end = dr1.model_state_sha256(uq_parameter_state(model))
    if uq_end != uq_start_sha256:
        raise RuntimeError("frozen UQ-only parameters changed")
    raw_before = dr1.model_state_sha256(model.state_dict())
    ema_payload = ema.seal(model)
    raw_after = dr1.model_state_sha256(model.state_dict())
    if raw_before != raw_after:
        raise RuntimeError("EMA seal mutated the raw model")
    state = ema_payload.pop("model_state")
    expected_updates = int(order["optimizer_updates"])
    if int(ema_payload["updates"]) != expected_updates:
        raise RuntimeError("EMA update count differs from optimizer events")
    # FP32 x <- decay*x + (1-decay)*x is not bitwise idempotent.  Although UQ
    # parameters never receive a gradient, repeatedly updating a full-model
    # EMA drifts some of them by rounding.  The endpoint is therefore explicit:
    # trainable point parameters use EMA, frozen UQ parameters use exact raw
    # values.  ``ema_payload`` remains the unmodified source-seal metadata and
    # its materialized hash intentionally describes the pre-override state.
    frozen_names = [name for name, _parameter in frozen_named_parameters(model)]
    for name, parameter in frozen_named_parameters(model):
        if name not in state:
            raise RuntimeError(f"frozen UQ parameter missing from EMA state: {name}")
        state[name] = parameter.detach().cpu().clone()
    final_frozen_sha = dr1.model_state_sha256(OrderedDict(
        (name, state[name]) for name in frozen_names))
    if final_frozen_sha != uq_start_sha256:
        raise RuntimeError("hybrid endpoint did not restore exact frozen UQ state")
    state_sha = dr1.model_state_sha256(state)
    config = model.architecture_config()
    normalizer = normalizer_payload(real_normalizer)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real-stage-domain-joint",
        "arm": arm,
        "endpoint_kind": "ema",
        "held_aoa": held_aoa,
        "optimizer_updates": expected_updates,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": config,
        "model_config_sha256": canonical_sha256(config),
        "start": dict(start),
        "real_normalizer": normalizer,
        "real_normalizer_sha256": canonical_sha256(normalizer),
        "event_order_sha256": order["sha256"],
        "training": dict(training),
        "ema_source_seal_before_frozen_override": ema_payload,
        "endpoint_materialization": {
            "kind": "trainable_parameter_ema_plus_exact_raw_frozen_uq",
            "trainable_parameters_from_ema": True,
            "frozen_uq_parameters_from_raw": True,
            "frozen_parameter_names": frozen_names,
            "source_pre_override_materialized_model_state_sha256": (
                ema_payload["materialized_model_state_sha256"]),
            "final_hybrid_model_state_sha256": state_sha,
            "final_frozen_uq_state_sha256": final_frozen_sha,
        },
        "uq_only_parameters_frozen": True,
        "uq_parameter_start_sha256": uq_start_sha256,
        "uq_parameter_end_sha256": uq_end,
        "uq_parameter_endpoint_sha256": final_frozen_sha,
        "held_target_opened_before_seal": False,
        "optimizer_state_saved": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    seal = {
        "arm": arm,
        "endpoint_kind": "ema",
        "held_aoa": held_aoa,
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": payload["model_config_sha256"],
        "real_normalizer_sha256": payload["real_normalizer_sha256"],
        "optimizer_updates": expected_updates,
        "event_order_sha256": order["sha256"],
    }
    del state
    return seal


def _make_loader(dataset, schedule: Mapping[str, object], device: torch.device):
    tagged = ensemble_train.TrajectoryTaggedDataset(dataset)
    return DataLoader(
        tagged, batch_sampler=schedule["batches"], num_workers=0,
        pin_memory=device.type == "cuda")


def train_arm(arm: str, start: Mapping[str, object], datasets: Mapping,
              schedules: Mapping[str, Mapping], normalizers: Mapping,
              order: Mapping[str, object], device: torch.device,
              held_aoa: int, output_dir: Path, deadline: float,
              *, save_raw: bool = False) -> dict:
    memory_stats_reset = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_stats_reset = True
    ensemble_train.seed_everything(SEED)
    model = build_start_model(start, device)
    ensemble_train.common.set_model_stats(model, normalizers["real"])
    start_state_sha = dr1.model_state_sha256(model.state_dict())
    trainable, trainability = configure_point_only_trainable(model)
    uq_start_sha = dr1.model_state_sha256(uq_parameter_state(model))
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    if optimizer.state:
        raise RuntimeError("paired arm optimizer did not start empty")
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    exposures = {
        domain: int(order["domain_exposures"].get(domain, 0))
        for domain in DOMAINS}
    active_domains = tuple(domain for domain in DOMAINS if exposures[domain] > 0)
    if not active_domains or sum(exposures.values()) != int(
            order["optimizer_updates"]):
        raise RuntimeError("event order has invalid domain exposure counts")
    loaders = {
        domain: iter(_make_loader(datasets[domain], schedules[domain], device))
        for domain in active_domains}
    actual_by_domain = {domain: [] for domain in DOMAINS}
    trace_by_domain = {domain: [] for domain in DOMAINS}
    sums = {domain: {"loss": 0.0} for domain in DOMAINS}
    counts = {domain: 0 for domain in DOMAINS}
    started = time.perf_counter()
    peak_memory = 0
    for event in order["rows"]:
        if time.perf_counter() > deadline:
            raise TimeoutError("domain-joint pilot wall budget expired")
        domain = str(event["domain"])
        domain_step = int(event["domain_step"])
        optimizer_step = int(event["optimizer_step"])
        batch = next(loaders[domain])
        expected = schedules[domain]["rows"][domain_step - 1]
        member_mask = validate_training_batch(batch, expected)
        actual_by_domain[domain].append(dict(expected))
        lr = domain_learning_rate(
            domain, domain_step, len(schedules[domain]["rows"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        ensemble_train.common.set_model_stats(model, normalizers[domain])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with domain_event_rng(domain, domain_step, device) as rng:
            values = backward_point_domain(
                model, batch, normalizers[domain], device, domain,
                domain_step, member_mask)
            if any(parameter.grad is not None
                   for _name, parameter in frozen_named_parameters(model)):
                raise RuntimeError("UQ-only parameter received a gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, GRAD_CLIP)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite clipped gradient")
            optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=optimizer_step)
        counts[domain] += 1
        sums[domain]["loss"] += values["loss"]
        trace_by_domain[domain].append({
            "domain_step": domain_step,
            "sample_row": expected,
            "learning_rate": lr,
            "rng_seed": rng["seed"],
            "rng_fingerprint": rng["fingerprint"],
            "rng_contract_fingerprint": rng["contract_fingerprint"],
        })
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        if optimizer_step % 50 == 0 or optimizer_step == order["optimizer_updates"]:
            print(json.dumps({
                "event": "domain_joint_progress", "arm": arm,
                "optimizer_step": optimizer_step, "domain": domain,
                "domain_step": domain_step, "loss": values["loss"],
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)
    for domain in DOMAINS:
        expected_rows = (schedules[domain]["rows"]
                         if exposures[domain] else [])
        if actual_by_domain[domain] != expected_rows:
            raise RuntimeError(f"{arm} did not replay the frozen {domain} stream")
    training = {
        "optimizer_updates": sum(counts.values()),
        "ema_updates": ema.updates,
        "domain_exposures": counts,
        "actual_stream_sha256": {
            domain: canonical_sha256(actual_by_domain[domain])
            for domain in DOMAINS},
        "paired_trace_sha256": {
            domain: canonical_sha256(trace_by_domain[domain])
            for domain in DOMAINS},
        "mean_loss": {
            domain: (sums[domain]["loss"] / counts[domain]
                     if counts[domain] else None) for domain in DOMAINS},
        "constant_domain_local_lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "real_mse_warmup_updates": 0,
        "real_augmentation": dict(dense4k.AUGMENTATION),
        "loss_contract": DOMAIN_LOSS,
        "trainability": trainability,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak_memory,
        "peak_memory_stats_reset_per_arm": memory_stats_reset,
        "rng_trace_sha256": {
            domain: canonical_sha256([{
                "domain_step": row["domain_step"],
                "rng_seed": row["rng_seed"],
                "rng_fingerprint": row["rng_fingerprint"],
                "rng_contract_fingerprint": row["rng_contract_fingerprint"],
            } for row in trace_by_domain[domain]]) for domain in DOMAINS},
        "lr_trace_sha256": {
            domain: canonical_sha256([{
                "domain_step": row["domain_step"],
                "learning_rate": row["learning_rate"],
            } for row in trace_by_domain[domain]]) for domain in DOMAINS},
    }
    seals = {}
    if save_raw:
        seals["raw"] = save_raw_endpoint(
            output_dir / f"{arm}_raw.pth", model, arm=arm,
            held_aoa=held_aoa, start=start,
            real_normalizer=normalizers["real"], order=order,
            training=training, uq_start_sha256=uq_start_sha)
    seals["ema"] = save_endpoint(
        output_dir / f"{arm}_ema999.pth", model, ema, arm=arm,
        held_aoa=held_aoa, start=start, real_normalizer=normalizers["real"],
        order=order, training=training, uq_start_sha256=uq_start_sha)
    result = {
        "start_state_sha256": start_state_sha,
        "training": training,
        # ``seal`` is retained as the Stage-A-compatible EMA alias.
        "seal": seals["ema"],
        "seals": seals,
    }
    del model, optimizer, ema, trainable, loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def paired_arm_audit(reports: Mapping[str, Mapping],
                     schedules: Mapping[str, Mapping], updates: int) -> dict:
    d0, d1 = (reports[arm] for arm in ARMS)
    checks = {
        "same_start_state": d0["start_state_sha256"] == d1["start_state_sha256"],
        "same_sim_stream": d0["training"]["actual_stream_sha256"]["sim"]
        == d1["training"]["actual_stream_sha256"]["sim"]
        == schedules["sim"]["sha256"],
        "same_real_stream": d0["training"]["actual_stream_sha256"]["real"]
        == d1["training"]["actual_stream_sha256"]["real"]
        == schedules["real"]["sha256"],
        "same_sim_rng_lr_trace": d0["training"]["paired_trace_sha256"]["sim"]
        == d1["training"]["paired_trace_sha256"]["sim"],
        "same_real_rng_lr_trace": d0["training"]["paired_trace_sha256"]["real"]
        == d1["training"]["paired_trace_sha256"]["real"],
        "two_n_optimizer_updates": all(
            report["training"]["optimizer_updates"] == 2 * updates
            for report in reports.values()),
        "n_exposures_per_domain": all(
            report["training"]["domain_exposures"]
            == {"sim": updates, "real": updates}
            for report in reports.values()),
        "ema_matches_optimizer_updates": all(
            report["training"]["ema_updates"] == 2 * updates
            for report in reports.values()),
    }
    return {"checks": checks, "pass": all(checks.values())}


def real_only_reference_audit(reports: Mapping[str, Mapping],
                              schedules: Mapping[str, Mapping],
                              updates: int) -> dict:
    """Verify D-minus-1 shares Real events but is not an equal-compute arm."""
    reference = reports[REFERENCE_ARM]
    causal = [reports[arm] for arm in ARMS]
    empty_sha = canonical_sha256([])
    checks = {
        "same_fresh_start_as_causal_pair": all(
            reference["start_state_sha256"] == row["start_state_sha256"]
            for row in causal),
        "same_real_stream_as_causal_pair": all(
            reference["training"]["actual_stream_sha256"]["real"]
            == row["training"]["actual_stream_sha256"]["real"]
            == schedules["real"]["sha256"] for row in causal),
        "same_real_rng_lr_trace_as_causal_pair": all(
            reference["training"]["paired_trace_sha256"]["real"]
            == row["training"]["paired_trace_sha256"]["real"]
            for row in causal),
        "no_sim_stream_or_exposure": (
            reference["training"]["actual_stream_sha256"]["sim"] == empty_sha
            and reference["training"]["domain_exposures"]["sim"] == 0),
        "n_real_optimizer_updates": (
            reference["training"]["optimizer_updates"] == updates
            and reference["training"]["domain_exposures"]["real"] == updates
            and reference["training"]["ema_updates"] == updates),
        "explicitly_not_equal_compute": True,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "interpretation": (
            "absolute same-start Real-only reference; N updates, not an "
            "equal-compute causal arm"),
    }


def load_endpoint_model(seal: Mapping[str, object], real_normalizer,
                        device: torch.device, *,
                        expected_parameters: int = EXPECTED_B210_PARAMETERS,
                        ) -> tuple[EnsembleUNet2d, dict]:
    """Hash-verify and strict-load one sealed raw/EMA endpoint."""
    path = Path(str(seal["path"]))
    if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("endpoint file changed after sealing")
    payload = _load_checkpoint(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != ENDPOINT_SCHEMA:
        raise ValueError("invalid domain-joint endpoint schema")
    if (payload.get("arm") != seal.get("arm")
            or payload.get("endpoint_kind") != seal.get("endpoint_kind")):
        raise RuntimeError("endpoint identity differs from its seal")
    for key in ("held_aoa", "optimizer_updates", "event_order_sha256"):
        if payload.get(key) != seal.get(key):
            raise RuntimeError(f"endpoint {key} differs from its seal")
    state = payload.get("model_state_dict")
    config = payload.get("model_config")
    if not isinstance(state, Mapping) or not isinstance(config, Mapping):
        raise ValueError("endpoint lacks model config/state")
    if (dr1.model_state_sha256(state) != payload.get("model_state_sha256")
            or payload.get("model_state_sha256") != seal.get("model_state_sha256")):
        raise RuntimeError("endpoint model-state hash failed")
    if (canonical_sha256(config) != payload.get("model_config_sha256")
            or payload.get("model_config_sha256") != seal.get(
                "model_config_sha256")):
        raise RuntimeError("endpoint model-config hash failed")
    expected_normalizer = normalizer_payload(real_normalizer)
    if (canonical_sha256(expected_normalizer)
            != payload.get("real_normalizer_sha256")
            or payload.get("real_normalizer_sha256") != seal.get(
                "real_normalizer_sha256")):
        raise RuntimeError("endpoint Real normalizer differs from evaluation")
    parsed = dict(config)
    if parsed.pop("architecture", None) != "ensemble_unet":
        raise ValueError("endpoint is not an ensemble U-Net")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**parsed)).float()
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != expected_parameters:
        raise RuntimeError(
            f"endpoint parameter count {parameter_count} != {expected_parameters}")
    model.load_state_dict(state, strict=True)
    loaded_sha = dr1.model_state_sha256(model.state_dict())
    if loaded_sha != seal["model_state_sha256"]:
        raise RuntimeError("strict-loaded endpoint differs from seal")
    report = {
        "path": str(path.resolve()),
        "file_sha256": seal["file_sha256"],
        "model_state_sha256": loaded_sha,
        "model_config_sha256": seal["model_config_sha256"],
        "real_normalizer_sha256": seal["real_normalizer_sha256"],
        "arm": seal["arm"],
        "endpoint_kind": seal["endpoint_kind"],
        "held_aoa": seal["held_aoa"],
        "optimizer_updates": seal["optimizer_updates"],
        "event_order_sha256": seal["event_order_sha256"],
        "parameters": parameter_count,
        "strict_load": True,
    }
    del payload, state
    gc.collect()
    return model.to(device).eval(), report


def _update_raw_tensor_hash(digest, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))


def point_metrics_from_official_errors(errors: np.ndarray) -> dict[str, object]:
    """Pool official per-sample errors, then apply released score maps."""
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("official errors must have non-empty shape (N,3)")
    means = values.mean(axis=0, dtype=np.float64)
    official = official_scoring()
    scores = {
        "rel_l2_score": official.score_error(float(means[0])),
        "tke_score": official.score_error(float(means[1])),
        "mvpe_score": official.score_error(float(means[2])),
    }
    point = float(sum(POINT_WEIGHTS[key] * scores[key]
                      for key in POINT_WEIGHTS))
    result: dict[str, object] = {
        "samples": int(len(values)),
        "mean_errors": {
            "rel_l2": float(means[0]),
            "tke_rel_l2": float(means[1]),
            "mvpe_rel_l2": float(means[2]),
        },
        **scores,
        "point_final_contribution_46_10_9": point,
        "point_formula": "0.46*Rel-L2 + 0.10*TKE + 0.09*MVPE",
    }
    if not all(math.isfinite(float(result[key])) for key in POINT_METRIC_KEYS):
        raise FloatingPointError("official point scores are non-finite")
    return result


@torch.inference_mode()
def evaluate_point_endpoint(model: EnsembleUNet2d, dataset,
                            sample_aoas: Sequence[int], real_normalizer,
                            device: torch.device, deadline: float) -> dict:
    """Evaluate one endpoint through the released v9 point scorer."""
    if len(dataset) <= 0 or len(sample_aoas) != len(dataset):
        raise ValueError("inner-validation dataset/AoA labels are empty or misaligned")
    batches = [
        list(range(start, min(start + EVAL_BATCH_SIZE, len(dataset))))
        for start in range(0, len(dataset), EVAL_BATCH_SIZE)]
    loader = DataLoader(
        dataset, batch_sampler=batches, num_workers=0,
        pin_memory=device.type == "cuda")
    errors = []
    per_aoa_errors: dict[int, list[np.ndarray]] = {
        aoa: [] for aoa in sorted(set(int(value) for value in sample_aoas))}
    input_hasher = hashlib.sha256()
    target_hasher = hashlib.sha256()
    scored_mask_hasher = hashlib.sha256()
    started = time.perf_counter()
    model.eval()
    official = official_scoring()
    for indices, (history, target) in zip(batches, loader, strict=True):
        if time.perf_counter() > deadline:
            raise TimeoutError("Stage-B quick evaluation wall budget expired")
        target_cpu = target.to(dtype=torch.float32).contiguous()
        _update_raw_tensor_hash(input_hasher, history)
        _update_raw_tensor_hash(target_hasher, target_cpu)
        _update_raw_tensor_hash(
            scored_mask_hasher,
            target_cpu[..., :2].ne(0.0).to(torch.uint8))
        history = history.to(
            device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(
            device=device, dtype=torch.float32, non_blocking=True)
        history_norm, _target_norm = real_normalizer.preprocess(history, target)
        prediction_norm, _members = forward_point_training(model, history_norm)
        if not isinstance(prediction_norm, torch.Tensor):
            raise TypeError("point endpoint forward did not return one tensor")
        prediction = real_normalizer.postprocess_prediction(prediction_norm)
        prediction_np = np.ascontiguousarray(
            prediction.detach().cpu().numpy(), dtype=np.float32)
        target_np = np.ascontiguousarray(
            target_cpu.numpy(), dtype=np.float32)
        official.validate_shapes(prediction_np, target_np)
        channels = official.measured_channels(target_np)
        if channels != 2:
            raise ValueError(f"inner validation must have two measured channels, got {channels}")
        batch_errors = np.stack((
            official.rel_l2_per_sample(prediction_np, target_np, channels),
            official.tke_rel_l2_per_sample(prediction_np, target_np, channels),
            official.mvpe_rel_l2_per_sample(prediction_np, target_np),
        ), axis=1).astype(np.float64, copy=False)
        errors.append(batch_errors)
        batch_aoas = np.asarray(
            [int(sample_aoas[index]) for index in indices], dtype=np.int64)
        for aoa in sorted(set(batch_aoas.tolist())):
            per_aoa_errors[aoa].append(batch_errors[batch_aoas == aoa])
    return {
        "metrics": point_metrics_from_official_errors(np.concatenate(errors)),
        "per_aoa_metrics": {
            str(aoa): point_metrics_from_official_errors(
                np.concatenate(per_aoa_errors[aoa]))
            for aoa in sorted(per_aoa_errors)},
        "ordered_raw_input_sha256": input_hasher.hexdigest(),
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": scored_mask_hasher.hexdigest(),
        "official_scoring_path": str(SCORING_PATH.resolve()),
        "official_scoring_file_sha256": dr1.sha256(SCORING_PATH),
        "aggregation": "pooled official per-sample errors, then score_error",
        "forward_api": (
            "forward_point_training(model,x)[0], bitwise-tested against "
            "forward_uq_training point prefix"),
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_seconds": time.perf_counter() - started,
        "sps": "not_computed/undefined",
        "final": "not_computed/undefined",
    }


def compare_point_metrics(candidate: Mapping[str, object],
                          baseline: Mapping[str, object]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in POINT_METRIC_KEYS}


def stage_b_quick_gate(raw_d1_minus_d0: Mapping[str, float],
                       raw_d1_minus_reference: Mapping[str, float]) -> dict:
    component_keys = tuple(POINT_WEIGHTS)
    checks = {
        "raw_d1_minus_d0_point_at_least_0_03": float(
            raw_d1_minus_d0["point_final_contribution_46_10_9"]
        ) >= QUICK_GATE["raw_d1_minus_d0_point_min"],
        "each_raw_d1_minus_d0_component_at_least_minus_0_20": all(
            float(raw_d1_minus_d0[key]) >= QUICK_GATE[
                "raw_d1_minus_d0_component_floor"] for key in component_keys),
        "raw_d1_not_worse_than_real_only_reference": float(
            raw_d1_minus_reference["point_final_contribution_46_10_9"]
        ) >= QUICK_GATE["raw_d1_minus_dminus1_point_min"],
    }
    go = all(checks.values())
    return {
        "decision": "GO" if go else "NO_GO",
        "pass": go,
        "primary_endpoint": "raw",
        "ema_role": "secondary_report_only",
        "thresholds": dict(QUICK_GATE),
        "checks": checks,
        "stage_c_auto_promotion": False,
        "stage_c_authorized": False,
    }


def prepare_sim(args: argparse.Namespace) -> tuple[object, list, dict]:
    if dr1.sha256(args.sim_manifest) != args.sim_manifest_sha256:
        raise RuntimeError("frozen Sim manifest file SHA changed")
    expected = json.loads(args.sim_manifest.read_text(encoding="utf-8"))
    if canonical_sha256(expected) != DEFAULT_SIM_MANIFEST_SEMANTIC_SHA256:
        raise RuntimeError("frozen Sim manifest semantic SHA changed")
    trajectories, windows, generated = (
        ensemble_train.common.scratch.load_simulations(
            args.sim_dir, 0, 20, True))
    if generated != expected:
        raise RuntimeError("loaded Sim data does not reproduce frozen manifest")
    dataset = ensemble_train.common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    return dataset, trajectories, {
        "path": str(args.sim_manifest.resolve()),
        "file_sha256": args.sim_manifest_sha256,
        "semantic_sha256": canonical_sha256(generated),
        "trajectories": len(trajectories), "windows": len(windows),
    }


def run_pilot(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B210 Stage-A pilot requires CUDA")
    if dr1.sha256(args.clean_manifest) != args.clean_manifest_sha256:
        raise RuntimeError("clean81 manifest file SHA changed")
    rows, real_manifest = dr1.load_manifest(
        args.clean_manifest, dr1.FORMAL_MANIFEST_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    if not fit_rows or not held_rows:
        raise RuntimeError("empty fit/held Real partition")
    start = inspect_start_checkpoint(
        args.start_sim, args.start_sim_sha256,
        args.expected_start_iteration)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gate = HeldTargetGate(args.held_aoa)
    fit_trajectories, fit_metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", gate)
    statistics = fit_only_real_statistics(
        fit_trajectories, fit_metadata, sparse_windows, args.held_aoa)
    real_normalizer = smu1.FoldNormalizer(statistics, device)
    dense_windows = dense4k.dense_windows(
        fit_trajectories, dense4k.CANDIDATE_STRIDE)
    real_dataset = ensemble_train.common.repro.WindowDataset(
        fit_trajectories, dense_windows)
    sim_dataset, sim_trajectories, sim_report = prepare_sim(args)
    sim_normalizer = make_sim_normalizer(start, device)
    schedules = {
        "sim": smu1.paired_batch_schedule(
            range(len(sim_dataset)), sim_dataset.windows,
            updates=args.domain_updates, seed=SEED),
        "real": smu1.paired_batch_schedule(
            range(len(real_dataset)), dense_windows,
            updates=args.domain_updates, seed=SEED),
    }
    orders = build_domain_event_orders(args.domain_updates)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "fit_normalizer.pt")
    reports = {}
    datasets = {"sim": sim_dataset, "real": real_dataset}
    normalizers = {"sim": sim_normalizer, "real": real_normalizer}
    for arm in ARMS:
        arm_dir = args.output_dir / arm
        arm_dir.mkdir()
        reports[arm] = train_arm(
            arm, start, datasets, schedules, normalizers, orders[arm], device,
            args.held_aoa, arm_dir, deadline)
        gate.seal(arm, reports[arm]["seal"])
    audit = paired_arm_audit(reports, schedules, args.domain_updates)
    if not audit["pass"] or not gate.all_sealed:
        raise RuntimeError("paired Stage-A pilot audit failed")
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "status": "PASS_STAGE_A_ENGINEERING_PILOT",
        "pilot": True,
        "held_aoa": args.held_aoa,
        "held_target_evaluated": False,
        "domain_updates_per_arm": args.domain_updates,
        "optimizer_updates_per_arm": 2 * args.domain_updates,
        "start": start,
        "sim_manifest": sim_report,
        "real_manifest": real_manifest,
        "real_fit": {
            "trajectories": len(fit_rows),
            "sparse_normalizer_windows": len(sparse_windows),
            "dense_optimizer_windows": len(dense_windows),
            "normalizer_kind": "other-four-AoA sparse stride20 prefixes only",
            "normalizer_sha256": smu1.statistics_digest(statistics),
            "held_condition_in_normalizer": False,
        },
        "streams": {
            domain: {
                "sha256": schedules[domain]["sha256"],
                "updates": args.domain_updates,
            } for domain in DOMAINS},
        "orders": {
            arm: {key: value for key, value in orders[arm].items()
                  if key != "rows"} for arm in ARMS},
        "arms": reports,
        "paired_audit": audit,
        "access_audit": gate.report(),
        "determinism": {
            "domain_rng_key": "(master_seed, domain, exposure_index)",
            "master_seed": SEED,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms_forced": False,
        },
        "git": git_identity(),
        "command": sys.argv,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_class": (
            "Real-stage LOAO with all released Sim conditions; Stage-A "
            "engineering pilot only, no held-target score"),
        "submission_authorized": False,
    }
    write_json_atomic(args.output_dir / "pilot_result.json", result)
    return result


def run_quick(args: argparse.Namespace) -> dict:
    """Run the frozen Stage-B fit-side quick gate without opening held AoA 10."""
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B210 Stage-B quick requires CUDA")
    if dr1.sha256(args.clean_manifest) != args.clean_manifest_sha256:
        raise RuntimeError("clean81 manifest file SHA changed")
    rows, real_manifest = dr1.load_manifest(
        args.clean_manifest, dr1.FORMAL_MANIFEST_SHA256)
    split = deterministic_stratified_trajectory_split(rows, args.held_aoa)
    inner_train_rows = split["inner_train_rows"]
    inner_val_rows = split["inner_val_rows"]
    start = inspect_start_checkpoint(
        args.start_sim, args.start_sim_sha256,
        args.expected_start_iteration)

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gate = QuickAccessGate(
        args.held_aoa, inner_train_rows, inner_val_rows)

    train_trajectories, train_metadata, sparse_train_windows = (
        dr1.load_prefix_rows(
            args.real_dir, inner_train_rows, "inner_train", gate))
    inner_train_files = [str(row["file"]) for row in inner_train_rows]
    statistics = fit_only_real_statistics(
        train_trajectories, train_metadata, sparse_train_windows,
        args.held_aoa, allowed_files=inner_train_files)
    real_normalizer = smu1.FoldNormalizer(statistics, device)
    dense_train_windows = dense4k.dense_windows(
        train_trajectories, dense4k.CANDIDATE_STRIDE)
    real_dataset = ensemble_train.common.repro.WindowDataset(
        train_trajectories, dense_train_windows)
    sim_dataset, sim_trajectories, sim_report = prepare_sim(args)
    sim_normalizer = make_sim_normalizer(start, device)
    schedules = {
        "sim": smu1.paired_batch_schedule(
            range(len(sim_dataset)), sim_dataset.windows,
            updates=args.domain_updates, seed=SEED),
        "real": smu1.paired_batch_schedule(
            range(len(real_dataset)), dense_train_windows,
            updates=args.domain_updates, seed=SEED),
    }
    orders = build_quick_event_orders(args.domain_updates)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "inner_train_normalizer.pt")

    reports = {}
    datasets = {"sim": sim_dataset, "real": real_dataset}
    normalizers = {"sim": sim_normalizer, "real": real_normalizer}
    for run in QUICK_RUNS:
        run_dir = args.output_dir / run
        run_dir.mkdir()
        reports[run] = train_arm(
            run, start, datasets, schedules, normalizers, orders[run], device,
            args.held_aoa, run_dir, deadline, save_raw=True)
        for kind in ENDPOINT_KINDS:
            gate.seal(run, kind, reports[run]["seals"][kind])

    causal_reports = {arm: reports[arm] for arm in ARMS}
    paired_audit = paired_arm_audit(
        causal_reports, schedules, args.domain_updates)
    reference_audit = real_only_reference_audit(
        reports, schedules, args.domain_updates)
    if not paired_audit["pass"] or not reference_audit["pass"]:
        raise RuntimeError("Stage-B event-stream audit failed")
    if not gate.all_sealed:
        raise RuntimeError("Stage-B endpoints are incomplete before inner-val")

    # Drop all optimizer-side datasets before the first inner-validation H5
    # open.  No selection, normalizer, dense windows, or augmentation setting
    # can therefore consult an inner-validation trajectory.
    del datasets, real_dataset, sim_dataset, sim_trajectories
    del train_trajectories, train_metadata
    gc.collect()
    torch.cuda.empty_cache()

    val_trajectories, val_metadata, val_windows = dr1.load_prefix_rows(
        args.real_dir, inner_val_rows, "inner_val", gate)
    val_dataset = ensemble_train.common.repro.WindowDataset(
        val_trajectories, val_windows)
    val_sample_aoas = [
        int(val_metadata[int(window.trajectory)]["aoa"])
        for window in val_windows]
    evaluations: dict[str, dict] = {}
    for run in QUICK_RUNS:
        evaluations[run] = {}
        for kind in ENDPOINT_KINDS:
            seal = reports[run]["seals"][kind]
            model, load_report = load_endpoint_model(
                seal, real_normalizer, device)
            endpoint_report = evaluate_point_endpoint(
                model, val_dataset, val_sample_aoas,
                real_normalizer, device, deadline)
            gate.record_evaluation(run, kind)
            endpoint_report["load"] = load_report
            evaluations[run][kind] = endpoint_report
            del model
            gc.collect()
            torch.cuda.empty_cache()

    comparisons = {}
    for kind in ENDPOINT_KINDS:
        d0_metrics = evaluations[ARMS[0]][kind]["metrics"]
        d1_metrics = evaluations[ARMS[1]][kind]["metrics"]
        reference_metrics = evaluations[REFERENCE_ARM][kind]["metrics"]
        comparisons[kind] = {
            "d1_minus_d0": compare_point_metrics(d1_metrics, d0_metrics),
            "d1_minus_dminus1": compare_point_metrics(
                d1_metrics, reference_metrics),
        }
    decision = stage_b_quick_gate(
        comparisons["raw"]["d1_minus_d0"],
        comparisons["raw"]["d1_minus_dminus1"])

    target_hashes = {
        evaluations[run][kind]["ordered_raw_target_sha256"]
        for run in QUICK_RUNS for kind in ENDPOINT_KINDS}
    input_hashes = {
        evaluations[run][kind]["ordered_raw_input_sha256"]
        for run in QUICK_RUNS for kind in ENDPOINT_KINDS}
    mask_hashes = {
        evaluations[run][kind]["ordered_scored_mask_sha256"]
        for run in QUICK_RUNS for kind in ENDPOINT_KINDS}
    evaluation_audit = {
        "all_endpoints_same_ordered_input": len(input_hashes) == 1,
        "all_endpoints_same_ordered_target": len(target_hashes) == 1,
        "all_endpoints_same_ordered_scored_mask": len(mask_hashes) == 1,
        "inner_val_windows": len(val_windows),
        "inner_val_trajectories": len(val_metadata),
        "inner_val_rows_sha256": split["inner_val_rows_sha256"],
    }
    if not all(value is True for key, value in evaluation_audit.items()
               if key.startswith("all_endpoints_")):
        raise RuntimeError("Stage-B endpoint evaluation streams differ")

    result = {
        "schema": QUICK_SCHEMA,
        "experiment": EXPERIMENT,
        "status": f"{decision['decision']}_STAGE_B_QUICK",
        "stage": "quick",
        "pilot": False,
        "held_aoa": args.held_aoa,
        "held_target_evaluated": False,
        "domain_updates": {
            ARMS[0]: {"sim": args.domain_updates, "real": args.domain_updates},
            ARMS[1]: {"sim": args.domain_updates, "real": args.domain_updates},
            REFERENCE_ARM: {"sim": 0, "real": args.domain_updates},
        },
        "optimizer_updates": {
            ARMS[0]: 2 * args.domain_updates,
            ARMS[1]: 2 * args.domain_updates,
            REFERENCE_ARM: args.domain_updates,
        },
        "start": start,
        "sim_manifest": sim_report,
        "real_manifest": {
            **real_manifest,
            "path": str(args.clean_manifest.resolve()),
            "file_sha256": args.clean_manifest_sha256,
        },
        "trajectory_split": split,
        "real_fit": {
            "inner_train_trajectories": len(inner_train_rows),
            "inner_val_trajectories": len(inner_val_rows),
            "held_trajectories_unopened": len(split["held_rows"]),
            "sparse_normalizer_windows": len(sparse_train_windows),
            "dense_optimizer_windows": len(dense_train_windows),
            "inner_val_windows": len(val_windows),
            "normalizer_kind": (
                "inner-train trajectories, sparse stride20 prefixes only"),
            "normalizer_sha256": smu1.statistics_digest(statistics),
            "normalizer_input_rows_sha256": split[
                "inner_train_rows_sha256"],
            "inner_val_in_normalizer": False,
            "held_condition_in_normalizer": False,
            "dense_optimizer_rows_sha256": split[
                "inner_train_rows_sha256"],
            "inner_val_in_dense_optimizer_windows": False,
            "augmentation_contract": dict(dense4k.AUGMENTATION),
            "augmentation_parameters_frozen_before_inner_val": True,
            "checkpoint_selection": "fixed terminal endpoint only",
            "checkpoint_selection_uses_inner_val": False,
        },
        "streams": {
            domain: {
                "sha256": schedules[domain]["sha256"],
                "updates": args.domain_updates,
            } for domain in DOMAINS},
        "orders": {
            run: {key: value for key, value in orders[run].items()
                  if key != "rows"} for run in QUICK_RUNS},
        "runs": reports,
        "paired_causal_audit": paired_audit,
        "real_only_reference_audit": reference_audit,
        "evaluations": evaluations,
        "evaluation_audit": evaluation_audit,
        "comparisons": comparisons,
        "quick_gate": decision,
        "access_audit": gate.report(),
        "determinism": {
            "domain_rng_key": "(master_seed, domain, exposure_index)",
            "master_seed": SEED,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms_forced": False,
        },
        "git": git_identity(),
        "command": sys.argv,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_class": (
            "fit-side trajectory-held Stage-B quick gate on other-four Real "
            "AoAs from a full-AoA Sim10k start; held AoA10 remains unopened"),
        "stage_c_auto_promotion": False,
        "stage_c_authorized": False,
        "submission_authorized": False,
    }
    write_json_atomic(args.output_dir / "quick_result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument(
        "--stage", choices=("pilot", "quick"),
        help="Run the frozen Stage-A pilot or Stage-B fit-side quick gate.")
    stage.add_argument(
        "--pilot", action="store_true",
        help="Legacy exact alias for --stage pilot.")
    parser.add_argument("--held-aoa", type=int, required=True)
    parser.add_argument("--domain-updates", type=int, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", required=True)
    parser.add_argument("--expected-start-iteration", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path,
                        default=Path("/root/autodl-tmp/realpde/data/sim"))
    parser.add_argument("--real-dir", type=Path,
                        default=Path("/root/autodl-tmp/realpde/data/real"))
    parser.add_argument("--sim-manifest", type=Path,
                        default=DEFAULT_SIM_MANIFEST)
    parser.add_argument("--sim-manifest-sha256",
                        default=DEFAULT_SIM_MANIFEST_FILE_SHA256)
    parser.add_argument("--clean-manifest", type=Path,
                        default=DEFAULT_REAL_MANIFEST)
    parser.add_argument("--clean-manifest-sha256",
                        default=DEFAULT_REAL_MANIFEST_FILE_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wall-budget-seconds", type=float, default=30 * 60)
    args = parser.parse_args(argv)
    if args.pilot:
        args.stage = "pilot"
    args.pilot = args.stage == "pilot"
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.stage not in {"pilot", "quick"}:
        raise ValueError("unknown domain-joint stage")
    expected_held = PILOT_HELD_AOA if args.stage == "pilot" else QUICK_HELD_AOA
    expected_updates = (PILOT_DOMAIN_UPDATES if args.stage == "pilot"
                        else QUICK_DOMAIN_UPDATES)
    if args.held_aoa != expected_held:
        raise ValueError(f"Stage-{args.stage} is frozen to held AoA 10")
    if args.domain_updates != expected_updates:
        raise ValueError(
            f"Stage-{args.stage} is frozen to N={expected_updates}/domain")
    if args.num_workers != 0:
        raise ValueError("paired event replay requires num-workers=0")
    if args.expected_start_iteration <= 0:
        raise ValueError("expected start iteration must be positive")
    if not math.isfinite(args.wall_budget_seconds) or args.wall_budget_seconds <= 0:
        raise ValueError("wall budget must be finite and positive")
    args.start_sim_sha256 = dr1.normalized_sha256(
        args.start_sim_sha256, "start-sim-sha256")
    args.sim_manifest_sha256 = dr1.normalized_sha256(
        args.sim_manifest_sha256, "sim-manifest-sha256")
    args.clean_manifest_sha256 = dr1.normalized_sha256(
        args.clean_manifest_sha256, "clean-manifest-sha256")


def main() -> None:
    args = parse_args()
    result = run_pilot(args) if args.stage == "pilot" else run_quick(args)
    filename = "pilot_result.json" if args.stage == "pilot" else "quick_result.json"
    event = ("b210_domain_joint_stage_a_complete"
             if args.stage == "pilot"
             else "b210_domain_joint_quick_complete")
    print(json.dumps({
        "event": event,
        "status": result["status"],
        "output": str(args.output_dir / filename),
    }), flush=True)


if __name__ == "__main__":
    main()
