#!/usr/bin/env python3
"""Train the 81.354 B210 Sim recipe on dense released-Sim phases.

Only the optimizer-visible window inventory changes from stride 20 to stride
5.  The Sim channel normalizer and the physical hgrad scales are deliberately
computed from the exact stride-20 inventory used by the seed-0 81.354 lineage.
The P1 cache, in contrast, must cover and be byte-bound to every stride-5
optimizer window.

This is a thin, opt-in launcher around ``train_b210_np_functional_sim``.  It
does not change that trainer's default stride-20 behavior and it never enables
the separate Real-closure or architecture research arms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_sim_p1_point_teacher_cache as point_cache  # noqa: E402
import train_b210_np_functional_sim as b210  # noqa: E402


METHOD = "b210_np_functional_sim_seed0_dense_stride5_sparse_stats_v1"
SCHEMA = "b210_dense_sim_stride5_support_v1"
CACHE_SCHEMA = "b210_dense_sim_stride5_teacher_cache_v1"
OPTIMIZER_STRIDE = 5
STATISTICS_STRIDE = 20
SEED = 0
FORMAL_TRAJECTORIES = 100
FORMAL_OPTIMIZER_WINDOWS = 19_200
FORMAL_STATISTICS_WINDOWS = 4_800

# Frozen identities from the scratch seed-0 Sim10k endpoint that feeds the
# point model behind official submission 908872 / Final 81.354225.
REFERENCE_SIM_CHECKPOINT_SHA256 = (
    "8a6d9a1055269f95a308b42e73ee94b00b005edbf550430dfa096a610e2e0f29")
REFERENCE_STATISTICS_MANIFEST_SHA256 = (
    "8b1650f3de3b3b92b8555335bc4de0a4b9ff4828613a2539d6591966cf99bd36")
REFERENCE_OPTIMIZER_MANIFEST_SHA256 = (
    "556cf83ea932c2c7f2596aee39d1e023c95ab4a61a122486894dc4f2e50c721b")
REFERENCE_OPTIMIZER_WINDOW_ORDER_SHA256 = (
    "19dc3475445442b3f5850fd36bd71a4c3bdeda28217c2edd91db59c71caa2fe7")
REFERENCE_NORMALIZER_SHA256 = (
    "ab8065e45470fd4c24eadaecad647d0f62b557c3e19e9a235e6f4c6867c21c65")
REFERENCE_HGRAD_CONTRACT_SHA256 = (
    "8f2c73b4c443689f1158fb08e8b4d4a0ac5daeb24d6523a11f29ac1cd80130f3")
REFERENCE_TEACHER_SHA256 = {
    "a": "e5498ba59db11287a7ee480c69d9a114ef09fe1ef431a378ca94e1662fb1d791",
    "b": "83e9f6364272360e2026987add03ac443238a933cb7dd674a99bf05245e37c9b",
    "c": "5d00dc6fef25a782b2ce9fe69d1c84b16b63e968024525c54d456378d781f9a3",
}

REFERENCE_TRAINING_FIELDS = {
    "phase": "sim",
    "updates": 10_000,
    "start_checkpoint": None,
    "num_workers": 0,
    "batch_size": 8,
    "eval_batch_size": 16,
    "eval_interval": 500,
    "checkpoint_interval": 10_000,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-5,
    "grad_clip": 1.0,
    "max_files": 0,
    "window_stride": 20,
    "input_noise_std": 0.0,
    "input_gain_std": 0.0,
    "input_blur_probability": 0.0,
    "input_blur_strength": 0.0,
    "temporal_jitter_max": 0,
    "point_loss_mode": "official_accuracy",
    "mse_weight": 0.25,
    "mse_warmup_updates": 0,
    "mean_loss_weight": 1.0,
    "member_loss_weight": 0.25,
    "member_loss_mode": "mse",
    "bootstrap_probability": 0.75,
    "quantile_loss_weight": 0.05,
    "soft_sps_loss_weight": 0.05,
    "soft_sps_tau_fraction": 0.05,
    "soft_sps_warmup_updates": 1500,
    "soft_sps_uq_only": False,
    "soft_sps_proxy_real_scale": True,
    "coverage_bce_weight": 0.0,
    "profile_loss_weight": 0.0,
    "accuracy_weights": (1.0, 4.0, 1.0),
    "distill_weight": 0.005,
    "trainable_scope": "all",
}

SUPPORT_CONTRACT_NAME = "dense_sim_support_contract.json"
STATISTICS_MANIFEST_NAME = "statistics_split_manifest.json"
CACHE_CONTRACT_NAME = "dense_teacher_cache_contract.json"

DISALLOWED_RESEARCH_OPTIONS = (
    "phase_aligned_re_mix",
    "phase_aligned_parameter_grid_mix",
    "real_closure_equivariance",
    "multiscale_tke_deep_supervision",
    "multiscale_energy_controller",
    "horizon_local_energy_controller",
    "history_energy_controller",
    "direct_history_tke_regression",
    "hybrid_history_multiscale_energy_controller",
    "cross_conditioned_backbone",
    "temporal_point_decoder",
    "temporal_point_amplitude_factorization",
    "isoenergetic_temporal_transport",
    "temporal_band_mixer",
    "official_point_objective",
    "official_point_controller_tke_floor",
    "pareto_tke_projection",
    "legacy_tke_gradient_floor",
    "member_integrated_legacy_tke_gradient_floor",
    "member_official_objective",
    "decoupled_member_residual",
    "member_feature_adapters",
    "standard_objective_member_feature_adapters",
    "fluctuation_role_tower",
    "conditional_future_energy",
    "conditional_future_energy_members_only",
    "conditional_future_energy_adapter_only",
    "conditional_future_crps_mean_preserving",
    "mean_preserving_member_adapters",
    "detach_member_adapter_context",
    "detach_uq_context",
    "isolate_distribution_tower",
    "budgeted_disagreement_allocation",
)

_ORIGINAL_TRAINING_ARGS = b210.training_args
_ORIGINAL_VALIDATE_ARGS = b210.validate_args
_ORIGINAL_PARSE_ARGS = b210.parse_args
_ORIGINAL_PREPARE_SIM = b210.rehearsal.ensemble_train.common.prepare_sim
_ORIGINAL_HGRAD_CONTRACT = b210.sim_hgrad.compute_sim_hgrad_scale_contract
_ORIGINAL_LOAD_CACHE = b210.rehearsal.load_and_validate_cache
_INSTALLED = False


@dataclass(frozen=True)
class DenseSupportContext:
    output_dir: Path
    optimizer_manifest: list[dict]
    optimizer_manifest_sha256: str
    optimizer_window_order_sha256: str
    statistics_manifest: list[dict]
    statistics_manifest_sha256: str
    statistics_dataset: object
    normalizer_sha256: str
    support_contract: dict


def canonical_with_digest(payload: Mapping[str, object]) -> dict:
    result = dict(payload)
    result["canonical_sha256"] = b210.rehearsal.canonical_sha256(result)
    return result


def parse_args() -> argparse.Namespace:
    """Expose only paths and the explicit new-cache acknowledgement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--research-equivalent-cache-sha256", required=True)
    parser.add_argument("--acknowledge-research-cache", required=True)
    return parser.parse_args()


def manifest_for_stride(
        source_manifest: Sequence[Mapping[str, object]], stride: int,
        ) -> tuple[list, list[dict]]:
    """Rebuild one window inventory without reopening trajectory files."""
    if stride <= 0:
        raise ValueError("window stride must be positive")
    if not source_manifest:
        raise ValueError("Sim manifest must not be empty")
    windows = []
    manifest = []
    files = set()
    for trajectory, source in enumerate(source_manifest):
        row = dict(source)
        file_name = row.get("file")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("Sim manifest row has no file name")
        if file_name in files:
            raise ValueError("Sim manifest contains duplicate files")
        files.add(file_name)
        frames = int(row.get("frames", -1))
        if frames <= 40:
            raise ValueError("Sim trajectory is too short for a 20-to-20 window")
        starts = list(range(0, frames - 40, stride))
        if not starts:
            raise ValueError("Sim trajectory has no optimizer window")
        row["starts"] = starts
        manifest.append(row)
        windows.extend(
            b210.rehearsal.ensemble_train.common.scratch.SimWindow(
                trajectory, start)
            for start in starts)
    return windows, manifest


def window_keys(windows: Sequence) -> list[tuple[int, int]]:
    return [
        (int(window.trajectory), int(window.start)) for window in windows]


def validate_loaded_optimizer_inventory(
        windows: Sequence, loaded_manifest: Sequence[Mapping[str, object]],
        ) -> list[dict]:
    expected_windows, expected_manifest = manifest_for_stride(
        loaded_manifest, OPTIMIZER_STRIDE)
    if list(loaded_manifest) != expected_manifest:
        raise RuntimeError(
            "released-Sim loader did not return the exact stride-5 manifest")
    if window_keys(windows) != window_keys(expected_windows):
        raise RuntimeError(
            "released-Sim loader window order is not the stride-5 manifest order")
    return expected_manifest


def validate_dense_only_args(args: argparse.Namespace) -> None:
    """Reject every scientific option outside the 81.354 seed-0 Sim arm."""
    if args.num_workers != 0:
        raise ValueError("formal dense B210 Sim freezes num-workers=0")
    if getattr(args, "real_closure_bank", None) is not None:
        raise ValueError("dense B210 Sim is separate from Real-closure")
    if getattr(args, "real_closure_bank_sha256", None) is not None:
        raise ValueError("dense B210 Sim must not receive a closure-bank SHA")
    enabled = [
        option for option in DISALLOWED_RESEARCH_OPTIONS
        if bool(getattr(args, option, False))]
    if enabled:
        raise ValueError(
            "dense B210 Sim changes only optimizer support; disallowed: "
            + ", ".join(enabled))
    research_sha = getattr(args, "research_equivalent_cache_sha256", None)
    if research_sha is None:
        raise ValueError(
            "stride-5 B210 Sim requires an explicit new research cache SHA")
    if str(research_sha).lower() != str(args.teacher_cache_sha256).lower():
        raise ValueError("research cache SHA must equal teacher-cache-sha256")
    if (getattr(args, "acknowledge_research_cache", "")
            != b210.RESEARCH_CACHE_ACKNOWLEDGEMENT):
        raise ValueError("stride-5 cache requires the research-cache acknowledgement")
    for name in (
            "split_manifest.json", SUPPORT_CONTRACT_NAME, STATISTICS_MANIFEST_NAME,
            CACHE_CONTRACT_NAME):
        if (args.output_dir / name).exists():
            raise FileExistsError(
                f"refusing to overwrite dense B210 contract {name}")


def dense_validate_args(args: argparse.Namespace) -> None:
    validate_dense_only_args(args)
    _ORIGINAL_VALIDATE_ARGS(args)


def dense_training_args(args: argparse.Namespace) -> argparse.Namespace:
    result = _ORIGINAL_TRAINING_ARGS(args)
    mismatches = {
        name: {"expected": expected, "actual": getattr(result, name, None)}
        for name, expected in REFERENCE_TRAINING_FIELDS.items()
        if getattr(result, name, None) != expected
    }
    expected_globals = {
        "BACKBONE_DROPOUT": 0.20,
        "HGRAD_WEIGHT": 0.020,
        "HGRAD_RAMP_UPDATES": 200,
        "TKE_AUX_WEIGHT": 0.020,
        "TKE_AUX_UPDATES": 1500,
        "EXPECTED_PARAMETERS": 178_705_575,
        "EXPECTED_CONTROLLER_PARAMETERS": 7_169,
    }
    global_mismatches = {
        name: {"expected": expected, "actual": getattr(b210, name, None)}
        for name, expected in expected_globals.items()
        if getattr(b210, name, None) != expected
    }
    if mismatches or global_mismatches:
        raise RuntimeError(
            "base B210 seed-0 Sim contract changed: "
            f"training={mismatches}, globals={global_mismatches}")
    result.seed = SEED
    result.window_stride = OPTIMIZER_STRIDE
    result.method = METHOD
    return result


def _normalizer_values(normalizer) -> dict:
    return b210.rehearsal.ensemble_train.common.normalizer_dict(normalizer)


def prepare_dense_sim(args: argparse.Namespace, device: torch.device):
    """Build stride-5 optimizer data and stride-20-only statistics data."""
    if (args.seed != SEED or args.window_stride != OPTIMIZER_STRIDE
            or args.max_files != 0 or args.temporal_jitter_max != 0):
        raise RuntimeError("dense B210 sampling arguments changed")
    scratch = b210.rehearsal.ensemble_train.common.scratch
    trajectories, optimizer_windows, loaded_manifest = scratch.load_simulations(
        args.sim_dir, args.max_files, OPTIMIZER_STRIDE, True)
    optimizer_manifest = validate_loaded_optimizer_inventory(
        optimizer_windows, loaded_manifest)
    statistics_windows, statistics_manifest = manifest_for_stride(
        optimizer_manifest, STATISTICS_STRIDE)
    if len(trajectories) != FORMAL_TRAJECTORIES:
        raise RuntimeError(
            f"dense Sim trajectories={len(trajectories)}, expected "
            f"{FORMAL_TRAJECTORIES}")
    if len(optimizer_windows) != FORMAL_OPTIMIZER_WINDOWS:
        raise RuntimeError(
            f"dense Sim windows={len(optimizer_windows)}, expected "
            f"{FORMAL_OPTIMIZER_WINDOWS}")
    if len(statistics_windows) != FORMAL_STATISTICS_WINDOWS:
        raise RuntimeError(
            f"statistics windows={len(statistics_windows)}, expected "
            f"{FORMAL_STATISTICS_WINDOWS}")

    statistics = scratch.window_statistics(trajectories, statistics_windows)
    normalizer = scratch.ChannelNormalizer(*statistics, device=device)
    normalizer_values = _normalizer_values(normalizer)
    normalizer_sha = b210.rehearsal.canonical_sha256(normalizer_values)
    statistics_manifest_sha = b210.rehearsal.canonical_sha256(
        statistics_manifest)
    if statistics_manifest_sha != REFERENCE_STATISTICS_MANIFEST_SHA256:
        raise RuntimeError(
            "stride-20 statistics manifest differs from the 81.354 Sim anchor")
    if normalizer_sha != REFERENCE_NORMALIZER_SHA256:
        raise RuntimeError(
            "stride-20 normalizer differs from the 81.354 Sim anchor")

    optimizer_dataset = scratch.SimWindowDataset(
        trajectories, optimizer_windows, mask_probability=0.0,
        noise_scale=0.0)
    statistics_dataset = scratch.SimWindowDataset(
        trajectories, statistics_windows, mask_probability=0.0,
        noise_scale=0.0)
    optimizer_loader = DataLoader(
        optimizer_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(SEED))

    optimizer_manifest_sha = b210.rehearsal.canonical_sha256(
        optimizer_manifest)
    optimizer_order_sha = b210.rehearsal.canonical_sha256(
        point_cache.ordered_window_keys(optimizer_windows, optimizer_manifest))
    if optimizer_manifest_sha != REFERENCE_OPTIMIZER_MANIFEST_SHA256:
        raise RuntimeError(
            "stride-5 optimizer manifest differs from the frozen inventory")
    if optimizer_order_sha != REFERENCE_OPTIMIZER_WINDOW_ORDER_SHA256:
        raise RuntimeError(
            "stride-5 optimizer window order differs from the frozen inventory")
    support_contract = canonical_with_digest({
        "schema": SCHEMA,
        "method": METHOD,
        "reference_sim_checkpoint_sha256": (
            REFERENCE_SIM_CHECKPOINT_SHA256),
        "seed": SEED,
        "trajectories": len(trajectories),
        "optimizer_window_stride": OPTIMIZER_STRIDE,
        "optimizer_windows": len(optimizer_windows),
        "optimizer_manifest_sha256": optimizer_manifest_sha,
        "optimizer_window_order_sha256": optimizer_order_sha,
        "statistics_window_stride": STATISTICS_STRIDE,
        "statistics_windows": len(statistics_windows),
        "statistics_manifest_sha256": statistics_manifest_sha,
        "normalizer_source": "released_sim_stride20_inventory",
        "normalizer_sha256": normalizer_sha,
        "hgrad_scale_source": "released_sim_stride20_inventory",
        "expected_hgrad_contract_sha256": (
            REFERENCE_HGRAD_CONTRACT_SHA256),
        "teacher_cache_source": "released_sim_stride5_optimizer_inventory",
        "real_closure_enabled": False,
        "external_data_or_model": False,
    })
    context = DenseSupportContext(
        output_dir=args.output_dir,
        optimizer_manifest=optimizer_manifest,
        optimizer_manifest_sha256=optimizer_manifest_sha,
        optimizer_window_order_sha256=optimizer_order_sha,
        statistics_manifest=statistics_manifest,
        statistics_manifest_sha256=statistics_manifest_sha,
        statistics_dataset=statistics_dataset,
        normalizer_sha256=normalizer_sha,
        support_contract=support_contract,
    )
    setattr(optimizer_dataset, "_b210_dense_support_context", context)
    b210.write_json(args.output_dir / "split_manifest.json", optimizer_manifest)
    b210.write_json(
        args.output_dir / STATISTICS_MANIFEST_NAME, statistics_manifest)
    b210.write_json(
        args.output_dir / SUPPORT_CONTRACT_NAME, support_contract)
    return (
        optimizer_loader, normalizer, len(optimizer_windows),
        len(trajectories))


def dense_support_context(dataset) -> DenseSupportContext:
    context = getattr(dataset, "_b210_dense_support_context", None)
    if not isinstance(context, DenseSupportContext):
        raise RuntimeError("dataset is not the formal dense B210 inventory")
    return context


def compute_stride20_hgrad_contract(
        optimizer_dataset, optimizer_manifest_sha256: str,
        normalizer_sha256: str) -> dict:
    """Compute hgrad scales from sparse statistics, never dense targets."""
    context = dense_support_context(optimizer_dataset)
    if optimizer_manifest_sha256 != context.optimizer_manifest_sha256:
        raise RuntimeError("hgrad call is not bound to the dense optimizer manifest")
    if normalizer_sha256 != context.normalizer_sha256:
        raise RuntimeError("hgrad call is not bound to the sparse normalizer")
    contract = _ORIGINAL_HGRAD_CONTRACT(
        context.statistics_dataset,
        context.statistics_manifest_sha256,
        context.normalizer_sha256)
    if contract.get("train_windows") != FORMAL_STATISTICS_WINDOWS:
        raise RuntimeError("hgrad scale did not use all stride-20 windows")
    if (contract.get("split_manifest_sha256")
            != REFERENCE_STATISTICS_MANIFEST_SHA256):
        raise RuntimeError("hgrad scale used the wrong statistics manifest")
    if contract.get("sim_normalizer_sha256") != REFERENCE_NORMALIZER_SHA256:
        raise RuntimeError("hgrad scale used the wrong Sim normalizer")
    if (contract.get("canonical_sha256")
            != REFERENCE_HGRAD_CONTRACT_SHA256):
        raise RuntimeError("hgrad scale differs from the 81.354 Sim anchor")
    return contract


def load_dense_teacher_cache(path: Path, manifest: list[dict], dataset):
    """Load once, then bind the cache header and tensors to stride-5 keys."""
    context = dense_support_context(dataset)
    if manifest != context.optimizer_manifest:
        raise RuntimeError("teacher cache received a different optimizer manifest")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("teacher cache payload must be a mapping")
    if int(payload.get("window_stride", -1)) != OPTIMIZER_STRIDE:
        raise ValueError("teacher cache header is not stride 5")
    if payload.get("exclude_final_window") is not True:
        raise ValueError("teacher cache must use the excluded-final convention")
    if int(payload.get("max_files", -1)) != 0:
        raise ValueError("teacher cache must cover the full released inventory")
    prediction, metadata = point_cache.validate_cache_payload(
        payload, manifest, dataset)
    actual_sha = point_cache.sha256(path)
    metadata = {
        **metadata,
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "window_stride": OPTIMIZER_STRIDE,
        "exclude_final_window": True,
        "max_files": 0,
    }
    if metadata["windows"] != FORMAL_OPTIMIZER_WINDOWS:
        raise RuntimeError("teacher cache does not cover all stride-5 windows")
    if (metadata["split_manifest_sha256"]
            != context.optimizer_manifest_sha256):
        raise RuntimeError("teacher cache manifest differs from optimizer support")
    if (metadata["window_order_sha256"]
            != context.optimizer_window_order_sha256):
        raise RuntimeError("teacher cache window order differs from optimizer support")
    identities = metadata.get("teacher_checkpoints")
    for name, expected in REFERENCE_TEACHER_SHA256.items():
        identity = identities.get(name) if isinstance(identities, dict) else None
        if (not isinstance(identity, dict)
                or identity.get("actual_sha256") != expected
                or identity.get("expected_sha256") != expected
                or identity.get("phase") != "sim"
                or identity.get("sha256_match") is not True):
            raise RuntimeError(
                f"teacher cache source {name} differs from the 81.354 P1 source")
    cache_contract = canonical_with_digest({
        "schema": CACHE_SCHEMA,
        "method": METHOD,
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "window_stride": OPTIMIZER_STRIDE,
        "windows": metadata["windows"],
        "split_manifest_sha256": metadata["split_manifest_sha256"],
        "window_order_sha256": metadata["window_order_sha256"],
        "prediction_shape": metadata["prediction_shape"],
        "prediction_dtype": metadata["prediction_dtype"],
        "teacher_checkpoints": metadata["teacher_checkpoints"],
    })
    b210.write_json(
        context.output_dir / CACHE_CONTRACT_NAME, cache_contract)
    del payload
    return prediction, metadata


def install_dense_contract() -> None:
    """Install process-local hooks; importing this module changes nothing."""
    global _INSTALLED
    if _INSTALLED:
        return
    if b210.training_args is not _ORIGINAL_TRAINING_ARGS:
        raise RuntimeError("B210 training_args was already replaced")
    if b210.parse_args is not _ORIGINAL_PARSE_ARGS:
        raise RuntimeError("B210 parse_args was already replaced")
    if b210.validate_args is not _ORIGINAL_VALIDATE_ARGS:
        raise RuntimeError("B210 validate_args was already replaced")
    if (b210.rehearsal.ensemble_train.common.prepare_sim
            is not _ORIGINAL_PREPARE_SIM):
        raise RuntimeError("B210 prepare_sim was already replaced")
    if (b210.sim_hgrad.compute_sim_hgrad_scale_contract
            is not _ORIGINAL_HGRAD_CONTRACT):
        raise RuntimeError("B210 hgrad contract builder was already replaced")
    if b210.rehearsal.load_and_validate_cache is not _ORIGINAL_LOAD_CACHE:
        raise RuntimeError("B210 teacher-cache loader was already replaced")
    b210.SEED = SEED
    b210.METHOD = METHOD
    b210.parse_args = parse_args
    b210.validate_args = dense_validate_args
    b210.training_args = dense_training_args
    b210.rehearsal.ensemble_train.common.prepare_sim = prepare_dense_sim
    b210.sim_hgrad.compute_sim_hgrad_scale_contract = (
        compute_stride20_hgrad_contract)
    b210.rehearsal.load_and_validate_cache = load_dense_teacher_cache
    _INSTALLED = True


def main() -> None:
    install_dense_contract()
    b210.main()


if __name__ == "__main__":
    main()
