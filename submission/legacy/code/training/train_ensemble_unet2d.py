#!/usr/bin/env python3
"""Train the shared-backbone ensemble U-Net and residual quantile decoder."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import time

import numpy as np
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (CODE_ROOT, Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from build_sim_p1_point_teacher_cache import (  # noqa: E402
    load_and_validate_cache,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sim", "real"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument(
        "--teacher-cache", type=Path,
        help="Current-P1 A/B/C point cache for optional Sim distillation.")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--baseline-split-manifest", type=Path,
        help=("Freeze Real val_starts/val_boundary from this baseline manifest "
              "while rebuilding optimizer starts at --window-stride."))
    parser.add_argument(
        "--real-all-windows", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--input-noise-std", type=float, default=0.0,
        help=("Training-only Gaussian noise in normalized u/v units; "
              "targets, pressure, and solid-body pixels remain unchanged."))
    parser.add_argument(
        "--input-gain-std", type=float, default=0.0,
        help=("Training-only sample/channel velocity gain jitter in physical "
              "units; targets, pressure, and solid-body pixels are unchanged."))
    parser.add_argument(
        "--input-blur-probability", type=float, default=0.0,
        help=("Probability per sample of applying a mask-normalized 3x3 "
              "Gaussian blur to normalized input u/v only."))
    parser.add_argument(
        "--input-blur-strength", type=float, default=0.0,
        help=("Interpolation strength between original and blurred input "
              "velocity; must lie in [0,1]."))
    parser.add_argument(
        "--temporal-jitter-max", type=int, default=0,
        help=("Sim-only random non-negative shift in released frames for "
              "each training window."))
    parser.add_argument(
        "--sim-audit-variant", choices=("control", "candidate"),
        help=("Write the formal C208-DR1 Sim stream/model attestation. "
              "Requires the frozen seed-4, stride-10, 10000-update contract."))

    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--unet-depth", type=int, default=4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--difference-steps", type=int, default=5)
    parser.add_argument(
        "--use-coordinates", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--use-regime-film", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--multi-scale-regime-film",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mean-fluctuation", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--ensemble-members", type=int, default=4)
    parser.add_argument("--uq-hidden-channels", type=int, default=64)
    parser.add_argument("--member-adapter-channels", type=int, default=0)
    parser.add_argument("--mean-profile-channels", type=int, default=0)
    parser.add_argument(
        "--uq-flow-context", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--residual-quantile-intervals",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--utility-refinement",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--utility-refinement-log-limit", type=float, default=0.75)
    parser.add_argument(
        "--utility-refinement-kernel", type=int, default=1)
    parser.add_argument(
        "--utility-refinement-blocks", type=int, default=0)
    parser.add_argument(
        "--temporal-uq-refinement",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-uq-channels", type=int, default=24)
    parser.add_argument("--temporal-uq-blocks", type=int, default=2)
    parser.add_argument("--temporal-uq-kernel", type=int, default=3)
    parser.add_argument(
        "--temporal-uq-logit-limit", type=float, default=1.5)
    parser.add_argument(
        "--coverage-calibration",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--coverage-calibration-kernel", type=int, default=3)
    parser.add_argument(
        "--coverage-calibration-blocks", type=int, default=1)
    parser.add_argument(
        "--coverage-calibration-logit-limit", type=float, default=2.0)
    parser.add_argument(
        "--coverage-calibration-mode",
        choices=("independent", "temperature"), default="independent")
    parser.add_argument(
        "--coverage-calibration-temperature-limit",
        type=float, default=0.75)
    parser.add_argument("--quantile-gap-bias", type=float, default=-4.0)
    parser.add_argument(
        "--disagreement-scale-fraction", type=float, default=0.01)

    parser.add_argument(
        "--point-loss-mode", choices=("mse", "official_accuracy"),
        default="official_accuracy")
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--mse-warmup-updates", type=int, default=0)
    parser.add_argument("--mean-loss-weight", type=float, default=1.0)
    parser.add_argument("--member-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--member-loss-mode", choices=("mse", "point"), default="mse")
    parser.add_argument("--bootstrap-probability", type=float, default=0.75)
    parser.add_argument("--quantile-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--soft-sps-loss-weight", type=float, default=0.0,
        help=("Jointly train native interval extents with a differentiable "
              "approximation of the official SPS utility."))
    parser.add_argument(
        "--soft-sps-tau-fraction", type=float, default=0.05)
    parser.add_argument(
        "--soft-sps-warmup-updates", type=int, default=0,
        help="Linearly ramp the joint SPS loss over this many updates.")
    parser.add_argument(
        "--soft-sps-uq-only",
        action=argparse.BooleanOptionalAction, default=False,
        help=("Backpropagate the SPS utility only through the distribution "
              "decoder, preserving the point/backbone optimization path."))
    parser.add_argument(
        "--soft-sps-proxy-real-scale",
        action=argparse.BooleanOptionalAction, default=True,
        help=("During Sim pretraining, map normalized Sim residuals onto the "
              "released Real u/v scale before applying the SPS utility."))
    parser.add_argument("--coverage-bce-weight", type=float, default=0.0)
    parser.add_argument("--profile-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--allow-real-start", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--trainable-scope", choices=("all", "mean_profile_refiner"),
        default="all")
    parser.add_argument(
        "--accuracy-weights", type=float, nargs=3,
        metavar=("REL", "TKE", "MVPE"), default=(5.0, 3.0, 2.0))
    parser.add_argument(
        "--selection-metric",
        choices=(
            "point_score", "native_sps_score", "mvpe_rel_l2_score"),
        default="native_sps_score")
    return parser.parse_args()


def model_config(args: argparse.Namespace) -> EnsembleUNet2dConfig:
    return EnsembleUNet2dConfig(
        base=args.base,
        depth=args.unet_depth,
        groups=args.groups,
        difference_steps=args.difference_steps,
        use_coordinates=args.use_coordinates,
        use_regime_film=args.use_regime_film,
        multi_scale_regime_film=args.multi_scale_regime_film,
        mean_fluctuation=args.mean_fluctuation,
        ensemble_members=args.ensemble_members,
        uq_hidden_channels=args.uq_hidden_channels,
        member_adapter_channels=args.member_adapter_channels,
        mean_profile_channels=args.mean_profile_channels,
        uq_flow_context=args.uq_flow_context,
        residual_quantile_intervals=args.residual_quantile_intervals,
        utility_refinement=args.utility_refinement,
        utility_refinement_log_limit=args.utility_refinement_log_limit,
        utility_refinement_kernel=args.utility_refinement_kernel,
        utility_refinement_blocks=args.utility_refinement_blocks,
        temporal_uq_refinement=args.temporal_uq_refinement,
        temporal_uq_channels=args.temporal_uq_channels,
        temporal_uq_blocks=args.temporal_uq_blocks,
        temporal_uq_kernel=args.temporal_uq_kernel,
        temporal_uq_logit_limit=args.temporal_uq_logit_limit,
        coverage_calibration=args.coverage_calibration,
        coverage_calibration_kernel=args.coverage_calibration_kernel,
        coverage_calibration_blocks=args.coverage_calibration_blocks,
        coverage_calibration_logit_limit=(
            args.coverage_calibration_logit_limit),
        coverage_calibration_mode=args.coverage_calibration_mode,
        coverage_calibration_temperature_limit=(
            args.coverage_calibration_temperature_limit),
        quantile_gap_bias=args.quantile_gap_bias,
        disagreement_scale_fraction=args.disagreement_scale_fraction,
    )


def canonical_sha256(value) -> str:
    """Hash JSON-compatible evidence using a stable canonical encoding."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_state_sha256(state_dict) -> str:
    """Hash tensor state independent of ``torch.save`` container details."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<q", int(dimension)))
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_fingerprint(device: torch.device) -> dict[str, object]:
    """Record the formal CUDA execution identity without touching RNG state."""
    if device.type != "cuda":
        return {
            "device_type": device.type,
            "torch_version": torch.__version__,
        }
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    smi = subprocess.check_output([
        "nvidia-smi", "-i", str(index),
        "--query-gpu=uuid,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",", maxsplit=1)
    if len(smi) != 2:
        raise RuntimeError("could not parse nvidia-smi CUDA identity")
    return {
        "device_type": "cuda",
        "device_index": int(index),
        "gpu_uuid": smi[0].strip(),
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "driver_version": smi[1].strip(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


class SimStreamAudit:
    """Observe actual sample/shift streams without drawing random numbers."""

    def __init__(self):
        self._base_samples = hashlib.sha256()
        self._jitter_shifts = hashlib.sha256()
        self.samples = 0

    def record(self, sample_index: int, jitter_shift: int) -> None:
        self._base_samples.update(struct.pack("<q", int(sample_index)))
        self._jitter_shifts.update(struct.pack("<q", int(jitter_shift)))
        self.samples += 1

    @property
    def base_sample_stream_sha256(self) -> str:
        return self._base_samples.hexdigest()

    @property
    def jitter_shift_stream_sha256(self) -> str:
        return self._jitter_shifts.hexdigest()


def augmentation_config(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "temporal_jitter_max": int(args.temporal_jitter_max),
        "input_gain_std": float(args.input_gain_std),
        "input_noise_std": float(args.input_noise_std),
        "input_blur_probability": float(args.input_blur_probability),
        "input_blur_strength": float(args.input_blur_strength),
    }


def json_training_config(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def validate_sim_audit_contract(args: argparse.Namespace) -> None:
    if args.phase != "sim":
        raise ValueError("formal Sim audit is restricted to the Sim phase")
    if args.teacher_cache is not None or args.distill_weight != 0.0:
        raise ValueError("formal Sim audit requires no teacher cache/distillation")
    if (args.seed, args.window_stride, args.updates) != (4, 10, 10000):
        raise ValueError(
            "formal Sim audit requires seed=4, window_stride=10, updates=10000")
    if args.num_workers != 0:
        raise ValueError("formal Sim stream audit requires num_workers=0")
    if torch.device(args.device).type != "cuda":
        raise ValueError("formal Sim audit requires a CUDA device")
    expected = {
        "control": {
            "temporal_jitter_max": 0,
            "input_gain_std": 0.0,
            "input_noise_std": 0.0,
            "input_blur_probability": 0.0,
            "input_blur_strength": 0.0,
        },
        "candidate": {
            "temporal_jitter_max": 9,
            "input_gain_std": 0.02,
            "input_noise_std": 0.02,
            "input_blur_probability": 0.25,
            "input_blur_strength": 0.25,
        },
    }[args.sim_audit_variant]
    if augmentation_config(args) != expected:
        raise ValueError(
            f"formal {args.sim_audit_variant} augmentation mismatch: "
            f"expected={expected} actual={augmentation_config(args)}")


def validate(args: argparse.Namespace) -> tuple[float, float, float]:
    input_gain_std = float(getattr(args, "input_gain_std", 0.0))
    input_blur_probability = float(getattr(
        args, "input_blur_probability", 0.0))
    input_blur_strength = float(getattr(args, "input_blur_strength", 0.0))
    temporal_jitter_max = int(getattr(args, "temporal_jitter_max", 0))
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.checkpoint_interval, args.lr,
        args.window_stride)
    if any(value <= 0 for value in positive):
        raise ValueError("training counts, learning rate and stride must be positive")
    nonnegative = (
        args.weight_decay, args.grad_clip, args.mse_weight,
        args.mean_loss_weight,
        args.member_loss_weight, args.quantile_loss_weight,
        args.profile_loss_weight, args.soft_sps_loss_weight,
        args.coverage_bce_weight, args.distill_weight,
        args.input_noise_std, input_gain_std)
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("loss and optimizer values must be finite/non-negative")
    if not 0.0 < args.bootstrap_probability <= 1.0:
        raise ValueError("bootstrap_probability must lie in (0,1]")
    if (not math.isfinite(args.soft_sps_tau_fraction)
            or args.soft_sps_tau_fraction <= 0):
        raise ValueError("soft SPS tau fraction must be finite/positive")
    if (not math.isfinite(input_blur_probability)
            or not 0.0 <= input_blur_probability <= 1.0):
        raise ValueError("input blur probability must lie in [0,1]")
    if (not math.isfinite(input_blur_strength)
            or not 0.0 <= input_blur_strength <= 1.0):
        raise ValueError("input blur strength must lie in [0,1]")
    if bool(input_blur_probability) != bool(input_blur_strength):
        raise ValueError(
            "input blur probability and strength must be enabled together")
    if temporal_jitter_max < 0:
        raise ValueError("temporal jitter must be non-negative")
    if args.phase != "sim" and temporal_jitter_max:
        raise ValueError("temporal jitter is restricted to the Sim phase")
    if args.coverage_bce_weight and not args.coverage_calibration:
        raise ValueError(
            "coverage BCE weight requires --coverage-calibration")
    if not 0 <= args.mse_warmup_updates <= args.updates:
        raise ValueError("MSE warmup must lie within the run")
    if not 0 <= args.soft_sps_warmup_updates <= args.updates:
        raise ValueError("soft SPS warmup must lie within the run")
    if args.phase == "sim" and args.start_checkpoint is not None:
        raise ValueError("Sim phase must start from random initialization")
    if args.phase == "real" and args.start_checkpoint is None:
        raise ValueError("Real phase requires a Sim checkpoint")
    if args.phase != "real" and args.real_all_windows:
        raise ValueError("real_all_windows is only valid in the Real phase")
    baseline_split_manifest = getattr(args, "baseline_split_manifest", None)
    if baseline_split_manifest is not None and args.phase != "real":
        raise ValueError("baseline split manifest is only valid in the Real phase")
    if baseline_split_manifest is not None and args.real_all_windows:
        raise ValueError(
            "frozen validation cannot be combined with --real-all-windows")
    if args.phase != "real" and args.allow_real_start:
        raise ValueError("allow_real_start is only valid in the Real phase")
    if args.phase != "real" and args.trainable_scope != "all":
        raise ValueError("restricted trainable scope is only valid for Real")
    if (args.teacher_cache is None) != (args.distill_weight == 0.0):
        raise ValueError(
            "teacher cache and positive distill weight must be used together")
    if args.teacher_cache is not None and args.phase != "sim":
        raise ValueError("point-cache distillation is only supported in Sim")
    if args.teacher_cache is not None and temporal_jitter_max:
        raise ValueError(
            "temporal jitter cannot use a cache indexed by unshifted windows")
    if (args.trainable_scope == "mean_profile_refiner"
            and args.mean_profile_channels <= 0):
        raise ValueError(
            "mean_profile_refiner scope requires mean-profile-channels > 0")
    weights = tuple(float(value) for value in args.accuracy_weights)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("accuracy weights must be finite/non-negative")
    if not any(weights):
        raise ValueError("at least one accuracy weight must be positive")
    model_config(args).validate()
    if args.sim_audit_variant is not None:
        validate_sim_audit_contract(args)
    return weights


def build_or_load(
        args: argparse.Namespace, device: torch.device) -> EnsembleUNet2d:
    requested = model_config(args)
    if args.phase == "sim":
        return EnsembleUNet2d(requested).to(device)
    payload = torch.load(
        args.start_checkpoint, map_location="cpu", weights_only=False)
    start_phase = payload.get("phase")
    if start_phase != "sim" and not (
            args.allow_real_start and start_phase == "real"):
        raise ValueError(
            "Real phase must load a Sim checkpoint unless "
            "--allow-real-start is set for a Real continuation")
    stored = dict(payload["model_config"])
    if stored.pop("architecture") != "ensemble_unet":
        raise ValueError("start checkpoint is not an ensemble_unet")
    config = EnsembleUNet2dConfig(**stored)
    if config != requested:
        raise ValueError(
            "Real architecture flags must exactly match the Sim checkpoint")
    model = EnsembleUNet2d(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_training_input_noise(
        history_normalized: torch.Tensor,
        history_physical: torch.Tensor, noise_std: float,
        ) -> torch.Tensor:
    """Perturb normalized fluid u/v only; return by identity when disabled."""
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("input noise std must be finite and non-negative")
    if noise_std == 0.0:
        return history_normalized
    if (history_normalized.shape != history_physical.shape
            or history_normalized.shape[-1] < 2):
        raise ValueError("normalized and physical histories are not aligned")
    fluid = history_physical[..., :2].abs().sum(
        dim=-1, keepdim=True).ne(0.0)
    perturbed = history_normalized.clone()
    perturbed[..., :2].add_(
        noise_std * torch.randn_like(perturbed[..., :2])
        * fluid.to(perturbed.dtype))
    return perturbed


def add_training_input_gain(
        history_physical: torch.Tensor, gain_std: float,
        ) -> torch.Tensor:
    """Apply sample/channel velocity gain jitter without changing the source."""
    if not math.isfinite(gain_std) or gain_std < 0.0:
        raise ValueError("input gain std must be finite and non-negative")
    if gain_std == 0.0:
        return history_physical
    if history_physical.shape[-1] < 2:
        raise ValueError("physical history must contain u/v channels")
    result = history_physical.clone()
    gain = 1.0 + gain_std * torch.randn(
        history_physical.shape[0], 1, 1, 1, 2,
        device=history_physical.device, dtype=history_physical.dtype)
    result[..., :2].mul_(gain)
    return result


def masked_velocity_blur(
        history_normalized: torch.Tensor,
        history_physical: torch.Tensor,
        probability: float, strength: float,
        ) -> torch.Tensor:
    """Apply sample-local low-pass augmentation without solid-body leakage."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("input blur probability must lie in [0,1]")
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("input blur strength must lie in [0,1]")
    if bool(probability) != bool(strength):
        raise ValueError(
            "input blur probability and strength must be enabled together")
    if probability == 0.0:
        return history_normalized
    if (history_normalized.shape != history_physical.shape
            or history_normalized.shape[-1] < 2):
        raise ValueError("normalized and physical histories are not aligned")

    batch, steps, height, width, _channels = history_normalized.shape
    velocity = history_normalized[..., :2].permute(
        0, 1, 4, 2, 3).reshape(batch * steps, 2, height, width)
    fluid = history_physical[..., :2].abs().sum(
        dim=-1, keepdim=True).ne(0.0)
    mask = fluid.expand(-1, -1, -1, -1, 2).permute(
        0, 1, 4, 2, 3).reshape(batch * steps, 2, height, width)
    mask = mask.to(dtype=velocity.dtype)
    kernel = velocity.new_tensor((
        (1.0, 2.0, 1.0),
        (2.0, 4.0, 2.0),
        (1.0, 2.0, 1.0),
    )).div_(16.0).view(1, 1, 3, 3).expand(2, -1, -1, -1)
    numerator = F.conv2d(velocity * mask, kernel, padding=1, groups=2)
    denominator = F.conv2d(mask, kernel, padding=1, groups=2).clamp_min_(1e-6)
    blurred = numerator / denominator
    selected = torch.rand(
        batch, 1, 1, 1, 1, device=history_normalized.device
    ).lt(probability)
    selected = selected.expand(-1, steps, 2, height, width).reshape(
        batch * steps, 2, height, width)
    augmented = velocity + strength * selected * mask * (blurred - velocity)
    result = history_normalized.clone()
    result[..., :2] = augmented.reshape(
        batch, steps, 2, height, width).permute(0, 1, 3, 4, 2)
    return result


class TemporalJitterDataset(Dataset):
    """Shift fixed Sim windows onto nearby released-frame time origins."""
    def __init__(self, base_dataset, max_shift: int,
                 audit: SimStreamAudit | None = None):
        if max_shift < 0:
            raise ValueError("temporal jitter must be non-negative")
        required = ("windows", "trajectories")
        if any(not hasattr(base_dataset, name) for name in required):
            raise TypeError("temporal jitter requires Sim window trajectories")
        self.base_dataset = base_dataset
        self.windows = base_dataset.windows
        self.trajectories = base_dataset.trajectories
        self.max_shift = int(max_shift)
        self.audit = audit

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        window = self.windows[index]
        trajectory = self.trajectories[window.trajectory]
        available = max(0, trajectory.shape[0] - 40 - window.start)
        maximum = min(self.max_shift, available)
        shift = int(torch.randint(maximum + 1, ()).item()) if maximum else 0
        start = window.start + shift
        data = trajectory[start:start + 40].clone()
        if self.audit is not None:
            self.audit.record(index, shift)
        return data[:20], data[20:]


class TrajectoryTaggedDataset(Dataset):
    def __init__(self, base_dataset, audit: SimStreamAudit | None = None):
        if not hasattr(base_dataset, "windows"):
            raise TypeError("training dataset must expose window trajectory IDs")
        self.base_dataset = base_dataset
        self.audit = audit
        self.base_records_audit = isinstance(base_dataset, TemporalJitterDataset)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        history, target = self.base_dataset[index]
        if self.audit is not None and not self.base_records_audit:
            self.audit.record(index, 0)
        trajectory = int(self.base_dataset.windows[index].trajectory)
        return history, target, trajectory, index


def tag_training_loader(loader, args: argparse.Namespace,
                        audit: SimStreamAudit | None = None):
    tagged = TrajectoryTaggedDataset(loader.dataset, audit)
    return DataLoader(
        tagged, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))


def rebuild_real_split_from_frozen_manifest(
        metadata: list[dict], frozen_manifest: list[dict],
        window_stride: int,
        ) -> tuple[list, list, list[dict]]:
    """Rebuild Real optimizer starts while preserving frozen validation keys."""
    if window_stride <= 0:
        raise ValueError("window stride must be positive")
    if not isinstance(frozen_manifest, list):
        raise TypeError("baseline split manifest must be a list")
    if len(frozen_manifest) != len(metadata):
        raise ValueError(
            "baseline split manifest metadata count does not match Real data")

    split_fields = {"train_starts", "val_starts", "val_boundary"}
    train_windows = []
    val_windows = []
    rebuilt_manifest = []
    train_keys = set()
    val_keys = set()
    window_frames = 40

    for trajectory, (item, frozen) in enumerate(zip(metadata, frozen_manifest)):
        if not isinstance(item, dict) or not isinstance(frozen, dict):
            raise TypeError("Real metadata and split entries must be objects")
        frozen_metadata = {
            key: value for key, value in frozen.items()
            if key not in split_fields
        }
        if frozen_metadata != item:
            raise ValueError(
                f"baseline split manifest metadata mismatch at trajectory "
                f"{trajectory}")

        frames = item.get("frames")
        if not isinstance(frames, int) or isinstance(frames, bool):
            raise TypeError(f"invalid frame count for {item.get('file', trajectory)}")
        if frames < window_frames:
            raise ValueError(f"trajectory {item.get('file', trajectory)} is too short")
        val_starts = frozen.get("val_starts")
        if not isinstance(val_starts, list) or not val_starts:
            raise ValueError(
                f"frozen validation starts are missing for "
                f"{item.get('file', trajectory)}")
        if any(not isinstance(start, int) or isinstance(start, bool)
               for start in val_starts):
            raise TypeError("frozen validation starts must be integer lists")
        if len(set(val_starts)) != len(val_starts):
            raise ValueError("frozen validation starts contain duplicate keys")
        if any(start < 0 or start + window_frames > frames
               for start in val_starts):
            raise ValueError("frozen validation window lies outside its trajectory")
        val_boundary = frozen.get("val_boundary")
        if not isinstance(val_boundary, int) or isinstance(val_boundary, bool):
            raise TypeError("frozen validation boundary must be an integer")

        all_starts = range(0, frames - window_frames + 1, window_stride)
        train_starts = [
            start for start in all_starts
            if start + window_frames <= val_boundary
        ]
        if not train_starts:
            raise ValueError(
                f"no optimizer windows remain for {item.get('file', trajectory)}")

        trajectory_train_keys = {
            (trajectory, start) for start in train_starts}
        trajectory_val_keys = {
            (trajectory, start) for start in val_starts}
        if trajectory_train_keys & trajectory_val_keys:
            raise ValueError(
                f"optimizer/frozen-validation key intersection for "
                f"{item.get('file', trajectory)}")
        overlap = next((
            (train_start, val_start)
            for train_start in train_starts
            for val_start in val_starts
            if (train_start < val_start + window_frames
                and val_start < train_start + window_frames)
        ), None)
        if overlap is not None:
            raise ValueError(
                f"optimizer interval overlaps frozen validation for "
                f"{item.get('file', trajectory)}: {overlap}")

        train_keys.update(trajectory_train_keys)
        val_keys.update(trajectory_val_keys)
        train_windows.extend(
            common.repro.Window(trajectory, start) for start in train_starts)
        val_windows.extend(
            common.repro.Window(trajectory, start) for start in val_starts)
        rebuilt_manifest.append({
            **item,
            "train_starts": train_starts,
            "val_starts": list(val_starts),
            "val_boundary": val_boundary,
        })

    if train_keys & val_keys:
        raise ValueError("optimizer/frozen-validation key intersection is not zero")
    return train_windows, val_windows, rebuilt_manifest


def prepare_real(args: argparse.Namespace, device: torch.device):
    """Use the historical loader unless a frozen validation split is requested."""
    frozen_path = getattr(args, "baseline_split_manifest", None)
    if frozen_path is None:
        return common.prepare_real(args, device)

    trajectories, metadata = common.repro.load_trajectories(
        args.real_dir, args.max_files)
    frozen_manifest = json.loads(
        frozen_path.read_text(encoding="utf-8"))
    train_windows, val_windows, manifest = (
        rebuild_real_split_from_frozen_manifest(
            metadata, frozen_manifest, args.window_stride))
    train_dataset = common.repro.WindowDataset(trajectories, train_windows)
    val_dataset = common.repro.WindowDataset(trajectories, val_windows)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return (
        train_loader, val_loader, normalizer, len(train_windows),
        len(val_windows), len(trajectories))


def trajectory_member_mask(
        trajectory_ids: torch.Tensor, probability: float, seed: int,
        member_count: int) -> torch.Tensor:
    """Return a deterministic trajectory-level member assignment mask."""
    trajectory_ids = trajectory_ids.to(dtype=torch.long, device="cpu").view(-1, 1)
    member_ids = torch.arange(member_count, dtype=torch.long).view(1, -1)
    modulus = 10007
    codes = (
        (trajectory_ids + 1) * 7919
        + (member_ids + 1) * 104729
        + (seed + 1) * 15485863
    ) % modulus
    threshold = int(round(probability * modulus))
    mask = codes < threshold
    empty_rows = ~mask.any(dim=1)
    if bool(empty_rows.any()):
        rows = torch.nonzero(empty_rows, as_tuple=False).flatten()
        fallback = (trajectory_ids[rows, 0] + seed) % member_count
        mask[rows, fallback] = True
    return mask


def bootstrap_member_mse(
        members: torch.Tensor, target: torch.Tensor,
        trajectory_ids: torch.Tensor, probability: float, seed: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample = (members - target[:, None]).square().mean(
        dim=(2, 3, 4, 5))
    mask = trajectory_member_mask(
        trajectory_ids, probability, seed, members.shape[1])
    mask = mask.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * mask).sum() / mask.sum(), mask.mean()


def residual_quantile_loss(
        model: EnsembleUNet2d, prediction: torch.Tensor,
        quantiles: torch.Tensor, target: torch.Tensor,
        normalizer, *, physical_target: torch.Tensor | None = None,
        ) -> torch.Tensor:
    target_std = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    residual = (
        target[..., :2] - prediction[..., :2].detach()) * target_std
    quantiles_physical = quantiles * target_std.unsqueeze(-1)
    error = residual.unsqueeze(-1) - quantiles_physical
    levels = error.new_tensor(model.config.quantile_levels).view(
        1, 1, 1, 1, 1, -1)
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    truth_physical = (
        common.to_physical_target(target, normalizer)[..., :2]
        if physical_target is None else physical_target[..., :2])
    if truth_physical.shape != target[..., :2].shape:
        raise ValueError("physical target is not aligned with normalized target")
    scored = truth_physical != 0.0
    reference = pinball.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2, 1)
    normalized = pinball / reference
    expanded = scored.unsqueeze(-1).expand_as(normalized)
    return normalized[expanded].mean()


def soft_native_sps_objective(
        prediction: torch.Tensor, target: torch.Tensor,
        lower_extent: torch.Tensor, upper_extent: torch.Tensor,
        normalizer, tau_fraction: float, metric_normalizer=None, *,
        physical_target: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable official-SPS proxy for joint representation learning."""
    prediction_physical = common.to_physical_target(
        prediction, normalizer)[..., :2]
    if (physical_target is not None and metric_normalizer is not None
            and metric_normalizer is not normalizer):
        raise ValueError(
            "physical target override requires the same metric normalizer")
    target_physical = (
        common.to_physical_target(target, normalizer)[..., :2]
        if physical_target is None else physical_target[..., :2])
    if target_physical.shape != target[..., :2].shape:
        raise ValueError("physical target is not aligned with normalized target")
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower = lower_extent * scale
    upper = upper_extent * scale
    residual = target_physical - (
        prediction_physical if __import__("os").environ.get("B210_SPS_POINT_COUPLING") == "1"
        else prediction_physical.detach())
    # Sim pretraining may deliberately evaluate interval widths on the Real
    # u/v scale.  Absolute point accuracy and the solid-body mask must still
    # come from the dataset's own physical normalization; interpreting a Sim
    # normalized zero with Real mean/std would turn masked cells into flow.
    metric_normalizer = metric_normalizer or normalizer
    prediction_metric = common.to_physical_target(
        prediction if __import__("os").environ.get("B210_SPS_POINT_COUPLING") == "1" else prediction.detach(),
        metric_normalizer)[..., :2]
    target_metric = (
        common.to_physical_target(target, metric_normalizer)[..., :2]
        if physical_target is None else target_physical)
    scored = target_metric != 0.0
    reference = lower.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2)
    tau = tau_fraction * reference
    soft_inside = (
        torch.sigmoid((residual + lower) / tau)
        * torch.sigmoid((upper - residual) / tau))
    efficiency = soft_inside * torch.exp(
        -(lower + upper) / common.SPS_SIGMA)
    count = scored.sum(dim=(1, 2, 3, 4)).clamp_min(1)
    per_sample = (
        (efficiency * scored).sum(dim=(1, 2, 3, 4)) / count)
    factor = common.sample_sps_factor(
        prediction_metric, target_metric)
    if not __import__("os").environ.get("B210_SPS_POINT_COUPLING") == "1":
        factor = factor.detach()
    soft_score = (factor * per_sample).mean()
    return -soft_score, soft_score.detach()


def coverage_calibration_objective(
        model: EnsembleUNet2d, prediction: torch.Tensor,
        quantiles: torch.Tensor, coverage_logits: torch.Tensor,
        target: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Proper binary loss for candidate-specific conditional hit rates."""
    lower, upper, valid, _masses = model._interval_candidates(quantiles)
    residual = (
        target[..., :2] - prediction[..., :2].detach()).unsqueeze(-1)
    inside = (residual >= -lower) & (residual <= upper)
    truth_physical = common.to_physical_target(
        target, normalizer)[..., :2]
    mask = truth_physical.ne(0.0).unsqueeze(-1).expand_as(inside) & valid
    return torch.nn.functional.binary_cross_entropy_with_logits(
        coverage_logits[mask], inside[mask].to(coverage_logits.dtype))


def uq_trainable_parameters(model: EnsembleUNet2d) -> list[torch.nn.Parameter]:
    """Return de-duplicated parameters that affect only interval prediction."""
    modules = [
        model.uq_projections,
        model.uq_fusion,
        model.quantile_head,
        model.utility_refinement_head,
        model.temporal_uq_refiner,
        model.coverage_calibration_head,
        model.residual_mixture_head,
    ]
    result = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def point_loss(
        args: argparse.Namespace, prediction: torch.Tensor,
        target: torch.Tensor, normalizer,
        weights: tuple[float, float, float], iteration: int,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mse = (prediction - target).square().mean()
    components = {"mse": mse.detach()}
    warmup = iteration <= args.mse_warmup_updates
    if args.point_loss_mode == "mse" or warmup:
        loss = mse
    else:
        accuracy, details = common.accuracy_objective(
            prediction, target, normalizer, weights)
        loss = accuracy + args.mse_weight * mse
        components.update(details)
    components["mse_warmup"] = mse.new_tensor(float(warmup))
    return loss, components


def mean_profile_objective(
        prediction: torch.Tensor, target: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Exact differentiable v9 MVPE error on the four probe profiles."""
    pred = common.to_physical_target(prediction, normalizer)[..., :2]
    truth = common.to_physical_target(target, normalizer)[..., :2]
    batch = pred.shape[0]
    center_y = pred.shape[2] // 2
    probe_y = [center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < pred.shape[2]]
    errors = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if not 0 <= probe_x < pred.shape[3]:
            continue
        pred_profile = pred[:, :, probe_y, probe_x].mean(
            dim=1).reshape(batch, -1)
        truth_profile = truth[:, :, probe_y, probe_x].mean(
            dim=1).reshape(batch, -1)
        errors.append(
            torch.linalg.vector_norm(pred_profile - truth_profile, dim=1)
            / torch.linalg.vector_norm(
                truth_profile, dim=1).clamp_min(1e-8))
    if not errors:
        raise ValueError("no valid MVPE probes for this spatial resolution")
    return torch.stack(errors).mean()


POINT_DISTILLATION_COMPONENT_WEIGHTS = {
    "temporal_mean": 2.0,
    "zero_mean_fluctuation": 1.0,
    "mean_profile": 2.0,
    "physical_tke_map": 0.25,
}
POINT_DISTILLATION_SCALES = {
    "temporal_mean_normalized": 0.025,
    "zero_mean_fluctuation_normalized": 0.020,
    "mean_profile_normalized": 0.025,
}


def mean_profile_distillation_loss(
        student_mean: torch.Tensor, teacher_mean: torch.Tensor,
        scale: float,
        ) -> torch.Tensor:
    center_y = student_mean.shape[2] // 2
    probe_y = [center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < student_mean.shape[2]]
    losses = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if 0 <= probe_x < student_mean.shape[3]:
            losses.append(F.smooth_l1_loss(
                student_mean[:, :, probe_y, probe_x] / scale,
                teacher_mean[:, :, probe_y, probe_x] / scale))
    if not losses:
        raise ValueError("no valid mean-profile probes")
    return torch.stack(losses).mean()


def physical_tke_map_distillation_loss(
        student_normalized: torch.Tensor,
        teacher_physical: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    student_physical = common.to_physical_target(
        student_normalized, normalizer)[..., :2].float()
    teacher_physical = teacher_physical.float()

    def kinetic_energy(value: torch.Tensor) -> torch.Tensor:
        fluctuation = value - value.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)

    student_tke = kinetic_energy(student_physical)
    teacher_tke = kinetic_energy(teacher_physical)
    batch = student_tke.shape[0]
    relative = torch.linalg.vector_norm(
        (student_tke - teacher_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        teacher_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)
    return relative.mean()


def component_point_distillation_loss(
        student_normalized: torch.Tensor,
        teacher_physical: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match the formal point teacher without changing truth or UQ losses."""
    if (teacher_physical.ndim != student_normalized.ndim
            or teacher_physical.shape[:-1] != student_normalized.shape[:-1]
            or teacher_physical.shape[-1] != 2):
        raise ValueError("teacher point batch is not aligned with the student")
    teacher_normalized = (
        (teacher_physical.float() - normalizer.mean_target[:2])
        / normalizer.std_target[:2])
    student_uv = student_normalized[..., :2].float()
    student_mean = student_uv.mean(dim=1, keepdim=True)
    teacher_mean = teacher_normalized.mean(dim=1, keepdim=True)
    student_fluctuation = student_uv - student_mean
    teacher_fluctuation = teacher_normalized - teacher_mean

    mean_loss = F.smooth_l1_loss(
        student_mean / POINT_DISTILLATION_SCALES[
            "temporal_mean_normalized"],
        teacher_mean / POINT_DISTILLATION_SCALES[
            "temporal_mean_normalized"])
    fluctuation_loss = F.smooth_l1_loss(
        student_fluctuation / POINT_DISTILLATION_SCALES[
            "zero_mean_fluctuation_normalized"],
        teacher_fluctuation / POINT_DISTILLATION_SCALES[
            "zero_mean_fluctuation_normalized"])
    profile_loss = mean_profile_distillation_loss(
        student_mean, teacher_mean,
        POINT_DISTILLATION_SCALES["mean_profile_normalized"])
    tke_loss = physical_tke_map_distillation_loss(
        student_normalized, teacher_physical, normalizer)
    loss = (
        POINT_DISTILLATION_COMPONENT_WEIGHTS["temporal_mean"] * mean_loss
        + POINT_DISTILLATION_COMPONENT_WEIGHTS[
            "zero_mean_fluctuation"] * fluctuation_loss
        + POINT_DISTILLATION_COMPONENT_WEIGHTS["mean_profile"] * profile_loss
        + POINT_DISTILLATION_COMPONENT_WEIGHTS[
            "physical_tke_map"] * tke_loss)
    return loss, {
        "distill_temporal_mean": mean_loss.detach(),
        "distill_zero_mean_fluctuation": fluctuation_loss.detach(),
        "distill_mean_profile": profile_loss.detach(),
        "distill_physical_tke_map": tke_loss.detach(),
    }


def add_point_distillation_loss(
        base_loss: torch.Tensor, student_normalized: torch.Tensor,
        teacher_physical: torch.Tensor | None, normalizer,
        distill_weight: float,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``base_loss`` by identity when distillation is disabled."""
    if teacher_physical is None:
        if distill_weight != 0.0:
            raise ValueError("positive distill weight requires a teacher batch")
        return base_loss, {}
    if not math.isfinite(distill_weight) or distill_weight <= 0.0:
        raise ValueError("teacher batch requires a finite positive distill weight")
    distillation, components = component_point_distillation_loss(
        student_normalized, teacher_physical, normalizer)
    return base_loss + distill_weight * distillation, {
        "distillation_loss": distillation.detach(),
        "distillation_weight": distillation.new_tensor(distill_weight),
        **components,
    }


def bootstrap_member_point_loss(
        args: argparse.Namespace, members: torch.Tensor,
        target: torch.Tensor, trajectory_ids: torch.Tensor,
        normalizer, weights: tuple[float, float, float], iteration: int,
        *, member_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Train each member only on its fixed trajectory bootstrap subset."""
    if member_mask is None:
        mask = trajectory_member_mask(
            trajectory_ids, args.bootstrap_probability, args.seed,
            members.shape[1])
    else:
        mask = member_mask.to(dtype=torch.bool, device="cpu")
        expected_shape = (members.shape[0], members.shape[1])
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"member mask shape {tuple(mask.shape)} != {expected_shape}")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("member mask contains an unassigned sample")
    losses = []
    for member_index in range(members.shape[1]):
        selected = mask[:, member_index].to(device=members.device)
        if bool(selected.any()):
            member_loss, _ = point_loss(
                args, members[selected, member_index], target[selected],
                normalizer, weights, iteration)
            losses.append(member_loss)
    if not losses:
        raise RuntimeError("trajectory bootstrap selected no ensemble members")
    fraction = mask.to(device=members.device, dtype=members.dtype).mean()
    return torch.stack(losses).mean(), fraction


def evaluate(
        model: EnsembleUNet2d, loader, normalizer,
        device: torch.device,
        weights: tuple[float, float, float]) -> dict[str, float]:
    metrics = common.repro.evaluate(model, loader, normalizer, device)
    common.add_weighted(metrics, weights)
    metrics.update(common.evaluate_native_sps(
        model, loader, normalizer, device))
    return metrics


def train(
        args: argparse.Namespace, model: EnsembleUNet2d,
        train_loader, normalizer, device: torch.device,
        weights: tuple[float, float, float],
        train_windows: int, trajectories: int,
        val_loader=None, val_windows: int = 0,
        soft_sps_normalizer=None,
        teacher_cache: torch.Tensor | None = None,
        ) -> dict:
    if args.trainable_scope == "mean_profile_refiner":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if model.mean_profile_refiner is None:
            raise ValueError("checkpoint has no mean-profile refiner")
        for parameter in model.mean_profile_refiner.parameters():
            parameter.requires_grad_(True)
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    if not trainable:
        raise ValueError("selected trainable scope contains no parameters")
    iterator = common.repro.cycling(train_loader)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    stats = common.normalizer_dict(normalizer)
    history = []
    sums: dict[str, float] = {}
    count = 0
    best_value = float("inf") if args.phase == "sim" else -float("inf")
    best_iteration = 0
    started = time.time()

    if val_loader is not None:
        initial = evaluate(model, val_loader, normalizer, device, weights)
        print(json.dumps({"event": "real_initial", **initial}), flush=True)
    else:
        initial = None

    for iteration in range(1, args.updates + 1):
        model.train()
        history_batch, target, trajectory_ids, sample_indices = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_input = add_training_input_gain(
            history_batch, float(getattr(args, "input_gain_std", 0.0)))
        history_norm, target_norm = normalizer.preprocess(history_input, target)
        history_norm = masked_velocity_blur(
            history_norm, history_batch,
            float(getattr(args, "input_blur_probability", 0.0)),
            float(getattr(args, "input_blur_strength", 0.0)))
        if args.input_noise_std:
            history_norm = add_training_input_noise(
                history_norm, history_batch, args.input_noise_std)
        optimizer.zero_grad(set_to_none=True)
        (prediction, members, quantiles, lower, upper, _features,
         coverage_logits) = model.forward_uq_training(history_norm)
        mean_loss, components = point_loss(
            args, prediction, target_norm, normalizer, weights, iteration)
        if args.member_loss_mode == "point":
            member_loss, bootstrap_fraction = bootstrap_member_point_loss(
                args, members, target_norm, trajectory_ids,
                normalizer, weights, iteration)
        else:
            member_loss, bootstrap_fraction = bootstrap_member_mse(
                members, target_norm, trajectory_ids,
                args.bootstrap_probability, args.seed)
        quantile_loss = residual_quantile_loss(
            model, prediction, quantiles, target_norm, normalizer)
        if args.soft_sps_loss_weight:
            soft_sps_loss, soft_sps_score = soft_native_sps_objective(
                prediction, target_norm, lower, upper,
                soft_sps_normalizer or normalizer,
                args.soft_sps_tau_fraction,
                metric_normalizer=normalizer)
        else:
            soft_sps_loss = prediction.new_zeros(())
            soft_sps_score = prediction.new_zeros(())
        coverage_bce = (
            coverage_calibration_objective(
                model, prediction, quantiles, coverage_logits,
                target_norm, normalizer)
            if coverage_logits is not None
            else prediction.new_zeros(()))
        profile_loss = (
            mean_profile_objective(
                prediction, target_norm, normalizer)
            if args.profile_loss_weight else prediction.new_zeros(()))
        soft_sps_weight = args.soft_sps_loss_weight * min(
            1.0,
            iteration / max(args.soft_sps_warmup_updates, 1))
        base_loss = (
            args.mean_loss_weight * mean_loss
            + args.member_loss_weight * member_loss
            + args.quantile_loss_weight * quantile_loss
            + args.coverage_bce_weight * coverage_bce
            + args.profile_loss_weight * profile_loss)
        teacher_batch = (
            teacher_cache[sample_indices].to(
                device=device, dtype=torch.float32, non_blocking=True)
            if teacher_cache is not None else None)
        supervised_loss, distillation_components = add_point_distillation_loss(
            base_loss, prediction, teacher_batch, normalizer,
            args.distill_weight)
        loss = supervised_loss + soft_sps_weight * soft_sps_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite ensemble loss at {iteration}")
        if args.soft_sps_uq_only and soft_sps_weight:
            supervised_loss.backward(retain_graph=True)
            interval_parameters = uq_trainable_parameters(model)
            interval_gradients = torch.autograd.grad(
                soft_sps_weight * soft_sps_loss,
                interval_parameters, allow_unused=True)
            for parameter, gradient in zip(
                    interval_parameters, interval_gradients):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
        else:
            loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        physical_spread = (
            members[..., :2].std(dim=1, unbiased=False)
            * normalizer.std_target[:2].view(1, 1, 1, 1, 2))
        values = {
            "loss": loss.detach(),
            "member_loss": member_loss.detach(),
            "bootstrap_fraction": bootstrap_fraction.detach(),
            "quantile_pinball": quantile_loss.detach(),
            "soft_sps_loss": soft_sps_loss.detach(),
            "soft_sps_score": soft_sps_score,
            "soft_sps_weight": prediction.new_tensor(soft_sps_weight),
            "coverage_bce": coverage_bce.detach(),
            "profile_loss": profile_loss.detach(),
            "member_spread_physical": physical_spread.mean().detach(),
            **components,
            **distillation_components,
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

        if iteration % args.eval_interval == 0 or iteration == args.updates:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
            }
            if val_loader is not None:
                metrics = evaluate(model, val_loader, normalizer, device, weights)
                row.update(metrics)
                selected = metrics[args.selection_metric]
                improved = selected > best_value
            else:
                selected = row["train_loss_mean"]
                improved = selected < best_value
            if improved:
                best_value = selected
                best_iteration = iteration
                common.save_checkpoint(
                    args.output_dir / "best.pth", model, args, iteration,
                    history + [row], stats, row)
            history.append(row)
            print(json.dumps({"event": f"{args.phase}_eval", **row}), flush=True)
            sums.clear()
            count = 0
        if iteration % args.checkpoint_interval == 0:
            common.save_checkpoint(
                args.output_dir / f"model_{iteration:05d}.pth",
                model, args, iteration, history, stats)

    final_metrics = (
        evaluate(model, val_loader, normalizer, device, weights)
        if val_loader is not None else history[-1])
    common.save_checkpoint(
        args.output_dir / "final.pth", model, args, args.updates,
        history, stats, final_metrics)
    return {
        "phase": args.phase,
        "updates": args.updates,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "initial_metrics": initial,
        "final_metrics": final_metrics,
        "best_iteration": best_iteration,
        "best_metric_value": best_value,
        "selection_metric": (
            "train_loss_mean" if args.phase == "sim"
            else args.selection_metric),
        "history": history,
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    weights = validate(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    audit_started_at_utc = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if args.sim_audit_variant is not None else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    model = build_or_load(args, device)
    parameter_count = sum(value.numel() for value in model.parameters())
    print(json.dumps({
        "event": "model", "phase": args.phase,
        "config": model.architecture_config(),
        "parameters": parameter_count,
    }), flush=True)

    sim_audit = (
        SimStreamAudit() if args.sim_audit_variant is not None else None)

    if args.phase == "sim":
        train_loader, normalizer, train_windows, trajectories = (
            common.prepare_sim(args, device))
        if args.temporal_jitter_max:
            jittered = TemporalJitterDataset(
                train_loader.dataset, args.temporal_jitter_max, sim_audit)
            train_loader = DataLoader(
                jittered, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True,
                generator=torch.Generator().manual_seed(args.seed))
        val_loader = None
        val_windows = 0
    else:
        (train_loader, val_loader, normalizer, train_windows,
         val_windows, trajectories) = prepare_real(args, device)
    teacher_cache = None
    teacher_cache_metadata = None
    if args.teacher_cache is not None:
        manifest = json.loads(
            (args.output_dir / "split_manifest.json").read_text(
                encoding="utf-8"))
        teacher_cache, teacher_cache_metadata = load_and_validate_cache(
            args.teacher_cache, manifest, train_loader.dataset)
        print(json.dumps({
            "event": "teacher_cache", **teacher_cache_metadata,
        }), flush=True)
    train_loader = tag_training_loader(train_loader, args, sim_audit)
    common.set_model_stats(model, normalizer)
    initial_model_state_sha256 = (
        model_state_sha256(model.state_dict())
        if sim_audit is not None else None)
    soft_sps_normalizer = normalizer
    if (args.phase == "sim" and args.soft_sps_loss_weight
            and args.soft_sps_proxy_real_scale):
        soft_sps_normalizer = common.repro.Normalizer(
            args.real_stats, device)
    summary = train(
        args, model, train_loader, normalizer, device, weights,
        train_windows, trajectories, val_loader, val_windows,
        soft_sps_normalizer, teacher_cache)
    summary.update({
        "model_config": model.architecture_config(),
        "parameters": parameter_count,
        "training_config": json_training_config(args),
    })
    if sim_audit is not None:
        model_config_payload = model.architecture_config()
        training_config_payload = json_training_config(args)
        endpoint_checkpoint = args.output_dir / "final.pth"
        endpoint_model_state_sha256 = model_state_sha256(model.state_dict())
        endpoint_checkpoint_sha256 = file_sha256(endpoint_checkpoint)
        audit_hardware = hardware_fingerprint(device)
        audit_completed_at_utc = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        audit_payload = {
            "schema": "c208_dr1_sim_audit_v1",
            "variant": args.sim_audit_variant,
            "run_ordinal": (
                1 if args.sim_audit_variant == "candidate" else 2),
            "started_at_utc": audit_started_at_utc,
            "completed_at_utc": audit_completed_at_utc,
            "initial_model_state_sha256": initial_model_state_sha256,
            "base_sample_stream_sha256": (
                sim_audit.base_sample_stream_sha256),
            "jitter_shift_stream_sha256": (
                sim_audit.jitter_shift_stream_sha256),
            "stream_samples": sim_audit.samples,
            "model_config": model_config_payload,
            "model_config_sha256": canonical_sha256(model_config_payload),
            "training_config": training_config_payload,
            "training_config_sha256": canonical_sha256(
                training_config_payload),
            "endpoint_model_state_sha256": endpoint_model_state_sha256,
            "endpoint_checkpoint_sha256": endpoint_checkpoint_sha256,
            "augmentation": augmentation_config(args),
            "teacher_cache": None,
            "distill_weight": 0.0,
            "seed": args.seed,
            "window_stride": args.window_stride,
            "updates": args.updates,
            "hardware_fingerprint": audit_hardware,
        }
        summary["sim_audit"] = audit_payload
        (args.output_dir / "sim_audit.json").write_text(
            json.dumps(
                audit_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8")
    if teacher_cache_metadata is not None:
        summary["point_distillation"] = {
            "distill_weight": args.distill_weight,
            "component_weights": POINT_DISTILLATION_COMPONENT_WEIGHTS,
            "component_scales": POINT_DISTILLATION_SCALES,
            "teacher_cache": teacher_cache_metadata,
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if summary["history"]:
        with (args.output_dir / "learning_curve.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            fields = list(dict.fromkeys(
                key for row in summary["history"] for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary["history"])
    print(json.dumps({
        "event": "complete", "phase": args.phase,
        "wall_seconds": summary["wall_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
