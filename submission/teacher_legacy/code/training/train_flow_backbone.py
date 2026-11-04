#!/usr/bin/env python3
"""Train point/distribution backbones with the common RealPDE protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import train_hybrid_fno2d as common  # noqa: E402
from models.dit2d import DiT2d, DiT2dConfig  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
from models.unet2d import UNet2d, UNet2dConfig  # noqa: E402
from models.temporal_transport import (  # noqa: E402
    MIXER_MODES,
    TemporalTransportConfig,
    TemporalTransportOperator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("unet", "dit", "temporal_transport"),
        required=True)
    parser.add_argument("--phase", choices=("sim", "real"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
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
        "--real-all-windows", action=argparse.BooleanOptionalAction,
        default=False,
        help=("Real refit only: optimize on both temporal-split partitions; "
              "validation metrics then become diagnostic, not held-out."))

    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--unet-depth", type=int, default=4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--difference-steps", type=int, default=5)
    parser.add_argument(
        "--use-coordinates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-regime-film", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mean-fluctuation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--predict-uncertainty", action=argparse.BooleanOptionalAction,
        default=None,
        help=("Native interval head. Defaults to enabled for U-Net/DiT and "
              "disabled for the point-only temporal-transport model."))
    parser.add_argument(
        "--learn-temporal-spread", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--dit-depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--temporal-hidden-channels", type=int, default=64)
    parser.add_argument("--temporal-spatial-blocks", type=int, default=2)
    parser.add_argument("--temporal-spatial-downsample", type=int, default=2)
    parser.add_argument("--temporal-mixer-blocks", type=int, default=3)
    parser.add_argument("--temporal-kernel", type=int, default=5)
    parser.add_argument(
        "--temporal-mixer-mode", choices=MIXER_MODES,
        default="causal_temporal",
        help=("Active explicit-time mixer or its parameter/MAC-matched "
              "per-frame spatial control."))
    parser.add_argument("--transport-max-displacement", type=float, default=4.0)
    parser.add_argument("--transport-residual-limit", type=float, default=4.0)
    parser.add_argument("--transport-output-init-std", type=float, default=1.0e-3)

    parser.add_argument(
        "--sim-loss-mode", choices=("mse", "official_accuracy"),
        default="official_accuracy")
    parser.add_argument("--sim-mse-weight", type=float, default=0.25)
    parser.add_argument(
        "--loss-mode", choices=("mse", "official_accuracy"),
        default="official_accuracy")
    parser.add_argument("--mse-warmup-updates", type=int, default=0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--spectrum-weight", type=float, default=0.0)
    parser.add_argument("--component-tke-weight", type=float, default=0.0)
    parser.add_argument("--component-tke-decay-updates", type=int, default=0)
    parser.add_argument(
        "--interval-loss-mode", choices=("laplace", "soft_sps"),
        default="soft_sps")
    parser.add_argument("--interval-loss-weight", type=float, default=0.025)
    parser.add_argument("--interval-nll-weight", type=float, default=0.10)
    parser.add_argument(
        "--interval-detach-mean", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--accuracy-weights", type=float, nargs=3,
        metavar=("REL", "TKE", "MVPE"), default=(1.0, 4.0, 1.0))
    parser.add_argument(
        "--selection-metric",
        choices=("rmse", "accuracy_score_mean", "weighted_accuracy_score",
                 "point_score", "official_point_contribution",
                 "tke_rel_l2_score"), default=None,
        help=("Checkpoint promotion metric. Defaults to the historical "
              "point_score for U-Net/DiT and the published 46/10/9 point "
              "contribution for temporal_transport."))
    return parser.parse_args()


def config_from_args(args: argparse.Namespace):
    predict_uncertainty = (
        args.architecture != "temporal_transport"
        if args.predict_uncertainty is None else args.predict_uncertainty)
    if args.architecture == "temporal_transport":
        return TemporalTransportConfig(
            hidden_channels=args.temporal_hidden_channels,
            spatial_blocks=args.temporal_spatial_blocks,
            spatial_downsample=args.temporal_spatial_downsample,
            temporal_blocks=args.temporal_mixer_blocks,
            temporal_kernel=args.temporal_kernel,
            mixer_mode=args.temporal_mixer_mode,
            groups=args.groups,
            use_coordinates=args.use_coordinates,
            max_displacement=args.transport_max_displacement,
            residual_limit=args.transport_residual_limit,
            output_initialization_std=args.transport_output_init_std,
            predict_uncertainty=predict_uncertainty,
        )
    shared = dict(
        difference_steps=args.difference_steps,
        use_coordinates=args.use_coordinates,
        mean_fluctuation=args.mean_fluctuation,
        predict_uncertainty=predict_uncertainty,
        learn_temporal_spread=args.learn_temporal_spread,
    )
    if args.architecture == "unet":
        return UNet2dConfig(
            base=args.base, depth=args.unet_depth, groups=args.groups,
            use_regime_film=args.use_regime_film, **shared)
    return DiT2dConfig(
        image_height=args.image_height, image_width=args.image_width,
        patch_size=args.patch_size, dimension=args.dimension,
        depth=args.dit_depth, heads=args.heads, mlp_ratio=args.mlp_ratio,
        **shared)


def construct_model(architecture: str, config: dict):
    config = dict(config)
    stored_architecture = config.pop("architecture", architecture)
    if stored_architecture != architecture:
        raise ValueError(
            f"checkpoint architecture {stored_architecture} != {architecture}")
    if architecture == "unet":
        return UNet2d(UNet2dConfig(**config))
    if architecture == "ensemble_unet":
        return EnsembleUNet2d(EnsembleUNet2dConfig(**config))
    if architecture == "dit":
        return DiT2d(DiT2dConfig(**config))
    if architecture == "temporal_transport":
        return TemporalTransportOperator(TemporalTransportConfig(**config))
    raise ValueError(f"unsupported architecture {architecture}")


def build_or_load(args: argparse.Namespace, device: torch.device):
    if args.phase == "sim":
        if args.start_checkpoint is not None:
            raise ValueError("Sim phase must start from random initialization")
        config = config_from_args(args)
        config.validate()
        return construct_model(args.architecture, as_config(config)).to(device)
    if args.start_checkpoint is None:
        raise ValueError("Real phase requires a Sim checkpoint")
    payload = torch.load(
        args.start_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("phase") != "sim":
        raise ValueError("Real phase must load a Sim-phase checkpoint")
    model = construct_model(args.architecture, payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def as_config(config) -> dict:
    return {key: getattr(config, key) for key in config.__dataclass_fields__}


def validate(args: argparse.Namespace) -> tuple[float, float, float]:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.checkpoint_interval, args.lr,
        args.window_stride,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("training counts, learning rate and stride must be positive")
    finite_nonnegative = (
        args.weight_decay, args.grad_clip, args.sim_mse_weight,
        args.mse_weight, args.spectrum_weight, args.component_tke_weight,
        args.interval_loss_weight, args.interval_nll_weight,
        args.distill_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in finite_nonnegative):
        raise ValueError("loss and optimizer values must be finite and non-negative")
    if not 0 <= args.mse_warmup_updates <= args.updates:
        raise ValueError("MSE warmup must lie within the run")
    if (args.teacher_checkpoint is None) != (args.distill_weight == 0.0):
        raise ValueError("teacher checkpoint and positive weight must be paired")
    if args.phase == "sim" and args.teacher_checkpoint is not None:
        raise ValueError("distillation is only supported during Real training")
    if args.phase != "real" and args.real_all_windows:
        raise ValueError("--real-all-windows is only valid in the Real phase")
    resolved_uncertainty = (
        args.architecture != "temporal_transport"
        if args.predict_uncertainty is None else args.predict_uncertainty)
    if args.architecture == "temporal_transport" and resolved_uncertainty:
        raise ValueError(
            "temporal_transport is point-only; use --no-predict-uncertainty")
    # Downstream common training helpers inspect both the model config and CLI
    # namespace.  Materialize the architecture-dependent default once here so
    # checkpoints and summaries never store an ambiguous null value.
    args.predict_uncertainty = resolved_uncertainty
    if not resolved_uncertainty:
        args.interval_loss_weight = 0.0
    if args.selection_metric is None:
        args.selection_metric = (
            "official_point_contribution"
            if args.architecture == "temporal_transport" else "point_score")
    weights = tuple(float(value) for value in args.accuracy_weights)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("accuracy weights must be finite and non-negative")
    if not any(weights):
        raise ValueError("at least one accuracy weight must be positive")
    config_from_args(args).validate()
    return weights


def main() -> None:
    args = parse_args()
    weights = validate(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.repro.set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = build_or_load(args, device)
    parameter_count = sum(value.numel() for value in model.parameters())
    print(json.dumps({
        "event": "model", "phase": args.phase,
        "config": model.architecture_config(),
        "parameters": parameter_count,
    }), flush=True)
    if args.phase == "sim":
        loader, normalizer, windows, trajectories = common.prepare_sim(args, device)
        common.set_model_stats(model, normalizer)
        summary = common.train_sim(
            args, model, loader, normalizer, device,
            windows, trajectories, weights)
    else:
        (train_loader, val_loader, normalizer, train_windows,
         val_windows, trajectories) = common.prepare_real(args, device)
        common.set_model_stats(model, normalizer)
        teacher = None
        if args.teacher_checkpoint is not None:
            baseline_loader = common.repro.load_baseline_module(args.kit_dir)
            teacher, _ = baseline_loader.load_baseline(
                "fno", str(args.teacher_checkpoint), device=str(device))
            teacher = teacher.to(device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        summary = common.train_real(
            args, model, train_loader, val_loader, normalizer, device,
            train_windows, val_windows, trajectories, weights, teacher)
    summary.update({
        "model_config": model.architecture_config(),
        "parameters": parameter_count,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    })
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
