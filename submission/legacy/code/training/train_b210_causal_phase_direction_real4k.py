#!/usr/bin/env python3
"""Train a matched causal/spatial phase adapter on the full Real4k stream.

The script deliberately freezes an existing B210 checkpoint and optimizes
only the small phase-direction adapter.  Both arms see the same clean81
prefixes, stride-5 windows, seed-4 sample order and augmentation RNG.  It does
not read clean716, select an endpoint by score, train UQ, or build a package.

Endpoints contain only adapter raw/EMA states plus the immutable B210 SHA-256;
the large private base checkpoint is never copied into Git-facing artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import asdict
from typing import Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
    FrozenB210CausalPhase,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_ensemble_unet2d as mature_real  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real4k  # noqa: E402


SCHEMA = "b210_causal_phase_direction_real4k_matched_v1"
ENDPOINT_SCHEMA = "b210_causal_phase_direction_adapter_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_B210_CAUSAL_PHASE_DIRECTION_MATCHED_REAL4K_V1"
ARMS = (
    "framewise_spatial_control",
    "causal_temporal",
)
UPDATES = 4_000
BATCH_SIZE = 8
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
SCHEDULER_T_MAX = 12_000
EXPECTED_ADAPTER_PARAMETERS = 41_667


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "checkpoint-smoke", "formal"),
        default="contract")
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-checkpoint-sha256")
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args(argv)


def experiment_contract() -> dict[str, object]:
    config = CausalPhaseDirectionConfig()
    return {
        "schema": SCHEMA,
        "question": (
            "does explicit causal observed-time information improve B210 "
            "future phase when mean and fluctuation RMS cannot move"),
        "base": {
            "architecture": "existing EnsembleUNet2d B210 checkpoint",
            "frozen": True,
            "owns": [
                "per-pixel/channel temporal mean",
                "per-pixel/channel temporal fluctuation RMS",
                "pressure",
                "absolute lower/upper bounds",
            ],
            "stored_in_adapter_endpoint": False,
        },
        "adapter": {
            "config": {
                **asdict(config),
                "architecture": "causal_phase_direction_adapter",
            },
            "parameters": EXPECTED_ADAPTER_PARAMETERS,
            "candidate": "strict left-causal mixer over observed time",
            "control": "same-shaped depthwise mixer over framewise spatial x",
            "matched_parameter_shapes": True,
            "matched_conv_macs": True,
            "output": "zero-mean unit-RMS u/v temporal direction only",
            "selection": (
                "history-conditioned bounded sample/horizon/pixel gate; no "
                "fixed streamwise threshold"),
            "projection": "restore frozen B210 temporal mean/RMS exactly",
            "pressure_passthrough": "bitwise",
            "sps_contract": (
                "absolute endpoints/coverage/width fixed; SPS accuracy "
                "multiplier may change with the point forecast"),
            "update_zero": "bitwise B210 identity",
        },
        "real": {
            "data": "competition clean81 optimizer-visible prefixes only",
            "normalizer_windows": 2_544,
            "optimizer_windows": 9_933,
            "optimizer_stride": 5,
            "updates_each_arm": UPDATES,
            "batch_size": BATCH_SIZE,
            "seed": real4k.SEED,
            "sample_stream": "same seed-4 epoch randperm for both arms",
            "augmentation_rng": "reset identically before each arm",
            "loss": (
                "official Rel-L2 score loss after deterministic projection"),
            "direction_diagnostic": (
                "2 * predicted_radius * target_radius weighted local "
                "fluctuation-direction cosine; logged but not added to loss"),
            "tke_or_mvpe_loss": None,
            "reason": (
                "projection fixes B210 temporal fluctuation RMS and temporal "
                "mean, so TKE/MVPE terms cannot supervise the adapter"),
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": f"CosineAnnealingLR(T_max={SCHEDULER_T_MAX})",
            "ema_decay": EMA_DECAY,
        },
        "evaluation": {
            "clean716_read_by_this_script": False,
            "endpoint_selection": False,
            "score_gate": None,
            "required_followup": (
                "seal both u4000 raw/EMA endpoints, then replay the same "
                "clean716 panel and report causal-minus-control plus mature "
                "B210 absolute reference"),
        },
        "submission_authorized": False,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> str:
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("base-checkpoint-sha256 must be 64 lowercase hex digits")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"base checkpoint changed: {actual} != {expected}")
    return actual


def _extract_base_checkpoint(
        payload: Mapping[str, object],
        ) -> tuple[EnsembleUNet2dConfig, Mapping[str, torch.Tensor], str]:
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("B210 checkpoint has no model_config mapping")
    config_values = dict(raw_config)
    if config_values.pop("architecture", None) != "ensemble_unet":
        raise ValueError("base checkpoint is not an EnsembleUNet2d B210")
    config = EnsembleUNet2dConfig(**config_values)
    state_key = next((key for key in (
        "model_state_dict", "model_state")
        if isinstance(payload.get(key), Mapping)), None)
    if state_key is None:
        raise ValueError("B210 checkpoint has no supported model state mapping")
    state = payload[state_key]
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("B210 model state contains a non-tensor value")
    return config, state, state_key


def load_base_payload(
        path: Path, expected_sha: str,
        ) -> tuple[Mapping[str, object], EnsembleUNet2dConfig,
                   Mapping[str, torch.Tensor], dict[str, object]]:
    actual_sha = require_sha(path, expected_sha)
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise TypeError("B210 checkpoint payload must be a mapping")
    config, state, state_key = _extract_base_checkpoint(payload)
    return payload, config, state, {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "state_key": state_key,
        "phase": payload.get("phase"),
        "iteration": payload.get("iteration"),
        "model_config_sha256": real4k.canonical_sha(dict(payload["model_config"])),
        "model_state_sha256": ema_lib.tensor_state_sha256(state),
    }


def build_base(
        config: EnsembleUNet2dConfig, state: Mapping[str, torch.Tensor],
        normalizer, device: torch.device) -> EnsembleUNet2d:
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    mature_real.common.set_model_stats(model, normalizer)
    model.requires_grad_(False)
    return model.to(device).eval()


def build_adapter(mode: str, device: torch.device) -> CausalPhaseDirectionAdapter:
    if mode not in ARMS:
        raise ValueError(f"unknown phase arm {mode!r}")
    # Reset immediately around construction so both arms begin with identical
    # tensor values.  The mode itself is non-tensor execution metadata.
    torch.manual_seed(real4k.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(real4k.SEED)
    adapter = CausalPhaseDirectionAdapter(
        CausalPhaseDirectionConfig(mixer_mode=mode)).float()
    count = sum(parameter.numel() for parameter in adapter.parameters())
    if count != EXPECTED_ADAPTER_PARAMETERS:
        raise RuntimeError(f"adapter parameter count changed: {count}")
    return adapter.to(device)


def normalizer_payload(normalizer) -> dict[str, torch.Tensor]:
    return {
        name: getattr(normalizer, name).detach().cpu().clone()
        for name in ("mean_input", "mean_target", "std_input", "std_target")
    }


def endpoint_payload(
        *, adapter: CausalPhaseDirectionAdapter, ema: ema_lib.FP32EMAShadow,
        arm: str, source: Mapping[str, object], normalizer,
        schedule_sha: str, trace: Sequence[Mapping[str, object]],
        elapsed_seconds: float, updates: int,
        ) -> dict[str, object]:
    raw = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    ema_state = ema.materialize_model_state(adapter)
    return {
        "schema": ENDPOINT_SCHEMA,
        "experiment_schema": SCHEMA,
        "arm": arm,
        "update": updates,
        "base_checkpoint": dict(source),
        "adapter_config": adapter.architecture_config(),
        "adapter_parameters": EXPECTED_ADAPTER_PARAMETERS,
        "raw_adapter_state": raw,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(raw),
        "ema_adapter_state": ema_state,
        "ema_adapter_state_sha256": ema_lib.tensor_state_sha256(ema_state),
        "ema": {key: value for key, value in ema.seal(adapter).items()
                if key != "model_state"},
        "normalizer": normalizer_payload(normalizer),
        "sample_schedule_sha256": schedule_sha,
        "training": {
            "updates": updates,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler_t_max": SCHEDULER_T_MAX,
            "ema_decay": EMA_DECAY,
            "objective": "official_projected_rel_score_loss",
            "direction_cosine_auxiliary_weight": 0.0,
            "tke_or_mvpe_loss": None,
            "augmentation": dict(real4k.AUGMENTATION),
            "trace": list(trace),
            "elapsed_seconds": float(elapsed_seconds),
        },
    }


def phase_objective(
        prediction: torch.Tensor, target: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize only the degrees of freedom left by the fixed projection."""
    prediction_physical = mature_real.common.to_physical_target(
        prediction, normalizer)[..., :2].float()
    target_physical = mature_real.common.to_physical_target(
        target, normalizer)[..., :2].float()
    return physical_phase_objective(prediction_physical, target_physical)


def physical_phase_objective(
        prediction_physical: torch.Tensor,
        target_physical: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official Rel-L2 mapping on physical u/v, shared by direct and KD fits.

    Keeping this primitive separate prevents the target-free KD stage from
    manufacturing a normalized pseudo-label merely to reuse the Real4k loss.
    Both stages therefore optimize the exact same physical-space scalar.
    """
    if (prediction_physical.shape != target_physical.shape
            or prediction_physical.ndim != 5
            or prediction_physical.shape[-1] != 2):
        raise ValueError(
            "physical phase objective requires aligned [B,T,H,W,2] tensors")
    prediction_physical = prediction_physical.float()
    target_physical = target_physical.float()
    batch = prediction_physical.shape[0]
    relative = torch.linalg.vector_norm(
        (prediction_physical - target_physical).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        target_physical.reshape(batch, -1), dim=1).clamp_min(1.0e-8)
    # The leaderboard maps error e to score 1/(1+0.5e).  Minimizing one minus
    # that score is the exact differentiable Rel-L2 score objective.
    mean_relative = relative.mean()
    rel_score_loss = 1.0 - 1.0 / (1.0 + 0.5 * mean_relative)

    prediction_fluctuation = prediction_physical - prediction_physical.mean(
        dim=1, keepdim=True)
    target_fluctuation = target_physical - target_physical.mean(
        dim=1, keepdim=True)
    dot = (prediction_fluctuation * target_fluctuation).sum(dim=1)
    prediction_energy = prediction_fluctuation.square().sum(dim=1)
    target_energy = target_fluctuation.square().sum(dim=1)
    active = (prediction_energy > 1.0e-12) & (target_energy > 1.0e-12)
    prediction_norm = torch.where(
        active, prediction_energy, torch.ones_like(prediction_energy)).sqrt()
    target_norm = torch.where(
        active, target_energy, torch.ones_like(target_energy)).sqrt()
    denominator = torch.where(
        active, prediction_norm * target_norm,
        torch.ones_like(prediction_norm))
    cosine = torch.where(active, dot / denominator, torch.ones_like(dot))
    # For two fixed-radius vectors the directional part of squared error is
    # 2*r_pred*r_target*(1-cos(theta)).  Radius-product weighting prevents
    # low-energy inlet/calm pixels from dominating energetic wake pixels.
    radius_product = torch.where(
        active, 2.0 * prediction_norm * target_norm,
        torch.zeros_like(prediction_norm))
    direction_weight = radius_product.sum().clamp_min(1.0e-12)
    weighted_cosine = (radius_product * cosine).sum() / direction_weight
    weighted_direction_loss = (
        radius_product * (1.0 - cosine)).sum() / direction_weight
    # The clean first experiment has one optimized objective.  The weighted
    # direction term is a mechanism diagnostic, not an arbitrarily weighted
    # second target; projected Rel-L2 already contains the same geometry.
    loss = rel_score_loss
    return loss, {
        "official_rel_score_loss": rel_score_loss.detach(),
        "relative_l2_error": mean_relative.detach(),
        "radius_product_weighted_direction_cosine": weighted_cosine.detach(),
        "radius_product_weighted_direction_loss": (
            weighted_direction_loss.detach()),
        "direction_radius_product_mean": radius_product.mean().detach(),
        "active_direction_fraction": active.float().mean().detach(),
        "total_phase_objective": loss.detach(),
    }


def train_arm(
        *, arm: str, config: EnsembleUNet2dConfig,
        base_state: Mapping[str, torch.Tensor], dataset, normalizer,
        schedule: Sequence[Sequence[int]], source: Mapping[str, object],
        output_dir: Path, device: torch.device,
        updates: int = UPDATES,
        ) -> dict[str, object]:
    if updates not in (1, 2, UPDATES):
        raise ValueError("only mechanics smoke u1/u2 or the complete u4000 endpoint is allowed")
    base = build_base(config, base_state, normalizer, device)
    adapter = build_adapter(arm, device)
    initial_adapter_state_sha = ema_lib.tensor_state_sha256(
        adapter.state_dict())
    wrapper = FrozenB210CausalPhase(base, adapter).train()
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise RuntimeError("frozen B210 unexpectedly has a trainable parameter")
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX)
    ema = ema_lib.FP32EMAShadow(adapter, EMA_DECAY)
    real4k.seed_real_recipe(device)
    trace: list[dict[str, object]] = []
    update0_checked = False
    head_gradient_u1 = False
    history_gradient_u2 = False
    gate_gradient_u2 = False
    started = time.perf_counter()
    for cursor in range(updates):
        update = cursor + 1
        indices = schedule[cursor]
        history, target = zip(*(dataset[index] for index in indices))
        history_physical = torch.stack(history).to(device)
        target_physical = torch.stack(target).to(device)
        adapter_history_normalized = real4k.augment_real_history(
            history_physical, normalizer)
        base_history_normalized, target_normalized = normalizer.preprocess(
            history_physical, target_physical)

        optimizer.zero_grad(set_to_none=True)
        components = wrapper.forward_components(
            base_history_normalized,
            adapter_history=adapter_history_normalized)
        if update == 1:
            update0_checked = torch.equal(
                components.prediction, components.base_prediction)
            if not update0_checked:
                raise RuntimeError("phase adapter is not exact identity at update zero")
        loss, details = phase_objective(
            components.prediction, target_normalized, normalizer)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at {arm} u{update}")
        loss.backward()
        head_gradient = adapter.direction_head.weight.grad
        if update == 1:
            head_gradient_u1 = bool(
                head_gradient is not None
                and torch.isfinite(head_gradient).all()
                and torch.count_nonzero(head_gradient))
            if not head_gradient_u1:
                raise RuntimeError("direction head lacks a finite u1 gradient")
        if update == 2:
            history_gradient = adapter.frame_encoder.stem.weight.grad
            history_gradient_u2 = bool(
                history_gradient is not None
                and torch.isfinite(history_gradient).all()
                and torch.count_nonzero(history_gradient))
            if not history_gradient_u2:
                raise RuntimeError("history path lacks a finite u2 gradient")
            gate_gradient = adapter.gate_head.weight.grad
            gate_gradient_u2 = bool(
                gate_gradient is not None
                and torch.isfinite(gate_gradient).all()
                and torch.count_nonzero(gate_gradient))
            if not gate_gradient_u2:
                raise RuntimeError("direction gate lacks a finite u2 gradient")
        gradient = nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError(f"non-finite gradient at {arm} u{update}")
        optimizer.step()
        ema.update_after_optimizer_step(adapter, optimizer_step=update)
        scheduler.step()

        if update in (1, 2, updates) or update % 500 == 0:
            base_uv = components.base_prediction[..., :2].float()
            output_uv = components.prediction[..., :2].float()
            base_mean = base_uv.mean(dim=1, keepdim=True)
            output_mean = output_uv.mean(dim=1, keepdim=True)
            base_rms = (base_uv - base_mean).square().mean(
                dim=1, keepdim=True).sqrt()
            output_rms = (output_uv - output_mean).square().mean(
                dim=1, keepdim=True).sqrt()
            row = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "max_temporal_mean_delta": float(
                    (output_mean - base_mean).abs().max()),
                "max_temporal_rms_delta": float(
                    (output_rms - base_rms).abs().max()),
                "direction_gate_mean": float(
                    components.direction_gate.detach().mean()),
                "direction_gate_min": float(
                    components.direction_gate.detach().amin()),
                "direction_gate_max": float(
                    components.direction_gate.detach().amax()),
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({
                "event": "causal_phase_real_progress", "arm": arm, **row,
            }, sort_keys=True), flush=True)

    if updates == UPDATES and not (history_gradient_u2 and gate_gradient_u2):
        raise RuntimeError("complete endpoint lacks adapter gradient evidence")
    elapsed = time.perf_counter() - started
    endpoint = endpoint_payload(
        adapter=adapter, ema=ema, arm=arm, source=source,
        normalizer=normalizer,
        schedule_sha=real4k.schedule_sha(schedule[:updates]),
        trace=trace, elapsed_seconds=elapsed, updates=updates)
    endpoint["mechanics"] = {
        "update0_bitwise_identity": update0_checked,
        "direction_head_gradient_u1": head_gradient_u1,
        "history_encoder_gradient_u2": history_gradient_u2,
        "direction_gate_gradient_u2": gate_gradient_u2,
        "initial_adapter_state_sha256": initial_adapter_state_sha,
    }
    path = output_dir / f"{arm}_adapter_u{updates}.pt"
    real4k.atomic_torch_save(path, endpoint)
    report = {
        "arm": arm,
        "path": str(path.resolve()),
        "file_sha256": sha256(path),
        "raw_adapter_state_sha256": endpoint["raw_adapter_state_sha256"],
        "ema_adapter_state_sha256": endpoint["ema_adapter_state_sha256"],
        "updates": updates,
        "elapsed_seconds": elapsed,
        "mechanics": endpoint["mechanics"],
        "trace": trace,
    }
    del wrapper, adapter, base, optimizer, scheduler, ema, endpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _validate_formal_args(args: argparse.Namespace) -> None:
    if not args.execute or args.acknowledge_protocol != ACKNOWLEDGEMENT:
        raise PermissionError(
            "formal Real4k requires --execute and the exact acknowledgement")
    required = (
        args.base_checkpoint, args.base_checkpoint_sha256,
        args.real_dir, args.clean_manifest, args.output_dir,
    )
    if any(value is None for value in required):
        raise ValueError("formal Real4k requires all path/SHA arguments")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal matched Real4k requires CUDA")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "contract":
        return experiment_contract()
    if args.mode == "formal":
        _validate_formal_args(args)
        arm_updates = UPDATES
    else:
        required = (
            args.base_checkpoint, args.base_checkpoint_sha256,
            args.real_dir, args.clean_manifest, args.output_dir,
        )
        if any(value is None for value in required):
            raise ValueError("checkpoint-smoke requires all path/SHA arguments")
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to reuse output {args.output_dir}")
        arm_updates = 2
    device = torch.device(args.device)
    payload, config, base_state, source = load_base_payload(
        args.base_checkpoint, args.base_checkpoint_sha256)
    rows, manifest_report = real4k.load_clean_manifest(args.clean_manifest)
    access = real4k.PrefixAccessAudit()
    trajectories, sparse_windows = real4k.load_real_rows(
        args.real_dir, rows, phase="fit", audit=access)
    normalizer, normalizer_report = real4k.normalizer_from_sparse_windows(
        trajectories, sparse_windows, device)
    optimizer_windows = real4k.dense_windows(trajectories)
    if len(optimizer_windows) != real4k.EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError("dense Real optimizer window count changed")
    dataset = real4k.RealWindowDataset(trajectories, optimizer_windows)
    schedule = real4k.make_sample_schedule(
        len(dataset), UPDATES, BATCH_SIZE)
    source.update({
        "clean_manifest_semantic_sha256": manifest_report["semantic_sha256"],
        "clean_manifest_file_sha256": sha256(args.clean_manifest),
        "normalizer_sha256": real4k.tensor_state_sha(
            normalizer_payload(normalizer)),
        "optimizer_windows": len(dataset),
        "optimizer_stride": real4k.OPTIMIZER_STRIDE,
    })
    args.output_dir.mkdir(parents=True)
    reports = []
    for arm in ARMS:
        reports.append(train_arm(
            arm=arm, config=config, base_state=base_state,
            dataset=dataset, normalizer=normalizer, schedule=schedule,
            source=source, output_dir=args.output_dir,
            device=device, updates=arm_updates))
    result = {
        "schema": SCHEMA,
        "mode": args.mode,
        "contract": experiment_contract(),
        "base": source,
        "manifest": manifest_report,
        "normalizer": normalizer_report,
        "arms": reports,
        "clean716_opened": False,
        "submission_authorized": False,
    }
    result_path = args.output_dir / "result.json"
    temporary = result_path.with_name(result_path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(result_path)
    del payload, base_state
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_causal_phase_direction_complete",
        "schema": result["schema"],
        "mode": args.mode,
        "output_dir": None if args.output_dir is None else str(args.output_dir),
    }, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
