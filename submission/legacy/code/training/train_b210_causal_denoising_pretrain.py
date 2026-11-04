#!/usr/bin/env python3
"""Pretrain the B210 backbone by denoising histories into true futures.

The task uses only released Sim windows and clean Real-prefix windows.  A
temporary multi-scale readout predicts the next 20 frames from a noisy 20-frame
history.  Only the shared backbone is retained; every deployed point/member/UQ
parameter stays bitwise equal to the standard seed-1 initialization.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

import train_b210_masked_reconstruction_pretrain as masked


METHOD = "b210_causal_denoising_future_sim_real_prefix_pretrain_v1"
TKE_AWARE_XCB_METHOD = (
    "b210_tke_aware_causal_future_xcb_member_sim_real_prefix_pretrain_v1")
UPDATES = 4_000
BATCH_PER_DOMAIN = 4
NOISE_ENERGY_SCALE = 5.0e-4
NOISE_SIGMA_MAX = 0.12
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
SEED = 1
LOG_INTERVAL = 250
TKE_HORIZONS = (5, 10, 20)
TKE_TARGET_GRADIENT_RATIO = 0.20
TKE_WEIGHT_MIN = 1.0e-4
TKE_WEIGHT_MAX = 10.0
XCB_SHARED_PREFIX = "cross_conditioner."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument(
        "--tke-aware-xcb-member", action="store_true",
        help=("pretrain the XCB/member architecture and add a physical "
              "multi-horizon TKE score loss with one-shot gradient scaling"))
    return parser.parse_args()


def history_noise_sigma(history: torch.Tensor) -> torch.Tensor:
    if history.ndim != 5:
        raise ValueError("history must be [B,T,H,W,C]")
    sigma = NOISE_ENERGY_SCALE * history.float().square().sum(
        dim=(1, 2, 3)).sqrt()
    return sigma.clamp(min=0.0, max=NOISE_SIGMA_MAX)


def corrupt_history(
        history: torch.Tensor, generator: torch.Generator,
        *, zero_pressure: bool) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = history_noise_sigma(history)
    noise = torch.randn(
        history.shape, dtype=history.dtype, device=history.device,
        generator=generator)
    if zero_pressure:
        if history.shape[-1] != 3:
            raise ValueError("Real history must have u/v/zero-pressure channels")
        noise[..., 2].zero_()
        sigma[:, 2].zero_()
    corrupted = history + noise * sigma[:, None, None, None, :]
    if zero_pressure:
        corrupted[..., 2].zero_()
    return corrupted, sigma


def balanced_relative_l2(
        prediction: torch.Tensor, target: torch.Tensor,
        channel_mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("future tensors must be aligned rank five")
    if channel_mask.shape != (prediction.shape[0], prediction.shape[-1]):
        raise ValueError("future channel mask is misaligned")
    if channel_mask.dtype != torch.bool or not bool(channel_mask.any()):
        raise ValueError("future channel mask must select boolean channels")
    error = (prediction.float() - target.float()).square().sum(dim=(1, 2, 3))
    energy = target.float().square().sum(dim=(1, 2, 3))
    relative = ((error + 1.0e-12) / (energy + 1.0e-8)).sqrt()
    return relative[channel_mask].mean()


def multi_horizon_physical_tke_score_loss(
        prediction_uv: torch.Tensor, target_uv: torch.Tensor,
        horizons: tuple[int, ...] = TKE_HORIZONS,
        ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Official-shaped physical TKE score loss at fixed prefix horizons."""
    if (prediction_uv.shape != target_uv.shape or prediction_uv.ndim != 5
            or prediction_uv.shape[-1] != 2):
        raise ValueError("physical TKE tensors must be aligned [B,T,H,W,2]")
    if (not horizons or tuple(sorted(set(horizons))) != horizons
            or horizons[-1] > prediction_uv.shape[1]
            or horizons[0] < 2):
        raise ValueError("TKE horizons must be unique ordered valid prefixes")
    losses: dict[int, torch.Tensor] = {}
    for horizon in horizons:
        prediction = prediction_uv[:, :horizon].float()
        target = target_uv[:, :horizon].float()
        prediction = prediction - prediction.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        prediction_tke = 0.5 * prediction.square().mean(dim=1).sum(dim=-1)
        target_tke = 0.5 * target.square().mean(dim=1).sum(dim=-1)
        batch = prediction_tke.shape[0]
        relative_error = torch.linalg.vector_norm(
            (prediction_tke - target_tke).reshape(batch, -1), dim=1
        ) / torch.linalg.vector_norm(
            target_tke.reshape(batch, -1), dim=1).clamp_min(1.0e-8)
        losses[horizon] = (
            1.0 - 1.0 / (1.0 + 0.5 * relative_error)).mean()
    return torch.stack(tuple(losses.values())).mean(), losses


def physical_tke_score_loss(
        prediction: torch.Tensor, target: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    prediction_physical = (
        masked.b210.rehearsal.ensemble_train.common.to_physical_target(
            prediction, normalizer)[..., :2])
    target_physical = (
        masked.b210.rehearsal.ensemble_train.common.to_physical_target(
            target, normalizer)[..., :2])
    return multi_horizon_physical_tke_score_loss(
        prediction_physical, target_physical)


def gradient_geometry(
        primary: torch.Tensor, auxiliary: torch.Tensor,
        parameters: tuple[nn.Parameter, ...],
        ) -> dict[str, float]:
    primary_gradients = torch.autograd.grad(
        primary, parameters, retain_graph=True, allow_unused=True)
    auxiliary_gradients = torch.autograd.grad(
        auxiliary, parameters, retain_graph=True, allow_unused=True)
    primary_square = primary.new_zeros(())
    auxiliary_square = auxiliary.new_zeros(())
    dot = primary.new_zeros(())
    for first, second in zip(primary_gradients, auxiliary_gradients):
        if first is not None:
            primary_square += first.float().square().sum()
        if second is not None:
            auxiliary_square += second.float().square().sum()
        if first is not None and second is not None:
            dot += (first.float() * second.float()).sum()
    primary_norm = primary_square.sqrt()
    auxiliary_norm = auxiliary_square.sqrt()
    if (not torch.isfinite(primary_norm) or not torch.isfinite(auxiliary_norm)
            or float(primary_norm) <= 0.0 or float(auxiliary_norm) <= 0.0):
        raise FloatingPointError("causal pretext gradient calibration is invalid")
    cosine = dot / (primary_norm * auxiliary_norm).clamp_min(1.0e-12)
    return {
        "primary_gradient_norm": float(primary_norm.detach()),
        "raw_tke_gradient_norm": float(auxiliary_norm.detach()),
        "gradient_cosine": float(cosine.detach()),
    }


def xcb_shared_parameter_names(model: nn.Module) -> tuple[str, ...]:
    names = tuple(masked.shared_parameter_names(model))
    xcb = tuple(
        name for name, _parameter in model.named_parameters()
        if name.startswith(XCB_SHARED_PREFIX))
    if not xcb:
        raise RuntimeError("TKE-aware causal pretraining found no XCB parameters")
    return names + xcb


def configure_named_parameters(
        model: nn.Module, names: tuple[str, ...]) -> tuple[nn.Parameter, ...]:
    selected = set(names)
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
        if parameter.requires_grad:
            trainable.append(parameter)
    if len(trainable) != len(selected):
        raise RuntimeError("causal pretext trainable parameter names changed")
    return tuple(trainable)


def validate_args(args: argparse.Namespace) -> None:
    if args.updates != UPDATES:
        raise ValueError(f"formal causal pretraining requires {UPDATES} updates")
    if args.num_workers != 0:
        raise ValueError("formal clean-prefix audit requires num-workers=0")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    missing = [
        str(path) for path in (
            args.sim_dir, args.real_dir, args.real_stats, args.clean_manifest)
        if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal causal pretraining requires CUDA")
    args.output_dir.mkdir(parents=True)
    masked.b210.rehearsal.ensemble_train.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")

    sim_args = argparse.Namespace(
        sim_dir=args.sim_dir, max_files=0, window_stride=20,
        batch_size=BATCH_PER_DOMAIN, num_workers=args.num_workers,
        output_dir=args.output_dir, seed=SEED)
    sim_loader, sim_normalizer, sim_windows, sim_trajectories = (
        masked.b210.rehearsal.ensemble_train.common.prepare_sim(
            sim_args, device))
    if (sim_windows != masked.EXPECTED_SIM_WINDOWS
            or sim_trajectories != masked.EXPECTED_SIM_TRAJECTORIES):
        raise RuntimeError("released Sim causal-pretext contract changed")
    real_loader, real_access = masked.real_prefix_loader(
        args.real_dir, args.clean_manifest, args.num_workers)
    real_normalizer = (
        masked.b210.rehearsal.ensemble_train.common.repro.Normalizer(
            args.real_stats, device))

    method = TKE_AWARE_XCB_METHOD if args.tke_aware_xcb_member else METHOD
    model = masked.b210.build_arm_model(
        "candidate", device,
        cross_conditioned_backbone=args.tke_aware_xcb_member,
        member_feature_adapters=args.tke_aware_xcb_member)
    masked.b210.rehearsal.ensemble_train.common.set_model_stats(
        model, sim_normalizer)
    shared_names = (
        xcb_shared_parameter_names(model)
        if args.tke_aware_xcb_member
        else masked.shared_parameter_names(model))
    shared_parameters = configure_named_parameters(model, shared_names)
    decoder_channels = [
        model.config.base * 2 ** level
        for level in range(model.config.depth - 2, -1, -1)]
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(SEED + 51_000)
        torch.cuda.manual_seed_all(SEED + 51_000)
        readout = masked.ReconstructionReadout(
            decoder_channels, model.config.out_steps,
            model.config.out_channels).to(device)
    trainable = shared_parameters + tuple(readout.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES)
    noise_generator = torch.Generator(device=device).manual_seed(
        SEED + 61_000)
    sim_stream, real_stream = masked.cycle(sim_loader), masked.cycle(real_loader)
    history = []
    started = time.perf_counter()
    if args.contract_only:
        contract = {
            "method": method,
            "parameters": sum(p.numel() for p in model.parameters()),
            "shared_trainable_parameters": sum(
                p.numel() for p in shared_parameters),
            "temporary_readout_parameters": sum(
                p.numel() for p in readout.parameters()),
            "real_access": real_access,
            "tke_aware_xcb_member": args.tke_aware_xcb_member,
            "tke_horizons": list(TKE_HORIZONS),
            "tke_target_gradient_ratio": TKE_TARGET_GRADIENT_RATIO,
        }
        (args.output_dir / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps({"event": "causal_pretrain_contract", **contract}),
              flush=True)
        return

    initial_state = masked.model_state(model)
    tke_calibration = None
    for iteration in range(1, UPDATES + 1):
        sim_history, sim_future = next(sim_stream)[:2]
        real_history, real_future = next(real_stream)
        sim_history = sim_history.to(device, non_blocking=True)
        sim_future = sim_future.to(device, non_blocking=True)
        real_history = real_history.to(device, non_blocking=True)
        real_future = real_future.to(device, non_blocking=True)
        sim_clean, sim_target = sim_normalizer.preprocess(
            sim_history, sim_future)
        real_clean, real_target = real_normalizer.preprocess(
            real_history, real_future)
        sim_corrupted, sim_sigma = corrupt_history(
            sim_clean, noise_generator, zero_pressure=False)
        real_corrupted, real_sigma = corrupt_history(
            real_clean, noise_generator, zero_pressure=True)
        corrupted = torch.cat((sim_corrupted, real_corrupted), dim=0)
        target = torch.cat((sim_target, real_target), dim=0)
        channel_mask = torch.ones(
            len(corrupted), target.shape[-1], dtype=torch.bool, device=device)
        channel_mask[BATCH_PER_DOMAIN:, 2] = False

        model.train()
        readout.train()
        optimizer.zero_grad(set_to_none=True)
        features = model._decode(corrupted)
        prediction = readout(features, (target.shape[2], target.shape[3]))
        primary_loss = balanced_relative_l2(prediction, target, channel_mask)
        sim_loss = balanced_relative_l2(
            prediction[:BATCH_PER_DOMAIN], target[:BATCH_PER_DOMAIN],
            channel_mask[:BATCH_PER_DOMAIN])
        real_loss = balanced_relative_l2(
            prediction[BATCH_PER_DOMAIN:], target[BATCH_PER_DOMAIN:],
            channel_mask[BATCH_PER_DOMAIN:])
        sim_tke_loss = primary_loss.new_zeros(())
        real_tke_loss = primary_loss.new_zeros(())
        tke_loss = primary_loss.new_zeros(())
        tke_weight = 0.0
        sim_tke_horizons: dict[int, torch.Tensor] = {}
        real_tke_horizons: dict[int, torch.Tensor] = {}
        if args.tke_aware_xcb_member:
            sim_tke_loss, sim_tke_horizons = physical_tke_score_loss(
                prediction[:BATCH_PER_DOMAIN], target[:BATCH_PER_DOMAIN],
                sim_normalizer)
            real_tke_loss, real_tke_horizons = physical_tke_score_loss(
                prediction[BATCH_PER_DOMAIN:], target[BATCH_PER_DOMAIN:],
                real_normalizer)
            tke_loss = 0.5 * (sim_tke_loss + real_tke_loss)
            if tke_calibration is None:
                tke_calibration = gradient_geometry(
                    primary_loss, tke_loss, shared_parameters)
                calibrated_weight = (
                    TKE_TARGET_GRADIENT_RATIO
                    * tke_calibration["primary_gradient_norm"]
                    / tke_calibration["raw_tke_gradient_norm"])
                if not TKE_WEIGHT_MIN <= calibrated_weight <= TKE_WEIGHT_MAX:
                    raise RuntimeError(
                        "TKE-aware causal calibration weight outside frozen "
                        f"bounds: {calibrated_weight}")
                tke_calibration.update({
                    "iteration": iteration,
                    "target_gradient_ratio": TKE_TARGET_GRADIENT_RATIO,
                    "calibrated_weight": calibrated_weight,
                    "calibrated_gradient_ratio": (
                        calibrated_weight
                        * tke_calibration["raw_tke_gradient_norm"]
                        / tke_calibration["primary_gradient_norm"]),
                })
                print(json.dumps({
                    "event": "causal_tke_calibration", **tke_calibration,
                }), flush=True)
            tke_weight = float(tke_calibration["calibrated_weight"])
        loss = primary_loss + tke_weight * tke_loss
        loss.backward()
        pre_clip_norm = masked.gradient_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        if (not math.isfinite(float(loss)) or pre_clip_norm <= 0.0
                or not math.isfinite(pre_clip_norm)):
            raise FloatingPointError("causal pretraining became non-finite")
        if iteration % LOG_INTERVAL == 0 or iteration == 1:
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "primary_loss": float(primary_loss.detach()),
                "sim_loss": float(sim_loss.detach()),
                "real_loss": float(real_loss.detach()),
                "tke_loss": float(tke_loss.detach()),
                "sim_tke_loss": float(sim_tke_loss.detach()),
                "real_tke_loss": float(real_tke_loss.detach()),
                "tke_weight": tke_weight,
                "sim_tke_by_horizon": {
                    str(key): float(value.detach())
                    for key, value in sim_tke_horizons.items()},
                "real_tke_by_horizon": {
                    str(key): float(value.detach())
                    for key, value in real_tke_horizons.items()},
                "sim_sigma_mean": float(sim_sigma.mean()),
                "real_sigma_mean": float(real_sigma[:, :2].mean()),
                "real_pressure_exact_zero": bool(
                    torch.count_nonzero(real_corrupted[..., 2]) == 0),
                "gradient_norm": pre_clip_norm,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(row)
            print(json.dumps({"event": "causal_pretrain_progress", **row}),
                  flush=True)

    final_state = masked.model_state(model)
    shared_names = set(shared_names)
    changed_shared = [
        name for name in shared_names
        if not torch.equal(initial_state[name], final_state[name])]
    untouched = [name for name in final_state if name not in shared_names]
    changed_untouched = [
        name for name in untouched
        if not torch.equal(initial_state[name], final_state[name])]
    if len(changed_shared) != len(shared_names):
        raise RuntimeError("not every shared backbone parameter was updated")
    if changed_untouched:
        raise RuntimeError(
            f"deployed non-backbone state changed: {changed_untouched[:5]}")

    checkpoint = {
        "phase": "sim_real_prefix_causal_pretrain",
        "iteration": UPDATES,
        "model_config": model.architecture_config(),
        "model_state_dict": final_state,
        "normalizer": (
            masked.b210.rehearsal.ensemble_train.common.normalizer_dict(
                sim_normalizer)),
        "training_config": {
            "method": method,
            "updates": UPDATES,
            "batch_per_domain": BATCH_PER_DOMAIN,
            "noise_energy_scale": NOISE_ENERGY_SCALE,
            "noise_sigma_max": NOISE_SIGMA_MAX,
            "loss": (
                "sample_channel_balanced_relative_l2_plus_"
                "gradient_calibrated_multihorizon_physical_tke_score"
                if args.tke_aware_xcb_member
                else "sample_channel_balanced_relative_l2"),
            "real_loss_channels": ["u", "v"],
            "tke_aware_xcb_member": args.tke_aware_xcb_member,
            "tke_horizons": list(TKE_HORIZONS),
            "tke_gradient_calibration": tke_calibration,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "seed": SEED,
            "sim_windows": sim_windows,
            "real_prefix_windows": real_access["windows"],
            "external_data_or_model": False,
            "tail_arrays_read": 0,
        },
        "pretext_readout_state_dict": masked.model_state(readout),
        "history": history,
        "pretraining_audit": {
            "shared_parameters": len(shared_names),
            "changed_shared_parameters": len(changed_shared),
            "untouched_deployed_state_tensors": len(untouched),
            "changed_untouched_state_tensors": changed_untouched,
            "initial_non_backbone_sha256": masked.state_sha256(OrderedDict(
                (name, initial_state[name]) for name in untouched)),
            "final_non_backbone_sha256": masked.state_sha256(OrderedDict(
                (name, final_state[name]) for name in untouched)),
            "real_access": real_access,
        },
    }
    checkpoint_path = args.output_dir / "causal_pretrain_4000.pth"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "method": METHOD,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": masked.sha256(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "terminal": history[-1],
        "training_config": checkpoint["training_config"],
        "pretraining_audit": checkpoint["pretraining_audit"],
        "next_stage_authorized": True,
        "submission_authorized": False,
    }
    summary_path = args.output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps({
        "event": "causal_pretrain_complete",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "terminal": history[-1],
        "elapsed_seconds": summary["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
