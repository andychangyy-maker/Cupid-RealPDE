#!/usr/bin/env python3
"""Controller-only Sim functional rehearsal for B210 Real adaptation.

The frozen Sim endpoint supplies a per-sample physical ``delta_logK`` target.
During Real training the current backbone is evaluated on released-Sim
histories without gradients, while only the energy controller is trained to
retain that physical function.  No teacher model is needed after the cache has
been built and the inference graph is unchanged.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_real_sim_rehearsal as rehearsal


SIM_STRIDE = 20
SIM_TRAJECTORIES = 100
CACHE_BATCH_SIZE = 16
CALIBRATION_UPDATE = 100
TARGET_CONTROLLER_GRADIENT_RATIO = 0.15
MIN_CALIBRATED_WEIGHT = 1.0e-4
MAX_CALIBRATED_WEIGHT = 10.0
INITIAL_IDENTITY_MAE_MAX = 5.0e-4


def _checkpoint_model(payload: Mapping[str, object],
                      device: torch.device) -> EnsembleUNet2d:
    raw = dict(payload.get("model_config", {}))
    if raw.pop("architecture", None) != "ensemble_unet":
        raise ValueError("functional teacher is not an ensemble U-Net")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**raw)).float()
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("functional teacher has no model state")
    model.load_state_dict(state, strict=True)
    if (model.energy_controller is None
            or not model.config.physical_energy_multiscale):
        raise ValueError(
            "functional rehearsal requires a multiscale energy controller")
    return model.to(device)


def bounded_energy_delta(model: EnsembleUNet2d,
                         history_norm: torch.Tensor,
                         *, detach_backbone: bool) -> torch.Tensor:
    """Return the deployed physical log-energy correction."""
    if detach_backbone:
        with torch.no_grad():
            features = model._decode(history_norm)
        features = [feature.detach() for feature in features]
        history_for_controller = history_norm.detach()
    else:
        features = model._decode(history_norm)
        history_for_controller = history_norm
    raw = model._physical_energy_delta(features, history_for_controller)
    return model.config.physical_energy_log_limit * torch.tanh(raw)


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def build_target_cache(
        start_payload: Mapping[str, object], dataset,
        sim_normalizer, device: torch.device,
        ) -> tuple[torch.Tensor, dict]:
    """Build a compact FP16 cache before the first Real optimizer step."""
    teacher = _checkpoint_model(start_payload, device).eval()
    rehearsal.ensemble_train.common.set_model_stats(
        teacher, sim_normalizer)
    loader = DataLoader(
        dataset, batch_size=CACHE_BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda")
    chunks = []
    with torch.inference_mode():
        for history, target in loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            history_norm, _target_norm = sim_normalizer.preprocess(
                history, target)
            chunks.append(bounded_energy_delta(
                teacher, history_norm, detach_backbone=False).half().cpu())
    cache = torch.cat(chunks, dim=0).contiguous()
    if len(cache) != len(dataset) or cache.ndim != 4 or cache.shape[1] != 1:
        raise RuntimeError("functional energy cache shape changed")
    finite = bool(torch.isfinite(cache).all())
    nontrivial = float((cache.float().abs() > 1.0e-3).float().mean())
    if not finite or nontrivial < 0.95:
        raise RuntimeError("functional energy teacher is trivial or non-finite")
    report = {
        "samples": len(cache),
        "shape": list(cache.shape),
        "dtype": str(cache.dtype),
        "sha256": tensor_sha256(cache),
        "delta_logk_mean": float(cache.float().mean()),
        "delta_logk_std": float(cache.float().std(unbiased=False)),
        "delta_logk_abs_mean": float(cache.float().abs().mean()),
        "delta_logk_abs_max": float(cache.float().abs().max()),
        "nontrivial_fraction": nontrivial,
    }
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache, report


def prepare_sim(args, start_payload: Mapping[str, object],
                device: torch.device) -> tuple[dict, object]:
    """Prepare released-Sim histories and a frozen correction cache."""
    sim_normalizer, values, normalizer_sha = (
        rehearsal.frozen_sim_normalizer(dict(start_payload), device))
    trajectories, windows, manifest = (
        rehearsal.ensemble_train.common.scratch.load_simulations(
            args.sim_dir, 0, SIM_STRIDE, True))
    if len(trajectories) != SIM_TRAJECTORIES or len(windows) != 4800:
        raise RuntimeError("released-Sim functional inventory changed")
    dataset = rehearsal.ensemble_train.common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    cache, cache_report = build_target_cache(
        start_payload, dataset, sim_normalizer, device)
    tagged = rehearsal.ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_size=small4k.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(small4k.SEED))
    return ({
        "loader": loader,
        "normalizer": sim_normalizer,
        "cache": cache,
        "cache_report": cache_report,
        "manifest": manifest,
        "manifest_sha256": dr1.canonical_sha256(manifest),
        "normalizer_values": values,
        "normalizer_sha256": normalizer_sha,
        "windows": len(windows),
        "trajectories": len(trajectories),
    }, dataset)


def _gradient_norm(values) -> torch.Tensor:
    pieces = [value.detach().float().square().sum()
              for value in values if value is not None]
    if not pieces:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(pieces).sum().sqrt()


def _gradient_cosine(left, right) -> float:
    dots = []
    left_norms = []
    right_norms = []
    for first, second in zip(left, right):
        if first is None or second is None:
            continue
        first = first.detach().float()
        second = second.detach().float()
        dots.append((first * second).sum())
        left_norms.append(first.square().sum())
        right_norms.append(second.square().sum())
    if not dots:
        return float("nan")
    denominator = (
        torch.stack(left_norms).sum().sqrt()
        * torch.stack(right_norms).sum().sqrt()).clamp_min(1.0e-12)
    return float(torch.stack(dots).sum() / denominator)


class ControllerFunctionalBackward:
    """Delayed, once-calibrated controller-only functional distillation."""

    def __init__(self, sim: dict, real_normalizer, device: torch.device,
                 *, calibration_update: int = CALIBRATION_UPDATE,
                 target_gradient_ratio: float = (
                     TARGET_CONTROLLER_GRADIENT_RATIO)) -> None:
        if calibration_update < 2:
            raise ValueError("functional calibration must follow initial drift")
        if not 0.0 < target_gradient_ratio < 1.0:
            raise ValueError("functional target gradient ratio must be in (0,1)")
        self.sim = sim
        self.real_normalizer = real_normalizer
        self.device = device
        self.calibration_update = int(calibration_update)
        self.target_gradient_ratio = float(target_gradient_ratio)
        self.iterator = rehearsal.ensemble_train.common.repro.cycling(
            sim["loader"])
        self.rows: list[dict] = []
        self.weight: float | None = None
        self.calibration: dict | None = None
        self.initial_identity_mae: float | None = None

    def contract(self) -> dict:
        return {
            "order": (
                "Real backward -> controller functional backward -> clip -> "
                "optimizer -> EMA"),
            "target": "frozen Sim24k physical delta_logK",
            "backbone_gradient": False,
            "controller_gradient": True,
            "calibration_update": self.calibration_update,
            "target_controller_gradient_ratio": self.target_gradient_ratio,
            "weight_bounds": [MIN_CALIBRATED_WEIGHT, MAX_CALIBRATED_WEIGHT],
            "inference_graph_changed": False,
        }

    def __call__(self, iteration: int,
                 model: EnsembleUNet2d) -> Mapping[str, float]:
        batch = next(self.iterator)
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            raise RuntimeError("functional Sim batch is not tuple4")
        history, target, trajectory_ids, sample_indices = batch
        sample_indices = sample_indices.to(dtype=torch.long)
        self.rows.append({
            "iteration": iteration,
            "trajectory_ids": trajectory_ids.to(dtype=torch.long).tolist(),
            "sample_indices": sample_indices.tolist(),
        })
        history = history.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        was_training = model.training
        # The frozen target cache is deterministic/eval-mode.  The Real loss
        # has already built its graph when this callback runs, so evaluating
        # the separate released-Sim graph without dropout does not alter the
        # Real forward or its gradients.  Restore train mode before returning.
        model.eval()
        rehearsal.ensemble_train.common.set_model_stats(
            model, self.sim["normalizer"])
        try:
            history_norm, _target_norm = self.sim["normalizer"].preprocess(
                history, target)
            student = bounded_energy_delta(
                model, history_norm, detach_backbone=True)
            teacher = self.sim["cache"][sample_indices].to(
                device=self.device, dtype=torch.float32, non_blocking=True)
            raw_loss = F.mse_loss(student.float(), teacher)
            identity_mae = float((student.detach().float() - teacher).abs().mean())
            if iteration == 1:
                self.initial_identity_mae = identity_mae
                if identity_mae > INITIAL_IDENTITY_MAE_MAX:
                    raise RuntimeError(
                        "functional teacher/student update-0 identity failed: "
                        f"mae={identity_mae:.8g}")

            raw_gradient_norm = 0.0
            real_gradient_norm = 0.0
            calibrated_ratio = 0.0
            # The shared trainer requires every diagnostic to be finite.
            # Zero means "not measured" while the separate active/calibration
            # fields distinguish it from an actual calibrated cosine.
            gradient_cosine = 0.0
            if iteration == self.calibration_update:
                parameters = tuple(model.energy_controller.parameters())
                real_gradients = tuple(
                    None if value.grad is None else value.grad.detach().clone()
                    for value in parameters)
                functional_gradients = torch.autograd.grad(
                    raw_loss, parameters, retain_graph=True,
                    allow_unused=True)
                raw_norm = _gradient_norm(functional_gradients)
                real_norm = _gradient_norm(real_gradients)
                raw_gradient_norm = float(raw_norm)
                real_gradient_norm = float(real_norm)
                if (not math.isfinite(raw_gradient_norm)
                        or not math.isfinite(real_gradient_norm)
                        or raw_gradient_norm <= 0.0
                        or real_gradient_norm <= 0.0):
                    raise RuntimeError(
                        "functional gradient calibration is non-finite or zero")
                proposed = (
                    self.target_gradient_ratio
                    * real_gradient_norm / raw_gradient_norm)
                if not MIN_CALIBRATED_WEIGHT <= proposed <= MAX_CALIBRATED_WEIGHT:
                    raise RuntimeError(
                        "functional calibrated weight is outside frozen bounds: "
                        f"{proposed:.8g}")
                self.weight = float(proposed)
                calibrated_ratio = (
                    self.weight * raw_gradient_norm / real_gradient_norm)
                gradient_cosine = _gradient_cosine(
                    real_gradients, functional_gradients)
                self.calibration = {
                    "iteration": iteration,
                    "raw_functional_gradient_norm": raw_gradient_norm,
                    "real_controller_gradient_norm": real_gradient_norm,
                    "calibrated_weight": self.weight,
                    "calibrated_gradient_ratio": calibrated_ratio,
                    "real_functional_gradient_cosine": gradient_cosine,
                }

            active = self.weight is not None
            weighted_loss = raw_loss * (0.0 if not active else self.weight)
            if active:
                if not torch.isfinite(weighted_loss):
                    raise FloatingPointError(
                        "non-finite controller functional loss")
                weighted_loss.backward()
            values = {
                "controller_functional_raw_loss": float(raw_loss.detach()),
                "controller_functional_weighted_loss": float(
                    weighted_loss.detach()),
                "controller_functional_identity_mae": identity_mae,
                "controller_functional_active": float(active),
                "controller_functional_weight": float(self.weight or 0.0),
                "controller_functional_raw_gradient_norm": raw_gradient_norm,
                "controller_functional_real_gradient_norm": real_gradient_norm,
                "controller_functional_calibrated_ratio": calibrated_ratio,
                "controller_functional_gradient_cosine": gradient_cosine,
            }
        finally:
            rehearsal.ensemble_train.common.set_model_stats(
                model, self.real_normalizer)
            model.train(was_training)
        return values

    def report(self) -> dict:
        if len(self.rows) != small4k.UPDATES:
            raise RuntimeError("functional Sim stream did not reach 4000")
        if self.weight is None or self.calibration is None:
            raise RuntimeError("functional gradient was never calibrated")
        return {
            "kind": "controller_only_sim_delta_logk_distillation",
            "updates": len(self.rows),
            "active_updates": len(self.rows) - self.calibration_update + 1,
            "stream_sha256": dr1.canonical_sha256(self.rows),
            "calibration": dict(self.calibration),
            "initial_identity_mae": self.initial_identity_mae,
            "cache": dict(self.sim["cache_report"]),
            "manifest_sha256": self.sim["manifest_sha256"],
            "normalizer_sha256": self.sim["normalizer_sha256"],
            "windows": self.sim["windows"],
            "trajectories": self.sim["trajectories"],
            "inference_graph_changed": False,
        }
