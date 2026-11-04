#!/usr/bin/env python3
"""Self-supervise the standard B210 backbone on masked Sim + Real histories.

Only the shared stem/encoder/bottleneck/decoder representation is updated.
All deployed point, member, controller and UQ parameters remain exactly at
the standard seed-1 initialization and are saved with the pretrained backbone
for a subsequent ordinary supervised Sim stage.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import train_b210_np_functional_sim as b210


METHOD = "b210_masked_reconstruction_sim_real_prefix_pretrain_v1"
UPDATES = 4_000
BATCH_PER_DOMAIN = 4
MASK_RATIO = 0.70
MASK_GRID = (8, 16)
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
SEED = 1
LOG_INTERVAL = 250
FORMAL_PREFIX_MANIFEST_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
EXPECTED_REAL_TRAJECTORIES = 81
EXPECTED_REAL_WINDOWS = 2544
EXPECTED_SIM_TRAJECTORIES = 100
EXPECTED_SIM_WINDOWS = 4800
SHARED_PREFIXES = (
    "stem.", "encoder.", "bottleneck.", "regime_film.", "up.",
    "decoder.", "decoder_regime_film.")


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
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def model_state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.state_dict().items())


def shared_parameter_names(model: nn.Module) -> tuple[str, ...]:
    names = tuple(
        name for name, _parameter in model.named_parameters()
        if name.startswith(SHARED_PREFIXES))
    if not names:
        raise RuntimeError("masked pretraining found no shared backbone")
    return names


def configure_shared_backbone(model: nn.Module) -> tuple[nn.Parameter, ...]:
    names = set(shared_parameter_names(model))
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
        if parameter.requires_grad:
            trainable.append(parameter)
    return tuple(trainable)


class ReconstructionReadout(nn.Module):
    """Temporary multi-scale history decoder; never enters deployment."""

    def __init__(self, channels: list[int], out_steps: int,
                 out_channels: int):
        super().__init__()
        self.out_steps = int(out_steps)
        self.out_channels = int(out_channels)
        self.heads = nn.ModuleList(
            nn.Conv2d(width, out_steps * out_channels, 1)
            for width in channels)

    def forward(self, features: list[torch.Tensor],
                size: tuple[int, int]) -> torch.Tensor:
        native = [head(feature) for head, feature in zip(self.heads, features)]
        raw = b210.EnsembleUNet2d._multi_scale_values(native, size)
        batch = raw.shape[0]
        return raw.reshape(
            batch, self.out_steps, self.out_channels, *size
        ).permute(0, 1, 3, 4, 2)


def spatial_mask(batch: int, height: int, width: int,
                 generator: torch.Generator, device: torch.device,
                 ratio: float = MASK_RATIO) -> torch.Tensor:
    if not 0.0 < ratio < 1.0:
        raise ValueError("mask ratio must lie in (0,1)")
    coarse = torch.rand(
        batch, 1, *MASK_GRID, generator=generator, device=device) < ratio
    mask = F.interpolate(
        coarse.float(), size=(height, width), mode="nearest").bool()
    return mask[:, 0]


def masked_reconstruction_loss(prediction: torch.Tensor,
                               target: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("reconstruction tensors must be aligned rank five")
    if mask.shape != (prediction.shape[0], prediction.shape[2],
                      prediction.shape[3]):
        raise ValueError("spatial reconstruction mask is misaligned")
    selected = mask[:, None, :, :, None].expand_as(prediction)
    if not bool(selected.any()):
        raise RuntimeError("masked reconstruction selected no values")
    return (prediction.float() - target.float()).square()[selected].mean()


def _normalized_manifest(path: Path) -> list[dict]:
    if sha256(path) != FORMAL_PREFIX_MANIFEST_SHA256:
        raise ValueError("clean Real prefix manifest changed")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != EXPECTED_REAL_TRAJECTORIES:
        raise RuntimeError("clean Real manifest must contain 81 trajectories")
    rows = []
    seen = set()
    for value in raw:
        filename = str(value["file"])
        if Path(filename).name != filename or filename in seen:
            raise ValueError(f"unsafe or duplicate Real file: {filename}")
        seen.add(filename)
        starts = [int(item) for item in value["train_starts"]]
        boundary = int(value["val_boundary"])
        if not starts or any(start + 40 > boundary for start in starts):
            raise RuntimeError("Real pretext window crosses the tail boundary")
        rows.append({
            "file": filename,
            "frames": int(value["frames"]),
            "train_starts": starts,
            "val_boundary": boundary,
        })
    if sum(len(row["train_starts"]) for row in rows) != EXPECTED_REAL_WINDOWS:
        raise RuntimeError("clean Real prefix window count changed")
    return rows


def real_velocity_channels(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Keep raw Real trajectories at u/v for the shared dataset adapter.

    ``repro.WindowDataset`` is the canonical place that appends one exact-zero
    pressure channel.  Adding it here as well would make four-channel batches.
    """
    if u.shape != v.shape or u.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError("Real u/v arrays must be aligned float32 values")
    value = np.stack((u, v), axis=-1)
    if value.shape[-1] != 2:
        raise RuntimeError("Real velocity channel construction failed")
    return np.ascontiguousarray(value)


def real_prefix_loader(real_dir: Path, manifest: Path,
                       num_workers: int) -> tuple[DataLoader, dict]:
    rows = _normalized_manifest(manifest)
    root = real_dir.resolve()
    trajectories = []
    windows = []
    max_read = {}
    for trajectory, row in enumerate(rows):
        path = (root / row["file"]).resolve()
        if path.parent != root or path.name != row["file"]:
            raise ValueError(f"unsafe Real path: {row['file']}")
        boundary = row["val_boundary"]
        with h5py.File(path, "r") as handle:
            if int(handle["u"].shape[0]) != row["frames"]:
                raise RuntimeError(f"Real frame count changed: {row['file']}")
            u = np.asarray(handle["u"][:boundary, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][:boundary, ::2, ::2], dtype=np.float32)
        trajectories.append(torch.from_numpy(real_velocity_channels(u, v)))
        windows.extend(
            b210.rehearsal.ensemble_train.common.repro.Window(
                trajectory, start) for start in row["train_starts"])
        max_read[row["file"]] = boundary
    dataset = b210.rehearsal.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    loader = DataLoader(
        dataset, batch_size=BATCH_PER_DOMAIN, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(SEED + 7))
    return loader, {
        "trajectories": len(rows),
        "windows": len(windows),
        "max_read_frame_exclusive": max_read,
        "tail_arrays_read": 0,
    }


def cycle(loader: DataLoader):
    while True:
        yield from loader


def gradient_norm(parameters: tuple[nn.Parameter, ...]) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def validate_args(args: argparse.Namespace) -> None:
    if args.updates != UPDATES:
        raise ValueError(f"formal masked pretraining requires {UPDATES} updates")
    if args.num_workers != 0:
        raise ValueError("formal clean-prefix audit requires num-workers=0")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required = (args.sim_dir, args.real_dir, args.real_stats,
                args.clean_manifest)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal masked pretraining requires CUDA")
    args.output_dir.mkdir(parents=True)
    b210.rehearsal.ensemble_train.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")

    sim_args = argparse.Namespace(
        sim_dir=args.sim_dir, max_files=0, window_stride=20,
        batch_size=BATCH_PER_DOMAIN, num_workers=args.num_workers,
        output_dir=args.output_dir, seed=SEED)
    sim_loader, sim_normalizer, sim_windows, sim_trajectories = (
        b210.rehearsal.ensemble_train.common.prepare_sim(sim_args, device))
    if (sim_windows != EXPECTED_SIM_WINDOWS
            or sim_trajectories != EXPECTED_SIM_TRAJECTORIES):
        raise RuntimeError("released Sim pretext data contract changed")
    real_loader, real_access = real_prefix_loader(
        args.real_dir, args.clean_manifest, args.num_workers)
    real_normalizer = b210.rehearsal.ensemble_train.common.repro.Normalizer(
        args.real_stats, device)

    model = b210.build_arm_model("candidate", device)
    b210.rehearsal.ensemble_train.common.set_model_stats(
        model, sim_normalizer)
    shared_parameters = configure_shared_backbone(model)
    decoder_channels = [
        model.config.base * 2 ** level
        for level in range(model.config.depth - 2, -1, -1)]
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(SEED + 31_000)
        torch.cuda.manual_seed_all(SEED + 31_000)
        readout = ReconstructionReadout(
            decoder_channels, model.config.in_steps,
            model.config.in_channels).to(device)
    trainable = shared_parameters + tuple(readout.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES)
    mask_generator = torch.Generator(device=device).manual_seed(SEED + 41_000)
    sim_stream, real_stream = cycle(sim_loader), cycle(real_loader)
    history = []
    started = time.perf_counter()
    if args.contract_only:
        write = {
            "method": METHOD,
            "parameters": sum(p.numel() for p in model.parameters()),
            "shared_trainable_parameters": sum(p.numel() for p in shared_parameters),
            "temporary_readout_parameters": sum(p.numel() for p in readout.parameters()),
            "real_access": real_access,
        }
        (args.output_dir / "contract.json").write_text(
            json.dumps(write, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps({"event": "masked_pretrain_contract", **write}), flush=True)
        return

    initial_state = model_state(model)
    for iteration in range(1, UPDATES + 1):
        sim_batch = next(sim_stream)
        real_history, real_target = next(real_stream)
        sim_history = sim_batch[0].to(device, non_blocking=True)
        real_history = real_history.to(device, non_blocking=True)
        sim_norm, sim_target = sim_normalizer.preprocess(
            sim_history, sim_history)
        real_norm, real_target_norm = real_normalizer.preprocess(
            real_history, real_history)
        clean = torch.cat((sim_norm, real_norm), dim=0)
        target = torch.cat((sim_target, real_target_norm), dim=0)
        mask = spatial_mask(
            len(clean), clean.shape[2], clean.shape[3],
            mask_generator, device)
        corrupted = clean.masked_fill(mask[:, None, :, :, None], 0.0)
        model.train()
        readout.train()
        optimizer.zero_grad(set_to_none=True)
        features = model._decode(corrupted)
        prediction = readout(features, (clean.shape[2], clean.shape[3]))
        loss = masked_reconstruction_loss(prediction, target, mask)
        sim_loss = masked_reconstruction_loss(
            prediction[:BATCH_PER_DOMAIN], target[:BATCH_PER_DOMAIN],
            mask[:BATCH_PER_DOMAIN])
        real_loss = masked_reconstruction_loss(
            prediction[BATCH_PER_DOMAIN:], target[BATCH_PER_DOMAIN:],
            mask[BATCH_PER_DOMAIN:])
        loss.backward()
        pre_clip_norm = gradient_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        if (not math.isfinite(float(loss)) or pre_clip_norm <= 0.0
                or not math.isfinite(pre_clip_norm)):
            raise FloatingPointError("masked pretraining became non-finite")
        if iteration % LOG_INTERVAL == 0 or iteration == 1:
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "sim_loss": float(sim_loss.detach()),
                "real_loss": float(real_loss.detach()),
                "mask_fraction": float(mask.float().mean()),
                "gradient_norm": pre_clip_norm,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(row)
            print(json.dumps({"event": "masked_pretrain_progress", **row}),
                  flush=True)

    final_state = model_state(model)
    shared_names = set(shared_parameter_names(model))
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

    normalizer_values = (
        b210.rehearsal.ensemble_train.common.normalizer_dict(sim_normalizer))
    checkpoint = {
        "phase": "sim_selfsupervised_pretrain",
        "iteration": UPDATES,
        "model_config": model.architecture_config(),
        "model_state_dict": final_state,
        "normalizer": normalizer_values,
        "training_config": {
            "method": METHOD,
            "updates": UPDATES,
            "batch_per_domain": BATCH_PER_DOMAIN,
            "mask_ratio": MASK_RATIO,
            "mask_grid": list(MASK_GRID),
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "seed": SEED,
            "sim_windows": sim_windows,
            "real_prefix_windows": real_access["windows"],
            "external_data_or_model": False,
            "tail_arrays_read": 0,
        },
        "pretext_readout_state_dict": model_state(readout),
        "history": history,
        "pretraining_audit": {
            "shared_parameters": len(shared_names),
            "changed_shared_parameters": len(changed_shared),
            "untouched_deployed_state_tensors": len(untouched),
            "changed_untouched_state_tensors": changed_untouched,
            "initial_non_backbone_sha256": state_sha256(OrderedDict(
                (name, initial_state[name]) for name in untouched)),
            "final_non_backbone_sha256": state_sha256(OrderedDict(
                (name, final_state[name]) for name in untouched)),
            "real_access": real_access,
        },
    }
    checkpoint_path = args.output_dir / "masked_pretrain_4000.pth"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "method": METHOD,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "terminal": history[-1],
        "training_config": checkpoint["training_config"],
        "pretraining_audit": checkpoint["pretraining_audit"],
        "next_stage_authorized": True,
        "submission_authorized": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "masked_pretrain_complete",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "terminal": history[-1],
        "elapsed_seconds": summary["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
