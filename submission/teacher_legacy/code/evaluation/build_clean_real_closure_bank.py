#!/usr/bin/env python3
"""Build a Real-closure bank from exactly the frozen clean training prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from physics.flow_rollout import FlowRolloutConfig, rollout_velocity  # noqa: E402


FORMAT = "realpde_clean_real_closure_bank_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(path: Path, expected_sha256: str) -> list[dict]:
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("train manifest SHA-256 must be 64 lowercase hex digits")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"train manifest SHA-256 {actual} != {expected}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("train manifest must be a non-empty list")
    files = [row.get("file") for row in rows]
    if any(not isinstance(name, str) or not name.endswith(".h5") for name in files):
        raise ValueError("manifest contains an invalid file name")
    if len(files) != len(set(files)):
        raise ValueError("manifest contains duplicate files")
    if "7575_0.h5" in files:
        raise ValueError("known invalid trajectory must not enter clean closure bank")
    return rows


@torch.no_grad()
def closure_for_row(real_dir: Path, row: dict, batch_size: int,
                    device: torch.device) -> dict:
    required = {"file", "frames", "re", "aoa", "train_starts", "val_boundary"}
    if set(row) != required:
        raise ValueError(f"unexpected clean manifest schema for {row.get('file')}")
    path = real_dir / row["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    prefix_frames = int(row["val_boundary"])
    if prefix_frames < 40 or not row["train_starts"]:
        raise ValueError(f"invalid clean training prefix for {path.name}")
    if max(row["train_starts"]) + 40 > prefix_frames:
        raise ValueError(f"optimizer window exceeds closure prefix for {path.name}")
    with h5py.File(path, "r") as handle:
        frames = int(handle["u"].shape[0])
        re_value = int(np.asarray(handle["re"]))
        aoa = int(np.asarray(handle["aoa"]))
        if (frames, re_value, aoa) != (
                int(row["frames"]), int(row["re"]), int(row["aoa"])):
            raise ValueError(f"manifest/HDF5 metadata mismatch for {path.name}")
        if prefix_frames > frames:
            raise ValueError(f"prefix exceeds trajectory for {path.name}")
        u = np.asarray(handle["u"][:prefix_frames, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle["v"][:prefix_frames, ::2, ::2], dtype=np.float32)
        x = np.asarray(handle["x"][::2, ::2], dtype=np.float64)
        y = np.asarray(handle["y"][::2, ::2], dtype=np.float64)
        t = np.asarray(handle["t"][:prefix_frames], dtype=np.float64)
    if u.shape[1:] != (32, 64) or v.shape != u.shape:
        raise ValueError(f"unexpected downsampled shape for {path.name}: {u.shape}")
    dx = float(np.median(np.abs(np.diff(x, axis=1))))
    dy = float(np.median(np.abs(np.diff(y, axis=0))))
    dt = float(np.median(np.diff(t)))
    if not all(np.isfinite(value) and value > 0 for value in (dx, dy, dt)):
        raise ValueError(f"invalid physical spacing for {path.name}")
    freestream = re_value * 1.0e-5
    velocity = torch.from_numpy(np.stack((u, v), axis=-1))
    config = FlowRolloutConfig(advection_scale=1.0, viscosity=1.0e-6)
    residuals = []
    for begin in range(0, prefix_frames - 1, batch_size):
        end = min(prefix_frames - 1, begin + batch_size)
        current = velocity[begin:end].unsqueeze(1).to(
            device=device, non_blocking=True)
        following = velocity[begin + 1:end + 1].to(
            device=device, non_blocking=True)
        predicted = rollout_velocity(
            current, 1, config, dt=dt, dx=dx, dy=dy,
            velocity_scale=1.0)[:, 0]
        fluid = following.abs().sum(dim=-1, keepdim=True).ne(0.0)
        residual = ((following - predicted) / freestream) * fluid
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError(f"non-finite closure residual for {path.name}")
        residuals.append(residual.to(device="cpu", dtype=torch.float16))
    return {
        "file": path.name,
        "re": re_value,
        "aoa": aoa,
        "frames": frames,
        "prefix_frames": prefix_frames,
        "train_starts": list(row["train_starts"]),
        "spacing": {"dt": dt, "dx": dx, "dy": dy},
        "residual": torch.cat(residuals, dim=0).contiguous(),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    rows = validate_manifest(args.train_manifest, args.train_manifest_sha256)
    expected_files = {row["file"] for row in rows}
    actual_files = {path.name for path in args.real_dir.glob("*.h5")}
    if actual_files != expected_files:
        raise ValueError(
            "Real directory must contain exactly the clean manifest files; "
            f"missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}")
    device = torch.device(args.device)
    started = time.monotonic()
    records = []
    for index, row in enumerate(rows, 1):
        record = closure_for_row(
            args.real_dir, row, args.batch_size, device)
        records.append(record)
        print(json.dumps({
            "event": "closure_file", "index": index, "total": len(rows),
            "file": record["file"],
            "prefix_frames": record["prefix_frames"],
            "residual_frames": int(record["residual"].shape[0]),
        }), flush=True)
    payload = {
        "format": FORMAT,
        "records": records,
        "source_contract": {
            "manifest_sha256": args.train_manifest_sha256.lower(),
            "records": len(records),
            "files": [row["file"] for row in rows],
            "prefix_rule": "frames [0,val_boundary) only",
            "validation_or_test_labels_read": False,
            "known_invalid_trajectory_excluded": True,
        },
        "operator": {
            "advection_scale": 1.0,
            "viscosity": 1.0e-6,
            "projection_strength": 0.0,
            "normalization": "physical residual divided by Re*1e-5",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.save(payload, args.output)
    print(json.dumps({
        "event": "closure_complete",
        "records": len(records),
        "residual_frames": sum(int(r["residual"].shape[0]) for r in records),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "elapsed_seconds": time.monotonic() - started,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
