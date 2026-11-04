#!/usr/bin/env python3
"""Train B210 with disposable Re/AoA representation supervision.

The released Sim trajectories form a complete Reynolds-number/angle-of-attack
grid.  This launcher reuses the audited B210 training loop, but replaces its
optional training-only multiscale TKE probes with multiscale regime probes.
The probes force all three decoder scales to retain enough global context to
recover the generating condition from history.  They are not part of the
saved point model and therefore add no deployment parameters or latency.
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as b210  # noqa: E402


METHOD = "b210_realclosure_multiscale_regime_parameter_supervision_v1"
HIDDEN_CHANNELS = 32
LOSS_WEIGHT = 0.005
GRADIENT_RATIO_RANGE = (0.002, 0.25)

_REGIME_TARGETS: torch.Tensor | None = None


def load_regime_targets(sim_dir: Path) -> torch.Tensor:
    """Return normalized [Re, AoA] labels in loader trajectory order."""
    files = sorted(sim_dir.glob("*.h5"))
    if len(files) != 100:
        raise RuntimeError(
            f"regime supervision requires 100 released Sim files, got {len(files)}")
    rows: list[tuple[float, float]] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            rows.append((
                float(handle["re"][()]),
                float(handle["aoa"][()]),
            ))
    values = torch.tensor(rows, dtype=torch.float32)
    re_values = torch.unique(values[:, 0])
    aoa_values = torch.unique(values[:, 1])
    if re_values.numel() != 20 or aoa_values.numel() != 5:
        raise RuntimeError(
            "released Sim regime grid is not the expected 20 Re x 5 AoA")
    if len({tuple(row) for row in rows}) != len(rows):
        raise RuntimeError("released Sim regime grid contains duplicates")
    lower = values.amin(dim=0)
    upper = values.amax(dim=0)
    if bool(torch.any(upper <= lower)):
        raise RuntimeError("released Sim regime range is degenerate")
    return 2.0 * (values - lower) / (upper - lower) - 1.0


class RegimeParameterSupervisor(nn.Module):
    """Disposable multiscale probes for normalized Re and AoA."""

    def __init__(self, channels):
        super().__init__()
        if _REGIME_TARGETS is None:
            raise RuntimeError("regime targets were not initialized")
        if len(channels) != 3:
            raise ValueError("regime supervision expects three decoder scales")
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(int(width), HIDDEN_CHANNELS, 1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(HIDDEN_CHANNELS, 2),
            )
            for width in channels)
        self.register_buffer(
            "regime_targets", _REGIME_TARGETS.clone(), persistent=True)
        self._trajectory_ids: torch.Tensor | None = None

    def set_trajectory_ids(self, trajectory_ids: torch.Tensor) -> None:
        ids = trajectory_ids.detach().to(dtype=torch.long, device="cpu")
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("trajectory ids must be one non-empty vector")
        if int(ids.min()) < 0 or int(ids.max()) >= self.regime_targets.shape[0]:
            raise ValueError("trajectory id lies outside the regime table")
        self._trajectory_ids = ids

    def forward(self, features, _target_physical_uv):
        if self._trajectory_ids is None:
            raise RuntimeError("regime supervisor has no current trajectory ids")
        if len(features) != len(self.heads):
            raise ValueError("regime feature/head contract changed")
        target = self.regime_targets[
            self._trajectory_ids.to(self.regime_targets.device)]
        losses = [
            F.smooth_l1_loss(head(feature).float(), target)
            for feature, head in zip(features, self.heads)
        ]
        weights = losses[0].new_tensor((0.25, 0.50, 1.00))
        total = torch.sum(torch.stack(losses) * weights) / weights.sum()
        return total, losses


_ORIGINAL_PARSE_ARGS = b210.parse_args
_ORIGINAL_SIM_UPDATE = b210.sim_update


def parse_args():
    args = _ORIGINAL_PARSE_ARGS()
    args.multiscale_tke_deep_supervision = True
    return args


def regime_sim_update(*args, **kwargs):
    # Positional contract: model, batch, ..., closure equivariance at 13 and
    # the optional training-only supervisor at index 14.
    batch = args[1]
    supervisor = kwargs.get("multiscale_tke_supervisor")
    if supervisor is None and len(args) > 14:
        supervisor = args[14]
    if isinstance(supervisor, RegimeParameterSupervisor):
        supervisor.set_trajectory_ids(batch[2])
    return _ORIGINAL_SIM_UPDATE(*args, **kwargs)


def main() -> None:
    global _REGIME_TARGETS
    # The base parser owns the public CLI.  Read only the value needed before
    # the base main constructs the disposable supervisor.
    sim_dir = Path("data/sim")
    for index, token in enumerate(sys.argv[:-1]):
        if token == "--sim-dir":
            sim_dir = Path(sys.argv[index + 1])
            break
    _REGIME_TARGETS = load_regime_targets(sim_dir)

    b210.parse_args = parse_args
    b210.MultiScaleTKEDeepSupervisor = RegimeParameterSupervisor
    b210.sim_update = regime_sim_update
    b210.MULTISCALE_TKE_METHOD = METHOD
    b210.MULTISCALE_TKE_WEIGHT = LOSS_WEIGHT
    b210.MULTISCALE_TKE_GRADIENT_RATIO_RANGE = GRADIENT_RATIO_RANGE
    b210.REAL_CLOSURE_SEED1_METHOD = METHOD
    b210.main()


if __name__ == "__main__":
    main()
