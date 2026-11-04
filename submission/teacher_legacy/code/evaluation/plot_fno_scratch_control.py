#!/usr/bin/env python3
"""Plot the complete FNO random-init control against released checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "runs/figures/fno_scratch_control.png",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty curve: {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def history_arrays(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = summary["history"]
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def style(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel("Optimizer step")


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    sim = history_arrays(read_json(
        root / "runs/logs/fno_sim_from_scratch_5000_seed0/summary.json"))
    scratch = read_csv(
        root / "runs/logs/fno_scratch_sim_real_finetune_5000_seed0/learning_curve.csv")
    released_start = read_csv(
        root / "runs/logs/finetune_repro_fno_finalstart_seed0/learning_curve.csv")
    scratch_summary = read_json(
        root / "runs/logs/fno_scratch_sim_real_finetune_5000_seed0/summary.json")
    old_sps = history_arrays(read_json(
        root / "runs/logs/structured_interval_deep_asym_reg_5000_seed1/summary.json"))
    scratch_sps = history_arrays(read_json(
        root / "runs/logs/structured_interval_scratch_final_deep_asym_reg_5000_seed1/summary.json"))

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    axis = axes[0]
    axis.plot(sim["iteration"], sim["train_mse"], label="interval mean")
    axis.plot(sim["iteration"], sim["train_mse_median"], label="interval median")
    axis.set_yscale("log")
    axis.set_title("1. Sim pretraining from random init")
    axis.set_ylabel("Normalized train MSE")
    axis.legend(frameon=False, fontsize=8)
    style(axis)

    axis = axes[1]
    axis.plot(
        released_start["iteration"], released_start["rmse"],
        label="released sim start", color="#2563EB")
    axis.plot(
        scratch["iteration"], scratch["rmse"],
        label="scratch sim start", color="#F97316")
    released_rmse = scratch_summary["metrics"]["released_official_finetune"]["rmse"]
    axis.axhline(
        released_rmse, color="#111827", linestyle="--", linewidth=1,
        label="released real-finetuned")
    axis.set_title("2. Real fine-tuning")
    axis.set_ylabel("Validation RMSE (lower is better)")
    axis.legend(frameon=False, fontsize=8)
    style(axis)

    axis = axes[2]
    axis.plot(
        released_start["iteration"], released_start["tke_rel_l2"],
        label="released sim start", color="#2563EB")
    axis.plot(
        scratch["iteration"], scratch["tke_rel_l2"],
        label="scratch sim start", color="#F97316")
    released_tke = scratch_summary["metrics"]["released_official_finetune"]["tke_rel_l2"]
    axis.axhline(
        released_tke, color="#111827", linestyle="--", linewidth=1,
        label="released real-finetuned")
    axis.set_title("2. Real fine-tuning: TKE")
    axis.set_ylabel("TKE relative L2 (lower is better)")
    axis.legend(frameon=False, fontsize=8)
    style(axis)

    axis = axes[3]
    axis.plot(
        old_sps["iteration"], old_sps["sps_score"],
        label="current TKE4 point model", color="#2563EB")
    axis.plot(
        scratch_sps["iteration"], scratch_sps["sps_score"],
        label="scratch-chain point model", color="#F97316")
    axis.set_title("3. Deep asymmetric SPS head")
    axis.set_ylabel("Validation SPS (higher is better)")
    axis.legend(frameon=False, fontsize=8)
    style(axis)

    figure.suptitle(
        "FNO random-init control: simulation pretraining, real fine-tuning, and SPS",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
