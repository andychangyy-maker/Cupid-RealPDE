#!/usr/bin/env python3
"""Plot the comparable Real curves for structural HybridFNO2d experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "flat joint": "hybrid_fno2d_wide64_joint_tke4_sps_real6500",
    "shared 3D": "hybrid_fno2d_temporal16x3_joint_tke4_sps_real6500",
    "distill 0.5": "hybrid_fno2d_wide64_joint_tke4_sps_distill050_real6500",
    "energy curriculum": "hybrid_fno2d_energy_gain_joint_tke4_sps_real6500",
}
COLORS = {
    "flat joint": "#1f77b4",
    "shared 3D": "#ff7f0e",
    "distill 0.5": "#2ca02c",
    "energy curriculum": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("runs/logs"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/figures/hybrid_fno2d_advanced_experiments.png"))
    return parser.parse_args()


def values(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    rows = [row for row in history if key in row]
    return ([row["iteration"] for row in rows], [row[key] for row in rows])


def main() -> None:
    args = parse_args()
    summaries = {
        label: json.loads((args.logs / name / "summary.json").read_text())
        for label, name in RUNS.items()
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for label, summary in summaries.items():
        color = COLORS[label]
        history = summary["history"]
        x, y = values(history, "point_score")
        axes[0, 0].plot(x, y, label=label, color=color)
        x, y = values(history, "tke_rel_l2_score")
        axes[0, 1].plot(x, y, label=label, color=color)
        x, y = values(history, "train_tke_rel_l2_mean")
        axes[1, 0].plot(x, y, label=label, color=color)

    axes[0, 0].axhline(
        92.288038, color="black", linestyle=":", label="released FNO + TKE4")
    axes[0, 0].set(
        title="Validation point score (0.5 / 0.3 / 0.2)",
        xlabel="Real update", ylabel="score")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].axhline(
        82.659216, color="black", linestyle=":", label="released FNO + TKE4")
    axes[0, 1].set(
        title="Validation TKE score", xlabel="Real update", ylabel="score")
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[1, 0].set(
        title="Training TKE error after TKE4 switch",
        xlabel="Real update", ylabel="relative error")
    axes[1, 0].legend(frameon=False, fontsize=9)

    labels = list(summaries)
    exact_sps = [
        summaries[label]["metrics"]["real_best"]["native_sps_score"]
        for label in labels
    ]
    bars = axes[1, 1].bar(
        labels, exact_sps, color=[COLORS[label] for label in labels])
    axes[1, 1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1, 1].set(
        title="Uncalibrated exact SPS at point-best checkpoint",
        ylabel="exact SPS", ylim=(45.5, 48.3))
    axes[1, 1].tick_params(axis="x", rotation=15)

    figure.suptitle(
        "HybridFNO2d structural experiments on the common 725-window Real split",
        fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
