#!/usr/bin/env python3
"""Plot architecture screening and joint TKE/SPS training curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES = {
    "high-mode 48x4": "hybrid_fno2d_hf48_m14x24_l4_k3",
    "wide 64x4": "hybrid_fno2d_wide64_m12x20_l4_k3",
    "deep 48x6 k5": "hybrid_fno2d_deep48_m12x20_l6_k5",
}
JOINT = "hybrid_fno2d_wide64_joint_tke4_sps"
TEMPORAL = "hybrid_fno2d_temporal16x3_joint_tke4_sps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir", type=Path, default=Path("runs/logs"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/figures/hybrid_fno2d_experiments.png"))
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def values(history: list[dict], key: str):
    selected = [(row["iteration"], row[key]) for row in history if key in row]
    return [item[0] for item in selected], [item[1] for item in selected]


def main() -> None:
    args = parse_args()
    logs = (REPO_ROOT / args.logs_dir).resolve()
    output = (REPO_ROOT / args.output).resolve()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for label, directory in BASELINES.items():
        sim = load(logs / f"{directory}_sim5000" / "summary.json")
        real = load(logs / f"{directory}_real5000" / "summary.json")
        x, y = values(sim["history"], "train_mse_median")
        axes[0, 0].plot(x, y, label=label)
        x, y = values(real["history"], "accuracy_score_mean")
        axes[0, 1].plot(x, y, label=label)

    axes[0, 0].set(title="MSE-only architecture screening: Sim",
                   xlabel="update", ylabel="median normalized MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].axhline(90.594003, color="0.35", linestyle="--",
                       label="released real FNO")
    axes[0, 1].axhline(91.973177, color="black", linestyle=":",
                       label="released FNO + TKE4")
    axes[0, 1].set(title="MSE-only architecture screening: Real",
                   xlabel="update", ylabel="equal mean of point subscores")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    sim_joint = load(logs / f"{JOINT}_sim10000" / "summary.json")
    real_joint = load(logs / f"{JOINT}_real6500" / "summary.json")
    point_curves = (
            ("train_rel_l2_mean", "Rel-L2 error"),
            ("train_tke_rel_l2_mean", "TKE error"),
            ("train_mvpe_rel_l2_mean", "MVPE error"))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for (key, label), color in zip(point_curves, colors):
        x, y = values(sim_joint["history"], key)
        axes[1, 0].plot(x, y, color=color, label=f"flat {label}")
    temporal_sim_path = logs / f"{TEMPORAL}_sim10000" / "summary.json"
    if temporal_sim_path.exists():
        temporal_sim = load(temporal_sim_path)
        for (key, label), color in zip(point_curves, colors):
            x, y = values(temporal_sim["history"], key)
            axes[1, 0].plot(
                x, y, color=color, linestyle="--", label=f"3D {label}")
    axes[1, 0].set(title="Joint pretraining: physical point objectives",
                   xlabel="Sim update", ylabel="relative error")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.2)

    x, width = values(real_joint["history"], "train_interval_mean_width_mean")
    left = axes[1, 1]
    left.plot(x, width, color="#1f77b4", label="flat mean total width")
    left.set(title="Native interval head during Real adaptation",
             xlabel="Real update", ylabel="physical mean total width")
    right = left.twinx()
    x, soft = values(real_joint["history"], "train_interval_soft_sps_mean")
    right.plot(x, [100.0 * value for value in soft], color="#d62728",
               label="flat soft-SPS")
    temporal_real_path = logs / f"{TEMPORAL}_real6500" / "summary.json"
    if temporal_real_path.exists():
        temporal_real = load(temporal_real_path)
        x, width = values(
            temporal_real["history"], "train_interval_mean_width_mean")
        left.plot(x, width, color="#1f77b4", linestyle="--",
                  label="3D mean total width")
        x, soft = values(
            temporal_real["history"], "train_interval_soft_sps_mean")
        right.plot(x, [100.0 * value for value in soft], color="#d62728",
                   linestyle="--", label="3D soft-SPS")
    right.set_ylabel("soft-SPS x 100")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], frameon=False)
    left.grid(alpha=0.2)

    figure.suptitle("HybridFNO2d: architecture and end-to-end TKE/SPS training")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
