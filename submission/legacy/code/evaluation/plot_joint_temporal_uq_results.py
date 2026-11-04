#!/usr/bin/env python3
"""Plot the controlled pretraining-level temporal-UQ experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-sim", type=Path,
        default=Path(
            "runs/logs/ensemble_unet_c160_crossprofile_jointuq_sim10000/"
            "summary.json"))
    parser.add_argument(
        "--aggressive-sim", type=Path,
        default=Path(
            "runs/logs/ensemble_unet_c160_temporal3d_utility_sps_sim10000/"
            "summary.json"))
    parser.add_argument(
        "--fixed-sim", type=Path,
        default=Path(
            "runs/logs/"
            "ensemble_unet_c160_temporal3d_utility_sps_sim10000_w005warm1500/"
            "summary.json"))
    parser.add_argument(
        "--old-real", type=Path,
        default=Path(
            "runs/logs/ensemble_unet_c160_crossprofile_jointuq_real5000/"
            "summary.json"))
    parser.add_argument(
        "--new-real", type=Path,
        default=Path(
            "runs/logs/"
            "ensemble_unet_c160_temporal3d_utility_sps_real5000_"
            "w005warm1500_uqonly/summary.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/joint_temporal_uq_curves.png"))
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def series(payload: dict, key: str) -> tuple[list[int], list[float]]:
    rows = [row for row in payload["history"] if key in row]
    return (
        [int(row["iteration"]) for row in rows],
        [float(row[key]) for row in rows],
    )


def plot_lines(axis, payloads, key: str, title: str, ylabel: str) -> None:
    for label, payload, style, color in payloads:
        x, y = series(payload, key)
        axis.plot(
            x, y, style, color=color, linewidth=2.0,
            markersize=4, label=label)
    axis.set_title(title)
    axis.set_xlabel("update")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def main() -> None:
    args = parse_args()
    old_sim = load(args.old_sim)
    aggressive_sim = load(args.aggressive_sim)
    fixed_sim = load(args.fixed_sim)
    old_real = load(args.old_real)
    new_real = load(args.new_real)

    sim_payloads = (
        ("old C160", old_sim, "o-", "#1f77b4"),
        ("temporal UQ, w=0.10", aggressive_sim, "x--", "#ff7f0e"),
        ("temporal UQ, fixed w=0.05", fixed_sim, "s-", "#2ca02c"),
    )
    real_payloads = (
        ("old C160", old_real, "o-", "#1f77b4"),
        ("pretraining-level temporal UQ", new_real, "s-", "#2ca02c"),
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    plot_lines(
        axes[0, 0], sim_payloads, "train_rel_l2_mean",
        "Sim relative error", "error (lower is better)")
    plot_lines(
        axes[0, 1], sim_payloads, "train_tke_rel_l2_mean",
        "Sim TKE error", "error (lower is better)")
    plot_lines(
        axes[0, 2], sim_payloads, "train_quantile_pinball_mean",
        "Sim residual-distribution loss", "normalized pinball")
    plot_lines(
        axes[1, 0], real_payloads, "point_score",
        "Real temporal-tail point score", "score (higher is better)")
    plot_lines(
        axes[1, 1], real_payloads, "native_sps_score",
        "Real temporal-tail native SPS", "score (higher is better)")
    plot_lines(
        axes[1, 2], real_payloads, "native_sps_coverage",
        "Real interval coverage", "coverage")
    axes[1, 2].axhline(0.86, color="#777777", linestyle=":", linewidth=1.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.suptitle(
        "RealPDE Track 1: pretraining-level spatiotemporal uncertainty",
        y=1.055, fontsize=15, fontweight="bold")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
