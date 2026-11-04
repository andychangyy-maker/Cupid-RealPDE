#!/usr/bin/env python3
"""Plot structured interval-head learning curves and robust SPS comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = (
    ("Pointwise asymmetric", "structured_interval_pointwise_asym_reg_5000_seed0"),
    ("Spatial symmetric", "structured_interval_spatial_sym_reg_5000_seed0"),
    ("Spatial asymmetric", "structured_interval_spatial_asym_reg_5000_seed0"),
    ("Spatial asymmetric, weak NLL", "structured_interval_spatial_asym_soft_5000_seed0"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("runs/logs"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/figures/structured_interval_ablation.png"))
    args = parser.parse_args()

    rows = []
    for label, directory in EXPERIMENTS:
        path = args.logs_dir / directory / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append((label, payload))

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    colors = plt.get_cmap("tab10").colors
    for index, (label, payload) in enumerate(rows):
        iterations = [entry["iteration"] for entry in payload["history"]]
        scores = [entry["sps_score"] for entry in payload["history"]]
        axes[0].plot(
            iterations, scores, marker="o", markersize=3.5,
            linewidth=1.8, label=label, color=colors[index])
    axes[0].axhline(
        51.721064, color="0.35", linestyle="--", linewidth=1.2,
        label="Dynamic v1 (pointwise symmetric)")
    axes[0].set_xlabel("Training update")
    axes[0].set_ylabel("Exact local SPS")
    axes[0].set_title("Temporal-tail validation")
    axes[0].legend(fontsize=8)

    positions = np.arange(len(rows))
    robust_local = [
        payload["robust_multiplier_on_local_metrics"]["sps_score"]
        for _label, payload in rows
    ]
    robust_stress = [
        payload["robust_stress_metrics"]["sps_score"]
        for _label, payload in rows
    ]
    width = 0.36
    axes[1].bar(
        positions - width / 2, robust_local, width,
        label="Robust multiplier on local", color="#4c78a8")
    axes[1].bar(
        positions + width / 2, robust_stress, width,
        label="1.65x residual stress", color="#f58518")
    axes[1].axhline(
        49.233081, color="#4c78a8", linestyle="--", linewidth=1.0)
    axes[1].axhline(
        44.280397, color="#f58518", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        ["Point/asym", "Spatial/sym", "Spatial/asym", "Spatial/asym\nweak NLL"],
        fontsize=8)
    axes[1].set_ylabel("Exact SPS")
    axes[1].set_title("Robust calibration and stress test")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(40, 54)

    figure.suptitle("Structured SPS interval-head ablation", fontsize=13)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
