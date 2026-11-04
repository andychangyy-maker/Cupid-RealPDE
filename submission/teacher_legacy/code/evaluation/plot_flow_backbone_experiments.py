#!/usr/bin/env python3
"""Plot matched-split U-Net/DiT learning curves and local score comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unet-screen", type=Path, required=True)
    parser.add_argument("--dit-screen", type=Path, required=True)
    parser.add_argument("--unet-full", type=Path, required=True)
    parser.add_argument("--unet-wide", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def best(summary: dict) -> dict:
    return summary["metrics"]["real_best"]


def main() -> None:
    args = parse_args()
    unet_screen = load(args.unet_screen)
    dit_screen = load(args.dit_screen)
    unet_full = load(args.unet_full)
    unet_wide = load(args.unet_wide)
    hybrid = load(args.hybrid)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    colors = {
        "U-Net short": "#2563eb",
        "DiT short": "#d97706",
        "U-Net64 full": "#059669",
        "U-Net96 full": "#7c3aed",
    }
    for label, summary in (
            ("U-Net short", unet_screen),
            ("DiT short", dit_screen),
            ("U-Net64 full", unet_full),
            ("U-Net96 full", unet_wide)):
        updates = [row["iteration"] for row in summary["history"]]
        scores = [row["point_score"] for row in summary["history"]]
        axes[0].plot(updates, scores, marker="o", markersize=3,
                     linewidth=2, label=label, color=colors[label])
    axes[0].set_title("Real temporal-tail point score")
    axes[0].set_xlabel("Real fine-tuning update")
    axes[0].set_ylabel("0.5 Rel + 0.3 TKE + 0.2 MVPE")
    axes[0].legend(frameon=True)

    labels = ["Hybrid FNO\ndistill", "U-Net64\nshort", "DiT\nshort",
              "U-Net64\nfull", "U-Net96\nfull"]
    summaries = [hybrid, unet_screen, dit_screen, unet_full, unet_wide]
    metric_keys = ["rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
                   "native_sps_score"]
    metric_labels = ["Rel", "TKE", "MVPE", "exact SPS"]
    x = np.arange(len(labels))
    width = 0.19
    for index, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [best(summary)[key] for summary in summaries]
        axes[1].bar(x + (index - 1.5) * width, values, width, label=label)
    axes[1].set_title("Matched 725-window local validation")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("score")
    axes[1].set_ylim(35, 100)
    axes[1].legend(ncols=2, frameon=True)

    figure.suptitle("RealPDE Track 1: architecture switch experiments")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
