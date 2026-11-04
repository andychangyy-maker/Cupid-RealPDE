#!/usr/bin/env python3
"""Plot the v9 point, robust-FNO, and conditioned-interval experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_curve(relative: str) -> list[dict[str, float]]:
    with (REPO_ROOT / relative).open(newline="") as handle:
        return [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(handle)
        ]


def series(relative: str, key: str) -> tuple[list[float], list[float]]:
    rows = read_curve(relative)
    return ([row["iteration"] for row in rows], [row[key] for row in rows])


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

    correctors = (
        ("mean objective", "runs/logs/conditioned_corrector_mean_seed0/learning_curve.csv"),
        ("group + spectrum", "runs/logs/conditioned_corrector_group_spec_seed0/learning_curve.csv"),
        ("high-Re holdout", "runs/logs/conditioned_corrector_ood_highre_seed0/learning_curve.csv"),
    )
    for label, path in correctors:
        x, y = series(path, "val_accuracy_score_mean")
        baseline = y[0]
        axes[0].plot(x, [value - baseline for value in y], marker="o", ms=3, label=label)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(title="Residual corrector", xlabel="updates",
                ylabel="validation accuracy gain")
    axes[0].legend(fontsize=8)

    fno_runs = (
        ("TKE4", "runs/logs/fno_tke_weight4_1000_lr1e5_seed0/learning_curve.csv"),
        ("+ group robust", "runs/logs/fno_tke_group050_1000_seed0/learning_curve.csv"),
        ("+ group + FFT", "runs/logs/fno_tke_group050_spec002_1000_seed0/learning_curve.csv"),
    )
    for label, path in fno_runs:
        axes[1].plot(*series(path, "accuracy_score_mean"), marker="o", ms=3,
                     label=label)
    axes[1].set(title="Full-FNO objectives", xlabel="updates",
                ylabel="validation accuracy")
    axes[1].legend(fontsize=8)

    interval_runs = (
        ("clean deep", "runs/logs/structured_interval_deep_asym_reg_5000_seed1/learning_curve.csv"),
        ("condition only", "runs/logs/conditioned_interval_condonly_mean_4000_seed1/learning_curve.csv"),
        ("condition + group", "runs/logs/conditioned_interval_condonly_group025_4000_seed1/learning_curve.csv"),
        ("point correction + interval", "runs/logs/conditioned_interval_full_group025_4000_seed1/learning_curve.csv"),
        ("+ causal continuity", "runs/logs/conditioned_interval_continuity_w050_group025_4000_seed1/learning_curve.csv"),
    )
    for label, path in interval_runs:
        axes[2].plot(*series(path, "sps_score"), marker="o", ms=3, label=label)
    axes[2].set(title="Structured interval heads", xlabel="updates",
                ylabel="uncalibrated local SPS")
    axes[2].legend(fontsize=8)

    figure.suptitle("Track 1 v9 OOD-oriented exploration (released-data validation)")
    figure.tight_layout()
    output = REPO_ROOT / "docs/figures/track1_v9_ood_exploration.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
