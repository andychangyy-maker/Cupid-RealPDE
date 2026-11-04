#!/usr/bin/env python3
"""Plot the three fine-tuning runs and the FNO score discrepancy.

The script only reads artifacts produced by ``reproduce_finetune.py`` and
``score_finetune_checkpoints.py``.  Figures are written under the git-ignored
``runs/figures`` directory by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


WEBSITE_FNO_SCORES = {
    "rel_l2_score": 94.151948,
    "tke_score": 73.819158,
    "mvpe_score": 92.627465,
    "time_score": 93.971720,
    "sps_score": 53.170530,
    "final_score": 81.548164,
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    directory: str
    color: str


RUNS = (
    RunSpec("cno", "CNO", "finetune_repro_cno_finalstart_seed0", "#3B82F6"),
    RunSpec("fno", "FNO", "finetune_repro_fno_finalstart_seed0", "#F97316"),
    RunSpec(
        "transolver",
        "Transolver",
        "finetune_repro_transolver_finalstart_seed0_b4_tmux",
        "#10B981",
    ),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_curve(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty learning curve: {path}")
    return {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in rows[0]
    }


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def draw_run(
    axes: list[plt.Axes] | np.ndarray,
    spec: RunSpec,
    curve: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    step = curve["iteration"]
    best_step = float(summary["best_iteration"])
    official = summary["metrics"]["released_official_finetune"]

    axis = axes[0]
    axis.plot(step, curve["train_mse_mean"], color=spec.color, label="Train MSE")
    axis.plot(
        step,
        curve["normalized_mse"],
        color="#6B7280",
        linestyle="--",
        label="Validation MSE",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Normalized MSE")
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    axis = axes[1]
    axis.plot(step, curve["rmse"], color=spec.color, label="Local run")
    axis.axhline(
        official["rmse"],
        color="#111827",
        linestyle="--",
        linewidth=1.1,
        label="Released baseline",
    )
    best_index = int(np.argmin(curve["rmse"]))
    axis.scatter(
        step[best_index],
        curve["rmse"][best_index],
        color="#DC2626",
        marker="*",
        s=75,
        zorder=4,
        label=f"Best @{int(best_step)}",
    )
    axis.set_ylabel("Validation RMSE")
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    axis = axes[2]
    axis.plot(step, curve["r2"], color=spec.color)
    axis.axhline(
        official["r2"], color="#111827", linestyle="--", linewidth=1.1
    )
    axis.set_ylabel("Validation R-squared")
    style_axis(axis)

    axis = axes[3]
    axis.plot(step, curve["rel_l2"], label="Rel-L2", color="#2563EB")
    axis.plot(step, curve["tke_rel_l2"], label="TKE", color="#DC2626")
    axis.plot(step, curve["mvpe_rel_l2"], label="MVPE", color="#059669")
    axis.set_yscale("log")
    axis.set_ylabel("Raw relative error")
    axis.legend(frameon=False, fontsize=8, ncol=3)
    style_axis(axis)

    for axis in axes:
        axis.axvline(best_step, color="#DC2626", alpha=0.22, linewidth=0.8)
        axis.set_xlabel("Optimizer step")


def save_learning_curves(
    repo_root: Path, output_dir: Path, dpi: int
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    loaded: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}

    for spec in RUNS:
        run_dir = repo_root / "runs" / "logs" / spec.directory
        curve = read_curve(run_dir / "learning_curve.csv")
        summary = read_json(run_dir / "summary.json")
        loaded[spec.key] = (curve, summary)

        figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
        draw_run(axes, spec, curve, summary)
        figure.suptitle(
            f"{spec.label} real-data fine-tuning from released sim_pretrain final "
            f"(batch={summary['batch_size']}, lr={summary['learning_rate']:g})",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(output_dir / f"training_curves_{spec.key}.png", dpi=dpi)
        plt.close(figure)

    figure, axes = plt.subplots(3, 4, figsize=(16, 12.5), sharex="col")
    for row, spec in enumerate(RUNS):
        curve, summary = loaded[spec.key]
        draw_run(axes[row], spec, curve, summary)
        axes[row, 0].text(
            0.97,
            0.95,
            spec.label,
            transform=axes[row, 0].transAxes,
            va="top",
            ha="right",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    for axis, title in zip(
        axes[0], ("Optimization loss", "Forecast RMSE", "Forecast R-squared", "Physics errors")
    ):
        axis.set_title(title)
    figure.suptitle(
        "RealPDE fine-tuning learning curves (same 82-trajectory temporal-tail split)",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(output_dir / "finetune_learning_curves.png", dpi=dpi)
    plt.close(figure)
    return loaded


def model_scores(path: Path) -> dict[str, dict[str, float]]:
    payload = read_json(path)
    return {
        name: values["subscores"] for name, values in payload["models"].items()
    }


def save_score_comparison(repo_root: Path, output_dir: Path, dpi: int) -> None:
    score_by_model: dict[str, dict[str, dict[str, float]]] = {}
    for spec in RUNS:
        path = (
            repo_root
            / "runs"
            / "logs"
            / spec.directory
            / "official_scores_local_val.json"
        )
        score_by_model[spec.key] = model_scores(path)

    fno_fp16_path = (
        repo_root
        / "runs/logs/finetune_repro_fno_finalstart_seed0/official_scores_fp16_local_val.json"
    )
    fno_fp16 = model_scores(fno_fp16_path)["released_official_finetune_fp16"]
    fno_single = read_json(
        repo_root
        / "runs/logs/eval/track1_fno_train3750_0/official_scores/scores.json"
    )

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))

    axis = axes[0]
    x = np.arange(len(RUNS), dtype=np.float64)
    width = 0.23
    categories = (
        ("released_sim_pretrain", "Pretrain start", "#94A3B8"),
        ("local_final", "Our final", "#3B82F6"),
        ("released_official_finetune", "Released baseline", "#10B981"),
    )
    for offset, (key, label, color) in zip((-width, 0.0, width), categories):
        values = [score_by_model[spec.key][key]["final_score"] for spec in RUNS]
        bars = axis.bar(x + offset, values, width=width, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axis.scatter(
        x[1] + width,
        WEBSITE_FNO_SCORES["final_score"],
        marker="D",
        s=70,
        facecolor="#DC2626",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="FNO CodaBench hidden",
    )
    axis.annotate(
        f"{WEBSITE_FNO_SCORES['final_score']:.2f}",
        (x[1] + width, WEBSITE_FNO_SCORES["final_score"]),
        xytext=(7, -15),
        textcoords="offset points",
        fontsize=8,
        color="#991B1B",
    )
    axis.set_xticks(x, [spec.label for spec in RUNS])
    axis.set_ylim(65, 87)
    axis.set_ylabel("Official-formula Final score")
    axis.set_title("Same formula, different checkpoints / evaluation data")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(axis)

    axis = axes[1]
    metric_keys = (
        "rel_l2_score",
        "tke_score",
        "mvpe_score",
        "time_score",
        "sps_score",
        "final_score",
    )
    metric_labels = ("Rel-L2", "TKE", "MVPE", "Time", "SPS", "Final")
    comparisons = (
        ("One released train trajectory", fno_single, "#94A3B8", "o", "--"),
        (
            "82-trajectory tail, FP32",
            score_by_model["fno"]["released_official_finetune"],
            "#2563EB",
            "o",
            "-",
        ),
        ("82-trajectory tail, FP16", fno_fp16, "#059669", "s", "-"),
        ("CodaBench hidden, FP16", WEBSITE_FNO_SCORES, "#DC2626", "D", "-"),
    )
    metric_x = np.arange(len(metric_keys))
    for label, scores, color, marker, linestyle in comparisons:
        axis.plot(
            metric_x,
            [scores[key] for key in metric_keys],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=5,
        )
    axis.set_xticks(metric_x, metric_labels)
    axis.set_ylim(48, 100)
    axis.set_ylabel("Subscore")
    axis.set_title("Official FNO: local data vs CodaBench hidden data")
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    style_axis(axis)

    figure.suptitle(
        "RealPDE baseline score comparison",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_dir / "fno_score_discrepancy.png", dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "runs" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    save_learning_curves(repo_root, output_dir, args.dpi)
    save_score_comparison(repo_root, output_dir, args.dpi)
    for name in (
        "training_curves_cno.png",
        "training_curves_fno.png",
        "training_curves_transolver.png",
        "finetune_learning_curves.png",
        "fno_score_discrepancy.png",
    ):
        print(output_dir / name)


if __name__ == "__main__":
    main()
