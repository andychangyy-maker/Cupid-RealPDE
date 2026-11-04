"""Read and visualize one HDF5 flow-field file without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def describe_h5(path: Path) -> dict[str, dict[str, object]]:
    """Return the top-level dataset shapes and dtypes in an HDF5 file."""
    with h5py.File(path, "r") as handle:
        return {
            name: {"shape": tuple(dataset.shape), "dtype": str(dataset.dtype)}
            for name, dataset in handle.items()
            if isinstance(dataset, h5py.Dataset)
        }


def render_frame(path: Path, field: str, frame: int, output: Path) -> Path:
    """Render one time frame of a 3-D field (time, height, width) to PNG."""
    with h5py.File(path, "r") as handle:
        if field not in handle:
            available = ", ".join(sorted(handle.keys()))
            raise KeyError(f"Field '{field}' is absent. Available datasets: {available}")
        dataset = handle[field]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
            raise ValueError(
                f"Field '{field}' must have shape (time, height, width); got {getattr(dataset, 'shape', None)}"
            )
        if frame < 0 or frame >= dataset.shape[0]:
            raise IndexError(f"Frame {frame} is outside valid range 0..{dataset.shape[0] - 1}")
        image = dataset[frame]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    heatmap = axis.imshow(image, origin="lower", aspect="auto", cmap="viridis")
    axis.set_title(f"{path.name} | {field}[{frame}]")
    axis.set_xlabel("x grid index")
    axis.set_ylabel("y grid index")
    figure.colorbar(heatmap, ax=axis, label=field)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def render_real_sim_triptych(
    real_path: Path, sim_path: Path, field: str, output: Path, scale_mode: str = "shared"
) -> Path:
    """Compare start/middle/end frames for paired data using shared or per-source scaling."""
    if scale_mode not in {"shared", "per_source"}:
        raise ValueError("scale_mode must be 'shared' or 'per_source'")
    selected_frames: list[list[np.ndarray]] = []
    labels: list[tuple[str, str]] = []
    for source_path, row_label in ((Path(real_path), "Real PIV"), (Path(sim_path), "Sim CFD")):
        with h5py.File(source_path, "r") as handle:
            if field not in handle:
                available = ", ".join(sorted(handle.keys()))
                raise KeyError(f"Field '{field}' is absent in {source_path.name}. Available datasets: {available}")
            dataset = handle[field]
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
                raise ValueError(f"Field '{field}' in {source_path.name} must have shape (time, height, width)")
            indices = (0, dataset.shape[0] // 2, dataset.shape[0] - 1)
            selected_frames.append([np.asarray(dataset[index]) for index in indices])
            labels.append((row_label, source_path.name))

    all_values = np.concatenate([frame.ravel() for row in selected_frames for frame in row])
    shared_scale = float(np.nanpercentile(np.abs(all_values), 99))
    if not np.isfinite(shared_scale) or shared_scale == 0:
        shared_scale = 1.0
    row_scales = []
    for row in selected_frames:
        scale = float(np.nanpercentile(np.abs(np.concatenate([frame.ravel() for frame in row])), 99))
        row_scales.append(scale if np.isfinite(scale) and scale > 0 else 1.0)
    scales = [shared_scale, shared_scale] if scale_mode == "shared" else row_scales

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    row_images = []
    for row_index, (row, (row_label, filename)) in enumerate(zip(selected_frames, labels)):
        row_image = None
        for column_index, image in enumerate(row):
            axis = axes[row_index, column_index]
            row_image = axis.imshow(
                image, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-scales[row_index], vmax=scales[row_index]
            )
            index_label = (0, "middle", "last")[column_index]
            axis.set_title(f"{row_label} · {index_label}")
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
        axes[row_index, 0].text(
            -0.28,
            0.5,
            f"{row_label}\n{filename}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        row_images.append(row_image)
    scale_label = "shared raw scale" if scale_mode == "shared" else "per-source contrast scale"
    figure.suptitle(
        f"NACA4418 paired trajectory comparison · field {field} · {scale_label}",
        fontsize=15,
        fontweight="bold",
    )
    for row_index, row_image in enumerate(row_images):
        if row_image is not None:
            figure.colorbar(row_image, ax=axes[row_index, :].tolist(), shrink=0.82, label=field)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True, help="Input .h5 file")
    parser.add_argument("--field", default="u", help="3-D field to render; commonly u, v, or p")
    parser.add_argument("--frame", type=int, default=0, help="Time index to render")
    parser.add_argument("--output", type=Path, help="Output PNG path; omit to only print metadata")
    args = parser.parse_args()

    metadata = describe_h5(args.file)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=list))
    if args.output:
        rendered = render_frame(args.file, args.field, args.frame, args.output)
        print(f"Wrote {rendered}")


if __name__ == "__main__":
    main()
