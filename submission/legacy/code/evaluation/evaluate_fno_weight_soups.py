#!/usr/bin/env python3
"""Evaluate single-FNO weight soups and causal post-processing by condition.

The script interpolates two released-data checkpoints in parameter space, so a
selected candidate has the same model size and inference cost as one FNO.  It
reports the ordinary temporal-tail score together with Reynolds/AoA slices to
avoid selecting exclusively on the optimistic same-condition aggregate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/uncertainty"))

from conditioned_residual_corrector import ConditionedResidualCorrector

_post_spec = importlib.util.spec_from_file_location(
    "realpde_causal_postprocess_soup",
    REPO_ROOT / "code/evaluation/explore_causal_postprocessing.py",
)
if _post_spec is None or _post_spec.loader is None:
    raise ImportError("Cannot import explore_causal_postprocessing.py")
_post = importlib.util.module_from_spec(_post_spec)
sys.modules[_post_spec.name] = _post
_post_spec.loader.exec_module(_post)
_repro = _post._repro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument(
        "--correction-checkpoint", type=Path,
        help="Optional frozen ConditionedResidualCorrector checkpoint.")
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument(
        "--correction-scales", type=float, nargs="+",
        help=("Optional correction-scale sweep evaluated after each FNO forward; "
              "overrides --correction-scale."))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--holdout-re", type=int, nargs="*", default=(),
        help="Evaluate all manifest windows from these Reynolds conditions.")
    parser.add_argument("--stats", type=Path, default=Path(
        "kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument("--kit-dir", type=Path, default=Path(
        "kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--raw-only", action="store_true",
        help="Skip continuity/fluctuation transforms for inexpensive sweeps.")
    return parser.parse_args()


def mixed_state(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor],
                alpha: float) -> dict[str, torch.Tensor]:
    if a.keys() != b.keys():
        raise ValueError("Checkpoint state dictionaries have different keys")
    result = {}
    for name in a:
        left = a[name]
        right = b[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Checkpoint tensor mismatch for {name}")
        if torch.is_floating_point(left) or torch.is_complex(left):
            result[name] = left + alpha * (right - left)
        else:
            result[name] = right if alpha >= 0.5 else left
    return result


def group_masks(windows, metadata: list[dict]) -> dict[str, np.ndarray]:
    re_values = np.asarray(
        [metadata[item.trajectory]["re"] for item in windows], dtype=np.int64)
    aoa_values = np.asarray(
        [metadata[item.trajectory]["aoa"] for item in windows], dtype=np.int64)
    masks = {"all": np.ones(len(windows), dtype=bool)}
    masks.update({
        "re_band_low": re_values <= 9000,
        "re_band_mid": (re_values > 9000) & (re_values < 17000),
        "re_band_high": re_values >= 17000,
    })
    for value in sorted(set(aoa_values.tolist())):
        masks[f"aoa_{value}"] = aoa_values == value
    return masks


def evaluation_windows(manifest_path: Path, filenames: list[str],
                       metadata: list[dict], holdout_re) -> list:
    if not holdout_re:
        return _post.validation_windows(manifest_path, filenames)
    values = {int(value) for value in holdout_re}
    available = {int(item["re"]) for item in metadata}
    missing = values - available
    if missing:
        raise ValueError(f"Unknown Reynolds holdout values: {sorted(missing)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_file = {entry["file"]: entry for entry in manifest}
    windows = []
    for trajectory, item in enumerate(metadata):
        if int(item["re"]) not in values:
            continue
        entry = by_file[item["file"]]
        starts = sorted(set(entry["train_starts"] + entry["val_starts"]))
        windows.extend(
            _repro.Window(trajectory, int(start)) for start in starts)
    if not windows:
        raise ValueError("Reynolds holdout produced no evaluation windows")
    return windows


def grouped_scores(prediction: np.ndarray, target: np.ndarray,
                   masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {"samples": int(mask.sum()), **_post.score_accuracy(
            prediction[mask], target[mask])}
        for name, mask in masks.items() if np.any(mask)
    }


def candidate_predictions(x: np.ndarray, prediction: np.ndarray,
                          raw_only: bool = False):
    yield "raw", prediction
    if raw_only:
        return
    for weight in (0.25, 0.5, 0.75):
        yield (
            f"continuity_w{weight:g}_p4",
            _post.continuity_correct(x, prediction, weight, 4.0),
        )
    for scale in (0.95, 1.05):
        scaled = _post.scale_predicted_fluctuations(prediction, scale)
        yield f"fluctuation_{scale:g}", scaled
        yield (
            f"fluctuation_{scale:g}_continuity_w0.5_p4",
            _post.continuity_correct(x, scaled, 0.5, 4.0),
        )


def apply_conditioned_correction(
        corrector: ConditionedResidualCorrector, x: np.ndarray,
        prediction: np.ndarray, scale: float, batch_size: int,
        device: torch.device) -> np.ndarray:
    corrected_parts = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            observation = torch.from_numpy(
                np.ascontiguousarray(x[start:start + batch_size])).to(device)
            point = torch.from_numpy(np.ascontiguousarray(
                prediction[start:start + batch_size, ..., :2])).to(device)
            corrected, _condition = corrector(point, observation)
            corrected_parts.append(
                (point + scale * (corrected - point)).cpu().numpy())
    result = prediction.copy()
    result[..., :2] = np.concatenate(corrected_parts)
    result[..., 2] = 0.0
    return result


def robust_summary(groups: dict[str, dict[str, float]]) -> dict[str, float]:
    condition_names = [name for name in groups if name != "all"]
    values = np.asarray(
        [groups[name]["accuracy_mean"] for name in condition_names],
        dtype=np.float64,
    )
    all_score = float(groups["all"]["accuracy_mean"])
    return {
        "all_accuracy_mean": all_score,
        "condition_accuracy_mean": float(values.mean()),
        "condition_accuracy_min": float(values.min()),
        "robust_selection_score": float(all_score + 0.25 * values.min()),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    correction_scales = (
        args.correction_scales
        if args.correction_scales is not None else [args.correction_scale])
    if any(not np.isfinite(value) or not 0.0 <= value <= 2.0
           for value in correction_scales):
        raise ValueError("correction scales must be finite and in [0, 2]")
    if not args.alphas or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in args.alphas):
        raise ValueError("alphas must be finite values in [0, 1]")
    device = torch.device(args.device)
    trajectories, metadata = _repro.load_trajectories(args.real_dir, 0)
    filenames = [item["file"] for item in metadata]
    windows = evaluation_windows(
        args.split_manifest, filenames, metadata, args.holdout_re)
    dataset = _repro.WindowDataset(trajectories, windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    normalizer = _repro.Normalizer(args.stats, device)
    baseline = _repro.load_baseline_module(args.kit_dir)

    model_a, meta_a = baseline.load_baseline(
        "fno", str(args.checkpoint_a), device="cpu")
    model_b, meta_b = baseline.load_baseline(
        "fno", str(args.checkpoint_b), device="cpu")
    state_a = {key: value.detach().clone() for key, value in model_a.state_dict().items()}
    state_b = {key: value.detach().clone() for key, value in model_b.state_dict().items()}
    del model_b

    corrector = None
    correction_metadata = None
    if args.correction_checkpoint is not None:
        correction_metadata = torch.load(
            args.correction_checkpoint, map_location=device, weights_only=False)
        corrector = ConditionedResidualCorrector(
            hidden=int(correction_metadata["hidden"]),
            context_hidden=int(correction_metadata["context_hidden"]),
            architecture=correction_metadata["architecture"],
        ).to(device)
        corrector.load_state_dict(correction_metadata["corrector_state_dict"])
        corrector.eval()

    masks = group_masks(windows, metadata)
    results = {}
    reference_input = reference_target = None
    started = time.time()
    for alpha in args.alphas:
        model_a.load_state_dict(mixed_state(state_a, state_b, float(alpha)))
        model_a.to(device)
        x, prediction, target = _post.predict_fno(
            model_a, loader, normalizer, device)
        model_a.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if reference_input is None:
            reference_input = x
            reference_target = target
        else:
            if not np.array_equal(x, reference_input):
                raise RuntimeError("Input order changed between soup evaluations")
            if not np.array_equal(target, reference_target):
                raise RuntimeError("Target order changed between soup evaluations")
            del x, target
        assert reference_input is not None and reference_target is not None
        scale_candidates = [(None, prediction)]
        if corrector is not None:
            scale_candidates = [
                (float(scale), apply_conditioned_correction(
                    corrector, reference_input, prediction, float(scale),
                    args.batch_size, device))
                for scale in correction_scales
            ]
        for correction_scale, scaled_prediction in scale_candidates:
            for transform, candidate in candidate_predictions(
                    reference_input, scaled_prediction, args.raw_only):
                scale_name = (
                    "" if correction_scale is None
                    else f"correction_{correction_scale:.3f}_")
                name = f"alpha_{float(alpha):.3f}_{scale_name}{transform}"
                groups = grouped_scores(candidate, reference_target, masks)
                results[name] = {
                    "alpha": float(alpha),
                    "correction_scale": correction_scale,
                    "transform": transform,
                    "groups": groups,
                    **robust_summary(groups),
                }
                print(json.dumps({"event": "score", "candidate": name,
                                  **robust_summary(groups)}), flush=True)
                if candidate is not scaled_prediction:
                    del candidate
            if scaled_prediction is not prediction:
                del scaled_prediction
        del prediction

    ranked = sorted(
        results, key=lambda name: results[name]["robust_selection_score"],
        reverse=True)
    payload = {
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "checkpoint_a_metadata": meta_a,
        "checkpoint_b_metadata": meta_b,
        "correction_checkpoint": (
            str(args.correction_checkpoint)
            if args.correction_checkpoint is not None else None),
        "correction_iteration": (
            int(correction_metadata["iteration"])
            if correction_metadata is not None else None),
        "correction_scales": correction_scales,
        "raw_only": args.raw_only,
        "alpha_definition": "state = (1-alpha)*checkpoint_a + alpha*checkpoint_b",
        "split_manifest": str(args.split_manifest),
        "holdout_re": list(args.holdout_re),
        "samples": len(windows),
        "groups": {name: int(mask.sum()) for name, mask in masks.items()},
        "ranked_candidates": ranked,
        "results": results,
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output),
                      "top": ranked[:10], "wall_seconds": payload["wall_seconds"]}),
          flush=True)


if __name__ == "__main__":
    main()
