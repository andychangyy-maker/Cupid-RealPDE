"""One frozen-old-C scalar/matrix head trajectory and native FP16 point screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "b210_uv_matrix_head_v1"
OLD_C_SHA = "96c0082c921c2aaeb449ef14ba5f94a7361395164b2064a191c407c337509903"
MANIFEST_SHA = "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e"
UPDATES, SEED, BATCH_SIZE = 1200, 4, 8


def plan():
    return {"schema": SCHEMA, "decision": "PLAN_ONLY_NO_DATA_NO_GPU", "old_C_sha256": OLD_C_SHA,
            "updates": UPDATES, "seed": SEED, "batch_size": BATCH_SIZE,
            "heads": {"scalar": 211, "matrix": 844}, "coefficient": ".25*tanh",
            "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": 1e-5,
                          "cosine_t_max": UPDATES, "clip_each_head": 1., "independent_per_head": True},
            "point_loss": {"accuracy_weights": [5., 3., 2.], "mse_weight": .25, "mse_warmup_updates": 0},
            "prefix_windows": 2544, "optimizer_windows": 9933, "stride": 5,
            "augmentation": False, "teacher": False, "uq_training": False, "ema": False,
            "frozen": "entire old C including controller, UQ and normalizer buffers; shared detached forward",
            "evaluation": "one native FP16 forward per batch for old_C/scalar/matrix on clean716",
            "gate": "matrix-old_C point-only fittedFinal >= .10; >=3/5 AoA positive; Rel nonnegative; matrix-scalar Final > 0",
            "wall_cap_seconds": 1200, "release_or_upload": False}


def check_deadline(deadline):
    if time.perf_counter() > deadline:
        raise TimeoutError("UV matrix diagnostic budget expired; no further stage admitted")


def evaluate(model, heads, normalizer_values, rows, args, deadline):
    import numpy as np
    import torch
    import evaluate_b210_transfer as replay
    from evaluate_b256_offline_teacher import summarize
    from models.uv_fluctuation_head import frozen_point_and_feature

    model.half().eval()
    heads.half().eval()
    stats = {key: torch.as_tensor(value, dtype=torch.float16, device=args.device)
             for key, value in normalizer_values.items()}
    official = replay.import_file(replay.SCORER, "uv_matrix_official_scoring")
    names = ("old_C", "scalar", "matrix")
    errors, prediction_hashes = {name: [] for name in names}, {name: hashlib.sha256() for name in names}
    input_hash, target_hash, labels = hashlib.sha256(), hashlib.sha256(), []
    with torch.no_grad():
        for inputs, truth, batch_labels in replay.tail_batches(rows, args.real_dir):
            check_deadline(deadline)
            values = torch.as_tensor(inputs, dtype=torch.float16, device=args.device)
            x = (values - stats["mean_input"]) / stats["std_input"]
            point, feature = frozen_point_and_feature(model, x)
            predictions = {"old_C": point, **{name: head(point, feature, stats["std_target"])
                                              for name, head in heads.items()}}
            for name, value in predictions.items():
                prediction = (value * stats["std_target"] + stats["mean_target"]).float().cpu().numpy()
                prediction[..., 2] = 0.
                if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                    raise ValueError("invalid native FP16 prediction: " + name)
                errors[name].append(np.stack((official.rel_l2_per_sample(prediction, truth, 2),
                    official.tke_rel_l2_per_sample(prediction, truth, 2),
                    official.mvpe_rel_l2_per_sample(prediction, truth)), axis=1))
                prediction_hashes[name].update(prediction.tobytes())
            input_hash.update(inputs.tobytes()); target_hash.update(truth.tobytes())
            labels.extend(batch_labels)
    if len(labels) != 716:
        raise ValueError("point screen must cover exactly clean716")
    aoa_by_file = {row["file"]: row["aoa"] for row in rows}
    aoas = np.asarray([aoa_by_file[label] for label in labels])
    result = {}
    for name in names:
        array = np.concatenate(errors[name])
        if not np.isfinite(array).all():
            raise ValueError("nonfinite point errors: " + name)
        result[name] = {"metrics": summarize(official, array),
            "by_aoa": {str(a): summarize(official, array[aoas == a]) for a in (0, 5, 10, 15, 20)},
            "per_window_errors_rel_tke_mvpe": array.tolist(),
            "input_sha256": input_hash.hexdigest(), "target_sha256": target_hash.hexdigest(),
            "prediction_sha256": prediction_hashes[name].hexdigest()}
    return result


def execute(args):
    started = time.perf_counter()
    if (any(getattr(args, key) is None for key in ("point_checkpoint", "clean_manifest", "real_dir", "output_dir"))
            or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit or "")
            or not math.isfinite(args.wall_budget_seconds) or not 0 < args.wall_budget_seconds <= 1200):
        raise ValueError("execution requires bound assets, full source commit and at most 1200 seconds")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    deadline = started + args.wall_budget_seconds
    for path in (ROOT / "code", ROOT / "code/training", ROOT / "code/evaluation"):
        sys.path.insert(0, str(path))
    import torch
    from torch.utils.data import DataLoader
    import train_b210_scratch_real_dense_aug as dense
    import train_b210_full_prefix_uq_refit as prefix
    import evaluate_b210_transfer as replay
    from evaluate_b256_offline_teacher import FIXED_SPS
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from models.uv_fluctuation_head import UVFluctuationHead, frozen_point_and_feature

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision("high")
    dr1, train = dense.dr1, dense.ft12.npt1.ensemble_train
    if replay.sha256(args.point_checkpoint) != OLD_C_SHA or replay.sha256(args.clean_manifest) != MANIFEST_SHA:
        raise ValueError("old C checkpoint or clean-prefix manifest bytes changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("execution requires CUDA; plan-only does not import torch")
    torch.cuda.set_device(device)
    payload = torch.load(args.point_checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["model_config"])
    if config.pop("architecture") != "ensemble_unet" or config["base"] != 210:
        raise ValueError("old C architecture changed")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False).eval()
    frozen_before = dr1.model_state_sha256(model.state_dict())
    normalizer, normalization = prefix.normalizer_from_point(payload, device)
    normalizer_values = payload["normalizer"]
    del payload
    model.to(device)
    rows, manifest = dr1.load_manifest(args.clean_manifest, dense.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    audit = prefix.PrefixOnlyAudit()
    trajectories, metadata, fit_windows = dr1.load_prefix_rows(args.real_dir, rows, "fit", audit)
    windows = dense.dense_windows(trajectories, 5)
    if len(fit_windows) != 2544 or len(windows) != 9933 or not audit.report(rows)["allowlist_exact"]:
        raise ValueError("prefix/dense window population changed")
    loaded_frames = {row["file"]: int(values.shape[0]) for row, values in zip(rows, trajectories)}
    if any(loaded_frames[row["file"]] != row["val_boundary"] for row in rows):
        raise ValueError("loaded arrays exceed the fixed prefix boundary")
    schedule = dense.smu1.paired_batch_schedule(range(len(windows)), windows, updates=UPDATES, seed=SEED)
    dataset = train.common.repro.WindowDataset(trajectories, windows)
    loader = DataLoader(train.TrajectoryTaggedDataset(dataset), batch_sampler=schedule["batches"], num_workers=0,
                        pin_memory=True, generator=torch.Generator().manual_seed(SEED))
    train.seed_everything(SEED)
    heads = torch.nn.ModuleDict({name: UVFluctuationHead(name == "matrix") for name in ("scalar", "matrix")}).to(device)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=1e-5) for name, head in heads.items()}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UPDATES) for name, opt in optimizers.items()}
    loss_args = argparse.Namespace(point_loss_mode="official_accuracy", mse_weight=.25, mse_warmup_updates=0)
    actual_rows, history = [], []
    for step, (inputs, target, trajectories_in_batch, indices) in enumerate(loader, 1):
        check_deadline(deadline)
        expected = schedule["rows"][step - 1]
        actual = {"sample_indices": indices.tolist(), "trajectory_ids": trajectories_in_batch.tolist(),
                  "bootstrap_mask": expected["bootstrap_mask"]}
        if actual != expected or len(indices) != BATCH_SIZE:
            raise ValueError("paired head sample stream changed")
        actual_rows.append(actual)
        x, y = normalizer.preprocess(inputs.to(device), target.to(device))
        point, feature = frozen_point_and_feature(model, x)
        losses = {}
        for name, head in heads.items():
            optimizers[name].zero_grad(set_to_none=True)
            prediction = head(point, feature, normalizer.std_target)
            loss, _ = train.point_loss(loss_args, prediction, y, normalizer, (5., 3., 2.), step)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite head loss: " + name)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            if not bool(torch.isfinite(norm)) or (step == 1 and float(norm) == 0):
                raise FloatingPointError("invalid head gradient: " + name)
            optimizers[name].step(); schedulers[name].step()
            losses[name] = float(loss.detach())
        if step == 1 or step % 200 == 0:
            entry = {"step": step, "losses": losses, "elapsed_seconds": time.perf_counter() - started}
            history.append(entry); print(json.dumps(entry), flush=True)
    if len(actual_rows) != UPDATES or dr1.canonical_sha256(actual_rows) != schedule["sha256"]:
        raise ValueError("incomplete shared head trajectory")
    frozen_after = dr1.model_state_sha256(model.state_dict())
    if frozen_after != frozen_before or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("head training changed the frozen old C model")
    check_deadline(deadline)
    contract = {**plan(), "decision": "HEADS_U1200_SEALED_POINT_SCREEN_PENDING", "sealed": True,
        "source_commit": args.source_commit, "point_checkpoint": str(args.point_checkpoint.resolve()),
        "source_sha256": {str(path.relative_to(ROOT)): replay.sha256(path) for path in
                          (Path(__file__), ROOT / "code/models/uv_fluctuation_head.py")},
        "manifest_sha256": MANIFEST_SHA, "manifest": manifest,
        "normalizer_sha256": normalization["sha256"], "normalizer": normalizer_values,
        "sample_stream_sha256": schedule["sha256"], "actual_stream_sha256": dr1.canonical_sha256(actual_rows),
        "bootstrap_masks_used_for_loss": False, "access_audit": audit.report(rows),
        "max_read_frame_exclusive_by_file": loaded_frames,
        "ordered_prefix_window_sha256": dense.smu1.ordered_held_key_digest(range(len(windows)), windows, metadata),
        "frozen_model_state_sha256": frozen_after, "frozen_model_unchanged": True,
        "head_parameters": {name: sum(p.numel() for p in head.parameters()) for name, head in heads.items()},
        "final_lr": {name: opt.param_groups[0]["lr"] for name, opt in optimizers.items()},
        "gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "training_precision": "FP32/high", "history": history, "training_elapsed_seconds": time.perf_counter() - started}
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "heads_u1200.pth"
    torch.save({**contract, "head_state_dict": {key: value.detach().cpu() for key, value in heads.state_dict().items()}}, checkpoint)
    receipt = {**contract, "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": replay.sha256(checkpoint)}
    dr1.write_json_atomic(args.output_dir / "heads_endpoint.json", receipt)
    # Evaluate the exact sealed small states, retaining old C's loaded buffers.
    heads.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["head_state_dict"], strict=True)
    check_deadline(deadline)
    results = evaluate(model, heads, normalizer_values, replay.clean_rows(args.clean_manifest), args, deadline)
    key = "point_only_fitted_final"
    matrix, old, scalar = (results[name] for name in ("matrix", "old_C", "scalar"))
    delta = matrix["metrics"][key] - old["metrics"][key]
    scalar_delta = matrix["metrics"][key] - scalar["metrics"][key]
    aoa = {a: matrix["by_aoa"][a][key] - old["by_aoa"][a][key] for a in matrix["by_aoa"]}
    rel = matrix["metrics"]["rel_l2_score"] - old["metrics"]["rel_l2_score"]
    passed = delta >= .10 and sum(value > 0 for value in aoa.values()) >= 3 and rel >= 0 and scalar_delta > 0
    check_deadline(deadline)
    result = {"schema": SCHEMA + "_point_result", "models": results,
        "head_receipt_sha256": replay.sha256(args.output_dir / "heads_endpoint.json"),
        "scorer_sha256": replay.sha256(replay.SCORER), "fixed_v3_sps": FIXED_SPS, "fixed_time_score": replay.TIME_SCORE,
        "precision": "all three native FP16; one shared old-C forward; FP32 physical scoring; no INT8",
        "gate": {"pass": passed, "matrix_minus_old_C": delta, "matrix_minus_scalar": scalar_delta,
                 "matrix_minus_old_C_rel": rel, "by_aoa_delta": aoa,
                 "min_point_only_final": .10, "min_positive_aoas": 3, "rel_floor": 0., "matrix_must_beat_scalar": True},
        "decision": "MATRIX_POINT_SIGNAL_REQUIRES_INDEPENDENT_CONFIRMATION" if passed else "STOP_UV_MATRIX_HEAD_CONFIG",
        "evidence": "explored clean716 frozen-head diagnostic, not official improvement or full-pipeline LOAO",
        "sps_measured": False, "time_measured": False, "release_verified": False,
        "elapsed_seconds": time.perf_counter() - started}
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    return {"decision": result["decision"], "gate": result["gate"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    for name in ("point-checkpoint", "clean-manifest", "real-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=1200)
    args = parser.parse_args()
    print(json.dumps(execute(args) if args.execute else plan(), indent=2))
