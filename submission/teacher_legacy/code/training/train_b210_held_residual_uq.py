#!/usr/bin/env python3
"""Fixed 49/16/16 residual-source experiment; each phase is explicit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from b210_held_residual_split import build_split, filter_rows

SCHEMA = "b210_held_residual_uq_v1"
MANIFEST_SHA = "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e"
SPLIT_SHA = "688766006d7403fd921881d70cd2fa213c524e4381beb9728a19c79ea7a6f025"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_bound(path, expected):
    if not expected or sha256(path) != expected:
        raise ValueError(f"asset identity changed: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split(path):
    rows = read_bound(path, MANIFEST_SHA)
    split = build_split(rows)
    if split["split_sha256"] != SPLIT_SHA:
        raise ValueError("fixed trajectory split changed")
    return rows, split


def validate_pure_start(sim):
    if (sim.get("schema") != "b210_pure_sim_start_v1"
            or sim.get("training_visibility") != "pure_sim_only"
            or sim.get("updates") != 10000 or sim.get("seed") != 0
            or sim.get("start") != "random_initialization"
            or sim.get("sealed") is not True
            or sim.get("real_closure") is not False
            or sim.get("pretext_checkpoint") is not None
            or "pretext_checkpoint" not in sim
            or sim.get("checkpoint_selection") is not False
            or any(sim.get(key) != 0 for key in (
                "real_arrays_read", "real_statistics_read", "teacher_reads",
                "p1_distill_weight", "held_arrays_read", "tail_arrays_read"))):
        raise ValueError("a sealed random-init pure-Sim10000 receipt without Real or teacher access is required")


def matched_windows(windows, metadata, split, role):
    """Equal counts per matched case pair, spanning each case's full prefix."""
    if role not in {"control_uq", "uq_calibration"}:
        raise ValueError("unknown UQ role")
    by_file = {str(row["file"]): [] for row in metadata}
    for window in windows:
        by_file[str(metadata[int(window.trajectory)]["file"])].append(window)
    source = {r["file"]: r for name in ("control_uq", "uq_calibration")
              for r in split["roles"][name]["prefix_rows"]}
    selected = []
    for pair in split["control_matches"]:
        count = min(len(source[pair[name]]["train_starts"])
                    for name in ("control_uq", "uq_calibration"))
        values = by_file[pair[role]]
        if count < 1 or len(values) < count:
            raise ValueError("matched prefix window inventory is empty or changed")
        positions = ([0] if count == 1 else
                     [i * (len(values) - 1) // (count - 1) for i in range(count)])
        selected.extend(values[i] for i in positions)
    return selected


def runtime():
    for path in (Path(__file__).parent, Path(__file__).parents[1],
                 Path(__file__).parents[1] / "evaluation"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    import train_b210_scratch_real_dense_aug as dense
    import train_b210_small4k_uq_release as native
    import train_b210_full_prefix_uq_refit as prefix
    return torch, dense, native, prefix


class RoleAudit:
    def __init__(self, rows):
        self.expected = {r["file"]: int(r["val_boundary"]) for r in rows}
        self.opened = []

    def before_open(self, row, phase):
        if (phase != "fit" or row["file"] not in self.expected
                or int(row["val_boundary"]) != self.expected[row["file"]]
                or row["file"] in self.opened):
            raise RuntimeError("data outside this training role reached the HDF5 opener")

    def record_open(self, row, phase):
        self.before_open(row, phase)
        self.opened.append(row["file"])

    def report(self):
        if set(self.opened) != set(self.expected):
            raise RuntimeError("training role access is incomplete")
        return {"opened": self.opened, "prefix_boundaries": self.expected,
                "tail_arrays_read": 0, "outer_trajectory_arrays_read": 0}


def point_phase(args, rows, split, deadline):
    torch, dense, native, prefix = runtime()
    import train_b210_transfer_real4k as transfer
    sim = read_bound(args.sim_receipt, args.sim_receipt_sha256)
    validate_pure_start(sim)
    fit_rows = filter_rows(rows, split, "point_fit")
    audit = RoleAudit(fit_rows)
    trajectories, metadata, prefix_windows = dense.dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", audit)
    stats = dense.smu1.fold_statistics(
        dense.smu1.window_sufficient_statistics(trajectories, prefix_windows),
        range(len(prefix_windows)))
    normalizer = dense.smu1.FoldNormalizer(stats, torch.device(args.device))
    windows = dense.dense_windows(trajectories, 5)
    dataset = native.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    schedule = dense.smu1.paired_batch_schedule(range(len(windows)), windows, updates=4000, seed=4)
    schedule["updates"] = 4000
    native.ensemble_train.seed_everything(4)
    torch.set_float32_matmul_precision("high")
    model, start = dense.ft12.validate_start_checkpoint(
        Path(sim["checkpoint_path"]), sim["checkpoint_sha256"], normalizer,
        torch.device(args.device), expected_status="PURE_SIM_10000_SEALED")
    start.update(pure_sim_receipt_sha256=args.sim_receipt_sha256,
                 split_sha256=SPLIT_SHA, training_role="point_fit",
                 real_fit_files=split["roles"]["point_fit"]["files"])
    access = dense.ArmSeal("point_fit")
    with transfer.endpoint_contract(dense.small4k):
        training, seal = dense.small4k.train_candidate(
            model, dataset, schedule, normalizer, torch.device(args.device),
            deadline, args.output_dir, start, access,
            input_augmentation=dense.AUGMENTATION,
            gradient_ratio_diagnostics_only=True)
    result = {"schema": SCHEMA + "_point", "split_sha256": SPLIT_SHA,
              "sim_receipt": sim, "sim_receipt_sha256": args.sim_receipt_sha256,
              "checkpoint_path": seal["path"], "checkpoint_sha256": seal["file_sha256"],
              "training": training, "training_role": "point_fit",
              "fit_files": split["roles"]["point_fit"]["files"],
              "normalizer_fit_files": split["roles"]["point_fit"]["files"],
              "access": audit.report(), "dense_windows": len(windows),
              "prefix_windows": len(prefix_windows), "sample_stream_sha256": schedule["sha256"]}
    write_json(args.output_dir / "point_endpoint.json", result)
    return result


def load_point(args, split):
    torch, dense, native, prefix = runtime()
    from models.ensemble_unet2d import EnsembleUNet2d
    receipt = read_bound(args.point_receipt, args.point_receipt_sha256)
    expected = split["roles"]["point_fit"]["files"]
    if (receipt.get("schema") != SCHEMA + "_point"
            or receipt.get("split_sha256") != SPLIT_SHA
            or receipt.get("fit_files") != expected
            or receipt.get("normalizer_fit_files") != expected
            or receipt.get("access", {}).get("outer_trajectory_arrays_read") != 0):
        raise ValueError("point or normalizer used a different Real training role")
    checkpoint = getattr(args, "point_checkpoint", None) or receipt["checkpoint_path"]
    if sha256(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("sealed point checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dense.ft12.validate_b210_model_config(payload["model_config"])
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    normalizer, norm_report = prefix.normalizer_from_point(payload, torch.device(args.device))
    return receipt, payload, model.to(args.device), normalizer, norm_report


def uq_phase(args, rows, split, deadline):
    torch, dense, native, prefix = runtime()
    from torch.utils.data import DataLoader
    from train_b210_transfer_uq import initialize_uq_state
    receipt, point, model, normalizer, norm_report = load_point(args, split)
    frozen = native.dr1.model_state_sha256(native.partition_state(model, uq=False))
    initialization = getattr(args, "initialization", "fresh")
    initialize_uq_state(model, native.UQ_PREFIXES, initialization)
    initial = native.dr1.model_state_sha256(native.partition_state(model, uq=True))
    role = "control_uq" if args.phase == "control" else "uq_calibration"
    fit_rows = filter_rows(rows, split, role)
    audit = RoleAudit(fit_rows)
    trajectories, metadata, windows = native.dr1.load_prefix_rows(args.real_dir, fit_rows, "fit", audit)
    windows = matched_windows(windows, metadata, split, role)
    dataset = native.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    tagged = native.ensemble_train.TrajectoryTaggedDataset(dataset)
    native.uq_train.seed_everything(18)
    torch.set_float32_matmul_precision("high")
    parameters = native.configure_uq_only(model)
    loader = DataLoader(tagged, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(18))
    batches = native.ensemble_train.common.repro.cycling(loader)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3200)
    objective = native.sharp2_args()
    stream = hashlib.sha256()
    started = time.perf_counter()
    for step in range(1, 3201):
        if time.perf_counter() > deadline:
            raise TimeoutError("UQ phase budget expired without endpoint")
        native.set_uq_training_mode(model)
        history, target, _, indices = next(batches)
        stream.update(indices.numpy().astype("<i8", copy=False).tobytes())
        x, y = normalizer.preprocess(history.to(args.device), target.to(args.device))
        optimizer.zero_grad(set_to_none=True)
        prediction, quantiles, extents, logits = native.uq_train.quantile_forward(model, x)
        loss, _ = native.uq_train.uq_objective(
            objective, model, prediction, quantiles, y, normalizer,
            interval_extents=extents, coverage_logits=logits)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite UQ loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not bool(torch.isfinite(norm)) or (step == 1 and float(norm) == 0):
            raise FloatingPointError("invalid UQ gradient")
        optimizer.step()
        scheduler.step()
        if step % 400 == 0:
            print(json.dumps({"phase": args.phase, "step": step, "loss": float(loss),
                              "seconds": time.perf_counter() - started}), flush=True)
    if native.dr1.model_state_sha256(native.partition_state(model, uq=False)) != frozen:
        raise RuntimeError("UQ training changed frozen point state")
    uq_state = {k: v.detach().cpu().clone() for k, v in native.partition_state(model, uq=True).items()}
    path = args.output_dir / "uq3200.pth"
    result = {"schema": SCHEMA + "_uq", "label": args.phase, "split_sha256": SPLIT_SHA,
              "point_receipt_sha256": args.point_receipt_sha256,
              "point_checkpoint_sha256": receipt["checkpoint_sha256"],
              "normalizer_sha256": norm_report["sha256"], "initial_uq_sha256": initial,
              "initialization": initialization,
              "point_frozen_sha256": frozen, "point_state_bitwise": True,
              "training_role": role, "fit_files": split["roles"][role]["files"],
              "fit_windows": len(windows), "sample_stream_sha256": stream.hexdigest(),
              "updates": 3200, "seed": 18, "access": audit.report(),
              "seconds": time.perf_counter() - started}
    torch.save({**result, "uq_state_dict": uq_state}, path)
    result.update(checkpoint_path=str(path.resolve()), checkpoint_sha256=sha256(path))
    write_json(args.output_dir / "uq_endpoint.json", result)
    return result


def validate_pair(receipts, split):
    for label, item in receipts.items():
        role = "control_uq" if label == "control" else "uq_calibration"
        if (item.get("schema") != SCHEMA + "_uq" or item.get("label") != label
                or item.get("split_sha256") != SPLIT_SHA or item.get("updates") != 3200
                or item.get("seed") != 18 or item.get("point_state_bitwise") is not True
                or item.get("training_role") != role
                or item.get("fit_files") != split["roles"][role]["files"]
                or item.get("access", {}).get("outer_trajectory_arrays_read") != 0):
            raise ValueError("invalid residual-source endpoint")
    for key in ("point_receipt_sha256", "point_checkpoint_sha256", "normalizer_sha256",
                "initial_uq_sha256", "point_frozen_sha256", "fit_windows", "sample_stream_sha256"):
        if not receipts["control"].get(key) or receipts["control"][key] != receipts["candidate"].get(key):
            raise ValueError("UQ comparison differs in " + key)


def absolute_distribution(prediction, lo, hi, normalizer):
    """Convert signed normalized residual extents to physical absolute bounds."""
    prediction = normalizer.postprocess_prediction(prediction.float())
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower, upper = prediction.new_zeros(prediction.shape), prediction.new_zeros(prediction.shape)
    lower[..., :2] = prediction[..., :2] - lo.float() * scale
    upper[..., :2] = prediction[..., :2] + hi.float() * scale
    prediction[..., 2] = 0
    return prediction, lower, upper


def eval_phase(args, rows, split, deadline):
    torch, dense, native, prefix = runtime()
    import numpy as np
    import evaluate_b210_transfer as evaluation
    receipts = {label: read_bound(getattr(args, label + "_receipt"),
                                getattr(args, label + "_receipt_sha256"))
                for label in ("control", "candidate")}
    validate_pair(receipts, split)
    point_receipt, point, model, normalizer, norm_report = load_point(args, split)
    if receipts["control"]["point_receipt_sha256"] != args.point_receipt_sha256:
        raise ValueError("evaluation point differs from UQ training")
    # Validate and load BOTH sealed endpoints before the first outer HDF5 access.
    states = {}
    for label, item in receipts.items():
        if sha256(item["checkpoint_path"]) != item["checkpoint_sha256"]:
            raise ValueError("sealed UQ checkpoint changed")
        payload = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
        if any(payload.get(k) != v for k, v in item.items()
               if k not in {"checkpoint_path", "checkpoint_sha256"}):
            raise ValueError("UQ receipt and checkpoint metadata differ")
        states[label] = payload["uq_state_dict"]
        if set(states[label]) != set(native.partition_state(model, uq=True)):
            raise ValueError("UQ state keys changed")
    official = evaluation.import_file(evaluation.SCORER, "held_residual_official_scoring")
    outer = set(split["roles"]["outer_test"]["files"])
    eval_rows = [row for row in evaluation.clean_rows(args.clean_manifest) if row["file"] in outer]
    results = {}
    for label in ("control", "candidate"):
        model.load_state_dict({**model.state_dict(), **states[label]}, strict=True)
        if native.dr1.model_state_sha256(native.partition_state(model, uq=False)) != receipts[label]["point_frozen_sha256"]:
            raise RuntimeError("evaluation changed frozen point")
        model.eval()
        by_file = {row["file"]: [] for row in eval_rows}
        input_hash, truth_hash, point_hash = (hashlib.sha256() for _ in range(3))
        with torch.inference_mode():
            for inputs, truth, labels in evaluation.tail_batches(eval_rows, args.real_dir):
                if time.perf_counter() > deadline:
                    raise TimeoutError("held evaluation budget expired")
                input_hash.update(inputs.tobytes()); truth_hash.update(truth.tobytes())
                x, _ = normalizer.preprocess(torch.from_numpy(inputs).to(args.device),
                                            torch.from_numpy(truth).to(args.device))
                pred, lo, hi = model.forward_distribution(x)
                pred, lower, upper = absolute_distribution(pred, lo, hi, normalizer)
                prediction, lower, upper = (value.cpu().numpy() for value in (pred, lower, upper))
                point_hash.update(prediction.tobytes())
                for name in dict.fromkeys(labels):
                    mask = np.asarray([item == name for item in labels])
                    by_file[name].append(evaluation.metric_state(
                        official, prediction[mask], truth[mask], lower[mask], upper[mask]))
        by_file = {name: evaluation.merge_states(value) for name, value in by_file.items()}
        pooled = evaluation.merge_states(list(by_file.values()))
        by_aoa = {str(aoa): evaluation.finalize(official, evaluation.merge_states(
                    [by_file[row["file"]] for row in eval_rows if row["aoa"] == aoa]))
                  for aoa in (0, 5, 10, 15, 20)}
        results[label] = {"metrics": evaluation.finalize(official, pooled), "by_aoa": by_aoa,
                          "by_trajectory": by_file, "input_sha256": input_hash.hexdigest(),
                          "truth_sha256": truth_hash.hexdigest(), "point_sha256": point_hash.hexdigest()}
    for key in ("input_sha256", "truth_sha256", "point_sha256"):
        if results["control"][key] != results["candidate"][key]:
            raise RuntimeError("paired evaluation mismatch: " + key)
    delta = {key: results["candidate"]["metrics"][key] - results["control"]["metrics"][key]
             for key in results["control"]["metrics"]}
    aoa_delta = {key: results["candidate"]["by_aoa"][key]["sps_score"] -
                     results["control"]["by_aoa"][key]["sps_score"] for key in results["control"]["by_aoa"]}
    passed = (delta["fixed_time_fitted_final"] >= .10
              and sum(v > 0 for v in aoa_delta.values()) >= 3 and min(aoa_delta.values()) >= -.10)
    report = {"schema": SCHEMA + "_result", "split_sha256": SPLIT_SHA,
              "decision": "PASS_TO_REPLICATION" if passed else "STOP_HELD_RESIDUAL_CONFIG",
              "results": results, "candidate_minus_control": delta, "aoa_sps_deltas": aoa_delta,
              "receipts": receipts, "outer_files": sorted(outer),
              "evidence": "FP32 independent trajectory holdout for this split; not official score or release qualification",
              "upload_authorized": False}
    write_json(args.output_dir / "decision.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("point", "control", "candidate", "eval"), required=True)
    parser.add_argument("--execute", action="store_true")
    for name in ("clean-manifest", "real-dir", "output-dir", "sim-receipt", "point-receipt",
                 "control-receipt", "candidate-receipt"):
        parser.add_argument("--" + name, type=Path)
    for name in ("sim", "point", "control", "candidate"):
        parser.add_argument("--" + name + "-receipt-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initialization", choices=("fresh", "point"), default="fresh")
    parser.add_argument("--point-checkpoint", type=Path)
    parser.add_argument("--wall-budget-seconds", type=float)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"schema": SCHEMA, "phase": args.phase, "decision": "PLAN_ONLY_NO_DATA_NO_GPU",
                          "split_sha256": SPLIT_SHA, "training_roles": [49, 16, 16]}))
        return
    if (not args.clean_manifest or not args.real_dir or not args.output_dir
            or not args.wall_budget_seconds or not math.isfinite(args.wall_budget_seconds)
            or args.wall_budget_seconds <= 0):
        raise ValueError("explicit data paths, new output directory, and wall budget are required")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows, split = load_split(args.clean_manifest)
    torch, *_ = runtime()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("execution requires CUDA")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "split.json", split)
    deadline = time.perf_counter() + args.wall_budget_seconds
    function = point_phase if args.phase == "point" else eval_phase if args.phase == "eval" else uq_phase
    result = function(args, rows, split, deadline)
    print(json.dumps({"phase": args.phase, "decision": result.get("decision", "SEALED"),
                      "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
