#!/usr/bin/env python3
"""Verify the exact B210 G3d import-prewarm probe without submitting it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = REPO_ROOT / "code/submissions"
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

import build_track1_b210_import_prewarm_probe as builder  # noqa: E402


SCHEMA = "b210_g3d_import_prewarm_probe_verify_v1"
FORMAL_BRANCH = "codex/b210-g3d-import-prewarm-probe"
FORMAL_ACKNOWLEDGEMENT = "B210_G3D_IMPORT_PREWARM_PROBE_GPU_VERIFY"
EXPECTED_SAMPLES = 725
EXTERNAL_BATCH_SIZE = 16
EXPECTED_CALLS = 46
EXPECTED_LAST_BATCH = 5
MAX_IMPORT_SECONDS = 15.0
MAX_PREDICT_RATIO = 0.75
MAX_TOTAL_RATIO = 1.10
MAX_CUDA_RESERVED_BYTES = 14 * 1024**3
MAX_RSS_BYTES = 16 * 1024**3
WALL_BUDGET_SECONDS = 20 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-cpu", action="store_true")
    mode.add_argument("--verify-gpu", action="store_true")
    mode.add_argument("--worker-cpu", action="store_true")
    mode.add_argument("--worker-gpu", action="store_true")
    parser.add_argument("--baseline-archive", type=Path)
    parser.add_argument("--probe-archive", type=Path)
    parser.add_argument("--probe-archive-sha256")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--arm", choices=("baseline", "probe"))
    parser.add_argument(
        "--wrapper", type=Path,
        default=REPO_ROOT / "code/submissions/track1_ensemble_unet2d.py")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--input-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary file {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def archive_audit(
    path: Path,
    expected_sha256: str,
    *,
    wrapper: Path | None,
    probe: bool,
) -> dict[str, object]:
    path = path.resolve()
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256.lower():
        raise ValueError(f"archive SHA-256 differs: {path}")
    if identity["bytes"] > builder.MAX_ARCHIVE_BYTES:
        raise ValueError(f"archive exceeds 256 MiB: {path}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        builder.validate_member_names(infos)
        if archive.testzip() is not None:
            raise ValueError(f"archive CRC validation failed: {path}")
        payloads = {info.filename: archive.read(info) for info in infos}
    members = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in payloads.items()
    }
    unchanged = {
        name: members[name]["sha256"] == expected
        for name, expected in builder.BASELINE_MEMBER_SHA256.items()
        if name != "submission.py"
    }
    if not all(unchanged.values()):
        raise ValueError(f"non-wrapper member differs: {unchanged}")
    if probe:
        if wrapper is None:
            raise ValueError("probe audit requires tracked wrapper")
        wrapper_payload = wrapper.resolve().read_bytes()
        builder.validate_wrapper_source(wrapper_payload)
        if payloads["submission.py"] != wrapper_payload:
            raise ValueError("probe submission.py differs from tracked wrapper")
        if (members["submission.py"]["sha256"]
                == builder.BASELINE_MEMBER_SHA256["submission.py"]):
            raise ValueError("probe wrapper is unchanged")
    elif {
            name: row["sha256"] for name, row in members.items()
    } != builder.BASELINE_MEMBER_SHA256:
        raise ValueError("baseline member ledger differs")
    return {
        **identity,
        "members": members,
        "non_wrapper_members_bitwise_identical": unchanged,
        "only_submission_member_changed": (
            probe and all(unchanged.values())
            and members["submission.py"]["sha256"]
            != builder.BASELINE_MEMBER_SHA256["submission.py"]),
    }


def safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        builder.validate_member_names(archive.infolist())
        archive.extractall(destination)


def load_submission(package_dir: Path):
    for name in tuple(sys.modules):
        if name == "models" or name.startswith("models."):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location(
        "g3d_evaluated_submission", package_dir / "submission.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load packaged submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_result(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(result, dict):
        raise TypeError("submission must return a prediction/bounds dict")
    values = tuple(np.asarray(result[name])
                   for name in ("prediction", "lower", "upper"))
    if any(value.dtype != np.float32 for value in values):
        raise TypeError("submission outputs must remain float32")
    return values


def run_cpu_worker(args: argparse.Namespace) -> dict[str, object]:
    if torch.cuda.is_available():
        raise RuntimeError("CPU smoke worker unexpectedly sees CUDA")
    archive = args.archive.resolve()
    if sha256(archive) != args.archive_sha256.lower():
        raise ValueError("CPU worker archive SHA differs")
    args.temp_root.mkdir(parents=True, exist_ok=True)
    original_load = torch.load

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("CPU import attempted to load the checkpoint")

    torch.load = forbidden_load
    try:
        with tempfile.TemporaryDirectory(
                prefix="g3d-cpu-import-", dir=args.temp_root) as temporary:
            package = Path(temporary)
            safe_extract(archive, package)
            started = time.perf_counter()
            submission = load_submission(package)
            import_seconds = time.perf_counter() - started
            state = submission._state
            if any(state[name] is not None for name in ("model", "device", "dtype")):
                raise RuntimeError("CPU import did not remain lazy")
            invalid_shape_guard = False
            try:
                submission.predict(
                    np.zeros((1, 19, 32, 64, 3), dtype=np.float32), metadata={})
            except ValueError as error:
                invalid_shape_guard = "shape" in str(error)
            if not invalid_shape_guard:
                raise RuntimeError("CPU import lost the public shape guard")
    finally:
        torch.load = original_load
    result = {
        "schema": SCHEMA,
        "kind": "cpu_import_worker",
        "archive": file_identity(archive),
        "cuda_available": False,
        "import_seconds": import_seconds,
        "state_model_is_none": state["model"] is None,
        "state_device_is_none": state["device"] is None,
        "state_dtype_is_none": state["dtype"] is None,
        "invalid_shape_guard": invalid_shape_guard,
        "checkpoint_load_forbidden": True,
        "peak_rss_bytes": peak_rss_bytes(),
        "decision": "PASS_G3D_CPU_IMPORT_WORKER",
    }
    write_json(args.worker_output, result)
    return result


def run_process(command: list[str], timeout: float,
                environment: dict[str, str] | None = None) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    if environment:
        env.update(environment)
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True,
        text=True, timeout=timeout, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed rc={completed.returncode}; "
            f"stdout={completed.stdout[-2000:]}; stderr={completed.stderr[-4000:]}")
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def run_static_cpu(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    probe_sha = args.probe_archive_sha256.lower()
    baseline = archive_audit(
        args.baseline_archive, builder.BASELINE_ARCHIVE_SHA256,
        wrapper=None, probe=False)
    probe = archive_audit(
        args.probe_archive, probe_sha, wrapper=args.wrapper, probe=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="g3d-static-cpu-", dir=args.temp_root) as temporary:
        worker_output = Path(temporary) / "cpu_worker.json"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker-cpu", "--archive", str(args.probe_archive.resolve()),
            "--archive-sha256", probe_sha,
            "--worker-output", str(worker_output),
            "--temp-root", str(Path(temporary) / "extract"),
        ]
        process = run_process(
            command, 60.0, {"CUDA_VISIBLE_DEVICES": ""})
        worker = json.loads(worker_output.read_text(encoding="utf-8"))
    gates = {
        "baseline_archive_exact": (
            baseline["sha256"] == builder.BASELINE_ARCHIVE_SHA256),
        "probe_archive_exact": probe["sha256"] == probe_sha,
        "only_submission_member_changed": (
            probe["only_submission_member_changed"] is True),
        "archive_under_256mib": probe["bytes"] <= builder.MAX_ARCHIVE_BYTES,
        "cpu_cuda_unavailable": worker["cuda_available"] is False,
        "cpu_import_le_15s": worker["import_seconds"] <= MAX_IMPORT_SECONDS,
        "cpu_checkpoint_remains_lazy": all(
            worker[key] is True for key in (
                "state_model_is_none", "state_device_is_none",
                "state_dtype_is_none", "checkpoint_load_forbidden")),
        "cpu_invalid_shape_guard": worker["invalid_shape_guard"] is True,
    }
    return {
        "schema": SCHEMA,
        "evidence_class": "g3d_static_cpu_archive_qualification",
        "baseline_archive": baseline,
        "probe_archive": probe,
        "wrapper": file_identity(args.wrapper.resolve()),
        "cpu_worker": worker,
        "cpu_process": process,
        "gates": gates,
        "submission_authorized": False,
        "decision": (
            "PASS_G3D_PROBE_STATIC"
            if all(gates.values()) else "STOP_G3D_PROBE_STATIC"),
    }


def build_fixed725_cache(
    real_dir: Path, real_stats: Path, output: Path,
) -> dict[str, object]:
    code_root = REPO_ROOT / "code"
    training_root = code_root / "training"
    for path in (code_root, training_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import train_hybrid_fno2d as common

    split_dir = output.parent / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    data_args = SimpleNamespace(
        real_dir=real_dir.resolve(), max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8, eval_batch_size=EXTERNAL_BATCH_SIZE,
        num_workers=0, real_stats=real_stats.resolve(), output_dir=split_dir,
        seed=0)
    (_train, val_loader, _normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(
         data_args, torch.device("cpu"))
    chunks = [history.numpy().astype(np.float32, copy=False)
              for history, _target in val_loader]
    values = np.concatenate(chunks)
    if values.shape != (EXPECTED_SAMPLES, 20, 32, 64, 3):
        raise ValueError(f"fixed725 input shape differs: {values.shape}")
    if not values.flags.c_contiguous or np.count_nonzero(values[..., 2]) != 0:
        raise ValueError("fixed725 input layout/pressure contract differs")
    np.save(output, values, allow_pickle=False)
    return {
        **file_identity(output),
        "array_shape": list(values.shape),
        "array_dtype": str(values.dtype),
        "array_sha256": sha256_bytes(values.tobytes(order="C")),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
    }


def update_output_hashes(
    hashers: dict[str, object], arrays: tuple[np.ndarray, ...],
    expected_shape: tuple[int, ...],
) -> None:
    for name, array in zip(("prediction", "lower", "upper"), arrays):
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape differs: {array.shape}")
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{name} is non-finite")
        hashers[name].update(np.ascontiguousarray(array).tobytes(order="C"))
    if np.any(arrays[1] > arrays[2]):
        raise FloatingPointError("submission bounds are unordered")
    if any(np.count_nonzero(array[..., 2]) != 0 for array in arrays):
        raise ValueError("submission output pressure is not exactly zero")


def run_gpu_worker(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU worker cannot see CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "4080 SUPER" not in gpu_name:
        raise RuntimeError(f"G3d is frozen to RTX 4080 SUPER, got {gpu_name}")
    archive = args.archive.resolve()
    if sha256(archive) != args.archive_sha256.lower():
        raise ValueError("GPU worker archive SHA differs")
    cache = np.load(args.input_cache.resolve(), mmap_mode="r", allow_pickle=False)
    if cache.shape != (EXPECTED_SAMPLES, 20, 32, 64, 3):
        raise ValueError("GPU worker input cache shape differs")
    input_cache = file_identity(args.input_cache.resolve())
    input_array_sha = sha256_bytes(
        np.ascontiguousarray(cache).tobytes(order="C"))
    args.temp_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(
            prefix=f"g3d-{args.arm}-", dir=args.temp_root) as temporary:
        package = Path(temporary)
        safe_extract(archive, package)
        torch.cuda.synchronize()
        import_started = time.perf_counter()
        submission = load_submission(package)
        torch.cuda.synchronize()
        import_seconds = time.perf_counter() - import_started
        import_state_materialized = submission._state["model"] is not None
        expected_materialized = args.arm == "probe"
        if import_state_materialized is not expected_materialized:
            raise RuntimeError(
                f"{args.arm} import materialization differs from contract")
        if submission._BATCH_SIZE != 16:
            raise RuntimeError("packaged wrapper batch size changed")

        hashers = {name: hashlib.sha256()
                   for name in ("prediction", "lower", "upper")}
        call_batch_sizes = []
        durations = []
        first_predict_seconds = None
        input_bitwise_unchanged = True
        for start in range(0, EXPECTED_SAMPLES, EXTERNAL_BATCH_SIZE):
            raw = np.ascontiguousarray(
                cache[start:start + EXTERNAL_BATCH_SIZE])
            frozen = raw.copy()
            torch.cuda.synchronize()
            started = time.perf_counter()
            arrays = normalized_result(submission.predict(raw, metadata={}))
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            if first_predict_seconds is None:
                first_predict_seconds = duration
            durations.append(duration)
            call_batch_sizes.append(len(raw))
            input_bitwise_unchanged &= bool(np.array_equal(raw, frozen))
            update_output_hashes(hashers, arrays, tuple(raw.shape))

        remainder = np.ascontiguousarray(cache[:17])
        frozen_remainder = remainder.copy()
        remainder_arrays = normalized_result(
            submission.predict(remainder, metadata={}))
        torch.cuda.synchronize()
        remainder_hashes = {
            name: sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))
            for name, array in zip(
                ("prediction", "lower", "upper"), remainder_arrays)
        }
        update_output_hashes(
            {name: hashlib.sha256()
             for name in ("prediction", "lower", "upper")},
            remainder_arrays, tuple(remainder.shape))
        input_bitwise_unchanged &= bool(
            np.array_equal(remainder, frozen_remainder))
        state = submission._state
        device_type = state["device"].type
        dtype = str(state["dtype"])
        if device_type != "cuda" or state["dtype"] != torch.float16:
            raise RuntimeError("packaged state is not CUDA FP16")
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

    predict_total = float(sum(durations))
    result = {
        "schema": SCHEMA,
        "kind": "gpu_archive_worker",
        "arm": args.arm,
        "archive": file_identity(archive),
        "input_cache": input_cache,
        "input_array_sha256": input_array_sha,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "import_state_materialized": import_state_materialized,
        "device_type": device_type,
        "dtype": dtype,
        "batch_size": submission._BATCH_SIZE,
        "samples": int(sum(call_batch_sizes)),
        "predict_calls": len(call_batch_sizes),
        "call_batch_sizes": call_batch_sizes,
        "first_predict_seconds": first_predict_seconds,
        "predict_seconds_total": predict_total,
        "predict_seconds_mean_per_call": predict_total / len(durations),
        "import_seconds": import_seconds,
        "import_plus_predict_seconds": import_seconds + predict_total,
        "ordered_output_sha256": {
            name: hasher.hexdigest() for name, hasher in hashers.items()},
        "remainder17_output_sha256": remainder_hashes,
        "input_bitwise_unchanged": input_bitwise_unchanged,
        "finite_ordered_pressure_zero": True,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_rss_bytes": peak_rss_bytes(),
        "decision": "PASS_G3D_GPU_ARCHIVE_WORKER",
    }
    write_json(args.worker_output, result)
    return result


def git_identity() -> dict[str, object]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True)
        return completed.stdout.strip()
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "status_porcelain": git("status", "--porcelain"),
    }


def gpu_worker_command(
    args: argparse.Namespace, arm: str, archive: Path,
    archive_sha: str, cache: Path, output: Path, extract_root: Path,
) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--worker-gpu",
        "--arm", arm, "--archive", str(archive.resolve()),
        "--archive-sha256", archive_sha, "--input-cache", str(cache),
        "--worker-output", str(output), "--temp-root", str(extract_root),
    ]


def run_gpu_verify(args: argparse.Namespace) -> dict[str, object]:
    identity = git_identity()
    if (identity["branch"] != FORMAL_BRANCH
            or identity["status_porcelain"] != ""):
        raise RuntimeError(f"formal branch/worktree identity differs: {identity}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    probe_sha = args.probe_archive_sha256.lower()
    baseline_audit = archive_audit(
        args.baseline_archive, builder.BASELINE_ARCHIVE_SHA256,
        wrapper=None, probe=False)
    probe_audit = archive_audit(
        args.probe_archive, probe_sha, wrapper=args.wrapper, probe=True)
    started = time.perf_counter()
    deadline = started + WALL_BUDGET_SECONDS
    args.temp_root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="g3d-fixed725-", dir=args.temp_root) as temporary:
        root = Path(temporary)
        cache = root / "fixed725.npy"
        cache_identity = build_fixed725_cache(
            args.real_dir, args.real_stats, cache)
        sequence = (
            ("baseline", "01_baseline_a.json"),
            ("probe", "02_probe_a.json"),
            ("probe", "03_probe_b.json"),
            ("baseline", "04_baseline_b.json"),
        )
        rows = []
        processes = []
        for arm, filename in sequence:
            worker_output = args.output.parent / filename
            if worker_output.exists():
                raise FileExistsError(
                    f"refusing to overwrite worker output {worker_output}")
            archive = (args.baseline_archive if arm == "baseline"
                       else args.probe_archive)
            archive_sha = (builder.BASELINE_ARCHIVE_SHA256 if arm == "baseline"
                           else probe_sha)
            process = run_process(
                gpu_worker_command(
                    args, arm, archive, archive_sha, cache, worker_output,
                    root / f"extract-{arm}-{filename[:2]}"),
                max(1.0, deadline - time.perf_counter()))
            rows.append(json.loads(worker_output.read_text(encoding="utf-8")))
            processes.append(process)

    baseline_rows = [rows[0], rows[3]]
    probe_rows = [rows[1], rows[2]]
    output_names = ("prediction", "lower", "upper")
    expected_batches = [16] * 45 + [5]
    ordered_hash_gates = {
        f"fixed725_{name}_four_block_bitwise":
            len({row["ordered_output_sha256"][name] for row in rows}) == 1
        for name in output_names
    }
    remainder_hash_gates = {
        f"remainder17_{name}_four_block_bitwise":
            len({row["remainder17_output_sha256"][name] for row in rows}) == 1
        for name in output_names
    }
    predict_ratio = (
        max(row["predict_seconds_total"] for row in probe_rows)
        / min(row["predict_seconds_total"] for row in baseline_rows))
    total_ratio = (
        max(row["import_plus_predict_seconds"] for row in probe_rows)
        / min(row["import_plus_predict_seconds"] for row in baseline_rows))
    max_probe_import = max(row["import_seconds"] for row in probe_rows)
    max_reserved = max(row["peak_cuda_reserved_bytes"] for row in probe_rows)
    rss_values = [row["peak_rss_bytes"] for row in probe_rows]
    max_rss = max(rss_values) if all(value is not None for value in rss_values) else None
    gates = {
        **ordered_hash_gates,
        **remainder_hash_gates,
        "fixed725_shape_and_calls": all(
            row["samples"] == EXPECTED_SAMPLES
            and row["predict_calls"] == EXPECTED_CALLS
            and row["call_batch_sizes"] == expected_batches
            for row in rows),
        "runtime_contract": all(
            row["batch_size"] == 16
            and row["device_type"] == "cuda"
            and row["dtype"] == "torch.float16"
            and row["input_bitwise_unchanged"] is True
            and row["finite_ordered_pressure_zero"] is True
            for row in rows),
        "baseline_lazy_on_import": all(
            row["import_state_materialized"] is False for row in baseline_rows),
        "probe_materialized_on_import": all(
            row["import_state_materialized"] is True for row in probe_rows),
        "adverse_predict_ratio_le_0p75": predict_ratio <= MAX_PREDICT_RATIO,
        "adverse_total_ratio_le_1p10": total_ratio <= MAX_TOTAL_RATIO,
        "probe_import_le_15s": max_probe_import <= MAX_IMPORT_SECONDS,
        "probe_peak_reserved_lt_14gib": (
            max_reserved < MAX_CUDA_RESERVED_BYTES),
        "probe_peak_rss_lt_16gib": (
            max_rss is not None and max_rss < MAX_RSS_BYTES),
        "wall_budget_le_20min": time.perf_counter() <= deadline,
    }
    return {
        "schema": SCHEMA,
        "evidence_class": "g3d_actual_zip_fixed725_deployment_qualification",
        "git": identity,
        "baseline_archive": baseline_audit,
        "probe_archive": probe_audit,
        "fixed725_input_cache": cache_identity,
        "sequence": [arm for arm, _filename in sequence],
        "worker_files": [file_identity(args.output.parent / filename)
                         for _arm, filename in sequence],
        "worker_processes": processes,
        "workers": rows,
        "ratios": {
            "adverse_probe_over_baseline_predict": predict_ratio,
            "adverse_probe_over_baseline_import_plus_predict": total_ratio,
            "max_probe_import_seconds": max_probe_import,
        },
        "memory": {
            "max_probe_cuda_reserved_bytes": max_reserved,
            "max_probe_rss_bytes": max_rss,
        },
        "gates": gates,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
        "release_classified": False,
        "decision": (
            "PASS_G3D_IMPORT_PREWARM_PROBE"
            if all(gates.values()) else "STOP_G3D_IMPORT_PREWARM_PROBE"),
    }


def require(value, label: str) -> None:
    if value is None:
        raise ValueError(f"missing required argument: {label}")


def validate_args(args: argparse.Namespace) -> None:
    if args.static_cpu:
        for value, label in (
            (args.baseline_archive, "baseline-archive"),
            (args.probe_archive, "probe-archive"),
            (args.probe_archive_sha256, "probe-archive-sha256"),
            (args.output, "output")):
            require(value, label)
    elif args.verify_gpu:
        for value, label in (
            (args.baseline_archive, "baseline-archive"),
            (args.probe_archive, "probe-archive"),
            (args.probe_archive_sha256, "probe-archive-sha256"),
            (args.output, "output")):
            require(value, label)
        if (not args.execute_formal_gpu
                or args.acknowledge_protocol != FORMAL_ACKNOWLEDGEMENT):
            raise ValueError("formal GPU verification lacks acknowledgement")
    elif args.worker_cpu:
        for value, label in (
            (args.archive, "archive"),
            (args.archive_sha256, "archive-sha256"),
            (args.worker_output, "worker-output")):
            require(value, label)
    else:
        for value, label in (
            (args.archive, "archive"),
            (args.archive_sha256, "archive-sha256"),
            (args.arm, "arm"),
            (args.input_cache, "input-cache"),
            (args.worker_output, "worker-output")):
            require(value, label)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.worker_cpu:
        result = run_cpu_worker(args)
    elif args.worker_gpu:
        result = run_gpu_worker(args)
    elif args.static_cpu:
        result = run_static_cpu(args)
        write_json(args.output, result)
    else:
        result = run_gpu_verify(args)
        write_json(args.output, result)
    print(json.dumps({
        "event": "b210_g3d_probe_verify_complete",
        "decision": result["decision"],
        "output": (None if args.output is None else str(args.output.resolve())),
    }), flush=True)


if __name__ == "__main__":
    main()
