#!/usr/bin/env python3
"""Build the official and TKE-weighted Track 1 v9 FNO submission archives."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=Path("models/sim_real_ft/sim_real_fno_fp16.pth"))
    parser.add_argument(
        "--candidate-checkpoint", type=Path,
        default=Path("runs/checkpoints/fno_tke4_1000_best_fp16.pth"))
    parser.add_argument(
        "--candidate-sps-wrapper", type=Path,
        default=Path("code/submissions/track1_fno_tke4_sps_calibrated.py"))
    parser.add_argument(
        "--dynamic-sps-wrapper", type=Path,
        default=Path("code/submissions/track1_fno_tke4_sps_dynamic.py"))
    parser.add_argument(
        "--dynamic-head", type=Path,
        help="Optional trained interval-head checkpoint to add as interval_head.pth.")
    parser.add_argument(
        "--structured-sps-wrapper", type=Path,
        default=Path("code/submissions/track1_fno_tke4_sps_structured.py"))
    parser.add_argument(
        "--structured-head", type=Path,
        help="Optional structured interval-head checkpoint to package.")
    parser.add_argument(
        "--structured-output-name",
        default="track1_v9_tke4_fno_fp16_sps_structured.zip",
        help="Archive filename for --structured-head (basename ending in .zip).")
    parser.add_argument(
        "--corrected-sps-wrapper", type=Path,
        default=Path("code/submissions/track1_fno_corrected_sps_structured.py"))
    parser.add_argument(
        "--correction-head", type=Path,
        help="Optional conditioned point-correction checkpoint to package.")
    parser.add_argument(
        "--corrected-interval-head", type=Path,
        help="Structured interval checkpoint trained on corrected predictions.")
    parser.add_argument(
        "--corrected-output-name",
        default="track1_v9_tke4_fno_fp16_corrected_sps_structured.zip",
        help="Archive filename for the corrected-point submission.")
    parser.add_argument(
        "--only-structured", action="store_true",
        help="Skip rebuilding the unchanged official, point, and fixed-bound archives.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/submissions"))
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_entries(kit_dir: Path, checkpoint: Path, wrapper: Path,
                   extra_files: tuple[tuple[Path, Path], ...] = ()):
    yield wrapper, Path("submission.py")
    yield kit_dir / "load_baseline.py", Path("load_baseline.py")
    yield checkpoint, Path("sim_real_fno_fp16.pth")
    yield from extra_files
    for directory in ("rpde_baselines", "_vendor"):
        root = kit_dir / directory
        for path in sorted(root.rglob("*")):
            if (path.is_file() and "__pycache__" not in path.parts
                    and path.suffix not in {".pyc", ".pyo"}):
                yield path, Path(directory) / path.relative_to(root)


def build_archive(output: Path, kit_dir: Path, checkpoint: Path, wrapper: Path,
                  max_size_mib: float,
                  extra_files: tuple[tuple[Path, Path], ...] = ()) -> dict[str, object]:
    required = [
        wrapper, kit_dir / "load_baseline.py",
        kit_dir / "rpde_baselines", kit_dir / "_vendor", checkpoint,
    ]
    required.extend(source for source, _archive_path in extra_files)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    names: set[str] = set()
    try:
        with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=6, allowZip64=True) as archive:
            for source, archive_path in source_entries(
                    kit_dir, checkpoint, wrapper, extra_files):
                name = archive_path.as_posix()
                if name in names:
                    raise RuntimeError(f"Duplicate archive path: {name}")
                names.add(name)
                archive.write(source, name)
        size = temporary.stat().st_size
        if size > max_size_mib * 1024 * 1024:
            raise RuntimeError(
                f"{temporary} is {size / 1024**2:.2f} MiB, above the "
                f"{max_size_mib:.2f} MiB limit")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        root_names = set(archive.namelist())
    if bad_member is not None:
        raise RuntimeError(f"Corrupt member in {output}: {bad_member}")
    required_root = {"submission.py", "load_baseline.py", "sim_real_fno_fp16.pth"}
    required_root.update(archive_path.as_posix()
                         for _source, archive_path in extra_files)
    if not required_root.issubset(root_names):
        raise RuntimeError(f"Required root files missing from {output}")
    return {
        "archive": str(output),
        "checkpoint": str(checkpoint),
        "wrapper": str(wrapper),
        "checkpoint_sha256": sha256(checkpoint),
        "archive_sha256": sha256(output),
        "size_mib": output.stat().st_size / 1024**2,
        "members": len(root_names),
    }


def main() -> None:
    args = parse_args()
    for flag, name in (
            ("--structured-output-name", args.structured_output_name),
            ("--corrected-output-name", args.corrected_output_name)):
        if Path(name).name != name or not name.endswith(".zip"):
            raise ValueError(f"{flag} must be a .zip basename")
    if (args.correction_head is None) != (args.corrected_interval_head is None):
        raise ValueError(
            "--correction-head and --corrected-interval-head must be supplied together")
    kit_dir = args.kit_dir.resolve()
    output_dir = args.output_dir.resolve()
    official_wrapper = kit_dir / "submission_example_fno.py"
    specs = [] if args.only_structured else [
        ("track1_v9_official_fno_fp16.zip", args.official_checkpoint.resolve(),
         official_wrapper, ()),
        ("track1_v9_tke4_fno_fp16.zip", args.candidate_checkpoint.resolve(),
         official_wrapper, ()),
        ("track1_v9_tke4_fno_fp16_sps_robust.zip", args.candidate_checkpoint.resolve(),
         args.candidate_sps_wrapper.resolve(), ()),
    ]
    if (
        args.only_structured
        and args.structured_head is None
        and args.correction_head is None
    ):
        raise ValueError(
            "--only-structured requires --structured-head or the corrected-head pair"
        )
    if args.dynamic_head is not None:
        specs.append((
            "track1_v9_tke4_fno_fp16_sps_dynamic.zip",
            args.candidate_checkpoint.resolve(),
            args.dynamic_sps_wrapper.resolve(),
            (
                (args.dynamic_head.resolve(), Path("interval_head.pth")),
                (REPO_ROOT / "code/uncertainty/fno_interval_head.py",
                 Path("fno_interval_head.py")),
            ),
        ))
    if args.structured_head is not None:
        specs.append((
            args.structured_output_name,
            args.candidate_checkpoint.resolve(),
            args.structured_sps_wrapper.resolve(),
            (
                (args.structured_head.resolve(), Path("interval_head.pth")),
                (REPO_ROOT / "code/uncertainty/structured_interval_head.py",
                 Path("structured_interval_head.py")),
            ),
        ))
    if args.correction_head is not None:
        specs.append((
            args.corrected_output_name,
            args.candidate_checkpoint.resolve(),
            args.corrected_sps_wrapper.resolve(),
            (
                (args.correction_head.resolve(), Path("correction_head.pth")),
                (args.corrected_interval_head.resolve(), Path("interval_head.pth")),
                (REPO_ROOT / "code/uncertainty/conditioned_residual_corrector.py",
                 Path("conditioned_residual_corrector.py")),
                (REPO_ROOT / "code/uncertainty/structured_interval_head.py",
                 Path("structured_interval_head.py")),
            ),
        ))
    for filename, checkpoint, wrapper, extra_files in specs:
        print(build_archive(
            output_dir / filename, kit_dir, checkpoint, wrapper,
            args.max_size_mib, extra_files))


if __name__ == "__main__":
    main()
