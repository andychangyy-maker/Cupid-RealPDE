#!/usr/bin/env python3
"""Clone the exact 81.007 B210 archive with only its wrapper replaced."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ARCHIVE_SHA256 = (
    "b655aa406e9925786101e91cd49c08e1e60467f74d9262c65e6b6eaa76cb4e7c"
)
MAX_ARCHIVE_BYTES = 256 * 1024**2
EXPECTED_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    "ensemble_unet2d.pth",
)
BASELINE_MEMBER_SHA256 = {
    "submission.py": (
        "3b5297e671ca2cc87d5659f1fab72d1f1200439cf361d6a8373ab4e83da533e9"
    ),
    "models/__init__.py": (
        "ca44e67d82b6d00c74b8043fd918b91fb17e186b133e45361155e3a949ca3187"
    ),
    "models/unet2d.py": (
        "9b881f7b6b5c9dd42c8a28233028253c0e84bef3700862c9157a63cf22faabb6"
    ),
    "models/ensemble_unet2d.py": (
        "e655ce7e2ec79c3d5885fc4229c95ec98703283941deb647b0fe82b9aa13d850"
    ),
    "ensemble_unet2d.pth": (
        "95bb30d6cb89553012b93d9cd6cff0d4029418aff2f779032d2172ab59619ed5"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument(
        "--wrapper", type=Path,
        default=REPO_ROOT / "code/submissions/track1_ensemble_unet2d.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
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


def member_identity(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def validate_member_names(infos: list[zipfile.ZipInfo]) -> None:
    names = [info.filename for info in infos]
    if names != list(EXPECTED_MEMBERS) or len(names) != len(set(names)):
        raise ValueError(f"baseline archive layout differs: {names}")
    for info in infos:
        path = Path(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        if path.is_absolute() or ".." in path.parts or mode == 0o120000:
            raise ValueError(f"unsafe archive member: {info.filename}")


def validate_wrapper_source(payload: bytes) -> None:
    source = payload.decode("utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    prewarm = functions.get("_prewarm_cuda_runtime")
    if prewarm is None:
        raise ValueError("wrapper lacks _prewarm_cuda_runtime")
    call_names = set()
    for node in ast.walk(prewarm):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            call_names.add(target.id)
        elif isinstance(target, ast.Attribute):
            parts = [target.attr]
            value = target.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            call_names.add(".".join(reversed(parts)))
    required = {
        "torch.cuda.is_available",
        "_get_model",
        "np.zeros",
        "predict",
        "torch.cuda.synchronize",
    }
    if not required.issubset(call_names):
        raise ValueError(
            f"wrapper prewarm calls differ: missing {sorted(required - call_names)}"
        )
    if "torch.compile" in call_names or "compile" in call_names:
        raise ValueError("torch.compile is forbidden in the G3d wrapper")
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_BATCH_SIZE"
                for target in node.targets)
    ]
    if (len(assignments) != 1
            or not isinstance(assignments[0].value, ast.Constant)
            or assignments[0].value.value != 16):
        raise ValueError("G3d wrapper batch size is not frozen to 16")
    top_level_calls = [
        node for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_prewarm_cuda_runtime"
    ]
    if len(top_level_calls) != 1:
        raise ValueError("G3d wrapper must invoke import prewarm exactly once")


def build_probe(
    baseline: Path,
    wrapper: Path,
    output: Path,
    *,
    expected_baseline_sha256: str = BASELINE_ARCHIVE_SHA256,
) -> dict[str, object]:
    baseline = baseline.resolve()
    wrapper = wrapper.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not baseline.is_file() or not wrapper.is_file():
        raise FileNotFoundError("baseline archive or wrapper is missing")
    baseline_identity = file_identity(baseline)
    if baseline_identity["sha256"] != expected_baseline_sha256:
        raise ValueError("baseline archive SHA-256 differs from the 81.007 anchor")
    wrapper_payload = wrapper.read_bytes()
    validate_wrapper_source(wrapper_payload)

    with zipfile.ZipFile(baseline) as source:
        infos = source.infolist()
        validate_member_names(infos)
        baseline_payloads = {info.filename: source.read(info) for info in infos}
    baseline_members = {
        name: member_identity(payload) for name, payload in baseline_payloads.items()
    }
    observed_shas = {
        name: row["sha256"] for name, row in baseline_members.items()
    }
    if observed_shas != BASELINE_MEMBER_SHA256:
        raise ValueError("baseline member SHA-256 ledger differs")
    if sha256_bytes(wrapper_payload) == BASELINE_MEMBER_SHA256["submission.py"]:
        raise ValueError("probe wrapper is byte-identical to the baseline wrapper")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary archive {temporary}")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as target:
            for info in infos:
                cloned = copy.copy(info)
                payload = (
                    wrapper_payload
                    if info.filename == "submission.py"
                    else baseline_payloads[info.filename]
                )
                target.writestr(cloned, payload)
        if temporary.stat().st_size > MAX_ARCHIVE_BYTES:
            raise RuntimeError("probe archive exceeds 256 MiB")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as probe:
        if probe.testzip() is not None:
            raise RuntimeError("probe archive failed CRC validation")
        probe_infos = probe.infolist()
        validate_member_names(probe_infos)
        probe_payloads = {
            info.filename: probe.read(info) for info in probe_infos
        }
    probe_members = {
        name: member_identity(payload) for name, payload in probe_payloads.items()
    }
    unchanged = {
        name: probe_members[name] == baseline_members[name]
        for name in EXPECTED_MEMBERS if name != "submission.py"
    }
    if not all(unchanged.values()):
        raise RuntimeError(f"non-wrapper archive member changed: {unchanged}")
    if probe_payloads["submission.py"] != wrapper_payload:
        raise RuntimeError("packaged wrapper does not equal the source wrapper")

    return {
        "schema": "b210_g3d_import_prewarm_probe_build_v1",
        "decision": "PASS_G3D_PROBE_BUILD",
        "baseline_archive": baseline_identity,
        "probe_archive": file_identity(output),
        "wrapper_source": file_identity(wrapper),
        "baseline_members": baseline_members,
        "probe_members": probe_members,
        "non_wrapper_members_bitwise_identical": unchanged,
        "only_submission_member_changed": (
            all(unchanged.values())
            and probe_members["submission.py"] != baseline_members["submission.py"]
        ),
        "max_archive_bytes": MAX_ARCHIVE_BYTES,
        "submission_authorized": False,
    }


def main() -> None:
    args = parse_args()
    if args.result.exists():
        raise FileExistsError(f"refusing to overwrite {args.result}")
    result = build_probe(args.baseline_archive, args.wrapper, args.output)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "b210_g3d_probe_build_complete",
        "decision": result["decision"],
        "archive": result["probe_archive"],
        "result": file_identity(args.result),
    }), flush=True)


if __name__ == "__main__":
    main()
