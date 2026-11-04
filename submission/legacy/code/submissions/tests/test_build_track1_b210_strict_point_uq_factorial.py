from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import zipfile

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "build_track1_b210_strict_point_uq_factorial.py"
)
SPEC = importlib.util.spec_from_file_location("strict_factorial_builder", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def payload(point: float, uq: float) -> dict:
    return {
        "phase": "real",
        "model_config": {"architecture": "ensemble_unet", "base": 2},
        "quantized_state_dict": {
            "stem.weight": torch.tensor([point], dtype=torch.float16),
            "member_point_heads.0.weight": torch.tensor(
                [point + 1], dtype=torch.float16
            ),
            "uq_fusion.0.weight": torch.tensor([uq], dtype=torch.float16),
            "temporal_uq_refiner.output.weight": torch.tensor(
                [uq + 1], dtype=torch.float16
            ),
        },
        "normalizer": {"mean_input": torch.zeros(1)},
        "inference_dtype": "float16",
        "storage_format": "symmetric_per_dim0_int8_v1",
    }


def write_archive(path: Path, value: dict, marker: bytes) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    contents = {
        "submission.py": marker,
        "models/__init__.py": b"",
        "models/unet2d.py": marker,
        "models/ensemble_unet2d.py": marker,
        subject.hybrid.CHECKPOINT_MEMBER: buffer.getvalue(),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name in subject.hybrid.EXPECTED_MEMBERS:
            archive.writestr(name, contents[name])


def read_payload(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return torch.load(
            io.BytesIO(archive.read(subject.hybrid.CHECKPOINT_MEMBER)),
            map_location="cpu",
            weights_only=False,
        )


def test_builds_actual_four_quadrants_with_one_runtime(tmp_path: Path) -> None:
    old = tmp_path / "old.zip"
    new = tmp_path / "new.zip"
    runtime = tmp_path / "runtime.zip"
    write_archive(old, payload(1.0, 10.0), b"old")
    write_archive(new, payload(2.0, 20.0), b"new")
    write_archive(runtime, payload(3.0, 30.0), b"runtime")

    result = subject.build_factorial(
        old_archive=old,
        new_archive=new,
        runtime_archive=runtime,
        output_dir=tmp_path / "arms",
        report=tmp_path / "report.json",
        max_size_mib=1.0,
    )
    expected = {
        "AA": (1.0, 10.0),
        "DA": (2.0, 10.0),
        "AD": (1.0, 20.0),
        "DD": (2.0, 20.0),
    }
    for arm, (point, uq) in expected.items():
        path = Path(result["arms"][arm]["archive"]["path"])
        stored = read_payload(path)["quantized_state_dict"]
        assert float(stored["stem.weight"]) == point
        assert float(stored["uq_fusion.0.weight"]) == uq
        with zipfile.ZipFile(path) as archive:
            assert archive.read("submission.py") == b"runtime"
    assert result["factorial_audit"]["strict_factorial"] is True
    assert result["serialized_archive_audit"]["strict_factorial"] is True
    assert result["old_vs_new_state"]["summary"]["point"]["changed_keys"] == 2
    assert result["old_vs_new_state"]["summary"]["uq"]["changed_keys"] == 2


def test_rejects_renamed_whole_archives() -> None:
    old = payload(1.0, 10.0)
    new = payload(2.0, 20.0)
    audit = subject.audit_factorial_payloads(
        {"AA": old, "DD": old, "DA": new, "AD": new}
    )
    assert audit["strict_factorial"] is False
    assert audit["identities"]["AA_point_equals_AD"] is False
    assert audit["identities"]["DA_point_equals_DD"] is False
