from __future__ import annotations

import io
import importlib.util
import zipfile
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "build_track1_point_uq_hybrid_archive.py"
)
SPEC = importlib.util.spec_from_file_location("point_uq_hybrid_builder", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def payload(point: float, uq: float) -> dict:
    return {
        "phase": "real",
        "model_config": {"architecture": "ensemble_unet", "base": 2},
        "quantized_state_dict": {
            "stem.weight": torch.tensor([point]),
            "member_point_heads.0.weight": torch.tensor([point + 1]),
            "uq_fusion.0.weight": torch.tensor([uq]),
            "temporal_uq_refiner.output.weight": torch.tensor([uq + 1]),
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
        subject.CHECKPOINT_MEMBER: buffer.getvalue(),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name in subject.EXPECTED_MEMBERS:
            archive.writestr(name, contents[name])


def read_payload(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return torch.load(
            io.BytesIO(archive.read(subject.CHECKPOINT_MEMBER)),
            map_location="cpu",
            weights_only=False,
        )


def test_hybrid_keeps_point_and_uq_responsibilities_exact(tmp_path: Path) -> None:
    point = tmp_path / "point.zip"
    uq = tmp_path / "uq.zip"
    runtime = tmp_path / "runtime.zip"
    output = tmp_path / "hybrid.zip"
    report = tmp_path / "hybrid.json"
    write_archive(point, payload(1.0, 10.0), b"point")
    write_archive(uq, payload(2.0, 20.0), b"uq")
    write_archive(runtime, payload(3.0, 30.0), b"runtime")

    result = subject.build_archive(
        point_archive=point,
        uq_archive=uq,
        runtime_archive=runtime,
        output=output,
        report=report,
        max_size_mib=1.0,
    )
    stored = read_payload(output)["quantized_state_dict"]
    assert torch.equal(stored["stem.weight"], torch.tensor([1.0]))
    assert torch.equal(
        stored["member_point_heads.0.weight"], torch.tensor([2.0])
    )
    assert torch.equal(stored["uq_fusion.0.weight"], torch.tensor([20.0]))
    assert torch.equal(
        stored["temporal_uq_refiner.output.weight"], torch.tensor([21.0])
    )
    with zipfile.ZipFile(output) as archive:
        assert archive.read("submission.py") == b"runtime"
    assert result["audit"]["point_source_exact"] is True
    assert result["audit"]["uq_source_exact"] is True
    assert result["audit"]["point_keys"] == 2
    assert result["audit"]["uq_keys"] == 2
    assert report.is_file()


def test_hybrid_rejects_model_config_mismatch() -> None:
    point = payload(1.0, 10.0)
    uq = payload(2.0, 20.0)
    uq["model_config"]["base"] = 3
    try:
        subject.build_hybrid_payload(point, uq)
    except ValueError as error:
        assert "configurations differ" in str(error)
    else:
        raise AssertionError("configuration mismatch was accepted")


def test_hybrid_accepts_only_known_explicit_defaults() -> None:
    point = payload(1.0, 10.0)
    uq = payload(2.0, 20.0)
    uq["model_config"].update(subject.COMPATIBLE_EXPLICIT_DEFAULTS)
    hybrid, _ = subject.build_hybrid_payload(point, uq)
    for key, value in subject.COMPATIBLE_EXPLICIT_DEFAULTS.items():
        assert hybrid["model_config"][key] == value
