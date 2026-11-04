import json
from pathlib import Path
import sys

import pytest


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import summarize_b210_abba_latency as subject  # noqa: E402


def block(archive_sha: str, median_ms: float, startup: float = 2.0) -> dict:
    return {
        "evidence_class": "s2_deployment_archive_latency",
        "archive": {"sha256": archive_sha, "bytes": 100_000_000},
        "example_h5": {"sha256": "input-sha"},
        "batch_size": 16,
        "warmup_repeats": 0,
        "repeats": 50,
        "device": "cuda:0",
        "gpu": "RTX 4080 SUPER",
        "torch_version": "2.x",
        "execution_contract": {
            "execution_mode": "eager",
            "torch_compile_requested": False,
            "channels_last_requested": False,
            "cudnn_benchmark": False,
            "input_c_contiguous": True,
            "input_pressure_zero": True,
            "input_bitwise_unchanged": True,
        },
        "outputs": {
            "prediction": {"finite": True},
            "lower": {"finite": True},
            "upper": {"finite": True},
            "bounds_ordered": True,
        },
        "benchmark_source": {"sha256": "benchmark-sha"},
        "startup_to_first_output_seconds": startup,
        "end_to_end": {"median_ms": median_ms},
        "peak_cuda_reserved_bytes": 5_000_000_000,
    }


def write_blocks(tmp_path, medians=(100.0, 102.0, 103.0, 101.0)):
    c208_sha = "a" * 64
    b210_sha = "b" * 64
    paths = []
    for index, (sha, median) in enumerate(zip(
            (c208_sha, b210_sha, b210_sha, c208_sha), medians)):
        path = tmp_path / f"block{index}.json"
        path.write_text(json.dumps(block(sha, median)), encoding="utf-8")
        paths.append(path)
    return paths, c208_sha, b210_sha


def test_passes_frozen_abba_gates(tmp_path):
    paths, c208_sha, b210_sha = write_blocks(tmp_path)
    result = subject.summarize(paths, c208_sha, b210_sha)
    assert result["decision"] == "PASS_B210_ABBA_LATENCY"
    assert result["ratios"]["pooled_b210_over_c208"] == pytest.approx(
        102.5 / 100.5)


def test_stops_adverse_ratio_even_when_pooled_ratio_passes(tmp_path):
    paths, c208_sha, b210_sha = write_blocks(
        tmp_path, medians=(90.0, 99.5, 90.0, 100.0))
    result = subject.summarize(paths, c208_sha, b210_sha)
    assert result["gates"]["adverse_b210_over_c208_le_1p10"] is False
    assert result["decision"] == "STOP_B210_ABBA_LATENCY"


def test_rejects_execution_contract_drift(tmp_path):
    paths, c208_sha, b210_sha = write_blocks(tmp_path)
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["execution_contract"]["cudnn_benchmark"] = True
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="execution contract"):
        subject.summarize(paths, c208_sha, b210_sha)


def test_rejects_archive_sha_drift(tmp_path):
    paths, c208_sha, b210_sha = write_blocks(tmp_path)
    payload = json.loads(paths[2].read_text(encoding="utf-8"))
    payload["archive"]["sha256"] = "c" * 64
    paths[2].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="archive SHA-256"):
        subject.summarize(paths, c208_sha, b210_sha)
