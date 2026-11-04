from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = (
    ROOT / "code/submissions/build_track1_b210_import_prewarm_probe.py")
SPEC = importlib.util.spec_from_file_location("g3d_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_tracked_wrapper_has_frozen_import_prewarm_contract():
    wrapper = ROOT / "code/submissions/track1_ensemble_unet2d.py"
    builder.validate_wrapper_source(wrapper.read_bytes())
    source = wrapper.read_text(encoding="utf-8")
    assert "torch.compile" not in source
    assert "_BATCH_SIZE = 16" in source


def test_builder_changes_only_submission_member(
        tmp_path: Path, monkeypatch):
    baseline = tmp_path / "baseline.zip"
    payloads = {
        name: f"baseline:{name}".encode("utf-8")
        for name in builder.EXPECTED_MEMBERS
    }
    with zipfile.ZipFile(baseline, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in builder.EXPECTED_MEMBERS:
            archive.writestr(name, payloads[name])
    ledger = {name: _sha(payload) for name, payload in payloads.items()}
    monkeypatch.setattr(builder, "BASELINE_MEMBER_SHA256", ledger)
    output = tmp_path / "probe.zip"
    wrapper = ROOT / "code/submissions/track1_ensemble_unet2d.py"
    result = builder.build_probe(
        baseline, wrapper, output,
        expected_baseline_sha256=builder.sha256(baseline))
    assert result["decision"] == "PASS_G3D_PROBE_BUILD"
    assert result["only_submission_member_changed"] is True
    assert all(result["non_wrapper_members_bitwise_identical"].values())
    with zipfile.ZipFile(output) as archive:
        assert archive.read("submission.py") == wrapper.read_bytes()
        for name in builder.EXPECTED_MEMBERS[1:]:
            assert archive.read(name) == payloads[name]
