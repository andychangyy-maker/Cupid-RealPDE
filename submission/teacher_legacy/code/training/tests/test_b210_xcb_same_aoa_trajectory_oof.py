from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))

import build_b210_xcb_same_aoa_point_oof_residual_cache as XCACHE  # noqa: E402
import train_b210_full_real_oof_native_uq as FULL_UQ  # noqa: E402
import train_b210_xcb_same_aoa_point_oof_fold as XPOINT  # noqa: E402
import train_b210_xcb_same_aoa_trajectory_native_uq as XUQ  # noqa: E402


def test_xcb_point_adapter_binds_exact_sim24k_topology():
    config, parameters, controller = XPOINT.architecture_contract()
    assert XPOINT.START_SHA256 == (
        "06ade419811d722e58235acdcb34e480dd6ae88555dd52037ebcb6bbeba2ce99")
    assert XPOINT.START_EXPECTED_ITERATION == 4_000
    assert XPOINT.START_EXPECTED_STATUS == (
        "PASS_B210_XCB_MEMBER_STANDARD24K_SIM")
    assert XPOINT.MODEL_LINEAGE == "b210_xcb_member_standard24k_sim"
    assert config.cross_conditioned_backbone is True
    assert config.member_adapter_channels == 64
    assert (parameters, controller) == (180_553_371, 7_169)
    assert XPOINT.LINEAGE_FIELD_REQUIRED is True


def test_xcb_point_adapter_scopes_and_restores_shared_runner():
    names = (
        "EXPERIMENT", "SCHEMA", "ENDPOINT_SCHEMA", "ACKNOWLEDGEMENT",
        "START_SHA256", "START_EXPECTED_ITERATION", "START_EXPECTED_STATUS",
        "MODEL_LINEAGE", "LINEAGE_FIELD_REQUIRED", "UQ_SOURCE_LABEL",
        "FINAL_POINT_RESULT_SCHEMA", "FINAL_POINT_ENDPOINT_SCHEMA",
        "FORMAL_FINAL_POINT_RESULT_SHA256", "architecture_contract")
    before = {name: getattr(XPOINT.base, name) for name in names}
    with XPOINT.configured_base():
        assert XPOINT.base.SCHEMA == XPOINT.SCHEMA
        assert XPOINT.base.START_SHA256 == XPOINT.START_SHA256
        assert XPOINT.base.architecture_contract() == (
            XPOINT.architecture_contract())
    assert {name: getattr(XPOINT.base, name) for name in names} == before


def test_xcb_cache_and_uq_adapters_only_replace_lineage_interfaces():
    cache_before = (XCACHE.base.point_fold, XCACHE.base.SCHEMA)
    with XCACHE.configured_base():
        assert XCACHE.base.point_fold is XPOINT
        assert XCACHE.base.SCHEMA == XCACHE.SCHEMA
        assert XCACHE.base._lineage_matches({
            "model_lineage": XPOINT.MODEL_LINEAGE}) is True
        assert XCACHE.base._lineage_matches({}) is False
    assert (XCACHE.base.point_fold, XCACHE.base.SCHEMA) == cache_before

    names = (
        "point_fold", "cache_builder", "EXPERIMENT", "FOLD_SCHEMA",
        "AGGREGATE_SCHEMA", "REFIT_SCHEMA", "REFIT_RESULT_SCHEMA",
        "ACKNOWLEDGEMENT", "FINAL_ENDPOINT")
    before = {name: getattr(XUQ.base, name) for name in names}
    with XUQ.configured_base():
        assert XUQ.base.point_fold is XPOINT
        assert XUQ.base.cache_builder is XCACHE
        assert XUQ.base.FOLD_SCHEMA == XUQ.FOLD_SCHEMA
        assert XUQ.base.FINAL_ENDPOINT == XUQ.FINAL_ENDPOINT
        assert XUQ.base._lineage_matches({
            "model_lineage": XPOINT.MODEL_LINEAGE}) is True
        assert XUQ.base._lineage_matches({}) is False
    assert {name: getattr(XUQ.base, name) for name in names} == before


def test_xcb_final_point_result_is_sha_and_schema_bound_when_available():
    path = XPOINT.FORMAL_FINAL_POINT_RESULT_PATH
    if not path.is_file():
        pytest.skip("formal XCB all81 point result is not on this host")
    assert XPOINT.dr1.sha256(path) == XPOINT.FORMAL_FINAL_POINT_RESULT_SHA256
    payload = json.loads(path.read_text(encoding="utf-8"))
    seal = payload["endpoints"]["ema_u4000"]["seal"]
    assert payload["schema"] == XPOINT.FINAL_POINT_RESULT_SCHEMA
    assert payload["start"]["checkpoint_sha256"] == XPOINT.START_SHA256
    assert payload["start"]["iteration"] == XPOINT.START_EXPECTED_ITERATION
    assert payload["start"]["status"] == XPOINT.START_EXPECTED_STATUS
    assert seal["schema"] == XPOINT.FINAL_POINT_ENDPOINT_SCHEMA
    assert seal["file_sha256"] == (
        "85f822f00a6bb16af68509cb8af1cc4477116fb0e7f37fc4fcedd5f04b3f98bf")


def test_general_reset_hook_accepts_sealed_sim24k_semantics(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.point = torch.nn.Linear(1, 1)
            self.uq_fusion = torch.nn.Linear(1, 1)

    model = TinyModel()
    source_model = TinyModel()
    with torch.no_grad():
        model.point.weight.fill_(5.0)
        model.uq_fusion.weight.fill_(7.0)
        source_model.point.weight.fill_(-1.0)
        source_model.uq_fusion.weight.fill_(3.0)
    source = tmp_path / "xcb_sim24k.pth"
    torch.save({
        "phase": "sim", "iteration": 4_000,
        "metrics": {"status": XPOINT.START_EXPECTED_STATUS},
        "model_config": {"architecture": "tiny"},
        "model_state_dict": source_model.state_dict(),
    }, source)
    digest = XPOINT.dr1.sha256(source)
    point_before = XPOINT.dr1.model_state_sha256(
        FULL_UQ.uq_release.partition_state(model, uq=False))
    report = FULL_UQ.reset_uq_from_pure_sim(
        model, source, digest, {"architecture": "tiny"},
        formal_start_sha256=digest, expected_iteration=4_000,
        expected_status=XPOINT.START_EXPECTED_STATUS,
        source_label="xcb_sim24k")
    assert report["policy"] == (
        "reset_all_existing_uq_only_tensors_from_xcb_sim24k")
    assert report["source_iteration"] == 4_000
    assert report["source_status"] == XPOINT.START_EXPECTED_STATUS
    assert report["point_partition_bitwise"] is True
    assert XPOINT.dr1.model_state_sha256(
        FULL_UQ.uq_release.partition_state(model, uq=False)) == point_before
    assert torch.equal(model.uq_fusion.weight, torch.full_like(
        model.uq_fusion.weight, 3.0))


def test_xcb_adapters_add_no_training_loop_or_forbidden_manifest_cli():
    for name in (
            "train_b210_xcb_same_aoa_point_oof_fold.py",
            "build_b210_xcb_same_aoa_point_oof_residual_cache.py",
            "train_b210_xcb_same_aoa_trajectory_native_uq.py"):
        source = (TRAINING / name).read_text(encoding="utf-8")
        assert 'add_argument("--clean716' not in source
        assert 'add_argument("--evaluation-manifest' not in source
        assert "DataLoader(" not in source
        assert "train_candidate(" not in source
    assert (XUQ.UPDATES_PER_FOLD, XUQ.CYCLES, XUQ.TOTAL_UPDATES) == (
        800, 4, 3_200)


def test_xcb_final_binding_rejects_any_other_result_sha(tmp_path):
    impostor = tmp_path / "point_stage_result.json"
    impostor.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside the formal lineage"):
        XUQ.load_final_point_artifact(
            impostor, XPOINT.dr1.sha256(impostor), torch.device("cpu"))
