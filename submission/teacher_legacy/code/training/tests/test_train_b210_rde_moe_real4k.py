from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_rde_moe_real4k as subject  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
from models.unet2d import history_regime_summary  # noqa: E402


def tiny_config(**overrides) -> EnsembleUNet2dConfig:
    values = {
        "base": 8,
        "depth": 3,
        "groups": 4,
        "difference_steps": 2,
        "ensemble_members": 3,
        "uq_hidden_channels": 8,
        "mean_profile_channels": 8,
        "multi_scale_regime_film": True,
        "residual_quantile_intervals": True,
        "physical_energy_controller": True,
        "physical_energy_member_sync": True,
    }
    values.update(overrides)
    return EnsembleUNet2dConfig(**values)


def test_frozen_protocol_has_no_effect_gate_and_uses_native_joint_loss():
    assert subject.ARM_ORDER == (
        "r0_single", "r1_fixed_average", "r2_history_router")
    assert subject.UPDATES == 4_000
    assert subject.SCHEDULER_T_MAX == 12_000
    assert subject.EMA_DECAY == pytest.approx(0.999)
    assert subject.ROUTER_UPDATES == 400
    assert subject.ROUTER_SEED == 20_260_904
    assert subject.ROUTER_POSITIVE_AOAS == (0, 20)
    assert subject.ROUTER_NEGATIVE_AOAS == (5, 10, 15)
    assert subject.ROUTER_POS_WEIGHT == pytest.approx(6065 / 3868)
    source = inspect.getsource(subject.run)
    assert "score_gate" in source
    assert "RDE_MOE_REAL4K_COMPLETE_NO_SCORE_GATE" in source
    train_source = inspect.getsource(subject.train_arm)
    assert "small4k.train_candidate" in train_source
    assert "input_augmentation=dense4k.AUGMENTATION" in train_source


def test_real_router_descriptor_matches_deployed_uvp_normalization(
        monkeypatch):
    window_type = subject.ft12.npt1.ensemble_train.common.repro.Window
    trajectories = []
    metadata = []
    windows = []
    aoas = (0, 5, 10, 15, 20)
    for trajectory, aoa in enumerate(aoas):
        value = torch.arange(40 * 2 * 3 * 2, dtype=torch.float32).reshape(
            40, 2, 3, 2)
        trajectories.append(value + trajectory)
        metadata.append({"aoa": aoa})
        windows.append(window_type(trajectory, 0))
    normalizer = SimpleNamespace(
        mean_input=torch.tensor([2.0, -1.0, 0.0]),
        std_input=torch.tensor([3.0, 4.0, 1.0]),
    )
    uvp = subject.real_history_uvp(trajectories, windows)
    assert uvp.shape == (5, 20, 2, 3, 3)
    assert torch.count_nonzero(uvp[..., 2]) == 0
    expected = history_regime_summary(
        (uvp - normalizer.mean_input) / normalizer.std_input)
    actual, labels, observed_aoas = subject.prefix_router_tensors_general(
        trajectories, windows, metadata, normalizer)
    assert torch.equal(actual, expected)
    assert labels.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert observed_aoas.tolist() == list(aoas)

    monkeypatch.setattr(subject, "EXPECTED_DENSE_WINDOWS", 5)
    monkeypatch.setattr(subject, "EXPECTED_ROUTER_POSITIVES", 2)
    monkeypatch.setattr(subject, "EXPECTED_ROUTER_NEGATIVES", 3)
    strict = subject.prefix_router_tensors(
        trajectories, windows, metadata, normalizer)
    assert all(torch.equal(left, right)
               for left, right in zip(strict, (actual, labels, observed_aoas)))


def test_router_install_covers_weights_standardization_and_freezes():
    torch.manual_seed(91)
    models = [EnsembleUNet2d(tiny_config(dual_decoder_mode=mode))
              for mode in subject.REGIME_EXPERT_DECODER_MODES]
    source = models[0]
    with torch.no_grad():
        for parameter in source.regime_expert_router.parameters():
            parameter.copy_(torch.randn_like(parameter))
        source.set_regime_expert_router_standardization(
            torch.randn(80), torch.rand(80) + 0.1)
    package = {
        "schema": subject.ROUTER_SCHEMA,
        "router_state_dict": {
            name: value.detach().clone()
            for name, value in source.regime_expert_router.state_dict().items()
        },
        "descriptor_mean": source.regime_expert_router_mean.detach().clone(),
        "descriptor_std": source.regime_expert_router_std.detach().clone(),
        "router_state_sha256": subject.router_state_sha256(source),
    }
    for model in models:
        subject.install_router(model, package)
        assert subject.router_state_sha256(model) == package[
            "router_state_sha256"]
        assert not any(parameter.requires_grad
                       for parameter in model.regime_expert_router.parameters())
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    for name, value in models[0].state_dict().items():
        assert value.shape == models[1].state_dict()[name].shape


def frozen_router_fixture():
    torch.manual_seed(912)
    model = EnsembleUNet2d(tiny_config(
        dual_decoder_mode="regime_expert_history_router"))
    with torch.no_grad():
        for parameter in model.regime_expert_router.parameters():
            parameter.copy_(torch.randn_like(parameter))
        model.set_regime_expert_router_standardization(
            torch.randn(80), torch.rand(80) + 0.1)
    package = {
        "schema": subject.ROUTER_SCHEMA,
        "router_state_dict": {
            name: value.detach().clone()
            for name, value in model.regime_expert_router.state_dict().items()
        },
        "descriptor_mean": model.regime_expert_router_mean.detach().clone(),
        "descriptor_std": model.regime_expert_router_std.detach().clone(),
    }
    raw = subject.packaged_router_state(package)
    ema = subject.deterministic_frozen_router_ema_state(package)
    endpoints = {
        arm: {
            "raw": {name: value.clone() for name, value in raw.items()},
            "ema": {name: value.clone() for name, value in ema.items()},
        }
        for arm in ("r1_fixed_average", "r2_history_router")
    }
    return package, raw, ema, endpoints


def test_frozen_router_ema_integrity_replays_exact_arithmetic():
    package, raw, ema, endpoints = frozen_router_fixture()
    report = subject.audit_frozen_router_endpoints(package, endpoints)
    assert report["raw_equals_router_package_bitwise"] is True
    assert report["r1_r2_ema_equal_bitwise"] is True
    assert report[
        "ema_equals_constant_source_fp32_replay_bitwise"] is True
    assert report["ema_replay"]["updates"] == subject.UPDATES
    assert report["ema_replay"]["operation"].startswith("shadow.mul_")
    for name in ("regime_expert_router_mean", "regime_expert_router_std"):
        assert torch.equal(raw[name], ema[name])


@pytest.mark.parametrize("kind", ("raw", "ema"))
def test_frozen_router_integrity_rejects_single_ulp_mutation(kind):
    package, _raw, _ema, endpoints = frozen_router_fixture()
    value = endpoints["r2_history_router"][kind][
        "regime_expert_router.0.weight"]
    value.view(-1)[0] = torch.nextafter(
        value.view(-1)[0], torch.tensor(float("inf")))
    with pytest.raises(RuntimeError, match="bitwise mismatch"):
        subject.audit_frozen_router_endpoints(package, endpoints)


def test_frozen_router_integrity_rejects_descriptor_buffer_mutation():
    package, _raw, _ema, endpoints = frozen_router_fixture()
    endpoints["r1_fixed_average"]["ema"][
        "regime_expert_router_std"][0] += 1.0e-6
    with pytest.raises(RuntimeError, match="bitwise mismatch"):
        subject.audit_frozen_router_endpoints(package, endpoints)


def test_frozen_router_is_removed_from_optimizer_without_changing_model(
        monkeypatch):
    model = EnsembleUNet2d(tiny_config(
        dual_decoder_mode="regime_expert_history_router"))
    total = sum(parameter.numel() for parameter in model.parameters())
    router = sum(parameter.numel()
                 for parameter in model.regime_expert_router.parameters())
    monkeypatch.setattr(subject, "expected_parameters", lambda _arm: total)
    with subject.frozen_router_trainability("r2_history_router"):
        trainable = subject.small4k.ft12.configure_trainable(model, total)
    assert sum(parameter.numel() for parameter in trainable) == total - router
    assert not any(parameter.requires_grad
                   for parameter in model.regime_expert_router.parameters())


def test_r1_r2_configs_differ_only_in_feature_mix_rule():
    base = tiny_config()
    r1 = subject.arm_config(base, "r1_fixed_average")
    r2 = subject.arm_config(base, "r2_history_router")
    difference = {
        key for key in r1.__dict__
        if r1.__dict__[key] != r2.__dict__[key]
    }
    assert difference == {"dual_decoder_mode"}
    assert replace(r1, dual_decoder_mode="none") == replace(
        r2, dual_decoder_mode="none") == base


def test_tail_audit_requires_all_endpoint_seals_and_detects_change(tmp_path):
    registry = subject.EndpointRegistry()
    with pytest.raises(RuntimeError, match="all 6 raw/EMA endpoints"):
        subject.TailAudit(registry)
    for arm in subject.ARM_ORDER:
        for kind in subject.ENDPOINT_ORDER:
            path = tmp_path / f"{arm}_{kind}.pth"
            path.write_bytes(f"{arm}:{kind}".encode())
            registry.seal(arm, kind, {
                "path": str(path),
                "file_sha256": subject.dr1.sha256(path),
            })
    audit = subject.TailAudit(registry)
    row = {"file": "held.h5"}
    audit.before_open(row, "held")
    audit.record_open(row, "held")
    audit.after_load()
    changed = tmp_path / "r2_history_router_ema.pth"
    changed.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        audit.after_load()


def test_fixed_clean716_hash_is_checked_only_after_all_endpoint_seals():
    assert subject.EXPECTED_EVALUATION_MANIFEST_SHA256 == (
        "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
    source = inspect.getsource(subject.run)
    complete = source.index("if not registry.complete()")
    tail_audit = source.index("tail_audit = TailAudit(registry)")
    manifest_hash = source.index(
        "evaluation_manifest_sha256 = dr1.sha256(args.evaluation_manifest)")
    manifest_load = source.index("evaluation_rows = dense4k.load_evaluation_rows")
    assert complete < tail_audit < manifest_hash < manifest_load
    # The pre-seal operation is deliberately only a filesystem existence
    # check; it must not read or hash the clean716 definition.
    assert source.index("if not args.evaluation_manifest.is_file()") < complete


def test_metric_comparison_uses_published_final_coefficients():
    reference = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 91.0,
        "native_sps_score": 40.0,
        "native_sps_coverage": 0.8,
        "native_sps_mean_width": 0.1,
    }
    candidate = dict(reference)
    candidate.update({
        "rel_l2_score": 91.0,
        "tke_rel_l2_score": 82.0,
        "mvpe_rel_l2_score": 94.0,
        "native_sps_score": 44.0,
    })
    report = subject.metric_comparison(candidate, reference)
    assert report["official_point_contribution_delta"] == pytest.approx(
        0.46 * 1.0 + 0.10 * 2.0 + 0.09 * 3.0)
    assert report["official_non_time_contribution_delta"] == pytest.approx(
        0.46 + 0.20 + 0.27 + 0.25 * 4.0)
