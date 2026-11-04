from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import Dataset


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_c208_real_hgrad.py"
SPEC = importlib.util.spec_from_file_location("c208_real_hgrad", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def formal_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        start_sim=Path("sim10k.pth"),
        start_sim_sha256=subject.rehearsal.FORMAL_START_SIM_SHA256,
        output_dir=tmp_path / "run",
        updates=5000,
        real_dir=Path("real"),
        real_stats=Path("stats.pt"),
        identity_summary=Path("identity.json"),
        identity_summary_sha256=(
            subject.rehearsal.FORMAL_IDENTITY_SUMMARY_SHA256),
        device="cpu",
        num_workers=0,
    )


def test_artifact_hashes_and_single_configuration_are_frozen(tmp_path):
    args = formal_args(tmp_path)
    subject.validate_args(args)
    assert args.start_sim_sha256 == subject.rehearsal.FORMAL_START_SIM_SHA256
    args.updates = 4000
    with pytest.raises(ValueError, match="updates 5000"):
        subject.validate_args(args)
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("rehearsal.ensemble_train.evaluate(") == 1
    assert source.count('args.output_dir / "model_04000.pth"') == 1


class GradientDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        y = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 4)
        x = torch.arange(4, dtype=torch.float32).view(1, 4).expand(3, 4)
        target = torch.zeros(2, 3, 4, 3)
        target[..., 0] = x
        target[..., 1] = 2.0 * y
        return target.clone(), target, index, index


def test_train_only_rms_scale_contract_is_exact_and_canonical():
    torch.manual_seed(1234)
    state = torch.get_rng_state().clone()
    contract = subject.compute_hgrad_scale_contract(
        GradientDataset(), "split", "stats", expected_windows=2)
    assert torch.equal(torch.get_rng_state(), state)
    assert contract["train_windows"] == 2
    assert contract["validation_windows_used"] == 0
    assert contract["scales"]["dx"] == pytest.approx([1.0, 1e-6])
    assert contract["scales"]["dy"] == pytest.approx([1e-6, 2.0])
    payload = dict(contract)
    digest = payload.pop("canonical_sha256")
    assert digest == subject.rehearsal.canonical_sha256(payload)


def test_unpadded_uv_forward_differences_and_hgrad_loss():
    value = torch.arange(2 * 3 * 4 * 2, dtype=torch.float32).reshape(
        1, 2, 3, 4, 2)
    dx, dy = subject.forward_differences(value)
    assert dx.shape == (1, 2, 3, 3, 2)
    assert dy.shape == (1, 2, 2, 4, 2)
    normalizer = argparse.Namespace(
        mean_target=torch.zeros(3), std_target=torch.ones(3))
    target = torch.cat((value, torch.zeros(1, 2, 3, 4, 1)), dim=-1)
    scales = {"scales": {"dx": [1.0, 1.0], "dy": [1.0, 1.0]}}
    zero = subject.hgrad_loss(target, target, normalizer, scales)
    shifted = target.clone()
    shifted[:, :, :, 1:, 0] += 1.0
    assert zero.item() == pytest.approx(1e-3, abs=1e-8)
    assert subject.hgrad_loss(shifted, target, normalizer, scales) > 0


def test_lambda_is_audit_calibrated_linear_ramp_to_point_zero_zero_four():
    assert subject.hgrad_weight(0) == 0.0
    assert subject.hgrad_weight(50) == pytest.approx(0.001)
    assert subject.hgrad_weight(200) == pytest.approx(0.004)
    assert subject.hgrad_weight(4000) == pytest.approx(0.004)


def test_output_gradient_audit_is_exact_and_does_not_accumulate_grad():
    prediction = torch.ones(4, requires_grad=True)
    point = 0.5 * prediction.square().sum()
    weighted_hgrad = 0.025 * prediction.square().sum()
    audit = subject.output_gradient_audit(
        point, weighted_hgrad, point, prediction)
    assert audit["finite_nonzero"] is True
    assert audit["gradient_ratio"] == pytest.approx(0.05)
    assert audit["cosine"] == pytest.approx(1.0)
    assert prediction.grad is None
    (point + weighted_hgrad).backward()
    assert prediction.grad is not None


def passing_audits():
    return [{
        "iteration": update,
        "finite_nonzero": True,
        "loss_ratio": 0.02,
        "gradient_ratio": 0.05,
        "cosine": 0.1,
    } for update in subject.PREFLIGHT_AUDIT_UPDATES]


def test_preflight_requires_exact_four_finite_bounded_audits():
    passed = subject.evaluate_preflight(passing_audits())
    assert passed["pass"] is True
    assert passed["status"] == "GO_HGRAD_TO_UPDATE2000"
    stopped = subject.evaluate_preflight(passing_audits()[:-1])
    assert stopped["pass"] is False
    assert stopped["status"] == "STOP_HGRAD_PREFLIGHT"


def candidate_metrics(baseline: dict, rel=0.10, tke=-0.20, sps=0.10):
    result = dict(baseline)
    result.update({
        "rel_l2_score": baseline["rel_l2_score"] + rel,
        "tke_rel_l2_score": baseline["tke_rel_l2_score"] + tke,
        "mvpe_rel_l2_score": baseline["mvpe_rel_l2_score"],
        "point_score": baseline["point_score"] - 0.05,
        "native_sps_score": baseline["native_sps_score"] + sps,
    })
    return result


def test_step2000_gate_is_shadow_led_and_single_access(monkeypatch):
    baseline = dict(subject.rehearsal.FORMAL_IDENTITY_STEP2000_METRICS)
    candidate = candidate_metrics(baseline)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "evaluate",
        lambda *_args, **_kwargs: candidate)
    real = {
        "validation_access_updates": [],
        "validation_loader": object(),
        "normalizer": object(),
    }
    gate = subject.evaluate_step2000_gate(
        object(), real, torch.device("cpu"), {"metrics": baseline})
    assert gate["pass"] is True
    assert gate["delta"]["point_score"] == pytest.approx(-0.05)
    assert gate["delta"]["tke_rel_l2_score"] == pytest.approx(-0.20)
    assert real["validation_access_updates"] == [2000]
    with pytest.raises(RuntimeError, match="already accessed"):
        subject.evaluate_step2000_gate(
            object(), real, torch.device("cpu"), {"metrics": baseline})


def test_terminal_promotion_uses_native_shadow_and_revised_floors():
    baseline = dict(subject.NATIVE_STEP4000_METRICS)
    candidate = candidate_metrics(baseline, rel=0.50, tke=-0.20, sps=0.10)
    for key in subject.rehearsal.IDENTITY_METRIC_KEYS:
        candidate.setdefault(key, 0.0)
    result = subject.evaluate_terminal_promotion(candidate)
    assert result["pass"] is True
    assert result["status"] == "PROMOTE_HGRAD_TO_OOF"
    candidate["tke_rel_l2_score"] = baseline["tke_rel_l2_score"] - 0.31
    assert subject.evaluate_terminal_promotion(candidate)["pass"] is False


@pytest.mark.parametrize("preflight_pass,gate_pass,expected_updates,saves", [
    (False, False, 2, 0),
    (True, False, 3, 0),
    (True, True, 4, 1),
])
def test_failure_exits_never_save_and_success_saves_only_terminal(
        monkeypatch, tmp_path, preflight_pass, gate_pass,
        expected_updates, saves):
    monkeypatch.setattr(subject, "PREFLIGHT_UPDATE", 2)
    monkeypatch.setattr(subject, "PREFLIGHT_AUDIT_UPDATES", (1, 2))
    monkeypatch.setattr(subject, "GATE_UPDATE", 3)
    monkeypatch.setattr(subject, "TERMINAL_UPDATE", 4)
    monkeypatch.setattr(subject.rehearsal, "FORMAL_LOG_INTERVAL", 2)
    model = torch.nn.Linear(1, 1, bias=False)
    calls = {"iteration": 0, "save": [], "gate": 0}

    def backward(model, *_args, **_kwargs):
        calls["iteration"] += 1
        model.weight.square().sum().backward()
        audit = {
            "iteration": calls["iteration"],
            "finite_nonzero": True,
            "loss_ratio": 0.02 if preflight_pass else 1.0,
            "gradient_ratio": 0.05,
            "cosine": 0.1,
        } if calls["iteration"] <= 2 else None
        return {"real_loss": 0.0}, audit

    monkeypatch.setattr(subject, "backward_real_hgrad", backward)
    def gate(*_args, **_kwargs):
        calls["gate"] += 1
        return {
            "status": "GO" if gate_pass else "STOP_HGRAD_UPDATE2000_GATE",
            "pass": gate_pass,
        }

    monkeypatch.setattr(subject, "evaluate_step2000_gate", gate)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "normalizer_dict",
        lambda _normalizer: {})
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "save_checkpoint",
        lambda path, *_args, **_kwargs: calls["save"].append(path))
    args = argparse.Namespace(updates=5, output_dir=tmp_path)
    real = {"loader": [None], "normalizer": object()}
    result = subject.train_endpoint(
        args, model, real, {}, {}, torch.device("cpu"))
    assert result["completed_updates"] == expected_updates
    assert len(calls["save"]) == saves
    assert calls["gate"] == (0 if not preflight_pass else 1)
    assert result["checkpoint"] is None if saves == 0 else result["checkpoint"]
