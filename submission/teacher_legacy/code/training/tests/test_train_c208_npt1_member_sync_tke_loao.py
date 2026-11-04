from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_npt1_member_sync_tke_loao as subject  # noqa: E402
from models.ensemble_unet2d import PhysicalEnergyController  # noqa: E402


def test_controller_zero_contract_only_requires_zero_output_projection():
    state = {
        "body.0.weight": torch.randn(4, 4, 1, 1),
        "body.0.bias": torch.randn(4),
        "body.5.weight": torch.zeros(1, 4, 1, 1),
        "body.5.bias": torch.zeros(1),
    }
    assert subject.controller_output_projection_is_zero(state)
    state["body.5.bias"][0] = 1.0
    assert not subject.controller_output_projection_is_zero(state)


def test_loss_scale_applies_to_every_backward_gradient_source():
    """Partitioned held-fold callers must not leave UQ/TKE gradients unscaled."""
    source = inspect.getsource(subject.backward_real_supervised_npt1)
    assert "loss_scale: float = 1.0" in source
    assert "supervised_loss.new_tensor(loss_scale)" in source
    assert "floor_correction * loss_scale" in source
    assert "loss_scale * auxiliary_weight * auxiliary_loss" in source
    assert "loss_scale * soft_sps_weight * soft_sps_loss" in source


def test_frozen_controller_ema_canonicalizes_only_roundoff_drift():
    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.energy_controller = PhysicalEnergyController(4, 4, 2)

    torch.manual_seed(4)
    model = MiniModel()
    raw = subject.controller_state(model)
    shadow = {
        f"energy_controller.{name}": value.clone()
        for name, value in raw.items()
    }
    shadow["energy_controller.body.0.weight"].add_(1e-6)
    ema = SimpleNamespace(_shadow=shadow)
    expected = subject.ema_lib.tensor_state_sha256(raw)
    report = subject.canonicalize_frozen_controller_ema(
        ema, model, expected)
    assert report["changed_elements_before"] > 0
    assert report["max_abs_drift_before"] > 0
    assert report["exact_after"] is True
    after = {
        name.removeprefix("energy_controller."): value
        for name, value in ema._shadow.items()
    }
    assert subject.tensor_state_equal(after, raw)


def test_four_endpoint_ledger_blocks_held_until_all_seals(tmp_path):
    ledger = subject.FourEndpointAccessAudit(10)
    held = {"file": "held.h5", "aoa": 10, "val_boundary": 80}
    with pytest.raises(RuntimeError, match="before four seals"):
        ledger.before_open(held, "held")
    for index, endpoint in enumerate(subject.ENDPOINT_ORDER):
        ledger.seal(endpoint, {
            "path": str(tmp_path / f"{endpoint}.pth"),
            "file_sha256": f"{index + 1}" * 64,
            "model_state_sha256": f"{index + 5}" * 64,
        })
    ledger.before_open(held, "held")
    ledger.record_open(held, "held")
    for endpoint in subject.ENDPOINT_ORDER:
        ledger.record_evaluation(endpoint)
    report = ledger.report(["held.h5"])
    assert report["held_first_open_after_four_seals"] is True
    assert report["held_evaluation_order"] == list(subject.ENDPOINT_ORDER)


def test_two_fold_aggregation_stops_when_both_nonpositive(monkeypatch):
    monkeypatch.setattr(subject, "validate_fold_rows", lambda rows: None)
    stopped = [
        {"held_aoa": 10, "primary": {"non_time_delta": -0.01}},
        {"held_aoa": 0, "primary": {"non_time_delta": 0.0}},
    ]
    assert subject.aggregate_fold_rows(stopped)["decision"] == (
        "STOP_NPT1_FIRST_TWO_NONPOSITIVE")
    with pytest.raises(ValueError, match="complete two-fold"):
        subject.aggregate_fold_rows(stopped[:1])


def _args(held_aoa: int, prior_aggregate=None) -> argparse.Namespace:
    return argparse.Namespace(
        start_sim_sha256=subject.FORMAL_START_SHA256,
        fold_worker=True,
        aggregate_only=False,
        contract_only=False,
        held_aoa=held_aoa,
        output_dir=Path("out"),
        clean_manifest=Path("manifest.json"),
        start_sim=Path("start.pth"),
        hardware_class="rtx4090",
        execute_formal_gpu=True,
        acknowledge_protocol=subject.FORMAL_ACKNOWLEDGEMENT,
        prior_aggregate=prior_aggregate,
        fold_result=None,
        aggregate_output=None,
    )


def test_fold_cli_enforces_wave_authorization_before_execution():
    subject.validate_args(_args(10))
    subject.validate_args(_args(0))
    with pytest.raises(ValueError, match="cannot receive"):
        subject.validate_args(_args(10, ["prior.json", "a" * 64]))
    with pytest.raises(ValueError, match="outside"):
        subject.validate_args(_args(20))
    with pytest.raises(ValueError, match="outside"):
        subject.validate_args(_args(20, ["prior.json", "a" * 64]))
    source = inspect.getsource(subject.run_fold)
    assert source.index("validate_prior_authorization") < source.index(
        "load_prefix_rows")


def test_primary_fold_gate_has_no_raw_endpoint_selection():
    assert subject.fold_gate({"non_time_delta": -0.051})["pass"] is False
    assert subject.fold_gate({"non_time_delta": -0.05})["pass"] is True


def test_tke_aux_weight_has_frozen_cosine_boundaries():
    assert subject.tke_aux_weight(1) == subject.TKE_AUX_WEIGHT
    assert subject.tke_aux_weight(subject.TKE_AUX_UPDATES) == 0.0
    assert subject.tke_aux_weight(subject.TKE_AUX_UPDATES + 1) == 0.0
    assert (subject.tke_aux_weight(1)
            > subject.tke_aux_weight(750)
            > subject.tke_aux_weight(1499))
    with pytest.raises(ValueError, match="positive"):
        subject.tke_aux_weight(0)


def test_truth_native_tke_relative_l2_matches_manual_identity_normalizer():
    class IdentityNormalizer:
        mean_target = torch.zeros(3)
        std_target = torch.ones(3)

    prediction = torch.tensor([[[[[1.0, 0.0, 0.0]]],
                                [[[-1.0, 0.0, 0.0]]]]])
    target = torch.tensor([[[[[2.0, 0.0, 0.0]]],
                            [[[-2.0, 0.0, 0.0]]]]])
    # K_pred=0.5 and K_truth=2.0 at the only spatial location.
    loss = subject.truth_native_tke_relative_l2(
        prediction, target, IdentityNormalizer())
    assert float(loss) == pytest.approx(0.75)


def test_official_tke_score_loss_matches_manual_transform():
    class IdentityNormalizer:
        mean_target = torch.zeros(3)
        std_target = torch.ones(3)

    prediction = torch.tensor([[[[[1.0, 0.0, 0.0]]],
                                [[[-1.0, 0.0, 0.0]]]]])
    target = torch.tensor([[[[[2.0, 0.0, 0.0]]],
                            [[[-2.0, 0.0, 0.0]]]]])
    loss = subject.official_tke_score_loss(
        prediction, target, IdentityNormalizer())
    assert float(loss) == pytest.approx(1.0 - 1.0 / (1.0 + 0.5 * 0.75))


def test_output_legacy_tke_gradient_floor_is_sample_local():
    prediction = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    primary = (
        prediction[0, 0] + prediction[0, 1]
        + 3.0 * prediction[1, 0] + prediction[1, 1])
    legacy = 2.0 * prediction[:, 0].sum()
    tke = prediction[:, 0].sum()
    correction, diagnostic = subject.output_legacy_tke_gradient_floor(
        primary, legacy, tke, prediction)
    assert correction is not None
    assert correction == pytest.approx(
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert diagnostic[
        "legacy_tke_floor_applied_fraction"] == pytest.approx(0.5)
    assert diagnostic["legacy_tke_floor_finite"] is True


def test_output_legacy_tke_gradient_floor_supports_member_tensor():
    members = torch.tensor(
        [[[1.0, 2.0], [2.0, 3.0]],
         [[3.0, 4.0], [4.0, 5.0]]], requires_grad=True)
    prediction = members.mean(dim=1)
    primary = prediction.square().mean() + 0.25 * members.square().mean()
    legacy = primary + members[..., 0].mean()
    tke = prediction[..., 0].mean()
    correction, diagnostic = subject.output_legacy_tke_gradient_floor(
        primary, legacy, tke, members)
    assert correction is not None
    assert correction.shape == members.shape
    primary.backward(retain_graph=True)
    before = members.grad.detach().clone()
    members.backward(correction)
    assert not torch.equal(members.grad, before)
    assert diagnostic["legacy_tke_floor_finite"] is True


def test_member_floor_tke_direction_is_connected_to_members():
    prediction = torch.randn(2, 3, 4, requires_grad=True)
    members = torch.randn(2, 4, 3, 4, requires_grad=True)
    protected = subject.tke_floor_prediction(prediction, members, True)
    gradient = torch.autograd.grad(protected.square().mean(), members)[0]
    assert protected.shape == prediction.shape
    assert gradient.shape == members.shape
    assert prediction.grad is None


def test_controller_only_aux_gradient_does_not_change_legacy_grad():
    legacy = torch.nn.Parameter(torch.tensor(2.0))
    controller = torch.nn.Parameter(torch.tensor(3.0))
    standard = (legacy + controller).square()
    standard.backward(retain_graph=True)
    legacy_before = legacy.grad.clone()
    controller_before = controller.grad.clone()
    auxiliary = (legacy * controller).square()
    gradients = torch.autograd.grad(auxiliary, (controller,))
    subject.add_parameter_gradients((controller,), gradients)
    assert torch.equal(legacy.grad, legacy_before)
    assert not torch.equal(controller.grad, controller_before)


def test_two_fold_promotion_gate_requires_joint_tke_sps_non_time():
    rows = [
        {"primary": {"non_time_delta": 0.11, "delta": {
            "tke_rel_l2_score": 0.21, "native_sps_score": 0.11}}},
        {"primary": {"non_time_delta": 0.12, "delta": {
            "tke_rel_l2_score": 0.22, "native_sps_score": 0.12}}},
    ]
    passing = {
        "delta": {
            "tke_rel_l2_score": 0.20,
            "native_sps_score": 0.10,
            "rel_l2_score": -0.05,
            "mvpe_rel_l2_score": -0.05,
        },
        "non_time_delta": 0.10,
    }
    assert subject.pooled_gate(passing, rows)["pass"] is True
    failing = {**passing, "delta": {**passing["delta"],
                                     "native_sps_score": 0.099}}
    assert subject.pooled_gate(failing, rows)["pass"] is False
