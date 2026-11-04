from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_ensemble_uq_oof.py"
SPEC = importlib.util.spec_from_file_location("ensemble_uq_oof_fixed", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def parse_args(monkeypatch, *extra: str):
    monkeypatch.setattr(sys, "argv", [
        "train_ensemble_uq_oof.py",
        "--point-checkpoint", "point.pth",
        "--output-dir", "output",
        *extra,
    ])
    return subject.parse_args()


def test_fixed_terminal_folds_is_opt_in(monkeypatch):
    default_args = parse_args(monkeypatch)
    fixed_args = parse_args(monkeypatch, "--fixed-terminal-folds")

    assert default_args.fixed_terminal_folds is False
    assert fixed_args.fixed_terminal_folds is True


def test_frozen_point_eval_mode_is_explicit_opt_in(monkeypatch):
    default_args = parse_args(monkeypatch)
    frozen_args = parse_args(monkeypatch, "--frozen-point-eval-mode")

    assert default_args.frozen_point_eval_mode is False
    assert frozen_args.frozen_point_eval_mode is True


def test_real_input_augmentation_is_explicit_opt_in(monkeypatch):
    default_args = parse_args(monkeypatch)
    augmented_args = parse_args(
        monkeypatch,
        "--real-input-gain-std", "0.01",
        "--real-input-noise-std", "0.01",
        "--real-input-blur-probability", "0.15",
        "--real-input-blur-strength", "0.20")
    assert default_args.real_input_gain_std == 0.0
    assert default_args.real_input_noise_std == 0.0
    assert default_args.real_input_blur_probability == 0.0
    assert default_args.real_input_blur_strength == 0.0
    assert augmented_args.real_input_gain_std == 0.01
    assert augmented_args.real_input_noise_std == 0.01
    assert augmented_args.real_input_blur_probability == 0.15
    assert augmented_args.real_input_blur_strength == 0.20


def test_robust_sps_stress_is_explicit_opt_in(monkeypatch):
    default_args = parse_args(monkeypatch)
    robust_args = parse_args(
        monkeypatch,
        "--robust-sps-weight", "0.5",
        "--robust-residual-jitter-fraction", "0.05",
        "--robust-residual-tail-scale", "1.05")
    assert default_args.robust_sps_weight == 0.0
    assert default_args.robust_residual_jitter_fraction == 0.0
    assert default_args.robust_residual_tail_scale == 1.0
    assert robust_args.robust_sps_weight == 0.5
    assert robust_args.robust_residual_jitter_fraction == 0.05
    assert robust_args.robust_residual_tail_scale == 1.05
    subject.validate(robust_args)


def test_peer_sps_stress_is_explicit_opt_in(monkeypatch):
    default_args = parse_args(monkeypatch)
    peer_args = parse_args(
        monkeypatch,
        "--peer-sps-weight", "0.5",
        "--peer-residual-mix-fraction", "0.25")
    assert default_args.peer_sps_weight == 0.0
    assert default_args.peer_residual_mix_fraction == 0.0
    assert peer_args.peer_sps_weight == 0.5
    assert peer_args.peer_residual_mix_fraction == 0.25
    subject.validate(peer_args)


def test_nearest_forecast_peer_never_selects_self():
    torch.manual_seed(7)
    prediction = torch.randn(4, 5, 3, 4, 2)
    peer, distance = subject.nearest_forecast_dynamics_peer(prediction)
    assert peer.shape == (4,)
    assert torch.all(peer != torch.arange(4))
    assert torch.isfinite(distance)
    assert distance >= 0.0


def test_residual_mixture_proper_loss_is_finite_and_trainable():
    class IdentityNormalizer:
        mean_target = torch.zeros(3)
        std_target = torch.ones(3)

    torch.manual_seed(13)
    prediction = torch.zeros(2, 3, 2, 2, 3)
    target = torch.randn_like(prediction) * 0.005
    logits = torch.randn(2, 3, 2, 2, 2, 3, requires_grad=True)
    means = torch.randn(2, 3, 2, 2, 2, 3, requires_grad=True)
    raw_scales = torch.randn(2, 3, 2, 2, 2, 3, requires_grad=True)
    scales = torch.nn.functional.softplus(raw_scales) + 0.025
    loss, metrics = subject.residual_mixture_proper_loss(
        prediction, (logits, means, scales), target, IdentityNormalizer())
    assert torch.isfinite(loss)
    assert set(metrics) == {
        "mixture_loss", "mixture_nll", "mixture_crps",
        "mixture_scale_mean", "mixture_weight_entropy"}
    loss.backward()
    for value in (logits.grad, means.grad, raw_scales.grad):
        assert value is not None
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) > 0


def test_frozen_point_eval_mode_only_trains_uq_modules():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Dropout(0.2))
            self.uq_projections = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.uq_fusion = torch.nn.Linear(2, 2)
            self.quantile_head = torch.nn.Linear(2, 2)
            self.utility_refinement_head = None
            self.temporal_uq_refiner = None
            self.coverage_calibration_head = None

    model = Tiny()
    args = type("Args", (), {"frozen_point_eval_mode": True})()
    subject.set_training_mode(args, model)
    assert model.training is False
    assert model.backbone.training is False
    assert model.uq_projections.training is True
    assert model.uq_fusion.training is True
    assert model.quantile_head.training is True


def test_fixed_terminal_contract_disables_held_fold_selection():
    default = subject.fold_selection_contract(False)
    fixed = subject.fold_selection_contract(True)

    assert default == {
        "select_best": True,
        "selection_rule": "best_held_fold_sps",
        "held_fold_checkpoint_selection": True,
    }
    assert fixed == {
        "select_best": False,
        "selection_rule": "fixed_terminal_no_held_fold_selection",
        "held_fold_checkpoint_selection": False,
    }
