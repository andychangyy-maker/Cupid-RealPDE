from __future__ import annotations

from dataclasses import asdict
import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fp32_ema_shadow as ema_lib  # noqa: E402
from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)
from models.ensemble_unet2d import EnsembleUNet2dConfig  # noqa: E402
import train_b210_causal_phase_direction_real4k as subject  # noqa: E402


class IdentityNormalizer:
    def __init__(self) -> None:
        self.mean_input = torch.zeros(3)
        self.mean_target = torch.zeros(3)
        self.std_input = torch.ones(3)
        self.std_target = torch.ones(3)


def test_contract_is_full_real4k_matched_and_has_no_score_gate() -> None:
    contract = subject.experiment_contract()
    assert contract["adapter"]["parameters"] == 41_667
    assert contract["real"]["updates_each_arm"] == 4_000
    assert contract["real"]["optimizer_windows"] == 9_933
    assert contract["real"]["tke_or_mvpe_loss"] is None
    assert contract["evaluation"]["clean716_read_by_this_script"] is False
    assert contract["evaluation"]["score_gate"] is None
    assert "fixed streamwise threshold" in contract["adapter"]["selection"]


def test_candidate_and_control_default_states_are_exactly_matched() -> None:
    candidate = subject.build_adapter("causal_temporal", torch.device("cpu"))
    control = subject.build_adapter(
        "framewise_spatial_control", torch.device("cpu"))
    assert sum(p.numel() for p in candidate.parameters()) == 41_667
    assert candidate.state_dict().keys() == control.state_dict().keys()
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, control.state_dict()[name]), name


def test_phase_objective_is_finite_and_differentiable() -> None:
    torch.manual_seed(41)
    prediction = torch.randn(2, 5, 4, 6, 3, requires_grad=True)
    target = torch.randn_like(prediction)
    loss, details = subject.phase_objective(
        prediction, target, IdentityNormalizer())
    assert torch.isfinite(loss)
    assert set(details) == {
        "official_rel_score_loss",
        "relative_l2_error",
        "radius_product_weighted_direction_cosine",
        "radius_product_weighted_direction_loss",
        "direction_radius_product_mean",
        "active_direction_fraction",
        "total_phase_objective",
    }
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    relative = torch.linalg.vector_norm(
        (prediction.detach()[..., :2] - target[..., :2]).reshape(2, -1),
        dim=1) / torch.linalg.vector_norm(
            target[..., :2].reshape(2, -1), dim=1).clamp_min(1.0e-8)
    expected = 1.0 - 1.0 / (1.0 + 0.5 * relative.mean())
    torch.testing.assert_close(details["official_rel_score_loss"], expected)


def test_base_checkpoint_parser_accepts_only_ensemble_state_mappings() -> None:
    config = asdict(EnsembleUNet2dConfig())
    payload = {
        "model_config": {"architecture": "ensemble_unet", **config},
        "model_state_dict": {"weight": torch.ones(2)},
    }
    parsed, state, key = subject._extract_base_checkpoint(payload)
    assert parsed == EnsembleUNet2dConfig()
    assert state is payload["model_state_dict"]
    assert key == "model_state_dict"

    changed = dict(payload)
    changed["model_config"] = {
        **payload["model_config"], "architecture": "fno"}
    with pytest.raises(ValueError, match="not an EnsembleUNet2d"):
        subject._extract_base_checkpoint(changed)


def test_endpoint_stores_adapter_only_and_references_base_by_digest() -> None:
    config = CausalPhaseDirectionConfig(
        in_steps=5, out_steps=5, hidden_channels=8,
        spatial_downsample=2, temporal_blocks=2,
        temporal_kernel=3, groups=4)
    adapter = CausalPhaseDirectionAdapter(config)
    ema = ema_lib.FP32EMAShadow(adapter, 0.999)
    payload = subject.endpoint_payload(
        adapter=adapter,
        ema=ema,
        arm="causal_temporal",
        source={"sha256": "a" * 64},
        normalizer=IdentityNormalizer(),
        schedule_sha="b" * 64,
        trace=[], elapsed_seconds=1.0, updates=2)
    assert payload["update"] == 2
    assert payload["base_checkpoint"] == {"sha256": "a" * 64}
    assert "raw_adapter_state" in payload
    assert "ema_adapter_state" in payload
    assert "model_state_dict" not in payload
    assert "base_state" not in payload
