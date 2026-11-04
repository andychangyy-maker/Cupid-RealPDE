from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for root in (CODE_ROOT, TRAINING_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import train_flow_backbone as trainer  # noqa: E402
from models.temporal_transport import (  # noqa: E402
    TemporalTransportConfig,
    TemporalTransportOperator,
)


class IdentityNormalizer:
    def __init__(self):
        self.mean_input = torch.zeros(3)
        self.mean_target = torch.zeros(3)
        self.std_input = torch.ones(3)
        self.std_target = torch.ones(3)

    @staticmethod
    def preprocess(
            history: torch.Tensor,
            target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return history, target


def parse_temporal_args(
        monkeypatch, output_dir: Path, phase: str = "sim",
        extras: tuple[str, ...] = ()):  # noqa: ANN001
    argv = [
        "train_flow_backbone.py",
        "--architecture", "temporal_transport",
        "--phase", phase,
        "--output-dir", str(output_dir),
        "--device", "cpu",
        "--temporal-hidden-channels", "8",
        "--temporal-spatial-blocks", "1",
        "--temporal-spatial-downsample", "2",
        "--temporal-mixer-blocks", "1",
        "--temporal-kernel", "3",
        "--groups", "4",
        "--updates", "2",
        "--batch-size", "1",
        "--eval-batch-size", "1",
        "--eval-interval", "2",
        "--checkpoint-interval", "10",
        "--sim-loss-mode", "mse",
        "--loss-mode", "mse",
        *extras,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return trainer.parse_args()


def test_temporal_cli_config_construct_and_official_selection(
        monkeypatch, tmp_path):
    args = parse_temporal_args(
        monkeypatch, tmp_path,
        extras=(
            "--transport-max-displacement", "2.5",
            "--transport-residual-limit", "3.0",
            "--temporal-mixer-mode", "framewise_spatial_control",
        ))
    trainer.validate(args)
    assert args.predict_uncertainty is False
    assert args.interval_loss_weight == 0.0
    assert args.selection_metric == "official_point_contribution"

    config = trainer.config_from_args(args)
    assert isinstance(config, TemporalTransportConfig)
    assert config.max_displacement == 2.5
    assert config.residual_limit == 3.0
    assert config.mixer_mode == "framewise_spatial_control"
    model = trainer.construct_model(
        "temporal_transport",
        {"architecture": "temporal_transport", **trainer.as_config(config)})
    assert isinstance(model, TemporalTransportOperator)
    assert model(torch.randn(1, 20, 8, 16, 3)).shape == (
        1, 20, 8, 16, 3)

    metrics = {
        "rel_l2_score": 94.0,
        "tke_rel_l2_score": 78.0,
        "mvpe_rel_l2_score": 93.0,
    }
    trainer.common.add_weighted(metrics, (1.0, 4.0, 1.0))
    assert metrics["official_point_contribution"] == pytest.approx(
        0.46 * 94.0 + 0.10 * 78.0 + 0.09 * 93.0)
    # The historical field deliberately keeps its former 50/30/20 meaning.
    assert metrics["point_score"] == pytest.approx(
        0.50 * 94.0 + 0.30 * 78.0 + 0.20 * 93.0)


def test_temporal_rejects_interval_head_request(monkeypatch, tmp_path):
    args = parse_temporal_args(
        monkeypatch, tmp_path, extras=("--predict-uncertainty",))
    with pytest.raises(ValueError, match="point-only"):
        trainer.validate(args)


def test_legacy_cli_defaults_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "train_flow_backbone.py",
        "--architecture", "unet",
        "--phase", "sim",
        "--output-dir", str(tmp_path),
        "--device", "cpu",
    ])
    args = trainer.parse_args()
    trainer.validate(args)
    assert args.predict_uncertainty is True
    assert args.selection_metric == "point_score"


def test_sim_random_build_and_real_checkpoint_load(monkeypatch, tmp_path):
    sim_args = parse_temporal_args(monkeypatch, tmp_path / "sim")
    trainer.validate(sim_args)
    torch.manual_seed(29)
    sim_model = trainer.build_or_load(sim_args, torch.device("cpu"))
    assert isinstance(sim_model, TemporalTransportOperator)

    checkpoint = tmp_path / "sim.pth"
    torch.save({
        "phase": "sim",
        "model_config": sim_model.architecture_config(),
        "model_state_dict": sim_model.state_dict(),
    }, checkpoint)
    real_args = parse_temporal_args(
        monkeypatch, tmp_path / "real", phase="real",
        extras=("--start-checkpoint", str(checkpoint)))
    trainer.validate(real_args)
    loaded = trainer.build_or_load(real_args, torch.device("cpu"))
    assert isinstance(loaded, TemporalTransportOperator)
    for name, value in sim_model.state_dict().items():
        torch.testing.assert_close(
            loaded.state_dict()[name], value, rtol=0.0, atol=0.0)

    sim_args.start_checkpoint = checkpoint
    with pytest.raises(ValueError, match="random initialization"):
        trainer.build_or_load(sim_args, torch.device("cpu"))


def test_common_sim_trainer_runs_two_point_only_updates(
        monkeypatch, tmp_path):
    output_dir = tmp_path / "two_steps"
    output_dir.mkdir()
    args = parse_temporal_args(monkeypatch, output_dir)
    weights = trainer.validate(args)
    torch.manual_seed(31)
    model = trainer.build_or_load(args, torch.device("cpu"))
    normalizer = IdentityNormalizer()
    trainer.common.set_model_stats(model, normalizer)

    history = torch.randn(2, 20, 8, 16, 3)
    target = torch.randn(2, 20, 8, 16, 3)
    loader = DataLoader(TensorDataset(history, target), batch_size=1)
    before = model.frame_encoder.stem.weight.detach().clone()
    summary = trainer.common.train_sim(
        args, model, loader, normalizer, torch.device("cpu"),
        dataset_size=2, trajectories=1, weights=weights)

    assert summary["updates"] == 2
    assert len(summary["history"]) == 1
    assert torch.isfinite(model.frame_encoder.stem.weight).all()
    assert not torch.equal(model.frame_encoder.stem.weight, before)
    assert (output_dir / "best.pth").is_file()
    assert (output_dir / "final.pth").is_file()
    payload = torch.load(
        output_dir / "final.pth", map_location="cpu", weights_only=False)
    assert payload["model_config"]["architecture"] == "temporal_transport"
    assert payload["model_config"]["predict_uncertainty"] is False
