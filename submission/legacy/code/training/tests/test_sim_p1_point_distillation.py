from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader


TRAINING_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TRAINING_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache_builder = load_script(
    "sim_p1_point_cache_test_subject",
    "build_sim_p1_point_teacher_cache.py")
trainer = load_script(
    "sim_p1_point_train_test_subject",
    "train_ensemble_unet2d.py")


class TinySimDataset:
    def __init__(self):
        self.windows = [
            SimpleNamespace(trajectory=0, start=0),
            SimpleNamespace(trajectory=0, start=20),
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        history = torch.full((3, 4, 5, 3), float(index))
        target = history + 1.0
        return history, target


class TinyTeacherInputDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        history = torch.zeros(20, 8, 8, 3)
        target = torch.zeros(20, 8, 8, 3)
        return history, target


def valid_cache_payload(dataset: TinySimDataset, manifest: list[dict]) -> dict:
    keys = cache_builder.ordered_window_keys(dataset.windows, manifest)
    prediction = torch.zeros(
        cache_builder.expected_prediction_shape(dataset), dtype=torch.float16)
    return {
        "schema_version": cache_builder.CACHE_SCHEMA_VERSION,
        "method": cache_builder.CACHE_METHOD,
        "composition": cache_builder.COMPOSITION,
        "teacher_checkpoints": {
            name: {
                "actual_sha256": character * 64,
                "expected_sha256": character * 64,
                "sha256_match": True,
                "phase": "sim",
            }
            for name, character in zip(("a", "b", "c"), ("a", "b", "c"))
        },
        "split_manifest": manifest,
        "split_manifest_sha256": cache_builder.canonical_sha256(manifest),
        "window_keys": keys,
        "window_order_sha256": cache_builder.canonical_sha256(keys),
        "prediction_shape": list(prediction.shape),
        "prediction_dtype": "float16",
        "prediction_uv_physical": prediction,
    }


def test_cache_contract_accepts_exact_manifest_order_and_float16_tensor():
    dataset = TinySimDataset()
    manifest = [{"file": "sim.h5", "starts": [0, 20]}]
    payload = valid_cache_payload(dataset, manifest)

    prediction, metadata = cache_builder.validate_cache_payload(
        payload, manifest, dataset)

    assert prediction is payload["prediction_uv_physical"]
    assert prediction.dtype == torch.float16
    assert metadata["windows"] == 2
    assert metadata["prediction_shape"] == [2, 3, 4, 5, 2]


def test_cache_contract_rejects_teacher_index_misalignment():
    dataset = TinySimDataset()
    manifest = [{"file": "sim.h5", "starts": [0, 20]}]
    payload = valid_cache_payload(dataset, manifest)
    payload["window_keys"] = list(reversed(payload["window_keys"]))

    with pytest.raises(ValueError, match="window order mismatch"):
        cache_builder.validate_cache_payload(payload, manifest, dataset)


def test_cache_composition_matches_current_p1_point_formula():
    prediction_a = torch.tensor(
        [[[[[1.0, 2.0]]], [[[3.0, 4.0]]]]], dtype=torch.float16)
    prediction_b = torch.tensor(
        [[[[[5.0, 6.0]]], [[[7.0, 8.0]]]]], dtype=torch.float16)
    prediction_c = torch.tensor(
        [[[[[2.0, 8.0]]], [[[6.0, 4.0]]]]], dtype=torch.float16)
    expected_base = prediction_a.float().lerp(
        prediction_b.float(), 0.5).to(torch.float16).float()
    expected = (
        expected_base
        + 0.5 * (
            prediction_c.float().mean(dim=1, keepdim=True)
            - expected_base.mean(dim=1, keepdim=True))
    ).to(torch.float16)

    cache_builder.blend_point_base_in_place(prediction_a, prediction_b)
    result = cache_builder.add_p1_temporal_mean_correction_in_place(
        prediction_a, prediction_c)

    assert result is prediction_a
    assert torch.equal(result, expected)


def test_checkpoint_loader_and_collection_use_stored_normalizer(tmp_path):
    config = cache_builder.EnsembleUNet2dConfig(
        base=4, depth=2, groups=1, ensemble_members=2,
        uq_hidden_channels=4, temporal_uq_channels=4)
    model = cache_builder.EnsembleUNet2d(config)
    checkpoint = tmp_path / "teacher.pth"
    torch.save({
        "phase": "sim",
        "model_config": model.architecture_config(),
        "model_state_dict": model.state_dict(),
        "normalizer": {
            "mean_input": [0.0, 0.0, 0.0],
            "mean_target": [0.0, 0.0, 0.0],
            "std_input": [1.0, 1.0, 1.0],
            "std_target": [1.0, 1.0, 1.0],
        },
    }, checkpoint)
    loader = DataLoader(TinyTeacherInputDataset(), batch_size=1, shuffle=False)

    prediction = cache_builder.collect_prediction(
        checkpoint, loader, torch.device("cpu"), torch.float32,
        expected_shape=(1, 20, 8, 8, 2))

    assert prediction.shape == (1, 20, 8, 8, 2)
    assert prediction.dtype == torch.float16
    assert torch.isfinite(prediction).all()
    prediction.add_(1.0)
    assert torch.isfinite(prediction).all()


def test_checkpoint_loader_rejects_non_sim_phase(tmp_path):
    config = cache_builder.EnsembleUNet2dConfig(
        base=4, depth=2, groups=1, ensemble_members=2,
        uq_hidden_channels=4, temporal_uq_channels=4)
    model = cache_builder.EnsembleUNet2d(config)
    checkpoint = tmp_path / "real_teacher.pth"
    torch.save({
        "phase": "real",
        "model_config": model.architecture_config(),
        "model_state_dict": model.state_dict(),
        "normalizer": {},
    }, checkpoint)

    with pytest.raises(ValueError, match="phase='sim'"):
        cache_builder.validate_checkpoint_phase(checkpoint)
    with pytest.raises(ValueError, match="phase='sim'"):
        cache_builder.load_teacher(checkpoint, torch.device("cpu"), torch.float32)


def test_checkpoint_identity_rejects_unexpected_sha256(tmp_path):
    checkpoint = tmp_path / "teacher.pth"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        cache_builder.checkpoint_identity(checkpoint, "0" * 64)

    actual = cache_builder.sha256(checkpoint)
    identity = cache_builder.checkpoint_identity(checkpoint, actual.upper())
    assert identity["actual_sha256"] == actual
    assert identity["expected_sha256"] == actual
    assert identity["sha256_match"] is True


def test_tagged_dataset_returns_original_teacher_cache_index():
    tagged = trainer.TrajectoryTaggedDataset(TinySimDataset())
    history, target, trajectory, index = tagged[1]
    assert trajectory == 0
    assert index == 1
    assert torch.equal(history, torch.full((3, 4, 5, 3), 1.0))
    assert torch.equal(target, torch.full((3, 4, 5, 3), 2.0))


def test_no_cache_returns_supervised_loss_bitwise_unchanged():
    supervised = torch.tensor(1.234567, dtype=torch.float32, requires_grad=True)
    prediction = torch.randn(1, 2, 3, 4, 3, requires_grad=True)

    result, components = trainer.add_point_distillation_loss(
        supervised, prediction, None, normalizer=None, distill_weight=0.0)

    assert result is supervised
    assert result.detach().view(torch.int32).item() == (
        supervised.detach().view(torch.int32).item())
    assert components == {}


def test_zero_input_noise_returns_by_identity_without_consuming_rng():
    normalized = torch.randn(1, 2, 3, 4, 3)
    physical = torch.randn_like(normalized)
    rng_before = torch.random.get_rng_state().clone()

    result = trainer.add_training_input_noise(
        normalized, physical, noise_std=0.0)

    assert result is normalized
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_input_noise_changes_only_fluid_uv_and_preserves_source():
    normalized = torch.arange(36, dtype=torch.float32).reshape(1, 2, 2, 3, 3)
    original = normalized.clone()
    physical = torch.zeros_like(normalized)
    physical[0, 0, 0, 1, 0] = 1.0
    fluid = physical[..., :2].abs().sum(dim=-1, keepdim=True).ne(0.0)
    torch.manual_seed(19)

    result = trainer.add_training_input_noise(
        normalized, physical, noise_std=0.2)

    assert result is not normalized
    assert torch.equal(normalized, original)
    assert torch.equal(result[..., 2], original[..., 2])
    solid_uv = (~fluid).expand_as(result[..., :2])
    assert torch.equal(result[..., :2][solid_uv], original[..., :2][solid_uv])
    fluid_uv = fluid.expand_as(result[..., :2])
    assert not torch.equal(
        result[..., :2][fluid_uv], original[..., :2][fluid_uv])


@pytest.mark.parametrize("noise_std", [-0.1, float("nan"), float("inf")])
def test_input_noise_rejects_invalid_scale(noise_std):
    values = torch.zeros(1, 2, 3, 4, 3)
    with pytest.raises(ValueError, match="finite and non-negative"):
        trainer.add_training_input_noise(values, values, noise_std)


def test_zero_input_gain_returns_by_identity_without_consuming_rng():
    physical = torch.randn(1, 2, 3, 4, 3)
    rng_before = torch.random.get_rng_state().clone()

    result = trainer.add_training_input_gain(physical, gain_std=0.0)

    assert result is physical
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_input_gain_changes_only_uv_and_preserves_source():
    physical = torch.ones(2, 3, 4, 5, 3)
    physical[..., 2] = 7.0
    physical[:, :, 0, 0, :2] = 0.0
    original = physical.clone()
    torch.manual_seed(23)

    result = trainer.add_training_input_gain(physical, gain_std=0.2)

    assert result is not physical
    assert torch.equal(physical, original)
    assert torch.equal(result[..., 2], original[..., 2])
    assert torch.count_nonzero(result[:, :, 0, 0, :2]) == 0
    for sample in range(physical.shape[0]):
        for channel in range(2):
            ratio = result[sample, :, 1:, :, channel]
            assert torch.allclose(ratio, ratio.reshape(-1)[0].expand_as(ratio))


@pytest.mark.parametrize("gain_std", [-0.1, float("nan"), float("inf")])
def test_input_gain_rejects_invalid_scale(gain_std):
    values = torch.zeros(1, 2, 3, 4, 3)
    with pytest.raises(ValueError, match="finite and non-negative"):
        trainer.add_training_input_gain(values, gain_std)


def test_zero_masked_blur_returns_by_identity_without_consuming_rng():
    normalized = torch.randn(1, 2, 3, 4, 3)
    physical = torch.randn_like(normalized)
    rng_before = torch.random.get_rng_state().clone()

    result = trainer.masked_velocity_blur(
        normalized, physical, probability=0.0, strength=0.0)

    assert result is normalized
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_mask_normalized_blur_does_not_leak_solid_zeros():
    normalized = torch.zeros(1, 1, 5, 5, 3)
    normalized[:, :, 1:4, 1:4, 0] = 3.0
    normalized[:, :, 1:4, 1:4, 1] = -2.0
    physical = torch.zeros_like(normalized)
    physical[:, :, 1:4, 1:4, :2] = 1.0

    result = trainer.masked_velocity_blur(
        normalized, physical, probability=1.0, strength=1.0)

    assert torch.allclose(
        result[:, :, 1:4, 1:4, :2],
        normalized[:, :, 1:4, 1:4, :2])
    assert torch.count_nonzero(result[:, :, (0, 4), :, :2]) == 0
    assert torch.count_nonzero(result[:, :, :, (0, 4), :2]) == 0


def test_masked_blur_changes_only_fluid_uv_and_preserves_source():
    normalized = torch.zeros(1, 1, 5, 5, 3)
    normalized[..., 0] = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    normalized[..., 1] = normalized[..., 0].flip(-1)
    normalized[..., 2] = 7.0
    original = normalized.clone()
    physical = torch.zeros_like(normalized)
    physical[:, :, 1:4, 1:4, :2] = 1.0
    fluid = physical[..., :2].abs().sum(dim=-1, keepdim=True).ne(0.0)

    result = trainer.masked_velocity_blur(
        normalized, physical, probability=1.0, strength=1.0)

    assert result is not normalized
    assert torch.equal(normalized, original)
    assert torch.equal(result[..., 2], original[..., 2])
    solid_uv = (~fluid).expand_as(result[..., :2])
    assert torch.equal(result[..., :2][solid_uv], original[..., :2][solid_uv])
    fluid_uv = fluid.expand_as(result[..., :2])
    assert not torch.equal(
        result[..., :2][fluid_uv], original[..., :2][fluid_uv])


@pytest.mark.parametrize(
    ("probability", "strength", "message"),
    [(-0.1, 0.5, "probability"),
     (1.1, 0.5, "probability"),
     (0.5, -0.1, "strength"),
     (0.5, 1.1, "strength"),
     (0.5, 0.0, "enabled together"),
     (0.0, 0.5, "enabled together")],
)
def test_masked_blur_rejects_invalid_contract(probability, strength, message):
    values = torch.zeros(1, 2, 3, 4, 3)
    with pytest.raises(ValueError, match=message):
        trainer.masked_velocity_blur(
            values, values, probability=probability, strength=strength)


def test_temporal_jitter_uses_shifted_released_frames(monkeypatch):
    trajectory = torch.arange(50, dtype=torch.float32).view(50, 1, 1, 1)
    base = SimpleNamespace(
        windows=[SimpleNamespace(trajectory=0, start=1)],
        trajectories=[trajectory],
    )
    monkeypatch.setattr(
        torch, "randint", lambda *_args, **_kwargs: torch.tensor(2))

    class DatasetWithLength:
        windows = base.windows
        trajectories = base.trajectories

        def __len__(self):
            return 1

    jittered = trainer.TemporalJitterDataset(DatasetWithLength(), max_shift=9)
    history, target = jittered[0]

    assert float(history[0]) == 3.0
    assert float(history[-1]) == 22.0
    assert float(target[0]) == 23.0
    assert float(target[-1]) == 42.0


def test_temporal_jitter_audit_is_rng_and_tensor_identity():
    trajectory = torch.arange(60, dtype=torch.float32).view(60, 1, 1, 1)

    class DatasetWithLength:
        windows = [SimpleNamespace(trajectory=0, start=1)]
        trajectories = [trajectory]

        def __len__(self):
            return 1

    torch.manual_seed(31)
    plain = trainer.TemporalJitterDataset(DatasetWithLength(), max_shift=9)
    plain_history, plain_target = plain[0]
    plain_rng_after = torch.random.get_rng_state().clone()

    audit = trainer.SimStreamAudit()
    torch.manual_seed(31)
    observed = trainer.TemporalJitterDataset(
        DatasetWithLength(), max_shift=9, audit=audit)
    observed_history, observed_target = observed[0]
    observed_rng_after = torch.random.get_rng_state().clone()

    assert torch.equal(observed_history, plain_history)
    assert torch.equal(observed_target, plain_target)
    assert torch.equal(observed_rng_after, plain_rng_after)
    assert audit.samples == 1
    assert audit.base_sample_stream_sha256 == hashlib.sha256(
        struct.pack("<q", 0)).hexdigest()
    shift = int(float(observed_history[0]) - 1)
    assert audit.jitter_shift_stream_sha256 == hashlib.sha256(
        struct.pack("<q", shift)).hexdigest()


def test_control_audit_is_dataset_and_rng_identity():
    plain = trainer.TrajectoryTaggedDataset(TinySimDataset())
    audit = trainer.SimStreamAudit()
    observed = trainer.TrajectoryTaggedDataset(TinySimDataset(), audit=audit)

    torch.manual_seed(37)
    plain_item = plain[1]
    plain_rng_after = torch.random.get_rng_state().clone()
    torch.manual_seed(37)
    observed_item = observed[1]
    observed_rng_after = torch.random.get_rng_state().clone()

    for observed_value, plain_value in zip(observed_item, plain_item):
        if isinstance(plain_value, torch.Tensor):
            assert torch.equal(observed_value, plain_value)
        else:
            assert observed_value == plain_value
    assert torch.equal(observed_rng_after, plain_rng_after)
    assert audit.samples == 1
    assert audit.base_sample_stream_sha256 == hashlib.sha256(
        struct.pack("<q", 1)).hexdigest()
    assert audit.jitter_shift_stream_sha256 == hashlib.sha256(
        struct.pack("<q", 0)).hexdigest()


def test_canonical_and_model_state_hashes_are_order_stable_and_rng_neutral():
    assert trainer.canonical_sha256({"b": 2, "a": [1, 3]}) == (
        trainer.canonical_sha256({"a": [1, 3], "b": 2}))
    first = {
        "z.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "a.count": torch.tensor(7, dtype=torch.int64),
    }
    reordered = {"a.count": first["a.count"], "z.weight": first["z.weight"]}
    rng_before = torch.random.get_rng_state().clone()

    digest = trainer.model_state_sha256(first)

    assert digest == trainer.model_state_sha256(reordered)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    changed = {key: value.clone() for key, value in first.items()}
    changed["z.weight"][0, 0] += 1
    assert trainer.model_state_sha256(changed) != digest


@pytest.mark.parametrize("variant", ["control", "candidate"])
def test_formal_sim_audit_contract_accepts_only_frozen_arm(
        variant, monkeypatch, tmp_path):
    argv = [
        "trainer", "--phase", "sim", "--output-dir", str(tmp_path),
        "--updates", "10000", "--seed", "4", "--window-stride", "10",
        "--sim-audit-variant", variant,
    ]
    if variant == "candidate":
        argv.extend([
            "--temporal-jitter-max", "9",
            "--input-gain-std", "0.02",
            "--input-noise-std", "0.02",
            "--input-blur-probability", "0.25",
            "--input-blur-strength", "0.25",
        ])
    monkeypatch.setattr("sys.argv", argv)

    args = trainer.parse_args()

    trainer.validate(args)
    assert args.teacher_cache is None
    assert args.distill_weight == 0.0


def test_formal_sim_audit_rejects_teacher_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "trainer", "--phase", "sim", "--output-dir", str(tmp_path),
        "--updates", "10000", "--seed", "4", "--window-stride", "10",
        "--sim-audit-variant", "control",
        "--teacher-cache", "teacher.pt", "--distill-weight", "0.1",
    ])
    args = trainer.parse_args()

    with pytest.raises(ValueError, match="no teacher cache/distillation"):
        trainer.validate(args)


def test_temporal_jitter_rejects_unshifted_teacher_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["trainer", "--phase", "sim", "--output-dir", str(tmp_path),
         "--teacher-cache", "teacher.pt", "--distill-weight", "0.005",
         "--temporal-jitter-max", "9"])
    args = trainer.parse_args()

    with pytest.raises(ValueError, match="cache indexed by unshifted windows"):
        trainer.validate(args)


def test_component_distillation_has_finite_nonzero_student_gradient():
    torch.manual_seed(7)
    normalizer = SimpleNamespace(
        mean_target=torch.tensor([0.15, -0.01, 0.0]),
        std_target=torch.tensor([0.10, 0.02, 1.0]),
    )
    student = torch.randn(2, 4, 20, 40, 3, requires_grad=True)
    teacher = (
        student.detach()[..., :2]
        * normalizer.std_target[:2]
        + normalizer.mean_target[:2])
    temporal_shift = torch.linspace(-0.004, 0.006, 4).view(1, 4, 1, 1, 1)
    teacher = teacher + temporal_shift

    loss, components = trainer.component_point_distillation_loss(
        student, teacher, normalizer)
    expected = (
        2.0 * components["distill_temporal_mean"]
        + components["distill_zero_mean_fluctuation"]
        + 2.0 * components["distill_mean_profile"]
        + 0.25 * components["distill_physical_tke_map"])
    assert torch.allclose(loss.detach(), expected)

    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert float(student.grad.abs().sum()) > 0.0
