from __future__ import annotations

from contextlib import contextmanager
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_hcobf_real4k as subject  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig  # noqa: E402


def _model() -> EnsembleUNet2d:
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8, depth=4, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8, mean_fluctuation=True,
        history_conditioned_operator_basis=True))
    model.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3))
    return model.eval()


def test_contract_reuses_mature_real4k_protocol_without_score_gate():
    assert subject.ARM_ORDER == ("aligned", "fixed_permuted")
    assert subject.UPDATES == subject.rde.UPDATES == 4_000
    assert subject.SCHEDULER_T_MAX == subject.rde.SCHEDULER_T_MAX
    assert subject.EMA_DECAY == subject.rde.EMA_DECAY
    assert subject.AUGMENTATION == subject.rde.dense4k.AUGMENTATION
    base = EnsembleUNet2dConfig(
        base=8, depth=4, groups=4, mean_fluctuation=True)
    assert subject.arm_config(base, "aligned") == subject.arm_config(
        base, "fixed_permuted")
    contract = subject.dry_contract()
    assert contract["score_gate"] is None
    assert contract["endpoint_selection"] is False
    assert subject.parse_args(["--dry-contract"]).dry_contract is True


def test_metrics_use_only_exact_published_point_and_sps_weights():
    metrics = subject.officialize_metrics({
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0,
        "native_sps_score": 40.0,
        "native_sps_coverage": 0.9,
        "native_sps_mean_width": 0.02,
        # This inherited proxy must never survive the formal result path.
        "point_score": -12345.0,
    })
    assert "point_score" not in metrics
    assert metrics["official_point_contribution"] == pytest.approx(
        0.46 * 90.0 + 0.10 * 80.0 + 0.09 * 70.0)
    assert metrics["official_non_time_contribution"] == pytest.approx(
        0.46 * 90.0 + 0.10 * 80.0 + 0.09 * 70.0 + 0.25 * 40.0)


def test_released_state_upgrade_allows_only_hcobf_keys_and_pairs_arms():
    base = EnsembleUNet2dConfig(
        base=8, depth=4, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8, mean_fluctuation=True)
    legacy = EnsembleUNet2d(base)
    candidate = subject.upgrade_released_start(base, legacy.state_dict(), "aligned")
    paired = subject.upgrade_released_start(
        base, legacy.state_dict(), "fixed_permuted")
    added = subject.new_state_keys(candidate, legacy.state_dict())
    assert added
    assert all(name.startswith((
        "history_operator_conditioner.", "history_operator_basis."))
        for name in added)
    assert all(torch.equal(value, paired.state_dict()[name])
               for name, value in candidate.state_dict().items())


def test_fixed_permutation_is_deterministic_bijective_and_cross_trajectory():
    keys = [(trajectory, ordinal)
            for ordinal in range(3) for trajectory in (10, 20, 30, 40)]
    first = subject.make_fixed_cross_trajectory_permutation(keys)
    assert first == subject.make_fixed_cross_trajectory_permutation(keys)
    assert sorted(first) == list(range(len(keys)))
    assert all(keys[index][1] == keys[donor][1]
               and keys[index][0] != keys[donor][0]
               for index, donor in enumerate(first))
    record = subject.permutation_record(keys)
    assert record["mapping"] == list(first)
    assert len(record["mapping_sha256"]) == 64


def test_permuted_override_uses_already_aligned_donor_history_and_cleans_up():
    torch.manual_seed(213)
    model = _model()
    history = torch.randn(3, 20, 16, 32, 3)
    donor_history = history.index_select(0, torch.tensor([1, 2, 0]))
    expected = model.history_operator_condition(donor_history)
    with subject.fixed_permuted_override(model, donor_history):
        assert torch.equal(model._history_operator_condition_override, expected)
    assert model._history_operator_condition_override is None


class _IndexedDataset:
    def __init__(self, count: int = 8):
        self.values = [torch.full((20, 16, 32, 3), float(index))
                       for index in range(count)]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], torch.zeros(20, 16, 32, 3)


def _normalizer():
    return SimpleNamespace(mean_input=torch.zeros(3), std_input=torch.ones(3))


def test_training_callback_resolves_global_sample_indices_not_batch_positions():
    dataset = _IndexedDataset()
    mapping = (1, 0, 3, 2, 5, 4, 7, 6)
    callback = subject.TrainingConditionCallback(
        "fixed_permuted", dataset, mapping)
    # Recipient indexes 6 and 2 are intentionally larger than this batch.
    batch = (torch.stack((dataset[6][0], dataset[2][0])),
             torch.zeros(2, 20, 16, 32, 3), torch.tensor([0, 1]),
             torch.tensor([6, 2]))
    histories, recipients, donors = callback._histories(batch)
    assert recipients == [6, 2]
    assert donors == [7, 3]
    assert torch.equal(histories[0], dataset[7][0])
    assert torch.equal(histories[1], dataset[3][0])


def test_small4k_default_context_is_rng_and_value_bitwise_and_callback_once():
    small4k = subject.small4k
    model = _model()
    normalizer = _normalizer()
    device = torch.device("cpu")
    batch = (torch.ones(1),) * 4
    torch.manual_seed(18)
    before = torch.get_rng_state().clone()
    with small4k.batch_forward_context(
            None, 1, model, batch, normalizer, device):
        default_value = torch.arange(7).square()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(default_value, torch.arange(7).square())

    calls = []

    @contextmanager
    def callback(iteration, callback_model, callback_batch,
                 callback_normalizer, callback_device):
        calls.append(("enter", iteration, callback_model is model,
                      callback_batch is batch))
        yield
        calls.append(("exit", callback_device, callback_normalizer is normalizer))

    with small4k.batch_forward_context(
            callback, 3, model, batch, normalizer, device):
        calls.append(("body",))
    assert calls == [
        ("enter", 3, True, True), ("body",),
        ("exit", device, True)]


def test_update0_five_outputs_are_bitwise_for_native_and_permuted_override():
    torch.manual_seed(219)
    model = _model()
    dataset = _IndexedDataset(4)
    mapping = (1, 0, 3, 2)
    native_sha, native = subject.update0_signature(
        model, dataset, mapping, _normalizer(), torch.device("cpu"), "native")
    permuted_sha, permuted = subject.update0_signature(
        model, dataset, mapping, _normalizer(), torch.device("cpu"),
        "fixed_permuted")
    assert native_sha == permuted_sha
    assert tuple(native) == (
        "prediction", "members", "quantiles", "lower", "upper")
    assert all(torch.equal(native[key], permuted[key]) for key in native)


def test_condition_graph_has_staggered_u1_u2_gradients_through_callback():
    torch.manual_seed(220)
    model = _model().train()
    with torch.no_grad():
        for heads in model.member_point_heads:
            for head in heads:
                channels = model.config.out_channels
                head.weight[channels:].normal_(std=0.03)
                head.bias[channels:].normal_(std=0.03)
    dataset = _IndexedDataset(4)
    callback = subject.TrainingConditionCallback(
        "fixed_permuted", dataset, (1, 0, 3, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sample_indices = torch.tensor([0, 2])
    history = torch.stack([dataset[index][0] for index in sample_indices])
    batch = (history, torch.zeros_like(history), torch.tensor([0, 1]),
             sample_indices)
    for update in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        with callback(update, model, batch, _normalizer(), torch.device("cpu")):
            model.forward_distribution(history)[0].square().mean().backward()
        optimizer.step()
    assert callback.gradient_audit == {
        "u1_projection_nonzero": True,
        "u1_condition_nonzero": False,
        "u1_gate_nonzero": False,
        "u2_projection_nonzero": True,
        "u2_condition_nonzero": True,
        "u2_gate_nonzero": True,
    }
    assert callback.rng_neutral


def test_hcobf_complex_spectral_parameter_trains_and_ema_seals_strictly():
    torch.manual_seed(221)
    model = _model().train()
    with torch.no_grad():
        for heads in model.member_point_heads:
            for head in heads:
                channels = model.config.out_channels
                head.weight[channels:].normal_(std=0.03)
                head.bias[channels:].normal_(std=0.03)
    ema = subject.small4k.ema_lib.FP32EMAShadow(model, decay=0.5)
    spectral_names = [
        name for name, _parameter in model.named_parameters()
        if name.endswith("spectral_weight")]
    assert len(spectral_names) == 3
    assert all(ema.shadow_parameter_state()[name].dtype == torch.complex64
               for name in spectral_names)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = torch.randn(2, 20, 16, 32, 3)
    raw_after_u1 = None
    for update in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        model.forward_distribution(history)[0].square().mean().backward()
        spectral = [basis.spectral_weight for basis
                    in model.history_operator_basis]
        if update == 1:
            assert all(parameter.grad is not None
                       and not bool(torch.count_nonzero(parameter.grad))
                       for parameter in spectral)
        else:
            assert all(parameter.grad is not None
                       and bool(torch.count_nonzero(parameter.grad))
                       for parameter in spectral)
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=update)
        if update == 1:
            raw_after_u1 = {
                name: model.state_dict()[name].detach().clone()
                for name in spectral_names}
    assert raw_after_u1 is not None
    assert any(not torch.equal(model.state_dict()[name], raw_after_u1[name])
               for name in spectral_names)
    ema_state = ema.shadow_parameter_state()
    assert all(ema_state[name].dtype == torch.complex64
               for name in spectral_names)
    assert any(not torch.equal(ema_state[name], model.state_dict()[name])
               for name in spectral_names)
    seal = ema.seal(model)
    restored = _model()
    restored.load_state_dict(seal["model_state"], strict=True)
    rehydrated = subject.small4k.ema_lib.FP32EMAShadow.from_seal(model, seal)
    assert all(torch.equal(
        rehydrated.shadow_parameter_state()[name], seal["model_state"][name])
        for name in spectral_names)


def test_tail_manifest_and_h5_are_after_four_seals_in_formal_source(tmp_path):
    registry = subject.rde.EndpointRegistry(subject.ARM_ORDER)
    with pytest.raises(RuntimeError, match="all 4 raw/EMA endpoints"):
        subject.rde.TailAudit(registry)
    for arm in subject.ARM_ORDER:
        for kind in subject.ENDPOINT_ORDER:
            path = tmp_path / f"{arm}_{kind}.pth"
            path.write_bytes(f"{arm}:{kind}".encode())
            registry.seal(arm, kind, {
                "path": str(path),
                "file_sha256": subject.dr1.sha256(path),
            })
    audit = subject.rde.TailAudit(registry)
    audit.before_open({"file": "tail.h5"}, "held")
    audit.record_open({"file": "tail.h5"}, "held")
    audit.after_load()

    source = inspect.getsource(subject.run)
    complete = source.index("if not registry.complete()")
    manifest_hash = source.index(
        "evaluation_manifest_sha256 = dr1.sha256(args.evaluation_manifest)")
    tail_mapping = source.index("tail_mapping = permutation_record(tail_keys)")
    tail_h5 = source.index("tail_trajectories, tail_metadata, tail_windows")
    assert complete < manifest_hash < tail_mapping < tail_h5
