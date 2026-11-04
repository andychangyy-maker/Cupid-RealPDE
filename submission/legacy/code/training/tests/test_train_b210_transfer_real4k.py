"""CPU contracts for transfer preparation, without full B210 or real data."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_b210_transfer_real4k as subject


@pytest.fixture(scope="module")
def dense():
    torch.set_num_threads(1)
    return subject.load_runtime()


def tiny_model(dense):
    config = replace(dense.sim_tke.b210_config("candidate"), base=4, groups=2,
                     uq_hidden_channels=4, temporal_uq_channels=4,
                     physical_energy_hidden_channels=4,
                     backbone_residual_dropout=0.)
    return dense.sim_tke.EnsembleUNet2d(config)


def test_default_plan_does_not_load_runtime_or_create_output(monkeypatch, tmp_path):
    def forbidden():
        raise AssertionError("plan imported the training runtime")
    monkeypatch.setattr(subject, "load_runtime", forbidden)
    args = subject.parse_args(["--arm", "B", "--output-dir", str(tmp_path / "out")])
    result = subject.run(args)
    assert result["decision"] == "PLAN_ONLY_NO_DATA_NO_GPU"
    assert result["auxiliary"]["weight"] == .005
    assert not args.output_dir.exists()
    assert "--start-sim-sha256" in result["missing_execution_arguments"]
    args.execute = True
    with pytest.raises(ValueError, match="missing execution bindings"):
        subject.run(args)


def test_constant_controller_tke_changes_only_fixed_real_schedule_and_pins_c(dense, tmp_path):
    import inspect
    small4k = dense.small4k
    assert inspect.signature(small4k.train_candidate).parameters["constant_controller_tke"].default is False
    legacy = [small4k.controller_tke_aux_weight(step) for step in range(1, 4001)]
    assert legacy == [dense.ft12.tke_aux_weight(step) for step in range(1, 4001)]
    assert sum(legacy) == pytest.approx(7.5)
    assert legacy[0] == .010 and legacy[1499:] == [0.] * 2501
    constant = [small4k.controller_tke_aux_weight(step, True) for step in range(1, 4001)]
    assert constant == [.010] * 4000 and sum(constant) == pytest.approx(40.)
    for step in (0, 4001):
        with pytest.raises(ValueError, match="1 through 4000"):
            small4k.controller_tke_aux_weight(step, True)
    original = subject.plan(subject.parse_args(["--arm", "C"]))
    changed = subject.plan(subject.parse_args([
        "--arm", "C", "--constant-controller-tke", "--source-commit", "a" * 40]))
    assert changed["real"] == original["real"] and changed["auxiliary"] == original["auxiliary"]
    assert changed["controller_tke_schedule"] == "constant_0p01_4000"
    assert changed["source_commit"] == "a" * 40 and "controller_tke_schedule" not in original
    with pytest.raises(SystemExit):
        subject.parse_args(["--arm", "D", "--constant-controller-tke"])
    args = subject.parse_args([
        "--arm", "C", "--constant-controller-tke", "--source-commit", "a" * 40,
        "--start-sim", str(tmp_path / "sim.pth"), "--start-sim-sha256", subject.C_START_SIM_SHA256,
        "--start-metadata", str(tmp_path / "sim.json"), "--start-metadata-sha256", subject.C_START_METADATA_SHA256,
        "--clean-manifest", "manifest.json", "--real-dir", "real", "--output-dir", str(tmp_path / "out"),
        "--wall-budget-seconds", "1500"])
    for field, value in (("start_sim_sha256", "b" * 64), ("start_metadata_sha256", "b" * 64),
                         ("source_commit", "short"), ("wall_budget_seconds", 1501)):
        previous = getattr(args, field)
        setattr(args, field, value)
        with pytest.raises(ValueError, match="exact old-C"):
            subject.validate_binding(args)  # reject before touching these nonexistent assets
        setattr(args, field, previous)


def test_binding_rejects_wrong_family_and_changed_checkpoint(tmp_path):
    checkpoint = tmp_path / "toy.bin"
    checkpoint.write_bytes(b"identity-test-only")
    metadata = tmp_path / "binding.json"
    row = {"schema": subject.METADATA_SCHEMA, "family": "standard24",
           "phase": "sim", "iteration": 4000, "status": "EXACT_STATUS",
           "checkpoint_sha256": subject.sha256(checkpoint),
           "training_visibility": "clean81_prefix_only",
           "prepared_source_commit": "d" * 40,
           "lineage": [{"kind": "fixture_only"}]}
    metadata.write_text(json.dumps(row), encoding="utf-8")
    args = subject.parse_args([
        "--arm", "A", "--start-sim", str(checkpoint),
        "--start-sim-sha256", subject.sha256(checkpoint),
        "--start-metadata", str(metadata),
        "--start-metadata-sha256", subject.sha256(metadata),
        "--clean-manifest", "manifest.json", "--real-dir", "real",
        "--output-dir", str(tmp_path / "out"), "--wall-budget-seconds", "3600"])
    assert subject.validate_binding(args) == row
    args.arm = "C"
    with pytest.raises(ValueError, match="does not bind this arm"):
        subject.validate_binding(args)
    args.arm = "A"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset SHA mismatch"):
        subject.validate_binding(args)


def test_fresh_heads_preserve_rng_and_stay_out_of_deploy_and_ema(dense, monkeypatch):
    model = tiny_model(dense)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("fresh CPU heads attempted to reseed CUDA")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    heads = subject.fresh_supervisor(model, torch.device("cpu"), dense.sim_tke)
    other = subject.fresh_supervisor(model, torch.device("cpu"), dense.sim_tke)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, other.state_dict()[key])
               for key, value in heads.state_dict().items())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert not set(map(id, heads.parameters())) & set(map(id, model.parameters()))
    ema = dense.small4k.ema_lib.FP32EMAShadow(model, .999)
    state = ema.materialize_model_state(model)
    assert set(state) == set(before)
    model.load_state_dict(state, strict=True)


def test_future_tke_uses_time_variance_and_survives_time_or_mean_shift(dense):
    truth = torch.randn(2, 20, 8, 8, 2) * .1
    tke = dense.sim_tke.MultiScaleTKEDeepSupervisor.physical_tke_map
    expected = .5 * truth.var(dim=1, unbiased=False).sum(dim=-1)
    assert torch.allclose(tke(truth), expected, atol=1e-8)
    assert torch.allclose(tke(truth.roll(7, dims=1)), expected, atol=1e-8)
    assert torch.allclose(tke(truth + 2.), expected, atol=1e-7)
    assert torch.count_nonzero(tke(torch.ones_like(truth))) == 0


def test_real_backward_zero_control_and_aux_reaches_shared_backbone(dense):
    torch.manual_seed(8)
    model = tiny_model(dense)
    heads = subject.fresh_supervisor(model, torch.device("cpu"), dense.sim_tke)
    normalizer = dense.smu1.FoldNormalizer(
        (torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3)),
        torch.device("cpu"))
    dense.ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    history = torch.randn(2, 20, 16, 16, 3) * .1
    target = torch.randn_like(history) * .1
    history[..., 2] = target[..., 2] = 0.
    batch = (history, target, torch.tensor([0, 1]), torch.tensor([0, 1]))
    def backward(arm, iteration=2):
        model.zero_grad(set_to_none=True)
        kwargs = subject.auxiliary_arguments(arm, heads, dense.sim_tke)
        kwargs.pop("gradient_ratio_diagnostics_only")
        return dense.ft12.npt1.backward_real_supervised_npt1(
            model, batch, normalizer, torch.device("cpu"), iteration,
            auxiliary_weight=0., member_mask=torch.ones(2, 4, dtype=torch.bool),
            audit_controller_gradients=False, **kwargs)
    # Cold point/member/quantile output weights are zero. Perform the real
    # first native update before auditing native feature gradients at update 2;
    # the formal runner instead starts from already-trained Sim weights.
    optimizer = torch.optim.AdamW(model.parameters(), lr=dense.small4k.LR,
                                  weight_decay=dense.small4k.WEIGHT_DECAY)
    backward("A", iteration=1)
    torch.nn.utils.clip_grad_norm_(model.parameters(), dense.small4k.GRAD_CLIP)
    optimizer.step()
    control = backward("A")
    native_gradients = {key: value.grad.clone() for key, value in model.named_parameters()
                        if value.grad is not None}
    for key in ("full_backbone_auxiliary_loss", "full_backbone_auxiliary_weight",
                "full_backbone_weighted_auxiliary_loss"):
        assert control[key] == 0.0
    assert control["real_supervised_loss"] == control["real_native_supervised_loss"]
    assert all(value.grad is None for value in heads.parameters())
    candidate = backward("B")
    assert candidate["full_backbone_gradient_finite_nonzero"] == 1.
    assert candidate["full_backbone_weighted_auxiliary_loss"] > 0.
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in heads.parameters())
    assert any(not torch.equal(value.grad, native_gradients[key])
               for key, value in model.named_parameters()
               if key.startswith(("stem.", "encoder.", "decoder."))
               and key in native_gradients)


def test_hdf5_tail_sentinel_cannot_reach_prefix_windows_or_statistics(dense, tmp_path):
    path = tmp_path / "toy.h5"
    values = np.ones((100, 4, 4), dtype=np.float32)
    values[60:] = np.nan
    with h5py.File(path, "w") as handle:
        handle["u"] = values
        handle["v"] = values * 2
        handle["re"] = 5000
        handle["aoa"] = 0
    rows = [{"file": path.name, "frames": 100, "re": 5000, "aoa": 0,
             "val_boundary": 60, "train_starts": [0, 20]}]
    audit = dense.PrefixAudit()
    trajectories, _, windows = dense.dr1.load_prefix_rows(tmp_path, rows, "fit", audit)
    assert trajectories[0].shape[0] == 60
    assert torch.isfinite(trajectories[0]).all()
    assert all(w.start + 40 <= 60 for w in dense.dense_windows(trajectories, 5))
    stats = dense.smu1.fold_statistics(
        dense.smu1.window_sufficient_statistics(trajectories, windows), range(len(windows)))
    assert all(torch.isfinite(value).all() for value in stats)
    with pytest.raises(RuntimeError, match="invalid clean-prefix open"):
        dense.PrefixAudit().before_open(rows[0], "held")
    with pytest.raises(RuntimeError, match="invalid clean-prefix open"):
        dense.PrefixAudit().before_open({**rows[0], "file": "7575_0.h5"}, "fit")


def test_full_support_reads_late_training_frames_without_opening_missing_held_file(dense, tmp_path):
    from b210_stage_fold import select_rows
    from train_b210_full_released_real_support import load_full_rows
    rows = [dict(file=f'{a}.h5', aoa=a, re=100, frames=52, val_boundary=40, train_starts=[0])
            for a in (0, 5, 10, 15, 20)]
    for row in rows[1:]:
        with h5py.File(tmp_path / row['file'], 'w') as h:
            h['u'] = np.broadcast_to(np.arange(52, dtype=np.float32)[:, None, None], (52, 4, 4))
            h['v'] = np.zeros((52, 4, 4), dtype=np.float32)
            h['aoa'], h['re'] = row['aoa'], row['re']
    selected = select_rows(rows, 0)
    prefix, _, _ = dense.dr1.load_prefix_rows(tmp_path, selected, 'fit', dense.PrefixAudit())
    audit = dense.PrefixAudit()
    full, _, = load_full_rows(tmp_path, selected, 'fit', audit)
    assert set(audit.opened) == {r['file'] for r in rows[1:]}
    assert full[0][-1, 0, 0, 0] == 51
    assert torch.equal(full[0][:40], prefix[0])
    assert len(dense.dense_windows(full, 5)) == 12
    assert len(dense.dense_windows(prefix, 5)) == 4
    with pytest.raises(SystemExit): subject.parse_args(['--arm', 'C', '--full-real-trajectories'])


def test_final_full_support_requires_explicit_mode_and_discloses_overlap():
    args = subject.parse_args(['--arm', 'C', '--full-real-trajectories', '--full-real-final'])
    report = subject.plan(args)
    assert report['optimizer_support'] == 'full_clean81_trajectories'
    assert report['normalizer_support'] == 'clean81_prefix2544'
    assert report['clean716_overlaps_point_training'] is True
    assert report['held_aoa'] is None
    for extra in (['--held-aoa', '0'], ['--arm', 'A']):
        with pytest.raises(SystemExit):
            subject.parse_args(['--arm', 'C', '--full-real-trajectories', '--full-real-final', *extra])
