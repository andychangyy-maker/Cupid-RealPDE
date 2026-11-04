"""Tiny CPU integration contracts: real Native-UQ APIs, no CUDA or real data."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

TRAINING = Path(__file__).resolve().parents[1]
for path in (TRAINING, TRAINING.parent):
    sys.path.insert(0, str(path))
import train_b210_transfer_uq as subject
import train_b210_small4k_uq_release as native
import train_b210_full_prefix_uq_refit as prefix
import train_b210_np_functional_sim as sim


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def tiny_model(seed):
    torch.random.default_generator.manual_seed(seed)
    config = replace(sim.b210_config("candidate"), base=4, groups=2,
                     uq_hidden_channels=4, temporal_uq_channels=4,
                     physical_energy_hidden_channels=4)
    return sim.EnsembleUNet2d(config).float()


def state_copy(model, uq):
    return {key: value.clone() for key, value in native.partition_state(model, uq=uq).items()}


def assert_same_state(expected, actual):
    assert set(expected) == set(actual)
    assert all(torch.equal(value, actual[key]) for key, value in expected.items())


def test_same_seed_uq_initialization_preserves_point_state_and_cpu_rng():
    first, second = tiny_model(1), tiny_model(2)
    frozen = [state_copy(model, False) for model in (first, second)]
    original_uq = [state_copy(model, True) for model in (first, second)]
    before_rng = torch.get_rng_state().clone()
    for model in (first, second):
        subject.initialize_uq_state(model, native.UQ_PREFIXES)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert_same_state(state_copy(first, True), state_copy(second, True))
    for model, expected in zip((first, second), frozen):
        assert_same_state(expected, native.partition_state(model, uq=False))
    for model, expected in zip((first, second), original_uq):
        assert any(not torch.equal(value, model.state_dict()[key]) for key, value in expected.items())
    assert any(not torch.equal(value, frozen[1][key]) for key, value in frozen[0].items())


def test_point_initialization_retains_loaded_uq_and_point_state_and_cpu_rng():
    model = tiny_model(8)
    before_point, before_uq = state_copy(model, False), state_copy(model, True)
    before_rng = torch.get_rng_state().clone()
    subject.initialize_uq_state(model, native.UQ_PREFIXES, "point")
    assert torch.equal(before_rng, torch.get_rng_state())
    assert_same_state(before_uq, native.partition_state(model, uq=True))
    assert_same_state(before_point, native.partition_state(model, uq=False))


def test_seed19_reset_changes_uq_without_changing_point_or_training_rng():
    model = tiny_model(8)
    before_point = state_copy(model, False)
    subject.initialize_uq_state(model, native.UQ_PREFIXES, seed=18)
    seed18 = state_copy(model, True)
    rng = torch.get_rng_state().clone()
    subject.initialize_uq_state(model, native.UQ_PREFIXES, seed=19)
    assert torch.equal(rng, torch.get_rng_state())
    assert_same_state(before_point, native.partition_state(model, uq=False))
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in seed18.items())
    args = subject.parse_args(['--label', 'fresh', '--seed', '19'])
    assert subject.plan(args)['initialization'] == 'fresh_same_seed19'
    assert subject.plan(args)['seed'] == 19
    with pytest.raises(SystemExit):
        subject.parse_args(['--label', 'raw', '--point-label', 'C', '--initialization',
                            'point', '--uq-ema999-pair', '--seed', '19'])


@pytest.mark.parametrize("initialization", ("fresh", "point"))
def test_sharp2_real_api_backward_and_optimizer_step_freeze_non_uq_state(initialization):
    model = tiny_model(8)
    normalizer = native.smu1.FoldNormalizer(
        (torch.tensor([.3, -.1, 0.]), torch.tensor([.2, -.2, 0.]),
         torch.tensor([.08, .05, 1.]), torch.tensor([.09, .04, 1.])),
        torch.device("cpu"))
    native.ensemble_train.common.set_model_stats(model, normalizer)
    subject.initialize_uq_state(model, native.UQ_PREFIXES, initialization)
    parameters = native.configure_uq_only(model)
    native.set_uq_training_mode(model)
    before_point, before_uq = state_copy(model, False), state_copy(model, True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
    history = torch.randn(2, 20, 16, 16, 3) * .1
    target = torch.randn_like(history) * .1
    history[..., 2] = target[..., 2] = 0.
    x, y = normalizer.preprocess(history, target)
    prediction, quantiles, extents, logits = native.uq_train.quantile_forward(model, x)
    loss, components = native.uq_train.uq_objective(
        native.sharp2_args(), model, prediction, quantiles, y, normalizer,
        interval_extents=extents, coverage_logits=logits)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert components
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    assert torch.isfinite(norm) and norm > 0
    assert all(parameter.grad is None for key, parameter in model.named_parameters()
               if not key.startswith(native.UQ_PREFIXES))
    optimizer.step()
    assert_same_state(before_point, native.partition_state(model, uq=False))
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before_uq.items())
    with torch.no_grad():
        after_prediction = native.uq_train.quantile_forward(model, x)[0]
    assert torch.equal(prediction.detach(), after_prediction)
    assert not model.training and not model.energy_controller.training


def test_point_receipt_payload_and_flat_normalizer_interfaces(tmp_path):
    receipt = {"schema": subject.POINT_RECEIPT_SCHEMA, "arm": "C", "updates": 4000,
               "ema_decay": .999, "tail_read": False, "seed": 4,
               "start_sim_sha256": "1" * 64, "sample_stream_sha256": "2" * 64,
               "checkpoint_path": "fixture.pth", "checkpoint_sha256": "3" * 64}
    normalizer = native.smu1.FoldNormalizer(
        (torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3)), torch.device("cpu"))
    normalizer_report = native.ft12.normalizer_payload(normalizer)
    point = {"schema": subject.POINT_SCHEMA, "phase": "real", "iteration": 4000,
             "endpoint": "ema_u4000", "primary": True,
             "ema": {"updates": 4000, "decay": .999},
             "start": {"transfer_arm": "C", "checkpoint_sha256": "1" * 64},
             "sample_stream_sha256": "2" * 64,
             "normalizer": normalizer_report["values"],
             "normalizer_sha256": normalizer_report["sha256"]}
    receipt_path = tmp_path / "point_endpoint.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    binding = ["--point-receipt", str(receipt_path),
               "--point-receipt-sha256", subject.sha256(receipt_path)]
    default_args = subject.parse_args(["--label", "C", *binding])
    assert default_args.initialization == "fresh" and default_args.point_label == "C"
    assert subject.load_point_receipt(default_args) == receipt
    for initialization, label in (("fresh", "fresh"), ("point", "inherit")):
        args = subject.parse_args(["--label", label, "--point-label", "C",
                                   "--initialization", initialization, *binding])
        loaded = subject.load_point_receipt(args)
        subject.validate_point_payload(point, loaded)
        assert loaded == receipt
        assert subject.plan(args)["point_source_label"] == "C"
        assert subject.plan(args)["initialization"] == subject.INITIALIZATIONS[initialization]
    for source_args in ([], ["--point-label", "A"]):
        args = subject.parse_args(["--label", "inherit", *source_args, *binding])
        with pytest.raises(ValueError, match="point receipt"):
            subject.load_point_receipt(args)
    restored, report = prefix.normalizer_from_point(point, torch.device("cpu"))
    assert report == normalizer_report
    assert torch.equal(restored.std_target, normalizer.std_target)
    with pytest.raises(ValueError, match="point receipt"):
        subject.validate_point_receipt({**receipt, "schema": "other"}, "C")
    with pytest.raises(ValueError, match="does not match"):
        subject.validate_point_payload({**point, "start": {**point["start"], "transfer_arm": "A"}}, receipt)
    with pytest.raises(ValueError, match="does not match"):
        subject.validate_point_payload({**point, "schema": subject.SCHEMA + "_endpoint",
                                        "iteration": 3200}, receipt)
    with pytest.raises(ValueError, match="does not match"):
        subject.validate_point_payload({**point, "sample_stream_sha256": "4" * 64}, receipt)
    with pytest.raises(RuntimeError, match="normalizer SHA changed"):
        prefix.normalizer_from_point({**point, "normalizer_sha256": "5" * 64}, torch.device("cpu"))


def test_weighted_pinball_uses_point_weights_quantile_denominator_mask_and_detach():
    normalizer = native.smu1.FoldNormalizer(
        (torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3)), torch.device("cpu"))
    levels = (.01, .05, .15, .5, .85, .95, .99)
    model = SimpleNamespace(config=SimpleNamespace(quantile_levels=levels))
    target = torch.zeros(2, 20, 8, 32, 3)
    target[0, ..., :2] = .02
    target[1, ..., :2] = .04
    target[:, :, 0, 0, 0] = 0.0
    prediction = torch.full_like(target, .01, requires_grad=True)
    quantiles = torch.full((*target.shape[:-1], 2, 7), .02, requires_grad=True)
    loss = subject.sps_weighted_quantile_loss(model, prediction, quantiles, target, normalizer)
    # Every scored residual is exactly .01 from every predicted quantile;
    # the seven symmetric levels average to .5, including their denominator.
    counts = (target[..., :2] != 0).sum(dim=(0, 1, 2, 3))
    expected = .005 * (counts / torch.tensor([.01075, .00550])).sum() / counts.sum()
    torch.testing.assert_close(loss, expected)
    changed = quantiles.detach().clone()
    changed[:, :, 0, 0, 0] = 999.0
    assert torch.equal(loss.detach(), subject.sps_weighted_quantile_loss(
        model, prediction, changed, target, normalizer).detach())
    loss.backward()
    assert prediction.grad is None
    assert torch.count_nonzero(quantiles.grad[:, :, 0, 0, 0]) == 0
    # Equal sample counts give an unweighted median a flat interval [.01,.03].
    # The more accurate first point gets higher actual SPS weight, selecting .01.
    factors = native.ensemble_train.common.sample_sps_factor(
        prediction.detach()[..., :2], target[..., :2])
    assert factors[0] > factors[1] > 0
    assert quantiles.grad[..., 3].sum() > 0


def test_weighted_removal_strict_reload_and_real_backward_preserve_point():
    model = tiny_model(8)
    config = {"architecture": "ensemble_unet", **asdict(model.config)}
    original = {key: value.clone() for key, value in model.state_dict().items()}
    before_point, before_uq = state_copy(model, False), state_copy(model, True)
    subject.initialize_uq_state(model, native.UQ_PREFIXES, "point")
    assert_same_state(before_uq, native.partition_state(model, uq=True))
    output_config = subject.remove_utility_refinement(model, config)
    assert config["utility_refinement"] is True and config["utility_refinement_log_limit"] > 0
    assert output_config == {**config, "utility_refinement": False}
    assert model.config.utility_refinement is False and model.utility_refinement_head is None
    retained = {key: value for key, value in original.items()
                if not key.startswith(subject.REFINEMENT_PREFIX)}
    assert len(retained) < len(original)
    assert_same_state(retained, model.state_dict())
    assert_same_state(before_point, native.partition_state(model, uq=False))
    reloaded = type(model)(model.config).float()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        reloaded.load_state_dict(original, strict=True)
    with pytest.raises(ValueError, match="original utility refinement"):
        subject.remove_utility_refinement(model, output_config)
    normalizer = native.smu1.FoldNormalizer(
        (torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3)), torch.device("cpu"))
    parameters = native.configure_uq_only(model)
    native.set_uq_training_mode(model)
    before_retained = state_copy(model, True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
    x = torch.randn(2, 20, 16, 16, 3) * .01
    y = torch.randn_like(x) * .01
    x[..., 2] = y[..., 2] = 0
    prediction, quantiles, _extents, _logits = native.uq_train.quantile_forward(model, x)
    loss = subject.sps_weighted_quantile_loss(model, prediction, quantiles, y, normalizer)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    assert torch.isfinite(loss) and torch.isfinite(norm) and norm > 0
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith(native.UQ_PREFIXES))
    optimizer.step()
    assert_same_state(before_point, native.partition_state(model, uq=False))
    assert any(not torch.equal(value, model.state_dict()[name])
               for name, value in before_retained.items())
    with torch.no_grad():
        assert torch.equal(prediction, native.uq_train.quantile_forward(model, x)[0])


def test_weighted_cli_and_exact_source_receipt_rejection(tmp_path):
    arguments = ["--objective", subject.WEIGHTED_OBJECTIVE, "--label", "weighted",
                 "--point-label", "C", "--initialization", "point"]
    args = subject.parse_args(arguments)
    assert subject.plan(args)["objective_kind"] == subject.WEIGHTED_OBJECTIVE
    assert subject.parse_args(["--label", "C"]).objective == "sharp2"
    for key, value in (("--label", "C"), ("--point-label", "A"), ("--initialization", "fresh")):
        invalid = list(arguments)
        invalid[invalid.index(key) + 1] = value
        with pytest.raises(SystemExit):
            subject.parse_args(invalid)
    receipt = {"schema": subject.POINT_RECEIPT_SCHEMA, "arm": "C", "updates": 4000,
               "ema_decay": .999, "tail_read": False, "seed": 4,
               "start_sim_sha256": "1" * 64, "sample_stream_sha256": "2" * 64,
               "checkpoint_path": "fixture.pth", "checkpoint_sha256": subject.C_POINT_CHECKPOINT_SHA}
    path = tmp_path / "point_endpoint.json"
    path.write_text(json.dumps(receipt))
    args.point_receipt, args.point_receipt_sha256 = path, subject.sha256(path)
    # A valid C-shaped receipt, even claiming the right checkpoint, is not the bound receipt.
    with pytest.raises(ValueError, match="exact original C"):
        subject.load_point_receipt(args)


def test_uq_ema_tracks_post_optimizer_states_without_changing_raw_point_or_buffers():
    from copy import deepcopy

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.point = torch.nn.Linear(2, 1)
            self.point.requires_grad_(False)
            self.quantile_head = torch.nn.Linear(2, 2)
            self.quantile_head.register_buffer("fixed_level", torch.tensor(.17))
            self.register_buffer("normalizer", torch.tensor([.13, .27]))

    torch.manual_seed(23)
    model = Tiny()
    baseline = deepcopy(model)
    rng = torch.get_rng_state().clone()
    view, ema = subject.make_uq_ema(model, ("quantile_head.",))
    assert torch.equal(rng, torch.get_rng_state())
    assert set(dict(view.named_parameters())) == {"quantile_head.weight", "quantile_head.bias"}
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    optimizers = [torch.optim.AdamW(m.quantile_head.parameters(), lr=.01, weight_decay=1e-5)
                  for m in (baseline, model)]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3) for opt in optimizers]
    post_optimizer = []
    x = torch.tensor([[.4, -.7], [.3, .1]])
    for step in (1, 2, 3):
        for current, optimizer, scheduler in zip((baseline, model), optimizers, schedulers):
            optimizer.zero_grad(set_to_none=True)
            loss = (current.quantile_head(x) - current.point(x).detach() - .1 * step).square().mean()
            loss.backward()
            optimizer.step()
            if current is model:
                post_optimizer.append(current.quantile_head.weight.detach().clone())
                ema.update_after_optimizer_step(view, optimizer_step=step)
            scheduler.step()
    raw = {name: value.clone() for name, value in model.state_dict().items()}
    averaged, report = subject.materialize_uq_ema(raw, view, ema)
    # Independent closed-form oracle distinguishes post-update from pre-update EMA.
    expected = (.999 ** 3 * initial["quantile_head.weight"]
                + .001 * (.999 ** 2 * post_optimizer[0] + .999 * post_optimizer[1] + post_optimizer[2]))
    torch.testing.assert_close(averaged["quantile_head.weight"], expected, rtol=1e-6, atol=1e-7)
    assert not torch.equal(raw["quantile_head.weight"], averaged["quantile_head.weight"])
    assert_same_state(baseline.state_dict(), model.state_dict())
    assert_same_state(raw, model.state_dict())
    for name, value in initial.items():
        if name not in dict(view.named_parameters()):
            assert torch.equal(value, raw[name]) and torch.equal(value, averaged[name])
    baseline.load_state_dict(averaged, strict=True)
    assert torch.equal(model.point(x), baseline.point(x))
    assert report["updates"] == 3 and report["decay"] == .999
    assert report["update_order"] == "after_optimizer_step_before_scheduler_step"
    with pytest.raises(ValueError, match="expected optimizer_step=4"):
        ema.update_after_optimizer_step(view, optimizer_step=3)
    assert "paired_ema" not in subject.plan(subject.parse_args(["--label", "C"]))
    paired = subject.parse_args(["--uq-ema999-pair", "--label", "raw", "--point-label", "C",
                                 "--initialization", "point"])
    assert subject.plan(paired)["paired_ema"]["labels"] == ["raw", "ema"]
    with pytest.raises(SystemExit):
        subject.parse_args(["--uq-ema999-pair", "--label", "raw", "--point-label", "C"])


def test_full_real_point_cannot_masquerade_as_prefix_point():
    r = dict(schema=subject.POINT_RECEIPT_SCHEMA, arm='C', updates=4000, ema_decay=.999,
             tail_read=True, seed=4, start_sim_sha256='sim', sample_stream_sha256='stream',
             checkpoint_path='point', checkpoint_sha256='sha', held_aoa=0,
             optimizer_support='full_other_four_trajectories', normalizer_support='other_four_prefixes',
             held_data_read=False, source_commit='a'*40)
    with pytest.raises(ValueError): subject.validate_point_receipt(r, 'C')
    subject.validate_point_receipt(r, 'C', True)
    r['held_data_read'] = True
    with pytest.raises(ValueError): subject.validate_point_receipt(r, 'C', True)


def test_final_point_cannot_claim_independent_holdout_or_tail_normalizer():
    receipt = {"schema": subject.POINT_RECEIPT_SCHEMA, "arm": "C", "updates": 4000,
               "ema_decay": .999, "tail_read": True, "seed": 4,
               "start_sim_sha256": "1" * 64, "sample_stream_sha256": "2" * 64,
               "checkpoint_path": "point.pth", "checkpoint_sha256": "3" * 64,
               "held_aoa": None, "optimizer_support": "full_clean81_trajectories",
               "normalizer_support": "clean81_prefix2544", "final_fit": True,
               "clean716_overlaps_point_training": True, "source_commit": "4" * 40}
    subject.validate_point_receipt(receipt, 'C', True, True)
    for field, value in [('held_aoa', 0), ('normalizer_support', 'full_clean81'),
                         ('clean716_overlaps_point_training', False)]:
        with pytest.raises(ValueError):
            subject.validate_point_receipt({**receipt, field: value}, 'C', True, True)
    with pytest.raises(ValueError):
        subject.validate_point_receipt(receipt, 'C', True)
