"""Check the actual LR intervention and teacher provenance before GPU work."""
from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path
import subprocess
import sys

import pytest
import torch

TRAINING = Path(__file__).resolve().parents[1]
for path in (TRAINING, TRAINING.parent):
    sys.path.insert(0, str(path))
import train_b210_teacher_real as subject
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_np_functional_sim as sim
import train_b210_transfer_uq as uq


def receipt(teacher):
    pure = subject.sim_start
    value = {
        "schema": pure.TEACHER_SCHEMA if teacher else pure.SCHEMA,
        "decision": pure.TEACHER_STATUS if teacher else pure.STATUS,
        "training_visibility": "sim_with_p1_teacher_no_direct_real" if teacher else "pure_sim_only",
        "phase": "sim", "start": "random_initialization", "updates": 10000,
        "iteration": 10000, "seed": 0, "ema_decay": .999, "ema_updates": 10000,
        "batch_size": 8, "real_arrays_read": 0, "real_statistics_read": 0,
        "held_arrays_read": 0, "tail_arrays_read": 0, "real_closure": False,
        "pretext_checkpoint": None, "checkpoint_selection": False, "sealed": True,
        "teacher_reads": int(teacher), "p1_distill_weight": .005 if teacher else 0.,
        "initial_model_state_sha256": pure.OFF_INITIAL_SHA256,
        "sample_stream_sha256": pure.OFF_STREAM_SHA256,
        "checkpoint_sha256": "b" * 64 if teacher else pure.OFF_CHECKPOINT_SHA256,
    }
    if teacher:
        value.update(teacher_cache_sha256=pure.TEACHER_CACHE_SHA256,
                     teacher_cache={"sha256": pure.TEACHER_CACHE_SHA256},
                     teacher_start_update=1, teacher_end_update=10000)
    return value


def test_plan_does_not_import_torch_or_touch_data(tmp_path):
    command = ("import runpy,sys;sys.path.insert(0,sys.argv[1]);"
               "sys.argv=[sys.argv[2],'--arm','B','--real-dir',sys.argv[3]];"
               "runpy.run_path(sys.argv[0],run_name='__main__');assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-B", "-c", command, str(TRAINING),
                    subject.__file__, str(tmp_path / "absent")], check=True, capture_output=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("teacher,arms", [(True, "AB"), (False, "CD")])
def test_wrong_teacher_arm_or_visibility_rejected(teacher, arms):
    bound = receipt(teacher)
    for arm in arms:
        subject.validate_sim_receipt(bound, arm)
    with pytest.raises(ValueError, match="teacher arm"):
        subject.validate_sim_receipt(bound, "C" if teacher else "A")
    for key, changed in (("training_visibility", "pure_sim_only" if teacher else "unknown"),
                         ("initial_model_state_sha256", "0" * 64),
                         ("sample_stream_sha256", "0" * 64), ("real_statistics_read", 1)):
        with pytest.raises(ValueError, match="teacher arm"):
            subject.validate_sim_receipt({**bound, key: changed}, arms[0])
    del bound["pretext_checkpoint"]
    with pytest.raises(ValueError):
        subject.validate_sim_receipt(bound, arms[0])


def test_real_recipe_does_not_depend_on_sim_teacher():
    assert subject.real_recipe("A") == subject.real_recipe("C")
    assert subject.real_recipe("B") == subject.real_recipe("D")
    assert subject.real_recipe("A")["scheduler_t_max"] == 12000
    assert subject.real_recipe("B")["scheduler_t_max"] == 4000
    assert inspect.signature(small4k.train_candidate).parameters["parameter_groups"].default is None


def test_b210_names_are_exhaustive_disjoint_and_trainable():
    config = replace(sim.b210_config("candidate"), base=4, depth=2, groups=1,
                     uq_hidden_channels=4, mean_profile_channels=4, temporal_uq_channels=4,
                     physical_energy_hidden_channels=4)
    model = sim.EnsembleUNet2d(config)
    groups = subject.parameter_groups(model, "B")
    small4k.validate_parameter_groups(tuple(model.parameters()), groups)
    actual = {name: (parameter, group["lr"]) for group in groups
              for name, parameter in zip(group["param_names"], group["params"])}
    assert set(actual) == dict(model.named_parameters()).keys()
    for name, parameter in model.named_parameters():
        assert actual[name][0] is parameter and parameter.requires_grad
        assert actual[name][1] == (2e-5 if name.startswith(subject.SLOW_PREFIXES) else 2e-4)
    assert any(name.startswith("regime_film.") for name in groups[0]["param_names"])
    assert any(name.startswith("energy_controller.") for name in groups[1]["param_names"])


def test_parameter_coverage_rejects_missing_duplicate_foreign_or_unnamed():
    parameters = tuple(torch.nn.Linear(2, 1).parameters())
    valid = [{"name": "all", "lr": 2e-4, "params": parameters}]
    assert small4k.validate_parameter_groups(parameters, None) is parameters
    small4k.validate_parameter_groups(parameters, valid)
    for bad in ([{"name": "all", "params": parameters[:1]}],
                [{"name": "all", "params": parameters + parameters[:1]}],
                [{"name": "all", "params": parameters + (torch.nn.Parameter(torch.ones(1)),)}],
                [{"params": parameters}], [{"name": "all", "params": parameters, "lr": 0}]):
        with pytest.raises(ValueError, match="exactly once"):
            small4k.validate_parameter_groups(parameters, bad)


def test_actual_adamw_preserves_default_layered_ratio_and_uq_endpoint_binding(tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Linear(1, 1, bias=False)
            self.decoder = torch.nn.Linear(1, 1, bias=False)

        def architecture_config(self):
            return {"architecture": "test_tiny"}

    legacy = Tiny()
    default, layered = deepcopy(legacy), deepcopy(legacy)
    models = (default, legacy, layered)
    opts = [torch.optim.AdamW(
        small4k.validate_parameter_groups(tuple(model.parameters()), groups),
        lr=small4k.LR, weight_decay=small4k.WEIGHT_DECAY)
        for model, groups in zip(models, (None, subject.parameter_groups(legacy, "A"),
                                         subject.parameter_groups(layered, "B")))]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=horizon)
                  for opt, horizon in zip(opts, (12000, 12000, 4000))]
    ema = small4k.ema_lib.FP32EMAShadow(legacy, .999)
    for update in range(4000):
        for model, opt, scheduler in zip(models, opts, schedulers):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            opt.step()
            if model is legacy:
                ema.update_after_optimizer_step(model, optimizer_step=update + 1)
            scheduler.step()
        if update < 3999:
            assert opts[2].param_groups[0]["lr"] / opts[2].param_groups[1]["lr"] == pytest.approx(.1)
    for left, right in zip(default.parameters(), legacy.parameters()):
        assert torch.equal(left, right)
    assert opts[0].param_groups[0]["lr"] == pytest.approx(1.5e-4)
    assert all(group["lr"] == 0 for group in opts[2].param_groups)
    assert not torch.equal(layered.stem.weight, legacy.stem.weight)
    old = (small4k.EXPERIMENT, small4k.ENDPOINT_SCHEMA, small4k.HELD_AOA)
    with subject.transfer.endpoint_contract(small4k):
        seal, _ = small4k.save_primary_endpoint(
            tmp_path / "point.pth", legacy, ema,
            start_report={"transfer_arm": "A", "checkpoint_sha256": "a" * 64},
            normalizer_report={"values": {}, "sha256": "b" * 64}, stream_sha256="c" * 64)
    assert (small4k.EXPERIMENT, small4k.ENDPOINT_SCHEMA, small4k.HELD_AOA) == old
    point = {"schema": uq.POINT_RECEIPT_SCHEMA, "arm": "A", "updates": 4000,
             "ema_decay": .999, "seed": 4, "tail_read": False,
             "checkpoint_path": seal["path"], "checkpoint_sha256": seal["file_sha256"],
             "sample_stream_sha256": "c" * 64, "start_sim_sha256": "a" * 64}
    uq.validate_point_receipt(point, "A")
    payload = torch.load(seal["path"], map_location="cpu", weights_only=False)
    uq.validate_point_payload(payload, point)
    with pytest.raises(ValueError, match="checkpoint"):
        uq.validate_point_payload(payload, {**point, "start_sim_sha256": "d" * 64})


@pytest.mark.parametrize("budget", [None, 0, -1, 1801, float("nan")])
def test_invalid_budget_fails_before_sim_or_real_read(tmp_path, budget):
    args = subject.parse_args(["--arm", "A", "--execute", "--sim-receipt", str(tmp_path / "absent"),
                              "--sim-receipt-sha256", "0" * 64, "--real-dir", str(tmp_path),
                              "--clean-manifest", str(tmp_path / "missing"),
                              "--output-dir", str(tmp_path / "new")])
    args.wall_budget_seconds = budget
    with pytest.raises(ValueError, match="wall budget"):
        subject.run(args)
    assert list(tmp_path.iterdir()) == []
