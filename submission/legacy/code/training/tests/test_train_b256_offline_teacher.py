"""CPU checks for B256 identity, fixed recipes, budget refusal and no probe continuation."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

TRAINING = Path(__file__).resolve().parents[1]
for path in (TRAINING, TRAINING.parent):
    sys.path.insert(0, str(path))
import train_b256_offline_teacher as subject
import train_b210_np_functional_sim as sim
import train_b210_np_functional_real_small4k_confirm_held0 as small4k


def test_plan_opens_no_data_and_does_not_import_torch(tmp_path):
    command = ("import runpy,sys;sys.path.insert(0,sys.argv[1]);"
               "sys.argv=[sys.argv[2],'--phase','preflight','--output-dir',sys.argv[3]];"
               "runpy.run_path(sys.argv[0],run_name='__main__');assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-B", "-c", command, str(TRAINING), subject.__file__,
                             str(tmp_path / "new")], check=True, capture_output=True, text=True)
    assert '263574885' in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_actual_meta_architecture_and_loader_contract_are_honest_b256():
    old, new = sim.b210_config("candidate"), subject.model_config(sim)
    assert {key for key in asdict(old) if asdict(old)[key] != asdict(new)[key]} == {"base"}
    with torch.device("meta"):
        baseline, model = sim.EnsembleUNet2d(old), sim.EnsembleUNet2d(new)
    assert sum(p.numel() for p in model.parameters()) == subject.PARAMETERS
    assert sum(p.numel() for p in model.energy_controller.parameters()) == subject.CONTROLLER_PARAMETERS
    norms = lambda m: {n: x.num_groups for n, x in m.named_modules() if isinstance(x, torch.nn.GroupNorm)}
    before, after = norms(baseline), norms(model)
    assert len([name for name in before if before[name] != after[name]]) == 5
    assert {(before[name], after[name]) for name in before if before[name] != after[name]} == {(7, 8)}
    assert small4k.ft12.validate_b210_model_config(model.architecture_config(), new) == new
    with pytest.raises(ValueError):
        small4k.ft12.validate_b210_model_config(model.architecture_config())


def test_sim_recipe_reuses_original_teacher_on_native_settings():
    args = subject.parse_args(["--phase", "sim", "--teacher-cache", "original.pt"])
    configured = subject.pure.training_args(sim, args)
    assert (configured.seed, configured.batch_size, configured.updates) == (0, 8, 10000)
    assert (configured.lr, configured.weight_decay, configured.distill_weight) == (3e-4, 1e-5, .005)
    assert configured.soft_sps_proxy_real_scale is False
    assert configured.input_noise_std == configured.input_gain_std == 0
    assert subject.real.real_recipe("A")["scheduler_t_max"] == 12000


def test_prior_receipt_rejects_wrong_model_teacher_probe_or_exhausted_budget():
    args = subject.parse_args(["--phase", "sim", "--source-commit", "a" * 40])
    value = {"parameters": subject.PARAMETERS, "source_commit": args.source_commit,
             "runner_sha256": subject.pure.sha256(subject.__file__),
             "schema": subject.SCHEMA + "_preflight", "decision": "PASS_HARDWARE_PREFLIGHT",
             "sealed": False, "preflight_weights_saved": False,
             "completed_updates": {"sim": 2, "real": 3}, "cumulative_wall_seconds": 10.}
    assert subject.validate_previous(value, args) == 10.
    for key, wrong in (("parameters", 178705575), ("schema", subject.pure.TEACHER_SCHEMA),
                       ("preflight_weights_saved", True), ("decision", "NEEDS_FASTER_GPU"),
                       ("cumulative_wall_seconds", 7800.), ("cumulative_wall_seconds", float("nan"))):
        with pytest.raises(ValueError):
            subject.validate_previous({**value, key: wrong}, args)
    args.phase = "real"
    value.update(schema=subject.SIM_SCHEMA, decision=subject.SIM_STATUS, phase="sim", sealed=True,
                 seed=0, batch_size=8, updates=10000, ema_updates=10000, ema_decay=.999,
                 p1_distill_weight=.005, teacher_start_update=1, teacher_end_update=10000,
                 teacher_cache_sha256=subject.pure.TEACHER_CACHE_SHA256,
                 training_visibility="sim_with_p1_teacher_no_direct_real",
                 real_arrays_read=0, real_statistics_read=0, tail_arrays_read=0,
                 sample_stream_sha256=subject.pure.OFF_STREAM_SHA256)
    subject.validate_previous(value, args)
    for key, wrong in (("p1_distill_weight", 0), ("ema_updates", 2), ("schema", subject.pure.TEACHER_SCHEMA),
                       ("real_statistics_read", 1), ("sample_stream_sha256", "0" * 64)):
        with pytest.raises(ValueError):
            subject.validate_previous({**value, key: wrong}, args)


def test_budget_requires_faster_hardware_without_changing_recipe():
    assert subject.budget_projection(.3, .4, 30, 30, 90)["decision"] == "PASS_HARDWARE_PREFLIGHT"
    assert subject.budget_projection(.8, .4, 30, 30, 90)["decision"] == "NEEDS_FASTER_GPU"
    assert subject.budget_projection(.3, .7, 30, 30, 90)["decision"] == "NEEDS_FASTER_GPU"


def test_probe_stops_before_fourth_backward_and_saves_no_endpoint(tmp_path):
    cuda = SimpleNamespace(synchronize=lambda _: None, reset_peak_memory_stats=lambda _: None,
                           max_memory_allocated=lambda _: 123)
    probe = subject.RealPreflight(SimpleNamespace(cuda=cuda), "cpu")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    updates = 0
    with pytest.raises(subject.PreflightComplete):
        for iteration in (1, 2, 3, 4, 5):
            optimizer.zero_grad(set_to_none=True)
            with probe(iteration):
                model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            updates += 1
    assert updates == 3 and len(probe.timings) == 3 and probe.seconds == probe.timings[2]
    assert probe.seconds > 0 and probe.peak_bytes == 123
    assert list(tmp_path.iterdir()) == []


def test_actual_ema_writer_uses_new_identity_and_restores_legacy_globals(tmp_path):
    class Tiny(torch.nn.Linear):
        def architecture_config(self):
            return {"test": "tiny"}
    model = Tiny(1, 1)
    ema = small4k.ema_lib.FP32EMAShadow(model, .999)
    for step in range(1, 4001):
        ema.update_after_optimizer_step(model, optimizer_step=step)
    old = (small4k.EXPERIMENT, small4k.ENDPOINT_SCHEMA, small4k.HELD_AOA)
    with subject.endpoint_contract(small4k):
        seal, _ = small4k.save_primary_endpoint(tmp_path / "point.pth", model, ema,
            start_report={"checkpoint_sha256": "a" * 64}, normalizer_report={"values": {}, "sha256": "b" * 64},
            stream_sha256="c" * 64)
    payload = torch.load(seal["path"], weights_only=False)
    assert payload["schema"] == subject.SCHEMA + "_point" and payload["experiment"] == subject.SCHEMA
    assert payload["iteration"] == 4000 and payload["held_aoa"] == -1
    assert (small4k.EXPERIMENT, small4k.ENDPOINT_SCHEMA, small4k.HELD_AOA) == old
    with pytest.raises(RuntimeError), subject.endpoint_contract(small4k):
        raise RuntimeError("test exception cleanup")
    assert (small4k.EXPERIMENT, small4k.ENDPOINT_SCHEMA, small4k.HELD_AOA) == old
