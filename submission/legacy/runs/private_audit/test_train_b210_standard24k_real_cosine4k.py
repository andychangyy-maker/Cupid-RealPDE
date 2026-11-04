from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_standard24k_real_cosine4k.py"
SPEC = importlib.util.spec_from_file_location("standard24_cosine4k", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_arm_schedule_is_single_variable():
    assert MODULE.scheduler_for_output(Path("run/control")) == 12_000
    assert MODULE.scheduler_for_output(Path("run/candidate")) == 4_000
    with pytest.raises(ValueError, match="infer paired arm"):
        MODULE.scheduler_for_output(Path("run/other"))


def test_bound_trainer_injects_and_records_horizon():
    calls = []

    def trainer(*args, **kwargs):
        calls.append(kwargs["scheduler_t_max"])
        return ({"scheduler_t_max": kwargs["scheduler_t_max"]}, {"ok": True})

    bound = MODULE.bind_schedule_trainer(trainer)
    prefix = (None,) * 6
    suffix = (None,) * 2
    control = bound(*prefix, Path("x/control"), *suffix)
    candidate = bound(*prefix, Path("x/candidate"), *suffix)
    assert control[0]["scheduler_t_max"] == 12_000
    assert candidate[0]["scheduler_t_max"] == 4_000
    assert calls == [12_000, 4_000]

    with pytest.raises(ValueError, match="owned"):
        bound(*prefix, Path("x/control"), *suffix,
              scheduler_t_max=4_000)
