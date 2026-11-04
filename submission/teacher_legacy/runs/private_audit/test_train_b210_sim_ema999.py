from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_sim_ema999.py"
SPEC = importlib.util.spec_from_file_location("sim_ema999", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("counter", torch.tensor(1, dtype=torch.int64))


def test_ema_is_initialized_exactly_and_updates_float_only():
    model = Tiny()
    ema = MODULE.initialize_ema(model)
    assert torch.equal(ema["weight"], model.weight)
    assert ema["weight"].data_ptr() != model.weight.data_ptr()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([3.0, 6.0]))
        model.counter.fill_(7)
    MODULE.update_ema(ema, model, decay=0.5)
    assert torch.allclose(ema["weight"], torch.tensor([2.0, 4.0]))
    assert int(ema["counter"]) == 7


def test_ema_state_loads_as_a_normal_model_state():
    model = Tiny()
    ema = MODULE.initialize_ema(model)
    with torch.no_grad():
        ema["weight"].add_(5.0)
    restored = Tiny()
    restored.load_state_dict(ema, strict=True)
    assert torch.equal(restored.weight, torch.tensor([6.0, 7.0]))
