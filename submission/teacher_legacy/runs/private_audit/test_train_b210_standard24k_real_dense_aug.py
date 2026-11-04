from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_standard24k_real_dense_aug.py"
SPEC = importlib.util.spec_from_file_location("standard24_real", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_binds_only_the_short_endpoint_contract():
    calls = []

    def validator(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    bound = MODULE.bind_standard24_validator(validator)
    assert bound("checkpoint", "sha", "normalizer", "cuda") == "ok"
    assert calls == [(('checkpoint', 'sha', 'normalizer', 'cuda'), {
        "expected_iteration": 4_000,
        "expected_status": MODULE.EXPECTED_STATUS,
    })]
