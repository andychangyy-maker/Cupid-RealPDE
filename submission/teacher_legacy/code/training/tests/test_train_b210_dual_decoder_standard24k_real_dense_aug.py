from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "runs" / "private_audit"
    / "train_b210_dual_decoder_standard24k_real_dense_aug.py")
SPEC = importlib.util.spec_from_file_location("dual_decoder_24k_real", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("mode", tuple(MODULE.CONTRACTS))
def test_cli_mode_is_required_and_removed(mode):
    argv = ["runner", "--output-dir", "out", "--dual-decoder-mode", mode]
    assert MODULE.extract_mode(argv) == mode
    assert argv == ["runner", "--output-dir", "out"]
    with pytest.raises(ValueError, match="required"):
        MODULE.extract_mode(["runner"])


@pytest.mark.parametrize("mode", tuple(MODULE.CONTRACTS))
def test_validator_binds_exact_short_endpoint(mode):
    calls = []

    def validator(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    bound = MODULE.bind_validator(validator, mode)
    assert bound("checkpoint", "sha", "normalizer", "cuda") == "ok"
    assert calls == [(("checkpoint", "sha", "normalizer", "cuda"), {
        "expected_iteration": MODULE.EXPECTED_ITERATION,
        "expected_status": MODULE.CONTRACTS[mode]["status"],
    })]


@pytest.mark.parametrize("mode", tuple(MODULE.CONTRACTS))
def test_architecture_contract_checks_role_mode_and_parameter_count(mode):
    expected = MODULE.CONTRACTS[mode]["parameters"]

    class Config:
        dual_decoder_mode = mode

    bound = MODULE.bind_architecture_contract(
        lambda *_args: (Config(), expected, 17), mode)
    assert bound(False, False)[1:] == (expected, 17)

    class WrongConfig:
        dual_decoder_mode = "wrong"

    wrong_mode = MODULE.bind_architecture_contract(
        lambda *_args: (WrongConfig(), expected, 17), mode)
    with pytest.raises(RuntimeError, match="differs from CLI"):
        wrong_mode(False, False)

    wrong_parameters = MODULE.bind_architecture_contract(
        lambda *_args: (Config(), expected + 1, 17), mode)
    with pytest.raises(RuntimeError, match="parameter contract"):
        wrong_parameters(False, False)
