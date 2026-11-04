from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "code", ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

PATH = ROOT / "runs/private_audit/b210_wide_base_contract.py"
SPEC = importlib.util.spec_from_file_location("wide_contract", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

import train_b210_np_functional_sim as b210  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d  # noqa: E402


class FakeModule:
    EXPECTED_PARAMETERS = b210.EXPECTED_PARAMETERS
    EXPECTED_CONTROLLER_PARAMETERS = b210.EXPECTED_CONTROLLER_PARAMETERS
    b210_config = staticmethod(b210.b210_config)


def test_only_base_and_count_contract_change():
    fake = FakeModule()
    legacy = fake.b210_config("candidate")
    MODULE.install_wide_base(fake)
    wide = fake.b210_config("candidate")
    assert legacy.base == 210
    assert wide.base == MODULE.WIDE_BASE
    assert wide.__dict__ == {**legacy.__dict__, "base": MODULE.WIDE_BASE}
    assert fake.EXPECTED_PARAMETERS == MODULE.EXPECTED_PARAMETERS
    assert fake.EXPECTED_CONTROLLER_PARAMETERS == 8_129


def test_exact_meta_parameter_contract():
    fake = FakeModule()
    MODULE.install_wide_base(fake)
    total, controller = MODULE.audit_parameter_count(
        EnsembleUNet2d, fake.b210_config("candidate"))
    assert total == MODULE.EXPECTED_PARAMETERS
    assert controller == MODULE.EXPECTED_CONTROLLER_PARAMETERS
