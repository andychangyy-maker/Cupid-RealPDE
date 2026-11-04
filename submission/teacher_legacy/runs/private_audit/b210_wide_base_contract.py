"""Install an audited wider-base contract into the frozen B210 trainer."""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType

import torch


WIDE_BASE = 240
EXPECTED_PARAMETERS = 232_160_565
EXPECTED_CONTROLLER_PARAMETERS = 8_129


def install_wide_base(b210: ModuleType) -> None:
    """Patch only the base width and its exact parameter-count contract."""
    if getattr(b210, "_wide_base_contract_installed", False):
        raise RuntimeError("wide-base contract was installed twice")
    original_config = b210.b210_config

    def wide_config(*args, **kwargs):
        config = original_config(*args, **kwargs)
        if int(config.base) != 210:
            raise RuntimeError("frozen source architecture is no longer B210")
        return replace(config, base=WIDE_BASE)

    b210.b210_config = wide_config
    b210.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    b210.EXPECTED_CONTROLLER_PARAMETERS = EXPECTED_CONTROLLER_PARAMETERS
    b210._wide_base_contract_installed = True


def audit_parameter_count(model_type, config) -> tuple[int, int]:
    """Count a meta-device B240 model without allocating its full weights."""
    with torch.device("meta"):
        model = model_type(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    controller = sum(
        parameter.numel() for parameter in model.energy_controller.parameters())
    return total, controller
