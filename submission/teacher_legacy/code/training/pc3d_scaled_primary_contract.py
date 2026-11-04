"""Frozen architecture identity for the first scaled-primary PC3D run.

The production trainers deliberately keep their audited default behaviour.
These small launch adapters install a distinct schema and the one preregistered
25.64M architecture before delegating to those trainers.
"""
from __future__ import annotations

from pathlib import Path
import sys


# The launchers are executable directly from the repository root.  In that
# mode Python adds ``code/training`` but not its sibling import root ``code``.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.pyramid_causal3d_operator import PyramidCausal3DOperatorConfig


SCALED_WIDTHS = (160, 256, 448)
EXPECTED_PARAMETERS = 25_638_640
SIM_SCHEMA = "pc3d_scaled25m_sim_all100_long_stage_s_v1"
SIM_ACKNOWLEDGEMENT = "RUN_PC3D_SCALED25M_SIM_ALL100_LONG_STAGE_S"
SIM_ENDPOINT_KIND = "raw_model_u10000_scaled25m"
REAL_SCHEMA = "pc3d_scaled25m_real_clean81_dense_aug_stage_r_v1"
REAL_ACKNOWLEDGEMENT = "RUN_PC3D_SCALED25M_REAL_DENSE_AUGMENTATION"
REAL_RESUME_SCHEMA = "pc3d_scaled25m_real_dense_aug_exact_resume_v1"
REAL_ENDPOINT_SCHEMA = "pc3d_scaled25m_real_dense_aug_ema_endpoint_v1"


def model_config(*, tiny: bool = False) -> PyramidCausal3DOperatorConfig:
    """Return the sole production config, or a test-only tiny config."""
    if tiny:
        return PyramidCausal3DOperatorConfig(
            e0_channels=8, e1_channels=24, e2_channels=32,
            e0_blocks=1, e1_blocks=1, e2_blocks=1)
    return PyramidCausal3DOperatorConfig(
        e0_channels=SCALED_WIDTHS[0],
        e1_channels=SCALED_WIDTHS[1],
        e2_channels=SCALED_WIDTHS[2],
    )


def install_sim_contract(module):
    """Bind the scaled identity into the audited Stage-S implementation."""
    module.SCHEMA = SIM_SCHEMA
    module.ACKNOWLEDGEMENT = SIM_ACKNOWLEDGEMENT
    module.ENDPOINT_KIND = SIM_ENDPOINT_KIND
    module.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    module.model_config = model_config
    return module


def install_real_contract(module, sim_module):
    """Bind Stage R to the same scaled Stage-S identity."""
    install_sim_contract(sim_module)
    if getattr(module, "sim_long", None) is not sim_module:
        raise RuntimeError("scaled Real launcher and Stage-S module identity differ")
    module.SCHEMA = REAL_SCHEMA
    module.ACKNOWLEDGEMENT = REAL_ACKNOWLEDGEMENT
    module.RESUME_SCHEMA = REAL_RESUME_SCHEMA
    module.ENDPOINT_SCHEMA = REAL_ENDPOINT_SCHEMA
    return module
