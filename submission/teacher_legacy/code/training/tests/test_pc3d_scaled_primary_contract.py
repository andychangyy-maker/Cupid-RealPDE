from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from models.pyramid_causal3d_operator import PyramidCausal3DOperator

import pc3d_scaled_primary_contract as subject


def test_scaled_architecture_identity_and_parameter_count() -> None:
    config = subject.model_config()
    assert (config.e0_channels, config.e1_channels, config.e2_channels) == (160, 256, 448)
    assert (config.e0_blocks, config.e1_blocks, config.e2_blocks) == (2, 2, 2)
    model = PyramidCausal3DOperator(config, "causal_temporal_candidate")
    assert sum(parameter.numel() for parameter in model.parameters()) == subject.EXPECTED_PARAMETERS


def test_sim_contract_install_is_complete() -> None:
    module = SimpleNamespace()
    assert subject.install_sim_contract(module) is module
    assert module.SCHEMA == subject.SIM_SCHEMA
    assert module.ACKNOWLEDGEMENT == subject.SIM_ACKNOWLEDGEMENT
    assert module.ENDPOINT_KIND == subject.SIM_ENDPOINT_KIND
    assert module.EXPECTED_PARAMETERS == subject.EXPECTED_PARAMETERS
    assert module.model_config().e2_channels == 448


def test_real_contract_is_bound_to_same_sim_module() -> None:
    sim = SimpleNamespace()
    real = SimpleNamespace(sim_long=sim)
    assert subject.install_real_contract(real, sim) is real
    assert sim.SCHEMA == subject.SIM_SCHEMA
    assert real.SCHEMA == subject.REAL_SCHEMA
    assert real.ACKNOWLEDGEMENT == subject.REAL_ACKNOWLEDGEMENT
    assert real.RESUME_SCHEMA == subject.REAL_RESUME_SCHEMA
    assert real.ENDPOINT_SCHEMA == subject.REAL_ENDPOINT_SCHEMA


def test_launchers_run_from_repo_root_without_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    scripts = (
        ("train_pyramid_causal3d_scaled_sim_long.py", subject.SIM_SCHEMA),
        ("train_pyramid_causal3d_scaled_real_dense_aug.py", subject.REAL_SCHEMA),
    )
    for filename, schema in scripts:
        completed = subprocess.run(
            [sys.executable, str(repo_root / "code" / "training" / filename),
             "--mode", "contract"],
            cwd=repo_root, env=environment, check=True,
            capture_output=True, text=True)
        payload = json.loads(completed.stdout)
        assert payload["schema"] == schema
    assert payload["scheduler_t_max"] == 12_000
