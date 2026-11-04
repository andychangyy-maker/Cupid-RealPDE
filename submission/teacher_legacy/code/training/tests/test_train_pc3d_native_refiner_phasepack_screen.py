from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_pc3d_native_refiner_phasepack_screen as subject


def _metric(value: float) -> dict[str, float]:
    return {"rel_l2_score": value, "tke_rel_l2_score": value,
            "mvpe_rel_l2_score": value, "point_contribution_46_10_9": value}


def _evaluation(*, base: float = subject.GATE["base_reference"],
                candidate: float | None = None, delta: float = .04) -> dict[str, object]:
    candidate = base + .05 if candidate is None else candidate
    return {"base": {"aggregate": _metric(base)}, "control_vs_candidate": {
        "aggregate": {subject.ARMS[0]: _metric(candidate - delta),
                      subject.ARMS[1]: _metric(candidate)},
        "delta_candidate_minus_control": {"rel_l2_score": .05,
                                           "tke_rel_l2_score": 0.,
                                           "mvpe_rel_l2_score": .05,
                                           "point_contribution_46_10_9": delta},
        "positive_trajectories": 15, "worst_trajectory_delta_point": -.05,
        "trajectory_count": 20, "windows_per_trajectory": 48}}


def test_frozen_contract_and_pair_execution_on_cpu_without_data_or_gpu():
    assert subject.HELD_AOA == 10
    assert subject.FIT_AOAS == (0, 5, 15, 20)
    assert subject.ARMS == ("phase_pack_2_control", "full_resolution_candidate")
    assert subject.ACKNOWLEDGEMENT == "RUN_PC3D_NATIVE_REFINER_PHASEPACK_HELD10"
    assert subject.EVIDENCE_CLASS == "selection_reused_whole_AoA10_Sim"
    assert (subject.EXPECTED_BASE_PARAMETERS, subject.EXPECTED_BASE_STATE_TENSORS) == (8_929_552, 184)
    base_model, base_identity = subject.construct_base(torch.device("cpu"))
    assert base_identity["initial_state_sha256"] == subject.EXPECTED_BASE_INITIAL_SHA256
    assert base_identity["initial_state_matches_expected"] is True
    del base_model
    assert (subject.EXPECTED_REFINER_PARAMETERS, subject.EXPECTED_REFINER_STATE_TENSORS) == (8_771, 22)
    models, identity = subject.construct_refiner_pair(torch.device("cpu"))
    assert identity["bitwise_identical"]
    assert identity["differing_architecture_keys"] == ["execution_mode"]
    contract = subject.refiner_execution_contract(models, torch.device("cpu"), formal=False)
    assert contract["pass"]
    assert contract["checks"]["random_pack_unpack_exact"]
    assert contract["checks"]["four_phase_impulse_pack_unpack_exact"]
    assert contract["checks"]["four_phase_impulse_order_exact"]
    assert contract["checks"]["update0_identity_both_modes"]
    formal_contract = subject.refiner_execution_contract(models, torch.device("cpu"), formal=True)
    assert formal_contract["pass"]
    assert formal_contract["raw_grid_before_existing_loader"] == [64, 128]
    assert formal_contract["effective_grid_from_existing_loader_::2"] == [32, 64]


def test_gate_separates_base_fidelity_from_science():
    identity = {"bitwise_identical": True}
    execution = mechanical = {"pass": True}
    good = subject.decide_gate(_evaluation(), identity, execution, mechanical)
    assert good["science_pass"] and good["base_fidelity_pass"]
    assert subject.decision_for(good) == "GO_NATIVE_REFINER_HELD20"
    inconclusive = subject.decide_gate(_evaluation(base=60.0), identity, execution, mechanical)
    assert inconclusive["science_pass"] and not inconclusive["base_fidelity_pass"]
    assert subject.decision_for(inconclusive) == "INCONCLUSIVE_BASE_FIDELITY"
    stopped = subject.decide_gate(_evaluation(delta=.039), identity, execution, mechanical)
    assert not stopped["science_pass"]
    assert subject.decision_for(stopped) == "STOP_NATIVE_REFINER_PHASEPACK"


def test_held_loader_is_guarded_by_base_and_two_refiner_seals(tmp_path: Path):
    endpoints = {"base": {"endpoint_state_sha256": "base"},
                 subject.ARMS[0]: {"endpoint_state_sha256": "control"},
                 subject.ARMS[1]: {"endpoint_state_sha256": "candidate"}}
    calls = []
    def loader(_path, rows): calls.append(rows); return "loaded"
    assert subject.load_held_fields_after_guards(tmp_path, ["held"], endpoints,
                                                 {"pass": True, "checks": {}}, loader=loader) == "loaded"
    with pytest.raises(RuntimeError, match="missing endpoint"):
        subject.load_held_fields_after_guards(tmp_path, ["held"], {"base": endpoints["base"]},
                                              {"pass": True, "checks": {}}, loader=loader)
    with pytest.raises(RuntimeError, match="before held load"):
        subject.load_held_fields_after_guards(tmp_path, ["held"], endpoints,
                                              {"pass": False, "checks": {"x": False}}, loader=loader)
    assert calls == [["held"]]


def test_formal_rejects_before_any_data_access_without_cuda(tmp_path: Path):
    args = SimpleNamespace(execute=True, acknowledge=subject.ACKNOWLEDGEMENT,
                           held_aoa=10, updates=2000, batch_size=8,
                           sim_dir=tmp_path, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        subject.run(args, torch.device("cpu"))
