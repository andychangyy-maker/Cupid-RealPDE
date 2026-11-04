from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_history_episodic_adaptation_crossfit_screen as heat  # noqa: E402


def _records() -> list[dict[str, object]]:
    return [
        {"file": f"a{aoa}_r{re:02d}.h5", "aoa": aoa, "re": re,
         "frames": 1_000, "shape": [1_000, 64, 128]}
        for aoa in heat.ALLOWED_AOAS for re in range(20)
    ]


def test_re_ordinal_fold_is_60_20_and_cross_aoa_re_disjoint():
    train, held, manifest = heat.select_fold(_records(), 0)
    assert len(train) == 60 and len(held) == 20
    assert {row["re"] for row in train}.isdisjoint({row["re"] for row in held})
    assert all(len(manifest["held_re_by_aoa"][str(aoa)]) == 5
               for aoa in heat.ALLOWED_AOAS)


def test_donor_map_is_same_aoa_start_and_other_trajectory():
    rows = []
    for aoa in heat.ALLOWED_AOAS:
        for re in (100, 200, 300):
            for start in (0, 20):
                rows.append({"file": f"a{aoa}_r{re}.h5", "aoa": aoa,
                             "re": re, "start": start, "trajectory": len(rows)})
    mapping = heat.donor_indexes(rows)
    assert mapping.dtype == np.int64
    for index, donor in enumerate(mapping.tolist()):
        source, other = rows[index], rows[donor]
        assert source["file"] != other["file"]
        assert source["aoa"] == other["aoa"]
        assert source["start"] == other["start"]
    audit = heat.donor_map_audit(rows, mapping)
    assert audit == {
        "windows": len(rows),
        "unique_source_files": 12,
        "valid_index": True,
        "different_trajectory": True,
        "same_aoa": True,
        "same_start": True,
        "passed": True,
    }


def test_schedule_is_fixed_shape_and_fit_indexed():
    schedule = heat.make_schedule([object()] * 19)
    assert schedule.shape == (heat.UPDATES, heat.BATCH_SIZE)
    assert schedule.min() >= 0 and schedule.max() < 19
    assert np.array_equal(schedule, heat.make_schedule([object()] * 19))


def _evaluation(*, support_self: float = .8, primary_point: float = .06,
                donor_point: float = .04, override_point: float = .03):
    components = {"rel_l2_score": .0, "tke_score": .0, "mvpe_score": .0,
                  "point_contribution_46_10_9": primary_point}
    return {
        "support_mechanism": {"self": support_self, "static": 1., "donor": .9},
        "B_self_minus_A_static": {"delta": components,
            "positive_trajectory_files": 15, "worst_trajectory_delta_point": -.06,
            "positive_aoa_groups": 3},
        "B_self_minus_C_donor": {"delta": {**components,
            "point_contribution_46_10_9": donor_point}},
        "horizons_11_20": {"B_self": {"point_contribution_46_10_9": 1.},
                             "A_static": {"point_contribution_46_10_9": 1.}},
        "B_same_checkpoint_overrides": {"self_minus_donor": {
            "point_contribution_46_10_9": override_point}},
        "mechanical_fast_state_audit": {
            "all_gradient_finite": True,
            "all_state_finite": True,
            "all_output_finite": True,
            "all_step_finite": True,
            "nondegenerate_step_radius_exact": True,
            "donor_structure": {"passed": True},
            "replay": {
                key: {
                    "batch_permutation_restored_bitwise": True,
                    "independent_call_bitwise": True,
                }
                for key in (*heat.ARM_NAMES, "B_override_self",
                            "B_override_static", "B_override_donor")
            },
        },
    }


def test_gate_requires_support_mechanism_and_all_preregistered_point_checks():
    assert heat.gate(_evaluation())[0] is True
    assert heat.gate(_evaluation(support_self=1.1))[0] is False
    assert heat.gate(_evaluation(primary_point=.049))[0] is False
    assert heat.gate(_evaluation(donor_point=.029))[0] is False
    assert heat.gate(_evaluation(override_point=.019))[0] is False

    nonfinite = _evaluation()
    nonfinite["mechanical_fast_state_audit"]["all_output_finite"] = False
    assert heat.gate(nonfinite)[0] is False

    nonreplayable = _evaluation()
    nonreplayable["mechanical_fast_state_audit"]["replay"][
        "C_donor"]["independent_call_bitwise"] = False
    assert heat.gate(nonreplayable)[0] is False

    invalid_donor = _evaluation()
    invalid_donor["mechanical_fast_state_audit"]["donor_structure"][
        "passed"] = False
    assert heat.gate(invalid_donor)[0] is False


def test_donor_map_rejects_singleton_same_aoa_start_group():
    with pytest.raises(ValueError, match="fewer than two"):
        heat.donor_indexes([{"file": "one.h5", "aoa": 0, "re": 1,
                             "start": 0, "trajectory": 0}])


def test_static_arm_executes_support_api_but_returns_fresh_zero_state():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def adapted_fast_state(self, history, *, radius, create_graph, mode):
            self.calls.append((history.clone(), radius, create_graph, mode))
            assert mode == "static"
            return torch.zeros(len(history), 3), {
                "per_sample_loss": torch.full((len(history),), .25)}

    class UnitNormalizer:
        mean_input = torch.zeros(3)
        std_input = torch.ones(3)

    model = FakeModel()
    history = torch.randn(2, 20, 1, 1, 3)
    state, diagnostics = heat.arm_fast_state(
        "A_static", model, history, history.flip(0), UnitNormalizer())
    assert len(model.calls) == 1
    assert model.calls[0][3] == "static"
    assert model.calls[0][1] == heat.FAST_RADIUS
    assert torch.equal(state, torch.zeros_like(state))
    assert torch.equal(diagnostics["per_sample_loss"], torch.full((2,), .25))


def test_support_wrapper_normalizes_history_then_uses_model_owned_split():
    class FakeModel:
        def __init__(self):
            self.seen = None

        def support_loss(self, history, fast):
            self.seen = history
            return fast.sum(dim=1)

    class Normalizer:
        mean_input = torch.full((3,), 2.)
        std_input = torch.full((3,), 4.)

    model = FakeModel()
    history = torch.full((2, 20, 1, 1, 3), 10.)
    output = heat.support_per_sample_loss(model, history, Normalizer(),
                                          torch.ones(2, 5))
    assert torch.equal(model.seen, torch.full_like(history, 2.))
    assert output.tolist() == [5., 5.]
