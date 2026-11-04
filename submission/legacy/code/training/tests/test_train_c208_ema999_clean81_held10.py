from __future__ import annotations

import ast
from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_ema999_clean81_held10 as subject  # noqa: E402


def comparison(**overrides):
    delta = {
        "point_score": 0.1,
        "rel_l2_score": 0.0,
        "tke_rel_l2_score": 0.0,
        "mvpe_rel_l2_score": 0.0,
        "native_sps_score": 0.0,
    }
    exact = overrides.pop("exact", 0.1)
    delta.update(overrides)
    return {"delta": delta, "exact_non_time_final_delta": exact}


def test_contract_binds_single_legacy_start_and_clean_prefix_only():
    assert subject.FORMAL_START_SHA256 == (
        "6674fcc6a1d777e6407b99bff1a14ee0e106c706a4b1cb67a9108e850ae809e3")
    assert subject.HELD_AOA == 10
    assert subject.CONTRACT["data"] == {
        "trajectories": 81,
        "prefix_windows": 2544,
        "manifest_sha256": (
            "d0f52e7ffdec0fad0cbd653ebb41a3d0d3c76dd2ee99a2b927cc9b0d14a76d09"),
        "excluded": "7575_0.h5",
        "tail_arrays_read": 0,
    }


def test_contract_binds_ema_and_native_training_schedule():
    assert subject.EMA_DECAY == 0.999
    assert subject.CONTRACT["ema"]["updates"] == 4000
    assert subject.CONTRACT["training"] == {
        "updates": 4000,
        "batch_size": 8,
        "seed": 4,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "scheduler_t_max": 5000,
        "grad_clip": 1.0,
    }


def test_quick_gate_passes_only_strictly_positive_release_signal():
    gate = subject.quick_gate(comparison())
    assert gate["pass"] is True
    assert gate["decision"] == "GO_EMA999_FULL_CLEAN81_LOAO"
    for field in ("point_score", "rel_l2_score", "tke_rel_l2_score",
                  "mvpe_rel_l2_score", "native_sps_score"):
        row = comparison(**{field: -1.0})
        assert subject.quick_gate(row)["pass"] is False
    assert subject.quick_gate(comparison(point_score=0.0))["pass"] is False
    assert subject.quick_gate(comparison(exact=0.0))["pass"] is False


def test_quick_gate_floor_edges_are_inclusive():
    gate = subject.quick_gate(comparison(
        rel_l2_score=-0.02,
        tke_rel_l2_score=-0.02,
        mvpe_rel_l2_score=-0.02,
        native_sps_score=-0.10,
    ))
    assert gate["pass"] is True


def test_update_order_is_optimizer_then_ema_then_scheduler():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "train_raw_with_ema")
    text = ast.get_source_segment(source, function)
    optimizer = text.index("optimizer.step()")
    ema = text.index("ema.update_after_optimizer_step")
    scheduler = text.index("scheduler.step()")
    assert optimizer < ema < scheduler
    assert "optimizer_step=iteration" in text


def test_source_has_no_submission_or_legacy_validation_split():
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    assert "codabench" not in source
    assert "clean716" in source  # Explicit prohibition in the module contract.
    assert "fixed725" not in source
    assert "val_starts" not in source
    assert subject.CONTRACT["release_checkpoint_written"] is False
    assert subject.CONTRACT["submission_authorized"] is False
