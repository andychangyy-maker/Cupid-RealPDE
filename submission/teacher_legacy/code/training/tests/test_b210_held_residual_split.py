"""Metadata-only checks against the exact bound clean81 prefix manifest."""

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import b210_held_residual_split as subject


@pytest.fixture(scope="module")
def rows():
    path = os.environ.get("REALPDE_CLEAN81_MANIFEST")
    if not path:
        pytest.skip("set REALPDE_CLEAN81_MANIFEST to the bound prefix manifest")
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == subject.MANIFEST_FILE_SHA256
    return json.loads(raw)


def test_partition_strata_and_prefix_visibility(rows):
    split = subject.build_split(rows)
    roles = split["roles"]
    point, calibration, outer, control = (
        set(roles[role]["files"]) for role in subject.ROLES)
    assert (len(point), len(calibration), len(outer), len(control)) == (49, 16, 16, 16)
    assert not point & calibration and not point & outer and not calibration & outer
    assert point | calibration | outer == {row["file"] for row in rows}
    assert control <= point
    assert split["normalizer_fit_role"] == "point_fit"
    for role in ("uq_calibration", "outer_test", "control_uq"):
        assert Counter(row["aoa"] for row in roles[role]["prefix_rows"]) == {
            0: 3, 5: 3, 10: 4, 15: 3, 20: 3}
    assert sum(roles[role]["prefix_window_count"] for role in
               ("point_fit", "uq_calibration", "outer_test")) == 2544
    strata_files = [name for band in split["re_strata"] for name in band["files"]]
    assert len(strata_files) == len(set(strata_files)) == 81
    for band in split["re_strata"]:
        assert len(calibration.intersection(band["files"])) == 1
        assert len(outer.intersection(band["files"])) == 1
    for role in subject.ROLES:
        selected = subject.filter_rows(rows, split, role)
        assert selected == roles[role]["prefix_rows"]
        assert all(set(row) == subject.PREFIX_FIELDS for row in selected)
        assert all(start + 40 <= row["val_boundary"]
                   for row in selected for start in row["train_starts"])


def test_control_matching_is_optimal_and_one_to_one(rows):
    split = subject.build_split(rows)
    pairs = split["control_matches"]
    assert len({pair["control_uq"] for pair in pairs}) == 16
    assert len({pair["uq_calibration"] for pair in pairs}) == 16
    by_name = {row["file"]: row for row in rows}
    for aoa in (0, 5, 10, 15, 20):
        selected = [pair for pair in pairs if pair["aoa"] == aoa]
        point = [row for row in split["roles"]["point_fit"]["prefix_rows"]
                 if row["aoa"] == aoa]
        calibration = [by_name[pair["uq_calibration"]] for pair in selected]
        optimum = min(sum(abs(c["re"] - q["re"]) for c, q in zip(candidate, calibration))
                      for candidate in permutations(point, len(calibration)))
        assert sum(pair["absolute_re_distance"] for pair in selected) == optimum
        assert all(by_name[pair["control_uq"]]["aoa"] == aoa for pair in selected)


def test_order_independent_and_input_not_mutated(rows):
    before = deepcopy(rows)
    split = subject.build_split(rows)
    assert subject.build_split(list(reversed(rows))) == split
    assert rows == before
    assert split["seed"] == "held-residual-20260906-v1"
    assert split["split_sha256"] == subject.canonical_sha256(
        {key: value for key, value in split.items() if key != "split_sha256"})
    filtered = subject.filter_rows(rows, split, "point_fit")
    filtered[0]["train_starts"].append(999999)
    assert rows == before
    assert filtered != split["roles"]["point_fit"]["prefix_rows"]


@pytest.mark.parametrize("change", ("tail", "duplicate", "excluded", "starts"))
def test_rejects_changed_or_nonprefix_manifest(rows, change):
    altered = deepcopy(rows)
    if change == "tail":
        altered[0]["val_starts"] = [altered[0]["val_boundary"]]
    elif change == "duplicate":
        altered[-1] = deepcopy(altered[0])
    elif change == "excluded":
        altered[0]["file"] = "7575_0.h5"
    else:
        altered[0]["train_starts"][-1] += 20
    with pytest.raises(ValueError):
        subject.build_split(altered)
