"""Fixed trajectory roles from clean81 prefix metadata; no tensor/data access."""

from copy import deepcopy
import hashlib
from itertools import combinations
import json


SEED = "held-residual-20260906-v1"
MANIFEST_FILE_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
MANIFEST_SEMANTIC_SHA256 = (
    "d0f52e7ffdec0fad0cbd653ebb41a3d0d3c76dd2ee99a2b927cc9b0d14a76d09")
PREFIX_FIELDS = {
    "file", "frames", "re", "aoa", "train_starts", "val_boundary"}
ROLES = ("point_fit", "uq_calibration", "outer_test", "control_uq")


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _validated_rows(rows) -> list[dict]:
    if any(set(row) != PREFIX_FIELDS for row in rows):
        raise ValueError("expected prefix-only manifest fields; no tail/labels")
    ordered = sorted(rows, key=lambda row: row["file"])
    names = [row["file"] for row in ordered]
    if (len(names) != 81 or len(set(names)) != 81
            or "7575_0.h5" in names):
        raise ValueError("expected unique clean81 trajectories")
    if canonical_sha256(ordered) != MANIFEST_SEMANTIC_SHA256:
        raise ValueError("clean81 prefix manifest semantic SHA changed")
    return ordered


def build_split(rows) -> dict:
    """Build the fixed 49/16/16 split; caller verifies original file bytes.

    Each AoA has three equal-rank Re strata (four at AoA 10). A seeded
    SHA ordering selects one calibration and one outer trajectory per stratum.
    Control matching minimizes total absolute Re distance within each AoA;
    sorted one-dimensional matching is optimal, with SHA-breaking subset ties.
    """
    ordered = _validated_rows(rows)
    selected = {role: [] for role in ROLES}
    strata = []
    matches = []
    for aoa in (0, 5, 10, 15, 20):
        group = sorted((row for row in ordered if row["aoa"] == aoa),
                       key=lambda row: (row["re"], row["file"]))
        count = 4 if aoa == 10 else 3
        calibration, outer = [], []
        for index in range(count):
            band = group[index * len(group) // count:
                         (index + 1) * len(group) // count]
            ranked = sorted(band, key=lambda row: canonical_sha256(
                [SEED, "role", aoa, index, row["file"]]))
            calibration.append(ranked[0])
            outer.append(ranked[1])
            strata.append({
                "aoa": aoa, "re_rank_stratum": index,
                "files": [row["file"] for row in band],
                "uq_calibration": ranked[0]["file"],
                "outer_test": ranked[1]["file"],
            })
        held = {row["file"] for row in calibration + outer}
        point = [row for row in group if row["file"] not in held]
        calibration.sort(key=lambda row: (row["re"], row["file"]))
        control = min(combinations(point, count), key=lambda subset: (
            sum(abs(c["re"] - q["re"])
                for c, q in zip(subset, calibration)),
            canonical_sha256([SEED, "match", aoa,
                              [row["file"] for row in subset]])))
        for q, c in zip(calibration, control):
            matches.append({
                "uq_calibration": q["file"], "control_uq": c["file"],
                "aoa": aoa, "calibration_re": q["re"],
                "control_re": c["re"], "absolute_re_distance": abs(
                    q["re"] - c["re"]),
            })
        for role, subset in zip(ROLES, (point, calibration, outer, control)):
            selected[role].extend(row["file"] for row in subset)

    roles = {}
    for role, names in selected.items():
        prefix = deepcopy([row for row in ordered if row["file"] in names])
        roles[role] = {
            "files": [row["file"] for row in prefix],
            "trajectory_count": len(prefix), "prefix_rows": prefix,
            "prefix_window_count": sum(len(row["train_starts"])
                                       for row in prefix),
            "prefix_rows_sha256": canonical_sha256(prefix),
        }
    result = {
        "schema": "b210_held_residual_split_v1", "seed": SEED,
        "source_manifest_required_file_sha256": MANIFEST_FILE_SHA256,
        "source_manifest_semantic_sha256": MANIFEST_SEMANTIC_SHA256,
        "normalizer_fit_role": "point_fit", "roles": roles,
        "re_strata": strata, "control_matches": matches,
        "selection_inputs": ["file", "aoa", "re"],
        "partition_roles": ["point_fit", "uq_calibration", "outer_test"],
    }
    result["split_sha256"] = canonical_sha256(result)
    return result


def filter_rows(rows, split: dict, role: str) -> list[dict]:
    """Return independent prefix rows, rejecting altered membership or starts."""
    ordered = _validated_rows(rows)
    selected = split["roles"][role]
    names = set(selected["files"])
    filtered = [row for row in ordered if row["file"] in names]
    if (len(filtered) != selected["trajectory_count"]
            or canonical_sha256(filtered) != selected["prefix_rows_sha256"]):
        raise ValueError("split role membership or prefix metadata changed")
    return deepcopy(filtered)
