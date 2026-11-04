#!/usr/bin/env python3
"""Seal exact quality and A800 evidence for the XCB-PW12 overall-score candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence
import zipfile


BASELINE_SHA256 = "61945bd279be2497c367120cd0cff4f3b8f1585cd4764ac12084689267c610c4"
SOURCE_SHA256 = "b77cc5e00f3137cb9861f55e2662da31286b071b2c8dee79499f0332a86f400c"
CANDIDATE_SHA256 = "1b682db6e9a46df902641897c651a647e11dbecac5822096e2911aaba268cf29"
CANDIDATE_BYTES = 159_425_134
QUALITY_SUMMARY_SHA256 = (
    "cfa00ccc8d369e08065a5a9bed1c5fa5251766b58d6152395ef3be1ab393b9a2"
)
EXPECTED_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    "ensemble_unet2d.pth",
)
SOURCE_WRAPPER_SHA256 = (
    "8436505f084b2cebc85f907c2f0e9454112575d583b96eb09e5b7dc9a1b9e150"
)
CANDIDATE_WRAPPER_SHA256 = (
    "1b92b0e61a90599fe95ec0c24e2bb19e2a43888dbb3afad05b812b1a0493761a"
)
DATA_FINGERPRINTS = {
    "clean716_semantic_sha256": (
        "c3443b99f290dbfb08b0b1e6e8eaa2bc84e37f552b087c4d3a7ecbe57cff0f20"
    ),
    "manifest_file_sha256": (
        "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e"
    ),
    "manifest_semantic_sha256": (
        "d0f52e7ffdec0fad0cbd653ebb41a3d0d3c76dd2ee99a2b927cc9b0d14a76d09"
    ),
    "ordered_key_sha256": (
        "e8e7eacf0647db5b4bb49c3ba846ab0b6dbfd9fe07ce35f239407706536637e7"
    ),
}
A800_JSON_SHA256 = {
    "01-baseline-A.json": (
        "f9ae0272d9b97170a6e493c61c442ff2f9eb9ca0984dfa5b4168dede4940c430"
    ),
    "02-candidate-A.json": (
        "4e29f123d55c4b730df2d5eef90d1546dc7c80bc25a58815d8f6cc08e2d5d391"
    ),
    "03-candidate-B.json": (
        "2cbb256123f98047798d44a152df9422ed99e71a55981b4ac6066f6094d7493e"
    ),
    "04-baseline-B.json": (
        "081ef7476618c1ddb7301a36909d14fed44cf11676432fb5bd0b84518fa6abcf"
    ),
}
EXPECTED_CACHE = {
    "file": "436dbc2db9bb63ef013db7a56774ae49d7acbebf3738ac1038923f3901fffbe6",
    "array": "574b2fdc423ba2fb9c10b444961ab014381b20cb1b75dcd7788725c360f76b68",
    "b16_stream": "e49ac67690c67414e57e311f7833a08a5e108cdf3d23ff182c69220a30ea805b",
}
EXPECTED_LOCAL_DELTAS = {
    "rel_l2_score": 0.028575668629983397,
    "tke_rel_l2_score": 0.33085524790750753,
    "mvpe_rel_l2_score": 0.039725046548937826,
    "native_sps_score": 0.535527806831503,
}
EXPECTED_AOA_SPS_DELTAS = {
    "0": 0.37874970660060114,
    "5": 0.33083794739540195,
    "10": 0.560422248747976,
    "15": 0.6388548198761583,
    "20": 0.7180630313332514,
}
OFFICIAL_BASELINE = {
    "rel_l2_score": 94.5822134538,
    "tke_rel_l2_score": 78.8439169015,
    "mvpe_rel_l2_score": 93.9067679058,
    "time_score": 90.7315600085,
    "native_sps_score": 43.2121556625,
    "final_score": 81.3531966340,
}
FINAL_POWER = 0.6854
FINAL_WEIGHTS = {
    "rel_l2_score": 0.5,
    "tke_rel_l2_score": 0.1,
    "mvpe_rel_l2_score": 0.1,
    "time_score": 0.1,
    "native_sps_score": 0.2,
}
MIN_PROJECTED_GAIN = 0.10
MAX_PAIR_SPREAD = 0.03


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def generalized_final(metrics: Mapping[str, float]) -> float:
    total = sum(
        FINAL_WEIGHTS[name] * float(metrics[name]) ** FINAL_POWER
        for name in FINAL_WEIGHTS
    )
    return float(total ** (1.0 / FINAL_POWER))


def time_score_from_ratio(ratio: float) -> float:
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("latency ratio must be finite and positive")
    baseline_root_ratio = 100.0 / OFFICIAL_BASELINE["time_score"] - 1.0
    return float(100.0 / (1.0 + math.sqrt(ratio) * baseline_root_ratio))


def pair_spread(values: Sequence[float]) -> float:
    if len(values) != 2 or min(values) <= 0.0:
        raise ValueError("pair spread requires two positive values")
    return float((max(values) - min(values)) / min(values))


def archive_audit(path: Path, expected_sha: str) -> dict[str, object]:
    path = path.resolve()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if names != EXPECTED_MEMBERS or len(names) != len(set(names)):
            raise ValueError(f"archive members differ: {path}")
        if archive.testzip() is not None:
            raise ValueError(f"archive CRC failed: {path}")
        members = {
            info.filename: {
                "bytes": info.file_size,
                "sha256": sha256_bytes(archive.read(info)),
            }
            for info in infos
        }
    identity = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if identity["sha256"] != expected_sha:
        raise ValueError(f"archive SHA-256 differs: {path}")
    return {**identity, "members": members}


def replay_signature(row: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "metrics",
        "metric_state",
        "by_aoa",
        "point_prediction_sha256",
        "lower_sha256",
        "upper_sha256",
        "samples",
        "device",
        "runtime_source_sha256",
    )
    return {name: row[name] for name in keys}


def load_quality(path: Path) -> dict[str, object]:
    if sha256(path) != QUALITY_SUMMARY_SHA256:
        raise ValueError("quality summary SHA-256 differs")
    report = json.loads(path.read_text(encoding="utf-8"))
    data = report["data"]
    arms = report["arms"]
    for name, expected in DATA_FINGERPRINTS.items():
        if data.get(name) != expected:
            raise ValueError(f"quality data fingerprint differs: {name}")
    if not (
        data.get("tail_windows") == 716
        and data.get("trajectories") == 81
        and data.get("excluded") == "7575_0.h5"
        and data["tail_access"].get("excluded_opened") is False
        and data["tail_access"].get("prefix_frames_read") == 0
        and data["tail_access"].get("allowlist_exact") is True
        and report.get("loader_batch_size") == 16
    ):
        raise ValueError("quality clean716 contract differs")
    for label in ("AA", "DD"):
        if arms[label]["archive"]["sha256"] != BASELINE_SHA256:
            raise ValueError("quality baseline archive differs")
    for label in ("DA", "AD"):
        if arms[label]["archive"]["sha256"] != CANDIDATE_SHA256:
            raise ValueError("quality candidate archive differs")
    baseline_repeat = canonical(replay_signature(arms["AA"])) == canonical(
        replay_signature(arms["DD"])
    )
    candidate_repeat = canonical(replay_signature(arms["DA"])) == canonical(
        replay_signature(arms["AD"])
    )
    deltas = {
        name: float(arms["DA"]["metrics"][name]) - float(arms["DD"]["metrics"][name])
        for name in EXPECTED_LOCAL_DELTAS
    }
    aoa = {
        name: float(arms["DA"]["by_aoa"][name]["native_sps_score"])
        - float(arms["DD"]["by_aoa"][name]["native_sps_score"])
        for name in EXPECTED_AOA_SPS_DELTAS
    }
    exact_deltas = all(
        math.isclose(deltas[name], expected, rel_tol=0.0, abs_tol=1e-12)
        for name, expected in EXPECTED_LOCAL_DELTAS.items()
    )
    exact_aoa = all(
        math.isclose(aoa[name], expected, rel_tol=0.0, abs_tol=1e-12)
        for name, expected in EXPECTED_AOA_SPS_DELTAS.items()
    )
    conditions = {
        "baseline_replay_exact": baseline_repeat,
        "candidate_replay_exact": candidate_repeat,
        "four_quality_deltas_exact": exact_deltas,
        "four_quality_metrics_positive": all(value > 0.0 for value in deltas.values()),
        "five_of_five_aoa_sps_positive": all(value > 0.0 for value in aoa.values()),
        "aoa_deltas_exact": exact_aoa,
    }
    return {
        "summary": {"path": str(path.resolve()), "sha256": QUALITY_SUMMARY_SHA256},
        "conditions": conditions,
        "pass": all(conditions.values()),
        "baseline_metrics": arms["DD"]["metrics"],
        "candidate_metrics": arms["DA"]["metrics"],
        "deltas": deltas,
        "aoa_sps_deltas": aoa,
        "candidate_output_sha256": {
            name: arms["DA"][name]
            for name in ("point_prediction_sha256", "lower_sha256", "upper_sha256")
        },
    }


def load_a800(directory: Path) -> dict[str, object]:
    rows = []
    for name, expected_sha in A800_JSON_SHA256.items():
        path = directory / name
        if sha256(path) != expected_sha:
            raise ValueError(f"A800 result SHA-256 differs: {name}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
        for suffix in ("before.csv", "after.csv"):
            idle = directory / name.replace(".json", f"-{suffix}")
            if not idle.is_file() or idle.stat().st_size != 0:
                raise ValueError(f"A800 idle evidence differs: {idle.name}")
    expected_arms = ("baseline", "candidate", "candidate", "baseline")
    if tuple(row["arm"] for row in rows) != expected_arms:
        raise ValueError("A800 arm order differs")
    fingerprints = {canonical(row["environment_fingerprint"]) for row in rows}
    expected_batches = [16] * 44 + [12]
    validity = {
        "fresh_abba_order": True,
        "runtime_fingerprint_exact": len(fingerprints) == 1,
        "a800_fp16": all(
            row["environment_fingerprint"]["gpu_name"] == "NVIDIA A800 80GB PCIe"
            and row["dtype"] == "torch.float16"
            for row in rows
        ),
        "clean716_call_sequence_exact": all(
            row["samples"] == 716
            and row["predict_calls"] == 45
            and row["call_batch_sizes"] == expected_batches
            for row in rows
        ),
        "cache_identity_exact": all(
            row["input_cache"]["sha256"] == EXPECTED_CACHE["file"]
            and row["input_array_sha256"] == EXPECTED_CACHE["array"]
            and row["input_b16_stream_sha256"] == EXPECTED_CACHE["b16_stream"]
            for row in rows
        ),
        "worker_validity_pass": all(
            row["decision"] == "PASS_TIME_Q5_A800_WORKER"
            and row["input_bitwise_unchanged"] is True
            and row["finite_ordered_pressure_zero"] is True
            for row in rows
        ),
        "archive_identity_exact": all(
            row["archive"]["sha256"]
            == (BASELINE_SHA256 if row["arm"] == "baseline" else CANDIDATE_SHA256)
            for row in rows
        ),
    }
    baseline = (rows[0], rows[3])
    candidate = (rows[1], rows[2])
    baseline_hashes = [canonical(row["ordered_output_sha256"]) for row in baseline]
    candidate_hashes = [canonical(row["ordered_output_sha256"]) for row in candidate]
    candidate_remainders = [
        canonical(row["remainder17_output_sha256"]) for row in candidate
    ]
    validity.update(
        {
            "baseline_output_repeat_exact": len(set(baseline_hashes)) == 1,
            "candidate_output_repeat_exact": len(set(candidate_hashes)) == 1,
            "candidate_remainder17_repeat_exact": len(set(candidate_remainders)) == 1,
        }
    )
    b45 = [float(row["predict_seconds_total_45_calls"]) for row in baseline]
    c45 = [float(row["predict_seconds_total_45_calls"]) for row in candidate]
    b16 = [float(row["b16_seconds_total_44_calls"]) for row in baseline]
    c16 = [float(row["b16_seconds_total_44_calls"]) for row in candidate]
    b12 = [float(row["b12_seconds"]) for row in baseline]
    c12 = [float(row["b12_seconds"]) for row in candidate]
    ratios = {
        "R45": sum(c45) / sum(b45),
        "R16": sum(c16) / sum(b16),
        "R12": sum(c12) / sum(b12),
    }
    spreads = {"baseline_R45": pair_spread(b45), "candidate_R45": pair_spread(c45)}
    validity["primary_pair_spreads_le_3pct"] = all(
        value <= MAX_PAIR_SPREAD for value in spreads.values()
    )
    return {
        "directory": str(directory.resolve()),
        "conditions": validity,
        "measurement_valid": all(validity.values()),
        "ratios": ratios,
        "pair_spreads": spreads,
        "candidate_ordered_output_sha256": candidate[0]["ordered_output_sha256"],
        "candidate_remainder17_output_sha256": candidate[0][
            "remainder17_output_sha256"
        ],
        "time_only_b16_pass": ratios["R16"] <= 1.0,
    }


def projected_scenario(deltas: Mapping[str, float], ratio: float) -> dict[str, float]:
    metrics = {
        name: float(OFFICIAL_BASELINE[name]) + float(deltas[name])
        for name in (
            "rel_l2_score",
            "tke_rel_l2_score",
            "mvpe_rel_l2_score",
            "native_sps_score",
        )
    }
    metrics["time_score"] = time_score_from_ratio(ratio)
    result = generalized_final(metrics)
    return {
        **metrics,
        "inferred_final": result,
        "inferred_gain": result - OFFICIAL_BASELINE["final_score"],
    }


def seal(
    baseline: Path,
    source: Path,
    candidate: Path,
    quality_summary: Path,
    a800_directory: Path,
) -> dict[str, object]:
    archives = {
        "baseline": archive_audit(baseline, BASELINE_SHA256),
        "source_xcb": archive_audit(source, SOURCE_SHA256),
        "candidate": archive_audit(candidate, CANDIDATE_SHA256),
    }
    wrapper_only = all(
        archives["source_xcb"]["members"][name]
        == archives["candidate"]["members"][name]
        for name in EXPECTED_MEMBERS
        if name != "submission.py"
    )
    archive_conditions = {
        "candidate_bytes_exact": archives["candidate"]["bytes"] == CANDIDATE_BYTES,
        "source_wrapper_exact": archives["source_xcb"]["members"]["submission.py"][
            "sha256"
        ]
        == SOURCE_WRAPPER_SHA256,
        "candidate_wrapper_exact": archives["candidate"]["members"]["submission.py"][
            "sha256"
        ]
        == CANDIDATE_WRAPPER_SHA256,
        "only_wrapper_changed_from_xcb": wrapper_only,
    }
    quality = load_quality(quality_summary)
    a800 = load_a800(a800_directory)
    scenarios = {
        "clean716_R45": projected_scenario(quality["deltas"], a800["ratios"]["R45"]),
        "conservative_B16_only": projected_scenario(
            quality["deltas"], a800["ratios"]["R16"]
        ),
    }
    integrity_pass = all(archive_conditions.values()) and quality["pass"] and a800[
        "measurement_valid"
    ]
    overall_score_pass = integrity_pass and all(
        row["inferred_gain"] >= MIN_PROJECTED_GAIN for row in scenarios.values()
    )
    return {
        "schema": "track1_xcb_pw12_candidate_seal_v1",
        "evidence_class": (
            "exact local clean716 plus same-family A800 deployment evidence; "
            "official score remains unobserved"
        ),
        "archives": archives,
        "archive_conditions": archive_conditions,
        "quality": quality,
        "a800": a800,
        "inferred_official_scenarios": scenarios,
        "decision_time_only": (
            "PASS_XCB_PW12_TIME_ONLY_B16"
            if a800["time_only_b16_pass"]
            else "STOP_XCB_PW12_TIME_ONLY_B16_REGRESSION"
        ),
        "decision_goal": (
            "PASS_XCB_PW12_OVERALL_SCORE_CANDIDATE"
            if overall_score_pass
            else "STOP_XCB_PW12_OVERALL_SCORE_CANDIDATE"
        ),
        "goal_gate": {
            "minimum_inferred_gain": MIN_PROJECTED_GAIN,
            "integrity_pass": integrity_pass,
            "clean716_R45_gain_pass": scenarios["clean716_R45"]["inferred_gain"]
            >= MIN_PROJECTED_GAIN,
            "conservative_B16_only_gain_pass": scenarios["conservative_B16_only"][
                "inferred_gain"
            ]
            >= MIN_PROJECTED_GAIN,
            "pass": overall_score_pass,
        },
        "recommended_for_user_submission": overall_score_pass,
        "automatic_submission_authorized": False,
        "risk": (
            "B12 is a clean716-specific tail shape, not an official contract; "
            "the conservative B16-only scenario is therefore the release floor."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-xcb", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--a800-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = seal(
        args.baseline,
        args.source_xcb,
        args.candidate,
        args.quality_summary,
        args.a800_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "decision_goal": result["decision_goal"],
        "decision_time_only": result["decision_time_only"],
        "candidate": result["archives"]["candidate"],
        "scenarios": result["inferred_official_scenarios"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
