#!/usr/bin/env python3
"""Plan, or replay all sealed matched-UQ arms and actual v3 on clean716.

Research quality replay only: common v3 runtime, INT8 storage / FP16 execution,
no latency gate, release ZIP, upload, or tail-based checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
V3_SHA = "61945bd279be2497c367120cd0cff4f3b8f1585cd4764ac12084689267c610c4"
MANIFEST_SHA = "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e"
TIME_SCORE = 90.7315600085
METRICS = ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score", "sps_score")
RUNTIME_FILES = ("submission.py", "models/__init__.py", "models/unet2d.py", "models/ensemble_unet2d.py")
SIM_INHERITED_POINTS = {
    "A_inherit": {
        "arm": "A",
        "point_receipt_sha256": "0f253382ff387e87fac497e67f35a6f5b0b5274c34debc83a0dfb064dc03b364",
        "checkpoint_sha256": "3f453fe38d471aecbc4748425bcfd0ce6a3505b457100aae7a05f10ae711d211",
        "start_sim_sha256": "4b56454ce64e301a9898dc1c066d23180de587cbebc7c53786fd3ca126981e68",
        "point_frozen_sha256": "ee33c57a8826319e27916f978f9d204f0be5a280b8cd57dd472ea36b8977edb0",
    },
    "inherit": {
        "arm": "C",
        "point_receipt_sha256": "3c9d425e1433ecce4e5d3ce5f94e968f17c10b05c32eae50187aa55481e9a891",
        "checkpoint_sha256": "96c0082c921c2aaeb449ef14ba5f94a7361395164b2064a191c407c337509903",
        "start_sim_sha256": "628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b",
        "point_frozen_sha256": "b818e67a39292dc53aaa2af266a25fe7b3f0fbc1539ade1c7bfc52649d34b3f3",
    },
}
REUSED_C_UQ_RECEIPT_SHA = "0c1295d2e8c97cfef3b2a805fcf436b77668d02b85bd16f181fe63971755923e"
REUSED_C_UQ_SHA = "04d8cd88805b913a316cf19141f74019d02c105a98945c002027e0aa4bf5004a"
SOURCE_C_INITIAL_UQ_SHA = "2c9f6b3781fc6d9950e08590d77c2d51300ec1e61bfd1d250a994871d598af29"
WEIGHTED_REMOVED_PREFIX = "utility_refinement_head."
UQ_EMA_PAIR_SCHEMA = "b210_uq_ema999_pair_v1"
UQ_EMA_STREAM_SHA = "d17a0520286fde93c61c8c29d40342d8c5e629f05c8950ee5f1af075e8d223f3"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def fixed_time_final(metrics):
    scores = [metrics[key] for key in METRICS[:3]] + [TIME_SCORE, metrics["sps_score"]]
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("Final requires finite nonnegative subscores")
    return sum(weight * value ** .6854 for weight, value in
               zip((.5, .1, .1, .1, .2), scores)) ** (1 / .6854)


def metric_state(official, prediction, truth, lower, upper):
    """Use absolute endpoints directly; preserve the scorer's point-dependent SPS."""
    import numpy as np
    channels = official.measured_channels(truth)
    if channels != 2:
        raise ValueError("clean Real replay must measure exactly u/v")
    for value in (prediction, lower, upper):
        official.validate_shapes(value, truth)
        if value.dtype != np.float32 or not np.isfinite(value).all() or np.any(value[..., 2] != 0):
            raise ValueError("public output must be finite FP32 with zero pressure")
    if np.any(lower > upper):
        raise ValueError("unordered absolute endpoints")
    errors = [official.rel_l2_per_sample(prediction, truth, channels),
              official.tke_rel_l2_per_sample(prediction, truth, channels),
              official.mvpe_rel_l2_per_sample(prediction, truth)]
    sps, coverage = official.aggregate_sps(prediction, truth, channels, lower, upper)
    mask = truth[..., :channels] != 0
    count = int(mask.sum())
    return {"samples": len(truth), "rel_error_sum": float(np.sum(errors[0], dtype=np.float64)),
            "tke_error_sum": float(np.sum(errors[1], dtype=np.float64)),
            "mvpe_error_sum": float(np.sum(errors[2], dtype=np.float64)),
            "sps_numerator": sps * count, "sps_scored": count,
            "sps_covered": int(round(coverage * count)),
            "width_sum": float(np.sum((upper[..., :channels] - lower[..., :channels])[mask], dtype=np.float64))}


def merge_states(states):
    if not states:
        raise ValueError("cannot merge empty metric states")
    return {key: sum(state[key] for state in states) for key in states[0]}


def finalize(official, state):
    if state["samples"] <= 0 or state["sps_scored"] <= 0:
        raise ValueError("empty score state")
    result = {key: official.score_error(state[source] / state["samples"])
              for key, source in zip(METRICS[:3], ("rel_error_sum", "tke_error_sum", "mvpe_error_sum"))}
    result.update(sps_score=official.score_sps(state["sps_numerator"] / state["sps_scored"]),
                  coverage=state["sps_covered"] / state["sps_scored"],
                  mean_width=state["width_sum"] / state["sps_scored"], time_score=TIME_SCORE)
    result["fixed_time_fitted_final"] = fixed_time_final(result)
    return result


def contrasts(results):
    keys = (*METRICS, "fixed_time_fitted_final")
    def delta(left, right):
        return {key: results[left][key] - results[right][key] for key in keys}
    output = {label + "-v3": delta(label, "v3") for label in results if label != "v3"}
    if set("ABCD").issubset(results):
        for left, right in (("B", "A"), ("D", "C"), ("C", "A"), ("D", "B")):
            output[left + "-" + right] = delta(left, right)
        output["interaction_(D-C)-(B-A)"] = {
            key: output["D-C"][key] - output["B-A"][key] for key in keys}
    return output


def validate_seals(receipts, expected_labels, study="matched-point", expected_fit_windows=2544, expected_seed=18):
    if len(expected_labels) != len(set(expected_labels)) or set(receipts) != set(expected_labels) or "v3" in receipts:
        raise ValueError("all requested candidate endpoints must be sealed before any tail read")
    if expected_seed not in (18, 19) or (expected_seed != 18 and study != "uq-initialization"):
        raise ValueError("seed19 is only the fixed UQ initialization replication")
    initializations = {label: f"fresh_same_seed{expected_seed}" for label in expected_labels}
    if study == "uq-initialization":
        if set(expected_labels) != {"fresh", "inherit"}:
            raise ValueError("UQ initialization study requires fresh and inherit")
        initializations["inherit"] = "point_checkpoint"
    elif study == "sim-inherited-uq":
        if set(expected_labels) != set(SIM_INHERITED_POINTS):
            raise ValueError("Sim inherited-UQ study requires A_inherit and unchanged inherit")
        initializations = dict.fromkeys(expected_labels, "point_checkpoint")
    elif study == "weighted-quantile":
        if set(expected_labels) != {"inherit", "weighted"}:
            raise ValueError("weighted quantile study requires unchanged inherit and weighted")
        initializations = dict.fromkeys(expected_labels, "point_checkpoint")
    elif study == "uq-ema":
        if set(expected_labels) != {"raw", "ema"}:
            raise ValueError("UQ EMA study requires raw and ema from one trajectory")
        initializations = dict.fromkeys(expected_labels, "point_checkpoint")
    elif study == "real-tke-constant":
        if set(expected_labels) != {"raw", "tke"}:
            raise ValueError("Real TKE study requires raw control and tke candidate")
        initializations = dict.fromkeys(expected_labels, "point_checkpoint")
    elif study == 'real-time-coverage':
        if set(expected_labels) != {'inherit', 'full'}:
            raise ValueError('Real coverage requires prefix control and full candidate')
        initializations = dict.fromkeys(expected_labels, 'point_checkpoint')
    elif study == 'real-reflection':
        if set(expected_labels) != {'control', 'reflection'}:
            raise ValueError('reflection study requires matched control and reflection')
        initializations = dict.fromkeys(expected_labels, 'point_checkpoint')
    elif study != "matched-point":
        raise ValueError("unknown study")
    for label, item in receipts.items():
        if (item.get("schema") != "b210_transfer_uq_v1_receipt" or item.get("label") != label
                or item.get("sealed") is not True or item.get("updates") != 3200
                or item.get("fit_windows") != expected_fit_windows or item.get("seed") != expected_seed
                or item.get("tail_arrays_read") != 0 or item.get("checkpoint_selection") is not False
                or item.get("point_state_bitwise") is not True
                or item.get("initialization") != initializations[label]
                or item.get("manifest_sha256") != MANIFEST_SHA):
            raise ValueError("wrong, incomplete, or tail-trained UQ endpoint: " + label)
    common = ["sample_stream_sha256", "normalizer_sha256", "model_config_sha256"]
    if study == "uq-initialization":
        common += ["point_receipt_sha256", "point_frozen_sha256", "point_source_label"]
        if any(not item.get("initial_uq_sha256") for item in receipts.values()):
            raise ValueError("missing UQ initialization identity")
        if len({item["point_receipt"]["checkpoint_sha256"] for item in receipts.values()}) != 1:
            raise ValueError("UQ initialization study requires the same source point checkpoint")
    elif study == "sim-inherited-uq":
        for label, expected in SIM_INHERITED_POINTS.items():
            item = receipts[label]
            point = item.get("point_receipt", {})
            if (item.get("point_source_label") != expected["arm"]
                    or item.get("point_receipt_sha256") != expected["point_receipt_sha256"]
                    or item.get("point_frozen_sha256") != expected["point_frozen_sha256"]
                    or not item.get("initial_uq_sha256")
                    or any(point.get(key) != expected[key] for key in
                           ("arm", "checkpoint_sha256", "start_sim_sha256"))
                    or point.get("auxiliary_enabled") is not False
                    or point.get("updates") != 4000 or point.get("ema_decay") != .999
                    or point.get("seed") != 4 or point.get("tail_read") is not False):
                raise ValueError("Sim inherited-UQ study requires the original A/C point identities: " + label)
        if receipts["inherit"].get("checkpoint_sha256") != REUSED_C_UQ_SHA:
            raise ValueError("Sim inherited-UQ study must reuse the sealed C UQ checkpoint")
    elif study == "weighted-quantile":
        common.remove("model_config_sha256")
        common += ["point_receipt_sha256", "point_frozen_sha256", "point_source_label", "initial_uq_sha256"]
        expected = SIM_INHERITED_POINTS["inherit"]
        for label, item in receipts.items():
            point = item.get("point_receipt", {})
            if (item.get("point_source_label") != "C"
                    or item.get("point_receipt_sha256") != expected["point_receipt_sha256"]
                    or item.get("point_frozen_sha256") != expected["point_frozen_sha256"]
                    or item.get("initial_uq_sha256") != SOURCE_C_INITIAL_UQ_SHA
                    or point.get("checkpoint_sha256") != expected["checkpoint_sha256"]
                    or point != receipts["inherit"].get("point_receipt")
                    or not item.get("model_config_sha256")):
                raise ValueError("weighted quantile study requires the same original C point and UQ initialization")
        item = receipts["weighted"]
        retained = item.get("retained_initial_uq_sha256", "")
        if (receipts["inherit"].get("checkpoint_sha256") != REUSED_C_UQ_SHA
                or item.get("objective_kind") != "sps-weighted-quantile"
                or item.get("source_initial_uq_sha256") != SOURCE_C_INITIAL_UQ_SHA
                or item.get("removed_state_prefixes") != [WEIGHTED_REMOVED_PREFIX]
                or item.get("model_config_overrides") != {"utility_refinement": False}
                or item["model_config_overrides"]["utility_refinement"] is not False
                or not isinstance(retained, str) or len(retained) != 64
                or any(char not in "0123456789abcdef" for char in retained)
                or retained == SOURCE_C_INITIAL_UQ_SHA):
            raise ValueError("weighted quantile objective or utility-head removal contract changed")
    elif study == "uq-ema":
        validate_uq_ema_seals(receipts)
        common.append("initial_uq_sha256")
    elif study == "real-tke-constant":
        expected = SIM_INHERITED_POINTS["inherit"]
        raw, candidate = receipts["raw"], receipts["tke"]
        if (raw.get("point_receipt_sha256") != expected["point_receipt_sha256"]
                or raw.get("point_frozen_sha256") != expected["point_frozen_sha256"]
                or raw.get("initial_uq_sha256") != SOURCE_C_INITIAL_UQ_SHA
                or raw.get("checkpoint_sha256") != "8a34645d5794a68cf0b7f9123b99604682edf006e8558ff97095b0745fea8512"
                or raw.get("variant") != "raw" or raw.get("ema") is not None
                or candidate.get("ema") is not None):
            raise ValueError("Real TKE control must be original C with raw inherited UQ")
        for label, item in receipts.items():
            point = item.get("point_receipt", {})
            if (item.get("point_source_label") != "C" or item.get("objective_kind") != "sharp2"
                    or not item.get("initial_uq_sha256")
                    or point.get("start_sim_sha256") != expected["start_sim_sha256"]
                    or point.get("auxiliary_enabled") is not False
                    or point.get("updates") != 4000 or point.get("ema_decay") != .999
                    or point.get("seed") != 4 or point.get("tail_read") is not False):
                raise ValueError("Real TKE point/UQ recipe differs: " + label)
        point = candidate["point_receipt"]
        if (point.get("controller_tke_schedule") != "constant_0p01_4000"
                or not point.get("source_commit")
                or point.get("checkpoint_sha256") == expected["checkpoint_sha256"]):
            raise ValueError("Real TKE candidate must bind the new constant-controller schedule")
    elif study in ('real-time-coverage', 'real-reflection'):
        for label, item in receipts.items():
            point = item.get('point_receipt', {})
            full = label == 'full' or study == 'real-reflection'
            if (item.get('point_source_label') != 'C' or item.get('objective_kind') != 'sharp2'
                    or point.get('held_aoa') not in (0, 5, 10, 15, 20)
                    or point.get('start_sim_sha256') != SIM_INHERITED_POINTS['inherit']['start_sim_sha256']
                    or point.get('auxiliary_enabled') is not False or point.get('updates') != 4000
                    or point.get('ema_decay') != .999 or point.get('seed') != 4
                    or point.get('tail_read') is not full):
                raise ValueError('Real coverage point recipe or visibility mismatch')
            if full and (point.get('optimizer_support') != 'full_other_four_trajectories'
                         or point.get('normalizer_support') != 'other_four_prefixes'
                         or point.get('held_data_read') is not False or not point.get('source_commit')):
                raise ValueError('Real coverage requires complete held-AoA exclusion')
            if study == 'real-reflection':
                expected = {'probability': .25, 'isolated_cpu_seed': 29009,
                            'space': 'physical_uvp_before_normalization'} if label == 'reflection' else None
                if point.get('vertical_reflection') != expected:
                    raise ValueError('reflection intervention identity changed')
    else:
        common.append("initial_uq_sha256")
    for key in common:
        values = {item.get(key) for item in receipts.values()}
        if None in values or len(values) != 1:
            raise ValueError("matched UQ contract differs: " + key)
    if study != 'real-time-coverage' and len({item["point_receipt"]["sample_stream_sha256"] for item in receipts.values()}) != 1:
        raise ValueError("point minibatch streams differ")


def validate_uq_ema_seals(receipts):
    raw, averaged = receipts["raw"], receipts["ema"]
    different = {"label", "checkpoint_path", "checkpoint_sha256", "variant", "ema"}
    if ({key: value for key, value in raw.items() if key not in different}
            != {key: value for key, value in averaged.items() if key not in different}):
        raise ValueError("UQ EMA receipts do not bind the same training trajectory")
    expected, point = SIM_INHERITED_POINTS["inherit"], raw.get("point_receipt", {})
    if (raw.get("point_source_label") != "C" or raw.get("objective_kind") != "sharp2"
            or raw.get("point_receipt_sha256") != expected["point_receipt_sha256"]
            or raw.get("point_frozen_sha256") != expected["point_frozen_sha256"]
            or raw.get("initial_uq_sha256") != SOURCE_C_INITIAL_UQ_SHA
            or raw.get("sample_stream_sha256") != UQ_EMA_STREAM_SHA
            or any(point.get(key) != expected[key] for key in ("arm", "checkpoint_sha256", "start_sim_sha256"))):
        raise ValueError("UQ EMA study requires the exact original C point, initialization and stream")
    pair = raw.get("paired_training", {})
    bound = {key: raw.get(key) for key in ("point_receipt_sha256", "initial_uq_sha256", "point_frozen_sha256",
             "sample_stream_sha256", "model_config_sha256", "normalizer_sha256", "seed", "updates")}
    bound.update(schema=UQ_EMA_PAIR_SCHEMA, parameters=3680422,
                 point_checkpoint_sha256=expected["checkpoint_sha256"], non_uq_and_all_buffers_bitwise=True,
                 source_sha256=sha256(ROOT / "code/training/train_b210_transfer_uq.py"))
    source_commit = pair.get("source_commit", "")
    if (any(pair.get(key) != value for key, value in bound.items())
            or not isinstance(source_commit, str) or len(source_commit) != 40
            or any(char not in "0123456789abcdef" for char in source_commit)
            or not pair.get("raw_uq_state_sha256") or not pair.get("ema_uq_state_sha256")
            or pair.get("raw_uq_state_sha256") == pair.get("ema_uq_state_sha256")
            or pair.get("trajectory_sha256") != canonical_sha256({key: value for key, value in pair.items()
                                                                  if key != "trajectory_sha256"})):
        raise ValueError("UQ EMA paired trajectory, source or state binding is invalid")
    ema = averaged.get("ema") or {}
    if (raw.get("variant") != "raw" or raw.get("ema") is not None
            or averaged.get("variant") != "ema999"
            or not raw.get("checkpoint_sha256") or not averaged.get("checkpoint_sha256")
            or raw["checkpoint_sha256"] == averaged["checkpoint_sha256"]
            or ema.get("schema") != "fp32_named_parameter_ema_seal_v1"
            or ema.get("decay") != .999 or ema.get("updates") != 3200
            or ema.get("shadow_dtype") != "torch.float32"
            or ema.get("update_order") != "after_optimizer_step_before_scheduler_step"):
        raise ValueError("UQ EMA study requires distinct raw and UQ-only EMA999 endpoints")


def validate_uq_ema_payloads(payloads, receipts):
    """Check actual paired tensors and the UQ EMA seal before any tail worker."""
    import torch
    from fp32_ema_shadow import tensor_state_sha256
    from train_c208_dr1_clean81_paired_loao import model_state_sha256
    from train_ensemble_uq_oof import UQ_PREFIXES

    pair, ema = receipts["raw"]["paired_training"], receipts["ema"]["ema"]
    states = {}
    for label, payload in payloads.items():
        receipt = receipts[label]
        if (payload.get("schema") != "b210_transfer_uq_v1_endpoint" or payload.get("phase") != "real"
                or payload.get("iteration") != 3200
                or any(payload.get(key) != receipt.get(key) for key in (
                    "initialization", "point_source_label", "point_receipt", "point_receipt_sha256",
                    "initial_uq_sha256", "objective_kind", "variant", "paired_training", "ema"))
                or canonical_sha256(payload["model_config"]) != receipt["model_config_sha256"]
                or canonical_sha256(payload["normalizer"]) != receipt["normalizer_sha256"]):
            raise ValueError("UQ EMA payload metadata differs from its sealed receipt")
        state = states[label] = payload["model_state_dict"]
        frozen = {key: value for key, value in state.items() if not key.startswith(UQ_PREFIXES)}
        uq = {key: value for key, value in state.items() if key.startswith(UQ_PREFIXES)}
        if model_state_sha256(frozen) != receipt["point_frozen_sha256"]:
            raise ValueError("UQ EMA payload changed the frozen point state")
        if model_state_sha256(uq) != pair[label + "_uq_state_sha256"]:
            raise ValueError("UQ EMA payload state differs from the paired trajectory")
    raw, averaged = states["raw"], states["ema"]
    manifest, buffers = ema.get("parameter_manifest", []), ema.get("buffer_names", [])
    names = [entry["name"] for entry in manifest]
    if (set(raw) != set(averaged) or len(names) != len(set(names)) or len(buffers) != len(set(buffers))
            or set(names).intersection(buffers)
            or set(names + buffers) != {key for key in raw if key.startswith(UQ_PREFIXES)}
            or sum(entry["numel"] for entry in manifest) != pair["parameters"]
            or any(raw[key].shape != averaged[key].shape or raw[key].dtype != averaged[key].dtype for key in raw)
            or any(not torch.equal(raw[key], averaged[key]) for key in raw if key not in set(names))):
        raise ValueError("UQ EMA must preserve every non-UQ tensor and all buffers")
    for index, entry in enumerate(manifest):
        value = averaged[entry["name"]]
        if entry != {"index": index, "name": entry["name"], "shape": list(value.shape),
                     "source_dtype": str(value.dtype), "numel": value.numel()}:
            raise ValueError("UQ EMA parameter manifest differs from actual tensors")
    hashes = {
        "parameter_manifest_sha256": canonical_sha256(manifest),
        "seal_metadata_sha256": canonical_sha256({key: value for key, value in ema.items() if key != "seal_metadata_sha256"}),
        "shadow_parameter_state_sha256": tensor_state_sha256({key: averaged[key] for key in names}),
        "buffer_state_sha256": tensor_state_sha256({key: averaged[key] for key in buffers}),
        "raw_model_state_sha256": tensor_state_sha256({key: raw[key] for key in names + buffers}),
        "materialized_model_state_sha256": tensor_state_sha256({key: averaged[key] for key in names + buffers}),
    }
    if any(ema.get(key) != value for key, value in hashes.items()):
        raise ValueError("UQ EMA seal hashes do not match actual raw/EMA tensors")
    return {"trajectory_sha256": pair["trajectory_sha256"], "non_uq_and_all_buffers_bitwise": True,
            "point_frozen_sha256": receipts["raw"]["point_frozen_sha256"], "uq_parameters": pair["parameters"],
            "actual_ema_seal_verified": True}


def validate_prediction_streams(results, study):
    if study in {"uq-initialization", "weighted-quantile", "uq-ema"}:
        labels = {"uq-initialization": ("fresh", "inherit"), "weighted-quantile": ("weighted", "inherit"),
                  "uq-ema": ("raw", "ema")}[study]
        hashes = [results[label].get("prediction_sha256") for label in labels]
        if any(not value for value in hashes) or hashes[0] != hashes[1]:
            raise ValueError("UQ initialization changed deployed point predictions")


def weighted_runtime_configs(payloads, receipts, template_config):
    """Admit exactly utility-head removal before any tail replay; keep payload config truthful."""
    import dataclasses
    from models.ensemble_unet2d import EnsembleUNet2dConfig
    from train_c208_dr1_clean81_paired_loao import canonical_sha256, model_state_sha256
    from train_ensemble_uq_oof import UQ_PREFIXES

    reference = dict(template_config)
    reference.pop("architecture")
    reference = dataclasses.asdict(EnsembleUNet2dConfig(**reference))
    output = {}
    for label in ("inherit", "weighted"):
        payload, receipt = payloads[label], receipts[label]
        config = dict(payload["model_config"])
        if canonical_sha256(config) != receipt["model_config_sha256"]:
            raise ValueError("weighted study payload model-config SHA differs from receipt")
        if config.pop("architecture") != "ensemble_unet":
            raise ValueError("weighted study payload architecture changed")
        expected = dict(reference)
        if label == "weighted":
            expected["utility_refinement"] = False
            if config.get("utility_refinement") is not False:
                raise ValueError("weighted utility refinement must be explicitly disabled")
            for key in ("objective_kind", "initial_uq_sha256", "source_initial_uq_sha256",
                        "retained_initial_uq_sha256", "removed_state_prefixes", "model_config_overrides"):
                if payload.get(key) != receipt[key]:
                    raise ValueError("weighted payload metadata differs from receipt: " + key)
        candidate_config = dataclasses.asdict(EnsembleUNet2dConfig(**config))
        if candidate_config != expected:
            raise ValueError("weighted study permits only utility_refinement=False")
        state = payload["model_state_dict"]
        frozen = {key: value for key, value in state.items() if not key.startswith(UQ_PREFIXES)}
        if model_state_sha256(frozen) != receipt["point_frozen_sha256"]:
            raise ValueError("weighted study payload changed the frozen point state")
        # The actual-v3 schema lacks newer default-only fields. Validate the full
        # config above, then project its values onto the runtime's supported keys.
        output[label] = ({key: ("ensemble_unet" if key == "architecture" else candidate_config[key])
                          for key in template_config} if label == "weighted" else template_config)
    original, weighted = (payloads[label]["model_state_dict"] for label in ("inherit", "weighted"))
    removed = {key for key in original if key.startswith(WEIGHTED_REMOVED_PREFIX)}
    if not removed or set(weighted) != set(original) - removed:
        raise ValueError("weighted payload must remove only utility_refinement_head state")
    if any(weighted[key].shape != original[key].shape or weighted[key].dtype != original[key].dtype
           for key in weighted):
        raise ValueError("weighted retained state shapes or dtypes changed")
    return output


def clean_rows(path):
    if sha256(path) != MANIFEST_SHA:
        raise ValueError("clean-prefix manifest SHA changed")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 81 or sum(len(row["train_starts"]) for row in rows) != 2544:
        raise ValueError("wrong clean81 prefix manifest")
    total = 0
    for row in rows:
        name = row["file"]
        if Path(name).name != name or name == "7575_0.h5":
            raise ValueError("unsafe or excluded trajectory")
        starts = list(range(0, int(row["frames"]) - 39, 20))
        row["val_starts"] = starts[-max(1, math.ceil(len(starts) * .2)):]
        if row["val_starts"][0] != row["val_boundary"]:
            raise ValueError("tail boundary drift")
        total += len(row["val_starts"])
    if total != 716:
        raise ValueError("wrong tail count")
    return rows


def tail_batches(rows, real_dir):
    import h5py
    import numpy as np
    root = real_dir.resolve()
    xs, ys, labels = [], [], []
    for row in rows:
        path = (root / row["file"]).resolve()
        if path.parent != root:
            raise ValueError("trajectory escaped the Real data root")
        boundary = row["val_boundary"]
        with h5py.File(path, "r") as handle:
            if (handle["u"].shape != handle["v"].shape or len(handle["u"]) != row["frames"]
                    or int(handle["aoa"][()]) != row["aoa"] or int(handle["re"][()]) != row["re"]):
                raise ValueError("trajectory metadata drift")
            u = np.asarray(handle["u"][boundary:, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][boundary:, ::2, ::2], dtype=np.float32)
        values = np.stack((u, v, np.zeros_like(u)), axis=-1)
        for start in row["val_starts"]:
            offset = start - boundary
            xs.append(values[offset:offset + 20]); ys.append(values[offset + 20:offset + 40])
            labels.append(row["file"])
            if len(xs) == 16:
                yield np.stack(xs), np.stack(ys), labels
                xs, ys, labels = [], [], []
    if xs:
        yield np.stack(xs), np.stack(ys), labels


def worker(args):
    import numpy as np
    import torch
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("formal quality replay requires CUDA")
    torch.cuda.set_device(torch.device(args.device))
    sys.path.insert(0, str(args.worker_runtime))
    submission = import_file(args.worker_runtime / "submission.py", "transfer_submission")
    official = import_file(SCORER, "transfer_official_scoring")
    rows = clean_rows(args.clean_manifest)
    if getattr(args, 'held_aoa', None) is not None:
        rows = [r for r in rows if r['aoa'] == args.held_aoa]
    states = {row["file"]: [] for row in rows}
    input_hash, target_hash, prediction_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for inputs, truth, labels in tail_batches(rows, args.real_dir):
        input_hash.update(inputs.tobytes()); target_hash.update(truth.tobytes())
        output = submission.predict(inputs)
        prediction_hash.update(output["prediction"].tobytes())
        for label in dict.fromkeys(labels):
            mask = np.asarray([value == label for value in labels])
            states[label].append(metric_state(official, output["prediction"][mask], truth[mask],
                                              output["lower"][mask], output["upper"][mask]))
    states = {label: merge_states(values) for label, values in states.items()}
    pooled = merge_states(list(states.values()))
    if pooled["samples"] != sum(len(r['val_starts']) for r in rows):
        raise ValueError("quality replay did not consume its fixed fold exactly")
    by_aoa = {str(aoa): finalize(official, merge_states([states[row["file"]] for row in rows if row["aoa"] == aoa]))
              for aoa in sorted({r['aoa'] for r in rows})}
    result = {"label": args.worker_label, "metrics": finalize(official, pooled), "pooled_state": pooled,
              "trajectory_states": states, "by_aoa": by_aoa,
              "by_trajectory": {key: finalize(official, state) for key, state in states.items()},
              "input_sha256": input_hash.hexdigest(), "target_sha256": target_hash.hexdigest(),
              "prediction_sha256": prediction_hash.hexdigest(),
              "scorer_sha256": sha256(SCORER), "precision": "v3 wrapper CUDA FP16, INT8 storage",
              "timing_measured": False}
    args.worker_output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def execute(args):
    started = time.perf_counter()
    required = (args.baseline_zip, args.clean_manifest, args.real_dir, args.output, args.scratch_dir, args.wall_budget_seconds)
    if any(value is None for value in required) or args.wall_budget_seconds <= 0 or not math.isfinite(args.wall_budget_seconds):
        raise ValueError("execution requires baseline, manifest, data, output, scratch and finite wall budget")
    if args.output.exists() or not args.scratch_dir.is_dir():
        raise ValueError("output must be new and scratch directory must exist")
    if args.study == "sim-inherited-uq" and args.compare != ["inherit:A_inherit"]:
        raise ValueError("Sim inherited-UQ primary comparison is fixed to inherit:A_inherit")
    if args.study == "weighted-quantile" and args.compare != ["weighted:inherit"]:
        raise ValueError("weighted quantile primary comparison is fixed to weighted:inherit")
    if args.study == "uq-ema" and (args.compare != ["ema:raw"] or args.wall_budget_seconds > 600):
        raise ValueError("UQ EMA primary comparison is fixed to ema:raw with at most 600 seconds")
    if args.study == "real-tke-constant" and (args.compare != ["tke:raw"] or args.wall_budget_seconds > 600):
        raise ValueError("Real TKE primary comparison is fixed to tke:raw with at most 600 seconds")
    receipts, receipt_paths = {}, {}
    for item in args.uq_receipt:
        label, value = item.split("=", 1)
        if label in receipts:
            raise ValueError("duplicate label")
        receipt_paths[label] = Path(value)
        receipts[label] = json.loads(Path(value).read_text(encoding="utf-8"))
    held_aoa = getattr(args, 'held_aoa', None)
    fit_rows = ([r for r in clean_rows(args.clean_manifest) if r['aoa'] != held_aoa]
                if held_aoa is not None else None)
    if held_aoa is not None:
        if (args.study, args.compare) not in (('uq-initialization', ['inherit:fresh']),
                                              ('real-time-coverage', ['full:inherit']),
                                              ('real-reflection', ['reflection:control'])):
            raise ValueError('stage confirmation comparison must match the fixed study')
        for item in receipts.values():
            point = item.get('point_receipt', {})
            if (item.get('held_aoa') != held_aoa or point.get('held_aoa') != held_aoa
                    or point.get('fit_files') != [r['file'] for r in fit_rows]):
                raise ValueError('stage confirmation training fold mismatch')
    elif any(item.get('held_aoa') is not None for item in receipts.values()):
        raise ValueError('stage-held endpoints cannot silently enter all-clean evaluation')
    if args.study in ('real-time-coverage', 'real-reflection') and held_aoa is None:
        raise ValueError('full-training coverage must be evaluated on a held AoA')
    validate_seals(receipts, args.expected_labels, args.study,
                   expected_fit_windows=(sum(len(r['train_starts']) for r in fit_rows)
                                         if fit_rows is not None else 2544),
                   expected_seed=getattr(args, "uq_seed", 18))
    if (args.study == "real-tke-constant" and sha256(receipt_paths["raw"])
            != "c2d574dd7ac5e1817e1794385471a69b7b6acda5c9d909c6b92a6c87e84efaab"):
        raise ValueError("Real TKE must reuse the fixed H800 raw control receipt")
    if (args.study in {"sim-inherited-uq", "weighted-quantile"}
            and sha256(receipt_paths["inherit"]) != REUSED_C_UQ_RECEIPT_SHA):
        raise ValueError("reused C UQ receipt bytes changed")
    if sha256(args.baseline_zip) != V3_SHA:
        raise ValueError("actual v3 ZIP SHA changed")
    for item in receipts.values():
        if sha256(item["checkpoint_path"]) != item["checkpoint_sha256"]:
            raise ValueError("sealed UQ checkpoint changed")
    clean_rows(args.clean_manifest)  # metadata only; no HDF5 before every seal is verified

    import dataclasses
    import torch
    sys.path.insert(0, str(ROOT / "code/submissions"))
    sys.path.insert(0, str(ROOT / "code"))
    if args.study in {"weighted-quantile", "uq-ema"}:
        sys.path.insert(0, str(ROOT / "code/training"))
    from build_track1_blended_ensemble_unet2d_package import compact_state_dict
    from models.ensemble_unet2d import EnsembleUNet2dConfig
    deadline = (started if args.study == "uq-ema" else time.perf_counter()) + args.wall_budget_seconds
    results, runtime_bindings = {}, {}
    with zipfile.ZipFile(args.baseline_zip) as archive, tempfile.TemporaryDirectory(prefix="transfer_eval_", dir=args.scratch_dir) as temporary:
        runtime = Path(temporary).resolve()
        if runtime.parent != args.scratch_dir.resolve():
            raise ValueError("temporary runtime escaped scratch root")
        for name in RUNTIME_FILES:
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
        checkpoint = runtime / "ensemble_unet2d.pth"
        checkpoint.write_bytes(archive.read("ensemble_unet2d.pth"))
        template = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reference_config = dict(template["model_config"])
        reference_config.pop("architecture")
        reference_config = dataclasses.asdict(EnsembleUNet2dConfig(**reference_config))
        weighted_payloads, weighted_configs = {}, {}
        ema_payloads, ema_audit = {}, None
        if args.study == "weighted-quantile":
            weighted_payloads = {label: torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
                                 for label, item in receipts.items()}
            weighted_configs = weighted_runtime_configs(weighted_payloads, receipts, template["model_config"])
        if args.study == "uq-ema":
            ema_payloads = {label: torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
                            for label, item in receipts.items()}
            ema_audit = validate_uq_ema_payloads(ema_payloads, receipts)
            for payload in ema_payloads.values():
                config = dict(payload["model_config"]); config.pop("architecture")
                if dataclasses.asdict(EnsembleUNet2dConfig(**config)) != reference_config:
                    raise ValueError("UQ EMA architecture is incompatible with actual-v3 runtime")
        for label in ["v3", *args.expected_labels]:
            if label != "v3":
                payload = (weighted_payloads.pop(label) if args.study == "weighted-quantile"
                           else ema_payloads.pop(label) if args.study == "uq-ema"
                           else torch.load(receipts[label]["checkpoint_path"], map_location="cpu", weights_only=False))
                config = dict(payload["model_config"]); config.pop("architecture")
                candidate_config = dataclasses.asdict(EnsembleUNet2dConfig(**config))
                if args.study != "weighted-quantile" and candidate_config != reference_config:
                    raise ValueError("candidate architecture is incompatible with actual-v3 runtime")
                compact, statistics = compact_state_dict(payload["model_state_dict"], 1024)
                torch.save({"model_config": weighted_configs.get(label, template["model_config"]), "quantized_state_dict": compact,
                            "normalizer": payload["normalizer"], "phase": "real", "inference_dtype": "float16",
                            "storage_format": "symmetric_per_dim0_int8_v1"}, checkpoint)
                del payload, compact
            runtime_bindings[label] = {"checkpoint_sha256": sha256(checkpoint),
                                       "source_members": {name: sha256(runtime / name) for name in RUNTIME_FILES}}
            output = runtime / "worker_result.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--worker-runtime", str(runtime),
                       "--worker-label", label, "--worker-output", str(output), "--device", args.device,
                       "--clean-manifest", str(args.clean_manifest), "--real-dir", str(args.real_dir), "--study", args.study]
            if held_aoa is not None:
                command += ['--held-aoa', str(held_aoa)]
            subprocess.run(command, check=True, timeout=max(.001, deadline - time.perf_counter()))
            results[label] = json.loads(output.read_text(encoding="utf-8"))
    for key in ("input_sha256", "target_sha256", "scorer_sha256"):
        if len({result[key] for result in results.values()}) != 1:
            raise ValueError("replay inputs or scorer differed: " + key)
    validate_prediction_streams(results, args.study)
    metrics = {label: result["metrics"] for label, result in results.items()}
    comparison = contrasts(metrics)
    for pair in args.compare:
        left, right = pair.split(":", 1)
        comparison[left + "-" + right] = {key: metrics[left][key] - metrics[right][key]
                                         for key in (*METRICS, "fixed_time_fitted_final")}
    result = {"schema": "b210_transfer_quality_v1", "evidence": "public clean716 matched deployment-quality diagnostic, not official gain or full-pipeline LOAO",
              "held_aoa": held_aoa,
              "holdout_scope": ('Real/UQ stages only; shared Sim ancestors not held out; v3 is descriptive, not matched fold control'
                                if held_aoa is not None else 'temporal-tail diagnostic'),
              "actual_v3_zip_sha256": V3_SHA, "fixed_time_score": TIME_SCORE,
              "study": args.study,
              "final_formula": {"p": .6854, "weights_Rel_TKE_MVPE_Time_SPS": [.5, .1, .1, .1, .2]},
              "receipts": {label: {"path": str(path), "sha256": sha256(path)} for label, path in receipt_paths.items()},
              "runtime_bindings": runtime_bindings, "arms": results, "contrasts": comparison,
              "point_pre_gate_used": args.study == 'real-reflection', "release_verified": False, "official_improvement_claimed": False}
    if args.study == "uq-ema":
        result["uq_ema_pair_audit"] = {**ema_audit, "deployed_point_predictions_bitwise": True,
                                      "prediction_sha256": results["raw"]["prediction_sha256"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"output": str(args.output), "contrasts": comparison, "release_verified": False}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-labels", nargs="+", default=list("ABCD"))
    parser.add_argument('--held-aoa', type=int, choices=(0, 5, 10, 15, 20))
    parser.add_argument("--uq-seed", type=int, choices=(18, 19), default=18)
    parser.add_argument("--study", choices=("matched-point", "uq-initialization", "sim-inherited-uq", "weighted-quantile", "uq-ema", "real-tke-constant", "real-time-coverage", "real-reflection"), default="matched-point")
    parser.add_argument("--uq-receipt", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--compare", action="append", default=[], metavar="LEFT:RIGHT")
    for name in ("baseline-zip", "clean-manifest", "real-dir", "output", "scratch-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float)
    for name in ("worker-runtime", "worker-output"):
        parser.add_argument("--" + name, type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.study == 'real-reflection':
        TIME_SCORE = 92.032660
    if not args.execute:
        uq_plan = {
            "matched-point": "fresh-seed18 prefix2544 u3200",
            "uq-initialization": f"seed{args.uq_seed} prefix2544 u3200; fresh versus inherited point UQ",
            "sim-inherited-uq": "one new A_inherit prefix2544 u3200; unchanged sealed C inherit; point initialization",
            "weighted-quantile": "same C point UQ; new weighted prefix2544 u3200 with utility head removed versus sealed inherit",
            "uq-ema": "same original C point and inherited UQ; one prefix2544 u3200 trajectory, raw versus UQ-only EMA999",
            "real-tke-constant": "same original C Sim; constant controller-only Real TKE versus original C, raw inherited UQ3200",
            "real-time-coverage": "same old C Sim; full other-four Real trajectories versus prefixes; same prefix normalizer and inherited UQ3200",
            "real-reflection": "same full Real support and old C Sim; 25 percent reflected training versus control, inherited prefix UQ3200",
        }
        print(json.dumps({"decision": "PLAN_ONLY_NO_DATA_NO_GPU", "expected_labels": args.expected_labels,
                          "requires_all_sealed_before_tail": True, "study": args.study,
                          "UQ": uq_plan[args.study],
                          "runtime": "actual-v3 sources; common INT8-storage/FP16", "fixed_time_score": TIME_SCORE,
                          "timing_gate": False, "release_or_upload": False}, indent=2))
    elif args.worker_runtime:
        worker(args)
    else:
        print(json.dumps(execute(args), indent=2))
