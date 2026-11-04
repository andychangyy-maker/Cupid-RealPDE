"""Check the real scorer aggregation and paired-endpoint admission on CPU."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = load(Path(__file__).resolve().parents[1] / "evaluate_b210_transfer.py", "transfer_eval_test")
official = load(subject.SCORER, "realpde_scorer_test")


def test_partitioned_scores_match_official_full_array_with_unequal_masks():
    rng = np.random.default_rng(41)
    truth = rng.normal(.1, .05, (5, 20, 32, 64, 3)).astype(np.float32)
    truth[..., 2] = 0.
    truth[0, :, :18, :, :2] = 0.
    truth[3, :, :7, :, :2] = 0.
    prediction = truth + rng.normal(0., .015, truth.shape).astype(np.float32)
    prediction[..., 2] = 0.
    lower, upper = prediction - .023, prediction + .041
    lower[..., 2] = upper[..., 2] = 0.
    parts = [subject.metric_state(official, prediction[s], truth[s], lower[s], upper[s])
             for s in (slice(0, 1), slice(1, 4), slice(4, 5))]
    scores = subject.finalize(official, subject.merge_states(parts))
    sps, coverage = official.aggregate_sps(prediction, truth, 2, lower, upper)
    expected = {
        "rel_l2_score": official.score_error(np.mean(official.rel_l2_per_sample(prediction, truth, 2), dtype=np.float64)),
        "tke_rel_l2_score": official.score_error(np.mean(official.tke_rel_l2_per_sample(prediction, truth, 2), dtype=np.float64)),
        "mvpe_rel_l2_score": official.score_error(np.mean(official.mvpe_rel_l2_per_sample(prediction, truth), dtype=np.float64)),
        "sps_score": official.score_sps(sps), "coverage": coverage,
    }
    for name, value in expected.items():
        assert scores[name] == pytest.approx(value, abs=1e-6)
    shifted = prediction.copy()
    shifted[..., :2] += .04
    changed, _ = official.aggregate_sps(shifted, truth, 2, lower, upper)
    assert changed != pytest.approx(sps)  # fixed bounds do not imply fixed SPS


def receipts():
    return {label: {
        "schema": "b210_transfer_uq_v1_receipt", "label": label, "sealed": True,
        "updates": 3200, "fit_windows": 2544, "seed": 18, "tail_arrays_read": 0,
        "checkpoint_selection": False, "point_state_bitwise": True,
        "initialization": "fresh_same_seed18", "manifest_sha256": subject.MANIFEST_SHA,
        "initial_uq_sha256": "same-uq", "sample_stream_sha256": "same-uq-stream",
        "normalizer_sha256": "same-stats", "model_config_sha256": "same-config",
        "point_receipt": {"sample_stream_sha256": "same-point-stream"},
    } for label in "ABCD"}


def test_missing_endpoint_or_unequal_training_contract_rejects_before_replay():
    original = receipts()
    subject.validate_seals(original, list("ABCD"))
    for key, value in (("initial_uq_sha256", "different"), ("tail_arrays_read", 1),
                       ("checkpoint_selection", True), ("point_state_bitwise", False)):
        broken = copy.deepcopy(original)
        broken["D"][key] = value
        with pytest.raises(ValueError):
            subject.validate_seals(broken, list("ABCD"))
    del original["D"]
    with pytest.raises(ValueError):
        subject.validate_seals(original, list("ABCD"))


def test_interaction_uses_complete_actual_subscores_and_preserves_sign():
    metrics = {label: {key: value for key in (*subject.METRICS, "fixed_time_fitted_final")}
               for label, value in {"v3": 80., "A": 70., "B": 71., "C": 72., "D": 75.}.items()}
    contrast = subject.contrasts(metrics)
    assert set(contrast["D-C"].values()) == {3.}
    assert set(contrast["interaction_(D-C)-(B-A)"].values()) == {2.}
    assert set(contrast["D-v3"].values()) == {-5.}


def test_uq_initialization_requires_explicit_mode_and_same_point():
    pair = {label: copy.deepcopy(receipts()["A"]) for label in ("fresh", "inherit")}
    for label, item in pair.items():
        item.update(label=label, point_receipt_sha256="same-receipt",
                    point_frozen_sha256="same-frozen", point_source_label="C")
        item["point_receipt"]["checkpoint_sha256"] = "same-point"
    pair["inherit"].update(initialization="point_checkpoint", initial_uq_sha256="trained-uq")
    subject.validate_seals(pair, list(pair), "uq-initialization")
    with pytest.raises(ValueError):
        subject.validate_seals(pair, list(pair))
    for key, value in (("point_receipt_sha256", "other"), ("point_frozen_sha256", "other"),
                       ("point_source_label", "D"), ("sample_stream_sha256", "other"),
                       ("tail_arrays_read", 1), ("sealed", False)):
        broken = copy.deepcopy(pair)
        broken["inherit"][key] = value
        with pytest.raises(ValueError):
            subject.validate_seals(broken, list(pair), "uq-initialization")
    pair["inherit"]["point_receipt"]["checkpoint_sha256"] = "other-point"
    with pytest.raises(ValueError):
        subject.validate_seals(pair, list(pair), "uq-initialization")


def test_same_point_study_rejects_changed_deployed_predictions():
    results = {"fresh": {"prediction_sha256": "same"}, "inherit": {"prediction_sha256": "same"},
               "v3": {"prediction_sha256": "different-baseline"}}
    subject.validate_prediction_streams(results, "uq-initialization")
    results["inherit"]["prediction_sha256"] = "changed"
    with pytest.raises(ValueError, match="point predictions"):
        subject.validate_prediction_streams(results, "uq-initialization")
    subject.validate_prediction_streams(results, "matched-point")


def test_replication_rejects_mixed_seed_and_preserves_old_seed18_contract():
    pair = {label: copy.deepcopy(receipts()['A']) for label in ('fresh', 'inherit')}
    for label, item in pair.items():
        item.update(label=label, seed=19, point_receipt_sha256='same-receipt',
                    point_frozen_sha256='same-point', point_source_label='C',
                    initialization='fresh_same_seed19' if label == 'fresh' else 'point_checkpoint')
        item['point_receipt']['checkpoint_sha256'] = 'same-point'
    subject.validate_seals(pair, list(pair), 'uq-initialization', expected_seed=19)
    with pytest.raises(ValueError):
        subject.validate_seals(pair, list(pair), 'uq-initialization')
    pair['inherit']['seed'] = 18
    with pytest.raises(ValueError):
        subject.validate_seals(pair, list(pair), 'uq-initialization', expected_seed=19)


def inherited_sim_pair():
    pair = {}
    for label, expected in subject.SIM_INHERITED_POINTS.items():
        item = copy.deepcopy(receipts()["A"])
        item.update(label=label, initialization="point_checkpoint",
                    initial_uq_sha256="trained-uq-" + label,
                    point_source_label=expected["arm"],
                    point_receipt_sha256=expected["point_receipt_sha256"],
                    point_frozen_sha256=expected["point_frozen_sha256"],
                    checkpoint_sha256=subject.REUSED_C_UQ_SHA if label == "inherit" else "new-A-uq")
        item["point_receipt"].update(
            {key: expected[key] for key in ("arm", "checkpoint_sha256", "start_sim_sha256")})
        item["point_receipt"].update(auxiliary_enabled=False, updates=4000,
                                     ema_decay=.999, seed=4, tail_read=False)
        pair[label] = item
    return pair


def test_sim_inherited_uq_allows_distinct_bound_points_without_loosening_other_studies():
    pair = inherited_sim_pair()
    subject.validate_seals(pair, list(pair), "sim-inherited-uq")
    subject.validate_prediction_streams(
        {label: {"prediction_sha256": label} for label in pair}, "sim-inherited-uq")
    for study in ("matched-point", "uq-initialization"):
        with pytest.raises(ValueError):
            subject.validate_seals(pair, list(pair), study)
    with pytest.raises(ValueError):
        subject.validate_seals({"inherit": pair["inherit"]}, ["inherit"], "sim-inherited-uq")


def test_sim_inherited_uq_rejects_wrong_endpoint_or_changed_training_contract():
    for key, value in (("point_source_label", "D"), ("point_receipt_sha256", "other"),
                       ("point_frozen_sha256", "other"), ("initialization", "fresh_same_seed18"),
                       ("initial_uq_sha256", None), ("sample_stream_sha256", "other"),
                       ("normalizer_sha256", "other"), ("model_config_sha256", "other"),
                       ("seed", 19), ("tail_arrays_read", 1), ("point_state_bitwise", False)):
        pair = inherited_sim_pair()
        pair["A_inherit"][key] = value
        with pytest.raises(ValueError):
            subject.validate_seals(pair, list(pair), "sim-inherited-uq")
    for key, value in (("arm", "B"), ("checkpoint_sha256", "other"),
                       ("start_sim_sha256", "other"), ("auxiliary_enabled", True)):
        pair = inherited_sim_pair()
        pair["A_inherit"]["point_receipt"][key] = value
        with pytest.raises(ValueError):
            subject.validate_seals(pair, list(pair), "sim-inherited-uq")
    pair = inherited_sim_pair()
    pair["inherit"]["checkpoint_sha256"] = "a-new-C-uq-run"
    with pytest.raises(ValueError, match="reuse"):
        subject.validate_seals(pair, list(pair), "sim-inherited-uq")


@pytest.mark.parametrize("study", ("sim-inherited-uq", "weighted-quantile"))
def test_reused_c_receipt_and_fixed_control_are_checked_before_data(tmp_path, monkeypatch, study):
    pair = inherited_sim_pair() if study == "sim-inherited-uq" else weighted_pair()
    comparison = "inherit:A_inherit" if study == "sim-inherited-uq" else "weighted:inherit"
    receipt_args = []
    for label, item in pair.items():
        path = tmp_path / (label + ".json")
        path.write_text(json.dumps(item), encoding="utf-8")
        receipt_args += ["--uq-receipt", label + "=" + str(path)]
    args = subject.parse_args([
        "--execute", "--study", study, "--expected-labels", *pair,
        "--compare", comparison, *receipt_args,
        "--baseline-zip", str(tmp_path / "v3.zip"), "--clean-manifest", str(tmp_path / "manifest.json"),
        "--real-dir", str(tmp_path), "--scratch-dir", str(tmp_path),
        "--output", str(tmp_path / "result.json"), "--wall-budget-seconds", "600"])
    monkeypatch.setattr(subject, "clean_rows", lambda _: pytest.fail("data admitted before receipt verification"))
    with pytest.raises(ValueError, match="receipt bytes changed"):
        subject.execute(args)
    args.compare = ["inherit:v3"]
    with pytest.raises(ValueError, match="primary comparison"):
        subject.execute(args)


def weighted_pair():
    original = inherited_sim_pair()["inherit"]
    original["initial_uq_sha256"] = subject.SOURCE_C_INITIAL_UQ_SHA
    weighted = copy.deepcopy(original)
    weighted.update(label="weighted", checkpoint_sha256="new-weighted-uq",
                    model_config_sha256="changed-config", objective_kind="sps-weighted-quantile",
                    source_initial_uq_sha256=subject.SOURCE_C_INITIAL_UQ_SHA,
                    retained_initial_uq_sha256="a" * 64,
                    removed_state_prefixes=[subject.WEIGHTED_REMOVED_PREFIX],
                    model_config_overrides={"utility_refinement": False})
    return {"inherit": original, "weighted": weighted}


def test_weighted_seals_allow_only_declared_removal_and_preserve_same_point_contract():
    pair = weighted_pair()
    subject.validate_seals(pair, list(pair), "weighted-quantile")
    for key, value in (("point_source_label", "A"), ("initial_uq_sha256", "other"),
                       ("source_initial_uq_sha256", "other"), ("retained_initial_uq_sha256", None),
                       ("point_frozen_sha256", "other"), ("sample_stream_sha256", "other"),
                       ("normalizer_sha256", "other"), ("tail_arrays_read", 1),
                       ("point_state_bitwise", False), ("objective_kind", "sharp2"),
                       ("model_config_overrides", {"utility_refinement": False, "base": 4}),
                       ("removed_state_prefixes", ["quantile_head."])):
        broken = copy.deepcopy(pair)
        broken["weighted"][key] = value
        with pytest.raises(ValueError):
            subject.validate_seals(broken, list(broken), "weighted-quantile")
    pair["weighted"]["point_receipt"]["checkpoint_sha256"] = "different-point"
    with pytest.raises(ValueError):
        subject.validate_seals(pair, list(pair), "weighted-quantile")
    streams = {label: {"prediction_sha256": "same-point"} for label in pair}
    subject.validate_prediction_streams(streams, "weighted-quantile")
    streams["weighted"]["prediction_sha256"] = "changed"
    with pytest.raises(ValueError, match="point predictions"):
        subject.validate_prediction_streams(streams, "weighted-quantile")


def tiny_weighted_payloads():
    import dataclasses
    import sys
    import torch
    for path in (subject.ROOT / "code", subject.ROOT / "code/training"):
        sys.path.insert(0, str(path))
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from train_c208_dr1_clean81_paired_loao import canonical_sha256, model_state_sha256
    from train_ensemble_uq_oof import UQ_PREFIXES
    torch.set_num_threads(1)
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=4, groups=2, uq_hidden_channels=4, utility_refinement=True,
        residual_quantile_intervals=True)).float()
    config = {"architecture": "ensemble_unet", **dataclasses.asdict(model.config)}
    state = model.state_dict()
    frozen_sha = model_state_sha256({k: v for k, v in state.items() if not k.startswith(UQ_PREFIXES)})
    payloads = {"inherit": {"model_config": config, "model_state_dict": state},
                "weighted": {"model_config": {**config, "utility_refinement": False},
                             "model_state_dict": {k: v.clone() for k, v in state.items()
                                                  if not k.startswith(subject.WEIGHTED_REMOVED_PREFIX)}}}
    pair = weighted_pair()
    for label, item in pair.items():
        item["model_config_sha256"] = canonical_sha256(payloads[label]["model_config"])
        item["point_frozen_sha256"] = frozen_sha
    for key in ("objective_kind", "initial_uq_sha256", "source_initial_uq_sha256",
                "retained_initial_uq_sha256", "removed_state_prefixes", "model_config_overrides"):
        payloads["weighted"][key] = pair["weighted"][key]
    return payloads, pair, config


def test_weighted_runtime_serializes_truthful_disabled_config_and_loads_exact_retained_state(tmp_path):
    import torch
    payloads, pair, template = tiny_weighted_payloads()
    configs = subject.weighted_runtime_configs(payloads, pair, template)
    assert configs["inherit"] is template
    assert configs["weighted"]["utility_refinement"] is False
    assert set(configs["weighted"]) == set(template)
    path = tmp_path / "runtime.pth"
    torch.save({"model_config": configs["weighted"], "model_state_dict": payloads["weighted"]["model_state_dict"]}, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(saved["model_config"])
    config.pop("architecture")
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    runtime = EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
    assert runtime.utility_refinement_head is None
    runtime.load_state_dict(saved["model_state_dict"], strict=True)


def test_weighted_payload_preflight_rejects_bad_config_hash_extra_override_or_extra_state_removal():
    payloads, pair, template = tiny_weighted_payloads()
    from train_c208_dr1_clean81_paired_loao import canonical_sha256
    pair["weighted"]["model_config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="config SHA"):
        subject.weighted_runtime_configs(payloads, pair, template)
    payloads["weighted"]["model_config"]["utility_refinement_log_limit"] = .5
    pair["weighted"]["model_config_sha256"] = canonical_sha256(payloads["weighted"]["model_config"])
    with pytest.raises(ValueError, match="only utility_refinement"):
        subject.weighted_runtime_configs(payloads, pair, template)
    payloads, pair, template = tiny_weighted_payloads()
    state = payloads["weighted"]["model_state_dict"]
    key = next(key for key in state if key.startswith("quantile_head."))
    state.pop(key)
    with pytest.raises(ValueError, match="remove only"):
        subject.weighted_runtime_configs(payloads, pair, template)
    payloads, pair, template = tiny_weighted_payloads()
    state = payloads["weighted"]["model_state_dict"]
    from train_ensemble_uq_oof import UQ_PREFIXES
    key = next(key for key in state if not key.startswith(UQ_PREFIXES) and state[key].is_floating_point())
    state[key] = state[key] + 1
    with pytest.raises(ValueError, match="frozen point"):
        subject.weighted_runtime_configs(payloads, pair, template)


def test_weighted_projection_is_accepted_by_actual_v3_config(tmp_path, monkeypatch):
    import dataclasses
    import os
    import sys
    import types
    import zipfile
    package = os.environ.get("REALPDE_V3_ZIP")
    if not package:
        pytest.skip("set REALPDE_V3_ZIP for the small actual-runtime schema check")
    module_name = "weighted_actual_v3_models"
    module = types.ModuleType(module_name)
    module.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, module_name, module)
    with zipfile.ZipFile(package) as archive:
        for name in ("unet2d.py", "ensemble_unet2d.py"):
            (tmp_path / name).write_bytes(archive.read("models/" + name))
    actual = subject.import_file(tmp_path / "ensemble_unet2d.py", module_name + ".ensemble_unet2d")
    payloads, pair, full_template = tiny_weighted_payloads()
    supported = {field.name for field in dataclasses.fields(actual.EnsembleUNet2dConfig)}
    template = {key: value for key, value in full_template.items() if key == "architecture" or key in supported}
    assert set(full_template) - set(template)  # real newer defaults must be omitted
    projected = subject.weighted_runtime_configs(payloads, pair, template)["weighted"]
    assert set(projected) == set(template)
    assert projected.pop("architecture") == "ensemble_unet"
    config = actual.EnsembleUNet2dConfig(**projected)
    assert config.utility_refinement is False
    assert config.utility_refinement_log_limit > 0


def uq_ema_pair():
    raw = inherited_sim_pair()["inherit"]
    raw.update(label="raw", initial_uq_sha256=subject.SOURCE_C_INITIAL_UQ_SHA,
               sample_stream_sha256=subject.UQ_EMA_STREAM_SHA, objective_kind="sharp2",
               checkpoint_sha256="a" * 64, variant="raw", ema=None)
    trajectory = {key: raw[key] for key in ("point_receipt_sha256", "initial_uq_sha256", "point_frozen_sha256",
                  "sample_stream_sha256", "model_config_sha256", "normalizer_sha256", "seed", "updates")}
    trajectory.update(schema=subject.UQ_EMA_PAIR_SCHEMA, parameters=3680422,
                      point_checkpoint_sha256=raw["point_receipt"]["checkpoint_sha256"],
                      source_commit="c" * 40,
                      source_sha256=subject.sha256(subject.ROOT / "code/training/train_b210_transfer_uq.py"),
                      raw_uq_state_sha256="d" * 64, ema_uq_state_sha256="e" * 64,
                      non_uq_and_all_buffers_bitwise=True)
    trajectory["trajectory_sha256"] = subject.canonical_sha256(trajectory)
    raw["paired_training"] = trajectory
    averaged = copy.deepcopy(raw)
    averaged.update(label="ema", checkpoint_sha256="b" * 64, variant="ema999", ema={
        "schema": "fp32_named_parameter_ema_seal_v1", "decay": .999, "updates": 3200,
        "shadow_dtype": "torch.float32", "update_order": "after_optimizer_step_before_scheduler_step"})
    return {"raw": raw, "ema": averaged}


def test_uq_ema_rejects_fake_pair_source_initialization_and_variant():
    pair = uq_ema_pair()
    subject.validate_seals(pair, list(pair), "uq-ema")
    for key, value in (("initialization", "fresh_same_seed18"), ("initial_uq_sha256", "other"),
                       ("variant", "raw"), ("ema", None), ("sample_stream_sha256", "other"),
                       ("paired_training", {**pair["ema"]["paired_training"], "started_epoch": 9.})):
        broken = copy.deepcopy(pair)
        broken["ema"][key] = value
        with pytest.raises(ValueError):
            subject.validate_seals(broken, list(broken), "uq-ema")
    for key in ("source_sha256", "point_checkpoint_sha256", "trajectory_sha256"):
        broken = copy.deepcopy(pair)
        for item in broken.values():
            item["paired_training"][key] = "f" * 64
        with pytest.raises(ValueError, match="binding"):
            subject.validate_seals(broken, list(broken), "uq-ema")
    with pytest.raises(ValueError):
        subject.validate_seals(pair, list(pair), "matched-point")
    streams = {label: {"prediction_sha256": "same"} for label in pair}
    subject.validate_prediction_streams(streams, "uq-ema")
    streams["ema"]["prediction_sha256"] = "changed"
    with pytest.raises(ValueError, match="point predictions"):
        subject.validate_prediction_streams(streams, "uq-ema")


def test_uq_ema_actual_tensors_reject_changed_point_buffer_or_seal():
    import sys
    import torch
    for path in (subject.ROOT / "code", subject.ROOT / "code/training"):
        sys.path.insert(0, str(path))
    from train_b210_transfer_uq import make_uq_ema, materialize_uq_ema
    from train_c208_dr1_clean81_paired_loao import model_state_sha256
    from train_ensemble_uq_oof import UQ_PREFIXES
    model = torch.nn.Module()
    model.point = torch.nn.Linear(2, 2).requires_grad_(False)
    model.quantile_head = torch.nn.Linear(2, 1)
    model.quantile_head.register_buffer("counter", torch.tensor(0))
    view, shadow = make_uq_ema(model, UQ_PREFIXES)
    optimizer = torch.optim.SGD(view.parameters(), lr=.1)
    for step in range(3):
        optimizer.zero_grad()
        view["quantile_head"](torch.tensor([[1., float(step)]])).square().sum().backward()
        optimizer.step()
        shadow.update_after_optimizer_step(view, optimizer_step=step + 1)
    raw_state = {key: value.clone() for key, value in model.state_dict().items()}
    averaged_state, seal = materialize_uq_ema(raw_state, view, shadow)
    pair, payloads = uq_ema_pair(), {}
    # Tiny states exercise the tensor guard; exact old-C identities are tested above.
    for label, state in (("raw", raw_state), ("ema", averaged_state)):
        item = pair[label]
        item["point_frozen_sha256"] = model_state_sha256({key: value for key, value in state.items()
                                                         if not key.startswith(UQ_PREFIXES)})
        item["model_config_sha256"] = item["normalizer_sha256"] = subject.canonical_sha256({})
        item["paired_training"]["parameters"] = sum(parameter.numel() for parameter in view.parameters())
        item["ema"] = seal if label == "ema" else None
        for other in pair.values():
            other["paired_training"][label + "_uq_state_sha256"] = model_state_sha256(
                {key: value for key, value in state.items() if key.startswith(UQ_PREFIXES)})
        payloads[label] = {key: item[key] for key in ("initialization", "point_source_label", "point_receipt",
                           "point_receipt_sha256", "initial_uq_sha256", "objective_kind", "variant", "paired_training", "ema")}
        payloads[label].update(schema="b210_transfer_uq_v1_endpoint", phase="real", iteration=3200,
                               model_config={}, normalizer={}, model_state_dict=state)
    audit = subject.validate_uq_ema_payloads(payloads, pair)
    assert audit["non_uq_and_all_buffers_bitwise"] and audit["actual_ema_seal_verified"]
    for key in ("point.weight", "quantile_head.counter", "quantile_head.weight"):
        broken = copy.deepcopy(payloads)
        broken["ema"]["model_state_dict"][key] += 1
        with pytest.raises(ValueError):
            subject.validate_uq_ema_payloads(broken, pair)
    broken = copy.deepcopy(pair)
    broken["ema"]["ema"]["shadow_parameter_state_sha256"] = "wrong"
    broken_payloads = copy.deepcopy(payloads)
    broken_payloads["ema"]["ema"] = broken["ema"]["ema"]
    with pytest.raises(ValueError, match="seal hashes"):
        subject.validate_uq_ema_payloads(broken_payloads, broken)


def test_uq_ema_fixed_comparison_rejects_before_any_asset_read(tmp_path):
    args = subject.parse_args([
        "--execute", "--study", "uq-ema", "--expected-labels", "raw", "ema", "--compare", "ema:v3",
        "--baseline-zip", str(tmp_path / "v3.zip"), "--clean-manifest", str(tmp_path / "manifest.json"),
        "--real-dir", str(tmp_path), "--scratch-dir", str(tmp_path),
        "--output", str(tmp_path / "result.json"), "--wall-budget-seconds", "600"])
    with pytest.raises(ValueError, match="primary comparison"):
        subject.execute(args)


def test_real_tke_requires_fixed_raw_control_and_single_point_intervention():
    raw = uq_ema_pair()["raw"]
    raw["checkpoint_sha256"] = "8a34645d5794a68cf0b7f9123b99604682edf006e8558ff97095b0745fea8512"
    candidate = copy.deepcopy(raw)
    candidate.update(label="tke", initial_uq_sha256="new-point-uq", checkpoint_sha256="new-uq")
    candidate["point_receipt"].update(checkpoint_sha256="new-point",
        controller_tke_schedule="constant_0p01_4000", source_commit="a" * 40)
    paired = {"raw": raw, "tke": candidate}
    subject.validate_seals(paired, ["raw", "tke"], "real-tke-constant")
    for label, key, value in (("raw", "variant", "ema999"),
                              ("tke", "initialization", "fresh_same_seed18"),
                              ("tke", "sample_stream_sha256", "different-stream")):
        broken = copy.deepcopy(paired)
        broken[label][key] = value
        with pytest.raises(ValueError):
            subject.validate_seals(broken, ["raw", "tke"], "real-tke-constant")
    del candidate["point_receipt"]["controller_tke_schedule"]
    with pytest.raises(ValueError):
        subject.validate_seals(paired, ["raw", "tke"], "real-tke-constant")


def test_real_coverage_keeps_uq_stream_normalizer_and_held_visibility_fixed():
    pair = {label: copy.deepcopy(receipts()['A']) for label in ('inherit', 'full')}
    for label, item in pair.items():
        item.update(label=label, initialization='point_checkpoint', point_source_label='C', objective_kind='sharp2')
        item['point_receipt'].update(held_aoa=0, start_sim_sha256=subject.SIM_INHERITED_POINTS['inherit']['start_sim_sha256'],
                                    auxiliary_enabled=False, updates=4000, ema_decay=.999, seed=4,
                                    tail_read=label == 'full', sample_stream_sha256=label)
    pair['full']['point_receipt'].update(optimizer_support='full_other_four_trajectories',
        normalizer_support='other_four_prefixes', held_data_read=False, source_commit='a'*40)
    subject.validate_seals(pair, list(pair), 'real-time-coverage')
    for key, value in (('normalizer_sha256', 'changed'), ('sample_stream_sha256', 'changed'), ('tail_arrays_read', 1)):
        broken = copy.deepcopy(pair); broken['full'][key] = value
        with pytest.raises(ValueError): subject.validate_seals(broken, list(pair), 'real-time-coverage')
    for key, value in (('tail_read', False), ('held_data_read', True), ('normalizer_support', 'full')):
        broken = copy.deepcopy(pair); broken['full']['point_receipt'][key] = value
        with pytest.raises(ValueError): subject.validate_seals(broken, list(pair), 'real-time-coverage')
