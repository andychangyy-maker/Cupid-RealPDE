from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_b210_np_functional_sim.py"
SPEC = importlib.util.spec_from_file_location("b210_np_sim", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def formal_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        teacher_cache=Path("teacher.pt"),
        teacher_cache_sha256=subject.TEACHER_CACHE_SHA256,
        output_dir=tmp_path / "run",
        sim_dir=Path("sim"),
        real_stats=Path("stats.pt"),
        device="cuda:0",
        num_workers=0,
    )


def anchor(*, point=90.0, rel=90.0, tke=80.0, mvpe=90.0,
           hgrad=2.0, p1=3.0, sps=0.5) -> dict[str, float]:
    return {
        "point_score": point,
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "hgrad_loss": hgrad,
        "p1_distill_loss": p1,
        "native_soft_sps_proxy": sps,
    }


def hgrad_audits() -> list[dict]:
    return [
        {
            "iteration": update,
            "loss_ratio": 0.02,
            "gradient_ratio": 0.05,
            "cosine": 0.10,
            "point_gradient_norm": 1.0,
            "hgrad_gradient_norm": 0.05,
            "finite_nonzero": True,
        }
        for update in subject.sim_hgrad.AUDIT_UPDATES
    ]


def test_contract_freezes_np_functional_bundle_and_sim_only_inputs(tmp_path):
    args = formal_args(tmp_path)
    subject.validate_args(args)
    training = subject.training_args(args)
    assert subject.METHOD == "b210_np_functional_sim_v3"
    assert subject.CONTROL_UPDATES == 200
    assert subject.TERMINAL_UPDATE == 10_000
    assert subject.WALL_BUDGET_SECONDS == 6 * 60 * 60
    assert subject.BACKBONE_DROPOUT == pytest.approx(0.20)
    assert subject.HGRAD_WEIGHT == pytest.approx(0.020)
    assert subject.TKE_AUX_WEIGHT == pytest.approx(0.020)
    assert subject.TKE_AUX_CALIBRATION_SOURCE_MEDIAN == pytest.approx(
        0.04955256027147056)
    assert subject.TKE_AUX_CALIBRATION_TARGET == pytest.approx(0.10)
    assert subject.TKE_GRADIENT_RATIO_RANGE == (0.05, 0.50)
    assert training.soft_sps_loss_weight == pytest.approx(0.05)
    assert training.soft_sps_warmup_updates == 1500
    assert training.soft_sps_uq_only is False
    assert training.distill_weight == pytest.approx(0.005)
    assert training.accuracy_weights == (1.0, 4.0, 1.0)
    assert training.start_checkpoint is None

    source = inspect.getsource(subject.parse_args)
    assert "--real-dir" not in source
    assert "--start-checkpoint" not in source
    assert "--held" not in source

    args.teacher_cache_sha256 = "a" * 64
    with pytest.raises(ValueError, match="teacher cache SHA"):
        subject.validate_args(args)


def test_output_protection_refuses_prior_terminal_or_stop(tmp_path):
    args = formal_args(tmp_path)
    args.output_dir.mkdir()
    (args.output_dir / "stop.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject.validate_args(args)


def test_research_cache_override_is_explicit_and_never_formal(tmp_path):
    args = formal_args(tmp_path)
    research_sha = "b" * 64
    args.teacher_cache_sha256 = research_sha
    args.research_equivalent_cache_sha256 = research_sha
    args.acknowledge_research_cache = subject.RESEARCH_CACHE_ACKNOWLEDGEMENT

    subject.validate_args(args)

    assert subject.expected_teacher_cache_sha256(args) == research_sha
    assert research_sha != subject.TEACHER_CACHE_SHA256

    args.acknowledge_research_cache = ""
    with pytest.raises(ValueError, match="requires acknowledgement"):
        subject.validate_args(args)


def test_b210_control_and_candidate_have_same_initial_state_and_exact_counts():
    device = torch.device("cpu")
    control = subject.build_arm_model("control", device)
    control_hash = subject.state_sha256(control.state_dict())
    control_config = control.architecture_config()
    control_trainable = subject.configure_trainable(control, "control")
    assert sum(value.numel() for value in control.parameters()) == (
        subject.EXPECTED_PARAMETERS)
    assert sum(value.numel() for value in control_trainable) == (
        subject.EXPECTED_PARAMETERS - subject.EXPECTED_CONTROLLER_PARAMETERS)
    assert control_config["base"] == 210
    assert control_config["groups"] == 8
    assert control_config["uq_hidden_channels"] == 160
    assert control_config["mean_profile_channels"] == 104
    assert control_config["physical_energy_controller"] is True
    assert control_config["physical_energy_member_sync"] is True
    assert control_config["backbone_residual_dropout"] == 0.0
    assert all(
        not parameter.requires_grad
        for parameter in control.energy_controller.parameters())
    del control, control_trainable
    gc.collect()

    candidate = subject.build_arm_model("candidate", device)
    candidate_config = candidate.architecture_config()
    candidate_trainable = subject.configure_trainable(candidate, "candidate")
    assert subject.state_sha256(candidate.state_dict()) == control_hash
    assert candidate_config["backbone_residual_dropout"] == pytest.approx(0.2)
    assert sum(value.numel() for value in candidate_trainable) == (
        subject.EXPECTED_PARAMETERS)
    assert all(
        parameter.requires_grad
        for parameter in candidate.energy_controller.parameters())


def test_controller_only_auxiliary_does_not_reach_legacy_parameter():
    legacy = torch.nn.Parameter(torch.tensor(1.0))
    controller = torch.nn.Parameter(torch.tensor(2.0))
    non_auxiliary = (legacy + controller).square()
    auxiliary = (3.0 * legacy + 5.0 * controller).square()
    diagnostic = subject.backward_with_controller_only_auxiliary(
        non_auxiliary, auxiliary, (controller,), 0.1)
    assert legacy.grad == pytest.approx(torch.tensor(6.0))
    assert controller.grad == pytest.approx(torch.tensor(19.0))
    assert diagnostic["native_controller_gradient_norm"] == pytest.approx(6.0)
    assert diagnostic["tke_aux_gradient_norm"] == pytest.approx(13.0)
    assert diagnostic["tke_aux_to_native_gradient_ratio"] == pytest.approx(
        13.0 / 6.0)


def test_output_pareto_projection_removes_only_destructive_tke_component():
    prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    primary = prediction[0]
    tke = -prediction[0] + prediction[1]
    correction, diagnostic = subject.output_tke_projection(
        primary, tke, prediction)
    assert diagnostic["pareto_projection_applied"] is True
    assert diagnostic["pareto_primary_tke_cosine"] == pytest.approx(
        -1.0 / (2.0 ** 0.5))
    assert correction is not None
    corrected = torch.tensor([1.0, 0.0]) + correction
    protected = torch.tensor([-1.0, 1.0])
    assert torch.dot(corrected, protected) == pytest.approx(0.0, abs=1e-7)
    assert corrected == pytest.approx(torch.tensor([0.5, 0.5]))

    aligned_prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    correction, aligned = subject.output_tke_projection(
        aligned_prediction.sum(), aligned_prediction[0], aligned_prediction)
    assert correction is None
    assert aligned["pareto_projection_applied"] is False


def test_pareto_tke_loss_matches_single_component_official_objective():
    class Normalizer:
        mean_target = torch.tensor([0.1, -0.2, 0.3])
        std_target = torch.tensor([0.4, 0.2, 0.5])

    prediction = torch.randn(3, 20, 8, 16, 3, requires_grad=True)
    target = torch.randn(3, 20, 8, 16, 3)
    expected, _details = (
        subject.rehearsal.ensemble_train.common.accuracy_objective(
            prediction, target, Normalizer(), (0.0, 1.0, 0.0)))
    observed = subject.official_tke_score_loss(
        prediction, target, Normalizer())
    assert float(observed.detach()) == pytest.approx(
        float(expected.detach()), abs=1e-7)
    observed.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_pareto_projection_summary_requires_real_conflicts():
    row = {
        "pareto_projection_applied": True,
        "pareto_primary_tke_cosine": -0.25,
        "pareto_primary_gradient_norm": 2.0,
        "pareto_tke_gradient_norm": 1.0,
        "pareto_correction_to_primary_ratio": 0.25,
        "pareto_projection_finite": True,
    }
    summary = subject.pareto_projection_summary(
        [row] * subject.CONTROL_UPDATES)
    assert summary["pass"] is True
    assert summary["conflict_fraction"] == pytest.approx(1.0)
    no_conflict = dict(row, pareto_projection_applied=False)
    summary = subject.pareto_projection_summary(
        [no_conflict] * subject.CONTROL_UPDATES)
    assert summary["pass"] is False


def test_legacy_tke_gradient_floor_restores_only_missing_parallel_strength():
    prediction = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    primary = (
        prediction[0, 0] + prediction[0, 1]
        + 3.0 * prediction[1, 0] + prediction[1, 1])
    legacy = 2.0 * prediction[:, 0].sum()
    tke = prediction[:, 0].sum()
    correction, diagnostic = subject.output_legacy_tke_gradient_floor(
        primary, legacy, tke, prediction)
    assert correction is not None
    assert correction == pytest.approx(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert diagnostic["legacy_tke_floor_applied_fraction"] == pytest.approx(0.5)
    assert diagnostic[
        "legacy_tke_floor_correction_ratio_median"] == pytest.approx(0.0)
    assert diagnostic["legacy_tke_floor_finite"] is True


def test_legacy_tke_floor_summary_has_fixed_mechanical_gate():
    row = {
        "legacy_tke_floor_finite": True,
        "legacy_tke_floor_applied_fraction": 0.75,
        "legacy_tke_floor_primary_parallel_median": 1.0,
        "legacy_tke_floor_reference_parallel_median": 2.0,
        "legacy_tke_floor_missing_parallel_median": 1.0,
        "legacy_tke_floor_correction_ratio_median": 0.5,
        "legacy_tke_floor_correction_ratio_max": 0.75,
    }
    summary = subject.legacy_tke_floor_summary(
        [row] * subject.CONTROL_UPDATES)
    assert summary["pass"] is True
    assert summary["mean_applied_fraction"] == pytest.approx(0.75)
    inactive = dict(row, legacy_tke_floor_applied_fraction=0.25)
    assert subject.legacy_tke_floor_summary(
        [inactive] * subject.CONTROL_UPDATES)["pass"] is False


def test_tke_auxiliary_schedule_is_frozen_and_early_only():
    assert subject.tke_aux_weight(1) == pytest.approx(0.020)
    assert 0.0 < subject.tke_aux_weight(750) < 0.020
    assert subject.tke_aux_weight(1500) == pytest.approx(0.0, abs=1e-15)
    assert subject.tke_aux_weight(1501) == 0.0
    with pytest.raises(ValueError, match="iteration must be positive"):
        subject.tke_aux_weight(0)


def test_official_point_controller_tke_floor_is_calibrated_and_persistent():
    floor = subject.TKE_AUX_PERSISTENT_FLOOR_WEIGHT
    assert floor == pytest.approx(0.004)
    assert subject.tke_aux_weight(1, persistent_floor=True) == pytest.approx(
        subject.TKE_AUX_WEIGHT)
    assert floor < subject.tke_aux_weight(
        750, persistent_floor=True) < subject.TKE_AUX_WEIGHT
    assert subject.tke_aux_weight(
        1500, persistent_floor=True) == pytest.approx(floor)
    assert subject.tke_aux_weight(
        10_000, persistent_floor=True) == pytest.approx(floor)


def test_multiscale_tke_supervisor_is_train_only_and_mean_invariant():
    features = [
        torch.randn(2, 12, 4, 8, requires_grad=True),
        torch.randn(2, 8, 8, 16, requires_grad=True),
        torch.randn(2, 4, 16, 32, requires_grad=True),
    ]
    target = torch.randn(2, 20, 16, 32, 2) * 0.1
    supervisor = subject.MultiScaleTKEDeepSupervisor((12, 8, 4))
    loss, levels = supervisor(features, target)
    shifted = target + torch.randn(2, 1, 16, 32, 2)
    assert torch.allclose(
        supervisor.physical_tke_map(target),
        supervisor.physical_tke_map(shifted), atol=2e-6, rtol=0.0)
    assert len(levels) == 3
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None and torch.count_nonzero(value.grad)
               for value in features)
    assert all(parameter.grad is not None
               for parameter in supervisor.parameters())


def test_multiscale_tke_contract_requires_closure_and_keeps_deploy_model_clean(
        tmp_path):
    args = formal_args(tmp_path)
    args.multiscale_tke_deep_supervision = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert training.multiscale_tke_deep_supervision is True
    assert subject.MULTISCALE_TKE_WEIGHT == pytest.approx(0.005)
    assert subject.MULTISCALE_TKE_PHYSICAL_SCALE == pytest.approx(0.010)
    assert subject.MULTISCALE_TKE_LEVEL_WEIGHTS == (0.25, 0.50, 1.00)
    assert subject.MULTISCALE_TKE_GRADIENT_AUDIT_UPDATE == 2
    model = subject.build_arm_model("candidate", torch.device("cpu"))
    before = set(model.state_dict())
    supervisor = subject.build_multiscale_tke_supervisor(
        model, torch.device("cpu"))
    assert sum(parameter.numel() for parameter in supervisor.parameters()) == 48195
    assert set(model.state_dict()) == before


def test_multiscale_energy_controller_is_a_distinct_deployed_arm(tmp_path):
    args = formal_args(tmp_path)
    args.multiscale_energy_controller = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert training.multiscale_energy_controller is True
    model = subject.build_arm_model(
        "candidate", torch.device("cpu"),
        multiscale_energy_controller=True)
    assert model.config.physical_energy_multiscale is True
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        subject.EXPECTED_MULTISCALE_ENERGY_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in model.energy_controller.parameters()) == (
            subject.EXPECTED_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS)
    assert torch.count_nonzero(model.energy_controller.body[-1].weight) == 0


def test_history_energy_controller_is_small_distinct_deployed_arm(tmp_path):
    args = formal_args(tmp_path)
    args.history_energy_controller = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert training.history_energy_controller is True
    model = subject.build_arm_model(
        "candidate", torch.device("cpu"),
        history_energy_controller=True)
    assert model.config.physical_energy_history_conditioned is True
    assert model.config.physical_energy_history_steps == 5
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        subject.EXPECTED_HISTORY_ENERGY_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in model.energy_controller.parameters()) == (
            subject.EXPECTED_HISTORY_ENERGY_CONTROLLER_PARAMETERS)
    assert torch.count_nonzero(model.energy_controller.body[-1].weight) == 0


def test_direct_history_tke_implies_history_arm_and_fixed_weight(tmp_path):
    args = formal_args(tmp_path)
    args.direct_history_tke_regression = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    assert args.history_energy_controller is True
    training = subject.training_args(args)
    assert training.direct_history_tke_regression is True
    assert training.history_energy_controller is True
    assert subject.DIRECT_HISTORY_TKE_WEIGHT == pytest.approx(0.010)


def test_hybrid_history_multiscale_energy_is_one_distinct_arm(tmp_path):
    args = formal_args(tmp_path)
    args.hybrid_history_multiscale_energy_controller = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    assert args.history_energy_controller is True
    assert args.multiscale_energy_controller is True
    training = subject.training_args(args)
    assert training.hybrid_history_multiscale_energy_controller is True
    model = subject.build_arm_model(
        "candidate", torch.device("cpu"),
        hybrid_history_multiscale_energy_controller=True)
    assert model.config.physical_energy_history_conditioned is True
    assert model.config.physical_energy_multiscale is True
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        subject.EXPECTED_HYBRID_HISTORY_MULTISCALE_ENERGY_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in model.energy_controller.parameters()) == (
            subject.EXPECTED_HYBRID_HISTORY_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS)
    assert torch.count_nonzero(model.energy_controller.body[-1].weight) == 0
    assert torch.count_nonzero(
        model.energy_controller.history.body[-1].weight) == 0

    conflict = formal_args(tmp_path)
    conflict.hybrid_history_multiscale_energy_controller = True
    conflict.direct_history_tke_regression = True
    conflict.real_closure_bank = Path("closure.pt")
    conflict.real_closure_bank_sha256 = "c" * 64
    with pytest.raises(ValueError, match="separate arms"):
        subject.validate_args(conflict)


def test_cross_conditioned_backbone_is_distinct_and_control_frozen(tmp_path):
    args = formal_args(tmp_path)
    args.cross_conditioned_backbone = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert training.cross_conditioned_backbone is True

    control = subject.build_arm_model(
        "control", torch.device("cpu"), cross_conditioned_backbone=True)
    assert control.config.cross_conditioned_backbone is True
    assert control.config.cross_conditioned_steps == 5
    assert control.config.cross_conditioned_channels == 64
    assert sum(parameter.numel() for parameter in control.parameters()) == (
        subject.EXPECTED_CROSS_CONDITIONED_BACKBONE_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in control.cross_conditioner.parameters()) == (
            subject.EXPECTED_CROSS_CONDITIONER_PARAMETERS)
    assert all(
        torch.count_nonzero(projection.weight) == 0
        for projection in control.cross_conditioner.projections)
    trainable = subject.configure_trainable(control, "control")
    assert all(
        not parameter.requires_grad
        for parameter in control.cross_conditioner.parameters())
    assert sum(parameter.numel() for parameter in trainable) == (
        subject.EXPECTED_CROSS_CONDITIONED_BACKBONE_PARAMETERS
        - subject.EXPECTED_CONTROLLER_PARAMETERS
        - subject.EXPECTED_CROSS_CONDITIONER_PARAMETERS)
    del control, trainable
    gc.collect()

    conflict = formal_args(tmp_path)
    conflict.cross_conditioned_backbone = True
    conflict.multiscale_energy_controller = True
    conflict.real_closure_bank = Path("closure.pt")
    conflict.real_closure_bank_sha256 = "c" * 64
    with pytest.raises(ValueError, match="separate architecture arm"):
        subject.validate_args(conflict)


def test_temporal_point_decoder_replaces_flat_heads_and_is_trainable(tmp_path):
    args = formal_args(tmp_path)
    args.temporal_point_decoder = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert training.temporal_point_decoder is True
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_TEMPORAL_POINT_DECODER_PARAMETERS)

    model = subject.build_arm_model(
        "candidate", torch.device("cpu"), temporal_point_decoder=True)
    assert model.config.temporal_point_decoder is True
    assert model.config.temporal_point_channels == 24
    assert model.config.temporal_point_blocks == 2
    assert model.config.temporal_point_kernel == 5
    assert len(model.member_point_heads) == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        subject.EXPECTED_TEMPORAL_POINT_DECODER_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in model.temporal_point_decoder.parameters()) == (
            subject.EXPECTED_TEMPORAL_POINT_PARAMETERS)
    assert all(
        parameter.requires_grad
        for parameter in model.temporal_point_decoder.parameters())
    assert all(
        torch.count_nonzero(head.weight) == 0
        for heads in model.temporal_point_decoder.member_fluctuation_heads
        for head in heads)
    del model
    gc.collect()

    conflict = formal_args(tmp_path)
    conflict.temporal_point_decoder = True
    conflict.cross_conditioned_backbone = True
    conflict.real_closure_bank = Path("closure.pt")
    conflict.real_closure_bank_sha256 = "c" * 64
    with pytest.raises(ValueError, match="separate architecture arm"):
        subject.validate_args(conflict)


def test_temporal_amplitude_factorization_has_fixed_bounded_contract(tmp_path):
    args = formal_args(tmp_path)
    args.temporal_point_amplitude_factorization = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    assert args.temporal_point_decoder is True
    training = subject.training_args(args)
    assert training.temporal_point_decoder is True
    assert training.temporal_point_amplitude_factorization is True
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_TEMPORAL_AMPLITUDE_FACTORIZATION_PARAMETERS)

    model = subject.build_arm_model(
        "candidate", torch.device("cpu"),
        temporal_point_decoder=True,
        temporal_point_amplitude_factorization=True)
    assert model.config.temporal_point_amplitude_factorization is True
    assert model.config.temporal_point_amplitude_limit == pytest.approx(3.0)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        subject.EXPECTED_TEMPORAL_AMPLITUDE_FACTORIZATION_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in model.temporal_point_decoder.parameters()) == (
            subject.EXPECTED_TEMPORAL_AMPLITUDE_PARAMETERS)
    assert all(
        torch.count_nonzero(head.weight) == 0
        for heads in model.temporal_point_decoder.member_amplitude_heads
        for head in heads)
    assert any(
        torch.count_nonzero(head.weight)
        for heads in model.temporal_point_decoder.member_fluctuation_heads
        for head in heads)
    del model
    gc.collect()


def test_isoenergetic_temporal_transport_has_frozen_control_contract(tmp_path):
    args = formal_args(tmp_path)
    args.isoenergetic_temporal_transport = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert training.isoenergetic_temporal_transport is True
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_TEMPORAL_TRANSPORT_PARAMETERS)

    model = subject.build_arm_model(
        "candidate", torch.device("cpu"),
        isoenergetic_temporal_transport=True)
    assert model.config.isoenergetic_temporal_transport is True
    assert model.config.temporal_transport_max_displacement == pytest.approx(1.5)
    assert len(model.member_point_heads) == 4
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        subject.EXPECTED_TEMPORAL_TRANSPORT_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in model.temporal_transport.parameters()) == (
            subject.EXPECTED_TEMPORAL_TRANSPORT_MODULE_PARAMETERS)
    assert all(
        torch.count_nonzero(parameter) == 0
        for parameter in model.temporal_transport.output.parameters())
    candidate_trainable = subject.configure_trainable(model, "candidate")
    assert all(parameter.requires_grad for parameter in candidate_trainable)
    del model, candidate_trainable
    gc.collect()

    control = subject.build_arm_model(
        "control", torch.device("cpu"),
        isoenergetic_temporal_transport=True)
    control_trainable = subject.configure_trainable(control, "control")
    assert all(
        not parameter.requires_grad
        for parameter in control.temporal_transport.parameters())
    assert sum(parameter.numel() for parameter in control_trainable) == (
        subject.EXPECTED_PARAMETERS
        - subject.EXPECTED_CONTROLLER_PARAMETERS)
    del control, control_trainable
    gc.collect()

    conflict = formal_args(tmp_path)
    conflict.isoenergetic_temporal_transport = True
    conflict.temporal_point_decoder = True
    conflict.real_closure_bank = Path("closure.pt")
    conflict.real_closure_bank_sha256 = "c" * 64
    with pytest.raises(ValueError, match="separate architecture arm"):
        subject.validate_args(conflict)

    objective = formal_args(tmp_path)
    objective.official_point_objective = True
    with pytest.raises(ValueError, match="requires isoenergetic"):
        subject.validate_args(objective)
    objective.isoenergetic_temporal_transport = True
    objective.real_closure_bank = Path("closure.pt")
    objective.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(objective)
    score_aligned = subject.training_args(objective)
    assert score_aligned.official_point_objective is True
    assert score_aligned.accuracy_weights == (46.0, 10.0, 9.0)

    floor = formal_args(tmp_path)
    floor.official_point_controller_tke_floor = True
    with pytest.raises(ValueError, match="requires official point"):
        subject.validate_args(floor)
    floor.official_point_objective = True
    floor.isoenergetic_temporal_transport = True
    floor.real_closure_bank = Path("closure.pt")
    floor.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(floor)
    score_aligned_floor = subject.training_args(floor)
    assert score_aligned_floor.official_point_controller_tke_floor is True
    assert score_aligned_floor.accuracy_weights == (46.0, 10.0, 9.0)


def test_pareto_tke_projection_is_training_only_and_score_aligned(tmp_path):
    args = formal_args(tmp_path)
    args.pareto_tke_projection = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert training.pareto_tke_projection is True
    assert training.accuracy_weights == (46.0, 10.0, 9.0)
    assert subject.contract_parameters(args) == subject.EXPECTED_PARAMETERS

    conflict = formal_args(tmp_path)
    conflict.pareto_tke_projection = True
    conflict.official_point_objective = True
    conflict.real_closure_bank = Path("closure.pt")
    conflict.real_closure_bank_sha256 = "c" * 64
    with pytest.raises(
            ValueError,
            match="requires isoenergetic|separate training-mechanism arm"):
        subject.validate_args(conflict)


def test_legacy_tke_gradient_floor_is_parameter_free_training_arm(tmp_path):
    args = formal_args(tmp_path)
    args.legacy_tke_gradient_floor = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert training.legacy_tke_gradient_floor is True
    assert training.accuracy_weights == (46.0, 10.0, 9.0)
    assert subject.contract_parameters(args) == subject.EXPECTED_PARAMETERS

    conflict = formal_args(tmp_path)
    conflict.legacy_tke_gradient_floor = True
    conflict.pareto_tke_projection = True
    conflict.real_closure_bank = Path("closure.pt")
    conflict.real_closure_bank_sha256 = "c" * 64
    with pytest.raises(ValueError, match="separate training"):
        subject.validate_args(conflict)


def test_member_integrated_tke_floor_aligns_mean_and_member_objectives(
        tmp_path):
    args = formal_args(tmp_path)
    args.member_integrated_legacy_tke_gradient_floor = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert args.legacy_tke_gradient_floor is True
    assert training.legacy_tke_gradient_floor is True
    assert training.member_integrated_legacy_tke_gradient_floor is True
    assert training.member_loss_mode == "official_accuracy"
    assert training.accuracy_weights == (46.0, 10.0, 9.0)
    assert subject.contract_parameters(args) == subject.EXPECTED_PARAMETERS

    source = inspect.getsource(subject.run_experiment)
    assert "if not (member_integrated_legacy_tke_gradient_floor" in source
    assert '"is_hard_gate": not (' in source


def test_member_integrated_floor_supports_full_member_gradient_tensor():
    members = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]],
         [[2.0, 1.0], [4.0, 3.0]]], requires_grad=True)
    primary = members[..., 0].sum() + 0.5 * members[..., 1].sum()
    legacy = 2.0 * members[..., 0].sum()
    tke = members.mean(dim=1)[..., 0].sum()
    correction, diagnostic = subject.output_legacy_tke_gradient_floor(
        primary, legacy, tke, members)
    assert correction is not None
    assert correction.shape == members.shape
    assert diagnostic["legacy_tke_floor_finite"] is True
    assert diagnostic["legacy_tke_floor_applied_fraction"] > 0.0


def test_member_official_objective_keeps_floor_on_mean_only(tmp_path):
    args = formal_args(tmp_path)
    args.member_official_objective = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert args.legacy_tke_gradient_floor is True
    assert training.member_official_objective is True
    assert training.member_integrated_legacy_tke_gradient_floor is False
    assert training.member_loss_mode == "official_accuracy"
    assert training.accuracy_weights == (46.0, 10.0, 9.0)
    assert subject.contract_parameters(args) == subject.EXPECTED_PARAMETERS


def test_decoupled_member_residual_is_distinct_frozen_control_arm(tmp_path):
    args = formal_args(tmp_path)
    args.decoupled_member_residual = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert args.member_official_objective is True
    assert args.legacy_tke_gradient_floor is True
    assert training.decoupled_member_residual is True
    assert training.member_official_objective is True
    assert training.member_integrated_legacy_tke_gradient_floor is False
    assert training.member_loss_mode == "official_accuracy"
    assert training.accuracy_weights == (46.0, 10.0, 9.0)
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_DECOUPLED_MEMBER_RESIDUAL_PARAMETERS)

    control = subject.build_arm_model(
        "control", torch.device("cpu"), decoupled_member_residual=True)
    control_hash = subject.state_sha256(control.state_dict())
    assert control.config.decoupled_member_residual_channels == 64
    assert control.config.decoupled_member_residual_blocks == 2
    assert sum(parameter.numel() for parameter in control.parameters()) == (
        subject.EXPECTED_DECOUPLED_MEMBER_RESIDUAL_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in control.decoupled_member_residual.parameters()) == (
            subject.EXPECTED_DECOUPLED_MEMBER_RESIDUAL_MODULE_PARAMETERS)
    control_trainable = subject.configure_trainable(control, "control")
    assert all(
        not parameter.requires_grad
        for parameter in control.decoupled_member_residual.parameters())
    assert sum(parameter.numel() for parameter in control_trainable) == (
        subject.EXPECTED_DECOUPLED_MEMBER_RESIDUAL_PARAMETERS
        - subject.EXPECTED_CONTROLLER_PARAMETERS
        - subject.EXPECTED_DECOUPLED_MEMBER_RESIDUAL_MODULE_PARAMETERS)
    del control, control_trainable
    gc.collect()

    candidate = subject.build_arm_model(
        "candidate", torch.device("cpu"), decoupled_member_residual=True)
    assert subject.state_sha256(candidate.state_dict()) == control_hash
    candidate_trainable = subject.configure_trainable(candidate, "candidate")
    assert all(
        parameter.requires_grad
        for parameter in candidate.decoupled_member_residual.parameters())
    assert sum(parameter.numel() for parameter in candidate_trainable) == (
        subject.EXPECTED_DECOUPLED_MEMBER_RESIDUAL_PARAMETERS)
    del candidate, candidate_trainable
    gc.collect()

    source = inspect.getsource(subject.main)
    assert "DECOUPLED_MEMBER_RESIDUAL_METHOD" in source


def test_member_feature_adapters_preserve_legacy_member_objective(tmp_path):
    args = formal_args(tmp_path)
    args.member_feature_adapters = True
    with pytest.raises(ValueError, match="requires a closure bank"):
        subject.validate_args(args)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert args.legacy_tke_gradient_floor is True
    assert training.member_feature_adapters is True
    assert training.member_official_objective is False
    assert training.member_integrated_legacy_tke_gradient_floor is False
    assert training.member_loss_mode == "mse"
    assert training.accuracy_weights == (46.0, 10.0, 9.0)
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_MEMBER_FEATURE_ADAPTER_PARAMETERS)

    control = subject.build_arm_model(
        "control", torch.device("cpu"), member_feature_adapters=True)
    control_hash = subject.state_sha256(control.state_dict())
    assert control.config.member_adapter_channels == 64
    assert sum(parameter.numel() for parameter in control.parameters()) == (
        subject.EXPECTED_MEMBER_FEATURE_ADAPTER_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in control.member_feature_adapters.parameters()) == (
            subject.EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS)
    control_trainable = subject.configure_trainable(control, "control")
    assert all(
        not parameter.requires_grad
        for parameter in control.member_feature_adapters.parameters())
    assert sum(parameter.numel() for parameter in control_trainable) == (
        subject.EXPECTED_MEMBER_FEATURE_ADAPTER_PARAMETERS
        - subject.EXPECTED_CONTROLLER_PARAMETERS
        - subject.EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS)
    del control, control_trainable
    gc.collect()

    candidate = subject.build_arm_model(
        "candidate", torch.device("cpu"), member_feature_adapters=True)
    assert subject.state_sha256(candidate.state_dict()) == control_hash
    candidate_trainable = subject.configure_trainable(candidate, "candidate")
    assert all(
        parameter.requires_grad
        for parameter in candidate.member_feature_adapters.parameters())
    assert sum(parameter.numel() for parameter in candidate_trainable) == (
        subject.EXPECTED_MEMBER_FEATURE_ADAPTER_PARAMETERS)
    del candidate, candidate_trainable
    gc.collect()


def test_conditional_future_adapter_only_contract_and_gradient_isolation(
        tmp_path):
    args = formal_args(tmp_path)
    args.conditional_future_energy_adapter_only = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert args.member_feature_adapters is True
    assert args.legacy_tke_gradient_floor is True
    assert training.conditional_future_energy_adapter_only is True
    assert training.conditional_future_energy_members_only is False
    assert training.member_loss_weight == pytest.approx(
        subject.CONDITIONAL_FUTURE_MEMBER_MSE_WEIGHT)

    conflict = formal_args(tmp_path)
    conflict.conditional_future_energy_adapter_only = True
    conflict.conditional_future_energy_members_only = True
    conflict.real_closure_bank = Path("closure.pt")
    conflict.real_closure_bank_sha256 = "c" * 64
    with pytest.raises(ValueError, match="mutually exclusive"):
        subject.validate_args(conflict)

    shared = torch.nn.Parameter(torch.tensor(1.0))
    adapter = torch.nn.Parameter(torch.tensor(2.0))
    base = (shared + adapter).square()
    energy = 0.05 * (shared + 3.0 * adapter).square()
    energy_gradient = torch.autograd.grad(
        energy, (adapter,), retain_graph=True, allow_unused=False)
    base.backward()
    shared_base_gradient = shared.grad.detach().clone()
    adapter_base_gradient = adapter.grad.detach().clone()
    subject.add_parameter_gradients((adapter,), energy_gradient)
    assert shared.grad == pytest.approx(shared_base_gradient)
    assert adapter.grad == pytest.approx(
        adapter_base_gradient + energy_gradient[0])


def test_conditional_future_crps_uses_mean_preserving_adapter_contract(
        tmp_path):
    args = formal_args(tmp_path)
    args.conditional_future_crps_mean_preserving = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert args.member_feature_adapters is True
    assert args.mean_preserving_member_adapters is True
    assert args.legacy_tke_gradient_floor is True
    assert training.conditional_future_crps_mean_preserving is True
    assert training.mean_preserving_member_adapters is True
    assert training.member_loss_weight == pytest.approx(
        subject.CONDITIONAL_FUTURE_MEMBER_MSE_WEIGHT)
    config = subject.b210_config(
        "candidate", member_feature_adapters=True,
        mean_preserving_member_adapters=True)
    assert config.member_adapter_channels == subject.MEMBER_FEATURE_ADAPTER_CHANNELS
    assert config.mean_preserving_member_adapters is True

    detached = formal_args(tmp_path)
    detached.conditional_future_crps_mean_preserving = True
    detached.detach_mean_preserving_adapter_context = True
    detached.real_closure_bank = Path("closure.pt")
    detached.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(detached)
    detached_training = subject.training_args(detached)
    assert detached.detach_member_adapter_context is True
    assert detached_training.detach_member_adapter_context is True
    detached_config = subject.b210_config(
        "candidate", member_feature_adapters=True,
        mean_preserving_member_adapters=True,
        detach_member_adapter_context=True)
    assert detached_config.detach_member_adapter_context is True

    invalid = formal_args(tmp_path)
    invalid.detach_mean_preserving_adapter_context = True
    with pytest.raises(ValueError, match="requires mean-preserving CRPS"):
        subject.validate_args(invalid)

    isolated = formal_args(tmp_path)
    isolated.conditional_future_crps_mean_preserving = True
    isolated.isolate_distribution_tower = True
    isolated.real_closure_bank = Path("closure.pt")
    isolated.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(isolated)
    isolated_training = subject.training_args(isolated)
    assert isolated.detach_member_adapter_context is True
    assert isolated.detach_uq_context is True
    assert isolated_training.isolate_distribution_tower is True
    assert isolated_training.detach_member_adapter_context is True
    assert isolated_training.detach_uq_context is True
    isolated_config = subject.b210_config(
        "candidate", member_feature_adapters=True,
        mean_preserving_member_adapters=True,
        detach_member_adapter_context=True,
        detach_uq_context=True)
    assert isolated_config.detach_uq_context is True

    missing_crps = formal_args(tmp_path)
    missing_crps.isolate_distribution_tower = True
    with pytest.raises(ValueError, match="requires mean-preserving CRPS"):
        subject.validate_args(missing_crps)

    budgeted = formal_args(tmp_path)
    budgeted.conditional_future_crps_mean_preserving = True
    budgeted.isolate_distribution_tower = True
    budgeted.budgeted_disagreement_allocation = True
    budgeted.real_closure_bank = Path("closure.pt")
    budgeted.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(budgeted)
    budgeted_training = subject.training_args(budgeted)
    assert budgeted_training.budgeted_disagreement_allocation is True
    assert subject.contract_parameters(budgeted) == (
        subject.EXPECTED_BUDGETED_DISAGREEMENT_PARAMETERS)
    budgeted_config = subject.b210_config(
        "candidate", member_feature_adapters=True,
        mean_preserving_member_adapters=True,
        detach_member_adapter_context=True, detach_uq_context=True,
        budgeted_disagreement_allocation=True)
    assert budgeted_config.budgeted_disagreement_allocation is True

    missing_isolation = formal_args(tmp_path)
    missing_isolation.conditional_future_crps_mean_preserving = True
    missing_isolation.budgeted_disagreement_allocation = True
    with pytest.raises(ValueError, match="requires isolated"):
        subject.validate_args(missing_isolation)


def test_cross_conditioned_member_adapters_have_joint_frozen_contract(
        tmp_path):
    args = formal_args(tmp_path)
    args.cross_conditioned_backbone = True
    args.member_feature_adapters = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)
    assert args.legacy_tke_gradient_floor is True
    assert training.cross_conditioned_backbone is True
    assert training.member_feature_adapters is True
    assert training.member_official_objective is False
    assert training.member_loss_mode == "mse"
    assert training.accuracy_weights == (46.0, 10.0, 9.0)
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS)

    control = subject.build_arm_model(
        "control", torch.device("cpu"),
        cross_conditioned_backbone=True, member_feature_adapters=True)
    initial_hash = subject.state_sha256(control.state_dict())
    assert sum(parameter.numel() for parameter in control.parameters()) == (
        subject.EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in control.cross_conditioner.parameters()) == (
            subject.EXPECTED_CROSS_CONDITIONER_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in control.member_feature_adapters.parameters()) == (
            subject.EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS)
    control_trainable = subject.configure_trainable(control, "control")
    assert all(
        not parameter.requires_grad
        for parameter in control.cross_conditioner.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in control.member_feature_adapters.parameters())
    assert sum(parameter.numel() for parameter in control_trainable) == (
        subject.EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS
        - subject.EXPECTED_CONTROLLER_PARAMETERS
        - subject.EXPECTED_CROSS_CONDITIONER_PARAMETERS
        - subject.EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS)
    del control, control_trainable
    gc.collect()

    candidate = subject.build_arm_model(
        "candidate", torch.device("cpu"),
        cross_conditioned_backbone=True, member_feature_adapters=True)
    assert subject.state_sha256(candidate.state_dict()) == initial_hash
    candidate_trainable = subject.configure_trainable(candidate, "candidate")
    assert all(parameter.requires_grad for parameter in candidate_trainable)
    assert sum(parameter.numel() for parameter in candidate_trainable) == (
        subject.EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS)
    del candidate, candidate_trainable
    gc.collect()


def test_cross_conditioned_member_adapters_can_retain_standard_objective(
        tmp_path):
    args = formal_args(tmp_path)
    args.cross_conditioned_backbone = True
    args.standard_objective_member_feature_adapters = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)

    assert args.member_feature_adapters is True
    assert getattr(args, "legacy_tke_gradient_floor", False) is False
    assert training.standard_objective_member_feature_adapters is True
    assert training.member_feature_adapters is True
    assert training.accuracy_weights == (1.0, 4.0, 1.0)
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS)

    duplicate = formal_args(tmp_path)
    duplicate.cross_conditioned_backbone = True
    duplicate.member_feature_adapters = True
    duplicate.standard_objective_member_feature_adapters = True
    with pytest.raises(ValueError, match="only one member-feature"):
        subject.validate_args(duplicate)

    missing_cross = formal_args(tmp_path)
    missing_cross.standard_objective_member_feature_adapters = True
    with pytest.raises(ValueError, match="require cross conditioning"):
        subject.validate_args(missing_cross)


def test_standard_xcb_member_can_add_multiscale_energy_controller(tmp_path):
    args = formal_args(tmp_path)
    args.cross_conditioned_backbone = True
    args.standard_objective_member_feature_adapters = True
    args.multiscale_energy_controller = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)

    assert args.member_feature_adapters is True
    assert training.accuracy_weights == (1.0, 4.0, 1.0)
    assert training.cross_conditioned_backbone is True
    assert training.multiscale_energy_controller is True
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_XCB_MEMBER_MULTISCALE_ENERGY_PARAMETERS)

    model = subject.build_arm_model(
        "candidate", torch.device("cpu"),
        multiscale_energy_controller=True,
        cross_conditioned_backbone=True,
        member_feature_adapters=True)
    assert model.config.physical_energy_multiscale is True
    assert model.config.cross_conditioned_backbone is True
    assert model.config.member_adapter_channels == 64
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        subject.EXPECTED_XCB_MEMBER_MULTISCALE_ENERGY_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in model.energy_controller.parameters()) == (
            subject.EXPECTED_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS)


def test_standard_xcb_member_can_separate_fluctuation_role(tmp_path):
    args = formal_args(tmp_path)
    args.cross_conditioned_backbone = True
    args.standard_objective_member_feature_adapters = True
    args.fluctuation_role_tower = True
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    training = subject.training_args(args)

    assert args.member_feature_adapters is True
    assert training.accuracy_weights == (1.0, 4.0, 1.0)
    assert training.fluctuation_role_tower is True
    assert subject.contract_parameters(args) == (
        subject.EXPECTED_XCB_MEMBER_FLUCTUATION_ROLE_PARAMETERS)

    control = subject.build_arm_model(
        "control", torch.device("cpu"),
        cross_conditioned_backbone=True,
        member_feature_adapters=True,
        fluctuation_role_tower=True)
    initial_hash = subject.state_sha256(control.state_dict())
    assert control.config.fluctuation_role_channels == 64
    assert sum(parameter.numel() for parameter in control.parameters()) == (
        subject.EXPECTED_XCB_MEMBER_FLUCTUATION_ROLE_PARAMETERS)
    assert sum(
        parameter.numel()
        for parameter in control.fluctuation_role_adapters.parameters()) == (
            subject.EXPECTED_FLUCTUATION_ROLE_MODULE_PARAMETERS)
    control_trainable = subject.configure_trainable(control, "control")
    assert all(
        not parameter.requires_grad
        for parameter in control.fluctuation_role_adapters.parameters())
    assert sum(parameter.numel() for parameter in control_trainable) == (
        subject.EXPECTED_XCB_MEMBER_FLUCTUATION_ROLE_PARAMETERS
        - subject.EXPECTED_CONTROLLER_PARAMETERS
        - subject.EXPECTED_CROSS_CONDITIONER_PARAMETERS
        - subject.EXPECTED_MEMBER_FEATURE_ADAPTER_MODULE_PARAMETERS
        - subject.EXPECTED_FLUCTUATION_ROLE_MODULE_PARAMETERS)
    del control, control_trainable
    gc.collect()

    candidate = subject.build_arm_model(
        "candidate", torch.device("cpu"),
        cross_conditioned_backbone=True,
        member_feature_adapters=True,
        fluctuation_role_tower=True)
    assert subject.state_sha256(candidate.state_dict()) == initial_hash
    candidate_trainable = subject.configure_trainable(candidate, "candidate")
    assert all(parameter.requires_grad for parameter in candidate_trainable)
    assert sum(parameter.numel() for parameter in candidate_trainable) == (
        subject.EXPECTED_XCB_MEMBER_FLUCTUATION_ROLE_PARAMETERS)

    invalid = formal_args(tmp_path)
    invalid.fluctuation_role_tower = True
    with pytest.raises(ValueError, match="requires standard XCB/member"):
        subject.validate_args(invalid)


def test_direct_history_tke_loss_trains_head_not_point_backbone():
    config = subject.EnsembleUNet2dConfig(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8,
        physical_energy_controller=True,
        physical_energy_member_sync=True,
        physical_energy_history_conditioned=True,
        physical_energy_history_steps=5,
        physical_energy_hidden_channels=8)
    model = subject.EnsembleUNet2d(config)

    class Normalizer:
        mean_target = torch.zeros(3)
        std_target = torch.tensor([0.2, 0.1, 0.3])

    history = torch.randn(2, 20, 8, 16, 3)
    features = [value.detach() for value in model._decode(history)]
    base = torch.randn(2, 20, 8, 16, 3)
    target = torch.randn(2, 20, 8, 16, 3) * 0.1
    loss, metrics = subject.direct_history_tke_regression_loss(
        model, base, target, history, features, Normalizer())
    assert torch.isfinite(loss)
    assert 0.0 <= metrics[
        "direct_history_tke_target_saturation_fraction"] <= 1.0
    loss.backward()
    controller_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.energy_controller.parameters()
        if parameter.grad is not None)
    backbone_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if not name.startswith("energy_controller.")
        and parameter.grad is not None)
    assert controller_grad > 0.0
    assert backbone_grad == 0.0


def test_multiscale_tke_feature_gradient_audit_reports_weighted_ratio():
    coarse = torch.randn(2, 3, requires_grad=True)
    fine = torch.randn(2, 4, requires_grad=True)
    native = coarse.square().mean() + fine.square().mean()
    auxiliary = 0.01 * (coarse.sum() + fine.sum())
    result = subject.multiscale_tke_feature_gradient_audit(
        native, auxiliary, (coarse, fine))
    assert result["finite_nonzero"] is True
    assert result["native_feature_gradient_norm"] > 0.0
    assert result["multiscale_tke_feature_gradient_norm"] > 0.0
    assert result[
        "multiscale_tke_to_native_feature_gradient_ratio"] > 0.0


def test_p95_is_diagnostic_and_directional_failure_does_not_stop_health():
    ratios = [0.20] * 189 + [10.0] * 11
    ratio = subject.ratio_diagnostic(ratios)
    assert ratio["pass"] is True
    assert ratio["median"] == pytest.approx(0.20)
    assert ratio["p95_diagnostic_only"] == pytest.approx(10.0)
    assert ratio["p95_is_hard_gate"] is False

    one_zero = subject.ratio_diagnostic([0.20] * 199 + [0.0])
    assert one_zero["pass"] is True
    assert one_zero["positive_samples"] == 199
    assert one_zero["zero_samples_diagnostic_only"] == 1
    assert one_zero["individual_zero_is_hard_gate"] is False

    initial = anchor()
    control = anchor(hgrad=2.0)
    candidate = anchor(
        point=89.99, rel=89.99, tke=79.99, mvpe=89.99,
        hgrad=1.96, p1=3.0, sps=0.5)
    gate = subject.prefix_gate(
        initial, control, candidate, ratios,
        {"pass": True},
        {"state_updated": True, "projection_updated": True,
         "front_updated": True, "projection_gradient_seen": True,
         "front_gradient_seen": True},
        hgrad_audits())
    assert gate["health_pass"] is True
    assert gate["directional_pass"] is False
    assert gate["directional_is_hard_gate"] is False
    assert gate["status"] == subject.HEALTH_PASS_STATUS


def test_health_gate_stops_only_broken_or_catastrophic_prefix():
    ratios = [0.20] * subject.CONTROL_UPDATES
    control = anchor()
    catastrophic = anchor(point=88.99, hgrad=1.9)
    gate = subject.prefix_gate(
        anchor(), control, catastrophic, ratios,
        {"pass": True},
        {"state_updated": True, "projection_updated": True,
         "front_updated": True, "projection_gradient_seen": True,
         "front_gradient_seen": True},
        hgrad_audits())
    assert gate["health_pass"] is False
    assert gate["health_checks"]["point_not_catastrophic"] is False
    assert gate["status"] == subject.HEALTH_STOP_STATUS

    broken_ratio = subject.prefix_gate(
        anchor(), control, anchor(hgrad=1.9), [0.9] * 200,
        {"pass": True},
        {"state_updated": True, "projection_updated": True,
         "front_updated": True, "projection_gradient_seen": True,
         "front_gradient_seen": True},
        hgrad_audits())
    assert broken_ratio["health_pass"] is False
    assert broken_ratio["health_checks"][
        "tke_controller_gradient_ratio"] is False


def test_update0_identity_checks_every_inference_tensor():
    first = {
        name: torch.tensor([index])
        for index, name in enumerate(
            ("point", "members", "quantiles", "lower", "upper"))
    }
    assert subject.update0_identity(first, first)["pass"] is True
    second = dict(first)
    second["members"] = second["members"] + 1
    result = subject.update0_identity(first, second)
    assert result["pass"] is False
    assert result["checks"]["members"] is False


def test_real_closure_teacher_transform_preserves_teacher_residual():
    target = torch.randn(2, 20, 4, 5, 2)
    teacher = target + 0.25
    delta = torch.randn_like(target) * 0.05
    transformed = subject.transform_teacher_for_real_closure(teacher, delta)
    assert torch.allclose(
        transformed - (target + delta), teacher - target, atol=2e-7, rtol=0.0)
    assert subject.transform_teacher_for_real_closure(teacher, None) is teacher
    with pytest.raises(ValueError, match="not teacher-aligned"):
        subject.transform_teacher_for_real_closure(
            teacher, delta[..., :3, :, :])


def test_real_closure_contract_is_fixed_and_requires_artifact_pair(tmp_path):
    args = formal_args(tmp_path)
    args.real_closure_bank = Path("closure.pt")
    args.real_closure_bank_sha256 = None
    with pytest.raises(ValueError, match="both bank and bank SHA"):
        subject.validate_args(args)
    args.real_closure_bank_sha256 = "c" * 64
    subject.validate_args(args)
    assert subject.REAL_CLOSURE_PROBABILITY == pytest.approx(0.5)
    assert subject.REAL_CLOSURE_AMPLITUDE == pytest.approx(0.05)
    assert subject.CLEAN_REAL_MANIFEST_SHA256 == (
        "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")

    args = formal_args(tmp_path)
    args.real_closure_equivariance = True
    with pytest.raises(ValueError, match="equivariance requires"):
        subject.validate_args(args)


def test_closure_equivariance_target_translates_uv_only():
    class Normalizer:
        std_target = torch.tensor([2.0, 4.0, 8.0])

    clean = torch.randn(2, 3, 4, 5, 3)
    physical_delta = torch.randn(2, 3, 4, 5, 2)
    expected = subject.closure_equivariance_target(
        clean, physical_delta, Normalizer())
    assert torch.allclose(expected[..., 0] - clean[..., 0],
                          physical_delta[..., 0] / 2.0, atol=2e-7)
    assert torch.allclose(expected[..., 1] - clean[..., 1],
                          physical_delta[..., 1] / 4.0, atol=2e-7)
    assert torch.equal(expected[..., 2], clean[..., 2])
    assert expected.requires_grad is False
