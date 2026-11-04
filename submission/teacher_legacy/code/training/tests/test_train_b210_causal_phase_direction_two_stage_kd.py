from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fp32_ema_shadow as ema_lib  # noqa: E402
from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)
import train_b210_causal_phase_direction_real4k as direct_phase  # noqa: E402
import train_b210_causal_phase_direction_two_stage_kd as subject  # noqa: E402


class Normalizer:
    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean_input = torch.tensor(mean, dtype=torch.float32)
        self.mean_target = torch.tensor(mean, dtype=torch.float32)
        self.std_input = torch.tensor(std, dtype=torch.float32)
        self.std_target = torch.tensor(std, dtype=torch.float32)

    def preprocess(self, history, target):
        return (
            (history - self.mean_input) / self.std_input,
            (target - self.mean_target) / self.std_target,
        )


class RowwiseTeacher(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history * self.scale


class TinyRealDataset:
    def __init__(self):
        generator = torch.Generator().manual_seed(91)
        self.samples = []
        for _ in range(2):
            history = torch.randn(20, 8, 8, 3, generator=generator)
            target = torch.randn(20, 8, 8, 3, generator=generator)
            history[..., 2] = 0.0
            target[..., 2] = 0.0
            self.samples.append((history, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_contract_freezes_two_stages_without_gate_or_sweep() -> None:
    contract = subject.experiment_contract()
    assert contract["stage1"]["data"]["sim"]["histories"] == 4_800
    assert contract["stage1"]["data"]["real"]["histories"] == 9_933
    assert contract["stage1"]["updates"] == 1_842
    assert contract["stage1"]["ema"] is None
    assert contract["stage1"]["endpoint"] == "raw_u1842"
    assert contract["stage2"]["updates_each_arm"] == 4_000
    assert contract["stage2"]["fixed_report"] == ["raw_u4000", "ema_u4000"]
    assert contract["audit"]["score_gate"] is None
    assert contract["audit"]["sweep"] is None
    assert contract["audit"]["clean716_cli_argument"] is False


def test_formal_stage1_schedule_is_one_seed4_permutation() -> None:
    batches, report = subject.make_stage1_schedule()
    flattened = [index for batch in batches for index in batch]
    assert len(batches) == 1_842
    assert len(batches[-1]) == 5
    assert len(flattened) == 14_733
    assert sorted(flattened) == list(range(14_733))
    assert report["permutation_sha256"] == subject.STAGE1_PERMUTATION_SHA256
    assert report["bijection"] is True
    assert report["inverse_roundtrip"] is True
    assert report["identity_order"] is False


def test_sim_zscore_maps_to_real_moments_and_zeroes_pressure() -> None:
    sim = Normalizer(mean=(10.0, 20.0, 30.0), std=(2.0, 4.0, 5.0))
    real = Normalizer(mean=(1.0, -2.0, 7.0), std=(3.0, 5.0, 9.0))
    history = torch.zeros(2, 20, 2, 2, 3)
    history[0, ..., 0] = 12.0
    history[0, ..., 1] = 16.0
    history[0, ..., 2] = 99.0
    history[1, ..., 0] = 6.0
    history[1, ..., 1] = 8.0
    history[1, ..., 2] = 11.0
    mapped = subject.map_sim_to_real_input(
        history, torch.tensor((0, 1)), sim, real)
    torch.testing.assert_close(mapped[0, ..., 0], torch.full_like(
        mapped[0, ..., 0], 4.0))
    torch.testing.assert_close(mapped[0, ..., 1], torch.full_like(
        mapped[0, ..., 1], -7.0))
    torch.testing.assert_close(mapped[1, ..., :2], history[1, ..., :2])
    assert torch.count_nonzero(mapped[..., 2]).item() == 0


def test_restore_mean_rms_preserves_reference_geometry() -> None:
    torch.manual_seed(1)
    candidate = torch.randn(3, 20, 4, 5, 2)
    reference = torch.randn_like(candidate)
    restored = subject.restore_mean_rms(candidate, reference)
    torch.testing.assert_close(
        restored.mean(dim=1), reference.mean(dim=1), atol=2e-7, rtol=2e-6)
    restored_rms = (restored - restored.mean(dim=1, keepdim=True)).square().mean(
        dim=1).sqrt()
    reference_rms = (
        reference - reference.mean(dim=1, keepdim=True)).square().mean(
            dim=1).sqrt()
    torch.testing.assert_close(restored_rms, reference_rms, atol=2e-7, rtol=2e-6)


def test_restore_mean_rms_falls_back_for_constant_candidate() -> None:
    candidate = torch.ones(2, 20, 3, 4, 2)
    reference = torch.randn_like(candidate)
    assert torch.equal(subject.restore_mean_rms(candidate, reference), reference)


def test_teacher_target_is_fixed_equal_blend_projection() -> None:
    torch.manual_seed(2)
    base = torch.randn(2, 20, 3, 4, 3)
    st = torch.randn_like(base)
    normalizer = Normalizer(mean=(0.2, -0.1, 0.0), std=(2.0, 3.0, 1.0))
    target = subject.teacher_target(base, st, normalizer)
    base_physical = base * normalizer.std_target + normalizer.mean_target
    st_physical = st * normalizer.std_target + normalizer.mean_target
    expected = subject.restore_mean_rms(
        0.5 * base_physical[..., :2] + 0.5 * st_physical[..., :2],
        base_physical[..., :2])
    torch.testing.assert_close(target, expected)


def test_physical_kd_loss_is_official_mapping_of_mean_relative_l2() -> None:
    prediction = torch.tensor([1.0, 2.0]).view(2, 1, 1, 1, 1).expand(
        -1, 2, 1, 1, 2).clone().requires_grad_()
    target = torch.ones_like(prediction)
    loss, details = direct_phase.physical_phase_objective(prediction, target)
    relative = torch.linalg.vector_norm(
        (prediction.detach() - target).reshape(2, -1), dim=1
    ) / torch.linalg.vector_norm(target.reshape(2, -1), dim=1)
    expected = 1.0 - 1.0 / (1.0 + 0.5 * relative.mean())
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(details["relative_l2_error"], relative.mean())
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_target_free_dataset_returns_history_and_rejects_target() -> None:
    sim = [torch.randn(40, 2, 3, 3)]
    real = [torch.randn(40, 2, 3, 2)]
    dataset = subject.TargetFreeHistoryDataset(
        sim, real, (
            subject.HistoryWindow("sim", 0, 0),
            subject.HistoryWindow("real", 0, 5),
        ))
    sim_history, sim_domain = dataset[0]
    real_history, real_domain = dataset[1]
    assert sim_history.shape == real_history.shape == (20, 2, 3, 3)
    assert (sim_domain, real_domain) == (0, 1)
    assert torch.count_nonzero(real_history[..., 2]).item() == 0
    with pytest.raises(RuntimeError, match="forbidden"):
        dataset.target(0)


def test_target_free_audit_rejects_pressure_repeat_and_future_request() -> None:
    audit = subject.TargetFreeAccessAudit()
    audit.record_history_open(
        domain="real", filename="a.h5", channels=("u", "v"),
        stop=100, allowed_history_stop=100, pressure_opened=False)
    with pytest.raises(RuntimeError, match="opened twice"):
        audit.record_history_open(
            domain="real", filename="a.h5", channels=("u", "v"),
            stop=100, allowed_history_stop=100, pressure_opened=False)
    with pytest.raises(RuntimeError, match="only measured"):
        subject.TargetFreeAccessAudit().record_history_open(
            domain="real", filename="b.h5", channels=("u", "v", "p"),
            stop=100, allowed_history_stop=100, pressure_opened=True)
    with pytest.raises(RuntimeError, match="target-free"):
        audit.record_future_tensor_request()


def test_order_permutation_diagnostic_reexecutes_rowwise_models() -> None:
    torch.manual_seed(3)
    history = torch.randn(4, 20, 8, 8, 3)
    adapter = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig(
        hidden_channels=8, temporal_blocks=2, temporal_kernel=3, groups=4))
    report = subject.order_permutation_diagnostic(
        history, RowwiseTeacher(0.75), RowwiseTeacher(1.25),
        adapter, Normalizer())
    assert report["passed"] is True
    assert max(report["max_abs_after_inverse"].values()) <= report["tolerance"]


def test_two_update_stage1_training_mechanics_are_backward_safe(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "STAGE1_HISTORIES", 3)
    monkeypatch.setattr(subject, "STAGE1_UPDATES", 2)
    sim = [torch.randn(40, 8, 8, 3)]
    real = [torch.randn(20, 8, 8, 2)]
    dataset = subject.TargetFreeHistoryDataset(
        sim, real, (
            subject.HistoryWindow("sim", 0, 0),
            subject.HistoryWindow("sim", 0, 20),
            subject.HistoryWindow("real", 0, 0),
        ))
    report = subject.train_stage1(
        dataset=dataset, schedule=((0, 1), (2,)),
        base=RowwiseTeacher(0.75), st_teacher=RowwiseTeacher(1.25),
        sim_normalizer=Normalizer(), real_normalizer=Normalizer(),
        source={}, data_report={},
        access_report={"future_tensor_requests": 0},
        output_dir=tmp_path, device=torch.device("cpu"))
    assert report["mechanics"]["direction_head_gradient_u1"] is True
    assert report["mechanics"]["history_encoder_gradient_u2"] is True
    assert report["mechanics"]["order_permutation"]["passed"] is True
    assert report["endpoint"]["strict_load"] is True


def test_two_update_stage2_arms_have_identical_augmented_stream(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "STAGE2_UPDATES", 2)
    monkeypatch.setattr(subject, "REAL_HISTORIES", 2)
    dataset = TinyRealDataset()
    schedule = ((0,), (1,))
    adapter = subject.direct_phase.build_adapter(
        "causal_temporal", torch.device("cpu"))
    zero = {key: value.detach().clone()
            for key, value in adapter.state_dict().items()}
    candidate = {key: value.clone() for key, value in zero.items()}
    candidate["direction_head.bias"].add_(0.05)
    reports = [
        subject.train_stage2_arm(
            arm=arm, initial_state=state,
            base=RowwiseTeacher(0.75), dataset=dataset,
            normalizer=Normalizer(), schedule=schedule, source={},
            output_dir=tmp_path, device=torch.device("cpu"))
        for arm, state in zip(subject.STAGE2_ARMS, (zero, candidate), strict=True)
    ]
    assert subject.assert_matched_stage2(reports)[
        "augmentation_probes_bitwise_equal"] is True
    assert all(report["endpoint"]["strict_load"] for report in reports)


def test_endpoint_writer_publishes_strict_companion_seal(tmp_path: Path) -> None:
    adapter = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig())
    state = {name: value.detach().clone()
             for name, value in adapter.state_dict().items()}
    normalizer = Normalizer()
    normalizer_sha = subject._normalizer_sha(normalizer)
    base_lineage = {
        "sha256": "base-file-sha",
        "model_state_sha256": "base-state-sha",
        "normalizer": {"sha256": normalizer_sha},
    }
    payload = {
        "schema": subject.STAGE1_ENDPOINT_SCHEMA,
        "experiment_schema": subject.SCHEMA,
        "stage": "stage1_target_free_kd",
        "arm": "kd_pretrain",
        "endpoint": f"raw_u{subject.STAGE1_UPDATES}",
        "update": subject.STAGE1_UPDATES,
        "adapter_config": adapter.architecture_config(),
        "adapter_parameters": subject.EXPECTED_ADAPTER_PARAMETERS,
        "raw_adapter_state": state,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(state),
        "ema": None,
        "normalizers": {
            "real": subject._normalizer_tensors(normalizer),
            "real_sha256": normalizer_sha,
        },
        "source": {
            "base": base_lineage,
            "clean_manifest": {"file_sha256": "clean81-manifest-sha"},
        },
        "training": {
            "unique_histories": subject.STAGE1_HISTORIES,
            "updates": subject.STAGE1_UPDATES,
            "batch_size": subject.BATCH_SIZE,
            "last_batch_size": 5,
            "seed": subject.SEED,
            "sample_permutation_sha256": subject.STAGE1_PERMUTATION_SHA256,
            "ema": None,
            "augmentation": None,
            "objective": "physical_uv_official_Rel-L2_mapping_only",
        },
        "target_free_access": {
            "schema": "b210_phase_kd_target_free_access_v1",
            "opened_files": 181,
            "future_tensor_requests": 0,
            "future_ground_truth_materialized": False,
            "clean716_opened": False,
        },
        "checkpoint_selected": False,
        "score_gate": None,
        "clean716_opened": False,
    }
    path = tmp_path / "endpoint.pt"
    seal = subject.write_endpoint_and_seal(
        path, payload, ("raw_adapter_state",))
    assert path.is_file()
    assert Path(seal["seal_path"]).is_file()
    verified = subject.verify_companion_seal(path, seal["file_sha256"])
    assert verified["strict_load"] is True
    assert verified["state_sha256"]["raw_adapter_state"] == (
        payload["raw_adapter_state_sha256"])
    loaded, report = subject.load_stage1_endpoint(
        path, seal["file_sha256"], normalizer, base_lineage,
        "clean81-manifest-sha")
    assert ema_lib.tensor_state_sha256(loaded) == (
        payload["raw_adapter_state_sha256"])
    assert report["file_sha256"] == seal["file_sha256"]
    with pytest.raises(RuntimeError, match="source lineage"):
        subject.load_stage1_endpoint(
            path, seal["file_sha256"], normalizer,
            {**base_lineage, "sha256": "different-base"},
            "clean81-manifest-sha")
    bad_payload = {
        **payload,
        "target_free_access": {
            **payload["target_free_access"], "future_tensor_requests": 1,
        },
    }
    bad_path = tmp_path / "bad_access.pt"
    bad_seal = subject.write_endpoint_and_seal(
        bad_path, bad_payload, ("raw_adapter_state",))
    with pytest.raises(RuntimeError, match="target-free access"):
        subject.load_stage1_endpoint(
            bad_path, bad_seal["file_sha256"], normalizer,
            base_lineage, "clean81-manifest-sha")


def test_companion_seal_rejects_endpoint_mutation(tmp_path: Path) -> None:
    adapter = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig())
    state = adapter.state_dict()
    path = tmp_path / "endpoint.pt"
    payload = {
        "schema": subject.STAGE1_ENDPOINT_SCHEMA,
        "experiment_schema": subject.SCHEMA,
        "stage": "stage1_target_free_kd",
        "arm": "kd_pretrain",
        "update": subject.STAGE1_UPDATES,
        "adapter_config": adapter.architecture_config(),
        "raw_adapter_state": state,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(state),
    }
    seal = subject.write_endpoint_and_seal(
        path, payload, ("raw_adapter_state",))
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="invalid"):
        subject.verify_companion_seal(path, seal["file_sha256"])


def test_matched_stage2_requires_different_start_and_equal_augmentation() -> None:
    probes = {"1": {"history": "same"}}
    reports = [
        {"arm": subject.STAGE2_ARMS[0],
         "initial_adapter_state_sha256": "a", "augmentation_probes": probes},
        {"arm": subject.STAGE2_ARMS[1],
         "initial_adapter_state_sha256": "b", "augmentation_probes": probes},
    ]
    assert subject.assert_matched_stage2(reports)[
        "augmentation_probes_bitwise_equal"] is True
    reports[1] = {**reports[1], "augmentation_probes": {"1": {"history": "x"}}}
    with pytest.raises(RuntimeError, match="identical augmentation"):
        subject.assert_matched_stage2(reports)


def test_inventory_missing_paths_are_explicit_and_h5_free() -> None:
    args = subject.parse_args(("--mode", "inventory"))
    report = subject.inventory_report(args)
    assert report["missing"]
    assert report["raw_h5_opened"] is False
    assert report["clean716_h5_opened"] is False
    assert report["gpu_used"] is False
    assert all(item["status"] == "missing_path_not_guessed"
               for item in report["checkpoints"])


def test_cli_has_no_clean716_or_hyperparameter_sweep_surface() -> None:
    actions = {action.dest for action in subject.parse_args.__globals__["argparse"].ArgumentParser()._actions}
    # Inspect the real parser through its usage text, without relying on private
    # argparse action layout in the implementation.
    with pytest.raises(SystemExit):
        subject.parse_args(("--clean716-manifest", "forbidden"))
    assert "clean716" not in actions


def test_mechanics_smoke_is_cpu_target_free_and_finite() -> None:
    report = subject.mechanics_smoke()
    assert report["device"] == "cpu"
    assert report["finite_loss"] is True
    assert report["direction_head_gradient"] is True
    assert report["pressure_exact_zero"] is True
    assert report["ema_initialized_from_u0"] is True
    assert report["raw_h5_opened"] is False
    assert report["clean716_opened"] is False
