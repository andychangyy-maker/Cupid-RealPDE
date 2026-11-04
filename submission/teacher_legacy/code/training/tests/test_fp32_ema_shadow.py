from __future__ import annotations

from collections import OrderedDict
import copy
import io
import sys
from pathlib import Path
import unittest
import warnings

import torch
from torch import nn


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fp32_ema_shadow import (  # noqa: E402
    EMA_SEAL_SCHEMA,
    UPDATE_ORDER_CONTRACT,
    FP32EMAShadow,
    canonical_sha256,
    tensor_state_sha256,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


class BufferedScalar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("running_value", torch.tensor([2.0]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.weight * value + self.running_value


class ComplexScalar(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.complex64) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0 + 2.0j], dtype=dtype))
        self.register_buffer("counter", torch.tensor([3], dtype=torch.int64))


class FP32EMAShadowTests(unittest.TestCase):
    def test_complex64_components_update_seal_and_strict_roundtrip(self):
        model = ComplexScalar()
        ema = FP32EMAShadow(model, decay=0.5)
        self.assertEqual(
            ema.shadow_parameter_state()["weight"].dtype, torch.complex64)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([3.0 + 6.0j]))
            model.counter.fill_(11)
        ema.update_after_optimizer_step(model, optimizer_step=1)
        torch.testing.assert_close(
            ema.shadow_parameter_state()["weight"],
            torch.tensor([2.0 + 4.0j], dtype=torch.complex64),
            rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            model.weight, torch.tensor([3.0 + 6.0j]), rtol=0.0, atol=0.0)

        seal = ema.seal(model)
        self.assertEqual(seal["shadow_dtype"], "torch.float32")
        self.assertEqual(seal["complex_shadow_dtype"], "torch.complex64")
        self.assertEqual(seal["model_state"]["weight"].dtype, torch.complex64)
        self.assertEqual(seal["model_state"]["counter"].dtype, torch.int64)
        self.assertEqual(int(seal["model_state"]["counter"]), 11)
        self.assertEqual(
            seal["materialized_model_state_sha256"],
            tensor_state_sha256(seal["model_state"]))

        buffer = io.BytesIO()
        torch.save(seal, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False)
        fresh = ComplexScalar()
        incompatible = fresh.load_state_dict(loaded["model_state"], strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(
            fresh.weight, torch.tensor([2.0 + 4.0j]), rtol=0.0, atol=0.0)
        restored = FP32EMAShadow.from_seal(model, loaded)
        torch.testing.assert_close(
            restored.shadow_parameter_state()["weight"],
            torch.tensor([2.0 + 4.0j]), rtol=0.0, atol=0.0)

    def test_complex_dtypes_other_than_complex64_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "complex64"):
            FP32EMAShadow(ComplexScalar(torch.complex128), decay=0.5)
        if hasattr(torch, "complex32"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with self.assertRaisesRegex(TypeError, "complex64"):
                    FP32EMAShadow(ComplexScalar(torch.complex32), decay=0.5)

    def test_real_only_seal_keeps_legacy_metadata_shape(self):
        model = BufferedScalar()
        ema = FP32EMAShadow(model, decay=0.5)
        seal = ema.seal(model)
        self.assertNotIn("complex_shadow_dtype", seal)
        self.assertEqual(seal["shadow_dtype"], "torch.float32")

    def test_post_optimizer_update_order_and_optimizer_isolation(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model_parameter_ids = tuple(id(value) for value in model.parameters())
        ema = FP32EMAShadow(model, decay=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        self.assertEqual(
            tuple(id(value) for value in model.parameters()),
            model_parameter_ids)
        self.assertEqual(
            tuple(id(value) for group in optimizer.param_groups
                  for value in group["params"]),
            model_parameter_ids)
        self.assertNotIsInstance(ema, nn.Module)

        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        raw_after_optimizer = model.weight.detach().clone()
        torch.testing.assert_close(
            ema.shadow_parameter_state()["weight"], torch.tensor([[1.0]]))

        ema.update_after_optimizer_step(model, optimizer_step=1)

        torch.testing.assert_close(raw_after_optimizer, torch.tensor([[0.5]]))
        torch.testing.assert_close(model.weight, raw_after_optimizer)
        torch.testing.assert_close(
            ema.shadow_parameter_state()["weight"], torch.tensor([[0.75]]))
        with self.assertRaisesRegex(ValueError, "expected optimizer_step=2"):
            ema.update_after_optimizer_step(model, optimizer_step=1)
        with self.assertRaisesRegex(ValueError, "expected optimizer_step=2"):
            ema.update_after_optimizer_step(model, optimizer_step=3)

    def test_fp32_manifest_and_exact_buffer_copy(self):
        model = BufferedScalar().double()
        ema = FP32EMAShadow(model, decay=0.5)
        self.assertEqual(
            [entry["name"] for entry in ema.parameter_manifest()],
            [name for name, _parameter in model.named_parameters()])
        self.assertEqual(
            ema.parameter_manifest_sha256,
            canonical_sha256(ema.parameter_manifest()))
        self.assertEqual(
            ema.shadow_parameter_state()["weight"].dtype, torch.float32)

        with torch.no_grad():
            model.weight.fill_(3.0)
            model.running_value.fill_(9.125)
        raw_before_update = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
        }
        ema.update_after_optimizer_step(model, optimizer_step=1)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, raw_before_update[name])

        materialized = ema.materialize_model_state(model)
        torch.testing.assert_close(
            materialized["weight"], torch.tensor([2.0], dtype=torch.float32))
        torch.testing.assert_close(
            materialized["running_value"],
            torch.tensor([9.125], dtype=torch.float64))
        self.assertEqual(
            materialized["running_value"].data_ptr()
            == model.running_value.data_ptr(), False)

    def test_4000_update_seal_hash_load_and_strict_materialize(self):
        model = BufferedScalar()
        ema = FP32EMAShadow(model, decay=0.999)
        for optimizer_step in range(1, 4001):
            with torch.no_grad():
                model.weight.fill_(optimizer_step / 4000.0)
            ema.update_after_optimizer_step(
                model, optimizer_step=optimizer_step)
        with torch.no_grad():
            model.running_value.fill_(17.25)
        raw_before_load = copy.deepcopy(model.state_dict())

        seal = ema.seal(model)

        self.assertEqual(seal["schema"], EMA_SEAL_SCHEMA)
        self.assertEqual(seal["updates"], 4000)
        self.assertEqual(seal["decay"], 0.999)
        self.assertEqual(seal["update_order"], UPDATE_ORDER_CONTRACT)
        self.assertEqual(seal["shadow_dtype"], "torch.float32")
        self.assertEqual(
            seal["parameter_manifest_sha256"],
            canonical_sha256(seal["parameter_manifest"]))
        self.assertEqual(
            seal["materialized_model_state_sha256"],
            tensor_state_sha256(seal["model_state"]))
        metadata = {
            key: value for key, value in seal.items()
            if key not in {"seal_metadata_sha256", "model_state"}
        }
        self.assertEqual(
            seal["seal_metadata_sha256"], canonical_sha256(metadata))
        self.assertEqual(
            seal["raw_model_state_sha256"],
            tensor_state_sha256(raw_before_load))
        parameter_state = OrderedDict(
            (entry["name"], seal["model_state"][entry["name"]])
            for entry in seal["parameter_manifest"])
        buffer_state = OrderedDict(
            (name, seal["model_state"][name])
            for name in seal["buffer_names"])
        self.assertEqual(
            seal["shadow_parameter_state_sha256"],
            tensor_state_sha256(parameter_state))
        self.assertEqual(
            seal["buffer_state_sha256"], tensor_state_sha256(buffer_state))

        fresh = BufferedScalar()
        incompatible = fresh.load_state_dict(seal["model_state"], strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(
            fresh.running_value, torch.tensor([17.25]))

        restored = FP32EMAShadow.from_seal(model, seal)
        self.assertEqual(restored.updates, 4000)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, raw_before_load[name])
        for name, value in restored.shadow_parameter_state().items():
            torch.testing.assert_close(value, seal["model_state"][name])

        tampered = dict(seal)
        tampered["materialized_model_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "metadata hash mismatch"):
            FP32EMAShadow.from_seal(model, tampered)

        tampered = dict(seal)
        tampered["raw_model_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "metadata hash mismatch"):
            FP32EMAShadow.from_seal(model, tampered)

        tampered = dict(seal)
        tampered_state = OrderedDict(
            (name, value.clone())
            for name, value in seal["model_state"].items())
        tampered_state["weight"].add_(1.0)
        tampered["model_state"] = tampered_state
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            FP32EMAShadow.from_seal(model, tampered)

    def test_shrunken_c208_raw_training_behavior_is_unchanged(self):
        torch.manual_seed(208)
        config = EnsembleUNet2dConfig(
            in_steps=20,
            out_steps=20,
            base=4,
            depth=2,
            groups=2,
            difference_steps=1,
            ensemble_members=2,
            uq_hidden_channels=4,
        )
        reference = EnsembleUNet2d(config)
        reference.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        candidate = copy.deepcopy(reference)
        ema = FP32EMAShadow(candidate, decay=0.999)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-3)
        candidate_optimizer = torch.optim.AdamW(
            candidate.parameters(), lr=1e-3)
        candidate_ids = {id(value) for value in candidate.parameters()}
        optimizer_ids = {
            id(value) for group in candidate_optimizer.param_groups
            for value in group["params"]
        }
        self.assertEqual(optimizer_ids, candidate_ids)
        self.assertEqual(
            list(candidate.state_dict()), list(reference.state_dict()))

        history = torch.randn(1, config.in_steps, 4, 4, 3)
        target = torch.randn(1, config.out_steps, 4, 4, 3)

        def train_step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            prediction, lower, upper = model.forward_distribution(history)
            loss = (
                (prediction - target).square().mean()
                + 0.01 * (lower.mean() + upper.mean()))
            loss.backward()
            optimizer.step()

        train_step(reference, reference_optimizer)
        train_step(candidate, candidate_optimizer)
        candidate_raw_after_optimizer = copy.deepcopy(candidate.state_dict())
        ema.update_after_optimizer_step(candidate, optimizer_step=1)

        for name, value in reference.state_dict().items():
            torch.testing.assert_close(
                value, candidate.state_dict()[name], rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                candidate.state_dict()[name],
                candidate_raw_after_optimizer[name], rtol=0.0, atol=0.0)
        reference.eval()
        candidate.eval()
        with torch.no_grad():
            reference_output = reference.forward_distribution(history)
            candidate_output = candidate.forward_distribution(history)
        for reference_value, candidate_value in zip(
                reference_output, candidate_output):
            torch.testing.assert_close(
                reference_value, candidate_value, rtol=0.0, atol=0.0)
        self.assertFalse(any(
            name.startswith("ema") or ".ema" in name
            for name in candidate.state_dict()))


if __name__ == "__main__":
    unittest.main()
