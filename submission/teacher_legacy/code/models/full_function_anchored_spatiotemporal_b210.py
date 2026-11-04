"""Full-function weight anchor with a zero-start causal-ST Real residual.

The source member uses a complete sealed Sim-trained ``SpatiotemporalB210``
weight state.  The source affine coordinate system is explicit: it can remain
the sealed Sim normalizer, or the same weights can be re-expressed with the
clean81 Real normalizer before the matched pair is constructed.  A second,
equally sized causal ST model consumes the same physical history through the
Real normalizer and supplies an additive correction.  A per-channel
zero-initialized gain makes the correction exactly zero at update zero without
bypassing its forward graph.

The matched control and candidate have identical state, parameters, and forward
calls.  The control updates only the complete source function; the candidate
updates only the additive Real residual while preserving the source function.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import torch
from torch import nn

from .spatiotemporal_b210 import (
    SpatiotemporalB210,
    SpatiotemporalB210Config,
)


FULL_FUNCTION_ADAPTATION_MODES = (
    "trainable_source_control",
    "frozen_source_candidate",
)

SOURCE_NORMALIZER_POLICIES = (
    "sealed_sim",
    "clean81_real_affine",
)

_NORMALIZER_FIELDS = (
    "mean_input", "mean_target", "std_input", "std_target")


def _validated_normalizer(
        values: Mapping[str, torch.Tensor], prefix: str,
        ) -> dict[str, torch.Tensor]:
    if set(values) != set(_NORMALIZER_FIELDS):
        raise ValueError(f"{prefix} normalizer inventory changed")
    result: dict[str, torch.Tensor] = {}
    for name in _NORMALIZER_FIELDS:
        value = values[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (3,):
            raise TypeError(f"{prefix}.{name} must be a three-channel tensor")
        value = value.detach().float().cpu().clone()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{prefix}.{name} is non-finite")
        if name.startswith("std_") and not bool((value > 0).all()):
            raise ValueError(f"{prefix}.{name} must be strictly positive")
        result[name] = value
    return result


class FullFunctionAnchoredSpatiotemporalB210(nn.Module):
    """One complete Sim weight source plus one zero-start Real residual."""

    def __init__(
            self,
            config: SpatiotemporalB210Config,
            adaptation_mode: str,
            sim_normalizer: Mapping[str, torch.Tensor],
            real_normalizer: Mapping[str, torch.Tensor],
            source_normalizer_policy: str = "sealed_sim",
            ) -> None:
        super().__init__()
        if adaptation_mode not in FULL_FUNCTION_ADAPTATION_MODES:
            raise ValueError(
                f"unsupported full-function adaptation mode {adaptation_mode!r}")
        if source_normalizer_policy not in SOURCE_NORMALIZER_POLICIES:
            raise ValueError(
                "unsupported source normalizer policy "
                f"{source_normalizer_policy!r}")
        config.validate()
        self.config = config
        self.adaptation_mode = str(adaptation_mode)
        self.source_normalizer_policy = str(source_normalizer_policy)
        sim = _validated_normalizer(sim_normalizer, "sim")
        real = _validated_normalizer(real_normalizer, "real")
        if (self.source_normalizer_policy == "clean81_real_affine"
                and any(not torch.equal(sim[name], real[name])
                        for name in _NORMALIZER_FIELDS)):
            raise ValueError(
                "clean81_real_affine requires the source and residual to use "
                "the same Real normalizer")

        self.source_model = SpatiotemporalB210(
            config, "causal_temporal_candidate")
        self.real_residual_model = SpatiotemporalB210(
            config, "causal_temporal_candidate")
        self.residual_gain = nn.Parameter(torch.zeros(config.out_channels))
        for domain, values in (("sim", sim), ("real", real)):
            for name in _NORMALIZER_FIELDS:
                self.register_buffer(
                    f"_{domain}_{name}", values[name].clone(), persistent=True)

        # These persistence buffers belong to their respective functions.  The
        # aligned policy supplies an explicitly audited common Real affine at
        # construction; callers cannot mutate the wrapper through a generic
        # normalizer installation path later.
        self.source_model.set_normalizer_statistics(
            sim["mean_input"], sim["mean_target"],
            sim["std_input"], sim["std_target"])
        self.real_residual_model.set_normalizer_statistics(
            real["mean_input"], real["mean_target"],
            real["std_input"], real["std_target"])
        self._apply_gradient_boundary()

    def _apply_gradient_boundary(self) -> None:
        source_trainable = (
            self.adaptation_mode == "trainable_source_control")
        self.source_model.requires_grad_(source_trainable)
        self.real_residual_model.requires_grad_(not source_trainable)
        self.residual_gain.requires_grad_(not source_trainable)

    @classmethod
    def from_single_model(
            cls,
            model: SpatiotemporalB210,
            adaptation_mode: str,
            sim_normalizer: Mapping[str, torch.Tensor],
            real_normalizer: Mapping[str, torch.Tensor],
            source_normalizer_policy: str = "sealed_sim",
            ) -> "FullFunctionAnchoredSpatiotemporalB210":
        if model.interaction_mode != "causal_temporal_candidate":
            raise ValueError("full-function anchor requires causal ST-B210")
        result = cls(
            model.config, adaptation_mode, sim_normalizer, real_normalizer,
            source_normalizer_policy=source_normalizer_policy)
        single_state = model.state_dict()
        # Both branches first receive the same sealed Sim weight state.  The
        # persistence buffers are then installed from their declared affine
        # coordinate systems.  This final installation is essential for the
        # Real-affine-aligned experiment: load_state_dict would otherwise put
        # the sealed Sim persistence buffers back into the source branch.
        result.source_model.load_state_dict(single_state, strict=True)
        result.real_residual_model.load_state_dict(single_state, strict=True)
        if source_normalizer_policy == "clean81_real_affine":
            source = result.normalizer_payload("sim")
            result.source_model.set_normalizer_statistics(
                source["mean_input"], source["mean_target"],
                source["std_input"], source["std_target"])
        real = result.normalizer_payload("real")
        result.real_residual_model.set_normalizer_statistics(
            real["mean_input"], real["mean_target"],
            real["std_input"], real["std_target"])
        result._assert_domain_normalizers_consistent()
        result._apply_gradient_boundary()
        return result

    def _assert_domain_normalizers_consistent(self) -> None:
        for domain, member in (
                ("sim", self.source_model),
                ("real", self.real_residual_model)):
            mean_input = getattr(self, f"_{domain}_mean_input")
            mean_target = getattr(self, f"_{domain}_mean_target")
            std_input = getattr(self, f"_{domain}_std_input")
            std_target = getattr(self, f"_{domain}_std_target")
            expected = {
                "persistence_scale": std_input / std_target,
                "persistence_bias": (
                    mean_input - mean_target) / std_target,
                "target_std": std_target,
            }
            actual = member.state_dict()
            for name, value in expected.items():
                if not torch.equal(
                        actual[name].detach().cpu(), value.detach().cpu()):
                    raise RuntimeError(
                        f"{domain} member persistence buffer does not match "
                        f"its {domain} normalizer: {name}")

    # Compatibility name retained for callers that audit the source boundary.
    def _assert_source_normalizer_matches_sim(self) -> None:
        self._assert_domain_normalizers_consistent()

    def set_normalizer_statistics(self, *_args, **_kwargs) -> None:
        """Reject the generic single-normalizer installation path."""
        if self.source_normalizer_policy == "clean81_real_affine":
            raise RuntimeError(
                "Real-affine FFAR normalizers are fixed at pair construction; "
                "generic mutation is forbidden")
        raise RuntimeError(
            "FFAR-ST owns immutable Sim and Real normalizers; a common "
            "normalizer must not be installed")

    def normalizer_payload(self, domain: str) -> dict[str, torch.Tensor]:
        if domain not in ("sim", "real"):
            raise ValueError("domain must be sim or real")
        return {
            name: getattr(self, f"_{domain}_{name}").detach().cpu().clone()
            for name in _NORMALIZER_FIELDS
        }

    def source_model_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.source_model.state_dict().items()
        }

    def source_function_state(self) -> dict[str, torch.Tensor]:
        result = {
            f"source_model.{name}": value.detach().cpu().clone()
            for name, value in self.source_model.state_dict().items()
        }
        result.update({
            f"_sim_{name}": value
            for name, value in self.normalizer_payload("sim").items()})
        return result

    def source_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"source_model.{name}"
            for name, _parameter in self.source_model.named_parameters())

    def residual_parameter_names(self) -> tuple[str, ...]:
        return (
            *(f"real_residual_model.{name}"
              for name, _parameter in self.real_residual_model.named_parameters()),
            "residual_gain",
        )

    @staticmethod
    def _view(statistic: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return statistic.to(value).view(1, 1, 1, 1, -1)

    def source_physical(self, history_physical: torch.Tensor) -> torch.Tensor:
        """Apply the source weights in their explicitly bound affine system."""
        sim_history = (
            history_physical - self._view(self._sim_mean_input, history_physical)
        ) / self._view(self._sim_std_input, history_physical)
        source_normalized = self.source_model(sim_history)
        return (
            source_normalized
            * self._view(self._sim_std_target, source_normalized)
            + self._view(self._sim_mean_target, source_normalized)
        )

    def forward_physical(
            self, history_physical: torch.Tensor,
            ) -> dict[str, torch.Tensor]:
        """Return physical prediction and an auditable decomposition."""
        source = self.source_physical(history_physical)
        real_history = (
            history_physical
            - self._view(self._real_mean_input, history_physical)
        ) / self._view(self._real_std_input, history_physical)
        residual_normalized = self.real_residual_model(real_history)
        gain = self.residual_gain.to(residual_normalized).view(1, 1, 1, 1, -1)
        residual_physical = (
            gain * residual_normalized
            * self._view(self._real_std_target, residual_normalized)
        )
        combined = source + residual_physical
        return {
            "prediction_physical": combined,
            "source_physical": source,
            "residual_physical": residual_physical,
            "residual_normalized": residual_normalized,
        }

    def forward_components(self, history_real_normalized: torch.Tensor):
        history_physical = (
            history_real_normalized
            * self._view(self._real_std_input, history_real_normalized)
            + self._view(self._real_mean_input, history_real_normalized)
        )
        components = self.forward_physical(history_physical)
        prediction = (
            components["prediction_physical"]
            - self._view(
                self._real_mean_target, components["prediction_physical"])
        ) / self._view(
            self._real_std_target, components["prediction_physical"])
        return {**components, "prediction": prediction}

    def forward(self, history_real_normalized: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history_real_normalized)["prediction"]

    def architecture_config(self) -> dict[str, object]:
        result = {
            "architecture": "full_function_anchored_spatiotemporal_b210",
            "adaptation_mode": self.adaptation_mode,
            "source_function": "complete_sealed_Sim_physical_ST_B210",
            "real_residual": "complete_causal_ST_B210_times_zero_start_gain",
            "residual_gain_initialization": "exact_zero_per_output_channel",
            "source_internal_normalizer": "immutable_Sim",
            "residual_internal_normalizer": "immutable_Real",
            "input_output_api": "Real_normalized",
            "physical_fusion": True,
            "unmeasured_pressure": (
                "trained_all3_compatibility_then_zeroed_by_official_evaluator"),
            "flattened_history_bypass": False,
            **asdict(self.config),
        }
        if self.source_normalizer_policy == "clean81_real_affine":
            result.update({
                "source_function": "complete_sealed_Sim_weight_ST_B210",
                "source_internal_normalizer": (
                    "clean81_Real_affine_installed_on_sealed_Sim_weights"),
                "source_normalizer_policy": self.source_normalizer_policy,
            })
        return result

    def execution_contract(self) -> dict[str, object]:
        return {
            "source": self.source_model.axis_execution_contract(),
            "real_residual": self.real_residual_model.axis_execution_contract(),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        # There is currently no dropout or running-stat normalization in the
        # source model, but eval mode makes the frozen-function promise robust
        # to future additions.
        if self.adaptation_mode == "frozen_source_candidate":
            self.source_model.eval()
        else:
            self.real_residual_model.eval()
        return self


# Short alias used by experiment code and endpoint metadata.
FullFunctionAnchoredSTB210 = FullFunctionAnchoredSpatiotemporalB210
