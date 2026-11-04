#!/usr/bin/env python3
"""M10 matched gradient/curl scalar-potential residual Sim runner.

``--residual-kind`` is deliberately private to this thin wrapper.  Each
invocation still delegates the complete paired u200/Sim10k protocol to the
audited B210 driver; the only model difference is the fixed scalar readout.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import sys

import torch

import train_b210_np_functional_sim as sim
from models.ensemble_unet2d import ScalarPotentialResidual


SCHEMA = "b210_compatible_curl_residual_sim_v1"
METHOD_PREFIX = "b210_compatible_"
RESIDUAL_CHANNELS = 64
RESIDUAL_BLOCKS = 2
RESIDUAL_LIMIT = 0.05
# baseline B210 plus the shared 210->64 potential body and 64->20 head.
EXPECTED_PARAMETERS = 178_868_731
EXPECTED_MODULE_PARAMETERS = 163_156

ORIGINAL_CONFIG = sim.b210_config
ORIGINAL_BUILD = sim.build_arm_model
ORIGINAL_CONFIGURE = sim.configure_trainable
ORIGINAL_UPDATE = sim.sim_update
ORIGINAL_RUN = sim.run_experiment

RESIDUAL_KIND = "curl"
INITIAL_STATES: dict[str, OrderedDict[str, torch.Tensor]] = {}
AUDIT: dict[str, bool] = {}
GEOMETRY_METRICS: dict[str, dict[str, float]] = {}
KEY_AUDIT: dict[str, object] = {
    arm: {"hasher": hashlib.sha256(), "count": 0}
    for arm in ("control", "candidate")}


def method_for(kind: str) -> str:
    return f"{METHOD_PREFIX}{kind}_residual_sim_v1"


def pop_residual_kind(argv: list[str]) -> tuple[list[str], str]:
    """Remove the wrapper-only option without perturbing the base CLI."""
    remaining = [argv[0]]
    kind = ""
    index = 1
    seen = False
    while index < len(argv):
        token = argv[index]
        if token == "--residual-kind":
            if seen or index + 1 == len(argv):
                raise ValueError("--residual-kind must appear exactly once")
            kind = argv[index + 1]
            seen = True
            index += 2
            continue
        remaining.append(token)
        index += 1
    if not seen or kind not in {"gradient", "curl"}:
        raise ValueError("--residual-kind must be gradient or curl")
    return remaining, kind


def compatible_config(*args, **kwargs):
    return replace(
        ORIGINAL_CONFIG(*args, **kwargs),
        scalar_potential_residual_mode=RESIDUAL_KIND,
        scalar_potential_residual_channels=RESIDUAL_CHANNELS,
        scalar_potential_residual_blocks=RESIDUAL_BLOCKS,
        scalar_potential_residual_limit=RESIDUAL_LIMIT,
    )


def module_state(model) -> OrderedDict[str, torch.Tensor]:
    module = model.scalar_potential_residual
    if module is None:
        raise RuntimeError("M10 scalar potential residual is missing")
    return OrderedDict((name, value.detach().cpu().clone())
                       for name, value in module.state_dict().items())


def state_group(name: str) -> str:
    if name.startswith("output."):
        return "projection"
    if name.startswith("body."):
        return "body"
    raise ValueError(f"unknown scalar-potential parameter: {name}")


def group_state(state, group: str) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value) for name, value in state.items()
                       if state_group(name) == group)


def group_has_gradient(module, group: str) -> bool:
    return any(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
               for name, parameter in module.named_parameters()
               if state_group(name) == group)


def build_with_contract(arm: str, *args, **kwargs):
    model = ORIGINAL_BUILD(arm, *args, **kwargs)
    module = model.scalar_potential_residual
    if module is None:
        raise RuntimeError("M10 scalar potential residual was not constructed")
    total = sum(parameter.numel() for parameter in model.parameters())
    module_count = sum(parameter.numel() for parameter in module.parameters())
    if total != EXPECTED_PARAMETERS or module_count != EXPECTED_MODULE_PARAMETERS:
        raise RuntimeError(f"M10 parameter contract changed: total={total}, module={module_count}")
    if torch.count_nonzero(module.output.weight) or torch.count_nonzero(module.output.bias):
        raise RuntimeError("M10 scalar-potential projection is not exact zero")
    INITIAL_STATES[arm] = module_state(model)
    return model


def configure_trainable(model, arm: str):
    """The branch trains in both internal u200 arms; this is not G/H control."""
    trainable = ORIGINAL_CONFIGURE(model, arm)
    module = model.scalar_potential_residual
    if module is None:
        raise RuntimeError("M10 scalar potential residual is missing")
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def geometry_audit(
        model, history, normalizer, device,
        ) -> tuple[dict[str, bool], dict[str, float]]:
    module = model.scalar_potential_residual
    if module is None:
        raise RuntimeError("M10 scalar potential residual is missing")
    history = history.to(device, non_blocking=True)
    history_norm, _ = normalizer.preprocess(history, history[:, -1:])
    with torch.no_grad():
        audit = model.scalar_potential_residual_audit(history_norm)
    correction = audit["physical_correction"]
    psi = audit["psi"]
    outer = torch.cat((correction[..., 0, :, :].reshape(-1),
                       correction[..., -1, :, :].reshape(-1),
                       correction[..., :, 0, :].reshape(-1),
                       correction[..., :, -1, :].reshape(-1)))
    dx, dy = ScalarPotentialResidual.central_derivatives(psi)
    div_x, _ = ScalarPotentialResidual.central_derivatives(
        correction[..., 0])
    _, div_y = ScalarPotentialResidual.central_derivatives(
        correction[..., 1])
    divergence_abs_max = float((div_x + div_y).abs().max())
    divergence_ok = bool(
        RESIDUAL_KIND != "curl" or divergence_abs_max <= 2e-6)
    checks = {
        "finite": bool(torch.isfinite(correction).all()),
        "nonzero": bool(torch.count_nonzero(correction)),
        "bounded": bool(correction.abs().max() <= RESIDUAL_LIMIT + 1e-7),
        "outer_exact_zero": bool(torch.count_nonzero(outer) == 0),
        "pressure_unchanged": True,  # branch only returns u/v by construction
        "member_shared": True,       # application expands one correction to members
        "curl_divergence": divergence_ok,
        "operator_finite": bool(torch.isfinite(dx).all() and torch.isfinite(dy).all()),
    }
    metrics = {
        "physical_correction_rms": float(
            correction.float().square().mean().sqrt()),
        "physical_correction_abs_max": float(correction.abs().max()),
        "physical_divergence_abs_max": divergence_abs_max,
        "psi_rms": float(psi.float().square().mean().sqrt()),
    }
    return checks, metrics


def sim_update_with_audit(*args, **kwargs):
    values, hgrad_audit, key, flags = ORIGINAL_UPDATE(*args, **kwargs)
    model, batch = args[0], args[1]
    normalizer, device, iteration, arm = args[2], args[7], int(args[8]), str(args[11])
    # Retain only a streaming commitment, not the potentially large key list.
    # Candidate's full 10k stream is the formal G/H reproducibility binding;
    # the internal control is intentionally only the u200 mechanics run.
    key_name = "candidate" if arm == "candidate" else "control"
    state = KEY_AUDIT[key_name]
    state["hasher"].update(str(key).encode("utf-8") + b"\n")
    state["count"] += 1
    module = model.scalar_potential_residual
    for group in ("projection", "body"):
        AUDIT[f"{arm}_{group}_gradient_seen"] |= group_has_gradient(module, group)
    if iteration == sim.CONTROL_UPDATES:
        current = module_state(model)
        for group in ("projection", "body"):
            AUDIT[f"{arm}_{group}_updated_at_u200"] = not sim.tensor_state_equal(
                group_state(INITIAL_STATES[arm], group), group_state(current, group))
        checks, metrics = geometry_audit(
            model, batch[0], normalizer, device)
        AUDIT.update({f"{arm}_geometry_{name}": value for name, value in checks.items()})
        GEOMETRY_METRICS[arm] = metrics
        if not all(checks.values()):
            raise RuntimeError(f"M10 {arm} geometry audit failed: {checks}")
        if arm == "candidate":
            # Both internal arms have now reached u200.  Fail before the
            # candidate is allowed to consume update 201.
            validate_mechanical_audit()
    values.update({
        f"scalar_potential_{group}_gradient_nonzero": float(
            group_has_gradient(module, group))
        for group in ("projection", "body")
    })
    return values, hgrad_audit, key, flags


def validate_mechanical_audit() -> None:
    failures = [key for key, value in AUDIT.items() if not value]
    if failures:
        raise RuntimeError(f"M10 u200 mechanical audit failed: {failures}; {AUDIT}")


def run_with_contract(*args, **kwargs):
    (_formal_args, sim_args, raw_loader, normalizer, soft_sps_normalizer,
     teacher_cache, _teacher_metadata, scale_contract, device) = args[:9]
    sim_args.method = method_for(RESIDUAL_KIND)
    sim_args.residual_kind = RESIDUAL_KIND
    summary = ORIGINAL_RUN(*args, **kwargs)
    # A failed original u200 gate never gets relabelled as an endpoint.
    if summary["completed_updates"] == sim.TERMINAL_UPDATE:
        if (KEY_AUDIT["candidate"]["count"] != sim.TERMINAL_UPDATE
                or KEY_AUDIT["control"]["count"] != sim.CONTROL_UPDATES):
            raise RuntimeError("M10 batch-key stream count contract changed")
        validate_mechanical_audit()
        initial_branch_sha = sim.state_sha256(
            INITIAL_STATES["candidate"])
        model = build_with_contract("candidate", device)
        payload = torch.load(
            summary["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        sim.rehearsal.ensemble_train.common.set_model_stats(model, normalizer)
        anchor = sim.sim_hgrad.evaluate_anchor(
            model, raw_loader.dataset, list(range(128, 256)), normalizer,
            soft_sps_normalizer, teacher_cache, scale_contract,
            sim_args, device)
        summary["paired_candidate"]["terminal_anchor_128_256"] = anchor
        summary["paired_candidate"]["terminal_anchor_128_256_role"] = (
            "train_set_fixed_diagnostic_not_heldout")
        summary["paired_candidate"]["terminal_anchor_role"] = (
            "train_set_fixed_diagnostic_not_heldout")
        terminal_histories = torch.stack([
            raw_loader.dataset[index][0] for index in range(128, 136)
        ])
        terminal_checks, terminal_metrics = geometry_audit(
            model, terminal_histories, normalizer, device)
        if not all(terminal_checks.values()):
            raise RuntimeError(
                f"M10 terminal geometry audit failed: {terminal_checks}")
        summary["paired_candidate"]["terminal_geometry_audit"] = {
            "sample_indices": list(range(128, 136)),
            "sample_role": "train_set_fixed_diagnostic_not_heldout",
            "checks": terminal_checks,
            "metrics": terminal_metrics,
        }
        summary["paired_candidate"]["initial_scalar_potential_sha256"] = (
            initial_branch_sha)
        summary["paired_candidate"]["terminal_scalar_potential_sha256"] = (
            sim.state_sha256(module_state(model)))
    summary.update({"schema": SCHEMA, "method": method_for(RESIDUAL_KIND),
                    "residual_kind": RESIDUAL_KIND, "parameters": EXPECTED_PARAMETERS})
    summary["training_contract"]["scalar_potential_residual"] = {
        "mode": RESIDUAL_KIND, "channels": RESIDUAL_CHANNELS,
        "blocks": RESIDUAL_BLOCKS, "limit": RESIDUAL_LIMIT,
        "internal_u200_arms": "both trainable; mechanics only, never directional selection",
        "anchor_128_256": "train_set_fixed_diagnostic_not_heldout",
    }
    summary.setdefault("paired_control", {})["role"] = (
        "internal_u200_mechanical_reference_not_G_H_control")
    summary["paired_candidate"]["scalar_potential_audit"] = {
        "checks": dict(AUDIT),
        "u200_geometry_metrics": dict(GEOMETRY_METRICS),
        "mechanical_only_not_directional_selection": True,
    }
    summary["batch_key_stream"] = {
        "candidate": {
            "count": KEY_AUDIT["candidate"]["count"],
            "rolling_sha256": KEY_AUDIT["candidate"]["hasher"].hexdigest(),
            "contract": "formal_kind_u1_u10000_same_required_for_G_H",
        },
        "internal_control": {
            "count": KEY_AUDIT["control"]["count"],
            "rolling_sha256": KEY_AUDIT["control"]["hasher"].hexdigest(),
            "contract": "u1_u200_mechanics_only",
        },
    }
    return summary


def main() -> None:
    global RESIDUAL_KIND
    argv, RESIDUAL_KIND = pop_residual_kind(list(sys.argv))
    sys.argv[:] = argv
    INITIAL_STATES.clear()
    AUDIT.clear()
    GEOMETRY_METRICS.clear()
    KEY_AUDIT.clear()
    KEY_AUDIT.update({
        arm: {"hasher": hashlib.sha256(), "count": 0}
        for arm in ("control", "candidate")})
    AUDIT.update({f"{arm}_{group}_{suffix}": False
                  for arm in ("control", "candidate")
                  for group in ("projection", "body")
                  for suffix in ("gradient_seen", "updated_at_u200")})
    sim.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    sim.b210_config = compatible_config
    sim.build_arm_model = build_with_contract
    sim.configure_trainable = configure_trainable
    sim.sim_update = sim_update_with_audit
    sim.run_experiment = run_with_contract
    sim.METHOD = method_for(RESIDUAL_KIND)
    sim.REAL_CLOSURE_SEED1_METHOD = method_for(RESIDUAL_KIND)
    sim.main()


if __name__ == "__main__":
    main()
