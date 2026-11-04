#!/usr/bin/env python3
"""Held-AoA20 discovery: deterministic PCGrad only on the shared decoder.

All four fit AoAs contribute two samples per update.  Gradients outside the
audited decoder scope are ordinary uniform ERM; only the selected gradients
are projected in fixed AoA order before the one optimizer step.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import train_b210_aoa_risk_held_fold as base


EXPERIMENT = "b210_pcgrad_shared_held_fold"
SCHEMA = "b210_pcgrad_shared_held_fold_v1"
ENDPOINT = "ema_u4000_pcgrad_shared"
ENDPOINT_SCHEMA = "b210_pcgrad_shared_endpoint_v1"
HELD_AOA = 20
PCGRAD_PREFIXES = ("up.", "decoder.", "decoder_regime_film.",
                   "mean_profile_refiner.")
SHARED_DECODER_PARAMETERS = 34_543_398


def deterministic_pcgrad(group_gradients: Sequence[torch.Tensor] | torch.Tensor
                         ) -> tuple[torch.Tensor, dict[str, float]]:
    """Fixed-order PCGrad via Gram coefficients, without per-group clones."""
    gradients = (group_gradients if isinstance(group_gradients, torch.Tensor)
                 else torch.stack(tuple(group_gradients)))
    if gradients.ndim != 2 or gradients.shape[0] == 0:
        raise ValueError("PCGrad requires at least one group")
    if not torch.isfinite(gradients).all():
        raise ValueError("PCGrad gradients must be finite matching vectors")
    group_count = gradients.shape[0]
    gram = gradients @ gradients.T
    diagonal = torch.diagonal(gram)
    if not torch.isfinite(gram).all() or not bool((diagonal > 0.0).all()):
        raise FloatingPointError("PCGrad Gram matrix is non-finite or singular")
    coefficients = torch.eye(group_count, device=gradients.device, dtype=gradients.dtype)
    conflicts = 0
    for source in range(group_count):
        # A fixed cyclic order is deterministic without privileging AoA0 as
        # the first reference for every task.
        for offset in range(1, group_count):
            target = (source + offset) % group_count
            dot = torch.dot(coefficients[source], gram[:, target])
            if dot < 0:
                coefficients[source, target].sub_(dot / diagonal[target])
                conflicts += 1
    ordinary_coefficients = torch.full((group_count,), 1.0 / group_count,
                                       device=gradients.device, dtype=gradients.dtype)
    result_coefficients = coefficients.mean(0)
    result = result_coefficients @ gradients
    correction = result_coefficients - ordinary_coefficients
    if not torch.isfinite(result).all():
        raise FloatingPointError("PCGrad produced a non-finite gradient")
    return result, {"conflict_pairs": float(conflicts),
                    "pair_count": float(group_count * (group_count - 1)),
                    "correction_norm": float(torch.sqrt((correction @ gram @ correction).clamp_min(0))),
                    "ordinary_norm": float(torch.sqrt((ordinary_coefficients @ gram @ ordinary_coefficients).clamp_min(0)))}


def shared_decoder_parameters(model, *, require_audited_size: bool = True) -> tuple[torch.nn.Parameter, ...]:
    selected = tuple(parameter for name, parameter in model.named_parameters()
                     if name.startswith(PCGRAD_PREFIXES) and parameter.requires_grad)
    if not selected:
        raise RuntimeError("PCGrad shared-decoder scope is empty")
    if require_audited_size and sum(parameter.numel() for parameter in selected) != SHARED_DECODER_PARAMETERS:
        raise RuntimeError("PCGrad shared-decoder audited parameter count changed")
    return selected


def _flatten_gradients(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("PCGrad shared-decoder gradient is missing")
    gradients = [parameter.grad.detach().reshape(-1) for parameter in parameters]
    return torch.cat(gradients)


def _assign_flattened_gradients(parameters: Sequence[torch.nn.Parameter], value: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad = value[offset:offset + size].reshape_as(parameter).clone()
        offset += size
    if offset != value.numel():
        raise RuntimeError("PCGrad gradient vector size changed")


def train_pcgrad_arm(model, dataset, schedule: Mapping[str, object], normalizer,
                     device: torch.device, deadline: float, output_dir: Path,
                     start_report: Mapping[str, object], held_aoa: int,
                     metadata: list[Mapping[str, object]]):
    trainable = base.ft12.configure_trainable(model, base.ft12.EXPECTED_PARAMETERS)
    shared = shared_decoder_parameters(model)
    shared_elements = sum(parameter.numel() for parameter in shared)
    loader = DataLoader(base.ft12.npt1.ensemble_train.TrajectoryTaggedDataset(dataset),
                        batch_sampler=schedule["batches"], num_workers=0,
                        pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(trainable, lr=base.small4k.LR,
                                  weight_decay=base.small4k.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12_000, eta_min=0.0)
    ema = base.small4k.ema_lib.FP32EMAShadow(model, base.small4k.EMA_DECAY)
    fit_aoas = tuple(aoa for aoa in base.AOA_VALUES if aoa != held_aoa)
    rows, history = [], []
    total_conflicts = 0.0
    total_pairs = 0.0
    correction_ratios = []
    for iteration, batch in enumerate(loader, start=1):
        if iteration > base.UPDATES:
            break
        if __import__('time').perf_counter() > deadline:
            raise TimeoutError("PCGrad held fold exceeded wall budget")
        history_value, target, trajectory_ids, sample_indices = batch
        expected = schedule["rows"][iteration - 1]
        actual = {"sample_indices": sample_indices.long().tolist(),
                  "trajectory_ids": trajectory_ids.long().tolist(),
                  "bootstrap_mask": expected["bootstrap_mask"]}
        if actual != expected:
            raise RuntimeError("PCGrad sample/bootstrap stream changed")
        rows.append(actual)
        partitions = base.partition_batch(trajectory_ids, metadata, fit_aoas)
        member_mask = torch.as_tensor(expected["bootstrap_mask"], dtype=torch.bool)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        group_gradients = torch.empty((len(fit_aoas), shared_elements), device=device)
        raw = []
        for group_index, aoa in enumerate(fit_aoas):
            indices = partitions[aoa]
            subbatch = tuple(value.index_select(0, indices) for value in
                             (history_value, target, trajectory_ids, sample_indices))
            values = base.ft12.npt1.backward_real_supervised_npt1(
                model, subbatch, normalizer, device, iteration,
                auxiliary_weight=base.ft12.tke_aux_weight(iteration),
                member_mask=member_mask.index_select(0, indices),
                audit_controller_gradients=False, input_augmentation=base.dense4k.AUGMENTATION,
                loss_scale=1.0)
            group_gradients[group_index].copy_(_flatten_gradients(shared)); raw.append(float(values["real_native_supervised_loss"]))
            for parameter in shared:
                parameter.grad = None
        corrected, diagnostics = deterministic_pcgrad(group_gradients)
        shared_ids = {id(parameter) for parameter in shared}
        for parameter in trainable:
            if id(parameter) not in shared_ids and parameter.grad is not None:
                parameter.grad.div_(len(fit_aoas))
        _assign_flattened_gradients(shared, corrected)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, base.small4k.GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("PCGrad clipped gradient is non-finite")
        optimizer.step(); ema.update_after_optimizer_step(model, optimizer_step=iteration); scheduler.step()
        row = {"iteration": iteration, "raw_native_supervised_loss": dict(zip(map(str, fit_aoas), raw)),
               "sample_count": {str(aoa): 2 for aoa in fit_aoas},
               "gradient_norm_before_clip": float(gradient_norm), **diagnostics,
               "conflict_trigger_rate": diagnostics["conflict_pairs"] / diagnostics["pair_count"],
               "correction_to_ordinary_norm": diagnostics["correction_norm"] / max(
                   diagnostics["ordinary_norm"], torch.finfo(torch.float32).tiny)}
        history.append(row)
        total_conflicts += diagnostics["conflict_pairs"]
        total_pairs += diagnostics["pair_count"]
        correction_ratios.append(row["correction_to_ordinary_norm"])
        if iteration == 1 or iteration % 500 == 0:
            print(json.dumps({"event": "b210_pcgrad_shared_progress", **row}), flush=True)
    if len(rows) != base.UPDATES or ema.updates != base.UPDATES:
        raise RuntimeError("PCGrad arm did not finish exactly 4000 updates")
    if base.dr1.canonical_sha256(rows) != schedule["sha256"]:
        raise RuntimeError("PCGrad stream digest changed")
    seal, ema_report = base.small4k.save_primary_endpoint(
        output_dir / "ema_u4000_pcgrad_shared.pth", model, ema, start_report=start_report,
        normalizer_report=base.ft12.normalizer_payload(normalizer), stream_sha256=schedule["sha256"])
    return ({"updates": base.UPDATES, "ema_decay": base.small4k.EMA_DECAY,
             "scheduler_t_max": 12_000, "sample_stream_sha256": schedule["sha256"],
             "sampling": schedule["sampling"], "pcgrad_prefixes": PCGRAD_PREFIXES,
             "pcgrad_shared_decoder_elements": shared_elements,
             "pcgrad_summary": {
                 "conflict_pairs": total_conflicts,
                 "pair_count": total_pairs,
                 "conflict_trigger_rate": total_conflicts / total_pairs,
                 "correction_to_ordinary_norm_mean": sum(correction_ratios) / len(correction_ratios),
                 "correction_to_ordinary_norm_max": max(correction_ratios),
             },
             "pcgrad_history": history, "ema": ema_report}, seal, rows)


@contextmanager
def endpoint_contract(held_aoa: int):
    replacements = {"EXPERIMENT": EXPERIMENT, "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
                    "PRIMARY_ENDPOINT": ENDPOINT, "UPDATES": base.UPDATES,
                    "SCHEDULER_T_MAX": 12_000, "HELD_AOA": held_aoa}
    original = {key: getattr(base.small4k, key) for key in replacements}
    try:
        for key, value in replacements.items(): setattr(base.small4k, key, value)
        yield
    finally:
        for key, value in original.items(): setattr(base.small4k, key, value)


def load_pcgrad_endpoint(seal: Mapping[str, object], normalizer,
                         device: torch.device, held_aoa: int):
    path = Path(str(seal["path"]))
    if base.dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("PCGrad endpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA or payload.get("experiment") != EXPERIMENT
            or payload.get("endpoint") != ENDPOINT or int(payload.get("held_aoa", -1)) != held_aoa
            or not isinstance(state, Mapping) or base.dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("PCGrad endpoint payload changed")
    config = base.ft12.validate_b210_model_config(payload["model_config"], base.sim_tke.b210_config("candidate"))
    model = base.small4k.EnsembleUNet2d(config).float(); model.load_state_dict(state, strict=True)
    base.ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def discovery_comparison(candidate: Mapping[str, float], control: Mapping[str, float]) -> dict:
    comparison = base.condition.metric_comparison(candidate, control)
    delta = comparison["delta"]
    gate = {"point": 0.03, "non_time": 0.05, "component": -0.05}
    checks = {"point": comparison["point_final_contribution_delta"] >= gate["point"],
              "non_time": comparison["non_time_final_contribution_delta"] >= gate["non_time"],
              "components": min(delta[key] for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")) >= gate["component"],
              "native_sps": delta["native_sps_score"] >= gate["component"]}
    return {**comparison, "gate": gate, "checks": checks,
            "decision": "PASS_PCGRAD_SHARED_DISCOVERY" if all(checks.values()) else "STOP_PCGRAD_SHARED_DISCOVERY"}


def run(args: argparse.Namespace) -> dict:
    if args.held_aoa != HELD_AOA:
        raise ValueError("PCGrad discovery is authorized only for held AoA20")
    original = {key: getattr(base, key) for key in ("EXPERIMENT", "SCHEMA", "ENDPOINT", "train_risk_arm", "endpoint_contract", "load_risk_endpoint", "discovery_comparison")}
    try:
        base.EXPERIMENT, base.SCHEMA, base.ENDPOINT = EXPERIMENT, SCHEMA, ENDPOINT
        base.train_risk_arm, base.endpoint_contract = train_pcgrad_arm, endpoint_contract
        base.load_risk_endpoint = load_pcgrad_endpoint
        base.discovery_comparison = discovery_comparison
        return base.run(args)
    finally:
        for key, value in original.items(): setattr(base, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-aoa", type=int, required=True)
    parser.add_argument("--dense-control-result", nargs=2, required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--balanced-result", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--start-sim", type=Path, required=True); parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    args = parser.parse_args(); result = run(args)
    print(json.dumps({"event": "b210_pcgrad_shared_held_fold_complete", "decision": result["dense4k_comparison"]["decision"]}), flush=True)


if __name__ == "__main__": main()
