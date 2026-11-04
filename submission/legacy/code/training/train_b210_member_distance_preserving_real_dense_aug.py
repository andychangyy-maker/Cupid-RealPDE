#!/usr/bin/env python3
"""Paired Real4k test of preserving Sim member-readout diversity.

The candidate adds no deployed parameters.  During Real fine-tuning it applies
a one-sided penalty only when pairwise RMS distances between the four
multi-scale member readout weights fall below their Sim-start values.  This is
the bounded, no-inference-cost counterpart of the last-layer weight diversity
regularizer proposed by DiverseNO.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader


EVALUATION_ROOT = Path(__file__).resolve().parents[1] / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_scratch_real_dense_aug as paired
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1
from models.ensemble_unet2d import EnsembleUNet2d


EXPERIMENT = "b210_member_distance_preserving_real_dense_aug"
SCHEMA = "b210_member_distance_preserving_real_dense_aug_v1"
DISTANCE_WEIGHT = 0.005
DISTANCE_TARGET_RATIO = 1.0
EXPECTED_MEMBERS = 4
EXPECTED_SCALES = 3
EXPECTED_PAIRS = EXPECTED_SCALES * EXPECTED_MEMBERS * (EXPECTED_MEMBERS - 1) // 2
ORIGINAL_TRAIN_CANDIDATE = small4k.train_candidate


def identical_dense_augmentation() -> dict[str, dict[str, float]]:
    return {
        "control": dict(paired.AUGMENTATION),
        "candidate": dict(paired.AUGMENTATION),
    }


def identical_dense_windows(trajectories, _manifest_windows) -> dict[str, list]:
    windows = paired.dense_windows(trajectories, 5)
    return {"control": list(windows), "candidate": list(windows)}


def member_readout_weights(model: EnsembleUNet2d) -> tuple[tuple[torch.Tensor, ...], ...]:
    heads = model.member_point_heads
    if len(heads) != EXPECTED_MEMBERS:
        raise RuntimeError("member-distance experiment requires four point members")
    result = tuple(tuple(head.weight for head in member) for member in heads)
    if any(len(member) != EXPECTED_SCALES for member in result):
        raise RuntimeError("member-distance experiment requires three readout scales")
    if any(weight.ndim != 4 for member in result for weight in member):
        raise RuntimeError("member readout weight rank changed")
    return result


def pairwise_readout_rms(model: EnsembleUNet2d) -> torch.Tensor:
    """Return deterministic scale-major pairwise member weight distances."""
    weights = member_readout_weights(model)
    distances = []
    for scale in range(EXPECTED_SCALES):
        for left in range(EXPECTED_MEMBERS):
            for right in range(left + 1, EXPECTED_MEMBERS):
                delta = weights[left][scale].float() - weights[right][scale].float()
                distances.append(delta.square().mean().sqrt())
    value = torch.stack(distances)
    if value.numel() != EXPECTED_PAIRS or not bool(torch.isfinite(value).all()):
        raise FloatingPointError("member readout distance vector is invalid")
    return value


def distance_preservation_loss(current: torch.Tensor,
                               reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if current.shape != reference.shape or current.numel() != EXPECTED_PAIRS:
        raise ValueError("member-distance current/reference contract changed")
    if bool((reference <= 0.0).any()) or not bool(torch.isfinite(reference).all()):
        raise ValueError("Sim member readouts have zero/non-finite diversity")
    ratio = current / reference.detach()
    shortfall = torch.relu(DISTANCE_TARGET_RATIO - ratio)
    return shortfall.square().mean(), ratio


class DistancePreserver:
    def __init__(self, model: EnsembleUNet2d):
        self.model = model
        self.reference = pairwise_readout_rms(model).detach().clone()
        self.parameters = tuple(
            weight for member in member_readout_weights(model) for weight in member)
        self.gradient_seen = False
        self.active_updates = 0
        self.minimum_ratio = math.inf
        self.last_ratio_mean = math.nan

    def __call__(self, iteration: int, model: EnsembleUNet2d) -> Mapping[str, float]:
        if model is not self.model:
            raise RuntimeError("member-distance callback received another model")
        current = pairwise_readout_rms(model)
        loss, ratio = distance_preservation_loss(current, self.reference)
        weighted = DISTANCE_WEIGHT * loss
        gradients = torch.autograd.grad(
            weighted, self.parameters, retain_graph=False, allow_unused=False)
        gradient_norm = float(torch.stack([
            value.detach().float().square().sum() for value in gradients
        ]).sum().sqrt())
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("member-distance gradient is non-finite")
        for parameter, gradient in zip(self.parameters, gradients, strict=True):
            if parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.add_(gradient.detach())
        minimum = float(ratio.detach().min())
        mean = float(ratio.detach().mean())
        active_fraction = float((ratio.detach() < DISTANCE_TARGET_RATIO).float().mean())
        self.gradient_seen |= gradient_norm > 0.0
        self.active_updates += int(active_fraction > 0.0)
        self.minimum_ratio = min(self.minimum_ratio, minimum)
        self.last_ratio_mean = mean
        return {
            "member_distance_preservation_loss": float(loss.detach()),
            "member_distance_weighted_loss": float(weighted.detach()),
            "member_distance_ratio_min": minimum,
            "member_distance_ratio_mean": mean,
            "member_distance_active_fraction": active_fraction,
            "member_distance_gradient_norm": gradient_norm,
            "member_distance_iteration": float(iteration),
        }

    def report(self) -> dict[str, float | bool | int]:
        return {
            "weight": DISTANCE_WEIGHT,
            "target_ratio": DISTANCE_TARGET_RATIO,
            "pairs": EXPECTED_PAIRS,
            "gradient_seen": self.gradient_seen,
            "active_updates": self.active_updates,
            "minimum_ratio": self.minimum_ratio,
            "terminal_ratio_mean": self.last_ratio_mean,
        }


def endpoint_distance_ratios(start_state: Mapping[str, torch.Tensor],
                             endpoint_state: Mapping[str, torch.Tensor]) -> dict[str, float]:
    def state_distances(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = []
        for scale in range(EXPECTED_SCALES):
            for left in range(EXPECTED_MEMBERS):
                for right in range(left + 1, EXPECTED_MEMBERS):
                    left_key = f"member_point_heads.{left}.{scale}.weight"
                    right_key = f"member_point_heads.{right}.{scale}.weight"
                    if left_key not in state or right_key not in state:
                        raise KeyError("member point readout missing from endpoint")
                    delta = state[left_key].float() - state[right_key].float()
                    values.append(delta.square().mean().sqrt())
        return torch.stack(values)

    start = state_distances(start_state)
    endpoint = state_distances(endpoint_state)
    if bool((start <= 0.0).any()):
        raise RuntimeError("start checkpoint has collapsed member readouts")
    ratio = endpoint / start
    return {
        "minimum": float(ratio.min()),
        "mean": float(ratio.mean()),
        "maximum": float(ratio.max()),
    }


def train_arm(*args, **kwargs):
    model = kwargs.get("model")
    if model is None and args:
        model = args[0]
    access = kwargs.get("access")
    if access is None and len(args) >= 9:
        access = args[8]
    arm = getattr(access, "arm", None)
    if not isinstance(model, EnsembleUNet2d) or arm not in {"control", "candidate"}:
        raise RuntimeError("member-distance paired arm contract is incomplete")
    preserver = DistancePreserver(model) if arm == "candidate" else None
    if preserver is not None:
        if kwargs.get("extra_backward") is not None:
            raise ValueError("member-distance experiment cannot stack extra backward hooks")
        kwargs["extra_backward"] = preserver
        kwargs["extra_backward_contract"] = {
            "method": "one-sided pairwise member-readout RMS preservation",
            "weight": DISTANCE_WEIGHT,
            "target_ratio": DISTANCE_TARGET_RATIO,
            "pairs": EXPECTED_PAIRS,
            "inference_parameters_added": 0,
        }
    training, seal = ORIGINAL_TRAIN_CANDIDATE(*args, **kwargs)
    training["member_distance_preservation"] = (
        None if preserver is None else preserver.report())
    return training, seal


class PostSealTailAudit:
    def __init__(self, seals: Mapping[str, Mapping[str, object]]):
        self.seals = seals

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "held" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid member-distance diagnostic tail open")
        for seal in self.seals.values():
            path = Path(str(seal["path"]))
            if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
                raise RuntimeError("endpoint changed before diagnostic")

    def record_open(self, _row: dict, _phase: str) -> None:
        return


def add_endpoint_diagnostics(args, result: dict) -> dict:
    """Measure endpoint weight diversity and error correlation post seal."""
    device = torch.device(args.device)
    start_payload = torch.load(args.start_sim, map_location="cpu", weights_only=False)
    start_state = start_payload["model_state_dict"]
    rows, _manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = paired.load_evaluation_rows(args.evaluation_manifest, rows)
    seals = {arm: result["arms"][arm]["seal"] for arm in ("control", "candidate")}
    audit = PostSealTailAudit(seals)
    trajectories, _metadata, windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", audit)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    statistics = torch.load(
        args.output_dir / "point_normalizer.pt", map_location="cpu",
        weights_only=False)
    normalizer = smu1.FoldNormalizer(statistics, device)
    expected_config, expected_parameters, expected_controller = (
        paired.architecture_contract(False, False))
    from diagnose_ensemble_disagreement import diagnose

    diagnostics = {}
    for arm in ("control", "candidate"):
        seal = seals[arm]
        payload = torch.load(
            Path(str(seal["path"])), map_location="cpu", weights_only=False)
        ratios = endpoint_distance_ratios(start_state, payload["model_state_dict"])
        model = small4k.load_primary_endpoint(
            seal, normalizer, device,
            expected_model_config=expected_config,
            expected_parameters=expected_parameters,
            expected_controller_parameters=expected_controller)
        raw = diagnose(model, loader, normalizer, device, element_stride=197)
        diagnostics[arm] = {
            "readout_distance_ratio_to_sim": ratios,
            **{key: float(raw[key]) for key in (
                "native_sps_score", "zero_disagreement_native_sps_score",
                "mean_member_spread", "element_spread_log_spearman",
                "sample_spread_spearman",
                "member_error_pairwise_correlation_mean",
                "top_spread_decile_error_amplification",
                "top_error_decile_recall_at_top_spread_decile")},
        }
        diagnostics[arm]["native_minus_collapsed_sps"] = (
            diagnostics[arm]["native_sps_score"]
            - diagnostics[arm]["zero_disagreement_native_sps_score"])
        del model
        torch.cuda.empty_cache()
    result["member_distance_diagnostic"] = diagnostics

    delta = result["comparison"]["delta"]
    point = (0.46 * float(delta["rel_l2_score"])
             + 0.10 * float(delta["tke_rel_l2_score"])
             + 0.09 * float(delta["mvpe_rel_l2_score"]))
    sps = float(delta["native_sps_score"])
    non_time = point + 0.25 * sps
    official_aoas = []
    for row in result["comparison"]["by_aoa"].values():
        values = row["delta"]
        official_aoas.append(
            0.46 * float(values["rel_l2_score"])
            + 0.10 * float(values["tke_rel_l2_score"])
            + 0.09 * float(values["mvpe_rel_l2_score"])
            + 0.25 * float(values["native_sps_score"]))
    control = diagnostics["control"]
    candidate = diagnostics["candidate"]
    conditions = {
        "point_not_damaged": point >= -0.03,
        "sps_improves": sps >= 0.10,
        "non_time_improves": non_time >= 0.03,
        "four_of_five_aoa_nonnegative": sum(value >= 0.0 for value in official_aoas) >= 4,
        "endpoint_distance_preserved_better": (
            candidate["readout_distance_ratio_to_sim"]["mean"]
            - control["readout_distance_ratio_to_sim"]["mean"] >= 0.02),
        "member_error_correlation_reduced": (
            candidate["member_error_pairwise_correlation_mean"]
            - control["member_error_pairwise_correlation_mean"] <= -0.03),
    }
    result["member_distance_gate"] = {
        "point_contribution_delta": point,
        "sps_delta": sps,
        "non_time_contribution_delta": non_time,
        "official_aoa_contribution_deltas": official_aoas,
        "conditions": conditions,
        "pass": all(conditions.values()),
        "decision": (
            "PASS_MEMBER_DISTANCE_PRESERVATION"
            if all(conditions.values()) else "STOP_MEMBER_DISTANCE_PRESERVATION"),
    }
    paired.dr1.write_json_atomic(args.output_dir / "result.json", result)
    return result


def install_contract() -> None:
    paired.EXPERIMENT = EXPERIMENT
    paired.SCHEMA = SCHEMA
    paired.CONTROL_STRIDE = 5
    paired.CANDIDATE_STRIDE = 5
    paired.arm_window_contract = identical_dense_windows
    paired.arm_augmentation_contract = identical_dense_augmentation
    paired.small4k.train_candidate = train_arm


def main() -> None:
    install_contract()
    args = paired.parse_args()
    result = paired.run(args)
    result = add_endpoint_diagnostics(args, result)
    print(json.dumps({
        "event": "b210_member_distance_preserving_complete",
        "gate": result["member_distance_gate"],
        "result": str(args.output_dir / "result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
