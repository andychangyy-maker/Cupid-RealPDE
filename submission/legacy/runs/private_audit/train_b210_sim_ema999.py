#!/usr/bin/env python3
"""Train B210 from scratch and publish the continuous EMA999 Sim endpoint."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as b210  # noqa: E402


METHOD = "b210_realclosure_sim10k_continuous_ema999_v1"
DECAY = 0.999
_ORIGINAL_SIM_UPDATE = b210.sim_update
_ORIGINAL_RUN_EXPERIMENT = b210.run_experiment
_EMA_STATE: OrderedDict[str, torch.Tensor] | None = None
_MODEL: torch.nn.Module | None = None


def initialize_ema(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().clone())
        for name, value in model.state_dict().items())


@torch.no_grad()
def update_ema(
        ema: OrderedDict[str, torch.Tensor], model: torch.nn.Module,
        decay: float = DECAY) -> None:
    current = model.state_dict()
    if current.keys() != ema.keys():
        raise RuntimeError("EMA/model state keys changed during training")
    grouped_ema: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    grouped_current: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for name, value in current.items():
        target = ema[name]
        if value.is_floating_point():
            group = (value.device, value.dtype)
            grouped_ema.setdefault(group, []).append(target)
            grouped_current.setdefault(group, []).append(value.detach())
        else:
            target.copy_(value)
    for group in grouped_ema:
        torch._foreach_lerp_(
            grouped_ema[group], grouped_current[group], 1.0 - decay)


def ema_sim_update(*args, **kwargs):
    global _EMA_STATE, _MODEL
    if len(args) < 12:
        raise TypeError("B210 sim_update positional contract changed")
    model = args[0]
    arm = str(args[11])
    if arm == "candidate" and _EMA_STATE is None:
        _EMA_STATE = initialize_ema(model)
        _MODEL = model
    result = _ORIGINAL_SIM_UPDATE(*args, **kwargs)
    if arm == "candidate":
        if _EMA_STATE is None or _MODEL is not model:
            raise RuntimeError("candidate EMA model identity changed")
        update_ema(_EMA_STATE, model)
        result[0]["ema_decay"] = DECAY
        result[0]["ema_updates"] = int(args[8])
    return result


def ema_run_experiment(*args, **kwargs):
    summary = _ORIGINAL_RUN_EXPERIMENT(*args, **kwargs)
    if summary.get("completed_updates") != b210.TERMINAL_UPDATE:
        return summary
    if _EMA_STATE is None or _MODEL is None:
        raise RuntimeError("terminal Sim run has no EMA state")

    public_args, sim_args = args[0], args[1]
    raw_loader, normalizer = args[2], args[3]
    soft_sps_normalizer, teacher_cache = args[4], args[5]
    scale_contract, device = args[7], args[8]
    raw_checkpoint = Path(summary["checkpoint"])
    raw_sha = summary["checkpoint_sha256"]
    renamed_raw = public_args.output_dir / "raw_model_10000.pth"
    if renamed_raw.exists():
        raise FileExistsError(renamed_raw)
    raw_checkpoint.rename(renamed_raw)

    _MODEL.load_state_dict(_EMA_STATE, strict=True)
    ema_anchor = b210.sim_hgrad.evaluate_anchor(
        _MODEL, raw_loader.dataset, list(range(b210.ANCHOR_SIZE)),
        normalizer, soft_sps_normalizer, teacher_cache, scale_contract,
        sim_args, device)
    checkpoint, checkpoint_sha = b210.save_terminal_checkpoint(
        public_args.output_dir, _MODEL, sim_args,
        summary["history"], normalizer)
    raw_anchor = summary["paired_candidate"]["terminal_anchor"]
    component_delta = {
        key: float(ema_anchor[key] - raw_anchor[key])
        for key in b210.POINT_COMPONENT_KEYS
    }
    summary["method"] = METHOD
    summary["checkpoint"] = checkpoint
    summary["checkpoint_sha256"] = checkpoint_sha
    summary["paired_candidate"]["raw_terminal_anchor"] = raw_anchor
    summary["paired_candidate"]["terminal_anchor"] = ema_anchor
    summary["ema999"] = {
        "decay": DECAY,
        "continuous_updates": b210.TERMINAL_UPDATE,
        "initialized_before_candidate_update1": True,
        "raw_checkpoint": str(renamed_raw),
        "raw_checkpoint_sha256": raw_sha,
        "ema_checkpoint": checkpoint,
        "ema_checkpoint_sha256": checkpoint_sha,
        "ema_anchor": ema_anchor,
        "ema_minus_raw_component_delta": component_delta,
        "ema_minus_raw_official_point_contribution": float(
            0.46 * component_delta["rel_l2_score"]
            + 0.10 * component_delta["tke_rel_l2_score"]
            + 0.09 * component_delta["mvpe_rel_l2_score"]),
        "deployment_graph_changed": False,
    }
    return summary


def main() -> None:
    b210.sim_update = ema_sim_update
    b210.run_experiment = ema_run_experiment
    b210.REAL_CLOSURE_SEED1_METHOD = METHOD
    b210.main()


if __name__ == "__main__":
    main()
