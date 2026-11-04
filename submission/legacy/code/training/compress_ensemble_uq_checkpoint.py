#!/usr/bin/env python3
"""Create a structured channel-pruned native quantile-UQ checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


UQ_PREFIXES = ("uq_projections.", "uq_fusion.", "quantile_head.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-channels", type=int, required=True)
    return parser.parse_args()


def grouped_topk(
        importance: torch.Tensor, groups: int,
        target_channels: int) -> torch.Tensor:
    source_channels = importance.numel()
    if source_channels % groups or target_channels % groups:
        raise ValueError("source and target channels must divide groups")
    keep_per_group = target_channels // groups
    source_per_group = source_channels // groups
    selected = []
    for group in range(groups):
        start = group * source_per_group
        values = importance[start:start + source_per_group]
        indices = torch.topk(
            values, k=keep_per_group, largest=True,
            sorted=True).indices
        selected.append(indices.sort().values + start)
    return torch.cat(selected)


def conv_importance(weight: torch.Tensor) -> torch.Tensor:
    return weight.float().abs().sum(dim=tuple(range(1, weight.ndim)))


def hidden_importance(
        state: dict[str, torch.Tensor], prefix: str,
        hidden: int) -> torch.Tensor:
    importance = torch.zeros(hidden)
    for key, value in state.items():
        if not key.startswith(prefix) or not key.endswith("weight"):
            continue
        if value.ndim == 1 and value.shape[0] == hidden:
            importance += value.float().abs()
        elif value.ndim >= 2:
            if value.shape[0] == hidden:
                importance += conv_importance(value)
            if value.shape[1] == hidden:
                reduce = (0, *range(2, value.ndim))
                importance += value.float().abs().sum(dim=reduce)
    return importance


def select_channels(
        source: dict[str, torch.Tensor], config: dict,
        target_hidden: int,
        ) -> tuple[list[torch.Tensor], torch.Tensor]:
    hidden = int(config["uq_hidden_channels"])
    groups = int(config["groups"])
    projection_count = int(config["depth"]) - 1
    fusion_weight = source["uq_fusion.0.weight"]
    projection_selections = []
    for scale in range(projection_count):
        start = scale * hidden
        fusion_input_importance = fusion_weight[:, start:start + hidden]
        importance = fusion_input_importance.float().abs().sum(
            dim=(0, 2, 3))
        importance += hidden_importance(
            source, f"uq_projections.{scale}.", hidden)
        projection_selections.append(grouped_topk(
            importance, groups, target_hidden))

    fusion_importance = hidden_importance(
        source, "uq_fusion.", hidden)
    fusion_importance += source["quantile_head.weight"].float().abs().sum(
        dim=(0, 2, 3))
    fusion_selection = grouped_topk(
        fusion_importance, groups, target_hidden)
    return projection_selections, fusion_selection


def copy_projection(
        destination: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor], scale: int,
        selected: torch.Tensor) -> set[str]:
    prefix = f"uq_projections.{scale}."
    assigned = set()
    first_weight = prefix + "0.weight"
    destination[first_weight] = source[first_weight][selected].clone()
    assigned.add(first_weight)
    for suffix in (
            "0.bias", "1.weight", "1.bias",
            "3.body.0.weight", "3.body.0.bias",
            "3.body.2.bias", "3.body.3.weight", "3.body.3.bias",
            "3.body.5.bias"):
        key = prefix + suffix
        destination[key] = source[key][selected].clone()
        assigned.add(key)
    for suffix in ("3.body.2.weight", "3.body.5.weight"):
        key = prefix + suffix
        destination[key] = source[key][selected][:, selected].clone()
        assigned.add(key)
    return assigned


def copy_fusion(
        destination: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        projection_selections: list[torch.Tensor],
        fusion_selection: torch.Tensor, source_hidden: int,
        ) -> set[str]:
    assigned = set()
    context_start = len(projection_selections) * source_hidden
    context_indices = torch.arange(
        context_start, source["uq_fusion.0.weight"].shape[1])
    input_selection = torch.cat([
        selected + scale * source_hidden
        for scale, selected in enumerate(projection_selections)
    ] + [context_indices])
    key = "uq_fusion.0.weight"
    destination[key] = source[key][fusion_selection][
        :, input_selection].clone()
    assigned.add(key)
    key = "uq_fusion.0.bias"
    destination[key] = source[key][fusion_selection].clone()
    assigned.add(key)
    for block in (1, 2):
        prefix = f"uq_fusion.{block}."
        for suffix in (
                "body.0.weight", "body.0.bias", "body.2.bias",
                "body.3.weight", "body.3.bias", "body.5.bias"):
            key = prefix + suffix
            destination[key] = source[key][fusion_selection].clone()
            assigned.add(key)
        for suffix in ("body.2.weight", "body.5.weight"):
            key = prefix + suffix
            destination[key] = source[key][fusion_selection][
                :, fusion_selection].clone()
            assigned.add(key)
    key = "quantile_head.weight"
    destination[key] = source[key][:, fusion_selection].clone()
    assigned.add(key)
    key = "quantile_head.bias"
    destination[key] = source[key].clone()
    assigned.add(key)
    return assigned


def main() -> None:
    args = parse_args()
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["model_config"])
    if config.get("architecture") != "ensemble_unet":
        raise ValueError("checkpoint must use ensemble_unet")
    source_hidden = int(config["uq_hidden_channels"])
    if not 0 < args.hidden_channels < source_hidden:
        raise ValueError("target hidden channels must be smaller than source")
    groups = int(config["groups"])
    if args.hidden_channels % groups:
        raise ValueError("target hidden channels must divide normalization groups")

    source = payload["model_state_dict"]
    projection_selections, fusion_selection = select_channels(
        source, config, args.hidden_channels)
    target_config = dict(config)
    target_config["uq_hidden_channels"] = args.hidden_channels
    architecture = target_config.pop("architecture")
    model = EnsembleUNet2d(
        EnsembleUNet2dConfig(**target_config))
    destination = model.state_dict()
    assigned = set()
    for key, value in source.items():
        if key.startswith(UQ_PREFIXES):
            continue
        if key not in destination or destination[key].shape != value.shape:
            raise ValueError(f"cannot copy non-UQ tensor {key}")
        destination[key] = value.clone()
        assigned.add(key)
    for scale, selected in enumerate(projection_selections):
        assigned.update(copy_projection(
            destination, source, scale, selected))
    assigned.update(copy_fusion(
        destination, source, projection_selections,
        fusion_selection, source_hidden))
    missing = sorted(set(destination) - assigned)
    if missing:
        raise RuntimeError(f"unassigned destination tensors: {missing}")
    model.load_state_dict(destination, strict=True)

    output_payload = copy.deepcopy(payload)
    output_payload["model_config"] = {
        "architecture": architecture, **target_config}
    output_payload["model_state_dict"] = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()}
    output_payload["uq_compression"] = {
        "method": "group-balanced structured magnitude pruning",
        "source_checkpoint": str(args.checkpoint),
        "source_hidden_channels": source_hidden,
        "target_hidden_channels": args.hidden_channels,
        "projection_selections": [
            value.tolist() for value in projection_selections],
        "fusion_selection": fusion_selection.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)
    print(json.dumps({
        "event": "compressed_uq_checkpoint",
        "output": str(args.output),
        "source_hidden_channels": source_hidden,
        "target_hidden_channels": args.hidden_channels,
        "source_parameters": sum(value.numel() for value in source.values()),
        "target_parameters": sum(
            value.numel() for value in destination.values()),
    }))


if __name__ == "__main__":
    main()
