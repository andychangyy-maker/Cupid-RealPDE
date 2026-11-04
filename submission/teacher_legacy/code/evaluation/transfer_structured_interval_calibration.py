#!/usr/bin/env python3
"""Transfer validated SPS calibration constants to a refitted interval head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


REQUIRED_CALIBRATION_KEYS = (
    "robust_lower_multiplier_uv",
    "robust_upper_multiplier_uv",
)
OPTIONAL_CALIBRATION_KEYS = (
    "lower_width_power_uv",
    "upper_width_power_uv",
    "horizon_calibration",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-head", type=Path, required=True)
    parser.add_argument("--calibration-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trained = torch.load(args.trained_head, map_location="cpu", weights_only=False)
    calibration = torch.load(
        args.calibration_head, map_location="cpu", weights_only=False)
    for key in ("head_hidden", "architecture", "asymmetric", "global_context"):
        if trained.get(key, False) != calibration.get(key, False):
            raise ValueError(f"Head mismatch for {key}")
    missing = [
        key for key in REQUIRED_CALIBRATION_KEYS if key not in calibration]
    if missing:
        raise ValueError(f"Calibration checkpoint missing keys: {missing}")
    output = dict(trained)
    for key in REQUIRED_CALIBRATION_KEYS + OPTIONAL_CALIBRATION_KEYS:
        if key not in calibration:
            continue
        output[key] = calibration[key]
    output["transferred_calibration"] = {
        "source": str(args.calibration_head),
        "target": str(args.trained_head),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
