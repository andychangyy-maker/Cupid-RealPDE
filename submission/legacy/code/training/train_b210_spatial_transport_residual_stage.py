#!/usr/bin/env python3
"""Run the frozen-base residual stage with a spatial transport branch."""

from __future__ import annotations

import sys

import train_b210_temporal_band_residual_stage as stage


def main() -> None:
    if "--spatial-warp" not in sys.argv:
        sys.argv.append("--spatial-warp")
    stage.main()


if __name__ == "__main__":
    main()
