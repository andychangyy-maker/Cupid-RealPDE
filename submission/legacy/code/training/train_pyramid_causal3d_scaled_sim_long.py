#!/usr/bin/env python3
"""Launch the frozen 25.64M scaled-primary PC3D Stage-S protocol."""
from __future__ import annotations

import pc3d_scaled_primary_contract as scaled
import train_pyramid_causal3d_sim_long as stage_s


def main() -> None:
    scaled.install_sim_contract(stage_s)
    stage_s.main()


if __name__ == "__main__":
    main()
