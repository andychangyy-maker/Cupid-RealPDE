#!/usr/bin/env python3
"""Launch the frozen 25.64M scaled-primary PC3D Stage-R protocol."""
from __future__ import annotations

import pc3d_scaled_primary_contract as scaled
import train_pyramid_causal3d_sim_long as stage_s

# Patch the imported Stage-S identity before importing Stage R.  Stage R then
# imports the same module object and validates the scaled checkpoint contract.
scaled.install_sim_contract(stage_s)
import train_pyramid_causal3d_real_dense_aug as stage_r  # noqa: E402


def main() -> None:
    scaled.install_real_contract(stage_r, stage_s)
    stage_r.main()


if __name__ == "__main__":
    main()
