# Third-party components

The Dockerfile fixes the environment versions. Distribution notices for the
added wheels are copied verbatim into `third_party_licenses/`; the original
wheels in `wheels/` also contain their license files. Dependencies already in
the pinned PyTorch image retain their installed license notices in that image.

| Component | Version/source | License and location |
|---|---|---|
| PyTorch | 2.4.1, CUDA 12.4 official runtime image | BSD-3-Clause; installed distribution notices in the base image |
| NumPy | 1.26.4 PyPI wheel | BSD-3-Clause and bundled-library notices; `third_party_licenses/numpy-*` |
| SciPy | 1.14.1 PyPI wheel | BSD-3-Clause and bundled-library notices; `third_party_licenses/scipy-*` |
| h5py | 3.11.0 PyPI wheel | BSD-3-Clause and bundled HDF5 notices; `third_party_licenses/h5py-*` |
| einops | 0.8.0 PyPI wheel | MIT; `third_party_licenses/einops-*` |
| Ninja Python distribution | 1.11.1.4 PyPI wheel | Apache-2.0; `third_party_licenses/ninja-*` |
| TinyCC | commit `0fb54300b56512754221d80adda85ddb9815bceb` | LGPL-2.1 and source-file notices; `runtime/track2/vendor/tinycc/COPYING`, bundled source archive and `BUILD.md` |

TinyCC's source archive and build provenance are distributed alongside its
binary. A retained, unused Track 1 vendor copy has the same accompanying
notices. Unused RealPDEBench reference-model source directories from the
historical development checkout are not part of this training package.
RealPDE data normalization, interfaces and scoring helpers originate
from the competition starting kits; their provenance is described in README
and METHOD. We do not claim ownership of these upstream components.
