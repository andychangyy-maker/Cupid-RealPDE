<div align="center">

<a href="https://github.com/andychangyy-maker/Cupid-RealPDE"><img src="docs/assets/cupid-hero.svg" alt="Cupid — Forecast the flow. Calibrate as observations arrive." width="100%"></a>

### Uncertainty-Aware Flow Forecasting<br>with Causal Online Calibration

**Yaoyu Chang · Wenhao Liu · Hao Shi**

NeurIPS 2026 RealPDE Competition · Track 2: Long-Term Test-Time Adaptation

**[Method](#how-cupid-works)** &nbsp; / &nbsp; **[Reproduce](#reproduce-cupid)** &nbsp; / &nbsp; **[Results](#results-with-provenance)** &nbsp; / &nbsp; **[Code map](docs/code-map.md)** &nbsp; / &nbsp; **[中文](README_ZH.md)**

</div>

Cupid forecasts future flow fields and their uncertainty while adapting to incoming
observations. **The neural networks stay frozen. Lightweight calibration learns
from targets that have already been revealed.** This is the complete release of
Team Cupid's competition solution, **submission 948801 (L33)**.

[![Official: joint 9th, 82.6104 ± 0.0653. Local Docker training: 31 stages on one A800 in 4 h 07 min. Release: 1,835 original package files.](docs/assets/cupid-evidence.svg)](docs/results.md)

## A real flow, in motion

![Observed public RealPDE velocity fields evolving over time; these are measurements, not model predictions.](docs/assets/real-flow-preview.gif)

<sub>Observed <i>u</i> and <i>v</i> from the public RealPDE trajectory <code>10125_10.h5</code>. Fixed color scales; illustrative playback speed. This is a data preview, not a model forecast. [Static image](docs/assets/real-flow-preview.png) · [Source, rendering details & attribution](docs/assets/README.md)</sub>

**The task:** use 20 observed frames to predict the next 20 frames, together with
lower and upper bounds. Evaluation uses a 32 × 64 grid; the Real pressure channel is zero.
The preview above shows the original 64 × 128 observation grid.

## How Cupid works

![Cupid pipeline: train teachers and student offline; use frozen networks and causally updated calibration online.](docs/assets/cupid-overview.svg)

1. **Learn offline.** Three independent Sim teachers supervise a multi-member
   U-Net. Real-data training fits point forecasts, fluctuation shape, mean
   calibration, and asymmetric uncertainty bounds.
2. **Calibrate online.** Revealed past targets update regional means, fluctuation
   amplitudes, residual distributions, and a 32-feature regularized linear readout.
   State resets for each trajectory; no neural backpropagation is used at inference.
3. **Run efficiently.** FP16 inference, Torch compilation, reusable buffers, and
   CUDA graphs execute the frozen runtime. Mixed INT8/FP16 **weight storage** does
   not imply INT8 compute kernels.

[Read the method →](docs/method.md) &nbsp; · &nbsp; [Exact 31-stage recipe →](submission/METHOD.md)

## Reproduce Cupid

**Build from source → train from scratch → load the Final API.**
The tested setup is Linux/amd64, Docker Engine, NVIDIA container support,
and one A800 80 GB. The historical private Docker Hub repository is not required.

First read the [public data guide](docs/data.md), then run from the repository root:

```bash
# Standard-library integrity check; no GPU required.
python scripts/verify_source.py

# Pinned base image; all additional Python wheels are bundled.
docker build --platform linux/amd64 -t cupid-realpde:948801 ./submission
```

<details>
<summary><strong>Run all 31 training stages with networking disabled</strong></summary>

The image build may download the pinned base image. Training itself is offline.
`DATA` must contain the official flat-HDF5 training layout; `OUTPUT` must be empty.

```bash
DATA=/absolute/path/to/public-training-data
OUTPUT=/absolute/path/to/new-empty-output
mkdir -p "$OUTPUT"

docker run --rm --network none --gpus '"device=0"' \
  --cpus 16 --memory 128000000000 --shm-size 32g \
  --mount type=bind,src="$DATA",dst=/data,readonly \
  --mount type=bind,src="$OUTPUT",dst=/output \
  cupid-realpde:948801 \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

This rebuilds every learned component, including teachers and caches, without
historical weights. The public interfaces are `train(data_dir, output_dir, device)`
and `get_ttt_model(output_dir, device)`.

</details>

**[Complete reproduction guide →](docs/reproduction.md)** · [Data layout →](docs/data.md) · [Execution records →](evidence/README.md)

## Results with provenance

| Evidence | Result | What was measured |
|:--|:--|:--|
| **Official final / private test** | **Joint 9th · 82.6104 ± 0.0653** | Organizer-reported mean ± SD over 15 evaluations |
| Official development submission | 82.186978 | Historical leaderboard score for 948801 |
| Local Docker training | **31 / 31 stages · 4 h 07 min 24 s** | One A800; 16 CPUs; 128 GB RAM; network disabled |
| Local paired public replay | 83.271780 fresh / 83.277454 sealed | 716-tail-window quality mean; −0.005674 points |

The [official final results](https://realpdecompetition.github.io/results.html)
combine three independent training runs and five evaluations per trained model.
The local replay is a **training-overlap diagnostic**. Its four-metric quality
mean excludes the official Time score and cannot be compared to official totals.

[Full results, metric definitions & limitations →](docs/results.md)

## Explore the release

| Start here | What you will find |
|:--|:--|
| [Method guide](docs/method.md) | Architecture, online state, and causal boundaries |
| [Code map](docs/code-map.md) | Entry points and the implementation behind each component |
| [Frozen submission](submission/) | All 1,835 original Final-package files, byte-for-byte |
| [Reproduction evidence](evidence/) | Docker execution, API checks, and paired public replay |
| [Source manifest](docs/frozen-source-manifest.json) | SHA256 values checked by `scripts/verify_source.py` |

<details>
<summary>About the preserved historical directories</summary>

`legacy/`, `teacher_legacy/`, and `runtime/track1/` inside the package are retained
training/runtime dependencies, not separate supported releases. The public
training entry point is **Track 2 only**. Original package documentation is kept
unchanged as a dated submission record. The later October 10 Docker Engine
reproduction is documented in [the current guide](docs/reproduction.md).

</details>

## Team, citation & acknowledgments

**Team Cupid:** Yaoyu Chang (team leader), Wenhao Liu, Hao Shi.
Software citation metadata is available in [CITATION.cff](CITATION.cff).
This repository releases a competition solution.

<details>
<summary>Copy the software citation</summary>

```bibtex
@software{cupid_realpde_2026,
  author  = {Chang, Yaoyu and Liu, Wenhao and Shi, Hao},
  title   = {Cupid: Uncertainty-Aware Flow Forecasting with Causal Online Calibration},
  year    = {2026},
  version = {948801-public.1},
  url     = {https://github.com/andychangyy-maker/Cupid-RealPDE}
}
```

</details>

We thank the [RealPDE organizers](https://realpdecompetition.github.io/) and
[RealPDEBench authors](https://github.com/AI4Science-WestlakeU/RealPDEBench)
for the benchmark, public data, and evaluation infrastructure. Please cite their
upstream work when using the benchmark.

---

Team-authored code and documentation: **[MIT](LICENSE)**. Third-party licenses
remain in [the original notices](submission/THIRD_PARTY_NOTICES.md).
The observed-flow preview derives from the **CC BY-NC 4.0** public dataset;
see [asset attribution](docs/assets/README.md) and [licensing](docs/licensing.md).
Raw training trajectories, model checkpoints, Docker layers, and credentials
are not bundled in this release.
