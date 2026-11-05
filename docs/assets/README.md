# Visual assets and attribution

[Repository home](../../README.md) · [Data guide](../data.md)

## Forecast gallery

`forecast-comparison.gif`, `forecast-comparison-mobile.gif`, their static PNG
counterparts, and `forecast-interval-trace.png` show actual Cupid predictions and
their relationship to public observations. These are separate from the
observation-only preview below.

- Data and derived visuals: **CC BY-NC 4.0**, credited to
  [AI4Science-WestlakeU / RealPDE and RealPDEBench contributors](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data).
- Model: Team Cupid's October 10 Docker-trained output for 948801; unchanged
  weights and runtime, with their hashes checked against the historical replay.
- Trajectory `10125_10.h5`; input 660–679; targets 680–699; all twenty horizons.
  The trajectory, window and trace point were selected before inference.
- Grid: official every-second-point subsampling to 32 × 64; original array
  orientation; stored dataset units. Bilinear display enlargement only.
- Scales: fixed across horizons, identical for truth/prediction within a channel;
  error and interval width have separate labeled scales, without clipping.
- Animation: 200 ms per horizon, illustrative playback. Static images show
  horizon 20. Desktop and mobile contain the same underlying fields.
- Trace: row 16, column 48 (zero-based); model lower–upper bounds are not a
  guarantee of out-of-sample coverage.

This is a **public-training-overlap illustration**, not a new official or held-out
result. The [gallery guide](../qualitative.md) gives causal replay details and
regeneration commands. [`forecast-provenance.json`](forecast-provenance.json)
records inputs, software, graph checks, source/asset hashes, scales and descriptive
metrics. No model checkpoint or raw trajectory is published with these graphics.

## Observed-flow preview

`real-flow-preview.gif` and its static `real-flow-preview.png` show **observations,
not predictions**, from the public RealPDE competition data released by
**AI4Science-WestlakeU / the RealPDE and RealPDEBench contributors**.

- Source: [RealPDE-Competition-Data on Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data).
- Dataset and these data-derived visualizations: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- Trajectory: `10125_10.h5`, `u` and `v`, original 64 × 128 grid.
- Zero-based frames: 0, 4, 8, …, 176; 45 frames in total.
- Changes: fixed color mapping, bilinear display enlargement, labels, and illustrative playback at 110 ms/frame. No inference, denoising, or field filtering.
- Fixed scales in stored dataset units: `u` from −0.03 to 0.19; `v` from −0.065 to 0.065. They cover all displayed values without clipping.
- Original array orientation is preserved: x increases to the right, y decreases downwards. Playback is not real-time.

The full source hash and rendering parameters are in
[`flow-preview-provenance.json`](flow-preview-provenance.json).
No raw HDF5 trajectory is included in this repository.

To regenerate from a separately obtained official flat-HDF5 file, install
NumPy, h5py, and Pillow in a visualization environment, then run:

```bash
python scripts/render_flow_preview.py /path/to/10125_10.h5 \
  --font /path/to/Arial.ttf
```

The font is a local rendering dependency; no font file is distributed here.
This visualization helper is separate from the frozen training environment.

## Team-authored graphics

- `cupid-hero.svg`: Cupid wordmark and **conceptual** streamlines; not a simulated or measured field.
- `cupid-overview.svg`: explanatory method schematic.
- `cupid-evidence.svg`: official final result and separately labeled local reproduction/source facts. Sources: [results guide](../results.md).
- `cupid-social.png`: repository link-preview cover, derived from the team-authored hero.
- `cupid-*-mobile.svg`: vertically arranged versions for narrow README layouts;
  they present the same method and evidence as the desktop graphics.

These team-authored graphics use the repository's MIT License. The root license
does not relicense the data-derived forecast gallery, observed-flow preview, or upstream dataset.
