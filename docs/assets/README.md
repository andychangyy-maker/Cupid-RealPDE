# Visual assets and attribution

[Repository home](../../README.md) · [Data guide](../data.md)

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

These team-authored graphics use the repository's MIT License. The root license
does not relicense the observed-flow preview or the upstream dataset.
