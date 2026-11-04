# Public training data and input layout

[Home](../README.md) · [Training guide](reproduction.md)

## Official source

Use the [official RealPDE Competition data release](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data)
and the [Track 2 Final starting kit](https://www.codabench.org/competitions/17385/).
The competition release concerns **NACA4418** flow; the broader RealPDEBench
collection contains other physical systems and is not a substitute for this
training set. No benchmark trajectories or external pretrained weights are
bundled in this repository.

The upstream data card marks the data CC BY-NC 4.0 and explicitly excludes
`7575_0.h5`, which duplicates another case. Preserve your downloaded archive;
omit that file from the directory mounted for training.

## Required layout

The frozen training package expects the competition-ready **flat HDF5** layout:

```text
public-training-data/
├── train_sim/                 # exactly 100 .h5 trajectories
├── train_real/                # exactly the 81 files in assets/manifest.json
└── mean_std_real.pt           # official normalization artifact
```

In the verified local release, each Real file has root datasets `u`, `v`, `x`,
`y`, `t`, `aoa`, and `re`; Sim files additionally have `p`. Raw spatial arrays
are 64 × 128. The recipe performs the competition subsampling; do not pre-resize
them to 32 × 64. `aoa` and `re` are scalar metadata.

The manifest fixes each Real filename, frame count, angle, and window selection.
It contains 2,544 prefix windows and 716 diagnostic tail windows. A complete
causal replay has 3,341 steps; this is a different view from the prefix/tail split.

The public normalization file is already included at
[`submission/assets/mean_std_real.pt`](../submission/assets/mean_std_real.pt).
Copy that small artifact to the data root; it is not a model checkpoint:

```bash
cp submission/assets/mean_std_real.pt "$DATA/mean_std_real.pt"
```

Expected SHA256:
`bae5235f73431ac7d76e6bdcea480b8ca9fda0ac01e57ab26af66bad20dc3b40`.

## Check layout before launching a GPU

The current upstream data card also describes raw training archives with fields
under `measured_data/`. **That nested layout is not directly accepted by this
frozen package**, whose `inspect_data()` reads root-level `u` and `aoa`.
Use the competition-ready files, or the optional staging helper below, then run
the CPU preflight in [reproduction.md](reproduction.md) before training.

[`scripts/prepare_data.py`](../scripts/prepare_data.py) creates a separate prepared
directory. Flat files are copied byte-for-byte. For nested files, it requires a
unique dataset for every required field and copies those datasets to the root,
preserving values, dtypes, and dataset attributes. It does not resize arrays,
invent metadata, change physical units, or infer angle/Reynolds values from
filenames. Ambiguous or missing fields cause an error before copying begins.

```bash
# RAW contains unpacked train_real/ and train_sim/ directories.
# PREPARED is separate and empty. These copies require additional disk space.
RAW=/absolute/path/to/unpacked-release
PREPARED=/absolute/path/to/new-empty-prepared-data
mkdir -p "$PREPARED"
docker run --rm --network none \
  --mount type=bind,src="$RAW",dst=/raw,readonly \
  --mount type=bind,src="$PREPARED",dst=/prepared \
  --mount type=bind,src="$(pwd)/scripts",dst=/tools,readonly \
  cupid-realpde:948801 python /tools/prepare_data.py \
  --real /raw/train_real --sim /raw/train_sim --output /prepared --package /workspace
DATA="$PREPARED"
```

The helper is a new packaging convenience, outside the unchanged training image
source. Its flat/nested copy behavior is checked with fixtures and its metadata
contract against our local training data. The current remote raw archive has not
been fully downloaded again for this repository preparation. A reorganized HDF5
file will have different container bytes; this is not a claim of byte equality
to our archived inputs. Preserve the generated `preparation_report.json` and
check the CPU preflight before any GPU use.

## Evaluation boundary

These are public training trajectories. Full Real fitting uses the public tails,
and the five calibration folds are stage-only OOF rather than full-pipeline
holdouts. Local replay is useful for checking reproduction, not for estimating
unseen-date, unseen-angle, or official private-test generalization.
