"""Stage public HDF5 files without changing arrays, units, dtype, or resolution.

Flat files are copied verbatim. Nested files are accepted only when each required
dataset name has exactly one match. No metadata is inferred from filenames.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py


def inspect_file(path, sim, row=None):
    required = ('u', 'v', 'x', 'y', 't', 'aoa', 're') + (('p',) if sim else ())
    with h5py.File(path, 'r') as h:
        datasets = []
        h.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        mapping = {}
        for key in required:
            matches = [key] if key in datasets else [n for n in datasets if n.rsplit('/', 1)[-1] == key]
            if len(matches) != 1:
                raise ValueError(f'{path.name}: expected one dataset named {key}; found {matches}')
            mapping[key] = matches[0]
        u = h[mapping['u']]
        if len(u.shape) != 3 or u.shape[1:] != (64, 128):
            raise ValueError(f'{path.name}: expected unresized [T,64,128], got {u.shape}')
        for key in ('v',) + (('p',) if sim else ()):
            if h[mapping[key]].shape != u.shape:
                raise ValueError(f'{path.name}: {key} shape differs from u')
        if h[mapping['t']].shape != (u.shape[0],):
            raise ValueError(f'{path.name}: t shape differs from frame count')
        for key in ('aoa', 're'):
            if h[mapping[key]].shape != ():
                raise ValueError(f'{path.name}: {key} must be scalar metadata')
        if row and (len(u) != row['frames'] or int(h[mapping['aoa']][()]) != row['aoa']
                    or int(h[mapping['re']][()]) != row['re']):
            raise ValueError(f'{path.name}: metadata differs from frozen Real manifest')
        return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--sim', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1] / 'submission')
    args = parser.parse_args()
    real, sim, output = args.real.resolve(), args.sim.resolve(), args.output.resolve()
    for source in (real, sim):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output and source trees must be separate')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty output directory; no files are overwritten')
    rows = json.loads((args.package / 'assets/manifest.json').read_text())
    expected = {r['file']: r for r in rows}
    real_files = {p.name: p for p in real.glob('*.h5') if p.name != '7575_0.h5'}
    if real_files.keys() != expected.keys():
        raise ValueError('Real file set must equal the frozen clean81 manifest after excluding 7575_0.h5')
    sim_files = sorted(sim.glob('*.h5'))
    if len(sim_files) != 100:
        raise ValueError('Expected exactly 100 Sim files')
    # Validate the whole input layout before copying any data.
    jobs = []
    for group, files in (('train_real', sorted(real_files.values())), ('train_sim', sim_files)):
        for path in files:
            mapping = inspect_file(path, group == 'train_sim', expected.get(path.name) if group == 'train_real' else None)
            jobs.append((group, path, mapping))
    records = []
    for group, path, mapping in jobs:
        target = output / group / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        flat = all(k == v for k, v in mapping.items())
        if flat:
            shutil.copy2(path, target)
        else:
            with h5py.File(path, 'r') as source, h5py.File(target, 'x') as destination:
                for key, value in source.attrs.items():
                    destination.attrs[key] = value
                for key, name in mapping.items():
                    source.copy(name, destination, name=key)
        records.append({'file':f'{group}/{path.name}', 'mode':'byte_copy' if flat else 'dataset_copy', 'mapping':mapping})
    stats = args.package / 'assets/mean_std_real.pt'
    shutil.copy2(stats, output / 'mean_std_real.pt')
    report = {'files':records, 'normalization_sha256':hashlib.sha256(stats.read_bytes()).hexdigest(),
              'excluded':['7575_0.h5'], 'array_transformations':[],
              'note':'Dataset-copy files have new container bytes; arrays and attributes are copied without numerical transforms.'}
    (output / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(records)} files; source files unchanged. Run pipeline.inspect_data next.')


if __name__ == '__main__':
    main()
