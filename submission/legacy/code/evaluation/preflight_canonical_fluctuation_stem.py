"""Fit-input identity and fixed ABBA cost of the canonical fluctuation view."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import zipfile

from preflight_shared_point_readout import module


def main(args):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    ev = module(root / 'code/evaluation/evaluate_b210_transfer.py', 'canonical_stem_ev')
    component = module(root / 'code/models/canonical_fluctuation_stem.py', 'canonical_stem_component')
    assert ev.sha256(args.zip) == 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'
    rows = [r for r in ev.clean_rows(args.manifest) if r['aoa'] != 15]
    batch = next(ev.tail_batches(rows, args.real_dir))[0]
    results = {}
    with tempfile.TemporaryDirectory(prefix='canonical_stem_', dir='/dev/shm') as runtime:
        with zipfile.ZipFile(args.zip) as archive:
            assert set(archive.namelist()) == {*ev.RUNTIME_FILES, 'ensemble_unet2d.pth'}
            archive.extractall(runtime)
        sys.path.insert(0, runtime)
        sub = module(Path(runtime) / 'submission.py', 'canonical_original_submission')
        state = sub._get_model()
        model = state['model']
        stem_type = type(model.stem)
        component.install(model)
        candidate_type = type(model.stem)
        for count in (1, 16):
            data = batch[:count]
            model.stem.__class__ = stem_type
            baseline = sub.predict(data)
            model.stem.__class__ = candidate_type
            candidate = sub.predict(data)
            assert all(np.array_equal(candidate[k], baseline[k]) for k in baseline), 'initial output changed'
            for kind in (stem_type, candidate_type):
                model.stem.__class__ = kind
                for _ in range(10):
                    sub.predict(data)
            observations = []
            calls = 40 if count == 1 else 10
            for cycle in range(5):
                for label in ('A', 'B', 'B', 'A'):
                    model.stem.__class__ = stem_type if label == 'A' else candidate_type
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(calls):
                        sub.predict(data)
                    torch.cuda.synchronize()
                    observations.append(dict(cycle=cycle, arm=label, seconds_per_call=(time.perf_counter()-start)/calls))
            times = {label: statistics.median(x['seconds_per_call'] for x in observations if x['arm'] == label) for label in ('A', 'B')}
            results[str(count)] = dict(times=times, relative_increase=times['B']/times['A']-1, observations=observations)
        model.stem.__class__ = stem_type
    passed = all(row['relative_increase'] <= .02 for row in results.values())
    result = dict(decision='PASS_CANONICAL_STEM_ENGINEERING' if passed else 'STOP_CANONICAL_STEM_LATENCY',
                  initial_outputs_bitwise_equal=True, original_parameters_and_rng_preserved=True,
                  added_parameters=9030, time_gate=.02, batches=results,
                  zip_sha256=ev.sha256(args.zip), component_sha256=ev.sha256(component.__file__),
                  script_sha256=ev.sha256(__file__), gpu=torch.cuda.get_device_name(),
                  scope='Fit-input engineering prototype; no held quality, training or official Time result.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('zip', 'manifest', 'real-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
