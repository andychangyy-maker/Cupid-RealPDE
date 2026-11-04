"""Zero-output identity and fixed ABBA latency of the width32 point branch."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import zipfile


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main(args):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    ev = module(root / 'code/evaluation/evaluate_b210_transfer.py', 'refiner_preflight_ev')
    refiner_module = module(root / 'code/models/shared_point_feature_refiner.py', 'refiner_component')
    assert ev.sha256(args.zip) == 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'
    rows = [r for r in ev.clean_rows(args.manifest) if r['aoa'] != 15]
    batch = next(ev.tail_batches(rows, args.real_dir))[0]
    results = {}
    with tempfile.TemporaryDirectory(prefix='shared_point_preflight_', dir='/dev/shm') as runtime:
        with zipfile.ZipFile(args.zip) as archive:
            assert set(archive.namelist()) == {*ev.RUNTIME_FILES, 'ensemble_unet2d.pth'}
            archive.extractall(runtime)
        sys.path.insert(0, runtime)
        sub = module(Path(runtime) / 'submission.py', 'refiner_original_submission')
        state = sub._get_model()
        model = state['model']
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(31011)
            refiner = refiner_module.SharedPointFeatureRefiner(model.config.base).to(device=state['device'], dtype=state['dtype']).eval()
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert model.config.base == 210
        assert sum(p.numel() for p in refiner.parameters()) == 13682
        original = model._members_from_features

        def refined(x, features):
            return original(x, [*features[:-1], refiner(features[-1])])

        for count in (1, 16):
            data = batch[:count]
            model._members_from_features = original
            baseline = sub.predict(data)
            model._members_from_features = refined
            candidate = sub.predict(data)
            assert all(np.array_equal(candidate[k], baseline[k]) for k in baseline), 'initial output changed'
            for function in (original, refined):
                model._members_from_features = function
                for _ in range(10):
                    sub.predict(data)
            observations = []
            calls = 40 if count == 1 else 10
            for cycle in range(5):
                for label in ('A', 'B', 'B', 'A'):
                    model._members_from_features = original if label == 'A' else refined
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(calls):
                        sub.predict(data)
                    torch.cuda.synchronize()
                    observations.append(dict(cycle=cycle, arm=label, seconds_per_call=(time.perf_counter()-start)/calls))
            times = {label: statistics.median(x['seconds_per_call'] for x in observations if x['arm'] == label) for label in ('A', 'B')}
            results[str(count)] = dict(times=times, relative_increase=times['B']/times['A']-1, observations=observations)
        model._members_from_features = original
    passed = all(result['relative_increase'] <= .02 for result in results.values())
    result = dict(decision='PASS_TO_REAL4000' if passed else 'STOP_POINT_REFINER_LATENCY',
                  initial_outputs_bitwise_equal=True, cpu_and_cuda_rng_unchanged=True,
                  parameters=13682, time_gate=.02, batches=results, zip_sha256=ev.sha256(args.zip),
                  component_sha256=ev.sha256(refiner_module.__file__), script_sha256=ev.sha256(__file__),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  scope='Same-process engineering prototype on our GPU; not official Time or trained quality.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('zip', 'manifest', 'real-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
