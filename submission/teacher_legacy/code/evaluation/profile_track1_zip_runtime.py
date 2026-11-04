"""Bounded profiling and full-predict timing of an actual Track 1 ZIP."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--batch-size', type=int, choices=(1, 8, 16), default=16)
    parser.add_argument('--single-call', action='store_true')
    parser.add_argument('--samples', type=int, choices=(1, 16, 63, 64, 716), default=716)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    assert sha(args.input) == '436dbc2db9bb63ef013db7a56774ae49d7acbebf3738ac1038923f3901fffbe6'
    inputs = np.load(args.input)
    assert inputs.shape == (716, 20, 32, 64, 3) and inputs.dtype == np.float32
    inputs = inputs[:args.samples]
    call_size = len(inputs) if args.single_call else args.batch_size
    result = {'zip_sha256': sha(args.zip), 'input_sha256': sha(args.input),
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'samples': len(inputs), 'batch': call_size,
              'single_call': args.single_call, 'python': sys.version,
              'cudnn': torch.backends.cudnn.version(), 'numpy': np.__version__,
              'official_time_prediction': False}
    with tempfile.TemporaryDirectory(dir=out.parent) as directory:
        root = Path(directory)
        with zipfile.ZipFile(args.zip) as archive:
            assert set(archive.namelist()) == {
                'submission.py', 'models/__init__.py', 'models/unet2d.py',
                'models/ensemble_unet2d.py', 'ensemble_unet2d.pth'}
            archive.extractall(root)
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('submission', root / 'submission.py')
        module = importlib.util.module_from_spec(spec)
        started = time.perf_counter()
        spec.loader.exec_module(module)
        torch.cuda.synchronize()
        result['import_seconds'] = time.perf_counter() - started
        result['config'] = vars(module._state['model'].config)
        result['runtime_flags'] = {
            'cpu_threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        }
        hashes = {k: hashlib.sha256() for k in ('prediction', 'lower', 'upper')}
        durations = []
        for start in range(0, len(inputs), call_size):
            batch = inputs[start:start + call_size]
            torch.cuda.synchronize()
            started = time.perf_counter()
            values = module.predict(batch)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            for key, value in values.items():
                assert value.shape == batch.shape and value.dtype == np.float32
                assert np.isfinite(value).all()
                hashes[key].update(value.tobytes())
            assert np.all(values['lower'] <= values['upper'])
        result.update(all_predict_seconds=sum(durations), call_seconds=durations,
                      output_sha256={k: v.hexdigest() for k, v in hashes.items()})
        out.write_text(json.dumps(result, indent=2) + '\n')
        if args.profile:
            model = module._state['model']
            names = ('_decode', '_members_from_features',
                     '_energy_controlled_prediction', '_energy_controlled_members',
                     '_members_with_shared_uv_mean', '_fused_uq_features',
                     '_quantiles_from_fused', '_coverage_logits', '_utility_interval',
                     '_refine_utility_interval', '_ordered_interval_extents')
            originals = {name: getattr(model, name) for name in names}
            events = {name: [] for name in names}
            for name, original in originals.items():
                def wrapped(*a, _name=name, _original=original, **kw):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    value = _original(*a, **kw)
                    end.record()
                    events[_name].append((begin, end))
                    return value
                setattr(model, name, wrapped)
            started = time.perf_counter()
            for start in range(0, 80, 16):
                module.predict(inputs[start:start + 16])
            torch.cuda.synchronize()
            result['instrumented_five_calls_seconds'] = time.perf_counter() - started
            result['stage_device_ms_per_call'] = {
                name: sum(a.elapsed_time(b) for a, b in pairs) / len(pairs)
                for name, pairs in events.items() if pairs}
            for name, original in originals.items():
                def annotated(*a, _name=name, _original=original, **kw):
                    with torch.profiler.record_function('stage::' + _name):
                        return _original(*a, **kw)
                setattr(model, name, annotated)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                for start in range(80, 128, 16):
                    module.predict(inputs[start:start + 16])
                torch.cuda.synchronize()
            result['profile_scope'] = 'instrumented diagnostics; not speedup evidence'
            keys = prof.key_averages()
            out.with_suffix('.profile.txt').write_text(
                keys.table(sort_by='self_device_time_total', row_limit=25) + '\n')
            result['annotated_stages'] = [
                {'key': item.key, 'count': item.count,
                 'cpu_us': item.cpu_time_total, 'device_us': item.device_time_total}
                for item in keys if item.key.startswith('stage::')]
            for name, original in originals.items():
                setattr(model, name, original)
            out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('config', 'call_seconds', 'annotated_stages')}, indent=2))


if __name__ == '__main__':
    main()
