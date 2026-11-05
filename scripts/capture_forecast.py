"""Replay one public trajectory causally and save one forecast for visualization.

Uses a completed Final output directory. No training, model patches, or scoring.
The resulting arrays are local rendering inputs, not part of the public release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_window(handle, begin):
    raw = np.zeros((1, 20, 32, 64, 3), dtype=np.float32)
    for channel, field in enumerate(('u', 'v')):
        raw[0, ..., channel] = handle[field][begin:begin + 20, ::2, ::2]
    return torch.from_numpy(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained-output', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True, help='Input-window origin, a multiple of 20')
    parser.add_argument('--output', type=Path, required=True, help='New local output directory')
    parser.add_argument('--device', choices=('cuda',), default='cuda',
                        help='The frozen L33 runtime requires CUDA Graphs')
    args = parser.parse_args()
    if args.start < 0 or args.start % 20:
        raise ValueError('Start must be a nonnegative multiple of 20')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'submission'))
    import solution

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('The frozen L33 runtime requires a CUDA GPU; CPU is unsupported')
    started = time.perf_counter()
    model = solution.get_ttt_model(str(args.trained_output), args.device)
    compile_status = {
        'common_compiled': bool(model.common_compiled),
        'common_graph': model._common_graph is not None,
        'refresh_graph': model._refresh_graph is not None,
        'readout_compiled': bool(model.readout.compiled),
        'readout_observe_graph': model.readout.observe_graph is not None,
        'readout_apply_graph': model.readout.apply_graph is not None,
    }
    if not all(compile_status.values()):
        raise RuntimeError(f'The original compiled runtime is required: {compile_status}')
    print(json.dumps({'event': 'model_ready', **compile_status}), flush=True)
    model.reset_ttt_state()
    stats = {key: value.cpu() for key, value in model.official_stats.items()}
    previous = previous_raw = first = selected = None
    steps = []
    with h5py.File(args.trajectory, 'r') as handle, torch.inference_mode():
        if args.start + 40 > len(handle['u']):
            raise ValueError('Forecast exceeds the available trajectory')
        for start in range(0, args.start + 1, 20):
            raw = raw_window(handle, start) if previous_raw is None else previous_raw
            x = (raw - stats['mean_input']) / stats['std_input']
            point, info = model.ttt_step(x, prev_target_norm=previous)
            bundle = torch.stack([point, info['lower'], info['upper']]).cpu().numpy().copy()
            if bundle.shape != (3, 1, 20, 32, 64, 3) or not np.isfinite(bundle).all():
                raise ValueError('Invalid model output')
            if not (bundle[1] <= bundle[2]).all() or np.count_nonzero(bundle[..., 2]):
                raise ValueError('Bounds or Real pressure contract violated')
            if first is None:
                first = (x.clone(), bundle.copy())
            # Only now read this forecast's target; it is feedback for the next step.
            target = raw_window(handle, start + 20)
            previous_raw = target
            previous = (target - stats['mean_target']) / stats['std_target']
            steps.append({'input_start': start, 'feedback_target_start': None if start == 0 else start,
                          'forecast_target_start': start + 20})
            if start == args.start:
                physical = bundle[:, 0] * stats['std_target'].numpy() + stats['mean_target'].numpy()
                selected = dict(input=raw[0].numpy(), truth=target[0].numpy(),
                                prediction=physical[0], lower=physical[1], upper=physical[2],
                                times=np.asarray(handle['t'][start + 20:start + 40]))
            print(json.dumps({'event': 'forecast', 'input_start': start}), flush=True)
        model.reset_ttt_state()
        point, info = model.ttt_step(first[0], prev_target_norm=None)
        repeated = torch.stack([point, info['lower'], info['upper']]).cpu().numpy()
        reset_error = float(np.abs(repeated - first[1]).max())
        if reset_error > 1e-6:
            raise ValueError(f'Trajectory reset changed the first forecast: {reset_error}')

    target = args.output / 'forecast.npz'
    np.savez_compressed(target, **selected)
    package = args.trained_output / 'model'
    report = {
        'kind': 'single public-training trajectory qualitative replay; not an official score',
        'complete': True, 'trajectory': args.trajectory.name, 'trajectory_sha256': sha(args.trajectory),
        'input_start': args.start, 'target_frame_indices': list(range(args.start + 20, args.start + 40)),
        'target_times': selected['times'].tolist(), 'replay_steps': steps,
        'device': args.device, 'torch': torch.__version__, 'model_dtype': str(model.dtype),
        'gpu_name': torch.cuda.get_device_name(), 'cuda_build': torch.version.cuda,
        'compile_status': compile_status,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'cpu_threads': torch.get_num_threads(), 'reset_maxabs': reset_error,
        'elapsed_seconds': time.perf_counter() - started,
        'timing_scope': 'Unoptimized qualitative replay including loading and checks; not a benchmark',
        'source_hashes': {p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob('*'))
                          if p.is_file() and '__pycache__' not in p.parts},
        'capture_script_sha256': sha(Path(__file__)), 'array_sha256': sha(target),
        'data_transform': 'Original float32 conversion and every-second-point subsampling; zero pressure',
        'causality': 'Reset at origin 0; target is read after its forecast, then supplied on the next step',
    }
    (args.output / 'capture.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'steps': len(steps), 'reset_maxabs': reset_error}), flush=True)


if __name__ == '__main__':
    main()
