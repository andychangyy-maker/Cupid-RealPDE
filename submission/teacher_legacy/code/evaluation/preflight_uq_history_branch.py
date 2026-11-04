"""Check fixed earlier-history UQ identity, trainability and <=2% inference cost."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/training'))
import train_b210_held_residual_uq as held


def main(args):
    started = time.perf_counter()
    torch, dense, native, prefix = held.runtime()
    import h5py
    import numpy as np
    from models.ensemble_unet2d import EnsembleUNet2d
    from extend_native_uq_history import extend_history_uq
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    assert held.sha256(args.point) == '0997ff5851e4ba198c62599e3202c3e56a67095d1b0e6e58219ae5d3501f9bc3'
    payload = torch.load(args.point, map_location='cpu', weights_only=False)
    original = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config'])).float()
    original.load_state_dict(payload['model_state_dict'], strict=True)
    assert native.dr1.model_state_sha256(native.partition_state(original, uq=True)) == 'a76eb8910d1193bec114165b910ce2c1323df4202116c70e230f096d41af70da'
    candidate, report = extend_history_uq(original, native.UQ_PREFIXES)
    parameters = native.configure_uq_only(candidate)
    assert candidate.uq_fusion[0].weight.requires_grad
    assert sum(p.numel() for p in parameters) == 3680422+43200
    assert native.dr1.model_state_sha256(native.partition_state(original, uq=False)) == native.dr1.model_state_sha256(native.partition_state(candidate, uq=False))
    norm, norm_report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    assert norm_report['sha256'] == 'e52dc4f6dca11a749067fdad59c61fa26b35ae897e657e5780ff70aea8ee379e'
    del payload
    file = args.real_dir / '10125_0.h5'
    with h5py.File(file, 'r') as handle:
        assert int(handle['aoa'][()]) == 0
        u = np.asarray(handle['u'][:320, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle['v'][:320, ::2, ::2], dtype=np.float32)
    uv = np.stack((u, v, np.zeros_like(u)), axis=-1)
    history = torch.from_numpy(np.stack([uv[s:s+20] for s in range(0, 301, 20)])).cuda()
    x, _ = norm.preprocess(history, torch.zeros_like(history))
    models = {'control': original.cuda().eval(), 'candidate': candidate.cuda().eval()}
    timing, identity = {}, {}
    with torch.inference_mode():
        for batch in (1, 16):
            matmul = torch.get_float32_matmul_precision()
            convolution_tf32 = torch.backends.cudnn.allow_tf32
            try:
                # Audit coefficient-preserving expansion in FP32, as in widen_native_uq.
                torch.set_float32_matmul_precision('highest')
                torch.backends.cudnn.allow_tf32 = False
                outputs = [model.forward_distribution(x[:batch]) for model in models.values()]
                assert torch.equal(outputs[0][0], outputs[1][0]), 'UQ expansion changed initial point'
                scale = norm.std_target[:2].view(1, 1, 1, 1, 2)
                difference = max(float(((a-b)*scale).abs().max()) for a, b in zip(outputs[0][1:], outputs[1][1:]))
                assert difference <= 2e-6, ('expanded convolution initial bounds drift', difference)
                identity[batch] = dict(point_bitwise=True, physical_interval_max_abs=difference, tf32=False)
                del outputs
            finally:
                torch.set_float32_matmul_precision(matmul)
                torch.backends.cudnn.allow_tf32 = convolution_tf32
            for model in models.values():
                for _ in range(4): model.forward_distribution(x[:batch])
            rows = {'control': [], 'candidate': []}
            for label in ('control', 'candidate', 'candidate', 'control'):
                model = models[label]
                torch.cuda.synchronize(); begin = time.perf_counter()
                for _ in range(20): model.forward_distribution(x[:batch])
                torch.cuda.synchronize()
                rows[label].append((time.perf_counter()-begin)/20)
            ratio = statistics.median(rows['candidate'])/statistics.median(rows['control'])-1
            timing[batch] = dict(seconds=rows, relative_change=ratio)
    passed = all(row['relative_change'] <= .02 for row in timing.values())
    result = dict(decision='PASS_UQ_HISTORY_ENGINEERING' if passed else 'STOP_UQ_HISTORY_LATENCY',
                  initial_function=identity, timing=timing, extension=report,
                  fit_input_file=str(file), held_data_read=False, elapsed_seconds=time.perf_counter()-started,
                  source_sha256=held.sha256(__file__), scope='Engineering only; no optimizer or held quality result')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('point', 'real-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
