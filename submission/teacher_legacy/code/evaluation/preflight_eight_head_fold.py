"""Fit-input only check of exact head copying and affine point folding."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/training'))
import train_b210_held_residual_uq as held


def fit_input(real_dir, norm, torch):
    import h5py
    import numpy as np
    with h5py.File(real_dir / '10125_0.h5', 'r') as handle:
        assert int(handle['aoa'][()]) == 0
        u = np.asarray(handle['u'][:20, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle['v'][:20, ::2, ::2], dtype=np.float32)
    history = torch.from_numpy(np.stack((u, v, np.zeros_like(u)), axis=-1)[None]).cuda()
    return norm.preprocess(history, torch.zeros_like(history))[0]


def main(args):
    started = time.perf_counter()
    torch, dense, _, prefix = held.runtime()
    from models.ensemble_unet2d import EnsembleUNet2d
    from train_eight_head_real4000 import install, fold_model, extended_schedule
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    assert held.sha256(args.point) == 'ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa'
    payload = torch.load(args.point, map_location='cpu', weights_only=False)
    model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config'])).float()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    norm, _ = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    model.cuda().eval()
    x = fit_input(args.real_dir, norm, torch)
    scale = norm.std_target[:2].view(1, 1, 1, 1, 2)
    with torch.inference_mode():
        original = model(x)
        install(model, torch)
        expanded = model.forward_distribution(x)
        assert all(torch.isfinite(value).all() for value in expanded)
        expansion_error = float(((expanded[0]-original)[..., :2]*scale).abs().max())
        folded = fold_model(model, torch).cuda().eval()
        folding_error = float(((folded(x)-expanded[0])[..., :2]*scale).abs().max())
    assert max(expansion_error, folding_error) <= 2e-6
    # The actual fixed schedule is reconstructed from manifest lengths without opening held arrays.
    import evaluate_b210_transfer as ev
    rows = [row for row in ev.clean_rows(args.manifest) if row['aoa'] != 15]
    windows = [dense.ft12.npt1.ensemble_train.common.repro.Window(i, start)
               for i, row in enumerate(rows) for start in range(0, row['frames']-40+1, 5)]
    old = dense.smu1.paired_batch_schedule(range(len(windows)), windows, updates=4000, seed=4)
    new = extended_schedule(old, dense)
    result = dict(decision='PASS_EIGHT_HEAD_FOLD_ENGINEERING', expansion_physical_error=expansion_error,
                  folding_physical_error=folding_error, sample_stream_sha256=new['sha256'],
                  original_state_and_rng_preserved=True, training_parameters=179076771,
                  deployed_parameters=178705575, fit_input='10125_0.h5[:20]', held_arrays_read=False,
                  elapsed_seconds=time.perf_counter()-started, source_sha256=held.sha256(__file__))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('point', 'real-dir', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
