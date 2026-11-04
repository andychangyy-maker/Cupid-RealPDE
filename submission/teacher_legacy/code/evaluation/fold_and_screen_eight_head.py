"""Seal the fixed affine 8-to-4 conversion before reading held15 quality."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/training'))
import train_b210_held_residual_uq as held


def main(args):
    torch, dense, _, prefix = held.runtime()
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from train_eight_head_real4000 import fold_model, INTERVENTION
    from preflight_eight_head_fold import fit_input
    from screen_random_real12000 import main as score
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    receipt = json.loads((args.root/'point8/point_endpoint.json').read_text())
    source_sha = held.sha256(receipt['checkpoint_path'])
    assert source_sha == receipt['checkpoint_sha256']
    assert receipt['updates'] == 4000 and receipt['held_aoa'] == 15 and receipt['seed'] == 4
    assert receipt['sample_stream_sha256'] == '7e7a4e6a384afd0a6edacb7607449a9070b1cbc5e655cdaaa4912bf1a2dd2682'
    assert all(receipt['eight_head_training'][k] == v for k, v in INTERVENTION.items())
    payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
    assert payload['iteration'] == 4000 and payload['held_aoa'] == 15
    assert payload['model_state_sha256'] == dense.dr1.model_state_sha256(payload['model_state_dict'])
    assert payload['sample_stream_sha256'] == receipt['sample_stream_sha256']
    config = dict(payload['model_config'])
    assert config.pop('architecture') == 'ensemble_unet'
    config = EnsembleUNet2dConfig(**config)
    control = torch.load(args.control_dir/'ema999_u4000_primary.pth', map_location='cpu', weights_only=False)
    control_config = dense.ft12.validate_b210_model_config(control['model_config'])
    assert config == replace(control_config, ensemble_members=8)
    assert payload['normalizer'] == control['normalizer']
    del control
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    sim = torch.load(args.sim, map_location='cpu', weights_only=False)
    assert held.sha256(args.sim) == receipt['start_sim_sha256']
    updates, pair_norms = [], []
    for i in range(8):
        updates.append(any(not torch.equal(value, sim['model_state_dict'][f'member_point_heads.{i%4}.{name}'])
                           for name, value in model.member_point_heads[i].state_dict().items()))
    for i in range(4):
        pair_norms.append(sum(float((value-model.member_point_heads[i+4].state_dict()[name]).double().square().sum())
                              for name, value in model.member_point_heads[i].state_dict().items())**.5)
    assert all(updates) and all(value > 0 for value in pair_norms)
    del sim
    norm, _ = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    x = fit_input(args.real_dir, norm, torch)
    model.cuda().eval()
    with torch.inference_mode():
        original = model(x)
        folded = fold_model(model, torch).cuda().eval()
        prediction = folded(x)
        scale = norm.std_target[:2].view(1, 1, 1, 1, 2)
        error = float(((original-prediction)[..., :2]*scale).abs().max())
    assert error <= 2e-6
    state = {name: value.detach().cpu() for name, value in folded.state_dict().items()}
    fold = dict(kind='equal_linear_head_pairs', pairs=INTERVENTION['folding_pairs'], source_checkpoint_sha256=source_sha,
                source_ema=payload['ema'], physical_fit_input_max_error=error,
                heads_updated=updates, pair_difference_l2=pair_norms, held_read=False)
    folded_payload = {k: payload[k] for k in ('iteration', 'held_aoa', 'normalizer', 'normalizer_sha256', 'sample_stream_sha256', 'start')}
    folded_payload.update(schema='eight_head_affine_fold_point_v1', model_config=folded.architecture_config(),
                          model_state_dict=state, model_state_sha256=dense.dr1.model_state_sha256(state), fold=fold)
    directory = args.root/'folded4'
    directory.mkdir()
    checkpoint = directory/'affine_fold_u4000.pth'
    torch.save(folded_payload, checkpoint)
    receipt.update(schema='eight_head_affine_fold_receipt_v1', checkpoint_path=str(checkpoint),
                   checkpoint_sha256=held.sha256(checkpoint), fold=fold)
    (directory/'point_endpoint.json').write_text(json.dumps(receipt, indent=2))
    (args.root/'FOLD.json').write_text(json.dumps({**fold, 'checkpoint_sha256':receipt['checkpoint_sha256']}, indent=2))
    del model, folded, norm, payload, folded_payload, state
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    score(SimpleNamespace(root=directory, control_dir=args.control_dir, manifest=args.manifest, real_dir=args.real_dir), folded_eight=True)
    (args.root/'RESULT.json').write_bytes((directory/'RESULT.json').read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'control-dir', 'manifest', 'real-dir', 'sim'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
