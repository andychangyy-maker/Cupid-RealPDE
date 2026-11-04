"""Frozen C: measure dropout-induced energy shift using only Real fit prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/training'))
import train_b210_held_residual_uq as held


def main(args):
    started = time.monotonic()
    torch, _, _, prefix = held.runtime()
    import h5py
    import numpy as np
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from models.unet2d import ResidualConvBlock

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(41109)
    point_sha = held.sha256(args.point)
    assert point_sha == 'ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa'
    assert held.sha256(args.manifest) == held.MANIFEST_SHA
    payload = torch.load(args.point, map_location='cpu', weights_only=False)
    assert payload['normalizer_sha256'] == '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
    config = dict(payload['model_config'])
    assert config.pop('architecture') == 'ensemble_unet'
    config = EnsembleUNet2dConfig(**config)
    assert config.backbone_residual_dropout == .2
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    norm, _ = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    model.cuda().eval()
    del payload
    blocks = [m for m in model.modules() if isinstance(m, ResidualConvBlock) and m.residual_dropout > 0]
    assert blocks and not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    rows = [r for r in ev.clean_rows(args.manifest) if r['aoa'] != 15]
    assert len(rows) == 64
    plan = [dict(file=r['file'], start=int(start)) for r in rows for start in (r['train_starts'][0], r['train_starts'][-1])]
    assert len(plan) == 128 and len({(p['file'], p['start']) for p in plan}) == 128
    records, opened = [], set()
    hashes = {k: hashlib.sha256() for k in ('input', 'truth')}

    def energy(uv):
        fluctuation = uv.double() - uv.double().mean(dim=1, keepdim=True)
        return .5 * fluctuation.square().mean(dim=1).sum(dim=-1)

    def physical(x):
        p = model(x)
        zeros = torch.zeros_like(p[..., :2])
        return held.absolute_distribution(p, zeros, zeros, norm)[0][..., :2]

    with torch.inference_mode():
        for offset in range(0, len(plan), 8):
            histories, truths = [], []
            batch_plan = plan[offset:offset + 8]
            for item in batch_plan:
                with h5py.File(args.real_dir / item['file'], 'r') as data:
                    uv = np.stack([np.asarray(data[k][item['start']:item['start'] + 40, ::2, ::2], dtype=np.float32) for k in ('u', 'v')], axis=-1)
                assert uv.shape == (40, 32, 64, 2) and np.isfinite(uv).all()
                full = np.concatenate([uv, np.zeros_like(uv[..., :1])], axis=-1)
                histories.append(full[:20]); truths.append(full[20:]); opened.add(item['file'])
            history, truth = np.stack(histories), np.stack(truths)
            for key, value in (('input', history), ('truth', truth)): hashes[key].update(value.tobytes())
            y = torch.from_numpy(truth).cuda()
            x, _ = norm.preprocess(torch.from_numpy(history).cuda(), y)
            for block in blocks: block.training = False
            kd, kt = energy(physical(x)), energy(y[..., :2])
            for block in blocks: block.training = True
            km = torch.stack([energy(physical(x)) for _ in range(8)]).mean(dim=0)
            for block in blocks: block.training = False
            base_error = (kd - kt).square().flatten(1).sum(dim=1)
            mc_error = (km - kt).square().flatten(1).sum(dim=1)
            truth_energy = kt.square().flatten(1).sum(dim=1).clamp_min(1e-30)
            explained = (base_error - mc_error) / base_error.clamp_min(1e-30)
            relative_shift = (km - kd).square().flatten(1).sum(dim=1).sqrt() / truth_energy.sqrt()
            values = torch.stack([explained, (base_error / truth_energy).sqrt(), (mc_error / truth_energy).sqrt(), relative_shift], dim=-1)
            assert torch.isfinite(values).all()
            for item, values_row in zip(batch_plan, values.cpu().tolist()):
                records.append({**item, **dict(zip(('explained_squared_energy_error', 'deterministic_tke_error', 'mean_dropout_energy_tke_error', 'energy_shift_relative_to_truth'), values_row))})
            if time.monotonic() - started > 120: raise TimeoutError('dropout diagnostic budget')
    assert opened == {r['file'] for r in rows}
    per_file = {r['file']: float(np.mean([v['explained_squared_energy_error'] for v in records if v['file'] == r['file']])) for r in rows}
    explained = float(np.mean(list(per_file.values())))
    positive = sum(v > 0 for v in per_file.values())
    det_score = 100 / (1 + .5 * np.mean([r['deterministic_tke_error'] for r in records]))
    mc_score = 100 / (1 + .5 * np.mean([r['mean_dropout_energy_tke_error'] for r in records]))
    passed = explained >= .20 and positive / 64 >= .70 and mc_score - det_score >= .20
    result = dict(decision=('PASS_' if passed else 'STOP_') + 'DROPOUT_ENERGY_GAP',
                  mean_explained_squared_energy_error=explained, positive_trajectories=positive,
                  trajectory_count=64, deterministic_tke_score=float(det_score), mean_dropout_energy_tke_score=float(mc_score),
                  tke_score_difference=float(mc_score - det_score), per_trajectory=per_file, samples=records,
                  checkpoint_sha256=point_sha, source_sha256=held.sha256(__file__), manifest_sha256=held.MANIFEST_SHA,
                  hashes={k: v.hexdigest() for k, v in hashes.items()}, dropout_blocks=len(blocks),
                  elapsed_seconds=time.monotonic() - started,
                  scope='128 fit-prefix windows only; eight stochastic passes; mean energy map is a mechanism counterfactual, not a submitted point prediction or held-score gain.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('decision', 'mean_explained_squared_energy_error', 'positive_trajectories', 'tke_score_difference', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('point', 'manifest', 'real-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
