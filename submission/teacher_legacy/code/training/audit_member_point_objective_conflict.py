"""Training-prefix diagnostic of member and deployed-mean objective directions."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import train_b210_held_residual_uq as held


def main(output):
    begun = time.monotonic()
    torch, dense, _native, prefix = held.runtime()
    import h5py
    import numpy as np
    import evaluate_b210_transfer as ev
    import train_c208_npt1_member_sync_tke_loao as npt1
    from models.ensemble_unet2d import EnsembleUNet2d

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    manifest = Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
    assert held.sha256(manifest) == held.MANIFEST_SHA
    rows = [r for r in ev.clean_rows(manifest) if r['aoa'] != 15]
    checkpoint = Path('/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held15/control/point/ema999_u4000_primary.pth')
    expected = '8711cbe9e279a806dc29be9ab423672f77e498035929855d14a5fac877eb7a77'
    assert held.sha256(checkpoint) == expected
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config']))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    normalizer, normalizer_report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    model.cuda().eval()
    del payload
    selected = [(name, value) for name, value in model.named_parameters()
                if name.startswith(('up.', 'decoder.', 'decoder_regime_film.',
                                    'mean_profile_refiner.', 'member_point_heads.',
                                    'energy_controller.'))]
    assert selected
    scopes = {'decoder': [], 'point_heads': []}
    for index, (name, _value) in enumerate(selected):
        scope = 'point_heads' if name.startswith(('member_point_heads.', 'energy_controller.')) else 'decoder'
        scopes[scope].append(index)
    assert all(scopes.values())
    parameters = [value for _name, value in selected]
    rng = np.random.default_rng(34010)
    plan, rounds = [], []
    sums = {key: {'dot': 0.0, 'point_norm2': 0.0, 'member_norm2': 0.0} for key in scopes}
    for iteration in range(8):
        batch_plan, histories, truths = [], [], []
        for angle in (0, 5, 10, 20):
            group = [r for r in rows if r['aoa'] == angle]
            for _ in range(2):
                row = group[int(rng.integers(len(group)))]
                start = int(rng.choice(row['train_starts']))
                batch_plan.append({'file': row['file'], 'start': start})
                with h5py.File(Path('/root/autodl-tmp/realpde/data/real') / row['file'], 'r') as data:
                    uv = np.stack([np.asarray(data[k][start:start + 40, ::2, ::2], dtype=np.float32)
                                   for k in ('u', 'v')], axis=-1)
                assert uv.shape[0] == 40
                full = np.concatenate([uv, np.zeros_like(uv[..., :1])], axis=-1)
                histories.append(full[:20]); truths.append(full[20:])
        plan.append(batch_plan)
        history = torch.from_numpy(np.stack(histories)).cuda()
        truth = torch.from_numpy(np.stack(truths)).cuda()
        x, y = normalizer.preprocess(history, truth)
        prediction, members = model.forward_uq_training(x)[:2]
        loss_args = npt1.joint_real.real_loss_args()
        weights = npt1.joint_real.REAL_ACCURACY_WEIGHTS
        point_loss = npt1.ensemble_train.point_loss(loss_args, prediction, y, normalizer, weights, 4000)[0]
        member_loss = torch.stack([npt1.ensemble_train.point_loss(
            loss_args, members[:, member], y, normalizer, weights, 4000)[0]
            for member in range(members.shape[1])]).mean()
        gp = torch.autograd.grad(point_loss, parameters, retain_graph=True, allow_unused=True)
        gm = torch.autograd.grad(member_loss, parameters, allow_unused=True)
        measurements = {}
        for scope, indices in scopes.items():
            dot = pn = mn = 0.0
            for index in indices:
                a, b = gp[index], gm[index]
                if a is not None: pn += float(a.square().sum())
                if b is not None: mn += float(b.square().sum())
                if a is not None and b is not None: dot += float((a * b).sum())
            assert pn > 0 and mn > 0 and np.isfinite([dot, pn, mn]).all()
            cosine = dot / (pn * mn) ** .5
            measurements[scope] = {'cosine': cosine, 'weighted_member_to_point_norm': 4 * (mn / pn) ** .5}
            for key, value in [('dot', dot), ('point_norm2', pn), ('member_norm2', mn)]: sums[scope][key] += value
        rounds.append(measurements)
        del gp, gm, point_loss, member_loss, prediction, members, history, truth, x, y
        gc.collect()
        if time.monotonic() - begun > 150: raise TimeoutError('member objective diagnostic budget')
    summary = {}
    for scope, values in sums.items():
        cosine = values['dot'] / (values['point_norm2'] * values['member_norm2']) ** .5
        negative = sum(row[scope]['cosine'] < 0 for row in rounds)
        summary[scope] = {'energy_weighted_cosine': cosine, 'negative_rounds': negative}
    passed = any(row['energy_weighted_cosine'] <= -.05 and row['negative_rounds'] >= 4 for row in summary.values())
    result = {'decision': 'MEMBER_POINT_CONFLICT_HEADROOM' if passed else 'STOP_MEMBER_POINT_CONFLICT_HYPOTHESIS',
              'checkpoint_sha256': expected, 'normalizer_sha256': normalizer_report['sha256'],
              'manifest_sha256': held.MANIFEST_SHA, 'sample_plan': plan,
              'sample_plan_sha256': hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
              'scope': 'training prefix only; deterministic all-members diagnostic; not full bootstrap training replay or score evidence',
              'parameter_counts': {key: sum(parameters[i].numel() for i in indices) for key, indices in scopes.items()},
              'rounds': rounds, 'summary': summary, 'seconds': time.monotonic() - begun}
    held.write_json(output, result)
    print(json.dumps({key: result[key] for key in ('decision', 'summary', 'seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
