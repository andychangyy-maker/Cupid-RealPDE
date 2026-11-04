"""Replace only UQ tensors inside the official 0908 archive."""
import argparse
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import torch
from build_track1_blended_ensemble_unet2d_package import compact_state_dict

BASELINE_SHA = 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'
POINT_SHA = 'dbe5d9514361a390a0523b330240f8bf2572fd51ef92a495d93e730f5abb4656'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    return a == b


def main(args):
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
    import train_b210_held_residual_uq as held
    _, _, native, _ = held.runtime()
    from models.ensemble_unet2d import EnsembleUNet2dConfig
    assert sha(args.baseline) == BASELINE_SHA
    receipt = json.loads(args.receipt.read_text())
    assert receipt['checkpoint_sha256'] == sha(args.checkpoint)
    assert receipt['point_receipt']['checkpoint_sha256'] == POINT_SHA
    assert receipt['compact_uq'] and receipt['point_state_bitwise'] and receipt['held_aoa'] is None
    assert receipt['updates'] == 3200 and receipt['seed'] == 18 and receipt['batch_size'] == 8
    patch = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    with zipfile.ZipFile(args.baseline) as archive:
        assert archive.testzip() is None
        infos = archive.infolist()
        members = {i.filename: archive.read(i) for i in infos}
    assert len(members) == 5 and 'ensemble_unet2d.pth' in members
    original = torch.load(io.BytesIO(members['ensemble_unet2d.pth']), map_location='cpu', weights_only=False)
    assert original['storage_format'] == 'symmetric_per_dim0_int8_v1' and original['inference_dtype'] == 'float16'
    def config(value):
        value = dict(value)
        assert value.pop('architecture') == 'ensemble_unet'
        return dataclasses.asdict(EnsembleUNet2dConfig(**value))
    assert config(patch['model_config']) == config(original['model_config'])
    assert equal(patch['normalizer'], original['normalizer'])
    state = original['quantized_state_dict']
    expected = {k for k in state if k.startswith(native.UQ_PREFIXES)}
    assert expected == set(patch['model_state_dict'])
    replacement, stats = compact_state_dict(patch['model_state_dict'], 1024)
    for key, value in replacement.items():
        old = state[key]
        assert type(value) is type(old)
        parts = value.items() if isinstance(value, dict) else [(None, value)]
        for subkey, tensor in parts:
            reference = old[subkey] if subkey is not None else old
            assert tensor.shape == reference.shape and tensor.dtype == reference.dtype and torch.isfinite(tensor).all()
    result = {**original, 'quantized_state_dict': {**state, **replacement}}
    buffer = io.BytesIO()
    torch.save(result, buffer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x') as archive:
        for info in infos:
            archive.writestr(info, buffer.getvalue() if info.filename == 'ensemble_unet2d.pth' else members[info.filename])
    assert args.output.stat().st_size < 256 * 1024**2
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None and set(archive.namelist()) == set(members)
        for name, data in members.items():
            if name != 'ensemble_unet2d.pth':
                assert archive.read(name) == data
        saved = torch.load(io.BytesIO(archive.read('ensemble_unet2d.pth')), map_location='cpu', weights_only=False)
        member_shas = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    assert equal(saved, result)
    assert all(equal(value, saved['quantized_state_dict'][key]) for key, value in state.items() if key not in expected)
    changed = [key for key in expected if not equal(state[key], replacement[key])]
    assert changed
    report = {'decision': 'PACKAGED_NEEDS_DEPLOYMENT_QUALITY_AND_COST_CHECKS', 'zip_sha256': sha(args.output),
              'zip_bytes': args.output.stat().st_size, 'baseline_sha256': BASELINE_SHA,
              'uq_checkpoint_sha256': sha(args.checkpoint), 'uq_receipt_sha256': sha(args.receipt),
              'runtime_members_byte_identical': True, 'non_uq_packed_state_bitwise': True,
              'normalizer_and_config_identical': True, 'uq_changed_tensors': len(changed),
              'member_sha256': member_shas, 'quantization': stats, 'packer_sha256': sha(__file__)}
    args.output.with_suffix('.package.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'checkpoint', 'receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
