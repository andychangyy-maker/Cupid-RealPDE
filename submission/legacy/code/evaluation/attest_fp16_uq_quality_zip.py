"""Attest every changed tensor of the fixed full-data FP16-UQ ZIP."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

for directory in ('submissions', 'training'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / directory))
from package_track1_combined_uq_quality import sha, equal, BASELINE_SHA, POINT_SHA
import train_b210_held_residual_uq as held


def main(args):
    torch, _, native, _ = held.runtime()
    torch.set_num_threads(4)
    assert sha(args.baseline) == BASELINE_SHA
    receipt = json.loads(args.receipt.read_text())
    assert receipt['checkpoint_sha256'] == sha(args.checkpoint)
    assert receipt['point_receipt']['checkpoint_sha256'] == POINT_SHA
    assert receipt['compact_uq'] and receipt['point_state_bitwise'] and receipt['held_aoa'] is None
    patch = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    payloads, members = {}, {}
    for label, path in [('baseline', args.baseline), ('candidate', args.candidate)]:
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None and len(archive.namelist()) == 5
            members[label] = {name: archive.read(name) for name in archive.namelist()}
        payloads[label] = torch.load(io.BytesIO(members[label]['ensemble_unet2d.pth']), map_location='cpu', weights_only=False)
    assert members['baseline'].keys() == members['candidate'].keys()
    assert all(members['baseline'][name] == data for name, data in members['candidate'].items() if name != 'ensemble_unet2d.pth')
    base, candidate = payloads['baseline'], payloads['candidate']
    assert equal({k: v for k, v in base.items() if k != 'quantized_state_dict'},
                 {k: v for k, v in candidate.items() if k != 'quantized_state_dict'})
    b, c = base['quantized_state_dict'], candidate['quantized_state_dict']
    assert b.keys() == c.keys()
    uq_keys = {k for k in b if k.startswith(native.UQ_PREFIXES)}
    assert uq_keys == set(patch['model_state_dict'])
    assert all(equal(v, c[k]) for k, v in b.items() if k not in uq_keys)
    assert all(equal(c[k], patch['model_state_dict'][k].half()) for k in uq_keys)
    assert equal(base['normalizer'], patch['normalizer'])
    report = {'decision': 'FP16_UQ_IDENTITY_PASS_NEEDS_RELEASE_CHECKS', 'zip_sha256': sha(args.candidate),
              'zip_bytes': args.candidate.stat().st_size, 'baseline_sha256': BASELINE_SHA,
              'uq_checkpoint_sha256': sha(args.checkpoint), 'uq_receipt_sha256': sha(args.receipt),
              'runtime_members_byte_identical': True, 'non_uq_packed_state_bitwise': True,
              'normalizer_and_config_identical': True, 'uq_tensors': len(uq_keys), 'uq_storage': 'float16',
              'point_storage': 'original per-dim0 int8; CUDA computation float16', 'verifier_sha256': sha(__file__)}
    args.candidate.with_suffix('.package.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'checkpoint', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
