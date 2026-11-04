"""Replace only the two three-element CUDA FP16 scans in the verified ZIP."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


HELPER = '''

def _short_gap_cumsum(values: torch.Tensor) -> torch.Tensor:
    # Match CUDA's Sklansky scan, including each FP16 rounding operation.
    # These three nonnegative gaps are the two halves of seven quantiles.
    if values.is_cuda and values.dtype == torch.float16 and values.shape[-1] == 3:
        first, second, third = values.unbind(dim=-1)
        pair01 = second + first
        return torch.stack((first, pair01, third + pair01), dim=-1)
    return torch.cumsum(values, dim=-1)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-cuda', action='store_true')
    args = parser.parse_args()
    base = Path(args.baseline)
    assert hashlib.sha256(base.read_bytes()).hexdigest() == '7b14d45c621621e532f219c8f98f123fcf550fa084ab93ef1ed2368eef69992e'
    rounding = None
    if args.check_cuda:
        import torch
        assert torch.cuda.is_available()
        torch.manual_seed(411308)
        namespace = {'torch': torch}
        exec(HELPER, namespace)
        scan = namespace['_short_gap_cumsum']
        values = torch.exp2(torch.rand(250000, 3, device='cuda') * 39 - 24).half()
        bits = torch.arange(0x7c00, dtype=torch.int16, device='cuda').view(torch.float16)
        patterns = torch.stack((bits, bits.flip(0), bits.roll(7919)), dim=-1)
        for data in (values, patterns, values.flip(-1)):
            assert torch.equal(scan(data), torch.cumsum(data, dim=-1))
        rounding = {'decision': 'PASS_EXACT_CUDA_FP16_SCAN3', 'seed': 411308,
                    'checked_rows': 2 * 250000 + len(bits), 'torch': torch.__version__}
    changed = 'models/ensemble_unet2d.py'
    with zipfile.ZipFile(base) as src, zipfile.ZipFile(args.output, 'w') as dst:
        for entry in src.infolist():
            data = src.read(entry.filename)
            if entry.filename == changed:
                text = data.decode('utf-8')
                marker = 'SPS_SIGMA = 0.0563870259\n'
                old_lower = 'torch.cumsum(\n            lower_gaps.flip(dims=(-1,)), dim=-1)'
                old_upper = 'torch.cumsum(upper_gaps, dim=-1)'
                assert text.count(marker) == text.count(old_lower) == text.count(old_upper) == 1
                text = text.replace(marker, marker + HELPER)
                text = text.replace(old_lower, '_short_gap_cumsum(\n            lower_gaps.flip(dims=(-1,)))')
                text = text.replace(old_upper, '_short_gap_cumsum(upper_gaps)')
                compile(text, changed, 'exec')
                data = text.encode('utf-8')
            dst.writestr(entry, data)
    members = {}
    with zipfile.ZipFile(base) as src, zipfile.ZipFile(args.output) as dst:
        assert dst.testzip() is None and dst.namelist() == src.namelist()
        for name in src.namelist():
            before, after = src.read(name), dst.read(name)
            assert (before != after) == (name == changed)
            members[name] = hashlib.sha256(after).hexdigest()
    result = {'zip_sha256': hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
              'only_changed_member': changed, 'member_sha256': members,
              'rounding_check': rounding, 'official_score': None}
    Path(args.output).with_suffix('.identity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
