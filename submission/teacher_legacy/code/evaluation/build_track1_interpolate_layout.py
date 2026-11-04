"""Local NHWC bilinear kernel experiment on the sealed pinned-output ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

BASE_SHA = 'fe80d3d2c5b24bc9c362839be4150524e9df07e0214f5fec3a11bc56f18d4ffa'
HELPER = '''

def _layout_interpolate(x, *, size, mode, align_corners):
    if (x.is_cuda and x.dtype == torch.float16 and x.ndim == 4
            and x.shape[1] >= 4 and mode == "bilinear" and not align_corners):
        return F.interpolate(x.contiguous(memory_format=torch.channels_last),
                             size=size, mode=mode,
                             align_corners=align_corners).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
'''

p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert hashlib.sha256(a.base.read_bytes()).hexdigest() == BASE_SHA
assert not a.output.exists()
changes = {}
with zipfile.ZipFile(a.base) as src, zipfile.ZipFile(a.output, 'w') as dst:
    for info in src.infolist():
        data = src.read(info.filename)
        if info.filename in ('models/unet2d.py', 'models/ensemble_unet2d.py'):
            source = data.decode()
            changes[info.filename] = source.count('F.interpolate(')
            assert changes[info.filename] == (2 if info.filename.endswith('/unet2d.py') else 10)
            source = source.replace('F.interpolate(', '_layout_interpolate(')
            marker = 'from torch.nn import functional as F'
            assert source.count(marker) == 1
            source = source.replace(marker, marker + HELPER)
            compile(source, info.filename, 'exec')
            data = source.encode()
        dst.writestr(info, data)
print(json.dumps({'base_sha256': BASE_SHA, 'candidate_sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
                  'replaced_calls': changes}))
