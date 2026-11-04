"""Avoid final NumPy concatenation while preserving GPU FP32 output conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

BASE_SHA = 'fe80d3d2c5b24bc9c362839be4150524e9df07e0214f5fec3a11bc56f18d4ffa'
p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert hashlib.sha256(a.base.read_bytes()).hexdigest() == BASE_SHA
assert not a.output.exists()
old = '''                host = torch.empty(bundle.shape, dtype=torch.float32,
                                   device="cpu", pin_memory=True)
                host.copy_(bundle, non_blocking=True)
                pending.append(host)'''
new = '''                if final_host is None:
                    final_host = [torch.empty(
                        (len(raw_input), *bundle.shape[2:]), dtype=torch.float32,
                        device="cpu", pin_memory=True) for _ in range(3)]
                for index, host in enumerate(final_host):
                    host[start:start + len(raw)].copy_(
                        bundle[index], non_blocking=True)'''
with zipfile.ZipFile(a.base) as src, zipfile.ZipFile(a.output, 'w') as dst:
    for info in src.infolist():
        data = src.read(info.filename)
        if info.filename == 'submission.py':
            text = data.decode()
            assert text.count(old) == 1
            text = text.replace(old, new).replace('    pending = []', '    pending = []\n    final_host = None')
            marker = '        torch.cuda.current_stream(state["device"]).synchronize()'
            assert text.count(marker) == 1
            text = text.replace(marker, marker + '''
        if final_host is not None:
            return dict(zip(("prediction", "lower", "upper"),
                            (host.numpy() for host in final_host)))''')
            compile(text, 'submission.py', 'exec')
            data = text.encode()
        dst.writestr(info, data)
print(json.dumps({'base_sha256': BASE_SHA, 'candidate_sha256': hashlib.sha256(a.output.read_bytes()).hexdigest()}))
