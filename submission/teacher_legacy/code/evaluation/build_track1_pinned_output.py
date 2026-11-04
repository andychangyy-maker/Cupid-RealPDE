"""One output-transfer experiment on the sealed 918343 ZIP; weights unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

BASE_SHA = 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'


def transform(source):
    old = '''            bundle = torch.stack(
                (prediction, lower, upper)).float().cpu().numpy()
            predictions.append(bundle[0])
            lower_bounds.append(bundle[1])
            upper_bounds.append(bundle[2])'''
    new = '''            bundle = torch.stack((prediction, lower, upper)).float()
            if state["device"].type == "cuda":
                host = torch.empty(bundle.shape, dtype=torch.float32,
                                   device="cpu", pin_memory=True)
                host.copy_(bundle, non_blocking=True)
                pending.append(host)
            else:
                pending.append(bundle.cpu())

    if state["device"].type == "cuda":
        torch.cuda.current_stream(state["device"]).synchronize()
    for host in pending:
        bundle = host.numpy()
        predictions.append(bundle[0])
        lower_bounds.append(bundle[1])
        upper_bounds.append(bundle[2])'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    assert source.count('    upper_bounds = []') == 1
    return source.replace('    upper_bounds = []', '    upper_bounds = []\n    pending = []')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert hashlib.sha256(a.base.read_bytes()).hexdigest() == BASE_SHA
    assert not a.output.exists()
    with zipfile.ZipFile(a.base) as src, zipfile.ZipFile(a.output, 'w', zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == 'submission.py':
                data = transform(data.decode()).encode()
                compile(data, 'submission.py', 'exec')
            dst.writestr(info, data)
    print(json.dumps({'base_sha256': BASE_SHA, 'candidate_sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
                      'change': 'FP32 pinned output transfer; synchronize before host reads; no global output alias'}))


if __name__ == '__main__':
    main()
