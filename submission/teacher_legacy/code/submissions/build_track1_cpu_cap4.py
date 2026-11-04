"""Limit excessive CPU threads in exact 918343; preserve an already limited runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

BASE_SHA='b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'


def build(base, output):
    assert hashlib.sha256(base.read_bytes()).hexdigest()==BASE_SHA
    with zipfile.ZipFile(base) as old,zipfile.ZipFile(output,'x') as new:
        for item in old.infolist():
            data=old.read(item.filename)
            if item.filename=='submission.py':
                needle=b'import torch  # noqa: E402\n'
                assert data.count(needle)==1
                data=data.replace(needle,needle+b'\nif torch.get_num_threads() > 4:\n    torch.set_num_threads(4)\n',1)
            new.writestr(item,data)
    with zipfile.ZipFile(base) as old,zipfile.ZipFile(output) as new:
        assert old.namelist()==new.namelist() and new.testzip() is None
        assert all(old.read(n)==new.read(n) for n in old.namelist() if n!='submission.py')
        members={n:hashlib.sha256(new.read(n)).hexdigest() for n in new.namelist()}
    return {'zip':str(output.resolve()),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'bytes':output.stat().st_size,'members':members,'base_sha256':BASE_SHA,
            'only_change':'if current PyTorch intra-op threads > 4, cap at 4'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.base,a.output),indent=2))
