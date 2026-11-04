"""Check that submission/ exactly matches the frozen Final source manifest."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'docs/frozen-source-manifest.json').read_text(encoding='utf-8'))
    expected = {item['path']: item for item in manifest['files']}
    package = root / 'submission'
    actual = {p.relative_to(package).as_posix(): p for p in package.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    failures = [f'missing: {name}' for name in sorted(expected.keys() - actual.keys())]
    failures += [f'unexpected: {name}' for name in sorted(actual.keys() - expected.keys())]
    for name in sorted(expected.keys() & actual.keys()):
        data = actual[name].read_bytes()
        item = expected[name]
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            failures.append(f'changed: {name}')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f"PASS: {len(expected)} frozen files, {manifest['bytes']:,} bytes; every SHA256 matches.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
