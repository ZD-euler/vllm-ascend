#!/usr/bin/env python3
"""Create and verify a relocatable SHA256 lock for a public reproduction tree."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

FORBIDDEN_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.key', '.pem'}
FORBIDDEN_NAMES = {'.env', 'id_rsa', 'id_ed25519', 'credentials'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_file(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('manifest path must stay inside bundle')
    path = root.joinpath(*relative.parts)
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
        raise ValueError('symlinks are not accepted in bundle')
    path.resolve().relative_to(root.resolve())
    return path


def create(root, output):
    root = root.resolve()
    records = []
    for path in sorted(root.rglob('*')):
        if '.git' in path.relative_to(root).parts or path.resolve() == output.resolve():
            continue
        if path.is_symlink():
            raise ValueError('symlinks are not accepted in bundle')
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError('excluded sensitive or model-weight file: ' + path.name)
        records.append({'path': path.relative_to(root).as_posix(),
                        'bytes': path.stat().st_size, 'sha256': sha256(path)})
    return {'schema_version': 1, 'algorithm': 'sha256', 'files': records}


def verify(root, data):
    if data.get('schema_version') != 1 or data.get('algorithm') != 'sha256':
        raise ValueError('unsupported manifest schema')
    seen = set()
    for item in data['files']:
        if item['path'] in seen:
            raise ValueError('duplicate manifest path')
        seen.add(item['path'])
        path = safe_file(root.resolve(), item['path'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('manifest mismatch: ' + item['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('create', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'create':
        args.manifest.write_text(json.dumps(create(args.root, args.manifest), indent=2) + '\n')
    else:
        verify(args.root, json.loads(args.manifest.read_text()))
        print('Manifest verified')


if __name__ == '__main__':
    main()
