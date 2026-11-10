#!/usr/bin/env python3
"""Verify bundled archive SHA256 and unpack the pinned core sources into a fresh root."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archives = HERE / 'artifacts/source'
    expected = json.loads((archives / 'REVISIONS.json').read_text())['archives']
    roles = {'core-native-runtime-source.tar.gz': 'core-native',
             'core-native-install-source.tar.gz': 'core-native'}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, role in roles.items():
        archive = archives / name
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected[name]:
            raise ValueError('archive identity mismatch: ' + name)
        target = args.output / role
        target.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = Path(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('unsafe source archive member: ' + member.name)
            tar.extractall(target)
    print('Unpacked verified pinned sources; no build or model launch.')


if __name__ == '__main__':
    main()
