#!/usr/bin/env python3
"""Download only the recorded public corpus archives and verify original SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-hf', action='store_true', help='Use huggingface.co with the exact same recorded path')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(HERE):
        raise ValueError('public archive output must stay outside the code bundle')
    args.output.mkdir(parents=True, exist_ok=True)
    for row in json.loads((HERE / 'references/public-artifacts.json').read_text()):
        name = row['local_file']
        if Path(name).name != name:
            raise ValueError('unsafe artifact filename')
        target = args.output / name
        if not target.exists():
            url = row['download_url']
            if args.official_hf:
                url = url.replace('https://hf-mirror.com/', 'https://huggingface.co/', 1)
            temporary = target.with_suffix(target.suffix + '.part')
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--connect-timeout', '15', '--max-time', '120',
                            '--max-filesize', str(64 * 1024 * 1024), url,
                            '--output', str(temporary)], check=True, timeout=130)
            if sha(temporary) != row['sha256']:
                raise ValueError('download archive hash mismatch: ' + name)
            temporary.replace(target)
        if sha(target) != row['sha256']:
            raise ValueError('cached archive hash mismatch: ' + name)
    print('All recorded public archives verified; no extraction or tokenization performed.')


if __name__ == '__main__':
    main()
