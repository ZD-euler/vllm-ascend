#!/usr/bin/env python3
"""Download hash-pinned public artifacts, rebuild canonical text and token windows.

Downloads and tokenizer execution occur only when explicitly invoked. Outputs
remain in the caller's new data directory and are never part of the code bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, help='Offline public archives, named by recorded local_file')
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text())
    metadata = []
    for name in ('canonical_manifest.json', 'canonical_solved_manifest.json'):
        for row in json.loads((HERE.parent / 'corpus/originals' / name).read_text()):
            metadata.append(row)
    rows = []
    for source in provenance['sources']:
        required_identity = ('sha256', 'canonical_sha256', 'traj_id')
        if any(key not in source for key in required_identity):
            raise ValueError('source artifact/canonical/trajectory provenance required')
        matches = [row for row in metadata if row['task_name'] == source['task_name'] and
                   all(row.get(key) == source[key] for key in required_identity)]
        unique = {json.dumps(row, sort_keys=True): row for row in matches}
        if len(unique) != 1:
            raise ValueError('canonical source is missing or ambiguous: ' + source['task_name'])
        row = next(iter(unique.values()))
        if not row.get('sha256') or not row.get('canonical_sha256'):
            raise ValueError('artifact/canonical hashes required')
        if urlsplit(row['download_url']).scheme != 'https':
            raise ValueError('HTTPS artifact URL required')
        for key in ('local_file', 'canonical_file'):
            if Path(row[key]).name != row[key]:
                raise ValueError('metadata output filename must be a basename')
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        target = args.output / row['local_file']
        if args.artifact_dir is not None:
            shutil.copyfile(args.artifact_dir.resolve(strict=True) / row['local_file'], target)
        else:
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--connect-timeout', '15', '--max-time', '120',
                        '--max-filesize', str(64 * 1024 * 1024), row['download_url'],
                            '--output', str(target)], check=True, timeout=130)
        if sha(target) != row['sha256']:
            raise ValueError('download hash mismatch: ' + row['task_name'])
    (args.output / 'downloaded_manifest.json').write_text(json.dumps(rows, indent=2))
    extractor = args.output / 'extract_canonical.py'
    shutil.copyfile(HERE.parent / 'corpus/originals/extract_canonical.py', extractor)
    subprocess.run([sys.executable, str(extractor)], check=True, timeout=300)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve(strict=True)), local_files_only=True)
    rebuilt = []
    for source, row in zip(provenance['sources'], rows):
        canonical = args.output / row['canonical_file']
        if sha(canonical) != row['canonical_sha256']:
            raise ValueError('canonical hash mismatch: ' + row['task_name'])
        tokens = tokenizer.encode(canonical.read_text(), add_special_tokens=False)
        digest = hashlib.sha256(struct.pack(f'<{len(tokens)}I', *tokens)).hexdigest()
        if len(tokens) != source['glm52_tokens'] or digest != source['tokens_sha256_uint32_le']:
            raise ValueError('tokenizer/token hash mismatch: ' + row['task_name'])
        token_file = source.get('token_file_name', Path(source['token_file']).name) if 'token_file' in source else source['token_file_name']
        if Path(token_file).name != token_file:
            raise ValueError('token output filename must be a basename')
        (args.output / token_file).write_text(json.dumps(tokens))
        if source.get('token_file_sha256') and sha(args.output / token_file) != source['token_file_sha256']:
            raise ValueError('serialized token file hash mismatch: ' + row['task_name'])
        rebuilt.append(dict(row, token_file=token_file, glm52_tokens=len(tokens),
                            tokens_sha256_uint32_le=digest))
    result = dict(synthetic_padding_tokens=False, sources=rebuilt, cases=provenance['cases'])
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Rebuilt hash-verified corpus; no model launched.')


if __name__ == '__main__':
    main()
