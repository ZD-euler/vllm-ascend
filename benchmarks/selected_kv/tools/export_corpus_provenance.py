#!/usr/bin/env python3
"""Export allowlisted corpus metadata and file hashes, never token/text contents."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def clean_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ('https', 'http', 'git', 'ssh'):
        return None
    return urlunsplit((parsed.scheme, parsed.hostname or '', parsed.path, '', ''))


def scalar_fields(value, names):
    return {key: value[key] for key in names if key in value and
            isinstance(value[key], (str, int, bool, float, type(None)))}


def export(manifest_path, token_root):
    data = json.loads(manifest_path.read_text())
    result = {'schema_version': 1, 'manifest_sha256': sha(manifest_path),
              'synthetic_padding_tokens': data.get('synthetic_padding_tokens'),
              'sources': [], 'cases': []}
    for source in data['sources']:
        row = scalar_fields(source, ('task_name', 'glm52_tokens', 'tokens_sha256_uint32_le',
                                     'commit', 'revision', 'repo_commit', 'tokenizer_sha256',
                                     'agent', 'canonical_file', 'canonical_sha256', 'sha256',
                                     'solved', 'traj_id'))
        for key in ('url', 'repo_url', 'source_url', 'repository', 'download_url'):
            if isinstance(source.get(key), str):
                row[key] = clean_url(source[key])
        path = Path(source['token_file'])
        if not path.is_absolute():
            path = token_root / path
        row['token_file_name'] = path.name
        row['token_file_sha256'] = sha(path)
        row['token_file_bytes'] = path.stat().st_size
        result['sources'].append(row)
    for case in data['cases']:
        row = scalar_fields(case, ('case', 'history', 'query', 'total_query_tokens', 'requests'))
        row['windows'] = [scalar_fields(window, ('task', 'start_token', 'token_sha256',
                                                'end_token', 'stop_token', 'length')) for window in case['windows']]
        result['cases'].append(row)
    # Top-level provenance schema differs across builders; export explicit known fields only.
    result['provenance'] = scalar_fields(data, ('tokenizer_revision', 'tokenizer_sha256',
                                               'builder_commit', 'source_commit'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--token-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export(args.manifest, args.token_root), indent=2) + '\n')


if __name__ == '__main__':
    main()
