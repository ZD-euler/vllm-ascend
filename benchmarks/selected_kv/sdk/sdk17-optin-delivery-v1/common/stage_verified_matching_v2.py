"""Fresh MMC-only stage from the complete qualified matching-pair archive.

No blanket 'output' basename exclusion: nlohmann's detail/output headers are
real sources. The old source, active candidate and existing MF binaries stay
untouched. This is a build-input integrity repair, not an allocator fix.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA = 'a93b02bb537674fe2a672f098edb2d9ff05301bc7eac1555d92bd46beab23dca'
MANIFEST_SHA = '5f634c2ef0e6c4c6ed2941f4fa3ee815481b92d5395f7465a9415ca4062fc08a'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stage(archive, manifest_path, output, addon=HERE):
    if output.exists():
        raise ValueError('fresh independent output directory required')
    if digest(archive.read_bytes()) != ARCHIVE_SHA or digest(manifest_path.read_bytes()) != MANIFEST_SHA:
        raise ValueError('qualified matching source archive/manifest differs')
    manifest = json.loads(manifest_path.read_text())
    expected = {name[4:]:sha for name,sha in manifest['files'].items() if name.startswith('mmc/')}
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive) as handle:
        for item in handle:
            if not item.name.startswith('mmc/'):
                continue
            relative = PurePosixPath(item.name[4:])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe archive member')
            target = output/str(relative)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not item.isfile() or str(relative) not in expected or str(relative) in seen:
                raise ValueError('unexpected or duplicated MMC archive member')
            data = handle.extractfile(item).read()
            if digest(data) != expected[str(relative)]:
                raise ValueError('full matching source file hash differs')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data); target.chmod(item.mode & 0o777)
            seen.add(str(relative))
    if seen != set(expected):
        raise ValueError('qualified source archive is incomplete')
    production = json.loads((addon/'manifest.json').read_text())
    overlays = [(addon/'overlay', production['production_files'])]
    overlays.append((addon/'build-tooling/overlay', json.loads((addon/'build-tooling/manifest.json').read_text())))
    changed = []
    for directory, entries in overlays:
        for name, hashes in entries.items():
            target = output/name; replacement = (directory/name).read_bytes()
            if digest(target.read_bytes()) != hashes['base_sha256'] or digest(replacement) != hashes['candidate_sha256']:
                raise ValueError('addon/build-tooling patch was not based on the qualified archive')
            target.write_bytes(replacement); expected[name] = digest(replacement); changed.append(name)
    for name, sha in expected.items():
        if digest((output/name).read_bytes()) != sha:
            raise ValueError('final whole-tree verification failed')
    (output/'dsa-sdk-addon-manifest.json').write_text(json.dumps(production,indent=2)+'\n')
    proof = dict(schema='verified_matching_MMC_stage_v2', archive_sha256=ARCHIVE_SHA,
        source_manifest_sha256=MANIFEST_SHA, complete_baseline_files=len(seen),
        intentional_changed_files=changed, final_source_files=expected, MF_binaries_modified=False,
        SDK_build_performed=False, NPU_initialized=False)
    (output/'dsa-matching-stage-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(status='full_matching_source_staged',files=len(seen),intentional_deltas=len(changed),
        missing_files=0,unqualified_runtime_deltas=0,MF_modified=False,NPU_initialized=False)))
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--addon-dir',type=Path,default=HERE)
    args=parser.parse_args()
    stage(args.archive.resolve(strict=True),args.source_manifest.resolve(strict=True),args.output.resolve(),args.addon_dir.resolve(strict=True))
