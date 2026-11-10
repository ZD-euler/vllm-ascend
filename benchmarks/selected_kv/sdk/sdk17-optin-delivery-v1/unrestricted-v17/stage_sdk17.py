"""Stage complete matching MMC + AddInfo + build flags + opt-in DDR patches."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(archive, source_manifest, output, addon):
    spec=importlib.util.spec_from_file_location('_strict_matching_stage',addon/'stage_verified_matching_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    proof=module.stage(archive,source_manifest,output,addon)
    patch=json.loads((HERE/'manifest.json').read_text())
    for name, hashes in patch['files'].items():
        target=output/name;replacement=HERE/'overlay'/name
        if digest(target)!=hashes['base_sha256'] or digest(replacement)!=hashes['candidate_sha256']:
            raise ValueError('opt-in patch base/candidate differs')
        target.write_bytes(replacement.read_bytes())
        proof['final_source_files'][name]=hashes['candidate_sha256']
        proof['intentional_changed_files'].append(name)
    for name, expected in proof['final_source_files'].items():
        if digest(output/name)!=expected:
            raise ValueError('final complete SDK17 source hash differs')
    proof.update(schema='verified_matching_MMC_sdk17_stage_v1',unrestricted_optin_key=
        'ock.mmc.local_service.dram.unrestricted.enabled',unrestricted_default=False,MF_binaries_modified=False)
    (output/'dsa-matching-stage-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(status='SDK17_complete_source_staged',full_files=len(proof['final_source_files']),
        intentional_deltas=len(proof['intentional_changed_files']),MF_modified=False,NPU_initialized=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--addon-dir',type=Path,default=HERE.parent)
    args=parser.parse_args()
    stage(args.archive.resolve(strict=True),args.source_manifest.resolve(strict=True),args.output.resolve(),args.addon_dir.resolve(strict=True))
