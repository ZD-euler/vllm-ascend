#!/usr/bin/env python3
"""Prepare a fresh portable harness; CPU only, never launches a model."""
import argparse
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
REQUIRED = ('model', 'model_reference', 'corpus', 'corpus_reference', 'token_root', 'source', 'core', 'sdk', 'deploy', 'cann')


def prepare(settings, recovered, output):
    settings = dict(settings)
    for key in REQUIRED:
        settings[key] = str(Path(settings[key]).expanduser().resolve(strict=True))
    if settings['arm'] not in ('native', 'm3v9'):
        raise ValueError('unsupported arm')
    expected_revision = {'native': 'fbb4d867d22095047473e3c7ca9db053f5f4c7b0',
                         'm3v9': 'db85ccd37d8fe90b9835903355f3bc92c1220e85'}
    if settings.get('revision') != expected_revision[settings['arm']]:
        raise ValueError('pinned source revision required')
    if output.exists():
        raise ValueError('output must be a new directory')
    originals = list(recovered.glob('engine-*.json'))
    if not originals:
        raise ValueError('missing recovered engine config')
    selected = recovered / ('engine-' + settings['arm'] + '.json')
    if not selected.is_file():
        raise ValueError('missing arm engine config')
    cfg = json.loads(selected.read_text())
    cfg['model'] = settings['model']
    additional = cfg.setdefault('additional_config', {})
    for key, relative in {'ascend_log_path': 'runs/' + settings['arm'] + '/logs',
                          'runtime_config_path': 'runtime/config/runtime_config.json',
                          'runtime_report_dir': 'runtime/report',
                          'runtime_dump_dir': 'runtime/dump'}.items():
        additional[key] = str(output.resolve() / relative)
    output.mkdir(parents=True)
    shutil.copytree(recovered / 'harness', output / 'harness')
    (output / 'portable-settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    (output / 'engine.json').write_text(json.dumps(cfg, indent=2) + '\n')
    shutil.copyfile(HERE / 'portable_preflight.py', output / 'harness/preflight.py')
    shutil.copyfile(HERE / 'build_receipts.py', output / 'harness/build_receipts.py')
    for name in ('controller.py', 'export_compare.py', 'export_mte.py', 'import_preflight.py'):
        path = output / 'harness' / name
        text = path.read_text()
        text = re.sub(r"(?m)^(ROOT|DEFAULT_ROOT)\s*=\s*Path\([^\n]+\)",
                      lambda m: m.group(1) + ' = Path(__file__).resolve().parent.parent', text)
        text = text.replace('preflight.BASE / "matrix42_15cases_20261031/deploy"', 'preflight.DEPLOY')
        text = text.replace('preflight.BASE/\'matrix42_15cases_20261031/deploy\'', 'preflight.DEPLOY')
        text = text.replace('str(Path("/tmp") / ("m3li-v4-" + out.name))', 'str(out / "tmp")')
        text = text.replace('(Path("/tmp") / ("m3li-v4-" + arm)).mkdir', '(out / "tmp").mkdir')
        path.write_text(text)
    path = output / 'harness/run_cases.py'
    text = path.read_text().replace("Path('/repro/reference-root/real_coding')", "Path(json.loads((Path(__file__).resolve().parent.parent / 'portable-settings.json').read_text())['token_root'])")
    path.write_text(text)
    # Preserve originals as evidence; record exact adapter scope for reviewers.
    (output / 'ADAPTATION.json').write_text(json.dumps({'changes': [
        'path relocation only in controller/exporters/input token-root',
        'preflight replaced by explicit portable settings and recursive runtime hashes',
        'engine model/log/report/dump paths relocated',
        'seed, burst, completion, profiler and worker math unchanged']}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--recovered', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.settings.read_text()), args.recovered, args.output)
    print('Prepared only. Enroll and verify runtime locks before explicit model launch.')


if __name__ == '__main__':
    main()
