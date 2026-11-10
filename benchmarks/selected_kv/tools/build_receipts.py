"""Typed CPU build receipts: source provenance and artifacts, never NPU PASS."""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

FEATURES = {'libsparse_prefill_union_pipeline.so', 'libsparse_prefill_fused_fixed_launch.so',
            'libdsa_packed_inplace.so', 'libdsa_selected_li_pages.so', 'sparse_prefill_fused_fixed.o'}
BUILDERS = {'base_ops': 'build_base_ops.py', 'sdk17': 'build_sdk.py', 'selected_features': 'build_features.py'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def allowed(recipe, name):
    if Path(name).is_absolute() or '..' in Path(name).parts:
        return False
    if recipe == 'selected_features':
        return name.startswith('vllm_ascend/lib/') and Path(name).name in FEATURES
    if recipe == 'base_ops':
        if name == 'vllm_ascend/_cann_ops_custom/.gitkeep':
            return False
        return (name == 'vllm_ascend/_version.py' or name.startswith('vllm_ascend.egg-info/') or name.startswith('vllm_ascend/_cann_ops_custom/') or
                bool(re.fullmatch(r'vllm_ascend/vllm_ascend_C[^/]*\.so', name)) or
                name == 'vllm_ascend/libvllm_ascend_kernels.so')
    if recipe == 'sdk17':
        return name.split('/')[0] in ('memcache_hybrid', 'memfabric_hybrid',
                                      'memcache_hybrid-1.3.0.dist-info', 'memfabric_hybrid-1.3.0.dist-info')
    return False


def files(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}


def expected_inputs(bundle, recipe, arm):
    pins = json.loads((bundle / 'references/pinned-source-identities.json').read_text())
    if recipe == 'base_ops':
        return pins[arm]
    if recipe == 'selected_features':
        return {k: v for k, v in pins['m3v9'].items() if k.startswith('csrc/sparse_prefill_native/')}
    if recipe == 'sdk17':
        return files(bundle / 'sdk')
    raise ValueError('unknown build recipe')


def check_inputs(bundle, recipe, arm, source):
    expected = expected_inputs(bundle, recipe, arm)
    for name, digest in expected.items():
        if sha(source / name) != digest:
            raise ValueError('build input differs: ' + name)
    return expected


def toolchain():
    result = {'platform': platform.platform()}
    for name, argv in {'compiler': ['c++', '--version'], 'cmake': ['cmake', '--version'],
                       'cann_compiler': ['bisheng', '--version']}.items():
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            result[name] = {'exit_code': p.returncode, 'version': p.stdout[:4096]}
        except OSError:
            result[name] = {'unavailable': True}
    import os
    cann = Path(os.environ.get('ASCEND_HOME_PATH', '/nonexistent-cann'))
    result['cann_path'] = str(cann)
    result['cann_version_files'] = {str(p.relative_to(cann)): {'sha256': sha(p), 'text': p.read_text(errors='replace')}
                                    for p in (cann / 'version.cfg', cann / 'version.info')
                                    if p.is_file() and p.stat().st_size <= 65536}
    import importlib.metadata
    for name in ('torch', 'torch-npu'):
        result[name] = importlib.metadata.version(name)
    return result


def require_outputs(recipe, artifacts):
    if recipe == 'base_ops':
        if (not {'vllm_ascend.egg-info/PKG-INFO', 'vllm_ascend.egg-info/entry_points.txt'}.issubset(artifacts) or
                'vllm_ascend/libvllm_ascend_kernels.so' not in artifacts or
                not any(re.fullmatch(r'vllm_ascend/vllm_ascend_C[^/]*\.so', k) for k in artifacts)):
            raise ValueError('base extension and kernel library required')
    if recipe == 'sdk17':
        required = {'memcache_hybrid/lib/libmf_memcache.so', 'memfabric_hybrid/lib/libmf_hybm_core.so',
                    'memfabric_hybrid/lib/libmf_smem.so', 'memcache_hybrid/config/mmc-local.conf',
                    'memcache_hybrid/config/mmc-meta.conf'}
        if not required.issubset(artifacts):
            raise ValueError('SDK runtime library/config output missing')
    if recipe == 'selected_features' and {Path(k).name for k in artifacts} != FEATURES:
        raise ValueError('five selected feature outputs required')


def emit(bundle, recipe, arm, source, output_root, inputs, commands, output):
    artifacts = {p.relative_to(output_root).as_posix(): sha(p) for p in output_root.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts
                 and allowed(recipe, p.relative_to(output_root).as_posix())}
    if not artifacts or any(row['exit_code'] != 0 for row in commands):
        raise ValueError('incomplete CPU build')
    if recipe == 'selected_features' and {Path(k).name for k in artifacts} != FEATURES:
        raise ValueError('five selected feature artifacts required')
    require_outputs(recipe, artifacts)
    data = dict(schema_version=1, recipe=recipe, recipe_version=1, arm=arm,
                status='compiled_not_NPU_validated', builder_sha256=sha(bundle / 'tools' / BUILDERS[recipe]),
                helper_sha256=sha(bundle / 'tools/build_receipts.py'),
                inputs=inputs, source=str(source.resolve()), output_root=str(output_root.resolve()),
                outputs=artifacts, commands=commands, toolchain=toolchain(), SOC_VERSION='ascend910_9382')
    if recipe == 'base_ops':
        from email.parser import Parser
        metadata = Parser().parsestr((output_root / 'vllm_ascend.egg-info/PKG-INFO').read_text())
        data['generated_distribution'] = {'name': metadata['Name'], 'version': metadata['Version']}
    output.write_text(json.dumps(data, indent=2) + '\n')


def verify(path, bundle, recipe, arm, runtime_root):
    data = json.loads(Path(path).read_text())
    if (data.get('schema_version') != 1 or data.get('recipe') != recipe or data.get('arm') != arm
            or data.get('status') != 'compiled_not_NPU_validated'
            or data.get('builder_sha256') != sha(bundle / 'tools' / BUILDERS[recipe])
            or data.get('helper_sha256') != sha(bundle / 'tools/build_receipts.py')
            or data.get('inputs') != expected_inputs(bundle, recipe, arm)
            or not data.get('commands') or any(row['exit_code'] != 0 for row in data['commands'])
            or Path(data['output_root']).resolve() != runtime_root.resolve()):
        raise ValueError('invalid typed build receipt: ' + recipe)
    artifacts = data['outputs']
    if not artifacts or any(not allowed(recipe, k) for k in artifacts):
        raise ValueError('receipt output namespace violation')
    if recipe == 'selected_features' and {Path(k).name for k in artifacts} != FEATURES:
        raise ValueError('five feature outputs required')
    require_outputs(recipe, artifacts)
    for name, digest in artifacts.items():
        target = (runtime_root / name).resolve()
        target.relative_to(runtime_root.resolve())
        if sha(target) != digest:
            raise ValueError('build output changed: ' + name)
    return artifacts


def install_build_signal_handlers():
    import signal
    def stop(signum, frame):
        raise InterruptedError('build received signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
