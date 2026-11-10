"""Download a bounded set of original public coding trajectory artifacts."""

import hashlib
import json
import pathlib
import subprocess
import time

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent
REPOSITORY = 'https://hf-mirror.com/datasets/Contextbench/Tracebench/resolve/main/'
LIMIT = 64 * 1024 * 1024


def fetch(relative, target):
    if target.exists():
        return target.read_bytes()
    for attempt in range(3):
        try:
            print('FETCH', relative, 'attempt', attempt + 1, flush=True)
            temporary = target.with_suffix(target.suffix + '.part')
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--connect-timeout', '10', '--max-time', '90', '--max-filesize', str(LIMIT),
                            REPOSITORY + relative, '--output', str(temporary)], check=True)
            data = temporary.read_bytes()
            if len(data) > LIMIT:
                raise ValueError('Artifact exceeds bounded download limit')
            temporary.replace(target)
            return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


ROOT.mkdir(exist_ok=True, parents=True)
fetch('bench_manifest.full.parquet', ROOT / 'bench_manifest.full.parquet')
rows = pq.read_table(ROOT / 'bench_manifest.full.parquet', columns=[
    'traj_id', 'agent', 'task_name', 'category', 'tags', 'step_count', 'artifact_path', 'source_relpath'
]).to_pylist()
coding = [r for r in rows if r.get('artifact_path') and
          (r.get('category') == 'software-engineering' or
           '__' in str(r.get('task_name', '')) or
           'coding' in str(r.get('tags', '')).lower())]
coding.sort(key=lambda r: int(r['step_count']), reverse=True)
chosen = []
for row in coding:
    if len(chosen) >= 8:
        break
    if any(r['task_name'] == row['task_name'] for r in chosen):
        continue
    relative = row['artifact_path']
    target = ROOT / pathlib.PurePosixPath(relative).name
    try:
        data = fetch(relative, target)
    except Exception as exc:
        print('SKIP', row['traj_id'], repr(exc), flush=True)
        continue
    record = dict(row, download_url=REPOSITORY + relative, local_file=target.name,
                  bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    chosen.append(record)
    (ROOT / 'downloaded_manifest.json').write_text(json.dumps(chosen, indent=2))
    print(json.dumps({k: record.get(k) for k in ['traj_id', 'task_name', 'step_count', 'bytes', 'sha256']}), flush=True)
print('DOWNLOAD_COMPLETE', len(chosen), flush=True)
