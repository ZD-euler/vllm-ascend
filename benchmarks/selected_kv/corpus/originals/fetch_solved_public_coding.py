"""Fetch additional solved real coding traces; bounded public artifacts only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
prior={r['task_name'] for r in json.loads((ROOT/'downloaded_manifest.json').read_text())}
chosen=[]
for r in json.loads((ROOT/'solved_long_candidates.json').read_text()):
    if r['agent']=='SWE-agent' or r['task_name'] in prior:continue
    chosen.append(r);prior.add(r['task_name'])
    if len(chosen)==8:break

def fetch(row):
    path=ROOT/Path(row['artifact_path']).name
    url='https://hf-mirror.com/datasets/Contextbench/Tracebench/resolve/main/'+row['artifact_path']
    if not path.exists():
        temp=path.with_suffix(path.suffix+'.part')
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','15',
            '--max-time','120','--max-filesize',str(64*1024*1024),url,'--output',str(temp)],check=True)
        temp.replace(path)
    data=path.read_bytes()
    result=dict(row,download_url=url,local_file=path.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    print(row['task_name'],len(data),flush=True)
    return result

with ThreadPoolExecutor(max_workers=3) as pool:
    outcomes=list(pool.map(fetch,chosen))
(ROOT/'downloaded_solved_manifest.json').write_text(json.dumps(outcomes,indent=2))
print('SOLVED_TRACES_DOWNLOADED',len(outcomes),flush=True)
