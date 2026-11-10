import pathlib,sys,json,time,datetime,re,hashlib
R=pathlib.Path('/repro/reference-root/m3v15_q4k_chunk6k_20261107/trace-m3v15');sys.path.insert(0,str(R/'harness'));import controller,preflight,spec,owned_process_utils_v1 as own
arm=sys.argv[1];assert arm in ('native','m3v9');launch=json.loads((R/('public-launch-'+arm+'.json')).read_text());cutoff=launch['stop_new_epoch'];out=R/'runs'/arm;parent=own.process(launch['pid']);assert parent and parent['startticks']==launch['startticks'];scope={parent['pid']:parent};snaps=[];last=None
assert json.loads((R/'public-CPU-ready-to-trace-v12.json').read_text())['status']=='PASS'
assert json.loads((R/'public-CPU-transport-contract.json').read_text())['status']=='PASS'
ip=json.loads((R/'public-CPU-trace-real-input-contract-v3.json').read_text());assert ip['status']=='PASS' and ip['input_validation_before_engine'] and ip['run_cases_sha256']==hashlib.sha256((R/'harness/run_cases.py').read_bytes()).hexdigest()
smoke_proof=json.loads((R.parent/'smoke4-input-v2/public-smoke-acceptance.json').read_text());assert smoke_proof['status']=='PASS_functional' and smoke_proof['remaining_owned']==0 and smoke_proof['client_returncode']==0
stopped=json.loads((R.parent/'full16/public-unprofiled-owned-exit-for-trace.json').read_text());assert stopped['remaining_owned']==0 and stopped['status']=='all_owned_stopped_for_requested_trace'

profile=preflight.check(arm);cfg=json.loads(pathlib.Path(profile['config']).read_text())
assert profile['revision']=={'m3v9':'db85ccd37d8fe90b9835903355f3bc92c1220e85','native':'fbb4d867d22095047473e3c7ca9db053f5f4c7b0'}[arm] and cfg['max_num_batched_tokens']==6144 and cfg['enable_chunked_prefill'] is True
while time.time()<cutoff:
 current=own.process(parent['pid'])
 if not current or current['startticks']!=parent['startticks']:raise SystemExit('controller exited before NPU admission')
 assert not(out/'admission.json').exists()
 ready=[json.loads(p.read_text()) for p in (out/'client/workers').glob('import-ready-*.json')]
 actual={(x['pid'],x['startticks']) for x in own.descendants(scope)}
 link=R/'public-CPU-runtime-library-job.json';linked=link.exists() and json.loads(link.read_text())['status']=='PASS'
 if len(ready)!=16 or len({x['pid'] for x in ready})!=16 or any((x['pid'],x['startticks']) not in actual or x.get('initialized') is not False for x in ready) or not linked:
  snaps=[];state=('CPU',len(ready),linked)
 else:
  rows=controller.hardware();state=('resources',tuple(x['device'] for x in rows if x['device'] in spec.DEVICES and (x['health']!='OK' or x['processes'] or x['utilization']>1 or x['used_MiB']>=5000)))
  if controller.idle(rows):
   snaps.append(dict(epoch=time.time(),hardware=rows))
   if len(snaps)>=2:break
  else:snaps=[]
 if state!=last:print(json.dumps(dict(status='waiting',state=state)),flush=True)
 last=state;time.sleep(2)
if len(snaps)<2:raise SystemExit('no NPU admission before deadline')
expected={'ock.mmc.local_service.dram.size':'10GB','ock.mmc.local_service.max.dram.size':'10GB','ock.mmc.local_service.world_size':'64','ock.mmc.local_service.protocol':'device_sdma','ock.mmc.local_service.hbm.size':'0GB'}
text=(out/'local.conf').read_text();values={k:re.search(r'^\s*'+re.escape(k)+r'\s*=\s*([^\r\n]+)',text,re.M).group(1).strip() for k in expected};assert values==expected
assert json.loads((out/'public-status.json').read_text())['unit_gate_passed'] is True
assert time.time()<cutoff
actual={(x['pid'],x['startticks']) for x in own.descendants(scope)};assert all((x['pid'],x['startticks']) in actual for x in ready)
controller.atomic(out/'admission.json',dict(allow_npu=True,workers=ready,expires=time.time()+600,idle_snapshots=snaps,root_authorized=True))
v=dict(status='NPU_admitted',arm=arm,source_commit=profile['revision'],layers=78,TP=4,DP=4,EP=16,H=524288,Q=4096,local_BS=1,global_BS=4,max_num_batched_tokens=6144,devices=list(spec.DEVICES),CPU_workers_ready=16,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());controller.atomic(R/('public-root-admission-'+arm+'.json'),v);print(json.dumps(v),flush=True)
