"""Allowlisted scalar export for the single-service LI transport comparison."""
import argparse
import json
import math
from pathlib import Path
from statistics import median
import spec

ROOT=Path('/repro/reference-root/m3v15_q4k_chunk6k_20261107/smoke4-input-v2')

def finite(x):
    y=float(x)
    if not math.isfinite(y): raise ValueError('nonfinite metric')
    return y

def samples(root, case):
    return [json.loads(p.read_text()) for p in sorted((root/'samples').glob(f'{case}-*.json'))]

def validate(value,H,Q,bs):
    rows=value['rank_rows']
    if len(rows)!=spec.DP*spec.TP or {(r['dp_rank'],r['tp_rank']) for r in rows}!={(d,t) for d in range(spec.DP) for t in range(spec.TP)}:
        raise ValueError('incomplete rank set')
    for r in rows:
        if not r['valid'] or r['issues'] or r['actual_H']!=[H]*(bs//spec.DP) or r['actual_Q']!=[Q]*(bs//spec.DP) or r['model_forward_calls']!=1:
            raise ValueError('rank geometry or path rejected')
        if r['li_transport']!=value['li_transport']: raise ValueError('mode mismatch')
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--client',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--include-q1024',action='store_true');args=parser.parse_args()
    cases=[(c['H'],c['request_Qs'][0],len(c['request_Qs'])) for c in spec.all_cases()]
    output=[]
    all_checks=[]
    for H,Q,bs in cases:
        case=spec.all_cases()[0]['case'];data=samples(args.client,case)
        for v in data: validate(v,H,Q,bs)
        oracle=[v for v in data if v['phase']=='oracle']
        if len(oracle)!=1 or oracle[0]['li_transport']!='mte': raise ValueError('single MTE oracle required')
        rank_proofs=oracle[0]['oracle']
        if len(rank_proofs)!=spec.DP*spec.TP or {(x['dp_rank'],x['tp_rank']) for x in rank_proofs}!={(d,t) for d in range(spec.DP) for t in range(spec.TP)} or any(not x['oracle'] or len(x['records'])!=3 for x in rank_proofs): raise ValueError('four ranks, three owners oracle required')
        checks=[r for x in rank_proofs for r in x['records']]
        all_checks.extend(checks)
        if any(not r['passed'] or any(r['mismatch_bytes']) or not r['selected_pages'] for r in checks): raise ValueError('LI byte oracle failed')
        for mode in ('sdk','mte'):
            warm=[v for v in data if v['phase']=='warmup' and v['li_transport']==mode]
            formal=[v for v in data if v['phase']=='measure' and v['li_transport']==mode]
            probes=[v for v in data if v['phase']=='probe' and v['li_transport']==mode]
            if len(warm)!=2 or len(formal)!=3 or len(probes)!=1: raise ValueError('phase counts invalid')
            if len({v['input_digest'] for v in data})!=len(data): raise ValueError('a suffix was reused')
            forward=[max(finite(r['model_npu_ms']) for r in v['rank_rows']) for v in formal]
            wall=[finite(v['all_first_tokens_ms']) for v in formal]
            peak=max(finite(r['allocated_peak_bytes']) for v in formal for r in v['rank_rows'])
            probe=probes[0]['rank_rows']
            if any(r['selected_union_error_flags']!=0 or len(r['ffn_calls'])!=spec.LAYERS or any(n!=1 for n in r['ffn_calls'].values()) or len(r['sfa_calls'])!=spec.LAYERS or any(n!=1 for n in r['sfa_calls'].values()) for r in probe): raise ValueError('probe path failed')
            output.append(dict(case=case,H=H,Q=Q,global_BS=bs,local_BS=bs//spec.DP,mode=mode,
                formal_forward_maxrank_ms=forward,forward_median_ms=median(forward),
                first_token_wall_ms=wall,first_token_wall_median_ms=median(wall),
                hbm_allocated_peak_bytes=int(peak),
                probe_per_rank=[{k:r.get(k) for k in ('dp_rank','tp_rank','sdk_load_wall_ms','selected_main_copy_ms','li_mte_callback_wall_ms','li_mte_callback_count','exposed_load_ms','selected_union_error_flags')} for r in probe]))
    proof=dict(owner_checks=len(all_checks),selected_pages=sum(r['selected_pages'] for r in all_checks),checked_bytes=sum(sum(r['checked_bytes']) for r in all_checks),mismatch_bytes=sum(sum(r['mismatch_bytes']) for r in all_checks),max_abs_byte_error=max(max(r['max_abs_byte_error']) for r in all_checks))
    result=dict(rows=output,oracle=proof,complete=True,correctness_scope='LI selected history pages byte-equivalent to SDK; distinct suffix performance samples are not paired full-model output equality',performance_scope='forward NPU event max across ranks; oracle and probes excluded',raw_data_exported=False)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='complete',rows=len(output),oracle=proof)))
if __name__=='__main__': main()
