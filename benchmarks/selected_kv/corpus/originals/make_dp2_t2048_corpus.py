"""Build a T=2048 global corpus for TP8/DP2/EP16 (T=1024 per DP).

Uses the existing real-window large-priority corpus. Does not overwrite the
T=1024 DP1 corpus.
"""
import json
from pathlib import Path

BASE = Path('/repro/reference-root')
SOURCE = BASE / 'c8_42layer_20261020/large-priority-corpus/manifest.json'
ROOT = BASE / '22layer_t1k_20261021'
TARGET = ROOT / 'corpus_dp2_t2048_v1'
HISTORIES = (65536, 131072, 262144, 524288)
QUERIES = (128, 256, 512, 1024)
TOTAL = 2048
DP = 2


def main():
    source = json.loads(SOURCE.read_text())
    by_key = {}
    for case in source['cases']:
        by_key.setdefault((int(case['history']), int(case['query'])), []).append(case)

    cases = []
    for history in HISTORIES:
        for query in QUERIES:
            assert TOTAL % query == 0
            requests = TOTAL // query
            candidates = by_key.get((history, query), [])
            candidates = [c for c in candidates
                          if int(c.get('total_query_tokens', 0)) >= TOTAL
                          and len(c.get('windows', [])) >= requests]
            assert candidates, (history, query)
            chosen = sorted(candidates, key=lambda c: int(c['total_query_tokens']))[0]
            windows = []
            for index, window in enumerate(chosen['windows'][:requests]):
                item = dict(window)
                item['request'] = index
                assert item['stop_token'] - item['start_token'] == history + query
                windows.append(item)
            cases.append(dict(
                case=f'h{history}-q{query}-t{TOTAL}',
                history=history,
                query=query,
                requests=requests,
                local_requests=requests // DP,
                total_query_tokens=TOTAL,
                windows=windows,
            ))

    manifest = dict(
        status='passed',
        corpus_version='t2k_22layer_dp2_real_windows_v1',
        cases=cases,
        sources=source['sources'],
        case_count=len(cases),
        total_requests=sum(c['requests'] for c in cases),
        source_manifests=[str(SOURCE)],
        source_kind=source.get('source_kind'),
        scope='Global T=2048; per-DP T=1024 at DP2; H64K..512K; Q128..1024; real coding windows',
        history_and_increment_contiguous=True,
        synthetic_padding_tokens=False,
        topology='TP8/DP2/EP16',
    )
    TARGET.mkdir(parents=True, exist_ok=False)
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='passed', target=str(TARGET),
                          case_count=len(cases),
                          total_requests=manifest['total_requests'],
                          cases=[c['case'] for c in cases]), indent=2), flush=True)


if __name__ == '__main__':
    main()
