"""Read public trace JSON as data; never execute commands or unpickle state."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pyarrow as pa

ROOT = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--manifest',default='downloaded_manifest.json')
parser.add_argument('--output-manifest',default='canonical_manifest.json')
cli_args=parser.parse_args()
manifest = json.loads((ROOT / cli_args.manifest).read_text())
outputs = []


def plain(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(plain(v.get('text', v.get('content', ''))) if isinstance(v, dict) else plain(v)
                         for v in value)
    return json.dumps(value, ensure_ascii=False)


for source in manifest:
    archive = ROOT / source['local_file']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == source['sha256']
    raw = pa.CompressedInputStream(pa.BufferReader(archive.read_bytes()), 'zstd').read()
    assert len(raw) < 128 * 1024 * 1024
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        members = tar.getmembers()
        selected = []
        events = []
        if source['agent'] == 'OpenHands':
            sessions = [m for m in members if '/sessions/' in m.name and m.name.endswith('.json')
                        and '/events/' not in m.name and '/event_cache/' not in m.name]
            if sessions:
                member = max(sessions, key=lambda m: m.size)
                selected = [member.name]
                stream = json.load(tar.extractfile(member))
            else:
                selected_members = [m for m in members if '/events/' in m.name and m.name.endswith('.json')]
                selected = [m.name for m in selected_members]
                stream = [json.load(tar.extractfile(m)) for m in selected_members]
                if not stream:
                    # SWE OpenHands artifacts store cumulative API snapshots,
                    # rather than an event session. Use one canonical snapshot.
                    snapshots = [m for m in members if m.name.endswith('.json')]
                    member = max(snapshots, key=lambda m: m.size)
                    selected = [member.name]
                    messages = json.load(tar.extractfile(member))['messages']
                    stream = []
                    for index, message in enumerate(messages):
                        text = plain(message.get('content'))
                        if message.get('tool_calls'):
                            text += '\n' + json.dumps(message['tool_calls'], ensure_ascii=False)
                        stream.append(dict(id=index, action='message', source=message.get('role', 'unknown'),
                                           args=dict(content=text)))
            unique = {}
            for event in stream:
                event_id = event.get('id')
                if event_id in unique:
                    assert unique[event_id] == event, event_id
                unique[event_id] = event
            for event_id, event in sorted(unique.items(), key=lambda x: int(x[0])):
                action = event.get('action')
                args = event.get('args') or {}
                if action:
                    if action in ('system', 'message'):
                        text = plain(args.get('content') or event.get('message'))
                    elif action in ('change_agent_state', 'recall'):
                        continue
                    else:
                        # Actual command/edit/read arguments, once. The event's
                        # message and model_response contain duplicate copies.
                        text = plain(args)
                else:
                    if event.get('observation') == 'agent_state_changed':
                        continue
                    text = plain(event.get('content') or event.get('message'))
                if text:
                    events.append((str(event_id), event.get('source', 'tool'),
                                   action or event.get('observation', 'observation'), text))
        else:
            snapshots = [m for m in members if m.name.endswith('/debug.json')]
            # Each debug file contains cumulative messages. Select one final
            # canonical conversation, never concatenate prompt snapshots.
            member = max(snapshots, key=lambda m: m.size)
            selected = [member.name]
            messages = json.load(tar.extractfile(member))['messages']
            for index, message in enumerate(messages):
                text = plain(message.get('content'))
                calls = message.get('tool_calls')
                if calls:
                    text += '\n' + json.dumps(calls, ensure_ascii=False)
                if text:
                    events.append((str(index), message.get('role', 'unknown'), 'message', text))
        pieces, boundaries, offset = [], [], 0
        for event_id, role, kind, text in events:
            piece = f'\n<event id="{event_id}" role="{role}" kind="{kind}">\n{text}\n</event>\n'
            pieces.append(piece)
            boundaries.append({'event_id': event_id, 'char_start': offset, 'char_end': offset + len(piece)})
            offset += len(piece)
        content = ''.join(pieces)
        output = ROOT / (source['task_name'] + '.txt')
        output.write_text(content)
        (ROOT / (source['task_name'] + '.events.json')).write_text(json.dumps(boundaries))
        row = dict(source, canonical_file=output.name, canonical_events=len(events),
                   canonical_chars=len(content), canonical_bytes=len(content.encode()),
                   canonical_sha256=hashlib.sha256(content.encode()).hexdigest(),
                   canonical_archive_members=selected,
                   extraction='one chronological event stream; duplicate storage snapshots excluded; original commands and outputs retained')
        outputs.append(row)
        print(source['task_name'], len(events), len(content), flush=True)
(ROOT / cli_args.output_manifest).write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
