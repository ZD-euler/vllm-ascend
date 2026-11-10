import argparse
import hashlib
import json
from pathlib import Path
import struct

from transformers import AutoTokenizer

BASE = Path('/repro/reference-root')
ROOT = BASE / 'real_coding'
tokenizer = AutoTokenizer.from_pretrained(str(BASE / 'models/GLM-5.2-w8a8-10layers-c8-ddr-baseline'),
                                         local_files_only=True)
parser=argparse.ArgumentParser()
parser.add_argument('--manifest',default='canonical_manifest.json')
parser.add_argument('--output-manifest',default='tokenized_manifest.json')
cli_args=parser.parse_args()
manifest = json.loads((ROOT / cli_args.manifest).read_text())
for row in manifest:
    text = (ROOT / row['canonical_file']).read_text()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    row['glm52_tokens'] = len(tokens)
    row['tokens_sha256_uint32_le'] = hashlib.sha256(struct.pack(f'<{len(tokens)}I', *tokens)).hexdigest()
    token_path = ROOT / (row['task_name'] + '.tokens.json')
    token_path.write_text(json.dumps(tokens))
    row['token_file'] = token_path.name
    print(row['task_name'], len(tokens), flush=True)
(ROOT / cli_args.output_manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
