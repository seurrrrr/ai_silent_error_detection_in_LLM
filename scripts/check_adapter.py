"""Check task-adapter integrity without importing or loading Qwen."""
from pathlib import Path
import hashlib
import json
import struct

root=Path(__file__).resolve().parents[1]
directory=root/'models/qwen_lora'
expected=json.loads((directory/'integrity.json').read_text())
for entry in expected['files']:
    path=directory/entry['filename']
    if not path.is_file():
        raise SystemExit('Missing adapter asset: '+entry['filename']+'. Obtain the separate model asset first.')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==entry['sha256'],entry['filename']
with (directory/'adapter_model.safetensors').open('rb') as stream:
    length=struct.unpack('<Q',stream.read(8))[0]
    header=json.loads(stream.read(length))
offsets=sorted(v['data_offsets'] for k,v in header.items() if k!='__metadata__')
assert offsets[0][0]==0
assert offsets[-1][1]==(directory/'adapter_model.safetensors').stat().st_size-8-length
assert all(a[1]==b[0] for a,b in zip(offsets,offsets[1:]))
print(json.dumps(dict(status='PASS',files=len(expected['files']),tensor_offsets_valid=True,model_loaded=False)))
