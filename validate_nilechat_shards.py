from pathlib import Path
from safetensors import safe_open

model_dir = Path('/workspace/models/Nile-Chat-12B')
bad = []
for p in sorted(model_dir.glob('model-0000*-of-00005.safetensors')):
    try:
        with safe_open(p, framework='pt') as f:
            keys = list(f.keys())
        print('OK:', p.name, '| tensors:', len(keys), '| size:', round(p.stat().st_size / 1024**3, 2), 'GB')
    except Exception as e:
        print('BAD:', p.name, '|', repr(e)); bad.append(p.name)
if bad:
    raise SystemExit('BAD SHARDS: ' + str(bad))
print('\nALL SHARDS VALID')
