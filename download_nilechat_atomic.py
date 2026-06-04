import os, re, time, getpass
from pathlib import Path
from urllib.parse import quote
import requests

REPO='MBZUAI-Paris/Nile-Chat-12B'
MODEL_DIR=Path('/workspace/models/Nile-Chat-12B')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FILES=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json','model-00001-of-00005.safetensors','model-00002-of-00005.safetensors','model-00003-of-00005.safetensors','model-00004-of-00005.safetensors','model-00005-of-00005.safetensors']
TOKEN=os.environ.get('HF_TOKEN') or getpass.getpass('Paste HF token: ').strip()
if not TOKEN: raise SystemExit('No HF token provided.')
S=requests.Session(); S.headers.update({'Authorization': f'Bearer {TOKEN}'})
def url(fn): return f'https://huggingface.co/{REPO}/resolve/main/{quote(fn)}'
def human(n):
    if n is None: return 'unknown'
    if n>=1024**3: return f'{n/1024**3:.2f}GB'
    if n>=1024**2: return f'{n/1024**2:.2f}MB'
    return f'{n}B'
def remote_size(fn):
    r=S.head(url(fn), allow_redirects=True, timeout=(30,60))
    if r.status_code in (401,403): raise RuntimeError(f'Auth error {r.status_code}. Check token/model access.')
    if r.status_code==404: raise RuntimeError(f'Not found: {fn}')
    cl=r.headers.get('Content-Length')
    return int(cl) if cl and cl.isdigit() else None
def total_from_range(v):
    m=re.search(r'/(\d+)$', v or '')
    return int(m.group(1)) if m else None

def download_one(fn, retries=20):
    out=MODEL_DIR/fn; tmp=MODEL_DIR/(fn+'.tmp')
    size=remote_size(fn)
    if out.exists() and out.stat().st_size>0:
        if size is None or out.stat().st_size==size:
            print(f'SKIP existing: {fn} | {human(out.stat().st_size)}'); return
        out.unlink()
    for attempt in range(1, retries+1):
        existing=tmp.stat().st_size if tmp.exists() else 0
        headers={'Range': f'bytes={existing}-'} if existing>0 else {}
        mode='ab' if existing>0 else 'wb'
        print(f'\n===== DOWNLOAD {fn} | attempt {attempt}/{retries} =====')
        print(f'resume_from={human(existing)} remote_size={human(size)}')
        try:
            with S.get(url(fn), headers=headers, stream=True, allow_redirects=True, timeout=(30,120)) as r:
                if r.status_code in (401,403): raise RuntimeError(f'Auth error {r.status_code}')
                if r.status_code==404: raise RuntimeError(f'Not found: {fn}')
                if existing>0 and r.status_code==200:
                    existing=0; mode='wb'; print('Server ignored Range; restarting file.')
                total=size or total_from_range(r.headers.get('Content-Range'))
                last=time.time(); chunk_bytes=0
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=8*1024*1024):
                        if not chunk: continue
                        f.write(chunk); chunk_bytes += len(chunk)
                        now=time.time()
                        if now-last>=10:
                            cur=tmp.stat().st_size; pct=f'{cur/total*100:.1f}%' if total else '?'
                            print(f'{fn}: {human(cur)} / {human(total)} ({pct}) | speed={human(int(chunk_bytes/(now-last)))}/s', flush=True)
                            chunk_bytes=0; last=now
            final=tmp.stat().st_size
            if size is not None and final!=size:
                raise RuntimeError(f'Size mismatch: got {final}, expected {size}')
            tmp.rename(out); print(f'DONE: {fn} | {human(out.stat().st_size)}'); return
        except Exception as e:
            print(f'ERROR: {fn}: {repr(e)}'); time.sleep(10)
    raise RuntimeError(f'Failed after retries: {fn}')

print('===== NileChat Atomic Downloader =====')
for f in FILES: download_one(f)
print('\n===== DOWNLOAD COMPLETE =====')
for f in FILES:
    p=MODEL_DIR/f; print(f, human(p.stat().st_size) if p.exists() else 'MISSING')
