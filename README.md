# NileChat RunPod Custom Template Files

Use these files to build a stable RunPod Docker image instead of relying on random pod versions.

## Recommended build

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/nilechat-runpod:v1 .
docker push YOUR_DOCKERHUB_USERNAME/nilechat-runpod:v1
```

Use this image in a RunPod custom template.

## RunPod template settings

```text
Container image: YOUR_DOCKERHUB_USERNAME/nilechat-runpod:v1
Expose HTTP ports: 9001,8000,8888
Volume mount path: /workspace
```

Environment variables:

```text
NILECHAT_API_KEY=nilechat-secret-123
NILECHAT_MAX_MODEL_LEN=16384
NILECHAT_GPU_UTIL=0.70
NILECHAT_MAX_NUM_SEQS=1
```

## One-time model download

Attach the 30GB volume to `/workspace`, then run:

```bash
python /opt/nilechat/download_nilechat_atomic.py
python /opt/nilechat/validate_nilechat_shards.py
```

After that, every new pod using the same volume will reuse the model.

## Start server

The container starts automatically with:

```bash
/opt/nilechat/start_nilechat.sh
```

Test:

```bash
curl http://127.0.0.1:9001/v1/models -H "Authorization: Bearer nilechat-secret-123"
```

## Optional proxy

```bash
/opt/nilechat/start_proxy.sh
```

## Don't run

```bash
pip install -U vllm
pip install -U transformers
pip install -U huggingface_hub
pip install -U "huggingface_hub[cli]"
pip install -U torch
```
