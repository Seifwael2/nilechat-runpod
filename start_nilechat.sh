#!/usr/bin/env bash
set -e

cd /workspace

echo "===== NILECHAT CUSTOM TEMPLATE STARTUP ====="

mkdir -p /workspace/logs
mkdir -p /workspace/models

echo "===== START JUPYTERLAB ON PORT 8888 ====="

jupyter lab \
  --allow-root \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888 \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_remote_access=True \
  --FileContentsManager.delete_to_trash=False \
  > /workspace/logs/jupyter.log 2>&1 &

echo "JupyterLab started."
echo "Logs: /workspace/logs/jupyter.log"

echo ""
echo "===== CHECK GPU ====="
nvidia-smi || true

echo ""
echo "===== CHECK MODEL FILES ====="

if [ ! -f /workspace/models/Nile-Chat-12B/config.json ]; then
  echo "NileChat model is not downloaded yet."
  echo ""
  echo "Open Jupyter/Terminal and run:"
  echo "python /opt/nilechat/download_nilechat_atomic.py"
  echo "python /opt/nilechat/validate_nilechat_shards.py"
  echo ""
  echo "After download, restart the pod or run:"
  echo "/opt/nilechat/start_nilechat.sh"
  echo ""
  echo "Container will stay alive so you can use Jupyter."
  tail -f /workspace/logs/jupyter.log
fi

ls -lh /workspace/models/Nile-Chat-12B/model-0000*-of-00005.safetensors || {
  echo "ERROR: Missing NileChat safetensors shards."
  echo "Run:"
  echo "python /opt/nilechat/download_nilechat_atomic.py"
  tail -f /workspace/logs/jupyter.log
}

echo ""
echo "===== VALIDATE SHARDS ====="
python /opt/nilechat/validate_nilechat_shards.py

echo ""
echo "===== PATCH NILECHAT / vLLM ====="
python /opt/nilechat/patch_nilechat_vllm.py

echo ""
echo "===== CLEAN OLD GPU PROCESSES ====="
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)"
for PID in $GPU_PIDS; do
  if [ "$PID" != "$$" ]; then
    echo "Killing old GPU PID: $PID"
    kill -9 "$PID" 2>/dev/null || true
  fi
done

sleep 5

echo ""
echo "===== START vLLM ON PORT 9001 ====="

export NILECHAT_MODEL_PATH=/workspace/models/Nile-Chat-12B
export NILECHAT_API_KEY="${NILECHAT_API_KEY:-nilechat-secret-123}"

export VLLM_USE_V1=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve "$NILECHAT_MODEL_PATH" \
  --served-model-name nilechat \
  --host 0.0.0.0 \
  --port 9001 \
  --api-key "$NILECHAT_API_KEY" \
  --dtype bfloat16 \
  --max-model-len "${NILECHAT_MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${NILECHAT_GPU_UTIL:-0.70}" \
  --max-num-seqs "${NILECHAT_MAX_NUM_SEQS:-1}" \
  --generation-config vllm \
  --enforce-eager