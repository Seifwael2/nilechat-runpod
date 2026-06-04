#!/usr/bin/env bash
set -e

cd /workspace

mkdir -p /workspace/logs

fuser -k 8000/tcp 2>/dev/null || true

: > /workspace/logs/nilechat_payload.log
: > /workspace/logs/nilechat_proxy.log

export PYTHONPATH=/opt/nilechat:${PYTHONPATH}

exec uvicorn nilechat_payload_proxy:app \
  --host 0.0.0.0 \
  --port 8000