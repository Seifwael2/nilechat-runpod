FROM vllm/vllm-openai:v0.8.5.post1

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV VLLM_USE_V1=0
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN python3 --version && \
    python3 -m pip --version && \
    ln -sf $(which python3) /usr/local/bin/python

RUN python3 -m pip install --no-cache-dir \
    --retries 20 \
    --timeout 120 \
    "jupyterlab==4.2.5" \
    "requests==2.32.3"

COPY patch_nilechat_vllm.py /opt/nilechat/patch_nilechat_vllm.py
COPY validate_nilechat_shards.py /opt/nilechat/validate_nilechat_shards.py
COPY download_nilechat_atomic.py /opt/nilechat/download_nilechat_atomic.py
COPY nilechat_payload_proxy.py /opt/nilechat/nilechat_payload_proxy.py
COPY start_nilechat.sh /opt/nilechat/start_nilechat.sh
COPY start_proxy.sh /opt/nilechat/start_proxy.sh

RUN chmod +x /opt/nilechat/start_nilechat.sh /opt/nilechat/start_proxy.sh

EXPOSE 9001
EXPOSE 8000
EXPOSE 8888

ENTRYPOINT ["/opt/nilechat/start_nilechat.sh"]
