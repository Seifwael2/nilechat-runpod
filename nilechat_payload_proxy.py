import json
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request, Response

VLLM_BASE = "http://127.0.0.1:9001"

LOG_PATH = Path("/workspace/logs/nilechat_payload.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI()
client = httpx.AsyncClient(timeout=300.0)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def preview(text, limit=2000):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [TRUNCATED {len(text) - limit} chars]"


def log_json(obj: Dict[str, Any]):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
        f.write("\n" + "=" * 100 + "\n")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    start = time.time()
    url = f"{VLLM_BASE}/{path}"

    body_bytes = await request.body()
    body_json = None

    if body_bytes:
        try:
            body_json = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            body_json = None

    headers = dict(request.headers)
    headers.pop("host", None)

    logged_headers = dict(headers)
    if "authorization" in logged_headers:
        logged_headers["authorization"] = "Bearer ***REDACTED***"

    req_log = {
        "time": now(),
        "direction": "incoming_request",
        "method": request.method,
        "path": "/" + path,
        "headers": logged_headers,
    }

    if body_json and path.endswith("chat/completions"):
        messages = body_json.get("messages", [])
        req_log.update({
            "params": {
                k: body_json.get(k)
                for k in [
                    "model",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "stream",
                    "stop",
                ]
                if k in body_json
            },
            "message_count": len(messages),
            "message_roles": [m.get("role") for m in messages],
            "message_char_lengths": [len(m.get("content") or "") for m in messages],
            "messages_preview": [
                {
                    "role": m.get("role"),
                    "chars": len(m.get("content") or ""),
                    "preview": preview(m.get("content") or ""),
                }
                for m in messages
            ],
        })
    elif body_json:
        req_log["body"] = body_json

    log_json(req_log)

    upstream = await client.request(
        request.method,
        url,
        content=body_bytes,
        headers=headers,
        params=dict(request.query_params),
    )

    elapsed = round(time.time() - start, 3)

    try:
        response_json = upstream.json()
    except Exception:
        response_json = None

    res_log = {
        "time": now(),
        "direction": "upstream_response",
        "path": "/" + path,
        "status_code": upstream.status_code,
        "latency_s": elapsed,
    }

    if response_json:
        res_log["usage"] = response_json.get("usage")
        if response_json.get("choices"):
            choice = response_json["choices"][0]
            res_log["finish_reason"] = choice.get("finish_reason")
            res_log["answer_preview"] = preview(choice.get("message", {}).get("content", ""))
        if upstream.status_code >= 400:
            res_log["error"] = response_json
    else:
        res_log["raw_preview"] = preview(upstream.text)

    log_json(res_log)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )