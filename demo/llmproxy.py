"""
LLM passthrough that bills the signed-in visitor.

Single-container deploys point the pipeline's OpenAI-compatible base URL at
this route (LLM_BASE_URL=http://127.0.0.1:7860/llmproxy/v1). Requests are
forwarded verbatim to the real provider (LLM_UPSTREAM_URL, default the HF
router) with the Authorization header swapped to the active visitor's OAuth
token (scope `inference-api`) — so LLM usage bills the visitor's own HF
credits, not the Space owner. s2sproxy registers/clears the active token as
the visitor's realtime socket connects/disconnects; the pool serves one
conversation at a time, so one active token suffices. Without a visitor
token (local dev, warmup at boot) the HF_TOKEN fallback is used.
"""

import logging
import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("s2s.llmproxy")

router = APIRouter()

UPSTREAM = os.environ.get("LLM_UPSTREAM_URL", "https://router.huggingface.co/v1").rstrip("/")
FALLBACK_TOKEN = os.environ.get("HF_TOKEN", "").strip()

_active_token: str | None = None


def set_active_token(token: str | None) -> None:
    global _active_token
    _active_token = token or None
    logger.info("active visitor token %s", "set" if _active_token else "cleared")


def clear_active_token() -> None:
    set_active_token(None)


@router.post("/llmproxy/v1/chat/completions")
async def chat_completions(request: Request):
    token = _active_token or FALLBACK_TOKEN
    if not token:
        return JSONResponse(
            {"error": {"message": "No credentials: sign in with Hugging Face to talk."}},
            status_code=401,
        )
    body = await request.body()
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))
    upstream_request = client.build_request(
        "POST",
        f"{UPSTREAM}/chat/completions",
        content=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": request.headers.get("accept", "*/*"),
        },
    )
    upstream = await client.send(upstream_request, stream=True)

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )
