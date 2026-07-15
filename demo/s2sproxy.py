"""
Same-origin WebSocket passthrough to an in-container speech-to-speech backend.

Deployment-only (single-container Space): HF Spaces expose one port, so the
browser can't dial the pipeline's internal port directly. This route pumps
`/v1/realtime` on the public port to the internal realtime server. Enabled by
setting S2S_INTERNAL_WS (e.g. ws://127.0.0.1:8765/v1/realtime); without it the
route refuses connections.
"""

import asyncio
import logging
import os

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import llmproxy
import slotqueue

logger = logging.getLogger("s2s.proxy")

router = APIRouter()

TARGET = os.environ.get("S2S_INTERNAL_WS", "").strip()
# When set, only signed-in visitors may talk: their OAuth token (scope
# inference-api) is registered with llmproxy so LLM usage bills THEM.
REQUIRE_LOGIN = os.environ.get("REQUIRE_LOGIN", "0") == "1"


def _visitor_token(ws: WebSocket) -> str | None:
    """The signed-in visitor's OAuth access token, if any. The login cookie
    rides the same-origin WS handshake; the session middleware attached by
    huggingface_hub populates .session for websockets too."""
    try:
        from huggingface_hub import parse_huggingface_oauth

        info = parse_huggingface_oauth(ws)  # duck-typed: reads .session
        return getattr(info, "access_token", None) if info else None
    except Exception:  # noqa: BLE001 - treated as signed out
        return None


@router.websocket("/v1/realtime")
async def realtime_proxy(ws: WebSocket):
    if not TARGET:
        await ws.close(code=1013)
        return
    token = _visitor_token(ws)
    if REQUIRE_LOGIN and not token:
        try:
            await ws.accept()
            await ws.close(code=4401, reason="Sign in with Hugging Face to talk")
        except Exception:  # noqa: BLE001 - client may already be gone
            pass
        return
    # With the in-server queue on, a socket must present the session_token from
    # its /api/session grant — that's what makes the queue unskippable. Claiming
    # binds the slot to this socket; the finally releases it on disconnect.
    session_token = ws.query_params.get("session_token", "")
    if slotqueue.ENABLED and not slotqueue.claim(session_token):
        try:
            await ws.accept()
            await ws.close(code=4409, reason="No session slot — reconnect from the page")
        except Exception:  # noqa: BLE001 - client may already be gone
            pass
        return
    llmproxy.set_active_token(token)
    await ws.accept()
    try:
        async with websockets.connect(TARGET, max_size=None) as upstream:

            async def pump_up():
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        # Normal termination — don't let it escape as an error.
                        return
                    if msg.get("text") is not None:
                        await upstream.send(msg["text"])
                    elif msg.get("bytes") is not None:
                        await upstream.send(msg["bytes"])

            async def pump_down():
                async for msg in upstream:
                    if isinstance(msg, str):
                        await ws.send_text(msg)
                    else:
                        await ws.send_bytes(msg)

            done, pending = await asyncio.wait(
                [asyncio.create_task(pump_up()), asyncio.create_task(pump_down())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("realtime proxy error: %r", exc)
    finally:
        if slotqueue.ENABLED:
            slotqueue.release(session_token)
        llmproxy.clear_active_token()
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
