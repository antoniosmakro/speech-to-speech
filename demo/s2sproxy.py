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

import auth
import llmproxy
import slotqueue

logger = logging.getLogger("s2s.proxy")

router = APIRouter()

TARGET = os.environ.get("S2S_INTERNAL_WS", "").strip()
# When set, only signed-in visitors may talk: their OAuth token (scope
# inference-api) is registered with llmproxy so LLM usage bills THEM.
REQUIRE_LOGIN = os.environ.get("REQUIRE_LOGIN", "0") == "1"
UPSTREAM_WAIT_S = 90


def _visitor_token(ws: WebSocket) -> str | None:
    """The signed-in visitor's OAuth access token, if any (and not expired).
    The login cookie rides the same-origin WS handshake; the session middleware
    attached by huggingface_hub populates .session for websockets too."""
    info = auth.current_oauth(ws)  # duck-typed: reads .session
    return getattr(info, "access_token", None) if info else None


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
    # Anything but a visitor-side disconnect closes with an error code: the
    # page reads a plain 1000 as its own Stop and would sit waiting.
    close_code, close_reason = 1000, ""
    try:
        try:
            upstream = await _connect_upstream()
        except OSError as exc:
            logger.warning("realtime proxy: pipeline unreachable: %r", exc)
            close_code, close_reason = 1013, "Still starting up — tap to try again"
            return
        async with upstream:

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

            up, down = asyncio.create_task(pump_up()), asyncio.create_task(pump_down())
            done, pending = await asyncio.wait([up, down], return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if down in done:
                close_code = 1012  # the pipeline ended the session: restart or shutdown
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("realtime proxy error: %r", exc)
        close_code = 1011
    finally:
        if slotqueue.ENABLED:
            slotqueue.release(session_token)
        llmproxy.clear_active_token()
        try:
            await ws.close(code=close_code, reason=close_reason)
        except Exception:  # noqa: BLE001
            pass


async def _connect_upstream():
    """The pipeline boots ~40 s after this app, which HF already serves; a
    visitor who starts in that window waits for it rather than failing."""
    deadline = asyncio.get_running_loop().time() + UPSTREAM_WAIT_S
    while True:
        try:
            return await websockets.connect(TARGET, max_size=None)
        except OSError:
            if asyncio.get_running_loop().time() >= deadline:
                raise
            await asyncio.sleep(1)
