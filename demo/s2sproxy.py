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

logger = logging.getLogger("s2s.proxy")

router = APIRouter()

TARGET = os.environ.get("S2S_INTERNAL_WS", "").strip()


@router.websocket("/v1/realtime")
async def realtime_proxy(ws: WebSocket):
    if not TARGET:
        await ws.close(code=1013)
        return
    await ws.accept()
    try:
        async with websockets.connect(TARGET, max_size=None) as upstream:

            async def pump_up():
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect(msg.get("code") or 1000)
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
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("realtime proxy error: %r", exc)
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
