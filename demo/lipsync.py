"""
Wav2Arkit lip-sync WebSocket route.

Turns assistant speech (PCM the browser is about to play) into 52 ARKit
blendshape weights at 30 FPS, driving the Nyx avatar's mouth. The model is
`myned-ai/wav2arkit_cpu` (ONNX, CPU real-time), vendored from
myned-ai/avatar-chat-server under `wav2arkit/`.

Protocol — JSON over `/api/wav2arkit`:

  client -> server:
    {"type": "audio", "audio": "<base64 PCM16 16 kHz mono>"}   assistant audio, in arrival order
    {"type": "reset"}    new turn or barge-in: drop buffered audio, frame counter back to 0
    {"type": "flush"}    turn audio complete: emit frames for the buffered tail

  server -> client:
    {"type": "ready"}                                     model loaded, stream me audio
    {"type": "frames", "start": N, "weights": [{...}]}    frames N..N+len-1 since the last
                                                          reset; frame k covers samples
                                                          [k*533.3, (k+1)*533.3) of the turn
    {"type": "error", "message": "..."}

Inference is chunked in fixed 300 ms blocks (9 frames): the model emits
floor(samples/533.3) frames per call, so uneven chunks would silently drop
tail samples and drift out of sync. One engine serves the app; calls are
serialized (the realtime pool is 1 conversation on this deployment anyway).
"""

import asyncio
import base64
import logging
import os

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("s2s.lipsync")

router = APIRouter()

MODEL_REPO = "myned-ai/wav2arkit_cpu"
MODEL_DIR = os.environ.get("WAV2ARKIT_DIR", "./pretrained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "wav2arkit_cpu.onnx")
# ONNX intra-op thread cap (0 = ONNX default). The model runs 12.8x realtime
# on 2 threads; an unbounded pool starves co-located inference.
THREADS = int(os.environ.get("WAV2ARKIT_THREADS", "2"))
SAMPLE_RATE = 16000
FPS = 30
# 300 ms per inference call — a multiple of the 533.33-samples-per-frame
# quantum (9 frames exactly), and the chunk size the model is known to
# behave well on (what avatar-chat-server feeds it).
BLOCK_SAMPLES = 4800

_engine = None
_engine_lock = asyncio.Lock()


def _load_engine():
    """Load (downloading the model first if absent). Blocking; run in a thread."""
    from wav2arkit import Wav2ArkitInference

    if not os.path.exists(MODEL_PATH):
        from huggingface_hub import snapshot_download

        logger.info("Downloading %s to %s ...", MODEL_REPO, MODEL_DIR)
        snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
    return Wav2ArkitInference(
        model_path=MODEL_PATH,
        audio_sr=SAMPLE_RATE,
        fps=FPS,
        intra_op_threads=THREADS or None,
    )


async def get_engine():
    """Lazy singleton. The lock also guards the one-time load."""
    global _engine
    async with _engine_lock:
        if _engine is None:
            _engine = await asyncio.to_thread(_load_engine)
    return _engine


def preload() -> None:
    """Fire-and-forget warmup at app startup so the first conversation
    doesn't pay the model download + ONNX/Numba warmup."""

    async def _warm():
        try:
            await get_engine()
            logger.info("wav2arkit ready")
        except Exception as exc:  # noqa: BLE001 - availability is optional
            logger.warning("wav2arkit preload failed (lip-sync unavailable): %r", exc)

    asyncio.create_task(_warm())


@router.websocket("/api/wav2arkit")
async def wav2arkit_ws(ws: WebSocket):
    await ws.accept()
    try:
        engine = await get_engine()
    except Exception as exc:  # noqa: BLE001
        logger.error("wav2arkit unavailable: %r", exc)
        await ws.send_json({"type": "error", "message": "lip-sync model unavailable"})
        await ws.close()
        return
    await ws.send_json({"type": "ready"})

    buf = np.zeros(0, dtype=np.float32)
    next_frame = 0

    async def infer_and_send(chunk: np.ndarray) -> None:
        nonlocal next_frame
        async with _engine_lock:
            result, _ = await asyncio.to_thread(engine.infer_streaming, chunk, SAMPLE_RATE)
        expression = result.get("expression")
        if result.get("code") != 0 or expression is None or len(expression) == 0:
            if result.get("code") != 0:
                logger.warning("inference failed: %s", result.get("error"))
            return
        frames = [engine.weights_to_dict(w) for w in expression]
        await ws.send_json({"type": "frames", "start": next_frame, "weights": frames})
        next_frame += len(frames)

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")

            if kind == "audio":
                raw = base64.b64decode(msg.get("audio", ""))
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                buf = np.concatenate([buf, pcm]) if buf.size else pcm
                while buf.size >= BLOCK_SAMPLES:
                    chunk, buf = buf[:BLOCK_SAMPLES], buf[BLOCK_SAMPLES:]
                    await infer_and_send(chunk)

            elif kind == "reset":
                buf = np.zeros(0, dtype=np.float32)
                next_frame = 0

            elif kind == "flush":
                # Pad the tail to a whole block so its frames aren't dropped;
                # the zero padding just closes the mouth at end of turn.
                if buf.size > 0:
                    padded = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
                    padded[: buf.size] = buf[:BLOCK_SAMPLES]
                    buf = np.zeros(0, dtype=np.float32)
                    await infer_and_send(padded)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - log and drop the socket
        logger.warning("wav2arkit session error: %r", exc)
