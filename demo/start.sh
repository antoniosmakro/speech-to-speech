#!/bin/bash
# Single-container entrypoint: the speech-to-speech pipeline (internal :8765)
# plus the demo app (public :7860). HF Spaces route traffic to 7860 only, so
# the browser reaches the pipeline through the same-origin /v1/realtime
# passthrough (s2sproxy.py), and the pipeline reaches the LLM through the
# visitor-billed passthrough (llmproxy.py) — the web app must therefore be
# up before the pipeline warms.
set -e

: "${LLM_MODEL:=Qwen/Qwen3-VL-30B-A3B-Instruct}"
: "${LLM_UPSTREAM_URL:=https://router.huggingface.co/v1}"
# The pipeline talks to the local passthrough, which swaps in the signed-in
# visitor's token (their HF credits) or falls back to HF_TOKEN (warmup, dev).
: "${LLM_BASE_URL:=http://127.0.0.1:7860/llmproxy/v1}"
: "${NUM_PIPELINES:=1}"
# TTS engine: kokoro or pocket (Rust-native inference — immune to the GIL
# starvation that slows Kokoro under pipeline load on CPU).
: "${TTS_ENGINE:=pocket}"
: "${POCKET_VOICE:=eponine}"
# Extra Pocket TTS models beside Kyutai's English one: comma-separated
# name=<config yaml> entries (see pocket_voices.py). Their voices join the
# Settings voice selector.
: "${POCKET_MODELS:=greek=hf://myned-ai/pocket-tts-greek/greek.yaml}"
export TTS_ENGINE POCKET_VOICE POCKET_MODELS
TTS_FLAGS=""
[ "$TTS_ENGINE" = "pocket" ] && TTS_FLAGS="--pocket_tts_voice $POCKET_VOICE --pocket_tts_device cpu"

# Cap math-library thread pools: oversubscribing hybrid/hyperthreaded CPUs
# slows CPU inference (Kokoro measured fastest at 4 threads on 20-vCPU host).
: "${TORCH_THREADS:=4}"
export OMP_NUM_THREADS="$TORCH_THREADS" MKL_NUM_THREADS="$TORCH_THREADS" OPENBLAS_NUM_THREADS="$TORCH_THREADS"
# wav2arkit ONNX cap (lipsync.py reads it; 12.8x realtime on 2 threads).
: "${WAV2ARKIT_THREADS:=2}"
export WAV2ARKIT_THREADS
# Sentences per TTS batch. Upstream default (3) holds short replies until the
# LLM finishes entirely — the avatar can't start speaking. 1 = speak after the
# first sentence.
: "${STREAM_BATCH_SENTENCES:=1}"
# Live user captions cost periodic STT bursts while speaking; 0 to disable.
: "${LIVE_TRANSCRIPTION:=1}"
LIVE_FLAG=""
[ "$LIVE_TRANSCRIPTION" = "1" ] && LIVE_FLAG="--enable_live_transcription"

# On a Space, only signed-in visitors may talk (their token pays for the LLM);
# locally anyone can (HF_TOKEN pays). Override with REQUIRE_LOGIN=0/1.
if [ -z "$REQUIRE_LOGIN" ]; then
  [ -n "$SPACE_HOST" ] && REQUIRE_LOGIN=1 || REQUIRE_LOGIN=0
fi
export REQUIRE_LOGIN LLM_UPSTREAM_URL

# In-server conversation queue (slotqueue.py): one visitor per pipeline talks,
# the next QUEUE_MAX (default 10) wait in line with live positions, beyond
# that the page says come back later. QUEUE_ENABLED=0 restores direct mode.
: "${QUEUE_ENABLED:=1}"
export QUEUE_ENABLED NUM_PIPELINES

if [ -z "$HF_TOKEN" ]; then
  echo "WARNING: HF_TOKEN is not set — warmup/fallback LLM calls will fail" >&2
fi

# Browser connects same-origin; the proxy pumps to the internal server.
export S2S_INTERNAL_WS="ws://127.0.0.1:8765/v1/realtime"
if [ -n "$SPACE_HOST" ]; then
  export SPEECH_TO_SPEECH_URL="wss://${SPACE_HOST}/v1/realtime"
else
  export SPEECH_TO_SPEECH_URL="${SPEECH_TO_SPEECH_URL:-ws://localhost:7860/v1/realtime}"
fi

uvicorn server:app --host 0.0.0.0 --port 7860 &
WEB_PID=$!
trap 'kill $WEB_PID 2>/dev/null' EXIT

# The pipeline's LLM warmup goes through the passthrough — wait for the app.
python - <<'PY'
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:7860/api/config", timeout=2)
        break
    except Exception:
        time.sleep(0.5)
PY

# run_pipeline.py = the speech-to-speech CLI with the multi-voice Pocket handler.
exec python run_pipeline.py \
  --mode realtime \
  --device cpu \
  --stt parakeet-tdt \
  --llm_backend chat-completions \
  --tts "$TTS_ENGINE" \
  $TTS_FLAGS \
  --model_name "$LLM_MODEL" \
  --responses_api_base_url "$LLM_BASE_URL" \
  --responses_api_api_key "${HF_TOKEN:-not-needed}" \
  --responses_api_stream \
  --stream_batch_sentences "$STREAM_BATCH_SENTENCES" \
  $LIVE_FLAG \
  --num_pipelines "$NUM_PIPELINES"
