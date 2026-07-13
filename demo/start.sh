#!/bin/bash
# Single-container entrypoint: the speech-to-speech pipeline (internal :8765)
# plus the demo app (public :7860). HF Spaces route traffic to 7860 only, so
# the browser reaches the pipeline through the same-origin /v1/realtime
# passthrough (s2sproxy.py).
set -e

: "${LLM_MODEL:=Qwen/Qwen3-VL-30B-A3B-Instruct}"
: "${LLM_BASE_URL:=https://router.huggingface.co/v1}"
: "${NUM_PIPELINES:=1}"

# Cap math-library thread pools: oversubscribing hybrid/hyperthreaded CPUs
# slows CPU inference well below realtime (measured 4x TTS regression).
: "${TORCH_THREADS:=8}"
export OMP_NUM_THREADS="$TORCH_THREADS" MKL_NUM_THREADS="$TORCH_THREADS" OPENBLAS_NUM_THREADS="$TORCH_THREADS"

if [ -z "$HF_TOKEN" ]; then
  echo "WARNING: HF_TOKEN is not set — LLM calls will fail" >&2
fi

speech-to-speech \
  --mode realtime \
  --device cpu \
  --stt parakeet-tdt \
  --llm_backend chat-completions \
  --tts kokoro \
  --model_name "$LLM_MODEL" \
  --responses_api_base_url "$LLM_BASE_URL" \
  --responses_api_api_key "$HF_TOKEN" \
  --responses_api_stream \
  --enable_live_transcription \
  --num_pipelines "$NUM_PIPELINES" \
  &
S2S_PID=$!

# Browser connects same-origin; the proxy pumps to the internal server.
export S2S_INTERNAL_WS="ws://127.0.0.1:8765/v1/realtime"
if [ -n "$SPACE_HOST" ]; then
  export SPEECH_TO_SPEECH_URL="wss://${SPACE_HOST}/v1/realtime"
else
  export SPEECH_TO_SPEECH_URL="${SPEECH_TO_SPEECH_URL:-ws://localhost:7860/v1/realtime}"
fi

trap 'kill $S2S_PID 2>/dev/null' EXIT
exec uvicorn server:app --host 0.0.0.0 --port 7860
