# Single container: serves the static front-end AND the /api/search proxy.
# HF Spaces (sdk: docker) routes traffic to $PORT, default 7860.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Wav2Arkit lip-sync for the avatar (audio -> 52 ARKit blendshapes, ONNX CPU).
# The module is fetched from myned-ai/avatar-chat-server (MIT) at a pinned ref
# — linked, not copied — and the model weights are baked into the image so the
# Space boots warm. See lipsync.py for the route it powers.
ARG WAV2ARKIT_REF=feat/wav2arkit-standalone
ADD https://github.com/myned-ai/avatar-chat-server/archive/refs/heads/${WAV2ARKIT_REF}.tar.gz /tmp/acs.tar.gz
RUN tar -xzf /tmp/acs.tar.gz -C /tmp \
    && cp -r /tmp/avatar-chat-server-*/src/wav2arkit /app/wav2arkit \
    && rm -rf /tmp/acs.tar.gz /tmp/avatar-chat-server-* \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('myned-ai/wav2arkit_cpu', local_dir='pretrained_models')"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
