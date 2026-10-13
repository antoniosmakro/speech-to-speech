"""
Pipeline entrypoint: the stock speech-to-speech CLI with three handlers
swapped, so the session's voice drives the conversation's language:

- Pocket TTS serves every voice in the catalog (pocket_voices.py) — Kyutai's
  English model plus any POCKET_MODELS — and follows the session's voice turn
  by turn, so the Settings voice selector works mid-conversation. (The stock
  handler loads one voice at launch and ignores the session.)
- STT transcribes with the Parakeet fine-tune for the session voice's
  language (PARAKEET_MODELS), else the base Parakeet, keeping one of them in
  memory at a time. The base model picks the language on its own and on short
  Greek phrases guesses English ("Γεια σου" -> "Yeah"); the Greek fine-tune
  in turn no longer understands English.
- VAD records each pipeline's session, since STT only receives bare audio,
  and tells STT when the voice changes so the swap starts right away.

Arguments are the CLI's own; start.sh passes them.
"""

import ctypes
import gc
import logging
import os
import threading
from time import perf_counter

import numpy as np
from speech_to_speech import s2s_pipeline
from speech_to_speech.LLM import chat_completions_language_model
from speech_to_speech.pipeline.messages import EndOfResponse
from speech_to_speech.s2s_pipeline import main
from speech_to_speech.STT import parakeet_tdt_handler
from speech_to_speech.TTS import pocket_tts_handler
from speech_to_speech.VAD.vad_handler import VADHandler

import pocket_voices
from pocket_voices import BASE_STT

logger = logging.getLogger(__name__)

# Language codes by the English name pocket_voices uses.
LANGUAGE_CODES = {
    "Bulgarian": "bg", "Croatian": "hr", "Czech": "cs", "Danish": "da", "Dutch": "nl",
    "Estonian": "et", "Finnish": "fi", "French": "fr", "German": "de", "Greek": "el",
    "Hungarian": "hu", "Italian": "it", "Latvian": "lv", "Lithuanian": "lt", "Maltese": "mt",
    "Polish": "pl", "Portuguese": "pt", "Romanian": "ro", "Russian": "ru", "Slovak": "sk",
    "Slovenian": "sl", "Spanish": "es", "Swedish": "sv", "Ukrainian": "uk",
}

# Parakeet relabels any language code outside this list with the previous
# turn's; it lacks Greek and Maltese, which the model does support.
parakeet_tdt_handler.SUPPORTED_LANGUAGES.extend(
    code for code in LANGUAGE_CODES.values() if code not in parakeet_tdt_handler.SUPPORTED_LANGUAGES
)

# Fine-tunes live in plain folders: onnxruntime can't follow the HF cache's
# symlinks to the external weights file. The Dockerfile bakes them in.
PARAKEET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parakeet_models")
PARAKEET_FILES = ["config.json", "vocab.txt", "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"]
# A turn spoken while its model loads waits this long at most, then goes to
# whichever model is loaded.
LOAD_WAIT_S = 90

# The session each pipeline's VAD saw last. A pipeline serves one
# conversation at a time, so that's the one its STT is transcribing.
_sessions = {}
_stt_handlers = []


def _session_voice(runtime_config):
    output = runtime_config.session.audio.output if runtime_config else None
    return str(output.voice) if output is not None and output.voice else None


class SessionTrackingVADHandler(VADHandler):
    def process(self, audio_chunk):
        if isinstance(audio_chunk, tuple):
            _sessions[self.pipeline_index] = audio_chunk[1]
            voice = _session_voice(audio_chunk[1])
            if voice != getattr(self, "_voice", None):
                self._voice = voice
                for stt in _stt_handlers:
                    if stt.pipeline_index == self.pipeline_index:
                        stt.request(stt.model_for(voice))
        yield from super().process(audio_chunk)


class SwappingSTTHandler(parakeet_tdt_handler.ParakeetTDTSTTHandler):
    """One STT model in memory: the base Parakeet, or the fine-tune for the
    session voice's language. A voice change loads the new model in the
    background; it replaces the old one once ready, and a turn spoken before
    then waits for it."""

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)  # loads the base Parakeet
        catalog = pocket_voices.catalog()
        self._repos = pocket_voices.parakeet_models()
        self._model_for_voice = {v["id"]: pocket_voices.stt_model(v) for v in catalog}
        self._codes = {v["model"]: LANGUAGE_CODES.get(v["language"]) for v in catalog if v["model"]}
        self._extra = None  # the loaded fine-tune, while one is active
        self._active = BASE_STT
        self._loading = None
        self._wanted = BASE_STT
        self._failed = set()
        self._state = threading.Condition()
        self._in_use = threading.Lock()  # held while a turn is transcribed
        _stt_handlers.append(self)
        self._write_status()
        self.request(self.model_for(pocket_voices.default_voice()))

    def model_for(self, voice):
        """Unknown voices are spoken by the default voice, so they get its model."""
        default = self._model_for_voice.get(pocket_voices.default_voice(), BASE_STT)
        return self._model_for_voice.get(voice, default)

    def request(self, name):
        """Start loading `name` unless it's loaded, loading, or failed before."""
        with self._state:
            self._wanted = name
            if name == self._active or self._loading is not None or name in self._failed:
                return
            self._loading = name
            self._write_status()
        threading.Thread(target=self._load, args=(name,), name=f"stt-load-{name}", daemon=True).start()

    def _load(self, name):
        logger.info("Loading STT model %r", name)
        start = perf_counter()
        try:
            model = self._open(name)
        except Exception as exc:
            logger.warning("STT model %r unavailable, keeping %r: %r", name, self._active, exc)
            model = None
        with self._in_use, self._state:
            if model is not None:
                previous = self._active
                self._install(name, model)
                logger.info("STT model %r ready in %.1fs, %r unloaded", name, perf_counter() - start, previous)
            else:
                self._failed.add(name)
            self._loading = None
            self._write_status()
            self._state.notify_all()
            follow = self._wanted if self._wanted != self._active and self._wanted not in self._failed else None
        del model
        gc.collect()
        # glibc keeps freed heap for reuse; without this the unloaded model's
        # ~2 GB stays resident and every swap adds more.
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        if follow is not None:
            self.request(follow)

    def _open(self, name):
        if name == BASE_STT:
            from nano_parakeet import from_pretrained

            model = from_pretrained(model_name=self.model_name, device=self.device)
            model.transcribe(np.zeros(16000, np.float32))
            return model
        import onnx_asr
        import onnxruntime as ort

        path = os.path.join(PARAKEET_DIR, name)
        if not all(os.path.exists(os.path.join(path, f)) for f in PARAKEET_FILES):
            from huggingface_hub import snapshot_download

            snapshot_download(self._repos[name], local_dir=path, allow_patterns=PARAKEET_FILES)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        model = onnx_asr.load_model("nemo-conformer-tdt", path, sess_options=options)
        model.recognize(np.zeros(16000, np.float32), sample_rate=16000)
        return model

    def _install(self, name, model):
        """Swap `model` in; the previous one has no references left after this."""
        if name == BASE_STT:
            self.model, self._extra = model, None
        else:
            self.model, self._extra = None, model
        if self.streaming_handler is not None:
            self.streaming_handler.model = self.model
        self._active = name

    def _write_status(self):
        pocket_voices.write_stt_status({"active": self._active, "loading": self._loading, "failed": sorted(self._failed)})

    def process(self, vad_audio):
        name = self.model_for(_session_voice(_sessions.get(self.pipeline_index)))
        self.request(name)
        # Live captions come from the base model only; a fine-tune captions
        # once, at the end of the turn.
        progressive = vad_audio.mode == "progressive"
        if progressive and (name != BASE_STT or self._active != BASE_STT):
            return
        with self._state:
            self._state.wait_for(lambda: self._active == name or name in self._failed, timeout=LOAD_WAIT_S)
        with self._in_use:
            if progressive and self._active != BASE_STT:
                return
            outputs = list(super().process(vad_audio))
        yield from outputs

    def _process_nano_parakeet(self, audio_input):
        if self._active == BASE_STT:
            return super()._process_nano_parakeet(audio_input)
        # VAD cuts segments tight to the speech; the fine-tune was measured
        # with half a second of silence either side.
        pad = np.zeros(8000, np.float32)
        text = self._extra.recognize(np.concatenate([pad, audio_input, pad]), sample_rate=16000).strip()
        code = self._codes.get(self._active)
        if code == "el":
            text = text.replace("?", ";")  # trained on "?"; Greek's question mark is ";"
        return text, code or self.last_language


class CatalogPocketTTSHandler(pocket_tts_handler.PocketTTSHandler):
    def setup(self, should_listen, voice: str = "alba", **kwargs) -> None:
        # The stock setup loads Kyutai's English model, so it needs an
        # English voice; a catalog id like "greek/eleni" is resolved below.
        super().setup(should_listen, voice=voice if "/" not in voice else "alba", **kwargs)
        from pocket_tts import TTSModel

        english = self.model
        models = {None: english}
        self._voices = {}
        # Every voice is primed now (a voice state computed mid-turn would
        # delay the reply by seconds). A model or voice that fails to load
        # drops out; the rest keep serving.
        for v in pocket_voices.catalog():
            if v["model"] not in models:
                models[v["model"]] = self._load_model(TTSModel, v["model"], v["config"])
            model = models[v["model"]]
            if model is None:
                continue
            try:
                if model is english and v["prompt"] == self.voice:
                    state = self.voice_state
                else:
                    state = model.get_state_for_audio_prompt(v["prompt"])
            except Exception as exc:
                logger.warning("Pocket voice %r unavailable: %r", v["id"], exc)
                continue
            self._voices[v["id"]] = (model, state)
        self._default = self._voices.get(voice, (english, self.voice_state))
        self.model, self.voice_state = self._default
        self._default_id = self._current = voice
        self._unknown = set()
        logger.info("Pocket TTS serving %d voices, default %r", len(self._voices), voice)

    def _load_model(self, TTSModel, name, config):
        logger.info("Loading Pocket TTS model %r from %s", name, config)
        try:
            model = TTSModel.load_model(config=config)
        except Exception as exc:
            logger.warning("Pocket model %r unavailable: %r", name, exc)
            return None
        return model if self.device == "cpu" else model.to(self.device)

    def process(self, tts_input):
        if not isinstance(tts_input, EndOfResponse):
            self._select_voice(tts_input)
        yield from super().process(tts_input)

    def _select_voice(self, tts_input) -> None:
        """Response-level voice first, then the session's; unknown ids (e.g.
        a Kokoro voice saved in an old browser) fall back to the default."""
        voice = None
        response = tts_input.response
        if response and response.audio and response.audio.output and response.audio.output.voice:
            voice = str(response.audio.output.voice)
        if not voice:
            voice = _session_voice(tts_input.runtime_config)
        if voice not in self._voices:
            if voice and voice not in self._unknown:
                self._unknown.add(voice)
                logger.info("Unknown Pocket voice %r — using the default", voice)
            voice = self._default_id
        if voice != self._current:
            logger.info("Pocket voice -> %s", voice)
            self._current = voice
        self.model, self.voice_state = self._voices.get(voice, self._default)


_llm_warmup = chat_completions_language_model.ChatCompletionsApiModelHandler.warmup


def _llm_warmup_or_warn(self) -> None:
    """A slow or unreachable LLM at boot must not stop the pipeline: HF
    restarts a crashed app a few times, then parks it in "Runtime error" until
    someone restarts it by hand, long after the LLM has recovered."""
    try:
        _llm_warmup(self)
    except Exception as exc:
        logger.warning("LLM warmup failed, starting anyway: %r", exc)


chat_completions_language_model.ChatCompletionsApiModelHandler.warmup = _llm_warmup_or_warn
pocket_tts_handler.PocketTTSHandler = CatalogPocketTTSHandler
parakeet_tdt_handler.ParakeetTDTSTTHandler = SwappingSTTHandler
s2s_pipeline.VADHandler = SessionTrackingVADHandler

if __name__ == "__main__":
    main()
