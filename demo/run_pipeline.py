"""
Pipeline entrypoint: the stock speech-to-speech CLI with three handlers
swapped, so the session's voice drives the conversation's language:

- Pocket TTS serves every voice in the catalog (pocket_voices.py) — Kyutai's
  English model plus any POCKET_MODELS — and follows the session's voice turn
  by turn, so the Settings voice selector works mid-conversation. (The stock
  handler loads one voice at launch and ignores the session.)
- STT transcribes with Canary-1b-v2, told the language, when the session's
  voice speaks one Canary knows (Greek, ...). Parakeet picks the language on
  its own, and on short phrases it guesses English: "Γεια σου" -> "Yeah".
- VAD records each pipeline's session, since STT only receives bare audio.

Arguments are the CLI's own; start.sh passes them.
"""

import logging
import os

import numpy as np
from speech_to_speech import s2s_pipeline
from speech_to_speech.pipeline.messages import EndOfResponse
from speech_to_speech.s2s_pipeline import main
from speech_to_speech.STT import parakeet_tdt_handler
from speech_to_speech.TTS import pocket_tts_handler
from speech_to_speech.VAD.vad_handler import VADHandler

import pocket_voices

logger = logging.getLogger(__name__)

# Canary-1b-v2's languages, by the English name pocket_voices uses. English
# stays with Parakeet.
CANARY_LANGUAGES = {
    "Bulgarian": "bg", "Croatian": "hr", "Czech": "cs", "Danish": "da", "Dutch": "nl",
    "Estonian": "et", "Finnish": "fi", "French": "fr", "German": "de", "Greek": "el",
    "Hungarian": "hu", "Italian": "it", "Latvian": "lv", "Lithuanian": "lt", "Maltese": "mt",
    "Polish": "pl", "Portuguese": "pt", "Romanian": "ro", "Russian": "ru", "Slovak": "sk",
    "Slovenian": "sl", "Spanish": "es", "Swedish": "sv", "Ukrainian": "uk",
}

# Parakeet relabels any language code outside this list with the previous
# turn's; it lacks Greek and Maltese, which the model does support.
parakeet_tdt_handler.SUPPORTED_LANGUAGES.extend(
    code for code in CANARY_LANGUAGES.values() if code not in parakeet_tdt_handler.SUPPORTED_LANGUAGES
)

# The session each pipeline's VAD saw last. A pipeline serves one
# conversation at a time, so that's the one its STT is transcribing.
_sessions = {}


def _session_voice(runtime_config):
    output = runtime_config.session.audio.output if runtime_config else None
    return str(output.voice) if output is not None and output.voice else None


class SessionTrackingVADHandler(VADHandler):
    def process(self, audio_chunk):
        if isinstance(audio_chunk, tuple):
            _sessions[self.pipeline_index] = audio_chunk[1]
        yield from super().process(audio_chunk)


def _load_canary():
    """Canary-1b-v2, int8 ONNX via onnx-asr: ~0.45 s per short phrase on 4
    threads, about Parakeet's speed. None if it can't load (logged); every
    turn then goes to Parakeet."""
    try:
        import onnx_asr
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        model = onnx_asr.load_model("nemo-canary-1b-v2", quantization="int8", sess_options=options)
        model.recognize(np.zeros(16000, np.float32), sample_rate=16000, language="en")
        return model
    except Exception as exc:
        logger.warning("Canary unavailable, Parakeet transcribes every turn: %r", exc)
        return None


class LanguageForcingSTTHandler(parakeet_tdt_handler.ParakeetTDTSTTHandler):
    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self._voice_languages = {v["id"]: CANARY_LANGUAGES.get(v["language"]) for v in pocket_voices.catalog()}
        self._canary = _load_canary() if any(self._voice_languages.values()) else None
        self._forced = None
        if self._canary:
            logger.info("Canary STT ready for %s", sorted({c for c in self._voice_languages.values() if c}))

    def process(self, vad_audio):
        voice = _session_voice(_sessions.get(self.pipeline_index))
        self._forced = self._voice_languages.get(voice) if self._canary else None
        # Live captions come from Parakeet, which would caption in whatever
        # language it guesses; forced turns caption once, from Canary.
        if self._forced and vad_audio.mode == "progressive":
            return
        yield from super().process(vad_audio)

    def _process_nano_parakeet(self, audio_input):
        if not self._forced:
            return super()._process_nano_parakeet(audio_input)
        # Canary misreads speech cut right at its edges, as VAD segments are;
        # half a second of silence either side fixes most of it (measured:
        # "Καίμερα" -> "Καλημέρα", "Ουίση" -> "Πού είσαι").
        pad = np.zeros(8000, np.float32)
        audio = np.concatenate([pad, audio_input, pad])
        text = self._canary.recognize(audio, sample_rate=16000, language=self._forced)
        return text.strip(), self._forced


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


pocket_tts_handler.PocketTTSHandler = CatalogPocketTTSHandler
parakeet_tdt_handler.ParakeetTDTSTTHandler = LanguageForcingSTTHandler
s2s_pipeline.VADHandler = SessionTrackingVADHandler

if __name__ == "__main__":
    main()
