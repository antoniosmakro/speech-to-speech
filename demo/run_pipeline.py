"""
Pipeline entrypoint: the stock speech-to-speech CLI, with the Pocket TTS
handler swapped for one that serves every voice in the catalog
(pocket_voices.py) — Kyutai's English model plus any POCKET_MODELS — and
follows the session's voice turn by turn, so the Settings voice selector
works mid-conversation. Arguments are the CLI's own; start.sh passes them.

The stock handler loads one model and one voice at launch and ignores the
session voice.
"""

import logging

from speech_to_speech.pipeline.messages import EndOfResponse
from speech_to_speech.s2s_pipeline import main
from speech_to_speech.TTS import pocket_tts_handler

import pocket_voices

logger = logging.getLogger(__name__)


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
        if not voice and tts_input.runtime_config:
            output = tts_input.runtime_config.session.audio.output
            voice = str(output.voice) if output is not None and output.voice else None
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

if __name__ == "__main__":
    main()
