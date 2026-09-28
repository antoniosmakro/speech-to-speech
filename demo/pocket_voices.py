"""
Pocket TTS voice catalog.

Besides Kyutai's built-in English model, the pipeline can serve extra Pocket
TTS models, listed in POCKET_MODELS as comma-separated `name=<config yaml>`
entries:

    POCKET_MODELS="greek=hf://myned-ai/pocket-tts-greek/greek.yaml"

An extra model's voices come from `voices/voices.json` next to its config
(the layout of myned-ai/pocket-tts-greek), each voice's audio prompt from
`voices/<name>.wav` beside it. Voice ids are `<model>/<voice>` ("greek/eleni");
built-in English voices keep their bare preset names ("eponine"). The model
name doubles as the language the voice speaks ("greek" -> "Greek").

Two readers: run_pipeline.py loads every model and primes every voice, and
/api/config hands the same list to the Settings voice selector.
"""

import functools
import json
import logging
import os

logger = logging.getLogger("s2s.voices")

# Kyutai's classic English presets, all resolved by name by pocket-tts.
ENGLISH_VOICES = [
    ("alba", "female"),
    ("azelma", "female"),
    ("cosette", "female"),
    ("eponine", "female"),
    ("fantine", "female"),
    ("javert", "male"),
    ("jean", "male"),
    ("marius", "male"),
]


def default_voice() -> str:
    return os.environ.get("POCKET_VOICE", "eponine").strip()


def parse_models(spec: str) -> list[tuple[str, str]]:
    """`"greek=hf://org/repo/greek.yaml, ..."` -> [("greek", "hf://org/repo/greek.yaml"), ...]"""
    models = []
    for entry in spec.split(","):
        name, sep, config = entry.strip().partition("=")
        if not entry.strip():
            continue
        if not sep or not name.strip() or not config.strip():
            logger.warning("POCKET_MODELS: ignoring malformed entry %r", entry.strip())
            continue
        models.append((name.strip().lower(), config.strip()))
    return models


def _sibling(config: str, rel: str) -> str:
    """Path of `rel` in the config's directory, same form as the config
    (hf://org/repo/dir/file.yaml[@rev] or a local path)."""
    if config.startswith("hf://"):
        body, _, rev = config[len("hf://"):].partition("@")
        base = body.rsplit("/", 1)[0]
        return f"hf://{base}/{rel}" + (f"@{rev}" if rev else "")
    return os.path.join(os.path.dirname(config), rel)


def _read_json(path: str):
    if path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        body, _, rev = path[len("hf://"):].partition("@")
        org, repo, filename = body.split("/", 2)
        path = hf_hub_download(f"{org}/{repo}", filename, revision=rev or None)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@functools.cache
def catalog() -> tuple[dict, ...]:
    """Every selectable voice: {id, name, description, language, model, config,
    prompt}. `model`/`config` are None for the built-in English voices; `prompt`
    is what pocket-tts' get_state_for_audio_prompt takes. A model whose voice
    manifest can't be read is skipped (logged), so the rest still serve."""
    voices = [
        {
            "id": name,
            "name": name.capitalize(),
            "description": gender,
            "language": "English",
            "model": None,
            "config": None,
            "prompt": name,
        }
        for name, gender in ENGLISH_VOICES
    ]
    for model, config in parse_models(os.environ.get("POCKET_MODELS", "")):
        try:
            manifest = _read_json(_sibling(config, "voices/voices.json"))
        except Exception as exc:
            logger.warning("Pocket model %r: no voice manifest (%r) — skipped", model, exc)
            continue
        language = model.replace("_", " ").title()
        for entry in manifest:
            name = entry["name"]
            filename = entry.get("file") or f"{name.lower()}.wav"
            voices.append(
                {
                    "id": f"{model}/{name.lower()}",
                    "name": name,
                    "description": entry.get("description", ""),
                    "language": language,
                    "model": model,
                    "config": config,
                    "prompt": _sibling(config, f"voices/{filename}"),
                }
            )
    return tuple(voices)
