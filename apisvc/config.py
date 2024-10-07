from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class InferenceConfig:
    # model_path can refer to a HF hub model_id, a local path, or a Weights & Biases artifact
    #    fixie-ai/ultravox
    #    runs/llama2_asr_gigaspeech/checkpoint-1000/
    #    wandb://fixie/ultravox/model-llama2_asr_gigaspeech:v0
    model_path: str = "fixie-ai/ultravox-v0_3"
    device: Optional[str] = None
    data_type: Optional[str] = None
    default_prompt: str = ""
    max_new_tokens: int = 50
    temperature: float = 0


import dotenv
import structlog

log: structlog.stdlib.BoundLogger = structlog.get_logger()
_loaded = False


def get_config(key: str) -> str | None:
    """Returns the secret specified by key.
    Throws KeyError if optional is false and key isn't found."""
    global _loaded
    if not _loaded:
        dotenv.load_dotenv()
        _loaded = True

    return os.getenv(key.upper())


def get_config_bool(key: str) -> bool:
    return (get_config(key) or "").lower() == "true"


def get_required_config(key: str) -> str:
    """Returns the secret specified by key.
    Throws KeyError if optional is false and key isn't found."""
    global _loaded
    if not _loaded:
        dotenv.load_dotenv()
        _loaded = True

    try:
        return os.environ[key.upper()]
    except KeyError:
        log.exception("Could not find required config value", key=key)
        raise
