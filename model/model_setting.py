from enum import Enum

from .mistral import MistralSettings
from .gemma import GemmaSettings


class ModelType(Enum):
    MISTRAL = "mistral"
    GEMMA = "gemma"


SUPPORTED_MODELS = {
    ModelType.GEMMA.value: GemmaSettings,
    ModelType.MISTRAL.value: MistralSettings,
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_setting(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings