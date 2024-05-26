from pathlib import Path
from typing import Any, Iterator, Union
import torch

from vllm import LLM, SamplingParams

from .llm_client import LlmClient, LlmClientType
from model.model import Model


class VLLMClient(LlmClient):
    def __init__(self, model_folder: Path, model_settings: Model):
        if LlmClientType.VLMM not in model_settings.clients:
            raise ValueError(
                f"{model_settings.file_name} is a not supported by the {LlmClientType.VLMM.value} client."
            )
        super().__init__(model_folder, model_settings)

    def _load_llm(self) -> Any:
        llm = LLM(model=str(self.model_path), quantization="AWQ", dtype=torch.float16, gpu_memory_utilization=0.5, enforce_eager=True, max_model_len=2048)
        return llm

    def _load_tokenizer(self) -> Any:
        return None

    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer based on the given prompt using the language model.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 512).

        Returns:
            str: The generated answer.
        """
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        prompt_none_pyvi = prompt.replace('_', ' ')
        outputs = self.llm.generate(prompt_none_pyvi, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
        answer = generated_text

        return answer

    async def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer based on the given prompt using the language model.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 512).

        Returns:
            str: The generated answer.
        """
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        prompt_none_pyvi = prompt.replace('_', ' ')
        outputs = self.llm.generate(prompt_none_pyvi, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
        answer = generated_text

        return answer

    def stream_answer(self, prompt: str, skip_prompt: bool = True, max_new_tokens: int = 512) -> str:
        """
        Generates an answer by streaming tokens using the TextStreamer.

        Args:
            prompt (str): The input prompt for generating the answer.
            skip_prompt (bool): Whether to skip the prompt tokens during streaming (default is True).
            max_new_tokens (int): The maximum number of new tokens to generate (default is 512).

        Returns:
            str: The generated answer.
        """
        return None

    def start_answer_iterator_streamer(
        self, prompt: str, skip_prompt: bool = True, max_new_tokens: int = 512
    ):
        return None

    async def async_start_answer_iterator_streamer(
        self, prompt: str, skip_prompt: bool = True, max_new_tokens: int = 512
    ):
       return None

    def parse_token(self, token):
        return token