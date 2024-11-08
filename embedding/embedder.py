from abc import ABC ### Abstract Base Classes
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedder(ABC):
    embedder: Any

    def get_embedding(self):
        return self.embedder


class EmbedderHuggingFace(Embedder):
    def __init__(self,

                model_name: str = "NghiemAbe/Vi-Legal-Bi-Encoder",
                # model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
                ):

        self.embedder = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'token': 'hf_PU...'})
