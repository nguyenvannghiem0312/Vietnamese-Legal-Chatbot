from abc import ABC ### Abstract Base Classes
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedder(ABC):
    embedder: Any

    def get_embedding(self):
        return self.embedder


class EmbedderHuggingFace(Embedder):
    def __init__(self,
                model_name: str = "NghiemAbe/Vi-Legal-Roberta-grad-cache-256-gist",
                ):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)