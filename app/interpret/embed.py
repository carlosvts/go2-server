"""Embeddings de frase. Só o codificador: nenhum modelo generativo roda aqui."""

import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

# Prefixo que cada família de modelo espera no texto. Os modelos e5 foram
# treinados com "query: "; sem ele a similaridade perde qualidade.
PREFIXES = {"e5": "query: "}


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Uma linha por texto, com norma 1 (o produto interno é o cosseno)."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, threads: int | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if threads:
            torch.set_num_threads(threads)
        self._model = SentenceTransformer(model_name, device="cpu")
        self._prefix = next((p for key, p in PREFIXES.items() if key in model_name.lower()), "")
        # O tokenizador rápido da Hugging Face não aceita uso simultâneo.
        self._lock = threading.Lock()
        self._one = lru_cache(maxsize=2048)(self._encode_one)

    def _encode_one(self, text: str) -> np.ndarray:
        with self._lock:
            vector = self._model.encode(
                [self._prefix + text], normalize_embeddings=True, show_progress_bar=False
            )[0]
        return np.asarray(vector, dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(text) for text in texts])
