"""Lazy, local EmbeddingGemma 2 inference using the official task conventions."""

import os
import threading
from pathlib import Path

import numpy as np

MODEL_ID = "google/embeddinggemma-2"


class Embedder:
    model_id = MODEL_ID

    def __init__(self, device: str | None = None):
        self.device = device or os.environ.get("SEMFS_DEVICE")
        self._model = None
        self._lock = threading.RLock()

    def _load(self):
        # All heavy imports stay here: ordinary file management never imports torch.
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            dtype = (
                torch.bfloat16
                if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else torch.float32
            )
            try:
                self._model = SentenceTransformer(
                    self.model_id,
                    device=device,
                    config_kwargs={"audio_config": None},
                    model_kwargs={"torch_dtype": dtype},
                )
            except torch.cuda.OutOfMemoryError:
                # Some laptops expose CUDA but do not have enough free GPU memory.
                torch.cuda.empty_cache()
                self._model = SentenceTransformer(
                    self.model_id,
                    device="cpu",
                    config_kwargs={"audio_config": None},
                    model_kwargs={"torch_dtype": torch.float32},
                )
        return self._model

    def _encode(self, inputs, **kwargs) -> np.ndarray:
        with self._lock:
            try:
                result = self._load().encode(
                    inputs,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=1,
                    **kwargs,
                )
            except Exception as error:
                raise RuntimeError(
                    f"EmbeddingGemma 2 inference failed: {error}. "
                    "Check the first-download internet connection and available memory. "
                    "Set SEMFS_DEVICE=cpu to force CPU inference."
                ) from error
        vectors = np.asarray(result, dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Model produced non-finite embeddings")
        return vectors

    def query(self, text: str) -> np.ndarray:
        return self._encode(text, prompt_name="SearchQuery").reshape(-1)

    def documents(self, texts: list[str], title: str) -> np.ndarray:
        # Titled documents must be formatted manually; Document adds 'title: none'.
        documents = [f"title: {title} | text: {text}" for text in texts]
        return self._encode(documents, prompt="")

    def image(self, path: Path) -> np.ndarray:
        # Native vision input. No captions and no text-task prefix.
        vector = self._encode({"image": str(path)}, prompt="")
        return vector.reshape(1, -1)
