from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmbeddingModel:
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for semantic embeddings. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
