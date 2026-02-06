from functools import lru_cache
from typing import Any

import structlog
from sentence_transformers import CrossEncoder

from src.config import settings

logger = structlog.get_logger(__name__)


class Reranker:
    """Cross-encoder based reranker for retrieval results."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model: CrossEncoder | None = None
        self._logger = logger.bind(component="reranker")

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            self._logger.info("loading_reranker_model", model_name=self._model_name)
            self._model = CrossEncoder(self._model_name, max_length=512)
            self._logger.info("reranker_model_loaded", model_name=self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        top_k = top_k or settings.rag.reranker_top_k
        pairs = [(query, doc.get(content_key, "")) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:top_k]


@lru_cache(maxsize=1)
def get_reranker(model_name: str | None = None) -> Reranker:
    """Get singleton reranker instance."""
    return Reranker(model_name)