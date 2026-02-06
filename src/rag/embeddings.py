"""
Embedding model management for RAG pipeline.

Uses sentence-transformers for local embedding generation.
Supports both CPU and GPU inference with automatic device selection.
"""

import hashlib
from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingManager:
    """
    Manages embedding model lifecycle and inference.
    
    Features:
    - Lazy model loading
    - Embedding caching
    - Batch processing
    - Automatic device selection (CPU/GPU)
    """

    _instance: "EmbeddingManager | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "EmbeddingManager":
        """Singleton pattern for embedding manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize embedding manager."""
        self._model_name = settings.llm.embedding_model
        self._cache: dict[str, list[float]] = {}
        self._logger = logger.bind(component="embedding_manager")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._logger.info(
                "loading_embedding_model",
                model_name=self._model_name,
            )
            self._model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,
            )
            self._logger.info(
                "embedding_model_loaded",
                model_name=self._model_name,
                device=str(self._model.device),
                embedding_dim=self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as list of floats
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        if use_cache:
            self._cache[cache_key] = embedding

        return embedding

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache for all texts
        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))

        # Embed texts not in cache
        if texts_to_embed:
            indices, uncached_texts = zip(*texts_to_embed)
            
            self._logger.debug(
                "embedding_texts",
                total=len(texts),
                cached=len(texts) - len(uncached_texts),
                to_embed=len(uncached_texts),
            )

            embeddings = self.model.encode(
                list(uncached_texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )

            for idx, embedding, text in zip(indices, embeddings, uncached_texts):
                embedding_list = embedding.tolist()
                results[idx] = embedding_list
                if use_cache:
                    self._cache[self._get_cache_key(text)] = embedding_list

        return results  # type: ignore

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query (may use different prompt template than documents).
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding vector
        """
        # Some models use different prefixes for queries vs documents
        # For now, we use the same embedding method
        return self.embed_text(query, use_cache=True)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        self._logger.info("cache_cleared", previous_size=cache_size)

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "model_name": self._model_name,
            "model_loaded": self._model is not None,
        }


@lru_cache(maxsize=1)
def get_embedding_manager() -> EmbeddingManager:
    """Get singleton embedding manager instance."""
    return EmbeddingManager()
