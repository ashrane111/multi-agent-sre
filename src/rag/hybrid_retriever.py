"""
Hybrid retriever combining BM25 sparse retrieval with dense semantic search.

Implements Reciprocal Rank Fusion (RRF) to combine results from both methods.
"""

import re
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from src.config import settings
from src.rag.document_processor import DocumentChunk
from src.rag.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining sparse (BM25) and dense (semantic) retrieval.
    
    Features:
    - BM25 for keyword matching
    - Dense embeddings for semantic similarity
    - Reciprocal Rank Fusion (RRF) for score combination
    - Configurable weights for each method
    """

    def __init__(
        self,
        vector_store: VectorStore,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for dense retrieval
            dense_weight: Weight for dense retrieval scores (0-1)
            sparse_weight: Weight for sparse retrieval scores (0-1)
        """
        self._vector_store = vector_store
        self._dense_weight = dense_weight or settings.rag.dense_weight
        self._sparse_weight = sparse_weight or settings.rag.sparse_weight
        self._logger = logger.bind(component="hybrid_retriever")

        # BM25 index (built lazily)
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict[str, Any]] = []
        self._bm25_tokenized: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens

    def build_bm25_index(self, chunks: list[DocumentChunk]) -> None:
        """
        Build BM25 index from document chunks.
        
        Args:
            chunks: List of document chunks to index
        """
        self._logger.info("building_bm25_index", chunk_count=len(chunks))

        self._bm25_corpus = [
            {
                "id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        self._bm25_tokenized = [
            self._tokenize(chunk.content) for chunk in chunks
        ]

        self._bm25 = BM25Okapi(self._bm25_tokenized)

        self._logger.info(
            "bm25_index_built",
            document_count=len(self._bm25_corpus),
            avg_doc_length=self._bm25.avgdl,
        )

    def rebuild_bm25_from_vector_store(self) -> None:
        """Rebuild BM25 index from vector store contents."""
        self._logger.info("rebuilding_bm25_from_vector_store")

        # Get all documents from vector store
        results = self._vector_store.collection.get(
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            self._logger.warning("vector_store_empty")
            return

        self._bm25_corpus = []
        self._bm25_tokenized = []

        for i in range(len(results["ids"])):
            doc_id = results["ids"][i]
            content = results["documents"][i] if results["documents"] else ""
            metadata = results["metadatas"][i] if results["metadatas"] else {}

            self._bm25_corpus.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
            })
            self._bm25_tokenized.append(self._tokenize(content))

        self._bm25 = BM25Okapi(self._bm25_tokenized)

        self._logger.info(
            "bm25_index_rebuilt",
            document_count=len(self._bm25_corpus),
        )

    def _sparse_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Perform BM25 sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of results with BM25 scores
        """
        if self._bm25 is None or not self._bm25_corpus:
            self._logger.warning("bm25_index_not_built")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        scored_docs = list(enumerate(scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:top_k]

        results = []
        for idx, score in top_docs:
            if score > 0:  # Only include docs with positive scores
                doc = self._bm25_corpus[idx]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "bm25_score": float(score),
                })

        return results

    def _dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Perform dense semantic retrieval.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of results with similarity scores
        """
        results = self._vector_store.search(query, top_k=top_k)
        
        # Rename score field for clarity
        for r in results:
            r["dense_score"] = r.pop("score", 0.0)
        
        return results

    def _reciprocal_rank_fusion(
        self,
        results_lists: list[list[dict[str, Any]]],
        weights: list[float],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(weight * 1 / (k + rank))
        
        Args:
            results_lists: List of result lists from different retrievers
            weights: Weights for each result list
            k: RRF constant (default 60)
            
        Returns:
            Combined and re-ranked results
        """
        fused_scores: dict[str, float] = {}
        doc_data: dict[str, dict[str, Any]] = {}

        for results, weight in zip(results_lists, weights):
            for rank, result in enumerate(results):
                doc_id = result["id"]
                
                # RRF score contribution
                rrf_score = weight * (1.0 / (k + rank + 1))
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score

                # Store document data (prefer first occurrence)
                if doc_id not in doc_data:
                    doc_data[doc_id] = result

        # Sort by fused score
        sorted_docs = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build final results
        results = []
        for doc_id, fused_score in sorted_docs:
            result = doc_data[doc_id].copy()
            result["fused_score"] = fused_score
            results.append(result)

        return results

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and dense retrieval.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            
        Returns:
            List of results with fused scores
        """
        top_k = top_k or settings.rag.top_k
        
        # Get more candidates from each method for better fusion
        candidate_k = top_k * 3

        self._logger.debug(
            "hybrid_search_starting",
            query_length=len(query),
            top_k=top_k,
        )

        # Sparse retrieval (BM25)
        sparse_results = self._sparse_search(query, candidate_k)
        
        # Dense retrieval (semantic)
        dense_results = self._dense_search(query, candidate_k)

        # Combine with RRF
        fused_results = self._reciprocal_rank_fusion(
            results_lists=[dense_results, sparse_results],
            weights=[self._dense_weight, self._sparse_weight],
        )

        # Return top-k
        final_results = fused_results[:top_k]

        self._logger.debug(
            "hybrid_search_complete",
            sparse_count=len(sparse_results),
            dense_count=len(dense_results),
            fused_count=len(final_results),
        )

        return final_results

    def search_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Perform hybrid search and return detailed scoring info.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Tuple of (results, scoring_info)
        """
        results = self.search(query, top_k)
        
        scoring_info = {
            "dense_weight": self._dense_weight,
            "sparse_weight": self._sparse_weight,
            "bm25_index_size": len(self._bm25_corpus),
            "query": query,
        }

        return results, scoring_info

    def stats(self) -> dict[str, Any]:
        """Get retriever statistics."""
        return {
            "dense_weight": self._dense_weight,
            "sparse_weight": self._sparse_weight,
            "bm25_index_size": len(self._bm25_corpus),
            "bm25_avg_doc_length": self._bm25.avgdl if self._bm25 else 0,
            "vector_store_size": self._vector_store.count,
        }
