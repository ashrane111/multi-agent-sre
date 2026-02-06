"""
RAG Pipeline orchestrating document processing, retrieval, and reranking.

Provides a high-level interface for the complete RAG workflow:
1. Document ingestion and chunking
2. Hybrid retrieval (BM25 + dense)
3. Cross-encoder reranking
4. Context formatting for LLM
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.config import settings
from src.rag.document_processor import Document, DocumentProcessor
from src.rag.embeddings import get_embedding_manager
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import get_reranker
from src.rag.vector_store import VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""

    query: str
    documents: list[dict[str, Any]]
    context: str
    metadata: dict[str, Any]


class RAGPipeline:
    """
    Complete RAG pipeline for runbook retrieval.
    
    Features:
    - Automatic document ingestion
    - Hybrid search (BM25 + semantic)
    - Cross-encoder reranking
    - Context formatting
    - Index management
    """

    def __init__(
        self,
        collection_name: str = "runbooks",
        runbooks_dir: str | None = None,
        auto_index: bool = True,
    ) -> None:
        """
        Initialize RAG pipeline.
        
        Args:
            collection_name: Name for the vector store collection
            runbooks_dir: Directory containing runbook documents
            auto_index: Whether to automatically index documents on init
        """
        self._collection_name = collection_name
        self._runbooks_dir = Path(runbooks_dir or "data/runbooks")
        self._logger = logger.bind(component="rag_pipeline")

        # Initialize components
        self._document_processor = DocumentProcessor()
        self._vector_store = VectorStore(collection_name=collection_name)
        self._hybrid_retriever = HybridRetriever(vector_store=self._vector_store)
        self._reranker = get_reranker()
        self._embedding_manager = get_embedding_manager()

        # Track indexed documents
        self._indexed_sources: set[str] = set(self._vector_store.get_all_sources())

        # Auto-index if requested and index is empty
        if auto_index and self._vector_store.count == 0:
            self._logger.info("auto_indexing_runbooks")
            self.index_directory(self._runbooks_dir)

        # Build BM25 index from existing documents
        if self._vector_store.count > 0:
            self._hybrid_retriever.rebuild_bm25_from_vector_store()

        self._logger.info(
            "rag_pipeline_initialized",
            collection_name=collection_name,
            document_count=self._vector_store.count,
            sources=list(self._indexed_sources),
        )

    def index_document(self, filepath: Path) -> Document:
        """
        Index a single document.
        
        Args:
            filepath: Path to the document
            
        Returns:
            Processed Document
        """
        self._logger.info("indexing_document", filepath=str(filepath))

        # Process document
        document = self._document_processor.process_file(filepath)

        # Remove old chunks if re-indexing
        if document.metadata.get("source") in self._indexed_sources:
            self._vector_store.delete_by_source(document.metadata["source"])

        # Add to vector store
        self._vector_store.add_chunks(document.chunks)

        # Update BM25 index
        self._hybrid_retriever.build_bm25_index(document.chunks)

        # Track source
        self._indexed_sources.add(document.metadata.get("source", ""))

        self._logger.info(
            "document_indexed",
            filepath=str(filepath),
            chunk_count=len(document.chunks),
        )

        return document

    def index_directory(self, directory: Path | str) -> list[Document]:
        """
        Index all documents in a directory.
        
        Args:
            directory: Directory containing documents
            
        Returns:
            List of processed Documents
        """
        directory = Path(directory)
        self._logger.info("indexing_directory", directory=str(directory))

        if not directory.exists():
            self._logger.warning("directory_not_found", directory=str(directory))
            return []

        # Process all documents
        documents = self._document_processor.process_directory(directory)

        # Collect all chunks
        all_chunks = []
        for doc in documents:
            # Remove old chunks if re-indexing
            source = doc.metadata.get("source", "")
            if source in self._indexed_sources:
                self._vector_store.delete_by_source(source)
            
            all_chunks.extend(doc.chunks)
            self._indexed_sources.add(source)

        # Add to vector store
        if all_chunks:
            self._vector_store.add_chunks(all_chunks)
            self._hybrid_retriever.build_bm25_index(all_chunks)

        self._logger.info(
            "directory_indexed",
            directory=str(directory),
            document_count=len(documents),
            chunk_count=len(all_chunks),
        )

        return documents

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool = True,
        rerank_top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of candidates for hybrid search
            rerank: Whether to apply cross-encoder reranking
            rerank_top_k: Number of final results after reranking
            
        Returns:
            RetrievalResult with documents and formatted context
        """
        top_k = top_k or settings.rag.top_k
        rerank_top_k = rerank_top_k or settings.rag.reranker_top_k

        self._logger.debug(
            "retrieving",
            query=query[:100],
            top_k=top_k,
            rerank=rerank,
        )

        # Hybrid search
        candidates = self._hybrid_retriever.search(query, top_k=top_k)

        # Rerank if requested
        if rerank and candidates:
            documents = self._reranker.rerank(
                query=query,
                documents=candidates,
                top_k=rerank_top_k,
            )
        else:
            documents = candidates[:rerank_top_k]

        # Format context
        context = self._format_context(documents)

        # Build metadata
        metadata = {
            "query": query,
            "candidate_count": len(candidates),
            "final_count": len(documents),
            "reranked": rerank,
            "sources": list({d.get("metadata", {}).get("source", "") for d in documents}),
        }

        self._logger.debug(
            "retrieval_complete",
            candidate_count=len(candidates),
            final_count=len(documents),
        )

        return RetrievalResult(
            query=query,
            documents=documents,
            context=context,
            metadata=metadata,
        )

    def _format_context(self, documents: list[dict[str, Any]]) -> str:
        """
        Format retrieved documents as context for LLM.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documentation found."

        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            section = metadata.get("section", "")
            content = doc.get("content", "")
            
            # Get score (prefer rerank_score, then fused_score)
            score = doc.get("rerank_score", doc.get("fused_score", 0.0))
            
            header = f"### Source {i}: {source}"
            if section:
                header += f" - {section}"
            header += f" (relevance: {score:.2f})"

            context_parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def search_simple(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Simple search returning just document content and metadata.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of simplified results
        """
        result = self.retrieve(query, top_k=top_k * 2, rerank_top_k=top_k)
        
        return [
            {
                "source": doc.get("metadata", {}).get("source", "unknown"),
                "section": doc.get("metadata", {}).get("section", ""),
                "content": doc.get("content", ""),
                "score": doc.get("rerank_score", doc.get("fused_score", 0.0)),
            }
            for doc in result.documents
        ]

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self._vector_store.clear()
        self._indexed_sources.clear()
        self._hybrid_retriever._bm25 = None
        self._hybrid_retriever._bm25_corpus = []
        self._logger.info("index_cleared")

    def reindex(self) -> None:
        """Clear and rebuild the entire index."""
        self.clear_index()
        self.index_directory(self._runbooks_dir)

    def stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "collection_name": self._collection_name,
            "runbooks_dir": str(self._runbooks_dir),
            "document_count": self._vector_store.count,
            "indexed_sources": list(self._indexed_sources),
            "vector_store": self._vector_store.stats(),
            "hybrid_retriever": self._hybrid_retriever.stats(),
            "embedding_model": self._embedding_manager.cache_stats(),
        }


# Singleton instance
_pipeline_instance: RAGPipeline | None = None


def get_rag_pipeline(
    collection_name: str = "runbooks",
    auto_index: bool = True,
) -> RAGPipeline:
    """
    Get or create the RAG pipeline singleton.
    
    Args:
        collection_name: Collection name for vector store
        auto_index: Whether to auto-index on creation
        
    Returns:
        RAGPipeline instance
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(
            collection_name=collection_name,
            auto_index=auto_index,
        )
    
    return _pipeline_instance
