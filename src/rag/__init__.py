"""
RAG (Retrieval-Augmented Generation) module.

Provides document retrieval capabilities for the SRE platform:
- Document processing and chunking
- Embedding generation
- Vector storage with ChromaDB
- Hybrid retrieval (BM25 + semantic)
- Cross-encoder reranking
"""

from src.rag.document_processor import Document, DocumentChunk, DocumentProcessor
from src.rag.embeddings import EmbeddingManager, get_embedding_manager
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.rag_pipeline import RAGPipeline, RetrievalResult, get_rag_pipeline
from src.rag.reranker import Reranker, get_reranker
from src.rag.vector_store import VectorStore

__all__ = [
    # Document processing
    "Document",
    "DocumentChunk",
    "DocumentProcessor",
    # Embeddings
    "EmbeddingManager",
    "get_embedding_manager",
    # Vector store
    "VectorStore",
    # Hybrid retrieval
    "HybridRetriever",
    # Reranking
    "Reranker",
    "get_reranker",
    # Pipeline
    "RAGPipeline",
    "RetrievalResult",
    "get_rag_pipeline",
]
