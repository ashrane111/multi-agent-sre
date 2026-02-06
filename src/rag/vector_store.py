"""
ChromaDB vector store integration for RAG pipeline.

Provides persistent vector storage with semantic search capabilities.
"""

from pathlib import Path
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.rag.document_processor import DocumentChunk
from src.rag.embeddings import get_embedding_manager

logger = structlog.get_logger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for document embeddings.
    
    Features:
    - Persistent storage
    - Semantic similarity search
    - Metadata filtering
    - Batch operations
    """

    def __init__(
        self,
        collection_name: str = "runbooks",
        persist_directory: str | None = None,
    ) -> None:
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        self._collection_name = collection_name
        self._persist_dir = persist_directory or settings.rag.chroma_persist_dir
        self._embedding_manager = get_embedding_manager()
        self._logger = logger.bind(
            component="vector_store",
            collection=collection_name,
        )

        # Ensure persist directory exists
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._logger.info(
            "vector_store_initialized",
            persist_dir=self._persist_dir,
            collection_name=collection_name,
            document_count=self._collection.count(),
        )

    @property
    def collection(self) -> chromadb.Collection:
        """Get the underlying ChromaDB collection."""
        return self._collection

    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 100,
    ) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunks to add
            batch_size: Batch size for processing
        """
        if not chunks:
            return

        self._logger.info("adding_chunks", count=len(chunks))

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]

            # Generate embeddings
            embeddings = self._embedding_manager.embed_texts(documents)

            # Add to collection
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            self._logger.debug(
                "batch_added",
                batch_start=i,
                batch_size=len(batch),
            )

        self._logger.info(
            "chunks_added",
            total_added=len(chunks),
            collection_size=self._collection.count(),
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            List of results with id, content, metadata, and distance
        """
        top_k = top_k or settings.rag.top_k

        # Generate query embedding
        query_embedding = self._embedding_manager.embed_query(query)

        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results: list[dict[str, Any]] = []
        
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "score": 1 - results["distances"][0][i] if results["distances"] else 1.0,
                })

        self._logger.debug(
            "search_complete",
            query_length=len(query),
            results_count=len(formatted_results),
        )

        return formatted_results

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            where: Metadata filter
            
        Returns:
            List of results
        """
        top_k = top_k or settings.rag.top_k

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: list[dict[str, Any]] = []
        
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "score": 1 - results["distances"][0][i] if results["distances"] else 1.0,
                })

        return formatted_results

    def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: The chunk ID to retrieve
            
        Returns:
            Chunk data or None if not found
        """
        results = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )

        if results["ids"]:
            return {
                "id": results["ids"][0],
                "content": results["documents"][0] if results["documents"] else "",
                "metadata": results["metadatas"][0] if results["metadatas"] else {},
            }
        return None

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source file.
        
        Args:
            source: Source filename to delete
            
        Returns:
            Number of chunks deleted
        """
        # Get IDs to delete
        results = self._collection.get(
            where={"source": source},
            include=[],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self._collection.delete(ids=results["ids"])

        self._logger.info(
            "chunks_deleted",
            source=source,
            count=count,
        )

        return count

    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._logger.info("collection_cleared")

    def get_all_sources(self) -> list[str]:
        """Get list of all source files in the collection."""
        results = self._collection.get(include=["metadatas"])
        
        sources = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
        
        return sorted(sources)

    def stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        return {
            "collection_name": self._collection_name,
            "document_count": self._collection.count(),
            "persist_directory": self._persist_dir,
            "sources": self.get_all_sources(),
        }
