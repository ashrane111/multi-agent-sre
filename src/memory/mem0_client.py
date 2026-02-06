"""
Mem0 client wrapper for episodic memory.

Provides persistent storage and retrieval of past incidents
to enable learning from historical data.
"""

from datetime import datetime
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class Mem0Client:
    """
    Wrapper around Mem0 for incident memory storage.
    
    Features:
    - Store incident memories with metadata
    - Semantic search for similar incidents
    - Memory consolidation and pruning
    - Fallback to local storage if Mem0 unavailable
    """

    def __init__(
        self,
        collection_name: str | None = None,
        use_local_fallback: bool = True,
    ) -> None:
        """
        Initialize Mem0 client.
        
        Args:
            collection_name: Name for the memory collection
            use_local_fallback: Use local storage if Mem0 fails
        """
        self._collection_name = collection_name or settings.memory.mem0_collection_name
        self._use_local_fallback = use_local_fallback
        self._logger = logger.bind(
            component="mem0_client",
            collection=self._collection_name,
        )
        
        # Initialize Mem0 (lazy loading)
        self._mem0 = None
        self._initialized = False
        
        # Local fallback storage
        self._local_memories: list[dict[str, Any]] = []

    def _ensure_initialized(self) -> bool:
        """
        Ensure Mem0 is initialized.
        
        Returns:
            True if Mem0 is available, False otherwise
        """
        if self._initialized:
            return self._mem0 is not None

        try:
            from mem0 import Memory
            
            # Configure Mem0
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": self._collection_name,
                        "path": settings.rag.chroma_persist_dir + "_mem0",
                    },
                },
            }
            
            self._mem0 = Memory.from_config(config)
            self._initialized = True
            
            self._logger.info(
                "mem0_initialized",
                collection=self._collection_name,
            )
            return True
            
        except ImportError:
            self._logger.warning("mem0_not_installed")
            self._initialized = True
            return False
        except Exception as e:
            self._logger.error(
                "mem0_initialization_failed",
                error=str(e),
            )
            self._initialized = True
            return False

    def add_memory(
        self,
        content: str,
        user_id: str = "sre_system",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Add a memory to the store.
        
        Args:
            content: Memory content (text description of incident)
            user_id: User/agent ID associated with memory
            metadata: Additional metadata
            
        Returns:
            Memory ID if successful, None otherwise
        """
        metadata = metadata or {}
        metadata["created_at"] = datetime.utcnow().isoformat()
        
        if self._ensure_initialized() and self._mem0:
            try:
                result = self._mem0.add(
                    content,
                    user_id=user_id,
                    metadata=metadata,
                )
                
                memory_id = result.get("id") if isinstance(result, dict) else str(result)
                
                self._logger.info(
                    "memory_added",
                    memory_id=memory_id,
                    content_length=len(content),
                )
                
                return memory_id
                
            except Exception as e:
                self._logger.error("memory_add_failed", error=str(e))
                if not self._use_local_fallback:
                    raise

        # Fallback to local storage
        if self._use_local_fallback:
            memory_id = f"local_{len(self._local_memories)}"
            self._local_memories.append({
                "id": memory_id,
                "content": content,
                "user_id": user_id,
                "metadata": metadata,
            })
            self._logger.debug("memory_added_locally", memory_id=memory_id)
            return memory_id
        
        return None

    def search_memories(
        self,
        query: str,
        user_id: str = "sre_system",
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Search for similar memories.
        
        Args:
            query: Search query
            user_id: User/agent ID to search within
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching memories with scores
        """
        if self._ensure_initialized() and self._mem0:
            try:
                results = self._mem0.search(
                    query,
                    user_id=user_id,
                    limit=top_k,
                )
                
                # Format results
                formatted = []
                for r in results:
                    score = r.get("score", r.get("similarity", 0.0))
                    if score >= threshold:
                        formatted.append({
                            "id": r.get("id", ""),
                            "content": r.get("memory", r.get("content", "")),
                            "metadata": r.get("metadata", {}),
                            "score": score,
                        })
                
                self._logger.debug(
                    "memory_search_complete",
                    query_length=len(query),
                    result_count=len(formatted),
                )
                
                return formatted
                
            except Exception as e:
                self._logger.error("memory_search_failed", error=str(e))
                if not self._use_local_fallback:
                    raise

        # Fallback to local search (simple keyword matching)
        if self._use_local_fallback and self._local_memories:
            return self._local_search(query, top_k, threshold)
        
        return []

    def _local_search(
        self,
        query: str,
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """
        Simple local search using keyword matching.
        
        Args:
            query: Search query
            top_k: Maximum results
            threshold: Minimum score threshold
            
        Returns:
            List of matching memories
        """
        query_words = set(query.lower().split())
        
        scored_memories = []
        for memory in self._local_memories:
            content_words = set(memory["content"].lower().split())
            
            # Simple Jaccard similarity
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)
            score = intersection / union if union > 0 else 0.0
            
            if score >= threshold:
                scored_memories.append({
                    "id": memory["id"],
                    "content": memory["content"],
                    "metadata": memory["metadata"],
                    "score": score,
                })
        
        # Sort by score and return top-k
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        return scored_memories[:top_k]

    def get_all_memories(
        self,
        user_id: str = "sre_system",
    ) -> list[dict[str, Any]]:
        """
        Get all memories for a user.
        
        Args:
            user_id: User/agent ID
            
        Returns:
            List of all memories
        """
        if self._ensure_initialized() and self._mem0:
            try:
                results = self._mem0.get_all(user_id=user_id)
                return [
                    {
                        "id": r.get("id", ""),
                        "content": r.get("memory", r.get("content", "")),
                        "metadata": r.get("metadata", {}),
                    }
                    for r in results
                ]
            except Exception as e:
                self._logger.error("get_all_memories_failed", error=str(e))

        # Return local memories as fallback
        return self._local_memories.copy()

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if self._ensure_initialized() and self._mem0:
            try:
                self._mem0.delete(memory_id)
                self._logger.info("memory_deleted", memory_id=memory_id)
                return True
            except Exception as e:
                self._logger.error("memory_delete_failed", error=str(e))

        # Delete from local storage
        self._local_memories = [
            m for m in self._local_memories if m["id"] != memory_id
        ]
        return True

    def clear_memories(self, user_id: str = "sre_system") -> int:
        """
        Clear all memories for a user.
        
        Args:
            user_id: User/agent ID
            
        Returns:
            Number of memories deleted
        """
        count = 0
        
        if self._ensure_initialized() and self._mem0:
            try:
                memories = self._mem0.get_all(user_id=user_id)
                for m in memories:
                    self._mem0.delete(m["id"])
                    count += 1
            except Exception as e:
                self._logger.error("clear_memories_failed", error=str(e))

        # Clear local memories too
        local_count = len(self._local_memories)
        self._local_memories.clear()
        count += local_count

        self._logger.info("memories_cleared", count=count)
        return count

    def stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        mem0_count = 0
        if self._ensure_initialized() and self._mem0:
            try:
                mem0_count = len(self._mem0.get_all(user_id="sre_system"))
            except Exception:
                pass

        return {
            "collection_name": self._collection_name,
            "mem0_available": self._mem0 is not None,
            "mem0_memory_count": mem0_count,
            "local_memory_count": len(self._local_memories),
            "use_local_fallback": self._use_local_fallback,
        }


# Singleton instance
_mem0_client: Mem0Client | None = None


def get_mem0_client() -> Mem0Client:
    """Get or create the Mem0 client singleton."""
    global _mem0_client
    
    if _mem0_client is None:
        _mem0_client = Mem0Client()
    
    return _mem0_client
