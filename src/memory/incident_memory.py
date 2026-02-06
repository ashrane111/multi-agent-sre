"""
Incident-specific episodic memory operations.

Provides high-level interface for storing and retrieving
incident memories with structured data.
"""

from dataclasses import asdict
from datetime import datetime
from typing import Any

import structlog

from src.memory.mem0_client import Mem0Client, get_mem0_client
from src.workflows.states import IncidentState, SimilarIncident

logger = structlog.get_logger(__name__)


class IncidentMemory:
    """
    Episodic memory system for SRE incidents.
    
    Features:
    - Store resolved incidents for future reference
    - Recall similar past incidents based on symptoms
    - Track resolution patterns and success rates
    - Learn from incident history
    """

    def __init__(self, mem0_client: Mem0Client | None = None) -> None:
        """
        Initialize incident memory.
        
        Args:
            mem0_client: Mem0 client instance (uses singleton if not provided)
        """
        self._client = mem0_client or get_mem0_client()
        self._logger = logger.bind(component="incident_memory")

    def store_incident(
        self,
        state: IncidentState,
        resolution_success: bool = True,
    ) -> str | None:
        """
        Store a resolved incident in memory.
        
        Args:
            state: Final state of the resolved incident
            resolution_success: Whether the resolution was successful
            
        Returns:
            Memory ID if stored successfully
        """
        # Calculate resolution time
        resolution_time = state.time_to_resolve_minutes

        # Build memory content
        content = self._build_incident_content(state)
        
        # Build metadata
        metadata = {
            "incident_id": state.incident_id,
            "severity": state.severity,
            "cluster": state.cluster,
            "namespace": state.namespace,
            "anomalies": state.anomalies,
            "root_cause": state.root_cause,
            "resolution_success": resolution_success,
            "resolution_time_minutes": resolution_time,
            "actions_taken": [a.action_type for a in state.proposed_actions] if state.proposed_actions else [],
            "memory_type": "incident",
        }

        memory_id = self._client.add_memory(
            content=content,
            user_id="sre_system",
            metadata=metadata,
        )

        self._logger.info(
            "incident_stored",
            incident_id=state.incident_id,
            memory_id=memory_id,
            resolution_time=resolution_time,
        )

        return memory_id

    def _build_incident_content(self, state: IncidentState) -> str:
        """
        Build a text description of the incident for memory storage.
        
        Args:
            state: Incident state
            
        Returns:
            Text description
        """
        parts = [
            f"Incident: {state.title}",
            f"Cluster: {state.cluster}, Namespace: {state.namespace}",
            f"Severity: {state.severity}",
            f"Symptoms: {', '.join(state.anomalies)}",
        ]

        if state.root_cause:
            parts.append(f"Root Cause: {state.root_cause}")

        if state.diagnosis:
            parts.append(f"Diagnosis: {state.diagnosis}")

        if state.proposed_actions:
            actions = [f"{a.action_type}: {a.description}" for a in state.proposed_actions]
            parts.append(f"Actions Taken: {'; '.join(actions)}")

        return "\n".join(parts)

    def recall_similar_incidents(
        self,
        state: IncidentState,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[SimilarIncident]:
        """
        Recall similar past incidents based on current incident state.
        
        Args:
            state: Current incident state
            top_k: Maximum number of similar incidents to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar past incidents
        """
        # Build search query from current incident
        query = self._build_search_query(state)

        self._logger.debug(
            "searching_similar_incidents",
            incident_id=state.incident_id,
            query_length=len(query),
        )

        # Search memories
        memories = self._client.search_memories(
            query=query,
            user_id="sre_system",
            top_k=top_k,
            threshold=threshold,
        )

        # Convert to SimilarIncident objects
        similar_incidents = []
        for memory in memories:
            try:
                incident = self._memory_to_similar_incident(memory)
                if incident:
                    similar_incidents.append(incident)
            except Exception as e:
                self._logger.warning(
                    "failed_to_parse_memory",
                    memory_id=memory.get("id"),
                    error=str(e),
                )

        self._logger.info(
            "similar_incidents_found",
            incident_id=state.incident_id,
            count=len(similar_incidents),
        )

        return similar_incidents

    def _build_search_query(self, state: IncidentState) -> str:
        """
        Build a search query from incident state.
        
        Args:
            state: Current incident state
            
        Returns:
            Search query string
        """
        parts = [
            f"Cluster: {state.cluster}",
            f"Symptoms: {', '.join(state.anomalies)}",
        ]

        if state.description:
            parts.append(state.description)

        return " ".join(parts)

    def _memory_to_similar_incident(
        self,
        memory: dict[str, Any],
    ) -> SimilarIncident | None:
        """
        Convert a memory result to SimilarIncident.
        
        Args:
            memory: Memory search result
            
        Returns:
            SimilarIncident or None if conversion fails
        """
        metadata = memory.get("metadata", {})
        
        # Skip if not an incident memory
        if metadata.get("memory_type") != "incident":
            return None

        # Parse occurred_at from metadata
        occurred_at = None
        created_at = metadata.get("created_at")
        if created_at:
            try:
                occurred_at = datetime.fromisoformat(created_at)
            except (ValueError, TypeError):
                occurred_at = datetime.utcnow()
        else:
            occurred_at = datetime.utcnow()

        return SimilarIncident(
            incident_id=metadata.get("incident_id", "unknown"),
            similarity_score=memory.get("score", 0.0),
            cluster=metadata.get("cluster", "unknown"),
            symptoms=metadata.get("anomalies", []),
            root_cause=metadata.get("root_cause", "Unknown"),
            fix_applied=", ".join(metadata.get("actions_taken", [])) or "Unknown",
            resolution_time_minutes=metadata.get("resolution_time_minutes", 0.0),
            occurred_at=occurred_at,
        )

    def get_incident_history(
        self,
        cluster: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get incident history, optionally filtered by cluster.
        
        Args:
            cluster: Optional cluster filter
            limit: Maximum number of results
            
        Returns:
            List of incident memories
        """
        all_memories = self._client.get_all_memories(user_id="sre_system")
        
        # Filter by memory type and optionally by cluster
        incidents = []
        for memory in all_memories:
            metadata = memory.get("metadata", {})
            if metadata.get("memory_type") == "incident":
                if cluster is None or metadata.get("cluster") == cluster:
                    incidents.append(memory)
        
        # Sort by created_at (newest first) and limit
        incidents.sort(
            key=lambda x: x.get("metadata", {}).get("created_at", ""),
            reverse=True,
        )
        
        return incidents[:limit]

    def get_resolution_stats(self) -> dict[str, Any]:
        """
        Get statistics about resolved incidents.
        
        Returns:
            Statistics dictionary
        """
        all_incidents = self.get_incident_history(limit=1000)
        
        total = len(all_incidents)
        successful = 0
        total_time = 0.0
        time_count = 0
        by_cluster: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        
        for incident in all_incidents:
            metadata = incident.get("metadata", {})
            
            if metadata.get("resolution_success"):
                successful += 1
            
            res_time = metadata.get("resolution_time_minutes")
            if res_time:
                total_time += res_time
                time_count += 1
            
            cluster = metadata.get("cluster", "unknown")
            by_cluster[cluster] = by_cluster.get(cluster, 0) + 1
            
            severity = metadata.get("severity", "unknown")
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_incidents": total,
            "successful_resolutions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_resolution_time_minutes": total_time / time_count if time_count > 0 else 0.0,
            "by_cluster": by_cluster,
            "by_severity": by_severity,
        }

    def clear_history(self) -> int:
        """
        Clear all incident memories.
        
        Returns:
            Number of memories cleared
        """
        return self._client.clear_memories(user_id="sre_system")

    def stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        client_stats = self._client.stats()
        resolution_stats = self.get_resolution_stats()
        
        return {
            **client_stats,
            "resolution_stats": resolution_stats,
        }


# Singleton instance
_incident_memory: IncidentMemory | None = None


def get_incident_memory() -> IncidentMemory:
    """Get or create the incident memory singleton."""
    global _incident_memory
    
    if _incident_memory is None:
        _incident_memory = IncidentMemory()
    
    return _incident_memory
