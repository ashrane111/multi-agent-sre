"""
Audit Trail for governance and compliance.

Provides persistent logging of all significant actions taken
by the SRE platform for accountability and compliance.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AuditAction(str, Enum):
    """Types of auditable actions."""
    
    # Incident lifecycle
    INCIDENT_CREATED = "incident_created"
    INCIDENT_CLASSIFIED = "incident_classified"
    INCIDENT_DIAGNOSED = "incident_diagnosed"
    INCIDENT_RESOLVED = "incident_resolved"
    INCIDENT_ESCALATED = "incident_escalated"
    
    # Approval workflow
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_ESCALATED = "approval_escalated"
    
    # Remediation
    ACTION_PROPOSED = "action_proposed"
    ACTION_APPROVED = "action_approved"
    ACTION_EXECUTED = "action_executed"
    ACTION_FAILED = "action_failed"
    ACTION_ROLLED_BACK = "action_rolled_back"
    
    # Policy
    POLICY_CHECKED = "policy_checked"
    POLICY_VIOLATION = "policy_violation"
    POLICY_OVERRIDE = "policy_override"
    
    # System
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_CHANGE = "configuration_change"


class AuditEntry(BaseModel):
    """A single audit log entry."""
    
    entry_id: str = Field(description="Unique entry ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # What happened
    action: AuditAction = Field(description="Type of action")
    description: str = Field(description="Human-readable description")
    
    # Context
    incident_id: str | None = Field(default=None)
    request_id: str | None = Field(default=None)  # Approval request ID
    agent: str | None = Field(default=None)  # Which agent performed the action
    
    # Actor
    actor_type: str = Field(default="system")  # system, human, agent
    actor_id: str | None = Field(default=None)  # User email or agent name
    
    # Details
    details: dict[str, Any] = Field(default_factory=dict)
    
    # Risk assessment
    risk_level: str = Field(default="low")  # low, medium, high, critical
    
    # Outcome
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None)


class AuditTrail:
    """
    Manages audit logging for compliance and accountability.
    
    Features:
    - Persistent storage (file-based, can extend to DB)
    - Structured queries
    - Retention policies
    - Export capabilities
    """

    def __init__(
        self,
        storage_path: str = "data/audit",
        retention_days: int = 90,
    ) -> None:
        """
        Initialize audit trail.
        
        Args:
            storage_path: Path for audit log storage
            retention_days: How long to retain logs
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._logger = logger.bind(component="audit_trail")
        
        # In-memory index for fast queries
        self._entries: list[AuditEntry] = []
        self._by_incident: dict[str, list[AuditEntry]] = {}
        self._by_action: dict[AuditAction, list[AuditEntry]] = {}
        
        # Load existing entries
        self._load_entries()
        
        self._entry_counter = len(self._entries)

    def _load_entries(self) -> None:
        """Load existing audit entries from storage."""
        audit_file = self._storage_path / "audit_log.jsonl"
        
        if not audit_file.exists():
            return
        
        try:
            with open(audit_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        entry = AuditEntry(**data)
                        self._index_entry(entry)
            
            self._logger.info(
                "audit_entries_loaded",
                count=len(self._entries),
            )
        except Exception as e:
            self._logger.error(
                "audit_load_failed",
                error=str(e),
            )

    def _index_entry(self, entry: AuditEntry) -> None:
        """Index an entry for fast querying."""
        self._entries.append(entry)
        
        if entry.incident_id:
            if entry.incident_id not in self._by_incident:
                self._by_incident[entry.incident_id] = []
            self._by_incident[entry.incident_id].append(entry)
        
        if entry.action not in self._by_action:
            self._by_action[entry.action] = []
        self._by_action[entry.action].append(entry)

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist an entry to storage."""
        audit_file = self._storage_path / "audit_log.jsonl"
        
        try:
            with open(audit_file, "a") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as e:
            self._logger.error(
                "audit_persist_failed",
                entry_id=entry.entry_id,
                error=str(e),
            )

    def log(
        self,
        action: AuditAction,
        description: str,
        incident_id: str | None = None,
        request_id: str | None = None,
        agent: str | None = None,
        actor_type: str = "system",
        actor_id: str | None = None,
        details: dict[str, Any] | None = None,
        risk_level: str = "low",
        success: bool = True,
        error_message: str | None = None,
    ) -> AuditEntry:
        """
        Create a new audit log entry.
        
        Args:
            action: Type of action
            description: Human-readable description
            incident_id: Associated incident ID
            request_id: Associated approval request ID
            agent: Agent that performed the action
            actor_type: Type of actor (system, human, agent)
            actor_id: ID of the actor
            details: Additional details
            risk_level: Risk assessment
            success: Whether the action succeeded
            error_message: Error message if failed
            
        Returns:
            Created AuditEntry
        """
        self._entry_counter += 1
        entry_id = f"AUD-{datetime.utcnow().strftime('%Y%m%d')}-{self._entry_counter:06d}"
        
        entry = AuditEntry(
            entry_id=entry_id,
            action=action,
            description=description,
            incident_id=incident_id,
            request_id=request_id,
            agent=agent,
            actor_type=actor_type,
            actor_id=actor_id,
            details=details or {},
            risk_level=risk_level,
            success=success,
            error_message=error_message,
        )
        
        # Index and persist
        self._index_entry(entry)
        self._persist_entry(entry)
        
        # Also log to structured logger
        self._logger.info(
            "audit_entry_created",
            entry_id=entry_id,
            action=action.value,
            incident_id=incident_id,
            risk_level=risk_level,
        )
        
        return entry

    def get_by_incident(self, incident_id: str) -> list[AuditEntry]:
        """Get all audit entries for an incident."""
        return self._by_incident.get(incident_id, [])

    def get_by_action(self, action: AuditAction) -> list[AuditEntry]:
        """Get all audit entries for an action type."""
        return self._by_action.get(action, [])

    def get_recent(
        self,
        limit: int = 100,
        action: AuditAction | None = None,
        risk_level: str | None = None,
        success: bool | None = None,
    ) -> list[AuditEntry]:
        """
        Get recent audit entries with optional filtering.
        
        Args:
            limit: Maximum number of entries
            action: Filter by action type
            risk_level: Filter by risk level
            success: Filter by success status
            
        Returns:
            List of matching entries (newest first)
        """
        entries = self._entries.copy()
        
        # Apply filters
        if action:
            entries = [e for e in entries if e.action == action]
        if risk_level:
            entries = [e for e in entries if e.risk_level == risk_level]
        if success is not None:
            entries = [e for e in entries if e.success == success]
        
        # Sort by timestamp (newest first) and limit
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def get_policy_violations(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditEntry]:
        """Get all policy violation entries within a date range."""
        violations = self._by_action.get(AuditAction.POLICY_VIOLATION, [])
        
        if start_date:
            violations = [v for v in violations if v.timestamp >= start_date]
        if end_date:
            violations = [v for v in violations if v.timestamp <= end_date]
        
        return violations

    def get_approval_history(self, incident_id: str) -> list[AuditEntry]:
        """Get approval-related entries for an incident."""
        incident_entries = self._by_incident.get(incident_id, [])
        
        approval_actions = {
            AuditAction.APPROVAL_REQUESTED,
            AuditAction.APPROVAL_GRANTED,
            AuditAction.APPROVAL_REJECTED,
            AuditAction.APPROVAL_EXPIRED,
            AuditAction.APPROVAL_ESCALATED,
        }
        
        return [e for e in incident_entries if e.action in approval_actions]

    def export_to_json(
        self,
        filepath: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """
        Export audit entries to JSON file.
        
        Args:
            filepath: Output file path
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            Number of entries exported
        """
        entries = self._entries.copy()
        
        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]
        if end_date:
            entries = [e for e in entries if e.timestamp <= end_date]
        
        with open(filepath, "w") as f:
            json.dump(
                [e.model_dump(mode="json") for e in entries],
                f,
                indent=2,
                default=str,
            )
        
        return len(entries)

    def stats(self) -> dict[str, Any]:
        """Get audit trail statistics."""
        action_counts = {}
        for action, entries in self._by_action.items():
            action_counts[action.value] = len(entries)
        
        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for entry in self._entries:
            if entry.risk_level in risk_counts:
                risk_counts[entry.risk_level] += 1
        
        failure_count = len([e for e in self._entries if not e.success])
        
        return {
            "total_entries": len(self._entries),
            "incidents_tracked": len(self._by_incident),
            "action_counts": action_counts,
            "risk_distribution": risk_counts,
            "failure_count": failure_count,
        }


# Singleton instance
_audit_trail: AuditTrail | None = None


def get_audit_trail() -> AuditTrail:
    """Get or create the audit trail singleton."""
    global _audit_trail
    
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    
    return _audit_trail
