"""
Human-in-the-Loop Approval Manager.

Manages approval requests for remediation actions that require
human review before execution.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog
from pydantic import BaseModel, Field

from src.config import settings

logger = structlog.get_logger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ESCALATED = "escalated"


class ApprovalRequest(BaseModel):
    """A request for human approval."""
    
    request_id: str = Field(description="Unique approval request ID")
    incident_id: str = Field(description="Associated incident ID")
    
    # What needs approval
    actions: list[dict[str, Any]] = Field(description="Actions requiring approval")
    policy_violations: list[dict[str, Any]] = Field(default_factory=list)
    blast_radius_score: float = Field(default=0.0)
    
    # Context
    severity: str = Field(description="Incident severity")
    root_cause: str = Field(default="")
    diagnosis_confidence: float = Field(default=0.0)
    
    # Status tracking
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(default=None)
    
    # Resolution
    resolved_at: datetime | None = Field(default=None)
    resolved_by: str | None = Field(default=None)
    resolution_notes: str | None = Field(default=None)
    
    # Notification tracking
    notifications_sent: list[dict[str, Any]] = Field(default_factory=list)
    escalation_level: int = Field(default=0)


class ApprovalDecision(BaseModel):
    """Decision made on an approval request."""
    
    request_id: str
    approved: bool
    approver: str
    notes: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ApprovalManager:
    """
    Manages human approval workflow for remediation actions.
    
    Features:
    - Approval request creation and tracking
    - Timeout handling with escalation
    - Notification dispatch (Slack, PagerDuty)
    - Audit logging
    - Async waiting for approvals
    """

    def __init__(
        self,
        default_timeout_minutes: int = 15,
        escalation_timeout_minutes: int = 5,
    ) -> None:
        """
        Initialize approval manager.
        
        Args:
            default_timeout_minutes: Default time before escalation
            escalation_timeout_minutes: Time between escalation levels
        """
        self._pending: dict[str, ApprovalRequest] = {}
        self._resolved: dict[str, ApprovalRequest] = {}
        self._default_timeout = timedelta(minutes=default_timeout_minutes)
        self._escalation_timeout = timedelta(minutes=escalation_timeout_minutes)
        self._logger = logger.bind(component="approval_manager")
        
        # Notification callbacks
        self._notification_handlers: list[Callable] = []
        
        # Event for signaling approvals
        self._approval_events: dict[str, asyncio.Event] = {}

    def register_notification_handler(self, handler: Callable) -> None:
        """Register a notification handler (e.g., Slack, PagerDuty)."""
        self._notification_handlers.append(handler)

    async def create_request(
        self,
        incident_id: str,
        actions: list[dict[str, Any]],
        severity: str,
        policy_violations: list[dict[str, Any]] | None = None,
        blast_radius_score: float = 0.0,
        root_cause: str = "",
        diagnosis_confidence: float = 0.0,
        timeout_minutes: int | None = None,
    ) -> ApprovalRequest:
        """
        Create a new approval request.
        
        Args:
            incident_id: Associated incident ID
            actions: Actions requiring approval
            severity: Incident severity
            policy_violations: Policy violations that triggered review
            blast_radius_score: Estimated blast radius
            root_cause: Diagnosed root cause
            diagnosis_confidence: Confidence in diagnosis
            timeout_minutes: Custom timeout (uses default if not specified)
            
        Returns:
            Created ApprovalRequest
        """
        request_id = f"APR-{incident_id}-{datetime.utcnow().strftime('%H%M%S')}"
        
        timeout = timedelta(minutes=timeout_minutes) if timeout_minutes else self._default_timeout
        
        request = ApprovalRequest(
            request_id=request_id,
            incident_id=incident_id,
            actions=actions,
            policy_violations=policy_violations or [],
            blast_radius_score=blast_radius_score,
            severity=severity,
            root_cause=root_cause,
            diagnosis_confidence=diagnosis_confidence,
            expires_at=datetime.utcnow() + timeout,
        )
        
        self._pending[request_id] = request
        self._approval_events[request_id] = asyncio.Event()
        
        self._logger.info(
            "approval_request_created",
            request_id=request_id,
            incident_id=incident_id,
            action_count=len(actions),
            expires_at=request.expires_at.isoformat(),
        )
        
        # Send notifications
        await self._send_notifications(request)
        
        return request

    async def _send_notifications(self, request: ApprovalRequest) -> None:
        """Send notifications for an approval request."""
        for handler in self._notification_handlers:
            try:
                await handler(request)
                request.notifications_sent.append({
                    "handler": handler.__name__,
                    "sent_at": datetime.utcnow().isoformat(),
                    "escalation_level": request.escalation_level,
                })
            except Exception as e:
                self._logger.error(
                    "notification_failed",
                    request_id=request.request_id,
                    handler=handler.__name__,
                    error=str(e),
                )

    async def approve(
        self,
        request_id: str,
        approver: str,
        notes: str = "",
    ) -> ApprovalDecision:
        """
        Approve a pending request.
        
        Args:
            request_id: ID of the request to approve
            approver: Email/ID of the approver
            notes: Optional approval notes
            
        Returns:
            ApprovalDecision
            
        Raises:
            ValueError: If request not found or not pending
        """
        request = self._pending.get(request_id)
        if not request:
            raise ValueError(f"Approval request {request_id} not found or already resolved")
        
        request.status = ApprovalStatus.APPROVED
        request.resolved_at = datetime.utcnow()
        request.resolved_by = approver
        request.resolution_notes = notes
        
        # Move to resolved
        self._resolved[request_id] = request
        del self._pending[request_id]
        
        # Signal waiting coroutines
        if request_id in self._approval_events:
            self._approval_events[request_id].set()
        
        self._logger.info(
            "approval_granted",
            request_id=request_id,
            incident_id=request.incident_id,
            approver=approver,
        )
        
        return ApprovalDecision(
            request_id=request_id,
            approved=True,
            approver=approver,
            notes=notes,
        )

    async def reject(
        self,
        request_id: str,
        approver: str,
        notes: str = "",
    ) -> ApprovalDecision:
        """
        Reject a pending request.
        
        Args:
            request_id: ID of the request to reject
            approver: Email/ID of the rejector
            notes: Rejection reason
            
        Returns:
            ApprovalDecision
        """
        request = self._pending.get(request_id)
        if not request:
            raise ValueError(f"Approval request {request_id} not found or already resolved")
        
        request.status = ApprovalStatus.REJECTED
        request.resolved_at = datetime.utcnow()
        request.resolved_by = approver
        request.resolution_notes = notes
        
        # Move to resolved
        self._resolved[request_id] = request
        del self._pending[request_id]
        
        # Signal waiting coroutines
        if request_id in self._approval_events:
            self._approval_events[request_id].set()
        
        self._logger.info(
            "approval_rejected",
            request_id=request_id,
            incident_id=request.incident_id,
            approver=approver,
            reason=notes,
        )
        
        return ApprovalDecision(
            request_id=request_id,
            approved=False,
            approver=approver,
            notes=notes,
        )

    async def wait_for_approval(
        self,
        request_id: str,
        timeout_seconds: float | None = None,
    ) -> ApprovalRequest:
        """
        Wait for an approval request to be resolved.
        
        Args:
            request_id: ID of the request to wait for
            timeout_seconds: Maximum time to wait (None for no timeout)
            
        Returns:
            Resolved ApprovalRequest
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
            ValueError: If request not found
        """
        if request_id not in self._approval_events:
            # Check if already resolved
            if request_id in self._resolved:
                return self._resolved[request_id]
            raise ValueError(f"Approval request {request_id} not found")
        
        event = self._approval_events[request_id]
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            # Check if expired
            request = self._pending.get(request_id)
            if request:
                request.status = ApprovalStatus.EXPIRED
                self._resolved[request_id] = request
                del self._pending[request_id]
            raise
        
        return self._resolved.get(request_id) or self._pending.get(request_id)

    def get_pending(self) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        return list(self._pending.values())

    def get_pending_for_incident(self, incident_id: str) -> ApprovalRequest | None:
        """Get pending approval for a specific incident."""
        for request in self._pending.values():
            if request.incident_id == incident_id:
                return request
        return None

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID."""
        return self._pending.get(request_id) or self._resolved.get(request_id)

    def get_resolved(self, limit: int = 100) -> list[ApprovalRequest]:
        """Get resolved approval requests."""
        requests = list(self._resolved.values())
        requests.sort(key=lambda r: r.resolved_at or datetime.min, reverse=True)
        return requests[:limit]

    async def check_expirations(self) -> list[ApprovalRequest]:
        """
        Check for expired requests and escalate/expire them.
        
        Returns:
            List of expired or escalated requests
        """
        now = datetime.utcnow()
        affected = []
        
        for request_id, request in list(self._pending.items()):
            if request.expires_at and now > request.expires_at:
                # Escalate or expire based on severity
                if request.severity in ("P1", "P2") and request.escalation_level < 2:
                    # Escalate high-severity incidents
                    request.escalation_level += 1
                    request.expires_at = now + self._escalation_timeout
                    request.status = ApprovalStatus.ESCALATED
                    
                    self._logger.warning(
                        "approval_escalated",
                        request_id=request_id,
                        escalation_level=request.escalation_level,
                    )
                    
                    await self._send_notifications(request)
                else:
                    # Expire the request
                    request.status = ApprovalStatus.EXPIRED
                    request.resolved_at = now
                    self._resolved[request_id] = request
                    del self._pending[request_id]
                    
                    if request_id in self._approval_events:
                        self._approval_events[request_id].set()
                    
                    self._logger.warning(
                        "approval_expired",
                        request_id=request_id,
                        incident_id=request.incident_id,
                    )
                
                affected.append(request)
        
        return affected

    def stats(self) -> dict[str, Any]:
        """Get approval manager statistics."""
        pending_by_severity = {}
        for r in self._pending.values():
            pending_by_severity[r.severity] = pending_by_severity.get(r.severity, 0) + 1
        
        return {
            "pending_count": len(self._pending),
            "resolved_count": len(self._resolved),
            "pending_by_severity": pending_by_severity,
        }


# Singleton instance
_approval_manager: ApprovalManager | None = None


def get_approval_manager() -> ApprovalManager:
    """Get or create the approval manager singleton."""
    global _approval_manager
    
    if _approval_manager is None:
        _approval_manager = ApprovalManager()
    
    return _approval_manager
