"""
FastAPI routes for HITL approvals and audit trail.

Provides REST API endpoints for:
- Viewing and managing approval requests
- Querying audit trail
- Health and status endpoints
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.governance.audit_trail import AuditAction, get_audit_trail
from src.hitl.approval_manager import ApprovalStatus, get_approval_manager

# Create routers
approvals_router = APIRouter(prefix="/api/v1/approvals", tags=["Approvals"])
audit_router = APIRouter(prefix="/api/v1/audit", tags=["Audit"])


# ==================== Request/Response Models ====================

class ApprovalDecisionRequest(BaseModel):
    """Request body for approval/rejection."""
    
    approver: str = Field(description="Email or ID of the approver")
    notes: str = Field(default="", description="Optional notes")


class ApprovalResponse(BaseModel):
    """Response for approval request details."""
    
    request_id: str
    incident_id: str
    status: str
    severity: str
    actions: list[dict[str, Any]]
    policy_violations: list[dict[str, Any]]
    blast_radius_score: float
    root_cause: str
    diagnosis_confidence: float
    created_at: datetime
    expires_at: datetime | None
    resolved_at: datetime | None
    resolved_by: str | None
    resolution_notes: str | None
    escalation_level: int


class ApprovalListResponse(BaseModel):
    """Response for list of approvals."""
    
    pending: list[ApprovalResponse]
    pending_count: int


class AuditEntryResponse(BaseModel):
    """Response for audit entry."""
    
    entry_id: str
    timestamp: datetime
    action: str
    description: str
    incident_id: str | None
    request_id: str | None
    agent: str | None
    actor_type: str
    actor_id: str | None
    details: dict[str, Any]
    risk_level: str
    success: bool
    error_message: str | None


class AuditListResponse(BaseModel):
    """Response for list of audit entries."""
    
    entries: list[AuditEntryResponse]
    total: int


class AuditStatsResponse(BaseModel):
    """Response for audit statistics."""
    
    total_entries: int
    incidents_tracked: int
    action_counts: dict[str, int]
    risk_distribution: dict[str, int]
    failure_count: int


# ==================== Approval Endpoints ====================

@approvals_router.get("/pending", response_model=ApprovalListResponse)
async def get_pending_approvals() -> ApprovalListResponse:
    """Get all pending approval requests."""
    manager = get_approval_manager()
    pending = manager.get_pending()
    
    return ApprovalListResponse(
        pending=[
            ApprovalResponse(
                request_id=r.request_id,
                incident_id=r.incident_id,
                status=r.status.value,
                severity=r.severity,
                actions=r.actions,
                policy_violations=r.policy_violations,
                blast_radius_score=r.blast_radius_score,
                root_cause=r.root_cause,
                diagnosis_confidence=r.diagnosis_confidence,
                created_at=r.created_at,
                expires_at=r.expires_at,
                resolved_at=r.resolved_at,
                resolved_by=r.resolved_by,
                resolution_notes=r.resolution_notes,
                escalation_level=r.escalation_level,
            )
            for r in pending
        ],
        pending_count=len(pending),
    )


@approvals_router.get("/stats")
async def get_approval_stats() -> dict[str, Any]:
    """Get approval manager statistics."""
    manager = get_approval_manager()
    return manager.stats()


@approvals_router.get("/incident/{incident_id}")
async def get_approval_for_incident(incident_id: str) -> ApprovalResponse | dict:
    """Get pending approval for a specific incident."""
    manager = get_approval_manager()
    request = manager.get_pending_for_incident(incident_id)
    
    if not request:
        return {"message": f"No pending approval for incident {incident_id}"}
    
    return ApprovalResponse(
        request_id=request.request_id,
        incident_id=request.incident_id,
        status=request.status.value,
        severity=request.severity,
        actions=request.actions,
        policy_violations=request.policy_violations,
        blast_radius_score=request.blast_radius_score,
        root_cause=request.root_cause,
        diagnosis_confidence=request.diagnosis_confidence,
        created_at=request.created_at,
        expires_at=request.expires_at,
        resolved_at=request.resolved_at,
        resolved_by=request.resolved_by,
        resolution_notes=request.resolution_notes,
        escalation_level=request.escalation_level,
    )


@approvals_router.get("/{request_id}", response_model=ApprovalResponse)
async def get_approval(request_id: str) -> ApprovalResponse:
    """Get a specific approval request."""
    manager = get_approval_manager()
    request = manager.get_request(request_id)
    
    if not request:
        raise HTTPException(status_code=404, detail=f"Approval request {request_id} not found")
    
    return ApprovalResponse(
        request_id=request.request_id,
        incident_id=request.incident_id,
        status=request.status.value,
        severity=request.severity,
        actions=request.actions,
        policy_violations=request.policy_violations,
        blast_radius_score=request.blast_radius_score,
        root_cause=request.root_cause,
        diagnosis_confidence=request.diagnosis_confidence,
        created_at=request.created_at,
        expires_at=request.expires_at,
        resolved_at=request.resolved_at,
        resolved_by=request.resolved_by,
        resolution_notes=request.resolution_notes,
        escalation_level=request.escalation_level,
    )


@approvals_router.post("/{request_id}/approve")
async def approve_request(
    request_id: str,
    decision: ApprovalDecisionRequest,
) -> dict[str, Any]:
    """Approve a pending approval request."""
    manager = get_approval_manager()
    audit = get_audit_trail()
    
    try:
        result = await manager.approve(
            request_id=request_id,
            approver=decision.approver,
            notes=decision.notes,
        )
        
        # Log to audit trail
        request = manager.get_request(request_id)
        audit.log(
            action=AuditAction.APPROVAL_GRANTED,
            description=f"Approval granted by {decision.approver}",
            incident_id=request.incident_id if request else None,
            request_id=request_id,
            actor_type="human",
            actor_id=decision.approver,
            details={"notes": decision.notes},
        )
        
        return {
            "success": True,
            "request_id": result.request_id,
            "approved": result.approved,
            "approver": result.approver,
            "timestamp": result.timestamp.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@approvals_router.post("/{request_id}/reject")
async def reject_request(
    request_id: str,
    decision: ApprovalDecisionRequest,
) -> dict[str, Any]:
    """Reject a pending approval request."""
    manager = get_approval_manager()
    audit = get_audit_trail()
    
    try:
        result = await manager.reject(
            request_id=request_id,
            approver=decision.approver,
            notes=decision.notes,
        )
        
        # Log to audit trail
        request = manager.get_request(request_id)
        audit.log(
            action=AuditAction.APPROVAL_REJECTED,
            description=f"Approval rejected by {decision.approver}: {decision.notes}",
            incident_id=request.incident_id if request else None,
            request_id=request_id,
            actor_type="human",
            actor_id=decision.approver,
            details={"notes": decision.notes},
            risk_level="medium",
        )
        
        return {
            "success": True,
            "request_id": result.request_id,
            "approved": result.approved,
            "approver": result.approver,
            "timestamp": result.timestamp.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ==================== Audit Endpoints ====================

@audit_router.get("/entries", response_model=AuditListResponse)
async def get_audit_entries(
    limit: int = Query(default=100, le=1000),
    action: str | None = Query(default=None),
    risk_level: str | None = Query(default=None),
    success: bool | None = Query(default=None),
) -> AuditListResponse:
    """Get recent audit entries with optional filtering."""
    audit = get_audit_trail()
    
    # Convert action string to enum if provided
    action_enum = None
    if action:
        try:
            action_enum = AuditAction(action)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")
    
    entries = audit.get_recent(
        limit=limit,
        action=action_enum,
        risk_level=risk_level,
        success=success,
    )
    
    return AuditListResponse(
        entries=[
            AuditEntryResponse(
                entry_id=e.entry_id,
                timestamp=e.timestamp,
                action=e.action.value,
                description=e.description,
                incident_id=e.incident_id,
                request_id=e.request_id,
                agent=e.agent,
                actor_type=e.actor_type,
                actor_id=e.actor_id,
                details=e.details,
                risk_level=e.risk_level,
                success=e.success,
                error_message=e.error_message,
            )
            for e in entries
        ],
        total=len(entries),
    )


@audit_router.get("/incident/{incident_id}")
async def get_audit_for_incident(incident_id: str) -> AuditListResponse:
    """Get all audit entries for a specific incident."""
    audit = get_audit_trail()
    entries = audit.get_by_incident(incident_id)
    
    return AuditListResponse(
        entries=[
            AuditEntryResponse(
                entry_id=e.entry_id,
                timestamp=e.timestamp,
                action=e.action.value,
                description=e.description,
                incident_id=e.incident_id,
                request_id=e.request_id,
                agent=e.agent,
                actor_type=e.actor_type,
                actor_id=e.actor_id,
                details=e.details,
                risk_level=e.risk_level,
                success=e.success,
                error_message=e.error_message,
            )
            for e in entries
        ],
        total=len(entries),
    )


@audit_router.get("/violations")
async def get_policy_violations(
    start_date: datetime | None = Query(default=None),
    end_date: datetime | None = Query(default=None),
) -> AuditListResponse:
    """Get policy violation audit entries."""
    audit = get_audit_trail()
    entries = audit.get_policy_violations(start_date, end_date)
    
    return AuditListResponse(
        entries=[
            AuditEntryResponse(
                entry_id=e.entry_id,
                timestamp=e.timestamp,
                action=e.action.value,
                description=e.description,
                incident_id=e.incident_id,
                request_id=e.request_id,
                agent=e.agent,
                actor_type=e.actor_type,
                actor_id=e.actor_id,
                details=e.details,
                risk_level=e.risk_level,
                success=e.success,
                error_message=e.error_message,
            )
            for e in entries
        ],
        total=len(entries),
    )


@audit_router.get("/stats", response_model=AuditStatsResponse)
async def get_audit_stats() -> AuditStatsResponse:
    """Get audit trail statistics."""
    audit = get_audit_trail()
    stats = audit.stats()
    
    return AuditStatsResponse(**stats)


@audit_router.post("/export")
async def export_audit_log(
    filepath: str = Query(default="data/audit/export.json"),
    start_date: datetime | None = Query(default=None),
    end_date: datetime | None = Query(default=None),
) -> dict[str, Any]:
    """Export audit log to JSON file."""
    audit = get_audit_trail()
    count = audit.export_to_json(filepath, start_date, end_date)
    
    return {
        "success": True,
        "filepath": filepath,
        "entries_exported": count,
    }