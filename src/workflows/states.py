"""
State definitions for the incident response workflow.

Defines the TypedDict state that flows through the LangGraph state machine.
All agents read from and write to this shared state.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class IncidentSeverity(str, Enum):
    """Incident severity levels."""

    P1 = "P1"  # Critical - Customer facing outage
    P2 = "P2"  # High - Significant degradation
    P3 = "P3"  # Medium - Minor impact
    P4 = "P4"  # Low - Minimal impact


class IncidentStatus(str, Enum):
    """Incident workflow status."""

    DETECTED = "detected"
    DIAGNOSING = "diagnosing"
    POLICY_CHECK = "policy_check"
    AWAITING_APPROVAL = "awaiting_approval"
    REMEDIATING = "remediating"
    RESOLVED = "resolved"
    FAILED = "failed"
    ESCALATED = "escalated"


class PolicyDecision(str, Enum):
    """Policy agent decisions."""

    APPROVED = "APPROVED"  # Safe to auto-execute
    BLOCKED = "BLOCKED"  # Violates immutable rules
    NEEDS_REVIEW = "NEEDS_REVIEW"  # Requires human approval


class AgentMessage(BaseModel):
    """Message passed between agents (A2A protocol)."""

    from_agent: str
    to_agent: str
    message_type: Literal["request", "response", "notification"]
    content: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str | None = None


class ProposedAction(BaseModel):
    """A proposed remediation action."""

    action_id: str
    action_type: Literal["scale", "restart", "rollback", "config_change", "custom"]
    command: str
    target_resource: str
    namespace: str
    description: str
    risk_level: Literal["low", "medium", "high", "critical"]
    estimated_impact: str
    rollback_command: str | None = None


class ExecutedAction(BaseModel):
    """Record of an executed action."""

    action: ProposedAction
    status: Literal["success", "failed", "skipped"]
    result: str | None = None
    error: str | None = None
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: float | None = None


class PolicyViolation(BaseModel):
    """A policy rule violation."""

    rule_id: str
    rule_name: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    action: str
    blocked: bool
    message: str


class SimilarIncident(BaseModel):
    """A similar past incident from episodic memory."""

    incident_id: str
    similarity_score: float
    cluster: str
    symptoms: list[str]
    root_cause: str
    fix_applied: str
    resolution_time_minutes: float
    occurred_at: datetime


class RunbookMatch(BaseModel):
    """A matching runbook section from RAG."""

    runbook_name: str
    section: str
    content: str
    relevance_score: float


class IncidentState(BaseModel):
    """
    Complete state for an incident flowing through the workflow.
    
    This state is passed through all agents in the LangGraph workflow.
    Each agent reads what it needs and adds its outputs.
    """

    # ==================== Incident Identity ====================
    incident_id: str = Field(description="Unique incident identifier")
    title: str = Field(description="Short incident title")
    description: str = Field(default="", description="Detailed incident description")

    # ==================== Classification ====================
    severity: IncidentSeverity = Field(default=IncidentSeverity.P3)
    status: IncidentStatus = Field(default=IncidentStatus.DETECTED)
    cluster: str = Field(description="Affected Kubernetes cluster")
    namespace: str = Field(default="default", description="Affected namespace")

    # ==================== Detection (Monitor Agent) ====================
    alert_source: str = Field(default="", description="Source of the alert")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Relevant metrics")
    anomalies: list[str] = Field(default_factory=list, description="Detected anomalies")
    affected_resources: list[str] = Field(default_factory=list, description="Affected K8s resources")

    # ==================== Diagnosis (Diagnose Agent) ====================
    diagnosis: str | None = Field(default=None, description="Diagnosis summary")
    root_cause: str | None = Field(default=None, description="Identified root cause")
    confidence: float | None = Field(default=None, description="Diagnosis confidence 0-1")
    similar_past_incidents: list[SimilarIncident] = Field(
        default_factory=list, description="Similar incidents from memory"
    )
    runbook_matches: list[RunbookMatch] = Field(
        default_factory=list, description="Matching runbook sections"
    )
    memory_informed: bool = Field(
        default=False, description="Whether diagnosis used episodic memory"
    )

    # ==================== Remediation Planning ====================
    proposed_actions: list[ProposedAction] = Field(
        default_factory=list, description="Proposed remediation actions"
    )

    # ==================== Policy Check (Policy Agent) ====================
    policy_decision: PolicyDecision | None = Field(
        default=None, description="Policy agent decision"
    )
    policy_violations: list[PolicyViolation] = Field(
        default_factory=list, description="Policy violations found"
    )
    blast_radius_score: float | None = Field(
        default=None, description="Estimated blast radius 0-1"
    )

    # ==================== Human Approval (HITL) ====================
    human_approved: bool | None = Field(
        default=None, description="Human approval status"
    )
    human_approver: str | None = Field(default=None, description="Who approved")
    approval_notes: str | None = Field(default=None, description="Approval notes")
    approval_requested_at: datetime | None = Field(default=None)
    approval_received_at: datetime | None = Field(default=None)

    # ==================== Execution (Remediate Agent) ====================
    executed_actions: list[ExecutedAction] = Field(
        default_factory=list, description="Actions that were executed"
    )
    rollback_performed: bool = Field(default=False, description="Whether rollback was needed")

    # ==================== Resolution (Report Agent) ====================
    resolution_summary: str | None = Field(default=None, description="Final resolution summary")
    time_to_resolve_minutes: float | None = Field(default=None)
    slack_thread_ts: str | None = Field(default=None, description="Slack thread ID")

    # ==================== Metadata ====================
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    agents_involved: list[str] = Field(default_factory=list)
    
    # ==================== Cost Tracking ====================
    total_llm_tokens: int = Field(default=0)
    total_llm_cost_usd: float = Field(default=0.0)
    
    # ==================== A2A Communication ====================
    messages: list[AgentMessage] = Field(
        default_factory=list, description="Inter-agent messages"
    )
    
    # ==================== Error Handling ====================
    errors: list[str] = Field(default_factory=list, description="Errors encountered")
    retry_count: int = Field(default=0)

    def add_agent(self, agent_name: str) -> None:
        """Record that an agent was involved."""
        if agent_name not in self.agents_involved:
            self.agents_involved.append(agent_name)
        self.updated_at = datetime.utcnow()

    def add_message(self, message: AgentMessage) -> None:
        """Add an A2A message."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def add_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(f"[{datetime.utcnow().isoformat()}] {error}")
        self.updated_at = datetime.utcnow()

    def update_cost(self, tokens: int, cost_usd: float) -> None:
        """Update cost tracking."""
        self.total_llm_tokens += tokens
        self.total_llm_cost_usd += cost_usd
        self.updated_at = datetime.utcnow()

    class Config:
        """Pydantic config."""
        use_enum_values = True


# Type alias for LangGraph state
# LangGraph expects a TypedDict, but we use Pydantic for validation
# We convert to/from dict at the boundaries
WorkflowState = dict[str, Any]


def state_to_dict(state: IncidentState) -> WorkflowState:
    """Convert IncidentState to dict for LangGraph."""
    return state.model_dump(mode="json")


def dict_to_state(data: WorkflowState) -> IncidentState:
    """Convert dict from LangGraph to IncidentState."""
    return IncidentState.model_validate(data)
