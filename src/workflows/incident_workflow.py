"""
LangGraph Incident Response Workflow.

Defines the state machine that orchestrates all agents:
Monitor → Diagnose → Policy → [Human Approval] → Remediate → Report

Features:
- Conditional routing based on policy decisions
- Human-in-the-loop integration
- State persistence with checkpointing
- Error handling and recovery
"""

import asyncio
import os
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

import structlog
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.diagnose_agent import DiagnoseAgent
from src.agents.monitor_agent import MonitorAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.remediate_agent import RemediateAgent
from src.agents.report_agent import ReportAgent
from src.governance.audit_trail import AuditAction, get_audit_trail
from src.hitl.approval_manager import ApprovalStatus, get_approval_manager
from src.hitl.slack_notifier import get_slack_notifier
from src.logging_config import set_correlation_id, set_incident_id
from src.workflows.states import (
    IncidentSeverity,
    IncidentState,
    IncidentStatus,
    PolicyDecision,
    WorkflowState,
)

logger = structlog.get_logger(__name__)


# ==================== Initialize Agents ====================

monitor_agent = MonitorAgent()
diagnose_agent = DiagnoseAgent()
policy_agent = PolicyAgent()
remediate_agent = RemediateAgent()
report_agent = ReportAgent()


# ==================== Routing Functions ====================


def route_after_policy(state: WorkflowState) -> Literal["remediate", "human_approval", "report"]:
    """
    Route based on policy decision.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name
    """
    decision = state.get("policy_decision")

    if decision == PolicyDecision.APPROVED.value:
        logger.debug("routing_to_remediate", reason="Policy approved")
        return "remediate"

    elif decision == PolicyDecision.NEEDS_REVIEW.value:
        logger.debug("routing_to_human_approval", reason="Policy needs review")
        return "human_approval"

    elif decision == PolicyDecision.BLOCKED.value:
        logger.debug("routing_to_report", reason="Policy blocked")
        return "report"

    # Default to human approval for safety
    logger.warning("routing_to_human_approval", reason="Unknown policy decision")
    return "human_approval"


def route_after_approval(state: WorkflowState) -> Literal["remediate", "report"]:
    """
    Route based on human approval.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name
    """
    if state.get("human_approved") is True:
        logger.debug("routing_to_remediate", reason="Human approved")
        return "remediate"
    else:
        logger.debug("routing_to_report", reason="Human rejected or timeout")
        return "report"


# ==================== Human Approval Node ====================


async def human_approval_node(state: WorkflowState) -> WorkflowState:
    """
    Human-in-the-loop approval node.
    
    Creates an approval request and either:
    - Auto-approves for P3/P4 severity (configurable)
    - Waits for human approval via API for P1/P2
    - Falls back to simulated approval in demo mode
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with approval status
    """
    incident_state = IncidentState.model_validate(state)
    approval_manager = get_approval_manager()
    audit_trail = get_audit_trail()
    slack_notifier = get_slack_notifier()
    
    # Register Slack notifier for approval notifications
    if slack_notifier not in approval_manager._notification_handlers:
        approval_manager.register_notification_handler(slack_notifier.send_approval_request)
    
    logger.info(
        "human_approval_requested",
        incident_id=incident_state.incident_id,
        severity=incident_state.severity,
        violations=[v.rule_name for v in incident_state.policy_violations],
    )

    incident_state.approval_requested_at = datetime.utcnow()
    
    # Prepare actions for approval request
    actions = [
        {
            "action_id": a.action_id,
            "action_type": a.action_type,
            "command": a.command,
            "target_resource": a.target_resource,
            "description": a.description,
            "risk_level": a.risk_level,
        }
        for a in incident_state.proposed_actions
    ]
    
    policy_violations = [
        {
            "rule_id": v.rule_id,
            "rule_name": v.rule_name,
            "severity": v.severity,
            "message": v.message,
        }
        for v in incident_state.policy_violations
    ]
    
    # Create approval request
    approval_request = await approval_manager.create_request(
        incident_id=incident_state.incident_id,
        actions=actions,
        severity=incident_state.severity.value if hasattr(incident_state.severity, 'value') else incident_state.severity,
        policy_violations=policy_violations,
        blast_radius_score=incident_state.blast_radius_score or 0.0,
        root_cause=incident_state.root_cause or "",
        diagnosis_confidence=incident_state.confidence or 0.0,
    )
    
    # Log to audit trail
    audit_trail.log(
        action=AuditAction.APPROVAL_REQUESTED,
        description=f"Approval requested for {len(actions)} actions",
        incident_id=incident_state.incident_id,
        request_id=approval_request.request_id,
        agent="workflow",
        details={
            "action_count": len(actions),
            "violation_count": len(policy_violations),
            "blast_radius": incident_state.blast_radius_score,
        },
        risk_level="medium" if incident_state.severity in ["P1", "P2", IncidentSeverity.P1.value, IncidentSeverity.P2.value] else "low",
    )

    # Determine approval mode
    interactive_mode = os.environ.get("SRE_INTERACTIVE_MODE", "false").lower() == "true"
    auto_approve_low_severity = os.environ.get("SRE_AUTO_APPROVE_LOW", "true").lower() == "true"
    
    severity_value = incident_state.severity.value if hasattr(incident_state.severity, 'value') else incident_state.severity
    is_low_severity = severity_value in [IncidentSeverity.P3.value, IncidentSeverity.P4.value, "P3", "P4"]
    
    if auto_approve_low_severity and is_low_severity:
        # Auto-approve low severity incidents
        await approval_manager.approve(
            request_id=approval_request.request_id,
            approver="auto-approval-system",
            notes="Automatically approved due to low severity",
        )
        incident_state.human_approved = True
        incident_state.human_approver = "auto-approval-low-severity"
        incident_state.approval_notes = "Automatically approved due to low severity"
        
        audit_trail.log(
            action=AuditAction.APPROVAL_GRANTED,
            description="Auto-approved due to low severity",
            incident_id=incident_state.incident_id,
            request_id=approval_request.request_id,
            actor_type="system",
            actor_id="auto-approval-system",
        )
        
        logger.info(
            "auto_approved",
            incident_id=incident_state.incident_id,
            reason="low_severity",
        )
        
    elif interactive_mode:
        # Wait for human approval via API
        logger.info(
            "waiting_for_human_approval",
            incident_id=incident_state.incident_id,
            request_id=approval_request.request_id,
            timeout_minutes=15,
        )
        
        try:
            # Wait for approval (15 minute timeout)
            resolved_request = await approval_manager.wait_for_approval(
                request_id=approval_request.request_id,
                timeout_seconds=900,  # 15 minutes
            )
            
            if resolved_request.status == ApprovalStatus.APPROVED:
                incident_state.human_approved = True
                incident_state.human_approver = resolved_request.resolved_by
                incident_state.approval_notes = resolved_request.resolution_notes
            else:
                incident_state.human_approved = False
                incident_state.human_approver = resolved_request.resolved_by
                incident_state.approval_notes = f"Rejected: {resolved_request.resolution_notes}"
                
        except asyncio.TimeoutError:
            # Approval timed out
            incident_state.human_approved = False
            incident_state.approval_notes = "Approval request timed out"
            
            audit_trail.log(
                action=AuditAction.APPROVAL_EXPIRED,
                description="Approval request timed out after 15 minutes",
                incident_id=incident_state.incident_id,
                request_id=approval_request.request_id,
                risk_level="high",
            )
            
            logger.warning(
                "approval_timeout",
                incident_id=incident_state.incident_id,
                request_id=approval_request.request_id,
            )
    else:
        # Demo mode: Simulate human approval after brief delay
        await asyncio.sleep(1)
        
        await approval_manager.approve(
            request_id=approval_request.request_id,
            approver="sre-oncall@example.com",
            notes="Approved after reviewing blast radius and actions",
        )
        
        incident_state.human_approved = True
        incident_state.human_approver = "sre-oncall@example.com"
        incident_state.approval_notes = "Approved after reviewing blast radius and actions"
        
        audit_trail.log(
            action=AuditAction.APPROVAL_GRANTED,
            description="Simulated approval in demo mode",
            incident_id=incident_state.incident_id,
            request_id=approval_request.request_id,
            actor_type="human",
            actor_id="sre-oncall@example.com",
        )
        
        logger.info(
            "human_approved",
            incident_id=incident_state.incident_id,
            approver=incident_state.human_approver,
        )

    incident_state.approval_received_at = datetime.utcnow()
    incident_state.status = IncidentStatus.REMEDIATING if incident_state.human_approved else IncidentStatus.CANCELLED

    return incident_state.model_dump(mode="json")


# ==================== Build Workflow Graph ====================


def create_incident_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for incident response.
    
    Workflow:
        monitor → diagnose → policy_check → [routing]
            → APPROVED: remediate → report → END
            → NEEDS_REVIEW: human_approval → [routing]
                → approved: remediate → report → END
                → rejected: report → END
            → BLOCKED: report → END
    
    Returns:
        Compiled LangGraph workflow
    """
    # Create the graph with state schema
    workflow = StateGraph(WorkflowState)

    # Add nodes (each node is an agent or function)
    workflow.add_node("monitor", monitor_agent)
    workflow.add_node("diagnose", diagnose_agent)
    workflow.add_node("policy_check", policy_agent)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("remediate", remediate_agent)
    workflow.add_node("report", report_agent)

    # Define edges
    # Linear flow until policy check
    workflow.add_edge("monitor", "diagnose")
    workflow.add_edge("diagnose", "policy_check")

    # Conditional routing after policy check
    workflow.add_conditional_edges(
        "policy_check",
        route_after_policy,
        {
            "remediate": "remediate",
            "human_approval": "human_approval",
            "report": "report",
        },
    )

    # Conditional routing after human approval
    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "remediate": "remediate",
            "report": "report",
        },
    )

    # Final edges to report and END
    workflow.add_edge("remediate", "report")
    workflow.add_edge("report", END)

    # Set entry point
    workflow.set_entry_point("monitor")

    return workflow


# ==================== Workflow Executor ====================


class IncidentWorkflow:
    """
    High-level interface for running incident workflows.
    
    Provides:
    - Easy workflow creation and execution
    - State persistence with checkpointing
    - Error handling and logging
    """

    def __init__(self, enable_checkpointing: bool = True) -> None:
        """
        Initialize the workflow.
        
        Args:
            enable_checkpointing: Whether to enable state checkpointing
        """
        self._graph = create_incident_workflow()
        
        # Setup checkpointing for state persistence
        if enable_checkpointing:
            self._checkpointer = MemorySaver()
            self._workflow = self._graph.compile(checkpointer=self._checkpointer)
        else:
            self._checkpointer = None
            self._workflow = self._graph.compile()

        logger.info("incident_workflow_initialized", checkpointing=enable_checkpointing)

    async def run(
        self,
        incident_id: str | None = None,
        title: str = "Infrastructure Incident",
        description: str = "",
        cluster: str = "production-cluster",
        namespace: str = "default",
        alert_source: str = "prometheus",
        severity: IncidentSeverity = IncidentSeverity.P3,
    ) -> IncidentState:
        """
        Run the incident workflow.
        
        Args:
            incident_id: Optional incident ID (generated if not provided)
            title: Incident title
            description: Incident description
            cluster: Affected cluster
            namespace: Affected namespace
            alert_source: Source of the alert
            severity: Initial severity assessment
            
        Returns:
            Final incident state after workflow completion
        """
        # Generate incident ID if not provided
        if incident_id is None:
            incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}"

        # Set logging context
        set_correlation_id(incident_id)
        set_incident_id(incident_id)

        # Create initial state
        initial_state = IncidentState(
            incident_id=incident_id,
            title=title,
            description=description,
            cluster=cluster,
            namespace=namespace,
            alert_source=alert_source,
            severity=severity,
            status=IncidentStatus.DETECTED,
        )

        logger.info(
            "workflow_started",
            incident_id=incident_id,
            title=title,
            cluster=cluster,
            namespace=namespace,
            severity=severity,
        )

        start_time = datetime.utcnow()

        try:
            # Run the workflow
            config = {"configurable": {"thread_id": incident_id}}
            
            final_state_dict = await self._workflow.ainvoke(
                initial_state.model_dump(mode="json"),
                config=config,
            )

            # Convert back to typed state
            final_state = IncidentState.model_validate(final_state_dict)

            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                "workflow_completed",
                incident_id=incident_id,
                status=final_state.status,
                duration_seconds=duration,
                total_cost_usd=final_state.total_llm_cost_usd,
                agents_involved=final_state.agents_involved,
            )

            return final_state

        except Exception as e:
            logger.error(
                "workflow_failed",
                incident_id=incident_id,
                error=str(e),
                exc_info=e,
            )
            raise

    async def resume(self, incident_id: str) -> IncidentState | None:
        """
        Resume a workflow from checkpoint.
        
        Args:
            incident_id: Incident ID to resume
            
        Returns:
            Final state or None if not found
        """
        if not self._checkpointer:
            logger.warning("checkpointing_disabled", incident_id=incident_id)
            return None

        config = {"configurable": {"thread_id": incident_id}}
        
        # Get current state from checkpoint
        state = await self._workflow.aget_state(config)
        
        if not state.values:
            logger.warning("no_checkpoint_found", incident_id=incident_id)
            return None

        logger.info(
            "workflow_resuming",
            incident_id=incident_id,
            current_status=state.values.get("status"),
        )

        # Resume execution
        final_state_dict = await self._workflow.ainvoke(None, config=config)
        
        return IncidentState.model_validate(final_state_dict)

    def get_workflow_graph(self) -> StateGraph:
        """Get the underlying workflow graph for visualization."""
        return self._graph


# ==================== Convenience Functions ====================


async def run_incident_workflow(
    title: str,
    cluster: str,
    namespace: str = "default",
    description: str = "",
    alert_source: str = "prometheus",
    severity: IncidentSeverity = IncidentSeverity.P3,
) -> IncidentState:
    """
    Convenience function to run an incident workflow.
    
    Args:
        title: Incident title
        cluster: Affected cluster
        namespace: Affected namespace
        description: Optional description
        alert_source: Source of the alert
        severity: Initial severity
        
    Returns:
        Final incident state
    """
    workflow = IncidentWorkflow()
    return await workflow.run(
        title=title,
        description=description,
        cluster=cluster,
        namespace=namespace,
        alert_source=alert_source,
        severity=severity,
    )
