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
    
    In production, this would:
    1. Create an approval request in the queue
    2. Send notification to Slack/PagerDuty
    3. Wait for response (with timeout)
    
    For now, auto-approves P3/P4 and simulates approval for P1/P2.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with approval status
    """
    incident_state = IncidentState.model_validate(state)
    
    logger.info(
        "human_approval_requested",
        incident_id=incident_state.incident_id,
        severity=incident_state.severity,
        violations=[v.rule_name for v in incident_state.policy_violations],
    )

    incident_state.approval_requested_at = datetime.utcnow()

    # For demo/testing: Auto-approve P3/P4, simulate quick approval for P1/P2
    # In production, this would wait for actual human response
    
    if incident_state.severity in [IncidentSeverity.P3.value, IncidentSeverity.P4.value]:
        # Auto-approve low severity
        incident_state.human_approved = True
        incident_state.human_approver = "auto-approval-low-severity"
        incident_state.approval_notes = "Automatically approved due to low severity"
        logger.info(
            "auto_approved",
            incident_id=incident_state.incident_id,
            reason="low_severity",
        )
    else:
        # Simulate human approval for high severity (in production, would wait)
        # For demo, we'll approve after a short delay
        await asyncio.sleep(1)  # Simulate approval time
        
        incident_state.human_approved = True
        incident_state.human_approver = "sre-oncall@example.com"
        incident_state.approval_notes = "Approved after reviewing blast radius and actions"
        logger.info(
            "human_approved",
            incident_id=incident_state.incident_id,
            approver=incident_state.human_approver,
        )

    incident_state.approval_received_at = datetime.utcnow()
    incident_state.status = IncidentStatus.REMEDIATING

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
