"""
Tests for the agent implementations.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.workflows.states import (
    IncidentSeverity,
    IncidentState,
    IncidentStatus,
    PolicyDecision,
    ProposedAction,
)


@pytest.fixture
def sample_incident_state():
    """Create a sample incident state for testing."""
    return IncidentState(
        incident_id="INC-TEST-001",
        title="Test High CPU Alert",
        description="CPU usage exceeded threshold",
        severity=IncidentSeverity.P3,
        status=IncidentStatus.DETECTED,
        cluster="test-cluster",
        namespace="test-namespace",
        alert_source="prometheus",
    )


class TestIncidentState:
    """Tests for IncidentState."""

    def test_create_incident_state(self, sample_incident_state):
        """Test creating an incident state."""
        assert sample_incident_state.incident_id == "INC-TEST-001"
        assert sample_incident_state.severity == IncidentSeverity.P3
        assert sample_incident_state.status == IncidentStatus.DETECTED

    def test_add_agent(self, sample_incident_state):
        """Test adding agent to state."""
        sample_incident_state.add_agent("monitor")
        sample_incident_state.add_agent("diagnose")
        
        assert "monitor" in sample_incident_state.agents_involved
        assert "diagnose" in sample_incident_state.agents_involved
        assert len(sample_incident_state.agents_involved) == 2

    def test_add_agent_no_duplicates(self, sample_incident_state):
        """Test that agents aren't duplicated."""
        sample_incident_state.add_agent("monitor")
        sample_incident_state.add_agent("monitor")
        
        assert sample_incident_state.agents_involved.count("monitor") == 1

    def test_update_cost(self, sample_incident_state):
        """Test cost tracking."""
        sample_incident_state.update_cost(100, 0.001)
        sample_incident_state.update_cost(200, 0.002)
        
        assert sample_incident_state.total_llm_tokens == 300
        assert sample_incident_state.total_llm_cost_usd == 0.003

    def test_add_error(self, sample_incident_state):
        """Test error recording."""
        sample_incident_state.add_error("Test error 1")
        sample_incident_state.add_error("Test error 2")
        
        assert len(sample_incident_state.errors) == 2
        assert "Test error 1" in sample_incident_state.errors[0]


class TestProposedAction:
    """Tests for ProposedAction."""

    def test_create_action(self):
        """Test creating a proposed action."""
        action = ProposedAction(
            action_id="action-001",
            action_type="scale",
            command="kubectl scale deployment/api --replicas=5",
            target_resource="deployment/api",
            namespace="production",
            description="Scale up API deployment",
            risk_level="low",
            estimated_impact="Increase capacity",
            rollback_command="kubectl scale deployment/api --replicas=3",
        )
        
        assert action.action_id == "action-001"
        assert action.action_type == "scale"
        assert action.risk_level == "low"


class TestMonitorAgent:
    """Tests for MonitorAgent."""

    @pytest.mark.asyncio
    async def test_monitor_agent_init(self):
        """Test monitor agent initialization."""
        from src.agents.monitor_agent import MonitorAgent
        
        agent = MonitorAgent()
        assert agent.name == "monitor"

    @pytest.mark.asyncio
    async def test_monitor_agent_process(self, sample_incident_state):
        """Test monitor agent processing."""
        from src.agents.monitor_agent import MonitorAgent
        
        agent = MonitorAgent()
        
        # Mock the LLM call
        with patch.object(agent, 'call_llm_structured') as mock_llm:
            from src.agents.monitor_agent import AlertClassification
            
            mock_llm.return_value = (
                AlertClassification(
                    severity="P3",
                    anomalies=["High CPU usage", "Pod restarts"],
                    affected_resources=["deployment/api"],
                    summary="CPU spike detected",
                    confidence=0.85,
                ),
                MagicMock(total_tokens=100, cost_usd=0.001),
            )
            
            result = await agent.process(sample_incident_state)
            
            assert result.status == IncidentStatus.DIAGNOSING
            assert len(result.anomalies) > 0


class TestPolicyAgent:
    """Tests for PolicyAgent."""

    @pytest.mark.asyncio
    async def test_policy_agent_init(self):
        """Test policy agent initialization."""
        from src.agents.policy_agent import PolicyAgent
        
        agent = PolicyAgent()
        assert agent.name == "policy"

    def test_policy_rule_matching(self):
        """Test policy rule pattern matching."""
        from src.agents.policy_agent import PolicyAgent
        
        agent = PolicyAgent()
        
        # Create a dangerous action
        action = ProposedAction(
            action_id="action-001",
            action_type="custom",
            command="kubectl delete namespace production",
            target_resource="namespace/production",
            namespace="production",
            description="Delete production namespace",
            risk_level="critical",
            estimated_impact="Complete outage",
        )
        
        violations = agent._check_action_against_rules(action)
        
        # Should detect PROD-001 violation
        blocked_violations = [v for v in violations if v.blocked]
        assert len(blocked_violations) > 0


class TestWorkflowStates:
    """Tests for workflow state transitions."""

    def test_severity_enum(self):
        """Test severity enum values."""
        assert IncidentSeverity.P1.value == "P1"
        assert IncidentSeverity.P4.value == "P4"

    def test_status_enum(self):
        """Test status enum values."""
        assert IncidentStatus.DETECTED.value == "detected"
        assert IncidentStatus.RESOLVED.value == "resolved"

    def test_policy_decision_enum(self):
        """Test policy decision enum values."""
        assert PolicyDecision.APPROVED.value == "APPROVED"
        assert PolicyDecision.BLOCKED.value == "BLOCKED"
        assert PolicyDecision.NEEDS_REVIEW.value == "NEEDS_REVIEW"
