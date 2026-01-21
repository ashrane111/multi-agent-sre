"""
Monitor Agent - Detects anomalies and classifies alerts.

Responsibilities:
- Query metrics from Prometheus (via MCP)
- Detect anomalies in metrics
- Classify incident severity (P1-P4)
- Identify affected resources
"""

from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.models.router import TaskType
from src.workflows.states import (
    IncidentSeverity,
    IncidentState,
    IncidentStatus,
)


class AlertClassification(BaseModel):
    """Structured output for alert classification."""

    severity: str = Field(description="Severity level: P1, P2, P3, or P4")
    anomalies: list[str] = Field(description="List of detected anomalies")
    affected_resources: list[str] = Field(description="List of affected Kubernetes resources")
    summary: str = Field(description="Brief summary of the alert")
    confidence: float = Field(description="Classification confidence 0-1")


class MonitorAgent(BaseAgent):
    """
    Agent responsible for monitoring and alert classification.
    
    Uses Prometheus metrics to:
    - Detect anomalies in system metrics
    - Classify alert severity
    - Identify affected resources
    """

    @property
    def name(self) -> str:
        return "monitor"

    @property
    def description(self) -> str:
        return "Monitors infrastructure and classifies alerts"

    async def process(self, state: IncidentState) -> IncidentState:
        """
        Process incoming alert and classify severity.
        
        Args:
            state: Current incident state with initial alert info
            
        Returns:
            Updated state with severity, anomalies, and affected resources
        """
        self._logger.info(
            "classifying_alert",
            incident_id=state.incident_id,
            alert_source=state.alert_source,
        )

        # Get metrics (will be replaced with MCP call in later phase)
        metrics = await self._get_metrics(state)

        # Classify the alert using LLM
        classification = await self._classify_alert(state, metrics)

        # Update state with classification results
        state.severity = IncidentSeverity(classification.severity)
        state.anomalies = classification.anomalies
        state.affected_resources = classification.affected_resources
        state.metrics = metrics
        state.status = IncidentStatus.DIAGNOSING

        self._logger.info(
            "alert_classified",
            incident_id=state.incident_id,
            severity=state.severity,
            anomaly_count=len(state.anomalies),
            affected_resource_count=len(state.affected_resources),
        )

        # Send A2A notification to diagnose agent
        self.create_message(
            to_agent="diagnose",
            message_type="notification",
            content={
                "event": "alert_classified",
                "severity": state.severity,
                "anomalies": state.anomalies,
            },
            state=state,
        )

        return state

    async def _get_metrics(self, state: IncidentState) -> dict:
        """
        Get relevant metrics for the incident.
        
        TODO: Replace with actual Prometheus MCP call in Phase 3.
        
        Args:
            state: Current incident state
            
        Returns:
            Dictionary of relevant metrics
        """
        # Mock metrics for now - will be replaced with MCP integration
        # This simulates what we'd get from Prometheus
        mock_metrics = {
            "cpu_usage_percent": 85.5,
            "memory_usage_percent": 72.3,
            "pod_restart_count": 5,
            "error_rate_5xx": 2.3,
            "latency_p99_ms": 450,
            "active_pods": 8,
            "desired_pods": 10,
            "container_status": {
                "running": 6,
                "waiting": 2,
                "terminated": 2,
            },
            "recent_events": [
                "Pod restart detected",
                "High CPU alert triggered",
                "Memory pressure warning",
            ],
        }

        self._logger.debug(
            "metrics_retrieved",
            cluster=state.cluster,
            namespace=state.namespace,
            metric_count=len(mock_metrics),
        )

        return mock_metrics

    async def _classify_alert(
        self,
        state: IncidentState,
        metrics: dict,
    ) -> AlertClassification:
        """
        Classify the alert severity using LLM.
        
        Args:
            state: Current incident state
            metrics: Retrieved metrics
            
        Returns:
            AlertClassification with severity and details
        """
        system_prompt = """You are an expert SRE monitoring agent responsible for classifying infrastructure alerts.

Your task is to analyze the provided metrics and alert information to:
1. Determine the incident severity (P1-P4)
2. Identify specific anomalies in the metrics
3. List affected Kubernetes resources

Severity Guidelines:
- P1 (Critical): Complete service outage, data loss risk, security breach
- P2 (High): Major degradation, >50% users affected, critical path broken
- P3 (Medium): Minor degradation, <50% users affected, workaround available
- P4 (Low): Minimal impact, cosmetic issues, non-critical path

Be precise and concise in your analysis."""

        user_prompt = f"""Analyze this infrastructure alert:

**Alert Source:** {state.alert_source}
**Cluster:** {state.cluster}
**Namespace:** {state.namespace}
**Initial Description:** {state.description}

**Current Metrics:**
- CPU Usage: {metrics.get('cpu_usage_percent', 'N/A')}%
- Memory Usage: {metrics.get('memory_usage_percent', 'N/A')}%
- Pod Restarts: {metrics.get('pod_restart_count', 'N/A')}
- 5xx Error Rate: {metrics.get('error_rate_5xx', 'N/A')}%
- P99 Latency: {metrics.get('latency_p99_ms', 'N/A')}ms
- Pods: {metrics.get('active_pods', 'N/A')}/{metrics.get('desired_pods', 'N/A')} running
- Container Status: {metrics.get('container_status', {})}

**Recent Events:**
{chr(10).join('- ' + e for e in metrics.get('recent_events', []))}

Classify this alert and identify all anomalies and affected resources."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        classification, _ = await self.call_llm_structured(
            messages=messages,
            response_model=AlertClassification,
            task_type=TaskType.CLASSIFICATION,
            state=state,
        )

        return classification
