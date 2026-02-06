"""
Prometheus MCP Server Stub.

Provides a Model Context Protocol (MCP) interface for querying
Prometheus metrics. This is a stub implementation for demonstration.

In production, this would connect to a real Prometheus server.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import random

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricResult:
    """Result from a Prometheus query."""
    
    metric_name: str
    labels: dict[str, str]
    value: float
    timestamp: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "labels": self.labels,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PrometheusQueryResult:
    """Result set from Prometheus query."""
    
    query: str
    result_type: str  # "vector", "matrix", "scalar"
    results: list[MetricResult] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "result_type": self.result_type,
            "results": [r.to_dict() for r in self.results],
        }


class PrometheusMCPServer:
    """
    MCP Server for Prometheus metrics queries.
    
    This is a stub implementation that generates mock data.
    In production, replace mock methods with actual Prometheus API calls.
    
    Supported tools:
    - query: Execute instant PromQL query
    - query_range: Execute range PromQL query
    - get_alerts: Get active alerts
    - get_targets: Get scrape targets status
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
    ) -> None:
        """
        Initialize Prometheus MCP server.
        
        Args:
            prometheus_url: URL of Prometheus server
        """
        self._prometheus_url = prometheus_url
        self._logger = logger.bind(
            component="prometheus_mcp",
            prometheus_url=prometheus_url,
        )
        self._logger.info("prometheus_mcp_initialized")

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Get available MCP tools.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "prometheus_query",
                "description": "Execute an instant PromQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "PromQL query string",
                        },
                        "time": {
                            "type": "string",
                            "description": "Evaluation timestamp (RFC3339 or Unix timestamp)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "prometheus_query_range",
                "description": "Execute a range PromQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "PromQL query string",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start timestamp",
                        },
                        "end": {
                            "type": "string",
                            "description": "End timestamp",
                        },
                        "step": {
                            "type": "string",
                            "description": "Query resolution step (e.g., '15s', '1m')",
                        },
                    },
                    "required": ["query", "start", "end", "step"],
                },
            },
            {
                "name": "prometheus_alerts",
                "description": "Get active Prometheus alerts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "description": "Optional label filter",
                        },
                    },
                },
            },
            {
                "name": "prometheus_targets",
                "description": "Get Prometheus scrape targets status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "enum": ["active", "dropped", "any"],
                            "description": "Filter by target state",
                        },
                    },
                },
            },
        ]

    async def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute an MCP tool.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        self._logger.debug(
            "executing_tool",
            tool_name=tool_name,
            parameters=parameters,
        )

        if tool_name == "prometheus_query":
            return await self._query(parameters)
        elif tool_name == "prometheus_query_range":
            return await self._query_range(parameters)
        elif tool_name == "prometheus_alerts":
            return await self._get_alerts(parameters)
        elif tool_name == "prometheus_targets":
            return await self._get_targets(parameters)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _query(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute instant query.
        
        STUB: Returns mock data based on query patterns.
        """
        query = params.get("query", "")
        
        # Generate mock results based on common query patterns
        results = self._generate_mock_results(query)
        
        return PrometheusQueryResult(
            query=query,
            result_type="vector",
            results=results,
        ).to_dict()

    async def _query_range(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute range query.
        
        STUB: Returns mock time series data.
        """
        query = params.get("query", "")
        
        # Generate mock time series
        results = self._generate_mock_results(query, count=10)
        
        return PrometheusQueryResult(
            query=query,
            result_type="matrix",
            results=results,
        ).to_dict()

    async def _get_alerts(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Get active alerts.
        
        STUB: Returns mock alerts.
        """
        return {
            "alerts": [
                {
                    "name": "HighCPUUsage",
                    "state": "firing",
                    "severity": "warning",
                    "labels": {
                        "namespace": "production",
                        "pod": "api-server-abc123",
                    },
                    "annotations": {
                        "summary": "High CPU usage detected",
                        "description": "CPU usage is above 80% for 5 minutes",
                    },
                    "activeAt": datetime.utcnow().isoformat(),
                },
                {
                    "name": "PodCrashLooping",
                    "state": "firing",
                    "severity": "critical",
                    "labels": {
                        "namespace": "production",
                        "pod": "worker-def456",
                    },
                    "annotations": {
                        "summary": "Pod is crash looping",
                        "description": "Pod has restarted 5 times in the last 10 minutes",
                    },
                    "activeAt": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                },
            ],
        }

    async def _get_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Get scrape targets.
        
        STUB: Returns mock target status.
        """
        state_filter = params.get("state", "any")
        
        targets = [
            {
                "job": "kubernetes-pods",
                "instance": "10.0.0.1:8080",
                "health": "up",
                "lastScrape": datetime.utcnow().isoformat(),
                "scrapeInterval": "15s",
            },
            {
                "job": "kubernetes-pods",
                "instance": "10.0.0.2:8080",
                "health": "up",
                "lastScrape": datetime.utcnow().isoformat(),
                "scrapeInterval": "15s",
            },
            {
                "job": "kubernetes-nodes",
                "instance": "node-1:9100",
                "health": "up",
                "lastScrape": datetime.utcnow().isoformat(),
                "scrapeInterval": "30s",
            },
        ]
        
        if state_filter == "active":
            targets = [t for t in targets if t["health"] == "up"]
        elif state_filter == "dropped":
            targets = [t for t in targets if t["health"] != "up"]
        
        return {"targets": targets}

    def _generate_mock_results(
        self,
        query: str,
        count: int = 3,
    ) -> list[MetricResult]:
        """
        Generate mock metric results based on query.
        
        Args:
            query: PromQL query string
            count: Number of results to generate
            
        Returns:
            List of mock MetricResults
        """
        results = []
        now = datetime.utcnow()
        
        # Detect metric type from query
        if "cpu" in query.lower():
            metric_name = "container_cpu_usage_seconds_total"
            base_value = 0.75
        elif "memory" in query.lower():
            metric_name = "container_memory_usage_bytes"
            base_value = 512 * 1024 * 1024  # 512MB
        elif "restart" in query.lower():
            metric_name = "kube_pod_container_status_restarts_total"
            base_value = 3.0
        elif "request" in query.lower() or "http" in query.lower():
            metric_name = "http_requests_total"
            base_value = 1000.0
        else:
            metric_name = "unknown_metric"
            base_value = 1.0
        
        pods = ["api-server", "worker", "scheduler"]
        namespaces = ["production", "staging"]
        
        for i in range(count):
            results.append(
                MetricResult(
                    metric_name=metric_name,
                    labels={
                        "pod": f"{pods[i % len(pods)]}-{i}",
                        "namespace": namespaces[i % len(namespaces)],
                        "container": "main",
                    },
                    value=base_value * (1 + random.uniform(-0.2, 0.2)),
                    timestamp=now - timedelta(seconds=i * 15),
                )
            )
        
        return results


# Singleton instance
_prometheus_server: PrometheusMCPServer | None = None


def get_prometheus_mcp() -> PrometheusMCPServer:
    """Get or create the Prometheus MCP server singleton."""
    global _prometheus_server
    
    if _prometheus_server is None:
        _prometheus_server = PrometheusMCPServer()
    
    return _prometheus_server
