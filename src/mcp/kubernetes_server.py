"""
Kubernetes MCP Server Stub.

Provides a Model Context Protocol (MCP) interface for interacting
with Kubernetes clusters. This is a stub implementation for demonstration.

In production, this would connect to a real Kubernetes API server.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import random

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KubernetesResource:
    """Represents a Kubernetes resource."""
    
    kind: str
    name: str
    namespace: str
    metadata: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                **self.metadata,
            },
            "spec": self.spec,
            "status": self.status,
        }


class KubernetesMCPServer:
    """
    MCP Server for Kubernetes cluster operations.
    
    This is a stub implementation that generates mock data.
    In production, replace mock methods with actual Kubernetes API calls.
    
    Supported tools:
    - get_pods: List pods in a namespace
    - get_pod: Get specific pod details
    - get_deployments: List deployments
    - get_events: Get cluster events
    - describe_pod: Get detailed pod description
    - get_logs: Get container logs
    - scale_deployment: Scale a deployment (HITL required)
    - rollback_deployment: Rollback a deployment (HITL required)
    """

    def __init__(
        self,
        kubeconfig: str | None = None,
        context: str | None = None,
    ) -> None:
        """
        Initialize Kubernetes MCP server.
        
        Args:
            kubeconfig: Path to kubeconfig file
            context: Kubernetes context to use
        """
        self._kubeconfig = kubeconfig
        self._context = context
        self._logger = logger.bind(
            component="kubernetes_mcp",
            context=context,
        )
        self._logger.info("kubernetes_mcp_initialized")

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Get available MCP tools.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "k8s_get_pods",
                "description": "List pods in a namespace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace",
                        },
                        "label_selector": {
                            "type": "string",
                            "description": "Label selector (e.g., 'app=nginx')",
                        },
                    },
                    "required": ["namespace"],
                },
            },
            {
                "name": "k8s_get_pod",
                "description": "Get details of a specific pod",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Pod name",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace",
                        },
                    },
                    "required": ["name", "namespace"],
                },
            },
            {
                "name": "k8s_get_deployments",
                "description": "List deployments in a namespace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace",
                        },
                    },
                    "required": ["namespace"],
                },
            },
            {
                "name": "k8s_get_events",
                "description": "Get events for a namespace or resource",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace",
                        },
                        "resource_name": {
                            "type": "string",
                            "description": "Optional: filter events for specific resource",
                        },
                    },
                    "required": ["namespace"],
                },
            },
            {
                "name": "k8s_get_logs",
                "description": "Get container logs from a pod",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Pod name",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace",
                        },
                        "container": {
                            "type": "string",
                            "description": "Container name (optional if pod has one container)",
                        },
                        "tail_lines": {
                            "type": "integer",
                            "description": "Number of lines from end of logs",
                        },
                        "previous": {
                            "type": "boolean",
                            "description": "Get logs from previous container instance",
                        },
                    },
                    "required": ["name", "namespace"],
                },
            },
            {
                "name": "k8s_scale_deployment",
                "description": "Scale a deployment (requires approval)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Deployment name",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace",
                        },
                        "replicas": {
                            "type": "integer",
                            "description": "Target replica count",
                        },
                    },
                    "required": ["name", "namespace", "replicas"],
                },
                "requires_approval": True,
            },
            {
                "name": "k8s_rollback_deployment",
                "description": "Rollback a deployment to previous revision (requires approval)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Deployment name",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace",
                        },
                        "revision": {
                            "type": "integer",
                            "description": "Target revision (0 for previous)",
                        },
                    },
                    "required": ["name", "namespace"],
                },
                "requires_approval": True,
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

        handlers = {
            "k8s_get_pods": self._get_pods,
            "k8s_get_pod": self._get_pod,
            "k8s_get_deployments": self._get_deployments,
            "k8s_get_events": self._get_events,
            "k8s_get_logs": self._get_logs,
            "k8s_scale_deployment": self._scale_deployment,
            "k8s_rollback_deployment": self._rollback_deployment,
        }

        handler = handlers.get(tool_name)
        if handler:
            return await handler(parameters)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _get_pods(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        List pods in namespace.
        
        STUB: Returns mock pod data.
        """
        namespace = params.get("namespace", "default")
        
        pods = self._generate_mock_pods(namespace)
        
        return {
            "pods": [p.to_dict() for p in pods],
            "total": len(pods),
        }

    async def _get_pod(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Get specific pod details.
        
        STUB: Returns mock pod data.
        """
        name = params.get("name", "")
        namespace = params.get("namespace", "default")
        
        pod = self._generate_mock_pod(name, namespace)
        return pod.to_dict()

    async def _get_deployments(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        List deployments in namespace.
        
        STUB: Returns mock deployment data.
        """
        namespace = params.get("namespace", "default")
        
        deployments = [
            KubernetesResource(
                kind="Deployment",
                name="api-server",
                namespace=namespace,
                metadata={
                    "creationTimestamp": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                    "labels": {"app": "api-server"},
                },
                spec={
                    "replicas": 3,
                    "selector": {"matchLabels": {"app": "api-server"}},
                },
                status={
                    "replicas": 3,
                    "readyReplicas": 3,
                    "availableReplicas": 3,
                    "conditions": [
                        {"type": "Available", "status": "True"},
                        {"type": "Progressing", "status": "True"},
                    ],
                },
            ),
            KubernetesResource(
                kind="Deployment",
                name="worker",
                namespace=namespace,
                metadata={
                    "creationTimestamp": (datetime.utcnow() - timedelta(days=15)).isoformat(),
                    "labels": {"app": "worker"},
                },
                spec={
                    "replicas": 5,
                    "selector": {"matchLabels": {"app": "worker"}},
                },
                status={
                    "replicas": 5,
                    "readyReplicas": 4,
                    "availableReplicas": 4,
                    "conditions": [
                        {"type": "Available", "status": "True"},
                        {"type": "Progressing", "status": "True"},
                    ],
                },
            ),
        ]
        
        return {
            "deployments": [d.to_dict() for d in deployments],
            "total": len(deployments),
        }

    async def _get_events(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Get cluster events.
        
        STUB: Returns mock events.
        """
        namespace = params.get("namespace", "default")
        resource_name = params.get("resource_name")
        
        events = [
            {
                "type": "Warning",
                "reason": "BackOff",
                "message": "Back-off restarting failed container",
                "involvedObject": {
                    "kind": "Pod",
                    "name": "worker-abc123",
                    "namespace": namespace,
                },
                "count": 5,
                "firstTimestamp": (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
                "lastTimestamp": datetime.utcnow().isoformat(),
            },
            {
                "type": "Normal",
                "reason": "Scheduled",
                "message": "Successfully assigned pod to node-1",
                "involvedObject": {
                    "kind": "Pod",
                    "name": "api-server-xyz789",
                    "namespace": namespace,
                },
                "count": 1,
                "firstTimestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                "lastTimestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            },
            {
                "type": "Warning",
                "reason": "OOMKilled",
                "message": "Container killed due to OOM",
                "involvedObject": {
                    "kind": "Pod",
                    "name": "worker-def456",
                    "namespace": namespace,
                },
                "count": 3,
                "firstTimestamp": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
                "lastTimestamp": (datetime.utcnow() - timedelta(minutes=2)).isoformat(),
            },
        ]
        
        if resource_name:
            events = [e for e in events if e["involvedObject"]["name"] == resource_name]
        
        return {"events": events}

    async def _get_logs(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Get container logs.
        
        STUB: Returns mock log lines.
        """
        name = params.get("name", "")
        namespace = params.get("namespace", "default")
        tail_lines = params.get("tail_lines", 100)
        previous = params.get("previous", False)
        
        # Generate mock logs
        log_lines = []
        base_time = datetime.utcnow() - timedelta(minutes=5)
        
        log_messages = [
            "INFO: Starting application...",
            "INFO: Connecting to database...",
            "INFO: Database connection established",
            "INFO: Loading configuration...",
            "WARN: High memory usage detected: 85%",
            "INFO: Processing request batch",
            "ERROR: Failed to connect to external service",
            "INFO: Retrying connection...",
            "INFO: Connection restored",
            "WARN: CPU usage above threshold",
        ]
        
        if previous:
            log_messages = [
                "ERROR: Out of memory",
                "ERROR: Application crashed",
                "INFO: Shutting down...",
            ]
        
        for i, msg in enumerate(log_messages[:tail_lines]):
            timestamp = (base_time + timedelta(seconds=i * 30)).isoformat()
            log_lines.append(f"{timestamp} {msg}")
        
        return {
            "pod": name,
            "namespace": namespace,
            "logs": "\n".join(log_lines),
            "previous": previous,
        }

    async def _scale_deployment(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Scale deployment.
        
        STUB: Returns mock scaling result.
        NOTE: In production, this requires HITL approval.
        """
        name = params.get("name", "")
        namespace = params.get("namespace", "default")
        replicas = params.get("replicas", 1)
        
        return {
            "success": True,
            "deployment": name,
            "namespace": namespace,
            "previousReplicas": 3,
            "newReplicas": replicas,
            "message": f"Deployment {name} scaled to {replicas} replicas",
            "requires_approval": True,
        }

    async def _rollback_deployment(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Rollback deployment.
        
        STUB: Returns mock rollback result.
        NOTE: In production, this requires HITL approval.
        """
        name = params.get("name", "")
        namespace = params.get("namespace", "default")
        revision = params.get("revision", 0)
        
        return {
            "success": True,
            "deployment": name,
            "namespace": namespace,
            "previousRevision": 5,
            "targetRevision": revision if revision > 0 else 4,
            "message": f"Deployment {name} rolled back",
            "requires_approval": True,
        }

    def _generate_mock_pods(
        self,
        namespace: str,
        count: int = 5,
    ) -> list[KubernetesResource]:
        """Generate mock pod resources."""
        pods = []
        
        statuses = ["Running", "Running", "Running", "CrashLoopBackOff", "Running"]
        apps = ["api-server", "worker", "scheduler"]
        
        for i in range(count):
            app = apps[i % len(apps)]
            status = statuses[i % len(statuses)]
            
            restart_count = 0 if status == "Running" else random.randint(3, 10)
            
            pods.append(
                KubernetesResource(
                    kind="Pod",
                    name=f"{app}-{i:03d}",
                    namespace=namespace,
                    metadata={
                        "labels": {"app": app},
                        "creationTimestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                    },
                    spec={
                        "containers": [
                            {
                                "name": "main",
                                "image": f"myregistry/{app}:latest",
                                "resources": {
                                    "limits": {"cpu": "500m", "memory": "512Mi"},
                                    "requests": {"cpu": "200m", "memory": "256Mi"},
                                },
                            },
                        ],
                    },
                    status={
                        "phase": status,
                        "containerStatuses": [
                            {
                                "name": "main",
                                "ready": status == "Running",
                                "restartCount": restart_count,
                                "state": {
                                    "running" if status == "Running" else "waiting": {
                                        "reason": "CrashLoopBackOff" if status != "Running" else None,
                                    },
                                },
                            },
                        ],
                    },
                )
            )
        
        return pods

    def _generate_mock_pod(
        self,
        name: str,
        namespace: str,
    ) -> KubernetesResource:
        """Generate a single mock pod."""
        return KubernetesResource(
            kind="Pod",
            name=name,
            namespace=namespace,
            metadata={
                "labels": {"app": name.split("-")[0]},
                "creationTimestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "uid": "abc123-def456-ghi789",
            },
            spec={
                "containers": [
                    {
                        "name": "main",
                        "image": f"myregistry/{name.split('-')[0]}:latest",
                        "resources": {
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                            "requests": {"cpu": "200m", "memory": "256Mi"},
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8080},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10,
                        },
                    },
                ],
                "nodeName": "node-1",
            },
            status={
                "phase": "Running",
                "podIP": "10.0.0.42",
                "hostIP": "192.168.1.10",
                "containerStatuses": [
                    {
                        "name": "main",
                        "ready": True,
                        "restartCount": 2,
                        "state": {"running": {"startedAt": datetime.utcnow().isoformat()}},
                    },
                ],
            },
        )


# Singleton instance
_kubernetes_server: KubernetesMCPServer | None = None


def get_kubernetes_mcp() -> KubernetesMCPServer:
    """Get or create the Kubernetes MCP server singleton."""
    global _kubernetes_server
    
    if _kubernetes_server is None:
        _kubernetes_server = KubernetesMCPServer()
    
    return _kubernetes_server
