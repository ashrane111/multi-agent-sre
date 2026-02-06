"""
MCP (Model Context Protocol) server implementations.

Provides external integrations for the SRE platform:
- Prometheus metrics retrieval
- Kubernetes cluster operations

These are stub implementations for demonstration.
In production, connect to real services.
"""

from src.mcp.kubernetes_server import KubernetesMCPServer, get_kubernetes_mcp
from src.mcp.prometheus_server import PrometheusMCPServer, get_prometheus_mcp

__all__ = [
    "PrometheusMCPServer",
    "get_prometheus_mcp",
    "KubernetesMCPServer",
    "get_kubernetes_mcp",
]
