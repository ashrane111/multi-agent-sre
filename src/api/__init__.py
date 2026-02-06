"""
API module for REST endpoints.

Provides FastAPI routes for:
- Approval management
- Audit trail queries
- Incident management
"""

from src.api.routes import approvals_router, audit_router

__all__ = [
    "approvals_router",
    "audit_router",
]
