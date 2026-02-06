"""
Human-in-the-Loop (HITL) module.

Provides approval workflows for remediation actions that
require human review before execution.
"""

from src.hitl.approval_manager import (
    ApprovalDecision,
    ApprovalManager,
    ApprovalRequest,
    ApprovalStatus,
    get_approval_manager,
)
from src.hitl.slack_notifier import SlackNotifier, get_slack_notifier

__all__ = [
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalStatus",
    "get_approval_manager",
    "SlackNotifier",
    "get_slack_notifier",
]
