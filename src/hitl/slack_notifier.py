"""
Slack notification handler for HITL approvals.

Sends formatted approval requests to Slack channels with
interactive buttons for approve/reject actions.
"""

from datetime import datetime
from typing import Any

import httpx
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class SlackNotifier:
    """
    Sends approval notifications to Slack.
    
    Features:
    - Formatted approval request messages
    - Interactive buttons (approve/reject)
    - Escalation notifications
    - Thread-based updates
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        channel: str = "#sre-incidents",
        bot_token: str | None = None,
    ) -> None:
        """
        Initialize Slack notifier.
        
        Args:
            webhook_url: Slack incoming webhook URL
            channel: Default channel for notifications
            bot_token: Bot token for interactive messages
        """
        self._webhook_url = webhook_url
        self._channel = channel
        self._bot_token = bot_token
        self._logger = logger.bind(component="slack_notifier")
        
        # Track message timestamps for threading
        self._message_threads: dict[str, str] = {}

    async def send_approval_request(self, request: Any) -> dict[str, Any]:
        """
        Send an approval request notification to Slack.
        
        Args:
            request: ApprovalRequest object
            
        Returns:
            Slack API response
        """
        # Build the message blocks
        blocks = self._build_approval_blocks(request)
        
        payload = {
            "channel": self._channel,
            "text": f"🔔 Approval Required: {request.incident_id}",
            "blocks": blocks,
        }
        
        # Add thread_ts if this is an escalation
        if request.incident_id in self._message_threads:
            payload["thread_ts"] = self._message_threads[request.incident_id]
        
        result = await self._send_message(payload)
        
        # Store thread timestamp for follow-ups
        if result.get("ts"):
            self._message_threads[request.incident_id] = result["ts"]
        
        self._logger.info(
            "approval_notification_sent",
            request_id=request.request_id,
            incident_id=request.incident_id,
            channel=self._channel,
        )
        
        return result

    def _build_approval_blocks(self, request: Any) -> list[dict]:
        """Build Slack Block Kit blocks for approval request."""
        severity_emoji = {
            "P1": "🔴",
            "P2": "🟠",
            "P3": "🟡",
            "P4": "🟢",
        }.get(request.severity, "⚪")
        
        escalation_text = ""
        if request.escalation_level > 0:
            escalation_text = f" ⚠️ *ESCALATION LEVEL {request.escalation_level}*"
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 Approval Required{escalation_text}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Incident:*\n{request.incident_id}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{severity_emoji} {request.severity}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Blast Radius:*\n{request.blast_radius_score:.0%}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence:*\n{request.diagnosis_confidence:.0%}",
                    },
                ],
            },
        ]
        
        # Root cause section
        if request.root_cause:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Root Cause:*\n{request.root_cause[:500]}",
                },
            })
        
        # Policy violations
        if request.policy_violations:
            violation_text = "\n".join([
                f"• [{v.get('severity', 'UNKNOWN')}] {v.get('rule_name', 'Unknown')}: {v.get('message', '')}"
                for v in request.policy_violations
            ])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Policy Violations:*\n{violation_text}",
                },
            })
        
        # Proposed actions
        actions_text = "\n".join([
            f"• `{a.get('action_type', 'unknown')}`: {a.get('description', 'No description')[:100]}"
            for a in request.actions[:5]
        ])
        if len(request.actions) > 5:
            actions_text += f"\n_...and {len(request.actions) - 5} more actions_"
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Proposed Actions ({len(request.actions)}):*\n{actions_text}",
            },
        })
        
        # Expiration warning
        if request.expires_at:
            expires_in = (request.expires_at - datetime.utcnow()).total_seconds() / 60
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⏰ Expires in {expires_in:.0f} minutes | Request ID: `{request.request_id}`",
                    },
                ],
            })
        
        # Action buttons
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "actions",
            "block_id": f"approval_{request.request_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "✅ Approve",
                        "emoji": True,
                    },
                    "style": "primary",
                    "action_id": "approve_action",
                    "value": request.request_id,
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "❌ Reject",
                        "emoji": True,
                    },
                    "style": "danger",
                    "action_id": "reject_action",
                    "value": request.request_id,
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📋 View Details",
                        "emoji": True,
                    },
                    "action_id": "view_details",
                    "value": request.request_id,
                    "url": f"http://localhost:8000/approvals/{request.request_id}",
                },
            ],
        })
        
        return blocks

    async def send_approval_result(
        self,
        request: Any,
        approved: bool,
        approver: str,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Send notification about approval result.
        
        Args:
            request: ApprovalRequest object
            approved: Whether it was approved
            approver: Who made the decision
            notes: Additional notes
            
        Returns:
            Slack API response
        """
        emoji = "✅" if approved else "❌"
        action = "Approved" if approved else "Rejected"
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} *{action}* by <@{approver}>",
                },
            },
        ]
        
        if notes:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📝 {notes}",
                    },
                ],
            })
        
        payload = {
            "channel": self._channel,
            "text": f"{emoji} Incident {request.incident_id} {action.lower()}",
            "blocks": blocks,
        }
        
        # Reply in thread if we have one
        if request.incident_id in self._message_threads:
            payload["thread_ts"] = self._message_threads[request.incident_id]
        
        return await self._send_message(payload)

    async def _send_message(self, payload: dict) -> dict[str, Any]:
        """
        Send message to Slack.
        
        Uses webhook URL if available, otherwise logs the message.
        """
        if not self._webhook_url:
            # Stub mode - just log
            self._logger.info(
                "slack_message_stub",
                channel=payload.get("channel"),
                text=payload.get("text"),
            )
            return {"ok": True, "ts": datetime.utcnow().timestamp()}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()
                return {"ok": True, "ts": datetime.utcnow().timestamp()}
        except Exception as e:
            self._logger.error(
                "slack_send_failed",
                error=str(e),
            )
            return {"ok": False, "error": str(e)}


# Singleton instance
_slack_notifier: SlackNotifier | None = None


def get_slack_notifier() -> SlackNotifier:
    """Get or create the Slack notifier singleton."""
    global _slack_notifier
    
    if _slack_notifier is None:
        _slack_notifier = SlackNotifier()
    
    return _slack_notifier
