"""
Report Agent - Generates incident reports and stores in memory.

Responsibilities:
- Generate incident summary and RCA report
- Send notifications to Slack
- Store resolved incidents in episodic memory for future recall
- Calculate resolution metrics
"""

from datetime import datetime

from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.models.router import TaskType
from src.workflows.states import (
    IncidentState,
    IncidentStatus,
)


class IncidentReport(BaseModel):
    """Structured incident report."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary of the incident")
    root_cause_analysis: str = Field(description="Detailed root cause analysis")
    actions_taken: list[str] = Field(description="List of actions taken")
    outcome: str = Field(description="Final outcome description")
    lessons_learned: list[str] = Field(description="Key lessons from this incident")
    prevention_recommendations: list[str] = Field(
        description="Recommendations to prevent recurrence"
    )
    slack_message: str = Field(description="Formatted Slack notification message")


class ReportAgent(BaseAgent):
    """
    Agent responsible for reporting and memory storage.
    
    Generates:
    - Incident summary reports
    - RCA (Root Cause Analysis) documents
    - Slack notifications
    
    Also stores resolved incidents in episodic memory for future recall.
    """

    @property
    def name(self) -> str:
        return "report"

    @property
    def description(self) -> str:
        return "Generates reports and stores incidents in memory"

    async def process(self, state: IncidentState) -> IncidentState:
        """
        Generate final incident report and store in memory.
        
        Args:
            state: Current incident state with execution results
            
        Returns:
            Updated state with report and metrics
        """
        self._logger.info(
            "report_generation_started",
            incident_id=state.incident_id,
            status=state.status,
        )

        # Calculate resolution time
        state.time_to_resolve_minutes = self._calculate_resolution_time(state)

        # Generate the report
        report = await self._generate_report(state)

        # Update state with report
        state.resolution_summary = report.summary

        # Send Slack notification
        await self._send_slack_notification(state, report)

        # Store in episodic memory if resolved
        if state.status == IncidentStatus.RESOLVED:
            await self._store_in_memory(state, report)

        self._logger.info(
            "report_generation_complete",
            incident_id=state.incident_id,
            resolution_time_minutes=state.time_to_resolve_minutes,
            status=state.status,
        )

        return state

    def _calculate_resolution_time(self, state: IncidentState) -> float:
        """
        Calculate total resolution time in minutes.
        
        Args:
            state: Current incident state
            
        Returns:
            Resolution time in minutes
        """
        if state.created_at:
            delta = datetime.utcnow() - state.created_at
            return delta.total_seconds() / 60
        return 0.0

    async def _generate_report(self, state: IncidentState) -> IncidentReport:
        """
        Generate comprehensive incident report using LLM.
        
        Args:
            state: Current incident state
            
        Returns:
            IncidentReport with all sections
        """
        # Build context for report generation
        actions_text = "\n".join(
            f"- [{a.status.upper()}] {a.action.description}: {a.result or a.error or 'No output'}"
            for a in state.executed_actions
        )

        similar_incidents_text = ""
        if state.similar_past_incidents:
            similar_incidents_text = "\n".join(
                f"- {inc.incident_id}: {inc.root_cause} (fixed by: {inc.fix_applied})"
                for inc in state.similar_past_incidents
            )

        system_prompt = """You are an expert SRE writing an incident report.

Generate a comprehensive yet concise incident report that includes:
1. Executive summary (2-3 sentences)
2. Root cause analysis (technical details)
3. Actions taken and their outcomes
4. Lessons learned
5. Prevention recommendations
6. A Slack message (with appropriate emoji indicators)

Use professional language. Be specific and actionable.
For the Slack message, use emoji like:
- 🟢 for resolved
- 🔴 for failed
- ⚠️ for warnings
- 🔧 for actions taken
- 📊 for metrics"""

        user_prompt = f"""Generate an incident report for:

## INCIDENT DETAILS
- **ID:** {state.incident_id}
- **Title:** {state.title}
- **Severity:** {state.severity}
- **Status:** {state.status}
- **Cluster:** {state.cluster}
- **Namespace:** {state.namespace}

## DIAGNOSIS
- **Root Cause:** {state.root_cause}
- **Confidence:** {state.confidence:.0%}
- **Diagnosis:** {state.diagnosis}

## ANOMALIES DETECTED
{chr(10).join('- ' + a for a in state.anomalies)}

## ACTIONS TAKEN
{actions_text if actions_text else "No actions executed"}

## SIMILAR PAST INCIDENTS
{similar_incidents_text if similar_incidents_text else "No similar incidents found"}

## POLICY VIOLATIONS
{chr(10).join('- ' + v.message for v in state.policy_violations) if state.policy_violations else "None"}

## METRICS
- Resolution Time: {state.time_to_resolve_minutes:.1f} minutes
- Total LLM Cost: ${state.total_llm_cost_usd:.4f}
- Agents Involved: {', '.join(state.agents_involved)}

Generate the incident report."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        report, _ = await self.call_llm_structured(
            messages=messages,
            response_model=IncidentReport,
            task_type=TaskType.SUMMARIZATION,
            state=state,
        )

        return report

    async def _send_slack_notification(
        self,
        state: IncidentState,
        report: IncidentReport,
    ) -> None:
        """
        Send notification to Slack.
        
        TODO: Replace with actual Slack MCP integration.
        
        Args:
            state: Current incident state
            report: Generated report
        """
        self._logger.info(
            "sending_slack_notification",
            incident_id=state.incident_id,
            channel="#sre-incidents",
        )

        # Mock Slack send - will be replaced with MCP
        # In production, this would use the Slack API/webhook

        slack_payload = {
            "channel": "#sre-incidents",
            "text": report.slack_message,
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{'🟢' if state.status == IncidentStatus.RESOLVED else '🔴'} {state.title}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Incident ID:*\n{state.incident_id}"},
                        {"type": "mrkdwn", "text": f"*Severity:*\n{state.severity}"},
                        {"type": "mrkdwn", "text": f"*Status:*\n{state.status}"},
                        {
                            "type": "mrkdwn",
                            "text": f"*Resolution Time:*\n{state.time_to_resolve_minutes:.1f} min",
                        },
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Root Cause:*\n{state.root_cause}",
                    },
                },
            ],
        }

        self._logger.debug(
            "slack_payload_prepared",
            channel=slack_payload["channel"],
            block_count=len(slack_payload["blocks"]),
        )

        # Store thread_ts for follow-up messages
        state.slack_thread_ts = f"mock-thread-{state.incident_id}"

    async def _store_in_memory(
        self,
        state: IncidentState,
        report: IncidentReport,
    ) -> None:
        """
        Store resolved incident in episodic memory for future recall.
        
        TODO: Replace with actual Mem0 integration.
        
        Args:
            state: Current incident state
            report: Generated report
        """
        self._logger.info(
            "storing_in_memory",
            incident_id=state.incident_id,
        )

        # Build memory entry
        memory_entry = {
            "incident_id": state.incident_id,
            "cluster": state.cluster,
            "namespace": state.namespace,
            "symptoms": state.anomalies,
            "root_cause": state.root_cause,
            "fix_applied": [a.action.description for a in state.executed_actions if a.status == "success"],
            "resolution_time_minutes": state.time_to_resolve_minutes,
            "severity": state.severity,
            "timestamp": datetime.utcnow().isoformat(),
            "lessons_learned": report.lessons_learned,
        }

        # Mock memory storage - will be replaced with Mem0
        # In production, this would add to the episodic memory store

        memory_text = f"""
Incident in {state.cluster}/{state.namespace}: {', '.join(state.anomalies[:3])}
Root cause: {state.root_cause}
Fixed by: {', '.join(memory_entry['fix_applied'][:2]) if memory_entry['fix_applied'] else 'Manual intervention'}
Resolution time: {state.time_to_resolve_minutes:.0f} minutes
"""

        self._logger.debug(
            "memory_entry_created",
            incident_id=state.incident_id,
            memory_text=memory_text[:200],
        )

        # In production: await self.memory.add(memory_text, metadata=memory_entry)
