"""
Remediate Agent - Executes approved remediation actions.

Responsibilities:
- Execute approved actions against Kubernetes (via MCP)
- Track execution status for each action
- Perform rollback on failure
- Handle human approval flow
"""

from datetime import datetime

import structlog

from src.agents.base_agent import BaseAgent
from src.workflows.states import (
    ExecutedAction,
    IncidentState,
    IncidentStatus,
    PolicyDecision,
    ProposedAction,
)

logger = structlog.get_logger(__name__)


class RemediateAgent(BaseAgent):
    """
    Agent responsible for executing remediation actions.
    
    Only executes actions that have been:
    - Approved by policy agent, OR
    - Approved by human (for NEEDS_REVIEW decisions)
    
    Handles rollback if any action fails.
    """

    @property
    def name(self) -> str:
        return "remediate"

    @property
    def description(self) -> str:
        return "Executes approved remediation actions"

    async def process(self, state: IncidentState) -> IncidentState:
        """
        Execute approved remediation actions.
        
        Args:
            state: Current incident state with approved actions
            
        Returns:
            Updated state with execution results
        """
        self._logger.info(
            "remediation_started",
            incident_id=state.incident_id,
            policy_decision=state.policy_decision,
            human_approved=state.human_approved,
            action_count=len(state.proposed_actions),
        )

        # Check if we should execute
        if not self._should_execute(state):
            self._logger.warning(
                "remediation_skipped",
                incident_id=state.incident_id,
                reason="Not approved for execution",
            )
            return state

        # Execute each action
        executed_actions: list[ExecutedAction] = []
        all_successful = True

        for action in state.proposed_actions:
            # Check if this specific action was blocked
            if self._is_action_blocked(state, action):
                self._logger.info(
                    "action_skipped_blocked",
                    action_id=action.action_id,
                )
                executed_actions.append(
                    ExecutedAction(
                        action=action,
                        status="skipped",
                        result="Blocked by policy",
                    )
                )
                continue

            # Execute the action
            result = await self._execute_action(state, action)
            executed_actions.append(result)

            if result.status == "failed":
                all_successful = False
                self._logger.error(
                    "action_execution_failed",
                    action_id=action.action_id,
                    error=result.error,
                )
                
                # Attempt rollback of previously successful actions
                if executed_actions:
                    await self._rollback(state, executed_actions)
                break

        state.executed_actions = executed_actions

        # Update status based on results
        if all_successful:
            state.status = IncidentStatus.RESOLVED
            self._logger.info(
                "remediation_successful",
                incident_id=state.incident_id,
                actions_executed=len([a for a in executed_actions if a.status == "success"]),
            )
        else:
            state.status = IncidentStatus.FAILED
            self._logger.error(
                "remediation_failed",
                incident_id=state.incident_id,
                rollback_performed=state.rollback_performed,
            )

        # Send A2A notification to report agent
        self.create_message(
            to_agent="report",
            message_type="notification",
            content={
                "event": "remediation_complete",
                "status": state.status,
                "executed_actions": [a.model_dump() for a in executed_actions],
            },
            state=state,
        )

        return state

    def _should_execute(self, state: IncidentState) -> bool:
        """
        Determine if remediation should proceed.
        
        Args:
            state: Current incident state
            
        Returns:
            True if execution should proceed
        """
        # Blocked by policy - don't execute
        if state.policy_decision == PolicyDecision.BLOCKED:
            return False

        # Auto-approved - execute
        if state.policy_decision == PolicyDecision.APPROVED:
            return True

        # Needs review - check human approval
        if state.policy_decision == PolicyDecision.NEEDS_REVIEW:
            return state.human_approved is True

        return False

    def _is_action_blocked(
        self,
        state: IncidentState,
        action: ProposedAction,
    ) -> bool:
        """
        Check if a specific action is blocked.
        
        Args:
            state: Current incident state
            action: The action to check
            
        Returns:
            True if this action is blocked
        """
        for violation in state.policy_violations:
            if violation.blocked and violation.action == action.command:
                return True
        return False

    async def _execute_action(
        self,
        state: IncidentState,
        action: ProposedAction,
    ) -> ExecutedAction:
        """
        Execute a single remediation action.
        
        TODO: Replace with actual Kubernetes MCP call in Phase 3.
        
        Args:
            state: Current incident state
            action: The action to execute
            
        Returns:
            ExecutedAction with result
        """
        start_time = datetime.utcnow()

        self._logger.info(
            "executing_action",
            action_id=action.action_id,
            action_type=action.action_type,
            command=action.command[:100],
            target=action.target_resource,
        )

        try:
            # Mock execution - will be replaced with K8s MCP
            # Simulates what we'd get from kubectl
            result = await self._mock_k8s_execute(action)

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return ExecutedAction(
                action=action,
                status="success" if result["success"] else "failed",
                result=result.get("output"),
                error=result.get("error"),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ExecutedAction(
                action=action,
                status="failed",
                error=str(e),
                execution_time_ms=execution_time,
            )

    async def _mock_k8s_execute(self, action: ProposedAction) -> dict:
        """
        Mock Kubernetes command execution.
        
        TODO: Replace with actual K8s MCP integration.
        
        Args:
            action: The action to execute
            
        Returns:
            Dict with success status and output/error
        """
        # Simulate different outcomes based on action type
        import asyncio
        import os
        import random

        # Simulate some execution time
        await asyncio.sleep(0.5)
        
        # Check if we should disable simulated failures (for clean demos)
        disable_failures = os.environ.get("SRE_DISABLE_SIMULATED_FAILURES", "false").lower() == "true"

        # For demo purposes, most actions succeed
        # In production, this would actually execute kubectl commands
        
        if action.action_type == "scale":
            return {
                "success": True,
                "output": f"deployment.apps/{action.target_resource} scaled",
            }

        elif action.action_type == "restart":
            return {
                "success": True,
                "output": f"pod/{action.target_resource} deleted (will be recreated)",
            }

        elif action.action_type == "rollback":
            return {
                "success": True,
                "output": f"deployment.apps/{action.target_resource} rolled back",
            }

        elif action.action_type == "config_change":
            return {
                "success": True,
                "output": f"deployment.apps/{action.target_resource} resource limits updated",
            }

        else:
            # Custom action - simulate with 90% success rate (unless disabled)
            if disable_failures or random.random() > 0.1:
                return {
                    "success": True,
                    "output": f"Command executed successfully on {action.target_resource}",
                }
            else:
                return {
                    "success": False,
                    "error": "Command execution failed: simulated error",
                }

    async def _rollback(
        self,
        state: IncidentState,
        executed_actions: list[ExecutedAction],
    ) -> None:
        """
        Rollback previously successful actions.
        
        Args:
            state: Current incident state
            executed_actions: List of executed actions to potentially rollback
        """
        self._logger.warning(
            "starting_rollback",
            incident_id=state.incident_id,
            actions_to_rollback=len([a for a in executed_actions if a.status == "success"]),
        )

        for executed in reversed(executed_actions):
            if executed.status != "success":
                continue

            if not executed.action.rollback_command:
                self._logger.warning(
                    "no_rollback_command",
                    action_id=executed.action.action_id,
                )
                continue

            self._logger.info(
                "rolling_back_action",
                action_id=executed.action.action_id,
                rollback_command=executed.action.rollback_command[:100],
            )

            try:
                # Create a rollback action
                rollback_action = ProposedAction(
                    action_id=f"{executed.action.action_id}-rollback",
                    action_type="custom",
                    command=executed.action.rollback_command,
                    target_resource=executed.action.target_resource,
                    namespace=executed.action.namespace,
                    description=f"Rollback of {executed.action.description}",
                    risk_level=executed.action.risk_level,
                    estimated_impact="Reverting previous change",
                )

                result = await self._mock_k8s_execute(rollback_action)

                if result["success"]:
                    self._logger.info(
                        "rollback_successful",
                        action_id=executed.action.action_id,
                    )
                else:
                    self._logger.error(
                        "rollback_failed",
                        action_id=executed.action.action_id,
                        error=result.get("error"),
                    )

            except Exception as e:
                self._logger.error(
                    "rollback_exception",
                    action_id=executed.action.action_id,
                    error=str(e),
                )

        state.rollback_performed = True
