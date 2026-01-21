"""
Policy Agent - AI Governance and security rule enforcement.

Responsibilities:
- Validate proposed actions against immutable security rules
- Calculate blast radius of proposed changes
- Create audit trail for all decisions
- Determine if human approval is needed
"""

import re
from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.config import settings
from src.models.router import TaskType
from src.workflows.states import (
    IncidentState,
    IncidentStatus,
    PolicyDecision,
    PolicyViolation,
    ProposedAction,
)

logger = structlog.get_logger(__name__)


class PolicyRule(BaseModel):
    """A policy rule from the rules file."""

    id: str
    name: str
    description: str | None = None
    pattern: str
    severity: str
    action: str
    message: str
    max_value: int | None = None


class BlastRadiusAnalysis(BaseModel):
    """Structured output for blast radius analysis."""

    score: float = Field(description="Blast radius score 0-1", ge=0, le=1)
    affected_pods_estimate: int = Field(description="Estimated number of affected pods")
    affected_services: list[str] = Field(description="List of potentially affected services")
    risk_factors: list[str] = Field(description="Identified risk factors")
    recommendation: str = Field(description="Recommendation based on analysis")


class PolicyAgent(BaseAgent):
    """
    Agent responsible for policy enforcement and governance.
    
    Validates all proposed actions against:
    - Immutable security rules (YAML config)
    - Blast radius limits
    - Time-based restrictions
    
    Creates audit trail for compliance.
    """

    def __init__(self) -> None:
        super().__init__()
        self._rules: list[PolicyRule] = []
        self._load_rules()

    @property
    def name(self) -> str:
        return "policy"

    @property
    def description(self) -> str:
        return "Enforces security policies and calculates blast radius"

    def _load_rules(self) -> None:
        """Load policy rules from YAML file."""
        rules_path = Path("data/policies/immutable_rules.yaml")
        
        if not rules_path.exists():
            self._logger.warning("policy_rules_not_found", path=str(rules_path))
            return

        try:
            with open(rules_path) as f:
                data = yaml.safe_load(f)
                
            for rule_data in data.get("rules", []):
                self._rules.append(PolicyRule(**rule_data))
                
            self._logger.info(
                "policy_rules_loaded",
                rule_count=len(self._rules),
            )
        except Exception as e:
            self._logger.error(
                "failed_to_load_policy_rules",
                error=str(e),
            )

    async def process(self, state: IncidentState) -> IncidentState:
        """
        Validate proposed actions against policies.
        
        Args:
            state: Current incident state with proposed actions
            
        Returns:
            Updated state with policy decision and violations
        """
        self._logger.info(
            "policy_check_started",
            incident_id=state.incident_id,
            action_count=len(state.proposed_actions),
        )

        # Step 1: Check each action against rules
        all_violations: list[PolicyViolation] = []
        
        for action in state.proposed_actions:
            violations = self._check_action_against_rules(action)
            all_violations.extend(violations)

        state.policy_violations = all_violations

        # Step 2: Calculate blast radius
        blast_radius = await self._calculate_blast_radius(state)
        state.blast_radius_score = blast_radius.score

        # Step 3: Make policy decision
        decision = self._make_decision(state, all_violations, blast_radius)
        state.policy_decision = decision

        # Step 4: Update status based on decision
        if decision == PolicyDecision.BLOCKED:
            state.status = IncidentStatus.FAILED
        elif decision == PolicyDecision.NEEDS_REVIEW:
            state.status = IncidentStatus.AWAITING_APPROVAL
        else:
            state.status = IncidentStatus.REMEDIATING

        # Step 5: Create audit entry
        await self._create_audit_entry(state, all_violations, blast_radius)

        self._logger.info(
            "policy_check_complete",
            incident_id=state.incident_id,
            decision=decision,
            violation_count=len(all_violations),
            blast_radius=blast_radius.score,
        )

        # Send A2A message based on decision
        if decision == PolicyDecision.NEEDS_REVIEW:
            self.create_message(
                to_agent="hitl",
                message_type="request",
                content={
                    "event": "approval_required",
                    "violations": [v.model_dump() for v in all_violations],
                    "blast_radius": blast_radius.model_dump(),
                },
                state=state,
            )
        else:
            self.create_message(
                to_agent="remediate",
                message_type="notification",
                content={
                    "event": "policy_decision",
                    "decision": decision,
                },
                state=state,
            )

        return state

    def _check_action_against_rules(
        self,
        action: ProposedAction,
    ) -> list[PolicyViolation]:
        """
        Check a single action against all policy rules.
        
        Args:
            action: The proposed action to check
            
        Returns:
            List of violations found
        """
        violations = []

        for rule in self._rules:
            try:
                # Check if action matches the rule pattern
                if re.search(rule.pattern, action.command, re.IGNORECASE):
                    # Check max_value constraint if applicable
                    if rule.max_value is not None:
                        match = re.search(rule.pattern, action.command, re.IGNORECASE)
                        if match and match.groups():
                            try:
                                value = int(match.group(1))
                                if value <= rule.max_value:
                                    continue  # Within limits, not a violation
                            except (ValueError, IndexError):
                                pass

                    violations.append(
                        PolicyViolation(
                            rule_id=rule.id,
                            rule_name=rule.name,
                            severity=rule.severity,
                            action=action.command,
                            blocked=rule.action == "BLOCK",
                            message=rule.message,
                        )
                    )

                    self._logger.warning(
                        "policy_violation_detected",
                        rule_id=rule.id,
                        rule_name=rule.name,
                        action=action.command[:100],
                        blocked=rule.action == "BLOCK",
                    )

            except re.error as e:
                self._logger.error(
                    "invalid_rule_pattern",
                    rule_id=rule.id,
                    pattern=rule.pattern,
                    error=str(e),
                )

        return violations

    async def _calculate_blast_radius(
        self,
        state: IncidentState,
    ) -> BlastRadiusAnalysis:
        """
        Calculate the blast radius of proposed actions.
        
        Uses LLM to analyze potential impact.
        
        Args:
            state: Current incident state
            
        Returns:
            BlastRadiusAnalysis with impact assessment
        """
        system_prompt = """You are an expert at analyzing the potential impact of Kubernetes operations.

Analyze the proposed actions and estimate:
1. Blast radius score (0-1, where 1 is maximum impact)
2. Number of pods that could be affected
3. Which services might be impacted
4. Risk factors to consider

Consider:
- Production namespaces have higher impact
- Operations affecting multiple pods are riskier
- Stateful applications need extra caution
- Peak traffic hours increase risk"""

        actions_text = "\n".join(
            f"- {a.action_type}: {a.command} (target: {a.target_resource})"
            for a in state.proposed_actions
        )

        user_prompt = f"""Analyze the blast radius of these proposed actions:

**Cluster:** {state.cluster}
**Namespace:** {state.namespace}
**Severity:** {state.severity}
**Affected Resources:** {', '.join(state.affected_resources)}

**Proposed Actions:**
{actions_text}

Provide your blast radius analysis."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            analysis, _ = await self.call_llm_structured(
                messages=messages,
                response_model=BlastRadiusAnalysis,
                task_type=TaskType.POLICY_CHECK,
                state=state,
            )
            return analysis

        except Exception as e:
            self._logger.error(
                "blast_radius_calculation_failed",
                error=str(e),
            )
            # Return conservative estimate on failure
            return BlastRadiusAnalysis(
                score=0.8,
                affected_pods_estimate=10,
                affected_services=state.affected_resources[:3],
                risk_factors=["Analysis failed - using conservative estimate"],
                recommendation="Manual review recommended due to analysis failure",
            )

    def _make_decision(
        self,
        state: IncidentState,
        violations: list[PolicyViolation],
        blast_radius: BlastRadiusAnalysis,
    ) -> PolicyDecision:
        """
        Make final policy decision based on all factors.
        
        Args:
            state: Current incident state
            violations: Found policy violations
            blast_radius: Blast radius analysis
            
        Returns:
            PolicyDecision
        """
        # Any blocked violation = BLOCKED
        if any(v.blocked for v in violations):
            return PolicyDecision.BLOCKED

        # High severity violations = NEEDS_REVIEW
        if any(v.severity in ["HIGH", "CRITICAL"] for v in violations):
            return PolicyDecision.NEEDS_REVIEW

        # High blast radius = NEEDS_REVIEW
        if blast_radius.score > 0.5:
            return PolicyDecision.NEEDS_REVIEW

        # P1/P2 incidents with any violations = NEEDS_REVIEW
        if state.severity in ["P1", "P2"] and violations:
            return PolicyDecision.NEEDS_REVIEW

        # P1/P2 incidents always need review for safety
        if state.severity in ["P1", "P2"]:
            return PolicyDecision.NEEDS_REVIEW

        # Low confidence diagnosis = NEEDS_REVIEW
        if state.confidence and state.confidence < 0.7:
            return PolicyDecision.NEEDS_REVIEW

        return PolicyDecision.APPROVED

    async def _create_audit_entry(
        self,
        state: IncidentState,
        violations: list[PolicyViolation],
        blast_radius: BlastRadiusAnalysis,
    ) -> None:
        """
        Create an audit log entry for this policy check.
        
        Args:
            state: Current incident state
            violations: Found violations
            blast_radius: Blast radius analysis
        """
        from datetime import datetime

        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "incident_id": state.incident_id,
            "agent": self.name,
            "severity": state.severity,
            "proposed_actions": [a.model_dump() for a in state.proposed_actions],
            "violations": [v.model_dump() for v in violations],
            "blast_radius_score": blast_radius.score,
            "decision": state.policy_decision,
            "root_cause": state.root_cause,
            "diagnosis_confidence": state.confidence,
        }

        # In production, this would write to a persistent audit log
        # For now, we just log it
        self._logger.info(
            "audit_entry_created",
            **audit_entry,
        )
