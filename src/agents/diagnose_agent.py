"""
Diagnose Agent - Root cause analysis using RAG and episodic memory.

Responsibilities:
- Query runbook documentation (RAG)
- Recall similar past incidents (Mem0)
- Perform root cause analysis
- Recommend remediation actions
- Provide confidence scores
"""

from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.models.router import TaskType
from src.workflows.states import (
    IncidentState,
    IncidentStatus,
    ProposedAction,
    RunbookMatch,
    SimilarIncident,
)


class DiagnosisResult(BaseModel):
    """Structured output for diagnosis."""

    root_cause: str = Field(description="Most likely root cause of the incident")
    confidence: float = Field(description="Confidence in diagnosis 0-1", ge=0, le=1)
    reasoning: str = Field(description="Step-by-step reasoning for the diagnosis")
    memory_informed: bool = Field(
        description="Whether similar past incidents influenced the diagnosis"
    )
    recommended_actions: list[dict] = Field(
        description="List of recommended remediation actions"
    )


class RecommendedAction(BaseModel):
    """Structured action recommendation."""

    action_type: str = Field(description="Type: scale, restart, rollback, config_change, custom")
    command: str = Field(description="The kubectl or API command to execute")
    target_resource: str = Field(description="Target resource (e.g., deployment/api-server)")
    description: str = Field(description="Human-readable description of what this does")
    risk_level: str = Field(description="Risk level: low, medium, high, critical")
    estimated_impact: str = Field(description="Expected impact of this action")
    rollback_command: str | None = Field(
        default=None, description="Command to rollback this action if needed"
    )


class DiagnoseAgent(BaseAgent):
    """
    Agent responsible for root cause analysis.
    
    Uses:
    - RAG to query runbook documentation
    - Episodic memory (Mem0) to recall similar past incidents
    - LLM reasoning for diagnosis
    """

    @property
    def name(self) -> str:
        return "diagnose"

    @property
    def description(self) -> str:
        return "Diagnoses incidents using RAG and episodic memory"

    async def process(self, state: IncidentState) -> IncidentState:
        """
        Perform root cause analysis on the incident.
        
        Args:
            state: Current incident state with metrics and anomalies
            
        Returns:
            Updated state with diagnosis and recommended actions
        """
        self._logger.info(
            "starting_diagnosis",
            incident_id=state.incident_id,
            severity=state.severity,
            anomaly_count=len(state.anomalies),
        )

        # Step 1: Check episodic memory for similar incidents
        similar_incidents = await self._recall_similar_incidents(state)
        state.similar_past_incidents = similar_incidents

        # Step 2: Query runbook documentation via RAG
        runbook_matches = await self._query_runbooks(state)
        state.runbook_matches = runbook_matches

        # Step 3: Perform diagnosis using LLM
        diagnosis = await self._diagnose(state, similar_incidents, runbook_matches)

        # Update state with diagnosis
        state.root_cause = diagnosis.root_cause
        state.confidence = diagnosis.confidence
        state.diagnosis = diagnosis.reasoning
        state.memory_informed = diagnosis.memory_informed

        # Step 4: Convert recommended actions to ProposedActions
        state.proposed_actions = await self._create_proposed_actions(
            state, diagnosis.recommended_actions
        )

        state.status = IncidentStatus.POLICY_CHECK

        self._logger.info(
            "diagnosis_complete",
            incident_id=state.incident_id,
            root_cause=state.root_cause[:100],
            confidence=state.confidence,
            action_count=len(state.proposed_actions),
            memory_informed=state.memory_informed,
        )

        # Send A2A notification
        self.create_message(
            to_agent="policy",
            message_type="request",
            content={
                "event": "diagnosis_complete",
                "root_cause": state.root_cause,
                "proposed_actions": [a.model_dump() for a in state.proposed_actions],
            },
            state=state,
        )

        return state

    async def _recall_similar_incidents(
        self,
        state: IncidentState,
    ) -> list[SimilarIncident]:
        """
        Recall similar incidents from episodic memory using Mem0.
        
        Args:
            state: Current incident state
            
        Returns:
            List of similar past incidents
        """
        self._logger.debug(
            "querying_episodic_memory",
            cluster=state.cluster,
            anomalies=state.anomalies,
        )

        try:
            # Use real Mem0-based incident memory
            from src.memory import get_incident_memory
            
            incident_memory = get_incident_memory()
            similar_incidents = incident_memory.recall_similar_incidents(
                state=state,
                top_k=5,
                threshold=0.3,
            )

            self._logger.debug(
                "memory_recall_complete",
                incident_count=len(similar_incidents),
                source="mem0",
            )

            return similar_incidents

        except Exception as e:
            self._logger.warning(
                "memory_recall_failed_using_fallback",
                error=str(e),
            )
            
            # Fallback to mock data if memory system fails
            return self._get_fallback_similar_incidents(state)

    def _get_fallback_similar_incidents(
        self,
        state: IncidentState,
    ) -> list[SimilarIncident]:
        """
        Fallback mock similar incidents when memory system is unavailable.
        
        Args:
            state: Current incident state
            
        Returns:
            List of mock similar incidents
        """
        from datetime import datetime, timedelta
        
        mock_incidents = []
        
        if any("cpu" in a.lower() for a in state.anomalies):
            mock_incidents.append(
                SimilarIncident(
                    incident_id="INC-2024-1234",
                    similarity_score=0.87,
                    cluster=state.cluster,
                    symptoms=["High CPU usage", "Pod restarts"],
                    root_cause="Memory leak in application causing GC thrashing",
                    fix_applied="Increased memory limits and restarted pods",
                    resolution_time_minutes=15.5,
                    occurred_at=datetime.utcnow() - timedelta(days=7),
                )
            )

        if any("restart" in a.lower() or "crashloop" in a.lower() for a in state.anomalies):
            mock_incidents.append(
                SimilarIncident(
                    incident_id="INC-2024-1189",
                    similarity_score=0.82,
                    cluster=state.cluster,
                    symptoms=["CrashLoopBackOff", "OOMKilled"],
                    root_cause="Insufficient memory limits for new feature",
                    fix_applied="Rolled back to previous deployment",
                    resolution_time_minutes=8.0,
                    occurred_at=datetime.utcnow() - timedelta(days=14),
                )
            )

        return mock_incidents

    async def _query_runbooks(self, state: IncidentState) -> list[RunbookMatch]:
        """
        Query runbook documentation using RAG pipeline.
        
        Args:
            state: Current incident state
            
        Returns:
            List of relevant runbook sections
        """
        self._logger.debug(
            "querying_runbooks",
            anomalies=state.anomalies,
        )

        try:
            # Use real RAG pipeline
            from src.rag import get_rag_pipeline
            
            rag_pipeline = get_rag_pipeline()
            
            # Build query from incident context
            query = self._build_rag_query(state)
            
            # Retrieve relevant documents
            retrieval_result = rag_pipeline.retrieve(
                query=query,
                top_k=5,
                rerank=True,
                rerank_top_k=3,
            )

            # Convert to RunbookMatch objects
            matches = []
            for doc in retrieval_result.documents:
                metadata = doc.get("metadata", {})
                matches.append(
                    RunbookMatch(
                        runbook_name=metadata.get("source", "unknown"),
                        section=metadata.get("section", ""),
                        content=doc.get("content", ""),
                        relevance_score=doc.get("rerank_score", doc.get("fused_score", 0.0)),
                    )
                )

            self._logger.debug(
                "runbook_query_complete",
                match_count=len(matches),
                source="rag_pipeline",
            )

            return matches

        except Exception as e:
            self._logger.warning(
                "rag_query_failed_using_fallback",
                error=str(e),
            )
            
            # Fallback to mock data if RAG fails
            return self._get_fallback_runbook_matches(state)

    def _build_rag_query(self, state: IncidentState) -> str:
        """
        Build a RAG query from incident state.
        
        Args:
            state: Current incident state
            
        Returns:
            Query string for RAG retrieval
        """
        parts = []
        
        # Include anomalies as primary search terms
        if state.anomalies:
            parts.append(" ".join(state.anomalies))
        
        # Include description if available
        if state.description:
            parts.append(state.description)
        
        # Include title
        if state.title:
            parts.append(state.title)
        
        return " ".join(parts)

    def _get_fallback_runbook_matches(
        self,
        state: IncidentState,
    ) -> list[RunbookMatch]:
        """
        Fallback mock runbook matches when RAG is unavailable.
        
        Args:
            state: Current incident state
            
        Returns:
            List of mock runbook matches
        """
        mock_matches = []

        if any("cpu" in a.lower() for a in state.anomalies):
            mock_matches.append(
                RunbookMatch(
                    runbook_name="high_cpu.md",
                    section="Remediation Actions",
                    content="""
## For Traffic Spikes
```bash
kubectl scale deployment/<deployment-name> --replicas=<count> -n <namespace>
```

## For Resource Limit Issues
```bash
kubectl set resources deployment/<deployment-name> --limits=cpu=500m,memory=512Mi -n <namespace>
```

## For Application Issues
```bash
kubectl rollout undo deployment/<deployment-name> -n <namespace>
```
""",
                    relevance_score=0.92,
                )
            )

        if any("restart" in a.lower() or "crash" in a.lower() for a in state.anomalies):
            mock_matches.append(
                RunbookMatch(
                    runbook_name="pod_crashloop.md",
                    section="Common Root Causes",
                    content="""
Common causes of CrashLoopBackOff:
1. Application error - Code bug causing crash on startup
2. Missing dependencies - Required services unavailable
3. Configuration error - Invalid environment variables
4. Resource exhaustion - OOMKilled due to memory limits
5. Failed health checks - Liveness probe failing

For OOMKilled:
```bash
kubectl set resources deployment/<deployment-name> --limits=memory=1Gi -n <namespace>
```

For Code Bugs:
```bash
kubectl rollout undo deployment/<deployment-name> -n <namespace>
```
""",
                    relevance_score=0.89,
                )
            )

        return mock_matches

    async def _diagnose(
        self,
        state: IncidentState,
        similar_incidents: list[SimilarIncident],
        runbook_matches: list[RunbookMatch],
    ) -> DiagnosisResult:
        """
        Perform diagnosis using LLM with all gathered context.
        
        Args:
            state: Current incident state
            similar_incidents: Similar past incidents from memory
            runbook_matches: Relevant runbook sections
            
        Returns:
            DiagnosisResult with root cause and recommendations
        """
        # Build memory context
        memory_context = ""
        if similar_incidents:
            memory_context = "\n## SIMILAR PAST INCIDENTS I REMEMBER:\n"
            for inc in similar_incidents:
                memory_context += f"""
**Incident {inc.incident_id}** (Similarity: {inc.similarity_score:.0%})
- Symptoms: {', '.join(inc.symptoms)}
- Root Cause: {inc.root_cause}
- Fix Applied: {inc.fix_applied}
- Resolution Time: {inc.resolution_time_minutes:.0f} minutes
"""

        # Build runbook context
        runbook_context = ""
        if runbook_matches:
            runbook_context = "\n## RELEVANT RUNBOOK DOCUMENTATION:\n"
            for match in runbook_matches:
                runbook_context += f"""
### From {match.runbook_name} - {match.section} (Relevance: {match.relevance_score:.0%})
{match.content}
"""

        system_prompt = """You are an expert SRE diagnosing an infrastructure incident.

Your task is to:
1. Analyze the incident data, metrics, and anomalies
2. Consider similar past incidents if available (use phrases like "I remember a similar incident where...")
3. Reference runbook documentation for remediation steps
4. Determine the most likely root cause
5. Recommend specific remediation actions

Be precise, technical, and actionable. Provide step-by-step reasoning."""

        user_prompt = f"""Diagnose this infrastructure incident:

## CURRENT INCIDENT
- **ID:** {state.incident_id}
- **Severity:** {state.severity}
- **Cluster:** {state.cluster}
- **Namespace:** {state.namespace}
- **Title:** {state.title}
- **Description:** {state.description}

## DETECTED ANOMALIES
{chr(10).join('- ' + a for a in state.anomalies)}

## CURRENT METRICS
{self._format_metrics(state.metrics)}

## AFFECTED RESOURCES
{chr(10).join('- ' + r for r in state.affected_resources)}

{memory_context if memory_context else "No similar past incidents found in memory."}

{runbook_context if runbook_context else "No matching runbook documentation found."}

Based on all this information, provide your diagnosis and recommended actions.
For each action, specify:
- action_type: scale, restart, rollback, config_change, or custom
- command: The exact kubectl command
- target_resource: What resource it affects
- description: What it does
- risk_level: low, medium, high, or critical
- estimated_impact: Expected result
- rollback_command: How to undo (if applicable)"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        diagnosis, _ = await self.call_llm_structured(
            messages=messages,
            response_model=DiagnosisResult,
            task_type=TaskType.DIAGNOSIS,
            state=state,
        )

        return diagnosis

    async def _create_proposed_actions(
        self,
        state: IncidentState,
        recommended_actions: list[dict],
    ) -> list[ProposedAction]:
        """
        Convert LLM recommended actions to ProposedAction objects.
        
        Args:
            state: Current incident state
            recommended_actions: Raw action recommendations from LLM
            
        Returns:
            List of validated ProposedAction objects
        """
        proposed = []
        
        for i, action in enumerate(recommended_actions):
            try:
                proposed.append(
                    ProposedAction(
                        action_id=f"{state.incident_id}-action-{i+1}",
                        action_type=action.get("action_type", "custom"),
                        command=action.get("command", ""),
                        target_resource=action.get("target_resource", "unknown"),
                        namespace=state.namespace,
                        description=action.get("description", ""),
                        risk_level=action.get("risk_level", "medium"),
                        estimated_impact=action.get("estimated_impact", ""),
                        rollback_command=action.get("rollback_command"),
                    )
                )
            except Exception as e:
                self._logger.warning(
                    "failed_to_parse_action",
                    action=action,
                    error=str(e),
                )

        return proposed

    def _format_metrics(self, metrics: dict) -> str:
        """Format metrics dict for prompt."""
        if not metrics:
            return "No metrics available"
        
        lines = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"- {key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"- {key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)
