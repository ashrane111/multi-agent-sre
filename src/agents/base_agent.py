"""
Base Agent class providing common functionality for all agents.

All specialized agents inherit from this base class which provides:
- LLM access through the gateway
- Structured output parsing
- Cost tracking
- Logging with context
- A2A message handling
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel

from src.config import settings
from src.logging_config import set_agent_name
from src.models.llm_gateway import LLMGateway, LLMResponse, get_llm_gateway
from src.models.router import ModelRouter, TaskType, get_model_router
from src.models.token_counter import TokenCounter, get_token_counter
from src.reliability.circuit_breaker import CircuitBreaker, get_circuit_breaker
from src.reliability.retry_policy import RetryPolicy
from src.reliability.timeout_manager import TimeoutManager, get_timeout_manager
from src.workflows.states import AgentMessage, IncidentState, WorkflowState

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """
    Abstract base class for all SRE agents.
    
    Provides common functionality:
    - LLM access with fallback and retry
    - Structured output parsing
    - Cost tracking
    - Logging with correlation
    - A2A messaging
    
    Subclasses must implement:
    - name: Agent identifier
    - process(): Main processing logic
    """

    def __init__(self) -> None:
        """Initialize the base agent."""
        self._llm_gateway: LLMGateway = get_llm_gateway()
        self._model_router: ModelRouter = get_model_router()
        self._token_counter: TokenCounter = get_token_counter()
        self._timeout_manager: TimeoutManager = get_timeout_manager()
        self._circuit_breaker: CircuitBreaker = get_circuit_breaker(f"agent_{self.name}")
        self._retry_policy = RetryPolicy(max_attempts=settings.reliability.max_retries)
        self._logger = structlog.get_logger(f"agent.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name/identifier."""
        ...

    @property
    def description(self) -> str:
        """Agent description for logging/debugging."""
        return f"{self.name} agent"

    @abstractmethod
    async def process(self, state: IncidentState) -> IncidentState:
        """
        Main processing logic for the agent.
        
        Args:
            state: Current incident state
            
        Returns:
            Updated incident state
        """
        ...

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node entry point.
        
        Converts between dict (LangGraph) and IncidentState (internal).
        Handles logging, timing, and error handling.
        
        Args:
            state: Workflow state dict from LangGraph
            
        Returns:
            Updated workflow state dict
        """
        # Set logging context
        set_agent_name(self.name)

        # Convert to typed state
        incident_state = IncidentState.model_validate(state)
        incident_state.add_agent(self.name)

        start_time = datetime.utcnow()
        self._logger.info(
            "agent_started",
            incident_id=incident_state.incident_id,
            status=incident_state.status,
        )

        try:
            # Process with timeout
            async with self._timeout_manager.timeout(
                seconds=settings.reliability.agent_workflow_timeout,
                operation=f"agent_{self.name}",
            ):
                # Process with circuit breaker
                async with self._circuit_breaker:
                    updated_state = await self.process(incident_state)

            # Calculate duration
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._logger.info(
                "agent_completed",
                incident_id=updated_state.incident_id,
                status=updated_state.status,
                duration_ms=duration_ms,
                tokens_used=updated_state.total_llm_tokens,
                cost_usd=updated_state.total_llm_cost_usd,
            )

            return updated_state.model_dump(mode="json")

        except Exception as e:
            self._logger.error(
                "agent_failed",
                incident_id=incident_state.incident_id,
                error=str(e),
                exc_info=e,
            )
            # Add error to state and return
            incident_state.add_error(f"{self.name}: {str(e)}")
            return incident_state.model_dump(mode="json")

    async def call_llm(
        self,
        messages: list[dict[str, str]],
        task_type: TaskType,
        temperature: float = 0,
        max_tokens: int = 2048,
        state: IncidentState | None = None,
    ) -> LLMResponse:
        """
        Call LLM through the gateway with routing and tracking.
        
        Args:
            messages: Chat messages
            task_type: Type of task for model routing
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            state: Optional state for cost tracking
            
        Returns:
            LLM response
        """
        # Get routed model
        model = self._model_router.get_model_for_task(
            task_type=task_type,
            agent_name=self.name,
        )

        self._logger.debug(
            "llm_call_starting",
            model=model,
            task_type=task_type.value,
            message_count=len(messages),
        )

        # Make the call
        response = await self._llm_gateway.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Track usage
        self._model_router.record_usage(
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            agent_name=self.name,
            task_type=task_type,
        )

        # Update state cost if provided
        if state:
            state.update_cost(response.total_tokens, response.cost_usd)

        return response

    async def call_llm_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        task_type: TaskType,
        state: IncidentState | None = None,
    ) -> tuple[T, LLMResponse]:
        """
        Call LLM and parse response into a Pydantic model.
        
        Args:
            messages: Chat messages
            response_model: Pydantic model class for response
            task_type: Type of task for model routing
            state: Optional state for cost tracking
            
        Returns:
            Tuple of (parsed response, raw LLM response)
        """
        # Get routed model
        model = self._model_router.get_model_for_task(
            task_type=task_type,
            agent_name=self.name,
        )

        parsed, response = await self._llm_gateway.complete_with_structured_output(
            messages=messages,
            response_model=response_model,
            model=model,
        )

        # Track usage
        self._model_router.record_usage(
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            agent_name=self.name,
            task_type=task_type,
        )

        # Update state cost if provided
        if state:
            state.update_cost(response.total_tokens, response.cost_usd)

        return parsed, response

    def create_message(
        self,
        to_agent: str,
        message_type: str,
        content: dict[str, Any],
        state: IncidentState,
    ) -> AgentMessage:
        """
        Create an A2A message to another agent.
        
        Args:
            to_agent: Target agent name
            message_type: Type of message (request, response, notification)
            content: Message content
            state: Current state for correlation
            
        Returns:
            AgentMessage object
        """
        message = AgentMessage(
            from_agent=self.name,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            correlation_id=state.incident_id,
        )
        state.add_message(message)
        
        self._logger.debug(
            "a2a_message_created",
            to_agent=to_agent,
            message_type=message_type,
        )
        
        return message

    def get_messages_for_me(self, state: IncidentState) -> list[AgentMessage]:
        """
        Get all A2A messages addressed to this agent.
        
        Args:
            state: Current state containing messages
            
        Returns:
            List of messages for this agent
        """
        return [m for m in state.messages if m.to_agent == self.name]

    def build_system_prompt(self, base_prompt: str, state: IncidentState) -> str:
        """
        Build a system prompt with context injection.
        
        Args:
            base_prompt: Base system prompt
            state: Current state for context
            
        Returns:
            Enhanced system prompt
        """
        context = f"""
Current Incident Context:
- Incident ID: {state.incident_id}
- Severity: {state.severity}
- Cluster: {state.cluster}
- Namespace: {state.namespace}
- Status: {state.status}
"""
        return f"{base_prompt}\n\n{context}"

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return self._token_counter.count(text)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        return self._token_counter.truncate_to_limit(text, max_tokens)
