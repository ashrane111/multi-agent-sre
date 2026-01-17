"""
Timeout management for async operations.

Provides:
- Configurable timeouts per operation type
- Graceful timeout handling
- Deadline propagation for nested calls
"""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, AsyncIterator, Callable, TypeVar

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Context variable for deadline propagation
_deadline: ContextVar[datetime | None] = ContextVar("deadline", default=None)


class TimeoutError(asyncio.TimeoutError):
    """Custom timeout error with additional context."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout: float | None = None,
        operation: str | None = None,
    ):
        self.timeout = timeout
        self.operation = operation
        super().__init__(message)


@dataclass
class TimeoutManager:
    """
    Manages timeouts for different operation types.
    
    Usage:
        manager = TimeoutManager()
        
        async with manager.timeout(30, operation="llm_call"):
            result = await llm.complete(...)
    """

    # Default timeouts by operation type
    default_timeouts: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.default_timeouts is None:
            self.default_timeouts = {
                "llm_call": settings.reliability.llm_timeout,
                "workflow": settings.reliability.agent_workflow_timeout,
                "mcp_call": 30.0,
                "database": 10.0,
                "http_request": 30.0,
            }

    def get_timeout(self, operation: str) -> float:
        """Get timeout for an operation type."""
        return self.default_timeouts.get(operation, 30.0)

    def get_remaining_time(self) -> float | None:
        """
        Get remaining time until current deadline.
        
        Returns:
            Remaining seconds or None if no deadline set
        """
        deadline = _deadline.get()
        if deadline is None:
            return None
        remaining = (deadline - datetime.now()).total_seconds()
        return max(0, remaining)

    @asynccontextmanager
    async def timeout(
        self,
        seconds: float | None = None,
        operation: str | None = None,
    ) -> AsyncIterator[None]:
        """
        Context manager for timeout-protected operations.
        
        Args:
            seconds: Timeout in seconds (or use default for operation)
            operation: Operation type for logging and default timeout
            
        Yields:
            None
            
        Raises:
            TimeoutError: If operation exceeds timeout
        """
        if seconds is None and operation:
            seconds = self.get_timeout(operation)
        elif seconds is None:
            seconds = 30.0

        # Check if we have an existing deadline that's shorter
        existing_deadline = _deadline.get()
        new_deadline = datetime.now() + timedelta(seconds=seconds)

        if existing_deadline and existing_deadline < new_deadline:
            # Use the existing stricter deadline
            remaining = (existing_deadline - datetime.now()).total_seconds()
            if remaining <= 0:
                raise TimeoutError(
                    f"Deadline already passed for {operation}",
                    timeout=seconds,
                    operation=operation,
                )
            seconds = remaining
            new_deadline = existing_deadline

        # Set new deadline in context
        token = _deadline.set(new_deadline)

        try:
            logger.debug(
                "timeout_started",
                operation=operation,
                timeout=seconds,
                deadline=new_deadline.isoformat(),
            )

            async with asyncio.timeout(seconds):
                yield

        except asyncio.TimeoutError as e:
            logger.error(
                "operation_timeout",
                operation=operation,
                timeout=seconds,
            )
            raise TimeoutError(
                f"Operation '{operation}' timed out after {seconds}s",
                timeout=seconds,
                operation=operation,
            ) from e

        finally:
            # Reset deadline
            _deadline.reset(token)

    async def run_with_timeout(
        self,
        coro: Any,
        seconds: float | None = None,
        operation: str | None = None,
    ) -> Any:
        """
        Run a coroutine with timeout.
        
        Args:
            coro: Coroutine to run
            seconds: Timeout in seconds
            operation: Operation type for logging
            
        Returns:
            Result of the coroutine
        """
        async with self.timeout(seconds, operation):
            return await coro


def with_timeout(
    seconds: float | None = None,
    operation: str | None = None,
) -> Callable:
    """
    Decorator to add timeout to an async function.
    
    Usage:
        @with_timeout(30, operation="llm_call")
        async def call_llm():
            ...
    
    Args:
        seconds: Timeout in seconds
        operation: Operation type for logging and default timeout
        
    Returns:
        Decorated function
    """
    manager = TimeoutManager()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            op = operation or func.__name__
            async with manager.timeout(seconds, op):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def get_remaining_deadline() -> float | None:
    """
    Get remaining time until the current deadline.
    
    This can be used by nested functions to respect the outer deadline.
    
    Returns:
        Remaining seconds or None if no deadline
    """
    deadline = _deadline.get()
    if deadline is None:
        return None
    remaining = (deadline - datetime.now()).total_seconds()
    return max(0, remaining)


# Singleton instance
_timeout_manager: TimeoutManager | None = None


def get_timeout_manager() -> TimeoutManager:
    """Get the singleton timeout manager instance."""
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = TimeoutManager()
    return _timeout_manager
