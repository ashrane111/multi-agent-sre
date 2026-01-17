"""
Circuit Breaker pattern implementation.

Prevents cascade failures by:
- Tracking failure rates per service
- Opening circuit when failures exceed threshold
- Automatically attempting recovery after timeout
- Supporting half-open state for gradual recovery
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, state: CircuitState, retry_after: float | None = None):
        self.name = name
        self.state = state
        self.retry_after = retry_after
        message = f"Circuit breaker '{name}' is {state.value}"
        if retry_after:
            message += f", retry after {retry_after:.1f}s"
        super().__init__(message)


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for protecting against cascade failures.
    
    Usage:
        breaker = CircuitBreaker(name="llm_provider")
        
        async def call_llm():
            async with breaker:
                return await llm.complete(...)
    
    States:
        - CLOSED: Normal operation, tracking failures
        - OPEN: Too many failures, rejecting all requests
        - HALF_OPEN: Testing recovery with single request
    """

    name: str
    failure_threshold: int = field(
        default_factory=lambda: settings.reliability.circuit_breaker_failure_threshold
    )
    recovery_timeout: int = field(
        default_factory=lambda: settings.reliability.circuit_breaker_recovery_timeout
    )
    half_open_max_calls: int = 1

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: datetime | None = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        elapsed = datetime.now() - self._last_failure_time
        return elapsed >= timedelta(seconds=self.recovery_timeout)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            old_state=old_state.value,
            new_state=new_state.value,
            failure_count=self._failure_count,
        )

    async def _check_state(self) -> None:
        """Check and potentially update circuit state."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._transition_to(CircuitState.CLOSED)
                self._failure_count = 0
                self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        logger.warning(
            "circuit_breaker_failure",
            name=self.name,
            failure_count=self._failure_count,
            threshold=self.failure_threshold,
            error=str(error) if error else None,
        )

        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery attempt, reopen
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        await self._check_state()

        if self._state == CircuitState.OPEN:
            retry_after = None
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                retry_after = max(0, self.recovery_timeout - elapsed)
            raise CircuitBreakerError(self.name, self._state, retry_after)

        if self._state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerError(self.name, self._state)
                self._half_open_calls += 1

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async context manager exit."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        logger.info("circuit_breaker_reset", name=self.name)

    def get_status(self) -> dict[str, Any]:
        """Get current status of the circuit breaker."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure": (
                self._last_failure_time.isoformat() if self._last_failure_time else None
            ),
            "time_until_recovery": (
                max(
                    0,
                    self.recovery_timeout
                    - (datetime.now() - self._last_failure_time).total_seconds(),
                )
                if self._last_failure_time and self._state == CircuitState.OPEN
                else None
            ),
        }


def with_circuit_breaker(breaker: CircuitBreaker) -> Callable:
    """
    Decorator to wrap a function with circuit breaker protection.
    
    Usage:
        breaker = CircuitBreaker(name="my_service")
        
        @with_circuit_breaker(breaker)
        async def call_service():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with breaker:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# Registry of circuit breakers for monitoring
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Unique name for the circuit breaker
        **kwargs: Arguments passed to CircuitBreaker constructor
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, dict[str, Any]]:
    """Get status of all registered circuit breakers."""
    return {name: breaker.get_status() for name, breaker in _circuit_breakers.items()}
