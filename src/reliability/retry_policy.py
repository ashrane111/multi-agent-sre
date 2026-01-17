"""
Retry policy with exponential backoff.

Provides configurable retry logic for:
- LLM API calls
- External service calls
- Database operations
"""

import asyncio
import random
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Sequence, TypeVar

import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# Default retryable exceptions
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    TimeoutError,
    ConnectionError,
    asyncio.TimeoutError,
)


@dataclass
class RetryPolicy:
    """
    Configurable retry policy with exponential backoff and jitter.
    
    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential calculation
        jitter: Add randomness to prevent thundering herd
        retryable_exceptions: Tuple of exception types to retry on
    """

    max_attempts: int = field(
        default_factory=lambda: settings.reliability.max_retries
    )
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = field(
        default_factory=lambda: settings.reliability.retry_backoff_base
    )
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.
        
        Args:
            attempt: Current attempt number (1-indexed)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ (attempt - 1))
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        # Add jitter (±25%)
        if self.jitter:
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            RetryError: If all attempts fail
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.debug(
                    "retry_attempt",
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    function=func.__name__,
                )
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.info(
                        "retry_succeeded",
                        attempt=attempt,
                        function=func.__name__,
                    )
                return result

            except self.retryable_exceptions as e:
                last_exception = e
                if attempt < self.max_attempts:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        "retry_after_failure",
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                        delay=delay,
                        error=str(e),
                        function=func.__name__,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "retry_exhausted",
                        attempts=attempt,
                        error=str(e),
                        function=func.__name__,
                    )

            except Exception as e:
                # Non-retryable exception, raise immediately
                logger.error(
                    "non_retryable_error",
                    error=str(e),
                    function=func.__name__,
                )
                raise

        # All attempts failed
        raise last_exception or RuntimeError("Retry failed with no exception captured")


def with_retry(
    max_attempts: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable:
    """
    Decorator to add retry logic to an async function.
    
    Usage:
        @with_retry(max_attempts=3)
        async def call_api():
            ...
    
    Args:
        max_attempts: Maximum retry attempts (default from settings)
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        retryable_exceptions: Exceptions to retry on
        
    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        policy = RetryPolicy(
            max_attempts=max_attempts or settings.reliability.max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            retryable_exceptions=retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS,
        )

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await policy.execute(func, *args, **kwargs)

        return wrapper

    return decorator


def create_tenacity_retry(
    max_attempts: int | None = None,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    retryable_exceptions: Sequence[type[Exception]] | None = None,
) -> AsyncRetrying:
    """
    Create a tenacity AsyncRetrying instance for more complex retry scenarios.
    
    Usage:
        async for attempt in create_tenacity_retry():
            with attempt:
                result = await call_api()
    
    Args:
        max_attempts: Maximum attempts
        min_wait: Minimum wait between retries
        max_wait: Maximum wait between retries
        retryable_exceptions: Exceptions to retry on
        
    Returns:
        AsyncRetrying instance
    """
    exceptions = tuple(retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS)

    return AsyncRetrying(
        stop=stop_after_attempt(max_attempts or settings.reliability.max_retries),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )
