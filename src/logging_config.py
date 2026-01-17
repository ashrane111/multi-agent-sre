"""
Structured logging configuration.

Uses structlog for:
- JSON-formatted logs in production
- Pretty console output in development
- Correlation IDs for request tracing
- Automatic context enrichment
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog
from structlog.types import Processor

from src.config import settings

# Context variables for correlation
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
incident_id: ContextVar[str] = ContextVar("incident_id", default="")
agent_name: ContextVar[str] = ContextVar("agent_name", default="")


def add_correlation_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add correlation IDs to log events."""
    if cid := correlation_id.get():
        event_dict["correlation_id"] = cid
    if iid := incident_id.get():
        event_dict["incident_id"] = iid
    if agent := agent_name.get():
        event_dict["agent"] = agent
    return event_dict


def add_service_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service context to log events."""
    event_dict["service"] = "multi-agent-sre"
    event_dict["environment"] = settings.app_env
    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    
    In development: Pretty-printed colorful console output
    In production: JSON-formatted logs for log aggregation
    """
    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_context,
        add_service_context,
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_development:
        # Development: Pretty console output
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
    else:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Bound structlog logger
    """
    return structlog.get_logger(name)


def set_correlation_id(cid: str | None = None) -> str:
    """
    Set a correlation ID for the current context.
    
    Args:
        cid: Correlation ID to set (generates one if not provided)
        
    Returns:
        The correlation ID that was set
    """
    if cid is None:
        cid = str(uuid4())
    correlation_id.set(cid)
    return cid


def set_incident_id(iid: str) -> None:
    """Set incident ID for the current context."""
    incident_id.set(iid)


def set_agent_name(name: str) -> None:
    """Set agent name for the current context."""
    agent_name.set(name)


def clear_context() -> None:
    """Clear all logging context variables."""
    correlation_id.set("")
    incident_id.set("")
    agent_name.set("")
