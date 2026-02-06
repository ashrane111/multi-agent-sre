"""
Multi-Agent SRE Platform - Main Application Entry Point.

FastAPI application with:
- Health check endpoints
- Incident management API
- HITL approval endpoints
- Observability integration
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.api.routes import approvals_router, audit_router
from src.config import settings
from src.logging_config import set_correlation_id, setup_logging
from src.reliability.circuit_breaker import get_all_circuit_breakers

# Setup logging first
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "application_starting",
        version=__version__,
        environment=settings.app_env,
    )

    # Initialize services here (Redis, ChromaDB, etc.)
    # These will be added in later phases

    yield

    # Shutdown
    logger.info("application_stopping")


# Create FastAPI app
app = FastAPI(
    title="Multi-Agent SRE Platform",
    description="Autonomous incident response system using multi-agent AI",
    version=__version__,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan,
)

# CORS middleware (configure based on environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(approvals_router)
app.include_router(audit_router)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Add correlation ID to each request."""
    # Get or generate correlation ID
    correlation_id = request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    response = await call_next(request)

    # Add correlation ID to response
    response.headers["X-Correlation-ID"] = correlation_id or ""
    return response


# ============== Health Check Endpoints ==============


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "environment": settings.app_env,
    }


@app.get("/health/ready", tags=["Health"])
async def readiness_check() -> dict:
    """
    Readiness check for Kubernetes.
    
    Checks if all dependencies are available.
    """
    checks = {
        "ollama": await _check_ollama(),
        "redis": await _check_redis(),
    }

    all_healthy = all(checks.values())

    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
    }


@app.get("/health/live", tags=["Health"])
async def liveness_check() -> dict:
    """Liveness check for Kubernetes."""
    return {"status": "alive"}


@app.get("/health/circuits", tags=["Health"])
async def circuit_breaker_status() -> dict:
    """Get status of all circuit breakers."""
    return {
        "circuit_breakers": get_all_circuit_breakers(),
    }


# ============== Placeholder Endpoints (to be implemented) ==============


@app.get("/api/v1/incidents", tags=["Incidents"])
async def list_incidents() -> dict:
    """List all incidents."""
    return {"incidents": [], "message": "To be implemented in Phase 2"}


@app.post("/api/v1/incidents", tags=["Incidents"])
async def create_incident() -> dict:
    """Create a new incident (trigger workflow)."""
    return {"message": "To be implemented in Phase 2"}


@app.get("/api/v1/incidents/{incident_id}", tags=["Incidents"])
async def get_incident(incident_id: str) -> dict:
    """Get incident details."""
    return {"incident_id": incident_id, "message": "To be implemented in Phase 2"}


@app.get("/api/v1/costs", tags=["Observability"])
async def get_cost_report() -> dict:
    """Get LLM cost report."""
    from src.models.router import get_model_router

    router = get_model_router()
    return router.get_cost_report()


# ============== Helper Functions ==============


async def _check_ollama() -> bool:
    """Check if Ollama is available."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.llm.ollama_base_url}/api/tags")
            return response.status_code == 200
    except Exception:
        return False


async def _check_redis() -> bool:
    """Check if Redis is available."""
    try:
        import redis.asyncio as redis

        client = redis.from_url(settings.memory.redis_url)
        await client.ping()
        await client.close()
        return True
    except Exception:
        return False


# ============== Exception Handlers ==============


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.is_development else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.is_development,
        workers=settings.api.workers,
    )
