"""Health check endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from app import __version__
from app.api.schemas import HealthResponse, ReadinessResponse
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


def get_vector_store(request: Request) -> VectorStoreService:
    """Dependency: return the shared VectorStoreService from app state."""
    return request.app.state.vector_store


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns basic health status of the service.",
)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=__version__,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Checks if the service is ready to handle requests (including DB connectivity).",
)
async def readiness_check(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> ReadinessResponse:
    """Readiness check including database connectivity."""
    logger.debug("Readiness check requested")

    try:
        # Check Qdrant connection
        is_healthy = vector_store.health_check()

        if not is_healthy:
            raise HTTPException(
                status_code=503,
                detail="Vector store is not healthy",
            )

        collection_info = vector_store.get_collection_info()

        return ReadinessResponse(
            status="ready",
            qdrant_connected=True,
            collection_info=collection_info,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}",
        )
