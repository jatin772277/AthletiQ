"""AthletiQ Centralized Backend Application."""

from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app import db
from backend.app.config import settings
from backend.app.schemas.models import HealthResponse
from backend.app.security import authorize_match
from backend.app.services.pipeline_runner import runner
from backend.app.api.routes import (
    auth,
    jobs,
    matches,
    players,
    buckets,
    dsa,
    possession,
    graph,
    rag,
    pipeline,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_production()
    print(f"[AthletiQ API] Starting {settings.app_name} v{settings.app_version}...")
    print(f"[AthletiQ API] Environment: {settings.app_environment}; pipeline mode: {settings.pipeline_mode}")
    if db.ping():
        db.ensure_indexes()
        runner.start()
        print(f"[AthletiQ API] MongoDB connected ({settings.mongodb_db}); pipeline worker running in '{settings.pipeline_mode}' mode")
    else:
        print("[AthletiQ API] WARNING: MongoDB is unreachable; sign-in and uploads will fail until it is available")
    yield
    runner.stop()
    print(f"[AthletiQ API] Shutting down {settings.app_name}...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AthletiQ Centralized Computer Vision & Tactical Kinematics Analytics API. "
        "Provides 32-player squad tracking, zero-loss collective metrics, constituent "
        "removed player buckets, DSA PriorityQueue rankings, interaction network graphs, "
        "RAG-ready tactical query endpoints, user accounts and video upload analysis jobs."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API Routers
app.include_router(auth.router)
app.include_router(jobs.router)
# Match data is only served to signed-in users, for demo matches or analyses they own.
for module in (matches, players, buckets, dsa, possession, graph, rag, pipeline):
    app.include_router(module.router, dependencies=[Depends(authorize_match)])


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health", response_model=HealthResponse, tags=["System Health"])
def health_check():
    database_ready = db.ping()
    return {
        "status": "healthy" if database_ready else "degraded",
        "version": settings.app_version,
        "environment": "development" if settings.debug else "production",
        "default_match": settings.default_match_id,
        "database": "connected" if database_ready else "unreachable",
        "database_ready": database_ready,
        "pipeline_mode": settings.pipeline_mode,
        "data_retention_policy": "zero_wastage_collective_metrics_verified",
    }
