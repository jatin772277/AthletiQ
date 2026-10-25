"""AthletiQ Centralized Backend Application."""

from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app.config import settings
from backend.app.schemas.models import HealthResponse
from backend.app.api.routes import (
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
    print(f"[AthletiQ API] Starting {settings.app_name} v{settings.app_version}...")
    print(f"[AthletiQ API] Root: {settings.project_root}, Outputs: {settings.outputs_dir}")
    yield
    print(f"[AthletiQ API] Shutting down {settings.app_name}...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AthletiQ Centralized Computer Vision & Tactical Kinematics Analytics API. "
        "Provides 32-player squad tracking, zero-loss collective metrics, constituent "
        "removed player buckets, DSA PriorityQueue rankings, interaction network graphs, "
        "and RAG-ready tactical query endpoints."
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
app.include_router(matches.router)
app.include_router(players.router)
app.include_router(buckets.router)
app.include_router(dsa.router)
app.include_router(possession.router)
app.include_router(graph.router)
app.include_router(rag.router)
app.include_router(pipeline.router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health", response_model=HealthResponse, tags=["System Health"])
def health_check():
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": "development" if settings.debug else "production",
        "default_match": settings.default_match_id,
        "data_retention_policy": "zero_wastage_collective_metrics_verified",
    }
